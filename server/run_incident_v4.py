#!/usr/bin/env python3
"""Durably prepare, build, and verify one dual-clock Incident V4 release."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from typing import Any, Iterator

from story_monitor.incident_context_v4 import validate_enriched_source_v4
from story_monitor.incident_release_v4 import normalize_released_at
from story_monitor.incident_validation_v4 import (
    validate_evidence_append,
    validate_evidence_directory,
)
from story_monitor.incident_viewer_v4 import validate_viewer_directory
from story_monitor.runner_process import (
    RunnerError,
    atomic_write_json,
    atomic_write_text,
    load_json,
    run_stage,
    utc_now,
    utc_tag,
)


DEFAULT_ROOT = Path("/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1")
VIEWER_MODE = "crop_incident_v4_dual_clock"
FOCUSED_TESTS = (
    "server.test_incident_policy_v4",
    "server.test_incident_context_v4",
    "server.test_prepare_incident_source_v4",
    "server.test_incident_validation_v4",
    "server.test_incident_motifs_v4",
    "server.test_incident_motif_workflow_v4",
    "server.test_run_incident_motifs_v4",
    "server.test_incident_viewer_v4",
    "server.test_story_map_server",
    "server.test_run_incident_v4",
)
NODE_SYNTAX_SCRIPT = """
const { spawnSync } = require('node:child_process');
for (const file of process.argv.slice(1)) {
  const result = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.status !== 0) process.exit(result.status || 1);
}
""".strip()


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def _job_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Job tag must be 1-64 letters, digits, dots, underscores, or hyphens"
        )
    return value


def _executable(value: Path) -> Path:
    return Path(os.path.abspath(value.expanduser()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build V4 daily evidence, export the dual-clock viewer, and run truth gates."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--generation-dir", type=Path, required=True)
    run.add_argument("--incident-dir", type=Path, required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--enriched-source-parquet",
        type=Path,
        help="Existing immutable V4-enriched daily source.",
    )
    source.add_argument(
        "--echo-deliverable",
        type=Path,
        help="Echo-aware daily deliverable to enrich inside this durable job.",
    )
    run.add_argument(
        "--full-parquet",
        type=Path,
        action="append",
        default=[],
        help="Rich field/day source; repeat for non-overlapping years.",
    )
    run.add_argument(
        "--source-acquisition-parquet",
        type=Path,
        action="append",
        default=[],
        help="Optional QA source used while enriching the daily source.",
    )
    run.add_argument(
        "--acquisition-parquet",
        type=Path,
        help=(
            "Partial or complete acquisition-attempt ledger merged with, rather "
            "than substituted for, acquisitions derived from the daily source."
        ),
    )
    run.add_argument(
        "--availability-mode",
        choices=("strict", "reconstructed"),
        default="reconstructed",
    )
    run.add_argument(
        "--released-at",
        help=(
            "Timezone-aware ingest/release watermark. Captured once in UTC when "
            "omitted and reused by every resumable stage."
        ),
    )
    release = run.add_mutually_exclusive_group(required=True)
    release.add_argument("--previous-evidence-dir", type=Path)
    release.add_argument("--first-release", action="store_true")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument(
        "--node", type=Path,
        default=Path(shutil.which("node")) if shutil.which("node") else None,
    )
    run.add_argument("--threads", type=_positive, default=32)
    run.add_argument("--memory-limit", default="96GB")
    run.add_argument("--temp-dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=_positive, default=30)
    run.add_argument("--job-tag", type=_job_tag)
    run.add_argument("--skip-tests", action="store_true")

    resume = commands.add_parser("resume")
    resume.add_argument("--job-dir", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    status.add_argument("--job-dir", type=Path)
    return parser


def _paths(root: Path, tag: str) -> dict[str, str]:
    job = root / "jobs" / f"incident_v4_{tag}"
    return {
        "job_dir": str(job),
        "runner_log": str(job / "runner.log"),
        "tests_stdout": str(job / "tests.stdout.log"),
        "tests_stderr": str(job / "tests.stderr.log"),
        "ui_tests_stdout": str(job / "ui-tests.stdout.log"),
        "ui_tests_stderr": str(job / "ui-tests.stderr.log"),
        "ui_syntax_stdout": str(job / "ui-syntax.stdout.log"),
        "ui_syntax_stderr": str(job / "ui-syntax.stderr.log"),
        "source_json": str(job / "source.json"),
        "source_stderr": str(job / "source.stderr.log"),
        "evidence_json": str(job / "evidence.json"),
        "evidence_stderr": str(job / "evidence.stderr.log"),
        "export_json": str(job / "export.json"),
        "export_stderr": str(job / "export.stderr.log"),
        "smoke_stdout": str(job / "smoke.stdout.log"),
        "smoke_stderr": str(job / "smoke.stderr.log"),
        "enriched_source": str(root / "sources" / f"incident_source_v4_{tag}.parquet"),
        "evidence_dir": str(root / "releases" / f"incident_evidence_v4_{tag}"),
        "viewer_dir": str(root / "releases" / f"incident_viewer_v4_{tag}"),
    }


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"incident-v4-runner.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    path = root / "logs" / "incident_v4_runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"Another Incident V4 runner holds {path}", 75) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield


def _save(state: dict[str, Any]) -> None:
    atomic_write_json(Path(state["paths"]["job_dir"]) / "state.json", state)


def _stage(
    state: dict[str, Any],
    logger: logging.Logger,
    name: str,
    label: str,
    command: list[str],
    stdout_key: str,
    stderr_key: str,
) -> None:
    state["current_stage"] = name
    state.setdefault("stages", {})[name] = {
        "status": "running", "updated_at": utc_now(),
    }
    _save(state)
    config = state["config"]
    env = os.environ.copy()
    server = str(Path(config["repo_dir"]) / "server")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = server + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    def heartbeat(pid: int, elapsed: int = 0, rss: str = "starting") -> None:
        state["active_process"] = {
            "label": label, "pid": pid, "elapsed_seconds": elapsed,
            "rss": rss, "heartbeat_at": utc_now(),
        }
        _save(state)

    try:
        code = run_stage(
            command,
            stdout_path=Path(state["paths"][stdout_key]),
            stderr_path=Path(state["paths"][stderr_key]),
            logger=logger,
            label=label,
            heartbeat_seconds=int(config["heartbeat_seconds"]),
            cwd=Path(config["repo_dir"]),
            env=env,
            on_start=heartbeat,
            on_heartbeat=heartbeat,
        )
    finally:
        state.pop("active_process", None)
        _save(state)
    if code:
        state["stages"][name].update(
            {"status": "failed", "exit_code": code, "updated_at": utc_now()}
        )
        _save(state)
        raise RunnerError(f"{label} failed with exit code {code}", code)
    state["stages"][name].update(
        {"status": "complete", "exit_code": 0, "updated_at": utc_now()}
    )
    _save(state)


def _artifact_complete(path: Path, mode: str | None = None) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    run = payload.get("run") or {}
    return str(run.get("status")) == "complete" and (
        mode is None or str(payload.get("mode") or run.get("mode")) == mode
    )


def _validated_viewer(path: Path) -> dict[str, Any] | None:
    if not _artifact_complete(path, VIEWER_MODE):
        return None
    return validate_viewer_directory(path)


def _source_complete(path: Path, mode: str | None = None) -> bool:
    try:
        validate_enriched_source_v4(path, expected_mode=mode)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    return True


def _node_syntax_command(node: str, repo: Path) -> list[str]:
    files = sorted((repo / "server" / "static").glob("*.js"))
    if not files:
        raise RunnerError("No static JavaScript files found", 2)
    return [node, "-e", NODE_SYNTAX_SCRIPT, *(str(path) for path in files)]


def _mark_skipped(state: dict[str, Any], name: str) -> None:
    state.setdefault("stages", {})[name] = {
        "status": "skipped", "reason": "--skip-tests", "updated_at": utc_now(),
    }
    _save(state)


def _evidence_command(state: dict[str, Any]) -> list[str]:
    config = state["config"]
    command = [
        config["python"], "server/weekly_story_monitor.py", "build-evidence-v4",
        "--generation-dir", config["generation_dir"],
        "--evidence-dir", state["paths"]["evidence_dir"],
        "--availability-mode", config["availability_mode"],
        "--released-at", config["released_at"],
        "--threads", str(config["threads"]),
        "--memory-limit", config["memory_limit"],
        "--temp-dir", config["temp_dir"],
    ]
    if config.get("enriched_source_parquet"):
        command.extend(["--enriched-source-parquet", config["enriched_source_parquet"]])
    if config.get("acquisition_parquet"):
        command.extend(["--acquisition-parquet", config["acquisition_parquet"]])
    return command


def _source_command(state: dict[str, Any]) -> list[str]:
    config = state["config"]
    command = [
        config["python"], "server/prepare_incident_source_v4.py",
        "--echo-deliverable", config["echo_deliverable"],
        "--output-parquet", config["enriched_source_parquet"],
        "--availability-mode", config["availability_mode"],
        "--released-at", config["released_at"],
        "--threads", str(config["threads"]),
        "--memory-limit", config["memory_limit"],
        "--temp-dir", config["temp_dir"],
    ]
    for path in config["full_parquets"]:
        command.extend(["--full-parquet", path])
    for path in config["source_acquisition_parquets"]:
        command.extend(["--acquisition-parquet", path])
    return command


def _execute(state: dict[str, Any], logger: logging.Logger) -> int:
    config, paths = state["config"], state["paths"]
    python = str(config["python"])
    try:
        if config["skip_tests"]:
            for name in ("python_tests", "ui_tests", "ui_syntax"):
                if name not in state.get("stages", {}):
                    _mark_skipped(state, name)
        else:
            node = str(config["node"])
            if state.get("stages", {}).get("python_tests", {}).get("status") != "complete":
                _stage(
                    state, logger, "python_tests", "[1/7 PYTHON TESTS]",
                    [python, "-m", "unittest", *FOCUSED_TESTS],
                    "tests_stdout", "tests_stderr",
                )
            if state.get("stages", {}).get("ui_tests", {}).get("status") != "complete":
                _stage(
                    state, logger, "ui_tests", "[2/7 UI TESTS]",
                    [node, "--test", "server/test_story_ui.mjs"],
                    "ui_tests_stdout", "ui_tests_stderr",
                )
            if state.get("stages", {}).get("ui_syntax", {}).get("status") != "complete":
                _stage(
                    state, logger, "ui_syntax", "[3/7 UI SYNTAX]",
                    _node_syntax_command(node, Path(config["repo_dir"])),
                    "ui_syntax_stdout", "ui_syntax_stderr",
                )

        enriched = Path(config["enriched_source_parquet"])
        if config.get("echo_deliverable"):
            source_ready = (
                _source_complete(enriched, config["availability_mode"])
                if enriched.exists() else False
            )
            if enriched.exists() and not source_ready:
                raise RunnerError(
                    "Generated enriched source exists without a complete immutable "
                    f"manifest: {enriched}",
                    2,
                )
            if not source_ready:
                _stage(
                    state, logger, "source_preparation", "[4/7 SOURCE PREPARATION]",
                    _source_command(state), "source_json", "source_stderr",
                )
                source_ready = _source_complete(enriched, config["availability_mode"])
            if not source_ready:
                raise RunnerError("Source preparation did not produce a valid release", 2)
        elif "source_preparation" not in state.get("stages", {}):
            state.setdefault("stages", {})["source_preparation"] = {
                "status": "external_immutable_input",
                "path": str(enriched),
                "updated_at": utc_now(),
            }
            _save(state)

        evidence = Path(paths["evidence_dir"])
        if not _artifact_complete(evidence):
            _stage(
                state, logger, "evidence", "[5/7 EVIDENCE BUILD]",
                _evidence_command(state), "evidence_json", "evidence_stderr",
            )
        validation = validate_evidence_directory(evidence)
        previous = config.get("previous_evidence_dir")
        if previous:
            validation["append_stability"] = validate_evidence_append(
                Path(previous), evidence
            )
        else:
            validation["append_stability"] = {
                "status": "first_release", "first_release": True,
            }
        state["evidence_validation"] = validation
        _save(state)

        viewer = Path(paths["viewer_dir"])
        viewer_validation = _validated_viewer(viewer)
        if viewer_validation is None:
            _stage(
                state, logger, "export", "[6/7 VIEWER EXPORT]",
                [
                    python, "server/export_incident_viewer_v4.py",
                    "--incident-dir", config["incident_dir"],
                    "--evidence-dir", paths["evidence_dir"],
                    "--source-generation-dir", config["generation_dir"],
                    "--output-dir", paths["viewer_dir"],
                    "--threads", str(config["threads"]),
                    "--memory-limit", config["memory_limit"],
                    "--temp-dir", config["temp_dir"],
                ],
                "export_json", "export_stderr",
            )
            viewer_validation = validate_viewer_directory(viewer)
        state["viewer_validation"] = viewer_validation
        _save(state)
        smoke = (
            "from pathlib import Path; import json; "
            "from story_map_server import Settings,StoryMapStore; "
            f"p=Path({paths['viewer_dir']!r}); "
            "s=Settings(run_dir=p,static_dir=Path('server/static'),host='127.0.0.1',"
            "port=8877,raster_tiles='',raster_attribution='',default_feature_limit=5000,"
            "max_feature_limit=20000,log_level='INFO'); st=StoryMapStore(s); "
            "assert st.health()['ok'] and st.has_incident_v4(); t=st.v4_timeline(); "
            "assert t['days']; d=next(x['calendar_date'] for x in t['days'] "
            "if x['source_day_present']); "
            "f=st.v4_frame(calendar_date=d,bbox=None,limit=0); "
            "m=f['meta']; assert m['complete_country_representation']; "
            "assert m['accounted_field_count']==m['source_field_count']; "
            "assert not m['country_representation_truncated']; "
            "print(json.dumps({'status':'ok','day':d,'grid':len(f['field_overview']['features']),"
            "'unmappable_warning':m['unmappable_warning']}))"
        )
        _stage(
            state, logger, "smoke", "[7/7 SERVER SMOKE]",
            [python, "-c", smoke], "smoke_stdout", "smoke_stderr",
        )
        state.update({
            "status": "complete", "exit_code": 0, "finished_at": utc_now(),
            "current_stage": "complete",
        })
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", "0\n")
        logger.info("Incident V4 dual-clock viewer complete — %s", viewer)
        return 0
    except KeyboardInterrupt:
        code, message = 130, "Interrupted"
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 143
        message = "Terminated"
    except RunnerError as exc:
        code, message = exc.exit_code, str(exc)
    except Exception as exc:  # runner boundary records unexpected failures durably
        logger.exception("Unexpected Incident V4 runner failure")
        code, message = 2, f"Unexpected failure: {exc}"
    logger.error("%s", message)
    state.update({
        "status": "interrupted" if code in {130, 143} else "failed",
        "exit_code": code, "finished_at": utc_now(), "error": message,
    })
    _save(state)
    atomic_write_text(Path(paths["job_dir"]) / "status", f"{code}\n")
    return code


def _new_state(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    generation = args.generation_dir.expanduser().resolve()
    incident = args.incident_dir.expanduser().resolve()
    previous = (
        args.previous_evidence_dir.expanduser().resolve()
        if args.previous_evidence_dir else None
    )
    supplied_enriched = (
        args.enriched_source_parquet.expanduser().resolve()
        if args.enriched_source_parquet else None
    )
    echo = args.echo_deliverable.expanduser().resolve() if args.echo_deliverable else None
    full = [path.expanduser().resolve() for path in args.full_parquet]
    source_acquisitions = [
        path.expanduser().resolve() for path in args.source_acquisition_parquet
    ]
    acquisition = args.acquisition_parquet.expanduser().resolve() if args.acquisition_parquet else None
    python = _executable(args.python)
    node = _executable(args.node) if args.node is not None else None
    repo = Path(__file__).resolve().parent.parent
    for label, path in (("root", root), ("generation", generation), ("incident", incident)):
        if not path.is_dir():
            raise RunnerError(f"{label} directory does not exist: {path}")
    for label, path in (
        ("enriched source", supplied_enriched),
        ("echo deliverable", echo),
        ("acquisition", acquisition),
    ):
        if path is not None and not path.is_file():
            raise RunnerError(f"{label} parquet does not exist: {path}")
    for label, candidates in (
        ("full source", full),
        ("source acquisition", source_acquisitions),
    ):
        missing = [str(path) for path in candidates if not path.is_file()]
        if missing:
            raise RunnerError(f"{label} parquet does not exist: {', '.join(missing)}")
    if echo is not None and not full:
        raise RunnerError("--echo-deliverable requires at least one --full-parquet")
    if supplied_enriched is not None and (full or source_acquisitions):
        raise RunnerError(
            "--full-parquet/--source-acquisition-parquet apply only with "
            "--echo-deliverable"
        )
    supplied_contract = None
    if supplied_enriched is not None:
        try:
            supplied_contract = validate_enriched_source_v4(
                supplied_enriched, expected_mode=args.availability_mode
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            raise RunnerError(
                "--enriched-source-parquet requires its complete immutable sidecar "
                f"manifest: {supplied_enriched}.manifest.json"
            ) from exc
    if previous is not None and not previous.is_dir():
        raise RunnerError(f"Previous evidence directory does not exist: {previous}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError(f"Python executable is invalid: {python}")
    if not args.skip_tests and (
        node is None or not node.is_file() or not os.access(node, os.X_OK)
    ):
        raise RunnerError("Node executable is required for UI verification")
    tag = args.job_tag or utc_tag()
    released_at = normalize_released_at(
        args.released_at
        or (supplied_contract or {}).get("released_at")
        or utc_now()
    )
    if (
        supplied_contract is not None
        and supplied_contract["released_at"] != released_at
    ):
        raise RunnerError(
            "--released-at must equal the immutable enriched-source watermark"
        )
    paths = _paths(root, tag)
    enriched = supplied_enriched or Path(paths["enriched_source"])
    job = Path(paths["job_dir"])
    if job.exists():
        raise RunnerError(f"Job tag already exists: {tag}")
    job.mkdir(parents=True)
    temp = (args.temp_dir or root / "duckdb_tmp").expanduser().resolve()
    temp.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "incident-v4-runner/1",
        "job_tag": tag, "status": "running", "started_at": utc_now(),
        "pid": os.getpid(), "current_stage": "starting", "paths": paths,
        "config": {
            "root": str(root), "repo_dir": str(repo),
            "generation_dir": str(generation), "incident_dir": str(incident),
            "previous_evidence_dir": str(previous) if previous else None,
            "first_release": bool(args.first_release),
            "enriched_source_parquet": str(enriched),
            "echo_deliverable": str(echo) if echo else None,
            "full_parquets": [str(path) for path in full],
            "source_acquisition_parquets": [
                str(path) for path in source_acquisitions
            ],
            "acquisition_parquet": str(acquisition) if acquisition else None,
            "availability_mode": args.availability_mode,
            "released_at": released_at,
            "python": str(python), "node": str(node) if node else None,
            "threads": args.threads, "memory_limit": args.memory_limit,
            "temp_dir": str(temp), "heartbeat_seconds": args.heartbeat_seconds,
            "skip_tests": args.skip_tests,
        },
        "stages": {},
    }


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_job_state(job_dir: Path) -> dict[str, Any]:
    state = load_json(job_dir.expanduser().resolve() / "state.json")
    active = state.get("active_process") or {}
    if state.get("status") != "running":
        state["liveness"] = "terminal"
    elif active.get("pid") and _pid_exists(int(active["pid"])):
        state["liveness"] = "child_alive"
    elif state.get("pid") and _pid_exists(int(state["pid"])):
        state["liveness"] = "runner_alive"
    else:
        state["liveness"] = "stale"
    return state


def _latest_job(root: Path) -> Path:
    pointer = root.expanduser().resolve() / "logs" / "latest_incident_v4_job.txt"
    if not pointer.is_file():
        raise RunnerError(f"No latest Incident V4 job pointer exists: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise RunnerError(f"Latest Incident V4 job pointer is empty: {pointer}")
    return Path(value)


def _handle_sigterm(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            job = args.job_dir or _latest_job(args.root)
            print(json.dumps(read_job_state(job), indent=2, sort_keys=True))
            return 0
        if args.command == "resume":
            state = load_json(args.job_dir.expanduser().resolve() / "state.json")
            root = Path(state["config"]["root"])
            with _lock(root):
                state.update({"status": "running", "pid": os.getpid(), "resumed_at": utc_now()})
                _save(state)
                return _execute(state, _logger(Path(state["paths"]["runner_log"])))
        state = _new_state(args)
        root = Path(state["config"]["root"])
        with _lock(root):
            _save(state)
            job = Path(state["paths"]["job_dir"])
            atomic_write_text(job / "runner.pid", f"{os.getpid()}\n")
            atomic_write_text(root / "logs" / "latest_incident_v4_job.txt", f"{job}\n")
            return _execute(state, _logger(Path(state["paths"]["runner_log"])))
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
