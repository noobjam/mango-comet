#!/usr/bin/env python3
"""Durably build, export, and smoke-test one Incident V3 viewer release."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import signal
import shutil
import sys
from typing import Any, Iterator

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
FOCUSED_TESTS = (
    "server.test_incident_context_v3",
    "server.test_incident_cells_v3",
    "server.test_incident_tracking_v3",
    "server.test_incident_exposures_v3",
    "server.test_incident_lineage_v3",
    "server.test_incident_story_states_v3",
    "server.test_incident_denominators_v3",
    "server.test_incident_archetypes_v3",
    "server.test_incident_validation_v3",
    "server.test_incident_workflow_v3",
    "server.test_incident_viewer_v3",
    "server.test_story_map_server",
    "server.test_run_incident_v3",
)

NODE_SYNTAX_SCRIPT = """
const { spawnSync } = require('node:child_process');
for (const file of process.argv.slice(1)) {
  const result = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.status !== 0) process.exit(result.status || 1);
  process.stdout.write(`syntax ok: ${file}\n`);
}
""".strip()


def _iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM-DD") from exc


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


def _executable_path(value: Path) -> Path:
    # A virtualenv's bin/python is commonly a symlink to the base interpreter.
    # Preserve the launcher path: resolving the symlink bypasses the virtualenv's
    # site-packages and can make installed dependencies disappear.
    return Path(os.path.abspath(value.expanduser()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Incident V3 tests, immutable build, viewer export, and smoke gate."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start a new durable Incident V3 release job.")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--generation-dir", type=Path, required=True)
    release_mode = run.add_mutually_exclusive_group(required=True)
    release_mode.add_argument("--previous-incident-dir", type=Path)
    release_mode.add_argument(
        "--first-release", action="store_true",
        help="Explicitly declare the first immutable Incident V3 release.",
    )
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument(
        "--node",
        type=Path,
        default=Path(shutil.which("node")) if shutil.which("node") else None,
    )
    run.add_argument("--baseline-through", type=_iso_date, required=True)
    run.add_argument("--threads", type=_positive, default=32)
    run.add_argument("--memory-limit", default="96GB")
    run.add_argument("--temp-dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=_positive, default=30)
    run.add_argument("--job-tag", type=_job_tag)
    run.add_argument("--skip-tests", action="store_true")
    run.add_argument(
        "--capture-stage9-replay",
        action="store_true",
        help=(
            "On the first stage-9 finalizer failure, atomically preserve one "
            "bounded read-only replay capsule in the job directory."
        ),
    )

    resume = commands.add_parser("resume", help="Resume missing stages of one job.")
    resume.add_argument("--job-dir", type=Path, required=True)

    status = commands.add_parser("status", help="Print one job's durable state.")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    status.add_argument("--job-dir", type=Path)
    return parser


def _paths(root: Path, tag: str) -> dict[str, str]:
    job = root / "jobs" / f"incident_v3_{tag}"
    return {
        "job_dir": str(job),
        "runner_log": str(job / "runner.log"),
        "tests_stdout": str(job / "tests.stdout.log"),
        "tests_stderr": str(job / "tests.stderr.log"),
        "ui_tests_stdout": str(job / "ui-tests.stdout.log"),
        "ui_tests_stderr": str(job / "ui-tests.stderr.log"),
        "ui_syntax_stdout": str(job / "ui-syntax.stdout.log"),
        "ui_syntax_stderr": str(job / "ui-syntax.stderr.log"),
        "build_json": str(job / "build.json"),
        "build_stderr": str(job / "build.stderr.log"),
        "finalizer_failure_capsule": str(job / "stage9-finalizer-capsule"),
        "export_json": str(job / "export.json"),
        "export_stderr": str(job / "export.stderr.log"),
        "smoke_stdout": str(job / "smoke.stdout.log"),
        "smoke_stderr": str(job / "smoke.stderr.log"),
        "incident_dir": str(root / "releases" / f"incidents_v3_{tag}"),
        "viewer_dir": str(root / "releases" / f"incident_viewer_v3_{tag}"),
    }


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"incident-v3-runner.{path.parent.name}")
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
    path = root / "logs" / "incident_v3_runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"Another Incident V3 runner holds {path}", 75) from exc
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
    stdout_name: str,
    stderr_name: str,
) -> None:
    state["current_stage"] = name
    state.setdefault("stages", {})[name] = {
        "status": "running", "updated_at": utc_now()
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
            "label": label,
            "pid": pid,
            "elapsed_seconds": elapsed,
            "rss": rss,
            "heartbeat_at": utc_now(),
        }
        _save(state)

    try:
        code = run_stage(
            command,
            stdout_path=Path(state["paths"][stdout_name]),
            stderr_path=Path(state["paths"][stderr_name]),
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


def _artifact_complete(path: Path, *, mode: str | None = None) -> bool:
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


def _require_nonempty_incident_release(path: Path) -> None:
    manifest = load_json(path / "manifest.json")
    counts = (manifest.get("validation") or {}).get("row_counts") or {}
    count = counts.get("incident_weekly_state")
    if count is None:
        raise RunnerError(
            "Incident V3 manifest does not report incident_weekly_state rows", 21
        )
    if int(count) < 1:
        raise RunnerError(
            "Incident V3 built zero crop stories; inspect thresholds/evidence before viewer export",
            21,
        )


def _mark_stage_skipped(
    state: dict[str, Any], name: str, reason: str
) -> None:
    state.setdefault("stages", {})[name] = {
        "status": "skipped",
        "reason": reason,
        "updated_at": utc_now(),
    }
    _save(state)


def _node_syntax_command(node: str, repo_dir: Path) -> list[str]:
    files = sorted((repo_dir / "server" / "static").glob("*.js"))
    if not files:
        raise RunnerError("No static JavaScript files found for syntax verification", 2)
    return [node, "-e", NODE_SYNTAX_SCRIPT, *(str(path) for path in files)]


def _require_append_gate(
    path: Path, previous: str | None, *, first_release: bool
) -> None:
    if bool(first_release) == bool(previous):
        raise RunnerError(
            "Runner release mode must be exactly one of first release or append",
            21,
        )
    manifest = load_json(path / "manifest.json")
    gate = (manifest.get("validation") or {}).get("append_stability") or {}
    expected = "passed" if previous else "first_release"
    if str(gate.get("status") or "") != expected:
        raise RunnerError(
            f"Incident V3 append-stability gate is {gate.get('status')!r}; "
            f"expected {expected!r}",
            21,
        )
    if previous:
        previous_manifest = load_json(Path(previous) / "manifest.json")
        previous_id = (previous_manifest.get("run") or {}).get("generation_id")
        if gate.get("previous_generation_id") != previous_id:
            raise RunnerError(
                "Incident V3 append-stability gate references a different previous release",
                21,
            )
    elif gate.get("first_release") is not True:
        raise RunnerError(
            "Incident V3 first-release gate is missing its explicit declaration",
            21,
        )


def _execute(state: dict[str, Any], logger: logging.Logger) -> int:
    config = state["config"]
    paths = state["paths"]
    paths.setdefault(
        "finalizer_failure_capsule",
        str(Path(paths["job_dir"]) / "stage9-finalizer-capsule"),
    )
    python = str(config["python"])
    try:
        if config["skip_tests"]:
            for name in ("python_tests", "ui_tests", "ui_syntax"):
                if name not in state.get("stages", {}):
                    _mark_stage_skipped(state, name, "--skip-tests")
        else:
            node = str(config["node"])
            if state.get("stages", {}).get("python_tests", {}).get("status") != "complete":
                _stage(
                    state, logger, "python_tests", "[1/6 PYTHON TESTS]",
                    [python, "-m", "unittest", *FOCUSED_TESTS],
                    "tests_stdout", "tests_stderr",
                )
            if state.get("stages", {}).get("ui_tests", {}).get("status") != "complete":
                _stage(
                    state, logger, "ui_tests", "[2/6 UI TESTS]",
                    [node, "--test", "server/test_story_ui.mjs"],
                    "ui_tests_stdout", "ui_tests_stderr",
                )
            if state.get("stages", {}).get("ui_syntax", {}).get("status") != "complete":
                _stage(
                    state, logger, "ui_syntax", "[3/6 UI SYNTAX]",
                    _node_syntax_command(node, Path(config["repo_dir"])),
                    "ui_syntax_stdout", "ui_syntax_stderr",
                )
        incident_dir = Path(paths["incident_dir"])
        if _artifact_complete(incident_dir):
            state.setdefault("stages", {})["build"] = {
                "status": "complete", "skipped_existing": True,
                "updated_at": utc_now(),
            }
            _save(state)
        else:
            build_command = [
                    python, "server/weekly_story_monitor.py", "build-incidents-v3",
                    "--generation-dir", config["generation_dir"],
                    "--output-dir", paths["incident_dir"],
                    "--baseline-through", config["baseline_through"],
                    "--threads", str(config["threads"]),
                    "--memory-limit", config["memory_limit"],
                    "--temp-dir", config["temp_dir"],
                ]
            if config.get("capture_stage9_replay", False):
                build_command.extend(
                    [
                        "--finalizer-failure-capsule",
                        paths["finalizer_failure_capsule"],
                    ]
                )
            if config.get("previous_incident_dir"):
                build_command.extend(
                    ["--previous-incident-dir", config["previous_incident_dir"]]
                )
            else:
                build_command.append("--first-release")
            _stage(
                state, logger, "build", "[4/6 INCIDENT BUILD]",
                build_command,
                "build_json", "build_stderr",
            )
        _require_nonempty_incident_release(incident_dir)
        _require_append_gate(
            incident_dir,
            config.get("previous_incident_dir"),
            first_release=bool(config.get("first_release")),
        )
        viewer_dir = Path(paths["viewer_dir"])
        if _artifact_complete(viewer_dir, mode="crop_incident_v3"):
            state.setdefault("stages", {})["export"] = {
                "status": "complete", "skipped_existing": True,
                "updated_at": utc_now(),
            }
            _save(state)
        else:
            _stage(
                state, logger, "export", "[5/6 VIEWER EXPORT]",
                [
                    python, "server/export_incident_viewer_v3.py",
                    "--incident-dir", paths["incident_dir"],
                    "--source-generation-dir", config["generation_dir"],
                    "--output-dir", paths["viewer_dir"],
                    "--threads", str(config["threads"]),
                    "--memory-limit", config["memory_limit"],
                    "--temp-dir", config["temp_dir"],
                ],
                "export_json", "export_stderr",
            )
        smoke = (
            "from pathlib import Path; import json; "
            "from story_map_server import Settings,StoryMapStore; "
            f"p=Path({paths['viewer_dir']!r}); "
            "s=Settings(run_dir=p,static_dir=Path('server/static'),host='127.0.0.1',"
            "port=8877,raster_tiles='',raster_attribution='',default_feature_limit=5000,"
            "max_feature_limit=20000,log_level='INFO'); st=StoryMapStore(s); "
            "assert st.health()['ok'] and st.has_incident_v3(); t=st.timeline(); "
            "assert t['buckets']; b=str(t['buckets'][0]['timeline_bucket']); "
            "f=st.incident_footprints(timeline_bucket=b); assert f['meta']['complete']; "
            "print(json.dumps({'status':'ok','bucket':b,'footprints':len(f['features'])}))"
        )
        _stage(
            state, logger, "smoke", "[6/6 SERVER SMOKE]",
            [python, "-c", smoke], "smoke_stdout", "smoke_stderr",
        )
        state.update(
            {
                "status": "complete", "exit_code": 0, "finished_at": utc_now(),
                "current_stage": "complete",
            }
        )
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", "0\n")
        logger.info("Incident V3 release complete — viewer %s", paths["viewer_dir"])
        return 0
    except KeyboardInterrupt:
        logger.error("Incident V3 runner interrupted")
        state.update(
            {
                "status": "interrupted", "exit_code": 130,
                "finished_at": utc_now(), "error": "Interrupted",
            }
        )
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", "130\n")
        return 130
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 143
        logger.error("Incident V3 runner terminated with status %s", code)
        state.update(
            {
                "status": "interrupted", "exit_code": code,
                "finished_at": utc_now(), "error": "Terminated",
            }
        )
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", f"{code}\n")
        return code
    except RunnerError as exc:
        logger.error("%s", exc)
        state.update(
            {
                "status": "failed", "exit_code": exc.exit_code,
                "finished_at": utc_now(), "error": str(exc),
            }
        )
        _save(state)
        atomic_write_text(
            Path(paths["job_dir"]) / "status", f"{exc.exit_code}\n"
        )
        return exc.exit_code
    except Exception as exc:
        logger.exception("Unexpected Incident V3 runner failure")
        state.update(
            {
                "status": "failed", "exit_code": 2,
                "finished_at": utc_now(), "error": f"Unexpected failure: {exc}",
            }
        )
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", "2\n")
        return 2


def _new_state(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    generation = args.generation_dir.expanduser().resolve()
    previous = (
        args.previous_incident_dir.expanduser().resolve()
        if args.previous_incident_dir is not None
        else None
    )
    python = _executable_path(args.python)
    node = _executable_path(args.node) if args.node is not None else None
    repo = Path(__file__).resolve().parent.parent
    if not root.is_dir():
        raise RunnerError(f"Runner root does not exist: {root}")
    if not generation.is_dir():
        raise RunnerError(f"Generation does not exist: {generation}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError(f"Python executable is invalid: {python}")
    if previous is not None and not previous.is_dir():
        raise RunnerError(f"Previous Incident V3 release does not exist: {previous}")
    if not args.skip_tests and (
        node is None or not node.is_file() or not os.access(node, os.X_OK)
    ):
        raise RunnerError("Node executable is required for UI verification")
    tag = args.job_tag or utc_tag()
    paths = _paths(root, tag)
    job = Path(paths["job_dir"])
    if job.exists():
        raise RunnerError(f"Job tag already exists: {tag}")
    job.mkdir(parents=True)
    temp = (args.temp_dir or root / "duckdb_tmp").expanduser().resolve()
    temp.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "incident-v3-runner/2",
        "job_tag": tag,
        "status": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "current_stage": "starting",
        "config": {
            "root": str(root), "repo_dir": str(repo),
            "generation_dir": str(generation), "python": str(python),
            "previous_incident_dir": str(previous) if previous is not None else None,
            "first_release": bool(args.first_release),
            "node": str(node) if node is not None else None,
            "baseline_through": args.baseline_through,
            "threads": args.threads, "memory_limit": args.memory_limit,
            "temp_dir": str(temp), "heartbeat_seconds": args.heartbeat_seconds,
            "skip_tests": args.skip_tests,
            "capture_stage9_replay": bool(args.capture_stage9_replay),
        },
        "paths": paths,
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
    pointer = root.expanduser().resolve() / "logs" / "latest_incident_v3_job.txt"
    if not pointer.is_file():
        raise RunnerError(f"No latest Incident V3 job pointer exists: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise RunnerError(f"Latest Incident V3 job pointer is empty: {pointer}")
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
                state.update(
                    {"status": "running", "pid": os.getpid(), "resumed_at": utc_now()}
                )
                _save(state)
                return _execute(state, _logger(Path(state["paths"]["runner_log"])))
        state = _new_state(args)
        root = Path(state["config"]["root"])
        with _lock(root):
            _save(state)
            job = Path(state["paths"]["job_dir"])
            atomic_write_text(job / "runner.pid", f"{os.getpid()}\n")
            atomic_write_text(root / "logs" / "latest_incident_v3_job.txt", f"{job}\n")
            return _execute(state, _logger(Path(state["paths"]["runner_log"])))
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
