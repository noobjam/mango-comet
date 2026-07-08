#!/usr/bin/env python3
"""Durably build and publish one immutable V4-native story replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
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
RUNNER_SCHEMA_VERSION = "incident-story-replay-v4-runner/1"
REPLAY_SCHEMA_VERSION = "crop-impact-incident-story-replay-v4/1"
REPLAY_MODE = "crop_incident_story_replay_v4"
CHECKPOINT_SCHEMA_VERSION = "incident-story-replay-checkpoint-v4/2"
VIEWER_SCHEMA_VERSION = "crop-incident-viewer-v4/2"
VIEWER_MODE = "crop_incident_v4_dual_clock"
EVIDENCE_FILES = (
    "crop_day_context_v4.parquet",
    "field_day_pressure_v4.parquet",
    "field_s2_acquisition_v4.parquet",
)
FOCUSED_TESTS = (
    "server.test_incident_replay_context_v4",
    "server.test_incident_crosswalk_v4",
    "server.test_incident_story_replay_v4",
    "server.test_incident_cells_v3",
    "server.test_incident_story_states_v3",
    "server.test_incident_viewer_v4",
    "server.test_story_map_server",
    "server.test_run_incident_story_replay_v4",
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


def _replay_partitions(value: str) -> int:
    parsed = _positive(value)
    if parsed > 1024:
        raise argparse.ArgumentTypeError("Replay partitions must be between 1 and 1024")
    return parsed


def _job_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Job tag must be 1-64 letters, digits, dots, underscores, or hyphens"
        )
    return value


def _executable_path(value: Path) -> Path:
    # Keep a virtualenv launcher path instead of resolving its interpreter symlink.
    return Path(os.path.abspath(value.expanduser()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Start a new durable V4-native story replay job."
    )
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--evidence-dir", type=Path, required=True)
    run.add_argument("--geometry-parquet", type=Path, required=True)
    run.add_argument("--audit-incident-dir", type=Path, required=True)
    run.add_argument("--baseline-through", type=_iso_date, required=True)
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument(
        "--node",
        type=Path,
        default=Path(shutil.which("node")) if shutil.which("node") else None,
    )
    run.add_argument("--threads", type=_positive, default=32)
    run.add_argument("--replay-partitions", type=_replay_partitions, default=64)
    run.add_argument("--memory-limit", default="96GB")
    run.add_argument("--temp-dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=_positive, default=30)
    run.add_argument("--job-tag", type=_job_tag)
    run.add_argument("--skip-tests", action="store_true")

    resume = commands.add_parser("resume", help="Resume missing stages of one job.")
    resume.add_argument("--job-dir", type=Path, required=True)
    status = commands.add_parser("status", help="Print one job's durable state.")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    status.add_argument("--job-dir", type=Path)
    return parser


def _build_stage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--geometry-parquet", type=Path, required=True)
    parser.add_argument("--audit-incident-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-through", type=_iso_date, required=True)
    parser.add_argument("--threads", type=_positive, required=True)
    parser.add_argument("--replay-partitions", type=_replay_partitions, required=True)
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    return parser


def _paths(root: Path, tag: str) -> dict[str, str]:
    job = root / "jobs" / f"incident_story_replay_v4_{tag}"
    return {
        "job_dir": str(job),
        "checkpoint_dir": str(job / "checkpoints"),
        "runner_log": str(job / "runner.log"),
        "tests_stdout": str(job / "tests.stdout.log"),
        "tests_stderr": str(job / "tests.stderr.log"),
        "ui_tests_stdout": str(job / "ui-tests.stdout.log"),
        "ui_tests_stderr": str(job / "ui-tests.stderr.log"),
        "ui_syntax_stdout": str(job / "ui-syntax.stdout.log"),
        "ui_syntax_stderr": str(job / "ui-syntax.stderr.log"),
        "build_json": str(job / "build.json"),
        "build_stderr": str(job / "build.stderr.log"),
        "export_json": str(job / "export.json"),
        "export_stderr": str(job / "export.stderr.log"),
        "smoke_stdout": str(job / "smoke.stdout.log"),
        "smoke_stderr": str(job / "smoke.stderr.log"),
        "incident_dir": str(root / "releases" / f"incidents_v4_replay_{tag}"),
        "viewer_dir": str(
            root / "releases" / f"incident_viewer_v4_replay_{tag}"
        ),
    }


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"incident-story-replay-v4.{path.parent.name}")
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
    path = root / "logs" / "incident_story_replay_v4_runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(
                f"Another Incident Story Replay V4 runner holds {path}", 75
            ) from exc
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
        "status": "running",
        "updated_at": utc_now(),
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
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    run = manifest.get("run") or {}
    return str(run.get("status")) == "complete" and (
        mode is None or str(manifest.get("mode") or run.get("mode")) == mode
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_replay(
    path: Path, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = load_json(manifest_path)
    run = manifest.get("run") or {}
    validation = manifest.get("validation") or {}
    if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("V4 replay schema_version does not match the runner")
    if str(manifest.get("mode") or run.get("mode") or "") != REPLAY_MODE:
        raise ValueError("Incident release is not a V4-native story replay")
    if run.get("status") != "complete" or run.get("immutable") is not True:
        raise ValueError("V4 replay is not complete and immutable")
    if validation.get("passed") is not True:
        raise ValueError("V4 replay final artifact validation did not pass")
    if int(validation.get("membership_counter_mismatch_count", -1)) != 0:
        raise ValueError("V4 replay membership counters do not reconcile")
    row_counts = validation.get("row_counts") or {}
    if int(row_counts.get("incident_weekly_state", 0)) < 1:
        raise ValueError("V4 replay contains no incident weekly states")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("V4 replay has no immutable artifact inventory")
    if "old_to_new_incident_crosswalk.parquet" not in artifacts:
        raise ValueError("V4 replay inventory is missing the audit crosswalk")
    disk_names = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if disk_names != set(artifacts):
        raise ValueError("V4 replay artifact inventory does not match disk")
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid V4 replay inventory entry: {name!r}")
        artifact = path / relative
        if (
            not isinstance(expected, dict)
            or artifact.stat().st_size != int(expected.get("size_bytes", -1))
            or _file_sha256(artifact) != str(expected.get("sha256") or "")
        ):
            raise ValueError(f"V4 replay artifact changed: {name}")
    if state is not None:
        config = state["config"]
        expected_source = {
            "evidence_manifest_sha256": _file_sha256(
                Path(config["evidence_dir"]) / "manifest.json"
            ),
            "geometry_sha256": _file_sha256(Path(config["geometry_parquet"])),
            "audit_incident_manifest_sha256": _file_sha256(
                Path(config["audit_incident_dir"]) / "manifest.json"
            ),
            "audit_incident_membership_sha256": _file_sha256(
                Path(config["audit_incident_dir"]) / "incident_membership.parquet"
            ),
        }
        source = manifest.get("source") or {}
        if any(source.get(name) != value for name, value in expected_source.items()):
            raise ValueError("V4 replay provenance does not match this runner job")
        if run.get("baseline_through") != config["baseline_through"]:
            raise ValueError("V4 replay baseline does not match this runner job")
    return {
        "status": "valid",
        "schema_version": REPLAY_SCHEMA_VERSION,
        "mode": REPLAY_MODE,
        "generation_id": run.get("generation_id"),
        "row_counts": row_counts,
        "crosswalk_rows": validation.get("crosswalk_rows"),
    }


def _require_native_viewer_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path / "manifest.json")
    run = manifest.get("run") or {}
    semantics = manifest.get("semantics") or {}
    if manifest.get("schema_version") != VIEWER_SCHEMA_VERSION:
        raise ValueError("Native replay viewer is not V4 schema version 2")
    if str(manifest.get("mode") or run.get("mode") or "") != VIEWER_MODE:
        raise ValueError("Native replay viewer changed the compatible V4 mode")
    if run.get("native_replay") is not True:
        raise ValueError("V4 viewer is not marked as a native replay")
    expected_true = (
        "lifecycle_state_recomputed_from_v4",
        "component_absence_replayed_from_v4",
        "full_lifecycle_replay_supported",
        "lifecycle_causal_ownership_claimed",
    )
    if any(semantics.get(name) is not True for name in expected_true):
        raise ValueError("Native replay viewer did not preserve V4 lifecycle ownership")
    if semantics.get("source_state_preserved") is not False:
        raise ValueError("Native replay viewer incorrectly preserved an old story spine")
    return manifest


def _validated_viewer(
    path: Path,
    state: dict[str, Any] | None = None,
    source_adapter: Path | None = None,
) -> dict[str, Any]:
    from story_monitor.incident_viewer_v4 import validate_viewer_directory

    validation = validate_viewer_directory(path)
    manifest = _require_native_viewer_manifest(path)
    if state is not None:
        paths, config = state["paths"], state["config"]
        adapter = source_adapter or (
            Path(paths["checkpoint_dir"]) / "01_context" / "source_generation"
        )
        expected_source = {
            "incident_manifest_sha256": _file_sha256(
                Path(paths["incident_dir"]) / "manifest.json"
            ),
            "evidence_manifest_sha256": _file_sha256(
                Path(config["evidence_dir"]) / "manifest.json"
            ),
            "source_generation_manifest_sha256": _file_sha256(
                adapter / "manifest.json"
            ),
        }
        source = manifest.get("source") or {}
        if any(source.get(name) != value for name, value in expected_source.items()):
            raise ValueError("Native V4 viewer provenance does not match this replay job")
    return validation


def _validated_source_adapter(state: dict[str, Any]) -> Path:
    paths = state["paths"]
    checkpoint = Path(paths["checkpoint_dir"]) / "01_context"
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = load_json(checkpoint_manifest_path)
    run = checkpoint_manifest.get("run") or {}
    if (
        checkpoint_manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or checkpoint_manifest.get("stage") != "context"
        or run.get("status") != "complete"
        or run.get("immutable") is not True
    ):
        raise ValueError("Replay context checkpoint is not complete and immutable")

    replay_manifest = load_json(Path(paths["incident_dir"]) / "manifest.json")
    expected_checkpoint = (replay_manifest.get("checkpoints") or {}).get("01_context")
    actual_checkpoint = {
        "sha256": _file_sha256(checkpoint_manifest_path),
        "size_bytes": checkpoint_manifest_path.stat().st_size,
    }
    if expected_checkpoint != actual_checkpoint:
        raise ValueError("Replay context checkpoint is not bound to the story release")

    artifacts = checkpoint_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Replay context checkpoint has no artifact inventory")
    disk_names = {
        item.relative_to(checkpoint).as_posix()
        for item in checkpoint.rglob("*")
        if item.is_file() and item != checkpoint_manifest_path
    }
    if disk_names != set(artifacts):
        raise ValueError("Replay context checkpoint inventory does not match disk")
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid replay checkpoint inventory entry: {name!r}")
        artifact = checkpoint / relative
        actual = {
            "sha256": _file_sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        if actual != expected:
            raise ValueError(f"Replay context checkpoint artifact changed: {name}")

    source_adapter = checkpoint / "source_generation"
    source_manifest = source_adapter / "manifest.json"
    if not source_adapter.is_dir() or not source_manifest.is_file():
        raise ValueError("Replay context checkpoint has no source adapter")
    expected_source_sha = (replay_manifest.get("source") or {}).get(
        "generation_manifest_sha256"
    )
    if expected_source_sha != _file_sha256(source_manifest):
        raise ValueError("Replay source adapter is not bound to the story release")
    return source_adapter


def _mark_stage_skipped(state: dict[str, Any], name: str, reason: str) -> None:
    state.setdefault("stages", {})[name] = {
        "status": "skipped",
        "reason": reason,
        "updated_at": utc_now(),
    }
    _save(state)


def _mark_existing(state: dict[str, Any], name: str) -> None:
    state.setdefault("stages", {})[name] = {
        "status": "complete",
        "skipped_existing": True,
        "updated_at": utc_now(),
    }
    _save(state)


def _node_syntax_command(node: str, repo: Path) -> list[str]:
    files = sorted((repo / "server" / "static").glob("*.js"))
    if not files:
        raise RunnerError("No static JavaScript files found", 2)
    return [node, "-e", NODE_SYNTAX_SCRIPT, *(str(path) for path in files)]


def _build_command(state: dict[str, Any]) -> list[str]:
    config, paths = state["config"], state["paths"]
    return [
        config["python"],
        "server/run_incident_story_replay_v4.py",
        "_build",
        "--evidence-dir",
        config["evidence_dir"],
        "--geometry-parquet",
        config["geometry_parquet"],
        "--audit-incident-dir",
        config["audit_incident_dir"],
        "--output-dir",
        paths["incident_dir"],
        "--checkpoint-dir",
        paths["checkpoint_dir"],
        "--baseline-through",
        config["baseline_through"],
        "--threads",
        str(config["threads"]),
        "--replay-partitions",
        str(config["replay_partitions"]),
        "--memory-limit",
        config["memory_limit"],
        "--temp-dir",
        config["temp_dir"],
    ]


def _export_command(state: dict[str, Any]) -> list[str]:
    config, paths = state["config"], state["paths"]
    source_adapter = Path(paths["checkpoint_dir"]) / "01_context" / "source_generation"
    return [
        config["python"],
        "server/export_incident_viewer_v4.py",
        "--incident-dir",
        paths["incident_dir"],
        "--evidence-dir",
        config["evidence_dir"],
        "--source-generation-dir",
        str(source_adapter),
        "--output-dir",
        paths["viewer_dir"],
        "--native-replay",
        "--threads",
        str(config["threads"]),
        "--memory-limit",
        config["memory_limit"],
        "--temp-dir",
        config["temp_dir"],
    ]


def _input_fingerprint(path: Path, *, digest: bool) -> dict[str, Any]:
    stat = path.stat()
    fingerprint: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if digest:
        fingerprint["sha256"] = _file_sha256(path)
    return fingerprint


def _verify_immutable_inputs(state: dict[str, Any]) -> None:
    for label, expected in state.get("input_fingerprints", {}).items():
        path = Path(expected["path"])
        try:
            actual = _input_fingerprint(path, digest="sha256" in expected)
        except OSError as exc:
            raise RunnerError(f"Immutable input disappeared: {label}: {path}", 21) from exc
        if actual != expected:
            raise RunnerError(f"Immutable input changed: {label}: {path}", 21)


def _require_complete_input_manifest(path: Path, label: str) -> None:
    try:
        manifest = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Cannot read {label} input manifest: {path}") from exc
    run = manifest.get("run") or {}
    if run.get("status") != "complete" or run.get("immutable") is not True:
        raise RunnerError(f"{label} input is not complete and immutable: {path}")


def _smoke_command(state: dict[str, Any]) -> list[str]:
    python = state["config"]["python"]
    viewer = state["paths"]["viewer_dir"]
    smoke = (
        "from pathlib import Path; import json; "
        "from story_map_server import Settings,StoryMapStore; "
        f"p=Path({viewer!r}); "
        "m=json.loads((p/'manifest.json').read_text()); "
        "assert m['schema_version']=='crop-incident-viewer-v4/2'; "
        "assert m['semantics']['lifecycle_state_recomputed_from_v4'] is True; "
        "s=Settings(run_dir=p,static_dir=Path('server/static'),host='127.0.0.1',"
        "port=8877,raster_tiles='',raster_attribution='',default_feature_limit=5000,"
        "max_feature_limit=20000,log_level='INFO'); st=StoryMapStore(s); "
        "assert st.health()['ok'] and st.has_incident_v4(); t=st.v4_timeline(); "
        "assert t['days']; d=next(x['calendar_date'] for x in t['days'] "
        "if x['source_day_present']); f=st.v4_frame(calendar_date=d,bbox=None,limit=0); "
        "x=f['meta']; assert x['complete_country_representation']; "
        "assert x['accounted_field_count']==x['source_field_count']; "
        "assert not x['country_representation_truncated']; "
        "print(json.dumps({'status':'ok','day':d,'grid':"
        "len(f['field_overview']['features'])}))"
    )
    return [python, "-c", smoke]


def _execute(state: dict[str, Any], logger: logging.Logger) -> int:
    config, paths = state["config"], state["paths"]
    python = str(config["python"])
    try:
        _verify_immutable_inputs(state)
        if config["skip_tests"]:
            for name in ("python_tests", "ui_tests", "ui_syntax"):
                if name not in state.get("stages", {}):
                    _mark_stage_skipped(state, name, "--skip-tests")
        else:
            node = str(config["node"])
            if state.get("stages", {}).get("python_tests", {}).get("status") != "complete":
                _stage(
                    state,
                    logger,
                    "python_tests",
                    "[1/6 PYTHON TESTS]",
                    [python, "-m", "unittest", *FOCUSED_TESTS],
                    "tests_stdout",
                    "tests_stderr",
                )
            if state.get("stages", {}).get("ui_tests", {}).get("status") != "complete":
                _stage(
                    state,
                    logger,
                    "ui_tests",
                    "[2/6 UI TESTS]",
                    [node, "--test", "server/test_story_ui.mjs"],
                    "ui_tests_stdout",
                    "ui_tests_stderr",
                )
            if state.get("stages", {}).get("ui_syntax", {}).get("status") != "complete":
                _stage(
                    state,
                    logger,
                    "ui_syntax",
                    "[3/6 UI SYNTAX]",
                    _node_syntax_command(node, Path(config["repo_dir"])),
                    "ui_syntax_stdout",
                    "ui_syntax_stderr",
                )

        incident = Path(paths["incident_dir"])
        if incident.exists() or incident.is_symlink():
            replay_validation = _validated_replay(incident, state)
            _mark_existing(state, "build")
        else:
            _stage(
                state,
                logger,
                "build",
                "[4/6 V4-NATIVE STORY REPLAY]",
                _build_command(state),
                "build_json",
                "build_stderr",
            )
            replay_validation = _validated_replay(incident, state)
        state["replay_validation"] = replay_validation
        _save(state)

        source_adapter = _validated_source_adapter(state)

        viewer = Path(paths["viewer_dir"])
        if viewer.exists() or viewer.is_symlink():
            viewer_validation = _validated_viewer(viewer, state, source_adapter)
            _mark_existing(state, "export")
        else:
            _stage(
                state,
                logger,
                "export",
                "[5/6 NATIVE V4 VIEWER EXPORT]",
                _export_command(state),
                "export_json",
                "export_stderr",
            )
            viewer_validation = _validated_viewer(viewer, state, source_adapter)
        state["viewer_validation"] = viewer_validation
        _save(state)

        if state.get("stages", {}).get("smoke", {}).get("status") != "complete":
            _stage(
                state,
                logger,
                "smoke",
                "[6/6 SERVER SMOKE]",
                _smoke_command(state),
                "smoke_stdout",
                "smoke_stderr",
            )
        state.update(
            {
                "status": "complete",
                "exit_code": 0,
                "finished_at": utc_now(),
                "current_stage": "complete",
            }
        )
        state.pop("error", None)
        _save(state)
        atomic_write_text(Path(paths["job_dir"]) / "status", "0\n")
        logger.info("V4-native story replay viewer complete — %s", viewer)
        return 0
    except KeyboardInterrupt:
        code, message = 130, "Interrupted"
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 143
        message = "Terminated"
    except RunnerError as exc:
        code, message = exc.exit_code, str(exc)
    except Exception as exc:  # runner boundary records validation and runtime failures
        logger.exception("Unexpected V4-native story replay runner failure")
        code, message = 2, f"Unexpected failure: {exc}"
    logger.error("%s", message)
    current_stage = str(state.get("current_stage") or "")
    stage = (state.get("stages") or {}).get(current_stage)
    if isinstance(stage, dict) and stage.get("status") == "running":
        stage.update(
            {
                "status": "interrupted" if code in {130, 143} else "failed",
                "exit_code": code,
                "updated_at": utc_now(),
            }
        )
    state.update(
        {
            "status": "interrupted" if code in {130, 143} else "failed",
            "exit_code": code,
            "finished_at": utc_now(),
            "error": message,
        }
    )
    _save(state)
    atomic_write_text(Path(paths["job_dir"]) / "status", f"{code}\n")
    return code


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _new_state(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    evidence = args.evidence_dir.expanduser().resolve()
    geometry = args.geometry_parquet.expanduser().resolve()
    audit = args.audit_incident_dir.expanduser().resolve()
    python = _executable_path(args.python)
    node = _executable_path(args.node) if args.node is not None else None
    repo = Path(__file__).resolve().parent.parent
    if not root.is_dir():
        raise RunnerError(f"Runner root does not exist: {root}")
    if not evidence.is_dir():
        raise RunnerError(f"V4 evidence directory does not exist: {evidence}")
    if not geometry.is_file():
        raise RunnerError(f"Geometry parquet does not exist: {geometry}")
    if not audit.is_dir():
        raise RunnerError(f"Audit incident directory does not exist: {audit}")
    required = {
        "evidence_manifest": evidence / "manifest.json",
        **{f"evidence_{name}": evidence / name for name in EVIDENCE_FILES},
        "geometry": geometry,
        "audit_manifest": audit / "manifest.json",
        "audit_membership": audit / "incident_membership.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RunnerError("Replay inputs are missing: " + ", ".join(missing))
    _require_complete_input_manifest(
        required["evidence_manifest"], "V4 evidence"
    )
    _require_complete_input_manifest(
        required["audit_manifest"], "audit incident"
    )
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RunnerError(f"Python executable is invalid: {python}")
    if not args.skip_tests and (
        node is None or not node.is_file() or not os.access(node, os.X_OK)
    ):
        raise RunnerError("Node executable is required for UI verification")

    tag = args.job_tag or utc_tag()
    paths = _paths(root, tag)
    job = Path(paths["job_dir"])
    incident = Path(paths["incident_dir"])
    viewer = Path(paths["viewer_dir"])
    temp = (args.temp_dir or root / "duckdb_tmp").expanduser().resolve()
    for target, label in ((job, "job"), (incident, "incident"), (viewer, "viewer"), (temp, "temp")):
        for source in (evidence, audit, geometry):
            if _paths_overlap(target, source):
                raise RunnerError(
                    f"Replay {label} path overlaps immutable input: {target} and {source}"
                )
    for target in (job, incident, viewer):
        if _paths_overlap(temp, target):
            raise RunnerError(
                f"Replay temp path overlaps a durable output: {temp} and {target}"
            )
    if any(
        target.exists() or target.is_symlink()
        for target in (job, incident, viewer)
    ):
        raise RunnerError(f"Job tag already exists: {tag}")

    input_fingerprints = {
        label: _input_fingerprint(
            path, digest=label in {"evidence_manifest", "audit_manifest"}
        )
        for label, path in required.items()
    }
    temp.mkdir(parents=True, exist_ok=True)
    job.mkdir(parents=True)
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "job_tag": tag,
        "status": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "current_stage": "starting",
        "input_fingerprints": input_fingerprints,
        "config": {
            "root": str(root),
            "repo_dir": str(repo),
            "evidence_dir": str(evidence),
            "geometry_parquet": str(geometry),
            "audit_incident_dir": str(audit),
            "baseline_through": args.baseline_through,
            "python": str(python),
            "node": str(node) if node is not None else None,
            "threads": args.threads,
            "replay_partitions": args.replay_partitions,
            "memory_limit": args.memory_limit,
            "temp_dir": str(temp),
            "heartbeat_seconds": args.heartbeat_seconds,
            "skip_tests": args.skip_tests,
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
    resolved_job = job_dir.expanduser().resolve()
    state = load_json(resolved_job / "state.json")
    active = state.get("active_process") or {}
    if state.get("status") != "running":
        state["liveness"] = "terminal"
    elif active.get("pid") and _pid_exists(int(active["pid"])):
        state["liveness"] = "child_alive"
    elif state.get("pid") and _pid_exists(int(state["pid"])):
        state["liveness"] = "runner_alive"
    else:
        state["liveness"] = "stale"
    checkpoint_dir = Path(
        (state.get("paths") or {}).get("checkpoint_dir")
        or resolved_job / "checkpoints"
    )
    state["checkpoint_progress"] = _replay_checkpoint_progress(
        checkpoint_dir,
        expected_partitions=int(
            (state.get("config") or {}).get("replay_partitions") or 0
        ),
    )
    return state


def _replay_checkpoint_progress(
    checkpoint_dir: Path, *, expected_partitions: int
) -> dict[str, Any]:
    completed_stages = sorted(
        path.parent.name
        for path in checkpoint_dir.glob("[0-9][0-9]_*/manifest.json")
        if path.is_file()
    )
    progress: dict[str, Any] = {"completed_stages": completed_stages}
    roots = sorted(
        path
        for path in checkpoint_dir.glob(
            ".01_context-*/01_context/.replay-partitions"
        )
        if path.is_dir()
    )
    if not roots:
        return progress
    root = roots[-1]
    completed = len(
        list((root / "output" / "event_state_snapshots").glob("part-*.parquet"))
    )
    actual_partitions = {
        path.name
        for ledger in ("crop", "pressure", "s2")
        for path in (root / "input" / ledger).glob("replay_partition=*")
        if path.is_dir()
    }
    progress["context_replay"] = {
        "completed_partitions": completed,
        "total_partitions": len(actual_partitions),
        "configured_partitions": expected_partitions,
        "partial_work_reusable_on_resume": False,
    }
    return progress


def _latest_job(root: Path) -> Path:
    pointer = (
        root.expanduser().resolve()
        / "logs"
        / "latest_incident_story_replay_v4_job.txt"
    )
    if not pointer.is_file():
        raise RunnerError(f"No latest V4 story replay job pointer exists: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise RunnerError(f"Latest V4 story replay job pointer is empty: {pointer}")
    return Path(value)


def _run_build_stage(argv: list[str]) -> int:
    args = _build_stage_parser().parse_args(argv)
    from story_monitor.incident_story_replay_v4 import build_incident_story_replay_v4

    result = build_incident_story_replay_v4(
        args.evidence_dir,
        args.geometry_parquet,
        args.audit_incident_dir,
        args.output_dir,
        args.checkpoint_dir,
        baseline_through=args.baseline_through,
        threads=args.threads,
        replay_partitions=args.replay_partitions,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _handle_sigterm(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["_build"]:
        return _run_build_stage(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
    try:
        if args.command == "status":
            job = args.job_dir or _latest_job(args.root)
            print(json.dumps(read_job_state(job), indent=2, sort_keys=True))
            return 0
        if args.command == "resume":
            state = load_json(args.job_dir.expanduser().resolve() / "state.json")
            if state.get("schema_version") != RUNNER_SCHEMA_VERSION:
                raise RunnerError("Job state is not a V4 story replay runner state")
            root = Path(state["config"]["root"])
            with _lock(root):
                state.update(
                    {"status": "running", "pid": os.getpid(), "resumed_at": utc_now()}
                )
                for key in ("error", "exit_code", "finished_at"):
                    state.pop(key, None)
                _save(state)
                return _execute(state, _logger(Path(state["paths"]["runner_log"])))
        state = _new_state(args)
        root = Path(state["config"]["root"])
        with _lock(root):
            _save(state)
            job = Path(state["paths"]["job_dir"])
            atomic_write_text(job / "runner.pid", f"{os.getpid()}\n")
            atomic_write_text(
                root / "logs" / "latest_incident_story_replay_v4_job.txt",
                f"{job}\n",
            )
            return _execute(state, _logger(Path(state["paths"]["runner_log"])))
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
