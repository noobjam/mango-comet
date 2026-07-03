"""Durable, resumable orchestration for the Archetype V2 Phase A workflow."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .archetype_runner_validation import (
    discover_generation,
    job_paths as _paths,
    resolve_rapids_python,
    validate_generation,
)
from .archetype_runner_stages import (
    build as _build,
    evaluate as _evaluate,
    gate_result as _gate_result,
    preflight as _preflight,
)
from .runner_process import (
    RunnerError,
    atomic_write_text,
    finish_state as _finish,
    load_json,
    runner_lock as _lock,
    save_state as _save,
    setup_logger,
    utc_tag,
    utc_now,
)


__all__ = ["read_job_state", "resume_job", "run_new_job"]


def _execute(state: dict[str, Any], logger: logging.Logger) -> int:
    try:
        _preflight(state, logger)
        _build(state, logger)
        return _evaluate(state, logger)
    except RunnerError as exc:
        logger.error("%s", exc)
        return _finish(state, "failed", exc.exit_code, error=str(exc))
    except KeyboardInterrupt:
        logger.error("Runner interrupted")
        return _finish(state, "interrupted", 130, error="Interrupted")
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 143
        logger.error("Runner terminated with status %s", code)
        return _finish(state, "interrupted", code, error="Terminated")
    except Exception as exc:
        logger.exception("Unexpected runner failure")
        return _finish(state, "failed", 2, error=f"Unexpected failure: {exc}")


def run_new_job(config: dict[str, Any], *, tag: str | None = None) -> int:
    root = Path(config["root"]).expanduser().resolve()
    if not root.is_dir():
        raise RunnerError(f"Runner root does not exist: {root}")
    with _lock(root):
        _assert_no_orphan_child(root)
        generation = Path(config["generation_dir"]) if config.get("generation_dir") else discover_generation(
            root, as_of=str(config["as_of"])
        )
        validate_generation(generation, as_of=config.get("as_of"))
        rapids = resolve_rapids_python(root, Path(config["rapids_python"]) if config.get("rapids_python") else None)
        job_tag = tag or utc_tag()
        paths = _paths(root, job_tag, str(config["training_cutoff"]))
        job = Path(paths["job_dir"])
        if any(Path(paths[name]).exists() for name in ("job_dir", "model_dir", "evaluation_dir")):
            raise RunnerError(f"Job tag already has output; choose another tag: {job_tag}")
        job.mkdir(parents=True)
        normalized = config | {
            "root": str(root), "generation_dir": str(generation.resolve()),
            "rapids_python": str(rapids), "repo_dir": str(Path(config["repo_dir"]).resolve()),
            "temp_dir": str(Path(config["temp_dir"]).expanduser().resolve()),
        }
        state = {
            "schema_version": "archetype-v2-runner/1", "job_tag": job_tag,
            "status": "running", "started_at": utc_now(), "pid": os.getpid(),
            "current_stage": "starting", "config": normalized, "paths": paths, "stages": {},
        }
        try:
            _save(state)
            atomic_write_text(job / "runner.pid", f"{os.getpid()}\n")
            atomic_write_text(root / "logs" / "latest_archetype_v2_job.txt", f"{job}\n")
            logger = setup_logger(Path(paths["runner_log"]))
        except Exception as exc:
            return _finish(state, "failed", 2, error=f"Runner startup failed: {exc}")
        logger.info("Archetype V2 job %s — outputs are diagnostic and are not published to the map", job_tag)
        return _execute(state, logger)


def resume_job(job_dir: Path) -> int:
    state = read_job_state(job_dir)
    root = Path(state["config"]["root"])
    with _lock(root):
        _assert_no_orphan_child(root)
        state.update({"status": "running", "pid": os.getpid(), "resumed_at": utc_now()})
        try:
            _save(state)
            atomic_write_text(job_dir / "runner.pid", f"{os.getpid()}\n")
            logger = setup_logger(Path(state["paths"]["runner_log"]))
        except Exception as exc:
            return _finish(state, "failed", 2, error=f"Resume startup failed: {exc}")
        logger.info("Resuming Archetype V2 job %s", state["job_tag"])
        return _execute(state, logger)


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


def _assert_no_orphan_child(root: Path) -> None:
    for path in sorted((root / "jobs").glob("archetype_v2_*/state.json")):
        try:
            previous = load_json(path)
            active = previous.get("active_process") or {}
            if previous.get("status") == "running" and active.get("pid") and _pid_exists(int(active["pid"])):
                raise RunnerError(f"Recorded child PID {active['pid']} is still alive in {path}", 75)
        except RunnerError:
            raise
        except (OSError, TypeError, ValueError):
            continue


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
