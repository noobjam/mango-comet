"""Process and atomic-state helpers for the Archetype V2 VM runner."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


__all__ = [
    "RunnerError",
    "atomic_write_json",
    "atomic_write_text",
    "finish_state",
    "load_json",
    "runner_lock",
    "run_job_stage",
    "run_stage",
    "save_state",
    "setup_logger",
    "stage_state",
    "utc_now",
    "utc_tag",
]


class RunnerError(RuntimeError):
    """Expected fail-closed runner error with a stable process exit code."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"Expected a JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(Path(state["paths"]["job_dir"]) / "state.json", state)


def stage_state(state: dict[str, Any], name: str, status: str, **values: Any) -> None:
    state["current_stage"] = name
    state.setdefault("stages", {}).setdefault(name, {}).update(
        {"status": status, "updated_at": utc_now(), **values}
    )
    save_state(state)


def finish_state(
    state: dict[str, Any], status: str, exit_code: int, **values: Any
) -> int:
    state.update(
        {"status": status, "exit_code": exit_code, "finished_at": utc_now(), **values}
    )
    save_state(state)
    atomic_write_text(Path(state["paths"]["job_dir"]) / "status", f"{exit_code}\n")
    return exit_code


@contextmanager
def runner_lock(root: Path) -> Iterator[None]:
    lock_path = root / "logs" / "archetype_v2_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"Another Archetype V2 runner holds {lock_path}", 75) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"archetype-v2-runner.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _rss_gib(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kibibytes = int(line.split()[1])
                return f"{kibibytes / 1024 / 1024:.1f} GiB"
    except (OSError, ValueError, IndexError):
        pass
    return "unavailable"


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()


def run_stage(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    logger: logging.Logger,
    label: str,
    heartbeat_seconds: int,
    cwd: Path,
    env: Mapping[str, str],
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int, int, str], None] | None = None,
) -> int:
    """Run one shell-free child and emit honest time/RSS heartbeats."""
    started = time.monotonic()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            process = subprocess.Popen(
                list(command), cwd=cwd, env=dict(env), stdout=stdout, stderr=stderr,
                text=True, shell=False, start_new_session=True,
            )
        except OSError as exc:
            raise RunnerError(f"{label} could not start: {exc}") from exc
        logger.info("%s started — pid %s", label, process.pid)
        try:
            if on_start:
                on_start(process.pid)
            while True:
                try:
                    return_code = process.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    rss = _rss_gib(process.pid)
                    if on_heartbeat:
                        on_heartbeat(process.pid, elapsed, rss)
                    logger.info(
                        "%s running — elapsed %02d:%02d:%02d — pid %s — RSS %s",
                        label,
                        elapsed // 3600,
                        (elapsed % 3600) // 60,
                        elapsed % 60,
                        process.pid,
                        rss,
                    )
        except BaseException:
            _terminate(process)
            raise
    elapsed = int(time.monotonic() - started)
    logger.info("%s finished — exit %s — elapsed %ss", label, return_code, elapsed)
    return return_code


def run_job_stage(
    state: dict[str, Any], logger: logging.Logger, label: str,
    command: list[str], stdout: Path, stderr: Path,
) -> int:
    config = state["config"]
    env = os.environ.copy()
    server = str(Path(config["repo_dir"]) / "server")
    env.update({
        "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": server + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    })

    def record(pid: int, elapsed: int = 0, rss: str = "starting") -> None:
        state["active_process"] = {
            "label": label, "pid": pid, "elapsed_seconds": elapsed,
            "rss": rss, "heartbeat_at": utc_now(),
        }
        save_state(state)

    try:
        return run_stage(
            command, stdout_path=stdout, stderr_path=stderr, logger=logger, label=label,
            heartbeat_seconds=int(config["heartbeat_seconds"]), cwd=Path(config["repo_dir"]),
            env=env, on_start=record, on_heartbeat=record,
        )
    finally:
        state.pop("active_process", None)
        save_state(state)
