#!/usr/bin/env python3
"""Run, resume, or inspect the complete Archetype V2 Phase A sequence."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import signal
import sys

from story_monitor.archetype_runner import read_job_state, resume_job, run_new_job
from story_monitor.runner_process import RunnerError


DEFAULT_ROOT = Path("/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1")


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


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Durably run Archetype V2 preflight, GPU build, and evaluation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Start a new immutable Phase A job.")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--generation-dir", type=Path)
    run.add_argument("--as-of", type=_iso_date, help="Required only for safe generation discovery.")
    run.add_argument("--rapids-python", type=Path)
    run.add_argument("--training-through", type=_iso_date, default="2025-12-31")
    run.add_argument("--gpu", type=_nonnegative, default=0)
    run.add_argument("--threads", type=_positive, default=32)
    run.add_argument("--memory-limit", default="96GB")
    run.add_argument("--temp-dir", type=Path)
    run.add_argument("--heartbeat-seconds", type=_positive, default=30)
    run.add_argument("--job-tag", help="Optional unique UTC-like tag for deterministic paths.")
    run.add_argument("--skip-tests", action="store_true", help="Skip focused V2 tests (not recommended).")

    resume = commands.add_parser("resume", help="Resume the same immutable job paths.")
    resume.add_argument("--job-dir", type=Path, required=True)

    status = commands.add_parser("status", help="Print the latest or selected job state.")
    status.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    status.add_argument("--job-dir", type=Path)
    return parser


def _latest_job(root: Path) -> Path:
    pointer = root.expanduser().resolve() / "logs" / "latest_archetype_v2_job.txt"
    if not pointer.is_file():
        raise RunnerError(f"No latest job pointer exists: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise RunnerError(f"Latest job pointer is empty: {pointer}")
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
            return resume_job(args.job_dir.expanduser().resolve())
        if args.generation_dir is None and args.as_of is None:
            raise RunnerError("Pass --generation-dir, or pass --as-of for safe discovery")
        root = args.root.expanduser().resolve()
        temp_dir = args.temp_dir or root / "duckdb_tmp"
        config = {
            "root": str(root),
            "repo_dir": str(Path(__file__).resolve().parent.parent),
            "generation_dir": str(args.generation_dir) if args.generation_dir else None,
            "as_of": args.as_of,
            "rapids_python": str(args.rapids_python) if args.rapids_python else None,
            "training_cutoff": args.training_through,
            "gpu": args.gpu,
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "temp_dir": str(temp_dir),
            "heartbeat_seconds": args.heartbeat_seconds,
            "skip_tests": args.skip_tests,
        }
        return run_new_job(config, tag=args.job_tag)
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
