#!/usr/bin/env python3
"""Run immutable, review-gated Incident V4 motif workflow stages."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
import time
from typing import Iterator

from story_monitor.incident_motif_workflow_v4 import (
    build_diagnostic_motif_release_v4,
    evaluate_prefix_release_v4,
    fit_reviewed_prefix_release_v4,
)
from story_monitor.incident_motifs_v4 import (
    MotifDiscoveryConfig,
    PrefixCalibrationConfig,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _integer_tuple(value: str, *, allow_zero: bool) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    minimum = 0 if allow_zero else 1
    if not parsed or any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(
            f"horizons must be comma-separated integers >= {minimum}"
        )
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("horizons must be sorted and unique")
    if allow_zero and parsed[0] != 0:
        raise argparse.ArgumentTypeError("S2 horizons must begin at zero")
    return parsed


def _weather_horizons(value: str) -> tuple[int, ...]:
    return _integer_tuple(value, allow_zero=False)


def _s2_horizons(value: str) -> tuple[int, ...]:
    return _integer_tuple(value, allow_zero=True)


def _add_prefix_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weather-day-horizons", type=_weather_horizons, default=(7, 14, 28, 56))
    parser.add_argument("--s2-acquisition-horizons", type=_s2_horizons, default=(0, 1, 2, 4))
    parser.add_argument("--minimum-training-support", type=_positive, default=20)
    parser.add_argument("--minimum-calibration-support", type=_positive, default=10)
    parser.add_argument("--radius-quantile", type=float, default=0.95)
    parser.add_argument("--margin-quantile", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    discover = commands.add_parser(
        "discover",
        help="Discover completed-story motifs and write a pending review template.",
    )
    discover.add_argument("--incident-dir", type=Path, required=True)
    discover.add_argument("--evidence-dir", type=Path, required=True)
    discover.add_argument("--viewer-dir", type=Path, required=True)
    discover.add_argument("--output-dir", type=Path, required=True)
    discover.add_argument("--train-through", type=_date, required=True)
    discover.add_argument("--calibration-through", type=_date, required=True)
    discover.add_argument("--evaluation-through", type=_date, required=True)
    discover.add_argument("--engine", choices=("cpu", "gpu"), required=True)
    discover.add_argument("--min-cluster-size", type=_positive, default=100)
    discover.add_argument("--min-samples", type=_positive, default=20)
    discover.add_argument("--diagnostic-radius-quantile", type=float, default=0.95)
    discover.add_argument("--threads", type=_positive, default=32)
    discover.add_argument("--memory-limit", default="96GB")
    discover.add_argument("--temp-dir", type=Path)
    discover.add_argument("--heartbeat-seconds", type=_positive, default=30)
    _add_prefix_config(discover)

    fit = commands.add_parser(
        "fit-prefix",
        help="Fit a frozen causal-prefix model from immutable expert reviews.",
    )
    fit.add_argument("--discovery-dir", type=Path, required=True)
    fit.add_argument("--review-overlay", type=Path, required=True)
    fit.add_argument("--reviewed-calibration-labels", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--heartbeat-seconds", type=_positive, default=30)
    _add_prefix_config(fit)

    evaluate = commands.add_parser(
        "evaluate",
        help="Replay a frozen prefix model on sealed holdout labels.",
    )
    evaluate.add_argument("--discovery-dir", type=Path, required=True)
    evaluate.add_argument("--prefix-model-dir", type=Path, required=True)
    evaluate.add_argument("--final-labels", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--heartbeat-seconds", type=_positive, default=30)
    return parser


def _prefix_config(args: argparse.Namespace) -> PrefixCalibrationConfig:
    return PrefixCalibrationConfig(
        weather_day_horizons=tuple(args.weather_day_horizons),
        s2_acquisition_horizons=tuple(args.s2_acquisition_horizons),
        minimum_training_support=args.minimum_training_support,
        minimum_calibration_support=args.minimum_calibration_support,
        radius_quantile=args.radius_quantile,
        margin_quantile=args.margin_quantile,
    )


def _rss() -> str:
    path = Path("/proc/self/status")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return " ".join(line.split()[1:3])
    return "unavailable"


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} {message}", file=sys.stderr, flush=True)


@contextmanager
def _heartbeat(label: str, seconds: int) -> Iterator[None]:
    stopped = Event()
    started = time.monotonic()

    def emit() -> None:
        while not stopped.wait(seconds):
            elapsed = int(time.monotonic() - started)
            _log(f"{label} running elapsed={elapsed}s rss={_rss()}")

    _log(f"{label} started pid={os.getpid()}")
    thread = Thread(target=emit, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1)
        _log(f"{label} finished elapsed={int(time.monotonic() - started)}s rss={_rss()}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with _heartbeat(args.command, args.heartbeat_seconds):
        if args.command == "discover":
            discovery = MotifDiscoveryConfig(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                diagnostic_radius_quantile=args.diagnostic_radius_quantile,
                engine=args.engine,
            )
            result = build_diagnostic_motif_release_v4(
                args.incident_dir,
                args.evidence_dir / "field_day_pressure_v4.parquet",
                args.evidence_dir / "field_s2_acquisition_v4.parquet",
                args.output_dir,
                train_through=args.train_through,
                calibration_through=args.calibration_through,
                evaluation_through=args.evaluation_through,
                config=discovery,
                prefix_config=_prefix_config(args),
                threads=args.threads,
                memory_limit=args.memory_limit,
                temp_dir=args.temp_dir,
                evidence_manifest_path=args.evidence_dir / "manifest.json",
                viewer_dir=args.viewer_dir,
            )
        elif args.command == "fit-prefix":
            result = fit_reviewed_prefix_release_v4(
                args.discovery_dir,
                args.review_overlay,
                args.reviewed_calibration_labels,
                args.output_dir,
                config=_prefix_config(args),
            )
        else:
            result = evaluate_prefix_release_v4(
                args.discovery_dir,
                args.prefix_model_dir,
                args.final_labels,
                args.output_dir,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "evaluate" and not result.get("hard_gates_passed", False):
        return 21
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
