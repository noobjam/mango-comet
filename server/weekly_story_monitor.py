from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import logging
from pathlib import Path
from typing import Any

from story_monitor.contracts import load_policy
from story_monitor.motif_export import export_motif_generation
from story_monitor.motifs import DiscoveryConfig, discover_motifs
from story_monitor.incident_policy_v3 import DEFAULT_INCIDENT_POLICY_V3_PATH
from story_monitor.incident_policy_v4 import DEFAULT_INCIDENT_POLICY_V4_PATH
from story_monitor.partitioned_pipeline import PartitionOptions, build_partitioned_generation
from story_monitor.pipeline import BOUNDED_V1_MAX_FIELDS, DEFAULT_POLICY_PATH, build_generation
from story_monitor.prefix_features import load_training_prefixes


DEFAULT_ECHO_PARQUET = Path(
    "/mnt/KSA-Oasis/fields_health_v2/rwanda_crop_risk_kb/final_field_daily_v4/"
    "rwanda_2025_2026_field_daily_risk_DELIVERABLE_WITH_CROP_AND_RISK_DRIVER_"
    "v4_WITH_SPECTRAL_ECHO_DAYS.parquet"
)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an ISO date (YYYY-MM-DD).") from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def _bounded_fields(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > BOUNDED_V1_MAX_FIELDS:
        raise argparse.ArgumentTypeError(
            f"Bounded V1 supports at most {BOUNDED_V1_MAX_FIELDS} fields."
        )
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_ECHO_PARQUET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--max-fields",
        type=_bounded_fields,
        help=(
            "Optional deterministic smoke-test bound (1-5000). Omit it to use the full "
            "single-scan field-partitioned pipeline."
        ),
    )
    parser.add_argument(
        "--partitions",
        type=_positive_int,
        default=128,
        help="Field-hash partitions for a full run (default: 128).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Concurrent partition workers; start with 4-8 on the VM after a smoke run.",
    )
    parser.add_argument("--threads", type=_positive_int, default=16, help="DuckDB source-scan threads.")
    parser.add_argument("--memory-limit", help="Optional DuckDB limit such as 96GB.")
    parser.add_argument("--temp-dir", type=Path, help="DuckDB spill directory on a large volume.")
    parser.add_argument(
        "--history-from",
        type=_date,
        help="Optional lower input date bound. Omitting it preserves all prior evidence.",
    )
    parser.add_argument(
        "--geometry-parquet",
        type=Path,
        help="Optional compatible field geometry copied into each viewer generation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable, prefix-safe weekly story-monitor generations. "
            "Starter thresholds are explicitly uncalibrated."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="Build one immutable as-of generation.")
    _add_common(update)
    update.add_argument("--as-of", type=_date, required=True)

    replay = subparsers.add_parser("replay", help="Build multiple immutable weekly cutoffs.")
    _add_common(replay)
    replay.add_argument("--date-from", "--from-as-of", dest="date_from", type=_date, required=True)
    replay.add_argument("--date-to", "--to-as-of", dest="date_to", type=_date, required=True)
    replay.add_argument("--step-days", type=_positive_int, default=7)

    train = subparsers.add_parser(
        "train-motifs", help="Discover a frozen HDBSCAN motif model from causal weekly prefixes."
    )
    train.add_argument("--generation-dir", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--training-through", type=_date, required=True)
    train.add_argument("--engine", choices=("cpu", "gpu", "auto"), default="cpu")
    train.add_argument("--min-cluster-size", type=_positive_int, default=100)
    train.add_argument("--min-samples", type=_positive_int, default=20)
    train.add_argument("--radius-quantile", type=float, default=0.95)
    train.add_argument("--assignment-margin", type=float, default=0.05)

    export = subparsers.add_parser(
        "export-motifs", help="Create an immutable viewer generation assigned to a frozen motif model."
    )
    export.add_argument("--generation-dir", type=Path, required=True)
    export.add_argument("--model-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)

    build_v2 = subparsers.add_parser(
        "build-archetypes-v2",
        help="Build one-anchor-per-event V2 diagnostic archetypes; does not publish to the map.",
    )
    build_v2.add_argument("--generation-dir", type=Path, required=True)
    build_v2.add_argument("--training-through", type=_date, required=True)
    build_v2.add_argument("--output-dir", type=Path, required=True)
    build_v2.add_argument("--engine", choices=("cpu", "gpu", "auto"), default="cpu")
    build_v2.add_argument("--radius-quantile", type=float, default=0.95)
    build_v2.add_argument("--assignment-margin", type=float, default=0.05)
    build_v2.add_argument("--threads", type=_positive_int, default=16)
    build_v2.add_argument("--memory-limit")
    build_v2.add_argument("--temp-dir", type=Path)

    evaluate_v2 = subparsers.add_parser(
        "evaluate-archetypes-v2",
        help="Run temporal-holdout and two-run stability gates for a frozen V2 model.",
    )
    evaluate_v2.add_argument("--model-dir", type=Path, required=True)
    evaluate_v2.add_argument("--output-dir", type=Path, required=True)
    evaluate_v2.add_argument(
        "--stability-runs", dest="stability_runs",
        type=_positive_int, default=2,
        help="Exactly two deterministic 80%% hazard-stratified subsample refits.",
    )
    incidents_v3 = subparsers.add_parser(
        "build-incidents-v3",
        help=(
            "Build immutable local crop-impact incident tracks and completed-story "
            "features; starter thresholds remain uncalibrated."
        ),
    )
    incidents_v3.add_argument("--generation-dir", type=Path, required=True)
    incidents_v3.add_argument("--output-dir", type=Path, required=True)
    incidents_v3.add_argument("--baseline-through", type=_date, required=True)
    incidents_v3.add_argument("--policy", type=Path, default=DEFAULT_INCIDENT_POLICY_V3_PATH)
    incidents_v3.add_argument("--threads", type=_positive_int, default=16)
    incidents_v3.add_argument("--memory-limit")
    incidents_v3.add_argument("--temp-dir", type=Path)
    incidents_v3.add_argument(
        "--finalizer-failure-capsule",
        type=Path,
        help=(
            "Opt-in directory written atomically only when the stage-9 story "
            "finalizer fails; successful builds create nothing there."
        ),
    )
    release_mode = incidents_v3.add_mutually_exclusive_group(required=True)
    release_mode.add_argument(
        "--previous-incident-dir",
        type=Path,
        help=(
            "Prior immutable Incident V3 release. Historical component, "
            "exposure, incident, and causal weekly content must remain stable."
        ),
    )
    release_mode.add_argument(
        "--first-release",
        action="store_true",
        help="Explicitly declare that no prior Incident V3 release exists.",
    )

    replay_incidents_v3 = subparsers.add_parser(
        "replay-incidents-v3-finalizer",
        help=(
            "Verify a stage-9 failure capsule and replay its captured story "
            "finalizer call once without publishing artifacts."
        ),
    )
    replay_incidents_v3.add_argument("--capsule-dir", type=Path, required=True)

    train_incident_archetypes_v3 = subparsers.add_parser(
        "train-incident-archetypes-v3",
        help=(
            "Discover immutable diagnostic archetype tags from completed Incident V3 "
            "stories; incident identity is never changed."
        ),
    )
    train_incident_archetypes_v3.add_argument("--incident-dir", type=Path, required=True)
    train_incident_archetypes_v3.add_argument("--model-dir", type=Path, required=True)
    train_incident_archetypes_v3.add_argument(
        "--training-through", type=_date, required=True
    )
    train_incident_archetypes_v3.add_argument(
        "--engine", choices=("cpu", "gpu"), default="cpu"
    )
    train_incident_archetypes_v3.add_argument(
        "--min-cluster-size", type=_positive_int, default=100
    )
    train_incident_archetypes_v3.add_argument(
        "--min-samples", type=_positive_int, default=20
    )
    train_incident_archetypes_v3.add_argument(
        "--radius-quantile", type=float, default=0.95
    )
    evidence_v4 = subparsers.add_parser(
        "build-evidence-v4",
        help=(
            "Build immutable daily pressure and acquisition-grain crop-evidence "
            "ledgers beside an existing source generation."
        ),
    )
    evidence_v4.add_argument("--generation-dir", type=Path, required=True)
    evidence_v4.add_argument("--evidence-dir", type=Path, required=True)
    evidence_v4.add_argument(
        "--released-at", required=True,
        help="Monotonic timezone-aware ingest/release watermark (normalized to UTC).",
    )
    evidence_v4.add_argument("--enriched-source-parquet", type=Path)
    evidence_v4.add_argument(
        "--acquisition-parquet",
        type=Path,
        help=(
            "Partial or complete attempt ledger merged with acquisitions derived "
            "from the enriched daily source."
        ),
    )
    evidence_v4.add_argument(
        "--availability-mode",
        choices=("strict", "reconstructed"),
        default="reconstructed",
    )
    evidence_v4.add_argument(
        "--policy", type=Path, default=DEFAULT_INCIDENT_POLICY_V4_PATH
    )
    evidence_v4.add_argument("--threads", type=_positive_int, default=16)
    evidence_v4.add_argument("--memory-limit")
    evidence_v4.add_argument("--temp-dir", type=Path)
    return parser


def _build(args: argparse.Namespace, as_of: date) -> dict[str, Any]:
    common = {
        "input_parquet": args.input_parquet,
        "output_dir": args.output_dir,
        "as_of_date": as_of,
        "policy": load_policy(args.policy),
        "history_from": args.history_from,
        "geometry_parquet": args.geometry_parquet,
    }
    if args.max_fields is None:
        result = build_partitioned_generation(
            **common,
            options=PartitionOptions(
                partitions=args.partitions,
                workers=args.workers,
                threads=args.threads,
                memory_limit=args.memory_limit,
                temp_dir=args.temp_dir,
            ),
        )
    else:
        result = build_generation(**common, max_fields=args.max_fields)
    return {
        "generation_id": result.generation_id,
        "generation_dir": str(result.generation_dir),
        "as_of_date": result.as_of_date.isoformat(),
        "row_count": result.row_count,
        "event_count": result.event_count,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "update":
        payload: dict[str, Any] = {
            "status": "written",
            "calibration_warning": "Starter thresholds are uncalibrated.",
            "generation": _build(args, args.as_of),
        }
    elif args.command == "replay":
        if args.max_fields is None:
            raise SystemExit(
                "Full replay would rescan the 39.7M-row source for every cutoff. Run one full "
                "`update` at the latest cutoff; its weekly snapshots are causal. Use replay only "
                "with --max-fields for acceptance testing."
            )
        if args.date_to < args.date_from:
            raise SystemExit("--date-to must be on or after --date-from.")
        generations: list[dict[str, Any]] = []
        cursor = args.date_from
        while cursor <= args.date_to:
            generations.append(_build(args, cursor))
            cursor += timedelta(days=args.step_days)
        payload = {
            "status": "written",
            "calibration_warning": "Starter thresholds are uncalibrated.",
            "generation_count": len(generations),
            "generations": generations,
        }
    elif args.command == "train-motifs":
        generation_manifest = json.loads(
            (args.generation_dir.expanduser().resolve() / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        policy_version = str(
            (generation_manifest.get("policy") or {}).get("version") or ""
        )
        policy_sha256 = str(
            (generation_manifest.get("policy") or {}).get("sha256") or ""
        )
        if not policy_version:
            raise SystemExit("Generation manifest is missing policy.version; refusing motif training.")
        if not policy_sha256:
            raise SystemExit("Generation manifest is missing policy.sha256; refusing motif training.")
        prefixes = load_training_prefixes(
            args.generation_dir,
            through=args.training_through.isoformat(),
            sample_age_buckets=True,
        )
        manifest = discover_motifs(
            prefixes,
            args.model_dir,
            config=DiscoveryConfig(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                radius_quantile=args.radius_quantile,
                assignment_margin=args.assignment_margin,
                engine=args.engine,
            ),
            training_cutoff=args.training_through.isoformat(),
            policy_version=policy_version,
            policy_sha256=policy_sha256,
        )
        payload = {
            "status": "written",
            "calibration_warning": "Discovered motifs are unreviewed and not outcome-validated.",
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "training_prefix_count": len(prefixes),
            "model": manifest,
        }
    elif args.command == "export-motifs":
        output = export_motif_generation(args.generation_dir, args.model_dir, args.output_dir)
        payload = {
            "status": "written",
            "output_dir": str(output),
            "warning": "Motif labels remain discovered_unreviewed until expert validation.",
        }
    elif args.command == "build-archetypes-v2":
        from story_monitor.archetype_workflow_v2 import build_archetype_model
        from story_monitor.archetypes_v2 import ArchetypeConfig

        payload = build_archetype_model(
            args.generation_dir,
            args.output_dir,
            training_cutoff=args.training_through.isoformat(),
            config=ArchetypeConfig(
                engine=args.engine,
                radius_quantile=args.radius_quantile,
                assignment_margin=args.assignment_margin,
            ),
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_dir=args.temp_dir,
        )
    elif args.command == "evaluate-archetypes-v2":
        from story_monitor.archetype_workflow_v2 import evaluate_archetype_release

        payload = evaluate_archetype_release(
            args.model_dir,
            args.output_dir,
            stability_runs=args.stability_runs,
        )
    elif args.command == "replay-incidents-v3-finalizer":
        from story_monitor.incident_workflow_v3 import (
            replay_finalizer_failure_capsule,
        )

        payload = replay_finalizer_failure_capsule(args.capsule_dir)
    elif args.command == "train-incident-archetypes-v3":
        from story_monitor.incident_archetype_discovery_v3 import (
            IncidentArchetypeDiscoveryConfig,
            train_incident_archetypes_v3,
        )

        model = train_incident_archetypes_v3(
            args.incident_dir,
            args.model_dir,
            training_through=args.training_through.isoformat(),
            config=IncidentArchetypeDiscoveryConfig(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                radius_quantile=args.radius_quantile,
                engine=args.engine,
            ),
        )
        payload = {
            "status": "written",
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "model": model,
            "warning": (
                "Incident archetypes are diagnostic_unreviewed tags and never replace "
                "incident_id."
            ),
        }
    elif args.command == "build-evidence-v4":
        from story_monitor.incident_context_v4 import build_incident_context_v4
        from story_monitor.incident_policy_v4 import load_incident_policy_v4

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)sZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        payload = build_incident_context_v4(
            args.generation_dir,
            args.evidence_dir,
            released_at=args.released_at,
            enriched_source_parquet=args.enriched_source_parquet,
            acquisition_parquet=args.acquisition_parquet,
            availability_mode=args.availability_mode,
            policy=load_incident_policy_v4(args.policy),
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_dir=args.temp_dir,
        )
    else:
        from story_monitor.incident_policy_v3 import load_incident_policy_v3
        from story_monitor.incident_workflow_v3 import build_incident_generation_v3

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)sZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        payload = build_incident_generation_v3(
            args.generation_dir,
            args.output_dir,
            baseline_through=args.baseline_through.isoformat(),
            policy=load_incident_policy_v3(args.policy),
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_dir=args.temp_dir,
            finalizer_failure_capsule=args.finalizer_failure_capsule,
            previous_incident_dir=args.previous_incident_dir,
            first_release=args.first_release,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
