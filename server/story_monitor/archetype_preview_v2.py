"""Build a separate, causal, diagnostic-only Archetype V2 viewer generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .archetype_preview_sql import (
    configured_connection,
    copy_auxiliary,
    create_sources,
    materialize_preview,
    validate_registry,
)
from .archetype_runner_validation import (
    RunnerError,
    validate_evaluation,
    validate_generation,
    validate_model,
)


LOGGER = logging.getLogger("archetype_preview_v2")
PREVIEW_SCHEMA_VERSION = "archetype-v2-diagnostic-preview/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_overlap(output: Path, protected: Path, label: str) -> None:
    if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError(f"Preview output must be separate from the immutable {label}: {protected}")


def _validate_gates(report: dict[str, Any], hard: bool) -> None:
    checks = ((report.get("gates") or {}).get("hard") or {}).get("checks") or {}
    if not hard or not checks or not all(value is True for value in checks.values()):
        failed = sorted(name for name, value in checks.items() if value is not True)
        suffix = f": {', '.join(failed)}" if failed else ""
        raise RunnerError(f"Archetype V2 hard gates did not all pass{suffix}")


def export_archetype_preview(
    generation_dir: Path,
    model_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    *,
    allow_failed_quality_gates: bool = False,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Materialize an immutable preview without promoting or mutating any source."""
    generation_dir = generation_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    evaluation_dir = evaluation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if threads < 1 or threads > 256:
        raise ValueError("threads must be between 1 and 256")
    for path, label in (
        (generation_dir, "generation"), (model_dir, "model"),
        (evaluation_dir, "evaluation"),
    ):
        _reject_overlap(output_dir, path, label)
        if temp_dir is not None:
            _reject_overlap(temp_dir.expanduser().resolve(), path, label)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable diagnostic preview already exists: {output_dir}")

    LOGGER.info("[1/5] validating generation, model, evaluation, and artifact hashes")
    generation_manifest_path = generation_dir / "manifest.json"
    model_manifest_path = model_dir / "archetype_manifest.json"
    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    generation = validate_generation(generation_dir)
    model = validate_model(model_dir, generation_manifest=generation_manifest_path)
    report, hard, quality = validate_evaluation(
        evaluation_dir, model_manifest=model_manifest_path
    )
    _validate_gates(report, hard)
    if not quality and not allow_failed_quality_gates:
        raise RunnerError(
            "Quality gates failed. Re-run only for an unpublishable diagnostic preview with "
            "--allow-failed-quality-gates."
        )
    if not (generation_dir / "event_state_snapshots.parquet").is_file():
        raise FileNotFoundError("Causal preview requires event_state_snapshots.parquet")

    run = generation.get("run") or {}
    as_of = str(run.get("as_of_date") or "")[:10]
    training_cutoff = str(model.get("training_cutoff") or "")[:10]
    if not as_of:
        raise ValueError("Source generation as-of date is missing")
    if not training_cutoff:
        raise ValueError("Archetype model training cutoff is missing")
    failed_quality = sorted(
        name for name, value in
        ((((report.get("gates") or {}).get("quality") or {}).get("checks") or {}).items())
        if value is not True
    )
    identity = hashlib.sha256(
        (_sha256(generation_manifest_path) + _sha256(model_manifest_path)
         + _sha256(evaluation_manifest_path)).encode("ascii")
    ).hexdigest()[:20]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".archetype-preview-v2-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        LOGGER.info("[2/5] opening streaming DuckDB transformation")
        with configured_connection(
            threads=threads, memory_limit=memory_limit, temp_dir=temp_dir
        ) as connection:
            create_sources(connection, generation_dir, model_dir, evaluation_dir)
            validate_registry(
                connection,
                str(model["model_version"]),
                str(model.get("feature_schema_sha256") or ""),
                training_cutoff,
            )
            LOGGER.info("[3/5] writing causal frames, trajectories, events, and labels")
            counts = materialize_preview(connection, stage, as_of=as_of)
        LOGGER.info("[4/5] copying geometry and writing diagnostic lineage")
        copy_auxiliary(generation_dir, stage)
        preview_manifest = _preview_manifest(
            generation, model, report, counts,
            preview_id=f"archetype_preview_{identity}",
            generation_sha=_sha256(generation_manifest_path),
            model_sha=_sha256(model_manifest_path),
            evaluation_sha=_sha256(evaluation_manifest_path),
            quality=quality,
            failed_quality=failed_quality,
        )
        (stage / "manifest.json").write_text(
            json.dumps(preview_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("[5/5] atomically installing immutable preview")
        os.replace(stage, output_dir)
    return {
        "status": "written",
        "phase": "phase_a_diagnostic_preview",
        "preview_dir": str(output_dir),
        "preview_id": f"archetype_preview_{identity}",
        "model_version": model["model_version"],
        "hard_gates_passed": True,
        "quality_gates_passed": quality,
        "publication_approved": False,
        "failed_quality_checks": failed_quality,
        "counts": counts,
    }


def _preview_manifest(
    generation: dict[str, Any],
    model: dict[str, Any],
    report: dict[str, Any],
    counts: dict[str, int],
    *,
    preview_id: str,
    generation_sha: str,
    model_sha: str,
    evaluation_sha: str,
    quality: bool,
    failed_quality: list[str],
) -> dict[str, Any]:
    source_run = generation.get("run") or {}
    warning = (
        "DIAGNOSTIC PREVIEW ONLY — quality gates failed; not approved for publication."
        if not quality else
        "DIAGNOSTIC PREVIEW ONLY — statistical gates are not agronomic validation."
    )
    return {
        "schema_version": PREVIEW_SCHEMA_VERSION,
        "run": {
            "status": "complete", "phase": "phase_a_diagnostic_preview",
            "generation_id": preview_id, "as_of_date": source_run.get("as_of_date"),
            "immutable": True, "diagnostic_preview": True,
            "publication_approved": False, "hard_gates_passed": True,
            "quality_gates_passed": quality, "failed_quality_checks": failed_quality,
            "viewer_ready": False, "viewer_bundle_required": True,
            "row_count": source_run.get("row_count"),
            "field_count": source_run.get("field_count"),
            "event_count": counts["event_windows"],
            "story_cluster_count": counts["story_cluster_count"],
            "motif_count": int(model.get("archetype_count") or 0),
            "archetype_count": int(model.get("archetype_count") or 0),
            "model_version": model.get("model_version"), "warning": warning,
        },
        "policy": generation.get("policy") or {},
        "semantics": {
            "prefix_safe": True,
            "story_cluster_id_alias": "diagnostic_archetype_display_id",
            "identity_reveal_rule": "holdout only: snapshot_as_of_date >= coalesce(anchor_date, generation_as_of_date)",
            "online_causal_scope": "holdout_only",
            "training_identity_rule": "training assignments remain masked as calibration_training for every historical frame",
            "pre_decision_state": "hazard_scoped_pending_anchor",
            "novel_state": "hazard_scoped_novel_unassigned",
            "event_id_retained_separately": True,
            "trajectory_interpretation": "field-footprint history; not geographic propagation",
        },
        "motifs": {
            "kind": "causal_event_archetype_v2", "model_version": model.get("model_version"),
            "catalog_status": "diagnostic_unreviewed", "quality_gates_passed": quality,
            "failed_quality_checks": failed_quality,
            "metrics": report.get("metrics") or {}, "warning": warning,
        },
        "parameters": {
            "map_color_by": "story_cluster", "map_top_scope": "global",
            "map_top_clusters": 12,
        },
        "map_geometry": {
            "mappable_event_field_count": counts["mapped_field_count"],
            "mappable_selected_field_count": counts["mapped_field_count"],
        },
        "lineage": {
            "source_generation_id": source_run.get("generation_id"),
            "source_generation_manifest_sha256": generation_sha,
            "model_manifest_sha256": model_sha,
            "evaluation_manifest_sha256": evaluation_sha,
        },
        "limitations": [
            warning,
            "Labels are diagnostic machine groupings, not agronomist-validated diagnoses.",
            "Pending, novel, and ineligible IDs are viewer states rather than learned archetypes.",
            "Training-cohort archetype identities stay masked as calibration_training.",
            "Map trails show changing field footprints and must not be presented as storm movement.",
        ],
        "outputs": {
            "map_frame_fields": "map_frame_fields.parquet",
            "event_story_cluster_labels": "event_story_cluster_labels.parquet",
            "event_windows": "event_windows.parquet",
            "story_day_membership": "story_day_membership.parquet",
            "event_state_snapshots": "event_state_snapshots.parquet",
            "map_field_geometry": "map_field_geometry.parquet",
        },
    }


__all__ = ["export_archetype_preview"]
