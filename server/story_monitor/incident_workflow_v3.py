"""Atomic end-to-end build for crop-impact incident stories V3."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from .incident_archetypes_v3 import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    extract_completed_incident_features,
)
from .incident_cells_v3 import (
    build_component_field_rows,
    build_stage_baseline,
    build_weekly_exposure_cells,
)
from .incident_context_v3 import build_incident_context_v3
from .incident_denominators_v3 import (
    SUMMARY_COLUMNS,
    build_incident_stage_summary,
    enrich_incident_weekly_state,
)
from .incident_exposures_v3 import track_exposures
from .incident_lineage_v3 import (
    build_incident_lineage_v3,
    remap_incident_lineage_segments,
)
from .incident_policy_v3 import IncidentPolicyV3, load_incident_policy_v3
from .incident_story_states_v3 import (
    build_crop_story_scaffold,
    build_incident_followup_evidence,
    finalize_crop_story_artifacts,
)
from .incident_tracking_v3 import build_weekly_components
from .incident_validation_v3 import (
    REQUIRED_SOURCE_ARTIFACTS,
    artifact_hashes,
    file_sha256,
    validate_append_stability,
    validate_final_artifact_directory,
    validate_source_generation,
)


SCHEMA_VERSION = "crop-impact-incident-generation-v3/1"
LOGGER = logging.getLogger("incident_workflow_v3")
V3_IMPLEMENTATION_INPUTS = (
    "incident_archetypes_v3.py",
    "incident_cells_v3.py",
    "incident_context_v3.py",
    "incident_denominators_v3.py",
    "incident_exposures_v3.py",
    "incident_lineage_v3.py",
    "incident_policy_v3.py",
    "incident_story_states_v3.py",
    "incident_tracking_v3.py",
    "incident_validation_v3.py",
    "incident_workflow_v3.py",
)


def build_incident_generation_v3(
    generation_dir: Path,
    output_dir: Path,
    *,
    baseline_through: str,
    policy: IncidentPolicyV3 | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
    previous_incident_dir: Path | None = None,
    first_release: bool = False,
) -> dict[str, Any]:
    """Build all V3 tracking artifacts and publish them in one atomic rename."""
    generation_dir = generation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    previous_incident_dir = (
        previous_incident_dir.expanduser().resolve()
        if previous_incident_dir is not None
        else None
    )
    if bool(first_release) == (previous_incident_dir is not None):
        raise ValueError(
            "Choose exactly one release mode: first_release=True or "
            "previous_incident_dir=<prior immutable release>"
        )
    policy = policy or load_incident_policy_v3()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable Incident V3 output already exists: {output_dir}")
    if output_dir == generation_dir or output_dir.is_relative_to(generation_dir):
        raise ValueError("Incident V3 output must not be inside its immutable source generation")
    if generation_dir.is_relative_to(output_dir):
        raise ValueError("Incident V3 output must not contain its source generation")
    if previous_incident_dir is not None:
        if not previous_incident_dir.is_dir():
            raise FileNotFoundError(
                f"Previous Incident V3 directory does not exist: {previous_incident_dir}"
            )
        if output_dir == previous_incident_dir:
            raise ValueError("Incident V3 output must differ from the previous release")
        if output_dir.is_relative_to(previous_incident_dir):
            raise ValueError("Incident V3 output must not be inside the previous release")
        if previous_incident_dir.is_relative_to(output_dir):
            raise ValueError("Incident V3 output must not contain the previous release")
    provenance = _capture_build_provenance(generation_dir, policy)
    source_manifest = validate_source_generation(generation_dir)
    _verify_source_manifest_provenance(provenance)
    baseline_date = pd.Timestamp(baseline_through).normalize()
    generation_as_of = pd.Timestamp((source_manifest.get("run") or {}).get("as_of_date")).normalize()
    if pd.isna(baseline_date) or baseline_date >= generation_as_of:
        raise ValueError("baseline_through must precede the generation as-of date")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-v3-", dir=output_dir.parent) as temporary:
        transaction = Path(temporary)
        context_dir = transaction / "context"
        stage = transaction / output_dir.name
        stage.mkdir()
        LOGGER.info("[1/10] building causal crop/week context")
        build_incident_context_v3(
            generation_dir, context_dir, policy=policy, threads=threads,
            memory_limit=memory_limit, temp_dir=temp_dir,
        )
        context_path = context_dir / "field_week_context.parquet"
        lanes_path = context_dir / "event_week_lanes.parquet"
        LOGGER.info("[2/10] fitting frozen stage-aware baseline through %s", baseline_through)
        baseline = build_stage_baseline(
            context_path, lanes_path, baseline_through=baseline_date.date().isoformat(),
            policy=policy, threads=threads, memory_limit=memory_limit, temp_dir=temp_dir,
        )
        LOGGER.info("[3/10] aggregating monitored cell coverage and significance")
        cells = build_weekly_exposure_cells(
            context_path, lanes_path, baseline, policy=policy,
            assignment_after=baseline_date.date().isoformat(),
            assignment_through=generation_as_of.date().isoformat(),
            threads=threads,
            memory_limit=memory_limit, temp_dir=temp_dir,
        )
        LOGGER.info("[4/10] loading canonical component field lanes")
        field_rows = build_component_field_rows(
            context_path, lanes_path, cells, policy=policy, threads=threads,
            frontier_distance_cells=policy.frontier_distance_cells,
            memory_limit=memory_limit, temp_dir=temp_dir,
        )
        LOGGER.info("[5/10] building deterministic weekly components")
        component_result = build_weekly_components(cells, field_rows, policy)
        LOGGER.info(
            "weekly components=%s memberships=%s",
            len(component_result.components), len(component_result.memberships),
        )
        LOGGER.info("[6/10] linking components into persistent exposures")
        exposure_result = track_exposures(
            component_result.components, component_result.memberships, policy
        )
        LOGGER.info("[7/10] building crop-specific exact-cell story scaffolds")
        story_scaffold = build_crop_story_scaffold(
            exposure_result.weekly_state,
            exposure_result.assignments,
            component_result.memberships,
            cells,
            policy,
        )
        LOGGER.info(
            "crop story scaffolds=%s weekly_rows=%s memberships=%s",
            len(story_scaffold.catalog),
            len(story_scaffold.weekly_state),
            len(story_scaffold.memberships),
        )
        lineage_result = build_incident_lineage_v3(
            exposure_result.lineage,
            story_scaffold.catalog,
            component_result.memberships,
        )
        reference_latitude = _single_reference_latitude(cells, policy)
        LOGGER.info("[8/10] computing initial crop/stage coverage")
        stage_summary = (
            build_incident_stage_summary(
                context_path,
                story_scaffold.weekly_state,
                story_scaffold.memberships,
                policy=policy,
                reference_latitude=reference_latitude,
                threads=threads,
                memory_limit=memory_limit,
                temp_dir=temp_dir,
            )
            if not story_scaffold.weekly_state.empty
            else _empty_stage_summary()
        )
        followup = (
            build_incident_followup_evidence(
                lanes_path,
                story_scaffold.weekly_state,
                story_scaffold.memberships,
                threads=threads,
                memory_limit=memory_limit,
                temp_dir=temp_dir,
            )
            if not story_scaffold.weekly_state.empty
            else pd.DataFrame()
        )
        LOGGER.info("[9/10] solving lifecycle and exact crop coverage to a fixed point")
        previous_signature: str | None = None
        story_result = None
        for iteration in range(1, 9):
            story_result = finalize_crop_story_artifacts(
                story_scaffold,
                stage_summary,
                policy,
                followup_evidence=followup,
                incident_lineage=lineage_result.lineage,
                weekly_cells=cells,
            )
            next_stage_summary = (
                build_incident_stage_summary(
                    context_path,
                    story_result.weekly_state,
                    story_result.memberships,
                    policy=policy,
                    reference_latitude=reference_latitude,
                    threads=threads,
                    memory_limit=memory_limit,
                    temp_dir=temp_dir,
                )
                if not story_result.weekly_state.empty
                else _empty_stage_summary()
            )
            signature = _story_coverage_fixed_point_signature(next_stage_summary)
            LOGGER.info(
                "lifecycle/coverage iteration=%s signature=%s",
                iteration,
                signature[:12],
            )
            stage_summary = next_stage_summary
            if signature == previous_signature:
                break
            previous_signature = signature
        else:
            raise RuntimeError(
                "Incident lifecycle and crop coverage did not converge after 8 iterations"
            )
        if story_result is None:
            raise RuntimeError("Incident lifecycle fixed-point solver did not run")
        segment_lineage_result = remap_incident_lineage_segments(
            lineage_result.lineage,
            story_result.weekly_state,
            story_result.catalog,
        )
        incident_weekly = enrich_incident_weekly_state(
            story_result.weekly_state, stage_summary
        )
        LOGGER.info("[10/10] extracting features and validating immutable artifacts")
        completed_features = (
            extract_completed_incident_features(
                incident_weekly, story_result.memberships
            )
            if not incident_weekly.empty
            else _empty_completed_features()
        )
        incident_catalog = story_result.catalog.merge(
            segment_lineage_result.incident_metadata.drop(
                columns=["exposure_id", "crop_name_normalized"], errors="ignore"
            ),
            on="incident_id", how="left", validate="one_to_one",
        )
        shutil.copy2(context_path, stage / "field_week_context.parquet")
        shutil.copy2(lanes_path, stage / "event_week_lanes.parquet")
        shutil.copy2(context_dir / "manifest.json", stage / "context_manifest.json")
        frames = {
            "stage_baseline.parquet": baseline,
            "weekly_exposure_cells.parquet": cells,
            "weekly_components.parquet": component_result.components,
            "component_membership.parquet": component_result.memberships,
            "exposure_component_assignments.parquet": exposure_result.assignments,
            "exposure_links.parquet": exposure_result.lineage,
            "exposure_weekly_state.parquet": exposure_result.weekly_state,
            "incident_catalog.parquet": incident_catalog,
            "incident_weekly_state.parquet": incident_weekly,
            "incident_stage_summary.parquet": stage_summary,
            "incident_membership.parquet": story_result.memberships,
            "incident_windows.parquet": story_result.windows,
            "incident_lineage.parquet": segment_lineage_result.lineage,
            "completed_incident_features.parquet": completed_features,
        }
        for name, frame in frames.items():
            frame.reset_index(drop=True).to_parquet(
                stage / name, index=False, compression="zstd"
            )
        validation = validate_final_artifact_directory(stage)
        validation["append_stability"] = (
            validate_append_stability(previous_incident_dir, stage)
            if previous_incident_dir is not None
            else {"status": "first_release", "first_release": True}
        )
        LOGGER.info("V3 artifact validation passed: %s", validation["row_counts"])
        manifest = _build_manifest(
            source_manifest=source_manifest,
            stage=stage,
            baseline_through=baseline_date.date().isoformat(),
            policy=policy,
            validation=validation,
            provenance=provenance,
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify_build_provenance(provenance)
        os.replace(stage, output_dir)
    return {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "generation_id": manifest["run"]["generation_id"],
        "output_dir": str(output_dir),
        "row_counts": validation["row_counts"],
        "append_stability": validation["append_stability"],
        "warning": policy.warning,
        "publication_status": "diagnostic_unreviewed_not_map_approved",
    }


def _single_reference_latitude(cells: pd.DataFrame, policy: IncidentPolicyV3) -> float:
    if cells.empty:
        return float(policy.grid_origin_lat)
    values = pd.to_numeric(cells.get("reference_latitude"), errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError("Weekly cells must contain exactly one metric-grid reference latitude")
    return float(values[0])


def _story_coverage_fixed_point_signature(
    stage_summary: pd.DataFrame,
) -> str:
    """Hash the compact deterministic input to the next lifecycle replay.

    Finalization is pure for a frozen scaffold/follow-up/lineage tuple, so two
    consecutive identical stage summaries imply identical lifecycle and
    membership outputs.  Avoid serializing the multi-million-row membership
    table into a giant transient Python string on every VM iteration.
    """

    columns = sorted(str(column) for column in stage_summary.columns)
    ordered = stage_summary.reindex(columns=columns).copy()
    sort_keys = [
        key
        for key in ("timeline_bucket", "incident_id", "stage_bucket")
        if key in ordered
    ]
    if sort_keys and not ordered.empty:
        ordered = ordered.sort_values(sort_keys, kind="mergesort")
    digest = hashlib.sha256(
        json.dumps(columns, separators=(",", ":")).encode("utf-8")
    )
    row_hashes = pd.util.hash_pandas_object(
        ordered.reset_index(drop=True), index=False, categorize=True
    )
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _empty_completed_features() -> pd.DataFrame:
    metadata = [
        "feature_schema_version", "incident_id", "exposure_id", "crop_name",
        "hazard_family", "stratification_key", "first_evidence_week",
        "last_evidence_week", "final_state",
    ]
    return pd.DataFrame(columns=[*metadata, *MODEL_FEATURE_COLUMNS])


def _empty_stage_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def _build_manifest(
    *,
    source_manifest: dict[str, Any],
    stage: Path,
    baseline_through: str,
    policy: IncidentPolicyV3,
    validation: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    artifact_names = sorted(
        path.name for path in stage.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    source_inputs = provenance["source_inputs"]
    implementation_inputs = provenance["implementation_inputs"]
    policy_input = provenance["policy_input"]
    source_provenance_sha256 = _provenance_sha256(source_inputs)
    implementation_sha256 = _provenance_sha256(implementation_inputs)
    policy_provenance_sha256 = _canonical_sha256(policy_input)
    effective_policy_sha256 = str(provenance["effective_policy_sha256"])
    identity = _generation_identity(
        baseline_through=baseline_through,
        source_provenance_sha256=source_provenance_sha256,
        implementation_sha256=implementation_sha256,
        policy_provenance_sha256=policy_provenance_sha256,
        effective_policy_sha256=effective_policy_sha256,
    )
    source_run = source_manifest.get("run") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "generation_id": f"incident_v3_{identity}",
            "immutable": True,
            "source_generation_id": source_run.get("generation_id"),
            "source_as_of_date": source_run.get("as_of_date"),
            "baseline_through": baseline_through,
            "tracking_window": "complete weekly buckets strictly after baseline_through",
            "publication_status": "diagnostic_unreviewed_not_map_approved",
        },
        "source": {
            "generation_manifest_sha256": source_inputs["manifest.json"]["sha256"],
            "provenance_sha256": source_provenance_sha256,
            "inputs": source_inputs,
        },
        "implementation": {
            "sha256": implementation_sha256,
            "inputs": implementation_inputs,
        },
        "implementation_sha256": implementation_sha256,
        "policy": {
            "version": policy.version,
            "sha256": policy.source_sha256,
            "provenance_sha256": policy_provenance_sha256,
            "effective_sha256": effective_policy_sha256,
            "input": policy_input,
            "schema_version": policy.schema_version,
            "calibration_status": policy.calibration_status,
            "warning": policy.warning,
        },
        "validation": validation,
        "semantics": {
            "primary_story_identity": "crop_impact_incident_id",
            "identity_hierarchy": ["component_id", "exposure_id", "incident_id"],
            "stage_is_dynamic_context": True,
            "archetype_is_optional_not_identity": True,
            "watch_cannot_establish_component": True,
            "low_coverage_freezes_lifecycle_clocks": True,
            "physical_movement_inferred": False,
            "crop_death_inferred": False,
            "denominator": "monitored_and_evaluable_crop_instances_by_story_footprint_and_stage",
        },
        "limitations": [
            "Starter spatial, significance, linking, and lifecycle thresholds are uncalibrated.",
            "Crop instances and stage aliases are derived monitoring constructs, not planting records.",
            "No survey, yield, crop-death, or causal propagation ground truth is available.",
            "Completed-story features are diagnostic until temporal and expert-review gates pass.",
        ],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "outputs": {name.removesuffix(".parquet"): name for name in artifact_names},
        "artifacts": artifact_hashes(stage, artifact_names),
    }


def _capture_build_provenance(
    generation_dir: Path, policy: IncidentPolicyV3
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    source_paths = {
        name: generation_dir / name
        for name in ("manifest.json", *REQUIRED_SOURCE_ARTIFACTS)
    }
    implementation_paths = {
        f"story_monitor/{name}": root / name for name in V3_IMPLEMENTATION_INPUTS
    }
    policy_path = policy.source_path.expanduser().resolve()
    policy_input = _file_fingerprint(policy_path)
    if policy_input["sha256"] != policy.source_sha256:
        raise RuntimeError("Incident V3 policy source changed after it was loaded")
    return {
        "source_paths": source_paths,
        "source_inputs": _capture_file_fingerprints(source_paths),
        "implementation_paths": implementation_paths,
        "implementation_inputs": _capture_file_fingerprints(implementation_paths),
        "policy_path": policy_path,
        "policy_input": policy_input,
        "effective_policy_sha256": _effective_policy_sha256(policy),
    }


def _capture_file_fingerprints(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _file_fingerprint(paths[name]) for name in sorted(paths)}


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Incident V3 provenance input: {resolved}")
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Incident V3 provenance input changed while hashing: {resolved.name}")
    return {"size_bytes": int(after.st_size), "sha256": digest}


def _verify_build_provenance(provenance: dict[str, Any]) -> None:
    checks = (
        ("source generation", provenance["source_paths"], provenance["source_inputs"]),
        (
            "V3 implementation",
            provenance["implementation_paths"],
            provenance["implementation_inputs"],
        ),
    )
    for label, paths, expected in checks:
        current = _capture_file_fingerprints(paths)
        changed = sorted(name for name in expected if current.get(name) != expected[name])
        if changed:
            raise RuntimeError(
                f"{label} inputs changed during Incident V3 build: {', '.join(changed)}"
            )
    current_policy = _file_fingerprint(provenance["policy_path"])
    if current_policy != provenance["policy_input"]:
        raise RuntimeError("Incident V3 policy input changed during Incident V3 build")


def _verify_source_manifest_provenance(provenance: dict[str, Any]) -> None:
    current = _file_fingerprint(provenance["source_paths"]["manifest.json"])
    if current != provenance["source_inputs"]["manifest.json"]:
        raise RuntimeError("Source generation manifest changed while it was being validated")


def _effective_policy_sha256(policy: IncidentPolicyV3) -> str:
    payload = asdict(policy)
    payload.pop("source_path", None)
    return _canonical_sha256(payload)


def _provenance_sha256(inputs: dict[str, dict[str, Any]]) -> str:
    return _canonical_sha256(inputs)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generation_identity(
    *,
    baseline_through: str,
    source_provenance_sha256: str,
    implementation_sha256: str,
    policy_provenance_sha256: str,
    effective_policy_sha256: str,
) -> str:
    digest = _canonical_sha256(
        {
            "baseline_through": baseline_through,
            "effective_policy_sha256": effective_policy_sha256,
            "implementation_sha256": implementation_sha256,
            "policy_provenance_sha256": policy_provenance_sha256,
            "schema_version": SCHEMA_VERSION,
            "source_provenance_sha256": source_provenance_sha256,
        }
    )
    return digest[:20]


__all__ = ["SCHEMA_VERSION", "build_incident_generation_v3"]
