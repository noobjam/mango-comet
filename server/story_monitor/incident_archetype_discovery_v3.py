"""Immutable crop-by-hazard discovery for completed Incident V3 stories.

The tracker owns ``incident_id``.  This module only learns diagnostic,
unreviewed archetype tags from terminal stories whose full history is known by
an explicit training cutoff.  It never rewrites, merges, or creates incidents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .incident_archetypes_v3 import (
    CENSORED_TERMINAL_STATES,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    OUTCOME_OBSERVED_TERMINAL_STATES,
    TERMINAL_STATES,
    fit_robust_feature_schema,
    temporal_split_completed_stories,
    transform_finite_feature_matrix,
)


MODEL_SCHEMA_VERSION = "completed-incident-archetype-model-v3/1"
SOURCE_ARTIFACTS = (
    "completed_incident_features.parquet",
    "incident_weekly_state.parquet",
    "incident_membership.parquet",
)
PROHIBITED_MODEL_FEATURE_TOKENS = (
    "timeline",
    "evidence_week",
    "calendar_date",
    "latitude",
    "longitude",
    "center_lat",
    "center_lon",
    "district",
    "sector",
    "village",
    "admin",
    "location",
)


@dataclass(frozen=True)
class IncidentArchetypeDiscoveryConfig:
    """Versioned discovery defaults; small support values are test-only."""

    min_cluster_size: int = 100
    min_samples: int = 20
    radius_quantile: float = 0.95
    engine: str = "cpu"

    def validate(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0.5 <= self.radius_quantile < 1:
            raise ValueError("radius_quantile must be in [0.5, 1)")
        if self.engine not in {"cpu", "gpu"}:
            raise ValueError("engine must be explicitly cpu or gpu")


def train_incident_archetypes_v3(
    incident_dir: Path,
    model_dir: Path,
    *,
    training_through: str | date,
    config: IncidentArchetypeDiscoveryConfig = IncidentArchetypeDiscoveryConfig(),
) -> dict[str, Any]:
    """Discover immutable unreviewed tags from pre-cutoff terminal stories."""
    config.validate()
    incident_dir = incident_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    cutoff = _cutoff(training_through)
    _validate_output_location(incident_dir, model_dir)
    source_manifest, source_provenance, source_paths = _load_source_provenance(incident_dir)
    _validate_cutoff_bounds(source_manifest, cutoff)
    policy = _policy_provenance(source_manifest)

    completed = pd.read_parquet(incident_dir / SOURCE_ARTIFACTS[0])
    weekly = pd.read_parquet(incident_dir / SOURCE_ARTIFACTS[1])
    completed, weekly = _prepare_source_frames(completed, weekly)
    excluded_censored = completed[
        completed["final_state"].isin(CENSORED_TERMINAL_STATES)
    ].copy()
    eligible = completed[
        completed["final_state"].isin(OUTCOME_OBSERVED_TERMINAL_STATES)
    ].copy()
    if len(eligible) + len(excluded_censored) != len(completed):
        raise AssertionError("Terminal-story model eligibility is not exhaustive")
    split = temporal_split_completed_stories(eligible, cutoff)
    training = split[split["temporal_split"] == "train"].copy()
    if training.empty:
        raise ValueError(
            "No outcome-observed Incident V3 stories end on or before "
            "training_through; censored, season-boundary, and merged fragments "
            "are intentionally ineligible for discovery."
        )
    _validate_training_history(training, weekly)
    training = training.sort_values(
        ["crop_name", "hazard_family", "incident_id"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_model_features(training)

    schema = fit_robust_feature_schema(
        training, strata_columns=("crop_name", "hazard_family")
    )
    _validate_schema_feature_boundary(schema)
    matrix = transform_finite_feature_matrix(training, schema)
    backend = _resolve_engine(config.engine)
    model_scope = _model_scope(schema, config, cutoff, policy)
    discovery = _discover_stratified(
        training,
        matrix,
        config=config,
        backend=backend,
        model_scope=model_scope,
    )
    prototypes = discovery["prototypes"]
    catalog = discovery["catalog"]
    assignments = discovery["assignments"]
    stratum_stats = discovery["strata"]
    if prototypes.empty:
        raise ValueError(
            "HDBSCAN discovered no supported archetypes across crop_name x hazard_family "
            "strata; inspect terminal cohort size and noise before changing thresholds."
        )

    model_version = _model_version(schema, config, cutoff, policy, prototypes)
    schema.update(
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_version": model_version,
            "training_through": cutoff,
            "radius_quantile": float(config.radius_quantile),
            "stratification": "strict_crop_name_x_hazard_family",
            "excluded_feature_families": [
                "incident_or_exposure_identity",
                "calendar_date",
                "location_or_administration",
            ],
        }
    )
    feature_schema_sha256 = _canonical_sha256(schema)
    for frame in (prototypes, catalog, assignments):
        frame["model_version"] = model_version
    assignments["feature_schema_sha256"] = feature_schema_sha256
    if assignments["incident_id"].tolist() != training["incident_id"].tolist():
        raise AssertionError("archetype discovery rewrote or reordered incident identity")

    # A discovered cluster is not a reviewed archetype.  Causal prefix
    # primitives remain available in incident_archetypes_v3, but publishing a
    # live prefix model requires a separate immutable expert-review overlay.
    # This also prevents legacy non-causal lineage totals from becoming prefix
    # features in models trained from older immutable V3 generations.
    prefix_manifest = {
        "status": "blocked_pending_review",
        "reason": "an immutable reviewed completed-story assignment overlay is required",
        "review_overlay_required": True,
        "artifacts_emitted": False,
        "causal_primitives": [
            "build_causal_prefix_features",
            "fit_prefix_prototypes",
        ],
        "lineage_leakage_guard": "no_unreviewed_prefix_model_emitted",
    }
    _verify_source_unchanged(source_paths, source_provenance)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-archetype-v3-", dir=model_dir.parent) as temporary:
        stage = Path(temporary) / model_dir.name
        stage.mkdir()
        _write_json(stage / "feature_schema.json", schema)
        _write_parquet(stage / "completed_assignments.parquet", assignments)
        _write_parquet(stage / "archetype_catalog.parquet", catalog)
        _write_parquet(stage / "prototypes.parquet", prototypes)
        artifact_names = [
            "feature_schema.json",
            "completed_assignments.parquet",
            "archetype_catalog.parquet",
            "prototypes.parquet",
        ]
        manifest = _build_manifest(
            source_manifest=source_manifest,
            source_provenance=source_provenance,
            stage=stage,
            artifact_names=artifact_names,
            cutoff=cutoff,
            config=config,
            backend=backend,
            model_version=model_version,
            feature_schema_sha256=feature_schema_sha256,
            split=split,
            assignments=assignments,
            catalog=catalog,
            stratum_stats=stratum_stats,
            prefix_manifest=prefix_manifest,
            policy=policy,
            source_terminal=completed,
            excluded_censored=excluded_censored,
        )
        _write_json(stage / "model_manifest.json", manifest)
        _verify_source_unchanged(source_paths, source_provenance)
        os.replace(stage, model_dir)
    return manifest


def _prepare_source_frames(
    completed: pd.DataFrame,
    weekly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = (
        "feature_schema_version",
        "incident_id",
        "exposure_id",
        "crop_name",
        "hazard_family",
        "first_evidence_week",
        "last_evidence_week",
        "final_state",
    )
    _require_columns(completed, (*metadata, *MODEL_FEATURE_COLUMNS), "completed features")
    if completed.empty:
        raise ValueError("completed incident features contain zero terminal stories")
    completed = completed.copy()
    for name in ("incident_id", "exposure_id"):
        _require_nonblank(completed, name, "completed features")
        completed[name] = completed[name].astype(str)
    if completed["incident_id"].duplicated().any():
        raise ValueError("completed incident features require exactly one row per incident_id")
    if set(completed["feature_schema_version"].astype(str)) != {FEATURE_SCHEMA_VERSION}:
        raise ValueError("completed incident features use an unsupported feature schema")
    for name in ("crop_name", "hazard_family"):
        _require_nonblank(completed, name, "completed features")
        completed[name] = _canonical_dimension(completed[name])
    completed["final_state"] = completed["final_state"].astype(str).str.upper()
    if not set(completed["final_state"]).issubset(TERMINAL_STATES):
        raise ValueError("completed incident features contain a non-terminal story")
    completed["first_evidence_week"] = _dates(
        completed["first_evidence_week"], "first_evidence_week"
    )
    completed["last_evidence_week"] = _dates(
        completed["last_evidence_week"], "last_evidence_week"
    )
    completed["stratification_key"] = (
        completed["crop_name"] + "::" + completed["hazard_family"]
    )

    weekly = weekly.copy()
    state_column = next(
        (name for name in ("current_state", "incident_state", "story_state") if name in weekly),
        None,
    )
    weekly_required = {
        "incident_id",
        "exposure_id",
        "timeline_bucket",
        "crop_name",
        "hazard_family",
    }
    _require_columns(weekly, weekly_required, "incident weekly state")
    if state_column is None:
        raise ValueError("incident weekly state is missing current_state")
    weekly["current_state"] = weekly[state_column].astype(str).str.upper()
    for name in ("incident_id", "exposure_id"):
        _require_nonblank(weekly, name, "incident weekly state")
        weekly[name] = weekly[name].astype(str)
    for name in ("crop_name", "hazard_family"):
        _require_nonblank(weekly, name, "incident weekly state")
        weekly[name] = _canonical_dimension(weekly[name])
    weekly["timeline_bucket"] = _dates(weekly["timeline_bucket"], "timeline_bucket")
    if weekly.duplicated(["incident_id", "timeline_bucket"]).any():
        raise ValueError("incident weekly state must be unique by incident and week")
    return completed, weekly


def _validate_training_history(training: pd.DataFrame, weekly: pd.DataFrame) -> None:
    latest = (
        weekly.sort_values(["incident_id", "timeline_bucket"], kind="mergesort")
        .groupby("incident_id", sort=False)
        .tail(1)
        .set_index("incident_id")
    )
    for row in training.itertuples(index=False):
        if row.incident_id not in latest.index:
            raise ValueError(f"training incident {row.incident_id} has no weekly history")
        final = latest.loc[row.incident_id]
        if isinstance(final, pd.DataFrame):
            raise ValueError(f"training incident {row.incident_id} has ambiguous final history")
        if final["timeline_bucket"] != row.last_evidence_week:
            raise ValueError(f"training incident {row.incident_id} end date does not reconcile")
        if str(final["current_state"]).upper() != str(row.final_state).upper():
            raise ValueError(f"training incident {row.incident_id} terminal state does not reconcile")
        for name in ("exposure_id", "crop_name", "hazard_family"):
            if str(final[name]) != str(getattr(row, name)):
                raise ValueError(f"training incident {row.incident_id} {name} does not reconcile")


def _validate_model_features(training: pd.DataFrame) -> None:
    values = training.loc[:, MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("training model features must all be finite")


def _validate_schema_feature_boundary(schema: dict[str, Any]) -> None:
    names = tuple(schema.get("feature_names") or ())
    if names != MODEL_FEATURE_COLUMNS:
        raise ValueError("V3 discovery schema does not match the approved model features")
    forbidden = [
        name
        for name in names
        if any(token in name.lower() for token in PROHIBITED_MODEL_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError("V3 model features include forbidden location/date/admin fields")


def _discover_stratified(
    training: pd.DataFrame,
    matrix: np.ndarray,
    *,
    config: IncidentArchetypeDiscoveryConfig,
    backend: str,
    model_scope: str,
) -> dict[str, Any]:
    archetype_ids = np.full(len(training), None, dtype=object)
    discovery_labels = np.full(len(training), -1, dtype=np.int64)
    membership = np.zeros(len(training), dtype=float)
    statuses = np.full(len(training), "unsupported_stratum", dtype=object)
    reasons = np.full(len(training), "stratum_below_min_cluster_size", dtype=object)
    prototypes: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    next_label = 0
    grouped = training.groupby(["crop_name", "hazard_family"], sort=True).groups
    for (crop, hazard), raw_indexes in grouped.items():
        positions = np.asarray(list(raw_indexes), dtype=np.int64)
        before = len(prototypes)
        if len(positions) >= config.min_cluster_size:
            local_labels, local_membership = _fit_hdbscan(
                matrix[positions], config, backend
            )
            _validate_hdbscan_output(local_labels, local_membership, len(positions))
            statuses[positions] = "noise"
            reasons[positions] = "hdbscan_noise"
            for local_label in sorted(int(value) for value in np.unique(local_labels) if value >= 0):
                local_mask = local_labels == local_label
                member_positions = positions[local_mask]
                if len(member_positions) < config.min_cluster_size:
                    reasons[member_positions] = "cluster_below_min_cluster_size"
                    continue
                vectors = matrix[member_positions]
                prototype = _observed_robust_prototype(vectors)
                distances = np.linalg.norm(vectors - prototype, axis=1)
                radius = max(
                    float(np.quantile(distances, config.radius_quantile)), 1e-6
                )
                archetype_id = _archetype_id(
                    str(crop), str(hazard), prototype, model_scope
                )
                label = _diagnostic_label(str(crop), str(hazard), archetype_id)
                archetype_ids[member_positions] = archetype_id
                discovery_labels[member_positions] = next_label
                membership[member_positions] = local_membership[local_mask]
                statuses[member_positions] = "diagnostic_unreviewed"
                reasons[member_positions] = "hdbscan_discovered_cluster"
                record: dict[str, Any] = {
                    "archetype_id": archetype_id,
                    "crop_name": str(crop),
                    "hazard_family": str(hazard),
                    "member_count": int(len(member_positions)),
                    "radius": radius,
                    "radius_quantile": float(config.radius_quantile),
                    "prototype_method": "observed_nearest_robust_center_v3",
                    "status": "diagnostic_unreviewed",
                    "label": label,
                }
                record.update(
                    {
                        f"f_{index:03d}": float(value)
                        for index, value in enumerate(prototype)
                    }
                )
                prototypes.append(record)
                catalog.append(
                    {
                        "archetype_id": archetype_id,
                        "crop_name": str(crop),
                        "hazard_family": str(hazard),
                        "member_count": int(len(member_positions)),
                        "training_membership_mean": float(
                            np.mean(local_membership[local_mask])
                        ),
                        "status": "diagnostic_unreviewed",
                        "publish_status": "not_reviewed",
                        "label": label,
                    }
                )
                next_label += 1
        stats.append(
            {
                "crop_name": str(crop),
                "hazard_family": str(hazard),
                "training_story_count": int(len(positions)),
                "archetype_count": int(len(prototypes) - before),
                "noise_or_unsupported_count": int(
                    np.sum(discovery_labels[positions] < 0)
                ),
                "min_cluster_size": int(config.min_cluster_size),
            }
        )
    prototypes_frame = pd.DataFrame(prototypes).sort_values(
        ["crop_name", "hazard_family", "archetype_id"], kind="mergesort"
    ).reset_index(drop=True) if prototypes else pd.DataFrame()
    catalog_frame = pd.DataFrame(catalog).sort_values(
        ["crop_name", "hazard_family", "archetype_id"], kind="mergesort"
    ).reset_index(drop=True) if catalog else pd.DataFrame()
    if not prototypes_frame.empty and prototypes_frame["archetype_id"].duplicated().any():
        raise ValueError("deterministic archetype ID collision")
    assignments = training[
        [
            "incident_id",
            "exposure_id",
            "crop_name",
            "hazard_family",
            "first_evidence_week",
            "last_evidence_week",
            "final_state",
        ]
    ].copy()
    assignments["archetype_id"] = archetype_ids
    assignments["assignment_status"] = statuses
    assignments["accepted"] = discovery_labels >= 0
    assignments["discovery_label"] = discovery_labels
    assignments["training_membership"] = membership
    assignments["assignment_reason"] = reasons
    assignments["assignment_method"] = "crop_hazard_hdbscan_discovery_v3"
    return {
        "prototypes": prototypes_frame,
        "catalog": catalog_frame,
        "assignments": assignments,
        "strata": stats,
    }


def _fit_hdbscan(
    matrix: np.ndarray,
    config: IncidentArchetypeDiscoveryConfig,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    min_samples = min(config.min_samples, len(matrix))
    if backend == "gpu":
        cupy, hdbscan = _gpu_dependencies()
        model = hdbscan(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
        ).fit(cupy.asarray(matrix, dtype=cupy.float32))
        return (
            cupy.asnumpy(model.labels_).astype(int),
            cupy.asnumpy(model.probabilities_).astype(float),
        )
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "CPU incident archetype discovery requires scikit-learn>=1.3"
        ) from exc
    model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        n_jobs=-1 if len(matrix) >= 10_000 else 1,
        copy=True,
    ).fit(matrix)
    return model.labels_.astype(int), model.probabilities_.astype(float)


def _resolve_engine(engine: str) -> str:
    if engine == "cpu":
        return "cpu"
    _gpu_dependencies()
    return "gpu"


def _gpu_dependencies() -> tuple[Any, Any]:
    try:
        import cupy
        from cuml.cluster.hdbscan import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "GPU incident archetype discovery requires RAPIDS cuML and CuPy"
        ) from exc
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except RuntimeError as exc:
        raise RuntimeError("GPU incident archetype discovery cannot access CUDA") from exc
    if device_count < 1:
        raise RuntimeError("GPU incident archetype discovery found no visible CUDA device")
    return cupy, HDBSCAN


def _validate_hdbscan_output(
    labels: np.ndarray,
    membership: np.ndarray,
    expected_count: int,
) -> None:
    labels = np.asarray(labels)
    membership = np.asarray(membership, dtype=float)
    if labels.shape != (expected_count,) or membership.shape != (expected_count,):
        raise ValueError("HDBSCAN returned an unexpected assignment shape")
    if not np.isfinite(membership).all() or (membership < 0).any() or (membership > 1).any():
        raise ValueError("HDBSCAN returned invalid membership probabilities")
    if not np.isfinite(labels.astype(float)).all():
        raise ValueError("HDBSCAN returned invalid labels")
    if not np.equal(labels, labels.astype(np.int64)).all():
        raise ValueError("HDBSCAN returned non-integer labels")


def _load_source_provenance(
    incident_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Path]]:
    paths = {
        "manifest.json": incident_dir / "manifest.json",
        **{name: incident_dir / name for name in SOURCE_ARTIFACTS},
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incident V3 source is missing: " + ", ".join(missing))
    provenance = {name: _fingerprint(path) for name, path in paths.items()}
    try:
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Incident V3 source manifest is not valid JSON") from exc
    run = manifest.get("run") or {}
    if str(run.get("status") or "") != "complete" or not bool(run.get("immutable")):
        raise ValueError("Archetype discovery requires a complete immutable Incident V3 source")
    declared = manifest.get("artifacts") or {}
    for name in SOURCE_ARTIFACTS:
        expected = str((declared.get(name) or {}).get("sha256") or "")
        if not expected or expected != provenance[name]["sha256"]:
            raise ValueError(f"Incident V3 source hash does not reconcile for {name}")
    semantics = manifest.get("semantics") or {}
    if not bool(semantics.get("archetype_is_optional_not_identity")):
        raise ValueError("Incident V3 manifest does not preserve story/archetype separation")
    return manifest, provenance, paths


def _policy_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    policy = manifest.get("policy") or {}
    version = str(policy.get("version") or "")
    source_sha256 = str(policy.get("sha256") or "")
    effective_sha256 = str(policy.get("effective_sha256") or source_sha256)
    if not version or not source_sha256 or not effective_sha256:
        raise ValueError("Incident V3 manifest is missing policy hashes")
    return {
        "version": version,
        "sha256": source_sha256,
        "effective_sha256": effective_sha256,
        "calibration_status": str(policy.get("calibration_status") or "unknown"),
    }


def _validate_cutoff_bounds(manifest: dict[str, Any], cutoff: str) -> None:
    run = manifest.get("run") or {}
    baseline_raw = run.get("baseline_through")
    as_of_raw = run.get("source_as_of_date") or run.get("as_of_date")
    if baseline_raw is None or as_of_raw is None:
        raise ValueError(
            "Incident V3 manifest must declare baseline_through and source_as_of_date"
        )
    baseline = pd.Timestamp(baseline_raw).normalize()
    as_of = pd.Timestamp(as_of_raw).normalize()
    boundary = pd.Timestamp(cutoff).normalize()
    if pd.isna(baseline) or pd.isna(as_of) or baseline >= as_of:
        raise ValueError("Incident V3 manifest has invalid source date bounds")
    if boundary <= baseline:
        raise ValueError("training_through must be after the incident baseline")
    if boundary > as_of:
        raise ValueError("training_through must be on or before the source as-of date")


def _build_manifest(
    *,
    source_manifest: dict[str, Any],
    source_provenance: dict[str, dict[str, Any]],
    stage: Path,
    artifact_names: list[str],
    cutoff: str,
    config: IncidentArchetypeDiscoveryConfig,
    backend: str,
    model_version: str,
    feature_schema_sha256: str,
    split: pd.DataFrame,
    assignments: pd.DataFrame,
    catalog: pd.DataFrame,
    stratum_stats: list[dict[str, Any]],
    prefix_manifest: dict[str, Any],
    policy: dict[str, str],
    source_terminal: pd.DataFrame,
    excluded_censored: pd.DataFrame,
) -> dict[str, Any]:
    source_run = source_manifest.get("run") or {}
    split_counts = split["temporal_split"].value_counts()
    return {
        "status": "complete",
        "phase": "diagnostic_unreviewed",
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": model_version,
        "training_through": cutoff,
        "engine_requested": config.engine,
        "engine_used": backend,
        "config": asdict(config),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": feature_schema_sha256,
        "source_terminal_story_count": int(len(source_terminal)),
        "eligible_outcome_observed_story_count": int(len(split)),
        "ineligible_censored_story_count": int(len(excluded_censored)),
        "ineligible_censored_by_state": {
            str(state): int(count)
            for state, count in excluded_censored["final_state"]
            .value_counts().sort_index().items()
        },
        "training_story_count": int((split["temporal_split"] == "train").sum()),
        "excluded_after_cutoff_count": int((split["temporal_split"] != "train").sum()),
        "holdout_story_count": int(split_counts.get("holdout", 0)),
        "embargo_story_count": int(split_counts.get("embargo", 0)),
        "archetype_count": int(len(catalog)),
        "noise_or_unsupported_count": int((~assignments["accepted"]).sum()),
        "strata": stratum_stats,
        "prefix_prototypes": prefix_manifest,
        "source": {
            "incident_generation_id": source_run.get("generation_id"),
            "manifest_sha256": source_provenance["manifest.json"]["sha256"],
            "artifacts": {
                name: source_provenance[name] for name in SOURCE_ARTIFACTS
            },
        },
        "policy": policy,
        "implementation": {
            "sha256": _fingerprint(Path(__file__).resolve())["sha256"],
            "software": _software_versions(backend),
        },
        "semantics": {
            "primary_story_identity": "incident_id",
            "archetype_is_optional_not_identity": True,
            "incident_identity_preserved": True,
            "stratification": "strict_crop_name_x_hazard_family",
            "noise_remains_unassigned": True,
            "location_date_admin_features_used": False,
            "prefix_tags_are_tentative": True,
            "feature_weighting": "equal_l2_energy_per_semantic_family",
            "crop_stage_squared_distance_budget": 1.0,
            "censored_or_merged_fragments_used_for_discovery": False,
            "eligible_terminal_states": sorted(
                OUTCOME_OBSERVED_TERMINAL_STATES
            ),
        },
        "evaluation": {
            "status": "not_run",
            "release_status": "blocked_pending_evaluation_and_review",
            "required_before_publication": [
                "temporal_holdout_novelty_and_acceptance",
                "deterministic_refit_stability",
                "same_stratum_prototype_separation",
                "expert_agronomic_review_overlay",
            ],
        },
        "publication_status": "diagnostic_unreviewed_not_map_approved",
        "warning": (
            "Statistical discovery is unreviewed and does not establish agronomic "
            "validity, crop death, causation, or map-publication approval."
        ),
        "artifacts": {name: _fingerprint(stage / name) for name in artifact_names},
    }


def _model_scope(
    schema: dict[str, Any],
    config: IncidentArchetypeDiscoveryConfig,
    cutoff: str,
    policy: dict[str, str],
) -> str:
    return _canonical_sha256(
        {
            "schema": schema,
            "config": asdict(config),
            "training_through": cutoff,
            "policy_version": policy["version"],
            "policy_effective_sha256": policy["effective_sha256"],
        }
    )[:16]


def _model_version(
    schema: dict[str, Any],
    config: IncidentArchetypeDiscoveryConfig,
    cutoff: str,
    policy: dict[str, str],
    prototypes: pd.DataFrame,
) -> str:
    records = prototypes.to_dict(orient="records")
    digest = _canonical_sha256(
        {
            "schema": schema,
            "config": asdict(config),
            "training_through": cutoff,
            "policy_version": policy["version"],
            "policy_effective_sha256": policy["effective_sha256"],
            "prototypes": records,
        }
    )
    return "incident-archetype-v3-" + digest[:16]


def _archetype_id(
    crop: str,
    hazard: str,
    prototype: np.ndarray,
    model_scope: str,
) -> str:
    payload = [
        crop,
        hazard,
        model_scope,
        [round(float(value), 8) for value in prototype],
    ]
    return "incident-archetype:" + _canonical_sha256(payload)[:20]


def _observed_robust_prototype(vectors: np.ndarray) -> np.ndarray:
    center = np.median(vectors, axis=0)
    return vectors[int(np.argmin(np.linalg.norm(vectors - center, axis=1)))].copy()


def _diagnostic_label(crop: str, hazard: str, archetype_id: str) -> str:
    crop_label = crop.replace("_", " ").title()
    hazard_label = hazard.replace("_", " ").title()
    return f"Diagnostic unreviewed {crop_label} × {hazard_label} {archetype_id[-8:]}"


def _validate_output_location(source: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Immutable incident archetype model already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Archetype model output must not modify the immutable Incident V3 source")
    if source.is_relative_to(output):
        raise ValueError("Archetype model output must not contain its Incident V3 source")


def _verify_source_unchanged(
    paths: dict[str, Path], expected: dict[str, dict[str, Any]]
) -> None:
    changed = [name for name, path in paths.items() if _fingerprint(path) != expected[name]]
    if changed:
        raise RuntimeError("Incident V3 source changed during training: " + ", ".join(changed))


def _fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path.name}")
    return {"sha256": digest.hexdigest(), "size_bytes": int(after.st_size)}


def _software_versions(backend: str) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if backend == "cpu":
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
    else:
        cupy, _ = _gpu_dependencies()
        import cuml

        versions["cupy"] = cupy.__version__
        versions["cuml"] = cuml.__version__
    return versions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.reset_index(drop=True).to_parquet(
        path, index=False, compression="zstd"
    )


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _canonical_dimension(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower()


def _cutoff(value: str | date) -> str:
    try:
        parsed = pd.Timestamp(value).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("training_through must be an ISO date") from exc
    if pd.isna(parsed):
        raise ValueError("training_through must be an ISO date")
    return parsed.date().isoformat()


def _dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    return parsed


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _require_nonblank(frame: pd.DataFrame, name: str, label: str) -> None:
    if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label}.{name} contains null or blank values")


__all__ = [
    "IncidentArchetypeDiscoveryConfig",
    "MODEL_SCHEMA_VERSION",
    "SOURCE_ARTIFACTS",
    "train_incident_archetypes_v3",
]
