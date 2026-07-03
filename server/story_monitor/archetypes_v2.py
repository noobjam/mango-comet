"""Hazard-stratified discovery and frozen assignment for event archetypes V2.

This module intentionally operates on exactly one causal anchor row per event.
It does not read weekly prefixes and it never uses lifecycle or terminal outcome
as a model feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .anchor_features_v2 import FEATURE_SCHEMA, FEATURE_VERSION, MODEL_FEATURE_COLUMNS


HAZARD_COLUMN = "hazard_family"
NOVEL_ARCHETYPE_ID = "novel_unassigned"
MODEL_SCHEMA_VERSION = "event_archetype_model_v2"


@dataclass(frozen=True)
class ArchetypeConfig:
    """Versioned production defaults; lower support values are test-only."""

    min_cluster_floor: int = 100
    min_cluster_fraction: float = 0.005
    min_samples: int = 20
    minimum_field_support: int = 50
    radius_quantile: float = 0.95
    assignment_margin: float = 0.05
    assignment_batch_size: int = 50_000
    engine: str = "cpu"

    def validate(self) -> None:
        if self.min_cluster_floor < 2:
            raise ValueError("min_cluster_floor must be at least 2")
        if not 0 <= self.min_cluster_fraction <= 1:
            raise ValueError("min_cluster_fraction must be in [0, 1]")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if self.minimum_field_support < 1:
            raise ValueError("minimum_field_support must be positive")
        if not 0.5 <= self.radius_quantile < 1:
            raise ValueError("radius_quantile must be in [0.5, 1)")
        if self.assignment_margin < 0:
            raise ValueError("assignment_margin must be non-negative")
        if self.assignment_batch_size < 1:
            raise ValueError("assignment_batch_size must be positive")
        if self.engine not in {"cpu", "gpu", "auto"}:
            raise ValueError("engine must be cpu, gpu, or auto")


def discover_archetypes(
    rows: pd.DataFrame,
    model_dir: Path,
    *,
    config: ArchetypeConfig = ArchetypeConfig(),
    training_cutoff: str,
    policy_version: str,
    policy_sha256: str,
) -> dict[str, Any]:
    """Fit independent HDBSCAN models per hazard and persist frozen prototypes."""
    config.validate()
    _require_columns(rows, ("event_id", "field_id", HAZARD_COLUMN, *MODEL_FEATURE_COLUMNS))
    if rows.empty:
        raise ValueError("archetype discovery requires at least one eligible event anchor")
    if rows["event_id"].astype(str).duplicated().any():
        raise ValueError("archetype discovery requires exactly one row per event_id")
    if not policy_version or not policy_sha256:
        raise ValueError("policy version and SHA-256 are required")

    ordered = rows.copy()
    ordered["event_id"] = ordered["event_id"].astype(str)
    ordered["field_id"] = ordered["field_id"].astype(str)
    ordered[HAZARD_COLUMN] = _hazards(ordered)
    ordered = ordered.sort_values([HAZARD_COLUMN, "event_id"], kind="mergesort").reset_index(drop=True)
    _validate_feature_values(ordered)
    schema = fit_feature_schema(ordered)
    matrix = transform_features(ordered, schema)
    if not np.isfinite(matrix).all():
        raise ValueError("transformed training features contain non-finite values")

    backend = _resolve_engine(config.engine)
    model_scope = _model_scope(schema, config, training_cutoff, policy_version, policy_sha256)
    assigned_ids = np.full(len(ordered), NOVEL_ARCHETYPE_ID, dtype=object)
    discovery_labels = np.full(len(ordered), -1, dtype=np.int64)
    membership = np.zeros(len(ordered), dtype=float)
    prototypes: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    hazard_stats: list[dict[str, Any]] = []

    next_label = 0
    for hazard, raw_index in ordered.groupby(HAZARD_COLUMN, sort=True).groups.items():
        positions = np.asarray(list(raw_index), dtype=np.int64)
        minimum = adaptive_min_cluster_size(len(positions), config)
        before = len(prototypes)
        if len(positions) >= minimum:
            local_labels, local_membership = _fit_hdbscan(
                matrix[positions], config, backend, min_cluster_size=minimum
            )
            if (
                not np.isfinite(local_membership).all()
                or (local_membership < 0).any()
                or (local_membership > 1).any()
            ):
                raise ValueError("HDBSCAN returned invalid membership probabilities")
            candidate_labels = sorted(int(value) for value in np.unique(local_labels) if value >= 0)
            for local_label in candidate_labels:
                local_mask = local_labels == local_label
                member_positions = positions[local_mask]
                field_support = int(ordered.iloc[member_positions]["field_id"].nunique())
                if len(member_positions) < minimum or field_support < config.minimum_field_support:
                    continue
                vectors = matrix[member_positions]
                prototype = _observed_robust_prototype(vectors)
                distances = np.linalg.norm(vectors - prototype, axis=1)
                radius = max(float(np.quantile(distances, config.radius_quantile)), 1e-9)
                archetype_id = _archetype_id(str(hazard), prototype, model_scope)
                assigned_ids[member_positions] = archetype_id
                discovery_labels[member_positions] = next_label
                membership[member_positions] = local_membership[local_mask]
                record: dict[str, Any] = {
                    "archetype_id": archetype_id,
                    HAZARD_COLUMN: str(hazard),
                    "member_count": int(len(member_positions)),
                    "field_count": field_support,
                    "radius": radius,
                    "radius_quantile": float(config.radius_quantile),
                    "prototype_method": "observed_nearest_robust_center_v2",
                }
                record.update(
                    {f"f_{index:03d}": float(value) for index, value in enumerate(prototype)}
                )
                prototypes.append(record)
                catalog.append(
                    {
                        "archetype_id": archetype_id,
                        HAZARD_COLUMN: str(hazard),
                        "member_count": int(len(member_positions)),
                        "field_count": field_support,
                        "training_membership_mean": float(np.mean(local_membership[local_mask])),
                        "status": "diagnostic_unreviewed",
                        "publish_status": "not_reviewed",
                        "label": _diagnostic_label(str(hazard), archetype_id),
                    }
                )
                next_label += 1
        hazard_stats.append(
            {
                HAZARD_COLUMN: str(hazard),
                "training_event_count": int(len(positions)),
                "min_cluster_size": int(minimum),
                "archetype_count": int(len(prototypes) - before),
            }
        )

    if not prototypes:
        raise ValueError(
            "HDBSCAN discovered no supported archetypes; inspect the anchor cohort and noise "
            "diagnostics before changing the versioned thresholds."
        )
    if not np.isfinite(pd.DataFrame(prototypes).filter(regex=r"^f_|^radius$").to_numpy()).all():
        raise ValueError("prototype artifacts contain non-finite values")

    model_version = _model_version(
        schema, config, training_cutoff, policy_version, policy_sha256, prototypes
    )
    schema.update(
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_version": model_version,
            "policy_version": policy_version,
            "policy_sha256": policy_sha256,
            "assignment_margin": float(config.assignment_margin),
            "assignment_batch_size": int(config.assignment_batch_size),
            "novel_archetype_id": NOVEL_ARCHETYPE_ID,
        }
    )
    for row in prototypes:
        row["model_version"] = model_version
    for row in catalog:
        row["model_version"] = model_version

    model_dir = model_dir.expanduser().resolve()
    if model_dir.exists():
        raise FileExistsError(f"Immutable archetype model directory already exists: {model_dir}")
    model_dir.mkdir(parents=True)
    _write_json(model_dir / "feature_schema.json", schema)
    pd.DataFrame(prototypes).to_parquet(model_dir / "prototypes.parquet", index=False)
    pd.DataFrame(catalog).to_parquet(model_dir / "archetype_catalog.parquet", index=False)
    training = ordered[["event_id", "field_id", HAZARD_COLUMN]].copy()
    training["archetype_id"] = assigned_ids
    training["accepted"] = discovery_labels >= 0
    training["discovery_label"] = discovery_labels
    training["training_membership"] = membership
    training["assignment_reason"] = np.where(
        training["accepted"], "discovered_cluster", "discovery_noise_or_support_filter"
    )
    training["assignment_method"] = "hdbscan_discovery_v2"
    training["model_version"] = model_version
    training["feature_schema_sha256"] = _json_sha256(schema)
    training.to_parquet(model_dir / "training_assignments.parquet", index=False)

    manifest: dict[str, Any] = {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model_version": model_version,
        "feature_version": FEATURE_VERSION,
        "feature_schema_sha256": _json_sha256(schema),
        "training_cutoff": training_cutoff,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "engine_requested": config.engine,
        "engine_used": backend,
        "training_event_count": int(len(ordered)),
        "archetype_count": int(len(prototypes)),
        "noise_count": int(np.sum(discovery_labels < 0)),
        "config": asdict(config),
        "hazards": hazard_stats,
        "warning": "Diagnostic unreviewed archetypes; not approved for map publication.",
    }
    manifest["artifacts"] = {
        name: _file_sha256(model_dir / name)
        for name in (
            "feature_schema.json",
            "prototypes.parquet",
            "archetype_catalog.parquet",
            "training_assignments.parquet",
        )
    }
    _write_json(model_dir / "archetype_manifest.json", manifest)
    return manifest


def fit_feature_schema(rows: pd.DataFrame) -> dict[str, Any]:
    """Fit train-only robust statistics independently inside each hazard."""
    schema: dict[str, Any] = {
        "version": FEATURE_VERSION,
        "source_feature_schema": FEATURE_SCHEMA,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "hazards": {},
    }
    source_specs = {item["name"]: item for item in FEATURE_SCHEMA["features"]}
    group_sizes: dict[str, int] = {}
    for item in FEATURE_SCHEMA["features"]:
        group_sizes[item["group"]] = group_sizes.get(item["group"], 0) + 1
    for hazard, frame in rows.groupby(HAZARD_COLUMN, sort=True):
        specs: list[dict[str, Any]] = []
        for name in MODEL_FEATURE_COLUMNS:
            source = source_specs[name]
            values = _numeric(frame, name)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            if source["kind"] == "continuous":
                q25, q75 = (
                    np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
                )
                scale = float(q75 - q25)
                if not np.isfinite(scale) or scale < 1e-9:
                    scale = 1.0
            else:
                scale = 1.0
            specs.append(
                {
                    "name": name,
                    "group": source["group"],
                    "kind": source["kind"],
                    "median": median,
                    "scale": scale,
                    "group_divisor": math.sqrt(group_sizes[source["group"]]),
                }
            )
        schema["hazards"][str(hazard)] = {"features": specs}
    return schema


def transform_features(rows: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    """Apply frozen hazard-local transforms without learning from assignment rows."""
    output = np.empty((len(rows), len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
    hazards = _hazards(rows)
    for hazard in sorted(set(hazards.tolist())):
        positions = np.flatnonzero(hazards.to_numpy() == hazard)
        hazard_schema = schema.get("hazards", {}).get(str(hazard))
        if hazard_schema is None:
            output[positions, :] = 0.0
            continue
        specs = hazard_schema["features"]
        for column_index, spec in enumerate(specs):
            values = _numeric(rows.iloc[positions], spec["name"])
            finite = np.isfinite(values)
            kind = str(spec["kind"])
            if kind == "continuous":
                values = np.where(finite, values, float(spec["median"]))
                values = np.clip(
                    (values - float(spec["median"])) / float(spec["scale"]), -5.0, 5.0
                )
            else:
                values = np.where(finite, values, 0.0)
                values = np.clip(values, 0.0, 1.0)
            output[positions, column_index] = (
                values / float(spec["group_divisor"])
            ).astype(np.float32)
    return output


def assign_frozen_archetypes(rows: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Assign one anchor per event in bounded vectorized batches with open-set rejection."""
    _require_columns(rows, ("event_id", HAZARD_COLUMN, *MODEL_FEATURE_COLUMNS))
    if rows["event_id"].astype(str).duplicated().any():
        raise ValueError("frozen assignment requires exactly one row per event_id")
    _validate_feature_values(rows)
    model_dir = model_dir.expanduser().resolve()
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    prototypes = pd.read_parquet(model_dir / "prototypes.parquet")
    manifest = json.loads(
        (model_dir / "archetype_manifest.json").read_text(encoding="utf-8")
    )
    _validate_assignment_model(model_dir, schema, prototypes, manifest)
    matrix = transform_features(rows, schema)
    feature_columns = [f"f_{index:03d}" for index in range(matrix.shape[1])]
    batch_size = int(schema.get("assignment_batch_size", 50_000))
    required_margin = float(schema.get("assignment_margin", 0.05))
    model_version = str(schema["model_version"])
    hazards = _hazards(rows)

    result = pd.DataFrame(
        {
            "event_id": rows["event_id"].astype(str).to_numpy(),
            HAZARD_COLUMN: hazards.to_numpy(),
            "archetype_id": NOVEL_ARCHETYPE_ID,
            "accepted": False,
            "assignment_reason": "no_hazard_prototype",
            "candidate_archetype_id": None,
            "runner_up_archetype_id": None,
            "assignment_distance": np.nan,
            "candidate_radius": np.nan,
            "distance_ratio": np.nan,
            "assignment_margin": np.nan,
            "model_version": model_version,
            "feature_schema_sha256": _json_sha256(schema),
            "assignment_method": "frozen_prototype_radius_v2",
        }
    )
    if "field_id" in rows:
        result.insert(1, "field_id", rows["field_id"].astype(str).to_numpy())

    for hazard in sorted(set(hazards.tolist())):
        positions = np.flatnonzero(hazards.to_numpy() == hazard)
        candidates = prototypes[prototypes[HAZARD_COLUMN].astype(str) == str(hazard)].reset_index(drop=True)
        if candidates.empty or str(hazard) not in schema.get("hazards", {}):
            continue
        candidate_vectors = candidates[feature_columns].to_numpy(dtype=np.float32)
        candidate_norm = np.sum(candidate_vectors * candidate_vectors, axis=1)
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            vectors = matrix[batch_positions]
            squared = np.maximum(
                np.sum(vectors * vectors, axis=1)[:, None]
                + candidate_norm[None, :]
                - 2.0 * vectors @ candidate_vectors.T,
                0.0,
            )
            best_index = np.argmin(squared, axis=1)
            best_distance = np.sqrt(squared[np.arange(len(vectors)), best_index])
            if len(candidates) > 1:
                second_index = np.argpartition(squared, 1, axis=1)[:, 1]
                second_distance = np.sqrt(squared[np.arange(len(vectors)), second_index])
                margin = second_distance - best_distance
                runner = candidates.iloc[second_index]["archetype_id"].astype(str).to_numpy()
            else:
                margin = np.full(len(vectors), np.inf)
                runner = np.full(len(vectors), None, dtype=object)
            radius = candidates.iloc[best_index]["radius"].to_numpy(dtype=float)
            ratio = best_distance / radius
            accepted = (best_distance <= radius) & (margin >= required_margin)
            candidate_id = candidates.iloc[best_index]["archetype_id"].astype(str).to_numpy()
            result.loc[batch_positions, "candidate_archetype_id"] = candidate_id
            result.loc[batch_positions, "runner_up_archetype_id"] = runner
            result.loc[batch_positions, "assignment_distance"] = best_distance
            result.loc[batch_positions, "candidate_radius"] = radius
            result.loc[batch_positions, "distance_ratio"] = ratio
            result.loc[batch_positions, "assignment_margin"] = np.where(
                np.isfinite(margin), margin, np.nan
            )
            result.loc[batch_positions, "accepted"] = accepted
            result.loc[batch_positions, "archetype_id"] = np.where(
                accepted, candidate_id, NOVEL_ARCHETYPE_ID
            )
            result.loc[batch_positions, "assignment_reason"] = np.where(
                accepted, "within_radius_and_margin", "outside_radius_or_ambiguous"
            )
    return result


def evaluate_archetype_model(
    training_rows: pd.DataFrame,
    holdout_rows: pd.DataFrame,
    model_dir: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist engineering/holdout diagnostics. Stability is a separate required gate."""
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == model_dir or output_dir.is_relative_to(model_dir):
        raise ValueError("evaluation output must not be inside the frozen model directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((model_dir / "archetype_manifest.json").read_text(encoding="utf-8"))
    training_assignments = pd.read_parquet(model_dir / "training_assignments.parquet")
    catalog = pd.read_parquet(model_dir / "archetype_catalog.parquet")
    prototypes = pd.read_parquet(model_dir / "prototypes.parquet")
    training_frozen_assignments = assign_frozen_archetypes(training_rows, model_dir)
    holdout_assignments = assign_frozen_archetypes(holdout_rows, model_dir)
    training_frozen_assignments.to_parquet(
        output_dir / "training_frozen_assignments.parquet", index=False
    )
    holdout_assignments.to_parquet(output_dir / "holdout_assignments.parquet", index=False)

    training_ids = set(training_rows["event_id"].astype(str))
    holdout_ids = set(holdout_rows["event_id"].astype(str))
    if training_ids & holdout_ids:
        raise ValueError("training and holdout event IDs overlap")
    noise_count = int((~training_assignments["accepted"].astype(bool)).sum())
    train_count = int(len(training_assignments))
    holdout_count = int(len(holdout_assignments))
    accepted_count = int(holdout_assignments["accepted"].astype(bool).sum())
    accepted_ratios = pd.to_numeric(
        holdout_assignments.loc[holdout_assignments["accepted"], "distance_ratio"], errors="coerce"
    ).dropna()
    p90_ratio = float(accepted_ratios.quantile(0.9)) if len(accepted_ratios) else None
    by_hazard: list[dict[str, Any]] = []
    hazards = sorted(
        set(training_rows[HAZARD_COLUMN].astype(str)) | set(holdout_rows[HAZARD_COLUMN].astype(str))
    )
    prototype_hazards = set(prototypes[HAZARD_COLUMN].astype(str))
    for hazard in hazards:
        train_n = int((training_rows[HAZARD_COLUMN].astype(str) == hazard).sum())
        subset = holdout_assignments[holdout_assignments[HAZARD_COLUMN].astype(str) == hazard]
        accepted_n = int(subset["accepted"].astype(bool).sum())
        by_hazard.append(
            {
                HAZARD_COLUMN: hazard,
                "training_event_count": train_n,
                "holdout_event_count": int(len(subset)),
                "holdout_accepted_count": accepted_n,
                "holdout_novel_count": int(len(subset) - accepted_n),
                "holdout_novelty_rate": (
                    float((len(subset) - accepted_n) / len(subset)) if len(subset) else None
                ),
                "has_prototype": hazard in prototype_hazards,
            }
        )
    by_hazard_frame = pd.DataFrame(by_hazard)
    by_hazard_frame.to_parquet(output_dir / "evaluation_by_hazard.parquet", index=False)
    overlap = prototype_overlap(prototypes)
    overlap.to_parquet(output_dir / "prototype_overlap.parquet", index=False)

    feature_columns = [name for name in prototypes if name.startswith("f_")]
    finite_prototypes = bool(
        np.isfinite(prototypes[[*feature_columns, "radius"]].to_numpy(dtype=float)).all()
        and (pd.to_numeric(prototypes["radius"], errors="coerce") > 0).all()
    )
    supported_hazard_rows = by_hazard_frame[
        (by_hazard_frame["training_event_count"] >= 1000)
        & (by_hazard_frame["holdout_event_count"] > 0)
    ]
    supported_novelty_ok = bool(
        supported_hazard_rows.empty
        or (pd.to_numeric(supported_hazard_rows["holdout_novelty_rate"]) <= 0.50).all()
    )
    prototype_coverage_ok = bool(
        by_hazard_frame[by_hazard_frame["training_event_count"] >= 1000]["has_prototype"].all()
    )
    prototype_ids = set(prototypes["archetype_id"].astype(str))
    catalog_ids = set(catalog["archetype_id"].astype(str))
    accepted_training_ids = set(
        training_assignments.loc[
            training_assignments["accepted"].astype(bool), "archetype_id"
        ].astype(str)
    )
    rejected_training_ids = set(
        training_assignments.loc[
            ~training_assignments["accepted"].astype(bool), "archetype_id"
        ].astype(str)
    )
    manifest_counts_reconcile = bool(
        int(manifest.get("training_event_count", -1)) == train_count
        and int(manifest.get("noise_count", -1)) == noise_count
        and int(manifest.get("archetype_count", -1)) == len(prototypes) == len(catalog)
    )
    catalog_counts_reconcile = bool(
        int(pd.to_numeric(catalog["member_count"], errors="coerce").sum())
        == int(training_assignments["accepted"].astype(bool).sum())
        and int(pd.to_numeric(prototypes["member_count"], errors="coerce").sum())
        == int(training_assignments["accepted"].astype(bool).sum())
    )
    nonoverlap_fraction = (
        float((overlap["overlap_ratio"] < 1).mean()) if not overlap.empty else 1.0
    )
    overlap_pair_count = int(len(overlap))
    nonoverlapping_pair_count = int(
        (pd.to_numeric(overlap["overlap_ratio"], errors="coerce") < 1).sum()
    )
    overlapping_pair_count = overlap_pair_count - nonoverlapping_pair_count
    hard_checks = {
        "unique_training_events": bool(training_assignments["event_id"].is_unique),
        "unique_training_frozen_events": bool(
            training_frozen_assignments["event_id"].is_unique
        ),
        "unique_holdout_events": bool(holdout_assignments["event_id"].is_unique),
        "training_assignment_cohort_matches": set(
            training_assignments["event_id"].astype(str)
        ) == training_ids,
        "training_frozen_assignment_cohort_matches": set(
            training_frozen_assignments["event_id"].astype(str)
        ) == training_ids,
        "holdout_assignment_cohort_matches": set(
            holdout_assignments["event_id"].astype(str)
        ) == holdout_ids,
        "train_holdout_disjoint": not bool(training_ids & holdout_ids),
        "one_archetype_per_event": bool(
            training_frozen_assignments["event_id"].is_unique
            and holdout_assignments["event_id"].is_unique
        ),
        "finite_prototypes_and_radii": finite_prototypes,
        "minimum_candidate_support": bool(
            (catalog["member_count"] >= 100).all() and (catalog["field_count"] >= 50).all()
        ),
        "no_duplicate_published_labels": _published_labels_unique(catalog),
        "supported_hazards_have_prototypes": prototype_coverage_ok,
        "manifest_counts_reconcile": manifest_counts_reconcile,
        "catalog_prototype_ids_match": bool(catalog_ids == prototype_ids),
        "catalog_member_counts_reconcile": catalog_counts_reconcile,
        "training_assignment_ids_valid": bool(
            accepted_training_ids <= prototype_ids
            and rejected_training_ids <= {NOVEL_ARCHETYPE_ID}
        ),
        "model_version_consistent": bool(
            set(training_assignments["model_version"].astype(str)) == {manifest["model_version"]}
            and set(training_frozen_assignments["model_version"].astype(str))
            <= {manifest["model_version"]}
            and set(holdout_assignments["model_version"].astype(str)) <= {manifest["model_version"]}
        ),
        "feature_schema_hash_consistent": bool(
            set(training_assignments["feature_schema_sha256"].astype(str))
            == {manifest["feature_schema_sha256"]}
            and set(training_frozen_assignments["feature_schema_sha256"].astype(str))
            <= {manifest["feature_schema_sha256"]}
            and set(holdout_assignments["feature_schema_sha256"].astype(str))
            <= {manifest["feature_schema_sha256"]}
        ),
    }
    discovery_noise_rate = float(noise_count / train_count) if train_count else None
    accepted_rate = float(accepted_count / holdout_count) if holdout_count else None
    novelty_rate = float(1 - accepted_rate) if accepted_rate is not None else None
    quality_checks = {
        "discovery_noise_rate": discovery_noise_rate is not None and discovery_noise_rate <= 0.60,
        "holdout_accepted_rate": accepted_rate is not None and accepted_rate >= 0.65,
        "holdout_novelty_rate": novelty_rate is not None and novelty_rate <= 0.35,
        "supported_hazard_novelty": supported_novelty_ok,
        "accepted_distance_ratio_p90": p90_ratio is not None and p90_ratio <= 0.90,
        "prototype_nonoverlap_fraction": nonoverlap_fraction >= 0.90,
        "subsample_stability": False,
    }
    report = {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_version": manifest["model_version"],
        "metrics": {
            "training_event_count": train_count,
            "discovery_noise_count": noise_count,
            "discovery_noise_rate": discovery_noise_rate,
            "holdout_event_count": holdout_count,
            "holdout_accepted_count": accepted_count,
            "holdout_accepted_rate": accepted_rate,
            "holdout_novelty_rate": novelty_rate,
            "accepted_distance_ratio_p90": p90_ratio,
            "prototype_nonoverlap_fraction": nonoverlap_fraction,
            "prototype_pair_count": overlap_pair_count,
            "prototype_nonoverlapping_pair_count": nonoverlapping_pair_count,
            "prototype_overlapping_pair_count": overlapping_pair_count,
            "stability_status": "not_run",
        },
        "gates": {
            "hard": {"passed": all(hard_checks.values()), "checks": hard_checks},
            "quality": {"passed": all(quality_checks.values()), "checks": quality_checks},
        },
        "warning": "Subsample-refit stability and expert/agronomic review are still required.",
    }
    _write_json(output_dir / "evaluation.json", report)
    return report


def prototype_overlap(prototypes: pd.DataFrame) -> pd.DataFrame:
    """Return every unordered same-hazard prototype-pair overlap diagnostic."""
    feature_columns = sorted(name for name in prototypes if name.startswith("f_"))
    records: list[dict[str, Any]] = []
    for hazard, frame in prototypes.groupby(HAZARD_COLUMN, sort=True):
        frame = frame.reset_index(drop=True)
        if len(frame) < 2:
            continue
        vectors = frame[feature_columns].to_numpy(dtype=float)
        for index in range(len(frame) - 1):
            deltas = vectors[index + 1 :] - vectors[index]
            distances = np.linalg.norm(deltas, axis=1)
            for offset, distance_value in enumerate(distances, start=index + 1):
                distance = float(distance_value)
                radius_sum = float(
                    frame.iloc[index]["radius"] + frame.iloc[offset]["radius"]
                )
                records.append(
                    {
                        HAZARD_COLUMN: str(hazard),
                        "archetype_id": str(frame.iloc[index]["archetype_id"]),
                        "other_archetype_id": str(frame.iloc[offset]["archetype_id"]),
                        "prototype_distance": distance,
                        "radius_sum": radius_sum,
                        "overlap_ratio": (
                            radius_sum / distance if distance > 0 else float("inf")
                        ),
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            HAZARD_COLUMN, "archetype_id", "other_archetype_id",
            "prototype_distance", "radius_sum", "overlap_ratio",
        ],
    )


def adaptive_min_cluster_size(event_count: int, config: ArchetypeConfig) -> int:
    return max(config.min_cluster_floor, int(math.ceil(config.min_cluster_fraction * event_count)))


def _fit_hdbscan(
    matrix: np.ndarray,
    config: ArchetypeConfig,
    backend: str,
    *,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    min_samples = min(config.min_samples, len(matrix))
    if backend == "gpu":
        import cupy as cp
        from cuml.cluster.hdbscan import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
        ).fit(cp.asarray(matrix, dtype=cp.float32))
        return cp.asnumpy(model.labels_).astype(int), cp.asnumpy(model.probabilities_).astype(float)
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "CPU archetype discovery requires scikit-learn>=1.3; install server/requirements.txt"
        ) from exc
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        n_jobs=-1 if len(matrix) >= 10_000 else 1,
        copy=True,
    ).fit(matrix)
    return model.labels_.astype(int), model.probabilities_.astype(float)


def _resolve_engine(engine: str) -> str:
    if engine == "cpu":
        return "cpu"
    try:
        import cupy  # noqa: F401
        import cuml  # noqa: F401
    except ImportError:
        if engine == "gpu":
            raise RuntimeError("GPU archetype discovery requires RAPIDS cuML and CuPy")
        return "cpu"
    return "gpu"


def _hazards(rows: pd.DataFrame) -> pd.Series:
    return rows[HAZARD_COLUMN].fillna("unattributed_decline").astype(str).str.strip().replace(
        "", "unattributed_decline"
    )


def _numeric(rows: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)


def _observed_robust_prototype(vectors: np.ndarray) -> np.ndarray:
    center = np.median(vectors, axis=0)
    return vectors[int(np.argmin(np.linalg.norm(vectors - center, axis=1)))].copy()


def _diagnostic_label(hazard: str, archetype_id: str) -> str:
    return f"{hazard.replace('_', ' ').title()} archetype {archetype_id[-8:]}"


def _archetype_id(hazard: str, prototype: np.ndarray, scope: str) -> str:
    payload = json.dumps(
        [hazard, scope, [round(float(value), 8) for value in prototype]],
        separators=(",", ":"),
    )
    return "archetype:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_scope(
    schema: dict[str, Any], config: ArchetypeConfig, cutoff: str,
    policy_version: str, policy_sha256: str,
) -> str:
    return _json_sha256(
        {
            "schema": schema,
            "config": asdict(config),
            "cutoff": cutoff,
            "policy_version": policy_version,
            "policy_sha256": policy_sha256,
        }
    )[:16]


def _model_version(
    schema: dict[str, Any],
    config: ArchetypeConfig,
    cutoff: str,
    policy_version: str,
    policy_sha256: str,
    prototypes: Iterable[dict[str, Any]],
) -> str:
    digest = _json_sha256(
        {
            "schema": schema,
            "config": asdict(config),
            "cutoff": cutoff,
            "policy_version": policy_version,
            "policy_sha256": policy_sha256,
            "prototypes": list(prototypes),
        }
    )
    return "archetype-v2-" + digest[:12]


def _published_labels_unique(catalog: pd.DataFrame) -> bool:
    if "publish_status" not in catalog:
        return True
    published = catalog[catalog["publish_status"].astype(str) == "published"]
    if published.empty:
        return True
    normalized = published["label"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return bool(normalized.is_unique)


def _json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_assignment_model(
    model_dir: Path,
    schema: dict[str, Any],
    prototypes: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    model_version = str(manifest.get("model_version") or "")
    if not model_version or str(schema.get("model_version")) != model_version:
        raise ValueError("model manifest and feature schema versions do not match")
    if _json_sha256(schema) != str(manifest.get("feature_schema_sha256") or ""):
        raise ValueError("feature schema hash does not match the model manifest")
    required = {"archetype_id", HAZARD_COLUMN, "radius", "model_version"}
    missing = sorted(required - set(prototypes.columns))
    if missing:
        raise ValueError("prototypes are missing columns: " + ", ".join(missing))
    if prototypes["archetype_id"].astype(str).duplicated().any():
        raise ValueError("prototype archetype IDs are not unique")
    if set(prototypes["model_version"].astype(str)) != {model_version}:
        raise ValueError("prototype model versions do not match the manifest")
    feature_columns = sorted(name for name in prototypes if name.startswith("f_"))
    if len(feature_columns) != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("prototype feature width does not match the feature schema")
    numeric = prototypes[[*feature_columns, "radius"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not (prototypes["radius"].astype(float) > 0).all():
        raise ValueError("prototype values and radii must be finite and positive")
    expected = manifest.get("artifacts") or {}
    for name in ("feature_schema.json", "prototypes.parquet"):
        digest = expected.get(name)
        if not digest or _file_sha256(model_dir / name) != str(digest):
            raise ValueError(f"model artifact hash mismatch: {name}")


def _require_columns(rows: pd.DataFrame, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError("missing archetype columns: " + ", ".join(missing))


def _validate_feature_values(rows: pd.DataFrame) -> None:
    specs = {item["name"]: item for item in FEATURE_SCHEMA["features"]}
    nullable = {
        "worst_attributed_ndvi_delta",
        "worst_attributed_ndmi_delta",
        "worst_attributed_psri_delta",
        "hazard_intensity",
    }
    for name, spec in specs.items():
        values = _numeric(rows, name)
        finite = np.isfinite(values)
        if name not in nullable and not finite.all():
            raise ValueError(f"archetype feature {name} must be finite")
        if spec["kind"] in {"binary", "bounded"}:
            invalid = finite & ((values < 0) | (values > 1))
            if invalid.any():
                raise ValueError(f"archetype feature {name} must be in [0, 1]")
        if spec["kind"] == "binary" and (finite & ~np.isin(values, [0.0, 1.0])).any():
            raise ValueError(f"archetype feature {name} must be binary")
    for value_name in nullable:
        flag_name = f"{value_name}_missing"
        value_missing = ~np.isfinite(_numeric(rows, value_name))
        flag = _numeric(rows, flag_name) == 1
        if not np.array_equal(value_missing, flag):
            raise ValueError(f"{flag_name} does not match missing {value_name} values")
