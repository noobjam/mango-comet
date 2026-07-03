"""Deterministic subsample stability checks for event archetypes V2.

The production check performs exactly two hazard-stratified refits.  Each
refit uses 80% of the events in every hazard, refits the hazard-local feature
scaler on that subsample, and reuses the discovery model's HDBSCAN
configuration.  It never uses holdout events.

Stability is measured against the frozen full-training discovery assignment:

* adjusted Rand index (ARI) uses every sampled event, including noise;
* each reference archetype is matched to the refit cluster with the largest
  Jaccard overlap on sampled event IDs;
* the conservative Jaccard aggregate is the minimum, across hazard/refit
  pairs, of the median best-match Jaccard for that pair; and
* an archetype is supported only when its best-match Jaccard is at least 0.70
  in both refits.

The sample fraction and persisted-config overrides exist for focused tests.
Production callers must use the defaults recorded in the returned summary.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .anchor_features_v2 import MODEL_FEATURE_COLUMNS
from .archetypes_v2 import (
    HAZARD_COLUMN,
    NOVEL_ARCHETYPE_ID,
    ArchetypeConfig,
    _file_sha256,
    _fit_hdbscan,
    _resolve_engine,
    _validate_assignment_model,
    adaptive_min_cluster_size,
    fit_feature_schema,
    transform_features,
)


DEFAULT_SAMPLE_FRACTION = 0.80
DEFAULT_SEEDS = (104_729, 130_363)
RUN_COUNT = 2
ARI_GATE = 0.80
JACCARD_GATE = 0.70
SUPPORTED_REFERENCE_FRACTION_GATE = 0.75


def evaluate_subsample_stability(
    training_rows: pd.DataFrame,
    model_dir: Path,
    *,
    output_dir: Path,
    runs: int = RUN_COUNT,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run two deterministic stratified refits and persist their diagnostics.

    ``training_rows`` must be the same one-anchor-per-event cohort used by
    :func:`discover_archetypes`.  Sampling is deterministic under input row
    reordering: event IDs are sorted before a hazard-specific seeded sample is
    drawn.  The sample size is ``floor(hazard_count * sample_fraction)``, with
    one row retained for non-empty hazards.
    """
    _validate_arguments(training_rows, runs, sample_fraction, seeds, config_overrides)
    model_dir = model_dir.expanduser().resolve()
    artifact_dir = output_dir.expanduser().resolve()
    if artifact_dir == model_dir or artifact_dir.is_relative_to(model_dir):
        raise ValueError("stability output must not be inside the frozen model directory")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    schema = _read_json(model_dir / "feature_schema.json")
    manifest = _read_json(model_dir / "archetype_manifest.json")
    assignments = pd.read_parquet(model_dir / "training_assignments.parquet")
    prototypes = pd.read_parquet(model_dir / "prototypes.parquet")
    config = _frozen_config(manifest, config_overrides)
    backend = _resolve_engine(config.engine)
    model_version = str(manifest["model_version"])
    _validate_assignment_model(model_dir, schema, prototypes, manifest)
    _validate_model_artifacts(schema, manifest, assignments, prototypes, model_version)
    expected_assignments_hash = (manifest.get("artifacts") or {}).get(
        "training_assignments.parquet"
    )
    if (
        not expected_assignments_hash
        or _file_sha256(model_dir / "training_assignments.parquet")
        != str(expected_assignments_hash)
    ):
        raise ValueError("model artifact hash mismatch: training_assignments.parquet")

    cohort = _aligned_reference(training_rows, assignments, prototypes)
    prototype_ids_by_hazard = {
        str(hazard): sorted(frame["archetype_id"].astype(str).tolist())
        for hazard, frame in prototypes.groupby(HAZARD_COLUMN, sort=True)
    }
    hazards = sorted(cohort[HAZARD_COLUMN].unique().tolist())
    missing_hazards = sorted(set(prototype_ids_by_hazard) - set(hazards))
    if missing_hazards:
        raise ValueError(
            "prototype hazards are missing from the training cohort: "
            + ", ".join(missing_hazards)
        )

    detail_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_index, seed in enumerate(tuple(int(value) for value in seeds), start=1):
        hazard_summaries: list[dict[str, Any]] = []
        for hazard in hazards:
            hazard_positions = np.flatnonzero(
                cohort[HAZARD_COLUMN].to_numpy(dtype=object) == hazard
            )
            sample_positions = _deterministic_sample_positions(
                cohort.iloc[hazard_positions]["event_id"].to_numpy(dtype=str),
                hazard_positions,
                sample_fraction,
                seed,
                hazard,
            )
            sample = training_rows.iloc[sample_positions]
            reference = cohort.iloc[sample_positions]
            # Stability covers the complete train-time fitting path. Reusing
            # the full-cohort scaler would hide unstable medians/IQRs.
            refit_schema = fit_feature_schema(sample)
            matrix = transform_features(sample, refit_schema)
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"transformed stability features are non-finite for hazard {hazard!r}"
                )
            minimum = adaptive_min_cluster_size(len(sample), config)
            refit_labels = _refit_labels(sample, matrix, config, backend, minimum)
            reference_ids = prototype_ids_by_hazard.get(hazard, [])
            reference_labels = _reference_label_codes(reference, reference_ids)
            ari = float(adjusted_rand_score(reference_labels, refit_labels))
            mutually_assigned = (reference_labels >= 0) & (refit_labels >= 0)
            mutual_non_noise_ari = (
                float(
                    adjusted_rand_score(
                        reference_labels[mutually_assigned], refit_labels[mutually_assigned]
                    )
                )
                if int(mutually_assigned.sum()) >= 2
                else 0.0
            )
            matches = _best_jaccard_matches(reference, refit_labels, reference_ids)
            jaccards = [float(item["best_jaccard"]) for item in matches]
            median_jaccard = float(np.median(jaccards)) if jaccards else None
            refit_cluster_count = int(len(set(refit_labels.tolist()) - {-1}))
            hazard_summary = {
                "run_index": run_index,
                "seed": seed,
                HAZARD_COLUMN: hazard,
                "sample_event_count": int(len(sample)),
                "reference_archetype_count": int(len(reference_ids)),
                "refit_cluster_count": refit_cluster_count,
                "min_cluster_size": int(minimum),
                "adjusted_rand_index": ari,
                "mutual_non_noise_adjusted_rand_index": mutual_non_noise_ari,
                "mutual_non_noise_event_count": int(mutually_assigned.sum()),
                "median_best_jaccard": median_jaccard,
            }
            hazard_summaries.append(hazard_summary)
            detail_records.append({"row_type": "hazard_summary", **hazard_summary})
            for match in matches:
                detail_records.append(
                    {
                        "row_type": "archetype_match",
                        **hazard_summary,
                        **match,
                        "meets_jaccard_0_70": bool(
                            float(match["best_jaccard"]) >= JACCARD_GATE
                        ),
                    }
                )
        run_summaries.append(
            {
                "run_index": run_index,
                "seed": seed,
                "hazards": hazard_summaries,
                "minimum_adjusted_rand_index": min(
                    item["adjusted_rand_index"] for item in hazard_summaries
                ),
                "minimum_mutual_non_noise_adjusted_rand_index": min(
                    item["mutual_non_noise_adjusted_rand_index"]
                    for item in hazard_summaries
                ),
            }
        )

    detail = pd.DataFrame(detail_records)
    match_rows = detail[detail["row_type"] == "archetype_match"].copy()
    support_by_reference = _cross_run_support(match_rows, prototypes)
    support_lookup = support_by_reference.set_index("reference_archetype_id")[
        "supported_both_runs"
    ]
    detail["supported_both_runs"] = detail.get("reference_archetype_id", pd.Series()).map(
        support_lookup
    )

    hazard_rows = detail[detail["row_type"] == "hazard_summary"]
    ari_minimum = float(hazard_rows["adjusted_rand_index"].min())
    mutual_non_noise_ari_minimum = float(
        hazard_rows["mutual_non_noise_adjusted_rand_index"].min()
    )
    matched_medians = pd.to_numeric(
        hazard_rows["median_best_jaccard"], errors="coerce"
    ).dropna()
    conservative_jaccard = float(matched_medians.min()) if len(matched_medians) else 0.0
    reference_count = int(len(support_by_reference))
    supported_count = int(support_by_reference["supported_both_runs"].sum())
    supported_fraction = float(supported_count / reference_count) if reference_count else 0.0
    checks = {
        "minimum_adjusted_rand_index": ari_minimum >= ARI_GATE,
        "minimum_mutual_non_noise_adjusted_rand_index": (
            mutual_non_noise_ari_minimum >= ARI_GATE
        ),
        "conservative_matched_jaccard": conservative_jaccard >= JACCARD_GATE,
        "supported_reference_fraction": (
            supported_fraction >= SUPPORTED_REFERENCE_FRACTION_GATE
        ),
    }

    references = []
    for row in support_by_reference.itertuples(index=False):
        references.append(
            {
                HAZARD_COLUMN: str(row.hazard_family),
                "reference_archetype_id": str(row.reference_archetype_id),
                "run_1_best_jaccard": float(row.run_1_best_jaccard),
                "run_2_best_jaccard": float(row.run_2_best_jaccard),
                "minimum_best_jaccard": float(row.minimum_best_jaccard),
                "supported_both_runs": bool(row.supported_both_runs),
            }
        )
    summary: dict[str, Any] = {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_version": model_version,
        "method": {
            "run_count": runs,
            "sample_fraction": float(sample_fraction),
            "sample_count_rule": "max(1, floor(hazard_event_count * sample_fraction))",
            "seeds": [int(value) for value in seeds],
            "stratification": HAZARD_COLUMN,
            "feature_transform": "refit_hazard_local_schema_per_subsample",
            "refit_engine": backend,
            "adjusted_rand_scope": "all_sampled_events_including_noise",
            "matched_jaccard_aggregate": (
                "minimum_across_hazard_runs_of_median_reference_best_match_jaccard"
            ),
            "support_rule": "best_match_jaccard_at_least_0.70_in_both_runs",
            "config_overrides": dict(config_overrides or {}),
        },
        "config": asdict(config),
        "runs": run_summaries,
        "references": references,
        "metrics": {
            "minimum_adjusted_rand_index": ari_minimum,
            "minimum_mutual_non_noise_adjusted_rand_index": (
                mutual_non_noise_ari_minimum
            ),
            "conservative_matched_jaccard": conservative_jaccard,
            "reference_archetype_count": reference_count,
            "supported_reference_count": supported_count,
            "supported_reference_fraction": supported_fraction,
        },
        "gates": {
            "thresholds": {
                "minimum_adjusted_rand_index": ARI_GATE,
                "minimum_mutual_non_noise_adjusted_rand_index": ARI_GATE,
                "conservative_matched_jaccard": JACCARD_GATE,
                "supported_reference_fraction": SUPPORTED_REFERENCE_FRACTION_GATE,
            },
            "checks": checks,
            "passed": bool(all(checks.values())),
        },
        "warning": "Engineering stability only; agronomic review remains required.",
    }
    _write_parquet_atomic(detail, artifact_dir / "subsample_stability.parquet")
    _write_json_atomic(summary, artifact_dir / "subsample_stability.json")
    return summary


def _validate_arguments(
    rows: pd.DataFrame,
    runs: int,
    sample_fraction: float,
    seeds: Sequence[int],
    config_overrides: Mapping[str, Any] | None,
) -> None:
    required = {"event_id", "field_id", HAZARD_COLUMN, *MODEL_FEATURE_COLUMNS}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError("missing stability columns: " + ", ".join(missing))
    if rows.empty:
        raise ValueError("stability evaluation requires training event anchors")
    if rows["event_id"].astype(str).duplicated().any():
        raise ValueError("stability evaluation requires exactly one row per event_id")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if runs != RUN_COUNT:
        raise ValueError("stability evaluation requires exactly two refit runs")
    if len(seeds) != runs:
        raise ValueError("stability evaluation requires exactly two deterministic seeds")
    if int(seeds[0]) == int(seeds[1]):
        raise ValueError("stability evaluation seeds must be distinct")
    allowed = {item.name for item in fields(ArchetypeConfig)}
    unknown = sorted(set(config_overrides or {}) - allowed)
    if unknown:
        raise ValueError("unknown archetype config overrides: " + ", ".join(unknown))


def _frozen_config(
    manifest: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> ArchetypeConfig:
    persisted = dict(manifest.get("config", {}))
    allowed = {item.name for item in fields(ArchetypeConfig)}
    values = {name: persisted[name] for name in allowed if name in persisted}
    # ``auto`` must not silently choose a different backend for a stability
    # refit; reuse the backend that built the frozen reference model.
    values["engine"] = str(manifest.get("engine_used", values.get("engine", "cpu")))
    values.update(dict(overrides or {}))
    config = ArchetypeConfig(**values)
    config.validate()
    return config


def _validate_model_artifacts(
    schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
    assignments: pd.DataFrame,
    prototypes: pd.DataFrame,
    model_version: str,
) -> None:
    assignment_required = {"event_id", HAZARD_COLUMN, "archetype_id", "accepted"}
    prototype_required = {"archetype_id", HAZARD_COLUMN, "model_version"}
    if missing := sorted(assignment_required - set(assignments.columns)):
        raise ValueError("training assignments are missing columns: " + ", ".join(missing))
    if missing := sorted(prototype_required - set(prototypes.columns)):
        raise ValueError("prototypes are missing columns: " + ", ".join(missing))
    if assignments["event_id"].astype(str).duplicated().any():
        raise ValueError("frozen training assignments contain duplicate event IDs")
    if prototypes["archetype_id"].astype(str).duplicated().any():
        raise ValueError("frozen prototypes contain duplicate archetype IDs")
    if str(schema.get("model_version")) != model_version:
        raise ValueError("feature schema model version does not match the manifest")
    if set(prototypes["model_version"].astype(str)) != {model_version}:
        raise ValueError("prototype model versions do not match the manifest")
    if "model_version" in assignments and set(assignments["model_version"].astype(str)) != {
        model_version
    }:
        raise ValueError("training assignment model versions do not match the manifest")


def _aligned_reference(
    training_rows: pd.DataFrame,
    assignments: pd.DataFrame,
    prototypes: pd.DataFrame,
) -> pd.DataFrame:
    anchors = pd.DataFrame(
        {
            "event_id": training_rows["event_id"].astype(str).to_numpy(),
            HAZARD_COLUMN: _normalize_hazards(training_rows[HAZARD_COLUMN]).to_numpy(),
        }
    )
    frozen = assignments[["event_id", HAZARD_COLUMN, "archetype_id", "accepted"]].copy()
    frozen["event_id"] = frozen["event_id"].astype(str)
    frozen[HAZARD_COLUMN] = _normalize_hazards(frozen[HAZARD_COLUMN])
    frozen["archetype_id"] = frozen["archetype_id"].astype(str)
    frozen = frozen.set_index("event_id")
    anchors["reference_hazard"] = anchors["event_id"].map(frozen[HAZARD_COLUMN])
    anchors["reference_archetype_id"] = anchors["event_id"].map(frozen["archetype_id"])
    anchors["reference_accepted"] = anchors["event_id"].map(frozen["accepted"])
    if anchors["reference_hazard"].isna().any() or len(anchors) != len(assignments):
        raise ValueError("training anchor event IDs do not exactly match frozen assignments")
    if not (anchors[HAZARD_COLUMN] == anchors["reference_hazard"]).all():
        raise ValueError("training anchor hazards do not match frozen assignments")
    prototype_ids = set(prototypes["archetype_id"].astype(str))
    accepted_ids = set(
        anchors.loc[anchors["reference_accepted"].astype(bool), "reference_archetype_id"]
        .astype(str)
        .tolist()
    )
    if not accepted_ids <= prototype_ids:
        raise ValueError("accepted frozen assignments reference unknown prototypes")
    anchors["reference_accepted"] = anchors["reference_accepted"].astype(bool)
    anchors.loc[~anchors["reference_accepted"], "reference_archetype_id"] = (
        NOVEL_ARCHETYPE_ID
    )
    return anchors


def _deterministic_sample_positions(
    event_ids: np.ndarray,
    cohort_positions: np.ndarray,
    fraction: float,
    seed: int,
    hazard: str,
) -> np.ndarray:
    order = np.argsort(event_ids, kind="mergesort")
    count = max(1, int(math.floor(len(order) * fraction)))
    count = min(count, len(order))
    digest = hashlib.sha256(f"{int(seed)}\0{hazard}".encode()).digest()
    hazard_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(hazard_seed)
    selected_in_order = np.sort(rng.choice(len(order), size=count, replace=False))
    return cohort_positions[order[selected_in_order]]


def _refit_labels(
    sample: pd.DataFrame,
    matrix: np.ndarray,
    config: ArchetypeConfig,
    backend: str,
    minimum: int,
) -> np.ndarray:
    if len(sample) < minimum:
        return np.full(len(sample), -1, dtype=np.int64)
    labels, _ = _fit_hdbscan(
        matrix,
        config,
        backend,
        min_cluster_size=minimum,
    )
    labels = labels.astype(np.int64, copy=True)
    fields_array = sample["field_id"].astype(str).to_numpy()
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        mask = labels == label
        if int(mask.sum()) < minimum or len(set(fields_array[mask].tolist())) < int(
            config.minimum_field_support
        ):
            labels[mask] = -1
    return labels


def _reference_label_codes(reference: pd.DataFrame, archetype_ids: list[str]) -> np.ndarray:
    codes = np.full(len(reference), -1, dtype=np.int64)
    values = reference["reference_archetype_id"].astype(str).to_numpy()
    accepted = reference["reference_accepted"].astype(bool).to_numpy()
    for code, archetype_id in enumerate(archetype_ids):
        codes[accepted & (values == archetype_id)] = code
    return codes


def _best_jaccard_matches(
    reference: pd.DataFrame,
    refit_labels: np.ndarray,
    archetype_ids: list[str],
) -> list[dict[str, Any]]:
    reference_values = reference["reference_archetype_id"].astype(str).to_numpy()
    reference_accepted = reference["reference_accepted"].astype(bool).to_numpy()
    refit_counts = {
        int(label): int((refit_labels == label).sum())
        for label in np.unique(refit_labels)
        if int(label) >= 0
    }
    pairs = pd.DataFrame(
        {
            "reference_archetype_id": reference_values,
            "reference_accepted": reference_accepted,
            "refit_label": refit_labels,
        }
    )
    intersections = (
        pairs[pairs["reference_accepted"] & (pairs["refit_label"] >= 0)]
        .groupby(["reference_archetype_id", "refit_label"], sort=False)
        .size()
    )
    intersections_by_reference: dict[str, list[tuple[int, int]]] = {}
    for (archetype_id, label), intersection in intersections.items():
        intersections_by_reference.setdefault(str(archetype_id), []).append(
            (int(label), int(intersection))
        )
    records: list[dict[str, Any]] = []
    for archetype_id in archetype_ids:
        reference_count = int(
            (reference_accepted & (reference_values == archetype_id)).sum()
        )
        best_label: int | None = None
        best_intersection = 0
        best_union = reference_count
        best_jaccard = 0.0
        if archetype_id in intersections_by_reference:
            for label, intersection in intersections_by_reference[archetype_id]:
                union = reference_count + refit_counts[label] - intersection
                jaccard = float(intersection / union) if union else 0.0
                if (jaccard, intersection, -label) > (
                    best_jaccard,
                    best_intersection,
                    -(best_label if best_label is not None else 2**31),
                ):
                    best_label = label
                    best_intersection = intersection
                    best_union = union
                    best_jaccard = jaccard
        records.append(
            {
                "reference_archetype_id": archetype_id,
                "reference_sample_count": reference_count,
                "best_refit_label": best_label,
                "matched_intersection": best_intersection,
                "matched_union": best_union,
                "best_jaccard": best_jaccard,
            }
        )
    return records


def _cross_run_support(
    matches: pd.DataFrame, prototypes: pd.DataFrame
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    match_lookup = {
        (int(row.run_index), str(row.reference_archetype_id)): float(row.best_jaccard)
        for row in matches.itertuples(index=False)
    }
    for row in prototypes.sort_values([HAZARD_COLUMN, "archetype_id"]).itertuples(index=False):
        archetype_id = str(row.archetype_id)
        run_1 = match_lookup.get((1, archetype_id), 0.0)
        run_2 = match_lookup.get((2, archetype_id), 0.0)
        minimum = min(run_1, run_2)
        records.append(
            {
                HAZARD_COLUMN: str(row.hazard_family),
                "reference_archetype_id": archetype_id,
                "run_1_best_jaccard": run_1,
                "run_2_best_jaccard": run_2,
                "minimum_best_jaccard": minimum,
                "supported_both_runs": bool(minimum >= JACCARD_GATE),
            }
        )
    return pd.DataFrame(records)


def _normalize_hazards(values: pd.Series) -> pd.Series:
    return values.fillna("unattributed_decline").astype(str).str.strip().replace(
        "", "unattributed_decline"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ARI_GATE",
    "DEFAULT_SAMPLE_FRACTION",
    "DEFAULT_SEEDS",
    "JACCARD_GATE",
    "SUPPORTED_REFERENCE_FRACTION_GATE",
    "evaluate_subsample_stability",
]
