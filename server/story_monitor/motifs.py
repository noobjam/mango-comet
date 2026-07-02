"""Versioned HDBSCAN motif discovery and frozen prototype assignment.

Clustering is an offline discovery operation.  Monitoring updates never refit
the clusters: they transform a causal event prefix with the stored feature
schema and assign it to a frozen observed prototype, or to ``novel_unassigned``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "current_risk_rank",
    "peak_risk_rank",
    "mean_risk_rank",
    "risk_slope",
    "escalation_count",
    "deescalation_count",
    "event_age_days",
    "active_days",
    "severe_days",
    "quiet_days",
    "stage_transition_count",
    "delta_ndvi",
    "delta_ndmi",
    "delta_psri",
    "unique_acquisition_count",
    "spectral_age_days",
    "min_spi",
    "max_ponding_mm",
    "max_apparent_temperature",
    "max_wind_speed",
    "observed_coverage",
)
CATEGORICAL_FEATURES = (
    "lifecycle_state",
    "stage_family",
    "response_class",
)
HAZARD_COLUMN = "hazard_family"
NOVEL_MOTIF_ID = "novel_unassigned"


@dataclass(frozen=True)
class DiscoveryConfig:
    min_cluster_size: int = 100
    min_samples: int = 20
    radius_quantile: float = 0.95
    assignment_margin: float = 0.05
    engine: str = "cpu"

    def validate(self) -> None:
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0.5 <= self.radius_quantile < 1:
            raise ValueError("radius_quantile must be in [0.5, 1)")
        if self.assignment_margin < 0:
            raise ValueError("assignment_margin must be non-negative")
        if self.engine not in {"cpu", "gpu", "auto"}:
            raise ValueError("engine must be cpu, gpu, or auto")


def fit_feature_schema(rows: pd.DataFrame) -> dict[str, Any]:
    """Fit a deterministic robust scaler and categorical vocabulary."""
    schema: dict[str, Any] = {
        "version": "causal_prefix_features_v1",
        "numeric": [],
        "categorical": [],
    }
    for name in NUMERIC_FEATURES:
        values = _numeric_series(rows, name)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if len(finite) else 0.0
        q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        schema["numeric"].append({"name": name, "median": median, "scale": scale})
    for name in CATEGORICAL_FEATURES:
        if name in rows:
            values = rows[name].fillna("missing").astype(str).str.strip().replace("", "missing")
            categories = sorted(set(values.tolist()) | {"missing", "unknown"})
        else:
            categories = ["missing", "unknown"]
        schema["categorical"].append({"name": name, "categories": categories})
    schema["feature_names"] = _transformed_feature_names(schema)
    return schema


def transform_features(rows: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    """Apply a stored schema without learning from monitoring rows."""
    columns: list[np.ndarray] = []
    for spec in schema["numeric"]:
        values = _numeric_series(rows, spec["name"])
        values = np.where(np.isfinite(values), values, float(spec["median"]))
        columns.append(((values - float(spec["median"])) / float(spec["scale"]))[:, None])
    for spec in schema["categorical"]:
        if spec["name"] in rows:
            raw = rows[spec["name"]].fillna("missing").astype(str).str.strip().replace("", "missing")
        else:
            raw = pd.Series(["missing"] * len(rows), index=rows.index)
        known = set(spec["categories"])
        values = np.asarray([value if value in known else "unknown" for value in raw])
        for category in spec["categories"]:
            columns.append((values == category).astype(float)[:, None])
    return np.hstack(columns) if columns else np.empty((len(rows), 0), dtype=float)


def discover_motifs(
    rows: pd.DataFrame,
    model_dir: Path,
    *,
    config: DiscoveryConfig = DiscoveryConfig(),
    training_cutoff: str,
    policy_version: str,
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Discover hazard-stratified density clusters and persist a frozen model."""
    config.validate()
    _require_columns(rows, (HAZARD_COLUMN, "event_id"))
    if rows.empty:
        raise ValueError("motif discovery requires at least one causal prefix")

    ordered = rows.copy()
    ordered[HAZARD_COLUMN] = ordered[HAZARD_COLUMN].fillna("unattributed").astype(str)
    sort_columns = [HAZARD_COLUMN, "event_id"]
    if "timeline_bucket" in ordered:
        sort_columns.append("timeline_bucket")
    ordered = ordered.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    schema = fit_feature_schema(ordered)
    matrix = transform_features(ordered, schema)
    labels = np.full(len(ordered), -1, dtype=int)
    membership = np.zeros(len(ordered), dtype=float)
    backend = _resolve_engine(config.engine)
    model_scope = _model_scope(
        schema, config, training_cutoff, policy_version, policy_sha256
    )

    prototypes: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    next_label = 0
    for hazard, index in ordered.groupby(HAZARD_COLUMN, sort=True).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if len(positions) < config.min_cluster_size:
            continue
        local_labels, local_membership = _fit_hdbscan(matrix[positions], config, backend)
        for local_label in sorted(label for label in set(local_labels.tolist()) if label >= 0):
            member_positions = positions[local_labels == local_label]
            if len(member_positions) < config.min_cluster_size:
                continue
            vectors = matrix[member_positions]
            prototype = _observed_robust_prototype(vectors)
            distances = np.linalg.norm(vectors - prototype, axis=1)
            radius = max(float(np.quantile(distances, config.radius_quantile)), 1e-9)
            motif_id = _motif_id(str(hazard), prototype, model_scope)
            labels[member_positions] = next_label
            membership[member_positions] = local_membership[local_labels == local_label]
            record = {
                "motif_id": motif_id,
                "hazard_family": str(hazard),
                "member_count": int(len(member_positions)),
                "radius": radius,
                "radius_quantile": config.radius_quantile,
                "prototype_method": "observed_nearest_robust_center_v1",
            }
            record.update({f"f_{i:03d}": float(value) for i, value in enumerate(prototype)})
            prototypes.append(record)
            catalog.append({
                "motif_id": motif_id,
                "hazard_family": str(hazard),
                "member_count": int(len(member_positions)),
                "training_membership_mean": float(np.mean(membership[member_positions])),
                "status": "discovered_unreviewed",
                "label": _motif_label(ordered.iloc[member_positions], str(hazard)),
            })
            next_label += 1

    if not prototypes:
        raise ValueError(
            "HDBSCAN discovered no publishable motifs. Inspect noise rates and validation data "
            "before changing min_cluster_size or min_samples."
        )
    model_version = _model_version(
        schema, config, training_cutoff, policy_version, policy_sha256, prototypes
    )
    model_dir.mkdir(parents=True, exist_ok=False)
    schema.update({
        "model_version": model_version,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "assignment_margin": config.assignment_margin,
        "novel_motif_id": NOVEL_MOTIF_ID,
    })
    (model_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(prototypes).to_parquet(model_dir / "prototypes.parquet", index=False)
    pd.DataFrame(catalog).to_parquet(model_dir / "motif_catalog.parquet", index=False)
    training = ordered[["event_id", HAZARD_COLUMN]].copy()
    training["discovery_label"] = labels
    training["training_membership"] = membership
    training.to_parquet(model_dir / "training_assignments.parquet", index=False)
    manifest = {
        "model_version": model_version,
        "training_cutoff": training_cutoff,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "feature_version": schema["version"],
        "engine_requested": config.engine,
        "engine_used": backend,
        "row_count": len(ordered),
        "motif_count": len(prototypes),
        "noise_count": int(np.sum(labels < 0)),
        "config": config.__dict__,
        "warning": "Unreviewed discovery model; thresholds and motifs require agronomic validation.",
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def assign_frozen_motifs(rows: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Assign causal prefixes to frozen prototypes with explicit novelty."""
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    prototypes = pd.read_parquet(model_dir / "prototypes.parquet")
    output = rows.copy()
    if HAZARD_COLUMN not in output:
        output[HAZARD_COLUMN] = "unattributed"
    output[HAZARD_COLUMN] = output[HAZARD_COLUMN].fillna("unattributed").astype(str)
    matrix = transform_features(output, schema)
    result: list[dict[str, Any]] = []
    feature_cols = [f"f_{i:03d}" for i in range(matrix.shape[1])]
    margin_required = float(schema.get("assignment_margin", 0.05))
    for position, (_, row) in enumerate(output.iterrows()):
        candidates = prototypes[prototypes[HAZARD_COLUMN] == row[HAZARD_COLUMN]]
        if candidates.empty:
            result.append(_novel_assignment("no_hazard_prototype", schema["model_version"]))
            continue
        vectors = candidates[feature_cols].to_numpy(dtype=float)
        distances = np.linalg.norm(vectors - matrix[position], axis=1)
        ordering = np.argsort(distances, kind="stable")
        best_index = int(ordering[0])
        best = candidates.iloc[best_index]
        best_distance = float(distances[best_index])
        second_distance = float(distances[ordering[1]]) if len(ordering) > 1 else float("inf")
        margin = second_distance - best_distance
        radius = float(best["radius"])
        accepted = best_distance <= radius and margin >= margin_required
        result.append({
            "motif_id": str(best["motif_id"]) if accepted else NOVEL_MOTIF_ID,
            "assignment_method": "frozen_prototype_radius_v1",
            "assignment_distance": best_distance,
            "distance_ratio": best_distance / radius if radius > 0 else None,
            "assignment_margin": margin if np.isfinite(margin) else None,
            "runner_up_motif_id": (
                str(candidates.iloc[int(ordering[1])]["motif_id"]) if len(ordering) > 1 else None
            ),
            "candidate_motif_id": str(best["motif_id"]),
            "assignment_reason": "within_radius_and_margin" if accepted else "outside_radius_or_ambiguous",
            "motif_model_version": schema["model_version"],
        })
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(result)], axis=1)


def _fit_hdbscan(
    matrix: np.ndarray, config: DiscoveryConfig, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "gpu":
        import cupy as cp
        from cuml.cluster.hdbscan import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min(config.min_samples, len(matrix)),
            cluster_selection_method="eom",
        )
        model.fit(cp.asarray(matrix, dtype=cp.float32))
        return cp.asnumpy(model.labels_).astype(int), cp.asnumpy(model.probabilities_).astype(float)
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "CPU motif discovery requires scikit-learn>=1.3; install server/requirements.txt"
        ) from exc
    model = HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=min(config.min_samples, len(matrix)),
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
            raise RuntimeError("GPU motif discovery requires RAPIDS cuML and CuPy")
        return "cpu"
    return "gpu"


def _observed_robust_prototype(vectors: np.ndarray) -> np.ndarray:
    center = np.median(vectors, axis=0)
    return vectors[int(np.argmin(np.linalg.norm(vectors - center, axis=1)))].copy()


def _motif_label(rows: pd.DataFrame, hazard: str) -> str:
    lifecycle = _mode(rows.get("lifecycle_state"), "active").lower()
    response = _mode(rows.get("response_class"), "")
    age = pd.to_numeric(rows.get("event_age_days"), errors="coerce").median()
    persistence = "persistent " if pd.notna(age) and float(age) >= 21 else ""
    phase = {
        "severe": "severe",
        "recovering": "recovering",
        "quiet_pending": "quiet-pending",
        "watch": "emerging",
    }.get(lifecycle, "active")
    response_suffix = {
        "severe_decline": " with severe crop-response evidence",
        "medium_decline": " with crop-response evidence",
        "recovery": " with recovery evidence",
    }.get(response, "")
    hazard_label = hazard.replace("_", " ")
    return f"{persistence}{phase} {hazard_label}{response_suffix}".strip().title()


def _mode(values: Any, fallback: str) -> str:
    if values is None:
        return fallback
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return fallback
    counts = series.value_counts()
    maximum = counts.max()
    return sorted(counts[counts == maximum].index)[0]


def _numeric_series(rows: pd.DataFrame, name: str) -> np.ndarray:
    if name not in rows:
        return np.full(len(rows), np.nan, dtype=float)
    return pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)


def _transformed_feature_names(schema: dict[str, Any]) -> list[str]:
    names = [item["name"] for item in schema["numeric"]]
    for item in schema["categorical"]:
        names.extend(f"{item['name']}={category}" for category in item["categories"])
    return names


def _motif_id(hazard: str, prototype: np.ndarray, model_scope: str) -> str:
    payload = json.dumps(
        [hazard, model_scope, [round(float(value), 8) for value in prototype]],
        separators=(",", ":"),
    )
    return "motif:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_scope(
    schema: dict[str, Any], config: DiscoveryConfig, cutoff: str,
    policy_version: str, policy_sha256: str | None
) -> str:
    payload = json.dumps(
        {
            "schema": schema,
            "config": config.__dict__,
            "cutoff": cutoff,
            "policy": policy_version,
            "policy_sha256": policy_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_version(
    schema: dict[str, Any],
    config: DiscoveryConfig,
    cutoff: str,
    policy_version: str,
    policy_sha256: str | None,
    prototypes: Iterable[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "schema": schema,
            "config": config.__dict__,
            "cutoff": cutoff,
            "policy": policy_version,
            "policy_sha256": policy_sha256,
            "prototypes": list(prototypes),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "motif-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _novel_assignment(reason: str, model_version: str) -> dict[str, Any]:
    return {
        "motif_id": NOVEL_MOTIF_ID,
        "assignment_method": "frozen_prototype_radius_v1",
        "assignment_distance": None,
        "distance_ratio": None,
        "assignment_margin": None,
        "runner_up_motif_id": None,
        "candidate_motif_id": None,
        "assignment_reason": reason,
        "motif_model_version": model_version,
    }


def _require_columns(rows: pd.DataFrame, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError("missing motif columns: " + ", ".join(missing))
