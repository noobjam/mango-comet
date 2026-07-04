"""Deterministic spatial-temporal incident tracking primitives.

The functions in this module deliberately stop below the policy, CLI, and UI
layers.  They accept pandas frames plus a plain mapping, never inspect future
rows, and keep three identities separate:

* a weekly component (same-hazard significant grid cells),
* a persistent crop-independent exposure assembled from weekly components,
* a crop-impact incident created only after grouping an exposure by crop, and
* changing crop/stage context, which never participates in either identity.

WATCH rows may be attached as a frontier, but cannot create a component or a
temporal continuation.  Likewise, crop impact is retained as context/lineage;
it is not silently converted into pressure or biological death.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


CORE_STATES = frozenset({"ACTIVE", "SEVERE"})
WATCH_STATES = frozenset({"WATCH"})
IMPACT_STATES = frozenset({"RECOVERING", "CLOSED_RESPONSE_UNRESOLVED"})
TERMINAL_INCIDENT_STATES = frozenset(
    {
        "CLOSED_CANDIDATE_EXPIRED",
        "CLOSED_RECOVERED",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED",
        "CLOSED_RESPONSE_UNRESOLVED",
        "CLOSED_SEASON_CENSORED",
        "CLOSED_DATA_CENSORED",
        "MERGED_INTO",
    }
)


DEFAULT_CONFIG: dict[str, Any] = {
    "identity_namespace": "incident-tracking-v3",
    "policy_version": "incident-tracking-v3-unversioned",
    "cell_size_km": 5.0,
    "origin_lon": 0.0,
    "origin_lat": 0.0,
    "reference_latitude": -2.0,
    "fdr_alpha": 0.05,
    "minimum_active_fields": 2,
    "minimum_monitored_fields": 5,
    "minimum_coverage_ratio": 0.0,
    "severe_override_min_fields": 2,
    "severe_override_min_fresh_response_fields": 1,
    "allow_severe_override": True,
    "frontier_distance_cells": 1,
    "max_gap_weeks": 2,
    "max_centroid_distance_km": 15.0,
    "episode_jaccard_weight": 0.35,
    "cell_jaccard_weight": 0.25,
    "field_jaccard_weight": 0.15,
    "distance_weight": 0.15,
    "stage_cosine_weight": 0.10,
    "gap_penalty": 0.10,
    "continuation_threshold": 0.45,
    "lineage_threshold": 0.30,
    "minimum_lineage_jaccard": 0.20,
    "confirmation_observed_weeks": 2,
    "candidate_expiry_observed_weeks": 2,
    "quiet_observed_weeks": 2,
    "recovery_observed_weeks": 1,
    "maximum_recovery_observed_weeks": 4,
    "severe_confirmation_min_fields": 2,
    "severe_confirmation_min_fresh_response_fields": 1,
}

_CONFIG_ALIASES = {
    "policy_version": "version",
    "cell_size_km": "grid_cell_size_km",
    "origin_lon": "grid_origin_lon",
    "origin_lat": "grid_origin_lat",
    "minimum_monitored_fields": "minimum_evaluable_fields",
    "max_gap_weeks": "max_link_gap_weeks",
    "max_centroid_distance_km": "spatial_scale_km",
    "continuation_threshold": "minimum_link_score",
    "confirmation_observed_weeks": "confirmation_weeks",
    "quiet_observed_weeks": "quiet_close_weeks",
    "maximum_recovery_observed_weeks": "recovery_grace_weeks",
    "reference_latitude": "grid_origin_lat",
}

_WEIGHT_ALIASES = {
    "episode_jaccard_weight": "active_episode_overlap",
    "cell_jaccard_weight": "cell_or_footprint_overlap",
    "field_jaccard_weight": "recent_member_overlap",
    "distance_weight": "centroid_proximity",
    "stage_cosine_weight": "stage_distribution_similarity",
}


COMPONENT_COLUMNS = (
    "timeline_bucket",
    "hazard_family",
    "component_id",
    "cell_ids_json",
    "core_cell_count",
    "active_field_count",
    "severe_field_count",
    "watch_frontier_field_count",
    "impact_field_count",
    "monitored_field_count",
    "center_x_km",
    "center_y_km",
    "center_lon",
    "center_lat",
    "footprint_area_km2",
    "max_z_score",
    "stage_distribution",
    "crop_distribution",
)

MEMBERSHIP_COLUMNS = (
    "timeline_bucket",
    "hazard_family",
    "component_id",
    "field_id",
    "crop_instance_id",
    "episode_id",
    "membership_role",
    "event_state",
    "response_class",
    "fresh_response_evidence",
    "evaluable",
    "is_data_gap",
    "stage_bucket",
    "crop_name",
    "grid_id",
    "knowledge_time",
)


@dataclass(frozen=True)
class ComponentBuildResult:
    components: pd.DataFrame
    memberships: pd.DataFrame


@dataclass(frozen=True)
class TrackingResult:
    assignments: pd.DataFrame
    lineage: pd.DataFrame
    previous_updates: pd.DataFrame


def _cfg(config: Mapping[str, Any] | None, name: str) -> Any:
    if config is None:
        return DEFAULT_CONFIG[name]
    alias = _CONFIG_ALIASES.get(name)
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        if alias and alias in config:
            return config[alias]
        weights = config.get("link_weights")
    else:
        if hasattr(config, name):
            return getattr(config, name)
        if alias and hasattr(config, alias):
            return getattr(config, alias)
        weights = getattr(config, "link_weights", None)
    weight_name = _WEIGHT_ALIASES.get(name)
    if weight_name and weights:
        weight_mapping = dict(weights)
        if weight_name in weight_mapping:
            return weight_mapping[weight_name]
    return DEFAULT_CONFIG[name]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _with_aliases(frame: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    """Copy a frame and expose canonical names without removing source names."""
    output = frame.copy()
    for canonical, source in aliases.items():
        if canonical not in output and source in output:
            output[canonical] = output[source]
    return output


def _stable_id(prefix: str, parts: Iterable[Any], *, length: int = 20) -> str:
    normalized = "\x1f".join("" if item is None else str(item) for item in parts)
    return f"{prefix}_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:length]}"


def _bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("timeline_bucket is required")
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("timeline_bucket is invalid")
    return parsed.date().isoformat()


def _hazard(value: Any) -> str:
    text = str(value or "other").strip().lower()
    return text or "other"


def _truth(value: Any, *, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    return bool(value)


def _crop(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(value or "unknown_crop").strip().lower()
    ).strip("_") or "unknown_crop"


def _identity_scope(config: Mapping[str, Any] | None) -> tuple[str, str]:
    if isinstance(config, Mapping):
        policy_sha = str(config.get("policy_sha256") or config.get("source_sha256") or "")
    else:
        policy_sha = str(getattr(config, "source_sha256", "") or "") if config is not None else ""
    policy_scope = str(_cfg(config, "policy_version"))
    if policy_sha:
        policy_scope = f"{policy_scope}:{policy_sha}"
    return str(_cfg(config, "identity_namespace")), policy_scope


def stable_component_id(
    hazard_family: str,
    timeline_bucket: Any,
    grid_ids: Iterable[str],
    config: Mapping[str, Any] | None = None,
) -> str:
    """Return a weekly component ID independent of crop, stage, and row order."""
    cells = tuple(sorted(set(str(value) for value in grid_ids)))
    if not cells:
        raise ValueError("stable component identity requires at least one grid cell")
    namespace, policy = _identity_scope(config)
    return _stable_id("component", (namespace, policy, _hazard(hazard_family), _bucket(timeline_bucket), *cells))


def stable_exposure_id(
    hazard_family: str,
    first_evidence_bucket: Any,
    seed_grid_ids: Iterable[str],
    config: Mapping[str, Any] | None = None,
) -> str:
    """Return a persistent track ID; dynamic stage/crop values are excluded."""
    cells = tuple(sorted(set(str(value) for value in seed_grid_ids)))
    if not cells:
        raise ValueError("stable exposure identity requires at least one seed grid cell")
    namespace, policy = _identity_scope(config)
    return _stable_id("exposure", (namespace, policy, _hazard(hazard_family), _bucket(first_evidence_bucket), *cells))


def stable_crop_incident_id(
    exposure_id: str,
    crop_name: Any,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Create a crop-impact story ID only after an exposure is grouped by crop."""
    if not str(exposure_id or "").startswith("exposure_"):
        raise ValueError("crop incident identity requires an exposure_ ID")
    crop = _crop(crop_name)
    namespace, policy = _identity_scope(config)
    return _stable_id("incident", (namespace, policy, str(exposure_id), crop))


def build_crop_incident_assignments(
    exposure_assignments: pd.DataFrame,
    component_memberships: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Group tracked exposure membership by crop and assign crop-impact IDs."""
    columns = [
        "exposure_id", "crop_name_normalized", "incident_id", "field_count",
        "episode_count",
    ]
    if exposure_assignments.empty or component_memberships.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        exposure_assignments, ("component_id", "exposure_id"), "exposure assignments"
    )
    _require_columns(
        component_memberships,
        ("component_id", "field_id", "membership_role", "crop_name"),
        "component memberships",
    )
    if exposure_assignments["component_id"].astype(str).duplicated().any():
        raise ValueError("exposure assignments must be unique by component_id")
    joined = component_memberships.merge(
        exposure_assignments[["component_id", "exposure_id"]],
        on="component_id",
        how="inner",
        validate="many_to_one",
    )
    joined = joined[
        joined["membership_role"].astype(str).isin({"pressure_core", "impact_lag"})
    ].copy()
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["crop_name_normalized"] = joined["crop_name"].map(_crop)
    records: list[dict[str, Any]] = []
    for (exposure_id, crop_name), group in joined.groupby(
        ["exposure_id", "crop_name_normalized"], sort=True
    ):
        episodes = (
            int(group["episode_id"].dropna().astype(str).nunique())
            if "episode_id" in group
            else 0
        )
        records.append(
            {
                "exposure_id": str(exposure_id),
                "crop_name_normalized": str(crop_name),
                "incident_id": stable_crop_incident_id(str(exposure_id), crop_name, config),
                "field_count": int(group["field_id"].astype(str).nunique()),
                "episode_count": episodes,
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["exposure_id", "crop_name_normalized"], kind="mergesort"
    ).reset_index(drop=True)


def assign_metric_grid(
    rows: pd.DataFrame, config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Assign centroids to a deterministic fixed-width local metric grid."""
    _require_columns(rows, ("centroid_lon", "centroid_lat"), "grid input")
    output = rows.copy()
    lon = pd.to_numeric(output["centroid_lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(output["centroid_lat"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("grid input contains non-finite centroids")
    if ((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).any():
        raise ValueError("grid input contains invalid longitude/latitude")
    size = float(_cfg(config, "cell_size_km"))
    if not math.isfinite(size) or size <= 0:
        raise ValueError("cell_size_km must be positive and finite")
    origin_lon = float(_cfg(config, "origin_lon"))
    origin_lat = float(_cfg(config, "origin_lat"))
    reference_latitude = float(_cfg(config, "reference_latitude"))
    x_scale = 111.32 * math.cos(math.radians(reference_latitude))
    y_scale = 110.574
    if abs(x_scale) < 1e-9:
        raise ValueError("reference_latitude produces an undefined longitude scale")
    metric_x = (lon - origin_lon) * x_scale
    metric_y = (lat - origin_lat) * y_scale
    grid_x = np.floor(metric_x / size).astype(np.int64)
    grid_y = np.floor(metric_y / size).astype(np.int64)
    center_x = (grid_x + 0.5) * size
    center_y = (grid_y + 0.5) * size
    output["metric_x_km"] = metric_x
    output["metric_y_km"] = metric_y
    output["grid_x"] = grid_x
    output["grid_y"] = grid_y
    output["grid_id"] = [f"g:{x}:{y}" for x, y in zip(grid_x, grid_y)]
    output["grid_center_x_km"] = center_x
    output["grid_center_y_km"] = center_y
    output["grid_center_lon"] = origin_lon + center_x / x_scale
    output["grid_center_lat"] = origin_lat + center_y / y_scale
    return output


def normal_tail_p_value(z_scores: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
    """One-sided standard-normal upper-tail probability without SciPy."""
    values = np.asarray(z_scores, dtype=float)
    flat = values.reshape(-1)
    result = np.array(
        [math.nan if not math.isfinite(value) else 0.5 * math.erfc(value / math.sqrt(2.0)) for value in flat],
        dtype=float,
    ).reshape(values.shape)
    result = np.clip(result, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def benjamini_hochberg(
    p_values: Sequence[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjusted q-values and rejection flags, preserving input order."""
    if not 0 < float(alpha) <= 1:
        raise ValueError("alpha must be in (0, 1]")
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    finite = np.isfinite(values)
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("finite p-values must be in [0, 1]")
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_indices = np.flatnonzero(finite)
    if len(valid_indices):
        order = valid_indices[np.argsort(values[valid_indices], kind="mergesort")]
        count = len(order)
        ranked = values[order] * count / np.arange(1, count + 1, dtype=float)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted[order] = np.clip(ranked, 0.0, 1.0)
    rejected = np.isfinite(adjusted) & (adjusted <= float(alpha))
    return adjusted, rejected


def mark_significant_cells(
    cells: pd.DataFrame, config: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Apply one-sided normal tests and BH-FDR inside each hazard/week."""
    output = _with_aliases(
        cells,
        {
            "grid_x": "cell_x",
            "grid_y": "cell_y",
            "active_field_count": "active_count",
            "monitored_field_count": "evaluable_count",
        },
    ).reset_index(drop=True)
    required = {
        "timeline_bucket", "hazard_family", "grid_x", "grid_y",
        "active_field_count", "monitored_field_count",
    }
    _require_columns(output, required, "cell frame")
    key = ["timeline_bucket", "hazard_family", "grid_x", "grid_y"]
    if output.duplicated(key).any():
        raise ValueError("cell frame must be unique by week, hazard, grid_x, grid_y")
    if "z_score" not in output:
        _require_columns(output, ("expected_active_count", "active_variance"), "cell frame")
        observed = pd.to_numeric(output["active_field_count"], errors="coerce")
        expected = pd.to_numeric(output["expected_active_count"], errors="coerce")
        variance = pd.to_numeric(output["active_variance"], errors="coerce")
        output["z_score"] = (observed - expected) / np.sqrt(np.maximum(variance, 1e-12))
    output["p_value"] = normal_tail_p_value(pd.to_numeric(output["z_score"], errors="coerce"))
    output["fdr_q_value"] = np.nan
    output["fdr_significant"] = False
    alpha = float(_cfg(config, "fdr_alpha"))
    for _, indices in output.groupby(["timeline_bucket", "hazard_family"], sort=True).groups.items():
        positions = np.asarray(sorted(indices), dtype=np.int64)
        adjusted, rejected = benjamini_hochberg(output.loc[positions, "p_value"].to_numpy(), alpha)
        output.loc[positions, "fdr_q_value"] = adjusted
        output.loc[positions, "fdr_significant"] = rejected
    monitored = pd.to_numeric(output["monitored_field_count"], errors="coerce").fillna(0)
    active = pd.to_numeric(output["active_field_count"], errors="coerce").fillna(0)
    coverage_ok = monitored.ge(int(_cfg(config, "minimum_monitored_fields")))
    if "coverage_ratio" in output:
        coverage_ok &= pd.to_numeric(output["coverage_ratio"], errors="coerce").fillna(0).ge(
            float(_cfg(config, "minimum_coverage_ratio"))
        )
    if "adequate_coverage" in output:
        coverage_ok &= output["adequate_coverage"].fillna(False).astype(bool)
    ordinary = (
        output["fdr_significant"].astype(bool)
        & active.ge(int(_cfg(config, "minimum_active_fields")))
        & coverage_ok
    )
    severe_override = pd.Series(False, index=output.index)
    if bool(_cfg(config, "allow_severe_override")) and {
        "severe_field_count", "fresh_response_field_count"
    }.issubset(output.columns):
        severe_override = (
            pd.to_numeric(output["severe_field_count"], errors="coerce").fillna(0).ge(
                int(_cfg(config, "severe_override_min_fields"))
            )
            & pd.to_numeric(output["fresh_response_field_count"], errors="coerce").fillna(0).ge(
                int(_cfg(config, "severe_override_min_fresh_response_fields"))
            )
            & coverage_ok
        )
    output["coverage_adequate"] = coverage_ok
    output["significance_reason"] = np.select(
        [ordinary, severe_override], ["fdr_excess", "severe_response_override"], default="not_significant"
    )
    output["significant"] = ordinary | severe_override
    output["timeline_bucket"] = output["timeline_bucket"].map(_bucket)
    output["hazard_family"] = output["hazard_family"].map(_hazard)
    return output.sort_values(key, kind="mergesort").reset_index(drop=True)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _distribution(values: pd.Series) -> str:
    normalized = values.fillna("unknown").astype(str).str.strip().replace("", "unknown")
    counts = normalized.value_counts(sort=False)
    total = int(counts.sum())
    payload = {key: round(int(counts[key]) / total, 8) for key in sorted(counts.index)} if total else {}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _component_cell_groups(group: pd.DataFrame) -> list[list[tuple[int, int]]]:
    coordinates = sorted({(int(row.grid_x), int(row.grid_y)) for row in group.itertuples()})
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}
    union = _UnionFind(len(coordinates))
    for index, (x_value, y_value) in enumerate(coordinates):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = lookup.get((x_value + dx, y_value + dy))
                if other is not None:
                    union.union(index, other)
    groups: dict[int, list[tuple[int, int]]] = {}
    for index, coordinate in enumerate(coordinates):
        groups.setdefault(union.find(index), []).append(coordinate)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])


def build_weekly_components(
    cells: pd.DataFrame,
    field_rows: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> ComponentBuildResult:
    """Build 8-neighbor pressure components and attach non-forming context."""
    cell_frame = _with_aliases(
        cells,
        {
            "grid_x": "cell_x",
            "grid_y": "cell_y",
            "significant": "is_significant",
            "monitored_field_count": "evaluable_count",
        },
    )
    fields = _with_aliases(
        field_rows,
        {
            "grid_x": "cell_x",
            "grid_y": "cell_y",
            "episode_id": "event_id",
            "stage_bucket": "stage_family",
            "response_class": "daily_response_class",
            "fresh_response_evidence": "new_response_evidence",
            "is_data_gap": "is_data_gap_snapshot",
        },
    )
    if "is_canonical_field_hazard_week" in fields:
        fields = fields[
            fields["is_canonical_field_hazard_week"].fillna(False).astype(bool)
        ].copy()
    if not {"grid_x", "grid_y"}.issubset(fields.columns) and {
        "centroid_lon", "centroid_lat"
    }.issubset(fields.columns):
        if "centroid_available" in fields:
            fields = fields[fields["centroid_available"].fillna(False).astype(bool)].copy()
        reference_values = (
            pd.to_numeric(cell_frame["reference_latitude"], errors="coerce").dropna().unique()
            if "reference_latitude" in cell_frame
            else np.asarray([], dtype=float)
        )
        if len(reference_values) > 1:
            raise ValueError("cell frame contains multiple metric-grid reference latitudes")
        grid_config = {
            "cell_size_km": float(_cfg(config, "cell_size_km")),
            "origin_lon": float(_cfg(config, "origin_lon")),
            "origin_lat": float(_cfg(config, "origin_lat")),
            "reference_latitude": (
                float(reference_values[0])
                if len(reference_values)
                else float(_cfg(config, "reference_latitude"))
            ),
        }
        fields = assign_metric_grid(fields, grid_config)
    _require_columns(
        cell_frame,
        (
            "timeline_bucket", "hazard_family", "grid_x", "grid_y", "significant",
            "active_field_count", "monitored_field_count",
        ),
        "cell frame",
    )
    _require_columns(
        fields,
        ("timeline_bucket", "hazard_family", "grid_x", "grid_y", "field_id", "event_state"),
        "field frame",
    )
    for frame in (cell_frame, fields):
        frame["timeline_bucket"] = frame["timeline_bucket"].map(_bucket)
        frame["hazard_family"] = frame["hazard_family"].map(_hazard)
        frame["grid_x"] = pd.to_numeric(frame["grid_x"], errors="raise").astype("int64")
        frame["grid_y"] = pd.to_numeric(frame["grid_y"], errors="raise").astype("int64")
    if cell_frame.duplicated(["timeline_bucket", "hazard_family", "grid_x", "grid_y"]).any():
        raise ValueError("cell frame contains duplicate hazard/week grid cells")
    if fields.duplicated(["timeline_bucket", "hazard_family", "field_id"]).any():
        raise ValueError("field frame must be canonical by week, hazard, field_id")
    fields["event_state"] = fields["event_state"].fillna("").astype(str).str.upper()
    fields["grid_id"] = [f"g:{x}:{y}" for x, y in zip(fields["grid_x"], fields["grid_y"])]
    if "stage_bucket" not in fields and "stage_family" in fields:
        fields["stage_bucket"] = fields["stage_family"]
    for name, fallback in (
        ("crop_instance_id", None), ("episode_id", None),
        ("stage_bucket", "unknown"), ("crop_name", "unknown"),
        ("response_class", "no_new_event_response"),
        ("fresh_response_evidence", False), ("evaluable", True),
        ("is_data_gap", False),
        ("knowledge_time", None),
    ):
        if name not in fields:
            fields[name] = fallback
    if "impact_active" not in fields:
        fields["impact_active"] = False
    significant = cell_frame[cell_frame["significant"].fillna(False).astype(bool)].copy()
    if significant.empty:
        return ComponentBuildResult(pd.DataFrame(columns=COMPONENT_COLUMNS), pd.DataFrame(columns=MEMBERSHIP_COLUMNS))

    size = float(_cfg(config, "cell_size_km"))
    frontier = int(_cfg(config, "frontier_distance_cells"))
    component_records: list[dict[str, Any]] = []
    membership_records: list[dict[str, Any]] = []
    for (week, hazard), cell_group in significant.groupby(
        ["timeline_bucket", "hazard_family"], sort=True
    ):
        coordinate_groups = _component_cell_groups(cell_group)
        component_meta: list[dict[str, Any]] = []
        for coordinates in coordinate_groups:
            grid_ids = [f"g:{x}:{y}" for x, y in coordinates]
            component_id = stable_component_id(hazard, week, grid_ids, config)
            coordinate_set = set(coordinates)
            selected_cells = cell_group[
                [(int(x), int(y)) in coordinate_set for x, y in zip(cell_group["grid_x"], cell_group["grid_y"])]
            ]
            component_meta.append(
                {
                    "component_id": component_id,
                    "coordinates": coordinate_set,
                    "grid_ids": grid_ids,
                    "cells": selected_cells,
                    "active_hint": float(
                        pd.to_numeric(selected_cells["active_field_count"], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                }
            )
        local_fields = fields[
            (fields["timeline_bucket"] == week) & (fields["hazard_family"] == hazard)
        ].sort_values("field_id", kind="mergesort")
        for row in local_fields.to_dict("records"):
            state = str(row["event_state"])
            is_core = state in CORE_STATES
            is_watch = state in WATCH_STATES
            # Spatial adjacency alone cannot attribute an old recovering episode
            # to this exposure. Only a fresh contemporaneous decline may seed an
            # impact-lag membership; later evolution follows that exact episode.
            is_impact = _truth(row.get("impact_active", False))
            if not (is_core or is_watch or is_impact):
                continue
            coordinate = (int(row["grid_x"]), int(row["grid_y"]))
            candidates: list[tuple[int, float, str, dict[str, Any]]] = []
            for meta in component_meta:
                distances = [max(abs(coordinate[0] - x), abs(coordinate[1] - y)) for x, y in meta["coordinates"]]
                distance = min(distances)
                allowed = distance == 0 if is_core else distance <= frontier
                if allowed:
                    candidates.append((distance, -meta["active_hint"], meta["component_id"], meta))
            if not candidates:
                continue
            meta = min(candidates, key=lambda item: item[:3])[3]
            # A fresh crop-response decline is explicit impact evidence, not a
            # generic watch row. Preserve pressure-core precedence, then impact,
            # then watch so an impact-only crop can causally seed this week.
            role = (
                "pressure_core" if is_core
                else "impact_lag" if is_impact
                else "watch_frontier"
            )
            membership_records.append(
                {
                    "timeline_bucket": week,
                    "hazard_family": hazard,
                    "component_id": meta["component_id"],
                    "field_id": str(row["field_id"]),
                    "crop_instance_id": (
                        None
                        if pd.isna(row.get("crop_instance_id"))
                        else str(row.get("crop_instance_id"))
                    ),
                    "episode_id": None if pd.isna(row.get("episode_id")) else str(row.get("episode_id")),
                    "membership_role": role,
                    "event_state": state,
                    "response_class": str(
                        row.get("response_class") or "no_new_event_response"
                    ),
                    "fresh_response_evidence": _truth(
                        row.get("fresh_response_evidence", False)
                    ),
                    "evaluable": _truth(row.get("evaluable", True), default=True),
                    "is_data_gap": _truth(row.get("is_data_gap", False)),
                    "stage_bucket": str(row.get("stage_bucket") or "unknown"),
                    "crop_name": str(row.get("crop_name") or "unknown"),
                    "grid_id": str(row["grid_id"]),
                    "knowledge_time": row.get("knowledge_time"),
                }
            )
        local_memberships = pd.DataFrame(membership_records, columns=MEMBERSHIP_COLUMNS)
        local_memberships = local_memberships[
            (local_memberships["timeline_bucket"] == week)
            & (local_memberships["hazard_family"] == hazard)
        ]
        for meta in component_meta:
            members = local_memberships[local_memberships["component_id"] == meta["component_id"]]
            core = members[members["membership_role"] == "pressure_core"]
            if core.empty:
                raise ValueError(
                    f"significant exposure component {meta['component_id']} has no ACTIVE/SEVERE fields"
                )
            selected_cells = meta["cells"]
            cell_x = pd.to_numeric(selected_cells.get("grid_center_x_km", (selected_cells["grid_x"] + 0.5) * size), errors="coerce")
            cell_y = pd.to_numeric(selected_cells.get("grid_center_y_km", (selected_cells["grid_y"] + 0.5) * size), errors="coerce")
            center_x, center_y = float(cell_x.mean()), float(cell_y.mean())
            if "grid_center_lon" in selected_cells:
                center_lon = float(
                    pd.to_numeric(selected_cells["grid_center_lon"], errors="coerce").mean()
                )
            elif {"min_lon", "max_lon"}.issubset(selected_cells.columns):
                center_lon = float(
                    (
                        pd.to_numeric(selected_cells["min_lon"], errors="coerce")
                        + pd.to_numeric(selected_cells["max_lon"], errors="coerce")
                    ).mean()
                    / 2.0
                )
            else:
                center_lon = math.nan
            if "grid_center_lat" in selected_cells:
                center_lat = float(
                    pd.to_numeric(selected_cells["grid_center_lat"], errors="coerce").mean()
                )
            elif {"min_lat", "max_lat"}.issubset(selected_cells.columns):
                center_lat = float(
                    (
                        pd.to_numeric(selected_cells["min_lat"], errors="coerce")
                        + pd.to_numeric(selected_cells["max_lat"], errors="coerce")
                    ).mean()
                    / 2.0
                )
            else:
                center_lat = math.nan
            component_records.append(
                {
                    "timeline_bucket": week,
                    "hazard_family": hazard,
                    "component_id": meta["component_id"],
                    "cell_ids_json": json.dumps(sorted(meta["grid_ids"]), separators=(",", ":")),
                    "core_cell_count": len(meta["coordinates"]),
                    "active_field_count": int(core["field_id"].nunique()),
                    "severe_field_count": int(core[core["event_state"] == "SEVERE"]["field_id"].nunique()),
                    "watch_frontier_field_count": int(members[members["membership_role"] == "watch_frontier"]["field_id"].nunique()),
                    "impact_field_count": int(members[members["membership_role"] == "impact_lag"]["field_id"].nunique()),
                    "monitored_field_count": int(pd.to_numeric(selected_cells["monitored_field_count"], errors="coerce").fillna(0).sum()),
                    "center_x_km": center_x,
                    "center_y_km": center_y,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "footprint_area_km2": len(meta["coordinates"]) * size * size,
                    "max_z_score": (
                        float(pd.to_numeric(selected_cells["z_score"], errors="coerce").max())
                        if "z_score" in selected_cells
                        else math.nan
                    ),
                    "stage_distribution": _distribution(core["stage_bucket"]),
                    "crop_distribution": _distribution(core["crop_name"]),
                }
            )
    components = pd.DataFrame(component_records, columns=COMPONENT_COLUMNS).sort_values(
        ["timeline_bucket", "hazard_family", "component_id"], kind="mergesort"
    ).reset_index(drop=True)
    memberships = pd.DataFrame(membership_records, columns=MEMBERSHIP_COLUMNS).sort_values(
        ["timeline_bucket", "hazard_family", "component_id", "membership_role", "field_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if memberships.duplicated(["timeline_bucket", "hazard_family", "field_id"]).any():
        raise RuntimeError("component attachment produced duplicate field memberships")
    return ComponentBuildResult(components, memberships)


def _json_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("cell_ids_json must contain a JSON list")
    return {str(item) for item in parsed}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    b = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _member_sets(memberships: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    output: dict[str, dict[str, set[str]]] = {}
    if memberships.empty:
        return output
    _require_columns(memberships, ("component_id", "field_id", "membership_role"), "component memberships")
    for component_id, group in memberships.groupby("component_id", sort=True):
        core = group[group["membership_role"].astype(str) == "pressure_core"]
        episodes = set()
        if "episode_id" in core:
            episodes = {str(value) for value in core["episode_id"].dropna() if str(value)}
        stages: dict[str, float] = {}
        stage_column = "stage_bucket" if "stage_bucket" in core else "stage_family"
        if stage_column in core and len(core):
            counts = core[stage_column].fillna("unknown").astype(str).value_counts()
            stages = {str(key): float(value / len(core)) for key, value in counts.items()}
        output[str(component_id)] = {
            "fields": set(core["field_id"].astype(str)),
            "episodes": episodes,
            "stages": stages,  # type: ignore[dict-item]
        }
    return output


def score_temporal_candidates(
    previous_components: pd.DataFrame,
    current_components: pd.DataFrame,
    previous_memberships: pd.DataFrame,
    current_memberships: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Score causal same-hazard links without using crop or stage identity."""
    columns = [
        "previous_component_id", "current_component_id", "previous_exposure_id",
        "hazard_family", "gap_weeks", "episode_jaccard", "cell_jaccard",
        "field_jaccard", "centroid_distance_km", "distance_similarity",
        "stage_cosine", "shared_episode_count", "shared_field_count", "score",
    ]
    if previous_components.empty or current_components.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "timeline_bucket", "hazard_family", "component_id", "cell_ids_json",
        "center_x_km", "center_y_km",
    }
    _require_columns(previous_components, required, "previous components")
    _require_columns(current_components, required, "current components")
    if previous_components["component_id"].astype(str).duplicated().any():
        raise ValueError("previous component IDs must be unique")
    if current_components["component_id"].astype(str).duplicated().any():
        raise ValueError("current component IDs must be unique")
    previous_sets = _member_sets(previous_memberships)
    current_sets = _member_sets(current_memberships)
    records: list[dict[str, Any]] = []
    max_gap = int(_cfg(config, "max_gap_weeks"))
    maximum_distance = float(_cfg(config, "max_centroid_distance_km"))
    identity_weights = {
        "episode": float(_cfg(config, "episode_jaccard_weight")),
        "cell": float(_cfg(config, "cell_jaccard_weight")),
        "field": float(_cfg(config, "field_jaccard_weight")),
        "distance": float(_cfg(config, "distance_weight")),
    }
    identity_weight_total = sum(identity_weights.values())
    if (
        not all(math.isfinite(value) and value >= 0 for value in identity_weights.values())
        or identity_weight_total <= 0
    ):
        raise ValueError("physical temporal-link weights must be finite, nonnegative, and nonzero")
    for previous in previous_components.sort_values("component_id", kind="mergesort").to_dict("records"):
        previous_id = str(previous["component_id"])
        previous_exposure = str(previous.get("exposure_id") or stable_exposure_id(
            str(previous["hazard_family"]), previous["timeline_bucket"], _json_set(previous["cell_ids_json"]), config
        ))
        for current in current_components.sort_values("component_id", kind="mergesort").to_dict("records"):
            if _hazard(previous["hazard_family"]) != _hazard(current["hazard_family"]):
                continue
            gap_days = (pd.Timestamp(current["timeline_bucket"]) - pd.Timestamp(previous["timeline_bucket"])).days
            if gap_days <= 0 or gap_days % 7 or gap_days // 7 > max_gap:
                continue
            gap_weeks = gap_days // 7
            current_id = str(current["component_id"])
            left = previous_sets.get(previous_id, {"fields": set(), "episodes": set(), "stages": {}})
            right = current_sets.get(current_id, {"fields": set(), "episodes": set(), "stages": {}})
            left_fields, right_fields = set(left["fields"]), set(right["fields"])
            left_episodes, right_episodes = set(left["episodes"]), set(right["episodes"])
            left_cells, right_cells = _json_set(previous["cell_ids_json"]), _json_set(current["cell_ids_json"])
            distance = math.hypot(
                float(previous["center_x_km"]) - float(current["center_x_km"]),
                float(previous["center_y_km"]) - float(current["center_y_km"]),
            )
            shared_fields = len(left_fields & right_fields)
            shared_episodes = len(left_episodes & right_episodes)
            cell_overlap = len(left_cells & right_cells)
            if not (shared_fields or shared_episodes or cell_overlap or distance <= maximum_distance):
                continue
            distance_similarity = math.exp(-distance / max(maximum_distance, 1e-9))
            episode_jaccard = _jaccard(left_episodes, right_episodes)
            cell_jaccard = _jaccard(left_cells, right_cells)
            field_jaccard = _jaccard(left_fields, right_fields)
            stage_cosine = _cosine(left["stages"], right["stages"])  # type: ignore[arg-type]
            # Stage is retained as a diagnostic annotation only.  Normalizing
            # the four physical/episode weights preserves the [0, 1] score
            # scale after excluding stage context from exposure identity.
            score = (
                (
                    identity_weights["episode"] * episode_jaccard
                    + identity_weights["cell"] * cell_jaccard
                    + identity_weights["field"] * field_jaccard
                    + identity_weights["distance"] * distance_similarity
                )
                / identity_weight_total
                - float(_cfg(config, "gap_penalty")) * (gap_weeks - 1)
            )
            records.append(
                {
                    "previous_component_id": previous_id,
                    "current_component_id": current_id,
                    "previous_exposure_id": previous_exposure,
                    "hazard_family": _hazard(current["hazard_family"]),
                    "gap_weeks": gap_weeks,
                    "episode_jaccard": episode_jaccard,
                    "cell_jaccard": cell_jaccard,
                    "field_jaccard": field_jaccard,
                    "centroid_distance_km": distance,
                    "distance_similarity": distance_similarity,
                    "stage_cosine": stage_cosine,
                    "shared_episode_count": shared_episodes,
                    "shared_field_count": shared_fields,
                    "score": score,
                }
            )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["score", "shared_episode_count", "shared_field_count", "previous_exposure_id", "previous_component_id", "current_component_id"],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _maximum_weight_primary_links(
    scores: pd.DataFrame,
    previous_component_ids: Iterable[str],
    current_component_ids: Iterable[str],
    continuation_threshold: float,
) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, str], float]]:
    """Return a deterministic globally optimal one-to-one continuation.

    Each previous component receives a private zero-weight dummy option, so no
    below-threshold or negative link is ever forced.  Real rows and columns are
    sorted before optimization; consequently the solver's deterministic tie
    handling is independent of input frame order.
    """
    previous_ids = sorted(set(str(value) for value in previous_component_ids))
    current_ids = sorted(set(str(value) for value in current_component_ids))
    if not previous_ids or not current_ids or scores.empty:
        return {}, {}, {}

    eligible: dict[tuple[str, str], float] = {}
    for row in scores.to_dict("records"):
        score = float(row["score"])
        if score >= continuation_threshold:
            eligible[
                (str(row["previous_component_id"]), str(row["current_component_id"]))
            ] = score
    if not eligible:
        return {}, {}, {}

    previous_index = {component_id: index for index, component_id in enumerate(previous_ids)}
    current_index = {component_id: index for index, component_id in enumerate(current_ids)}
    scale = max(1.0, *(abs(value) for value in eligible.values()))
    invalid_benefit = -scale - 1.0
    # Real current columns are followed by one dummy column per previous row.
    benefits = np.full(
        (len(previous_ids), len(current_ids) + len(previous_ids)),
        invalid_benefit,
        dtype=float,
    )
    for row_index in range(len(previous_ids)):
        benefits[row_index, len(current_ids) + row_index] = 0.0
    for (previous_id, current_id), score in eligible.items():
        benefits[previous_index[previous_id], current_index[current_id]] = score

    row_indices, column_indices = linear_sum_assignment(benefits, maximize=True)
    primary_by_previous: dict[str, str] = {}
    primary_by_current: dict[str, str] = {}
    primary_score: dict[tuple[str, str], float] = {}
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        if column_index >= len(current_ids):
            continue
        previous_id = previous_ids[row_index]
        current_id = current_ids[column_index]
        pair = (previous_id, current_id)
        if pair not in eligible:
            continue
        score = eligible[pair]
        # An unmatched dummy is preferable to a negative eligible edge.
        if score < 0:
            continue
        primary_by_previous[previous_id] = current_id
        primary_by_current[current_id] = previous_id
        primary_score[pair] = score
    return primary_by_previous, primary_by_current, primary_score


def link_weekly_components(
    previous_components: pd.DataFrame,
    current_components: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> TrackingResult:
    """Track crop-independent exposures and record split/merge lineage."""
    assignment_columns = [
        "timeline_bucket", "hazard_family", "component_id", "exposure_id",
        "assignment_kind", "previous_component_id", "link_score",
    ]
    lineage_columns = [
        "parent_exposure_id", "child_exposure_id", "previous_component_id",
        "current_component_id", "lineage_type", "score",
    ]
    update_columns = ["exposure_id", "previous_component_id", "update_status", "target_exposure_id"]
    if current_components.empty:
        updates = []
        for row in previous_components.to_dict("records"):
            exposure = str(row.get("exposure_id") or stable_exposure_id(
                str(row["hazard_family"]), row["timeline_bucket"],
                _json_set(row["cell_ids_json"]), config,
            ))
            updates.append({"exposure_id": exposure, "previous_component_id": str(row["component_id"]), "update_status": "unmatched", "target_exposure_id": None})
        return TrackingResult(
            pd.DataFrame(columns=assignment_columns), pd.DataFrame(columns=lineage_columns),
            pd.DataFrame(updates, columns=update_columns),
        )
    _require_columns(current_components, ("timeline_bucket", "hazard_family", "component_id", "cell_ids_json"), "current components")
    if current_components["component_id"].astype(str).duplicated().any():
        raise ValueError("current component IDs must be unique")
    previous_exposure: dict[str, str] = {}
    for row in previous_components.to_dict("records"):
        component_id = str(row["component_id"])
        if component_id in previous_exposure:
            raise ValueError("previous component IDs must be unique")
        previous_exposure[component_id] = str(row.get("exposure_id") or stable_exposure_id(
            str(row["hazard_family"]), row["timeline_bucket"], _json_set(row["cell_ids_json"]), config
        ))
    if len(set(previous_exposure.values())) != len(previous_exposure):
        raise ValueError("one previous weekly component is required per exposure_id")
    valid_previous = set(previous_exposure)
    valid_current = set(current_components["component_id"].astype(str))
    scores = candidate_scores.copy()
    if not scores.empty:
        _require_columns(
            scores,
            (
                "previous_component_id", "current_component_id", "score",
                "episode_jaccard", "cell_jaccard", "field_jaccard",
            ),
            "candidate scores",
        )
        scores["previous_component_id"] = scores["previous_component_id"].astype(str)
        scores["current_component_id"] = scores["current_component_id"].astype(str)
        if scores.duplicated(["previous_component_id", "current_component_id"]).any():
            raise ValueError("candidate scores contain duplicate component pairs")
        numeric_scores = pd.to_numeric(scores["score"], errors="coerce")
        if not np.isfinite(numeric_scores.to_numpy(dtype=float)).all():
            raise ValueError("candidate scores must be finite")
        scores["score"] = numeric_scores
        if not set(scores["previous_component_id"]) <= valid_previous or not set(scores["current_component_id"]) <= valid_current:
            raise ValueError("candidate scores reference unknown components")
        for name in ("shared_episode_count", "shared_field_count"):
            if name not in scores:
                scores[name] = 0
        scores["previous_exposure_id"] = scores["previous_component_id"].map(previous_exposure)
        scores = scores.sort_values(
            ["score", "shared_episode_count", "shared_field_count", "previous_exposure_id", "previous_component_id", "current_component_id"],
            ascending=[False, False, False, True, True, True], kind="mergesort",
        )
    continuation = float(_cfg(config, "continuation_threshold"))
    primary_by_previous, primary_by_current, primary_score = (
        _maximum_weight_primary_links(
            scores, valid_previous, valid_current, continuation
        )
    )

    assignments: list[dict[str, Any]] = []
    current_exposure: dict[str, str] = {}
    for row in current_components.sort_values("component_id", kind="mergesort").to_dict("records"):
        current_id = str(row["component_id"])
        parent = primary_by_current.get(current_id)
        if parent is not None:
            exposure = previous_exposure[parent]
            kind = "continued"
            score = primary_score[(parent, current_id)]
        else:
            exposure = stable_exposure_id(
                str(row["hazard_family"]), row["timeline_bucket"], _json_set(row["cell_ids_json"]), config
            )
            kind, score = "new", math.nan
        current_exposure[current_id] = exposure
        assignments.append(
            {
                "timeline_bucket": _bucket(row["timeline_bucket"]),
                "hazard_family": _hazard(row["hazard_family"]),
                "component_id": current_id,
                "exposure_id": exposure,
                "assignment_kind": kind,
                "previous_component_id": parent,
                "link_score": score,
            }
        )

    # Primary continuation is represented by the unchanged exposure_id on the
    # weekly assignment.  Persisting it as a parent->child self-edge would make
    # the split/merge lineage graph cyclic.
    lineage: list[dict[str, Any]] = []
    lineage_threshold = float(_cfg(config, "lineage_threshold"))
    minimum_lineage_jaccard = float(_cfg(config, "minimum_lineage_jaccard"))
    for row in scores.to_dict("records"):
        previous_id, current_id, score = str(row["previous_component_id"]), str(row["current_component_id"]), float(row["score"])
        overlap = max(
            float(row.get("episode_jaccard", 0.0) or 0.0),
            float(row.get("cell_jaccard", 0.0) or 0.0),
            float(row.get("field_jaccard", 0.0) or 0.0),
        )
        if (
            score < lineage_threshold
            or overlap < minimum_lineage_jaccard
            or primary_by_previous.get(previous_id) == current_id
        ):
            continue
        previous_continues = previous_id in primary_by_previous
        current_continues = current_id in primary_by_current
        if current_continues and not previous_continues:
            relation = "merge"
        elif previous_continues and not current_continues:
            relation = "split"
        else:
            relation = "related_unmatched"
        lineage.append(
            {
                "parent_exposure_id": previous_exposure[previous_id],
                "child_exposure_id": current_exposure[current_id],
                "previous_component_id": previous_id,
                "current_component_id": current_id,
                "lineage_type": relation,
                "score": score,
            }
        )

    updates: list[dict[str, Any]] = []
    lineage_frame = pd.DataFrame(lineage, columns=lineage_columns)
    for previous_id, exposure in sorted(previous_exposure.items()):
        if previous_id in primary_by_previous:
            status, target = "continued", exposure
        else:
            merge_rows = lineage_frame[
                (lineage_frame["previous_component_id"] == previous_id)
                & (lineage_frame["lineage_type"] == "merge")
            ]
            if len(merge_rows):
                best = merge_rows.sort_values(["score", "child_exposure_id"], ascending=[False, True], kind="mergesort").iloc[0]
                status, target = "merged", str(best["child_exposure_id"])
            else:
                status, target = "unmatched", None
        updates.append(
            {"exposure_id": exposure, "previous_component_id": previous_id, "update_status": status, "target_exposure_id": target}
        )
    assignment_frame = pd.DataFrame(assignments, columns=assignment_columns).sort_values(
        ["component_id"], kind="mergesort"
    ).reset_index(drop=True)
    lineage_frame = lineage_frame.sort_values(
        ["lineage_type", "parent_exposure_id", "child_exposure_id", "previous_component_id", "current_component_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    update_frame = pd.DataFrame(updates, columns=update_columns).sort_values(
        ["exposure_id", "previous_component_id"], kind="mergesort"
    ).reset_index(drop=True)
    return TrackingResult(assignment_frame, lineage_frame, update_frame)


def apply_tracking_assignments(
    components: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Attach one crop-independent exposure ID for the next causal step."""
    _require_columns(components, ("component_id",), "components")
    _require_columns(assignments, ("component_id", "exposure_id"), "assignments")
    if assignments["component_id"].astype(str).duplicated().any():
        raise ValueError("assignments must be unique by component_id")
    output = components.drop(columns=["exposure_id"], errors="ignore").merge(
        assignments[["component_id", "exposure_id"]], on="component_id", how="left", validate="one_to_one"
    )
    if output["exposure_id"].isna().any():
        raise ValueError("every component requires an exposure assignment")
    return output


def is_terminal_incident_state(value: Any) -> bool:
    return str(value or "").upper() in TERMINAL_INCIDENT_STATES


def initialize_incident_lifecycle(
    incident_id: str,
    hazard_family: str,
    observation: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a lifecycle and consume the first causal weekly observation."""
    if not incident_id:
        raise ValueError("incident_id is required")
    state = {
        "incident_id": str(incident_id),
        "hazard_family": _hazard(hazard_family),
        "incident_state": "CANDIDATE",
        "first_evidence_week": None,
        "confirmed_week": None,
        "pressure_off_week": None,
        "recovered_week": None,
        "closed_week": None,
        "merged_into_incident_id": None,
        "support_streak": 0,
        "candidate_absence_streak": 0,
        "quiet_streak": 0,
        "recovery_streak": 0,
        "unresolved_streak": 0,
        "relapse_count": 0,
        "data_gap_count": 0,
        "coverage_gap_streak": 0,
        "last_timeline_bucket": None,
    }
    return advance_incident_lifecycle(state, observation, config)


def advance_incident_lifecycle(
    lifecycle: Mapping[str, Any],
    observation: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance a causal incident state; low coverage freezes all evidence clocks."""
    output = dict(lifecycle)
    state = str(output.get("incident_state") or "CANDIDATE").upper()
    if "DEAD" in state:
        raise ValueError("incident lifecycle must never contain a DEAD state")
    week = _bucket(observation.get("timeline_bucket"))
    previous_week = output.get("last_timeline_bucket")
    if previous_week is not None and pd.Timestamp(week) <= pd.Timestamp(previous_week):
        raise ValueError("lifecycle observations must be strictly chronological")
    output["last_timeline_bucket"] = week
    if is_terminal_incident_state(state):
        return output
    if observation.get("merged_into_incident_id"):
        output.update(
            {
                "incident_state": "MERGED_INTO",
                "merged_into_incident_id": str(observation["merged_into_incident_id"]),
                "closed_week": week,
            }
        )
        return output
    if _truth(observation.get("season_boundary", False)):
        output.update({"incident_state": "CLOSED_SEASON_CENSORED", "closed_week": week})
        return output
    if _truth(observation.get("data_censored", False)):
        # Boundary censoring is a provisional closure of this same inadequate
        # observation.  Record its cumulative coverage history now so a later
        # append can reopen the row without rewriting audit counters.
        output["data_gap_count"] = int(output.get("data_gap_count", 0)) + 1
        output["coverage_gap_streak"] = int(
            output.get("coverage_gap_streak", 0)
        ) + 1
        output.update({"incident_state": "CLOSED_DATA_CENSORED", "closed_week": week})
        return output
    if not _truth(observation.get("adequate_coverage", True), default=True):
        output["data_gap_count"] = int(output.get("data_gap_count", 0)) + 1
        output["coverage_gap_streak"] = int(
            output.get("coverage_gap_streak", 0)
        ) + 1
        return output

    # ``data_gap_count`` is a cumulative audit metric.  Only a consecutive
    # coverage gap may trigger boundary censoring, so keep a separate streak
    # and reset it as soon as an adequately observed week arrives.
    output["coverage_gap_streak"] = 0

    present = _truth(observation.get("component_present", False))
    severe = int(observation.get("severe_field_count", 0) or 0)
    fresh_decline = int(
        observation.get(
            "fresh_decline_field_count",
            observation.get("fresh_response_field_count", 0),
        )
        or 0
    )
    fresh_recovery = int(observation.get("fresh_recovery_field_count", 0) or 0)
    impact = int(observation.get("impact_field_count", 0) or 0)
    recovered_impact = int(observation.get("recovered_impact_field_count", 0) or 0)
    recovery_evidence = _truth(
        observation.get(
            "recovery_evidence", fresh_recovery > 0 or recovered_impact > 0
        )
    )
    story_evidence_present = _truth(
        observation.get(
            "story_evidence_present",
            present or impact > 0 or fresh_decline > 0,
        )
    )
    confirmation_support_present = _truth(
        observation.get("confirmation_support_present", story_evidence_present)
    )

    if state == "CANDIDATE":
        if confirmation_support_present:
            output["first_evidence_week"] = output.get("first_evidence_week") or week
            output["support_streak"] = int(output.get("support_streak", 0)) + 1
            output["candidate_absence_streak"] = 0
            severe_confirmed = (
                severe >= int(_cfg(config, "severe_confirmation_min_fields"))
                and fresh_decline >= int(_cfg(config, "severe_confirmation_min_fresh_response_fields"))
            )
            if severe_confirmed or int(output["support_streak"]) >= int(_cfg(config, "confirmation_observed_weeks")):
                output["incident_state"] = "CONFIRMED"
                output["confirmed_week"] = week
        elif story_evidence_present:
            # Carried unresolved evidence keeps the candidate observable but
            # is not a second independent observation and cannot confirm it.
            # It also cannot keep an unconfirmed candidate alive forever.
            output["support_streak"] = 0
            output["candidate_absence_streak"] = int(
                output.get("candidate_absence_streak", 0)
            ) + 1
            if int(output["candidate_absence_streak"]) >= int(
                _cfg(config, "candidate_expiry_observed_weeks")
            ):
                output.update(
                    {
                        "incident_state": "CLOSED_CANDIDATE_EXPIRED",
                        "closed_week": week,
                    }
                )
        else:
            output["support_streak"] = 0
            output["candidate_absence_streak"] = int(output.get("candidate_absence_streak", 0)) + 1
            if int(output["candidate_absence_streak"]) >= int(_cfg(config, "candidate_expiry_observed_weeks")):
                output.update({"incident_state": "CLOSED_CANDIDATE_EXPIRED", "closed_week": week})
        return output

    if state in {"CONFIRMED", "RELAPSED"}:
        if present:
            output["incident_state"] = "ACTIVE"
            output["quiet_streak"] = 0
            return output
        output["incident_state"] = "PRESSURE_QUIET"
        output["pressure_off_week"] = output.get("pressure_off_week") or week
        output["quiet_streak"] = 1
        state = "PRESSURE_QUIET"

    if state == "ACTIVE":
        if present:
            output["quiet_streak"] = 0
            return output
        output["incident_state"] = "PRESSURE_QUIET"
        output["pressure_off_week"] = output.get("pressure_off_week") or week
        output["quiet_streak"] = 1
        state = "PRESSURE_QUIET"

    if state == "PRESSURE_QUIET":
        if present:
            output["incident_state"] = "RELAPSED"
            output["pressure_off_week"] = None
            output["quiet_streak"] = 0
            output["relapse_count"] = int(output.get("relapse_count", 0)) + 1
            return output
        if int(output.get("quiet_streak", 0)) < 1:
            output["quiet_streak"] = 1
        elif output.get("pressure_off_week") != week:
            output["quiet_streak"] = int(output["quiet_streak"]) + 1
        if int(output["quiet_streak"]) >= int(_cfg(config, "quiet_observed_weeks")):
            if impact > 0:
                output["incident_state"] = "RECOVERING"
                output["unresolved_streak"] = 1
            elif recovery_evidence:
                output.update(
                    {"incident_state": "CLOSED_RECOVERED", "recovered_week": week, "closed_week": week}
                )
            else:
                output.update(
                    {"incident_state": "CLOSED_PRESSURE_QUIET_UNCONFIRMED", "closed_week": week}
                )
        return output

    if state == "RECOVERING":
        if present:
            output["incident_state"] = "RELAPSED"
            output["pressure_off_week"] = None
            output["quiet_streak"] = 0
            output["recovery_streak"] = 0
            output["unresolved_streak"] = 0
            output["relapse_count"] = int(output.get("relapse_count", 0)) + 1
            return output
        if recovery_evidence and impact == 0:
            output["recovery_streak"] = int(output.get("recovery_streak", 0)) + 1
            if int(output["recovery_streak"]) >= int(_cfg(config, "recovery_observed_weeks")):
                output.update(
                    {"incident_state": "CLOSED_RECOVERED", "recovered_week": week, "closed_week": week}
                )
        elif impact > 0:
            output["recovery_streak"] = 0
            output["unresolved_streak"] = int(output.get("unresolved_streak", 0)) + 1
            if int(output["unresolved_streak"]) >= int(_cfg(config, "maximum_recovery_observed_weeks")):
                output.update({"incident_state": "CLOSED_RESPONSE_UNRESOLVED", "closed_week": week})
        else:
            output.update({"incident_state": "CLOSED_RESPONSE_UNRESOLVED", "closed_week": week})
        return output

    raise ValueError(f"unknown non-terminal incident state: {state}")


__all__ = [
    "ComponentBuildResult",
    "TrackingResult",
    "advance_incident_lifecycle",
    "apply_tracking_assignments",
    "assign_metric_grid",
    "benjamini_hochberg",
    "build_crop_incident_assignments",
    "build_weekly_components",
    "initialize_incident_lifecycle",
    "is_terminal_incident_state",
    "link_weekly_components",
    "mark_significant_cells",
    "normal_tail_p_value",
    "score_temporal_candidates",
    "stable_component_id",
    "stable_crop_incident_id",
    "stable_exposure_id",
]
