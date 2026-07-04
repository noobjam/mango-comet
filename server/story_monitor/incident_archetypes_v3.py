"""Deterministic feature and prototype primitives for V3 incident archetypes.

This module deliberately does not discover clusters.  It turns completed
crop-impact stories into leakage-safe numeric features and supplies the frozen
prefix machinery needed after a reviewed discovery model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "completed-incident-features-v3/1"
PREFIX_SCHEMA_VERSION = "causal-incident-prefix-features-v3/1"
PREFIX_HORIZONS = (1, 2, 4, 8)
STAGE_BUCKETS = (
    "emergence", "vegetative", "flowering", "fruiting_or_grain_fill",
    "maturity_or_harvest", "off_season", "unknown",
)
TERMINAL_STATES = frozenset(
    {
        "CLOSED_RECOVERED",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED",
        "CLOSED_RESPONSE_UNRESOLVED",
        "CLOSED_SEASON_CENSORED",
        "CLOSED_SEASON_BOUNDARY",
        "CLOSED_DATA_CENSORED",
        "CLOSED_WATCH_QUIET",
        "MERGED_INTO",
    }
)
OUTCOME_OBSERVED_TERMINAL_STATES = frozenset(
    {
        "CLOSED_RECOVERED",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED",
        "CLOSED_RESPONSE_UNRESOLVED",
        "CLOSED_WATCH_QUIET",
    }
)
CENSORED_TERMINAL_STATES = TERMINAL_STATES - OUTCOME_OBSERVED_TERMINAL_STATES

_STAGE_FEATURES = tuple(
    f"{moment}_stage_{stage}_fraction"
    for moment in ("onset", "peak")
    for stage in STAGE_BUCKETS
)
MODEL_FEATURE_COLUMNS = (
    "observed_week_count",
    "pressure_duration_fraction",
    "recovery_lag_fraction",
    "recovery_observed",
    "peak_timing_fraction",
    "peak_affected_rate",
    "cumulative_affected_rate",
    "severe_fraction",
    "early_footprint_growth",
    "maximum_area_km2",
    "footprint_area_missing_fraction",
    "mean_retention_fraction",
    "retention_missing",
    "recovery_member_fraction",
    "unresolved_member_fraction",
    "data_gap_fraction",
    "relapse_count",
    "split_count",
    "merge_count",
    "stage_entropy",
    *_STAGE_FEATURES,
    "hazard_intensity_mean",
    "hazard_intensity_peak",
    "hazard_intensity_slope",
    "hazard_intensity_missing_fraction",
)
FEATURE_GROUPS = {
    "temporal_shape": (
        "observed_week_count", "pressure_duration_fraction",
        "recovery_lag_fraction", "recovery_observed", "peak_timing_fraction",
    ),
    "crop_impact": (
        "peak_affected_rate", "cumulative_affected_rate", "severe_fraction",
        "recovery_member_fraction", "unresolved_member_fraction",
        "data_gap_fraction",
    ),
    "spatial_evolution": (
        "early_footprint_growth", "maximum_area_km2",
        "footprint_area_missing_fraction", "mean_retention_fraction",
        "retention_missing",
    ),
    "lineage_and_relapse": ("relapse_count", "split_count", "merge_count"),
    "crop_stage": ("stage_entropy", *_STAGE_FEATURES),
    "hazard_intensity": (
        "hazard_intensity_mean", "hazard_intensity_peak",
        "hazard_intensity_slope", "hazard_intensity_missing_fraction",
    ),
}
_FEATURE_GROUP_BY_NAME = {
    name: group for group, names in FEATURE_GROUPS.items() for name in names
}
if set(_FEATURE_GROUP_BY_NAME) != set(MODEL_FEATURE_COLUMNS):
    raise RuntimeError("V3 archetype feature groups must partition every model feature")

WEEKLY_REQUIRED_COLUMNS = (
    "incident_id", "exposure_id", "timeline_bucket", "crop_name",
    "hazard_family", "active_count", "severe_count", "affected_count",
    "monitored_count", "evaluable_count",
)
MEMBERSHIP_REQUIRED_COLUMNS = (
    "incident_id", "timeline_bucket", "crop_instance_id", "field_id",
    "episode_id", "membership_role", "stage_bucket",
)


@dataclass(frozen=True)
class PrefixPrototypeArtifacts:
    """Serializable schema plus wide frozen prototype table."""

    feature_schema: dict[str, Any]
    prototypes: pd.DataFrame


def extract_completed_incident_features(
    incident_weekly_state: pd.DataFrame,
    incident_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly one feature row per terminal crop-impact story."""
    weekly, membership = _prepare_inputs(incident_weekly_state, incident_membership)
    membership_groups = {
        str(incident_id): rows
        for incident_id, rows in membership.groupby("incident_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for _, story in weekly.groupby("incident_id", sort=True):
        story = story.sort_values("timeline_bucket", kind="mergesort")
        if str(story.iloc[-1]["current_state"]).upper() not in TERMINAL_STATES:
            continue
        story_members = membership_groups[str(story.iloc[0]["incident_id"])]
        rows.append(_aggregate_story(story, story_members))
    return _records_frame(rows)


def build_causal_prefix_features(
    incident_weekly_state: pd.DataFrame,
    incident_membership: pd.DataFrame,
    *,
    horizons: Sequence[int] = PREFIX_HORIZONS,
) -> pd.DataFrame:
    """Build first-N-observed-week vectors without reading later story rows."""
    checked_horizons = _validate_horizons(horizons)
    weekly, membership = _prepare_inputs(incident_weekly_state, incident_membership)
    membership_groups = {
        str(incident_id): rows
        for incident_id, rows in membership.groupby("incident_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for incident_id, story in weekly.groupby("incident_id", sort=True):
        story = story.sort_values("timeline_bucket", kind="mergesort").reset_index(drop=True)
        story_members = membership_groups[str(incident_id)]
        for horizon in checked_horizons:
            if len(story) < horizon:
                continue
            prefix = story.iloc[:horizon].copy()
            through = prefix.iloc[-1]["timeline_bucket"]
            prefix_members = story_members[story_members["timeline_bucket"] <= through]
            record = _aggregate_story(prefix, prefix_members)
            record.update(
                {
                    "feature_schema_version": PREFIX_SCHEMA_VERSION,
                    "horizon_weeks": horizon,
                    "prefix_through_week": through.date(),
                    "prefix_current_state": str(prefix.iloc[-1]["current_state"]),
                }
            )
            rows.append(record)
    extra = ("horizon_weeks", "prefix_through_week", "prefix_current_state")
    return _records_frame(rows, extra_metadata=extra)


def temporal_split_completed_stories(
    features: pd.DataFrame,
    cutoff: str | date,
    *,
    start_column: str = "first_evidence_week",
    end_column: str = "last_evidence_week",
) -> pd.DataFrame:
    """Label train/holdout stories and embargo every cutoff-crossing story."""
    _require_columns(features, ("incident_id", start_column, end_column), "story features")
    if features["incident_id"].duplicated().any():
        raise ValueError("temporal split requires one row per incident_id")
    output = features.copy()
    starts = pd.to_datetime(output[start_column], errors="coerce").dt.normalize()
    ends = pd.to_datetime(output[end_column], errors="coerce").dt.normalize()
    if starts.isna().any() or ends.isna().any() or (starts > ends).any():
        raise ValueError("temporal split requires valid ordered story start/end dates")
    boundary = pd.Timestamp(cutoff).normalize()
    output["temporal_split"] = np.where(
        ends <= boundary, "train", np.where(starts > boundary, "holdout", "embargo")
    )
    return output


def fit_robust_feature_schema(
    rows: pd.DataFrame,
    *,
    strata_columns: Sequence[str] = ("crop_name", "hazard_family"),
) -> dict[str, Any]:
    """Fit median/IQR transforms using only the supplied training rows."""
    strata = tuple(strata_columns)
    if not strata:
        raise ValueError("at least one stratification column is required")
    _require_columns(rows, (*strata, *MODEL_FEATURE_COLUMNS), "feature rows")
    if rows.empty:
        raise ValueError("cannot fit a feature schema from zero rows")
    records: list[dict[str, Any]] = []
    grouper: Any = strata[0] if len(strata) == 1 else list(strata)
    for key, frame in rows.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(strata) == 1 else tuple(key)
        specs: list[dict[str, Any]] = []
        for index, name in enumerate(MODEL_FEATURE_COLUMNS):
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale < 1e-9:
                scale = 1.0
            group = _FEATURE_GROUP_BY_NAME[name]
            weight = 1.0 / math.sqrt(len(FEATURE_GROUPS[group]))
            specs.append(
                {
                    "index": index,
                    "name": name,
                    "median": median,
                    "scale": scale,
                    "feature_group": group,
                    "weight": weight,
                }
            )
        records.append(
            {
                "key": {name: _key_value(value) for name, value in zip(strata, keys)},
                "features": specs,
            }
        )
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "strata_columns": list(strata),
        "clip": 5.0,
        "feature_weighting": {
            "method": "equal_l2_energy_per_semantic_family",
            "groups": {
                group: {
                    "features": list(group_features),
                    "per_feature_weight": 1.0 / math.sqrt(len(group_features)),
                }
                for group, group_features in FEATURE_GROUPS.items()
            },
        },
        "strata": records,
    }


def transform_finite_feature_matrix(
    rows: pd.DataFrame,
    feature_schema: dict[str, Any],
) -> np.ndarray:
    """Apply a frozen robust transform, imputing non-finite values by train medians."""
    names = tuple(feature_schema.get("feature_names") or ())
    strata = tuple(feature_schema.get("strata_columns") or ())
    if names != MODEL_FEATURE_COLUMNS or not strata:
        raise ValueError("unsupported or incomplete V3 feature schema")
    _require_columns(rows, (*strata, *MODEL_FEATURE_COLUMNS), "feature rows")
    lookup = {
        tuple(item["key"][name] for name in strata): item["features"]
        for item in feature_schema.get("strata", ())
    }
    matrix = np.empty((len(rows), len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
    clip = float(feature_schema.get("clip", 5.0))
    for position, (_, row) in enumerate(rows.iterrows()):
        key = tuple(_key_value(row[name]) for name in strata)
        specs = lookup.get(key)
        if specs is None:
            raise ValueError(f"feature schema has no fitted stratum for {key}")
        for spec in specs:
            value = pd.to_numeric(pd.Series([row[spec["name"]]]), errors="coerce").iloc[0]
            value = float(value) if pd.notna(value) and np.isfinite(value) else float(spec["median"])
            weight = float(spec.get("weight", math.nan))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("feature schema contains an invalid semantic-family weight")
            matrix[position, int(spec["index"])] = np.float32(
                np.clip(
                    (value - float(spec["median"])) / float(spec["scale"]),
                    -clip,
                    clip,
                )
                * weight
            )
    if not np.isfinite(matrix).all():
        raise ValueError("V3 feature transform produced non-finite values")
    return matrix


def fit_prefix_prototypes(
    prefix_features: pd.DataFrame,
    completed_assignments: pd.DataFrame,
    *,
    model_version: str,
    radius_quantile: float = 0.95,
    minimum_support: int = 2,
) -> PrefixPrototypeArtifacts:
    """Fit deterministic horizon-specific centers/radii from reviewed assignments."""
    if not model_version.strip():
        raise ValueError("model_version is required")
    if not 0.5 <= radius_quantile < 1.0 or minimum_support < 1:
        raise ValueError("invalid prototype radius quantile or minimum support")
    _require_columns(
        prefix_features,
        ("incident_id", "crop_name", "hazard_family", "horizon_weeks", *MODEL_FEATURE_COLUMNS),
        "prefix features",
    )
    if prefix_features.duplicated(["incident_id", "horizon_weeks"]).any():
        raise ValueError("prefix features must be unique by incident_id and horizon")
    if not set(pd.to_numeric(prefix_features["horizon_weeks"], errors="coerce")).issubset(
        PREFIX_HORIZONS
    ):
        raise ValueError(f"prefix horizons must be selected from {PREFIX_HORIZONS}")
    _require_columns(completed_assignments, ("incident_id", "archetype_id"), "assignments")
    if completed_assignments["incident_id"].duplicated().any():
        raise ValueError("completed assignments must be unique by incident_id")
    assigned = completed_assignments[["incident_id", "archetype_id"]].copy()
    assigned = assigned[
        assigned["archetype_id"].notna()
        & ~assigned["archetype_id"].astype(str).isin({"novel", "novel_unassigned", "pending"})
    ]
    training = prefix_features.merge(assigned, on="incident_id", how="inner", validate="many_to_one")
    if training.empty:
        raise ValueError("no reviewed assignments have causal prefix features")
    strata = ("crop_name", "hazard_family", "horizon_weeks")
    schema = fit_robust_feature_schema(training, strata_columns=strata)
    schema.update(
        {
            "schema_version": PREFIX_SCHEMA_VERSION,
            "model_version": model_version,
            "radius_quantile": radius_quantile,
            "minimum_support": minimum_support,
            "supported_horizons": sorted(
                set(int(value) for value in training["horizon_weeks"])
            ),
        }
    )
    matrix = transform_finite_feature_matrix(training, schema)
    vector_columns = [f"f_{index:03d}" for index in range(matrix.shape[1])]
    records: list[dict[str, Any]] = []
    for key, indexes in training.groupby([*strata, "archetype_id"], sort=True).groups.items():
        positions = training.index.get_indexer(list(indexes))
        group_vectors = matrix[positions]
        if len(group_vectors) < minimum_support:
            continue
        center = np.median(group_vectors, axis=0)
        distances = np.linalg.norm(group_vectors - center, axis=1)
        radius = max(float(np.quantile(distances, radius_quantile)), 1e-6)
        record = dict(zip((*strata, "archetype_id"), key))
        record.update(
            {
                "support": len(group_vectors),
                "radius": radius,
                "radius_quantile": radius_quantile,
                "model_version": model_version,
                **dict(zip(vector_columns, center.astype(float))),
            }
        )
        records.append(record)
    prototypes = pd.DataFrame(records)
    if prototypes.empty:
        raise ValueError("no prefix archetype met minimum support")
    prototypes = prototypes.sort_values([*strata, "archetype_id"], kind="mergesort").reset_index(drop=True)
    numeric = prototypes[["radius", *vector_columns]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (prototypes["radius"] <= 0).any():
        raise ValueError("prefix prototypes must be finite with positive radii")
    return PrefixPrototypeArtifacts(schema, prototypes)


def assign_open_set_prefixes(
    prefix_features: pd.DataFrame,
    artifacts: PrefixPrototypeArtifacts,
    *,
    runner_up_margin: float = 0.05,
) -> pd.DataFrame:
    """Assign latest supported prefixes as tentative, novel, or pending."""
    if runner_up_margin < 0:
        raise ValueError("runner_up_margin must be non-negative")
    _require_columns(
        prefix_features,
        ("incident_id", "crop_name", "hazard_family", "horizon_weeks", *MODEL_FEATURE_COLUMNS),
        "prefix features",
    )
    if prefix_features.duplicated(["incident_id", "horizon_weeks"]).any():
        raise ValueError("prefix features must be unique by incident_id and horizon")
    prototypes = artifacts.prototypes
    required_prototype_columns = {
        "crop_name", "hazard_family", "horizon_weeks", "archetype_id", "radius",
        "model_version",
    }
    missing = sorted(required_prototype_columns - set(prototypes.columns))
    if missing:
        raise ValueError("prefix prototypes are missing columns: " + ", ".join(missing))
    if prototypes.duplicated(
        ["crop_name", "hazard_family", "horizon_weeks", "archetype_id"]
    ).any():
        raise ValueError("prefix prototype keys must be unique")
    versions = set(prototypes["model_version"].astype(str))
    if len(versions) != 1 or not next(iter(versions)).strip():
        raise ValueError("prefix prototypes must use one non-empty model_version")
    if str(artifacts.feature_schema.get("model_version") or "") != next(iter(versions)):
        raise ValueError("prefix feature schema and prototypes use different model versions")
    vector_columns = sorted(name for name in prototypes if name.startswith("f_"))
    if len(vector_columns) != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("prototype feature width does not match V3 feature schema")
    prototype_numeric = prototypes[["radius", *vector_columns]].to_numpy(dtype=float)
    if not np.isfinite(prototype_numeric).all() or (prototype_numeric[:, 0] <= 0).any():
        raise ValueError("open-set assignment received invalid prototypes")
    output: list[dict[str, Any]] = []
    for incident_id, candidates in prefix_features.groupby("incident_id", sort=True):
        candidates = candidates.sort_values("horizon_weeks", ascending=False, kind="mergesort")
        selected = None
        local = pd.DataFrame()
        for _, candidate in candidates.iterrows():
            local = prototypes[
                (prototypes["crop_name"].astype(str) == str(candidate["crop_name"]))
                & (prototypes["hazard_family"].astype(str) == str(candidate["hazard_family"]))
                & (prototypes["horizon_weeks"].astype(int) == int(candidate["horizon_weeks"]))
            ]
            if not local.empty:
                selected = candidate
                break
        if selected is None:
            latest = candidates.iloc[0]
            output.append(_pending_assignment(latest, prototypes))
            continue
        row_frame = pd.DataFrame([selected])
        vector = transform_finite_feature_matrix(row_frame, artifacts.feature_schema)[0]
        centers = local[vector_columns].to_numpy(dtype=float)
        radii = local["radius"].to_numpy(dtype=float)
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError("open-set assignment received invalid prototypes")
        distances = np.linalg.norm(centers - vector, axis=1)
        order = np.argsort(distances, kind="stable")
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else None
        best_distance = float(distances[best])
        separation = math.inf if second is None else float(distances[second] - best_distance)
        within = best_distance <= float(radii[best])
        separated = separation >= runner_up_margin
        accepted = within and separated
        status = "tentative" if accepted else "novel"
        reason = (
            "within_radius_and_margin" if accepted
            else "outside_radius" if not within
            else "ambiguous_runner_up"
        )
        output.append(
            {
                "incident_id": incident_id,
                "crop_name": selected["crop_name"],
                "hazard_family": selected["hazard_family"],
                "horizon_weeks": int(selected["horizon_weeks"]),
                "assignment_status": status,
                "archetype_id": local.iloc[best]["archetype_id"] if accepted else None,
                "candidate_archetype_id": local.iloc[best]["archetype_id"],
                "runner_up_archetype_id": (
                    local.iloc[second]["archetype_id"] if second is not None else None
                ),
                "assignment_distance": best_distance,
                "candidate_radius": float(radii[best]),
                "distance_ratio": best_distance / float(radii[best]),
                "runner_up_separation": separation if np.isfinite(separation) else None,
                "assignment_reason": reason,
                "model_version": str(local.iloc[best]["model_version"]),
            }
        )
    result = pd.DataFrame(output)
    if set(result["incident_id"].tolist()) != set(prefix_features["incident_id"].tolist()):
        raise AssertionError("open-set assignment rewrote or dropped an incident_id")
    return result.sort_values("incident_id", kind="mergesort").reset_index(drop=True)


def supported_prefix_horizon(
    observed_week_count: int, horizons: Sequence[int] = PREFIX_HORIZONS
) -> int | None:
    """Return the latest configured horizon supported by a causal story age."""
    if (
        isinstance(observed_week_count, bool)
        or not isinstance(observed_week_count, (int, np.integer))
        or observed_week_count < 0
    ):
        raise ValueError("observed_week_count must be a non-negative integer")
    available = [value for value in _validate_horizons(horizons) if value <= observed_week_count]
    return max(available) if available else None


def _prepare_inputs(
    weekly: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = weekly.copy()
    if "current_state" not in weekly:
        state_alias = next(
            (name for name in ("story_state", "incident_state") if name in weekly), None
        )
        if state_alias is not None:
            weekly["current_state"] = weekly[state_alias]
    _require_columns(weekly, (*WEEKLY_REQUIRED_COLUMNS, "current_state"), "incident weekly state")
    _require_columns(membership, MEMBERSHIP_REQUIRED_COLUMNS, "incident membership")
    for label, frame, id_columns in (
        ("incident weekly state", weekly, ("incident_id", "exposure_id")),
        ("incident membership", membership, ("incident_id", "field_id")),
    ):
        for name in id_columns:
            if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
                raise ValueError(f"{label} contains null or empty {name}")
    weekly["timeline_bucket"] = _dates(weekly["timeline_bucket"], "weekly timeline_bucket")
    membership = membership.copy()
    membership["timeline_bucket"] = _dates(
        membership["timeline_bucket"], "membership timeline_bucket"
    )
    if weekly.duplicated(["incident_id", "timeline_bucket"]).any():
        raise ValueError("incident weekly state must be unique by incident_id and timeline_bucket")
    for name in ("active_count", "severe_count", "affected_count", "monitored_count", "evaluable_count"):
        weekly[name] = pd.to_numeric(weekly[name], errors="coerce")
        if not np.isfinite(weekly[name]).all() or (weekly[name] < 0).any():
            raise ValueError(f"incident weekly state {name} must be finite and non-negative")
    if (weekly["affected_count"] > weekly["monitored_count"]).any():
        raise ValueError("affected_count cannot exceed monitored_count")
    if (weekly["severe_count"] > weekly["affected_count"]).any():
        raise ValueError("severe_count cannot exceed affected_count")
    if (weekly["evaluable_count"] > weekly["monitored_count"]).any():
        raise ValueError("evaluable_count cannot exceed monitored_count")
    known = set(zip(weekly["incident_id"], weekly["timeline_bucket"]))
    member_keys = set(zip(membership["incident_id"], membership["timeline_bucket"]))
    if not member_keys.issubset(known):
        raise ValueError("incident membership contains incident/week rows absent from weekly state")
    missing_members = set(weekly["incident_id"]) - set(membership["incident_id"])
    if missing_members:
        raise ValueError("every incident requires membership lineage")
    membership["stage_bucket"] = membership["stage_bucket"].map(_stage)
    return weekly.sort_values(["incident_id", "timeline_bucket"]), membership


def _aggregate_story(story: pd.DataFrame, members: pd.DataFrame) -> dict[str, Any]:
    story = story.sort_values("timeline_bucket", kind="mergesort").reset_index(drop=True)
    incident_id = story.iloc[0]["incident_id"]
    exposure_id = _single(story, "exposure_id", incident_id)
    crop = _single(story, "crop_name", incident_id)
    hazard = _single(story, "hazard_family", incident_id)
    affected = story["affected_count"].to_numpy(dtype=float)
    monitored = story["monitored_count"].to_numpy(dtype=float)
    evaluable = story["evaluable_count"].to_numpy(dtype=float)
    affected_rate = np.divide(affected, monitored, out=np.zeros_like(affected), where=monitored > 0)
    peak_index = int(np.argmax(affected_rate))
    roles = members["membership_role"].fillna("").astype(str).str.lower()
    responses = (
        members["response_class"].fillna("").astype(str).str.lower()
        if "response_class" in members else pd.Series("", index=members.index)
    )
    event_states = (
        members["event_state"].fillna("").astype(str).str.upper()
        if "event_state" in members else pd.Series("", index=members.index)
    )
    recovery_members = int(
        (roles.isin({"recovering", "recovered"}) | responses.eq("recovery")).sum()
    )
    unresolved_members = int(
        (
            roles.isin({"unresolved", "unresolved_review"})
            | event_states.eq("CLOSED_RESPONSE_UNRESOLVED")
        ).sum()
    )
    recovery_indexes = [
        index for index, value in enumerate(story["current_state"].astype(str).str.upper())
        if value in {"RECOVERING", "CLOSED_RECOVERED"}
    ]
    first_recovery = next((index for index in recovery_indexes if index >= peak_index), None)
    areas = _optional_numeric(story, ("footprint_area_km2", "area_km2"))
    intensities = _optional_numeric(
        story, ("hazard_intensity_mean", "mean_hazard_intensity", "hazard_intensity")
    )
    retention = _retention(story, members)
    onset_stage = _stage_fractions(members, story.iloc[0]["timeline_bucket"])
    peak_stage = _stage_fractions(members, story.iloc[peak_index]["timeline_bucket"])
    all_stages = members["stage_bucket"].value_counts()
    entropy = _entropy([int(all_stages.get(stage, 0)) for stage in STAGE_BUCKETS])
    pressure_weeks = (
        story["active_count"].to_numpy(dtype=float)
        + story["severe_count"].to_numpy(dtype=float)
    ) > 0
    total_affected = max(float(affected.sum()), 1.0)
    total_monitored = max(float(monitored.sum()), 1.0)
    states = story["current_state"].astype(str).str.upper().tolist()
    relapse_transitions = sum(
        state == "RELAPSED" and (index == 0 or states[index - 1] != "RELAPSED")
        for index, state in enumerate(states)
    )
    count = len(story)
    record: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "incident_id": incident_id,
        "exposure_id": exposure_id,
        "crop_name": crop,
        "hazard_family": hazard,
        "stratification_key": f"{str(crop).strip().lower()}::{str(hazard).strip().lower()}",
        "first_evidence_week": story.iloc[0]["timeline_bucket"].date(),
        "last_evidence_week": story.iloc[-1]["timeline_bucket"].date(),
        "final_state": states[-1],
        "observed_week_count": float(count),
        "pressure_duration_fraction": float(np.mean(pressure_weeks)),
        "recovery_lag_fraction": (
            float((first_recovery - peak_index) / max(count - 1, 1))
            if first_recovery is not None else 0.0
        ),
        "recovery_observed": float(first_recovery is not None),
        "peak_timing_fraction": float(peak_index / max(count - 1, 1)),
        "peak_affected_rate": float(np.max(affected_rate)),
        "cumulative_affected_rate": float(np.sum(affected_rate)),
        "severe_fraction": float(story["severe_count"].sum() / total_affected),
        "early_footprint_growth": _early_growth(areas),
        "maximum_area_km2": _finite_max(areas),
        "footprint_area_missing_fraction": float(np.mean(~np.isfinite(areas))),
        "mean_retention_fraction": float(np.mean(retention)) if retention else 0.0,
        "retention_missing": float(not retention),
        "recovery_member_fraction": recovery_members / total_affected,
        "unresolved_member_fraction": unresolved_members / total_affected,
        "data_gap_fraction": float(np.maximum(monitored - evaluable, 0).sum() / total_monitored),
        "relapse_count": float(max(relapse_transitions, _optional_count(story, "relapse_count"))),
        "split_count": float(_optional_count(story, "split_count")),
        "merge_count": float(max(states[-1] == "MERGED_INTO", _optional_count(story, "merge_count"))),
        "stage_entropy": entropy,
        "hazard_intensity_mean": _finite_mean(intensities),
        "hazard_intensity_peak": _finite_max(intensities),
        "hazard_intensity_slope": _finite_slope(intensities),
        "hazard_intensity_missing_fraction": float(np.mean(~np.isfinite(intensities))),
    }
    for moment, values in (("onset", onset_stage), ("peak", peak_stage)):
        record.update(
            {f"{moment}_stage_{stage}_fraction": values[stage] for stage in STAGE_BUCKETS}
        )
    for name in MODEL_FEATURE_COLUMNS:
        value = float(record[name])
        record[name] = value if np.isfinite(value) else 0.0
    return record


def _retention(story: pd.DataFrame, members: pd.DataFrame) -> list[float]:
    member_sets = {
        week: set(group.loc[~group["membership_role"].astype(str).str.lower().eq("data_gap"), "field_id"])
        for week, group in members.groupby("timeline_bucket", sort=True)
    }
    values: list[float] = []
    weeks = story["timeline_bucket"].tolist()
    for previous, current in zip(weeks, weeks[1:]):
        prior = member_sets.get(previous, set())
        if prior:
            values.append(len(prior & member_sets.get(current, set())) / len(prior))
    return values


def _stage_fractions(members: pd.DataFrame, week: pd.Timestamp) -> dict[str, float]:
    counts = members.loc[members["timeline_bucket"] == week, "stage_bucket"].value_counts()
    total = max(int(counts.sum()), 1)
    return {stage: float(counts.get(stage, 0) / total) for stage in STAGE_BUCKETS}


def _entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=float)
    if values.sum() <= 0:
        return 0.0
    probabilities = values[values > 0] / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(STAGE_BUCKETS)))


def _optional_numeric(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    name = next((candidate for candidate in names if candidate in frame), None)
    if name is None:
        return np.full(len(frame), np.nan, dtype=float)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    if "area" in name and np.any(values[np.isfinite(values)] < 0):
        raise ValueError("footprint area cannot be negative")
    return values


def _optional_count(frame: pd.DataFrame, name: str) -> int:
    if name not in frame:
        return 0
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) and np.min(finite) < 0:
        raise ValueError(f"{name} cannot be negative")
    return int(np.max(finite)) if len(finite) else 0


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else 0.0


def _finite_slope(values: np.ndarray) -> float:
    indexes = np.flatnonzero(np.isfinite(values))
    if len(indexes) < 2:
        return 0.0
    return float(np.polyfit(indexes.astype(float), values[indexes], 1)[0])


def _early_growth(values: np.ndarray) -> float:
    indexes = np.flatnonzero(np.isfinite(values))
    if len(indexes) < 2:
        return 0.0
    early = indexes[indexes <= 2]
    end = int(early[-1]) if len(early) > 1 else int(indexes[1])
    start = int(indexes[0])
    return float(np.clip((values[end] - values[start]) / max(abs(values[start]), 1.0), -10, 10))


def _pending_assignment(row: pd.Series, prototypes: pd.DataFrame) -> dict[str, Any]:
    versions = sorted(set(prototypes.get("model_version", pd.Series(dtype=str)).astype(str)))
    return {
        "incident_id": row["incident_id"],
        "crop_name": row["crop_name"],
        "hazard_family": row["hazard_family"],
        "horizon_weeks": int(row["horizon_weeks"]),
        "assignment_status": "pending",
        "archetype_id": None,
        "candidate_archetype_id": None,
        "runner_up_archetype_id": None,
        "assignment_distance": None,
        "candidate_radius": None,
        "distance_ratio": None,
        "runner_up_separation": None,
        "assignment_reason": "no_supported_prefix_prototype",
        "model_version": versions[0] if len(versions) == 1 else None,
    }


def _records_frame(
    rows: list[dict[str, Any]], *, extra_metadata: Sequence[str] = ()
) -> pd.DataFrame:
    metadata = (
        "feature_schema_version", "incident_id", "exposure_id", "crop_name",
        "hazard_family", "stratification_key", "first_evidence_week",
        "last_evidence_week", "final_state", *extra_metadata,
    )
    return pd.DataFrame(rows, columns=[*metadata, *MODEL_FEATURE_COLUMNS])


def _single(frame: pd.DataFrame, name: str, incident_id: Any) -> Any:
    values = frame[name].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"incident {incident_id} must have one invariant {name}")
    return values[0]


def _dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    return parsed


def _stage(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in STAGE_BUCKETS else "unknown"


def _key_value(value: Any) -> str:
    if pd.isna(value):
        return "<null>"
    return str(value)


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(horizons)))
    if not values or any(isinstance(value, bool) or value not in PREFIX_HORIZONS for value in values):
        raise ValueError(f"prefix horizons must be selected from {PREFIX_HORIZONS}")
    return values


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "MEMBERSHIP_REQUIRED_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "PREFIX_HORIZONS",
    "PREFIX_SCHEMA_VERSION",
    "PrefixPrototypeArtifacts",
    "STAGE_BUCKETS",
    "TERMINAL_STATES",
    "WEEKLY_REQUIRED_COLUMNS",
    "assign_open_set_prefixes",
    "build_causal_prefix_features",
    "extract_completed_incident_features",
    "fit_prefix_prototypes",
    "fit_robust_feature_schema",
    "supported_prefix_horizon",
    "temporal_split_completed_stories",
    "transform_finite_feature_matrix",
]
