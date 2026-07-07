"""Frozen policy contract for V3 crop-incident monitoring context.

This policy controls display-oriented stage bucketing, weekly cohort semantics,
and deterministic lane ordering.  It does not diagnose crop loss or causation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


POLICY_SCHEMA_VERSION = "incident-context-policy-v3/3"
CONTROLLED_STAGE_BUCKETS = (
    "emergence",
    "vegetative",
    "flowering",
    "fruiting_or_grain_fill",
    "maturity_or_harvest",
    "off_season",
    "unknown",
)
LINK_WEIGHT_NAMES = frozenset(
    {
        "active_episode_overlap",
        "cell_or_footprint_overlap",
        "recent_member_overlap",
        "centroid_proximity",
        "stage_distribution_similarity",
    }
)
CANONICAL_LANE_ORDER = (
    "SEVERE",
    "ACTIVE",
    "QUIET_PENDING",
    "WATCH",
    "RECOVERING",
    "DATA_GAP",
)
DEFAULT_INCIDENT_POLICY_V3_PATH = (
    Path(__file__).resolve().parent / "policies" / "incident_policy_v3.json"
)


@dataclass(frozen=True)
class StageAlias:
    raw_stage: str
    stage_bucket: str


@dataclass(frozen=True)
class CropStageAlias:
    raw_crop: str
    raw_stage: str
    stage_bucket: str


@dataclass(frozen=True)
class LaneStatePriority:
    event_state: str
    priority: int
    is_open: bool


@dataclass(frozen=True)
class IncidentPolicyV3:
    schema_version: str
    version: str
    calibration_status: str
    warning: str
    week_start: str
    monitored_rule: str
    evaluable_rule: str
    identity_namespace: str
    grid_cell_size_km: float
    grid_origin_lon: float
    grid_origin_lat: float
    reference_latitude_strategy: str
    baseline_prior_strength: float
    minimum_source_field_centroid_coverage: float
    minimum_source_crop_instance_week_centroid_coverage: float
    minimum_known_stage_coverage: float
    minimum_known_stage_coverage_per_supported_crop: float
    minimum_stage_coverage_crop_instance_weeks: int
    minimum_evaluable_fields: int
    minimum_coverage_ratio: float
    minimum_active_fields: int
    severe_override_min_fields: int
    severe_override_min_fresh_response_fields: int
    allow_severe_override: bool
    frontier_distance_cells: int
    fdr_alpha: float
    minimum_link_score: float
    lineage_threshold: float
    minimum_lineage_jaccard: float
    same_hazard_link_required: bool
    max_link_gap_weeks: int
    spatial_scale_km: float
    gap_penalty: float
    confirmation_weeks: int
    candidate_expiry_observed_weeks: int
    quiet_close_weeks: int
    recovery_observed_weeks: int
    recovery_grace_weeks: int
    severe_confirmation_min_fields: int
    severe_confirmation_min_fresh_response_fields: int
    minimum_crop_monitored_instances: int
    minimum_crop_evaluable_instances: int
    maximum_data_gap_weeks: int
    link_weights: tuple[tuple[str, float], ...]
    stage_buckets: tuple[str, ...]
    stage_aliases: tuple[StageAlias, ...]
    crop_stage_aliases: tuple[CropStageAlias, ...]
    lane_state_priorities: tuple[LaneStatePriority, ...]
    source_path: Path
    source_sha256: str

    def stage_bucket_for(self, raw_stage: Any, crop_name: Any = None) -> str:
        """Return an exact controlled alias match, never a fuzzy inference.

        Crop-qualified aliases take precedence.  This keeps source-specific
        labels such as maize ``silking`` from being applied to unrelated crops.
        """
        normalized = normalize_stage_token(raw_stage)
        if crop_name is not None:
            crop = normalize_stage_token(crop_name)
            for alias in self.crop_stage_aliases:
                if alias.raw_crop == crop and alias.raw_stage == normalized:
                    return alias.stage_bucket
        for alias in self.stage_aliases:
            if alias.raw_stage == normalized:
                return alias.stage_bucket
        return "unknown"

    def state_priority_for(self, event_state: Any) -> LaneStatePriority:
        normalized = str(event_state or "").strip().upper()
        for item in self.lane_state_priorities:
            if item.event_state == normalized:
                return item
        return LaneStatePriority(normalized or "UNKNOWN", 0, False)


def normalize_stage_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").strip().lower()).strip("_") or "unknown"


def load_incident_policy_v3(
    path: Path = DEFAULT_INCIDENT_POLICY_V3_PATH,
) -> IncidentPolicyV3:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Incident V3 policy is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Incident V3 policy root must be an object")

    schema_version = _text(payload, "policy_schema_version")
    version = _text(payload, "policy_version")
    calibration_status = _text(payload, "calibration_status")
    warning = _text(payload, "warning")
    week_start = _text(payload, "week_start").lower()
    monitored_rule = _text(payload, "monitored_rule")
    evaluable_rule = _text(payload, "evaluable_rule")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Incident V3 policy schema: {schema_version}")
    if "UNCALIBRATED" not in calibration_status.upper() or "UNCALIBRATED" not in warning.upper():
        raise ValueError("Incident V3 policy must explicitly warn that it is uncalibrated")
    if week_start != "monday":
        raise ValueError("Incident V3 currently requires Monday reporting weeks")

    starter = payload.get("tracker_starter_parameters")
    if not isinstance(starter, dict):
        raise ValueError("Incident V3 tracker_starter_parameters must be an object")
    identity_namespace = _starter_text(starter, "identity_namespace")
    grid_cell_size_km = _positive_number(starter, "grid_cell_size_km")
    grid_origin_lon = _bounded_number(starter, "grid_origin_lon", -180.0, 180.0)
    grid_origin_lat = _bounded_number(starter, "grid_origin_lat", -90.0, 90.0)
    reference_latitude_strategy = _starter_text(starter, "reference_latitude_strategy")
    if reference_latitude_strategy != "fixed_origin":
        raise ValueError(
            "Incident V3 requires fixed_origin so ordinary appends cannot rewrite grid IDs"
        )
    baseline_prior_strength = _positive_number(starter, "baseline_prior_strength")
    minimum_source_field_centroid_coverage = _bounded_number(
        starter, "minimum_source_field_centroid_coverage", 0.0, 1.0
    )
    minimum_source_crop_instance_week_centroid_coverage = _bounded_number(
        starter,
        "minimum_source_crop_instance_week_centroid_coverage",
        0.0,
        1.0,
    )
    minimum_known_stage_coverage = _bounded_number(
        starter, "minimum_known_stage_coverage", 0.0, 1.0
    )
    minimum_known_stage_coverage_per_supported_crop = _bounded_number(
        starter,
        "minimum_known_stage_coverage_per_supported_crop",
        0.0,
        1.0,
    )
    minimum_stage_coverage_crop_instance_weeks = _positive_integer(
        starter, "minimum_stage_coverage_crop_instance_weeks"
    )
    minimum_evaluable_fields = _positive_integer(starter, "minimum_evaluable_fields")
    minimum_coverage_ratio = _bounded_number(
        starter, "minimum_coverage_ratio", 0.0, 1.0
    )
    minimum_active_fields = _positive_integer(starter, "minimum_active_fields")
    severe_override_min_fields = _positive_integer(starter, "severe_override_min_fields")
    severe_override_min_fresh_response_fields = _positive_integer(
        starter, "severe_override_min_fresh_response_fields"
    )
    allow_severe_override = starter.get("allow_severe_override")
    if not isinstance(allow_severe_override, bool):
        raise ValueError("Incident V3 allow_severe_override must be boolean")
    frontier_distance_cells = _nonnegative_integer(
        starter, "frontier_distance_cells"
    )
    fdr_alpha = _bounded_number(starter, "fdr_alpha", 0.0, 1.0, exclusive=True)
    minimum_link_score = _bounded_number(starter, "minimum_link_score", 0.0, 1.0)
    lineage_threshold = _bounded_number(
        starter, "lineage_threshold", 0.0, 1.0
    )
    minimum_lineage_jaccard = _bounded_number(
        starter, "minimum_lineage_jaccard", 0.0, 1.0
    )
    same_hazard_link_required = starter.get("same_hazard_link_required")
    if same_hazard_link_required is not True:
        raise ValueError("Incident V3 links must use a hard same-hazard gate")
    max_link_gap_weeks = _positive_integer(starter, "max_link_gap_weeks")
    spatial_scale_km = _positive_number(starter, "spatial_scale_km")
    gap_penalty = _bounded_number(starter, "gap_penalty", 0.0, 1.0)
    confirmation_weeks = _positive_integer(starter, "confirmation_weeks")
    candidate_expiry_observed_weeks = _positive_integer(
        starter, "candidate_expiry_observed_weeks"
    )
    quiet_close_weeks = _positive_integer(starter, "quiet_close_weeks")
    recovery_observed_weeks = _positive_integer(
        starter, "recovery_observed_weeks"
    )
    recovery_grace_weeks = _positive_integer(starter, "recovery_grace_weeks")
    severe_confirmation_min_fields = _positive_integer(
        starter, "severe_confirmation_min_fields"
    )
    severe_confirmation_min_fresh_response_fields = _positive_integer(
        starter, "severe_confirmation_min_fresh_response_fields"
    )
    minimum_crop_monitored_instances = _positive_integer(
        starter, "minimum_crop_monitored_instances"
    )
    minimum_crop_evaluable_instances = _positive_integer(
        starter, "minimum_crop_evaluable_instances"
    )
    maximum_data_gap_weeks = _positive_integer(
        starter, "maximum_data_gap_weeks"
    )
    weights_raw = starter.get("link_weights")
    if not isinstance(weights_raw, dict) or not weights_raw:
        raise ValueError("Incident V3 link_weights must be a non-empty object")
    link_weights: list[tuple[str, float]] = []
    for name, value in sorted(weights_raw.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Incident V3 link weight names must be non-empty strings")
        weight = _bounded_value(value, f"link weight {name}", 0.0, 1.0)
        link_weights.append((name.strip(), weight))
    if {name for name, _ in link_weights} != LINK_WEIGHT_NAMES:
        raise ValueError("Incident V3 link weights do not match the frozen contract")
    if abs(sum(value for _, value in link_weights) - 1.0) > 1e-9:
        raise ValueError("Incident V3 link weights must sum to exactly 1.0")

    buckets_raw = payload.get("stage_buckets")
    if not isinstance(buckets_raw, list) or not buckets_raw:
        raise ValueError("Incident V3 stage_buckets must be a non-empty list")
    stage_buckets = tuple(_normalized_text(value, "stage bucket") for value in buckets_raw)
    if stage_buckets != CONTROLLED_STAGE_BUCKETS:
        raise ValueError("Incident V3 stage buckets do not match the frozen contract")

    aliases_raw = payload.get("stage_aliases")
    if not isinstance(aliases_raw, dict):
        raise ValueError("Incident V3 stage_aliases must be an object")
    if set(aliases_raw) != set(stage_buckets):
        raise ValueError(
            "Incident V3 stage_aliases must define every frozen stage bucket exactly"
        )
    aliases: list[StageAlias] = []
    seen_aliases: set[str] = set()
    for bucket, values in aliases_raw.items():
        normalized_bucket = _normalized_text(bucket, "stage alias bucket")
        if normalized_bucket not in stage_buckets:
            raise ValueError(f"Stage aliases reference unsupported bucket: {bucket}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Stage aliases for {bucket} must be a non-empty list")
        if normalized_bucket not in {
            _normalized_text(value, "stage alias") for value in values
        }:
            raise ValueError(
                f"Incident V3 stage aliases must include canonical bucket {bucket}"
            )
        for value in values:
            alias = _normalized_text(value, "stage alias")
            if alias in seen_aliases:
                raise ValueError(f"Duplicate Incident V3 stage alias: {alias}")
            seen_aliases.add(alias)
            aliases.append(StageAlias(alias, normalized_bucket))

    crop_aliases_raw = payload.get("crop_stage_aliases")
    if not isinstance(crop_aliases_raw, list) or not crop_aliases_raw:
        raise ValueError("Incident V3 crop_stage_aliases must be a non-empty list")
    crop_aliases: list[CropStageAlias] = []
    seen_crop_aliases: set[tuple[str, str]] = set()
    for row in crop_aliases_raw:
        if not isinstance(row, dict):
            raise ValueError("Incident V3 crop stage alias rows must be objects")
        crop = _normalized_text(row.get("crop_name"), "crop stage alias crop_name")
        stage = _normalized_text(row.get("raw_stage"), "crop stage alias raw_stage")
        bucket = _normalized_text(row.get("stage_bucket"), "crop stage alias stage_bucket")
        if bucket not in stage_buckets:
            raise ValueError(
                f"Crop stage alias references unsupported bucket: {bucket}"
            )
        if stage in seen_aliases:
            raise ValueError(
                f"Incident V3 crop-qualified stage alias must not also be global: {stage}"
            )
        key = (crop, stage)
        if key in seen_crop_aliases:
            raise ValueError(
                f"Duplicate Incident V3 crop stage alias: {crop}/{stage}"
            )
        seen_crop_aliases.add(key)
        crop_aliases.append(CropStageAlias(crop, stage, bucket))

    priorities_raw = payload.get("lane_state_priority")
    if not isinstance(priorities_raw, list) or not priorities_raw:
        raise ValueError("Incident V3 lane_state_priority must be a non-empty list")
    priorities: list[LaneStatePriority] = []
    seen_states: set[str] = set()
    seen_values: set[int] = set()
    for row in priorities_raw:
        if not isinstance(row, dict):
            raise ValueError("Incident V3 lane priority rows must be objects")
        state = _text(row, "event_state").upper()
        priority = row.get("priority")
        is_open = row.get("is_open")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 1:
            raise ValueError(f"Lane priority for {state} must be a positive integer")
        if not isinstance(is_open, bool):
            raise ValueError(f"Lane is_open for {state} must be boolean")
        if state in seen_states or priority in seen_values:
            raise ValueError("Incident V3 lane states and priority values must be unique")
        seen_states.add(state)
        seen_values.add(priority)
        priorities.append(LaneStatePriority(state, priority, is_open))
    ranked_states = tuple(
        item.event_state for item in sorted(priorities, key=lambda item: -item.priority)[:6]
    )
    if ranked_states != CANONICAL_LANE_ORDER:
        raise ValueError("Incident V3 lane priority does not match the frozen contract")

    return IncidentPolicyV3(
        schema_version=schema_version,
        version=version,
        calibration_status=calibration_status,
        warning=warning,
        week_start=week_start,
        monitored_rule=monitored_rule,
        evaluable_rule=evaluable_rule,
        identity_namespace=identity_namespace,
        grid_cell_size_km=grid_cell_size_km,
        grid_origin_lon=grid_origin_lon,
        grid_origin_lat=grid_origin_lat,
        reference_latitude_strategy=reference_latitude_strategy,
        baseline_prior_strength=baseline_prior_strength,
        minimum_source_field_centroid_coverage=minimum_source_field_centroid_coverage,
        minimum_source_crop_instance_week_centroid_coverage=(
            minimum_source_crop_instance_week_centroid_coverage
        ),
        minimum_known_stage_coverage=minimum_known_stage_coverage,
        minimum_known_stage_coverage_per_supported_crop=(
            minimum_known_stage_coverage_per_supported_crop
        ),
        minimum_stage_coverage_crop_instance_weeks=(
            minimum_stage_coverage_crop_instance_weeks
        ),
        minimum_evaluable_fields=minimum_evaluable_fields,
        minimum_coverage_ratio=minimum_coverage_ratio,
        minimum_active_fields=minimum_active_fields,
        severe_override_min_fields=severe_override_min_fields,
        severe_override_min_fresh_response_fields=(
            severe_override_min_fresh_response_fields
        ),
        allow_severe_override=allow_severe_override,
        frontier_distance_cells=frontier_distance_cells,
        fdr_alpha=fdr_alpha,
        minimum_link_score=minimum_link_score,
        lineage_threshold=lineage_threshold,
        minimum_lineage_jaccard=minimum_lineage_jaccard,
        same_hazard_link_required=same_hazard_link_required,
        max_link_gap_weeks=max_link_gap_weeks,
        spatial_scale_km=spatial_scale_km,
        gap_penalty=gap_penalty,
        confirmation_weeks=confirmation_weeks,
        candidate_expiry_observed_weeks=candidate_expiry_observed_weeks,
        quiet_close_weeks=quiet_close_weeks,
        recovery_observed_weeks=recovery_observed_weeks,
        recovery_grace_weeks=recovery_grace_weeks,
        severe_confirmation_min_fields=severe_confirmation_min_fields,
        severe_confirmation_min_fresh_response_fields=(
            severe_confirmation_min_fresh_response_fields
        ),
        minimum_crop_monitored_instances=minimum_crop_monitored_instances,
        minimum_crop_evaluable_instances=minimum_crop_evaluable_instances,
        maximum_data_gap_weeks=maximum_data_gap_weeks,
        link_weights=tuple(link_weights),
        stage_buckets=stage_buckets,
        stage_aliases=tuple(aliases),
        crop_stage_aliases=tuple(crop_aliases),
        lane_state_priorities=tuple(priorities),
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Incident V3 policy {key} must be a non-empty string")
    return value.strip()


def _normalized_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Incident V3 {label} must be a non-empty string")
    return normalize_stage_token(value)


def _starter_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Incident V3 starter parameter {key} must be a non-empty string")
    return value.strip()


def _positive_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Incident V3 starter parameter {key} must be a positive integer")
    return value


def _nonnegative_integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Incident V3 starter parameter {key} must be a non-negative integer"
        )
    return value


def _positive_number(payload: dict[str, Any], key: str) -> float:
    value = _bounded_value(payload.get(key), key, 0.0, float("inf"))
    if value <= 0:
        raise ValueError(f"Incident V3 starter parameter {key} must be positive")
    return value


def _bounded_number(
    payload: dict[str, Any], key: str, minimum: float, maximum: float, *, exclusive: bool = False
) -> float:
    value = _bounded_value(payload.get(key), key, minimum, maximum)
    if exclusive and not minimum < value < maximum:
        raise ValueError(
            f"Incident V3 starter parameter {key} must be between {minimum} and {maximum}"
        )
    return value


def _bounded_value(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Incident V3 {label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"Incident V3 {label} must be in [{minimum}, {maximum}]")
    return normalized


__all__ = [
    "DEFAULT_INCIDENT_POLICY_V3_PATH",
    "CANONICAL_LANE_ORDER",
    "CONTROLLED_STAGE_BUCKETS",
    "CropStageAlias",
    "IncidentPolicyV3",
    "LaneStatePriority",
    "LINK_WEIGHT_NAMES",
    "POLICY_SCHEMA_VERSION",
    "StageAlias",
    "load_incident_policy_v3",
    "normalize_stage_token",
]
