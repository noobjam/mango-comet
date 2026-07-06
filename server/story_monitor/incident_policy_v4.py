"""Frozen source/cadence policy for acquisition-aware Incident V4 evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


POLICY_SCHEMA_VERSION = "incident-source-policy-v4/1"
AVAILABILITY_MODES = ("strict", "reconstructed")
HAZARD_FAMILIES = ("drought", "ponding_flooding", "heat", "damaging_wind")
CONTROLLED_STAGE_BUCKETS = (
    "emergence", "vegetative", "flowering", "fruiting_or_grain_fill",
    "maturity_or_harvest", "off_season", "unknown",
)
DEFAULT_INCIDENT_POLICY_V4_PATH = (
    Path(__file__).resolve().parent / "policies" / "incident_policy_v4.json"
)


@dataclass(frozen=True)
class IncidentPolicyV4:
    schema_version: str
    version: str
    calibration_status: str
    warning: str
    availability_modes: tuple[str, ...]
    hazard_families: tuple[str, ...]
    freshness_fresh_days: int
    freshness_aging_days: int
    reference_min_days: int
    reference_max_days: int
    minimum_valid_pixel_fraction: float
    maximum_cloud_pct: float
    rejected_quality_flags: tuple[str, ...]
    medium_ndvi_delta: float
    medium_ndmi_delta: float
    medium_psri_delta: float
    severe_ndvi_delta: float
    severe_ndmi_delta: float
    severe_psri_delta: float
    recovery_ndvi_delta: float
    recovery_ndmi_delta: float
    recovery_psri_delta: float
    pressure_low_medium: float
    pressure_medium_high: float
    pressure_high: float
    stage_buckets: tuple[str, ...]
    stage_aliases: tuple[tuple[str, str], ...]
    source_path: Path
    source_sha256: str

    def validate_availability_mode(self, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in self.availability_modes:
            raise ValueError(
                "availability_mode must be one of: "
                + ", ".join(self.availability_modes)
            )
        return normalized

    def stage_bucket_for(self, value: Any) -> str:
        normalized = normalize_stage_token(value)
        return dict(self.stage_aliases).get(normalized, "unknown")


def normalize_stage_token(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").strip().lower())
    return text.strip("_") or "unknown"


def load_incident_policy_v4(
    path: Path = DEFAULT_INCIDENT_POLICY_V4_PATH,
) -> IncidentPolicyV4:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Incident V4 policy is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Incident V4 policy root must be an object")

    schema_version = _text(payload, "policy_schema_version")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Incident V4 policy schema: {schema_version}")
    version = _text(payload, "policy_version")
    calibration_status = _text(payload, "calibration_status")
    warning = _text(payload, "warning")
    if "UNCALIBRATED" not in calibration_status.upper() or "UNCALIBRATED" not in warning.upper():
        raise ValueError("Incident V4 policy must explicitly warn that it is uncalibrated")

    availability_modes = _exact_string_tuple(
        payload, "availability_modes", AVAILABILITY_MODES
    )
    hazard_families = _exact_string_tuple(
        payload, "hazard_families", HAZARD_FAMILIES
    )
    freshness = _object(payload, "spectral_freshness_days")
    reference = _object(payload, "reference_window_days")
    quality = _object(payload, "spectral_quality")
    spectral = _object(payload, "spectral_change_thresholds")
    pressure = _object(payload, "pressure_rank_thresholds")
    fresh = _nonnegative_int(freshness, "fresh")
    aging = _nonnegative_int(freshness, "aging")
    reference_min = _positive_int(reference, "minimum")
    reference_max = _positive_int(reference, "maximum")
    if fresh > aging:
        raise ValueError("Incident V4 spectral freshness thresholds are out of order")
    if reference_min > reference_max:
        raise ValueError("Incident V4 reference window thresholds are out of order")

    minimum_valid = _bounded_number(quality, "minimum_valid_pixel_fraction", 0.0, 1.0)
    maximum_cloud = _bounded_number(quality, "maximum_cloud_pct", 0.0, 100.0)
    rejected = quality.get("rejected_quality_flags")
    if not isinstance(rejected, list) or not rejected or any(
        not isinstance(item, str) or not item.strip() for item in rejected
    ):
        raise ValueError("Incident V4 rejected_quality_flags must be non-empty strings")

    low_medium = _number(pressure, "low_medium")
    medium_high = _number(pressure, "medium_high")
    high = _number(pressure, "high")
    if not 0 <= low_medium < medium_high < high:
        raise ValueError("Incident V4 pressure rank thresholds are out of order")

    buckets = _exact_string_tuple(payload, "stage_buckets", CONTROLLED_STAGE_BUCKETS)
    aliases_raw = _object(payload, "stage_aliases")
    if set(aliases_raw) != set(buckets):
        raise ValueError("Incident V4 stage_aliases must define every frozen stage bucket exactly")
    aliases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for bucket, values in aliases_raw.items():
        normalized_bucket = normalize_stage_token(bucket)
        if (
            normalized_bucket not in buckets
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(f"Invalid Incident V4 stage aliases for {bucket}")
        if normalized_bucket not in {normalize_stage_token(value) for value in values}:
            raise ValueError(
                f"Incident V4 stage aliases must include canonical bucket {bucket}"
            )
        for value in values:
            alias = normalize_stage_token(value)
            if alias in seen:
                raise ValueError(f"Duplicate Incident V4 stage alias: {alias}")
            seen.add(alias)
            aliases.append((alias, normalized_bucket))

    values = {
        name: _number(spectral, name)
        for name in (
            "medium_ndvi_delta", "medium_ndmi_delta", "medium_psri_delta",
            "severe_ndvi_delta", "severe_ndmi_delta", "severe_psri_delta",
            "recovery_ndvi_delta", "recovery_ndmi_delta", "recovery_psri_delta",
        )
    }
    if not values["severe_ndvi_delta"] <= values["medium_ndvi_delta"] < 0:
        raise ValueError("Incident V4 NDVI decline thresholds are out of order")
    if not values["severe_ndmi_delta"] <= values["medium_ndmi_delta"] < 0:
        raise ValueError("Incident V4 NDMI decline thresholds are out of order")
    if not 0 < values["medium_psri_delta"] <= values["severe_psri_delta"]:
        raise ValueError("Incident V4 PSRI decline thresholds are out of order")
    if not (
        values["recovery_ndvi_delta"] > 0
        and values["recovery_ndmi_delta"] > 0
        and values["recovery_psri_delta"] < 0
    ):
        raise ValueError("Incident V4 recovery thresholds have invalid directions")
    return IncidentPolicyV4(
        schema_version=schema_version, version=version,
        calibration_status=calibration_status, warning=warning,
        availability_modes=availability_modes, hazard_families=hazard_families,
        freshness_fresh_days=fresh, freshness_aging_days=aging,
        reference_min_days=reference_min, reference_max_days=reference_max,
        minimum_valid_pixel_fraction=minimum_valid,
        maximum_cloud_pct=maximum_cloud,
        rejected_quality_flags=tuple(sorted({item.strip().lower() for item in rejected})),
        **values,
        pressure_low_medium=low_medium, pressure_medium_high=medium_high,
        pressure_high=high, stage_buckets=buckets,
        stage_aliases=tuple(aliases), source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Incident V4 policy {key} must be a non-empty string")
    return value.strip()


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Incident V4 policy {key} must be an object")
    return value


def _exact_string_tuple(
    payload: dict[str, Any], key: str, expected: tuple[str, ...]
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Incident V4 policy {key} must be a string list")
    result = tuple(item.strip() for item in value)
    if result != expected:
        raise ValueError(f"Incident V4 policy {key} does not match the frozen contract")
    return result


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Incident V4 policy {key} must be finite numeric")
    return float(value)


def _bounded_number(
    payload: dict[str, Any], key: str, minimum: float, maximum: float
) -> float:
    value = _number(payload, key)
    if not minimum <= value <= maximum:
        raise ValueError(f"Incident V4 policy {key} must be in [{minimum}, {maximum}]")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Incident V4 policy {key} must be a positive integer")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Incident V4 policy {key} must be a non-negative integer")
    return value


__all__ = [
    "AVAILABILITY_MODES", "DEFAULT_INCIDENT_POLICY_V4_PATH", "HAZARD_FAMILIES",
    "IncidentPolicyV4", "load_incident_policy_v4", "normalize_stage_token",
]
