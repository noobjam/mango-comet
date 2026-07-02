from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = "weekly-story-monitor/v1"
OPEN_STATES = frozenset({"WATCH", "ACTIVE", "SEVERE", "QUIET_PENDING", "RECOVERING"})
CLOSED_STATES = frozenset(
    {
        "CLOSED_WATCH_QUIET",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED",
        "CLOSED_RECOVERED",
        "CLOSED_RESPONSE_UNRESOLVED",
        "CLOSED_SEASON_BOUNDARY",
    }
)

RISK_RANKS = {
    "NONE": 0,
    "LOW": 1,
    "LOW-MED": 2,
    "MED-HIGH": 3,
    "HIGH": 4,
}


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Policy value {key!r} must be numeric.")
    return float(value)


@dataclass(frozen=True)
class MonitorPolicy:
    version: str
    calibration_status: str
    crop_instance_gap_days: int
    freshness_fresh_days: int
    freshness_aging_days: int
    reference_min_days: int
    reference_max_days: int
    medium_ndvi_delta: float
    medium_ndmi_delta: float
    medium_psri_delta: float
    severe_ndvi_delta: float
    severe_ndmi_delta: float
    severe_psri_delta: float
    recovery_ndvi_delta: float
    recovery_ndmi_delta: float
    recovery_psri_delta: float
    data_gap_days: int
    max_recovery_days: int
    quiet_days_by_hazard: dict[str, int]
    source_path: Path
    source_sha256: str

    def quiet_days(self, hazard: str) -> int:
        return self.quiet_days_by_hazard.get(
            hazard, self.quiet_days_by_hazard.get("other", 7)
        )


def load_policy(path: Path) -> MonitorPolicy:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Policy is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Policy root must be a JSON object.")

    spectral = payload.get("spectral_change_thresholds")
    freshness = payload.get("spectral_freshness_days")
    reference = payload.get("reference_window_days")
    quiet = payload.get("quiet_observed_days_by_hazard")
    if not all(isinstance(item, dict) for item in (spectral, freshness, reference, quiet)):
        raise ValueError("Policy spectral, freshness, reference, and quiet sections are required.")
    assert isinstance(spectral, dict)
    assert isinstance(freshness, dict)
    assert isinstance(reference, dict)
    assert isinstance(quiet, dict)

    calibration_status = str(payload.get("calibration_status", ""))
    if "UNCALIBRATED" not in calibration_status.upper():
        raise ValueError("V1 policy must explicitly identify its starter thresholds as uncalibrated.")
    quiet_days = {str(key): int(value) for key, value in quiet.items()}
    if not quiet_days or min(quiet_days.values()) < 1:
        raise ValueError("Quiet-day thresholds must be positive integers.")

    policy = MonitorPolicy(
        version=str(payload.get("policy_version", "")).strip(),
        calibration_status=calibration_status,
        crop_instance_gap_days=int(payload.get("crop_instance_gap_days", 60)),
        freshness_fresh_days=int(freshness.get("fresh", 7)),
        freshness_aging_days=int(freshness.get("aging", 14)),
        reference_min_days=int(reference.get("minimum", 7)),
        reference_max_days=int(reference.get("maximum", 21)),
        medium_ndvi_delta=_require_number(spectral, "medium_ndvi_delta"),
        medium_ndmi_delta=_require_number(spectral, "medium_ndmi_delta"),
        medium_psri_delta=_require_number(spectral, "medium_psri_delta"),
        severe_ndvi_delta=_require_number(spectral, "severe_ndvi_delta"),
        severe_ndmi_delta=_require_number(spectral, "severe_ndmi_delta"),
        severe_psri_delta=_require_number(spectral, "severe_psri_delta"),
        recovery_ndvi_delta=_require_number(spectral, "recovery_ndvi_delta"),
        recovery_ndmi_delta=_require_number(spectral, "recovery_ndmi_delta"),
        recovery_psri_delta=_require_number(spectral, "recovery_psri_delta"),
        data_gap_days=int(payload.get("data_gap_days", 7)),
        max_recovery_days=int(payload.get("max_recovery_days", 28)),
        quiet_days_by_hazard=quiet_days,
        source_path=path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if not policy.version:
        raise ValueError("policy_version is required.")
    if not 0 <= policy.freshness_fresh_days <= policy.freshness_aging_days:
        raise ValueError("Spectral freshness thresholds are out of order.")
    if not 1 <= policy.reference_min_days <= policy.reference_max_days:
        raise ValueError("Reference-window thresholds are out of order.")
    return policy


def stable_id(prefix: str, parts: Iterable[Any], *, length: int = 24) -> str:
    normalized = "\x1f".join("" if value is None else str(value) for value in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def normalize_risk(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", "-", str(value or "NONE").strip().upper()).strip("-")
    aliases = {
        "": "NONE",
        "NO-RISK": "NONE",
        "LOW-MEDIUM": "LOW-MED",
        "MEDIUM-HIGH": "MED-HIGH",
        "MED": "LOW-MED",
        "MEDIUM": "LOW-MED",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in RISK_RANKS else "NONE"


def normalize_hazard(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not text or text in {"none", "null", "nan", "no_risk"}:
        return "none"
    if "drought" in text or "dry" in text:
        return "drought"
    if "pond" in text or "flood" in text or "waterlog" in text:
        return "ponding_flooding"
    if "heat" in text or "hot" in text or "temperature" in text:
        return "heat"
    if "wind" in text:
        return "damaging_wind"
    return text


def normalize_stage(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").strip().lower()).strip("_")
    return text or "unknown"
