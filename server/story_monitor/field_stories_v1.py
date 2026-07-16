"""Deterministic multi-hazard field/crop stories over V4 evidence ledgers.

A field story is a fixed-location open-concern interval.  Hazard lanes remain
auditable inside the story, but a hazard or crop-stage change does not rewrite
story identity.  This module intentionally contains no clustering, learned
labels, spatial incident tracking, or visualization logic.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import stable_id


POLICY_PATH = Path(__file__).with_name("policies") / "field_story_policy_v1.json"
DECLINE_RESPONSES = frozenset({"medium_decline", "severe_decline"})
POSITIVE_RESPONSES = frozenset({*DECLINE_RESPONSES, "recovery"})
OPEN_STATES = frozenset(
    {"CANDIDATE", "ACTIVE", "SEVERE", "QUIET_PENDING", "RECOVERING", "DATA_GAP"}
)

_CROP_COLUMNS = {
    "field_id",
    "crop_instance_id",
    "observation_date",
    "knowledge_time",
    "crop_name",
    "crop_season",
    "stage_bucket",
}
_PRESSURE_COLUMNS = {
    "field_id",
    "crop_instance_id",
    "knowledge_time",
    "hazard_family",
    "pressure_observed",
    "pressure_rank",
}
_RESPONSE_COLUMNS = {
    "field_id",
    "crop_instance_id",
    "spectral_source_date",
    "knowledge_time",
    "spectral_usable",
    "new_response_evidence",
    "response_class",
}


@dataclass(frozen=True)
class FieldStoryPolicy:
    version: str
    calibration_status: str
    confirmation_window_observed_days: int
    confirmation_min_elevated_days: int
    severe_pressure_rank: int
    quiet_observed_days: int
    candidate_expiry_observed_days: int
    maximum_data_gap_days: int
    maximum_recovery_observed_days: int


@dataclass(frozen=True)
class FieldStoryArtifacts:
    daily_state: pd.DataFrame
    chapters: pd.DataFrame
    windows: pd.DataFrame
    hazard_daily: pd.DataFrame


@dataclass
class _HazardThread:
    hazard_family: str
    first_elevated_date: date
    last_elevated_date: date
    max_risk_rank: int
    quiet_observed_days: int = 0
    open: bool = True


@dataclass
class _StoryRuntime:
    story_id: str
    field_id: str
    crop_instance_id: str
    crop_name: str
    crop_season: str
    first_evidence_date: date
    story_known_time: pd.Timestamp
    last_updated_time: pd.Timestamp
    confirmed_time: pd.Timestamp | None = None
    state: str = "CANDIDATE"
    last_decision_date: date | None = None
    last_stage: str = "unknown"
    threads: dict[str, _HazardThread] = field(default_factory=dict)
    encountered_hazards: list[str] = field(default_factory=list)
    max_risk_rank: int = 0
    observed_day_count: int = 0
    reportable_day_count: int = 0
    consecutive_data_gap_days: int = 0
    recovery_observed_days: int = 0
    last_decline_source_date: date | None = None
    last_recovery_source_date: date | None = None
    response_status: str = "none_observed"
    close_reason: str | None = None
    requires_review: bool = False

    @property
    def unresolved_decline(self) -> bool:
        return self.last_decline_source_date is not None and (
            self.last_recovery_source_date is None
            or self.last_recovery_source_date <= self.last_decline_source_date
        )

    @property
    def recovered_decline(self) -> bool:
        return self.last_decline_source_date is not None and not self.unresolved_decline

    @property
    def open_hazards(self) -> list[str]:
        return sorted(name for name, item in self.threads.items() if item.open)


def load_field_story_policy(path: Path | None = None) -> FieldStoryPolicy:
    source = path or POLICY_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    calibration = str(payload.get("calibration_status") or "")
    if "UNCALIBRATED" not in calibration.upper():
        raise ValueError(
            "Field-story policy must identify its thresholds as uncalibrated"
        )
    policy = FieldStoryPolicy(
        version=str(payload.get("policy_version") or "").strip(),
        calibration_status=calibration,
        confirmation_window_observed_days=int(
            payload["confirmation"]["window_observed_days"]
        ),
        confirmation_min_elevated_days=int(
            payload["confirmation"]["minimum_elevated_days"]
        ),
        severe_pressure_rank=int(payload["confirmation"]["severe_pressure_rank"]),
        quiet_observed_days=int(payload["closure"]["quiet_observed_days"]),
        candidate_expiry_observed_days=int(
            payload["closure"]["candidate_expiry_observed_days"]
        ),
        maximum_data_gap_days=int(payload["closure"]["maximum_data_gap_days"]),
        maximum_recovery_observed_days=int(
            payload["closure"]["maximum_recovery_observed_days"]
        ),
    )
    if not policy.version:
        raise ValueError("Field-story policy_version is required")
    positive = (
        policy.confirmation_window_observed_days,
        policy.confirmation_min_elevated_days,
        policy.quiet_observed_days,
        policy.candidate_expiry_observed_days,
        policy.maximum_data_gap_days,
        policy.maximum_recovery_observed_days,
    )
    if min(positive) < 1:
        raise ValueError("Field-story policy day thresholds must be positive")
    if policy.confirmation_min_elevated_days > policy.confirmation_window_observed_days:
        raise ValueError("Confirmation minimum cannot exceed its observed-day window")
    if policy.severe_pressure_rank not in {2, 3, 4}:
        raise ValueError("severe_pressure_rank must be in [2, 4]")
    return policy


def build_field_stories(
    crop_days: pd.DataFrame,
    pressure: pd.DataFrame,
    responses: pd.DataFrame,
    *,
    policy: FieldStoryPolicy | None = None,
) -> FieldStoryArtifacts:
    """Compose deterministic field/crop stories from V4-shaped ledgers.

    Evidence is processed on the first calendar day it could have been known.
    A late observation can affect the story from that decision day onward but
    never rewrites an earlier state.
    """
    selected_policy = policy or load_field_story_policy()
    crops = _normalize_crops(crop_days)
    pressures = _normalize_pressure(pressure)
    accepted_responses = _normalize_responses(responses)
    _validate_ownership(crops, pressures, accepted_responses)

    crop_lookup = {
        key: group.reset_index(drop=True)
        for key, group in crops.groupby(
            ["field_id", "crop_instance_id"], sort=True, dropna=False
        )
    }
    pressure_groups = {
        key: group
        for key, group in pressures.groupby(
            ["field_id", "crop_instance_id"], sort=True, dropna=False
        )
    }
    response_groups = {
        key: group
        for key, group in accepted_responses.groupby(
            ["field_id", "crop_instance_id"], sort=True, dropna=False
        )
    }
    keys = sorted(set(pressure_groups) | set(response_groups))

    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    for key in keys:
        crop_history = crop_lookup.get(key)
        if crop_history is None or crop_history.empty:
            raise ValueError(
                f"Story evidence has no crop context for {key[0]}/{key[1]}"
            )
        group_pressure = pressure_groups.get(key, pressures.iloc[0:0])
        group_responses = response_groups.get(key, accepted_responses.iloc[0:0])
        group_daily, group_windows, group_hazards = _compose_crop_instance(
            str(key[0]),
            str(key[1]),
            crop_history,
            group_pressure,
            group_responses,
            selected_policy,
        )
        daily_rows.extend(group_daily)
        window_rows.extend(group_windows)
        hazard_rows.extend(group_hazards)

    daily = _frame(daily_rows, _DAILY_COLUMNS)
    windows = _frame(window_rows, _WINDOW_COLUMNS)
    hazards = _frame(hazard_rows, _HAZARD_COLUMNS)
    chapters = _build_chapters(daily)
    _validate_outputs(daily, chapters, windows, hazards)
    return FieldStoryArtifacts(daily, chapters, windows, hazards)


def _compose_crop_instance(
    field_id: str,
    crop_instance_id: str,
    crop_history: pd.DataFrame,
    pressure: pd.DataFrame,
    responses: pd.DataFrame,
    policy: FieldStoryPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pressure_by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in pressure.to_dict("records"):
        pressure_by_day[pd.Timestamp(row["decision_date"]).date()].append(row)
    response_by_day: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in responses.to_dict("records"):
        response_by_day[pd.Timestamp(row["decision_date"]).date()].append(row)
    days = sorted(set(pressure_by_day) | set(response_by_day))
    crop_by_day = _crop_context_as_of(crop_history, days)
    histories: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=policy.confirmation_window_observed_days)
    )
    runtime: _StoryRuntime | None = None
    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []

    for decision_date in days:
        crop = crop_by_day.get(decision_date)
        if crop is None:
            raise ValueError(
                "Story replay would use evidence before crop ownership was knowable: "
                f"{field_id}/{crop_instance_id}/{decision_date.isoformat()}"
            )
        lanes = {
            str(row["hazard_family"]): row
            for row in pressure_by_day.get(decision_date, [])
        }
        for hazard, lane in lanes.items():
            if bool(lane["pressure_observed"]):
                histories[hazard].append(int(lane["pressure_rank"]))
        day_responses = response_by_day.get(decision_date, [])
        _reject_conflicting_responses(day_responses, decision_date)
        elevated = sorted(
            hazard
            for hazard, lane in lanes.items()
            if bool(lane["pressure_observed"]) and int(lane["pressure_rank"]) >= 2
        )
        declines = [
            row
            for row in day_responses
            if str(row["response_class"]) in DECLINE_RESPONSES
        ]
        if runtime is None and not elevated and not declines:
            continue
        if runtime is None:
            knowledge_values = [crop["knowledge_time"]]
            knowledge_values.extend(row["knowledge_time"] for row in lanes.values())
            knowledge_values.extend(row["knowledge_time"] for row in day_responses)
            evidence_source_dates = [
                pd.Timestamp(lanes[hazard]["pressure_observation_date"]).date()
                for hazard in elevated
            ]
            evidence_source_dates.extend(
                pd.Timestamp(row["spectral_source_date"]).date() for row in declines
            )
            first_evidence_date = min(evidence_source_dates)
            story_known_time = max(pd.Timestamp(value) for value in knowledge_values)
            runtime = _StoryRuntime(
                story_id=stable_id(
                    "field_story",
                    (
                        policy.version,
                        field_id,
                        crop_instance_id,
                        first_evidence_date.isoformat(),
                        decision_date.isoformat(),
                    ),
                ),
                field_id=field_id,
                crop_instance_id=crop_instance_id,
                crop_name=str(crop["crop_name"]),
                crop_season=str(crop["crop_season"]),
                first_evidence_date=first_evidence_date,
                story_known_time=story_known_time,
                last_updated_time=story_known_time,
                last_decision_date=decision_date,
            )

        visible_state, active_hazards = _advance_story(
            runtime,
            decision_date,
            crop,
            lanes,
            day_responses,
            histories,
            policy,
        )
        daily_row = _daily_record(
            runtime, decision_date, crop, lanes, active_hazards, visible_state, policy
        )
        daily_rows.append(daily_row)
        hazard_rows.extend(
            _hazard_records(runtime, decision_date, lanes, active_hazards)
        )
        if runtime.state not in OPEN_STATES:
            window_rows.append(_window_record(runtime, policy, right_censored=False))
            runtime = None
            histories.clear()

    if runtime is not None:
        window_rows.append(_window_record(runtime, policy, right_censored=True))
    return daily_rows, window_rows, hazard_rows


def _advance_story(
    runtime: _StoryRuntime,
    decision_date: date,
    crop: dict[str, Any],
    lanes: dict[str, dict[str, Any]],
    responses: list[dict[str, Any]],
    histories: dict[str, deque[int]],
    policy: FieldStoryPolicy,
) -> tuple[str, list[str]]:
    runtime.last_decision_date = decision_date
    runtime.last_stage = str(crop["stage_bucket"])
    runtime.last_updated_time = max(
        [runtime.last_updated_time, pd.Timestamp(crop["knowledge_time"])]
        + [pd.Timestamp(row["knowledge_time"]) for row in lanes.values()]
        + [pd.Timestamp(row["knowledge_time"]) for row in responses]
    )
    observed_lanes = {
        hazard: row for hazard, row in lanes.items() if bool(row["pressure_observed"])
    }
    active_hazards = sorted(
        hazard
        for hazard, row in observed_lanes.items()
        if int(row["pressure_rank"]) >= 2
    )
    if observed_lanes:
        runtime.observed_day_count += 1
    if active_hazards or any(
        str(row["response_class"]) in DECLINE_RESPONSES for row in responses
    ):
        runtime.reportable_day_count += 1

    for hazard in active_hazards:
        rank = int(observed_lanes[hazard]["pressure_rank"])
        runtime.max_risk_rank = max(runtime.max_risk_rank, rank)
        thread = runtime.threads.get(hazard)
        if thread is None or not thread.open:
            runtime.threads[hazard] = _HazardThread(
                hazard_family=hazard,
                first_elevated_date=decision_date,
                last_elevated_date=decision_date,
                max_risk_rank=rank,
            )
            if hazard not in runtime.encountered_hazards:
                runtime.encountered_hazards.append(hazard)
        else:
            thread.last_elevated_date = decision_date
            thread.max_risk_rank = max(thread.max_risk_rank, rank)
            thread.quiet_observed_days = 0

    missing_open_hazard = False
    for hazard in runtime.open_hazards:
        if hazard in active_hazards:
            continue
        lane = lanes.get(hazard)
        if lane is None or not bool(lane["pressure_observed"]):
            missing_open_hazard = True
            continue
        thread = runtime.threads[hazard]
        thread.quiet_observed_days += 1
        quiet_threshold = (
            policy.candidate_expiry_observed_days
            if runtime.confirmed_time is None
            else policy.quiet_observed_days
        )
        if thread.quiet_observed_days >= quiet_threshold:
            thread.open = False

    for response in responses:
        response_class = str(response["response_class"])
        source_date = pd.Timestamp(response["spectral_source_date"]).date()
        if response_class in DECLINE_RESPONSES:
            if (
                runtime.last_decline_source_date is None
                or source_date > runtime.last_decline_source_date
            ):
                runtime.last_decline_source_date = source_date
            runtime.response_status = (
                "aligned_decline" if active_hazards else "unattributed_decline"
            )
            runtime.requires_review = runtime.requires_review or not active_hazards
        elif response_class == "recovery":
            if (
                runtime.last_recovery_source_date is None
                or source_date > runtime.last_recovery_source_date
            ):
                runtime.last_recovery_source_date = source_date
            runtime.response_status = (
                "recovery_observed"
                if runtime.recovered_decline
                else "recovery_without_prior_decline"
            )
            runtime.requires_review = (
                runtime.requires_review or runtime.last_decline_source_date is None
            )

    coverage_adequate = bool(observed_lanes) and not missing_open_hazard
    needs_observed_followup = bool(runtime.open_hazards) or runtime.unresolved_decline
    has_fresh_response = bool(responses)
    if (
        not coverage_adequate
        and needs_observed_followup
        and not (has_fresh_response and not runtime.open_hazards)
    ):
        runtime.consecutive_data_gap_days += 1
        if runtime.consecutive_data_gap_days >= policy.maximum_data_gap_days:
            runtime.state = "CLOSED_DATA_CENSORED"
            runtime.close_reason = "maximum_missing_evidence_gap"
            runtime.requires_review = True
            return runtime.state, active_hazards
        return "DATA_GAP", active_hazards
    runtime.consecutive_data_gap_days = 0

    severe = any(
        int(observed_lanes[hazard]["pressure_rank"]) >= policy.severe_pressure_rank
        for hazard in active_hazards
    ) or any(str(row["response_class"]) == "severe_decline" for row in responses)
    persistent = any(
        sum(rank >= 2 for rank in list(histories[hazard]))
        >= policy.confirmation_min_elevated_days
        for hazard in active_hazards
    )
    aligned_decline = any(
        str(row["response_class"]) in DECLINE_RESPONSES for row in responses
    ) and bool(active_hazards)

    if active_hazards:
        runtime.recovery_observed_days = 0
        if severe:
            runtime.state = "SEVERE"
        elif runtime.confirmed_time is not None or persistent or aligned_decline:
            runtime.state = "ACTIVE"
        else:
            runtime.state = "CANDIDATE"
        if runtime.state in {"ACTIVE", "SEVERE"} and runtime.confirmed_time is None:
            runtime.confirmed_time = runtime.last_updated_time
        return runtime.state, active_hazards

    if runtime.open_hazards:
        runtime.state = "QUIET_PENDING"
        return runtime.state, active_hazards

    if runtime.recovered_decline:
        runtime.state = "CLOSED_RECOVERED"
        runtime.close_reason = "later_usable_recovery_observation"
    elif runtime.unresolved_decline:
        runtime.recovery_observed_days += 1
        if runtime.recovery_observed_days >= policy.maximum_recovery_observed_days:
            runtime.state = "CLOSED_RESPONSE_UNRESOLVED"
            runtime.close_reason = "response_unresolved_at_observed_deadline"
            runtime.requires_review = True
        else:
            runtime.state = "RECOVERING" if runtime.encountered_hazards else "CANDIDATE"
    elif runtime.confirmed_time is None:
        quiet_days = max(
            (thread.quiet_observed_days for thread in runtime.threads.values()),
            default=runtime.observed_day_count,
        )
        if quiet_days >= policy.candidate_expiry_observed_days:
            runtime.state = "CLOSED_CANDIDATE_EXPIRED"
            runtime.close_reason = "candidate_evidence_did_not_confirm"
        else:
            runtime.state = "QUIET_PENDING"
    else:
        runtime.state = "CLOSED_PRESSURE_QUIET_NO_RESPONSE"
        runtime.close_reason = "all_hazard_threads_observed_quiet"
    return runtime.state, active_hazards


def _daily_record(
    runtime: _StoryRuntime,
    decision_date: date,
    crop: dict[str, Any],
    lanes: dict[str, dict[str, Any]],
    active_hazards: list[str],
    visible_state: str,
    policy: FieldStoryPolicy,
) -> dict[str, Any]:
    risk_vector = {
        hazard: (int(row["pressure_rank"]) if bool(row["pressure_observed"]) else None)
        for hazard, row in sorted(lanes.items())
    }
    return {
        "story_id": runtime.story_id,
        "field_id": runtime.field_id,
        "crop_instance_id": runtime.crop_instance_id,
        "crop_name": runtime.crop_name,
        "crop_season": runtime.crop_season,
        "decision_date": decision_date,
        "first_evidence_date": runtime.first_evidence_date,
        "story_known_time": runtime.story_known_time,
        "confirmed_time": runtime.confirmed_time,
        "state_known_time": runtime.last_updated_time,
        "story_state": visible_state,
        "stage_bucket": str(crop["stage_bucket"]),
        "active_hazards_json": json.dumps(active_hazards),
        "open_hazards_json": json.dumps(runtime.open_hazards),
        "encountered_hazards_json": json.dumps(runtime.encountered_hazards),
        "pressure_rank_by_hazard_json": json.dumps(risk_vector, sort_keys=True),
        "current_max_risk_rank": max(
            (rank for rank in risk_vector.values() if rank is not None), default=None
        ),
        "max_risk_rank": runtime.max_risk_rank,
        "response_status": runtime.response_status,
        "last_decline_source_date": runtime.last_decline_source_date,
        "last_recovery_source_date": runtime.last_recovery_source_date,
        "coverage_adequate": visible_state not in {"DATA_GAP", "CLOSED_DATA_CENSORED"},
        "requires_review": runtime.requires_review,
        "right_censored": visible_state in OPEN_STATES,
        "policy_version": policy.version,
    }


def _hazard_records(
    runtime: _StoryRuntime,
    decision_date: date,
    lanes: dict[str, dict[str, Any]],
    active_hazards: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for hazard in sorted(set(lanes) | set(runtime.threads)):
        lane = lanes.get(hazard)
        thread = runtime.threads.get(hazard)
        records.append(
            {
                "story_id": runtime.story_id,
                "field_id": runtime.field_id,
                "crop_instance_id": runtime.crop_instance_id,
                "decision_date": decision_date,
                "hazard_family": hazard,
                "pressure_observed": bool(
                    lane is not None and lane["pressure_observed"]
                ),
                "pressure_rank": (
                    int(lane["pressure_rank"])
                    if lane is not None and bool(lane["pressure_observed"])
                    else None
                ),
                "pressure_source_date": (
                    pd.Timestamp(lane["pressure_observation_date"]).date()
                    if lane is not None
                    else None
                ),
                "pressure_knowledge_time": (
                    lane["knowledge_time"] if lane is not None else None
                ),
                "hazard_active": hazard in active_hazards,
                "thread_open": bool(thread is not None and thread.open),
                "thread_first_elevated_date": (
                    thread.first_elevated_date if thread is not None else None
                ),
                "thread_last_elevated_date": (
                    thread.last_elevated_date if thread is not None else None
                ),
                "quiet_observed_day_count": (
                    thread.quiet_observed_days if thread is not None else 0
                ),
            }
        )
    return records


def _window_record(
    runtime: _StoryRuntime,
    policy: FieldStoryPolicy,
    *,
    right_censored: bool,
) -> dict[str, Any]:
    return {
        "story_id": runtime.story_id,
        "field_id": runtime.field_id,
        "crop_instance_id": runtime.crop_instance_id,
        "crop_name": runtime.crop_name,
        "crop_season": runtime.crop_season,
        "first_evidence_date": runtime.first_evidence_date,
        "story_known_time": runtime.story_known_time,
        "confirmed_time": runtime.confirmed_time,
        "last_updated_time": runtime.last_updated_time,
        "story_end_date": None if right_censored else runtime.last_decision_date,
        "story_state": runtime.state,
        "encountered_hazards_json": json.dumps(runtime.encountered_hazards),
        "max_risk_rank": runtime.max_risk_rank,
        "response_status": runtime.response_status,
        "last_decline_source_date": runtime.last_decline_source_date,
        "last_recovery_source_date": runtime.last_recovery_source_date,
        "close_reason": runtime.close_reason or "input_boundary_right_censored",
        "right_censored": right_censored,
        "requires_review": runtime.requires_review,
        "policy_version": policy.version,
    }


def _build_chapters(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_CHAPTER_COLUMNS)
    source = daily.sort_values(["story_id", "decision_date"], kind="mergesort").copy()
    signature_columns = [
        "story_state",
        "stage_bucket",
        "active_hazards_json",
        "open_hazards_json",
        "response_status",
        "coverage_adequate",
    ]
    source["_signature"] = (
        source[signature_columns].astype(str).agg("\x1f".join, axis=1)
    )
    source["_new"] = source.groupby("story_id", sort=False)["_signature"].transform(
        lambda values: values.ne(values.shift()).astype(int)
    )
    source["_chapter"] = source.groupby("story_id", sort=False)["_new"].cumsum()
    records: list[dict[str, Any]] = []
    for (story_id, chapter_number), group in source.groupby(
        ["story_id", "_chapter"], sort=True
    ):
        first = group.iloc[0]
        records.append(
            {
                "story_id": story_id,
                "chapter_number": int(chapter_number),
                "chapter_start_date": group["decision_date"].min(),
                "chapter_end_date": group["decision_date"].max(),
                "story_state": first["story_state"],
                "stage_bucket": first["stage_bucket"],
                "active_hazards_json": first["active_hazards_json"],
                "open_hazards_json": first["open_hazards_json"],
                "response_status": first["response_status"],
                "coverage_adequate": bool(first["coverage_adequate"]),
            }
        )
    return pd.DataFrame(records, columns=_CHAPTER_COLUMNS)


def _normalize_crops(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _CROP_COLUMNS, "crop_day_context_v4")
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["observation_date"] = pd.to_datetime(
        output["observation_date"]
    ).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    natural = ["field_id", "crop_instance_id", "observation_date"]
    if output.duplicated(natural).any():
        raise ValueError("crop_day_context_v4 contains duplicate natural keys")
    return output.sort_values(
        ["field_id", "crop_instance_id", "knowledge_time", "observation_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _PRESSURE_COLUMNS, "field_day_pressure_v4")
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["hazard_family"] = output["hazard_family"].astype(str).str.lower()
    effective_column = (
        "pressure_observation_date"
        if "pressure_observation_date" in output
        else "observation_date"
    )
    if effective_column not in output:
        raise ValueError("field_day_pressure_v4 requires a pressure observation date")
    output["pressure_observation_date"] = pd.to_datetime(
        output[effective_column]
    ).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    output["pressure_observed"] = _coerce_bool(output["pressure_observed"])
    output["pressure_rank"] = pd.to_numeric(output["pressure_rank"], errors="coerce")
    invalid = output["pressure_observed"] & ~output["pressure_rank"].isin(
        [0, 1, 2, 3, 4]
    )
    if invalid.any():
        raise ValueError("Observed pressure rows require an exact rank in [0, 4]")
    output.loc[~output["pressure_observed"], "pressure_rank"] = 0
    output["pressure_rank"] = output["pressure_rank"].astype(int)
    knowledge_date = (
        output["knowledge_time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    output["decision_date"] = pd.concat(
        [output["pressure_observation_date"], knowledge_date], axis=1
    ).max(axis=1)
    natural = [
        "field_id",
        "crop_instance_id",
        "pressure_observation_date",
        "hazard_family",
    ]
    if output.duplicated(natural).any():
        raise ValueError("field_day_pressure_v4 contains duplicate natural keys")
    output = output.sort_values(
        [
            "field_id",
            "crop_instance_id",
            "decision_date",
            "hazard_family",
            "pressure_observation_date",
            "knowledge_time",
        ],
        kind="mergesort",
    )
    return output.drop_duplicates(
        ["field_id", "crop_instance_id", "decision_date", "hazard_family"],
        keep="last",
    ).reset_index(drop=True)


def _normalize_responses(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _RESPONSE_COLUMNS, "field_s2_acquisition_v4")
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["spectral_source_date"] = pd.to_datetime(
        output["spectral_source_date"]
    ).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    output["response_class"] = output["response_class"].astype(str).str.lower()
    accepted = (
        _coerce_bool(output["spectral_usable"])
        & _coerce_bool(output["new_response_evidence"])
        & output["response_class"].isin(POSITIVE_RESPONSES)
    )
    output = output.loc[accepted].copy()
    knowledge_date = (
        output["knowledge_time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    decision_parts = [output["spectral_source_date"], knowledge_date]
    if "crop_assignment_available_at" in output:
        assignment = (
            pd.to_datetime(
                output["crop_assignment_available_at"], errors="coerce", utc=True
            )
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        decision_parts.append(assignment)
    output["decision_date"] = pd.concat(decision_parts, axis=1).max(axis=1)
    key_column = "acquisition_id" if "acquisition_id" in output else None
    if key_column and output[key_column].astype(str).duplicated().any():
        raise ValueError("field_s2_acquisition_v4 contains duplicate acquisition IDs")
    return output.sort_values(
        [
            "field_id",
            "crop_instance_id",
            "decision_date",
            "spectral_source_date",
            "knowledge_time",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _crop_context_as_of(
    crop_history: pd.DataFrame, days: list[date]
) -> dict[date, dict[str, Any]]:
    output: dict[date, dict[str, Any]] = {}
    for decision_date in days:
        day = pd.Timestamp(decision_date)
        known_by = pd.Timestamp(decision_date, tz="UTC") + pd.Timedelta(days=1)
        eligible = crop_history[
            (crop_history["observation_date"] <= day)
            & (crop_history["knowledge_time"] < known_by)
        ]
        if eligible.empty:
            continue
        row = eligible.sort_values(
            ["observation_date", "knowledge_time"], kind="mergesort"
        ).iloc[-1]
        output[decision_date] = row.to_dict()
    return output


def _validate_ownership(
    crops: pd.DataFrame, pressure: pd.DataFrame, responses: pd.DataFrame
) -> None:
    crop_keys = set(zip(crops["field_id"], crops["crop_instance_id"]))
    evidence_keys = set(zip(pressure["field_id"], pressure["crop_instance_id"]))
    evidence_keys |= set(zip(responses["field_id"], responses["crop_instance_id"]))
    missing = sorted(evidence_keys - crop_keys)
    if missing:
        field_id, crop_instance_id = missing[0]
        raise ValueError(
            "Story evidence references unknown crop ownership: "
            f"{field_id}/{crop_instance_id}"
        )


def _reject_conflicting_responses(
    responses: list[dict[str, Any]], decision_date: date
) -> None:
    classes = {str(row["response_class"]) for row in responses}
    if classes & DECLINE_RESPONSES and "recovery" in classes:
        raise ValueError(
            "One decision day cannot assert both fresh decline and recovery: "
            f"{decision_date.isoformat()}"
        )


def _validate_outputs(
    daily: pd.DataFrame,
    chapters: pd.DataFrame,
    windows: pd.DataFrame,
    hazards: pd.DataFrame,
) -> None:
    if daily.duplicated(["story_id", "decision_date"]).any():
        raise RuntimeError("Field-story daily state is not unique by story/day")
    if chapters.duplicated(["story_id", "chapter_number"]).any():
        raise RuntimeError("Field-story chapter numbers are not unique")
    if windows["story_id"].duplicated().any():
        raise RuntimeError("Field-story windows are not unique by story_id")
    if hazards.duplicated(["story_id", "decision_date", "hazard_family"]).any():
        raise RuntimeError("Field-story hazard rows are not unique")
    if not daily.empty:
        story_counts = windows.groupby("story_id").size()
        if not set(daily["story_id"]) <= set(story_counts.index):
            raise RuntimeError("Every daily story requires one window row")
        known = pd.to_datetime(daily["state_known_time"], utc=True)
        decision_end = pd.to_datetime(daily["decision_date"], utc=True) + pd.Timedelta(
            days=1
        )
        if (known >= decision_end).any():
            raise RuntimeError(
                "Story state was emitted before its evidence was knowable"
            )


def _coerce_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError("Boolean evidence columns contain invalid values")
    return normalized.isin({"true", "1"})


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    if "decision_date" in columns:
        candidates = (
            "field_id",
            "crop_instance_id",
            "decision_date",
            "story_id",
            "hazard_family",
        )
    elif "first_evidence_date" in columns:
        candidates = (
            "field_id",
            "crop_instance_id",
            "first_evidence_date",
            "story_id",
        )
    else:
        candidates = ("story_id", "chapter_number")
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values([name for name in candidates if name in columns], kind="mergesort")
        .reset_index(drop=True)
    )


_DAILY_COLUMNS = [
    "story_id",
    "field_id",
    "crop_instance_id",
    "crop_name",
    "crop_season",
    "decision_date",
    "first_evidence_date",
    "story_known_time",
    "confirmed_time",
    "state_known_time",
    "story_state",
    "stage_bucket",
    "active_hazards_json",
    "open_hazards_json",
    "encountered_hazards_json",
    "pressure_rank_by_hazard_json",
    "current_max_risk_rank",
    "max_risk_rank",
    "response_status",
    "last_decline_source_date",
    "last_recovery_source_date",
    "coverage_adequate",
    "requires_review",
    "right_censored",
    "policy_version",
]
_CHAPTER_COLUMNS = [
    "story_id",
    "chapter_number",
    "chapter_start_date",
    "chapter_end_date",
    "story_state",
    "stage_bucket",
    "active_hazards_json",
    "open_hazards_json",
    "response_status",
    "coverage_adequate",
]
_WINDOW_COLUMNS = [
    "story_id",
    "field_id",
    "crop_instance_id",
    "crop_name",
    "crop_season",
    "first_evidence_date",
    "story_known_time",
    "confirmed_time",
    "last_updated_time",
    "story_end_date",
    "story_state",
    "encountered_hazards_json",
    "max_risk_rank",
    "response_status",
    "last_decline_source_date",
    "last_recovery_source_date",
    "close_reason",
    "right_censored",
    "requires_review",
    "policy_version",
]
_HAZARD_COLUMNS = [
    "story_id",
    "field_id",
    "crop_instance_id",
    "decision_date",
    "hazard_family",
    "pressure_observed",
    "pressure_rank",
    "pressure_source_date",
    "pressure_knowledge_time",
    "hazard_active",
    "thread_open",
    "thread_first_elevated_date",
    "thread_last_elevated_date",
    "quiet_observed_day_count",
]


__all__ = [
    "FieldStoryArtifacts",
    "FieldStoryPolicy",
    "build_field_stories",
    "load_field_story_policy",
]
