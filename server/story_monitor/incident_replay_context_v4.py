"""Causal day-major context adapter for a V4-native incident replay.

The V4 evidence release has one pressure row per field/crop/hazard/day and a
separate acquisition ledger.  The legacy V1 machine accepts one dominant
hazard row per day, so feeding V4 rows into it would advance other hazards'
clocks multiple times and make the result input-order dependent.  This module
keeps one clock tick per decision day and evaluates all hazard lanes together.

Episode identity is based on the day evidence became knowable, not merely its
effective source date.  That prevents late evidence from rewriting an episode
that was already visible in an earlier prefix replay.  Effective dates and
record IDs remain on every emitted row for audit and viewer attribution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from typing import Any

import pandas as pd

from .contracts import stable_id
from .incident_policy_v3 import IncidentPolicyV3
from .incident_policy_v4 import IncidentPolicyV4


DECLINE_RESPONSES = frozenset({"medium_decline", "severe_decline"})
POSITIVE_RESPONSES = frozenset({*DECLINE_RESPONSES, "recovery"})
OPEN_EPISODE_STATES = frozenset(
    {"WATCH", "ACTIVE", "SEVERE", "QUIET_PENDING", "RECOVERING"}
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
    "record_id",
    "field_id",
    "crop_instance_id",
    "pressure_observation_date",
    "knowledge_time",
    "hazard_family",
    "pressure_observed",
    "pressure_rank",
    "pressure_band",
}
_S2_COLUMNS = {
    "acquisition_id",
    "field_id",
    "crop_instance_id",
    "spectral_source_date",
    "knowledge_time",
    "crop_assignment_effective_date",
    "crop_assignment_available_at",
    "spectral_usable",
    "new_response_evidence",
    "response_class",
}


@dataclass(frozen=True)
class V4EpisodeReplay:
    daily_episode_state: pd.DataFrame
    episode_windows: pd.DataFrame
    episode_membership: pd.DataFrame
    daily_signals: pd.DataFrame
    diagnostics: dict[str, int]


@dataclass
class _Runtime:
    episode_id: str
    field_id: str
    crop_instance_id: str
    crop_name: str
    crop_season: str
    hazard_family: str
    start_date: date
    state: str
    last_decision_date: date
    last_pressure_source_date: date | None
    last_pressure_record_id: str | None
    max_risk_rank: int = 0
    reportable_day_count: int = 0
    response_day_count: int = 0
    quiet_observed_days: int = 0
    last_decline_source_date: date | None = None
    last_recovery_source_date: date | None = None
    cumulative_knowledge_time: pd.Timestamp | None = None
    close_reason: str | None = None

    @property
    def unresolved_decline(self) -> bool:
        return self.last_decline_source_date is not None and (
            self.last_recovery_source_date is None
            or self.last_recovery_source_date <= self.last_decline_source_date
        )


def replay_daily_episodes_v4(
    crop_days: pd.DataFrame,
    pressure: pd.DataFrame,
    acquisitions: pd.DataFrame,
    *,
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
) -> V4EpisodeReplay:
    """Replay deterministic field/crop/hazard episodes from V4 ledgers.

    Only usable acquisitions with a reference and an exact positive V4
    response enter the state machine.  Missing pressure is emitted as a data
    gap for an open episode but never increments a quiet clock.
    """
    _require_columns(crop_days, _CROP_COLUMNS, "crop_day_context_v4")
    _require_columns(pressure, _PRESSURE_COLUMNS, "field_day_pressure_v4")
    _require_columns(acquisitions, _S2_COLUMNS, "field_s2_acquisition_v4")

    crops = _normalize_crops(crop_days)
    pressures = _normalize_pressure(pressure, source_policy)
    responses, rejected_response_count = _normalize_responses(acquisitions)
    _validate_crop_ownership(crops, pressures, responses)

    crop_lookup = _crop_lookup(crops)
    pressure_groups = {
        key: group
        for key, group in pressures.groupby(
            ["field_id", "crop_instance_id"], sort=True, dropna=False
        )
    }
    response_groups = {
        key: group
        for key, group in responses.groupby(
            ["field_id", "crop_instance_id"], sort=True, dropna=False
        )
    }
    keys = sorted(set(pressure_groups) | set(response_groups))

    daily_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    ambiguous_declines = 0
    ambiguous_recoveries = 0

    quiet_close_days = int(tracker_policy.quiet_close_weeks) * 7
    for key in keys:
        field_id, crop_instance_id = (str(key[0]), str(key[1]))
        group_pressure = pressure_groups.get(key, _empty_pressure())
        group_responses = response_groups.get(key, _empty_responses())
        crop_history = crop_lookup.get(key)
        if crop_history is None or crop_history.empty:
            raise ValueError(
                "V4 replay evidence has no causally available crop context for "
                f"{field_id}/{crop_instance_id}"
            )
        (
            group_daily,
            group_windows,
            group_memberships,
            group_signals,
            group_ambiguous_declines,
            group_ambiguous_recoveries,
        ) = _replay_crop_instance(
            field_id,
            crop_instance_id,
            crop_history,
            group_pressure,
            group_responses,
            source_policy=source_policy,
            tracker_policy=tracker_policy,
            quiet_close_days=quiet_close_days,
        )
        daily_rows.extend(group_daily)
        window_rows.extend(group_windows)
        membership_rows.extend(group_memberships)
        signal_rows.extend(group_signals)
        ambiguous_declines += group_ambiguous_declines
        ambiguous_recoveries += group_ambiguous_recoveries

    daily = _frame(daily_rows, _DAILY_COLUMNS)
    windows = _frame(window_rows, _WINDOW_COLUMNS)
    memberships = _frame(membership_rows, _MEMBERSHIP_COLUMNS)
    signals = _frame(signal_rows, _SIGNAL_COLUMNS)
    _validate_replay_outputs(daily, windows, memberships, signals)
    diagnostics = {
        "daily_episode_state_count": len(daily),
        "episode_count": len(windows),
        "episode_membership_count": len(memberships),
        "daily_signal_count": len(signals),
        "ignored_nonpositive_or_unusable_acquisition_count": rejected_response_count,
        "ambiguous_decline_attribution_count": ambiguous_declines,
        "ambiguous_recovery_attribution_count": ambiguous_recoveries,
    }
    return V4EpisodeReplay(daily, windows, memberships, signals, diagnostics)


def _replay_crop_instance(
    field_id: str,
    crop_instance_id: str,
    crop_history: pd.DataFrame,
    pressure: pd.DataFrame,
    responses: pd.DataFrame,
    *,
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
    quiet_close_days: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
]:
    pressure_by_day = {
        pd.Timestamp(day).date(): group
        for day, group in pressure.groupby("decision_date", sort=True)
    }
    responses_by_day = {
        pd.Timestamp(day).date(): group
        for day, group in responses.groupby("decision_date", sort=True)
    }
    days = sorted(set(pressure_by_day) | set(responses_by_day))
    histories: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=3))
    active: dict[str, _Runtime] = {}
    daily_rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    ambiguous_declines = 0
    ambiguous_recoveries = 0

    for decision_date in days:
        pressure_day = pressure_by_day.get(decision_date, _empty_pressure())
        response_day = responses_by_day.get(decision_date, _empty_responses())
        crop = _crop_as_of(crop_history, decision_date)
        if crop is None:
            raise ValueError(
                "V4 replay would use crop ownership before it was knowable: "
                f"{field_id}/{crop_instance_id}/{decision_date.isoformat()}"
            )

        lanes: dict[str, dict[str, Any]] = {}
        for row in pressure_day.to_dict("records"):
            hazard = str(row["hazard_family"])
            lanes[hazard] = row
            if bool(row["pressure_observed"]):
                histories[hazard].append(int(row["pressure_rank"]))

        response_rows = response_day.sort_values(
            ["spectral_source_date", "knowledge_time", "acquisition_id"],
            kind="mergesort",
        ).to_dict("records")
        targeted: dict[str, list[dict[str, Any]]] = defaultdict(list)
        provisional = {
            hazard: (
                runtime.last_decline_source_date,
                runtime.last_recovery_source_date,
            )
            for hazard, runtime in active.items()
        }
        for response in response_rows:
            response_class = str(response["response_class"])
            if response_class in DECLINE_RESPONSES:
                hazard = _decline_hazard(lanes, active)
                if hazard is None:
                    ambiguous_declines += 1
                    continue
            else:
                unresolved = sorted(
                    hazard
                    for hazard, (decline, recovery) in provisional.items()
                    if decline is not None
                    and (recovery is None or recovery <= decline)
                )
                if len(unresolved) != 1:
                    ambiguous_recoveries += 1
                    continue
                hazard = unresolved[0]
            targeted[hazard].append(response)
            decline, recovery = provisional.get(hazard, (None, None))
            source_date = pd.Timestamp(response["spectral_source_date"]).date()
            if response_class in DECLINE_RESPONSES:
                decline = _latest_date(decline, source_date)
            else:
                recovery = _latest_date(recovery, source_date)
            provisional[hazard] = (decline, recovery)

        candidates = {
            hazard
            for hazard, lane in lanes.items()
            if bool(lane["pressure_observed"])
            and int(lane["pressure_rank"]) >= 2
        } | set(targeted)
        for hazard in sorted(candidates):
            if hazard in active:
                continue
            lane = lanes.get(hazard)
            hazard_responses = targeted.get(hazard, [])
            rank = (
                int(lane["pressure_rank"])
                if lane is not None and bool(lane["pressure_observed"])
                else 0
            )
            response_class = _onset_response_class(hazard_responses)
            onset = _onset_state(histories[hazard], rank, response_class)
            if onset is None:
                continue
            identity = stable_id(
                "episode_v4",
                (
                    tracker_policy.identity_namespace,
                    _effective_policy_sha256(tracker_policy),
                    _effective_policy_sha256(source_policy),
                    field_id,
                    crop_instance_id,
                    hazard,
                    decision_date.isoformat(),
                ),
            )
            active[hazard] = _Runtime(
                episode_id=identity,
                field_id=field_id,
                crop_instance_id=crop_instance_id,
                crop_name=str(crop["crop_name"]),
                crop_season=str(crop["crop_season"]),
                hazard_family=hazard,
                start_date=decision_date,
                state=onset,
                last_decision_date=decision_date,
                last_pressure_source_date=None,
                last_pressure_record_id=None,
            )

        for hazard in sorted(tuple(active)):
            runtime = active[hazard]
            lane = lanes.get(hazard)
            hazard_responses = targeted.get(hazard, [])
            response = _representative_response(hazard_responses)
            _advance_runtime(
                runtime,
                decision_date,
                lane,
                hazard_responses,
                quiet_close_days=quiet_close_days,
            )
            row = _daily_record(runtime, decision_date, crop, lane, response)
            daily_rows.append(row)
            memberships.append(_membership_record(row))
            if runtime.state not in OPEN_EPISODE_STATES:
                windows.append(_window_record(runtime))
                del active[hazard]
                histories[hazard].clear()

        signals.append(
            _signal_record(
                field_id,
                crop_instance_id,
                decision_date,
                crop,
                lanes,
                response_rows,
            )
        )

    windows.extend(_window_record(runtime, right_censored=True) for runtime in active.values())
    return (
        daily_rows,
        windows,
        memberships,
        signals,
        ambiguous_declines,
        ambiguous_recoveries,
    )


def _advance_runtime(
    runtime: _Runtime,
    decision_date: date,
    lane: dict[str, Any] | None,
    responses: list[dict[str, Any]],
    *,
    quiet_close_days: int,
) -> None:
    observed = bool(lane is not None and lane["pressure_observed"])
    rank = int(lane["pressure_rank"]) if observed else 0
    response = _representative_response(responses)
    response_class = str(response["response_class"]) if response else None
    runtime.last_decision_date = decision_date
    runtime.max_risk_rank = max(runtime.max_risk_rank, rank)
    if lane is not None:
        runtime.cumulative_knowledge_time = _max_timestamp(
            runtime.cumulative_knowledge_time, lane["knowledge_time"]
        )
        if observed:
            runtime.last_pressure_source_date = pd.Timestamp(
                lane["pressure_observation_date"]
            ).date()
            runtime.last_pressure_record_id = str(lane["record_id"])
    if responses:
        runtime.response_day_count += 1
        for item in responses:
            runtime.cumulative_knowledge_time = _max_timestamp(
                runtime.cumulative_knowledge_time, item["knowledge_time"]
            )
            item_class = str(item["response_class"])
            source_date = pd.Timestamp(item["spectral_source_date"]).date()
            if item_class in DECLINE_RESPONSES:
                runtime.last_decline_source_date = _latest_date(
                    runtime.last_decline_source_date, source_date
                )
            elif item_class == "recovery":
                runtime.last_recovery_source_date = _latest_date(
                    runtime.last_recovery_source_date, source_date
                )
    if observed and rank >= 2 or any(
        str(item["response_class"]) in DECLINE_RESPONSES for item in responses
    ):
        runtime.reportable_day_count += 1

    if response_class == "severe_decline":
        runtime.state = "SEVERE"
        runtime.quiet_observed_days = 0
        return
    if observed and rank >= 2:
        runtime.quiet_observed_days = 0
        if rank >= 4:
            runtime.state = "SEVERE"
        elif runtime.state not in {"SEVERE"}:
            runtime.state = "ACTIVE" if rank >= 3 else "WATCH"
        return
    if not observed:
        if response_class == "recovery" and runtime.unresolved_decline is False:
            runtime.state = "CLOSED_RECOVERED"
            runtime.close_reason = "later_usable_recovery_acquisition"
        return

    runtime.quiet_observed_days += 1
    if response_class == "recovery" and runtime.unresolved_decline is False:
        runtime.state = "CLOSED_RECOVERED"
        runtime.close_reason = "later_usable_recovery_acquisition"
    elif runtime.quiet_observed_days >= quiet_close_days:
        if runtime.unresolved_decline:
            runtime.state = "RECOVERING"
        else:
            runtime.state = "CLOSED_PRESSURE_QUIET_UNCONFIRMED"
            runtime.close_reason = "observed_pressure_quiet_without_recovery"
    else:
        runtime.state = "QUIET_PENDING"


def _decline_hazard(
    lanes: dict[str, dict[str, Any]], active: dict[str, _Runtime]
) -> str | None:
    ranked = [
        (int(row["pressure_rank"]), hazard)
        for hazard, row in lanes.items()
        if bool(row["pressure_observed"]) and int(row["pressure_rank"]) >= 2
    ]
    if ranked:
        maximum = max(rank for rank, _ in ranked)
        tied = sorted(hazard for rank, hazard in ranked if rank == maximum)
        active_tied = [hazard for hazard in tied if hazard in active]
        if len(tied) == 1:
            return tied[0]
        if len(active_tied) == 1:
            return active_tied[0]
        return None
    if len(active) == 1:
        return next(iter(active))
    return None


def _onset_state(
    history: deque[int], rank: int, response_class: str | None
) -> str | None:
    values = list(history)
    if rank >= 4 or response_class == "severe_decline":
        return "SEVERE"
    if sum(value >= 3 for value in values[-3:]) >= 2:
        return "ACTIVE"
    if response_class == "medium_decline" and rank >= 2:
        return "ACTIVE"
    if sum(value >= 2 for value in values[-3:]) >= 2:
        return "WATCH"
    return None


def _daily_record(
    runtime: _Runtime,
    decision_date: date,
    crop: dict[str, Any],
    lane: dict[str, Any] | None,
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    observed = bool(lane is not None and lane["pressure_observed"])
    response_class = (
        str(response["response_class"])
        if response is not None
        else "no_new_event_response"
    )
    visible_state = (
        "DATA_GAP"
        if lane is not None and not observed and response is None
        and runtime.state in OPEN_EPISODE_STATES
        else runtime.state
    )
    knowledge = _max_timestamp(
        runtime.cumulative_knowledge_time, crop["knowledge_time"]
    )
    return {
        "decision_date": decision_date,
        "observation_date": decision_date,
        "field_id": runtime.field_id,
        "crop_instance_id": runtime.crop_instance_id,
        "crop_name": runtime.crop_name,
        "crop_season": runtime.crop_season,
        "stage_bucket": str(crop["stage_bucket"]),
        "stage_source_date": pd.Timestamp(crop["observation_date"]).date(),
        "episode_id": runtime.episode_id,
        "event_id": runtime.episode_id,
        "hazard_family": runtime.hazard_family,
        "event_state": visible_state,
        "pressure_observed": observed,
        "pressure_rank": int(lane["pressure_rank"]) if observed else None,
        "pressure_band": str(lane["pressure_band"]) if lane is not None else "UNKNOWN",
        "pressure_record_id": str(lane["record_id"]) if lane is not None else None,
        "pressure_source_date": (
            pd.Timestamp(lane["pressure_observation_date"]).date()
            if lane is not None else None
        ),
        "response_class": response_class,
        "fresh_response_evidence": response is not None,
        "response_acquisition_id": (
            str(response["acquisition_id"]) if response is not None else None
        ),
        "response_source_date": (
            pd.Timestamp(response["spectral_source_date"]).date()
            if response is not None else None
        ),
        "knowledge_time": knowledge,
        "max_risk_rank": runtime.max_risk_rank,
        "reportable_day_count": runtime.reportable_day_count,
        "response_day_count": runtime.response_day_count,
        "quiet_observed_day_count": runtime.quiet_observed_days,
        "right_censored": runtime.state in OPEN_EPISODE_STATES,
        "is_data_gap_snapshot": visible_state == "DATA_GAP",
        "close_reason": runtime.close_reason,
    }


def _membership_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": row["field_id"],
        "crop_instance_id": row["crop_instance_id"],
        "episode_id": row["episode_id"],
        "event_id": row["episode_id"],
        "story_cluster_id": row["episode_id"],
        "observation_date": row["decision_date"],
        "event_state": row["event_state"],
        "hazard_signature": row["hazard_family"],
        "daily_pressure_rank": row["pressure_rank"],
        "daily_response_class": row["response_class"],
        "pressure_observed": row["pressure_observed"],
        "knowledge_time": row["knowledge_time"],
        "pressure_record_id": row["pressure_record_id"],
        "response_acquisition_id": row["response_acquisition_id"],
    }


def _window_record(runtime: _Runtime, *, right_censored: bool = False) -> dict[str, Any]:
    closed = not right_censored and runtime.state not in OPEN_EPISODE_STATES
    return {
        "field_id": runtime.field_id,
        "crop_name": runtime.crop_name,
        "crop_season": runtime.crop_season,
        "crop_instance_id": runtime.crop_instance_id,
        "event_id": runtime.episode_id,
        "episode_id": runtime.episode_id,
        "story_cluster_id": runtime.episode_id,
        "event_start_date": runtime.start_date,
        "active_end_date": runtime.last_pressure_source_date,
        "event_end_date": runtime.last_decision_date if closed else None,
        "event_state": runtime.state,
        "max_risk_rank": runtime.max_risk_rank,
        "max_risk_band": _rank_band(runtime.max_risk_rank),
        "hazard_signature": runtime.hazard_family,
        "stage_signature": "unknown",
        "response_signature": (
            "unresolved_decline" if runtime.unresolved_decline else "none"
        ),
        "close_reason": runtime.close_reason or "input_boundary_right_censored",
        "reportable_days": runtime.reportable_day_count,
        "response_day_count": runtime.response_day_count,
        "window_span_days": (runtime.last_decision_date - runtime.start_date).days + 1,
        "right_censored": not closed,
        "knowledge_time": runtime.cumulative_knowledge_time,
    }


def _signal_record(
    field_id: str,
    crop_instance_id: str,
    decision_date: date,
    crop: dict[str, Any],
    lanes: dict[str, dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [row for row in lanes.values() if bool(row["pressure_observed"])]
    dominant = (
        sorted(
            observed,
            key=lambda row: (-int(row["pressure_rank"]), str(row["hazard_family"])),
        )[0]
        if observed else None
    )
    positives = [
        row for row in response_rows if str(row["response_class"]) in POSITIVE_RESPONSES
    ]
    response = _representative_response(positives)
    knowledge_values = [crop["knowledge_time"]]
    knowledge_values.extend(row["knowledge_time"] for row in lanes.values())
    knowledge_values.extend(row["knowledge_time"] for row in positives)
    return {
        "field_id": field_id,
        "crop_instance_id": crop_instance_id,
        "observation_date": decision_date,
        "crop_instance_start_date": pd.Timestamp(
            crop["crop_instance_start_date"]
        ).date(),
        "crop_name": str(crop["crop_name"]),
        "crop_season": str(crop["crop_season"]),
        "crop_stage": str(crop.get("crop_stage_raw") or crop["stage_bucket"]),
        "stage_family": str(crop.get("stage_family_raw") or crop["stage_bucket"]),
        "pressure_observed": dominant is not None,
        "risk_rank": int(dominant["pressure_rank"]) if dominant is not None else 0,
        "risk_band": str(dominant["pressure_band"]) if dominant is not None else "UNKNOWN",
        "hazard_family": (
            str(dominant["hazard_family"]) if dominant is not None else "none"
        ),
        "response_class": (
            str(response["response_class"])
            if response is not None else "no_new_event_response"
        ),
        "new_response_evidence": response is not None,
        "knowledge_time": max(pd.Timestamp(value) for value in knowledge_values),
    }


def _normalize_crops(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["observation_date"] = pd.to_datetime(output["observation_date"]).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    if "crop_instance_start_date" not in output:
        output["crop_instance_start_date"] = output.groupby(
            ["field_id", "crop_instance_id"], sort=False
        )["observation_date"].transform("min")
    else:
        output["crop_instance_start_date"] = pd.to_datetime(
            output["crop_instance_start_date"]
        ).dt.normalize()
    if "crop_stage_raw" not in output:
        output["crop_stage_raw"] = output["stage_bucket"]
    if "stage_family_raw" not in output:
        output["stage_family_raw"] = output["stage_bucket"]
    if output.duplicated(["field_id", "crop_instance_id", "observation_date"]).any():
        raise ValueError("crop_day_context_v4 contains duplicate natural keys")
    return output.sort_values(
        ["field_id", "crop_instance_id", "knowledge_time", "observation_date"],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_pressure(
    frame: pd.DataFrame, source_policy: IncidentPolicyV4
) -> pd.DataFrame:
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["hazard_family"] = output["hazard_family"].astype(str).str.lower()
    invalid_hazards = sorted(set(output["hazard_family"]) - set(source_policy.hazard_families))
    if invalid_hazards:
        raise ValueError("field_day_pressure_v4 has unsupported hazards: " + ", ".join(invalid_hazards))
    output["pressure_observation_date"] = pd.to_datetime(
        output["pressure_observation_date"]
    ).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    output["pressure_observed"] = _coerce_boolean_series(
        output["pressure_observed"]
    )
    output["pressure_rank"] = pd.to_numeric(output["pressure_rank"], errors="coerce")
    bad_rank = output["pressure_observed"] & ~output["pressure_rank"].isin([0, 1, 2, 3, 4])
    if bad_rank.any():
        raise ValueError("Observed V4 pressure rows require an exact rank in [0, 4]")
    output.loc[~output["pressure_observed"], "pressure_rank"] = 0
    output["pressure_rank"] = output["pressure_rank"].astype(int)
    decision = pd.concat(
        [
            output["pressure_observation_date"],
            output["knowledge_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize(),
        ],
        axis=1,
    ).max(axis=1)
    output["decision_date"] = decision
    natural = [
        "field_id", "crop_instance_id", "pressure_observation_date", "hazard_family"
    ]
    if output.duplicated(natural).any():
        raise ValueError("field_day_pressure_v4 contains duplicate natural keys")
    output = output.sort_values(
        [
            "field_id", "crop_instance_id", "decision_date", "hazard_family",
            "pressure_observation_date", "knowledge_time", "record_id",
        ],
        kind="mergesort",
    )
    # Several late effective observations may become knowable on one decision
    # day.  One day is one clock tick; the latest effective observation wins.
    return output.drop_duplicates(
        ["field_id", "crop_instance_id", "decision_date", "hazard_family"],
        keep="last",
    ).reset_index(drop=True)


def _normalize_responses(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = frame.copy()
    output["field_id"] = output["field_id"].astype(str)
    output["crop_instance_id"] = output["crop_instance_id"].astype(str)
    output["spectral_source_date"] = pd.to_datetime(
        output["spectral_source_date"]
    ).dt.normalize()
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], utc=True)
    output["response_class"] = output["response_class"].astype(str).str.lower()
    usable = _coerce_boolean_series(output["spectral_usable"])
    fresh = _coerce_boolean_series(output["new_response_evidence"])
    positive = output["response_class"].isin(POSITIVE_RESPONSES)
    accepted = usable & fresh & positive
    rejected = int((~accepted).sum())
    output = output.loc[accepted].copy()
    decision_values = [
        output["spectral_source_date"],
        output["knowledge_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize(),
    ]
    crop_available = pd.to_datetime(
        output["crop_assignment_available_at"], utc=True
    ).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    decision_values.append(crop_available)
    output["decision_date"] = pd.concat(decision_values, axis=1).max(axis=1)
    natural = ["field_id", "crop_instance_id", "spectral_source_date"]
    if output.duplicated(natural).any():
        raise ValueError("field_s2_acquisition_v4 contains duplicate source dates")
    return (
        output.sort_values(
            [
                "field_id", "crop_instance_id", "decision_date",
                "spectral_source_date", "knowledge_time", "acquisition_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True),
        rejected,
    )


def _validate_crop_ownership(
    crops: pd.DataFrame, pressure: pd.DataFrame, responses: pd.DataFrame
) -> None:
    crop_keys = set(zip(crops["field_id"], crops["crop_instance_id"]))
    evidence_keys = set(zip(pressure["field_id"], pressure["crop_instance_id"])) | set(
        zip(responses["field_id"], responses["crop_instance_id"])
    )
    missing = sorted(evidence_keys - crop_keys)
    if missing:
        sample = ", ".join(f"{field}/{crop}" for field, crop in missing[:5])
        raise ValueError("V4 evidence references crop instances absent from crop context: " + sample)
    effective = pd.to_datetime(
        responses["crop_assignment_effective_date"], errors="coerce"
    ).dt.normalize()
    available = pd.to_datetime(
        responses["crop_assignment_available_at"], errors="coerce", utc=True
    )
    if effective.isna().any() or available.isna().any():
        raise ValueError("Positive S2 evidence requires causal crop assignment provenance")
    if (effective > responses["spectral_source_date"]).any():
        raise ValueError("S2 crop ownership uses a future effective crop assignment")
    if (available > responses["knowledge_time"]).any():
        raise ValueError("S2 crop ownership was not knowable by acquisition knowledge_time")


def _crop_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    output: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in frame.groupby(["field_id", "crop_instance_id"], sort=True):
        copy = group.copy()
        known_date = copy["knowledge_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        copy["decision_date"] = pd.concat([copy["observation_date"], known_date], axis=1).max(axis=1)
        output[(str(key[0]), str(key[1]))] = copy.sort_values(
            ["decision_date", "observation_date", "knowledge_time"], kind="mergesort"
        )
    return output


def _crop_as_of(frame: pd.DataFrame, decision_date: date) -> dict[str, Any] | None:
    eligible = frame[frame["decision_date"].dt.date <= decision_date]
    if eligible.empty:
        return None
    # Arrival order controls when a row becomes eligible, but once eligible the
    # latest effective crop observation remains authoritative.  A late-arriving
    # older row must not roll a known crop stage backward.
    return eligible.sort_values(
        ["observation_date", "knowledge_time"], kind="mergesort"
    ).iloc[-1].to_dict()


def _representative_response(
    responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not responses:
        return None
    severity = {"recovery": 0, "medium_decline": 1, "severe_decline": 2}
    return max(
        responses,
        key=lambda row: (
            pd.Timestamp(row["spectral_source_date"]),
            pd.Timestamp(row["knowledge_time"]),
            severity.get(str(row["response_class"]), -1),
            str(row["acquisition_id"]),
        ),
    )


def _onset_response_class(responses: list[dict[str, Any]]) -> str | None:
    decline = [
        row for row in responses
        if str(row["response_class"]) in DECLINE_RESPONSES
    ]
    response = _representative_response(decline or responses)
    return str(response["response_class"]) if response is not None else None


def _effective_policy_sha256(policy: object) -> str:
    payload = asdict(policy)  # type: ignore[arg-type]
    payload.pop("source_path", None)
    payload.pop("source_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rank_band(rank: int) -> str:
    return {0: "NONE", 1: "LOW", 2: "LOW-MED", 3: "MED-HIGH", 4: "HIGH"}.get(
        int(rank), "UNKNOWN"
    )


def _max_timestamp(left: Any, right: Any) -> pd.Timestamp:
    values = [pd.Timestamp(value) for value in (left, right) if value is not None and not pd.isna(value)]
    if not values:
        raise ValueError("V4 replay state lacks a knowledge timestamp")
    normalized = [value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC") for value in values]
    return max(normalized)


def _latest_date(left: date | None, right: date) -> date:
    return right if left is None else max(left, right)


def _coerce_boolean_series(values: pd.Series) -> pd.Series:
    """Match fail-closed ledger boolean semantics without Python truthiness."""
    truthy = {"1", "t", "true", "y", "yes"}

    def coerce(value: Any) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        return str(value).strip().lower() in truthy

    return values.map(coerce).astype(bool)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _frame(records: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(columns))
    return pd.DataFrame.from_records(records).reindex(columns=list(columns)).sort_values(
        [column for column in ("field_id", "crop_instance_id", "decision_date", "episode_id") if column in columns],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_replay_outputs(
    daily: pd.DataFrame,
    windows: pd.DataFrame,
    memberships: pd.DataFrame,
    signals: pd.DataFrame,
) -> None:
    if daily.duplicated(["episode_id", "decision_date"]).any():
        raise ValueError("V4 daily episode replay is not canonical by episode/day")
    if windows.duplicated(["episode_id"]).any():
        raise ValueError("V4 episode windows contain duplicate episode IDs")
    if memberships.duplicated(["episode_id", "observation_date"]).any():
        raise ValueError("V4 episode membership is not canonical by episode/day")
    if signals.duplicated(["field_id", "crop_instance_id", "observation_date"]).any():
        raise ValueError("V4 daily signal adapter is not canonical by crop/day")
    if not daily.empty:
        missing_weather = ~daily["pressure_observed"].astype(bool)
        invalid_missing = missing_weather & daily["pressure_rank"].notna()
        if invalid_missing.any():
            raise ValueError("Missing V4 pressure was converted to a zero pressure rank")


def _empty_pressure() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_PRESSURE_COLUMNS | {"decision_date"}))


def _empty_responses() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_S2_COLUMNS | {"decision_date"}))


_DAILY_COLUMNS = (
    "decision_date", "observation_date", "field_id", "crop_instance_id",
    "crop_name", "crop_season", "stage_bucket", "stage_source_date",
    "episode_id", "event_id", "hazard_family", "event_state",
    "pressure_observed", "pressure_rank", "pressure_band",
    "pressure_record_id", "pressure_source_date", "response_class",
    "fresh_response_evidence", "response_acquisition_id",
    "response_source_date", "knowledge_time", "max_risk_rank",
    "reportable_day_count", "response_day_count", "quiet_observed_day_count",
    "right_censored", "is_data_gap_snapshot", "close_reason",
)
_WINDOW_COLUMNS = (
    "field_id", "crop_name", "crop_season", "crop_instance_id", "event_id",
    "episode_id", "story_cluster_id", "event_start_date", "active_end_date",
    "event_end_date", "event_state", "max_risk_rank", "max_risk_band",
    "hazard_signature", "stage_signature", "response_signature",
    "close_reason", "reportable_days", "response_day_count",
    "window_span_days", "right_censored", "knowledge_time",
)
_MEMBERSHIP_COLUMNS = (
    "field_id", "crop_instance_id", "episode_id", "event_id",
    "story_cluster_id", "observation_date", "event_state", "hazard_signature",
    "daily_pressure_rank", "daily_response_class", "pressure_observed",
    "knowledge_time", "pressure_record_id", "response_acquisition_id",
)
_SIGNAL_COLUMNS = (
    "field_id", "crop_instance_id", "observation_date",
    "crop_instance_start_date", "crop_name", "crop_season", "crop_stage",
    "stage_family", "pressure_observed", "risk_rank", "risk_band",
    "hazard_family", "response_class", "new_response_evidence",
    "knowledge_time",
)


__all__ = ["V4EpisodeReplay", "replay_daily_episodes_v4"]
