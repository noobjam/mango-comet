from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .contracts import MonitorPolicy, OPEN_STATES, iso_date, stable_id


@dataclass
class MachineResult:
    daily_records: pd.DataFrame
    events: pd.DataFrame
    memberships: pd.DataFrame


@dataclass
class EventRuntime:
    event_id: str
    field_id: str
    crop_name: str
    crop_season: str
    crop_instance_id: str
    hazard: str
    start_date: date
    state: str
    last_observation_date: date
    last_pressure_date: date | None
    max_risk_rank: int = 0
    reportable_days: int = 0
    response_day_count: int = 0
    quiet_observed_days: int = 0
    quiet_started_date: date | None = None
    recovering_observed_days: int = 0
    last_attributed_decline_date: date | None = None
    last_attributed_recovery_date: date | None = None
    stages: set[str] = field(default_factory=set)
    responses: set[str] = field(default_factory=set)
    close_reason: str | None = None

    def observe_stats(
        self, row: dict[str, Any], pressure_rank: int, targeted_response: str | None
    ) -> None:
        current_date = _as_date(row["observation_date"])
        self.last_observation_date = current_date
        self.max_risk_rank = max(self.max_risk_rank, int(pressure_rank))
        self.stages.add(str(row["stage_family"]))
        if targeted_response in {"medium_decline", "severe_decline", "recovery"}:
            self.responses.add(targeted_response)
            self.response_day_count += 1
        if targeted_response in {"medium_decline", "severe_decline"}:
            self.last_attributed_decline_date = current_date
        elif targeted_response == "recovery":
            self.last_attributed_recovery_date = current_date
        if pressure_rank >= 2 or targeted_response in {"medium_decline", "severe_decline"}:
            self.reportable_days += 1

    def has_unresolved_decline(self) -> bool:
        return self.last_attributed_decline_date is not None and (
            self.last_attributed_recovery_date is None
            or self.last_attributed_recovery_date < self.last_attributed_decline_date
        )

    def has_recovered_decline(self) -> bool:
        return self.last_attributed_decline_date is not None and not self.has_unresolved_decline()


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _rank_to_band(rank: int) -> str:
    return {0: "NONE", 1: "LOW", 2: "LOW-MED", 3: "MED-HIGH", 4: "HIGH"}.get(
        int(rank), "NONE"
    )


def _requires_review(event: EventRuntime, state: str) -> bool:
    return event.hazard == "unattributed_decline" or state in {
        "SEVERE",
        "RECOVERING",
        "CLOSED_RESPONSE_UNRESOLVED",
    }


def _onset_state(history: deque[int], response: str) -> str | None:
    values = list(history)
    last_three = values[-3:]
    if (values and values[-1] >= 4) or response == "severe_decline":
        return "SEVERE"
    if (
        sum(rank >= 3 for rank in last_three) >= 2
        or (response == "medium_decline" and values and values[-1] >= 2)
    ):
        return "ACTIVE"
    if sum(rank >= 2 for rank in last_three) >= 2 or response == "medium_decline":
        return "WATCH"
    return None


def _gap_mondays(previous: date, current: date, minimum_gap: int) -> list[date]:
    if (current - previous).days <= minimum_gap:
        return []
    cursor = previous + timedelta(days=(7 - previous.weekday()) % 7 or 7)
    output: list[date] = []
    while cursor < current:
        output.append(cursor)
        cursor += timedelta(days=7)
    return output


def _record(
    event: EventRuntime,
    when: date,
    *,
    state: str | None = None,
    gap: bool = False,
    pressure_rank: int = 0,
    pressure_observed: bool = True,
    response: str | None = None,
) -> dict[str, Any]:
    visible_state = state or event.state
    return {
        "field_id": event.field_id,
        "crop_name": event.crop_name,
        "crop_season": event.crop_season,
        "crop_instance_id": event.crop_instance_id,
        "event_id": event.event_id,
        "story_cluster_id": event.event_id,
        "observation_date": when,
        "event_state": visible_state,
        "hazard_signature": event.hazard,
        "max_risk_rank": event.max_risk_rank,
        "max_risk_band": _rank_to_band(event.max_risk_rank),
        "current_risk_rank": int(pressure_rank) if pressure_observed else None,
        "current_risk_band": _rank_to_band(pressure_rank) if pressure_observed else "UNKNOWN",
        "reportable_day_count": event.reportable_days,
        "response_day_count": event.response_day_count,
        "right_censored": visible_state in OPEN_STATES or visible_state == "DATA_GAP",
        "is_data_gap_snapshot": gap,
        "requires_review": (
            _requires_review(event, visible_state) or _requires_review(event, event.state)
        ),
        "daily_pressure_rank": int(pressure_rank),
        "pressure_observed": bool(pressure_observed),
        "daily_response_class": response or "no_new_event_response",
        "last_attributed_decline_date": event.last_attributed_decline_date,
        "last_attributed_recovery_date": event.last_attributed_recovery_date,
    }


def _event_summary(event: EventRuntime, as_of: date) -> dict[str, Any]:
    closed = event.state not in OPEN_STATES
    end_date = event.last_observation_date
    return {
        "field_id": event.field_id,
        "crop_name": event.crop_name,
        "crop_season": event.crop_season,
        "crop_instance_id": event.crop_instance_id,
        "event_id": event.event_id,
        "event_start_date": event.start_date,
        "active_end_date": event.last_pressure_date,
        "event_end_date": end_date if closed else None,
        "event_state": event.state,
        "max_risk_band": _rank_to_band(event.max_risk_rank),
        "max_risk_rank": event.max_risk_rank,
        "hazard_signature": event.hazard,
        "stage_signature": ">".join(sorted(event.stages)) or "unknown",
        "response_signature": ">".join(sorted(event.responses)) or "spectral_missing",
        "close_reason": event.close_reason or "input_boundary_right_censored",
        "reportable_days": event.reportable_days,
        "response_day_count": event.response_day_count,
        "window_span_days": (end_date - event.start_date).days + 1,
        "story_cluster_id": event.event_id,
        "right_censored": not closed,
        "as_of_date": as_of,
        "requires_review": _requires_review(event, event.state),
    }


def _advance_event(
    event: EventRuntime,
    row: dict[str, Any],
    pressure_rank: int,
    pressure_observed: bool,
    targeted_response: str | None,
    promotion_state: str | None,
    policy: MonitorPolicy,
) -> None:
    event.observe_stats(row, pressure_rank, targeted_response)
    current_date = _as_date(row["observation_date"])
    elevated_pressure = pressure_observed and pressure_rank >= 3
    watch_pressure = pressure_observed and pressure_rank >= 2

    if targeted_response == "severe_decline":
        # A fresh, event-attributed severe crop response is independently
        # urgent even after upstream pressure has eased. It is not relabeled as
        # pressure, so active_end_date still means the last pressure date.
        event.state = "SEVERE"
        if elevated_pressure:
            event.last_pressure_date = current_date
        event.quiet_observed_days = 0
        event.quiet_started_date = None
        event.recovering_observed_days = 0
        return

    if elevated_pressure:
        event.last_pressure_date = current_date
        event.quiet_observed_days = 0
        event.quiet_started_date = None
        event.recovering_observed_days = 0
        if event.state == "WATCH":
            if pressure_rank >= 4 or promotion_state == "SEVERE":
                event.state = "SEVERE"
            elif promotion_state == "ACTIVE":
                event.state = "ACTIVE"
        elif event.state == "RECOVERING":
            event.state = "SEVERE" if pressure_rank >= 4 else "ACTIVE"
        elif pressure_rank >= 4:
            event.state = "SEVERE"
        elif event.state != "SEVERE":
            event.state = "ACTIVE"
        return

    if not pressure_observed:
        if (
            event.state == "RECOVERING"
            and targeted_response == "recovery"
            and event.has_recovered_decline()
        ):
            event.state = "CLOSED_RECOVERED"
            event.close_reason = "attributed_recovery_confirmed"
        return

    if event.state == "WATCH" and watch_pressure:
        event.last_pressure_date = current_date
        event.quiet_observed_days = 0
        event.quiet_started_date = None
        if promotion_state in {"ACTIVE", "SEVERE"}:
            event.state = promotion_state
        return

    if event.state == "RECOVERING":
        if targeted_response == "recovery" and event.has_recovered_decline():
            event.state = "CLOSED_RECOVERED"
            event.close_reason = "attributed_recovery_confirmed"
            return
        event.recovering_observed_days += 1
        if event.recovering_observed_days >= policy.max_recovery_days:
            event.state = "CLOSED_RESPONSE_UNRESOLVED"
            event.close_reason = "response_unresolved_observed_deadline"
        return

    event.quiet_observed_days += 1
    event.quiet_started_date = event.quiet_started_date or current_date
    deadline = policy.quiet_days(event.hazard)
    if event.state == "WATCH":
        if event.quiet_observed_days >= deadline:
            if event.has_unresolved_decline():
                event.state = "RECOVERING"
                event.recovering_observed_days = 0
            elif event.has_recovered_decline():
                event.state = "CLOSED_RECOVERED"
                event.close_reason = "attributed_recovery_confirmed"
            else:
                event.state = "CLOSED_WATCH_QUIET"
                event.close_reason = "watch_quiet_observed_deadline"
        return
    if event.state in {"ACTIVE", "SEVERE"}:
        event.state = "QUIET_PENDING"
    if event.state == "QUIET_PENDING" and event.quiet_observed_days >= deadline:
        if event.has_unresolved_decline():
            event.state = "RECOVERING"
            event.recovering_observed_days = 0
        elif event.has_recovered_decline():
            event.state = "CLOSED_RECOVERED"
            event.close_reason = "attributed_recovery_confirmed"
        else:
            event.state = "CLOSED_PRESSURE_QUIET_UNCONFIRMED"
            event.close_reason = "pressure_quiet_without_fresh_recovery_evidence"


def run_state_machine(
    signals: pd.DataFrame, policy: MonitorPolicy, *, as_of_date: date
) -> MachineResult:
    if signals.empty:
        return MachineResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    rows_out: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    instance_meta = (
        signals.groupby(["field_id", "crop_name", "crop_season", "crop_instance_id"], sort=True)
        ["crop_instance_start_date"]
        .min()
        .reset_index()
    )
    instance_meta = instance_meta.sort_values(
        ["field_id", "crop_instance_start_date", "crop_instance_id"], kind="mergesort"
    )
    instance_meta["next_instance_start_date"] = instance_meta.groupby(
        "field_id", sort=False
    )["crop_instance_start_date"].shift(-1)
    next_instance_lookup = dict(
        zip(instance_meta["crop_instance_id"], instance_meta["next_instance_start_date"])
    )

    for instance_id, group in signals.groupby("crop_instance_id", sort=True):
        active: dict[str, EventRuntime] = {}
        histories: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))
        previous_date: date | None = None
        group_rows = group.sort_values("observation_date", kind="mergesort").to_dict("records")
        for row in group_rows:
            current_date = _as_date(row["observation_date"])
            if previous_date is not None:
                for gap_date in _gap_mondays(previous_date, current_date, policy.data_gap_days):
                    rows_out.extend(
                        _record(
                            event,
                            gap_date,
                            state="DATA_GAP",
                            gap=True,
                            pressure_observed=False,
                        )
                        for event in active.values()
                    )
            previous_date = current_date
            row_hazard = str(row["hazard_family"])
            pressure_value = row.get("pressure_observed", True)
            pressure_observed = False if pd.isna(pressure_value) else bool(pressure_value)
            response = str(row["response_class"]) if bool(row["new_response_evidence"]) else None
            response_hazard = None
            if response in {"medium_decline", "severe_decline"}:
                response_hazard = (
                    row_hazard
                    if pressure_observed and row_hazard != "none" and int(row["risk_rank"]) >= 2
                    else "unattributed_decline"
                )
            recovery_hazard = None
            if response == "recovery":
                unresolved = [
                    hazard for hazard, event in active.items() if event.has_unresolved_decline()
                ]
                if len(unresolved) == 1:
                    recovery_hazard = unresolved[0]
            hazards = set(histories) | set(active)
            if pressure_observed and row_hazard != "none":
                hazards.add(row_hazard)
            if response_hazard:
                hazards.add(response_hazard)
            if pressure_observed:
                for hazard in hazards:
                    histories[hazard].append(int(row["risk_rank"]) if hazard == row_hazard else 0)

            candidates = {hazard for hazard in (response_hazard,) if hazard}
            if pressure_observed and row_hazard != "none":
                candidates.add(row_hazard)
            newly_opened: set[str] = set()
            for hazard in sorted(candidates):
                if hazard in active:
                    continue
                targeted = response if hazard == response_hazard else None
                onset = _onset_state(histories[hazard], targeted or "")
                if onset is None:
                    continue
                event_id = stable_id(
                    "event",
                    (
                        row["field_id"],
                        instance_id,
                        hazard,
                        current_date,
                        policy.version,
                        policy.source_sha256,
                    ),
                )
                active[hazard] = EventRuntime(
                    event_id=event_id,
                    field_id=str(row["field_id"]),
                    crop_name=str(row["crop_name"]),
                    crop_season=str(row["crop_season"]),
                    crop_instance_id=str(instance_id),
                    hazard=hazard,
                    start_date=current_date,
                    state=onset,
                    last_observation_date=current_date,
                    last_pressure_date=(
                        current_date
                        if pressure_observed and int(row["risk_rank"]) >= 2
                        else None
                    ),
                )
                newly_opened.add(hazard)

            for hazard, event in list(active.items()):
                pressure_rank = (
                    int(row["risk_rank"])
                    if pressure_observed and hazard == row_hazard
                    else 0
                )
                targeted = response if hazard in {response_hazard, recovery_hazard} else None
                promotion_state = _onset_state(histories[hazard], targeted or "")
                if hazard in newly_opened:
                    event.observe_stats(row, pressure_rank, targeted)
                else:
                    _advance_event(
                        event,
                        row,
                        pressure_rank,
                        pressure_observed,
                        targeted,
                        promotion_state,
                        policy,
                    )
                visible_gap = (
                    not pressure_observed
                    and targeted is None
                    and event.state in OPEN_STATES
                )
                state_record = _record(
                    event,
                    current_date,
                    state="DATA_GAP" if visible_gap else None,
                    gap=visible_gap,
                    pressure_rank=pressure_rank,
                    pressure_observed=pressure_observed,
                    response=targeted,
                )
                rows_out.append(state_record)
                memberships.append(
                    {
                        "field_id": event.field_id,
                        "event_id": event.event_id,
                        "story_cluster_id": event.event_id,
                        "crop_instance_id": event.crop_instance_id,
                        "observation_date": current_date,
                        "event_state": state_record["event_state"],
                        "hazard_signature": event.hazard,
                        "daily_pressure_rank": state_record["daily_pressure_rank"],
                        "daily_response_class": state_record["daily_response_class"],
                        "pressure_observed": state_record["pressure_observed"],
                    }
                )
                if event.state not in OPEN_STATES:
                    events_out.append(_event_summary(event, as_of_date))
                    del active[hazard]
                    histories[hazard].clear()

        next_instance_start = next_instance_lookup.get(instance_id)
        if next_instance_start is not None and not pd.isna(next_instance_start):
            boundary_date = _as_date(next_instance_start)
            for event in active.values():
                event.last_observation_date = boundary_date
                event.state = "CLOSED_SEASON_BOUNDARY"
                event.close_reason = "next_crop_instance_observed"
                rows_out.append(_record(event, boundary_date))
                events_out.append(_event_summary(event, as_of_date))
            active.clear()
        else:
            events_out.extend(_event_summary(event, as_of_date) for event in active.values())

    daily = pd.DataFrame(rows_out)
    if not daily.empty:
        daily["event_state_id"] = daily.apply(
            lambda row: stable_id(
                "state",
                (row["event_id"], iso_date(row["observation_date"]), row["event_state"], 1),
            ),
            axis=1,
        )
        daily["revision"] = 1
    membership_frame = pd.DataFrame(memberships).drop_duplicates() if memberships else pd.DataFrame()
    return MachineResult(daily, pd.DataFrame(events_out), membership_frame)
