"""Causal orchestration for weekly components into persistent exposures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .incident_tracking_v3 import (
    link_weekly_components,
    score_temporal_candidates,
)


@dataclass(frozen=True)
class ExposureArtifacts:
    assignments: pd.DataFrame
    lineage: pd.DataFrame
    weekly_state: pd.DataFrame


def track_exposures(
    components: pd.DataFrame,
    memberships: pd.DataFrame,
    config: Mapping[str, Any] | Any,
) -> ExposureArtifacts:
    """Link all component weeks in causal order, retaining one latest row per exposure."""
    assignment_columns = [
        "timeline_bucket", "hazard_family", "component_id", "exposure_id",
        "assignment_kind", "previous_component_id", "link_score",
    ]
    lineage_columns = [
        "timeline_bucket", "parent_exposure_id", "child_exposure_id",
        "previous_component_id", "current_component_id", "lineage_type", "score",
    ]
    if components.empty:
        return ExposureArtifacts(
            pd.DataFrame(columns=assignment_columns),
            pd.DataFrame(columns=lineage_columns),
            _empty_weekly_state(),
        )
    _require(components, {"timeline_bucket", "component_id", "hazard_family"}, "components")
    _require(memberships, {"component_id", "field_id", "membership_role"}, "memberships")
    source = components.copy()
    source["timeline_bucket"] = pd.to_datetime(source["timeline_bucket"], errors="raise").dt.normalize()
    if source["component_id"].astype(str).duplicated().any():
        raise ValueError("component_id must be globally unique")
    max_gap = int(_config_value(config, "max_link_gap_weeks", "max_gap_weeks", 2))
    if max_gap < 1:
        raise ValueError("max link gap must be positive")

    latest: dict[str, dict[str, Any]] = {}
    assignment_parts: list[pd.DataFrame] = []
    lineage_parts: list[pd.DataFrame] = []
    for week, current in source.groupby("timeline_bucket", sort=True):
        prior_rows = []
        for exposure_id, row in sorted(latest.items()):
            gap_days = (week - pd.Timestamp(row["timeline_bucket"])).days
            if 0 < gap_days <= max_gap * 7 and gap_days % 7 == 0:
                prior_rows.append({**row, "exposure_id": exposure_id})
        previous = pd.DataFrame(prior_rows, columns=[*source.columns, "exposure_id"])
        previous_ids = set(previous.get("component_id", pd.Series(dtype=str)).astype(str))
        current_ids = set(current["component_id"].astype(str))
        previous_members = memberships[memberships["component_id"].astype(str).isin(previous_ids)]
        current_members = memberships[memberships["component_id"].astype(str).isin(current_ids)]
        scores = score_temporal_candidates(
            previous, current, previous_members, current_members, config
        )
        linked = link_weekly_components(previous, current, scores, config)
        assignments = linked.assignments.copy()
        assignments["timeline_bucket"] = pd.to_datetime(
            assignments["timeline_bucket"], errors="raise"
        ).dt.normalize()
        assignment_parts.append(assignments)
        if not linked.lineage.empty:
            lineage = linked.lineage.copy()
            lineage.insert(0, "timeline_bucket", week)
            lineage_parts.append(lineage)

        for update in linked.previous_updates.to_dict("records"):
            if str(update.get("update_status")) == "merged":
                latest.pop(str(update["exposure_id"]), None)
        rows = current.merge(
            assignments[["component_id", "exposure_id"]],
            on="component_id", how="inner", validate="one_to_one",
        )
        for row in rows.to_dict("records"):
            latest[str(row["exposure_id"])] = row

    assignments = pd.concat(assignment_parts, ignore_index=True)
    assignments = assignments.loc[:, assignment_columns].sort_values(
        ["timeline_bucket", "hazard_family", "component_id"], kind="mergesort"
    ).reset_index(drop=True)
    if assignments["component_id"].duplicated().any():
        raise RuntimeError("tracking assigned a component more than once")
    lineage = (
        pd.concat(lineage_parts, ignore_index=True).loc[:, lineage_columns]
        if lineage_parts else pd.DataFrame(columns=lineage_columns)
    )
    lineage = lineage.sort_values(
        ["timeline_bucket", "lineage_type", "parent_exposure_id", "child_exposure_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    weekly = _build_weekly_state(source, memberships, assignments)
    return ExposureArtifacts(assignments, lineage, weekly)


def _build_weekly_state(
    components: pd.DataFrame,
    memberships: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    weekly = components.merge(
        assignments,
        on=["timeline_bucket", "hazard_family", "component_id"],
        how="inner", validate="one_to_one",
    )
    member_assignments = memberships.merge(
        assignments[["component_id", "exposure_id"]],
        on="component_id", how="inner", validate="many_to_one",
    )
    transitions: dict[tuple[str, str], dict[str, Any]] = {}
    for exposure_id, group in member_assignments.groupby("exposure_id", sort=True):
        previous_fields: set[str] = set()
        previous_week: pd.Timestamp | None = None
        for week, rows in group.groupby("timeline_bucket", sort=True):
            week_ts = pd.Timestamp(week).normalize()
            core = rows[rows["membership_role"].astype(str) == "pressure_core"]
            current_fields = set(core["field_id"].astype(str))
            union = previous_fields | current_fields
            transitions[(str(exposure_id), week_ts.date().isoformat())] = {
                "prior_timeline_bucket": previous_week,
                "persisting_field_count": len(previous_fields & current_fields),
                "entering_field_count": len(current_fields - previous_fields),
                "exiting_field_count": len(previous_fields - current_fields),
                "field_union_count": len(union),
                "field_overlap_jaccard": (
                    len(previous_fields & current_fields) / len(union) if union else 0.0
                ),
                "is_consecutive_week": bool(
                    previous_week is not None and (week_ts - previous_week).days == 7
                ),
            }
            previous_fields, previous_week = current_fields, week_ts
    records = []
    for row in weekly.to_dict("records"):
        week = pd.Timestamp(row["timeline_bucket"]).normalize()
        transition = transitions.get(
            (str(row["exposure_id"]), week.date().isoformat()),
            {
                "prior_timeline_bucket": None, "persisting_field_count": 0,
                "entering_field_count": int(row.get("active_field_count") or 0),
                "exiting_field_count": 0,
                "field_union_count": int(row.get("active_field_count") or 0),
                "field_overlap_jaccard": 0.0, "is_consecutive_week": False,
            },
        )
        records.append(
            {
                **row,
                **transition,
                "evidence_scope": "local_significant_cell_component",
                "is_physical_movement": False,
            }
        )
    output = pd.DataFrame(records)
    if output.duplicated(["exposure_id", "timeline_bucket"]).any():
        raise RuntimeError("one exposure acquired multiple primary components in one week")
    return output.sort_values(
        ["timeline_bucket", "hazard_family", "exposure_id"], kind="mergesort"
    ).reset_index(drop=True)


def _config_value(config: Mapping[str, Any] | Any, primary: str, alias: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(primary, config.get(alias, default))
    return getattr(config, primary, getattr(config, alias, default))


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _empty_weekly_state() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timeline_bucket", "hazard_family", "component_id", "exposure_id",
            "cell_ids_json", "active_field_count", "severe_field_count",
            "persisting_field_count", "entering_field_count", "exiting_field_count",
        ]
    )


__all__ = ["ExposureArtifacts", "track_exposures"]
