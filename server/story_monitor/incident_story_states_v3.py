"""Crop-specific story scaffolds, follow-up evidence, and causal lifecycles.

The production workflow deliberately separates geometry/membership assembly from
lifecycle decisions.  A story is first expanded into crop-specific weekly
footprints, then crop/stage denominators and episode follow-up are computed, and
only then may lifecycle clocks advance.  This prevents another crop in the same
weather footprint from silently closing the story.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import duckdb
import pandas as pd

from .incident_tracking_v3 import (
    advance_incident_lifecycle,
    build_crop_incident_assignments,
    initialize_incident_lifecycle,
    is_terminal_incident_state,
)


@dataclass(frozen=True)
class CropStoryScaffold:
    catalog: pd.DataFrame
    weekly_state: pd.DataFrame
    memberships: pd.DataFrame


@dataclass(frozen=True)
class CropStoryArtifacts:
    catalog: pd.DataFrame
    weekly_state: pd.DataFrame
    memberships: pd.DataFrame
    windows: pd.DataFrame


def build_crop_story_scaffold(
    exposure_weekly_state: pd.DataFrame,
    exposure_assignments: pd.DataFrame,
    component_memberships: pd.DataFrame,
    weekly_cells: pd.DataFrame,
    config: Mapping[str, Any] | Any,
    *,
    through_week: str | None = None,
) -> CropStoryScaffold:
    """Expand exposures into crop-specific, exact-cell weekly story rows.

    The scaffold contains no lifecycle judgment.  Pressure, impact, and watch
    cell sets remain separate and the combined footprint is an exact union of
    observed cells, never an interpolated path or convex hull.
    """
    catalog = build_crop_incident_assignments(
        exposure_assignments, component_memberships, config
    )
    if catalog.empty:
        return CropStoryScaffold(catalog, _empty_scaffold(), _empty_membership())
    memberships = _incident_memberships(
        component_memberships, exposure_assignments, catalog
    )
    exposure = exposure_weekly_state.copy()
    exposure["timeline_bucket"] = pd.to_datetime(
        exposure["timeline_bucket"], errors="raise"
    ).dt.normalize()
    cells = _normalize_cells(weekly_cells)
    available_maxima = [exposure["timeline_bucket"].max()]
    if not cells.empty:
        available_maxima.append(cells["timeline_bucket"].max())
    maximum_week = (
        pd.Timestamp(through_week).normalize()
        if through_week is not None
        else max(value for value in available_maxima if pd.notna(value))
    )

    exposure_groups = {
        str(exposure_id): {
            pd.Timestamp(week).normalize(): rows.iloc[0].to_dict()
            for week, rows in group.groupby("timeline_bucket", sort=True)
        }
        for exposure_id, group in exposure.groupby("exposure_id", sort=False)
    }
    member_groups: dict[str, dict[pd.Timestamp, pd.DataFrame]] = {}
    for incident_id, group in memberships.groupby("incident_id", sort=False):
        member_groups[str(incident_id)] = {
            pd.Timestamp(week).normalize(): rows
            for week, rows in group.groupby("timeline_bucket", sort=True)
        }

    records: list[dict[str, Any]] = []
    cell_size = float(_config_value(config, "grid_cell_size_km", "cell_size_km", 5.0))
    for item in catalog.sort_values(
        ["exposure_id", "crop_name_normalized"], kind="mergesort"
    ).to_dict("records"):
        incident_id = str(item["incident_id"])
        exposure_id = str(item["exposure_id"])
        incident_weeks = member_groups.get(incident_id, {})
        exposure_weeks = exposure_groups.get(exposure_id, {})
        if not incident_weeks:
            continue
        if not exposure_weeks:
            raise ValueError(f"Incident {incident_id} has no exposure weekly state")
        tail_weeks = _story_tail_weeks(config)
        story_weeks = sorted(
            {
                candidate
                for evidence_week in incident_weeks
                for candidate in (
                    evidence_week + pd.Timedelta(days=7 * offset)
                    for offset in range(tail_weeks + 1)
                )
                if candidate <= maximum_week
            }
        )
        hazard_values = {
            str(row.get("hazard_family") or "") for row in exposure_weeks.values()
        }
        if len(hazard_values) != 1:
            raise ValueError(f"Exposure {exposure_id} crosses hazard families")
        hazard = next(iter(hazard_values))
        prior_core: set[str] = set()
        last_combined_cells: set[str] = set()
        for week in story_weeks:
            component = exposure_weeks.get(week)
            current = incident_weeks.get(week, _empty_membership())
            core = current[current["membership_role"].astype(str).eq("pressure_core")]
            watch = current[current["membership_role"].astype(str).eq("watch_frontier")]
            impact = current[current["membership_role"].astype(str).eq("impact_lag")]
            core_fields = set(core["field_id"].astype(str))
            pressure_cells = set(core["grid_id"].dropna().astype(str))
            watch_cells = set(watch["grid_id"].dropna().astype(str))
            impact_cells = set(impact["grid_id"].dropna().astype(str))
            observed_cells = pressure_cells | watch_cells | impact_cells
            if observed_cells:
                last_combined_cells = observed_cells
            footprint = observed_cells or last_combined_cells
            cell_coverage_adequate = _footprint_cells_adequate(
                cells, week, hazard, footprint
            )
            center_lon, center_lat = _footprint_center(footprint, config)
            union = prior_core | core_fields
            stage_distribution = _distribution(
                current.drop_duplicates("crop_instance_id")["stage_bucket"]
                if not current.empty else pd.Series(dtype=str)
            )
            severe_count = core[
                core["event_state"].astype(str).str.upper().eq("SEVERE")
            ]["field_id"].nunique()
            current_component_id = (
                str(component["component_id"])
                if component is not None and not current.empty
                else None
            )
            knowledge_values = pd.to_datetime(
                current.get("knowledge_time", pd.Series(dtype="datetime64[ns]")),
                errors="coerce",
            ).dropna()
            knowledge_time = (
                knowledge_values.max()
                if not knowledge_values.empty
                else week + pd.Timedelta(days=6)
            )
            records.append(
                {
                    "timeline_bucket": week,
                    "incident_id": incident_id,
                    "exposure_id": exposure_id,
                    "crop_name": str(item["crop_name_normalized"]),
                    "hazard_family": hazard,
                    "component_id": current_component_id,
                    "knowledge_time": knowledge_time,
                    "knowledge_time_inferred": knowledge_values.empty,
                    "pressure_core_field_count": len(core_fields),
                    "severe_field_count": int(severe_count),
                    "watch_frontier_field_count": int(watch["field_id"].nunique()),
                    "impact_lag_field_count": int(impact["field_id"].nunique()),
                    "entering_field_count": len(core_fields - prior_core),
                    "persisting_field_count": len(core_fields & prior_core),
                    "exiting_field_count": len(prior_core - core_fields),
                    "field_overlap_jaccard": (
                        len(core_fields & prior_core) / len(union) if union else 0.0
                    ),
                    "stage_distribution": stage_distribution,
                    "stage_bucket_count": len(json.loads(stage_distribution)),
                    "pressure_cell_ids_json": _json_set(pressure_cells),
                    "impact_cell_ids_json": _json_set(impact_cells),
                    "watch_cell_ids_json": _json_set(watch_cells),
                    "footprint_cell_ids_json": _json_set(footprint),
                    "footprint_carried_forward": not observed_cells and bool(footprint),
                    "cell_coverage_adequate": cell_coverage_adequate,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "footprint_area_km2": len(footprint) * cell_size * cell_size,
                    "hazard_intensity": (
                        None if component is None else component.get("max_z_score")
                    ),
                    "is_physical_movement": False,
                }
            )
            prior_core = core_fields
    weekly = pd.DataFrame(records)
    if weekly.empty:
        weekly = _empty_scaffold()
    else:
        weekly = weekly.sort_values(
            ["timeline_bucket", "hazard_family", "crop_name", "incident_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    return CropStoryScaffold(catalog, weekly, memberships)


def _story_tail_weeks(config: Mapping[str, Any] | Any) -> int:
    """Bound post-evidence scaffold rows to every possible closure clock."""
    candidate = int(
        _config_value(
            config,
            "candidate_expiry_observed_weeks",
            "candidate_expiry_observed_weeks",
            2,
        )
    )
    quiet = int(
        _config_value(config, "quiet_close_weeks", "quiet_observed_weeks", 2)
    )
    recovery = int(
        _config_value(
            config,
            "recovery_grace_weeks",
            "maximum_recovery_observed_weeks",
            4,
        )
    )
    data_gap = int(
        _config_value(
            config, "maximum_data_gap_weeks", "maximum_data_gap_weeks", 4
        )
    )
    values = (candidate, quiet + recovery, data_gap)
    if any(value < 1 for value in values):
        raise ValueError("Story closure horizons must be positive")
    return max(values)


def build_incident_followup_evidence(
    lanes_path: Path,
    scaffold_weekly_state: pd.DataFrame,
    incident_memberships: pd.DataFrame,
    *,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Follow known episode IDs after pressure disappears from a component.

    This keeps recovery attributable to the episode that seeded the incident;
    it does not reuse an unrelated field-level recovery signal.
    """
    columns = [
        "timeline_bucket", "incident_id", "field_id", "crop_instance_id",
        "episode_id", "hazard_family", "event_state", "response_class",
        "stage_bucket", "knowledge_time", "fresh_decline_evidence",
        "fresh_recovery_evidence",
    ]
    if scaffold_weekly_state.empty or incident_memberships.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "incident_id", "timeline_bucket", "field_id", "crop_instance_id",
        "episode_id", "hazard_family",
    }
    missing = sorted(required - set(incident_memberships.columns))
    if missing:
        raise ValueError("incident memberships are missing follow-up keys: " + ", ".join(missing))
    lanes_path = lanes_path.expanduser().resolve()
    if not lanes_path.is_file():
        raise FileNotFoundError(f"Missing event-week lanes: {lanes_path}")
    seeds = (
        incident_memberships.loc[:, list(required)]
        .assign(
            timeline_bucket=lambda frame: pd.to_datetime(
                frame["timeline_bucket"], errors="raise"
            ).dt.date,
            episode_id=lambda frame: frame["episode_id"].astype(str),
        )
    )
    seeds = seeds[~seeds["episode_id"].isin({"", "none", "nan", "unknown_episode"})]
    if seeds.empty:
        return pd.DataFrame(columns=columns)
    seeds = seeds.groupby(
        ["incident_id", "field_id", "crop_instance_id", "episode_id", "hazard_family"],
        as_index=False,
        sort=True,
    )["timeline_bucket"].min().rename(columns={"timeline_bucket": "first_seed_week"})
    weeks = scaffold_weekly_state[["incident_id", "timeline_bucket"]].copy()
    weeks["timeline_bucket"] = pd.to_datetime(
        weeks["timeline_bucket"], errors="raise"
    ).dt.date
    weeks = weeks.drop_duplicates()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved)])
        connection.register("incident_seeds", seeds)
        connection.register("story_weeks", weeks)
        connection.read_parquet(str(lanes_path)).create_view("event_lanes")
        lane_columns = {
            str(row[0]) for row in connection.execute("DESCRIBE event_lanes").fetchall()
        }
        response_name = (
            "signal_response_class"
            if "signal_response_class" in lane_columns
            else "daily_response_class"
        )
        if response_name not in lane_columns:
            response_sql = "CAST('' AS VARCHAR)"
        else:
            response_sql = f"LOWER(COALESCE(CAST(l.{response_name} AS VARCHAR), ''))"
        fresh_sql = (
            "COALESCE(TRY_CAST(l.fresh_response_evidence AS BOOLEAN), FALSE)"
            if "fresh_response_evidence" in lane_columns
            else "FALSE"
        )
        stage_sql = (
            "COALESCE(CAST(l.stage_bucket AS VARCHAR), 'unknown')"
            if "stage_bucket" in lane_columns else "CAST('unknown' AS VARCHAR)"
        )
        result = connection.execute(
            f"""
            WITH matched AS (
                SELECT
                    w.timeline_bucket, s.incident_id,
                    CAST(l.field_id AS VARCHAR) AS field_id,
                    CAST(l.crop_instance_id AS VARCHAR) AS crop_instance_id,
                    CAST(l.event_id AS VARCHAR) AS episode_id,
                    CAST(l.hazard_family AS VARCHAR) AS hazard_family,
                    UPPER(CAST(l.event_state AS VARCHAR)) AS event_state,
                    {stage_sql} AS stage_bucket,
                    CAST(l.snapshot_as_of_date AS DATE) AS knowledge_time,
                    {response_sql} AS response_class,
                    {fresh_sql} AS fresh_response_evidence
                FROM story_weeks w
                JOIN incident_seeds s
                  ON s.incident_id = w.incident_id
                 AND w.timeline_bucket >= s.first_seed_week
                JOIN event_lanes l
                  ON CAST(l.timeline_bucket AS DATE) = w.timeline_bucket
                 AND CAST(l.event_id AS VARCHAR) = s.episode_id
                 AND CAST(l.field_id AS VARCHAR) = s.field_id
                 AND CAST(l.crop_instance_id AS VARCHAR) = s.crop_instance_id
                 AND CAST(l.hazard_family AS VARCHAR) = s.hazard_family
            )
            SELECT
                timeline_bucket, incident_id, field_id, crop_instance_id,
                episode_id, hazard_family,
                ARG_MAX(event_state,
                    CASE event_state
                        WHEN 'SEVERE' THEN 6 WHEN 'ACTIVE' THEN 5
                        WHEN 'QUIET_PENDING' THEN 4 WHEN 'RECOVERING' THEN 3
                        WHEN 'DATA_GAP' THEN 2 ELSE 1 END
                ) AS event_state,
                ARG_MAX(stage_bucket,
                    CASE event_state
                        WHEN 'SEVERE' THEN 6 WHEN 'ACTIVE' THEN 5
                        WHEN 'QUIET_PENDING' THEN 4 WHEN 'RECOVERING' THEN 3
                        WHEN 'DATA_GAP' THEN 2 ELSE 1 END
                ) AS stage_bucket,
                MAX(knowledge_time) AS knowledge_time,
                CASE
                    WHEN BOOL_OR(fresh_response_evidence AND response_class = 'severe_decline')
                        THEN 'severe_decline'
                    WHEN BOOL_OR(fresh_response_evidence AND response_class = 'medium_decline')
                        THEN 'medium_decline'
                    WHEN BOOL_OR(fresh_response_evidence AND response_class = 'recovery')
                        THEN 'recovery'
                    ELSE 'no_new_event_response'
                END AS response_class,
                BOOL_OR(fresh_response_evidence AND response_class IN
                    ('medium_decline', 'severe_decline')) AS fresh_decline_evidence,
                BOOL_OR(fresh_response_evidence AND response_class = 'recovery')
                    AS fresh_recovery_evidence
            FROM matched
            GROUP BY 1, 2, 3, 4, 5, 6
            ORDER BY 1, 2, 3, 5
            """
        ).fetchdf()
    finally:
        connection.close()
    return result.loc[:, columns]


def finalize_crop_story_artifacts(
    scaffold: CropStoryScaffold,
    stage_summary: pd.DataFrame,
    config: Mapping[str, Any] | Any,
    *,
    followup_evidence: pd.DataFrame | None = None,
    incident_lineage: pd.DataFrame | None = None,
    weekly_cells: pd.DataFrame | None = None,
) -> CropStoryArtifacts:
    """Advance all crop stories in causal week order.

    Week-major scheduling is essential here: unresolved crop-response episodes
    may cross a split/merge edge or receive an exact direct claim from another
    current component.  Ownership moves before current-week evidence is
    applied, so a recovery on the new owner can resolve prior evidence;
    terminal segments are cleared immediately and cannot leak into recurrence.
    """
    if scaffold.catalog.empty:
        return CropStoryArtifacts(
            scaffold.catalog, _empty_weekly(), scaffold.memberships, _empty_windows()
        )
    coverage = _coverage_lookup(stage_summary)
    followup = _normalize_followup(followup_evidence)
    lineage = _normalize_lineage(incident_lineage)
    normalized_weekly_cells = (
        _normalize_cells(weekly_cells) if weekly_cells is not None else None
    )
    member_weeks = {
        (str(incident_id), pd.Timestamp(week).normalize()): rows
        for (incident_id, week), rows in scaffold.memberships.groupby(
            ["incident_id", "timeline_bucket"], sort=False
        )
    }
    story_rows_by_key = {
        (str(row["incident_id"]), pd.Timestamp(row["timeline_bucket"]).normalize()): row
        for row in scaffold.weekly_state.to_dict("records")
    }
    followup_by_week = {
        pd.Timestamp(week).normalize(): rows
        for week, rows in followup.groupby("timeline_bucket", sort=False)
    }
    lineage_by_week = {
        pd.Timestamp(week).normalize(): rows
        for week, rows in lineage.groupby("timeline_bucket", sort=False)
    }
    lineage_by_parent = {
        str(parent): rows.sort_values(
            ["timeline_bucket", "child_incident_id"], kind="mergesort"
        )
        for parent, rows in lineage.groupby("parent_incident_id", sort=False)
    } if not lineage.empty else {}
    split_weeks = _lineage_event_weeks(lineage, "split")
    merge_weeks = _lineage_event_weeks(lineage, "merge")

    catalog_items = {
        str(item["incident_id"]): item
        for item in scaffold.catalog.sort_values(
            ["exposure_id", "crop_name_normalized"], kind="mergesort"
        ).to_dict("records")
    }
    runtimes: dict[str, dict[str, Any]] = {}
    for base_incident_id, item in catalog_items.items():
        base_story = scaffold.weekly_state[
            scaffold.weekly_state["incident_id"].astype(str).eq(base_incident_id)
        ]
        if base_story.empty:
            continue
        runtimes[base_incident_id] = {
            "item": item,
            "hazard": str(base_story.iloc[0]["hazard_family"]),
            "final_week": pd.Timestamp(base_story["timeline_bucket"].max()).normalize(),
            "segment_index": 0,
            "incident_id": None,
            "segment_item": None,
            "lifecycle": None,
            "unresolved": {},
            "recovered": set(),
            "seed_keys": set(),
            "seed_weeks": {},
            "story_rows": [],
        }

    weekly_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    catalog_records: list[dict[str, Any]] = []
    carried_membership_records: list[dict[str, Any]] = []
    owned_followup_records: list[dict[str, Any]] = []
    emitted_keys: set[tuple[str, pd.Timestamp]] = set()
    emitted_segment_by_base_week: dict[tuple[str, pd.Timestamp], str] = {}
    redirects: dict[tuple[str, tuple[str, str, str]], str] = {}
    all_weeks = sorted({key[1] for key in story_rows_by_key})

    for week in all_weeks:
        bases_this_week = sorted(
            base for base in runtimes if (base, week) in story_rows_by_key
        )
        current_by_base = {
            base: member_weeks.get((base, week), _empty_membership())
            for base in bases_this_week
        }

        # Admit segments only from pressure/impact evidence in this prefix.
        # WATCH and follow-up rows cannot create a story by themselves.
        for base in bases_this_week:
            runtime = runtimes[base]
            current = current_by_base[base]
            if runtime["incident_id"] is None and _segment_start_evidence(current):
                segment_index = int(runtime["segment_index"])
                incident_id = (
                    base if segment_index == 0
                    else _recurrence_incident_id(base, week)
                )
                onset_row = story_rows_by_key[(base, week)]
                onset_cells = (
                    _parse_json_set(onset_row.get("pressure_cell_ids_json"))
                    | _parse_json_set(onset_row.get("watch_cell_ids_json"))
                    | _parse_json_set(onset_row.get("impact_cell_ids_json"))
                )
                if not onset_cells:
                    onset_cells = _parse_json_set(
                        onset_row.get("footprint_cell_ids_json")
                    )
                _, _, onset_adequate = _coverage_for_story_cells(
                    onset_row,
                    incident_id,
                    base,
                    week,
                    runtime["hazard"],
                    onset_cells,
                    coverage,
                    normalized_weekly_cells,
                    config,
                )
                if not onset_adequate:
                    continue
                segment_item = {
                    **runtime["item"],
                    "incident_id": incident_id,
                    "base_incident_id": base,
                    "segment_index": segment_index,
                    "segment_start_week": week,
                }
                runtime.update(
                    {
                        "incident_id": incident_id,
                        "segment_item": segment_item,
                        "lifecycle": None,
                        "unresolved": {},
                        "recovered": set(),
                        "seed_keys": set(),
                        "seed_weeks": {},
                        "story_rows": [],
                    }
                )
                catalog_records.append(segment_item)
            if runtime["incident_id"] is not None:
                for row in current.to_dict("records"):
                    evidence_key = _evidence_key(row)
                    runtime["seed_keys"].add(evidence_key)
                    runtime["seed_weeks"][evidence_key] = max(
                        pd.Timestamp(
                            runtime["seed_weeks"].get(evidence_key, week)
                        ).normalize(),
                        week,
                    )

        _transfer_lineage_registries_for_week(
            week,
            lineage_by_week.get(week, lineage.iloc[0:0]),
            runtimes,
            current_by_base,
            redirects,
        )

        # Impact-only memberships are intentionally excluded from physical
        # exposure tracking.  Consequently, an exact crop-response episode can
        # move to another current component without a split/merge edge.  A
        # direct current unresolved claim is stronger than stale carried
        # ownership, so hand the registry entry over atomically before either
        # story consumes this week's response evidence.
        _reconcile_direct_unresolved_claims(
            week,
            runtimes,
            current_by_base,
            redirects,
            lineage_by_week.get(week, lineage.iloc[0:0]),
        )

        owned_followup_by_key: dict[
            tuple[str, tuple[str, str, str]], dict[str, Any]
        ] = {}
        direct_owners = _direct_claim_owner_lookup(
            week, runtimes, current_by_base
        )
        unresolved_owners = _unresolved_owner_lookup(runtimes)
        recovered_owners = _recovered_owner_lookup(runtimes)
        seed_owners = _causal_seed_owner_lookup(runtimes)
        live_owners = {
            base
            for base in bases_this_week
            if runtimes[base].get("incident_id") is not None
        }

        registry_current_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        actionable: dict[
            tuple[str, tuple[str, str, str]],
            list[tuple[str, str, dict[str, Any]]],
        ] = defaultdict(list)
        for base in bases_this_week:
            for row in current_by_base[base].to_dict("records"):
                if _row_reports_fresh_recovery(row):
                    evidence_key = _evidence_key(row)
                    owner_key = (
                        str(row.get("hazard_family") or ""), evidence_key
                    )
                    actionable[owner_key].append(("current", base, row))
                else:
                    registry_current_records[base].append(row)

        for row in followup_by_week.get(week, followup.iloc[0:0]).to_dict("records"):
            if not bool(row.get("fresh_decline_evidence")) and not bool(
                row.get("fresh_recovery_evidence")
            ):
                continue
            source = str(row["incident_id"])
            evidence_key = _evidence_key(row)
            owner_key = (str(row.get("hazard_family") or ""), evidence_key)
            actionable[owner_key].append(("followup", source, row))

        for owner_key, rows in sorted(actionable.items()):
            sources = {source for _, source, _ in rows}
            owner = _resolve_episode_owner(
                week,
                sources,
                owner_key,
                direct_owners,
                unresolved_owners,
                recovered_owners,
                redirects,
                seed_owners,
                live_owners,
            )
            if owner is None:
                continue
            runtime = runtimes.get(owner)
            if runtime is None or owner not in live_owners:
                raise ValueError(
                    "Evidence owner has no active causal story row: "
                    f"week={week.date()}, owner={owner}, evidence={owner_key[1]}"
                )
            _establish_episode_seed_owner(
                owner_key, owner, week, runtimes, redirects
            )
            evidence_key = owner_key[1]
            for kind, source, row in rows:
                if owner != source:
                    redirects[(source, evidence_key)] = owner
                if kind == "current":
                    if owner == source:
                        registry_current_records[owner].append(row)
                    else:
                        remapped = _current_recovery_as_followup(row, owner)
                        followup_key = (owner, evidence_key)
                        previous = owned_followup_by_key.get(followup_key)
                        if previous is not None:
                            _assert_equivalent_followup(previous, remapped)
                        else:
                            owned_followup_by_key[followup_key] = remapped
                    continue
                remapped = {**row, "incident_id": owner}
                followup_key = (owner, evidence_key)
                previous = owned_followup_by_key.get(followup_key)
                if previous is not None:
                    _assert_equivalent_followup(previous, remapped)
                else:
                    owned_followup_by_key[followup_key] = remapped

        registry_current_by_base = {
            base: pd.DataFrame(
                registry_current_records.get(base, []),
                columns=scaffold.memberships.columns,
            )
            for base in bases_this_week
        }

        owned_followup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (owner, _), row in sorted(owned_followup_by_key.items()):
            owned_followup[owner].append(row)
            owned_followup_records.append(row)

        for base in bases_this_week:
            runtime = runtimes[base]
            incident_id = runtime["incident_id"]
            segment_item = runtime["segment_item"]
            if incident_id is None or segment_item is None:
                continue
            base_row = story_rows_by_key[(base, week)]
            current_members = registry_current_by_base[base]
            current_followup = pd.DataFrame(
                owned_followup.get(base, []), columns=followup.columns
            )
            decline_fields, recovery_fields = _apply_impact_registry(
                runtime["unresolved"],
                runtime["recovered"],
                current_members,
                current_followup,
            )
            _assert_unique_unresolved_owners(runtimes)
            _recovered_owner_lookup(runtimes)

            unresolved_field_count = _registry_field_count(runtime["unresolved"])
            recovered_field_count = _evidence_field_count(runtime["recovered"])
            fresh_decline_field_count = _evidence_field_count(decline_fields)
            fresh_recovery_field_count = _evidence_field_count(recovery_fields)
            pressure_cells = _parse_json_set(base_row.get("pressure_cell_ids_json"))
            watch_cells = _parse_json_set(base_row.get("watch_cell_ids_json"))
            current_impact_cells = _parse_json_set(
                base_row.get("impact_cell_ids_json")
            )
            carried_impact_cells = {
                str(evidence.get("grid_id"))
                for evidence in runtime["unresolved"].values()
                if evidence.get("grid_id")
            }
            impact_cells = current_impact_cells | carried_impact_cells
            current_role_cells = pressure_cells | watch_cells | current_impact_cells
            combined_cells = pressure_cells | watch_cells | impact_cells
            if not combined_cells:
                combined_cells = _parse_json_set(
                    base_row.get("footprint_cell_ids_json")
                )
            (
                dynamic_cell_coverage_adequate,
                crop_coverage,
                adequate,
            ) = _coverage_for_story_cells(
                base_row,
                incident_id,
                base,
                week,
                runtime["hazard"],
                combined_cells,
                coverage,
                normalized_weekly_cells,
                config,
            )
            season_boundary = bool(
                adequate
                and int(base_row["pressure_core_field_count"]) == 0
                and crop_coverage["monitored_crop_instance_count"] > 0
                and crop_coverage["off_season_monitored_crop_instance_count"]
                == crop_coverage["monitored_crop_instance_count"]
            )
            maximum_data_gap_weeks = int(
                _config_value(
                    config,
                    "maximum_data_gap_weeks",
                    "maximum_data_gap_weeks",
                    4,
                )
            )
            projected_gap_streak = int(
                (runtime["lifecycle"] or {}).get("coverage_gap_streak", 0)
            ) + (0 if adequate else 1)
            merge_target = _merge_target(
                lineage_by_parent.get(base, pd.DataFrame()), week
            )
            data_censored = bool(
                not adequate
                and week == runtime["final_week"]
                and projected_gap_streak >= maximum_data_gap_weeks
                and merge_target is None
            )
            observation = {
                "timeline_bucket": week,
                "adequate_coverage": adequate,
                "component_present": int(base_row["pressure_core_field_count"]) > 0,
                "story_evidence_present": bool(
                    int(base_row["pressure_core_field_count"]) > 0
                    or unresolved_field_count > 0
                    or fresh_decline_field_count > 0
                ),
                "confirmation_support_present": bool(
                    int(base_row["pressure_core_field_count"]) > 0
                    or int(base_row.get("impact_lag_field_count", 0)) > 0
                    or fresh_decline_field_count > 0
                ),
                "severe_field_count": int(base_row["severe_field_count"]),
                "fresh_decline_field_count": fresh_decline_field_count,
                "fresh_recovery_field_count": fresh_recovery_field_count,
                "impact_field_count": unresolved_field_count,
                "recovered_impact_field_count": recovered_field_count,
                "recovery_evidence": bool(runtime["recovered"]),
                "merged_into_incident_id": merge_target,
                "season_boundary": season_boundary,
                "data_censored": data_censored,
            }
            lifecycle = (
                initialize_incident_lifecycle(
                    incident_id, runtime["hazard"], observation, config
                )
                if runtime["lifecycle"] is None
                else advance_incident_lifecycle(
                    runtime["lifecycle"], observation, config
                )
            )
            runtime["lifecycle"] = lifecycle
            center_lon, center_lat = _footprint_center(combined_cells, config)
            cell_size = float(
                _config_value(config, "grid_cell_size_km", "cell_size_km", 5.0)
            )
            record = {
                **base_row,
                "incident_id": incident_id,
                "base_incident_id": base,
                "segment_index": int(runtime["segment_index"]),
                "pressure_cell_ids_json": _json_set(pressure_cells),
                "watch_cell_ids_json": _json_set(watch_cells),
                "impact_cell_ids_json": _json_set(impact_cells),
                "footprint_cell_ids_json": _json_set(combined_cells),
                "footprint_carried_forward": bool(combined_cells)
                and not bool(current_role_cells),
                "center_lon": center_lon,
                "center_lat": center_lat,
                "footprint_area_km2": len(combined_cells) * cell_size * cell_size,
                "incident_state": lifecycle["incident_state"],
                "current_state": lifecycle["incident_state"],
                "first_evidence_week": lifecycle.get("first_evidence_week"),
                "confirmed_week": lifecycle.get("confirmed_week"),
                "pressure_off_week": lifecycle.get("pressure_off_week"),
                "recovered_week": lifecycle.get("recovered_week"),
                "closed_week": lifecycle.get("closed_week"),
                "merged_into_incident_id": lifecycle.get("merged_into_incident_id"),
                "unresolved_carried_field_count": unresolved_field_count,
                "recovered_field_count": recovered_field_count,
                "fresh_decline_field_count": fresh_decline_field_count,
                "fresh_recovery_field_count": fresh_recovery_field_count,
                "coverage_monitored_field_count": crop_coverage["monitored_field_count"],
                "coverage_evaluable_field_count": crop_coverage["evaluable_field_count"],
                "coverage_monitored_crop_instance_count": crop_coverage[
                    "monitored_crop_instance_count"
                ],
                "coverage_evaluable_crop_instance_count": crop_coverage[
                    "evaluable_crop_instance_count"
                ],
                "coverage_adequate": adequate,
                "cell_coverage_adequate": dynamic_cell_coverage_adequate,
                "coverage_missing_cell_count": crop_coverage[
                    "coverage_missing_cell_count"
                ],
                "season_boundary_observed": season_boundary,
                "data_censored_at_boundary": (
                    lifecycle["incident_state"] == "CLOSED_DATA_CENSORED"
                ),
                "split_count": _lineage_count_through(
                    split_weeks,
                    base,
                    week,
                    since=segment_item["segment_start_week"],
                ),
                "merge_count": _lineage_count_through(
                    merge_weeks,
                    base,
                    week,
                    since=segment_item["segment_start_week"],
                ),
                "right_censored": not is_terminal_incident_state(
                    lifecycle["incident_state"]
                ),
                "relapse_count": int(lifecycle.get("relapse_count", 0)),
                "data_gap_count": int(lifecycle.get("data_gap_count", 0)),
                "coverage_gap_streak": int(
                    lifecycle.get("coverage_gap_streak", 0)
                ),
            }
            weekly_records.append(record)
            runtime["story_rows"].append(record)
            emitted_keys.add((incident_id, week))
            emitted_segment_by_base_week[(base, week)] = incident_id
            present_evidence = {
                _evidence_key(row) for row in current_members.to_dict("records")
            }
            for evidence_key, evidence in sorted(runtime["unresolved"].items()):
                if evidence_key not in present_evidence:
                    carried_membership_records.append(
                        _carried_membership_record(
                            segment_item,
                            runtime["hazard"],
                            week,
                            evidence_key,
                            evidence,
                        )
                    )

            if is_terminal_incident_state(lifecycle["incident_state"]):
                window_records.append(
                    _story_window(
                        segment_item,
                        runtime["hazard"],
                        lifecycle,
                        runtime["story_rows"],
                    )
                )
                for redirect_key, owner in list(redirects.items()):
                    if owner == base:
                        redirects.pop(redirect_key, None)
                runtime.update(
                    {
                        "segment_index": int(runtime["segment_index"]) + 1,
                        "incident_id": None,
                        "segment_item": None,
                        "lifecycle": None,
                        "unresolved": {},
                        "recovered": set(),
                        "seed_keys": set(),
                        "seed_weeks": {},
                        "story_rows": [],
                    }
                )

    for runtime in runtimes.values():
        if (
            runtime["incident_id"] is not None
            and runtime["segment_item"] is not None
            and runtime["lifecycle"] is not None
        ):
            window_records.append(
                _story_window(
                    runtime["segment_item"],
                    runtime["hazard"],
                    runtime["lifecycle"],
                    runtime["story_rows"],
                )
            )

    _remap_merge_targets(
        weekly_records, window_records, emitted_segment_by_base_week
    )

    weekly = pd.DataFrame(weekly_records)
    if weekly.empty:
        weekly = _empty_weekly()
    else:
        weekly = weekly.sort_values(
            ["timeline_bucket", "hazard_family", "crop_name", "incident_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    windows = pd.DataFrame(window_records)
    if windows.empty:
        windows = _empty_windows()
    else:
        windows = windows.sort_values("incident_id", kind="mergesort").reset_index(drop=True)
    memberships = _remap_incident_weeks(
        scaffold.memberships, emitted_segment_by_base_week
    )
    owned_followup_frame = (
        pd.DataFrame(owned_followup_records, columns=followup.columns)
        if owned_followup_records else followup.iloc[0:0].copy()
    )
    remapped_followup = _remap_incident_weeks(
        owned_followup_frame, emitted_segment_by_base_week
    )
    if carried_membership_records:
        memberships = pd.concat(
            [memberships, pd.DataFrame(carried_membership_records)],
            ignore_index=True,
        )
    memberships = _augment_followup_memberships(
        memberships, remapped_followup, emitted_keys
    )
    if memberships.empty:
        memberships = _empty_membership()
    else:
        memberships = memberships[
            [
                (str(incident_id), pd.Timestamp(week).normalize()) in emitted_keys
                for incident_id, week in zip(
                    memberships["incident_id"], memberships["timeline_bucket"]
                )
            ]
        ].sort_values(
            ["timeline_bucket", "incident_id", "membership_role", "field_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        _assert_unique_membership_episode_owners(memberships)
    catalog = pd.DataFrame(catalog_records)
    if not catalog.empty and not memberships.empty:
        counts = memberships.groupby("incident_id", as_index=False, sort=False).agg(
            field_count=("field_id", "nunique"),
            episode_count=("episode_id", "nunique"),
        )
        catalog = catalog.drop(
            columns=["field_count", "episode_count"], errors="ignore"
        ).merge(counts, on="incident_id", how="left", validate="one_to_one")
        catalog[["field_count", "episode_count"]] = catalog[
            ["field_count", "episode_count"]
        ].fillna(0).astype("int64")
    return CropStoryArtifacts(catalog, weekly, memberships, windows)


def build_crop_story_artifacts(
    exposure_weekly_state: pd.DataFrame,
    exposure_assignments: pd.DataFrame,
    component_memberships: pd.DataFrame,
    weekly_cells: pd.DataFrame,
    config: Mapping[str, Any] | Any,
    *,
    through_week: str | None = None,
    stage_summary: pd.DataFrame | None = None,
    followup_evidence: pd.DataFrame | None = None,
    incident_lineage: pd.DataFrame | None = None,
) -> CropStoryArtifacts:
    """Compatibility wrapper for bounded tests and callers with prepared evidence.

    Production must pass a crop-specific ``stage_summary`` produced from the
    field-week context.  The fallback only adapts small synthetic cell fixtures.
    """
    scaffold = build_crop_story_scaffold(
        exposure_weekly_state, exposure_assignments, component_memberships,
        weekly_cells, config, through_week=through_week,
    )
    summary = (
        stage_summary
        if stage_summary is not None
        else _fallback_stage_summary(scaffold.weekly_state, weekly_cells)
    )
    return finalize_crop_story_artifacts(
        scaffold, summary, config, followup_evidence=followup_evidence,
        incident_lineage=incident_lineage, weekly_cells=weekly_cells,
    )


def _incident_memberships(
    memberships: pd.DataFrame,
    assignments: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    source = memberships.copy()
    source["crop_name_normalized"] = source["crop_name"].map(_crop)
    source["timeline_bucket"] = pd.to_datetime(
        source["timeline_bucket"], errors="raise"
    ).dt.normalize()
    joined = source.merge(
        assignments[["component_id", "exposure_id"]],
        on="component_id", how="inner", validate="many_to_one",
    ).merge(
        catalog[["exposure_id", "crop_name_normalized", "incident_id"]],
        on=["exposure_id", "crop_name_normalized"], how="inner",
        validate="many_to_one",
    )
    for name, fallback in (
        ("crop_instance_id", "unknown_crop_instance"),
        ("episode_id", "unknown_episode"),
        ("response_class", "no_new_event_response"),
        ("fresh_response_evidence", False),
        ("evaluable", False),
        ("is_data_gap", False),
        ("stage_bucket", "unknown"),
        ("grid_id", None),
        ("knowledge_time", None),
    ):
        if name not in joined:
            joined[name] = fallback
        if fallback is not None:
            joined[name] = joined[name].fillna(fallback)
    joined["knowledge_time"] = pd.to_datetime(
        joined["knowledge_time"], errors="coerce"
    ).dt.normalize()
    if joined.duplicated(["incident_id", "timeline_bucket", "field_id"]).any():
        raise ValueError("Incident membership is not canonical by story, week, and field")
    columns = [
        "timeline_bucket", "incident_id", "exposure_id", "component_id",
        "crop_name_normalized", "hazard_family", "field_id", "crop_instance_id",
        "episode_id", "membership_role", "event_state", "response_class",
        "fresh_response_evidence", "evaluable", "is_data_gap", "stage_bucket",
        "grid_id",
        "knowledge_time",
    ]
    return joined.loc[:, columns].sort_values(
        ["timeline_bucket", "incident_id", "membership_role", "field_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    output = cells.copy()
    aliases = {"grid_x": "cell_x", "grid_y": "cell_y"}
    for canonical, source in aliases.items():
        if canonical not in output and source in output:
            output[canonical] = output[source]
    if output.empty:
        for name in ("timeline_bucket", "hazard_family", "grid_x", "grid_y"):
            if name not in output:
                output[name] = pd.Series(dtype="object")
        return output
    required = {"timeline_bucket", "hazard_family", "grid_x", "grid_y"}
    missing = sorted(required - set(output.columns))
    if missing:
        raise ValueError("weekly cells are missing columns: " + ", ".join(missing))
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="raise"
    ).dt.normalize()
    return output


def _footprint_cells_adequate(
    cells: pd.DataFrame,
    week: pd.Timestamp,
    hazard: str,
    footprint: set[str],
) -> bool:
    if not footprint or "passes_coverage_gate" not in cells:
        return False
    current = cells[
        cells["timeline_bucket"].eq(pd.Timestamp(week).normalize())
        & cells["hazard_family"].astype(str).eq(str(hazard))
    ].copy()
    if current.empty:
        return False
    current["grid_id"] = (
        "g:" + current["grid_x"].astype("int64").astype(str)
        + ":" + current["grid_y"].astype("int64").astype(str)
    )
    gate_by_cell = {
        str(row["grid_id"]): bool(row["passes_coverage_gate"])
        for row in current.to_dict("records")
    }
    return all(gate_by_cell.get(cell_id, False) for cell_id in footprint)


def _coverage_lookup(stage_summary: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], dict[str, int]]:
    required = {
        "timeline_bucket", "incident_id", "monitored_field_count",
        "evaluable_field_count", "monitored_crop_instance_count",
        "evaluable_crop_instance_count",
    }
    missing = sorted(required - set(stage_summary.columns))
    if missing:
        raise ValueError("incident stage summary is missing lifecycle coverage: " + ", ".join(missing))
    if stage_summary.empty:
        return {}
    source = stage_summary.copy()
    source["timeline_bucket"] = pd.to_datetime(
        source["timeline_bucket"], errors="raise"
    ).dt.normalize()
    aggregations: dict[str, tuple[str, str]] = {
        "monitored_field_count": ("monitored_field_count", "sum"),
        "evaluable_field_count": ("evaluable_field_count", "sum"),
        "monitored_crop_instance_count": ("monitored_crop_instance_count", "sum"),
        "evaluable_crop_instance_count": ("evaluable_crop_instance_count", "sum"),
    }
    source["off_season_monitored_crop_instance_count"] = source[
        "monitored_crop_instance_count"
    ].where(source["stage_bucket"].astype(str).eq("off_season"), 0)
    aggregations["off_season_monitored_crop_instance_count"] = (
        "off_season_monitored_crop_instance_count", "sum"
    )
    if "coverage_missing_cell_count" in source:
        aggregations["coverage_missing_cell_count"] = (
            "coverage_missing_cell_count", "max"
        )
    grouped = source.groupby(
        ["incident_id", "timeline_bucket"], as_index=False, sort=False
    ).agg(**aggregations)
    if "coverage_missing_cell_count" not in grouped:
        grouped["coverage_missing_cell_count"] = 0
    return {
        (str(row["incident_id"]), pd.Timestamp(row["timeline_bucket"]).normalize()): {
            name: int(row[name])
            for name in (
                "monitored_field_count", "evaluable_field_count",
                "monitored_crop_instance_count", "evaluable_crop_instance_count",
                "coverage_missing_cell_count",
                "off_season_monitored_crop_instance_count",
            )
        }
        for row in grouped.to_dict("records")
    }


def _coverage_for_story_cells(
    base_row: Mapping[str, Any],
    incident_id: str,
    base_incident_id: str,
    week: pd.Timestamp,
    hazard: str,
    footprint: set[str],
    coverage: Mapping[tuple[str, pd.Timestamp], dict[str, int]],
    weekly_cells: pd.DataFrame | None,
    config: Mapping[str, Any] | Any,
) -> tuple[bool, dict[str, int], bool]:
    dynamic_cell_coverage_adequate = (
        _footprint_cells_adequate(weekly_cells, week, hazard, footprint)
        if weekly_cells is not None
        else bool(base_row.get("cell_coverage_adequate", False))
    )
    crop_coverage = coverage.get(
        (str(incident_id), week),
        coverage.get((str(base_incident_id), week), _empty_coverage(base_row)),
    )
    minimum_crop = int(
        _config_value(
            config,
            "minimum_crop_monitored_instances",
            "minimum_crop_monitored_instances",
            1,
        )
    )
    minimum_evaluable_crop = int(
        _config_value(
            config,
            "minimum_crop_evaluable_instances",
            "minimum_crop_evaluable_instances",
            1,
        )
    )
    adequate = bool(
        dynamic_cell_coverage_adequate
        and crop_coverage["coverage_missing_cell_count"] == 0
        and crop_coverage["monitored_crop_instance_count"] >= minimum_crop
        and crop_coverage["evaluable_crop_instance_count"]
        >= minimum_evaluable_crop
    )
    return dynamic_cell_coverage_adequate, crop_coverage, adequate


def _empty_coverage(base: Mapping[str, Any]) -> dict[str, int]:
    footprint = _parse_json_set(base.get("footprint_cell_ids_json"))
    return {
        "monitored_field_count": 0,
        "evaluable_field_count": 0,
        "monitored_crop_instance_count": 0,
        "evaluable_crop_instance_count": 0,
        "coverage_missing_cell_count": len(footprint),
        "off_season_monitored_crop_instance_count": 0,
    }


def _normalize_followup(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "timeline_bucket", "incident_id", "field_id", "crop_instance_id",
        "episode_id", "hazard_family", "event_state", "response_class",
        "stage_bucket", "knowledge_time", "fresh_decline_evidence",
        "fresh_recovery_evidence",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("follow-up evidence is missing: " + ", ".join(missing))
    output = frame.loc[:, columns].copy()
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="raise"
    ).dt.normalize()
    return output


def _current_recovery_as_followup(
    row: Mapping[str, Any], owner: str
) -> dict[str, Any]:
    """Represent routed recovery separately from its physical context row."""
    return {
        "timeline_bucket": pd.Timestamp(row["timeline_bucket"]).normalize(),
        "incident_id": owner,
        "field_id": str(row.get("field_id") or "unknown_field"),
        "crop_instance_id": str(
            row.get("crop_instance_id") or "unknown_crop_instance"
        ),
        "episode_id": str(row.get("episode_id") or "unknown_episode"),
        "hazard_family": str(row.get("hazard_family") or ""),
        "event_state": str(row.get("event_state") or "RECOVERING"),
        "response_class": "recovery",
        "stage_bucket": str(row.get("stage_bucket") or "unknown"),
        "knowledge_time": row.get("knowledge_time"),
        "fresh_decline_evidence": False,
        "fresh_recovery_evidence": True,
    }


def _assert_equivalent_followup(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    """Canonicalize duplicate historical owners only when evidence agrees."""
    columns = (
        "timeline_bucket",
        "field_id",
        "crop_instance_id",
        "episode_id",
        "hazard_family",
        "event_state",
        "response_class",
        "stage_bucket",
        "knowledge_time",
        "fresh_decline_evidence",
        "fresh_recovery_evidence",
    )
    conflicts: list[str] = []
    for name in columns:
        left_value = left.get(name)
        right_value = right.get(name)
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if name in {"timeline_bucket", "knowledge_time"}:
            equal = pd.Timestamp(left_value) == pd.Timestamp(right_value)
        elif name in {"fresh_decline_evidence", "fresh_recovery_evidence"}:
            equal = bool(left_value) == bool(right_value)
        else:
            equal = str(left_value) == str(right_value)
        if not equal:
            conflicts.append(name)
    if conflicts:
        raise ValueError(
            "Duplicate remapped follow-up evidence conflicts in: "
            + ", ".join(conflicts)
        )


def _normalize_lineage(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "timeline_bucket", "parent_incident_id", "child_incident_id",
        "lineage_type",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("incident lineage is missing lifecycle columns: " + ", ".join(missing))
    output = frame.copy()
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="raise"
    ).dt.normalize()
    return output


def _lineage_event_weeks(
    lineage: pd.DataFrame, kind: str
) -> dict[str, tuple[pd.Timestamp, ...]]:
    if lineage.empty:
        return {}
    rows = lineage[lineage["lineage_type"].astype(str).str.lower().eq(kind)]
    events: dict[str, list[pd.Timestamp]] = {}
    for row in rows.to_dict("records"):
        week = pd.Timestamp(row["timeline_bucket"]).normalize()
        for raw_incident_id in (
            row.get("parent_incident_id"), row.get("child_incident_id")
        ):
            if pd.isna(raw_incident_id):
                continue
            events.setdefault(str(raw_incident_id), []).append(week)
    return {
        incident_id: tuple(sorted(weeks))
        for incident_id, weeks in events.items()
    }


def _lineage_count_through(
    events: dict[str, tuple[pd.Timestamp, ...]],
    incident_id: str,
    week: pd.Timestamp,
    *,
    since: pd.Timestamp | None = None,
) -> int:
    values = events.get(incident_id, ())
    upper = bisect_right(values, pd.Timestamp(week).normalize())
    if since is None:
        return upper
    lower = bisect_right(
        values,
        pd.Timestamp(since).normalize() - pd.Timedelta(days=1),
    )
    return max(0, upper - lower)


def _remap_merge_targets(
    weekly_records: list[dict[str, Any]],
    window_records: list[dict[str, Any]],
    mapping: Mapping[tuple[str, pd.Timestamp], str],
) -> None:
    for record in weekly_records:
        target = record.get("merged_into_incident_id")
        if not target:
            continue
        week = pd.Timestamp(record["timeline_bucket"]).normalize()
        record["merged_into_incident_id"] = mapping.get(
            (str(target), week), str(target)
        )
    for record in window_records:
        target = record.get("merged_into_incident_id")
        closed_week = record.get("closed_week")
        if not target or closed_week is None or pd.isna(closed_week):
            continue
        week = pd.Timestamp(closed_week).normalize()
        record["merged_into_incident_id"] = mapping.get(
            (str(target), week), str(target)
        )


def _merge_target(lineage: pd.DataFrame, week: pd.Timestamp) -> str | None:
    if lineage.empty:
        return None
    rows = lineage[
        lineage["lineage_type"].astype(str).str.lower().eq("merge")
        & lineage["timeline_bucket"].eq(week)
    ]
    if rows.empty:
        return None
    return str(rows.iloc[0]["child_incident_id"])


def _row_claims_unresolved_ownership(row: Mapping[str, Any]) -> bool:
    """Return whether a current row directly owns unresolved crop response."""
    response = str(row.get("response_class") or "").lower()
    fresh = bool(row.get("fresh_response_evidence"))
    return bool(
        str(row.get("membership_role") or "") == "impact_lag"
        or str(row.get("event_state") or "").upper()
        == "CLOSED_RESPONSE_UNRESOLVED"
        or (fresh and response in {"medium_decline", "severe_decline"})
    )


def _row_reports_fresh_recovery(row: Mapping[str, Any]) -> bool:
    return bool(row.get("fresh_response_evidence")) and str(
        row.get("response_class") or ""
    ).lower() == "recovery"


def _unresolved_owner_lookup(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, tuple[str, str, str]], str]:
    _assert_unique_unresolved_owners(runtimes)
    owners: dict[tuple[str, tuple[str, str, str]], str] = {}
    for base, runtime in runtimes.items():
        if runtime.get("incident_id") is None:
            continue
        hazard = str(runtime.get("hazard") or "")
        for evidence_key in runtime.get("unresolved", {}):
            owners[(hazard, evidence_key)] = base
    return owners


def _recovered_owner_lookup(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, tuple[str, str, str]], str]:
    unresolved = _unresolved_owner_lookup(runtimes)
    owner_sets = _recovered_owner_sets(runtimes)
    owners: dict[tuple[str, tuple[str, str, str]], str] = {}
    for owner_key, values in sorted(owner_sets.items()):
        recovered = sorted(values)
        if len(recovered) > 1:
            hazard, evidence_key = owner_key
            raise ValueError(
                "Recovered episode has multiple active crop-story owners: "
                f"hazard={hazard}, evidence={evidence_key}, "
                f"owners={','.join(recovered)}"
            )
        recovered_owner = recovered[0]
        unresolved_owner = unresolved.get(owner_key)
        if unresolved_owner is not None:
            hazard, evidence_key = owner_key
            raise ValueError(
                "Episode has conflicting unresolved and recovered owners: "
                f"hazard={hazard}, evidence={evidence_key}, "
                f"unresolved_owner={unresolved_owner}, "
                f"recovered_owner={recovered_owner}"
            )
        owners[owner_key] = recovered_owner
    return owners


def _recovered_owner_sets(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, tuple[str, str, str]], set[str]]:
    owners: dict[tuple[str, tuple[str, str, str]], set[str]] = defaultdict(set)
    for base, runtime in runtimes.items():
        if runtime.get("incident_id") is None:
            continue
        hazard = str(runtime.get("hazard") or "")
        for evidence_key in runtime.get("recovered", set()):
            owners[(hazard, evidence_key)].add(base)
    return owners


def _causal_seed_owner_lookup(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> dict[
    tuple[str, tuple[str, str, str]], dict[str, pd.Timestamp | None]
]:
    owners: dict[
        tuple[str, tuple[str, str, str]], dict[str, pd.Timestamp | None]
    ] = defaultdict(dict)
    for base, runtime in runtimes.items():
        if runtime.get("incident_id") is None:
            continue
        hazard = str(runtime.get("hazard") or "")
        for evidence_key in runtime.get("seed_keys", set()):
            raw_week = runtime.get("seed_weeks", {}).get(evidence_key)
            owners[(hazard, evidence_key)][base] = (
                None
                if raw_week is None or pd.isna(raw_week)
                else pd.Timestamp(raw_week).normalize()
            )
    return owners


def _resolve_episode_owner(
    week: pd.Timestamp,
    sources: set[str],
    owner_key: tuple[str, tuple[str, str, str]],
    direct_owners: Mapping[tuple[str, tuple[str, str, str]], str],
    unresolved_owners: Mapping[tuple[str, tuple[str, str, str]], str],
    recovered_owners: Mapping[tuple[str, tuple[str, str, str]], str],
    redirects: Mapping[tuple[str, tuple[str, str, str]], str],
    seed_owners: Mapping[
        tuple[str, tuple[str, str, str]], Mapping[str, pd.Timestamp | None]
    ],
    live_owners: set[str],
) -> str | None:
    owner = (
        direct_owners.get(owner_key)
        or unresolved_owners.get(owner_key)
        or recovered_owners.get(owner_key)
    )
    if owner is not None:
        return owner
    redirect_targets = {
        target
        for source in sources
        for target in [redirects.get((source, owner_key[1]))]
        if target is not None and target in live_owners
    }
    if len(redirect_targets) > 1:
        raise ValueError(
            "Evidence has conflicting live ownership redirects: "
            f"week={pd.Timestamp(week).date()}, hazard={owner_key[0]}, "
            f"evidence={owner_key[1]}, owners={','.join(sorted(redirect_targets))}"
        )
    if redirect_targets:
        return next(iter(redirect_targets))
    candidates = seed_owners.get(owner_key, {})
    if not candidates:
        return None
    if len(candidates) > 1 and any(value is None for value in candidates.values()):
        raise ValueError(
            "Evidence has active story seeds without causal recency: "
            f"week={pd.Timestamp(week).date()}, hazard={owner_key[0]}, "
            f"evidence={owner_key[1]}, owners={','.join(sorted(candidates))}"
        )
    known = {
        base: seed_week
        for base, seed_week in candidates.items()
        if seed_week is not None
    }
    if known:
        latest = max(known.values())
        winners = sorted(base for base, seed_week in known.items() if seed_week == latest)
    else:
        winners = sorted(candidates)
        latest = None
    if len(winners) != 1:
        raise ValueError(
            "Evidence has ambiguous latest causal story seeds: "
            f"week={pd.Timestamp(week).date()}, hazard={owner_key[0]}, "
            f"evidence={owner_key[1]}, seed_week={latest}, "
            f"owners={','.join(winners)}"
        )
    return winners[0]


def _establish_episode_seed_owner(
    owner_key: tuple[str, tuple[str, str, str]],
    owner: str,
    week: pd.Timestamp,
    runtimes: dict[str, dict[str, Any]],
    redirects: dict[tuple[str, tuple[str, str, str]], str],
) -> None:
    hazard, evidence_key = owner_key
    for base, runtime in runtimes.items():
        if runtime.get("incident_id") is None:
            continue
        if str(runtime.get("hazard") or "") != hazard:
            continue
        if base != owner and evidence_key in runtime.get("seed_keys", set()):
            runtime["seed_keys"].discard(evidence_key)
            runtime.setdefault("seed_weeks", {}).pop(evidence_key, None)
            redirects[(base, evidence_key)] = owner
    owner_runtime = runtimes[owner]
    owner_runtime["seed_keys"].add(evidence_key)
    owner_runtime.setdefault("seed_weeks", {})[evidence_key] = pd.Timestamp(
        week
    ).normalize()


def _direct_claim_owner_lookup(
    week: pd.Timestamp,
    runtimes: Mapping[str, Mapping[str, Any]],
    current_by_base: Mapping[str, pd.DataFrame],
) -> dict[tuple[str, tuple[str, str, str]], str]:
    claimants: dict[
        tuple[str, tuple[str, str, str]], set[str]
    ] = defaultdict(set)
    for base in sorted(current_by_base):
        runtime = runtimes.get(base)
        if runtime is None or runtime.get("incident_id") is None:
            continue
        hazard = str(runtime.get("hazard") or "")
        for row in current_by_base[base].to_dict("records"):
            if _row_claims_unresolved_ownership(row):
                claimants[(hazard, _evidence_key(row))].add(base)
    owners: dict[tuple[str, tuple[str, str, str]], str] = {}
    for owner_key, values in sorted(claimants.items()):
        direct = sorted(values)
        if len(direct) > 1:
            hazard, evidence_key = owner_key
            raise ValueError(
                "Unresolved episode has multiple direct current claimants: "
                f"week={pd.Timestamp(week).date()}, hazard={hazard}, "
                f"evidence={evidence_key}, claimants={','.join(direct)}"
            )
        owners[owner_key] = direct[0]
    return owners


def _reconcile_direct_unresolved_claims(
    week: pd.Timestamp,
    runtimes: dict[str, dict[str, Any]],
    current_by_base: Mapping[str, pd.DataFrame],
    redirects: dict[tuple[str, tuple[str, str, str]], str],
    lineage_edges: pd.DataFrame,
) -> None:
    """Move stale carried ownership to the one admitted current claimant."""
    owners = _unresolved_owner_lookup(runtimes)
    recovered_owner_sets = _recovered_owner_sets(runtimes)
    outgoing_merge_parents: set[str] = set()
    if not lineage_edges.empty:
        outgoing_merge_parents = set(
            lineage_edges.loc[
                lineage_edges["lineage_type"].astype(str).str.lower().eq("merge"),
                "parent_incident_id",
            ].astype(str)
        )
    direct_owners = _direct_claim_owner_lookup(week, runtimes, current_by_base)
    for owner_key, current_owner in sorted(direct_owners.items()):
        if current_owner in outgoing_merge_parents:
            hazard, evidence_key = owner_key
            raise ValueError(
                "Outgoing merge parent has a direct unresolved claim: "
                f"week={pd.Timestamp(week).date()}, hazard={hazard}, "
                f"evidence={evidence_key}, claimant={current_owner}"
            )

    for owner_key, current_owner in sorted(direct_owners.items()):
        previous_owner = owners.get(owner_key)
        evidence_key = owner_key[1]
        current_runtime = runtimes[current_owner]
        stale_owners = set(recovered_owner_sets.get(owner_key, set()))
        if previous_owner is not None:
            stale_owners.add(previous_owner)
        stale_owners.discard(current_owner)

        if previous_owner is not None and previous_owner != current_owner:
            previous_runtime = runtimes[previous_owner]
            evidence = previous_runtime["unresolved"].pop(evidence_key)
            previous_runtime["seed_keys"].discard(evidence_key)
            previous_runtime.setdefault("seed_weeks", {}).pop(evidence_key, None)
            current_runtime["unresolved"][evidence_key] = evidence
        for stale_owner in stale_owners:
            runtimes[stale_owner]["recovered"].discard(evidence_key)
            runtimes[stale_owner]["seed_keys"].discard(evidence_key)
            runtimes[stale_owner].setdefault("seed_weeks", {}).pop(
                evidence_key, None
            )
        current_runtime["recovered"].discard(evidence_key)
        current_runtime["seed_keys"].add(evidence_key)
        for origin_key, owner in list(redirects.items()):
            if origin_key[1] == evidence_key:
                redirects[origin_key] = current_owner
        for stale_owner in stale_owners:
            redirects[(stale_owner, evidence_key)] = current_owner
        _establish_episode_seed_owner(
            owner_key, current_owner, week, runtimes, redirects
        )

    _assert_unique_unresolved_owners(runtimes)
    _recovered_owner_lookup(runtimes)


def _apply_impact_registry(
    unresolved: dict[tuple[str, str, str], dict[str, Any]],
    recovered: set[tuple[str, str, str]],
    current: pd.DataFrame,
    followup: pd.DataFrame,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    decline_fields: set[tuple[str, str, str]] = set()
    recovery_fields: set[tuple[str, str, str]] = set()
    for row in current.to_dict("records"):
        evidence_key = _evidence_key(row)
        response = str(row.get("response_class") or "").lower()
        fresh = bool(row.get("fresh_response_evidence"))
        is_unresolved = _row_claims_unresolved_ownership(row)
        if is_unresolved:
            unresolved[evidence_key] = row
        if fresh and response in {"medium_decline", "severe_decline"}:
            decline_fields.add(evidence_key)
            recovered.discard(evidence_key)
        if fresh and response == "recovery":
            recovery_fields.add(evidence_key)
    for row in followup.to_dict("records"):
        evidence_key = _evidence_key(row)
        if bool(row.get("fresh_decline_evidence")):
            previous = unresolved.get(evidence_key, {})
            merged = {**previous, **row}
            if not merged.get("grid_id") and previous.get("grid_id"):
                merged["grid_id"] = previous["grid_id"]
            unresolved[evidence_key] = merged
            decline_fields.add(evidence_key)
            recovered.discard(evidence_key)
        if bool(row.get("fresh_recovery_evidence")):
            recovery_fields.add(evidence_key)
    # If decline and recovery are both reported in the same weekly bucket, keep
    # the field unresolved because the within-week causal order is unknown.
    for evidence_key in recovery_fields - decline_fields:
        unresolved.pop(evidence_key, None)
        recovered.add(evidence_key)
    return decline_fields, recovery_fields


def _transfer_lineage_registries_for_week(
    week: pd.Timestamp,
    edges: pd.DataFrame,
    runtimes: dict[str, dict[str, Any]],
    current_by_base: Mapping[str, pd.DataFrame],
    redirects: dict[tuple[str, tuple[str, str, str]], str],
) -> None:
    """Transfer only prior, unresolved evidence across this week's edges."""
    if edges.empty:
        return
    supported = edges[
        edges["lineage_type"].astype(str).str.lower().isin({"split", "merge"})
    ]
    if supported.empty:
        return
    _assert_unique_unresolved_owners(runtimes)
    for parent in _lineage_parent_order(supported):
        parent_edges = supported[
            supported["parent_incident_id"].astype(str).eq(parent)
        ]
        kinds = set(parent_edges["lineage_type"].astype(str).str.lower())
        if len(kinds) != 1:
            raise ValueError(
                f"Incident {parent} has mixed split/merge ownership at {week.date()}"
            )
        kind = next(iter(kinds))
        children = sorted(set(parent_edges["child_incident_id"].astype(str)))
        if kind == "merge" and len(children) != 1:
            raise ValueError(
                f"Incident {parent} has ambiguous merge ownership at {week.date()}"
            )
        parent_runtime = runtimes.get(parent)
        if parent_runtime is None or parent_runtime["incident_id"] is None:
            continue
        for evidence_key, evidence in list(parent_runtime["unresolved"].items()):
            if kind == "merge":
                candidates = children
            else:
                exact: list[str] = []
                field_crop: list[str] = []
                for child in children:
                    child_members = current_by_base.get(child, _empty_membership())
                    child_keys = {
                        _evidence_key(row)
                        for row in child_members.to_dict("records")
                    }
                    if evidence_key in child_keys:
                        exact.append(child)
                    elif evidence_key[:2] in {key[:2] for key in child_keys}:
                        field_crop.append(child)
                candidates = exact or field_crop
            if len(candidates) > 1:
                raise ValueError(
                    "Unresolved episode has ambiguous lineage ownership: "
                    f"incident={parent}, week={week.date()}, evidence={evidence_key}"
                )
            if not candidates:
                # On a split, an unmatched episode remains with the continuing
                # primary parent.  Nothing is inferred from spatial proximity.
                continue
            child = candidates[0]
            child_runtime = runtimes.get(child)
            if child_runtime is None or child_runtime["incident_id"] is None:
                raise ValueError(
                    "Lineage transfer target has no causally admitted crop story: "
                    f"incident={child}, week={week.date()}"
                )
            if evidence_key in child_runtime["unresolved"]:
                raise ValueError(
                    "Unresolved episode already has two lineage owners: "
                    f"week={week.date()}, evidence={evidence_key}"
                )
            parent_runtime["unresolved"].pop(evidence_key, None)
            parent_runtime["recovered"].discard(evidence_key)
            parent_runtime["seed_keys"].discard(evidence_key)
            parent_runtime["seed_weeks"].pop(evidence_key, None)
            child_runtime["unresolved"][evidence_key] = dict(evidence)
            child_runtime["recovered"].discard(evidence_key)
            child_runtime["seed_keys"].add(evidence_key)
            child_runtime["seed_weeks"][evidence_key] = week
            for origin_key, owner in list(redirects.items()):
                if origin_key[1] == evidence_key and owner == parent:
                    redirects[origin_key] = child
            redirects[(parent, evidence_key)] = child
    _assert_unique_unresolved_owners(runtimes)


def _lineage_parent_order(edges: pd.DataFrame) -> list[str]:
    """Return deterministic topological parent order for same-week transfers."""
    parents = set(edges["parent_incident_id"].astype(str))
    children = set(edges["child_incident_id"].astype(str))
    nodes = parents | children
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for row in edges.to_dict("records"):
        parent = str(row["parent_incident_id"])
        child = str(row["child_incident_id"])
        if child not in outgoing[parent]:
            outgoing[parent].add(child)
            indegree[child] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        if node in parents:
            ordered.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if any(degree > 0 for degree in indegree.values()):
        raise ValueError(f"Incident lineage contains a cycle at {edges.iloc[0]['timeline_bucket']}")
    return ordered


def _assert_unique_unresolved_owners(
    runtimes: Mapping[str, Mapping[str, Any]],
) -> None:
    owners: dict[tuple[str, tuple[str, str, str]], str] = {}
    for base, runtime in runtimes.items():
        if runtime.get("incident_id") is None:
            continue
        hazard = str(runtime.get("hazard") or "")
        for evidence_key in runtime.get("unresolved", {}):
            owner_key = (hazard, evidence_key)
            previous = owners.get(owner_key)
            if previous is not None and previous != base:
                raise ValueError(
                    "Unresolved episode has multiple active crop-story owners: "
                    f"hazard={hazard}, evidence={evidence_key}, "
                    f"owners={previous},{base}"
                )
            owners[owner_key] = base


def _assert_unique_membership_episode_owners(memberships: pd.DataFrame) -> None:
    """Fail if one ownership-bearing episode is published under two stories."""
    ownership_rows = memberships[
        memberships.apply(
            lambda row: (
                _row_claims_unresolved_ownership(row)
                or str(row.get("membership_role") or "")
                in {"unresolved", "recovered"}
            ),
            axis=1,
        )
    ].copy()
    if ownership_rows.empty:
        return
    key = [
        "timeline_bucket",
        "hazard_family",
        "field_id",
        "crop_instance_id",
        "episode_id",
    ]
    conflicts = (
        ownership_rows.groupby(key, dropna=False, sort=True)["incident_id"]
        .agg(lambda values: tuple(sorted(set(str(value) for value in values))))
    )
    conflicts = conflicts[conflicts.map(len) > 1]
    if conflicts.empty:
        return
    evidence, owners = conflicts.index[0], conflicts.iloc[0]
    raise ValueError(
        "Incident membership has multiple unresolved episode owners: "
        f"evidence={evidence}, owners={','.join(owners)}"
    )


def _segment_start_evidence(current: pd.DataFrame) -> bool:
    if current.empty:
        return False
    roles = current["membership_role"].fillna("").astype(str)
    if roles.isin({"pressure_core", "impact_lag"}).any():
        return True
    # Watch/frontier rows never seed crop-story onset, even if a source row
    # happens to carry fresh response evidence. Impact onset must have been
    # explicitly attributed as impact_lag in that same weekly component.
    return False


def _filter_followup_to_seeds(
    followup: pd.DataFrame,
    seeds: set[tuple[str, str, str]],
) -> pd.DataFrame:
    if followup.empty or not seeds:
        return followup.iloc[0:0].copy()
    keep = [
        _evidence_key(row) in seeds for row in followup.to_dict("records")
    ]
    return followup.loc[keep].copy()


def _recurrence_incident_id(
    base_incident_id: str, start_week: pd.Timestamp
) -> str:
    payload = (
        "crop-incident-recurrence-v1\0"
        + str(base_incident_id)
        + "\0"
        + pd.Timestamp(start_week).strftime("%Y-%m-%d")
    )
    return "incident_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _remap_incident_weeks(
    frame: pd.DataFrame,
    mapping: Mapping[tuple[str, pd.Timestamp], str],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    weeks = pd.to_datetime(output["timeline_bucket"], errors="raise").dt.normalize()
    remapped = [
        mapping.get((str(incident_id), pd.Timestamp(week).normalize()))
        for incident_id, week in zip(output["incident_id"], weeks)
    ]
    output["incident_id"] = remapped
    output = output[output["incident_id"].notna()].copy()
    output["timeline_bucket"] = weeks.loc[output.index]
    return output.reset_index(drop=True)


def _evidence_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("field_id") or "unknown_field"),
        str(row.get("crop_instance_id") or "unknown_crop_instance"),
        str(row.get("episode_id") or "unknown_episode"),
    )


def _evidence_field_count(values: set[tuple[str, str, str]]) -> int:
    return len({value[0] for value in values})


def _registry_field_count(
    values: dict[tuple[str, str, str], dict[str, Any]]
) -> int:
    return len({value[0] for value in values})


def _carried_membership_record(
    catalog_row: Mapping[str, Any],
    hazard: str,
    week: pd.Timestamp,
    evidence_key: tuple[str, str, str],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "timeline_bucket": week,
        "incident_id": str(catalog_row["incident_id"]),
        "exposure_id": str(catalog_row["exposure_id"]),
        "component_id": None,
        "crop_name_normalized": str(catalog_row["crop_name_normalized"]),
        "hazard_family": hazard,
        "field_id": evidence_key[0],
        "crop_instance_id": evidence_key[1],
        "episode_id": evidence_key[2],
        "membership_role": "unresolved",
        "event_state": str(
            evidence.get("event_state") or "CLOSED_RESPONSE_UNRESOLVED"
        ),
        "response_class": str(
            evidence.get("response_class") or "no_new_event_response"
        ),
        "fresh_response_evidence": False,
        "evaluable": False,
        "is_data_gap": False,
        "stage_bucket": str(evidence.get("stage_bucket") or "unknown"),
        "grid_id": evidence.get("grid_id"),
        "knowledge_time": evidence.get("knowledge_time"),
    }


def _augment_followup_memberships(
    memberships: pd.DataFrame,
    followup: pd.DataFrame,
    emitted_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    if followup.empty:
        return memberships.copy()
    existing = {
        (str(row.incident_id), pd.Timestamp(row.timeline_bucket).normalize(), str(row.field_id))
        for row in memberships[["incident_id", "timeline_bucket", "field_id"]].itertuples(index=False)
    }
    exposure_by_incident = (
        memberships.groupby("incident_id", sort=False)["exposure_id"].first().astype(str).to_dict()
    )
    crop_by_incident = (
        memberships.groupby("incident_id", sort=False)["crop_name_normalized"].first().astype(str).to_dict()
    )
    records: list[dict[str, Any]] = []
    for row in followup.to_dict("records"):
        incident_id = str(row["incident_id"])
        week = pd.Timestamp(row["timeline_bucket"]).normalize()
        field_id = str(row["field_id"])
        if (incident_id, week) not in emitted_keys or (incident_id, week, field_id) in existing:
            continue
        decline = bool(row.get("fresh_decline_evidence"))
        recovery = bool(row.get("fresh_recovery_evidence"))
        if not decline and not recovery:
            continue
        records.append(
            {
                "timeline_bucket": week,
                "incident_id": incident_id,
                "exposure_id": exposure_by_incident[incident_id],
                "component_id": None,
                "crop_name_normalized": crop_by_incident[incident_id],
                "hazard_family": str(row["hazard_family"]),
                "field_id": field_id,
                "crop_instance_id": str(row["crop_instance_id"]),
                "episode_id": str(row["episode_id"]),
                "membership_role": "unresolved" if decline else "recovered",
                "event_state": str(row["event_state"]),
                "response_class": str(row["response_class"]),
                "fresh_response_evidence": True,
                "evaluable": True,
                "is_data_gap": False,
                "stage_bucket": str(row.get("stage_bucket") or "unknown"),
                "grid_id": None,
                "knowledge_time": row.get("knowledge_time"),
            }
        )
    if not records:
        return memberships.copy()
    output = pd.concat([memberships, pd.DataFrame(records)], ignore_index=True)
    return output.sort_values(
        ["timeline_bucket", "incident_id", "membership_role", "field_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fallback_stage_summary(scaffold: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Adapt bounded synthetic fixtures; not used by the production workflow."""
    normalized = _normalize_cells(cells)
    records: list[dict[str, Any]] = []
    for row in scaffold.to_dict("records"):
        week = pd.Timestamp(row["timeline_bucket"]).normalize()
        hazard = str(row["hazard_family"])
        coords = {_grid_coordinate(value) for value in _parse_json_set(row["footprint_cell_ids_json"])}
        subset = normalized[
            normalized["timeline_bucket"].eq(week)
            & normalized["hazard_family"].astype(str).eq(hazard)
        ]
        if coords:
            subset = subset[
                [(int(x), int(y)) in coords for x, y in zip(subset["grid_x"], subset["grid_y"])]
            ]
        monitored_name = (
            "monitored_field_count" if "monitored_field_count" in subset else "monitored_count"
        )
        evaluable_name = (
            "evaluable_field_count" if "evaluable_field_count" in subset else "evaluable_count"
        )
        monitored = int(pd.to_numeric(subset.get(monitored_name, 0), errors="coerce").fillna(0).sum())
        evaluable = int(pd.to_numeric(subset.get(evaluable_name, 0), errors="coerce").fillna(0).sum())
        records.append(
            {
                "timeline_bucket": week,
                "incident_id": row["incident_id"],
                "stage_bucket": "unknown",
                "monitored_field_count": monitored,
                "evaluable_field_count": evaluable,
                "monitored_crop_instance_count": monitored,
                "evaluable_crop_instance_count": evaluable,
                "coverage_missing_cell_count": max(0, len(coords) - len(subset)),
            }
        )
    return pd.DataFrame(records)


def _story_window(
    catalog_row: dict[str, Any],
    hazard: str,
    lifecycle: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if lifecycle is None or not rows:
        raise ValueError("Cannot close an empty crop-impact story")
    terminal = str(lifecycle["incident_state"])
    peak = max(
        rows,
        key=lambda row: (
            int(row.get("pressure_core_field_count", 0))
            + int(row.get("unresolved_carried_field_count", 0)),
            -pd.Timestamp(row["timeline_bucket"]).value,
        ),
    )
    return {
        "incident_id": str(catalog_row["incident_id"]),
        "exposure_id": str(catalog_row["exposure_id"]),
        "crop_name": str(catalog_row["crop_name_normalized"]),
        "hazard_family": hazard,
        "first_evidence_week": lifecycle.get("first_evidence_week"),
        "confirmed_week": lifecycle.get("confirmed_week"),
        "pressure_off_week": lifecycle.get("pressure_off_week"),
        "recovered_week": lifecycle.get("recovered_week"),
        "closed_week": lifecycle.get("closed_week"),
        "merged_into_incident_id": lifecycle.get("merged_into_incident_id"),
        "terminal_state": terminal,
        "right_censored": not is_terminal_incident_state(terminal),
        "observed_week_count": len(rows),
        "active_component_week_count": sum(
            int(row["pressure_core_field_count"]) > 0 for row in rows
        ),
        "peak_week": peak["timeline_bucket"],
        "peak_affected_field_count": (
            int(peak["pressure_core_field_count"])
            + int(peak["unresolved_carried_field_count"])
        ),
        "relapse_count": int(lifecycle.get("relapse_count", 0)),
        "data_gap_count": int(lifecycle.get("data_gap_count", 0)),
        "split_count": int(rows[-1].get("split_count", 0)),
        "merge_count": int(rows[-1].get("merge_count", 0)),
        "outcome_evidence": "monitoring_signals_only_no_crop_death_inference",
    }


def _footprint_center(
    cells: set[str], config: Mapping[str, Any] | Any
) -> tuple[float | None, float | None]:
    if not cells:
        return None, None
    size = float(_config_value(config, "grid_cell_size_km", "cell_size_km", 5.0))
    origin_lon = float(_config_value(config, "grid_origin_lon", "origin_lon", 0.0))
    origin_lat = float(_config_value(config, "grid_origin_lat", "origin_lat", 0.0))
    reference = float(
        _config_value(config, "grid_origin_lat", "reference_latitude", origin_lat)
    )
    scale_lon = 111.32 * math.cos(math.radians(reference))
    coordinates = [_grid_coordinate(value) for value in cells]
    lon = sum(origin_lon + (x + 0.5) * size / scale_lon for x, _ in coordinates) / len(coordinates)
    lat = sum(origin_lat + (y + 0.5) * size / 110.574 for _, y in coordinates) / len(coordinates)
    return lon, lat


def _crop(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(value or "unknown_crop").strip().lower()
    ).strip("_") or "unknown_crop"


def _json_set(values: set[str]) -> str:
    return json.dumps(sorted(values), separators=(",", ":"))


def _parse_json_set(value: Any) -> set[str]:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return set()
    parsed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("cell IDs must be a JSON list")
    return {str(item) for item in parsed}


def _grid_coordinate(value: str) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) != 3 or parts[0] != "g":
        raise ValueError(f"Invalid grid ID: {value}")
    return int(parts[1]), int(parts[2])


def _distribution(values: pd.Series) -> str:
    if values.empty:
        return "{}"
    counts = values.fillna("unknown").astype(str).value_counts(sort=False)
    total = int(counts.sum())
    return json.dumps(
        {str(key): round(int(counts[key]) / total, 8) for key in sorted(counts.index)},
        sort_keys=True, separators=(",", ":"),
    )


def _config_value(
    config: Mapping[str, Any] | Any,
    primary: str,
    alias: str,
    default: Any,
) -> Any:
    if isinstance(config, Mapping):
        return config.get(primary, config.get(alias, default))
    return getattr(config, primary, getattr(config, alias, default))


def _empty_scaffold() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timeline_bucket", "incident_id", "exposure_id", "crop_name",
            "hazard_family", "component_id", "pressure_core_field_count",
            "knowledge_time", "knowledge_time_inferred",
            "severe_field_count", "watch_frontier_field_count",
            "impact_lag_field_count", "pressure_cell_ids_json",
            "impact_cell_ids_json", "watch_cell_ids_json",
            "footprint_cell_ids_json", "footprint_carried_forward",
            "center_lon", "center_lat", "footprint_area_km2",
            "hazard_intensity",
            "is_physical_movement",
        ]
    )


def _empty_weekly() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *_empty_scaffold().columns,
            "incident_state", "current_state", "coverage_adequate",
            "coverage_gap_streak",
        ]
    )


def _empty_membership() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timeline_bucket", "incident_id", "exposure_id", "component_id",
            "crop_name_normalized", "hazard_family", "field_id",
            "crop_instance_id", "episode_id", "membership_role", "event_state",
            "response_class", "fresh_response_evidence", "evaluable",
            "is_data_gap", "stage_bucket", "grid_id",
            "knowledge_time",
        ]
    )


def _empty_windows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["incident_id", "exposure_id", "terminal_state"]
    )


__all__ = [
    "CropStoryArtifacts",
    "CropStoryScaffold",
    "build_crop_story_artifacts",
    "build_crop_story_scaffold",
    "build_incident_followup_evidence",
    "finalize_crop_story_artifacts",
]
