"""Crop/stage denominators for exact pressure and impact story footprints."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

import duckdb
import pandas as pd

from .incident_policy_v3 import IncidentPolicyV3


SUMMARY_SCHEMA_VERSION = "incident-stage-summary-v3/2"


SUMMARY_COLUMNS = [
    "timeline_bucket", "incident_id", "exposure_id", "crop_name",
    "hazard_family", "stage_bucket", "monitored_field_count",
    "evaluable_field_count", "monitored_crop_instance_count",
    "evaluable_crop_instance_count", "pressure_core_crop_instance_count",
    "severe_crop_instance_count", "watch_frontier_crop_instance_count",
    "impact_lag_crop_instance_count", "affected_crop_instance_count",
    "pressure_signal_rate", "impact_signal_rate", "footprint_cell_count",
    "crop_observed_cell_count", "coverage_missing_cell_count",
    "global_crop_week_unmappable_instance_count", "denominator_scope",
    "schema_version", "policy_version", "policy_sha256",
]


def build_incident_stage_summary(
    context_path: Path,
    incident_weekly_state: pd.DataFrame,
    incident_memberships: pd.DataFrame,
    *,
    policy: IncidentPolicyV3,
    reference_latitude: float,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Return explicit crop-stage numerators and denominators per story week.

    Denominator cells are the union of the crop-specific story footprint and
    every pressure/watch/impact membership cell.  An impact-lag field therefore
    cannot sit outside its denominator.
    """
    footprints = _expand_footprints(incident_weekly_state, incident_memberships)
    if footprints.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    context_path = context_path.expanduser().resolve()
    if not context_path.is_file():
        raise FileNotFoundError(f"Missing field-week context: {context_path}")
    if not math.isfinite(reference_latitude):
        raise ValueError("reference_latitude must be finite")
    memberships = _normalize_memberships(incident_memberships)
    scale_lon = 111.32 * math.cos(math.radians(reference_latitude))
    if abs(scale_lon) < 1e-9:
        raise ValueError("Metric grid longitude scale is undefined")
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved)])
        connection.read_parquet(str(context_path)).create_view("context_source")
        connection.register("story_footprints", footprints)
        connection.register("incident_memberships", memberships)
        result = connection.execute(
            """
            WITH context_cells AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(field_id AS VARCHAR) AS field_id,
                    CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
                    REGEXP_REPLACE(LOWER(TRIM(CAST(crop_name AS VARCHAR))),
                        '[^a-z0-9]+', '_', 'g') AS crop_name,
                    COALESCE(CAST(stage_bucket AS VARCHAR), 'unknown') AS stage_bucket,
                    COALESCE(TRY_CAST(monitored AS BOOLEAN), TRUE) AS monitored,
                    COALESCE(TRY_CAST(evaluable AS BOOLEAN), FALSE) AS evaluable,
                    COALESCE(TRY_CAST(centroid_available AS BOOLEAN), FALSE)
                        AS centroid_available,
                    FLOOR((TRY_CAST(centroid_lon AS DOUBLE) - ?) * ? / ?)::BIGINT AS grid_x,
                    FLOOR((TRY_CAST(centroid_lat AS DOUBLE) - ?) * 110.574 / ?)::BIGINT AS grid_y
                FROM context_source
            ), footprint_stats AS (
                SELECT timeline_bucket, incident_id, exposure_id, crop_name,
                    hazard_family, COUNT(*)::BIGINT AS footprint_cell_count
                FROM story_footprints
                GROUP BY 1, 2, 3, 4, 5
            ), coverage AS (
                SELECT
                    f.timeline_bucket, f.incident_id, f.exposure_id, f.crop_name,
                    f.hazard_family,
                    COUNT(DISTINCT CASE WHEN c.crop_instance_id IS NOT NULL
                        THEN CAST(f.grid_x AS VARCHAR) || ':' || CAST(f.grid_y AS VARCHAR)
                    END)::BIGINT AS crop_observed_cell_count
                FROM story_footprints f
                LEFT JOIN context_cells c
                  ON c.timeline_bucket = f.timeline_bucket
                 AND c.crop_name = f.crop_name
                 AND c.centroid_available
                 AND c.grid_x = f.grid_x AND c.grid_y = f.grid_y
                GROUP BY 1, 2, 3, 4, 5
            ), local_denominators AS (
                SELECT
                    f.timeline_bucket, f.incident_id, f.exposure_id, f.crop_name,
                    f.hazard_family, COALESCE(c.stage_bucket, 'unknown') AS stage_bucket,
                    COUNT(DISTINCT CASE WHEN c.monitored THEN c.field_id END)::BIGINT
                        AS monitored_field_count,
                    COUNT(DISTINCT CASE WHEN c.evaluable THEN c.field_id END)::BIGINT
                        AS evaluable_field_count,
                    COUNT(DISTINCT CASE WHEN c.monitored THEN c.crop_instance_id END)::BIGINT
                        AS monitored_crop_instance_count,
                    COUNT(DISTINCT CASE WHEN c.evaluable THEN c.crop_instance_id END)::BIGINT
                        AS evaluable_crop_instance_count
                FROM story_footprints f
                LEFT JOIN context_cells c
                  ON c.timeline_bucket = f.timeline_bucket
                 AND c.crop_name = f.crop_name
                 AND c.centroid_available
                 AND c.grid_x = f.grid_x AND c.grid_y = f.grid_y
                GROUP BY 1, 2, 3, 4, 5, 6
            ), membership_staged AS (
                SELECT
                    m.*,
                    COALESCE(c.stage_bucket, m.stage_bucket, 'unknown')
                        AS current_stage_bucket,
                    c.crop_instance_id IS NOT NULL AS has_current_crop_context
                FROM incident_memberships m
                LEFT JOIN context_cells c
                  ON c.timeline_bucket = m.timeline_bucket
                 AND c.crop_instance_id = m.crop_instance_id
                 AND c.crop_name = m.crop_name
            ), affected AS (
                SELECT
                    timeline_bucket, incident_id, exposure_id, crop_name,
                    hazard_family, current_stage_bucket AS stage_bucket,
                    COUNT(DISTINCT CASE WHEN has_current_crop_context
                        AND membership_role = 'pressure_core'
                        THEN crop_instance_id END)::BIGINT AS pressure_core_crop_instance_count,
                    COUNT(DISTINCT CASE WHEN has_current_crop_context
                        AND membership_role = 'pressure_core'
                        AND UPPER(event_state) = 'SEVERE' THEN crop_instance_id END)::BIGINT
                        AS severe_crop_instance_count,
                    COUNT(DISTINCT CASE WHEN has_current_crop_context
                        AND membership_role = 'watch_frontier'
                        THEN crop_instance_id END)::BIGINT AS watch_frontier_crop_instance_count,
                    COUNT(DISTINCT CASE WHEN has_current_crop_context
                        AND membership_role IN
                        ('impact_lag', 'unresolved', 'recovered')
                        THEN crop_instance_id END)::BIGINT AS impact_lag_crop_instance_count,
                    COUNT(DISTINCT CASE WHEN has_current_crop_context
                        AND membership_role IN
                        ('pressure_core', 'impact_lag', 'unresolved', 'recovered')
                        THEN crop_instance_id END)::BIGINT AS affected_crop_instance_count
                FROM membership_staged
                GROUP BY 1, 2, 3, 4, 5, 6
            ), dimensions AS (
                SELECT timeline_bucket, incident_id, exposure_id, crop_name,
                    hazard_family, stage_bucket FROM local_denominators
                UNION
                SELECT timeline_bucket, incident_id, exposure_id, crop_name,
                    hazard_family, stage_bucket FROM affected
            ), missing_geometry AS (
                SELECT CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    REGEXP_REPLACE(LOWER(TRIM(CAST(crop_name AS VARCHAR))),
                        '[^a-z0-9]+', '_', 'g') AS crop_name,
                    COUNT(DISTINCT crop_instance_id)::BIGINT AS missing_count
                FROM context_source
                WHERE NOT COALESCE(TRY_CAST(centroid_available AS BOOLEAN), FALSE)
                GROUP BY 1, 2
            )
            SELECT
                d.timeline_bucket, d.incident_id, d.exposure_id, d.crop_name,
                d.hazard_family, d.stage_bucket,
                COALESCE(n.monitored_field_count, 0) AS monitored_field_count,
                COALESCE(n.evaluable_field_count, 0) AS evaluable_field_count,
                COALESCE(n.monitored_crop_instance_count, 0)
                    AS monitored_crop_instance_count,
                COALESCE(n.evaluable_crop_instance_count, 0)
                    AS evaluable_crop_instance_count,
                COALESCE(a.pressure_core_crop_instance_count, 0)
                    AS pressure_core_crop_instance_count,
                COALESCE(a.severe_crop_instance_count, 0) AS severe_crop_instance_count,
                COALESCE(a.watch_frontier_crop_instance_count, 0)
                    AS watch_frontier_crop_instance_count,
                COALESCE(a.impact_lag_crop_instance_count, 0)
                    AS impact_lag_crop_instance_count,
                COALESCE(a.affected_crop_instance_count, 0)
                    AS affected_crop_instance_count,
                COALESCE(a.pressure_core_crop_instance_count, 0)::DOUBLE
                    / NULLIF(n.monitored_crop_instance_count, 0) AS pressure_signal_rate,
                COALESCE(a.affected_crop_instance_count, 0)::DOUBLE
                    / NULLIF(n.monitored_crop_instance_count, 0) AS impact_signal_rate,
                s.footprint_cell_count,
                c.crop_observed_cell_count,
                GREATEST(s.footprint_cell_count - c.crop_observed_cell_count, 0)::BIGINT
                    AS coverage_missing_cell_count,
                COALESCE(m.missing_count, 0)
                    AS global_crop_week_unmappable_instance_count,
                CAST('crop_instances_in_pressure_watch_and_impact_cells' AS VARCHAR)
                    AS denominator_scope,
                CAST(? AS VARCHAR) AS schema_version,
                CAST(? AS VARCHAR) AS policy_version,
                CAST(? AS VARCHAR) AS policy_sha256
            FROM dimensions d
            LEFT JOIN local_denominators n USING (
                timeline_bucket, incident_id, exposure_id, crop_name, hazard_family, stage_bucket
            )
            LEFT JOIN affected a USING (
                timeline_bucket, incident_id, exposure_id, crop_name, hazard_family, stage_bucket
            )
            JOIN footprint_stats s USING (
                timeline_bucket, incident_id, exposure_id, crop_name, hazard_family
            )
            JOIN coverage c USING (
                timeline_bucket, incident_id, exposure_id, crop_name, hazard_family
            )
            LEFT JOIN missing_geometry m USING (timeline_bucket, crop_name)
            ORDER BY d.timeline_bucket, d.hazard_family, d.crop_name,
                d.incident_id, d.stage_bucket
            """,
            [
                policy.grid_origin_lon, scale_lon, policy.grid_cell_size_km,
                policy.grid_origin_lat, policy.grid_cell_size_km,
                SUMMARY_SCHEMA_VERSION, policy.version, policy.source_sha256,
            ],
        ).fetchdf()
    finally:
        connection.close()
    if result.duplicated(["timeline_bucket", "incident_id", "stage_bucket"]).any():
        raise RuntimeError("Incident stage summary contains duplicate story-week-stage rows")
    for name in ("pressure_signal_rate", "impact_signal_rate"):
        rate = pd.to_numeric(result[name], errors="coerce").dropna()
        if (~rate.between(0, 1)).any():
            raise ValueError(f"{name} is outside [0, 1]")
    return result.loc[:, SUMMARY_COLUMNS]


def enrich_incident_weekly_state(
    weekly_state: pd.DataFrame,
    stage_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach crop-specific totals using the explicit stage denominator table."""
    if weekly_state.empty:
        return weekly_state.copy()
    required = {
        "timeline_bucket", "incident_id", "monitored_crop_instance_count",
        "evaluable_crop_instance_count", "pressure_core_crop_instance_count",
        "severe_crop_instance_count", "impact_lag_crop_instance_count",
        "affected_crop_instance_count",
    }
    missing = sorted(required - set(stage_summary.columns))
    if missing:
        raise ValueError("incident stage summary is missing: " + ", ".join(missing))
    totals = stage_summary.groupby(
        ["timeline_bucket", "incident_id"], as_index=False, sort=True
    ).agg(
        monitored_count=("monitored_crop_instance_count", "sum"),
        evaluable_count=("evaluable_crop_instance_count", "sum"),
        pressure_core_count=("pressure_core_crop_instance_count", "sum"),
        severe_count=("severe_crop_instance_count", "sum"),
        impact_lag_count=("impact_lag_crop_instance_count", "sum"),
        affected_count=("affected_crop_instance_count", "sum"),
        global_crop_week_unmappable_instance_count=(
            "global_crop_week_unmappable_instance_count", "max"
        ),
    )
    totals["active_count"] = (
        totals["pressure_core_count"] - totals["severe_count"]
    ).clip(lower=0)
    output = weekly_state.copy()
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="raise"
    ).dt.normalize()
    totals["timeline_bucket"] = pd.to_datetime(
        totals["timeline_bucket"], errors="raise"
    ).dt.normalize()
    output = output.merge(
        totals, on=["timeline_bucket", "incident_id"], how="left",
        validate="one_to_one",
    )
    count_columns = [
        "monitored_count", "evaluable_count", "pressure_core_count",
        "severe_count", "impact_lag_count", "affected_count",
        "global_crop_week_unmappable_instance_count", "active_count",
    ]
    output[count_columns] = output[count_columns].fillna(0).astype("int64")
    output["current_state"] = output.get("current_state", output["incident_state"])
    if (output["evaluable_count"] > output["monitored_count"]).any():
        raise ValueError("Crop-specific evaluable count exceeds monitored count")
    if (output["affected_count"] > output["monitored_count"]).any():
        raise ValueError("Crop-specific affected count exceeds expanded footprint denominator")
    return output


def _expand_footprints(
    stories: pd.DataFrame,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "timeline_bucket", "incident_id", "exposure_id", "crop_name",
        "hazard_family", "footprint_cell_ids_json",
    }
    missing = sorted(required - set(stories.columns))
    if missing:
        raise ValueError("incident weekly state is missing: " + ", ".join(missing))
    records: list[dict[str, Any]] = []
    for row in stories.to_dict("records"):
        parsed = json.loads(str(row["footprint_cell_ids_json"] or "[]"))
        if not isinstance(parsed, list):
            raise ValueError("Story footprint cells must be a JSON list")
        for grid_id in sorted(set(str(value) for value in parsed)):
            records.append(_footprint_record(row, grid_id))
    if not memberships.empty and "grid_id" in memberships:
        story_dimensions = stories[[
            "timeline_bucket", "incident_id", "exposure_id", "crop_name", "hazard_family"
        ]].copy()
        story_dimensions["timeline_bucket"] = pd.to_datetime(
            story_dimensions["timeline_bucket"], errors="raise"
        ).dt.normalize()
        member_cells = memberships.dropna(subset=["grid_id"]).copy()
        member_cells["timeline_bucket"] = pd.to_datetime(
            member_cells["timeline_bucket"], errors="raise"
        ).dt.normalize()
        member_cells = member_cells.merge(
            story_dimensions,
            on=["timeline_bucket", "incident_id", "exposure_id", "hazard_family"],
            how="inner", validate="many_to_one",
        )
        for row in member_cells.to_dict("records"):
            records.append(_footprint_record(row, str(row["grid_id"])))
    columns = [
        "timeline_bucket", "incident_id", "exposure_id", "crop_name",
        "hazard_family", "grid_x", "grid_y",
    ]
    return pd.DataFrame(records, columns=columns).drop_duplicates()


def _footprint_record(row: dict[str, Any], grid_id: str) -> dict[str, Any]:
    x_value, y_value = _grid_coordinate(grid_id)
    return {
        "timeline_bucket": pd.Timestamp(row["timeline_bucket"]).date(),
        "incident_id": str(row["incident_id"]),
        "exposure_id": str(row["exposure_id"]),
        "crop_name": _crop(row["crop_name"]),
        "hazard_family": str(row["hazard_family"]),
        "grid_x": x_value,
        "grid_y": y_value,
    }


def _normalize_memberships(memberships: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timeline_bucket", "incident_id", "exposure_id", "crop_name_normalized",
        "hazard_family", "stage_bucket", "crop_instance_id", "membership_role",
        "event_state", "grid_id",
    }
    source = memberships.copy()
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("incident memberships are missing: " + ", ".join(missing))
    source["timeline_bucket"] = pd.to_datetime(
        source["timeline_bucket"], errors="raise"
    ).dt.date
    source["crop_name"] = source["crop_name_normalized"].map(_crop)
    return source


def _grid_coordinate(value: str) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) != 3 or parts[0] != "g":
        raise ValueError(f"Invalid grid ID: {value}")
    return int(parts[1]), int(parts[2])


def _crop(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(value or "unknown_crop").strip().lower()
    ).strip("_") or "unknown_crop"


__all__ = [
    "SUMMARY_COLUMNS",
    "SUMMARY_SCHEMA_VERSION",
    "build_incident_stage_summary",
    "enrich_incident_weekly_state",
]
