"""Export immutable crop-incident V3 artifacts for the story-map viewer.

The exporter is deliberately a projection, not another clustering pass.  It
keeps ``incident_id`` as the stable story identity, retains the V3 evidence
columns, and materializes exact unions of the metric grid rectangles recorded
in ``footprint_cell_ids_json``.  It never infers movement from centroids and it
never substitutes a convex hull for the tracked footprint.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import duckdb
import pandas as pd
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from build_story_map_bundle import (
    DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
    DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
    GEOMETRY_OUTPUT_COLUMNS,
    advisory_build_lock,
    build_geometry,
    validate_coverage_threshold,
)

from .incident_validation_v3 import file_sha256, validate_source_generation


SCHEMA_VERSION = "crop-incident-viewer-v3/1"
MODE = "crop_incident_v3"
PUBLICATION_STATUS = "diagnostic_uncalibrated_not_map_approved"

REQUIRED_INCIDENT_FILES = (
    "manifest.json",
    "field_week_context.parquet",
    "event_week_lanes.parquet",
    "weekly_exposure_cells.parquet",
    "incident_catalog.parquet",
    "incident_weekly_state.parquet",
    "incident_stage_summary.parquet",
    "incident_membership.parquet",
    "incident_windows.parquet",
    "exposure_component_assignments.parquet",
    "exposure_links.parquet",
    "exposure_weekly_state.parquet",
    "incident_lineage.parquet",
)

DRILLDOWN_COPIES = (
    "incident_catalog.parquet",
    "incident_weekly_state.parquet",
    "incident_stage_summary.parquet",
    "incident_membership.parquet",
    "incident_windows.parquet",
    "exposure_component_assignments.parquet",
    "exposure_links.parquet",
    "exposure_weekly_state.parquet",
    "incident_lineage.parquet",
)

COMPATIBLE_SCHEMAS: dict[str, set[str]] = {
    "frame_fields.parquet": {
        "timeline_bucket",
        "field_id",
        "story_cluster_id",
        "max_risk_band",
        "hazard_signature",
        "response_signature",
        "reportable_day_count",
        "event_count",
        "max_risk_rank",
        "response_day_count",
    },
    "cluster_labels.parquet": {
        "story_cluster_id",
        "short_label",
        "max_risk_band",
        "hazard_signature",
        "response_signature",
        "event_count",
        "field_count",
        "crop_count",
        "median_window_span_days",
        "median_reportable_days",
    },
    "event_windows.parquet": {
        "field_id",
        "crop_name",
        "crop_season",
        "event_id",
        "event_start_date",
        "active_end_date",
        "max_risk_band",
        "hazard_signature",
        "stage_signature",
        "response_signature",
        "close_reason",
        "reportable_days",
        "window_span_days",
        "story_cluster_id",
    },
    "story_day_membership.parquet": {
        "field_id",
        "event_id",
        "story_cluster_id",
    },
}


def export_incident_viewer_v3(
    incident_dir: Path,
    source_generation_dir: Path,
    output_dir: Path,
    *,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
    min_valid_geometry_coverage: float = DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
    min_frame_geometry_coverage: float = DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
) -> dict[str, Any]:
    """Build and atomically publish a server-compatible incident viewer bundle."""
    incident_dir = incident_dir.expanduser().resolve()
    source_generation_dir = source_generation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be positive")
    min_valid_geometry_coverage = validate_coverage_threshold(
        "min_valid_geometry_coverage", min_valid_geometry_coverage
    )
    min_frame_geometry_coverage = validate_coverage_threshold(
        "min_frame_geometry_coverage", min_frame_geometry_coverage
    )
    _validate_paths(incident_dir, source_generation_dir, output_dir, temp_dir)
    source_manifest = validate_source_generation(source_generation_dir)
    incident_manifest = _validate_incident_source(
        incident_dir, source_generation_dir, source_manifest
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with advisory_build_lock(output_dir):
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Immutable viewer output already exists: {output_dir}")
        with TemporaryDirectory(prefix=".incident-viewer-v3-", dir=output_dir.parent) as temporary:
            transaction = Path(temporary)
            stage = transaction / output_dir.name
            stage.mkdir()
            build_geometry(
                source_generation_dir,
                stage,
                profile_output=Path("field_geometry.parquet"),
                min_valid_coverage=min_valid_geometry_coverage,
            )
            _sanitize_geometry_profile(stage / "geometry_profile.json")
            with _connection(threads, memory_limit, temp_dir) as connection:
                _register_inputs(connection, incident_dir, source_generation_dir)
                _write_frame_fields(connection, stage / "frame_fields.parquet")
                _write_event_windows(connection, stage / "event_windows.parquet")
                connection.read_parquet(str(stage / "event_windows.parquet")).create_view(
                    "viewer_event_windows"
                )
                _write_cluster_labels(connection, stage / "cluster_labels.parquet")
                _write_story_day_membership(
                    connection, stage / "story_day_membership.parquet"
                )
                summaries = stage / "gpu_summaries"
                summaries.mkdir()
                _write_timeline_summary(
                    connection,
                    stage / "frame_fields.parquet",
                    summaries / "timeline_summary.parquet",
                )
            _write_incident_footprints(incident_dir, stage / "incident_footprints.parquet")
            _copy_drilldown_artifacts(incident_dir, stage)
            validation = _validate_stage(
                stage,
                incident_dir,
                min_frame_geometry_coverage=min_frame_geometry_coverage,
            )
            manifest = _build_manifest(
                incident_dir=incident_dir,
                source_generation_dir=source_generation_dir,
                stage=stage,
                incident_manifest=incident_manifest,
                source_manifest=source_manifest,
                validation=validation,
            )
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if output_dir.exists() or output_dir.is_symlink():
                raise FileExistsError(
                    f"Immutable viewer output appeared during build: {output_dir}"
                )
            os.replace(stage, output_dir)

    return {
        "status": "complete",
        "mode": MODE,
        "schema_version": SCHEMA_VERSION,
        "viewer_bundle_id": manifest["run"]["viewer_bundle_id"],
        "output_dir": str(output_dir),
        "counts": validation["counts"],
        "publication_status": PUBLICATION_STATUS,
        "warning": manifest["warning"],
    }


@contextmanager
def _connection(
    threads: int, memory_limit: str | None, temp_dir: Path | None
) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads = ?", [threads])
        if memory_limit:
            connection.execute("SET memory_limit = ?", [str(memory_limit)])
        if temp_dir:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory = ?", [str(resolved)])
        yield connection
    finally:
        connection.close()


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    incident_dir: Path,
    source_generation_dir: Path,
) -> None:
    inputs = {
        "field_week_context": incident_dir / "field_week_context.parquet",
        "incident_membership": incident_dir / "incident_membership.parquet",
        "incident_weekly_state": incident_dir / "incident_weekly_state.parquet",
        "incident_windows_source": incident_dir / "incident_windows.parquet",
        "event_week_lanes": incident_dir / "event_week_lanes.parquet",
        "source_story_days": source_generation_dir / "story_day_membership.parquet",
    }
    for view, path in inputs.items():
        connection.read_parquet(str(path)).create_view(view)
    _require_columns(
        connection,
        "field_week_context",
        {"timeline_bucket", "crop_instance_id", "crop_name", "stage_bucket"},
    )
    _require_columns(
        connection,
        "incident_membership",
        {
            "timeline_bucket", "incident_id", "exposure_id", "component_id",
            "crop_name_normalized", "hazard_family", "field_id",
            "crop_instance_id", "episode_id", "membership_role", "event_state",
            "response_class", "fresh_response_evidence", "evaluable",
            "is_data_gap", "stage_bucket", "grid_id",
            "knowledge_time",
        },
    )
    _require_columns(
        connection,
        "incident_weekly_state",
        {
            "timeline_bucket", "incident_id", "exposure_id", "crop_name",
            "hazard_family", "incident_state", "footprint_cell_ids_json",
            "pressure_core_field_count", "severe_field_count",
            "watch_frontier_field_count", "impact_lag_field_count",
            "fresh_decline_field_count", "fresh_recovery_field_count",
            "stage_distribution", "coverage_adequate", "footprint_carried_forward",
            "footprint_area_km2", "right_censored", "monitored_count",
            "evaluable_count", "pressure_core_count", "severe_count",
            "impact_lag_count", "global_crop_week_unmappable_instance_count",
            "active_count",
            "affected_count",
        },
    )
    _require_columns(
        connection,
        "incident_windows_source",
        {
            "incident_id", "exposure_id", "crop_name", "hazard_family",
            "terminal_state", "right_censored", "observed_week_count",
        },
    )
    _require_columns(
        connection,
        "event_week_lanes",
        {
            "timeline_bucket", "event_id", "field_id", "crop_name",
            "snapshot_as_of_date", "crop_season", "stage_bucket",
            "hazard_family", "event_state",
            "event_start_date", "event_end_date", "close_reason",
            "current_risk_rank", "current_risk_band", "max_risk_rank",
            "max_risk_band", "daily_response_class", "fresh_response_evidence",
            "reportable_day_count", "response_day_count",
        },
    )
    _require_columns(
        connection,
        "source_story_days",
        {"field_id", "event_id", "crop_instance_id", "observation_date"},
    )


def _write_frame_fields(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    connection.execute(
        """
        COPY (
            SELECT
                CAST(m.timeline_bucket AS DATE) AS timeline_bucket,
                TRIM(CAST(m.field_id AS VARCHAR)) AS field_id,
                CAST(m.incident_id AS VARCHAR) AS story_cluster_id,
                CAST(m.incident_id AS VARCHAR) AS incident_id,
                CAST(m.exposure_id AS VARCHAR) AS exposure_id,
                CAST(m.component_id AS VARCHAR) AS component_id,
                CAST(m.episode_id AS VARCHAR) AS event_id,
                CAST(m.episode_id AS VARCHAR) AS episode_id,
                CAST(m.crop_instance_id AS VARCHAR) AS crop_instance_id,
                COALESCE(NULLIF(CAST(c.crop_name AS VARCHAR), ''),
                    NULLIF(CAST(l.crop_name AS VARCHAR), ''),
                    CAST(m.crop_name_normalized AS VARCHAR)) AS crop_name,
                COALESCE(NULLIF(CAST(c.stage_bucket AS VARCHAR), ''),
                    CAST(m.stage_bucket AS VARCHAR), 'unknown') AS stage_bucket,
                CAST(m.membership_role AS VARCHAR) AS membership_role,
                UPPER(CAST(m.event_state AS VARCHAR)) AS event_state,
                UPPER(CAST(m.event_state AS VARCHAR)) AS field_event_state,
                CAST(s.incident_state AS VARCHAR) AS incident_state,
                CAST(m.hazard_family AS VARCHAR) AS hazard_signature,
                CAST(m.hazard_family AS VARCHAR) AS motif_family,
                CAST(m.response_class AS VARCHAR) AS response_signature,
                CAST(m.response_class AS VARCHAR) AS response_class,
                COALESCE(NULLIF(CAST(l.max_risk_band AS VARCHAR), ''),
                    CASE UPPER(CAST(m.event_state AS VARCHAR))
                        WHEN 'SEVERE' THEN 'HIGH'
                        WHEN 'ACTIVE' THEN 'MED-HIGH'
                        WHEN 'WATCH' THEN 'LOW-MED'
                        ELSE 'NONE'
                    END) AS max_risk_band,
                COALESCE(NULLIF(CAST(l.current_risk_band AS VARCHAR), ''),
                    NULLIF(CAST(l.max_risk_band AS VARCHAR), ''), 'NONE')
                    AS current_risk_band,
                COALESCE(TRY_CAST(l.max_risk_rank AS INTEGER),
                    CASE UPPER(CAST(m.event_state AS VARCHAR))
                        WHEN 'SEVERE' THEN 4 WHEN 'ACTIVE' THEN 3
                        WHEN 'WATCH' THEN 2 WHEN 'RECOVERING' THEN 1 ELSE 0 END)
                    AS max_risk_rank,
                COALESCE(TRY_CAST(l.current_risk_rank AS INTEGER),
                    TRY_CAST(l.max_risk_rank AS INTEGER), 0) AS current_risk_rank,
                GREATEST(COALESCE(TRY_CAST(l.reportable_day_count AS BIGINT), 0), 0)
                    AS reportable_day_count,
                CAST(1 AS BIGINT) AS event_count,
                GREATEST(COALESCE(TRY_CAST(l.response_day_count AS BIGINT), 0), 0)
                    AS response_day_count,
                COALESCE(TRY_CAST(m.fresh_response_evidence AS BOOLEAN), FALSE)
                    AS fresh_response_evidence,
                COALESCE(TRY_CAST(l.fresh_response_evidence AS BOOLEAN), FALSE)
                    AS lane_fresh_response_evidence,
                COALESCE(TRY_CAST(m.evaluable AS BOOLEAN), FALSE) AS evaluable,
                COALESCE(TRY_CAST(m.is_data_gap AS BOOLEAN), FALSE) AS is_data_gap,
                CAST(m.knowledge_time AS DATE) AS knowledge_time,
                CAST(m.grid_id AS VARCHAR) AS grid_id,
                COALESCE(TRY_CAST(s.coverage_adequate AS BOOLEAN), FALSE)
                    AS coverage_adequate,
                COALESCE(TRY_CAST(s.right_censored AS BOOLEAN), FALSE) AS right_censored,
                COALESCE(TRY_CAST(s.footprint_carried_forward AS BOOLEAN), FALSE)
                    AS footprint_carried_forward,
                COALESCE(TRY_CAST(s.fresh_decline_field_count AS BIGINT), 0)
                    AS fresh_decline_field_count,
                COALESCE(TRY_CAST(s.fresh_recovery_field_count AS BIGINT), 0)
                    AS fresh_recovery_field_count,
                COALESCE(TRY_CAST(s.monitored_count AS BIGINT), 0) AS monitored_count,
                COALESCE(TRY_CAST(s.evaluable_count AS BIGINT), 0) AS evaluable_count,
                COALESCE(TRY_CAST(s.affected_count AS BIGINT), 0) AS affected_count,
                CAST(FALSE AS BOOLEAN) AS is_physical_movement
            FROM incident_membership AS m
            JOIN incident_weekly_state AS s
              ON CAST(s.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
             AND CAST(s.incident_id AS VARCHAR) = CAST(m.incident_id AS VARCHAR)
            LEFT JOIN field_week_context AS c
              ON CAST(c.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
             AND CAST(c.crop_instance_id AS VARCHAR)
                    = CAST(m.crop_instance_id AS VARCHAR)
            LEFT JOIN event_week_lanes AS l
              ON CAST(l.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
             AND CAST(l.event_id AS VARCHAR) = CAST(m.episode_id AS VARCHAR)
             AND CAST(l.field_id AS VARCHAR) = CAST(m.field_id AS VARCHAR)
            ORDER BY timeline_bucket, story_cluster_id, field_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_event_windows(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    connection.execute(
        """
        COPY (
            WITH grouped AS (
                SELECT
                    CAST(m.incident_id AS VARCHAR) AS incident_id,
                    CAST(m.exposure_id AS VARCHAR) AS exposure_id,
                    TRIM(CAST(m.field_id AS VARCHAR)) AS field_id,
                    CAST(m.episode_id AS VARCHAR) AS event_id,
                    COALESCE(NULLIF(MAX(CAST(l.crop_name AS VARCHAR)), ''),
                        MAX(CAST(m.crop_name_normalized AS VARCHAR))) AS crop_name,
                    COALESCE(NULLIF(MAX(CAST(l.crop_season AS VARCHAR)), ''),
                        'unknown') AS crop_season,
                    MIN(COALESCE(TRY_CAST(l.event_start_date AS DATE),
                        CAST(m.timeline_bucket AS DATE))) AS event_start_date,
                    MAX(COALESCE(TRY_CAST(l.event_end_date AS DATE),
                        TRY_CAST(l.snapshot_as_of_date AS DATE),
                        CAST(m.timeline_bucket AS DATE) + INTERVAL 6 DAY))::DATE
                        AS active_end_date,
                    ARG_MAX(
                        COALESCE(NULLIF(CAST(l.max_risk_band AS VARCHAR), ''), 'NONE'),
                        COALESCE(TRY_CAST(l.max_risk_rank AS INTEGER), 0)
                    ) AS max_risk_band,
                    MAX(CAST(m.hazard_family AS VARCHAR)) AS hazard_signature,
                    STRING_AGG(DISTINCT CAST(m.stage_bucket AS VARCHAR), ' -> '
                        ORDER BY CAST(m.stage_bucket AS VARCHAR)) AS stage_signature,
                    STRING_AGG(DISTINCT CAST(m.response_class AS VARCHAR), ' | '
                        ORDER BY CAST(m.response_class AS VARCHAR)) AS response_signature,
                    COALESCE(NULLIF(MAX(CAST(l.close_reason AS VARCHAR)), ''),
                        MAX(CAST(w.terminal_state AS VARCHAR))) AS close_reason,
                    MAX(GREATEST(COALESCE(TRY_CAST(l.reportable_day_count AS BIGINT), 0), 0))
                        AS reportable_days,
                    MAX(CAST(w.terminal_state AS VARCHAR)) AS incident_state,
                    BOOL_OR(COALESCE(TRY_CAST(m.fresh_response_evidence AS BOOLEAN), FALSE))
                        AS fresh_response_evidence,
                    BOOL_OR(COALESCE(TRY_CAST(m.evaluable AS BOOLEAN), FALSE)) AS evaluable,
                    BOOL_OR(COALESCE(TRY_CAST(m.is_data_gap AS BOOLEAN), FALSE)) AS is_data_gap
                FROM incident_membership AS m
                JOIN incident_windows_source AS w
                  ON CAST(w.incident_id AS VARCHAR) = CAST(m.incident_id AS VARCHAR)
                LEFT JOIN event_week_lanes AS l
                  ON CAST(l.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
                 AND CAST(l.event_id AS VARCHAR) = CAST(m.episode_id AS VARCHAR)
                 AND CAST(l.field_id AS VARCHAR) = CAST(m.field_id AS VARCHAR)
                GROUP BY m.incident_id, m.exposure_id, m.field_id, m.episode_id
            )
            SELECT
                field_id, crop_name, crop_season, event_id,
                event_start_date, active_end_date, max_risk_band,
                hazard_signature, stage_signature, response_signature,
                close_reason, reportable_days,
                DATE_DIFF('day', event_start_date, active_end_date) + 1 AS window_span_days,
                incident_id AS story_cluster_id,
                incident_id, exposure_id, incident_state,
                fresh_response_evidence, evaluable, is_data_gap
            FROM grouped
            ORDER BY story_cluster_id, event_start_date, field_id, event_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_cluster_labels(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    connection.execute(
        """
        COPY (
            WITH window_stats AS (
                SELECT
                    CAST(w.incident_id AS VARCHAR) AS incident_id,
                    MAX(CAST(w.exposure_id AS VARCHAR)) AS exposure_id,
                    MAX(CAST(w.crop_name AS VARCHAR)) AS crop_name,
                    MAX(CAST(w.hazard_family AS VARCHAR)) AS hazard_family,
                    MAX(CAST(w.terminal_state AS VARCHAR)) AS terminal_state,
                    BOOL_OR(COALESCE(TRY_CAST(w.right_censored AS BOOLEAN), FALSE))
                        AS right_censored,
                    MAX(COALESCE(TRY_CAST(w.observed_week_count AS BIGINT), 0))
                        AS observed_week_count
                FROM incident_windows_source AS w
                GROUP BY w.incident_id
            ), event_stats AS (
                SELECT
                    story_cluster_id AS incident_id,
                    ARG_MAX(max_risk_band,
                        CASE UPPER(max_risk_band)
                            WHEN 'EXTREME' THEN 5 WHEN 'HIGH' THEN 4
                            WHEN 'MED-HIGH' THEN 3 WHEN 'MEDIUM' THEN 3
                            WHEN 'LOW-MED' THEN 2 WHEN 'LOW' THEN 1 ELSE 0 END)
                        AS max_risk_band,
                    MAX(hazard_signature) AS hazard_signature,
                    STRING_AGG(DISTINCT response_signature, ' | '
                        ORDER BY response_signature) AS response_signature,
                    COUNT(DISTINCT event_id) AS event_count,
                    COUNT(DISTINCT field_id) AS field_count,
                    MEDIAN(window_span_days) AS median_window_span_days,
                    MEDIAN(reportable_days) AS median_reportable_days
                FROM viewer_event_windows
                GROUP BY story_cluster_id
            )
            SELECT
                s.incident_id AS story_cluster_id,
                REPLACE(s.hazard_family, '_', ' ') || ' / '
                    || REPLACE(s.crop_name, '_', ' ') || ' incident' AS short_label,
                e.max_risk_band,
                e.hazard_signature,
                e.response_signature,
                e.event_count,
                e.field_count,
                CAST(1 AS BIGINT) AS crop_count,
                e.median_window_span_days,
                e.median_reportable_days,
                s.incident_id,
                s.exposure_id,
                s.crop_name,
                s.terminal_state,
                s.right_censored,
                s.observed_week_count,
                CAST(? AS VARCHAR) AS publication_status
            FROM window_stats AS s
            JOIN event_stats AS e USING (incident_id)
            ORDER BY e.event_count DESC, e.field_count DESC, s.incident_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        # DuckDB binds COPY's destination before parameters inside its query.
        [str(output_path), PUBLICATION_STATUS],
    )


def _write_story_day_membership(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    source_columns = _columns(connection, "source_story_days")
    state_sql = _optional_sql(
        source_columns, "d", "event_state", "CAST(m.event_state AS VARCHAR)"
    )
    pressure_sql = _optional_sql(
        source_columns, "d", "daily_pressure_rank", "COALESCE(l.current_risk_rank, 0)"
    )
    response_sql = _optional_sql(
        source_columns, "d", "daily_response_class", "CAST(m.response_class AS VARCHAR)"
    )
    observed_sql = _optional_sql(
        source_columns,
        "d",
        "pressure_observed",
        "UPPER(CAST(m.event_state AS VARCHAR)) IN ('ACTIVE', 'SEVERE')",
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                TRIM(CAST(d.field_id AS VARCHAR)) AS field_id,
                CAST(d.event_id AS VARCHAR) AS event_id,
                CAST(m.incident_id AS VARCHAR) AS story_cluster_id,
                CAST(m.incident_id AS VARCHAR) AS incident_id,
                CAST(m.exposure_id AS VARCHAR) AS exposure_id,
                CAST(d.crop_instance_id AS VARCHAR) AS crop_instance_id,
                CAST(d.observation_date AS DATE) AS observation_date,
                DATE_TRUNC('week', CAST(d.observation_date AS DATE))::DATE
                    AS timeline_bucket,
                CAST(m.crop_name_normalized AS VARCHAR) AS crop_name,
                CAST(m.stage_bucket AS VARCHAR) AS stage_bucket,
                CAST(m.membership_role AS VARCHAR) AS membership_role,
                CAST({state_sql} AS VARCHAR) AS event_state,
                CAST(s.incident_state AS VARCHAR) AS incident_state,
                CAST(m.hazard_family AS VARCHAR) AS hazard_signature,
                COALESCE(TRY_CAST({pressure_sql} AS INTEGER), 0) AS daily_pressure_rank,
                COALESCE(TRY_CAST({pressure_sql} AS INTEGER), 0) AS risk_rank,
                CAST({response_sql} AS VARCHAR) AS daily_response_class,
                COALESCE(TRY_CAST({observed_sql} AS BOOLEAN), FALSE) AS pressure_observed,
                COALESCE(TRY_CAST({observed_sql} AS BOOLEAN), FALSE)
                    AS is_reportable_story_day,
                COALESCE(TRY_CAST(m.fresh_response_evidence AS BOOLEAN), FALSE)
                    AS fresh_response_evidence,
                COALESCE(TRY_CAST(m.evaluable AS BOOLEAN), FALSE) AS evaluable,
                COALESCE(TRY_CAST(m.is_data_gap AS BOOLEAN), FALSE) AS is_data_gap
            FROM source_story_days AS d
            JOIN incident_membership AS m
              ON CAST(m.timeline_bucket AS DATE)
                    = DATE_TRUNC('week', CAST(d.observation_date AS DATE))::DATE
             AND CAST(m.field_id AS VARCHAR) = CAST(d.field_id AS VARCHAR)
             AND CAST(m.episode_id AS VARCHAR) = CAST(d.event_id AS VARCHAR)
            JOIN incident_weekly_state AS s
              ON CAST(s.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
             AND CAST(s.incident_id AS VARCHAR) = CAST(m.incident_id AS VARCHAR)
            LEFT JOIN event_week_lanes AS l
              ON CAST(l.timeline_bucket AS DATE) = CAST(m.timeline_bucket AS DATE)
             AND CAST(l.event_id AS VARCHAR) = CAST(m.episode_id AS VARCHAR)
             AND CAST(l.field_id AS VARCHAR) = CAST(m.field_id AS VARCHAR)
            ORDER BY observation_date, story_cluster_id, field_id, event_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_timeline_summary(
    connection: duckdb.DuckDBPyConnection,
    frame_path: Path,
    output_path: Path,
) -> None:
    connection.execute(
        """
        COPY (
            WITH incident_buckets AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    COUNT(DISTINCT CAST(incident_id AS VARCHAR))
                        AS story_cluster_count,
                    COUNT(DISTINCT CAST(exposure_id AS VARCHAR)) AS exposure_count
                FROM incident_weekly_state
                GROUP BY timeline_bucket
            ), frame_buckets AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    COUNT(DISTINCT field_id) AS field_count,
                    SUM(reportable_day_count) AS reportable_day_count,
                    SUM(event_count) AS event_count,
                    MAX(max_risk_rank) AS max_risk_rank,
                    COUNT_IF(fresh_response_evidence) AS fresh_response_field_count,
                    COUNT_IF(is_data_gap) AS data_gap_field_count
                FROM read_parquet(?)
                GROUP BY timeline_bucket
            )
            SELECT
                i.timeline_bucket,
                COALESCE(f.field_count, 0) AS field_count,
                i.story_cluster_count,
                COALESCE(f.reportable_day_count, 0) AS reportable_day_count,
                COALESCE(f.event_count, 0) AS event_count,
                COALESCE(f.max_risk_rank, 0) AS max_risk_rank,
                i.exposure_count,
                COALESCE(f.fresh_response_field_count, 0)
                    AS fresh_response_field_count,
                COALESCE(f.data_gap_field_count, 0) AS data_gap_field_count
            FROM incident_buckets AS i
            LEFT JOIN frame_buckets AS f USING (timeline_bucket)
            ORDER BY i.timeline_bucket
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        # DuckDB binds COPY's destination before parameters inside its query.
        [str(output_path), str(frame_path)],
    )


def _write_incident_footprints(incident_dir: Path, output_path: Path) -> None:
    cells_path = incident_dir / "weekly_exposure_cells.parquet"
    weekly_path = incident_dir / "incident_weekly_state.parquet"
    with duckdb.connect(":memory:") as connection:
        cells = connection.execute(
            """
            SELECT DISTINCT
                'g:' || CAST(cell_x AS VARCHAR) || ':' || CAST(cell_y AS VARCHAR) AS grid_id,
                TRY_CAST(min_lon AS DOUBLE) AS min_lon,
                TRY_CAST(min_lat AS DOUBLE) AS min_lat,
                TRY_CAST(max_lon AS DOUBLE) AS max_lon,
                TRY_CAST(max_lat AS DOUBLE) AS max_lat,
                TRY_CAST(cell_size_km AS DOUBLE) AS cell_size_km
            FROM read_parquet(?)
            ORDER BY grid_id, min_lon, min_lat, max_lon, max_lat
            """,
            [str(cells_path)],
        ).fetchdf()
        weekly = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timeline_bucket, incident_id",
            [str(weekly_path)],
        ).fetchdf()
    if weekly.empty:
        raise ValueError("incident_weekly_state must contain at least one story week")
    if weekly.duplicated(["timeline_bucket", "incident_id"]).any():
        raise ValueError("incident_weekly_state is not unique by story and week")
    catalog: dict[str, tuple[float, float, float, float, float]] = {}
    for grid_id, rows in cells.groupby("grid_id", sort=True):
        variants = {
            tuple(float(row[name]) for name in (
                "min_lon", "min_lat", "max_lon", "max_lat", "cell_size_km"
            ))
            for row in rows.to_dict("records")
        }
        if len(variants) != 1:
            raise ValueError(f"Grid cell {grid_id} has changing bounds or size")
        values = next(iter(variants))
        min_lon, min_lat, max_lon, max_lat, cell_size_km = values
        if (
            not all(math.isfinite(value) for value in values)
            or min_lon >= max_lon
            or min_lat >= max_lat
            or cell_size_km <= 0
        ):
            raise ValueError(f"Grid cell {grid_id} has invalid geometry")
        catalog[str(grid_id)] = values

    geometry_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def exact_geometry(cell_ids: list[str]) -> dict[str, Any]:
        missing = sorted(set(cell_ids) - set(catalog))
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"Incident footprint references {len(missing)} unknown grid cells: {sample}"
            )
        key = tuple(cell_ids)
        cached = geometry_cache.get(key)
        if cached is not None:
            return cached
        rectangles = [box(*catalog[cell_id][:4]) for cell_id in key]
        geometry = unary_union(rectangles)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Exact grid-cell union produced invalid geometry")
        min_lon, min_lat, max_lon, max_lat = (
            float(value) for value in geometry.bounds
        )
        centroid = geometry.centroid
        cached = {
            "geometry_geojson": json.dumps(
                mapping(geometry), separators=(",", ":"), sort_keys=True
            ),
            "geometry_type": geometry.geom_type,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "centroid_lon": float(centroid.x),
            "centroid_lat": float(centroid.y),
            "footprint_area_km2": float(
                sum(catalog[cell_id][4] ** 2 for cell_id in key)
            ),
        }
        geometry_cache[key] = cached
        return cached

    records: list[dict[str, Any]] = []
    for item in weekly.to_dict("records"):
        cell_ids = _json_string_list(item.get("footprint_cell_ids_json"))
        if not cell_ids:
            raise ValueError(
                f"Incident {item.get('incident_id')} has an empty footprint at "
                f"{item.get('timeline_bucket')}"
            )
        key = tuple(cell_ids)
        cached = exact_geometry(cell_ids)
        record = dict(item)
        record["tracker_footprint_area_km2"] = record.get("footprint_area_km2")
        record.update(cached)
        record.update(
            {
                "story_cluster_id": str(item["incident_id"]),
                "cell_ids_json": json.dumps(key, separators=(",", ":")),
                "footprint_cell_ids_json": json.dumps(key, separators=(",", ":")),
                "footprint_cell_count": len(key),
                "stage_bucket": _dominant_stage(item.get("stage_distribution")),
                "fresh_decline_evidence": int(item.get("fresh_decline_field_count") or 0) > 0,
                "fresh_recovery_evidence": int(item.get("fresh_recovery_field_count") or 0) > 0,
                "pressure_core_evidence": int(item.get("pressure_core_field_count") or 0) > 0,
                "denominator_scope": (
                    "monitored_and_evaluable_crop_instances_by_story_footprint_and_stage"
                ),
                "footprint_geometry_method": "exact_union_of_grid_rectangles",
                "is_physical_movement": False,
                "low_zoom_omitted": False,
            }
        )
        role_cells = {
            "pressure": _optional_json_string_list(item.get("pressure_cell_ids_json")),
            "impact": _optional_json_string_list(item.get("impact_cell_ids_json")),
            "watch": _optional_json_string_list(item.get("watch_cell_ids_json")),
        }
        if not role_cells["pressure"] and int(item.get("pressure_core_field_count") or 0) > 0:
            role_cells["pressure"] = list(cell_ids)
        if not role_cells["impact"] and int(item.get("impact_lag_field_count") or 0) > 0:
            role_cells["impact"] = list(cell_ids)
        for role, values in role_cells.items():
            record[f"{role}_cell_ids_json"] = json.dumps(
                values, separators=(",", ":")
            )
            record[f"{role}_cell_count"] = len(values)
            if values:
                role_geometry = exact_geometry(values)
                record[f"{role}_geometry_geojson"] = role_geometry[
                    "geometry_geojson"
                ]
                record[f"{role}_geometry_type"] = role_geometry["geometry_type"]
                record[f"{role}_area_km2"] = role_geometry["footprint_area_km2"]
            else:
                record[f"{role}_geometry_geojson"] = None
                record[f"{role}_geometry_type"] = None
                record[f"{role}_area_km2"] = 0.0
        records.append(record)
    footprints = pd.DataFrame(records)
    footprints["coincident_incident_count"] = 1
    footprints["coincident_incident_index"] = 0
    footprints["coincident_crop_names_json"] = "[]"
    footprints["coincident_incident_ids_json"] = "[]"
    footprints["coincident_group_id"] = ""
    for _, indices in footprints.groupby(
        ["timeline_bucket", "footprint_cell_ids_json"], sort=True
    ).groups.items():
        ordered = footprints.loc[list(indices)].sort_values(
            ["crop_name", "incident_id"], kind="mergesort"
        )
        crops = sorted(set(ordered["crop_name"].astype(str)))
        incident_ids = ordered["incident_id"].astype(str).tolist()
        group_payload = (
            str(ordered.iloc[0]["timeline_bucket"])
            + "\0"
            + str(ordered.iloc[0]["footprint_cell_ids_json"])
        )
        group_id = "coincident_" + hashlib.sha256(
            group_payload.encode("utf-8")
        ).hexdigest()[:20]
        footprints.loc[ordered.index, "coincident_incident_count"] = len(ordered)
        footprints.loc[ordered.index, "coincident_incident_index"] = range(len(ordered))
        footprints.loc[ordered.index, "coincident_crop_names_json"] = json.dumps(
            crops, separators=(",", ":")
        )
        footprints.loc[ordered.index, "coincident_incident_ids_json"] = json.dumps(
            incident_ids, separators=(",", ":")
        )
        footprints.loc[ordered.index, "coincident_group_id"] = group_id
    footprints.to_parquet(
        output_path, index=False, compression="zstd"
    )


def _copy_drilldown_artifacts(incident_dir: Path, stage: Path) -> None:
    for name in DRILLDOWN_COPIES:
        shutil.copy2(incident_dir / name, stage / name)


def _validate_stage(
    stage: Path,
    incident_dir: Path,
    *,
    min_frame_geometry_coverage: float,
) -> dict[str, Any]:
    required_paths = [
        stage / "field_geometry.parquet",
        stage / "incident_footprints.parquet",
        stage / "gpu_summaries" / "timeline_summary.parquet",
        *(stage / name for name in COMPATIBLE_SCHEMAS),
    ]
    missing = [str(path.relative_to(stage)) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Viewer stage is missing: " + ", ".join(missing))
    counts: dict[str, int] = {}
    with duckdb.connect(":memory:") as connection:
        for name, schema in COMPATIBLE_SCHEMAS.items():
            path = stage / name
            columns = _parquet_columns(connection, path)
            absent = sorted(schema - columns)
            if absent:
                raise ValueError(f"{name} is missing columns: {', '.join(absent)}")
            count = _row_count(connection, path)
            if count < 1:
                raise ValueError(f"{name} must contain at least one row")
            counts[name] = count
        geometry_columns = _parquet_columns(connection, stage / "field_geometry.parquet")
        absent_geometry = sorted(set(GEOMETRY_OUTPUT_COLUMNS) - geometry_columns)
        if absent_geometry:
            raise ValueError(
                "field_geometry.parquet is missing columns: " + ", ".join(absent_geometry)
            )
        counts["field_geometry.parquet"] = _row_count(
            connection, stage / "field_geometry.parquet"
        )
        counts["incident_footprints.parquet"] = _row_count(
            connection, stage / "incident_footprints.parquet"
        )
        counts["gpu_summaries/timeline_summary.parquet"] = _row_count(
            connection, stage / "gpu_summaries" / "timeline_summary.parquet"
        )
        for name in DRILLDOWN_COPIES:
            counts[name] = _row_count(connection, stage / name)

        unique_keys = {
            "frame_fields.parquet": ("timeline_bucket", "story_cluster_id", "field_id"),
            "cluster_labels.parquet": ("story_cluster_id",),
            "event_windows.parquet": ("story_cluster_id", "field_id", "event_id"),
            "story_day_membership.parquet": (
                "story_cluster_id", "field_id", "event_id", "observation_date",
            ),
            "incident_footprints.parquet": ("timeline_bucket", "incident_id"),
            "gpu_summaries/timeline_summary.parquet": ("timeline_bucket",),
        }
        for name, key in unique_keys.items():
            path = stage / name
            duplicates = _duplicate_key_count(connection, path, key)
            if duplicates:
                raise ValueError(f"{name} contains {duplicates} duplicate natural keys")

        for name in (
            "frame_fields.parquet", "cluster_labels.parquet", "event_windows.parquet",
            "story_day_membership.parquet", "incident_footprints.parquet",
        ):
            path = stage / name
            mismatched = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM read_parquet(?)
                    WHERE incident_id IS NULL OR story_cluster_id IS NULL
                       OR CAST(incident_id AS VARCHAR) <> CAST(story_cluster_id AS VARCHAR)
                    """,
                    [str(path)],
                ).fetchone()[0]
            )
            if mismatched:
                raise ValueError(f"{name} does not preserve incident_id as story_cluster_id")

        frame_path = stage / "frame_fields.parquet"
        geometry_path = stage / "field_geometry.parquet"
        frame_fields, joined_fields, noncanonical = connection.execute(
            """
            WITH fields AS (
                SELECT DISTINCT field_id FROM read_parquet(?)
            )
            SELECT
                (SELECT COUNT(*) FROM fields),
                (SELECT COUNT(*) FROM fields f JOIN read_parquet(?) g USING (field_id)),
                (SELECT COUNT(*) FROM read_parquet(?)
                 WHERE field_id IS NULL OR TRIM(CAST(field_id AS VARCHAR)) = ''
                    OR CAST(field_id AS VARCHAR) <> TRIM(CAST(field_id AS VARCHAR)))
            """,
            [str(frame_path), str(geometry_path), str(frame_path)],
        ).fetchone()
        if noncanonical:
            raise ValueError("frame_fields contains blank or noncanonical field IDs")
        frame_geometry_coverage = joined_fields / frame_fields if frame_fields else 0.0
        if frame_geometry_coverage < min_frame_geometry_coverage:
            raise ValueError(
                "Frame-to-geometry field coverage is below the configured minimum: "
                f"{joined_fields}/{frame_fields} ({frame_geometry_coverage:.2%}) "
                f"< {min_frame_geometry_coverage:.2%}"
            )

        source_weekly = incident_dir / "incident_weekly_state.parquet"
        footprint_keys = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM read_parquet(?)) AS source_rows,
                (SELECT COUNT(*) FROM read_parquet(?)) AS footprint_rows,
                (SELECT COUNT(*)
                 FROM read_parquet(?) s
                 FULL OUTER JOIN read_parquet(?) f
                   ON CAST(s.timeline_bucket AS DATE) = CAST(f.timeline_bucket AS DATE)
                  AND CAST(s.incident_id AS VARCHAR) = CAST(f.incident_id AS VARCHAR)
                 WHERE s.incident_id IS NULL OR f.incident_id IS NULL) AS missing_keys,
                (SELECT COUNT(*) FROM read_parquet(?)
                 WHERE low_zoom_omitted OR footprint_geometry_method
                    <> 'exact_union_of_grid_rectangles') AS inexact_rows
            """,
            [
                str(source_weekly), str(stage / "incident_footprints.parquet"),
                str(source_weekly), str(stage / "incident_footprints.parquet"),
                str(stage / "incident_footprints.parquet"),
            ],
        ).fetchone()
        if footprint_keys[0] != footprint_keys[1] or footprint_keys[2] or footprint_keys[3]:
            raise ValueError(
                "Incident footprint completeness failed: "
                f"source={footprint_keys[0]}, exported={footprint_keys[1]}, "
                f"missing_keys={footprint_keys[2]}, inexact={footprint_keys[3]}"
            )

    profile_path = stage / "geometry_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.update(
        {
            "frame_field_count": int(frame_fields),
            "joined_frame_field_count": int(joined_fields),
            "frame_geometry_coverage": frame_geometry_coverage,
            "min_frame_geometry_coverage": min_frame_geometry_coverage,
        }
    )
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "passed": True,
        "counts": counts,
        "frame_geometry_coverage": frame_geometry_coverage,
        "complete_incident_footprints": True,
        "low_zoom_footprints_dropped": False,
    }


def _build_manifest(
    *,
    incident_dir: Path,
    source_generation_dir: Path,
    stage: Path,
    incident_manifest: dict[str, Any],
    source_manifest: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    incident_manifest_hash = file_sha256(incident_dir / "manifest.json")
    source_manifest_hash = file_sha256(source_generation_dir / "manifest.json")
    implementation_hash = file_sha256(Path(__file__).resolve())
    artifacts = _artifact_inventory(stage)
    bundle_content_hash = hashlib.sha256(
        json.dumps(
            {name: metadata["sha256"] for name, metadata in artifacts.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    identity = hashlib.sha256(
        (
            SCHEMA_VERSION + incident_manifest_hash + source_manifest_hash
            + implementation_hash + bundle_content_hash
        ).encode("ascii")
    ).hexdigest()[:20]
    policy = incident_manifest.get("policy") or {}
    source_run = source_manifest.get("run") or {}
    incident_run = incident_manifest.get("run") or {}
    warning = (
        "Incident V3 thresholds and labels are uncalibrated. This viewer bundle is "
        "diagnostic only and is not map-publication approved until tracking, lifecycle, "
        "and agronomic review gates pass."
    )
    counts = validation.get("counts") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "run": {
            "status": "complete",
            "mode": MODE,
            "viewer_bundle_id": f"incident_viewer_v3_{identity}",
            "immutable": True,
            "viewer_ready": True,
            "viewer_bundle_required": False,
            "geometry_optimized": True,
            "api_ui_gate_passed": False,
            "map_publication_approved": False,
            "publication_status": PUBLICATION_STATUS,
            "source_generation_id": source_run.get("generation_id"),
            "source_incident_generation_id": incident_run.get("generation_id"),
            "as_of_date": source_run.get("as_of_date"),
            "incident_count": int(counts.get("cluster_labels.parquet") or 0),
        },
        "source": {
            "generation_manifest_sha256": source_manifest_hash,
            "incident_manifest_sha256": incident_manifest_hash,
            "implementation_sha256": implementation_hash,
            "bundle_content_sha256": bundle_content_hash,
        },
        "policy": {
            "version": policy.get("version"),
            "sha256": policy.get("sha256"),
            "calibration_status": policy.get("calibration_status", "uncalibrated"),
            "warning": policy.get("warning") or warning,
        },
        "semantics": {
            "primary_story_identity": "incident_id",
            "story_cluster_id_alias": "incident_id",
            "trajectory_representation": "exact_union_of_grid_rectangles",
            "footprint_key": ["timeline_bucket", "incident_id"],
            "footprint_source": "footprint_cell_ids_json",
            "centroid_trails_used": False,
            "convex_hulls_used": False,
            "is_physical_movement": False,
            "all_story_week_footprints_exported": True,
            "low_zoom_footprints_dropped": False,
            "crop_death_inferred": False,
        },
        "validation": validation,
        "warning": warning,
        "limitations": [
            "The V3 tracker uses uncalibrated starter thresholds.",
            "Footprints show monitored exposure extent, not physical movement or causal propagation.",
            "Lifecycle states do not establish biological crop death or validated agronomic outcomes.",
            "Real-VM visual and latency acceptance remain required before operational promotion.",
        ],
        "outputs": {
            "field_geometry": "field_geometry.parquet",
            "frame_fields": "frame_fields.parquet",
            "cluster_labels": "cluster_labels.parquet",
            "event_windows": "event_windows.parquet",
            "story_day_membership": "story_day_membership.parquet",
            "incident_footprints": "incident_footprints.parquet",
            "timeline_summary": "gpu_summaries/timeline_summary.parquet",
            "geometry_profile": "geometry_profile.json",
            "drilldown": list(DRILLDOWN_COPIES),
        },
        "artifacts": artifacts,
        "manifest_self_hash_excluded": True,
    }


def _artifact_inventory(stage: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    with duckdb.connect(":memory:") as connection:
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = path.relative_to(stage).as_posix()
            entry: dict[str, Any] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix == ".parquet":
                entry["row_count"] = _row_count(connection, path)
            inventory[relative] = entry
    return inventory


def _validate_paths(
    incident_dir: Path,
    source_generation_dir: Path,
    output_dir: Path,
    temp_dir: Path | None,
) -> None:
    for label, path in (
        ("Incident V3 directory", incident_dir),
        ("Source generation directory", source_generation_dir),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable viewer output already exists: {output_dir}")
    for source in (incident_dir, source_generation_dir):
        if output_dir == source or output_dir.is_relative_to(source):
            raise ValueError("Viewer output must not be inside an immutable input")
        if source.is_relative_to(output_dir):
            raise ValueError("Viewer output must not contain an immutable input")
    if temp_dir is not None:
        resolved = temp_dir.expanduser().resolve()
        if resolved == output_dir or resolved.is_relative_to(output_dir):
            raise ValueError("DuckDB temp directory must not be inside viewer output")
        for source in (incident_dir, source_generation_dir):
            if resolved == source or resolved.is_relative_to(source):
                raise ValueError("DuckDB temp directory must not modify an immutable input")


def _validate_incident_source(
    incident_dir: Path,
    source_generation_dir: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_INCIDENT_FILES if not (incident_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Incident V3 directory is missing: " + ", ".join(missing))
    try:
        manifest = json.loads((incident_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Incident V3 manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Incident V3 manifest must be a JSON object")
    run = manifest.get("run") or {}
    if run.get("status") != "complete" or run.get("immutable") is not True:
        raise ValueError("Viewer export requires a completed immutable Incident V3 run")
    source_run = source_manifest.get("run") or {}
    linked_id = run.get("source_generation_id")
    if linked_id and linked_id != source_run.get("generation_id"):
        raise ValueError("Incident V3 run does not reference the supplied source generation")
    recorded_source_hash = (manifest.get("source") or {}).get(
        "generation_manifest_sha256"
    )
    actual_source_hash = file_sha256(source_generation_dir / "manifest.json")
    if recorded_source_hash and recorded_source_hash != actual_source_hash:
        raise ValueError("Source generation manifest hash does not match Incident V3 lineage")
    artifacts = manifest.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        raise ValueError("Incident V3 manifest artifacts must be an object")
    for name in REQUIRED_INCIDENT_FILES:
        if name == "manifest.json":
            continue
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Incident V3 manifest does not declare required artifact: {name}")
        path = incident_dir / name
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        if expected and file_sha256(path) != expected:
            raise ValueError(f"Incident V3 artifact hash mismatch: {name}")
    return manifest


def _sanitize_geometry_profile(path: Path) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["source"] = Path(str(profile.get("source") or "map_field_geometry.parquet")).name
    profile["output"] = "field_geometry.parquet"
    path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _columns(connection: duckdb.DuckDBPyConnection, view: str) -> set[str]:
    return {str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()}


def _require_columns(
    connection: duckdb.DuckDBPyConnection, view: str, required: set[str]
) -> None:
    missing = sorted(required - _columns(connection, view))
    if missing:
        raise ValueError(f"{view} is missing columns: {', '.join(missing)}")


def _optional_sql(columns: set[str], alias: str, name: str, fallback: str) -> str:
    return f'{alias}."{name}"' if name in columns else fallback


def _parquet_columns(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> set[str]:
    cursor = connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
    return {str(item[0]) for item in (cursor.description or [])}


def _row_count(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    )


def _duplicate_key_count(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    columns: tuple[str, ...],
) -> int:
    identifiers = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
    return int(
        connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {identifiers}, COUNT(*) AS row_count
                FROM read_parquet(?)
                GROUP BY {identifiers}
                HAVING row_count > 1
            )
            """,
            [str(path)],
        ).fetchone()[0]
    )


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parsed = list(value)
    else:
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("footprint_cell_ids_json must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("footprint_cell_ids_json must contain a JSON list")
    values = sorted({str(item).strip() for item in parsed if str(item).strip()})
    return values


def _optional_json_string_list(value: Any) -> list[str]:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    return _json_string_list(value)


def _dominant_stage(value: Any) -> str:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "unknown"
    if not isinstance(parsed, dict) or not parsed:
        return "unknown"
    ranked = []
    for key, amount in parsed.items():
        try:
            ranked.append((float(amount), str(key)))
        except (TypeError, ValueError):
            continue
    return max(ranked, default=(0.0, "unknown"), key=lambda item: (item[0], item[1]))[1]


__all__ = ["MODE", "SCHEMA_VERSION", "export_incident_viewer_v3"]
