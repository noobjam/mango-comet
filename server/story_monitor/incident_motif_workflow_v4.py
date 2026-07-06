"""Immutable file workflows for diagnostic Incident V4 motif learning.

The workflow consumes causally bounded V4 viewer story checkpoints plus
separate daily pressure/S2 evidence.  It never writes into its source releases
and exposes no viewer or map publication function.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import duckdb
import pandas as pd

from .incident_motifs_v4 import (
    CompletedMotifModel,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_SCHEMA_VERSION,
    PREFIX_SCHEMA_VERSION,
    MotifDiscoveryConfig,
    PrefixCalibrationConfig,
    PrefixMotifModel,
    assign_open_set_prefixes,
    build_causal_prefix_features,
    build_completed_story_features,
    build_eligibility_ledger,
    build_review_overlay_template,
    discover_completed_motifs,
    evaluate_prefix_replay,
    fit_calibrated_prefix_model,
    reviewed_incident_assignments,
    temporal_split_ledger,
)
from .incident_validation_v4 import validate_evidence_directory
from .incident_viewer_v4 import OUTPUT_FILES as VIEWER_OUTPUT_FILES
from .incident_viewer_v4 import validate_viewer_directory


WORKFLOW_SCHEMA_VERSION = "incident-motif-workflow-v4/1"
SOURCE_NAMES = {
    "membership": "incident_membership.parquet",
    "windows": "incident_windows.parquet",
    "lineage": "incident_lineage.parquet",
}
VIEWER_STORY_CHECKPOINT_NAME = VIEWER_OUTPUT_FILES["story_checkpoints"]


def materialize_causal_incident_evidence_v4(
    incident_membership_path: Path,
    field_pressure_path: Path,
    field_s2_path: Path,
    incident_daily_output_path: Path,
    incident_s2_output_path: Path,
    *,
    threads: int = 16,
    memory_limit: str | None = "8GB",
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream raw field ledgers through DuckDB into bounded incident evidence.

    The large pressure and S2 inputs are never converted to pandas.  DuckDB
    performs membership filtering, dual-clock joining, and incident/day
    aggregation before the two output parquet files are exposed to callers.
    Pair installation uses an incomplete-transaction marker plus exception
    rollback; a hard process crash remains detectable by the marker.
    """

    membership_path = incident_membership_path.expanduser().resolve()
    pressure_path = field_pressure_path.expanduser().resolve()
    s2_path = field_s2_path.expanduser().resolve()
    daily_output = incident_daily_output_path.expanduser().resolve()
    s2_output = incident_s2_output_path.expanduser().resolve()
    _require_files(
        {
            "incident_membership": membership_path,
            "field_pressure": pressure_path,
            "field_s2": s2_path,
        }
    )
    if int(threads) < 1:
        raise ValueError("DuckDB evidence adapter threads must be positive")
    if daily_output == s2_output:
        raise ValueError("incident pressure and S2 outputs must differ")
    if daily_output.parent != s2_output.parent:
        raise ValueError("incident evidence outputs must share one transaction directory")
    transaction_marker = daily_output.parent / (
        f".{daily_output.name}.{s2_output.name}.incomplete.json"
    )
    if transaction_marker.exists() or transaction_marker.is_symlink():
        raise RuntimeError(
            f"incomplete incident evidence transaction requires inspection: {transaction_marker}"
        )
    for output in (daily_output, s2_output):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"incident evidence output already exists: {output}")
        if output in {membership_path, pressure_path, s2_path}:
            raise ValueError("incident evidence output must not replace a source ledger")
    daily_output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(
        prefix=".incident-evidence-stage-v4-", dir=daily_output.parent
    ) as stage_temporary:
        stage_root = Path(stage_temporary)
        staged_daily = stage_root / daily_output.name
        staged_s2 = stage_root / s2_output.name
        if temp_dir is None:
            with TemporaryDirectory(
                prefix=".duckdb-spill-v4-", dir=daily_output.parent
            ) as spill_temporary:
                stats = _duckdb_materialize_incident_evidence(
                    membership_path,
                    pressure_path,
                    s2_path,
                    staged_daily,
                    staged_s2,
                    threads=int(threads),
                    memory_limit=memory_limit,
                    temp_dir=Path(spill_temporary),
                )
        else:
            spill = temp_dir.expanduser().resolve()
            spill.mkdir(parents=True, exist_ok=True)
            stats = _duckdb_materialize_incident_evidence(
                membership_path,
                pressure_path,
                s2_path,
                staged_daily,
                staged_s2,
                threads=int(threads),
                memory_limit=memory_limit,
                temp_dir=spill,
            )
        _write_json_exclusive(
            transaction_marker,
            {
                "status": "installing",
                "schema_version": "incident-evidence-path-adapter-v4/1",
                "outputs": [daily_output.name, s2_output.name],
            },
        )
        installed: list[Path] = []
        try:
            os.replace(staged_daily, daily_output)
            installed.append(daily_output)
            os.replace(staged_s2, s2_output)
            installed.append(s2_output)
        except BaseException:
            for path in installed:
                path.unlink(missing_ok=True)
            transaction_marker.unlink(missing_ok=True)
            raise
        transaction_marker.unlink()
    return stats


def _duckdb_materialize_incident_evidence(
    membership_path: Path,
    pressure_path: Path,
    s2_path: Path,
    daily_output: Path,
    s2_output: Path,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path,
) -> dict[str, Any]:
    connection = duckdb.connect(":memory:")
    try:
        # Never let the host/VM timezone move UTC evidence across a day or week
        # boundary (notably during DST/Ramadan transitions).
        connection.execute("SET TimeZone='UTC'")
        connection.execute("SET threads=?", [threads])
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        connection.execute("SET temp_directory=?", [str(temp_dir)])
        membership_columns = _parquet_columns(connection, membership_path)
        _require_column_names(
            membership_columns,
            (
                "incident_id",
                "field_id",
                "crop_instance_id",
                "hazard_family",
                "timeline_bucket",
                "knowledge_time",
            ),
            "incident membership",
        )
        membership_stage = _optional_column_sql(
            membership_columns, ("stage_bucket",), "'unknown'"
        )
        membership_role = _optional_column_sql(
            membership_columns, ("membership_role",), "'unknown'"
        )
        fresh_response = _optional_column_sql(
            membership_columns, ("fresh_response_evidence",), "FALSE"
        )
        membership_response = _optional_column_sql(
            membership_columns, ("response_class",), "'unknown'"
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW membership_raw_v4 AS
            SELECT
                TRIM(CAST({_q('incident_id')} AS VARCHAR)) AS incident_id,
                TRIM(CAST({_q('field_id')} AS VARCHAR)) AS field_id,
                TRIM(CAST({_q('crop_instance_id')} AS VARCHAR)) AS crop_instance_id,
                LOWER(TRIM(CAST({_q('hazard_family')} AS VARCHAR))) AS hazard_family,
                DATE_TRUNC('week', TRY_CAST({_q('timeline_bucket')} AS TIMESTAMP))
                    AS membership_week,
                TRY_CAST({_q('knowledge_time')} AS TIMESTAMP)
                    AS membership_available_at,
                LOWER(COALESCE(NULLIF(TRIM(CAST({membership_stage} AS VARCHAR)), ''),
                    'unknown')) AS stage_bucket,
                LOWER(COALESCE(NULLIF(TRIM(CAST({membership_role} AS VARCHAR)), ''),
                    'unknown')) AS membership_role,
                COALESCE(TRY_CAST({fresh_response} AS BOOLEAN), FALSE)
                    AS fresh_response_evidence,
                LOWER(COALESCE(NULLIF(TRIM(CAST({membership_response} AS VARCHAR)), ''),
                    'unknown')) AS response_class
            FROM read_parquet({_sql_string(str(membership_path))})
            """
        )
        invalid_membership = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM membership_raw_v4
                WHERE incident_id = '' OR field_id = '' OR crop_instance_id = ''
                   OR hazard_family = '' OR membership_week IS NULL
                   OR membership_available_at IS NULL
                   OR membership_available_at < membership_week
                """
            ).fetchone()[0]
        )
        if invalid_membership:
            raise ValueError(
                f"incident membership contains {invalid_membership} invalid causal rows"
            )
        conflicting_membership = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT incident_id, membership_week, field_id,
                        crop_instance_id, hazard_family
                    FROM membership_raw_v4
                    GROUP BY ALL
                    HAVING COUNT(DISTINCT membership_available_at) > 1
                )
                """
            ).fetchone()[0]
        )
        if conflicting_membership:
            raise ValueError("incident membership has conflicting knowledge times")
        connection.execute(
            """
            CREATE TEMP TABLE membership_v4 AS
            SELECT incident_id, membership_week, field_id, crop_instance_id,
                hazard_family,
                MAX(membership_available_at) AS membership_available_at,
                MIN(stage_bucket) AS stage_bucket,
                MIN(membership_role) AS membership_role,
                BOOL_OR(fresh_response_evidence) AS fresh_response_evidence,
                MIN(response_class) FILTER (WHERE fresh_response_evidence)
                    AS response_class
            FROM membership_raw_v4
            GROUP BY incident_id, membership_week, field_id, crop_instance_id,
                hazard_family
            """
        )

        pressure_columns = _parquet_columns(connection, pressure_path)
        pressure_date = _pick_column(
            pressure_columns,
            ("observation_date", "pressure_observation_date", "timeline_date", "calendar_date"),
            "field pressure observation",
        )
        pressure_available = _pick_column(
            pressure_columns,
            ("knowledge_time", "weather_available_at", "feature_available_at"),
            "field pressure knowledge",
        )
        _require_column_names(
            pressure_columns,
            ("field_id", "crop_instance_id", "hazard_family", "pressure_observed"),
            "field pressure",
        )
        pressure_score = _optional_column_sql(
            pressure_columns,
            ("pressure_score", "weather_intensity", "pressure_rank", "risk_rank", "daily_pressure_rank"),
            "NULL",
        )
        pressure_rank = _optional_column_sql(
            pressure_columns,
            ("pressure_rank", "risk_rank", "daily_pressure_rank"),
            "NULL",
        )
        pressure_active = _optional_column_sql(
            pressure_columns,
            ("pressure_active",),
            f"(TRY_CAST({pressure_rank} AS DOUBLE) >= 2)",
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW pressure_raw_v4 AS
            SELECT
                TRIM(CAST({_q('field_id')} AS VARCHAR)) AS field_id,
                TRIM(CAST({_q('crop_instance_id')} AS VARCHAR)) AS crop_instance_id,
                LOWER(TRIM(CAST({_q('hazard_family')} AS VARCHAR))) AS hazard_family,
                TRY_CAST({_q(pressure_date)} AS TIMESTAMP) AS timeline_date,
                TRY_CAST({_q(pressure_available)} AS TIMESTAMP)
                    AS pressure_available_at,
                COALESCE(TRY_CAST({_q('pressure_observed')} AS BOOLEAN), FALSE)
                    AS pressure_observed,
                TRY_CAST({pressure_score} AS DOUBLE) AS weather_intensity,
                TRY_CAST({pressure_rank} AS DOUBLE) AS pressure_rank,
                COALESCE(TRY_CAST({pressure_active} AS BOOLEAN), FALSE)
                    AS pressure_active
            FROM read_parquet({_sql_string(str(pressure_path))})
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE pressure_joined_v4 AS
            SELECT
                m.incident_id,
                p.field_id,
                p.crop_instance_id,
                p.hazard_family,
                p.timeline_date,
                p.pressure_available_at,
                GREATEST(p.pressure_available_at,
                    m.membership_available_at) AS feature_available_at,
                p.field_id || CHR(31) || p.crop_instance_id AS instance_id,
                p.pressure_observed,
                CASE WHEN p.pressure_observed
                    THEN p.weather_intensity END AS observed_intensity,
                p.pressure_observed AND p.pressure_active
                    AS weather_pressure_active,
                p.pressure_observed AND p.pressure_rank >= 4
                    AS weather_severe,
                m.membership_role IN ('impact_lag', 'unresolved', 'unresolved_review',
                    'recovering', 'recovered')
                    OR m.fresh_response_evidence AS affected,
                m.fresh_response_evidence
                    AND m.response_class = 'severe_decline' AS severe,
                m.stage_bucket
            FROM pressure_raw_v4 p
            JOIN membership_v4 m
              ON DATE_TRUNC('week', p.timeline_date) = m.membership_week
             AND p.field_id = m.field_id
             AND p.crop_instance_id = m.crop_instance_id
             AND p.hazard_family = m.hazard_family
            """
        )
        invalid_pressure = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pressure_joined_v4
                WHERE pressure_available_at IS NULL
                   OR pressure_available_at < timeline_date
                """
            ).fetchone()[0]
        )
        if invalid_pressure:
            raise ValueError(
                f"incident-owned pressure contains {invalid_pressure} invalid causal rows"
            )
        duplicate_pressure = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT incident_id, field_id, crop_instance_id,
                        hazard_family, timeline_date
                    FROM pressure_joined_v4 GROUP BY ALL HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if duplicate_pressure:
            raise ValueError("incident-owned pressure contains duplicate natural keys")
        connection.execute(
            f"""
            COPY (
                WITH daily AS (
                    SELECT incident_id, timeline_date,
                        MAX(feature_available_at) AS feature_available_at,
                        BOOL_OR(pressure_observed) AS pressure_observed,
                        AVG(observed_intensity) AS weather_intensity,
                        BOOL_OR(weather_pressure_active) AS pressure_active,
                        BOOL_OR(weather_severe) AS severe_pressure,
                        COUNT(DISTINCT instance_id)::BIGINT AS monitored_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE pressure_observed)::BIGINT
                            AS evaluable_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE affected)::BIGINT
                            AS affected_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE severe)::BIGINT
                            AS severe_count
                    FROM pressure_joined_v4 GROUP BY incident_id, timeline_date
                ), stage_counts AS (
                    SELECT incident_id, timeline_date, stage_bucket, COUNT(*) AS n
                    FROM pressure_joined_v4
                    GROUP BY incident_id, timeline_date, stage_bucket
                ), dominant_stage AS (
                    SELECT incident_id, timeline_date, stage_bucket
                    FROM stage_counts
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY incident_id, timeline_date
                        ORDER BY n DESC, stage_bucket ASC
                    ) = 1
                )
                SELECT d.incident_id, d.timeline_date, d.feature_available_at,
                    d.pressure_observed, d.weather_intensity,
                    d.monitored_count, d.evaluable_count,
                    d.affected_count, d.severe_count,
                    d.pressure_active, d.severe_pressure,
                    NULL::DOUBLE AS footprint_area_km2,
                    TRUE AS footprint_carried_forward,
                    COALESCE(s.stage_bucket, 'unknown') AS stage_bucket
                FROM daily d
                LEFT JOIN dominant_stage s
                  ON d.incident_id = s.incident_id
                 AND d.timeline_date = s.timeline_date
                ORDER BY d.incident_id, d.timeline_date
            ) TO {_sql_string(str(daily_output))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        daily_rows = _parquet_row_count(connection, daily_output)
        if daily_rows < 1:
            raise ValueError("field pressure did not join to any V3 incident membership")

        s2_columns = _parquet_columns(connection, s2_path)
        s2_source = _pick_column(
            s2_columns,
            ("spectral_source_date", "acquisition_date", "source_date"),
            "field S2 source date",
        )
        s2_available = _pick_column(
            s2_columns,
            ("knowledge_time", "spectral_available_at", "feature_available_at", "known_date"),
            "field S2 knowledge",
        )
        _require_column_names(
            s2_columns,
            ("field_id", "crop_instance_id", "acquisition_attempted", "spectral_usable"),
            "field S2",
        )
        acquisition_id = _optional_column_sql(
            s2_columns, ("acquisition_id",), "NULL"
        )
        response_class = _optional_column_sql(
            s2_columns, ("response_class", "daily_response_class"), "NULL"
        )
        new_response = _optional_column_sql(
            s2_columns, ("new_response_evidence",), "FALSE"
        )
        echo_days = _optional_column_sql(
            s2_columns, ("spectral_echo_days", "evidence_age_days"), "NULL"
        )
        acquisition_status = _optional_column_sql(
            s2_columns, ("acquisition_status",), "NULL"
        )
        ndvi_delta = _optional_column_sql(s2_columns, ("ndvi_delta",), "NULL")
        ndmi_delta = _optional_column_sql(s2_columns, ("ndmi_delta",), "NULL")
        psri_delta = _optional_column_sql(s2_columns, ("psri_delta",), "NULL")
        connection.execute(
            f"""
            CREATE TEMP VIEW s2_raw_v4 AS
            SELECT
                TRIM(CAST({_q('field_id')} AS VARCHAR)) AS field_id,
                TRIM(CAST({_q('crop_instance_id')} AS VARCHAR)) AS crop_instance_id,
                CAST({acquisition_id} AS VARCHAR) AS acquisition_id,
                TRY_CAST({_q(s2_source)} AS TIMESTAMP) AS spectral_source_date,
                TRY_CAST({_q(s2_available)} AS TIMESTAMP) AS s2_available_at,
                COALESCE(TRY_CAST({_q('acquisition_attempted')} AS BOOLEAN), FALSE)
                    AS acquisition_attempted,
                COALESCE(TRY_CAST({_q('spectral_usable')} AS BOOLEAN), FALSE)
                    AS spectral_usable,
                CAST({acquisition_status} AS VARCHAR) AS acquisition_status,
                LOWER(CAST({response_class} AS VARCHAR)) AS response_class,
                COALESCE(TRY_CAST({new_response} AS BOOLEAN), FALSE)
                    AS new_response_evidence,
                TRY_CAST({echo_days} AS DOUBLE) AS spectral_echo_days,
                TRY_CAST({ndvi_delta} AS DOUBLE) AS ndvi_delta,
                TRY_CAST({ndmi_delta} AS DOUBLE) AS ndmi_delta,
                TRY_CAST({psri_delta} AS DOUBLE) AS psri_delta
            FROM read_parquet({_sql_string(str(s2_path))})
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE s2_joined_v4 AS
            WITH ownership AS (
                SELECT incident_id, membership_week, field_id, crop_instance_id,
                    MAX(membership_available_at) AS membership_available_at
                FROM membership_v4
                GROUP BY incident_id, membership_week, field_id, crop_instance_id
            )
            SELECT
                m.incident_id,
                s.field_id,
                s.crop_instance_id,
                s.acquisition_id,
                s.spectral_source_date,
                s.s2_available_at,
                GREATEST(s.s2_available_at, m.membership_available_at)
                    AS feature_available_at,
                s.acquisition_attempted,
                s.spectral_usable,
                s.acquisition_status,
                s.response_class,
                s.new_response_evidence,
                s.spectral_echo_days,
                s.ndvi_delta,
                s.ndmi_delta,
                s.psri_delta
            FROM s2_raw_v4 s
            JOIN ownership m
              ON DATE_TRUNC('week', s.spectral_source_date) = m.membership_week
             AND s.field_id = m.field_id
             AND s.crop_instance_id = m.crop_instance_id
            """
        )
        invalid_s2 = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM s2_joined_v4
                WHERE s2_available_at IS NULL
                   OR s2_available_at < spectral_source_date
                   OR (spectral_usable AND NOT acquisition_attempted)
                """
            ).fetchone()[0]
        )
        if invalid_s2:
            raise ValueError(
                f"incident-owned S2 contains {invalid_s2} invalid causal rows"
            )
        duplicate_s2 = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT incident_id, field_id, crop_instance_id,
                        spectral_source_date, COALESCE(acquisition_id, '')
                            AS acquisition_key
                    FROM s2_joined_v4 GROUP BY ALL HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if duplicate_s2:
            raise ValueError("incident-owned S2 contains duplicate acquisition keys")
        connection.execute(
            f"""
            COPY (
                SELECT incident_id, field_id, crop_instance_id, acquisition_id,
                    spectral_source_date, feature_available_at,
                    acquisition_attempted, spectral_usable, acquisition_status,
                    response_class, new_response_evidence, spectral_echo_days,
                    ndvi_delta, ndmi_delta, psri_delta
                FROM s2_joined_v4
                ORDER BY feature_available_at, incident_id, field_id,
                    crop_instance_id, spectral_source_date
            ) TO {_sql_string(str(s2_output))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        s2_rows = _parquet_row_count(connection, s2_output)
        augmented_daily = daily_output.parent / (
            f".{daily_output.name}.with-s2-impact.parquet"
        )
        connection.execute(
            f"""
            COPY (
                WITH s2_impact AS (
                    SELECT incident_id,
                        CAST(spectral_source_date AS DATE) AS source_day,
                        field_id || CHR(31) || crop_instance_id AS instance_id,
                        BOOL_OR(
                            acquisition_attempted AND spectral_usable
                            AND new_response_evidence AND response_class IN
                                ('medium_decline', 'severe_decline', 'recovery')
                        ) AS impact_response,
                        BOOL_OR(
                            acquisition_attempted AND spectral_usable
                            AND new_response_evidence
                            AND response_class = 'severe_decline'
                        ) AS severe_response,
                        MAX(feature_available_at) FILTER (
                            WHERE acquisition_attempted AND spectral_usable
                              AND new_response_evidence
                        ) AS response_available_at
                    FROM s2_joined_v4
                    GROUP BY incident_id, source_day, instance_id
                ), enriched AS (
                    SELECT p.* EXCLUDE (affected, severe, feature_available_at),
                        GREATEST(
                            p.feature_available_at,
                            COALESCE(s.response_available_at, p.feature_available_at)
                        ) AS feature_available_at,
                        p.affected OR COALESCE(s.impact_response, FALSE) AS affected,
                        p.severe OR COALESCE(s.severe_response, FALSE) AS severe
                    FROM pressure_joined_v4 p
                    LEFT JOIN s2_impact s
                      ON p.incident_id = s.incident_id
                     AND CAST(p.timeline_date AS DATE) = s.source_day
                     AND p.instance_id = s.instance_id
                ), daily AS (
                    SELECT incident_id, timeline_date,
                        MAX(feature_available_at) AS feature_available_at,
                        BOOL_OR(pressure_observed) AS pressure_observed,
                        AVG(observed_intensity) AS weather_intensity,
                        BOOL_OR(weather_pressure_active) AS pressure_active,
                        BOOL_OR(weather_severe) AS severe_pressure,
                        COUNT(DISTINCT instance_id)::BIGINT AS monitored_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE pressure_observed)::BIGINT
                            AS evaluable_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE affected)::BIGINT
                            AS affected_count,
                        COUNT(DISTINCT instance_id) FILTER (WHERE severe)::BIGINT
                            AS severe_count
                    FROM enriched GROUP BY incident_id, timeline_date
                ), stage_counts AS (
                    SELECT incident_id, timeline_date, stage_bucket, COUNT(*) AS n
                    FROM enriched GROUP BY incident_id, timeline_date, stage_bucket
                ), dominant_stage AS (
                    SELECT incident_id, timeline_date, stage_bucket
                    FROM stage_counts
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY incident_id, timeline_date
                        ORDER BY n DESC, stage_bucket ASC
                    ) = 1
                )
                SELECT d.incident_id, d.timeline_date, d.feature_available_at,
                    d.pressure_observed, d.weather_intensity,
                    d.monitored_count, d.evaluable_count,
                    d.affected_count, d.severe_count,
                    d.pressure_active, d.severe_pressure,
                    NULL::DOUBLE AS footprint_area_km2,
                    TRUE AS footprint_carried_forward,
                    COALESCE(s.stage_bucket, 'unknown') AS stage_bucket
                FROM daily d
                LEFT JOIN dominant_stage s
                  ON d.incident_id = s.incident_id
                 AND d.timeline_date = s.timeline_date
                ORDER BY d.incident_id, d.timeline_date
            ) TO {_sql_string(str(augmented_daily))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        os.replace(augmented_daily, daily_output)
        daily_rows = _parquet_row_count(connection, daily_output)
        return {
            "engine": "duckdb",
            "schema_version": "incident-evidence-path-adapter-v4/1",
            "raw_field_ledgers_loaded_into_pandas": False,
            "raw_scan_strategy": "single_membership_join_scan_per_field_ledger",
            "pandas_materialization_boundary": "incident_aggregates_and_owned_acquisitions",
            "threads": threads,
            "memory_limit": str(memory_limit) if memory_limit else None,
            "spill_enabled": True,
            "incident_daily_row_count": daily_rows,
            "incident_s2_row_count": s2_rows,
            "pressure_materialization_grain": "incident_id_x_timeline_date",
            "s2_materialization_grain": "incident_owned_distinct_acquisition",
            "map_publication_supported": False,
        }
    finally:
        connection.close()


def _maturity_case_sql(column: str, horizons: tuple[int, ...]) -> str:
    branches = " ".join(
        f"WHEN {column} >= {int(value)} THEN {int(value)}"
        for value in reversed(horizons)
    )
    return f"CASE {branches} ELSE 0 END"


def _materialize_learning_features_v4(
    story_checkpoints_path: Path,
    daily_path: Path,
    s2_path: Path,
    eligibility_path: Path,
    completed_output: Path,
    prefix_output: Path,
    *,
    prefix_config: PrefixCalibrationConfig,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> None:
    """Build causal terminal/prefix vectors with bounded DuckDB state.

    This replaces the former per-incident pandas rescans.  Evidence is reduced
    once into cumulative modality snapshots and attached to unioned knowledge
    cutoffs with ASOF joins.  The potentially tens-of-millions of prefixes are
    written directly to parquet.
    """

    prefix_config.validate()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.execute("SET threads=?", [int(threads)])
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir is not None:
            spill = temp_dir.expanduser().resolve()
            spill.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(spill)])
        story_checkpoints = _sql_string(str(story_checkpoints_path))
        daily = _sql_string(str(daily_path))
        s2 = _sql_string(str(s2_path))
        eligibility = _sql_string(str(eligibility_path))
        prefix_target = _sql_string(str(prefix_output))
        completed_target = _sql_string(str(completed_output))
        weekly_columns = _parquet_columns(connection, story_checkpoints_path)
        if "story_known_time" not in weekly_columns:
            raise ValueError(
                "V4 story checkpoints require the full causal story_known_time"
            )
        if not {"footprint_area_km2", "footprint_carried_forward"} <= weekly_columns:
            raise ValueError(
                "weekly state requires exact footprint_area_km2 and "
                "footprint_carried_forward audit columns"
            )
        weekly_state_name = next(
            (name for name in ("current_state", "incident_state") if name in weekly_columns),
            None,
        )
        weekly_state = f"w.{_q(weekly_state_name)}" if weekly_state_name else "''"
        weekly_area = f"TRY_CAST(w.{_q('footprint_area_km2')} AS DOUBLE)"
        weekly_carried = (
            f"COALESCE(TRY_CAST(w.{_q('footprint_carried_forward')} AS BOOLEAN), TRUE)"
        )
        weather_maturity = _maturity_case_sql(
            "weather_observed_day_count", prefix_config.weather_day_horizons
        )
        s2_maturity = _maturity_case_sql(
            "s2_usable_acquisition_count", prefix_config.s2_acquisition_horizons
        )
        feature_select = """
            CAST(DATE_DIFF('day', first_timeline_date, last_timeline_date) + 1 AS DOUBLE)
                AS duration_days,
            CAST(COALESCE(checkpoint_count, 0) AS DOUBLE) AS checkpoint_count,
            CAST(COALESCE(weather_observed_count, 0) AS DOUBLE)
                AS weather_observed_day_count,
            CAST(COALESCE(weather_observed_count, 0) /
                GREATEST(DATE_DIFF('day', first_timeline_date, last_timeline_date) + 1, 1)
                AS DOUBLE) AS weather_coverage_fraction,
            CAST(COALESCE(pressure_active_count, 0) /
                GREATEST(daily_count, 1) AS DOUBLE) AS pressure_day_fraction,
            CAST(COALESCE(severe_pressure_count, 0) /
                GREATEST(daily_count, 1) AS DOUBLE) AS severe_pressure_day_fraction,
            CAST(COALESCE(intensity_sum / NULLIF(intensity_count, 0), 0) AS DOUBLE)
                AS weather_intensity_mean,
            CAST(COALESCE(intensity_peak, 0) AS DOUBLE) AS weather_intensity_peak,
            CAST(COALESCE(
                (intensity_count * intensity_xy_sum - intensity_x_sum * intensity_sum)
                / NULLIF(intensity_count * intensity_x2_sum
                    - intensity_x_sum * intensity_x_sum, 0), 0) AS DOUBLE)
                AS weather_intensity_slope,
            CAST(COALESCE(positive_intensity_sum, 0) AS DOUBLE)
                AS weather_cumulative_intensity,
            CAST((daily_count - COALESCE(intensity_count, 0)) /
                GREATEST(daily_count, 1) AS DOUBLE)
                AS weather_intensity_missing_fraction,
            CAST(COALESCE(affected_rate_sum / GREATEST(daily_count, 1), 0) AS DOUBLE)
                AS affected_rate_mean,
            CAST(COALESCE(affected_rate_peak, 0) AS DOUBLE) AS affected_rate_peak,
            CAST(COALESCE(severe_sum / NULLIF(affected_sum, 0), 0) AS DOUBLE)
                AS severe_affected_fraction,
            CAST(COALESCE(maximum_observed_area_km2, 0) AS DOUBLE)
                AS maximum_observed_area_km2,
            CAST(COALESCE(observed_footprint_count, 0) /
                GREATEST(footprint_observation_count, 1) AS DOUBLE)
                AS observed_footprint_fraction,
            CAST(COALESCE(data_gap_sum / NULLIF(monitored_sum, 0), 0) AS DOUBLE)
                AS data_gap_fraction,
            CAST(COALESCE(relapse_count, 0) AS DOUBLE) AS relapse_count,
            CAST(COALESCE(s2_opportunity_count, 0) AS DOUBLE)
                AS s2_usable_acquisition_count,
            CAST(COALESCE(s2_instance_count, 0) /
                GREATEST(max_monitored_count, 1) AS DOUBLE)
                AS s2_crop_instance_coverage_fraction,
            CAST(COALESCE(cutoff_epoch_day - s2_source_epoch_mean, 0) AS DOUBLE)
                AS s2_echo_age_mean,
            CAST(COALESCE(cutoff_epoch_day - s2_source_epoch_min, 0) AS DOUBLE)
                AS s2_echo_age_max,
            CAST(COALESCE(s2_decline_count / NULLIF(s2_row_count, 0), 0) AS DOUBLE)
                AS s2_decline_fraction,
            CAST(COALESCE(s2_recovery_count / NULLIF(s2_row_count, 0), 0) AS DOUBLE)
                AS s2_recovery_fraction,
            CAST(COALESCE(ndvi_sum / NULLIF(ndvi_count, 0), 0) AS DOUBLE)
                AS s2_ndvi_delta_mean,
            CAST(COALESCE(ndvi_min, 0) AS DOUBLE) AS s2_ndvi_delta_min,
            CAST(COALESCE(ndmi_sum / NULLIF(ndmi_count, 0), 0) AS DOUBLE)
                AS s2_ndmi_delta_mean,
            CAST(COALESCE(ndmi_min, 0) AS DOUBLE) AS s2_ndmi_delta_min,
            CAST(COALESCE(psri_sum / NULLIF(psri_count, 0), 0) AS DOUBLE)
                AS s2_psri_delta_mean,
            CAST(COALESCE(psri_max, 0) AS DOUBLE) AS s2_psri_delta_max
        """
        query = f"""
        CREATE TEMP VIEW eligible_v4 AS
        SELECT CAST(incident_id AS VARCHAR) AS incident_id,
            CAST(exposure_id AS VARCHAR) AS exposure_id,
            LOWER(CAST(crop_name AS VARCHAR)) AS crop_name,
            LOWER(CAST(hazard_family AS VARCHAR)) AS hazard_family,
            CAST(lineage_family_id AS VARCHAR) AS lineage_family_id,
            CAST(first_available_at AS TIMESTAMP) AS first_available_at,
            CAST(feature_available_at AS TIMESTAMP) AS terminal_available_at
        FROM read_parquet({eligibility}) WHERE COALESCE(eligible, FALSE);

        CREATE TEMP VIEW daily_rows_v4 AS
        SELECT d.*, DATE_DIFF('day', DATE '1970-01-01', CAST(timeline_date AS DATE))::DOUBLE AS x,
            affected_count::DOUBLE / NULLIF(monitored_count, 0) AS affected_rate,
            GREATEST(monitored_count - evaluable_count, 0)::DOUBLE AS data_gap,
            NOT COALESCE(footprint_carried_forward, TRUE)
                AND footprint_area_km2 IS NOT NULL AS observed_footprint
        FROM read_parquet({daily}) d JOIN eligible_v4 USING (incident_id);

        CREATE TEMP TABLE daily_running_v4 AS
        WITH events AS (
            SELECT incident_id, CAST(feature_available_at AS TIMESTAMP) AS available_at,
                MIN(CAST(timeline_date AS DATE)) AS first_date,
                MAX(CAST(timeline_date AS DATE)) AS last_date,
                COUNT(*)::DOUBLE AS n,
                COUNT_IF(pressure_observed)::DOUBLE AS observed_n,
                COUNT_IF(pressure_active)::DOUBLE AS active_n,
                COUNT_IF(severe_pressure)::DOUBLE AS severe_pressure_n,
                COUNT(weather_intensity)::DOUBLE AS intensity_n,
                SUM(weather_intensity) AS intensity_sum,
                MAX(weather_intensity) AS intensity_peak,
                SUM(GREATEST(weather_intensity, 0)) AS positive_intensity_sum,
                SUM(x) FILTER (WHERE weather_intensity IS NOT NULL) AS intensity_x_sum,
                SUM(x*x) FILTER (WHERE weather_intensity IS NOT NULL) AS intensity_x2_sum,
                SUM(x*weather_intensity) AS intensity_xy_sum,
                SUM(COALESCE(affected_rate, 0)) AS affected_rate_sum,
                MAX(COALESCE(affected_rate, 0)) AS affected_rate_peak,
                SUM(affected_count)::DOUBLE AS affected_sum,
                SUM(severe_count)::DOUBLE AS severe_sum,
                MAX(footprint_area_km2) FILTER (WHERE observed_footprint)
                    AS observed_area_max,
                COUNT_IF(observed_footprint)::DOUBLE AS observed_footprint_n,
                SUM(data_gap) AS data_gap_sum,
                SUM(monitored_count)::DOUBLE AS monitored_sum,
                MAX(monitored_count)::DOUBLE AS max_monitored,
                ARG_MAX(COALESCE(stage_bucket, 'unknown'), timeline_date) AS latest_stage
            FROM daily_rows_v4 GROUP BY incident_id, available_at
        )
        SELECT incident_id, available_at,
            MIN(first_date) OVER cumulative AS first_timeline_date,
            MAX(last_date) OVER cumulative AS last_timeline_date,
            SUM(n) OVER cumulative AS daily_count,
            SUM(observed_n) OVER cumulative AS weather_observed_count,
            SUM(active_n) OVER cumulative AS pressure_active_count,
            SUM(severe_pressure_n) OVER cumulative AS severe_pressure_count,
            SUM(intensity_n) OVER cumulative AS intensity_count,
            SUM(intensity_sum) OVER cumulative AS intensity_sum,
            MAX(intensity_peak) OVER cumulative AS intensity_peak,
            SUM(positive_intensity_sum) OVER cumulative AS positive_intensity_sum,
            SUM(intensity_x_sum) OVER cumulative AS intensity_x_sum,
            SUM(intensity_x2_sum) OVER cumulative AS intensity_x2_sum,
            SUM(intensity_xy_sum) OVER cumulative AS intensity_xy_sum,
            SUM(affected_rate_sum) OVER cumulative AS affected_rate_sum,
            MAX(affected_rate_peak) OVER cumulative AS affected_rate_peak,
            SUM(affected_sum) OVER cumulative AS affected_sum,
            SUM(severe_sum) OVER cumulative AS severe_sum,
            MAX(observed_area_max) OVER cumulative AS daily_maximum_observed_area_km2,
            SUM(observed_footprint_n) OVER cumulative AS daily_observed_footprint_count,
            SUM(data_gap_sum) OVER cumulative AS data_gap_sum,
            SUM(monitored_sum) OVER cumulative AS monitored_sum,
            MAX(max_monitored) OVER cumulative AS max_monitored_count,
            LAST_VALUE(latest_stage) OVER cumulative AS dominant_stage
        FROM events WINDOW cumulative AS (PARTITION BY incident_id ORDER BY available_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

        CREATE TEMP TABLE weekly_running_v4 AS
        WITH ordered AS (
            SELECT CAST(w.incident_id AS VARCHAR) AS incident_id,
                CAST(w.story_known_time AS TIMESTAMP) AS available_at,
                UPPER(COALESCE(CAST({weekly_state} AS VARCHAR), '')) AS state,
                {weekly_area} AS footprint_area_km2,
                NOT ({weekly_carried}) AND {weekly_area} IS NOT NULL
                    AS observed_footprint,
                LAG(UPPER(COALESCE(CAST({weekly_state} AS VARCHAR), ''))) OVER
                    (PARTITION BY w.incident_id ORDER BY w.story_known_time) AS prior_state
            FROM read_parquet({story_checkpoints}) w JOIN eligible_v4 e
              ON CAST(w.incident_id AS VARCHAR) = e.incident_id
        ), events AS (
            SELECT incident_id, available_at, COUNT(*)::DOUBLE AS checkpoint_n,
                COUNT_IF(state = 'RELAPSED' AND COALESCE(prior_state, '') <> 'RELAPSED')::DOUBLE
                    AS relapse_n,
                MAX(footprint_area_km2) FILTER (WHERE observed_footprint)
                    AS observed_area_max,
                COUNT_IF(observed_footprint)::DOUBLE AS observed_footprint_n
            FROM ordered GROUP BY incident_id, available_at
        )
        SELECT incident_id, available_at,
            SUM(checkpoint_n) OVER weekly_cumulative AS checkpoint_count,
            SUM(relapse_n) OVER weekly_cumulative AS relapse_count,
            MAX(observed_area_max) OVER weekly_cumulative
                AS weekly_maximum_observed_area_km2,
            SUM(observed_footprint_n) OVER weekly_cumulative
                AS weekly_observed_footprint_count
        FROM events WINDOW weekly_cumulative AS (PARTITION BY incident_id ORDER BY available_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

        CREATE TEMP VIEW usable_s2_v4 AS
        SELECT s.*, DATE_DIFF('day', DATE '1970-01-01',
            CAST(spectral_source_date AS DATE))::DOUBLE AS source_epoch_day
        FROM read_parquet({s2}) s JOIN eligible_v4 USING (incident_id)
        WHERE COALESCE(acquisition_attempted, FALSE)
          AND COALESCE(spectral_usable, FALSE)
          AND spectral_source_date IS NOT NULL;

        CREATE TEMP TABLE s2_running_v4 AS
        WITH events AS (
            SELECT incident_id, CAST(feature_available_at AS TIMESTAMP) AS available_at,
                COUNT(*)::DOUBLE AS row_n,
                SUM(source_epoch_day) AS source_epoch_sum,
                MIN(source_epoch_day) AS source_epoch_min,
                COUNT_IF(response_class IN ('medium_decline','severe_decline'))::DOUBLE
                    AS decline_n,
                COUNT_IF(response_class = 'recovery')::DOUBLE AS recovery_n,
                COUNT(ndvi_delta)::DOUBLE AS ndvi_n, SUM(ndvi_delta) AS ndvi_sum,
                MIN(ndvi_delta) AS ndvi_min,
                COUNT(ndmi_delta)::DOUBLE AS ndmi_n, SUM(ndmi_delta) AS ndmi_sum,
                MIN(ndmi_delta) AS ndmi_min,
                COUNT(psri_delta)::DOUBLE AS psri_n, SUM(psri_delta) AS psri_sum,
                MAX(psri_delta) AS psri_max
            FROM usable_s2_v4 GROUP BY incident_id, available_at
        )
        SELECT incident_id, available_at,
            SUM(row_n) OVER s2_cumulative AS s2_row_count,
            SUM(source_epoch_sum) OVER s2_cumulative / NULLIF(SUM(row_n) OVER s2_cumulative, 0)
                AS s2_source_epoch_mean,
            MIN(source_epoch_min) OVER s2_cumulative AS s2_source_epoch_min,
            SUM(decline_n) OVER s2_cumulative AS s2_decline_count,
            SUM(recovery_n) OVER s2_cumulative AS s2_recovery_count,
            SUM(ndvi_n) OVER s2_cumulative AS ndvi_count, SUM(ndvi_sum) OVER s2_cumulative AS ndvi_sum,
            MIN(ndvi_min) OVER s2_cumulative AS ndvi_min,
            SUM(ndmi_n) OVER s2_cumulative AS ndmi_count, SUM(ndmi_sum) OVER s2_cumulative AS ndmi_sum,
            MIN(ndmi_min) OVER s2_cumulative AS ndmi_min,
            SUM(psri_n) OVER s2_cumulative AS psri_count, SUM(psri_sum) OVER s2_cumulative AS psri_sum,
            MAX(psri_max) OVER s2_cumulative AS psri_max
        FROM events WINDOW s2_cumulative AS (PARTITION BY incident_id ORDER BY available_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

        CREATE TEMP TABLE s2_opportunity_running_v4 AS
        WITH first_known AS (
            SELECT incident_id, CAST(spectral_source_date AS DATE) AS source_date,
                MIN(CAST(feature_available_at AS TIMESTAMP)) AS available_at
            FROM usable_s2_v4 GROUP BY incident_id, source_date
        ), running AS (
        SELECT incident_id, available_at,
            COUNT(*) OVER (PARTITION BY incident_id ORDER BY available_at, source_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::DOUBLE
                AS s2_opportunity_count
        FROM first_known)
        SELECT incident_id, available_at, MAX(s2_opportunity_count)
            AS s2_opportunity_count
        FROM running GROUP BY incident_id, available_at;

        CREATE TEMP TABLE s2_instance_running_v4 AS
        WITH first_known AS (
            SELECT incident_id, CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
                MIN(CAST(feature_available_at AS TIMESTAMP)) AS available_at
            FROM usable_s2_v4 GROUP BY incident_id, crop_instance_id
        ), running AS (
        SELECT incident_id, available_at,
            COUNT(*) OVER (PARTITION BY incident_id ORDER BY available_at, crop_instance_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::DOUBLE
                AS s2_instance_count
        FROM first_known)
        SELECT incident_id, available_at, MAX(s2_instance_count) AS s2_instance_count
        FROM running GROUP BY incident_id, available_at;

        CREATE TEMP TABLE cutoffs_v4 AS
        SELECT incident_id, cutoff FROM (
            SELECT incident_id, CAST(feature_available_at AS TIMESTAMP) AS cutoff
                FROM daily_rows_v4
            UNION SELECT e.incident_id, CAST(w.story_known_time AS TIMESTAMP)
                FROM read_parquet({story_checkpoints}) w JOIN eligible_v4 e
                  ON CAST(w.incident_id AS VARCHAR) = e.incident_id
            UNION SELECT incident_id, CAST(feature_available_at AS TIMESTAMP)
                FROM read_parquet({s2}) s JOIN eligible_v4 USING (incident_id)
            UNION SELECT incident_id, terminal_available_at FROM eligible_v4
        ) c JOIN eligible_v4 e USING (incident_id)
        WHERE cutoff BETWEEN e.first_available_at AND e.terminal_available_at;

        CREATE TEMP TABLE feature_base_v4 AS
        SELECT c.incident_id, e.exposure_id, e.crop_name, e.hazard_family,
            e.lineage_family_id, c.cutoff,
            DATE_DIFF('day', DATE '1970-01-01', CAST(c.cutoff AS DATE))::DOUBLE
                AS cutoff_epoch_day,
            d.* EXCLUDE (incident_id, available_at),
            w.* EXCLUDE (incident_id, available_at),
            s.* EXCLUDE (incident_id, available_at),
            o.s2_opportunity_count, i.s2_instance_count,
            COALESCE(w.weekly_maximum_observed_area_km2,
                d.daily_maximum_observed_area_km2) AS maximum_observed_area_km2,
            CASE WHEN COALESCE(w.checkpoint_count, 0) > 0
                THEN w.weekly_observed_footprint_count
                ELSE d.daily_observed_footprint_count END AS observed_footprint_count,
            CASE WHEN COALESCE(w.checkpoint_count, 0) > 0
                THEN w.checkpoint_count ELSE d.daily_count END
                AS footprint_observation_count
        FROM cutoffs_v4 c JOIN eligible_v4 e USING (incident_id)
        ASOF LEFT JOIN daily_running_v4 d
          ON c.incident_id = d.incident_id AND c.cutoff >= d.available_at
        ASOF LEFT JOIN weekly_running_v4 w
          ON c.incident_id = w.incident_id AND c.cutoff >= w.available_at
        ASOF LEFT JOIN s2_running_v4 s
          ON c.incident_id = s.incident_id AND c.cutoff >= s.available_at
        ASOF LEFT JOIN s2_opportunity_running_v4 o
          ON c.incident_id = o.incident_id AND c.cutoff >= o.available_at
        ASOF LEFT JOIN s2_instance_running_v4 i
          ON c.incident_id = i.incident_id AND c.cutoff >= i.available_at
        WHERE d.first_timeline_date IS NOT NULL;

        COPY (
            SELECT '{PREFIX_SCHEMA_VERSION}' AS feature_schema_version,
                incident_id, exposure_id, crop_name, hazard_family,
                lineage_family_id, cutoff AS feature_available_at,
                cutoff AS prefix_as_of_time,
                {feature_select},
                {weather_maturity}::BIGINT AS weather_day_horizon,
                {s2_maturity}::BIGINT AS s2_acquisition_horizon,
                COALESCE(dominant_stage, 'unknown') AS dominant_stage,
                NULL::DOUBLE AS stage_entropy,
                CAST(JSON_OBJECT('audit_status', 'latest_stage_only',
                    'latest_stage', COALESCE(dominant_stage, 'unknown')) AS VARCHAR)
                    AS stage_distribution_json
            FROM feature_base_v4 ORDER BY incident_id, cutoff
        ) TO {prefix_target} (FORMAT PARQUET, COMPRESSION ZSTD);

        COPY (
            SELECT '{FEATURE_SCHEMA_VERSION}' AS feature_schema_version,
                f.incident_id, f.exposure_id, f.crop_name, f.hazard_family,
                f.lineage_family_id, f.cutoff AS feature_available_at,
                {feature_select},
                COALESCE(f.dominant_stage, 'unknown') AS dominant_stage,
                NULL::DOUBLE AS stage_entropy,
                CAST(JSON_OBJECT('audit_status', 'latest_stage_only',
                    'latest_stage', COALESCE(f.dominant_stage, 'unknown')) AS VARCHAR)
                    AS stage_distribution_json
            FROM feature_base_v4 f JOIN eligible_v4 e USING (incident_id)
            WHERE f.cutoff = e.terminal_available_at
            ORDER BY f.incident_id
        ) TO {completed_target} (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        connection.execute(query)
    finally:
        connection.close()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str,
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _discover_completed_from_parquet_v4(
    completed_path: Path,
    split_path: Path,
    stage: Path,
    *,
    training_through: Any,
    config: MotifDiscoveryConfig,
    provenance: Mapping[str, Any],
) -> CompletedMotifModel:
    """Cluster one crop×hazard cohort at a time and spool assignments to disk."""

    config.validate()
    connection = duckdb.connect(":memory:")
    parts = stage / ".assignment_parts"
    parts.mkdir()
    schemas: list[dict[str, Any]] = []
    prototypes: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []
    training_count = 0
    noise_count = 0
    try:
        cohorts = connection.execute(
            """
            SELECT c.crop_name, c.hazard_family, COUNT(*) n
            FROM read_parquet(?) c JOIN read_parquet(?) s USING (incident_id)
            WHERE s.eligible AND s.temporal_split = 'train'
            GROUP BY 1, 2 ORDER BY 1, 2
            """,
            [str(completed_path), str(split_path)],
        ).fetchall()
        part_index = 0

        def write_unsupported(crop: str, hazard: str, count: int) -> None:
            nonlocal part_index, training_count, noise_count
            target = parts / f"part-{part_index:05d}.parquet"
            connection.execute(
                f"""
                COPY (
                    SELECT c.incident_id, c.crop_name, c.hazard_family,
                        NULL::VARCHAR AS discovered_motif_id,
                        -1::BIGINT AS discovery_label,
                        0.0::DOUBLE AS training_membership,
                        FALSE AS accepted
                    FROM read_parquet({_sql_string(str(completed_path))}) c
                    JOIN read_parquet({_sql_string(str(split_path))}) s USING (incident_id)
                    WHERE s.eligible AND s.temporal_split = 'train'
                      AND c.crop_name = {_sql_string(str(crop))}
                      AND c.hazard_family = {_sql_string(str(hazard))}
                    ORDER BY c.incident_id
                ) TO {_sql_string(str(target))} (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            training_count += int(count)
            noise_count += int(count)
            part_index += 1

        for crop, hazard, count in cohorts:
            if int(count) < config.min_cluster_size:
                write_unsupported(str(crop), str(hazard), int(count))
                continue
            frame = connection.execute(
                """
                SELECT c.* FROM read_parquet(?) c
                JOIN read_parquet(?) s USING (incident_id)
                WHERE s.eligible AND s.temporal_split = 'train'
                  AND c.crop_name = ? AND c.hazard_family = ?
                ORDER BY c.incident_id
                """,
                [str(completed_path), str(split_path), crop, hazard],
            ).fetchdf()
            try:
                local = discover_completed_motifs(
                    frame, training_through=training_through, config=config,
                    provenance=provenance,
                )
            except ValueError as exc:
                if "no supported completed motifs" in str(exc):
                    write_unsupported(str(crop), str(hazard), int(count))
                    continue
                raise
            schemas.extend(local.feature_schema["strata"])
            prototypes.append(local.prototypes.drop(columns=["model_version"]))
            catalogs.append(local.catalog.drop(columns=["model_version"]))
            assignments = local.assignments.drop(columns=["model_version"])
            training_count += len(assignments)
            noise_count += int((assignments["discovery_label"] < 0).sum())
            assignments.to_parquet(parts / f"part-{part_index:05d}.parquet", index=False)
            part_index += 1
        if not prototypes:
            raise ValueError("discovery produced no supported completed motifs")
        prototype_frame = pd.concat(prototypes, ignore_index=True).sort_values(
            "discovered_motif_id", kind="mergesort"
        ).reset_index(drop=True)
        catalog_frame = pd.concat(catalogs, ignore_index=True).sort_values(
            "discovered_motif_id", kind="mergesort"
        ).reset_index(drop=True)
        global_scope = _stable_hash(
            {"config": asdict(config), "training_through": str(training_through),
             "provenance": dict(provenance)}
        )[:16]
        model_version = "incident-motif-v4-" + _stable_hash(
            {"scope": global_scope,
             "prototypes": prototype_frame.to_dict("records")}
        )[:16]
        prototype_frame["model_version"] = model_version
        catalog_frame["model_version"] = model_version
        feature_schema = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(MODEL_FEATURE_COLUMNS),
            "strata_columns": ["crop_name", "hazard_family"],
            "clip": 5.0,
            "stage_distance_weight": 0.0,
            "strata": schemas,
            "model_version": model_version,
        }
        part_glob = str(parts / "*.parquet")
        connection.execute(
            f"""
            COPY (SELECT *, {_sql_string(model_version)} AS model_version
                  FROM read_parquet({_sql_string(part_glob)}))
            TO {_sql_string(str(stage / 'completed_assignments.parquet'))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        _write_json(stage / "feature_schema.json", feature_schema)
        _write_parquet(stage / "completed_prototypes.parquet", prototype_frame)
        _write_parquet(stage / "completed_catalog.parquet", catalog_frame)
        manifest = {
            "status": "diagnostic_unreviewed",
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_version": model_version,
            "training_through": str(training_through),
            "config": asdict(config),
            "engine_used": config.engine,
            "training_story_count": training_count,
            "motif_count": len(prototype_frame),
            "noise_or_unsupported_count": noise_count,
            "stage_distance_weight": 0.0,
            "incident_identity_preserved": True,
            "publication_status": "blocked_pending_review_and_evaluation",
            "provenance": dict(provenance),
            "bounded_clustering_grain": "crop_name_x_hazard_family",
        }
        # Assignments live on disk; the review template needs only the catalog.
        return CompletedMotifModel(
            feature_schema, prototype_frame, catalog_frame,
            pd.DataFrame(), manifest,
        )
    finally:
        connection.close()
        for path in parts.glob("*.parquet"):
            path.unlink()
        parts.rmdir()


def _validate_viewer_story_checkpoint(
    viewer_dir: Path, viewer_manifest: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Bind motif learning to the exact causal checkpoint artifact in a viewer."""

    outputs = viewer_manifest.get("outputs") or {}
    if not isinstance(outputs, Mapping):
        raise ValueError("V4 viewer outputs must be a manifest object")
    checkpoint_name = str(outputs.get("story_checkpoints") or "")
    if checkpoint_name != VIEWER_STORY_CHECKPOINT_NAME:
        raise ValueError(
            "V4 motif learning requires viewer/story_checkpoints_v4.parquet"
        )
    artifacts = viewer_manifest.get("artifacts") or {}
    expected = artifacts.get(checkpoint_name) if isinstance(artifacts, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError("V4 viewer manifest is missing the story checkpoint artifact")
    checkpoint_path = (viewer_dir / checkpoint_name).resolve()
    if not checkpoint_path.is_relative_to(viewer_dir) or not checkpoint_path.is_file():
        raise ValueError("V4 viewer story checkpoint path escapes or is missing")
    actual = _fingerprint(checkpoint_path)
    try:
        expected_size = int(expected["size_bytes"])
        expected_rows = int(expected["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "V4 viewer story checkpoint has invalid size or row_count metadata"
        ) from exc
    if (
        str(expected.get("sha256") or "") != actual["sha256"]
        or expected_size != actual["size_bytes"]
    ):
        raise ValueError(
            "V4 viewer story checkpoint hash/size does not match its manifest"
        )
    with duckdb.connect(":memory:") as connection:
        actual_rows = _parquet_row_count(connection, checkpoint_path)
        columns = _parquet_columns(connection, checkpoint_path)
    if expected_rows < 0 or actual_rows != expected_rows:
        raise ValueError(
            "V4 viewer story checkpoint row_count does not match its manifest"
        )
    if "story_known_time" not in columns:
        raise ValueError(
            "V4 viewer story checkpoints require the full causal story_known_time"
        )
    return checkpoint_path, {**actual, "row_count": actual_rows}


def build_diagnostic_motif_release_v4(
    incident_dir: Path,
    daily_pressure_path: Path,
    s2_acquisition_path: Path,
    output_dir: Path,
    *,
    train_through: Any,
    calibration_through: Any,
    evaluation_through: Any,
    evidence_manifest_path: Path,
    viewer_dir: Path,
    config: MotifDiscoveryConfig = MotifDiscoveryConfig(),
    prefix_config: PrefixCalibrationConfig = PrefixCalibrationConfig(),
    threads: int = 16,
    memory_limit: str | None = "8GB",
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Build immutable discovery inputs/model and a pending review template."""

    incident_dir = incident_dir.expanduser().resolve()
    daily_pressure_path = daily_pressure_path.expanduser().resolve()
    s2_acquisition_path = s2_acquisition_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    viewer_dir = viewer_dir.expanduser().resolve()
    viewer_validation = validate_viewer_directory(viewer_dir)
    viewer_manifest_path = viewer_dir / "manifest.json"
    story_checkpoints_path = viewer_dir / VIEWER_STORY_CHECKPOINT_NAME
    sources = {
        **{name: incident_dir / file_name for name, file_name in SOURCE_NAMES.items()},
        "story_checkpoints": story_checkpoints_path,
        "daily_pressure": daily_pressure_path,
        "s2_acquisitions": s2_acquisition_path,
        "incident_manifest": incident_dir / "manifest.json",
        "viewer_manifest": viewer_manifest_path,
    }
    evidence_manifest_path = evidence_manifest_path.expanduser().resolve()
    evidence_root = evidence_manifest_path.parent
    if evidence_manifest_path != evidence_root / "manifest.json":
        raise ValueError("evidence_manifest_path must be the release manifest.json")
    if daily_pressure_path.parent != evidence_root or s2_acquisition_path.parent != evidence_root:
        raise ValueError("pressure, S2, and evidence manifest must belong to one release directory")
    evidence_validation = validate_evidence_directory(evidence_root)
    sources["evidence_manifest"] = evidence_manifest_path
    _validate_new_output(
        output_dir,
        protected=(
            incident_dir,
            viewer_dir,
            daily_pressure_path,
            s2_acquisition_path,
        ),
    )
    _require_files(sources)
    source_fingerprints = {name: _fingerprint(path) for name, path in sources.items()}
    viewer_manifest = _read_json(sources["viewer_manifest"])
    bound_checkpoint_path, story_checkpoint_binding = (
        _validate_viewer_story_checkpoint(viewer_dir, viewer_manifest)
    )
    if bound_checkpoint_path != story_checkpoints_path or any(
        source_fingerprints["story_checkpoints"][name]
        != story_checkpoint_binding[name]
        for name in ("sha256", "size_bytes")
    ):
        raise RuntimeError(
            "V4 viewer story checkpoint changed while binding motif sources"
        )
    source_manifest = _read_json(sources["incident_manifest"])
    run = source_manifest.get("run") or {}
    if run.get("status") != "complete" or run.get("immutable") is not True:
        raise ValueError("V4 motif learning requires an immutable complete Incident V3 release")
    if not bool((source_manifest.get("semantics") or {}).get("archetype_is_optional_not_identity")):
        raise ValueError("source manifest does not preserve incident/motif separation")
    source_generation_id = str(run.get("source_generation_id") or "").strip()
    evidence_manifest = _read_json(sources["evidence_manifest"])
    evidence_generation_id = str(
        (evidence_manifest.get("run") or {}).get("source_generation_id") or ""
    ).strip()
    if evidence_generation_id != str(
        evidence_validation.get("source_generation_id") or ""
    ).strip():
        raise RuntimeError("V4 evidence manifest changed while validating motif sources")
    if not source_generation_id or source_generation_id != evidence_generation_id:
        raise ValueError(
            "incident and evidence releases must have the same nonblank "
            "run.source_generation_id"
        )
    viewer_source = viewer_manifest.get("source") or {}
    if not isinstance(viewer_source, Mapping):
        raise ValueError("V4 viewer source bindings must be a manifest object")
    expected_viewer_bindings = {
        "incident_manifest_sha256": source_fingerprints["incident_manifest"]["sha256"],
        "evidence_manifest_sha256": source_fingerprints["evidence_manifest"]["sha256"],
    }
    for name, expected_sha256 in expected_viewer_bindings.items():
        if str(viewer_source.get(name) or "") != str(expected_sha256):
            raise ValueError(
                f"V4 viewer {name} does not match the selected source release"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Incident metadata is orders of magnitude smaller than the daily field
    # ledgers.  All daily/S2 joins and prefix expansion below remain in DuckDB
    # and on disk; pandas only sees one crop×hazard clustering cohort at a time.
    checkpoints = pd.read_parquet(sources["story_checkpoints"])
    if "story_known_time" not in checkpoints:
        raise ValueError(
            "V4 viewer story checkpoints require the full causal story_known_time"
        )
    checkpoints = checkpoints.copy()
    checkpoints["knowledge_time"] = checkpoints["story_known_time"]
    windows = pd.read_parquet(sources["windows"])
    lineage = pd.read_parquet(sources["lineage"])
    ledger = build_eligibility_ledger(windows, lineage, checkpoints)
    split = temporal_split_ledger(
        ledger,
        train_through=train_through,
        calibration_through=calibration_through,
        evaluation_through=evaluation_through,
    )
    with TemporaryDirectory(prefix=".incident-motif-v4-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        _write_parquet(stage / "eligibility_ledger.parquet", ledger)
        _write_parquet(stage / "temporal_split.parquet", split)
        joined_daily_path = stage / "incident_daily_pressure.parquet"
        joined_s2_path = stage / "incident_s2_acquisitions.parquet"
        adapter_stats = materialize_causal_incident_evidence_v4(
            sources["membership"], daily_pressure_path, s2_acquisition_path,
            joined_daily_path, joined_s2_path, threads=threads,
            memory_limit=memory_limit, temp_dir=temp_dir,
        )
        _materialize_learning_features_v4(
            sources["story_checkpoints"], joined_daily_path, joined_s2_path,
            stage / "eligibility_ledger.parquet",
            stage / "completed_story_features.parquet",
            stage / "causal_prefix_features.parquet",
            prefix_config=prefix_config, threads=threads,
            memory_limit=memory_limit, temp_dir=temp_dir,
        )
        model = _discover_completed_from_parquet_v4(
            stage / "completed_story_features.parquet",
            stage / "temporal_split.parquet",
            stage,
            training_through=train_through,
            config=config,
            provenance={
                "incident_generation_id": run.get("generation_id"),
                "source_generation_id": source_generation_id,
                "source_fingerprints": source_fingerprints,
            },
        )
        review = build_review_overlay_template(model)
        _write_parquet(stage / "review_overlay_template.parquet", review)
        manifest = {
            **model.manifest,
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "phase": "diagnostic_completed_discovery",
            "source": source_fingerprints,
            "split_counts": {
                str(key): int(value)
                for key, value in split["temporal_split"].value_counts().sort_index().items()
            },
            "eligibility_counts": {
                str(key): int(value)
                for key, value in ledger["eligibility_reason"].value_counts().sort_index().items()
            },
            "prefix_config": asdict(prefix_config),
            "evidence_path_adapter": adapter_stats,
            "evidence_validation": evidence_validation,
            "viewer_validation": viewer_validation,
            "release_binding": {
                "source_generation_id": source_generation_id,
                **expected_viewer_bindings,
                "viewer_manifest_sha256": source_fingerprints["viewer_manifest"][
                    "sha256"
                ],
                "story_checkpoints": story_checkpoint_binding,
            },
            "review_overlay_status": "pending_immutable_expert_review",
            "prefix_model_status": "blocked_pending_review",
            "map_publication_supported": False,
        }
        manifest["artifacts"] = _artifact_fingerprints(stage)
        _write_json(stage / "model_manifest.json", manifest)
        _verify_unchanged(sources, source_fingerprints)
        os.replace(stage, output_dir)
    return {
        "status": "diagnostic_unreviewed",
        "output_dir": str(output_dir),
        "model_version": model.manifest["model_version"],
        "training_story_count": int(model.manifest["training_story_count"]),
        "motif_count": len(model.prototypes),
        "prefix_model_status": "blocked_pending_review",
        "map_publication_supported": False,
    }


def _fit_prefix_from_parquet_v4(
    prefix_path: Path,
    reviewed: pd.DataFrame,
    split_path: Path,
    *,
    completed_model_version: str,
    config: PrefixCalibrationConfig,
) -> PrefixMotifModel:
    """Fit one exact maturity stratum at a time; never load global prefixes."""

    with TemporaryDirectory(prefix=".prefix-labels-v4-", dir=prefix_path.parent) as temporary:
        labels_path = Path(temporary) / "reviewed.parquet"
        reviewed[["incident_id", "reviewed_motif_id"]].to_parquet(
            labels_path, index=False, compression="zstd"
        )
        connection = duckdb.connect(":memory:")
        schemas: list[dict[str, Any]] = []
        prototypes: list[pd.DataFrame] = []
        try:
            strata = connection.execute(
                """
                SELECT DISTINCT p.crop_name, p.hazard_family,
                    p.weather_day_horizon, p.s2_acquisition_horizon
                FROM read_parquet(?) p
                JOIN read_parquet(?) r USING (incident_id)
                JOIN read_parquet(?) s USING (incident_id)
                WHERE s.temporal_split IN ('train', 'calibration')
                ORDER BY 1, 2, 3, 4
                """,
                [str(prefix_path), str(labels_path), str(split_path)],
            ).fetchall()
            for crop, hazard, weather_horizon, s2_horizon in strata:
                rows = connection.execute(
                    """
                    SELECT * EXCLUDE (rn) FROM (
                        SELECT p.*, r.reviewed_motif_id, s.temporal_split,
                            ROW_NUMBER() OVER (
                                PARTITION BY p.incident_id, p.crop_name,
                                    p.hazard_family, p.weather_day_horizon,
                                    p.s2_acquisition_horizon
                                ORDER BY p.prefix_as_of_time DESC
                            ) AS rn
                        FROM read_parquet(?) p
                        JOIN read_parquet(?) r USING (incident_id)
                        JOIN read_parquet(?) s USING (incident_id)
                        WHERE s.temporal_split IN ('train', 'calibration')
                          AND p.crop_name = ? AND p.hazard_family = ?
                          AND p.weather_day_horizon = ?
                          AND p.s2_acquisition_horizon = ?
                    ) WHERE rn = 1 ORDER BY incident_id
                    """,
                    [
                        str(prefix_path), str(labels_path), str(split_path), crop,
                        hazard, weather_horizon, s2_horizon,
                    ],
                ).fetchdf()
                if rows.empty or set(rows["temporal_split"]) != {"train", "calibration"}:
                    continue
                local_labels = rows[["incident_id", "reviewed_motif_id"]].drop_duplicates()
                local_split = rows[["incident_id", "temporal_split"]].drop_duplicates()
                features = rows.drop(columns=["reviewed_motif_id", "temporal_split"])
                try:
                    local = fit_calibrated_prefix_model(
                        features, local_labels, local_split,
                        model_version=completed_model_version, config=config,
                    )
                except ValueError as exc:
                    if any(
                        marker in str(exc)
                        for marker in (
                            "no reviewed prefix center met training support",
                            "no prefix center met calibration support",
                            "calibration cohort has no train-supported maturity strata",
                        )
                    ):
                        continue
                    raise
                schemas.extend(local.feature_schema["strata"])
                prototypes.append(local.prototypes.drop(columns=["model_version"]))
            if not prototypes:
                raise ValueError("no prefix maturity stratum met train/calibration support")
            combined = pd.concat(prototypes, ignore_index=True).sort_values(
                ["crop_name", "hazard_family", "weather_day_horizon",
                 "s2_acquisition_horizon", "reviewed_motif_id"],
                kind="mergesort",
            ).reset_index(drop=True)
            model_version = "incident-prefix-v4-" + _stable_hash(
                {"model": completed_model_version, "config": asdict(config),
                 "prototypes": combined.to_dict("records")}
            )[:16]
            combined["model_version"] = model_version
            schema = {
                "schema_version": PREFIX_SCHEMA_VERSION,
                "model_version": model_version,
                "feature_names": list(MODEL_FEATURE_COLUMNS),
                "strata_columns": ["crop_name", "hazard_family",
                    "weather_day_horizon", "s2_acquisition_horizon"],
                "clip": 5.0,
                "stage_distance_weight": 0.0,
                "strata": schemas,
            }
            manifest = {
                "status": "frozen_diagnostic",
                "schema_version": PREFIX_SCHEMA_VERSION,
                "model_version": model_version,
                "reviewed_completed_model_version": completed_model_version,
                "config": asdict(config),
                "center_fit_split": "train",
                "radius_margin_fit_split": "calibration",
                "stage_distance_weight": 0.0,
                "prototype_count": len(combined),
                "publication_status": "blocked_pending_rolling_evaluation",
                "bounded_fit_grain": "crop_hazard_weather_maturity_s2_maturity",
            }
            return PrefixMotifModel(schema, combined, manifest)
        finally:
            connection.close()


def fit_reviewed_prefix_release_v4(
    discovery_dir: Path,
    review_overlay_path: Path,
    reviewed_calibration_labels_path: Path,
    output_dir: Path,
    *,
    config: PrefixCalibrationConfig = PrefixCalibrationConfig(),
) -> dict[str, Any]:
    """Fit a frozen prefix model after an external immutable review decision."""

    discovery_dir = discovery_dir.expanduser().resolve()
    review_overlay_path = review_overlay_path.expanduser().resolve()
    reviewed_calibration_labels_path = reviewed_calibration_labels_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_new_output(
        output_dir,
        protected=(
            discovery_dir,
            review_overlay_path,
            reviewed_calibration_labels_path,
        ),
    )
    manifest_path = discovery_dir / "model_manifest.json"
    paths = {
        "manifest": manifest_path,
        "prefixes": discovery_dir / "causal_prefix_features.parquet",
        "assignments": discovery_dir / "completed_assignments.parquet",
        "split": discovery_dir / "temporal_split.parquet",
        "review_overlay": review_overlay_path,
        "reviewed_calibration_labels": reviewed_calibration_labels_path,
    }
    _require_files(paths)
    source_fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    manifest = _read_json(manifest_path)
    _verify_manifest_artifacts(
        manifest,
        {
            "causal_prefix_features.parquet": paths["prefixes"],
            "completed_assignments.parquet": paths["assignments"],
            "temporal_split.parquet": paths["split"],
        },
    )
    discovery_prefix_config = manifest.get("prefix_config") or {}
    expected_weather = tuple(discovery_prefix_config.get("weather_day_horizons") or ())
    expected_s2 = tuple(discovery_prefix_config.get("s2_acquisition_horizons") or ())
    if (
        expected_weather != tuple(config.weather_day_horizons)
        or expected_s2 != tuple(config.s2_acquisition_horizons)
    ):
        raise ValueError(
            "fit-prefix maturity horizons must exactly match the discovery feature release"
        )
    overlay = pd.read_parquet(review_overlay_path)
    approved_overlay = overlay[overlay["review_status"].isin({"approved", "merged"})]
    review_versions = set(approved_overlay["review_version"].dropna().astype(str))
    if len(review_versions) != 1 or approved_overlay["review_version"].isna().any():
        raise ValueError("approved review overlay requires one immutable review_version")
    review_version = next(iter(review_versions))
    discovery_assignments = pd.read_parquet(paths["assignments"])
    reviewed_training = reviewed_incident_assignments(discovery_assignments, overlay)
    if reviewed_training["reviewed_motif_id"].notna().sum() == 0:
        raise ValueError("review overlay approves no completed motif assignments")
    split = pd.read_parquet(paths["split"])
    calibration_labels = pd.read_parquet(reviewed_calibration_labels_path)
    _require_reviewed_calibration_labels(
        calibration_labels,
        split,
        overlay,
        review_overlay_sha256=source_fingerprints["review_overlay"]["sha256"],
    )
    reviewed = pd.concat(
        [
            reviewed_training[["incident_id", "reviewed_motif_id"]].dropna(),
            calibration_labels[["incident_id", "reviewed_motif_id"]],
        ],
        ignore_index=True,
    )
    if reviewed["incident_id"].astype(str).duplicated().any():
        raise ValueError("reviewed train and calibration incident labels overlap")
    model = _fit_prefix_from_parquet_v4(
        paths["prefixes"], reviewed, paths["split"],
        completed_model_version=str(manifest["model_version"]), config=config,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-prefix-v4-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        _write_json(stage / "prefix_feature_schema.json", model.feature_schema)
        _write_parquet(stage / "prefix_prototypes.parquet", model.prototypes)
        _write_parquet(stage / "reviewed_incident_assignments.parquet", reviewed)
        prefix_manifest = {
            **model.manifest,
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "source": source_fingerprints,
            "review_overlay_sha256": source_fingerprints["review_overlay"]["sha256"],
            "review_version": review_version,
            "reviewed_calibration_labels_sha256": source_fingerprints[
                "reviewed_calibration_labels"
            ]["sha256"],
            "map_publication_supported": False,
        }
        prefix_manifest["artifacts"] = _artifact_fingerprints(stage)
        _write_json(stage / "prefix_manifest.json", prefix_manifest)
        _verify_unchanged(paths, source_fingerprints)
        os.replace(stage, output_dir)
    return {
        "status": "frozen_diagnostic",
        "output_dir": str(output_dir),
        "model_version": model.manifest["model_version"],
        "prototype_count": len(model.prototypes),
        "map_publication_supported": False,
    }


def _stream_holdout_assignments_v4(
    prefix_path: Path,
    split_path: Path,
    model: PrefixMotifModel,
    output_path: Path,
    *,
    batch_rows: int = 100_000,
) -> None:
    """Assign sealed holdout prefixes in bounded Arrow record batches."""

    parts = output_path.parent / ".prefix_assignment_parts"
    parts.mkdir()
    connection = duckdb.connect(":memory:")
    part_index = 0
    try:
        reader = connection.execute(
            """
            SELECT p.* FROM read_parquet(?) p
            JOIN read_parquet(?) s USING (incident_id)
            WHERE s.eligible AND s.temporal_split = 'holdout'
            ORDER BY p.incident_id, p.prefix_as_of_time
            """,
            [str(prefix_path), str(split_path)],
        ).to_arrow_reader(batch_size=int(batch_rows))
        for batch in reader:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            assigned = assign_open_set_prefixes(frame, model)
            assigned.to_parquet(
                parts / f"part-{part_index:06d}.parquet",
                index=False,
                compression="zstd",
            )
            part_index += 1
        if part_index == 0:
            raise ValueError("rolling replay has no sealed holdout prefixes")
        connection.execute(
            f"""
            COPY (SELECT * FROM read_parquet({_sql_string(str(parts / '*.parquet'))}))
            TO {_sql_string(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()
        for path in parts.glob("*.parquet"):
            path.unlink()
        parts.rmdir()


def _evaluate_assignment_parquet_v4(
    assignments_path: Path, final_labels_path: Path, split_path: Path
) -> dict[str, Any]:
    """Compute replay gates/metrics in DuckDB without global pandas materialization."""

    connection = duckdb.connect(":memory:")
    try:
        assignments = _sql_string(str(assignments_path))
        labels = _sql_string(str(final_labels_path))
        split = _sql_string(str(split_path))
        duplicate_count = int(connection.execute(
            f"""SELECT COUNT(*) FROM (
                SELECT incident_id, prefix_as_of_time, COUNT(*) n
                FROM read_parquet({assignments}) GROUP BY 1,2 HAVING n > 1)"""
        ).fetchone()[0])
        overlap = connection.execute(
            f"""
            WITH groups AS (
                SELECT purge_group_id,
                    BOOL_OR(temporal_split='train') train,
                    BOOL_OR(temporal_split='calibration') calibration,
                    BOOL_OR(temporal_split='holdout') holdout
                FROM read_parquet({split}) GROUP BY purge_group_id)
            SELECT COUNT_IF(train AND calibration), COUNT_IF(train AND holdout),
                COUNT_IF(calibration AND holdout) FROM groups
            """
        ).fetchone()
        missing_labels = int(connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT a.incident_id FROM read_parquet({assignments}) a
                ANTI JOIN read_parquet({labels}) l USING (incident_id))
            """
        ).fetchone()[0])
        future_prefixes = int(connection.execute(
            f"""
            SELECT COUNT(*) FROM read_parquet({assignments}) a
            JOIN read_parquet({split}) s USING (incident_id)
            WHERE CAST(a.prefix_as_of_time AS TIMESTAMP)
                > CAST(s.feature_available_at AS TIMESTAMP)
            """
        ).fetchone()[0])
        replay_count = int(connection.execute(
            f"""SELECT COUNT(*) FROM read_parquet({assignments}) a
                 JOIN read_parquet({labels}) l USING (incident_id)"""
        ).fetchone()[0])
        if replay_count == 0:
            raise ValueError("rolling replay has no sealed labeled holdout rows")
        totals = connection.execute(
            f"""
            SELECT COUNT(*)::BIGINT AS n,
                COUNT_IF(assignment_status='tentative')::BIGINT AS accepted,
                COUNT_IF(assignment_status='tentative'
                    AND final_assignment_status='accepted'
                    AND a.reviewed_motif_id=l.reviewed_motif_id)::BIGINT AS correct,
                COUNT_IF(final_assignment_status='novel_unassigned')::BIGINT AS novel,
                COUNT_IF(assignment_status='tentative'
                    AND final_assignment_status='novel_unassigned')::BIGINT AS false_accept
            FROM read_parquet({assignments}) a
            JOIN read_parquet({labels}) l USING (incident_id)
            """
        ).fetchone()

        def grouped(
            columns: str, order: str, *, group_by: str | None = None
        ) -> list[dict[str, Any]]:
            rows = connection.execute(
                f"""
                SELECT {columns}, COUNT(*)::BIGINT AS prefix_count,
                    AVG((assignment_status='tentative')::INTEGER)::DOUBLE
                        AS accepted_coverage,
                    COUNT_IF(assignment_status='tentative'
                        AND final_assignment_status='accepted'
                        AND a.reviewed_motif_id=l.reviewed_motif_id)::DOUBLE
                        / GREATEST(COUNT_IF(assignment_status='tentative'),1)
                        AS accepted_known_precision,
                    COUNT_IF(assignment_status='tentative'
                        AND final_assignment_status='novel_unassigned')::DOUBLE
                        / GREATEST(COUNT_IF(final_assignment_status='novel_unassigned'),1)
                        AS final_novel_false_accept_rate
                FROM read_parquet({assignments}) a
                JOIN read_parquet({labels}) l USING (incident_id)
                GROUP BY {group_by or columns} ORDER BY {order}
                """
            ).fetchdf()
            return rows.to_dict("records")

        by_maturity = grouped(
            "weather_day_horizon, s2_acquisition_horizon", "1,2"
        )
        by_origin = grouped(
            "CAST(prefix_as_of_time AS DATE) AS origin", "1",
            group_by="CAST(prefix_as_of_time AS DATE)",
        )
        for row in by_origin:
            row["origin"] = pd.Timestamp(row["origin"], tz="UTC").isoformat()
        hard = {
            "unique_incident_as_of": duplicate_count == 0,
            "train_calibration_disjoint": int(overlap[0]) == 0,
            "train_holdout_disjoint": int(overlap[1]) == 0,
            "calibration_holdout_disjoint": int(overlap[2]) == 0,
            "holdout_only_evaluation": True,
            "complete_holdout_final_labels": missing_labels == 0,
            "prefixes_do_not_follow_final_knowledge": future_prefixes == 0,
        }
        n, accepted, correct, novel, false_accept = map(int, totals)
        return {
            "status": "complete" if all(hard.values()) else "failed_hard_gates",
            "phase": "diagnostic_rolling_replay",
            "hard_gates": {"passed": all(hard.values()), "checks": hard},
            "metrics": {
                "holdout_prefix_count": n,
                "accepted_coverage": accepted / max(n, 1),
                "accepted_known_precision": correct / max(accepted, 1),
                "final_novel_false_accept_rate": false_accept / max(novel, 1),
                "by_maturity": by_maturity,
                "by_origin": by_origin,
            },
            "warning": "Engineering replay only; not agronomic or outcome validation.",
        }
    finally:
        connection.close()


def evaluate_prefix_release_v4(
    discovery_dir: Path,
    prefix_model_dir: Path,
    final_labels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Replay the frozen model over the sealed holdout and persist diagnostics."""

    discovery_dir = discovery_dir.expanduser().resolve()
    prefix_model_dir = prefix_model_dir.expanduser().resolve()
    final_labels_path = final_labels_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_new_output(output_dir, protected=(discovery_dir, prefix_model_dir, final_labels_path))
    paths = {
        "discovery_manifest": discovery_dir / "model_manifest.json",
        "prefixes": discovery_dir / "causal_prefix_features.parquet",
        "split": discovery_dir / "temporal_split.parquet",
        "schema": prefix_model_dir / "prefix_feature_schema.json",
        "prototypes": prefix_model_dir / "prefix_prototypes.parquet",
        "prefix_manifest": prefix_model_dir / "prefix_manifest.json",
        "final_labels": final_labels_path,
    }
    _require_files(paths)
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    discovery_manifest = _read_json(paths["discovery_manifest"])
    prefix_manifest = _read_json(paths["prefix_manifest"])
    _verify_manifest_artifacts(
        discovery_manifest,
        {
            "causal_prefix_features.parquet": paths["prefixes"],
            "temporal_split.parquet": paths["split"],
        },
    )
    _verify_manifest_artifacts(
        prefix_manifest,
        {
            "prefix_feature_schema.json": paths["schema"],
            "prefix_prototypes.parquet": paths["prototypes"],
        },
    )
    source = prefix_manifest.get("source") or {}
    for source_name in ("manifest", "prefixes", "split"):
        expected = (source.get(source_name) or {}).get("sha256")
        actual = fingerprints[
            "discovery_manifest" if source_name == "manifest" else source_name
        ]["sha256"]
        if not expected or str(expected) != str(actual):
            raise ValueError(
                f"prefix model source hash does not match current {source_name} artifact"
            )
    prefix_schema = _read_json(paths["schema"])
    prefix_prototypes = pd.read_parquet(paths["prototypes"])
    final_labels = pd.read_parquet(final_labels_path)
    _validate_evaluation_chain(
        discovery_manifest,
        prefix_manifest,
        prefix_schema,
        prefix_prototypes,
        final_labels,
        fingerprints,
    )
    model = PrefixMotifModel(
        prefix_schema,
        prefix_prototypes,
        prefix_manifest,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-prefix-eval-v4-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        assignments_path = stage / "prefix_replay_assignments.parquet"
        _stream_holdout_assignments_v4(
            paths["prefixes"], paths["split"], model, assignments_path
        )
        report = _evaluate_assignment_parquet_v4(
            assignments_path, final_labels_path, paths["split"]
        )
        report["source"] = fingerprints
        report["map_publication_supported"] = False
        _write_json(stage / "evaluation.json", report)
        report["artifacts"] = _artifact_fingerprints(stage)
        _write_json(stage / "evaluation_manifest.json", report)
        _verify_unchanged(paths, fingerprints)
        os.replace(stage, output_dir)
    hard_passed = bool(report["hard_gates"]["passed"])
    return {
        "status": "complete" if hard_passed else "failed_hard_gates",
        "output_dir": str(output_dir),
        "hard_gates_passed": hard_passed,
        "metrics": report["metrics"],
        "map_publication_supported": False,
    }


def _validate_new_output(output: Path, *, protected: tuple[Path, ...]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable V4 output already exists: {output}")
    for path in protected:
        resolved = path.expanduser().resolve()
        if output == resolved or (resolved.is_dir() and output.is_relative_to(resolved)):
            raise ValueError("V4 output must not replace a protected input")


def _validate_evaluation_chain(
    discovery_manifest: Mapping[str, Any],
    prefix_manifest: Mapping[str, Any],
    prefix_schema: Mapping[str, Any],
    prefix_prototypes: pd.DataFrame,
    final_labels: pd.DataFrame,
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "incident_id",
        "final_assignment_status",
        "reviewed_motif_id",
        "discovery_model_version",
        "review_version",
        "review_overlay_sha256",
        "prefix_model_version",
        "prefix_manifest_sha256",
    }
    missing = sorted(required - set(final_labels.columns))
    if missing:
        raise ValueError(
            "sealed holdout labels are missing immutable bindings: "
            + ", ".join(missing)
        )
    discovery_version = str(discovery_manifest.get("model_version") or "")
    prefix_version = str(prefix_manifest.get("model_version") or "")
    review_version = str(prefix_manifest.get("review_version") or "")
    review_hash = str(prefix_manifest.get("review_overlay_sha256") or "")
    if not all((discovery_version, prefix_version, review_version, review_hash)):
        raise ValueError("model/review chain contains blank immutable identifiers")
    if str(prefix_manifest.get("reviewed_completed_model_version") or "") != discovery_version:
        raise ValueError("prefix model is not bound to the discovery model")
    source_manifest_hash = str(
        ((prefix_manifest.get("source") or {}).get("manifest") or {}).get("sha256")
        or ""
    )
    if source_manifest_hash != str(fingerprints["discovery_manifest"]["sha256"]):
        raise ValueError("prefix model discovery-manifest hash does not match")
    if str(prefix_schema.get("model_version") or "") != prefix_version:
        raise ValueError("prefix feature schema model version does not match")
    if prefix_prototypes.empty or set(
        prefix_prototypes["model_version"].astype(str)
    ) != {prefix_version}:
        raise ValueError("prefix prototypes model version does not match")
    frozen_motifs = set(prefix_prototypes["reviewed_motif_id"].dropna().astype(str))
    accepted_motifs = set(
        final_labels.loc[
            final_labels["final_assignment_status"].eq("accepted"),
            "reviewed_motif_id",
        ]
        .dropna()
        .astype(str)
    )
    if not accepted_motifs <= frozen_motifs:
        raise ValueError("sealed holdout labels reference motifs outside the frozen prefix model")
    expected = {
        "discovery_model_version": discovery_version,
        "review_version": review_version,
        "review_overlay_sha256": review_hash,
        "prefix_model_version": prefix_version,
        "prefix_manifest_sha256": str(fingerprints["prefix_manifest"]["sha256"]),
    }
    for column, value in expected.items():
        if set(final_labels[column].astype(str)) != {value}:
            raise ValueError(f"sealed holdout labels mismatch immutable {column}")


def _require_reviewed_calibration_labels(
    labels: pd.DataFrame,
    split: pd.DataFrame,
    overlay: pd.DataFrame,
    *,
    review_overlay_sha256: str,
) -> None:
    required = {
        "incident_id",
        "model_version",
        "reviewed_motif_id",
        "review_status",
        "review_version",
        "review_overlay_sha256",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(
            "reviewed calibration labels are missing columns: " + ", ".join(missing)
        )
    if labels.empty or labels["incident_id"].astype(str).duplicated().any():
        raise ValueError("reviewed calibration labels must be nonempty and unique")
    dispositions = labels["review_status"].astype(str)
    if not set(dispositions) <= {"approved", "novel_unassigned"}:
        raise ValueError(
            "every calibration incident requires approved or novel_unassigned disposition"
        )
    invalid_approved = dispositions.eq("approved") & labels["reviewed_motif_id"].isna()
    invalid_novel = dispositions.eq("novel_unassigned") & labels[
        "reviewed_motif_id"
    ].notna()
    if invalid_approved.any() or invalid_novel.any():
        raise ValueError(
            "approved calibration labels require a motif and novel dispositions require null"
        )
    overlay_versions = set(overlay["model_version"].dropna().astype(str))
    if len(overlay_versions) != 1 or set(labels["model_version"].astype(str)) != overlay_versions:
        raise ValueError("calibration labels do not match the reviewed discovery model")
    approved_overlay = overlay[overlay["review_status"].isin({"approved", "merged"})]
    review_versions = set(approved_overlay["review_version"].dropna().astype(str))
    if len(review_versions) != 1 or set(labels["review_version"].astype(str)) != review_versions:
        raise ValueError("calibration labels do not match the immutable review version")
    if set(labels["review_overlay_sha256"].astype(str)) != {review_overlay_sha256}:
        raise ValueError("calibration labels do not match the review overlay hash")
    calibration_mask = split["temporal_split"].eq("calibration")
    if "eligible" in split:
        calibration_mask &= split["eligible"].fillna(False).astype(bool)
    calibration_ids = set(split.loc[calibration_mask, "incident_id"].astype(str))
    supplied_ids = set(labels["incident_id"].astype(str))
    if supplied_ids != calibration_ids:
        missing = len(calibration_ids - supplied_ids)
        extra = len(supplied_ids - calibration_ids)
        raise ValueError(
            "reviewed calibration labels must exhaustively disposition every eligible "
            f"calibration incident (missing={missing}, extra={extra})"
        )
    approved_motifs = set(
        overlay.loc[
            overlay["review_status"].isin({"approved", "merged"}),
            "reviewed_motif_id",
        ]
        .dropna()
        .astype(str)
    )
    if not set(labels["reviewed_motif_id"].dropna().astype(str)) <= approved_motifs:
        raise ValueError("calibration labels reference motifs absent from approved overlay")


def _require_files(paths: Mapping[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V4 motif workflow is missing inputs: " + ", ".join(missing))


def _parquet_columns(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
        ).fetchall()
    }


def _require_column_names(
    columns: set[str], required: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _pick_column(
    columns: set[str], candidates: tuple[str, ...], label: str
) -> str:
    selected = next((name for name in candidates if name in columns), None)
    if selected is None:
        raise ValueError(f"{label} requires one of: {', '.join(candidates)}")
    return selected


def _optional_column_sql(
    columns: set[str], candidates: tuple[str, ...], fallback: str
) -> str:
    selected = next((name for name in candidates if name in columns), None)
    return _q(selected) if selected is not None else fallback


def _parquet_row_count(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> int:
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
        ).fetchone()[0]
    )


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": int(after.st_size)}


def _artifact_fingerprints(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: _fingerprint(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.name.endswith("manifest.json")
    }


def _verify_manifest_artifacts(
    manifest: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    artifacts = manifest.get("artifacts") or {}
    for name, path in paths.items():
        expected = artifacts.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f"producing manifest is missing artifact hash for {name}")
        actual = _fingerprint(path)
        if (
            str(expected.get("sha256") or "") != actual["sha256"]
            or int(expected.get("size_bytes", -1)) != actual["size_bytes"]
        ):
            raise ValueError(f"artifact does not match producing manifest: {name}")


def _verify_unchanged(
    paths: Mapping[str, Path], expected: Mapping[str, dict[str, Any]]
) -> None:
    changed = [name for name, path in paths.items() if _fingerprint(path) != expected[name]]
    if changed:
        raise RuntimeError("V4 motif source changed during workflow: " + ", ".join(changed))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")


__all__ = [
    "WORKFLOW_SCHEMA_VERSION",
    "build_diagnostic_motif_release_v4",
    "evaluate_prefix_release_v4",
    "fit_reviewed_prefix_release_v4",
    "materialize_causal_incident_evidence_v4",
]
