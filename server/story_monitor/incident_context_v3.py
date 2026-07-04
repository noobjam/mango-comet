"""DuckDB-first causal weekly context artifacts for the V3 incident tracker.

The builder retains monitoring coverage and event-state evidence.  It does not
estimate crop loss, biological outcome, or causal effect.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb

from .incident_policy_v3 import IncidentPolicyV3, load_incident_policy_v3


SCHEMA_VERSION = "crop-impact-incident-context-v3/1"
FIELD_WEEK_FILE = "field_week_context.parquet"
EVENT_WEEK_FILE = "event_week_lanes.parquet"
MANIFEST_FILE = "manifest.json"

_REQUIRED_COLUMNS = {
    "signals_source": {
        "field_id", "observation_date", "crop_name", "crop_season",
        "crop_instance_id", "crop_stage", "stage_family", "pressure_observed",
        "new_response_evidence", "risk_rank", "risk_band", "hazard_family",
        "response_class",
    },
    "snapshots_source": {
        "timeline_bucket", "snapshot_as_of_date", "field_id", "crop_name",
        "crop_season", "crop_instance_id", "event_id", "event_state",
        "hazard_signature", "max_risk_rank", "max_risk_band",
        "current_risk_rank", "current_risk_band", "reportable_day_count",
        "response_day_count", "right_censored", "is_data_gap_snapshot",
        "requires_review", "daily_response_class",
    },
    "events_source": {
        "event_id", "event_start_date", "event_end_date", "close_reason",
    },
    "memberships_source": {
        "event_id", "field_id", "crop_instance_id", "observation_date",
        "daily_response_class",
    },
    "geometry_source": {"field_id"},
}


def build_incident_context_v3(
    generation_dir: Path,
    output_dir: Path,
    *,
    policy: IncidentPolicyV3 | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Build immutable weekly cohort and event-lane Parquet artifacts."""
    generation_dir = generation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    policy = policy or load_incident_policy_v3()
    _validate_paths(generation_dir, output_dir, temp_dir)
    if isinstance(threads, bool) or not 1 <= int(threads) <= 256:
        raise ValueError("threads must be between 1 and 256")

    paths = {
        "signals_source": generation_dir / "daily_causal_signals.parquet",
        "snapshots_source": generation_dir / "event_state_snapshots.parquet",
        "events_source": generation_dir / "event_windows.parquet",
        "memberships_source": generation_dir / "story_day_membership.parquet",
        "geometry_source": _pick_geometry(generation_dir),
    }
    manifest_path = generation_dir / "manifest.json"
    for path in (*paths.values(), manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Generation is missing required Incident V3 artifact: {path}")
    source_manifest = _read_generation_manifest(manifest_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-context-v3-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        connection = _configured_connection(
            paths, policy, threads=int(threads), memory_limit=memory_limit, temp_dir=temp_dir
        )
        try:
            _validate_source_lineage(connection, source_manifest)
            connection.sql(_FIELD_WEEK_QUERY).write_parquet(
                str(stage / FIELD_WEEK_FILE), compression="zstd"
            )
            connection.read_parquet(str(stage / FIELD_WEEK_FILE)).create_view("field_week_context")
            connection.sql(_EVENT_WEEK_QUERY).write_parquet(
                str(stage / EVENT_WEEK_FILE), compression="zstd"
            )
            counts = _validate_outputs(connection, stage, policy)
        finally:
            connection.close()

        manifest = _context_manifest(
            source_manifest, manifest_path, paths, stage, counts, policy
        )
        (stage / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
    return {
        "status": "written",
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "policy_version": policy.version,
        **counts,
        "warning": policy.warning,
    }


def _configured_connection(
    paths: dict[str, Path],
    policy: IncidentPolicyV3,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET threads={threads}")
        if memory_limit:
            connection.execute("SET memory_limit=?", [memory_limit])
        if temp_dir is not None:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved)])
        for name, path in paths.items():
            relation = connection.read_parquet(str(path))
            missing = sorted(_REQUIRED_COLUMNS[name] - set(relation.columns))
            if missing:
                raise ValueError(f"{path.name} is missing Incident V3 columns: {', '.join(missing)}")
            if name == "geometry_source":
                relation.create_view("geometry_source_raw")
                _create_geometry_view(connection, set(relation.columns))
            else:
                relation.create_view(name)
        connection.execute(
            "CREATE TEMP TABLE stage_aliases(raw_stage VARCHAR PRIMARY KEY, stage_bucket VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO stage_aliases VALUES (?, ?)",
            [(item.raw_stage, item.stage_bucket) for item in policy.stage_aliases],
        )
        connection.execute(
            "CREATE TEMP TABLE lane_priorities(event_state VARCHAR PRIMARY KEY, priority INTEGER, is_open BOOLEAN)"
        )
        connection.executemany(
            "INSERT INTO lane_priorities VALUES (?, ?, ?)",
            [
                (item.event_state, item.priority, item.is_open)
                for item in policy.lane_state_priorities
            ],
        )
        connection.execute(
            "CREATE TEMP TABLE incident_policy(policy_version VARCHAR, policy_sha256 VARCHAR)"
        )
        connection.execute(
            "INSERT INTO incident_policy VALUES (?, ?)",
            [policy.version, policy.source_sha256],
        )
        return connection
    except Exception:
        connection.close()
        raise


def _create_geometry_view(
    connection: duckdb.DuckDBPyConnection, columns: set[str]
) -> None:
    def optional(name: str, sql_type: str) -> str:
        if name in columns:
            return f'TRY_CAST("{name}" AS {sql_type}) AS {name}'
        return f"CAST(NULL AS {sql_type}) AS {name}"

    connection.execute(
        f"""
        CREATE TEMP VIEW geometry_source AS
        SELECT
            TRIM(CAST(field_id AS VARCHAR)) AS field_id,
            {optional('centroid_lon', 'DOUBLE')},
            {optional('centroid_lat', 'DOUBLE')},
            {optional('district', 'VARCHAR')},
            {optional('sector', 'VARCHAR')},
            {optional('cell', 'VARCHAR')},
            {optional('village', 'VARCHAR')},
            TRUE AS geometry_record_present
        FROM geometry_source_raw
        """
    )


def _pick_geometry(generation_dir: Path) -> Path:
    for name in ("map_field_geometry.parquet", "field_geometry.parquet"):
        path = generation_dir / name
        if path.is_file():
            return path
    return generation_dir / "map_field_geometry.parquet"


_NORMALIZED_STAGE_SQL = """
COALESCE(NULLIF(TRIM(BOTH '_' FROM REGEXP_REPLACE(
    LOWER(COALESCE(CAST(stage_family_raw AS VARCHAR), 'unknown')),
    '[^a-z0-9]+', '_', 'g'
)), ''), 'unknown')
"""


_FIELD_WEEK_QUERY = f"""
WITH source AS (
    SELECT
        DATE_TRUNC('week', CAST(observation_date AS DATE))::DATE AS timeline_bucket,
        CAST(field_id AS VARCHAR) AS field_id,
        CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
        CAST(observation_date AS DATE) AS observation_date,
        CAST(crop_name AS VARCHAR) AS crop_name,
        CAST(crop_season AS VARCHAR) AS crop_season,
        CAST(crop_stage AS VARCHAR) AS stage_raw,
        CAST(stage_family AS VARCHAR) AS stage_family_raw,
        COALESCE(TRY_CAST(pressure_observed AS BOOLEAN), FALSE) AS pressure_observed,
        COALESCE(TRY_CAST(new_response_evidence AS BOOLEAN), FALSE) AS new_response_evidence,
        TRY_CAST(risk_rank AS INTEGER) AS risk_rank,
        CAST(risk_band AS VARCHAR) AS risk_band,
        CAST(hazard_family AS VARCHAR) AS hazard_family,
        CAST(response_class AS VARCHAR) AS response_class
    FROM signals_source
    WHERE field_id IS NOT NULL AND crop_instance_id IS NOT NULL
      AND observation_date IS NOT NULL
), latest AS (
    SELECT *
    FROM source
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY timeline_bucket, field_id, crop_instance_id
        ORDER BY observation_date DESC, crop_name, crop_season, stage_family_raw, stage_raw
    ) = 1
), aggregate AS (
    SELECT
        timeline_bucket, field_id, crop_instance_id,
        MAX(observation_date) AS week_last_observation_date,
        COUNT(DISTINCT observation_date) AS observation_day_count,
        COUNT_IF(pressure_observed) AS pressure_observed_day_count,
        COUNT_IF(new_response_evidence) AS new_response_evidence_day_count,
        COUNT_IF(
            new_response_evidence
            AND LOWER(response_class) IN ('medium_decline', 'severe_decline')
        ) AS fresh_decline_day_count,
        COUNT_IF(
            new_response_evidence AND LOWER(response_class) = 'severe_decline'
        ) AS fresh_severe_decline_day_count,
        COUNT_IF(
            new_response_evidence AND LOWER(response_class) = 'recovery'
        ) AS fresh_recovery_day_count,
        BOOL_OR(pressure_observed OR new_response_evidence) AS evaluable
    FROM source
    GROUP BY timeline_bucket, field_id, crop_instance_id
), named AS (
    SELECT l.*, {_NORMALIZED_STAGE_SQL} AS stage_family_normalized
    FROM latest l
)
SELECT
    a.timeline_bucket, a.field_id, a.crop_instance_id,
    n.crop_name, n.crop_season,
    n.stage_raw,
    n.stage_family_raw,
    n.stage_family_normalized,
    COALESCE(sa.stage_bucket, 'unknown') AS stage_bucket,
    n.observation_date AS stage_source_date,
    a.week_last_observation_date,
    a.observation_day_count,
    a.pressure_observed_day_count,
    a.new_response_evidence_day_count,
    a.fresh_decline_day_count,
    a.fresh_severe_decline_day_count,
    a.fresh_recovery_day_count,
    TRUE AS monitored,
    a.evaluable,
    n.risk_rank AS latest_risk_rank,
    n.risk_band AS latest_risk_band,
    n.hazard_family AS latest_hazard_family,
    n.response_class AS latest_response_class,
    g.centroid_lon,
    g.centroid_lat,
    g.district,
    g.sector,
    g.cell,
    g.village,
    g.field_id IS NOT NULL AS geometry_present,
    COALESCE(
        ISFINITE(g.centroid_lon) AND ISFINITE(g.centroid_lat)
        AND g.centroid_lon BETWEEN -180 AND 180
        AND g.centroid_lat BETWEEN -90 AND 90,
        FALSE
    )
        AS centroid_available,
    CASE
        WHEN g.field_id IS NULL THEN 'geometry_missing'
        WHEN COALESCE(
            ISFINITE(g.centroid_lon) AND ISFINITE(g.centroid_lat)
            AND g.centroid_lon BETWEEN -180 AND 180
            AND g.centroid_lat BETWEEN -90 AND 90,
            FALSE
        )
            THEN 'centroid_available'
        ELSE 'geometry_without_centroid'
    END AS geometry_join_status,
    p.policy_version,
    p.policy_sha256
FROM aggregate a
JOIN named n USING (timeline_bucket, field_id, crop_instance_id)
LEFT JOIN stage_aliases sa ON sa.raw_stage = n.stage_family_normalized
LEFT JOIN geometry_source g ON g.field_id = a.field_id
CROSS JOIN incident_policy p
ORDER BY a.timeline_bucket, a.field_id, a.crop_instance_id
"""


_EVENT_WEEK_QUERY = f"""
WITH snapshots AS (
    SELECT
        CAST(timeline_bucket AS DATE) AS timeline_bucket,
        CAST(snapshot_as_of_date AS DATE) AS snapshot_as_of_date,
        CAST(field_id AS VARCHAR) AS field_id,
        CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
        CAST(event_id AS VARCHAR) AS event_id,
        CAST(crop_name AS VARCHAR) AS crop_name,
        CAST(crop_season AS VARCHAR) AS crop_season,
        UPPER(CAST(event_state AS VARCHAR)) AS event_state,
        CAST(hazard_signature AS VARCHAR) AS hazard_family,
        TRY_CAST(max_risk_rank AS INTEGER) AS max_risk_rank,
        CAST(max_risk_band AS VARCHAR) AS max_risk_band,
        TRY_CAST(current_risk_rank AS INTEGER) AS current_risk_rank,
        CAST(current_risk_band AS VARCHAR) AS current_risk_band,
        TRY_CAST(reportable_day_count AS BIGINT) AS reportable_day_count,
        TRY_CAST(response_day_count AS BIGINT) AS response_day_count,
        COALESCE(TRY_CAST(right_censored AS BOOLEAN), FALSE) AS right_censored,
        COALESCE(TRY_CAST(is_data_gap_snapshot AS BOOLEAN), FALSE) AS is_data_gap_snapshot,
        COALESCE(TRY_CAST(requires_review AS BOOLEAN), FALSE) AS requires_review,
        CAST(daily_response_class AS VARCHAR) AS daily_response_class
    FROM snapshots_source
), signal_stage AS (
    SELECT
        CAST(field_id AS VARCHAR) AS field_id,
        CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
        CAST(observation_date AS DATE) AS stage_source_date,
        CAST(crop_stage AS VARCHAR) AS stage_raw,
        CAST(stage_family AS VARCHAR) AS stage_family_raw,
        COALESCE(TRY_CAST(new_response_evidence AS BOOLEAN), FALSE)
            AS signal_new_response_evidence,
        CAST(response_class AS VARCHAR) AS signal_response_class
    FROM signals_source
    WHERE observation_date IS NOT NULL
), event_week_response AS (
    SELECT
        DATE_TRUNC('week', CAST(observation_date AS DATE))::DATE AS timeline_bucket,
        CAST(event_id AS VARCHAR) AS event_id,
        COUNT_IF(
            LOWER(COALESCE(CAST(daily_response_class AS VARCHAR), ''))
                IN ('medium_decline', 'severe_decline', 'recovery')
        ) AS fresh_response_day_count,
        COUNT_IF(
            LOWER(COALESCE(CAST(daily_response_class AS VARCHAR), ''))
                IN ('medium_decline', 'severe_decline')
        ) AS fresh_decline_day_count,
        COUNT_IF(
            LOWER(COALESCE(CAST(daily_response_class AS VARCHAR), '')) = 'severe_decline'
        ) AS fresh_severe_decline_day_count,
        COUNT_IF(
            LOWER(COALESCE(CAST(daily_response_class AS VARCHAR), '')) = 'recovery'
        ) AS fresh_recovery_day_count
    FROM memberships_source
    GROUP BY 1, 2
), causal_stage AS (
    SELECT s.*, d.stage_source_date, d.stage_raw, d.stage_family_raw,
        d.signal_new_response_evidence, d.signal_response_class
    FROM snapshots s
    ASOF LEFT JOIN signal_stage d
      ON s.field_id = d.field_id
     AND s.crop_instance_id = d.crop_instance_id
     AND s.snapshot_as_of_date >= d.stage_source_date
), named AS (
    SELECT s.*, {_NORMALIZED_STAGE_SQL} AS stage_family_normalized
    FROM causal_stage s
), enriched AS (
    SELECT
        s.*,
        COALESCE(sa.stage_bucket, 'unknown') AS stage_bucket,
        DATE_DIFF('day', s.stage_source_date, s.snapshot_as_of_date) AS stage_age_days,
        CAST(e.event_start_date AS DATE) AS event_start_date,
        CAST(e.event_end_date AS DATE) AS event_end_date,
        CAST(e.close_reason AS VARCHAR) AS close_reason,
        COALESCE(lp.priority, 0) AS lane_state_priority,
        COALESCE(lp.is_open, FALSE) AS is_open,
        COALESCE(m.monitored, FALSE) AS monitored,
        COALESCE(m.evaluable, FALSE) AS evaluable,
        m.centroid_lon,
        m.centroid_lat,
        m.district,
        m.sector,
        m.cell,
        m.village,
        COALESCE(m.geometry_present, FALSE) AS geometry_present,
        COALESCE(m.centroid_available, FALSE) AS centroid_available,
        COALESCE(m.geometry_join_status, 'no_weekly_context') AS geometry_join_status,
        COALESCE(r.fresh_response_day_count, 0) AS fresh_response_day_count,
        COALESCE(r.fresh_decline_day_count, 0) AS fresh_decline_day_count,
        COALESCE(r.fresh_severe_decline_day_count, 0) AS fresh_severe_decline_day_count,
        COALESCE(r.fresh_recovery_day_count, 0) AS fresh_recovery_day_count,
        p.policy_version,
        p.policy_sha256
    FROM named s
    LEFT JOIN stage_aliases sa ON sa.raw_stage = s.stage_family_normalized
    LEFT JOIN events_source e ON CAST(e.event_id AS VARCHAR) = s.event_id
    LEFT JOIN lane_priorities lp ON lp.event_state = s.event_state
    LEFT JOIN field_week_context m
      ON m.timeline_bucket = s.timeline_bucket
     AND m.field_id = s.field_id
     AND m.crop_instance_id = s.crop_instance_id
    LEFT JOIN event_week_response r
      ON r.timeline_bucket = s.timeline_bucket
     AND r.event_id = s.event_id
    CROSS JOIN incident_policy p
), ranked AS (
    SELECT *,
        COUNT(*) OVER (
            PARTITION BY timeline_bucket, field_id, hazard_family
        ) AS concurrent_field_hazard_episode_count,
        ROW_NUMBER() OVER (
            PARTITION BY timeline_bucket, field_id, hazard_family
            ORDER BY lane_state_priority DESC,
                COALESCE(current_risk_rank, max_risk_rank, 0) DESC,
                COALESCE(max_risk_rank, 0) DESC,
                COALESCE(reportable_day_count, 0) DESC,
                event_id ASC
        ) AS field_hazard_lane_rank
    FROM enriched
)
SELECT
    timeline_bucket, snapshot_as_of_date,
    event_id AS episode_id, event_id, field_id, crop_instance_id,
    crop_name, crop_season,
    stage_raw, stage_family_raw, stage_family_normalized, stage_bucket,
    stage_source_date, stage_age_days,
    hazard_family, event_state, lane_state_priority, is_open,
    event_start_date, event_end_date, close_reason,
    COALESCE(
        event_start_date BETWEEN timeline_bucket AND timeline_bucket + INTERVAL 6 DAY,
        FALSE
    ) AS is_new_this_week,
    COALESCE(
        event_end_date BETWEEN timeline_bucket AND timeline_bucket + INTERVAL 6 DAY,
        FALSE
    ) AS is_closed_this_week,
    current_risk_rank, current_risk_band, max_risk_rank, max_risk_band,
    daily_response_class,
    fresh_response_day_count > 0 AS fresh_response_evidence,
    CASE
        WHEN fresh_severe_decline_day_count > 0 THEN 'severe_decline'
        WHEN fresh_decline_day_count > 0 THEN 'medium_decline'
        WHEN fresh_recovery_day_count > 0 THEN 'recovery'
        ELSE 'no_new_event_response'
    END AS signal_response_class,
    fresh_response_day_count, fresh_decline_day_count,
    fresh_severe_decline_day_count, fresh_recovery_day_count,
    reportable_day_count, response_day_count,
    right_censored, is_data_gap_snapshot, requires_review,
    monitored, evaluable,
    centroid_lon, centroid_lat, district, sector, cell, village,
    geometry_present, centroid_available, geometry_join_status,
    concurrent_field_hazard_episode_count,
    field_hazard_lane_rank,
    field_hazard_lane_rank = 1 AS is_canonical_field_hazard_week,
    field_hazard_lane_rank = 1 AS is_canonical_field_hazard_lane,
    policy_version, policy_sha256
FROM ranked
ORDER BY timeline_bucket, field_id, hazard_family, field_hazard_lane_rank, event_id
"""


def _validate_source_lineage(
    connection: duckdb.DuckDBPyConnection, manifest: dict[str, Any]
) -> None:
    as_of = str((manifest.get("run") or {}).get("as_of_date") or "")[:10]
    invalid_signal_keys = int(connection.execute(
        """
        SELECT COUNT(*) FROM signals_source
        WHERE field_id IS NULL OR TRIM(CAST(field_id AS VARCHAR)) = ''
           OR crop_instance_id IS NULL OR TRIM(CAST(crop_instance_id AS VARCHAR)) = ''
           OR TRY_CAST(observation_date AS DATE) IS NULL
        """
    ).fetchone()[0])
    invalid_snapshot_keys = int(connection.execute(
        """
        SELECT COUNT(*) FROM snapshots_source
        WHERE event_id IS NULL OR TRIM(CAST(event_id AS VARCHAR)) = ''
           OR field_id IS NULL OR TRIM(CAST(field_id AS VARCHAR)) = ''
           OR crop_instance_id IS NULL OR TRIM(CAST(crop_instance_id AS VARCHAR)) = ''
           OR TRY_CAST(timeline_bucket AS DATE) IS NULL
           OR TRY_CAST(snapshot_as_of_date AS DATE) IS NULL
           OR DATE_TRUNC('week', TRY_CAST(timeline_bucket AS DATE))::DATE
                IS DISTINCT FROM TRY_CAST(timeline_bucket AS DATE)
           OR DATE_TRUNC('week', TRY_CAST(snapshot_as_of_date AS DATE))::DATE
                IS DISTINCT FROM TRY_CAST(timeline_bucket AS DATE)
        """
    ).fetchone()[0])
    duplicate_signals = int(connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT CAST(field_id AS VARCHAR), CAST(crop_instance_id AS VARCHAR),
                 CAST(observation_date AS DATE), COUNT(*)
          FROM signals_source GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0])
    duplicate_snapshots = int(connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT CAST(event_id AS VARCHAR), CAST(timeline_bucket AS DATE), COUNT(*)
          FROM snapshots_source GROUP BY 1, 2 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0])
    duplicate_events = int(connection.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT CAST(event_id AS VARCHAR)) FROM events_source"
    ).fetchone()[0])
    duplicate_geometry = int(connection.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT CAST(field_id AS VARCHAR)) FROM geometry_source"
    ).fetchone()[0])
    orphan_snapshots = int(connection.execute(
        """
        SELECT COUNT(*) FROM snapshots_source s
        LEFT JOIN events_source e
          ON CAST(e.event_id AS VARCHAR) = CAST(s.event_id AS VARCHAR)
        WHERE e.event_id IS NULL
        """
    ).fetchone()[0])
    orphan_memberships = int(connection.execute(
        """
        SELECT COUNT(*) FROM memberships_source m
        LEFT JOIN events_source e
          ON CAST(e.event_id AS VARCHAR) = CAST(m.event_id AS VARCHAR)
        WHERE e.event_id IS NULL
        """
    ).fetchone()[0])
    future_signals = int(connection.execute(
        "SELECT COUNT(*) FROM signals_source WHERE CAST(observation_date AS DATE) > CAST(? AS DATE)",
        [as_of],
    ).fetchone()[0])
    future_snapshots = int(connection.execute(
        """
        SELECT COUNT(*) FROM snapshots_source
        WHERE CAST(snapshot_as_of_date AS DATE) > CAST(? AS DATE)
           OR CAST(timeline_bucket AS DATE) > CAST(? AS DATE)
        """,
        [as_of, as_of],
    ).fetchone()[0])
    if (
        invalid_signal_keys or invalid_snapshot_keys
        or duplicate_signals or duplicate_snapshots or duplicate_events or duplicate_geometry
        or orphan_snapshots or orphan_memberships or future_signals or future_snapshots
    ):
        raise ValueError(
            "Incident V3 source lineage failed: "
            f"invalid_signal_keys={invalid_signal_keys}, "
            f"invalid_snapshot_keys={invalid_snapshot_keys}, "
            f"duplicate_signals={duplicate_signals}, duplicate_snapshots={duplicate_snapshots}, "
            f"duplicate_events={duplicate_events}, duplicate_geometry={duplicate_geometry}, "
            f"orphan_snapshots={orphan_snapshots}, orphan_memberships={orphan_memberships}, "
            f"future_signals={future_signals}, future_snapshots={future_snapshots}"
        )


def _validate_outputs(
    connection: duckdb.DuckDBPyConnection,
    stage: Path,
    policy: IncidentPolicyV3,
) -> dict[str, Any]:
    field_path = stage / FIELD_WEEK_FILE
    lane_path = stage / EVENT_WEEK_FILE
    (
        field_rows,
        monitored_rows,
        evaluable_rows,
        geometry_present_rows,
        geometry_missing_rows,
        centroid_available_rows,
    ) = connection.execute(
        """
        SELECT COUNT(*),
            COUNT_IF(monitored),
            COUNT_IF(evaluable),
            COUNT_IF(geometry_present),
            COUNT_IF(NOT geometry_present),
            COUNT_IF(centroid_available)
        FROM read_parquet(?)
        """,
        [str(field_path)],
    ).fetchone()
    duplicate_fields = int(connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT timeline_bucket, field_id, crop_instance_id, COUNT(*)
          FROM read_parquet(?) GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
        )
        """,
        [str(field_path)],
    ).fetchone()[0])
    (
        source_field_count,
        centroid_field_count,
        source_crop_instance_week_count,
        centroid_crop_instance_week_count,
    ) = connection.execute(
        """
        SELECT
            COUNT(DISTINCT CAST(field_id AS VARCHAR)),
            COUNT(DISTINCT CAST(field_id AS VARCHAR))
                FILTER (WHERE centroid_available),
            COUNT(*),
            COUNT_IF(centroid_available)
        FROM read_parquet(?)
        WHERE monitored
        """,
        [str(field_path)],
    ).fetchone()
    field_centroid_coverage = (
        float(centroid_field_count) / float(source_field_count)
        if source_field_count else 0.0
    )
    crop_instance_week_centroid_coverage = (
        float(centroid_crop_instance_week_count)
        / float(source_crop_instance_week_count)
        if source_crop_instance_week_count else 0.0
    )
    known_stage_rows = int(
        connection.execute(
            """
            SELECT COUNT_IF(stage_bucket <> 'unknown')
            FROM read_parquet(?) WHERE monitored
            """,
            [str(field_path)],
        ).fetchone()[0]
    )
    known_stage_coverage = (
        float(known_stage_rows) / float(source_crop_instance_week_count)
        if source_crop_instance_week_count else 0.0
    )
    stage_coverage_by_crop = [
        {
            "crop_name": str(row[0]),
            "crop_instance_week_count": int(row[1]),
            "known_stage_crop_instance_week_count": int(row[2]),
            "known_stage_coverage": float(row[3]),
        }
        for row in connection.execute(
            """
            SELECT
                LOWER(TRIM(CAST(crop_name AS VARCHAR))) AS crop_name,
                COUNT(*) AS crop_instance_weeks,
                COUNT_IF(stage_bucket <> 'unknown') AS known_stage_weeks,
                COUNT_IF(stage_bucket <> 'unknown')::DOUBLE / COUNT(*)
                    AS known_stage_coverage
            FROM read_parquet(?)
            WHERE monitored
            GROUP BY 1
            ORDER BY crop_instance_weeks DESC, crop_name
            """,
            [str(field_path)],
        ).fetchall()
    ]
    stage_coverage_below_policy = [
        row
        for row in stage_coverage_by_crop
        if row["crop_instance_week_count"]
        >= policy.minimum_stage_coverage_crop_instance_weeks
        and row["known_stage_coverage"]
        < policy.minimum_known_stage_coverage_per_supported_crop
    ]
    top_unmapped_stage_labels = [
        {
            "stage_family_raw": str(row[0]),
            "stage_family_normalized": str(row[1]),
            "crop_instance_week_count": int(row[2]),
        }
        for row in connection.execute(
            """
            SELECT
                COALESCE(CAST(stage_family_raw AS VARCHAR), '<null>'),
                COALESCE(CAST(stage_family_normalized AS VARCHAR), 'unknown'),
                COUNT(*) AS crop_instance_weeks
            FROM read_parquet(?)
            WHERE monitored AND stage_bucket = 'unknown'
            GROUP BY 1, 2
            ORDER BY crop_instance_weeks DESC, 1
            LIMIT 20
            """,
            [str(field_path)],
        ).fetchall()
    ]
    lane_rows, future_stage = connection.execute(
        """
        SELECT COUNT(*),
            COUNT_IF(stage_source_date > snapshot_as_of_date)
        FROM read_parquet(?)
        """,
        [str(lane_path)],
    ).fetchone()
    duplicate_lanes = int(connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT timeline_bucket, event_id, COUNT(*)
          FROM read_parquet(?) GROUP BY 1, 2 HAVING COUNT(*) > 1
        )
        """,
        [str(lane_path)],
    ).fetchone()[0])
    bad_canonical = int(connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT timeline_bucket, field_id, hazard_family,
                 COUNT_IF(is_canonical_field_hazard_lane) AS canonical_count
          FROM read_parquet(?) GROUP BY 1, 2, 3
          HAVING canonical_count <> 1
        )
        """,
        [str(lane_path)],
    ).fetchone()[0])
    if (
        not field_rows or monitored_rows != field_rows
        or geometry_present_rows + geometry_missing_rows != field_rows
        or duplicate_fields or duplicate_lanes or future_stage or bad_canonical
        or field_centroid_coverage
            < policy.minimum_source_field_centroid_coverage
        or crop_instance_week_centroid_coverage
            < policy.minimum_source_crop_instance_week_centroid_coverage
        or known_stage_coverage < policy.minimum_known_stage_coverage
        or stage_coverage_below_policy
    ):
        raise ValueError(
            "Incident V3 output reconciliation failed: "
            f"field_rows={field_rows}, duplicate_fields={duplicate_fields}, "
            f"monitored_rows={monitored_rows}, evaluable_rows={evaluable_rows}, "
            f"geometry_present={geometry_present_rows}, geometry_missing={geometry_missing_rows}, "
            f"centroid_fields={centroid_field_count}/{source_field_count} "
            f"({field_centroid_coverage:.2%}, minimum "
            f"{policy.minimum_source_field_centroid_coverage:.2%}), "
            f"centroid_crop_instance_weeks={centroid_crop_instance_week_count}/"
            f"{source_crop_instance_week_count} "
            f"({crop_instance_week_centroid_coverage:.2%}, minimum "
            f"{policy.minimum_source_crop_instance_week_centroid_coverage:.2%}), "
            f"known_stage_crop_instance_weeks={known_stage_rows}/"
            f"{source_crop_instance_week_count} "
            f"({known_stage_coverage:.2%}, minimum "
            f"{policy.minimum_known_stage_coverage:.2%}), "
            f"top_unmapped_stage_labels={top_unmapped_stage_labels[:5]}, "
            f"supported_crops_below_stage_policy={stage_coverage_below_policy}, "
            f"duplicate_lanes={duplicate_lanes}, future_stage={future_stage}, "
            f"bad_canonical_groups={bad_canonical}"
        )
    return {
        "field_week_row_count": int(field_rows),
        "monitored_crop_instance_week_count": int(monitored_rows),
        "evaluable_crop_instance_week_count": int(evaluable_rows),
        "geometry_present_crop_instance_week_count": int(geometry_present_rows),
        "geometry_missing_crop_instance_week_count": int(geometry_missing_rows),
        "centroid_available_crop_instance_week_count": int(centroid_available_rows),
        "source_field_count": int(source_field_count),
        "centroid_available_field_count": int(centroid_field_count),
        "source_field_centroid_coverage": field_centroid_coverage,
        "source_crop_instance_week_centroid_coverage": (
            crop_instance_week_centroid_coverage
        ),
        "known_stage_crop_instance_week_count": known_stage_rows,
        "source_known_stage_coverage": known_stage_coverage,
        "stage_coverage_by_crop": stage_coverage_by_crop,
        "stage_coverage_below_policy": stage_coverage_below_policy,
        "minimum_stage_coverage_crop_instance_weeks": (
            policy.minimum_stage_coverage_crop_instance_weeks
        ),
        "top_unmapped_stage_labels": top_unmapped_stage_labels,
        "event_week_lane_count": int(lane_rows),
    }


def _read_generation_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Generation manifest is invalid: {path}") from exc
    run = payload.get("run") if isinstance(payload, dict) else None
    if not isinstance(run, dict) or run.get("status") != "complete" or run.get("immutable") is not True:
        raise ValueError("Incident V3 requires a completed immutable generation")
    as_of = str(run.get("as_of_date") or "")[:10]
    if not as_of:
        raise ValueError("Generation manifest is missing run.as_of_date")
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError("Generation manifest run.as_of_date is invalid") from exc
    return payload


def _context_manifest(
    source: dict[str, Any],
    source_manifest_path: Path,
    paths: dict[str, Path],
    stage: Path,
    counts: dict[str, Any],
    policy: IncidentPolicyV3,
) -> dict[str, Any]:
    run = source.get("run") or {}
    starter_parameters = {
        name: getattr(policy, name)
        for name in (
            "identity_namespace",
            "grid_cell_size_km", "grid_origin_lon", "grid_origin_lat",
            "reference_latitude_strategy", "baseline_prior_strength",
            "minimum_source_field_centroid_coverage",
            "minimum_source_crop_instance_week_centroid_coverage",
            "minimum_known_stage_coverage",
            "minimum_known_stage_coverage_per_supported_crop",
            "minimum_stage_coverage_crop_instance_weeks",
            "minimum_evaluable_fields", "minimum_active_fields",
            "minimum_coverage_ratio", "severe_override_min_fields",
            "severe_override_min_fresh_response_fields",
            "allow_severe_override", "frontier_distance_cells",
            "fdr_alpha", "minimum_link_score", "lineage_threshold",
            "minimum_lineage_jaccard", "same_hazard_link_required",
            "max_link_gap_weeks", "spatial_scale_km", "gap_penalty",
            "confirmation_weeks", "candidate_expiry_observed_weeks",
            "quiet_close_weeks", "recovery_observed_weeks",
            "recovery_grace_weeks", "severe_confirmation_min_fields",
            "severe_confirmation_min_fresh_response_fields",
            "minimum_crop_monitored_instances",
            "minimum_crop_evaluable_instances", "maximum_data_gap_weeks",
        )
    }
    starter_parameters["link_weights"] = dict(policy.link_weights)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generation": {
            "generation_id": run.get("generation_id"),
            "as_of_date": run.get("as_of_date"),
            "manifest_sha256": _sha256(source_manifest_path),
            "artifacts": {
                name: {"file": path.name, "size_bytes": path.stat().st_size}
                for name, path in paths.items()
            },
        },
        "policy": {
            "version": policy.version,
            "sha256": policy.source_sha256,
            "schema_version": policy.schema_version,
            "calibration_status": policy.calibration_status,
            "warning": policy.warning,
            "monitored_rule": policy.monitored_rule,
            "evaluable_rule": policy.evaluable_rule,
            "stage_buckets": list(policy.stage_buckets),
            "tracker_starter_parameters": starter_parameters,
        },
        "semantics": {
            "week_start": policy.week_start,
            "field_week_key": ["timeline_bucket", "field_id", "crop_instance_id"],
            "event_week_key": ["timeline_bucket", "event_id"],
            "stage_join": "latest_same_crop_instance_signal_on_or_before_snapshot_as_of_date",
            "stage_aliasing": "exact_controlled_alias_or_unknown_raw_preserved",
            "canonical_lane": "one_priority_rank_1_lane_per_field_hazard_week_all_lanes_retained",
            "claims": "monitoring signals only; no crop-loss, biological-outcome, diagnosis, or causation claim",
        },
        "counts": counts,
        "outputs": {
            "field_week_context": FIELD_WEEK_FILE,
            "event_week_lanes": EVENT_WEEK_FILE,
        },
        "artifact_sha256": {
            FIELD_WEEK_FILE: _sha256(stage / FIELD_WEEK_FILE),
            EVENT_WEEK_FILE: _sha256(stage / EVENT_WEEK_FILE),
        },
    }


def _validate_paths(generation_dir: Path, output_dir: Path, temp_dir: Path | None) -> None:
    if not generation_dir.is_dir():
        raise FileNotFoundError(f"Generation directory does not exist: {generation_dir}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable Incident V3 output already exists: {output_dir}")
    if output_dir == generation_dir or output_dir.is_relative_to(generation_dir):
        raise ValueError("Incident V3 output must not be inside the immutable generation")
    if generation_dir.is_relative_to(output_dir):
        raise ValueError("Incident V3 output must not contain the immutable generation")
    if temp_dir is not None:
        resolved = temp_dir.expanduser().resolve()
        if resolved == generation_dir or resolved.is_relative_to(generation_dir):
            raise ValueError("DuckDB temp directory must not be inside the immutable generation")
        if resolved == output_dir or resolved.is_relative_to(output_dir):
            raise ValueError("DuckDB temp directory must not be inside the immutable output")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EVENT_WEEK_FILE",
    "FIELD_WEEK_FILE",
    "MANIFEST_FILE",
    "SCHEMA_VERSION",
    "build_incident_context_v3",
]
