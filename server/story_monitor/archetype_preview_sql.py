"""Streaming SQL materialization for the Archetype V2 diagnostic map preview."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import duckdb


PREVIEW_COLUMNS = {
    "story_cluster_id", "motif_family", "motif_id", "archetype_id",
    "archetype_display_state", "anchor_date", "anchor_kind", "anchor_status",
    "eligible_for_training", "accepted", "split", "assignment_method",
    "assignment_reason", "candidate_archetype_id", "runner_up_archetype_id",
    "assignment_distance", "candidate_radius", "distance_ratio", "assignment_margin",
    "motif_model_version", "feature_schema_sha256",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def configured_connection(
    *, threads: int, memory_limit: str | None, temp_dir: Path | None
) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        connection.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {sql_string(str(temp_dir))}")
    return connection


def create_sources(
    connection: duckdb.DuckDBPyConnection,
    generation_dir: Path,
    model_dir: Path,
    evaluation_dir: Path,
) -> None:
    sources = {
        "source_frames": generation_dir / "map_frame_fields.parquet",
        "source_snapshots": generation_dir / "event_state_snapshots.parquet",
        "source_events": generation_dir / "event_windows.parquet",
        "source_memberships": generation_dir / "story_day_membership.parquet",
        "anchors": model_dir / "event_anchors.parquet",
        "catalog": model_dir / "archetype_catalog.parquet",
        "registry": evaluation_dir / "event_archetype_assignments.parquet",
    }
    for name, path in sources.items():
        connection.execute(
            f"CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet({sql_string(str(path))})"
        )
    connection.execute(
        """
        CREATE TEMP TABLE preview_ledger AS
        SELECT
            CAST(a.event_id AS VARCHAR) AS event_id,
            CAST(a.field_id AS VARCHAR) AS field_id,
            CAST(a.hazard_family AS VARCHAR) AS hazard_family,
            REGEXP_REPLACE(LOWER(COALESCE(NULLIF(TRIM(CAST(a.hazard_family AS VARCHAR)), ''),
                'other')), '[^a-z0-9]+', '_', 'g') AS hazard_slug,
            CAST(a.anchor_date AS DATE) AS anchor_date,
            CAST(a.anchor_kind AS VARCHAR) AS anchor_kind,
            CAST(a.anchor_outcome AS VARCHAR) AS anchor_status,
            COALESCE(CAST(a.eligible_for_training AS BOOLEAN), FALSE) AS eligible,
            CAST(r.accepted AS BOOLEAN) AS accepted,
            CAST(r.archetype_id AS VARCHAR) AS archetype_id,
            CAST(r.split AS VARCHAR) AS split,
            CAST(r.assignment_method AS VARCHAR) AS assignment_method,
            CAST(r.assignment_reason AS VARCHAR) AS assignment_reason,
            CAST(r.candidate_archetype_id AS VARCHAR) AS candidate_archetype_id,
            CAST(r.runner_up_archetype_id AS VARCHAR) AS runner_up_archetype_id,
            TRY_CAST(r.assignment_distance AS DOUBLE) AS assignment_distance,
            TRY_CAST(r.candidate_radius AS DOUBLE) AS candidate_radius,
            TRY_CAST(r.distance_ratio AS DOUBLE) AS distance_ratio,
            TRY_CAST(r.assignment_margin AS DOUBLE) AS assignment_margin,
            CAST(r.model_version AS VARCHAR) AS model_version,
            CAST(r.feature_schema_sha256 AS VARCHAR) AS feature_schema_sha256
        FROM anchors a
        LEFT JOIN registry r USING (event_id)
        """
    )


def validate_registry(
    connection: duckdb.DuckDBPyConnection,
    model_version: str,
    feature_schema_sha256: str,
    training_cutoff: str,
) -> None:
    total, distinct_events = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT event_id) FROM anchors"
    ).fetchone()
    if int(total) != int(distinct_events):
        raise ValueError("Archetype anchor ledger contains duplicate event IDs")
    registry_total, registry_distinct = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT event_id) FROM registry"
    ).fetchone()
    if int(registry_total) != int(registry_distinct):
        raise ValueError("Evaluation registry contains duplicate event IDs")
    mismatch = connection.execute(
        """
        SELECT COUNT(*)
        FROM anchors a FULL OUTER JOIN registry r USING (event_id)
        WHERE (a.event_id IS NULL)
           OR (COALESCE(a.eligible_for_training, FALSE) AND r.event_id IS NULL)
           OR (NOT COALESCE(a.eligible_for_training, FALSE) AND r.event_id IS NOT NULL)
        """
    ).fetchone()[0]
    if int(mismatch):
        raise ValueError("Evaluation assignments do not exactly match eligible event anchors")
    invalid = connection.execute(
        """
        SELECT COUNT(*)
        FROM registry r
        JOIN anchors a USING (event_id)
        LEFT JOIN catalog c ON r.archetype_id = c.archetype_id
        WHERE r.accepted IS NULL
           OR r.model_version IS DISTINCT FROM ?
           OR r.feature_schema_sha256 IS DISTINCT FROM ?
           OR CAST(r.hazard_family AS VARCHAR) IS DISTINCT FROM CAST(a.hazard_family AS VARCHAR)
           OR r.split IS DISTINCT FROM CASE
                WHEN CAST(a.anchor_date AS DATE) <= CAST(? AS DATE) THEN 'training'
                ELSE 'holdout' END
           OR (r.accepted AND (c.archetype_id IS NULL OR c.hazard_family IS DISTINCT FROM a.hazard_family))
           OR (NOT r.accepted AND r.archetype_id IS DISTINCT FROM 'novel_unassigned')
        """,
        [model_version, feature_schema_sha256, training_cutoff],
    ).fetchone()[0]
    if int(invalid):
        raise ValueError("Evaluation registry has inconsistent model, hazard, or archetype identity")
    duplicate_snapshots = connection.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT event_id, CAST(timeline_bucket AS DATE), COUNT(*) AS rows
          FROM source_snapshots GROUP BY event_id, CAST(timeline_bucket AS DATE)
          HAVING COUNT(*) <> 1
        )
        """
    ).fetchone()[0]
    if int(duplicate_snapshots):
        raise ValueError("Causal snapshots are not unique by event and timeline bucket")
    frame_count, matched_count = connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM source_frames),
          (SELECT COUNT(*) FROM source_frames f JOIN source_snapshots s
             ON f.event_id = s.event_id
            AND CAST(f.timeline_bucket AS DATE) = CAST(s.timeline_bucket AS DATE))
        """
    ).fetchone()
    if int(frame_count) != int(matched_count):
        raise ValueError("Map frames do not have exactly one matching causal snapshot")


def _base_columns(
    connection: duckdb.DuckDBPyConnection, view: str, alias: str
) -> str:
    columns = [row[0] for row in connection.execute(f"DESCRIBE {view}").fetchall()]
    kept = [name for name in columns if str(name) not in PREVIEW_COLUMNS]
    return ",\n            ".join(f'{alias}."{str(name).replace(chr(34), chr(34) * 2)}"' for name in kept)


def _display_parts(date_sql: str, as_of: str) -> dict[str, str]:
    decision = f"COALESCE(l.anchor_date, CAST({sql_string(as_of)} AS DATE))"
    anchor_ready = f"CAST({date_sql} AS DATE) >= {decision}"
    calibration = f"({anchor_ready}) AND l.eligible AND l.split = 'training'"
    revealed = f"({anchor_ready}) AND NOT ({calibration})"
    pending = "'diag:v2:' || l.hazard_slug || ':pending_anchor'"
    training = "'diag:v2:' || l.hazard_slug || ':calibration_training'"
    final = (
        "CASE WHEN NOT l.eligible THEN 'diag:v2:' || l.hazard_slug || ':' || "
        "REGEXP_REPLACE(LOWER(l.anchor_status), '[^a-z0-9]+', '_', 'g') "
        "WHEN l.accepted THEN l.archetype_id "
        "ELSE 'diag:v2:' || l.hazard_slug || ':novel_unassigned' END"
    )
    state = (
        f"CASE WHEN NOT ({anchor_ready}) THEN 'pending_anchor' "
        f"WHEN {calibration} THEN 'calibration_training' "
        "WHEN NOT l.eligible THEN l.anchor_status WHEN l.accepted THEN 'accepted' "
        "ELSE 'novel_unassigned' END"
    )
    display = (
        f"CASE WHEN NOT ({anchor_ready}) THEN {pending} "
        f"WHEN {calibration} THEN {training} ELSE {final} END"
    )
    return {
        "anchor_ready": anchor_ready, "calibration": calibration,
        "revealed": revealed, "display_id": display, "state": state,
    }


def _diagnostic_columns(parts: dict[str, str]) -> str:
    anchor_ready = parts["anchor_ready"]
    calibration = parts["calibration"]
    revealed = parts["revealed"]
    return f"""
            {parts['display_id']} AS story_cluster_id,
            {parts['display_id']} AS motif_id,
            l.hazard_family AS motif_family,
            {parts['state']} AS archetype_display_state,
            CASE WHEN {anchor_ready} THEN l.anchor_date END AS anchor_date,
            CASE WHEN {anchor_ready} THEN l.anchor_kind END AS anchor_kind,
            CASE WHEN {anchor_ready} THEN l.anchor_status ELSE 'pending_anchor' END AS anchor_status,
            CASE WHEN {anchor_ready} THEN l.eligible END AS eligible_for_training,
            CASE WHEN {revealed} AND l.eligible THEN l.accepted END AS accepted,
            CASE WHEN {revealed} AND l.eligible THEN l.archetype_id END AS archetype_id,
            CASE WHEN {revealed} AND l.eligible THEN l.split END AS split,
            CASE WHEN {revealed} AND l.eligible THEN l.assignment_method END AS assignment_method,
            CASE WHEN NOT ({anchor_ready}) THEN 'awaiting_causal_anchor'
                 WHEN {calibration} THEN 'training_calibration_assignment_masked'
                 WHEN NOT l.eligible THEN 'anchor_' || l.anchor_status
                 ELSE l.assignment_reason END AS assignment_reason,
            CASE WHEN {revealed} AND l.eligible THEN l.candidate_archetype_id END AS candidate_archetype_id,
            CASE WHEN {revealed} AND l.eligible THEN l.runner_up_archetype_id END AS runner_up_archetype_id,
            CASE WHEN {revealed} AND l.eligible THEN l.assignment_distance END AS assignment_distance,
            CASE WHEN {revealed} AND l.eligible THEN l.candidate_radius END AS candidate_radius,
            CASE WHEN {revealed} AND l.eligible THEN l.distance_ratio END AS distance_ratio,
            CASE WHEN {revealed} AND l.eligible THEN l.assignment_margin END AS assignment_margin,
            CASE WHEN {revealed} AND l.eligible THEN l.model_version END AS motif_model_version,
            CASE WHEN {revealed} AND l.eligible THEN l.feature_schema_sha256 END AS feature_schema_sha256
    """


def copy_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    connection.execute(
        f"COPY ({query}) TO {sql_string(str(path))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def materialize_preview(
    connection: duckdb.DuckDBPyConnection,
    stage: Path,
    *,
    as_of: str,
) -> dict[str, int]:
    frame_parts = _display_parts("s.snapshot_as_of_date", as_of)
    frame_query = f"""
        SELECT {_base_columns(connection, 'source_frames', 'f')},
               {_diagnostic_columns(frame_parts)}
        FROM source_frames f
        JOIN source_snapshots s ON f.event_id = s.event_id
          AND CAST(f.timeline_bucket AS DATE) = CAST(s.timeline_bucket AS DATE)
        JOIN preview_ledger l ON f.event_id = l.event_id
        ORDER BY f.timeline_bucket, f.field_id, f.event_id
    """
    copy_query(connection, frame_query, stage / "map_frame_fields.parquet")

    membership_parts = _display_parts("m.observation_date", as_of)
    membership_query = f"""
        SELECT {_base_columns(connection, 'source_memberships', 'm')},
               {_diagnostic_columns(membership_parts)}
        FROM source_memberships m JOIN preview_ledger l USING (event_id)
        ORDER BY m.field_id, m.observation_date, m.event_id
    """
    copy_query(connection, membership_query, stage / "story_day_membership.parquet")

    snapshot_parts = _display_parts("s.snapshot_as_of_date", as_of)
    snapshot_query = f"""
        SELECT {_base_columns(connection, 'source_snapshots', 's')},
               {_diagnostic_columns(snapshot_parts)}
        FROM source_snapshots s JOIN preview_ledger l USING (event_id)
        ORDER BY s.timeline_bucket, s.field_id, s.event_id
    """
    copy_query(connection, snapshot_query, stage / "event_state_snapshots.parquet")

    final_parts = _display_parts(f"CAST({sql_string(as_of)} AS DATE)", as_of)
    event_query = f"""
        SELECT {_base_columns(connection, 'source_events', 'e')},
               {_diagnostic_columns(final_parts)}
        FROM source_events e JOIN preview_ledger l USING (event_id)
        ORDER BY e.field_id, e.event_start_date, e.event_id
    """
    copy_query(connection, event_query, stage / "event_windows.parquet")

    labels_query = f"""
        SELECT
            f.story_cluster_id,
            'DIAGNOSTIC — ' || CASE
              WHEN BOOL_OR(f.archetype_display_state = 'accepted') THEN COALESCE(ANY_VALUE(c.label), 'Accepted archetype')
              WHEN BOOL_OR(f.archetype_display_state = 'novel_unassigned') THEN 'Novel / unassigned ' || REPLACE(ANY_VALUE(f.motif_family), '_', ' ')
              WHEN BOOL_OR(f.archetype_display_state = 'calibration_training') THEN 'Training calibration period · ' || REPLACE(ANY_VALUE(f.motif_family), '_', ' ')
              WHEN BOOL_OR(f.archetype_display_state = 'pending_anchor') THEN 'Pending causal anchor · ' || REPLACE(ANY_VALUE(f.motif_family), '_', ' ')
              ELSE REPLACE(ANY_VALUE(f.archetype_display_state), '_', ' ') || ' · ' || REPLACE(ANY_VALUE(f.motif_family), '_', ' ')
            END AS short_label,
            ARG_MAX(f.max_risk_band, COALESCE(f.max_risk_rank, 0)) AS max_risk_band,
            ANY_VALUE(f.hazard_signature) AS hazard_signature,
            ANY_VALUE(f.motif_family) AS motif_family,
            ARG_MAX(f.response_signature, COALESCE(f.max_risk_rank, 0)) AS response_signature,
            COUNT(DISTINCT f.event_id) AS event_count,
            COUNT(DISTINCT f.field_id) AS field_count,
            COUNT(DISTINCT f.crop_name) AS crop_count,
            CAST(NULL AS DOUBLE) AS median_window_span_days,
            MEDIAN(CAST(f.reportable_day_count AS DOUBLE)) AS median_reportable_days
        FROM read_parquet({sql_string(str(stage / 'map_frame_fields.parquet'))}) f
        LEFT JOIN catalog c ON f.story_cluster_id = c.archetype_id
        GROUP BY f.story_cluster_id
        ORDER BY f.story_cluster_id
    """
    copy_query(connection, labels_query, stage / "event_story_cluster_labels.parquet")

    names = ("map_frame_fields", "story_day_membership", "event_state_snapshots", "event_windows")
    source_names = ("source_frames", "source_memberships", "source_snapshots", "source_events")
    counts: dict[str, int] = {}
    for name, source in zip(names, source_names):
        output = stage / f"{name}.parquet"
        source_count = int(connection.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0])
        output_count = int(connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(output)]).fetchone()[0])
        if source_count != output_count:
            raise ValueError(f"Preview row-count mismatch for {name}: {output_count} != {source_count}")
        counts[name] = output_count
    counts["story_cluster_count"] = int(connection.execute(
        "SELECT COUNT(*) FROM read_parquet(?)", [str(stage / "event_story_cluster_labels.parquet")]
    ).fetchone()[0])
    counts["mapped_field_count"] = int(connection.execute(
        "SELECT COUNT(DISTINCT field_id) FROM read_parquet(?)",
        [str(stage / "map_frame_fields.parquet")],
    ).fetchone()[0])
    return counts


def copy_auxiliary(generation_dir: Path, stage: Path) -> None:
    geometry = next(
        (generation_dir / name for name in ("map_field_geometry.parquet", "field_geometry.parquet")
         if (generation_dir / name).is_file()),
        None,
    )
    if geometry is None:
        raise FileNotFoundError("Source generation has no field geometry parquet")
    shutil.copy2(geometry, stage / "map_field_geometry.parquet")
    crop_instances = generation_dir / "crop_instances.parquet"
    if crop_instances.is_file():
        shutil.copy2(crop_instances, stage / crop_instances.name)
