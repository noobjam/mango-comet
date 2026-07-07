"""Export the dual-clock Incident V4 map projection.

V4 is a viewer projection over two immutable inputs.  ``incident_dir`` owns the
weekly Incident V3 story spine; ``evidence_dir`` owns daily pressure, crop
context, and sparse Sentinel-2 acquisition ledgers.  The exporter never changes
either source and never turns a carried spectral value into an acquisition.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping, Sequence

import duckdb

from build_story_map_bundle import (
    DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
    DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
    advisory_build_lock,
)

from .incident_validation_v3 import file_sha256
from .incident_validation_v4 import validate_evidence_directory
from .incident_viewer_v3 import export_incident_viewer_v3


SCHEMA_VERSION = "crop-incident-viewer-v4/2"
MODE = "crop_incident_v4_dual_clock"
NATIVE_REPLAY_INCIDENT_SCHEMA_VERSION = "crop-impact-incident-story-replay-v4/1"
NATIVE_REPLAY_INCIDENT_MODE = "crop_incident_story_replay_v4"
PUBLICATION_STATUS = "diagnostic_uncalibrated_not_map_approved"
LIFECYCLE_RECONCILIATION_SCHEMA_VERSION = (
    "incident-lifecycle-evidence-reconciliation-v4/1"
)

EVIDENCE_FILES = {
    "crop": "crop_day_context_v4.parquet",
    "pressure": "field_day_pressure_v4.parquet",
    "s2": "field_s2_acquisition_v4.parquet",
}

OUTPUT_FILES = {
    "timeline": "daily_timeline_v4.parquet",
    "field_state": "field_day_state_v4.parquet",
    "pressure_observations": "pressure_observations_v4.parquet",
    "grid": "daily_field_grid_v4.parquet",
    "pressure_grid": "daily_pressure_grid_v4.parquet",
    "s2_attempts": "s2_attempts_v4.parquet",
    "s2_updates": "s2_updates_v4.parquet",
    "story_checkpoints": "story_checkpoints_v4.parquet",
    "lifecycle_reconciliation": "lifecycle_reconciliation_v4.parquet",
    "story_footprints": "story_footprints_v4.parquet",
}
REQUIRED_NONEMPTY_OUTPUTS = frozenset({"timeline", "field_state", "grid"})

_DATE_ALIASES = ("calendar_date", "observation_date", "pressure_date", "as_of_date")
_KNOWLEDGE_ALIASES = (
    "knowledge_time", "knowledge_date", "available_date", "observation_date",
)
_SOURCE_DATE_ALIASES = (
    "spectral_source_date", "acquisition_date", "evidence_date", "source_date",
)


def export_incident_viewer_v4(
    incident_dir: Path,
    evidence_dir: Path,
    source_generation_dir: Path,
    output_dir: Path,
    *,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
    min_valid_geometry_coverage: float = DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
    min_frame_geometry_coverage: float = DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
    display_grid_degrees: float = 0.05,
    freshness_fresh_days: int = 7,
    freshness_aging_days: int = 14,
    native_replay: bool = False,
) -> dict[str, Any]:
    """Build an immutable V4 bundle while retaining every V3 artifact/API."""
    incident_dir = incident_dir.expanduser().resolve()
    evidence_dir = evidence_dir.expanduser().resolve()
    source_generation_dir = source_generation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be positive")
    if not 0 < float(display_grid_degrees) <= 1:
        raise ValueError("display_grid_degrees must be in (0, 1]")
    if not 0 <= int(freshness_fresh_days) <= int(freshness_aging_days):
        raise ValueError("freshness thresholds are out of order")
    for name, directory in (
        ("incident_dir", incident_dir),
        ("evidence_dir", evidence_dir),
        ("source_generation_dir", source_generation_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} does not exist: {directory}")
    if native_replay:
        _require_native_replay_incident_manifest(incident_dir)
    evidence_validation = validate_evidence_directory(evidence_dir)
    source_manifest_path = source_generation_dir / "manifest.json"
    evidence_manifest_path = evidence_dir / "manifest.json"
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V4 viewer inputs require valid source/evidence manifests") from exc
    source_run = source_manifest.get("run") or {}
    evidence_run = evidence_manifest.get("run") or {}
    source_generation_id = str(source_run.get("generation_id") or "")
    if not source_generation_id or source_generation_id != str(
        evidence_run.get("source_generation_id") or ""
    ):
        raise ValueError("V4 evidence is not bound to the selected source generation")
    source_as_of = str(source_run.get("as_of_date") or "")[:10]
    evidence_as_of = str(evidence_run.get("release_as_of") or "")[:10]
    if not source_as_of or source_as_of != evidence_as_of:
        raise ValueError("V4 evidence/source release boundaries do not match")
    immutable_inputs = (incident_dir, evidence_dir, source_generation_dir)
    if any(
        output_dir == source or output_dir.is_relative_to(source)
        for source in immutable_inputs
    ):
        raise ValueError(
            "output_dir must be separate from and outside every immutable input"
        )
    evidence_paths = {name: evidence_dir / filename for name, filename in EVIDENCE_FILES.items()}
    missing = [str(path) for path in evidence_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V4 evidence directory is missing: " + ", ".join(missing))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with advisory_build_lock(output_dir):
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Immutable viewer output already exists: {output_dir}")
        with TemporaryDirectory(prefix=".incident-viewer-v4-", dir=output_dir.parent) as temporary:
            transaction = Path(temporary)
            v3_base = transaction / "v3-base"
            export_incident_viewer_v3(
                incident_dir,
                source_generation_dir,
                v3_base,
                threads=threads,
                memory_limit=memory_limit,
                temp_dir=temp_dir,
                min_valid_geometry_coverage=min_valid_geometry_coverage,
                min_frame_geometry_coverage=min_frame_geometry_coverage,
            )
            stage = transaction / output_dir.name
            os.replace(v3_base, stage)
            with _connection(threads, memory_limit, temp_dir) as connection:
                metadata = _register_canonical_inputs(connection, evidence_paths, stage)
                _write_pressure_observations(
                    connection, stage / OUTPUT_FILES["pressure_observations"]
                )
                _write_s2_attempts(connection, stage / OUTPUT_FILES["s2_attempts"])
                connection.read_parquet(
                    str(stage / OUTPUT_FILES["s2_attempts"])
                ).create_view("s2_attempts_v4")
                _write_s2_updates(connection, stage / OUTPUT_FILES["s2_updates"])
                connection.read_parquet(
                    str(stage / OUTPUT_FILES["s2_updates"])
                ).create_view("s2_updates_v4")
                _write_field_day_state(
                    connection,
                    stage / OUTPUT_FILES["field_state"],
                    freshness_fresh_days=int(freshness_fresh_days),
                    freshness_aging_days=int(freshness_aging_days),
                )
                connection.read_parquet(
                    str(stage / OUTPUT_FILES["field_state"])
                ).create_view("field_day_state_v4")
                _write_story_checkpoints(
                    connection,
                    stage / OUTPUT_FILES["story_checkpoints"],
                    availability_mode=str(evidence_validation["availability_mode"]),
                    native_replay=native_replay,
                )
                connection.read_parquet(
                    str(stage / OUTPUT_FILES["story_checkpoints"])
                ).create_view("story_checkpoints_v4")
                _write_lifecycle_reconciliation(
                    connection,
                    stage / OUTPUT_FILES["lifecycle_reconciliation"],
                    availability_mode=str(evidence_validation["availability_mode"]),
                    native_replay=native_replay,
                )
                _write_story_footprints(
                    connection, stage / OUTPUT_FILES["story_footprints"]
                )
                _write_daily_grid(
                    connection,
                    stage / OUTPUT_FILES["grid"],
                    cell_degrees=float(display_grid_degrees),
                )
                connection.read_parquet(
                    str(stage / OUTPUT_FILES["grid"])
                ).create_view("daily_field_grid_v4")
                _write_daily_pressure_grid(
                    connection,
                    stage / OUTPUT_FILES["pressure_grid"],
                    cell_degrees=float(display_grid_degrees),
                )
                _write_daily_timeline(
                    connection, stage / OUTPUT_FILES["timeline"]
                )
                validation = _validate_outputs(
                    connection, stage, native_replay=native_replay
                )

            base_manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
            manifest = _manifest(
                stage=stage,
                incident_dir=incident_dir,
                evidence_dir=evidence_dir,
                source_generation_dir=source_generation_dir,
                base_manifest=base_manifest,
                input_metadata=metadata,
                validation=validation,
                evidence_validation=evidence_validation,
                display_grid_degrees=float(display_grid_degrees),
                freshness_fresh_days=int(freshness_fresh_days),
                freshness_aging_days=int(freshness_aging_days),
                native_replay=native_replay,
            )
            (stage / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        "native_replay": native_replay,
        "output_dir": str(output_dir),
        **validation,
    }


@contextmanager
def _connection(
    threads: int, memory_limit: str | None, temp_dir: Path | None
) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET preserve_insertion_order=false")
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved)])
        yield connection
    finally:
        connection.close()


def _columns(connection: duckdb.DuckDBPyConnection, path: Path) -> frozenset[str]:
    description = connection.execute(
        "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
    ).description or []
    return frozenset(str(item[0]) for item in description)


def _name(columns: Sequence[str], aliases: Sequence[str], *, label: str) -> str:
    for alias in aliases:
        if alias in columns:
            return alias
    raise ValueError(f"{label} requires one of: {', '.join(aliases)}")


def _optional(columns: Sequence[str], aliases: Sequence[str], fallback: str) -> str:
    for alias in aliases:
        if alias in columns:
            return f'"{alias}"'
    return fallback


def _require_native_replay_incident_manifest(incident_dir: Path) -> None:
    manifest_path = incident_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Native replay requires a valid V4-native incident manifest"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Native replay incident manifest must be a JSON object")
    if (
        manifest.get("mode") != NATIVE_REPLAY_INCIDENT_MODE
        or manifest.get("schema_version")
        != NATIVE_REPLAY_INCIDENT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Native replay requires incident manifest mode "
            f"{NATIVE_REPLAY_INCIDENT_MODE!r} and schema_version "
            f"{NATIVE_REPLAY_INCIDENT_SCHEMA_VERSION!r}"
        )


def _register_canonical_inputs(
    connection: duckdb.DuckDBPyConnection,
    paths: Mapping[str, Path],
    stage: Path,
) -> dict[str, Any]:
    pressure_columns = _columns(connection, paths["pressure"])
    crop_columns = _columns(connection, paths["crop"])
    s2_columns = _columns(connection, paths["s2"])
    geometry_columns = _columns(connection, stage / "field_geometry.parquet")
    weekly_columns = _columns(connection, stage / "incident_weekly_state.parquet")
    footprint_columns = _columns(connection, stage / "incident_footprints.parquet")
    membership_columns = _columns(connection, stage / "incident_membership.parquet")
    connection.read_parquet(str(paths["pressure"])).create_view("pressure_source_v4_raw")
    connection.read_parquet(str(paths["crop"])).create_view("crop_source_v4_raw")
    connection.read_parquet(str(paths["s2"])).create_view("s2_source_v4_raw")

    pressure_date = _name(pressure_columns, _DATE_ALIASES, label="field_day_pressure_v4")
    pressure_field = _name(pressure_columns, ("field_id",), label="field_day_pressure_v4")
    pressure_instance = _name(
        pressure_columns, ("crop_instance_id",), label="field_day_pressure_v4"
    )
    crop_date = _name(crop_columns, _DATE_ALIASES, label="crop_day_context_v4")
    crop_field = _name(crop_columns, ("field_id",), label="crop_day_context_v4")
    crop_instance = _name(
        crop_columns, ("crop_instance_id",), label="crop_day_context_v4"
    )
    s2_field = _name(s2_columns, ("field_id",), label="field_s2_acquisition_v4")
    s2_instance = _name(
        s2_columns, ("crop_instance_id",), label="field_s2_acquisition_v4"
    )
    s2_knowledge = _name(s2_columns, _KNOWLEDGE_ALIASES, label="field_s2_acquisition_v4")
    s2_source = _name(s2_columns, _SOURCE_DATE_ALIASES, label="field_s2_acquisition_v4")
    for required in ("field_id", "centroid_lon", "centroid_lat"):
        if required not in geometry_columns:
            raise ValueError(f"field_geometry.parquet is missing {required}")
    if "knowledge_time" not in weekly_columns:
        raise ValueError("incident_weekly_state.parquet requires knowledge_time for V4")

    pressure_observed = _optional(
        pressure_columns, ("pressure_observed",), "TRUE"
    )
    pressure_active = _optional(
        pressure_columns, ("pressure_active",), pressure_observed
    )
    risk_rank = _optional(
        pressure_columns,
        (
            "pressure_rank", "risk_rank", "daily_pressure_rank",
            "current_risk_rank", "max_risk_rank",
        ),
        "0",
    )
    risk_band = _optional(
        pressure_columns,
        ("pressure_band", "risk_band", "current_risk_band", "max_risk_band"),
        "'NONE'",
    )
    pressure_score = _optional(
        pressure_columns, ("pressure_score",), risk_rank
    )
    pressure_knowledge = _optional(
        pressure_columns, ("knowledge_time",), f'"{pressure_date}"'
    )
    weather_available = _optional(
        pressure_columns, ("weather_available_at",), pressure_knowledge
    )
    hazard = _optional(
        pressure_columns,
        ("hazard_family", "hazard_signature", "primary_risk_driver"),
        "'none'",
    )
    connection.execute(
        f"""
        CREATE VIEW pressure_v4 AS
        SELECT
            CAST("{pressure_date}" AS DATE) AS calendar_date,
            CAST("{pressure_date}" AS DATE) AS pressure_observation_date,
            CAST({pressure_knowledge} AS TIMESTAMP) AS pressure_knowledge_time,
            CAST({weather_available} AS TIMESTAMP) AS weather_available_at,
            TRIM(CAST("{pressure_field}" AS VARCHAR)) AS field_id,
            CAST("{pressure_instance}" AS VARCHAR) AS crop_instance_id,
            LOWER(COALESCE(CAST({hazard} AS VARCHAR), 'none')) AS hazard_family,
            COALESCE(TRY_CAST({risk_rank} AS INTEGER), 0) AS risk_rank,
            UPPER(COALESCE(CAST({risk_band} AS VARCHAR), 'NONE')) AS risk_band,
            TRY_CAST({pressure_score} AS DOUBLE) AS pressure_score,
            COALESCE(TRY_CAST({pressure_observed} AS BOOLEAN), FALSE)
                AS pressure_observed,
            COALESCE(TRY_CAST({pressure_active} AS BOOLEAN), FALSE)
                AS pressure_active
        FROM pressure_source_v4_raw
        WHERE TRY_CAST("{pressure_date}" AS DATE) IS NOT NULL
          AND NULLIF(TRIM(CAST("{pressure_field}" AS VARCHAR)), '') IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY pressure_observation_date, field_id, crop_instance_id,
                hazard_family, pressure_knowledge_time
            ORDER BY pressure_knowledge_time DESC NULLS LAST,
                risk_rank DESC, hazard_family, risk_band
        ) = 1
        """,
    )

    crop_name = _optional(crop_columns, ("crop_name", "crop_name_normalized"), "'unknown'")
    crop_season = _optional(crop_columns, ("crop_season",), "'unknown'")
    stage_expr = _optional(
        crop_columns, ("stage_bucket", "stage_family", "crop_stage"), "'unknown'"
    )
    stage_source = _optional(
        crop_columns,
        ("stage_source_date", "stage_effective_date"),
        f'"{crop_date}"',
    )
    monitored = _optional(crop_columns, ("monitored",), "TRUE")
    evaluable = _optional(crop_columns, ("evaluable",), "TRUE")
    crop_knowledge = _optional(
        crop_columns, ("knowledge_time",), f'"{crop_date}"'
    )
    crop_observed = _optional(
        crop_columns, ("crop_context_observed",), "TRUE"
    )
    connection.execute(
        f"""
        CREATE VIEW crop_events_v4 AS
        SELECT
            CAST("{crop_date}" AS DATE) AS crop_observation_date,
            CAST({crop_knowledge} AS TIMESTAMP) AS crop_knowledge_time,
            TRIM(CAST("{crop_field}" AS VARCHAR)) AS field_id,
            CAST("{crop_instance}" AS VARCHAR) AS crop_instance_id,
            COALESCE(NULLIF(CAST({crop_name} AS VARCHAR), ''), 'unknown') AS crop_name,
            COALESCE(NULLIF(CAST({crop_season} AS VARCHAR), ''), 'unknown') AS crop_season,
            LOWER(COALESCE(NULLIF(CAST({stage_expr} AS VARCHAR), ''), 'unknown')) AS stage_bucket,
            CAST({stage_source} AS DATE) AS stage_effective_date,
            COALESCE(TRY_CAST({monitored} AS BOOLEAN), TRUE) AS monitored,
            COALESCE(TRY_CAST({evaluable} AS BOOLEAN), FALSE) AS evaluable,
            COALESCE(TRY_CAST({crop_observed} AS BOOLEAN), FALSE)
                AS crop_context_observed
        FROM crop_source_v4_raw
        WHERE TRY_CAST("{crop_date}" AS DATE) IS NOT NULL
          AND NULLIF(TRIM(CAST("{crop_field}" AS VARCHAR)), '') IS NOT NULL
          AND NULLIF(TRIM(CAST("{crop_instance}" AS VARCHAR)), '') IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY crop_observation_date, field_id, crop_instance_id
            ORDER BY crop_knowledge_time DESC NULLS LAST,
                stage_effective_date DESC, crop_name, stage_bucket
        ) = 1
        """,
    )
    connection.execute(
        """
        CREATE VIEW crop_context_v4 AS
        WITH spine AS (
            SELECT crop_observation_date AS calendar_date,
                field_id, crop_instance_id
            FROM crop_events_v4
            GROUP BY ALL
        )
        SELECT
            d.calendar_date,
            known.crop_observation_date,
            known.crop_knowledge_time,
            d.field_id,
            d.crop_instance_id,
            COALESCE(known.crop_name, 'unknown') AS crop_name,
            COALESCE(known.crop_season, 'unknown') AS crop_season,
            COALESCE(known.stage_bucket, 'unknown') AS stage_bucket,
            known.stage_effective_date AS stage_source_date,
            COALESCE(known.monitored, TRUE) AS monitored,
            COALESCE(known.evaluable, FALSE) AS evaluable,
            COALESCE(known.crop_context_observed, FALSE) AS crop_context_observed
        FROM spine d
        LEFT JOIN LATERAL (
            SELECT e.*
            FROM crop_events_v4 e
            WHERE e.field_id = d.field_id
              AND e.crop_instance_id = d.crop_instance_id
              AND e.stage_effective_date <= d.calendar_date
              AND e.crop_knowledge_time
                    < CAST(d.calendar_date AS TIMESTAMP) + INTERVAL 1 DAY
            ORDER BY e.stage_effective_date DESC,
                e.crop_knowledge_time DESC,
                e.crop_observation_date DESC,
                e.crop_name,
                e.stage_bucket
            LIMIT 1
        ) known ON TRUE
        """
    )

    s2_crop_name = _optional(s2_columns, ("crop_name", "crop_name_normalized"), "NULL")
    s2_stage = _optional(s2_columns, ("stage_bucket", "stage_family", "crop_stage"), "NULL")
    echo_days = _optional(s2_columns, ("spectral_echo_days", "echo_days"), "NULL")
    freshness = _optional(s2_columns, ("spectral_freshness", "freshness"), "NULL")
    reference_date = _optional(s2_columns, ("reference_source_date",), "NULL")
    response_class = _optional(
        s2_columns, ("response_class", "daily_response_class"), "'insufficient_reference'"
    )
    new_response = _optional(
        s2_columns, ("new_response_evidence",), "FALSE"
    )
    acquisition_attempted = _optional(
        s2_columns, ("acquisition_attempted",), "TRUE"
    )
    spectral_usable = _optional(
        s2_columns,
        ("spectral_usable", "acquisition_usable"),
        "TRUE",
    )
    new_acquisition = _optional(
        s2_columns,
        ("is_new_acquisition",),
        f"({acquisition_attempted} AND {spectral_usable})",
    )
    acquisition_id = _optional(s2_columns, ("acquisition_id",), "NULL")
    acquisition_status = _optional(
        s2_columns, ("acquisition_status",), "NULL"
    )
    quality_status = _optional(
        s2_columns,
        ("s2_field_quality_flag", "qa_status", "qa_reason", "qa"),
        "NULL",
    )
    rejection_reason = _optional(
        s2_columns,
        ("rejection_reason", "spectral_rejection_reason"),
        acquisition_status,
    )
    connection.execute(
        f"""
        CREATE VIEW s2_source_v4 AS
        SELECT
            TRIM(CAST("{s2_field}" AS VARCHAR)) AS field_id,
            CAST("{s2_instance}" AS VARCHAR) AS crop_instance_id,
            CAST({acquisition_id} AS VARCHAR) AS acquisition_id,
            CAST("{s2_knowledge}" AS TIMESTAMP) AS knowledge_time,
            CAST("{s2_knowledge}" AS DATE) AS knowledge_date,
            CAST("{s2_source}" AS DATE) AS spectral_source_date,
            COALESCE(TRY_CAST({new_acquisition} AS BOOLEAN), FALSE)
                AS is_new_acquisition,
            COALESCE(TRY_CAST({acquisition_attempted} AS BOOLEAN), FALSE)
                AS acquisition_attempted,
            COALESCE(TRY_CAST({spectral_usable} AS BOOLEAN), FALSE)
                AS spectral_usable,
            CAST({rejection_reason} AS VARCHAR) AS rejection_reason,
            CAST({acquisition_status} AS VARCHAR) AS acquisition_status,
            CAST({quality_status} AS VARCHAR) AS quality_status,
            TRY_CAST({echo_days} AS INTEGER) AS spectral_echo_days,
            LOWER(CAST({freshness} AS VARCHAR)) AS source_freshness,
            CAST({reference_date} AS DATE) AS reference_source_date,
            LOWER(COALESCE(CAST({response_class} AS VARCHAR), 'insufficient_reference'))
                AS response_class,
            COALESCE(TRY_CAST({new_response} AS BOOLEAN), FALSE)
                AS new_response_evidence,
            CAST({s2_crop_name} AS VARCHAR) AS crop_name,
            LOWER(CAST({s2_stage} AS VARCHAR)) AS stage_bucket
        FROM s2_source_v4_raw
        WHERE TRY_CAST("{s2_knowledge}" AS TIMESTAMP) IS NOT NULL
          AND NULLIF(TRIM(CAST("{s2_field}" AS VARCHAR)), '') IS NOT NULL
          AND NULLIF(TRIM(CAST("{s2_instance}" AS VARCHAR)), '') IS NOT NULL
        """,
    )
    connection.read_parquet(str(stage / "field_geometry.parquet")).create_view("geometry_v4")
    connection.read_parquet(str(stage / "incident_weekly_state.parquet")).create_view(
        "incident_weekly_v4"
    )
    connection.read_parquet(str(stage / "incident_footprints.parquet")).create_view(
        "incident_footprints_v4_source"
    )
    connection.read_parquet(str(stage / "incident_membership.parquet")).create_view(
        "incident_membership_v4"
    )
    return {
        "pressure_columns": sorted(pressure_columns),
        "crop_columns": sorted(crop_columns),
        "s2_columns": sorted(s2_columns),
        "footprint_columns": sorted(footprint_columns),
        "membership_columns": sorted(membership_columns),
    }


def _write_pressure_observations(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    """Persist only the sparse late-known supplement to the normal daily state."""
    connection.execute(
        """
        COPY (
            SELECT *, calendar_date AS pressure_effective_date
            FROM pressure_v4
            WHERE CAST(pressure_knowledge_time AS DATE) > calendar_date
            ORDER BY field_id, crop_instance_id, pressure_effective_date,
                pressure_knowledge_time, hazard_family
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_s2_attempts(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    connection.execute(
        """
        COPY (
            SELECT *,
                CASE
                    WHEN is_new_acquisition AND spectral_usable
                         AND spectral_source_date IS NOT NULL THEN 'acquisition'
                    WHEN acquisition_attempted AND NOT spectral_usable THEN 'rejected'
                    ELSE 'echo'
                END AS marker_type
            FROM s2_source_v4
            WHERE (is_new_acquisition AND spectral_usable
                   AND spectral_source_date IS NOT NULL)
               OR (acquisition_attempted AND NOT spectral_usable)
            ORDER BY knowledge_time, field_id, crop_instance_id,
                spectral_source_date NULLS LAST, acquisition_id NULLS LAST
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_s2_updates(connection: duckdb.DuckDBPyConnection, output_path: Path) -> None:
    duplicates = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT field_id, crop_instance_id, spectral_source_date,
                    COUNT(*) AS n
                FROM s2_attempts_v4
                WHERE marker_type = 'acquisition'
                GROUP BY 1, 2, 3 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    if duplicates:
        raise ValueError("field_s2_acquisition_v4 contains duplicate new acquisitions")
    invalid = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM s2_attempts_v4
            WHERE marker_type = 'acquisition'
              AND CAST(spectral_source_date AS TIMESTAMP) > knowledge_time
            """
        ).fetchone()[0]
    )
    if invalid:
        raise ValueError("S2 source date cannot follow its knowledge date")
    connection.execute(
        """
        COPY (
            SELECT * EXCLUDE (marker_type) FROM s2_attempts_v4
            WHERE marker_type = 'acquisition'
            ORDER BY knowledge_time, field_id, crop_instance_id,
                spectral_source_date, acquisition_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_field_day_state(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    freshness_fresh_days: int,
    freshness_aging_days: int,
) -> None:
    connection.execute(
        """
        COPY (
            WITH base AS (
                SELECT
                    c.*,
                    COALESCE(p.hazard_family, 'none') AS hazard_family,
                    COALESCE(p.risk_rank, 0) AS risk_rank,
                    COALESCE(p.risk_band, 'NONE') AS risk_band,
                    p.pressure_score,
                    COALESCE(p.pressure_observed, FALSE) AS pressure_observed,
                    COALESCE(p.pressure_active, FALSE) AS pressure_active,
                    p.pressure_observation_date,
                    p.pressure_knowledge_time,
                    p.weather_available_at
                FROM crop_context_v4 c
                LEFT JOIN pressure_v4 p
                  ON p.calendar_date = c.calendar_date
                 AND p.field_id = c.field_id
                 AND p.crop_instance_id = c.crop_instance_id
                 AND p.pressure_knowledge_time
                        < CAST(c.calendar_date AS TIMESTAMP) + INTERVAL 1 DAY
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY c.calendar_date, c.field_id, c.crop_instance_id,
                        COALESCE(p.hazard_family, 'none')
                    ORDER BY p.pressure_knowledge_time DESC NULLS LAST,
                        p.risk_rank DESC
                ) = 1
            ), held AS (
                SELECT
                    b.*,
                    a.acquisition_id AS s2_acquisition_id,
                    a.knowledge_time AS s2_knowledge_time,
                    a.knowledge_date AS s2_knowledge_date,
                    a.spectral_source_date,
                    a.spectral_echo_days,
                    a.reference_source_date,
                    a.response_class,
                    a.new_response_evidence,
                    DATE_DIFF('day', a.spectral_source_date, b.calendar_date)
                        AS evidence_age_days,
                    CAST(a.knowledge_time AS DATE) = b.calendar_date AS new_s2_today
                FROM base b
                LEFT JOIN LATERAL (
                    SELECT candidate.*
                    FROM s2_updates_v4 candidate
                    WHERE candidate.field_id = b.field_id
                      AND candidate.crop_instance_id = b.crop_instance_id
                      AND candidate.knowledge_time
                            < CAST(b.calendar_date AS TIMESTAMP) + INTERVAL 1 DAY
                    ORDER BY candidate.knowledge_time DESC,
                        candidate.spectral_source_date DESC,
                        candidate.acquisition_id DESC NULLS LAST
                    LIMIT 1
                ) a ON TRUE
            )
            SELECT *,
                CASE
                    WHEN spectral_source_date IS NULL THEN 'missing'
                    WHEN evidence_age_days <= ? THEN 'fresh'
                    WHEN evidence_age_days <= ? THEN 'aging'
                    ELSE 'stale'
                END AS evidence_freshness,
                COALESCE(
                    new_response_evidence
                    AND response_class IN ('medium_decline', 'severe_decline', 'recovery'),
                    FALSE
                ) AS crop_impact_active
            FROM held
            ORDER BY calendar_date, field_id, crop_instance_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path), freshness_fresh_days, freshness_aging_days],
    )


def _write_story_checkpoints(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    availability_mode: str,
    native_replay: bool = False,
) -> None:
    if availability_mode not in {"strict", "reconstructed"}:
        raise ValueError("Story checkpoint availability mode is invalid")
    weekly_columns = {
        str(row[0]) for row in connection.execute("DESCRIBE incident_weekly_v4").fetchall()
    }
    membership_columns = {
        str(row[0])
        for row in connection.execute("DESCRIBE incident_membership_v4").fetchall()
    }
    required_membership = {
        "incident_id", "timeline_bucket", "field_id", "crop_instance_id",
        "hazard_family", "membership_role", "event_state", "response_class",
        "fresh_response_evidence", "knowledge_time",
    }
    missing_membership = sorted(required_membership - membership_columns)
    if missing_membership:
        message = (
            "Incident membership cannot attribute checkpoint evidence: "
            + ", ".join(missing_membership)
        )
        if availability_mode == "strict":
            raise ValueError(message)
        raise ValueError(message)
    inferred_sql = (
        "COALESCE(TRY_CAST(w.knowledge_time_inferred AS BOOLEAN), TRUE)"
        if "knowledge_time_inferred" in weekly_columns else "TRUE"
    )
    pressure_decision_week = (
        "CAST(p.pressure_knowledge_time AS DATE) BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
        if native_replay
        else "p.pressure_observation_date BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
    )
    crop_decision_week = (
        "CAST(c.crop_knowledge_time AS DATE) BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
        if native_replay
        else "c.stage_effective_date BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
    )
    s2_decision_week = (
        "a.knowledge_date BETWEEN m.story_week AND m.story_week + INTERVAL 6 DAY"
        if native_replay
        else "a.spectral_source_date BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
    )
    invalid = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM incident_weekly_v4
            WHERE TRY_CAST(knowledge_time AS DATE) IS NULL
               OR CAST(knowledge_time AS DATE) < CAST(timeline_bucket AS DATE)
            """
        ).fetchone()[0]
    )
    if invalid:
        raise ValueError("Incident weekly states contain invalid knowledge_time")
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW story_checkpoint_bounds_v4 AS
        WITH checkpoints AS (
            SELECT w.* EXCLUDE (knowledge_time),
                CAST(w.timeline_bucket AS DATE) AS story_week,
                CAST(w.knowledge_time AS TIMESTAMP)
                    AS source_checkpoint_knowledge_time,
                {inferred_sql} AS source_checkpoint_knowledge_time_inferred
            FROM incident_weekly_v4 AS w
        ), members AS (
            SELECT DISTINCT
                NULLIF(TRIM(CAST(m.incident_id AS VARCHAR)), '') AS incident_id,
                CAST(m.timeline_bucket AS DATE) AS story_week,
                NULLIF(TRIM(CAST(m.field_id AS VARCHAR)), '') AS field_id,
                NULLIF(TRIM(CAST(m.crop_instance_id AS VARCHAR)), '')
                    AS crop_instance_id,
                NULLIF(LOWER(TRIM(CAST(m.hazard_family AS VARCHAR))), '')
                    AS hazard_family,
                LOWER(COALESCE(CAST(m.membership_role AS VARCHAR), ''))
                    AS membership_role,
                UPPER(COALESCE(CAST(m.event_state AS VARCHAR), '')) AS event_state,
                LOWER(COALESCE(CAST(m.response_class AS VARCHAR), ''))
                    AS response_class,
                COALESCE(TRY_CAST(m.fresh_response_evidence AS BOOLEAN), FALSE)
                    AS fresh_response_evidence,
                TRY_CAST(m.knowledge_time AS TIMESTAMP) AS membership_knowledge_time
            FROM incident_membership_v4 AS m
        ), member_evidence AS (
            SELECT m.*,
                m.membership_role IN ('pressure_core', 'watch_frontier')
                    OR m.event_state IN ('SEVERE', 'ACTIVE', 'WATCH')
                    AS pressure_attribution_required,
                m.membership_role = 'impact_lag'
                    OR (
                        m.fresh_response_evidence
                        AND m.response_class IN (
                            'medium_decline', 'severe_decline', 'recovery'
                        )
                    ) AS s2_attribution_required,
                (
                    SELECT MAX(p.pressure_knowledge_time)
                    FROM pressure_v4 AS p
                    WHERE p.field_id = m.field_id
                      AND p.crop_instance_id = m.crop_instance_id
                      AND p.hazard_family = m.hazard_family
                      AND {pressure_decision_week}
                ) AS pressure_knowledge_time,
                (
                    SELECT MAX(p.pressure_knowledge_time)
                    FROM pressure_v4 AS p
                    WHERE p.field_id = m.field_id
                      AND p.crop_instance_id = m.crop_instance_id
                      AND p.hazard_family = m.hazard_family
                      AND {pressure_decision_week}
                      AND p.pressure_observed
                ) AS observed_pressure_knowledge_time,
                (
                    SELECT MAX(c.crop_knowledge_time)
                    FROM crop_events_v4 AS c
                    WHERE c.field_id = m.field_id
                      AND c.crop_instance_id = m.crop_instance_id
                      AND {crop_decision_week}
                ) AS crop_context_knowledge_time,
                (
                    SELECT MAX(a.knowledge_time)
                    FROM s2_updates_v4 AS a
                    WHERE a.field_id = m.field_id
                      AND a.crop_instance_id = m.crop_instance_id
                      AND {s2_decision_week}
                      AND a.new_response_evidence
                      AND a.response_class IN (
                          'medium_decline', 'severe_decline', 'recovery'
                      )
                ) AS s2_response_knowledge_time
            FROM members AS m
        ), bounds AS (
            SELECT incident_id, story_week,
                COUNT(*)::BIGINT AS attributed_membership_count,
                COUNT_IF(
                    field_id IS NULL OR crop_instance_id IS NULL
                    OR hazard_family IS NULL OR membership_knowledge_time IS NULL
                )::BIGINT AS invalid_membership_attribution_count,
                COUNT_IF(crop_context_knowledge_time IS NULL)::BIGINT
                    AS missing_crop_attribution_count,
                COUNT_IF(
                    pressure_attribution_required
                    AND observed_pressure_knowledge_time IS NULL
                )::BIGINT AS missing_pressure_attribution_count,
                COUNT_IF(
                    s2_attribution_required AND s2_response_knowledge_time IS NULL
                )::BIGINT AS missing_s2_attribution_count,
                MAX(membership_knowledge_time) AS membership_knowledge_time,
                MAX(pressure_knowledge_time) AS pressure_knowledge_time,
                MAX(crop_context_knowledge_time) AS crop_context_knowledge_time,
                MAX(s2_response_knowledge_time) AS s2_response_knowledge_time
            FROM member_evidence
            GROUP BY incident_id, story_week
        ), joined AS (
            SELECT c.*, 
                COALESCE(b.attributed_membership_count, 0)
                    AS attributed_membership_count,
                COALESCE(b.invalid_membership_attribution_count, 0)
                    AS invalid_membership_attribution_count,
                COALESCE(b.missing_crop_attribution_count, 0)
                    AS missing_crop_attribution_count,
                COALESCE(b.missing_pressure_attribution_count, 0)
                    AS missing_pressure_attribution_count,
                COALESCE(b.missing_s2_attribution_count, 0)
                    AS missing_s2_attribution_count,
                b.membership_knowledge_time,
                b.pressure_knowledge_time,
                b.crop_context_knowledge_time,
                b.s2_response_knowledge_time,
                GREATEST(
                    b.membership_knowledge_time,
                    b.pressure_knowledge_time,
                    b.crop_context_knowledge_time,
                    b.s2_response_knowledge_time
                ) AS contributing_evidence_knowledge_time,
                GREATEST(
                    COALESCE(TRY_CAST(c.pressure_core_field_count AS BIGINT), 0)
                    + COALESCE(TRY_CAST(c.watch_frontier_field_count AS BIGINT), 0)
                    + COALESCE(TRY_CAST(c.impact_lag_field_count AS BIGINT), 0),
                    COALESCE(TRY_CAST(c.affected_count AS BIGINT), 0)
                ) AS required_membership_count
            FROM checkpoints AS c
            LEFT JOIN bounds AS b
              ON b.incident_id = CAST(c.incident_id AS VARCHAR)
             AND b.story_week = c.story_week
        ), bounded AS (
            SELECT *,
                GREATEST(
                    source_checkpoint_knowledge_time,
                    contributing_evidence_knowledge_time
                ) AS story_known_time,
                attributed_membership_count >= required_membership_count
                    AND invalid_membership_attribution_count = 0
                    AND missing_crop_attribution_count = 0
                    AND missing_pressure_attribution_count = 0
                    AND missing_s2_attribution_count = 0
                    AS knowledge_bound_complete
            FROM joined
        )
        SELECT *,
            story_known_time AS knowledge_time,
            CAST(story_known_time AS DATE) AS story_known_date,
            story_known_time > source_checkpoint_knowledge_time
                AS story_known_time_raised,
            CAST('{availability_mode}' AS VARCHAR) AS checkpoint_bound_mode
        FROM bounded
        """
    )
    if availability_mode == "strict":
        inferred, under_bound, incomplete = connection.execute(
            """
            SELECT
                COUNT_IF(source_checkpoint_knowledge_time_inferred),
                COUNT_IF(story_known_time_raised),
                COUNT_IF(NOT knowledge_bound_complete)
            FROM story_checkpoint_bounds_v4
            """
        ).fetchone()
        if int(inferred or 0):
            raise ValueError("Strict story checkpoints may not use inferred knowledge time")
        if int(under_bound or 0):
            raise ValueError(
                "Strict story checkpoint knowledge time is below contributing evidence"
            )
        if int(incomplete or 0):
            raise ValueError(
                "Strict story checkpoint evidence attribution is incomplete"
            )
    connection.execute(
        """
        COPY (
            SELECT * FROM story_checkpoint_bounds_v4
            ORDER BY story_known_time, incident_id, story_week
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_lifecycle_reconciliation(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    availability_mode: str,
    native_replay: bool = False,
) -> None:
    """Fail closed when positive lifecycle claims contradict V4 evidence.

    Default projection mode preserves the V3 lifecycle and proves only that its
    positive pressure/response claims reconcile.  Native mode consumes the
    lifecycle already replayed from V4 ledgers and records that stronger source
    contract while retaining the same contradiction and counter gates.
    """
    if availability_mode not in {"strict", "reconstructed"}:
        raise ValueError("Lifecycle reconciliation availability mode is invalid")
    weekly_columns = {
        str(row[0]) for row in connection.execute("DESCRIBE story_checkpoints_v4").fetchall()
    }
    required_weekly = {
        "incident_id", "story_week", "story_known_time", "incident_state",
        "pressure_core_field_count", "impact_lag_field_count",
        "fresh_decline_field_count", "fresh_recovery_field_count",
        "knowledge_bound_complete",
    }
    missing_weekly = sorted(required_weekly - weekly_columns)
    if missing_weekly:
        raise ValueError(
            "Incident V4 lifecycle reconciliation requires checkpoint columns: "
            + ", ".join(missing_weekly)
        )
    pressure_decision_week = (
        "CAST(p.pressure_knowledge_time AS DATE) BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
        if native_replay
        else "p.pressure_observation_date BETWEEN m.story_week "
        "AND m.story_week + INTERVAL 6 DAY"
    )
    s2_decision_week = (
        "a.knowledge_date BETWEEN p.story_week AND p.story_week + INTERVAL 6 DAY"
        if native_replay
        else "(a.spectral_source_date BETWEEN p.story_week "
        "AND p.story_week + INTERVAL 6 DAY OR (a.knowledge_date BETWEEN "
        "p.story_week AND p.story_week + INTERVAL 6 DAY "
        "AND a.spectral_source_date < p.story_week))"
    )
    source_state_preserved = "FALSE" if native_replay else "TRUE"
    native_lifecycle_replayed = "TRUE" if native_replay else "FALSE"
    active_quiet_status = (
        "active_field_pressure_present_after_component_absence_replay"
        if native_replay
        else "active_field_pressure_component_absence_not_replayed"
    )
    reconciliation_scope = (
        "V4-native lifecycle, component absence, coverage registries, and policy "
        "replayed upstream; positive claims and membership counters reconciled"
        if native_replay
        else "positive V3 pressure/response claims only; component absence, "
        "coverage thresholds, registries, and lifecycle policy are not replayed"
    )

    connection.execute(
        f"""
        COPY (
            WITH checkpoints AS (
                SELECT
                    CAST(incident_id AS VARCHAR) AS incident_id,
                    CAST(story_week AS DATE) AS story_week,
                    CAST(story_known_time AS TIMESTAMP) AS story_known_time,
                    UPPER(CAST(incident_state AS VARCHAR)) AS source_incident_state,
                    COALESCE(TRY_CAST(pressure_core_field_count AS BIGINT), 0)
                        AS source_pressure_core_field_count,
                    COALESCE(TRY_CAST(impact_lag_field_count AS BIGINT), 0)
                        AS source_impact_lag_field_count,
                    COALESCE(TRY_CAST(fresh_decline_field_count AS BIGINT), 0)
                        AS source_fresh_decline_field_count,
                    COALESCE(TRY_CAST(fresh_recovery_field_count AS BIGINT), 0)
                        AS source_fresh_recovery_field_count,
                    COALESCE(TRY_CAST(knowledge_bound_complete AS BOOLEAN), FALSE)
                        AS knowledge_bound_complete
                FROM story_checkpoints_v4
            ), members AS (
                SELECT DISTINCT
                    c.incident_id, c.story_week, c.story_known_time,
                    NULLIF(TRIM(CAST(m.field_id AS VARCHAR)), '') AS field_id,
                    NULLIF(TRIM(CAST(m.crop_instance_id AS VARCHAR)), '')
                        AS crop_instance_id,
                    NULLIF(LOWER(TRIM(CAST(m.hazard_family AS VARCHAR))), '')
                        AS hazard_family,
                    LOWER(COALESCE(CAST(m.membership_role AS VARCHAR), ''))
                        AS membership_role,
                    LOWER(COALESCE(CAST(m.response_class AS VARCHAR), ''))
                        AS response_class,
                    COALESCE(TRY_CAST(m.fresh_response_evidence AS BOOLEAN), FALSE)
                        AS fresh_response_evidence
                FROM incident_membership_v4 AS m
                JOIN checkpoints AS c
                  ON CAST(m.incident_id AS VARCHAR) = c.incident_id
                 AND CAST(m.timeline_bucket AS DATE) = c.story_week
            ), pressure_support AS (
                SELECT m.*,
                    COUNT(DISTINCT p.pressure_observation_date) FILTER (
                        WHERE p.pressure_observed
                    )::BIGINT AS observed_pressure_day_count,
                    COUNT(DISTINCT p.pressure_observation_date) FILTER (
                        WHERE p.pressure_observed AND p.pressure_active
                    )::BIGINT AS active_pressure_day_count
                FROM members AS m
                LEFT JOIN pressure_v4 AS p
                  ON p.field_id = m.field_id
                 AND p.crop_instance_id = m.crop_instance_id
                 AND p.hazard_family = m.hazard_family
                 AND {pressure_decision_week}
                 AND p.pressure_knowledge_time <= m.story_known_time
                GROUP BY ALL
            ), member_support AS (
                SELECT p.*,
                    COUNT(a.acquisition_id) FILTER (
                        WHERE a.new_response_evidence
                          AND (
                              (
                                  a.response_class IN (
                                      'medium_decline', 'severe_decline'
                                  )
                                  AND p.response_class IN (
                                      'medium_decline', 'severe_decline'
                                  )
                              )
                              OR (
                                  a.response_class = 'recovery'
                                  AND p.response_class = 'recovery'
                              )
                          )
                    )::BIGINT AS matching_s2_response_count
                FROM pressure_support AS p
                LEFT JOIN s2_updates_v4 AS a
                 ON a.field_id = p.field_id
                 AND a.crop_instance_id = p.crop_instance_id
                 AND a.knowledge_time <= p.story_known_time
                 AND {s2_decision_week}
                GROUP BY ALL
            ), membership_evidence AS (
                SELECT incident_id, story_week,
                    COUNT(DISTINCT field_id)::BIGINT AS attributed_member_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE membership_role = 'pressure_core'
                    )::BIGINT AS membership_pressure_core_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE membership_role = 'impact_lag'
                    )::BIGINT AS membership_impact_lag_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE fresh_response_evidence
                          AND response_class IN ('medium_decline', 'severe_decline')
                    )::BIGINT AS membership_fresh_decline_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE fresh_response_evidence AND response_class = 'recovery'
                    )::BIGINT AS membership_fresh_recovery_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE fresh_response_evidence
                          AND response_class NOT IN (
                              'medium_decline', 'severe_decline', 'recovery'
                          )
                    )::BIGINT AS unclassified_fresh_response_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE membership_role = 'pressure_core'
                          AND active_pressure_day_count > 0
                    )::BIGINT AS supported_pressure_core_field_count,
                    COUNT(*) FILTER (
                        WHERE membership_role = 'pressure_core'
                          AND active_pressure_day_count = 0
                    )::BIGINT AS unsupported_pressure_core_membership_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE membership_role = 'pressure_core'
                          AND observed_pressure_day_count = 0
                    )::BIGINT AS pressure_core_missing_weather_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE membership_role = 'pressure_core'
                          AND observed_pressure_day_count > 0
                          AND active_pressure_day_count = 0
                    )::BIGINT AS pressure_core_observed_inactive_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE fresh_response_evidence
                          AND response_class IN ('medium_decline', 'severe_decline')
                          AND matching_s2_response_count > 0
                    )::BIGINT AS supported_fresh_decline_field_count,
                    COUNT(*) FILTER (
                        WHERE fresh_response_evidence
                          AND response_class IN ('medium_decline', 'severe_decline')
                          AND matching_s2_response_count = 0
                    )::BIGINT AS unsupported_fresh_decline_membership_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE fresh_response_evidence AND response_class = 'recovery'
                          AND matching_s2_response_count > 0
                    )::BIGINT AS supported_fresh_recovery_field_count,
                    COUNT(*) FILTER (
                        WHERE fresh_response_evidence AND response_class = 'recovery'
                          AND matching_s2_response_count = 0
                    )::BIGINT AS unsupported_fresh_recovery_membership_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE observed_pressure_day_count = 0
                    )::BIGINT AS missing_weather_member_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE observed_pressure_day_count > 0
                          AND active_pressure_day_count = 0
                    )::BIGINT AS observed_quiet_member_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE active_pressure_day_count > 0
                    )::BIGINT AS observed_active_member_field_count
                FROM member_support
                GROUP BY incident_id, story_week
            ), compared AS (
                SELECT c.*,
                    COALESCE(m.attributed_member_field_count, 0)
                        AS attributed_member_field_count,
                    COALESCE(m.membership_pressure_core_field_count, 0)
                        AS membership_pressure_core_field_count,
                    COALESCE(m.membership_impact_lag_field_count, 0)
                        AS membership_impact_lag_field_count,
                    COALESCE(m.membership_fresh_decline_field_count, 0)
                        AS membership_fresh_decline_field_count,
                    COALESCE(m.membership_fresh_recovery_field_count, 0)
                        AS membership_fresh_recovery_field_count,
                    COALESCE(m.unclassified_fresh_response_field_count, 0)
                        AS unclassified_fresh_response_field_count,
                    COALESCE(m.supported_pressure_core_field_count, 0)
                        AS supported_pressure_core_field_count,
                    COALESCE(m.unsupported_pressure_core_membership_count, 0)
                        AS unsupported_pressure_core_membership_count,
                    COALESCE(m.pressure_core_missing_weather_field_count, 0)
                        AS pressure_core_missing_weather_field_count,
                    COALESCE(m.pressure_core_observed_inactive_field_count, 0)
                        AS pressure_core_observed_inactive_field_count,
                    COALESCE(m.supported_fresh_decline_field_count, 0)
                        AS supported_fresh_decline_field_count,
                    COALESCE(m.unsupported_fresh_decline_membership_count, 0)
                        AS unsupported_fresh_decline_membership_count,
                    COALESCE(m.supported_fresh_recovery_field_count, 0)
                        AS supported_fresh_recovery_field_count,
                    COALESCE(m.unsupported_fresh_recovery_membership_count, 0)
                        AS unsupported_fresh_recovery_membership_count,
                    COALESCE(m.missing_weather_member_field_count, 0)
                        AS missing_weather_member_field_count,
                    COALESCE(m.observed_quiet_member_field_count, 0)
                        AS observed_quiet_member_field_count,
                    COALESCE(m.observed_active_member_field_count, 0)
                        AS observed_active_member_field_count
                FROM checkpoints AS c
                LEFT JOIN membership_evidence AS m USING (incident_id, story_week)
            ), checks AS (
                SELECT *,
                    source_pressure_core_field_count
                        <> membership_pressure_core_field_count
                        AS pressure_membership_count_mismatch,
                    source_impact_lag_field_count
                        <> membership_impact_lag_field_count
                        AS impact_membership_count_mismatch,
                    source_fresh_decline_field_count
                        <> membership_fresh_decline_field_count
                        AS decline_membership_count_mismatch,
                    source_fresh_recovery_field_count
                        <> membership_fresh_recovery_field_count
                        AS recovery_membership_count_mismatch,
                    unsupported_pressure_core_membership_count > 0
                        AS unsupported_pressure_core_claim,
                    unsupported_fresh_decline_membership_count > 0
                        AS unsupported_fresh_decline_claim,
                    unsupported_fresh_recovery_membership_count > 0
                        AS unsupported_fresh_recovery_claim
                FROM compared
            ), scored AS (
                SELECT *,
                    CAST(pressure_membership_count_mismatch AS INTEGER)
                    + CAST(impact_membership_count_mismatch AS INTEGER)
                    + CAST(decline_membership_count_mismatch AS INTEGER)
                    + CAST(recovery_membership_count_mismatch AS INTEGER)
                    + CAST(unsupported_pressure_core_claim AS INTEGER)
                    + CAST(unsupported_fresh_decline_claim AS INTEGER)
                    + CAST(unsupported_fresh_recovery_claim AS INTEGER)
                    + CAST(unclassified_fresh_response_field_count > 0 AS INTEGER)
                        AS contradiction_count,
                    CONCAT_WS(', ',
                        CASE WHEN pressure_membership_count_mismatch
                            THEN 'pressure_membership_count_mismatch' END,
                        CASE WHEN impact_membership_count_mismatch
                            THEN 'impact_membership_count_mismatch' END,
                        CASE WHEN decline_membership_count_mismatch
                            THEN 'decline_membership_count_mismatch' END,
                        CASE WHEN recovery_membership_count_mismatch
                            THEN 'recovery_membership_count_mismatch' END,
                        CASE WHEN unsupported_pressure_core_claim
                            THEN 'unsupported_pressure_core_claim' END,
                        CASE WHEN unsupported_fresh_decline_claim
                            THEN 'unsupported_fresh_decline_claim' END,
                        CASE WHEN unsupported_fresh_recovery_claim
                            THEN 'unsupported_fresh_recovery_claim' END,
                        CASE WHEN unclassified_fresh_response_field_count > 0
                            THEN 'unclassified_fresh_response_claim' END
                    ) AS contradiction_reasons
                FROM checks
            )
            SELECT
                '{LIFECYCLE_RECONCILIATION_SCHEMA_VERSION}' AS schema_version,
                CAST('{availability_mode}' AS VARCHAR) AS availability_mode,
                incident_id, story_week, source_incident_state,
                source_pressure_core_field_count,
                source_impact_lag_field_count,
                source_fresh_decline_field_count,
                source_fresh_recovery_field_count,
                attributed_member_field_count,
                membership_pressure_core_field_count,
                membership_impact_lag_field_count,
                membership_fresh_decline_field_count,
                membership_fresh_recovery_field_count,
                supported_pressure_core_field_count,
                supported_fresh_decline_field_count,
                supported_fresh_recovery_field_count,
                unsupported_pressure_core_membership_count,
                unsupported_fresh_decline_membership_count,
                unsupported_fresh_recovery_membership_count,
                pressure_core_missing_weather_field_count,
                pressure_core_observed_inactive_field_count,
                missing_weather_member_field_count,
                observed_quiet_member_field_count,
                observed_active_member_field_count,
                unclassified_fresh_response_field_count,
                knowledge_bound_complete,
                contradiction_count,
                contradiction_reasons,
                contradiction_count = 0 AS positive_claim_reconciliation_complete,
                CASE
                    WHEN source_pressure_core_field_count > 0
                        THEN 'not_applicable_pressure_present'
                    WHEN attributed_member_field_count = 0
                        THEN 'insufficient_no_attributed_members'
                    WHEN missing_weather_member_field_count > 0
                        THEN 'missing_weather'
                    WHEN observed_active_member_field_count > 0
                        THEN '{active_quiet_status}'
                    ELSE 'observed_quiet_for_attributed_members'
                END AS quiet_evidence_status,
                CASE WHEN contradiction_count > 0 THEN 'contradiction'
                    WHEN '{availability_mode}' = 'strict'
                        THEN 'strict_positive_claims_reconciled'
                    ELSE 'diagnostic_positive_claims_reconciled'
                END AS reconciliation_status,
                {source_state_preserved} AS source_state_preserved,
                {native_lifecycle_replayed} AS lifecycle_state_recomputed,
                {native_lifecycle_replayed} AS component_absence_replayed,
                {native_lifecycle_replayed} AS full_lifecycle_replay_supported,
                {native_lifecycle_replayed} AS lifecycle_causal_claim_supported,
                CAST('{reconciliation_scope}' AS VARCHAR) AS reconciliation_scope
            FROM scored
            ORDER BY story_week, incident_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )
    contradictions = connection.execute(
        """
        SELECT incident_id, story_week, contradiction_reasons
        FROM read_parquet(?)
        WHERE contradiction_count > 0
        ORDER BY story_week, incident_id
        LIMIT 5
        """,
        [str(output_path)],
    ).fetchall()
    if contradictions:
        examples = "; ".join(
            f"{incident_id}@{story_week}[{reasons}]"
            for incident_id, story_week, reasons in contradictions
        )
        total = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?) WHERE contradiction_count > 0",
                [str(output_path)],
            ).fetchone()[0]
        )
        raise ValueError(
            "Incident V4 lifecycle evidence reconciliation failed closed: "
            f"contradictory_checkpoints={total}; {examples}"
        )


def _write_story_footprints(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    columns = {str(row[0]) for row in connection.execute(
        "DESCRIBE incident_footprints_v4_source"
    ).fetchall()}
    role_replacements: list[str] = []
    excluded: list[str] = []
    for role in ("pressure", "impact", "watch"):
        geometry = f"{role}_geometry_geojson"
        cell_ids = f"{role}_cell_ids_json"
        if geometry in columns and cell_ids in columns:
            excluded.append(geometry)
            role_replacements.append(
                f"CASE WHEN REPLACE(TRIM(CAST(c.{cell_ids} AS VARCHAR)), ' ', '') "
                f"NOT IN ('', '[]', 'null') THEN f.{geometry} ELSE NULL END AS {geometry}"
            )
    exclude_sql = f" EXCLUDE ({', '.join(excluded)})" if excluded else ""
    replacement_sql = (",\n                " + ",\n                ".join(role_replacements)) if role_replacements else ""
    connection.execute(
        f"""
        COPY (
            SELECT f.*{exclude_sql},
                c.story_week,
                c.story_known_date,
                c.story_known_time,
                c.source_checkpoint_knowledge_time,
                c.source_checkpoint_knowledge_time_inferred,
                c.contributing_evidence_knowledge_time,
                c.knowledge_bound_complete,
                c.story_known_time_raised,
                c.checkpoint_bound_mode,
                c.knowledge_time
                {replacement_sql}
            FROM incident_footprints_v4_source f
            JOIN story_checkpoints_v4 c
              ON CAST(f.incident_id AS VARCHAR) = CAST(c.incident_id AS VARCHAR)
             AND CAST(f.timeline_bucket AS DATE) = c.story_week
            ORDER BY c.story_known_date, f.incident_id, c.story_week
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _write_daily_grid(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    cell_degrees: float,
) -> None:
    connection.execute(
        """
        COPY (
            WITH mapped AS (
                SELECT
                    s.*,
                    TRY_CAST(g.centroid_lon AS DOUBLE) AS centroid_lon,
                    TRY_CAST(g.centroid_lat AS DOUBLE) AS centroid_lat,
                    FLOOR((TRY_CAST(g.centroid_lon AS DOUBLE) + 180.0) / ?)::BIGINT
                        AS grid_x,
                    FLOOR((TRY_CAST(g.centroid_lat AS DOUBLE) + 90.0) / ?)::BIGINT
                        AS grid_y
                FROM field_day_state_v4 s
                JOIN geometry_v4 g USING (field_id)
                WHERE TRY_CAST(g.centroid_lon AS DOUBLE) BETWEEN -180 AND 180
                  AND TRY_CAST(g.centroid_lat AS DOUBLE) BETWEEN -90 AND 90
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.calendar_date, s.field_id
                    ORDER BY s.monitored DESC, s.evaluable DESC,
                        s.crop_context_observed DESC,
                        s.stage_source_date DESC NULLS LAST,
                        s.crop_observation_date DESC NULLS LAST,
                        s.crop_instance_id, s.hazard_family
                ) = 1
            ), crop_stage_counts AS (
                SELECT calendar_date, grid_x, grid_y, crop_name, stage_bucket,
                    COUNT(DISTINCT field_id)::BIGINT AS represented_field_count
                FROM mapped
                GROUP BY calendar_date, grid_x, grid_y, crop_name, stage_bucket
            ), crop_stage_mode AS (
                SELECT calendar_date, grid_x, grid_y,
                    crop_name AS dominant_crop_name,
                    stage_bucket AS dominant_stage_bucket
                FROM crop_stage_counts
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY calendar_date, grid_x, grid_y
                    ORDER BY represented_field_count DESC, crop_name, stage_bucket
                ) = 1
            ), aggregated AS (
                SELECT
                    calendar_date, grid_x, grid_y,
                    COUNT(DISTINCT field_id)::BIGINT AS represented_field_count,
                    COUNT(DISTINCT crop_instance_id)::BIGINT AS crop_instance_count,
                    COUNT(DISTINCT field_id) FILTER (WHERE pressure_observed)::BIGINT
                        AS pressure_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE pressure_observed AND risk_rank >= 2
                    )::BIGINT AS elevated_pressure_field_count,
                    MAX(risk_rank)::INTEGER AS max_risk_rank,
                    ARG_MAX(hazard_family, risk_rank) AS dominant_hazard_family,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE crop_impact_active AND response_class IN
                            ('medium_decline', 'severe_decline')
                    )::BIGINT AS decline_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE crop_impact_active AND response_class = 'recovery'
                    )::BIGINT AS recovery_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE spectral_source_date IS NOT NULL
                    )::BIGINT AS crop_evidence_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE evidence_freshness = 'stale'
                    )::BIGINT AS stale_evidence_field_count,
                    MAX(spectral_source_date) AS newest_spectral_source_date,
                    MAX(s2_knowledge_date) AS newest_s2_knowledge_date
                FROM mapped
                GROUP BY calendar_date, grid_x, grid_y
            )
            SELECT
                a.calendar_date,
                'v4:' || CAST(a.grid_x AS VARCHAR) || ':' || CAST(a.grid_y AS VARCHAR)
                    AS grid_id,
                a.grid_x,
                a.grid_y,
                -180.0 + a.grid_x * ? AS min_lon,
                -90.0 + a.grid_y * ? AS min_lat,
                -180.0 + (a.grid_x + 1) * ? AS max_lon,
                -90.0 + (a.grid_y + 1) * ? AS max_lat,
                a.represented_field_count,
                a.crop_instance_count,
                a.pressure_field_count,
                a.elevated_pressure_field_count,
                a.max_risk_rank,
                a.dominant_hazard_family,
                m.dominant_crop_name,
                m.dominant_stage_bucket,
                a.decline_field_count,
                a.recovery_field_count,
                a.crop_evidence_field_count,
                a.stale_evidence_field_count,
                a.newest_spectral_source_date,
                a.newest_s2_knowledge_date,
                CAST(? AS DOUBLE) AS grid_cell_degrees,
                CAST('complete_centroid_aggregation' AS VARCHAR) AS representation_method
            FROM aggregated AS a
            JOIN crop_stage_mode AS m USING (calendar_date, grid_x, grid_y)
            ORDER BY a.calendar_date, a.grid_x, a.grid_y
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [
            str(output_path),
            cell_degrees, cell_degrees,
            cell_degrees, cell_degrees, cell_degrees, cell_degrees,
            cell_degrees,
        ],
    )


def _write_daily_pressure_grid(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    *,
    cell_degrees: float,
) -> None:
    """Precompute one grid lane per day and hazard without collapsing hazards."""
    connection.execute(
        """
        COPY (
            WITH mapped AS (
                SELECT
                    s.calendar_date,
                    s.field_id,
                    s.hazard_family,
                    s.risk_rank,
                    s.risk_band,
                    s.pressure_observed,
                    FLOOR((TRY_CAST(g.centroid_lon AS DOUBLE) + 180.0) / ?)::BIGINT
                        AS grid_x,
                    FLOOR((TRY_CAST(g.centroid_lat AS DOUBLE) + 90.0) / ?)::BIGINT
                        AS grid_y
                FROM field_day_state_v4 s
                JOIN geometry_v4 g USING (field_id)
                WHERE TRY_CAST(g.centroid_lon AS DOUBLE) BETWEEN -180 AND 180
                  AND TRY_CAST(g.centroid_lat AS DOUBLE) BETWEEN -90 AND 90
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.calendar_date, s.field_id, s.hazard_family
                    ORDER BY s.pressure_observed DESC, s.risk_rank DESC,
                        s.crop_instance_id
                ) = 1
            )
            SELECT
                calendar_date,
                hazard_family,
                'v4-pressure:' || CAST(grid_x AS VARCHAR) || ':'
                    || CAST(grid_y AS VARCHAR) || ':' || hazard_family AS grid_id,
                grid_x, grid_y,
                -180.0 + grid_x * ? AS min_lon,
                -90.0 + grid_y * ? AS min_lat,
                -180.0 + (grid_x + 1) * ? AS max_lon,
                -90.0 + (grid_y + 1) * ? AS max_lat,
                COUNT(DISTINCT field_id)::BIGINT AS represented_field_count,
                COUNT(DISTINCT field_id) FILTER (WHERE pressure_observed)::BIGINT
                    AS pressure_field_count,
                COUNT(DISTINCT field_id) FILTER (
                    WHERE pressure_observed AND risk_rank >= 2
                )::BIGINT AS elevated_pressure_field_count,
                MAX(risk_rank)::INTEGER AS max_risk_rank,
                ARG_MAX(risk_band, risk_rank) AS max_risk_band,
                CAST(? AS DOUBLE) AS grid_cell_degrees,
                CAST('complete_per_hazard_pressure_aggregation' AS VARCHAR)
                    AS representation_method
            FROM mapped
            WHERE pressure_observed AND hazard_family NOT IN ('', 'none', 'null')
            GROUP BY calendar_date, hazard_family, grid_x, grid_y
            ORDER BY calendar_date, hazard_family, grid_x, grid_y
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [
            str(output_path),
            cell_degrees, cell_degrees,
            cell_degrees, cell_degrees, cell_degrees, cell_degrees,
            cell_degrees,
        ],
    )


def _write_daily_timeline(
    connection: duckdb.DuckDBPyConnection, output_path: Path
) -> None:
    first_day, last_day = connection.execute(
        """
        SELECT MIN(day), MAX(day) FROM (
            SELECT MIN(calendar_date) AS day FROM field_day_state_v4
            UNION ALL SELECT MAX(calendar_date) FROM field_day_state_v4
            UNION ALL SELECT MIN(knowledge_date) FROM s2_updates_v4
            UNION ALL SELECT MAX(knowledge_date) FROM s2_updates_v4
            UNION ALL SELECT MIN(story_known_date) FROM story_checkpoints_v4
            UNION ALL SELECT MAX(story_known_date) FROM story_checkpoints_v4
        ) WHERE day IS NOT NULL
        """
    ).fetchone()
    if first_day is None or last_day is None:
        raise ValueError("V4 timeline has no valid evidence dates")
    span_days = int(
        connection.execute("SELECT DATE_DIFF('day', ?, ?)", [first_day, last_day]).fetchone()[0]
    )
    if span_days < 0 or span_days > 3660:
        raise ValueError(
            f"V4 timeline span is invalid or exceeds 10 years: {span_days} days"
        )
    connection.execute(
        """
        COPY (
            WITH bounds AS (
                SELECT MIN(day) AS first_day, MAX(day) AS last_day FROM (
                    SELECT MIN(calendar_date) AS day FROM field_day_state_v4
                    UNION ALL SELECT MAX(calendar_date) FROM field_day_state_v4
                    UNION ALL SELECT MIN(knowledge_date) FROM s2_updates_v4
                    UNION ALL SELECT MAX(knowledge_date) FROM s2_updates_v4
                    UNION ALL SELECT MIN(story_known_date) FROM story_checkpoints_v4
                    UNION ALL SELECT MAX(story_known_date) FROM story_checkpoints_v4
                ) WHERE day IS NOT NULL
            ), days AS (
                SELECT CAST(day AS DATE) AS calendar_date
                FROM bounds,
                UNNEST(GENERATE_SERIES(first_day, last_day, INTERVAL 1 DAY)) AS t(day)
            ), field_stats AS (
                SELECT calendar_date,
                    COUNT(DISTINCT field_id)::BIGINT AS monitored_field_count,
                    COUNT(DISTINCT crop_instance_id)::BIGINT AS monitored_crop_instance_count,
                    COUNT(DISTINCT field_id) FILTER (WHERE pressure_observed)::BIGINT
                        AS pressure_observed_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE pressure_observed AND risk_rank >= 2
                    )::BIGINT AS elevated_pressure_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE evidence_freshness = 'fresh'
                    )::BIGINT AS fresh_evidence_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE evidence_freshness = 'aging'
                    )::BIGINT AS aging_evidence_field_count,
                    COUNT(DISTINCT field_id) FILTER (
                        WHERE evidence_freshness = 'stale'
                    )::BIGINT AS stale_evidence_field_count
                FROM field_day_state_v4 GROUP BY calendar_date
            ), s2_stats AS (
                SELECT knowledge_date AS calendar_date,
                    COUNT(DISTINCT field_id)::BIGINT AS new_s2_field_count,
                    COUNT(DISTINCT crop_instance_id)::BIGINT AS new_s2_crop_instance_count,
                    COUNT(DISTINCT field_id) FILTER (WHERE new_response_evidence)::BIGINT
                        AS new_s2_response_field_count,
                    MIN(spectral_source_date) AS oldest_new_s2_source_date,
                    MAX(spectral_source_date) AS newest_new_s2_source_date
                FROM s2_updates_v4 GROUP BY knowledge_date
            ), attempt_stats AS (
                SELECT knowledge_date AS calendar_date,
                    COUNT(*) FILTER (WHERE marker_type = 'rejected')::BIGINT
                        AS rejected_s2_attempt_count
                FROM s2_attempts_v4 GROUP BY knowledge_date
            ), story_stats AS (
                SELECT story_known_date AS calendar_date,
                    COUNT(DISTINCT incident_id)::BIGINT AS story_checkpoint_count
                FROM story_checkpoints_v4 GROUP BY story_known_date
            ), represented AS (
                SELECT calendar_date,
                    SUM(represented_field_count)::BIGINT AS represented_field_count
                FROM daily_field_grid_v4 GROUP BY calendar_date
            )
            SELECT d.calendar_date,
                COALESCE(f.monitored_field_count, 0) AS monitored_field_count,
                COALESCE(f.monitored_field_count, 0) AS source_field_count,
                COALESCE(f.monitored_crop_instance_count, 0)
                    AS monitored_crop_instance_count,
                COALESCE(r.represented_field_count, 0) AS represented_field_count,
                GREATEST(
                    COALESCE(f.monitored_field_count, 0)
                    - COALESCE(r.represented_field_count, 0), 0
                ) AS unmappable_field_count,
                COALESCE(r.represented_field_count, 0) + GREATEST(
                    COALESCE(f.monitored_field_count, 0)
                    - COALESCE(r.represented_field_count, 0), 0
                ) AS accounted_field_count,
                COALESCE(f.pressure_observed_field_count, 0)
                    AS pressure_observed_field_count,
                COALESCE(f.elevated_pressure_field_count, 0)
                    AS elevated_pressure_field_count,
                COALESCE(s.new_s2_field_count, 0) AS new_s2_field_count,
                COALESCE(s.new_s2_crop_instance_count, 0)
                    AS new_s2_crop_instance_count,
                COALESCE(s.new_s2_response_field_count, 0)
                    AS new_s2_response_field_count,
                COALESCE(a.rejected_s2_attempt_count, 0)
                    AS rejected_s2_attempt_count,
                s.oldest_new_s2_source_date,
                s.newest_new_s2_source_date,
                COALESCE(f.fresh_evidence_field_count, 0) AS fresh_evidence_field_count,
                COALESCE(f.aging_evidence_field_count, 0) AS aging_evidence_field_count,
                COALESCE(f.stale_evidence_field_count, 0) AS stale_evidence_field_count,
                COALESCE(st.story_checkpoint_count, 0) AS story_checkpoint_count,
                f.monitored_field_count IS NOT NULL AS source_day_present,
                f.monitored_field_count IS NOT NULL
                    AND COALESCE(r.represented_field_count, 0) + GREATEST(
                        COALESCE(f.monitored_field_count, 0)
                        - COALESCE(r.represented_field_count, 0), 0
                    ) = COALESCE(f.monitored_field_count, 0)
                    AS coverage_reconciled,
                f.monitored_field_count IS NOT NULL
                    AND COALESCE(r.represented_field_count, 0)
                        = COALESCE(f.monitored_field_count, 0)
                    AS all_monitored_fields_mapped
            FROM days d
            LEFT JOIN field_stats f USING (calendar_date)
            LEFT JOIN s2_stats s USING (calendar_date)
            LEFT JOIN attempt_stats a USING (calendar_date)
            LEFT JOIN story_stats st USING (calendar_date)
            LEFT JOIN represented r USING (calendar_date)
            ORDER BY d.calendar_date
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(output_path)],
    )


def _validate_outputs(
    connection: duckdb.DuckDBPyConnection,
    stage: Path,
    *,
    native_replay: bool = False,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for name, filename in OUTPUT_FILES.items():
        path = stage / filename
        if not path.is_file():
            raise FileNotFoundError(f"V4 exporter did not write {filename}")
        count = int(
            connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        )
        if count < 1 and name in REQUIRED_NONEMPTY_OUTPUTS:
            raise ValueError(f"{filename} must contain at least one row")
        counts[name] = count
    echoed_markers = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM read_parquet(?)
            WHERE NOT is_new_acquisition
            """,
            [str(stage / OUTPUT_FILES["s2_updates"])],
        ).fetchone()[0]
    )
    nonmarkers = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM read_parquet(?)
            WHERE marker_type NOT IN ('acquisition', 'rejected')
            """,
            [str(stage / OUTPUT_FILES["s2_attempts"])],
        ).fetchone()[0]
    )
    future_story = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM read_parquet(?)
            WHERE story_known_date < story_week
            """,
            [str(stage / OUTPUT_FILES["story_checkpoints"])],
        ).fetchone()[0]
    )
    invalid_story_bound, raised_story_bound, incomplete_story_bound = (
        int(value or 0) for value in connection.execute(
            """
            SELECT
                COUNT_IF(
                    story_known_time IS NULL
                    OR CAST(story_known_time AS DATE) <> story_known_date
                    OR story_known_time < source_checkpoint_knowledge_time
                    OR (
                        contributing_evidence_knowledge_time IS NOT NULL
                        AND story_known_time < contributing_evidence_knowledge_time
                    )
                ),
                COUNT_IF(story_known_time_raised),
                COUNT_IF(NOT knowledge_bound_complete)
            FROM read_parquet(?)
            """,
            [str(stage / OUTPUT_FILES["story_checkpoints"])],
        ).fetchone()
    )
    (
        lifecycle_contradictions,
        invalid_lifecycle_contract,
        quiet_observed_checkpoints,
        quiet_missing_checkpoints,
        full_replay_supported_checkpoints,
    ) = (
        int(value or 0) for value in connection.execute(
            """
            SELECT
                COUNT_IF(contradiction_count > 0),
                COUNT_IF(
                    schema_version <> ?
                    OR source_state_preserved IS DISTINCT FROM ?
                    OR lifecycle_state_recomputed IS DISTINCT FROM ?
                    OR component_absence_replayed IS DISTINCT FROM ?
                    OR full_lifecycle_replay_supported IS DISTINCT FROM ?
                    OR lifecycle_causal_claim_supported IS DISTINCT FROM ?
                    OR positive_claim_reconciliation_complete
                        IS DISTINCT FROM (contradiction_count = 0)
                ),
                COUNT_IF(quiet_evidence_status = 'observed_quiet_for_attributed_members'),
                COUNT_IF(quiet_evidence_status = 'missing_weather'),
                COUNT_IF(full_lifecycle_replay_supported)
            FROM read_parquet(?)
            """,
            [
                LIFECYCLE_RECONCILIATION_SCHEMA_VERSION,
                not native_replay,
                native_replay,
                native_replay,
                native_replay,
                native_replay,
                str(stage / OUTPUT_FILES["lifecycle_reconciliation"]),
            ],
        ).fetchone()
    )
    lifecycle_key_mismatch = int(
        connection.execute(
            """
            WITH checkpoints AS (
                SELECT CAST(incident_id AS VARCHAR) AS incident_id,
                    CAST(story_week AS DATE) AS story_week
                FROM read_parquet(?)
            ), reconciled AS (
                SELECT CAST(incident_id AS VARCHAR) AS incident_id,
                    CAST(story_week AS DATE) AS story_week
                FROM read_parquet(?)
            )
            SELECT
                (SELECT COUNT(*) FROM checkpoints ANTI JOIN reconciled USING (
                    incident_id, story_week
                ))
                + (SELECT COUNT(*) FROM reconciled ANTI JOIN checkpoints USING (
                    incident_id, story_week
                ))
            """,
            [
                str(stage / OUTPUT_FILES["story_checkpoints"]),
                str(stage / OUTPUT_FILES["lifecycle_reconciliation"]),
            ],
        ).fetchone()[0]
    )
    unreconciled = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM read_parquet(?) WHERE NOT coverage_reconciled
            """,
            [str(stage / OUTPUT_FILES["timeline"])],
        ).fetchone()[0]
    )
    if (
        echoed_markers or nonmarkers or future_story
        or invalid_story_bound or lifecycle_contradictions
        or invalid_lifecycle_contract or lifecycle_key_mismatch or unreconciled
    ):
        raise ValueError(
            "V4 truth contract failed: "
            f"echoed_markers={echoed_markers}, nonmarkers={nonmarkers}, "
            f"future_story={future_story}, "
            f"invalid_story_bound={invalid_story_bound}, "
            f"lifecycle_contradictions={lifecycle_contradictions}, "
            f"invalid_lifecycle_contract={invalid_lifecycle_contract}, "
            f"lifecycle_key_mismatch={lifecycle_key_mismatch}, "
            f"unreconciled_days={unreconciled}"
        )
    return {
        "validation_passed": True,
        "daily_count": counts["timeline"],
        "field_day_state_count": counts["field_state"],
        "pressure_observation_count": counts["pressure_observations"],
        "grid_row_count": counts["grid"],
        "pressure_grid_row_count": counts["pressure_grid"],
        "s2_attempt_count": counts["s2_attempts"],
        "s2_update_count": counts["s2_updates"],
        "story_checkpoint_count": counts["story_checkpoints"],
        "story_checkpoint_raised_count": raised_story_bound,
        "story_checkpoint_incomplete_bound_count": incomplete_story_bound,
        "lifecycle_reconciliation_count": counts["lifecycle_reconciliation"],
        "lifecycle_reconciliation_contradiction_count": lifecycle_contradictions,
        "lifecycle_full_replay_supported_count": full_replay_supported_checkpoints,
        "lifecycle_quiet_observed_checkpoint_count": quiet_observed_checkpoints,
        "lifecycle_quiet_missing_checkpoint_count": quiet_missing_checkpoints,
        "counts": counts,
    }


def _manifest(
    *,
    stage: Path,
    incident_dir: Path,
    evidence_dir: Path,
    source_generation_dir: Path,
    base_manifest: dict[str, Any],
    input_metadata: dict[str, Any],
    validation: dict[str, Any],
    evidence_validation: dict[str, Any],
    display_grid_degrees: float,
    freshness_fresh_days: int,
    freshness_aging_days: int,
    native_replay: bool = False,
) -> dict[str, Any]:
    artifacts = _inventory(stage)
    content_hash = hashlib.sha256(
        json.dumps(
            {name: row["sha256"] for name, row in artifacts.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    identity = hashlib.sha256((SCHEMA_VERSION + content_hash).encode("ascii")).hexdigest()[:20]
    base_run = dict(base_manifest.get("run") or {})
    base_run.update(
        {
            "mode": MODE,
            "viewer_bundle_id": f"incident_viewer_v4_{identity}",
            "immutable": True,
            "viewer_ready": True,
            "dual_clock": True,
            "daily_timeline": True,
            "native_replay": native_replay,
            "map_publication_approved": False,
            "publication_status": PUBLICATION_STATUS,
        }
    )
    return {
        **base_manifest,
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "run": base_run,
        "source": {
            **dict(base_manifest.get("source") or {}),
            "incident_manifest_sha256": file_sha256(incident_dir / "manifest.json"),
            "evidence_manifest_sha256": (
                file_sha256(evidence_dir / "manifest.json")
                if (evidence_dir / "manifest.json").is_file() else None
            ),
            "source_generation_manifest_sha256": file_sha256(
                source_generation_dir / "manifest.json"
            ),
            "bundle_content_sha256": content_hash,
            **(
                {
                    "incident_manifest_mode": NATIVE_REPLAY_INCIDENT_MODE,
                    "incident_manifest_schema_version": (
                        NATIVE_REPLAY_INCIDENT_SCHEMA_VERSION
                    ),
                }
                if native_replay
                else {}
            ),
        },
        "semantics": {
            **dict(base_manifest.get("semantics") or {}),
            "story_spine_source": (
                "v4_native_story_replay" if native_replay else "v3_story_projection"
            ),
            "calendar_clock": "daily_as_of_date",
            "pressure_clock": (
                "effective_weather_observation_visible_no_earlier_than_knowledge_time"
            ),
            "pressure_observation_ledger": (
                "sparse_late_known_supplement_union_with_field_day_state"
            ),
            "late_known_pressure_backprojected": False,
            "crop_evidence_clock": "latest_new_s2_acquisition_known_on_or_before_day",
            "story_clock": "latest_weekly_checkpoint_with_knowledge_time_on_or_before_day",
            "story_checkpoint_knowledge_bound": (
                "max_source_checkpoint_membership_pressure_crop_context_"
                "and_applicable_s2_response_knowledge_times"
            ),
            "story_checkpoint_full_timestamp_preserved": True,
            "lifecycle_evidence_reconciliation": (
                "fail_closed_native_lifecycle_claims_against_v4_ledgers"
                if native_replay
                else "fail_closed_positive_v3_pressure_and_response_claims_against_v4_ledgers"
            ),
            "lifecycle_reconciliation_schema_version": (
                LIFECYCLE_RECONCILIATION_SCHEMA_VERSION
            ),
            "lifecycle_state_recomputed_from_v4": native_replay,
            "component_absence_replayed_from_v4": native_replay,
            "source_state_preserved": not native_replay,
            "full_lifecycle_replay_supported": native_replay,
            "lifecycle_causal_ownership_claimed": native_replay,
            "lifecycle_reconciliation_scope": (
                "native_lifecycle_component_absence_coverage_registries_"
                "and_policy_replayed_upstream"
                if native_replay
                else "positive_claims_only_component_absence_coverage_registries_"
                "and_policy_not_replayed"
            ),
            "decision_week_evidence_attribution": (
                "evidence_knowledge_time"
                if native_replay
                else "effective_or_source_date_with_late_known_s2_support"
            ),
            "strict_checkpoint_bound_enforced": (
                evidence_validation.get("availability_mode") == "strict"
            ),
            "reconstructed_checkpoint_bounds_diagnostic": (
                evidence_validation.get("availability_mode") == "reconstructed"
            ),
            "s2_echo_rows_are_markers": False,
            "crop_evidence_interpolated": False,
            "pressure_crop_story_geometry_substitution": False,
            "country_field_representation": "complete_centroid_aggregation",
            "field_polygons": "high_zoom_drilldown",
            "is_physical_movement": False,
        },
        "freshness_policy": {
            "fresh_max_days": freshness_fresh_days,
            "aging_max_days": freshness_aging_days,
            "calibration_status": "uncalibrated",
        },
        "display_grid": {
            "cell_degrees": display_grid_degrees,
            "purpose": "complete low-zoom field representation",
            "physical_area_claim": False,
        },
        "validation": {
            **dict(base_manifest.get("validation") or {}),
            **validation,
            "evidence": evidence_validation,
            "evidence_generation_binding": True,
        },
        "input_schemas": input_metadata,
        "outputs": {
            **dict(base_manifest.get("outputs") or {}),
            **OUTPUT_FILES,
        },
        "artifacts": artifacts,
        "warning": (
            "Incident V4 is a native-replay dual-clock diagnostic viewer. Freshness "
            "and response thresholds remain uncalibrated; pressure is a field-risk "
            "signal unless an independently identified weather-grid source is supplied."
            if native_replay
            else "Incident V4 is a dual-clock diagnostic viewer. Freshness, lifecycle, "
            "and response thresholds remain uncalibrated; pressure is a field-risk "
            "signal unless an independently identified weather-grid source is supplied."
        ),
    }


def _inventory(stage: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    with duckdb.connect(":memory:") as connection:
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = path.relative_to(stage).as_posix()
            if relative == "manifest.json":
                continue
            row: dict[str, Any] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix == ".parquet":
                row["row_count"] = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
                    ).fetchone()[0]
                )
            inventory[relative] = row
    return inventory


def validate_viewer_directory(viewer_dir: Path) -> dict[str, Any]:
    """Verify an immutable V4 viewer inventory before it is reused or served."""
    root = viewer_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"V4 viewer manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"V4 viewer manifest is invalid: {manifest_path}") from exc
    run = manifest.get("run") or {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("V4 viewer schema_version does not match the validator")
    if str(manifest.get("mode") or run.get("mode") or "") != MODE:
        raise ValueError("Viewer manifest is not a dual-clock Incident V4 bundle")
    native_replay_value = run.get("native_replay", False)
    if not isinstance(native_replay_value, bool):
        raise ValueError("V4 viewer native_replay marker must be boolean")
    native_replay = native_replay_value
    if (
        run.get("status") != "complete"
        or run.get("immutable") is not True
        or run.get("viewer_ready") is not True
    ):
        raise ValueError("V4 viewer manifest is not complete, immutable, and ready")
    source = manifest.get("source") or {}
    if native_replay and (
        source.get("incident_manifest_mode") != NATIVE_REPLAY_INCIDENT_MODE
        or source.get("incident_manifest_schema_version")
        != NATIVE_REPLAY_INCIDENT_SCHEMA_VERSION
    ):
        raise ValueError("Native V4 viewer source manifest contract is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("V4 viewer manifest has no artifact inventory")
    disk_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    manifest_names = {str(name) for name in artifacts}
    if disk_names != manifest_names:
        missing = sorted(manifest_names - disk_names)
        unexpected = sorted(disk_names - manifest_names)
        raise ValueError(
            "V4 viewer inventory does not match disk: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    verified_rows: dict[str, int] = {}
    total_bytes = 0
    with duckdb.connect(":memory:") as connection:
        for name in sorted(manifest_names):
            expected = artifacts.get(name)
            relative = Path(name)
            if (
                not isinstance(expected, dict)
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError(f"Invalid V4 viewer inventory entry: {name!r}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"V4 viewer artifact escapes or is missing: {name}")
            try:
                expected_size = int(expected["size_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"V4 viewer artifact has invalid size: {name}") from exc
            actual_size = path.stat().st_size
            if expected_size < 0 or actual_size != expected_size:
                raise ValueError(
                    f"V4 viewer artifact size mismatch: {name} "
                    f"expected={expected_size} actual={actual_size}"
                )
            expected_sha = str(expected.get("sha256") or "")
            if len(expected_sha) != 64 or file_sha256(path) != expected_sha:
                raise ValueError(f"V4 viewer artifact hash mismatch: {name}")
            total_bytes += actual_size
            if path.suffix == ".parquet":
                try:
                    expected_rows = int(expected["row_count"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"V4 viewer parquet has invalid row_count: {name}"
                    ) from exc
                actual_rows = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
                    ).fetchone()[0]
                )
                if expected_rows < 0 or actual_rows != expected_rows:
                    raise ValueError(
                        f"V4 viewer parquet row_count mismatch: {name} "
                        f"expected={expected_rows} actual={actual_rows}"
                    )
                verified_rows[name] = actual_rows
        checkpoint_name = OUTPUT_FILES["story_checkpoints"]
        if checkpoint_name not in verified_rows:
            raise ValueError("V4 viewer inventory is missing story checkpoints")
        try:
            invalid_bound, strict_violation = connection.execute(
                """
                SELECT
                    COUNT_IF(
                        story_known_time IS NULL
                        OR CAST(story_known_time AS DATE) <> story_known_date
                        OR story_known_time < source_checkpoint_knowledge_time
                        OR (
                            contributing_evidence_knowledge_time IS NOT NULL
                            AND story_known_time < contributing_evidence_knowledge_time
                        )
                    ),
                    COUNT_IF(
                        checkpoint_bound_mode = 'strict'
                        AND (
                            source_checkpoint_knowledge_time_inferred
                            OR story_known_time_raised
                            OR NOT knowledge_bound_complete
                        )
                    )
                FROM read_parquet(?)
                """,
                [str(root / checkpoint_name)],
            ).fetchone()
        except duckdb.Error as exc:
            raise ValueError(
                "V4 viewer story checkpoints lack the causal knowledge-bound schema"
            ) from exc
        if int(invalid_bound or 0) or int(strict_violation or 0):
            raise ValueError(
                "V4 viewer story checkpoint knowledge bounds are invalid"
            )
        lifecycle_name = OUTPUT_FILES["lifecycle_reconciliation"]
        if lifecycle_name not in verified_rows:
            raise ValueError("V4 viewer inventory is missing lifecycle reconciliation")
        try:
            invalid_lifecycle, lifecycle_contradictions = connection.execute(
                """
                SELECT
                    COUNT_IF(
                        schema_version <> ?
                        OR source_state_preserved IS DISTINCT FROM ?
                        OR lifecycle_state_recomputed IS DISTINCT FROM ?
                        OR component_absence_replayed IS DISTINCT FROM ?
                        OR full_lifecycle_replay_supported IS DISTINCT FROM ?
                        OR lifecycle_causal_claim_supported IS DISTINCT FROM ?
                        OR positive_claim_reconciliation_complete
                            IS DISTINCT FROM (contradiction_count = 0)
                    ),
                    COUNT_IF(contradiction_count > 0)
                FROM read_parquet(?)
                """,
                [
                    LIFECYCLE_RECONCILIATION_SCHEMA_VERSION,
                    not native_replay,
                    native_replay,
                    native_replay,
                    native_replay,
                    native_replay,
                    str(root / lifecycle_name),
                ],
            ).fetchone()
            lifecycle_key_mismatch = connection.execute(
                """
                WITH checkpoints AS (
                    SELECT CAST(incident_id AS VARCHAR) AS incident_id,
                        CAST(story_week AS DATE) AS story_week
                    FROM read_parquet(?)
                ), reconciled AS (
                    SELECT CAST(incident_id AS VARCHAR) AS incident_id,
                        CAST(story_week AS DATE) AS story_week
                    FROM read_parquet(?)
                )
                SELECT
                    (SELECT COUNT(*) FROM checkpoints ANTI JOIN reconciled USING (
                        incident_id, story_week
                    ))
                    + (SELECT COUNT(*) FROM reconciled ANTI JOIN checkpoints USING (
                        incident_id, story_week
                    ))
                """,
                [str(root / checkpoint_name), str(root / lifecycle_name)],
            ).fetchone()[0]
        except duckdb.Error as exc:
            raise ValueError(
                "V4 viewer lifecycle reconciliation schema is invalid"
            ) from exc
        if (
            int(invalid_lifecycle or 0)
            or int(lifecycle_contradictions or 0)
            or int(lifecycle_key_mismatch or 0)
        ):
            raise ValueError(
                "V4 viewer lifecycle evidence reconciliation is invalid"
            )

    outputs = manifest.get("outputs") or {}
    counts = (manifest.get("validation") or {}).get("counts") or {}
    for logical_name, expected_count in counts.items():
        filename = str(outputs.get(logical_name) or "")
        if filename not in verified_rows or verified_rows[filename] != int(expected_count):
            raise ValueError(
                f"V4 viewer validation count does not reconcile: {logical_name}"
            )
    content_sha = hashlib.sha256(
        json.dumps(
            {name: artifacts[name]["sha256"] for name in sorted(manifest_names)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if content_sha != str((manifest.get("source") or {}).get("bundle_content_sha256") or ""):
        raise ValueError("V4 viewer bundle content hash does not match its inventory")
    semantics = manifest.get("semantics") or {}
    source_state_preserved = semantics.get(
        "source_state_preserved", True if not native_replay else None
    )
    full_lifecycle_replay_supported = semantics.get(
        "full_lifecycle_replay_supported", False if not native_replay else None
    )
    if (
        semantics.get("lifecycle_reconciliation_schema_version")
        != LIFECYCLE_RECONCILIATION_SCHEMA_VERSION
        or semantics.get("lifecycle_state_recomputed_from_v4") is not native_replay
        or semantics.get("component_absence_replayed_from_v4") is not native_replay
        or source_state_preserved is not (not native_replay)
        or full_lifecycle_replay_supported is not native_replay
        or semantics.get("lifecycle_causal_ownership_claimed") is not native_replay
    ):
        raise ValueError("V4 viewer lifecycle reconciliation semantics are invalid")
    return {
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "native_replay": native_replay,
        "artifact_count": len(manifest_names),
        "parquet_count": len(verified_rows),
        "total_size_bytes": total_bytes,
        "bundle_content_sha256": content_sha,
    }
