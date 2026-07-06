#!/usr/bin/env python3
"""Atomically enrich the echo deliverable for Incident V4 evidence building."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Sequence

import duckdb

from story_monitor.incident_policy_v4 import load_incident_policy_v4
from story_monitor.incident_release_v4 import (
    CORRECTION_POLICY,
    normalize_released_at,
)


SCHEMA_VERSION = "incident-enriched-source-v4/1"
_CANONICAL = {
    "sentinel_observation_date": (("sentinel_observation_date", "spectral_source_date"), "DATE"),
    "sentinel_days_stale": (("sentinel_days_stale", "spectral_echo_days"), "INTEGER"),
    "valid_pixel_fraction": (("valid_pixel_fraction",), "DOUBLE"),
    "cloud_pct": (("cloud_pct",), "DOUBLE"),
    "s2_field_quality_flag": (("s2_field_quality_flag",), "VARCHAR"),
    "s2_good_observation": (("s2_good_observation",), "BOOLEAN"),
    "ndvi": (("ndvi",), "DOUBLE"),
    "ndmi": (("ndmi",), "DOUBLE"),
    "psri": (("psri",), "DOUBLE"),
    "drought_risk_score": (("drought_risk_score",), "DOUBLE"),
    "ponding_risk_score": (("ponding_risk_score",), "DOUBLE"),
    "heat_risk_score": (("heat_risk_score",), "DOUBLE"),
    "wind_risk_score": (("wind_risk_score",), "DOUBLE"),
    "drought_hazard_level": (("drought_hazard_level",), "VARCHAR"),
    "ponding_hazard_level": (("ponding_hazard_level",), "VARCHAR"),
    "heatwave_category": (("heatwave_category",), "VARCHAR"),
    "wind_hazard_level": (("wind_hazard_level",), "VARCHAR"),
    "spi_index": (("spi_index",), "DOUBLE"),
    "ponding_mm": (("ponding_mm",), "DOUBLE"),
    "apparent_temperature": (("apparent_temperature",), "DOUBLE"),
    "temperature": (("temperature",), "DOUBLE"),
    "humidity": (("humidity",), "DOUBLE"),
    "wind_speed": (("wind_speed",), "DOUBLE"),
    "wind_gust": (("wind_gust", "wind_gust_kmh"), "DOUBLE"),
    "season_calendar_source": (("season_calendar_source", "stage_source"), "VARCHAR"),
    "planting_date": (("planting_date",), "DATE"),
}


def prepare_incident_source_v4(
    echo_deliverable: Path,
    full_sources: Sequence[Path],
    output_parquet: Path,
    *,
    released_at: str,
    acquisition_sources: Sequence[Path] = (),
    availability_mode: str = "reconstructed",
    threads: int = 8,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    policy = load_incident_policy_v4()
    release_watermark = normalize_released_at(released_at)
    mode = policy.validate_availability_mode(availability_mode)
    deliverable = _file(echo_deliverable)
    full_paths = tuple(_file(path) for path in full_sources)
    acquisition_paths = tuple(_file(path) for path in acquisition_sources)
    if not full_paths:
        raise ValueError("At least one rich full-source parquet is required")
    output = output_parquet.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Immutable enriched source already exists: {output}")
    if not 1 <= int(threads) <= 256:
        raise ValueError("threads must be between 1 and 256")
    output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    try:
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=?", [int(threads)])
        if memory_limit:
            con.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory=?", [str(resolved)])

        deliverable_cols = _view(con, "deliverable_v4", (deliverable,))
        full_cols = _view(con, "full_source_v4", full_paths)
        _require(deliverable_cols, {"field_id", "observation_date", "spectral_echo_days"}, "echo deliverable")
        _require(full_cols, {"field_id", "observation_date"}, "rich full source")
        acquisition_cols: set[str] = set()
        if acquisition_paths:
            acquisition_cols = _view(con, "acquisition_input_v4", acquisition_paths)
            _create_acquisition_qa_view(con, acquisition_cols)
        else:
            con.execute(
                """CREATE TEMP VIEW acquisition_qa_v4 AS SELECT
                CAST(NULL AS VARCHAR) field_id, CAST(NULL AS DATE) sentinel_observation_date,
                CAST(NULL AS DOUBLE) acquisition_valid_pixel_fraction,
                CAST(NULL AS DOUBLE) acquisition_cloud_pct,
                CAST(NULL AS VARCHAR) acquisition_quality_flag,
                CAST(NULL AS BOOLEAN) acquisition_good,
                CAST(NULL AS TIMESTAMP) acquisition_available_at WHERE FALSE"""
            )

        duplicate_deliverable = _duplicate_keys(con, "deliverable_v4")
        duplicate_full = _duplicate_keys(con, "full_source_v4")
        if duplicate_deliverable or duplicate_full:
            raise ValueError(
                f"Incident V4 preprocessing requires unique field/day keys: "
                f"deliverable={duplicate_deliverable}, full={duplicate_full}"
            )
        _validate_join(con, full_cols)
        if mode == "strict":
            _validate_strict_columns(full_cols, acquisition_cols)

        select_prefix = ", ".join(
            f"d.{_q(name)}" for name in sorted(deliverable_cols)
            if name not in set(_CANONICAL) | {
                "weather_available_at", "spectral_available_at", "stage_available_at",
                "availability_reconstructed", "source_row_present",
            }
        )
        rich = {
            name: _source_expr(full_cols, aliases, sql_type)
            for name, (aliases, sql_type) in _CANONICAL.items()
        }
        sentinel = rich["sentinel_observation_date"]
        if mode == "strict":
            weather_available = _source_expr(
                full_cols, ("weather_available_at", "weather_knowledge_time"), "TIMESTAMP"
            )
            stage_available = _source_expr(
                full_cols, ("stage_available_at", "stage_knowledge_time"), "TIMESTAMP"
            )
            spectral_full = _source_expr(
                full_cols, ("spectral_available_at", "sentinel_available_at"), "TIMESTAMP"
            )
            spectral_available = f"COALESCE(a.acquisition_available_at, {spectral_full})"
        else:
            weather_available = "CAST(d.observation_date AS TIMESTAMP)"
            stage_available = "CAST(d.observation_date AS TIMESTAMP)"
            spectral_available = (
                f"CASE WHEN {sentinel} IS NULL THEN NULL ELSE "
                f"MIN(CAST(d.observation_date AS TIMESTAMP)) OVER "
                f"(PARTITION BY CAST(d.field_id AS VARCHAR), {sentinel}) END"
            )
        rich["valid_pixel_fraction"] = f"COALESCE(a.acquisition_valid_pixel_fraction, {rich['valid_pixel_fraction']})"
        rich["cloud_pct"] = f"COALESCE(a.acquisition_cloud_pct, {rich['cloud_pct']})"
        rich["s2_field_quality_flag"] = f"COALESCE(a.acquisition_quality_flag, {rich['s2_field_quality_flag']})"
        rich["s2_good_observation"] = f"COALESCE(a.acquisition_good, {rich['s2_good_observation']})"
        canonical_sql = ", ".join(f"{expr} AS {_q(name)}" for name, expr in rich.items())
        query = f"""
            SELECT {select_prefix}, {canonical_sql},
              {weather_available} AS weather_available_at,
              {spectral_available} AS spectral_available_at,
              {stage_available} AS stage_available_at,
              {str(mode == 'reconstructed').upper()} AS availability_reconstructed,
              f.field_id IS NOT NULL AS source_row_present
            FROM deliverable_v4 d
            LEFT JOIN full_source_v4 f
              ON CAST(f.field_id AS VARCHAR) = CAST(d.field_id AS VARCHAR)
             AND CAST(f.observation_date AS DATE) = CAST(d.observation_date AS DATE)
            LEFT JOIN acquisition_qa_v4 a
              ON a.field_id = CAST(d.field_id AS VARCHAR)
             AND a.sentinel_observation_date = {sentinel}
            ORDER BY CAST(d.field_id AS VARCHAR), CAST(d.observation_date AS DATE)
        """
        _validate_enriched_query(con, query, mode, release_watermark)
        with NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            con.sql(query).write_parquet(str(temporary), compression="zstd")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        row_count = int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(output)]).fetchone()[0])
    finally:
        con.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "immutable": True,
        "released_at": release_watermark,
        "correction_policy": CORRECTION_POLICY,
        "availability": {
            "mode": mode,
            "diagnostic_reconstruction": mode == "reconstructed",
            "reconstruction_rule": (
                "weather/stage=observation_date; spectral=first daily appearance"
                if mode == "reconstructed" else None
            ),
        },
        "policy": {
            "version": policy.version,
            "sha256": policy.source_sha256,
            "calibration_status": policy.calibration_status,
            "warning": policy.warning,
        },
        "inputs": {
            "echo_deliverable": _metadata(deliverable),
            "full_sources": [_metadata(path) for path in full_paths],
            "acquisition_sources": [_metadata(path) for path in acquisition_paths],
        },
        "output": {**_metadata(output), "row_count": row_count},
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "status": "written", "schema_version": SCHEMA_VERSION,
        "output_parquet": str(output), "manifest": str(manifest_path),
        "availability_mode": mode, "released_at": release_watermark,
        "row_count": row_count,
    }


def _create_acquisition_qa_view(
    con: duckdb.DuckDBPyConnection, columns: set[str]
) -> None:
    source = _first(columns, ("sentinel_observation_date", "spectral_source_date", "prediction_observation_date"))
    if source is None:
        raise ValueError("acquisition source is missing a Sentinel source date")
    available = _external(columns, ("spectral_available_at", "sentinel_available_at", "available_at"), "TIMESTAMP")
    con.execute(
        f"""
        CREATE TEMP VIEW acquisition_qa_v4 AS
        SELECT CAST(field_id AS VARCHAR) AS field_id,
          CAST({_q(source)} AS DATE) AS sentinel_observation_date,
          ARG_MIN({_external(columns, ('valid_pixel_fraction',), 'DOUBLE')}, CAST({_q(source)} AS DATE)) AS acquisition_valid_pixel_fraction,
          ARG_MIN({_external(columns, ('cloud_pct',), 'DOUBLE')}, CAST({_q(source)} AS DATE)) AS acquisition_cloud_pct,
          ARG_MIN({_external(columns, ('s2_field_quality_flag',), 'VARCHAR')}, CAST({_q(source)} AS DATE)) AS acquisition_quality_flag,
          ARG_MIN({_external(columns, ('s2_good_observation',), 'BOOLEAN')}, CAST({_q(source)} AS DATE)) AS acquisition_good,
          MIN({available}) AS acquisition_available_at
        FROM acquisition_input_v4
        WHERE {_q(source)} IS NOT NULL
        GROUP BY 1, 2
        """
    )


def _validate_join(con: duckdb.DuckDBPyConnection, full_cols: set[str]) -> None:
    source = _first(full_cols, ("sentinel_observation_date", "spectral_source_date"))
    stale = _first(full_cols, ("sentinel_days_stale", "spectral_echo_days"))
    if source is None or stale is None:
        raise ValueError("rich full source requires Sentinel source date and staleness")
    unmatched, mismatch = con.execute(
        f"""
        SELECT COUNT_IF(f.field_id IS NULL),
          COUNT_IF(f.field_id IS NOT NULL AND (
            (TRY_CAST(d.spectral_echo_days AS INTEGER) IS NULL)
              <> (TRY_CAST(f.{_q(stale)} AS INTEGER) IS NULL)
            OR (TRY_CAST(f.{_q(stale)} AS INTEGER) IS NULL)
              <> (TRY_CAST(f.{_q(source)} AS DATE) IS NULL)
            OR (
              TRY_CAST(d.spectral_echo_days AS INTEGER) IS NOT NULL
              AND TRY_CAST(f.{_q(stale)} AS INTEGER) IS NOT NULL
              AND TRY_CAST(f.{_q(source)} AS DATE) IS NOT NULL
              AND (
                TRY_CAST(d.spectral_echo_days AS INTEGER)
                  <> TRY_CAST(f.{_q(stale)} AS INTEGER)
                OR TRY_CAST(d.spectral_echo_days AS INTEGER)
                  <> DATE_DIFF('day', CAST(f.{_q(source)} AS DATE), CAST(d.observation_date AS DATE))
              )
            )
          ))
        FROM deliverable_v4 d LEFT JOIN full_source_v4 f
          ON CAST(f.field_id AS VARCHAR) = CAST(d.field_id AS VARCHAR)
         AND CAST(f.observation_date AS DATE) = CAST(d.observation_date AS DATE)
        """
    ).fetchone()
    if unmatched or mismatch:
        raise ValueError(
            f"Incident V4 source join failed: unmatched={unmatched}, echo_mismatch={mismatch}"
        )


def _validate_strict_columns(full_cols: set[str], acquisition_cols: set[str]) -> None:
    for label, candidates in (
        ("weather_available_at", ("weather_available_at", "weather_knowledge_time")),
        ("stage_available_at", ("stage_available_at", "stage_knowledge_time")),
    ):
        if _first(full_cols, candidates) is None:
            raise ValueError(f"strict availability requires {label} in rich source")
    if _first(full_cols, ("spectral_available_at", "sentinel_available_at")) is None and _first(
        acquisition_cols, ("spectral_available_at", "sentinel_available_at", "available_at")
    ) is None:
        raise ValueError("strict availability requires spectral_available_at")


def _validate_enriched_query(
    con: duckdb.DuckDBPyConnection, query: str, mode: str, released_at: str
) -> None:
    future, negative, mismatch, missing = con.execute(
        f"""SELECT
          COUNT_IF(sentinel_observation_date > CAST(observation_date AS DATE)),
          COUNT_IF(sentinel_days_stale < 0),
          COUNT_IF(
            (sentinel_observation_date IS NULL) <> (sentinel_days_stale IS NULL)
            OR (sentinel_observation_date IS NOT NULL AND sentinel_days_stale IS NOT NULL
              AND sentinel_days_stale
                <> DATE_DIFF('day', sentinel_observation_date, CAST(observation_date AS DATE)))
          ),
          COUNT_IF(source_row_present = FALSE)
        FROM ({query}) q"""
    ).fetchone()
    if future or negative or mismatch or missing:
        raise ValueError(
            f"Incident V4 enriched source failed: future_source={future}, negative_echo={negative}, "
            f"echo_mismatch={mismatch}, missing_full_rows={missing}"
        )
    post_release = int(con.execute(
        f"""SELECT COUNT(*) FROM ({query}) q
        WHERE TRY_CAST(weather_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ)
           OR TRY_CAST(stage_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ)
           OR TRY_CAST(spectral_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ)""",
        [released_at, released_at, released_at],
    ).fetchone()[0])
    if post_release:
        raise ValueError(
            f"Incident V4 enriched source has {post_release} post-release timestamps"
        )
    if mode == "strict":
        invalid = int(con.execute(
            f"""SELECT COUNT(*) FROM ({query}) q
            WHERE weather_available_at IS NULL OR stage_available_at IS NULL
               OR weather_available_at < CAST(observation_date AS TIMESTAMP)
               OR stage_available_at < CAST(observation_date AS TIMESTAMP)
               OR (sentinel_observation_date IS NOT NULL AND spectral_available_at IS NULL)
               OR spectral_available_at < CAST(sentinel_observation_date AS TIMESTAMP)"""
        ).fetchone()[0])
        if invalid:
            raise ValueError(f"strict availability has {invalid} invalid timestamps")


def _view(
    con: duckdb.DuckDBPyConnection, name: str, paths: Sequence[Path]
) -> set[str]:
    relation = con.read_parquet([str(path) for path in paths], union_by_name=True)
    relation.create_view(name)
    return set(relation.columns)


def _source_expr(columns: set[str], aliases: Iterable[str], sql_type: str) -> str:
    name = _first(columns, aliases)
    return f"TRY_CAST(f.{_q(name)} AS {sql_type})" if name else f"CAST(NULL AS {sql_type})"


def _external(columns: set[str], aliases: Iterable[str], sql_type: str) -> str:
    name = _first(columns, aliases)
    return f"TRY_CAST({_q(name)} AS {sql_type})" if name else f"CAST(NULL AS {sql_type})"


def _first(columns: set[str], aliases: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _duplicate_keys(con: duckdb.DuckDBPyConnection, view: str) -> int:
    return int(con.execute(
        f"SELECT COUNT(*) FROM (SELECT CAST(field_id AS VARCHAR), CAST(observation_date AS DATE), COUNT(*) FROM {_q(view)} GROUP BY 1,2 HAVING COUNT(*) > 1)"
    ).fetchone()[0])


def _require(columns: set[str], required: set[str], label: str) -> None:
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _atomic_text(path: Path, value: str) -> None:
    with NamedTemporaryFile(
        mode="w", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        delete=False, encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echo-deliverable", type=Path, required=True)
    parser.add_argument("--full-parquet", type=Path, action="append", required=True)
    parser.add_argument("--acquisition-parquet", type=Path, action="append", default=[])
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument(
        "--released-at", required=True,
        help="Monotonic timezone-aware ingest/release watermark (normalized to UTC).",
    )
    parser.add_argument("--availability-mode", choices=("strict", "reconstructed"), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit")
    parser.add_argument("--temp-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare_incident_source_v4(
        args.echo_deliverable, args.full_parquet, args.output_parquet,
        released_at=args.released_at,
        acquisition_sources=args.acquisition_parquet,
        availability_mode=args.availability_mode, threads=args.threads,
        memory_limit=args.memory_limit, temp_dir=args.temp_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
