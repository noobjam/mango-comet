"""Build immutable daily pressure and acquisition-timed Incident V4 evidence.

V4 is an evidence projection beside, never inside, immutable Incident V3
analytics.  The existing generation remains authoritative for crop-instance
identity; an optional enriched source supplies weather/Sentinel provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import duckdb

from .incident_policy_v4 import IncidentPolicyV4, load_incident_policy_v4
from .incident_release_v4 import (
    CORRECTION_POLICY,
    normalize_released_at,
    validate_correction_policy,
)


SCHEMA_VERSION = "crop-impact-incident-evidence-v4/1"
CROP_DAY_FILE = "crop_day_context_v4.parquet"
PRESSURE_FILE = "field_day_pressure_v4.parquet"
S2_FILE = "field_s2_acquisition_v4.parquet"
MANIFEST_FILE = "manifest.json"

_SIGNAL_REQUIRED = {
    "field_id", "observation_date", "crop_name", "crop_season", "crop_stage",
    "stage_family", "crop_instance_id", "pressure_observed", "risk_rank",
    "risk_band", "hazard_family", "ndvi", "ndmi", "psri",
}
_RICH_COLUMNS: dict[str, tuple[str, ...]] = {
    "spectral_source_date": ("sentinel_observation_date", "spectral_source_date"),
    "spectral_echo_days": ("spectral_echo_days", "sentinel_days_stale"),
    "weather_available_at_raw": ("weather_available_at", "weather_knowledge_time"),
    "spectral_available_at_raw": ("spectral_available_at", "sentinel_available_at"),
    "stage_available_at_raw": ("stage_available_at", "stage_knowledge_time"),
    "valid_pixel_fraction": ("valid_pixel_fraction",),
    "cloud_pct": ("cloud_pct",),
    "s2_field_quality_flag": ("s2_field_quality_flag",),
    "s2_good_observation": ("s2_good_observation",),
    "drought_risk_score": ("drought_risk_score",),
    "ponding_risk_score": ("ponding_risk_score",),
    "heat_risk_score": ("heat_risk_score",),
    "wind_risk_score": ("wind_risk_score",),
    "spi_index": ("spi_index",),
    "ponding_mm": ("ponding_mm",),
    "apparent_temperature": ("apparent_temperature",),
    "temperature": ("temperature",),
    "humidity": ("humidity",),
    "wind_speed": ("wind_speed",),
    "wind_gust": ("wind_gust", "wind_gust_kmh"),
    "stage_source": ("season_calendar_source", "stage_source"),
    "planting_date": ("planting_date",),
}


def build_incident_context_v4(
    generation_dir: Path,
    evidence_dir: Path,
    *,
    released_at: str,
    enriched_source_parquet: Path | None = None,
    acquisition_parquet: Path | None = None,
    availability_mode: str = "reconstructed",
    policy: IncidentPolicyV4 | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Write an immutable standalone evidence directory with three V4 ledgers."""
    generation_dir = generation_dir.expanduser().resolve()
    evidence_dir = evidence_dir.expanduser().resolve()
    enriched_source_parquet = _optional_file(enriched_source_parquet)
    acquisition_parquet = _optional_file(acquisition_parquet)
    policy = policy or load_incident_policy_v4()
    release_watermark = normalize_released_at(released_at)
    mode = policy.validate_availability_mode(availability_mode)
    if evidence_dir.exists() or evidence_dir.is_symlink():
        raise FileExistsError(f"Immutable Incident V4 evidence already exists: {evidence_dir}")
    if not 1 <= int(threads) <= 256:
        raise ValueError("threads must be between 1 and 256")

    manifest_path = generation_dir / "manifest.json"
    signals_path = generation_dir / "daily_causal_signals.parquet"
    source_manifest = _generation_manifest(manifest_path)
    if not signals_path.is_file():
        raise FileNotFoundError(f"Generation is missing daily causal signals: {signals_path}")
    if mode == "strict" and enriched_source_parquet is None:
        raise ValueError("strict availability requires enriched_source_parquet")
    enriched_source_contract = (
        validate_enriched_source_v4(enriched_source_parquet, expected_mode=mode)
        if enriched_source_parquet is not None else None
    )
    if (
        enriched_source_contract is not None
        and enriched_source_contract["released_at"] != release_watermark
    ):
        raise ValueError(
            "Evidence released_at must equal the immutable enriched-source "
            "released_at watermark"
        )

    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".incident-evidence-v4-", dir=evidence_dir.parent) as tmp:
        stage = Path(tmp) / evidence_dir.name
        stage.mkdir()
        con = _connection(threads, memory_limit, temp_dir)
        try:
            signal_cols = _register_parquet(con, "signals_source", signals_path)
            missing = sorted(_SIGNAL_REQUIRED - signal_cols)
            if missing:
                raise ValueError("daily_causal_signals.parquet is missing V4 columns: " + ", ".join(missing))
            enriched_cols: set[str] = set()
            if enriched_source_parquet is not None:
                enriched_cols = _register_parquet(
                    con, "enriched_source", enriched_source_parquet
                )
                _require_columns(
                    enriched_cols, {"field_id", "observation_date"}, "enriched source"
                )
            acquisition_cols: set[str] = set()
            if acquisition_parquet is not None:
                acquisition_cols = _register_parquet(
                    con, "acquisition_source", acquisition_parquet
                )
                _require_columns(acquisition_cols, {"field_id"}, "acquisition source")

            _validate_unique_source_keys(con, enriched_source_parquet is not None)
            source_reconciliation = _validate_source_key_coverage(
                con, enriched_source_parquet is not None
            )
            _create_policy_tables(con, policy, mode)
            _create_source_view(con, signal_cols, enriched_cols, mode)
            _validate_source_causality(con, mode)
            _write_crop_day(con, stage / CROP_DAY_FILE)
            _write_pressure(con, stage / PRESSURE_FILE, policy, bool(enriched_cols))
            _create_acquisition_base(
                con, acquisition_cols, mode, policy, use_external=acquisition_parquet is not None
            )
            _validate_acquisition_causality(con, mode)
            acquisition_reconciliation = _acquisition_reconciliation(con)
            _write_acquisitions(con, stage / S2_FILE, policy)
            counts = _counts(con, stage)
        finally:
            con.close()

        inputs = {
            "generation_manifest": _file_metadata(manifest_path),
            "daily_causal_signals": _file_metadata(signals_path),
            "enriched_source": (
                dict(enriched_source_contract["source"])
                if enriched_source_contract is not None else None
            ),
            "enriched_source_manifest": (
                dict(enriched_source_contract["manifest"])
                if enriched_source_contract is not None else None
            ),
            "acquisition_source": (
                _file_metadata(acquisition_parquet)
                if acquisition_parquet is not None else None
            ),
        }
        artifacts = {
            "crop": {
                **_file_metadata(stage / CROP_DAY_FILE),
                "row_count": counts["crop_day_count"],
            },
            "pressure": {
                **_file_metadata(stage / PRESSURE_FILE),
                "row_count": counts["pressure_day_hazard_count"],
            },
            "s2": {
                **_file_metadata(stage / S2_FILE),
                "row_count": counts["spectral_acquisition_count"],
            },
        }
        source_as_of = (source_manifest.get("run") or {}).get("as_of_date")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "status": "complete", "immutable": True,
                "source_generation_id": (source_manifest.get("run") or {}).get("generation_id"),
                "source_as_of_date": source_as_of,
                # Consumers use this as the evidence release boundary.  Keep
                # the conventional alias as well so generic release validators
                # do not need generation-specific knowledge.
                "release_as_of": source_as_of,
                "as_of_date": source_as_of,
                "released_at": release_watermark,
            },
            "correction_policy": CORRECTION_POLICY,
            "availability": {
                "mode": mode,
                "diagnostic_reconstruction": mode == "reconstructed",
                "strict_source_timestamps_required": mode == "strict",
                "reconstruction_rule": (
                    "weather/stage use observation_date; spectral uses first daily appearance"
                    if mode == "reconstructed" else None
                ),
            },
            "policy": {
                "version": policy.version, "sha256": policy.source_sha256,
                "calibration_status": policy.calibration_status, "warning": policy.warning,
            },
            "inputs": inputs,
            "enriched_source_contract": enriched_source_contract,
            "reconciliation": {
                "source_field_day": source_reconciliation,
                "s2_acquisitions": acquisition_reconciliation,
            },
            "artifacts": artifacts,
            "counts": counts,
            "semantics": {
                "weather_and_spectral_observability_are_separate": True,
                "missing_weather_is_not_zero_pressure": True,
                "spectral_carry_forward_is_new_evidence": False,
                "reference_requires_prior_usable_acquisition": True,
                "simultaneous_hazard_lanes": True,
            },
            "outputs": {
                "crop_day_context": CROP_DAY_FILE,
                "field_day_pressure": PRESSURE_FILE,
                "field_s2_acquisition": S2_FILE,
            },
        }
        (stage / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, evidence_dir)
    return {
        "status": "written", "schema_version": SCHEMA_VERSION,
        "evidence_dir": str(evidence_dir), "availability_mode": mode,
        "released_at": release_watermark, **counts,
    }


def _create_source_view(
    con: duckdb.DuckDBPyConnection,
    signal_cols: set[str],
    enriched_cols: set[str],
    mode: str,
) -> None:
    join = "LEFT JOIN enriched_source e ON CAST(e.field_id AS VARCHAR) = CAST(s.field_id AS VARCHAR) AND CAST(e.observation_date AS DATE) = CAST(s.observation_date AS DATE)" if enriched_cols else ""
    values: dict[str, str] = {}
    for canonical, candidates in _RICH_COLUMNS.items():
        sql_type = "TIMESTAMP" if canonical.endswith("_at_raw") else (
            "DATE" if canonical in {"spectral_source_date", "planting_date"} else
            "BOOLEAN" if canonical == "s2_good_observation" else
            "VARCHAR" if canonical in {"s2_field_quality_flag", "stage_source"} else "DOUBLE"
        )
        values[canonical] = _best_expr(
            signal_cols, enriched_cols, candidates, sql_type
        )
    for name in ("ndvi", "ndmi", "psri"):
        values[name] = _best_expr(signal_cols, enriched_cols, (name,), "DOUBLE")
    con.execute(
        f"""
        CREATE TEMP VIEW source_raw_v4 AS
        SELECT
            CAST(s.field_id AS VARCHAR) AS field_id,
            CAST(s.observation_date AS DATE) AS observation_date,
            CAST(s.crop_instance_id AS VARCHAR) AS crop_instance_id,
            CAST(s.crop_name AS VARCHAR) AS crop_name,
            CAST(s.crop_season AS VARCHAR) AS crop_season,
            CAST(s.crop_stage AS VARCHAR) AS crop_stage_raw,
            CAST(s.stage_family AS VARCHAR) AS stage_family_raw,
            COALESCE(TRY_CAST(s.pressure_observed AS BOOLEAN), FALSE) AS pressure_observed_v1,
            TRY_CAST(s.risk_rank AS INTEGER) AS risk_rank_v1,
            CAST(s.risk_band AS VARCHAR) AS risk_band_v1,
            CAST(s.hazard_family AS VARCHAR) AS primary_hazard,
            {', '.join(f'{expr} AS {name}' for name, expr in values.items())}
        FROM signals_source s
        {join}
        """
    )
    if mode == "reconstructed":
        weather = "CAST(observation_date AS TIMESTAMP)"
        stage = "CAST(observation_date AS TIMESTAMP)"
        spectral = "CASE WHEN spectral_source_date IS NULL THEN NULL ELSE MIN(CAST(observation_date AS TIMESTAMP)) OVER (PARTITION BY field_id, spectral_source_date) END"
    else:
        weather = "weather_available_at_raw"
        stage = "stage_available_at_raw"
        spectral = "spectral_available_at_raw"
    con.execute(
        f"""
        CREATE TEMP VIEW source_v4 AS
        SELECT *, {weather} AS weather_available_at,
            {stage} AS stage_available_at,
            {spectral} AS spectral_available_at
        FROM source_raw_v4
        """
    )


def _write_crop_day(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    con.sql(
        f"""
        SELECT
            '{SCHEMA_VERSION}' AS schema_version,
            'crop_' || SUBSTR(MD5(field_id || '|' || crop_instance_id || '|' || CAST(observation_date AS VARCHAR)), 1, 24) AS record_id,
            field_id, crop_instance_id, observation_date,
            stage_available_at AS knowledge_time,
            crop_name, crop_season, crop_stage_raw, stage_family_raw,
            COALESCE(a.stage_bucket, 'unknown') AS stage_bucket,
            observation_date AS stage_effective_date,
            stage_available_at,
            COALESCE(NULLIF(stage_source, ''), 'generation_daily_causal_signals') AS stage_source,
            planting_date,
            COALESCE(NULLIF(TRIM(crop_name), ''), '') <> ''
                AND COALESCE(NULLIF(TRIM(crop_stage_raw), ''), '') <> '' AS crop_context_observed,
            CASE
                WHEN COALESCE(NULLIF(TRIM(crop_name), ''), '') = '' THEN 'crop_missing'
                WHEN COALESCE(NULLIF(TRIM(crop_stage_raw), ''), '') = '' THEN 'stage_missing'
                ELSE NULL
            END AS crop_context_missing_reason,
            av.mode AS availability_mode,
            p.policy_version, p.policy_sha256
        FROM source_v4 s
        LEFT JOIN stage_aliases_v4 a ON a.raw_stage = COALESCE(NULLIF(TRIM(BOTH '_' FROM REGEXP_REPLACE(LOWER(COALESCE(stage_family_raw, 'unknown')), '[^a-z0-9]+', '_', 'g')), ''), 'unknown')
        CROSS JOIN incident_policy_v4 p
        CROSS JOIN availability_mode_v4 av
        ORDER BY field_id, crop_instance_id, observation_date
        """
    ).write_parquet(str(path), compression="zstd")


def _write_pressure(
    con: duckdb.DuckDBPyConnection, path: Path, policy: IncidentPolicyV4,
    has_enriched: bool,
) -> None:
    observed = """
        CASE h.hazard_family
          WHEN 'drought' THEN spi_index IS NOT NULL
          WHEN 'ponding_flooding' THEN ponding_mm IS NOT NULL
          WHEN 'heat' THEN COALESCE(apparent_temperature, temperature) IS NOT NULL AND humidity IS NOT NULL
          WHEN 'damaging_wind' THEN wind_speed IS NOT NULL
        END
    """
    if not has_enriched:
        observed = f"(({observed}) OR (pressure_observed_v1 AND primary_hazard = h.hazard_family))"
    score = "CASE h.hazard_family WHEN 'drought' THEN drought_risk_score WHEN 'ponding_flooding' THEN ponding_risk_score WHEN 'heat' THEN heat_risk_score WHEN 'damaging_wind' THEN wind_risk_score END"
    rank = f"""
        CASE WHEN NOT pressure_observed THEN NULL
             WHEN pressure_score >= {policy.pressure_high} THEN 4
             WHEN pressure_score >= {policy.pressure_medium_high} THEN 3
             WHEN pressure_score >= {policy.pressure_low_medium} THEN 2
             WHEN pressure_score <= 0 THEN 0
             WHEN pressure_score IS NOT NULL THEN 1
             WHEN primary_hazard = hazard_family THEN risk_rank_v1
             ELSE 0 END
    """
    con.sql(
        f"""
        WITH lanes AS (
          SELECT s.*, h.hazard_family,
            COALESCE(({observed}), FALSE) AS pressure_observed,
            {score} AS pressure_score
          FROM source_v4 s CROSS JOIN hazard_families_v4 h
        ), ranked AS (
          SELECT *, {rank} AS pressure_rank FROM lanes
        )
        SELECT
          '{SCHEMA_VERSION}' AS schema_version,
          'pressure_' || SUBSTR(MD5(field_id || '|' || crop_instance_id || '|' || CAST(observation_date AS VARCHAR) || '|' || hazard_family), 1, 24) AS record_id,
          field_id, crop_instance_id, observation_date,
          observation_date AS pressure_observation_date,
          weather_available_at AS knowledge_time,
          weather_available_at, hazard_family, pressure_observed,
          pressure_score, pressure_rank,
          CASE pressure_rank WHEN 4 THEN 'HIGH' WHEN 3 THEN 'MED-HIGH'
            WHEN 2 THEN 'LOW-MED' WHEN 1 THEN 'LOW' WHEN 0 THEN 'NONE'
            ELSE 'UNKNOWN' END AS pressure_band,
          COALESCE(pressure_rank >= 2, FALSE) AS pressure_active,
          CASE WHEN pressure_observed THEN NULL ELSE 'required_weather_driver_missing' END AS pressure_missing_reason,
          primary_hazard, spi_index, ponding_mm, apparent_temperature,
          temperature, humidity, wind_speed, wind_gust,
          av.mode AS availability_mode, p.policy_version, p.policy_sha256
        FROM ranked
        CROSS JOIN incident_policy_v4 p CROSS JOIN availability_mode_v4 av
        ORDER BY observation_date, field_id, crop_instance_id, hazard_family
        """
    ).write_parquet(str(path), compression="zstd")


def _create_acquisition_base(
    con: duckdb.DuckDBPyConnection, columns: set[str], mode: str,
    policy: IncidentPolicyV4, *, use_external: bool,
) -> None:
    con.execute(
        """
        CREATE TEMP VIEW derived_acquisition_candidates_v4 AS
        SELECT field_id, spectral_source_date,
          MIN(spectral_available_at) AS spectral_available_at,
          MIN(observation_date) AS first_seen_observation_date,
          ARG_MIN(valid_pixel_fraction, observation_date) AS valid_pixel_fraction,
          ARG_MIN(cloud_pct, observation_date) AS cloud_pct,
          ARG_MIN(s2_field_quality_flag, observation_date) AS s2_field_quality_flag,
          ARG_MIN(s2_good_observation, observation_date) AS s2_good_observation,
          ARG_MIN(ndvi, observation_date) AS ndvi,
          ARG_MIN(ndmi, observation_date) AS ndmi,
          ARG_MIN(psri, observation_date) AS psri
        FROM source_v4 WHERE spectral_source_date IS NOT NULL
        GROUP BY field_id, spectral_source_date
        """
    )
    if use_external:
        source_col = _first(columns, ("sentinel_observation_date", "spectral_source_date", "prediction_observation_date"))
        if source_col is None:
            raise ValueError("acquisition source is missing a Sentinel source date")
        available_col = _first(columns, ("spectral_available_at", "sentinel_available_at", "available_at"))
        if mode == "strict" and available_col is None:
            raise ValueError("strict availability requires spectral_available_at in acquisition source")
        first_seen_col = _first(
            columns, ("first_seen_observation_date", "observation_date")
        )
        first_seen = (
            f"TRY_CAST(a.{_q(first_seen_col)} AS DATE)" if first_seen_col
            else f"CAST(a.{_q(source_col)} AS DATE)"
        )
        available = (
            f"TRY_CAST(a.{_q(available_col)} AS TIMESTAMP)"
            if mode == "strict" and available_col
            else f"CAST({first_seen} AS TIMESTAMP)"
        )
        fields = {name: _external_expr(columns, (name,), "DOUBLE") for name in ("valid_pixel_fraction", "cloud_pct", "ndvi", "ndmi", "psri")}
        quality = _external_expr(columns, ("s2_field_quality_flag",), "VARCHAR")
        good = _external_expr(columns, ("s2_good_observation",), "BOOLEAN")
        con.execute(
            f"""
            CREATE TEMP VIEW external_acquisition_rows_v4 AS
            SELECT CAST(a.field_id AS VARCHAR) AS field_id,
              CAST(a.{_q(source_col)} AS DATE) AS spectral_source_date,
              {available} AS spectral_available_at,
              {first_seen} AS first_seen_observation_date,
              {fields['valid_pixel_fraction']} AS valid_pixel_fraction,
              {fields['cloud_pct']} AS cloud_pct,
              {quality} AS s2_field_quality_flag,
              {good} AS s2_good_observation,
              {fields['ndvi']} AS ndvi, {fields['ndmi']} AS ndmi, {fields['psri']} AS psri
            FROM acquisition_source a
            """
        )
        con.execute(
            """
            CREATE TEMP VIEW external_acquisition_candidates_v4 AS
            SELECT field_id, spectral_source_date,
              MIN(spectral_available_at) AS spectral_available_at,
              MIN(first_seen_observation_date) AS first_seen_observation_date,
              MIN(valid_pixel_fraction) AS valid_pixel_fraction,
              MAX(cloud_pct) AS cloud_pct,
              MIN(s2_field_quality_flag) AS s2_field_quality_flag,
              BOOL_AND(s2_good_observation) AS s2_good_observation,
              ARG_MIN(ndvi, spectral_available_at) AS ndvi,
              ARG_MIN(ndmi, spectral_available_at) AS ndmi,
              ARG_MIN(psri, spectral_available_at) AS psri
            FROM external_acquisition_rows_v4
            WHERE spectral_source_date IS NOT NULL
            GROUP BY field_id, spectral_source_date
            """
        )
    else:
        con.execute(
            """
            CREATE TEMP VIEW external_acquisition_candidates_v4 AS
            SELECT CAST(NULL AS VARCHAR) AS field_id,
              CAST(NULL AS DATE) AS spectral_source_date,
              CAST(NULL AS TIMESTAMP) AS spectral_available_at,
              CAST(NULL AS DATE) AS first_seen_observation_date,
              CAST(NULL AS DOUBLE) AS valid_pixel_fraction,
              CAST(NULL AS DOUBLE) AS cloud_pct,
              CAST(NULL AS VARCHAR) AS s2_field_quality_flag,
              CAST(NULL AS BOOLEAN) AS s2_good_observation,
              CAST(NULL AS DOUBLE) AS ndvi,
              CAST(NULL AS DOUBLE) AS ndmi,
              CAST(NULL AS DOUBLE) AS psri
            WHERE FALSE
            """
        )
    con.execute(
        """
        CREATE TEMP VIEW merged_acquisition_candidates_v4 AS
        SELECT COALESCE(e.field_id, d.field_id) AS field_id,
          COALESCE(e.spectral_source_date, d.spectral_source_date)
            AS spectral_source_date,
          COALESCE(e.spectral_available_at, d.spectral_available_at)
            AS spectral_available_at,
          COALESCE(e.first_seen_observation_date, d.first_seen_observation_date)
            AS first_seen_observation_date,
          COALESCE(e.valid_pixel_fraction, d.valid_pixel_fraction)
            AS valid_pixel_fraction,
          COALESCE(e.cloud_pct, d.cloud_pct) AS cloud_pct,
          COALESCE(e.s2_field_quality_flag, d.s2_field_quality_flag)
            AS s2_field_quality_flag,
          COALESCE(e.s2_good_observation, d.s2_good_observation)
            AS s2_good_observation,
          COALESCE(e.ndvi, d.ndvi) AS ndvi,
          COALESCE(e.ndmi, d.ndmi) AS ndmi,
          COALESCE(e.psri, d.psri) AS psri,
          CASE
            WHEN e.field_id IS NOT NULL AND d.field_id IS NOT NULL
              THEN 'external_and_derived'
            WHEN e.field_id IS NOT NULL THEN 'external_attempt'
            ELSE 'derived_daily_source'
          END AS acquisition_origin
        FROM derived_acquisition_candidates_v4 d
        FULL OUTER JOIN external_acquisition_candidates_v4 e
          ON e.field_id = d.field_id
         AND e.spectral_source_date = d.spectral_source_date
        """
    )
    con.execute(
        """
        CREATE TEMP VIEW acquisition_candidates_v4 AS
        SELECT m.*, crop.crop_instance_id,
          crop.observation_date AS crop_assignment_effective_date,
          crop.stage_available_at AS crop_assignment_available_at,
          CASE
            WHEN m.spectral_available_at IS NULL THEN NULL
            WHEN crop.stage_available_at IS NULL THEN m.spectral_available_at
            WHEN crop.stage_available_at > m.spectral_available_at
              THEN crop.stage_available_at
            ELSE m.spectral_available_at
          END AS knowledge_time
        FROM merged_acquisition_candidates_v4 m
        LEFT JOIN LATERAL (
          SELECT c.crop_instance_id, c.observation_date, c.stage_available_at
          FROM source_v4 c
          WHERE c.field_id = m.field_id
            AND c.observation_date <= m.spectral_source_date
          ORDER BY c.observation_date DESC, c.stage_available_at DESC,
            c.crop_instance_id
          LIMIT 1
        ) crop ON TRUE
        """
    )
    rejected_flags = ",".join("'" + value.replace("'", "''") + "'" for value in policy.rejected_quality_flags)
    con.execute(
        f"""
        CREATE TEMP TABLE acquisition_base_v4 AS
        SELECT *,
          's2_' || SUBSTR(MD5(field_id || '|' || COALESCE(crop_instance_id, '') || '|' || CAST(spectral_source_date AS VARCHAR)), 1, 24) AS acquisition_id,
          CASE
            WHEN s2_good_observation IS NULL
              AND valid_pixel_fraction IS NULL
              AND cloud_pct IS NULL
              AND NULLIF(TRIM(COALESCE(s2_field_quality_flag, '')), '') IS NULL
              THEN 'rejected_qa_unknown'
            WHEN COALESCE(s2_good_observation, TRUE) = FALSE THEN 'rejected_no_valid_pixels'
            WHEN valid_pixel_fraction IS NOT NULL AND valid_pixel_fraction < {policy.minimum_valid_pixel_fraction} THEN 'rejected_sparse'
            WHEN cloud_pct IS NOT NULL AND cloud_pct > {policy.maximum_cloud_pct} THEN 'rejected_cloud'
            WHEN LOWER(COALESCE(s2_field_quality_flag, '')) IN ({rejected_flags}) THEN 'rejected_quality'
            WHEN ndvi IS NULL AND ndmi IS NULL AND psri IS NULL THEN 'rejected_missing_indices'
            ELSE 'usable' END AS acquisition_status
        FROM acquisition_candidates_v4
        WHERE spectral_source_date IS NOT NULL
        """
    )


def _write_acquisitions(
    con: duckdb.DuckDBPyConnection, path: Path, policy: IncidentPolicyV4
) -> None:
    con.execute(
        f"""
        CREATE TEMP VIEW acquisition_referenced_v4 AS
        SELECT c.*, r.acquisition_id AS reference_acquisition_id,
          r.spectral_source_date AS reference_source_date,
          CASE WHEN c.acquisition_status = 'usable' THEN c.ndvi - r.ndvi END AS ndvi_delta,
          CASE WHEN c.acquisition_status = 'usable' THEN c.ndmi - r.ndmi END AS ndmi_delta,
          CASE WHEN c.acquisition_status = 'usable' THEN c.psri - r.psri END AS psri_delta
        FROM acquisition_base_v4 c
        LEFT JOIN acquisition_base_v4 r
         ON r.field_id = c.field_id
         AND r.crop_instance_id = c.crop_instance_id
         AND c.acquisition_status = 'usable'
         AND r.acquisition_status = 'usable'
         AND r.knowledge_time <= c.knowledge_time
         AND r.spectral_source_date < c.spectral_source_date
         AND DATE_DIFF('day', r.spectral_source_date, c.spectral_source_date)
             BETWEEN {policy.reference_min_days} AND {policy.reference_max_days}
        QUALIFY ROW_NUMBER() OVER (
          PARTITION BY c.field_id, c.crop_instance_id, c.spectral_source_date
          ORDER BY r.spectral_source_date DESC NULLS LAST
        ) = 1
        """
    )
    con.sql(
        f"""
        WITH classified AS (
          SELECT *,
            CASE
              WHEN acquisition_status <> 'usable' THEN 'not_evaluable'
              WHEN reference_acquisition_id IS NULL THEN 'insufficient_reference'
              WHEN ndvi_delta <= {policy.severe_ndvi_delta}
                OR ndmi_delta <= {policy.severe_ndmi_delta}
                OR psri_delta >= {policy.severe_psri_delta} THEN 'severe_decline'
              WHEN ndvi_delta <= {policy.medium_ndvi_delta}
                OR ndmi_delta <= {policy.medium_ndmi_delta}
                OR psri_delta >= {policy.medium_psri_delta} THEN 'medium_decline'
              WHEN ndvi_delta >= {policy.recovery_ndvi_delta}
                OR ndmi_delta >= {policy.recovery_ndmi_delta}
                OR psri_delta <= {policy.recovery_psri_delta} THEN 'recovery'
              ELSE 'no_material_change' END AS response_class
          FROM acquisition_referenced_v4
        )
        SELECT '{SCHEMA_VERSION}' AS schema_version,
          acquisition_id, field_id, crop_instance_id, spectral_source_date,
          knowledge_time, spectral_available_at,
          crop_assignment_effective_date, crop_assignment_available_at,
          acquisition_origin,
          first_seen_observation_date, TRUE AS acquisition_attempted,
          acquisition_status = 'usable' AS spectral_usable,
          acquisition_status, valid_pixel_fraction, cloud_pct,
          s2_field_quality_flag, ndvi, ndmi, psri,
          reference_acquisition_id, reference_source_date,
          ndvi_delta, ndmi_delta, psri_delta, response_class,
          CASE
            WHEN response_class IN ('medium_decline', 'severe_decline') THEN 'adverse_index_change'
            WHEN response_class = 'recovery' THEN 'recovery_index_change'
            ELSE 'none' END AS response_evidence,
          acquisition_status = 'usable' AND reference_acquisition_id IS NOT NULL
            AS new_response_evidence,
          av.mode AS availability_mode, p.policy_version, p.policy_sha256
        FROM classified
        CROSS JOIN incident_policy_v4 p CROSS JOIN availability_mode_v4 av
        ORDER BY spectral_source_date, field_id, crop_instance_id
        """
    ).write_parquet(str(path), compression="zstd")


def _validate_source_causality(con: duckdb.DuckDBPyConnection, mode: str) -> None:
    duplicate = int(con.execute(
        "SELECT COUNT(*) FROM (SELECT field_id, crop_instance_id, observation_date, COUNT(*) FROM source_v4 GROUP BY 1,2,3 HAVING COUNT(*) > 1)"
    ).fetchone()[0])
    future, negative, mismatch = con.execute(
        """
        SELECT COUNT_IF(spectral_source_date > observation_date),
          COUNT_IF(spectral_echo_days < 0),
          COUNT_IF(
            (spectral_source_date IS NULL) <> (spectral_echo_days IS NULL)
            OR (spectral_source_date IS NOT NULL AND spectral_echo_days IS NOT NULL
              AND CAST(spectral_echo_days AS INTEGER)
                <> DATE_DIFF('day', spectral_source_date, observation_date))
          )
        FROM source_v4
        """
    ).fetchone()
    if duplicate or future or negative or mismatch:
        raise ValueError(
            "Incident V4 causal source failed: "
            f"duplicate_keys={duplicate}, future_spectral_source={future}, "
            f"negative_echo={negative}, echo_mismatch={mismatch}"
        )
    if mode == "strict":
        invalid = int(con.execute(
            """
            SELECT COUNT(*) FROM source_v4
            WHERE weather_available_at IS NULL OR stage_available_at IS NULL
               OR weather_available_at < CAST(observation_date AS TIMESTAMP)
               OR stage_available_at < CAST(observation_date AS TIMESTAMP)
               OR (spectral_source_date IS NOT NULL AND spectral_available_at IS NULL)
               OR spectral_available_at < CAST(spectral_source_date AS TIMESTAMP)
            """
        ).fetchone()[0])
        if invalid:
            raise ValueError(f"strict availability has {invalid} missing or impossible timestamps")


def _validate_acquisition_causality(con: duckdb.DuckDBPyConnection, mode: str) -> None:
    invalid = int(con.execute(
        """
        SELECT COUNT(*) FROM acquisition_base_v4
        WHERE crop_instance_id IS NULL OR spectral_available_at IS NULL
           OR knowledge_time IS NULL
           OR spectral_available_at < CAST(spectral_source_date AS TIMESTAMP)
           OR crop_assignment_effective_date > spectral_source_date
           OR knowledge_time < spectral_available_at
           OR knowledge_time < crop_assignment_available_at
        """
    ).fetchone()[0])
    if invalid:
        label = "strict" if mode == "strict" else "reconstructed"
        raise ValueError(f"{label} acquisition source has {invalid} invalid causal rows")


def _acquisition_reconciliation(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    row = con.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE acquisition_origin = 'derived_daily_source'),
          COUNT(*) FILTER (WHERE acquisition_origin = 'external_attempt'),
          COUNT(*) FILTER (WHERE acquisition_origin = 'external_and_derived'),
          COUNT(*)
        FROM acquisition_base_v4
        """
    ).fetchone()
    return {
        "derived_only_count": int(row[0] or 0),
        "external_only_count": int(row[1] or 0),
        "external_and_derived_count": int(row[2] or 0),
        "published_acquisition_count": int(row[3] or 0),
    }


def _validate_unique_source_keys(con: duckdb.DuckDBPyConnection, enriched: bool) -> None:
    duplicate_signals = int(con.execute(
        "SELECT COUNT(*) FROM (SELECT CAST(field_id AS VARCHAR), CAST(observation_date AS DATE), COUNT(*) FROM signals_source GROUP BY 1,2 HAVING COUNT(*) > 1)"
    ).fetchone()[0])
    duplicate_enriched = 0
    if enriched:
        duplicate_enriched = int(con.execute(
            "SELECT COUNT(*) FROM (SELECT CAST(field_id AS VARCHAR), CAST(observation_date AS DATE), COUNT(*) FROM enriched_source GROUP BY 1,2 HAVING COUNT(*) > 1)"
        ).fetchone()[0])
    if duplicate_signals or duplicate_enriched:
        raise ValueError(
            f"Incident V4 source keys are not unique: signals={duplicate_signals}, enriched={duplicate_enriched}"
        )


def _validate_source_key_coverage(
    con: duckdb.DuckDBPyConnection, enriched: bool
) -> dict[str, Any]:
    signal_count = int(con.execute("SELECT COUNT(*) FROM signals_source").fetchone()[0])
    if not enriched:
        return {
            "signal_field_day_count": signal_count,
            "enriched_field_day_count": None,
            "missing_enriched_field_day_count": None,
            "extra_enriched_field_day_count": None,
            "exact_key_coverage": None,
        }
    enriched_count, missing, extra = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM enriched_source),
          (SELECT COUNT(*) FROM signals_source s
             ANTI JOIN enriched_source e
               ON CAST(e.field_id AS VARCHAR) = CAST(s.field_id AS VARCHAR)
              AND CAST(e.observation_date AS DATE) = CAST(s.observation_date AS DATE)),
          (SELECT COUNT(*) FROM enriched_source e
             ANTI JOIN signals_source s
               ON CAST(e.field_id AS VARCHAR) = CAST(s.field_id AS VARCHAR)
              AND CAST(e.observation_date AS DATE) = CAST(s.observation_date AS DATE))
        """
    ).fetchone()
    reconciliation = {
        "signal_field_day_count": signal_count,
        "enriched_field_day_count": int(enriched_count),
        "missing_enriched_field_day_count": int(missing),
        "extra_enriched_field_day_count": int(extra),
        "exact_key_coverage": not missing and not extra,
    }
    if missing or extra:
        raise ValueError(
            "Incident V4 enriched source does not exactly cover generation field/day "
            f"keys: missing={missing}, extra={extra}"
        )
    return reconciliation


def _connection(
    threads: int, memory_limit: str | None, temp_dir: Path | None
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=?", [int(threads)])
    if memory_limit:
        con.execute("SET memory_limit=?", [str(memory_limit)])
    if temp_dir:
        resolved = temp_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        con.execute("SET temp_directory=?", [str(resolved)])
    return con


def _create_policy_tables(
    con: duckdb.DuckDBPyConnection, policy: IncidentPolicyV4, mode: str
) -> None:
    con.execute("CREATE TEMP TABLE incident_policy_v4(policy_version VARCHAR, policy_sha256 VARCHAR)")
    con.execute("INSERT INTO incident_policy_v4 VALUES (?, ?)", [policy.version, policy.source_sha256])
    con.execute("CREATE TEMP TABLE availability_mode_v4(mode VARCHAR)")
    con.execute("INSERT INTO availability_mode_v4 VALUES (?)", [mode])
    con.execute("CREATE TEMP TABLE hazard_families_v4(hazard_family VARCHAR)")
    con.executemany("INSERT INTO hazard_families_v4 VALUES (?)", [(x,) for x in policy.hazard_families])
    con.execute("CREATE TEMP TABLE stage_aliases_v4(raw_stage VARCHAR PRIMARY KEY, stage_bucket VARCHAR)")
    con.executemany("INSERT INTO stage_aliases_v4 VALUES (?, ?)", list(policy.stage_aliases))


def _register_parquet(
    con: duckdb.DuckDBPyConnection, name: str, path: Path
) -> set[str]:
    relation = con.read_parquet(str(path))
    relation.create_view(name)
    return set(relation.columns)


def _best_expr(
    signal_cols: set[str], enriched_cols: set[str], candidates: Iterable[str],
    sql_type: str,
) -> str:
    enriched = _first(enriched_cols, candidates)
    signal = _first(signal_cols, candidates)
    expressions = []
    if enriched:
        expressions.append(f"TRY_CAST(e.{_q(enriched)} AS {sql_type})")
    if signal:
        expressions.append(f"TRY_CAST(s.{_q(signal)} AS {sql_type})")
    return "COALESCE(" + ", ".join(expressions) + ")" if len(expressions) > 1 else (
        expressions[0] if expressions else f"CAST(NULL AS {sql_type})"
    )


def _external_expr(columns: set[str], candidates: Iterable[str], sql_type: str) -> str:
    name = _first(columns, candidates)
    return f"TRY_CAST(a.{_q(name)} AS {sql_type})" if name else f"CAST(NULL AS {sql_type})"


def _first(columns: set[str], candidates: Iterable[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _require_columns(columns: set[str], required: set[str], label: str) -> None:
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _generation_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read immutable generation manifest: {path}") from exc
    run = payload.get("run") or {}
    if run.get("status") != "complete" or not bool(run.get("immutable", True)):
        raise ValueError("Incident V4 requires a complete immutable generation")
    if not str(run.get("as_of_date") or "").strip():
        raise ValueError("Incident V4 generation manifest is missing run.as_of_date")
    return payload


def validate_enriched_source_v4(
    path: Path, *, expected_mode: str | None = None
) -> dict[str, Any]:
    """Validate and bind an immutable enriched source to its availability clock."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Enriched source sidecar does not exist: {manifest_path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read enriched source sidecar: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Enriched source sidecar root must be an object")
    availability = payload.get("availability") or {}
    mode = str(availability.get("mode") or "")
    if mode not in {"strict", "reconstructed"}:
        raise ValueError("Enriched source sidecar has an invalid availability mode")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(
            "Enriched source availability mode does not match the requested evidence "
            f"mode: source={mode}, requested={expected_mode}"
        )
    released_at = normalize_released_at(str(payload.get("released_at") or ""))
    validate_correction_policy(payload.get("correction_policy"))
    output = payload.get("output") or {}
    source_metadata = _file_metadata(source)
    if (
        payload.get("schema_version") != "incident-enriched-source-v4/1"
        or payload.get("status") != "complete"
        or payload.get("immutable") is not True
        or str(output.get("name") or "") != source.name
        or output.get("size_bytes") is None
        or int(output["size_bytes"]) != source_metadata["size_bytes"]
        or str(output.get("sha256") or "") != source_metadata["sha256"]
    ):
        raise ValueError("Enriched source sidecar does not match its immutable parquet")
    with duckdb.connect(":memory:") as connection:
        columns = set(connection.read_parquet(str(source)).columns)
        availability_columns = {
            "weather_available_at", "spectral_available_at", "stage_available_at",
        }
        missing = sorted(availability_columns - columns)
        if missing:
            raise ValueError(
                "Enriched source is missing availability columns: " + ", ".join(missing)
            )
        post_release = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?) WHERE "
                "TRY_CAST(weather_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ) "
                "OR TRY_CAST(stage_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ) "
                "OR TRY_CAST(spectral_available_at AS TIMESTAMPTZ) > CAST(? AS TIMESTAMPTZ)",
                [str(source), released_at, released_at, released_at],
            ).fetchone()[0]
        )
    if post_release:
        raise ValueError(
            f"Enriched source contains {post_release} availability timestamps after "
            "released_at"
        )
    return {
        "schema_version": payload["schema_version"],
        "availability_mode": mode,
        "diagnostic_reconstruction": bool(
            availability.get("diagnostic_reconstruction")
        ),
        "released_at": released_at,
        "source": source_metadata,
        "manifest": _file_metadata(manifest_path),
    }


def _optional_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _counts(con: duckdb.DuckDBPyConnection, stage: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, name in (
        ("crop_day_count", CROP_DAY_FILE),
        ("pressure_day_hazard_count", PRESSURE_FILE),
        ("spectral_acquisition_count", S2_FILE),
    ):
        result[key] = int(con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(stage / name)]).fetchone()[0])
    return result


__all__ = [
    "CROP_DAY_FILE", "MANIFEST_FILE", "PRESSURE_FILE", "S2_FILE",
    "SCHEMA_VERSION", "build_incident_context_v4", "validate_enriched_source_v4",
]
