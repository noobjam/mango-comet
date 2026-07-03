"""Causal one-anchor-per-event features for V2 archetype discovery.

``build_event_anchors`` is convenient for bounded fixtures and analysis.  Full
VM runs should use ``write_event_anchors``: DuckDB writes the result directly to
Parquet, so Python never materializes the event table.
"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd


FEATURE_VERSION = "causal_event_anchor_features_v2"
MATURITY_USABLE_DAYS = 21
MINIMUM_USABLE_DAYS = 7


def _feature(name: str, group: str, kind: str, definition: str) -> dict[str, str]:
    return {"name": name, "group": group, "kind": kind, "definition": definition}


FEATURE_SCHEMA: dict[str, Any] = {
    "version": FEATURE_VERSION,
    "anchor_contract": "21st_usable_day_or_eligible_early_closure_v1",
    "features": [
        _feature("peak_risk_rank", "risk_shape", "continuous", "Maximum usable-day risk rank."),
        _feature("mean_risk_rank", "risk_shape", "continuous", "Mean usable-day risk rank."),
        _feature(
            "risk_slope", "risk_shape", "continuous",
            "OLS risk slope over calendar time normalized from first usable day to anchor.",
        ),
        _feature(
            "elevated_day_fraction", "risk_shape", "bounded",
            "Fraction of usable days at MED-HIGH or HIGH (rank >= 3).",
        ),
        _feature("high_day_fraction", "risk_shape", "bounded", "Fraction of usable days at HIGH."),
        _feature(
            "longest_elevated_run_fraction", "risk_shape", "bounded",
            "Longest consecutive rank >= 3 run in usable-row order, divided by usable days.",
        ),
        _feature("attributed_decline_any", "response", "binary", "Any attributed decline by anchor."),
        _feature(
            "attributed_severe_decline_any", "response", "binary",
            "Any attributed severe decline by anchor.",
        ),
        _feature(
            "attributed_decline_day_fraction", "response", "bounded",
            "Attributed decline rows divided by usable days, clipped to [0, 1].",
        ),
        _feature(
            "first_attributed_decline_position", "response", "bounded",
            "First decline calendar position from event start to anchor, clipped to [0, 1].",
        ),
        _feature(
            "attributed_recovery_after_decline", "response", "binary",
            "An attributed recovery occurs strictly after the first attributed decline.",
        ),
        _feature(
            "worst_attributed_ndvi_delta", "spectral", "continuous",
            "Minimum NDVI delta on an attributed decline or recovery row; NULL when missing.",
        ),
        _feature("worst_attributed_ndvi_delta_missing", "spectral", "binary", "No attributed NDVI delta."),
        _feature(
            "worst_attributed_ndmi_delta", "spectral", "continuous",
            "Minimum NDMI delta on an attributed decline or recovery row; NULL when missing.",
        ),
        _feature("worst_attributed_ndmi_delta_missing", "spectral", "binary", "No attributed NDMI delta."),
        _feature(
            "worst_attributed_psri_delta", "spectral", "continuous",
            "Maximum PSRI delta on an attributed decline or recovery row; NULL when missing.",
        ),
        _feature("worst_attributed_psri_delta_missing", "spectral", "binary", "No attributed PSRI delta."),
        _feature(
            "hazard_intensity", "hazard_intensity", "continuous",
            "One hazard-canonical extreme: min SPI or max ponding/apparent temperature/wind; NULL when missing.",
        ),
        _feature("hazard_intensity_missing", "hazard_intensity", "binary", "Canonical hazard intensity unavailable."),
        _feature(
            "usable_days_fraction", "duration", "bounded",
            "Usable pressure-observed days divided by 21.",
        ),
    ],
}

MODEL_FEATURE_COLUMNS = tuple(item["name"] for item in FEATURE_SCHEMA["features"])
DIAGNOSTIC_COLUMNS = (
    "anchor_window_day_count",
    "usable_day_count",
    "missing_pressure_day_count",
    "pressure_observed_coverage",
    "evidence_max_date",
    "spectral_source_max_date",
    "post_anchor_row_count",
)

_REQUIRED_COLUMNS = {
    "events_source": {
        "event_id", "field_id", "event_start_date", "event_end_date",
        "event_state", "hazard_signature", "right_censored",
    },
    "memberships_source": {
        "event_id", "field_id", "crop_instance_id", "observation_date",
        "event_state", "daily_pressure_rank", "daily_response_class", "pressure_observed",
    },
    "signals_source": {
        "field_id", "crop_instance_id", "observation_date", "ndvi_delta", "ndmi_delta",
        "psri_delta", "spi_index", "ponding_mm", "apparent_temperature", "temperature",
        "wind_speed",
    },
}


_ANCHOR_QUERY = f"""
WITH event_scope AS (
    SELECT
        CAST(e.event_id AS VARCHAR) AS event_id,
        CAST(e.field_id AS VARCHAR) AS field_id,
        CAST(e.hazard_signature AS VARCHAR) AS hazard_family,
        CAST(e.event_start_date AS DATE) AS event_start_date,
        CAST(e.event_end_date AS DATE) AS event_end_date,
        UPPER(CAST(e.event_state AS VARCHAR)) AS final_event_state,
        p.cutoff,
        CAST(e.event_end_date AS DATE) IS NOT NULL
            AND CAST(e.event_end_date AS DATE) <= p.cutoff AS closed_as_of_cutoff
    FROM events_source e
    CROSS JOIN anchor_parameters p
    WHERE CAST(e.event_start_date AS DATE) <= p.cutoff
), membership_rows AS (
    SELECT
        e.*,
        CAST(m.observation_date AS DATE) AS observation_date,
        UPPER(CAST(m.event_state AS VARCHAR)) AS observed_event_state,
        TRY_CAST(m.daily_pressure_rank AS DOUBLE) AS risk_rank,
        COALESCE(TRY_CAST(m.pressure_observed AS BOOLEAN), FALSE)
            AND TRY_CAST(m.daily_pressure_rank AS DOUBLE) IS NOT NULL AS usable,
        LOWER(COALESCE(CAST(m.daily_response_class AS VARCHAR), '')) AS response_class,
        TRY_CAST(s.ndvi_delta AS DOUBLE) AS ndvi_delta,
        TRY_CAST(s.ndmi_delta AS DOUBLE) AS ndmi_delta,
        TRY_CAST(s.psri_delta AS DOUBLE) AS psri_delta,
        CAST(s.spectral_source_date_v2 AS DATE) AS spectral_source_date,
        TRY_CAST(s.spi_index AS DOUBLE) AS spi_index,
        TRY_CAST(s.ponding_mm AS DOUBLE) AS ponding_mm,
        COALESCE(
            TRY_CAST(s.apparent_temperature AS DOUBLE), TRY_CAST(s.temperature AS DOUBLE)
        ) AS apparent_temperature,
        TRY_CAST(s.wind_speed AS DOUBLE) AS wind_speed
    FROM event_scope e
    JOIN memberships_source m USING (event_id)
    LEFT JOIN signals_source s
      ON CAST(s.field_id AS VARCHAR) = CAST(m.field_id AS VARCHAR)
     AND CAST(s.crop_instance_id AS VARCHAR) = CAST(m.crop_instance_id AS VARCHAR)
     AND CAST(s.observation_date AS DATE) = CAST(m.observation_date AS DATE)
    WHERE CAST(m.observation_date AS DATE) <= e.cutoff
), ranked AS (
    SELECT *,
        SUM(CASE WHEN usable THEN 1 ELSE 0 END) OVER (
            PARTITION BY event_id ORDER BY observation_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS usable_ordinal
    FROM membership_rows
), event_progress AS (
    SELECT
        event_id,
        COUNT_IF(usable) AS usable_days_before_cutoff,
        MIN(CASE WHEN usable AND usable_ordinal = {MATURITY_USABLE_DAYS}
                 THEN observation_date END) AS day_21_date
    FROM ranked
    GROUP BY event_id
), candidates AS (
    SELECT
        e.*,
        COALESCE(p.usable_days_before_cutoff, 0) AS usable_days_before_cutoff,
        p.day_21_date,
        CASE
            WHEN p.day_21_date IS NOT NULL THEN p.day_21_date
            WHEN e.closed_as_of_cutoff THEN e.event_end_date
            ELSE NULL
        END AS anchor_date,
        CASE
            WHEN p.day_21_date IS NOT NULL THEN 'day_21'
            WHEN e.closed_as_of_cutoff THEN 'early_closure'
            ELSE NULL
        END AS anchor_kind
    FROM event_scope e
    LEFT JOIN event_progress p USING (event_id)
), window_rows AS (
    SELECT r.*, c.anchor_date, c.anchor_kind
    FROM ranked r
    JOIN candidates c USING (event_id)
    WHERE r.observation_date <= COALESCE(c.anchor_date, c.cutoff)
), window_summary AS (
    SELECT
        event_id,
        COUNT(*) AS anchor_window_day_count,
        COUNT_IF(usable) AS usable_day_count,
        COUNT_IF(NOT usable) AS missing_pressure_day_count,
        MAX(observation_date) AS evidence_max_date,
        MIN(CASE WHEN usable THEN observation_date END) AS first_usable_date,
        BOOL_OR(observed_event_state IN ('ACTIVE', 'SEVERE')) AS reached_active_or_severe,
        MIN(CASE WHEN response_class IN ('medium_decline', 'severe_decline')
                 THEN observation_date END) AS first_decline_date
    FROM window_rows
    GROUP BY event_id
), classified AS (
    SELECT
        c.*,
        COALESCE(w.anchor_window_day_count, 0) AS anchor_window_day_count,
        COALESCE(w.usable_day_count, 0) AS usable_day_count,
        COALESCE(w.missing_pressure_day_count, 0) AS missing_pressure_day_count,
        w.evidence_max_date,
        w.first_usable_date,
        w.first_decline_date,
        COALESCE(w.reached_active_or_severe, FALSE) AS reached_active_or_severe,
        CASE
            WHEN c.closed_as_of_cutoff
             AND c.final_event_state = 'CLOSED_SEASON_BOUNDARY'
             AND c.day_21_date IS NULL THEN 'season_boundary_before_maturity'
            WHEN NOT COALESCE(w.reached_active_or_severe, FALSE) THEN 'watch_only'
            WHEN COALESCE(w.usable_day_count, 0) < {MINIMUM_USABLE_DAYS}
                THEN 'insufficient_evidence'
            WHEN NOT c.closed_as_of_cutoff AND c.day_21_date IS NULL
                THEN 'insufficient_evidence'
            ELSE 'eligible'
        END AS anchor_outcome
    FROM candidates c
    LEFT JOIN window_summary w USING (event_id)
), risk_rows AS (
    SELECT
        w.*,
        CASE
            WHEN DATE_DIFF('day', c.first_usable_date, c.anchor_date) <= 0 THEN 0.0
            ELSE DATE_DIFF('day', c.first_usable_date, w.observation_date)::DOUBLE
                 / DATE_DIFF('day', c.first_usable_date, c.anchor_date)
        END AS normalized_time
    FROM window_rows w
    JOIN classified c USING (event_id)
    WHERE w.usable AND c.anchor_outcome = 'eligible'
), risk_marked AS (
    SELECT *,
        SUM(CASE WHEN risk_rank < 3 THEN 1 ELSE 0 END) OVER (
            PARTITION BY event_id ORDER BY observation_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS elevated_run_group
    FROM risk_rows
), risk_summary AS (
    SELECT
        event_id,
        MAX(risk_rank) AS peak_risk_rank,
        AVG(risk_rank) AS mean_risk_rank,
        COALESCE(REGR_SLOPE(risk_rank, normalized_time), 0.0) AS risk_slope,
        COUNT_IF(risk_rank >= 3)::DOUBLE / COUNT(*) AS elevated_day_fraction,
        COUNT_IF(risk_rank >= 4)::DOUBLE / COUNT(*) AS high_day_fraction
    FROM risk_rows
    GROUP BY event_id
), elevated_runs AS (
    SELECT event_id, elevated_run_group, COUNT(*) AS run_days
    FROM risk_marked
    WHERE risk_rank >= 3
    GROUP BY event_id, elevated_run_group
), longest_runs AS (
    SELECT event_id, MAX(run_days) AS longest_elevated_run
    FROM elevated_runs
    GROUP BY event_id
), response_summary AS (
    SELECT
        c.event_id,
        COUNT_IF(w.response_class IN ('medium_decline', 'severe_decline')) AS decline_rows,
        COUNT_IF(w.response_class = 'severe_decline') AS severe_decline_rows,
        COUNT_IF(
            w.response_class = 'recovery'
            AND c.first_decline_date IS NOT NULL
            AND w.observation_date > c.first_decline_date
        ) AS recovery_after_decline_rows,
        MIN(CASE WHEN w.response_class IN ('medium_decline', 'severe_decline', 'recovery')
                 THEN w.ndvi_delta END) AS adverse_ndvi,
        MIN(CASE WHEN w.response_class IN ('medium_decline', 'severe_decline', 'recovery')
                 THEN w.ndmi_delta END) AS adverse_ndmi,
        MAX(CASE WHEN w.response_class IN ('medium_decline', 'severe_decline', 'recovery')
                 THEN w.psri_delta END) AS adverse_psri,
        MAX(CASE WHEN w.response_class IN ('medium_decline', 'severe_decline', 'recovery')
                 THEN w.spectral_source_date END) AS spectral_source_max_date,
        CASE c.hazard_family
            WHEN 'drought' THEN MIN(w.spi_index)
            WHEN 'ponding_flooding' THEN MAX(w.ponding_mm)
            WHEN 'heat' THEN MAX(w.apparent_temperature)
            WHEN 'damaging_wind' THEN MAX(w.wind_speed)
            ELSE NULL
        END AS canonical_hazard_intensity
    FROM classified c
    LEFT JOIN window_rows w USING (event_id)
    GROUP BY c.event_id, c.hazard_family, c.first_decline_date
), post_anchor AS (
    SELECT c.event_id, COUNT(*) AS post_anchor_row_count
    FROM classified c
    JOIN memberships_source m USING (event_id)
    WHERE c.anchor_date IS NOT NULL
      AND CAST(m.observation_date AS DATE) > c.anchor_date
    GROUP BY c.event_id
)
SELECT
    c.event_id,
    c.field_id,
    c.hazard_family,
    c.anchor_date,
    c.anchor_kind,
    c.anchor_outcome,
    c.anchor_outcome AS anchor_status,
    c.anchor_outcome = 'eligible' AS eligible_for_training,
    c.anchor_window_day_count,
    c.usable_day_count,
    c.missing_pressure_day_count,
    CASE WHEN c.anchor_window_day_count = 0 THEN 0.0
         ELSE c.usable_day_count::DOUBLE / c.anchor_window_day_count END
        AS pressure_observed_coverage,
    c.evidence_max_date,
    s.spectral_source_max_date,
    COALESCE(p.post_anchor_row_count, 0) AS post_anchor_row_count,
    CASE WHEN c.anchor_outcome = 'eligible' THEN r.peak_risk_rank END AS peak_risk_rank,
    CASE WHEN c.anchor_outcome = 'eligible' THEN r.mean_risk_rank END AS mean_risk_rank,
    CASE WHEN c.anchor_outcome = 'eligible' THEN r.risk_slope END AS risk_slope,
    CASE WHEN c.anchor_outcome = 'eligible' THEN r.elevated_day_fraction END AS elevated_day_fraction,
    CASE WHEN c.anchor_outcome = 'eligible' THEN r.high_day_fraction END AS high_day_fraction,
    CASE WHEN c.anchor_outcome = 'eligible'
         THEN COALESCE(l.longest_elevated_run, 0)::DOUBLE / c.usable_day_count END
        AS longest_elevated_run_fraction,
    CASE WHEN c.anchor_outcome = 'eligible' THEN CAST(s.decline_rows > 0 AS INTEGER) END
        AS attributed_decline_any,
    CASE WHEN c.anchor_outcome = 'eligible' THEN CAST(s.severe_decline_rows > 0 AS INTEGER) END
        AS attributed_severe_decline_any,
    CASE WHEN c.anchor_outcome = 'eligible'
         THEN LEAST(1.0, s.decline_rows::DOUBLE / c.usable_day_count) END
        AS attributed_decline_day_fraction,
    CASE WHEN c.anchor_outcome <> 'eligible' THEN NULL
         WHEN c.first_decline_date IS NULL THEN 0.0
         WHEN DATE_DIFF('day', c.event_start_date, c.anchor_date) <= 0 THEN 0.0
         ELSE LEAST(1.0, GREATEST(0.0,
             DATE_DIFF('day', c.event_start_date, c.first_decline_date)::DOUBLE
             / DATE_DIFF('day', c.event_start_date, c.anchor_date)
         )) END AS first_attributed_decline_position,
    CASE WHEN c.anchor_outcome = 'eligible'
         THEN CAST(s.recovery_after_decline_rows > 0 AS INTEGER) END
        AS attributed_recovery_after_decline,
    CASE WHEN c.anchor_outcome = 'eligible' THEN s.adverse_ndvi END
        AS worst_attributed_ndvi_delta,
    CASE WHEN c.anchor_outcome = 'eligible' THEN CAST(s.adverse_ndvi IS NULL AS INTEGER) END
        AS worst_attributed_ndvi_delta_missing,
    CASE WHEN c.anchor_outcome = 'eligible' THEN s.adverse_ndmi END
        AS worst_attributed_ndmi_delta,
    CASE WHEN c.anchor_outcome = 'eligible' THEN CAST(s.adverse_ndmi IS NULL AS INTEGER) END
        AS worst_attributed_ndmi_delta_missing,
    CASE WHEN c.anchor_outcome = 'eligible' THEN s.adverse_psri END
        AS worst_attributed_psri_delta,
    CASE WHEN c.anchor_outcome = 'eligible' THEN CAST(s.adverse_psri IS NULL AS INTEGER) END
        AS worst_attributed_psri_delta_missing,
    CASE WHEN c.anchor_outcome = 'eligible' THEN s.canonical_hazard_intensity END
        AS hazard_intensity,
    CASE WHEN c.anchor_outcome = 'eligible'
         THEN CAST(s.canonical_hazard_intensity IS NULL AS INTEGER) END
        AS hazard_intensity_missing,
    CASE WHEN c.anchor_outcome = 'eligible'
         THEN c.usable_day_count::DOUBLE / {MATURITY_USABLE_DAYS} END AS usable_days_fraction
FROM classified c
LEFT JOIN risk_summary r USING (event_id)
LEFT JOIN longest_runs l USING (event_id)
LEFT JOIN response_summary s USING (event_id)
LEFT JOIN post_anchor p USING (event_id)
ORDER BY c.event_id
"""


def build_event_anchors(
    generation_dir: Path,
    *,
    through: str | date | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Return one anchor record per event visible at ``through``.

    This materializes the result and is intended for bounded data and tests.
    Use :func:`write_event_anchors` for the full VM generation.
    """
    with _configured_connection(
        generation_dir, through=through, threads=threads,
        memory_limit=memory_limit, temp_dir=temp_dir,
    ) as connection:
        output = connection.execute(_ANCHOR_QUERY).fetchdf()
    _validate_anchor_output(output)
    return output


def write_event_anchors(
    generation_dir: Path,
    output_path: Path,
    *,
    through: str | date | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> Path:
    """Stream one-row-per-event V2 anchors directly to an immutable Parquet."""
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Anchor output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".event-anchors-v2-", dir=output_path.parent) as directory:
        stage = Path(directory) / output_path.name
        with _configured_connection(
            generation_dir, through=through, threads=threads,
            memory_limit=memory_limit, temp_dir=temp_dir,
        ) as connection:
            connection.execute(
                f"COPY ({_ANCHOR_QUERY}) TO {_sql_string(str(stage))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            leakage_count = int(connection.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE eligible_for_training
                  AND (
                    evidence_max_date > anchor_date
                    OR spectral_source_max_date > anchor_date
                  )
                """,
                [str(stage)],
            ).fetchone()[0])
            if leakage_count:
                raise ValueError(
                    f"V2 anchor output contains {leakage_count} post-anchor evidence rows"
                )
        os.replace(stage, output_path)
    return output_path


def _configured_connection(
    generation_dir: Path,
    *,
    through: str | date | None,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> duckdb.DuckDBPyConnection:
    generation_dir = generation_dir.expanduser().resolve()
    paths = {
        "events_source": generation_dir / "event_windows.parquet",
        "memberships_source": generation_dir / "story_day_membership.parquet",
        "signals_source": generation_dir / "daily_causal_signals.parquet",
    }
    manifest_path = generation_dir / "manifest.json"
    for path in (*paths.values(), manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Generation is missing required artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str((manifest.get("run") or {}).get("status") or "") != "complete":
        raise ValueError("V2 anchors require a completed immutable generation")
    generation_cutoff = date.fromisoformat(str(manifest["run"]["as_of_date"])[:10])
    cutoff = _as_date(through) if through is not None else generation_cutoff
    if cutoff > generation_cutoff:
        raise ValueError(
            f"Anchor cutoff {cutoff} exceeds generation as-of date {generation_cutoff}."
        )
    if isinstance(threads, bool) or int(threads) < 1:
        raise ValueError("threads must be a positive integer")

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET threads={int(threads)}")
        if memory_limit:
            connection.execute("SET memory_limit=?", [memory_limit])
        if temp_dir is not None:
            resolved_temp = temp_dir.expanduser().resolve()
            resolved_temp.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved_temp)])
        for view, path in paths.items():
            relation = connection.read_parquet(str(path))
            missing = sorted(_REQUIRED_COLUMNS[view] - set(relation.columns))
            if missing:
                raise ValueError(
                    f"{path.name} is missing V2 anchor columns: {', '.join(missing)}"
                )
            relation.create_view("signals_source_raw" if view == "signals_source" else view)
            if view == "signals_source":
                source_expression = (
                    "CAST(spectral_source_date AS DATE)"
                    if "spectral_source_date" in relation.columns
                    else "NULL::DATE"
                )
                connection.execute(
                    "CREATE TEMP VIEW signals_source AS "
                    f"SELECT *, {source_expression} AS spectral_source_date_v2 "
                    "FROM signals_source_raw"
                )
        duplicates = int(connection.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM events_source"
        ).fetchone()[0])
        if duplicates:
            raise ValueError(f"event_windows.parquet contains {duplicates} duplicate event IDs")
        events_without_membership = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM events_source e
            LEFT JOIN memberships_source m USING (event_id)
            WHERE m.event_id IS NULL
            """
        ).fetchone()[0])
        orphan_memberships = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM memberships_source m
            LEFT JOIN events_source e USING (event_id)
            WHERE e.event_id IS NULL
            """
        ).fetchone()[0])
        if events_without_membership or orphan_memberships:
            raise ValueError(
                "event/membership lineage is incomplete: "
                f"{events_without_membership} events without membership and "
                f"{orphan_memberships} orphan memberships"
            )
        duplicate_signals = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT field_id, crop_instance_id, CAST(observation_date AS DATE), COUNT(*)
                FROM signals_source_raw
                GROUP BY field_id, crop_instance_id, CAST(observation_date AS DATE)
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0])
        if duplicate_signals:
            raise ValueError(
                f"daily_causal_signals.parquet contains {duplicate_signals} duplicate join keys"
            )
        memberships_without_signal = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM memberships_source m
            LEFT JOIN signals_source_raw s
              ON CAST(s.field_id AS VARCHAR) = CAST(m.field_id AS VARCHAR)
             AND CAST(s.crop_instance_id AS VARCHAR) = CAST(m.crop_instance_id AS VARCHAR)
             AND CAST(s.observation_date AS DATE) = CAST(m.observation_date AS DATE)
            WHERE s.observation_date IS NULL
            """
        ).fetchone()[0])
        if memberships_without_signal:
            raise ValueError(
                "event membership is missing matching causal-signal lineage for "
                f"{memberships_without_signal} rows"
            )
        duplicate_usable = connection.execute(
            """
            SELECT event_id, CAST(observation_date AS DATE), COUNT(*)
            FROM memberships_source
            WHERE COALESCE(TRY_CAST(pressure_observed AS BOOLEAN), FALSE)
              AND TRY_CAST(daily_pressure_rank AS DOUBLE) IS NOT NULL
            GROUP BY event_id, CAST(observation_date AS DATE)
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate_usable is not None:
            raise ValueError(
                "story_day_membership.parquet contains duplicate usable "
                f"(event_id, date): {duplicate_usable[0]}, {duplicate_usable[1]}"
            )
        invalid_ranks = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM memberships_source
            WHERE (
                daily_pressure_rank IS NOT NULL
                AND (
                    TRY_CAST(daily_pressure_rank AS DOUBLE) IS NULL
                    OR NOT isfinite(TRY_CAST(daily_pressure_rank AS DOUBLE))
                    OR TRY_CAST(daily_pressure_rank AS DOUBLE) < 0
                    OR TRY_CAST(daily_pressure_rank AS DOUBLE) > 4
                )
            ) OR (
                COALESCE(TRY_CAST(pressure_observed AS BOOLEAN), FALSE)
                AND TRY_CAST(daily_pressure_rank AS DOUBLE) IS NULL
            )
            """
        ).fetchone()[0])
        if invalid_ranks:
            raise ValueError(
                f"story_day_membership.parquet contains {invalid_ranks} observed risk ranks "
                "outside the finite [0, 4] domain"
            )
        connection.execute("CREATE TEMP TABLE anchor_parameters(cutoff DATE)")
        connection.execute("INSERT INTO anchor_parameters VALUES (?)", [cutoff])
        return connection
    except Exception:
        connection.close()
        raise


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_anchor_output(output: pd.DataFrame) -> None:
    if output["event_id"].duplicated().any():
        raise ValueError("V2 anchor output contains duplicate event IDs")
    eligible = output["eligible_for_training"].fillna(False).astype(bool)
    evidence_leak = (
        pd.to_datetime(output.loc[eligible, "evidence_max_date"])
        > pd.to_datetime(output.loc[eligible, "anchor_date"])
    ).fillna(False)
    spectral_leak = (
        pd.to_datetime(output.loc[eligible, "spectral_source_max_date"])
        > pd.to_datetime(output.loc[eligible, "anchor_date"])
    ).fillna(False)
    if evidence_leak.any() or spectral_leak.any():
        raise ValueError("V2 anchor output contains post-anchor evidence")
