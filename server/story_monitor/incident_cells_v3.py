"""Frozen stage baseline and significant local exposure-cell preparation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import numpy as np
import pandas as pd

from .incident_policy_v3 import IncidentPolicyV3


BASELINE_SCHEMA_VERSION = "stage-aware-baseline-v3/1"
CELL_SCHEMA_VERSION = "weekly-exposure-cells-v3/1"


def build_stage_baseline(
    context_path: Path,
    lanes_path: Path,
    *,
    baseline_through: str,
    policy: IncidentPolicyV3,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Fit a frozen stage-aware expected affected rate from pre-cutoff weeks."""
    cutoff = pd.Timestamp(baseline_through).normalize()
    with _connection(threads, memory_limit, temp_dir) as connection:
        _parquet_view(connection, "context_source", context_path)
        _parquet_view(connection, "lane_source_raw", lanes_path)
        _normalize_lane_source(connection)
        _require_columns(
            connection, "context_source",
            {"timeline_bucket", "crop_instance_id", "stage_bucket", "evaluable"},
        )
        _require_columns(
            connection, "lane_source",
            {
                "timeline_bucket", "crop_instance_id", "hazard_family", "event_state",
                "is_canonical_field_hazard_week",
            },
        )
        baseline = connection.execute(
            """
            WITH cohort AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(stage_bucket AS VARCHAR) AS stage_bucket,
                    COUNT(DISTINCT CAST(crop_instance_id AS VARCHAR)) AS denominator
                FROM context_source
                WHERE COALESCE(TRY_CAST(evaluable AS BOOLEAN), FALSE)
                  AND CAST(timeline_bucket AS DATE) + INTERVAL 6 DAY <= CAST(? AS DATE)
                GROUP BY 1, 2
            ), hazards AS (
                SELECT DISTINCT CAST(hazard_family AS VARCHAR) AS hazard_family
                FROM lane_source
                WHERE CAST(timeline_bucket AS DATE) + INTERVAL 6 DAY <= CAST(? AS DATE)
                  AND LOWER(CAST(hazard_family AS VARCHAR)) NOT IN ('', 'none', 'null')
            ), active AS (
                SELECT
                    CAST(l.timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(l.hazard_family AS VARCHAR) AS hazard_family,
                    CAST(c.stage_bucket AS VARCHAR) AS stage_bucket,
                    COUNT(DISTINCT CAST(l.crop_instance_id AS VARCHAR)) AS affected
                FROM lane_source l
                JOIN context_source c
                  ON CAST(c.timeline_bucket AS DATE) = CAST(l.timeline_bucket AS DATE)
                 AND CAST(c.crop_instance_id AS VARCHAR) = CAST(l.crop_instance_id AS VARCHAR)
                WHERE CAST(l.timeline_bucket AS DATE) + INTERVAL 6 DAY <= CAST(? AS DATE)
                  AND COALESCE(TRY_CAST(l.is_canonical_field_hazard_week AS BOOLEAN), FALSE)
                  AND UPPER(CAST(l.event_state AS VARCHAR)) IN ('ACTIVE', 'SEVERE')
                  AND COALESCE(TRY_CAST(c.evaluable AS BOOLEAN), FALSE)
                GROUP BY 1, 2, 3
            ), weekly AS (
                SELECT
                    h.hazard_family, c.timeline_bucket, c.stage_bucket, c.denominator,
                    COALESCE(a.affected, 0) AS affected,
                    EXTRACT(WEEK FROM c.timeline_bucket)::INTEGER AS iso_week
                FROM cohort c
                CROSS JOIN hazards h
                LEFT JOIN active a
                  ON a.timeline_bucket = c.timeline_bucket
                 AND a.hazard_family = h.hazard_family
                 AND a.stage_bucket = c.stage_bucket
            ), grouped AS (
                SELECT
                    hazard_family, stage_bucket, iso_week,
                    SUM(affected)::BIGINT AS affected_count,
                    SUM(denominator)::BIGINT AS evaluable_count,
                    COUNT(*)::BIGINT AS training_week_count
                FROM weekly
                GROUP BY 1, 2, 3
            ), global AS (
                SELECT
                    hazard_family, iso_week,
                    SUM(affected)::DOUBLE / NULLIF(SUM(denominator), 0) AS global_probability,
                    SUM(affected)::BIGINT AS global_affected_count,
                    SUM(denominator)::BIGINT AS global_evaluable_count
                FROM weekly
                GROUP BY 1, 2
            )
            SELECT
                g.hazard_family, g.stage_bucket, g.iso_week,
                g.affected_count, g.evaluable_count, g.training_week_count,
                x.global_affected_count, x.global_evaluable_count,
                x.global_probability,
                (g.affected_count + ? * x.global_probability)
                    / NULLIF(g.evaluable_count + ?, 0) AS expected_probability,
                CAST(? AS DOUBLE) AS prior_strength,
                CAST(? AS DATE) AS baseline_through,
                CAST(? AS VARCHAR) AS schema_version,
                CAST(? AS VARCHAR) AS policy_version,
                CAST(? AS VARCHAR) AS policy_sha256
            FROM grouped g
            JOIN global x USING (hazard_family, iso_week)
            WHERE g.evaluable_count > 0
            ORDER BY g.hazard_family, g.stage_bucket, g.iso_week
            """,
            [
                cutoff.date(), cutoff.date(), cutoff.date(),
                policy.baseline_prior_strength, policy.baseline_prior_strength,
                policy.baseline_prior_strength, cutoff.date(), BASELINE_SCHEMA_VERSION,
                policy.version, policy.source_sha256,
            ],
        ).fetchdf()
    if baseline.empty:
        raise ValueError("No evaluable pre-cutoff weeks exist for the frozen stage baseline")
    _finite_probability(baseline, "expected_probability")
    return baseline


def build_weekly_exposure_cells(
    context_path: Path,
    lanes_path: Path,
    baseline: pd.DataFrame | Path,
    *,
    policy: IncidentPolicyV3,
    assignment_after: str | None = None,
    assignment_through: str | None = None,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Aggregate candidate cells and apply stage adjustment plus per-week FDR."""
    assignment_after_date = (
        pd.Timestamp(assignment_after).normalize()
        if assignment_after is not None else None
    )
    assignment_through_date = (
        pd.Timestamp(assignment_through).normalize()
        if assignment_through is not None else None
    )
    if assignment_after_date is not None and pd.isna(assignment_after_date):
        raise ValueError("assignment_after must be a valid date")
    if assignment_through_date is not None and pd.isna(assignment_through_date):
        raise ValueError("assignment_through must be a valid date")
    with _connection(threads, memory_limit, temp_dir) as connection:
        _parquet_view(connection, "context_source", context_path)
        _parquet_view(connection, "lane_source_raw", lanes_path)
        _normalize_lane_source(connection)
        if isinstance(baseline, Path):
            _parquet_view(connection, "baseline_source", baseline)
        else:
            connection.register("baseline_source", baseline)
        _require_columns(
            connection, "context_source",
            {
                "timeline_bucket", "field_id", "crop_instance_id", "crop_name",
                "stage_bucket", "evaluable", "centroid_lon", "centroid_lat",
            },
        )
        _require_columns(
            connection, "lane_source",
            {
                "timeline_bucket", "event_id", "field_id", "crop_instance_id",
                "hazard_family", "event_state", "is_canonical_field_hazard_week",
            },
        )
        if assignment_after_date is not None:
            _assert_supported_assignment_hazards(
                connection,
                assignment_after=assignment_after_date,
                assignment_through=assignment_through_date,
            )
        reference_latitude = _reference_latitude(connection, policy)
        scale_lon = 111.32 * math.cos(math.radians(reference_latitude))
        if abs(scale_lon) < 1e-9:
            raise ValueError("Metric grid longitude scale is undefined at the reference latitude")
        cells = connection.execute(
            """
            WITH context_cells AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(field_id AS VARCHAR) AS field_id,
                    CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
                    CAST(crop_name AS VARCHAR) AS crop_name,
                    CAST(stage_bucket AS VARCHAR) AS stage_bucket,
                    COALESCE(TRY_CAST(monitored AS BOOLEAN), TRUE) AS monitored,
                    COALESCE(TRY_CAST(evaluable AS BOOLEAN), FALSE) AS evaluable,
                    TRY_CAST(centroid_lon AS DOUBLE) AS centroid_lon,
                    TRY_CAST(centroid_lat AS DOUBLE) AS centroid_lat,
                    FLOOR((TRY_CAST(centroid_lon AS DOUBLE) - ?) * ? / ?)::BIGINT AS cell_x,
                    FLOOR((TRY_CAST(centroid_lat AS DOUBLE) - ?) * 110.574 / ?)::BIGINT AS cell_y
                FROM context_source
                WHERE TRY_CAST(centroid_lon AS DOUBLE) BETWEEN -180 AND 180
                  AND TRY_CAST(centroid_lat AS DOUBLE) BETWEEN -90 AND 90
            ), active_rows AS (
                SELECT
                    c.*, CAST(l.event_id AS VARCHAR) AS episode_id,
                    CAST(l.hazard_family AS VARCHAR) AS hazard_family,
                    UPPER(CAST(l.event_state AS VARCHAR)) AS event_state,
                    l.v3_current_risk_rank AS current_risk_rank,
                    l.v3_daily_response_class AS response_class,
                    l.v3_fresh_response_evidence AS fresh_response_evidence
                FROM lane_source l
                JOIN context_cells c
                  ON c.timeline_bucket = CAST(l.timeline_bucket AS DATE)
                 AND c.crop_instance_id = CAST(l.crop_instance_id AS VARCHAR)
                WHERE COALESCE(TRY_CAST(l.is_canonical_field_hazard_week AS BOOLEAN), FALSE)
            ), hazards AS (
                SELECT DISTINCT CAST(hazard_family AS VARCHAR) AS hazard_family
                FROM baseline_source
            ), cell_keys AS (
                SELECT DISTINCT
                    c.timeline_bucket, h.hazard_family, c.cell_x, c.cell_y
                FROM context_cells c
                CROSS JOIN hazards h
            ), denoms AS (
                SELECT
                    k.timeline_bucket, k.hazard_family, k.cell_x, k.cell_y,
                    c.stage_bucket,
                    COUNT(DISTINCT CASE WHEN c.monitored THEN c.crop_instance_id END)::BIGINT
                        AS monitored_count,
                    COUNT(DISTINCT CASE WHEN c.evaluable THEN c.crop_instance_id END)::BIGINT
                        AS evaluable_count
                FROM cell_keys k
                JOIN context_cells c
                  ON c.timeline_bucket = k.timeline_bucket
                 AND c.cell_x = k.cell_x AND c.cell_y = k.cell_y
                GROUP BY 1, 2, 3, 4, 5
            ), cell_denoms AS (
                SELECT
                    k.timeline_bucket, k.hazard_family, k.cell_x, k.cell_y,
                    COUNT(DISTINCT CASE WHEN c.monitored THEN c.field_id END)::BIGINT
                        AS monitored_field_count,
                    COUNT(DISTINCT CASE WHEN c.evaluable THEN c.field_id END)::BIGINT
                        AS evaluable_field_count
                FROM cell_keys k
                JOIN context_cells c
                  ON c.timeline_bucket = k.timeline_bucket
                 AND c.cell_x = k.cell_x AND c.cell_y = k.cell_y
                GROUP BY 1, 2, 3, 4
            ), baseline_global AS (
                SELECT hazard_family, iso_week, MAX(global_probability) AS global_probability
                FROM baseline_source
                GROUP BY 1, 2
            ), expected AS (
                SELECT
                    d.timeline_bucket, d.hazard_family, d.cell_x, d.cell_y,
                    SUM(d.monitored_count)::BIGINT AS monitored_count,
                    SUM(d.evaluable_count)::BIGINT AS evaluable_count,
                    SUM(d.evaluable_count * COALESCE(b.expected_probability, g.global_probability, 0))
                        AS expected_active_count,
                    SUM(
                        d.evaluable_count
                        * COALESCE(b.expected_probability, g.global_probability, 0)
                        * (1 - COALESCE(b.expected_probability, g.global_probability, 0))
                        * (1 + (d.evaluable_count - 1) / (? + 1))
                    ) AS expected_variance,
                    COUNT_IF(b.expected_probability IS NULL AND g.global_probability IS NULL)
                        AS missing_baseline_stage_count
                FROM denoms d
                LEFT JOIN baseline_source b
                  ON b.hazard_family = d.hazard_family
                 AND b.stage_bucket = d.stage_bucket
                 AND b.iso_week = EXTRACT(WEEK FROM d.timeline_bucket)::INTEGER
                LEFT JOIN baseline_global g
                  ON g.hazard_family = d.hazard_family
                 AND g.iso_week = EXTRACT(WEEK FROM d.timeline_bucket)::INTEGER
                GROUP BY 1, 2, 3, 4
            ), observed AS (
                SELECT
                    timeline_bucket, hazard_family, cell_x, cell_y,
                    COUNT(DISTINCT CASE WHEN event_state IN ('ACTIVE', 'SEVERE')
                        THEN crop_instance_id END)::BIGINT AS active_count,
                    COUNT(DISTINCT CASE WHEN event_state IN ('ACTIVE', 'SEVERE')
                        THEN field_id END)::BIGINT AS active_field_count,
                    COUNT(DISTINCT CASE WHEN event_state = 'SEVERE'
                        THEN field_id END)::BIGINT AS severe_field_count,
                    COUNT(DISTINCT CASE WHEN event_state = 'WATCH'
                        THEN field_id END)::BIGINT AS watch_field_count,
                    COUNT(DISTINCT CASE WHEN fresh_response_evidence
                        AND response_class IN ('medium_decline', 'severe_decline')
                        THEN field_id END)::BIGINT AS fresh_response_field_count,
                    COUNT(DISTINCT episode_id)::BIGINT AS episode_count,
                    MAX(current_risk_rank) AS peak_risk_rank,
                    AVG(centroid_lon) AS active_center_lon,
                    AVG(centroid_lat) AS active_center_lat
                FROM active_rows
                GROUP BY 1, 2, 3, 4
            )
            SELECT
                e.timeline_bucket, e.hazard_family,
                'cell_' || CAST(e.cell_x AS VARCHAR) || '_' || CAST(e.cell_y AS VARCHAR) AS cell_id,
                e.cell_x, e.cell_y,
                ? + e.cell_x * ? / ? AS min_lon,
                ? + (e.cell_x + 1) * ? / ? AS max_lon,
                ? + e.cell_y * ? / 110.574 AS min_lat,
                ? + (e.cell_y + 1) * ? / 110.574 AS max_lat,
                o.active_center_lon, o.active_center_lat,
                e.monitored_count, d.monitored_field_count,
                e.evaluable_count, d.evaluable_field_count,
                COALESCE(o.active_count, 0) AS active_count,
                COALESCE(o.active_field_count, 0) AS active_field_count,
                COALESCE(o.severe_field_count, 0) AS severe_field_count,
                COALESCE(o.watch_field_count, 0) AS watch_field_count,
                COALESCE(o.fresh_response_field_count, 0) AS fresh_response_field_count,
                COALESCE(o.episode_count, 0) AS episode_count,
                COALESCE(o.peak_risk_rank, 0) AS peak_risk_rank,
                e.expected_active_count, e.expected_variance,
                (COALESCE(o.active_count, 0) - e.expected_active_count)
                    / SQRT(GREATEST(e.expected_variance, 1e-9)) AS z_score,
                e.missing_baseline_stage_count,
                CAST(? AS DOUBLE) AS reference_latitude,
                CAST(? AS DOUBLE) AS cell_size_km,
                CAST(? AS VARCHAR) AS schema_version,
                CAST(? AS VARCHAR) AS policy_version,
                CAST(? AS VARCHAR) AS policy_sha256
            FROM expected e
            LEFT JOIN observed o USING (timeline_bucket, hazard_family, cell_x, cell_y)
            JOIN cell_denoms d USING (timeline_bucket, hazard_family, cell_x, cell_y)
            ORDER BY e.timeline_bucket, e.hazard_family, e.cell_x, e.cell_y
            """,
            [
                policy.grid_origin_lon, scale_lon, policy.grid_cell_size_km,
                policy.grid_origin_lat, policy.grid_cell_size_km,
                policy.baseline_prior_strength,
                policy.grid_origin_lon, policy.grid_cell_size_km, scale_lon,
                policy.grid_origin_lon, policy.grid_cell_size_km, scale_lon,
                policy.grid_origin_lat, policy.grid_cell_size_km,
                policy.grid_origin_lat, policy.grid_cell_size_km,
                reference_latitude, policy.grid_cell_size_km, CELL_SCHEMA_VERSION,
                policy.version, policy.source_sha256,
            ],
        ).fetchdf()
    if cells.empty:
        return _empty_cells()
    if assignment_after_date is not None:
        cells = cells[
            pd.to_datetime(cells["timeline_bucket"], errors="raise").dt.normalize()
            > assignment_after_date
        ].copy()
        if cells.empty:
            return _empty_cells()
    if assignment_through_date is not None:
        buckets = pd.to_datetime(cells["timeline_bucket"], errors="raise").dt.normalize()
        cells = cells[
            buckets + pd.Timedelta(days=6) <= assignment_through_date
        ].copy()
        if cells.empty:
            return _empty_cells()
    cells["p_value"] = [
        0.5 * math.erfc(float(value) / math.sqrt(2.0))
        if np.isfinite(value) else 1.0
        for value in cells["z_score"]
    ]
    cells["q_value"] = 1.0
    for _, indices in cells.groupby(["timeline_bucket", "hazard_family"], sort=True).groups.items():
        cells.loc[list(indices), "q_value"] = _benjamini_hochberg(
            cells.loc[list(indices), "p_value"].to_numpy(dtype=float)
        )
    coverage_eligible = (
        (cells["evaluable_field_count"] >= policy.minimum_evaluable_fields)
        & (cells["missing_baseline_stage_count"] == 0)
    )
    eligible = coverage_eligible & (
        cells["active_field_count"] >= policy.minimum_active_fields
    )
    severe_override = (
        coverage_eligible
        & (cells["severe_field_count"] >= policy.severe_override_min_fields)
        & (
            cells["fresh_response_field_count"]
            >= policy.severe_override_min_fresh_response_fields
        )
        & policy.allow_severe_override
    )
    cells["passes_coverage_gate"] = coverage_eligible
    cells["passes_denominator_gate"] = eligible
    cells["severe_response_override"] = severe_override
    cells["is_significant"] = (eligible & (cells["q_value"] <= policy.fdr_alpha)) | severe_override
    cells["significance_reason"] = np.select(
        [severe_override, eligible & (cells["q_value"] <= policy.fdr_alpha)],
        ["multi_field_severe_fresh_response_override", "stage_adjusted_fdr"],
        default="not_significant",
    )
    cells["grid_x"] = cells["cell_x"].astype("int64")
    cells["grid_y"] = cells["cell_y"].astype("int64")
    cells["grid_center_x_km"] = (cells["grid_x"] + 0.5) * policy.grid_cell_size_km
    cells["grid_center_y_km"] = (cells["grid_y"] + 0.5) * policy.grid_cell_size_km
    cells["grid_center_lon"] = (cells["min_lon"] + cells["max_lon"]) / 2
    cells["grid_center_lat"] = (cells["min_lat"] + cells["max_lat"]) / 2
    cells["active_variance"] = cells["expected_variance"]
    cells["fdr_q_value"] = cells["q_value"]
    cells["significant"] = cells["is_significant"]
    return cells.sort_values(
        ["timeline_bucket", "hazard_family", "cell_x", "cell_y"], kind="mergesort"
    ).reset_index(drop=True)


def build_component_field_rows(
    context_path: Path,
    lanes_path: Path,
    cells: pd.DataFrame | Path,
    *,
    policy: IncidentPolicyV3,
    frontier_distance_cells: int = 1,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pd.DataFrame:
    """Load only canonical event lanes in or beside significant component cells."""
    if frontier_distance_cells < 0:
        raise ValueError("frontier_distance_cells cannot be negative")
    if isinstance(cells, Path):
        significant = pd.read_parquet(cells)
    else:
        significant = cells.copy()
    marker = "is_significant" if "is_significant" in significant else "significant"
    if marker not in significant:
        raise ValueError("weekly cells are missing a significance marker")
    significant = significant[significant[marker].fillna(False).astype(bool)].copy()
    output_columns = [
        "timeline_bucket", "hazard_family", "field_id", "crop_instance_id",
        "episode_id", "event_state", "stage_family", "crop_name", "grid_x", "grid_y",
        "centroid_lon", "centroid_lat", "impact_active", "response_class",
        "fresh_response_evidence", "evaluable", "is_data_gap", "is_data_gap_snapshot",
        "knowledge_time",
    ]
    if significant.empty:
        return pd.DataFrame(columns=output_columns)
    rename = {}
    if "cell_x" in significant and "grid_x" not in significant:
        rename["cell_x"] = "grid_x"
    if "cell_y" in significant and "grid_y" not in significant:
        rename["cell_y"] = "grid_y"
    significant = significant.rename(columns=rename)
    required = {"timeline_bucket", "hazard_family", "grid_x", "grid_y"}
    missing = sorted(required - set(significant.columns))
    if missing:
        raise ValueError("weekly cells are missing columns: " + ", ".join(missing))
    significant = significant.loc[:, sorted(required)].drop_duplicates().reset_index(drop=True)

    with _connection(threads, memory_limit, temp_dir) as connection:
        _parquet_view(connection, "context_source", context_path)
        _parquet_view(connection, "lane_source_raw", lanes_path)
        _normalize_lane_source(connection)
        connection.register("significant_cells", significant)
        reference_latitude = (
            float(cells["reference_latitude"].dropna().iloc[0])
            if not isinstance(cells, Path)
            and "reference_latitude" in cells
            and not cells["reference_latitude"].dropna().empty
            else _reference_latitude(connection, policy)
        )
        scale_lon = 111.32 * math.cos(math.radians(reference_latitude))
        frame = connection.execute(
            """
            WITH context_cells AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(field_id AS VARCHAR) AS field_id,
                    CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
                    CAST(crop_name AS VARCHAR) AS crop_name,
                    CAST(stage_bucket AS VARCHAR) AS stage_family,
                    COALESCE(TRY_CAST(evaluable AS BOOLEAN), FALSE) AS evaluable,
                    TRY_CAST(centroid_lon AS DOUBLE) AS centroid_lon,
                    TRY_CAST(centroid_lat AS DOUBLE) AS centroid_lat,
                    FLOOR((TRY_CAST(centroid_lon AS DOUBLE) - ?) * ? / ?)::BIGINT AS grid_x,
                    FLOOR((TRY_CAST(centroid_lat AS DOUBLE) - ?) * 110.574 / ?)::BIGINT AS grid_y
                FROM context_source
                WHERE TRY_CAST(centroid_lon AS DOUBLE) BETWEEN -180 AND 180
                  AND TRY_CAST(centroid_lat AS DOUBLE) BETWEEN -90 AND 90
            ), lanes AS (
                SELECT
                    c.*,
                    CAST(l.event_id AS VARCHAR) AS episode_id,
                    CAST(l.hazard_family AS VARCHAR) AS hazard_family,
                    UPPER(CAST(l.event_state AS VARCHAR)) AS event_state,
                    l.v3_daily_response_class AS response_class,
                    l.v3_fresh_response_evidence AS fresh_response_evidence,
                    l.v3_is_data_gap_snapshot AS is_data_gap_snapshot,
                    l.v3_knowledge_time AS knowledge_time
                FROM lane_source l
                JOIN context_cells c
                  ON c.timeline_bucket = CAST(l.timeline_bucket AS DATE)
                 AND c.crop_instance_id = CAST(l.crop_instance_id AS VARCHAR)
                WHERE COALESCE(TRY_CAST(l.is_canonical_field_hazard_week AS BOOLEAN), FALSE)
                  AND UPPER(CAST(l.event_state AS VARCHAR)) IN (
                      'ACTIVE', 'SEVERE', 'WATCH', 'RECOVERING',
                      'QUIET_PENDING', 'CLOSED_RESPONSE_UNRESOLVED'
                  )
            )
            SELECT DISTINCT
                l.timeline_bucket, l.hazard_family, l.field_id, l.crop_instance_id,
                l.episode_id, l.event_state, l.stage_family, l.crop_name,
                l.grid_x, l.grid_y, l.centroid_lon, l.centroid_lat,
                l.fresh_response_evidence
                    AND l.response_class IN ('medium_decline', 'severe_decline')
                    AS impact_active,
                l.response_class,
                l.fresh_response_evidence,
                l.evaluable, l.is_data_gap_snapshot AS is_data_gap,
                l.is_data_gap_snapshot, l.knowledge_time
            FROM lanes l
            WHERE EXISTS (
                SELECT 1 FROM significant_cells s
                WHERE CAST(s.timeline_bucket AS DATE) = l.timeline_bucket
                  AND CAST(s.hazard_family AS VARCHAR) = l.hazard_family
                  AND ABS(TRY_CAST(s.grid_x AS BIGINT) - l.grid_x) <= ?
                  AND ABS(TRY_CAST(s.grid_y AS BIGINT) - l.grid_y) <= ?
            )
            ORDER BY l.timeline_bucket, l.hazard_family, l.field_id
            """,
            [
                policy.grid_origin_lon, scale_lon, policy.grid_cell_size_km,
                policy.grid_origin_lat, policy.grid_cell_size_km,
                frontier_distance_cells, frontier_distance_cells,
            ],
        ).fetchdf()
    if frame.duplicated(["timeline_bucket", "hazard_family", "field_id"]).any():
        raise ValueError("Canonical component field rows contain duplicate field-hazard-weeks")
    return frame.loc[:, output_columns]


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _reference_latitude(connection: duckdb.DuckDBPyConnection, policy: IncidentPolicyV3) -> float:
    if policy.reference_latitude_strategy == "fixed_origin":
        return policy.grid_origin_lat
    value = connection.execute(
        """
        SELECT AVG(TRY_CAST(centroid_lat AS DOUBLE))
        FROM context_source
        WHERE TRY_CAST(centroid_lat AS DOUBLE) BETWEEN -90 AND 90
        """
    ).fetchone()[0]
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Cannot derive metric-grid reference latitude from context geometry")
    return float(value)


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Benjamini-Hochberg expects a one-dimensional p-value array")
    if len(values) == 0:
        return values.astype(float)
    p = np.clip(np.nan_to_num(values.astype(float), nan=1.0, posinf=1.0, neginf=1.0), 0, 1)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _parquet_view(connection: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing V3 source artifact: {resolved}")
    connection.read_parquet(str(resolved)).create_view(name, replace=True)


def _normalize_lane_source(connection: duckdb.DuckDBPyConnection) -> None:
    columns = {str(row[0]) for row in connection.execute("DESCRIBE lane_source_raw").fetchall()}
    if "is_canonical_field_hazard_week" in columns:
        expression = "TRY_CAST(is_canonical_field_hazard_week AS BOOLEAN)"
    elif "is_canonical_field_hazard_lane" in columns:
        expression = "TRY_CAST(is_canonical_field_hazard_lane AS BOOLEAN)"
    else:
        raise ValueError(
            "lane_source is missing is_canonical_field_hazard_lane/week"
        )
    if "signal_response_class" in columns and "daily_response_class" in columns:
        response = (
            "LOWER(COALESCE(NULLIF(CAST(signal_response_class AS VARCHAR), ''), "
            "CAST(daily_response_class AS VARCHAR), ''))"
        )
    elif "signal_response_class" in columns:
        response = "LOWER(COALESCE(CAST(signal_response_class AS VARCHAR), ''))"
    elif "daily_response_class" in columns:
        response = "LOWER(COALESCE(CAST(daily_response_class AS VARCHAR), ''))"
    else:
        response = "CAST('' AS VARCHAR)"
    gap = (
        "COALESCE(TRY_CAST(is_data_gap_snapshot AS BOOLEAN), FALSE)"
        if "is_data_gap_snapshot" in columns else "FALSE"
    )
    risk = (
        "COALESCE(TRY_CAST(current_risk_rank AS DOUBLE), 0)"
        if "current_risk_rank" in columns else "CAST(0 AS DOUBLE)"
    )
    fresh = (
        "COALESCE(TRY_CAST(fresh_response_evidence AS BOOLEAN), FALSE)"
        if "fresh_response_evidence" in columns
        else "FALSE"
    )
    knowledge = (
        "CAST(snapshot_as_of_date AS DATE)"
        if "snapshot_as_of_date" in columns
        else "CAST(timeline_bucket AS DATE) + INTERVAL 6 DAY"
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW lane_source AS
        SELECT *,
            {expression} AS is_canonical_field_hazard_week,
            {response} AS v3_daily_response_class,
            {gap} AS v3_is_data_gap_snapshot,
            {risk} AS v3_current_risk_rank,
            {fresh} AS v3_fresh_response_evidence,
            {knowledge} AS v3_knowledge_time
        FROM lane_source_raw
        """
    )


def _connection(
    threads: int, memory_limit: str | None, temp_dir: Path | None
) -> duckdb.DuckDBPyConnection:
    if int(threads) < 1:
        raise ValueError("threads must be positive")
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        connection.execute("SET memory_limit = ?", [str(memory_limit)])
    if temp_dir:
        resolved = temp_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        connection.execute("SET temp_directory = ?", [str(resolved)])
    return connection


def _require_columns(
    connection: duckdb.DuckDBPyConnection, view: str, required: set[str]
) -> None:
    columns = {str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{view} is missing columns: {', '.join(missing)}")


def _finite_probability(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or (~values.between(0, 1)).any():
        raise ValueError(f"Frozen baseline contains invalid {column} values")


def _assert_supported_assignment_hazards(
    connection: duckdb.DuckDBPyConnection,
    *,
    assignment_after: pd.Timestamp,
    assignment_through: pd.Timestamp | None,
) -> None:
    """Fail closed when monitoring sees a hazard absent from the frozen baseline."""
    through_clause = ""
    parameters: list[object] = [assignment_after.date()]
    if assignment_through is not None:
        through_clause = (
            "AND CAST(timeline_bucket AS DATE) + INTERVAL 6 DAY <= CAST(? AS DATE)"
        )
        parameters.append(assignment_through.date())
    unsupported = connection.execute(
        f"""
        SELECT DISTINCT TRIM(CAST(hazard_family AS VARCHAR)) AS hazard_family
        FROM lane_source
        WHERE COALESCE(TRY_CAST(is_canonical_field_hazard_week AS BOOLEAN), FALSE)
          AND CAST(timeline_bucket AS DATE) > CAST(? AS DATE)
          {through_clause}
          AND LOWER(TRIM(CAST(hazard_family AS VARCHAR))) NOT IN ('', 'none', 'null')
        EXCEPT
        SELECT DISTINCT TRIM(CAST(hazard_family AS VARCHAR))
        FROM baseline_source
        ORDER BY hazard_family
        """,
        parameters,
    ).fetchall()
    names = [str(row[0]) for row in unsupported]
    if names:
        raise ValueError(
            "unsupported_new_hazard: frozen baseline has no expectation for "
            + ", ".join(names)
        )


def _empty_cells() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timeline_bucket", "hazard_family", "cell_id", "cell_x", "cell_y",
            "min_lon", "max_lon", "min_lat", "max_lat", "active_center_lon",
            "active_center_lat", "monitored_count", "monitored_field_count",
            "evaluable_count", "evaluable_field_count", "active_count",
            "active_field_count", "severe_field_count", "watch_field_count",
            "fresh_response_field_count", "episode_count", "peak_risk_rank",
            "expected_active_count", "expected_variance", "z_score",
            "missing_baseline_stage_count", "reference_latitude", "cell_size_km",
            "schema_version", "policy_version", "policy_sha256", "p_value",
            "q_value", "passes_coverage_gate", "passes_denominator_gate",
            "severe_response_override", "is_significant", "significance_reason",
            "grid_x", "grid_y", "grid_center_x_km", "grid_center_y_km",
            "grid_center_lon", "grid_center_lat", "active_variance",
            "fdr_q_value", "significant",
        ]
    )


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "CELL_SCHEMA_VERSION",
    "build_stage_baseline",
    "build_component_field_rows",
    "build_weekly_exposure_cells",
    "write_parquet_atomic",
]
