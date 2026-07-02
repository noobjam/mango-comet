"""Build causal temporal feature vectors from immutable monitor generations."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def load_training_prefixes(
    generation_dir: Path,
    *,
    through: str | None = None,
    sample_age_buckets: bool = True,
) -> pd.DataFrame:
    """Return weekly event prefixes; every window is backward-looking."""
    generation_dir = generation_dir.expanduser().resolve()
    signals = generation_dir / "daily_causal_signals.parquet"
    memberships = generation_dir / "story_day_membership.parquet"
    snapshots = generation_dir / "event_state_snapshots.parquet"
    events = generation_dir / "event_windows.parquet"
    for path in (signals, memberships, snapshots, events):
        if not path.is_file():
            raise FileNotFoundError(f"Generation is missing required artifact: {path}")
    with duckdb.connect(":memory:") as connection:
        description = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(memberships)]
        ).description
    membership_columns = {str(item[0]) for item in description or []}
    required_membership = {
        "daily_pressure_rank", "daily_response_class", "pressure_observed"
    }
    missing_membership = sorted(required_membership - membership_columns)
    if missing_membership:
        raise ValueError(
            "Generation predates event-specific prefix evidence; rebuild it before motif "
            "training. Missing story_day_membership columns: "
            + ", ".join(missing_membership)
        )
    source_cutoff_clause = ""
    cutoff_clause = ""
    params: list[object] = [str(memberships), str(signals), str(events)]
    if through:
        source_cutoff_clause = "AND CAST(m.observation_date AS DATE) <= CAST(? AS DATE)"
        params.append(through)
    params.append(str(snapshots))
    if through:
        # A weekly snapshot is the latest state in its Monday-Sunday bucket.
        # Exclude the boundary week unless it is fully inside the cutoff so a
        # Dec-31 train split cannot ingest Jan-1..4 evidence.
        cutoff_clause = "WHERE timeline_bucket + INTERVAL 6 DAY <= CAST(? AS DATE)"
        params.append(through)
    sampling = ""
    if sample_age_buckets:
        sampling = """
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY event_id,
                CASE
                    WHEN event_age_days <= 7 THEN 0
                    WHEN event_age_days <= 21 THEN 1
                    WHEN event_age_days <= 49 THEN 2
                    ELSE 3
                END
            ORDER BY timeline_bucket DESC
        ) = 1
        """
    with duckdb.connect(":memory:") as connection:
        return connection.execute(
            f"""
            WITH joined AS (
                SELECT
                    m.event_id,
                    m.field_id,
                    m.crop_instance_id,
                    CAST(m.observation_date AS DATE) AS observation_date,
                    DATE_TRUNC('week', CAST(m.observation_date AS DATE))::DATE AS timeline_bucket,
                    s.crop_name,
                    s.crop_season,
                    e.hazard_signature AS hazard_family,
                    s.stage_family,
                    m.daily_pressure_rank AS risk_rank,
                    m.daily_response_class AS response_class,
                    CASE WHEN m.daily_response_class IN ('medium_decline', 'severe_decline', 'recovery')
                        THEN s.ndvi_delta END AS ndvi_delta,
                    CASE WHEN m.daily_response_class IN ('medium_decline', 'severe_decline', 'recovery')
                        THEN s.ndmi_delta END AS ndmi_delta,
                    CASE WHEN m.daily_response_class IN ('medium_decline', 'severe_decline', 'recovery')
                        THEN s.psri_delta END AS psri_delta,
                    s.is_new_acquisition,
                    s.spectral_echo_days,
                    s.spi_index,
                    s.ponding_mm,
                    COALESCE(s.apparent_temperature, s.temperature) AS apparent_temperature,
                    s.wind_speed
                FROM read_parquet(?) AS m
                JOIN read_parquet(?) AS s
                  ON s.field_id = m.field_id
                 AND s.crop_instance_id = m.crop_instance_id
                 AND CAST(s.observation_date AS DATE) = CAST(m.observation_date AS DATE)
                JOIN read_parquet(?) AS e USING (event_id)
                WHERE (
                    COALESCE(m.pressure_observed, TRUE)
                    OR m.daily_response_class IN ('medium_decline', 'severe_decline', 'recovery')
                )
                {source_cutoff_clause}
            ),
            changes AS (
                SELECT
                    *,
                    CASE WHEN risk_rank > LAG(risk_rank) OVER event_order THEN 1 ELSE 0 END AS escalated,
                    CASE WHEN risk_rank < LAG(risk_rank) OVER event_order THEN 1 ELSE 0 END AS deescalated,
                    CASE WHEN stage_family IS DISTINCT FROM LAG(stage_family) OVER event_order THEN 1 ELSE 0 END AS stage_changed
                FROM joined
                WINDOW event_order AS (PARTITION BY event_id ORDER BY observation_date)
            ),
            cumulative AS (
                SELECT
                    event_id,
                    field_id,
                    crop_instance_id,
                    crop_name,
                    crop_season,
                    hazard_family,
                    timeline_bucket,
                    observation_date,
                    risk_rank AS current_risk_rank,
                    MAX(risk_rank) OVER prefix AS peak_risk_rank,
                    AVG(risk_rank) OVER prefix AS mean_risk_rank,
                    (
                        risk_rank - FIRST_VALUE(risk_rank) OVER prefix
                    ) / GREATEST(1, DATE_DIFF('day', MIN(observation_date) OVER prefix, observation_date)) AS risk_slope,
                    SUM(escalated) OVER prefix AS escalation_count,
                    SUM(deescalated) OVER prefix AS deescalation_count,
                    DATE_DIFF('day', MIN(observation_date) OVER prefix, observation_date) + 1 AS event_age_days,
                    SUM(CASE WHEN risk_rank >= 3 OR response_class IN ('medium_decline', 'severe_decline') THEN 1 ELSE 0 END) OVER prefix AS active_days,
                    SUM(CASE WHEN risk_rank >= 4 OR response_class = 'severe_decline' THEN 1 ELSE 0 END) OVER prefix AS severe_days,
                    SUM(CASE WHEN risk_rank < 3 THEN 1 ELSE 0 END) OVER prefix AS quiet_days,
                    GREATEST(0, SUM(stage_changed) OVER prefix - 1) AS stage_transition_count,
                    LAST_VALUE(ndvi_delta IGNORE NULLS) OVER prefix AS delta_ndvi,
                    LAST_VALUE(ndmi_delta IGNORE NULLS) OVER prefix AS delta_ndmi,
                    LAST_VALUE(psri_delta IGNORE NULLS) OVER prefix AS delta_psri,
                    SUM(CASE WHEN is_new_acquisition THEN 1 ELSE 0 END) OVER prefix AS unique_acquisition_count,
                    spectral_echo_days AS spectral_age_days,
                    MIN(spi_index) OVER prefix AS min_spi,
                    MAX(ponding_mm) OVER prefix AS max_ponding_mm,
                    MAX(apparent_temperature) OVER prefix AS max_apparent_temperature,
                    MAX(wind_speed) OVER prefix AS max_wind_speed,
                    COUNT(*) OVER prefix::DOUBLE
                        / GREATEST(1, DATE_DIFF('day', MIN(observation_date) OVER prefix, observation_date) + 1)
                        AS observed_coverage,
                    stage_family,
                    response_class
                FROM changes
                WINDOW prefix AS (
                    PARTITION BY event_id ORDER BY observation_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
            ),
            weekly AS (
                SELECT *
                FROM cumulative
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY event_id, timeline_bucket ORDER BY observation_date DESC
                ) = 1
            ),
            enriched AS (
                SELECT
                    weekly.*,
                    COALESCE(snap.event_state, 'UNKNOWN') AS lifecycle_state
                FROM weekly
                LEFT JOIN read_parquet(?) AS snap
                  ON snap.event_id = weekly.event_id
                 AND CAST(snap.timeline_bucket AS DATE) = weekly.timeline_bucket
            ),
            filtered AS (
                SELECT * FROM enriched
                {cutoff_clause}
            )
            SELECT * FROM filtered
            {sampling}
            ORDER BY hazard_family, event_id, timeline_bucket
            """,
            params,
        ).fetchdf()
