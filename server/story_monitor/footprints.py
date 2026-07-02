"""Truthful aggregate motif-footprint summaries for map playback."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def build_motif_weekly_footprints(
    states: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    bucket_column: str = "timeline_bucket",
    motif_column: str = "motif_id",
) -> pd.DataFrame:
    """Summarize changing active sets; this does not infer propagation."""
    required_states = {bucket_column, motif_column, "field_id"}
    required_geometry = {"field_id", "centroid_lon", "centroid_lat"}
    _require(states, required_states, "states")
    _require(geometry, required_geometry, "geometry")

    columns = [
        bucket_column,
        motif_column,
        "motif_family",
        "field_count",
        "event_count",
        "center_lon",
        "center_lat",
        "center_method",
        "dispersion_p50_km",
        "dispersion_p90_km",
        "prior_week",
        "persisting_field_count",
        "entering_field_count",
        "exiting_field_count",
        "field_union_count",
        "field_overlap_jaccard",
        "is_consecutive_week",
        "trail_segment_allowed",
        "trail_break_reason",
        "evidence_scope",
        "is_physical_movement",
    ]
    if states.empty:
        return pd.DataFrame(columns=columns)

    geometry_unique = geometry.drop_duplicates("field_id", keep="last")
    joined = states.merge(
        geometry_unique[["field_id", "centroid_lon", "centroid_lat"]],
        on="field_id",
        how="inner",
        validate="many_to_one",
    )
    joined[bucket_column] = pd.to_datetime(joined[bucket_column], errors="raise").dt.normalize()
    joined = joined.dropna(subset=[motif_column, "centroid_lon", "centroid_lat"])
    records: list[dict[str, Any]] = []
    previous: dict[str, tuple[pd.Timestamp, set[str]]] = {}

    for (motif_id, bucket), group in joined.groupby(
        [motif_column, bucket_column], sort=True, dropna=False
    ):
        field_rows = group.drop_duplicates("field_id", keep="last")
        field_ids = set(field_rows["field_id"].astype(str))
        center_lon, center_lat = spherical_mean(
            field_rows["centroid_lon"].to_numpy(dtype=float),
            field_rows["centroid_lat"].to_numpy(dtype=float),
        )
        distances = haversine_km(
            field_rows["centroid_lon"].to_numpy(dtype=float),
            field_rows["centroid_lat"].to_numpy(dtype=float),
            center_lon,
            center_lat,
        )
        prior = previous.get(str(motif_id))
        prior_week: pd.Timestamp | None = prior[0] if prior else None
        prior_fields = prior[1] if prior else set()
        persisting = field_ids & prior_fields
        entering = field_ids - prior_fields
        exiting = prior_fields - field_ids
        union = field_ids | prior_fields
        consecutive = bool(prior_week is not None and (bucket - prior_week).days == 7)
        overlap = len(persisting) / len(union) if union else 0.0
        if prior is None:
            allowed, break_reason = False, "first_observation"
        elif not consecutive:
            allowed, break_reason = False, "nonconsecutive_week"
        elif not persisting:
            allowed, break_reason = False, "zero_field_overlap"
        else:
            allowed, break_reason = True, None
        family = None
        if "motif_family" in group:
            non_null = group["motif_family"].dropna()
            family = str(non_null.iloc[0]) if len(non_null) else None
        records.append(
            {
                bucket_column: bucket.date().isoformat(),
                motif_column: motif_id,
                "motif_family": family,
                "field_count": len(field_ids),
                "event_count": int(group["event_id"].nunique()) if "event_id" in group else len(field_ids),
                "center_lon": center_lon,
                "center_lat": center_lat,
                "center_method": "unweighted_field_spherical_mean_v1",
                "dispersion_p50_km": float(np.quantile(distances, 0.5)),
                "dispersion_p90_km": float(np.quantile(distances, 0.9)),
                "prior_week": prior_week.date().isoformat() if prior_week is not None else None,
                "persisting_field_count": len(persisting),
                "entering_field_count": len(entering),
                "exiting_field_count": len(exiting),
                "field_union_count": len(union),
                "field_overlap_jaccard": overlap,
                "is_consecutive_week": consecutive,
                "trail_segment_allowed": allowed,
                "trail_break_reason": break_reason,
                "evidence_scope": "aggregate_active_field_set",
                "is_physical_movement": False,
            }
        )
        previous[str(motif_id)] = (bucket, field_ids)
    return pd.DataFrame.from_records(records, columns=columns)


def spherical_mean(longitudes: np.ndarray, latitudes: np.ndarray) -> tuple[float, float]:
    if not len(longitudes) or len(longitudes) != len(latitudes):
        raise ValueError("spherical_mean requires equal non-empty coordinate arrays")
    lon = np.radians(longitudes)
    lat = np.radians(latitudes)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    mean_x, mean_y, mean_z = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
    norm = math.sqrt(mean_x * mean_x + mean_y * mean_y + mean_z * mean_z)
    if norm < 1e-12:
        raise ValueError("activity center is undefined for antipodal coordinates")
    return (
        math.degrees(math.atan2(mean_y, mean_x)),
        math.degrees(math.atan2(mean_z, math.sqrt(mean_x * mean_x + mean_y * mean_y))),
    )


def haversine_km(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> np.ndarray:
    lon = np.radians(longitudes)
    lat = np.radians(latitudes)
    center_lon_rad = math.radians(center_lon)
    center_lat_rad = math.radians(center_lat)
    delta_lon = lon - center_lon_rad
    delta_lat = lat - center_lat_rad
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat) * math.cos(center_lat_rad) * np.sin(delta_lon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0.0, 1 - value)))


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")
