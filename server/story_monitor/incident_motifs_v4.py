"""Cadence-aware, diagnostic motif learning for Incident V3 stories.

Incident identity and lifecycle remain owned by the deterministic V3 tracker.
This module only learns optional, reviewed pattern tags from V3 checkpoints plus
daily pressure and sparse Sentinel-2 evidence.  It has no map publication path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "incident-motif-features-v4/1"
MODEL_SCHEMA_VERSION = "incident-motif-model-v4/1"
PREFIX_SCHEMA_VERSION = "incident-motif-prefix-v4/1"

ELIGIBLE_OPERATIONAL_CLOSURES = frozenset(
    {
        "CLOSED_RECOVERED",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED",
        "CLOSED_RESPONSE_UNRESOLVED",
    }
)
EXCLUDED_TERMINAL_REASONS = {
    "CLOSED_CANDIDATE_EXPIRED": "candidate_expired",
    "CLOSED_DATA_CENSORED": "data_censored",
    "CLOSED_SEASON_CENSORED": "season_censored",
    "CLOSED_SEASON_BOUNDARY": "season_censored",
    "MERGED_INTO": "merged_fragment",
    "CLOSED_WATCH_QUIET": "watch_only_not_incident",
}

# Stage is deliberately absent. It remains an audit/facet output with zero
# distance weight, so phenology cannot become motif identity by accident.
MODEL_FEATURE_COLUMNS = (
    "duration_days",
    "checkpoint_count",
    "weather_observed_day_count",
    "weather_coverage_fraction",
    "pressure_day_fraction",
    "severe_pressure_day_fraction",
    "weather_intensity_mean",
    "weather_intensity_peak",
    "weather_intensity_slope",
    "weather_cumulative_intensity",
    "weather_intensity_missing_fraction",
    "affected_rate_mean",
    "affected_rate_peak",
    "severe_affected_fraction",
    "maximum_observed_area_km2",
    "observed_footprint_fraction",
    "data_gap_fraction",
    "relapse_count",
    "s2_usable_acquisition_count",
    "s2_crop_instance_coverage_fraction",
    "s2_echo_age_mean",
    "s2_echo_age_max",
    "s2_decline_fraction",
    "s2_recovery_fraction",
    "s2_ndvi_delta_mean",
    "s2_ndvi_delta_min",
    "s2_ndmi_delta_mean",
    "s2_ndmi_delta_min",
    "s2_psri_delta_mean",
    "s2_psri_delta_max",
)

STAGE_AUDIT_COLUMNS = (
    "dominant_stage",
    "stage_entropy",
    "stage_distribution_json",
)


@dataclass(frozen=True)
class MotifDiscoveryConfig:
    min_cluster_size: int = 100
    min_samples: int = 20
    diagnostic_radius_quantile: float = 0.95
    engine: str = "cpu"

    def validate(self) -> None:
        if self.min_cluster_size < 2 or self.min_samples < 1:
            raise ValueError("invalid HDBSCAN support configuration")
        if not 0.5 <= self.diagnostic_radius_quantile < 1:
            raise ValueError("diagnostic_radius_quantile must be in [0.5, 1)")
        if self.engine not in {"cpu", "gpu"}:
            raise ValueError("engine must be cpu or gpu")


@dataclass(frozen=True)
class PrefixCalibrationConfig:
    weather_day_horizons: tuple[int, ...] = (7, 14, 28, 56)
    s2_acquisition_horizons: tuple[int, ...] = (0, 1, 2, 4)
    minimum_training_support: int = 20
    minimum_calibration_support: int = 10
    radius_quantile: float = 0.95
    margin_quantile: float = 0.05

    def validate(self) -> None:
        if (
            not self.weather_day_horizons
            or any(value < 1 for value in self.weather_day_horizons)
            or tuple(sorted(set(self.weather_day_horizons))) != self.weather_day_horizons
        ):
            raise ValueError("weather horizons must be sorted unique positive integers")
        if (
            not self.s2_acquisition_horizons
            or self.s2_acquisition_horizons[0] != 0
            or any(value < 0 for value in self.s2_acquisition_horizons)
            or tuple(sorted(set(self.s2_acquisition_horizons)))
            != self.s2_acquisition_horizons
        ):
            raise ValueError("S2 horizons must be sorted unique integers beginning at zero")
        if self.minimum_training_support < 2 or self.minimum_calibration_support < 1:
            raise ValueError("prefix support thresholds are invalid")
        if not 0.5 <= self.radius_quantile < 1 or not 0 <= self.margin_quantile <= 0.5:
            raise ValueError("prefix calibration quantiles are invalid")


@dataclass(frozen=True)
class CompletedMotifModel:
    feature_schema: dict[str, Any]
    prototypes: pd.DataFrame
    catalog: pd.DataFrame
    assignments: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PrefixMotifModel:
    feature_schema: dict[str, Any]
    prototypes: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class IncidentDailyEvidence:
    """Causally joined incident-level views of field evidence."""

    daily_pressure: pd.DataFrame
    s2_acquisitions: pd.DataFrame


def build_causal_incident_evidence(
    incident_membership: pd.DataFrame,
    field_daily_pressure: pd.DataFrame,
    field_s2_acquisitions: pd.DataFrame,
) -> IncidentDailyEvidence:
    """Join raw V4 field evidence to V3 story ownership without future leakage.

    Ownership is taken from the V3 membership snapshot covering the evidence
    source week.  A joined row is not available until both that ownership
    checkpoint and the modality-specific evidence are known.
    """

    membership = _prepare_incident_membership(incident_membership)
    pressure = _join_field_pressure(membership, field_daily_pressure)
    s2 = _join_field_s2(membership, field_s2_acquisitions)
    return IncidentDailyEvidence(pressure, s2)


def build_eligibility_ledger(
    incident_windows: pd.DataFrame,
    incident_lineage: pd.DataFrame | None = None,
    incident_weekly_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one explicit learning decision for every incident window."""

    windows = incident_windows.copy()
    _require(windows, ("incident_id", "exposure_id", "crop_name", "hazard_family"), "windows")
    terminal_name = _first_column(windows, ("terminal_state", "final_state"))
    if terminal_name is None:
        raise ValueError("windows require terminal_state")
    if windows["incident_id"].astype(str).duplicated().any():
        raise ValueError("eligibility requires one window per incident_id")
    windows["incident_id"] = _nonblank(windows["incident_id"], "incident_id")
    windows["exposure_id"] = _nonblank(windows["exposure_id"], "exposure_id")
    windows["crop_name"] = _dimension(windows["crop_name"])
    windows["hazard_family"] = _dimension(windows["hazard_family"])
    windows["terminal_state"] = windows[terminal_name].fillna("").astype(str).str.upper()
    right = (
        windows["right_censored"].fillna(False).astype(bool)
        if "right_censored" in windows
        else windows["terminal_state"].eq("")
    )
    confirmed_name = _first_column(windows, ("confirmed_time", "confirmed_date", "confirmed_week"))
    confirmed = (
        pd.to_datetime(windows[confirmed_name], errors="coerce").notna()
        if confirmed_name
        else ~windows["terminal_state"].eq("CLOSED_CANDIDATE_EXPIRED")
    )
    first_name = _first_column(
        windows, ("first_available_at", "first_knowledge_time", "first_evidence_date", "first_evidence_week")
    )
    end_name = _first_column(
        windows, ("feature_available_at", "closed_knowledge_time", "knowledge_time", "closed_date", "closed_week")
    )
    first = _time_column(windows, first_name, "window start") if first_name else pd.Series(pd.NaT, index=windows.index)
    end = _time_column(windows, end_name, "window availability") if end_name else pd.Series(pd.NaT, index=windows.index)

    if incident_weekly_state is not None and not incident_weekly_state.empty:
        weekly = incident_weekly_state.copy()
        _require(weekly, ("incident_id", "knowledge_time"), "weekly state")
        weekly["incident_id"] = weekly["incident_id"].astype(str)
        knowledge = pd.to_datetime(weekly["knowledge_time"], errors="coerce", utc=True)
        if knowledge.isna().any():
            raise ValueError("weekly state contains invalid knowledge_time")
        grouped = pd.DataFrame({"incident_id": weekly["incident_id"], "knowledge": knowledge}).groupby(
            "incident_id", sort=False
        )["knowledge"].agg(["min", "max"])
        first = windows["incident_id"].map(grouped["min"]).fillna(first)
        end = windows["incident_id"].map(grouped["max"]).fillna(end)
    if first.isna().any() or end.isna().any() or (first > end).any():
        raise ValueError("eligibility requires ordered first and final knowledge times")

    family = _lineage_families(windows, incident_lineage)
    reasons: list[str] = []
    eligible: list[bool] = []
    for state, is_right, is_confirmed in zip(windows["terminal_state"], right, confirmed):
        if bool(is_right):
            reason = "right_censored"
        elif state in EXCLUDED_TERMINAL_REASONS:
            reason = EXCLUDED_TERMINAL_REASONS[state]
        elif not bool(is_confirmed):
            reason = "not_confirmed"
        elif state in ELIGIBLE_OPERATIONAL_CLOSURES:
            reason = "eligible_operational_closure"
        else:
            reason = "unsupported_terminal_state"
        reasons.append(reason)
        eligible.append(reason == "eligible_operational_closure")
    return pd.DataFrame(
        {
            "incident_id": windows["incident_id"],
            "exposure_id": windows["exposure_id"],
            "crop_name": windows["crop_name"],
            "hazard_family": windows["hazard_family"],
            "lineage_family_id": windows["incident_id"].map(family),
            "purge_group_id": windows["incident_id"].map(family),
            "first_available_at": first,
            "feature_available_at": end,
            "terminal_state": windows["terminal_state"],
            "right_censored": right.to_numpy(bool),
            "confirmed": confirmed.to_numpy(bool),
            "eligible": eligible,
            "eligibility_reason": reasons,
        }
    ).sort_values("incident_id", kind="mergesort").reset_index(drop=True)


def temporal_split_ledger(
    ledger: pd.DataFrame,
    *,
    train_through: Any,
    calibration_through: Any,
    evaluation_through: Any | None = None,
) -> pd.DataFrame:
    """Create knowledge-time splits and purge later lineage/exposure relatives."""

    _require(
        ledger,
        ("incident_id", "first_available_at", "feature_available_at", "purge_group_id"),
        "eligibility ledger",
    )
    output = ledger.copy()
    start = pd.to_datetime(output["first_available_at"], errors="coerce", utc=True)
    end = pd.to_datetime(output["feature_available_at"], errors="coerce", utc=True)
    train = _through_timestamp(train_through)
    calibration = _through_timestamp(calibration_through)
    evaluation = _through_timestamp(evaluation_through) if evaluation_through is not None else pd.Timestamp.max.tz_localize("UTC")
    if not train < calibration < evaluation:
        raise ValueError("temporal boundaries must satisfy train < calibration < evaluation")
    split = np.full(len(output), "embargo_crossing_or_out_of_range", dtype=object)
    split[end <= train] = "train"
    split[(start > train) & (end <= calibration)] = "calibration"
    split[(start > calibration) & (end <= evaluation)] = "holdout"
    output["temporal_split"] = split
    rank = {"train": 0, "calibration": 1, "holdout": 2}
    for _, indexes in output.groupby("purge_group_id", sort=False).groups.items():
        positions = list(indexes)
        present = [rank[value] for value in output.loc[positions, "temporal_split"] if value in rank]
        if len(set(present)) <= 1:
            continue
        keep = min(present)
        for index in positions:
            value = output.at[index, "temporal_split"]
            if value in rank and rank[value] != keep:
                output.at[index, "temporal_split"] = "embargo_lineage_or_exposure_purge"
    output["train_through"] = train
    output["calibration_through"] = calibration
    output["evaluation_through"] = evaluation
    return output


def build_completed_story_features(
    incident_weekly_state: pd.DataFrame,
    daily_pressure: pd.DataFrame,
    day_membership: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Build one cadence-aware vector per eligible operational closure."""

    eligible = eligibility[eligibility["eligible"].fillna(False).astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for item in eligible.to_dict("records"):
        cutoff = _timestamp(item["feature_available_at"])
        rows.append(
            _aggregate_as_of(
                str(item["incident_id"]), cutoff, item,
                incident_weekly_state, daily_pressure, day_membership,
            )
        )
    return _feature_frame(rows, prefix=False)


def build_causal_prefix_features(
    incident_weekly_state: pd.DataFrame,
    daily_pressure: pd.DataFrame,
    day_membership: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    config: PrefixCalibrationConfig = PrefixCalibrationConfig(),
) -> pd.DataFrame:
    """Replay every available daily prefix using exact weather/S2 maturity bins."""

    config.validate()
    pressure = _prepare_daily(daily_pressure)
    rows: list[dict[str, Any]] = []
    lookup = eligibility.set_index("incident_id", drop=False)
    for incident_id, group in pressure.groupby("incident_id", sort=True):
        if str(incident_id) not in lookup.index:
            raise ValueError(f"daily pressure references unknown incident {incident_id}")
        item = lookup.loc[str(incident_id)]
        if isinstance(item, pd.DataFrame):
            raise ValueError("eligibility contains duplicate incident_id")
        upper = _timestamp(item["feature_available_at"])
        lower = group["feature_available_at"].min()
        as_of_values = _prefix_knowledge_times(
            str(incident_id),
            group,
            incident_weekly_state,
            day_membership,
            lower=_timestamp(lower),
            upper=upper,
        )
        for value in as_of_values:
            cutoff = _timestamp(value)
            record = _aggregate_as_of(
                str(incident_id), cutoff, item.to_dict(),
                incident_weekly_state, pressure, day_membership,
            )
            weather_count = int(record["weather_observed_day_count"])
            s2_count = int(record["s2_usable_acquisition_count"])
            record.update(
                {
                    "feature_schema_version": PREFIX_SCHEMA_VERSION,
                    "prefix_as_of_time": cutoff,
                    "weather_day_horizon": _maturity(weather_count, config.weather_day_horizons),
                    "s2_acquisition_horizon": _maturity(s2_count, config.s2_acquisition_horizons),
                }
            )
            rows.append(record)
    return _feature_frame(rows, prefix=True)


def discover_completed_motifs(
    training_features: pd.DataFrame,
    *,
    training_through: Any,
    config: MotifDiscoveryConfig = MotifDiscoveryConfig(),
    provenance: Mapping[str, Any] | None = None,
) -> CompletedMotifModel:
    """Discover deterministic, unreviewed crop×hazard completed-story motifs."""

    config.validate()
    rows = training_features.copy().reset_index(drop=True)
    _validate_features(rows, prefix=False)
    if rows.empty:
        raise ValueError("completed motif discovery requires training stories")
    cutoff = _through_timestamp(training_through)
    available = pd.to_datetime(rows["feature_available_at"], errors="coerce", utc=True)
    if available.isna().any() or (available > cutoff).any():
        raise ValueError("training features contain post-cutoff knowledge")
    schema = _fit_schema(rows, ("crop_name", "hazard_family"))
    matrix = _transform(rows, schema)
    scope = _sha({"schema": schema, "config": asdict(config), "cutoff": cutoff.isoformat(), "provenance": dict(provenance or {})})[:16]
    labels = np.full(len(rows), -1, dtype=int)
    memberships = np.zeros(len(rows), dtype=float)
    motif_ids = np.full(len(rows), None, dtype=object)
    prototype_records: list[dict[str, Any]] = []
    catalog_records: list[dict[str, Any]] = []
    next_label = 0
    backend = _resolve_engine(config.engine)
    for (crop, hazard), indexes in rows.groupby(["crop_name", "hazard_family"], sort=True).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        if len(positions) < config.min_cluster_size:
            continue
        local_labels, local_membership = _fit_hdbscan(matrix[positions], config, backend)
        for local_label in sorted(int(value) for value in np.unique(local_labels) if value >= 0):
            mask = local_labels == local_label
            member_positions = positions[mask]
            if len(member_positions) < config.min_cluster_size:
                continue
            vectors = matrix[member_positions]
            median = np.median(vectors, axis=0)
            center = vectors[int(np.argmin(np.linalg.norm(vectors - median, axis=1)))]
            distances = np.linalg.norm(vectors - center, axis=1)
            motif_id = "incident-motif:" + _sha([crop, hazard, scope, np.round(center, 8).tolist()])[:20]
            labels[member_positions] = next_label
            memberships[member_positions] = local_membership[mask]
            motif_ids[member_positions] = motif_id
            record = {
                "discovered_motif_id": motif_id,
                "crop_name": crop,
                "hazard_family": hazard,
                "member_count": len(member_positions),
                "diagnostic_radius": max(float(np.quantile(distances, config.diagnostic_radius_quantile)), 1e-9),
                **{f"f_{i:03d}": float(value) for i, value in enumerate(center)},
            }
            prototype_records.append(record)
            catalog_records.append(
                {
                    "discovered_motif_id": motif_id,
                    "crop_name": crop,
                    "hazard_family": hazard,
                    "member_count": len(member_positions),
                    "status": "diagnostic_unreviewed",
                }
            )
            next_label += 1
    if not prototype_records:
        raise ValueError("discovery produced no supported completed motifs")
    prototypes = pd.DataFrame(prototype_records).sort_values("discovered_motif_id").reset_index(drop=True)
    model_version = "incident-motif-v4-" + _sha(
        {"scope": scope, "prototypes": prototypes.to_dict("records")}
    )[:16]
    schema.update({"model_version": model_version, "schema_version": FEATURE_SCHEMA_VERSION})
    prototypes["model_version"] = model_version
    catalog = pd.DataFrame(catalog_records).sort_values("discovered_motif_id").reset_index(drop=True)
    catalog["model_version"] = model_version
    assignments = rows[["incident_id", "crop_name", "hazard_family"]].copy()
    assignments["discovered_motif_id"] = motif_ids
    assignments["discovery_label"] = labels
    assignments["training_membership"] = memberships
    assignments["accepted"] = labels >= 0
    assignments["model_version"] = model_version
    manifest = {
        "status": "diagnostic_unreviewed",
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": model_version,
        "training_through": cutoff.isoformat(),
        "config": asdict(config),
        "engine_used": backend,
        "training_story_count": len(rows),
        "motif_count": len(prototypes),
        "noise_or_unsupported_count": int((labels < 0).sum()),
        "stage_distance_weight": 0.0,
        "incident_identity_preserved": True,
        "publication_status": "blocked_pending_review_and_evaluation",
        "provenance": dict(provenance or {}),
    }
    return CompletedMotifModel(schema, prototypes, catalog, assignments, manifest)


def build_review_overlay_template(model: CompletedMotifModel) -> pd.DataFrame:
    """Return an immutable-review template; no discovered label is auto-approved."""

    overlay = model.catalog[["model_version", "discovered_motif_id", "crop_name", "hazard_family"]].copy()
    overlay["review_status"] = "pending"
    overlay["reviewed_motif_id"] = None
    overlay["display_name"] = None
    overlay["narrative"] = None
    overlay["review_version"] = None
    return overlay


def reviewed_incident_assignments(
    discovery_assignments: pd.DataFrame, review_overlay: pd.DataFrame
) -> pd.DataFrame:
    _require(discovery_assignments, ("incident_id", "discovered_motif_id", "model_version"), "assignments")
    _require(review_overlay, ("discovered_motif_id", "model_version", "review_status", "reviewed_motif_id"), "review overlay")
    if review_overlay.duplicated(["model_version", "discovered_motif_id"]).any():
        raise ValueError("review overlay keys must be unique")
    merged = discovery_assignments.merge(
        review_overlay[["model_version", "discovered_motif_id", "review_status", "reviewed_motif_id"]],
        on=["model_version", "discovered_motif_id"], how="left", validate="many_to_one",
    )
    approved = merged["review_status"].isin({"approved", "merged"}) & merged["reviewed_motif_id"].notna()
    merged.loc[~approved, "reviewed_motif_id"] = None
    return merged


def fit_calibrated_prefix_model(
    prefix_features: pd.DataFrame,
    reviewed_assignments: pd.DataFrame,
    split_ledger: pd.DataFrame,
    *,
    model_version: str,
    config: PrefixCalibrationConfig = PrefixCalibrationConfig(),
) -> PrefixMotifModel:
    """Fit train-only centers and calibration-only radii/margins."""

    config.validate()
    _validate_features(prefix_features, prefix=True)
    _require(reviewed_assignments, ("incident_id", "reviewed_motif_id"), "reviewed assignments")
    _require(split_ledger, ("incident_id", "temporal_split"), "split ledger")
    labels = reviewed_assignments[["incident_id", "reviewed_motif_id"]].dropna().copy()
    if labels["incident_id"].duplicated().any():
        raise ValueError("reviewed assignments must be unique by incident_id")
    rows = prefix_features.merge(labels, on="incident_id", how="inner", validate="many_to_one")
    rows = rows.merge(
        split_ledger[["incident_id", "temporal_split"]], on="incident_id", how="inner", validate="many_to_one"
    )
    maturity = ("crop_name", "hazard_family", "weather_day_horizon", "s2_acquisition_horizon")
    # One deterministic landmark per incident/maturity prevents long dwell times
    # from giving one story repeated training weight.
    rows = rows.sort_values("prefix_as_of_time", kind="mergesort").groupby(
        ["incident_id", *maturity], as_index=False, sort=True
    ).tail(1)
    train = rows[rows["temporal_split"] == "train"].reset_index(drop=True)
    calibration = rows[rows["temporal_split"] == "calibration"].reset_index(drop=True)
    if train.empty or calibration.empty:
        raise ValueError("prefix fitting requires nonempty train and calibration cohorts")
    schema = _fit_schema(train, maturity)
    train_matrix = _transform(train, schema)
    train = train.copy()
    train["_position"] = np.arange(len(train))
    centers: list[dict[str, Any]] = []
    for key, group in train.groupby([*maturity, "reviewed_motif_id"], sort=True):
        if len(group) < config.minimum_training_support:
            continue
        vectors = train_matrix[group["_position"].to_numpy(int)]
        center = np.median(vectors, axis=0)
        centers.append(
            {
                **dict(zip((*maturity, "reviewed_motif_id"), key)),
                "training_support": len(group),
                **{f"f_{i:03d}": float(value) for i, value in enumerate(center)},
            }
        )
    prototypes = pd.DataFrame(centers)
    if prototypes.empty:
        raise ValueError("no reviewed prefix center met training support")
    supported_calibration = _supported_schema_rows(calibration, schema)
    calibration = calibration.loc[supported_calibration].reset_index(drop=True)
    if calibration.empty:
        raise ValueError("calibration cohort has no train-supported maturity strata")
    calibration_matrix = _transform(calibration, schema)
    vector_columns = [f"f_{i:03d}" for i in range(len(MODEL_FEATURE_COLUMNS))]
    calibrated: list[dict[str, Any]] = []
    for index, prototype in prototypes.iterrows():
        mask = np.ones(len(calibration), dtype=bool)
        for name in maturity:
            mask &= calibration[name].astype(str).to_numpy() == str(prototype[name])
        mask &= calibration["reviewed_motif_id"].astype(str).to_numpy() == str(prototype["reviewed_motif_id"])
        positions = np.flatnonzero(mask)
        if len(positions) < config.minimum_calibration_support:
            continue
        center = prototype[vector_columns].to_numpy(float)
        own = np.linalg.norm(calibration_matrix[positions] - center, axis=1)
        record = prototype.to_dict()
        record["radius"] = max(float(np.quantile(own, config.radius_quantile)), 1e-9)
        record["calibration_support"] = len(positions)
        calibrated.append(record)
    prototypes = pd.DataFrame(calibrated)
    if prototypes.empty:
        raise ValueError("no prefix center met calibration support")
    # Calibrate runner-up separation only after every candidate radius is frozen.
    margin_values: dict[tuple[str, ...], list[float]] = {}
    for position, row in calibration.iterrows():
        local = _local_prototypes(prototypes, row, maturity)
        if local.empty or str(row["reviewed_motif_id"]) not in set(local["reviewed_motif_id"].astype(str)):
            continue
        vector = calibration_matrix[position]
        ratios = np.linalg.norm(local[vector_columns].to_numpy(float) - vector, axis=1) / local["radius"].to_numpy(float)
        own_index = int(np.flatnonzero(local["reviewed_motif_id"].astype(str).to_numpy() == str(row["reviewed_motif_id"]))[0])
        other = np.delete(ratios, own_index)
        separation = float(np.min(other) - ratios[own_index]) if len(other) else math.inf
        key = tuple(str(row[name]) for name in (*maturity, "reviewed_motif_id"))
        if np.isfinite(separation):
            margin_values.setdefault(key, []).append(separation)
    margins = []
    for _, row in prototypes.iterrows():
        key = tuple(str(row[name]) for name in (*maturity, "reviewed_motif_id"))
        values = margin_values.get(key, [])
        margins.append(max(0.0, float(np.quantile(values, config.margin_quantile))) if values else 0.0)
    prototypes["runner_up_margin"] = margins
    prefix_version = "incident-prefix-v4-" + _sha(
        {"model": model_version, "schema": schema, "config": asdict(config), "prototypes": prototypes.to_dict("records")}
    )[:16]
    prototypes["model_version"] = prefix_version
    schema.update({"schema_version": PREFIX_SCHEMA_VERSION, "model_version": prefix_version})
    manifest = {
        "status": "frozen_diagnostic",
        "schema_version": PREFIX_SCHEMA_VERSION,
        "model_version": prefix_version,
        "reviewed_completed_model_version": model_version,
        "config": asdict(config),
        "center_fit_split": "train",
        "radius_margin_fit_split": "calibration",
        "stage_distance_weight": 0.0,
        "prototype_count": len(prototypes),
        "publication_status": "blocked_pending_rolling_evaluation",
    }
    return PrefixMotifModel(schema, prototypes.reset_index(drop=True), manifest)


def assign_open_set_prefixes(
    prefix_features: pd.DataFrame, model: PrefixMotifModel
) -> pd.DataFrame:
    """Assign every causal prefix as pending, novel_unassigned, or tentative."""

    _validate_features(prefix_features, prefix=True)
    maturity = ("crop_name", "hazard_family", "weather_day_horizon", "s2_acquisition_horizon")
    vectors = _transform(prefix_features, model.feature_schema, allow_missing_strata=True)
    vector_columns = [f"f_{i:03d}" for i in range(len(MODEL_FEATURE_COLUMNS))]
    output: list[dict[str, Any]] = []
    for position, row in prefix_features.reset_index(drop=True).iterrows():
        local = _local_prototypes(model.prototypes, row, maturity)
        base = {
            "incident_id": row["incident_id"],
            "prefix_as_of_time": row["prefix_as_of_time"],
            **{name: row[name] for name in maturity},
            "model_version": model.manifest["model_version"],
        }
        if local.empty or vectors[position] is None:
            output.append({**base, "assignment_status": "pending", "reviewed_motif_id": None, "candidate_motif_id": None, "distance_ratio": None, "runner_up_separation": None, "assignment_reason": "unsupported_maturity_stratum"})
            continue
        vector = vectors[position]
        ratios = np.linalg.norm(local[vector_columns].to_numpy(float) - vector, axis=1) / local["radius"].to_numpy(float)
        order = np.argsort(ratios, kind="stable")
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        separation = math.inf if second is None else float(ratios[second] - ratios[best])
        accepted = ratios[best] <= 1.0 and separation >= float(local.iloc[best]["runner_up_margin"])
        output.append(
            {
                **base,
                "assignment_status": "tentative" if accepted else "novel_unassigned",
                "reviewed_motif_id": local.iloc[best]["reviewed_motif_id"] if accepted else None,
                "candidate_motif_id": local.iloc[best]["reviewed_motif_id"],
                "distance_ratio": float(ratios[best]),
                "runner_up_separation": separation if np.isfinite(separation) else None,
                "assignment_reason": "within_calibrated_radius_and_margin" if accepted else "outside_radius_or_ambiguous",
            }
        )
    return pd.DataFrame(output)


def evaluate_prefix_replay(
    assignments: pd.DataFrame,
    final_labels: pd.DataFrame,
    split_ledger: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate sealed holdout daily replay without changing a frozen model."""

    _require(assignments, ("incident_id", "prefix_as_of_time", "weather_day_horizon", "s2_acquisition_horizon", "assignment_status", "reviewed_motif_id"), "prefix assignments")
    _require(final_labels, ("incident_id", "final_assignment_status", "reviewed_motif_id"), "final labels")
    _require(
        split_ledger,
        ("incident_id", "temporal_split", "feature_available_at", "purge_group_id"),
        "split ledger",
    )
    if assignments.duplicated(["incident_id", "prefix_as_of_time"]).any():
        raise ValueError("replay assignments are not unique by incident/as-of")
    if split_ledger["incident_id"].astype(str).duplicated().any():
        raise ValueError("split ledger must be unique by incident_id")
    if final_labels["incident_id"].astype(str).duplicated().any():
        raise ValueError("final labels must be unique by incident_id")
    allowed_final = {"accepted", "novel_unassigned"}
    if not set(final_labels["final_assignment_status"].astype(str)) <= allowed_final:
        raise ValueError("final labels contain an unsupported assignment status")
    invalid_known = final_labels["final_assignment_status"].eq("accepted") & final_labels[
        "reviewed_motif_id"
    ].isna()
    if invalid_known.any():
        raise ValueError("accepted final labels require reviewed_motif_id")
    holdout_ids = set(split_ledger.loc[split_ledger["temporal_split"] == "holdout", "incident_id"].astype(str))
    expected_holdout_ids = set(assignments["incident_id"].astype(str)) & holdout_ids
    supplied_final_ids = set(final_labels["incident_id"].astype(str))
    replay = assignments[assignments["incident_id"].astype(str).isin(holdout_ids)].merge(
        final_labels, on="incident_id", how="inner", suffixes=("_prefix", "_final"), validate="many_to_one"
    ).merge(
        split_ledger[["incident_id", "feature_available_at"]],
        on="incident_id",
        how="inner",
        validate="many_to_one",
    )
    if replay.empty:
        raise ValueError("rolling replay has no sealed holdout rows")
    final_known = replay["final_assignment_status"].eq("accepted")
    prefix_accepted = replay["assignment_status"].eq("tentative")
    correct = prefix_accepted & final_known & (
        replay["reviewed_motif_id_prefix"].astype(str) == replay["reviewed_motif_id_final"].astype(str)
    )
    final_novel = replay["final_assignment_status"].eq("novel_unassigned")
    by_maturity: list[dict[str, Any]] = []
    for key, frame in replay.groupby(["weather_day_horizon", "s2_acquisition_horizon"], sort=True):
        accepted = frame["assignment_status"].eq("tentative")
        known = frame["final_assignment_status"].eq("accepted")
        matching = accepted & known & (
            frame["reviewed_motif_id_prefix"].astype(str) == frame["reviewed_motif_id_final"].astype(str)
        )
        by_maturity.append(
            {
                "weather_day_horizon": int(key[0]),
                "s2_acquisition_horizon": int(key[1]),
                "prefix_count": len(frame),
                "accepted_coverage": float(accepted.mean()),
                "accepted_known_precision": float(
                    matching.sum() / max(int(accepted.sum()), 1)
                ),
                "final_novel_false_accept_rate": float((accepted & frame["final_assignment_status"].eq("novel_unassigned")).sum() / max(int(frame["final_assignment_status"].eq("novel_unassigned").sum()), 1)),
            }
        )
    by_origin: list[dict[str, Any]] = []
    replay["evaluation_origin"] = pd.to_datetime(
        replay["prefix_as_of_time"], errors="coerce", utc=True
    ).dt.normalize()
    for origin, frame in replay.groupby("evaluation_origin", sort=True):
        accepted = frame["assignment_status"].eq("tentative")
        matching = accepted & frame["final_assignment_status"].eq("accepted") & (
            frame["reviewed_motif_id_prefix"].astype(str)
            == frame["reviewed_motif_id_final"].astype(str)
        )
        novel = frame["final_assignment_status"].eq("novel_unassigned")
        by_origin.append(
            {
                "origin": _timestamp(origin).isoformat(),
                "prefix_count": len(frame),
                "accepted_coverage": float(accepted.mean()),
                "accepted_known_precision": float(
                    matching.sum() / max(int(accepted.sum()), 1)
                ),
                "final_novel_false_accept_rate": float(
                    (accepted & novel).sum() / max(int(novel.sum()), 1)
                ),
            }
        )
    train_groups = set(split_ledger.loc[split_ledger["temporal_split"] == "train", "purge_group_id"].astype(str))
    calibration_groups = set(split_ledger.loc[split_ledger["temporal_split"] == "calibration", "purge_group_id"].astype(str))
    holdout_groups = set(split_ledger.loc[split_ledger["temporal_split"] == "holdout", "purge_group_id"].astype(str))
    hard = {
        "unique_incident_as_of": not assignments.duplicated(["incident_id", "prefix_as_of_time"]).any(),
        "train_calibration_disjoint": not bool(train_groups & calibration_groups),
        "train_holdout_disjoint": not bool(train_groups & holdout_groups),
        "calibration_holdout_disjoint": not bool(calibration_groups & holdout_groups),
        "holdout_only_evaluation": set(replay["incident_id"].astype(str)) <= holdout_ids,
        "complete_holdout_final_labels": expected_holdout_ids <= supplied_final_ids,
        "prefixes_do_not_follow_final_knowledge": bool(
            (
                pd.to_datetime(replay["prefix_as_of_time"], errors="coerce", utc=True)
                <= pd.to_datetime(replay["feature_available_at"], errors="coerce", utc=True)
            ).all()
        ),
    }
    return {
        "status": "complete",
        "phase": "diagnostic_rolling_replay",
        "hard_gates": {"passed": all(hard.values()), "checks": hard},
        "metrics": {
            "holdout_prefix_count": len(replay),
            "accepted_coverage": float(prefix_accepted.mean()),
            "accepted_known_precision": float(
                correct.sum() / max(int(prefix_accepted.sum()), 1)
            ),
            "final_novel_false_accept_rate": float((prefix_accepted & final_novel).sum() / max(int(final_novel.sum()), 1)),
            "by_maturity": by_maturity,
            "by_origin": by_origin,
        },
        "warning": "Engineering replay only; not agronomic or outcome validation.",
    }


def _aggregate_as_of(
    incident_id: str,
    cutoff: pd.Timestamp,
    ledger_row: Mapping[str, Any],
    weekly_state: pd.DataFrame,
    daily_pressure: pd.DataFrame,
    day_membership: pd.DataFrame,
) -> dict[str, Any]:
    pressure = _prepare_daily(daily_pressure)
    local = pressure[(pressure["incident_id"] == incident_id) & (pressure["feature_available_at"] <= cutoff)].copy()
    if local.empty:
        raise ValueError(f"incident {incident_id} has no causal daily pressure through cutoff")
    if (local["timeline_date"] > cutoff).any():
        raise ValueError("daily pressure evidence occurs after prefix cutoff")
    weekly = _prepare_weekly(weekly_state)
    checkpoints = weekly[(weekly["incident_id"] == incident_id) & (weekly["knowledge_time"] <= cutoff)].copy()
    s2 = _usable_s2(day_membership, incident_id, cutoff)
    intensity = _weather_intensity(local, str(ledger_row["hazard_family"]))
    observed_flag = _boolean(local, ("pressure_observed",), default=None)
    observed = observed_flag if observed_flag is not None else np.isfinite(intensity)
    dates = local["timeline_date"].sort_values()
    span = max((dates.max() - dates.min()).days + 1, 1)
    affected = _numeric(local, ("affected_count", "affected_crop_instance_count"), 0.0)
    monitored = _numeric(local, ("monitored_count", "monitored_crop_instance_count"), 0.0)
    severe = _numeric(local, ("severe_count", "severe_crop_instance_count"), 0.0)
    evaluable = _numeric(local, ("evaluable_count", "evaluable_crop_instance_count"), monitored)
    rates = np.divide(affected, monitored, out=np.zeros_like(affected), where=monitored > 0)
    pressure_active = _boolean(local, ("pressure_active",), default=None)
    if pressure_active is None:
        active = _numeric(local, ("active_count", "pressure_core_count"), 0.0)
        pressure_active = (active + severe) > 0
    severe_active = _boolean(local, ("severe_pressure",), default=None)
    if severe_active is None:
        severe_active = severe > 0
    daily_areas = _numeric(local, ("footprint_area_km2",), np.nan)
    daily_observed_footprint = ~_boolean(
        local,
        ("footprint_carried_forward",),
        default=np.zeros(len(local), dtype=bool),
    )
    weekly_areas = _numeric(checkpoints, ("footprint_area_km2",), np.nan)
    weekly_observed_footprint = ~_boolean(
        checkpoints,
        ("footprint_carried_forward",),
        default=np.zeros(len(checkpoints), dtype=bool),
    )
    if np.isfinite(weekly_areas).any():
        areas = weekly_areas
        observed_footprint = weekly_observed_footprint & np.isfinite(weekly_areas)
    else:
        areas = daily_areas
        observed_footprint = daily_observed_footprint & np.isfinite(daily_areas)
    stages = local.get("stage_bucket", pd.Series("unknown", index=local.index)).fillna("unknown").astype(str).str.lower()
    stage_counts = stages.value_counts().rename_axis("stage").reset_index(name="count")
    stage_counts = stage_counts.sort_values(
        ["count", "stage"], ascending=[False, True], kind="mergesort"
    )
    stage_total = max(int(stage_counts["count"].sum()), 1)
    probabilities = stage_counts["count"].to_numpy(float) / stage_total
    stage_entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(max(len(stage_counts), 2)))
    s2_instances = int(s2["crop_instance_id"].nunique()) if not s2.empty else 0
    s2_opportunities = (
        int(s2["spectral_source_date"].dt.normalize().nunique())
        if not s2.empty
        else 0
    )
    monitored_instances = max(int(local.get("monitored_count", pd.Series(0, index=local.index)).max()), 1)
    echo = pd.to_numeric(s2.get("spectral_echo_days", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    response = s2.get("response_class", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    ndvi = pd.to_numeric(s2.get("ndvi_delta", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    ndmi = pd.to_numeric(s2.get("ndmi_delta", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    psri = pd.to_numeric(s2.get("psri_delta", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    states = checkpoints.get("current_state", checkpoints.get("incident_state", pd.Series(dtype=str))).astype(str).str.upper().tolist()
    record = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "incident_id": incident_id,
        "exposure_id": str(ledger_row["exposure_id"]),
        "crop_name": _dimension_value(ledger_row["crop_name"]),
        "hazard_family": _dimension_value(ledger_row["hazard_family"]),
        "lineage_family_id": str(ledger_row["lineage_family_id"]),
        "feature_available_at": cutoff,
        "duration_days": float(span),
        "checkpoint_count": float(len(checkpoints)),
        "weather_observed_day_count": float(observed.sum()),
        "weather_coverage_fraction": float(observed.sum() / span),
        "pressure_day_fraction": float(np.mean(pressure_active)),
        "severe_pressure_day_fraction": float(np.mean(severe_active)),
        "weather_intensity_mean": _finite_mean(intensity),
        "weather_intensity_peak": _finite_max(intensity),
        "weather_intensity_slope": _finite_slope(intensity),
        "weather_cumulative_intensity": float(np.nansum(np.maximum(intensity, 0))) if observed.any() else 0.0,
        "weather_intensity_missing_fraction": float(np.mean(~np.isfinite(intensity))),
        "affected_rate_mean": float(np.mean(rates)),
        "affected_rate_peak": float(np.max(rates)),
        "severe_affected_fraction": float(severe.sum() / max(float(affected.sum()), 1.0)),
        "maximum_observed_area_km2": _finite_max(areas[observed_footprint]),
        "observed_footprint_fraction": (
            float(np.mean(observed_footprint)) if len(observed_footprint) else 0.0
        ),
        "data_gap_fraction": float(np.maximum(monitored - evaluable, 0).sum() / max(float(monitored.sum()), 1.0)),
        "relapse_count": float(sum(value == "RELAPSED" and (i == 0 or states[i - 1] != "RELAPSED") for i, value in enumerate(states))),
        # Maturity is cadence/opportunity based. Multiple fields observed by
        # the same satellite pass increase spatial coverage, not time maturity.
        "s2_usable_acquisition_count": float(s2_opportunities),
        "s2_crop_instance_coverage_fraction": float(s2_instances / monitored_instances),
        "s2_echo_age_mean": _finite_mean(echo),
        "s2_echo_age_max": _finite_max(echo),
        "s2_decline_fraction": float(response.isin({"medium_decline", "severe_decline"}).mean()) if len(response) else 0.0,
        "s2_recovery_fraction": float(response.eq("recovery").mean()) if len(response) else 0.0,
        "s2_ndvi_delta_mean": _finite_mean(ndvi),
        "s2_ndvi_delta_min": _finite_min(ndvi),
        "s2_ndmi_delta_mean": _finite_mean(ndmi),
        "s2_ndmi_delta_min": _finite_min(ndmi),
        "s2_psri_delta_mean": _finite_mean(psri),
        "s2_psri_delta_max": _finite_max(psri),
        "dominant_stage": str(stage_counts.iloc[0]["stage"]) if len(stage_counts) else "unknown",
        "stage_entropy": stage_entropy,
        "stage_distribution_json": json.dumps(
            {
                str(row.stage): float(row.count / stage_total)
                for row in stage_counts.itertuples(index=False)
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for name in MODEL_FEATURE_COLUMNS:
        value = float(record[name])
        record[name] = value if np.isfinite(value) else 0.0
    return record


def _prepare_incident_membership(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    _require(
        output,
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
    for name in ("incident_id", "field_id", "crop_instance_id"):
        output[name] = _nonblank(output[name], name)
    output["hazard_family"] = _dimension(output["hazard_family"])
    output["membership_week"] = _week_start(
        _time_column(output, "timeline_bucket", "membership timeline")
    )
    output["membership_available_at"] = _time_column(
        output, "knowledge_time", "membership knowledge"
    )
    if (output["membership_available_at"] < output["membership_week"]).any():
        raise ValueError("incident membership is known before its source week")
    output["stage_bucket"] = (
        output.get("stage_bucket", pd.Series("unknown", index=output.index))
        .fillna("unknown")
        .astype(str)
        .map(_dimension_value)
    )
    output["membership_role"] = (
        output.get("membership_role", pd.Series("unknown", index=output.index))
        .fillna("unknown")
        .astype(str)
        .map(_dimension_value)
    )
    output["fresh_response_evidence"] = (
        output.get(
            "fresh_response_evidence", pd.Series(False, index=output.index)
        )
        .fillna(False)
        .astype(bool)
    )
    output["response_class"] = (
        output.get("response_class", pd.Series("", index=output.index))
        .fillna("")
        .astype(str)
        .map(_dimension_value)
    )
    key = [
        "incident_id",
        "membership_week",
        "field_id",
        "crop_instance_id",
        "hazard_family",
    ]
    conflicting = output.groupby(key, dropna=False)["membership_available_at"].nunique()
    if (conflicting > 1).any():
        raise ValueError("incident membership key has conflicting knowledge times")
    return (
        output.sort_values([*key, "stage_bucket"], kind="mergesort")
        .drop_duplicates(key, keep="first")
        .loc[
            :,
            [
                *key,
                "membership_available_at",
                "stage_bucket",
                "membership_role",
                "fresh_response_evidence",
                "response_class",
            ],
        ]
        .reset_index(drop=True)
    )


def _join_field_pressure(
    membership: pd.DataFrame, field_pressure: pd.DataFrame
) -> pd.DataFrame:
    pressure = field_pressure.copy()
    date_name = _first_column(
        pressure, ("observation_date", "pressure_observation_date", "timeline_date", "calendar_date")
    )
    available_name = _first_column(
        pressure,
        ("knowledge_time", "weather_available_at", "feature_available_at"),
    )
    if date_name is None or available_name is None:
        raise ValueError("field pressure requires observation and knowledge timestamps")
    _require(
        pressure,
        ("field_id", "crop_instance_id", "hazard_family", "pressure_observed"),
        "field pressure",
    )
    for name in ("field_id", "crop_instance_id"):
        pressure[name] = _nonblank(pressure[name], name)
    pressure["hazard_family"] = _dimension(pressure["hazard_family"])
    pressure["timeline_date"] = _time_column(
        pressure, date_name, "pressure observation"
    ).dt.normalize()
    pressure["pressure_available_at"] = _time_column(
        pressure, available_name, "pressure knowledge"
    )
    if (pressure["pressure_available_at"] < pressure["timeline_date"]).any():
        raise ValueError("field pressure is known before its observation date")
    pressure["membership_week"] = _week_start(pressure["timeline_date"])
    pressure_key = [
        "field_id",
        "crop_instance_id",
        "hazard_family",
        "timeline_date",
    ]
    if pressure.duplicated(pressure_key).any():
        raise ValueError("field pressure contains duplicate natural keys")
    # Stage ownership comes from the V3 incident membership snapshot.  A stage
    # column carried by a weather file is neither authoritative nor needed.
    pressure = pressure.drop(columns=["stage_bucket"], errors="ignore")
    joined = pressure.merge(
        membership,
        on=["membership_week", "field_id", "crop_instance_id", "hazard_family"],
        how="inner",
        validate="many_to_many",
    )
    if joined.empty:
        raise ValueError("field pressure did not join to any V3 incident membership")
    joined["feature_available_at"] = joined[
        ["pressure_available_at", "membership_available_at"]
    ].max(axis=1)
    joined["pressure_observed"] = joined["pressure_observed"].fillna(False).astype(bool)
    score_name = _first_column(
        joined,
        ("pressure_score", "weather_intensity", "pressure_rank", "risk_rank", "daily_pressure_rank"),
    )
    joined["_intensity"] = (
        pd.to_numeric(joined[score_name], errors="coerce")
        if score_name is not None
        else np.nan
    )
    active_name = _first_column(joined, ("pressure_active",))
    rank_name = _first_column(
        joined, ("pressure_rank", "risk_rank", "daily_pressure_rank")
    )
    rank = (
        pd.to_numeric(joined[rank_name], errors="coerce")
        if rank_name is not None
        else pd.Series(np.nan, index=joined.index)
    )
    active = (
        joined[active_name].fillna(False).astype(bool)
        if active_name is not None
        else rank.ge(2)
    )
    instance = joined["field_id"].astype(str) + "\x1f" + joined["crop_instance_id"].astype(str)
    joined["_instance"] = instance
    joined["_evaluable_instance"] = instance.where(joined["pressure_observed"])
    impact = joined["membership_role"].isin(
        {"impact_lag", "unresolved", "unresolved_review", "recovering", "recovered"}
    ) | joined["fresh_response_evidence"]
    severe_impact = joined["fresh_response_evidence"] & joined[
        "response_class"
    ].eq("severe_decline")
    joined["_affected_instance"] = instance.where(impact)
    joined["_severe_instance"] = instance.where(severe_impact)
    joined["_weather_active"] = joined["pressure_observed"] & active
    joined["_weather_severe"] = joined["pressure_observed"] & rank.ge(4)
    joined["_observed_intensity"] = joined["_intensity"].where(
        joined["pressure_observed"]
    )
    group_names = ["incident_id", "timeline_date"]
    summary = joined.groupby(group_names, sort=True, as_index=False).agg(
        feature_available_at=("feature_available_at", "max"),
        pressure_observed=("pressure_observed", "any"),
        weather_intensity=("_observed_intensity", "mean"),
        monitored_count=("_instance", "nunique"),
        evaluable_count=("_evaluable_instance", "nunique"),
        affected_count=("_affected_instance", "nunique"),
        severe_count=("_severe_instance", "nunique"),
        pressure_active=("_weather_active", "any"),
        severe_pressure=("_weather_severe", "any"),
    )
    summary["footprint_area_km2"] = np.nan
    summary["footprint_carried_forward"] = True
    stages = (
        joined.groupby([*group_names, "stage_bucket"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            [*group_names, "count", "stage_bucket"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(group_names, keep="first")
        .rename(columns={"stage_bucket": "stage_bucket_dominant"})
    )
    summary = summary.merge(
        stages[[*group_names, "stage_bucket_dominant"]],
        on=group_names,
        how="left",
        validate="one_to_one",
    ).rename(columns={"stage_bucket_dominant": "stage_bucket"})
    return summary.sort_values(group_names, kind="mergesort").reset_index(drop=True)


def _join_field_s2(
    membership: pd.DataFrame, field_s2: pd.DataFrame
) -> pd.DataFrame:
    if field_s2.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "field_id",
                "crop_instance_id",
                "spectral_source_date",
                "feature_available_at",
                "acquisition_attempted",
                "spectral_usable",
            ]
        )
    s2 = field_s2.copy()
    source_name = _first_column(
        s2, ("spectral_source_date", "acquisition_date", "source_date")
    )
    available_name = _first_column(
        s2,
        ("knowledge_time", "spectral_available_at", "feature_available_at", "known_date"),
    )
    if source_name is None or available_name is None:
        raise ValueError("field S2 evidence requires source and knowledge timestamps")
    _require(s2, ("field_id", "crop_instance_id"), "field S2 evidence")
    for name in ("field_id", "crop_instance_id"):
        s2[name] = _nonblank(s2[name], name)
    s2["spectral_source_date"] = _time_column(
        s2, source_name, "S2 source date"
    ).dt.normalize()
    s2["s2_available_at"] = _time_column(s2, available_name, "S2 knowledge")
    if (s2["s2_available_at"] < s2["spectral_source_date"]).any():
        raise ValueError("S2 evidence is known before its source date")
    s2["membership_week"] = _week_start(s2["spectral_source_date"])
    ownership = membership.drop(columns=["hazard_family"]).drop_duplicates(
        [
            "incident_id",
            "membership_week",
            "field_id",
            "crop_instance_id",
        ]
    )
    joined = s2.merge(
        ownership,
        on=["membership_week", "field_id", "crop_instance_id"],
        how="inner",
        validate="many_to_many",
    )
    joined["feature_available_at"] = joined[
        ["s2_available_at", "membership_available_at"]
    ].max(axis=1)
    key = ["incident_id", "field_id", "crop_instance_id", "spectral_source_date"]
    if "acquisition_id" in joined:
        key.append("acquisition_id")
    if joined.duplicated(key).any():
        raise ValueError("S2 acquisition joins more than once to one incident")
    return joined.sort_values(
        ["feature_available_at", *key], kind="mergesort"
    ).reset_index(drop=True)


def _week_start(values: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(values, errors="coerce", utc=True).dt.normalize()
    return normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")


def _prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    _require(output, ("incident_id", "timeline_date", "feature_available_at"), "daily pressure")
    output["incident_id"] = output["incident_id"].astype(str)
    output["timeline_date"] = pd.to_datetime(output["timeline_date"], errors="coerce", utc=True)
    output["feature_available_at"] = pd.to_datetime(output["feature_available_at"], errors="coerce", utc=True)
    if output[["timeline_date", "feature_available_at"]].isna().any().any():
        raise ValueError("daily pressure contains invalid dates")
    if (output["feature_available_at"] < output["timeline_date"]).any():
        raise ValueError("daily pressure is known before its evidence date")
    if output.duplicated(["incident_id", "timeline_date"]).any():
        raise ValueError("daily pressure must be unique by incident_id and timeline_date")
    return output


def _prepare_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    _require(output, ("incident_id", "knowledge_time"), "weekly checkpoints")
    output["incident_id"] = output["incident_id"].astype(str)
    output["knowledge_time"] = pd.to_datetime(output["knowledge_time"], errors="coerce", utc=True)
    if output["knowledge_time"].isna().any():
        raise ValueError("weekly checkpoints contain invalid knowledge_time")
    if output.duplicated(["incident_id", "knowledge_time"]).any():
        raise ValueError("weekly checkpoints must be unique by incident and knowledge time")
    return output


def _prefix_knowledge_times(
    incident_id: str,
    pressure: pd.DataFrame,
    weekly: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
) -> list[pd.Timestamp]:
    values = {
        _timestamp(value)
        for value in pressure["feature_available_at"].dropna().tolist()
    }
    prepared_weekly = _prepare_weekly(weekly)
    values.update(
        _timestamp(value)
        for value in prepared_weekly.loc[
            prepared_weekly["incident_id"].eq(incident_id), "knowledge_time"
        ].dropna()
    )
    if not s2.empty:
        available_name = _first_column(
            s2,
            (
                "feature_available_at",
                "knowledge_time",
                "spectral_available_at",
                "known_date",
            ),
        )
        if available_name is None or "incident_id" not in s2:
            raise ValueError("joined S2 evidence requires incident and knowledge columns")
        available = pd.to_datetime(s2[available_name], errors="coerce", utc=True)
        if available.isna().any():
            raise ValueError("joined S2 evidence contains invalid knowledge times")
        values.update(
            _timestamp(value)
            for value in available[s2["incident_id"].astype(str).eq(incident_id)]
        )
    return sorted(value for value in values if lower <= value <= upper)


def _usable_s2(frame: pd.DataFrame, incident_id: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    _require(output, ("incident_id", "crop_instance_id"), "day membership")
    source_name = _first_column(
        output, ("spectral_source_date", "acquisition_date", "source_date")
    )
    available_name = _first_column(
        output,
        ("feature_available_at", "knowledge_time", "spectral_available_at", "known_date"),
    )
    attempted_name = _first_column(
        output, ("is_new_acquisition", "acquisition_attempted")
    )
    if source_name is None or available_name is None or attempted_name is None:
        raise ValueError(
            "day membership requires acquisition source, knowledge, and attempted columns"
        )
    if source_name != "spectral_source_date":
        output["spectral_source_date"] = output[source_name]
    if available_name != "feature_available_at":
        output["feature_available_at"] = output[available_name]
    output["incident_id"] = output["incident_id"].astype(str)
    output["spectral_source_date"] = pd.to_datetime(output["spectral_source_date"], errors="coerce", utc=True)
    output["feature_available_at"] = pd.to_datetime(output["feature_available_at"], errors="coerce", utc=True)
    output = output[(output["incident_id"] == incident_id) & (output["feature_available_at"] <= cutoff)].copy()
    output = output[output[attempted_name].fillna(False).astype(bool)]
    if output.empty:
        return output
    usable = (
        output["spectral_usable"].fillna(False).astype(bool)
        if "spectral_usable" in output
        else output["usable_s2"].fillna(False).astype(bool)
        if "usable_s2" in output
        else output["new_response_evidence"].fillna(False).astype(bool)
        if "new_response_evidence" in output
        else output.get("spectral_freshness", pd.Series("fresh", index=output.index)).astype(str).str.lower().isin({"fresh", "aging"})
    )
    output = output[usable].copy()
    key = ["incident_id"]
    if "field_id" in output:
        key.append("field_id")
    key.extend(["crop_instance_id", "spectral_source_date"])
    if output.duplicated(key).any():
        raise ValueError("one usable S2 acquisition was marked new more than once")
    if (output["feature_available_at"] < output["spectral_source_date"]).any():
        raise ValueError("S2 acquisition is known before its source date")
    if "evidence_age_days" in output and "spectral_echo_days" not in output:
        output["spectral_echo_days"] = output["evidence_age_days"]
    derived_echo = (
        cutoff.normalize() - output["spectral_source_date"].dt.normalize()
    ).dt.days
    if "spectral_echo_days" not in output:
        output["spectral_echo_days"] = derived_echo
    else:
        supplied_echo = pd.to_numeric(output["spectral_echo_days"], errors="coerce")
        output["spectral_echo_days"] = supplied_echo.where(
            supplied_echo.notna(), derived_echo
        )
    return output.sort_values(["feature_available_at", *key], kind="mergesort")


def _feature_frame(rows: list[dict[str, Any]], *, prefix: bool) -> pd.DataFrame:
    metadata = [
        "feature_schema_version", "incident_id", "exposure_id", "crop_name",
        "hazard_family", "lineage_family_id", "feature_available_at",
    ]
    if prefix:
        metadata += ["prefix_as_of_time", "weather_day_horizon", "s2_acquisition_horizon"]
    return pd.DataFrame(rows, columns=[*metadata, *MODEL_FEATURE_COLUMNS, *STAGE_AUDIT_COLUMNS])


def _validate_features(frame: pd.DataFrame, *, prefix: bool) -> None:
    required = ["incident_id", "crop_name", "hazard_family", "feature_available_at", *MODEL_FEATURE_COLUMNS]
    if prefix:
        required += ["prefix_as_of_time", "weather_day_horizon", "s2_acquisition_horizon"]
    _require(frame, required, "motif features")
    if prefix and frame.duplicated(["incident_id", "prefix_as_of_time"]).any():
        raise ValueError("prefix features must be unique by incident/as-of")
    values = frame.loc[:, MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("motif features must be finite")


def _fit_schema(rows: pd.DataFrame, strata: Sequence[str]) -> dict[str, Any]:
    records = []
    grouper: Any = strata[0] if len(strata) == 1 else list(strata)
    for key, group in rows.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(strata) == 1 else tuple(key)
        specs = []
        for index, name in enumerate(MODEL_FEATURE_COLUMNS):
            values = pd.to_numeric(group[name], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            q25, q75 = np.quantile(finite, [0.25, 0.75]) if len(finite) else (0.0, 1.0)
            scale = max(float(q75 - q25), 1e-9)
            specs.append({"index": index, "name": name, "median": median, "scale": scale})
        records.append({"key": {name: _dimension_value(value) for name, value in zip(strata, keys)}, "features": specs})
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "strata_columns": list(strata),
        "clip": 5.0,
        "stage_distance_weight": 0.0,
        "strata": records,
    }


def _transform(
    rows: pd.DataFrame, schema: Mapping[str, Any], *, allow_missing_strata: bool = False
) -> Any:
    strata = tuple(schema["strata_columns"])
    lookup = {
        tuple(item["key"][name] for name in strata): item["features"]
        for item in schema["strata"]
    }
    output: list[np.ndarray | None] = []
    for _, row in rows.iterrows():
        key = tuple(_dimension_value(row[name]) for name in strata)
        specs = lookup.get(key)
        if specs is None:
            if allow_missing_strata:
                output.append(None)
                continue
            raise ValueError(f"feature schema has no fitted stratum {key}")
        vector = np.zeros(len(MODEL_FEATURE_COLUMNS), dtype=np.float32)
        for spec in specs:
            value = float(row[spec["name"]])
            vector[int(spec["index"])] = np.float32(
                np.clip((value - spec["median"]) / spec["scale"], -schema["clip"], schema["clip"])
                / math.sqrt(len(MODEL_FEATURE_COLUMNS))
            )
        output.append(vector)
    if allow_missing_strata:
        return output
    matrix = np.asarray(output, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("feature transform produced non-finite values")
    return matrix


def _supported_schema_rows(
    rows: pd.DataFrame, schema: Mapping[str, Any]
) -> np.ndarray:
    strata = tuple(schema["strata_columns"])
    supported = {
        tuple(item["key"][name] for name in strata)
        for item in schema["strata"]
    }
    return np.asarray(
        [
            tuple(_dimension_value(row[name]) for name in strata) in supported
            for _, row in rows.iterrows()
        ],
        dtype=bool,
    )


def _fit_hdbscan(
    matrix: np.ndarray, config: MotifDiscoveryConfig, backend: str
) -> tuple[np.ndarray, np.ndarray]:
    samples = min(config.min_samples, len(matrix))
    if backend == "gpu":
        import cupy
        from cuml.cluster.hdbscan import HDBSCAN
        model = HDBSCAN(min_cluster_size=config.min_cluster_size, min_samples=samples, cluster_selection_method="eom").fit(cupy.asarray(matrix, dtype=cupy.float32))
        return cupy.asnumpy(model.labels_).astype(int), cupy.asnumpy(model.probabilities_).astype(float)
    from sklearn.cluster import HDBSCAN
    model = HDBSCAN(min_cluster_size=config.min_cluster_size, min_samples=samples, cluster_selection_method="eom", copy=True).fit(matrix)
    return model.labels_.astype(int), model.probabilities_.astype(float)


def _resolve_engine(engine: str) -> str:
    if engine == "cpu":
        return "cpu"
    try:
        import cupy
        from cuml.cluster.hdbscan import HDBSCAN  # noqa: F401
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("GPU motif discovery requires usable CuPy/cuML") from exc
    return "gpu"


def _lineage_families(windows: pd.DataFrame, lineage: pd.DataFrame | None) -> dict[str, str]:
    ids = windows["incident_id"].astype(str).tolist()
    parent = {value: value for value in ids}
    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for _, group in windows.groupby("exposure_id", sort=False):
        values = sorted(group["incident_id"].astype(str))
        for value in values[1:]:
            union(values[0], value)
    if lineage is not None and not lineage.empty:
        _require(lineage, ("parent_incident_id", "child_incident_id"), "lineage")
        for row in lineage[["parent_incident_id", "child_incident_id"]].dropna().itertuples(index=False):
            if str(row.parent_incident_id) in parent and str(row.child_incident_id) in parent:
                union(str(row.parent_incident_id), str(row.child_incident_id))
    groups: dict[str, list[str]] = {}
    for value in ids:
        groups.setdefault(find(value), []).append(value)
    result = {}
    for members in groups.values():
        family = "lineage-family:" + _sha(sorted(members))[:16]
        for value in members:
            result[value] = family
    return result


def _weather_intensity(frame: pd.DataFrame, hazard: str) -> np.ndarray:
    direct = _first_column(frame, ("weather_intensity", "weather_pressure_score", "risk_rank", "current_risk_rank"))
    if direct:
        return pd.to_numeric(frame[direct], errors="coerce").to_numpy(float)
    hazard = hazard.lower()
    candidates = (
        ("apparent_temperature", "temperature") if "heat" in hazard
        else ("spi_index",) if "drought" in hazard
        else ("ponding_mm",) if "flood" in hazard or "ponding" in hazard
        else ("wind_speed",) if "wind" in hazard
        else ()
    )
    name = _first_column(frame, candidates)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float) if name else np.full(len(frame), np.nan)
    return -values if name == "spi_index" else values


def _numeric(frame: pd.DataFrame, names: Sequence[str], default: Any) -> np.ndarray:
    name = _first_column(frame, names)
    if name is None:
        if isinstance(default, np.ndarray):
            return default.astype(float)
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _boolean(frame: pd.DataFrame, names: Sequence[str], default: Any) -> Any:
    name = _first_column(frame, names)
    if name is None:
        return default
    return frame[name].astype("boolean").fillna(False).to_numpy(dtype=bool)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else 0.0


def _finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else 0.0


def _finite_slope(values: np.ndarray) -> float:
    positions = np.flatnonzero(np.isfinite(values))
    return float(np.polyfit(positions, values[positions], 1)[0]) if len(positions) >= 2 else 0.0


def _local_prototypes(prototypes: pd.DataFrame, row: pd.Series, names: Sequence[str]) -> pd.DataFrame:
    mask = np.ones(len(prototypes), dtype=bool)
    for name in names:
        mask &= prototypes[name].astype(str).to_numpy() == str(row[name])
    return prototypes.loc[mask].reset_index(drop=True)


def _maturity(count: int, horizons: Sequence[int]) -> int:
    supported = [value for value in horizons if value <= count]
    return max(supported) if supported else 0


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame), None)


def _time_column(frame: pd.DataFrame, name: str, label: str) -> pd.Series:
    values = pd.to_datetime(frame[name], errors="coerce", utc=True)
    if values.isna().any():
        raise ValueError(f"{label} contains invalid timestamps")
    return values


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("timestamp is invalid")
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _through_timestamp(value: Any) -> pd.Timestamp:
    """Interpret a CLI-style YYYY-MM-DD boundary as inclusive through that day."""

    parsed = _timestamp(value)
    if (
        isinstance(value, str) and len(value.strip()) == 10
    ) or (isinstance(value, date) and not isinstance(value, datetime)):
        return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return parsed


def _dimension(values: pd.Series) -> pd.Series:
    return values.map(_dimension_value)


def _dimension_value(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace(" ", "_")


def _nonblank(values: pd.Series, label: str) -> pd.Series:
    output = values.astype(str)
    if values.isna().any() or output.str.strip().eq("").any():
        raise ValueError(f"{label} contains null or blank values")
    return output


def _require(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


__all__ = [
    "CompletedMotifModel",
    "ELIGIBLE_OPERATIONAL_CLOSURES",
    "FEATURE_SCHEMA_VERSION",
    "IncidentDailyEvidence",
    "MODEL_FEATURE_COLUMNS",
    "MotifDiscoveryConfig",
    "PrefixCalibrationConfig",
    "PrefixMotifModel",
    "assign_open_set_prefixes",
    "build_causal_incident_evidence",
    "build_causal_prefix_features",
    "build_completed_story_features",
    "build_eligibility_ledger",
    "build_review_overlay_template",
    "discover_completed_motifs",
    "evaluate_prefix_replay",
    "fit_calibrated_prefix_model",
    "reviewed_incident_assignments",
    "temporal_split_ledger",
]
