from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from .contracts import MonitorPolicy, RISK_RANKS, normalize_hazard, normalize_risk, normalize_stage, stable_id


SPECTRAL_COLUMNS = ("ndvi", "ndmi", "psri")


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _freshness(echo_days: Any, policy: MonitorPolicy) -> str:
    if echo_days is None or pd.isna(echo_days):
        return "missing"
    age = int(echo_days)
    if age <= policy.freshness_fresh_days:
        return "fresh"
    if age <= policy.freshness_aging_days:
        return "aging"
    return "stale"


def _response_class(
    deltas: dict[str, float | None], policy: MonitorPolicy
) -> tuple[str, str]:
    adverse = {
        "ndvi": deltas["ndvi"] is not None and deltas["ndvi"] <= policy.medium_ndvi_delta,
        "ndmi": deltas["ndmi"] is not None and deltas["ndmi"] <= policy.medium_ndmi_delta,
        "psri": deltas["psri"] is not None and deltas["psri"] >= policy.medium_psri_delta,
    }
    severe = {
        "ndvi": deltas["ndvi"] is not None and deltas["ndvi"] <= policy.severe_ndvi_delta,
        "ndmi": deltas["ndmi"] is not None and deltas["ndmi"] <= policy.severe_ndmi_delta,
        "psri": deltas["psri"] is not None and deltas["psri"] >= policy.severe_psri_delta,
    }
    recovery = {
        "ndvi": deltas["ndvi"] is not None and deltas["ndvi"] >= policy.recovery_ndvi_delta,
        "ndmi": deltas["ndmi"] is not None and deltas["ndmi"] >= policy.recovery_ndmi_delta,
        "psri": deltas["psri"] is not None and deltas["psri"] <= policy.recovery_psri_delta,
    }
    if any(severe.values()):
        return "severe_decline", "+".join(key for key, flag in severe.items() if flag)
    if any(adverse.values()):
        return "medium_decline", "+".join(key for key, flag in adverse.items() if flag)
    if any(recovery.values()):
        return "recovery", "+".join(key for key, flag in recovery.items() if flag)
    if any(value is not None for value in deltas.values()):
        return "no_material_change", "none"
    return "insufficient_reference", "none"


def _assign_crop_instances(frame: pd.DataFrame, policy: MonitorPolicy) -> pd.DataFrame:
    output = frame.copy()
    output["crop_instance_start_date"] = pd.NaT
    for _, indices in output.groupby("field_id", sort=True, dropna=False).groups.items():
        previous_date: pd.Timestamp | None = None
        previous_stage = "unknown"
        previous_regime: tuple[str, str] | None = None
        instance_start: pd.Timestamp | None = None
        for index in sorted(indices, key=lambda item: output.at[item, "observation_date"]):
            current_date = pd.Timestamp(output.at[index, "observation_date"])
            stage = str(output.at[index, "stage_family"])
            regime = (str(output.at[index, "crop_name"]), str(output.at[index, "crop_season"]))
            starts_after_gap = (
                previous_date is not None
                and (current_date - previous_date).days > policy.crop_instance_gap_days
            )
            starts_after_off_season = previous_stage == "off_season" and stage != "off_season"
            if (
                instance_start is None
                or regime != previous_regime
                or starts_after_gap
                or starts_after_off_season
            ):
                instance_start = current_date
            output.at[index, "crop_instance_start_date"] = instance_start
            previous_date = current_date
            previous_stage = stage
            previous_regime = regime
    output["crop_instance_id"] = output.apply(
        lambda row: stable_id(
            "crop",
            (
                row["field_id"],
                row["crop_name"],
                row["crop_season"],
                pd.Timestamp(row["crop_instance_start_date"]).date().isoformat(),
            ),
        ),
        axis=1,
    )
    return output


def prepare_causal_signals(frame: pd.DataFrame, policy: MonitorPolicy) -> pd.DataFrame:
    """Create acquisition-aware deltas using only observations available on each row."""
    missing = {
        "field_id",
        "observation_date",
        "crop_name",
        "crop_season",
        "crop_stage",
        "risk_level",
        "primary_risk_driver",
        "spectral_echo_days",
        *SPECTRAL_COLUMNS,
    }.difference(frame.columns)
    if missing:
        raise ValueError("Canonical input is missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        return _empty_causal_frame(frame)

    output = frame.copy()
    output["observation_date"] = pd.to_datetime(output["observation_date"]).dt.normalize()
    for column, fallback in (
        ("crop_name", "unknown_crop"),
        ("crop_season", "unknown_season"),
        ("crop_stage", "unknown_stage"),
    ):
        output[column] = output[column].fillna(fallback).astype(str).str.strip().replace("", fallback)
    raw_risk = output["risk_level"]
    output["pressure_observed"] = raw_risk.notna() & raw_risk.astype("string").str.strip().ne("")
    output["risk_band"] = [
        normalize_risk(value) if observed else "NONE"
        for value, observed in zip(raw_risk, output["pressure_observed"])
    ]
    output["risk_rank"] = output["risk_band"].map(RISK_RANKS).astype("int8")
    output["hazard_family"] = output["primary_risk_driver"].fillna("").map(normalize_hazard)
    output["stage_family"] = output["crop_stage"].map(normalize_stage)
    dedup_keys = ["field_id", "observation_date"]
    output = output.sort_values(
        dedup_keys + ["risk_rank", "crop_name", "crop_season", "hazard_family"],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    ).drop_duplicates(dedup_keys, keep="first")
    output = _assign_crop_instances(output.reset_index(drop=True), policy)
    output["spectral_echo_days"] = pd.to_numeric(output["spectral_echo_days"], errors="coerce")
    if (output["spectral_echo_days"].dropna() < 0).any():
        raise ValueError("spectral_echo_days cannot be negative.")
    output["spectral_source_date"] = output.apply(
        lambda row: pd.NaT
        if pd.isna(row["spectral_echo_days"])
        else row["observation_date"] - timedelta(days=int(row["spectral_echo_days"])),
        axis=1,
    )

    feature_rows: list[dict[str, Any]] = []
    for _, group in output.groupby("crop_instance_id", sort=True):
        acquisitions: list[dict[str, Any]] = []
        seen_source_dates: set[pd.Timestamp] = set()
        for row in group.sort_values("observation_date", kind="mergesort").to_dict("records"):
            source_date = row["spectral_source_date"]
            freshness = _freshness(row["spectral_echo_days"], policy)
            is_new = not pd.isna(source_date) and source_date not in seen_source_dates
            reference: dict[str, Any] | None = None
            if is_new:
                eligible = [
                    item
                    for item in acquisitions
                    if policy.reference_min_days
                    <= (source_date - item["source_date"]).days
                    <= policy.reference_max_days
                ]
                if eligible:
                    reference = max(eligible, key=lambda item: item["source_date"])

            deltas = {name: None for name in SPECTRAL_COLUMNS}
            if reference is not None:
                for name in SPECTRAL_COLUMNS:
                    current = _finite_or_none(row.get(name))
                    previous = reference[name]
                    if current is not None and previous is not None:
                        deltas[name] = current - previous

            if pd.isna(source_date):
                response_class, evidence = "spectral_missing", "none"
            elif not is_new:
                response_class, evidence = "no_new_acquisition", "none"
            elif freshness == "stale":
                response_class, evidence = "stale_acquisition", "none"
            elif reference is None:
                response_class, evidence = "insufficient_reference", "none"
            else:
                response_class, evidence = _response_class(deltas, policy)

            row.update(
                {
                    "spectral_freshness": freshness,
                    "is_new_acquisition": bool(is_new),
                    "reference_source_date": None if reference is None else reference["source_date"],
                    "ndvi_delta": deltas["ndvi"],
                    "ndmi_delta": deltas["ndmi"],
                    "psri_delta": deltas["psri"],
                    "response_class": response_class,
                    "response_evidence": evidence,
                    "new_response_evidence": bool(
                        is_new and freshness in {"fresh", "aging"} and reference is not None
                    ),
                }
            )
            feature_rows.append(row)
            if is_new:
                seen_source_dates.add(source_date)
                acquisitions.append(
                    {
                        "source_date": source_date,
                        **{name: _finite_or_none(row.get(name)) for name in SPECTRAL_COLUMNS},
                    }
                )

    result = pd.DataFrame(feature_rows)
    return result.sort_values(
        ["field_id", "crop_instance_id", "observation_date"], kind="mergesort"
    ).reset_index(drop=True)


def _empty_causal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns) + [
        "risk_band",
        "risk_rank",
        "pressure_observed",
        "hazard_family",
        "stage_family",
        "crop_instance_start_date",
        "crop_instance_id",
        "spectral_source_date",
        "spectral_freshness",
        "is_new_acquisition",
        "reference_source_date",
        "ndvi_delta",
        "ndmi_delta",
        "psri_delta",
        "response_class",
        "response_evidence",
        "new_response_evidence",
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))
