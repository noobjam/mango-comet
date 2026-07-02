from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd

from .causal_features import prepare_causal_signals
from .contracts import MonitorPolicy, SCHEMA_VERSION, iso_date, stable_id
from .state_machine import MachineResult, run_state_machine


DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "onset_policy_v1.json"
BOUNDED_V1_MAX_FIELDS = 5000

ALIASES = {
    "field_id": ("field_id", "Field_ID"),
    "observation_date": ("observation_date", "Observation_Date"),
    "crop_name": ("crop_name", "Crop_Name"),
    "crop_season": ("crop_season", "Crop_Season_Code"),
    "crop_stage": ("crop_stage", "Crop_Stage"),
    "risk_level": ("risk_level", "Crop_Health_Daily_RIsk"),
    "primary_risk_driver": ("primary_risk_driver", "Primary_Risk_Driver"),
    "spectral_echo_days": ("spectral_echo_days", "Spectral_Echo_Days"),
    "ndvi": ("ndvi", "NDVI_AVG"),
    "ndmi": ("ndmi", "NDMI_AVG"),
    "psri": ("psri", "PSRI_AVG"),
    "spi_index": ("spi_index", "SPI_Index_AVG"),
    "ponding_mm": ("ponding_mm", "Ponding_mm_AVG"),
    "temperature": ("temperature", "Temperature_C_AVG"),
    "apparent_temperature": ("apparent_temperature",),
    "humidity": ("humidity", "Humidity_Precent_AVG"),
    "wind_speed": ("wind_speed", "Wind_Speed_KM_HR_AVG"),
}
REQUIRED_COLUMNS = frozenset(
    {
        "field_id",
        "observation_date",
        "crop_name",
        "crop_season",
        "crop_stage",
        "risk_level",
        "primary_risk_driver",
        "spectral_echo_days",
        "ndvi",
        "ndmi",
        "psri",
    }
)
TEXT_COLUMNS = frozenset(
    {"field_id", "crop_name", "crop_season", "crop_stage", "risk_level", "primary_risk_driver"}
)

EVENT_COLUMNS = (
    "field_id", "crop_name", "crop_season", "crop_instance_id", "event_id",
    "event_start_date", "active_end_date", "event_end_date", "event_state",
    "max_risk_band", "max_risk_rank", "hazard_signature", "stage_signature",
    "response_signature", "close_reason", "reportable_days", "window_span_days",
    "story_cluster_id", "right_censored", "as_of_date", "requires_review",
)
MEMBERSHIP_COLUMNS = (
    "field_id", "event_id", "story_cluster_id", "crop_instance_id", "observation_date",
    "event_state", "hazard_signature", "daily_pressure_rank", "daily_response_class",
    "pressure_observed",
)
SNAPSHOT_COLUMNS = (
    "timeline_bucket", "snapshot_as_of_date", "field_id", "crop_name", "crop_season",
    "crop_instance_id", "event_id", "story_cluster_id", "event_state_id", "event_state",
    "hazard_signature", "max_risk_rank", "max_risk_band", "current_risk_rank",
    "current_risk_band", "reportable_day_count",
    "response_day_count", "right_censored", "is_data_gap_snapshot", "requires_review",
    "daily_pressure_rank", "daily_response_class", "revision", "generation_as_of_date",
)


@dataclass(frozen=True)
class GenerationResult:
    generation_id: str
    generation_dir: Path
    as_of_date: date
    row_count: int
    event_count: int


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _resolve_columns(actual_columns: list[str]) -> dict[str, str]:
    normalized = {_normalized_name(column): column for column in actual_columns}
    mapping: dict[str, str] = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            match = normalized.get(_normalized_name(candidate))
            if match is not None:
                mapping[canonical] = match
                break
    missing = sorted(REQUIRED_COLUMNS.difference(mapping))
    if missing:
        raise ValueError("Input parquet is missing required column(s): " + ", ".join(missing))
    return mapping


def _canonical_select(mapping: dict[str, str]) -> str:
    expressions: list[str] = []
    for canonical in ALIASES:
        actual = mapping.get(canonical)
        if actual is None:
            expression = "CAST(NULL AS DOUBLE)"
        elif canonical == "field_id":
            expression = f"TRIM(CAST({_quote_identifier(actual)} AS VARCHAR))"
        elif canonical in TEXT_COLUMNS:
            expression = f"CAST({_quote_identifier(actual)} AS VARCHAR)"
        elif canonical == "observation_date":
            expression = f"TRY_CAST({_quote_identifier(actual)} AS DATE)"
        elif canonical == "spectral_echo_days":
            expression = f"TRY_CAST({_quote_identifier(actual)} AS INTEGER)"
        else:
            expression = f"TRY_CAST({_quote_identifier(actual)} AS DOUBLE)"
        expressions.append(f'{expression} AS {_quote_identifier(canonical)}')
    return ",\n".join(expressions)


def read_canonical_input(
    input_parquet: Path,
    *,
    as_of_date: date,
    history_from: date | None = None,
    max_fields: int | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    input_parquet = input_parquet.expanduser().resolve()
    if not input_parquet.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_parquet}")
    if max_fields is not None and max_fields < 1:
        raise ValueError("max_fields must be a positive integer.")
    if max_fields is None:
        raise ValueError(
            "Bounded V1 requires --max-fields; refusing to load the full parquet into pandas. "
            "Use at most 5000 fields for smoke/replay validation. Full 39.7M-row operation still "
            "requires the planned partitioned transactional store."
        )
    if max_fields > BOUNDED_V1_MAX_FIELDS:
        raise ValueError(f"Bounded V1 max_fields cannot exceed {BOUNDED_V1_MAX_FIELDS}.")
    with duckdb.connect(":memory:") as connection:
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_parquet)]
        ).fetchall()
        mapping = _resolve_columns([str(row[0]) for row in description])
        source = _canonical_select(mapping)
        date_column = _quote_identifier(mapping["observation_date"])
        conditions = [f"TRY_CAST({date_column} AS DATE) <= ?"]
        parameters: list[Any] = [str(input_parquet), as_of_date.isoformat()]
        if history_from is not None:
            conditions.append(f"TRY_CAST({date_column} AS DATE) >= ?")
            parameters.append(history_from.isoformat())
        source_sql = f"SELECT {source} FROM read_parquet(?) WHERE {' AND '.join(conditions)}"
        if max_fields is None:
            query = f"SELECT * FROM ({source_sql}) AS source ORDER BY field_id, observation_date"
        else:
            query = f"""
                WITH source AS ({source_sql}),
                selected_fields AS (
                    SELECT field_id
                    FROM source
                    WHERE field_id IS NOT NULL
                    GROUP BY field_id
                    ORDER BY md5(field_id), field_id
                    LIMIT {int(max_fields)}
                )
                SELECT source.*
                FROM source
                JOIN selected_fields USING (field_id)
                ORDER BY field_id, observation_date
            """
        frame = connection.execute(query, parameters).fetchdf()
    if frame.empty:
        raise ValueError(f"Input contains no rows on or before {as_of_date.isoformat()}.")
    if frame["observation_date"].isna().any():
        raise ValueError("Input contains null or invalid observation_date values in the selected range.")
    frame["field_id"] = frame["field_id"].astype("string").str.strip()
    if frame["field_id"].isna().any() or frame["field_id"].eq("").any():
        raise ValueError("Input contains null or empty field_id values in the selected range.")
    if (frame["spectral_echo_days"].dropna() < 0).any():
        raise ValueError("spectral_echo_days cannot be negative.")
    return frame, mapping


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _ensure_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = None
    return output.loc[:, list(columns)]


def _weekly_snapshots(machine: MachineResult, as_of_date: date) -> pd.DataFrame:
    if machine.daily_records.empty:
        return _ensure_columns(pd.DataFrame(), SNAPSHOT_COLUMNS)
    daily = machine.daily_records.copy()
    observed = pd.to_datetime(daily["observation_date"])
    daily["timeline_bucket"] = (observed - pd.to_timedelta(observed.dt.weekday, unit="D")).dt.date
    daily["snapshot_as_of_date"] = observed.dt.date
    daily["generation_as_of_date"] = as_of_date
    daily = daily.sort_values(
        ["event_id", "timeline_bucket", "snapshot_as_of_date"], kind="mergesort"
    )
    snapshots = daily.groupby(["event_id", "timeline_bucket"], sort=True).tail(1)
    return _ensure_columns(snapshots.reset_index(drop=True), SNAPSHOT_COLUMNS)


def _map_frames(machine: MachineResult, snapshots: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timeline_bucket", "field_id", "story_cluster_id", "event_id", "event_state_id",
        "event_state", "crop_name", "crop_season", "crop_instance_id", "max_risk_band",
        "current_risk_band", "hazard_signature", "motif_family", "response_signature",
        "reportable_day_count", "event_count", "max_risk_rank", "current_risk_rank",
        "response_day_count", "right_censored",
        "requires_review",
    ]
    if snapshots.empty:
        return pd.DataFrame(columns=columns)
    daily = machine.daily_records.copy()
    observed = pd.to_datetime(daily["observation_date"])
    daily["timeline_bucket"] = (observed - pd.to_timedelta(observed.dt.weekday, unit="D")).dt.date
    daily["is_reportable"] = (
        daily["daily_pressure_rank"].ge(2)
        | daily["daily_response_class"].isin({"medium_decline", "severe_decline"})
    ).astype("int64")
    daily["has_response"] = daily["daily_response_class"].isin(
        {"medium_decline", "severe_decline", "recovery"}
    ).astype("int64")
    counts = daily.groupby(["event_id", "timeline_bucket"], as_index=False).agg(
        reportable_day_count=("is_reportable", "sum"),
        response_day_count=("has_response", "sum"),
    )
    frame = snapshots.drop(columns=["reportable_day_count", "response_day_count"]).merge(
        counts, on=["event_id", "timeline_bucket"], how="left", validate="one_to_one"
    )
    frame["motif_family"] = frame["hazard_signature"]
    frame["response_signature"] = frame["daily_response_class"]
    frame["event_count"] = 1
    frame["timeline_bucket"] = frame["timeline_bucket"].map(iso_date)
    return frame.loc[:, columns].sort_values(
        ["timeline_bucket", "field_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _event_labels(machine: MachineResult) -> pd.DataFrame:
    columns = [
        "story_cluster_id", "short_label", "max_risk_band", "hazard_signature",
        "motif_family", "response_signature", "event_count", "field_count", "crop_count",
        "median_window_span_days", "median_reportable_days",
    ]
    if machine.events.empty or machine.daily_records.empty:
        return pd.DataFrame(columns=columns)
    onset = machine.daily_records.sort_values("observation_date", kind="mergesort").groupby(
        "event_id", as_index=False
    ).first()
    labels = onset.copy()
    labels["story_cluster_id"] = labels["event_id"]
    labels["short_label"] = labels["hazard_signature"].map(
        lambda value: f"{str(value).replace('_', ' ').title()} monitored event"
    )
    labels["motif_family"] = labels["hazard_signature"]
    labels["response_signature"] = labels["daily_response_class"]
    labels["event_count"] = 1
    labels["field_count"] = 1
    labels["crop_count"] = 1
    # Static labels must not backfill complete-event duration into an earlier map frame.
    labels["median_window_span_days"] = 1.0
    labels["median_reportable_days"] = (
        labels["daily_pressure_rank"].ge(2)
        | labels["daily_response_class"].isin({"medium_decline", "severe_decline"})
    ).astype("float64")
    return labels.loc[:, columns].sort_values("story_cluster_id").reset_index(drop=True)


def _crop_instances(signals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "crop_instance_id", "field_id", "crop_name", "crop_season",
        "crop_instance_start_date", "last_observation_date", "observation_count",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    frame = signals.groupby(
        ["crop_instance_id", "field_id", "crop_name", "crop_season"], as_index=False
    ).agg(
        crop_instance_start_date=("crop_instance_start_date", "min"),
        last_observation_date=("observation_date", "max"),
        observation_count=("observation_date", "size"),
    )
    return frame.loc[:, columns].sort_values(["field_id", "crop_instance_start_date"])


def _write_parquet(frame: pd.DataFrame, path: Path, sort_by: list[str] | None = None) -> None:
    output = frame.copy()
    if sort_by and not output.empty:
        output = output.sort_values(sort_by, kind="mergesort")
    output.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")


def build_generation(
    *,
    input_parquet: Path,
    output_dir: Path,
    as_of_date: date,
    policy: MonitorPolicy,
    history_from: date | None = None,
    max_fields: int | None = None,
    geometry_parquet: Path | None = None,
) -> GenerationResult:
    input_parquet = input_parquet.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_metadata = _file_metadata(input_parquet)
    geometry_metadata = None
    if geometry_parquet is not None:
        geometry_parquet = geometry_parquet.expanduser().resolve()
        if not geometry_parquet.is_file():
            raise FileNotFoundError(f"Geometry parquet does not exist: {geometry_parquet}")
        geometry_metadata = _file_metadata(geometry_parquet)
    generation_id = stable_id(
        "generation",
        (
            as_of_date.isoformat(), history_from, max_fields, policy.source_sha256,
            source_metadata["name"], source_metadata["size_bytes"], source_metadata["mtime_ns"],
            None if geometry_metadata is None else geometry_metadata["mtime_ns"],
        ),
        length=20,
    )
    generations_dir = output_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    final_dir = generations_dir / f"{as_of_date.isoformat()}_{generation_id}"
    if final_dir.exists():
        raise FileExistsError(f"Immutable generation already exists: {final_dir}")

    raw, column_mapping = read_canonical_input(
        input_parquet,
        as_of_date=as_of_date,
        history_from=history_from,
        max_fields=max_fields,
    )
    if _file_metadata(input_parquet) != source_metadata:
        raise RuntimeError("Input parquet changed while the generation was being read; retry.")
    signals = prepare_causal_signals(raw, policy)
    machine = run_state_machine(signals, policy, as_of_date=as_of_date)
    events = _ensure_columns(machine.events, EVENT_COLUMNS)
    memberships = _ensure_columns(machine.memberships, MEMBERSHIP_COLUMNS)
    snapshots = _weekly_snapshots(machine, as_of_date)
    frames = _map_frames(machine, snapshots)
    labels = _event_labels(machine)
    instances = _crop_instances(signals)

    with TemporaryDirectory(prefix=".weekly-story-", dir=generations_dir) as temporary:
        stage = Path(temporary) / final_dir.name
        stage.mkdir()
        _write_parquet(signals, stage / "daily_causal_signals.parquet", ["field_id", "observation_date"])
        _write_parquet(instances, stage / "crop_instances.parquet", ["field_id", "crop_instance_start_date"])
        _write_parquet(events, stage / "event_windows.parquet", ["field_id", "event_start_date"])
        _write_parquet(snapshots, stage / "event_state_snapshots.parquet", ["timeline_bucket", "field_id"])
        _write_parquet(memberships, stage / "story_day_membership.parquet", ["field_id", "observation_date"])
        _write_parquet(frames, stage / "map_frame_fields.parquet", ["timeline_bucket", "field_id"])
        _write_parquet(labels, stage / "event_story_cluster_labels.parquet", ["story_cluster_id"])
        if geometry_parquet is not None:
            shutil.copy2(geometry_parquet, stage / "map_field_geometry.parquet")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "status": "complete",
                "generation_id": generation_id,
                "as_of_date": as_of_date.isoformat(),
                "immutable": True,
                "row_count": int(len(signals)),
                "field_count": int(signals["field_id"].nunique()),
                "crop_instance_count": int(signals["crop_instance_id"].nunique()),
                "event_count": int(len(events)),
                "snapshot_count": int(len(snapshots)),
                "viewer_ready": False,
                "viewer_bundle_required": geometry_parquet is not None and not frames.empty,
            },
            "input": {
                "source": source_metadata,
                "history_from": None if history_from is None else history_from.isoformat(),
                "max_fields": max_fields,
                "column_mapping": column_mapping,
                "requires_spectral_echo_days": True,
            },
            "policy": {
                "version": policy.version,
                "sha256": policy.source_sha256,
                "calibration_status": policy.calibration_status,
                "warning": "Starter numeric thresholds are uncalibrated and require agronomist validation.",
            },
            "semantics": {
                "prefix_safe": True,
                "full_history_z_scores": False,
                "spectral_carry_forward_is_new_evidence": False,
                "open_events_are_right_censored": True,
                "story_cluster_id_alias": "event_id",
                "persistent_event_registry": False,
                "event_id_stability": "ordinary future appends only; earlier corrections can move onset-derived IDs",
                "event_state_id_stability": "immutable within this generation; no cross-generation registry guarantee",
                "motif_family": "hazard-family compatibility facet; not a learned motif",
                "causal_prefix_feature_version": "causal_prefix_features_v1",
            },
            "limitations": [
                "This bounded V1 requires max_fields <= 5000 and recomputes that selected prefix in memory.",
                "The 39.7M-row full dataset is intentionally rejected until hash-partitioned streaming and a transactional store are implemented.",
                "Late corrections create a new immutable generation but no automatic supersession lineage.",
                "Event IDs are stable under ordinary future appends, but an earlier late-arriving onset can change identity.",
                "No learned motif model, calibrated uncertainty, causal geographic propagation, or outcome validation is included.",
            ],
            "outputs": {
                "daily_causal_signals": "daily_causal_signals.parquet",
                "crop_instances": "crop_instances.parquet",
                "event_windows": "event_windows.parquet",
                "event_state_snapshots": "event_state_snapshots.parquet",
                "story_day_membership": "story_day_membership.parquet",
                "map_frame_fields": "map_frame_fields.parquet",
                "event_story_cluster_labels": "event_story_cluster_labels.parquet",
                "map_field_geometry": "map_field_geometry.parquet" if geometry_parquet else None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, final_dir)
    return GenerationResult(generation_id, final_dir, as_of_date, len(signals), len(events))
