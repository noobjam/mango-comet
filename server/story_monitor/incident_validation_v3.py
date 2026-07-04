"""Fail-closed validation for immutable crop-impact incident artifacts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd


SOURCE_SCHEMA_VERSION = "crop-incident-source-v3/1"
REQUIRED_SOURCE_ARTIFACTS = (
    "daily_causal_signals.parquet",
    "event_state_snapshots.parquet",
    "event_windows.parquet",
    "story_day_membership.parquet",
    "map_field_geometry.parquet",
)

FINAL_ARTIFACT_KEYS: dict[str, tuple[str, ...]] = {
    "field_week_context": ("timeline_bucket", "field_id", "crop_instance_id"),
    "stage_baseline": ("hazard_family", "stage_bucket", "iso_week"),
    "weekly_exposure_cells": ("timeline_bucket", "hazard_family", "cell_id"),
    "weekly_components": ("component_id",),
    "component_membership": ("component_id", "crop_instance_id", "episode_id"),
    "exposure_weekly_state": ("exposure_id", "timeline_bucket"),
    "incident_weekly_state": ("incident_id", "timeline_bucket"),
    "incident_stage_summary": (
        "incident_id",
        "timeline_bucket",
        "stage_bucket",
    ),
    "incident_membership": (
        "incident_id",
        "timeline_bucket",
        "crop_instance_id",
        "membership_role",
    ),
    "incident_windows": ("incident_id",),
}

FINAL_ARTIFACT_FILES = {
    **{name: f"{name}.parquet" for name in FINAL_ARTIFACT_KEYS},
    "incident_lineage": "incident_lineage.parquet",
    "completed_incident_features": "completed_incident_features.parquet",
}

APPEND_STABILITY_FILES = (
    "weekly_components.parquet",
    "exposure_component_assignments.parquet",
    "incident_weekly_state.parquet",
    "incident_stage_summary.parquet",
    "incident_membership.parquet",
    "incident_lineage.parquet",
    "incident_windows.parquet",
)

STAGE_SUMMARY_COUNT_COLUMNS = (
    "monitored_field_count",
    "evaluable_field_count",
    "monitored_crop_instance_count",
    "evaluable_crop_instance_count",
    "pressure_core_crop_instance_count",
    "severe_crop_instance_count",
    "watch_frontier_crop_instance_count",
    "impact_lag_crop_instance_count",
    "affected_crop_instance_count",
    "footprint_cell_count",
    "crop_observed_cell_count",
    "coverage_missing_cell_count",
    "global_crop_week_unmappable_instance_count",
)

STAGE_SUMMARY_RATE_COLUMNS = (
    "pressure_signal_rate",
    "impact_signal_rate",
)

STAGE_SUMMARY_REQUIRED_COLUMNS = (
    "timeline_bucket",
    "incident_id",
    "exposure_id",
    "crop_name",
    "hazard_family",
    "stage_bucket",
    *STAGE_SUMMARY_COUNT_COLUMNS,
    *STAGE_SUMMARY_RATE_COLUMNS,
    "denominator_scope",
    "schema_version",
    "policy_version",
    "policy_sha256",
)

# These weekly values are causal evidence at that week and must never rewrite.
APPEND_STABLE_WEEKLY_CONTENT_COLUMNS = (
    "base_incident_id",
    "segment_index",
    "exposure_id",
    "crop_name",
    "hazard_family",
    "component_id",
    "knowledge_time",
    "knowledge_time_inferred",
    "pressure_core_field_count",
    "severe_field_count",
    "watch_frontier_field_count",
    "impact_lag_field_count",
    "entering_field_count",
    "persisting_field_count",
    "exiting_field_count",
    "field_overlap_jaccard",
    "stage_distribution",
    "pressure_cell_ids_json",
    "impact_cell_ids_json",
    "watch_cell_ids_json",
    "footprint_cell_ids_json",
    "footprint_carried_forward",
    "cell_coverage_adequate",
    "footprint_area_km2",
    "hazard_intensity",
    "is_physical_movement",
    "unresolved_carried_field_count",
    "recovered_field_count",
    "fresh_decline_field_count",
    "fresh_recovery_field_count",
    "coverage_monitored_field_count",
    "coverage_evaluable_field_count",
    "coverage_monitored_crop_instance_count",
    "coverage_evaluable_crop_instance_count",
    "coverage_adequate",
    "coverage_missing_cell_count",
    "season_boundary_observed",
    "split_count",
    "merge_count",
)

APPEND_STABLE_LIFECYCLE_COLUMNS = (
    "incident_state",
    "current_state",
    "first_evidence_week",
    "confirmed_week",
    "pressure_off_week",
    "recovered_week",
    "closed_week",
    "merged_into_incident_id",
    "right_censored",
    "relapse_count",
    "data_gap_count",
    "coverage_gap_streak",
    "data_censored_at_boundary",
)

# A max-week data-censor marker is a provisional closure caused solely by the
# release boundary.  On a later append only these presentation/closure fields
# may resolve on that already-published row.  Evidence dates and cumulative
# history remain immutable.
BOUNDARY_MUTABLE_LIFECYCLE_COLUMNS = (
    "incident_state",
    "current_state",
    "closed_week",
    "right_censored",
    "data_censored_at_boundary",
)

BOUNDARY_FROZEN_LIFECYCLE_COLUMNS = tuple(
    column
    for column in APPEND_STABLE_LIFECYCLE_COLUMNS
    if column not in BOUNDARY_MUTABLE_LIFECYCLE_COLUMNS
)

WINDOW_REQUIRED_COLUMNS = (
    "incident_id",
    "exposure_id",
    "crop_name",
    "hazard_family",
    "first_evidence_week",
    "confirmed_week",
    "pressure_off_week",
    "recovered_week",
    "closed_week",
    "merged_into_incident_id",
    "terminal_state",
    "right_censored",
    "observed_week_count",
    "active_component_week_count",
    "peak_week",
    "peak_affected_field_count",
    "relapse_count",
    "data_gap_count",
    "split_count",
    "merge_count",
    "outcome_evidence",
)

WINDOW_ALWAYS_STABLE_COLUMNS = (
    "exposure_id",
    "crop_name",
    "hazard_family",
    "first_evidence_week",
    "outcome_evidence",
)

WINDOW_MONOTONIC_COUNT_COLUMNS = (
    "observed_week_count",
    "active_component_week_count",
    "peak_affected_field_count",
    "relapse_count",
    "data_gap_count",
    "split_count",
    "merge_count",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_generation(generation_dir: Path) -> dict[str, Any]:
    """Validate the immutable V1 source required by the V3 offline build."""
    root = generation_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read a valid generation manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Generation manifest must be a JSON object")
    run = manifest.get("run") or {}
    if str(run.get("status") or "") != "complete":
        raise ValueError("V3 requires a completed immutable source generation")
    if not bool(run.get("immutable", True)):
        raise ValueError("V3 refuses a source generation not marked immutable")
    policy = manifest.get("policy") or {}
    if not str(policy.get("version") or "") or not str(policy.get("sha256") or ""):
        raise ValueError("Source generation is missing policy version or SHA-256")
    missing = [name for name in REQUIRED_SOURCE_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise ValueError("Source generation is missing: " + ", ".join(missing))
    return manifest


def artifact_hashes(directory: Path, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in sorted(set(names)):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Cannot hash missing artifact: {path}")
        output[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return output


def validate_final_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate the in-memory final tables before immutable publication."""
    missing = sorted(set(FINAL_ARTIFACT_KEYS) - set(frames))
    if missing:
        raise ValueError("V3 output is missing frames: " + ", ".join(missing))
    counts: dict[str, int] = {}
    for name, key in FINAL_ARTIFACT_KEYS.items():
        frame = frames[name]
        _require_columns(frame, key, name)
        _require_nonblank(frame, key, name)
        if frame.duplicated(list(key)).any():
            raise ValueError(f"{name} contains duplicate natural keys: {key}")
        _reject_death_claims(frame, name)
        counts[name] = len(frame)

    _validate_context_causality(frames["field_week_context"])
    _validate_incident_knowledge_time(frames["incident_weekly_state"])
    _validate_stage_summary_frame(frames["incident_stage_summary"])
    _validate_story_references(frames)
    lineage = frames.get("incident_lineage")
    if lineage is not None:
        _reject_death_claims(lineage, "incident_lineage")
        _validate_lineage(lineage)
        counts["incident_lineage"] = len(lineage)
    return {"passed": True, "row_counts": counts}


def validate_final_artifact_directory(directory: Path) -> dict[str, Any]:
    """Reconcile a staged V3 directory without loading large Parquet files."""
    root = directory.expanduser().resolve()
    missing_files = [name for name in FINAL_ARTIFACT_FILES.values() if not (root / name).is_file()]
    if missing_files:
        raise ValueError("V3 output is missing artifacts: " + ", ".join(missing_files))
    counts: dict[str, int] = {}
    connection = duckdb.connect(":memory:")
    try:
        for name, key in FINAL_ARTIFACT_KEYS.items():
            path = root / FINAL_ARTIFACT_FILES[name]
            relation = connection.read_parquet(str(path))
            columns = set(relation.columns)
            missing = sorted(set(key) - columns)
            if missing:
                raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
            relation.create_view("artifact", replace=True)
            count = int(connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0])
            distinct = int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT "
                    + ", ".join(_quote_identifier(column) for column in key)
                    + " FROM artifact GROUP BY "
                    + ", ".join(_quote_identifier(column) for column in key)
                    + ")"
                ).fetchone()[0]
            )
            null_predicate = " OR ".join(
                f"{_quote_identifier(column)} IS NULL OR TRIM(CAST({_quote_identifier(column)} AS VARCHAR)) = ''"
                for column in key
            )
            nulls = int(
                connection.execute(f"SELECT COUNT(*) FROM artifact WHERE {null_predicate}").fetchone()[0]
            )
            if count != distinct or nulls:
                raise ValueError(
                    f"{name} natural-key reconciliation failed: rows={count}, "
                    f"distinct={distinct}, null_keys={nulls}"
                )
            counts[name] = count

        context_path = root / FINAL_ARTIFACT_FILES["field_week_context"]
        future_stage = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM read_parquet(?)
                WHERE CAST(stage_source_date AS DATE) > CAST(week_last_observation_date AS DATE)
                """,
                [str(context_path)],
            ).fetchone()[0]
        )
        if future_stage:
            raise ValueError("field_week_context contains future stage evidence")
        _validate_directory_references(connection, root)
        _validate_directory_stage_summary(connection, root)
        _validate_directory_windows(connection, root)
        incident_path = root / FINAL_ARTIFACT_FILES["incident_weekly_state"]
        incident_columns = set(connection.read_parquet(str(incident_path)).columns)
        if "knowledge_time" in incident_columns:
            invalid_knowledge = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM read_parquet(?)
                    WHERE CAST(knowledge_time AS TIMESTAMP)
                        < CAST(timeline_bucket AS TIMESTAMP)
                    """,
                    [str(incident_path)],
                ).fetchone()[0]
            )
            if invalid_knowledge:
                raise ValueError("incident_weekly_state contains pre-evidence knowledge time")

        for file_name, state_column in (
            ("incident_weekly_state.parquet", "incident_state"),
            ("incident_windows.parquet", "terminal_state"),
        ):
            death = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM read_parquet(?)
                    WHERE REGEXP_MATCHES(UPPER(CAST({_quote_identifier(state_column)} AS VARCHAR)),
                        '(^|_)DEAD($|_)')
                    """,
                    [str(root / file_name)],
                ).fetchone()[0]
            )
            if death:
                raise ValueError(f"{file_name} contains an unsupported crop-death claim")

        lineage = pd.read_parquet(root / "incident_lineage.parquet")
        _validate_lineage(lineage)
        counts["incident_lineage"] = len(lineage)
        features = pd.read_parquet(root / "completed_incident_features.parquet")
        if "incident_id" not in features or features["incident_id"].astype(str).duplicated().any():
            raise ValueError("completed incident features must be unique by incident_id")
        counts["completed_incident_features"] = len(features)
    finally:
        connection.close()
    return {"passed": True, "row_counts": counts}


def assert_append_stability(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    natural_key: tuple[str, ...],
    identity_column: str,
) -> None:
    """Fail when an ordinary append rewrites an already-published identity."""
    required = (*natural_key, identity_column)
    _require_columns(previous, required, "previous")
    _require_columns(current, required, "current")
    left = previous.loc[:, required]
    right = current.loc[:, required]
    if left.duplicated(list(natural_key)).any():
        raise ValueError(f"previous contains duplicate append keys: {natural_key}")
    if right.duplicated(list(natural_key)).any():
        raise ValueError(f"current contains duplicate append keys: {natural_key}")
    missing = left.merge(
        right.loc[:, list(natural_key)],
        on=list(natural_key),
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] == "left_only"]
    if not missing.empty:
        raise ValueError(
            f"Ordinary append dropped {len(missing)} historical natural keys"
        )
    joined = left.merge(right, on=list(natural_key), how="inner", suffixes=("_old", "_new"))
    changed = joined[
        joined[f"{identity_column}_old"].astype(str)
        != joined[f"{identity_column}_new"].astype(str)
    ]
    if not changed.empty:
        raise ValueError(
            f"Ordinary append rewrote {len(changed)} historical {identity_column} values"
        )


def validate_append_stability(
    previous_directory: Path,
    current_directory: Path,
) -> dict[str, Any]:
    """Verify that an appended V3 release preserves every historical identity."""
    previous_root = previous_directory.expanduser().resolve()
    current_root = current_directory.expanduser().resolve()
    if previous_root == current_root:
        raise ValueError("Previous and current Incident V3 directories must differ")
    previous_manifest = _validate_previous_append_source(previous_root)
    missing_current = [
        name for name in APPEND_STABILITY_FILES
        if not (current_root / name).is_file()
    ]
    if missing_current:
        raise ValueError(
            "Current Incident V3 stage is missing append artifacts: "
            + ", ".join(missing_current)
        )
    before = artifact_hashes(previous_root, APPEND_STABILITY_FILES)

    previous_components = pd.read_parquet(
        previous_root / "weekly_components.parquet",
        columns=["timeline_bucket", "hazard_family", "cell_ids_json", "component_id"],
    )
    current_components = pd.read_parquet(
        current_root / "weekly_components.parquet",
        columns=["timeline_bucket", "hazard_family", "cell_ids_json", "component_id"],
    )
    for frame in (previous_components, current_components):
        frame["timeline_bucket"] = _canonical_week(frame["timeline_bucket"])
        frame["hazard_family"] = frame["hazard_family"].astype(str)
        frame["cell_set"] = frame["cell_ids_json"].map(_canonical_json_set)
        frame["component_id"] = frame["component_id"].astype(str)
    component_key = ("timeline_bucket", "hazard_family", "cell_set")
    assert_append_stability(
        previous_components,
        current_components,
        natural_key=component_key,
        identity_column="component_id",
    )
    _assert_no_historical_insertions(
        previous_components, current_components, component_key
    )

    assignment_columns = [
        "timeline_bucket", "component_id", "exposure_id"
    ]
    previous_assignments = pd.read_parquet(
        previous_root / "exposure_component_assignments.parquet",
        columns=assignment_columns,
    )
    current_assignments = pd.read_parquet(
        current_root / "exposure_component_assignments.parquet",
        columns=assignment_columns,
    )
    for frame in (previous_assignments, current_assignments):
        frame["timeline_bucket"] = _canonical_week(frame["timeline_bucket"])
        frame["component_id"] = frame["component_id"].astype(str)
        frame["exposure_id"] = frame["exposure_id"].astype(str)
    exposure_key = ("timeline_bucket", "component_id")
    assert_append_stability(
        previous_assignments,
        current_assignments,
        natural_key=exposure_key,
        identity_column="exposure_id",
    )
    _assert_no_historical_insertions(
        previous_assignments, current_assignments, exposure_key
    )

    incident_required = {
        "timeline_bucket", "exposure_id", "crop_name", "incident_id",
        "hazard_family", "footprint_cell_ids_json",
    }
    previous_weekly_path = previous_root / "incident_weekly_state.parquet"
    current_weekly_path = current_root / "incident_weekly_state.parquet"
    previous_columns = _parquet_column_names(previous_weekly_path)
    current_columns = _parquet_column_names(current_weekly_path)
    missing_previous = sorted(incident_required - previous_columns)
    missing_current = sorted(incident_required - current_columns)
    if missing_previous:
        raise ValueError(
            "previous incident weekly state is missing columns: "
            + ", ".join(missing_previous)
        )
    if missing_current:
        raise ValueError(
            "current incident weekly state is missing columns: "
            + ", ".join(missing_current)
        )
    stable_columns = set(APPEND_STABLE_WEEKLY_CONTENT_COLUMNS) | set(
        APPEND_STABLE_LIFECYCLE_COLUMNS
    )
    missing_previous_stable = sorted(stable_columns - previous_columns)
    missing_current_stable = sorted(stable_columns - current_columns)
    if missing_previous_stable:
        raise ValueError(
            "previous incident weekly state is missing append-stable columns: "
            + ", ".join(missing_previous_stable)
        )
    if missing_current_stable:
        raise ValueError(
            "current incident weekly state is missing append-stable columns: "
            + ", ".join(missing_current_stable)
        )
    content_columns = list(APPEND_STABLE_WEEKLY_CONTENT_COLUMNS)
    lifecycle_columns = list(APPEND_STABLE_LIFECYCLE_COLUMNS)
    selected_columns = list(
        dict.fromkeys(
            [
                "timeline_bucket", "incident_id", "exposure_id", "crop_name",
                *content_columns, *lifecycle_columns,
            ]
        )
    )
    previous_weekly = pd.read_parquet(
        previous_weekly_path, columns=selected_columns
    )
    current_weekly = pd.read_parquet(
        current_weekly_path, columns=selected_columns
    )
    for frame in (previous_weekly, current_weekly):
        frame["timeline_bucket"] = _canonical_week(frame["timeline_bucket"])
        for column in ("exposure_id", "crop_name", "incident_id"):
            frame[column] = frame[column].astype(str)
    incident_key = ("timeline_bucket", "exposure_id", "crop_name")
    assert_append_stability(
        previous_weekly,
        current_weekly,
        natural_key=incident_key,
        identity_column="incident_id",
    )
    _assert_no_historical_insertions(
        previous_weekly, current_weekly, incident_key
    )

    for required in ("exposure_id", "crop_name", "hazard_family", "footprint_cell_ids_json"):
        if required not in content_columns:
            raise ValueError(
                f"Incident weekly append content is missing stable column: {required}"
            )
    previous_content = _weekly_content_hashes(previous_weekly, content_columns)
    current_content = _weekly_content_hashes(current_weekly, content_columns)
    weekly_key = ("timeline_bucket", "incident_id")
    assert_append_stability(
        previous_content,
        current_content,
        natural_key=weekly_key,
        identity_column="content_sha256",
    )

    previous_weeks = pd.to_datetime(
        previous_weekly["timeline_bucket"], errors="raise"
    ).dt.normalize()
    previous_max_week = previous_weeks.max()
    boundary_exception = (
        previous_weekly["data_censored_at_boundary"].fillna(False).astype(bool)
        & previous_weeks.eq(previous_max_week)
    )
    previous_lifecycle = _weekly_content_hashes(
        previous_weekly.loc[~boundary_exception].copy(), lifecycle_columns
    )
    current_lifecycle = _weekly_content_hashes(
        current_weekly, lifecycle_columns
    )
    assert_append_stability(
        previous_lifecycle,
        current_lifecycle,
        natural_key=weekly_key,
        identity_column="content_sha256",
    )
    previous_boundary_lifecycle = _weekly_content_hashes(
        previous_weekly.loc[boundary_exception].copy(),
        list(BOUNDARY_FROZEN_LIFECYCLE_COLUMNS),
    )
    current_boundary_lifecycle = _weekly_content_hashes(
        current_weekly, list(BOUNDARY_FROZEN_LIFECYCLE_COLUMNS)
    )
    assert_append_stability(
        previous_boundary_lifecycle,
        current_boundary_lifecycle,
        natural_key=weekly_key,
        identity_column="content_sha256",
    )
    _validate_boundary_censor_resolution(
        previous_weekly.loc[boundary_exception].copy(),
        current_weekly,
        previous_weekly=previous_weekly,
        previous_max_week=previous_max_week,
    )

    stage_key = ("incident_id", "timeline_bucket", "stage_bucket")
    previous_stage, current_stage = _append_artifact_content_hashes(
        previous_root / "incident_stage_summary.parquet",
        current_root / "incident_stage_summary.parquet",
        natural_key=stage_key,
        label="incident stage summary",
    )
    assert_append_stability(
        previous_stage,
        current_stage,
        natural_key=stage_key,
        identity_column="content_sha256",
    )
    _assert_no_historical_insertions(
        previous_stage,
        current_stage,
        stage_key,
        historical_through=previous_max_week,
    )

    membership_key = (
        "timeline_bucket",
        "incident_id",
        "field_id",
        "crop_instance_id",
        "episode_id",
        "membership_role",
    )
    previous_membership, current_membership = _append_artifact_content_hashes(
        previous_root / "incident_membership.parquet",
        current_root / "incident_membership.parquet",
        natural_key=membership_key,
        label="incident membership",
    )
    assert_append_stability(
        previous_membership,
        current_membership,
        natural_key=membership_key,
        identity_column="content_sha256",
    )
    _assert_no_historical_insertions(
        previous_membership, current_membership, membership_key
    )

    lineage_key = (
        "timeline_bucket",
        "lineage_type",
        "crop_name_normalized",
        "parent_incident_id",
        "child_incident_id",
        "previous_component_id",
        "current_component_id",
    )
    previous_lineage, current_lineage = _append_artifact_content_hashes(
        previous_root / "incident_lineage.parquet",
        current_root / "incident_lineage.parquet",
        natural_key=lineage_key,
        label="incident lineage",
    )
    assert_append_stability(
        previous_lineage,
        current_lineage,
        natural_key=lineage_key,
        identity_column="content_sha256",
    )
    _assert_no_historical_insertions(
        previous_lineage,
        current_lineage,
        lineage_key,
        historical_through=previous_max_week,
    )

    window_key = ("incident_id",)
    previous_windows_raw = pd.read_parquet(
        previous_root / "incident_windows.parquet"
    )
    current_windows_raw = pd.read_parquet(
        current_root / "incident_windows.parquet"
    )
    _require_columns(
        previous_windows_raw, WINDOW_REQUIRED_COLUMNS,
        "previous incident windows",
    )
    _require_columns(
        current_windows_raw, WINDOW_REQUIRED_COLUMNS,
        "current incident windows",
    )
    _validate_windows_against_weekly(
        previous_windows_raw, previous_weekly, label="previous incident windows"
    )
    _validate_windows_against_weekly(
        current_windows_raw, current_weekly, label="current incident windows"
    )
    previous_window_ids = previous_windows_raw[["incident_id"]].assign(
        stable_incident_id=lambda frame: frame["incident_id"].astype(str)
    )
    current_window_ids = current_windows_raw[["incident_id"]].assign(
        stable_incident_id=lambda frame: frame["incident_id"].astype(str)
    )
    assert_append_stability(
        previous_window_ids,
        current_window_ids,
        natural_key=window_key,
        identity_column="stable_incident_id",
    )
    previous_windows, current_windows = _append_artifact_content_hashes(
        previous_root / "incident_windows.parquet",
        current_root / "incident_windows.parquet",
        natural_key=window_key,
        label="incident windows",
    )
    boundary_incident_ids = set(
        previous_weekly.loc[boundary_exception, "incident_id"].astype(str)
    )
    boundary_window_ids = set(
        previous_windows_raw.loc[
            previous_windows_raw["incident_id"].astype(str).isin(
                boundary_incident_ids
            )
            & previous_windows_raw["terminal_state"].astype(str).str.upper().eq(
                "CLOSED_DATA_CENSORED"
            )
            & ~previous_windows_raw["right_censored"].fillna(False).astype(bool),
            "incident_id",
        ].astype(str)
    )
    if boundary_window_ids != boundary_incident_ids:
        raise ValueError(
            "Prior max-week boundary-censored rows and windows do not reconcile"
        )
    terminal_ids = set(
        previous_windows_raw.loc[
            ~previous_windows_raw["right_censored"].fillna(False).astype(bool),
            "incident_id",
        ].astype(str)
    ) - boundary_window_ids
    strict_previous_windows = previous_windows[
        previous_windows["incident_id"].astype(str).isin(terminal_ids)
    ]
    assert_append_stability(
        strict_previous_windows,
        current_windows,
        natural_key=window_key,
        identity_column="content_sha256",
    )
    extendable_ids = set(
        previous_windows_raw.loc[
            previous_windows_raw["right_censored"].fillna(False).astype(bool),
            "incident_id",
        ].astype(str)
    ) | boundary_window_ids
    _validate_extendable_windows(
        previous_windows_raw,
        current_windows_raw,
        current_weekly,
        incident_ids=extendable_ids,
        previous_max_week=previous_max_week,
    )
    prior_incident_ids = set(previous_windows_raw["incident_id"].astype(str))
    new_windows = current_windows_raw[
        ~current_windows_raw["incident_id"].astype(str).isin(prior_incident_ids)
    ]
    historical_new_windows = new_windows[
        pd.to_datetime(new_windows["first_evidence_week"], errors="raise").dt.normalize()
        <= previous_max_week
    ]
    if not historical_new_windows.empty:
        raise ValueError(
            "Ordinary append inserted new incident windows into already-published weeks"
        )

    after = artifact_hashes(previous_root, APPEND_STABILITY_FILES)
    if before != after:
        raise RuntimeError(
            "Previous Incident V3 release changed during append-stability verification"
        )
    previous_run = previous_manifest.get("run") or {}
    return {
        "status": "passed",
        "previous_generation_id": previous_run.get("generation_id"),
        "comparisons": {
            "component_id": _append_counts(
                previous_components, current_components, component_key
            ),
            "exposure_id": _append_counts(
                previous_assignments, current_assignments, exposure_key
            ),
            "incident_id": _append_counts(
                previous_weekly, current_weekly, incident_key
            ),
            "historical_weekly_content": {
                **_append_counts(previous_content, current_content, weekly_key),
                "columns": content_columns,
            },
            "historical_lifecycle_content": {
                **_append_counts(
                    previous_lifecycle, current_lifecycle, weekly_key
                ),
                "columns": lifecycle_columns,
                "prior_boundary_resolutions": int(boundary_exception.sum()),
                "boundary_mutable_columns": list(
                    BOUNDARY_MUTABLE_LIFECYCLE_COLUMNS
                ),
                "boundary_frozen_columns": list(
                    BOUNDARY_FROZEN_LIFECYCLE_COLUMNS
                ),
            },
            "historical_stage_denominators": _append_counts(
                previous_stage, current_stage, stage_key
            ),
            "historical_incident_membership": _append_counts(
                previous_membership, current_membership, membership_key
            ),
            "historical_incident_lineage": _append_counts(
                previous_lineage, current_lineage, lineage_key
            ),
            "historical_terminal_windows": {
                **_append_counts(
                    strict_previous_windows, current_windows, window_key
                ),
                "prior_right_censored_allowed_to_extend": int(
                    previous_windows_raw["right_censored"].fillna(False).astype(bool).sum()
                ),
                "prior_boundary_censored_allowed_to_reopen": len(
                    boundary_window_ids
                ),
            },
        },
    }


def _validate_previous_append_source(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Previous Incident V3 directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Previous Incident V3 manifest is invalid") from exc
    run = manifest.get("run") or {}
    if str(run.get("status") or "") != "complete" or run.get("immutable") is not True:
        raise ValueError("Previous Incident V3 release must be complete and immutable")
    artifacts = manifest.get("artifacts") or {}
    for name in APPEND_STABILITY_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"Previous Incident V3 release is missing: {name}")
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Previous Incident V3 manifest does not declare: {name}")
        expected_size = metadata.get("size_bytes")
        expected_hash = metadata.get("sha256")
        if expected_size is None or not expected_hash:
            raise ValueError(f"Previous Incident V3 manifest has incomplete hash metadata: {name}")
        if int(expected_size) != path.stat().st_size or str(expected_hash) != file_sha256(path):
            raise ValueError(f"Previous Incident V3 artifact hash mismatch: {name}")
    return manifest


def _canonical_week(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise").dt.normalize()
    return parsed.dt.strftime("%Y-%m-%d")


def _parquet_column_names(path: Path) -> set[str]:
    connection = duckdb.connect(":memory:")
    try:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
        )
        return {str(item[0]) for item in (cursor.description or [])}
    finally:
        connection.close()


def _canonical_json_set(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Append-stability cell set must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("Append-stability cell set must be a JSON list")
    return json.dumps(
        sorted({str(item) for item in parsed}), separators=(",", ":")
    )


def _weekly_content_hashes(
    frame: pd.DataFrame, content_columns: list[str]
) -> pd.DataFrame:
    output = frame.loc[:, ["timeline_bucket", "incident_id", *content_columns]].copy()
    json_set_columns = {
        "pressure_cell_ids_json", "impact_cell_ids_json", "watch_cell_ids_json",
        "footprint_cell_ids_json",
    }
    output["content_sha256"] = [
        hashlib.sha256(
            json.dumps(
                {
                    column: (
                        _canonical_json_set(row[column])
                        if column in json_set_columns
                        else _canonical_scalar(row[column])
                    )
                    for column in content_columns
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        for row in output.to_dict("records")
    ]
    return output.loc[:, ["timeline_bucket", "incident_id", "content_sha256"]]


def _append_artifact_content_hashes(
    previous_path: Path,
    current_path: Path,
    *,
    natural_key: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalize and hash complete rows for an append-protected artifact."""
    previous = pd.read_parquet(previous_path)
    current = pd.read_parquet(current_path)
    previous_columns = set(previous.columns)
    current_columns = set(current.columns)
    if previous_columns != current_columns:
        missing_previous = sorted(current_columns - previous_columns)
        missing_current = sorted(previous_columns - current_columns)
        details = []
        if missing_previous:
            details.append("missing from previous: " + ", ".join(missing_previous))
        if missing_current:
            details.append("missing from current: " + ", ".join(missing_current))
        raise ValueError(f"{label} append schema differs ({'; '.join(details)})")
    _require_columns(previous, natural_key, f"previous {label}")
    _require_columns(current, natural_key, f"current {label}")
    content_columns = sorted(previous_columns - set(natural_key))

    def hashed(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for column in natural_key:
            if column == "timeline_bucket":
                output[column] = _canonical_week(output[column])
            else:
                output[column] = output[column].astype(str)
        output["content_sha256"] = [
            hashlib.sha256(
                json.dumps(
                    {
                        column: _canonical_scalar(row[column])
                        for column in content_columns
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            for row in output.to_dict("records")
        ]
        return output.loc[:, [*natural_key, "content_sha256"]]

    return hashed(previous), hashed(current)


def _validate_windows_against_weekly(
    windows: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    label: str,
) -> None:
    """Recompute every window field from its causal weekly ledger."""
    _require_columns(windows, WINDOW_REQUIRED_COLUMNS, label)
    weekly_required = (
        "incident_id", "timeline_bucket", "exposure_id", "crop_name",
        "hazard_family", "incident_state", "right_censored",
        "first_evidence_week", "confirmed_week", "pressure_off_week",
        "recovered_week", "closed_week", "merged_into_incident_id",
        "pressure_core_field_count", "unresolved_carried_field_count",
        "relapse_count", "data_gap_count", "split_count", "merge_count",
    )
    _require_columns(weekly, weekly_required, f"{label} weekly ledger")
    if windows["incident_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate incident IDs")
    weekly_source = weekly.copy()
    weekly_source["incident_id"] = weekly_source["incident_id"].astype(str)
    weekly_source["timeline_bucket"] = pd.to_datetime(
        weekly_source["timeline_bucket"], errors="raise"
    ).dt.normalize()
    if weekly_source.duplicated(["incident_id", "timeline_bucket"]).any():
        raise ValueError(f"{label} weekly ledger contains duplicate incident weeks")
    window_ids = set(windows["incident_id"].astype(str))
    weekly_ids = set(weekly_source["incident_id"])
    if window_ids != weekly_ids:
        raise ValueError(f"{label} and weekly ledger incident IDs do not reconcile")

    by_incident = {
        incident_id: group.sort_values("timeline_bucket", kind="mergesort")
        for incident_id, group in weekly_source.groupby("incident_id", sort=False)
    }
    for window in windows.to_dict("records"):
        incident_id = str(window["incident_id"])
        rows = by_incident[incident_id]
        right_censored = bool(window["right_censored"])
        terminal_state = str(window["terminal_state"] or "").upper()
        closed_week = _normalize_optional_week(window["closed_week"])
        recovered_week = _normalize_optional_week(window["recovered_week"])
        if right_censored and closed_week is not None:
            raise ValueError(f"{label} closes a right-censored incident: {incident_id}")
        if not right_censored and closed_week is None:
            raise ValueError(f"{label} terminal incident has no closed_week: {incident_id}")
        if terminal_state == "CLOSED_RECOVERED":
            if recovered_week is None or recovered_week != closed_week:
                raise ValueError(
                    f"{label} recovered incident does not reconcile recovery and closure"
                )
        elif recovered_week is not None:
            raise ValueError(
                f"{label} has recovered_week without CLOSED_RECOVERED state"
            )
        for column in ("exposure_id", "crop_name", "hazard_family"):
            values = rows[column].dropna().astype(str).unique()
            if len(values) != 1:
                raise ValueError(
                    f"{label} weekly ledger changes {column} within {incident_id}"
                )
            if not _append_values_equal(window[column], values[0], column):
                raise ValueError(f"{label}.{column} does not reconcile for {incident_id}")
        last = rows.iloc[-1]
        affected = (
            pd.to_numeric(rows["pressure_core_field_count"], errors="raise")
            + pd.to_numeric(
                rows["unresolved_carried_field_count"], errors="raise"
            )
        )
        peak_value = affected.max()
        peak_index = affected[affected.eq(peak_value)].index[0]
        peak_row = rows.loc[peak_index]
        expected = {
            "first_evidence_week": last["first_evidence_week"],
            "confirmed_week": last["confirmed_week"],
            "pressure_off_week": last["pressure_off_week"],
            "recovered_week": last["recovered_week"],
            "closed_week": last["closed_week"],
            "merged_into_incident_id": last["merged_into_incident_id"],
            "terminal_state": last["incident_state"],
            "right_censored": last["right_censored"],
            "observed_week_count": len(rows),
            "active_component_week_count": int(
                (pd.to_numeric(rows["pressure_core_field_count"], errors="raise") > 0).sum()
            ),
            "peak_week": peak_row["timeline_bucket"],
            "peak_affected_field_count": int(peak_value),
            "relapse_count": int(last["relapse_count"]),
            "data_gap_count": int(last["data_gap_count"]),
            "split_count": int(last["split_count"]),
            "merge_count": int(last["merge_count"]),
            "outcome_evidence": "monitoring_signals_only_no_crop_death_inference",
        }
        for column, value in expected.items():
            if not _append_values_equal(window[column], value, column):
                raise ValueError(
                    f"{label}.{column} does not reconcile with weekly rows for {incident_id}"
                )


def _validate_extendable_windows(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    current_weekly: pd.DataFrame,
    *,
    incident_ids: set[str],
    previous_max_week: pd.Timestamp,
) -> None:
    """Protect open/provisional summaries while allowing future-supported extension."""
    if not incident_ids:
        return
    previous_source = previous.copy()
    current_source = current.copy()
    previous_source["incident_id"] = previous_source["incident_id"].astype(str)
    current_source["incident_id"] = current_source["incident_id"].astype(str)
    left = previous_source[
        previous_source["incident_id"].isin(incident_ids)
    ]
    joined = left.merge(
        current_source,
        on="incident_id",
        how="left",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    weekly = current_weekly.copy()
    weekly["incident_id"] = weekly["incident_id"].astype(str)
    weekly["timeline_bucket"] = pd.to_datetime(
        weekly["timeline_bucket"], errors="raise"
    ).dt.normalize()
    max_week = pd.Timestamp(previous_max_week).normalize()
    for row in joined.to_dict("records"):
        incident_id = str(row["incident_id"])
        if pd.isna(row.get("exposure_id_new")):
            raise ValueError(f"Extendable incident window disappeared: {incident_id}")
        for column in WINDOW_ALWAYS_STABLE_COLUMNS:
            if not _append_values_equal(
                row[f"{column}_old"], row[f"{column}_new"], column
            ):
                raise ValueError(
                    f"Ordinary append rewrote historical incident window {column}"
                )
        old_confirmed = _normalize_optional_week(row.get("confirmed_week_old"))
        new_confirmed = _normalize_optional_week(row.get("confirmed_week_new"))
        if old_confirmed is not None and old_confirmed != new_confirmed:
            raise ValueError(
                "Ordinary append rewrote historical incident window confirmed_week"
            )
        if old_confirmed is None and new_confirmed is not None and new_confirmed <= max_week:
            raise ValueError(
                "Ordinary append inserted a historical incident confirmation"
            )
        for column in (
            "pressure_off_week", "recovered_week", "closed_week"
        ):
            old_milestone = _normalize_optional_week(row.get(f"{column}_old"))
            new_milestone = _normalize_optional_week(row.get(f"{column}_new"))
            if old_milestone != new_milestone and new_milestone is not None:
                if new_milestone <= max_week:
                    raise ValueError(
                        f"Ordinary append rewrote historical incident window {column}"
                    )
        for column in WINDOW_MONOTONIC_COUNT_COLUMNS:
            old_value = int(row[f"{column}_old"])
            new_value = int(row[f"{column}_new"])
            if new_value < old_value:
                raise ValueError(
                    f"Ordinary append decreased incident window {column}"
                )

        future = weekly[
            weekly["incident_id"].eq(incident_id)
            & weekly["timeline_bucket"].gt(max_week)
        ]
        changed_columns = [
            column
            for column in WINDOW_REQUIRED_COLUMNS
            if column != "incident_id"
            and not _append_values_equal(
                row[f"{column}_old"], row[f"{column}_new"], column
            )
        ]
        if changed_columns and future.empty:
            raise ValueError(
                "Incident window changed without supporting future weekly rows: "
                + ", ".join(changed_columns)
            )
        terminal_changed = not _append_values_equal(
            row["terminal_state_old"], row["terminal_state_new"],
            "terminal_state",
        )
        if terminal_changed and not bool(row["right_censored_new"]):
            new_closed = _normalize_optional_week(row["closed_week_new"])
            if new_closed is None or new_closed <= max_week:
                raise ValueError(
                    "Incident window terminal-state change is not supported by a future closure"
                )
        merge_changed = not _append_values_equal(
            row["merged_into_incident_id_old"],
            row["merged_into_incident_id_new"],
            "merged_into_incident_id",
        )
        if merge_changed:
            new_closed = _normalize_optional_week(row["closed_week_new"])
            if new_closed is None or new_closed <= max_week:
                raise ValueError(
                    "Incident window merge target changed without a future merge week"
                )
        expected_observed_delta = len(future)
        actual_observed_delta = int(row["observed_week_count_new"]) - int(
            row["observed_week_count_old"]
        )
        if actual_observed_delta != expected_observed_delta:
            raise ValueError(
                "Incident window observed_week_count is not supported by future weekly rows"
            )
        expected_active_delta = int(
            (
                pd.to_numeric(
                    future["pressure_core_field_count"], errors="raise"
                ) > 0
            ).sum()
        )
        actual_active_delta = int(row["active_component_week_count_new"]) - int(
            row["active_component_week_count_old"]
        )
        if actual_active_delta != expected_active_delta:
            raise ValueError(
                "Incident window active_component_week_count is not supported by future weekly rows"
            )
        if not _append_values_equal(
            row["peak_week_old"], row["peak_week_new"], "peak_week"
        ):
            new_peak = _normalize_optional_week(row["peak_week_new"])
            if new_peak is None or new_peak <= max_week:
                raise ValueError(
                    "Incident window peak rewrite is not supported by a future week"
                )


def _append_values_equal(left: Any, right: Any, column: str) -> bool:
    if column.endswith("_week") or column == "timeline_bucket":
        return _normalize_optional_week(left) == _normalize_optional_week(right)
    if column in {"right_censored", "data_censored_at_boundary"}:
        if _is_missing(left) or _is_missing(right):
            return _is_missing(left) and _is_missing(right)
        return bool(left) == bool(right)
    if column.endswith("_count"):
        if _is_missing(left) or _is_missing(right):
            return _is_missing(left) and _is_missing(right)
        return int(left) == int(right)
    return _canonical_scalar(left) == _canonical_scalar(right)


def _normalize_optional_week(value: Any) -> pd.Timestamp | None:
    if _is_missing(value):
        return None
    return pd.Timestamp(value).normalize()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_canonical_scalar(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_scalar(value[key]) for key in sorted(value)
        }
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _canonical_scalar(value.item())
    return value


def _append_counts(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    natural_key: tuple[str, ...],
) -> dict[str, int]:
    overlap = previous.loc[:, list(natural_key)].merge(
        current.loc[:, list(natural_key)], on=list(natural_key), how="inner"
    )
    return {
        "previous": len(previous),
        "current": len(current),
        "overlap": len(overlap),
    }


def _assert_no_historical_insertions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    natural_key: tuple[str, ...],
    *,
    historical_through: Any | None = None,
) -> None:
    """Reject backfills into any week already covered by the prior release."""
    if previous.empty and historical_through is None:
        return
    if "timeline_bucket" not in natural_key:
        raise ValueError("Historical insertion checks require timeline_bucket")
    previous_weeks = pd.to_datetime(
        previous["timeline_bucket"], errors="raise"
    ).dt.normalize()
    current_weeks = pd.to_datetime(
        current["timeline_bucket"], errors="raise"
    ).dt.normalize()
    previous_max = (
        pd.Timestamp(historical_through).normalize()
        if historical_through is not None
        else previous_weeks.max()
    )
    prior_keys = previous.loc[:, list(natural_key)].copy()
    historical_current = current.loc[
        current_weeks <= previous_max, list(natural_key)
    ].copy()
    inserted = historical_current.merge(
        prior_keys,
        on=list(natural_key),
        how="left",
        indicator=True,
    )
    inserted = inserted[inserted["_merge"] == "left_only"]
    if not inserted.empty:
        raise ValueError(
            f"Ordinary append inserted {len(inserted)} new historical natural keys"
        )


def _validate_boundary_censor_resolution(
    previous_boundary: pd.DataFrame,
    current_weekly: pd.DataFrame,
    *,
    previous_weekly: pd.DataFrame,
    previous_max_week: pd.Timestamp,
) -> None:
    """Allow only the deterministic reopening of a max-week data censor."""
    if previous_boundary.empty:
        return
    key = ["timeline_bucket", "incident_id"]
    current = current_weekly.copy()
    current["timeline_bucket"] = _canonical_week(current["timeline_bucket"])
    previous = previous_boundary.copy()
    previous["timeline_bucket"] = _canonical_week(previous["timeline_bucket"])
    joined = previous.merge(
        current,
        on=key,
        how="left",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    history = previous_weekly.copy()
    history["incident_id"] = history["incident_id"].astype(str)
    history["timeline_bucket"] = pd.to_datetime(
        history["timeline_bucket"], errors="raise"
    ).dt.normalize()
    max_week = pd.Timestamp(previous_max_week).normalize()
    for row in joined.to_dict("records"):
        incident_id = str(row["incident_id"])
        if pd.isna(row.get("incident_state_new")):
            raise ValueError(
                f"Boundary-censored incident row disappeared: {incident_id}"
            )
        old_state = str(row.get("incident_state_old") or "").upper()
        old_current = str(row.get("current_state_old") or "").upper()
        old_closed = _normalize_optional_week(row.get("closed_week_old"))
        old_right_censored = bool(row.get("right_censored_old"))
        if (
            old_state != "CLOSED_DATA_CENSORED"
            or old_current != "CLOSED_DATA_CENSORED"
            or old_closed != max_week
            or old_right_censored
        ):
            raise ValueError(
                "Prior max-week data-censor marker is not a valid provisional closure"
            )
        new_state = str(row.get("incident_state_new") or "").upper()
        new_current = str(row.get("current_state_new") or "").upper()
        predecessor = history[
            history["incident_id"].eq(incident_id)
            & history["timeline_bucket"].lt(max_week)
        ].sort_values("timeline_bucket", kind="mergesort").tail(1)
        if predecessor.empty:
            raise ValueError(
                "Prior max-week data censor has no preceding published incident state"
            )
        expected_state = str(
            predecessor.iloc[0]["incident_state"] or ""
        ).upper()
        expected_current = str(
            predecessor.iloc[0]["current_state"] or ""
        ).upper()
        if expected_state.startswith("CLOSED_") or expected_state == "MERGED_INTO":
            raise ValueError(
                "Prior max-week data censor follows an already-terminal incident state"
            )
        if (
            new_state != expected_state
            or new_current != expected_current
            or new_current != new_state
        ):
            raise ValueError(
                "Boundary data-censor resolution does not preserve the preceding lifecycle state"
            )
        if bool(row.get("data_censored_at_boundary_new")):
            raise ValueError(
                "Historical max-week data-censor flag did not resolve on append"
            )
        if not bool(row.get("right_censored_new")):
            raise ValueError(
                "Reopened historical boundary row must be right-censored at that week"
            )
        if _normalize_optional_week(row.get("closed_week_new")) is not None:
            raise ValueError(
                "Reopened historical boundary row must clear its provisional closed_week"
            )


def _validate_context_causality(context: pd.DataFrame) -> None:
    observation_cutoff = (
        "last_observation_date"
        if "last_observation_date" in context
        else "week_last_observation_date"
    )
    for evidence, cutoff in (
        ("stage_source_date", observation_cutoff),
        ("evidence_max_date", "timeline_bucket"),
    ):
        if {evidence, cutoff}.issubset(context.columns):
            left = pd.to_datetime(context[evidence], errors="coerce")
            right = pd.to_datetime(context[cutoff], errors="coerce")
            if (left.notna() & right.notna() & (left > right)).any():
                raise ValueError(f"field_week_context contains future {evidence}")


def _validate_incident_knowledge_time(incidents: pd.DataFrame) -> None:
    if "knowledge_time" not in incidents:
        return
    knowledge = pd.to_datetime(incidents["knowledge_time"], errors="coerce")
    evidence = pd.to_datetime(incidents["timeline_bucket"], errors="coerce")
    if knowledge.isna().any() or (knowledge < evidence).any():
        raise ValueError("incident_weekly_state contains invalid knowledge time")


def _validate_stage_summary_frame(summary: pd.DataFrame) -> None:
    _require_columns(summary, STAGE_SUMMARY_REQUIRED_COLUMNS, "incident_stage_summary")
    _require_nonblank(
        summary,
        (
            "incident_id", "timeline_bucket", "stage_bucket", "exposure_id",
            "crop_name", "hazard_family", "denominator_scope",
            "schema_version", "policy_version", "policy_sha256",
        ),
        "incident_stage_summary",
    )
    counts = summary.loc[:, STAGE_SUMMARY_COUNT_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ValueError("incident_stage_summary contains invalid negative or non-finite counts")
    if ((counts % 1) != 0).any().any():
        raise ValueError("incident_stage_summary counts must be whole numbers")
    relationships = (
        ("evaluable_field_count", "monitored_field_count"),
        ("evaluable_crop_instance_count", "monitored_crop_instance_count"),
        ("pressure_core_crop_instance_count", "monitored_crop_instance_count"),
        ("severe_crop_instance_count", "pressure_core_crop_instance_count"),
        ("watch_frontier_crop_instance_count", "monitored_crop_instance_count"),
        ("impact_lag_crop_instance_count", "monitored_crop_instance_count"),
        ("affected_crop_instance_count", "monitored_crop_instance_count"),
        ("pressure_core_crop_instance_count", "affected_crop_instance_count"),
        ("impact_lag_crop_instance_count", "affected_crop_instance_count"),
        ("crop_observed_cell_count", "footprint_cell_count"),
    )
    for numerator, denominator in relationships:
        if (counts[numerator] > counts[denominator]).any():
            raise ValueError(
                f"incident_stage_summary {numerator} exceeds {denominator}"
            )
    if (counts["footprint_cell_count"] <= 0).any():
        raise ValueError("incident_stage_summary footprint_cell_count must be positive")
    expected_missing = (
        counts["footprint_cell_count"] - counts["crop_observed_cell_count"]
    )
    if not counts["coverage_missing_cell_count"].equals(expected_missing):
        raise ValueError("incident_stage_summary coverage cell counts do not reconcile")

    denominator = counts["monitored_crop_instance_count"]
    for rate_column, numerator_column in (
        ("pressure_signal_rate", "pressure_core_crop_instance_count"),
        ("impact_signal_rate", "affected_crop_instance_count"),
    ):
        rates = pd.to_numeric(summary[rate_column], errors="coerce")
        has_denominator = denominator > 0
        if rates[has_denominator].isna().any() or (
            ~rates[has_denominator].between(0, 1)
        ).any():
            raise ValueError(f"incident_stage_summary {rate_column} is invalid")
        if rates[~has_denominator].notna().any():
            raise ValueError(
                f"incident_stage_summary {rate_column} must be null without a denominator"
            )
        expected = counts.loc[has_denominator, numerator_column] / denominator[has_denominator]
        if ((rates[has_denominator] - expected).abs() > 1e-12).any():
            raise ValueError(f"incident_stage_summary {rate_column} does not reconcile")


def _validate_story_references(frames: Mapping[str, pd.DataFrame]) -> None:
    components = set(frames["weekly_components"]["component_id"].astype(str))
    membership_components = set(frames["component_membership"]["component_id"].astype(str))
    if not membership_components.issubset(components):
        raise ValueError("Component membership references unknown components")
    exposures = set(frames["exposure_weekly_state"]["exposure_id"].astype(str))
    incidents = frames["incident_weekly_state"]
    if "exposure_id" not in incidents:
        raise ValueError("incident_weekly_state is missing exposure_id")
    if not set(incidents["exposure_id"].astype(str)).issubset(exposures):
        raise ValueError("Crop-impact stories reference unknown exposures")
    known_incidents = set(incidents["incident_id"].astype(str))
    for name in ("incident_membership", "incident_windows"):
        unknown = set(frames[name]["incident_id"].astype(str)) - known_incidents
        if unknown:
            raise ValueError(f"{name} references unknown incident IDs")
    incident_weeks = set(
        zip(
            incidents["incident_id"].astype(str),
            pd.to_datetime(incidents["timeline_bucket"], errors="raise").dt.normalize(),
        )
    )
    stage_summary = frames["incident_stage_summary"]
    stage_weeks = set(
        zip(
            stage_summary["incident_id"].astype(str),
            pd.to_datetime(
                stage_summary["timeline_bucket"], errors="raise"
            ).dt.normalize(),
        )
    )
    if not stage_weeks.issubset(incident_weeks):
        raise ValueError("incident_stage_summary references unknown incident weeks")
    if incident_weeks != stage_weeks:
        raise ValueError("incident_stage_summary does not cover every incident week")


def _validate_lineage(lineage: pd.DataFrame) -> None:
    candidates = (
        ("parent_exposure_id", "child_exposure_id"),
        ("parent_incident_id", "child_incident_id"),
    )
    edges: list[tuple[str, str]] = []
    for parent, child in candidates:
        if {parent, child}.issubset(lineage.columns):
            subset = lineage.dropna(subset=[parent, child])
            edges.extend(zip(subset[parent].astype(str), subset[child].astype(str)))
    graph: dict[str, set[str]] = defaultdict(set)
    for parent, child in edges:
        if parent == child:
            raise ValueError("Incident lineage contains a self-cycle")
        graph[parent].add(child)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("Incident lineage contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def _validate_directory_stage_summary(
    connection: duckdb.DuckDBPyConnection, root: Path
) -> None:
    summary_path = root / "incident_stage_summary.parquet"
    incident_path = root / "incident_weekly_state.parquet"
    columns = set(connection.read_parquet(str(summary_path)).columns)
    missing = sorted(set(STAGE_SUMMARY_REQUIRED_COLUMNS) - columns)
    if missing:
        raise ValueError(
            "incident_stage_summary is missing columns: " + ", ".join(missing)
        )
    connection.read_parquet(str(summary_path)).create_view(
        "incident_stage_summary", replace=True
    )
    connection.read_parquet(str(incident_path)).create_view(
        "incident_week_source", replace=True
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW incident_weeks AS "
        "SELECT DISTINCT CAST(incident_id AS VARCHAR) AS incident_id, "
        "CAST(timeline_bucket AS DATE) AS timeline_bucket "
        "FROM incident_week_source"
    )

    count_predicates = []
    for column in STAGE_SUMMARY_COUNT_COLUMNS:
        quoted = _quote_identifier(column)
        value = f"TRY_CAST({quoted} AS DOUBLE)"
        count_predicates.extend(
            (
                f"{value} IS NULL",
                f"NOT ISFINITE({value})",
                f"{value} < 0",
                f"{value} <> FLOOR({value})",
            )
        )
    invalid_counts = int(
        connection.execute(
            "SELECT COUNT(*) FROM incident_stage_summary WHERE "
            + " OR ".join(count_predicates)
        ).fetchone()[0]
    )
    if invalid_counts:
        raise ValueError(
            "incident_stage_summary contains invalid negative, non-finite, or fractional counts"
        )

    invalid_relationships = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM incident_stage_summary
            WHERE evaluable_field_count > monitored_field_count
               OR evaluable_crop_instance_count > monitored_crop_instance_count
               OR pressure_core_crop_instance_count > monitored_crop_instance_count
               OR severe_crop_instance_count > pressure_core_crop_instance_count
               OR watch_frontier_crop_instance_count > monitored_crop_instance_count
               OR impact_lag_crop_instance_count > monitored_crop_instance_count
               OR affected_crop_instance_count > monitored_crop_instance_count
               OR pressure_core_crop_instance_count > affected_crop_instance_count
               OR impact_lag_crop_instance_count > affected_crop_instance_count
               OR footprint_cell_count <= 0
               OR crop_observed_cell_count > footprint_cell_count
               OR coverage_missing_cell_count
                    <> footprint_cell_count - crop_observed_cell_count
            """
        ).fetchone()[0]
    )
    if invalid_relationships:
        raise ValueError("incident_stage_summary denominator counts do not reconcile")

    invalid_rates = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM incident_stage_summary
            WHERE (
                monitored_crop_instance_count = 0
                AND (pressure_signal_rate IS NOT NULL OR impact_signal_rate IS NOT NULL)
            ) OR (
                monitored_crop_instance_count > 0
                AND (
                    pressure_signal_rate IS NULL
                    OR NOT ISFINITE(TRY_CAST(pressure_signal_rate AS DOUBLE))
                    OR pressure_signal_rate < 0 OR pressure_signal_rate > 1
                    OR ABS(
                        pressure_signal_rate
                        - pressure_core_crop_instance_count::DOUBLE
                            / monitored_crop_instance_count
                    ) > 1e-12
                    OR impact_signal_rate IS NULL
                    OR NOT ISFINITE(TRY_CAST(impact_signal_rate AS DOUBLE))
                    OR impact_signal_rate < 0 OR impact_signal_rate > 1
                    OR ABS(
                        impact_signal_rate
                        - affected_crop_instance_count::DOUBLE
                            / monitored_crop_instance_count
                    ) > 1e-12
                )
            )
            """
        ).fetchone()[0]
    )
    if invalid_rates:
        raise ValueError("incident_stage_summary signal rates do not reconcile")

    metadata_predicate = " OR ".join(
        f"{_quote_identifier(column)} IS NULL "
        f"OR TRIM(CAST({_quote_identifier(column)} AS VARCHAR)) = ''"
        for column in (
            "exposure_id", "crop_name", "hazard_family", "denominator_scope",
            "schema_version", "policy_version", "policy_sha256",
        )
    )
    invalid_metadata = int(
        connection.execute(
            f"SELECT COUNT(*) FROM incident_stage_summary WHERE {metadata_predicate}"
        ).fetchone()[0]
    )
    if invalid_metadata:
        raise ValueError("incident_stage_summary contains blank denominator metadata")

    unknown = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM incident_stage_summary s
            LEFT JOIN incident_weeks i
              ON CAST(s.incident_id AS VARCHAR) = i.incident_id
             AND CAST(s.timeline_bucket AS DATE) = i.timeline_bucket
            WHERE i.incident_id IS NULL
            """
        ).fetchone()[0]
    )
    if unknown:
        raise ValueError(
            f"incident_stage_summary references unknown incident weeks: {unknown}"
        )
    uncovered = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM incident_weeks i
            LEFT JOIN (
                SELECT DISTINCT CAST(incident_id AS VARCHAR) AS incident_id,
                    CAST(timeline_bucket AS DATE) AS timeline_bucket
                FROM incident_stage_summary
            ) s USING (incident_id, timeline_bucket)
            WHERE s.incident_id IS NULL
            """
        ).fetchone()[0]
    )
    if uncovered:
        raise ValueError(
            f"incident_stage_summary does not cover every incident week: {uncovered}"
        )


def _validate_directory_windows(
    connection: duckdb.DuckDBPyConnection, root: Path
) -> None:
    """Reconcile first-release window summaries in DuckDB without bulk loading."""
    window_path = root / "incident_windows.parquet"
    weekly_path = root / "incident_weekly_state.parquet"
    window_columns = set(connection.read_parquet(str(window_path)).columns)
    missing_windows = sorted(set(WINDOW_REQUIRED_COLUMNS) - window_columns)
    if missing_windows:
        raise ValueError(
            "incident_windows is missing columns: " + ", ".join(missing_windows)
        )
    weekly_required = {
        "incident_id", "timeline_bucket", "exposure_id", "crop_name",
        "hazard_family", "incident_state", "right_censored",
        "first_evidence_week", "confirmed_week", "pressure_off_week",
        "recovered_week", "closed_week", "merged_into_incident_id",
        "pressure_core_field_count", "unresolved_carried_field_count",
        "relapse_count", "data_gap_count", "split_count", "merge_count",
    }
    weekly_columns = set(connection.read_parquet(str(weekly_path)).columns)
    missing_weekly = sorted(weekly_required - weekly_columns)
    if missing_weekly:
        raise ValueError(
            "incident_weekly_state cannot reconcile windows; missing columns: "
            + ", ".join(missing_weekly)
        )
    connection.read_parquet(str(window_path)).create_view(
        "incident_window_source", replace=True
    )
    connection.read_parquet(str(weekly_path)).create_view(
        "incident_window_weekly_source", replace=True
    )
    invalid_weekly_identity = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT incident_id
                FROM incident_window_weekly_source
                GROUP BY incident_id
                HAVING COUNT(DISTINCT CAST(exposure_id AS VARCHAR)) <> 1
                    OR COUNT(DISTINCT CAST(crop_name AS VARCHAR)) <> 1
                    OR COUNT(DISTINCT CAST(hazard_family AS VARCHAR)) <> 1
            )
            """
        ).fetchone()[0]
    )
    if invalid_weekly_identity:
        raise ValueError("incident weekly ledger changes window identity dimensions")
    mismatches = int(
        connection.execute(
            """
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY incident_id
                        ORDER BY CAST(timeline_bucket AS DATE) DESC
                    ) AS last_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY incident_id
                        ORDER BY (
                            TRY_CAST(pressure_core_field_count AS BIGINT)
                            + TRY_CAST(unresolved_carried_field_count AS BIGINT)
                        ) DESC, CAST(timeline_bucket AS DATE) ASC
                    ) AS peak_rank
                FROM incident_window_weekly_source
            ), expected AS (
                SELECT
                    CAST(incident_id AS VARCHAR) AS incident_id,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(exposure_id AS VARCHAR) END)
                        AS exposure_id,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(crop_name AS VARCHAR) END)
                        AS crop_name,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(hazard_family AS VARCHAR) END)
                        AS hazard_family,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(first_evidence_week AS DATE) END)
                        AS first_evidence_week,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(confirmed_week AS DATE) END)
                        AS confirmed_week,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(pressure_off_week AS DATE) END)
                        AS pressure_off_week,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(recovered_week AS DATE) END)
                        AS recovered_week,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(closed_week AS DATE) END)
                        AS closed_week,
                    MAX(CASE WHEN last_rank = 1
                        THEN CAST(merged_into_incident_id AS VARCHAR) END)
                        AS merged_into_incident_id,
                    MAX(CASE WHEN last_rank = 1 THEN CAST(incident_state AS VARCHAR) END)
                        AS terminal_state,
                    MAX(CASE WHEN last_rank = 1 THEN TRY_CAST(right_censored AS BOOLEAN) END)
                        AS right_censored,
                    COUNT(*)::BIGINT AS observed_week_count,
                    COUNT_IF(TRY_CAST(pressure_core_field_count AS BIGINT) > 0)::BIGINT
                        AS active_component_week_count,
                    MAX(CASE WHEN peak_rank = 1 THEN CAST(timeline_bucket AS DATE) END)
                        AS peak_week,
                    MAX(
                        TRY_CAST(pressure_core_field_count AS BIGINT)
                        + TRY_CAST(unresolved_carried_field_count AS BIGINT)
                    )::BIGINT AS peak_affected_field_count,
                    MAX(CASE WHEN last_rank = 1 THEN TRY_CAST(relapse_count AS BIGINT) END)
                        AS relapse_count,
                    MAX(CASE WHEN last_rank = 1 THEN TRY_CAST(data_gap_count AS BIGINT) END)
                        AS data_gap_count,
                    MAX(CASE WHEN last_rank = 1 THEN TRY_CAST(split_count AS BIGINT) END)
                        AS split_count,
                    MAX(CASE WHEN last_rank = 1 THEN TRY_CAST(merge_count AS BIGINT) END)
                        AS merge_count
                FROM ranked
                GROUP BY incident_id
            )
            SELECT COUNT(*)
            FROM incident_window_source w
            FULL OUTER JOIN expected e
              ON CAST(w.incident_id AS VARCHAR) = e.incident_id
            WHERE w.incident_id IS NULL OR e.incident_id IS NULL
               OR CAST(w.exposure_id AS VARCHAR) IS DISTINCT FROM e.exposure_id
               OR CAST(w.crop_name AS VARCHAR) IS DISTINCT FROM e.crop_name
               OR CAST(w.hazard_family AS VARCHAR) IS DISTINCT FROM e.hazard_family
               OR CAST(w.first_evidence_week AS DATE) IS DISTINCT FROM e.first_evidence_week
               OR CAST(w.confirmed_week AS DATE) IS DISTINCT FROM e.confirmed_week
               OR CAST(w.pressure_off_week AS DATE) IS DISTINCT FROM e.pressure_off_week
               OR CAST(w.recovered_week AS DATE) IS DISTINCT FROM e.recovered_week
               OR CAST(w.closed_week AS DATE) IS DISTINCT FROM e.closed_week
               OR CAST(w.merged_into_incident_id AS VARCHAR)
                    IS DISTINCT FROM e.merged_into_incident_id
               OR CAST(w.terminal_state AS VARCHAR) IS DISTINCT FROM e.terminal_state
               OR TRY_CAST(w.right_censored AS BOOLEAN) IS DISTINCT FROM e.right_censored
               OR TRY_CAST(w.observed_week_count AS BIGINT)
                    IS DISTINCT FROM e.observed_week_count
               OR TRY_CAST(w.active_component_week_count AS BIGINT)
                    IS DISTINCT FROM e.active_component_week_count
               OR CAST(w.peak_week AS DATE) IS DISTINCT FROM e.peak_week
               OR TRY_CAST(w.peak_affected_field_count AS BIGINT)
                    IS DISTINCT FROM e.peak_affected_field_count
               OR TRY_CAST(w.relapse_count AS BIGINT) IS DISTINCT FROM e.relapse_count
               OR TRY_CAST(w.data_gap_count AS BIGINT) IS DISTINCT FROM e.data_gap_count
               OR TRY_CAST(w.split_count AS BIGINT) IS DISTINCT FROM e.split_count
               OR TRY_CAST(w.merge_count AS BIGINT) IS DISTINCT FROM e.merge_count
               OR CAST(w.outcome_evidence AS VARCHAR) IS DISTINCT FROM
                    'monitoring_signals_only_no_crop_death_inference'
               OR (
                    TRY_CAST(w.right_censored AS BOOLEAN)
                    AND CAST(w.closed_week AS DATE) IS NOT NULL
               )
               OR (
                    NOT TRY_CAST(w.right_censored AS BOOLEAN)
                    AND CAST(w.closed_week AS DATE) IS NULL
               )
               OR (
                    UPPER(CAST(w.terminal_state AS VARCHAR)) = 'CLOSED_RECOVERED'
                    AND (
                        CAST(w.recovered_week AS DATE) IS NULL
                        OR CAST(w.recovered_week AS DATE)
                            IS DISTINCT FROM CAST(w.closed_week AS DATE)
                    )
               )
               OR (
                    UPPER(CAST(w.terminal_state AS VARCHAR)) <> 'CLOSED_RECOVERED'
                    AND CAST(w.recovered_week AS DATE) IS NOT NULL
               )
            """
        ).fetchone()[0]
    )
    if mismatches:
        raise ValueError(
            f"incident_windows do not reconcile with causal weekly rows: {mismatches}"
        )


def _validate_directory_references(
    connection: duckdb.DuckDBPyConnection, root: Path
) -> None:
    queries = (
        (
            "component memberships reference unknown components",
            """
            SELECT COUNT(*) FROM read_parquet(?) m
            LEFT JOIN read_parquet(?) c USING (component_id)
            WHERE c.component_id IS NULL
            """,
            [root / "component_membership.parquet", root / "weekly_components.parquet"],
        ),
        (
            "crop-impact stories reference unknown exposures",
            """
            SELECT COUNT(*) FROM read_parquet(?) i
            LEFT JOIN read_parquet(?) e USING (exposure_id)
            WHERE e.exposure_id IS NULL
            """,
            [root / "incident_weekly_state.parquet", root / "exposure_weekly_state.parquet"],
        ),
        (
            "incident memberships reference unknown stories",
            """
            SELECT COUNT(*) FROM read_parquet(?) m
            LEFT JOIN (SELECT DISTINCT incident_id FROM read_parquet(?)) i USING (incident_id)
            WHERE i.incident_id IS NULL
            """,
            [root / "incident_membership.parquet", root / "incident_weekly_state.parquet"],
        ),
        (
            "incident windows reference unknown stories",
            """
            SELECT COUNT(*) FROM read_parquet(?) w
            LEFT JOIN (SELECT DISTINCT incident_id FROM read_parquet(?)) i USING (incident_id)
            WHERE i.incident_id IS NULL
            """,
            [root / "incident_windows.parquet", root / "incident_weekly_state.parquet"],
        ),
    )
    for message, query, paths in queries:
        count = int(connection.execute(query, [str(path) for path in paths]).fetchone()[0])
        if count:
            raise ValueError(f"{message}: {count}")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _reject_death_claims(frame: pd.DataFrame, name: str) -> None:
    semantic = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("state", "status", "outcome", "reason"))
    ]
    for column in semantic:
        values = frame[column].dropna().astype(str).str.upper()
        if values.str.contains(r"(?:^|_)DEAD(?:$|_)", regex=True).any():
            raise ValueError(f"{name}.{column} makes an unsupported crop-death claim")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _require_nonblank(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}.{column} contains null or blank keys")


__all__ = [
    "FINAL_ARTIFACT_KEYS",
    "REQUIRED_SOURCE_ARTIFACTS",
    "artifact_hashes",
    "assert_append_stability",
    "file_sha256",
    "validate_append_stability",
    "validate_final_frames",
    "validate_final_artifact_directory",
    "validate_source_generation",
]
