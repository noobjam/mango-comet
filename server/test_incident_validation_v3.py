from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_validation_v3 import (
    APPEND_STABLE_LIFECYCLE_COLUMNS,
    APPEND_STABLE_WEEKLY_CONTENT_COLUMNS,
    artifact_hashes,
    assert_append_stability,
    validate_append_stability,
    validate_final_artifact_directory,
    validate_final_frames,
    validate_source_generation,
)


def _stage_summary_row(
    *,
    week: str = "2026-01-05",
    incident_id: str = "incident-1",
    exposure_id: str = "exposure-1",
    stage_bucket: str = "vegetative",
    monitored_count: int = 2,
) -> dict[str, object]:
    pressure_count = min(1, monitored_count)
    rate = pressure_count / monitored_count if monitored_count else None
    return {
        "timeline_bucket": week,
        "incident_id": incident_id,
        "exposure_id": exposure_id,
        "crop_name": "maize",
        "hazard_family": "heat",
        "stage_bucket": stage_bucket,
        "monitored_field_count": monitored_count,
        "evaluable_field_count": monitored_count,
        "monitored_crop_instance_count": monitored_count,
        "evaluable_crop_instance_count": monitored_count,
        "pressure_core_crop_instance_count": pressure_count,
        "severe_crop_instance_count": 0,
        "watch_frontier_crop_instance_count": 0,
        "impact_lag_crop_instance_count": 0,
        "affected_crop_instance_count": pressure_count,
        "pressure_signal_rate": rate,
        "impact_signal_rate": rate,
        "footprint_cell_count": 2,
        "crop_observed_cell_count": 2,
        "coverage_missing_cell_count": 0,
        "global_crop_week_unmappable_instance_count": 0,
        "denominator_scope": "crop_instances_in_pressure_watch_and_impact_cells",
        "schema_version": "incident-stage-summary-v3/2",
        "policy_version": "policy-v3",
        "policy_sha256": "f" * 64,
    }


def _frames() -> dict[str, pd.DataFrame]:
    week = "2026-01-05"
    return {
        "field_week_context": pd.DataFrame(
            [{
                "timeline_bucket": week, "field_id": "f1", "crop_instance_id": "crop1",
                "stage_source_date": week, "last_observation_date": week,
            }]
        ),
        "stage_baseline": pd.DataFrame(
            [{"hazard_family": "heat", "stage_bucket": "vegetative", "iso_week": 1}]
        ),
        "weekly_exposure_cells": pd.DataFrame(
            [{"timeline_bucket": week, "hazard_family": "heat", "cell_id": "c1"}]
        ),
        "weekly_components": pd.DataFrame([{"component_id": "component-1"}]),
        "component_membership": pd.DataFrame(
            [{"component_id": "component-1", "crop_instance_id": "crop1", "episode_id": "e1"}]
        ),
        "exposure_weekly_state": pd.DataFrame(
            [{"exposure_id": "exposure-1", "timeline_bucket": week}]
        ),
        "incident_weekly_state": pd.DataFrame(
            [{
                "incident_id": "incident-1", "exposure_id": "exposure-1",
                "crop_name": "maize", "hazard_family": "heat",
                "timeline_bucket": week, "incident_state": "ACTIVE",
                "right_censored": True, "first_evidence_week": week,
                "confirmed_week": None, "pressure_off_week": None,
                "recovered_week": None, "closed_week": None,
                "merged_into_incident_id": None,
                "pressure_core_field_count": 1,
                "unresolved_carried_field_count": 0,
                "relapse_count": 0, "data_gap_count": 0,
                "split_count": 0, "merge_count": 0,
            }]
        ),
        "incident_stage_summary": pd.DataFrame([_stage_summary_row(week=week)]),
        "incident_membership": pd.DataFrame(
            [{
                "incident_id": "incident-1", "timeline_bucket": week,
                "crop_instance_id": "crop1", "membership_role": "pressure_core",
            }]
        ),
        "incident_windows": pd.DataFrame(
            [{
                "incident_id": "incident-1", "exposure_id": "exposure-1",
                "crop_name": "maize", "hazard_family": "heat",
                "first_evidence_week": week, "confirmed_week": None,
                "pressure_off_week": None, "recovered_week": None,
                "closed_week": None, "merged_into_incident_id": None,
                "terminal_state": "ACTIVE", "right_censored": True,
                "observed_week_count": 1, "active_component_week_count": 1,
                "peak_week": week, "peak_affected_field_count": 1,
                "relapse_count": 0, "data_gap_count": 0,
                "split_count": 0, "merge_count": 0,
                "outcome_evidence": "monitoring_signals_only_no_crop_death_inference",
            }]
        ),
        "incident_lineage": pd.DataFrame(
            columns=["parent_exposure_id", "child_exposure_id"]
        ),
    }


class IncidentValidationV3Tests(unittest.TestCase):
    def test_source_generation_requires_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "run": {"status": "complete", "immutable": True},
                        "policy": {"version": "v1", "sha256": "f" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_source_generation(root)
            for name in (
                "daily_causal_signals.parquet", "event_state_snapshots.parquet",
                "event_windows.parquet", "story_day_membership.parquet",
                "map_field_geometry.parquet",
            ):
                (root / name).touch()
            manifest = validate_source_generation(root)
            self.assertEqual(manifest["run"]["status"], "complete")

    def test_final_validation_rejects_future_stage_and_death_claim(self) -> None:
        frames = _frames()
        self.assertTrue(validate_final_frames(frames)["passed"])
        future = _frames()
        future["field_week_context"].loc[0, "stage_source_date"] = "2026-01-06"
        with self.assertRaisesRegex(ValueError, "future stage_source_date"):
            validate_final_frames(future)
        death = _frames()
        death["incident_windows"].loc[0, "terminal_state"] = "CLOSED_DEAD"
        with self.assertRaisesRegex(ValueError, "death claim"):
            validate_final_frames(death)

    def test_lineage_cycles_and_unknown_references_fail_closed(self) -> None:
        cycle = _frames()
        cycle["incident_lineage"] = pd.DataFrame(
            [
                {"parent_exposure_id": "a", "child_exposure_id": "b"},
                {"parent_exposure_id": "b", "child_exposure_id": "a"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_final_frames(cycle)
        unknown = _frames()
        unknown["component_membership"].loc[0, "component_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown components"):
            validate_final_frames(unknown)

    def test_stage_summary_validation_is_reconciled_and_referential(self) -> None:
        invalid_rate = _frames()
        invalid_rate["incident_stage_summary"].loc[0, "pressure_signal_rate"] = 0.9
        with self.assertRaisesRegex(ValueError, "pressure_signal_rate does not reconcile"):
            validate_final_frames(invalid_rate)

        invalid_count = _frames()
        invalid_count["incident_stage_summary"].loc[
            0, "affected_crop_instance_count"
        ] = 3
        with self.assertRaisesRegex(ValueError, "exceeds monitored_crop_instance_count"):
            validate_final_frames(invalid_count)

        unknown_week = _frames()
        unknown_week["incident_stage_summary"].loc[0, "timeline_bucket"] = "2026-01-12"
        with self.assertRaisesRegex(ValueError, "unknown incident weeks"):
            validate_final_frames(unknown_week)

    def test_append_stability_compares_overlapping_natural_keys(self) -> None:
        old = pd.DataFrame(
            [{"week": "2026-01-05", "component": "c1", "exposure_id": "x1"}]
        )
        extended = pd.concat(
            [
                old,
                pd.DataFrame(
                    [{"week": "2026-01-12", "component": "c2", "exposure_id": "x1"}]
                ),
            ],
            ignore_index=True,
        )
        assert_append_stability(
            old, extended, natural_key=("week", "component"), identity_column="exposure_id"
        )
        changed = extended.copy()
        changed.loc[0, "exposure_id"] = "rewritten"
        with self.assertRaisesRegex(ValueError, "rewrote"):
            assert_append_stability(
                old, changed,
                natural_key=("week", "component"), identity_column="exposure_id",
            )
        dropped = old.iloc[0:0].copy()
        with self.assertRaisesRegex(ValueError, "dropped"):
            assert_append_stability(
                old,
                dropped,
                natural_key=("week", "component"),
                identity_column="exposure_id",
            )

    def test_release_append_stability_checks_all_identity_layers_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_append_release(previous)
            _write_append_release(current, include_future=True, reverse_cells=True)

            result = validate_append_stability(previous, current)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["comparisons"]["component_id"]["overlap"], 1
            )
            self.assertEqual(
                result["comparisons"]["historical_weekly_content"]["overlap"],
                1,
            )
            self.assertEqual(
                result["comparisons"]["historical_stage_denominators"]["overlap"],
                1,
            )

            drift_cases = (
                ("component", {"component_id": "component-rewritten"}),
                ("exposure", {"exposure_id": "exposure-rewritten"}),
                ("incident", {"incident_id": "incident-rewritten"}),
                ("content", {"pressure_core_field_count": 99}),
            )
            for label, changes in drift_cases:
                drifted = root / f"drift-{label}"
                _write_append_release(drifted, **changes)
                with self.subTest(label=label), self.assertRaisesRegex(
                    ValueError, "rewrote"
                ):
                    validate_append_stability(previous, drifted)

            lifecycle_drift = root / "drift-lifecycle"
            _write_append_release(
                lifecycle_drift, incident_state="CLOSED_RECOVERED"
            )
            with self.assertRaisesRegex(ValueError, "rewrote"):
                validate_append_stability(previous, lifecycle_drift)

            missing_schema = root / "missing-stable-column"
            _write_append_release(
                missing_schema, drop_incident_column="relapse_count"
            )
            with self.assertRaisesRegex(
                ValueError, "missing append-stable columns.*relapse_count"
            ):
                validate_append_stability(previous, missing_schema)

            stage_drift = root / "drift-stage-denominator"
            _write_append_release(stage_drift, stage_monitored_count=3)
            with self.assertRaisesRegex(ValueError, "rewrote"):
                validate_append_stability(previous, stage_drift)

            stage_backfill = root / "backfill-stage"
            _write_append_release(stage_backfill, include_historical_stage=True)
            with self.assertRaisesRegex(
                ValueError, "inserted .* historical natural keys"
            ):
                validate_append_stability(previous, stage_backfill)

    def test_release_append_stability_rejects_tampered_previous_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_append_release(previous)
            _write_append_release(current)
            assignments = pd.read_parquet(
                previous / "exposure_component_assignments.parquet"
            )
            assignments.loc[0, "exposure_id"] = "tampered"
            assignments.to_parquet(
                previous / "exposure_component_assignments.parquet", index=False
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_append_stability(previous, current)

    def test_append_allows_only_explicit_prior_boundary_censor_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_append_release(
                previous,
                incident_state="CLOSED_DATA_CENSORED",
                data_censored_at_boundary=True,
                include_predecessor=True,
            )
            _write_append_release(
                current,
                incident_state="ACTIVE",
                data_censored_at_boundary=False,
                lifecycle_data_gap_count=1,
                lifecycle_coverage_gap_streak=1,
                include_predecessor=True,
                include_future=True,
            )
            result = validate_append_stability(previous, current)
            self.assertEqual(
                result["comparisons"]["historical_lifecycle_content"]
                ["prior_boundary_resolutions"],
                1,
            )
            self.assertEqual(
                result["comparisons"]["historical_terminal_windows"]
                ["prior_boundary_censored_allowed_to_reopen"],
                1,
            )
            wrong_state = root / "wrong-reopened-state"
            _write_append_release(
                wrong_state,
                incident_state="RECOVERING",
                lifecycle_data_gap_count=1,
                lifecycle_coverage_gap_streak=1,
                include_predecessor=True,
                include_future=True,
            )
            with self.assertRaisesRegex(ValueError, "preceding lifecycle state"):
                validate_append_stability(previous, wrong_state)

    def test_boundary_resolution_freezes_causal_milestones_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            _write_append_release(
                previous,
                incident_state="CLOSED_DATA_CENSORED",
                data_censored_at_boundary=True,
                include_predecessor=True,
            )
            tamper_cases = {
                "first-evidence": ("first_evidence_week", "2025-12-22"),
                "confirmed": ("confirmed_week", "2026-01-05"),
                "relapse": ("relapse_count", 7),
            }
            for label, (column, value) in tamper_cases.items():
                current = root / label
                _write_append_release(
                    current,
                    incident_state="ACTIVE",
                    include_future=True,
                    lifecycle_data_gap_count=1,
                    lifecycle_coverage_gap_streak=1,
                    include_predecessor=True,
                )
                weekly_path = current / "incident_weekly_state.parquet"
                weekly = pd.read_parquet(weekly_path)
                weekly.loc[0, column] = value
                weekly.to_parquet(weekly_path, index=False)
                with self.subTest(label=label), self.assertRaisesRegex(
                    ValueError, "rewrote"
                ):
                    validate_append_stability(previous, current)

    def test_extendable_window_tampering_requires_causal_future_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            _write_append_release(previous)
            tamper_cases = {
                "identity": ("crop_name", "tampered_crop"),
                "first-evidence": ("first_evidence_week", "2025-12-29"),
                "outcome": ("outcome_evidence", "unsupported_outcome"),
                "decreasing-count": ("observed_week_count", 0),
                "unsupported-terminal": ("terminal_state", "CLOSED_RECOVERED"),
                "unsupported-peak": ("peak_week", "2026-01-12"),
            }
            for label, (column, value) in tamper_cases.items():
                current = root / label
                _write_append_release(current, include_future=True)
                window_path = current / "incident_windows.parquet"
                windows = pd.read_parquet(window_path)
                windows.loc[0, column] = value
                windows.to_parquet(window_path, index=False)
                with self.subTest(label=label), self.assertRaisesRegex(
                    ValueError, "reconcile|rewrote|decreased|supported"
                ):
                    validate_append_stability(previous, current)

    def test_extendable_window_rejects_backdated_or_incoherent_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            _write_append_release(previous)

            backdated_pressure = root / "backdated-pressure"
            _write_append_release(backdated_pressure, include_future=True)
            weekly_path = backdated_pressure / "incident_weekly_state.parquet"
            window_path = backdated_pressure / "incident_windows.parquet"
            weekly = pd.read_parquet(weekly_path)
            windows = pd.read_parquet(window_path)
            weekly.loc[weekly["timeline_bucket"] == "2026-01-12", "pressure_off_week"] = (
                "2026-01-05"
            )
            windows.loc[0, "pressure_off_week"] = "2026-01-05"
            weekly.to_parquet(weekly_path, index=False)
            windows.to_parquet(window_path, index=False)
            with self.assertRaisesRegex(ValueError, "pressure_off_week"):
                validate_append_stability(previous, backdated_pressure)

            backdated_recovery = root / "backdated-recovery"
            _write_append_release(backdated_recovery, include_future=True)
            weekly_path = backdated_recovery / "incident_weekly_state.parquet"
            window_path = backdated_recovery / "incident_windows.parquet"
            weekly = pd.read_parquet(weekly_path)
            windows = pd.read_parquet(window_path)
            future = weekly["timeline_bucket"] == "2026-01-12"
            weekly.loc[future, ["incident_state", "current_state"]] = (
                "CLOSED_RECOVERED"
            )
            weekly.loc[future, "right_censored"] = False
            weekly.loc[future, ["recovered_week", "closed_week"]] = "2026-01-05"
            windows.loc[0, "terminal_state"] = "CLOSED_RECOVERED"
            windows.loc[0, "right_censored"] = False
            windows.loc[0, ["recovered_week", "closed_week"]] = "2026-01-05"
            weekly.to_parquet(weekly_path, index=False)
            windows.to_parquet(window_path, index=False)
            with self.assertRaisesRegex(ValueError, "recovered_week|closed_week"):
                validate_append_stability(previous, backdated_recovery)

            incoherent_open = root / "incoherent-open"
            _write_append_release(incoherent_open, include_future=True)
            weekly_path = incoherent_open / "incident_weekly_state.parquet"
            window_path = incoherent_open / "incident_windows.parquet"
            weekly = pd.read_parquet(weekly_path)
            windows = pd.read_parquet(window_path)
            weekly.loc[
                weekly["timeline_bucket"] == "2026-01-12", "closed_week"
            ] = "2026-01-12"
            windows.loc[0, "closed_week"] = "2026-01-12"
            weekly.to_parquet(weekly_path, index=False)
            windows.to_parquet(window_path, index=False)
            with self.assertRaisesRegex(ValueError, "right-censored"):
                validate_append_stability(previous, incoherent_open)

    def test_append_rejects_backfilled_keys_in_published_weeks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_append_release(previous)
            _write_append_release(current, include_historical_extra=True)
            with self.assertRaisesRegex(
                ValueError, "inserted .* historical natural keys"
            ):
                validate_append_stability(previous, current)

    def test_append_protects_membership_terminal_windows_and_empty_lineage_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            _write_append_release(
                previous, incident_state="CLOSED_RECOVERED"
            )

            membership_drift = root / "membership-drift"
            _write_append_release(
                membership_drift,
                incident_state="CLOSED_RECOVERED",
                membership_event_state="SEVERE",
            )
            with self.assertRaisesRegex(ValueError, "rewrote"):
                validate_append_stability(previous, membership_drift)

            window_drift = root / "window-drift"
            _write_append_release(
                window_drift,
                incident_state="CLOSED_RECOVERED",
                window_observed_week_count=99,
            )
            with self.assertRaisesRegex(ValueError, "rewrote|does not reconcile"):
                validate_append_stability(previous, window_drift)

            lineage_backfill = root / "lineage-backfill"
            _write_append_release(
                lineage_backfill, incident_state="CLOSED_RECOVERED"
            )
            pd.DataFrame(
                [
                    {
                        "lineage_id": "lineage-backfill",
                        "timeline_bucket": "2026-01-05",
                        "lineage_type": "split",
                        "crop_name_normalized": "maize",
                        "parent_exposure_id": "exposure-1",
                        "child_exposure_id": "exposure-2",
                        "parent_incident_id": "incident-1",
                        "child_incident_id": "incident-2",
                        "previous_component_id": "component-1",
                        "current_component_id": "component-2",
                        "score": 0.9,
                        "schema_version": "crop-incident-lineage-v3/1",
                    }
                ]
            ).to_parquet(lineage_backfill / "incident_lineage.parquet", index=False)
            with self.assertRaisesRegex(
                ValueError, "inserted .* historical natural keys"
            ):
                validate_append_stability(previous, lineage_backfill)

    def test_staged_directory_validation_reconciles_without_bulk_loading(self) -> None:
        frames = _frames()
        frames["field_week_context"] = frames["field_week_context"].rename(
            columns={"last_observation_date": "week_last_observation_date"}
        )
        frames["incident_weekly_state"]["incident_state"] = "ACTIVE"
        frames["weekly_components"]["exposure_id"] = "exposure-1"
        frames["component_membership"]["field_id"] = "f1"
        frames["incident_lineage"] = pd.DataFrame(
            columns=["parent_exposure_id", "child_exposure_id", "lineage_type"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in frames.items():
                frame.to_parquet(root / f"{name}.parquet", index=False)
            pd.DataFrame(columns=["incident_id"]).to_parquet(
                root / "completed_incident_features.parquet", index=False
            )
            result = validate_final_artifact_directory(root)
            windows = pd.read_parquet(root / "incident_windows.parquet")
            windows.loc[0, "peak_week"] = "2026-01-12"
            windows.to_parquet(root / "incident_windows.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "do not reconcile"):
                validate_final_artifact_directory(root)
        self.assertTrue(result["passed"])
        self.assertEqual(result["row_counts"]["incident_stage_summary"], 1)

    def test_staged_directory_requires_stage_summary_artifact(self) -> None:
        frames = _frames()
        frames["field_week_context"] = frames["field_week_context"].rename(
            columns={"last_observation_date": "week_last_observation_date"}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in frames.items():
                if name != "incident_stage_summary":
                    frame.to_parquet(root / f"{name}.parquet", index=False)
            pd.DataFrame(columns=["incident_id"]).to_parquet(
                root / "completed_incident_features.parquet", index=False
            )
            with self.assertRaisesRegex(
                ValueError, "incident_stage_summary.parquet"
            ):
                validate_final_artifact_directory(root)


def _write_append_release(
    root: Path,
    *,
    component_id: str = "component-1",
    exposure_id: str = "exposure-1",
    incident_id: str = "incident-1",
    pressure_core_field_count: int = 2,
    incident_state: str = "ACTIVE",
    data_censored_at_boundary: bool = False,
    drop_incident_column: str | None = None,
    include_historical_extra: bool = False,
    include_future: bool = False,
    reverse_cells: bool = False,
    membership_event_state: str = "ACTIVE",
    window_right_censored: bool | None = None,
    window_observed_week_count: int | None = None,
    stage_monitored_count: int = 2,
    include_historical_stage: bool = False,
    lifecycle_data_gap_count: int | None = None,
    lifecycle_coverage_gap_streak: int | None = None,
    include_predecessor: bool = False,
) -> None:
    root.mkdir()
    cells = '["g:1:0","g:0:0"]' if reverse_cells else '["g:0:0","g:1:0"]'
    components = [
        {
            "timeline_bucket": "2026-01-05",
            "hazard_family": "heat",
            "cell_ids_json": cells,
            "component_id": component_id,
        }
    ]
    assignments = [
        {
            "timeline_bucket": "2026-01-05",
            "component_id": component_id,
            "exposure_id": exposure_id,
        }
    ]
    incidents = [
        {
            "timeline_bucket": "2026-01-05",
            "incident_id": incident_id,
            "exposure_id": exposure_id,
            "crop_name": "maize",
            "hazard_family": "heat",
            "component_id": component_id,
            "footprint_cell_ids_json": cells,
            "pressure_cell_ids_json": cells,
            "impact_cell_ids_json": "[]",
            "watch_cell_ids_json": "[]",
            "pressure_core_field_count": pressure_core_field_count,
            "incident_state": incident_state,
            "data_censored_at_boundary": data_censored_at_boundary,
        }
    ]
    is_terminal = incident_state in {
        "CLOSED_CANDIDATE_EXPIRED", "CLOSED_RECOVERED",
        "CLOSED_PRESSURE_QUIET_UNCONFIRMED", "CLOSED_RESPONSE_UNRESOLVED",
        "CLOSED_SEASON_CENSORED", "CLOSED_DATA_CENSORED", "MERGED_INTO",
    }
    closed_week = "2026-01-05" if is_terminal else None
    recovered_week = "2026-01-05" if incident_state == "CLOSED_RECOVERED" else None
    inferred_right_censored = not is_terminal
    first_evidence_week = "2025-12-29" if include_predecessor else "2026-01-05"
    defaults: dict[str, object] = {
        "base_incident_id": incident_id,
        "segment_index": 0,
        "knowledge_time": "2026-01-11",
        "knowledge_time_inferred": False,
        "stage_distribution": "{}",
        "footprint_carried_forward": False,
        "cell_coverage_adequate": True,
        "is_physical_movement": False,
        "coverage_adequate": True,
        "season_boundary_observed": False,
        "current_state": incident_state,
        "first_evidence_week": first_evidence_week,
        "confirmed_week": None,
        "pressure_off_week": None,
        "recovered_week": recovered_week,
        "closed_week": closed_week,
        "merged_into_incident_id": None,
        "right_censored": inferred_right_censored,
        "data_gap_count": (
            lifecycle_data_gap_count
            if lifecycle_data_gap_count is not None
            else (1 if data_censored_at_boundary else 0)
        ),
        "coverage_gap_streak": (
            lifecycle_coverage_gap_streak
            if lifecycle_coverage_gap_streak is not None
            else (1 if data_censored_at_boundary else 0)
        ),
        "data_censored_at_boundary": data_censored_at_boundary,
    }
    for column in (
        *APPEND_STABLE_WEEKLY_CONTENT_COLUMNS,
        *APPEND_STABLE_LIFECYCLE_COLUMNS,
    ):
        incidents[0].setdefault(column, defaults.get(column, 0))
    base_incident = dict(incidents[0])
    if include_predecessor:
        components.insert(
            0,
            {
                "timeline_bucket": "2025-12-29",
                "hazard_family": "heat",
                "cell_ids_json": cells,
                "component_id": "component-0",
            },
        )
        assignments.insert(
            0,
            {
                "timeline_bucket": "2025-12-29",
                "component_id": "component-0",
                "exposure_id": exposure_id,
            },
        )
        incidents.insert(
            0,
            {
                **base_incident,
                "timeline_bucket": "2025-12-29",
                "component_id": "component-0",
                "incident_state": "ACTIVE",
                "current_state": "ACTIVE",
                "closed_week": None,
                "recovered_week": None,
                "right_censored": True,
                "data_censored_at_boundary": False,
                "data_gap_count": 0,
                "coverage_gap_streak": 0,
            },
        )
    if include_historical_extra:
        components.append(
            {
                "timeline_bucket": "2026-01-05",
                "hazard_family": "heat",
                "cell_ids_json": '["g:9:9"]',
                "component_id": "component-backfill",
            }
        )
        assignments.append(
            {
                "timeline_bucket": "2026-01-05",
                "component_id": "component-backfill",
                "exposure_id": "exposure-backfill",
            }
        )
        backfill = {
            **base_incident,
            "incident_id": "incident-backfill",
            "base_incident_id": "incident-backfill",
            "exposure_id": "exposure-backfill",
            "component_id": "component-backfill",
            "footprint_cell_ids_json": '["g:9:9"]',
            "pressure_cell_ids_json": '["g:9:9"]',
        }
        incidents.append(backfill)
    if include_future:
        components.append(
            {
                "timeline_bucket": "2026-01-12",
                "hazard_family": "heat",
                "cell_ids_json": '["g:1:0"]',
                "component_id": "component-2",
            }
        )
        assignments.append(
            {
                "timeline_bucket": "2026-01-12",
                "component_id": "component-2",
                "exposure_id": exposure_id,
            }
        )
        incidents.append(
            {
                **base_incident,
                "timeline_bucket": "2026-01-12",
                "component_id": "component-2",
                "footprint_cell_ids_json": '["g:1:0"]',
                "pressure_cell_ids_json": '["g:1:0"]',
            }
        )
    incident_frame = pd.DataFrame(incidents)
    if drop_incident_column:
        incident_frame = incident_frame.drop(columns=[drop_incident_column])
    memberships = [
        {
            "timeline_bucket": "2026-01-05",
            "incident_id": incident_id,
            "exposure_id": exposure_id,
            "component_id": component_id,
            "crop_name_normalized": "maize",
            "hazard_family": "heat",
            "field_id": "field-1",
            "crop_instance_id": "crop-1",
            "episode_id": "episode-1",
            "membership_role": "pressure_core",
            "event_state": membership_event_state,
            "response_class": "no_material_change",
            "fresh_response_evidence": False,
            "evaluable": True,
            "is_data_gap": False,
            "stage_bucket": "vegetative",
            "grid_id": "g:0:0",
            "knowledge_time": "2026-01-11",
        }
    ]
    if include_predecessor:
        memberships.insert(
            0,
            {
                **memberships[0],
                "timeline_bucket": "2025-12-29",
                "component_id": "component-0",
                "knowledge_time": "2026-01-04",
            },
        )
    if include_future:
        memberships.append(
            {
                **memberships[0],
                "timeline_bucket": "2026-01-12",
                "component_id": "component-2",
                "knowledge_time": "2026-01-18",
            }
        )
    lineage = pd.DataFrame(
        columns=[
            "lineage_id", "timeline_bucket", "lineage_type",
            "crop_name_normalized", "parent_exposure_id", "child_exposure_id",
            "parent_incident_id", "child_incident_id",
            "previous_component_id", "current_component_id", "score",
            "schema_version",
        ]
    )
    windows = pd.DataFrame(
        [
            {
                "incident_id": incident_id,
                "exposure_id": exposure_id,
                "crop_name": "maize",
                "hazard_family": "heat",
                "first_evidence_week": first_evidence_week,
                "confirmed_week": None,
                "pressure_off_week": None,
                "recovered_week": recovered_week,
                "closed_week": closed_week,
                "merged_into_incident_id": None,
                "terminal_state": incident_state,
                "right_censored": (
                    inferred_right_censored
                    if window_right_censored is None
                    else window_right_censored
                ),
                "observed_week_count": (
                    window_observed_week_count
                    if window_observed_week_count is not None
                    else (
                        1 + int(include_predecessor) + int(include_future)
                    )
                ),
                "active_component_week_count": (
                    1 + int(include_predecessor) + int(include_future)
                ),
                "peak_week": first_evidence_week,
                "peak_affected_field_count": pressure_core_field_count,
                "relapse_count": 0,
                "data_gap_count": (
                    lifecycle_data_gap_count
                    if lifecycle_data_gap_count is not None
                    else (1 if data_censored_at_boundary else 0)
                ),
                "split_count": 0,
                "merge_count": 0,
                "outcome_evidence": "monitoring_signals_only_no_crop_death_inference",
            }
        ]
    )
    stage_summaries = [
        _stage_summary_row(
            incident_id=incident_id,
            exposure_id=exposure_id,
            monitored_count=stage_monitored_count,
        )
    ]
    if include_predecessor:
        stage_summaries.insert(
            0,
            _stage_summary_row(
                week="2025-12-29",
                incident_id=incident_id,
                exposure_id=exposure_id,
                monitored_count=stage_monitored_count,
            ),
        )
    if include_historical_stage:
        stage_summaries.append(
            _stage_summary_row(
                incident_id=incident_id,
                exposure_id=exposure_id,
                stage_bucket="flowering",
                monitored_count=stage_monitored_count,
            )
        )
    if include_future:
        stage_summaries.append(
            _stage_summary_row(
                week="2026-01-12",
                incident_id=incident_id,
                exposure_id=exposure_id,
                monitored_count=stage_monitored_count,
            )
        )
    frames = {
        "weekly_components.parquet": pd.DataFrame(components),
        "exposure_component_assignments.parquet": pd.DataFrame(assignments),
        "incident_weekly_state.parquet": incident_frame,
        "incident_stage_summary.parquet": pd.DataFrame(stage_summaries),
        "incident_membership.parquet": pd.DataFrame(memberships),
        "incident_lineage.parquet": lineage,
        "incident_windows.parquet": windows,
    }
    for name, frame in frames.items():
        frame.to_parquet(root / name, index=False)
    artifacts = artifact_hashes(root, frames)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "generation_id": f"generation-{root.name}",
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
