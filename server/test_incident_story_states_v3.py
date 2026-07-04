from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_story_states_v3 import (
    build_crop_story_artifacts,
    build_crop_story_scaffold,
    build_incident_followup_evidence,
    finalize_crop_story_artifacts,
)


class IncidentStoryStatesV3Tests(unittest.TestCase):
    def test_crop_story_confirms_then_closes_without_rewriting_stage_identity(self) -> None:
        weeks = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"])
        exposure_id = "exposure_abc"
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[index], "hazard_family": "heat",
                    "component_id": f"component-{index}", "exposure_id": exposure_id,
                }
                for index in range(2)
            ]
        )
        exposure = pd.DataFrame(
            [
                {
                    **row,
                    "cell_ids_json": json.dumps(["g:1:1"]),
                    "center_lon": 30.0, "center_lat": -2.0,
                    "footprint_area_km2": 25.0,
                }
                for row in assignments.to_dict("records")
            ]
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[index], "hazard_family": "heat",
                    "component_id": f"component-{index}", "field_id": "f1",
                    "crop_instance_id": "crop-1", "episode_id": "episode-1",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": "no_material_change", "fresh_response_evidence": False,
                    "evaluable": True, "is_data_gap": False,
                    "stage_bucket": "vegetative" if index == 0 else "flowering",
                    "crop_name": "Maize", "grid_id": "g:1:1",
                }
                for index in range(2)
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": 1, "grid_y": 1, "monitored_field_count": 30,
                    "evaluable_field_count": 30,
                    "passes_coverage_gate": True,
                }
                for week in weeks
            ]
        )
        config = {
            "policy_version": "test", "minimum_evaluable_fields": 5,
            "confirmation_observed_weeks": 2, "quiet_observed_weeks": 2,
        }
        result = build_crop_story_artifacts(
            exposure, assignments, memberships, cells, config
        )
        self.assertEqual(result.catalog["incident_id"].nunique(), 1)
        self.assertEqual(result.weekly_state["incident_id"].nunique(), 1)
        states = result.weekly_state["incident_state"].tolist()
        self.assertEqual(states[:2], ["CANDIDATE", "CONFIRMED"])
        self.assertEqual(states[-1], "CLOSED_PRESSURE_QUIET_UNCONFIRMED")
        self.assertEqual(result.windows.iloc[0]["terminal_state"], states[-1])
        self.assertFalse(bool(result.windows.iloc[0]["right_censored"]))
        self.assertEqual(
            json.loads(result.weekly_state.iloc[0]["stage_distribution"]),
            {"vegetative": 1.0},
        )
        self.assertEqual(
            json.loads(result.weekly_state.iloc[1]["stage_distribution"]),
            {"flowering": 1.0},
        )

    def test_low_coverage_freezes_quiet_clock(self) -> None:
        weeks = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
        assignments = pd.DataFrame(
            [{"timeline_bucket": weeks[0], "hazard_family": "heat", "component_id": "c1", "exposure_id": "exposure_x"}]
        )
        exposure = assignments.assign(
            cell_ids_json=json.dumps(["g:1:1"]), center_lon=30.0,
            center_lat=-2.0, footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [{
                "timeline_bucket": weeks[0], "hazard_family": "heat", "component_id": "c1",
                "field_id": "f", "crop_instance_id": "crop", "episode_id": "episode",
                "membership_role": "pressure_core", "event_state": "SEVERE",
                "response_class": "severe_decline", "fresh_response_evidence": True,
                "evaluable": True, "is_data_gap": False, "stage_bucket": "flowering",
                "crop_name": "beans", "grid_id": "g:1:1",
            }]
        )
        cells = pd.DataFrame(
            [
                {"timeline_bucket": weeks[0], "hazard_family": "heat", "grid_x": 1, "grid_y": 1, "evaluable_field_count": 30, "monitored_field_count": 30, "passes_coverage_gate": True},
                {"timeline_bucket": weeks[1], "hazard_family": "heat", "grid_x": 1, "grid_y": 1, "evaluable_field_count": 0, "monitored_field_count": 30, "passes_coverage_gate": False},
                {"timeline_bucket": weeks[2], "hazard_family": "heat", "grid_x": 1, "grid_y": 1, "evaluable_field_count": 30, "monitored_field_count": 30, "passes_coverage_gate": True},
            ]
        )
        result = build_crop_story_artifacts(
            exposure, assignments, memberships, cells,
            {
                "policy_version": "test", "minimum_evaluable_fields": 5,
                "severe_confirmation_min_fields": 1,
                "severe_confirmation_min_fresh_response_fields": 1,
            },
        )
        self.assertEqual(result.weekly_state.iloc[1]["incident_state"], "CONFIRMED")
        self.assertEqual(result.weekly_state.iloc[1]["data_gap_count"], 1)

    def test_post_pressure_episode_recovery_closes_the_same_story(self) -> None:
        weeks = pd.to_datetime(
            ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"]
        )
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "exposure_id": "exposure_recovery",
                }
                for index, week in enumerate(weeks[:2])
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0, center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "component_id": f"component-{index}", "field_id": "f1",
                    "crop_instance_id": "crop-1", "episode_id": "episode-1",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": (
                        "medium_decline" if index == 1 else "no_material_change"
                    ),
                    "fresh_response_evidence": index == 1,
                    "evaluable": True, "is_data_gap": False,
                    "stage_bucket": "flowering", "crop_name": "maize",
                    "grid_id": "g:1:1",
                }
                for index, week in enumerate(weeks[:2])
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": 1, "grid_y": 1,
                    "monitored_field_count": 10, "evaluable_field_count": 10,
                    "passes_coverage_gate": True,
                }
                for week in weeks
            ]
        )
        config = {
            "policy_version": "test", "minimum_evaluable_fields": 1,
            "confirmation_observed_weeks": 2, "quiet_observed_weeks": 2,
            "recovery_observed_weeks": 1,
        }
        scaffold = build_crop_story_scaffold(
            exposure, assignments, memberships, cells, config
        )
        incident_id = str(scaffold.catalog.iloc[0]["incident_id"])
        summary = _coverage_summary(incident_id, weeks)
        followup = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[-1], "incident_id": incident_id,
                    "field_id": "f1", "crop_instance_id": "crop-1",
                    "episode_id": "episode-1", "hazard_family": "heat",
                    "event_state": "RECOVERING", "response_class": "recovery",
                    "stage_bucket": "flowering",
                    "knowledge_time": weeks[-1],
                    "fresh_decline_evidence": False,
                    "fresh_recovery_evidence": True,
                }
            ]
        )
        future_lineage = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[-1],
                    "parent_incident_id": incident_id,
                    "child_incident_id": "future-split-child",
                    "lineage_type": "split",
                }
            ]
        )
        result = finalize_crop_story_artifacts(
            scaffold,
            summary,
            config,
            followup_evidence=followup,
            incident_lineage=future_lineage,
        )
        self.assertEqual(
            result.weekly_state["incident_state"].tolist(),
            ["CANDIDATE", "CONFIRMED", "PRESSURE_QUIET", "CLOSED_RECOVERED"],
        )
        recovered = result.memberships[
            (result.memberships["timeline_bucket"] == weeks[-1])
            & (result.memberships["field_id"] == "f1")
        ]
        self.assertEqual(recovered.iloc[0]["membership_role"], "recovered")
        self.assertEqual(result.windows.iloc[0]["terminal_state"], "CLOSED_RECOVERED")
        self.assertEqual(result.weekly_state["split_count"].tolist(), [0, 0, 0, 1])

    def test_other_crops_cannot_advance_a_crop_story_clock(self) -> None:
        weeks = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-maize",
                    "exposure_id": "exposure_crop_coverage",
                }
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0, center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-maize", "field_id": "maize-field",
                    "crop_instance_id": "maize-crop", "episode_id": "episode-maize",
                    "membership_role": "pressure_core", "event_state": "SEVERE",
                    "response_class": "severe_decline",
                    "fresh_response_evidence": True, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "flowering",
                    "crop_name": "maize", "grid_id": "g:1:1",
                }
            ]
        )
        # These all-crop counts stay healthy, but the explicit maize denominator
        # below is absent in week two.
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": 1, "grid_y": 1,
                    "monitored_field_count": 30, "evaluable_field_count": 30,
                    "passes_coverage_gate": True,
                }
                for week in weeks
            ]
        )
        config = {
            "policy_version": "test", "minimum_evaluable_fields": 1,
            "severe_confirmation_min_fields": 1,
            "severe_confirmation_min_fresh_response_fields": 1,
        }
        scaffold = build_crop_story_scaffold(
            exposure, assignments, memberships, cells, config
        )
        incident_id = str(scaffold.catalog.iloc[0]["incident_id"])
        summary = _coverage_summary(incident_id, weeks)
        summary.loc[summary["timeline_bucket"] == weeks[1], [
            "monitored_field_count", "evaluable_field_count",
            "monitored_crop_instance_count", "evaluable_crop_instance_count",
        ]] = 0
        summary.loc[
            summary["timeline_bucket"] == weeks[1], "coverage_missing_cell_count"
        ] = 1
        summary.loc[
            summary["timeline_bucket"] == weeks[2], "stage_bucket"
        ] = "off_season"
        result = finalize_crop_story_artifacts(
            scaffold, summary, config, weekly_cells=cells
        )
        self.assertEqual(result.weekly_state.iloc[0]["incident_state"], "CONFIRMED")
        self.assertEqual(result.weekly_state.iloc[1]["incident_state"], "CONFIRMED")
        self.assertFalse(bool(result.weekly_state.iloc[1]["coverage_adequate"]))
        self.assertEqual(result.weekly_state.iloc[1]["data_gap_count"], 1)
        self.assertEqual(
            result.weekly_state.iloc[-1]["incident_state"],
            "CLOSED_SEASON_CENSORED",
        )

        censored_summary = _coverage_summary(incident_id, weeks)
        for week in weeks[1:]:
            mask = censored_summary["timeline_bucket"] == week
            censored_summary.loc[mask, [
                "monitored_field_count", "evaluable_field_count",
                "monitored_crop_instance_count", "evaluable_crop_instance_count",
            ]] = 0
            censored_summary.loc[mask, "coverage_missing_cell_count"] = 1
        censored = finalize_crop_story_artifacts(
            scaffold,
            censored_summary,
            {**config, "maximum_data_gap_weeks": 2},
        )
        self.assertEqual(
            censored.weekly_state.iloc[-1]["incident_state"],
            "CLOSED_DATA_CENSORED",
        )

    def test_followup_is_joined_by_episode_not_unrelated_field_signal(self) -> None:
        scaffold = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "incident_id": "incident-1",
                }
                for week in pd.to_datetime(["2026-01-05", "2026-01-12"])
            ]
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-05", "incident_id": "incident-1",
                    "field_id": "field-1", "crop_instance_id": "crop-1",
                    "episode_id": "episode-1", "hazard_family": "heat",
                }
            ]
        )
        lanes = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-12", "event_id": "episode-1",
                    "snapshot_as_of_date": "2026-01-14",
                    "field_id": "field-1", "crop_instance_id": "crop-1",
                    "hazard_family": "heat", "event_state": "RECOVERING",
                    "signal_response_class": "recovery",
                    "fresh_response_evidence": True,
                },
                {
                    "timeline_bucket": "2026-01-12", "event_id": "unrelated-episode",
                    "snapshot_as_of_date": "2026-01-14",
                    "field_id": "field-1", "crop_instance_id": "crop-1",
                    "hazard_family": "heat", "event_state": "SEVERE",
                    "signal_response_class": "severe_decline",
                    "fresh_response_evidence": True,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lanes.parquet"
            lanes.to_parquet(path, index=False)
            followup = build_incident_followup_evidence(
                path, scaffold, memberships, threads=1
            )
        self.assertEqual(len(followup), 1)
        self.assertEqual(followup.iloc[0]["episode_id"], "episode-1")
        self.assertTrue(bool(followup.iloc[0]["fresh_recovery_evidence"]))
        self.assertFalse(bool(followup.iloc[0]["fresh_decline_evidence"]))

    def test_carried_unresolved_impact_cell_remains_in_exact_footprint(self) -> None:
        weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-a", "exposure_id": "exposure_carry",
                },
                {
                    "timeline_bucket": weeks[1], "hazard_family": "heat",
                    "component_id": "component-b", "exposure_id": "exposure_carry",
                },
            ]
        )
        exposure = assignments.assign(
            cell_ids_json=['["g:1:1"]', '["g:2:1"]'],
            center_lon=[30.0, 30.1], center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-a", "field_id": "field-a",
                    "crop_instance_id": "crop-a", "episode_id": "episode-a",
                    "membership_role": "pressure_core", "event_state": "SEVERE",
                    "response_class": "severe_decline",
                    "fresh_response_evidence": True, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "flowering",
                    "crop_name": "maize", "grid_id": "g:1:1",
                },
                {
                    "timeline_bucket": weeks[1], "hazard_family": "heat",
                    "component_id": "component-b", "field_id": "field-b",
                    "crop_instance_id": "crop-b", "episode_id": "episode-b",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "flowering",
                    "crop_name": "maize", "grid_id": "g:2:1",
                },
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": x, "grid_y": 1, "monitored_field_count": 10,
                    "evaluable_field_count": 10, "passes_coverage_gate": True,
                }
                for week in weeks for x in (1, 2)
            ]
        )
        scaffold = build_crop_story_scaffold(
            exposure, assignments, memberships, cells,
            {"policy_version": "test", "minimum_evaluable_fields": 1},
        )
        incident_id = str(scaffold.catalog.iloc[0]["incident_id"])
        result = finalize_crop_story_artifacts(
            scaffold,
            _coverage_summary(incident_id, weeks),
            {"policy_version": "test", "minimum_evaluable_fields": 1},
        )
        second = result.weekly_state.iloc[1]
        self.assertEqual(
            set(json.loads(second["pressure_cell_ids_json"])), {"g:2:1"}
        )
        self.assertEqual(
            set(json.loads(second["impact_cell_ids_json"])), {"g:1:1"}
        )
        self.assertEqual(
            set(json.loads(second["footprint_cell_ids_json"])),
            {"g:1:1", "g:2:1"},
        )
        carried = result.memberships[
            (result.memberships["timeline_bucket"] == weeks[1])
            & (result.memberships["field_id"] == "field-a")
        ]
        self.assertEqual(carried.iloc[0]["membership_role"], "unresolved")

    def test_unobservable_carried_impact_cell_freezes_lifecycle_coverage(self) -> None:
        weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-a", "exposure_id": "exposure_coverage",
                },
                {
                    "timeline_bucket": weeks[1], "hazard_family": "heat",
                    "component_id": "component-b", "exposure_id": "exposure_coverage",
                },
            ]
        )
        exposure = assignments.assign(
            cell_ids_json=['["g:1:1"]', '["g:2:1"]'],
            center_lon=[30.0, 30.1], center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-a", "field_id": "field-a",
                    "crop_instance_id": "crop-a", "episode_id": "episode-a",
                    "membership_role": "pressure_core", "event_state": "SEVERE",
                    "response_class": "severe_decline",
                    "fresh_response_evidence": True, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "flowering",
                    "crop_name": "maize", "grid_id": "g:1:1",
                },
                {
                    "timeline_bucket": weeks[1], "hazard_family": "heat",
                    "component_id": "component-b", "field_id": "field-b",
                    "crop_instance_id": "crop-b", "episode_id": "episode-b",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "flowering",
                    "crop_name": "maize", "grid_id": "g:2:1",
                },
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": x, "grid_y": 1, "monitored_field_count": 10,
                    "evaluable_field_count": 10,
                    "passes_coverage_gate": not (week == weeks[1] and x == 1),
                }
                for week in weeks for x in (1, 2)
            ]
        )
        config = {
            "policy_version": "test", "minimum_evaluable_fields": 1,
            "severe_confirmation_min_fields": 1,
            "severe_confirmation_min_fresh_response_fields": 1,
        }
        scaffold = build_crop_story_scaffold(
            exposure, assignments, memberships, cells, config
        )
        incident_id = str(scaffold.catalog.iloc[0]["incident_id"])
        summary = _coverage_summary(incident_id, weeks)
        result = finalize_crop_story_artifacts(
            scaffold, summary, config, weekly_cells=cells
        )
        second = result.weekly_state.iloc[1]
        self.assertFalse(bool(second["coverage_adequate"]))
        self.assertEqual(second["incident_state"], "CONFIRMED")
        self.assertEqual(second["data_gap_count"], 1)

    def test_crop_reappearing_after_terminal_close_starts_new_incident_segment(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=5, freq="7D")
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[index],
                    "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "exposure_id": "exposure_recurrence",
                }
                for index in (0, 1, 4)
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0,
            center_lat=-2.0, footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[index],
                    "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "episode_id": f"episode-{index}",
                    "membership_role": "pressure_core",
                    "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False,
                    "evaluable": True,
                    "is_data_gap": False,
                    "stage_bucket": "vegetative",
                    "crop_name": "maize",
                    "grid_id": "g:1:1",
                }
                for index in (0, 1, 4)
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "grid_x": 1, "grid_y": 1, "monitored_field_count": 10,
                    "evaluable_field_count": 10, "passes_coverage_gate": True,
                }
                for week in weeks
            ]
        )
        result = build_crop_story_artifacts(
            exposure,
            assignments,
            memberships,
            cells,
            {
                "policy_version": "test",
                "minimum_evaluable_fields": 1,
                "confirmation_observed_weeks": 2,
                "quiet_observed_weeks": 2,
            },
        )
        self.assertEqual(len(result.catalog), 2)
        first_id, second_id = result.catalog.sort_values("segment_index")[
            "incident_id"
        ].tolist()
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_id, result.catalog.iloc[0]["base_incident_id"])
        first = result.weekly_state[result.weekly_state["incident_id"] == first_id]
        second = result.weekly_state[result.weekly_state["incident_id"] == second_id]
        self.assertEqual(first["timeline_bucket"].tolist(), list(weeks[:4]))
        self.assertEqual(first.iloc[-1]["incident_state"], "CLOSED_PRESSURE_QUIET_UNCONFIRMED")
        self.assertEqual(second["timeline_bucket"].tolist(), [weeks[4]])
        self.assertEqual(second.iloc[0]["incident_state"], "CANDIDATE")
        self.assertEqual(
            result.memberships.loc[
                result.memberships["timeline_bucket"] == weeks[4], "incident_id"
            ].tolist(),
            [second_id],
        )
        self.assertEqual(set(result.windows["incident_id"]), {first_id, second_id})

    def test_carried_one_off_impact_cannot_self_confirm(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=2, freq="7D")
        assignments = pd.DataFrame(
            [{
                "timeline_bucket": weeks[0], "hazard_family": "heat",
                "component_id": "component-impact", "exposure_id": "exposure_impact",
            }]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0,
            center_lat=-2.0, footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [{
                "timeline_bucket": weeks[0], "hazard_family": "heat",
                "component_id": "component-impact", "field_id": "field-impact",
                "crop_instance_id": "crop-impact", "episode_id": "episode-impact",
                "membership_role": "impact_lag", "event_state": "RECOVERING",
                "response_class": "medium_decline", "fresh_response_evidence": True,
                "evaluable": True, "is_data_gap": False,
                "stage_bucket": "flowering", "crop_name": "maize",
                "grid_id": "g:1:1",
            }]
        )
        cells = pd.DataFrame(
            [{
                "timeline_bucket": week, "hazard_family": "heat",
                "grid_x": 1, "grid_y": 1, "monitored_field_count": 10,
                "evaluable_field_count": 10, "passes_coverage_gate": True,
            } for week in weeks]
        )
        result = build_crop_story_artifacts(
            exposure, assignments, memberships, cells,
            {
                "policy_version": "test", "minimum_evaluable_fields": 1,
                "confirmation_observed_weeks": 2,
            },
        )
        self.assertEqual(
            result.weekly_state["incident_state"].tolist(),
            ["CANDIDATE", "CANDIDATE"],
        )
        self.assertTrue(result.weekly_state["confirmed_week"].isna().all())

    def test_one_off_unconfirmed_impact_expires_in_bounded_observed_time(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=5, freq="7D")
        assignments = pd.DataFrame(
            [{
                "timeline_bucket": weeks[0], "hazard_family": "heat",
                "component_id": "component-impact", "exposure_id": "exposure_expiry",
            }]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0,
            center_lat=-2.0, footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [{
                "timeline_bucket": weeks[0], "hazard_family": "heat",
                "component_id": "component-impact", "field_id": "field-impact",
                "crop_instance_id": "crop-impact", "episode_id": "episode-impact",
                "membership_role": "impact_lag", "event_state": "RECOVERING",
                "response_class": "medium_decline", "fresh_response_evidence": True,
                "evaluable": True, "is_data_gap": False,
                "stage_bucket": "flowering", "crop_name": "maize",
                "grid_id": "g:1:1",
            }]
        )
        cells = pd.DataFrame(
            [{
                "timeline_bucket": week, "hazard_family": "heat",
                "grid_x": 1, "grid_y": 1, "monitored_field_count": 10,
                "evaluable_field_count": 10, "passes_coverage_gate": True,
            } for week in weeks]
        )
        result = build_crop_story_artifacts(
            exposure, assignments, memberships, cells,
            {
                "policy_version": "test", "minimum_evaluable_fields": 1,
                "confirmation_observed_weeks": 2,
                "candidate_expiry_observed_weeks": 2,
            },
        )
        self.assertEqual(
            result.weekly_state["incident_state"].tolist(),
            ["CANDIDATE", "CANDIDATE", "CLOSED_CANDIDATE_EXPIRED"],
        )
        self.assertFalse(bool(result.windows.iloc[0]["right_censored"]))

    def test_merge_transfers_prior_unresolved_before_child_recovery(self) -> None:
        fixture = _lineage_fixture("merge")
        parent_id = fixture["parent_id"]
        child_id = fixture["child_id"]
        recovery = pd.DataFrame(
            [
                {
                    "timeline_bucket": fixture["weeks"][1],
                    "incident_id": parent_id,
                    "field_id": "field-a",
                    "crop_instance_id": "crop-a",
                    "episode_id": "episode-a",
                    "hazard_family": "heat",
                    "event_state": "RECOVERING",
                    "response_class": "recovery",
                    "stage_bucket": "flowering",
                    "knowledge_time": fixture["weeks"][1],
                    "fresh_decline_evidence": False,
                    "fresh_recovery_evidence": True,
                }
            ]
        )
        result = finalize_crop_story_artifacts(
            fixture["scaffold"],
            fixture["summary"],
            fixture["config"],
            followup_evidence=recovery,
            incident_lineage=fixture["lineage"],
            weekly_cells=fixture["cells"],
        )
        parent_edge = result.weekly_state[
            (result.weekly_state["base_incident_id"] == parent_id)
            & (result.weekly_state["timeline_bucket"] == fixture["weeks"][1])
        ].iloc[0]
        child_edge = result.weekly_state[
            (result.weekly_state["base_incident_id"] == child_id)
            & (result.weekly_state["timeline_bucket"] == fixture["weeks"][1])
        ].iloc[0]
        self.assertEqual(parent_edge["incident_state"], "MERGED_INTO")
        self.assertEqual(parent_edge["unresolved_carried_field_count"], 0)
        self.assertEqual(child_edge["fresh_recovery_field_count"], 1)
        self.assertEqual(child_edge["recovered_field_count"], 1)
        self.assertEqual(child_edge["unresolved_carried_field_count"], 1)
        episode_a = result.memberships[
            (result.memberships["timeline_bucket"] == fixture["weeks"][1])
            & (result.memberships["episode_id"] == "episode-a")
        ]
        self.assertEqual(set(episode_a["incident_id"]), {child_id})
        self.assertEqual(set(episode_a["membership_role"]), {"recovered"})

    def test_final_week_merge_cannot_also_be_marked_data_censored(self) -> None:
        fixture = _lineage_fixture("merge")
        lineage = fixture["lineage"].copy()
        lineage["timeline_bucket"] = fixture["weeks"][-1]
        summary = fixture["summary"].copy()
        mask = (
            summary["incident_id"].eq(fixture["parent_id"])
            & summary["timeline_bucket"].eq(fixture["weeks"][-1])
        )
        summary.loc[mask, [
            "monitored_field_count", "evaluable_field_count",
            "monitored_crop_instance_count", "evaluable_crop_instance_count",
        ]] = 0
        summary.loc[mask, "coverage_missing_cell_count"] = 1

        result = finalize_crop_story_artifacts(
            fixture["scaffold"],
            summary,
            {**fixture["config"], "maximum_data_gap_weeks": 1},
            incident_lineage=lineage,
            weekly_cells=fixture["cells"],
        )
        parent = result.weekly_state[
            result.weekly_state["base_incident_id"].eq(fixture["parent_id"])
            & result.weekly_state["timeline_bucket"].eq(fixture["weeks"][-1])
        ].iloc[0]
        self.assertEqual(parent["incident_state"], "MERGED_INTO")
        self.assertFalse(bool(parent["data_censored_at_boundary"]))

    def test_split_transfers_exact_episode_and_keeps_unmatched_with_parent(self) -> None:
        fixture = _lineage_fixture("split")
        result = finalize_crop_story_artifacts(
            fixture["scaffold"],
            fixture["summary"],
            fixture["config"],
            incident_lineage=fixture["lineage"],
            weekly_cells=fixture["cells"],
        )
        edge = result.weekly_state[
            result.weekly_state["timeline_bucket"] == fixture["weeks"][1]
        ].set_index("base_incident_id")
        self.assertEqual(
            edge.loc[fixture["parent_id"], "unresolved_carried_field_count"], 1
        )
        self.assertEqual(
            edge.loc[fixture["child_id"], "unresolved_carried_field_count"], 1
        )
        owners = result.memberships[
            (result.memberships["timeline_bucket"] == fixture["weeks"][1])
            & result.memberships["episode_id"].isin(["episode-a", "episode-b"])
        ].groupby("episode_id")["incident_id"].agg(lambda values: set(values))
        self.assertEqual(owners["episode-a"], {fixture["parent_id"]})
        self.assertEqual(owners["episode-b"], {fixture["child_id"]})

    def test_same_week_impact_onset_is_prefix_stable_when_crop_later_has_core(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=2, freq="7D")
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "exposure_id": "exposure_onset",
                }
                for index, week in enumerate(weeks)
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]', center_lon=30.0,
            center_lat=-2.0, footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-0", "field_id": "beans-core",
                    "crop_instance_id": "beans-crop", "episode_id": "beans-event",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "vegetative",
                    "crop_name": "beans", "grid_id": "g:1:1",
                },
                {
                    "timeline_bucket": weeks[0], "hazard_family": "heat",
                    "component_id": "component-0", "field_id": "maize-watch",
                    "crop_instance_id": "maize-crop", "episode_id": "maize-watch-event",
                    "membership_role": "watch_frontier", "event_state": "WATCH",
                    "response_class": "medium_decline",
                    "fresh_response_evidence": True, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "vegetative",
                    "crop_name": "maize", "grid_id": "g:1:1",
                },
                {
                    "timeline_bucket": weeks[1], "hazard_family": "heat",
                    "component_id": "component-1", "field_id": "maize-core",
                    "crop_instance_id": "maize-crop", "episode_id": "maize-core-event",
                    "membership_role": "pressure_core", "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False, "evaluable": True,
                    "is_data_gap": False, "stage_bucket": "vegetative",
                    "crop_name": "maize", "grid_id": "g:1:1",
                },
            ]
        )
        cells = pd.DataFrame(
            [{
                "timeline_bucket": week, "hazard_family": "heat",
                "grid_x": 1, "grid_y": 1, "monitored_field_count": 10,
                "evaluable_field_count": 10, "passes_coverage_gate": True,
            } for week in weeks]
        )
        config = {"policy_version": "test", "minimum_evaluable_fields": 1}
        # Synthetic membership is promoted to the same explicit impact role
        # that production component construction assigns to fresh decline.
        for frame in (memberships,):
            frame.loc[
                frame["field_id"].eq("maize-watch"), "membership_role"
            ] = "impact_lag"
        full = build_crop_story_artifacts(
            exposure, assignments, memberships, cells, config
        )
        truncated = build_crop_story_artifacts(
            exposure.iloc[:1], assignments.iloc[:1], memberships.iloc[:2],
            cells.iloc[:1], config,
        )
        full_prefix = full.weekly_state[
            (full.weekly_state["crop_name"] == "maize")
            & (full.weekly_state["timeline_bucket"] == weeks[0])
        ]
        truncated_maize = truncated.weekly_state[
            truncated.weekly_state["crop_name"] == "maize"
        ]
        self.assertEqual(len(full_prefix), 1)
        self.assertEqual(len(truncated_maize), 1)
        self.assertEqual(
            full_prefix.iloc[0]["incident_id"],
            truncated_maize.iloc[0]["incident_id"],
        )
        self.assertEqual(
            full_prefix.iloc[0]["incident_state"],
            truncated_maize.iloc[0]["incident_state"],
        )

    def test_low_coverage_impact_prelude_is_not_admitted_as_a_story(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=2, freq="7D")
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": week,
                    "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "exposure_id": "exposure_coverage_onset",
                }
                for index, week in enumerate(weeks)
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]',
            center_lon=30.0,
            center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[index],
                    "hazard_family": "heat",
                    "component_id": f"component-{index}",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "episode_id": f"episode-{index}",
                    "membership_role": "impact_lag" if index == 0 else "pressure_core",
                    "event_state": "RECOVERING" if index == 0 else "ACTIVE",
                    "response_class": "medium_decline" if index == 0 else "no_material_change",
                    "fresh_response_evidence": index == 0,
                    "evaluable": True,
                    "is_data_gap": False,
                    "stage_bucket": "flowering",
                    "crop_name": "maize",
                    "grid_id": "g:1:1",
                }
                for index in range(2)
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week,
                    "hazard_family": "heat",
                    "grid_x": 1,
                    "grid_y": 1,
                    "monitored_field_count": 10,
                    "evaluable_field_count": 10,
                    "passes_coverage_gate": index == 1,
                }
                for index, week in enumerate(weeks)
            ]
        )
        config = {"policy_version": "test", "minimum_evaluable_fields": 1}
        full = build_crop_story_artifacts(
            exposure, assignments, memberships, cells, config
        )
        truncated = build_crop_story_artifacts(
            exposure.iloc[:1],
            assignments.iloc[:1],
            memberships.iloc[:1],
            cells.iloc[:1],
            config,
        )
        self.assertTrue(truncated.catalog.empty)
        self.assertTrue(truncated.weekly_state.empty)
        self.assertEqual(full.weekly_state["timeline_bucket"].tolist(), [weeks[1]])
        self.assertEqual(
            pd.Timestamp(full.weekly_state.iloc[0]["first_evidence_week"]),
            weeks[1],
        )

    def test_scaffold_tail_is_bounded_in_long_monitoring_history(self) -> None:
        weeks = pd.date_range("2026-01-05", periods=100, freq="7D")
        assignments = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0],
                    "hazard_family": "heat",
                    "component_id": "component-0",
                    "exposure_id": "exposure_bounded_tail",
                }
            ]
        )
        exposure = assignments.assign(
            cell_ids_json='["g:1:1"]',
            center_lon=30.0,
            center_lat=-2.0,
            footprint_area_km2=25.0,
        )
        memberships = pd.DataFrame(
            [
                {
                    "timeline_bucket": weeks[0],
                    "hazard_family": "heat",
                    "component_id": "component-0",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "episode_id": "episode-1",
                    "membership_role": "pressure_core",
                    "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False,
                    "evaluable": True,
                    "is_data_gap": False,
                    "stage_bucket": "vegetative",
                    "crop_name": "maize",
                    "grid_id": "g:1:1",
                }
            ]
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": week,
                    "hazard_family": "heat",
                    "grid_x": 1,
                    "grid_y": 1,
                    "monitored_field_count": 10,
                    "evaluable_field_count": 10,
                    "passes_coverage_gate": True,
                }
                for week in weeks
            ]
        )
        scaffold = build_crop_story_scaffold(
            exposure,
            assignments,
            memberships,
            cells,
            {
                "policy_version": "test",
                "candidate_expiry_observed_weeks": 2,
                "quiet_close_weeks": 2,
                "recovery_grace_weeks": 2,
                "maximum_data_gap_weeks": 4,
            },
        )
        self.assertEqual(len(scaffold.weekly_state), 5)
        self.assertEqual(scaffold.weekly_state["timeline_bucket"].max(), weeks[4])


def _coverage_summary(
    incident_id: str, weeks: pd.DatetimeIndex
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timeline_bucket": week, "incident_id": incident_id,
                "stage_bucket": "flowering", "monitored_field_count": 1,
                "evaluable_field_count": 1,
                "monitored_crop_instance_count": 1,
                "evaluable_crop_instance_count": 1,
                "coverage_missing_cell_count": 0,
            }
            for week in weeks
        ]
    )


def _lineage_fixture(kind: str) -> dict[str, object]:
    weeks = pd.date_range("2026-01-05", periods=3, freq="7D")
    assignments = [
        {
            "timeline_bucket": weeks[0],
            "hazard_family": "heat",
            "component_id": "component-parent-0",
            "exposure_id": "exposure_parent",
        },
        {
            "timeline_bucket": weeks[1],
            "hazard_family": "heat",
            "component_id": "component-child-1",
            "exposure_id": "exposure_child",
        },
        {
            "timeline_bucket": weeks[2],
            "hazard_family": "heat",
            "component_id": "component-child-2",
            "exposure_id": "exposure_child",
        },
    ]
    if kind == "split":
        assignments.extend(
            [
                {
                    "timeline_bucket": weeks[1],
                    "hazard_family": "heat",
                    "component_id": "component-parent-1",
                    "exposure_id": "exposure_parent",
                },
                {
                    "timeline_bucket": weeks[2],
                    "hazard_family": "heat",
                    "component_id": "component-parent-2",
                    "exposure_id": "exposure_parent",
                },
            ]
        )
    assignments_frame = pd.DataFrame(assignments)
    exposure = assignments_frame.assign(
        cell_ids_json=assignments_frame["component_id"].map(
            lambda value: (
                '["g:1:1"]' if "parent" in value else '["g:2:1"]'
            )
        ),
        center_lon=30.0,
        center_lat=-2.0,
        footprint_area_km2=25.0,
    )
    membership_records = [
        {
            "timeline_bucket": weeks[0],
            "hazard_family": "heat",
            "component_id": "component-parent-0",
            "field_id": field,
            "crop_instance_id": crop,
            "episode_id": episode,
            "membership_role": "impact_lag",
            "event_state": "RECOVERING",
            "response_class": "medium_decline",
            "fresh_response_evidence": True,
            "evaluable": True,
            "is_data_gap": False,
            "stage_bucket": "flowering",
            "crop_name": "maize",
            "grid_id": grid,
        }
        for field, crop, episode, grid in (
            ("field-a", "crop-a", "episode-a", "g:1:1"),
            ("field-b", "crop-b", "episode-b", "g:2:1"),
        )
    ]
    if kind == "split":
        for index in (1, 2):
            membership_records.extend(
                [
                    {
                        "timeline_bucket": weeks[index],
                        "hazard_family": "heat",
                        "component_id": f"component-parent-{index}",
                        "field_id": "field-a",
                        "crop_instance_id": "crop-a",
                        "episode_id": "episode-a",
                        "membership_role": "pressure_core",
                        "event_state": "ACTIVE",
                        "response_class": "no_material_change",
                        "fresh_response_evidence": False,
                        "evaluable": True,
                        "is_data_gap": False,
                        "stage_bucket": "flowering",
                        "crop_name": "maize",
                        "grid_id": "g:1:1",
                    },
                    {
                        "timeline_bucket": weeks[index],
                        "hazard_family": "heat",
                        "component_id": f"component-child-{index}",
                        "field_id": "field-b",
                        "crop_instance_id": "crop-b",
                        "episode_id": "episode-b",
                        "membership_role": "pressure_core",
                        "event_state": "ACTIVE",
                        "response_class": "no_material_change",
                        "fresh_response_evidence": False,
                        "evaluable": True,
                        "is_data_gap": False,
                        "stage_bucket": "flowering",
                        "crop_name": "maize",
                        "grid_id": "g:2:1",
                    },
                ]
            )
    else:
        for index in (1, 2):
            membership_records.append(
                {
                    "timeline_bucket": weeks[index],
                    "hazard_family": "heat",
                    "component_id": f"component-child-{index}",
                    "field_id": "field-c",
                    "crop_instance_id": "crop-c",
                    "episode_id": "episode-c",
                    "membership_role": "pressure_core",
                    "event_state": "ACTIVE",
                    "response_class": "no_material_change",
                    "fresh_response_evidence": False,
                    "evaluable": True,
                    "is_data_gap": False,
                    "stage_bucket": "flowering",
                    "crop_name": "maize",
                    "grid_id": "g:3:1",
                }
            )
    memberships = pd.DataFrame(membership_records)
    cells = pd.DataFrame(
        [
            {
                "timeline_bucket": week,
                "hazard_family": "heat",
                "grid_x": x,
                "grid_y": 1,
                "monitored_field_count": 10,
                "evaluable_field_count": 10,
                "passes_coverage_gate": True,
            }
            for week in weeks
            for x in (1, 2, 3)
        ]
    )
    config = {
        "policy_version": "test",
        "minimum_evaluable_fields": 1,
        "confirmation_observed_weeks": 2,
    }
    scaffold = build_crop_story_scaffold(
        exposure, assignments_frame, memberships, cells, config
    )
    ids = scaffold.catalog.set_index("exposure_id")["incident_id"].to_dict()
    summary = pd.DataFrame(
        [
            {
                "timeline_bucket": row.timeline_bucket,
                "incident_id": row.incident_id,
                "stage_bucket": "flowering",
                "monitored_field_count": 3,
                "evaluable_field_count": 3,
                "monitored_crop_instance_count": 3,
                "evaluable_crop_instance_count": 3,
                "coverage_missing_cell_count": 0,
            }
            for row in scaffold.weekly_state.itertuples(index=False)
        ]
    )
    lineage = pd.DataFrame(
        [
            {
                "timeline_bucket": weeks[1],
                "parent_incident_id": ids["exposure_parent"],
                "child_incident_id": ids["exposure_child"],
                "lineage_type": kind,
            }
        ]
    )
    return {
        "weeks": weeks,
        "scaffold": scaffold,
        "cells": cells,
        "config": config,
        "summary": summary,
        "lineage": lineage,
        "parent_id": ids["exposure_parent"],
        "child_id": ids["exposure_child"],
    }


if __name__ == "__main__":
    unittest.main()
