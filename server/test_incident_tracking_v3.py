from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from story_monitor.incident_tracking_v3 import (
    advance_incident_lifecycle,
    apply_tracking_assignments,
    assign_metric_grid,
    benjamini_hochberg,
    build_crop_incident_assignments,
    build_weekly_components,
    initialize_incident_lifecycle,
    link_weekly_components,
    mark_significant_cells,
    normal_tail_p_value,
    score_temporal_candidates,
    stable_crop_incident_id,
)


CONFIG = {
    "policy_version": "test-policy-v3",
    "identity_namespace": "test-incidents",
    "cell_size_km": 5.0,
    "minimum_active_fields": 1,
    "minimum_monitored_fields": 1,
    "continuation_threshold": 0.45,
    "lineage_threshold": 0.30,
}


def _cell(
    week: str,
    x_value: int,
    y_value: int,
    *,
    hazard: str = "heat",
    active: int = 2,
    significant: bool = True,
    z_score: float = 4.0,
) -> dict[str, object]:
    return {
        "timeline_bucket": week,
        "hazard_family": hazard,
        "grid_x": x_value,
        "grid_y": y_value,
        "grid_center_x_km": (x_value + 0.5) * 5.0,
        "grid_center_y_km": (y_value + 0.5) * 5.0,
        "grid_center_lon": 30.0 + x_value * 0.05,
        "grid_center_lat": -2.0 + y_value * 0.05,
        "active_field_count": active,
        "monitored_field_count": 20,
        "z_score": z_score,
        "significant": significant,
    }


def _field(
    week: str,
    field_id: str,
    episode_id: str,
    x_value: int,
    y_value: int,
    state: str,
    *,
    hazard: str = "heat",
    stage: str = "vegetative",
    crop: str = "maize",
    impact: bool = False,
    response_class: str = "no_new_event_response",
    fresh_response: bool = False,
) -> dict[str, object]:
    return {
        "timeline_bucket": week,
        "hazard_family": hazard,
        "field_id": field_id,
        "crop_instance_id": f"crop-{field_id}",
        "episode_id": episode_id,
        "grid_x": x_value,
        "grid_y": y_value,
        "event_state": state,
        "stage_family": stage,
        "crop_name": crop,
        "impact_active": impact,
        "response_class": response_class,
        "fresh_response_evidence": fresh_response,
        "evaluable": True,
        "is_data_gap": False,
    }


def _component(
    week: str,
    component_id: str,
    cells: list[str],
    center_x: float,
    *,
    exposure_id: str | None = None,
    hazard: str = "heat",
) -> dict[str, object]:
    result: dict[str, object] = {
        "timeline_bucket": week,
        "hazard_family": hazard,
        "component_id": component_id,
        "cell_ids_json": json.dumps(cells),
        "center_x_km": center_x,
        "center_y_km": 0.0,
    }
    if exposure_id is not None:
        result["exposure_id"] = exposure_id
    return result


def _membership(
    component_id: str,
    field_id: str,
    episode_id: str,
    *,
    stage: str = "vegetative",
) -> dict[str, str]:
    return {
        "component_id": component_id,
        "field_id": field_id,
        "episode_id": episode_id,
        "membership_role": "pressure_core",
        "stage_family": stage,
    }


class IncidentTrackingV3Tests(unittest.TestCase):
    def test_metric_grid_and_ids_are_deterministic_under_reordering(self) -> None:
        rows = pd.DataFrame(
            [
                {"field_id": "b", "centroid_lon": 30.12, "centroid_lat": -1.97},
                {"field_id": "a", "centroid_lon": 30.01, "centroid_lat": -2.01},
            ]
        )
        first = assign_metric_grid(rows, CONFIG).sort_values("field_id").reset_index(drop=True)
        second = assign_metric_grid(rows.iloc[::-1], CONFIG).sort_values("field_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["grid_id"].str.match(r"g:-?\d+:-?\d+").all())

    def test_normal_tail_and_bh_are_stable_and_preserve_missing_values(self) -> None:
        p_values = np.asarray(normal_tail_p_value([4.0, 3.0, 0.0, np.nan]))
        adjusted, rejected = benjamini_hochberg(p_values, alpha=0.05)

        self.assertLess(p_values[0], p_values[1])
        self.assertTrue(rejected[0])
        self.assertTrue(rejected[1])
        self.assertFalse(rejected[2])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_cell_significance_is_group_local_and_order_independent(self) -> None:
        cells = pd.DataFrame(
            [
                _cell("2026-01-05", 0, 0, z_score=4.0),
                _cell("2026-01-05", 1, 0, z_score=0.0),
                _cell("2026-01-12", 0, 0, z_score=3.2),
            ]
        ).drop(columns=["significant"])
        first = mark_significant_cells(cells, CONFIG)
        second = mark_significant_cells(cells.iloc[::-1], CONFIG)
        columns = ["timeline_bucket", "grid_x", "grid_y", "significant", "fdr_q_value"]

        pd.testing.assert_frame_equal(first[columns], second[columns])
        self.assertTrue(bool(first.iloc[0]["significant"]))
        self.assertFalse(bool(first.iloc[1]["significant"]))
        self.assertTrue(bool(first.iloc[2]["significant"]))

    def test_components_use_eight_neighbors_and_watch_is_frontier_only(self) -> None:
        week = "2026-01-05"
        cells = pd.DataFrame(
            [
                _cell(week, 0, 0),
                _cell(week, 1, 1),
                _cell(week, 5, 5, active=1),
            ]
        )
        fields = pd.DataFrame(
            [
                _field(week, "a", "ea", 0, 0, "ACTIVE"),
                _field(
                    week, "b", "eb", 1, 1, "SEVERE",
                    response_class="severe_decline", fresh_response=True,
                ),
                _field(week, "watch", "ew", 2, 2, "WATCH"),
                _field(week, "c", "ec", 5, 5, "ACTIVE"),
                _field(week, "impact", "ei", 4, 4, "RECOVERING", impact=True),
            ]
        )

        result = build_weekly_components(cells, fields, CONFIG)
        reordered = build_weekly_components(
            cells.iloc[::-1].reset_index(drop=True),
            fields.iloc[::-1].reset_index(drop=True),
            CONFIG,
        )

        self.assertEqual(len(result.components), 2)
        pd.testing.assert_frame_equal(result.components, reordered.components)
        pd.testing.assert_frame_equal(result.memberships, reordered.memberships)
        self.assertEqual(sorted(result.components["active_field_count"]), [1, 2])
        roles = dict(zip(result.memberships["field_id"], result.memberships["membership_role"]))
        self.assertEqual(roles["watch"], "watch_frontier")
        self.assertEqual(roles["impact"], "impact_lag")
        self.assertEqual(
            result.memberships.set_index("field_id").loc["a", "crop_instance_id"],
            "crop-a",
        )
        severe_membership = result.memberships.set_index("field_id").loc["b"]
        self.assertEqual(severe_membership["response_class"], "severe_decline")
        self.assertTrue(bool(severe_membership["fresh_response_evidence"]))
        self.assertFalse(result.memberships.duplicated(["timeline_bucket", "hazard_family", "field_id"]).any())

        watch_only = pd.DataFrame([_cell(week, 10, 10, active=0)])
        with self.assertRaisesRegex(ValueError, "no ACTIVE/SEVERE fields"):
            build_weekly_components(
                watch_only,
                pd.DataFrame([_field(week, "only-watch", "ew2", 10, 10, "WATCH")]),
                CONFIG,
            )

    def test_parallel_policy_and_cell_schema_aliases_are_accepted(self) -> None:
        policy = SimpleNamespace(
            version="parallel-policy-v3",
            grid_cell_size_km=5.0,
            grid_origin_lon=0.0,
            grid_origin_lat=0.0,
        )
        cells = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-05",
                    "hazard_family": "heat",
                    "cell_x": 0,
                    "cell_y": 0,
                    "reference_latitude": -2.0,
                    "active_field_count": 1,
                    "evaluable_count": 20,
                    "z_score": 4.0,
                    "is_significant": True,
                }
            ]
        )
        lanes = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-05",
                    "hazard_family": "heat",
                    "field_id": "a",
                    "crop_instance_id": "crop-a",
                    "event_id": "episode-a",
                    "event_state": "ACTIVE",
                    "daily_response_class": "medium_decline",
                    "new_response_evidence": True,
                    "evaluable": True,
                    "is_data_gap_snapshot": False,
                    "stage_bucket": "vegetative",
                    "crop_name": "maize",
                    "centroid_lon": 0.01,
                    "centroid_lat": 0.01,
                }
            ]
        )

        result = build_weekly_components(cells, lanes, policy)

        self.assertEqual(len(result.components), 1)
        self.assertTrue(result.components.iloc[0]["component_id"].startswith("component_"))
        self.assertEqual(result.memberships.iloc[0]["episode_id"], "episode-a")
        self.assertEqual(result.memberships.iloc[0]["crop_instance_id"], "crop-a")
        self.assertEqual(result.memberships.iloc[0]["stage_bucket"], "vegetative")
        self.assertEqual(result.memberships.iloc[0]["response_class"], "medium_decline")
        self.assertTrue(bool(result.memberships.iloc[0]["fresh_response_evidence"]))

    def test_stage_and_crop_change_context_but_not_exposure_identity(self) -> None:
        week = "2026-01-05"
        cells = pd.DataFrame([_cell(week, 0, 0, active=1)])
        early = build_weekly_components(
            cells,
            pd.DataFrame([_field(week, "a", "e", 0, 0, "ACTIVE", stage="vegetative", crop="maize")]),
            CONFIG,
        )
        late = build_weekly_components(
            cells,
            pd.DataFrame([_field(week, "a", "e", 0, 0, "ACTIVE", stage="flowering", crop="beans")]),
            CONFIG,
        )

        self.assertEqual(early.components.iloc[0]["component_id"], late.components.iloc[0]["component_id"])
        self.assertTrue(str(early.components.iloc[0]["component_id"]).startswith("component_"))
        self.assertNotEqual(early.components.iloc[0]["stage_distribution"], late.components.iloc[0]["stage_distribution"])
        self.assertNotEqual(early.components.iloc[0]["crop_distribution"], late.components.iloc[0]["crop_distribution"])

        exposure_id = "exposure_1234567890"
        maize = stable_crop_incident_id(exposure_id, "Maize", CONFIG)
        beans = stable_crop_incident_id(exposure_id, "Beans", CONFIG)
        self.assertTrue(maize.startswith("incident_"))
        self.assertNotEqual(maize, beans)

        exposure_assignments = pd.DataFrame(
            [{"component_id": early.components.iloc[0]["component_id"], "exposure_id": exposure_id}]
        )
        crop_incidents = build_crop_incident_assignments(
            exposure_assignments,
            pd.DataFrame(
                [
                    {"component_id": early.components.iloc[0]["component_id"],
                     "field_id": "a", "episode_id": "e-a", "membership_role": "pressure_core",
                     "crop_name": "Maize", "stage_bucket": "vegetative"},
                    {"component_id": early.components.iloc[0]["component_id"],
                     "field_id": "b", "episode_id": "e-b", "membership_role": "impact_lag",
                     "crop_name": "Beans", "stage_bucket": "flowering"},
                    {"component_id": early.components.iloc[0]["component_id"],
                     "field_id": "c", "episode_id": "e-c", "membership_role": "watch_frontier",
                     "crop_name": "Rice", "stage_bucket": "emergence"},
                ]
            ),
            CONFIG,
        )
        self.assertEqual(set(crop_incidents["crop_name_normalized"]), {"maize", "beans"})
        self.assertTrue(crop_incidents["incident_id"].str.startswith("incident_").all())

    def test_split_retains_parent_on_one_child_and_records_other_child(self) -> None:
        previous = pd.DataFrame(
            [_component("2026-01-05", "p", ["g:0:0", "g:1:0"], 5.0, exposure_id="exposure-parent")]
        )
        current = pd.DataFrame(
            [
                _component("2026-01-12", "c1", ["g:0:0"], 2.5),
                _component("2026-01-12", "c2", ["g:1:0"], 7.5),
            ]
        )
        previous_members = pd.DataFrame(
            [_membership("p", field, f"e-{field}") for field in ("a", "b", "c", "d")]
        )
        current_members = pd.DataFrame(
            [_membership("c1", "a", "e-a"), _membership("c1", "b", "e-b"),
             _membership("c2", "c", "e-c"), _membership("c2", "d", "e-d")]
        )

        scores = score_temporal_candidates(previous, current, previous_members, current_members, CONFIG)
        tracking = link_weekly_components(previous, current, scores, CONFIG)

        self.assertEqual((tracking.assignments["exposure_id"] == "exposure-parent").sum(), 1)
        self.assertIn("split", set(tracking.lineage["lineage_type"]))
        split = tracking.lineage[tracking.lineage["lineage_type"] == "split"].iloc[0]
        self.assertEqual(split["parent_exposure_id"], "exposure-parent")
        self.assertNotEqual(split["child_exposure_id"], "exposure-parent")

    def test_merge_keeps_one_parent_and_marks_the_other_merged(self) -> None:
        previous = pd.DataFrame(
            [
                _component("2026-01-05", "p1", ["g:0:0"], 2.5, exposure_id="exposure-one"),
                _component("2026-01-05", "p2", ["g:1:0"], 7.5, exposure_id="exposure-two"),
            ]
        )
        current = pd.DataFrame([_component("2026-01-12", "c", ["g:0:0", "g:1:0"], 5.0)])
        previous_members = pd.DataFrame(
            [_membership("p1", "a", "e-a"), _membership("p1", "b", "e-b"),
             _membership("p2", "c", "e-c"), _membership("p2", "d", "e-d")]
        )
        current_members = pd.DataFrame(
            [_membership("c", "a", "e-a"), _membership("c", "c", "e-c")]
        )

        scores = score_temporal_candidates(previous, current, previous_members, current_members, CONFIG)
        tracking = link_weekly_components(previous, current, scores, CONFIG)

        self.assertIn(tracking.assignments.iloc[0]["exposure_id"], {"exposure-one", "exposure-two"})
        self.assertIn("merge", set(tracking.lineage["lineage_type"]))
        self.assertEqual((tracking.previous_updates["update_status"] == "merged").sum(), 1)

    def test_secondary_cross_edge_between_two_continuations_is_not_a_merge(self) -> None:
        previous = pd.DataFrame(
            [
                _component("2026-01-05", "p1", ["g:0:0"], 2.5, exposure_id="exposure-one"),
                _component("2026-01-05", "p2", ["g:2:0"], 12.5, exposure_id="exposure-two"),
            ]
        )
        current = pd.DataFrame(
            [
                _component("2026-01-12", "c1", ["g:0:0"], 2.5),
                _component("2026-01-12", "c2", ["g:2:0"], 12.5),
            ]
        )
        scores = pd.DataFrame(
            [
                {"previous_component_id": "p1", "current_component_id": "c1", "score": 0.90},
                {"previous_component_id": "p2", "current_component_id": "c2", "score": 0.85},
                {"previous_component_id": "p1", "current_component_id": "c2", "score": 0.80},
                {"previous_component_id": "p2", "current_component_id": "c1", "score": 0.75},
            ]
        )
        for name in ("episode_jaccard", "cell_jaccard", "field_jaccard"):
            scores[name] = 0.5

        tracking = link_weekly_components(previous, current, scores, CONFIG)

        self.assertEqual(set(tracking.lineage["lineage_type"]), {"related_unmatched"})
        self.assertEqual(set(tracking.previous_updates["update_status"]), {"continued"})
        self.assertNotIn("merge", set(tracking.lineage["lineage_type"]))

    def test_primary_links_maximize_total_weight_and_ignore_input_order(self) -> None:
        previous = pd.DataFrame(
            [
                _component(
                    "2026-01-05", "p1", ["g:0:0"], 2.5,
                    exposure_id="exposure-one",
                ),
                _component(
                    "2026-01-05", "p2", ["g:1:0"], 7.5,
                    exposure_id="exposure-two",
                ),
            ]
        )
        current = pd.DataFrame(
            [
                _component("2026-01-12", "c1", ["g:0:0"], 2.5),
                _component("2026-01-12", "c2", ["g:1:0"], 7.5),
            ]
        )
        # Greedy selection takes p1->c1 (0.90) and strands p2.  The globally
        # optimal one-to-one continuation is p1->c2 plus p2->c1 (1.65).
        scores = pd.DataFrame(
            [
                {"previous_component_id": "p1", "current_component_id": "c1", "score": 0.90},
                {"previous_component_id": "p1", "current_component_id": "c2", "score": 0.80},
                {"previous_component_id": "p2", "current_component_id": "c1", "score": 0.85},
                {"previous_component_id": "p2", "current_component_id": "c2", "score": 0.10},
            ]
        )
        for name in ("episode_jaccard", "cell_jaccard", "field_jaccard"):
            scores[name] = 0.0

        first = link_weekly_components(previous, current, scores, CONFIG)
        reordered = link_weekly_components(
            previous.iloc[::-1].reset_index(drop=True),
            current.iloc[::-1].reset_index(drop=True),
            scores.iloc[::-1].reset_index(drop=True),
            CONFIG,
        )

        assignment = first.assignments.set_index("component_id")
        self.assertEqual(assignment.loc["c1", "exposure_id"], "exposure-two")
        self.assertEqual(assignment.loc["c2", "exposure_id"], "exposure-one")
        self.assertEqual(set(assignment["assignment_kind"]), {"continued"})
        self.assertAlmostEqual(float(assignment["link_score"].sum()), 1.65)
        pd.testing.assert_frame_equal(first.assignments, reordered.assignments)
        pd.testing.assert_frame_equal(first.lineage, reordered.lineage)
        pd.testing.assert_frame_equal(first.previous_updates, reordered.previous_updates)

    def test_primary_link_ties_and_unmatched_options_are_deterministic(self) -> None:
        previous = pd.DataFrame(
            [
                _component(
                    "2026-01-05", component_id, [f"g:{index}:0"], index * 5.0,
                    exposure_id=f"exposure-{index}",
                )
                for index, component_id in enumerate(("p1", "p2", "p3"), start=1)
            ]
        )
        current = pd.DataFrame(
            [
                _component("2026-01-12", "c1", ["g:1:0"], 5.0),
                _component("2026-01-12", "c2", ["g:2:0"], 10.0),
            ]
        )
        scores = pd.DataFrame(
            [
                {
                    "previous_component_id": previous_id,
                    "current_component_id": current_id,
                    "score": 0.80 if previous_id != "p3" else 0.20,
                    "episode_jaccard": 0.0,
                    "cell_jaccard": 0.0,
                    "field_jaccard": 0.0,
                }
                for previous_id in ("p1", "p2", "p3")
                for current_id in ("c1", "c2")
            ]
        )

        first = link_weekly_components(previous, current, scores, CONFIG)
        reordered = link_weekly_components(
            previous.sample(frac=1.0, random_state=7).reset_index(drop=True),
            current.iloc[::-1].reset_index(drop=True),
            scores.sample(frac=1.0, random_state=11).reset_index(drop=True),
            CONFIG,
        )

        self.assertEqual(set(first.assignments["assignment_kind"]), {"continued"})
        unmatched = first.previous_updates[
            first.previous_updates["update_status"] == "unmatched"
        ]
        self.assertEqual(list(unmatched["previous_component_id"]), ["p3"])
        pd.testing.assert_frame_equal(first.assignments, reordered.assignments)
        pd.testing.assert_frame_equal(first.lineage, reordered.lineage)
        pd.testing.assert_frame_equal(first.previous_updates, reordered.previous_updates)

    def test_stage_and_crop_context_do_not_change_candidate_weight_or_link(self) -> None:
        previous = pd.DataFrame(
            [_component(
                "2026-01-05", "p", ["g:0:0"], 2.5,
                exposure_id="exposure-stable",
            )]
        )
        current = pd.DataFrame(
            [_component("2026-01-12", "c", ["g:0:0"], 2.5)]
        )
        previous_members = pd.DataFrame(
            [{**_membership("p", "a", "episode-a", stage="vegetative"), "crop_name": "maize"}]
        )
        same_context = pd.DataFrame(
            [{**_membership("c", "a", "episode-a", stage="vegetative"), "crop_name": "maize"}]
        )
        changed_context = pd.DataFrame(
            [{**_membership("c", "a", "episode-a", stage="flowering"), "crop_name": "beans"}]
        )

        baseline = score_temporal_candidates(
            previous, current, previous_members, same_context, CONFIG
        )
        changed = score_temporal_candidates(
            previous, current, previous_members, changed_context, CONFIG
        )

        self.assertEqual(float(baseline.iloc[0]["stage_cosine"]), 1.0)
        self.assertEqual(float(changed.iloc[0]["stage_cosine"]), 0.0)
        self.assertAlmostEqual(
            float(baseline.iloc[0]["score"]), float(changed.iloc[0]["score"])
        )
        baseline_tracking = link_weekly_components(
            previous, current, baseline, CONFIG
        )
        changed_tracking = link_weekly_components(
            previous, current, changed, CONFIG
        )
        pd.testing.assert_frame_equal(
            baseline_tracking.assignments, changed_tracking.assignments
        )

    def test_stage_change_does_not_break_temporal_identity(self) -> None:
        previous = pd.DataFrame(
            [_component("2026-01-05", "p", ["g:0:0"], 2.5, exposure_id="exposure-stable")]
        )
        current = pd.DataFrame([_component("2026-01-12", "c", ["g:0:0"], 2.5)])
        previous_members = pd.DataFrame([_membership("p", "a", "episode-a", stage="vegetative")])
        current_members = pd.DataFrame([_membership("c", "a", "episode-a", stage="flowering")])

        scores = score_temporal_candidates(previous, current, previous_members, current_members, CONFIG)
        self.assertEqual(float(scores.iloc[0]["stage_cosine"]), 0.0)
        tracking = link_weekly_components(previous, current, scores, CONFIG)
        self.assertEqual(tracking.assignments.iloc[0]["exposure_id"], "exposure-stable")
        self.assertTrue(tracking.lineage.empty)

    def test_append_does_not_change_existing_exposure_ids(self) -> None:
        weeks = ["2026-01-05", "2026-01-12", "2026-01-19"]
        components = [
            pd.DataFrame([_component(week, f"c{index}", ["g:0:0"], 2.5)])
            for index, week in enumerate(weeks, start=1)
        ]
        memberships = [
            pd.DataFrame([_membership(f"c{index}", "a", "episode-a")])
            for index in range(1, 4)
        ]

        first = link_weekly_components(pd.DataFrame(), components[0], pd.DataFrame(), CONFIG)
        week_one = apply_tracking_assignments(components[0], first.assignments)
        scores_two = score_temporal_candidates(week_one, components[1], memberships[0], memberships[1], CONFIG)
        second = link_weekly_components(week_one, components[1], scores_two, CONFIG)
        week_two = apply_tracking_assignments(components[1], second.assignments)
        original_ids = [week_one.iloc[0]["exposure_id"], week_two.iloc[0]["exposure_id"]]

        scores_three = score_temporal_candidates(week_two, components[2], memberships[1], memberships[2], CONFIG)
        third = link_weekly_components(week_two, components[2], scores_three, CONFIG)
        week_three = apply_tracking_assignments(components[2], third.assignments)

        self.assertEqual(original_ids[0], original_ids[1])
        self.assertTrue(str(original_ids[0]).startswith("exposure_"))
        self.assertEqual(week_three.iloc[0]["exposure_id"], original_ids[0])
        self.assertEqual(week_one.iloc[0]["exposure_id"], original_ids[0])
        self.assertEqual(week_two.iloc[0]["exposure_id"], original_ids[1])

    def test_low_coverage_freezes_lifecycle_clocks_and_recovery_closes(self) -> None:
        lifecycle = initialize_incident_lifecycle(
            "incident-a",
            "heat",
            {"timeline_bucket": "2026-01-05", "component_present": True, "adequate_coverage": True},
            CONFIG,
        )
        frozen = advance_incident_lifecycle(
            lifecycle,
            {"timeline_bucket": "2026-01-12", "component_present": True, "adequate_coverage": False},
            CONFIG,
        )
        self.assertEqual(frozen["incident_state"], "CANDIDATE")
        self.assertEqual(frozen["support_streak"], 1)
        self.assertEqual(frozen["data_gap_count"], 1)

        active = advance_incident_lifecycle(
            frozen,
            {"timeline_bucket": "2026-01-19", "component_present": True, "adequate_coverage": True},
            CONFIG,
        )
        self.assertEqual(active["incident_state"], "CONFIRMED")
        active = advance_incident_lifecycle(
            active,
            {"timeline_bucket": "2026-01-26", "component_present": True, "adequate_coverage": True},
            CONFIG,
        )
        self.assertEqual(active["incident_state"], "ACTIVE")
        quiet = advance_incident_lifecycle(
            active,
            {"timeline_bucket": "2026-02-02", "component_present": False, "adequate_coverage": True},
            CONFIG,
        )
        quiet_frozen = advance_incident_lifecycle(
            quiet,
            {"timeline_bucket": "2026-02-09", "component_present": False, "adequate_coverage": False},
            CONFIG,
        )
        self.assertEqual(quiet_frozen["quiet_streak"], 1)
        recovering = advance_incident_lifecycle(
            quiet_frozen,
            {"timeline_bucket": "2026-02-16", "component_present": False,
             "adequate_coverage": True, "impact_field_count": 3},
            CONFIG,
        )
        self.assertEqual(recovering["incident_state"], "RECOVERING")
        closed = advance_incident_lifecycle(
            recovering,
            {"timeline_bucket": "2026-02-23", "component_present": False,
             "adequate_coverage": True, "impact_field_count": 0,
             "recovery_evidence": True, "recovered_impact_field_count": 3},
            CONFIG,
        )
        self.assertEqual(closed["incident_state"], "CLOSED_RECOVERED")
        self.assertNotIn("DEAD", closed["incident_state"])

    def test_partial_recovery_cannot_close_unresolved_impact(self) -> None:
        recovering = {
            "incident_id": "incident-a",
            "hazard_family": "heat",
            "incident_state": "RECOVERING",
            "last_timeline_bucket": "2026-02-16",
            "recovery_streak": 0,
            "unresolved_streak": 0,
            "relapse_count": 0,
            "data_gap_count": 0,
        }

        updated = advance_incident_lifecycle(
            recovering,
            {
                "timeline_bucket": "2026-02-23",
                "component_present": False,
                "adequate_coverage": True,
                "impact_field_count": 9,
                "fresh_recovery_field_count": 1,
                "recovered_impact_field_count": 1,
                "recovery_evidence": True,
            },
            {**CONFIG, "recovery_observed_weeks": 1},
        )

        self.assertEqual(updated["incident_state"], "RECOVERING")
        self.assertEqual(updated["recovery_streak"], 0)
        self.assertEqual(updated["unresolved_streak"], 1)

    def test_relapse_clears_pressure_off_week_before_a_new_quiet_interval(self) -> None:
        for state in ("PRESSURE_QUIET", "RECOVERING"):
            with self.subTest(state=state):
                lifecycle = {
                    "incident_id": "incident-a",
                    "hazard_family": "heat",
                    "incident_state": state,
                    "last_timeline_bucket": "2026-02-02",
                    "pressure_off_week": "2026-02-02",
                    "quiet_streak": 1,
                    "recovery_streak": 1,
                    "unresolved_streak": 1,
                    "relapse_count": 0,
                    "data_gap_count": 0,
                }
                relapsed = advance_incident_lifecycle(
                    lifecycle,
                    {
                        "timeline_bucket": "2026-02-09",
                        "component_present": True,
                        "adequate_coverage": True,
                    },
                    CONFIG,
                )
                self.assertEqual(relapsed["incident_state"], "RELAPSED")
                self.assertIsNone(relapsed["pressure_off_week"])

                active = advance_incident_lifecycle(
                    relapsed,
                    {
                        "timeline_bucket": "2026-02-16",
                        "component_present": True,
                        "adequate_coverage": True,
                    },
                    CONFIG,
                )
                quiet = advance_incident_lifecycle(
                    active,
                    {
                        "timeline_bucket": "2026-02-23",
                        "component_present": False,
                        "adequate_coverage": True,
                    },
                    CONFIG,
                )
                self.assertEqual(quiet["incident_state"], "PRESSURE_QUIET")
                self.assertEqual(quiet["pressure_off_week"], "2026-02-23")

    def test_lifecycle_rejects_dead_state_and_duplicate_time(self) -> None:
        lifecycle = initialize_incident_lifecycle(
            "incident-a", "heat",
            {"timeline_bucket": "2026-01-05", "component_present": True}, CONFIG,
        )
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            advance_incident_lifecycle(
                lifecycle, {"timeline_bucket": "2026-01-05", "component_present": True}, CONFIG
            )
        with self.assertRaisesRegex(ValueError, "DEAD"):
            advance_incident_lifecycle(
                {**lifecycle, "incident_state": "DEAD"},
                {"timeline_bucket": "2026-01-12", "component_present": False}, CONFIG,
            )

    def test_coverage_gap_streak_resets_without_erasing_audit_count(self) -> None:
        lifecycle = initialize_incident_lifecycle(
            "incident-a",
            "heat",
            {
                "timeline_bucket": "2026-01-05",
                "component_present": True,
                "adequate_coverage": False,
            },
            CONFIG,
        )
        self.assertEqual(lifecycle["data_gap_count"], 1)
        self.assertEqual(lifecycle["coverage_gap_streak"], 1)
        observed = advance_incident_lifecycle(
            lifecycle,
            {
                "timeline_bucket": "2026-01-12",
                "component_present": True,
                "adequate_coverage": True,
            },
            CONFIG,
        )
        self.assertEqual(observed["data_gap_count"], 1)
        self.assertEqual(observed["coverage_gap_streak"], 0)
        separated = advance_incident_lifecycle(
            observed,
            {
                "timeline_bucket": "2026-01-19",
                "component_present": True,
                "adequate_coverage": False,
            },
            CONFIG,
        )
        self.assertEqual(separated["data_gap_count"], 2)
        self.assertEqual(separated["coverage_gap_streak"], 1)

    def test_boundary_censor_records_same_frozen_history_as_append_replay(self) -> None:
        lifecycle = initialize_incident_lifecycle(
            "incident-a",
            "heat",
            {
                "timeline_bucket": "2026-01-05",
                "component_present": True,
                "adequate_coverage": True,
            },
            CONFIG,
        )
        first_gap = advance_incident_lifecycle(
            lifecycle,
            {
                "timeline_bucket": "2026-01-12",
                "component_present": True,
                "adequate_coverage": False,
            },
            CONFIG,
        )
        boundary = advance_incident_lifecycle(
            first_gap,
            {
                "timeline_bucket": "2026-01-19",
                "component_present": True,
                "adequate_coverage": False,
                "data_censored": True,
            },
            CONFIG,
        )
        replayed_after_append = advance_incident_lifecycle(
            first_gap,
            {
                "timeline_bucket": "2026-01-19",
                "component_present": True,
                "adequate_coverage": False,
                "data_censored": False,
            },
            CONFIG,
        )

        self.assertEqual(boundary["incident_state"], "CLOSED_DATA_CENSORED")
        self.assertEqual(replayed_after_append["incident_state"], "CANDIDATE")
        for column in (
            "first_evidence_week", "confirmed_week", "pressure_off_week",
            "recovered_week", "merged_into_incident_id", "relapse_count",
            "data_gap_count", "coverage_gap_streak",
        ):
            self.assertEqual(boundary[column], replayed_after_append[column], column)
        self.assertEqual(boundary["data_gap_count"], 2)
        self.assertEqual(boundary["coverage_gap_streak"], 2)


if __name__ == "__main__":
    unittest.main()
