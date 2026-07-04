from __future__ import annotations

import json
import unittest

import pandas as pd

from story_monitor.incident_exposures_v3 import track_exposures


def _component(week: str, component: str, fields: int = 2) -> dict[str, object]:
    return {
        "timeline_bucket": week, "hazard_family": "heat", "component_id": component,
        "cell_ids_json": json.dumps(["g:1:1"]), "active_field_count": fields,
        "severe_field_count": 0, "center_x_km": 7.5, "center_y_km": 7.5,
    }


def _members(week: str, component: str) -> list[dict[str, object]]:
    return [
        {
            "timeline_bucket": week, "hazard_family": "heat", "component_id": component,
            "field_id": field, "crop_instance_id": f"crop-{field}",
            "episode_id": f"episode-{field}", "membership_role": "pressure_core",
            "event_state": "ACTIVE", "stage_bucket": "vegetative", "crop_name": "maize",
            "grid_id": "g:1:1",
        }
        for field in ("a", "b")
    ]


def _selected_members(
    week: str, component: str, fields: tuple[str, ...], grid_id: str
) -> list[dict[str, object]]:
    return [
        {
            "timeline_bucket": week, "hazard_family": "heat",
            "component_id": component, "field_id": field,
            "crop_instance_id": f"crop-{field}", "episode_id": f"episode-{field}",
            "membership_role": "pressure_core", "event_state": "ACTIVE",
            "stage_bucket": "vegetative", "crop_name": "maize",
            "grid_id": grid_id,
        }
        for field in fields
    ]


class IncidentExposuresV3Tests(unittest.TestCase):
    def test_tracks_one_exposure_and_preserves_history_on_append(self) -> None:
        components = pd.DataFrame(
            [_component("2026-01-05", "component-a"), _component("2026-01-12", "component-b")]
        )
        members = pd.DataFrame(
            _members("2026-01-05", "component-a") + _members("2026-01-12", "component-b")
        )
        config = {
            "policy_version": "test-v3", "max_gap_weeks": 2,
            "continuation_threshold": 0.2, "lineage_threshold": 0.1,
        }
        first = track_exposures(components, members, config)
        self.assertEqual(first.assignments["exposure_id"].nunique(), 1)
        self.assertEqual(first.weekly_state.iloc[1]["persisting_field_count"], 2)
        self.assertFalse(bool(first.weekly_state.iloc[1]["is_physical_movement"]))

        extended_components = pd.concat(
            [components, pd.DataFrame([_component("2026-01-19", "component-c")])],
            ignore_index=True,
        )
        extended_members = pd.concat(
            [members, pd.DataFrame(_members("2026-01-19", "component-c"))],
            ignore_index=True,
        )
        extended = track_exposures(extended_components, extended_members, config)
        before = first.assignments.set_index("component_id")["exposure_id"].to_dict()
        after = extended.assignments.set_index("component_id")["exposure_id"].to_dict()
        self.assertEqual(before, {key: after[key] for key in before})
        self.assertEqual(extended.assignments["exposure_id"].nunique(), 1)

    def test_integrated_split_emits_parent_child_lineage(self) -> None:
        components = pd.DataFrame(
            [
                {**_component("2026-01-05", "component-parent", 4),
                 "cell_ids_json": '["g:1:1","g:1:2"]'},
                {**_component("2026-01-12", "component-child-a", 2),
                 "cell_ids_json": '["g:1:1"]'},
                {**_component("2026-01-12", "component-child-b", 2),
                 "cell_ids_json": '["g:1:2"]'},
            ]
        )
        members = pd.DataFrame(
            _selected_members(
                "2026-01-05", "component-parent", ("a", "b", "c", "d"), "g:1:1"
            )
            + _selected_members(
                "2026-01-12", "component-child-a", ("a", "b"), "g:1:1"
            )
            + _selected_members(
                "2026-01-12", "component-child-b", ("c", "d"), "g:1:2"
            )
        )
        result = track_exposures(
            components,
            members,
            {
                "policy_version": "test-v3", "max_gap_weeks": 2,
                "continuation_threshold": 0.2, "lineage_threshold": 0.1,
                "minimum_lineage_jaccard": 0.1,
            },
        )
        split = result.lineage[result.lineage["lineage_type"] == "split"]
        self.assertEqual(len(split), 1)
        self.assertTrue(split.iloc[0]["parent_exposure_id"].startswith("exposure_"))
        self.assertTrue(split.iloc[0]["child_exposure_id"].startswith("exposure_"))
        self.assertNotEqual(
            split.iloc[0]["parent_exposure_id"], split.iloc[0]["child_exposure_id"]
        )

    def test_integrated_merge_closes_losing_exposure_into_winner(self) -> None:
        components = pd.DataFrame(
            [
                {**_component("2026-01-05", "component-parent-a", 2),
                 "cell_ids_json": '["g:1:1"]'},
                {**_component("2026-01-05", "component-parent-b", 2),
                 "cell_ids_json": '["g:1:2"]'},
                {**_component("2026-01-12", "component-child", 4),
                 "cell_ids_json": '["g:1:1","g:1:2"]'},
            ]
        )
        members = pd.DataFrame(
            _selected_members(
                "2026-01-05", "component-parent-a", ("a", "b"), "g:1:1"
            )
            + _selected_members(
                "2026-01-05", "component-parent-b", ("c", "d"), "g:1:2"
            )
            + _selected_members(
                "2026-01-12", "component-child", ("a", "b", "c", "d"), "g:1:1"
            )
        )
        result = track_exposures(
            components,
            members,
            {
                "policy_version": "test-v3", "max_gap_weeks": 2,
                "continuation_threshold": 0.2, "lineage_threshold": 0.1,
                "minimum_lineage_jaccard": 0.1,
            },
        )
        merge = result.lineage[result.lineage["lineage_type"] == "merge"]
        self.assertEqual(len(merge), 1)
        parent = str(merge.iloc[0]["parent_exposure_id"])
        child = str(merge.iloc[0]["child_exposure_id"])
        self.assertNotEqual(parent, child)
        self.assertNotIn(
            parent,
            set(
                result.assignments.loc[
                    result.assignments["timeline_bucket"] == pd.Timestamp("2026-01-12"),
                    "exposure_id",
                ].astype(str)
            ),
        )


if __name__ == "__main__":
    unittest.main()
