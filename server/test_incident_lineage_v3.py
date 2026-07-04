from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.incident_lineage_v3 import (
    LINEAGE_SCHEMA_VERSION,
    build_incident_lineage_v3 as _build_incident_lineage_v3,
    remap_incident_lineage_segments,
)


def _edge(
    parent: str,
    child: str,
    lineage_type: str,
    *,
    week: str = "2026-01-12",
    previous_component: str | None = None,
    current_component: str | None = None,
    score: float = 0.9,
) -> dict[str, object]:
    return {
        "timeline_bucket": week,
        "parent_exposure_id": parent,
        "child_exposure_id": child,
        "lineage_type": lineage_type,
        "score": score,
        "previous_component_id": previous_component or f"component-{parent}",
        "current_component_id": current_component or f"component-{child}",
    }


def _catalog(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["exposure_id", "crop_name_normalized", "incident_id"],
    )


def build_incident_lineage_v3(
    lineage: pd.DataFrame, catalog: pd.DataFrame
):
    records: list[dict[str, str]] = []
    crops_by_exposure = {
        str(exposure): set(group["crop_name_normalized"].astype(str).str.lower())
        for exposure, group in catalog.groupby("exposure_id", sort=False)
    }
    for edge_index, edge in enumerate(lineage.to_dict("records")):
        parent = str(edge.get("parent_exposure_id") or edge.get("source_exposure_id"))
        child = str(edge.get("child_exposure_id") or edge.get("target_exposure_id"))
        previous = str(edge.get("previous_component_id"))
        current = str(edge.get("current_component_id"))
        for crop in sorted(crops_by_exposure.get(parent, set()) & crops_by_exposure.get(child, set())):
            records.extend(
                [
                    {
                        "component_id": previous, "crop_name": crop,
                        "membership_role": "pressure_core",
                        "field_id": f"previous-{edge_index}-{crop}",
                    },
                    {
                        "component_id": current, "crop_name": crop,
                        "membership_role": "pressure_core",
                        "field_id": f"current-{edge_index}-{crop}",
                    },
                ]
            )
    memberships = pd.DataFrame(
        records,
        columns=["component_id", "crop_name", "membership_role", "field_id"],
    )
    return _build_incident_lineage_v3(lineage, catalog, memberships)


class IncidentLineageV3Tests(unittest.TestCase):
    def test_split_maps_shared_crops_and_is_deterministic(self) -> None:
        lineage = pd.DataFrame([_edge("parent", "child", "split")])
        catalog = _catalog(
            ("parent", "Maize", "incident-parent-maize"),
            ("parent", "Beans", "incident-parent-beans"),
            ("child", "maize", "incident-child-maize"),
            ("child", "beans", "incident-child-beans"),
        )

        first = build_incident_lineage_v3(lineage, catalog)
        shuffled = build_incident_lineage_v3(
            lineage.sample(frac=1.0, random_state=3).reset_index(drop=True),
            catalog.sample(frac=1.0, random_state=7).reset_index(drop=True),
        )

        self.assertEqual(len(first.lineage), 2)
        self.assertEqual(set(first.lineage["crop_name_normalized"]), {"beans", "maize"})
        self.assertEqual(set(first.lineage["lineage_type"]), {"split"})
        self.assertEqual(set(first.lineage["schema_version"]), {LINEAGE_SCHEMA_VERSION})
        assert_frame_equal(first.lineage, shuffled.lineage)
        assert_frame_equal(first.incident_metadata, shuffled.incident_metadata)

        metadata = first.incident_metadata.set_index("incident_id")
        for crop in ("beans", "maize"):
            parent = metadata.loc[f"incident-parent-{crop}"]
            child = metadata.loc[f"incident-child-{crop}"]
            self.assertEqual(parent["split_count"], 1)
            self.assertEqual(parent["split_out_count"], 1)
            self.assertEqual(parent["split_from_count"], 0)
            self.assertEqual(child["split_count"], 1)
            self.assertEqual(child["split_out_count"], 0)
            self.assertEqual(child["split_from_count"], 1)

    def test_merge_preserves_multiple_parents_and_terminal_metadata(self) -> None:
        lineage = pd.DataFrame(
            [
                _edge("parent-a", "child", "merge", previous_component="component-a"),
                _edge("parent-b", "child", "merge", previous_component="component-b"),
            ]
        )
        catalog = _catalog(
            ("parent-a", "maize", "incident-a"),
            ("parent-b", "maize", "incident-b"),
            ("child", "maize", "incident-child"),
        )

        artifacts = build_incident_lineage_v3(lineage, catalog)

        self.assertEqual(len(artifacts.lineage), 2)
        self.assertEqual(
            set(
                zip(
                    artifacts.lineage["parent_incident_id"],
                    artifacts.lineage["child_incident_id"],
                )
            ),
            {("incident-a", "incident-child"), ("incident-b", "incident-child")},
        )
        metadata = artifacts.incident_metadata.set_index("incident_id")
        for parent_id in ("incident-a", "incident-b"):
            parent = metadata.loc[parent_id]
            self.assertEqual(parent["merge_count"], 1)
            self.assertEqual(parent["merge_out_count"], 1)
            self.assertEqual(parent["merge_in_count"], 0)
            self.assertEqual(parent["merged_into_incident_id"], "incident-child")
            self.assertEqual(parent["merged_week"], pd.Timestamp("2026-01-12"))
        child = metadata.loc["incident-child"]
        self.assertEqual(child["merge_count"], 2)
        self.assertEqual(child["merge_in_count"], 2)
        self.assertEqual(child["merge_out_count"], 0)
        self.assertTrue(pd.isna(child["merged_into_incident_id"]))
        self.assertTrue(pd.isna(child["merged_week"]))

    def test_crop_missing_on_either_side_is_not_fabricated(self) -> None:
        lineage = pd.DataFrame(
            [
                _edge("parent", "child", "split"),
                _edge(
                    "other-parent",
                    "other-child",
                    "merge",
                    previous_component="component-other-parent",
                    current_component="component-other-child",
                ),
            ]
        )
        catalog = _catalog(
            ("parent", "maize", "incident-parent-maize"),
            ("parent", "beans", "incident-parent-beans"),
            ("child", "maize", "incident-child-maize"),
            ("other-parent", "beans", "incident-other-parent-beans"),
            ("other-child", "maize", "incident-other-child-maize"),
        )

        artifacts = build_incident_lineage_v3(lineage, catalog)

        self.assertEqual(len(artifacts.lineage), 1)
        row = artifacts.lineage.iloc[0]
        self.assertEqual(row["crop_name_normalized"], "maize")
        self.assertEqual(row["parent_incident_id"], "incident-parent-maize")
        self.assertEqual(row["child_incident_id"], "incident-child-maize")
        metadata = artifacts.incident_metadata.set_index("incident_id")
        self.assertEqual(metadata.loc["incident-parent-beans", "split_count"], 0)
        self.assertEqual(metadata.loc["incident-other-parent-beans", "merge_count"], 0)
        self.assertEqual(metadata.loc["incident-other-child-maize", "merge_count"], 0)

    def test_source_target_aliases_and_related_unmatched_are_supported(self) -> None:
        lineage = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-12",
                    "source_exposure_id": "parent",
                    "target_exposure_id": "child",
                    "lineage_type": "split",
                    "score": 0.8,
                    "previous_component_id": "component-parent",
                    "current_component_id": "component-child",
                },
                {
                    "timeline_bucket": "2026-01-19",
                    "source_exposure_id": "child",
                    "target_exposure_id": "unmatched",
                    "lineage_type": "related_unmatched",
                    "score": 0.2,
                    "previous_component_id": "component-child",
                    "current_component_id": "component-unmatched",
                },
            ]
        )
        catalog = _catalog(
            ("parent", "maize", "incident-parent"),
            ("child", "maize", "incident-child"),
            ("unmatched", "maize", "incident-unmatched"),
        )

        artifacts = build_incident_lineage_v3(lineage, catalog)

        self.assertEqual(len(artifacts.lineage), 1)
        self.assertEqual(artifacts.lineage.iloc[0]["parent_exposure_id"], "parent")
        self.assertEqual(artifacts.lineage.iloc[0]["child_exposure_id"], "child")
        unmatched = artifacts.incident_metadata.set_index("incident_id").loc[
            "incident-unmatched"
        ]
        self.assertEqual(unmatched["split_count"], 0)
        self.assertEqual(unmatched["merge_count"], 0)

    def test_duplicate_component_edge_is_rejected(self) -> None:
        edge = _edge("parent", "child", "split")
        with self.assertRaisesRegex(ValueError, "duplicate component edges"):
            build_incident_lineage_v3(
                pd.DataFrame([edge, edge]),
                _catalog(
                    ("parent", "maize", "incident-parent"),
                    ("child", "maize", "incident-child"),
                ),
            )

    def test_duplicate_catalog_keys_and_incident_ids_are_rejected(self) -> None:
        lineage = pd.DataFrame([_edge("parent", "child", "split")])
        with self.assertRaisesRegex(ValueError, "duplicates exposure and crop"):
            build_incident_lineage_v3(
                lineage,
                _catalog(
                    ("parent", "Maize", "incident-parent-a"),
                    ("parent", "maize", "incident-parent-b"),
                    ("child", "maize", "incident-child"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "one incident_id more than once"):
            build_incident_lineage_v3(
                lineage,
                _catalog(
                    ("parent", "maize", "incident-shared"),
                    ("child", "maize", "incident-shared"),
                ),
            )

    def test_cycle_is_rejected(self) -> None:
        lineage = pd.DataFrame(
            [
                _edge("a", "b", "split", week="2026-01-12"),
                _edge("b", "a", "merge", week="2026-01-19"),
            ]
        )
        catalog = _catalog(
            ("a", "maize", "incident-a"),
            ("b", "maize", "incident-b"),
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            build_incident_lineage_v3(lineage, catalog)

    def test_one_incident_cannot_merge_into_multiple_targets(self) -> None:
        lineage = pd.DataFrame(
            [
                _edge("parent", "child-a", "merge", current_component="component-a"),
                _edge("parent", "child-b", "merge", current_component="component-b"),
            ]
        )
        catalog = _catalog(
            ("parent", "maize", "incident-parent"),
            ("child-a", "maize", "incident-child-a"),
            ("child-b", "maize", "incident-child-b"),
        )

        with self.assertRaisesRegex(ValueError, "merges into multiple targets"):
            build_incident_lineage_v3(lineage, catalog)

    def test_crop_absent_from_edge_component_cannot_create_historical_lineage(self) -> None:
        lineage = pd.DataFrame(
            [_edge("parent", "child", "merge", week="2026-01-12")]
        )
        catalog = _catalog(
            ("parent", "maize", "incident-parent-maize"),
            ("child", "maize", "incident-child-maize"),
            ("child", "beans", "incident-child-beans"),
        )
        memberships = pd.DataFrame(
            [
                {
                    "component_id": "component-parent", "crop_name": "maize",
                    "membership_role": "pressure_core",
                },
                {
                    "component_id": "component-child", "crop_name": "beans",
                    "membership_role": "pressure_core",
                },
            ]
        )
        result = _build_incident_lineage_v3(lineage, catalog, memberships)
        self.assertTrue(result.lineage.empty)
        metadata = result.incident_metadata.set_index("incident_id")
        self.assertEqual(metadata.loc["incident-parent-maize", "merge_count"], 0)

    def test_exposure_edge_is_remapped_to_recurrence_segments_active_that_week(self) -> None:
        base = build_incident_lineage_v3(
            pd.DataFrame(
                [_edge("parent", "child", "merge", week="2026-02-02")]
            ),
            _catalog(
                ("parent", "maize", "incident-parent-base"),
                ("child", "maize", "incident-child-base"),
            ),
        )
        weekly = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-02-02",
                    "base_incident_id": "incident-parent-base",
                    "incident_id": "incident-parent-recurrence",
                },
                {
                    "timeline_bucket": "2026-02-02",
                    "base_incident_id": "incident-child-base",
                    "incident_id": "incident-child-recurrence",
                },
            ]
        )
        segmented_catalog = _catalog(
            ("parent", "maize", "incident-parent-base"),
            ("parent", "maize", "incident-parent-recurrence"),
            ("child", "maize", "incident-child-base"),
            ("child", "maize", "incident-child-recurrence"),
        )
        result = remap_incident_lineage_segments(
            base.lineage, weekly, segmented_catalog
        )
        self.assertEqual(len(result.lineage), 1)
        edge = result.lineage.iloc[0]
        self.assertEqual(edge["parent_incident_id"], "incident-parent-recurrence")
        self.assertEqual(edge["child_incident_id"], "incident-child-recurrence")
        metadata = result.incident_metadata.set_index("incident_id")
        self.assertEqual(
            metadata.loc["incident-parent-recurrence", "merged_into_incident_id"],
            "incident-child-recurrence",
        )
        self.assertEqual(metadata.loc["incident-parent-base", "merge_count"], 0)


if __name__ == "__main__":
    unittest.main()
