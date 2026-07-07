from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_cells_v3 import (
    build_component_field_rows,
    build_stage_baseline,
    build_weekly_exposure_cells,
)
from story_monitor.incident_policy_v3 import load_incident_policy_v3


class IncidentCellsV3Tests(unittest.TestCase):
    def test_stage_baseline_and_cell_significance_are_deterministic(self) -> None:
        policy = replace(
            load_incident_policy_v3(),
            minimum_evaluable_fields=3,
            minimum_active_fields=2,
            severe_override_min_fields=2,
            fdr_alpha=0.2,
        )
        context_rows = []
        lane_rows = []
        for week_index, week in enumerate(("2025-01-06", "2026-01-05")):
            for index in range(8):
                context_rows.append(
                    {
                        "timeline_bucket": week,
                        "field_id": f"f{index}",
                        "crop_instance_id": f"crop{index}",
                        "crop_name": "maize",
                        "stage_bucket": "vegetative",
                        "monitored": True,
                        "evaluable": True,
                        "centroid_lon": 30.000 + index * 0.001,
                        "centroid_lat": -1.000 + index * 0.001,
                    }
                )
            active_count = 1 if week_index == 0 else 5
            for index in range(active_count):
                lane_rows.append(
                    {
                        "timeline_bucket": week,
                        "event_id": f"event-{week_index}-{index}",
                        "field_id": f"f{index}",
                        "crop_instance_id": f"crop{index}",
                        "hazard_family": "heat",
                        "event_state": "SEVERE" if week_index and index < 2 else "ACTIVE",
                        "is_canonical_field_hazard_week": True,
                        "current_risk_rank": 4 if week_index else 3,
                        "daily_response_class": (
                            "severe_decline" if week_index and index < 2 else "no_material_change"
                        ),
                        "fresh_response_evidence": bool(week_index and index < 2),
                        "knowledge_time": f"{week}T18:30:00Z",
                    }
                )
        context_rows.append(
            {
                "timeline_bucket": "2025-12-29", "field_id": "crossing",
                "crop_instance_id": "crossing-crop", "crop_name": "maize",
                "stage_bucket": "flowering", "monitored": True, "evaluable": True,
                "centroid_lon": 30.0, "centroid_lat": -1.0,
            }
        )
        lane_rows.append(
            {
                "timeline_bucket": "2025-12-29", "event_id": "crossing-event",
                "field_id": "crossing", "crop_instance_id": "crossing-crop",
                "hazard_family": "drought", "event_state": "SEVERE",
                "is_canonical_field_hazard_week": True, "current_risk_rank": 4,
                "daily_response_class": "severe_decline",
                "fresh_response_evidence": True,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_path = root / "context.parquet"
            lanes_path = root / "lanes.parquet"
            pd.DataFrame(context_rows).to_parquet(context_path, index=False)
            pd.DataFrame(lane_rows).to_parquet(lanes_path, index=False)
            baseline = build_stage_baseline(
                context_path, lanes_path,
                baseline_through="2025-12-31", policy=policy, threads=1,
            )
            self.assertEqual(set(baseline["hazard_family"]), {"heat"})
            self.assertEqual(set(baseline["stage_bucket"]), {"vegetative"})
            cells = build_weekly_exposure_cells(
                context_path, lanes_path, baseline, policy=policy, threads=1
            )
            future = cells[cells["timeline_bucket"] == pd.Timestamp("2026-01-05")]
            self.assertGreaterEqual(len(future), 1)
            significant = future[future["is_significant"]]
            self.assertEqual(len(significant), 1)
            self.assertEqual(
                significant.iloc[0]["significance_reason"],
                "multi_field_severe_fresh_response_override",
            )
            fields = build_component_field_rows(
                context_path, lanes_path, cells, policy=policy, threads=1
            )
            future_fields = fields[fields["timeline_bucket"] == pd.Timestamp("2026-01-05")]
            self.assertEqual(len(future_fields), 5)
            self.assertTrue(future_fields["episode_id"].is_unique)
            self.assertEqual(set(future_fields["stage_family"]), {"vegetative"})
            self.assertTrue(
                pd.to_datetime(future_fields["knowledge_time"], utc=True)
                .eq(pd.Timestamp("2026-01-05T18:30:00Z"))
                .all()
            )
            repeated = build_weekly_exposure_cells(
                context_path, lanes_path, baseline, policy=policy, threads=1
            )
            pd.testing.assert_frame_equal(cells, repeated)
            partial_week = build_weekly_exposure_cells(
                context_path,
                lanes_path,
                baseline,
                policy=policy,
                assignment_after="2025-12-31",
                assignment_through="2026-01-08",
                threads=1,
            )
            self.assertTrue(partial_week.empty)
            complete_week = build_weekly_exposure_cells(
                context_path,
                lanes_path,
                baseline,
                policy=policy,
                assignment_after="2025-12-31",
                assignment_through="2026-01-11",
                threads=1,
            )
            self.assertFalse(complete_week.empty)
            empty_after_latest = build_weekly_exposure_cells(
                context_path,
                lanes_path,
                baseline,
                policy=policy,
                assignment_after="2027-01-01",
                assignment_through="2027-01-10",
                threads=1,
            )
            self.assertEqual(
                empty_after_latest.columns.tolist(),
                complete_week.columns.tolist(),
            )

            # A geographically distant future append must not shift the fixed
            # grid or rewrite any already-built cell identity.
            appended = pd.concat(
                [
                    pd.DataFrame(context_rows),
                    pd.DataFrame(
                        [
                            {
                                "timeline_bucket": "2026-01-12",
                                "field_id": "far-future",
                                "crop_instance_id": "far-future-crop",
                                "crop_name": "maize",
                                "stage_bucket": "vegetative",
                                "monitored": True,
                                "evaluable": True,
                                "centroid_lon": 42.0,
                                "centroid_lat": 18.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            appended.to_parquet(context_path, index=False)
            extended = build_weekly_exposure_cells(
                context_path, lanes_path, baseline, policy=policy, threads=1
            )
            key_columns = [
                "timeline_bucket", "hazard_family", "cell_id", "cell_x", "cell_y",
                "reference_latitude",
            ]
            before = cells.loc[
                cells["timeline_bucket"] <= pd.Timestamp("2026-01-05"), key_columns
            ].reset_index(drop=True)
            after = extended.loc[
                extended["timeline_bucket"] <= pd.Timestamp("2026-01-05"), key_columns
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(before, after)

    def test_field_denominator_gate_cannot_be_satisfied_by_many_crop_instances(self) -> None:
        policy = replace(
            load_incident_policy_v3(),
            minimum_evaluable_fields=2,
            minimum_active_fields=1,
            severe_override_min_fields=2,
        )
        context_rows = []
        lane_rows = []
        for week in ("2025-01-06", "2026-01-05"):
            for index in range(3):
                crop_instance = f"crop-{week}-{index}"
                context_rows.append(
                    {
                        "timeline_bucket": week,
                        "field_id": "one-field",
                        "crop_instance_id": crop_instance,
                        "crop_name": "maize",
                        "stage_bucket": "vegetative",
                        "monitored": True,
                        "evaluable": True,
                        "centroid_lon": 30.0,
                        "centroid_lat": -1.0,
                    }
                )
                lane_rows.append(
                    {
                        "timeline_bucket": week,
                        "event_id": f"event-{week}-{index}",
                        "field_id": "one-field",
                        "crop_instance_id": crop_instance,
                        "hazard_family": "heat",
                        "event_state": "ACTIVE",
                        "is_canonical_field_hazard_week": True,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_path = root / "context.parquet"
            lanes_path = root / "lanes.parquet"
            pd.DataFrame(context_rows).to_parquet(context_path, index=False)
            pd.DataFrame(lane_rows).to_parquet(lanes_path, index=False)
            baseline = build_stage_baseline(
                context_path,
                lanes_path,
                baseline_through="2025-12-31",
                policy=policy,
                threads=1,
            )
            cells = build_weekly_exposure_cells(
                context_path,
                lanes_path,
                baseline,
                policy=policy,
                assignment_after="2025-12-31",
                assignment_through="2026-01-11",
                threads=1,
            )
        self.assertEqual(int(cells.iloc[0]["evaluable_count"]), 3)
        self.assertEqual(int(cells.iloc[0]["evaluable_field_count"]), 1)
        self.assertFalse(bool(cells.iloc[0]["passes_denominator_gate"]))
        self.assertFalse(bool(cells.iloc[0]["is_significant"]))

    def test_baseline_rejects_empty_training_cohort(self) -> None:
        policy = load_incident_policy_v3()
        context = pd.DataFrame(
            [{
                "timeline_bucket": "2026-01-05", "field_id": "f", "crop_instance_id": "c",
                "stage_bucket": "unknown", "evaluable": True,
            }]
        )
        lanes = pd.DataFrame(
            [{
                "timeline_bucket": "2026-01-05", "event_id": "e", "field_id": "f",
                "crop_instance_id": "c", "hazard_family": "heat", "event_state": "ACTIVE",
                "is_canonical_field_hazard_week": True,
            }]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context.to_parquet(root / "context.parquet", index=False)
            lanes.to_parquet(root / "lanes.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "No evaluable pre-cutoff"):
                build_stage_baseline(
                    root / "context.parquet", root / "lanes.parquet",
                    baseline_through="2025-12-31", policy=policy, threads=1,
                )

    def test_new_post_baseline_hazard_fails_closed(self) -> None:
        policy = replace(
            load_incident_policy_v3(),
            minimum_evaluable_fields=1,
            minimum_active_fields=1,
        )
        context = pd.DataFrame(
            [
                {
                    "timeline_bucket": week,
                    "field_id": "field-1",
                    "crop_instance_id": f"crop-{week}",
                    "crop_name": "maize",
                    "stage_bucket": "vegetative",
                    "monitored": True,
                    "evaluable": True,
                    "centroid_lon": 30.0,
                    "centroid_lat": -1.0,
                }
                for week in ("2025-01-06", "2026-01-05")
            ]
        )
        lanes = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2025-01-06",
                    "event_id": "baseline-heat",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-2025-01-06",
                    "hazard_family": "heat",
                    "event_state": "ACTIVE",
                    "is_canonical_field_hazard_week": True,
                },
                {
                    "timeline_bucket": "2026-01-05",
                    "event_id": "new-drought",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-2026-01-05",
                    "hazard_family": "drought",
                    "event_state": "ACTIVE",
                    "is_canonical_field_hazard_week": True,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context.to_parquet(root / "context.parquet", index=False)
            lanes.to_parquet(root / "lanes.parquet", index=False)
            baseline = build_stage_baseline(
                root / "context.parquet",
                root / "lanes.parquet",
                baseline_through="2025-12-31",
                policy=policy,
                threads=1,
            )
            with self.assertRaisesRegex(ValueError, "unsupported_new_hazard.*drought"):
                build_weekly_exposure_cells(
                    root / "context.parquet",
                    root / "lanes.parquet",
                    baseline,
                    policy=policy,
                    assignment_after="2025-12-31",
                    assignment_through="2026-01-11",
                    threads=1,
                )


if __name__ == "__main__":
    unittest.main()
