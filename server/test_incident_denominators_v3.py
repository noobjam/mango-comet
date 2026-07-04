from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_denominators_v3 import (
    build_incident_stage_summary,
    enrich_incident_weekly_state,
)
from story_monitor.incident_policy_v3 import load_incident_policy_v3


class IncidentDenominatorsV3Tests(unittest.TestCase):
    def test_denominator_is_crop_and_stage_specific(self) -> None:
        policy = load_incident_policy_v3()
        reference_latitude = -2.0
        scale_lon = 111.32 * __import__("math").cos(__import__("math").radians(reference_latitude))
        lon = policy.grid_origin_lon + (1.5 * policy.grid_cell_size_km) / scale_lon
        lat = policy.grid_origin_lat + (1.5 * policy.grid_cell_size_km) / 110.574
        context = pd.DataFrame(
            [
                {
                    "timeline_bucket": "2026-01-05", "field_id": f"m{index}",
                    "crop_instance_id": f"maize-{index}", "crop_name": "Maize",
                    "stage_bucket": "flowering", "monitored": True,
                    "evaluable": index < 2, "centroid_available": True,
                    "centroid_lon": lon, "centroid_lat": lat,
                }
                for index in range(3)
            ]
            + [
                {
                    "timeline_bucket": "2026-01-05", "field_id": f"b{index}",
                    "crop_instance_id": f"beans-{index}", "crop_name": "Beans",
                    "stage_bucket": "flowering", "monitored": True,
                    "evaluable": True, "centroid_available": True,
                    "centroid_lon": lon, "centroid_lat": lat,
                }
                for index in range(2)
            ]
        )
        weekly = pd.DataFrame(
            [{
                "timeline_bucket": "2026-01-05", "incident_id": "incident-1",
                "exposure_id": "exposure-1", "crop_name": "maize",
                "hazard_family": "heat", "footprint_cell_ids_json": '["g:1:1"]',
            }]
        )
        memberships = pd.DataFrame(
            [{
                "timeline_bucket": "2026-01-05", "incident_id": "incident-1",
                "exposure_id": "exposure-1", "crop_name_normalized": "maize",
                # Membership history may retain an earlier stage. The current
                # field-week context must own the weekly denominator stage.
                "hazard_family": "heat", "stage_bucket": "vegetative",
                "field_id": "m0", "crop_instance_id": "maize-0", "episode_id": "e",
                "membership_role": "pressure_core", "event_state": "SEVERE",
                "grid_id": "g:1:1",
            }]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.parquet"
            context.to_parquet(path, index=False)
            summary = build_incident_stage_summary(
                path, weekly, memberships, policy=policy,
                reference_latitude=reference_latitude, threads=1,
            )
        row = summary.iloc[0]
        self.assertEqual(row["stage_bucket"], "flowering")
        self.assertEqual(row["monitored_crop_instance_count"], 3)
        self.assertEqual(row["evaluable_crop_instance_count"], 2)
        self.assertEqual(row["pressure_core_crop_instance_count"], 1)
        self.assertEqual(row["severe_crop_instance_count"], 1)
        self.assertAlmostEqual(row["impact_signal_rate"], 1 / 3)
        self.assertEqual(
            row["denominator_scope"],
            "crop_instances_in_pressure_watch_and_impact_cells",
        )
        self.assertEqual(row["coverage_missing_cell_count"], 0)
        enriched = enrich_incident_weekly_state(
            weekly.assign(incident_state="ACTIVE"), summary
        ).iloc[0]
        self.assertEqual(enriched["monitored_count"], 3)
        self.assertEqual(enriched["evaluable_count"], 2)
        self.assertEqual(enriched["affected_count"], 1)
        self.assertEqual(enriched["severe_count"], 1)
        self.assertEqual(enriched["current_state"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()
