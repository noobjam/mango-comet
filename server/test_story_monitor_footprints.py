from __future__ import annotations

import unittest

import pandas as pd

from story_monitor.footprints import build_motif_weekly_footprints


class MotifFootprintTests(unittest.TestCase):
    def test_trail_connects_only_consecutive_overlapping_active_sets(self) -> None:
        states = pd.DataFrame(
            [
                {"timeline_bucket": "2026-01-05", "motif_id": "m1", "field_id": "a"},
                {"timeline_bucket": "2026-01-05", "motif_id": "m1", "field_id": "b"},
                {"timeline_bucket": "2026-01-12", "motif_id": "m1", "field_id": "b"},
                {"timeline_bucket": "2026-01-12", "motif_id": "m1", "field_id": "c"},
                {"timeline_bucket": "2026-01-19", "motif_id": "m1", "field_id": "d"},
                {"timeline_bucket": "2026-02-02", "motif_id": "m1", "field_id": "d"},
            ]
        )
        geometry = pd.DataFrame(
            [
                {"field_id": "a", "centroid_lon": 30.0, "centroid_lat": -1.0},
                {"field_id": "b", "centroid_lon": 30.1, "centroid_lat": -1.0},
                {"field_id": "c", "centroid_lon": 30.2, "centroid_lat": -1.0},
                {"field_id": "d", "centroid_lon": 31.0, "centroid_lat": -1.0},
            ]
        )

        result = build_motif_weekly_footprints(states, geometry)

        self.assertEqual(result["trail_segment_allowed"].tolist(), [False, True, False, False])
        self.assertEqual(result.loc[1, "persisting_field_count"], 1)
        self.assertAlmostEqual(result.loc[1, "field_overlap_jaccard"], 1 / 3)
        self.assertEqual(result.loc[2, "trail_break_reason"], "zero_field_overlap")
        self.assertEqual(result.loc[3, "trail_break_reason"], "nonconsecutive_week")
        self.assertFalse(result["is_physical_movement"].any())


if __name__ == "__main__":
    unittest.main()
