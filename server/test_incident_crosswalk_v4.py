from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.incident_crosswalk_v4 import (
    CROSSWALK_COLUMNS,
    MEMBERSHIP_FILENAME,
    SCHEMA_VERSION,
    build_incident_crosswalk_v4,
)


def _memberships(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["incident_id", "timeline_bucket", "field_id"]
    )


class IncidentCrosswalkV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.old = _memberships(
            ("old-a", "2026-01-05", "field-1"),
            ("old-a", "2026-01-12", "field-2"),
            ("old-a", "2026-01-19", "field-3"),
            ("old-b", "2026-01-05", "field-9"),
        )
        self.new = _memberships(
            ("new-x", "2026-01-05", "field-1"),
            ("new-x", "2026-01-12", "field-2"),
            ("new-x", "2026-01-26", "field-4"),
            ("new-y", "2026-01-19", "field-3"),
            ("new-z", "2026-01-05", "field-8"),
        )

    def test_emits_overlap_metrics_and_unmatched_incidents(self) -> None:
        crosswalk = build_incident_crosswalk_v4(self.old, self.new)

        self.assertEqual(tuple(crosswalk.columns), CROSSWALK_COLUMNS)
        self.assertEqual(
            list(crosswalk["match_status"]),
            ["overlap", "overlap", "unmatched_old", "unmatched_new"],
        )
        first = crosswalk.iloc[0]
        self.assertEqual(
            (first.old_incident_id, first.new_incident_id), ("old-a", "new-x")
        )
        self.assertEqual(
            (
                first.old_field_week_count,
                first.new_field_week_count,
                first.overlap_field_week_count,
                first.union_field_week_count,
            ),
            (3, 3, 2, 4),
        )
        self.assertAlmostEqual(first.jaccard_similarity, 0.5)
        self.assertAlmostEqual(first.old_coverage_fraction, 2 / 3)
        self.assertAlmostEqual(first.new_coverage_fraction, 2 / 3)

        second = crosswalk.iloc[1]
        self.assertEqual(
            (second.old_incident_id, second.new_incident_id), ("old-a", "new-y")
        )
        self.assertEqual(second.overlap_field_week_count, 1)
        self.assertEqual(second.union_field_week_count, 3)
        self.assertAlmostEqual(second.jaccard_similarity, 1 / 3)
        self.assertAlmostEqual(second.old_coverage_fraction, 1 / 3)
        self.assertAlmostEqual(second.new_coverage_fraction, 1.0)

        unmatched_old = crosswalk.iloc[2]
        self.assertEqual(unmatched_old.old_incident_id, "old-b")
        self.assertTrue(pd.isna(unmatched_old.new_incident_id))
        self.assertEqual(unmatched_old.old_field_week_count, 1)
        self.assertEqual(unmatched_old.new_field_week_count, 0)
        self.assertEqual(unmatched_old.union_field_week_count, 1)
        self.assertEqual(unmatched_old.jaccard_similarity, 0.0)

        unmatched_new = crosswalk.iloc[3]
        self.assertTrue(pd.isna(unmatched_new.old_incident_id))
        self.assertEqual(unmatched_new.new_incident_id, "new-z")
        self.assertEqual(unmatched_new.old_field_week_count, 0)
        self.assertEqual(unmatched_new.new_field_week_count, 1)
        self.assertEqual(unmatched_new.union_field_week_count, 1)
        self.assertEqual(set(crosswalk["schema_version"]), {SCHEMA_VERSION})

    def test_output_is_independent_of_input_order(self) -> None:
        expected = build_incident_crosswalk_v4(self.old, self.new)

        shuffled = build_incident_crosswalk_v4(
            self.old.sample(frac=1.0, random_state=13).reset_index(drop=True),
            self.new.sample(frac=1.0, random_state=29).reset_index(drop=True),
        )

        assert_frame_equal(expected, shuffled)

    def test_reads_membership_parquets_from_release_directories(self) -> None:
        expected = build_incident_crosswalk_v4(self.old, self.new)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_dir = root / "old"
            new_dir = root / "new"
            old_dir.mkdir()
            new_dir.mkdir()
            self.old.to_parquet(old_dir / MEMBERSHIP_FILENAME, index=False)
            self.new.to_parquet(new_dir / MEMBERSHIP_FILENAME, index=False)

            actual = build_incident_crosswalk_v4(old_dir, new_dir)

        assert_frame_equal(expected, actual)

    def test_empty_release_emits_every_incident_on_the_other_side(self) -> None:
        empty = _memberships()

        unmatched_new = build_incident_crosswalk_v4(empty, self.new)
        unmatched_old = build_incident_crosswalk_v4(self.old, empty)
        both_empty = build_incident_crosswalk_v4(empty, empty)

        self.assertEqual(
            list(unmatched_new["new_incident_id"]), ["new-x", "new-y", "new-z"]
        )
        self.assertEqual(set(unmatched_new["match_status"]), {"unmatched_new"})
        self.assertEqual(
            list(unmatched_old["old_incident_id"]), ["old-a", "old-b"]
        )
        self.assertEqual(set(unmatched_old["match_status"]), {"unmatched_old"})
        self.assertEqual(tuple(both_empty.columns), CROSSWALK_COLUMNS)
        self.assertTrue(both_empty.empty)

    def test_deduplicates_audit_membership_but_rejects_new_duplicates(self) -> None:
        duplicate_old = pd.concat(
            [
                self.old,
                _memberships(("old-a", "2026-01-05T12:00:00", " field-1 ")),
            ],
            ignore_index=True,
        )
        duplicate_new = pd.concat(
            [
                self.new,
                _memberships(("new-x", "2026-01-05T18:00:00", "field-1")),
            ],
            ignore_index=True,
        )

        expected = build_incident_crosswalk_v4(self.old, self.new)
        actual = build_incident_crosswalk_v4(duplicate_old, self.new)
        assert_frame_equal(expected, actual)
        with self.assertRaisesRegex(ValueError, "new.*not canonical"):
            build_incident_crosswalk_v4(self.old, duplicate_new)

    def test_old_identity_changes_cannot_change_new_identity_or_metrics(self) -> None:
        first = build_incident_crosswalk_v4(self.old, self.new)
        renamed_old = self.old.copy()
        renamed_old["incident_id"] = renamed_old["incident_id"].replace(
            {"old-a": "audit-only-a", "old-b": "audit-only-b"}
        )

        second = build_incident_crosswalk_v4(renamed_old, self.new)

        assert_frame_equal(
            first.drop(columns="old_incident_id"),
            second.drop(columns="old_incident_id"),
        )
        self.assertEqual(
            set(second["new_incident_id"].dropna()), {"new-x", "new-y", "new-z"}
        )


if __name__ == "__main__":
    unittest.main()
