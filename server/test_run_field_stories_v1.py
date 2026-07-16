from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from run_field_stories_v1 import ARTIFACTS, build_field_story_release


class RunFieldStoriesV1Tests(unittest.TestCase):
    def test_builds_partitioned_immutable_release(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            output = root / "field-stories"
            evidence.mkdir()
            _write_evidence(evidence)

            with patch(
                "run_field_stories_v1.validate_evidence_directory",
                return_value={"status": "complete"},
            ) as validate:
                manifest = build_field_story_release(
                    evidence,
                    output,
                    partitions=3,
                    threads=1,
                )

            validate.assert_called_once_with(evidence.resolve())
            self.assertEqual(manifest["schema_version"], "field-stories-v1/1")
            self.assertTrue(manifest["semantics"]["multi_hazard"])
            self.assertFalse(manifest["semantics"]["machine_learning"])
            self.assertEqual(manifest["artifacts"]["windows"]["row_count"], 1)
            for filename, _ in ARTIFACTS.values():
                self.assertTrue((output / filename).is_file())
            stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)
            daily = pd.read_parquet(output / ARTIFACTS["daily_state"][0])
            self.assertEqual(daily["story_id"].nunique(), 1)
            self.assertEqual(
                json.loads(daily.iloc[1]["active_hazards_json"]),
                ["drought", "heat"],
            )

            with self.assertRaises(FileExistsError):
                build_field_story_release(evidence, output, partitions=1, threads=1)


def _write_evidence(evidence: Path) -> None:
    days = ["2026-01-01", "2026-01-02", "2026-01-03"]
    crop = pd.DataFrame(
        [
            {
                "field_id": "field-1",
                "crop_instance_id": "crop-1",
                "observation_date": day,
                "knowledge_time": f"{day}T08:00:00Z",
                "crop_name": "maize",
                "crop_season": "2026-A",
                "stage_bucket": "vegetative",
            }
            for day in days
        ]
    )
    pressure_rows = []
    ranks = {
        "2026-01-01": {"drought": 2, "heat": 0},
        "2026-01-02": {"drought": 3, "heat": 3},
        "2026-01-03": {"drought": 0, "heat": 3},
    }
    for day in days:
        for hazard in ("drought", "heat"):
            pressure_rows.append(
                {
                    "record_id": f"{day}-{hazard}",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "pressure_observation_date": day,
                    "knowledge_time": f"{day}T09:00:00Z",
                    "hazard_family": hazard,
                    "pressure_observed": True,
                    "pressure_rank": ranks[day][hazard],
                }
            )
    responses = pd.DataFrame(
        columns=[
            "acquisition_id",
            "field_id",
            "crop_instance_id",
            "spectral_source_date",
            "knowledge_time",
            "spectral_usable",
            "new_response_evidence",
            "response_class",
        ]
    )
    crop.to_parquet(evidence / "crop_day_context_v4.parquet", index=False)
    pd.DataFrame(pressure_rows).to_parquet(
        evidence / "field_day_pressure_v4.parquet", index=False
    )
    responses.to_parquet(evidence / "field_s2_acquisition_v4.parquet", index=False)
    (evidence / "manifest.json").write_text(
        json.dumps({"schema_version": "test-v4", "status": "complete"}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
