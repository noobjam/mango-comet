from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_release_v4 import CORRECTION_POLICY
from story_monitor.incident_validation_v4 import (
    validate_evidence_append,
    validate_evidence_directory,
)


class IncidentValidationV4Tests(unittest.TestCase):
    def test_valid_release_and_ordinary_append_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_release(previous, "2025-01-08")
            _write_release(current, "2025-01-09", append=True)

            result = validate_evidence_directory(previous)
            append = validate_evidence_append(previous, current)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(append["appended_row_counts"], {
                "crop": 1, "pressure": 1, "s2": 1,
            })

    def test_append_rewrite_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_release(previous, "2025-01-08")
            _write_release(current, "2025-01-09", append=True)
            pressure = pd.read_parquet(current / "field_day_pressure_v4.parquet")
            pressure.loc[0, "pressure_score"] = 999.0
            pressure.to_parquet(current / "field_day_pressure_v4.parquet", index=False)
            _refresh_signed_artifacts(current)

            with self.assertRaisesRegex(ValueError, "rewrote"):
                validate_evidence_append(previous, current)

    def test_same_day_append_uses_monotonic_released_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            _write_release(
                previous, "2025-01-09",
                released_at="2025-01-09T08:00:00Z",
            )
            _write_release(
                current, "2025-01-09", append=True,
                released_at="2025-01-09T12:00:00Z",
            )

            result = validate_evidence_append(previous, current)

            self.assertEqual(
                result["previous_release_as_of"], result["current_release_as_of"]
            )
            self.assertLess(
                result["previous_released_at"], result["current_released_at"]
            )

    def test_append_rejects_equal_or_older_released_at(self) -> None:
        for watermark in (
            "2025-01-09T12:00:00Z",
            "2025-01-09T13:00:00+01:00",
            "2025-01-09T11:59:59Z",
        ):
            with self.subTest(watermark=watermark), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                previous = root / "previous"
                current = root / "current"
                _write_release(
                    previous, "2025-01-09",
                    released_at="2025-01-09T12:00:00Z",
                )
                _write_release(
                    current, "2025-01-09", append=True, released_at=watermark,
                )
                with self.assertRaisesRegex(ValueError, "advance released_at"):
                    validate_evidence_append(previous, current)

    def test_rejects_same_day_knowledge_after_released_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            _write_release(
                root, "2025-01-08", released_at="2025-01-08T12:00:00Z"
            )
            pressure_path = root / "field_day_pressure_v4.parquet"
            pressure = pd.read_parquet(pressure_path)
            pressure.loc[0, "knowledge_time"] = "2025-01-08T12:00:01Z"
            pressure.to_parquet(pressure_path, index=False)
            _refresh_signed_artifacts(root)

            with self.assertRaisesRegex(ValueError, "after released_at"):
                validate_evidence_directory(root)

    def test_rejects_naive_release_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            _write_release(
                root, "2025-01-08", released_at="2025-01-08T12:00:00"
            )
            with self.assertRaisesRegex(ValueError, "UTC offset"):
                validate_evidence_directory(root)

    def test_future_source_and_bad_reference_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            _write_release(root, "2025-01-08")
            s2 = pd.read_parquet(root / "field_s2_acquisition_v4.parquet")
            s2.loc[0, "spectral_source_date"] = "2025-01-09"
            s2.to_parquet(root / "field_s2_acquisition_v4.parquet", index=False)
            _refresh_signed_artifacts(root)
            with self.assertRaisesRegex(ValueError, "evidence time"):
                validate_evidence_directory(root)

    def test_rejects_pre_effective_knowledge_and_usable_unknown_qa(self) -> None:
        for mutation, error in (
            ("pressure_clock", "pressure knowledge"),
            ("crop_clock", "crop knowledge"),
            ("unknown_qa", "unknown QA"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "release"
                _write_release(root, "2025-01-08")
                if mutation == "pressure_clock":
                    path = root / "field_day_pressure_v4.parquet"
                    frame = pd.read_parquet(path)
                    frame.loc[0, "knowledge_time"] = "2025-01-07 23:59:59"
                elif mutation == "crop_clock":
                    path = root / "crop_day_context_v4.parquet"
                    frame = pd.read_parquet(path)
                    frame.loc[0, "knowledge_time"] = "2025-01-07 23:59:59"
                else:
                    path = root / "field_s2_acquisition_v4.parquet"
                    frame = pd.read_parquet(path)
                    frame["s2_good_observation"] = frame[
                        "s2_good_observation"
                    ].astype("boolean")
                    frame.loc[0, [
                        "valid_pixel_fraction",
                        "cloud_pct",
                        "s2_field_quality_flag",
                        "s2_good_observation",
                    ]] = None
                frame.to_parquet(path, index=False)
                _refresh_signed_artifacts(root)
                with self.assertRaisesRegex(ValueError, error):
                    validate_evidence_directory(root)


def _write_release(
    root: Path,
    as_of: str,
    *,
    append: bool = False,
    released_at: str | None = None,
) -> None:
    root.mkdir()
    dates = ["2025-01-08", "2025-01-09"] if append else ["2025-01-08"]
    common = {
        "field_id": "field-1",
        "crop_instance_id": "crop-1",
        "policy_version": "v4-test",
        "policy_sha256": "a" * 64,
        "availability_mode": "reconstructed",
    }
    pd.DataFrame([
        {
            **common, "observation_date": day, "knowledge_time": day,
            "stage_effective_date": day,
            "crop_name": "Maize", "stage_bucket": "vegetative",
        }
        for day in dates
    ]).to_parquet(root / "crop_day_context_v4.parquet", index=False)
    pd.DataFrame([
        {
            **common, "observation_date": day, "knowledge_time": day,
            "hazard_family": "heat", "pressure_score": 4.0,
            "pressure_observed": True,
        }
        for day in dates
    ]).to_parquet(root / "field_day_pressure_v4.parquet", index=False)
    acquisitions = [
        {
            **common, "acquisition_id": "acq-1",
            "spectral_source_date": "2025-01-08", "knowledge_time": "2025-01-08",
            "acquisition_attempted": True, "spectral_usable": True,
            "valid_pixel_fraction": 0.9, "cloud_pct": 5.0,
            "s2_field_quality_flag": "good", "s2_good_observation": True,
            "reference_acquisition_id": None,
        }
    ]
    if append:
        acquisitions.append(
            {
                **common, "acquisition_id": "acq-2",
                "spectral_source_date": "2025-01-09", "knowledge_time": "2025-01-09",
                "acquisition_attempted": True, "spectral_usable": True,
                "valid_pixel_fraction": 0.9, "cloud_pct": 5.0,
                "s2_field_quality_flag": "good", "s2_good_observation": True,
                "reference_acquisition_id": "acq-1",
            }
        )
    pd.DataFrame(acquisitions).to_parquet(
        root / "field_s2_acquisition_v4.parquet", index=False
    )
    manifest = {
        "run": {
            "status": "complete",
            "immutable": True,
            "release_as_of": as_of,
            "released_at": released_at or f"{as_of}T23:59:59Z",
            "source_generation_id": "generation-v4-test",
        },
        "correction_policy": CORRECTION_POLICY,
        "policy": {"version": "v4-test", "sha256": "a" * 64},
        "availability": {"mode": "reconstructed"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_signed_artifacts(root)


def _refresh_signed_artifacts(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {
        "crop": "crop_day_context_v4.parquet",
        "pressure": "field_day_pressure_v4.parquet",
        "s2": "field_s2_acquisition_v4.parquet",
    }
    artifacts = {}
    for label, filename in files.items():
        path = root / filename
        artifacts[label] = {
            "name": filename,
            "size_bytes": path.stat().st_size,
            "row_count": len(pd.read_parquet(path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
