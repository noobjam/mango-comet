from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prepare_incident_source_v4 import prepare_incident_source_v4
from story_monitor.incident_release_v4 import CORRECTION_POLICY


RELEASED_AT = "2025-02-01T12:00:00Z"


class PrepareIncidentSourceV4Tests(unittest.TestCase):
    def test_enriches_echo_source_with_weather_scores_and_acquisition_qa(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deliverable = root / "echo.parquet"
            full = root / "full.parquet"
            acquisition = root / "acquisition.parquet"
            output = root / "enriched.parquet"
            _deliverable_frame().to_parquet(deliverable, index=False)
            _full_frame().to_parquet(full, index=False)
            pd.DataFrame(
                [
                    {
                        "field_id": "field-1",
                        "sentinel_observation_date": "2025-01-01",
                        "valid_pixel_fraction": 0.91,
                        "cloud_pct": 2.0,
                        "s2_field_quality_flag": "acquisition-good",
                        "s2_good_observation": True,
                    }
                ]
            ).to_parquet(acquisition, index=False)

            result = prepare_incident_source_v4(
                deliverable,
                (full,),
                output,
                released_at=RELEASED_AT,
                acquisition_sources=(acquisition,),
                availability_mode="reconstructed",
                threads=1,
            )

            self.assertEqual(result["row_count"], 2)
            enriched = pd.read_parquet(output)
            self.assertEqual(len(enriched), 2)
            self.assertEqual(enriched["spectral_echo_days"].tolist(), [0, 1])
            self.assertEqual(enriched["sentinel_days_stale"].tolist(), [0, 1])
            self.assertEqual(set(enriched["heat_risk_score"]), {8.5})
            self.assertEqual(set(enriched["wind_risk_score"]), {6.0})
            self.assertEqual(set(enriched["valid_pixel_fraction"]), {0.91})
            self.assertEqual(set(enriched["s2_field_quality_flag"]), {"acquisition-good"})
            self.assertTrue(enriched["availability_reconstructed"].all())
            self.assertTrue(enriched["source_row_present"].all())
            self.assertEqual(
                enriched["spectral_available_at"].dt.strftime("%Y-%m-%d").tolist(),
                ["2025-01-01", "2025-01-01"],
            )

            manifest_path = Path(result["manifest"])
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["availability"]["mode"], "reconstructed")
            self.assertEqual(manifest["released_at"], "2025-02-01T12:00:00.000000Z")
            self.assertEqual(manifest["correction_policy"], CORRECTION_POLICY)
            self.assertTrue(manifest["availability"]["diagnostic_reconstruction"])
            self.assertEqual(manifest["output"]["row_count"], 2)
            self.assertEqual(
                manifest["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            with self.assertRaises(FileExistsError):
                prepare_incident_source_v4(
                    deliverable, (full,), output,
                    released_at=RELEASED_AT, threads=1,
                )

    def test_echo_mismatch_and_missing_strict_timestamps_fail_atomically(self) -> None:
        cases = (
            ("mismatch", "reconstructed", True, "echo_mismatch=1"),
            ("strict", "strict", False, "strict availability requires"),
        )
        for label, mode, mismatch, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                deliverable = root / "echo.parquet"
                full = root / "full.parquet"
                output = root / "enriched.parquet"
                deliverable_frame = _deliverable_frame()
                if mismatch:
                    deliverable_frame.loc[1, "spectral_echo_days"] = 2
                deliverable_frame.to_parquet(deliverable, index=False)
                _full_frame().to_parquet(full, index=False)

                with self.assertRaisesRegex(ValueError, error):
                    prepare_incident_source_v4(
                        deliverable,
                        (full,),
                        output,
                        released_at=RELEASED_AT,
                        availability_mode=mode,
                        threads=1,
                    )
                self.assertFalse(output.exists())
                self.assertFalse(
                    output.with_suffix(output.suffix + ".manifest.json").exists()
                )

    def test_strict_source_rejects_weather_or_stage_available_before_observation(self) -> None:
        for column in ("weather_available_at", "stage_available_at"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                deliverable = root / "echo.parquet"
                full = root / "full.parquet"
                output = root / "must-not-exist.parquet"
                _deliverable_frame().to_parquet(deliverable, index=False)
                frame = _full_frame()
                frame["weather_available_at"] = [
                    f"{day} 06:00:00" for day in frame["observation_date"]
                ]
                frame["stage_available_at"] = [
                    f"{day} 07:00:00" for day in frame["observation_date"]
                ]
                frame["spectral_available_at"] = [
                    f"{day} 10:00:00" for day in frame["observation_date"]
                ]
                frame.loc[1, column] = "2025-01-01 23:59:59"
                frame.to_parquet(full, index=False)

                with self.assertRaisesRegex(ValueError, "invalid timestamps"):
                    prepare_incident_source_v4(
                        deliverable,
                        (full,),
                        output,
                        released_at=RELEASED_AT,
                        availability_mode="strict",
                        threads=1,
                    )
                self.assertFalse(output.exists())

    def test_strict_source_rejects_same_day_availability_after_released_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deliverable = root / "echo.parquet"
            full = root / "full.parquet"
            output = root / "must-not-exist.parquet"
            _deliverable_frame().iloc[:1].to_parquet(deliverable, index=False)
            frame = _full_frame().iloc[:1].copy()
            frame["weather_available_at"] = [
                f"{day} 06:00:00" for day in frame["observation_date"]
            ]
            frame["stage_available_at"] = [
                f"{day} 07:00:00" for day in frame["observation_date"]
            ]
            frame["spectral_available_at"] = [
                f"{day} 10:00:00" for day in frame["observation_date"]
            ]
            frame.to_parquet(full, index=False)

            with self.assertRaisesRegex(ValueError, "post-release timestamps"):
                prepare_incident_source_v4(
                    deliverable,
                    (full,),
                    output,
                    released_at="2025-01-01T08:00:00Z",
                    availability_mode="strict",
                    threads=1,
                )
            self.assertFalse(output.exists())


def _deliverable_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_id": "field-1",
                "observation_date": "2025-01-01",
                "spectral_echo_days": 0,
                "crop_name": "Maize",
            },
            {
                "field_id": "field-1",
                "observation_date": "2025-01-02",
                "spectral_echo_days": 1,
                "crop_name": "Maize",
            },
        ]
    )


def _full_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_id": "field-1",
                "observation_date": day,
                "sentinel_observation_date": "2025-01-01",
                "sentinel_days_stale": stale,
                "valid_pixel_fraction": 0.5,
                "cloud_pct": 10.0,
                "s2_field_quality_flag": "full-good",
                "s2_good_observation": True,
                "ndvi": 0.6,
                "ndmi": 0.3,
                "psri": 0.1,
                "drought_risk_score": 2.0,
                "ponding_risk_score": 1.0,
                "heat_risk_score": 8.5,
                "wind_risk_score": 6.0,
                "drought_hazard_level": "LOW",
                "ponding_hazard_level": "LOW",
                "heatwave_category": "HIGH",
                "wind_hazard_level": "MED-HIGH",
                "spi_index": -0.5,
                "ponding_mm": 0.0,
                "apparent_temperature": 38.0,
                "temperature": 35.0,
                "humidity": 40.0,
                "wind_speed": 25.0,
                "wind_gust_kmh": 45.0,
                "season_calendar_source": "test-calendar",
                "planting_date": "2024-11-01",
            }
            for day, stale in (("2025-01-01", 0), ("2025-01-02", 1))
        ]
    )


if __name__ == "__main__":
    unittest.main()
