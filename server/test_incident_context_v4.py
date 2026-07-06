from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from story_monitor.incident_context_v4 import build_incident_context_v4
from story_monitor.incident_release_v4 import CORRECTION_POLICY
from story_monitor.incident_validation_v4 import validate_evidence_directory


RELEASED_AT = "2025-02-01T12:00:00Z"


class IncidentContextV4Tests(unittest.TestCase):
    def test_builds_separate_daily_and_acquisition_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "enriched.parquet"
            _write_enriched(enriched)
            evidence = root / "evidence"

            result = build_incident_context_v4(
                generation,
                evidence,
                released_at=RELEASED_AT,
                enriched_source_parquet=enriched,
                availability_mode="reconstructed",
                threads=1,
            )

            self.assertEqual(result["crop_day_count"], 4)
            self.assertEqual(result["pressure_day_hazard_count"], 16)
            self.assertEqual(result["spectral_acquisition_count"], 4)

            crop = pd.read_parquet(evidence / "crop_day_context_v4.parquet")
            pressure = pd.read_parquet(evidence / "field_day_pressure_v4.parquet")
            s2 = pd.read_parquet(evidence / "field_s2_acquisition_v4.parquet")
            self.assertEqual(set(crop["stage_bucket"]), {"flowering"})
            self.assertEqual(set(crop["availability_mode"]), {"reconstructed"})
            self.assertFalse(
                pressure.duplicated(
                    ["field_id", "crop_instance_id", "observation_date", "hazard_family"]
                ).any()
            )

            day = pressure[pressure["observation_date"].astype(str) == "2025-01-08"]
            heat = day[day["hazard_family"] == "heat"].iloc[0]
            wind = day[day["hazard_family"] == "damaging_wind"].iloc[0]
            drought = day[day["hazard_family"] == "drought"].iloc[0]
            ponding = day[day["hazard_family"] == "ponding_flooding"].iloc[0]
            self.assertTrue(heat["pressure_observed"])
            self.assertEqual(heat["pressure_rank"], 4)
            self.assertTrue(wind["pressure_observed"])
            self.assertEqual(wind["pressure_rank"], 3)
            self.assertFalse(drought["pressure_observed"])
            self.assertTrue(pd.isna(drought["pressure_rank"]))
            self.assertEqual(drought["pressure_missing_reason"], "required_weather_driver_missing")
            self.assertTrue(ponding["pressure_observed"])
            self.assertEqual(ponding["pressure_rank"], 0)
            self.assertEqual(ponding["pressure_band"], "NONE")
            self.assertTrue(pd.isna(ponding["pressure_missing_reason"]))

            self.assertFalse(s2.duplicated(["field_id", "spectral_source_date"]).any())
            by_date = {
                str(row.spectral_source_date)[:10]: row
                for row in s2.itertuples(index=False)
            }
            self.assertEqual(by_date["2025-01-08"].response_class, "medium_decline")
            self.assertEqual(by_date["2025-01-15"].acquisition_status, "rejected_cloud")
            # Parquet preserves this as a null; pandas may materialize a null
            # string column as either ``None``, ``pd.NA``, or ``NaN``.
            self.assertTrue(
                pd.isna(by_date["2025-01-15"].reference_acquisition_id)
            )
            self.assertFalse(by_date["2025-01-15"].new_response_evidence)
            self.assertEqual(
                str(by_date["2025-01-22"].reference_source_date)[:10], "2025-01-08"
            )
            self.assertEqual(by_date["2025-01-22"].response_class, "recovery")
            self.assertTrue(by_date["2025-01-22"].new_response_evidence)

            manifest = json.loads((evidence / "manifest.json").read_text())
            self.assertEqual(manifest["run"]["release_as_of"], "2025-01-31")
            self.assertEqual(
                manifest["run"]["released_at"], "2025-02-01T12:00:00.000000Z"
            )
            self.assertEqual(manifest["correction_policy"], CORRECTION_POLICY)
            self.assertEqual(manifest["run"]["as_of_date"], "2025-01-31")
            self.assertEqual(manifest["availability"]["mode"], "reconstructed")
            self.assertEqual(
                manifest["enriched_source_contract"]["availability_mode"],
                "reconstructed",
            )
            self.assertTrue(
                manifest["reconciliation"]["source_field_day"]["exact_key_coverage"]
            )
            self.assertEqual(
                validate_evidence_directory(evidence)["availability_mode"],
                "reconstructed",
            )
            self.assertTrue(manifest["semantics"]["missing_weather_is_not_zero_pressure"])
            with self.assertRaises(FileExistsError):
                build_incident_context_v4(
                    generation, evidence, released_at=RELEASED_AT, threads=1
                )

    def test_strict_mode_uses_explicit_availability_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "strict.parquet"
            _write_enriched(enriched, strict=True)
            evidence = root / "strict-evidence"

            build_incident_context_v4(
                generation,
                evidence,
                released_at=RELEASED_AT,
                enriched_source_parquet=enriched,
                availability_mode="strict",
                threads=1,
            )

            crop = pd.read_parquet(evidence / "crop_day_context_v4.parquet")
            pressure = pd.read_parquet(evidence / "field_day_pressure_v4.parquet")
            s2 = pd.read_parquet(evidence / "field_s2_acquisition_v4.parquet")
            self.assertEqual(set(crop["availability_mode"]), {"strict"})
            self.assertTrue((crop["knowledge_time"].dt.hour == 7).all())
            self.assertTrue((pressure["knowledge_time"].dt.hour == 6).all())
            self.assertTrue((s2["knowledge_time"].dt.hour == 10).all())

    def test_strict_mode_rejects_weather_or_stage_known_before_effective_day(self) -> None:
        for column in ("weather_available_at", "stage_available_at"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generation = root / "generation"
                generation.mkdir()
                _write_generation(generation)
                enriched = root / "strict-invalid.parquet"
                frame = _enriched_frame(strict=True)
                frame.loc[1, column] = "2025-01-07 23:59:59"
                frame.to_parquet(enriched, index=False)
                _write_enriched_manifest(enriched, "strict")

                with self.assertRaisesRegex(ValueError, "missing or impossible timestamps"):
                    build_incident_context_v4(
                        generation,
                        root / "must-not-exist",
                        released_at=RELEASED_AT,
                        enriched_source_parquet=enriched,
                        availability_mode="strict",
                        threads=1,
                    )
                self.assertFalse((root / "must-not-exist").exists())

    def test_unknown_acquisition_qa_is_rejected_not_used_as_crop_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "qa-unknown.parquet"
            frame = _enriched_frame()
            frame["s2_good_observation"] = frame["s2_good_observation"].astype(
                "boolean"
            )
            frame.loc[0, [
                "valid_pixel_fraction",
                "cloud_pct",
                "s2_field_quality_flag",
                "s2_good_observation",
            ]] = None
            frame.to_parquet(enriched, index=False)
            _write_enriched_manifest(enriched, "reconstructed")
            evidence = root / "evidence"

            build_incident_context_v4(
                generation,
                evidence,
                released_at=RELEASED_AT,
                enriched_source_parquet=enriched,
                availability_mode="reconstructed",
                threads=1,
            )

            acquisitions = pd.read_parquet(
                evidence / "field_s2_acquisition_v4.parquet"
            )
            unknown = acquisitions[
                acquisitions["spectral_source_date"].astype(str).eq("2025-01-01")
            ].iloc[0]
            self.assertEqual(unknown["acquisition_status"], "rejected_qa_unknown")
            self.assertFalse(unknown["spectral_usable"])
            self.assertFalse(unknown["new_response_evidence"])
            self.assertTrue(pd.isna(unknown["reference_acquisition_id"]))

    def test_external_attempts_augment_derived_history_and_assign_crop_causally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "enriched.parquet"
            enriched_frame = _enriched_frame(strict=True)
            enriched_frame.loc[
                enriched_frame["observation_date"].eq("2025-01-08"),
                "stage_available_at",
            ] = "2025-01-14 07:00:00"
            enriched_frame.to_parquet(enriched, index=False)
            _write_enriched_manifest(enriched, "strict")
            attempts = root / "attempts.parquet"
            rejected = {
                "field_id": "field-1",
                "sentinel_observation_date": "2025-01-12",
                "first_seen_observation_date": "2025-01-13",
                "spectral_available_at": "2025-01-13 10:00:00",
                "valid_pixel_fraction": 0.1,
                "cloud_pct": 99.0,
                "s2_field_quality_flag": "cloudy",
                "s2_good_observation": False,
                "ndvi": None,
                "ndmi": None,
                "psri": None,
            }
            pd.DataFrame(
                [
                    rejected,
                    dict(rejected),
                    {
                        "field_id": "field-1",
                        "sentinel_observation_date": "2025-01-15",
                        "first_seen_observation_date": "2025-01-15",
                        "spectral_available_at": "2025-01-15 10:00:00",
                        "valid_pixel_fraction": 0.95,
                        "cloud_pct": 1.0,
                        "s2_field_quality_flag": "good",
                        "s2_good_observation": True,
                        "ndvi": None,
                        "ndmi": None,
                        "psri": None,
                    },
                ]
            ).to_parquet(attempts, index=False)

            evidence = root / "evidence"
            result = build_incident_context_v4(
                generation,
                evidence,
                released_at=RELEASED_AT,
                enriched_source_parquet=enriched,
                acquisition_parquet=attempts,
                availability_mode="strict",
                threads=1,
            )

            self.assertEqual(result["spectral_acquisition_count"], 5)
            acquisitions = pd.read_parquet(
                evidence / "field_s2_acquisition_v4.parquet"
            )
            self.assertEqual(
                set(acquisitions["spectral_source_date"].astype(str)),
                {
                    "2025-01-01", "2025-01-08", "2025-01-12",
                    "2025-01-15", "2025-01-22",
                },
            )
            unmatched = acquisitions[
                acquisitions["spectral_source_date"].astype(str).eq("2025-01-12")
            ].iloc[0]
            self.assertEqual(unmatched["acquisition_origin"], "external_attempt")
            self.assertEqual(unmatched["acquisition_status"], "rejected_no_valid_pixels")
            self.assertEqual(
                unmatched["crop_instance_id"], "field-1::maize::2025-A"
            )
            self.assertEqual(
                str(unmatched["crop_assignment_effective_date"])[:10], "2025-01-08"
            )
            self.assertEqual(str(unmatched["knowledge_time"])[:10], "2025-01-14")
            overlap = acquisitions[
                acquisitions["spectral_source_date"].astype(str).eq("2025-01-15")
            ].iloc[0]
            self.assertEqual(overlap["acquisition_origin"], "external_and_derived")
            self.assertTrue(overlap["spectral_usable"])

    def test_future_echo_mismatch_and_missing_strict_time_fail_atomically(self) -> None:
        mutations = (
            (
                "echo_mismatch",
                lambda frame: frame.__setitem__("spectral_echo_days", [0, 2, 0, 0]),
                "echo_mismatch=1",
                "reconstructed",
            ),
            (
                "future_source",
                lambda frame: frame.__setitem__(
                    "sentinel_observation_date",
                    ["2025-01-01", "2025-01-30", "2025-01-15", "2025-01-22"],
                ),
                "future_spectral_source=1",
                "reconstructed",
            ),
            (
                "strict_missing_time",
                lambda frame: frame.drop(
                    columns=[
                        "weather_available_at", "stage_available_at",
                        "spectral_available_at",
                    ],
                    inplace=True,
                ),
                "availability columns",
                "strict",
            ),
        )
        for label, mutate, error, mode in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generation = root / "generation"
                generation.mkdir()
                _write_generation(generation)
                enriched = root / "enriched.parquet"
                frame = _enriched_frame()
                mutate(frame)
                frame.to_parquet(enriched, index=False)
                _write_enriched_manifest(enriched, mode)
                evidence = root / "evidence"
                with self.assertRaisesRegex(ValueError, error):
                    build_incident_context_v4(
                        generation,
                        evidence,
                        released_at=RELEASED_AT,
                        enriched_source_parquet=enriched,
                        availability_mode=mode,
                        threads=1,
                    )
                self.assertFalse(evidence.exists())

    def test_rejects_availability_laundering_and_inexact_enriched_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "reconstructed.parquet"
            _write_enriched(enriched)

            with self.assertRaisesRegex(ValueError, "does not match"):
                build_incident_context_v4(
                    generation,
                    root / "strict-must-not-exist",
                    released_at=RELEASED_AT,
                    enriched_source_parquet=enriched,
                    availability_mode="strict",
                    threads=1,
                )

            partial = root / "partial.parquet"
            _enriched_frame().iloc[:-1].to_parquet(partial, index=False)
            _write_enriched_manifest(partial, "reconstructed")
            with self.assertRaisesRegex(ValueError, "exactly cover.*missing=1"):
                build_incident_context_v4(
                    generation,
                    root / "partial-must-not-exist",
                    released_at=RELEASED_AT,
                    enriched_source_parquet=partial,
                    availability_mode="reconstructed",
                    threads=1,
                )

            extra = root / "extra.parquet"
            extra_frame = _enriched_frame()
            extra_row = extra_frame.iloc[-1].copy()
            extra_row["field_id"] = "field-not-in-generation"
            pd.concat(
                [extra_frame, pd.DataFrame([extra_row])], ignore_index=True
            ).to_parquet(extra, index=False)
            _write_enriched_manifest(extra, "reconstructed")
            with self.assertRaisesRegex(ValueError, "exactly cover.*extra=1"):
                build_incident_context_v4(
                    generation,
                    root / "extra-must-not-exist",
                    released_at=RELEASED_AT,
                    enriched_source_parquet=extra,
                    availability_mode="reconstructed",
                    threads=1,
                )

    def test_rejects_enriched_availability_after_release_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            enriched = root / "post-release.parquet"
            frame = _enriched_frame(strict=True)
            frame.loc[0, "weather_available_at"] = "2025-02-01T12:00:01Z"
            frame.to_parquet(enriched, index=False)
            _write_enriched_manifest(enriched, "strict")

            with self.assertRaisesRegex(ValueError, "after released_at"):
                build_incident_context_v4(
                    generation,
                    root / "must-not-exist",
                    released_at=RELEASED_AT,
                    enriched_source_parquet=enriched,
                    availability_mode="strict",
                    threads=1,
                )

def _write_generation(root: Path) -> None:
    dates = ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]
    pd.DataFrame(
        [
            {
                "field_id": "field-1",
                "observation_date": day,
                "crop_name": "Maize",
                "crop_season": "2025-A",
                "crop_stage": "Flowering",
                "stage_family": "Flowering",
                "crop_instance_id": "field-1::maize::2025-A",
                "pressure_observed": True,
                "risk_rank": 4,
                "risk_band": "HIGH",
                "hazard_family": "heat",
                "ndvi": ndvi,
                "ndmi": ndmi,
                "psri": psri,
            }
            for day, ndvi, ndmi, psri in zip(
                dates,
                (0.70, 0.60, 0.58, 0.68),
                (0.40, 0.33, 0.31, 0.39),
                (0.10, 0.16, 0.18, 0.11),
            )
        ]
    ).to_parquet(root / "daily_causal_signals.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "generation_id": "generation-v4-test",
                    "as_of_date": "2025-01-31",
                }
            }
        ),
        encoding="utf-8",
    )


def _enriched_frame(*, strict: bool = False) -> pd.DataFrame:
    dates = ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]
    frame = pd.DataFrame(
        [
            {
                "field_id": "field-1",
                "observation_date": day,
                "sentinel_observation_date": day,
                "spectral_echo_days": 0,
                "valid_pixel_fraction": valid,
                "cloud_pct": cloud,
                "s2_field_quality_flag": "good",
                "s2_good_observation": True,
                "ndvi": ndvi,
                "ndmi": ndmi,
                "psri": psri,
                "drought_risk_score": 5.0,
                "ponding_risk_score": 0.0,
                "heat_risk_score": 9.0,
                "wind_risk_score": 6.0,
                "spi_index": None,
                "ponding_mm": 0.0,
                "apparent_temperature": 38.0,
                "temperature": 35.0,
                "humidity": 45.0,
                "wind_speed": 30.0,
                "wind_gust": 50.0,
                "season_calendar_source": "test-calendar",
                "planting_date": "2024-11-01",
            }
            for day, ndvi, ndmi, psri, valid, cloud in zip(
                dates,
                (0.70, 0.60, 0.58, 0.68),
                (0.40, 0.33, 0.31, 0.39),
                (0.10, 0.16, 0.18, 0.11),
                (0.9, 0.9, 0.9, 0.9),
                (2.0, 3.0, 90.0, 4.0),
            )
        ]
    )
    frame["weather_available_at"] = [f"{day} 00:00:00" for day in dates]
    frame["stage_available_at"] = [f"{day} 00:00:00" for day in dates]
    frame["spectral_available_at"] = [f"{day} 00:00:00" for day in dates]
    if strict:
        frame["weather_available_at"] = [f"{day} 06:00:00" for day in dates]
        frame["stage_available_at"] = [f"{day} 07:00:00" for day in dates]
        frame["spectral_available_at"] = [f"{day} 10:00:00" for day in dates]
    return frame


def _write_enriched(path: Path, *, strict: bool = False) -> None:
    _enriched_frame(strict=strict).to_parquet(path, index=False)
    _write_enriched_manifest(path, "strict" if strict else "reconstructed")


def _write_enriched_manifest(path: Path, mode: str) -> None:
    payload = {
        "schema_version": "incident-enriched-source-v4/1",
        "status": "complete",
        "immutable": True,
        "released_at": RELEASED_AT,
        "correction_policy": CORRECTION_POLICY,
        "availability": {
            "mode": mode,
            "diagnostic_reconstruction": mode == "reconstructed",
        },
        "output": {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    unittest.main()
