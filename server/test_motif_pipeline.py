from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from build_story_map_bundle import build_bundle
from story_monitor.contracts import load_policy
from story_monitor.motif_export import export_motif_generation
from story_monitor.motifs import DiscoveryConfig, discover_motifs
from story_monitor.pipeline import DEFAULT_POLICY_PATH, build_generation
from story_monitor.prefix_features import load_training_prefixes


def monitor_row(day: str, *, risk: str, driver: str | None) -> dict[str, object]:
    return {
        "field_id": "field-A", "observation_date": day, "crop_name": "Maize",
        "crop_season": "Season A", "crop_stage": "Vegetative Growth",
        "risk_level": risk, "primary_risk_driver": driver, "spectral_echo_days": 0,
        "ndvi": 0.5, "ndmi": 0.2, "psri": 0.1, "spi_index": -0.2,
        "ponding_mm": 0.0, "temperature": 25.0, "apparent_temperature": 26.0,
        "humidity": 55.0, "wind_speed": 4.0,
    }


class MotifPipelineTests(unittest.TestCase):
    def test_concurrent_hazards_keep_event_specific_pressure(self) -> None:
        rows = [
            monitor_row("2025-01-01", risk="MED-HIGH", driver="heat"),
            monitor_row("2025-01-02", risk="MED-HIGH", driver="heat"),
            monitor_row("2025-01-03", risk="MED-HIGH", driver="drought"),
            monitor_row("2025-01-04", risk="MED-HIGH", driver="drought"),
            monitor_row("2025-01-05", risk="MED-HIGH", driver="heat"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "echo.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            generation = build_generation(
                input_parquet=source,
                output_dir=root / "monitor",
                as_of_date=date(2025, 1, 5),
                policy=load_policy(DEFAULT_POLICY_PATH),
                max_fields=10,
            )
            prefixes = load_training_prefixes(
                generation.generation_dir,
                through="2025-01-05",
                sample_age_buckets=False,
            )

        latest = prefixes.sort_values("observation_date").groupby("hazard_family").tail(1)
        risks = dict(zip(latest["hazard_family"], latest["current_risk_rank"]))
        self.assertEqual(risks["heat"], 3)
        self.assertEqual(risks["drought"], 0)

    def test_training_cutoff_excludes_partial_boundary_week(self) -> None:
        rows = []
        for timestamp in pd.date_range("2025-12-20", "2026-01-04", freq="D"):
            rows.append(
                monitor_row(timestamp.date().isoformat(), risk="MED-HIGH", driver="heat")
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "echo.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            generation = build_generation(
                input_parquet=source,
                output_dir=root / "monitor",
                as_of_date=date(2026, 1, 4),
                policy=load_policy(DEFAULT_POLICY_PATH),
                max_fields=10,
            )
            prefixes = load_training_prefixes(
                generation.generation_dir,
                through="2025-12-31",
                sample_age_buckets=False,
            )

        self.assertFalse(prefixes.empty)
        self.assertLessEqual(str(prefixes["observation_date"].max())[:10], "2025-12-28")
        self.assertNotIn("2025-12-29", set(prefixes["timeline_bucket"].astype(str).str[:10]))

    def test_prefix_discovery_export_and_bundle_are_compatible(self) -> None:
        rows = []
        fields = []
        for group in range(2):
            for field_number in range(8):
                field_id = f"field-{group}-{field_number}"
                fields.append(field_id)
                for offset in range(14):
                    pressure_days = 5 if group == 0 else 7
                    rows.append(
                        {
                            "field_id": field_id,
                            "observation_date": (
                                pd.Timestamp("2025-01-01") + pd.Timedelta(days=offset)
                            ).date().isoformat(),
                            "crop_name": "Maize",
                            "crop_season": "Season A",
                            "crop_stage": "Vegetative Growth",
                            "risk_level": (
                                "MED-HIGH" if group == 0 and offset < pressure_days
                                else "HIGH" if group == 1 and offset < pressure_days
                                else "LOW"
                            ),
                            "primary_risk_driver": "heat" if offset < pressure_days else None,
                            "spectral_echo_days": 0 if offset in {0, 8} else offset if offset < 8 else offset - 8,
                            "ndvi": 0.5 if group == 0 else 0.25,
                            "ndmi": 0.2,
                            "psri": 0.1,
                            "spi_index": -0.2,
                            "ponding_mm": 0.0,
                            "temperature": 25.0,
                            "apparent_temperature": 26.0,
                            "humidity": 55.0,
                            "wind_speed": 4.0,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "echo.parquet"
            geometry = root / "geometry.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            pd.DataFrame(
                [
                    {"field_id": field_id, "geometry_wkt": f"POINT ({30 + index * 0.001} -1)"}
                    for index, field_id in enumerate(fields)
                ]
            ).to_parquet(geometry, index=False)
            policy = load_policy(DEFAULT_POLICY_PATH)
            generation = build_generation(
                input_parquet=source,
                output_dir=root / "monitor",
                as_of_date=date(2025, 1, 14),
                policy=policy,
                max_fields=100,
                geometry_parquet=geometry,
            )
            prefixes = load_training_prefixes(
                generation.generation_dir,
                through="2025-01-14",
                sample_age_buckets=True,
            )
            model_dir = root / "model"
            manifest = discover_motifs(
                prefixes,
                model_dir,
                config=DiscoveryConfig(min_cluster_size=4, min_samples=2),
                training_cutoff="2025-01-14",
                policy_version=policy.version,
                policy_sha256=policy.source_sha256,
            )
            motif_generation = export_motif_generation(
                generation.generation_dir, model_dir, root / "motif-generation"
            )
            bundle = root / "bundle"
            build_bundle(motif_generation, bundle)

            frames = pd.read_parquet(bundle / "frame_fields.parquet")
            labels = pd.read_parquet(bundle / "cluster_labels.parquet")

        self.assertGreaterEqual(manifest["motif_count"], 2)
        self.assertTrue(frames["story_cluster_id"].str.startswith("motif:").all())
        self.assertEqual(set(frames["story_cluster_id"]), set(labels["story_cluster_id"]))
        self.assertTrue(labels["short_label"].str.contains("Heat", case=False).all())


if __name__ == "__main__":
    unittest.main()
