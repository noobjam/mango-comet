from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.contracts import load_policy
from story_monitor.partitioned_pipeline import PartitionOptions, build_partitioned_generation
from story_monitor.pipeline import DEFAULT_POLICY_PATH, build_generation


def source_row(field_id: str, day: str, risk: str, driver: str | None, echo: int) -> dict[str, object]:
    return {
        "field_id": field_id,
        "observation_date": day,
        "crop_name": "Maize",
        "crop_season": "Season A",
        "crop_stage": "Vegetative Growth",
        "risk_level": risk,
        "primary_risk_driver": driver,
        "spectral_echo_days": echo,
        "ndvi": 0.5,
        "ndmi": 0.2,
        "psri": 0.1,
        "spi_index": -0.2,
        "ponding_mm": 0.0,
        "temperature": 25.0,
        "apparent_temperature": 26.0,
        "humidity": 55.0,
        "wind_speed": 4.0,
    }


class PartitionedGenerationTests(unittest.TestCase):
    def test_partitioned_generation_rejects_negative_echo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "echo.parquet"
            pd.DataFrame(
                [source_row("field-A", "2025-01-01", "HIGH", "heat", -1)]
            ).to_parquet(source, index=False)
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                build_partitioned_generation(
                    input_parquet=source,
                    output_dir=root / "partitioned",
                    as_of_date=date(2025, 1, 1),
                    policy=load_policy(DEFAULT_POLICY_PATH),
                    options=PartitionOptions(partitions=1, workers=1, threads=1),
                )

    def test_partitioned_generation_matches_bounded_generation(self) -> None:
        rows = []
        for field_id in ("field-A", "field-B", "field-C", "field-D"):
            for offset in range(9):
                day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=offset)).date().isoformat()
                rows.append(
                    source_row(
                        field_id,
                        day,
                        "MED-HIGH" if offset < 2 else "LOW",
                        "heat" if offset < 2 else None,
                        0 if offset in {0, 8} else offset,
                    )
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "echo.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            policy = load_policy(DEFAULT_POLICY_PATH)
            bounded = build_generation(
                input_parquet=source,
                output_dir=root / "bounded",
                as_of_date=date(2025, 1, 9),
                policy=policy,
                max_fields=10,
            )
            partitioned = build_partitioned_generation(
                input_parquet=source,
                output_dir=root / "partitioned",
                as_of_date=date(2025, 1, 9),
                policy=policy,
                options=PartitionOptions(partitions=4, workers=2, threads=2),
            )
            for name, keys in {
                "daily_causal_signals.parquet": ["field_id", "observation_date"],
                "event_windows.parquet": ["field_id", "event_start_date", "event_id"],
                "event_state_snapshots.parquet": ["timeline_bucket", "field_id", "event_id"],
                "map_frame_fields.parquet": ["timeline_bucket", "field_id", "event_id"],
            }.items():
                expected = pd.read_parquet(bounded.generation_dir / name).sort_values(keys).reset_index(drop=True)
                actual = pd.read_parquet(partitioned.generation_dir / name).sort_values(keys).reset_index(drop=True)
                assert_frame_equal(expected, actual, check_dtype=False)

        self.assertEqual(partitioned.row_count, len(rows))
        self.assertEqual(partitioned.event_count, bounded.event_count)


if __name__ == "__main__":
    unittest.main()
