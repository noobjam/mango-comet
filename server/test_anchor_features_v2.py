from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.anchor_features_v2 import (
    FEATURE_SCHEMA,
    FEATURE_VERSION,
    MODEL_FEATURE_COLUMNS,
    build_event_anchors,
    write_event_anchors,
)


START = pd.Timestamp("2025-01-01")


def _day(number: int) -> str:
    return (START + pd.Timedelta(days=number - 1)).date().isoformat()


class AnchorFeatureV2Tests(unittest.TestCase):
    def _generation(self, root: Path) -> Path:
        generation = root / "generation"
        generation.mkdir()
        events: list[dict[str, object]] = []
        memberships: list[dict[str, object]] = []
        signals: list[dict[str, object]] = []

        def add_event(
            event_id: str,
            days: int,
            *,
            final_state: str,
            event_end: int | None,
            membership_state: str,
            hazard: str = "heat",
            missing_pressure: set[int] = frozenset(),
            responses: dict[int, str] | None = None,
            ranks: list[int] | None = None,
        ) -> None:
            field_id = f"field-{event_id}"
            instance_id = f"crop-{event_id}"
            events.append({
                "event_id": event_id,
                "field_id": field_id,
                "event_start_date": _day(1),
                "event_end_date": None if event_end is None else _day(event_end),
                "event_state": final_state,
                "hazard_signature": hazard,
                "right_censored": event_end is None,
            })
            responses = responses or {}
            ranks = ranks or [3] * days
            for number in range(1, days + 1):
                observed = number not in missing_pressure
                memberships.append({
                    "event_id": event_id,
                    "field_id": field_id,
                    "crop_instance_id": instance_id,
                    "observation_date": _day(number),
                    "event_state": membership_state,
                    "daily_pressure_rank": ranks[number - 1],
                    "daily_response_class": responses.get(number, "no_new_event_response"),
                    "pressure_observed": observed,
                })
                ndvi = ndmi = psri = None
                if event_id == "mature":
                    if number == 3:
                        ndvi, ndmi, psri = -0.1, -0.2, 0.1
                    elif number == 5:
                        ndvi, ndmi, psri = -0.3, -0.1, 0.4
                    elif number == 8:
                        ndvi, ndmi, psri = 0.2, 0.1, -0.2
                    elif number == 22:
                        ndvi, ndmi, psri = -9.0, -9.0, 9.0
                signals.append({
                    "field_id": field_id,
                    "crop_instance_id": instance_id,
                    "observation_date": _day(number),
                    "spectral_source_date": _day(number),
                    "ndvi_delta": ndvi,
                    "ndmi_delta": ndmi,
                    "psri_delta": psri,
                    "spi_index": -0.1 * number,
                    "ponding_mm": float(number),
                    "apparent_temperature": 99.0 if number == 22 else 40.0 if number == 20 else 30.0,
                    "temperature": 29.0,
                    "wind_speed": float(number),
                })

        mature_ranks = [3] * 10 + [4] * 5 + [2] * 6 + [4] * 4
        add_event(
            "mature", 25, final_state="ACTIVE", event_end=None,
            membership_state="ACTIVE", ranks=mature_ranks,
            responses={3: "medium_decline", 5: "severe_decline", 8: "recovery", 22: "severe_decline"},
        )
        add_event(
            "early", 10, final_state="CLOSED_RECOVERED", event_end=10,
            membership_state="ACTIVE", hazard="drought", missing_pressure={4, 6},
            responses={4: "medium_decline", 8: "recovery"},
        )
        add_event(
            "young", 10, final_state="ACTIVE", event_end=None,
            membership_state="ACTIVE",
        )
        add_event(
            "watch", 21, final_state="WATCH", event_end=None,
            membership_state="WATCH", ranks=[2] * 21,
        )
        add_event(
            "boundary", 10, final_state="CLOSED_SEASON_BOUNDARY", event_end=10,
            membership_state="ACTIVE",
        )

        pd.DataFrame(events).to_parquet(generation / "event_windows.parquet", index=False)
        pd.DataFrame(memberships).to_parquet(
            generation / "story_day_membership.parquet", index=False
        )
        pd.DataFrame(signals).to_parquet(
            generation / "daily_causal_signals.parquet", index=False
        )
        (generation / "manifest.json").write_text(
            json.dumps({"run": {"status": "complete", "as_of_date": _day(25)}}),
            encoding="utf-8",
        )
        return generation

    def test_anchor_outcomes_are_one_row_per_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            anchors = build_event_anchors(self._generation(Path(directory)), threads=1)

        self.assertEqual(len(anchors), 5)
        self.assertTrue(anchors["event_id"].is_unique)
        rows = anchors.set_index("event_id")
        self.assertEqual(rows.at["mature", "anchor_outcome"], "eligible")
        self.assertEqual(str(rows.at["mature", "anchor_date"])[:10], _day(21))
        self.assertEqual(rows.at["mature", "anchor_kind"], "day_21")
        self.assertEqual(rows.at["early", "anchor_outcome"], "eligible")
        self.assertEqual(rows.at["early", "anchor_kind"], "early_closure")
        self.assertEqual(rows.at["young", "anchor_outcome"], "insufficient_evidence")
        self.assertEqual(rows.at["watch", "anchor_outcome"], "watch_only")
        self.assertEqual(
            rows.at["boundary", "anchor_outcome"], "season_boundary_before_maturity"
        )
        self.assertTrue(rows.loc[["young", "watch", "boundary"], MODEL_FEATURE_COLUMNS].isna().all().all())

    def test_mature_features_stop_at_day_21(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            anchors = build_event_anchors(self._generation(Path(directory)), threads=1)
        row = anchors.set_index("event_id").loc["mature"]

        risks = np.asarray([3] * 10 + [4] * 5 + [2] * 6, dtype=float)
        normalized_time = np.arange(21, dtype=float) / 20.0
        self.assertAlmostEqual(row["peak_risk_rank"], 4.0)
        self.assertAlmostEqual(row["mean_risk_rank"], risks.mean())
        self.assertAlmostEqual(row["risk_slope"], np.polyfit(normalized_time, risks, 1)[0])
        self.assertAlmostEqual(row["elevated_day_fraction"], 15 / 21)
        self.assertAlmostEqual(row["high_day_fraction"], 5 / 21)
        self.assertAlmostEqual(row["longest_elevated_run_fraction"], 15 / 21)
        self.assertEqual(row["attributed_decline_any"], 1)
        self.assertEqual(row["attributed_severe_decline_any"], 1)
        self.assertAlmostEqual(row["attributed_decline_day_fraction"], 2 / 21)
        self.assertAlmostEqual(row["first_attributed_decline_position"], 2 / 20)
        self.assertEqual(row["attributed_recovery_after_decline"], 1)
        self.assertAlmostEqual(row["worst_attributed_ndvi_delta"], -0.3)
        self.assertAlmostEqual(row["worst_attributed_ndmi_delta"], -0.2)
        self.assertAlmostEqual(row["worst_attributed_psri_delta"], 0.4)
        self.assertAlmostEqual(row["hazard_intensity"], 40.0)
        self.assertAlmostEqual(row["usable_days_fraction"], 1.0)
        self.assertEqual(str(row["evidence_max_date"])[:10], _day(21))
        self.assertEqual(str(row["spectral_source_max_date"])[:10], _day(8))
        self.assertEqual(row["post_anchor_row_count"], 4)

    def test_early_closure_uses_usable_denominators_and_missing_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            anchors = build_event_anchors(self._generation(Path(directory)), threads=1)
        row = anchors.set_index("event_id").loc["early"]

        self.assertEqual(row["usable_day_count"], 8)
        self.assertEqual(row["missing_pressure_day_count"], 2)
        self.assertAlmostEqual(row["pressure_observed_coverage"], 0.8)
        self.assertAlmostEqual(row["usable_days_fraction"], 8 / 21)
        self.assertAlmostEqual(row["attributed_decline_day_fraction"], 1 / 8)
        self.assertEqual(row["attributed_recovery_after_decline"], 1)
        self.assertTrue(pd.isna(row["worst_attributed_ndvi_delta"]))
        self.assertEqual(row["worst_attributed_ndvi_delta_missing"], 1)
        self.assertAlmostEqual(row["hazard_intensity"], -1.0)

    def test_cutoff_hides_later_closure_and_copy_matches_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = self._generation(root)
            cutoff = build_event_anchors(generation, through=_day(5), threads=1)
            early = cutoff.set_index("event_id").loc["early"]
            self.assertEqual(early["anchor_outcome"], "insufficient_evidence")
            self.assertTrue(pd.isna(early["anchor_date"]))

            output = root / "anchors.parquet"
            spill = root / "duckdb-temp"
            write_event_anchors(
                generation, output, threads=1, memory_limit="256MB", temp_dir=spill
            )
            written = pd.read_parquet(output)
            direct = build_event_anchors(
                generation, threads=1, memory_limit="256MB", temp_dir=spill
            )
            for column in ("anchor_date", "evidence_max_date", "spectral_source_max_date"):
                written[column] = pd.to_datetime(written[column]).dt.date
                direct[column] = pd.to_datetime(direct[column]).dt.date
            assert_frame_equal(written, direct, check_dtype=False)
            with self.assertRaises(FileExistsError):
                write_event_anchors(generation, output, threads=1)

    def test_invalid_or_duplicate_usable_risk_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            path = generation / "story_day_membership.parquet"
            membership = pd.read_parquet(path)
            duplicate = membership[
                (membership["event_id"] == "mature")
                & (membership["observation_date"].astype(str).str[:10] == _day(1))
            ]
            pd.concat([membership, duplicate], ignore_index=True).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate usable"):
                build_event_anchors(generation, threads=1)

        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            path = generation / "story_day_membership.parquet"
            membership = pd.read_parquet(path)
            membership["daily_pressure_rank"] = membership["daily_pressure_rank"].astype(float)
            membership.loc[membership.index[0], "daily_pressure_rank"] = float("inf")
            membership.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, r"finite \[0, 4\]"):
                build_event_anchors(generation, threads=1)

    def test_future_spectral_source_date_fails_leakage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            path = generation / "daily_causal_signals.parquet"
            signals = pd.read_parquet(path)
            mask = (
                signals["field_id"].eq("field-mature")
                & signals["observation_date"].astype(str).str[:10].eq(_day(3))
            )
            signals.loc[mask, "spectral_source_date"] = _day(25)
            signals.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "post-anchor evidence"):
                build_event_anchors(generation, threads=1)

    def test_missing_event_signal_lineage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            path = generation / "daily_causal_signals.parquet"
            signals = pd.read_parquet(path).iloc[1:].copy()
            signals.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing matching causal-signal lineage"):
                build_event_anchors(generation, threads=1)

    def test_schema_is_serializable_and_matches_columns(self) -> None:
        self.assertEqual(FEATURE_SCHEMA["version"], FEATURE_VERSION)
        self.assertEqual(
            tuple(item["name"] for item in FEATURE_SCHEMA["features"]),
            MODEL_FEATURE_COLUMNS,
        )
        json.dumps(FEATURE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
