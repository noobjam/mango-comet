from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.causal_features import prepare_causal_signals
from story_monitor.contracts import load_policy
from story_monitor.pipeline import build_generation, read_canonical_input
from story_monitor.state_machine import run_state_machine


POLICY_PATH = Path(__file__).resolve().parent / "story_monitor" / "policies" / "onset_policy_v1.json"


def _source_row(
    day: str,
    *,
    field_id: str = "field-A",
    crop: str = "Maize",
    season: str = "Season A",
    stage: str = "Vegetative",
    risk: str = "LOW",
    driver: str | None = None,
    echo: int | None = 0,
    ndvi: float = 0.5,
    ndmi: float = 0.2,
    psri: float = 0.1,
) -> dict[str, object]:
    return {
        "field_id": field_id,
        "observation_date": day,
        "crop_name": crop,
        "crop_season": season,
        "crop_stage": stage,
        "risk_level": risk,
        "primary_risk_driver": driver,
        "spectral_echo_days": echo,
        "ndvi": ndvi,
        "ndmi": ndmi,
        "psri": psri,
        "spi_index": None,
        "ponding_mm": None,
        "temperature": 24.0,
        "apparent_temperature": 25.0,
        "humidity": 55.0,
        "wind_speed": 4.0,
    }


class CausalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(POLICY_PATH)

    def test_echo_rows_do_not_become_new_response_evidence(self) -> None:
        base = pd.DataFrame(
            [
                _source_row("2025-01-01", echo=0),
                _source_row("2025-01-02", echo=1),
                _source_row("2025-01-08", echo=0, ndvi=0.40, ndmi=0.10, psri=0.20),
                _source_row("2025-01-09", echo=1, ndvi=0.40, ndmi=0.10, psri=0.20),
            ]
        )
        signals = prepare_causal_signals(base, self.policy)

        self.assertEqual(signals["is_new_acquisition"].tolist(), [True, False, True, False])
        self.assertEqual(signals.iloc[1]["response_class"], "no_new_acquisition")
        self.assertEqual(signals.iloc[2]["response_class"], "severe_decline")
        self.assertFalse(bool(signals.iloc[3]["new_response_evidence"]))

        appended = pd.concat(
            [base, pd.DataFrame([_source_row("2025-01-16", echo=0, ndvi=0.7)])],
            ignore_index=True,
        )
        replayed = prepare_causal_signals(appended, self.policy).iloc[: len(signals)]
        columns = [
            "spectral_source_date", "reference_source_date", "ndvi_delta", "ndmi_delta",
            "psri_delta", "response_class", "new_response_evidence", "crop_instance_id",
        ]
        assert_frame_equal(
            signals[columns].reset_index(drop=True),
            replayed[columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_crop_instances_follow_contiguous_field_regimes(self) -> None:
        frame = pd.DataFrame(
            [
                _source_row("2025-01-01", crop="Maize", season="A"),
                _source_row("2025-01-02", crop="Beans", season="B"),
                _source_row("2025-01-03", crop="Maize", season="A"),
            ]
        )
        signals = prepare_causal_signals(frame, self.policy)
        self.assertEqual(signals["crop_instance_id"].nunique(), 3)
        self.assertNotEqual(signals.iloc[0]["crop_instance_id"], signals.iloc[2]["crop_instance_id"])

    def test_negative_spectral_echo_is_rejected_before_date_derivation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            prepare_causal_signals(
                pd.DataFrame([_source_row("2025-01-01", echo=-1)]), self.policy
            )


class StateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(POLICY_PATH)

    def _signals(self, risks: list[str], drivers: list[str | None]) -> pd.DataFrame:
        rows = []
        for offset, (risk, driver) in enumerate(zip(risks, drivers)):
            day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=offset)).date().isoformat()
            rows.append(_source_row(day, risk=risk, driver=driver, echo=offset))
        return prepare_causal_signals(pd.DataFrame(rows), self.policy)

    def _direct_signals(
        self, entries: list[tuple[str, int, str | None]]
    ) -> pd.DataFrame:
        rows = []
        for offset, (hazard, rank, response) in enumerate(entries):
            rows.append(
                {
                    "field_id": "field-A",
                    "crop_name": "Maize",
                    "crop_season": "A",
                    "crop_instance_id": "crop-instance-A",
                    "crop_instance_start_date": date(2025, 1, 1),
                    "observation_date": date(2025, 1, 1) + pd.Timedelta(days=offset),
                    "hazard_family": hazard,
                    "risk_rank": rank,
                    "pressure_observed": True,
                    "stage_family": "vegetative",
                    "response_class": response or "no_new_acquisition",
                    "new_response_evidence": response is not None,
                }
            )
        return pd.DataFrame(rows)

    def test_low_med_stays_watch_without_aligned_response(self) -> None:
        signals = self._signals(["LOW-MED"] * 6, ["heat"] * 6)
        result = run_state_machine(signals, self.policy, as_of_date=date(2025, 1, 6))
        self.assertEqual(result.events.iloc[0]["event_state"], "WATCH")
        self.assertNotIn("ACTIVE", set(result.daily_records["event_state"]))
        self.assertTrue(bool(result.events.iloc[0]["right_censored"]))

    def test_quiet_event_closes_without_waiting_for_a_future_event(self) -> None:
        risks = ["MED-HIGH", "MED-HIGH", "LOW", "LOW", "LOW", "LOW"]
        drivers = ["heat", "heat", None, None, None, None]
        result = run_state_machine(
            self._signals(risks, drivers), self.policy, as_of_date=date(2025, 1, 6)
        )
        event = result.events.iloc[0]
        self.assertEqual(event["event_state"], "CLOSED_PRESSURE_QUIET_UNCONFIRMED")
        self.assertEqual(str(event["event_end_date"]), "2025-01-06")
        self.assertFalse(bool(event["right_censored"]))

    def test_null_risk_freezes_onset_history_and_open_event_quiet_clock(self) -> None:
        onset = prepare_causal_signals(
            pd.DataFrame(
                [
                    _source_row("2025-01-01", risk="LOW-MED", driver="heat"),
                    _source_row("2025-01-02", risk=None, driver="heat", echo=1),
                    _source_row("2025-01-03", risk="", driver="heat", echo=2),
                    _source_row("2025-01-04", risk="LOW-MED", driver="heat", echo=3),
                ]
            ),
            self.policy,
        )
        onset_result = run_state_machine(onset, self.policy, as_of_date=date(2025, 1, 4))
        self.assertEqual(str(onset_result.events.iloc[0]["event_start_date"]), "2025-01-04")

        active = prepare_causal_signals(
            pd.DataFrame(
                [
                    _source_row("2025-01-01", risk="MED-HIGH", driver="heat"),
                    _source_row("2025-01-02", risk="MED-HIGH", driver="heat", echo=1),
                    _source_row("2025-01-03", risk=None, driver="heat", echo=2),
                    _source_row("2025-01-04", risk="", driver="heat", echo=3),
                    _source_row("2025-01-05", risk="LOW", driver=None, echo=4),
                ]
            ),
            self.policy,
        )
        result = run_state_machine(active, self.policy, as_of_date=date(2025, 1, 5))
        gaps = result.daily_records[result.daily_records["event_state"] == "DATA_GAP"]
        self.assertEqual(len(gaps), 2)
        self.assertTrue((~gaps["pressure_observed"]).all())
        self.assertTrue((gaps["current_risk_band"] == "UNKNOWN").all())
        self.assertEqual(result.events.iloc[0]["event_state"], "QUIET_PENDING")

    def test_one_med_high_day_does_not_promote_watch(self) -> None:
        signals = self._signals(
            ["LOW-MED", "LOW-MED", "MED-HIGH"], ["heat", "heat", "heat"]
        )
        result = run_state_machine(signals, self.policy, as_of_date=date(2025, 1, 3))
        self.assertEqual(result.events.iloc[0]["event_state"], "WATCH")
        self.assertEqual(result.daily_records.iloc[-1]["event_state"], "WATCH")

    def test_aligned_severe_response_escalates_existing_event(self) -> None:
        result = run_state_machine(
            self._direct_signals(
                [("heat", 3, None), ("heat", 3, None), ("heat", 2, "severe_decline")]
            ),
            self.policy,
            as_of_date=date(2025, 1, 3),
        )
        heat_rows = result.daily_records[result.daily_records["hazard_signature"] == "heat"]
        self.assertEqual(heat_rows.iloc[-1]["event_state"], "SEVERE")
        self.assertTrue(bool(heat_rows.iloc[-1]["requires_review"]))

    def test_fresh_severe_response_is_visible_when_pressure_is_missing(self) -> None:
        signals = self._direct_signals([("none", 0, "severe_decline")])
        signals.loc[signals.index[0], "pressure_observed"] = False
        result = run_state_machine(signals, self.policy, as_of_date=date(2025, 1, 1))
        row = result.daily_records.iloc[0]
        self.assertEqual(row["event_state"], "SEVERE")
        self.assertEqual(row["daily_response_class"], "severe_decline")
        self.assertTrue(bool(row["requires_review"]))

    def test_attributed_recovery_does_not_contaminate_concurrent_hazard(self) -> None:
        signals = self._direct_signals(
            [
                ("heat", 3, None),
                ("heat", 3, None),
                ("drought", 3, None),
                ("drought", 3, None),
                ("heat", 2, "medium_decline"),
                ("heat", 2, "medium_decline"),
                ("none", 1, "recovery"),
            ]
        )
        signals.loc[signals.index[-1], "pressure_observed"] = False
        result = run_state_machine(signals, self.policy, as_of_date=date(2025, 1, 7))
        heat = result.events[result.events["hazard_signature"] == "heat"].iloc[0]
        drought = result.events[result.events["hazard_signature"] == "drought"].iloc[0]
        self.assertEqual(heat["event_state"], "CLOSED_RECOVERED")
        self.assertEqual(int(heat["response_day_count"]), 3)
        self.assertNotEqual(drought["event_state"], "CLOSED_RECOVERED")
        self.assertEqual(int(drought["response_day_count"]), 0)
        drought_rows = result.daily_records[result.daily_records["hazard_signature"] == "drought"]
        self.assertEqual(drought_rows.iloc[-1]["event_state"], "DATA_GAP")
        self.assertEqual(drought_rows.iloc[-1]["daily_response_class"], "no_new_event_response")

    def test_recovering_event_relapses_or_closes_unresolved(self) -> None:
        prefix = [
            ("heat", 3, None),
            ("heat", 3, None),
            ("drought", 3, None),
            ("drought", 3, None),
            ("heat", 2, "medium_decline"),
            ("heat", 2, "medium_decline"),
        ]
        relapse = run_state_machine(
            self._direct_signals(prefix + [("heat", 3, None)]),
            self.policy,
            as_of_date=date(2025, 1, 7),
        )
        heat = relapse.events[relapse.events["hazard_signature"] == "heat"].iloc[0]
        self.assertEqual(heat["event_state"], "ACTIVE")

        short_deadline = replace(self.policy, max_recovery_days=2)
        unresolved = run_state_machine(
            self._direct_signals(prefix + [("none", 1, None), ("none", 1, None)]),
            short_deadline,
            as_of_date=date(2025, 1, 8),
        )
        heat = unresolved.events[unresolved.events["hazard_signature"] == "heat"].iloc[0]
        self.assertEqual(heat["event_state"], "CLOSED_RESPONSE_UNRESOLVED")
        self.assertTrue(bool(heat["requires_review"]))

    def test_later_crop_instance_closure_does_not_rewrite_old_prefix(self) -> None:
        early_rows = [
            _source_row("2025-01-01", crop="Maize", season="A", risk="MED-HIGH", driver="heat"),
            _source_row("2025-01-02", crop="Maize", season="A", risk="MED-HIGH", driver="heat", echo=1),
            _source_row("2025-01-03", crop="Maize", season="A", risk="LOW", driver=None, echo=2),
        ]
        early_signals = prepare_causal_signals(pd.DataFrame(early_rows), self.policy)
        early = run_state_machine(early_signals, self.policy, as_of_date=date(2025, 1, 3))
        appended_signals = prepare_causal_signals(
            pd.DataFrame(
                early_rows
                + [_source_row("2025-01-08", crop="Beans", season="B", risk="LOW")]
            ),
            self.policy,
        )
        appended = run_state_machine(appended_signals, self.policy, as_of_date=date(2025, 1, 8))
        event_id = early.events.iloc[0]["event_id"]
        old = early.daily_records[early.daily_records["event_id"] == event_id]
        replayed_old = appended.daily_records[
            (appended.daily_records["event_id"] == event_id)
            & (pd.to_datetime(appended.daily_records["observation_date"]).dt.date <= date(2025, 1, 3))
        ]
        columns = ["observation_date", "event_state", "event_state_id", "right_censored"]
        assert_frame_equal(
            old[columns].reset_index(drop=True), replayed_old[columns].reset_index(drop=True)
        )
        closure = appended.daily_records[
            (appended.daily_records["event_id"] == event_id)
            & (appended.daily_records["event_state"] == "CLOSED_SEASON_BOUNDARY")
        ]
        self.assertEqual(str(closure.iloc[0]["observation_date"]), "2025-01-08")

    def test_decline_without_elevated_driver_is_unattributed(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "field_id": "field-A",
                    "crop_name": "Maize",
                    "crop_season": "A",
                    "crop_instance_id": "crop-instance-A",
                    "crop_instance_start_date": date(2025, 1, 1),
                    "observation_date": date(2025, 1, 1),
                    "hazard_family": "heat",
                    "risk_rank": 1,
                    "stage_family": "vegetative",
                    "response_class": "severe_decline",
                    "new_response_evidence": True,
                }
            ]
        )
        result = run_state_machine(signals, self.policy, as_of_date=date(2025, 1, 1))
        event = result.events.iloc[0]
        self.assertEqual(event["hazard_signature"], "unattributed_decline")
        self.assertTrue(bool(event["requires_review"]))


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(POLICY_PATH)

    def _write_input(self, path: Path) -> None:
        rows = []
        for offset in range(12):
            day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=offset)
            rows.append(
                _source_row(
                    day.date().isoformat(),
                    risk="MED-HIGH" if offset < 2 else "LOW",
                    driver="heat" if offset < 2 else None,
                    echo=0 if offset in {0, 8} else (offset if offset < 8 else offset - 8),
                )
            )
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_generation_is_immutable_and_historical_prefix_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "echo.parquet"
            output = root / "monitor"
            self._write_input(input_path)
            early = build_generation(
                input_parquet=input_path,
                output_dir=output,
                as_of_date=date(2025, 1, 5),
                policy=self.policy,
                max_fields=10,
            )
            late = build_generation(
                input_parquet=input_path,
                output_dir=output,
                as_of_date=date(2025, 1, 10),
                policy=self.policy,
                max_fields=10,
            )

            early_frames = pd.read_parquet(early.generation_dir / "map_frame_fields.parquet")
            late_frames = pd.read_parquet(late.generation_dir / "map_frame_fields.parquet")
            historical = late_frames[late_frames["timeline_bucket"] == "2024-12-30"]
            assert_frame_equal(early_frames.reset_index(drop=True), historical.reset_index(drop=True))
            self.assertEqual(early_frames.iloc[0]["event_id"], late_frames.iloc[0]["event_id"])

            with self.assertRaisesRegex(FileExistsError, "Immutable generation"):
                build_generation(
                    input_parquet=input_path,
                    output_dir=output,
                    as_of_date=date(2025, 1, 5),
                    policy=self.policy,
                    max_fields=10,
                )
            manifest = json.loads((early.generation_dir / "manifest.json").read_text())
            self.assertTrue(manifest["semantics"]["prefix_safe"])
            self.assertFalse(manifest["semantics"]["persistent_event_registry"])
            self.assertIn("uncalibrated", manifest["policy"]["warning"].lower())

    def test_full_unbounded_dataframe_load_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "echo.parquet"
            self._write_input(input_path)
            with self.assertRaisesRegex(ValueError, "requires --max-fields"):
                read_canonical_input(
                    input_path,
                    as_of_date=date(2025, 1, 5),
                    max_fields=None,
                )


if __name__ == "__main__":
    unittest.main()
