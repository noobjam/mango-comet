from __future__ import annotations

from dataclasses import replace
import random
import unittest

import pandas as pd

from story_monitor.incident_policy_v3 import load_incident_policy_v3
from story_monitor.incident_policy_v4 import load_incident_policy_v4
from story_monitor.incident_replay_context_v4 import replay_daily_episodes_v4


class IncidentReplayContextV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_policy = load_incident_policy_v4()
        self.tracker_policy = replace(
            load_incident_policy_v3(), quiet_close_weeks=1
        )

    def test_day_major_replay_is_input_order_invariant(self) -> None:
        crops = _crops("2026-01-01", 3)
        pressure = pd.DataFrame(
            [
                _pressure("2026-01-01", "heat", 4),
                _pressure("2026-01-01", "drought", 0),
                _pressure("2026-01-02", "heat", 4),
                _pressure("2026-01-02", "drought", 0),
                _pressure("2026-01-03", "heat", 1),
                _pressure("2026-01-03", "drought", 0),
            ]
        )
        acquisitions = _empty_acquisitions()

        expected = self._run(crops, pressure, acquisitions)
        shuffled_pressure = pressure.sample(frac=1, random_state=21).reset_index(drop=True)
        shuffled_crops = crops.sample(frac=1, random_state=9).reset_index(drop=True)
        actual = self._run(shuffled_crops, shuffled_pressure, acquisitions)

        pd.testing.assert_frame_equal(
            expected.daily_episode_state, actual.daily_episode_state
        )
        pd.testing.assert_frame_equal(expected.episode_windows, actual.episode_windows)
        self.assertEqual(
            expected.daily_episode_state["decision_date"].nunique(), 3
        )
        heat = expected.daily_episode_state.query("hazard_family == 'heat'")
        self.assertEqual(len(heat), 3)
        self.assertEqual(heat.iloc[-1]["event_state"], "QUIET_PENDING")

    def test_unusable_or_nonpositive_s2_never_creates_response(self) -> None:
        crops = _crops("2026-01-01", 3)
        pressure = pd.DataFrame(
            [
                _pressure("2026-01-01", "heat", 4),
                _pressure("2026-01-02", "heat", 1),
                _pressure("2026-01-03", "heat", 1),
            ]
        )
        attempts = pd.DataFrame(
            [
                _acquisition(
                    "2026-01-02", "severe_decline", usable=False,
                    fresh=False, acquisition_id="rejected",
                ),
                _acquisition(
                    "2026-01-03", "insufficient_reference", usable=True,
                    fresh=False, acquisition_id="no-reference",
                ),
            ]
        )

        result = self._run(crops, pressure, attempts)

        self.assertFalse(result.daily_episode_state["fresh_response_evidence"].any())
        self.assertEqual(result.episode_windows.iloc[0]["response_day_count"], 0)
        self.assertEqual(
            result.diagnostics["ignored_nonpositive_or_unusable_acquisition_count"],
            2,
        )

    def test_string_false_pressure_is_missing_not_observed(self) -> None:
        pressure = _pressure("2026-01-01", "heat", 4)
        pressure["pressure_observed"] = "false"

        result = self._run(
            _crops("2026-01-01", 1),
            pd.DataFrame([pressure]),
            _empty_acquisitions(),
        )

        self.assertTrue(result.daily_episode_state.empty)
        self.assertFalse(result.daily_signals.iloc[0]["pressure_observed"])

    def test_string_false_s2_flags_never_create_fresh_decline(self) -> None:
        attempt = _acquisition("2026-01-02", "severe_decline")
        attempt["spectral_usable"] = "true"
        attempt["new_response_evidence"] = "false"
        result = self._run(
            _crops("2026-01-01", 2),
            pd.DataFrame(
                [
                    _pressure("2026-01-01", "heat", 4),
                    _pressure("2026-01-02", "heat", 0),
                ]
            ),
            pd.DataFrame([attempt]),
        )

        self.assertFalse(result.daily_episode_state["fresh_response_evidence"].any())
        self.assertEqual(result.episode_windows.iloc[0]["response_day_count"], 0)
        self.assertEqual(
            result.diagnostics["ignored_nonpositive_or_unusable_acquisition_count"],
            1,
        )

    def test_exact_v4_response_severity_controls_transition(self) -> None:
        crops = _crops("2026-01-01", 2)
        pressure = pd.DataFrame(
            [
                _pressure("2026-01-01", "heat", 4),
                _pressure("2026-01-02", "heat", 1),
            ]
        )
        severe = self._run(
            crops,
            pressure,
            pd.DataFrame([_acquisition("2026-01-02", "severe_decline")]),
        )
        medium = self._run(
            crops,
            pressure,
            pd.DataFrame([_acquisition("2026-01-02", "medium_decline")]),
        )

        self.assertEqual(severe.daily_episode_state.iloc[-1]["event_state"], "SEVERE")
        self.assertEqual(
            medium.daily_episode_state.iloc[-1]["event_state"], "QUIET_PENDING"
        )
        self.assertEqual(
            severe.daily_episode_state.iloc[-1]["response_class"],
            "severe_decline",
        )
        self.assertEqual(
            medium.daily_episode_state.iloc[-1]["response_class"],
            "medium_decline",
        )

    def test_missing_weather_does_not_advance_quiet_clock(self) -> None:
        days = pd.date_range("2026-01-01", periods=12, freq="D")
        crops = _crops("2026-01-01", len(days))
        rows = [_pressure("2026-01-01", "heat", 4)]
        rows.extend(
            _pressure(day, "heat", None, observed=False)
            for day in days[1:6]
        )
        rows.extend(_pressure(day, "heat", 0) for day in days[6:])

        result = self._run(crops, pd.DataFrame(rows), _empty_acquisitions())

        final = result.daily_episode_state.iloc[-1]
        self.assertEqual(final["event_state"], "QUIET_PENDING")
        self.assertEqual(final["quiet_observed_day_count"], 6)
        gaps = result.daily_episode_state.query("event_state == 'DATA_GAP'")
        self.assertEqual(len(gaps), 5)
        self.assertTrue(gaps["pressure_rank"].isna().all())

    def test_only_later_usable_recovery_closes_decline(self) -> None:
        crops = _crops("2026-01-01", 4)
        pressure = pd.DataFrame(
            [
                _pressure("2026-01-01", "heat", 4),
                _pressure("2026-01-02", "heat", 0),
                _pressure("2026-01-03", "heat", 0),
                _pressure("2026-01-04", "heat", 0),
            ]
        )
        attempts = pd.DataFrame(
            [
                _acquisition("2026-01-01", "severe_decline", acquisition_id="decline"),
                _acquisition(
                    "2026-01-02", "recovery", usable=False, fresh=False,
                    acquisition_id="rejected-recovery",
                ),
                _acquisition("2026-01-04", "recovery", acquisition_id="usable-recovery"),
            ]
        )

        result = self._run(crops, pressure, attempts)

        self.assertEqual(
            result.daily_episode_state.iloc[-1]["event_state"], "CLOSED_RECOVERED"
        )
        self.assertEqual(
            result.episode_windows.iloc[0]["close_reason"],
            "later_usable_recovery_acquisition",
        )
        self.assertEqual(result.episode_windows.iloc[0]["response_day_count"], 2)

    def test_late_known_older_recovery_cannot_close_newer_decline(self) -> None:
        crops = _crops("2026-01-01", 8)
        pressure = pd.DataFrame(
            [
                _pressure(
                    "2026-01-05", "heat", 4,
                    knowledge_time="2026-01-06T09:00:00Z",
                ),
                _pressure("2026-01-07", "heat", 0),
            ]
        )
        decline = _acquisition("2026-01-05", "severe_decline")
        decline["knowledge_time"] = "2026-01-06T10:00:00Z"
        recovery = _acquisition("2026-01-04", "recovery")
        recovery["knowledge_time"] = "2026-01-07T10:00:00Z"

        result = self._run(
            crops,
            pressure,
            pd.DataFrame([decline, recovery]),
        )

        self.assertEqual(
            result.daily_episode_state.iloc[-1]["event_state"], "QUIET_PENDING"
        )
        self.assertTrue(result.episode_windows.iloc[0]["right_censored"])
        self.assertEqual(
            result.episode_windows.iloc[0]["close_reason"],
            "input_boundary_right_censored",
        )

    def test_same_knowledge_day_applies_decline_then_later_recovery(self) -> None:
        crops = _crops("2026-01-01", 7)
        decline = _acquisition("2026-01-03", "severe_decline")
        decline["knowledge_time"] = "2026-01-07T10:00:00Z"
        recovery = _acquisition("2026-01-04", "recovery")
        recovery["knowledge_time"] = "2026-01-07T10:00:00Z"

        result = self._run(
            crops,
            pd.DataFrame(
                [
                    _pressure("2026-01-01", "heat", 4),
                    _pressure("2026-01-07", "heat", 0),
                ]
            ),
            pd.DataFrame([recovery, decline]),
        )

        self.assertEqual(
            result.daily_episode_state.iloc[-1]["event_state"],
            "CLOSED_RECOVERED",
        )
        self.assertEqual(
            result.daily_episode_state.iloc[-1]["response_class"], "recovery"
        )
        self.assertEqual(result.episode_windows.iloc[0]["response_day_count"], 1)
        self.assertEqual(result.diagnostics["ambiguous_recovery_attribution_count"], 0)

    def test_late_evidence_does_not_rewrite_visible_prefix(self) -> None:
        crops = _crops("2026-01-01", 5)
        pressure = pd.DataFrame(
            [
                _pressure("2026-01-01", "heat", 4),
                _pressure(
                    "2026-01-02", "heat", 0,
                    knowledge_time="2026-01-05T08:00:00Z",
                ),
            ]
        )
        full = self._run(crops, pressure, _empty_acquisitions())
        cutoff = pd.Timestamp("2026-01-03T23:59:59Z")
        prefix = self._run(
            crops[pd.to_datetime(crops["knowledge_time"], utc=True) <= cutoff],
            pressure[pd.to_datetime(pressure["knowledge_time"], utc=True) <= cutoff],
            _empty_acquisitions(),
        )
        visible = full.daily_episode_state[
            pd.to_datetime(full.daily_episode_state["knowledge_time"], utc=True)
            <= cutoff
        ].reset_index(drop=True)

        pd.testing.assert_frame_equal(visible, prefix.daily_episode_state)
        self.assertEqual(
            full.daily_episode_state.iloc[1]["decision_date"],
            pd.Timestamp("2026-01-05").date(),
        )

    def test_future_crop_assignment_is_rejected(self) -> None:
        crops = _crops("2026-01-01", 2)
        attempt = _acquisition("2026-01-01", "severe_decline")
        attempt["crop_assignment_effective_date"] = "2026-01-02"
        attempt["crop_assignment_available_at"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "future effective crop assignment"):
            self._run(
                crops,
                pd.DataFrame([_pressure("2026-01-01", "heat", 4)]),
                pd.DataFrame([attempt]),
            )

    def test_late_older_crop_row_cannot_roll_stage_backward(self) -> None:
        crops = _crops("2026-01-01", 2)
        crops.loc[0, "stage_bucket"] = "germination"
        crops.loc[1, "stage_bucket"] = "flowering"
        older = crops.iloc[0].copy()
        older["observation_date"] = pd.Timestamp("2025-12-31")
        older["knowledge_time"] = pd.Timestamp("2026-01-03", tz="UTC")
        older["stage_bucket"] = "harvest"
        crops = pd.concat([crops, older.to_frame().T], ignore_index=True)

        result = self._run(
            crops,
            pd.DataFrame([_pressure("2026-01-03", "heat", 4)]),
            _empty_acquisitions(),
        )

        self.assertEqual(result.daily_episode_state.iloc[0]["stage_bucket"], "flowering")
        self.assertEqual(
            result.daily_episode_state.iloc[0]["stage_source_date"],
            pd.Timestamp("2026-01-02").date(),
        )

    def _run(
        self,
        crops: pd.DataFrame,
        pressure: pd.DataFrame,
        acquisitions: pd.DataFrame,
    ):
        return replay_daily_episodes_v4(
            crops,
            pressure,
            acquisitions,
            source_policy=self.source_policy,
            tracker_policy=self.tracker_policy,
        )


def _crops(start: str, periods: int) -> pd.DataFrame:
    rows = []
    for day in pd.date_range(start, periods=periods, freq="D"):
        rows.append(
            {
                "field_id": "field-1",
                "crop_instance_id": "crop-1",
                "observation_date": day,
                "knowledge_time": day.tz_localize("UTC"),
                "crop_name": "maize",
                "crop_season": "2026-a",
                "stage_bucket": "flowering",
                "crop_stage_raw": "silking",
                "stage_family_raw": "flowering",
                "crop_instance_start_date": pd.Timestamp(start),
            }
        )
    return pd.DataFrame(rows)


def _pressure(
    day: object,
    hazard: str,
    rank: int | None,
    *,
    observed: bool = True,
    knowledge_time: str | None = None,
) -> dict[str, object]:
    timestamp = pd.Timestamp(day)
    return {
        "record_id": f"pressure-{timestamp.date()}-{hazard}",
        "field_id": "field-1",
        "crop_instance_id": "crop-1",
        "pressure_observation_date": timestamp,
        "knowledge_time": knowledge_time or timestamp.tz_localize("UTC"),
        "hazard_family": hazard,
        "pressure_observed": observed,
        "pressure_rank": rank,
        "pressure_band": {
            None: "UNKNOWN", 0: "NONE", 1: "LOW", 2: "LOW-MED",
            3: "MED-HIGH", 4: "HIGH",
        }[rank],
    }


def _acquisition(
    day: str,
    response: str,
    *,
    usable: bool = True,
    fresh: bool = True,
    acquisition_id: str | None = None,
) -> dict[str, object]:
    return {
        "acquisition_id": acquisition_id or f"s2-{day}-{response}",
        "field_id": "field-1",
        "crop_instance_id": "crop-1",
        "spectral_source_date": day,
        "knowledge_time": f"{day}T10:00:00Z",
        "crop_assignment_effective_date": day,
        "crop_assignment_available_at": f"{day}T08:00:00Z",
        "spectral_usable": usable,
        "new_response_evidence": fresh,
        "response_class": response,
    }


def _empty_acquisitions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "acquisition_id", "field_id", "crop_instance_id",
            "spectral_source_date", "knowledge_time", "spectral_usable",
            "crop_assignment_effective_date", "crop_assignment_available_at",
            "new_response_evidence", "response_class",
        ]
    )


if __name__ == "__main__":
    unittest.main()
