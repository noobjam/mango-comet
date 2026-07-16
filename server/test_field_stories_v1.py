from __future__ import annotations

import json
import unittest

import pandas as pd

from story_monitor.field_stories_v1 import (
    FieldStoryPolicy,
    build_field_stories,
    load_field_story_policy,
)


HAZARDS = ("drought", "ponding_flooding", "heat", "damaging_wind")
TEST_POLICY = FieldStoryPolicy(
    version="field-story-test-v1-uncalibrated",
    calibration_status="UNCALIBRATED_TEST_POLICY",
    confirmation_window_observed_days=3,
    confirmation_min_elevated_days=2,
    severe_pressure_rank=4,
    quiet_observed_days=2,
    candidate_expiry_observed_days=2,
    maximum_data_gap_days=2,
    maximum_recovery_observed_days=3,
)


def _evidence(
    daily: dict[str, dict[str, int | None]],
    *,
    stages: dict[str, str] | None = None,
    responses: list[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crop_rows = []
    pressure_rows = []
    for day, ranks in sorted(daily.items()):
        crop_rows.append(
            {
                "field_id": "field-1",
                "crop_instance_id": "crop-1",
                "observation_date": day,
                "knowledge_time": f"{day}T08:00:00Z",
                "crop_name": "maize",
                "crop_season": "2026-A",
                "stage_bucket": (stages or {}).get(day, "vegetative"),
            }
        )
        for hazard in HAZARDS:
            rank = ranks.get(hazard, 0)
            pressure_rows.append(
                {
                    "record_id": f"{day}-{hazard}",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "pressure_observation_date": day,
                    "knowledge_time": f"{day}T09:00:00Z",
                    "hazard_family": hazard,
                    "pressure_observed": rank is not None,
                    "pressure_rank": 0 if rank is None else rank,
                }
            )
    response_rows = []
    for index, (source_day, known_day, response_class) in enumerate(responses or []):
        response_rows.append(
            {
                "acquisition_id": f"s2-{index}",
                "field_id": "field-1",
                "crop_instance_id": "crop-1",
                "spectral_source_date": source_day,
                "knowledge_time": f"{known_day}T10:00:00Z",
                "spectral_usable": True,
                "new_response_evidence": True,
                "response_class": response_class,
            }
        )
    responses_frame = pd.DataFrame(
        response_rows,
        columns=[
            "acquisition_id",
            "field_id",
            "crop_instance_id",
            "spectral_source_date",
            "knowledge_time",
            "spectral_usable",
            "new_response_evidence",
            "response_class",
        ],
    )
    return pd.DataFrame(crop_rows), pd.DataFrame(pressure_rows), responses_frame


class FieldStoryV1Tests(unittest.TestCase):
    def test_default_policy_is_explicitly_uncalibrated(self) -> None:
        policy = load_field_story_policy()
        self.assertIn("UNCALIBRATED", policy.calibration_status)
        self.assertGreater(policy.quiet_observed_days, 0)

    def test_concurrent_and_sequential_hazards_remain_one_story(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-01-01": {"drought": 2, "heat": 2},
                "2026-01-02": {"drought": 3, "heat": 3},
                "2026-01-03": {"drought": 3, "heat": 0},
                "2026-01-04": {"drought": 0, "heat": 0},
                "2026-01-05": {"drought": 0, "heat": 0},
            }
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(result.daily_state["story_id"].nunique(), 1)
        concurrent = result.daily_state.iloc[0]
        self.assertEqual(
            json.loads(concurrent["active_hazards_json"]), ["drought", "heat"]
        )
        drought_only = result.daily_state.loc[
            result.daily_state["decision_date"].eq(pd.Timestamp("2026-01-03").date())
        ].iloc[0]
        self.assertEqual(json.loads(drought_only["active_hazards_json"]), ["drought"])
        window = result.windows.iloc[0]
        self.assertEqual(
            json.loads(window["encountered_hazards_json"]), ["drought", "heat"]
        )
        self.assertEqual(window["story_state"], "CLOSED_PRESSURE_QUIET_NO_RESPONSE")

    def test_later_hazard_joins_before_the_open_concern_closes(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-01-10": {"drought": 4},
                "2026-01-11": {"drought": 0},
                "2026-01-12": {"drought": 0, "heat": 3},
                "2026-01-13": {"drought": 0, "heat": 3},
            }
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(result.daily_state["story_id"].nunique(), 1)
        heat_entry = result.daily_state.loc[
            result.daily_state["decision_date"].eq(pd.Timestamp("2026-01-12").date())
        ].iloc[0]
        self.assertEqual(json.loads(heat_entry["active_hazards_json"]), ["heat"])
        self.assertEqual(
            json.loads(heat_entry["encountered_hazards_json"]),
            ["drought", "heat"],
        )

    def test_stage_change_creates_a_chapter_not_a_new_story(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-02-01": {"heat": 2},
                "2026-02-02": {"heat": 3},
                "2026-02-03": {"heat": 3},
            },
            stages={"2026-02-03": "flowering"},
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(result.daily_state["story_id"].nunique(), 1)
        self.assertEqual(result.windows["story_id"].nunique(), 1)
        self.assertEqual(
            result.chapters["stage_bucket"].tolist(),
            ["vegetative", "vegetative", "flowering"],
        )

    def test_unconfirmed_story_expires_as_candidate(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-02-10": {"heat": 2},
                "2026-02-11": {"heat": 0},
                "2026-02-12": {"heat": 0},
            }
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(
            result.windows["story_state"].tolist(), ["CLOSED_CANDIDATE_EXPIRED"]
        )
        self.assertTrue(pd.isna(result.windows.iloc[0]["confirmed_time"]))

    def test_fully_closed_concern_then_later_hazard_starts_new_story(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-03-01": {"drought": 4},
                "2026-03-02": {"drought": 0},
                "2026-03-03": {"drought": 0},
                "2026-03-04": {},
                "2026-03-05": {"ponding_flooding": 4},
            }
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(result.windows["story_id"].nunique(), 2)
        self.assertEqual(
            [json.loads(value) for value in result.windows["encountered_hazards_json"]],
            [["drought"], ["ponding_flooding"]],
        )
        self.assertFalse(bool(result.windows.iloc[0]["right_censored"]))
        self.assertTrue(bool(result.windows.iloc[1]["right_censored"]))

    def test_missing_active_hazard_freezes_quiet_then_censors(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-04-01": {"heat": 4},
                "2026-04-02": {"heat": None},
                "2026-04-03": {"heat": None},
            }
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(
            result.daily_state["story_state"].tolist(),
            ["SEVERE", "DATA_GAP", "CLOSED_DATA_CENSORED"],
        )
        heat = result.hazard_daily.loc[result.hazard_daily["hazard_family"].eq("heat")]
        self.assertEqual(heat["quiet_observed_day_count"].tolist(), [0, 0, 0])
        self.assertFalse(bool(result.daily_state.iloc[-1]["coverage_adequate"]))

    def test_decline_then_recovery_closes_same_story(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-05-01": {"heat": 4},
                "2026-05-02": {"heat": 0},
                "2026-05-03": {"heat": 0},
                "2026-05-04": {"heat": 0},
            },
            responses=[
                ("2026-05-01", "2026-05-01", "medium_decline"),
                ("2026-05-04", "2026-05-04", "recovery"),
            ],
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(result.windows["story_state"].tolist(), ["CLOSED_RECOVERED"])
        self.assertEqual(result.windows.iloc[0]["response_status"], "recovery_observed")
        self.assertEqual(result.daily_state.iloc[-2]["story_state"], "RECOVERING")

    def test_missing_followup_does_not_advance_unresolved_deadline(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-05-10": {"heat": 4},
                "2026-05-11": {"heat": 0},
                "2026-05-12": {"heat": 0},
                "2026-05-13": {hazard: None for hazard in HAZARDS},
                "2026-05-14": {"heat": 0},
            },
            responses=[("2026-05-10", "2026-05-10", "medium_decline")],
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        states = result.daily_state.set_index("decision_date")["story_state"]
        self.assertEqual(states[pd.Timestamp("2026-05-12").date()], "RECOVERING")
        self.assertEqual(states[pd.Timestamp("2026-05-13").date()], "DATA_GAP")
        self.assertEqual(states[pd.Timestamp("2026-05-14").date()], "RECOVERING")
        self.assertTrue(bool(result.windows.iloc[0]["right_censored"]))

    def test_late_response_is_visible_only_when_known(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-06-01": {},
                "2026-06-02": {},
                "2026-06-03": {},
            },
            responses=[("2026-06-01", "2026-06-03", "medium_decline")],
        )
        result = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)

        self.assertEqual(
            result.daily_state["decision_date"].min(), pd.Timestamp("2026-06-03").date()
        )
        self.assertEqual(
            result.daily_state.iloc[0]["first_evidence_date"],
            pd.Timestamp("2026-06-01").date(),
        )
        self.assertEqual(
            result.daily_state.iloc[0]["response_status"], "unattributed_decline"
        )
        self.assertTrue(bool(result.daily_state.iloc[0]["requires_review"]))

    def test_future_rows_do_not_change_earlier_as_of_state(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-07-01": {"drought": 2},
                "2026-07-02": {"drought": 3},
                "2026-07-03": {"drought": 0},
                "2026-07-04": {"drought": 0},
            },
            responses=[("2026-07-01", "2026-07-04", "medium_decline")],
        )
        full = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)
        crop_prefix = crop[pd.to_datetime(crop["observation_date"]) <= "2026-07-02"]
        pressure_prefix = pressure[
            pd.to_datetime(pressure["pressure_observation_date"]) <= "2026-07-02"
        ]
        response_prefix = responses[
            pd.to_datetime(responses["knowledge_time"], utc=True)
            < pd.Timestamp("2026-07-03", tz="UTC")
        ]
        prefix = build_field_stories(
            crop_prefix, pressure_prefix, response_prefix, policy=TEST_POLICY
        )

        columns = [
            "story_id",
            "decision_date",
            "story_state",
            "active_hazards_json",
            "open_hazards_json",
            "max_risk_rank",
        ]
        pd.testing.assert_frame_equal(
            full.daily_state.loc[
                full.daily_state["decision_date"] <= pd.Timestamp("2026-07-02").date(),
                columns,
            ].reset_index(drop=True),
            prefix.daily_state[columns].reset_index(drop=True),
        )

    def test_output_is_invariant_to_input_row_order(self) -> None:
        crop, pressure, responses = _evidence(
            {
                "2026-08-01": {"drought": 2, "heat": 2},
                "2026-08-02": {"drought": 3, "heat": 3},
                "2026-08-03": {"drought": 0, "heat": 3},
            }
        )
        first = build_field_stories(crop, pressure, responses, policy=TEST_POLICY)
        second = build_field_stories(
            crop.iloc[::-1].reset_index(drop=True),
            pressure.sample(frac=1, random_state=11).reset_index(drop=True),
            responses,
            policy=TEST_POLICY,
        )

        pd.testing.assert_frame_equal(first.daily_state, second.daily_state)
        pd.testing.assert_frame_equal(first.chapters, second.chapters)
        pd.testing.assert_frame_equal(first.windows, second.windows)
        pd.testing.assert_frame_equal(first.hazard_daily, second.hazard_daily)


if __name__ == "__main__":
    unittest.main()
