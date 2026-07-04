from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd

from story_monitor.incident_context_v3 import (
    _validate_outputs,
    build_incident_context_v3,
)
from story_monitor.incident_policy_v3 import (
    DEFAULT_INCIDENT_POLICY_V3_PATH,
    load_incident_policy_v3,
)


class IncidentPolicyV3Tests(unittest.TestCase):
    def test_default_policy_is_frozen_hashed_and_contract_aligned(self) -> None:
        policy = load_incident_policy_v3()
        self.assertEqual(
            policy.source_sha256,
            hashlib.sha256(DEFAULT_INCIDENT_POLICY_V3_PATH.read_bytes()).hexdigest(),
        )
        self.assertIn("UNCALIBRATED", policy.warning.upper())
        self.assertEqual(
            policy.stage_buckets,
            (
                "emergence", "vegetative", "flowering",
                "fruiting_or_grain_fill", "maturity_or_harvest", "off_season",
                "unknown",
            ),
        )
        self.assertEqual(policy.stage_bucket_for("Flowering"), "flowering")
        self.assertEqual(policy.stage_bucket_for("invented sensitive stage"), "unknown")
        self.assertTrue(policy.same_hazard_link_required)
        self.assertEqual(policy.reference_latitude_strategy, "fixed_origin")
        self.assertEqual(policy.grid_origin_lat, -2.0)
        self.assertEqual(policy.identity_namespace, "crop-impact-incident-v3")
        self.assertEqual(policy.frontier_distance_cells, 1)
        self.assertEqual(policy.maximum_data_gap_weeks, 4)
        self.assertEqual(policy.candidate_expiry_observed_weeks, 2)
        self.assertAlmostEqual(sum(dict(policy.link_weights).values()), 1.0)
        self.assertEqual(
            set(dict(policy.link_weights)),
            {
                "active_episode_overlap", "cell_or_footprint_overlap",
                "recent_member_overlap", "centroid_proximity",
                "stage_distribution_similarity",
            },
        )
        self.assertEqual(
            [item.event_state for item in policy.lane_state_priorities[:6]],
            ["SEVERE", "ACTIVE", "QUIET_PENDING", "WATCH", "RECOVERING", "DATA_GAP"],
        )
        with self.assertRaises(FrozenInstanceError):
            policy.version = "changed"  # type: ignore[misc]

    def test_policy_rejects_missing_uncalibrated_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(DEFAULT_INCIDENT_POLICY_V3_PATH.read_text())
            payload["calibration_status"] = "CALIBRATED"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uncalibrated"):
                load_incident_policy_v3(path)

    def test_policy_rejects_append_unstable_dataset_centroid_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(DEFAULT_INCIDENT_POLICY_V3_PATH.read_text())
            payload["tracker_starter_parameters"][
                "reference_latitude_strategy"
            ] = "dataset_centroid"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot rewrite grid IDs"):
                load_incident_policy_v3(path)

    def test_policy_rejects_malformed_tracker_parameters_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            payload = json.loads(DEFAULT_INCIDENT_POLICY_V3_PATH.read_text())
            payload["tracker_starter_parameters"] = ["not", "an", "object"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                load_incident_policy_v3(path)


class IncidentContextV3Tests(unittest.TestCase):
    def test_builds_causal_week_context_and_preserves_all_episode_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            output = root / "context"

            result = build_incident_context_v3(
                generation, output, policy=_test_policy(), threads=1
            )

            self.assertEqual(result["field_week_row_count"], 2)
            self.assertEqual(result["event_week_lane_count"], 3)
            context = pd.read_parquet(output / "field_week_context.parquet")
            lanes = pd.read_parquet(output / "event_week_lanes.parquet")
            self.assertFalse(context.duplicated(["timeline_bucket", "field_id", "crop_instance_id"]).any())
            self.assertFalse(lanes.duplicated(["timeline_bucket", "event_id"]).any())

            field_a = context[context["field_id"] == "A"].iloc[0]
            self.assertTrue(field_a["monitored"])
            self.assertTrue(field_a["evaluable"])
            self.assertEqual(field_a["stage_raw"], "source::Flowering")
            self.assertEqual(field_a["stage_family_raw"], "Flowering")
            self.assertEqual(field_a["stage_bucket"], "flowering")
            self.assertEqual(field_a["geometry_join_status"], "centroid_available")
            self.assertEqual(field_a["district"], "North")

            field_b = context[context["field_id"] == "B"].iloc[0]
            self.assertTrue(field_b["monitored"])
            self.assertFalse(field_b["evaluable"])
            self.assertEqual(field_b["stage_family_raw"], "mystery phase")
            self.assertEqual(field_b["stage_bucket"], "unknown")
            self.assertEqual(field_b["geometry_join_status"], "geometry_missing")

            first_snapshot = lanes[
                (lanes["event_id"] == "event-a")
                & (lanes["timeline_bucket"].astype(str).str[:10] == "2025-01-06")
            ].iloc[0]
            self.assertEqual(str(first_snapshot["stage_source_date"])[:10], "2025-01-06")
            self.assertEqual(first_snapshot["stage_raw"], "source::vegetative")
            self.assertEqual(first_snapshot["stage_family_raw"], "vegetative")
            self.assertEqual(first_snapshot["stage_bucket"], "vegetative")
            self.assertLessEqual(first_snapshot["stage_source_date"], first_snapshot["snapshot_as_of_date"])

            week_one = lanes[lanes["timeline_bucket"].astype(str).str[:10] == "2025-01-06"]
            severe = week_one[week_one["event_id"] == "event-b"].iloc[0]
            active = week_one[week_one["event_id"] == "event-a"].iloc[0]
            self.assertTrue(severe["fresh_response_evidence"])
            self.assertEqual(severe["signal_response_class"], "medium_decline")
            self.assertFalse(active["fresh_response_evidence"])
            self.assertEqual(severe["field_hazard_lane_rank"], 1)
            self.assertTrue(severe["is_canonical_field_hazard_lane"])
            self.assertTrue(severe["is_canonical_field_hazard_week"])
            self.assertEqual(active["field_hazard_lane_rank"], 2)
            self.assertFalse(active["is_canonical_field_hazard_lane"])

            gap = lanes[lanes["timeline_bucket"].astype(str).str[:10] == "2025-01-13"].iloc[0]
            self.assertFalse(gap["monitored"])
            self.assertFalse(gap["evaluable"])
            self.assertEqual(str(gap["stage_source_date"])[:10], "2025-01-08")

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["monitored_crop_instance_week_count"], 2)
            self.assertEqual(manifest["counts"]["evaluable_crop_instance_week_count"], 1)
            self.assertEqual(manifest["counts"]["geometry_present_crop_instance_week_count"], 1)
            self.assertEqual(manifest["counts"]["geometry_missing_crop_instance_week_count"], 1)
            self.assertEqual(manifest["counts"]["source_field_count"], 2)
            self.assertEqual(
                manifest["counts"]["source_field_centroid_coverage"], 0.5
            )
            self.assertEqual(
                manifest["counts"]["source_known_stage_coverage"], 0.5
            )
            self.assertEqual(
                manifest["counts"]["stage_coverage_by_crop"][0]
                ["crop_instance_week_count"],
                1,
            )
            self.assertEqual(
                manifest["counts"]["top_unmapped_stage_labels"][0]
                ["stage_family_normalized"],
                "mystery_phase",
            )
            self.assertTrue(
                manifest["policy"]["tracker_starter_parameters"]["same_hazard_link_required"]
            )
            with self.assertRaises(FileExistsError):
                build_incident_context_v3(
                    generation, output, policy=_test_policy(), threads=1
                )

    def test_duplicate_source_key_fails_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            path = generation / "daily_causal_signals.parquet"
            signals = pd.read_parquet(path)
            pd.concat([signals, signals.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
            output = root / "context"
            with self.assertRaisesRegex(ValueError, "duplicate_signals=1"):
                build_incident_context_v3(generation, output, threads=1)
            self.assertFalse(output.exists())

    def test_weekly_fresh_response_survives_later_no_new_snapshot_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            snapshots_path = generation / "event_state_snapshots.parquet"
            snapshots = pd.read_parquet(snapshots_path)
            mask = (snapshots["event_id"] == "event-a") & (
                pd.to_datetime(snapshots["timeline_bucket"]) == pd.Timestamp("2025-01-06")
            )
            snapshots.loc[mask, "snapshot_as_of_date"] = "2025-01-10"
            snapshots.to_parquet(snapshots_path, index=False)
            membership_path = generation / "story_day_membership.parquet"
            memberships = pd.read_parquet(membership_path)
            memberships = pd.concat(
                [
                    memberships,
                    pd.DataFrame(
                        [
                            {
                                "event_id": "event-a", "field_id": "A",
                                "crop_instance_id": "crop-a", "observation_date": "2025-01-07",
                                "daily_response_class": "medium_decline",
                            },
                            {
                                "event_id": "event-a", "field_id": "A",
                                "crop_instance_id": "crop-a", "observation_date": "2025-01-10",
                                "daily_response_class": "no_new_event_response",
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )
            memberships.to_parquet(membership_path, index=False)
            output = root / "context"
            build_incident_context_v3(
                generation, output, policy=_test_policy(), threads=1
            )
            lanes = pd.read_parquet(output / "event_week_lanes.parquet")
            event = lanes[lanes["event_id"] == "event-a"].iloc[0]
            self.assertTrue(event["fresh_response_evidence"])
            self.assertEqual(event["signal_response_class"], "medium_decline")
            self.assertEqual(event["fresh_decline_day_count"], 1)

    def test_source_centroid_coverage_gate_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            _write_generation(generation)
            output = root / "context"
            with self.assertRaisesRegex(
                ValueError, "centroid_fields=1/2.*minimum 95.00%"
            ):
                build_incident_context_v3(generation, output, threads=1)
            self.assertFalse(output.exists())

    def test_supported_minority_crop_stage_gap_fails_even_when_global_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            week = "2025-01-06"
            rows = []
            for index in range(100):
                beans = index >= 90
                rows.append(
                    {
                        "timeline_bucket": week,
                        "field_id": f"field-{index}",
                        "crop_instance_id": f"crop-{index}",
                        "crop_name": "beans" if beans else "maize",
                        "stage_family_raw": "unmapped beans" if beans else "vegetative",
                        "stage_family_normalized": "unmapped_beans" if beans else "vegetative",
                        "stage_bucket": "unknown" if beans else "vegetative",
                        "monitored": True,
                        "evaluable": True,
                        "geometry_present": True,
                        "centroid_available": True,
                    }
                )
            pd.DataFrame(rows).to_parquet(
                stage / "field_week_context.parquet", index=False
            )
            pd.DataFrame(
                [
                    {
                        "timeline_bucket": week,
                        "event_id": "event-1",
                        "field_id": "field-0",
                        "hazard_family": "heat",
                        "stage_source_date": week,
                        "snapshot_as_of_date": week,
                        "is_canonical_field_hazard_lane": True,
                    }
                ]
            ).to_parquet(stage / "event_week_lanes.parquet", index=False)
            policy = replace(
                load_incident_policy_v3(),
                minimum_source_field_centroid_coverage=1.0,
                minimum_source_crop_instance_week_centroid_coverage=1.0,
                minimum_known_stage_coverage=0.8,
                minimum_known_stage_coverage_per_supported_crop=0.7,
                minimum_stage_coverage_crop_instance_weeks=10,
            )
            with duckdb.connect(":memory:") as connection, self.assertRaisesRegex(
                ValueError,
                "known_stage_crop_instance_weeks=90/100.*beans",
            ):
                _validate_outputs(connection, stage, policy)


def _test_policy():
    return replace(
        load_incident_policy_v3(),
        minimum_source_field_centroid_coverage=0.5,
        minimum_source_crop_instance_week_centroid_coverage=0.5,
        minimum_known_stage_coverage=0.5,
    )


def _write_generation(root: Path) -> None:
    pd.DataFrame(
        [
            _signal("A", "crop-a", "2025-01-06", "vegetative", True, False, 3, "MED-HIGH"),
            _signal("A", "crop-a", "2025-01-08", "Flowering", False, True, 2, "LOW-MED"),
            _signal("B", "crop-b", "2025-01-07", "mystery phase", False, False, 0, "NONE"),
        ]
    ).to_parquet(root / "daily_causal_signals.parquet", index=False)
    pd.DataFrame(
        [
            _snapshot("event-a", "2025-01-06", "2025-01-06", "ACTIVE", 3),
            _snapshot("event-b", "2025-01-06", "2025-01-08", "SEVERE", 4),
            _snapshot("event-a", "2025-01-13", "2025-01-13", "DATA_GAP", None),
        ]
    ).to_parquet(root / "event_state_snapshots.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "event-a", "event_start_date": "2025-01-06",
                "event_end_date": None, "close_reason": "input_boundary_right_censored",
            },
            {
                "event_id": "event-b", "event_start_date": "2025-01-08",
                "event_end_date": None, "close_reason": "input_boundary_right_censored",
            },
        ]
    ).to_parquet(root / "event_windows.parquet", index=False)
    pd.DataFrame(
        [
            {"event_id": "event-a", "field_id": "A", "crop_instance_id": "crop-a", "observation_date": "2025-01-06", "daily_response_class": "no_new_event_response"},
            {"event_id": "event-b", "field_id": "A", "crop_instance_id": "crop-a", "observation_date": "2025-01-08", "daily_response_class": "medium_decline"},
        ]
    ).to_parquet(root / "story_day_membership.parquet", index=False)
    pd.DataFrame(
        [
            {
                "field_id": "A", "centroid_lon": 30.1, "centroid_lat": -1.9,
                "district": "North", "sector": "N1", "cell": "C1", "village": "V1",
            }
        ]
    ).to_parquet(root / "map_field_geometry.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete", "immutable": True,
                    "generation_id": "generation-test", "as_of_date": "2025-01-31",
                }
            }
        ),
        encoding="utf-8",
    )


def _signal(
    field_id: str,
    crop_instance_id: str,
    date: str,
    stage: str,
    pressure: bool,
    response: bool,
    risk: int,
    band: str,
) -> dict[str, object]:
    return {
        "field_id": field_id,
        "observation_date": date,
        "crop_name": "maize" if field_id == "A" else "beans",
        "crop_season": "2025-A",
        "crop_instance_id": crop_instance_id,
        "crop_stage": f"source::{stage}",
        "stage_family": stage,
        "pressure_observed": pressure,
        "new_response_evidence": response,
        "risk_rank": risk,
        "risk_band": band,
        "hazard_family": "heat" if field_id == "A" else "none",
        "response_class": "medium_decline" if response else "no_material_change",
    }


def _snapshot(
    event_id: str,
    bucket: str,
    as_of: str,
    state: str,
    risk: int | None,
) -> dict[str, object]:
    return {
        "timeline_bucket": bucket,
        "snapshot_as_of_date": as_of,
        "field_id": "A",
        "crop_name": "maize",
        "crop_season": "2025-A",
        "crop_instance_id": "crop-a",
        "event_id": event_id,
        "event_state": state,
        "hazard_signature": "heat",
        "max_risk_rank": 4 if event_id == "event-b" else 3,
        "max_risk_band": "HIGH" if event_id == "event-b" else "MED-HIGH",
        "current_risk_rank": risk,
        "current_risk_band": "UNKNOWN" if risk is None else ("HIGH" if risk == 4 else "MED-HIGH"),
        "reportable_day_count": 2,
        "response_day_count": 0,
        "right_censored": state in {"ACTIVE", "SEVERE", "DATA_GAP"},
        "is_data_gap_snapshot": state == "DATA_GAP",
        "requires_review": state == "SEVERE",
        "daily_response_class": "no_new_event_response",
    }


if __name__ == "__main__":
    unittest.main()
