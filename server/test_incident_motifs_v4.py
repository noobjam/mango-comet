from __future__ import annotations

from datetime import date, datetime, timezone
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.incident_motifs_v4 import (
    MODEL_FEATURE_COLUMNS,
    MotifDiscoveryConfig,
    PrefixCalibrationConfig,
    assign_open_set_prefixes,
    build_causal_incident_evidence,
    build_causal_prefix_features,
    build_completed_story_features,
    build_eligibility_ledger,
    build_live_scoring_ledger,
    build_review_overlay_template,
    discover_completed_motifs,
    evaluate_prefix_replay,
    fit_calibrated_prefix_model,
    temporal_split_ledger,
)


UTC = timezone.utc


def _weekly(incident_id: str = "i-1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": incident_id,
                "knowledge_time": datetime(2025, 1, 7, tzinfo=UTC),
                "current_state": "ACTIVE",
            },
            {
                "incident_id": incident_id,
                "knowledge_time": datetime(2025, 1, 14, tzinfo=UTC),
                "current_state": "CLOSED_RECOVERED",
            },
        ]
    )


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "i-1", "exposure_id": "e-1", "crop_name": "Maize",
                "hazard_family": "Heat", "terminal_state": "CLOSED_RECOVERED",
                "right_censored": False, "confirmed_week": "2025-01-06",
                "first_evidence_week": "2025-01-01", "closed_week": "2025-01-14",
            },
            {
                "incident_id": "i-expired", "exposure_id": "e-2", "crop_name": "Maize",
                "hazard_family": "Heat", "terminal_state": "CLOSED_CANDIDATE_EXPIRED",
                "right_censored": False, "confirmed_week": None,
                "first_evidence_week": "2025-01-01", "closed_week": "2025-01-08",
            },
            {
                "incident_id": "i-data", "exposure_id": "e-3", "crop_name": "Beans",
                "hazard_family": "Drought", "terminal_state": "CLOSED_DATA_CENSORED",
                "right_censored": False, "confirmed_week": "2025-01-01",
                "first_evidence_week": "2025-01-01", "closed_week": "2025-01-08",
            },
            {
                "incident_id": "i-later", "exposure_id": "e-4", "crop_name": "Maize",
                "hazard_family": "Heat", "terminal_state": "CLOSED_RESPONSE_UNRESOLVED",
                "right_censored": False, "confirmed_week": "2025-02-03",
                "first_evidence_week": "2025-02-01", "closed_week": "2025-02-14",
            },
        ]
    )


def _feature_row(incident_id: str, value: float, available: str = "2025-01-10") -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": "incident-motif-features-v4/1",
        "incident_id": incident_id,
        "exposure_id": "e-" + incident_id,
        "crop_name": "maize",
        "hazard_family": "heat",
        "lineage_family_id": "f-" + incident_id,
        "feature_available_at": available,
        "dominant_stage": "vegetative",
        "stage_entropy": 0.0,
        "stage_distribution_json": '{"vegetative":1.0}',
    }
    row.update({name: value for name in MODEL_FEATURE_COLUMNS})
    return row


def _prefix_row(
    incident_id: str,
    value: float,
    as_of: str,
    *,
    weather: int = 7,
    s2: int = 1,
) -> dict[str, object]:
    row = _feature_row(incident_id, value, available=as_of)
    row.update(
        {
            "feature_schema_version": "incident-motif-prefix-v4/1",
            "prefix_as_of_time": as_of,
            "weather_day_horizon": weather,
            "s2_acquisition_horizon": s2,
        }
    )
    row["weather_observed_day_count"] = float(weather)
    row["s2_usable_acquisition_count"] = float(s2)
    return row


class IncidentMotifsV4Tests(unittest.TestCase):
    def test_live_ledger_is_historical_as_of_not_eventual_outcome(self) -> None:
        windows = _windows().iloc[[0]].copy()
        checkpoints = pd.DataFrame(
            [
                {
                    "incident_id": "i-1",
                    "knowledge_time": "2025-01-07T12:00:00Z",
                    "current_state": "ACTIVE",
                },
                {
                    "incident_id": "i-1",
                    "knowledge_time": "2025-01-14T12:00:00Z",
                    "current_state": "CLOSED_RECOVERED",
                },
            ]
        )
        before_close = build_live_scoring_ledger(
            windows, pd.DataFrame(), checkpoints, as_of="2025-01-10T00:00:00Z"
        )
        self.assertEqual(list(before_close["incident_id"]), ["i-1"])
        self.assertEqual(before_close.iloc[0]["terminal_state"], "ACTIVE")
        self.assertEqual(
            before_close.iloc[0]["feature_available_at"],
            pd.Timestamp("2025-01-10T00:00:00Z"),
        )
        after_close = build_live_scoring_ledger(
            windows, pd.DataFrame(), checkpoints, as_of="2025-01-15T00:00:00Z"
        )
        self.assertTrue(after_close.empty)

    def test_live_ledger_excludes_candidate_until_confirmation_is_known(self) -> None:
        windows = _windows().iloc[[0]].copy()
        checkpoints = pd.DataFrame(
            [
                {
                    "incident_id": "i-1",
                    "knowledge_time": "2025-01-07T12:00:00Z",
                    "current_state": "CANDIDATE",
                },
                {
                    "incident_id": "i-1",
                    "knowledge_time": "2025-01-14T12:00:00Z",
                    "current_state": "ACTIVE",
                },
            ]
        )

        candidate = build_live_scoring_ledger(
            windows, pd.DataFrame(), checkpoints, as_of="2025-01-10T00:00:00Z"
        )
        self.assertTrue(candidate.empty)

        confirmed = build_live_scoring_ledger(
            windows, pd.DataFrame(), checkpoints, as_of="2025-01-15T00:00:00Z"
        )
        self.assertEqual(list(confirmed["incident_id"]), ["i-1"])
        self.assertTrue(bool(confirmed.iloc[0]["confirmed"]))

    def test_daily_weather_carries_known_ownership_but_stops_at_lineage_boundary(self) -> None:
        membership = pd.DataFrame(
            [
                {
                    "incident_id": "parent",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "hazard_family": "heat",
                    "timeline_bucket": "2025-01-06",
                    "knowledge_time": "2025-01-10",
                    "membership_role": "impact_lag",
                },
                {
                    "incident_id": "child",
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "hazard_family": "heat",
                    "timeline_bucket": "2025-01-20",
                    "knowledge_time": "2025-01-20",
                    "membership_role": "impact_lag",
                },
            ]
        )
        pressure = pd.DataFrame(
            [
                {
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "hazard_family": "heat",
                    "observation_date": day,
                    "knowledge_time": day,
                    "pressure_observed": True,
                    "pressure_score": 1.0,
                    "pressure_rank": 3,
                }
                for day in ("2025-01-13", "2025-01-20")
            ]
        )
        windows = pd.DataFrame(
            [
                {
                    "incident_id": "parent",
                    "terminal_state": "ACTIVE",
                    "right_censored": True,
                    "feature_available_at": "2025-01-20",
                },
                {
                    "incident_id": "child",
                    "terminal_state": "ACTIVE",
                    "right_censored": True,
                    "feature_available_at": "2025-01-20",
                },
            ]
        )
        lineage = pd.DataFrame(
            [
                {
                    "parent_incident_id": "parent",
                    "child_incident_id": "child",
                    "lineage_type": "split",
                    "timeline_bucket": "2025-01-20",
                }
            ]
        )
        evidence = build_causal_incident_evidence(
            membership,
            pressure,
            pd.DataFrame(),
            windows,
            lineage,
        ).daily_pressure
        self.assertEqual(
            evidence.set_index("timeline_date")["incident_id"].to_dict(),
            {
                pd.Timestamp("2025-01-13T00:00:00Z"): "parent",
                pd.Timestamp("2025-01-20T00:00:00Z"): "child",
            },
        )

    def test_raw_field_evidence_joins_through_v3_membership_and_dual_clocks(self) -> None:
        membership = pd.DataFrame(
            [
                {
                    "incident_id": "i-1", "field_id": "f-1",
                    "crop_instance_id": "c-1", "hazard_family": "Heat",
                    "timeline_bucket": "2025-01-06", "knowledge_time": "2025-01-07",
                    "stage_bucket": "vegetative", "membership_role": "impact_lag",
                    "fresh_response_evidence": True,
                    "response_class": "severe_decline",
                },
                {
                    "incident_id": "i-1", "field_id": "f-2",
                    "crop_instance_id": "c-2", "hazard_family": "Heat",
                    "timeline_bucket": "2025-01-06", "knowledge_time": "2025-01-07",
                    "stage_bucket": "flowering", "membership_role": "pressure_core",
                    "fresh_response_evidence": False, "response_class": "stable",
                },
            ]
        )
        pressure = pd.DataFrame(
            [
                {
                    "field_id": "f-1", "crop_instance_id": "c-1",
                    "hazard_family": "Heat", "observation_date": "2025-01-06",
                    "knowledge_time": "2025-01-06", "pressure_observed": True,
                    "pressure_score": 0.8, "pressure_rank": 4,
                    "pressure_active": True,
                },
                {
                    "field_id": "f-2", "crop_instance_id": "c-2",
                    "hazard_family": "Heat", "observation_date": "2025-01-06",
                    "knowledge_time": "2025-01-06", "pressure_observed": True,
                    "pressure_score": 0.2, "pressure_rank": 4,
                    "pressure_active": True,
                },
            ]
        )
        s2 = pd.DataFrame(
            [
                {
                    "acquisition_id": "a-1", "field_id": "f-1",
                    "crop_instance_id": "c-1", "spectral_source_date": "2025-01-06",
                    "spectral_available_at": "2025-01-08",
                    "acquisition_attempted": True, "spectral_usable": True,
                },
                {
                    "acquisition_id": "a-2", "field_id": "f-2",
                    "crop_instance_id": "c-2", "spectral_source_date": "2025-01-06",
                    "spectral_available_at": "2025-01-08",
                    "acquisition_attempted": True, "spectral_usable": False,
                },
            ]
        )
        evidence = build_causal_incident_evidence(membership, pressure, s2)
        day = evidence.daily_pressure.iloc[0]
        self.assertEqual(day["incident_id"], "i-1")
        self.assertEqual(day["feature_available_at"], pd.Timestamp("2025-01-07", tz="UTC"))
        self.assertEqual(day["monitored_count"], 2)
        self.assertEqual(day["affected_count"], 1)
        self.assertEqual(day["severe_count"], 1)
        self.assertAlmostEqual(day["weather_intensity"], 0.5)
        self.assertEqual(set(evidence.s2_acquisitions["incident_id"]), {"i-1"})
        self.assertTrue(
            evidence.s2_acquisitions["feature_available_at"].eq(
                pd.Timestamp("2025-01-08", tz="UTC")
            ).all()
        )

    def test_eligibility_is_exhaustive_and_temporal_split_purges_lineage(self) -> None:
        weekly = pd.concat(
            [
                _weekly("i-1"),
                _weekly("i-expired").assign(
                    knowledge_time=pd.to_datetime(["2025-01-01", "2025-01-08"], utc=True)
                ),
                _weekly("i-data").assign(
                    knowledge_time=pd.to_datetime(["2025-01-01", "2025-01-08"], utc=True)
                ),
                _weekly("i-later").assign(
                    knowledge_time=pd.to_datetime(["2025-02-01", "2025-02-14"], utc=True)
                ),
            ],
            ignore_index=True,
        )
        lineage = pd.DataFrame(
            [{"parent_incident_id": "i-1", "child_incident_id": "i-later"}]
        )
        ledger = build_eligibility_ledger(_windows(), lineage, weekly).set_index("incident_id")

        self.assertEqual(len(ledger), 4)
        self.assertTrue(ledger.loc["i-1", "eligible"])
        self.assertEqual(ledger.loc["i-expired", "eligibility_reason"], "candidate_expired")
        self.assertEqual(ledger.loc["i-data", "eligibility_reason"], "data_censored")
        self.assertEqual(
            ledger.loc["i-1", "lineage_family_id"],
            ledger.loc["i-later", "lineage_family_id"],
        )

        split = temporal_split_ledger(
            ledger.reset_index(),
            train_through="2025-01-31",
            calibration_through="2025-02-28",
            evaluation_through="2025-03-31",
        ).set_index("incident_id")
        self.assertEqual(split.loc["i-1", "temporal_split"], "train")
        self.assertEqual(
            split.loc["i-later", "temporal_split"],
            "embargo_lineage_or_exposure_purge",
        )
        boundary = ledger.reset_index().loc[lambda frame: frame["incident_id"].eq("i-data")].copy()
        boundary["first_available_at"] = pd.Timestamp("2025-01-31T10:00:00Z")
        boundary["feature_available_at"] = pd.Timestamp("2025-01-31T12:00:00Z")
        boundary["purge_group_id"] = "boundary-only"
        boundary_split = temporal_split_ledger(
            boundary,
            train_through=date(2025, 1, 31),
            calibration_through=date(2025, 2, 28),
            evaluation_through=date(2025, 3, 31),
        )
        self.assertEqual(boundary_split.iloc[0]["temporal_split"], "train")

    def test_cadence_features_count_explicit_weather_and_only_usable_s2(self) -> None:
        windows = _windows().iloc[[0]].copy()
        ledger = build_eligibility_ledger(windows, pd.DataFrame(), _weekly())
        daily = []
        for index, day in enumerate(pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")):
            daily.append(
                {
                    "incident_id": "i-1",
                    "timeline_date": day,
                    "feature_available_at": day,
                    "pressure_observed": True,
                    "weather_intensity": np.nan if index == 4 else float(index),
                    "active_count": 1,
                    "severe_count": int(index == 8),
                    "affected_count": 2,
                    "monitored_count": 10,
                    "evaluable_count": 10,
                    "footprint_area_km2": 4 + index,
                    "footprint_carried_forward": False,
                    "stage_bucket": "flowering" if index >= 3 else "vegetative",
                }
            )
        pressure = pd.DataFrame(daily)
        s2 = pd.DataFrame(
            [
                {
                    "incident_id": "i-1", "crop_instance_id": "crop-1",
                    "acquisition_date": "2025-01-03", "known_date": "2025-01-04",
                    "acquisition_attempted": True, "spectral_usable": True,
                    "evidence_age_days": 1, "response_class": "medium_decline",
                    "ndvi_delta": -0.2, "ndmi_delta": -0.1, "psri_delta": 0.1,
                },
                {
                    "incident_id": "i-1", "crop_instance_id": "crop-1",
                    "acquisition_date": "2025-01-06", "known_date": "2025-01-07",
                    "acquisition_attempted": True, "spectral_usable": False,
                    "evidence_age_days": 1, "response_class": "spectral_missing",
                },
                {
                    "incident_id": "i-1", "crop_instance_id": "crop-1",
                    "acquisition_date": "2025-01-03", "known_date": "2025-01-08",
                    "acquisition_attempted": False, "spectral_usable": True,
                    "evidence_age_days": 5, "response_class": "medium_decline",
                },
                {
                    "incident_id": "i-1", "crop_instance_id": "crop-2",
                    "acquisition_date": "2025-01-03", "known_date": "2025-01-04",
                    "acquisition_attempted": True, "spectral_usable": True,
                    "evidence_age_days": 1, "response_class": "no_material_change",
                    "ndvi_delta": 0.0, "ndmi_delta": 0.0, "psri_delta": 0.0,
                },
            ]
        )
        completed = build_completed_story_features(_weekly(), pressure, s2, ledger).iloc[0]
        self.assertEqual(completed["weather_observed_day_count"], 10)
        self.assertAlmostEqual(completed["weather_intensity_missing_fraction"], 0.1)
        self.assertEqual(completed["s2_usable_acquisition_count"], 1)
        self.assertEqual(completed["dominant_stage"], "flowering")
        self.assertNotIn("dominant_stage", MODEL_FEATURE_COLUMNS)

        config = PrefixCalibrationConfig(
            weather_day_horizons=(7, 14),
            s2_acquisition_horizons=(0, 1),
            minimum_training_support=2,
            minimum_calibration_support=1,
        )
        before = build_causal_prefix_features(_weekly(), pressure, s2, ledger, config=config)
        extended = pd.concat(
            [
                pressure,
                pressure.iloc[[0]].assign(
                    timeline_date=pd.Timestamp("2025-01-12", tz="UTC"),
                    feature_available_at=pd.Timestamp("2025-01-12", tz="UTC"),
                    weather_intensity=999.0,
                ),
            ],
            ignore_index=True,
        )
        # Extend the terminal knowledge bound without changing the historical prefix.
        later_weekly = pd.concat(
            [
                _weekly(),
                pd.DataFrame(
                    [{"incident_id": "i-1", "knowledge_time": pd.Timestamp("2025-01-12", tz="UTC"), "current_state": "ACTIVE"}]
                ),
            ],
            ignore_index=True,
        ).sort_values("knowledge_time")
        later_ledger = ledger.copy()
        later_ledger["feature_available_at"] = pd.Timestamp("2025-01-14", tz="UTC")
        after = build_causal_prefix_features(later_weekly, extended, s2, later_ledger, config=config)
        columns = [*MODEL_FEATURE_COLUMNS, "weather_day_horizon", "s2_acquisition_horizon"]
        assert_frame_equal(
            before.loc[before["prefix_as_of_time"] == pd.Timestamp("2025-01-07", tz="UTC"), columns].reset_index(drop=True),
            after.loc[after["prefix_as_of_time"] == pd.Timestamp("2025-01-07", tz="UTC"), columns].reset_index(drop=True),
        )

    def test_discovery_is_deterministic_and_review_starts_pending(self) -> None:
        features = pd.DataFrame(
            [
                _feature_row("a-1", 0.0), _feature_row("a-2", 0.1),
                _feature_row("b-1", 3.0), _feature_row("b-2", 3.1),
            ]
        )
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.ones(4)
        config = MotifDiscoveryConfig(min_cluster_size=2, min_samples=1)
        with patch(
            "story_monitor.incident_motifs_v4._fit_hdbscan",
            return_value=(labels, probabilities),
        ):
            first = discover_completed_motifs(
                features, training_through="2025-01-31", config=config
            )
            changed_stage = features.copy()
            changed_stage["dominant_stage"] = "maturity_or_harvest"
            second = discover_completed_motifs(
                changed_stage, training_through="2025-01-31", config=config
            )
        self.assertEqual(first.manifest["model_version"], second.manifest["model_version"])
        self.assertEqual(len(first.prototypes), 2)
        overlay = build_review_overlay_template(first)
        self.assertEqual(set(overlay["review_status"]), {"pending"})
        self.assertTrue(overlay["reviewed_motif_id"].isna().all())
        self.assertEqual(first.manifest["stage_distance_weight"], 0.0)

    def test_prefix_calibration_excludes_unseen_maturity_and_replay_is_open_set(self) -> None:
        train = [
            _prefix_row("a-1", 0.0, "2025-01-07"),
            _prefix_row("a-2", 0.1, "2025-01-08"),
            _prefix_row("b-1", 3.0, "2025-01-07"),
            _prefix_row("b-2", 3.1, "2025-01-08"),
        ]
        calibration = [
            _prefix_row("a-c", 0.05, "2025-02-07"),
            _prefix_row("b-c", 3.05, "2025-02-07"),
            # Calibration-only maturity must be ignored, not fitted or crashed.
            _prefix_row("a-unseen", 0.05, "2025-02-14", weather=14),
        ]
        prefixes = pd.DataFrame([*train, *calibration])
        labels = pd.DataFrame(
            {
                "incident_id": ["a-1", "a-2", "b-1", "b-2", "a-c", "b-c", "a-unseen"],
                "reviewed_motif_id": ["A", "A", "B", "B", "A", "B", "A"],
            }
        )
        split = pd.DataFrame(
            {
                "incident_id": labels["incident_id"],
                "temporal_split": ["train"] * 4 + ["calibration"] * 3,
            }
        )
        config = PrefixCalibrationConfig(
            weather_day_horizons=(7, 14),
            s2_acquisition_horizons=(0, 1),
            minimum_training_support=2,
            minimum_calibration_support=1,
        )
        model = fit_calibrated_prefix_model(
            prefixes, labels, split, model_version="reviewed-v4", config=config
        )
        self.assertEqual(set(model.prototypes["weather_day_horizon"]), {7})
        queries = pd.DataFrame(
            [
                _prefix_row("hold-known", 0.05, "2025-03-07"),
                _prefix_row("hold-novel", 1.5, "2025-03-07"),
                _prefix_row("hold-pending", 0.05, "2025-03-14", weather=14),
            ]
        )
        assigned = assign_open_set_prefixes(queries, model).set_index("incident_id")
        self.assertEqual(
            assigned.loc["hold-known", "assignment_status"],
            "tentative_crop_evidence_supported",
        )
        self.assertEqual(assigned.loc["hold-novel", "assignment_status"], "novel_unassigned")
        self.assertEqual(assigned.loc["hold-pending", "assignment_status"], "pending")

        replay_assignments = assigned.reset_index().iloc[[0, 1]].copy()
        final = pd.DataFrame(
            {
                "incident_id": ["hold-known", "hold-novel"],
                "final_assignment_status": ["accepted", "novel_unassigned"],
                "reviewed_motif_id": ["A", None],
            }
        )
        replay_split = pd.DataFrame(
            {
                "incident_id": ["train", "cal", "hold-known", "hold-novel"],
                "temporal_split": ["train", "calibration", "holdout", "holdout"],
                "purge_group_id": ["g1", "g2", "g3", "g4"],
                "feature_available_at": pd.to_datetime(
                    ["2025-01-01", "2025-02-01", "2025-03-31", "2025-03-31"], utc=True
                ),
            }
        )
        report = evaluate_prefix_replay(replay_assignments, final, replay_split)
        self.assertTrue(report["hard_gates"]["passed"])
        self.assertEqual(report["metrics"]["accepted_known_precision"], 1.0)
        self.assertEqual(report["metrics"]["final_novel_false_accept_rate"], 0.0)

    def test_novel_calibration_tightens_radius_and_readiness_is_explicit(self) -> None:
        prefixes = pd.DataFrame(
            [
                _prefix_row("train-1", 0.0, "2025-01-07"),
                _prefix_row("train-2", 1.0, "2025-01-08"),
                _prefix_row("cal-known", 0.8, "2025-02-07"),
                _prefix_row("cal-novel", 0.55, "2025-02-08"),
            ]
        )
        base_labels = pd.DataFrame(
            {
                "incident_id": ["train-1", "train-2", "cal-known"],
                "reviewed_motif_id": ["A", "A", "A"],
            }
        )
        base_split = pd.DataFrame(
            {
                "incident_id": ["train-1", "train-2", "cal-known"],
                "temporal_split": ["train", "train", "calibration"],
            }
        )
        config = PrefixCalibrationConfig(
            minimum_training_support=2,
            minimum_calibration_support=1,
            maximum_novel_false_accept_rate=0.0,
        )
        unconstrained = fit_calibrated_prefix_model(
            prefixes[prefixes["incident_id"].ne("cal-novel")],
            base_labels,
            base_split,
            model_version="reviewed-v4",
            config=config,
        )
        constrained = fit_calibrated_prefix_model(
            prefixes,
            pd.concat(
                [
                    base_labels,
                    pd.DataFrame(
                        [{"incident_id": "cal-novel", "reviewed_motif_id": None}]
                    ),
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    base_split,
                    pd.DataFrame(
                        [{"incident_id": "cal-novel", "temporal_split": "calibration"}]
                    ),
                ],
                ignore_index=True,
            ),
            model_version="reviewed-v4",
            config=config,
        )
        self.assertLess(
            constrained.prototypes.iloc[0]["radius"],
            unconstrained.prototypes.iloc[0]["radius"],
        )
        self.assertTrue(constrained.prototypes.iloc[0]["radius_constrained_by_novel"])

        query = pd.DataFrame([_prefix_row("query", 0.7, "2025-03-07")])
        self.assertIn(
            assign_open_set_prefixes(query, unconstrained).iloc[0]["assignment_status"],
            {"tentative_crop_evidence_supported", "tentative_weather_only"},
        )
        self.assertEqual(
            assign_open_set_prefixes(query, constrained).iloc[0]["assignment_status"],
            "novel_unassigned",
        )

        weather_prefixes = prefixes[prefixes["incident_id"].ne("cal-novel")].copy()
        weather_prefixes["s2_acquisition_horizon"] = 0
        weather_prefixes["s2_usable_acquisition_count"] = 0.0
        weather_model = fit_calibrated_prefix_model(
            weather_prefixes,
            base_labels,
            base_split,
            model_version="reviewed-v4",
            config=config,
        )
        early = pd.DataFrame(
            [
                _prefix_row("too-early", 0.5, "2025-03-01", weather=6, s2=0),
                _prefix_row("weather-only", 0.5, "2025-03-02", weather=7, s2=0),
            ]
        )
        states = assign_open_set_prefixes(early, weather_model).set_index("incident_id")
        self.assertEqual(states.loc["too-early", "assignment_status"], "pending")
        self.assertEqual(
            states.loc["weather-only", "assignment_status"],
            "tentative_weather_only",
        )
        crop_supported = pd.DataFrame(
            [_prefix_row("crop-supported", 0.5, "2025-03-03", weather=7, s2=1)]
        )
        self.assertEqual(
            assign_open_set_prefixes(crop_supported, unconstrained).iloc[0][
                "assignment_status"
            ],
            "tentative_crop_evidence_supported",
        )

    def test_novel_false_accept_budget_is_joint_across_stratum_prototypes(self) -> None:
        train = [
            _prefix_row("a-train-1", 0.0, "2025-01-07"),
            _prefix_row("a-train-2", 0.2, "2025-01-08"),
            _prefix_row("b-train-1", 10.0, "2025-01-07"),
            _prefix_row("b-train-2", 10.2, "2025-01-08"),
        ]
        known = [
            _prefix_row("a-cal-known", 0.6, "2025-02-07"),
            _prefix_row("b-cal-known", 9.6, "2025-02-07"),
        ]
        near_novel = [
            _prefix_row("novel-a-1", 0.11, "2025-02-08"),
            _prefix_row("novel-a-2", 0.12, "2025-02-08"),
            _prefix_row("novel-b-1", 10.11, "2025-02-08"),
            _prefix_row("novel-b-2", 10.12, "2025-02-08"),
        ]
        far_novel = [
            _prefix_row(f"novel-far-{index}", 50.0 + index, "2025-02-08")
            for index in range(16)
        ]
        prefixes = pd.DataFrame([*train, *known, *near_novel, *far_novel])
        labels = pd.DataFrame(
            {
                "incident_id": prefixes["incident_id"],
                "reviewed_motif_id": [
                    "A", "A", "B", "B", "A", "B", *([None] * 20)
                ],
            }
        )
        split = pd.DataFrame(
            {
                "incident_id": prefixes["incident_id"],
                "temporal_split": ["train"] * 4 + ["calibration"] * 22,
            }
        )
        config = PrefixCalibrationConfig(
            minimum_training_support=2,
            minimum_calibration_support=1,
            maximum_novel_false_accept_rate=0.1,
        )
        model = fit_calibrated_prefix_model(
            prefixes, labels, split, model_version="reviewed-v4", config=config
        )

        novel = prefixes[prefixes["incident_id"].str.startswith("novel-")]
        assigned = assign_open_set_prefixes(novel, model)
        accepted = assigned["assignment_status"].isin(
            {"tentative_weather_only", "tentative_crop_evidence_supported"}
        )
        allowed = int(np.floor(config.maximum_novel_false_accept_rate * len(novel)))
        self.assertLessEqual(int(accepted.sum()), allowed)
        self.assertEqual(
            model.manifest["novel_negative_policy"],
            "joint_stratum_final_assignment_false_accept_cap",
        )


if __name__ == "__main__":
    unittest.main()
