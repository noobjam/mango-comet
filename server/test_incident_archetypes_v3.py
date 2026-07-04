from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from story_monitor.incident_archetypes_v3 import (
    FEATURE_GROUPS,
    MODEL_FEATURE_COLUMNS,
    PrefixPrototypeArtifacts,
    assign_open_set_prefixes,
    build_causal_prefix_features,
    extract_completed_incident_features,
    fit_prefix_prototypes,
    fit_robust_feature_schema,
    supported_prefix_horizon,
    temporal_split_completed_stories,
    transform_finite_feature_matrix,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = {
        "incident-train": {
            "crop": "Maize", "hazard": "heat", "start": "2025-01-06", "weeks": 4,
            "states": ["CANDIDATE", "ACTIVE", "RECOVERING", "CLOSED_RECOVERED"],
        },
        "incident-open": {
            "crop": "Maize", "hazard": "heat", "start": "2025-01-20", "weeks": 2,
            "states": ["CANDIDATE", "ACTIVE"],
        },
        "incident-cross": {
            "crop": "Beans", "hazard": "drought", "start": "2025-01-27", "weeks": 3,
            "states": ["ACTIVE", "RECOVERING", "CLOSED_RESPONSE_UNRESOLVED"],
        },
        "incident-holdout": {
            "crop": "Beans", "hazard": "drought", "start": "2025-02-10", "weeks": 2,
            "states": ["ACTIVE", "CLOSED_PRESSURE_QUIET_UNCONFIRMED"],
        },
    }
    weekly: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    for incident_id, spec in definitions.items():
        for index, bucket in enumerate(pd.date_range(spec["start"], periods=spec["weeks"], freq="7D")):
            affected = 2 + index
            weekly.append(
                {
                    "incident_id": incident_id,
                    "exposure_id": "exposure-" + incident_id,
                    "timeline_bucket": bucket,
                    "crop_name": spec["crop"],
                    "hazard_family": spec["hazard"],
                    "story_state": spec["states"][index],
                    "active_count": affected if index < 2 else 0,
                    "severe_count": 1 if index == 1 else 0,
                    "affected_count": affected,
                    "monitored_count": 10,
                    "evaluable_count": 9 if index == 2 else 10,
                    "footprint_area_km2": 1.0 + index,
                    "hazard_intensity_mean": 30.0 + index,
                    "relapse_count": 0,
                    "split_count": 0,
                    "merge_count": 0,
                }
            )
            for member_index in range(affected):
                members.append(
                    {
                        "incident_id": incident_id,
                        "timeline_bucket": bucket,
                        "crop_instance_id": f"crop-{incident_id}-{member_index}",
                        "field_id": f"field-{incident_id}-{member_index}",
                        "episode_id": f"episode-{incident_id}-{member_index}",
                        "membership_role": "recovering" if index >= 2 else "pressure_core",
                        "stage_bucket": "vegetative" if index < 2 else "flowering",
                    }
                )
    return pd.DataFrame(weekly), pd.DataFrame(members)


def _model_row(
    incident_id: str,
    value: float,
    *,
    crop: str = "Maize",
    hazard: str = "heat",
    horizon: int = 1,
) -> dict[str, object]:
    row: dict[str, object] = {
        "incident_id": incident_id,
        "crop_name": crop,
        "hazard_family": hazard,
        "horizon_weeks": horizon,
    }
    row.update({name: 0.0 for name in MODEL_FEATURE_COLUMNS})
    row["peak_affected_rate"] = value
    return row


class IncidentArchetypesV3Tests(unittest.TestCase):
    def test_completed_features_are_one_per_story_and_leakage_safe(self) -> None:
        weekly, membership = _fixture()
        features = extract_completed_incident_features(weekly, membership)

        self.assertEqual(
            set(features["incident_id"]),
            {"incident-train", "incident-cross", "incident-holdout"},
        )
        self.assertTrue(features["incident_id"].is_unique)
        self.assertEqual(
            features.set_index("incident_id").loc["incident-train", "stratification_key"],
            "maize::heat",
        )
        self.assertTrue(np.isfinite(features[list(MODEL_FEATURE_COLUMNS)].to_numpy(float)).all())
        forbidden = {
            "incident_id", "exposure_id", "field_id", "episode_id", "crop_instance_id",
            "timeline_bucket", "first_evidence_week", "last_evidence_week", "longitude",
            "latitude", "district", "sector",
        }
        self.assertFalse(forbidden.intersection(MODEL_FEATURE_COLUMNS))

    def test_temporal_split_embargoes_cutoff_crossing_stories(self) -> None:
        weekly, membership = _fixture()
        features = extract_completed_incident_features(weekly, membership)
        split = temporal_split_completed_stories(features, date(2025, 2, 3)).set_index(
            "incident_id"
        )

        self.assertEqual(split.loc["incident-train", "temporal_split"], "train")
        self.assertEqual(split.loc["incident-cross", "temporal_split"], "embargo")
        self.assertEqual(split.loc["incident-holdout", "temporal_split"], "holdout")
        self.assertIn("incident-cross", split.index)

    def test_prefixes_are_causal_and_use_only_supported_ages(self) -> None:
        weekly, membership = _fixture()
        before = build_causal_prefix_features(weekly, membership)
        changed = weekly.copy()
        late = (changed["incident_id"] == "incident-train") & (
            pd.to_datetime(changed["timeline_bucket"]) >= pd.Timestamp("2025-01-20")
        )
        changed.loc[late, ["affected_count", "active_count"]] = [8, 8]
        changed.loc[late, "footprint_area_km2"] = 500.0
        changed.loc[late, "hazard_intensity_mean"] = 999.0
        after = build_causal_prefix_features(changed, membership)

        early_before = before[
            (before["incident_id"] == "incident-train") & (before["horizon_weeks"] == 2)
        ][list(MODEL_FEATURE_COLUMNS)].reset_index(drop=True)
        early_after = after[
            (after["incident_id"] == "incident-train") & (after["horizon_weeks"] == 2)
        ][list(MODEL_FEATURE_COLUMNS)].reset_index(drop=True)
        assert_frame_equal(early_before, early_after)
        self.assertEqual(
            before.loc[before["incident_id"] == "incident-train", "horizon_weeks"].tolist(),
            [1, 2, 4],
        )
        self.assertEqual(supported_prefix_horizon(7), 4)
        self.assertIsNone(supported_prefix_horizon(0))

    def test_robust_transform_imputes_nonfinite_values_without_metadata(self) -> None:
        weekly, membership = _fixture()
        features = extract_completed_incident_features(weekly, membership)
        features.loc[features.index[0], "hazard_intensity_mean"] = np.inf
        features.loc[features.index[1], "maximum_area_km2"] = np.nan
        schema = fit_robust_feature_schema(features)
        matrix = transform_finite_feature_matrix(features, schema)

        self.assertEqual(matrix.shape, (len(features), len(MODEL_FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(matrix).all())
        unknown = features.iloc[[0]].copy()
        unknown["crop_name"] = "Unsupported crop"
        with self.assertRaisesRegex(ValueError, "no fitted stratum"):
            transform_finite_feature_matrix(unknown, schema)

        weighting = schema["feature_weighting"]
        self.assertEqual(weighting["method"], "equal_l2_energy_per_semantic_family")
        self.assertEqual(
            set(weighting["groups"]),
            set(FEATURE_GROUPS),
        )
        for group in weighting["groups"].values():
            self.assertAlmostEqual(
                len(group["features"]) * group["per_feature_weight"] ** 2,
                1.0,
            )

    def test_prefix_prototypes_and_open_set_assignment(self) -> None:
        training = pd.DataFrame(
            [
                _model_row("a-1", 0.09), _model_row("a-2", 0.10),
                _model_row("a-3", 0.11), _model_row("b-1", 0.89),
                _model_row("b-2", 0.90), _model_row("b-3", 0.91),
            ]
        )
        assignments = pd.DataFrame(
            {
                "incident_id": ["a-1", "a-2", "a-3", "b-1", "b-2", "b-3"],
                "archetype_id": ["archetype-a"] * 3 + ["archetype-b"] * 3,
            }
        )
        artifacts = fit_prefix_prototypes(
            training, assignments, model_version="v3-test", minimum_support=3
        )
        self.assertEqual(len(artifacts.prototypes), 2)
        self.assertTrue((artifacts.prototypes["radius"] > 0).all())

        queries = pd.DataFrame(
            [
                _model_row("tentative", 0.10, horizon=2),
                _model_row("tentative", 0.10, horizon=1),
                _model_row("novel", 0.50),
                _model_row("pending", 0.10, crop="Beans"),
            ]
        )
        result = assign_open_set_prefixes(queries, artifacts).set_index("incident_id")
        self.assertEqual(result.loc["tentative", "assignment_status"], "tentative")
        self.assertEqual(result.loc["tentative", "horizon_weeks"], 1)
        self.assertEqual(result.loc["novel", "assignment_status"], "novel")
        self.assertEqual(result.loc["pending", "assignment_status"], "pending")
        self.assertEqual(set(result.index), {"tentative", "novel", "pending"})

        wide = artifacts.prototypes.copy()
        wide["radius"] = 10.0
        ambiguous = assign_open_set_prefixes(
            pd.DataFrame([_model_row("ambiguous", 0.50)]),
            PrefixPrototypeArtifacts(artifacts.feature_schema, wide),
            runner_up_margin=0.05,
        ).iloc[0]
        self.assertEqual(ambiguous["assignment_status"], "novel")
        self.assertEqual(ambiguous["assignment_reason"], "ambiguous_runner_up")
        self.assertEqual(ambiguous["incident_id"], "ambiguous")

    def test_inputs_fail_closed_on_missing_or_inconsistent_lineage(self) -> None:
        weekly, membership = _fixture()
        with self.assertRaisesRegex(ValueError, "missing columns"):
            extract_completed_incident_features(
                weekly.drop(columns="affected_count"), membership
            )
        inconsistent = weekly.copy()
        mask = inconsistent["incident_id"] == "incident-train"
        inconsistent.loc[inconsistent.index[mask][-1], "crop_name"] = "Beans"
        with self.assertRaisesRegex(ValueError, "one invariant crop_name"):
            extract_completed_incident_features(inconsistent, membership)


if __name__ == "__main__":
    unittest.main()
