from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from story_monitor.anchor_features_v2 import FEATURE_SCHEMA, MODEL_FEATURE_COLUMNS
from story_monitor.archetype_stability_v2 import (
    DEFAULT_SEEDS,
    evaluate_subsample_stability,
)
from story_monitor.archetypes_v2 import ArchetypeConfig, discover_archetypes


BOUNDED_FEATURES = {
    item["name"]
    for item in FEATURE_SCHEMA["features"]
    if item["kind"] == "bounded"
}
BINARY_FEATURES = {
    item["name"]
    for item in FEATURE_SCHEMA["features"]
    if item["kind"] == "binary"
}


def _well_separated_events() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    records: list[dict[str, object]] = []
    for hazard in ("drought", "heat"):
        for cluster_index, center in enumerate((0.12, 0.86)):
            for event_index in range(45):
                record: dict[str, object] = {
                    "event_id": f"{hazard}-event-{cluster_index}-{event_index:03d}",
                    "field_id": f"{hazard}-field-{cluster_index}-{event_index:03d}",
                    "hazard_family": hazard,
                }
                for feature_index, name in enumerate(MODEL_FEATURE_COLUMNS):
                    if name in BINARY_FEATURES:
                        value = float(cluster_index)
                    elif name in BOUNDED_FEATURES:
                        value = float(np.clip(center + rng.normal(0, 0.015), 0, 1))
                    else:
                        value = float(center * (feature_index + 1) + rng.normal(0, 0.02))
                    record[name] = value
                record["worst_attributed_ndvi_delta_missing"] = 0.0
                record["worst_attributed_ndmi_delta_missing"] = 0.0
                record["worst_attributed_psri_delta_missing"] = 0.0
                record["hazard_intensity_missing"] = 0.0
                records.append(record)
    return pd.DataFrame(records)


class ArchetypeStabilityV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _well_separated_events()
        self.config = ArchetypeConfig(
            min_cluster_floor=8,
            min_cluster_fraction=0,
            min_samples=3,
            minimum_field_support=4,
            assignment_margin=0,
            engine="cpu",
        )

    def _discover(self, directory: str) -> Path:
        model_dir = Path(directory) / "model"
        discover_archetypes(
            self.rows,
            model_dir,
            config=self.config,
            training_cutoff="2025-12-31",
            policy_version="stability-test-policy",
            policy_sha256="d" * 64,
        )
        return model_dir

    def test_two_deterministic_eighty_percent_refits_persist_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self._discover(directory)
            artifact_dir = Path(directory) / "evaluation"
            report = evaluate_subsample_stability(
                self.rows, model_dir, output_dir=artifact_dir, runs=2
            )

            self.assertEqual(report["method"]["run_count"], 2)
            self.assertEqual(report["method"]["sample_fraction"], 0.8)
            self.assertEqual(report["method"]["seeds"], list(DEFAULT_SEEDS))
            self.assertEqual(len(report["runs"]), 2)
            self.assertTrue(report["gates"]["passed"])
            self.assertEqual(report["metrics"]["reference_archetype_count"], 4)
            self.assertEqual(report["metrics"]["supported_reference_fraction"], 1.0)
            self.assertGreaterEqual(
                report["metrics"]["minimum_mutual_non_noise_adjusted_rand_index"],
                0.80,
            )
            self.assertEqual(
                report["method"]["feature_transform"],
                "refit_hazard_local_schema_per_subsample",
            )
            self.assertTrue(
                all(item["supported_both_runs"] for item in report["references"])
            )
            self.assertTrue(
                all(
                    hazard["sample_event_count"] == 72
                    for run in report["runs"]
                    for hazard in run["hazards"]
                )
            )
            # The returned object is safe to embed in evaluation.json without
            # NumPy scalars or non-finite JSON values.
            json.dumps(report, allow_nan=False)

            detail = pd.read_parquet(artifact_dir / "subsample_stability.parquet")
            self.assertEqual(int((detail["row_type"] == "hazard_summary").sum()), 4)
            self.assertEqual(int((detail["row_type"] == "archetype_match").sum()), 8)
            matches = detail[detail["row_type"] == "archetype_match"]
            self.assertTrue((matches["best_jaccard"] >= 0.70).all())
            self.assertTrue(matches["supported_both_runs"].astype(bool).all())
            persisted = json.loads(
                (artifact_dir / "subsample_stability.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["metrics"], report["metrics"])

            shuffled = self.rows.sample(frac=1, random_state=91).reset_index(drop=True)
            repeated = evaluate_subsample_stability(
                shuffled, model_dir, output_dir=artifact_dir, runs=2
            )
            self.assertEqual(repeated["runs"], report["runs"])
            self.assertEqual(repeated["references"], report["references"])
            self.assertEqual(repeated["metrics"], report["metrics"])

    def test_small_test_overrides_are_explicit_and_seed_count_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = self._discover(directory)
            report = evaluate_subsample_stability(
                self.rows,
                model_dir,
                output_dir=Path(directory) / "evaluation",
                sample_fraction=0.75,
                config_overrides={"min_cluster_floor": 6},
            )
            self.assertEqual(report["method"]["sample_fraction"], 0.75)
            self.assertEqual(report["method"]["config_overrides"], {"min_cluster_floor": 6})
            self.assertEqual(report["config"]["min_cluster_floor"], 6)
            self.assertTrue(
                all(
                    hazard["sample_event_count"] == 67
                    for run in report["runs"]
                    for hazard in run["hazards"]
                )
            )

            with self.assertRaisesRegex(ValueError, "exactly two"):
                evaluate_subsample_stability(
                    self.rows, model_dir,
                    output_dir=Path(directory) / "bad-seeds", seeds=(1,)
                )
            with self.assertRaisesRegex(ValueError, "exactly two"):
                evaluate_subsample_stability(
                    self.rows, model_dir,
                    output_dir=Path(directory) / "bad-runs", runs=3
                )
            with self.assertRaisesRegex(ValueError, "unknown archetype config"):
                evaluate_subsample_stability(
                    self.rows, model_dir,
                    output_dir=Path(directory) / "bad-config",
                    config_overrides={"not_a_setting": 1}
                )


if __name__ == "__main__":
    unittest.main()
