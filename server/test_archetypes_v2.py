from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from story_monitor.anchor_features_v2 import FEATURE_SCHEMA, MODEL_FEATURE_COLUMNS
from story_monitor.archetypes_v2 import (
    ArchetypeConfig,
    NOVEL_ARCHETYPE_ID,
    adaptive_min_cluster_size,
    assign_frozen_archetypes,
    discover_archetypes,
    evaluate_archetype_model,
    fit_feature_schema,
    prototype_overlap,
    transform_features,
)


BOUNDED = {
    item["name"] for item in FEATURE_SCHEMA["features"] if item["kind"] in {"binary", "bounded"}
}
BINARY = {item["name"] for item in FEATURE_SCHEMA["features"] if item["kind"] == "binary"}


def _training_rows() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for cluster, center in enumerate((0.15, 0.82)):
        for index in range(45):
            row: dict[str, object] = {
                "event_id": f"event-{cluster}-{index:03d}",
                "field_id": f"field-{cluster}-{index:03d}",
                "hazard_family": "heat",
                "anchor_date": pd.Timestamp("2025-06-01") + pd.Timedelta(days=index),
            }
            for feature_index, name in enumerate(MODEL_FEATURE_COLUMNS):
                if name in BINARY:
                    value = float(cluster)
                elif name in BOUNDED:
                    value = float(np.clip(center + rng.normal(0, 0.025), 0, 1))
                else:
                    value = float(center * (feature_index + 1) + rng.normal(0, 0.04))
                row[name] = value
            if index % 9 == 0:
                row["worst_attributed_ndvi_delta"] = np.nan
                row["worst_attributed_ndvi_delta_missing"] = 1.0
            else:
                row["worst_attributed_ndvi_delta_missing"] = 0.0
            row["worst_attributed_ndmi_delta_missing"] = 0.0
            row["worst_attributed_psri_delta_missing"] = 0.0
            row["hazard_intensity_missing"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


class ArchetypeV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _training_rows()
        self.config = ArchetypeConfig(
            min_cluster_floor=8,
            min_cluster_fraction=0,
            min_samples=3,
            minimum_field_support=4,
            assignment_margin=0,
            engine="cpu",
        )

    def test_schema_is_hazard_local_bounded_and_finite(self) -> None:
        drought = self.rows.iloc[:10].copy()
        drought["event_id"] = [f"drought-{index}" for index in range(len(drought))]
        drought["hazard_family"] = "drought"
        drought["peak_risk_rank"] = drought["peak_risk_rank"] + 100
        rows = pd.concat([self.rows, drought], ignore_index=True)
        schema = fit_feature_schema(rows)
        heat_specs = {item["name"]: item for item in schema["hazards"]["heat"]["features"]}
        drought_specs = {
            item["name"]: item for item in schema["hazards"]["drought"]["features"]
        }

        self.assertNotEqual(
            heat_specs["peak_risk_rank"]["median"], drought_specs["peak_risk_rank"]["median"]
        )
        self.assertEqual(heat_specs["high_day_fraction"]["kind"], "bounded")
        matrix = transform_features(rows, schema)
        self.assertEqual(matrix.shape, (len(rows), len(MODEL_FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertLessEqual(float(np.abs(matrix).max()), 5.0)

    def test_discovery_and_frozen_assignment_are_one_row_per_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            manifest = discover_archetypes(
                self.rows,
                model_dir,
                config=self.config,
                training_cutoff="2025-12-31",
                policy_version="test-policy",
                policy_sha256="a" * 64,
            )
            near = self.rows.iloc[:8].copy()
            near["event_id"] = [f"holdout-{index}" for index in range(len(near))]
            unknown = near.iloc[[0]].copy()
            unknown["event_id"] = "unknown-hazard"
            unknown["hazard_family"] = "unknown"
            assignments = assign_frozen_archetypes(
                pd.concat([near, unknown], ignore_index=True), model_dir
            )

            self.assertEqual(manifest["feature_version"], "causal_event_anchor_features_v2")
            self.assertGreaterEqual(manifest["archetype_count"], 2)
            self.assertTrue(assignments["event_id"].is_unique)
            # A 95th-percentile training radius intentionally rejects a small
            # tail even when replaying observed training-shaped rows.
            self.assertGreaterEqual(int(assignments.iloc[: len(near)]["accepted"].sum()), 7)
            self.assertEqual(assignments.iloc[-1]["archetype_id"], NOVEL_ARCHETYPE_ID)
            self.assertEqual(assignments.iloc[-1]["assignment_reason"], "no_hazard_prototype")
            self.assertTrue((model_dir / "archetype_manifest.json").is_file())
            self.assertTrue((model_dir / "feature_schema.json").is_file())
            self.assertTrue((model_dir / "training_assignments.parquet").is_file())

    def test_evaluation_is_fail_closed_until_stability_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            discover_archetypes(
                self.rows,
                model_dir,
                config=self.config,
                training_cutoff="2025-12-31",
                policy_version="test-policy",
                policy_sha256="b" * 64,
            )
            holdout = self.rows.iloc[:12].copy()
            holdout["event_id"] = [f"future-{index}" for index in range(len(holdout))]
            evaluation_dir = Path(directory) / "evaluation"
            model_files_before = sorted(path.name for path in model_dir.iterdir())
            report = evaluate_archetype_model(
                self.rows, holdout, model_dir, output_dir=evaluation_dir
            )

            self.assertFalse(report["gates"]["quality"]["passed"])
            self.assertFalse(report["gates"]["quality"]["checks"]["subsample_stability"])
            self.assertEqual(report["metrics"]["stability_status"], "not_run")
            self.assertEqual(
                sorted(path.name for path in model_dir.iterdir()), model_files_before
            )
            self.assertTrue((evaluation_dir / "training_frozen_assignments.parquet").is_file())
            self.assertTrue((evaluation_dir / "holdout_assignments.parquet").is_file())
            self.assertTrue((evaluation_dir / "evaluation.json").is_file())

    def test_duplicate_event_is_rejected_and_adaptive_default_is_exact(self) -> None:
        duplicate = pd.concat([self.rows, self.rows.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one row per event_id"):
                discover_archetypes(
                    duplicate,
                    Path(directory) / "model",
                    config=self.config,
                    training_cutoff="2025-12-31",
                    policy_version="test-policy",
                    policy_sha256="c" * 64,
                )
        self.assertEqual(adaptive_min_cluster_size(1_000, ArchetypeConfig()), 100)
        self.assertEqual(adaptive_min_cluster_size(100_001, ArchetypeConfig()), 501)

    def test_invalid_bounded_or_missing_indicator_values_fail_closed(self) -> None:
        invalid = self.rows.copy()
        invalid.loc[0, "high_day_fraction"] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, r"high_day_fraction must be in \[0, 1\]"):
                discover_archetypes(
                    invalid,
                    Path(directory) / "model",
                    config=self.config,
                    training_cutoff="2025-12-31",
                    policy_version="test-policy",
                    policy_sha256="d" * 64,
                )

        inconsistent = self.rows.copy()
        inconsistent.loc[1, "worst_attributed_ndmi_delta_missing"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not match"):
                discover_archetypes(
                    inconsistent,
                    Path(directory) / "model",
                    config=self.config,
                    training_cutoff="2025-12-31",
                    policy_version="test-policy",
                    policy_sha256="e" * 64,
                )

    def test_assignment_rejects_mixed_or_tampered_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            discover_archetypes(
                self.rows,
                model_dir,
                config=self.config,
                training_cutoff="2025-12-31",
                policy_version="test-policy",
                policy_sha256="f" * 64,
            )
            prototypes = pd.read_parquet(model_dir / "prototypes.parquet")
            prototypes.loc[0, "radius"] = float(prototypes.loc[0, "radius"]) * 2
            prototypes.to_parquet(model_dir / "prototypes.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                assign_frozen_archetypes(self.rows.iloc[:2], model_dir)

    def test_overlap_checks_every_same_hazard_pair_not_only_nearest_centers(self) -> None:
        records = []
        for archetype_id, position, radius in (
            ("a", 0.0, 0.1),
            ("b", 1.0, 0.1),
            ("c", 3.0, 3.2),
        ):
            record = {
                "archetype_id": archetype_id,
                "hazard_family": "heat",
                "radius": radius,
            }
            record.update(
                {
                    f"f_{index:03d}": position if index == 0 else 0.0
                    for index in range(len(MODEL_FEATURE_COLUMNS))
                }
            )
            records.append(record)
        overlap = prototype_overlap(pd.DataFrame(records))
        self.assertEqual(len(overlap), 3)
        pair = overlap[
            overlap["archetype_id"].eq("a")
            & overlap["other_archetype_id"].eq("c")
        ].iloc[0]
        self.assertGreater(pair["overlap_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
