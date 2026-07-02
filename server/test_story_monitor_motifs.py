from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from story_monitor.motifs import (
    NOVEL_MOTIF_ID,
    DiscoveryConfig,
    assign_frozen_motifs,
    discover_motifs,
    fit_feature_schema,
    transform_features,
)
from story_monitor.motif_export import (
    _validate_model_compatibility,
    _write_enriched_frames,
    _write_motif_labels,
)


class MotifAssignmentTests(unittest.TestCase):
    def test_data_gap_carries_prior_assignment_and_prefix_age(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            pd.DataFrame(
                [
                    {
                        "event_id": "event-A", "timeline_bucket": "2025-01-01",
                        "event_state": "ACTIVE", "field_id": "field-A",
                        "max_risk_rank": 3, "hazard_signature": "heat",
                        "response_signature": "none", "crop_name": "Maize",
                        "reportable_day_count": 2,
                    },
                    {
                        "event_id": "event-A", "timeline_bucket": "2025-01-08",
                        "event_state": "DATA_GAP", "field_id": "field-A",
                        "max_risk_rank": 3, "hazard_signature": "heat",
                        "response_signature": "none", "crop_name": "Maize",
                        "reportable_day_count": 0,
                    },
                ]
            ).to_parquet(stage / "map_frame_fields.parquet", index=False)
            assignments = pd.DataFrame(
                [
                    {
                        "event_id": "event-A", "timeline_bucket": "2025-01-01",
                        "event_age_days": 5, "motif_id": "motif:heat",
                        "assignment_reason": "within_radius_and_margin",
                    }
                ]
            )
            _write_enriched_frames(stage, assignments)
            frames = pd.read_parquet(stage / "map_frame_fields.parquet")

        gap = frames.loc[frames["event_state"] == "DATA_GAP"].iloc[0]
        self.assertEqual(gap["motif_id"], "motif:heat")
        self.assertEqual(gap["assignment_reason"], "carried_through_data_gap")
        self.assertEqual(gap["event_age_days"], 5)

    def test_motif_labels_use_prefix_metrics_not_completed_event_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            pd.DataFrame(
                [
                    {
                        "motif_id": "motif:heat", "event_id": "event-A",
                        "field_id": "field-A", "max_risk_rank": 3,
                        "hazard_signature": "heat", "response_signature": "none",
                        "crop_name": "Maize", "event_age_days": 7,
                        "reportable_day_count": 2,
                    }
                ]
            ).to_parquet(stage / "map_frame_fields.parquet", index=False)
            catalog = pd.DataFrame(
                [{"motif_id": "motif:heat", "label": "Heat motif", "hazard_family": "heat"}]
            )
            _write_motif_labels(stage, catalog)
            label = pd.read_parquet(stage / "event_story_cluster_labels.parquet").iloc[0]

        self.assertEqual(label["median_window_span_days"], 7)
        self.assertEqual(label["median_reportable_days"], 2)

    def test_export_rejects_policy_hash_mismatch(self) -> None:
        source = {"policy": {"version": "policy-v1", "sha256": "source-hash"}}
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            schema = {
                "version": "causal_prefix_features_v1",
                "model_version": "model-v1",
                "policy_version": "policy-v1",
                "policy_sha256": "different-hash",
            }
            (model / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            (model / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "model_version": "model-v1", "policy_version": "policy-v1",
                        "policy_sha256": "different-hash",
                        "feature_version": "causal_prefix_features_v1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "policy hash"):
                _validate_model_compatibility(source, model)

    def test_schema_fit_is_row_order_invariant(self) -> None:
        rows = pd.DataFrame(
            [
                {"current_risk_rank": 3, "lifecycle_state": "ACTIVE", "stage_family": "vegetative"},
                {"current_risk_rank": 1, "lifecycle_state": "WATCH", "stage_family": "establishment"},
            ]
        )
        schema_a = fit_feature_schema(rows)
        schema_b = fit_feature_schema(rows.iloc[::-1].reset_index(drop=True))
        self.assertEqual(schema_a, schema_b)

    def test_frozen_assignment_accepts_near_and_rejects_far_prefix(self) -> None:
        training = pd.DataFrame(
            [{"current_risk_rank": 1, "hazard_family": "heat", "lifecycle_state": "WATCH"}]
        )
        schema = fit_feature_schema(training)
        schema.update(
            {
                "model_version": "test-model",
                "assignment_margin": 0.05,
                "novel_motif_id": NOVEL_MOTIF_ID,
            }
        )
        vector = transform_features(training, schema)[0]
        record = {
            "motif_id": "motif:heat",
            "hazard_family": "heat",
            "radius": 1.0,
        }
        record.update({f"f_{i:03d}": value for i, value in enumerate(vector)})
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
            pd.DataFrame([record]).to_parquet(model_dir / "prototypes.parquet", index=False)
            candidates = pd.DataFrame(
                [
                    {"current_risk_rank": 1, "hazard_family": "heat", "lifecycle_state": "WATCH"},
                    {"current_risk_rank": 100, "hazard_family": "heat", "lifecycle_state": "WATCH"},
                    {"current_risk_rank": 1, "hazard_family": "flood", "lifecycle_state": "WATCH"},
                ]
            )
            assigned = assign_frozen_motifs(candidates, model_dir)

        self.assertEqual(assigned["motif_id"].tolist(), ["motif:heat", NOVEL_MOTIF_ID, NOVEL_MOTIF_ID])
        self.assertEqual(assigned.loc[2, "assignment_reason"], "no_hazard_prototype")

    def test_density_discovery_publishes_noise_aware_frozen_model(self) -> None:
        rows = []
        for group, center in enumerate((1.0, 4.0)):
            for index in range(12):
                rows.append(
                    {
                        "event_id": f"event-{group}-{index:02d}",
                        "hazard_family": "heat",
                        "current_risk_rank": center + (index % 3) * 0.01,
                        "event_age_days": center * 10 + (index % 4) * 0.01,
                        "lifecycle_state": "WATCH" if group == 0 else "SEVERE",
                        "stage_family": "vegetative",
                        "response_class": "none" if group == 0 else "severe_decline",
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            manifest = discover_motifs(
                pd.DataFrame(rows),
                model_dir,
                config=DiscoveryConfig(min_cluster_size=5, min_samples=3),
                training_cutoff="2025-12-31",
                policy_version="starter_policy_v1",
            )
            catalog = pd.read_parquet(model_dir / "motif_catalog.parquet")
            training = pd.read_parquet(model_dir / "training_assignments.parquet")

        self.assertGreaterEqual(manifest["motif_count"], 2)
        self.assertTrue((catalog["status"] == "discovered_unreviewed").all())
        self.assertTrue((training["discovery_label"] >= -1).all())


if __name__ == "__main__":
    unittest.main()
