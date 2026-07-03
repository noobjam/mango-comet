from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from build_story_map_bundle import build_bundle
from story_map_server import Settings, StoryMapStore
from story_monitor.archetype_preview_v2 import export_archetype_preview
from story_monitor.runner_process import RunnerError


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


class ArchetypePreviewV2Tests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, dict, dict, dict]:
        generation, model, evaluation = root / "generation", root / "model", root / "evaluation"
        generation.mkdir(); model.mkdir(); evaluation.mkdir()
        generation_payload = {
            "run": {"status": "complete", "immutable": True, "generation_id": "g1",
                    "as_of_date": "2025-01-31", "event_count": 4, "field_count": 4,
                    "row_count": 12},
            "input": {"max_fields": None},
            "policy": {"version": "p1", "sha256": "a" * 64},
        }
        (generation / "manifest.json").write_text(json.dumps(generation_payload))
        rows = []
        snapshots = []
        memberships = []
        for event, field, hazard in (
            ("accepted", "field-a", "heat"),
            ("novel", "field-b", "heat"),
            ("watch", "field-c", "drought"),
            ("training", "field-d", "heat"),
        ):
            for bucket, observed in (("2025-01-06", "2025-01-09"),
                                     ("2025-01-13", "2025-01-14"),
                                     ("2025-01-20", "2025-01-20")):
                common = {
                    "timeline_bucket": bucket, "field_id": field,
                    "story_cluster_id": event, "event_id": event,
                    "event_state": "ACTIVE", "crop_name": "Maize", "crop_season": "A",
                    "max_risk_band": "MED-HIGH", "current_risk_band": "MED-HIGH",
                    "hazard_signature": hazard, "motif_family": hazard,
                    "response_signature": "medium_decline", "reportable_day_count": 1,
                    "event_count": 1, "max_risk_rank": 3, "current_risk_rank": 3,
                    "response_day_count": 1,
                }
                rows.append(common)
                snapshots.append({**common, "snapshot_as_of_date": observed})
                memberships.append({
                    "field_id": field, "event_id": event, "story_cluster_id": event,
                    "crop_instance_id": f"crop-{field}", "observation_date": observed,
                    "event_state": "ACTIVE", "hazard_signature": hazard,
                    "daily_pressure_rank": 3, "daily_response_class": "medium_decline",
                    "pressure_observed": True,
                })
        _write(pd.DataFrame(rows), generation / "map_frame_fields.parquet")
        _write(pd.DataFrame(snapshots), generation / "event_state_snapshots.parquet")
        _write(pd.DataFrame(memberships), generation / "story_day_membership.parquet")
        events = pd.DataFrame([
            {"field_id": field, "crop_name": "Maize", "crop_season": "A",
             "event_id": event, "event_start_date": "2025-01-01", "active_end_date": "2025-01-20",
             "event_end_date": "2025-01-20", "event_state": "ACTIVE",
             "max_risk_band": "MED-HIGH", "max_risk_rank": 3,
             "hazard_signature": hazard, "stage_signature": "vegetative",
             "response_signature": "medium_decline", "close_reason": None,
             "reportable_days": 3, "window_span_days": 20, "story_cluster_id": event,
             "right_censored": False, "as_of_date": "2025-01-31", "requires_review": False}
            for event, field, hazard in (("accepted", "field-a", "heat"),
                                         ("novel", "field-b", "heat"),
                                         ("watch", "field-c", "drought"),
                                         ("training", "field-d", "heat"))
        ])
        _write(events, generation / "event_windows.parquet")
        _write(pd.DataFrame({"field_id": ["field-a", "field-b", "field-c", "field-d"],
                             "geometry_wkt": ["POINT (30 -1)", "POINT (31 -1)",
                                              "POINT (32 -1)", "POINT (33 -1)"]}),
               generation / "map_field_geometry.parquet")

        anchors = pd.DataFrame([
            {"event_id": "accepted", "field_id": "field-a", "hazard_family": "heat",
             "anchor_date": "2025-01-15", "anchor_kind": "day_21", "anchor_outcome": "eligible",
             "anchor_status": "eligible", "eligible_for_training": True},
            {"event_id": "novel", "field_id": "field-b", "hazard_family": "heat",
             "anchor_date": "2025-01-15", "anchor_kind": "day_21", "anchor_outcome": "eligible",
             "anchor_status": "eligible", "eligible_for_training": True},
            {"event_id": "watch", "field_id": "field-c", "hazard_family": "drought",
             "anchor_date": "2025-01-10", "anchor_kind": "early_closure", "anchor_outcome": "watch_only",
             "anchor_status": "watch_only", "eligible_for_training": False},
            {"event_id": "training", "field_id": "field-d", "hazard_family": "heat",
             "anchor_date": "2025-01-10", "anchor_kind": "day_21", "anchor_outcome": "eligible",
             "anchor_status": "eligible", "eligible_for_training": True},
        ])
        _write(anchors, model / "event_anchors.parquet")
        _write(pd.DataFrame([{"archetype_id": "heat-a1", "hazard_family": "heat",
                              "label": "Heat archetype 1"}]), model / "archetype_catalog.parquet")
        registry = pd.DataFrame([
            {"event_id": "accepted", "field_id": "field-a", "hazard_family": "heat",
             "archetype_id": "heat-a1", "accepted": True, "split": "holdout",
             "assignment_method": "frozen_prototype_radius_v2", "assignment_reason": "within_radius_and_margin",
             "candidate_archetype_id": "heat-a1", "runner_up_archetype_id": None,
             "assignment_distance": .2, "candidate_radius": .5, "distance_ratio": .4,
             "assignment_margin": .3, "model_version": "m1", "feature_schema_sha256": "f" * 64},
            {"event_id": "novel", "field_id": "field-b", "hazard_family": "heat",
             "archetype_id": "novel_unassigned", "accepted": False, "split": "holdout",
             "assignment_method": "frozen_prototype_radius_v2", "assignment_reason": "outside_radius_or_ambiguous",
             "candidate_archetype_id": "heat-a1", "runner_up_archetype_id": None,
             "assignment_distance": .8, "candidate_radius": .5, "distance_ratio": 1.6,
             "assignment_margin": .1, "model_version": "m1", "feature_schema_sha256": "f" * 64},
            {"event_id": "training", "field_id": "field-d", "hazard_family": "heat",
             "archetype_id": "heat-a1", "accepted": True, "split": "training",
             "assignment_method": "frozen_prototype_radius_v2", "assignment_reason": "within_radius_and_margin",
             "candidate_archetype_id": "heat-a1", "runner_up_archetype_id": None,
             "assignment_distance": .2, "candidate_radius": .5, "distance_ratio": .4,
             "assignment_margin": .3, "model_version": "m1", "feature_schema_sha256": "f" * 64},
        ])
        _write(registry, evaluation / "event_archetype_assignments.parquet")
        (model / "archetype_manifest.json").write_text("{}")
        (evaluation / "evaluation_manifest.json").write_text("{}")
        model_payload = {"model_version": "m1", "archetype_count": 1,
                         "feature_schema_sha256": "f" * 64,
                         "training_cutoff": "2025-01-14"}
        report = {"gates": {"hard": {"passed": True, "checks": {"lineage": True}},
                             "quality": {"passed": False, "checks": {"holdout": False}}},
                  "metrics": {"holdout_accepted_rate": .5}}
        return generation, model, evaluation, generation_payload, model_payload, report

    def test_causal_preview_and_viewer_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, model, evaluation, gen_meta, model_meta, report = self._fixture(root)
            output = root / "preview"
            with patch("story_monitor.archetype_preview_v2.validate_generation", return_value=gen_meta), \
                 patch("story_monitor.archetype_preview_v2.validate_model", return_value=model_meta), \
                 patch("story_monitor.archetype_preview_v2.validate_evaluation",
                       return_value=(report, True, False)):
                result = export_archetype_preview(
                    generation, model, evaluation, output,
                    allow_failed_quality_gates=True, threads=1,
                )
            frames = pd.read_parquet(output / "map_frame_fields.parquet")
            accepted = frames[frames.event_id == "accepted"].sort_values("timeline_bucket")
            self.assertEqual(accepted.iloc[1].archetype_display_state, "pending_anchor")
            self.assertTrue(pd.isna(accepted.iloc[1].archetype_id))
            self.assertEqual(accepted.iloc[2].story_cluster_id, "heat-a1")
            novel = frames[(frames.event_id == "novel") & (frames.timeline_bucket == "2025-01-20")].iloc[0]
            self.assertEqual(novel.story_cluster_id, "diag:v2:heat:novel_unassigned")
            watch = frames[frames.event_id == "watch"].sort_values("timeline_bucket")
            self.assertEqual(watch.iloc[0].archetype_display_state, "pending_anchor")
            self.assertEqual(watch.iloc[1].archetype_display_state, "watch_only")
            training = frames[frames.event_id == "training"].sort_values("timeline_bucket")
            self.assertEqual(training.iloc[0].archetype_display_state, "pending_anchor")
            self.assertTrue((training.iloc[1:].archetype_display_state == "calibration_training").all())
            self.assertTrue(training.archetype_id.isna().all())
            labels = pd.read_parquet(output / "event_story_cluster_labels.parquet")
            self.assertTrue(labels.short_label.str.startswith("DIAGNOSTIC —").all())
            self.assertFalse(result["quality_gates_passed"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["run"]["publication_approved"])
            bundle = root / "bundle"
            build_bundle(output, bundle)
            self.assertTrue((bundle / "frame_fields.parquet").is_file())
            self.assertTrue(json.loads((bundle / "manifest.json").read_text())["run"]["viewer_ready"])
            store = StoryMapStore(Settings(
                run_dir=bundle, static_dir=Path(__file__).parent / "static",
                host="127.0.0.1", port=0, raster_tiles="", raster_attribution="",
                default_feature_limit=100, max_feature_limit=1000, log_level="ERROR",
            ))
            state = store.frame_state(
                timeline_bucket="2025-01-20", bbox=None, limit=0,
                filters={"story_cluster_id": "heat-a1"},
            )
            self.assertEqual(state["rows"][0]["archetype_display_state"], "accepted")
            trajectory = store.field_trajectory("field-a", 100)
            self.assertEqual(
                [row["archetype_display_state"] for row in trajectory["states"]],
                ["pending_anchor", "pending_anchor", "accepted"],
            )
            exact_trail = store.trail_features(
                timeline_bucket="2025-01-20", filters={"story_cluster_id": "heat-a1"},
                lookback=8, limit=0,
            )
            self.assertEqual(exact_trail["features"], [])

    def test_quality_and_hard_gates_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, model, evaluation, gen_meta, model_meta, report = self._fixture(root)
            patches = (
                patch("story_monitor.archetype_preview_v2.validate_generation", return_value=gen_meta),
                patch("story_monitor.archetype_preview_v2.validate_model", return_value=model_meta),
            )
            with patches[0], patches[1], patch(
                "story_monitor.archetype_preview_v2.validate_evaluation",
                return_value=(report, True, False),
            ):
                with self.assertRaisesRegex(RunnerError, "Quality gates failed"):
                    export_archetype_preview(generation, model, evaluation, root / "quality")
            bad = {**report, "gates": {**report["gates"],
                   "hard": {"passed": True, "checks": {"lineage": False}}}}
            with patches[0], patches[1], patch(
                "story_monitor.archetype_preview_v2.validate_evaluation",
                return_value=(bad, True, False),
            ):
                with self.assertRaisesRegex(RunnerError, "hard gates"):
                    export_archetype_preview(
                        generation, model, evaluation, root / "hard",
                        allow_failed_quality_gates=True,
                    )

    def test_registry_split_cannot_backdate_a_learned_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, model, evaluation, gen_meta, model_meta, report = self._fixture(root)
            registry_path = evaluation / "event_archetype_assignments.parquet"
            registry = pd.read_parquet(registry_path)
            registry.loc[registry.event_id == "accepted", "split"] = "training"
            _write(registry, registry_path)
            with patch("story_monitor.archetype_preview_v2.validate_generation", return_value=gen_meta), \
                 patch("story_monitor.archetype_preview_v2.validate_model", return_value=model_meta), \
                 patch("story_monitor.archetype_preview_v2.validate_evaluation",
                       return_value=(report, True, False)):
                with self.assertRaisesRegex(ValueError, "inconsistent model, hazard, or archetype"):
                    export_archetype_preview(
                        generation, model, evaluation, root / "preview",
                        allow_failed_quality_gates=True,
                    )


if __name__ == "__main__":
    unittest.main()
