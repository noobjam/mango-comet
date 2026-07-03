from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from story_monitor.archetype_workflow_v2 import (
    build_archetype_model,
    evaluate_archetype_release,
)
from story_monitor.archetypes_v2 import ArchetypeConfig
from test_archetypes_v2 import _training_rows
from weekly_story_monitor import build_parser


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArchetypeWorkflowV2Tests(unittest.TestCase):
    def test_cli_exposes_only_versioned_phase_a_commands(self) -> None:
        parser = build_parser()
        build = parser.parse_args(
            [
                "build-archetypes-v2",
                "--generation-dir", "/tmp/generation",
                "--training-through", "2025-12-31",
                "--output-dir", "/tmp/model",
                "--engine", "gpu",
            ]
        )
        evaluate = parser.parse_args(
            [
                "evaluate-archetypes-v2",
                "--model-dir", "/tmp/model",
                "--output-dir", "/tmp/evaluation",
            ]
        )
        self.assertEqual(build.command, "build-archetypes-v2")
        self.assertEqual(build.engine, "gpu")
        self.assertEqual(evaluate.command, "evaluate-archetypes-v2")
        self.assertEqual(evaluate.stability_runs, 2)

    def test_atomic_build_and_separate_evaluation(self) -> None:
        training = _training_rows()
        training["anchor_date"] = pd.Timestamp("2025-06-01")
        holdout = training.iloc[:18].copy()
        holdout["event_id"] = [f"holdout-{index:03d}" for index in range(len(holdout))]
        holdout["field_id"] = [f"holdout-field-{index:03d}" for index in range(len(holdout))]
        holdout["anchor_date"] = pd.Timestamp("2026-02-01")
        anchors = pd.concat([training, holdout], ignore_index=True)
        anchors["eligible_for_training"] = True
        anchors["anchor_outcome"] = "eligible"
        anchors["anchor_status"] = "eligible"
        anchors["anchor_kind"] = "day_21"
        anchors["evidence_max_date"] = anchors["anchor_date"]
        anchors["spectral_source_max_date"] = anchors["anchor_date"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            generation.mkdir()
            (generation / "manifest.json").write_text(
                json.dumps(
                    {
                        "run": {
                            "status": "complete",
                            "generation_id": "generation-test",
                            "as_of_date": "2026-05-17",
                        },
                        "policy": {"version": "test-policy", "sha256": "f" * 64},
                    }
                ),
                encoding="utf-8",
            )
            model = root / "model"

            with self.assertRaisesRegex(ValueError, "source generation"):
                build_archetype_model(
                    generation,
                    generation / "model",
                    training_cutoff="2025-12-31",
                    config=ArchetypeConfig(
                        min_cluster_floor=8,
                        min_cluster_fraction=0,
                        minimum_field_support=4,
                    ),
                )

            def fake_anchor_writer(_generation: Path, output: Path, **_: object) -> Path:
                anchors.to_parquet(output, index=False)
                return output

            with patch(
                "story_monitor.archetype_workflow_v2.write_event_anchors",
                side_effect=fake_anchor_writer,
            ):
                result = build_archetype_model(
                    generation,
                    model,
                    training_cutoff="2025-12-31",
                    config=ArchetypeConfig(
                        min_cluster_floor=8,
                        min_cluster_fraction=0,
                        min_samples=3,
                        minimum_field_support=4,
                        assignment_margin=0,
                    ),
                    threads=1,
                )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["anchor_counts"]["training_events"], len(training))
            self.assertEqual(result["anchor_counts"]["holdout_events"], len(holdout))
            manifest = json.loads((model / "archetype_manifest.json").read_text())
            self.assertTrue(manifest["artifacts"])
            before = {path.name: _digest(path) for path in model.iterdir() if path.is_file()}

            manifest_path = model / "archetype_manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            wrong_implementation = dict(manifest)
            wrong_implementation["implementation_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(wrong_implementation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implementation hash differs"):
                evaluate_archetype_release(
                    model, root / "wrong-implementation-eval", stability_runs=2
                )
            manifest_path.write_text(original_manifest, encoding="utf-8")
            before = {path.name: _digest(path) for path in model.iterdir() if path.is_file()}

            with self.assertRaisesRegex(ValueError, "frozen model"):
                evaluate_archetype_release(
                    model, model / "evaluation", stability_runs=2
                )

            evaluation = root / "evaluation"
            evaluated = evaluate_archetype_release(model, evaluation, stability_runs=2)
            after = {path.name: _digest(path) for path in model.iterdir() if path.is_file()}

            self.assertEqual(before, after, "evaluation must not mutate the frozen model")
            self.assertEqual(evaluated["status"], "complete")
            self.assertTrue((evaluation / "evaluation.json").is_file())
            self.assertTrue((evaluation / "subsample_stability.parquet").is_file())
            assignments = pd.read_parquet(
                evaluation / "event_archetype_assignments.parquet"
            )
            self.assertEqual(len(assignments), len(training) + len(holdout))
            self.assertTrue(assignments["event_id"].is_unique)
            self.assertEqual(set(assignments["split"]), {"training", "holdout"})
            self.assertEqual(
                set(assignments["assignment_method"]), {"frozen_prototype_radius_v2"}
            )

            with self.assertRaises(FileExistsError):
                evaluate_archetype_release(model, evaluation, stability_runs=2)


if __name__ == "__main__":
    unittest.main()
