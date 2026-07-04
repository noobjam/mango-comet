from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from story_monitor import incident_workflow_v3 as workflow_v3
from story_monitor.incident_policy_v3 import load_incident_policy_v3
from story_monitor.incident_workflow_v3 import build_incident_generation_v3
from story_monitor.incident_validation_v3 import REQUIRED_SOURCE_ARTIFACTS
from test_incident_context_v3 import _write_generation
from weekly_story_monitor import build_parser


def _write_workflow_generation(root: Path) -> Path:
    generation = root / "source"
    generation.mkdir()
    _write_generation(generation)
    signals_path = generation / "daily_causal_signals.parquet"
    signals = pd.read_parquet(signals_path)
    baseline_signal = signals.iloc[0].copy()
    baseline_signal["observation_date"] = "2024-01-08"
    pd.concat([pd.DataFrame([baseline_signal]), signals], ignore_index=True).to_parquet(
        signals_path, index=False
    )

    snapshots_path = generation / "event_state_snapshots.parquet"
    snapshots = pd.read_parquet(snapshots_path)
    baseline_snapshot = snapshots.iloc[0].copy()
    baseline_snapshot["event_id"] = "event-baseline"
    baseline_snapshot["timeline_bucket"] = "2024-01-08"
    baseline_snapshot["snapshot_as_of_date"] = "2024-01-08"
    pd.concat([pd.DataFrame([baseline_snapshot]), snapshots], ignore_index=True).to_parquet(
        snapshots_path, index=False
    )

    events_path = generation / "event_windows.parquet"
    events = pd.read_parquet(events_path)
    baseline_event = events.iloc[0].copy()
    baseline_event["event_id"] = "event-baseline"
    baseline_event["event_start_date"] = "2024-01-08"
    pd.concat([pd.DataFrame([baseline_event]), events], ignore_index=True).to_parquet(
        events_path, index=False
    )

    memberships_path = generation / "story_day_membership.parquet"
    memberships = pd.read_parquet(memberships_path)
    baseline_membership = memberships.iloc[0].copy()
    baseline_membership["event_id"] = "event-baseline"
    baseline_membership["observation_date"] = "2024-01-08"
    pd.concat(
        [pd.DataFrame([baseline_membership]), memberships], ignore_index=True
    ).to_parquet(memberships_path, index=False)

    source_manifest = json.loads((generation / "manifest.json").read_text())
    source_manifest["policy"] = {"version": "source-v1", "sha256": "f" * 64}
    (generation / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    return generation


class IncidentWorkflowV3Tests(unittest.TestCase):
    def test_cli_exposes_versioned_incident_build(self) -> None:
        parsed = build_parser().parse_args(
            [
                "build-incidents-v3", "--generation-dir", "/tmp/source",
                "--output-dir", "/tmp/v3", "--baseline-through", "2025-12-31",
                "--threads", "32", "--previous-incident-dir", "/tmp/previous",
            ]
        )
        self.assertEqual(parsed.command, "build-incidents-v3")
        self.assertEqual(parsed.threads, 32)
        self.assertEqual(parsed.baseline_through.isoformat(), "2025-12-31")
        self.assertEqual(parsed.previous_incident_dir, Path("/tmp/previous"))

    def test_atomic_end_to_end_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = _write_workflow_generation(root)
            policy = replace(
                load_incident_policy_v3(),
                minimum_evaluable_fields=1,
                minimum_active_fields=1,
                severe_override_min_fields=1,
                minimum_source_field_centroid_coverage=0.5,
                minimum_source_crop_instance_week_centroid_coverage=0.5,
                minimum_known_stage_coverage=0.5,
            )
            output = root / "incident-v3"
            result = build_incident_generation_v3(
                generation, output, baseline_through="2024-01-14",
                policy=policy, threads=1, first_release=True,
            )
            self.assertEqual(result["status"], "complete")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["run"]["status"], "complete")
            self.assertEqual(
                manifest["validation"]["append_stability"]["status"],
                "first_release",
            )
            self.assertEqual(
                manifest["validation"]["row_counts"]["incident_stage_summary"],
                len(pd.read_parquet(output / "incident_stage_summary.parquet")),
            )
            self.assertEqual(
                manifest["semantics"]["identity_hierarchy"],
                ["component_id", "exposure_id", "incident_id"],
            )
            self.assertFalse(manifest["semantics"]["crop_death_inferred"])
            self.assertEqual(
                set(manifest["source"]["inputs"]),
                {"manifest.json", *REQUIRED_SOURCE_ARTIFACTS},
            )
            for name, fingerprint in manifest["source"]["inputs"].items():
                path = generation / name
                self.assertEqual(fingerprint["size_bytes"], path.stat().st_size)
                self.assertEqual(
                    fingerprint["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
            self.assertEqual(
                set(manifest["implementation"]["inputs"]),
                {
                    f"story_monitor/{name}"
                    for name in workflow_v3.V3_IMPLEMENTATION_INPUTS
                },
            )
            implementation_root = Path(workflow_v3.__file__).resolve().parent
            for logical_name, fingerprint in manifest["implementation"]["inputs"].items():
                path = implementation_root / logical_name.removeprefix("story_monitor/")
                self.assertEqual(fingerprint["size_bytes"], path.stat().st_size)
                self.assertEqual(
                    fingerprint["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
            self.assertEqual(len(manifest["implementation"]["sha256"]), 64)
            self.assertEqual(
                manifest["implementation_sha256"],
                manifest["implementation"]["sha256"],
            )
            self.assertEqual(manifest["policy"]["input"]["sha256"], policy.source_sha256)
            self.assertEqual(
                manifest["policy"]["input"]["size_bytes"], policy.source_path.stat().st_size
            )
            self.assertEqual(len(manifest["policy"]["provenance_sha256"]), 64)
            self.assertEqual(len(manifest["policy"]["effective_sha256"]), 64)
            for name in (
                "field_week_context.parquet", "stage_baseline.parquet",
                "weekly_exposure_cells.parquet", "weekly_components.parquet",
                "component_membership.parquet", "exposure_weekly_state.parquet",
                "incident_weekly_state.parquet", "incident_stage_summary.parquet",
                "incident_membership.parquet", "incident_windows.parquet",
                "completed_incident_features.parquet",
            ):
                self.assertTrue((output / name).is_file(), name)
            cells = pd.read_parquet(output / "weekly_exposure_cells.parquet")
            self.assertIn("evaluable_field_count", cells)
            self.assertTrue(
                cells.empty
                or (pd.to_datetime(cells["timeline_bucket"]) > pd.Timestamp("2024-01-14")).all()
            )
            incidents = pd.read_parquet(output / "incident_weekly_state.parquet")
            self.assertGreater(len(incidents), 0)
            self.assertIn("knowledge_time", incidents)
            self.assertTrue(
                (
                    pd.to_datetime(incidents["knowledge_time"])
                    >= pd.to_datetime(incidents["timeline_bucket"])
                ).all()
            )
            repeated = build_incident_generation_v3(
                generation, root / "incident-v3-repeat", baseline_through="2024-01-14",
                policy=policy, threads=1, previous_incident_dir=output,
            )
            self.assertEqual(result["generation_id"], repeated["generation_id"])
            self.assertEqual(repeated["append_stability"]["status"], "passed")
            changed_policy = replace(policy, minimum_active_fields=2)
            changed = build_incident_generation_v3(
                generation, root / "incident-v3-policy-change",
                baseline_through="2024-01-14", policy=changed_policy, threads=1,
                first_release=True,
            )
            self.assertNotEqual(result["generation_id"], changed["generation_id"])
            drift_output = root / "incident-v3-drift"
            with patch.object(
                workflow_v3,
                "validate_append_stability",
                side_effect=ValueError("rewrote historical identity"),
            ):
                with self.assertRaisesRegex(ValueError, "rewrote"):
                    build_incident_generation_v3(
                        generation,
                        drift_output,
                        baseline_through="2024-01-14",
                        policy=policy,
                        threads=1,
                        previous_incident_dir=output,
                    )
            self.assertFalse(drift_output.exists())
            with self.assertRaises(FileExistsError):
                build_incident_generation_v3(
                    generation, output, baseline_through="2024-01-14",
                    policy=policy, threads=1, first_release=True,
                )

    def test_generation_identity_changes_for_each_provenance_dimension(self) -> None:
        inputs = {
            "baseline_through": "2025-12-31",
            "source_provenance_sha256": "a" * 64,
            "implementation_sha256": "b" * 64,
            "policy_provenance_sha256": "c" * 64,
            "effective_policy_sha256": "d" * 64,
        }
        identity = workflow_v3._generation_identity(**inputs)
        self.assertEqual(identity, workflow_v3._generation_identity(**inputs))
        for name in inputs:
            changed = dict(inputs)
            changed[name] = "2026-01-01" if name == "baseline_through" else "e" * 64
            self.assertNotEqual(identity, workflow_v3._generation_identity(**changed), name)
        self.assertNotEqual(
            workflow_v3._provenance_sha256(
                {"input": {"size_bytes": 1, "sha256": "f" * 64}}
            ),
            workflow_v3._provenance_sha256(
                {"input": {"size_bytes": 2, "sha256": "f" * 64}}
            ),
        )

    def test_release_mode_must_be_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = _write_workflow_generation(root)
            with self.assertRaisesRegex(ValueError, "exactly one release mode"):
                build_incident_generation_v3(
                    generation,
                    root / "incident-v3",
                    baseline_through="2024-01-14",
                    policy=replace(
                        load_incident_policy_v3(),
                        minimum_source_field_centroid_coverage=0.5,
                        minimum_source_crop_instance_week_centroid_coverage=0.5,
                        minimum_known_stage_coverage=0.5,
                    ),
                    threads=1,
                )
        self.assertNotEqual(
            workflow_v3._provenance_sha256(
                {"input": {"size_bytes": 1, "sha256": "f" * 64}}
            ),
            workflow_v3._provenance_sha256(
                {"input": {"size_bytes": 1, "sha256": "0" * 64}}
            ),
        )

    def test_source_mutation_aborts_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = _write_workflow_generation(root)
            output = root / "incident-v3"
            policy = replace(
                load_incident_policy_v3(),
                minimum_evaluable_fields=1,
                minimum_active_fields=1,
                severe_override_min_fields=1,
                minimum_source_field_centroid_coverage=0.5,
                minimum_source_crop_instance_week_centroid_coverage=0.5,
                minimum_known_stage_coverage=0.5,
            )
            validate = workflow_v3.validate_final_artifact_directory

            def validate_then_mutate(stage: Path) -> dict[str, object]:
                result = validate(stage)
                with (generation / "daily_causal_signals.parquet").open("r+b") as handle:
                    first = handle.read(1)
                    handle.seek(0)
                    handle.write(b"X" if first != b"X" else b"Y")
                return result

            with patch.object(
                workflow_v3,
                "validate_final_artifact_directory",
                side_effect=validate_then_mutate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "source generation inputs changed"
                ):
                    build_incident_generation_v3(
                        generation,
                        output,
                        baseline_through="2024-01-14",
                        policy=policy,
                        threads=1,
                        first_release=True,
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
