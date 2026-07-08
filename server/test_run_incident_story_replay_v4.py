from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_incident_story_replay_v4 import (
    CHECKPOINT_SCHEMA_VERSION,
    FOCUSED_TESTS,
    REPLAY_MODE,
    REPLAY_SCHEMA_VERSION,
    _build_command,
    _execute,
    _export_command,
    _new_state,
    _paths,
    _require_native_viewer_manifest,
    _replay_checkpoint_progress,
    _run_build_stage,
    _validated_replay,
    _validated_source_adapter,
    _verify_immutable_inputs,
    build_parser,
)
from story_monitor.runner_process import RunnerError


class IncidentStoryReplayV4RunnerTests(unittest.TestCase):
    def test_status_reports_context_partition_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            (checkpoints / "01_context").mkdir()
            (checkpoints / "01_context" / "manifest.json").write_text("{}")
            output = (
                checkpoints
                / ".01_context-work"
                / "01_context"
                / ".replay-partitions"
                / "output"
                / "event_state_snapshots"
            )
            output.mkdir(parents=True)
            for index in range(3):
                (output / f"part-{index:04d}.parquet").write_bytes(b"part")

            self.assertEqual(
                _replay_checkpoint_progress(
                    checkpoints, expected_partitions=64
                ),
                {
                    "completed_stages": ["01_context"],
                    "context_replay": {
                        "completed_partitions": 3,
                        "expected_partitions": 64,
                        "partial_work_reusable_on_resume": False,
                    },
                },
            )

    def test_parser_paths_and_subprocess_commands_are_deterministic(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--evidence-dir",
                "/tmp/evidence",
                "--geometry-parquet",
                "/tmp/geometry.parquet",
                "--audit-incident-dir",
                "/tmp/audit",
                "--baseline-through",
                "2025-12-31",
                "--replay-partitions",
                "128",
                "--job-tag",
                "replay-tag",
            ]
        )
        self.assertEqual(args.baseline_through, "2025-12-31")
        self.assertEqual(args.replay_partitions, 128)
        self.assertIn("server.test_incident_cells_v3", FOCUSED_TESTS)
        self.assertIn("server.test_incident_story_states_v3", FOCUSED_TESTS)
        paths = _paths(Path("/tmp/root"), "replay-tag")
        self.assertEqual(
            paths["job_dir"],
            "/tmp/root/jobs/incident_story_replay_v4_replay-tag",
        )
        self.assertEqual(
            paths["incident_dir"],
            "/tmp/root/releases/incidents_v4_replay_replay-tag",
        )
        self.assertEqual(
            paths["viewer_dir"],
            "/tmp/root/releases/incident_viewer_v4_replay_replay-tag",
        )
        self.assertEqual(
            paths["checkpoint_dir"],
            "/tmp/root/jobs/incident_story_replay_v4_replay-tag/checkpoints",
        )

        state = {
            "config": {
                "python": "/venv/bin/python",
                "evidence_dir": "/input/evidence",
                "geometry_parquet": "/input/geometry.parquet",
                "audit_incident_dir": "/input/audit",
                "baseline_through": "2025-12-31",
                "threads": 8,
                "replay_partitions": 128,
                "memory_limit": "32GB",
                "temp_dir": "/tmp/duckdb",
            },
            "paths": paths,
        }
        build = _build_command(state)
        self.assertEqual(build[2], "_build")
        self.assertEqual(build[build.index("--replay-partitions") + 1], "128")
        self.assertEqual(
            build[build.index("--checkpoint-dir") + 1], paths["checkpoint_dir"]
        )
        export = _export_command(state)
        self.assertIn("--native-replay", export)
        self.assertEqual(
            export[export.index("--source-generation-dir") + 1],
            str(
                Path(paths["checkpoint_dir"])
                / "01_context"
                / "source_generation"
            ),
        )

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "run",
                    "--evidence-dir",
                    "/tmp/evidence",
                    "--geometry-parquet",
                    "/tmp/geometry.parquet",
                    "--audit-incident-dir",
                    "/tmp/audit",
                    "--baseline-through",
                    "2025-12-31",
                    "--replay-partitions",
                    "1025",
                ]
            )

    def test_new_state_freezes_inputs_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, geometry, audit = _write_inputs(root)
            args = build_parser().parse_args(
                [
                    "run",
                    "--root",
                    str(root),
                    "--evidence-dir",
                    str(evidence),
                    "--geometry-parquet",
                    str(geometry),
                    "--audit-incident-dir",
                    str(audit),
                    "--baseline-through",
                    "2025-12-31",
                    "--python",
                    "/bin/sh",
                    "--skip-tests",
                    "--job-tag",
                    "immutable",
                ]
            )

            state = _new_state(args)

            self.assertEqual(state["config"]["replay_partitions"], 64)
            self.assertEqual(
                Path(state["paths"]["checkpoint_dir"]).parent,
                Path(state["paths"]["job_dir"]),
            )
            self.assertNotEqual(state["paths"]["incident_dir"], str(evidence))
            _verify_immutable_inputs(state)
            (evidence / "manifest.json").write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RunnerError, "Immutable input changed"):
                _verify_immutable_inputs(state)

    def test_new_state_rejects_incomplete_audit_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, geometry, audit = _write_inputs(root)
            (audit / "manifest.json").write_text("{}\n", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "run",
                    "--root",
                    str(root),
                    "--evidence-dir",
                    str(evidence),
                    "--geometry-parquet",
                    str(geometry),
                    "--audit-incident-dir",
                    str(audit),
                    "--baseline-through",
                    "2025-12-31",
                    "--python",
                    "/bin/sh",
                    "--skip-tests",
                    "--job-tag",
                    "incomplete-audit",
                ]
            )

            with self.assertRaisesRegex(RunnerError, "not complete and immutable"):
                _new_state(args)

    def test_execute_uses_mocked_build_export_and_smoke_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, geometry, audit = _write_inputs(root)
            args = build_parser().parse_args(
                [
                    "run",
                    "--root",
                    str(root),
                    "--evidence-dir",
                    str(evidence),
                    "--geometry-parquet",
                    str(geometry),
                    "--audit-incident-dir",
                    str(audit),
                    "--baseline-through",
                    "2025-12-31",
                    "--python",
                    "/bin/sh",
                    "--skip-tests",
                    "--job-tag",
                    "execute",
                ]
            )
            state = _new_state(args)
            adapter = (
                Path(state["paths"]["checkpoint_dir"])
                / "01_context"
                / "source_generation"
            )
            adapter.mkdir(parents=True)
            (adapter / "manifest.json").write_text("{}\n", encoding="utf-8")
            calls: list[tuple[str, list[str]]] = []

            def fake_stage(
                stage_state: dict,
                _logger: logging.Logger,
                name: str,
                _label: str,
                command: list[str],
                _stdout: str,
                _stderr: str,
            ) -> None:
                calls.append((name, command))
                stage_state.setdefault("stages", {})[name] = {"status": "complete"}

            logger = logging.getLogger("test-story-replay-runner")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            with (
                patch(
                    "run_incident_story_replay_v4._stage", side_effect=fake_stage
                ),
                patch(
                    "run_incident_story_replay_v4._validated_replay",
                    return_value={"status": "valid"},
                ),
                patch(
                    "run_incident_story_replay_v4._validated_viewer",
                    return_value={"status": "valid"},
                ),
                patch(
                    "run_incident_story_replay_v4._validated_source_adapter",
                    return_value=adapter,
                ),
            ):
                code = _execute(state, logger)

            self.assertEqual(code, 0)
            self.assertEqual([name for name, _ in calls], ["build", "export", "smoke"])
            export = dict(calls)["export"]
            self.assertIn("--native-replay", export)
            self.assertEqual(
                (Path(state["paths"]["job_dir"]) / "status").read_text(), "0\n"
            )

    def test_private_build_stage_forwards_bounded_replay_configuration(self) -> None:
        result = {"status": "complete", "generation_id": "generation-1"}
        stdout = io.StringIO()
        with (
            patch(
                "story_monitor.incident_story_replay_v4.build_incident_story_replay_v4",
                return_value=result,
            ) as build,
            redirect_stdout(stdout),
        ):
            code = _run_build_stage(
                [
                    "--evidence-dir",
                    "/input/evidence",
                    "--geometry-parquet",
                    "/input/geometry.parquet",
                    "--audit-incident-dir",
                    "/input/audit",
                    "--output-dir",
                    "/output/incidents",
                    "--checkpoint-dir",
                    "/job/checkpoints",
                    "--baseline-through",
                    "2025-12-31",
                    "--threads",
                    "8",
                    "--replay-partitions",
                    "32",
                    "--memory-limit",
                    "24GB",
                    "--temp-dir",
                    "/tmp/duckdb",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs["replay_partitions"], 32)
        self.assertEqual(build.call_args.kwargs["memory_limit"], "24GB")
        self.assertEqual(json.loads(stdout.getvalue()), result)

    def test_replay_validation_verifies_inventory_and_crosswalk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            crosswalk = release / "old_to_new_incident_crosswalk.parquet"
            crosswalk.write_bytes(b"crosswalk")
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "manifest.json").write_bytes(b"evidence-manifest")
            audit = root / "audit"
            audit.mkdir()
            (audit / "manifest.json").write_bytes(b"audit-manifest")
            (audit / "incident_membership.parquet").write_bytes(b"membership")
            geometry = root / "geometry.parquet"
            geometry.write_bytes(b"geometry")
            manifest = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "mode": REPLAY_MODE,
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "generation_id": "replay-1",
                    "baseline_through": "2025-12-31",
                },
                "source": {
                    "evidence_manifest_sha256": hashlib.sha256(
                        (evidence / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "geometry_sha256": hashlib.sha256(
                        geometry.read_bytes()
                    ).hexdigest(),
                    "audit_incident_manifest_sha256": hashlib.sha256(
                        (audit / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "audit_incident_membership_sha256": hashlib.sha256(
                        (audit / "incident_membership.parquet").read_bytes()
                    ).hexdigest(),
                },
                "validation": {
                    "passed": True,
                    "membership_counter_mismatch_count": 0,
                    "row_counts": {"incident_weekly_state": 4},
                    "crosswalk_rows": 2,
                },
                "artifacts": {
                    crosswalk.name: {
                        "size_bytes": crosswalk.stat().st_size,
                        "sha256": hashlib.sha256(crosswalk.read_bytes()).hexdigest(),
                    }
                },
            }
            (release / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            state = {
                "config": {
                    "evidence_dir": str(evidence),
                    "geometry_parquet": str(geometry),
                    "audit_incident_dir": str(audit),
                    "baseline_through": "2025-12-31",
                }
            }
            self.assertEqual(_validated_replay(release, state)["status"], "valid")
            (audit / "incident_membership.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                _validated_replay(release, state)
            (audit / "incident_membership.parquet").write_bytes(b"membership")
            crosswalk.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                _validated_replay(release)

    def test_native_viewer_manifest_keeps_compatible_mode_and_true_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            viewer = Path(directory)
            manifest = {
                "schema_version": "crop-incident-viewer-v4/2",
                "mode": "crop_incident_v4_dual_clock",
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "native_replay": True,
                },
                "semantics": {
                    "lifecycle_state_recomputed_from_v4": True,
                    "component_absence_replayed_from_v4": True,
                    "full_lifecycle_replay_supported": True,
                    "lifecycle_causal_ownership_claimed": True,
                    "source_state_preserved": False,
                },
            }
            (viewer / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                _require_native_viewer_manifest(viewer)["mode"],
                "crop_incident_v4_dual_clock",
            )
            manifest["semantics"]["lifecycle_state_recomputed_from_v4"] = False
            (viewer / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "lifecycle ownership"):
                _require_native_viewer_manifest(viewer)

    def test_source_adapter_is_bound_to_release_and_checkpoint_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "job" / "checkpoints" / "01_context"
            adapter = checkpoint / "source_generation"
            adapter.mkdir(parents=True)
            source_manifest = adapter / "manifest.json"
            source_manifest.write_text('{"run":{"status":"complete"}}\n')
            payload = adapter / "event_state_snapshots.parquet"
            payload.write_bytes(b"snapshots")
            artifacts = {
                item.relative_to(checkpoint).as_posix(): {
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                    "size_bytes": item.stat().st_size,
                }
                for item in (source_manifest, payload)
            }
            checkpoint_manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage": "context",
                "run": {"status": "complete", "immutable": True},
                "artifacts": artifacts,
            }
            checkpoint_manifest_path = checkpoint / "manifest.json"
            checkpoint_manifest_path.write_text(json.dumps(checkpoint_manifest))
            incident = root / "incident"
            incident.mkdir()
            replay_manifest = {
                "source": {
                    "generation_manifest_sha256": hashlib.sha256(
                        source_manifest.read_bytes()
                    ).hexdigest()
                },
                "checkpoints": {
                    "01_context": {
                        "sha256": hashlib.sha256(
                            checkpoint_manifest_path.read_bytes()
                        ).hexdigest(),
                        "size_bytes": checkpoint_manifest_path.stat().st_size,
                    }
                },
            }
            (incident / "manifest.json").write_text(json.dumps(replay_manifest))
            state = {
                "paths": {
                    "checkpoint_dir": str(checkpoint.parent),
                    "incident_dir": str(incident),
                }
            }

            self.assertEqual(_validated_source_adapter(state), adapter)
            payload.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                _validated_source_adapter(state)


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    evidence = root / "evidence"
    evidence.mkdir()
    complete_manifest = {"run": {"status": "complete", "immutable": True}}
    (evidence / "manifest.json").write_text(
        json.dumps(complete_manifest) + "\n", encoding="utf-8"
    )
    for name in (
        "crop_day_context_v4.parquet",
        "field_day_pressure_v4.parquet",
        "field_s2_acquisition_v4.parquet",
    ):
        (evidence / name).write_bytes(name.encode("utf-8"))
    geometry = root / "geometry.parquet"
    geometry.write_bytes(b"geometry")
    audit = root / "audit"
    audit.mkdir()
    (audit / "manifest.json").write_text(
        json.dumps(complete_manifest) + "\n", encoding="utf-8"
    )
    (audit / "incident_membership.parquet").write_bytes(b"membership")
    return evidence, geometry, audit


if __name__ == "__main__":
    unittest.main()
