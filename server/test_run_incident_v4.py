from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from run_incident_v4 import (
    _artifact_complete,
    _evidence_command,
    _new_state,
    _source_command,
    _validated_viewer,
    build_parser,
)
from story_monitor.incident_release_v4 import CORRECTION_POLICY
from story_monitor.runner_process import RunnerError


class IncidentV4RunnerTests(unittest.TestCase):
    def test_parser_requires_explicit_release_mode(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run", "--generation-dir", "/tmp/generation",
                    "--incident-dir", "/tmp/incident",
                ]
            )

    def test_new_state_uses_separate_immutable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            incident = root / "incident"
            enriched = root / "enriched.parquet"
            generation.mkdir()
            incident.mkdir()
            _write_enriched_parquet(enriched)
            enriched.with_suffix(".parquet.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "incident-enriched-source-v4/1",
                        "status": "complete",
                        "immutable": True,
                        "released_at": "2025-02-01T12:00:00Z",
                        "correction_policy": CORRECTION_POLICY,
                        "availability": {
                            "mode": "reconstructed",
                            "diagnostic_reconstruction": True,
                        },
                        "output": {
                            "name": enriched.name,
                            "size_bytes": enriched.stat().st_size,
                            "sha256": hashlib.sha256(enriched.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args(
                [
                    "run", "--root", str(root),
                    "--generation-dir", str(generation),
                    "--incident-dir", str(incident),
                    "--enriched-source-parquet", str(enriched),
                    "--python", "/bin/sh", "--node", "/bin/sh",
                    "--first-release", "--job-tag", "test-v4",
                ]
            )

            state = _new_state(args)

            self.assertNotEqual(state["paths"]["evidence_dir"], str(incident))
            self.assertNotEqual(state["paths"]["viewer_dir"], str(incident))
            self.assertTrue(state["config"]["first_release"])
            self.assertEqual(state["config"]["availability_mode"], "reconstructed")
            self.assertEqual(
                state["config"]["released_at"], "2025-02-01T12:00:00.000000Z"
            )
            self.assertEqual(
                _evidence_command(state)[
                    _evidence_command(state).index("--released-at") + 1
                ],
                state["config"]["released_at"],
            )

    def test_new_state_rejects_reconstructed_source_requested_as_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            incident = root / "incident"
            enriched = root / "enriched.parquet"
            generation.mkdir()
            incident.mkdir()
            _write_enriched_parquet(enriched)
            enriched.with_suffix(".parquet.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "incident-enriched-source-v4/1",
                        "status": "complete",
                        "immutable": True,
                        "released_at": "2025-02-01T12:00:00Z",
                        "correction_policy": CORRECTION_POLICY,
                        "availability": {
                            "mode": "reconstructed",
                            "diagnostic_reconstruction": True,
                        },
                        "output": {
                            "name": enriched.name,
                            "size_bytes": enriched.stat().st_size,
                            "sha256": hashlib.sha256(enriched.read_bytes()).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "run", "--root", str(root),
                    "--generation-dir", str(generation),
                    "--incident-dir", str(incident),
                    "--enriched-source-parquet", str(enriched),
                    "--availability-mode", "strict",
                    "--python", "/bin/sh", "--node", "/bin/sh",
                    "--first-release",
                ]
            )

            with self.assertRaisesRegex(RunnerError, "complete immutable sidecar"):
                _new_state(args)

    def test_echo_source_requires_rich_full_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            incident = root / "incident"
            echo = root / "echo.parquet"
            generation.mkdir()
            incident.mkdir()
            echo.touch()
            args = build_parser().parse_args(
                [
                    "run", "--root", str(root),
                    "--generation-dir", str(generation),
                    "--incident-dir", str(incident),
                    "--echo-deliverable", str(echo),
                    "--python", "/bin/sh", "--node", "/bin/sh",
                    "--first-release",
                ]
            )
            with self.assertRaisesRegex(RunnerError, "at least one --full-parquet"):
                _new_state(args)

    def test_internal_source_and_evidence_share_one_release_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "generation"
            incident = root / "incident"
            echo = root / "echo.parquet"
            full = root / "full.parquet"
            generation.mkdir()
            incident.mkdir()
            echo.touch()
            full.touch()
            args = build_parser().parse_args(
                [
                    "run", "--root", str(root),
                    "--generation-dir", str(generation),
                    "--incident-dir", str(incident),
                    "--echo-deliverable", str(echo),
                    "--full-parquet", str(full),
                    "--released-at", "2025-02-01T13:14:15+00:00",
                    "--python", "/bin/sh", "--node", "/bin/sh",
                    "--first-release", "--job-tag", "watermark-v4",
                ]
            )

            state = _new_state(args)
            source_command = _source_command(state)
            evidence_command = _evidence_command(state)

            self.assertEqual(state["config"]["released_at"], "2025-02-01T13:14:15.000000Z")
            self.assertEqual(
                source_command[source_command.index("--released-at") + 1],
                state["config"]["released_at"],
            )
            self.assertEqual(
                evidence_command[evidence_command.index("--released-at") + 1],
                state["config"]["released_at"],
            )

    def test_artifact_completion_checks_v4_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "mode": "crop_incident_v4_dual_clock",
                        "run": {"status": "complete"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_artifact_complete(root, "crop_incident_v4_dual_clock"))
            self.assertFalse(_artifact_complete(root, "crop_incident_v3"))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                _validated_viewer(root)


def _write_enriched_parquet(path: Path) -> None:
    pd.DataFrame(
        [{
            "weather_available_at": "2025-01-01T06:00:00Z",
            "spectral_available_at": "2025-01-01T10:00:00Z",
            "stage_available_at": "2025-01-01T07:00:00Z",
        }]
    ).to_parquet(path, index=False)


if __name__ == "__main__":
    unittest.main()
