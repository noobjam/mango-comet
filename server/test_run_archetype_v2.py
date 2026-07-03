from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from story_monitor.archetype_runner_validation import (
    discover_generation,
    validate_evaluation,
)
from story_monitor.runner_process import RunnerError, run_stage


def _write(path: Path, value: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _generation(root: Path, name: str, *, as_of: str, status: str = "complete") -> Path:
    generation = root / "generations" / name
    manifest = {
        "run": {
            "status": status,
            "immutable": True,
            "as_of_date": as_of,
            "generation_id": name,
        },
        "input": {"max_fields": None},
        "policy": {"version": "test-v1", "sha256": "a" * 64},
    }
    _write(generation / "manifest.json", json.dumps(manifest).encode())
    for name in (
        "event_windows.parquet",
        "story_day_membership.parquet",
        "daily_causal_signals.parquet",
    ):
        _write(generation / name)
    return generation


def _evaluation(
    root: Path, *, hard: bool, quality: bool, model_manifest: Path | None = None
) -> Path:
    evaluation = root / "evaluation"
    model = json.loads(model_manifest.read_text()) if model_manifest else {}
    report = {
        "model_version": model.get("model_version", "model-test"),
        "gates": {"hard": {"passed": hard}, "quality": {"passed": quality}},
    }
    _write(evaluation / "evaluation.json", json.dumps(report).encode())
    for name in (
        "event_archetype_assignments.parquet",
        "evaluation_by_hazard.parquet",
        "holdout_assignments.parquet",
        "prototype_overlap.parquet",
        "subsample_stability.parquet",
        "subsample_stability.json",
        "training_frozen_assignments.parquet",
    ):
        _write(evaluation / name)
    artifacts = {}
    for name in sorted(path.name for path in evaluation.iterdir()):
        artifacts[name] = hashlib.sha256((evaluation / name).read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_version": model.get("model_version", "model-test"),
        "model_manifest_sha256": hashlib.sha256(model_manifest.read_bytes()).hexdigest() if model_manifest else None,
        "implementation_sha256": model.get("implementation_sha256"),
        "software_versions": model.get("software_versions"),
        "artifacts": artifacts,
    }
    _write(evaluation / "evaluation_manifest.json", json.dumps(manifest).encode())
    return evaluation


def _model(root: Path, generation: Path) -> Path:
    model = root / "model"
    for name in (
        "feature_schema.json",
        "event_anchors.parquet",
        "prototypes.parquet",
        "archetype_catalog.parquet",
        "training_assignments.parquet",
    ):
        _write(model / name)
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in model.iterdir()
    }
    generation_payload = json.loads((generation / "manifest.json").read_text())
    manifest = {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_version": "model-test",
        "generation_id": generation_payload["run"]["generation_id"],
        "generation_manifest_sha256": hashlib.sha256((generation / "manifest.json").read_bytes()).hexdigest(),
        "training_cutoff": "2025-12-31",
        "implementation_sha256": "b" * 64,
        "software_versions": {"python": "test"},
        "artifacts": artifacts,
    }
    _write(model / "archetype_manifest.json", json.dumps(manifest).encode())
    return model


def _state(root: Path) -> dict[str, object]:
    job = root / "job"
    job.mkdir()
    return {
        "status": "running",
        "paths": {
            "job_dir": str(job),
            "model_dir": str(root / "model"),
            "evaluation_dir": str(root / "evaluation"),
        },
        "config": {},
        "stages": {},
    }


class _FakeProcess:
    pid = 4321

    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.wait_count = 0
        kwargs["stdout"].write('{"status":"complete"}\n')  # type: ignore[union-attr]
        kwargs["stderr"].write("child warning\n")  # type: ignore[union-attr]

    def wait(self, timeout: int | None = None) -> int:
        self.wait_count += 1
        if self.wait_count == 1:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return 0

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


class ArchetypeV2RunnerTests(unittest.TestCase):
    def test_generation_discovery_is_full_immutable_and_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _generation(root, "full", as_of="2026-05-17")
            _generation(root, "old", as_of="2026-05-10")
            _generation(root, "incomplete", as_of="2026-05-17", status="building")
            self.assertEqual(discover_generation(root, as_of="2026-05-17"), expected.resolve())

            _generation(root, "also-full", as_of="2026-05-17")
            with self.assertRaisesRegex(RunnerError, "Multiple full generations"):
                discover_generation(root, as_of="2026-05-17")

    def test_stage_heartbeats_without_shell_or_fake_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stream = io.StringIO()
            logger = logging.getLogger(f"runner-test-{id(self)}")
            logger.handlers = [logging.StreamHandler(stream)]
            logger.setLevel(logging.INFO)
            with mock.patch(
                "story_monitor.runner_process.subprocess.Popen", side_effect=_FakeProcess
            ) as popen:
                code = run_stage(
                    ["python", "worker.py"],
                    stdout_path=root / "stdout.json",
                    stderr_path=root / "stderr.log",
                    logger=logger,
                    label="[1/2 BUILD]",
                    heartbeat_seconds=1,
                    cwd=root,
                    env={"SAFE": "1"},
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads((root / "stdout.json").read_text())["status"], "complete")
            self.assertIn("child warning", (root / "stderr.log").read_text())
            self.assertIn("running — elapsed", stream.getvalue())
            self.assertNotIn("%", stream.getvalue())
            self.assertFalse(popen.call_args.kwargs["shell"])
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.args[0], ["python", "worker.py"])

    def test_evaluation_validation_preserves_gate_failure_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _evaluation(Path(directory), hard=True, quality=False)
            _, hard, quality = validate_evaluation(evaluation)
            self.assertTrue(hard)
            self.assertFalse(quality)

            manifest_path = evaluation / "evaluation_manifest.json"
            original_manifest = manifest_path.read_text()
            manifest = json.loads(original_manifest)
            manifest["artifacts"].pop("subsample_stability.parquet")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RunnerError, "registry differs"):
                validate_evaluation(evaluation)
            manifest_path.write_text(original_manifest)

            (evaluation / "subsample_stability.parquet").write_bytes(b"tampered")
            with self.assertRaisesRegex(RunnerError, "hash mismatch"):
                validate_evaluation(evaluation)

if __name__ == "__main__":
    unittest.main()
