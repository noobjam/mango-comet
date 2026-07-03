from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from story_monitor import archetype_runner, runner_process
from story_monitor.archetype_runner_validation import (
    job_paths,
    validate_evaluation,
    validate_model,
)
from story_monitor.runner_process import RunnerError, run_stage
from test_run_archetype_v2 import _evaluation, _generation, _model, _state


class ArchetypeRunnerSafetyTests(unittest.TestCase):
    def test_resume_artifacts_are_bound_to_generation_and_model_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = _generation(root, "full", as_of="2026-05-17")
            model = _model(root, generation)
            model_manifest = model / "archetype_manifest.json"
            validate_model(
                model,
                generation_manifest=generation / "manifest.json",
                training_cutoff="2025-12-31",
            )
            with self.assertRaisesRegex(RunnerError, "training cutoff"):
                validate_model(
                    model,
                    generation_manifest=generation / "manifest.json",
                    training_cutoff="2025-12-30",
                )
            evaluation = _evaluation(
                root, hard=True, quality=True, model_manifest=model_manifest
            )
            validate_evaluation(evaluation, model_manifest=model_manifest)
            payload = json.loads((evaluation / "evaluation_manifest.json").read_text())
            payload["model_version"] = "wrong-model"
            (evaluation / "evaluation_manifest.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(RunnerError, "does not match"):
                validate_evaluation(evaluation, model_manifest=model_manifest)

    def test_build_failure_is_terminal_and_does_not_start_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = _state(Path(directory))
            logger = logging.getLogger(f"runner-failure-{id(self)}")
            logger.handlers = [logging.NullHandler()]
            with mock.patch.object(archetype_runner, "_preflight"), mock.patch.object(
                archetype_runner, "_build", side_effect=RunnerError("build failed", 10)
            ), mock.patch.object(archetype_runner, "_evaluate") as evaluate:
                code = archetype_runner._execute(state, logger)
            self.assertEqual(code, 10)
            evaluate.assert_not_called()
            persisted = json.loads((Path(directory) / "job" / "state.json").read_text())
            self.assertEqual(persisted["status"], "failed")

    def test_unexpected_failure_is_persisted_instead_of_staying_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = _state(Path(directory))
            logger = logging.getLogger(f"runner-unexpected-{id(self)}")
            logger.handlers = [logging.NullHandler()]
            with mock.patch.object(archetype_runner, "_preflight"), mock.patch.object(
                archetype_runner, "_build", side_effect=OSError("launch failed")
            ):
                code = archetype_runner._execute(state, logger)
            self.assertEqual(code, 2)
            persisted = json.loads((Path(directory) / "job" / "state.json").read_text())
            self.assertEqual(persisted["status"], "failed")
            self.assertIn("launch failed", persisted["error"])

    def test_gate_combinations_have_distinct_terminal_exit_codes(self) -> None:
        cases = ((False, True, 20), (True, False, 21), (False, False, 22), (True, True, 0))
        for hard, quality, expected in cases:
            with self.subTest(hard=hard, quality=quality), tempfile.TemporaryDirectory() as directory:
                state = _state(Path(directory))
                code = archetype_runner._gate_result(state, hard, quality)
                self.assertEqual(code, expected)

    def test_resume_refuses_partial_model_instead_of_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = _state(root)
            generation = _generation(root, "full", as_of="2026-05-17")
            state["config"] = {
                "rapids_python": "python",
                "generation_dir": str(generation),
                "training_cutoff": "2025-12-31",
            }
            (root / "model").mkdir()
            logger = logging.getLogger(f"runner-partial-{id(self)}")
            logger.handlers = [logging.NullHandler()]
            with self.assertRaisesRegex(RunnerError, "will not be overwritten") as raised:
                archetype_runner._build(state, logger)
            self.assertEqual(raised.exception.exit_code, 11)

    def test_job_tag_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RunnerError, "Job tag"):
                job_paths(Path(directory), "../../escape", "2025-12-31")

    def test_new_job_refuses_output_collision_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = _generation(root, "full", as_of="2026-05-17")
            (root / "jobs" / "archetype_v2_duplicate").mkdir(parents=True)
            config = {
                "root": str(root),
                "repo_dir": str(Path(__file__).resolve().parent.parent),
                "generation_dir": str(generation),
                "as_of": None,
                "rapids_python": sys.executable,
                "training_cutoff": "2025-12-31",
                "gpu": 0,
                "threads": 1,
                "memory_limit": "1GB",
                "temp_dir": str(root / "tmp"),
                "heartbeat_seconds": 1,
                "skip_tests": True,
            }
            with self.assertRaisesRegex(RunnerError, "already has output"):
                archetype_runner.run_new_job(config, tag="duplicate")

    def test_status_marks_dead_running_job_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = _state(Path(directory))
            state["pid"] = 999_999_999
            (Path(directory) / "job" / "state.json").write_text(json.dumps(state))
            self.assertEqual(
                archetype_runner.read_job_state(Path(directory) / "job")["liveness"],
                "stale",
            )

    def test_live_orphan_child_blocks_a_new_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orphan = root / "jobs" / "archetype_v2_orphan"
            orphan.mkdir(parents=True)
            state = {"status": "running", "active_process": {"pid": os.getpid()}}
            (orphan / "state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(RunnerError, "still alive") as raised:
                archetype_runner._assert_no_orphan_child(root)
            self.assertEqual(raised.exception.exit_code, 75)

    def test_orphan_from_another_job_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orphan = root / "jobs" / "archetype_v2_orphan"
            target = root / "jobs" / "archetype_v2_target"
            orphan.mkdir(parents=True)
            target.mkdir(parents=True)
            orphan_state = {
                "status": "running",
                "active_process": {"pid": os.getpid()},
                "paths": {"job_dir": str(orphan)},
            }
            target_state = {
                "status": "failed",
                "job_tag": "target",
                "config": {"root": str(root)},
                "paths": {"job_dir": str(target), "runner_log": str(target / "runner.log")},
            }
            (orphan / "state.json").write_text(json.dumps(orphan_state))
            (target / "state.json").write_text(json.dumps(target_state))
            with self.assertRaisesRegex(RunnerError, "still alive") as raised:
                archetype_runner.resume_job(target)
            self.assertEqual(raised.exception.exit_code, 75)

    def test_interrupt_terminates_the_child_process_group(self) -> None:
        process = mock.Mock(pid=4321)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 10), -9]
        with mock.patch("story_monitor.runner_process.os.killpg") as kill_group:
            runner_process._terminate(process)
        self.assertEqual(
            kill_group.call_args_list,
            [mock.call(4321, signal.SIGTERM), mock.call(4321, signal.SIGKILL)],
        )

    def test_process_launch_failure_is_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logger = logging.getLogger(f"runner-launch-{id(self)}")
            logger.handlers = [logging.NullHandler()]
            with mock.patch(
                "story_monitor.runner_process.subprocess.Popen",
                side_effect=FileNotFoundError("gone"),
            ), self.assertRaisesRegex(RunnerError, "could not start"):
                run_stage(
                    ["missing"], stdout_path=root / "out", stderr_path=root / "err",
                    logger=logger, label="[1/2 BUILD]", heartbeat_seconds=1,
                    cwd=root, env={},
                )


if __name__ == "__main__":
    unittest.main()
