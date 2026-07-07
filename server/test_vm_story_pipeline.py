from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

from story_monitor.incident_workflow_v3 import V3_IMPLEMENTATION_INPUTS


SCRIPT = Path(__file__).with_name("vm_story_pipeline.sh")


class VmStoryPipelineTests(unittest.TestCase):
    def test_script_has_valid_bash_syntax_and_help_needs_no_env(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            [str(SCRIPT), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("launch starts one detached, logged pipeline", help_result.stdout)
        self.assertIn("mandatory review gate", help_result.stdout)
        self.assertIn("continue [ENV_FILE]", help_result.stdout)
        self.assertIn("without\nbuilding V3", help_result.stdout)
        self.assertIn("--capture-stage9-replay", SCRIPT.read_text(encoding="utf-8"))

    def test_continue_path_cannot_reach_v3_source_or_evidence_builds(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        continuation = script.split("continue_pipeline() {", 1)[1].split(
            "show_status() {", 1
        )[0]
        self.assertIn("prepare_completed_v4_continuation", continuation)
        self.assertIn("run_downstream_from_completed_v4", continuation)
        self.assertNotIn("ensure_v3_incident_dir", continuation)
        self.assertNotIn("run_incident_v4.py", continuation)
        self.assertNotIn("prepare_incident_source_v4.py", continuation)
        self.assertNotIn("build-evidence-v4", continuation)
        self.assertIn(
            "REQUIRED_COMMIT=1ac6e4b534fbf84bd11207663df9ea26168547a4",
            script,
        )

    def test_missing_env_fails_before_any_pipeline_action(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "status", "/definitely/missing/.env.vm"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("environment file does not exist", result.stderr)

    def test_status_reports_dead_process_without_requiring_source_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            repo = Path(temporary) / "repo"
            logs = root / "logs"
            logs.mkdir(parents=True)
            repo.mkdir()
            tag = "dead-job"
            base = logs / f"vm_story_pipeline_{tag}"
            (logs / "latest_vm_story_pipeline.txt").write_text(f"{tag}\n")
            Path(f"{base}.pid").write_text("99999999\n")
            Path(f"{base}.phase").write_text("PREFLIGHT\n")
            Path(f"{base}.log").write_text("started\n")
            env_file = Path(temporary) / ".env.vm"
            env_file.write_text(
                "\n".join(
                    (
                        f"ROOT={root}",
                        f"REPO={repo}",
                        f"PYTHON={sys.executable}",
                        "MAP_HOST=127.0.0.1",
                        "MAP_PORT=9",
                    )
                )
                + "\n"
            )
            result = subprocess.run(
                [str(SCRIPT), "status", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STATUS=DEAD_WITHOUT_STATUS", result.stdout)
            self.assertIn("PHASE=PREFLIGHT", result.stdout)

    def test_v3_compatibility_uses_generation_schema_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            generation = Path(temporary) / "generation"
            incident = Path(temporary) / "incident"
            job = root / "jobs" / "incident_v3_fixture"
            logs = root / "logs"
            for path in (generation, incident, job, logs):
                path.mkdir(parents=True, exist_ok=True)
            generation_manifest = generation / "manifest.json"
            generation_manifest.write_text(
                json.dumps({"run": {"generation_id": "generation-fixture"}})
            )
            policy_path = SCRIPT.parent / "story_monitor" / "policies" / "incident_policy_v3.json"
            implementation_root = SCRIPT.parent / "story_monitor"
            implementation_inputs = {}
            for name in V3_IMPLEMENTATION_INPUTS:
                implementation_path = implementation_root / name
                implementation_inputs[f"story_monitor/{name}"] = {
                    "sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
                    "size_bytes": implementation_path.stat().st_size,
                }
            (incident / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "crop-impact-incident-generation-v3/1",
                        "run": {
                            "status": "complete",
                            "source_generation_id": "generation-fixture",
                            "baseline_through": "2025-12-31",
                        },
                        "source": {
                            "generation_manifest_sha256": hashlib.sha256(
                                generation_manifest.read_bytes()
                            ).hexdigest(),
                        },
                        "policy": {
                            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                        },
                        "implementation": {
                            "inputs": implementation_inputs,
                        },
                    }
                )
            )
            (job / "status").write_text("0\n")
            (job / "state.json").write_text(
                json.dumps({"paths": {"incident_dir": str(incident)}})
            )
            (logs / "latest_incident_v3_job.txt").write_text(f"{job}\n")
            env_file = Path(temporary) / ".env.vm"
            env_file.write_text(
                "\n".join(
                    (
                        f"ROOT={root}",
                        f"PYTHON={sys.executable}",
                        f"GEN={generation}",
                        "V3_BASELINE_THROUGH=2025-12-31",
                    )
                )
                + "\n"
            )
            compatible = subprocess.run(
                [str(SCRIPT), "check-v3", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compatible.returncode, 0, compatible.stderr)
            self.assertIn(f"INCIDENT_DIR={incident}", compatible.stdout)

            incident_manifest = json.loads((incident / "manifest.json").read_text())
            incident_manifest["policy"]["sha256"] = "0" * 64
            (incident / "manifest.json").write_text(json.dumps(incident_manifest))
            stale_policy = subprocess.run(
                [str(SCRIPT), "check-v3", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(stale_policy.returncode, 2)
            self.assertIn("no successful V3 release matches GEN", stale_policy.stderr)

            incident_manifest["policy"]["sha256"] = hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest()
            (incident / "manifest.json").write_text(json.dumps(incident_manifest))

            manifest = json.loads((generation / "manifest.json").read_text())
            manifest["run"]["generation_id"] = "different-generation"
            (generation / "manifest.json").write_text(json.dumps(manifest))
            mismatch = subprocess.run(
                [str(SCRIPT), "check-v3", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("no successful V3 release matches GEN", mismatch.stderr)

    def test_explicit_node_path_is_validated(self) -> None:
        node = shutil.which("node")
        self.assertIsNotNone(node)
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env.vm"
            env_file.write_text(f"NODE={node}\n")
            result = subprocess.run(
                [str(SCRIPT), "check-node", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"NODE={node}", result.stdout)
            self.assertRegex(result.stdout, r"NODE_VERSION=v\d+")


if __name__ == "__main__":
    unittest.main()
