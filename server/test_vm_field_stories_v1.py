from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("vm_field_stories_v1.sh")


class VmFieldStoriesV1Tests(unittest.TestCase):
    def test_script_has_valid_syntax_and_env_free_help(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("run [ENV_FILE]", help_result.stdout)
        self.assertIn("status [ENV_FILE]", help_result.stdout)
        self.assertIn("logs [ENV_FILE]", help_result.stdout)

    def test_run_updates_then_restarts_the_checked_out_wrapper(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        run_case = script.split("  run)", 1)[1].split("    ;;", 1)[0]
        self.assertIn("git pull --ff-only origin main", run_case)
        self.assertIn('exec "$REPO/server/vm_field_stories_v1.sh" __launch', run_case)

    def test_detached_launch_uses_only_the_field_story_runner(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        launch = script.split("launch_release() {", 1)[1].split("run_release() {", 1)[0]
        run = script.split("run_release() {", 1)[1].split("show_status() {", 1)[0]
        self.assertIn("nohup", launch)
        self.assertIn("run_field_stories_v1.py", run)
        for argument in (
            "--evidence-dir",
            "--output-dir",
            "--partitions",
            "--threads",
            "--memory-limit",
            "--temp-dir",
        ):
            self.assertIn(argument, run)
        for forbidden in (
            "run_incident_v4.py",
            "run_incident_story_replay_v4.py",
            "story_map_server.py",
            "run_incident_motifs_v4.py",
        ):
            self.assertNotIn(forbidden, launch + run)

    def test_missing_env_fails_before_action(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "status", "/definitely/missing/.env.vm"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("environment file does not exist", result.stderr)

    def test_status_reports_dead_process_without_source_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            logs = root / "logs"
            logs.mkdir(parents=True)
            tag = "dead-job"
            base = logs / f"field_stories_v1_{tag}"
            (logs / "latest_field_stories_v1_tag.txt").write_text(f"{tag}\n")
            Path(f"{base}.pid").write_text("99999999\n")
            Path(f"{base}.log").write_text("started\n")
            env_file = Path(temporary) / ".env.vm"
            env_file.write_text(f"ROOT={root}\nPYTHON={sys.executable}\n")
            result = subprocess.run(
                ["bash", str(SCRIPT), "status", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STATUS=DEAD_WITHOUT_STATUS", result.stdout)
            self.assertIn(
                f"OUTPUT={root}/releases/field_stories_v1_{tag}", result.stdout
            )

    def test_status_reports_completed_artifact_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            logs = root / "logs"
            release = root / "releases" / "field_stories_v1_complete-job"
            logs.mkdir(parents=True)
            release.mkdir(parents=True)
            tag = "complete-job"
            base = logs / f"field_stories_v1_{tag}"
            (logs / "latest_field_stories_v1_tag.txt").write_text(f"{tag}\n")
            Path(f"{base}.pid").write_text("99999999\n")
            Path(f"{base}.status").write_text("0\n")
            Path(f"{base}.log").write_text("complete\n")
            (release / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "daily_state": {"row_count": 12},
                            "windows": {"row_count": 3},
                        }
                    }
                )
            )
            env_file = Path(temporary) / ".env.vm"
            env_file.write_text(f"ROOT={root}\nPYTHON={sys.executable}\n")

            result = subprocess.run(
                ["bash", str(SCRIPT), "status", str(env_file)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("STATUS=0", result.stdout)
            self.assertIn("DAILY_STATE_ROWS=12", result.stdout)
            self.assertIn("WINDOWS_ROWS=3", result.stdout)


if __name__ == "__main__":
    unittest.main()
