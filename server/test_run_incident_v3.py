from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from run_incident_v3 import (
    FOCUSED_TESTS,
    _artifact_complete,
    _executable_path,
    _node_syntax_command,
    _paths,
    _require_append_gate,
    _require_nonempty_incident_release,
    build_parser,
)
from story_monitor.runner_process import RunnerError


class IncidentV3RunnerTests(unittest.TestCase):
    def test_parser_and_paths_are_deterministic(self) -> None:
        args = build_parser().parse_args(
            [
                "run", "--generation-dir", "/tmp/source",
                "--baseline-through", "2025-12-31", "--job-tag", "tag",
                "--previous-incident-dir", "/tmp/previous",
            ]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.baseline_through, "2025-12-31")
        self.assertEqual(args.previous_incident_dir, Path("/tmp/previous"))
        self.assertIn("server.test_story_map_server", FOCUSED_TESTS)
        paths = _paths(Path("/tmp/root"), "tag")
        self.assertEqual(paths["job_dir"], "/tmp/root/jobs/incident_v3_tag")
        self.assertEqual(
            paths["viewer_dir"], "/tmp/root/releases/incident_viewer_v3_tag"
        )

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "run", "--generation-dir", "/tmp/source",
                    "--baseline-through", "2025-12-31", "--job-tag", "../escape",
                ]
            )

    def test_virtualenv_python_symlink_is_not_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "base" / "python3.13"
            target.parent.mkdir()
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
            virtualenv_python = root / "venv" / "bin" / "python"
            virtualenv_python.parent.mkdir(parents=True)
            virtualenv_python.symlink_to(target)

            selected = _executable_path(virtualenv_python)

            self.assertEqual(selected, virtualenv_python.absolute())
            self.assertNotEqual(selected, target.resolve())

    def test_artifact_complete_checks_status_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(_artifact_complete(root))
            (root / "manifest.json").write_text(
                '{"run":{"status":"complete","mode":"crop_incident_v3"}}',
                encoding="utf-8",
            )
            self.assertTrue(_artifact_complete(root))
            self.assertTrue(_artifact_complete(root, mode="crop_incident_v3"))
            self.assertFalse(_artifact_complete(root, mode="other"))

    def test_zero_incident_release_fails_before_viewer_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                '{"validation":{"row_counts":{"incident_weekly_state":0}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RunnerError, "zero crop stories"):
                _require_nonempty_incident_release(root)

    def test_append_gate_and_node_syntax_command_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            previous = root / "previous"
            release.mkdir()
            previous.mkdir()
            (previous / "manifest.json").write_text(
                '{"run":{"generation_id":"previous-generation"}}',
                encoding="utf-8",
            )
            manifest = release / "manifest.json"
            manifest.write_text(
                '{"validation":{"append_stability":{"status":"passed",'
                '"previous_generation_id":"previous-generation"}}}',
                encoding="utf-8",
            )
            _require_append_gate(release, str(previous), first_release=False)
            with self.assertRaisesRegex(RunnerError, "expected 'first_release'"):
                _require_append_gate(release, None, first_release=True)

            repo = root / "repo"
            static = repo / "server" / "static"
            static.mkdir(parents=True)
            (static / "app.js").write_text("export {};\n", encoding="utf-8")
            (static / "timeline.js").write_text("export {};\n", encoding="utf-8")
            command = _node_syntax_command("/usr/bin/node", repo)
            self.assertEqual(command[:2], ["/usr/bin/node", "-e"])
            self.assertEqual(
                [Path(value).name for value in command[3:]],
                ["app.js", "timeline.js"],
            )


if __name__ == "__main__":
    unittest.main()
