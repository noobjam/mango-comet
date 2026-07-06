from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from run_incident_motifs_v4 import _prefix_config, build_parser, main


class IncidentMotifV4CliTests(unittest.TestCase):
    def test_discovery_requires_explicit_engine_and_ordered_boundaries(self) -> None:
        args = build_parser().parse_args(
            [
                "discover",
                "--incident-dir", "/tmp/incidents",
                "--evidence-dir", "/tmp/evidence",
                "--viewer-dir", "/tmp/viewer",
                "--output-dir", "/tmp/output",
                "--train-through", "2025-12-31",
                "--calibration-through", "2026-03-31",
                "--evaluation-through", "2026-06-30",
                "--engine", "gpu",
            ]
        )
        self.assertEqual(args.engine, "gpu")
        self.assertEqual(args.viewer_dir, Path("/tmp/viewer"))
        self.assertEqual(_prefix_config(args).s2_acquisition_horizons, (0, 1, 2, 4))

    def test_discovery_requires_viewer_directory(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "discover",
                    "--incident-dir", "/tmp/incidents",
                    "--evidence-dir", "/tmp/evidence",
                    "--output-dir", "/tmp/output",
                    "--train-through", "2025-12-31",
                    "--calibration-through", "2026-03-31",
                    "--evaluation-through", "2026-06-30",
                    "--engine", "gpu",
                ]
            )

    def test_horizon_parser_rejects_unsorted_or_nonzero_first_s2(self) -> None:
        parser = build_parser()
        common = [
            "discover", "--incident-dir", "/tmp/i", "--evidence-dir", "/tmp/e",
            "--viewer-dir", "/tmp/v", "--output-dir", "/tmp/o",
            "--train-through", "2025-01-01",
            "--calibration-through", "2025-02-01",
            "--evaluation-through", "2025-03-01", "--engine", "cpu",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args([*common, "--weather-day-horizons", "14,7"])
        with self.assertRaises(SystemExit):
            parser.parse_args([*common, "--s2-acquisition-horizons", "1,2"])

    def test_discovery_forwards_viewer_directory(self) -> None:
        with patch(
            "run_incident_motifs_v4.build_diagnostic_motif_release_v4",
            return_value={"status": "complete"},
        ) as build:
            code = main(
                [
                    "discover",
                    "--incident-dir", "/tmp/incidents",
                    "--evidence-dir", "/tmp/evidence",
                    "--viewer-dir", "/tmp/viewer",
                    "--output-dir", "/tmp/output",
                    "--train-through", "2025-12-31",
                    "--calibration-through", "2026-03-31",
                    "--evaluation-through", "2026-06-30",
                    "--engine", "cpu",
                    "--heartbeat-seconds", "1",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs["viewer_dir"], Path("/tmp/viewer"))

    def test_failed_evaluation_hard_gates_exit_nonzero(self) -> None:
        with patch(
            "run_incident_motifs_v4.evaluate_prefix_release_v4",
            return_value={
                "status": "failed_hard_gates",
                "hard_gates_passed": False,
                "metrics": {},
            },
        ):
            code = main(
                [
                    "evaluate", "--discovery-dir", "/tmp/discovery",
                    "--prefix-model-dir", "/tmp/prefix", "--final-labels",
                    "/tmp/labels.parquet", "--output-dir", "/tmp/evaluation",
                    "--heartbeat-seconds", "1",
                ]
            )
        self.assertEqual(code, 21)


if __name__ == "__main__":
    unittest.main()
