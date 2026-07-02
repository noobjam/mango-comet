from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from build_story_map_bundle import advisory_build_lock, build_bundle, clear_known_outputs


def write_source(run_dir: Path, geometry_text: str = "POINT (0 0)") -> None:
    run_dir.mkdir()
    pd.DataFrame(
        [{"field_id": "field-A", "geometry_text": geometry_text}]
    ).to_parquet(run_dir / "map_field_geometry.parquet", index=False)
    pd.DataFrame(
        [
            {
                "timeline_bucket": "2025-01-01",
                "field_id": "field-A",
                "story_cluster_id": "story-A",
                "max_risk_band": "HIGH",
                "hazard_signature": "heat",
                "response_signature": "stable",
                "reportable_day_count": 1,
                "event_count": 1,
                "max_risk_rank": 3,
                "response_day_count": 1,
            }
        ]
    ).to_parquet(run_dir / "map_frame_fields.parquet", index=False)
    pd.DataFrame(
        [
            {
                "story_cluster_id": "story-A",
                "short_label": "Heat story",
                "max_risk_band": "HIGH",
                "hazard_signature": "heat",
                "response_signature": "stable",
                "event_count": 1,
                "field_count": 1,
                "crop_count": 1,
                "median_window_span_days": 1.0,
                "median_reportable_days": 1.0,
            }
        ]
    ).to_parquet(run_dir / "event_story_cluster_labels.parquet", index=False)
    pd.DataFrame(
        [
            {
                "field_id": "field-A",
                "crop_name": "crop",
                "crop_season": "season",
                "event_id": "event-A",
                "event_start_date": "2025-01-01",
                "active_end_date": "2025-01-02",
                "max_risk_band": "HIGH",
                "hazard_signature": "heat",
                "stage_signature": "growth",
                "response_signature": "stable",
                "close_reason": "boundary",
                "reportable_days": 1,
                "window_span_days": 1,
                "story_cluster_id": "story-A",
            }
        ]
    ).to_parquet(run_dir / "event_windows.parquet", index=False)
    pd.DataFrame(
        [{"field_id": "field-A", "event_id": "event-A", "story_cluster_id": "story-A"}]
    ).to_parquet(run_dir / "story_day_membership.parquet", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"run": {"status": "complete"}}), encoding="utf-8")


class StoryMapBundleTests(unittest.TestCase):
    def test_overwrite_cleanup_removes_owned_artifacts_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "story_motifs.parquet").write_bytes(b"stale")
            (out_dir / "field_geometry.parquet").write_bytes(b"stale")
            (out_dir / "notes.txt").write_text("keep", encoding="utf-8")
            summaries = out_dir / "gpu_summaries"
            summaries.mkdir()
            (summaries / "timeline_summary.parquet").write_bytes(b"stale")

            clear_known_outputs(out_dir)

            self.assertFalse((out_dir / "story_motifs.parquet").exists())
            self.assertFalse((out_dir / "field_geometry.parquet").exists())
            self.assertFalse(summaries.exists())
            self.assertEqual((out_dir / "notes.txt").read_text(encoding="utf-8"), "keep")

    def test_same_run_and_output_directory_is_rejected_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            marker = run_dir / "field_geometry.parquet"
            marker.write_bytes(b"existing-output")

            with self.assertRaisesRegex(ValueError, "must be different"):
                build_bundle(run_dir, run_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"existing-output")

    def test_missing_source_artifacts_are_rejected_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            run_dir.mkdir()
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"existing-output")
            pd.DataFrame(
                [{"field_id": "field-A", "geometry_text": "POINT (0 0)"}]
            ).to_parquet(run_dir / "map_field_geometry.parquet", index=False)

            with self.assertRaises(FileNotFoundError):
                build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"existing-output")

    def test_missing_source_directory_is_rejected_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "missing-run"
            out_dir = root / "output"
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"existing-output")

            with self.assertRaisesRegex(FileNotFoundError, "Run directory"):
                build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"existing-output")

    def test_all_invalid_geometry_leaves_existing_output_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir, geometry_text="not valid WKT")
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"known-good-output")

            with self.assertRaisesRegex(ValueError, "no valid geometries"):
                build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"known-good-output")

    def test_copy_failure_leaves_existing_output_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"known-good-output")

            with patch("build_story_map_bundle.shutil.copy2", side_effect=OSError("copy failed")):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"known-good-output")

    def test_corrupt_required_parquet_leaves_existing_output_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            (run_dir / "map_frame_fields.parquet").write_bytes(b"not parquet")
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"known-good-output")

            with self.assertRaisesRegex(ValueError, "Required Parquet artifact is unreadable"):
                build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"known-good-output")

    def test_required_parquet_schema_and_nonempty_rows_are_enforced(self) -> None:
        for case in ("missing-column", "empty"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / "run"
                out_dir = root / "output"
                write_source(run_dir)
                frames = pd.read_parquet(run_dir / "map_frame_fields.parquet")
                if case == "missing-column":
                    frames = frames.drop(columns=["response_day_count"])
                    message = "missing columns: response_day_count"
                else:
                    frames = frames.iloc[0:0]
                    message = "contains no rows"
                frames.to_parquet(run_dir / "map_frame_fields.parquet", index=False)

                with self.assertRaisesRegex(ValueError, message):
                    build_bundle(run_dir, out_dir, overwrite=True)

    def test_malformed_manifest_leaves_existing_output_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
            out_dir.mkdir()
            marker = out_dir / "field_geometry.parquet"
            marker.write_bytes(b"known-good-output")

            with self.assertRaisesRegex(ValueError, "valid JSON"):
                build_bundle(run_dir, out_dir, overwrite=True)

            self.assertEqual(marker.read_bytes(), b"known-good-output")

    def test_near_total_geometry_parse_loss_fails_default_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            rows = [{"field_id": "field-A", "geometry_text": "POINT (0 0)"}]
            rows.extend(
                {"field_id": f"field-{index}", "geometry_text": "not valid WKT"}
                for index in range(1, 20)
            )
            pd.DataFrame(rows).to_parquet(run_dir / "map_field_geometry.parquet", index=False)

            with self.assertRaisesRegex(ValueError, "Valid geometry coverage"):
                build_bundle(run_dir, out_dir, overwrite=True)

            build_bundle(
                run_dir,
                out_dir,
                overwrite=True,
                min_valid_geometry_coverage=0.05,
            )
            geometry = pd.read_parquet(out_dir / "field_geometry.parquet")
            self.assertEqual(geometry["field_id"].tolist(), ["field-A"])

    def test_poor_distinct_frame_field_join_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            base = pd.read_parquet(run_dir / "map_frame_fields.parquet").iloc[0].to_dict()
            frames = [{**base, "field_id": f"field-{index}"} for index in range(20)]
            frames[0]["field_id"] = "field-A"
            pd.DataFrame(frames).to_parquet(run_dir / "map_frame_fields.parquet", index=False)

            with self.assertRaisesRegex(ValueError, "Frame-to-geometry field coverage"):
                build_bundle(run_dir, out_dir, overwrite=True)

    def test_frame_field_ids_must_match_runtime_join_semantics(self) -> None:
        for field_id in (" field-A ", "   ", None):
            with self.subTest(field_id=field_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / "run"
                out_dir = root / "output"
                write_source(run_dir)
                frames = pd.read_parquet(run_dir / "map_frame_fields.parquet")
                frames.loc[0, "field_id"] = field_id
                frames.to_parquet(run_dir / "map_frame_fields.parquet", index=False)

                with self.assertRaisesRegex(ValueError, "empty or noncanonical field IDs"):
                    build_bundle(run_dir, out_dir, overwrite=True)

    def test_duplicate_empty_and_unusable_geometry_are_rejected(self) -> None:
        cases = {
            "duplicate": (
                [
                    {"field_id": "field-A", "geometry_text": "POINT (0 0)"},
                    {"field_id": "field-A", "geometry_text": "POINT (1 1)"},
                ],
                "duplicate field_id",
            ),
            "empty-id": ([{"field_id": "  ", "geometry_text": "POINT (0 0)"}], "empty field_id"),
            "empty-geometry": ([{"field_id": "field-A", "geometry_text": "POINT EMPTY"}], "no valid geometries"),
            "invalid-geometry": (
                [
                    {
                        "field_id": "field-A",
                        "geometry_text": "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))",
                    }
                ],
                "no valid geometries",
            ),
            "nonfinite-geometry": (
                [{"field_id": "field-A", "geometry_text": "POINT (Infinity 0)"}],
                "no valid geometries",
            ),
        }
        for name, (rows, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / "run"
                out_dir = root / "output"
                write_source(run_dir)
                pd.DataFrame(rows).to_parquet(run_dir / "map_field_geometry.parquet", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    build_bundle(run_dir, out_dir, overwrite=True)

    def test_coverage_override_ranges_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            for argument, value in (
                ("min_valid_geometry_coverage", -0.01),
                ("min_frame_geometry_coverage", 1.01),
            ):
                with self.subTest(argument=argument):
                    with self.assertRaisesRegex(ValueError, r"finite number in \[0, 1\]"):
                        build_bundle(run_dir, out_dir, overwrite=True, **{argument: value})

    def test_owned_directories_and_gpu_summary_symlink_are_cleaned_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            external_summaries = root / "external-summaries"
            write_source(run_dir)
            out_dir.mkdir()
            stale_owned = out_dir / "story_motifs.parquet"
            stale_owned.mkdir()
            (stale_owned / "stale").write_text("stale", encoding="utf-8")
            external_summaries.mkdir()
            external_marker = external_summaries / "timeline_summary.parquet"
            external_marker.write_bytes(b"external-data")
            (out_dir / "gpu_summaries").symlink_to(external_summaries, target_is_directory=True)

            build_bundle(run_dir, out_dir, overwrite=True)

            self.assertFalse(stale_owned.exists())
            self.assertFalse((out_dir / "gpu_summaries").exists())
            self.assertFalse((out_dir / "gpu_summaries").is_symlink())
            self.assertEqual(external_marker.read_bytes(), b"external-data")

    def test_install_phase_failure_rolls_back_files_directories_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            external_summaries = root / "external-summaries"
            write_source(run_dir)
            out_dir.mkdir()
            (out_dir / "field_geometry.parquet").write_bytes(b"old-geometry")
            (out_dir / "frame_fields.parquet").write_bytes(b"old-frames")
            stale_owned = out_dir / "story_motifs.parquet"
            stale_owned.mkdir()
            (stale_owned / "stale").write_text("old-motif", encoding="utf-8")
            external_summaries.mkdir()
            (external_summaries / "timeline_summary.parquet").write_bytes(b"external-data")
            summaries_link = out_dir / "gpu_summaries"
            summaries_link.symlink_to(external_summaries, target_is_directory=True)

            real_replace = Path.replace
            failure_injected = False

            def fail_during_install(path: Path, target: Path) -> Path:
                nonlocal failure_injected
                if not failure_injected and path.parent.name == "stage" and path.name == "frame_fields.parquet":
                    failure_injected = True
                    raise OSError("install failed")
                return real_replace(path, target)

            with patch.object(Path, "replace", fail_during_install):
                with self.assertRaisesRegex(OSError, "install failed"):
                    build_bundle(run_dir, out_dir, overwrite=True)

            self.assertTrue(failure_injected)
            self.assertEqual((out_dir / "field_geometry.parquet").read_bytes(), b"old-geometry")
            self.assertEqual((out_dir / "frame_fields.parquet").read_bytes(), b"old-frames")
            self.assertEqual((stale_owned / "stale").read_text(encoding="utf-8"), "old-motif")
            self.assertTrue(summaries_link.is_symlink())
            self.assertEqual(
                (summaries_link / "timeline_summary.parquet").read_bytes(),
                b"external-data",
            )

    def test_parent_scoped_lock_rejects_overlapping_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)

            with advisory_build_lock(out_dir):
                with self.assertRaisesRegex(RuntimeError, "already targeting"):
                    build_bundle(run_dir, out_dir, overwrite=True)

    def test_successful_overwrite_installs_staged_bundle_and_preserves_unknown_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            out_dir = root / "output"
            write_source(run_dir)
            out_dir.mkdir()
            (out_dir / "field_geometry.parquet").write_bytes(b"stale-geometry")
            (out_dir / "story_motifs.parquet").write_bytes(b"stale-motif")
            (out_dir / "notes.txt").write_text("preserve", encoding="utf-8")
            summaries = out_dir / "gpu_summaries"
            summaries.mkdir()
            (summaries / "timeline_summary.parquet").write_bytes(b"stale-summary")

            build_bundle(run_dir, out_dir, overwrite=True)

            geometry = pd.read_parquet(out_dir / "field_geometry.parquet")
            self.assertEqual(geometry["field_id"].tolist(), ["field-A"])
            self.assertIn("geometry_geojson", geometry.columns)
            frames = pd.read_parquet(out_dir / "frame_fields.parquet")
            self.assertEqual(frames["field_id"].tolist(), ["field-A"])
            self.assertFalse((out_dir / "story_motifs.parquet").exists())
            self.assertFalse(summaries.exists())
            self.assertEqual((out_dir / "notes.txt").read_text(encoding="utf-8"), "preserve")
            profile = json.loads((out_dir / "geometry_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["output"], str((out_dir / "field_geometry.parquet").resolve()))
            self.assertEqual(profile["valid_geometry_coverage"], 1.0)
            self.assertEqual(profile["frame_geometry_coverage"], 1.0)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["run"]["viewer_ready"])
            self.assertEqual(manifest["outputs"]["frame_fields"], "frame_fields.parquet")
            self.assertEqual(manifest["outputs"]["field_geometry"], "field_geometry.parquet")
            self.assertTrue(
                all((out_dir / name).is_file() for name in manifest["outputs"].values())
            )


if __name__ == "__main__":
    unittest.main()
