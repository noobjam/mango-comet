from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from shapely.geometry import shape

from story_map_server import Settings, StoryMapStore
from story_monitor.incident_viewer_v3 import export_incident_viewer_v3


class IncidentViewerV3Tests(unittest.TestCase):
    def test_exports_server_tables_and_exact_complete_footprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            output = root / "viewer"
            _write_source(source)
            _write_incident(incident, source)

            result = export_incident_viewer_v3(
                incident, source, output, threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mode"], "crop_incident_v3")
            frames = pd.read_parquet(output / "frame_fields.parquet")
            self.assertEqual(len(frames), 3)
            self.assertTrue((frames["story_cluster_id"] == frames["incident_id"]).all())
            current_stage = frames.loc[
                (frames["timeline_bucket"].astype(str) == "2025-01-13")
                & (frames["field_id"] == "field-1"),
                "stage_bucket",
            ].iloc[0]
            self.assertEqual(current_stage, "flowering")
            for column in (
                "crop_name", "stage_bucket", "exposure_id", "incident_state",
                "fresh_response_evidence", "evaluable", "is_data_gap",
            ):
                self.assertIn(column, frames.columns)

            labels = pd.read_parquet(output / "cluster_labels.parquet")
            self.assertEqual(labels["story_cluster_id"].tolist(), ["incident-1"])
            events = pd.read_parquet(output / "event_windows.parquet")
            self.assertEqual(set(events["event_id"]), {"event-1", "event-2"})
            story_days = pd.read_parquet(output / "story_day_membership.parquet")
            self.assertEqual(len(story_days), 3)
            self.assertTrue((story_days["story_cluster_id"] == "incident-1").all())

            footprints = pd.read_parquet(output / "incident_footprints.parquet")
            self.assertEqual(len(footprints), 2)
            self.assertFalse(footprints["low_zoom_omitted"].any())
            self.assertEqual(
                set(footprints["footprint_geometry_method"]),
                {"exact_union_of_grid_rectangles"},
            )
            first = shape(json.loads(footprints.iloc[0]["geometry_geojson"]))
            second = shape(json.loads(footprints.iloc[1]["geometry_geojson"]))
            self.assertEqual(first.geom_type, "Polygon")
            self.assertEqual(second.geom_type, "MultiPolygon")
            self.assertLess(second.area, second.envelope.area)
            self.assertEqual(footprints["footprint_area_km2"].tolist(), [50.0, 50.0])
            self.assertIsNotNone(footprints.iloc[0]["pressure_geometry_geojson"])
            self.assertEqual(footprints.iloc[0]["pressure_cell_count"], 2)
            self.assertIsNone(footprints.iloc[0]["impact_geometry_geojson"])
            self.assertIsNotNone(footprints.iloc[1]["impact_geometry_geojson"])
            self.assertEqual(footprints.iloc[1]["impact_cell_count"], 2)

            timeline = pd.read_parquet(
                output / "gpu_summaries" / "timeline_summary.parquet"
            )
            self.assertEqual(timeline["field_count"].tolist(), [2, 1])
            store = StoryMapStore(
                Settings(
                    run_dir=output, static_dir=root, host="127.0.0.1", port=8877,
                    raster_tiles="", raster_attribution="", default_feature_limit=5000,
                    max_feature_limit=20000, log_level="INFO",
                )
            )
            self.assertTrue(store.health()["ok"])
            self.assertEqual(store.timeline()["source"], "gpu_summary")
            self.assertEqual(
                store.frame_features(
                    timeline_bucket="2025-01-06", bbox=None, limit=0
                )["meta"]["feature_count"],
                2,
            )
            footprint_payload = store.incident_footprints(
                timeline_bucket="2025-01-06"
            )
            self.assertTrue(footprint_payload["meta"]["complete"])
            footprint_properties = footprint_payload["features"][0]["properties"]
            self.assertNotIn("pressure_geometry", footprint_properties)
            self.assertNotIn("impact_geometry", footprint_properties)
            detail = store.incident_detail("incident-1")
            self.assertEqual(
                detail["footprints"][0]["pressure_geometry"]["type"], "Polygon"
            )
            self.assertIsNone(detail["footprints"][0]["impact_geometry"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["mode"], "crop_incident_v3")
            self.assertTrue(manifest["run"]["geometry_optimized"])
            self.assertEqual(manifest["run"]["incident_count"], 1)
            self.assertFalse(manifest["run"]["api_ui_gate_passed"])
            self.assertFalse(manifest["run"]["map_publication_approved"])
            self.assertFalse(manifest["semantics"]["centroid_trails_used"])
            self.assertFalse(manifest["semantics"]["convex_hulls_used"])
            self.assertIn("row_count", manifest["artifacts"]["incident_footprints.parquet"])
            for name in (
                "incident_stage_summary.parquet", "incident_windows.parquet",
                "incident_membership.parquet", "exposure_weekly_state.parquet",
                "exposure_links.parquet", "incident_lineage.parquet",
            ):
                self.assertTrue((output / name).is_file(), name)

    def test_missing_frame_geometry_fails_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            output = root / "viewer"
            _write_source(source, include_second_geometry=False)
            _write_incident(incident, source)

            with self.assertRaisesRegex(ValueError, "Frame-to-geometry field coverage"):
                export_incident_viewer_v3(
                    incident, source, output, threads=1,
                    min_valid_geometry_coverage=1.0,
                    min_frame_geometry_coverage=1.0,
                )
            self.assertFalse(output.exists())

    def test_timeline_keeps_incident_week_without_field_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            output = root / "viewer"
            _write_source(source)
            _write_incident(incident, source, membership_free_week=True)

            export_incident_viewer_v3(
                incident,
                source,
                output,
                threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )

            timeline = pd.read_parquet(
                output / "gpu_summaries" / "timeline_summary.parquet"
            )
            self.assertEqual(
                timeline["timeline_bucket"].astype(str).tolist(),
                ["2025-01-06", "2025-01-13", "2025-01-20"],
            )
            self.assertEqual(timeline["field_count"].tolist(), [2, 1, 0])
            self.assertEqual(
                timeline["story_cluster_count"].tolist(), [1, 1, 1]
            )
            self.assertEqual(timeline["event_count"].tolist(), [2, 1, 0])
            footprints = pd.read_parquet(output / "incident_footprints.parquet")
            self.assertEqual(len(footprints), 3)

    def test_unknown_footprint_cell_and_existing_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            output = root / "viewer"
            _write_source(source)
            _write_incident(incident, source, unknown_cell=True)

            with self.assertRaisesRegex(ValueError, "unknown grid cells"):
                export_incident_viewer_v3(incident, source, output, threads=1)
            self.assertFalse(output.exists())

            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("immutable", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_incident_viewer_v3(incident, source, output, threads=1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "immutable")


def _write_source(root: Path, *, include_second_geometry: bool = True) -> None:
    root.mkdir()
    pd.DataFrame(
        [
            {"field_id": "field-1", "geometry_text": "POLYGON ((30 -2, 30.08 -2, 30.08 -1.92, 30 -1.92, 30 -2))", "district": "D"},
            *(
                [{"field_id": "field-2", "geometry_text": "POLYGON ((30.1 -2, 30.18 -2, 30.18 -1.92, 30.1 -1.92, 30.1 -2))", "district": "D"}]
                if include_second_geometry else []
            ),
        ]
    ).to_parquet(root / "map_field_geometry.parquet", index=False)
    pd.DataFrame(
        [{"field_id": "field-1", "crop_instance_id": "crop-1", "observation_date": "2025-01-06"}]
    ).to_parquet(root / "daily_causal_signals.parquet", index=False)
    pd.DataFrame(
        [{"event_id": "event-1", "timeline_bucket": "2025-01-06"}]
    ).to_parquet(root / "event_state_snapshots.parquet", index=False)
    pd.DataFrame(
        [{"event_id": "event-1", "event_start_date": "2025-01-06"}]
    ).to_parquet(root / "event_windows.parquet", index=False)
    pd.DataFrame(
        [
            _source_day("field-1", "event-1", "crop-1", "2025-01-06", 3, True),
            _source_day("field-2", "event-2", "crop-2", "2025-01-08", 2, True),
            _source_day("field-1", "event-1", "crop-1", "2025-01-13", 1, False),
        ]
    ).to_parquet(root / "story_day_membership.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete", "immutable": True,
                    "generation_id": "generation-test", "as_of_date": "2025-01-31",
                },
                "policy": {"version": "source-v1", "sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )


def _write_incident(
    root: Path,
    source: Path,
    *,
    unknown_cell: bool = False,
    membership_free_week: bool = False,
) -> None:
    root.mkdir()
    cells = []
    for week in ("2025-01-06", "2025-01-13"):
        for x in (0, 1, 3):
            cells.append(
                {
                    "timeline_bucket": week, "hazard_family": "heat",
                    "cell_id": f"cell_{x}_0", "cell_x": x, "cell_y": 0,
                    "min_lon": 30.0 + x * 0.1, "max_lon": 30.1 + x * 0.1,
                    "min_lat": -2.0, "max_lat": -1.9, "cell_size_km": 5.0,
                }
            )
    pd.DataFrame(cells).to_parquet(root / "weekly_exposure_cells.parquet", index=False)
    second_cells = '["g:1:0","g:99:0"]' if unknown_cell else '["g:1:0","g:3:0"]'
    weekly_rows = [
        _weekly("2025-01-06", '["g:0:0","g:1:0"]', "ACTIVE", 2, 1, 0),
        _weekly("2025-01-13", second_cells, "RECOVERING", 0, 0, 1),
    ]
    if membership_free_week:
        weekly_rows.append(
            _weekly(
                "2025-01-20",
                '["g:1:0","g:3:0"]',
                "PRESSURE_QUIET",
                0,
                0,
                0,
            )
        )
    pd.DataFrame(weekly_rows).to_parquet(
        root / "incident_weekly_state.parquet", index=False
    )
    pd.DataFrame(
        [
            _membership("2025-01-06", "field-1", "crop-1", "event-1", "pressure_core", "ACTIVE", "vegetative", "g:0:0", True),
            _membership("2025-01-06", "field-2", "crop-2", "event-2", "watch_frontier", "WATCH", "flowering", "g:1:0", False),
            _membership("2025-01-13", "field-1", "crop-1", "event-1", "impact_lag", "RECOVERING", "vegetative", "g:1:0", True),
        ]
    ).to_parquet(root / "incident_membership.parquet", index=False)
    pd.DataFrame(
        [
            {
                "timeline_bucket": "2025-01-06", "field_id": "field-1",
                "crop_instance_id": "crop-1", "crop_name": "maize",
                "stage_bucket": "vegetative",
            },
            {
                "timeline_bucket": "2025-01-06", "field_id": "field-2",
                "crop_instance_id": "crop-2", "crop_name": "maize",
                "stage_bucket": "flowering",
            },
            {
                "timeline_bucket": "2025-01-13", "field_id": "field-1",
                "crop_instance_id": "crop-1", "crop_name": "maize",
                "stage_bucket": "flowering",
            },
        ]
    ).to_parquet(root / "field_week_context.parquet", index=False)
    pd.DataFrame(
        [
            _lane("2025-01-06", "field-1", "crop-1", "event-1", "ACTIVE", "MED-HIGH", 3, "stable"),
            _lane("2025-01-06", "field-2", "crop-2", "event-2", "WATCH", "LOW-MED", 2, "stable"),
            _lane("2025-01-13", "field-1", "crop-1", "event-1", "RECOVERING", "LOW", 1, "recovery"),
        ]
    ).to_parquet(root / "event_week_lanes.parquet", index=False)
    pd.DataFrame(
        [
            {
                "incident_id": "incident-1", "exposure_id": "exposure-1",
                "crop_name": "maize", "hazard_family": "heat",
                "first_evidence_week": "2025-01-06", "confirmed_week": "2025-01-06",
                "closed_week": None, "terminal_state": "RECOVERING",
                "right_censored": True, "observed_week_count": 2,
                "active_component_week_count": 1, "peak_week": "2025-01-06",
                "peak_affected_field_count": 1, "relapse_count": 0,
                "data_gap_count": 0,
            }
        ]
    ).to_parquet(root / "incident_windows.parquet", index=False)
    pd.DataFrame(
        [{"incident_id": "incident-1", "exposure_id": "exposure-1", "crop_name_normalized": "maize"}]
    ).to_parquet(root / "incident_catalog.parquet", index=False)
    stage_rows = [
            {
                "timeline_bucket": "2025-01-06", "incident_id": "incident-1",
                "exposure_id": "exposure-1", "crop_name": "maize",
                "hazard_family": "heat", "stage_bucket": "vegetative",
                "monitored_crop_instance_count": 2, "evaluable_crop_instance_count": 2,
                "pressure_core_crop_instance_count": 1, "severe_crop_instance_count": 0,
                "watch_frontier_crop_instance_count": 1, "impact_lag_crop_instance_count": 0,
                "impact_signal_rate": 0.5,
            },
            {
                "timeline_bucket": "2025-01-13", "incident_id": "incident-1",
                "exposure_id": "exposure-1", "crop_name": "maize",
                "hazard_family": "heat", "stage_bucket": "flowering",
                "monitored_crop_instance_count": 2, "evaluable_crop_instance_count": 2,
                "pressure_core_crop_instance_count": 0, "severe_crop_instance_count": 0,
                "watch_frontier_crop_instance_count": 0, "impact_lag_crop_instance_count": 1,
                "impact_signal_rate": 0.0,
            },
        ]
    if membership_free_week:
        stage_rows.append(
            {
                "timeline_bucket": "2025-01-20",
                "incident_id": "incident-1",
                "exposure_id": "exposure-1",
                "crop_name": "maize",
                "hazard_family": "heat",
                "stage_bucket": "flowering",
                "monitored_crop_instance_count": 0,
                "evaluable_crop_instance_count": 0,
                "pressure_core_crop_instance_count": 0,
                "severe_crop_instance_count": 0,
                "watch_frontier_crop_instance_count": 0,
                "impact_lag_crop_instance_count": 0,
                "impact_signal_rate": 0.0,
            }
        )
    pd.DataFrame(stage_rows).to_parquet(
        root / "incident_stage_summary.parquet", index=False
    )
    pd.DataFrame(
        [
            {"timeline_bucket": "2025-01-06", "hazard_family": "heat", "component_id": "component-1", "exposure_id": "exposure-1"},
            {"timeline_bucket": "2025-01-13", "hazard_family": "heat", "component_id": "component-2", "exposure_id": "exposure-1"},
        ]
    ).to_parquet(root / "exposure_component_assignments.parquet", index=False)
    pd.DataFrame(columns=["parent_exposure_id", "child_exposure_id", "lineage_type"]).to_parquet(
        root / "exposure_links.parquet", index=False
    )
    pd.DataFrame(
        [
            {"timeline_bucket": "2025-01-06", "exposure_id": "exposure-1", "component_id": "component-1"},
            {"timeline_bucket": "2025-01-13", "exposure_id": "exposure-1", "component_id": "component-2"},
        ]
    ).to_parquet(root / "exposure_weekly_state.parquet", index=False)
    pd.DataFrame(columns=["parent_exposure_id", "child_exposure_id", "lineage_type"]).to_parquet(
        root / "incident_lineage.parquet", index=False
    )

    artifact_names = [
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    ]
    artifacts = {
        name: {"sha256": _sha256(root / name), "size_bytes": (root / name).stat().st_size}
        for name in artifact_names
    }
    source_hash = _sha256(source / "manifest.json")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete", "immutable": True,
                    "generation_id": "incident-generation-test",
                    "source_generation_id": "generation-test",
                },
                "source": {"generation_manifest_sha256": source_hash},
                "policy": {
                    "version": "incident-v3-test", "sha256": "a" * 64,
                    "calibration_status": "uncalibrated",
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _source_day(
    field_id: str, event_id: str, crop_instance_id: str, date: str,
    rank: int, observed: bool,
) -> dict[str, object]:
    return {
        "field_id": field_id, "event_id": event_id,
        "story_cluster_id": "old-story", "crop_instance_id": crop_instance_id,
        "observation_date": date,
        "event_state": "ACTIVE" if observed else "RECOVERING",
        "hazard_signature": "heat", "daily_pressure_rank": rank,
        "daily_response_class": "stable" if observed else "recovery",
        "pressure_observed": observed,
    }


def _weekly(
    week: str, cells: str, state: str, core: int, decline: int, recovery: int,
) -> dict[str, object]:
    return {
        "timeline_bucket": week, "incident_id": "incident-1",
        "exposure_id": "exposure-1", "crop_name": "maize",
        "hazard_family": "heat", "component_id": "component-1",
        "incident_state": state, "pressure_core_field_count": core,
        "severe_field_count": 0, "watch_frontier_field_count": 1 if core else 0,
        "impact_lag_field_count": 1 if recovery else 0,
        "fresh_decline_field_count": decline,
        "fresh_recovery_field_count": recovery,
        "stage_distribution": '{"flowering":0.5,"vegetative":0.5}',
        "coverage_adequate": True, "footprint_cell_ids_json": cells,
        "knowledge_time": week, "knowledge_time_inferred": False,
        "pressure_cell_ids_json": cells if core else "[]",
        "impact_cell_ids_json": cells if recovery else "[]",
        "watch_cell_ids_json": cells if core else "[]",
        "footprint_carried_forward": False, "footprint_area_km2": 50.0,
        "right_censored": True, "monitored_count": 2, "evaluable_count": 2,
        "pressure_core_count": core, "severe_count": 0,
        "impact_lag_count": recovery,
        "global_crop_week_unmappable_instance_count": 0,
        "active_count": core, "affected_count": max(core, recovery),
    }


def _membership(
    week: str, field_id: str, crop_instance: str, event: str, role: str,
    state: str, stage: str, grid: str, fresh: bool,
) -> dict[str, object]:
    return {
        "timeline_bucket": week, "incident_id": "incident-1",
        "exposure_id": "exposure-1", "component_id": "component-1",
        "crop_name_normalized": "maize", "hazard_family": "heat",
        "field_id": field_id, "crop_instance_id": crop_instance,
        "episode_id": event, "membership_role": role, "event_state": state,
        "response_class": "recovery" if state == "RECOVERING" else "stable",
        "fresh_response_evidence": fresh, "evaluable": True,
        "is_data_gap": False, "stage_bucket": stage, "grid_id": grid,
        "knowledge_time": week,
    }


def _lane(
    week: str, field: str, crop_instance: str, event: str, state: str,
    band: str, rank: int, response: str,
) -> dict[str, object]:
    return {
        "timeline_bucket": week, "snapshot_as_of_date": week,
        "event_id": event, "field_id": field, "crop_instance_id": crop_instance,
        "crop_name": "maize", "crop_season": "A",
        "stage_bucket": "vegetative", "hazard_family": "heat",
        "event_state": state, "event_start_date": "2025-01-06",
        "event_end_date": None, "close_reason": "right_censored",
        "current_risk_rank": rank, "current_risk_band": band,
        "max_risk_rank": rank, "max_risk_band": band,
        "daily_response_class": response, "fresh_response_evidence": response != "stable",
        "reportable_day_count": 2, "response_day_count": 1,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
