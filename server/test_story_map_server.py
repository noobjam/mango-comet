from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import tempfile
from threading import Event, Lock, Thread
import time
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler, Request, build_opener
from urllib.error import HTTPError

import pandas as pd

from story_map_server import (
    BoundedThreadingHTTPServer,
    MAX_GEOMETRY_REQUEST_BYTES,
    ResponseCache,
    _cache_byte_budgets,
    Settings,
    StoryMapStore,
    make_handler,
)
from http.server import ThreadingHTTPServer


# These integration tests only call their own loopback HTTP servers. Bypass
# machine-wide corporate proxy settings so 127.0.0.1 can never be sent to an
# outbound proxy on developer or VM hosts.
urlopen = build_opener(ProxyHandler({})).open


def _geometry_row(field_id: str, x: float, y: float) -> dict[str, object]:
    coordinates = [[x, y], [x + 0.2, y], [x + 0.2, y + 0.2], [x, y + 0.2], [x, y]]
    return {
        "field_id": field_id,
        "geometry_geojson": json.dumps({"type": "Polygon", "coordinates": [coordinates]}),
        "min_lon": x,
        "min_lat": y,
        "max_lon": x + 0.2,
        "max_lat": y + 0.2,
        "centroid_lon": x + 0.1,
        "centroid_lat": y + 0.1,
        "district": "district",
        "sector": "sector",
        "cell": "cell",
        "village": field_id,
    }


def _frame_row(bucket: str, field_id: str, story_id: str) -> dict[str, object]:
    return {
        "timeline_bucket": bucket,
        "field_id": field_id,
        "story_cluster_id": story_id,
        "max_risk_band": "HIGH",
        "hazard_signature": "heat",
        "response_signature": "no_material_response_proxy",
        "reportable_day_count": 2,
        "event_count": 1,
        "max_risk_rank": 3,
        "response_day_count": 0,
    }


def _incident_footprint_row(
    incident_id: str,
    *,
    crop_name: str,
    hazard_family: str,
    incident_state: str,
    stage_bucket: str,
    geometry: dict[str, object],
    monitored: int,
    evaluable: int,
    affected: int,
    severe: int,
    carried: bool = False,
) -> dict[str, object]:
    points: list[list[float]] = []

    def collect(value: object) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and all(isinstance(item, (int, float)) for item in value[:2])
        ):
            points.append([float(value[0]), float(value[1])])
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(geometry.get("coordinates"))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "timeline_bucket": "2025-01-01",
        "incident_id": incident_id,
        "exposure_id": f"exposure-{incident_id}",
        "crop_name": crop_name,
        "hazard_family": hazard_family,
        "incident_state": incident_state,
        "stage_bucket": stage_bucket,
        "geometry_geojson": json.dumps(geometry),
        "geometry_type": geometry["type"],
        "min_lon": min(xs),
        "min_lat": min(ys),
        "max_lon": max(xs),
        "max_lat": max(ys),
        "footprint_geometry_method": "exact_union_of_grid_rectangles",
        "low_zoom_omitted": False,
        "monitored_count": monitored,
        "evaluable_count": evaluable,
        "affected_count": affected,
        "severe_count": severe,
        "pressure_core_field_count": affected,
        "severe_field_count": severe,
        "watch_frontier_field_count": 1,
        "impact_lag_field_count": 1 if incident_state == "RECOVERING" else 0,
        "footprint_carried_forward": carried,
        "pressure_geometry_geojson": json.dumps(geometry),
        "impact_geometry_geojson": None,
        "watch_geometry_geojson": None,
        "pressure_cell_count": 1,
        "impact_cell_count": 1,
        "watch_cell_count": 1,
        "footprint_area_km2": 1.5,
        "coincident_group_id": f"coincident-{incident_id}",
        "coincident_incident_count": 1,
        "coincident_incident_index": 0,
        "coincident_crop_names_json": json.dumps([crop_name]),
        "footprint_cell_ids_json": '["g:0:0","g:1:0"]',
        "pressure_cell_ids_json": '["g:0:0"]',
        "impact_cell_ids_json": '["g:1:0"]',
        "watch_cell_ids_json": '["g:2:0"]',
        "first_evidence_week": "2025-01-01",
        "confirmed_week": "2025-01-01",
        "pressure_off_week": None,
        "recovered_week": None,
        "closed_week": None,
        "relapse_count": 0,
        "data_gap_count": 0,
        "right_censored": True,
        "is_physical_movement": False,
    }


def _write_incident_api_artifacts(run_dir: Path) -> None:
    polygon_a = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4], [0.0, 0.0]]],
    }
    multipolygon_b = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[1.0, 1.0], [1.2, 1.0], [1.2, 1.2], [1.0, 1.2], [1.0, 1.0]]],
            [[[1.4, 1.4], [1.6, 1.4], [1.6, 1.6], [1.4, 1.6], [1.4, 1.4]]],
        ],
    }
    polygon_c = {
        "type": "Polygon",
        "coordinates": [[[10.0, 10.0], [10.2, 10.0], [10.2, 10.2], [10.0, 10.2], [10.0, 10.0]]],
    }
    footprints = pd.DataFrame(
        [
            _incident_footprint_row(
                "story-A", crop_name="maize", hazard_family="heat",
                incident_state="ACTIVE", stage_bucket="vegetative",
                geometry=polygon_a, monitored=10, evaluable=8, affected=3, severe=1,
            ),
            _incident_footprint_row(
                "story-B", crop_name="beans", hazard_family="ponding_flooding",
                incident_state="RECOVERING", stage_bucket="flowering",
                geometry=multipolygon_b, monitored=5, evaluable=4, affected=2,
                severe=0, carried=True,
            ),
            _incident_footprint_row(
                "story-C", crop_name="maize", hazard_family="heat",
                incident_state="ACTIVE", stage_bucket="flowering",
                geometry=polygon_c, monitored=6, evaluable=6, affected=1, severe=0,
            ),
        ]
    )
    weekly = footprints.drop(
        columns=["geometry_geojson", "footprint_geometry_method", "low_zoom_omitted"]
    )
    later = weekly[weekly["incident_id"] == "story-A"].iloc[0].copy()
    later["timeline_bucket"] = "2025-01-08"
    later["incident_state"] = "RECOVERING"
    later["footprint_carried_forward"] = True
    no_membership = later.copy()
    no_membership["timeline_bucket"] = "2025-01-22"
    no_membership["incident_state"] = "PRESSURE_QUIET"
    for column in (
        "monitored_count", "evaluable_count", "affected_count", "severe_count",
        "pressure_core_field_count", "severe_field_count",
        "watch_frontier_field_count", "impact_lag_field_count",
    ):
        no_membership[column] = 0
    pd.concat(
        [weekly, pd.DataFrame([later, no_membership])], ignore_index=True
    ).to_parquet(
        run_dir / "incident_weekly_state.parquet", index=False
    )
    later_footprint = footprints[footprints["incident_id"] == "story-A"].iloc[0].copy()
    later_footprint["timeline_bucket"] = "2025-01-08"
    later_footprint["incident_state"] = "RECOVERING"
    later_footprint["stage_bucket"] = "flowering"
    later_footprint["footprint_area_km2"] = 1.2
    later_footprint["footprint_carried_forward"] = True
    quiet_footprint = later_footprint.copy()
    quiet_footprint["timeline_bucket"] = "2025-01-22"
    quiet_footprint["incident_state"] = "PRESSURE_QUIET"
    quiet_footprint["footprint_area_km2"] = 1.1
    pd.concat(
        [footprints, pd.DataFrame([later_footprint, quiet_footprint])],
        ignore_index=True,
    ).to_parquet(run_dir / "incident_footprints.parquet", index=False)
    pd.DataFrame(
        [
            {
                "timeline_bucket": "2025-01-01", "incident_id": "story-A",
                "crop_name": "maize", "hazard_family": "heat",
                "stage_bucket": "vegetative", "monitored_crop_instance_count": 10,
                "evaluable_crop_instance_count": 8,
                "pressure_core_crop_instance_count": 3,
                "severe_crop_instance_count": 1,
                "affected_crop_instance_count": 8,
            },
            {
                "timeline_bucket": "2025-01-01", "incident_id": "story-A",
                "crop_name": "maize", "hazard_family": "heat",
                "stage_bucket": "flowering", "monitored_crop_instance_count": 2,
                "evaluable_crop_instance_count": 2,
                "pressure_core_crop_instance_count": 1,
                "severe_crop_instance_count": 0,
                "affected_crop_instance_count": 1,
            },
            {
                "timeline_bucket": "2025-01-01", "incident_id": "story-A",
                "crop_name": "maize", "hazard_family": "heat",
                "stage_bucket": "maturity_or_harvest",
                "monitored_crop_instance_count": 5,
                "evaluable_crop_instance_count": 5,
                "pressure_core_crop_instance_count": 0,
                "severe_crop_instance_count": 0,
                "affected_crop_instance_count": 0,
            },
            {
                "timeline_bucket": "2025-01-08", "incident_id": "story-A",
                "crop_name": "maize", "hazard_family": "heat",
                "stage_bucket": "flowering", "monitored_crop_instance_count": 9,
                "evaluable_crop_instance_count": 7,
                "pressure_core_crop_instance_count": 0,
                "severe_crop_instance_count": 0,
                "affected_crop_instance_count": 1,
            },
        ]
    ).to_parquet(run_dir / "incident_stage_summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "incident_id": "story-A", "exposure_id": "exposure-story-A",
                "crop_name": "maize", "hazard_family": "heat",
                "first_evidence_week": "2025-01-01", "confirmed_week": "2025-01-01",
                "closed_week": None, "terminal_state": "RECOVERING",
                "right_censored": True, "observed_week_count": 2,
            }
        ]
    ).to_parquet(run_dir / "incident_windows.parquet", index=False)
    pd.DataFrame(
        [
            {
                "lineage_id": "lineage-in", "timeline_bucket": "2025-01-01",
                "lineage_type": "split", "parent_incident_id": "incident-parent",
                "child_incident_id": "story-A", "score": 0.8,
            },
            {
                "lineage_id": "lineage-out", "timeline_bucket": "2025-01-08",
                "lineage_type": "merge", "parent_incident_id": "story-A",
                "child_incident_id": "incident-child", "score": 0.9,
            },
        ]
    ).to_parquet(run_dir / "incident_lineage.parquet", index=False)


class StoryMapServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        run_dir = Path(cls.temp_dir.name)
        pd.DataFrame(
            [
                _geometry_row("A", 0.0, 0.0),
                _geometry_row("B", 1.0, 1.0),
                _geometry_row("C", 10.0, 10.0),
                _geometry_row("D", 1.5, 1.5),
            ]
        ).to_parquet(run_dir / "field_geometry.parquet", index=False)
        pd.DataFrame(
            [
                _frame_row("2025-01-01", "A", "story-A"),
                _frame_row("2025-01-01", "B", "story-B"),
                _frame_row("2025-01-01", "C", "story-C"),
                _frame_row("2025-01-08", "A", "story-A"),
                _frame_row("2025-01-08", "C", "story-C"),
                _frame_row("2025-01-15", "A", "story-A"),
                _frame_row("2025-01-15", "D", "story-D"),
            ]
        ).to_parquet(run_dir / "frame_fields.parquet", index=False)
        labels = []
        for field_id in "ABCDE":
            labels.append(
                {
                    "story_cluster_id": f"story-{field_id}",
                    "short_label": f"Heat story {field_id}",
                    "max_risk_band": "HIGH",
                    "hazard_signature": "heat",
                    "response_signature": "no_material_response_proxy",
                    "event_count": 1,
                    "field_count": 1,
                    "crop_count": 1,
                    "median_window_span_days": 2.0,
                    "median_reportable_days": 2.0,
                }
            )
        pd.DataFrame(labels).to_parquet(run_dir / "cluster_labels.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "field_id": "A",
                    "story_cluster_id": "story-A",
                    "event_id": f"event-A-{day}",
                    "event_start_date": f"2025-01-{day:02d}",
                    "active_end_date": f"2025-01-{day + 1:02d}",
                }
                for day in (1, 8, 15)
            ]
        ).to_parquet(run_dir / "event_windows.parquet", index=False)
        pd.DataFrame(
            [{"field_id": "A", "story_cluster_id": "story-A", "event_id": "event-A"}]
        ).to_parquet(run_dir / "story_day_membership.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "timeline_bucket": "2025-01-01",
                    "snapshot_as_of_date": "2025-01-05",
                    "field_id": "A",
                    "crop_name": "Maize",
                    "crop_season": "Season A",
                    "event_id": "event-A",
                    "event_state": "ACTIVE",
                    "hazard_signature": "heat",
                    "max_risk_rank": 3,
                    "max_risk_band": "MED-HIGH",
                    "daily_pressure_rank": 3,
                    "daily_response_class": "no_new_acquisition",
                    "right_censored": True,
                    "requires_review": False,
                    "revision": 1,
                },
                {
                    "timeline_bucket": "2025-01-08",
                    "snapshot_as_of_date": "2025-01-12",
                    "field_id": "A",
                    "crop_name": "Maize",
                    "crop_season": "Season A",
                    "event_id": "event-A",
                    "event_state": "QUIET_PENDING",
                    "hazard_signature": "heat",
                    "max_risk_rank": 3,
                    "max_risk_band": "MED-HIGH",
                    "daily_pressure_rank": 1,
                    "daily_response_class": "recovery",
                    "right_censored": True,
                    "requires_review": False,
                    "revision": 1,
                },
            ]
        ).to_parquet(run_dir / "event_state_snapshots.parquet", index=False)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run": {
                        "story_cluster_count": 4,
                        "output_dir": "/private/story-run",
                    },
                    "input": {"input_parquet": "/private/source.parquet"},
                    "parameters": {
                        "map_top_clusters": 4,
                        "temp_dir": "/private/duckdb-tmp",
                    },
                    "outputs": {"event_windows": "/private/event_windows.parquet"},
                }
            ),
            encoding="utf-8",
        )

        cls.settings = Settings(
            run_dir=run_dir,
            static_dir=Path(__file__).resolve().parent / "static",
            host="127.0.0.1",
            port=0,
            raster_tiles="",
            raster_attribution="",
            default_feature_limit=2,
            max_feature_limit=10,
            log_level="ERROR",
            cache_seconds=60,
            cache_entries=16,
            gzip_min_bytes=1,
        )
        cls.store = StoryMapStore(cls.settings)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.store, cls.settings))
        cls.server_thread = Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join(timeout=3)
        cls.temp_dir.cleanup()

    def test_exact_story_filter_and_motif_contract(self) -> None:
        frame = self.store.frame_features(
            timeline_bucket="2025-01-15",
            bbox=None,
            limit=0,
            filters={"story_cluster_id": "story-A"},
        )
        self.assertEqual([item["properties"]["field_id"] for item in frame["features"]], ["A"])
        self.assertEqual(frame["features"][0]["properties"]["motif_family"], "heat")

        motifs = self.store.motifs(None, 10)
        self.assertEqual(motifs["motifs"], motifs["exact_stories"])
        self.assertNotIn("story-E", {item["story_cluster_id"] for item in motifs["exact_stories"]})
        self.assertIn("motif_family", motifs["facets"])
        self.assertEqual(motifs["taxonomy"]["source"], "hazard_signature_fallback")

    def test_frame_state_excludes_geometry_coordinates_and_static_admin(self) -> None:
        state = self.store.frame_state(
            timeline_bucket="2025-01-01",
            bbox=(-1.0, -1.0, 3.0, 3.0),
            limit=10,
            filters={"motif_family": "heat"},
        )
        self.assertEqual([row["field_id"] for row in state["rows"]], ["A", "B"])
        self.assertEqual(state["meta"]["source_row_count"], 2)
        self.assertEqual(state["meta"]["state_count"], 2)
        self.assertTrue(state["meta"]["bbox_applied"])
        self.assertNotIn("bbox", state["meta"])
        forbidden = {
            "geometry",
            "geometry_geojson",
            "geometry_text",
            "geometry_format",
            "bbox",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
            "centroid_lon",
            "centroid_lat",
            "district",
            "sector",
            "cell",
            "village",
        }
        self.assertTrue(all(forbidden.isdisjoint(row) for row in state["rows"]))
        self.assertTrue(state["geometry_version"].startswith("geom-sha256-"))
        self.assertEqual(state["geometry_version"], self.store.geometry_version())
        self.assertEqual(state["geometry_version"], StoryMapStore(self.settings).geometry_version())

        url = (
            f"http://127.0.0.1:{self.httpd.server_port}"
            "/api/frame-state/2025-01-01?bbox=-1,-1,3,3&limit=10"
        )
        with urlopen(url, timeout=3) as response:
            http_state = json.loads(response.read())
        self.assertEqual([row["field_id"] for row in http_state["rows"]], ["A", "B"])
        self.assertTrue(all(forbidden.isdisjoint(row) for row in http_state["rows"]))

    def test_frame_is_one_field_row_and_prioritizes_live_current_risk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "field_geometry.parquet", "cluster_labels.parquet",
                "event_windows.parquet", "story_day_membership.parquet", "manifest.json",
            ):
                shutil.copy2(self.settings.run_dir / name, run_dir / name)
            rows = [
                {
                    **_frame_row("2025-01-01", "A", "story-A"),
                    "event_id": "event-old", "event_state": "CLOSED_RECOVERED",
                    "current_risk_band": "LOW", "current_risk_rank": 1,
                },
                {
                    **_frame_row("2025-01-01", "A", "story-B"),
                    "event_id": "event-live", "event_state": "SEVERE",
                    "current_risk_band": "HIGH", "current_risk_rank": 4,
                },
                {
                    **_frame_row("2025-01-01", "B", "story-B"),
                    "event_id": "event-quiet", "event_state": "QUIET_PENDING",
                    "current_risk_band": "LOW", "current_risk_rank": 1,
                },
            ]
            pd.DataFrame(rows).to_parquet(run_dir / "frame_fields.parquet", index=False)
            store = StoryMapStore(Settings(**{**self.settings.__dict__, "run_dir": run_dir}))

            frame = store.frame_state(
                timeline_bucket="2025-01-01", bbox=None, limit=10, filters={}
            )
            high = store.frame_state(
                timeline_bucket="2025-01-01", bbox=None, limit=10,
                filters={"current_risk_band": "HIGH"},
            )

        self.assertEqual(frame["meta"]["source_row_count"], 2)
        self.assertEqual(len(frame["rows"]), 2)
        selected_a = next(row for row in frame["rows"] if row["field_id"] == "A")
        self.assertEqual(selected_a["event_id"], "event-live")
        self.assertEqual(selected_a["concurrent_event_count"], 2)
        self.assertEqual([row["field_id"] for row in high["rows"]], ["A"])

    def test_geometry_post_is_versioned_bounded_and_deduplicated(self) -> None:
        base_url = f"http://127.0.0.1:{self.httpd.server_port}/api/geometry"
        version = self.store.geometry_version()

        payload = json.dumps(
            {
                "geometry_version": version,
                "field_ids": ["D", "A", "D", "missing"],
            }
        ).encode("utf-8")
        request = Request(
            base_url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=3) as response:
            geometry = json.loads(response.read())
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(geometry["geometry_version"], version)
        self.assertEqual(
            [feature["properties"]["field_id"] for feature in geometry["features"]],
            ["D", "A"],
        )
        self.assertEqual(geometry["meta"]["requested_field_count"], 3)
        self.assertEqual(geometry["meta"]["feature_count"], 2)
        self.assertEqual(geometry["meta"]["missing_field_ids"], ["missing"])
        self.assertEqual(geometry["features"][0]["properties"]["village"], "D")
        self.assertIn("bbox", geometry["features"][0]["properties"])
        self.assertEqual(geometry["features"][0]["geometry"]["type"], "Polygon")

        stale_request = Request(
            base_url,
            data=json.dumps(
                {"geometry_version": "geom-sha256-stale", "field_ids": ["A"]}
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(stale_request, timeout=3)
        self.assertEqual(raised.exception.code, 409)
        conflict = json.loads(raised.exception.read())
        self.assertEqual(conflict["geometry_version"], version)

        excessive_request = Request(
            base_url,
            data=json.dumps(
                {"geometry_version": version, "field_ids": ["A"] * 2001}
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(excessive_request, timeout=3)
        self.assertEqual(raised.exception.code, 413)

        oversized_request = Request(
            base_url,
            data=b"{" + (b" " * MAX_GEOMETRY_REQUEST_BYTES) + b"}",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(oversized_request, timeout=3)
        self.assertEqual(raised.exception.code, 413)

    def test_evolution_reports_nonphysical_centers_and_nonzero_overlap(self) -> None:
        evolution = self.store.evolution({"story_cluster_id": "story-A"})
        self.assertEqual(evolution["kind"], "aggregate_activity_center")
        self.assertFalse(evolution["is_physical_movement"])
        self.assertEqual(evolution["bucket_count"], 3)
        self.assertNotIn("features", evolution)
        first, second, third = evolution["points"]
        self.assertEqual(first["break_reason"], "start")
        self.assertFalse(first["trail_segment_allowed"])
        self.assertEqual(first["entering_field_count"], 1)
        self.assertEqual(first["center_lon"], 0.1)
        self.assertEqual(first["center_lat"], 0.1)
        for point in (second, third):
            self.assertTrue(point["consecutive"])
            self.assertTrue(point["trail_segment_allowed"])
            self.assertIsNone(point["break_reason"])
            self.assertEqual(point["persisting_field_count"], 1)
            self.assertEqual(point["jaccard_overlap"], 1.0)
            self.assertEqual(point["p50_dispersion_km"], 0.0)
            self.assertEqual(point["p90_dispersion_km"], 0.0)

        base_url = f"http://127.0.0.1:{self.httpd.server_port}"
        with urlopen(f"{base_url}/api/evolution?story_cluster_id=story-A", timeout=3) as response:
            http_evolution = json.loads(response.read())
        self.assertEqual(http_evolution["points"], evolution["points"])
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{base_url}/api/evolution", timeout=3)
        self.assertEqual(raised.exception.code, 400)

    def test_field_trajectory_returns_causal_weekly_prefix_states(self) -> None:
        trajectory = self.store.field_trajectory("A", 20)
        self.assertTrue(trajectory["available"])
        self.assertEqual(trajectory["mode"], "causal_weekly_event_prefix")
        self.assertEqual(
            [state["event_state"] for state in trajectory["states"]],
            ["ACTIVE", "QUIET_PENDING"],
        )
        url = (
            f"http://127.0.0.1:{self.httpd.server_port}"
            "/api/field/A/trajectory?limit=20"
        )
        with urlopen(url, timeout=3) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["states"], trajectory["states"])

    def test_incident_v3_complete_footprints_filters_and_drilldown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "field_geometry.parquet", "cluster_labels.parquet",
                "event_windows.parquet", "story_day_membership.parquet", "manifest.json",
            ):
                shutil.copy2(self.settings.run_dir / name, run_dir / name)
            frames = pd.read_parquet(self.settings.run_dir / "frame_fields.parquet")
            frames["incident_id"] = frames["story_cluster_id"]
            frames["crop_name"] = frames["field_id"].map(
                {"A": "maize", "B": "beans", "C": "maize", "D": "maize"}
            )
            frames["stage_bucket"] = frames["field_id"].map(
                {"A": "vegetative", "B": "flowering", "C": "flowering", "D": "vegetative"}
            )
            frames["incident_state"] = frames["field_id"].map(
                {"A": "ACTIVE", "B": "RECOVERING", "C": "ACTIVE", "D": "ACTIVE"}
            )
            frames.to_parquet(run_dir / "frame_fields.parquet", index=False)
            _write_incident_api_artifacts(run_dir)

            settings = Settings(
                **{
                    **self.settings.__dict__,
                    "run_dir": run_dir,
                    "default_feature_limit": 1,
                    "max_feature_limit": 1,
                }
            )
            store = StoryMapStore(settings)
            facets = store.motifs(None, 10)["facets"]
            self.assertEqual(
                {row["crop_name"] for row in facets["crop_name"]},
                {"beans", "maize"},
            )
            self.assertEqual(
                {row["stage_bucket"] for row in facets["stage_bucket"]},
                {"flowering", "vegetative"},
            )
            self.assertEqual(
                {row["incident_state"] for row in facets["incident_state"]},
                {"ACTIVE", "PRESSURE_QUIET", "RECOVERING"},
            )
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, settings))
            thread = Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{httpd.server_port}"
            try:
                self.assertTrue(store.has_incident_v3())
                timeline = store.timeline()
                self.assertEqual(timeline["source"], "incident_artifacts")
                self.assertEqual(timeline["activity_unit"], "incident_stories")
                self.assertEqual(
                    [row["timeline_bucket"] for row in timeline["buckets"]],
                    ["2025-01-01", "2025-01-08", "2025-01-22"],
                )
                self.assertEqual(timeline["buckets"][-1]["field_count"], 0)
                self.assertEqual(timeline["buckets"][-1]["activity_count"], 1)
                self.assertEqual(
                    timeline["buckets"][-1]["activity_unit"],
                    "incident_stories",
                )
                footprints_url = (
                    f"{base_url}/api/incident-footprints/2025-01-01"
                    "?limit=1&complete_case=1"
                )
                with urlopen(footprints_url, timeout=3) as response:
                    footprints_body = response.read().decode("utf-8")
                    footprints = json.loads(footprints_body)
                    self.assertEqual(response.headers["X-Cache"], "MISS")
                with urlopen(footprints_url, timeout=3) as response:
                    self.assertEqual(response.headers["X-Cache"], "HIT")
                    response.read()
                self.assertEqual(len(footprints["features"]), 3)
                self.assertEqual(footprints["meta"]["source_footprint_count"], 3)
                self.assertEqual(footprints["meta"]["matching_footprint_count"], 3)
                self.assertTrue(footprints["meta"]["complete"])
                self.assertFalse(footprints["meta"]["truncated"])
                self.assertFalse(footprints["meta"]["feature_cap_applied"])
                self.assertNotIn(str(run_dir), footprints_body)
                types = {feature["geometry"]["type"] for feature in footprints["features"]}
                self.assertEqual(types, {"Polygon", "MultiPolygon"})

                filtered_url = (
                    f"{base_url}/api/incident-footprints/2025-01-01"
                    "?incident_id=story-A&crop_name=maize&hazard_family=heat"
                    "&incident_state=ACTIVE&stage_bucket=vegetative&filter_case=1"
                )
                with urlopen(filtered_url, timeout=3) as response:
                    filtered = json.loads(response.read())
                self.assertEqual(len(filtered["features"]), 1)
                properties = filtered["features"][0]["properties"]
                self.assertEqual(properties["incident_id"], "story-A")
                self.assertEqual(properties["monitored_count"], 10)
                self.assertEqual(properties["evaluable_count"], 8)
                self.assertEqual(properties["affected_count"], 3)
                self.assertEqual(properties["severe_count"], 1)
                self.assertNotIn("pressure_cell_ids_json", properties)
                self.assertNotIn("impact_cell_ids_json", properties)
                self.assertNotIn("watch_cell_ids_json", properties)
                self.assertFalse(properties["is_physical_movement"])
                self.assertEqual(filtered["meta"]["matching_footprint_count"], 1)
                self.assertEqual(filtered["meta"]["source_footprint_count"], 3)

                secondary_stage_url = (
                    f"{base_url}/api/incident-footprints/2025-01-01"
                    "?incident_id=story-A&stage_bucket=flowering&secondary_stage=1"
                )
                with urlopen(secondary_stage_url, timeout=3) as response:
                    secondary_stage = json.loads(response.read())
                self.assertEqual(len(secondary_stage["features"]), 1)
                self.assertEqual(
                    secondary_stage["features"][0]["properties"]["stage_bucket"],
                    "vegetative",
                )
                self.assertEqual(
                    secondary_stage["meta"]["filters"]["stage_bucket"],
                    "flowering",
                )
                denominator_only = store.incident_footprints(
                    timeline_bucket="2025-01-01",
                    filters={
                        "incident_id": "story-A",
                        "stage_bucket": "maturity_or_harvest",
                    },
                )
                self.assertEqual(denominator_only["features"], [])

                unreadable_frames = run_dir / "unreadable-frames.parquet"
                unreadable_frames.write_text("not parquet", encoding="utf-8")
                with patch.object(store, "frame_path", unreadable_frames):
                    activity = store.activity(
                        {"incident_id": "story-A", "crop_name": "maize"}
                    )
                self.assertEqual(activity["source"], "incident_artifacts")
                self.assertFalse(activity["uses_frame_fields"])
                self.assertEqual(
                    [row["timeline_bucket"] for row in activity["buckets"]],
                    ["2025-01-01", "2025-01-08", "2025-01-22"],
                )
                self.assertEqual(activity["buckets"][-1]["field_count"], 0)
                self.assertEqual(
                    activity["buckets"][-1]["story_cluster_count"], 1
                )
                stage_activity = store.activity(
                    {"incident_id": "story-A", "stage_bucket": "flowering"}
                )
                self.assertEqual(
                    [row["timeline_bucket"] for row in stage_activity["buckets"]],
                    ["2025-01-01", "2025-01-08"],
                )
                self.assertEqual(
                    store.activity(
                        {
                            "incident_id": "story-A",
                            "stage_bucket": "maturity_or_harvest",
                        }
                    )["buckets"],
                    [],
                )

                with urlopen(f"{base_url}/api/incident/story-A?detail_case=1", timeout=3) as response:
                    detail_body = response.read().decode("utf-8")
                    detail = json.loads(detail_body)
                self.assertEqual(detail["window"]["incident_id"], "story-A")
                self.assertEqual(
                    detail["footprints"][0]["pressure_geometry"]["type"],
                    "Polygon",
                )
                self.assertEqual(
                    [row["geometry"]["type"] for row in detail["footprints"]],
                    ["Polygon", "Polygon", "Polygon"],
                )
                self.assertEqual(
                    [row["timeline_bucket"] for row in detail["footprints"]],
                    ["2025-01-01", "2025-01-08", "2025-01-22"],
                )
                self.assertTrue(
                    all(not row["is_physical_movement"] for row in detail["footprints"])
                )
                self.assertNotIn("geometry_geojson", detail["footprints"][0])
                self.assertEqual(
                    [row["timeline_bucket"] for row in detail["weekly_state"]],
                    ["2025-01-01", "2025-01-08", "2025-01-22"],
                )
                self.assertEqual(
                    [row["stage_bucket"] for row in detail["stage_rows"]],
                    [
                        "flowering", "maturity_or_harvest", "vegetative",
                        "flowering",
                    ],
                )
                self.assertEqual(
                    [row["lineage_id"] for row in detail["lineage"]["incoming"]],
                    ["lineage-in"],
                )
                self.assertEqual(
                    [row["lineage_id"] for row in detail["lineage"]["outgoing"]],
                    ["lineage-out"],
                )
                self.assertNotIn(str(run_dir), detail_body)

                frame_url = (
                    f"{base_url}/api/frame/2025-01-01?limit=10&incident_id=story-A"
                    "&crop_name=maize&stage_bucket=vegetative&incident_state=ACTIVE"
                )
                with urlopen(frame_url, timeout=3) as response:
                    frame = json.loads(response.read())
                self.assertEqual(
                    [feature["properties"]["field_id"] for feature in frame["features"]],
                    ["A"],
                )
                self.assertEqual(frame["features"][0]["properties"]["incident_id"], "story-A")

                with self.assertRaises(HTTPError) as raised:
                    urlopen(f"{base_url}/api/incident/missing?detail_case=2", timeout=3)
                self.assertEqual(raised.exception.code, 404)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=3)

    def test_incident_v3_overlap_presentation_metadata_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "field_geometry.parquet", "frame_fields.parquet",
                "cluster_labels.parquet", "event_windows.parquet",
                "story_day_membership.parquet", "manifest.json",
            ):
                shutil.copy2(self.settings.run_dir / name, run_dir / name)
            _write_incident_api_artifacts(run_dir)
            footprint_path = run_dir / "incident_footprints.parquet"
            footprints = pd.read_parquet(footprint_path).drop(
                columns=[
                    "coincident_group_id", "coincident_incident_count",
                    "coincident_incident_index", "coincident_crop_names_json",
                ]
            )
            footprints.to_parquet(footprint_path, index=False)
            store = StoryMapStore(Settings(**{
                **self.settings.__dict__, "run_dir": run_dir,
            }))

            payload = store.incident_footprints(
                timeline_bucket="2025-01-01",
                filters={"incident_id": "story-A"},
            )
            self.assertEqual(len(payload["features"]), 1)
            properties = payload["features"][0]["properties"]
            self.assertIsNone(properties["coincident_group_id"])
            self.assertIsNone(properties["coincident_incident_count"])
            self.assertIsNone(properties["coincident_incident_index"])

    def test_incident_routes_fail_cleanly_for_legacy_bundle_and_bad_dates(self) -> None:
        self.assertFalse(self.store.has_incident_v3())
        aliased = self.store.frame_features(
            timeline_bucket="2025-01-01", bbox=None, limit=0,
            filters={"incident_id": "story-A"},
        )
        self.assertEqual(
            [feature["properties"]["field_id"] for feature in aliased["features"]],
            ["A"],
        )
        unsupported_crop = self.store.frame_features(
            timeline_bucket="2025-01-01", bbox=None, limit=0,
            filters={"crop_name": "maize"},
        )
        self.assertEqual(len(unsupported_crop["features"]), 3)

        base_url = f"http://127.0.0.1:{self.httpd.server_port}"
        for path in (
            "/api/incident-footprints/2025-01-01?legacy_case=1",
            "/api/incident/story-A?legacy_case=2",
        ):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                urlopen(base_url + path, timeout=3)
            self.assertEqual(raised.exception.code, 404)
            body = raised.exception.read().decode("utf-8")
            self.assertIn("not available", body)
            self.assertNotIn(str(self.settings.run_dir), body)

        with self.assertRaises(HTTPError) as raised:
            urlopen(
                f"{base_url}/api/incident-footprints/not-a-date?legacy_bad_date=1",
                timeout=3,
            )
        self.assertEqual(raised.exception.code, 400)
        self.assertIn("YYYY-MM-DD", raised.exception.read().decode("utf-8"))

    def test_evolution_breaks_zero_overlap_but_allows_later_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            pd.DataFrame(
                [
                    _geometry_row("A", 0.0, 0.0),
                    _geometry_row("B", 1.0, 1.0),
                ]
            ).to_parquet(run_dir / "field_geometry.parquet", index=False)
            pd.DataFrame(
                [
                    _frame_row("2025-01-01", "A", "story-A"),
                    _frame_row("2025-01-08", "B", "story-A"),
                    _frame_row("2025-01-15", "B", "story-A"),
                ]
            ).to_parquet(run_dir / "frame_fields.parquet", index=False)
            for name in (
                "cluster_labels.parquet",
                "event_windows.parquet",
                "story_day_membership.parquet",
                "manifest.json",
            ):
                shutil.copy2(self.settings.run_dir / name, run_dir / name)
            store = StoryMapStore(
                Settings(
                    **{
                        **self.settings.__dict__,
                        "run_dir": run_dir,
                    }
                )
            )
            evolution = store.evolution({"story_cluster_id": "story-A"})

        first, zero_overlap, overlap = evolution["points"]
        self.assertEqual(first["break_reason"], "start")
        self.assertTrue(zero_overlap["consecutive"])
        self.assertEqual(zero_overlap["persisting_field_count"], 0)
        self.assertEqual(zero_overlap["entering_field_count"], 1)
        self.assertEqual(zero_overlap["exiting_field_count"], 1)
        self.assertEqual(zero_overlap["jaccard_overlap"], 0.0)
        self.assertFalse(zero_overlap["trail_segment_allowed"])
        self.assertEqual(zero_overlap["break_reason"], "zero_field_overlap")
        self.assertTrue(overlap["consecutive"])
        self.assertEqual(overlap["persisting_field_count"], 1)
        self.assertEqual(overlap["jaccard_overlap"], 1.0)
        self.assertTrue(overlap["trail_segment_allowed"])
        self.assertIsNone(overlap["break_reason"])

    def test_activity_contains_no_spatial_movement_coordinates(self) -> None:
        activity = self.store.activity({"story_cluster_id": "story-A"})
        self.assertEqual(activity["bucket_count"], 3)
        forbidden = {
            "centroid_lon",
            "centroid_lat",
            "aggregate_centroid_lon",
            "aggregate_centroid_lat",
            "representative_field_id",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
        }
        self.assertTrue(all(forbidden.isdisjoint(bucket) for bucket in activity["buckets"]))

    def test_empty_frame_diagnostic_error_is_generic_publicly(self) -> None:
        secret = f"sensitive failure at {self.settings.run_dir}/private.parquet"
        with self.assertLogs("story_map_server", level="ERROR") as captured:
            with patch.object(
                self.store, "bounds", side_effect=RuntimeError(secret)
            ):
                frame = self.store.frame_features(
                    timeline_bucket="2099-01-01",
                    bbox=(-1.0, -1.0, 1.0, 1.0),
                    limit=10,
                )

        diagnostics = frame["meta"]["diagnostics"]
        self.assertEqual(
            diagnostics["diagnostic_error"], "diagnostic_query_failed"
        )
        public_payload = json.dumps(frame)
        self.assertNotIn(secret, public_payload)
        self.assertNotIn(str(self.settings.run_dir), public_payload)
        self.assertIn(secret, "\n".join(captured.output))

    def test_frame_bbox_and_limit_expose_truncation(self) -> None:
        frame = self.store.frame_features(
            timeline_bucket="2025-01-15",
            bbox=(-1.0, -1.0, 3.0, 3.0),
            limit=1,
            filters={"motif_family": "heat"},
        )
        self.assertEqual(frame["meta"]["source_row_count"], 2)
        self.assertEqual(frame["meta"]["feature_count"], 1)
        self.assertTrue(frame["meta"]["truncated"])
        self.assertEqual(frame["meta"]["query_row_count"], 2)

    def test_public_nonpositive_limit_is_bounded_but_internal_zero_is_unlimited(self) -> None:
        base_url = f"http://127.0.0.1:{self.httpd.server_port}/api/frame/2025-01-01"
        for raw_limit in ("0", "-9"):
            with self.subTest(limit=raw_limit):
                with urlopen(f"{base_url}?limit={raw_limit}", timeout=3) as response:
                    payload = json.loads(response.read())
                self.assertEqual(payload["meta"]["limit"], self.settings.default_feature_limit)
                self.assertEqual(payload["meta"]["feature_count"], self.settings.default_feature_limit)
                self.assertFalse(payload["meta"]["unlimited"])
                self.assertTrue(payload["meta"]["truncated"])

        internal = self.store.frame_features(
            timeline_bucket="2025-01-01",
            bbox=None,
            limit=0,
        )
        self.assertEqual(internal["meta"]["feature_count"], 3)
        self.assertTrue(internal["meta"]["unlimited"])

    def test_raw_geometry_bbox_filters_before_render_limit(self) -> None:
        raw_dir = self.settings.run_dir / "raw-run"
        raw_dir.mkdir(exist_ok=True)
        geometry_rows = []
        for field_id, x, y in (("A", 0.0, 0.0), ("B", 1.0, 1.0), ("C", 10.0, 10.0)):
            geometry_rows.append(
                {
                    "field_id": field_id,
                    "geometry_text": f"POLYGON (({x} {y}, {x + 0.2} {y}, {x + 0.2} {y + 0.2}, {x} {y + 0.2}, {x} {y}))",
                    "geometry_format": "wkt",
                    "district": "district",
                    "sector": "sector",
                    "cell": "cell",
                    "village": field_id,
                }
            )
        pd.DataFrame(geometry_rows).to_parquet(raw_dir / "map_field_geometry.parquet", index=False)
        shutil.copy2(self.settings.run_dir / "frame_fields.parquet", raw_dir / "map_frame_fields.parquet")
        shutil.copy2(self.settings.run_dir / "cluster_labels.parquet", raw_dir / "event_story_cluster_labels.parquet")
        for name in ("event_windows.parquet", "story_day_membership.parquet", "manifest.json"):
            shutil.copy2(self.settings.run_dir / name, raw_dir / name)
        raw_settings = Settings(
            **{
                **self.settings.__dict__,
                "run_dir": raw_dir,
                "default_feature_limit": 1,
            }
        )
        raw_store = StoryMapStore(raw_settings)
        frame = raw_store.frame_features(
            timeline_bucket="2025-01-01",
            bbox=(9.0, 9.0, 11.0, 11.0),
            limit=1,
        )
        self.assertEqual([item["properties"]["field_id"] for item in frame["features"]], ["C"])
        self.assertEqual(frame["meta"]["source_row_count"], 1)
        self.assertEqual(frame["meta"]["story_cluster_count"], 1)
        self.assertEqual(frame["meta"]["reportable_day_count"], 2)
        self.assertEqual(frame["meta"]["event_count"], 1)
        self.assertEqual(frame["meta"]["query_row_count"], 3)
        self.assertFalse(frame["meta"]["truncated"])

        bounded_viewport = raw_store.frame_features(
            timeline_bucket="2025-01-01",
            bbox=(-1.0, -1.0, 2.0, 2.0),
            limit=1,
        )
        self.assertEqual(bounded_viewport["meta"]["source_row_count"], 2)
        self.assertEqual(bounded_viewport["meta"]["story_cluster_count"], 2)
        self.assertEqual(bounded_viewport["meta"]["reportable_day_count"], 4)
        self.assertEqual(bounded_viewport["meta"]["event_count"], 2)
        self.assertEqual(bounded_viewport["meta"]["feature_count"], 1)
        self.assertTrue(bounded_viewport["meta"]["truncated"])

    def test_trail_is_one_nearest_prior_footprint_per_field(self) -> None:
        trail = self.store.trail_features(
            timeline_bucket="2025-01-15",
            filters={"motif_family": "heat"},
            lookback=2,
            bbox=(-1.0, -1.0, 3.0, 3.0),
        )
        properties = [item["properties"] for item in trail["features"]]
        self.assertEqual({item["field_id"] for item in properties}, {"A", "B"})
        self.assertEqual(len(properties), len({item["field_id"] for item in properties}))
        self.assertTrue(all(item["timeline_bucket"] != "2025-01-15" for item in properties))
        a = next(item for item in properties if item["field_id"] == "A")
        self.assertEqual(a["timeline_bucket"], "2025-01-08")
        self.assertTrue(a["persists_to_current"])
        self.assertEqual(trail["meta"]["current_field_count"], 2)
        self.assertEqual(trail["meta"]["prior_field_count"], 2)
        self.assertEqual(trail["meta"]["previous_field_count"], 1)
        self.assertEqual(trail["meta"]["persisting_field_count"], 1)
        self.assertEqual(trail["meta"]["departed_field_count"], 0)
        self.assertEqual(trail["meta"]["new_current_field_count"], 1)
        self.assertEqual(trail["meta"]["transition_scope"], "open_previous_bucket")

        bounded = self.store.trail_features(
            timeline_bucket="2025-01-15",
            filters={"motif_family": "heat"},
            lookback=2,
            bbox=(-1.0, -1.0, 3.0, 3.0),
            limit=1,
        )
        self.assertEqual(len(bounded["features"]), 1)
        self.assertEqual(bounded["meta"]["prior_field_count"], 2)
        self.assertTrue(bounded["meta"]["truncated"])

    def test_trail_open_transitions_use_immediately_previous_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "field_geometry.parquet", "cluster_labels.parquet",
                "event_windows.parquet", "story_day_membership.parquet", "manifest.json",
            ):
                shutil.copy2(self.settings.run_dir / name, run_dir / name)
            rows = []
            for bucket, event_state, risk, rank in (
                ("2025-01-01", "ACTIVE", "MED-HIGH", 3),
                ("2025-01-08", "CLOSED_RECOVERED", "LOW", 1),
                ("2025-01-15", "ACTIVE", "MED-HIGH", 3),
            ):
                rows.append(
                    {
                        **_frame_row(bucket, "A", "story-A"),
                        "event_id": f"event-{bucket}", "event_state": event_state,
                        "current_risk_band": risk, "current_risk_rank": rank,
                    }
                )
            pd.DataFrame(rows).to_parquet(run_dir / "frame_fields.parquet", index=False)
            store = StoryMapStore(Settings(**{**self.settings.__dict__, "run_dir": run_dir}))
            trail = store.trail_features(
                timeline_bucket="2025-01-15",
                filters={"story_cluster_id": "story-A"},
                lookback=2,
            )

        self.assertEqual(trail["meta"]["previous_field_count"], 0)
        self.assertEqual(trail["meta"]["persisting_field_count"], 0)
        self.assertEqual(trail["meta"]["new_current_field_count"], 1)
        self.assertEqual(len(trail["features"]), 1)
        prior = trail["features"][0]["properties"]
        self.assertEqual(prior["timeline_bucket"], "2025-01-01")
        self.assertEqual(prior["event_state"], "ACTIVE")
        self.assertEqual(prior["current_risk_band"], "MED-HIGH")

    def test_recent_field_events_and_invalid_bbox(self) -> None:
        events = self.store.field_events("A", 2)["events"]
        self.assertEqual([item["event_start_date"] for item in events], ["2025-01-15", "2025-01-08"])

        url = f"http://127.0.0.1:{self.httpd.server_port}/api/frame/2025-01-15?bbox=nan,0,1,1"
        with self.assertRaises(HTTPError) as raised:
            urlopen(url, timeout=3)
        self.assertEqual(raised.exception.code, 400)
        self.assertIn("finite numbers", raised.exception.read().decode("utf-8"))

        invalid_limit_url = (
            f"http://127.0.0.1:{self.httpd.server_port}"
            "/api/frame/2025-01-15?limit=not-an-integer"
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(invalid_limit_url, timeout=3)
        self.assertEqual(raised.exception.code, 400)
        self.assertIn("limit must be an integer", raised.exception.read().decode("utf-8"))

    def test_internal_value_error_is_generic_http_500(self) -> None:
        original_activity = self.store.activity

        def fail_activity(_filters: object) -> dict[str, object]:
            raise ValueError("sensitive internal invariant")

        self.store.activity = fail_activity  # type: ignore[method-assign]
        try:
            url = (
                f"http://127.0.0.1:{self.httpd.server_port}"
                "/api/activity?internal_failure_regression=1"
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(url, timeout=3)
            body = raised.exception.read().decode("utf-8")
        finally:
            self.store.activity = original_activity  # type: ignore[method-assign]
        self.assertEqual(raised.exception.code, 500)
        self.assertIn("server could not complete", body)
        self.assertNotIn("sensitive internal invariant", body)

    def test_http_gzip_and_raw_response_cache_headers(self) -> None:
        url = (
            f"http://127.0.0.1:{self.httpd.server_port}"
            "/api/activity?story_cluster_id=story-A&cache_test=1"
        )
        request = Request(url, headers={"Accept-Encoding": "gzip"})
        with urlopen(request, timeout=3) as response:
            first_body = gzip.decompress(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(response.version, 11)
            self.assertEqual(response.headers["Content-Encoding"], "gzip")
            self.assertEqual(response.headers["X-Cache"], "MISS")
            self.assertEqual(response.headers["Cache-Control"], "private, max-age=60")
            self.assertEqual(response.headers["Vary"], "Accept-Encoding")
        with urlopen(request, timeout=3) as response:
            second_body = gzip.decompress(response.read())
            self.assertEqual(response.headers["X-Cache"], "HIT")
        self.assertEqual(first_body, second_body)

    def test_public_api_metadata_does_not_disclose_host_paths(self) -> None:
        base_url = f"http://127.0.0.1:{self.httpd.server_port}"
        with urlopen(f"{base_url}/api/health", timeout=3) as response:
            health_body = response.read().decode("utf-8")
            health = json.loads(health_body)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertNotIn("paths", health)
        self.assertNotIn(str(self.settings.run_dir), health_body)

        with urlopen(f"{base_url}/api/manifest?path_redaction=1", timeout=3) as response:
            manifest_body = response.read().decode("utf-8")
            manifest = json.loads(manifest_body)
            self.assertEqual(response.headers["Cache-Control"], "private, max-age=60")
        self.assertNotIn("/private/", manifest_body)
        self.assertNotIn(str(self.settings.run_dir), manifest_body)
        self.assertNotIn("input", manifest)
        self.assertNotIn("outputs", manifest)
        self.assertNotIn("output_dir", manifest["run"])
        self.assertNotIn("temp_dir", manifest["parameters"])
        self.assertNotIn("run_dir", manifest["server"])

    def test_explicit_gzip_zero_overrides_wildcard_in_any_order(self) -> None:
        url = (
            f"http://127.0.0.1:{self.httpd.server_port}"
            "/api/activity?story_cluster_id=story-A&gzip_precedence=1"
        )
        for header in ("*;q=1, gzip;q=0", "gzip;q=0, *;q=1"):
            with self.subTest(accept_encoding=header):
                request = Request(url, headers={"Accept-Encoding": header})
                with urlopen(request, timeout=3) as response:
                    body = response.read()
                    self.assertIsNone(response.headers["Content-Encoding"])
                self.assertEqual(json.loads(body)["bucket_count"], 3)

        request = Request(url, headers={"Accept-Encoding": "*;q=0, gzip;q=0.5"})
        with urlopen(request, timeout=3) as response:
            self.assertEqual(response.headers["Content-Encoding"], "gzip")
            self.assertEqual(json.loads(gzip.decompress(response.read()))["bucket_count"], 3)

    def test_cached_request_bypasses_per_query_concurrency_gate(self) -> None:
        gate_settings = Settings(
            **{
                **self.settings.__dict__,
                "query_concurrency": 1,
            }
        )
        httpd = BoundedThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.store, gate_settings),
            max_concurrency=1,
        )
        server_thread = Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        cached_url = f"{base_url}/api/timeline?query_gate_cached=1"
        with urlopen(cached_url, timeout=3) as response:
            response.read()
            self.assertEqual(response.headers["X-Cache"], "MISS")

        started = Event()
        release = Event()
        worker_errors: list[BaseException] = []
        original_activity = self.store.activity

        def blocking_activity(filters: dict[str, str] | None) -> dict[str, object]:
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("query gate test timed out")
            return original_activity(filters)

        def request_slow_activity() -> None:
            try:
                with urlopen(f"{base_url}/api/activity?query_gate_slow=1", timeout=4) as response:
                    response.read()
            except BaseException as exc:  # pragma: no cover - asserted below.
                worker_errors.append(exc)

        self.store.activity = blocking_activity  # type: ignore[method-assign]
        slow_thread = Thread(target=request_slow_activity, daemon=True)
        slow_thread.start()
        try:
            self.assertTrue(started.wait(timeout=1), "uncached query did not acquire the query gate")
            request_started = time.perf_counter()
            with urlopen(cached_url, timeout=1) as response:
                response.read()
                self.assertEqual(response.headers["X-Cache"], "HIT")
            self.assertLess(time.perf_counter() - request_started, 1.0)
        finally:
            release.set()
            self.store.activity = original_activity  # type: ignore[method-assign]
            slow_thread.join(timeout=4)
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3)
        self.assertFalse(slow_thread.is_alive())
        self.assertEqual(worker_errors, [])

    def test_uncached_query_admission_fails_fast_and_releases(self) -> None:
        gate_settings = Settings(
            **{
                **self.settings.__dict__,
                "query_concurrency": 1,
            }
        )
        httpd = BoundedThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.store, gate_settings),
            max_concurrency=1,
        )
        server_thread = Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        started = Event()
        release = Event()
        worker_errors: list[BaseException] = []
        original_activity = self.store.activity

        def blocking_activity(filters: dict[str, str] | None) -> dict[str, object]:
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("query admission test timed out")
            return original_activity(filters)

        def request_slow_activity() -> None:
            try:
                with urlopen(f"{base_url}/api/activity?admission_holder=1", timeout=4) as response:
                    response.read()
            except BaseException as exc:  # pragma: no cover - asserted below.
                worker_errors.append(exc)

        self.store.activity = blocking_activity  # type: ignore[method-assign]
        slow_thread = Thread(target=request_slow_activity, daemon=True)
        slow_thread.start()
        try:
            self.assertTrue(started.wait(timeout=1), "first query did not acquire the only slot")

            with urlopen(f"{base_url}/api/health", timeout=1) as response:
                self.assertEqual(response.status, 200)
            with urlopen(f"{base_url}/", timeout=1) as response:
                self.assertEqual(response.status, 200)

            rejected_url = f"{base_url}/api/timeline?admission_rejected=1"
            request_started = time.perf_counter()
            with self.assertRaises(HTTPError) as raised:
                urlopen(rejected_url, timeout=1)
            elapsed = time.perf_counter() - request_started
            self.assertEqual(raised.exception.code, 503)
            self.assertLess(elapsed, 1.0)
            self.assertEqual(raised.exception.headers["Retry-After"], "1")
            self.assertEqual(raised.exception.headers["Cache-Control"], "no-store")
            self.assertEqual(raised.exception.headers["X-Cache"], "BYPASS")
            self.assertIn("server is busy", raised.exception.read().decode("utf-8"))
        finally:
            release.set()
            self.store.activity = original_activity  # type: ignore[method-assign]
            slow_thread.join(timeout=4)

        try:
            with urlopen(f"{base_url}/api/timeline?admission_after_release=1", timeout=3) as response:
                self.assertEqual(response.status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3)
        self.assertFalse(slow_thread.is_alive())
        self.assertEqual(worker_errors, [])

    def test_uncached_response_work_is_bounded_and_cache_hits_bypass_it(self) -> None:
        gate_settings = Settings(
            **{
                **self.settings.__dict__,
                "query_concurrency": 1,
            }
        )
        httpd = BoundedThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(self.store, gate_settings),
            max_concurrency=1,
        )
        server_thread = Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{httpd.server_port}"
        cached_url = f"{base_url}/api/timeline?response_gate_cached=1"
        with urlopen(cached_url, timeout=3) as response:
            response.read()
            self.assertEqual(response.headers["X-Cache"], "MISS")

        encode_started = Event()
        release_encode = Event()
        counter_lock = Lock()
        active_encodes = 0
        maximum_active_encodes = 0
        first_timeline_encode = True
        worker_errors: list[BaseException] = []

        from story_map_server import _encode_json_body as original_encode_json_body

        def blocking_encode(payload: object) -> bytes:
            nonlocal active_encodes, maximum_active_encodes, first_timeline_encode
            is_timeline = isinstance(payload, dict) and "buckets" in payload
            if not is_timeline:
                return original_encode_json_body(payload)
            with counter_lock:
                active_encodes += 1
                maximum_active_encodes = max(maximum_active_encodes, active_encodes)
                should_block = first_timeline_encode
                first_timeline_encode = False
            try:
                if should_block:
                    encode_started.set()
                    if not release_encode.wait(timeout=3):
                        raise RuntimeError("response gate test timed out")
                return original_encode_json_body(payload)
            finally:
                with counter_lock:
                    active_encodes -= 1

        def request_response_holder() -> None:
            try:
                with urlopen(
                    f"{base_url}/api/timeline?response_gate_holder=1",
                    timeout=4,
                ) as response:
                    response.read()
            except BaseException as exc:  # pragma: no cover - asserted below.
                worker_errors.append(exc)

        holder = Thread(target=request_response_holder, daemon=True)
        with patch("story_map_server._encode_json_body", side_effect=blocking_encode):
            holder.start()
            try:
                self.assertTrue(
                    encode_started.wait(timeout=1),
                    "first response did not acquire the response-work slot",
                )

                request_started = time.perf_counter()
                with urlopen(cached_url, timeout=1) as response:
                    response.read()
                    self.assertEqual(response.headers["X-Cache"], "HIT")
                self.assertLess(time.perf_counter() - request_started, 1.0)

                rejected_url = f"{base_url}/api/timeline?response_gate_rejected=1"
                request_started = time.perf_counter()
                with self.assertRaises(HTTPError) as raised:
                    urlopen(rejected_url, timeout=1)
                self.assertEqual(raised.exception.code, 503)
                self.assertLess(time.perf_counter() - request_started, 1.0)
                self.assertEqual(raised.exception.headers["Retry-After"], "1")
                self.assertEqual(raised.exception.headers["Cache-Control"], "no-store")
                self.assertEqual(raised.exception.headers["X-Cache"], "BYPASS")
                self.assertIn(
                    "server is busy",
                    raised.exception.read().decode("utf-8"),
                )

                with urlopen(f"{base_url}/api/health", timeout=1) as response:
                    self.assertEqual(response.status, 200)
            finally:
                release_encode.set()
                holder.join(timeout=4)

            with urlopen(
                f"{base_url}/api/timeline?response_gate_after_release=1",
                timeout=3,
            ) as response:
                self.assertEqual(response.status, 200)
        try:
            self.assertFalse(holder.is_alive())
            self.assertEqual(worker_errors, [])
            self.assertEqual(maximum_active_encodes, 1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join(timeout=3)

    def test_zero_gzip_threshold_disables_application_compression(self) -> None:
        cache = ResponseCache(ttl_seconds=60, capacity=1, gzip_min_bytes=0)
        self.assertIsNone(cache.put("payload", b"compressible payload").gzip_body)

    def test_response_cache_is_bounded_by_raw_and_compressed_bytes(self) -> None:
        cache = ResponseCache(
            ttl_seconds=60,
            capacity=10,
            gzip_min_bytes=0,
            max_bytes=8,
        )
        cache.put("first", b"12345678")
        self.assertEqual(cache.size_bytes, 8)
        cache.put("second", b"abcdefgh")
        self.assertIsNone(cache.get("first"))
        self.assertEqual(cache.get("second").body, b"abcdefgh")
        oversized = cache.put("oversized", b"012345678")
        self.assertEqual(oversized.body, b"012345678")
        self.assertIsNone(cache.get("oversized"))
        self.assertLessEqual(cache.size_bytes, cache.max_bytes)

    def test_api_and_static_caches_share_one_process_byte_budget(self) -> None:
        api_bytes, static_bytes = _cache_byte_budgets(512)
        self.assertEqual((api_bytes, static_bytes), (480, 32))
        self.assertEqual(api_bytes + static_bytes, 512)
        self.assertEqual(_cache_byte_budgets(0), (0, 0))


if __name__ == "__main__":
    unittest.main()
