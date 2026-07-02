from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import tempfile
from threading import Event, Thread
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd

from story_map_server import (
    BoundedThreadingHTTPServer,
    ResponseCache,
    Settings,
    StoryMapStore,
    make_handler,
)
from http.server import ThreadingHTTPServer


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
        self.assertEqual(trail["meta"]["persisting_field_count"], 1)
        self.assertEqual(trail["meta"]["departed_field_count"], 1)
        self.assertEqual(trail["meta"]["new_current_field_count"], 1)

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

    def test_zero_gzip_threshold_disables_application_compression(self) -> None:
        cache = ResponseCache(ttl_seconds=60, capacity=1, gzip_min_bytes=0)
        self.assertIsNone(cache.put("payload", b"compressible payload").gzip_body)


if __name__ == "__main__":
    unittest.main()
