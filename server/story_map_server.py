from __future__ import annotations

import json
import logging
import math
import mimetypes
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import duckdb


LOGGER = logging.getLogger("story_map_server")

STORY_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
    "#c2410c",
    "#0f766e",
    "#7c3aed",
    "#b45309",
    "#0369a1",
    "#a21caf",
    "#15803d",
]

MOTIF_FAMILY_TAXONOMY = [
    {"id": "compound", "label": "Compound hazard", "description": "Two or more hazard families."},
    {"id": "heat", "label": "Heat", "description": "Heat or temperature stress."},
    {"id": "wind", "label": "Wind", "description": "Damaging wind or storm exposure."},
    {"id": "drought", "label": "Drought", "description": "Dryness or drought exposure."},
    {"id": "flood", "label": "Flood / ponding", "description": "Flooding, ponding, or excess water."},
    {"id": "none", "label": "No named hazard", "description": "No named hazard family."},
    {"id": "other", "label": "Other hazard", "description": "Hazard not covered by the fallback taxonomy."},
]

PUBLIC_MANIFEST_SECTIONS = {
    "run",
    "event_window_rules",
    "parameters",
    "eligibility",
    "constraints",
    "map_geometry",
}
MANIFEST_PATH_KEYS = {
    "path",
    "paths",
    "output_dir",
    "input_parquet",
    "temp_dir",
    "field_geometry_parquet",
}

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "cross_crop_story_clustering"
    / "vm_one_parquet"
    / "output"
    / "event_map_sample_50"
)


class RequestValidationError(ValueError):
    """A malformed public request that is safe to report as HTTP 400."""


class ServerBusyError(RuntimeError):
    """The bounded query executor has no immediately available capacity."""


@dataclass(frozen=True)
class Settings:
    run_dir: Path
    static_dir: Path
    host: str
    port: int
    raster_tiles: str
    raster_attribution: str
    default_feature_limit: int
    max_feature_limit: int
    log_level: str
    cache_seconds: float = 300.0
    cache_entries: int = 256
    gzip_min_bytes: int = 1024
    query_concurrency: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file(SERVER_DIR / ".env")
        run_dir = resolve_portable_path(os.getenv("STORY_MAP_RUN_DIR", str(DEFAULT_RUN_DIR)))
        static_dir = resolve_portable_path(
            os.getenv("STORY_MAP_STATIC_DIR", str(SERVER_DIR / "static")),
            prefer_server_dir=True,
        )
        return cls(
            run_dir=run_dir,
            static_dir=static_dir,
            host=os.getenv("STORY_MAP_HOST", "127.0.0.1"),
            port=int(os.getenv("STORY_MAP_PORT", "8877")),
            raster_tiles=os.getenv(
                "STORY_MAP_RASTER_TILES",
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ),
            raster_attribution=os.getenv(
                "STORY_MAP_RASTER_ATTRIBUTION",
                "Tiles (C) Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            ),
            default_feature_limit=int(os.getenv("STORY_MAP_DEFAULT_FEATURE_LIMIT", "5000")),
            max_feature_limit=int(os.getenv("STORY_MAP_MAX_FEATURE_LIMIT", "20000")),
            log_level=os.getenv("STORY_MAP_LOG_LEVEL", "INFO"),
            cache_seconds=float(os.getenv("STORY_MAP_CACHE_SECONDS", "300")),
            cache_entries=int(os.getenv("STORY_MAP_CACHE_ENTRIES", "256")),
            gzip_min_bytes=int(os.getenv("STORY_MAP_GZIP_MIN_BYTES", "1024")),
            query_concurrency=int(os.getenv("STORY_MAP_QUERY_CONCURRENCY", "8")),
        )


@dataclass(frozen=True)
class CachedBody:
    body: bytes
    gzip_body: bytes | None


class ResponseCache:
    """Small process-local TTL/LRU cache for immutable run artifacts."""

    def __init__(self, *, ttl_seconds: float, capacity: int, gzip_min_bytes: int) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.capacity = max(0, int(capacity))
        self.gzip_min_bytes = max(0, int(gzip_min_bytes))
        self._items: OrderedDict[str, tuple[float, CachedBody]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> CachedBody | None:
        if not key or self.capacity == 0 or self.ttl_seconds == 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, cached = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return cached

    def put(self, key: str, body: bytes) -> CachedBody:
        gzip_body = (
            gzip.compress(body, compresslevel=5)
            if self.gzip_min_bytes > 0 and len(body) >= self.gzip_min_bytes
            else None
        )
        cached = CachedBody(body=body, gzip_body=gzip_body)
        if not key or self.capacity == 0 or self.ttl_seconds == 0:
            return cached
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._items[key] = (expires_at, cached)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return cached


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded transport; query work is bounded inside the request handler."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: type, *, max_concurrency: int) -> None:
        self.max_query_concurrency = max(1, int(max_concurrency))
        super().__init__(server_address, handler)


class StoryMapStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.run_dir = settings.run_dir
        self.geometry_path = self._pick("field_geometry.parquet", "map_field_geometry.parquet")
        self.frame_path = self._pick("frame_fields.parquet", "map_frame_fields.parquet")
        self.labels_path = self._pick("cluster_labels.parquet", "event_story_cluster_labels.parquet")
        self.events_path = self._pick("event_windows.parquet")
        self.story_days_path = self._pick("story_day_membership.parquet")
        self.manifest_path = self._pick("manifest.json")
        self.timeline_summary_path = self.run_dir / "gpu_summaries" / "timeline_summary.parquet"
        self.frame_columns = self._parquet_columns(self.frame_path)
        self.timeline_summary_columns = self._parquet_columns(self.timeline_summary_path)
        self.timeline_summary_mapping = self._timeline_summary_mapping()
        LOGGER.info(
            "story_map_store run_dir=%s optimized_geometry=%s frame_columns=%s timeline_summary=%s",
            self.run_dir,
            self._has_optimized_geometry(),
            ",".join(sorted(self.frame_columns)),
            bool(self.timeline_summary_mapping),
        )
        for name, path in self._artifact_paths().items():
            LOGGER.info(
                "story_map_artifact name=%s path=%s exists=%s size_bytes=%s",
                name,
                path,
                path.exists(),
                _file_size(path),
            )

    def _pick(self, *names: str) -> Path:
        for name in names:
            path = self.run_dir / name
            if path.exists():
                return path
        return self.run_dir / names[0]

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            "run_dir": self.run_dir,
            "geometry": self.geometry_path,
            "frames": self.frame_path,
            "labels": self.labels_path,
            "events": self.events_path,
            "story_days": self.story_days_path,
            "manifest": self.manifest_path,
        }

    def _parquet_columns(self, path: Path) -> frozenset[str]:
        if not path.exists():
            return frozenset()
        try:
            with duckdb.connect(":memory:") as con:
                cursor = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
                return frozenset(str(item[0]) for item in (cursor.description or []))
        except Exception:
            LOGGER.exception("parquet_schema_failed path=%s", path)
            return frozenset()

    def _timeline_summary_mapping(self) -> dict[str, str] | None:
        columns = self.timeline_summary_columns
        required = {"timeline_bucket", "reportable_day_count", "event_count", "max_risk_rank"}
        if not required.issubset(columns):
            return None
        field_count = "field_count" if "field_count" in columns else "field_id" if "field_id" in columns else None
        story_count = (
            "story_cluster_count"
            if "story_cluster_count" in columns
            else "story_cluster_id"
            if "story_cluster_id" in columns
            else None
        )
        if field_count is None or story_count is None:
            return None
        return {"field_count": field_count, "story_cluster_count": story_count}

    def _motif_family_sql(self, alias: str) -> str:
        if "motif_family" in self.frame_columns:
            fallback = _hazard_family_sql(f"{alias}.hazard_signature")
            return f"COALESCE(NULLIF(TRIM(CAST({alias}.motif_family AS VARCHAR)), ''), {fallback})"
        return _hazard_family_sql(f"{alias}.hazard_signature")

    def motif_taxonomy(self) -> dict[str, Any]:
        return {
            "field": "motif_family",
            "source": "frame_fields.motif_family" if "motif_family" in self.frame_columns else "hazard_signature_fallback",
            "fallback_version": "hazard_family_v1",
            "families": MOTIF_FAMILY_TAXONOMY,
        }

    def health(self) -> dict[str, Any]:
        paths = self._artifact_paths()
        checks = {name: path.exists() for name, path in paths.items()}
        return {
            "ok": all(checks.values()),
            "checks": checks,
        }

    def require_ready(self) -> None:
        health = self.health()
        missing = [name for name, exists in health["checks"].items() if not exists]
        if missing:
            missing_paths = {
                name: str(path)
                for name, path in self._artifact_paths().items()
                if name in missing
            }
            LOGGER.error(
                "story_map_not_ready missing=%s paths=%s",
                ",".join(missing),
                _json_for_log(missing_paths),
            )
            raise ValueError(f"Story map run directory is missing required files: {', '.join(missing)}")

    def config(self) -> dict[str, Any]:
        return {
            "raster": {
                "tiles": [self.settings.raster_tiles],
                "tileSize": 256,
                "attribution": self.settings.raster_attribution,
            },
            "limits": {
                "defaultFeatureLimit": self.settings.default_feature_limit,
                "maxFeatureLimit": self.settings.max_feature_limit,
            },
        }

    def manifest(self) -> dict[str, Any]:
        self.require_ready()
        manifest = _public_manifest(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )
        bounds = self.bounds()
        manifest["server"] = {
            "bounds": bounds,
            "optimized_geometry": self._has_optimized_geometry(),
            "story_palette": self.story_palette(manifest),
        }
        LOGGER.info(
            "manifest_loaded run_dir=%s optimized_geometry=%s bounds=%s",
            self.run_dir,
            self._has_optimized_geometry(),
            _json_for_log(bounds),
        )
        return manifest

    def story_palette(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        self.require_ready()
        if manifest is None:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        parameters = manifest.get("parameters", {}) if isinstance(manifest, dict) else {}
        top_clusters = int(parameters.get("map_top_clusters") or 12)
        top_scope = str(parameters.get("map_top_scope") or "global")
        color_by = str(parameters.get("map_color_by") or "story_cluster")
        top_clusters = max(1, top_clusters)
        label_join = ""
        label_select = "NULL AS short_label"
        params: list[Any] = [str(self.frame_path)]
        if self.labels_path.exists():
            label_join = "LEFT JOIN read_parquet(?) AS l USING (story_cluster_id)"
            label_select = "MAX(l.short_label) AS short_label"
            params.append(str(self.labels_path))
        else:
            label_select = "NULL AS short_label"
        params.append(top_clusters)

        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                f"""
                SELECT
                    f.story_cluster_id,
                    SUM(f.reportable_day_count) AS total_reportable_day_count,
                    SUM(f.event_count) AS total_event_count,
                    MAX(f.max_risk_rank) AS max_risk_rank,
                    {label_select}
                FROM read_parquet(?) AS f
                {label_join}
                GROUP BY f.story_cluster_id
                ORDER BY
                    total_reportable_day_count DESC,
                    total_event_count DESC,
                    max_risk_rank DESC,
                    f.story_cluster_id
                LIMIT ?
                """,
                params,
            ).fetchdf()

        clusters = []
        for index, row in enumerate(_records(rows)):
            clusters.append(
                {
                    "story_cluster_id": row.get("story_cluster_id"),
                    "short_label": row.get("short_label"),
                    "total_reportable_day_count": row.get("total_reportable_day_count"),
                    "total_event_count": row.get("total_event_count"),
                    "color": STORY_PALETTE[index % len(STORY_PALETTE)],
                }
            )
        return {
            "color_by": color_by,
            "top_scope": top_scope,
            "top_clusters": top_clusters,
            "palette": STORY_PALETTE,
            "other_color": "#94a3b8",
            "clusters": clusters,
        }

    def bounds(self) -> dict[str, float]:
        return _bounds_for_geometry(str(self.geometry_path), self._has_optimized_geometry())

    def timeline(self) -> dict[str, Any]:
        self.require_ready()
        started = time.perf_counter()
        source = "frame_fields"
        with duckdb.connect(":memory:") as con:
            if self.timeline_summary_mapping:
                source = "gpu_summary"
                field_count = self.timeline_summary_mapping["field_count"]
                story_count = self.timeline_summary_mapping["story_cluster_count"]
                rows = con.execute(
                    f"""
                    SELECT
                        timeline_bucket,
                        {field_count} AS field_count,
                        {story_count} AS story_cluster_count,
                        reportable_day_count,
                        event_count,
                        max_risk_rank
                    FROM read_parquet(?)
                    ORDER BY timeline_bucket
                    """,
                    [str(self.timeline_summary_path)],
                ).fetchdf()
            else:
                rows = con.execute(
                    """
                    SELECT
                        timeline_bucket,
                        COUNT(DISTINCT field_id) AS field_count,
                        COUNT(DISTINCT story_cluster_id) AS story_cluster_count,
                        SUM(reportable_day_count) AS reportable_day_count,
                        SUM(event_count) AS event_count,
                        MAX(max_risk_rank) AS max_risk_rank
                    FROM read_parquet(?)
                    GROUP BY timeline_bucket
                    ORDER BY timeline_bucket
                    """,
                    [str(self.frame_path)],
                ).fetchdf()
        buckets = _records(rows)
        LOGGER.info(
            "timeline_loaded source=%s buckets=%s first_bucket=%s last_bucket=%s elapsed_ms=%.1f",
            source,
            len(buckets),
            buckets[0]["timeline_bucket"] if buckets else None,
            buckets[-1]["timeline_bucket"] if buckets else None,
            (time.perf_counter() - started) * 1000,
        )
        if not buckets:
            LOGGER.warning("timeline_empty frame_path=%s", self.frame_path)
        return {"buckets": buckets, "source": source}

    def frame_features(
        self,
        *,
        timeline_bucket: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.require_ready()
        started = time.perf_counter()
        requested_limit = limit
        effective_limit = _feature_limit(limit, self.settings.max_feature_limit)
        filters = _clean_filters(filters)
        optimized = self._has_optimized_geometry()
        python_bbox_filter = bool(bbox and not optimized)
        bbox_clause = ""
        motif_family_sql = self._motif_family_sql("f")
        filter_clause, filter_params = _filter_sql(filters, "f", motif_family_sql=motif_family_sql)
        if optimized and bbox:
            bbox_clause = """
              AND NOT (
                g.max_lon < ? OR g.min_lon > ? OR g.max_lat < ? OR g.min_lat > ?
              )
            """

        label_join = ""
        label_select = "NULL AS short_label"
        if self.labels_path.exists():
            label_join = "LEFT JOIN read_parquet(?) AS l USING (story_cluster_id)"
            label_select = "l.short_label"

        geometry_select = (
            "g.geometry_geojson, 'geojson' AS geometry_format, "
            "g.min_lon, g.min_lat, g.max_lon, g.max_lat, g.centroid_lon, g.centroid_lat"
            if optimized
            else "g.geometry_text, g.geometry_format, NULL AS min_lon, NULL AS min_lat, "
            "NULL AS max_lon, NULL AS max_lat, NULL AS centroid_lon, NULL AS centroid_lat"
        )

        sql = f"""
            SELECT
                f.timeline_bucket,
                f.field_id,
                f.story_cluster_id,
                f.max_risk_band,
                f.hazard_signature,
                f.response_signature,
                f.reportable_day_count,
                f.event_count,
                f.max_risk_rank,
                f.response_day_count,
                {motif_family_sql} AS motif_family,
                {label_select},
                g.district,
                g.sector,
                g.cell,
                g.village,
                {geometry_select},
                COUNT(*) OVER () AS _source_row_count,
                COUNT(DISTINCT f.story_cluster_id) OVER () AS _source_story_cluster_count,
                SUM(f.reportable_day_count) OVER () AS _source_reportable_day_count,
                SUM(f.event_count) OVER () AS _source_event_count
            FROM read_parquet(?) AS f
            JOIN read_parquet(?) AS g USING (field_id)
            {label_join}
            WHERE f.timeline_bucket = ?
            {filter_clause}
            {bbox_clause}
            ORDER BY f.max_risk_rank DESC, f.reportable_day_count DESC, f.field_id
        """
        params: list[Any] = [str(self.frame_path), str(self.geometry_path)]
        if self.labels_path.exists():
            params.append(str(self.labels_path))
        params.append(timeline_bucket)
        params.extend(filter_params)
        if optimized and bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            params.extend([min_lon, max_lon, min_lat, max_lat])
        if effective_limit is not None and not python_bbox_filter:
            sql += "\nLIMIT ?"
            params.append(effective_limit + 1)

        with duckdb.connect(":memory:") as con:
            rows = con.execute(sql, params).fetchdf()

        features = []
        parse_failures = 0
        bbox_filtered = 0
        parse_failure_samples: list[dict[str, Any]] = []
        sample_features: list[dict[str, Any]] = []
        source_rows = _records(rows)
        source_row_count = int(source_rows[0].get("_source_row_count") or 0) if source_rows else 0
        source_story_cluster_count = (
            int(source_rows[0].get("_source_story_cluster_count") or 0) if source_rows else 0
        )
        source_reportable_day_count = (
            float(source_rows[0].get("_source_reportable_day_count") or 0) if source_rows else 0
        )
        source_event_count = float(source_rows[0].get("_source_event_count") or 0) if source_rows else 0
        viewport_source_row_count = 0
        viewport_story_clusters: set[str] = set()
        viewport_reportable_day_count = 0.0
        viewport_event_count = 0.0
        truncated = False
        for row in source_rows:
            try:
                geometry, geom_bbox = _geometry_to_geojson_and_bbox(
                    row.get("geometry_geojson") or row.get("geometry_text"),
                    str(row.get("geometry_format") or "geojson"),
                )
            except Exception as exc:
                parse_failures += 1
                if len(parse_failure_samples) < 3:
                    parse_failure_samples.append(
                        {
                            "field_id": row.get("field_id"),
                            "geometry_format": row.get("geometry_format"),
                            "error": str(exc)[:240],
                        }
                    )
                continue
            if bbox and not _intersects(geom_bbox, bbox):
                bbox_filtered += 1
                continue
            if python_bbox_filter:
                viewport_source_row_count += 1
                if row.get("story_cluster_id") is not None:
                    viewport_story_clusters.add(str(row["story_cluster_id"]))
                viewport_reportable_day_count += float(row.get("reportable_day_count") or 0)
                viewport_event_count += float(row.get("event_count") or 0)
            if effective_limit is not None and len(features) >= effective_limit:
                truncated = True
                if python_bbox_filter:
                    continue
                break
            properties = {
                key: row.get(key)
                for key in [
                    "timeline_bucket",
                    "field_id",
                    "story_cluster_id",
                    "max_risk_band",
                    "hazard_signature",
                    "response_signature",
                    "reportable_day_count",
                    "event_count",
                    "max_risk_rank",
                    "response_day_count",
                    "motif_family",
                    "short_label",
                    "district",
                    "sector",
                    "cell",
                    "village",
                ]
            }
            properties["bbox"] = geom_bbox
            features.append({"type": "Feature", "geometry": geometry, "properties": properties})
            if len(sample_features) < 3:
                sample_features.append(
                    {
                        "field_id": row.get("field_id"),
                        "story_cluster_id": row.get("story_cluster_id"),
                        "max_risk_band": row.get("max_risk_band"),
                        "bbox": geom_bbox,
                    }
                )
        if python_bbox_filter:
            source_row_count = viewport_source_row_count
            source_story_cluster_count = len(viewport_story_clusters)
            source_reportable_day_count = viewport_reportable_day_count
            source_event_count = viewport_event_count
            truncated = effective_limit is not None and source_row_count > effective_limit
        elif effective_limit is not None:
            truncated = truncated or source_row_count > effective_limit
        limit_hit = truncated
        diagnostics = None
        if parse_failure_samples:
            LOGGER.warning(
                "frame_geometry_parse_failures bucket=%s samples=%s",
                timeline_bucket,
                _json_for_log(parse_failure_samples),
            )
        if not features:
            diagnostics = self._diagnose_empty_frame(timeline_bucket, bbox, optimized, filters)
            LOGGER.warning(
                "frame_empty bucket=%s filters=%s bbox=%s optimized_geometry=%s diagnostics=%s",
                timeline_bucket,
                _json_for_log(filters),
                _json_for_log(bbox),
                optimized,
                _json_for_log(diagnostics),
            )
        LOGGER.info(
            (
                "frame_features bucket=%s bbox=%s optimized_geometry=%s "
                "filters=%s requested_limit=%s effective_limit=%s source_rows=%s rendered=%s "
                "bbox_filtered=%s parse_failures=%s limit_hit=%s elapsed_ms=%.1f samples=%s"
            ),
            timeline_bucket,
            _json_for_log(bbox),
            optimized,
            _json_for_log(filters),
            requested_limit,
            effective_limit,
            source_row_count,
            len(features),
            bbox_filtered,
            parse_failures,
            limit_hit,
            (time.perf_counter() - started) * 1000,
            _json_for_log(sample_features),
        )

        meta = {
            "timeline_bucket": timeline_bucket,
            "feature_count": len(features),
            "source_row_count": source_row_count,
            "query_row_count": len(source_rows),
            "bbox_filtered_count": bbox_filtered,
            "parse_failures": parse_failures,
            "story_cluster_count": source_story_cluster_count,
            "reportable_day_count": source_reportable_day_count,
            "event_count": source_event_count,
            "limit": effective_limit,
            "requested_limit": requested_limit,
            "unlimited": effective_limit is None,
            "limit_hit": limit_hit,
            "truncated": truncated,
            "optimized_geometry": optimized,
            "bbox": bbox,
            "filters": filters,
        }
        if diagnostics is not None:
            meta["diagnostics"] = diagnostics
        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": meta,
        }

    def motifs(self, q: str | None, limit: int) -> dict[str, Any]:
        self.require_ready()
        limit = max(1, min(limit, 1000))
        frame_family_sql = self._motif_family_sql("f")
        label_fallback_sql = _hazard_family_sql("l.hazard_signature")
        enriched_cte = f"""
            WITH frame_families AS (
                SELECT f.story_cluster_id, MIN({frame_family_sql}) AS motif_family
                FROM read_parquet(?) AS f
                GROUP BY f.story_cluster_id
            ),
            labels_enriched AS (
                SELECT
                    l.story_cluster_id,
                    l.short_label,
                    l.max_risk_band,
                    l.hazard_signature,
                    l.response_signature,
                    l.event_count,
                    l.field_count,
                    l.crop_count,
                    l.median_window_span_days,
                    l.median_reportable_days,
                    COALESCE(ff.motif_family, {label_fallback_sql}) AS motif_family
                FROM read_parquet(?) AS l
                LEFT JOIN frame_families AS ff USING (story_cluster_id)
                WHERE ff.story_cluster_id IS NOT NULL
            )
        """
        params: list[Any] = [str(self.frame_path), str(self.labels_path)]
        where = ""
        if q:
            where = """
                WHERE CAST(story_cluster_id AS VARCHAR) ILIKE ?
                   OR short_label ILIKE ?
                   OR hazard_signature ILIKE ?
                   OR response_signature ILIKE ?
                   OR motif_family ILIKE ?
            """
            needle = f"%{q}%"
            params.extend([needle, needle, needle, needle, needle])
        params.append(limit)
        with duckdb.connect(":memory:") as con:
            motifs = con.execute(
                f"""
                {enriched_cte}
                SELECT
                    story_cluster_id,
                    short_label,
                    max_risk_band,
                    hazard_signature,
                    response_signature,
                    motif_family,
                    event_count,
                    field_count,
                    crop_count,
                    median_window_span_days,
                    median_reportable_days
                FROM labels_enriched
                {where}
                ORDER BY event_count DESC, field_count DESC, story_cluster_id
                LIMIT ?
                """,
                params,
            ).fetchdf()
            risks = con.execute(
                """
                SELECT max_risk_band, SUM(event_count) AS event_count
                FROM read_parquet(?)
                WHERE max_risk_band IS NOT NULL
                GROUP BY max_risk_band
                ORDER BY event_count DESC, max_risk_band
                """,
                [str(self.labels_path)],
            ).fetchdf()
            hazards = con.execute(
                """
                SELECT hazard_signature, SUM(event_count) AS event_count
                FROM read_parquet(?)
                WHERE hazard_signature IS NOT NULL
                GROUP BY hazard_signature
                ORDER BY event_count DESC, hazard_signature
                LIMIT 200
                """,
                [str(self.labels_path)],
            ).fetchdf()
            responses = con.execute(
                """
                SELECT response_signature, SUM(event_count) AS event_count
                FROM read_parquet(?)
                WHERE response_signature IS NOT NULL
                GROUP BY response_signature
                ORDER BY event_count DESC, response_signature
                LIMIT 200
                """,
                [str(self.labels_path)],
            ).fetchdf()
            motif_families = con.execute(
                f"""
                {enriched_cte}
                SELECT motif_family, SUM(event_count) AS event_count
                FROM labels_enriched
                WHERE motif_family IS NOT NULL
                GROUP BY motif_family
                ORDER BY event_count DESC, motif_family
                """,
                [str(self.frame_path), str(self.labels_path)],
            ).fetchdf()
        motif_records = _records(motifs)
        LOGGER.info("motifs_loaded query=%s limit=%s motifs=%s", q, limit, len(motif_records))
        return {
            "motifs": motif_records,
            "exact_stories": motif_records,
            "facets": {
                "max_risk_band": _records(risks),
                "hazard_signature": _records(hazards),
                "response_signature": _records(responses),
                "motif_family": _records(motif_families),
            },
            "taxonomy": self.motif_taxonomy(),
        }

    def activity(self, filters: dict[str, str] | None) -> dict[str, Any]:
        self.require_ready()
        filters = _clean_filters(filters)
        motif_family_sql = self._motif_family_sql("f")
        filter_clause, filter_params = _filter_sql(filters, "f", motif_family_sql=motif_family_sql)
        started = time.perf_counter()
        params: list[Any] = [str(self.frame_path)]
        params.extend(filter_params)
        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                f"""
                SELECT
                    f.timeline_bucket,
                    COUNT(DISTINCT f.field_id) AS field_count,
                    COUNT(DISTINCT f.story_cluster_id) AS story_cluster_count,
                    COUNT(DISTINCT {motif_family_sql}) AS motif_family_count,
                    SUM(f.reportable_day_count) AS reportable_day_count,
                    SUM(f.event_count) AS event_count,
                    MAX(f.max_risk_rank) AS max_risk_rank
                FROM read_parquet(?) AS f
                WHERE 1 = 1
                {filter_clause}
                GROUP BY f.timeline_bucket
                ORDER BY f.timeline_bucket
                """,
                params,
            ).fetchdf()
        buckets = _records(rows)
        LOGGER.info(
            "activity_loaded filters=%s buckets=%s elapsed_ms=%.1f",
            _json_for_log(filters),
            len(buckets),
            (time.perf_counter() - started) * 1000,
        )
        return {
            "filters": filters,
            "bucket_count": len(buckets),
            "buckets": buckets,
        }

    def trajectory(self, filters: dict[str, str] | None) -> dict[str, Any]:
        result = self.activity(filters)
        result["deprecated"] = {
            "replacement": "/api/activity",
            "reason": "Spatial representative-field movement was removed; buckets are activity aggregates.",
        }
        return result

    def trail_features(
        self,
        *,
        timeline_bucket: str,
        filters: dict[str, str] | None,
        lookback: int,
        bbox: tuple[float, float, float, float] | None = None,
        limit: int = 0,
    ) -> dict[str, Any]:
        self.require_ready()
        filters = _clean_filters(filters)
        requested_limit = limit
        effective_limit = _feature_limit(limit, self.settings.max_feature_limit)
        if not filters:
            return {
                "type": "FeatureCollection",
                "features": [],
                "meta": {
                    "timeline_bucket": timeline_bucket,
                    "filters": filters,
                    "bbox": bbox,
                    "bucket_count": 0,
                    "feature_count": 0,
                    "current_field_count": 0,
                    "prior_field_count": 0,
                    "persisting_field_count": 0,
                    "departed_field_count": 0,
                    "new_current_field_count": 0,
                    "transition_counts_available": False,
                    "limit": effective_limit,
                    "requested_limit": requested_limit,
                    "truncated": False,
                },
            }
        lookback = max(0, min(lookback, 24))
        optimized = self._has_optimized_geometry()
        if not optimized:
            raise ValueError("Trail requires optimized field_geometry.parquet.")
        motif_family_sql = self._motif_family_sql("f")
        filter_clause, filter_params = _filter_sql(filters, "f", motif_family_sql=motif_family_sql)
        bbox_clause = ""
        bbox_params: list[Any] = []
        if bbox:
            bbox_clause = """
              AND NOT (
                g.max_lon < ? OR g.min_lon > ? OR g.max_lat < ? OR g.min_lat > ?
              )
            """
            min_lon, min_lat, max_lon, max_lat = bbox
            bbox_params = [min_lon, max_lon, min_lat, max_lat]

        label_join = ""
        label_select = "NULL AS short_label"
        if self.labels_path.exists():
            label_join = "LEFT JOIN read_parquet(?) AS l USING (story_cluster_id)"
            label_select = "l.short_label"

        sql = f"""
            WITH all_buckets AS (
                SELECT
                    timeline_bucket,
                    ROW_NUMBER() OVER (ORDER BY timeline_bucket) - 1 AS bucket_index
                FROM (
                    SELECT DISTINCT timeline_bucket
                    FROM read_parquet(?)
                )
            ),
            target_bucket AS (
                SELECT timeline_bucket, bucket_index
                FROM all_buckets
                WHERE timeline_bucket <= ?
                ORDER BY timeline_bucket DESC
                LIMIT 1
            ),
            selected_prior_buckets AS (
                SELECT
                    a.timeline_bucket,
                    a.bucket_index,
                    t.bucket_index - a.bucket_index AS age_index
                FROM all_buckets AS a
                CROSS JOIN target_bucket AS t
                WHERE a.bucket_index BETWEEN t.bucket_index - ? AND t.bucket_index - 1
            ),
            current_fields AS (
                SELECT DISTINCT f.field_id
                FROM target_bucket AS t
                JOIN read_parquet(?) AS f USING (timeline_bucket)
                JOIN read_parquet(?) AS g USING (field_id)
                WHERE 1 = 1
                {filter_clause}
                {bbox_clause}
            ),
            prior_ranked AS (
                SELECT
                    s.bucket_index,
                    s.age_index,
                    f.timeline_bucket,
                    f.field_id,
                    f.story_cluster_id,
                    f.max_risk_band,
                    f.hazard_signature,
                    f.response_signature,
                    f.reportable_day_count,
                    f.event_count,
                    f.max_risk_rank,
                    f.response_day_count,
                    {motif_family_sql} AS motif_family,
                    {label_select},
                    g.district,
                    g.sector,
                    g.cell,
                    g.village,
                    g.geometry_geojson,
                    'geojson' AS geometry_format,
                    g.min_lon,
                    g.min_lat,
                    g.max_lon,
                    g.max_lat,
                    g.centroid_lon,
                    g.centroid_lat,
                    c.field_id IS NOT NULL AS persists_to_current,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.field_id
                        ORDER BY s.bucket_index DESC, f.max_risk_rank DESC, f.reportable_day_count DESC
                    ) AS prior_rank
                FROM selected_prior_buckets AS s
                JOIN read_parquet(?) AS f USING (timeline_bucket)
                JOIN read_parquet(?) AS g USING (field_id)
                {label_join}
                LEFT JOIN current_fields AS c USING (field_id)
                WHERE 1 = 1
                {filter_clause}
                {bbox_clause}
            ),
            prior_features AS (
                SELECT * EXCLUDE (prior_rank)
                FROM prior_ranked
                WHERE prior_rank = 1
            ),
            stats AS (
                SELECT
                    (SELECT timeline_bucket FROM target_bucket) AS resolved_timeline_bucket,
                    (SELECT COUNT(*) FROM current_fields) AS current_field_count,
                    (SELECT COUNT(*) FROM prior_features) AS prior_field_count,
                    (SELECT COUNT(*) FROM prior_features WHERE persists_to_current) AS persisting_field_count
            )
            SELECT p.*, s.*
            FROM stats AS s
            LEFT JOIN prior_features AS p ON TRUE
            ORDER BY p.age_index DESC, p.max_risk_rank DESC, p.reportable_day_count DESC, p.field_id
        """
        if effective_limit is not None:
            sql += "\nLIMIT ?"
        params: list[Any] = [str(self.frame_path), timeline_bucket, lookback]
        params.extend([str(self.frame_path), str(self.geometry_path), *filter_params, *bbox_params])
        params.extend([str(self.frame_path), str(self.geometry_path)])
        if self.labels_path.exists():
            params.append(str(self.labels_path))
        params.extend([*filter_params, *bbox_params])
        if effective_limit is not None:
            params.append(effective_limit + 1)

        started = time.perf_counter()
        with duckdb.connect(":memory:") as con:
            rows = con.execute(sql, params).fetchdf()

        features = []
        parse_failures = 0
        bucket_names = set()
        records = _records(rows)
        stats_row = records[0] if records else {}
        truncated = (
            effective_limit is not None
            and int(stats_row.get("prior_field_count") or 0) > effective_limit
        )
        render_records = records[:effective_limit] if effective_limit is not None else records
        for row in render_records:
            if row.get("field_id") is None:
                continue
            try:
                geometry, geom_bbox = _geometry_to_geojson_and_bbox(
                    row.get("geometry_geojson"),
                    str(row.get("geometry_format") or "geojson"),
                )
            except Exception:
                parse_failures += 1
                continue
            bucket_names.add(row.get("timeline_bucket"))
            properties = {
                key: row.get(key)
                for key in [
                    "bucket_index",
                    "age_index",
                    "timeline_bucket",
                    "field_id",
                    "story_cluster_id",
                    "max_risk_band",
                    "hazard_signature",
                    "response_signature",
                    "reportable_day_count",
                    "event_count",
                    "max_risk_rank",
                    "response_day_count",
                    "motif_family",
                    "short_label",
                    "district",
                    "sector",
                    "cell",
                    "village",
                    "persists_to_current",
                ]
            }
            properties["bbox"] = geom_bbox
            features.append({"type": "Feature", "geometry": geometry, "properties": properties})

        LOGGER.info(
            "trail_features bucket=%s filters=%s bbox=%s lookback=%s buckets=%s features=%s parse_failures=%s elapsed_ms=%.1f",
            timeline_bucket,
            _json_for_log(filters),
            _json_for_log(bbox),
            lookback,
            len(bucket_names),
            len(features),
            parse_failures,
            (time.perf_counter() - started) * 1000,
        )
        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "timeline_bucket": timeline_bucket,
                "resolved_timeline_bucket": stats_row.get("resolved_timeline_bucket"),
                "filters": filters,
                "bbox": bbox,
                "lookback": lookback,
                "bucket_count": len(bucket_names),
                "feature_count": len(features),
                "parse_failures": parse_failures,
                "current_field_count": int(stats_row.get("current_field_count") or 0),
                "prior_field_count": int(stats_row.get("prior_field_count") or 0),
                "persisting_field_count": int(stats_row.get("persisting_field_count") or 0),
                "departed_field_count": max(
                    0,
                    int(stats_row.get("prior_field_count") or 0)
                    - int(stats_row.get("persisting_field_count") or 0),
                ),
                "new_current_field_count": max(
                    0,
                    int(stats_row.get("current_field_count") or 0)
                    - int(stats_row.get("persisting_field_count") or 0),
                ),
                "transition_counts_available": True,
                "limit": effective_limit,
                "requested_limit": requested_limit,
                "truncated": truncated,
            },
        }

    def field_events(self, field_id: str, limit: int) -> dict[str, Any]:
        self.require_ready()
        limit = max(1, min(limit, 500))
        with duckdb.connect(":memory:") as con:
            events = con.execute(
                """
                SELECT *
                FROM read_parquet(?)
                WHERE field_id = ?
                ORDER BY event_start_date DESC, active_end_date DESC
                LIMIT ?
                """,
                [str(self.events_path), field_id, limit],
            ).fetchdf()
        records = _records(events)
        LOGGER.info("field_events field_id=%s limit=%s events=%s", field_id, limit, len(records))
        return {"field_id": field_id, "events": records}

    def cluster(self, story_cluster_id: str, limit: int) -> dict[str, Any]:
        self.require_ready()
        limit = max(1, min(limit, 500))
        with duckdb.connect(":memory:") as con:
            label = con.execute(
                """
                SELECT *
                FROM read_parquet(?)
                WHERE story_cluster_id = ?
                LIMIT 1
                """,
                [str(self.labels_path), story_cluster_id],
            ).fetchdf()
            events = con.execute(
                """
                SELECT
                    field_id,
                    crop_name,
                    crop_season,
                    event_id,
                    event_start_date,
                    active_end_date,
                    max_risk_band,
                    hazard_signature,
                    stage_signature,
                    response_signature,
                    close_reason,
                    reportable_days,
                    window_span_days
                FROM read_parquet(?)
                WHERE story_cluster_id = ?
                ORDER BY event_start_date, field_id
                LIMIT ?
                """,
                [str(self.events_path), story_cluster_id, limit],
            ).fetchdf()
        label_records = _records(label)
        event_records = _records(events)
        LOGGER.info(
            "cluster_loaded story_cluster_id=%s limit=%s label_found=%s events=%s",
            story_cluster_id,
            limit,
            bool(label_records),
            len(event_records),
        )
        return {
            "story_cluster_id": story_cluster_id,
            "label": label_records[0] if label_records else None,
            "events": event_records,
        }

    def field_search(self, q: str, limit: int) -> dict[str, Any]:
        self.require_ready()
        limit = max(1, min(limit, 50))
        needle = f"%{q}%"
        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                """
                SELECT DISTINCT field_id
                FROM read_parquet(?)
                WHERE field_id ILIKE ?
                ORDER BY field_id
                LIMIT ?
                """,
                [str(self.geometry_path), needle, limit],
            ).fetchdf()
        records = _records(rows)
        LOGGER.info("field_search query=%s limit=%s matches=%s", q, limit, len(records))
        return {"fields": records}

    def _has_optimized_geometry(self) -> bool:
        return self.geometry_path.name == "field_geometry.parquet"

    def _diagnose_empty_frame(
        self,
        timeline_bucket: str,
        bbox: tuple[float, float, float, float] | None,
        optimized: bool,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        filters = _clean_filters(filters)
        filter_clause, filter_params = _filter_sql(
            filters,
            "f",
            motif_family_sql=self._motif_family_sql("f"),
        )
        try:
            with duckdb.connect(":memory:") as con:
                diagnostics["frame_rows_for_bucket"] = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet(?) AS f
                    WHERE timeline_bucket = ?
                    {filter_clause}
                    """,
                    [str(self.frame_path), timeline_bucket, *filter_params],
                ).fetchone()[0]
                diagnostics["geometry_rows"] = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [str(self.geometry_path)],
                ).fetchone()[0]
                diagnostics["joined_rows_without_bbox"] = con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM read_parquet(?) AS f
                    JOIN read_parquet(?) AS g USING (field_id)
                    WHERE f.timeline_bucket = ?
                    {filter_clause}
                    """,
                    [str(self.frame_path), str(self.geometry_path), timeline_bucket, *filter_params],
                ).fetchone()[0]
                diagnostics["frame_field_samples"] = [
                    row[0]
                    for row in con.execute(
                        f"""
                        SELECT field_id
                        FROM read_parquet(?) AS f
                        WHERE timeline_bucket = ?
                        {filter_clause}
                        ORDER BY field_id
                        LIMIT 5
                        """,
                        [str(self.frame_path), timeline_bucket, *filter_params],
                    ).fetchall()
                ]
                diagnostics["geometry_field_samples"] = [
                    row[0]
                    for row in con.execute(
                        """
                        SELECT field_id
                        FROM read_parquet(?)
                        ORDER BY field_id
                        LIMIT 5
                        """,
                        [str(self.geometry_path)],
                    ).fetchall()
                ]
                diagnostics["timeline_bucket_samples"] = [
                    row[0]
                    for row in con.execute(
                        """
                        SELECT DISTINCT timeline_bucket
                        FROM read_parquet(?)
                        ORDER BY timeline_bucket
                        LIMIT 5
                        """,
                        [str(self.frame_path)],
                    ).fetchall()
                ]
                if bbox and optimized:
                    min_lon, min_lat, max_lon, max_lat = bbox
                    diagnostics["joined_rows_inside_bbox"] = con.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM read_parquet(?) AS f
                        JOIN read_parquet(?) AS g USING (field_id)
                        WHERE f.timeline_bucket = ?
                          {filter_clause}
                          AND NOT (
                            g.max_lon < ? OR g.min_lon > ? OR g.max_lat < ? OR g.min_lat > ?
                          )
                        """,
                        [
                            str(self.frame_path),
                            str(self.geometry_path),
                            timeline_bucket,
                            *filter_params,
                            min_lon,
                            max_lon,
                            min_lat,
                            max_lat,
                        ],
                    ).fetchone()[0]
            if bbox:
                diagnostics["requested_bbox"] = list(bbox)
                diagnostics["geometry_bounds"] = self.bounds()
            diagnostics["filters"] = filters
        except Exception as exc:
            diagnostics["diagnostic_error"] = str(exc)
        return diagnostics


def make_handler(store: StoryMapStore, settings: Settings) -> type[BaseHTTPRequestHandler]:
    api_cache = ResponseCache(
        ttl_seconds=settings.cache_seconds,
        capacity=settings.cache_entries,
        gzip_min_bytes=settings.gzip_min_bytes,
    )
    static_cache = ResponseCache(
        ttl_seconds=settings.cache_seconds,
        capacity=settings.cache_entries,
        gzip_min_bytes=settings.gzip_min_bytes,
    )
    query_slots = BoundedSemaphore(max(1, int(settings.query_concurrency)))
    static_root = settings.static_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._request_started = time.perf_counter()
            self._api_cache_key: str | None = None
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path
            if path.startswith("/api/"):
                LOGGER.info(
                    "http_request method=GET path=%s query=%s remote=%s",
                    path,
                    parsed.query,
                    self.client_address[0] if self.client_address else None,
                )
            try:
                if path.startswith("/api/") and path != "/api/health":
                    self._api_cache_key = self.path
                    cached = api_cache.get(self.path)
                    if cached is not None:
                        self._send_cached_json(cached, cache_status="HIT")
                        return
                if path == "/api/health":
                    self._json(store.health())
                    return
                if path == "/api/config":
                    self._json(store.config())
                    return
                if path == "/api/manifest":
                    self._json(self._query(store.manifest))
                    return
                if path == "/api/timeline":
                    self._json(self._query(store.timeline))
                    return
                if path == "/api/motifs":
                    self._json(
                        self._query(
                            lambda: store.motifs(
                                _first(query, "q"),
                                _int_query(query, "limit", 250, 1000),
                            )
                        )
                    )
                    return
                if path == "/api/activity":
                    self._json(self._query(lambda: store.activity(_filters_from_query(query))))
                    return
                if path == "/api/trajectory":
                    self._json(self._query(lambda: store.trajectory(_filters_from_query(query))))
                    return
                if path == "/api/trail":
                    bucket = _first(query, "bucket")
                    if not bucket:
                        raise RequestValidationError("bucket is required")
                    self._json(
                        self._query(
                            lambda: store.trail_features(
                                timeline_bucket=bucket,
                                filters=_filters_from_query(query),
                                lookback=_int_query(query, "lookback", 5, 24),
                                bbox=_parse_bbox(_first(query, "bbox")),
                                limit=_feature_limit_query(
                                    query,
                                    "limit",
                                    settings.default_feature_limit,
                                    settings.max_feature_limit,
                                ),
                            ),
                        )
                    )
                    return
                if path.startswith("/api/frame/"):
                    bucket = unquote(path.removeprefix("/api/frame/"))
                    self._json(
                        self._query(
                            lambda: store.frame_features(
                                timeline_bucket=bucket,
                                bbox=_parse_bbox(_first(query, "bbox")),
                                filters=_filters_from_query(query),
                                limit=_feature_limit_query(
                                    query,
                                    "limit",
                                    settings.default_feature_limit,
                                    settings.max_feature_limit,
                                ),
                            ),
                        )
                    )
                    return
                if path.startswith("/api/field/") and path.endswith("/events"):
                    field_id = unquote(path.removeprefix("/api/field/").removesuffix("/events"))
                    self._json(
                        self._query(
                            lambda: store.field_events(field_id, _int_query(query, "limit", 100, 500))
                        )
                    )
                    return
                if path.startswith("/api/cluster/"):
                    cluster_id = unquote(path.removeprefix("/api/cluster/"))
                    self._json(
                        self._query(
                            lambda: store.cluster(cluster_id, _int_query(query, "limit", 100, 500))
                        )
                    )
                    return
                if path == "/api/search/fields":
                    q = _first(query, "q")
                    if not q:
                        raise RequestValidationError("q is required")
                    self._json(
                        self._query(lambda: store.field_search(q, _int_query(query, "limit", 20, 50)))
                    )
                    return
                self._static(path)
            except RequestValidationError as exc:
                LOGGER.warning(
                    "http_bad_request method=GET path=%s query=%s remote=%s error=%s elapsed_ms=%.1f",
                    path,
                    parsed.query,
                    self.client_address[0] if self.client_address else None,
                    str(exc),
                    (time.perf_counter() - self._request_started) * 1000,
                )
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except ServerBusyError:
                LOGGER.warning(
                    "http_server_busy method=GET path=%s query=%s remote=%s elapsed_ms=%.1f",
                    path,
                    parsed.query,
                    self.client_address[0] if self.client_address else None,
                    (time.perf_counter() - self._request_started) * 1000,
                )
                self._json(
                    {"error": "The server is busy. Retry this request shortly."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    extra_headers={"Retry-After": "1"},
                )
            except Exception:
                LOGGER.exception(
                    "http_error method=GET path=%s query=%s remote=%s elapsed_ms=%.1f",
                    path,
                    parsed.query,
                    self.client_address[0] if self.client_address else None,
                    (time.perf_counter() - self._request_started) * 1000,
                )
                self._json(
                    {"error": "The server could not complete this request."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _query(self, operation: Any) -> Any:
            if not query_slots.acquire(blocking=False):
                raise ServerBusyError("query capacity exhausted")
            try:
                return operation()
            finally:
                query_slots.release()

        def _json(
            self,
            payload: object,
            status: HTTPStatus = HTTPStatus.OK,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(_clean(payload), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            cache_key = getattr(self, "_api_cache_key", None)
            if status == HTTPStatus.OK and cache_key:
                cached = api_cache.put(cache_key, body)
                cache_status = "MISS" if api_cache.capacity and api_cache.ttl_seconds else "BYPASS"
                cache_control = f"private, max-age={max(0, int(settings.cache_seconds))}"
            else:
                cached = api_cache.put("", body)
                cache_status = "BYPASS"
                cache_control = "no-store"
            self._send_body(
                cached,
                status=status,
                content_type="application/json; charset=utf-8",
                cache_control=cache_control,
                cache_status=cache_status,
                extra_headers=extra_headers,
            )
            self._log_api_response(status, len(body))

        def _send_cached_json(self, cached: CachedBody, *, cache_status: str) -> None:
            self._send_body(
                cached,
                status=HTTPStatus.OK,
                content_type="application/json; charset=utf-8",
                cache_control=f"private, max-age={max(0, int(settings.cache_seconds))}",
                cache_status=cache_status,
            )
            self._log_api_response(HTTPStatus.OK, len(cached.body))

        def _send_body(
            self,
            cached: CachedBody,
            *,
            status: HTTPStatus,
            content_type: str,
            cache_control: str,
            cache_status: str,
            etag: str | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            use_gzip = (
                cached.gzip_body is not None
                and _accepts_gzip(self.headers.get("Accept-Encoding"))
                and _compressible_content_type(content_type)
            )
            body = cached.gzip_body if use_gzip else cached.body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", cache_control)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("X-Cache", cache_status)
            if etag:
                self.send_header("ETag", etag)
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _log_api_response(self, status: HTTPStatus, raw_bytes: int) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/"):
                started = getattr(self, "_request_started", time.perf_counter())
                LOGGER.info(
                    "http_response method=GET path=%s status=%s bytes=%s elapsed_ms=%.1f",
                    path,
                    int(status),
                    raw_bytes,
                    (time.perf_counter() - started) * 1000,
                )

        def _static(self, raw_path: str) -> None:
            rel = "index.html" if raw_path in {"", "/"} else unquote(raw_path).lstrip("/")
            path = (static_root / rel).resolve()
            try:
                path.relative_to(static_root)
            except ValueError:
                LOGGER.warning("static_path_rejected raw_path=%s resolved_path=%s", raw_path, path)
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not path.exists() or path.is_dir():
                LOGGER.warning("static_not_found raw_path=%s resolved_path=%s", raw_path, path)
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            stat = path.stat()
            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            cache_control = f"public, max-age={max(0, int(settings.cache_seconds))}"
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("Cache-Control", cache_control)
                self.send_header("Vary", "Accept-Encoding")
                self.send_header("X-Cache", "VALIDATED")
                self.send_header("ETag", etag)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            cache_key = f"{path}:{etag}"
            cached = static_cache.get(cache_key)
            if cached is None:
                cached = static_cache.put(cache_key, path.read_bytes())
                cache_status = "MISS" if static_cache.capacity and static_cache.ttl_seconds else "BYPASS"
            else:
                cache_status = "HIT"
            self._send_body(
                cached,
                status=HTTPStatus.OK,
                content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                cache_control=cache_control,
                cache_status=cache_status,
                etag=etag,
            )

    return Handler


def setup_logging(level: str) -> None:
    normalized = str(level or "INFO").upper()
    numeric_level = getattr(logging, normalized, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        normalized = "INFO"
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    LOGGER.info("logging_configured level=%s", normalized)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_portable_path(raw: str, *, prefer_server_dir: bool = False) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        [SERVER_DIR / path, Path.cwd() / path]
        if prefer_server_dir
        else [Path.cwd() / path, SERVER_DIR / path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


FILTER_COLUMNS = {
    "story_cluster_id",
    "max_risk_band",
    "hazard_signature",
    "response_signature",
    "motif_family",
}


def _filters_from_query(query: dict[str, list[str]]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for key in FILTER_COLUMNS:
        value = _first(query, key)
        if value:
            filters[key] = value
    return filters


def _clean_filters(filters: dict[str, str] | None) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in (filters or {}).items():
        if key in FILTER_COLUMNS and value is not None and str(value) != "":
            clean[key] = str(value)
    return clean


def _filter_sql(
    filters: dict[str, str] | None,
    alias: str,
    *,
    motif_family_sql: str | None = None,
) -> tuple[str, list[Any]]:
    filters = _clean_filters(filters)
    clauses = []
    params: list[Any] = []
    for key in sorted(filters):
        column_sql = motif_family_sql if key == "motif_family" and motif_family_sql else f"{alias}.{key}"
        clauses.append(f"AND {column_sql} = ?")
        params.append(filters[key])
    return ("\n".join(clauses), params)


def _hazard_family_sql(column_sql: str) -> str:
    value = f"LOWER(COALESCE(CAST({column_sql} AS VARCHAR), ''))"
    heat = f"CASE WHEN {value} LIKE '%heat%' OR {value} LIKE '%temperature%' THEN 1 ELSE 0 END"
    wind = f"CASE WHEN {value} LIKE '%wind%' OR {value} LIKE '%storm%' THEN 1 ELSE 0 END"
    drought = f"CASE WHEN {value} LIKE '%drought%' OR {value} LIKE '%dry%' THEN 1 ELSE 0 END"
    flood = (
        f"CASE WHEN {value} LIKE '%flood%' OR {value} LIKE '%ponding%' "
        f"OR {value} LIKE '%waterlog%' THEN 1 ELSE 0 END"
    )
    recognized_count = f"(({heat}) + ({wind}) + ({drought}) + ({flood}))"
    return f"""
        CASE
            WHEN {recognized_count} >= 2 THEN 'compound'
            WHEN ({heat}) = 1 THEN 'heat'
            WHEN ({wind}) = 1 THEN 'wind'
            WHEN ({drought}) = 1 THEN 'drought'
            WHEN ({flood}) = 1 THEN 'flood'
            WHEN TRIM({value}) IN ('', 'none', 'unknown', 'no_hazard') THEN 'none'
            ELSE 'other'
        END
    """


def _file_size(path: Path) -> int | None:
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return None
    return None


def _accepts_gzip(value: str | None) -> bool:
    explicit_gzip_quality: float | None = None
    wildcard_quality: float | None = None
    for item in str(value or "").split(","):
        encoding, *parameters = item.strip().lower().split(";")
        if encoding not in {"gzip", "*"}:
            continue
        quality = 1.0
        for parameter in parameters:
            name, separator, raw_quality = parameter.strip().partition("=")
            if separator and name == "q":
                try:
                    quality = float(raw_quality)
                except ValueError:
                    quality = 0.0
        if encoding == "gzip":
            explicit_gzip_quality = quality
        else:
            wildcard_quality = quality
    if explicit_gzip_quality is not None:
        return explicit_gzip_quality > 0
    return wildcard_quality is not None and wildcard_quality > 0


def _compressible_content_type(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/javascript", "application/xml", "image/svg+xml"}
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _json_for_log(value: Any) -> str:
    return json.dumps(_clean(value), separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _int_query(query: dict[str, list[str]], key: str, default: int, maximum: int) -> int:
    raw = _first(query, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{key} must be an integer") from exc
    return max(1, min(value, maximum))


def _feature_limit_query(query: dict[str, list[str]], key: str, default: int, maximum: int) -> int:
    raw = _first(query, key)
    bounded_maximum = maximum if maximum > 0 else default if default > 0 else 5000
    bounded_default = default if default > 0 else bounded_maximum
    bounded_default = max(1, min(bounded_default, bounded_maximum))
    if raw is None or raw == "":
        return bounded_default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{key} must be an integer") from exc
    if value <= 0:
        return bounded_default
    return max(1, min(value, bounded_maximum))


def _feature_limit(limit: int, maximum: int) -> int | None:
    if limit <= 0:
        return None
    if maximum <= 0:
        return max(1, limit)
    return max(1, min(limit, maximum))


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    raw_parts = raw.split(",")
    if len(raw_parts) != 4:
        raise RequestValidationError("bbox must be minLon,minLat,maxLon,maxLat")
    try:
        parts = [float(item.strip()) for item in raw_parts]
    except (TypeError, ValueError) as exc:
        raise RequestValidationError("bbox values must be numbers") from exc
    min_lon, min_lat, max_lon, max_lat = parts
    if not all(math.isfinite(value) for value in parts):
        raise RequestValidationError("bbox values must be finite numbers")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise RequestValidationError("bbox longitude must be between -180 and 180")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise RequestValidationError("bbox latitude must be between -90 and 90")
    if min_lon > max_lon or min_lat > max_lat:
        raise RequestValidationError("bbox min values must be <= max values")
    return min_lon, min_lat, max_lon, max_lat


@lru_cache(maxsize=4)
def _bounds_for_geometry(path: str, optimized: bool) -> dict[str, float]:
    if optimized:
        with duckdb.connect(":memory:") as con:
            row = con.execute(
                """
                SELECT
                    MIN(min_lon) AS min_lon,
                    MIN(min_lat) AS min_lat,
                    MAX(max_lon) AS max_lon,
                    MAX(max_lat) AS max_lat
                FROM read_parquet(?)
                """,
                [path],
            ).fetchone()
        return {"minLon": row[0], "minLat": row[1], "maxLon": row[2], "maxLat": row[3]}

    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            """
            SELECT geometry_text, geometry_format
            FROM read_parquet(?)
            """,
            [path],
        ).fetchall()
    bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    for text, fmt in rows:
        _, bbox = _geometry_to_geojson_and_bbox(text, fmt)
        bounds[0] = min(bounds[0], bbox[0])
        bounds[1] = min(bounds[1], bbox[1])
        bounds[2] = max(bounds[2], bbox[2])
        bounds[3] = max(bounds[3], bbox[3])
    if not math.isfinite(bounds[0]):
        return {"minLon": 0.0, "minLat": 0.0, "maxLon": 1.0, "maxLat": 1.0}
    return {"minLon": bounds[0], "minLat": bounds[1], "maxLon": bounds[2], "maxLat": bounds[3]}


def _geometry_to_geojson_and_bbox(text: Any, fmt: str) -> tuple[dict[str, Any], list[float]]:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import mapping, shape

    if not text:
        raise ValueError("empty geometry")
    if fmt == "geojson":
        geometry = json.loads(text) if isinstance(text, str) else text
        geom = shape(geometry)
    else:
        geom = shapely_wkt.loads(str(text))
        geometry = mapping(geom)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    return dict(geometry), [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


def _intersects(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _records(df: Any) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    clean_df = df.where(df.notnull(), None)
    return clean_df.to_dict(orient="records")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _public_manifest(manifest: Any) -> dict[str, Any]:
    """Return only browser-useful manifest sections with host paths removed."""
    if not isinstance(manifest, dict):
        return {}
    public: dict[str, Any] = {}
    for key in PUBLIC_MANIFEST_SECTIONS:
        if key not in manifest:
            continue
        sanitized = _sanitize_manifest_value(manifest[key])
        if sanitized is not _MANIFEST_DROP:
            public[key] = sanitized
    return public


_MANIFEST_DROP = object()


def _sanitize_manifest_value(value: Any, key: str = "") -> Any:
    normalized_key = str(key).lower()
    if (
        normalized_key in MANIFEST_PATH_KEYS
        or normalized_key.endswith("_path")
        or normalized_key.endswith("_dir")
        or normalized_key.endswith("_parquet")
    ):
        return _MANIFEST_DROP
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            sanitized = _sanitize_manifest_value(child_value, str(child_key))
            if sanitized is not _MANIFEST_DROP:
                result[str(child_key)] = sanitized
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            sanitized = _sanitize_manifest_value(item)
            if sanitized is not _MANIFEST_DROP:
                result.append(sanitized)
        return result
    if isinstance(value, str) and _looks_like_absolute_path(value):
        return _MANIFEST_DROP
    return value


def _looks_like_absolute_path(value: str) -> bool:
    text = value.strip()
    return (
        text.startswith(("/", "\\\\"))
        or (len(text) >= 3 and text[1] == ":" and text[2] in {"/", "\\"})
    )


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    store = StoryMapStore(settings)
    server = BoundedThreadingHTTPServer(
        (settings.host, settings.port),
        make_handler(store, settings),
        max_concurrency=settings.query_concurrency,
    )
    LOGGER.info("server_start url=http://%s:%s", settings.host, settings.port)
    LOGGER.info("server_paths run_dir=%s static_dir=%s", settings.run_dir, settings.static_dir)
    LOGGER.info("server_health %s", _json_for_log(store.health()))
    server.serve_forever()


if __name__ == "__main__":
    main()
