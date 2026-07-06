from __future__ import annotations

import hashlib
import json
import logging
import math
import mimetypes
from numbers import Real
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
import gzip
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import duckdb


LOGGER = logging.getLogger("story_map_server")

MAX_GEOMETRY_IDS = 2000
MAX_GEOMETRY_REQUEST_BYTES = 256 * 1024
V4_INCIDENT_DEFAULT_LOOKBACK_DAYS = 366
V4_INCIDENT_MAX_LOOKBACK_DAYS = 3660
V4_INCIDENT_DEFAULT_HISTORY_LIMIT = 5000
V4_INCIDENT_MAX_HISTORY_LIMIT = 20000

FRAME_STATE_FIELDS = (
    "timeline_bucket",
    "field_id",
    "story_cluster_id",
    "incident_id",
    "crop_name",
    "stage_bucket",
    "incident_state",
    "event_id",
    "event_state",
    "motif_id",
    "archetype_display_state",
    "anchor_status",
    "accepted",
    "max_risk_band",
    "current_risk_band",
    "hazard_signature",
    "response_signature",
    "reportable_day_count",
    "event_count",
    "max_risk_rank",
    "current_risk_rank",
    "response_day_count",
    "motif_family",
    "short_label",
    "assignment_distance",
    "distance_ratio",
    "assignment_reason",
    "motif_model_version",
    "concurrent_event_count",
)

INCIDENT_V3_ARTIFACT_NAMES = {
    "footprints": "incident_footprints.parquet",
    "weekly_state": "incident_weekly_state.parquet",
    "stage_summary": "incident_stage_summary.parquet",
    "windows": "incident_windows.parquet",
    "lineage": "incident_lineage.parquet",
}

INCIDENT_V4_ARTIFACT_NAMES = {
    "timeline": "daily_timeline_v4.parquet",
    "field_state": "field_day_state_v4.parquet",
    "pressure_observations": "pressure_observations_v4.parquet",
    "grid": "daily_field_grid_v4.parquet",
    "pressure_grid": "daily_pressure_grid_v4.parquet",
    "s2_attempts": "s2_attempts_v4.parquet",
    "s2_updates": "s2_updates_v4.parquet",
    "story_checkpoints": "story_checkpoints_v4.parquet",
    "story_footprints": "story_footprints_v4.parquet",
}
INCIDENT_V4_CLOSED_STORY_RETENTION_DAYS = 28
INCIDENT_V4_STORY_CHECKPOINT_LOOKBACK_DAYS = 56

INCIDENT_FOOTPRINT_FILTER_COLUMNS = frozenset(
    {
        "incident_id",
        "crop_name",
        "hazard_family",
        "incident_state",
        "stage_bucket",
    }
)

API_CACHE_QUERY_KEYS = frozenset(
    {
        "as_of", "bbox", "bucket", "limit", "lookback", "lookback_days",
        "history_limit", "q",
        "story_cluster_id", "incident_id", "crop_instance_id", "crop_name", "stage_bucket",
        "incident_state", "max_risk_band", "current_risk_band",
        "hazard_signature", "hazard_family", "response_signature",
        "motif_family",
    }
)

INCIDENT_FOOTPRINT_REQUIRED_COLUMNS = frozenset(
    {
        "timeline_bucket",
        "incident_id",
        "crop_name",
        "hazard_family",
        "incident_state",
        "stage_bucket",
        "geometry_geojson",
        "geometry_type",
        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",
        "footprint_geometry_method",
        "low_zoom_omitted",
        "monitored_count",
        "evaluable_count",
        "affected_count",
        "severe_count",
        "footprint_carried_forward",
        "is_physical_movement",
    }
)

INCIDENT_FOOTPRINT_PROPERTY_FIELDS = (
    "timeline_bucket",
    "incident_id",
    "story_cluster_id",
    "exposure_id",
    "crop_name",
    "hazard_family",
    "incident_state",
    "current_state",
    "stage_bucket",
    "stage_distribution",
    "component_id",
    "monitored_count",
    "evaluable_count",
    "affected_count",
    "active_count",
    "severe_count",
    "pressure_core_count",
    "impact_lag_count",
    "global_crop_week_unmappable_instance_count",
    "pressure_core_field_count",
    "severe_field_count",
    "watch_frontier_field_count",
    "impact_lag_field_count",
    "coverage_adequate",
    "coverage_missing_cell_count",
    "unresolved_carried_field_count",
    "recovered_field_count",
    "fresh_decline_field_count",
    "fresh_recovery_field_count",
    "footprint_carried_forward",
    "footprint_cell_count",
    "pressure_cell_count",
    "impact_cell_count",
    "watch_cell_count",
    "relapse_count",
    "data_gap_count",
    "right_censored",
    "fresh_decline_evidence",
    "fresh_recovery_evidence",
    "pressure_core_evidence",
    "footprint_geometry_method",
    "footprint_area_km2",
    "coincident_group_id",
    "coincident_incident_count",
    "coincident_incident_index",
    "coincident_crop_names_json",
    "is_physical_movement",
)

FRAME_STATE_META_FIELDS = (
    "timeline_bucket",
    "source_row_count",
    "query_row_count",
    "story_cluster_count",
    "reportable_day_count",
    "event_count",
    "limit",
    "requested_limit",
    "unlimited",
    "limit_hit",
    "truncated",
    "optimized_geometry",
    "filters",
)

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
    "policy",
    "semantics",
    "motifs",
    "limitations",
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


class ResourceNotFoundError(LookupError):
    """A missing optional public resource that is safe to report as HTTP 404."""


class ServerBusyError(RuntimeError):
    """The bounded query executor has no immediately available capacity."""


class GeometryVersionMismatchError(RuntimeError):
    """The client requested geometry from a different immutable artifact."""


class RequestBodyTooLargeError(RuntimeError):
    """A public request exceeded an explicitly bounded request size."""


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
    cache_max_bytes: int = 536_870_912
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
            cache_max_bytes=int(
                os.getenv("STORY_MAP_CACHE_MAX_BYTES", "536870912")
            ),
            gzip_min_bytes=int(os.getenv("STORY_MAP_GZIP_MIN_BYTES", "1024")),
            query_concurrency=int(os.getenv("STORY_MAP_QUERY_CONCURRENCY", "8")),
        )


@dataclass(frozen=True)
class CachedBody:
    body: bytes
    gzip_body: bytes | None


class ResponseCache:
    """Small process-local TTL/LRU cache for immutable run artifacts."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        capacity: int,
        gzip_min_bytes: int,
        max_bytes: int = 536_870_912,
    ) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.capacity = max(0, int(capacity))
        self.gzip_min_bytes = max(0, int(gzip_min_bytes))
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[str, tuple[float, CachedBody]] = OrderedDict()
        self._size_bytes = 0
        self._lock = RLock()

    def get(self, key: str) -> CachedBody | None:
        if (
            not key
            or self.capacity == 0
            or self.ttl_seconds == 0
            or self.max_bytes == 0
        ):
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, cached = item
            if expires_at <= now:
                self._items.pop(key, None)
                self._size_bytes -= _cached_body_size(cached)
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
        if (
            not key
            or self.capacity == 0
            or self.ttl_seconds == 0
            or self.max_bytes == 0
        ):
            return cached
        cached_size = _cached_body_size(cached)
        if cached_size > self.max_bytes:
            return cached
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._size_bytes -= _cached_body_size(previous[1])
            self._items[key] = (expires_at, cached)
            self._size_bytes += cached_size
            self._items.move_to_end(key)
            while (
                len(self._items) > self.capacity
                or self._size_bytes > self.max_bytes
            ):
                _, (_, evicted) = self._items.popitem(last=False)
                self._size_bytes -= _cached_body_size(evicted)
        return cached

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return self._size_bytes


def _cached_body_size(value: CachedBody) -> int:
    return len(value.body) + (len(value.gzip_body) if value.gzip_body else 0)


def _cache_byte_budgets(total_bytes: int) -> tuple[int, int]:
    """Split one process cache budget between API payloads and static assets."""

    total = max(0, int(total_bytes))
    static = total // 16
    return total - static, static


def _encode_json_body(payload: object) -> bytes:
    """Clean and encode one JSON response before optional cache compression."""

    return json.dumps(
        _clean(payload),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


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
        self.state_snapshots_path = self.run_dir / "event_state_snapshots.parquet"
        self.manifest_path = self._pick("manifest.json")
        self.timeline_summary_path = self.run_dir / "gpu_summaries" / "timeline_summary.parquet"
        self.incident_footprints_path = self.run_dir / INCIDENT_V3_ARTIFACT_NAMES["footprints"]
        self.incident_weekly_state_path = self.run_dir / INCIDENT_V3_ARTIFACT_NAMES["weekly_state"]
        self.incident_stage_summary_path = self.run_dir / INCIDENT_V3_ARTIFACT_NAMES["stage_summary"]
        self.incident_windows_path = self.run_dir / INCIDENT_V3_ARTIFACT_NAMES["windows"]
        self.incident_lineage_path = self.run_dir / INCIDENT_V3_ARTIFACT_NAMES["lineage"]
        self.incident_membership_path = self.run_dir / "incident_membership.parquet"
        self.v4_timeline_path = self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["timeline"]
        self.v4_field_state_path = self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["field_state"]
        self.v4_pressure_observations_path = (
            self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["pressure_observations"]
        )
        self.v4_grid_path = self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["grid"]
        self.v4_pressure_grid_path = (
            self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["pressure_grid"]
        )
        self.v4_s2_attempts_path = (
            self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["s2_attempts"]
        )
        self.v4_s2_updates_path = self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["s2_updates"]
        self.v4_story_checkpoints_path = (
            self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["story_checkpoints"]
        )
        self.v4_story_footprints_path = (
            self.run_dir / INCIDENT_V4_ARTIFACT_NAMES["story_footprints"]
        )
        self.v4_viewer_validation: dict[str, Any] | None = None
        if any(path.exists() for path in self._incident_v4_paths().values()):
            from story_monitor.incident_viewer_v4 import validate_viewer_directory

            self.v4_viewer_validation = validate_viewer_directory(self.run_dir)
        self.frame_columns = self._parquet_columns(self.frame_path)
        self.timeline_summary_columns = self._parquet_columns(self.timeline_summary_path)
        self.incident_footprint_columns = self._parquet_columns(self.incident_footprints_path)
        self.incident_weekly_state_columns = self._parquet_columns(self.incident_weekly_state_path)
        self.incident_stage_summary_columns = self._parquet_columns(self.incident_stage_summary_path)
        self.incident_window_columns = self._parquet_columns(self.incident_windows_path)
        self.incident_lineage_columns = self._parquet_columns(self.incident_lineage_path)
        self.incident_membership_columns = self._parquet_columns(
            self.incident_membership_path
        )
        self.v4_field_state_columns = self._parquet_columns(self.v4_field_state_path)
        self.v4_grid_columns = self._parquet_columns(self.v4_grid_path)
        self.v4_story_checkpoint_columns = self._parquet_columns(
            self.v4_story_checkpoints_path
        )
        self.v4_story_footprint_columns = self._parquet_columns(
            self.v4_story_footprints_path
        )
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

    def _incident_v3_paths(self) -> dict[str, Path]:
        return {
            "footprints": self.incident_footprints_path,
            "weekly_state": self.incident_weekly_state_path,
            "stage_summary": self.incident_stage_summary_path,
            "windows": self.incident_windows_path,
            "lineage": self.incident_lineage_path,
        }

    def _incident_v4_paths(self) -> dict[str, Path]:
        return {
            "timeline": self.v4_timeline_path,
            "field_state": self.v4_field_state_path,
            "pressure_observations": self.v4_pressure_observations_path,
            "grid": self.v4_grid_path,
            "pressure_grid": self.v4_pressure_grid_path,
            "s2_attempts": self.v4_s2_attempts_path,
            "s2_updates": self.v4_s2_updates_path,
            "story_checkpoints": self.v4_story_checkpoints_path,
            "story_footprints": self.v4_story_footprints_path,
        }

    def has_incident_v3(self) -> bool:
        """Return whether the complete optional Incident V3 API bundle is present."""
        return all(path.is_file() for path in self._incident_v3_paths().values())

    def has_incident_v4(self) -> bool:
        """Return whether the complete optional dual-clock projection is present."""
        return self.has_incident_v3() and all(
            path.is_file() for path in self._incident_v4_paths().values()
        )

    def require_incident_v4(self) -> None:
        if not self.has_incident_v4():
            raise ResourceNotFoundError(
                "Incident V4 dual-clock data is not available for this viewer bundle."
            )

    def require_incident_v3(self) -> None:
        if not self.has_incident_v3():
            raise ResourceNotFoundError(
                "Incident V3 data is not available for this viewer bundle."
            )

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

    def _optional_frame_sql(self, alias: str, name: str, fallback: str = "NULL") -> str:
        return f"{alias}.{name} AS {name}" if name in self.frame_columns else f"{fallback} AS {name}"

    def _effective_frame_filters(self, filters: dict[str, str] | None) -> dict[str, str]:
        """Map the live-risk filter to old bundles without changing new semantics."""
        clean = _clean_filters(filters)
        incident_id = clean.pop("incident_id", None)
        if incident_id is not None:
            story_cluster_id = clean.get("story_cluster_id")
            if story_cluster_id is not None and story_cluster_id != incident_id:
                raise RequestValidationError(
                    "incident_id and story_cluster_id must match when both are provided"
                )
            clean["story_cluster_id"] = incident_id
        for optional_name in ("crop_name", "stage_bucket", "incident_state"):
            if optional_name not in self.frame_columns:
                clean.pop(optional_name, None)
        if "current_risk_band" in clean and "current_risk_band" not in self.frame_columns:
            clean["max_risk_band"] = clean.pop("current_risk_band")
        return clean

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
            "capabilities": {"incident_v3": self.has_incident_v3()},
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
            "capabilities": {"incident_v3": self.has_incident_v3()},
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
            "incident_v3_available": self.has_incident_v3(),
        }
        LOGGER.info(
            "manifest_loaded run_dir=%s optimized_geometry=%s bounds=%s",
            self.run_dir,
            self._has_optimized_geometry(),
            _json_for_log(bounds),
        )
        return manifest

    @staticmethod
    def _require_incident_columns(
        artifact_name: str,
        columns: frozenset[str],
        required: set[str] | frozenset[str],
    ) -> None:
        missing = sorted(set(required) - set(columns))
        if missing:
            raise RuntimeError(
                f"Incident V3 {artifact_name} schema is missing required columns: "
                + ", ".join(missing)
            )

    def incident_footprints(
        self,
        *,
        timeline_bucket: str,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return every exact incident footprint for one week, without a field cap."""
        self.require_incident_v3()
        self._require_incident_columns(
            "footprints",
            self.incident_footprint_columns,
            INCIDENT_FOOTPRINT_REQUIRED_COLUMNS,
        )
        clean_filters = _clean_incident_filters(filters)
        public_filters = dict(clean_filters)
        stage_bucket = clean_filters.pop("stage_bucket", None)
        filter_clause, filter_params = _incident_filter_sql(clean_filters, "f")
        if stage_bucket is not None:
            self._require_incident_columns(
                "stage summary",
                self.incident_stage_summary_columns,
                {
                    "incident_id", "timeline_bucket", "stage_bucket",
                    "affected_crop_instance_count",
                },
            )
            filter_clause += """
                AND EXISTS (
                    SELECT 1
                    FROM read_parquet(?) AS stage
                    WHERE CAST(stage.timeline_bucket AS DATE)
                            = CAST(f.timeline_bucket AS DATE)
                      AND CAST(stage.incident_id AS VARCHAR)
                            = CAST(f.incident_id AS VARCHAR)
                      AND CAST(stage.stage_bucket AS VARCHAR) = ?
                      AND COALESCE(
                            TRY_CAST(stage.affected_crop_instance_count AS BIGINT), 0
                          ) > 0
                )
            """
            filter_params.extend(
                [str(self.incident_stage_summary_path), stage_bucket]
            )
        geometry_fields = (
            "geometry_geojson", "geometry_type",
            "min_lon", "min_lat", "max_lon", "max_lat",
        )
        projection = ",\n                    ".join(
            (
                f'f."{column}"'
                if column in self.incident_footprint_columns
                else f'NULL AS "{column}"'
            )
            for column in dict.fromkeys(
                (*geometry_fields, *INCIDENT_FOOTPRINT_PROPERTY_FIELDS)
            )
        )
        with duckdb.connect(":memory:") as con:
            summary = con.execute(
                """
                SELECT
                    COUNT(*) AS source_count,
                    COUNT(DISTINCT CAST(incident_id AS VARCHAR)) AS distinct_incident_count,
                    SUM(CASE WHEN COALESCE(TRY_CAST(low_zoom_omitted AS BOOLEAN), TRUE)
                        THEN 1 ELSE 0 END) AS omitted_count,
                    SUM(CASE WHEN COALESCE(CAST(footprint_geometry_method AS VARCHAR), '')
                        <> 'exact_union_of_grid_rectangles' THEN 1 ELSE 0 END) AS inexact_count,
                    SUM(CASE WHEN COALESCE(TRY_CAST(is_physical_movement AS BOOLEAN), TRUE)
                        THEN 1 ELSE 0 END) AS physical_movement_count
                FROM read_parquet(?)
                WHERE CAST(timeline_bucket AS DATE) = CAST(? AS DATE)
                """,
                [str(self.incident_footprints_path), timeline_bucket],
            ).fetchone()
            rows = con.execute(
                f"""
                SELECT {projection}
                FROM read_parquet(?) AS f
                WHERE CAST(f.timeline_bucket AS DATE) = CAST(? AS DATE)
                {filter_clause}
                ORDER BY CAST(f.incident_id AS VARCHAR)
                """,
                [str(self.incident_footprints_path), timeline_bucket, *filter_params],
            ).fetchdf()

        source_count = int(summary[0] or 0)
        if source_count != int(summary[1] or 0):
            raise RuntimeError("Incident V3 footprints are not unique by incident and week")
        if any(int(value or 0) for value in summary[2:]):
            raise RuntimeError(
                "Incident V3 footprint completeness contract was violated"
            )

        features: list[dict[str, Any]] = []
        for row in _records(rows):
            geometry, geom_bbox = _precomputed_geojson_and_bbox(
                row.get("geometry_geojson"),
                row.get("geometry_type"),
                (
                    row.get("min_lon"), row.get("min_lat"),
                    row.get("max_lon"), row.get("max_lat"),
                ),
            )
            if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                raise RuntimeError("Incident V3 footprint geometry must be Polygon or MultiPolygon")
            properties = {
                key: row.get(key)
                for key in INCIDENT_FOOTPRINT_PROPERTY_FIELDS
            }
            incident_id = row.get("incident_id")
            properties["incident_id"] = incident_id
            properties["story_cluster_id"] = row.get("story_cluster_id") or incident_id
            properties["footprint_geometry_method"] = "exact_union_of_grid_rectangles"
            properties["is_physical_movement"] = False
            properties["bbox"] = geom_bbox
            features.append(
                {"type": "Feature", "geometry": geometry, "properties": properties}
            )

        LOGGER.info(
            "incident_footprints bucket=%s filters=%s source=%s matching=%s",
            timeline_bucket,
            _json_for_log(public_filters),
            source_count,
            len(features),
        )
        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "timeline_bucket": timeline_bucket,
                "source_footprint_count": source_count,
                "matching_footprint_count": len(features),
                "feature_count": len(features),
                "complete": True,
                "truncated": False,
                "feature_cap_applied": False,
                "low_zoom_footprints_dropped": False,
                "footprint_geometry_method": "exact_union_of_grid_rectangles",
                "is_physical_movement": False,
                "filters": public_filters,
            },
        }

    def incident_detail(self, incident_id: str) -> dict[str, Any]:
        """Return a complete V3 incident lifecycle, stage, and lineage drill-down."""
        self.require_incident_v3()
        self._require_incident_columns(
            "windows", self.incident_window_columns, {"incident_id"}
        )
        self._require_incident_columns(
            "weekly state",
            self.incident_weekly_state_columns,
            {"incident_id", "timeline_bucket"},
        )
        self._require_incident_columns(
            "stage summary",
            self.incident_stage_summary_columns,
            {"incident_id", "timeline_bucket"},
        )
        self._require_incident_columns(
            "footprints",
            self.incident_footprint_columns,
            {
                "incident_id", "timeline_bucket",
                "geometry_geojson", "footprint_geometry_method",
                "low_zoom_omitted", "is_physical_movement",
            },
        )

        detail_footprint_fields = (
            "timeline_bucket", "incident_id", "geometry_geojson",
            "footprint_geometry_method", "low_zoom_omitted",
            "is_physical_movement", "footprint_area_km2",
            "footprint_carried_forward", "incident_state", "stage_bucket",
            "crop_name", "hazard_family",
            "pressure_geometry_geojson", "impact_geometry_geojson",
            "watch_geometry_geojson", "pressure_cell_count",
            "impact_cell_count", "watch_cell_count",
        )
        detail_footprint_projection = ",\n                        ".join(
            (
                f'f."{column}"'
                if column in self.incident_footprint_columns
                else f'NULL AS "{column}"'
            )
            for column in detail_footprint_fields
        )

        window_order = _optional_order_by(
            self.incident_window_columns, ("first_evidence_week", "incident_id")
        )
        with duckdb.connect(":memory:") as con:
            window_rows = _records(
                con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                    {window_order}
                    """,
                    [str(self.incident_windows_path), incident_id],
                ).fetchdf()
            )
            if not window_rows:
                raise ResourceNotFoundError("Incident was not found.")
            if len(window_rows) != 1:
                raise RuntimeError("Incident V3 windows are not unique by incident_id")

            weekly_state = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                    ORDER BY timeline_bucket
                    """,
                    [str(self.incident_weekly_state_path), incident_id],
                ).fetchdf()
            )
            stage_rows = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                    ORDER BY timeline_bucket, stage_bucket
                    """,
                    [str(self.incident_stage_summary_path), incident_id],
                ).fetchdf()
            )
            footprint_rows = _records(
                con.execute(
                    f"""
                    SELECT {detail_footprint_projection}
                    FROM read_parquet(?) AS f
                    WHERE CAST(f.incident_id AS VARCHAR) = ?
                    ORDER BY timeline_bucket
                    """,
                    [str(self.incident_footprints_path), incident_id],
                ).fetchdf()
            )
            for footprint in footprint_rows:
                if (
                    str(footprint.get("footprint_geometry_method") or "")
                    != "exact_union_of_grid_rectangles"
                    or bool(footprint.get("low_zoom_omitted"))
                    or bool(footprint.get("is_physical_movement"))
                ):
                    raise RuntimeError(
                        "Incident V3 detail requires exact non-movement footprints"
                    )
                main_geometry, _ = _geometry_to_geojson_and_bbox(
                    footprint.pop("geometry_geojson", None), "geojson"
                )
                if main_geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                    raise RuntimeError(
                        "Incident V3 detail footprint geometry must be Polygon or MultiPolygon"
                    )
                footprint["geometry"] = main_geometry
                footprint["is_physical_movement"] = False
                for role in ("pressure", "impact", "watch"):
                    raw_geometry = footprint.pop(
                        f"{role}_geometry_geojson", None
                    )
                    if raw_geometry:
                        geometry, _ = _geometry_to_geojson_and_bbox(
                            raw_geometry, "geojson"
                        )
                        footprint[f"{role}_geometry"] = geometry
                    else:
                        footprint[f"{role}_geometry"] = None

            lineage_required = {"parent_incident_id", "child_incident_id"}
            if lineage_required.issubset(self.incident_lineage_columns):
                lineage_order = _optional_order_by(
                    self.incident_lineage_columns, ("timeline_bucket", "lineage_id")
                )
                incoming = _records(
                    con.execute(
                        f"""
                        SELECT * FROM read_parquet(?)
                        WHERE CAST(child_incident_id AS VARCHAR) = ?
                        {lineage_order}
                        """,
                        [str(self.incident_lineage_path), incident_id],
                    ).fetchdf()
                )
                outgoing = _records(
                    con.execute(
                        f"""
                        SELECT * FROM read_parquet(?)
                        WHERE CAST(parent_incident_id AS VARCHAR) = ?
                        {lineage_order}
                        """,
                        [str(self.incident_lineage_path), incident_id],
                    ).fetchdf()
                )
            else:
                incoming = []
                outgoing = []

        LOGGER.info(
            "incident_detail incident_id=%s weekly=%s stages=%s incoming=%s outgoing=%s",
            incident_id,
            len(weekly_state),
            len(stage_rows),
            len(incoming),
            len(outgoing),
        )
        return {
            "incident_id": incident_id,
            "window": window_rows[0],
            "weekly_state": weekly_state,
            "stage_summary": stage_rows,
            "stage_rows": stage_rows,
            "footprints": footprint_rows,
            "lineage": {"incoming": incoming, "outgoing": outgoing},
        }

    def v4_timeline(self) -> dict[str, Any]:
        """Return the precomputed daily pressure/S2/story clock ledger."""
        self.require_incident_v4()
        with duckdb.connect(":memory:") as con:
            rows = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?) ORDER BY calendar_date
                    """,
                    [str(self.v4_timeline_path)],
                ).fetchdf()
            )
        for row in rows:
            day = str(row.get("calendar_date") or "")[:10]
            row["calendar_date"] = day
            # Compatibility aliases let the existing accessible slider consume
            # V4 without changing any V3 timeline payload.
            row["timeline_bucket"] = day
            row["activity_count"] = int(row.get("elevated_pressure_field_count") or 0)
            row["activity_unit"] = "daily_pressure_fields"
        return {
            "days": rows,
            "buckets": rows,
            "source": "precomputed_dual_clock_v4",
            "clock": "daily_as_of",
            "complete": True,
        }

    def v4_frame(
        self,
        *,
        calendar_date: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return one precomputed low-zoom grid plus optional high-zoom fields."""
        self.require_incident_v4()
        started = time.perf_counter()
        clean_filters = _clean_incident_filters(filters)
        grid_clauses: list[str] = []
        grid_params: list[Any] = []
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            grid_clauses.append(
                "AND NOT (max_lon < ? OR min_lon > ? OR max_lat < ? OR min_lat > ?)"
            )
            grid_params.extend([min_lon, max_lon, min_lat, max_lat])
        grid_where = "\n".join(grid_clauses)
        with duckdb.connect(":memory:") as con:
            timeline_rows = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(calendar_date AS DATE) = CAST(? AS DATE)
                    """,
                    [str(self.v4_timeline_path), calendar_date],
                ).fetchdf()
            )
            if not timeline_rows:
                raise ResourceNotFoundError("Daily V4 frame was not found.")
            grid_rows = _records(
                con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(calendar_date AS DATE) = CAST(? AS DATE)
                    {grid_where}
                    ORDER BY grid_x, grid_y
                    """,
                    [str(self.v4_grid_path), calendar_date, *grid_params],
                ).fetchdf()
            )
            pressure_clause = ""
            pressure_params: list[Any] = []
            if clean_filters.get("hazard_family"):
                pressure_clause = "AND CAST(hazard_family AS VARCHAR) = ?"
                pressure_params.append(clean_filters["hazard_family"])
            if bbox:
                pressure_clause += (
                    "\nAND NOT (max_lon < ? OR min_lon > ? "
                    "OR max_lat < ? OR min_lat > ?)"
                )
                pressure_params.extend([min_lon, max_lon, min_lat, max_lat])
            pressure_rows = _records(
                con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(calendar_date AS DATE) = CAST(? AS DATE)
                    {pressure_clause}
                    ORDER BY hazard_family, grid_x, grid_y
                    """,
                    [
                        str(self.v4_pressure_grid_path), calendar_date,
                        *pressure_params,
                    ],
                ).fetchdf()
            )
            story_filter, story_params = _incident_filter_sql(clean_filters, "f")
            story_spatial = ""
            story_spatial_params: list[Any] = []
            if bbox:
                story_spatial = (
                    "AND NOT (f.max_lon < ? OR f.min_lon > ? "
                    "OR f.max_lat < ? OR f.min_lat > ?)"
                )
                story_spatial_params.extend([min_lon, max_lon, min_lat, max_lat])
            story_rows = _records(
                con.execute(
                    f"""
                    WITH eligible AS (
                        SELECT f.*,
                            ROW_NUMBER() OVER (
                                PARTITION BY CAST(f.incident_id AS VARCHAR)
                                ORDER BY CAST(f.story_known_date AS DATE) DESC,
                                    CAST(f.story_week AS DATE) DESC
                            ) AS _rank
                        FROM read_parquet(?) AS f
                        WHERE CAST(f.story_known_date AS DATE)
                          BETWEEN CAST(? AS DATE)
                              - INTERVAL '{INCIDENT_V4_STORY_CHECKPOINT_LOOKBACK_DAYS} days'
                          AND CAST(? AS DATE)
                    ), latest AS (
                        SELECT * EXCLUDE (_rank) FROM eligible WHERE _rank = 1
                    )
                    SELECT f.* FROM latest AS f
                    WHERE 1 = 1
                      {story_filter}
                      {story_spatial}
                      AND (
                        (
                          UPPER(COALESCE(CAST(incident_state AS VARCHAR), ''))
                              NOT LIKE 'CLOSED_%'
                          AND UPPER(COALESCE(CAST(incident_state AS VARCHAR), ''))
                              <> 'MERGED_INTO'
                        )
                        OR DATE_DIFF(
                            'day', CAST(story_known_date AS DATE), CAST(? AS DATE)
                        ) <= {INCIDENT_V4_CLOSED_STORY_RETENTION_DAYS}
                      )
                    ORDER BY CAST(incident_id AS VARCHAR)
                    """,
                    [
                        str(self.v4_story_footprints_path), calendar_date, calendar_date,
                        *story_params, *story_spatial_params, calendar_date,
                    ],
                ).fetchdf()
            )
            field_rows: list[dict[str, Any]] = []
            source_field_count = 0
            truncated = False
            if bbox:
                field_rows, source_field_count, truncated = self._v4_field_rows(
                    con,
                    calendar_date=calendar_date,
                    bbox=bbox,
                    limit=limit,
                    filters=clean_filters,
                )

        overview_features = [_v4_grid_feature(row, "field_overview") for row in grid_rows]
        pressure_features = [
            _v4_grid_feature(row, "pressure")
            for row in pressure_rows
            if int(row.get("pressure_field_count") or 0) > 0
        ]
        crop_features = [
            _v4_grid_feature(row, "crop_impact")
            for row in grid_rows
            if int(row.get("decline_field_count") or 0)
            + int(row.get("recovery_field_count") or 0) > 0
        ]
        story_features = [_v4_story_feature(row) for row in story_rows]
        timeline = timeline_rows[0]
        represented = int(timeline.get("represented_field_count") or 0)
        monitored = int(timeline.get("monitored_field_count") or 0)
        unmappable = int(timeline.get("unmappable_field_count") or 0)
        accounted = int(
            timeline.get("accounted_field_count")
            if timeline.get("accounted_field_count") is not None
            else represented + unmappable
        )
        source_day_present = bool(timeline.get("source_day_present"))
        country_truncated = False
        complete_country = bool(
            bbox is None
            and source_day_present
            and not country_truncated
            and accounted == monitored
        )
        representation_warnings = (
            [{
                "code": "unmappable_fields",
                "field_count": unmappable,
                "message": (
                    f"{unmappable} monitored fields are accounted for but cannot "
                    "be drawn because valid map coordinates are unavailable."
                ),
            }]
            if unmappable else []
        )
        latest_story_known = max(
            (str(row.get("story_known_date") or "")[:10] for row in story_rows),
            default=None,
        )
        latest_story_known_time = max(
            (
                str(row.get("story_known_time") or row.get("knowledge_time") or "")
                for row in story_rows
            ),
            default=None,
        )
        payload = {
            "calendar_date": calendar_date,
            "clocks": {
                "pressure_as_of_date": calendar_date,
                "crop_knowledge_cutoff": calendar_date,
                "story_knowledge_cutoff": calendar_date,
                "latest_story_known_date": latest_story_known,
                "latest_story_known_time": latest_story_known_time,
            },
            "field_overview": {
                "type": "FeatureCollection", "features": overview_features,
                "meta": {
                    "complete": complete_country,
                    "truncated": country_truncated,
                    "representation": (
                        "complete_centroid_aggregation"
                        if bbox is None else "viewport_centroid_aggregation"
                    ),
                    "represented_field_count": represented,
                    "monitored_field_count": monitored,
                    "source_field_count": monitored,
                    "unmappable_field_count": unmappable,
                    "accounted_field_count": accounted,
                    "unmappable_warning": bool(unmappable),
                    "warnings": representation_warnings,
                    "source_day_present": source_day_present,
                    "filtered": False,
                    "filter_scope": "unfiltered_all_field_context",
                    "viewport_scoped": bbox is not None,
                },
            },
            "pressure": {
                "type": "FeatureCollection", "features": pressure_features,
                "meta": {
                    "as_of_date": calendar_date,
                    "source": "daily_field_pressure",
                    "complete": True,
                    "geometry_substituted": False,
                },
            },
            "crop_impact": {
                "type": "FeatureCollection", "features": crop_features,
                "meta": {
                    "knowledge_cutoff": calendar_date,
                    "step_held": True,
                    "interpolated": False,
                    "geometry_substituted": False,
                },
            },
            "story_footprints": {
                "type": "FeatureCollection", "features": story_features,
                "meta": {
                    "knowledge_cutoff": calendar_date,
                    "complete": True,
                    "truncated": False,
                    "is_physical_movement": False,
                },
            },
            "fields": {
                "type": "FeatureCollection", "features": field_rows,
                "meta": {
                    "viewport_only": bbox is not None,
                    "source_field_count": source_field_count,
                    "feature_count": len(field_rows),
                    "truncated": truncated,
                },
            },
            "timeline": timeline,
            "meta": {
                "mode": "crop_incident_v4_dual_clock",
                "calendar_date": calendar_date,
                "complete_country_representation": complete_country,
                "source_day_present": source_day_present,
                "country_representation_truncated": country_truncated,
                "represented_field_count": represented,
                "source_field_count": monitored,
                "unmappable_field_count": unmappable,
                "accounted_field_count": accounted,
                "unmappable_warning": bool(unmappable),
                "warnings": representation_warnings,
                "all_monitored_fields_mapped": bool(
                    timeline.get("all_monitored_fields_mapped")
                ),
                "closed_story_retention_days": (
                    INCIDENT_V4_CLOSED_STORY_RETENTION_DAYS
                ),
                "story_checkpoint_query_lookback_days": (
                    INCIDENT_V4_STORY_CHECKPOINT_LOOKBACK_DAYS
                ),
                "filters": clean_filters,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        }
        return payload

    def _v4_field_rows(
        self,
        connection: duckdb.DuckDBPyConnection,
        *,
        calendar_date: str,
        bbox: tuple[float, float, float, float],
        limit: int,
        filters: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int, bool]:
        effective_limit = _feature_limit(limit, self.settings.max_feature_limit)
        clauses: list[str] = []
        params: list[Any] = []
        names = {
            "crop_name": "s.crop_name",
            "hazard_family": "s.hazard_family",
            "stage_bucket": "s.stage_bucket",
        }
        for key, expression in names.items():
            if key in filters:
                clauses.append(f"AND CAST({expression} AS VARCHAR) = ?")
                params.append(filters[key])
        min_lon, min_lat, max_lon, max_lat = bbox
        sql = f"""
            WITH state_day AS (
                SELECT * FROM read_parquet(?)
                WHERE CAST(calendar_date AS DATE) = CAST(? AS DATE)
            ), hazards AS (
                SELECT field_id,
                    STRING_AGG(
                        DISTINCT CAST(hazard_family AS VARCHAR), ','
                        ORDER BY CAST(hazard_family AS VARCHAR)
                    ) FILTER (WHERE pressure_active) AS active_hazards,
                    MAX(risk_rank) FILTER (WHERE pressure_observed)
                        AS field_max_risk_rank,
                    ARG_MAX(hazard_family, risk_rank) FILTER (WHERE pressure_observed)
                        AS highest_pressure_hazard
                FROM state_day GROUP BY field_id
            ), eligible AS (
                SELECT s.*, h.active_hazards, h.field_max_risk_rank,
                    h.highest_pressure_hazard,
                    g.geometry_geojson,
                    CAST('geojson' AS VARCHAR) AS geometry_format,
                    g.min_lon, g.min_lat, g.max_lon, g.max_lat,
                    COUNT(DISTINCT s.field_id) OVER () AS _source_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.field_id
                        ORDER BY s.pressure_active DESC, s.risk_rank DESC,
                            s.monitored DESC, s.evaluable DESC,
                            s.s2_knowledge_date DESC NULLS LAST,
                            s.crop_instance_id, s.hazard_family
                    ) AS _rank
                FROM state_day s
                JOIN read_parquet(?) g USING (field_id)
                JOIN hazards h USING (field_id)
                WHERE NOT (
                    g.max_lon < ? OR g.min_lon > ? OR g.max_lat < ? OR g.min_lat > ?
                )
                {' '.join(clauses)}
            )
            SELECT * EXCLUDE (_rank) FROM eligible WHERE _rank = 1
            ORDER BY field_max_risk_rank DESC NULLS LAST, field_id
        """
        query_params: list[Any] = [
            str(self.v4_field_state_path), calendar_date, str(self.geometry_path),
            min_lon, max_lon, min_lat, max_lat, *params,
        ]
        if effective_limit is not None:
            sql += "\nLIMIT ?"
            query_params.append(effective_limit + 1)
        rows = _records(connection.execute(sql, query_params).fetchdf())
        source_count = int(rows[0].get("_source_count") or 0) if rows else 0
        truncated = effective_limit is not None and source_count > effective_limit
        if effective_limit is not None:
            rows = rows[:effective_limit]
        features: list[dict[str, Any]] = []
        for row in rows:
            raw_geometry = row.get("geometry_geojson")
            geometry = (
                json.loads(raw_geometry)
                if isinstance(raw_geometry, str) else dict(raw_geometry or {})
            )
            if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
                raise RuntimeError("V4 field geometry must be Polygon or MultiPolygon")
            geom_bbox = [
                float(row.get("min_lon")), float(row.get("min_lat")),
                float(row.get("max_lon")), float(row.get("max_lat")),
            ]
            properties = {
                key: value for key, value in row.items()
                if key not in {
                    "geometry_geojson", "geometry_format", "min_lon", "min_lat",
                    "max_lon", "max_lat", "_source_count",
                }
            }
            properties["bbox"] = geom_bbox
            properties["timeline_bucket"] = calendar_date
            features.append(
                {"type": "Feature", "geometry": geometry, "properties": properties}
            )
        return features, source_count, truncated

    def v4_frame_state(
        self,
        *,
        calendar_date: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return bounded V4 dynamic rows for hydration with cached geometry."""
        self.require_incident_v4()
        started = time.perf_counter()
        requested_limit = limit
        effective_limit = _feature_limit(limit, self.settings.max_feature_limit)
        clean_filters = _clean_incident_filters(filters)
        clauses: list[str] = []
        params: list[Any] = []
        for key, expression in {
            "crop_name": "s.crop_name",
            "hazard_family": "s.hazard_family",
            "stage_bucket": "s.stage_bucket",
        }.items():
            if key in clean_filters:
                clauses.append(f"AND CAST({expression} AS VARCHAR) = ?")
                params.append(clean_filters[key])
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
        sql = f"""
            WITH state_day AS (
                SELECT * FROM read_parquet(?)
                WHERE CAST(calendar_date AS DATE) = CAST(? AS DATE)
            ), hazards AS (
                SELECT field_id,
                    STRING_AGG(
                        DISTINCT CAST(hazard_family AS VARCHAR), ','
                        ORDER BY CAST(hazard_family AS VARCHAR)
                    ) FILTER (WHERE pressure_active) AS active_hazards,
                    MAX(risk_rank) FILTER (WHERE pressure_observed)
                        AS field_max_risk_rank,
                    ARG_MAX(hazard_family, risk_rank) FILTER (WHERE pressure_observed)
                        AS highest_pressure_hazard
                FROM state_day GROUP BY field_id
            ), eligible AS (
                SELECT s.*, h.active_hazards, h.field_max_risk_rank,
                    h.highest_pressure_hazard,
                    COUNT(DISTINCT s.field_id) OVER () AS _source_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.field_id
                        ORDER BY s.pressure_active DESC, s.risk_rank DESC,
                            s.monitored DESC, s.evaluable DESC,
                            s.s2_knowledge_date DESC NULLS LAST,
                            s.crop_instance_id, s.hazard_family
                    ) AS _rank
                FROM state_day AS s
                JOIN read_parquet(?) AS g USING (field_id)
                JOIN hazards AS h USING (field_id)
                WHERE 1 = 1
                {bbox_clause}
                {' '.join(clauses)}
            )
            SELECT * EXCLUDE (_rank) FROM eligible WHERE _rank = 1
            ORDER BY field_max_risk_rank DESC NULLS LAST, field_id
        """
        query_params: list[Any] = [
            str(self.v4_field_state_path), calendar_date, str(self.geometry_path),
            *bbox_params, *params,
        ]
        if effective_limit is not None:
            sql += "\nLIMIT ?"
            query_params.append(effective_limit + 1)
        with duckdb.connect(":memory:") as con:
            source_rows = _records(con.execute(sql, query_params).fetchdf())
        source_count = int(source_rows[0].get("_source_count") or 0) if source_rows else 0
        render_rows = source_rows[:effective_limit] if effective_limit is not None else source_rows
        rows: list[dict[str, Any]] = []
        for source in render_rows:
            row = {key: value for key, value in source.items() if key != "_source_count"}
            row.update(
                {
                    "timeline_bucket": calendar_date,
                    "current_risk_band": row.get("risk_band"),
                    "max_risk_band": row.get("risk_band"),
                    "current_risk_rank": row.get("risk_rank"),
                    "max_risk_rank": row.get("risk_rank"),
                    "hazard_signature": row.get("highest_pressure_hazard")
                        or row.get("hazard_family"),
                    "response_signature": row.get("response_class"),
                }
            )
            rows.append(row)
        truncated = effective_limit is not None and source_count > effective_limit
        return {
            "geometry_version": self.geometry_version(),
            "rows": rows,
            "meta": {
                "mode": "crop_incident_v4_dual_clock",
                "calendar_date": calendar_date,
                "timeline_bucket": calendar_date,
                "state_count": len(rows),
                "source_row_count": source_count,
                "query_row_count": len(source_rows),
                "limit": effective_limit,
                "requested_limit": requested_limit,
                "truncated": truncated,
                "limit_hit": truncated,
                "bbox_applied": bbox is not None,
                "filters": clean_filters,
                "transport": "compact_v4_state_plus_cached_geometry",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        }

    def v4_field_detail(
        self,
        field_id: str,
        *,
        as_of_date: str | None,
        crop_instance_id: str | None = None,
        lookback_days: int = V4_INCIDENT_DEFAULT_LOOKBACK_DAYS,
        history_limit: int = V4_INCIDENT_DEFAULT_HISTORY_LIMIT,
    ) -> dict[str, Any]:
        """Return three bounded causal evidence lanes for one mapped field."""
        self.require_incident_v4()
        lookback_days = max(1, min(int(lookback_days), V4_INCIDENT_MAX_LOOKBACK_DAYS))
        history_limit = max(1, min(int(history_limit), V4_INCIDENT_MAX_HISTORY_LIMIT))
        crop_instance_id = str(crop_instance_id or "").strip() or None
        with duckdb.connect(":memory:") as con:
            exists = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM read_parquet(?)
                    WHERE CAST(field_id AS VARCHAR) = ?
                    """,
                    [str(self.geometry_path), field_id],
                ).fetchone()[0]
            )
            if not exists:
                raise ResourceNotFoundError("Field was not found.")
            if as_of_date is None:
                value = con.execute(
                    "SELECT MAX(calendar_date) FROM read_parquet(?)",
                    [str(self.v4_timeline_path)],
                ).fetchone()[0]
                if value is None:
                    raise ResourceNotFoundError("Incident V4 timeline is empty.")
                as_of_date = str(value)[:10]
            as_of_date = _parse_iso_date_segment(str(as_of_date))
            history_start = (
                date.fromisoformat(as_of_date) - timedelta(days=lookback_days - 1)
            ).isoformat()

            state_crop_clause = ""
            state_crop_params: list[Any] = []
            if crop_instance_id:
                state_crop_clause = "AND CAST(s.crop_instance_id AS VARCHAR) = ?"
                state_crop_params.append(crop_instance_id)
            state_date_clause = (
                "AND CAST(s.calendar_date AS DATE) <= CAST(? AS DATE)"
                if crop_instance_id
                else "AND CAST(s.calendar_date AS DATE) = CAST(? AS DATE)"
            )
            current_rows = _records(
                con.execute(
                    f"""
                    SELECT s.* FROM read_parquet(?) AS s
                    WHERE CAST(s.field_id AS VARCHAR) = ?
                      {state_date_clause}
                      {state_crop_clause}
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY s.crop_instance_id, s.hazard_family
                        ORDER BY s.calendar_date DESC, s.risk_rank DESC
                    ) = 1
                    ORDER BY s.crop_instance_id, s.hazard_family
                    LIMIT ?
                    """,
                    [
                        str(self.v4_field_state_path), field_id, as_of_date,
                        *state_crop_params, history_limit + 1,
                    ],
                ).fetchdf()
            )
            current_truncated = len(current_rows) > history_limit
            current_rows = current_rows[:history_limit]

            membership_knowledge = (
                "COALESCE(TRY_CAST(m.knowledge_time AS DATE), "
                "TRY_CAST(m.timeline_bucket AS DATE))"
                if "knowledge_time" in self.incident_membership_columns
                else "TRY_CAST(m.timeline_bucket AS DATE)"
            )
            membership_crop_clause = ""
            membership_crop_params: list[Any] = []
            if crop_instance_id:
                membership_crop_clause = "AND CAST(m.crop_instance_id AS VARCHAR) = ?"
                membership_crop_params.append(crop_instance_id)
            pressure_crop_clause = ""
            pressure_crop_params: list[Any] = []
            if crop_instance_id:
                pressure_crop_clause = "AND CAST(p.crop_instance_id AS VARCHAR) = ?"
                pressure_crop_params.append(crop_instance_id)
            pressure_desc = con.execute(
                f"""
                WITH memberships AS (
                    SELECT DISTINCT
                        CAST(m.crop_instance_id AS VARCHAR) AS crop_instance_id,
                        LOWER(CAST(m.hazard_family AS VARCHAR)) AS hazard_family,
                        CAST(m.timeline_bucket AS DATE) AS membership_week,
                        CAST(m.incident_id AS VARCHAR) AS incident_id
                    FROM read_parquet(?) AS m
                    WHERE CAST(m.field_id AS VARCHAR) = ?
                      AND CAST(m.timeline_bucket AS DATE) <= CAST(? AS DATE)
                      AND CAST(m.timeline_bucket AS DATE) + INTERVAL 6 DAY
                            >= CAST(? AS DATE)
                      AND {membership_knowledge} <= CAST(? AS DATE)
                      {membership_crop_clause}
                ), pressure_candidates AS (
                    SELECT
                        s.calendar_date,
                        s.pressure_observation_date,
                        CAST(s.pressure_observation_date AS DATE)
                            AS pressure_effective_date,
                        s.pressure_knowledge_time,
                        s.weather_available_at,
                        s.field_id, s.crop_instance_id, s.hazard_family,
                        s.risk_rank, s.risk_band, s.pressure_score,
                        s.pressure_observed, s.pressure_active
                    FROM read_parquet(?) AS s
                    WHERE CAST(s.field_id AS VARCHAR) = ?
                      AND CAST(s.pressure_observation_date AS DATE)
                          BETWEEN CAST(? AS DATE)
                          AND CAST(? AS DATE)
                      AND CAST(s.pressure_knowledge_time AS TIMESTAMP)
                            < CAST(? AS DATE) + INTERVAL 1 DAY
                      AND COALESCE(TRY_CAST(s.pressure_observed AS BOOLEAN), FALSE)
                      {state_crop_clause}
                    UNION ALL
                    SELECT
                        p.calendar_date, p.pressure_observation_date,
                        p.pressure_effective_date, p.pressure_knowledge_time,
                        p.weather_available_at, p.field_id, p.crop_instance_id,
                        p.hazard_family, p.risk_rank, p.risk_band,
                        p.pressure_score, p.pressure_observed, p.pressure_active
                    FROM read_parquet(?) AS p
                    WHERE CAST(p.field_id AS VARCHAR) = ?
                      AND CAST(p.pressure_effective_date AS DATE)
                          BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                      AND CAST(p.pressure_knowledge_time AS TIMESTAMP)
                            < CAST(? AS DATE) + INTERVAL 1 DAY
                      {pressure_crop_clause}
                ), pressure_rows AS (
                    SELECT * FROM pressure_candidates AS p
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY p.pressure_effective_date, p.crop_instance_id,
                            p.hazard_family
                        ORDER BY p.pressure_knowledge_time DESC,
                            p.pressure_active DESC, p.risk_rank DESC
                    ) = 1
                )
                SELECT p.*,
                    STRING_AGG(DISTINCT m.incident_id, ',' ORDER BY m.incident_id)
                        AS attributed_incident_ids
                FROM pressure_rows AS p
                LEFT JOIN memberships AS m
                  ON CAST(p.crop_instance_id AS VARCHAR) = m.crop_instance_id
                 AND LOWER(CAST(p.hazard_family AS VARCHAR)) = m.hazard_family
                 AND CAST(p.calendar_date AS DATE) BETWEEN m.membership_week
                     AND m.membership_week + INTERVAL 6 DAY
                GROUP BY ALL
                ORDER BY p.calendar_date DESC, p.hazard_family DESC,
                    p.crop_instance_id DESC
                LIMIT ?
                """,
                [
                    str(self.incident_membership_path), field_id, as_of_date,
                    history_start, as_of_date, *membership_crop_params,
                    str(self.v4_field_state_path), field_id, history_start,
                    as_of_date, as_of_date, *state_crop_params,
                    str(self.v4_pressure_observations_path), field_id,
                    history_start, as_of_date, as_of_date, *pressure_crop_params,
                    history_limit + 1,
                ],
            ).fetchdf()
            daily_pressure, pressure_truncated = _bounded_recent_records(
                pressure_desc, history_limit
            )

            attempt_crop_clause = ""
            attempt_crop_params: list[Any] = []
            if crop_instance_id:
                attempt_crop_clause = "AND CAST(a.crop_instance_id AS VARCHAR) = ?"
                attempt_crop_params.append(crop_instance_id)
            attempt_desc = con.execute(
                f"""
                SELECT a.* FROM read_parquet(?) AS a
                WHERE CAST(a.field_id AS VARCHAR) = ?
                  AND CAST(a.knowledge_date AS DATE) BETWEEN CAST(? AS DATE)
                      AND CAST(? AS DATE)
                  {attempt_crop_clause}
                ORDER BY a.knowledge_date DESC, a.spectral_source_date DESC NULLS LAST,
                    a.crop_instance_id DESC
                LIMIT ?
                """,
                [
                    str(self.v4_s2_attempts_path), field_id, history_start,
                    as_of_date, *attempt_crop_params, history_limit + 1,
                ],
            ).fetchdf()
            s2_attempts, attempts_truncated = _bounded_recent_records(
                attempt_desc, history_limit
            )

            story_desc = con.execute(
                f"""
                WITH memberships AS (
                    SELECT
                        CAST(m.incident_id AS VARCHAR) AS incident_id,
                        CAST(m.exposure_id AS VARCHAR) AS exposure_id,
                        CAST(m.crop_instance_id AS VARCHAR) AS crop_instance_id,
                        LOWER(CAST(m.hazard_family AS VARCHAR)) AS hazard_family,
                        CAST(m.timeline_bucket AS DATE) AS story_week,
                        CAST({membership_knowledge} AS DATE) AS membership_known_date,
                        CAST(m.stage_bucket AS VARCHAR) AS stage_bucket,
                        CAST(m.membership_role AS VARCHAR) AS membership_role,
                        CAST(m.event_state AS VARCHAR) AS field_event_state,
                        ROW_NUMBER() OVER (
                            PARTITION BY m.incident_id, m.crop_instance_id,
                                m.hazard_family, CAST(m.timeline_bucket AS DATE)
                            ORDER BY {membership_knowledge} DESC,
                                CAST(m.episode_id AS VARCHAR)
                        ) AS _membership_rank
                    FROM read_parquet(?) AS m
                    WHERE CAST(m.field_id AS VARCHAR) = ?
                      AND {membership_knowledge} <= CAST(? AS DATE)
                      AND CAST(m.timeline_bucket AS DATE) <= CAST(? AS DATE)
                      {membership_crop_clause}
                ), joined AS (
                    SELECT m.* EXCLUDE (_membership_rank),
                        CAST(c.story_known_date AS DATE) AS story_known_date,
                        CAST(c.incident_state AS VARCHAR) AS incident_state,
                        CAST(c.stage_distribution AS VARCHAR) AS stage_distribution,
                        COALESCE(TRY_CAST(c.right_censored AS BOOLEAN), FALSE)
                            AS right_censored,
                        ROW_NUMBER() OVER (
                            PARTITION BY m.incident_id, m.crop_instance_id,
                                m.hazard_family
                            ORDER BY c.story_known_date DESC, c.story_week DESC
                        ) AS _latest_rank
                    FROM memberships AS m
                    JOIN read_parquet(?) AS c
                      ON CAST(c.incident_id AS VARCHAR) = m.incident_id
                     AND CAST(c.story_week AS DATE) = m.story_week
                    WHERE m._membership_rank = 1
                      AND CAST(c.story_known_date AS DATE) <= CAST(? AS DATE)
                )
                SELECT * EXCLUDE (_latest_rank) FROM joined
                WHERE CAST(story_known_date AS DATE) >= CAST(? AS DATE)
                   OR _latest_rank = 1
                ORDER BY story_known_date DESC, story_week DESC,
                    incident_id DESC, hazard_family DESC
                LIMIT ?
                """,
                [
                    str(self.incident_membership_path), field_id, as_of_date,
                    as_of_date, *membership_crop_params,
                    str(self.v4_story_checkpoints_path), as_of_date, history_start,
                    history_limit + 1,
                ],
            ).fetchdf()
            story_checkpoints, story_truncated = _bounded_recent_records(
                story_desc, history_limit
            )

        truncated = {
            "current_state": current_truncated,
            "daily_pressure": pressure_truncated,
            "s2_attempts": attempts_truncated,
            "story_checkpoints": story_truncated,
        }
        returned = {
            "current_state": len(current_rows),
            "daily_pressure": len(daily_pressure),
            "s2_attempts": len(s2_attempts),
            "story_checkpoints": len(story_checkpoints),
        }
        return {
            "field_id": field_id,
            "crop_instance_id": crop_instance_id,
            "as_of_date": as_of_date,
            "current_state_scope": (
                "explicit_crop_latest_known_state"
                if crop_instance_id else "crops_active_on_as_of_date"
            ),
            "current_state": current_rows,
            "daily_pressure": daily_pressure,
            "s2_attempts": s2_attempts,
            "story_checkpoints": story_checkpoints,
            "lanes": {
                "daily_pressure": daily_pressure,
                "s2_attempts": s2_attempts,
                "story_checkpoints": story_checkpoints,
            },
            "history": {
                "contract": "bounded_causal_field_evidence_v1",
                "window_start": history_start,
                "window_end": as_of_date,
                "lookback_days": lookback_days,
                "history_limit_per_collection": history_limit,
                "truncated": truncated,
                "any_truncated": any(truncated.values()),
                "returned": returned,
            },
            "clocks": {
                "pressure_cutoff": as_of_date,
                "s2_knowledge_cutoff": as_of_date,
                "story_knowledge_cutoff": as_of_date,
            },
        }

    def v4_incident_detail(
        self,
        incident_id: str,
        *,
        as_of_date: str | None,
        lookback_days: int = V4_INCIDENT_DEFAULT_LOOKBACK_DAYS,
        history_limit: int = V4_INCIDENT_DEFAULT_HISTORY_LIMIT,
    ) -> dict[str, Any]:
        """Return one bounded incident history using only evidence known as-of."""
        self.require_incident_v4()
        lookback_days = max(1, min(int(lookback_days), V4_INCIDENT_MAX_LOOKBACK_DAYS))
        history_limit = max(1, min(int(history_limit), V4_INCIDENT_MAX_HISTORY_LIMIT))
        with duckdb.connect(":memory:") as con:
            if as_of_date is None:
                value = con.execute(
                    "SELECT MAX(calendar_date) FROM read_parquet(?)",
                    [str(self.v4_timeline_path)],
                ).fetchone()[0]
                if value is None:
                    raise ResourceNotFoundError("Incident V4 timeline is empty.")
                as_of_date = str(value)[:10]
            as_of_date = _parse_iso_date_segment(str(as_of_date))
            history_start = (
                date.fromisoformat(as_of_date) - timedelta(days=lookback_days - 1)
            ).isoformat()

            base_windows = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                    LIMIT 2
                    """,
                    [str(self.incident_windows_path), incident_id],
                ).fetchdf()
            )
            if not base_windows:
                raise ResourceNotFoundError("Incident was not found.")
            if len(base_windows) != 1:
                raise RuntimeError("Incident V3 windows are not unique by incident_id")

            current_rows = _records(
                con.execute(
                    """
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                      AND CAST(story_known_date AS DATE) <= CAST(? AS DATE)
                    ORDER BY story_known_date DESC, story_week DESC
                    LIMIT 1
                    """,
                    [str(self.v4_story_checkpoints_path), incident_id, as_of_date],
                ).fetchdf()
            )
            if not current_rows:
                raise ResourceNotFoundError(
                    "Incident was not known by the requested as-of date."
                )
            current_checkpoint = current_rows[0]
            gap_count_sql = (
                "COUNT(*) FILTER (WHERE COALESCE(TRY_CAST(is_data_gap AS BOOLEAN), FALSE))"
                if "is_data_gap" in self.v4_story_checkpoint_columns
                else "0"
            )
            checkpoint_summary = con.execute(
                f"""
                SELECT MIN(CAST(story_week AS DATE)) AS first_story_week,
                    COUNT(*)::BIGINT AS observed_week_count,
                    ({gap_count_sql})::BIGINT AS data_gap_count
                FROM read_parquet(?)
                WHERE CAST(incident_id AS VARCHAR) = ?
                  AND CAST(story_known_date AS DATE) <= CAST(? AS DATE)
                """,
                [str(self.v4_story_checkpoints_path), incident_id, as_of_date],
            ).fetchone()

            checkpoint_desc = con.execute(
                """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        ORDER BY story_known_date DESC, story_week DESC
                    ) AS _rank
                    FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                      AND CAST(story_known_date AS DATE) <= CAST(? AS DATE)
                )
                SELECT * EXCLUDE (_rank) FROM ranked
                WHERE CAST(story_known_date AS DATE) >= CAST(? AS DATE)
                   OR _rank = 1
                ORDER BY story_known_date DESC, story_week DESC
                LIMIT ?
                """,
                [
                    str(self.v4_story_checkpoints_path), incident_id, as_of_date,
                    history_start, history_limit + 1,
                ],
            ).fetchdf()
            checkpoints, checkpoints_truncated = _bounded_recent_records(
                checkpoint_desc, history_limit
            )

            stage_desc = con.execute(
                """
                WITH ranked AS (
                    SELECT story_week, story_known_date, ROW_NUMBER() OVER (
                        ORDER BY story_known_date DESC, story_week DESC
                    ) AS _rank
                    FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                      AND CAST(story_known_date AS DATE) <= CAST(? AS DATE)
                ), causal_weeks AS (
                    SELECT DISTINCT CAST(story_week AS DATE) AS story_week
                    FROM ranked
                    WHERE CAST(story_known_date AS DATE) >= CAST(? AS DATE)
                       OR _rank = 1
                )
                SELECT s.* FROM read_parquet(?) AS s
                JOIN causal_weeks c
                  ON CAST(s.timeline_bucket AS DATE) = c.story_week
                WHERE CAST(s.incident_id AS VARCHAR) = ?
                ORDER BY s.timeline_bucket DESC, s.stage_bucket DESC
                LIMIT ?
                """,
                [
                    str(self.v4_story_checkpoints_path), incident_id, as_of_date,
                    history_start, str(self.incident_stage_summary_path), incident_id,
                    history_limit + 1,
                ],
            ).fetchdf()
            stage_rows, stage_truncated = _bounded_recent_records(
                stage_desc, history_limit
            )

            footprint_desc = con.execute(
                """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        ORDER BY story_known_date DESC, story_week DESC
                    ) AS _rank
                    FROM read_parquet(?)
                    WHERE CAST(incident_id AS VARCHAR) = ?
                      AND CAST(story_known_date AS DATE) <= CAST(? AS DATE)
                )
                SELECT * EXCLUDE (_rank) FROM ranked
                WHERE CAST(story_known_date AS DATE) >= CAST(? AS DATE)
                   OR _rank = 1
                ORDER BY story_known_date DESC, story_week DESC
                LIMIT ?
                """,
                [
                    str(self.v4_story_footprints_path), incident_id, as_of_date,
                    history_start, history_limit + 1,
                ],
            ).fetchdf()
            footprints, footprints_truncated = _bounded_recent_records(
                footprint_desc, history_limit
            )

            membership_knowledge = (
                "COALESCE(TRY_CAST(m.knowledge_time AS DATE), "
                "TRY_CAST(m.timeline_bucket AS DATE))"
                if "knowledge_time" in self.incident_membership_columns
                else "TRY_CAST(m.timeline_bucket AS DATE)"
            )
            membership_cte = f"""
                members AS (
                    SELECT DISTINCT
                        CAST(m.field_id AS VARCHAR) AS field_id,
                        CAST(m.crop_instance_id AS VARCHAR) AS crop_instance_id,
                        CAST(m.timeline_bucket AS DATE) AS membership_week,
                        LOWER(CAST(m.hazard_family AS VARCHAR)) AS hazard_family
                    FROM read_parquet(?) AS m
                    WHERE CAST(m.incident_id AS VARCHAR) = ?
                      AND CAST(m.timeline_bucket AS DATE) <= CAST(? AS DATE)
                      AND CAST(m.timeline_bucket AS DATE) + INTERVAL 6 DAY
                            >= CAST(? AS DATE)
                      AND {membership_knowledge} <= CAST(? AS DATE)
                ), member_fields AS (
                    SELECT DISTINCT field_id, crop_instance_id, membership_week
                    FROM members
                )
            """
            member_params: list[Any] = [
                str(self.incident_membership_path), incident_id, as_of_date,
                history_start, as_of_date,
            ]

            s2_update_desc = con.execute(
                f"""
                WITH {membership_cte}
                SELECT a.* FROM read_parquet(?) AS a
                JOIN member_fields m
                  ON CAST(a.field_id AS VARCHAR) = m.field_id
                 AND CAST(a.crop_instance_id AS VARCHAR) = m.crop_instance_id
                 AND CAST(a.knowledge_date AS DATE) BETWEEN m.membership_week
                     AND m.membership_week + INTERVAL 6 DAY
                WHERE CAST(a.knowledge_date AS DATE) BETWEEN CAST(? AS DATE)
                    AND CAST(? AS DATE)
                ORDER BY a.knowledge_date DESC, a.field_id DESC,
                    a.crop_instance_id DESC, a.spectral_source_date DESC
                LIMIT ?
                """,
                [
                    *member_params, str(self.v4_s2_updates_path), history_start,
                    as_of_date, history_limit + 1,
                ],
            ).fetchdf()
            s2_updates, s2_updates_truncated = _bounded_recent_records(
                s2_update_desc, history_limit
            )

            s2_attempt_desc = con.execute(
                f"""
                WITH {membership_cte}
                SELECT a.* FROM read_parquet(?) AS a
                JOIN member_fields m
                  ON CAST(a.field_id AS VARCHAR) = m.field_id
                 AND CAST(a.crop_instance_id AS VARCHAR) = m.crop_instance_id
                 AND CAST(a.knowledge_date AS DATE) BETWEEN m.membership_week
                     AND m.membership_week + INTERVAL 6 DAY
                WHERE CAST(a.knowledge_date AS DATE) BETWEEN CAST(? AS DATE)
                    AND CAST(? AS DATE)
                ORDER BY a.knowledge_date DESC, a.field_id DESC,
                    a.crop_instance_id DESC, a.spectral_source_date DESC NULLS LAST
                LIMIT ?
                """,
                [
                    *member_params, str(self.v4_s2_attempts_path), history_start,
                    as_of_date, history_limit + 1,
                ],
            ).fetchdf()
            s2_attempts, s2_attempts_truncated = _bounded_recent_records(
                s2_attempt_desc, history_limit
            )

            pressure_desc = con.execute(
                f"""
                WITH {membership_cte}
                SELECT s.calendar_date,
                    COUNT(DISTINCT s.field_id)::BIGINT AS monitored_field_count,
                    COUNT(DISTINCT s.field_id) FILTER (
                        WHERE s.pressure_observed
                    )::BIGINT AS pressure_field_count,
                    MAX(s.risk_rank)::INTEGER AS max_risk_rank,
                    LOWER(CAST(s.hazard_family AS VARCHAR)) AS hazard_family
                FROM read_parquet(?) AS s
                JOIN members m
                 ON CAST(s.field_id AS VARCHAR) = m.field_id
                 AND CAST(s.crop_instance_id AS VARCHAR) = m.crop_instance_id
                 AND LOWER(CAST(s.hazard_family AS VARCHAR)) = m.hazard_family
                 AND CAST(s.calendar_date AS DATE) BETWEEN m.membership_week
                     AND m.membership_week + INTERVAL 6 DAY
                WHERE CAST(s.calendar_date AS DATE) BETWEEN CAST(? AS DATE)
                    AND CAST(? AS DATE)
                GROUP BY s.calendar_date, s.hazard_family
                ORDER BY s.calendar_date DESC, s.hazard_family DESC
                LIMIT ?
                """,
                [
                    *member_params, str(self.v4_field_state_path), history_start,
                    as_of_date, history_limit + 1,
                ],
            ).fetchdf()
            daily_pressure, pressure_truncated = _bounded_recent_records(
                pressure_desc, history_limit
            )

            lineage_time_column = next(
                (
                    name for name in (
                        "knowledge_time", "story_known_date", "timeline_bucket",
                        "effective_date", "lineage_date",
                    )
                    if name in self.incident_lineage_columns
                ),
                None,
            )
            incoming: list[dict[str, Any]] = []
            outgoing: list[dict[str, Any]] = []
            incoming_truncated = False
            outgoing_truncated = False
            if lineage_time_column and {
                "parent_incident_id", "child_incident_id"
            }.issubset(self.incident_lineage_columns):
                time_sql = f'TRY_CAST("{lineage_time_column}" AS DATE)'
                incoming_desc = con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(child_incident_id AS VARCHAR) = ?
                      AND {time_sql} BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                    ORDER BY {time_sql} DESC, CAST(parent_incident_id AS VARCHAR) DESC
                    LIMIT ?
                    """,
                    [
                        str(self.incident_lineage_path), incident_id, history_start,
                        as_of_date, history_limit + 1,
                    ],
                ).fetchdf()
                incoming, incoming_truncated = _bounded_recent_records(
                    incoming_desc, history_limit
                )
                outgoing_desc = con.execute(
                    f"""
                    SELECT * FROM read_parquet(?)
                    WHERE CAST(parent_incident_id AS VARCHAR) = ?
                      AND {time_sql} BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                    ORDER BY {time_sql} DESC, CAST(child_incident_id AS VARCHAR) DESC
                    LIMIT ?
                    """,
                    [
                        str(self.incident_lineage_path), incident_id, history_start,
                        as_of_date, history_limit + 1,
                    ],
                ).fetchdf()
                outgoing, outgoing_truncated = _bounded_recent_records(
                    outgoing_desc, history_limit
                )

        for row in footprints:
            geometry, _ = _precomputed_geojson_and_bbox(
                row.pop("geometry_geojson", None),
                row.get("geometry_type"),
                (
                    row.get("min_lon"), row.get("min_lat"),
                    row.get("max_lon"), row.get("max_lat"),
                ),
            )
            row["geometry"] = geometry
            for role in ("pressure", "impact", "watch"):
                raw = row.pop(f"{role}_geometry_geojson", None)
                row[f"{role}_geometry"] = (
                    _geometry_to_geojson_and_bbox(raw, "geojson")[0]
                    if _has_geometry_value(raw) else None
                )

        first_story_week = str(checkpoint_summary[0])[:10]
        causal_window = _causal_v4_incident_window(
            base_windows[0],
            current_checkpoint,
            as_of_date=as_of_date,
            first_story_week=first_story_week,
            observed_week_count=int(checkpoint_summary[1] or 0),
            data_gap_count=int(checkpoint_summary[2] or 0),
            split_count=sum(
                1 for row in outgoing
                if str(row.get("lineage_type") or "").lower() == "split"
            ),
            merge_count=sum(
                1 for row in incoming
                if str(row.get("lineage_type") or "").lower() == "merge"
            ),
        )
        truncated = {
            "weekly_state": checkpoints_truncated,
            "stage_summary": stage_truncated,
            "footprints": footprints_truncated,
            "daily_pressure": pressure_truncated,
            "s2_updates": s2_updates_truncated,
            "s2_attempts": s2_attempts_truncated,
            "lineage_incoming": incoming_truncated,
            "lineage_outgoing": outgoing_truncated,
        }
        returned = {
            "weekly_state": len(checkpoints),
            "stage_summary": len(stage_rows),
            "footprints": len(footprints),
            "daily_pressure": len(daily_pressure),
            "s2_updates": len(s2_updates),
            "s2_attempts": len(s2_attempts),
            "lineage_incoming": len(incoming),
            "lineage_outgoing": len(outgoing),
        }
        return {
            "incident_id": incident_id,
            "window": causal_window,
            "as_of_date": as_of_date,
            "weekly_state": checkpoints,
            "stage_summary": stage_rows,
            "stage_rows": stage_rows,
            "footprints": footprints,
            "lineage": {"incoming": incoming, "outgoing": outgoing},
            "daily_pressure": daily_pressure,
            "s2_updates": s2_updates,
            "s2_attempts": s2_attempts,
            "history": {
                "contract": "bounded_causal_incident_history_v1",
                "window_start": history_start,
                "window_end": as_of_date,
                "lookback_days": lookback_days,
                "history_limit_per_collection": history_limit,
                "current_checkpoint_always_included": True,
                "current_checkpoint_outside_lookback": (
                    str(current_checkpoint.get("story_known_date") or "")[:10]
                    < history_start
                ),
                "membership_scope": (
                    "evidence_knowledge_date_must_fall_in_a_causally_known_"
                    "incident_membership_week"
                ),
                "lineage_time_field": lineage_time_column,
                "undated_lineage_omitted": lineage_time_column is None,
                "truncated": truncated,
                "any_truncated": any(truncated.values()),
                "returned": returned,
            },
            "clocks": {
                "pressure_as_of_date": as_of_date,
                "crop_knowledge_cutoff": as_of_date,
                "story_knowledge_cutoff": as_of_date,
            },
        }

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
        if self.has_incident_v3() and not self.timeline_summary_mapping:
            activity = self.activity({})
            buckets = activity["buckets"]
            LOGGER.info(
                "timeline_loaded source=incident_artifacts buckets=%s "
                "first_bucket=%s last_bucket=%s elapsed_ms=%.1f",
                len(buckets),
                buckets[0]["timeline_bucket"] if buckets else None,
                buckets[-1]["timeline_bucket"] if buckets else None,
                (time.perf_counter() - started) * 1000,
            )
            return {
                "buckets": buckets,
                "source": "incident_artifacts",
                "activity_unit": "incident_stories",
            }
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
        if self.has_incident_v3():
            for bucket in buckets:
                incident_count = int(bucket.get("story_cluster_count") or 0)
                bucket["incident_count"] = incident_count
                bucket["activity_count"] = incident_count
                bucket["activity_unit"] = "incident_stories"
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
        filters = self._effective_frame_filters(filters)
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
        optional_frame_select = ",\n                ".join(
            self._optional_frame_sql("f", name, fallback)
            for name, fallback in (
                ("incident_id", "f.story_cluster_id"),
                ("crop_name", "NULL"),
                ("stage_bucket", "NULL"),
                ("incident_state", "NULL"),
                ("event_id", "NULL"),
                ("event_state", "NULL"),
                ("motif_id", "NULL"),
                ("archetype_display_state", "NULL"),
                ("anchor_status", "NULL"),
                ("accepted", "NULL"),
                ("current_risk_band", "f.max_risk_band"),
                ("current_risk_rank", "f.max_risk_rank"),
                ("assignment_distance", "NULL"),
                ("distance_ratio", "NULL"),
                ("assignment_reason", "NULL"),
                ("motif_model_version", "NULL"),
            )
        )
        current_rank_sql = (
            "COALESCE(TRY_CAST(f.current_risk_rank AS INTEGER), "
            "TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
            if "current_risk_rank" in self.frame_columns
            else "COALESCE(TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
        )
        state_sql = (
            "UPPER(COALESCE(CAST(f.event_state AS VARCHAR), ''))"
            if "event_state" in self.frame_columns
            else "''"
        )
        state_priority_sql = f"""CASE {state_sql}
            WHEN 'SEVERE' THEN 6 WHEN 'ACTIVE' THEN 5 WHEN 'WATCH' THEN 4
            WHEN 'RECOVERING' THEN 3 WHEN 'QUIET_PENDING' THEN 2
            WHEN 'DATA_GAP' THEN 1 ELSE 0 END"""

        sql = f"""
            SELECT
                f.timeline_bucket,
                f.field_id,
                f.story_cluster_id,
                {optional_frame_select},
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
                COUNT(DISTINCT f.field_id) OVER () AS _source_row_count,
                COUNT(DISTINCT f.story_cluster_id) OVER () AS _source_story_cluster_count,
                SUM(f.reportable_day_count) OVER () AS _source_reportable_day_count,
                SUM(f.event_count) OVER () AS _source_event_count,
                COUNT(*) OVER (PARTITION BY f.field_id) AS concurrent_event_count
            FROM read_parquet(?) AS f
            JOIN read_parquet(?) AS g USING (field_id)
            {label_join}
            WHERE f.timeline_bucket = ?
            {filter_clause}
            {bbox_clause}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY f.field_id
                ORDER BY {state_priority_sql} DESC, {current_rank_sql} DESC,
                    f.max_risk_rank DESC, f.reportable_day_count DESC,
                    CAST(f.story_cluster_id AS VARCHAR)
            ) = 1
            ORDER BY {state_priority_sql} DESC, {current_rank_sql} DESC,
                f.reportable_day_count DESC, f.field_id
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
                    "incident_id",
                    "crop_name",
                    "stage_bucket",
                    "incident_state",
                    "event_id",
                    "event_state",
                    "motif_id",
                    "archetype_display_state",
                    "anchor_status",
                    "accepted",
                    "max_risk_band",
                    "current_risk_band",
                    "hazard_signature",
                    "response_signature",
                    "reportable_day_count",
                    "event_count",
                    "max_risk_rank",
                    "current_risk_rank",
                    "response_day_count",
                    "motif_family",
                    "short_label",
                    "assignment_distance",
                    "distance_ratio",
                    "assignment_reason",
                    "motif_model_version",
                    "concurrent_event_count",
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

    def geometry_version(self) -> str:
        """Return a content-derived version for the immutable geometry artifact."""
        stat = self.geometry_path.stat()
        return _geometry_artifact_version(
            str(self.geometry_path),
            stat.st_size,
            stat.st_mtime_ns,
        )

    def frame_state(
        self,
        *,
        timeline_bucket: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Return dynamic field state without retransmitting static geometry."""
        self.require_ready()
        if not self._has_optimized_geometry():
            return self._raw_frame_state(
                timeline_bucket=timeline_bucket,
                bbox=bbox,
                limit=limit,
                filters=filters,
            )

        started = time.perf_counter()
        requested_limit = limit
        effective_limit = _feature_limit(limit, self.settings.max_feature_limit)
        filters = self._effective_frame_filters(filters)
        motif_family_sql = self._motif_family_sql("f")
        filter_clause, filter_params = _filter_sql(
            filters,
            "f",
            motif_family_sql=motif_family_sql,
        )
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
        optional_frame_select = ",\n                ".join(
            self._optional_frame_sql("f", name, fallback)
            for name, fallback in (
                ("incident_id", "f.story_cluster_id"),
                ("crop_name", "NULL"),
                ("stage_bucket", "NULL"),
                ("incident_state", "NULL"),
                ("event_id", "NULL"),
                ("event_state", "NULL"),
                ("motif_id", "NULL"),
                ("archetype_display_state", "NULL"),
                ("anchor_status", "NULL"),
                ("accepted", "NULL"),
                ("current_risk_band", "f.max_risk_band"),
                ("current_risk_rank", "f.max_risk_rank"),
                ("assignment_distance", "NULL"),
                ("distance_ratio", "NULL"),
                ("assignment_reason", "NULL"),
                ("motif_model_version", "NULL"),
            )
        )
        current_rank_sql = (
            "COALESCE(TRY_CAST(f.current_risk_rank AS INTEGER), "
            "TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
            if "current_risk_rank" in self.frame_columns
            else "COALESCE(TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
        )
        state_sql = (
            "UPPER(COALESCE(CAST(f.event_state AS VARCHAR), ''))"
            if "event_state" in self.frame_columns
            else "''"
        )
        state_priority_sql = f"""CASE {state_sql}
            WHEN 'SEVERE' THEN 6 WHEN 'ACTIVE' THEN 5 WHEN 'WATCH' THEN 4
            WHEN 'RECOVERING' THEN 3 WHEN 'QUIET_PENDING' THEN 2
            WHEN 'DATA_GAP' THEN 1 ELSE 0 END"""

        sql = f"""
            SELECT
                f.timeline_bucket,
                f.field_id,
                f.story_cluster_id,
                {optional_frame_select},
                f.max_risk_band,
                f.hazard_signature,
                f.response_signature,
                f.reportable_day_count,
                f.event_count,
                f.max_risk_rank,
                f.response_day_count,
                {motif_family_sql} AS motif_family,
                {label_select},
                COUNT(DISTINCT f.field_id) OVER () AS _source_row_count,
                COUNT(DISTINCT f.story_cluster_id) OVER () AS _source_story_cluster_count,
                SUM(f.reportable_day_count) OVER () AS _source_reportable_day_count,
                SUM(f.event_count) OVER () AS _source_event_count,
                COUNT(*) OVER (PARTITION BY f.field_id) AS concurrent_event_count
            FROM read_parquet(?) AS f
            JOIN read_parquet(?) AS g USING (field_id)
            {label_join}
            WHERE f.timeline_bucket = ?
            {filter_clause}
            {bbox_clause}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY f.field_id
                ORDER BY {state_priority_sql} DESC, {current_rank_sql} DESC,
                    f.max_risk_rank DESC, f.reportable_day_count DESC,
                    CAST(f.story_cluster_id AS VARCHAR)
            ) = 1
            ORDER BY {state_priority_sql} DESC, {current_rank_sql} DESC,
                f.reportable_day_count DESC, f.field_id
        """
        params: list[Any] = [str(self.frame_path), str(self.geometry_path)]
        if self.labels_path.exists():
            params.append(str(self.labels_path))
        params.append(timeline_bucket)
        params.extend(filter_params)
        params.extend(bbox_params)
        if effective_limit is not None:
            sql += "\nLIMIT ?"
            params.append(effective_limit + 1)

        with duckdb.connect(":memory:") as con:
            rows = con.execute(sql, params).fetchdf()

        source_rows = _records(rows)
        source_row_count = int(source_rows[0].get("_source_row_count") or 0) if source_rows else 0
        source_story_cluster_count = (
            int(source_rows[0].get("_source_story_cluster_count") or 0) if source_rows else 0
        )
        source_reportable_day_count = (
            float(source_rows[0].get("_source_reportable_day_count") or 0) if source_rows else 0
        )
        source_event_count = float(source_rows[0].get("_source_event_count") or 0) if source_rows else 0
        render_rows = source_rows[:effective_limit] if effective_limit is not None else source_rows
        states = [
            {key: row.get(key) for key in FRAME_STATE_FIELDS}
            for row in render_rows
        ]
        truncated = effective_limit is not None and source_row_count > effective_limit
        LOGGER.info(
            (
                "frame_state bucket=%s bbox=%s filters=%s requested_limit=%s "
                "effective_limit=%s source_rows=%s states=%s limit_hit=%s elapsed_ms=%.1f"
            ),
            timeline_bucket,
            _json_for_log(bbox),
            _json_for_log(filters),
            requested_limit,
            effective_limit,
            source_row_count,
            len(states),
            truncated,
            (time.perf_counter() - started) * 1000,
        )
        return {
            "geometry_version": self.geometry_version(),
            "rows": states,
            "meta": {
                "timeline_bucket": timeline_bucket,
                "state_count": len(states),
                "source_row_count": source_row_count,
                "query_row_count": len(source_rows),
                "story_cluster_count": source_story_cluster_count,
                "reportable_day_count": source_reportable_day_count,
                "event_count": source_event_count,
                "limit": effective_limit,
                "requested_limit": requested_limit,
                "unlimited": effective_limit is None,
                "limit_hit": truncated,
                "truncated": truncated,
                "optimized_geometry": True,
                "bbox_applied": bbox is not None,
                "filters": filters,
            },
        }

    def _raw_frame_state(
        self,
        *,
        timeline_bucket: str,
        bbox: tuple[float, float, float, float] | None,
        limit: int,
        filters: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Compatibility path; optimized bundles avoid this geometry parse."""
        frame = self.frame_features(
            timeline_bucket=timeline_bucket,
            bbox=bbox,
            limit=limit,
            filters=filters,
        )
        states = [
            {key: feature.get("properties", {}).get(key) for key in FRAME_STATE_FIELDS}
            for feature in frame.get("features", [])
        ]
        legacy_meta = frame.get("meta", {})
        meta = {
            key: legacy_meta.get(key)
            for key in FRAME_STATE_META_FIELDS
        }
        meta["state_count"] = len(states)
        meta["bbox_applied"] = bbox is not None
        return {
            "geometry_version": self.geometry_version(),
            "rows": states,
            "meta": meta,
        }

    def geometry_features(
        self,
        *,
        geometry_version: str,
        field_ids: list[str],
    ) -> dict[str, Any]:
        """Return static geometry for a bounded, deduplicated field-ID request."""
        self.require_ready()
        current_version = self.geometry_version()
        if geometry_version != current_version:
            raise GeometryVersionMismatchError("geometry version does not match the active artifact")
        if len(field_ids) > MAX_GEOMETRY_IDS:
            raise RequestBodyTooLargeError(
                f"field_ids may contain at most {MAX_GEOMETRY_IDS} items"
            )
        unique_ids = list(dict.fromkeys(field_ids))
        if not unique_ids:
            return {
                "geometry_version": current_version,
                "type": "FeatureCollection",
                "features": [],
                "meta": {
                    "requested_field_count": 0,
                    "feature_count": 0,
                    "missing_field_ids": [],
                },
            }

        optimized = self._has_optimized_geometry()
        geometry_select = (
            "g.geometry_geojson, 'geojson' AS geometry_format, "
            "g.min_lon, g.min_lat, g.max_lon, g.max_lat"
            if optimized
            else "g.geometry_text, g.geometry_format, NULL AS min_lon, NULL AS min_lat, "
            "NULL AS max_lon, NULL AS max_lat"
        )
        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                f"""
                SELECT
                    g.field_id,
                    {geometry_select},
                    g.district,
                    g.sector,
                    g.cell,
                    g.village
                FROM read_parquet(?) AS g
                JOIN UNNEST(?) AS wanted(field_id) USING (field_id)
                """,
                [str(self.geometry_path), unique_ids],
            ).fetchdf()

        by_field_id: dict[str, dict[str, Any]] = {}
        for row in _records(rows):
            field_id = str(row.get("field_id"))
            if optimized:
                raw_geometry = row.get("geometry_geojson")
                geometry = json.loads(raw_geometry) if isinstance(raw_geometry, str) else raw_geometry
                geom_bbox = [
                    float(row["min_lon"]),
                    float(row["min_lat"]),
                    float(row["max_lon"]),
                    float(row["max_lat"]),
                ]
            else:
                geometry, geom_bbox = _geometry_to_geojson_and_bbox(
                    row.get("geometry_text"),
                    str(row.get("geometry_format") or "geojson"),
                )
            by_field_id[field_id] = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "field_id": field_id,
                    "bbox": geom_bbox,
                    "district": row.get("district"),
                    "sector": row.get("sector"),
                    "cell": row.get("cell"),
                    "village": row.get("village"),
                },
            }

        features = [by_field_id[field_id] for field_id in unique_ids if field_id in by_field_id]
        missing = [field_id for field_id in unique_ids if field_id not in by_field_id]
        LOGGER.info(
            "geometry_features requested=%s returned=%s missing=%s optimized_geometry=%s",
            len(unique_ids),
            len(features),
            len(missing),
            optimized,
        )
        return {
            "geometry_version": current_version,
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "requested_field_count": len(unique_ids),
                "feature_count": len(features),
                "missing_field_ids": missing,
            },
        }

    def motifs(self, q: str | None, limit: int) -> dict[str, Any]:
        self.require_ready()
        limit = max(1, min(limit, 1000))
        frame_family_sql = self._motif_family_sql("f")
        label_fallback_sql = _hazard_family_sql("l.hazard_signature")
        current_risk_sql = (
            "COALESCE(current_risk_band, max_risk_band)"
            if "current_risk_band" in self.frame_columns
            else "max_risk_band"
        )
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
                f"""
                SELECT {current_risk_sql} AS current_risk_band,
                    COUNT(DISTINCT field_id) AS field_count,
                    COUNT(*) AS event_count
                FROM read_parquet(?)
                WHERE {current_risk_sql} IS NOT NULL
                GROUP BY {current_risk_sql}
                ORDER BY field_count DESC, current_risk_band
                """,
                [str(self.frame_path)],
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
            incident_facets: dict[str, list[dict[str, Any]]] = {}
            if self.has_incident_v3():
                facet_sources = {
                    "crop_name": (
                        self.incident_weekly_state_path,
                        self.incident_weekly_state_columns,
                    ),
                    "stage_bucket": (
                        self.incident_stage_summary_path,
                        self.incident_stage_summary_columns,
                    ),
                    "incident_state": (
                        self.incident_weekly_state_path,
                        self.incident_weekly_state_columns,
                    ),
                }
                for column, (source_path, source_columns) in facet_sources.items():
                    if column not in source_columns or "incident_id" not in source_columns:
                        continue
                    if (
                        column == "stage_bucket"
                        and "affected_crop_instance_count" not in source_columns
                    ):
                        continue
                    affected_stage_clause = (
                        "AND COALESCE(TRY_CAST(s.affected_crop_instance_count AS BIGINT), 0) > 0"
                        if column == "stage_bucket"
                        else ""
                    )
                    rows = con.execute(
                        f"""
                        SELECT CAST(s.{column} AS VARCHAR) AS {column},
                               COUNT(DISTINCT s.incident_id) AS count
                        FROM read_parquet(?) AS s
                        WHERE s.{column} IS NOT NULL
                          AND TRIM(CAST(s.{column} AS VARCHAR)) <> ''
                          {affected_stage_clause}
                        GROUP BY s.{column}
                        ORDER BY count DESC, {column}
                        """,
                        [str(source_path)],
                    ).fetchdf()
                    incident_facets[column] = _records(rows)
        motif_records = _records(motifs)
        LOGGER.info("motifs_loaded query=%s limit=%s motifs=%s", q, limit, len(motif_records))
        return {
            "motifs": motif_records,
            "exact_stories": motif_records,
            "facets": {
                "current_risk_band": _records(risks),
                "hazard_signature": _records(hazards),
                "response_signature": _records(responses),
                "motif_family": _records(motif_families),
                **incident_facets,
            },
            "taxonomy": self.motif_taxonomy(),
        }

    def activity(self, filters: dict[str, str] | None) -> dict[str, Any]:
        self.require_ready()
        if self.has_incident_v3():
            return self._incident_activity(filters)
        filters = self._effective_frame_filters(filters)
        motif_family_sql = self._motif_family_sql("f")
        current_rank_sql = (
            "COALESCE(f.current_risk_rank, f.max_risk_rank)"
            if "current_risk_rank" in self.frame_columns
            else "f.max_risk_rank"
        )
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
                    MAX({current_rank_sql}) AS max_risk_rank
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

    def _incident_activity(
        self, filters: dict[str, str] | None
    ) -> dict[str, Any]:
        """Aggregate V3 activity from authoritative incident/week artifacts."""
        self._require_incident_columns(
            "footprints",
            self.incident_footprint_columns,
            {
                "timeline_bucket", "incident_id", "crop_name",
                "hazard_family", "incident_state", "monitored_count",
                "affected_count", "severe_count",
            },
        )
        self._require_incident_columns(
            "weekly state",
            self.incident_weekly_state_columns,
            {
                "timeline_bucket", "incident_id", "crop_name",
                "hazard_family", "incident_state", "monitored_count",
                "affected_count", "severe_count",
            },
        )
        clean = _clean_filters(filters)
        incident_id = clean.pop("incident_id", None)
        story_cluster_id = clean.pop("story_cluster_id", None)
        if (
            incident_id is not None
            and story_cluster_id is not None
            and incident_id != story_cluster_id
        ):
            raise RequestValidationError(
                "incident_id and story_cluster_id must match when both are provided"
            )
        if incident_id is not None:
            clean["story_cluster_id"] = incident_id
        elif story_cluster_id is not None:
            clean["story_cluster_id"] = story_cluster_id

        frame_filter_keys = {
            "max_risk_band", "current_risk_band", "response_signature"
        }
        frame_filters = {
            key: value for key, value in clean.items() if key in frame_filter_keys
        }
        if (
            "current_risk_band" in frame_filters
            and "current_risk_band" not in self.frame_columns
        ):
            frame_filters["max_risk_band"] = frame_filters.pop(
                "current_risk_band"
            )
        incident_filters = {
            key: value for key, value in clean.items() if key not in frame_filter_keys
        }
        frame_clause, frame_params = _filter_sql(frame_filters, "fr")

        incident_clauses: list[str] = []
        incident_params: list[Any] = []
        incident_columns = {
            "story_cluster_id": "incident_id",
            "crop_name": "crop_name",
            "incident_state": "incident_state",
            "hazard_signature": "hazard_family",
            "motif_family": "hazard_family",
        }
        stage_bucket = incident_filters.pop("stage_bucket", None)
        for key in sorted(incident_filters):
            column = incident_columns.get(key)
            if column is None:
                continue
            incident_clauses.append(
                f"AND CAST(i.{column} AS VARCHAR) = ?"
            )
            incident_params.append(incident_filters[key])
        if stage_bucket is not None:
            self._require_incident_columns(
                "stage summary",
                self.incident_stage_summary_columns,
                {
                    "incident_id", "timeline_bucket", "stage_bucket",
                    "affected_crop_instance_count",
                },
            )
            incident_clauses.append(
                """
                AND EXISTS (
                    SELECT 1
                    FROM read_parquet(?) AS stage
                    WHERE CAST(stage.timeline_bucket AS DATE)
                            = CAST(i.timeline_bucket AS DATE)
                      AND CAST(stage.incident_id AS VARCHAR)
                            = CAST(i.incident_id AS VARCHAR)
                      AND CAST(stage.stage_bucket AS VARCHAR) = ?
                      AND COALESCE(
                            TRY_CAST(stage.affected_crop_instance_count AS BIGINT), 0
                          ) > 0
                )
                """
            )
            incident_params.extend(
                [str(self.incident_stage_summary_path), stage_bucket]
            )
        if frame_filters:
            incident_clauses.append(
                """
                AND EXISTS (
                    SELECT 1 FROM filtered_frames AS matched_frame
                    WHERE CAST(matched_frame.timeline_bucket AS DATE)
                            = CAST(i.timeline_bucket AS DATE)
                      AND CAST(matched_frame.story_cluster_id AS VARCHAR)
                            = CAST(i.incident_id AS VARCHAR)
                )
                """
            )
        incident_clause = "\n".join(incident_clauses)
        started = time.perf_counter()
        incident_weeks_sql = """
            incident_weeks AS (
                SELECT
                    CAST(timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(incident_id AS VARCHAR) AS incident_id,
                    CAST(crop_name AS VARCHAR) AS crop_name,
                    CAST(hazard_family AS VARCHAR) AS hazard_family,
                    CAST(incident_state AS VARCHAR) AS incident_state,
                    COALESCE(TRY_CAST(monitored_count AS BIGINT), 0) AS monitored_count,
                    COALESCE(TRY_CAST(affected_count AS BIGINT), 0) AS affected_count,
                    COALESCE(TRY_CAST(severe_count AS BIGINT), 0) AS severe_count
                FROM read_parquet(?)
                UNION ALL
                SELECT
                    CAST(f.timeline_bucket AS DATE) AS timeline_bucket,
                    CAST(f.incident_id AS VARCHAR) AS incident_id,
                    CAST(f.crop_name AS VARCHAR) AS crop_name,
                    CAST(f.hazard_family AS VARCHAR) AS hazard_family,
                    CAST(f.incident_state AS VARCHAR) AS incident_state,
                    COALESCE(TRY_CAST(f.monitored_count AS BIGINT), 0)
                        AS monitored_count,
                    COALESCE(TRY_CAST(f.affected_count AS BIGINT), 0)
                        AS affected_count,
                    COALESCE(TRY_CAST(f.severe_count AS BIGINT), 0) AS severe_count
                FROM read_parquet(?) AS f
                WHERE NOT EXISTS (
                    SELECT 1 FROM read_parquet(?) AS state
                    WHERE CAST(state.timeline_bucket AS DATE)
                            = CAST(f.timeline_bucket AS DATE)
                      AND CAST(state.incident_id AS VARCHAR)
                            = CAST(f.incident_id AS VARCHAR)
                )
            )
        """
        incident_paths = [
            str(self.incident_weekly_state_path),
            str(self.incident_footprints_path),
            str(self.incident_weekly_state_path),
        ]
        with duckdb.connect(":memory:") as con:
            if frame_filters:
                current_rank_sql = (
                    "COALESCE(fr.current_risk_rank, fr.max_risk_rank)"
                    if "current_risk_rank" in self.frame_columns
                    else "fr.max_risk_rank"
                )
                rows = con.execute(
                    f"""
                    WITH filtered_frames AS (
                        SELECT fr.* FROM read_parquet(?) AS fr
                        WHERE 1 = 1 {frame_clause}
                    ), {incident_weeks_sql}, matching_incidents AS (
                        SELECT DISTINCT i.* FROM incident_weeks AS i
                        WHERE 1 = 1 {incident_clause}
                    ), incident_stats AS (
                        SELECT timeline_bucket,
                            COUNT(DISTINCT incident_id) AS story_cluster_count,
                            COUNT(DISTINCT hazard_family) AS motif_family_count
                        FROM matching_incidents GROUP BY timeline_bucket
                    ), frame_stats AS (
                        SELECT m.timeline_bucket,
                            COUNT(DISTINCT fr.field_id) AS field_count,
                            SUM(fr.reportable_day_count) AS reportable_day_count,
                            SUM(fr.event_count) AS event_count,
                            MAX({current_rank_sql}) AS max_risk_rank
                        FROM matching_incidents AS m
                        LEFT JOIN filtered_frames AS fr
                          ON CAST(fr.timeline_bucket AS DATE) = m.timeline_bucket
                         AND CAST(fr.story_cluster_id AS VARCHAR) = m.incident_id
                        GROUP BY m.timeline_bucket
                    )
                    SELECT CAST(i.timeline_bucket AS VARCHAR) AS timeline_bucket,
                        COALESCE(f.field_count, 0) AS field_count,
                        i.story_cluster_count,
                        i.story_cluster_count AS incident_count,
                        i.story_cluster_count AS activity_count,
                        CAST('incident_stories' AS VARCHAR) AS activity_unit,
                        i.motif_family_count,
                        COALESCE(f.reportable_day_count, 0) AS reportable_day_count,
                        COALESCE(f.event_count, 0) AS event_count,
                        COALESCE(f.max_risk_rank, 0) AS max_risk_rank
                    FROM incident_stats AS i
                    LEFT JOIN frame_stats AS f USING (timeline_bucket)
                    ORDER BY i.timeline_bucket
                    """,
                    [
                        str(self.frame_path), *frame_params, *incident_paths,
                        *incident_params,
                    ],
                ).fetchdf()
            else:
                rows = con.execute(
                    f"""
                    WITH {incident_weeks_sql}, matching_incidents AS (
                        SELECT DISTINCT i.* FROM incident_weeks AS i
                        WHERE 1 = 1 {incident_clause}
                    )
                    SELECT CAST(timeline_bucket AS VARCHAR) AS timeline_bucket,
                        SUM(monitored_count)::BIGINT AS field_count,
                        COUNT(DISTINCT incident_id) AS story_cluster_count,
                        COUNT(DISTINCT incident_id) AS incident_count,
                        COUNT(DISTINCT incident_id) AS activity_count,
                        CAST('incident_stories' AS VARCHAR) AS activity_unit,
                        COUNT(DISTINCT hazard_family) AS motif_family_count,
                        CAST(0 AS BIGINT) AS reportable_day_count,
                        SUM(affected_count)::BIGINT AS event_count,
                        MAX(CASE
                            WHEN severe_count > 0 OR UPPER(incident_state) = 'SEVERE'
                                THEN 4
                            WHEN UPPER(incident_state) IN ('ACTIVE', 'CONFIRMED') THEN 3
                            WHEN UPPER(incident_state) IN
                                ('CANDIDATE', 'WATCH', 'PRESSURE_QUIET') THEN 2
                            ELSE 1 END) AS max_risk_rank
                    FROM matching_incidents
                    GROUP BY timeline_bucket
                    ORDER BY timeline_bucket
                    """,
                    [*incident_paths, *incident_params],
                ).fetchdf()
        buckets = _records(rows)
        public_filters = dict(clean)
        LOGGER.info(
            "activity_loaded source=incident_artifacts filters=%s buckets=%s "
            "elapsed_ms=%.1f",
            _json_for_log(public_filters),
            len(buckets),
            (time.perf_counter() - started) * 1000,
        )
        return {
            "filters": public_filters,
            "source": "incident_artifacts",
            "activity_unit": "incident_stories",
            "uses_frame_fields": bool(frame_filters),
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

    def evolution(self, filters: dict[str, str] | None) -> dict[str, Any]:
        """Summarize retrospective activity-center change without implying movement."""
        self.require_ready()
        filters = self._effective_frame_filters(filters)
        if not filters:
            raise RequestValidationError("at least one story or evidence filter is required")
        if not self._has_optimized_geometry():
            raise RequestValidationError("evolution requires optimized field_geometry.parquet")

        motif_family_sql = self._motif_family_sql("f")
        filter_clause, filter_params = _filter_sql(
            filters,
            "f",
            motif_family_sql=motif_family_sql,
        )
        open_state_clause = (
            "AND UPPER(CAST(f.event_state AS VARCHAR)) IN "
            "('WATCH','ACTIVE','SEVERE','QUIET_PENDING','RECOVERING','DATA_GAP')"
            if "event_state" in self.frame_columns
            else ""
        )
        started = time.perf_counter()
        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                f"""
                WITH all_buckets AS (
                    SELECT
                        timeline_bucket,
                        ROW_NUMBER() OVER (ORDER BY timeline_bucket) - 1 AS bucket_index
                    FROM (
                        SELECT DISTINCT timeline_bucket
                        FROM read_parquet(?)
                    )
                ),
                selected AS (
                    SELECT DISTINCT
                        f.timeline_bucket,
                        f.field_id,
                        RADIANS(g.centroid_lat) AS latitude_radians,
                        RADIANS(g.centroid_lon) AS longitude_radians
                    FROM read_parquet(?) AS f
                    JOIN read_parquet(?) AS g USING (field_id)
                    WHERE g.centroid_lon IS NOT NULL
                      AND g.centroid_lat IS NOT NULL
                      {filter_clause}
                      {open_state_clause}
                ),
                vectors AS (
                    SELECT
                        timeline_bucket,
                        COUNT(*) AS field_count,
                        AVG(COS(latitude_radians) * COS(longitude_radians)) AS mean_x,
                        AVG(COS(latitude_radians) * SIN(longitude_radians)) AS mean_y,
                        AVG(SIN(latitude_radians)) AS mean_z
                    FROM selected
                    GROUP BY timeline_bucket
                ),
                centers AS (
                    SELECT
                        timeline_bucket,
                        field_count,
                        ATAN2(mean_y, mean_x) AS center_longitude_radians,
                        ATAN2(mean_z, SQRT(mean_x * mean_x + mean_y * mean_y)) AS center_latitude_radians
                    FROM vectors
                ),
                distances AS (
                    SELECT
                        s.timeline_bucket,
                        s.field_id,
                        c.field_count,
                        c.center_longitude_radians,
                        c.center_latitude_radians,
                        2.0 * 6371.0088 * ASIN(
                            SQRT(
                                LEAST(
                                    1.0,
                                    GREATEST(
                                        0.0,
                                        POWER(SIN((s.latitude_radians - c.center_latitude_radians) / 2.0), 2)
                                        + COS(c.center_latitude_radians) * COS(s.latitude_radians)
                                        * POWER(SIN((s.longitude_radians - c.center_longitude_radians) / 2.0), 2)
                                    )
                                )
                            )
                        ) AS distance_km
                    FROM selected AS s
                    JOIN centers AS c USING (timeline_bucket)
                ),
                bucket_stats AS (
                    SELECT
                        timeline_bucket,
                        ANY_VALUE(field_count) AS field_count,
                        DEGREES(ANY_VALUE(center_longitude_radians)) AS center_lon,
                        DEGREES(ANY_VALUE(center_latitude_radians)) AS center_lat,
                        QUANTILE_CONT(distance_km, 0.5) AS p50_dispersion_km,
                        QUANTILE_CONT(distance_km, 0.9) AS p90_dispersion_km
                    FROM distances
                    GROUP BY timeline_bucket
                ),
                indexed AS (
                    SELECT stats.*, buckets.bucket_index
                    FROM bucket_stats AS stats
                    JOIN all_buckets AS buckets USING (timeline_bucket)
                ),
                pairs AS (
                    SELECT
                        *,
                        LAG(timeline_bucket) OVER (ORDER BY timeline_bucket) AS previous_timeline_bucket,
                        LAG(bucket_index) OVER (ORDER BY timeline_bucket) AS previous_bucket_index,
                        LAG(field_count) OVER (ORDER BY timeline_bucket) AS previous_field_count
                    FROM indexed
                ),
                persistence AS (
                    SELECT
                        p.timeline_bucket,
                        COUNT(prior.field_id) AS persisting_field_count
                    FROM pairs AS p
                    JOIN selected AS current_state
                      ON current_state.timeline_bucket = p.timeline_bucket
                    LEFT JOIN selected AS prior
                      ON prior.timeline_bucket = p.previous_timeline_bucket
                     AND prior.field_id = current_state.field_id
                    GROUP BY p.timeline_bucket
                )
                SELECT
                    p.timeline_bucket,
                    p.previous_timeline_bucket,
                    p.bucket_index,
                    p.previous_bucket_index,
                    p.field_count,
                    p.previous_field_count,
                    p.center_lon,
                    p.center_lat,
                    p.p50_dispersion_km,
                    p.p90_dispersion_km,
                    persistence.persisting_field_count
                FROM pairs AS p
                JOIN persistence USING (timeline_bucket)
                ORDER BY p.timeline_bucket
                """,
                [str(self.frame_path), str(self.frame_path), str(self.geometry_path), *filter_params],
            ).fetchdf()

        points: list[dict[str, Any]] = []
        for row in _records(rows):
            field_count = int(row.get("field_count") or 0)
            timeline_bucket = _clean(row.get("timeline_bucket"))
            previous_bucket = _clean(row.get("previous_timeline_bucket"))
            has_previous = previous_bucket is not None
            previous_field_count_value = _clean(row.get("previous_field_count"))
            previous_field_count = int(previous_field_count_value or 0)
            persisting = int(row.get("persisting_field_count") or 0) if has_previous else 0
            entering = max(0, field_count - persisting)
            exiting = max(0, previous_field_count - persisting) if has_previous else 0
            denominator = field_count + previous_field_count - persisting
            jaccard = (persisting / denominator) if has_previous and denominator else None
            adjacent_index = (
                int(row.get("bucket_index") or 0)
                - int(_clean(row.get("previous_bucket_index")) or 0)
                == 1
            )
            consecutive = bool(
                has_previous
                and _weekly_buckets_are_consecutive(
                    previous_bucket,
                    timeline_bucket,
                    fallback=adjacent_index,
                )
            )
            if not has_previous:
                break_reason = "start"
            elif not consecutive:
                break_reason = "timeline_gap"
            elif persisting == 0:
                break_reason = "zero_field_overlap"
            else:
                break_reason = None
            points.append(
                {
                    "timeline_bucket": timeline_bucket,
                    "previous_timeline_bucket": previous_bucket,
                    "field_count": field_count,
                    "center_lon": _rounded_float(row.get("center_lon"), 7),
                    "center_lat": _rounded_float(row.get("center_lat"), 7),
                    "p50_dispersion_km": _rounded_float(row.get("p50_dispersion_km"), 3),
                    "p90_dispersion_km": _rounded_float(row.get("p90_dispersion_km"), 3),
                    "entering_field_count": entering,
                    "persisting_field_count": persisting,
                    "exiting_field_count": exiting,
                    "jaccard_overlap": _rounded_float(jaccard, 6),
                    "consecutive": consecutive,
                    "trail_segment_allowed": bool(consecutive and persisting > 0),
                    "break_reason": break_reason,
                }
            )
        LOGGER.info(
            "evolution_loaded filters=%s buckets=%s elapsed_ms=%.1f",
            _json_for_log(filters),
            len(points),
            (time.perf_counter() - started) * 1000,
        )
        return {
            "kind": "aggregate_activity_center",
            "is_physical_movement": False,
            "center_method": "unweighted_spherical_mean_of_field_centroids",
            "dispersion_method": "haversine_distance_to_activity_center",
            "filters": filters,
            "bucket_count": len(points),
            "points": points,
        }

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
        filters = self._effective_frame_filters(filters)
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
        optional_trail_select = ",\n                    ".join(
            self._optional_frame_sql("f", name, fallback)
            for name, fallback in (
                ("event_id", "NULL"),
                ("event_state", "NULL"),
                ("archetype_display_state", "NULL"),
                ("anchor_status", "NULL"),
                ("accepted", "NULL"),
                ("current_risk_band", "f.max_risk_band"),
                ("current_risk_rank", "f.max_risk_rank"),
            )
        )
        current_rank_sql = (
            "COALESCE(TRY_CAST(f.current_risk_rank AS INTEGER), "
            "TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
            if "current_risk_rank" in self.frame_columns
            else "COALESCE(TRY_CAST(f.max_risk_rank AS INTEGER), 0)"
        )
        state_sql = (
            "UPPER(COALESCE(CAST(f.event_state AS VARCHAR), ''))"
            if "event_state" in self.frame_columns
            else "''"
        )
        state_priority_sql = f"""CASE {state_sql}
            WHEN 'SEVERE' THEN 6 WHEN 'ACTIVE' THEN 5 WHEN 'WATCH' THEN 4
            WHEN 'RECOVERING' THEN 3 WHEN 'QUIET_PENDING' THEN 2
            WHEN 'DATA_GAP' THEN 1 ELSE 0 END"""
        open_state_clause = (
            "AND UPPER(CAST(f.event_state AS VARCHAR)) IN "
            "('WATCH','ACTIVE','SEVERE','QUIET_PENDING','RECOVERING','DATA_GAP')"
            if "event_state" in self.frame_columns
            else ""
        )

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
                {open_state_clause}
            ),
            previous_open_fields AS (
                SELECT DISTINCT f.field_id
                FROM all_buckets AS a
                CROSS JOIN target_bucket AS t
                JOIN read_parquet(?) AS f ON f.timeline_bucket = a.timeline_bucket
                JOIN read_parquet(?) AS g USING (field_id)
                WHERE a.bucket_index = t.bucket_index - 1
                {filter_clause}
                {bbox_clause}
                {open_state_clause}
            ),
            prior_ranked AS (
                SELECT
                    s.bucket_index,
                    s.age_index,
                    f.timeline_bucket,
                    f.field_id,
                    f.story_cluster_id,
                    {optional_trail_select},
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
                        ORDER BY s.bucket_index DESC, {state_priority_sql} DESC,
                            {current_rank_sql} DESC, f.max_risk_rank DESC,
                            f.reportable_day_count DESC, CAST(f.story_cluster_id AS VARCHAR)
                    ) AS prior_rank
                FROM selected_prior_buckets AS s
                JOIN read_parquet(?) AS f USING (timeline_bucket)
                JOIN read_parquet(?) AS g USING (field_id)
                {label_join}
                LEFT JOIN current_fields AS c USING (field_id)
                WHERE 1 = 1
                {filter_clause}
                {bbox_clause}
                {open_state_clause}
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
                    (SELECT COUNT(*) FROM previous_open_fields) AS previous_field_count,
                    (SELECT COUNT(*) FROM prior_features) AS prior_field_count,
                    (
                        SELECT COUNT(*)
                        FROM current_fields AS current_state
                        JOIN previous_open_fields AS previous_state USING (field_id)
                    ) AS persisting_field_count,
                    EXISTS (
                        SELECT 1 FROM all_buckets AS a CROSS JOIN target_bucket AS t
                        WHERE a.bucket_index = t.bucket_index - 1
                    ) AS previous_bucket_available
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
                    "event_id",
                    "event_state",
                    "archetype_display_state",
                    "anchor_status",
                    "accepted",
                    "max_risk_band",
                    "current_risk_band",
                    "hazard_signature",
                    "response_signature",
                    "reportable_day_count",
                    "event_count",
                    "max_risk_rank",
                    "current_risk_rank",
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
                "previous_field_count": int(stats_row.get("previous_field_count") or 0),
                "persisting_field_count": int(stats_row.get("persisting_field_count") or 0),
                "departed_field_count": max(
                    0,
                    int(stats_row.get("previous_field_count") or 0)
                    - int(stats_row.get("persisting_field_count") or 0),
                ),
                "new_current_field_count": max(
                    0,
                    int(stats_row.get("current_field_count") or 0)
                    - int(stats_row.get("persisting_field_count") or 0),
                ),
                "transition_counts_available": bool(stats_row.get("previous_bucket_available")),
                "transition_scope": "open_previous_bucket",
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

    def field_trajectory(self, field_id: str, limit: int) -> dict[str, Any]:
        """Return causal weekly prefix states when the monitoring artifact exists."""
        self.require_ready()
        limit = max(1, min(limit, 1000))
        if not self.state_snapshots_path.is_file():
            return {
                "field_id": field_id,
                "available": False,
                "mode": "retrospective_event_windows_only",
                "states": [],
            }
        columns = self._parquet_columns(self.state_snapshots_path)

        def optional(name: str, fallback: str = "NULL") -> str:
            return f"{name}" if name in columns else f"{fallback} AS {name}"

        with duckdb.connect(":memory:") as con:
            rows = con.execute(
                f"""
                WITH latest_buckets AS (
                    SELECT DISTINCT timeline_bucket
                    FROM read_parquet(?)
                    WHERE field_id = ?
                    ORDER BY timeline_bucket DESC
                    LIMIT ?
                )
                SELECT
                    states.timeline_bucket,
                    {optional('snapshot_as_of_date')},
                    states.field_id,
                    {optional('crop_name')},
                    {optional('crop_season')},
                    {optional('event_id')},
                    {optional('event_state')},
                    {optional('story_cluster_id')},
                    {optional('motif_id')},
                    {optional('archetype_display_state')},
                    {optional('anchor_date')},
                    {optional('anchor_status')},
                    {optional('accepted')},
                    {optional('assignment_reason')},
                    {optional('hazard_signature')},
                    {optional('max_risk_rank', '0')},
                    {optional('max_risk_band')},
                    {optional('current_risk_rank', 'daily_pressure_rank')},
                    {optional('current_risk_band', 'max_risk_band')},
                    {optional('daily_pressure_rank', '0')},
                    {optional('daily_response_class')},
                    {optional('right_censored', 'FALSE')},
                    {optional('requires_review', 'FALSE')},
                    {optional('revision', '1')}
                FROM read_parquet(?) AS states
                JOIN latest_buckets USING (timeline_bucket)
                WHERE states.field_id = ?
                ORDER BY states.timeline_bucket, event_id
                """,
                [
                    str(self.state_snapshots_path), field_id, limit,
                    str(self.state_snapshots_path), field_id,
                ],
            ).fetchdf()
        states = _records(rows)
        return {
            "field_id": field_id,
            "available": True,
            "mode": "causal_weekly_event_prefix",
            "state_count": len(states),
            "states": states,
        }

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
        filters = self._effective_frame_filters(filters)
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
        except Exception:
            LOGGER.exception(
                "empty_frame_diagnostic_failed bucket=%s filters=%s bbox=%s "
                "optimized_geometry=%s",
                timeline_bucket,
                _json_for_log(filters),
                _json_for_log(bbox),
                optimized,
            )
            diagnostics["diagnostic_error"] = "diagnostic_query_failed"
        return diagnostics


def _v4_grid_feature(row: dict[str, Any], layer_role: str) -> dict[str, Any]:
    min_lon = float(row["min_lon"])
    min_lat = float(row["min_lat"])
    max_lon = float(row["max_lon"])
    max_lat = float(row["max_lat"])
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat],
        ]],
    }
    properties = dict(row)
    day = str(properties.get("calendar_date") or "")[:10]
    properties.update(
        {
            "calendar_date": day,
            "timeline_bucket": day,
            "layer_role": layer_role,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "hazard_signature": properties.get("hazard_family")
                or properties.get("dominant_hazard_family"),
            "stage_bucket": properties.get("dominant_stage_bucket"),
            "crop_name": properties.get("dominant_crop_name"),
            "is_physical_movement": False,
        }
    )
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _v4_story_feature(row: dict[str, Any]) -> dict[str, Any]:
    geometry, geom_bbox = _precomputed_geojson_and_bbox(
        row.get("geometry_geojson"),
        row.get("geometry_type"),
        (
            row.get("min_lon"), row.get("min_lat"),
            row.get("max_lon"), row.get("max_lat"),
        ),
    )
    properties = {
        key: value for key, value in row.items()
        if key != "geometry_geojson"
    }
    for role in ("pressure", "impact", "watch"):
        raw = properties.pop(f"{role}_geometry_geojson", None)
        properties[f"{role}_geometry"] = raw
    properties["timeline_bucket"] = str(
        properties.get("story_week") or properties.get("timeline_bucket") or ""
    )[:10]
    properties["story_known_date"] = str(
        properties.get("story_known_date") or properties.get("knowledge_time") or ""
    )[:10]
    properties["bbox"] = geom_bbox
    properties["is_physical_movement"] = False
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def make_handler(store: StoryMapStore, settings: Settings) -> type[BaseHTTPRequestHandler]:
    api_cache_bytes, static_cache_bytes = _cache_byte_budgets(
        settings.cache_max_bytes
    )
    api_cache = ResponseCache(
        ttl_seconds=settings.cache_seconds,
        capacity=settings.cache_entries,
        gzip_min_bytes=settings.gzip_min_bytes,
        max_bytes=api_cache_bytes,
    )
    static_cache = ResponseCache(
        ttl_seconds=settings.cache_seconds,
        capacity=settings.cache_entries,
        gzip_min_bytes=settings.gzip_min_bytes,
        max_bytes=static_cache_bytes,
    )
    query_slots = BoundedSemaphore(max(1, int(settings.query_concurrency)))
    # DuckDB/GeoJSON work and JSON cleaning/encoding/gzip are separate CPU and
    # memory-heavy stages. Bound both stages independently so a burst cannot
    # move an unbounded number of completed queries into response construction.
    response_slots = BoundedSemaphore(max(1, int(settings.query_concurrency)))
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
                    self._api_cache_key = _canonical_api_cache_key(path, query)
                    cached = api_cache.get(self._api_cache_key)
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
                if path == "/api/v4/timeline":
                    self._json(self._query(store.v4_timeline))
                    return
                if path.startswith("/api/v4/frame-state/"):
                    day = _parse_iso_date_segment(
                        unquote(path.removeprefix("/api/v4/frame-state/"))
                    )
                    self._json(
                        self._query(
                            lambda: store.v4_frame_state(
                                calendar_date=day,
                                bbox=_parse_bbox(_first(query, "bbox")),
                                filters=_incident_filters_from_query(query),
                                limit=_feature_limit_query(
                                    query,
                                    "limit",
                                    settings.default_feature_limit,
                                    settings.max_feature_limit,
                                ),
                            )
                        )
                    )
                    return
                if path == "/api/v4/frame-state":
                    raise RequestValidationError(
                        "date path parameter is required in YYYY-MM-DD format"
                    )
                if path.startswith("/api/v4/frame/"):
                    day = _parse_iso_date_segment(
                        unquote(path.removeprefix("/api/v4/frame/"))
                    )
                    self._json(
                        self._query(
                            lambda: store.v4_frame(
                                calendar_date=day,
                                bbox=_parse_bbox(_first(query, "bbox")),
                                filters=_incident_filters_from_query(query),
                                limit=_feature_limit_query(
                                    query,
                                    "limit",
                                    settings.default_feature_limit,
                                    settings.max_feature_limit,
                                ),
                            )
                        )
                    )
                    return
                if path == "/api/v4/frame":
                    raise RequestValidationError(
                        "date path parameter is required in YYYY-MM-DD format"
                    )
                if path.startswith("/api/v4/field/"):
                    field_id = _validate_field_id(
                        unquote(path.removeprefix("/api/v4/field/"))
                    )
                    raw_as_of = _first(query, "as_of")
                    raw_crop_instance = _first(query, "crop_instance_id")
                    self._json(
                        self._query(
                            lambda: store.v4_field_detail(
                                field_id,
                                as_of_date=(
                                    _parse_iso_date_segment(raw_as_of)
                                    if raw_as_of else None
                                ),
                                crop_instance_id=(
                                    _validate_crop_instance_id(raw_crop_instance)
                                    if raw_crop_instance else None
                                ),
                                lookback_days=_int_query(
                                    query,
                                    "lookback_days",
                                    V4_INCIDENT_DEFAULT_LOOKBACK_DAYS,
                                    V4_INCIDENT_MAX_LOOKBACK_DAYS,
                                ),
                                history_limit=_int_query(
                                    query,
                                    "history_limit",
                                    V4_INCIDENT_DEFAULT_HISTORY_LIMIT,
                                    V4_INCIDENT_MAX_HISTORY_LIMIT,
                                ),
                            )
                        )
                    )
                    return
                if path == "/api/v4/field":
                    raise RequestValidationError("field_id path parameter is required")
                if path.startswith("/api/v4/incident/"):
                    incident_id = _validate_incident_id(
                        unquote(path.removeprefix("/api/v4/incident/"))
                    )
                    raw_as_of = _first(query, "as_of")
                    as_of = (
                        _parse_iso_date_segment(raw_as_of) if raw_as_of else None
                    )
                    self._json(
                        self._query(
                            lambda: store.v4_incident_detail(
                                incident_id,
                                as_of_date=as_of,
                                lookback_days=_int_query(
                                    query,
                                    "lookback_days",
                                    V4_INCIDENT_DEFAULT_LOOKBACK_DAYS,
                                    V4_INCIDENT_MAX_LOOKBACK_DAYS,
                                ),
                                history_limit=_int_query(
                                    query,
                                    "history_limit",
                                    V4_INCIDENT_DEFAULT_HISTORY_LIMIT,
                                    V4_INCIDENT_MAX_HISTORY_LIMIT,
                                ),
                            )
                        )
                    )
                    return
                if path == "/api/v4/incident":
                    raise RequestValidationError("incident_id path parameter is required")
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
                if path == "/api/evolution":
                    self._json(self._query(lambda: store.evolution(_filters_from_query(query))))
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
                if path.startswith("/api/incident-footprints/"):
                    bucket = _parse_iso_date_segment(
                        unquote(path.removeprefix("/api/incident-footprints/"))
                    )
                    self._json(
                        self._query(
                            lambda: store.incident_footprints(
                                timeline_bucket=bucket,
                                filters=_incident_filters_from_query(query),
                            )
                        )
                    )
                    return
                if path == "/api/incident-footprints":
                    raise RequestValidationError(
                        "date path parameter is required in YYYY-MM-DD format"
                    )
                if path.startswith("/api/incident/"):
                    incident_id = _validate_incident_id(
                        unquote(path.removeprefix("/api/incident/"))
                    )
                    self._json(
                        self._query(lambda: store.incident_detail(incident_id))
                    )
                    return
                if path == "/api/incident":
                    raise RequestValidationError("incident_id path parameter is required")
                if path.startswith("/api/frame-state/"):
                    bucket = unquote(path.removeprefix("/api/frame-state/"))
                    self._json(
                        self._query(
                            lambda: store.frame_state(
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
                if path.startswith("/api/field/") and path.endswith("/trajectory"):
                    field_id = unquote(
                        path.removeprefix("/api/field/").removesuffix("/trajectory")
                    )
                    self._json(
                        self._query(
                            lambda: store.field_trajectory(
                                field_id, _int_query(query, "limit", 250, 1000)
                            )
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
            except ResourceNotFoundError as exc:
                LOGGER.info(
                    "http_not_found method=GET path=%s remote=%s error=%s elapsed_ms=%.1f",
                    path,
                    self.client_address[0] if self.client_address else None,
                    str(exc),
                    (time.perf_counter() - self._request_started) * 1000,
                )
                self._json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            except ServerBusyError as exc:
                LOGGER.warning(
                    "http_server_busy method=GET path=%s query=%s remote=%s "
                    "reason=%s elapsed_ms=%.1f",
                    path,
                    parsed.query,
                    self.client_address[0] if self.client_address else None,
                    str(exc),
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

        def do_POST(self) -> None:  # noqa: N802
            self._request_started = time.perf_counter()
            self._api_cache_key = None
            parsed = urlparse(self.path)
            path = parsed.path
            LOGGER.info(
                "http_request method=POST path=%s remote=%s",
                path,
                self.client_address[0] if self.client_address else None,
            )
            try:
                if path != "/api/geometry":
                    self._json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = self._read_json_body()
                geometry_version = payload.get("geometry_version")
                if not isinstance(geometry_version, str) or not geometry_version:
                    raise RequestValidationError("geometry_version must be a nonempty string")
                field_ids = payload.get("field_ids")
                if not isinstance(field_ids, list):
                    raise RequestValidationError("field_ids must be an array")
                if len(field_ids) > MAX_GEOMETRY_IDS:
                    raise RequestBodyTooLargeError(
                        f"field_ids may contain at most {MAX_GEOMETRY_IDS} items"
                    )
                validated_ids: list[str] = []
                for value in field_ids:
                    if not isinstance(value, str) or not value or value != value.strip():
                        raise RequestValidationError(
                            "field_ids must contain nonempty strings without surrounding whitespace"
                        )
                    if len(value) > 512:
                        raise RequestValidationError("field_ids values may not exceed 512 characters")
                    validated_ids.append(value)
                self._json(
                    self._query(
                        lambda: store.geometry_features(
                            geometry_version=geometry_version,
                            field_ids=validated_ids,
                        )
                    )
                )
            except RequestBodyTooLargeError as exc:
                LOGGER.warning("http_request_too_large method=POST path=%s error=%s", path, str(exc))
                self._json(
                    {"error": str(exc)},
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            except GeometryVersionMismatchError as exc:
                LOGGER.info("http_geometry_version_conflict path=%s error=%s", path, str(exc))
                self._json(
                    {
                        "error": str(exc),
                        "geometry_version": store.geometry_version(),
                    },
                    status=HTTPStatus.CONFLICT,
                )
            except RequestValidationError as exc:
                LOGGER.warning("http_bad_request method=POST path=%s error=%s", path, str(exc))
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except ServerBusyError as exc:
                LOGGER.warning(
                    "http_server_busy method=POST path=%s reason=%s",
                    path,
                    str(exc),
                )
                self._json(
                    {"error": "The server is busy. Retry this request shortly."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    extra_headers={"Retry-After": "1"},
                )
            except Exception:
                LOGGER.exception("http_error method=POST path=%s", path)
                self._json(
                    {"error": "The server could not complete this request."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise RequestValidationError("Content-Length is required")
            try:
                content_length = int(raw_length)
            except ValueError as exc:
                raise RequestValidationError("Content-Length must be an integer") from exc
            if content_length < 0:
                raise RequestValidationError("Content-Length must be nonnegative")
            if content_length > MAX_GEOMETRY_REQUEST_BYTES:
                # Drain modest over-limit bodies before replying so ordinary
                # clients do not race the 413 with an in-flight upload. Never
                # drain an arbitrarily large claimed body.
                if content_length <= MAX_GEOMETRY_REQUEST_BYTES * 2:
                    self.rfile.read(content_length)
                else:
                    # The unread bytes make HTTP/1.1 reuse unsafe.
                    self.close_connection = True
                raise RequestBodyTooLargeError(
                    f"request body may not exceed {MAX_GEOMETRY_REQUEST_BYTES} bytes"
                )
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestValidationError("request body must be valid UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise RequestValidationError("request body must be a JSON object")
            return payload

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
            path = urlparse(self.path).path
            gate_response_work = (
                status == HTTPStatus.OK
                and path.startswith("/api/")
                and path != "/api/health"
            )
            if gate_response_work and not response_slots.acquire(blocking=False):
                raise ServerBusyError("response capacity exhausted")
            try:
                body = _encode_json_body(payload)
                cache_key = getattr(self, "_api_cache_key", None)
                if status == HTTPStatus.OK and cache_key:
                    cached = api_cache.put(cache_key, body)
                    cache_status = (
                        "MISS"
                        if api_cache.capacity
                        and api_cache.ttl_seconds
                        and api_cache.max_bytes
                        and _cached_body_size(cached) <= api_cache.max_bytes
                        else "BYPASS"
                    )
                    cache_control = f"private, max-age={max(0, int(settings.cache_seconds))}"
                else:
                    cached = api_cache.put("", body)
                    cache_status = "BYPASS"
                    cache_control = "no-store"
            finally:
                if gate_response_work:
                    response_slots.release()
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
            if self.close_connection:
                self.send_header("Connection", "close")
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
                cache_status = (
                    "MISS"
                    if static_cache.capacity
                    and static_cache.ttl_seconds
                    and static_cache.max_bytes
                    and _cached_body_size(cached) <= static_cache.max_bytes
                    else "BYPASS"
                )
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
    "incident_id",
    "crop_name",
    "stage_bucket",
    "incident_state",
    "max_risk_band",
    "current_risk_band",
    "hazard_signature",
    "response_signature",
    "motif_family",
}


def _canonical_api_cache_key(
    path: str, query: dict[str, list[str]],
) -> str:
    """Ignore unknown cache-buster parameters and normalize query ordering."""
    items = [
        (key, value)
        for key in sorted(query)
        if key in API_CACHE_QUERY_KEYS
        for value in sorted(str(item) for item in query[key])
    ]
    suffix = urlencode(items, doseq=True)
    return path + (f"?{suffix}" if suffix else "")


def _incident_filters_from_query(query: dict[str, list[str]]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for key in INCIDENT_FOOTPRINT_FILTER_COLUMNS:
        value = _first(query, key)
        if value:
            filters[key] = value
    return filters


def _clean_incident_filters(filters: dict[str, str] | None) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in (filters or {}).items()
        if key in INCIDENT_FOOTPRINT_FILTER_COLUMNS
        and value is not None
        and str(value) != ""
    }


def _incident_filter_sql(
    filters: dict[str, str] | None,
    alias: str,
) -> tuple[str, list[Any]]:
    clean = _clean_incident_filters(filters)
    clauses: list[str] = []
    params: list[Any] = []
    for key in sorted(clean):
        clauses.append(f"AND {alias}.{key} = ?")
        params.append(clean[key])
    return ("\n".join(clauses), params)


def _parse_iso_date_segment(raw: str) -> str:
    if len(raw) != 10:
        raise RequestValidationError("date must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise RequestValidationError("date must use YYYY-MM-DD format") from exc
    if parsed.isoformat() != raw:
        raise RequestValidationError("date must use YYYY-MM-DD format")
    return raw


def _validate_incident_id(raw: str) -> str:
    if not raw or raw != raw.strip():
        raise RequestValidationError("incident_id must be a nonempty string")
    if len(raw) > 512:
        raise RequestValidationError("incident_id may not exceed 512 characters")
    return raw


def _validate_field_id(raw: str) -> str:
    if not raw or raw != raw.strip():
        raise RequestValidationError("field_id must be a nonempty string")
    if len(raw) > 512:
        raise RequestValidationError("field_id may not exceed 512 characters")
    return raw


def _validate_crop_instance_id(raw: str) -> str:
    if not raw or raw != raw.strip():
        raise RequestValidationError("crop_instance_id must be a nonempty string")
    if len(raw) > 512:
        raise RequestValidationError("crop_instance_id may not exceed 512 characters")
    return raw


def _optional_order_by(columns: frozenset[str], candidates: tuple[str, ...]) -> str:
    selected = [name for name in candidates if name in columns]
    return "ORDER BY " + ", ".join(selected) if selected else ""


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


@lru_cache(maxsize=8)
def _geometry_artifact_version(path: str, size: int, mtime_ns: int) -> str:
    """Hash immutable geometry bytes; stat values only invalidate the local cache."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"geom-sha256-{digest.hexdigest()}"


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


def _has_geometry_value(value: Any) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    if isinstance(value, Real):
        return False
    return True


def _precomputed_geojson_and_bbox(
    raw_geometry: Any,
    geometry_type: Any,
    raw_bbox: tuple[Any, Any, Any, Any],
) -> tuple[dict[str, Any], list[float]]:
    """Decode exporter-validated geometry without per-request Shapely work."""
    if not raw_geometry:
        raise ValueError("empty geometry")
    try:
        geometry = (
            json.loads(raw_geometry)
            if isinstance(raw_geometry, str)
            else dict(raw_geometry)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid precomputed GeoJSON geometry") from exc
    declared = str(geometry_type or geometry.get("type") or "")
    if declared not in {"Polygon", "MultiPolygon"}:
        raise ValueError("unsupported precomputed footprint geometry type")
    if str(geometry.get("type") or "") != declared:
        raise ValueError("precomputed footprint geometry type does not reconcile")
    try:
        bbox = [float(value) for value in raw_bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid precomputed footprint bounds") from exc
    if (
        not all(math.isfinite(value) for value in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
        or not isinstance(geometry.get("coordinates"), list)
    ):
        raise ValueError("invalid precomputed footprint geometry metadata")
    return geometry, bbox


def _intersects(a: list[float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _records(df: Any) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    # ``DataFrame.where(..., None)`` retains IEEE NaN values in float-backed
    # columns.  That leaks non-JSON values through the direct store API and is
    # especially visible for nullable identifiers produced by LEFT JOINs.
    # Object dtype is required before pandas can represent the missing value as
    # an actual Python ``None``.
    clean_df = df.astype(object).where(df.notnull(), None)
    return clean_df.to_dict(orient="records")


def _bounded_recent_records(df: Any, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """Trim a newest-first result and restore chronological presentation order."""
    records = _records(df)
    truncated = len(records) > limit
    selected = records[:limit]
    selected.reverse()
    return selected, truncated


def _causal_v4_incident_window(
    base_window: dict[str, Any],
    current_checkpoint: dict[str, Any],
    *,
    as_of_date: str,
    first_story_week: str,
    observed_week_count: int,
    data_gap_count: int,
    split_count: int,
    merge_count: int,
) -> dict[str, Any]:
    """Project a V3 lifetime window into facts knowable at one V4 cutoff."""

    def first(*values: Any) -> Any:
        return next((value for value in values if value not in (None, "")), None)

    def causal_date(*values: Any) -> str | None:
        for value in values:
            if value in (None, ""):
                continue
            normalized = str(value)[:10]
            try:
                if date.fromisoformat(normalized) <= date.fromisoformat(as_of_date):
                    return normalized
            except ValueError:
                continue
        return None

    state = first(
        current_checkpoint.get("incident_state"),
        current_checkpoint.get("current_state"),
        "UNKNOWN",
    )
    window: dict[str, Any] = {
        "incident_id": first(
            current_checkpoint.get("incident_id"), base_window.get("incident_id")
        ),
        "exposure_id": first(
            current_checkpoint.get("exposure_id"), base_window.get("exposure_id")
        ),
        "crop_name": first(
            current_checkpoint.get("crop_name"), base_window.get("crop_name")
        ),
        "hazard_family": first(
            current_checkpoint.get("hazard_family"), base_window.get("hazard_family")
        ),
        "first_evidence_week": first_story_week,
        "confirmed_week": causal_date(
            current_checkpoint.get("confirmed_week")
        ),
        "pressure_off_week": causal_date(
            current_checkpoint.get("pressure_off_week")
        ),
        "recovered_week": causal_date(
            current_checkpoint.get("recovered_week")
        ),
        "closed_week": causal_date(
            current_checkpoint.get("closed_week")
        ),
        "peak_week": causal_date(
            current_checkpoint.get("peak_week")
        ),
        "terminal_state": state,
        "incident_state": state,
        "right_censored": bool(current_checkpoint.get("right_censored")),
        "observed_week_count": observed_week_count,
        "relapse_count": int(current_checkpoint.get("relapse_count") or 0),
        "data_gap_count": data_gap_count,
        "split_count": split_count,
        "merge_count": merge_count,
        "snapshot_as_of_date": as_of_date,
        "story_known_date": str(
            current_checkpoint.get("story_known_date") or ""
        )[:10],
    }
    for key in (
        "component_id", "stage_bucket", "stage_distribution", "monitored_count",
        "evaluable_count", "affected_count", "active_count", "severe_count",
        "pressure_core_field_count", "impact_lag_field_count",
        "fresh_decline_field_count", "fresh_recovery_field_count",
        "coverage_adequate", "coverage_missing_cell_count",
        "global_crop_week_unmappable_instance_count", "footprint_carried_forward",
    ):
        if key in current_checkpoint:
            window[key] = current_checkpoint.get(key)
    return window


def _rounded_float(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        return None
    return round(normalized, digits)


def _weekly_buckets_are_consecutive(previous: Any, current: Any, *, fallback: bool) -> bool:
    try:
        previous_date = date.fromisoformat(str(previous)[:10])
        current_date = date.fromisoformat(str(current)[:10])
    except ValueError:
        return fallback
    return (current_date - previous_date).days == 7


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
