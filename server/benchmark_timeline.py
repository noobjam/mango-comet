#!/usr/bin/env python3
"""Compare legacy GeoJSON playback with geometry-once compact state playback."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--weeks", type=int, default=20)
    parser.add_argument("--bbox", default=None, help="minLon,minLat,maxLon,maxLat")
    parser.add_argument("--filter", action="append", default=[], help="key=value; repeatable")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, int]:
    encoded = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=encoded, headers=headers, method=method)
    started = time.perf_counter()
    with urlopen(request, timeout=120) as response:
        raw = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        decoded = gzip.decompress(raw) if response.headers.get("Content-Encoding") == "gzip" else raw
        return json.loads(decoded), elapsed_ms, len(raw)


def query_url(base_url: str, path: str, query: dict[str, Any]) -> str:
    values = {key: value for key, value in query.items() if value not in (None, "")}
    return f"{base_url.rstrip('/')}{path}?{urlencode(values)}"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[position]


def summarize(times: list[float], compressed_bytes: int) -> dict[str, Any]:
    return {
        "requests": len(times),
        "p50_ms": round(statistics.median(times), 1) if times else None,
        "p95_ms": round(percentile(times, 0.95) or 0, 1) if times else None,
        "max_ms": round(max(times), 1) if times else None,
        "compressed_bytes": compressed_bytes,
    }


def main() -> None:
    args = parse_args()
    filters: dict[str, str] = {}
    for item in args.filter:
        if "=" not in item:
            raise SystemExit(f"Invalid --filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        filters[key] = value
    timeline, _, _ = request_json("GET", f"{args.base_url.rstrip('/')}/api/timeline")
    buckets = [str(row["timeline_bucket"]) for row in timeline.get("buckets", [])][-args.weeks :]
    if not buckets:
        raise SystemExit("Timeline contains no buckets")
    common: dict[str, Any] = {**filters, "bbox": args.bbox, "limit": args.limit}

    legacy_times: list[float] = []
    legacy_bytes = 0
    for bucket in buckets:
        _, elapsed, size = request_json(
            "GET", query_url(args.base_url, f"/api/frame/{bucket}", common)
        )
        legacy_times.append(elapsed)
        legacy_bytes += size

    compact_times: list[float] = []
    geometry_times: list[float] = []
    compact_bytes = 0
    geometry_bytes = 0
    geometry_cache: set[str] = set()
    geometry_requests = 0
    for bucket in buckets:
        state, elapsed, size = request_json(
            "GET", query_url(args.base_url, f"/api/frame-state/{bucket}", common)
        )
        compact_times.append(elapsed)
        compact_bytes += size
        missing = [
            str(row["field_id"])
            for row in state.get("rows", [])
            if str(row["field_id"]) not in geometry_cache
        ]
        for offset in range(0, len(missing), 2000):
            batch = missing[offset : offset + 2000]
            payload, geometry_elapsed, geometry_size = request_json(
                "POST",
                f"{args.base_url.rstrip('/')}/api/geometry",
                {"geometry_version": state["geometry_version"], "field_ids": batch},
            )
            geometry_times.append(geometry_elapsed)
            compact_bytes += geometry_size
            geometry_bytes += geometry_size
            geometry_requests += 1
            geometry_cache.update(
                str(feature["properties"]["field_id"])
                for feature in payload.get("features", [])
            )

    legacy = summarize(legacy_times, legacy_bytes)
    compact = summarize(compact_times, compact_bytes)
    reduction = 1 - (compact_bytes / legacy_bytes) if legacy_bytes else None
    report = {
        "weeks": buckets,
        "query": common,
        "legacy": legacy,
        "geometry_once": {
            **compact,
            "geometry_requests": geometry_requests,
            "unique_geometry_fields": len(geometry_cache),
            "geometry_bootstrap": summarize(geometry_times, geometry_bytes),
        },
        "compressed_byte_reduction": round(reduction, 4) if reduction is not None else None,
        "gates": {
            "subsequent_request_p95_below_300ms": bool(
                compact["p95_ms"] is not None and compact["p95_ms"] < 300
            ),
            "compressed_bytes_reduced_70pct": bool(reduction is not None and reduction >= 0.70),
        },
        "note": "Network/server benchmark only; record browser parse/layer-update separately.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} from {exc.url}: {exc.read().decode(errors='replace')}") from exc
