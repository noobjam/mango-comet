#!/usr/bin/env python3
"""Benchmark daily Incident V4 timeline scrubbing on the deployment VM."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--random-requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--filter", action="append", default=[], help="key=value")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fetch(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
    )
    started = time.perf_counter()
    try:
        response = urlopen(request, timeout=180)
    except HTTPError as exc:
        wire = exc.read()
        return {
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "wire_bytes": len(wire),
            "decoded_bytes": len(wire),
            "feature_count": 0,
            "cache": exc.headers.get("X-Cache"),
            "http_status": int(exc.code),
            "error": wire.decode("utf-8", errors="replace")[:500],
        }
    with response:
        wire = response.read()
        decoded = (
            gzip.decompress(wire)
            if response.headers.get("Content-Encoding") == "gzip"
            else wire
        )
        payload = json.loads(decoded)
        meta = payload.get("meta") or {}
        feature_count = sum(
            len((payload.get(name) or {}).get("features") or [])
            for name in (
                "field_overview", "pressure", "crop_impact",
                "story_footprints", "fields",
            )
        )
        return {
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "wire_bytes": len(wire),
            "decoded_bytes": len(decoded),
            "feature_count": feature_count,
            "cache": response.headers.get("X-Cache"),
            "http_status": int(response.status),
            "complete_country_representation": bool(
                meta.get("complete_country_representation")
            ),
            "source_day_present": bool(meta.get("source_day_present")),
            "country_representation_truncated": bool(
                meta.get("country_representation_truncated")
            ),
            "source_field_count": int(meta.get("source_field_count") or 0),
            "represented_field_count": int(meta.get("represented_field_count") or 0),
            "unmappable_field_count": int(meta.get("unmappable_field_count") or 0),
            "accounted_field_count": int(meta.get("accounted_field_count") or 0),
            "unmappable_warning": bool(meta.get("unmappable_warning")),
        }


def frame_url(base: str, day: str, query: dict[str, str]) -> str:
    suffix = urlencode(query)
    return f"{base.rstrip('/')}/api/v4/frame/{day}" + (f"?{suffix}" if suffix else "")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def rounded(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row for row in rows if 200 <= int(row.get("http_status", 0)) < 300
    ]
    times = [float(row["elapsed_ms"]) for row in successful]
    wire = [int(row["wire_bytes"]) for row in successful]
    decoded = [int(row["decoded_bytes"]) for row in successful]
    return {
        "requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "p50_ms": rounded(percentile(times, 0.50)),
        "p95_ms": rounded(percentile(times, 0.95)),
        "max_ms": rounded(max(times) if times else None),
        "median_wire_bytes": int(statistics.median(wire)) if wire else 0,
        "max_wire_bytes": max(wire, default=0),
        "median_decoded_bytes": int(statistics.median(decoded)) if decoded else 0,
        "max_decoded_bytes": max(decoded, default=0),
        "complete_country_responses": sum(
            bool(row.get("complete_country_representation")) for row in successful
        ),
        "responses_with_unmappable_warning": sum(
            bool(row.get("unmappable_warning")) for row in successful
        ),
        "cache_statuses": {
            name: sum(str(row.get("cache") or "NONE") == name for row in rows)
            for name in sorted({str(row.get("cache") or "NONE") for row in rows})
        },
        "http_statuses": {
            str(status): sum(int(row.get("http_status", 0)) == status for row in rows)
            for status in sorted({int(row.get("http_status", 0)) for row in rows})
        },
    }


def rss_bytes(pid: int | None) -> int | None:
    if not pid:
        return None
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _filters(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"Invalid --filter {item!r}; expected key=value")
        key, value = item.split("=", 1)
        if key not in {"crop_name", "hazard_family", "stage_bucket"}:
            raise SystemExit(f"Unsupported V4 filter: {key}")
        parsed[key] = value
    return parsed


def main() -> None:
    args = parse_args()
    if args.days < 2 or args.random_requests < 1 or args.concurrency < 1:
        raise SystemExit("--days >=2, --random-requests >=1, and --concurrency >=1")
    filters = _filters(args.filter)
    with urlopen(f"{args.base_url.rstrip('/')}/api/v4/timeline", timeout=60) as response:
        timeline = json.load(response)
    days = [str(row["calendar_date"])[:10] for row in timeline.get("days", [])]
    days = days[-args.days :]
    if len(days) < 2:
        raise SystemExit("Incident V4 timeline has fewer than two days")

    if len(days) < 3:
        raise SystemExit("Incident V4 benchmark requires at least three daily frames")
    before_rss = rss_bytes(args.server_pid)
    burst_count = min(args.concurrency, max(1, len(days) // 3))
    adjacent_days = days[-burst_count:]
    sequential_days = days[:-burst_count]
    cold_concurrent_targets = [
        frame_url(args.base_url, day, filters) for day in adjacent_days
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        cold_concurrent = list(executor.map(fetch, cold_concurrent_targets))
    cold_concurrent_wall_ms = (time.perf_counter() - started) * 1000

    cold_targets = [frame_url(args.base_url, day, filters) for day in sequential_days]
    cold = [fetch(target) for target in cold_targets]
    warm = [fetch(target) for target in cold_targets]

    base_targets = [frame_url(args.base_url, day, filters) for day in days]
    for target in base_targets:
        fetch(target)
    rng = random.Random(17)
    random_rows = [fetch(rng.choice(base_targets)) for _ in range(args.random_requests)]
    adjacent = base_targets[-min(len(base_targets), args.concurrency) :]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        concurrent = list(executor.map(fetch, adjacent))
    concurrent_wall_ms = (time.perf_counter() - started) * 1000
    after_rss = rss_bytes(args.server_pid)

    cold_summary = summary(cold)
    cold_concurrent_summary = summary(cold_concurrent)
    warm_summary = summary(warm)
    random_summary = summary(random_rows)
    concurrent_summary = summary(concurrent)
    all_complete = all(
        row.get("complete_country_representation") is True
        for rows in (cold, warm, cold_concurrent, random_rows, concurrent)
        for row in rows
        if 200 <= int(row.get("http_status", 0)) < 300
    )
    all_accounted = all(
        row.get("source_day_present") is True
        and not row.get("country_representation_truncated")
        and int(row.get("accounted_field_count", -1))
            == int(row.get("source_field_count", -2))
        for rows in (cold, warm, cold_concurrent, random_rows, concurrent)
        for row in rows
        if 200 <= int(row.get("http_status", 0)) < 300
    )
    report = {
        "mode": "crop_incident_v4_dual_clock",
        "days": days,
        "filters": filters,
        "cold_unique": cold_summary,
        "cold_concurrent_adjacent": {
            **cold_concurrent_summary,
            "wall_ms": rounded(cold_concurrent_wall_ms),
            "workers": args.concurrency,
        },
        "warm_same_url": warm_summary,
        "random_scrub_cached": random_summary,
        "concurrent_adjacent": {
            **concurrent_summary,
            "wall_ms": rounded(concurrent_wall_ms),
            "workers": args.concurrency,
        },
        "server_rss": {
            "pid": args.server_pid,
            "before_bytes": before_rss,
            "after_bytes": after_rss,
            "delta_bytes": (
                after_rss - before_rss
                if before_rss is not None and after_rss is not None else None
            ),
        },
        "gates": {
            "all_country_responses_complete": all_complete,
            "all_source_fields_accounted": all_accounted,
            "cold_requests_were_cache_misses": all(
                str(row.get("cache") or "") == "MISS"
                for row in (*cold, *cold_concurrent)
            ),
            "cold_p95_below_1500ms": bool(
                cold_summary["p95_ms"] is not None
                and cold_summary["p95_ms"] < 1500
                and cold_summary["failed_requests"] == 0
            ),
            "cold_concurrent_p95_below_2500ms": bool(
                cold_concurrent_summary["p95_ms"] is not None
                and cold_concurrent_summary["p95_ms"] < 2500
                and cold_concurrent_summary["failed_requests"] == 0
            ),
            "warm_p95_below_250ms": bool(
                warm_summary["p95_ms"] is not None
                and warm_summary["p95_ms"] < 250
                and warm_summary["failed_requests"] == 0
            ),
            "random_scrub_p95_below_300ms": bool(
                random_summary["p95_ms"] is not None
                and random_summary["p95_ms"] < 300
                and random_summary["failed_requests"] == 0
            ),
            "cached_concurrent_p95_below_500ms": bool(
                concurrent_summary["p95_ms"] is not None
                and concurrent_summary["p95_ms"] < 500
                and concurrent_summary["failed_requests"] == 0
            ),
        },
        "browser_acceptance_required": {
            "timeline_scrub": "record Performance over 28 adjacent daily frames",
            "render_p95_ms": "target <100 ms after response arrival",
            "long_task_ms": "no repeated >50 ms main-thread tasks",
            "heap": "stable after two back-and-forth scrub passes",
            "visual": (
                "complete country grid; daily pressure, S2 evidence, and weekly "
                "story checkpoints remain visually distinct"
            ),
        },
        "measurement_note": (
            "Run immediately after starting a fresh server; unknown query-string "
            "cache busters are intentionally canonicalized away."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not all(bool(value) for value in report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
