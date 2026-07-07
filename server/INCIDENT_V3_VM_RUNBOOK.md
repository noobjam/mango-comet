# Crop-Impact Incident V3 — authoritative VM runbook

This sequence builds an immutable crop-incident release, exports the map-ready
viewer bundle, verifies it, starts the app, and measures timeline latency. The
runner writes durable state and can resume completed stages safely.

V3 identity is:

```text
weekly component_* -> persistent exposure_* -> crop-specific incident_*
```

Crop stage is changing evidence context, never identity. Thresholds are hashed
but still uncalibrated; successful execution does not establish crop death,
causation, propagation, or agronomic validity.

Stage bucketing is exact and crop-qualified. The frozen policy checks
`(crop_name, source stage)` aliases before a small set of canonical or genuinely
generic aliases. Crop-specific labels cannot fall through a global alias for a
different crop; unlisted labels remain `unknown`, and the 80% global / 70%
supported-crop coverage gates still stop publication. The current Rwanda
mappings follow the source metadata crosswalk for maize, rice, wheat, Irish
potatoes, and generic, bush, and climbing beans.

The context builder accepts a valid source `centroid_lon`/`centroid_lat` pair
when present. Otherwise it derives the centroid from a Polygon/MultiPolygon in
`geometry_geojson`, `geometry_text`, `geometry_wkt`, or text/binary `geometry`.
The source geometry contract is WGS84 longitude/latitude; the adapter validates
finite global coordinate bounds but cannot infer a missing CRS or axis-order
declaration. Invalid geometry remains without a centroid; it is not imputed,
and the unchanged 95% field and crop-instance-week gates stop publication.

## 1. Exact paths and preflight

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet
source .venv/bin/activate

export REPO=/mnt/KSA-Oasis/El-Mohammed/mango-comet
export PYTHON=/mnt/KSA-Oasis/El-Mohammed/mango-comet/.venv/bin/python
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export GEN=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/generations/2026-05-17_generation_7a715df05da10c3b3300
export BASELINE_THROUGH=2025-12-31
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

test -x "$PYTHON"
test -f "$GEN/manifest.json"
test -f "$GEN/daily_causal_signals.parquet"
test -f "$GEN/event_state_snapshots.parquet"
test -f "$GEN/event_windows.parquet"
test -f "$GEN/story_day_membership.parquet"
test -f "$GEN/map_field_geometry.parquet"
mkdir -p "$ROOT/duckdb_tmp" "$ROOT/logs" "$ROOT/jobs" "$ROOT/releases"

git pull --ff-only
git status --short
node --version
"$PYTHON" --version
```

The source generation was built from the echo-aware parquet. The V3 runner
consumes its immutable causal artifacts; it does not rescan the original daily
parquet.

## 2. First V3 release

Use `--first-release` exactly once. The runner executes Python tests, Node UI
tests, JavaScript syntax checks, the V3 build, viewer export, and server smoke.

```bash
export JOB_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export JOB_DIR="$ROOT/jobs/incident_v3_$JOB_TAG"
export LAUNCH_LOG="$ROOT/logs/incident_v3_launcher_$JOB_TAG.log"

nohup "$PYTHON" server/run_incident_v3.py run \
  --root "$ROOT" \
  --generation-dir "$GEN" \
  --python "$PYTHON" \
  --baseline-through "$BASELINE_THROUGH" \
  --first-release \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  --capture-stage9-replay \
  --job-tag "$JOB_TAG" \
  >"$LAUNCH_LOG" 2>&1 </dev/null &

export LAUNCHER_PID=$!
printf 'PID=%s\nJOB_DIR=%s\nLOG=%s\n' \
  "$LAUNCHER_PID" "$JOB_DIR" "$LAUNCH_LOG"
```

Do not add RAPIDS to this command. DuckDB, causal state construction, exact
grid unions, and HTTP bundle export are CPU/memory/I/O work; H100s do not reduce
interactive GeoJSON or browser latency.

## 3. Monitor, inspect failure, and resume

```bash
"$PYTHON" server/run_incident_v3.py status --job-dir "$JOB_DIR"
tail -f "$JOB_DIR/runner.log"
```

After reconnecting in a new shell:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet
source .venv/bin/activate
export PYTHON=/mnt/KSA-Oasis/El-Mohammed/mango-comet/.venv/bin/python
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export BASELINE_THROUGH=2025-12-31
export JOB_DIR=$(cat "$ROOT/logs/latest_incident_v3_job.txt")
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

"$PYTHON" server/run_incident_v3.py status --job-dir "$JOB_DIR"
tail -n 200 "$JOB_DIR/runner.log"
cat "$JOB_DIR/status" 2>/dev/null || echo "still running"
```

If the process stopped after a completed stage, resume the same immutable job:

```bash
export RESUME_LOG="$JOB_DIR/resume.launcher.log"
nohup "$PYTHON" server/run_incident_v3.py resume \
  --job-dir "$JOB_DIR" \
  >"$RESUME_LOG" 2>&1 </dev/null &
echo "$!" | tee "$JOB_DIR/resume.pid"
```

Never delete a partial release or reuse a job tag to conceal a failure. Read the
stage-specific `*.stderr.log` recorded under `JOB_DIR`.

When `--capture-stage9-replay` is enabled and the stage-9 story finalizer fails,
the runner preserves its exact failing call in
`JOB_DIR/stage9-finalizer-capsule`. A job that never fails creates no capsule.
The first capsule in a job is immutable: a later resume never overwrites it, so
start a fresh job if a different failure must be captured. After pulling a
candidate code fix, verify and replay that call once without rebuilding stages
1-8 or publishing anything:

Capture fails closed above a 32 GiB input-memory estimate or 16 GiB serialized
size, and requires an additional 8 GiB free-space reserve. A limit breach
removes the partial capsule and never replaces the original build exception.

```bash
test -f "$JOB_DIR/stage9-finalizer-capsule/manifest.json"
"$PYTHON" server/weekly_story_monitor.py \
  replay-incidents-v3-finalizer \
  --capsule-dir "$JOB_DIR/stage9-finalizer-capsule"
```

Replay verifies every captured Parquet/JSON hash before loading data. A replay
success is only a focused regression check; run the normal immutable pipeline
again before treating any release as built.

## 4. Resolve and audit successful outputs

```bash
test "$(cat "$JOB_DIR/status")" = "0"

export INCIDENT_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["incident_dir"])' \
  "$JOB_DIR/state.json")
export VIEWER_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["viewer_dir"])' \
  "$JOB_DIR/state.json")

test -f "$INCIDENT_DIR/manifest.json"
test -f "$VIEWER_DIR/manifest.json"
printf 'INCIDENT_DIR=%s\nVIEWER_DIR=%s\n' "$INCIDENT_DIR" "$VIEWER_DIR"
"$PYTHON" -m json.tool "$JOB_DIR/build.json"
"$PYTHON" -m json.tool "$JOB_DIR/export.json"
```

Inspect crop-stage mapping before presenting:

```bash
"$PYTHON" -c '
import json, os
p = os.path.join(os.environ["INCIDENT_DIR"], "context_manifest.json")
m = json.load(open(p))
c = m["counts"]
print("overall known-stage coverage:", c["source_known_stage_coverage"])
print("per crop:")
for row in c["stage_coverage_by_crop"]:
    print(row)
print("top unmapped raw labels:")
for row in c["top_unmapped_stage_labels"]:
    print(row)
'
```

The build fails if overall known-stage coverage is below 80%, or a crop with at
least 100 crop-instance-weeks is below 70%. Fix versioned aliases and rebuild;
do not relabel unknown stages ad hoc in the UI.

Inspect story/lifecycle volumes:

```bash
"$PYTHON" -c '
import duckdb, os
root = os.environ["INCIDENT_DIR"]
for name in (
    "weekly_components", "exposure_weekly_state", "incident_weekly_state",
    "incident_stage_summary", "incident_membership", "incident_windows",
    "incident_lineage",
):
    n = duckdb.sql("SELECT COUNT(*) FROM read_parquet(?)", params=[f"{root}/{name}.parquet"]).fetchone()[0]
    print(f"{name}: {n:,}")
print(duckdb.sql("""
  SELECT crop_name, current_state, COUNT(*) AS story_weeks,
         SUM(affected_count) AS affected_crop_instance_weeks
  FROM read_parquet(?) GROUP BY 1,2 ORDER BY story_weeks DESC LIMIT 40
""", params=[f"{root}/incident_weekly_state.parquet"]).df().to_string(index=False))
'
```

## 5. Weekly append release

Build the next immutable source generation first. Then use the prior
`INCIDENT_DIR`, not the prior viewer directory:

```bash
export PREVIOUS_INCIDENT_DIR=/absolute/path/to/prior/incidents_v3_TAG
export GEN=/absolute/path/to/new/immutable/source_generation
export JOB_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export JOB_DIR="$ROOT/jobs/incident_v3_$JOB_TAG"
export LAUNCH_LOG="$ROOT/logs/incident_v3_launcher_$JOB_TAG.log"

nohup "$PYTHON" server/run_incident_v3.py run \
  --root "$ROOT" \
  --generation-dir "$GEN" \
  --python "$PYTHON" \
  --baseline-through "$BASELINE_THROUGH" \
  --previous-incident-dir "$PREVIOUS_INCIDENT_DIR" \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  --capture-stage9-replay \
  --job-tag "$JOB_TAG" \
  >"$LAUNCH_LOG" 2>&1 </dev/null &
```

Append validation rejects rewritten historical components, exposure IDs,
incident evidence/lifecycle, crop-stage denominators, memberships, lineage, and
terminal history. For a prior maximum-week row explicitly marked
`data_censored_at_boundary`, only `incident_state`, `current_state`,
`closed_week`, `right_censored`, and the boundary flag may reopen; onset,
confirmation, pressure/recovery milestones, lineage targets, and cumulative
counters remain frozen, and the reopened state must equal the immediately
preceding published nonterminal state. A prior right-censored window—or its matching terminal
`CLOSED_DATA_CENSORED` boundary window—may extend only when later weekly rows
support the change. Its immutable pre-boundary evidence cannot change.

## 6. Launch the map-ready V3 bundle

First re-resolve the paths from the successful job you intend to serve. This is
required after an append; do not reuse `VIEWER_DIR` from the prior shell state:

```bash
test "$(cat "$JOB_DIR/status")" = "0"
export INCIDENT_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["incident_dir"])' \
  "$JOB_DIR/state.json")
export VIEWER_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["viewer_dir"])' \
  "$JOB_DIR/state.json")
test -f "$INCIDENT_DIR/manifest.json"
test -f "$VIEWER_DIR/manifest.json"
```

Write `server/.env` with that actual exported viewer path:

```bash
cat > /tmp/incident-v3.env <<EOF
STORY_MAP_RUN_DIR=$VIEWER_DIR
STORY_MAP_STATIC_DIR=./static
STORY_MAP_HOST=127.0.0.1
STORY_MAP_PORT=8877
STORY_MAP_LOG_LEVEL=INFO
STORY_MAP_RASTER_TILES=https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
STORY_MAP_RASTER_ATTRIBUTION=Tiles (C) Esri, Maxar, Earthstar Geographics, and the GIS User Community
STORY_MAP_DEFAULT_FEATURE_LIMIT=5000
STORY_MAP_MAX_FEATURE_LIMIT=20000
STORY_MAP_CACHE_SECONDS=300
STORY_MAP_CACHE_ENTRIES=256
STORY_MAP_CACHE_MAX_BYTES=536870912
STORY_MAP_QUERY_CONCURRENCY=8
STORY_MAP_GZIP_MIN_BYTES=1024
EOF
cp /tmp/incident-v3.env server/.env
```

Start on loopback:

```bash
# Refuse to let a stale server make the health check validate an old release.
if "$PYTHON" -c 'import socket; s=socket.socket(); s.settimeout(1); raise SystemExit(0 if s.connect_ex(("127.0.0.1", 8877)) == 0 else 1)'; then
  echo "Port 8877 is already in use; stop or intentionally promote the old server first."
  exit 1
fi

# story_map_server.py deliberately lets exported variables override .env.
# Clear stale values so the exact file written above is authoritative.
unset STORY_MAP_RUN_DIR STORY_MAP_STATIC_DIR STORY_MAP_HOST STORY_MAP_PORT
unset STORY_MAP_LOG_LEVEL STORY_MAP_RASTER_TILES STORY_MAP_RASTER_ATTRIBUTION
unset STORY_MAP_DEFAULT_FEATURE_LIMIT STORY_MAP_MAX_FEATURE_LIMIT
unset STORY_MAP_CACHE_SECONDS STORY_MAP_CACHE_ENTRIES STORY_MAP_CACHE_MAX_BYTES
unset STORY_MAP_QUERY_CONCURRENCY STORY_MAP_GZIP_MIN_BYTES

export SERVER_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export SERVER_LOG="$ROOT/logs/incident_v3_server_$SERVER_TAG.log"
export SERVER_PID="$ROOT/logs/incident_v3_server_$SERVER_TAG.pid"

nohup "$PYTHON" server/story_map_server.py \
  >"$SERVER_LOG" 2>&1 </dev/null &
echo "$!" | tee "$SERVER_PID"
sleep 3
kill -0 "$(cat "$SERVER_PID")"
curl -fsS http://127.0.0.1:8877/api/health | "$PYTHON" -m json.tool
curl -fsS http://127.0.0.1:8877/api/timeline | "$PYTHON" -m json.tool | head -n 60

export EXPECTED_VIEWER_ID=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["run"]["viewer_bundle_id"])' \
  "$VIEWER_DIR/manifest.json")
export SERVED_VIEWER_ID=$(curl -fsS http://127.0.0.1:8877/api/manifest | \
  "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["run"]["viewer_bundle_id"])')
test "$SERVED_VIEWER_ID" = "$EXPECTED_VIEWER_ID"
printf 'Serving verified viewer bundle: %s\n' "$SERVED_VIEWER_ID"
```

From your laptop:

```bash
ssh -N -L 8877:127.0.0.1:8877 opc@YOUR_VM_HOST
```

Open `http://127.0.0.1:8877`. At country zoom every filtered incident is an
exact grid-cell footprint. Fields appear only after zooming in for evidence
drill-down. Selecting a story shows crop stage, lifecycle, pressure/impact/
unresolved counts, and exact age-banded prior footprints—never centroid motion.

## 7. VM latency and visual acceptance

```bash
export SERVER_PID_VALUE=$(cat "$SERVER_PID")
"$PYTHON" server/benchmark_incident_v3.py \
  --base-url http://127.0.0.1:8877 \
  --weeks 20 \
  --random-requests 60 \
  --concurrency 8 \
  --server-pid "$SERVER_PID_VALUE" \
  --output "$ROOT/logs/incident_v3_benchmark_$SERVER_TAG.json"
```

Do not present a latency claim unless the saved full-VM result has no failed
requests and passes sequential-cold p95 under 1.5 s, concurrent cold-adjacent
p95 under 2.5 s, warm p95 under 250 ms, random scrub p95 under 300 ms, and
prewarmed concurrent-adjacent p95 under 500 ms. A 503 in either cold burst is a
failed gate, even if the successful-request latency looks fast. Also record a
browser Performance trace over 20 adjacent weeks: response-to-render p95 under
100 ms, no repeated main-thread tasks over 50 ms, and stable heap after two
back-and-forth passes. Record wire bytes, decoded JSON bytes, HTTP status
counts, and server RSS.

Visual acceptance:

- country zoom never drops complete story footprints;
- coincident crops remain selectable;
- crop and crop stage are visible in the inspector/story arc;
- selected history uses exact polygon outlines only;
- no line, arrow, or animation implies physical movement;
- low-coverage/data-censored states are visually explicit;
- field polygons are a high-zoom evidence layer, not the national overview.

## 8. Optional archetype discovery

Completed-story HDBSCAN is diagnostic and separate from story identity. It
excludes data/season-censored and merged fragments, and prefix publication is
blocked until an immutable expert-review overlay and holdout evaluation exist.
One H100 is sufficient when that experiment is authorized; eight GPUs do not
improve the live map path.
