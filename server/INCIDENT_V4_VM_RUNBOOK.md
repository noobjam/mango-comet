# Crop-incident V4 — authoritative VM runbook

This is the no-shortcut path for the current VM. It prepares a rich daily
source, keeps daily weather and irregular Sentinel-2 on separate evidence
clocks, projects the immutable weekly V3 story spine onto a daily playhead,
exports the map bundle, verifies it, and measures real latency.

The current historical files do not contain original ingestion timestamps, so
the first build must use `reconstructed` availability and remains diagnostic.
Operational monitoring must switch to `strict` after the upstream pipeline
retains `weather_available_at`, `spectral_available_at`, and
`stage_available_at`. Its V3 incident release must also retain explicit,
non-inferred checkpoint knowledge timestamps and complete membership
attribution; strict V4 export fails closed when either is absent.

## 1. Pull, paths, and preflight

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet
source .venv/bin/activate

export REPO=/mnt/KSA-Oasis/El-Mohammed/mango-comet
export PYTHON=/mnt/KSA-Oasis/El-Mohammed/mango-comet/.venv/bin/python
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export GEN=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/generations/2026-05-17_generation_7a715df05da10c3b3300
export ECHO=/mnt/KSA-Oasis/fields_health_v2/rwanda_crop_risk_kb/final_field_daily_v4/rwanda_2025_2026_field_daily_risk_DELIVERABLE_WITH_CROP_AND_RISK_DRIVER_v4_WITH_SPECTRAL_ECHO_DAYS.parquet
export FULL_2025=/mnt/KSA-Oasis/fields_health/KB/rwanda_crop_risk_kb_vm_transfer_20260513/rwanda_crop_risk_kb/final_field_daily_v3/rwanda_2025_field_daily_risk.parquet
export FULL_2026=/mnt/KSA-Oasis/fields_health_v2/rwanda_crop_risk_kb/final_field_daily_v4/rwanda_2026_field_daily_risk_v4.parquet
export ACQ_2026=/mnt/KSA-Oasis/fields_health_v2/rwanda_crop_risk_kb/final_field_daily_v4/field_s2_prediction_weekly_v4.parquet
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

git pull --ff-only
git status --short

test -x "$PYTHON"
test -f "$GEN/manifest.json"
test -f "$ECHO"
test -f "$FULL_2025"
test -f "$FULL_2026"
test -f "$ACQ_2026"
node --version
"$PYTHON" --version
mkdir -p "$ROOT/duckdb_tmp" "$ROOT/logs" "$ROOT/jobs" "$ROOT/releases" "$ROOT/sources"
```

Resolve the successful V3 story release. V4 deliberately reuses its stable
`component -> exposure -> crop incident` identity; it does not recluster fields
into a new identity.

```bash
export V3_JOB=$(cat "$ROOT/logs/latest_incident_v3_job.txt")
test "$(cat "$V3_JOB/status")" = "0"
export INCIDENT_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["incident_dir"])' \
  "$V3_JOB/state.json")
test -f "$INCIDENT_DIR/manifest.json"
printf 'GEN=%s\nINCIDENT_DIR=%s\n' "$GEN" "$INCIDENT_DIR"
```

If `latest_incident_v3_job.txt` does not exist, complete
[`INCIDENT_V3_VM_RUNBOOK.md`](INCIDENT_V3_VM_RUNBOOK.md) first. Do not point V4
at the old V1 motif model or the Archetype V2 diagnostic output.

## 2. Start the complete V4 build

The durable runner performs tests, source enrichment, three evidence ledgers,
append/truth validation, viewer export, and an API smoke test. Source enrichment
is inside the runner, so its PID, RSS, elapsed time, stderr, and restart state
are recorded instead of disappearing behind a silent DuckDB command.

`ACQ_2026` is used twice: as QA enrichment while preparing the daily source and
as a partial attempt ledger while building evidence. External attempts are
merged by field/source date with acquisitions derived from the complete
2025–2026 source; they never replace 2025 history. This retains unmatched
rejected attempts and deduplicates overlaps. The enriched-source sidecar mode
must exactly match `--availability-mode`, and enriched field/day keys must
exactly reconcile with the selected generation.

```bash
export JOB_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export RELEASED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export JOB_DIR="$ROOT/jobs/incident_v4_$JOB_TAG"
export LAUNCH_LOG="$ROOT/logs/incident_v4_launcher_$JOB_TAG.log"

nohup "$PYTHON" server/run_incident_v4.py run \
  --root "$ROOT" \
  --generation-dir "$GEN" \
  --incident-dir "$INCIDENT_DIR" \
  --echo-deliverable "$ECHO" \
  --full-parquet "$FULL_2025" \
  --full-parquet "$FULL_2026" \
  --source-acquisition-parquet "$ACQ_2026" \
  --acquisition-parquet "$ACQ_2026" \
  --availability-mode reconstructed \
  --released-at "$RELEASED_AT" \
  --first-release \
  --python "$PYTHON" \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  --job-tag "$JOB_TAG" \
  >"$LAUNCH_LOG" 2>&1 </dev/null &

export LAUNCHER_PID=$!
printf 'PID=%s\nJOB_DIR=%s\nLOG=%s\n' \
  "$LAUNCHER_PID" "$JOB_DIR" "$LAUNCH_LOG"
```

Do not add RAPIDS to this command. The heavy operations here are DuckDB scans,
joins, Parquet writes, spatial aggregation, JSON serving, and browser rendering.
H100s help only the later offline HDBSCAN experiment.

## 3. Monitor and resume

```bash
"$PYTHON" server/run_incident_v4.py status --job-dir "$JOB_DIR"
tail -f "$JOB_DIR/runner.log"
```

After reconnecting:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet
source .venv/bin/activate
export PYTHON=/mnt/KSA-Oasis/El-Mohammed/mango-comet/.venv/bin/python
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export JOB_DIR=$(cat "$ROOT/logs/latest_incident_v4_job.txt")

"$PYTHON" server/run_incident_v4.py status --job-dir "$JOB_DIR"
cat "$JOB_DIR/status" 2>/dev/null || echo "still running"
tail -n 200 "$JOB_DIR/runner.log"
```

Resume a stopped job without rerunning completed immutable stages:

```bash
export RESUME_LOG="$JOB_DIR/resume.launcher.log"
nohup "$PYTHON" server/run_incident_v4.py resume \
  --job-dir "$JOB_DIR" \
  >"$RESUME_LOG" 2>&1 </dev/null &
echo "$!" | tee "$JOB_DIR/resume.pid"
```

Never delete an incomplete output just to reuse its job tag. Read the exact
stage `*.stderr.log`; a generated source without its complete immutable
manifest intentionally blocks resume.

## 4. Resolve and audit the release

```bash
test "$(cat "$JOB_DIR/status")" = "0"
export ENRICHED_SOURCE=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["config"]["enriched_source_parquet"])' \
  "$JOB_DIR/state.json")
export EVIDENCE_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["evidence_dir"])' \
  "$JOB_DIR/state.json")
export VIEWER_DIR=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["viewer_dir"])' \
  "$JOB_DIR/state.json")

test -f "$ENRICHED_SOURCE.manifest.json"
test -f "$EVIDENCE_DIR/manifest.json"
test -f "$VIEWER_DIR/manifest.json"
printf 'ENRICHED_SOURCE=%s\nEVIDENCE_DIR=%s\nVIEWER_DIR=%s\n' \
  "$ENRICHED_SOURCE" "$EVIDENCE_DIR" "$VIEWER_DIR"
PYTHONPATH=server "$PYTHON" -c \
  'import json,os; from pathlib import Path; from story_monitor.incident_viewer_v4 import validate_viewer_directory; print(json.dumps(validate_viewer_directory(Path(os.environ["VIEWER_DIR"])),indent=2))'

"$PYTHON" -c '
import json, os
manifest = json.load(open(os.path.join(os.environ["VIEWER_DIR"], "manifest.json")))
assert manifest["schema_version"] == "crop-incident-viewer-v4/2", manifest["schema_version"]
assert manifest["semantics"]["lifecycle_state_recomputed_from_v4"] is False
assert manifest["semantics"]["lifecycle_causal_ownership_claimed"] is False
print("Fresh V4/2 viewer and bounded reconciliation semantics confirmed")
'
```

Do not reuse a V4/1 viewer directory. V4/2 requires the immutable lifecycle
reconciliation artifact and the server rejects a missing, altered, or
contradictory ledger.

Audit the three clocks and simultaneous hazard lanes:

```bash
"$PYTHON" -c '
import duckdb, json, os
e = os.environ["EVIDENCE_DIR"]
con = duckdb.connect()
for name in ("crop_day_context_v4", "field_day_pressure_v4", "field_s2_acquisition_v4"):
    n = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [f"{e}/{name}.parquet"]).fetchone()[0]
    print(f"{name}: {n:,}")
print(con.execute("""
  SELECT hazard_family, COUNT(*) AS rows,
         COUNT_IF(pressure_observed) AS observed,
         COUNT_IF(pressure_active) AS active
  FROM read_parquet(?) GROUP BY 1 ORDER BY 1
""", [f"{e}/field_day_pressure_v4.parquet"]).fetchdf().to_string(index=False))
print(con.execute("""
  SELECT acquisition_status, COUNT(*) AS attempts,
         COUNT_IF(spectral_usable) AS usable,
         COUNT_IF(new_response_evidence) AS comparable
  FROM read_parquet(?) GROUP BY 1 ORDER BY attempts DESC
""", [f"{e}/field_s2_acquisition_v4.parquet"]).fetchdf().to_string(index=False))
manifest = json.load(open(f"{e}/manifest.json"))
source_manifest = json.load(open(os.environ["ENRICHED_SOURCE"] + ".manifest.json"))
print("released_at:", manifest["run"]["released_at"])
assert source_manifest["released_at"] == manifest["run"]["released_at"]
print(json.dumps(manifest["reconciliation"], indent=2, sort_keys=True))
assert manifest["reconciliation"]["source_field_day"]["exact_key_coverage"] is True
v = os.environ["VIEWER_DIR"]
print(con.execute("""
  SELECT COUNT(*) AS pressure_observations,
         COUNT_IF(CAST(pressure_knowledge_time AS DATE) > pressure_effective_date)
           AS late_known_observations
  FROM read_parquet(?)
""", [f"{v}/pressure_observations_v4.parquet"]).fetchdf().to_string(index=False))
print(con.execute("""
  SELECT reconciliation_status, quiet_evidence_status,
         COUNT(*) AS checkpoints, SUM(contradiction_count) AS contradictions
  FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1,2
""", [f"{v}/lifecycle_reconciliation_v4.parquet"]).fetchdf().to_string(index=False))
assert con.execute(
  "SELECT COUNT(*) FROM read_parquet(?) WHERE contradiction_count > 0",
  [f"{v}/lifecycle_reconciliation_v4.parquet"],
).fetchone()[0] == 0
'
```

The release is invalid if only one hazard family exists, if echo days created
extra acquisition rows, if rejected attempts became references, or if the
manifest does not say `diagnostic_reconstruction: true` for this historical
build.

## 5. Serve the V4 viewer

```bash
cat > /tmp/incident-v4.env <<EOF
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
cp /tmp/incident-v4.env server/.env

if "$PYTHON" -c 'import socket; s=socket.socket(); s.settimeout(1); raise SystemExit(0 if s.connect_ex(("127.0.0.1",8877)) == 0 else 1)'; then
  echo "Port 8877 is already serving another release; stop it before validation."
  exit 1
fi

unset STORY_MAP_RUN_DIR STORY_MAP_STATIC_DIR STORY_MAP_HOST STORY_MAP_PORT
unset STORY_MAP_LOG_LEVEL STORY_MAP_RASTER_TILES STORY_MAP_RASTER_ATTRIBUTION
unset STORY_MAP_DEFAULT_FEATURE_LIMIT STORY_MAP_MAX_FEATURE_LIMIT
unset STORY_MAP_CACHE_SECONDS STORY_MAP_CACHE_ENTRIES STORY_MAP_CACHE_MAX_BYTES
unset STORY_MAP_QUERY_CONCURRENCY STORY_MAP_GZIP_MIN_BYTES

export SERVER_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export SERVER_LOG="$ROOT/logs/incident_v4_server_$SERVER_TAG.log"
export SERVER_PID="$ROOT/logs/incident_v4_server_$SERVER_TAG.pid"
nohup "$PYTHON" server/story_map_server.py \
  >"$SERVER_LOG" 2>&1 </dev/null &
echo "$!" | tee "$SERVER_PID"

SERVER_READY=0
for attempt in $(seq 1 150); do
  if curl -fsS --max-time 2 http://127.0.0.1:8877/api/health >/dev/null 2>&1; then
    SERVER_READY=1
    break
  fi
  if ! kill -0 "$(cat "$SERVER_PID")" 2>/dev/null; then
    echo "V4 viewer exited before becoming healthy." >&2
    tail -n 100 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$attempt" -lt 150 ]; then sleep 2; fi
done
if [ "$SERVER_READY" -ne 1 ]; then
  echo "V4 viewer did not become healthy within the bounded startup window." >&2
  tail -n 100 "$SERVER_LOG" >&2
  exit 1
fi

curl -fsS http://127.0.0.1:8877/api/health | "$PYTHON" -m json.tool
curl -fsS http://127.0.0.1:8877/api/v4/timeline | "$PYTHON" -m json.tool | head -n 80
export LAST_DAY=$(curl -fsS http://127.0.0.1:8877/api/v4/timeline | \
  "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["days"][-1]["calendar_date"])')
curl -fsS "http://127.0.0.1:8877/api/v4/frame/$LAST_DAY" | \
  "$PYTHON" -c 'import json,sys; p=json.load(sys.stdin); print(json.dumps({"day":p["calendar_date"],"clocks":p["clocks"],"meta":p["meta"],"overview":p["field_overview"]["meta"],"pressure_features":len(p["pressure"]["features"]),"story_features":len(p["story_footprints"]["features"])},indent=2))'
```

From the laptop:

```bash
ssh -N -L 8877:127.0.0.1:8877 opc@YOUR_VM_HOST
```

Open `http://127.0.0.1:8877`.

## 6. Latency and visual acceptance

Run this immediately after the fresh server starts, before manually scrubbing;
unknown query-string cache busters are deliberately ignored.

```bash
export SERVER_PID_VALUE=$(cat "$SERVER_PID")
"$PYTHON" server/benchmark_incident_v4.py \
  --base-url http://127.0.0.1:8877 \
  --days 28 \
  --random-requests 60 \
  --concurrency 8 \
  --server-pid "$SERVER_PID_VALUE" \
  --output "$ROOT/logs/incident_v4_benchmark_$SERVER_TAG.json"
```

Required server gates: every country response has a source day, is untruncated,
and accounts for every monitored field as `represented + unmappable = source`;
unmappable fields raise a warning but do not by themselves fail completeness. Also require no failed requests,
cold p95 under 1.5 s, cold concurrent p95 under 2.5 s, warm p95 under 250 ms,
cached random p95 under 300 ms, and cached concurrent p95 under 500 ms. Also
record a browser Performance trace over 28 adjacent days: response-to-render
p95 under 100 ms, no repeated tasks over 50 ms, and stable heap after two
back-and-forth passes.

Visual acceptance:

- the muted national grid is present without zooming; any unmappable fields are
  explicitly warned while the completeness gate confirms that every source
  field is either represented or accounted as unmappable;
- simultaneous daily hazards use separately colored pressure bands;
- a usable S2 acquisition advances crop evidence once, while echoes do not;
- rejected acquisitions are visible as rejected markers;
- crop name and stage are visible, and stage changes do not create a new story;
- a story appears only after its weekly checkpoint is known;
- the selected trajectory has separate hazard lanes, explicit missing versus
  observed-low pressure, S2 source-to-known clocks, freshness aging, a crop
  stage band, and one row per story;
- history truncation/capping is visible instead of looking complete;
- selected history is made of exact age-faded footprint polygons/outlines, with no
  arrow, centroid motion, interpolation, or propagation claim;
- terminal stories leave the overview after the documented 28-day review
  window, so frame payloads do not grow forever.

## 7. Operational append cadence

Run an immutable V4 release after every daily weather ingest and after every
irregular Sentinel-2 ingest. The weekly V3 checkpoint still advances only when
a full week is sealed; daily/S2 releases between checkpoints retain the latest
known weekly story state. Two releases on the same calendar day deliberately
share `release_as_of` but must have strictly increasing UTC `released_at`
watermarks.

For each release, first build the new immutable source generation and its V3
incident release. Then point V4 at the immediately prior **evidence** release.
Use the upstream ingest-completion timestamp when available; otherwise capture
one timestamp once and pass it to the durable runner:

```bash
export PREVIOUS_EVIDENCE_DIR=/absolute/path/to/prior/incident_evidence_v4_TAG
export GEN=/absolute/path/to/new/generation
export INCIDENT_DIR=/absolute/path/to/new/incidents_v3_TAG
export ECHO=/absolute/path/to/new/echo-aware-deliverable.parquet
export JOB_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export RELEASED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export JOB_DIR="$ROOT/jobs/incident_v4_$JOB_TAG"
export LAUNCH_LOG="$ROOT/logs/incident_v4_launcher_$JOB_TAG.log"

nohup "$PYTHON" server/run_incident_v4.py run \
  --root "$ROOT" \
  --generation-dir "$GEN" \
  --incident-dir "$INCIDENT_DIR" \
  --echo-deliverable "$ECHO" \
  --full-parquet "$FULL_2025" \
  --full-parquet "$FULL_2026" \
  --source-acquisition-parquet "$ACQ_2026" \
  --acquisition-parquet "$ACQ_2026" \
  --availability-mode reconstructed \
  --released-at "$RELEASED_AT" \
  --previous-evidence-dir "$PREVIOUS_EVIDENCE_DIR" \
  --python "$PYTHON" \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  --job-tag "$JOB_TAG" \
  >"$LAUNCH_LOG" 2>&1 </dev/null &
```

Append validation rejects any silent rewrite of published crop/day, hazard/day,
or acquisition facts, and rejects an equal or older `released_at` even when the
playhead date is unchanged. A late correction needs a future explicit revision
ID plus supersedes contract; that contract is not implemented. Until it is, a
correction fails closed and cannot mutate the historical release in place.

## 8. Learned motifs are a gated second layer

The map above already contains real stories: persistent crop-specific incidents
with daily pressure, sparse crop response, lifecycle, stage context, and exact
footprint evolution. HDBSCAN learns a similarity vocabulary over eligible
completed stories; it never decides where a story begins or ends and never
changes `incident_id`.

Do not describe the current single historical horizon as validated motifs. A
publishable learned layer requires three knowledge-time-separated cohorts,
exposure/lineage purging, immutable expert review of discovered training motifs,
separately reviewed calibration incidents, and sealed holdout replay. The V4
motif workflow intentionally emits `map_publication_supported=false` until a
later product approval gate is defined. Exact discovery, review, calibration,
and evaluation commands are printed by:

```bash
"$PYTHON" server/run_incident_motifs_v4.py --help
```

For a strictly diagnostic run over the current horizon, the explicit temporal
split is:

```bash
export RAPIDS_PYTHON=/mnt/KSA-Oasis/El-Mohammed/mango-comet/.venv-rapids-26.06-cu13-20260702T233116Z/bin/python
export TRAIN_THROUGH=2026-02-28
export CALIBRATION_THROUGH=2026-03-31
export EVALUATION_THROUGH=2026-05-17
export MOTIF_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export DISCOVERY_DIR="$ROOT/models/incident_motif_v4_discovery_$MOTIF_TAG"
export MOTIF_JSON="$ROOT/logs/incident_motif_v4_discovery_$MOTIF_TAG.json"
export MOTIF_LOG="$ROOT/logs/incident_motif_v4_discovery_$MOTIF_TAG.log"

test -x "$RAPIDS_PYTHON"
test -f "$INCIDENT_DIR/incident_membership.parquet"
test -f "$EVIDENCE_DIR/field_day_pressure_v4.parquet"
test -f "$EVIDENCE_DIR/field_s2_acquisition_v4.parquet"
test -f "$VIEWER_DIR/manifest.json"
test -f "$VIEWER_DIR/story_checkpoints_v4.parquet"

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$RAPIDS_PYTHON" server/run_incident_motifs_v4.py discover \
  --incident-dir "$INCIDENT_DIR" \
  --evidence-dir "$EVIDENCE_DIR" \
  --viewer-dir "$VIEWER_DIR" \
  --output-dir "$DISCOVERY_DIR" \
  --train-through "$TRAIN_THROUGH" \
  --calibration-through "$CALIBRATION_THROUGH" \
  --evaluation-through "$EVALUATION_THROUGH" \
  --engine gpu \
  --min-cluster-size 100 \
  --min-samples 20 \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  >"$MOTIF_JSON" 2>"$MOTIF_LOG" </dev/null &
echo "$!" | tee "$ROOT/logs/incident_motif_v4_discovery_$MOTIF_TAG.pid"
tail -f "$MOTIF_LOG"
```

The command is allowed to fail with “no supported motifs/strata” if this short,
single-season split is insufficient. Do not weaken support thresholds until it
passes. A successful output is still unreviewed and not map-approved.

Discovery validates and hash-binds `"$EVIDENCE_DIR/manifest.json"` and the V4
viewer release in `"$VIEWER_DIR"`; loose, mixed, or generation-mismatched
inputs are rejected. Daily evidence and causal prefixes stay in DuckDB/parquet.
Pandas receives only one crop-by-hazard discovery cohort at a time; prefix
fitting is bounded by crop, hazard, weather maturity, and usable S2 maturity,
and holdout replay is record-batched. Ensure `--temp-dir` has enough spill
space for the cumulative on-disk tables.

An agronomist must make an immutable reviewed copy of
`review_overlay_template.parquet`, and separately label calibration incidents.
Both files must retain the exact discovery `model_version`; do not auto-approve
the template. The calibration label parquet must contain `incident_id`,
`model_version`, `reviewed_motif_id`, `review_status`, `review_version`, and
`review_overlay_sha256`. It must exhaustively disposition every eligible
calibration incident: `approved` with a reviewed motif, or `novel_unassigned`
with a null motif. Partial/easy-case calibration is rejected. After those
artifacts exist:

```bash
export REVIEW_OVERLAY=/absolute/path/to/reviewed_overlay.parquet
export CALIBRATION_LABELS=/absolute/path/to/reviewed_calibration_labels.parquet
export PREFIX_DIR="$ROOT/models/incident_motif_v4_prefix_$MOTIF_TAG"

"$PYTHON" server/run_incident_motifs_v4.py fit-prefix \
  --discovery-dir "$DISCOVERY_DIR" \
  --review-overlay "$REVIEW_OVERLAY" \
  --reviewed-calibration-labels "$CALIBRATION_LABELS" \
  --output-dir "$PREFIX_DIR"
```

Finally, obtain a sealed holdout label file created without looking at prefix
predictions. It must contain `incident_id`, `final_assignment_status`,
`reviewed_motif_id`, `discovery_model_version`, `review_version`,
`review_overlay_sha256`, `prefix_model_version`, and
`prefix_manifest_sha256`; these bindings prevent labels from being silently
reused with another model or review. Then replay:

```bash
export HOLDOUT_LABELS=/absolute/path/to/sealed_holdout_labels.parquet
export EVALUATION_DIR="$ROOT/evaluations/incident_motif_v4_$MOTIF_TAG"

"$PYTHON" server/run_incident_motifs_v4.py evaluate \
  --discovery-dir "$DISCOVERY_DIR" \
  --prefix-model-dir "$PREFIX_DIR" \
  --final-labels "$HOLDOUT_LABELS" \
  --output-dir "$EVALUATION_DIR"

"$PYTHON" -m json.tool "$EVALUATION_DIR/evaluation.json"
```

`evaluate` exits with code `21` when any hard replay gate fails, while retaining
the immutable diagnostic report for inspection. Do not treat a failed-gate
directory as a successful model release.

After a reviewed prefix model passes the engineering gates, score one immutable
diagnostic delta for a requested as-of. The command excludes unconfirmed
`CANDIDATE` and terminal incidents, uses daily post-checkpoint weather only
while causal incident ownership remains valid, and never publishes motifs to
the map:

```bash
export AS_OF=2026-05-17
export LIVE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export LIVE_DIR="$ROOT/live_scores/score_${AS_OF}_${LIVE_TAG}"
export LIVE_JSON="$ROOT/logs/incident_motif_v4_live_${LIVE_TAG}.json"
export LIVE_LOG="$ROOT/logs/incident_motif_v4_live_${LIVE_TAG}.log"

nohup env PYTHONUNBUFFERED=1 \
  "$PYTHON" server/run_incident_motifs_v4.py score-live \
  --incident-dir "$INCIDENT_DIR" \
  --evidence-dir "$EVIDENCE_DIR" \
  --viewer-dir "$VIEWER_DIR" \
  --prefix-model-dir "$PREFIX_DIR" \
  --output-dir "$LIVE_DIR" \
  --as-of "$AS_OF" \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp" \
  --heartbeat-seconds 30 \
  >"$LIVE_JSON" 2>"$LIVE_LOG" </dev/null &
echo "$!" | tee "$ROOT/logs/incident_motif_v4_live_${LIVE_TAG}.pid"
tail -f "$LIVE_LOG"
```

Every run needs a fresh output directory. Inspect
`live_score_manifest.json`, `live_incident_ledger.parquet`, and
`live_prefix_assignments.parquet`; accepted rows remain tentative and
`map_publication_supported` must remain `false`. A release with no eligible
active confirmed incident fails explicitly instead of writing a misleading
empty delta.

Use one H100 for discovery; the strata are independent offline work, but eight
GPUs do not make the HTTP timeline or browser faster. CPU and GPU HDBSCAN are
separate implementations, so the selected engine is model provenance rather
than an assumed equivalent backend. Compare them on a fixed labelled fixture
before interpreting cross-engine motif differences.
