# Story Activity Map Server

Portable, no-Docker server for reviewing precomputed crop-risk stories on a
map. MapLibre/deck.gl render in the browser; the Python server reads Parquet
artifacts with DuckDB.

## Crop-impact Incident V4 authority

[`CROP_INCIDENT_STORIES_V4.md`](CROP_INCIDENT_STORIES_V4.md) is the current
monitoring contract. V4 retains the deterministic V3 identity hierarchy while
separating daily weather pressure, irregular acquisition-grain crop evidence,
and knowledge-gated weekly story checkpoints. If an old V3 spine fails V4
reconciliation, `run_incident_story_replay_v4.py` recomputes new identities
from the V4 ledgers and publishes the old IDs only in an audit crosswalk. The
map uses a daily playhead, shows crop stage, preserves simultaneous hazard
lanes, reports rejected S2 attempts, and never treats echoed spectral values as
new observations.

Use [`INCIDENT_V4_VM_RUNBOOK.md`](INCIDENT_V4_VM_RUNBOOK.md) for the exact
durable source-preparation/build/serve/benchmark sequence. Point
`STORY_MAP_RUN_DIR` only at its completed `incident_viewer_v4_*` output.
Learned motifs are a separate immutable diagnostic workflow: they cannot change
story identity and cannot appear as approved labels without expert review,
calibration, and sealed holdout replay.

The repository also contains an isolated deterministic field-story foundation
in [`FIELD_STORIES_V1.md`](FIELD_STORIES_V1.md). It composes concurrent and
sequential V4 hazard lanes into one fixed-location field/crop concern interval,
without clustering or machine learning. It is executable and canonically
tested, and `run_field_stories_v1.py` can publish a standalone immutable Parquet
release from completed V4 evidence. It is not yet wired into the viewer; V4
remains the promoted operational contract until that integration is designed
and validated.

## Crop-impact Incident V3 foundation

[`CROP_INCIDENT_STORIES_V3.md`](CROP_INCIDENT_STORIES_V3.md) is the current
definition of a product **story**. A weekly local component receives a
`component_id`; components linked through time retain a crop-independent
`exposure_id`; each crop affected by that exposure receives a stable
`incident_id`. Crop stage changes do not rewrite identity. Learned archetypes
are optional tags trained only on completed crop-impact stories.

Use [`INCIDENT_V3_VM_RUNBOOK.md`](INCIDENT_V3_VM_RUNBOOK.md) for the exact
tested/nohup build sequence. `run_incident_v3.py` produces both an immutable
`incidents_v3_*` analytics release and a verified `incident_viewer_v3_*`
bundle. Point `STORY_MAP_RUN_DIR` only at the latter. The UI renders complete
weekly incident footprints, crop/stage lifecycle arcs, and selected-only exact
footprint history without centroid trajectories.

## Archetype V2 Phase A provenance

[`ARCHETYPE_V2.md`](ARCHETYPE_V2.md) records the failed diagnostic predecessor
and its VM workflow. Its preferred
`run_archetype_v2.py` command safely runs tests, GPU build, evaluation, logging,
status, and resume. Phase A builds and evaluates one fixed causal anchor per
eligible event; it does not publish archetypes to this viewer. When all hard
gates pass but quality gates fail—as the real VM run did—the V2 runbook documents a separate,
unpublishable diagnostic preview for inspecting the result on port 8878.

The completed V1 experiment's **10,901 HDBSCAN prefix groups are diagnostic and
not publishable**. They came from multiple age-dependent prefix rows per event,
remain `discovered_unreviewed`, and must not be described as 10,901 story types
or exported as the map taxonomy. The V1 commands below remain for provenance
and bounded compatibility tests, not as the V2 release path.

## Legacy V1/V2 interpretation

This section describes compatibility artifacts from the older event/fingerprint
and motif paths. It does not override Incident V3: V3 uses stable
`exposure_id × crop` story identity and keeps crop stage as changing context.

Three related identifiers may be present:

- An exact story cluster is a deterministic audit fingerprint of an event's
  encoded sequence. It is useful for traceability, but is usually too granular
  for a product legend.
- Older motif runs contain a coarser deterministic taxonomy. New monitor runs
  can instead contain HDBSCAN-discovered causal prefix motifs, frozen into a
  versioned prototype/radius model with an explicit `novel_unassigned` result.
- An event ID identifies one field episode. It is not a cluster ID.

Crop is retained as event metadata rather than forced into the cross-crop key.
That design permits cross-crop motifs; it does not by itself prove that a motif
generalizes across crops.

Important: legacy runs summarize complete observed events and remain
retrospective. Generations built by `weekly_story_monitor.py` instead publish
causal weekly event prefixes. The viewer exposes the active generation's mode;
it must never project a complete-event label backward. Weekly footprints and
aggregate centers do not imply physical movement or propagation.

### Council concept audit

The bundled 50-field development sample supports the audit-fingerprint use case,
but not yet a claim of learned or validated story clustering:

- 1,312 event windows produce 761 exact fingerprints; 596 fingerprints (78.3%)
  are singletons, representing 45.4% of all events.
- Only 4 fingerprints (0.5%) contain more than one crop in this sample, so the
  cross-crop generalization claim remains largely untested here.
- 197 events (15.0%) contain more than the configured 12 sequence tokens. Their
  fingerprint uses the retained prefix, so later episodes do not affect identity.
- A complete-event fingerprint is future-aware when displayed on an earlier
  week. It is suitable for retrospective review, not live causal detection.

Treat exact fingerprints as stable audit IDs. Before calling broader groups
"similar stories," validate a separate motif layer for stability across seasons
and crops, agronomist agreement, and useful outcome separation. A live product
would also need prefix-safe identities that use only evidence available by the
selected date.

The new `weekly_story_monitor.py` path provides those causal prefixes for
ordinary append-only updates. Its starter thresholds and discovered motifs are
still uncalibrated and require agronomist/outcome validation. Read
[`MONITORING_STORIES.md`](MONITORING_STORIES.md) before presenting the method.
For the V2 event anchor, exact eligibility/status contract, immutable assignment,
gates, Phase A artifacts, and VM commands, read
[`ARCHETYPE_V2.md`](ARCHETYPE_V2.md).
For the current V3 crop-impact story contract and full VM build, read
[`CROP_INCIDENT_STORIES_V3.md`](CROP_INCIDENT_STORIES_V3.md) and
[`INCIDENT_V3_VM_RUNBOOK.md`](INCIDENT_V3_VM_RUNBOOK.md).
For the daily-weather/irregular-S2 successor, read
[`CROP_INCIDENT_STORIES_V4.md`](CROP_INCIDENT_STORIES_V4.md) and run
[`INCIDENT_V4_VM_RUNBOOK.md`](INCIDENT_V4_VM_RUNBOOK.md).
For the exact first full-release sequence, durable `nohup` commands, quality
gates, current full-scale export blocker, bundle promotion, and map acceptance
checks, use [`VM_MAP_RELEASE_RUNBOOK.md`](VM_MAP_RELEASE_RUNBOOK.md).

## Build monitoring data on the VM

Use the echo-aware deliverable and the matching geometry:

```bash
DATA=/mnt/KSA-Oasis/fields_health_v2/rwanda_crop_risk_kb/final_field_daily_v4/rwanda_2025_2026_field_daily_risk_DELIVERABLE_WITH_CROP_AND_RISK_DRIVER_v4_WITH_SPECTRAL_ECHO_DAYS.parquet
GEOM=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/full_event_adaptive_28d/map_field_geometry.parquet
ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
AS_OF=2026-05-17
```

Run a bounded acceptance test first:

```bash
python server/weekly_story_monitor.py update \
  --input-parquet "$DATA" \
  --geometry-parquet "$GEOM" \
  --output-dir "$ROOT/smoke" \
  --as-of "$AS_OF" \
  --max-fields 500 \
  --threads 8
```

For all fields, DuckDB scans the source once into deterministic field-hash
partitions. Workers load bounded partitions rather than the complete 39.7M rows
into one DataFrame:

```bash
mkdir -p "$ROOT/duckdb_tmp"

UPDATE_RESULT=$(python server/weekly_story_monitor.py update \
  --input-parquet "$DATA" \
  --geometry-parquet "$GEOM" \
  --output-dir "$ROOT" \
  --as-of "$AS_OF" \
  --partitions 128 \
  --workers 8 \
  --threads 32 \
  --memory-limit 96GB \
  --temp-dir "$ROOT/duckdb_tmp")

printf '%s\n' "$UPDATE_RESULT"
GEN=$(printf '%s' "$UPDATE_RESULT" | python -c 'import json,sys; print(json.load(sys.stdin)["generation"]["generation_dir"])')
test -n "$GEN"
test -d "$GEN"
echo "$GEN"
```

That generation already contains causal weekly snapshots. Do not rescan the
full source once per historical week; `replay` is intended for bounded
acceptance fixtures.

This command is an immutable batch update, not a continuously mutating stream.
For each new weekly delivery, build a new generation, run validation, export
with the same frozen motif model, build a new bundle, then promote that release
and restart the service. The browser does not poll for a new generation. The
current full update also rescans and repartitions retained history; a persistent
incremental event registry and automatic late-correction lineage remain future
production work.

The following command records the historical V1 prefix experiment. It is not
the Archetype V2 command and must not be used to produce a publishable map.
V1 used 2025 prefixes, leaving 2026 available for diagnostics. One H100 is
enough for its hazard-stratified discovery:

```bash
MODEL="$ROOT/models/motif_v1_train_2025"

CUDA_VISIBLE_DEVICES=0 python server/weekly_story_monitor.py train-motifs \
  --generation-dir "$GEN" \
  --training-through 2025-12-31 \
  --model-dir "$MODEL" \
  --engine gpu \
  --min-cluster-size 100 \
  --min-samples 20 \
  --radius-quantile 0.95 \
  --assignment-margin 0.05
```

The cutoff is applied to source observation dates, and only fully completed
Monday-Sunday buckets are admitted. Therefore `2025-12-31` safely ends on the
week of `2025-12-22`; the partial `2025-12-29` week is excluded rather than
borrowing January 2026 evidence. Concurrent hazards use event-specific daily
pressure/response columns, so one field's heat event cannot inherit its drought
event's pressure signal.

Use `--engine cpu` only for bounded fixtures or a separately measured
reproducibility run. Do not use `--engine auto` for the 3.1-million-event
generation: a missing RAPIDS installation could silently select CPU HDBSCAN.
In either case, weekly assignment uses the frozen model; it does not recluster
as new weeks arrive.

The following export command describes the intended artifact sequence, but it
is **not currently safe for the first full generation**. The exporter on commit
`eb6fa0a` materializes all weekly prefixes and performs row-wise pandas
assignment before copying and rewriting the generation. At 3.1 million events,
run the scalable-export gate in
[`VM_MAP_RELEASE_RUNBOOK.md`](VM_MAP_RELEASE_RUNBOOK.md) before using it.

After that gate is implemented and verified, export motif assignments and build
the geometry-optimized bundle used by the app:

```bash
MOTIF_RUN="$ROOT/releases/${AS_OF}_motifs"
BUNDLE="$ROOT/releases/${AS_OF}_bundle"

python server/weekly_story_monitor.py export-motifs \
  --generation-dir "$GEN" \
  --model-dir "$MODEL" \
  --output-dir "$MOTIF_RUN"

python server/build_story_map_bundle.py \
  --run-dir "$MOTIF_RUN" \
  --out-dir "$BUNDLE"
```

Use this `.env` block for that release:

```dotenv
STORY_MAP_RUN_DIR=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/releases/2026-05-17_bundle
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
```

After starting the server, measure the real VM rather than guessing:

```bash
python server/benchmark_timeline.py \
  --base-url http://127.0.0.1:8877 \
  --weeks 20 \
  --output "$ROOT/timeline_benchmark.json"
```

The acceptance gate is subsequent-week p95 below 300 ms and at least 70% fewer
compressed bytes than geometry-every-frame playback.

The development fixture measured 76% fewer compressed timeline bytes after
splitting static geometry from dynamic state. That is evidence for the transport
design, not a VM acceptance result; record the VM benchmark before presenting a
latency number.

For the crop-incident V3 app, use the footprint-specific benchmark instead:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
python server/benchmark_incident_v3.py \
  --base-url http://127.0.0.1:8877 \
  --weeks 20 \
  --random-requests 60 \
  --concurrency 8 \
  --output "$ROOT/incident_v3_benchmark.json"
```

It records sequential cold, concurrent cold-adjacent, warm, random-scrub, and
prewarmed concurrent-adjacent p50/p95; HTTP/503 counts; wire and decoded JSON
bytes; and, when passed the optional numeric `--server-pid`, server RSS. The
authoritative V3 runbook captures that PID during launch. Browser
parse/render/heap acceptance remains a required Performance trace because H100s
cannot accelerate the browser.

## Artifacts

The legacy exact-story path reads:

- `map_field_geometry.parquet` or optimized `field_geometry.parquet`
- `map_frame_fields.parquet` or optimized `frame_fields.parquet`
- `event_story_cluster_labels.parquet` or optimized `cluster_labels.parquet`
- `event_windows.parquet`
- `story_day_membership.parquet`
- `manifest.json`

A newer motif run can additionally contain:

- `daily_causal_signals.parquet`
- `crop_instances.parquet`
- `event_state_snapshots.parquet` (generation-local records; no automatic
  cross-generation supersession lineage yet)
- `motif_assignments.parquet` and `motif_catalog.parquet`
- `event_motif_membership.parquet`
- `field_motif_timeline.parquet`
- `story_motifs.parquet`
- `motif_prototypes.parquet`
- `motif_labels.parquet`
- `motif_timeline.parquet`
- `llm_narration_queue.parquet` and `llm_narration_queue.jsonl`

The bundle builder copies those motif artifacts only when the source run
contains them. The current map consumes `motif_family` when it is embedded in
`map_frame_fields.parquet`; the standalone motif tables are preserved for
offline narration and future drill-down, not yet exposed as a direct motif-ID
selector. Older exact-story runs remain valid and use a labeled hazard-family
proxy rather than fabricated motif membership.

## Local Run

From the repository root:

```bash
python -m venv .venv-map
source .venv-map/bin/activate
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  -r server/requirements.txt
cp server/env.example server/.env
python server/story_map_server.py
```

This public repository contains no run data. Before starting, edit
`server/.env` so `STORY_MAP_RUN_DIR` points to a raw run or optimized bundle on
the machine.

Open `http://127.0.0.1:8877`.

## Build A Portable Bundle

The optimized bundle converts source geometry to compact GeoJSON strings and
adds bounding boxes and centroids for viewport filtering. Builds are staged and
validated before owned output files are replaced, so a failed overwrite leaves
the previous bundle intact:

```bash
python server/build_story_map_bundle.py \
  --run-dir /path/to/event_story_run \
  --out-dir server/story_map_bundle \
  --overwrite
```

The builder validates required Parquet schemas, manifest JSON, geometry
integrity, unique field IDs, and frame-to-geometry joins. By default at least
95% of source geometries must parse and at least 95% of distinct frame fields
must join to geometry. The explicit `--min-valid-geometry-coverage` and
`--min-frame-geometry-coverage` overrides exist for audited exceptions; lowering
them should be treated as a data-quality decision, not a performance setting.

Run against it with:

```bash
STORY_MAP_RUN_DIR=server/story_map_bundle python server/story_map_server.py
```

For a VM deployment, copy `server/` and the chosen bundle or full run directory,
then set `STORY_MAP_RUN_DIR` to that location.

For a running VM, build into a new versioned release directory, start or probe a
server against that completed directory, then update `STORY_MAP_RUN_DIR` and
restart the service. Do not use `--overwrite` on the directory a live process is
reading: replacement of several Parquet files cannot be crash-atomic for that
reader. The builder serializes competing builds and rolls back ordinary install
errors, but a generation-directory switch is the safe deployment boundary.

## API And Bounded GeoJSON

Useful endpoints include:

- `GET /api/timeline` for available reporting buckets.
- `GET /api/incident-footprints/<bucket>?crop_name=...&hazard_family=...` for
  every complete exact Incident V3 footprint in a week. This country-overview
  endpoint is never subject to the field feature cap.
- `GET /api/incident/<incident_id>` for one complete crop story: causal weekly
  lifecycle, crop-stage denominators, lineage, and exact main/role footprint
  geometry through time.
- `GET /api/motifs` for mappable exact-story labels and situation facets,
  including the live `current_risk_band` facet (peak risk remains audit context).
- `GET /api/frame/<bucket>?bbox=minLon,minLat,maxLon,maxLat&limit=N` for field
  GeoJSON in the current viewport.
- `GET /api/frame-state/<bucket>?bbox=...` for one canonical, highest-urgency
  state per field without geometry, coordinates, or static administration
  fields. `concurrent_event_count` flags additional same-week event lanes.
- `POST /api/geometry` for at most 2,000 missing field geometries, pinned to an
  immutable geometry version and cached across dates by the browser.
- `GET /api/activity?...filters` for non-spatial retrospective counts per
  active bucket. It contains no representative point or traveled path.
- `GET /api/trail?bucket=YYYY-MM-DD&lookback=5&limit=N&...filters` for bounded
  historical field footprints behind the selected bucket. History is enabled
  after choosing an exact story or shared-evidence filter. It requires the
  optimized bundle geometry; the raw-run UI disables history while keeping
  current-frame filtering available.
- `GET /api/evolution?...filters` for a selected motif/signature's spherical
  activity center, p50/p90 dispersion, field-set overlap, and explicit segment
  breaks across the full mapped extent. It is serialized and loaded outside the
  frame critical path. It is an aggregate footprint summary, not physical
  movement.
- `GET /api/field/<field_id>/trajectory` for causal weekly event-prefix states
  when monitoring snapshots are present. Concurrent hazards remain separate
  event/hazard lanes rather than being drawn as one sequential story.

Use `bbox` and a finite `limit` for interactive requests. Unbounded GeoJSON is
acceptable for small development samples only: parsing every polygon on every
pan does not scale. The optimized `field_geometry.parquet` lets DuckDB apply
bounding-box filtering before geometry serialization.

Exact fingerprint filters can be useful for audit drill-down. For broader
activity review, prefer motif family, motif, risk, hazard, or response facets
when those columns exist.

## Deployment And Caching

Copy `server/env.example` to `server/.env`, set the VM paths, and keep finite
feature bounds in production.

The API exposes field geometry and event evidence and has no authentication.
Keep the default `127.0.0.1` bind for SSH-tunnel testing. Use `0.0.0.0` only
behind explicit firewall/security-group controls or an authenticated reverse
proxy. Health and browser manifest responses omit host filesystem paths, and
API responses are marked `private` to prevent storage by shared proxy caches.

The example uses a 5,000-feature default, a 20,000-feature hard cap, a
five-minute/256-entry in-process cache, a 512 MiB combined raw+gzip process
cache byte budget, and gzip for responses of at least 1 KiB. The process splits
that byte budget 15/16 to API responses and 1/16 to static assets, so the two
caches cannot each consume the full configured amount.
Tune `STORY_MAP_DEFAULT_FEATURE_LIMIT`, `STORY_MAP_MAX_FEATURE_LIMIT`,
`STORY_MAP_CACHE_SECONDS`, `STORY_MAP_CACHE_ENTRIES`, and
`STORY_MAP_CACHE_MAX_BYTES`, and `STORY_MAP_GZIP_MIN_BYTES` from measured
payloads and memory use. Oversized responses bypass the process cache. The
portable server uses `STORY_MAP_QUERY_CONCURRENCY` as the independent cap for
both simultaneous DuckDB/GeoJSON queries and simultaneous JSON cleaning,
encoding, and gzip work, so a rapid scrub burst cannot merely move an unbounded
queue from querying into response construction. Cache hits and `/api/health`
bypass those work gates. Excess uncached work fails fast with HTTP 503 and
`Retry-After: 1` instead of joining a stale request queue; tune the cap against
CPU cores, RSS, 503 counts, and measured sequential and concurrent-cold p95
latency.

Put a front proxy such as Nginx or Caddy in front of the Python process to add
TLS, gzip or Brotli compression, and access logs. Cache static vendor assets for
a long duration, but honor the API's `private` cache headers. Add shared API
caching only under an explicit single-audience protected policy with any access
identity included in the cache key. Never cache error responses as successful
data.

The in-process cache is a small latency aid, not a shared production cache.
Multiple processes or VMs will have independent cache hit rates; do not add a
shared cache until its access-control and invalidation policy is defined.

The GeoJSON API remains a bounded prototype serving path. For full production
data, materialize stable field geometry as PMTiles or MVT and serve filtered
dynamic event layers from vector tiles. Martin/PostGIS is one option; raster
layers can use TiTiler with COG/STAC inputs.

## The 8xH100 VM

H100s can help offline work: full-dataset clustering, motif materialization,
large joins and aggregates with RAPIDS, and preparation of tile or summary
artifacts. They do not make per-request GeoJSON serialization, HTTP transfer, or
browser rendering faster, so the request server should remain CPU-safe and read
precomputed outputs.

Measure a representative full run first. DuckDB on CPU is often sufficient and
Dask coordination can cost more than it saves. Move a measured bottleneck to
cuDF or Dask-cuDF only when the data size and partitioning justify it. The
existing optional summary command is:

```bash
python server/gpu_precompute_story_map.py \
  --run-dir /path/to/full_story_run \
  --out-dir /path/to/full_story_run/gpu_summaries \
  --engine auto
```

On the 8xH100 host, `--engine dask-cudf` is available after the optional RAPIDS
dependencies are installed, but it should follow measurement rather than be the
default assumption. Regardless of compute engine, production map delivery
should prefer prebuilt PMTiles/MVT over large per-request GeoJSON payloads.

## Offline Basemap Note

MapLibre and deck.gl are vendored under `server/static/vendor`; no Node build is
required. The default satellite source still needs network access. Set
`STORY_MAP_RASTER_TILES` to a local or self-hosted source for an offline VM.
