# Mango Comet

Echo-aware story monitoring, offline motif discovery, and a portable agronomic
activity viewer. The repository is intentionally code-only: field geometry,
Parquet runs, manifests, generated bundles, and credentials are not published.

## VM quick start

```bash
git clone https://github.com/noobjam/mango-comet.git
cd mango-comet

python3 --version  # Python 3.11 or newer
python3 -m venv .venv
source .venv/bin/activate

# Ignore machine-wide Oracle pip configuration and use only public PyPI.
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  -r server/requirements.txt

cp server/env.example server/.env
```

This command bypasses machine-wide pip configuration and permits only the
official `https://pypi.org/simple` index. If it still reports a proxy connection
failure, inspect `env | grep -i proxy`; the VM may also have stale `HTTP_PROXY`,
`HTTPS_PROXY`, or `ALL_PROXY` variables unrelated to pip's package index.

Edit `server/.env` and point `STORY_MAP_RUN_DIR` at the directory you intend to
serve. For the current dual-clock app this must be the completed
`incident_viewer_v4_*` directory emitted by `run_incident_v4.py`. Incident V3
similarly requires `incident_viewer_v3_*`; intermediate analytics/evidence
directories are not viewer bundles. Legacy raw runs remain supported for
bounded current-frame exploration.

```bash
python server/story_map_server.py
```

The safe default listens only on the VM loopback interface. In a second local
terminal, create an SSH tunnel and open `http://127.0.0.1:8877`:

```bash
ssh -N -L 8877:127.0.0.1:8877 VM_USER@VM_HOST
```

This prototype has no authentication. Bind it to `0.0.0.0` only after adding
explicit VM firewall/security-group restrictions or a protected reverse proxy.

## Build an optimized release

Build into a new versioned directory rather than overwriting the directory used
by a running process:

```bash
python server/build_story_map_bundle.py \
  --run-dir /srv/story-map-data/source-run \
  --out-dir /srv/story-map-data/releases/mango-comet-v1
```

Then set this in `server/.env` and restart the server:

```dotenv
STORY_MAP_RUN_DIR=/srv/story-map-data/releases/mango-comet-v1
```

The builder validates required schemas, geometry integrity, unique field IDs,
and geometry coverage before installing a release.

`weekly_story_monitor.py` builds immutable weekly monitoring generations from
the VM echo-aware parquet. It is currently a scheduled batch-release workflow,
not a parquet tailer or hot-reloading browser service; the exact VM commands and
that production boundary are documented in `server/README.md`. The current
full-generation checkpoint and the gated path from RAPIDS installation through
learned motifs, bundle promotion, map verification, and latency acceptance are
in [`server/VM_MAP_RELEASE_RUNBOOK.md`](server/VM_MAP_RELEASE_RUNBOOK.md).
The current Phase A archetype experiment should be launched with the durable
`server/run_archetype_v2.py` runner documented in
[`server/ARCHETYPE_V2.md`](server/ARCHETYPE_V2.md); it provides one command for
preflight, GPU build, evaluation, status, resume, and truthful heartbeats.

The product-level successor is the deterministic crop-impact incident V3
pipeline. Its story identity is local exposure × crop, with crop stage retained
as changing weekly context. Build and inspect it with
[`server/INCIDENT_V3_VM_RUNBOOK.md`](server/INCIDENT_V3_VM_RUNBOOK.md). The V3
pipeline emits a dedicated `incident_viewer_v3_*` directory that is the
map-ready `STORY_MAP_RUN_DIR`. Use the durable runner and promotion/latency
gates in the runbook; the intermediate `incidents_v3_*` analytics directory is
not a drop-in viewer bundle.

Incident V4 keeps that identity and corrects the monitoring clocks: weather is
daily, Sentinel-2 is acquisition-grain and irregular, and weekly story states
appear only after their knowledge time. It adds country-scale field coverage,
daily pressure bands, usable/rejected S2 markers, crop stage, causal story
drilldown, and a separate review-gated motif-learning workflow. Use
[`server/INCIDENT_V4_VM_RUNBOOK.md`](server/INCIDENT_V4_VM_RUNBOOK.md) for the
single durable VM sequence and
[`server/CROP_INCIDENT_STORIES_V4.md`](server/CROP_INCIDENT_STORIES_V4.md) for
the truth contract.

## Verify on the VM

```bash
python -m py_compile server/*.py
python -m unittest discover -s server -p 'test_*.py'
curl --fail http://127.0.0.1:8877/api/health
```

For interpretation, API endpoints, performance controls, optional GPU
precomputation, and the production vector-tile direction, see
[`server/README.md`](server/README.md).

For the current lifecycle semantics, as-of causality rule, exact-footprint
trajectory contract, validation gates, and safe presentation language, read
[`server/CROP_INCIDENT_STORIES_V3.md`](server/CROP_INCIDENT_STORIES_V3.md).
The V4 dual-clock successor is documented in
[`server/CROP_INCIDENT_STORIES_V4.md`](server/CROP_INCIDENT_STORIES_V4.md).
[`server/MONITORING_STORIES.md`](server/MONITORING_STORIES.md) remains the V1
field-episode provenance contract.

## Data policy

Keep run artifacts outside the repository. The `.gitignore` blocks common
geospatial, Parquet, bundle, environment, and cache paths as a second line of
defense, but inspect `git status` before every push.
