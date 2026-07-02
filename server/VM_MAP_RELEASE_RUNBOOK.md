# GPU VM to Learned-Motif Map Runbook

This is the operational sequence for turning the first full causal monitoring
generation into the data shown by the map. It is deliberately stricter than a
list of commands: every stage has an artifact, a gate, and a failure rule.

## Current checkpoint

The first full generation completed successfully on 2026-07-02:

```text
as_of_date:    2026-05-17
generation:    generation_7a715df05da10c3b3300
daily rows:    39,695,363
events:        3,131,245
generation_dir:
/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/generations/2026-05-17_generation_7a715df05da10c3b3300
```

This proves that the causal state-machine generation finished. It does **not**
yet prove that the event thresholds or learned motifs are agronomically valid.

The remaining sequence is:

```text
RAPIDS ready
  -> train frozen motifs on 2025 prefixes
  -> inspect motif/noise diagnostics
  -> pass the scalable-export code gate
  -> assign the frozen model to all weekly prefixes
  -> build the optimized map bundle
  -> launch and benchmark the VM server
  -> inspect the map and field trajectories
```

## Definition of done for the map

The desired map release must provide all of the following:

- one canonical field polygon per selected week, colored and ordered by its
  **current** lifecycle/risk rather than its eventual peak;
- learned `motif_id` values from a frozen model, plus an explicit
  `novel_unassigned` result for unfamiliar or ambiguous prefixes;
- causal field drill-down with concurrent hazards in separate lanes;
- a weather-map-style motif footprint trail showing activity center,
  dispersion, overlap, entering, persisting, and exiting fields;
- no claim that the aggregate trail is physical hazard movement;
- compact timeline state responses with geometry fetched once;
- measured subsequent-week p95 below 300 ms and at least 70% fewer compressed
  bytes than geometry-on-every-frame playback.

## 1. Restore the release variables

Run this at the start of every new VM shell:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet

export REPO=/mnt/KSA-Oasis/El-Mohammed/mango-comet
export PYTHON="$REPO/.venv/bin/python"
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export GEN="$ROOT/generations/2026-05-17_generation_7a715df05da10c3b3300"
export AS_OF=2026-05-17
export TRAINING_CUTOFF=2025-12-31

test -x "$PYTHON"
test -s "$GEN/manifest.json"
```

The 2025 cutoff leaves 2026 out of discovery so it can be used as a temporal
holdout. The partial week containing 2025-12-31 is excluded by the prefix
loader; the last admitted training bucket is therefore fully historical.

## 2. Finish and verify the RAPIDS installation

The server `.venv` intentionally contains only CPU/runtime dependencies. The
CUDA 13 RAPIDS environment is separate and its Python path is recorded here:

```bash
export RAPIDS_PYTHON=$(cat "$ROOT/logs/latest_rapids_python.txt")
test -x "$RAPIDS_PYTHON"
echo "$RAPIDS_PYTHON"
```

Find the active or latest installation job:

```bash
export INSTALL_PID_FILE=$(ls -1t "$ROOT"/logs/rapids_install_*.pid 2>/dev/null | head -n 1)
test -n "$INSTALL_PID_FILE"

export INSTALL_BASE=${INSTALL_PID_FILE%.pid}
export INSTALL_LOG="$INSTALL_BASE.log"
export INSTALL_STATUS="$INSTALL_BASE.status"

ps -fp "$(cat "$INSTALL_PID_FILE")" || true
tail -n 100 "$INSTALL_LOG"

if [ ! -f "$INSTALL_STATUS" ]; then
  echo "RAPIDS INSTALL STILL RUNNING"
elif [ "$(cat "$INSTALL_STATUS")" = "0" ]; then
  echo "RAPIDS INSTALL SUCCEEDED"
else
  echo "RAPIDS INSTALL FAILED: status $(cat "$INSTALL_STATUS")"
  tail -n 200 "$INSTALL_LOG"
fi
```

Do not reuse a partially installed environment after a nonzero status. Create a
fresh timestamped environment and rerun the CUDA 13 installation instead.

After status `0`, validate RAPIDS and the exact HDBSCAN path:

```bash
CUDA_VISIBLE_DEVICES=0 "$RAPIDS_PYTHON" -m cuml.health_checks -v

CUDA_VISIBLE_DEVICES=0 "$RAPIDS_PYTHON" -c 'import cupy as cp, cuml; from cuml.cluster.hdbscan import HDBSCAN; X=cp.asarray([[0,0],[0,.1],[.1,0],[.1,.1],[8,8],[8,8.1],[8.1,8],[8.1,8.1]],dtype=cp.float32); model=HDBSCAN(min_cluster_size=2,min_samples=1).fit(X); cp.cuda.Stream.null.synchronize(); print("GPU BACKEND READY"); print("cuML:",cuml.__version__); print("CuPy:",cp.__version__); print("labels:",cp.asnumpy(model.labels_))'
```

Stop here if either command fails. Do not use `--engine auto`: on this dataset
it could silently choose CPU HDBSCAN.

## 3. Train the frozen motif model

Training uses causal 2025 event prefixes and an explicit GPU backend. The
current implementation uses one H100; the other seven GPUs do not reduce this
job because hazard strata are fitted sequentially.

Preflight host resources:

```bash
free -h
df -h "$ROOT"
du -sh "$GEN"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

The age-bucket sampler can retain up to four prefixes per event. With this
generation the theoretical ceiling is 12,524,980 training rows. Prefix
construction is CPU/RAM/Disk work, so GPU utilization may remain at zero for a
while before HDBSCAN begins.

Launch a uniquely named training job:

```bash
export TRAIN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export MODEL="$ROOT/models/motif_v1_train_2025_$TRAIN_TAG"
export TRAIN_JSON="$ROOT/logs/motif_train_$TRAIN_TAG.json"
export TRAIN_LOG="$ROOT/logs/motif_train_$TRAIN_TAG.log"
export TRAIN_STATUS="$ROOT/logs/motif_train_$TRAIN_TAG.status"
export TRAIN_PID="$ROOT/logs/motif_train_$TRAIN_TAG.pid"

mkdir -p "$ROOT/models" "$ROOT/logs"
printf '%s\n' "$MODEL" > "$ROOT/logs/latest_motif_model_path.txt"

export REPO RAPIDS_PYTHON GEN TRAINING_CUTOFF MODEL
export TRAIN_JSON TRAIN_LOG TRAIN_STATUS

nohup bash -c '
  cd "$REPO"

  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$RAPIDS_PYTHON" server/weekly_story_monitor.py train-motifs \
      --generation-dir "$GEN" \
      --training-through "$TRAINING_CUTOFF" \
      --model-dir "$MODEL" \
      --engine gpu \
      --min-cluster-size 100 \
      --min-samples 20 \
      --radius-quantile 0.95 \
      --assignment-margin 0.05 \
      >"$TRAIN_JSON" 2>"$TRAIN_LOG"

  status=$?
  printf "%s\n" "$status" > "$TRAIN_STATUS"
  exit "$status"
' </dev/null >/dev/null 2>&1 &

echo "$!" | tee "$TRAIN_PID"
```

Monitor without assuming an empty log means a hang:

```bash
ps -fp "$(cat "$TRAIN_PID")" || true
tail -n 100 "$TRAIN_LOG"
free -h
nvidia-smi

if [ ! -f "$TRAIN_STATUS" ]; then
  echo "MOTIF TRAINING STILL RUNNING"
elif [ "$(cat "$TRAIN_STATUS")" = "0" ]; then
  echo "MOTIF TRAINING SUCCEEDED"
  "$PYTHON" -m json.tool "$TRAIN_JSON"
else
  echo "MOTIF TRAINING FAILED: status $(cat "$TRAIN_STATUS")"
  tail -n 200 "$TRAIN_LOG"
fi
```

On failure, use a fresh `TRAIN_TAG` and model directory. Never reuse a partial
model directory.

## 4. Apply the model-quality gate

Restore the successful model path:

```bash
export MODEL=$(cat "$ROOT/logs/latest_motif_model_path.txt")
test -s "$MODEL/training_manifest.json"
test -s "$MODEL/feature_schema.json"
test -s "$MODEL/prototypes.parquet"
test -s "$MODEL/motif_catalog.parquet"
test -s "$MODEL/training_assignments.parquet"
```

Validate hard engineering invariants and print the top-level diagnostics:

```bash
MODEL="$MODEL" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

root = Path(os.environ["MODEL"])
manifest = json.loads((root / "training_manifest.json").read_text())
schema = json.loads((root / "feature_schema.json").read_text())
catalog = pd.read_parquet(root / "motif_catalog.parquet")
prototypes = pd.read_parquet(root / "prototypes.parquet")
assignments = str(root / "training_assignments.parquet")

with duckdb.connect(":memory:") as con:
    rows, noise, bad_membership = con.execute("""
        SELECT
          COUNT(*),
          COUNT(*) FILTER (WHERE discovery_label < 0),
          COUNT(*) FILTER (
            WHERE NOT isfinite(training_membership)
               OR training_membership < 0
               OR training_membership > 1
          )
        FROM read_parquet(?)
    """, [assignments]).fetchone()

assert manifest["engine_requested"] == "gpu", manifest
assert manifest["engine_used"] == "gpu", manifest
assert manifest["row_count"] > 0, manifest
assert manifest["motif_count"] > 0, manifest
assert 0 <= manifest["noise_count"] < manifest["row_count"], manifest
assert manifest["model_version"] == schema["model_version"]
assert schema.get("policy_sha256")
assert rows == manifest["row_count"]
assert noise == manifest["noise_count"]
assert bad_membership == 0
assert len(catalog) == len(prototypes) == manifest["motif_count"]
assert catalog["motif_id"].is_unique and prototypes["motif_id"].is_unique
assert set(catalog["motif_id"]) == set(prototypes["motif_id"])
assert int(catalog["member_count"].sum()) == rows - noise
assert (catalog["member_count"] >= manifest["config"]["min_cluster_size"]).all()
assert (catalog["status"] == "discovered_unreviewed").all()

feature_columns = sorted(name for name in prototypes if name.startswith("f_"))
assert feature_columns
assert np.isfinite(prototypes[feature_columns].to_numpy(dtype=float)).all()
assert np.isfinite(prototypes["radius"].to_numpy(dtype=float)).all()
assert (prototypes["radius"] > 0).all()

summary = {
    "model_version": manifest["model_version"],
    "training_cutoff": manifest["training_cutoff"],
    "training_prefixes": manifest["row_count"],
    "motifs": manifest["motif_count"],
    "noise_prefixes": manifest["noise_count"],
    "noise_pct": round(100 * manifest["noise_count"] / manifest["row_count"], 2),
    "smallest_motif": int(catalog["member_count"].min()),
    "median_motif": float(catalog["member_count"].median()),
    "engine": manifest["engine_used"],
    "catalog_status": sorted(catalog["status"].unique().tolist()),
}
print(json.dumps(summary, indent=2))
PY
```

Inspect hazard coverage, noise, and motif-size imbalance:

```bash
MODEL="$MODEL" "$PYTHON" - <<'PY'
import os
from pathlib import Path
import duckdb

root = Path(os.environ["MODEL"])
assignments = str(root / "training_assignments.parquet")
catalog = str(root / "motif_catalog.parquet")

with duckdb.connect(":memory:") as con:
    print("\nPREFIX AND NOISE SUMMARY")
    print(con.execute("""
        SELECT
          hazard_family,
          COUNT(*) AS prefix_count,
          COUNT(*) FILTER (WHERE discovery_label < 0) AS noise_count,
          ROUND(100.0 * COUNT(*) FILTER (WHERE discovery_label < 0)
                / COUNT(*), 2) AS noise_pct,
          COUNT(DISTINCT discovery_label)
            FILTER (WHERE discovery_label >= 0) AS motif_count
        FROM read_parquet(?)
        GROUP BY hazard_family
        ORDER BY prefix_count DESC
    """, [assignments]).fetchdf().to_string(index=False))

    print("\nPUBLISHED MOTIF SIZE SUMMARY")
    print(con.execute("""
        SELECT
          hazard_family,
          COUNT(*) AS motif_count,
          MIN(member_count) AS smallest_motif,
          MEDIAN(member_count) AS median_motif,
          MAX(member_count) AS largest_motif
        FROM read_parquet(?)
        GROUP BY hazard_family
        ORDER BY hazard_family
    """, [catalog]).fetchdf().to_string(index=False))
PY
```

These summaries are diagnostics, not agronomic validation. Before publication:

- investigate a hazard with training prefixes but zero motifs;
- investigate extreme noise, a single overwhelmingly dominant motif, or an
  impractically large number of near-duplicate motifs;
- review motif labels and representative events with an agronomist;
- keep catalog status `discovered_unreviewed` until that review exists;
- do not tune thresholds solely to make the map look visually balanced.

## 5. Full-scale export is a mandatory code gate

**Do not run full `export-motifs` on commit `eb6fa0a`.** The current exporter:

- materializes every weekly prefix in one pandas DataFrame;
- assigns prototypes with a Python row loop;
- loads and rewrites full frame/membership tables in pandas;
- copies the entire generation before rewriting several large files.

At 3.1 million events this can require many tens of GB, excessive temporary
disk, and hours or days of avoidable CPU work. Eight H100s do not fix this
Python/pandas bottleneck.

The export gate is open only after the repository provides and tests:

1. DuckDB prefix materialization with an explicit spill directory;
2. bounded Arrow/Parquet batches;
3. vectorized prototype distances with stable tie behavior;
4. incremental assignment Parquet writing;
5. DuckDB joins/windows for frames, memberships, events, and labels;
6. global gap carry-forward across row-group boundaries;
7. atomic staging without duplicating unchanged large artifacts;
8. row-count, uniqueness, model-version, non-null motif, and scalar-versus-
   vectorized equivalence tests.

Until this gate is implemented, the technically correct next action is to
inspect the trained model, not to attempt a full release export.

## 6. Export and bundle after the scalable-export gate opens

This section is conditional. Run it only after pulling a reviewed commit that
implements the gate above and after its full tests pass:

```bash
cd "$REPO"
git pull --ff-only
PYTHONPATH=server "$PYTHON" -m unittest discover -s server -p 'test_*.py'
```

Create immutable release paths:

```bash
export MODEL=$(cat "$ROOT/logs/latest_motif_model_path.txt")
export RELEASE_TAG="$AS_OF-$(date -u +%Y%m%dT%H%M%SZ)"
export MOTIF_RUN="$ROOT/releases/$RELEASE_TAG-motifs"
export BUNDLE="$ROOT/releases/$RELEASE_TAG-bundle"

test ! -e "$MOTIF_RUN"
test ! -e "$BUNDLE"
```

Launch the assignment/export job:

```bash
export EXPORT_LOG="$ROOT/logs/export_$RELEASE_TAG.log"
export EXPORT_STATUS="$ROOT/logs/export_$RELEASE_TAG.status"
export EXPORT_PID="$ROOT/logs/export_$RELEASE_TAG.pid"

export REPO PYTHON GEN MODEL MOTIF_RUN EXPORT_LOG EXPORT_STATUS

nohup bash -c '
  cd "$REPO"
  "$PYTHON" server/weekly_story_monitor.py export-motifs \
    --generation-dir "$GEN" \
    --model-dir "$MODEL" \
    --output-dir "$MOTIF_RUN" \
    >"$EXPORT_LOG" 2>&1
  status=$?
  printf "%s\n" "$status" > "$EXPORT_STATUS"
  exit "$status"
' </dev/null >/dev/null 2>&1 &

echo "$!" | tee "$EXPORT_PID"
```

Require status `0` before bundling:

```bash
if [ ! -f "$EXPORT_STATUS" ]; then
  echo "EXPORT STILL RUNNING"
elif [ "$(cat "$EXPORT_STATUS")" = "0" ]; then
  echo "EXPORT SUCCEEDED"
else
  echo "EXPORT FAILED: status $(cat "$EXPORT_STATUS")"
  tail -n 200 "$EXPORT_LOG"
fi
```

Then build the optimized bundle:

```bash
export BUNDLE_LOG="$ROOT/logs/bundle_$RELEASE_TAG.log"
export BUNDLE_STATUS="$ROOT/logs/bundle_$RELEASE_TAG.status"
export BUNDLE_PID="$ROOT/logs/bundle_$RELEASE_TAG.pid"

export REPO PYTHON MOTIF_RUN BUNDLE BUNDLE_LOG BUNDLE_STATUS

nohup bash -c '
  cd "$REPO"
  "$PYTHON" server/build_story_map_bundle.py \
    --run-dir "$MOTIF_RUN" \
    --out-dir "$BUNDLE" \
    >"$BUNDLE_LOG" 2>&1
  status=$?
  printf "%s\n" "$status" > "$BUNDLE_STATUS"
  exit "$status"
' </dev/null >/dev/null 2>&1 &

echo "$!" | tee "$BUNDLE_PID"
```

Require status `0`, then verify the release artifacts:

```bash
cat "$BUNDLE_STATUS"

for name in \
  frame_fields.parquet \
  field_geometry.parquet \
  cluster_labels.parquet \
  event_windows.parquet \
  story_day_membership.parquet \
  event_state_snapshots.parquet \
  motif_assignments.parquet \
  motif_catalog.parquet \
  manifest.json \
  geometry_profile.json
do
  test -s "$BUNDLE/$name" || echo "MISSING: $name"
done

BUNDLE="$BUNDLE" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BUNDLE"])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["run"]["viewer_ready"] is True, manifest["run"]
assert manifest["run"]["geometry_optimized"] is True, manifest["run"]
assert manifest["semantics"]["story_cluster_id_alias"] == "motif_id", manifest["semantics"]
print(json.dumps({
    "run": manifest["run"],
    "motifs": manifest.get("motifs"),
    "outputs": manifest.get("outputs"),
}, indent=2))
PY

printf '%s\n' "$BUNDLE" > "$ROOT/latest_bundle_path.txt"
```

## 7. Point the server at the bundle

Create the runtime environment from the checked-in example and change only the
release path:

```bash
export BUNDLE=$(cat "$ROOT/latest_bundle_path.txt")

cp server/env.example server/.env
sed -i "s|^STORY_MAP_RUN_DIR=.*|STORY_MAP_RUN_DIR=$BUNDLE|" server/.env
grep '^STORY_MAP_' server/.env
```

Keep `STORY_MAP_HOST=127.0.0.1` for SSH-tunnel access. The prototype has no
authentication and must not be exposed directly on a public interface.

Check whether an old server already owns the port:

```bash
curl --silent --show-error http://127.0.0.1:8877/api/health || true
ss -ltnp 2>/dev/null | grep ':8877' || true
```

Stop or move the old process intentionally before starting the new release.
Then launch:

```bash
export SERVER_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export SERVER_LOG="$ROOT/logs/server_$SERVER_TAG.log"
export SERVER_PID="$ROOT/logs/server_$SERVER_TAG.pid"

nohup "$PYTHON" server/story_map_server.py \
  >"$SERVER_LOG" 2>&1 </dev/null &

echo "$!" | tee "$SERVER_PID"
sleep 2
tail -n 100 "$SERVER_LOG"
```

The startup log must name the exact value in `$BUNDLE` and report
`optimized_geometry=True`:

```bash
grep -F "run_dir=$BUNDLE" "$SERVER_LOG"
grep -F 'optimized_geometry=True' "$SERVER_LOG"
```

Verify the public API:

```bash
curl --fail --silent http://127.0.0.1:8877/api/health \
  | "$PYTHON" -m json.tool

curl --fail --silent http://127.0.0.1:8877/api/manifest \
  | "$PYTHON" -m json.tool

curl --fail --silent http://127.0.0.1:8877/api/timeline \
  | "$PYTHON" -m json.tool

curl --fail --silent http://127.0.0.1:8877/api/motifs \
  | "$PYTHON" -m json.tool
```

Require the learned-motif column rather than the legacy hazard fallback:

```bash
curl --fail --silent http://127.0.0.1:8877/api/motifs \
  | "$PYTHON" -c 'import json,sys; payload=json.load(sys.stdin); source=payload["taxonomy"]["source"]; print("taxonomy source:",source); assert source == "frame_fields.motif_family", payload["taxonomy"]'
```

From the local workstation, tunnel the VM loopback port:

```bash
ssh -N -L 8877:127.0.0.1:8877 VM_USER@VM_HOST
```

Open `http://127.0.0.1:8877` locally.

## 8. Measure timeline latency

Record the first compact-frame request separately. It computes and caches a
content hash of the optimized geometry artifact, so it can be materially slower
than subsequent dates and can be hidden by a 20-week p95:

```bash
export FIRST_BUCKET=$(curl --fail --silent http://127.0.0.1:8877/api/timeline \
  | "$PYTHON" -c 'import json,sys; rows=json.load(sys.stdin)["buckets"]; print(rows[-1]["timeline_bucket"])')

curl --fail --silent --output /dev/null \
  --write-out 'cold_frame_state_seconds=%{time_total}\n' \
  "http://127.0.0.1:8877/api/frame-state/$FIRST_BUCKET?limit=5000"
```

Run the warm transport benchmark on the VM:

```bash
export BUNDLE=$(cat "$ROOT/latest_bundle_path.txt")

"$PYTHON" server/benchmark_timeline.py \
  --base-url http://127.0.0.1:8877 \
  --weeks 20 \
  --limit 5000 \
  --output "$ROOT/logs/timeline_benchmark_$(basename "$BUNDLE").json"
```

Required automated gates in its JSON output:

```text
subsequent_request_p95_below_300ms = true
compressed_bytes_reduced_70pct     = true
```

This measures server/network payload behavior. Also scrub at least 20 weeks in
the browser and check that polygon updates, selection, inspector rendering, and
aggregate-trail updates remain responsive. Record browser behavior separately;
do not present the fixture's 76% reduction as a VM measurement.

## 9. Visual acceptance checklist

On the final map, verify all of the following:

1. Changing week updates dynamic field state without refetching all geometry.
2. A field appears once per week even when it has concurrent hazard events.
3. Current risk/lifecycle controls color and urgency; peak risk remains audit
   context only.
4. Selecting a learned motif shows its label and model version.
5. `novel_unassigned` remains visible rather than being forced into a motif.
6. Field inspection shows causal weekly states and separate hazard/event lanes.
7. Missing evidence appears as `DATA_GAP`/unknown, not healthy/none.
8. The footprint trail breaks across weak-overlap or nonconsecutive weeks.
9. The UI describes the trail as aggregate footprint change, never movement or
   propagation.
10. No label claims crop death; unresolved severe response is marked for
    review.

## 10. Weekly operating model

This release is scheduled batch monitoring, not a live parquet tailer:

1. ingest the next weekly delivery;
2. build a new immutable generation;
3. compare/review late corrections;
4. assign with the same frozen motif model;
5. build and validate a new bundle;
6. change the release pointer and restart/refresh the server;
7. retrain motifs only on a governed schedule, producing a new model version.

Do not overwrite the bundle read by a live process. Promote a new immutable
bundle directory only after every gate above passes.

For the scientific contract and safe presentation language, read
[`MONITORING_STORIES.md`](MONITORING_STORIES.md).
