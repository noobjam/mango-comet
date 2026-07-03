# Event Archetype V2: Phase A Contract and VM Runbook

This document is the authoritative contract for Event Archetype V2 Phase A.
It supersedes the V1 prefix-motif workflow for any claim about learned story
types or any future map release. `MONITORING_STORIES.md` remains authoritative
for the event-state machine and evidence semantics.

## Status and decision

Phase A is an offline, diagnostic build-and-evaluate phase. It produces one
causal anchor row and at most one immutable archetype assignment per eligible
event. It does **not** modify a monitoring generation, create a viewer bundle,
or authorize publishing learned archetypes in the app.

The completed V1 experiment discovered **10,901 HDBSCAN groups from sampled
weekly event prefixes**. That result is useful as a fragmentation and scale
diagnostic only. It is not 10,901 validated stories, is not a product taxonomy,
and must not be exported to the public map. V2 deliberately changes the unit of
learning from multiple age-dependent prefixes to one fixed causal anchor per
event.

The implemented `build-archetypes-v2` and `evaluate-archetypes-v2` commands
below create immutable model and evaluation directories. Their outputs remain
diagnostic and unreviewed even when every engineering and statistical gate
passes.

## 1. Objects and hierarchy

V2 keeps four objects separate:

| Level | Object | Meaning | Identity rule |
|---|---|---|---|
| 1 | Hazard family | Coarse mechanism used to prevent unrelated processes from clustering together | Existing normalized event hazard |
| 2 | Event archetype | Learned pattern at one fixed causal event anchor | Stable `archetype_id` within one immutable model version |
| 3 | Event | One monitored field episode | Existing `event_id`; never replaced by an archetype ID |
| 4 | Weekly state | What was known about that event in week `t` | Existing causal state snapshot |

Hazard family is the eventual map-color and legend level. Archetype is a
curated drill-down level. Event state and current risk communicate phase and
urgency. The map must never allocate a separate color to every archetype.

## 2. Required inputs and lineage

Phase A reads one completed immutable generation containing at least:

- `manifest.json`;
- `daily_causal_signals.parquet`;
- `event_windows.parquet`;
- `story_day_membership.parquet`.

The model manifest records the generation ID and as-of date, source-generation
manifest SHA-256, generation policy version and SHA-256, feature schema hash,
clustering configuration, training cutoff, implementation SHA-256, and hashes
of the emitted model artifacts. Source-input lineage remains in the immutable
generation manifest. A model is compatible only with generations using the
same causal policy and feature schema.

The recorded implementation SHA-256 covers the V2 anchor,
discovery/assignment, subsample-stability, and workflow modules; the CLI; and
the CPU and optional-GPU requirement files. Model and evaluation manifests also
record exact Python, DuckDB, NumPy, pandas, PyArrow, scikit-learn, CuPy, and
cuML versions. Evaluation refuses implementation-hash or software-version
drift from the frozen model.

For the current VM run:

```text
generation as-of:  2026-05-17
training cutoff:   2025-12-31
holdout:           anchor dates after 2025-12-31
```

The split is by whole event using `anchor_date`. No event, field-event prefix,
or later row from a training event may appear in holdout.

## 3. Causal eligibility and anchor

### 3.1 Usable pressure rows

For each `event_id`, order `story_day_membership` rows by
`observation_date`. A usable pressure row is a row where:

```text
pressure_observed = true
```

Its `daily_pressure_rank` must be finite and within the policy's valid rank
domain. Duplicate `(event_id, observation_date)` usable rows are a hard error;
they are not silently deduplicated.

### 3.2 Anchor rule

An event is eligible only if it reached `ACTIVE` or `SEVERE` on or before its
candidate anchor and has at least seven usable pressure days.

The anchor is deterministic:

1. If the event has at least 21 usable pressure rows, `anchor_date` is the date
   of the 21st usable pressure row and `usable_days = 21`.
2. Otherwise, the event is eligible only when it is closed for a reason other
   than `CLOSED_SEASON_BOUNDARY`, has at least seven usable pressure rows
   through closure, and has a non-null `event_end_date`. In that case,
   `anchor_date = event_end_date` and `usable_days` is the count of usable
   pressure rows through closure.
3. Open/right-censored events with fewer than 21 usable days, watch-only events,
   season-boundary closures before maturity, and events with fewer than seven
   usable pressure days are excluded from Phase A discovery and evaluation.

This is equivalently the earlier of the 21st usable pressure day and an
eligible natural closure, subject to the seven-day minimum. An event that only
reaches `ACTIVE` or `SEVERE` after that date is ineligible; later state cannot
retroactively qualify an earlier anchor.

Every feature source observation date and every spectral source date must be
less than or equal to `anchor_date`. A post-anchor value anywhere in the
feature lineage is a hard failure.

### 3.3 One anchor ledger and its four statuses

`event_anchors.parquet` contains exactly one ledger row for every event in
`event_windows.parquet`. `anchor_outcome` and `anchor_status` carry one of four
exact values, evaluated in this order:

1. `season_boundary_before_maturity`: the event closed at a season boundary
   before a 21st usable pressure day;
2. `watch_only`: no `ACTIVE` or `SEVERE` state occurred in the candidate anchor
   window;
3. `insufficient_evidence`: fewer than seven usable days, or an open event has
   not yet reached its 21st usable day;
4. `eligible`: the remaining events that reached `ACTIVE` or `SEVERE` and have
   either a 21st usable day or an eligible natural closure with at least seven
   usable days.

Only `eligible` rows have model features and `eligible_for_training=true`;
other rows retain null feature values plus audit diagnostics. Duplicate event
IDs, duplicate usable `(event_id, observation_date)` rows, invalid pressure
ranks, and missing required artifacts or columns are hard input failures. They
are not additional ledger statuses.

## 4. Exact V2 feature contract

V2 has 20 model inputs. It has no categorical model inputs. Risk features use
only usable pressure rows through the anchor. Response features use event
membership rows through the anchor. Spectral extrema use only event-attributed
response rows through the anchor.

The canonical matrix order is:

```text
peak_risk_rank
mean_risk_rank
risk_slope
elevated_day_fraction
high_day_fraction
longest_elevated_run_fraction
attributed_decline_any
attributed_severe_decline_any
attributed_decline_day_fraction
first_attributed_decline_position
attributed_recovery_after_decline
worst_attributed_ndvi_delta
worst_attributed_ndvi_delta_missing
worst_attributed_ndmi_delta
worst_attributed_ndmi_delta_missing
worst_attributed_psri_delta
worst_attributed_psri_delta_missing
hazard_intensity
hazard_intensity_missing
usable_days_fraction
```

### 4.1 Risk-shape features: 6

Let the usable rows through the anchor be ordered by date, let `r_i` be
`daily_pressure_rank`, and let `U = usable_days`.

| Feature | Definition |
|---|---|
| `peak_risk_rank` | `max(r_i)` |
| `mean_risk_rank` | `mean(r_i)` |
| `elevated_day_fraction` | `count(r_i >= 3) / U` |
| `high_day_fraction` | `count(r_i >= 4) / U` |
| `risk_slope` | OLS coefficient of `r_i` against normalized time `x_i = (date_i - first_date) / (anchor_date - first_date)`; exactly `0` when the date span is zero |
| `longest_elevated_run_fraction` | Longest consecutive run of `r_i >= 3` in the ordered usable-row sequence, divided by `U` |

The run is consecutive in usable-observation sequence. Missing-pressure
calendar days are recorded as diagnostics but do not become synthetic low-risk
observations.

### 4.2 Response features: 5

A decline row has `daily_response_class` in
`{medium_decline, severe_decline}`.

| Feature | Definition |
|---|---|
| `attributed_decline_any` | `1` if any decline row occurs by anchor, otherwise `0` |
| `attributed_severe_decline_any` | `1` if any `severe_decline` occurs by anchor, otherwise `0` |
| `attributed_decline_day_fraction` | `min(1, decline_row_count / U)` |
| `first_attributed_decline_position` | For a decline: `clip((first_decline_date - event_start_date) / max(1, anchor_date - event_start_date), 0, 1)`; exactly `0` when no decline exists or the anchor span is zero |
| `attributed_recovery_after_decline` | `1` when a later event-attributed `recovery` occurs after a decline and by anchor, otherwise `0` |

`attributed_decline_any=0` disambiguates the no-decline value at position `0`
from a decline observed at event onset. The position is a finite bounded
feature, so it is not median-imputed.

### 4.3 Adverse spectral extrema and missingness: 6

Use only rows whose event-attributed response class is `medium_decline`,
`severe_decline`, or `recovery`, and only deltas already produced by the
echo-aware causal feature pipeline.

| Feature | Definition |
|---|---|
| `worst_attributed_ndvi_delta` | Minimum finite `ndvi_delta` by anchor |
| `worst_attributed_ndvi_delta_missing` | `1` when the preceding value is absent, else `0` |
| `worst_attributed_ndmi_delta` | Minimum finite `ndmi_delta` by anchor |
| `worst_attributed_ndmi_delta_missing` | `1` when the preceding value is absent, else `0` |
| `worst_attributed_psri_delta` | Maximum finite `psri_delta` by anchor |
| `worst_attributed_psri_delta_missing` | `1` when the preceding value is absent, else `0` |

Carried Sentinel echoes are context, not repeated acquisitions. Phase A must
not reconstruct adverse deltas from carried raw index values.

### 4.4 Hazard intensity and missingness: 2

Compute exactly one continuous intensity feature inside each hazard stratum:

| Hazard | `hazard_intensity` |
|---|---|
| `drought` | Minimum finite `spi_index` by anchor |
| `ponding_flooding` | Maximum finite `ponding_mm` by anchor |
| `heat` | Maximum finite `COALESCE(apparent_temperature, temperature)` by anchor |
| `damaging_wind` | Maximum finite `wind_speed` by anchor |
| `unattributed_decline` | Missing by definition |
| other normalized hazard | Missing unless a versioned schema explicitly defines its driver |

`hazard_intensity_missing` is `1` when intensity is absent and `0` otherwise.

### 4.5 Duration feature: 1

```text
usable_days_fraction = clip(usable_days / 21, 0, 1)
```

### 4.6 Scaling and missing values

- Clustering is performed separately per normalized hazard family.
- Continuous features are imputed with training-only, per-hazard medians,
  scaled with training-only, per-hazard median/IQR parameters, and clipped to
  `[-5, 5]`.
- An absent or non-finite IQR uses scale `1` and is recorded in the schema.
- Bounded and binary features are clipped to `[0, 1]` and are not
  robust-scaled.
- Every transformed feature is divided by the square root of the number of
  features in its feature group, preventing the six-feature groups from
  dominating the one-feature duration group solely by width.
- Holdout and later assignments only apply the frozen training schema.
- No vocabulary or imputation statistic may be fitted on holdout.

## 5. Explicit model exclusions

The following are prohibited model inputs in V2:

- current/anchor-day risk as a special feature distinct from the risk shape;
- escalation or de-escalation counts;
- raw event age, raw active/severe/quiet day counts, or raw duration;
- lifecycle state, stage, crop, season, or any other categorical feature;
- acquisition count, spectral age, or coverage;
- field ID, event ID, crop-instance ID, geometry, coordinates, or
  administrative location;
- absolute calendar date or year;
- event end, close reason, terminal state, recovery after the anchor, yield,
  crop-death labels, or any other post-anchor outcome;
- complete-season statistics, future-aware baselines, or post-anchor spectral
  acquisitions.

Calendar span, observed coverage, missing-pressure-day count, crop, stage,
season, and location may be retained in a diagnostic table for bias and support
audits. They must not enter the feature matrix or distance calculation.

## 6. Discovery, split, and frozen assignment

### 6.1 Split

- Train: `anchor_date <= 2025-12-31`.
- Holdout: `anchor_date > 2025-12-31`.
- Split identity is stored per event and is immutable.
- Parameter selection, feature fitting, HDBSCAN, prototype construction, and
  radii use train only.

### 6.2 Discovery

Run HDBSCAN independently for each hazard with `N_train_hazard` eligible
training events:

```text
min_cluster_size = max(100, ceil(0.005 * N_train_hazard))
min_samples       = 20
cluster_selection = EOM
```

Noise remains noise. It is not forced into the nearest training cluster during
discovery. Each accepted cluster stores an observed robust prototype: compute
the coordinate-wise median and choose the observed training vector nearest to
that median. Store a positive, finite training-distance radius and its
quantile/configuration in the model.

### 6.3 Assignment

Assignment is hazard-local and uses only the frozen schema, prototypes, radii,
and configured separation rule. Store:

- nearest candidate archetype;
- runner-up archetype;
- distance;
- candidate radius;
- `distance_ratio = distance / radius`;
- separation margin;
- accepted/novel outcome and reason;
- model version and feature schema hash.

For two or more candidates in the hazard,
`separation_margin = runner_up_distance - nearest_distance`. Acceptance
requires both `nearest_distance <= candidate_radius` and
`separation_margin >= configured_assignment_margin` (`0.05` in the VM command).
With only one candidate, the margin test is vacuously satisfied; the persisted
margin is null and the radius test still applies.

Distance is not a probability or calibrated confidence. A missing hazard
prototype, out-of-radius event, or ambiguous nearest pair yields an explicit
novel outcome.

The evaluation registry assigns each eligible event at most once, at its
anchor, using the frozen radius and separation rule. Ordinary future-date
appends do not change that assignment within one immutable model/generation.
Data gaps may carry the display assignment for continuity, but cannot create
or change it.

A backfill or source correction creates a new immutable generation and may
change the anchor or assignment. The repository does not yet maintain an
automatic cross-generation supersession registry, so an operator must compare
the old and new generations. Do not claim correction-stable identity or
automatic bitemporal reconciliation.

## 7. Phase A artifacts

A successful Phase A result consists of an immutable model directory and a
separate immutable evaluation directory. Evaluation verifies the model hashes
and does not modify the model directory.

| Directory | Artifact | Contract |
|---|---|---|
| Model | `archetype_manifest.json` | Status, generation/policy/implementation hashes, software versions, cutoff, schema/model versions, configuration, counts, and model-artifact hashes |
| Model | `feature_schema.json` | Ordered 20-feature schema and frozen per-hazard training medians, IQR scales, group weights, and assignment settings |
| Model | `event_anchors.parquet` | One row per source event: the four-way anchor status, causal audit diagnostics, and model features only for eligible rows |
| Model | `prototypes.parquet` | One finite observed prototype and positive radius per discovered archetype |
| Model | `archetype_catalog.parquet` | Archetype ID, hazard, training event/field support, diagnostic label/status, and model version |
| Model | `training_assignments.parquet` | HDBSCAN discovery membership or explicit discovery-noise outcome per eligible training event; not the frozen-radius registry assignment |
| Evaluation | `training_frozen_assignments.parquet` | Training anchors replayed through the same frozen radius and separation rule used for holdout |
| Evaluation | `holdout_assignments.parquet` | One frozen open-set assignment outcome per eligible holdout event |
| Evaluation | `evaluation_by_hazard.parquet` | Training and holdout counts, holdout novelty, and prototype availability by hazard |
| Evaluation | `prototype_overlap.parquet` | Separation and overlap ratio for every unordered same-hazard prototype pair |
| Evaluation | `subsample_stability.parquet` | Detailed rows from two deterministic, hazard-stratified 80% subsample refits |
| Evaluation | `subsample_stability.json` | Refit method, ARIs, matched-Jaccard, support metrics, and their gates |
| Evaluation | `event_archetype_assignments.parquet` | Combined frozen-radius training and holdout registry, one row per eligible event |
| Evaluation | `evaluation.json` | Combined hard-gate and quality-gate report with exact metrics |
| Evaluation | `evaluation_manifest.json` | Evaluation status, model lineage, V2 implementation hash, software versions, and evaluation-artifact hashes |

Discovery membership and frozen registry assignment answer different
questions and may legitimately differ for a training event. Never concatenate
`training_assignments.parquet` into the registry. The combined registry uses
`training_frozen_assignments.parquet` plus `holdout_assignments.parquet`, so
both splits obey the same open-set radius and runner-up-margin rule.

Features, eligible and non-eligible outcomes, and anchor audit columns
intentionally share `event_anchors.parquet`; Phase A does not emit duplicate
feature or exclusion files. It also does not create a label-review queue.

All Phase A catalog rows remain `diagnostic_unreviewed`. Passing engineering
and statistical gates does not change that status or prove that archetypes are
semantically or narratively distinct.

## 8. Gates

### 8.1 Hard gates

Any failure blocks the model:

1. exactly one anchor-ledger row per source `event_id`, using only the four
   statuses in section 3.3, and a non-null anchor date for every eligible row;
2. train and holdout event IDs are disjoint and match the anchor-date cutoff;
3. every observation and spectral source date used by a feature is on or before
   its anchor;
4. zero events with multiple archetypes or an archetype change;
5. the frozen V2 implementation hash, software versions, and model-artifact
   hashes verify before evaluation, and model versions and feature-schema
   hashes agree across assignments;
6. all transformed features, prototypes, and radii are finite and every radius
   is positive;
7. every publishable candidate has at least 100 distinct events and 50 distinct
   fields;
8. no duplicate published display labels after normalization;
9. every hazard with at least 1,000 eligible training events has at least one
   prototype;
10. all required artifacts exist, keys are unique, and reported counts reconcile.

Phase A does not actually mark rows published. Gates 7 and 8 identify which
rows could enter human review in Phase B.

### 8.2 Quality gates required before Phase B

All must pass:

| Metric | Gate |
|---|---:|
| Discovery noise rate | `<= 60%` of eligible training events |
| Overall holdout accepted rate | `>= 65%` |
| Overall holdout novelty rate | `<= 35%` |
| Supported-hazard holdout novelty | `<= 50%` for each hazard with at least 1,000 train events and nonempty holdout |
| Accepted holdout distance ratio | p90 `<= 0.90` |
| Subsample-refit adjusted Rand index | Minimum across hazard/refit rows, including noise, `>= 0.80` |
| Mutual-non-noise adjusted Rand index | Minimum across hazard/refit rows on events assigned to a cluster by both reference and refit, `>= 0.80` |
| Matched-archetype Jaccard | Minimum across hazard/refit rows of the median reference-archetype best-match Jaccard `>= 0.70` |
| Subsample-refit support | At least `75%` of reference archetypes meet the match gate in both required runs |
| Prototype overlap | At least `90%` of all unordered same-hazard prototype pairs have overlap ratio `< 1` |

For prototypes `i` and `j`, the reported overlap ratio is:

```text
overlap_ratio(i, j) = (radius_i + radius_j) / distance(prototype_i, prototype_j)
```

Each unordered same-hazard pair appears once. Equal prototype vectors have an
infinite overlap ratio. If there are no same-hazard pairs, the implementation
reports pair counts of zero and a vacuous non-overlap fraction of `1.0`.

The evaluation report must contain raw numerators and denominators, not only
rounded percentages. Missing metrics fail closed.

The stability stage is deterministic two-run 80% hazard-stratified **subsample
refit stability**, sampled without replacement. Each run uses
`max(1, floor(0.80 * hazard_event_count))` training events per hazard and
refits that subsample's hazard-local medians and IQRs before HDBSCAN. The first
ARI includes noise labels; the mutual-non-noise ARI guards against a large
shared noise class hiding cluster instability. A reference archetype counts as
supported only when its best refit-cluster Jaccard is at least `0.70` in both
runs. Reference archetypes choose their best refit cluster independently; the
matching is not one-to-one. Fewer than two mutually non-noise events yields a
mutual-non-noise ARI of `0`, failing that gate.

These are model-quality gates, not evidence of agronomic validity or proof that
the discovered groups form semantically or narratively distinct stories.

## 9. Blocked claims

Phase A does not authorize any claim that the model:

- detects or proves crop death;
- identifies a causal mechanism;
- shows physical geographic propagation;
- returns calibrated confidence or probability;
- is agronomically validated;
- generalizes across crops, seasons, or countries;
- predicts yield or final outcome;
- operates as continuous real-time streaming.

Permitted language is:

> Phase A evaluates whether one fixed, causal event-anchor representation can
> produce stable, separated statistical archetypes on a temporal holdout. The
> archetypes remain unreviewed and are not yet published to the map.

## 10. Phase A and Phase B sequence

### Phase A: current authorized scope

1. discover the latest completed immutable generation;
2. build the one-row-per-event anchor ledger and the eligible 20-feature matrix;
3. fit train-only per-hazard schemas and HDBSCAN models;
4. freeze prototypes and assign holdout events once;
5. run hard gates, holdout gates, two deterministic subsample refits, and
   all-pairs prototype-overlap diagnostics;
6. inspect the catalog, support diagnostics, prototypes, and assignments;
7. retain the outputs as diagnostic and unreviewed.

Do not run V1 `export-motifs`, build a V2 map bundle, change `server/.env`, or
restart the map server as part of Phase A.

### Phase B: blocked until every Phase A gate passes

Phase B will define expert label/family review, a curated map hierarchy,
scalable full-generation materialization, map summaries, bundle validation,
viewer language, release promotion, and runtime benchmarks. Phase B requires a
separate implementation and release decision; it is not implied by a successful
Phase A command.

## 11. Copy-paste VM preparation

Run from a fresh VM shell:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet

export REPO=/mnt/KSA-Oasis/El-Mohammed/mango-comet
export PYTHON="$REPO/.venv/bin/python"
export ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
export AS_OF=2026-05-17
export TRAINING_CUTOFF=2025-12-31

test -x "$PYTHON"
mkdir -p "$ROOT/models" "$ROOT/evaluations" "$ROOT/logs" "$ROOT/duckdb_tmp"
```

Discover the latest completed generation for the requested as-of date by
reading manifests, rather than relying on directory-name ordering:

```bash
export ROOT AS_OF
export GEN=$("$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"]) / "generations"
as_of = os.environ["AS_OF"]
candidates = []
for manifest_path in root.glob("*/manifest.json"):
    try:
        payload = json.loads(manifest_path.read_text())
        run = payload.get("run") or {}
        if run.get("status") != "complete" or str(run.get("as_of_date")) != as_of:
            continue
        generation_id = str(run.get("generation_id") or "")
        if not generation_id:
            continue
        candidates.append((manifest_path.stat().st_mtime_ns, generation_id, manifest_path.parent))
    except (OSError, ValueError, TypeError):
        continue
if not candidates:
    raise SystemExit(f"No completed generation found for {as_of} under {root}")
print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
)

test -s "$GEN/manifest.json"
echo "GEN=$GEN"
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); r=p["run"]; assert r["status"]=="complete"; print(json.dumps({"generation_id":r["generation_id"],"as_of_date":r["as_of_date"],"event_count":r["event_count"]},indent=2))' "$GEN/manifest.json"
```

Resolve the RAPIDS interpreter used for GPU discovery:

```bash
export RAPIDS_PYTHON=$(tr -d '\r\n' < "$ROOT/logs/latest_rapids_python.txt")
test -x "$RAPIDS_PYTHON"
CUDA_VISIBLE_DEVICES=0 "$RAPIDS_PYTHON" -c 'import cupy,cuml; print({"cupy":cupy.__version__,"cuml":cuml.__version__})'
```

The V1 pointer is diagnostic only. Never reuse it as the V2 model:

```bash
if [ -s "$ROOT/logs/latest_motif_model_path.txt" ]; then
  export V1_DIAGNOSTIC_MODEL=$(tr -d '\r\n' < "$ROOT/logs/latest_motif_model_path.txt")
  echo "V1 diagnostic only: $V1_DIAGNOSTIC_MODEL"
fi
```

## 12. Phase A build command

Use this durable-launch pattern:

```bash
export ARCH_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export ARCH_MODEL="$ROOT/models/archetype_v2_anchor21_train_2025_$ARCH_TAG"
export BUILD_JSON="$ROOT/logs/archetype_v2_build_$ARCH_TAG.json"
export BUILD_LOG="$ROOT/logs/archetype_v2_build_$ARCH_TAG.log"
export BUILD_STATUS="$ROOT/logs/archetype_v2_build_$ARCH_TAG.status"
export BUILD_PID="$ROOT/logs/archetype_v2_build_$ARCH_TAG.pid"

test ! -e "$ARCH_MODEL"
export REPO RAPIDS_PYTHON GEN TRAINING_CUTOFF ARCH_MODEL
export BUILD_JSON BUILD_LOG BUILD_STATUS

nohup bash -c '
  cd "$REPO" || {
    printf "%s\n" 98 >"$BUILD_STATUS"
    exit 98
  }

  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$RAPIDS_PYTHON" server/weekly_story_monitor.py build-archetypes-v2 \
      --generation-dir "$GEN" \
      --training-through "$TRAINING_CUTOFF" \
      --output-dir "$ARCH_MODEL" \
      --engine gpu \
      --radius-quantile .95 \
      --assignment-margin .05 \
      --threads 32 \
      --memory-limit 96GB \
      --temp-dir "$ROOT/duckdb_tmp" \
      >"$BUILD_JSON" 2>"$BUILD_LOG"

  status=$?
  printf "%s\n" "$status" >"$BUILD_STATUS"
  exit "$status"
' </dev/null >/dev/null 2>&1 &

printf '%s\n' "$!" | tee "$BUILD_PID"
```

Monitor it without treating an initially quiet log as a failure:

```bash
if [ -s "$BUILD_STATUS" ]; then
  echo "BUILD STATUS: $(cat "$BUILD_STATUS")"
elif kill -0 "$(cat "$BUILD_PID")" 2>/dev/null; then
  echo "ARCHETYPE V2 BUILD STILL RUNNING"
  ps -fp "$(cat "$BUILD_PID")"
else
  echo "BUILD PROCESS EXITED WITHOUT A STATUS FILE"
fi

tail -n 100 "$BUILD_LOG"
free -h
nvidia-smi
```

On status `0`, command stdout must be one JSON object:

```bash
"$PYTHON" -m json.tool "$BUILD_JSON"

ARCH_MODEL="$ARCH_MODEL" BUILD_JSON="$BUILD_JSON" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["BUILD_JSON"]).read_text())
model = Path(os.environ["ARCH_MODEL"])
model_manifest = json.loads((model / "archetype_manifest.json").read_text())
assert set(result) == {
    "status", "phase", "model_dir", "model_version", "feature_version",
    "anchor_counts", "quality_gate_status", "warning",
}, result
assert result["status"] == "complete", result
assert result["phase"] == "phase_a_diagnostic", result
assert Path(result["model_dir"]).resolve() == model.resolve(), result
assert result["feature_version"] == "causal_event_anchor_features_v2", result
assert result["quality_gate_status"] == "not_evaluated", result
assert len(model_manifest["implementation_sha256"]) == 64, model_manifest
assert set(model_manifest["software_versions"]) == {
    "python", "duckdb", "numpy", "pandas", "pyarrow", "scikit_learn",
    "cupy", "cuml",
}, model_manifest

counts = result["anchor_counts"]
outcomes = counts["outcomes"]
allowed = {
    "eligible", "watch_only", "insufficient_evidence",
    "season_boundary_before_maturity",
}
assert set(outcomes) <= allowed, outcomes
assert sum(outcomes.values()) == counts["total_events"], counts
assert outcomes.get("eligible", 0) == counts["eligible_events"], counts
assert counts["training_events"] + counts["holdout_events"] == counts["eligible_events"], counts

for name in (
    "archetype_manifest.json", "feature_schema.json", "event_anchors.parquet",
    "prototypes.parquet", "archetype_catalog.parquet",
    "training_assignments.parquet",
):
    assert (model / name).is_file(), name
print(json.dumps({
    "build": "PASS",
    "model_dir": str(model),
    "model_version": result["model_version"],
    "anchor_counts": counts,
}, indent=2))
PY

printf '%s\n' "$ARCH_MODEL" > "$ROOT/logs/latest_archetype_v2_path.txt"
```

Never write the latest-model pointer before the build status and JSON checks
pass.

## 13. Phase A evaluation command

Discover the latest valid V2 model from its manifest, using the pointer only as
a fast path:

```bash
export ROOT
export ARCH_MODEL=$("$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
pointer = root / "logs/latest_archetype_v2_path.txt"
candidates = []
if pointer.is_file():
    candidates.append(Path(pointer.read_text().strip()))
candidates.extend(path.parent for path in (root / "models").glob("archetype_v2_*/archetype_manifest.json"))
valid = []
for model in candidates:
    manifest = model / "archetype_manifest.json"
    try:
        payload = json.loads(manifest.read_text())
        if payload.get("status") == "complete" and payload.get("phase") == "phase_a_diagnostic":
            valid.append((manifest.stat().st_mtime_ns, str(model.resolve())))
    except (OSError, ValueError, TypeError):
        pass
if not valid:
    raise SystemExit("No completed Archetype V2 model found")
print(max(valid)[1])
PY
)

test -s "$ARCH_MODEL/archetype_manifest.json"
echo "ARCH_MODEL=$ARCH_MODEL"
```

Use this launch pattern:

```bash
export EVAL_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export EVAL_DIR="$ROOT/evaluations/archetype_v2_$EVAL_TAG"
export EVAL_JSON="$ROOT/logs/archetype_v2_eval_$EVAL_TAG.json"
export EVAL_LOG="$ROOT/logs/archetype_v2_eval_$EVAL_TAG.log"
export EVAL_STATUS="$ROOT/logs/archetype_v2_eval_$EVAL_TAG.status"
export EVAL_PID="$ROOT/logs/archetype_v2_eval_$EVAL_TAG.pid"

test ! -e "$EVAL_DIR"
export REPO RAPIDS_PYTHON ARCH_MODEL EVAL_DIR
export EVAL_JSON EVAL_LOG EVAL_STATUS

nohup bash -c '
  cd "$REPO" || {
    printf "%s\n" 98 >"$EVAL_STATUS"
    exit 98
  }

  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$RAPIDS_PYTHON" server/weekly_story_monitor.py evaluate-archetypes-v2 \
      --model-dir "$ARCH_MODEL" \
      --output-dir "$EVAL_DIR" \
      --stability-runs 2 \
      >"$EVAL_JSON" 2>"$EVAL_LOG"

  status=$?
  printf "%s\n" "$status" >"$EVAL_STATUS"
  exit "$status"
' </dev/null >/dev/null 2>&1 &

printf '%s\n' "$!" | tee "$EVAL_PID"
```

Monitor and inspect status exactly as for the build, substituting `EVAL_*` for
`BUILD_*`. A status of `0` means evaluation completed; it does not mean gates
passed.

Validate both the command result and the persisted evaluation report. The
command returning status `0` only means the report was written; the final two
assertions below enforce the Phase A gates:

```bash
test -s "$EVAL_JSON"
test -s "$EVAL_DIR/evaluation.json"
"$PYTHON" -m json.tool "$EVAL_JSON"
"$PYTHON" -m json.tool "$EVAL_DIR/evaluation.json"

EVAL_JSON="$EVAL_JSON" EVAL_DIR="$EVAL_DIR" ARCH_MODEL="$ARCH_MODEL" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["EVAL_JSON"]).read_text())
evaluation_dir = Path(os.environ["EVAL_DIR"])
model_dir = Path(os.environ["ARCH_MODEL"])
report = json.loads((evaluation_dir / "evaluation.json").read_text())
manifest = json.loads((model_dir / "archetype_manifest.json").read_text())
evaluation_manifest = json.loads(
    (evaluation_dir / "evaluation_manifest.json").read_text()
)

assert set(result) == {
    "status", "phase", "evaluation_dir", "model_version",
    "hard_gates_passed", "quality_gates_passed", "metrics", "warning",
}, result
assert result["status"] == "complete", result
assert result["phase"] == "phase_a_diagnostic", result
assert Path(result["evaluation_dir"]).resolve() == evaluation_dir.resolve(), result
assert result["model_version"] == manifest["model_version"], result

for name in (
    "training_frozen_assignments.parquet", "holdout_assignments.parquet",
    "evaluation_by_hazard.parquet", "prototype_overlap.parquet",
    "subsample_stability.parquet", "subsample_stability.json",
    "event_archetype_assignments.parquet",
    "evaluation.json", "evaluation_manifest.json",
):
    assert (evaluation_dir / name).is_file(), name

assert set(evaluation_manifest) == {
    "status", "phase", "model_version", "model_manifest_sha256",
    "implementation_sha256", "software_versions", "artifacts",
}, evaluation_manifest
assert evaluation_manifest["status"] == "complete", evaluation_manifest
assert evaluation_manifest["phase"] == "phase_a_diagnostic", evaluation_manifest
assert evaluation_manifest["model_version"] == manifest["model_version"], evaluation_manifest
assert (
    evaluation_manifest["implementation_sha256"]
    == manifest["implementation_sha256"]
), evaluation_manifest
assert (
    evaluation_manifest["software_versions"] == manifest["software_versions"]
), evaluation_manifest
assert set(evaluation_manifest["artifacts"]) == {
    "training_frozen_assignments.parquet", "holdout_assignments.parquet",
    "evaluation_by_hazard.parquet", "prototype_overlap.parquet",
    "subsample_stability.parquet", "subsample_stability.json",
    "event_archetype_assignments.parquet", "evaluation.json",
}, evaluation_manifest

assert set(report) == {
    "status", "phase", "model_version", "metrics", "gates", "warning",
    "model_artifact_hashes_verified", "model_manifest_sha256",
}, report
assert report["status"] == "complete", report
assert report["phase"] == "phase_a_diagnostic", report
assert report["model_version"] == manifest["model_version"], report
assert report["model_artifact_hashes_verified"] is True, report

hard = report["gates"]["hard"]
quality = report["gates"]["quality"]
assert set(hard["checks"]) == {
    "unique_training_events", "unique_training_frozen_events",
    "unique_holdout_events", "training_assignment_cohort_matches",
    "training_frozen_assignment_cohort_matches",
    "holdout_assignment_cohort_matches",
    "train_holdout_disjoint", "one_archetype_per_event",
    "finite_prototypes_and_radii", "minimum_candidate_support",
    "no_duplicate_published_labels", "supported_hazards_have_prototypes",
    "manifest_counts_reconcile", "catalog_prototype_ids_match",
    "catalog_member_counts_reconcile", "training_assignment_ids_valid",
    "model_version_consistent", "feature_schema_hash_consistent",
    "one_anchor_ledger_row_per_event",
    "eligible_anchor_dates_present", "no_post_anchor_feature_evidence",
    "training_split_nonempty", "holdout_split_nonempty",
    "anchor_counts_match_manifest", "combined_assignment_unique",
    "model_artifact_hashes",
}, hard
assert set(quality["checks"]) == {
    "discovery_noise_rate", "holdout_accepted_rate", "holdout_novelty_rate",
    "supported_hazard_novelty", "accepted_distance_ratio_p90",
    "prototype_nonoverlap_fraction", "stability_ari",
    "stability_mutual_non_noise_ari", "stability_matched_jaccard",
    "stability_support",
}, quality
assert hard["passed"] == all(hard["checks"].values()), hard
assert quality["passed"] == all(quality["checks"].values()), quality
hard_failed = sorted(name for name, value in hard["checks"].items() if not value)
quality_failed = sorted(name for name, value in quality["checks"].items() if not value)

metrics = report["metrics"]
assert set(metrics) == {
    "training_event_count", "discovery_noise_count", "discovery_noise_rate",
    "holdout_event_count", "holdout_accepted_count", "holdout_accepted_rate",
    "holdout_novelty_rate", "accepted_distance_ratio_p90",
    "prototype_nonoverlap_fraction", "prototype_pair_count",
    "prototype_nonoverlapping_pair_count", "prototype_overlapping_pair_count",
    "stability_status", "minimum_adjusted_rand_index",
    "minimum_mutual_non_noise_adjusted_rand_index",
    "conservative_matched_jaccard",
    "reference_archetype_count", "supported_reference_count",
    "supported_reference_fraction",
}, metrics
assert metrics["stability_status"] == "complete", metrics
assert (
    metrics["prototype_nonoverlapping_pair_count"]
    + metrics["prototype_overlapping_pair_count"]
    == metrics["prototype_pair_count"]
), metrics
assert result["metrics"] == metrics, result
assert result["hard_gates_passed"] == hard["passed"], result
assert result["quality_gates_passed"] == quality["passed"], result

print(json.dumps({
    "hard_passed": hard["passed"],
    "hard_failed": hard_failed,
    "quality_passed": quality["passed"],
    "quality_failed": quality_failed,
    "metrics": metrics,
}, indent=2))

assert hard["passed"] and not hard_failed
assert quality["passed"] and not quality_failed
PY
```

If this check fails, retain the output, diagnose the failed metric, and create a
new versioned experiment. Do not tune until the map looks attractive, overwrite
the failed model, export assignments to the viewer, or weaken a gate without a
recorded methodology decision.
