# Crop-impact stories V4: dual-clock monitoring contract

Status: implementation contract. Numeric thresholds remain uncalibrated and
must not be presented as agronomic validation, diagnosis, yield loss, crop
death, or causal effect.

## What a story is

A story is one stable local same-hazard exposure viewed for one crop:

```text
daily field pressure
  -> exact local component
  -> persistent exposure_id
  -> exposure_id x crop = stable incident_id
```

The deterministic V3 identity hierarchy remains authoritative. A learned
motif may describe a completed or sufficiently mature incident, but it never
creates, merges, splits, closes, or renames an incident.

V4 fixes the evidence clocks around that identity. It does not silently turn
the weekly reporting bucket into the time at which weather or Sentinel-2
evidence was observed.

## Three clocks

### Daily pressure clock

Pressure is tracked independently for every supported hazard and calendar day.
Missing weather inputs and observed low pressure are different states. Only an
adequately observed pressure day can advance onset, persistence, or quiet
clocks. Sentinel-2 evidence never makes a weather day observable.

The viewer keeps normal same-day observations in `field_day_state_v4.parquet`
and writes only late-known weather to the sparse
`pressure_observations_v4.parquet` supplement, ordered for field drilldown.
Field history unions both sources and applies `pressure_knowledge_time <= as-of`
before displaying the effective observation date; late arrivals are never
projected into an earlier as-of response.

The current source is a per-field weather-derived pressure product, not a raw
meteorological raster. The UI and documentation must use that wording unless a
separate weather-grid source is supplied.

### Acquisition-grain crop-evidence clock

One usable Sentinel-2 source date can create at most one crop-evidence update
per crop instance. Carried values and echo rows age the previous observation;
they are never new decline or recovery evidence.

Every acquisition attempt retains:

- source/acquisition date;
- first-known or explicit availability time;
- usability and rejection reason;
- valid-pixel/cloud/quality metadata;
- spectral values and the prior usable reference acquisition;
- response deltas and response class.

A rejected acquisition is an observed opportunity but cannot become a
reference. Recovery requires a later usable acquisition. If no usable
follow-up exists, the incident is censored/unknown rather than recovered,
unresolved damage, or dead.

Partial external attempt ledgers augment the acquisitions derived from the
daily source. They are deduplicated by field/source date, retain unmatched
rejected attempts, and never replace historical derived acquisitions. Crop
instance assignment uses only crop context effective on or before the source
date and becomes knowable no earlier than both the attempt and crop context.

### Story checkpoint clock

Incident identity and lifecycle remain immutable V3 checkpoints. On an as-of
day, a checkpoint is visible only when its full `story_known_time` is not later
than the playhead day. That timestamp is the maximum of the source checkpoint
time and the attributed membership, pressure, crop/stage, and applicable S2
impact/recovery knowledge times for the checkpoint week; `story_known_date` is
only its display date. Daily pressure and sparse Sentinel-2 evidence can appear
as an unassigned prelude before the first known checkpoint; the future incident
ID must not be projected backward.

Strict releases reject inferred source checkpoint clocks, source clocks below
their contributing evidence bound, and membership that cannot attribute the
required field/crop/hazard evidence. Reconstructed releases may raise the bound
but remain explicitly diagnostic and retain the source and component timestamps
for audit.

Viewer schema `crop-incident-viewer-v4/2` adds a required
`lifecycle_reconciliation_v4.parquet` ledger. For every displayed checkpoint it
reconciles the V3 membership counts and every positive pressure, decline, and
recovery claim against evidence known by that checkpoint. Any contradiction
blocks the bundle atomically. This is deliberately narrower than replaying the
V3 lifecycle: V4 does **not** recompute lifecycle state, infer component
absence, or claim that V4 alone owns story start/end. The correct presentation
claim is “knowledge-gated V3 lifecycle with fail-closed V4 positive-evidence
reconciliation.” Older V4/1 bundles are rejected and must be rebuilt.

## Authoritative V4 evidence ledgers

V4 evidence is published as a separate immutable directory. It never mutates
the source generation or the V3 incident release.

### `crop_day_context_v4.parquet`

Natural key: `(field_id, crop_instance_id, observation_date)`.

It contains crop, season, raw and controlled stage, stage effective/knowledge
time, observability, source/release provenance, and policy identity. Crop
stage is changing context and has zero incident-identity weight.

### `field_day_pressure_v4.parquet`

Natural key:
`(field_id, crop_instance_id, observation_date, hazard_family)`.

It contains pressure observation/availability time, hazard-specific score,
rank and band, raw driver inputs, source identity, completeness flags, and an
explicit missing reason. Simultaneous hazards remain separate rows.

### `field_s2_acquisition_v4.parquet`

Natural key: `(field_id, crop_instance_id, acquisition_id)`.

It contains acquisition/source and availability time, acquisition attempt and
usability flags, QA, spectral values, prior usable reference, deltas, response,
rejection reason, and provenance. A source date is unique within a crop
instance. A repeated/corrected source date is rejected until the explicit
revision/supersession contract described below exists.

An acquisition whose source date precedes the field's first causally known crop
assignment is not attached to that future crop. It is excluded from this
crop-qualified ledger and counted by origin under
`reconciliation.s2_acquisitions.excluded_without_causal_crop_count` in the
evidence manifest.

## Availability modes

`strict` mode requires retained source availability timestamps and is eligible
for operational replay.

An enriched daily source is accepted only with its immutable sidecar. The
sidecar availability mode must equal the evidence-build mode, its output hash
must match the parquet, and its field/day keys must exactly reconcile with the
selected generation. A reconstructed source cannot be relabelled `strict`.

`reconstructed` mode uses the first daily row on which an acquisition/source
is present and uses the daily observation date for pressure knowledge. It is
useful for historical development but must remain labelled diagnostic because
the current historical parquets do not retain the original ingest timestamps.

Both modes enforce:

```text
evidence_time <= knowledge_time
knowledge_time <= released_at
effective calendar date <= release_as_of
```

`release_as_of` is the daily map playhead. `released_at` is a timezone-aware,
monotonically increasing UTC ingest watermark. Multiple immutable releases may
share one `release_as_of` date—for example, daily weather followed by a later
Sentinel-2 ingest—but each must advance `released_at`.

V4 intentionally has no partial correction mechanism. Its manifests declare
`append_only_no_revisions`: a late correction to an already published natural
key fails closed. Supporting that case requires a future explicit revision ID,
supersedes link, and knowledge-time projection; operators must not overwrite a
prior release or disguise a correction as an ordinary append.

## Learning

Learning uses eligible completed incidents only. It operates after incident
tracking and cannot change the operational story graph.

1. Build one terminal trajectory vector per eligible completed incident.
2. Fit robust transforms and HDBSCAN only on the training interval, stratified
   by crop and hazard.
3. Purge exposure/lineage families across train, calibration, and holdout.
4. Calibrate open-set radii and runner-up margins on a later calibration
   interval; never on outer holdout.
5. Replay prefixes using only evidence whose knowledge time is available at
   each as-of.
6. Compare a live prefix only with prototypes having compatible daily-weather
   maturity and distinct usable-S2-acquisition maturity.
7. Constrain the **combined** nearest-prototype assignment rule with reviewed
   novel calibration incidents, so the whole crop/hazard/maturity stratum—not
   each prototype independently—respects the configured false-accept ceiling.

`score-live` writes one immutable as-of delta for confirmed, non-terminal
incidents only. `CANDIDATE` checkpoints are not eligible. Daily weather after a
weekly checkpoint is assigned to the latest causally known ownership whose
effective week is not later than the weather day; close/merge boundaries stop
carry and split fields transition only when their child membership is known.
Sentinel-2 remains acquisition-week owned.

Live outputs are explicitly `pending`, `novel_unassigned`,
`tentative_weather_only`, or `tentative_crop_evidence_supported`. They are
hash-bound to the incident, evidence, viewer, and frozen prefix-model releases,
remain `map_publication_supported=false`, and cannot create, merge, split,
close, or rename a story. Stage, district, and season are audit/facet
dimensions by default and have zero core trajectory-distance weight.
Discovered labels remain unreviewed until an immutable review overlay approves,
merges, or rejects them.

## Map contract

The V4 viewer uses one daily playhead and linked map/detail views. The map has
three visually separate evidence layers:

- daily pressure footprint;
- acquisition-to-acquisition crop evidence, step-held and visibly aging;
- latest causally available incident checkpoint and exact footprint.

Pressure, crop-impact, and full story geometries never substitute for one
another. Selected histories use exact footprint outlines with age fading; no
centroid arrows, interpolated hulls, smooth morphs, or implied propagation are
allowed.

The field and selected-incident inspectors use one aligned linear-time
trajectory with:

- one non-overlapping lane per hazard, distinguishing missing, partial,
  observed-low, and elevated pressure;
- Sentinel-2 source and knowledge markers, rejected attempts, response class,
  and step-held fresh/aging/stale state using the bundle's freshness policy;
- a continuous known crop-stage band;
- one row per incident, with prelude, lifecycle blocks, and explicit
  start/recovery/pressure-off/close milestones.

Long histories are deterministically bounded for rendering while API
truncation is disclosed. Aggregated Sentinel-2 markers do not draw a fictitious
source-to-known connector. At field zoom, normal click selects a field and
Shift-click selects/cycles the incident footprint beneath it.

Every mappable field contributes at country scale through a complete
precomputed aggregate/density representation. Middle zoom may use centroids;
high zoom uses exact field polygons. The UI reports source, represented,
unmappable, and truncated counts.

## Release gates

A V4 release fails closed when any of the following occurs:

- future evidence or knowledge-time leakage;
- an echo row becoming a new acquisition;
- a rejected acquisition becoming a reference;
- weather missingness advancing pressure clocks;
- a Sentinel update making weather coverage adequate;
- duplicate source-date weighting;
- historical append rewrites without an explicit superseding revision;
- cross-fold exposure or lineage-family leakage;
- unsupported maturity falling back to another stratum;
- incomplete country overview or geometry-role substitution;
- request-time clustering or trajectory reconstruction.

The viewer is precomputed. H100s may accelerate independent offline clustering
folds or preprocessing only after measurement; they do not solve HTTP or
browser-rendering latency.

The exact VM paths, durable runner, status/resume commands, evidence audits,
server launch, latency gates, and operational append sequence are in
[`INCIDENT_V4_VM_RUNBOOK.md`](INCIDENT_V4_VM_RUNBOOK.md).
