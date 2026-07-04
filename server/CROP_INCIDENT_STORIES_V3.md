# Crop-Impact Incident Stories V3

This document is the authoritative implementation contract for V3. It
supersedes V1/V2 whenever the product uses the words **story**, **incident**,
or **story clustering**. V1 remains the causal field-episode source. V2 remains
an immutable failed diagnostic experiment; it must not be promoted as the V3
story taxonomy.

## 1. Product objective

The product monitors crops, not weather in isolation. It must answer:

1. What crop-impact incident has started locally?
2. Which crop instances and stages are affected now?
3. Is the incident growing, persisting, relapsing, recovering, unresolved, or
   data-censored?
4. How did its pressure and crop-impact footprints evolve?
5. Which reviewed historical story archetypes does the live partial story
   tentatively resemble?

A memoryless weekly weather/proximity component is the baseline, not the
product claim. V3 earns the word “story” only by preserving causal identity and
lifecycle across weeks.

Here **causal** means prefix-safe/as-of computation: a state uses no evidence
that arrived after that state. It does not mean the system has identified a
biophysical cause or estimated a causal effect.

## 2. Non-overloaded objects

| Object | ID | Contract |
|---|---|---|
| Crop instance | `crop_instance_id` | One field, crop, season/regime, and observed growing cycle. |
| Field episode | `episode_id` (`event_id` compatibility alias) | One crop instance’s hazard-specific causal lifecycle. |
| Weekly exposure component | `component_id` | One significant local same-hazard footprint in one week. Never stable across weeks. |
| Exposure incident | `exposure_id` | Weekly exposure components linked causally across time, independent of crop identity. |
| Crop-impact story | `incident_id` | One exposure incident’s impact on one crop. This is the primary user-facing story identity. |
| Archetype | `archetype_id` | Optional, model-versioned tag learned from completed crop-impact stories. Never identity. |

One exposure may have several linked crop-impact stories. A heat exposure that
affects maize and beans has one `exposure_id` and two `incident_id` values.
Crop stage changes never create a new story ID.

## 3. Required source artifacts

V3 builds from a full immutable monitoring generation, not from the current
viewer bundle:

- `daily_causal_signals.parquet`: all monitored crop instances, crop/stage,
  pressure coverage, echo-aware response, and hazard/intensity context;
- `event_state_snapshots.parquet`: causal weekly affected-episode state;
- `event_windows.parquet`: episode closure/censoring audit;
- `story_day_membership.parquet`: episode evidence lineage;
- `map_field_geometry.parquet`: field geometry and administrative context;
- `manifest.json`: generation, policy, and source lineage.

`event_state_snapshots` contains affected episodes only and omits stage. It
must never be used as the monitored-field denominator. The denominator comes
from weekly canonical rows derived from `daily_causal_signals`.

## 4. Crop and stage semantics

Every weekly crop-instance context records the raw stage and one versioned
stage bucket:

- `emergence`;
- `vegetative`;
- `flowering`;
- `fruiting_or_grain_fill`;
- `maturity_or_harvest`;
- `off_season`;
- `unknown`.

Unknown source values remain `unknown`; the code must not invent agronomic
sensitivity. Stage is time-varying context. Each story records stage at first
evidence, confirmation, peak, and every weekly update.

The build fails closed when known-stage mapping is below the versioned overall
gate, or when a crop above the support floor misses its per-crop gate. The
context manifest records per-crop coverage and the most common unmapped raw
labels so aliases can be reviewed and versioned instead of patched in the UI.

All rates displayed to users have an explicit denominator, preferably:

```text
affected crop instances for crop C and stage S
------------------------------------------------
all monitored crop instances for crop C and stage S
```

The API must report monitored, affected, rendered, missing-evidence, and
missing-geometry counts separately.

## 5. V3 build phases

### 5.1 Weekly field context

Materialize one causal row per `(timeline_bucket, field_id, crop_instance_id)`
from all monitored rows. Join at most one episode per
`(timeline_bucket, field_id, hazard)` using deterministic urgency order:

```text
SEVERE > ACTIVE > QUIET_PENDING > WATCH > RECOVERING > DATA_GAP
then current risk, then stable episode ID
```

Concurrent hazards remain separate episode lanes.

### 5.2 Frozen stage-aware baseline

Fit only on weeks at or before the configured baseline cutoff. For each
`hazard × stage_bucket × ISO-week`, estimate the expected ACTIVE/SEVERE rate
with Beta-binomial shrinkage toward the hazard/week global rate. Persist the
baseline and its training cutoff. Assignment weeks only apply that frozen
baseline.

### 5.3 Metric cells and significant local components

Use a versioned fixed metric grid; the MVP default is 5 km. For each
`week × hazard × cell`, compute monitored denominators, active/severe counts,
crop/stage composition, evidence coverage, and expected counts. Apply an
FDR-controlled significance test per `week × hazard`, with a versioned
multi-field severe-response override.

WATCH fields may form a displayed frontier but cannot establish a component.
Low-coverage cells cannot start or close an incident.

Connect significant 8-neighbor cells with deterministic union-find. Display
the union of actual cells, not a convex hull across unrelated points.

### 5.4 Temporal incident tracking

Compare only same-hazard components from the previous configured observed
weeks. Candidate edges require shared episodes/fields, overlapping cells, or
bounded footprint distance. Score edges using versioned weights for:

- active-episode overlap;
- cell/footprint overlap;
- recent incident-member overlap;
- normalized centroid distance;
- observed-week gap penalty.

Crop-stage similarity is retained as a diagnostic column but has zero identity
weight. The physical/episode weights are normalized after excluding stage, so
phenology cannot split or preserve an `exposure_id`.

Primary one-to-one matches preserve `exposure_id`. Additional qualifying edges
record lineage. On split, the best child keeps identity and other children
receive new IDs with `split_from`. On merge, the best parent keeps identity and
other parents close as `merged_into`. Sorting and tie breaks are deterministic.

### 5.5 Crop-impact stories

For every exposure/week, group members by crop. The crop-impact story keeps a
stable `incident_id` while stage composition changes. Weekly story state
includes:

- monitored and affected denominators by stage;
- pressure-core, watch-frontier, impact-lag, recovering, unresolved, and
  data-gap members;
- entering, persisting, exiting, and recovered fields;
- current and peak severity;
- pressure and impact footprint geometry;
- crop/stage distribution;
- response evidence and acquisition freshness;
- split/merge/relapse milestones.

## 6. Story lifecycle

- `CANDIDATE`: first significant local component, admitted only when the exact
  footprint and crop denominator pass the same-week coverage gates.
- `CONFIRMED`: linked support in a second observed week, or versioned
  multi-field severe pressure plus fresh aligned crop response.
- `ACTIVE`: confirmed pressure core exists.
- `PRESSURE_QUIET`: confirmed pressure has stopped while the response/recovery
  grace window remains open.
- `RECOVERING`: pressure core is quiet but attributed impact remains.
- `RELAPSED`: pressure returns during the recovery/quiet grace period.
- `CLOSED_RECOVERED`: recovery evidence satisfies the calibrated contract.
- `CLOSED_PRESSURE_QUIET_UNCONFIRMED`: pressure ended without fresh response
  confirmation.
- `CLOSED_CANDIDATE_EXPIRED`: first-week evidence did not receive enough
  observed-week support to become a confirmed incident.
- `CLOSED_RESPONSE_UNRESOLVED`: adverse crop response remains beyond the
  calibrated window.
- `CLOSED_SEASON_CENSORED`: crop/season boundary; never automatic recovery.
- `CLOSED_DATA_CENSORED`: inadequate observations prevent a conclusion.
- `MERGED_INTO`: lineage closure after a component merge.

Low-coverage evidence remains a prelude, not a story start.

Persist `first_evidence_week`, `confirmed_week`, evidence time, and knowledge
time separately. Knowledge time is the earliest time the system could have
known the weekly state; it may never precede the evidence it summarizes.
Missing/low-coverage weeks freeze start/end clocks. A field that exits remains
in immutable membership history. Recovery can close a story only after its
attributed unresolved registry is empty; recovery in one field cannot hide
unresolved response in another.

There is no `DEAD` outcome. Death requires independent survey, harvest/yield,
or a separately validated outcome model.

Unresolved episode ownership crosses an exact crop split/merge edge before the
new week’s recovery evidence is applied. Each unresolved
`(field, crop_instance, episode)` has one same-hazard story owner; ambiguity
fails closed. A terminal story clears that registry before any recurrence.

## 7. Story clustering paradigm

Story clustering remains part of V3, but it consumes completed crop-impact
stories. It does not create story identity.

Completed-story features include pressure duration, recovery lag, peak timing,
peak/cumulative affected rate, severe fraction, early footprint growth,
maximum area, week-to-week retention, relapse and split/merge counts, recovery
and unresolved fractions, data-gap fraction, onset/peak stage distributions,
crop/stage entropy, and hazard-intensity trajectory summaries.

Discovery is initially stratified by `crop × hazard`. Noise remains noise.
Location, absolute date, IDs, and administrative names are diagnostic only.
Every group is `diagnostic_unreviewed` until stability and expert narration
gates pass. Robust-scaled features are weighted by semantic family so the
fourteen onset/peak stage fractions cannot dominate duration, crop impact,
spatial evolution, lineage, or hazard intensity merely by dimensionality.

Training split:

- train: stories ending on/before the cutoff;
- holdout: stories beginning after the cutoff;
- embargo: stories crossing the cutoff.

Data-censored, season-censored, and `MERGED_INTO` fragments are not completed
outcomes and are excluded from archetype discovery. Their counts remain in the
model manifest.

District, season, and crop/stage support audits are mandatory.

## 8. Causal live archetype matching

Do not compare an early live story directly with a completed-story vector.
After completed-story discovery, reconstruct training prefixes at 1, 2, 4,
and 8 observed weeks. Fit frozen horizon-specific prototypes/radii for each
`crop × hazard × archetype × horizon`.

A live story uses the latest supported horizon not exceeding its age. It must
pass both radius and runner-up separation; otherwise it remains
`novel_unassigned`. The provisional tag may change as evidence accumulates;
`incident_id` never changes. The UI calls this a **tentative pattern**, never a
story identity or outcome probability.

## 9. Artifacts

An immutable V3 generation contains at minimum:

- `field_week_context.parquet`;
- `stage_baseline.parquet`;
- `weekly_exposure_cells.parquet`;
- `weekly_components.parquet`;
- `component_membership.parquet`;
- `exposure_weekly_state.parquet`;
- `incident_weekly_state.parquet`;
- `incident_stage_summary.parquet`;
- `incident_membership.parquet`;
- `incident_windows.parquet`;
- `incident_lineage.parquet`;
- `completed_incident_features.parquet`;
- `manifest.json` with source/policy/config hashes and reconciliation counts.

Optional model artifacts are stored separately and include completed-story
assignments, prototypes, a review catalog, prefix prototypes, temporal-holdout
assignments, and evaluation reports.

## 9.1 Map trajectory contract

The map’s “trajectory” is an exact footprint history, not a traveled path.
Country view renders every complete weekly incident footprint. Selecting one
story shows prior exact grid-union outlines with causal age bands plus a weekly
arc for lifecycle, crop stage, pressure, impact, unresolved fields, and area.
No centroid line, arrow, hull, or animation may imply movement or propagation.
Field polygons appear only at high zoom as evidence drill-down.

## 9.2 Monitoring/append contract

New weekly data creates a new immutable source generation and a new V3 release.
Ordinary append validation preserves all published component/exposure/incident
IDs, weekly evidence and lifecycle, crop-stage denominator summaries,
memberships, lineage, and terminal windows.
The only lifecycle exception is a prior maximum-week row explicitly marked
`data_censored_at_boundary`: only `incident_state`, `current_state`,
`closed_week`, `right_censored`, and the boundary flag may reopen. Onset,
confirmation, pressure/recovery milestones, lineage targets, and cumulative
counters stay frozen, and the reopened state must equal the immediately
preceding published nonterminal state. A prior right-censored window—or its matching terminal
`CLOSED_DATA_CENSORED` boundary window—may extend only from later weekly rows,
while its immutable pre-boundary evidence remains fixed.
The browser does not mutate or recluster the active release in place; promotion
switches to a fully built viewer directory.

## 10. Required comparisons and gates

Baselines:

- B0: independent weekly district/sector hazard brief;
- B1: independent weekly significant spatial components;
- B2: weekly proximity/weather clustering without temporal identity;
- B3: field episodes without incident tracking.

Tracking claims require reviewed links. Proposed pilot gates are:

- same-incident link precision at least 0.85;
- link recall at least 0.75;
- false merges at most 10%;
- at least 30% fewer fragments and +10 points identity/link F1 versus B2;
- at least 95% historical ID retention after an ordinary weekly append;
- expert same-narration agreement at least 80% and Cohen’s kappa at least 0.70.

Impact claims additionally require onset/closure F1 at least 0.80 within one
weekly bucket, median episode temporal IoU at least 0.70, recovery precision at
least 0.85, and measurable scouting value at a fixed review budget.

Archetype gates remain separate: holdout acceptance at least 65%, novelty at
most 35%, prototype non-overlap at least 90%, stable matched-cluster Jaccard at
least 0.70, expert same-narration at least 80%, and semantic false merges at
most 10%.

Archetype failure removes the tags; it must not invalidate a separately
validated incident tracker.

## 11. Safe release language

Before reviewed gates pass, call V3:

> A causal crop-impact incident tracking and diagnostic story-archetype
> framework with uncalibrated starter thresholds.

Immediately define “causal” as **as-of/no-future-leakage**, not causal
attribution.

Never claim validated crop death, causal propagation, real-time streaming, or
production story taxonomy from engineering tests alone.
