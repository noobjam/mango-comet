# Architecture decisions

## ADR-001: Stable event archetypes use one causal anchor

Status: accepted, 2026-07-03.

Each eligible event contributes exactly one vector. Its anchor is the 21st usable pressure-observed event day, or the closure date when an event closes earlier after at least seven usable days. Open events younger than 21 usable days are `insufficient_evidence`; WATCH-only events are `watch_only`; season-boundary closures before day 21 are excluded.

The vector may use evidence only through its anchor. It excludes terminal outcome, lifecycle state, dates, crop, field, geography, season, and future observations. The assigned archetype is immutable for the life of the event.

Why: V1 clustered multiple evolving prefixes from the same event. On the full VM run, 50.09% of events occupied multiple motifs and 10,901 internal clusters collapsed to only 54 repeated generated labels. That model is useful for diagnostics but is not stable story identity.

Rejected alternatives:

- Latest-prefix clustering: future data can rewrite identity and mixes event maturity with archetype.
- A fixed 28-calendar-day window: it is not observation-aware and biases events with sparse acquisitions.
- Lifecycle labels as archetypes: phase and urgency evolve, while identity must remain stable.

## ADR-002: Discovery and map publication are separate gates

Status: accepted, 2026-07-03.

Phase A produces a versioned V2 model and evaluation report. It does not overwrite V1 artifacts or publish clusters to the UI. Phase B starts only after all hard gates pass and the statistical quality report is reviewed. Human labels and publication status live in an immutable review overlay keyed by model version and archetype ID.

## ADR-003: Operational hierarchy

Status: accepted, 2026-07-03.

The hierarchy is: hazard family → stable event archetype → event → evolving lifecycle/current severity. Crop, stage, geography, and outcome are metadata and facets, not archetype-defining dimensions.

## ADR-004: Stability and separation gates cover the fitted pipeline

Status: accepted, 2026-07-03.

Two deterministic 80% hazard-stratified subsample runs refit their own
hazard-local median/IQR transform before HDBSCAN. Evaluation reports ARI both
with noise and on mutually assigned non-noise events, matched-archetype
Jaccard, and two-run support. Prototype separation examines every unordered
same-hazard pair rather than only nearest centers. These are engineering and
statistical gates, not agronomic validation.

## ADR-005: Phase A runs through one fail-closed orchestrator

Status: accepted, 2026-07-03.

The preferred VM interface is `server/run_archetype_v2.py`. It performs
focused tests, RAPIDS/GPU preflight, immutable build, evaluation, artifact and
lineage verification, and gate-aware exit handling in one job. State and
heartbeats are atomic; subprocesses are shell-free and isolated in a process
group; resume refuses partial, mismatched, or still-running work.

Progress is stage-, time-, PID-, and RSS-based. A percentage bar is rejected
until the build/evaluation internals expose a real work denominator. Phase A
success remains diagnostic and does not create a map publication pointer.

## ADR-006: V3 story identity is crop-impact incident tracking

Status: accepted, 2026-07-03. Supersedes ADR-003 for the V3 product and
supersedes ADR-001 only for story identity; V2 remains immutable provenance.

The primary story is a local same-hazard exposure tracked through weekly
components, viewed separately for each affected crop. `incident_id` is stable
through crop-stage changes. `archetype_id` is an optional provisional tag
learned from completed stories and never controls incident identity.

Why: V2 found coherent training cores but failed temporal holdout coverage and
prototype separation. More importantly, one 21-day field-event anchor cannot
represent a local evolving crop incident. V3 keeps the causal field state
machine, adds stage-aware monitored denominators and deterministic spatial-
temporal lineage, then clusters completed stories.

Rejected alternatives:

- Treat V2 archetype cohorts as stories: globally scattered members and no
  local incident identity.
- Cluster independent weekly components: repeats the memoryless baseline.
- Put crop stage in the story ID: fragments one crop story as phenology
  advances.
- Infer crop death from satellite response: unsupported without independent
  outcome evidence.

## ADR-007: Story onset and closure are causal lifecycle decisions

Status: accepted, 2026-07-04.

A story starts only on a week where an exact local footprint has adequate
monitored/evaluable crop denominators and satisfies persistent confirmation or
the explicitly hashed severe-override policy. WATCH evidence and low-coverage
preludes do not establish a story. Quiet, recovery, unresolved response,
recurrence, split/merge, and data-censored endings are separate operational
states. Low coverage freezes clocks, and bounded follow-up scaffolds prevent
open incidents from expanding across all future monitoring weeks.

Why: monitoring must distinguish “we saw damage evidence,” “we stopped seeing
pressure,” “we saw recovery evidence,” and “we lost observability.” Collapsing
those into alive/dead or open/closed would overstate the satellite evidence.

## ADR-008: Map trajectories are exact footprint histories

Status: accepted, 2026-07-04.

The viewer renders the exact union of significant fixed-grid rectangles for
each incident/week. The selected incident may display prior weekly polygons in
age bands beside its crop-stage/lifecycle arc. Centroid lines, arrows,
interpolated hulls, and animations that imply movement or propagation are
rejected.

Countrywide incident footprints are precomputed into the viewer release. The
server and browser use byte-bounded caches; field polygons are a high-zoom
detail layer, while the complete incident footprint layer remains visible at
national zoom. VM latency and browser visual gates are release requirements.

## ADR-009: Monitoring publication is immutable and append-only

Status: accepted, 2026-07-04.

Every production build declares exactly one mode: `--first-release`, or
`--previous-incident-dir` for an append. Append validation rejects rewritten
historical components, exposure assignments, incident weeks, memberships,
lineage, and terminal windows, including insertion into a prior time range when
the corresponding prior artifact was empty. Stage mapping must pass an overall
coverage floor and a per-supported-crop floor before the release is accepted.

The analytics `incidents_v3_*` directory is the append/audit source. Only its
derived `incident_viewer_v3_*` directory may be used as `STORY_MAP_RUN_DIR`.
The server performs no clustering or lifecycle reconstruction on request.

## ADR-010: V4 separates pressure, crop evidence, and story knowledge clocks

Status: accepted, 2026-07-06.

The V3 `component_id -> exposure_id -> incident_id` hierarchy remains story
identity. V4 adds independent daily hazard-pressure and acquisition-grain
Sentinel-2 ledgers plus a daily as-of viewer projection. Incident checkpoints
are visible only after their recorded knowledge time; an incident ID is never
back-projected into a daily prelude.

Repeated spectral echoes only age the last acquisition. Missing weather cannot
advance a quiet clock, rejected/cloudy acquisitions cannot become spectral
references, and absent usable post-pressure observations produce censoring
rather than recovery, unresolved damage, or crop-death claims.

Historical data without retained ingest timestamps uses an explicitly
diagnostic reconstructed-availability mode. Operational replay requires source
availability timestamps. Learned motifs are reviewed, completed-story
descriptors with weather- and acquisition-matched prefix prototypes; they never
control incident identity or lifecycle.
