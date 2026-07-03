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
