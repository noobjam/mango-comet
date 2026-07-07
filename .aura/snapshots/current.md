# Current snapshot — 2026-07-07

## Active handoff

- Status: paused after real-VM V4 truth-gate failure.
- Resume from: `server/V4_STORY_REPLAY_HANDOFF.md`.
- Decision: build a V4-native story replay from the completed evidence ledgers;
  do not resume the old viewer or rebuild evidence.
- Implementation baseline: `646f68f`; four confirmed V3/V4 reconciliation fix
  files remain intentionally uncommitted and are listed in the handoff.

- Branch/baseline: `main` at `7e06b80`; V4/2 reconciliation, live motif scoring,
  and trajectory UX are locally verified and ready to commit.
- Story authority: deterministic V3
  `component_id → exposure_id → exposure × crop incident_id`. Learned motifs
  describe eligible completed stories and never create, merge, split, start,
  close, or rename them.
- Clocks: daily hazard pressure, irregular acquisition-grain Sentinel-2 crop
  evidence, full-timestamp story knowledge, a daily map playhead, and a
  monotonic UTC `released_at` watermark are distinct. Multiple releases can
  share one playhead date; late corrections fail closed until a revision and
  supersession contract exists.
- Crop monitoring: crop instance and controlled stage remain visible context
  with zero incident/motif distance weight. Rejected and unknown-QA S2 attempts
  cannot become references; echoes age evidence and never create an update.
- Viewer: complete accounted country density, simultaneous daily hazard bands,
  step-held acquisition evidence, knowledge-gated weekly checkpoints, compact
  state plus cached static geometry, exact age-faded footprint history, and an
  aligned field/incident trajectory with explicit missing versus low pressure,
  source-to-known S2 timing/freshness, crop stage, and one row per story. No
  centroid movement or propagation.
- Learning: completed-story HDBSCAN discovery is crop×hazard bounded; causal
  feature extraction is on-disk DuckDB, prefix fitting is maturity-stratum
  bounded, and holdout replay is record-batched. Review, exhaustive
  calibration, sealed holdout labels, open-set assignment, and artifact hashes
  are mandatory. Immutable live deltas score confirmed non-terminal stories,
  with explicit readiness and causal daily-weather ownership; map publication
  remains false.
- Integrity/performance: evidence and viewer inventories are SHA-256/size/row
  bound; server and browser caches are byte-bounded; rapid scrubbing uses
  compact state, static geometry hydration, adjacent-day prefetch, and explicit
  cold/warm/concurrent VM gates.
- Local verification: 284 Python tests and 44 Node UI tests pass; all server
  Python/static JavaScript compiles and `git diff --check` passes.
- VM work remaining: pull the committed release, run the documented first V4
  job against the immutable generation, validate real counts and clocks, launch
  the emitted V4/2 viewer, run the latency benchmark, and complete browser
  visual/performance acceptance. The reviewed prefix model must be rebuilt for
  the joint novelty threshold before live scoring. Motif discovery must then be
  measured on the 3.1M-event release and reviewed; current historical
  availability is diagnostic reconstructed data, not strict operational replay.
