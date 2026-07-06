# Current snapshot — 2026-07-06

- Branch/baseline: `main` at `f24bbe9` before the uncommitted Incident V4
  dual-clock implementation.
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
  state plus cached static geometry, exact age-faded footprint history, and a
  field-level pressure/S2/story ribbon. No centroid movement or propagation.
- Learning: completed-story HDBSCAN discovery is crop×hazard bounded; causal
  feature extraction is on-disk DuckDB, prefix fitting is maturity-stratum
  bounded, and holdout replay is record-batched. Review, exhaustive
  calibration, sealed holdout labels, open-set assignment, and artifact hashes
  are mandatory; map publication remains false.
- Integrity/performance: evidence and viewer inventories are SHA-256/size/row
  bound; server and browser caches are byte-bounded; rapid scrubbing uses
  compact state, static geometry hydration, adjacent-day prefetch, and explicit
  cold/warm/concurrent VM gates.
- Local verification: 273 Python tests and 36 Node UI tests pass; all static
  JavaScript and server Python files compile; `git diff --check` passes.
- VM work remaining: pull the committed release, run the documented first V4
  job against the immutable generation, validate real counts and clocks, launch
  the emitted `incident_viewer_v4_*`, run the latency benchmark, and complete
  browser visual/performance acceptance. Motif discovery must then be measured
  on the 3.1M-event release and reviewed; current historical availability is
  diagnostic reconstructed data, not strict operational replay.
