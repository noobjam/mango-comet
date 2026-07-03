# Current snapshot — 2026-07-03

- Main branch baseline: `4a1371e32297c1f7e52e011cf4bccc46f8ebf48f`
- V1 full generation: 39,695,363 daily rows and 3,131,245 events.
- V1 GPU discovery: 4,914,446 prefixes, 10,901 motifs, 17.54% noise.
- V1 verdict: diagnostic baseline only; blocked from map publication because it models prefix-state microvariants rather than immutable event archetypes.
- Phase A implementation: complete locally. It includes causal event anchors,
  hazard-stratified discovery, uniform frozen assignment, temporal holdout,
  deterministic two-run subsample-refit stability, all-pairs prototype overlap,
  immutable artifacts, hashes, reconciliation gates, tests, and the VM runbook.
- Local verification: 85 Python tests and 11 browser-logic tests passed before
  final handoff; GPU/full-scale validation remains VM-only.
- Deferred until every real-data V2 gate passes: map hierarchy, scalable
  exporter, API/UI changes, trajectories, and timeline benchmark/promotion.
- Durable V2 runner: implemented locally with run/resume/status, RAPIDS/GPU
  preflight, exact artifact and lineage verification, atomic state and
  heartbeats, orphan-child protection, gate-aware exit codes, and persistent
  launcher logs. No `tqdm` percentage is shown because the stages expose no
  valid work denominator.
- Current verification: 100 Python tests and 11 browser-logic tests pass;
  compilation, diff checks, and documentation Bash syntax pass. Independent
  runner and documentation reviews pass. VM execution remains.
