# Current snapshot — 2026-07-04

- Branch/baseline: `main` at `e8b17dd` before the uncommitted Incident V3
  release implementation.
- Product authority: stable crop-impact incidents, with hierarchy
  `component_id → exposure_id → incident_id`; crop stage is dynamic context and
  has zero identity weight.
- Story lifecycle: prefix-safe/as-of onset, confirmation, active/quiet,
  recovery evidence, relapse, unresolved response, recurrence, split/merge,
  and explicit data/season censoring. No crop-death or causal-effect claim.
- Monitoring publication: first release is explicit; later releases validate
  immutable historical components, exposure/incident identity, evidence,
  crop-stage denominators, memberships, lineage, and terminal history.
- Viewer: complete exact weekly incident footprints at country zoom, high-zoom
  field evidence, continuous selected-story arc, and selected-only age-banded
  exact footprint history. No centroid movement trails.
- Performance: precomputed viewer artifacts, bounded DuckDB request admission,
  gzip, one combined server cache byte budget, a byte-bounded browser footprint
  cache, adjacent-week prefetch, and a VM benchmark for cold/warm/random scrub.
- Operations: `server/run_incident_v3.py` performs tests, immutable build,
  viewer export, smoke validation, durable status/resume, and append gating.
  `server/INCIDENT_V3_VM_RUNBOOK.md` contains exact VM commands and verifies the
  served viewer bundle ID before acceptance.
- Local verification: 220/220 Python tests and 31/31 Node UI tests pass; all
  static JavaScript and Incident V3 Python compile checks pass; documentation
  Bash syntax and `git diff --check` pass.
- VM work remaining: pull the committed release, run the first V3 job against
  the existing immutable echo-aware generation, inspect stage coverage/story
  volumes, launch the emitted `incident_viewer_v3_*`, run the saved latency
  benchmark, and complete the browser visual/performance acceptance trace.
