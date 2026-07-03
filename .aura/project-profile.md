# Aura project profile

- ConfigHash: `6829e8fe3df3db561237234733b68dc9b8ce7c37`
- Recorded: 2026-07-03
- Product: field-health story monitoring and map explorer
- Backend: Python 3, standard-library HTTP server, DuckDB, Parquet
- Offline pipeline: Python CLI under `server/weekly_story_monitor.py`
- Frontend: browser-native JavaScript, MapLibre GL, deck.gl, static CSS
- Package management: pip requirements; no JavaScript package manager or bundler
- Tests: Python `unittest`, Node `node:test`, Python compile checks
- Runtime data: generated on the GPU VM; source Parquet and generated artifacts are not committed

## Change strategy

The V2 archetype work is deliberately split into two gates:

1. Build and validate a causal one-anchor-per-event archetype model without changing the V1 pipeline or map contract.
2. Only after the VM evaluation gates pass, export the reviewed hierarchy and precomputed summaries used by the map.

No new runtime dependency is required for Phase A. GPU discovery continues to use the optional RAPIDS environment.
