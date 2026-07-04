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

## Current product strategy

Incident V3 supersedes learned archetypes as map story identity:

1. Build causal/as-of weekly components from exact significant cells and crop
   denominators.
2. Link them into crop-independent exposures and stable crop-specific
   incidents, with stage retained only as changing context.
3. Export a precomputed `incident_viewer_v3_*` bundle and serve exact weekly
   footprints; no clustering runs in the request path.
4. Gate every append against the prior immutable analytics release, then gate
   publication on latency, visual truthfulness, and data-quality checks.

DuckDB performs the production build. RAPIDS remains optional for offline
completed-story archetype experiments and does not improve HTTP or browser
rendering latency.
