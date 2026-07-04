# Team conventions

- Preserve V1/V2 commands and artifacts as provenance; all V3 entry points,
  policies, outputs, and schemas are explicitly versioned.
- "Causal" means prefix-safe/as-of, never causal inference. Every source
  column has a documented semantic role and knowledge time.
- Incident identity is deterministic and immutable. Crop stage, lifecycle,
  severity, and optional archetype tags never rewrite `incident_id`.
- Discovery is optional, completed-story-only, and stratified by hazard; do
  not compare distances across hazards or use archetypes as story IDs.
- Deterministic sorting, seeds, hashes, and model manifests are mandatory.
- Fail closed on future leakage, historical append rewrites, duplicate weekly
  identities, mixed policy versions, or non-finite model artifacts.
- Keep generated VM data out of Git. Document commands with absolute VM paths.
- Draw exact weekly footprint unions only; centroid trails may not imply motion.
- Add focused tests for each behavior change, then run the full Python and JavaScript suites.
