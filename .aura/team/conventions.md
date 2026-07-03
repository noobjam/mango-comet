# Team conventions

- Preserve V1 commands and artifacts; all V2 entry points and schemas are explicitly versioned.
- Features are causal and event-level. Every source column has a documented semantic role.
- Assignment is immutable. Unknown evidence is explicit (`motif_pending`, `watch_only`, `insufficient_evidence`, or `novel_unassigned`).
- Discovery is stratified by hazard; do not compare distances across hazards.
- Deterministic sorting, seeds, hashes, and model manifests are mandatory.
- Fail closed on data leakage, duplicate event anchors, mixed model versions, or non-finite model artifacts.
- Keep generated VM data out of Git. Document commands with absolute VM paths.
- Add focused tests for each behavior change, then run the full Python and JavaScript suites.
