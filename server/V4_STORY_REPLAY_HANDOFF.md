# V4 story replay handoff — 2026-07-07

## Current stopping point

Do **not** resume the failed V4 viewer export and do **not** run the outer
pipeline continuation. The evidence build succeeded and is durable; viewer
publication correctly failed because the old V3 story spine contains response
claims that V4 cannot support.

VM paths:

```text
ROOT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1
SOURCE_GENERATION=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/generations/2026-05-17_generation_7a715df05da10c3b3300
V4_JOB=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/jobs/incident_v4_20260707T132154Z
V4_EVIDENCE=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/releases/incident_evidence_v4_20260707T132154Z
OLD_V3_INCIDENT=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/releases/incidents_v3_20260707T132154Z
DIAGNOSTIC_JSON=/mnt/KSA-Oasis/fields_health_v2/clusters/runs/weekly_monitor_v1/jobs/incident_v4_20260707T132154Z/lifecycle_reconciliation_diagnostic.json
```

The V4 job remains status `1` only because stage 6 failed. Stage 5 completed in
1,335 seconds. Preserve the completed evidence directory and the captured
`export.stderr.failed-20260707T143138Z.log`.

Repository state at handoff:

- Implementation baseline before this handoff commit: `646f68f`
  (`Print compact V4 diagnostic summary`).
- Confirmed but intentionally uncommitted fixes are present in:
  - `server/story_monitor/incident_story_states_v3.py`
  - `server/story_monitor/incident_viewer_v4.py`
  - `server/test_incident_story_states_v3.py`
  - `server/test_incident_viewer_v4.py`
- Those fixes merge follow-up evidence into canonical memberships and compare
  medium/severe as one decline family. The full Python suite passed (325 tests),
  but these fixes alone cannot make the old story spine publishable.

## Evidence from the real VM

The diagnostic classified 19,139 V3 fresh-decline field claims:

| Classification | Field claims | Meaning |
|---|---:|---|
| Exact V3/V4 response | 17,462 | Supported |
| Same decline family, different severity | 32 | Binary decline is supported; exact severity differs |
| Rejected by V4 QA | 1,214 | Unsupported/unknown, not “no decline” |
| Usable acquisition without a new decline | 411 | Unsupported V3 fresh-decline claim |
| Acquisition without decline | 20 | Unsupported V3 fresh-decline claim |

There are also 16 V3 weekly-versus-membership count mismatches. V4 acquisition
reconciliation is:

```text
candidate acquisitions: 3,020,513
published acquisitions: 3,011,420
excluded before any causal crop owner: 9,093
```

The exporter failed on 48 contradictory checkpoints. This was a truth-gate
failure, not an OOM or infrastructure failure.

## Council decision

Build a **V4-native story replay from the existing completed V4 ledgers**. Keep
the old V3 release only as an audit/comparison input. Do not rebuild the
22-minute evidence stage.

Reusable authoritative inputs:

- `crop_day_context_v4.parquet`
- `field_day_pressure_v4.parquet`
- `field_s2_acquisition_v4.parquet`
- validated V4 evidence/source manifests
- field geometry/admin data
- causal crop-instance identities

Recompute, in order:

1. Field/crop/hazard daily episodes from V4 pressure and acquisition evidence.
2. Frozen stage-aware baseline.
3. Weekly significant cells.
4. Spatial components.
5. Persistent exposures and lineage.
6. Crop-specific incidents.
7. Lifecycle, memberships, counters, windows, and knowledge times.
8. Viewer bundle and trajectories.
9. Motif features/models after the new story trajectories are accepted.

Do not force old incident IDs when components change. Produce an explicit
old-to-new overlap/lineage crosswalk.

## Rejected shortcuts

- Do not filter contradictions only in the viewer.
- Do not relax the fail-closed reconciliation gate.
- Do not resume the old exporter after only the medium/severe matching fix.
- Do not rebuild V3 from the same unsupported response inputs.
- Do not label rejected, incomparable, or missing S2 as stable or recovered.
- Do not reuse motif models trained on the old trajectories.

Unsupported evidence is **unknown**, not evidence of no decline. Weather may
still sustain a story. Recovery requires a later usable recovery acquisition.

## Required acceptance gates

- Rejected/unknown-QA/carried/insufficient-reference S2 never creates fresh
  decline or recovery.
- Crop ownership is evaluated at the S2 source date; no future crop assignment.
- Exact V4 severity drives severity-specific transitions.
- Missing weather never advances pressure-off or quiet clocks.
- One canonical membership exists per story/week/field.
- Every weekly role/response counter equals distinct membership evidence.
- Every positive response claim is supported by V4 evidence known by the
  checkpoint time.
- Batch replay equals prefix replay at historical cutoffs.
- Input order does not change components, IDs, states, or counters.
- Final viewer reconciliation reports zero contradictions and zero count
  mismatches.

## First task when resuming work

Implement an immutable `V4-native story replay` runner that accepts
`V4_EVIDENCE`, geometry, and the old V3 release only for crosswalk/audit. It
must checkpoint each stage, reuse the existing evidence, emit a new story
release plus old-to-new crosswalk, and then export a new viewer. Add a VM wrapper
command for this path; do not mutate the failed V4 job or either immutable input.
