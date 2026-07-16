# Deterministic field stories V1

Status: executable foundation, not yet a viewer or promoted release contract.
The starter thresholds are uncalibrated and require agronomist validation.

## Definition

A field story is one fixed field/crop instance's maximal open-concern interval.
It composes every supported hazard lane in causal decision order:

```text
V4 crop context + daily hazard pressure + usable crop-response evidence
  -> one field/crop story
  -> deterministic lifecycle states and chapters
```

Hazard changes create chapters inside the same story. Crop-stage changes also
remain inside the story. A later hazard starts a new story only after the prior
concern has closed. Story identity is therefore independent of the current
hazard set, stage, lifecycle state, and chapter.

This layer deliberately does not learn motifs, clusters, prototypes, or prefix
models. It does not replace V4's separate same-hazard regional exposure replay.

## Evidence contract

`build_field_stories` consumes the three V4-shaped evidence ledgers:

- `crop_day_context_v4.parquet` for crop ownership and stage;
- `field_day_pressure_v4.parquet` for one observable/missing pressure lane per
  field, crop instance, hazard, and source day;
- `field_s2_acquisition_v4.parquet` for fresh, usable decline or recovery
  observations.

Evidence is applied on the first calendar day it could have been known:

```text
decision_date = max(effective source date, knowledge date)
```

A late observation retains its effective source date but cannot rewrite an
earlier daily state. Missing pressure is not low pressure. It freezes quiet and
unresolved-response clocks, then closes as data-censored at the policy limit.

Remote crop response is story-level supporting evidence. A decline with no
currently elevated pressure is retained as `unattributed_decline` and requires
review; it is not forced into one hazard lane.

## Lifecycle

- `CANDIDATE`: first elevated pressure or fresh adverse crop response.
- `ACTIVE`: persistent pressure or aligned pressure plus fresh decline.
- `SEVERE`: starter severe-pressure override, or severe decline aligned with
  currently elevated pressure.
- `QUIET_PENDING`: no current elevation while at least one hazard thread owns
  an observed quiet grace clock.
- `RECOVERING`: pressure threads closed but an attributed decline remains
  unresolved.
- `DATA_GAP`: required follow-up is not observable; clocks freeze.
- `CLOSED_CANDIDATE_EXPIRED`: initial evidence did not confirm.
- `CLOSED_PRESSURE_QUIET_NO_RESPONSE`: confirmed pressure closed without an
  observed adverse response.
- `CLOSED_RECOVERED`: a later usable acquisition supports recovery after a
  decline.
- `CLOSED_RESPONSE_UNRESOLVED`: observed follow-up expired without recovery.
- `CLOSED_DATA_CENSORED`: missing follow-up exceeded the allowed gap.

`confirmed_time` is separate from lifecycle state. This prevents an
unconfirmed candidate in quiet grace from silently acquiring confirmed-story
closure behavior.

## Outputs

`FieldStoryArtifacts` contains four deterministic tables:

| Table | Grain | Purpose |
|---|---|---|
| `daily_state` | story x decision day | Current lifecycle, hazard vector, stage, response, coverage, and clocks |
| `chapters` | consecutive material story state | Run-length encoding of lifecycle/hazard/stage/response changes |
| `windows` | story | Stable identity, complete/right-censored boundary, hazards, and conclusion |
| `hazard_daily` | story x decision day x hazard | Auditable pressure lane and thread state |

The public in-memory API is:

```python
from story_monitor import build_field_stories

artifacts = build_field_stories(crop_days, pressure, acquisitions)
```

The same inputs in any row order produce identical outputs. State at an as-of
day is unchanged when later rows are added.

## Standalone release build

`run_field_stories_v1.py` validates a completed immutable V4 evidence release,
hash-partitions all three ledgers once by field/crop ownership, composes each
partition, and atomically publishes four globally sorted Parquet artifacts plus
a hash-bound `manifest.json`. Existing output directories are never replaced.

On the current VM, the gitignored `.env.vm` provides the repository, Python,
evidence, run-root, partition, and DuckDB settings. Pull once to install the
wrapper:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/mango-comet
git pull --ff-only origin main
```

After that, every new release is one command. `run` pulls `origin/main`, restarts
the updated wrapper, performs preflight checks, and launches the build under
`nohup` with durable PID, status, log, and output pointers:

```bash
server/vm_field_stories_v1.sh run
```

Inspect it without reconstructing paths:

```bash
server/vm_field_stories_v1.sh status
server/vm_field_stories_v1.sh logs
```

Pass an explicit environment file after the action only when it is not the
default repository `.env.vm`.

The release is intentionally not a viewer bundle. Do not point
`STORY_MAP_RUN_DIR` at it.

## Spatial boundary

A field story never moves: its field geometry is fixed. These artifacts do not
claim spatial propagation.

Two separate future views may consume them:

1. a prevalence query showing fields that currently satisfy an explicit,
   human-authored story condition; and
2. the existing same-hazard regional exposure lineage showing observed
   appearance, continuation, split, merge, reorganization, or disappearance.

The first is a distribution of field stories, not one shared event. The second
is regional hazard evidence, not crop-story identity.

## Validation

Run the canonical deterministic cases:

```bash
cd server
python -m unittest -v \
  test_field_stories_v1.py \
  test_run_field_stories_v1.py \
  test_vm_field_stories_v1.py
```

The suite covers concurrent and sequential hazards, stage changes, complete
closure followed by a new story, candidate expiry, missingness, unresolved
follow-up, recovery, late-known evidence, row-order invariance, and prefix-safe
as-of replay.

Before map integration, agronomists still need to adjudicate the same/new-story
boundary cases and calibrate the numeric policy in
`story_monitor/policies/field_story_policy_v1.json`.
