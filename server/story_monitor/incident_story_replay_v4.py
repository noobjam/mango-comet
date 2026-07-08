"""Immutable, checkpointed V4-native crop-incident story replay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, Iterator

import duckdb
import pandas as pd

from .contracts import stable_id
from .incident_archetypes_v3 import extract_completed_incident_features
from .incident_cells_v3 import (
    build_component_field_rows,
    build_stage_baseline,
    build_weekly_exposure_cells,
)
from .incident_context_v3 import build_incident_context_v3
from .incident_crosswalk_v4 import build_incident_crosswalk_v4
from .incident_denominators_v3 import (
    build_incident_stage_summary,
    enrich_incident_weekly_state,
)
from .incident_exposures_v3 import track_exposures
from .incident_lineage_v3 import (
    build_incident_lineage_v3,
    remap_incident_lineage_segments,
)
from .incident_policy_v3 import IncidentPolicyV3, load_incident_policy_v3
from .incident_policy_v4 import IncidentPolicyV4, load_incident_policy_v4
from .incident_replay_context_v4 import replay_daily_episodes_v4
from .incident_story_states_v3 import (
    CropStoryScaffold,
    build_crop_story_scaffold,
    build_incident_followup_evidence,
    finalize_crop_story_artifacts,
)
from .incident_tracking_v3 import build_weekly_components
from .incident_validation_v3 import (
    FINAL_ARTIFACT_FILES,
    artifact_hashes,
    file_sha256,
    validate_final_artifact_directory,
)
from .incident_validation_v4 import validate_evidence_directory


SCHEMA_VERSION = "crop-impact-incident-story-replay-v4/1"
CHECKPOINT_SCHEMA_VERSION = "incident-story-replay-checkpoint-v4/2"
CONTEXT_SCHEMA_VERSION = "incident-story-replay-context-v4/1"
SOURCE_ADAPTER_SCHEMA_VERSION = "incident-story-replay-source-adapter-v4/1"
MODE = "crop_incident_story_replay_v4"
REPLAY_ALGORITHM_REVISION = "v4-native-story-replay-2026-07-07.4"
LIFECYCLE_ALGORITHM_REVISION = (
    "v4-native-lifecycle-routed-recovery-reconciliation-2026-07-08.1"
)
LIFECYCLE_CHECKPOINT_NAME = "08_lifecycle_reconciled"

EVIDENCE_FILES = {
    "crop": "crop_day_context_v4.parquet",
    "pressure": "field_day_pressure_v4.parquet",
    "s2": "field_s2_acquisition_v4.parquet",
}


def build_incident_story_replay_v4(
    evidence_dir: Path,
    geometry_parquet: Path,
    audit_incident_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    baseline_through: str,
    source_policy: IncidentPolicyV4 | None = None,
    tracker_policy: IncidentPolicyV3 | None = None,
    threads: int = 16,
    replay_partitions: int = 64,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Build one new immutable story release without mutating any input."""
    evidence_dir = evidence_dir.expanduser().resolve()
    geometry_parquet = geometry_parquet.expanduser().resolve()
    audit_incident_dir = audit_incident_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    source_policy = source_policy or load_incident_policy_v4()
    tracker_policy = tracker_policy or load_incident_policy_v3()
    tracker_policy_source_sha256 = tracker_policy.source_sha256
    tracker_policy = replace(
        tracker_policy,
        source_sha256=_effective_policy_sha256(tracker_policy),
    )
    baseline_date = pd.Timestamp(baseline_through).normalize()
    if isinstance(replay_partitions, bool) or not 1 <= int(replay_partitions) <= 1024:
        raise ValueError("replay_partitions must be between 1 and 1024")
    _validate_build_inputs(
        evidence_dir,
        geometry_parquet,
        audit_incident_dir,
        output_dir,
        checkpoint_dir,
        baseline_date,
    )
    evidence_validation = validate_evidence_directory(evidence_dir)
    evidence_manifest = _read_json(evidence_dir / "manifest.json")
    release_as_of = pd.Timestamp(
        (evidence_manifest.get("run") or {}).get("release_as_of")
        or (evidence_manifest.get("run") or {}).get("as_of_date")
    ).normalize()
    if baseline_date >= release_as_of:
        raise ValueError("baseline_through must precede the V4 evidence release boundary")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    build_inputs = {
        "replay_algorithm_revision": REPLAY_ALGORITHM_REVISION,
        "evidence_manifest": _fingerprint(evidence_dir / "manifest.json"),
        "geometry": _fingerprint(geometry_parquet),
        "audit_incident_manifest": _fingerprint(audit_incident_dir / "manifest.json"),
        "audit_incident_membership": _fingerprint(
            audit_incident_dir / "incident_membership.parquet"
        ),
        "source_policy": _fingerprint(source_policy.source_path),
        "tracker_policy": _fingerprint(tracker_policy.source_path),
        "source_policy_effective_sha256": _effective_policy_sha256(source_policy),
        "tracker_policy_effective_sha256": tracker_policy.source_sha256,
        "tracker_policy_source_sha256": tracker_policy_source_sha256,
        "baseline_through": baseline_date.date().isoformat(),
        "replay_partitions": int(replay_partitions),
    }

    context = _checkpoint(
        checkpoint_dir / "01_context",
        stage_name="context",
        inputs=build_inputs,
        builder=lambda stage: _build_context_checkpoint(
            evidence_dir,
            geometry_parquet,
            stage,
            evidence_manifest=evidence_manifest,
            source_policy=source_policy,
            tracker_policy=tracker_policy,
            threads=threads,
            replay_partitions=int(replay_partitions),
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )
    context_inputs = {**build_inputs, "context_checkpoint": _fingerprint(context / "manifest.json")}
    baseline = _checkpoint(
        checkpoint_dir / "02_baseline",
        stage_name="baseline",
        inputs=context_inputs,
        builder=lambda stage: _build_baseline_checkpoint(
            context,
            stage,
            baseline_through=baseline_date.date().isoformat(),
            tracker_policy=tracker_policy,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )
    baseline_inputs = {**context_inputs, "baseline_checkpoint": _fingerprint(baseline / "manifest.json")}
    cells = _checkpoint(
        checkpoint_dir / "03_cells",
        stage_name="cells",
        inputs=baseline_inputs,
        builder=lambda stage: _build_cells_checkpoint(
            context,
            baseline,
            stage,
            assignment_after=baseline_date.date().isoformat(),
            assignment_through=release_as_of.date().isoformat(),
            tracker_policy=tracker_policy,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )
    cells_inputs = {**baseline_inputs, "cells_checkpoint": _fingerprint(cells / "manifest.json")}
    components = _checkpoint(
        checkpoint_dir / "04_components",
        stage_name="components",
        inputs=cells_inputs,
        builder=lambda stage: _build_components_checkpoint(
            context,
            cells,
            stage,
            tracker_policy=tracker_policy,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )
    component_inputs = {**cells_inputs, "components_checkpoint": _fingerprint(components / "manifest.json")}
    exposures = _checkpoint(
        checkpoint_dir / "05_exposures",
        stage_name="exposures",
        inputs=component_inputs,
        builder=lambda stage: _build_exposures_checkpoint(
            components, stage, tracker_policy=tracker_policy
        ),
    )
    exposure_inputs = {**component_inputs, "exposures_checkpoint": _fingerprint(exposures / "manifest.json")}
    scaffold = _checkpoint(
        checkpoint_dir / "06_scaffold",
        stage_name="scaffold",
        inputs=exposure_inputs,
        builder=lambda stage: _build_scaffold_checkpoint(
            context,
            cells,
            components,
            exposures,
            stage,
            tracker_policy=tracker_policy,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )
    scaffold_inputs = {**exposure_inputs, "scaffold_checkpoint": _fingerprint(scaffold / "manifest.json")}
    lifecycle_inputs = {
        **scaffold_inputs,
        "lifecycle_algorithm_revision": LIFECYCLE_ALGORITHM_REVISION,
    }
    lifecycle = _checkpoint(
        checkpoint_dir / LIFECYCLE_CHECKPOINT_NAME,
        stage_name="lifecycle_reconciled",
        inputs=lifecycle_inputs,
        builder=lambda stage: _build_lifecycle_checkpoint(
            context,
            cells,
            scaffold,
            stage,
            tracker_policy=tracker_policy,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        ),
    )

    _publish_release(
        output_dir,
        context=context,
        baseline=baseline,
        cells=cells,
        components=components,
        exposures=exposures,
        lifecycle=lifecycle,
        audit_incident_dir=audit_incident_dir,
        evidence_dir=evidence_dir,
        geometry_parquet=geometry_parquet,
        checkpoint_dir=checkpoint_dir,
        baseline_through=baseline_date.date().isoformat(),
        release_as_of=release_as_of.date().isoformat(),
        evidence_manifest=evidence_manifest,
        evidence_validation=evidence_validation,
        source_policy=source_policy,
        tracker_policy=tracker_policy,
        active_checkpoints={
            "01_context": context,
            "02_baseline": baseline,
            "03_cells": cells,
            "04_components": components,
            "05_exposures": exposures,
            "06_scaffold": scaffold,
            LIFECYCLE_CHECKPOINT_NAME: lifecycle,
        },
    )
    manifest = _read_json(output_dir / "manifest.json")
    return {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "source_adapter_dir": str(context / "source_generation"),
        "generation_id": (manifest.get("run") or {}).get("generation_id"),
        "row_counts": (manifest.get("validation") or {}).get("row_counts"),
        "crosswalk_rows": (manifest.get("validation") or {}).get("crosswalk_rows"),
    }


def _build_context_checkpoint(
    evidence_dir: Path,
    geometry_parquet: Path,
    stage: Path,
    *,
    evidence_manifest: dict[str, Any],
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    replay_partitions: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    source = stage / "source_generation"
    source.mkdir()
    diagnostics = _materialize_partitioned_replay(
        evidence_dir,
        stage,
        source,
        evidence_manifest=evidence_manifest,
        source_policy=source_policy,
        tracker_policy=tracker_policy,
        replay_partitions=replay_partitions,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    shutil.copy2(geometry_parquet, source / "map_field_geometry.parquet")

    source_manifest = _source_adapter_manifest(
        source,
        evidence_dir=evidence_dir,
        evidence_manifest=evidence_manifest,
        geometry_parquet=geometry_parquet,
        source_policy=source_policy,
        tracker_policy=tracker_policy,
        diagnostics=diagnostics,
    )
    (source / "manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_context = stage / "context-raw"
    build_incident_context_v3(
        source,
        raw_context,
        policy=tracker_policy,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    context = stage / "context"
    context.mkdir()
    shutil.copy2(
        raw_context / "field_week_context.parquet",
        context / "field_week_context.parquet",
    )
    _write_context_lanes_with_knowledge(
        raw_context / "event_week_lanes.parquet",
        source / "event_state_snapshots.parquet",
        context / "event_week_lanes.parquet",
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    lane_count = _parquet_count(context / "event_week_lanes.parquet")
    context_manifest = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "run": {"status": "complete", "immutable": True},
        "source": {
            "evidence_manifest_sha256": file_sha256(evidence_dir / "manifest.json"),
            "source_adapter_manifest_sha256": file_sha256(source / "manifest.json"),
        },
        "policies": _policy_manifest(source_policy, tracker_policy),
        "counts": diagnostics,
        "artifacts": artifact_hashes(
            context, ["field_week_context.parquet", "event_week_lanes.parquet"]
        ),
    }
    (context / "manifest.json").write_text(
        json.dumps(context_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(raw_context)
    return {
        **diagnostics,
        "field_week_context_count": _parquet_count(
            context / "field_week_context.parquet"
        ),
        "event_week_lane_count": lane_count,
    }


def _materialize_partitioned_replay(
    evidence_dir: Path,
    stage: Path,
    source: Path,
    *,
    evidence_manifest: dict[str, Any],
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
    replay_partitions: int,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, int]:
    partition_root = stage / ".replay-partitions"
    input_root = partition_root / "input"
    output_root = partition_root / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir()
    _partition_evidence(
        evidence_dir,
        input_root,
        replay_partitions=replay_partitions,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    output_names = (
        "daily_episode_state",
        "episode_windows",
        "episode_membership",
        "daily_signals",
        "event_state_snapshots",
    )
    for name in output_names:
        (output_root / name).mkdir()
    diagnostic_keys = (
        "daily_episode_state_count",
        "episode_count",
        "episode_membership_count",
        "daily_signal_count",
        "ignored_nonpositive_or_unusable_acquisition_count",
        "ambiguous_decline_attribution_count",
        "ambiguous_recovery_attribution_count",
    )
    diagnostics = {name: 0 for name in diagnostic_keys}
    partition_ids = _replay_partition_ids(input_root)
    if not partition_ids:
        raise ValueError("V4 evidence contains no replay input partitions")
    print(
        f"V4 replay input partitioning complete: {len(partition_ids)} partitions",
        file=sys.stderr,
        flush=True,
    )
    empty_inputs = {
        name: _empty_parquet_frame(evidence_dir / filename)
        for name, filename in EVIDENCE_FILES.items()
    }
    for position, partition_id in enumerate(partition_ids, start=1):
        partition_started = time.monotonic()
        frames: dict[str, pd.DataFrame] = {}
        for name in EVIDENCE_FILES:
            directory = input_root / name / f"replay_partition={partition_id}"
            frames[name] = (
                pd.read_parquet(directory)
                if directory.is_dir()
                else empty_inputs[name].copy()
            )
        print(
            "V4 replay partition "
            f"{position}/{len(partition_ids)} id={partition_id} started "
            f"crop={len(frames['crop'])} pressure={len(frames['pressure'])} "
            f"s2={len(frames['s2'])}",
            file=sys.stderr,
            flush=True,
        )
        replay = replay_daily_episodes_v4(
            frames["crop"],
            frames["pressure"],
            frames["s2"],
            source_policy=source_policy,
            tracker_policy=tracker_policy,
        )
        snapshots = _weekly_snapshots(
            replay.daily_episode_state, evidence_manifest
        )
        part_name = f"part-{partition_id:04d}.parquet"
        replay.daily_episode_state.to_parquet(
            output_root / "daily_episode_state" / part_name,
            index=False,
            compression="zstd",
        )
        replay.episode_windows.to_parquet(
            output_root / "episode_windows" / part_name,
            index=False,
            compression="zstd",
        )
        replay.episode_membership.to_parquet(
            output_root / "episode_membership" / part_name,
            index=False,
            compression="zstd",
        )
        replay.daily_signals.to_parquet(
            output_root / "daily_signals" / part_name,
            index=False,
            compression="zstd",
        )
        snapshots.to_parquet(
            output_root / "event_state_snapshots" / part_name,
            index=False,
            compression="zstd",
        )
        for name in diagnostic_keys:
            diagnostics[name] += int(replay.diagnostics[name])
        print(
            "V4 replay partition "
            f"{position}/{len(partition_ids)} id={partition_id} complete "
            f"elapsed_seconds={time.monotonic() - partition_started:.1f}",
            file=sys.stderr,
            flush=True,
        )

    destinations = {
        "daily_episode_state": stage / "daily_episode_state_v4.parquet",
        "episode_windows": source / "event_windows.parquet",
        "episode_membership": source / "story_day_membership.parquet",
        "daily_signals": source / "daily_causal_signals.parquet",
        "event_state_snapshots": source / "event_state_snapshots.parquet",
    }
    for name, destination in destinations.items():
        _concat_parquet_parts(
            output_root / name,
            destination,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        )
    shutil.rmtree(partition_root)
    return diagnostics


def _replay_partition_ids(input_root: Path) -> list[int]:
    """Return every partition containing any replay ledger.

    Evidence-only partitions must reach the causal ownership checks.  Building
    the worklist from crop rows alone would silently discard orphan pressure or
    S2 evidence in a bucket with no crop context.
    """
    return sorted(
        {
            int(path.name.split("=", 1)[1])
            for name in EVIDENCE_FILES
            for path in (input_root / name).glob("replay_partition=*")
            if path.is_dir()
        }
    )


def _partition_evidence(
    evidence_dir: Path,
    output_root: Path,
    *,
    replay_partitions: int,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> None:
    with _duckdb_connection(threads, memory_limit, temp_dir) as connection:
        for name, filename in EVIDENCE_FILES.items():
            source = evidence_dir / filename
            destination = output_root / name
            connection.execute(
                f"""
                COPY (
                    SELECT *, CAST(
                        HASH(CAST(field_id AS VARCHAR)) % {int(replay_partitions)}
                        AS INTEGER
                    ) AS replay_partition
                    FROM read_parquet({_sql_literal(source)})
                ) TO {_sql_literal(destination)} (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    PARTITION_BY (replay_partition)
                )
                """
            )


def _concat_parquet_parts(
    source_dir: Path,
    destination: Path,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> None:
    glob = source_dir / "*.parquet"
    with _duckdb_connection(threads, memory_limit, temp_dir) as connection:
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet(
                    {_sql_literal(glob)}, union_by_name=TRUE
                )
            ) TO {_sql_literal(destination)} (
                FORMAT PARQUET, COMPRESSION ZSTD
            )
            """
        )


def _write_context_lanes_with_knowledge(
    raw_lanes: Path,
    snapshots: Path,
    destination: Path,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> None:
    with _duckdb_connection(threads, memory_limit, temp_dir) as connection:
        lane_columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(raw_lanes)})"
            ).fetchall()
        }
        select_lanes = (
            "l.* EXCLUDE (knowledge_time)"
            if "knowledge_time" in lane_columns else "l.*"
        )
        connection.execute(
            f"""
            COPY (
                SELECT {select_lanes}, s.knowledge_time
                FROM read_parquet({_sql_literal(raw_lanes)}) AS l
                JOIN read_parquet({_sql_literal(snapshots)}) AS s
                  ON CAST(l.timeline_bucket AS DATE)
                    = CAST(s.timeline_bucket AS DATE)
                 AND CAST(l.event_id AS VARCHAR)
                    = CAST(s.event_id AS VARCHAR)
                QUALIFY COUNT(*) OVER (
                    PARTITION BY l.timeline_bucket, l.event_id
                ) = 1
                ORDER BY l.timeline_bucket, l.field_id,
                    l.hazard_family, l.field_hazard_lane_rank, l.event_id
            ) TO {_sql_literal(destination)} (
                FORMAT PARQUET, COMPRESSION ZSTD
            )
            """
        )
        missing = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet({_sql_literal(destination)})
                WHERE knowledge_time IS NULL
                """
            ).fetchone()[0]
        )
        if missing:
            raise ValueError("V4 replay context lost event knowledge timestamps")


def _empty_parquet_frame(path: Path) -> pd.DataFrame:
    with duckdb.connect(":memory:") as connection:
        return connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
        ).fetchdf()


def _build_baseline_checkpoint(
    context: Path,
    stage: Path,
    *,
    baseline_through: str,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    baseline = build_stage_baseline(
        context / "context" / "field_week_context.parquet",
        context / "context" / "event_week_lanes.parquet",
        baseline_through=baseline_through,
        policy=tracker_policy,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    baseline.to_parquet(
        stage / "stage_baseline.parquet", index=False, compression="zstd"
    )
    return {"row_count": len(baseline), "baseline_through": baseline_through}


def _build_cells_checkpoint(
    context: Path,
    baseline: Path,
    stage: Path,
    *,
    assignment_after: str,
    assignment_through: str,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    cells = build_weekly_exposure_cells(
        context / "context" / "field_week_context.parquet",
        context / "context" / "event_week_lanes.parquet",
        baseline / "stage_baseline.parquet",
        policy=tracker_policy,
        assignment_after=assignment_after,
        assignment_through=assignment_through,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    cells.to_parquet(
        stage / "weekly_exposure_cells.parquet", index=False, compression="zstd"
    )
    return {
        "row_count": len(cells),
        "significant_count": int(cells.get("is_significant", pd.Series(dtype=bool)).sum()),
        "assignment_after": assignment_after,
        "assignment_through": assignment_through,
    }


def _build_components_checkpoint(
    context: Path,
    cells: Path,
    stage: Path,
    *,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    cell_frame = pd.read_parquet(cells / "weekly_exposure_cells.parquet")
    field_rows = build_component_field_rows(
        context / "context" / "field_week_context.parquet",
        context / "context" / "event_week_lanes.parquet",
        cell_frame,
        policy=tracker_policy,
        frontier_distance_cells=tracker_policy.frontier_distance_cells,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    result = build_weekly_components(cell_frame, field_rows, tracker_policy)
    _write_frames(
        stage,
        {
            "component_field_rows.parquet": field_rows,
            "weekly_components.parquet": result.components,
            "component_membership.parquet": result.memberships,
        },
    )
    return {
        "component_field_row_count": len(field_rows),
        "component_count": len(result.components),
        "component_membership_count": len(result.memberships),
    }


def _build_exposures_checkpoint(
    components: Path,
    stage: Path,
    *,
    tracker_policy: IncidentPolicyV3,
) -> dict[str, Any]:
    result = track_exposures(
        pd.read_parquet(components / "weekly_components.parquet"),
        pd.read_parquet(components / "component_membership.parquet"),
        tracker_policy,
    )
    _write_frames(
        stage,
        {
            "exposure_component_assignments.parquet": result.assignments,
            "exposure_links.parquet": result.lineage,
            "exposure_weekly_state.parquet": result.weekly_state,
        },
    )
    return {
        "assignment_count": len(result.assignments),
        "lineage_count": len(result.lineage),
        "weekly_state_count": len(result.weekly_state),
    }


def _build_scaffold_checkpoint(
    context: Path,
    cells: Path,
    components: Path,
    exposures: Path,
    stage: Path,
    *,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    cell_frame = pd.read_parquet(cells / "weekly_exposure_cells.parquet")
    component_memberships = pd.read_parquet(
        components / "component_membership.parquet"
    )
    exposure_assignments = pd.read_parquet(
        exposures / "exposure_component_assignments.parquet"
    )
    exposure_weekly = pd.read_parquet(exposures / "exposure_weekly_state.parquet")
    scaffold = build_crop_story_scaffold(
        exposure_weekly,
        exposure_assignments,
        component_memberships,
        cell_frame,
        tracker_policy,
    )
    if scaffold.weekly_state.empty:
        raise ValueError(
            "V4-native replay produced zero crop stories; inspect the frozen "
            "baseline, significance gates, and V4 evidence before publication"
        )
    lineage = build_incident_lineage_v3(
        pd.read_parquet(exposures / "exposure_links.parquet"),
        scaffold.catalog,
        component_memberships,
    )
    stage_summary = build_incident_stage_summary(
        context / "context" / "field_week_context.parquet",
        scaffold.weekly_state,
        scaffold.memberships,
        policy=tracker_policy,
        reference_latitude=_reference_latitude(cell_frame, tracker_policy),
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    followup = build_incident_followup_evidence(
        context / "context" / "event_week_lanes.parquet",
        scaffold.weekly_state,
        scaffold.memberships,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    _write_frames(
        stage,
        {
            "scaffold_catalog.parquet": scaffold.catalog,
            "scaffold_weekly_state.parquet": scaffold.weekly_state,
            "scaffold_memberships.parquet": scaffold.memberships,
            "incident_lineage_initial.parquet": lineage.lineage,
            "incident_lineage_metadata_initial.parquet": lineage.incident_metadata,
            "stage_summary_initial.parquet": stage_summary,
            "followup_evidence.parquet": followup,
        },
    )
    return {
        "catalog_count": len(scaffold.catalog),
        "weekly_state_count": len(scaffold.weekly_state),
        "membership_count": len(scaffold.memberships),
        "followup_count": len(followup),
    }


def _build_lifecycle_checkpoint(
    context: Path,
    cells: Path,
    scaffold_dir: Path,
    stage: Path,
    *,
    tracker_policy: IncidentPolicyV3,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> dict[str, Any]:
    scaffold = CropStoryScaffold(
        pd.read_parquet(scaffold_dir / "scaffold_catalog.parquet"),
        pd.read_parquet(scaffold_dir / "scaffold_weekly_state.parquet"),
        pd.read_parquet(scaffold_dir / "scaffold_memberships.parquet"),
    )
    initial_lineage = pd.read_parquet(
        scaffold_dir / "incident_lineage_initial.parquet"
    )
    stage_summary = pd.read_parquet(scaffold_dir / "stage_summary_initial.parquet")
    followup = pd.read_parquet(scaffold_dir / "followup_evidence.parquet")
    cell_frame = pd.read_parquet(cells / "weekly_exposure_cells.parquet")
    previous_signature: str | None = None
    story_result = None
    iterations = 0
    for iterations in range(1, 9):
        story_result = finalize_crop_story_artifacts(
            scaffold,
            stage_summary,
            tracker_policy,
            followup_evidence=followup,
            incident_lineage=initial_lineage,
            weekly_cells=cell_frame,
        )
        next_summary = (
            build_incident_stage_summary(
                context / "context" / "field_week_context.parquet",
                story_result.weekly_state,
                story_result.memberships,
                policy=tracker_policy,
                reference_latitude=_reference_latitude(cell_frame, tracker_policy),
                threads=threads,
                memory_limit=memory_limit,
                temp_dir=temp_dir,
            )
            if not story_result.weekly_state.empty
            else _empty_stage_summary()
        )
        signature = _frame_signature(next_summary)
        stage_summary = next_summary
        if signature == previous_signature:
            break
        previous_signature = signature
    else:
        raise RuntimeError("V4 replay lifecycle/coverage did not converge")
    if story_result is None:
        raise RuntimeError("V4 replay lifecycle solver did not run")

    remapped = remap_incident_lineage_segments(
        initial_lineage, story_result.weekly_state, story_result.catalog
    )
    weekly = _bind_native_weekly_knowledge(
        enrich_incident_weekly_state(story_result.weekly_state, stage_summary),
        story_result.memberships,
        context / "source_generation" / "daily_causal_signals.parquet",
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    features = (
        extract_completed_incident_features(weekly, story_result.memberships)
        if not weekly.empty
        else _empty_completed_features()
    )
    catalog = story_result.catalog.merge(
        remapped.incident_metadata.drop(
            columns=["exposure_id", "crop_name_normalized"], errors="ignore"
        ),
        on="incident_id",
        how="left",
        validate="one_to_one",
    )
    _write_frames(
        stage,
        {
            "incident_catalog.parquet": catalog,
            "incident_weekly_state.parquet": weekly,
            "incident_stage_summary.parquet": stage_summary,
            "incident_membership.parquet": story_result.memberships,
            "incident_windows.parquet": story_result.windows,
            "incident_lineage.parquet": remapped.lineage,
            "completed_incident_features.parquet": features,
        },
    )
    return {
        "fixed_point_iterations": iterations,
        "incident_count": len(catalog),
        "weekly_state_count": len(weekly),
        "membership_count": len(story_result.memberships),
    }


def _publish_release(
    output_dir: Path,
    *,
    context: Path,
    baseline: Path,
    cells: Path,
    components: Path,
    exposures: Path,
    lifecycle: Path,
    audit_incident_dir: Path,
    evidence_dir: Path,
    geometry_parquet: Path,
    checkpoint_dir: Path,
    baseline_through: str,
    release_as_of: str,
    evidence_manifest: dict[str, Any],
    evidence_validation: dict[str, Any],
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
    active_checkpoints: dict[str, Path],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable V4 replay output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        "field_week_context.parquet": context / "context" / "field_week_context.parquet",
        "event_week_lanes.parquet": context / "context" / "event_week_lanes.parquet",
        "context_manifest.json": context / "context" / "manifest.json",
        "daily_episode_state_v4.parquet": context / "daily_episode_state_v4.parquet",
        "stage_baseline.parquet": baseline / "stage_baseline.parquet",
        "weekly_exposure_cells.parquet": cells / "weekly_exposure_cells.parquet",
        "weekly_components.parquet": components / "weekly_components.parquet",
        "component_membership.parquet": components / "component_membership.parquet",
        "exposure_component_assignments.parquet": exposures / "exposure_component_assignments.parquet",
        "exposure_links.parquet": exposures / "exposure_links.parquet",
        "exposure_weekly_state.parquet": exposures / "exposure_weekly_state.parquet",
    }
    sources.update(
        {
            name: lifecycle / name
            for name in (
                "incident_catalog.parquet",
                "incident_weekly_state.parquet",
                "incident_stage_summary.parquet",
                "incident_membership.parquet",
                "incident_windows.parquet",
                "incident_lineage.parquet",
                "completed_incident_features.parquet",
            )
        }
    )
    with TemporaryDirectory(prefix=".incident-story-replay-v4-", dir=output_dir.parent) as tmp:
        stage = Path(tmp) / output_dir.name
        stage.mkdir()
        for name, source in sources.items():
            shutil.copy2(source, stage / name)
        audit_duplicate_membership_count = _audit_duplicate_membership_count(
            audit_incident_dir / "incident_membership.parquet"
        )
        crosswalk = build_incident_crosswalk_v4(
            audit_incident_dir, stage / "incident_membership.parquet"
        )
        crosswalk.to_parquet(
            stage / "old_to_new_incident_crosswalk.parquet",
            index=False,
            compression="zstd",
        )
        validation = validate_final_artifact_directory(stage)
        mismatch_details = _membership_counter_mismatch_details(stage)
        mismatch_count = _membership_counter_mismatch_count(mismatch_details)
        if mismatch_count:
            metric_counts = ", ".join(
                f"{metric}={count}"
                for metric, count in mismatch_details["metric"]
                .value_counts(sort=False)
                .sort_index()
                .items()
            )
            examples = "; ".join(
                f"{row.incident_id}@{pd.Timestamp(row.timeline_bucket).date()}:"
                f"{row.metric} weekly={row.weekly_count} membership={row.membership_count}"
                for row in mismatch_details.head(6).itertuples(index=False)
            )
            raise ValueError(
                "V4 replay weekly/member counter reconciliation has "
                f"{mismatch_count} mismatched incident-weeks across "
                f"{len(mismatch_details)} counters; metrics[{metric_counts}]; "
                f"examples[{examples}]"
            )
        artifact_names = sorted(path.name for path in stage.iterdir() if path.is_file())
        source_run = evidence_manifest.get("run") or {}
        generation_id = stable_id(
            "incident_replay_v4",
            (
                file_sha256(evidence_dir / "manifest.json"),
                file_sha256(geometry_parquet),
                source_policy.source_sha256,
                tracker_policy.source_sha256,
                baseline_through,
                REPLAY_ALGORITHM_REVISION,
                LIFECYCLE_ALGORITHM_REVISION,
            ),
            length=20,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "run": {
                "status": "complete",
                "immutable": True,
                "generation_id": generation_id,
                "source_generation_id": source_run.get("source_generation_id"),
                "as_of_date": release_as_of,
                "baseline_through": baseline_through,
                "lifecycle_state_recomputed_from_v4": True,
            },
            "source": {
                "evidence_manifest_sha256": file_sha256(evidence_dir / "manifest.json"),
                "geometry_sha256": file_sha256(geometry_parquet),
                "audit_incident_manifest_sha256": file_sha256(
                    audit_incident_dir / "manifest.json"
                ),
                "audit_incident_membership_sha256": file_sha256(
                    audit_incident_dir / "incident_membership.parquet"
                ),
                "generation_manifest_sha256": file_sha256(
                    context / "source_generation" / "manifest.json"
                ),
                "old_release_role": "overlap_crosswalk_and_audit_only",
            },
            "policy": {
                "version": tracker_policy.version,
                "sha256": tracker_policy.source_sha256,
                "calibration_status": tracker_policy.calibration_status,
                "warning": tracker_policy.warning,
            },
            "policies": _policy_manifest(source_policy, tracker_policy),
            "semantics": {
                "story_spine_source": "v4_evidence_ledgers",
                "old_incident_ids_seed_new_ids": False,
                "lifecycle_state_recomputed_from_v4": True,
                "component_absence_replayed_from_v4": True,
                "missing_weather_advances_quiet_clocks": False,
                "unsupported_s2_is_no_decline": False,
                "unsupported_s2_is_unknown": True,
                "exact_v4_response_severity_preserved": True,
                "decision_identity_clock": "max_effective_and_knowledge_time",
                "motif_models_reused": False,
                "replay_algorithm_revision": REPLAY_ALGORITHM_REVISION,
                "lifecycle_algorithm_revision": LIFECYCLE_ALGORITHM_REVISION,
            },
            "evidence_validation": evidence_validation,
            "validation": {
                **validation,
                "membership_counter_mismatch_count": mismatch_count,
                "crosswalk_rows": len(crosswalk),
                "crosswalk_overlap_rows": int(crosswalk["match_status"].eq("overlap").sum()),
                "audit_duplicate_story_week_field_membership_count": (
                    audit_duplicate_membership_count
                ),
            },
            "checkpoints": {
                name: _fingerprint(path / "manifest.json")
                for name, path in sorted(active_checkpoints.items())
            },
            "artifacts": artifact_hashes(stage, artifact_names),
        }
        superseded_lifecycle = checkpoint_dir / "07_lifecycle"
        if (
            superseded_lifecycle.is_dir()
            and (superseded_lifecycle / "manifest.json").is_file()
        ):
            manifest["superseded_checkpoints"] = {
                "07_lifecycle": {
                    **_fingerprint(superseded_lifecycle / "manifest.json"),
                    "reason": "routed_current_recovery_attribution_reconciled",
                    "replacement": LIFECYCLE_CHECKPOINT_NAME,
                }
            }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output_dir)


def _checkpoint(
    path: Path,
    *,
    stage_name: str,
    inputs: dict[str, Any],
    builder: Callable[[Path], dict[str, Any]],
) -> Path:
    if path.exists() or path.is_symlink():
        manifest = _read_json(path / "manifest.json")
        if (
            manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or (manifest.get("run") or {}).get("status") != "complete"
            or manifest.get("stage") != stage_name
            or manifest.get("inputs") != inputs
        ):
            raise ValueError(f"Replay checkpoint is incompatible and will not be overwritten: {path}")
        _verify_inventory(path, manifest.get("artifacts") or {})
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
        stage = Path(tmp) / path.name
        stage.mkdir()
        metadata = builder(stage)
        files = sorted(
            item.relative_to(stage).as_posix()
            for item in stage.rglob("*")
            if item.is_file()
            and item.relative_to(stage).as_posix() != "manifest.json"
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": stage_name,
            "run": {"status": "complete", "immutable": True},
            "inputs": inputs,
            "metadata": metadata,
            "artifacts": {
                name: _fingerprint(stage / name)
                for name in files
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, path)
    return path


def _weekly_snapshots(
    daily: pd.DataFrame, evidence_manifest: dict[str, Any]
) -> pd.DataFrame:
    columns = [
        "timeline_bucket", "snapshot_as_of_date", "field_id", "crop_name",
        "crop_season", "crop_instance_id", "event_id", "story_cluster_id",
        "event_state_id", "event_state", "hazard_signature", "max_risk_rank",
        "max_risk_band", "current_risk_rank", "current_risk_band",
        "reportable_day_count", "response_day_count", "right_censored",
        "is_data_gap_snapshot", "requires_review", "daily_pressure_rank",
        "daily_response_class", "revision", "generation_as_of_date",
        "knowledge_time",
    ]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    frame = daily.copy()
    observed = pd.to_datetime(frame["decision_date"])
    frame["timeline_bucket"] = (
        observed - pd.to_timedelta(observed.dt.weekday, unit="D")
    ).dt.normalize()
    frame["snapshot_as_of_date"] = observed.dt.normalize()
    frame = frame.sort_values(
        ["event_id", "timeline_bucket", "snapshot_as_of_date"], kind="mergesort"
    ).groupby(["event_id", "timeline_bucket"], sort=True).tail(1)
    frame["story_cluster_id"] = frame["event_id"]
    frame["event_state_id"] = frame.apply(
        lambda row: stable_id(
            "state_v4",
            (row["event_id"], row["snapshot_as_of_date"], row["event_state"]),
        ),
        axis=1,
    )
    frame["hazard_signature"] = frame["hazard_family"]
    frame["max_risk_band"] = frame["max_risk_rank"].map(_risk_band)
    frame["current_risk_rank"] = frame["pressure_rank"]
    frame["current_risk_band"] = frame["pressure_band"]
    frame["daily_pressure_rank"] = frame["pressure_rank"]
    frame["daily_response_class"] = frame["response_class"]
    frame["requires_review"] = frame["event_state"].isin(
        {"SEVERE", "RECOVERING", "CLOSED_RESPONSE_UNRESOLVED"}
    )
    frame["revision"] = 1
    frame["generation_as_of_date"] = (
        (evidence_manifest.get("run") or {}).get("release_as_of")
        or (evidence_manifest.get("run") or {}).get("as_of_date")
    )
    return frame.reindex(columns=columns).reset_index(drop=True)


def _source_adapter_manifest(
    source: Path,
    *,
    evidence_dir: Path,
    evidence_manifest: dict[str, Any],
    geometry_parquet: Path,
    source_policy: IncidentPolicyV4,
    tracker_policy: IncidentPolicyV3,
    diagnostics: dict[str, int],
) -> dict[str, Any]:
    run = evidence_manifest.get("run") or {}
    artifacts = sorted(path.name for path in source.iterdir() if path.is_file())
    return {
        "schema_version": SOURCE_ADAPTER_SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "immutable": True,
            "generation_id": run.get("source_generation_id"),
            "as_of_date": run.get("release_as_of") or run.get("as_of_date"),
            "source_generation_id": run.get("source_generation_id"),
        },
        "policy": {
            "version": tracker_policy.version,
            "sha256": tracker_policy.source_sha256,
            "calibration_status": tracker_policy.calibration_status,
            "warning": tracker_policy.warning,
        },
        "policies": _policy_manifest(source_policy, tracker_policy),
        "source": {
            "evidence_manifest_sha256": file_sha256(evidence_dir / "manifest.json"),
            "geometry_sha256": file_sha256(geometry_parquet),
        },
        "semantics": {
            "adapter_only": True,
            "story_source": "v4_evidence_ledgers",
            "old_v3_response_inputs_used": False,
            "simultaneous_hazard_day_major_replay": True,
        },
        "counts": diagnostics,
        "artifacts": artifact_hashes(source, artifacts),
    }


def _membership_counter_mismatch_details(release: Path) -> pd.DataFrame:
    weekly = pd.read_parquet(release / "incident_weekly_state.parquet")
    memberships = pd.read_parquet(release / "incident_membership.parquet")
    if weekly.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "timeline_bucket",
                "metric",
                "weekly_count",
                "membership_count",
                "delta",
            ]
        )
    keys = ["incident_id", "timeline_bucket"]
    membership = memberships.copy()
    membership["timeline_bucket"] = pd.to_datetime(membership["timeline_bucket"]).dt.normalize()
    weekly["timeline_bucket"] = pd.to_datetime(weekly["timeline_bucket"]).dt.normalize()
    grouped = membership.groupby(keys, sort=False).agg(
        membership_pressure_core_field_count=(
            "field_id",
            lambda values: values[membership.loc[values.index, "membership_role"].eq("pressure_core")].nunique(),
        ),
        membership_severe_field_count=(
            "field_id",
            lambda values: values[
                membership.loc[values.index, "membership_role"].eq("pressure_core")
                & membership.loc[values.index, "event_state"].eq("SEVERE")
            ].nunique(),
        ),
        membership_watch_frontier_field_count=(
            "field_id",
            lambda values: values[membership.loc[values.index, "membership_role"].eq("watch_frontier")].nunique(),
        ),
        membership_impact_lag_field_count=(
            "field_id",
            lambda values: values[membership.loc[values.index, "membership_role"].eq("impact_lag")].nunique(),
        ),
        membership_fresh_decline_field_count=(
            "field_id",
            lambda values: values[
                membership.loc[values.index, "fresh_response_evidence"].astype(bool)
                & membership.loc[values.index, "response_class"].isin(
                    {"medium_decline", "severe_decline"}
                )
            ].nunique(),
        ),
        membership_fresh_recovery_field_count=(
            "field_id",
            lambda values: values[
                membership.loc[values.index, "fresh_response_evidence"].astype(bool)
                & membership.loc[values.index, "response_class"].eq("recovery")
            ].nunique(),
        ),
    ).reset_index()
    joined = weekly.merge(grouped, on=keys, how="left", validate="one_to_one")
    pairs = (
        ("pressure_core_field_count", "membership_pressure_core_field_count"),
        ("severe_field_count", "membership_severe_field_count"),
        ("watch_frontier_field_count", "membership_watch_frontier_field_count"),
        ("impact_lag_field_count", "membership_impact_lag_field_count"),
        ("fresh_decline_field_count", "membership_fresh_decline_field_count"),
        ("fresh_recovery_field_count", "membership_fresh_recovery_field_count"),
    )
    details: list[pd.DataFrame] = []
    for weekly_column, membership_column in pairs:
        weekly_count = (
            pd.to_numeric(joined[weekly_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        membership_count = (
            pd.to_numeric(joined[membership_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        mismatch = weekly_count.ne(membership_count)
        if not mismatch.any():
            continue
        detail = joined.loc[mismatch, keys].copy()
        detail["metric"] = weekly_column
        detail["weekly_count"] = weekly_count.loc[mismatch].to_numpy()
        detail["membership_count"] = membership_count.loc[mismatch].to_numpy()
        detail["delta"] = (
            detail["membership_count"] - detail["weekly_count"]
        )
        details.append(detail)
    if not details:
        return pd.DataFrame(
            columns=[
                *keys,
                "metric",
                "weekly_count",
                "membership_count",
                "delta",
            ]
        )
    return pd.concat(details, ignore_index=True).sort_values(
        [*keys, "metric"], kind="mergesort"
    ).reset_index(drop=True)


def _membership_counter_mismatch_count(details: pd.DataFrame) -> int:
    if details.empty:
        return 0
    return int(
        details[["incident_id", "timeline_bucket"]]
        .drop_duplicates()
        .shape[0]
    )


def _membership_counter_mismatches(release: Path) -> int:
    return _membership_counter_mismatch_count(
        _membership_counter_mismatch_details(release)
    )


def _audit_duplicate_membership_count(path: Path) -> int:
    frame = pd.read_parquet(
        path, columns=["incident_id", "timeline_bucket", "field_id"]
    )
    frame["incident_id"] = frame["incident_id"].astype(str).str.strip()
    frame["field_id"] = frame["field_id"].astype(str).str.strip()
    frame["timeline_bucket"] = pd.to_datetime(
        frame["timeline_bucket"], errors="raise", utc=True
    ).dt.tz_localize(None).dt.normalize()
    return int(
        frame.duplicated(
            ["incident_id", "timeline_bucket", "field_id"], keep=False
        ).sum()
    )


def _validate_build_inputs(
    evidence_dir: Path,
    geometry_parquet: Path,
    audit_incident_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    baseline_date: pd.Timestamp,
) -> None:
    if pd.isna(baseline_date):
        raise ValueError("baseline_through is invalid")
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"V4 evidence directory does not exist: {evidence_dir}")
    if not geometry_parquet.is_file():
        raise FileNotFoundError(f"Geometry parquet does not exist: {geometry_parquet}")
    if not audit_incident_dir.is_dir() or not (
        audit_incident_dir / "incident_membership.parquet"
    ).is_file():
        raise FileNotFoundError(
            f"Audit incident release is missing incident membership: {audit_incident_dir}"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Immutable V4 replay output already exists: {output_dir}")
    immutable = (evidence_dir, audit_incident_dir)
    for target, label in ((output_dir, "output"), (checkpoint_dir, "checkpoint")):
        for source in immutable:
            if target == source or target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError(f"Replay {label} directory must not overlap immutable inputs")
    if output_dir == checkpoint_dir or output_dir.is_relative_to(checkpoint_dir) or checkpoint_dir.is_relative_to(output_dir):
        raise ValueError("Replay output and checkpoint directories must be separate")
    for target, label in ((output_dir, "output"), (checkpoint_dir, "checkpoint")):
        if geometry_parquet == target or geometry_parquet.is_relative_to(target):
            raise ValueError(
                f"Replay {label} directory must not contain immutable geometry"
            )


def _policy_manifest(
    source_policy: IncidentPolicyV4, tracker_policy: IncidentPolicyV3
) -> dict[str, Any]:
    return {
        "v4_evidence": {
            "version": source_policy.version,
            "sha256": source_policy.source_sha256,
            "effective_sha256": _effective_policy_sha256(source_policy),
            "calibration_status": source_policy.calibration_status,
        },
        "story_tracking": {
            "version": tracker_policy.version,
            "sha256": tracker_policy.source_sha256,
            "source_file_sha256": file_sha256(tracker_policy.source_path),
            "calibration_status": tracker_policy.calibration_status,
        },
    }


def _effective_policy_sha256(policy: object) -> str:
    payload = asdict(policy)  # type: ignore[arg-type]
    payload.pop("source_path", None)
    payload.pop("source_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reference_latitude(cells: pd.DataFrame, policy: IncidentPolicyV3) -> float:
    if cells.empty or "reference_latitude" not in cells:
        return float(policy.grid_origin_lat)
    values = pd.to_numeric(cells["reference_latitude"], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError("V4 replay cells do not have one frozen reference latitude")
    return float(values[0])


def _bind_native_weekly_knowledge(
    weekly: pd.DataFrame,
    memberships: pd.DataFrame,
    daily_signals_path: Path,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> pd.DataFrame:
    """Bind checkpoints to exact member or current component-absence evidence."""
    if weekly.empty:
        return weekly.copy()
    required = {
        "incident_id",
        "timeline_bucket",
        "crop_name",
        "knowledge_time",
        "knowledge_time_inferred",
    }
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise ValueError(
            "Native weekly states cannot bind exact knowledge time: "
            + ", ".join(missing)
        )
    member_required = {"incident_id", "timeline_bucket", "knowledge_time"}
    missing_members = sorted(member_required - set(memberships.columns))
    if missing_members:
        raise ValueError(
            "Native memberships cannot bind checkpoint knowledge time: "
            + ", ".join(missing_members)
        )

    output = weekly.copy()
    output["timeline_bucket"] = pd.to_datetime(
        output["timeline_bucket"], errors="raise"
    ).dt.normalize()
    output["_source_knowledge"] = pd.to_datetime(
        output["knowledge_time"], errors="coerce", utc=True
    )
    output["_was_inferred"] = (
        output["knowledge_time_inferred"].fillna(True).astype(bool)
    )

    member = memberships.loc[:, list(member_required)].copy()
    member["timeline_bucket"] = pd.to_datetime(
        member["timeline_bucket"], errors="raise"
    ).dt.normalize()
    member["_membership_knowledge"] = pd.to_datetime(
        member.pop("knowledge_time"), errors="coerce", utc=True
    )
    member_max = (
        member.groupby(["incident_id", "timeline_bucket"], sort=True)[
            "_membership_knowledge"
        ]
        .max()
        .reset_index()
    )
    output = output.merge(
        member_max,
        on=["incident_id", "timeline_bucket"],
        how="left",
        validate="one_to_one",
    )
    absence_bounds = _native_crop_week_knowledge_bounds(
        daily_signals_path,
        threads=threads,
        memory_limit=memory_limit,
        temp_dir=temp_dir,
    )
    output["_crop_name_normalized"] = output["crop_name"].map(
        _normalize_crop_name
    )
    output = output.merge(
        absence_bounds,
        on=["timeline_bucket", "_crop_name_normalized"],
        how="left",
        validate="many_to_one",
    )
    missing_absence = output["_was_inferred"] & output[
        "_absence_knowledge"
    ].isna()
    if missing_absence.any():
        examples = output.loc[
            missing_absence, ["incident_id", "timeline_bucket", "crop_name"]
        ].head(5).to_dict("records")
        raise ValueError(
            "Native component-absence checkpoints lack current V4 knowledge bounds: "
            f"{examples}"
        )

    candidates = output[["_source_knowledge", "_membership_knowledge"]].copy()
    candidates["_absence_knowledge"] = output["_absence_knowledge"].where(
        output["_was_inferred"]
    )
    output["knowledge_time"] = candidates.max(axis=1)
    if output["knowledge_time"].isna().any():
        raise ValueError("Native story checkpoints have no exact knowledge time")
    under_membership = (
        output["_membership_knowledge"].notna()
        & (output["knowledge_time"] < output["_membership_knowledge"])
    )
    if under_membership.any():
        raise ValueError("Native story checkpoint precedes membership knowledge")
    output["knowledge_time_inferred"] = False
    return output.drop(
        columns=[
            "_source_knowledge",
            "_membership_knowledge",
            "_was_inferred",
            "_crop_name_normalized",
            "_absence_knowledge",
        ]
    )


def _native_crop_week_knowledge_bounds(
    daily_signals_path: Path,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> pd.DataFrame:
    with _duckdb_connection(threads, memory_limit, temp_dir) as connection:
        frame = connection.execute(
            f"""
            SELECT
                DATE_TRUNC('week', TRY_CAST(observation_date AS DATE))::DATE
                    AS timeline_bucket,
                COALESCE(NULLIF(TRIM(BOTH '_' FROM REGEXP_REPLACE(
                    LOWER(COALESCE(CAST(crop_name AS VARCHAR), 'unknown')),
                    '[^a-z0-9]+', '_', 'g'
                )), ''), 'unknown') AS _crop_name_normalized,
                MAX(TRY_CAST(knowledge_time AS TIMESTAMP)) AS _absence_knowledge
            FROM read_parquet({_sql_literal(daily_signals_path)})
            WHERE TRY_CAST(observation_date AS DATE) IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
    frame["timeline_bucket"] = pd.to_datetime(
        frame["timeline_bucket"], errors="raise"
    ).dt.normalize()
    frame["_absence_knowledge"] = pd.to_datetime(
        frame["_absence_knowledge"], errors="coerce", utc=True
    )
    return frame


def _normalize_crop_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").lower()).strip(
        "_"
    ) or "unknown"


def _frame_signature(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(columns=sorted(frame.columns)).copy()
    sort_columns = [
        name for name in ("timeline_bucket", "incident_id", "stage_bucket")
        if name in ordered
    ]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns, kind="mergesort")
    payload = pd.util.hash_pandas_object(
        ordered.reset_index(drop=True), index=True, categorize=True
    ).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _empty_stage_summary() -> pd.DataFrame:
    from .incident_denominators_v3 import SUMMARY_COLUMNS

    return pd.DataFrame(columns=list(SUMMARY_COLUMNS))


def _empty_completed_features() -> pd.DataFrame:
    from .incident_archetypes_v3 import MODEL_FEATURE_COLUMNS

    return pd.DataFrame(columns=["incident_id", *MODEL_FEATURE_COLUMNS])


def _write_frames(directory: Path, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.reset_index(drop=True).to_parquet(
            directory / name, index=False, compression="zstd"
        )


def _risk_band(value: Any) -> str:
    return {0: "NONE", 1: "LOW", 2: "LOW-MED", 3: "MED-HIGH", 4: "HIGH"}.get(
        int(value or 0), "UNKNOWN"
    )


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_inventory(root: Path, inventory: dict[str, Any]) -> None:
    for name, expected in inventory.items():
        path = root / name
        if not path.is_file() or _fingerprint(path) != expected:
            raise ValueError(f"Replay checkpoint artifact changed: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return payload


def _parquet_count(path: Path) -> int:
    with duckdb.connect(":memory:") as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
            ).fetchone()[0]
        )


@contextmanager
def _duckdb_connection(
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET preserve_insertion_order=false")
        if memory_limit:
            connection.execute("SET memory_limit=?", [str(memory_limit)])
        if temp_dir is not None:
            resolved = temp_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            connection.execute("SET temp_directory=?", [str(resolved)])
        yield connection
    finally:
        connection.close()


def _sql_literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CONTEXT_SCHEMA_VERSION",
    "MODE",
    "REPLAY_ALGORITHM_REVISION",
    "SCHEMA_VERSION",
    "SOURCE_ADAPTER_SCHEMA_VERSION",
    "build_incident_story_replay_v4",
]
