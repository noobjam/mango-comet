"""Full-dataset story generation without loading the source into one DataFrame."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd

from .causal_features import prepare_causal_signals
from .contracts import MonitorPolicy, SCHEMA_VERSION, stable_id
from .pipeline import (
    ALIASES,
    BOUNDED_V1_MAX_FIELDS,
    EVENT_COLUMNS,
    MEMBERSHIP_COLUMNS,
    SNAPSHOT_COLUMNS,
    GenerationResult,
    _canonical_select,
    _crop_instances,
    _ensure_columns,
    _event_labels,
    _file_metadata,
    _map_frames,
    _resolve_columns,
    _weekly_snapshots,
    _write_parquet,
)
from .state_machine import run_state_machine


ARTIFACTS: dict[str, tuple[str, list[str]]] = {
    "signals": ("daily_causal_signals.parquet", ["field_id", "observation_date"]),
    "instances": ("crop_instances.parquet", ["field_id", "crop_instance_start_date"]),
    "events": ("event_windows.parquet", ["field_id", "event_start_date"]),
    "snapshots": ("event_state_snapshots.parquet", ["timeline_bucket", "field_id", "event_id"]),
    "memberships": ("story_day_membership.parquet", ["field_id", "observation_date", "event_id"]),
    "frames": ("map_frame_fields.parquet", ["timeline_bucket", "field_id", "event_id"]),
    "labels": ("event_story_cluster_labels.parquet", ["story_cluster_id"]),
}
EMPTY_COLUMNS: dict[str, list[str]] = {
    "events": list(EVENT_COLUMNS),
    "snapshots": list(SNAPSHOT_COLUMNS),
    "memberships": list(MEMBERSHIP_COLUMNS),
    "frames": [
        "timeline_bucket", "field_id", "story_cluster_id", "event_id", "event_state_id",
        "event_state", "crop_name", "crop_season", "crop_instance_id", "max_risk_band",
        "current_risk_band", "hazard_signature", "motif_family", "response_signature",
        "reportable_day_count", "event_count", "max_risk_rank", "current_risk_rank",
        "response_day_count", "right_censored",
        "requires_review",
    ],
    "labels": [
        "story_cluster_id", "short_label", "max_risk_band", "hazard_signature",
        "motif_family", "response_signature", "event_count", "field_count", "crop_count",
        "median_window_span_days", "median_reportable_days",
    ],
}


@dataclass(frozen=True)
class PartitionOptions:
    partitions: int = 128
    workers: int = 1
    threads: int = 16
    memory_limit: str | None = None
    temp_dir: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.partitions <= 4096:
            raise ValueError("partitions must be between 1 and 4096")
        if not 1 <= self.workers <= 64:
            raise ValueError("workers must be between 1 and 64")
        if not 1 <= self.threads <= 256:
            raise ValueError("threads must be between 1 and 256")


def build_partitioned_generation(
    *,
    input_parquet: Path,
    output_dir: Path,
    as_of_date: date,
    policy: MonitorPolicy,
    history_from: date | None = None,
    geometry_parquet: Path | None = None,
    options: PartitionOptions = PartitionOptions(),
) -> GenerationResult:
    """Build one generation after a single source scan into field partitions."""
    options.validate()
    input_parquet = input_parquet.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_parquet.is_file():
        raise FileNotFoundError(f"Input parquet does not exist: {input_parquet}")
    source_metadata = _file_metadata(input_parquet)
    geometry_metadata = None
    if geometry_parquet is not None:
        geometry_parquet = geometry_parquet.expanduser().resolve()
        if not geometry_parquet.is_file():
            raise FileNotFoundError(f"Geometry parquet does not exist: {geometry_parquet}")
        geometry_metadata = _file_metadata(geometry_parquet)
    generation_id = stable_id(
        "generation",
        (
            "partitioned-v1",
            as_of_date.isoformat(),
            history_from,
            options.partitions,
            policy.source_sha256,
            source_metadata["name"],
            source_metadata["size_bytes"],
            source_metadata["mtime_ns"],
            None if geometry_metadata is None else geometry_metadata["mtime_ns"],
        ),
        length=20,
    )
    generations_dir = output_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    final_dir = generations_dir / f"{as_of_date.isoformat()}_{generation_id}"
    if final_dir.exists():
        raise FileExistsError(f"Immutable generation already exists: {final_dir}")

    with TemporaryDirectory(prefix=".weekly-story-full-", dir=generations_dir) as temporary:
        transaction = Path(temporary)
        stage = transaction / final_dir.name
        stage.mkdir()
        canonical_root = transaction / "canonical"
        parts_root = transaction / "parts"
        canonical_root.mkdir()
        parts_root.mkdir()
        mapping = _partition_source(
            input_parquet,
            canonical_root,
            as_of_date=as_of_date,
            history_from=history_from,
            options=options,
            transaction_dir=transaction,
        )
        partition_files = _partition_file_groups(canonical_root)
        if not partition_files:
            raise ValueError(f"Input contains no rows on or before {as_of_date.isoformat()}.")
        jobs = [
            (index, files, parts_root, policy, as_of_date)
            for index, files in enumerate(partition_files)
        ]
        if options.workers == 1:
            summaries = [_process_partition(job) for job in jobs]
        else:
            try:
                executor = ProcessPoolExecutor(
                    max_workers=options.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            except (OSError, PermissionError):
                # Restricted runtimes may deny POSIX semaphore inspection. The
                # partition contract remains correct when processed serially.
                summaries = [_process_partition(job) for job in jobs]
            else:
                with executor:
                    summaries = list(executor.map(_process_partition, jobs))

        _assemble_parts(parts_root, stage)
        if geometry_parquet is not None:
            shutil.copy2(geometry_parquet, stage / "map_field_geometry.parquet")
        if _file_metadata(input_parquet) != source_metadata:
            raise RuntimeError("Input parquet changed during generation; staged outputs were discarded.")

        totals = _summarize_generation(stage)
        manifest = _manifest(
            generation_id=generation_id,
            as_of_date=as_of_date,
            history_from=history_from,
            source_metadata=source_metadata,
            geometry_metadata=geometry_metadata,
            mapping=mapping,
            policy=policy,
            options=options,
            partition_summaries=summaries,
            totals=totals,
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, final_dir)
    return GenerationResult(
        generation_id,
        final_dir,
        as_of_date,
        int(totals["row_count"]),
        int(totals["event_count"]),
    )


def _partition_source(
    input_parquet: Path,
    destination: Path,
    *,
    as_of_date: date,
    history_from: date | None,
    options: PartitionOptions,
    transaction_dir: Path,
) -> dict[str, str]:
    with duckdb.connect(":memory:") as connection:
        description = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_parquet)]
        ).fetchall()
        mapping = _resolve_columns([str(row[0]) for row in description])
        source = _canonical_select(mapping)
        date_column = _quote_identifier(mapping["observation_date"])
        predicates = [f"TRY_CAST({date_column} AS DATE) <= DATE {_sql_string(as_of_date.isoformat())}"]
        if history_from is not None:
            predicates.append(
                f"TRY_CAST({date_column} AS DATE) >= DATE {_sql_string(history_from.isoformat())}"
            )
        temp_dir = (options.temp_dir or (transaction_dir / "duckdb-tmp")).expanduser().resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"PRAGMA threads={options.threads}")
        connection.execute(f"PRAGMA temp_directory={_sql_string(str(temp_dir))}")
        if options.memory_limit:
            connection.execute(f"SET memory_limit={_sql_string(options.memory_limit)}")
        connection.execute(
            f"""
            COPY (
                WITH canonical AS (
                    SELECT {source}
                    FROM read_parquet({_sql_string(str(input_parquet))})
                    WHERE {' AND '.join(predicates)}
                )
                SELECT
                    *,
                    CAST(hash(field_id) % {options.partitions} AS INTEGER) AS field_partition
                FROM canonical
                WHERE field_id IS NOT NULL
                  AND TRIM(field_id) <> ''
                  AND observation_date IS NOT NULL
            ) TO {_sql_string(str(destination))}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (field_partition))
            """
        )
    return mapping


def _partition_file_groups(root: Path) -> list[list[str]]:
    grouped: dict[Path, list[str]] = {}
    for path in sorted(root.rglob("*.parquet")):
        grouped.setdefault(path.parent, []).append(str(path))
    return [grouped[key] for key in sorted(grouped, key=str)]


def _process_partition(
    job: tuple[int, list[str], Path, MonitorPolicy, date]
) -> dict[str, int]:
    index, files, parts_root, policy, as_of_date = job
    canonical_columns = list(ALIASES)
    with duckdb.connect(":memory:") as connection:
        frame = connection.execute(
            f"""
            SELECT {', '.join(_quote_identifier(name) for name in canonical_columns)}
            FROM read_parquet(?, union_by_name=true, hive_partitioning=false)
            ORDER BY field_id, observation_date
            """,
            [files],
        ).fetchdf()
    field_count = int(frame["field_id"].nunique())
    if field_count > BOUNDED_V1_MAX_FIELDS:
        raise ValueError(
            f"Partition {index} contains {field_count} fields; increase --partitions so each "
            f"partition has at most {BOUNDED_V1_MAX_FIELDS}."
        )
    signals = prepare_causal_signals(frame, policy)
    machine = run_state_machine(signals, policy, as_of_date=as_of_date)
    events = _ensure_columns(machine.events, EVENT_COLUMNS)
    memberships = _ensure_columns(machine.memberships, MEMBERSHIP_COLUMNS)
    snapshots = _weekly_snapshots(machine, as_of_date)
    artifacts = {
        "signals": signals,
        "instances": _crop_instances(signals),
        "events": events,
        "snapshots": _ensure_columns(snapshots, SNAPSHOT_COLUMNS),
        "memberships": memberships,
        "frames": _map_frames(machine, snapshots),
        "labels": _event_labels(machine),
    }
    for key, frame_part in artifacts.items():
        if frame_part.empty:
            continue
        artifact_dir = parts_root / key
        artifact_dir.mkdir(exist_ok=True)
        _write_parquet(frame_part, artifact_dir / f"part-{index:05d}.parquet")
    return {
        "partition": index,
        "field_count": field_count,
        "row_count": int(len(signals)),
        "event_count": int(len(events)),
        "snapshot_count": int(len(snapshots)),
    }


def _assemble_parts(parts_root: Path, stage: Path) -> None:
    for key, (name, sort_columns) in ARTIFACTS.items():
        files = sorted((parts_root / key).glob("*.parquet"))
        if not files:
            _write_parquet(pd.DataFrame(columns=EMPTY_COLUMNS.get(key, [])), stage / name)
            continue
        columns = _parquet_columns(files[0])
        valid_sort = [column for column in sort_columns if column in columns]
        order = ", ".join(_quote_identifier(column) for column in valid_sort)
        order_clause = f"ORDER BY {order}" if order else ""
        file_sql = "[" + ",".join(_sql_string(str(path)) for path in files) + "]"
        with duckdb.connect(":memory:") as connection:
            connection.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet({file_sql}, union_by_name=true)
                    {order_clause}
                ) TO {_sql_string(str(stage / name))}
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )


def _summarize_generation(stage: Path) -> dict[str, int]:
    with duckdb.connect(":memory:") as connection:
        row_count, field_count, instance_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT field_id), COUNT(DISTINCT crop_instance_id)
            FROM read_parquet(?)
            """,
            [str(stage / "daily_causal_signals.parquet")],
        ).fetchone()
        event_count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(stage / "event_windows.parquet")]
        ).fetchone()[0]
        snapshot_count = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(stage / "event_state_snapshots.parquet")]
        ).fetchone()[0]
    return {
        "row_count": int(row_count),
        "field_count": int(field_count),
        "crop_instance_count": int(instance_count),
        "event_count": int(event_count),
        "snapshot_count": int(snapshot_count),
    }


def _manifest(**values: Any) -> dict[str, Any]:
    policy: MonitorPolicy = values["policy"]
    options: PartitionOptions = values["options"]
    totals: dict[str, int] = values["totals"]
    geometry_metadata = values["geometry_metadata"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "status": "complete",
            "generation_id": values["generation_id"],
            "as_of_date": values["as_of_date"].isoformat(),
            "immutable": True,
            **totals,
            "viewer_ready": False,
            "viewer_bundle_required": geometry_metadata is not None and totals["snapshot_count"] > 0,
        },
        "input": {
            "source": values["source_metadata"],
            "history_from": (
                None if values["history_from"] is None else values["history_from"].isoformat()
            ),
            "max_fields": None,
            "column_mapping": values["mapping"],
            "requires_spectral_echo_days": True,
        },
        "policy": {
            "version": policy.version,
            "sha256": policy.source_sha256,
            "calibration_status": policy.calibration_status,
            "warning": "Starter numeric thresholds are uncalibrated and require agronomist validation.",
        },
        "processing": {
            "mode": "single_scan_field_hash_partitions_v1",
            "partitions": options.partitions,
            "workers": options.workers,
            "threads": options.threads,
            "partition_summaries": values["partition_summaries"],
        },
        "semantics": {
            "prefix_safe": True,
            "full_history_z_scores": False,
            "spectral_carry_forward_is_new_evidence": False,
            "open_events_are_right_censored": True,
            "story_cluster_id_alias": "event_id",
            "persistent_event_registry": False,
            "event_id_stability": "ordinary future appends only; earlier corrections can move onset-derived IDs",
            "motif_family": "hazard-family compatibility facet; not a learned motif",
            "causal_prefix_feature_version": "causal_prefix_features_v1",
        },
        "limitations": [
            "Late corrections create a new immutable generation but no automatic supersession lineage.",
            "Event IDs are stable under ordinary appends; an earlier late-arriving onset can change identity.",
            "Motif discovery is a separate frozen-model operation.",
            "Starter policy thresholds require agronomist and outcome validation.",
        ],
        "outputs": {
            key: name for key, (name, _) in ARTIFACTS.items()
        } | {"map_field_geometry": "map_field_geometry.parquet" if geometry_metadata else None},
    }


def _parquet_columns(path: Path) -> set[str]:
    with duckdb.connect(":memory:") as connection:
        description = connection.execute(
            "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
        ).description
    return {str(item[0]) for item in description or []}


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
