#!/usr/bin/env python3
"""Build a deterministic field-story V1 release from completed V4 ledgers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd

from story_monitor.field_stories_v1 import (
    POLICY_PATH,
    FieldStoryArtifacts,
    build_field_stories,
    load_field_story_policy,
)
from story_monitor.incident_validation_v4 import validate_evidence_directory


EVIDENCE_FILES = {
    "crop": "crop_day_context_v4.parquet",
    "pressure": "field_day_pressure_v4.parquet",
    "responses": "field_s2_acquisition_v4.parquet",
}
ARTIFACTS = {
    "daily_state": (
        "field_story_daily_state_v1.parquet",
        ["field_id", "crop_instance_id", "decision_date", "story_id"],
    ),
    "chapters": (
        "field_story_chapters_v1.parquet",
        ["story_id", "chapter_number"],
    ),
    "windows": (
        "field_story_windows_v1.parquet",
        ["field_id", "crop_instance_id", "first_evidence_date", "story_id"],
    ),
    "hazard_daily": (
        "field_story_hazard_daily_v1.parquet",
        ["field_id", "crop_instance_id", "decision_date", "story_id", "hazard_family"],
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--partitions", type=_positive_int, default=64)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--memory-limit")
    parser.add_argument("--temp-dir", type=Path)
    return parser


def build_field_story_release(
    evidence_dir: Path,
    output_dir: Path,
    *,
    policy_path: Path | None = None,
    partitions: int = 64,
    threads: int = 8,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Write one immutable, hash-bound field-story release."""
    evidence = evidence_dir.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Field-story output already exists: {output}")
    if partitions < 1 or threads < 1:
        raise ValueError("partitions and threads must be positive")

    source_paths = {
        name: evidence / filename for name, filename in EVIDENCE_FILES.items()
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing V4 evidence ledgers: " + ", ".join(missing))
    source_manifest = evidence / "manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"V4 evidence manifest is missing: {source_manifest}")

    evidence_validation = validate_evidence_directory(evidence)
    selected_policy_path = (policy_path or POLICY_PATH).expanduser().resolve()
    policy = load_field_story_policy(selected_policy_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    configured_temp = temp_dir.expanduser().resolve() if temp_dir else None
    if configured_temp:
        configured_temp.mkdir(parents=True, exist_ok=True)

    totals = {name: 0 for name in ARTIFACTS}
    templates: dict[str, pd.DataFrame] = {}
    with TemporaryDirectory(
        prefix=f".{output.name}.partial-", dir=output.parent
    ) as temporary:
        transaction = Path(temporary)
        stage = transaction / "release"
        parts_root = transaction / "parts"
        input_root = transaction / "partitioned-inputs"
        stage.mkdir()
        parts_root.mkdir()
        input_root.mkdir()

        with duckdb.connect(":memory:") as connection:
            _configure_connection(
                connection,
                threads=threads,
                memory_limit=memory_limit,
                temp_dir=configured_temp,
            )
            for name, path in source_paths.items():
                connection.read_parquet(str(path)).create_view(f"{name}_input")
            _partition_inputs(connection, input_root, partitions)
            empty_inputs = {
                name: connection.execute(
                    f"SELECT * FROM {name}_input LIMIT 0"
                ).fetchdf()
                for name in EVIDENCE_FILES
            }

            for partition in range(partitions):
                frames = {
                    name: _read_partition(
                        connection,
                        input_root / name,
                        partition,
                        empty_inputs[name],
                    )
                    for name in EVIDENCE_FILES
                }
                artifacts = build_field_stories(
                    frames["crop"],
                    frames["pressure"],
                    frames["responses"],
                    policy=policy,
                )
                _write_partition(parts_root, partition, artifacts, totals, templates)
                print(
                    "Field-story partition "
                    f"{partition + 1}/{partitions} complete "
                    f"crop={len(frames['crop'])} pressure={len(frames['pressure'])} "
                    f"responses={len(frames['responses'])}",
                    file=sys.stderr,
                    flush=True,
                )

            _assemble_artifacts(connection, parts_root, stage, templates, totals)

        manifest = {
            "schema_version": "field-stories-v1/1",
            "mode": "deterministic_multi_hazard_field_story",
            "status": "complete",
            "created_at": _utc_now(),
            "source": {
                "evidence_manifest_sha256": _sha256(source_manifest),
                "evidence_directory": str(evidence),
            },
            "policy": {
                "version": policy.version,
                "calibration_status": policy.calibration_status,
                "sha256": _sha256(selected_policy_path),
            },
            "run": {
                "partitions": partitions,
                "threads": threads,
                "memory_limit": memory_limit,
            },
            "evidence_validation": evidence_validation,
            "semantics": {
                "story_identity": "fixed_field_crop_open_concern_interval",
                "multi_hazard": True,
                "machine_learning": False,
                "spatial_propagation_claimed": False,
            },
            "artifacts": {
                logical_name: {
                    "filename": filename,
                    "row_count": totals[logical_name],
                    "size_bytes": (stage / filename).stat().st_size,
                    "sha256": _sha256(stage / filename),
                }
                for logical_name, (filename, _) in ARTIFACTS.items()
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    return manifest


def _configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> None:
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET threads=?", [threads])
    if memory_limit:
        connection.execute("SET memory_limit=?", [memory_limit])
    if temp_dir:
        connection.execute("SET temp_directory=?", [str(temp_dir)])


def _partition_inputs(
    connection: duckdb.DuckDBPyConnection,
    input_root: Path,
    partitions: int,
) -> None:
    for name in EVIDENCE_FILES:
        destination = connection.sql(
            f"""
            SELECT *, CAST(
                hash(CAST(field_id AS VARCHAR), CAST(crop_instance_id AS VARCHAR))
                % {partitions} AS INTEGER
            ) AS story_partition
            FROM {name}_input
            """
        )
        destination.write_parquet(
            str(input_root / name),
            compression="zstd",
            partition_by=["story_partition"],
        )


def _read_partition(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    partition: int,
    empty: pd.DataFrame,
) -> pd.DataFrame:
    files = sorted((source / f"story_partition={partition}").glob("*.parquet"))
    if not files:
        return empty.copy()
    return connection.read_parquet(
        [str(path) for path in files], hive_partitioning=False
    ).fetchdf()


def _write_partition(
    parts_root: Path,
    partition: int,
    artifacts: FieldStoryArtifacts,
    totals: dict[str, int],
    templates: dict[str, pd.DataFrame],
) -> None:
    for logical_name in ARTIFACTS:
        frame = getattr(artifacts, logical_name)
        templates[logical_name] = frame.iloc[0:0].copy()
        totals[logical_name] += len(frame)
        if frame.empty:
            continue
        destination = parts_root / logical_name
        destination.mkdir(exist_ok=True)
        frame.to_parquet(
            destination / f"part-{partition:05d}.parquet",
            index=False,
            compression="zstd",
        )


def _assemble_artifacts(
    connection: duckdb.DuckDBPyConnection,
    parts_root: Path,
    stage: Path,
    templates: dict[str, pd.DataFrame],
    totals: dict[str, int],
) -> None:
    for logical_name, (filename, sort_columns) in ARTIFACTS.items():
        output = stage / filename
        files = sorted((parts_root / logical_name).glob("*.parquet"))
        if not files:
            templates[logical_name].to_parquet(output, index=False, compression="zstd")
        else:
            view = f"{logical_name}_parts"
            connection.read_parquet(
                [str(path) for path in files], union_by_name=True
            ).create_view(view)
            columns = {
                str(row[0]) for row in connection.execute(f"DESCRIBE {view}").fetchall()
            }
            order = ", ".join(
                f'"{column}"' for column in sort_columns if column in columns
            )
            query = f"SELECT * FROM {view}" + (f" ORDER BY {order}" if order else "")
            connection.sql(query).write_parquet(str(output), compression="zstd")
        actual = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(output)]
            ).fetchone()[0]
        )
        if actual != totals[logical_name]:
            raise RuntimeError(
                f"Assembled {logical_name} row count {actual} != {totals[logical_name]}"
            )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_field_story_release(
        args.evidence_dir,
        args.output_dir,
        policy_path=args.policy,
        partitions=args.partitions,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
