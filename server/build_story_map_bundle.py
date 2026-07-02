from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import duckdb
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping, shape


ARTIFACT_COPIES = {
    "map_frame_fields.parquet": "frame_fields.parquet",
    "event_story_cluster_labels.parquet": "cluster_labels.parquet",
    "event_windows.parquet": "event_windows.parquet",
    "story_day_membership.parquet": "story_day_membership.parquet",
    "manifest.json": "manifest.json",
    "comparison_summary.json": "comparison_summary.json",
    "event_story_report.md": "event_story_report.md",
}

REQUIRED_SOURCE_ARTIFACTS = (
    "map_frame_fields.parquet",
    "event_story_cluster_labels.parquet",
    "event_windows.parquet",
    "story_day_membership.parquet",
    "manifest.json",
)

REQUIRED_PARQUET_SCHEMAS = {
    "map_frame_fields.parquet": {
        "timeline_bucket",
        "field_id",
        "story_cluster_id",
        "max_risk_band",
        "hazard_signature",
        "response_signature",
        "reportable_day_count",
        "event_count",
        "max_risk_rank",
        "response_day_count",
    },
    "event_story_cluster_labels.parquet": {
        "story_cluster_id",
        "short_label",
        "max_risk_band",
        "hazard_signature",
        "response_signature",
        "event_count",
        "field_count",
        "crop_count",
        "median_window_span_days",
        "median_reportable_days",
    },
    "event_windows.parquet": {
        "field_id",
        "crop_name",
        "crop_season",
        "event_id",
        "event_start_date",
        "active_end_date",
        "max_risk_band",
        "hazard_signature",
        "stage_signature",
        "response_signature",
        "close_reason",
        "reportable_days",
        "window_span_days",
        "story_cluster_id",
    },
    "story_day_membership.parquet": {"field_id", "event_id", "story_cluster_id"},
}

STAGED_REQUIRED_PARQUET_SCHEMAS = {
    ARTIFACT_COPIES[source_name]: columns
    for source_name, columns in REQUIRED_PARQUET_SCHEMAS.items()
}

# A small amount of source geometry loss is tolerated for known upstream defects,
# but a bundle must preserve at least 95% of source and frame-field coverage by default.
DEFAULT_MIN_VALID_GEOMETRY_COVERAGE = 0.95
DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE = 0.95

GEOMETRY_SOURCE_NAMES = ("map_field_geometry.parquet", "field_geometry.parquet")

GEOMETRY_OUTPUT_COLUMNS = (
    "field_id",
    "geometry_geojson",
    "min_lon",
    "min_lat",
    "max_lon",
    "max_lat",
    "centroid_lon",
    "centroid_lat",
    "district",
    "sector",
    "cell",
    "village",
)

OPTIONAL_MOTIF_ARTIFACTS = (
    "motif_assignments.parquet",
    "motif_catalog.parquet",
    "event_motif_membership.parquet",
    "field_motif_timeline.parquet",
    "story_motifs.parquet",
    "motif_prototypes.parquet",
    "motif_labels.parquet",
    "motif_timeline.parquet",
    "llm_narration_queue.parquet",
    "llm_narration_queue.jsonl",
)

OPTIONAL_MONITORING_ARTIFACTS = (
    "crop_instances.parquet",
    "event_state_snapshots.parquet",
)

OWNED_OUTPUT_NAMES = frozenset(
    {
        "field_geometry.parquet",
        "geometry_profile.json",
        *ARTIFACT_COPIES.values(),
        *OPTIONAL_MOTIF_ARTIFACTS,
        *OPTIONAL_MONITORING_ARTIFACTS,
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable story-map bundle.")
    parser.add_argument("--run-dir", required=True, help="Event story run directory.")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min-valid-geometry-coverage",
        type=float,
        default=DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
        help="Minimum valid geometry rows divided by source rows, in [0, 1] (default: 0.95).",
    )
    parser.add_argument(
        "--min-frame-geometry-coverage",
        type=float,
        default=DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
        help="Minimum distinct frame fields joined to geometry, in [0, 1] (default: 0.95).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    build_bundle(
        run_dir,
        out_dir,
        overwrite=args.overwrite,
        min_valid_geometry_coverage=args.min_valid_geometry_coverage,
        min_frame_geometry_coverage=args.min_frame_geometry_coverage,
    )
    print(json.dumps({"status": "written", "out_dir": str(out_dir)}, indent=2))


def build_bundle(
    run_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
    min_valid_geometry_coverage: float = DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
    min_frame_geometry_coverage: float = DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
) -> None:
    run_dir = run_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    min_valid_geometry_coverage = validate_coverage_threshold(
        "min_valid_geometry_coverage", min_valid_geometry_coverage
    )
    min_frame_geometry_coverage = validate_coverage_threshold(
        "min_frame_geometry_coverage", min_frame_geometry_coverage
    )
    validate_source(run_dir, out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with advisory_build_lock(out_dir):
        if out_dir.exists() and not out_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {out_dir}")
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise SystemExit(f"Output directory is not empty: {out_dir}. Pass --overwrite.")

        with TemporaryDirectory(prefix=".story-map-build-", dir=out_dir.parent) as directory:
            transaction_dir = Path(directory)
            stage_dir = transaction_dir / "stage"
            stage_dir.mkdir()
            build_geometry(
                run_dir,
                stage_dir,
                profile_output=out_dir / "field_geometry.parquet",
                min_valid_coverage=min_valid_geometry_coverage,
            )
            copy_artifacts(run_dir, stage_dir)
            validate_staged_bundle(
                stage_dir,
                min_frame_geometry_coverage=min_frame_geometry_coverage,
            )
            mark_bundle_ready(stage_dir)
            install_staged_bundle(stage_dir, out_dir, transaction_dir / "backup")


def mark_bundle_ready(stage_dir: Path) -> None:
    """Mark readiness only after normalized geometry and joins have validated."""
    path = stage_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run = manifest.setdefault("run", {})
    run["viewer_ready"] = True
    run["viewer_bundle_required"] = False
    run["geometry_optimized"] = True
    output_candidates = {
        "field_geometry": "field_geometry.parquet",
        "frame_fields": "frame_fields.parquet",
        "cluster_labels": "cluster_labels.parquet",
        "event_windows": "event_windows.parquet",
        "story_day_membership": "story_day_membership.parquet",
        "geometry_profile": "geometry_profile.json",
        "crop_instances": "crop_instances.parquet",
        "event_state_snapshots": "event_state_snapshots.parquet",
        "motif_assignments": "motif_assignments.parquet",
        "motif_catalog": "motif_catalog.parquet",
    }
    manifest["outputs"] = {
        key: name for key, name in output_candidates.items() if (stage_dir / name).is_file()
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_coverage_threshold(name: str, value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0, 1].") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return normalized


@contextmanager
def advisory_build_lock(out_dir: Path) -> Iterator[None]:
    """Fail fast when another builder targets the same output directory."""
    lock_name = f".{out_dir.name or 'root'}.story-map-build.lock"
    lock_path = out_dir.parent / lock_name
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another story-map build is already targeting: {out_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_source(run_dir: Path, out_dir: Path) -> None:
    if run_dir == out_dir:
        raise ValueError("Run directory and output directory must be different.")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist or is not a directory: {run_dir}")

    geometry_path = pick(run_dir, *GEOMETRY_SOURCE_NAMES)
    with duckdb.connect(":memory:") as con:
        cursor = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(geometry_path)])
        columns = {str(item[0]) for item in (cursor.description or [])}
    if "field_id" not in columns or not columns.intersection(
        {"geometry_geojson", "geometry_text", "geometry_wkt", "geometry"}
    ):
        raise ValueError(
            f"Geometry artifact must contain field_id and a supported geometry column: {geometry_path}"
        )

    missing = [name for name in REQUIRED_SOURCE_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run directory is missing required source artifacts: {', '.join(missing)}"
        )
    validate_required_artifacts(
        run_dir,
        parquet_schemas=REQUIRED_PARQUET_SCHEMAS,
        manifest_name="manifest.json",
    )


def validate_required_artifacts(
    directory: Path,
    *,
    parquet_schemas: dict[str, set[str]],
    manifest_name: str,
) -> None:
    for name, required_columns in parquet_schemas.items():
        path = directory / name
        try:
            with duckdb.connect(":memory:") as con:
                cursor = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
                columns = {str(item[0]) for item in (cursor.description or [])}
                row_count = int(
                    con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
                )
        except Exception as exc:
            raise ValueError(f"Required Parquet artifact is unreadable: {path}") from exc
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise ValueError(
                f"Required Parquet artifact {path} is missing columns: {', '.join(missing_columns)}"
            )
        if row_count < 1:
            raise ValueError(f"Required Parquet artifact contains no rows: {path}")

    manifest_path = directory / manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Manifest is not readable valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest root must be a JSON object: {manifest_path}")


def value_is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
        try:
            if bool(missing):
                return False
        except ValueError:
            pass
    except (TypeError, ValueError):
        pass
    return not isinstance(value, str) or bool(value.strip())


def normalize_field_id(value: Any) -> str:
    if not value_is_present(value):
        raise ValueError("Geometry artifact contains an empty field_id.")
    field_id = str(value).strip()
    if not field_id:
        raise ValueError("Geometry artifact contains an empty field_id.")
    return field_id


def first_present_value(item: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = item.get(name)
        if value_is_present(value):
            return value
    raise ValueError("empty geometry")


def build_geometry(
    run_dir: Path,
    out_dir: Path,
    *,
    profile_output: Path | None = None,
    min_valid_coverage: float = DEFAULT_MIN_VALID_GEOMETRY_COVERAGE,
) -> None:
    src = pick(run_dir, *GEOMETRY_SOURCE_NAMES)
    with duckdb.connect(":memory:") as con:
        df = con.execute("SELECT * FROM read_parquet(?) ORDER BY field_id", [str(src)]).fetchdf()

    source_count = len(df)
    if source_count < 1:
        raise ValueError(f"Geometry artifact contains no source rows: {src}")

    rows: list[dict[str, Any]] = []
    failures = 0
    seen_field_ids: set[str] = set()
    for item in df.to_dict(orient="records"):
        field_id = normalize_field_id(item.get("field_id"))
        if field_id in seen_field_ids:
            raise ValueError(f"Geometry artifact contains duplicate field_id: {field_id}")
        seen_field_ids.add(field_id)
        try:
            geometry = item.get("geometry_geojson")
            if value_is_present(geometry):
                geometry_obj = json.loads(geometry) if isinstance(geometry, str) else geometry
                geom = shape(geometry_obj)
            else:
                text = first_present_value(item, "geometry_text", "geometry_wkt", "geometry")
                geom = shapely_wkt.loads(str(text))
                geometry_obj = mapping(geom)
            if geom.is_empty:
                raise ValueError("empty geometry")
            if not geom.is_valid:
                raise ValueError("invalid geometry")
            centroid = geom.centroid
            bounds = tuple(float(value) for value in geom.bounds)
            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                raise ValueError("geometry has non-finite bounds")
            if centroid.is_empty or not math.isfinite(float(centroid.x)) or not math.isfinite(float(centroid.y)):
                raise ValueError("geometry has a non-finite centroid")
            min_lon, min_lat, max_lon, max_lat = bounds
        except Exception:
            failures += 1
            continue
        rows.append(
            {
                "field_id": field_id,
                "geometry_geojson": json.dumps(geometry_obj, separators=(",", ":")),
                "min_lon": float(min_lon),
                "min_lat": float(min_lat),
                "max_lon": float(max_lon),
                "max_lat": float(max_lat),
                "centroid_lon": float(centroid.x),
                "centroid_lat": float(centroid.y),
                "district": item.get("district"),
                "sector": item.get("sector"),
                "cell": item.get("cell"),
                "village": item.get("village"),
            }
        )

    if not rows:
        raise ValueError(f"Geometry artifact contains no valid geometries: {src}")
    valid_coverage = len(rows) / source_count
    if valid_coverage < min_valid_coverage:
        raise ValueError(
            "Valid geometry coverage is below the configured minimum: "
            f"{len(rows)}/{source_count} ({valid_coverage:.2%}) < {min_valid_coverage:.2%}"
        )

    out = out_dir / "field_geometry.parquet"
    pd.DataFrame(rows, columns=GEOMETRY_OUTPUT_COLUMNS).to_parquet(out, index=False)
    profile = {
        "source": str(src),
        "output": str(profile_output or out),
        "source_field_count": source_count,
        "field_count": len(rows),
        "parse_failures": failures,
        "valid_geometry_coverage": valid_coverage,
        "min_valid_geometry_coverage": min_valid_coverage,
    }
    (out_dir / "geometry_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_artifacts(run_dir: Path, out_dir: Path) -> None:
    optional = (*OPTIONAL_MOTIF_ARTIFACTS, *OPTIONAL_MONITORING_ARTIFACTS)
    copies = {**ARTIFACT_COPIES, **{name: name for name in optional}}
    for source_name, target_name in copies.items():
        source = run_dir / source_name
        if source.exists():
            shutil.copy2(source, out_dir / target_name)


def validate_staged_bundle(
    stage_dir: Path,
    *,
    min_frame_geometry_coverage: float = DEFAULT_MIN_FRAME_GEOMETRY_COVERAGE,
) -> None:
    geometry_path = stage_dir / "field_geometry.parquet"
    with duckdb.connect(":memory:") as con:
        cursor = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(geometry_path)])
        columns = {str(item[0]) for item in (cursor.description or [])}
        row_count = int(
            con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(geometry_path)]).fetchone()[0]
        )
    missing_columns = sorted(set(GEOMETRY_OUTPUT_COLUMNS) - columns)
    if missing_columns:
        raise ValueError(
            f"Staged geometry is missing expected columns: {', '.join(missing_columns)}"
        )
    if row_count < 1:
        raise ValueError("Staged geometry must contain at least one valid feature.")

    required_targets = {ARTIFACT_COPIES[name] for name in REQUIRED_SOURCE_ARTIFACTS}
    missing_targets = sorted(name for name in required_targets if not (stage_dir / name).is_file())
    if missing_targets:
        raise FileNotFoundError(
            f"Staged bundle is missing required artifacts: {', '.join(missing_targets)}"
        )
    validate_required_artifacts(
        stage_dir,
        parquet_schemas=STAGED_REQUIRED_PARQUET_SCHEMAS,
        manifest_name="manifest.json",
    )

    frame_path = stage_dir / "frame_fields.parquet"
    with duckdb.connect(":memory:") as con:
        noncanonical_frame_field_count = int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM read_parquet(?)
                WHERE field_id IS NULL
                   OR TRIM(CAST(field_id AS VARCHAR)) = ''
                   OR CAST(field_id AS VARCHAR) <> TRIM(CAST(field_id AS VARCHAR))
                """,
                [str(frame_path)],
            ).fetchone()[0]
        )
        frame_field_count = int(
            con.execute(
                """
                SELECT COUNT(DISTINCT field_id)
                FROM read_parquet(?)
                """,
                [str(frame_path)],
            ).fetchone()[0]
        )
        joined_field_count = int(
            con.execute(
                """
                SELECT COUNT(DISTINCT f.field_id)
                FROM read_parquet(?) AS f
                JOIN read_parquet(?) AS g USING (field_id)
                """,
                [str(frame_path), str(geometry_path)],
            ).fetchone()[0]
        )
    if noncanonical_frame_field_count:
        raise ValueError(
            "Staged frame artifact contains empty or noncanonical field IDs; "
            "field_id values must be nonempty and have no surrounding whitespace."
        )
    if frame_field_count < 1:
        raise ValueError("Staged frame artifact contains no nonempty field IDs.")
    frame_geometry_coverage = joined_field_count / frame_field_count
    if frame_geometry_coverage < min_frame_geometry_coverage:
        raise ValueError(
            "Frame-to-geometry field coverage is below the configured minimum: "
            f"{joined_field_count}/{frame_field_count} ({frame_geometry_coverage:.2%}) "
            f"< {min_frame_geometry_coverage:.2%}"
        )

    profile_path = stage_dir / "geometry_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.update(
        {
            "frame_field_count": frame_field_count,
            "joined_frame_field_count": joined_field_count,
            "frame_geometry_coverage": frame_geometry_coverage,
            "min_frame_geometry_coverage": min_frame_geometry_coverage,
        }
    )
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def install_staged_bundle(stage_dir: Path, out_dir: Path, backup_dir: Path) -> None:
    """Replace owned outputs transactionally while preserving unrelated files."""
    staged_files = sorted(path for path in stage_dir.iterdir() if path.is_file())
    unexpected = sorted(path.name for path in staged_files if path.name not in OWNED_OUTPUT_NAMES)
    if unexpected:
        raise ValueError(f"Staged bundle contains unexpected outputs: {', '.join(unexpected)}")

    out_existed = out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir()
    backed_up: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for name in sorted(OWNED_OUTPUT_NAMES):
            destination = out_dir / name
            if destination.exists() or destination.is_symlink():
                backup = backup_dir / name
                destination.replace(backup)
                backed_up.append((backup, destination))

        summaries = out_dir / "gpu_summaries"
        if summaries.exists() or summaries.is_symlink():
            backup = backup_dir / summaries.name
            summaries.replace(backup)
            backed_up.append((backup, summaries))

        for source in staged_files:
            destination = out_dir / source.name
            source.replace(destination)
            installed.append(destination)
    except BaseException:
        for destination in reversed(installed):
            remove_path(destination)
        for backup, destination in reversed(backed_up):
            backup.replace(destination)
        if not out_existed and not any(out_dir.iterdir()):
            out_dir.rmdir()
        raise


def clear_known_outputs(out_dir: Path) -> None:
    """Remove only files this builder owns so stale run artifacts cannot survive."""
    if not out_dir.exists():
        return
    for name in OWNED_OUTPUT_NAMES:
        remove_path(out_dir / name)
    remove_path(out_dir / "gpu_summaries")


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def pick(directory: Path, *names: str) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist in {directory}: {', '.join(names)}")


if __name__ == "__main__":
    main()
