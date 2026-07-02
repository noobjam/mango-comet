from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-optional story-map summaries.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--engine",
        choices=["auto", "dask-cudf", "cudf", "pandas"],
        default="auto",
        help="Use RAPIDS on the GPU VM when available; pandas fallback is portable.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = choose_engine(args.engine)
    if engine == "dask-cudf":
        result = run_dask_cudf(run_dir, out_dir)
    elif engine == "cudf":
        result = run_cudf(run_dir, out_dir)
    else:
        result = run_pandas(run_dir, out_dir)
    result["engine"] = engine
    (out_dir / "gpu_precompute_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def choose_engine(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import dask_cudf  # noqa: F401
        import dask_cuda  # noqa: F401

        return "dask-cudf"
    except Exception:
        pass
    try:
        import cudf  # noqa: F401

        return "cudf"
    except Exception:
        return "pandas"


def run_dask_cudf(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    import dask_cudf

    cluster = LocalCUDACluster()
    client = Client(cluster)
    try:
        frame = dask_cudf.read_parquet(str(pick(run_dir, "frame_fields.parquet", "map_frame_fields.parquet")))
        timeline = frame.groupby("timeline_bucket").agg(
            {
                "field_id": "nunique",
                "story_cluster_id": "nunique",
                "reportable_day_count": "sum",
                "event_count": "sum",
                "max_risk_rank": "max",
            }
        )
        timeline = timeline.reset_index().compute()
        timeline.to_parquet(out_dir / "timeline_summary.parquet", index=False)

        story_days_path = pick(run_dir, "story_day_membership.parquet")
        story_days = dask_cudf.read_parquet(str(story_days_path))
        cluster_summary = story_days.groupby("story_cluster_id").agg(
            {
                "field_id": "nunique",
                "event_id": "nunique",
                "risk_rank": "max",
                "is_reportable_story_day": "sum",
            }
        )
        cluster_summary = cluster_summary.reset_index().compute()
        cluster_summary.to_parquet(out_dir / "cluster_day_summary.parquet", index=False)
    finally:
        client.close()
        cluster.close()
    return {
        "status": "written",
        "outputs": [
            str(out_dir / "timeline_summary.parquet"),
            str(out_dir / "cluster_day_summary.parquet"),
        ],
    }


def run_cudf(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    import cudf

    frame = cudf.read_parquet(str(pick(run_dir, "frame_fields.parquet", "map_frame_fields.parquet")))
    timeline = frame.groupby("timeline_bucket").agg(
        {
            "field_id": "nunique",
            "story_cluster_id": "nunique",
            "reportable_day_count": "sum",
            "event_count": "sum",
            "max_risk_rank": "max",
        }
    )
    timeline = timeline.reset_index()
    timeline.to_parquet(str(out_dir / "timeline_summary.parquet"), index=False)

    story_days = cudf.read_parquet(str(pick(run_dir, "story_day_membership.parquet")))
    cluster_summary = story_days.groupby("story_cluster_id").agg(
        {
            "field_id": "nunique",
            "event_id": "nunique",
            "risk_rank": "max",
            "is_reportable_story_day": "sum",
        }
    )
    cluster_summary = cluster_summary.reset_index()
    cluster_summary.to_parquet(str(out_dir / "cluster_day_summary.parquet"), index=False)
    return {
        "status": "written",
        "outputs": [
            str(out_dir / "timeline_summary.parquet"),
            str(out_dir / "cluster_day_summary.parquet"),
        ],
    }


def run_pandas(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_parquet(pick(run_dir, "frame_fields.parquet", "map_frame_fields.parquet"))
    timeline = (
        frame.groupby("timeline_bucket")
        .agg(
            field_id=("field_id", "nunique"),
            story_cluster_id=("story_cluster_id", "nunique"),
            reportable_day_count=("reportable_day_count", "sum"),
            event_count=("event_count", "sum"),
            max_risk_rank=("max_risk_rank", "max"),
        )
        .reset_index()
    )
    timeline.to_parquet(out_dir / "timeline_summary.parquet", index=False)

    story_days = pd.read_parquet(pick(run_dir, "story_day_membership.parquet"))
    cluster_summary = (
        story_days.groupby("story_cluster_id")
        .agg(
            field_id=("field_id", "nunique"),
            event_id=("event_id", "nunique"),
            risk_rank=("risk_rank", "max"),
            is_reportable_story_day=("is_reportable_story_day", "sum"),
        )
        .reset_index()
    )
    cluster_summary.to_parquet(out_dir / "cluster_day_summary.parquet", index=False)
    return {
        "status": "written",
        "outputs": [
            str(out_dir / "timeline_summary.parquet"),
            str(out_dir / "cluster_day_summary.parquet"),
        ],
    }


def pick(directory: Path, *names: str) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist in {directory}: {', '.join(names)}")


if __name__ == "__main__":
    main()
