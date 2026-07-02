"""Export a frozen motif model as a viewer-compatible immutable generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pandas as pd

from .motifs import assign_frozen_motifs
from .prefix_features import load_training_prefixes


FEATURE_VERSION = "causal_prefix_features_v1"


def export_motif_generation(
    generation_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> Path:
    generation_dir = generation_dir.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == generation_dir or generation_dir in output_dir.parents:
        raise ValueError("Motif output must be outside the immutable source generation.")
    if output_dir.exists():
        raise FileExistsError(f"Motif generation already exists: {output_dir}")
    source_manifest = json.loads((generation_dir / "manifest.json").read_text(encoding="utf-8"))
    schema = _validate_model_compatibility(source_manifest, model_dir)
    prefixes = load_training_prefixes(generation_dir, sample_age_buckets=False)
    assignments = assign_frozen_motifs(prefixes, model_dir)
    assignment_columns = [
        "event_id",
        "timeline_bucket",
        "event_age_days",
        "motif_id",
        "assignment_method",
        "assignment_distance",
        "distance_ratio",
        "assignment_margin",
        "runner_up_motif_id",
        "candidate_motif_id",
        "assignment_reason",
        "motif_model_version",
    ]
    assignments = assignments[assignment_columns].copy()
    assignments["timeline_bucket"] = assignments["timeline_bucket"].astype(str).str[:10]
    catalog = pd.read_parquet(model_dir / "motif_catalog.parquet")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".motif-export-", dir=output_dir.parent) as directory:
        stage = Path(directory) / output_dir.name
        shutil.copytree(generation_dir, stage)
        assignments.to_parquet(stage / "motif_assignments.parquet", index=False, compression="zstd")
        catalog.to_parquet(stage / "motif_catalog.parquet", index=False, compression="zstd")
        _write_enriched_frames(stage, assignments)
        _write_motif_labels(stage, catalog)
        _write_enriched_memberships(stage, assignments)
        _write_enriched_events(stage, assignments)
        source_manifest.setdefault("run", {})["motif_count"] = int(catalog["motif_id"].nunique())
        source_manifest["run"]["motif_model_version"] = schema["model_version"]
        source_manifest["motifs"] = {
            "model_version": schema["model_version"],
            "assignment": "frozen_prototype_radius_v1",
            "novel_outcome": "novel_unassigned",
            "catalog_status": "discovered_unreviewed",
        }
        source_manifest.setdefault("semantics", {})["story_cluster_id_alias"] = "motif_id"
        source_manifest["semantics"]["event_id_retained_separately"] = True
        (stage / "manifest.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
    return output_dir


def _validate_model_compatibility(
    source_manifest: dict[str, object], model_dir: Path
) -> dict[str, object]:
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    training = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    source_policy = str((source_manifest.get("policy") or {}).get("version") or "")
    source_policy_sha = str((source_manifest.get("policy") or {}).get("sha256") or "")
    model_policy = str(schema.get("policy_version") or "")
    if not source_policy or source_policy != model_policy:
        raise ValueError(
            "Motif model policy does not match the source generation: "
            f"generation={source_policy or 'missing'}, model={model_policy or 'missing'}."
        )
    model_policy_sha = str(schema.get("policy_sha256") or "")
    if source_policy_sha and source_policy_sha != model_policy_sha:
        raise ValueError(
            "Motif model policy hash does not match the source generation: "
            f"generation={source_policy_sha}, model={model_policy_sha or 'missing'}."
        )
    if schema.get("version") != FEATURE_VERSION:
        raise ValueError(
            f"Unsupported motif feature schema {schema.get('version')!r}; expected {FEATURE_VERSION!r}."
        )
    source_feature = str(
        (source_manifest.get("semantics") or {}).get("causal_prefix_feature_version") or ""
    )
    if source_feature and source_feature != schema.get("version"):
        raise ValueError(
            "Motif feature schema does not match the source generation: "
            f"generation={source_feature}, model={schema.get('version')}."
        )
    if (
        training.get("model_version") != schema.get("model_version")
        or training.get("policy_version") != model_policy
        or str(training.get("policy_sha256") or "") != model_policy_sha
        or training.get("feature_version") != schema.get("version")
    ):
        raise ValueError("Motif model metadata is internally inconsistent.")
    return schema


def _write_enriched_frames(stage: Path, assignments: pd.DataFrame) -> None:
    frames = pd.read_parquet(stage / "map_frame_fields.parquet")
    frames["timeline_bucket"] = frames["timeline_bucket"].astype(str).str[:10]
    enriched = frames.merge(
        assignments,
        on=["event_id", "timeline_bucket"],
        how="left",
        validate="many_to_one",
    )
    assignment_values = [
        column for column in assignments.columns
        if column not in {"event_id", "timeline_bucket"}
    ]
    ordered = enriched.sort_values(
        ["event_id", "timeline_bucket"], kind="mergesort"
    )
    carried = ordered.groupby("event_id", sort=False)[assignment_values].ffill()
    carryable = ordered["event_state"].isin({"DATA_GAP", "CLOSED_SEASON_BOUNDARY"})
    missing_carryable = carryable & ordered["motif_id"].isna()
    ordered.loc[missing_carryable, assignment_values] = carried.loc[
        missing_carryable, assignment_values
    ]
    carried_gap = (
        missing_carryable
        & ordered["event_state"].eq("DATA_GAP")
        & ordered["motif_id"].notna()
    )
    ordered.loc[carried_gap, "assignment_reason"] = "carried_through_data_gap"
    carried_boundary = (
        missing_carryable
        & ordered["event_state"].eq("CLOSED_SEASON_BOUNDARY")
        & ordered["motif_id"].notna()
    )
    ordered.loc[carried_boundary, "assignment_reason"] = "carried_to_season_boundary"
    enriched = ordered.sort_index()
    enriched["motif_id"] = enriched["motif_id"].fillna("novel_unassigned")
    enriched["story_cluster_id"] = enriched["motif_id"]
    enriched.to_parquet(stage / "map_frame_fields.parquet", index=False, compression="zstd")


def _write_motif_labels(stage: Path, catalog: pd.DataFrame) -> None:
    frames = pd.read_parquet(stage / "map_frame_fields.parquet")
    stats = frames.groupby("motif_id", as_index=False).agg(
        max_risk_rank=("max_risk_rank", "max"),
        hazard_signature=("hazard_signature", "first"),
        response_signature=("response_signature", "first"),
        event_count=("event_id", "nunique"),
        field_count=("field_id", "nunique"),
        crop_count=("crop_name", "nunique"),
        median_window_span_days=("event_age_days", "median"),
        median_reportable_days=("reportable_day_count", "median"),
    )
    labels = stats.merge(
        catalog[["motif_id", "label", "hazard_family"]], on="motif_id", how="left"
    )
    labels["story_cluster_id"] = labels["motif_id"]
    labels["short_label"] = labels["label"].fillna(
        labels["hazard_signature"].astype(str).str.replace("_", " ").str.title()
    )
    labels["max_risk_band"] = labels["max_risk_rank"].map(
        {0: "NONE", 1: "LOW", 2: "LOW-MED", 3: "MED-HIGH", 4: "HIGH"}
    )
    labels["motif_family"] = labels["hazard_family"].fillna(labels["hazard_signature"])
    columns = [
        "story_cluster_id", "short_label", "max_risk_band", "hazard_signature",
        "motif_family", "response_signature", "event_count", "field_count", "crop_count",
        "median_window_span_days", "median_reportable_days",
    ]
    labels[columns].to_parquet(
        stage / "event_story_cluster_labels.parquet", index=False, compression="zstd"
    )


def _write_enriched_memberships(stage: Path, assignments: pd.DataFrame) -> None:
    membership = pd.read_parquet(stage / "story_day_membership.parquet")
    membership["timeline_bucket"] = pd.to_datetime(membership["observation_date"]).dt.to_period("W-SUN").dt.start_time.astype(str).str[:10]
    frames = pd.read_parquet(stage / "map_frame_fields.parquet")
    frames["timeline_bucket"] = frames["timeline_bucket"].astype(str).str[:10]
    assignment_keys = frames[["event_id", "timeline_bucket", "motif_id"]].drop_duplicates(
        ["event_id", "timeline_bucket"]
    )
    membership = membership.merge(
        assignment_keys, on=["event_id", "timeline_bucket"], how="left", validate="many_to_one"
    )
    membership["story_cluster_id"] = membership["motif_id"].fillna("novel_unassigned")
    membership.to_parquet(stage / "story_day_membership.parquet", index=False, compression="zstd")


def _write_enriched_events(stage: Path, assignments: pd.DataFrame) -> None:
    events = pd.read_parquet(stage / "event_windows.parquet")
    latest = assignments.sort_values("timeline_bucket").drop_duplicates("event_id", keep="last")
    events = events.drop(columns=["story_cluster_id"], errors="ignore").merge(
        latest[["event_id", "motif_id", "motif_model_version"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    events["story_cluster_id"] = events["motif_id"].fillna("novel_unassigned")
    events.to_parquet(stage / "event_windows.parquet", index=False, compression="zstd")
