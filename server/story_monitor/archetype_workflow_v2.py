"""Atomic Phase A workflows for event-anchor archetype build and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd

from .anchor_features_v2 import FEATURE_VERSION, write_event_anchors
from .archetypes_v2 import (
    ArchetypeConfig,
    discover_archetypes,
    evaluate_archetype_model,
)


def build_archetype_model(
    generation_dir: Path,
    output_dir: Path,
    *,
    training_cutoff: str,
    config: ArchetypeConfig,
    threads: int = 16,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, Any]:
    """Build anchors and a frozen training model, then publish atomically."""
    generation_dir = generation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _reject_descendant(output_dir, generation_dir, "model output", "source generation")
    if temp_dir is not None:
        _reject_descendant(
            temp_dir.expanduser().resolve(), generation_dir,
            "DuckDB temp directory", "source generation",
        )
    if output_dir.exists():
        raise FileExistsError(f"Immutable archetype model already exists: {output_dir}")
    generation_manifest_path = generation_dir / "manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    run = generation_manifest.get("run") or {}
    if str(run.get("status") or "") != "complete":
        raise ValueError("Archetype V2 requires a completed immutable generation")
    generation_as_of = date.fromisoformat(str(run.get("as_of_date") or "")[:10])
    cutoff_date = date.fromisoformat(str(training_cutoff)[:10])
    if cutoff_date >= generation_as_of:
        raise ValueError("training cutoff must precede generation as-of date for temporal holdout")
    policy = generation_manifest.get("policy") or {}
    policy_version = str(policy.get("version") or "")
    policy_sha256 = str(policy.get("sha256") or "")
    if not policy_version or not policy_sha256:
        raise ValueError("Generation manifest must contain policy.version and policy.sha256")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".archetype-v2-build-", dir=output_dir.parent) as temporary:
        transaction = Path(temporary)
        anchors_path = transaction / "event_anchors.parquet"
        stage = transaction / output_dir.name
        write_event_anchors(
            generation_dir,
            anchors_path,
            threads=threads,
            memory_limit=memory_limit,
            temp_dir=temp_dir,
        )
        training_rows = _load_anchor_split(
            anchors_path, training_cutoff=training_cutoff, split="training"
        )
        if training_rows.empty:
            raise ValueError("No eligible event anchors exist on or before the training cutoff")
        manifest = discover_archetypes(
            training_rows,
            stage,
            config=config,
            training_cutoff=training_cutoff,
            policy_version=policy_version,
            policy_sha256=policy_sha256,
        )
        os.replace(anchors_path, stage / "event_anchors.parquet")
        anchor_counts = _anchor_counts(stage / "event_anchors.parquet", training_cutoff)
        expected_event_count = run.get("event_count")
        if (
            expected_event_count is not None
            and int(expected_event_count) != anchor_counts["total_events"]
        ):
            raise ValueError(
                "Anchor ledger count does not match generation manifest event_count"
            )
        manifest.update(
            {
                "generation_id": str((generation_manifest.get("run") or {}).get("generation_id") or ""),
                "generation_as_of_date": str(
                    (generation_manifest.get("run") or {}).get("as_of_date") or ""
                ),
                "generation_manifest_sha256": _file_sha256(generation_manifest_path),
                "anchor_counts": anchor_counts,
                "implementation_sha256": _implementation_sha256(),
                "software_versions": _software_versions(),
            }
        )
        manifest["artifacts"] = _artifact_hashes(stage)
        _write_json(stage / "archetype_manifest.json", manifest)
        os.replace(stage, output_dir)
    return {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "model_dir": str(output_dir),
        "model_version": manifest["model_version"],
        "feature_version": FEATURE_VERSION,
        "anchor_counts": anchor_counts,
        "quality_gate_status": "not_evaluated",
        "warning": "Diagnostic unreviewed model; do not export it to the map.",
    }


def evaluate_archetype_release(
    model_dir: Path,
    output_dir: Path,
    *,
    stability_runs: int = 2,
) -> dict[str, Any]:
    """Evaluate temporal holdout and stability into a separate immutable directory."""
    if stability_runs != 2:
        raise ValueError("The V2 contract requires exactly two subsample stability runs")
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _reject_descendant(output_dir, model_dir, "evaluation output", "frozen model")
    if output_dir.exists():
        raise FileExistsError(f"Immutable evaluation directory already exists: {output_dir}")
    manifest_path = model_dir / "archetype_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_implementation = _implementation_sha256()
    if str(manifest.get("implementation_sha256") or "") != current_implementation:
        raise ValueError(
            "V2 implementation hash differs from the frozen model; evaluate with the build commit"
        )
    current_versions = _software_versions()
    if (manifest.get("software_versions") or {}) != current_versions:
        raise ValueError(
            "V2 software versions differ from the frozen model; evaluate in the build environment"
        )
    _verify_artifact_hashes(model_dir, manifest.get("artifacts") or {})
    training_cutoff = str(manifest["training_cutoff"])
    anchors_path = model_dir / "event_anchors.parquet"
    training_rows = _load_anchor_split(
        anchors_path, training_cutoff=training_cutoff, split="training"
    )
    holdout_rows = _load_anchor_split(
        anchors_path, training_cutoff=training_cutoff, split="holdout"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".archetype-v2-eval-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / output_dir.name
        stage.mkdir()
        report = evaluate_archetype_model(
            training_rows, holdout_rows, model_dir, output_dir=stage
        )
        report["gates"]["hard"]["checks"].update(
            _anchor_hard_checks(anchors_path, training_cutoff)
        )
        report["gates"]["hard"]["checks"]["anchor_counts_match_manifest"] = bool(
            _anchor_counts(anchors_path, training_cutoff)
            == (manifest.get("anchor_counts") or {})
        )
        from .archetype_stability_v2 import evaluate_subsample_stability

        stability = evaluate_subsample_stability(
            training_rows,
            model_dir,
            output_dir=stage,
            runs=stability_runs,
        )
        quality = report["gates"]["quality"]["checks"]
        stability_checks = stability["gates"]["checks"]
        quality["stability_ari"] = bool(
            stability_checks["minimum_adjusted_rand_index"]
        )
        quality["stability_mutual_non_noise_ari"] = bool(
            stability_checks["minimum_mutual_non_noise_adjusted_rand_index"]
        )
        quality["stability_matched_jaccard"] = bool(
            stability_checks["conservative_matched_jaccard"]
        )
        quality["stability_support"] = bool(
            stability_checks["supported_reference_fraction"]
        )
        quality.pop("subsample_stability", None)
        report["gates"]["quality"]["passed"] = all(quality.values())
        report["metrics"].update(stability["metrics"])
        report["metrics"]["stability_status"] = "complete"
        report["model_artifact_hashes_verified"] = True
        report["model_manifest_sha256"] = _file_sha256(manifest_path)
        report["warning"] = (
            "Statistical gates do not constitute agronomic validation or map publication approval."
        )
        training_assignments = pd.read_parquet(
            stage / "training_frozen_assignments.parquet"
        )
        holdout_assignments = pd.read_parquet(stage / "holdout_assignments.parquet")
        training_assignments["split"] = "training"
        holdout_assignments["split"] = "holdout"
        combined = pd.concat([training_assignments, holdout_assignments], ignore_index=True, sort=False)
        if not combined["event_id"].astype(str).is_unique:
            raise ValueError("Evaluation produced multiple archetype assignments for one event")
        report["gates"]["hard"]["checks"]["combined_assignment_unique"] = True
        report["gates"]["hard"]["checks"]["model_artifact_hashes"] = True
        report["gates"]["hard"]["passed"] = all(
            report["gates"]["hard"]["checks"].values()
        )
        combined.to_parquet(stage / "event_archetype_assignments.parquet", index=False)
        _write_json(stage / "evaluation.json", report)
        _write_json(
            stage / "evaluation_manifest.json",
            {
                "status": "complete",
                "phase": "phase_a_diagnostic",
                "model_version": manifest["model_version"],
                "model_manifest_sha256": _file_sha256(manifest_path),
                "implementation_sha256": current_implementation,
                "software_versions": current_versions,
                "artifacts": _artifact_hashes(stage),
            },
        )
        os.replace(stage, output_dir)
    return {
        "status": "complete",
        "phase": "phase_a_diagnostic",
        "evaluation_dir": str(output_dir),
        "model_version": manifest["model_version"],
        "hard_gates_passed": bool(report["gates"]["hard"]["passed"]),
        "quality_gates_passed": bool(report["gates"]["quality"]["passed"]),
        "metrics": report["metrics"],
        "warning": report["warning"],
    }


def _load_anchor_split(
    path: Path, *, training_cutoff: str, split: str
) -> pd.DataFrame:
    operator = "<=" if split == "training" else ">"
    if split not in {"training", "holdout"}:
        raise ValueError("split must be training or holdout")
    with duckdb.connect(":memory:") as connection:
        return connection.execute(
            f"""
            SELECT *
            FROM read_parquet(?)
            WHERE eligible_for_training
              AND CAST(anchor_date AS DATE) {operator} CAST(? AS DATE)
            ORDER BY hazard_family, event_id
            """,
            [str(path), training_cutoff],
        ).fetchdf()


def _anchor_counts(path: Path, training_cutoff: str) -> dict[str, Any]:
    with duckdb.connect(":memory:") as connection:
        total, eligible, training, holdout = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT_IF(eligible_for_training),
                COUNT_IF(eligible_for_training AND anchor_date <= CAST(? AS DATE)),
                COUNT_IF(eligible_for_training AND anchor_date > CAST(? AS DATE))
            FROM read_parquet(?)
            """,
            [training_cutoff, training_cutoff, str(path)],
        ).fetchone()
        outcomes = connection.execute(
            """
            SELECT anchor_outcome, COUNT(*) AS event_count
            FROM read_parquet(?)
            GROUP BY anchor_outcome
            ORDER BY anchor_outcome
            """,
            [str(path)],
        ).fetchall()
    return {
        "total_events": int(total),
        "eligible_events": int(eligible),
        "training_events": int(training),
        "holdout_events": int(holdout),
        "outcomes": {str(name): int(count) for name, count in outcomes},
    }


def _anchor_hard_checks(path: Path, training_cutoff: str) -> dict[str, bool]:
    with duckdb.connect(":memory:") as connection:
        total, distinct_events, bad_anchor, leakage, train_count, holdout_count = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT event_id),
                COUNT_IF(eligible_for_training AND anchor_date IS NULL),
                COUNT_IF(
                    eligible_for_training
                    AND (
                        evidence_max_date > anchor_date
                        OR spectral_source_max_date > anchor_date
                    )
                ),
                COUNT_IF(eligible_for_training AND anchor_date <= CAST(? AS DATE)),
                COUNT_IF(eligible_for_training AND anchor_date > CAST(? AS DATE))
            FROM read_parquet(?)
            """,
            [training_cutoff, training_cutoff, str(path)],
        ).fetchone()
    return {
        "one_anchor_ledger_row_per_event": int(total) == int(distinct_events),
        "eligible_anchor_dates_present": int(bad_anchor) == 0,
        "no_post_anchor_feature_evidence": int(leakage) == 0,
        "training_split_nonempty": int(train_count) > 0,
        "holdout_split_nonempty": int(holdout_count) > 0,
    }


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: _file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in {"archetype_manifest.json", "evaluation_manifest.json"}
    }


def _reject_descendant(path: Path, protected: Path, label: str, protected_label: str) -> None:
    if path == protected or path.is_relative_to(protected):
        raise ValueError(f"{label} must not be inside the {protected_label}: {path}")


def _verify_artifact_hashes(directory: Path, expected: dict[str, str]) -> None:
    if not expected:
        raise ValueError("Model manifest has no artifact hashes")
    required = {
        "feature_schema.json",
        "event_anchors.parquet",
        "prototypes.parquet",
        "archetype_catalog.parquet",
        "training_assignments.parquet",
    }
    missing = sorted(required - set(expected))
    if missing:
        raise ValueError("Model manifest is missing artifact hashes: " + ", ".join(missing))
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or _file_sha256(path) != str(digest):
            raise ValueError(f"Model artifact hash mismatch: {name}")


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).with_name("anchor_features_v2.py"),
        Path(__file__).with_name("archetypes_v2.py"),
        Path(__file__).with_name("archetype_stability_v2.py"),
        Path(__file__),
        Path(__file__).parents[1] / "weekly_story_monitor.py",
        Path(__file__).parents[1] / "requirements.txt",
        Path(__file__).parents[1] / "requirements-gpu-optional.txt",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _software_versions() -> dict[str, str | None]:
    import numpy
    import pandas
    import pyarrow
    import sklearn

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__,
        "cupy": None,
        "cuml": None,
    }
    try:
        import cupy
        import cuml
    except ImportError:
        pass
    else:
        versions["cupy"] = str(cupy.__version__)
        versions["cuml"] = str(cuml.__version__)
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
