"""Fail-closed input and artifact validation for the Archetype V2 runner."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .runner_process import RunnerError, load_json


__all__ = [
    "discover_generation",
    "ensure_clean_repository",
    "job_paths",
    "resolve_rapids_python",
    "validate_evaluation",
    "validate_generation",
    "validate_model",
]


GENERATION_FILES = (
    "manifest.json",
    "event_windows.parquet",
    "story_day_membership.parquet",
    "daily_causal_signals.parquet",
)
MODEL_FILES = (
    "archetype_manifest.json",
    "feature_schema.json",
    "event_anchors.parquet",
    "prototypes.parquet",
    "archetype_catalog.parquet",
    "training_assignments.parquet",
)
EVALUATION_FILES = (
    "evaluation.json",
    "evaluation_manifest.json",
    "event_archetype_assignments.parquet",
    "evaluation_by_hazard.parquet",
    "holdout_assignments.parquet",
    "prototype_overlap.parquet",
    "subsample_stability.parquet",
    "subsample_stability.json",
    "training_frozen_assignments.parquet",
)
MODEL_ARTIFACTS = frozenset(MODEL_FILES[1:])
EVALUATION_ARTIFACTS = frozenset(name for name in EVALUATION_FILES if name != "evaluation_manifest.json")


def job_paths(root: Path, tag: str, training_cutoff: str) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", tag):
        raise RunnerError("Job tag must be 1-80 characters using only letters, digits, dot, underscore, or dash")
    job = root / "jobs" / f"archetype_v2_{tag}"
    return {
        "job_dir": str(job),
        "model_dir": str(root / "models" / f"archetype_v2_anchor21_train_{training_cutoff[:4]}_{tag}"),
        "evaluation_dir": str(root / "evaluations" / f"archetype_v2_{tag}"),
        "runner_log": str(job / "runner.log"),
        "build_json": str(job / "build.json"),
        "build_stderr": str(job / "build.stderr.log"),
        "evaluation_json": str(job / "evaluation.json"),
        "evaluation_stderr": str(job / "evaluation.stderr.log"),
    }


def _require_files(directory: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    empty = [name for name in names if (directory / name).is_file() and not (directory / name).stat().st_size]
    if missing or empty:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if empty:
            details.append(f"empty={','.join(empty)}")
        raise RunnerError(f"Invalid {label} at {directory}: {'; '.join(details)}")


def validate_generation(path: Path, *, as_of: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    _require_files(path, GENERATION_FILES, "generation")
    manifest = load_json(path / "manifest.json")
    run = manifest.get("run") or {}
    inputs = manifest.get("input") or {}
    problems = []
    if run.get("status") != "complete":
        problems.append("run.status is not complete")
    if run.get("immutable") is not True:
        problems.append("run.immutable is not true")
    if inputs.get("max_fields") is not None:
        problems.append("input.max_fields is not null (bounded smoke generation)")
    if as_of and str(run.get("as_of_date") or "")[:10] != as_of:
        problems.append(f"run.as_of_date does not equal {as_of}")
    if not str(run.get("generation_id") or ""):
        problems.append("run.generation_id is missing")
    policy = manifest.get("policy") or {}
    if not policy.get("version") or not policy.get("sha256"):
        problems.append("policy version/hash is missing")
    if problems:
        raise RunnerError(f"Invalid generation at {path}: {'; '.join(problems)}")
    return manifest


def discover_generation(root: Path, *, as_of: str) -> Path:
    candidates: list[Path] = []
    for manifest_path in sorted((root / "generations").glob("*/manifest.json")):
        try:
            validate_generation(manifest_path.parent, as_of=as_of)
        except RunnerError:
            continue
        candidates.append(manifest_path.parent.resolve())
    if not candidates:
        raise RunnerError(f"No full immutable generation for {as_of} under {root / 'generations'}")
    if len(candidates) > 1:
        listing = "\n  ".join(str(path) for path in candidates)
        raise RunnerError(f"Multiple full generations match {as_of}; pass --generation-dir:\n  {listing}")
    return candidates[0]


def resolve_rapids_python(root: Path, explicit: Path | None) -> Path:
    candidate: Path | None = explicit
    if candidate is None and os.environ.get("RAPIDS_PYTHON"):
        candidate = Path(os.environ["RAPIDS_PYTHON"])
    if candidate is None:
        pointer = root / "logs" / "latest_rapids_python.txt"
        if pointer.is_file():
            candidate = Path(pointer.read_text(encoding="utf-8").strip())
    if candidate is None:
        raise RunnerError("RAPIDS Python is unknown; pass --rapids-python or restore latest_rapids_python.txt")
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RunnerError(f"RAPIDS Python is not executable: {candidate}")
    return candidate


def ensure_clean_repository(repo: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RunnerError(f"Cannot inspect repository state: {result.stderr.strip()}")
    if result.stdout.strip():
        raise RunnerError("Tracked repository changes exist; commit/stash them before a reproducible V2 run")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hashes(directory: Path, artifacts: Any, expected: frozenset[str]) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise RunnerError(f"Artifact hashes are missing from {directory}")
    if set(artifacts) != expected:
        missing = sorted(expected - set(artifacts))
        extra = sorted(set(artifacts) - expected)
        raise RunnerError(f"Artifact hash registry differs at {directory}: missing={missing}; extra={extra}")
    for name, expected in sorted(artifacts.items()):
        path = directory / str(name)
        if not path.is_file():
            raise RunnerError(f"Hashed artifact is missing: {path}")
        if _sha256(path) != str(expected):
            raise RunnerError(f"Artifact hash mismatch: {path}")


def validate_model(
    path: Path, *, generation_manifest: Path | None = None,
    training_cutoff: str | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    _require_files(path, MODEL_FILES, "Archetype V2 model")
    manifest = load_json(path / "archetype_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("phase") != "phase_a_diagnostic":
        raise RunnerError(f"Model manifest is not a completed Phase A model: {path}")
    _verify_hashes(path, manifest.get("artifacts"), MODEL_ARTIFACTS)
    if generation_manifest is not None:
        generation = load_json(generation_manifest)
        expected_run = generation.get("run") or {}
        if manifest.get("generation_id") != expected_run.get("generation_id"):
            raise RunnerError("Model generation_id does not match the runner generation")
        if manifest.get("generation_manifest_sha256") != _sha256(generation_manifest):
            raise RunnerError("Model generation manifest hash does not match the runner generation")
    if training_cutoff is not None and manifest.get("training_cutoff") != training_cutoff:
        raise RunnerError("Model training cutoff does not match the runner configuration")
    return manifest


def validate_evaluation(
    path: Path, *, model_manifest: Path | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    path = path.expanduser().resolve()
    _require_files(path, EVALUATION_FILES, "Archetype V2 evaluation")
    manifest = load_json(path / "evaluation_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("phase") != "phase_a_diagnostic":
        raise RunnerError(f"Evaluation manifest is not complete: {path}")
    _verify_hashes(path, manifest.get("artifacts"), EVALUATION_ARTIFACTS)
    report = load_json(path / "evaluation.json")
    if model_manifest is not None:
        model = load_json(model_manifest)
        lineage = {
            "model_version": model.get("model_version"),
            "model_manifest_sha256": _sha256(model_manifest),
            "implementation_sha256": model.get("implementation_sha256"),
            "software_versions": model.get("software_versions"),
        }
        for key, expected in lineage.items():
            if manifest.get(key) != expected:
                raise RunnerError(f"Evaluation {key} does not match the runner model")
        if report.get("model_version") != model.get("model_version"):
            raise RunnerError("Evaluation report model_version does not match the runner model")
    gates = report.get("gates") or {}
    hard = (gates.get("hard") or {}).get("passed")
    quality = (gates.get("quality") or {}).get("passed")
    if not isinstance(hard, bool) or not isinstance(quality, bool):
        raise RunnerError(f"Evaluation gate booleans are missing: {path / 'evaluation.json'}")
    return report, hard, quality
