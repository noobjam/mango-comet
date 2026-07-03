"""Preflight, build, evaluation, and gate stages for the V2 runner."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any

from .archetype_runner_validation import (
    ensure_clean_repository,
    validate_evaluation,
    validate_generation,
    validate_model,
)
from .runner_process import (
    RunnerError,
    finish_state,
    load_json,
    run_job_stage,
    stage_state,
)


__all__ = ["build", "evaluate", "gate_result", "preflight"]

EXIT_BUILD = 10
EXIT_MODEL = 11
EXIT_EVALUATE = 12
EXIT_EVALUATION = 13
EXIT_HARD_GATES = 20
EXIT_QUALITY_GATES = 21
EXIT_BOTH_GATES = 22


def _model(state: dict[str, Any]) -> dict[str, Any]:
    return validate_model(
        Path(state["paths"]["model_dir"]),
        generation_manifest=Path(state["config"]["generation_dir"]) / "manifest.json",
        training_cutoff=state["config"]["training_cutoff"],
    )


def _evaluation(state: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    return validate_evaluation(
        Path(state["paths"]["evaluation_dir"]),
        model_manifest=Path(state["paths"]["model_dir"]) / "archetype_manifest.json",
    )


def preflight(state: dict[str, Any], logger: logging.Logger) -> None:
    config = state["config"]
    job = Path(state["paths"]["job_dir"])
    stage_state(state, "preflight", "running")
    ensure_clean_repository(Path(config["repo_dir"]))
    manifest = validate_generation(Path(config["generation_dir"]))
    generation_as_of = str((manifest.get("run") or {}).get("as_of_date") or "")[:10]
    if config["training_cutoff"] >= generation_as_of:
        raise RunnerError("Training cutoff must precede the generation as-of date")
    Path(config["temp_dir"]).mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(config["root"]).free / 1024**3
    logger.info("Preflight — generation %s — free disk %.1f GiB", config["generation_dir"], free_gib)
    gpu_code = run_job_stage(
        state,
        logger,
        "[PREFLIGHT GPU]",
        [
            config["rapids_python"],
            "-c",
            "import cupy,cuml; from cuml.cluster.hdbscan import HDBSCAN; "
            "count=cupy.cuda.runtime.getDeviceCount(); assert count>0; "
            "print({'gpu_count':count,'cupy':cupy.__version__,'cuml':cuml.__version__})",
        ],
        job / "gpu.stdout.log",
        job / "gpu.stderr.log",
    )
    if gpu_code:
        raise RunnerError(f"GPU/RAPIDS preflight failed; see {job / 'gpu.stderr.log'}")
    if not config["skip_tests"]:
        tests = [
            "server.test_anchor_features_v2",
            "server.test_archetypes_v2",
            "server.test_archetype_stability_v2",
            "server.test_archetype_workflow_v2",
        ]
        test_code = run_job_stage(
            state,
            logger,
            "[PREFLIGHT TESTS]",
            [config["rapids_python"], "-m", "unittest", *tests],
            job / "tests.stdout.log",
            job / "tests.stderr.log",
        )
        if test_code:
            raise RunnerError(f"Focused V2 tests failed; see {job / 'tests.stderr.log'}")
    stage_state(state, "preflight", "complete")


def build(state: dict[str, Any], logger: logging.Logger) -> None:
    config, paths = state["config"], state["paths"]
    model = Path(paths["model_dir"])
    if model.exists():
        try:
            _model(state)
        except RunnerError as exc:
            raise RunnerError(f"Existing model is invalid and will not be overwritten: {exc}", EXIT_MODEL) from exc
        logger.info("[1/2 BUILD] valid model already exists; skipping")
        stage_state(state, "build", "complete", resumed=True)
        return
    stage_state(state, "build", "running")
    command = [
        config["rapids_python"], "server/weekly_story_monitor.py", "build-archetypes-v2",
        "--generation-dir", config["generation_dir"], "--training-through", config["training_cutoff"],
        "--output-dir", str(model), "--engine", "gpu", "--radius-quantile", "0.95",
        "--assignment-margin", "0.05", "--threads", str(config["threads"]),
        "--memory-limit", config["memory_limit"], "--temp-dir", config["temp_dir"],
    ]
    try:
        code = run_job_stage(
            state, logger, "[1/2 BUILD]", command,
            Path(paths["build_json"]), Path(paths["build_stderr"]),
        )
    except RunnerError as exc:
        raise RunnerError(f"Archetype build could not run: {exc}", EXIT_BUILD) from exc
    if code:
        raise RunnerError(f"Archetype build exited {code}; see {paths['build_stderr']}", EXIT_BUILD)
    try:
        result = load_json(Path(paths["build_json"]))
        if result.get("status") != "complete" or Path(str(result.get("model_dir"))).resolve() != model.resolve():
            raise RunnerError("Build JSON does not identify the completed requested model")
        manifest = _model(state)
        if result.get("model_version") != manifest.get("model_version"):
            raise RunnerError("Build JSON model_version does not match the model manifest")
    except RunnerError as exc:
        raise RunnerError(f"Completed model validation failed: {exc}", EXIT_MODEL) from exc
    stage_state(state, "build", "complete", model_version=result.get("model_version"))


def gate_result(state: dict[str, Any], hard: bool, quality: bool) -> int:
    summary = {"hard_gates_passed": hard, "quality_gates_passed": quality}
    if hard and quality:
        return finish_state(state, "passed", 0, gate_summary=summary)
    code = EXIT_HARD_GATES if not hard and quality else EXIT_QUALITY_GATES if hard else EXIT_BOTH_GATES
    return finish_state(state, "gates_failed", code, gate_summary=summary)


def evaluate(state: dict[str, Any], logger: logging.Logger) -> int:
    config, paths = state["config"], state["paths"]
    evaluation = Path(paths["evaluation_dir"])
    if evaluation.exists():
        try:
            _, hard, quality = _evaluation(state)
        except RunnerError as exc:
            raise RunnerError(f"Existing evaluation is invalid and will not be overwritten: {exc}", EXIT_EVALUATION) from exc
        logger.info("[2/2 EVALUATE] valid evaluation already exists; reporting its gates")
        stage_state(state, "evaluate", "complete", resumed=True)
        return gate_result(state, hard, quality)
    stage_state(state, "evaluate", "running")
    command = [
        config["rapids_python"], "server/weekly_story_monitor.py", "evaluate-archetypes-v2",
        "--model-dir", paths["model_dir"], "--output-dir", str(evaluation), "--stability-runs", "2",
    ]
    try:
        code = run_job_stage(
            state, logger, "[2/2 EVALUATE]", command,
            Path(paths["evaluation_json"]), Path(paths["evaluation_stderr"]),
        )
    except RunnerError as exc:
        raise RunnerError(f"Archetype evaluation could not run: {exc}", EXIT_EVALUATE) from exc
    if code:
        raise RunnerError(f"Archetype evaluation exited {code}; see {paths['evaluation_stderr']}", EXIT_EVALUATE)
    try:
        result = load_json(Path(paths["evaluation_json"]))
        if result.get("status") != "complete" or Path(str(result.get("evaluation_dir"))).resolve() != evaluation.resolve():
            raise RunnerError("Evaluation JSON does not identify the completed requested evaluation")
        _, hard, quality = _evaluation(state)
        if result.get("hard_gates_passed") is not hard or result.get("quality_gates_passed") is not quality:
            raise RunnerError("Evaluation command and persisted gate results disagree")
    except RunnerError as exc:
        raise RunnerError(f"Completed evaluation validation failed: {exc}", EXIT_EVALUATION) from exc
    stage_state(state, "evaluate", "complete")
    return gate_result(state, hard, quality)
