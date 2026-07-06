"""Fail-closed validation for immutable dual-clock evidence releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from .incident_release_v4 import (
    normalize_released_at,
    validate_correction_policy,
)


SCHEMA_VERSION = "incident-evidence-validation-v4/1"
EVIDENCE_FILES = {
    "crop": "crop_day_context_v4.parquet",
    "pressure": "field_day_pressure_v4.parquet",
    "s2": "field_s2_acquisition_v4.parquet",
}
NATURAL_KEYS = {
    "crop": ("field_id", "crop_instance_id", "observation_date"),
    "pressure": (
        "field_id", "crop_instance_id", "observation_date", "hazard_family",
    ),
    "s2": ("acquisition_id",),
}


def validate_evidence_directory(directory: Path) -> dict[str, Any]:
    """Validate keys, modality clocks, references, and release knowledge bounds."""
    root = directory.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V4 evidence manifest does not exist: {manifest_path}")
    manifest = _manifest(manifest_path)
    run = manifest.get("run") or {}
    if str(run.get("status") or "") != "complete" or run.get("immutable") is not True:
        raise ValueError("V4 evidence manifest must describe one complete immutable release")
    release_as_of = str(run.get("release_as_of") or run.get("as_of_date") or "")[:10]
    if not release_as_of:
        raise ValueError("V4 evidence manifest is missing run.release_as_of")
    released_at = normalize_released_at(str(run.get("released_at") or ""))
    validate_correction_policy(manifest.get("correction_policy"))
    source_generation_id = str(run.get("source_generation_id") or "").strip()
    if not source_generation_id:
        raise ValueError("V4 evidence manifest is missing run.source_generation_id")
    policy = manifest.get("policy") or {}
    expected_policy = (
        str(policy.get("version") or ""), str(policy.get("sha256") or ""),
    )
    if not expected_policy[0] or len(expected_policy[1]) != 64:
        raise ValueError("V4 evidence manifest has invalid policy provenance")
    availability = manifest.get("availability") or {}
    expected_mode = str(availability.get("mode") or "")
    if expected_mode not in {"strict", "reconstructed"}:
        raise ValueError("V4 evidence manifest has invalid availability mode")
    inputs = manifest.get("inputs") or {}
    enriched_input = inputs.get("enriched_source") if isinstance(inputs, dict) else None
    if enriched_input is not None:
        contract = manifest.get("enriched_source_contract") or {}
        sidecar_input = inputs.get("enriched_source_manifest")
        reconciliation = (
            (manifest.get("reconciliation") or {}).get("source_field_day") or {}
        )
        if (
            not isinstance(enriched_input, dict)
            or not isinstance(sidecar_input, dict)
            or not isinstance(contract, dict)
            or str(contract.get("availability_mode") or "") != expected_mode
            or normalize_released_at(str(contract.get("released_at") or ""))
                != released_at
            or (contract.get("source") or {}).get("sha256")
                != enriched_input.get("sha256")
            or (contract.get("manifest") or {}).get("sha256")
                != sidecar_input.get("sha256")
            or reconciliation.get("exact_key_coverage") is not True
        ):
            raise ValueError(
                "V4 evidence manifest does not bind an exact enriched source contract"
            )
    artifacts = manifest.get("artifacts") or {}

    counts: dict[str, int] = {}
    with duckdb.connect(":memory:") as connection:
        for label, filename in EVIDENCE_FILES.items():
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"V4 evidence release is missing {path}")
            columns = _columns(connection, path)
            required = {
                *NATURAL_KEYS[label], "knowledge_time", "policy_version",
                "policy_sha256", "availability_mode",
            }
            if label == "s2":
                required.update(
                    {
                        "field_id",
                        "crop_instance_id",
                        "spectral_source_date",
                        "acquisition_attempted",
                        "spectral_usable",
                    }
                )
            missing = sorted(required - columns)
            if missing:
                raise ValueError(f"{filename} is missing columns: {', '.join(missing)}")
            key_sql = ", ".join(_quote(name) for name in NATURAL_KEYS[label])
            duplicates = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM (SELECT {key_sql}, COUNT(*) n "
                    f"FROM read_parquet(?) GROUP BY {key_sql} HAVING COUNT(*) > 1)",
                    [str(path)],
                ).fetchone()[0]
            )
            if duplicates:
                raise ValueError(f"{filename} contains {duplicates} duplicate natural keys")
            future = int(
                connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?) "
                    "WHERE TRY_CAST(knowledge_time AS DATE) IS NULL "
                    "OR CAST(knowledge_time AS DATE) > CAST(? AS DATE)",
                    [str(path), release_as_of],
                ).fetchone()[0]
            )
            if future:
                raise ValueError(
                    f"{filename} contains {future} invalid or post-release knowledge times"
                )
            post_release = int(
                connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?) "
                    "WHERE TRY_CAST(knowledge_time AS TIMESTAMPTZ) "
                    "> CAST(? AS TIMESTAMPTZ)",
                    [str(path), released_at],
                ).fetchone()[0]
            )
            if post_release:
                raise ValueError(
                    f"{filename} contains {post_release} knowledge times after released_at"
                )
            if label in {"crop", "pressure"}:
                effective_column = (
                    "stage_effective_date"
                    if label == "crop" and "stage_effective_date" in columns
                    else "observation_date"
                )
                before_effective = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM read_parquet(?) "
                        f"WHERE TRY_CAST(knowledge_time AS TIMESTAMP) "
                        f"< TRY_CAST({_quote(effective_column)} AS TIMESTAMP)",
                        [str(path)],
                    ).fetchone()[0]
                )
                if before_effective:
                    raise ValueError(
                        f"{label} knowledge precedes its effective observation time"
                    )
            counts[label] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
                ).fetchone()[0]
            )
            artifact = artifacts.get(label) if isinstance(artifacts, dict) else None
            if not isinstance(artifact, dict):
                raise ValueError(f"V4 evidence manifest is missing artifact metadata for {label}")
            artifact_size = artifact.get("size_bytes")
            artifact_rows = artifact.get("row_count")
            if (
                str(artifact.get("name") or "") != filename
                or artifact_size is None
                or int(artifact_size) != path.stat().st_size
                or artifact_rows is None
                or int(artifact_rows) != counts[label]
                or str(artifact.get("sha256") or "") != _sha256(path)
            ):
                raise ValueError(f"V4 evidence artifact provenance does not match {filename}")
            signatures = connection.execute(
                "SELECT DISTINCT CAST(availability_mode AS VARCHAR), "
                "CAST(policy_version AS VARCHAR), CAST(policy_sha256 AS VARCHAR) "
                "FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
            if counts[label] and signatures != [
                (expected_mode, expected_policy[0], expected_policy[1])
            ]:
                raise ValueError(
                    f"{filename} policy/availability provenance does not match manifest"
                )

        s2_path = root / EVIDENCE_FILES["s2"]
        invalid_source = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?) "
                "WHERE spectral_source_date IS NOT NULL AND "
                "(TRY_CAST(spectral_source_date AS DATE) IS NULL OR "
                "CAST(spectral_source_date AS DATE) > CAST(knowledge_time AS DATE))",
                [str(s2_path)],
            ).fetchone()[0]
        )
        if invalid_source:
            raise ValueError("S2 evidence time must not follow its knowledge time")
        impossible_usable = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?) "
                "WHERE COALESCE(TRY_CAST(spectral_usable AS BOOLEAN), FALSE) "
                "AND (NOT COALESCE(TRY_CAST(acquisition_attempted AS BOOLEAN), FALSE) "
                "OR spectral_source_date IS NULL)",
                [str(s2_path)],
            ).fetchone()[0]
        )
        if impossible_usable:
            raise ValueError("Usable S2 rows require an attempted acquisition and source date")
        s2_columns = _columns(connection, s2_path)
        if "new_response_evidence" in s2_columns:
            if "reference_acquisition_id" not in s2_columns:
                raise ValueError(
                    "S2 response evidence requires reference_acquisition_id provenance"
                )
            invalid_response = int(
                connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?) "
                    "WHERE COALESCE(TRY_CAST(new_response_evidence AS BOOLEAN), FALSE) "
                    "AND (NOT COALESCE(TRY_CAST(acquisition_attempted AS BOOLEAN), FALSE) "
                    "OR NOT COALESCE(TRY_CAST(spectral_usable AS BOOLEAN), FALSE) "
                    "OR reference_acquisition_id IS NULL)",
                    [str(s2_path)],
                ).fetchone()[0]
            )
            if invalid_response:
                raise ValueError(
                    "New S2 response evidence requires an attempted usable acquisition "
                    "and an earlier reference"
                )
        good = (
            'TRY_CAST("s2_good_observation" AS BOOLEAN)'
            if "s2_good_observation" in s2_columns else "NULL"
        )
        valid = (
            'TRY_CAST("valid_pixel_fraction" AS DOUBLE)'
            if "valid_pixel_fraction" in s2_columns else "NULL"
        )
        cloud = (
            'TRY_CAST("cloud_pct" AS DOUBLE)'
            if "cloud_pct" in s2_columns else "NULL"
        )
        quality = (
            'NULLIF(TRIM(CAST("s2_field_quality_flag" AS VARCHAR)), \'\')'
            if "s2_field_quality_flag" in s2_columns else "NULL"
        )
        usable_unknown_qa = int(
            connection.execute(
                f"SELECT COUNT(*) FROM read_parquet(?) "
                f"WHERE COALESCE(TRY_CAST(spectral_usable AS BOOLEAN), FALSE) "
                f"AND {good} IS NULL AND {valid} IS NULL "
                f"AND {cloud} IS NULL AND {quality} IS NULL",
                [str(s2_path)],
            ).fetchone()[0]
        )
        if usable_unknown_qa:
            raise ValueError("Usable S2 evidence cannot have unknown QA")
        repeated_source = int(
            connection.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT field_id, crop_instance_id, spectral_source_date, COUNT(*) n "
                "FROM read_parquet(?) WHERE spectral_source_date IS NOT NULL "
                "GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)",
                [str(s2_path)],
            ).fetchone()[0]
        )
        if repeated_source:
            raise ValueError(
                "S2 source dates must be unique per field/crop without a revision contract"
            )
        bad_references = int(
            connection.execute(
                """
                WITH source AS (
                    SELECT * FROM read_parquet(?)
                )
                SELECT COUNT(*)
                FROM source current
                LEFT JOIN source prior
                  ON CAST(current.reference_acquisition_id AS VARCHAR)
                   = CAST(prior.acquisition_id AS VARCHAR)
                WHERE current.reference_acquisition_id IS NOT NULL
                  AND (
                    prior.acquisition_id IS NULL
                    OR NOT COALESCE(TRY_CAST(prior.spectral_usable AS BOOLEAN), FALSE)
                    OR CAST(prior.spectral_source_date AS DATE)
                        >= CAST(current.spectral_source_date AS DATE)
                    OR CAST(prior.knowledge_time AS DATE)
                        > CAST(current.knowledge_time AS DATE)
                  )
                """,
                [str(s2_path)],
            ).fetchone()[0]
        )
        if bad_references:
            raise ValueError("S2 references must point to an earlier usable known acquisition")
    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "release_as_of": release_as_of,
        "released_at": released_at,
        "source_generation_id": source_generation_id,
        "availability_mode": expected_mode,
        "policy_version": expected_policy[0],
        "policy_sha256": expected_policy[1],
        "row_counts": counts,
    }


def validate_evidence_append(
    previous_directory: Path,
    current_directory: Path,
) -> dict[str, Any]:
    """Prove that an ordinary append did not rewrite prior evidence rows."""
    previous = previous_directory.expanduser().resolve()
    current = current_directory.expanduser().resolve()
    previous_validation = validate_evidence_directory(previous)
    current_validation = validate_evidence_directory(current)
    previous_as_of = previous_validation["release_as_of"]
    if current_validation["release_as_of"] < previous_as_of:
        raise ValueError("A V4 append release cannot move release_as_of backward")
    previous_released_at = previous_validation["released_at"]
    current_released_at = current_validation["released_at"]
    if current_released_at <= previous_released_at:
        raise ValueError("A V4 append release must advance released_at")

    append_counts: dict[str, int] = {}
    with duckdb.connect(":memory:") as connection:
        for label, filename in EVIDENCE_FILES.items():
            before = previous / filename
            after = current / filename
            before_columns = _ordered_columns(connection, before)
            after_columns = _ordered_columns(connection, after)
            if before_columns != after_columns:
                raise ValueError(f"V4 append changed the {filename} schema")
            keys = NATURAL_KEYS[label]
            key_join = " AND ".join(
                f"p.{_quote(name)} IS NOT DISTINCT FROM c.{_quote(name)}"
                for name in keys
            )
            missing = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM read_parquet(?) p "
                    f"ANTI JOIN read_parquet(?) c ON {key_join}",
                    [str(before), str(after)],
                ).fetchone()[0]
            )
            if missing:
                raise ValueError(f"V4 append deleted {missing} prior rows from {filename}")
            equality = " AND ".join(
                f"p.{_quote(name)} IS NOT DISTINCT FROM c.{_quote(name)}"
                for name in before_columns
            ) or "TRUE"
            changed = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM read_parquet(?) p "
                    f"JOIN read_parquet(?) c ON {key_join} WHERE NOT ({equality})",
                    [str(before), str(after)],
                ).fetchone()[0]
            )
            if changed:
                raise ValueError(f"V4 append rewrote {changed} prior rows in {filename}")
            append_counts[label] = int(current_validation["row_counts"][label]) - int(
                previous_validation["row_counts"][label]
            )
    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "previous_release_as_of": previous_as_of,
        "current_release_as_of": current_validation["release_as_of"],
        "previous_released_at": previous_released_at,
        "current_released_at": current_released_at,
        "appended_row_counts": append_counts,
    }


def _manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid V4 evidence manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("V4 evidence manifest root must be an object")
    return payload


def _columns(connection: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return set(_ordered_columns(connection, path))


def _ordered_columns(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> tuple[str, ...]:
    description = connection.execute(
        "SELECT * FROM read_parquet(?) LIMIT 0", [str(path)]
    ).description or []
    return tuple(str(item[0]) for item in description)


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EVIDENCE_FILES", "NATURAL_KEYS", "SCHEMA_VERSION",
    "validate_evidence_append", "validate_evidence_directory",
]
