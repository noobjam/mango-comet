#!/usr/bin/env python3
"""Classify V3 response claims against immutable V4 acquisition evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def _rows(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    frame = connection.execute(query).fetchdf()
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def diagnose(job_dir: Path, *, example_limit: int = 50) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve()
    state_path = job_dir / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"V4 job state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    incident_dir = Path(state["config"]["incident_dir"]).expanduser().resolve()
    evidence_dir = Path(state["paths"]["evidence_dir"]).expanduser().resolve()
    membership = incident_dir / "incident_membership.parquet"
    weekly = incident_dir / "incident_weekly_state.parquet"
    acquisitions = evidence_dir / "field_s2_acquisition_v4.parquet"
    evidence_manifest_path = evidence_dir / "manifest.json"
    for path in (membership, weekly, acquisitions, evidence_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required diagnostic input does not exist: {path}")

    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    connection = duckdb.connect(":memory:")
    try:
        connection.read_parquet(str(membership)).create_view("membership_v3")
        connection.read_parquet(str(weekly)).create_view("weekly_v3")
        connection.read_parquet(str(acquisitions)).create_view("acquisitions_v4")
        connection.execute(
            """
            CREATE TEMP VIEW decline_claims AS
            SELECT
              CAST(incident_id AS VARCHAR) AS incident_id,
              CAST(timeline_bucket AS DATE) AS story_week,
              CAST(field_id AS VARCHAR) AS field_id,
              CAST(crop_instance_id AS VARCHAR) AS crop_instance_id,
              CASE
                WHEN BOOL_OR(LOWER(CAST(response_class AS VARCHAR)) = 'severe_decline')
                  THEN 'severe_decline'
                ELSE 'medium_decline'
              END AS claimed_response,
              MAX(TRY_CAST(knowledge_time AS TIMESTAMP)) AS claim_known
            FROM membership_v3
            WHERE COALESCE(TRY_CAST(fresh_response_evidence AS BOOLEAN), FALSE)
              AND LOWER(CAST(response_class AS VARCHAR)) IN (
                'medium_decline', 'severe_decline'
              )
            GROUP BY incident_id, story_week, field_id, crop_instance_id
            """
        )
        connection.execute(
            """
            CREATE TEMP VIEW claim_matches AS
            SELECT c.*,
              COUNT(a.field_id) AS acquisition_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) = c.claimed_response
              ) AS exact_response_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) IN (
                  'medium_decline', 'severe_decline'
                )
              ) AS same_decline_family_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) IN (
                  'medium_decline', 'severe_decline'
                )
                AND LOWER(CAST(a.response_class AS VARCHAR)) <> c.claimed_response
              ) AS different_severity_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) IN (
                  'medium_decline', 'severe_decline'
                )
                AND TRY_CAST(a.knowledge_time AS TIMESTAMP) <= c.claim_known
              ) AS decline_known_by_v3_claim_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) IN (
                  'medium_decline', 'severe_decline'
                )
                AND TRY_CAST(a.spectral_source_date AS DATE) < c.story_week
                AND TRY_CAST(a.knowledge_time AS DATE) BETWEEN c.story_week
                  AND c.story_week + INTERVAL 6 DAY
              ) AS delayed_knowledge_decline_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND NOT COALESCE(a.spectral_usable, FALSE)
              ) AS rejected_same_crop_count,
              COUNT_IF(
                a.crop_instance_id = c.crop_instance_id
                AND COALESCE(a.spectral_usable, FALSE)
                AND NOT COALESCE(a.new_response_evidence, FALSE)
              ) AS usable_without_new_response_count,
              COUNT_IF(
                a.crop_instance_id <> c.crop_instance_id
                AND a.new_response_evidence
                AND LOWER(CAST(a.response_class AS VARCHAR)) IN (
                  'medium_decline', 'severe_decline'
                )
              ) AS decline_on_different_crop_count
            FROM decline_claims c
            LEFT JOIN acquisitions_v4 a
              ON CAST(a.field_id AS VARCHAR) = c.field_id
             AND (
               CAST(a.spectral_source_date AS DATE) BETWEEN c.story_week
                 AND c.story_week + INTERVAL 6 DAY
               OR (
                 CAST(a.knowledge_time AS DATE) BETWEEN c.story_week
                   AND c.story_week + INTERVAL 6 DAY
                 AND CAST(a.spectral_source_date AS DATE) < c.story_week
               )
             )
            GROUP BY ALL
            """
        )
        connection.execute(
            """
            CREATE TEMP VIEW classified_claims AS
            SELECT *,
              CASE
                WHEN exact_response_count > 0 THEN 'exact_response'
                WHEN different_severity_count > 0
                  THEN 'same_decline_family_different_severity'
                WHEN same_decline_family_count > 0
                     AND decline_known_by_v3_claim_count = 0
                  THEN 'decline_after_v3_claim_clock'
                WHEN rejected_same_crop_count > 0 THEN 'rejected_by_v4_qa'
                WHEN usable_without_new_response_count > 0
                  THEN 'usable_without_new_decline'
                WHEN decline_on_different_crop_count > 0
                  THEN 'decline_owned_by_different_crop'
                WHEN acquisition_count > 0 THEN 'acquisition_without_decline'
                ELSE 'no_v4_acquisition_in_claim_week'
              END AS classification
            FROM claim_matches
            """
        )
        summary = _rows(
            connection,
            """
            SELECT classification, COUNT(*) AS field_claim_count,
              COUNT(DISTINCT incident_id || '|' || CAST(story_week AS VARCHAR))
                AS checkpoint_count,
              SUM(same_decline_family_count) AS matching_decline_count,
              SUM(different_severity_count) AS different_severity_count,
              SUM(decline_known_by_v3_claim_count) AS known_by_claim_clock_count,
              SUM(delayed_knowledge_decline_count) AS delayed_knowledge_count
            FROM classified_claims
            GROUP BY classification
            ORDER BY field_claim_count DESC, classification
            """,
        )
        examples = _rows(
            connection,
            f"""
            SELECT * FROM classified_claims
            WHERE classification <> 'exact_response'
            ORDER BY classification, story_week, incident_id, field_id
            LIMIT {int(example_limit)}
            """,
        )
        count_mismatches = _rows(
            connection,
            """
            WITH member_counts AS (
              SELECT CAST(incident_id AS VARCHAR) AS incident_id,
                CAST(timeline_bucket AS DATE) AS story_week,
                COUNT(DISTINCT field_id) FILTER (
                  WHERE fresh_response_evidence
                    AND LOWER(CAST(response_class AS VARCHAR)) IN (
                      'medium_decline', 'severe_decline'
                    )
                ) AS member_decline_fields
              FROM membership_v3
              GROUP BY incident_id, story_week
            )
            SELECT CAST(w.incident_id AS VARCHAR) AS incident_id,
              CAST(w.timeline_bucket AS DATE) AS story_week,
              TRY_CAST(w.fresh_decline_field_count AS BIGINT)
                AS weekly_decline_fields,
              COALESCE(m.member_decline_fields, 0) AS member_decline_fields
            FROM weekly_v3 w
            LEFT JOIN member_counts m
              ON m.incident_id = CAST(w.incident_id AS VARCHAR)
             AND m.story_week = CAST(w.timeline_bucket AS DATE)
            WHERE COALESCE(TRY_CAST(w.fresh_decline_field_count AS BIGINT), 0)
              <> COALESCE(m.member_decline_fields, 0)
            ORDER BY story_week, incident_id
            """,
        )
        claim_count = int(
            connection.execute("SELECT COUNT(*) FROM classified_claims").fetchone()[0]
        )
    finally:
        connection.close()

    return {
        "status": "complete",
        "job_dir": str(job_dir),
        "incident_dir": str(incident_dir),
        "evidence_dir": str(evidence_dir),
        "decline_field_claim_count": claim_count,
        "classification_summary": summary,
        "non_exact_examples": examples,
        "weekly_membership_count_mismatches": count_mismatches,
        "evidence_acquisition_reconciliation": (
            (evidence_manifest.get("reconciliation") or {}).get("s2_acquisitions")
        ),
        "interpretation": {
            "exact_response": "V3 and V4 agree exactly.",
            "same_decline_family_different_severity": (
                "V4 supports the published decline family at different severity."
            ),
            "decline_after_v3_claim_clock": (
                "A V4 decline exists but became knowable after the V3 claim clock."
            ),
            "rejected_by_v4_qa": "Only QA-rejected V4 evidence was found.",
            "usable_without_new_decline": (
                "V4 has a usable acquisition but no newly supported decline."
            ),
            "decline_owned_by_different_crop": (
                "The decline belongs to another causal crop instance."
            ),
            "no_v4_acquisition_in_claim_week": (
                "No V4 acquisition can support the V3 claim window."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=50)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.example_limit <= 1000:
        parser.error("--example-limit must be between 1 and 1000")
    result = diagnose(args.job_dir, example_limit=args.example_limit)
    output = args.output or args.job_dir / "lifecycle_reconciliation_diagnostic.json"
    output = output.expanduser().resolve()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"DECLINE_FIELD_CLAIMS={result['decline_field_claim_count']}")
        for row in result["classification_summary"]:
            print(
                "CLASSIFICATION "
                f"name={row['classification']} "
                f"field_claims={row['field_claim_count']} "
                f"checkpoints={row['checkpoint_count']} "
                f"matching_declines={row['matching_decline_count']} "
                f"different_severity={row['different_severity_count']} "
                f"known_by_claim_clock={row['known_by_claim_clock_count']} "
                f"delayed_knowledge={row['delayed_knowledge_count']}"
            )
        mismatches = result["weekly_membership_count_mismatches"]
        print(f"WEEKLY_MEMBERSHIP_COUNT_MISMATCHES={len(mismatches)}")
        for row in mismatches[:10]:
            print(
                "COUNT_MISMATCH "
                f"incident={row['incident_id']} week={str(row['story_week'])[:10]} "
                f"weekly={row['weekly_decline_fields']} "
                f"membership={row['member_decline_fields']}"
            )
        reconciliation = result.get("evidence_acquisition_reconciliation") or {}
        print(
            "ACQUISITION_RECONCILIATION "
            f"candidates={reconciliation.get('candidate_acquisition_count')} "
            f"published={reconciliation.get('published_acquisition_count')} "
            "excluded_without_causal_crop="
            f"{reconciliation.get('excluded_without_causal_crop_count')}"
        )
    print(f"DIAGNOSTIC_JSON={output}")


if __name__ == "__main__":
    main()
