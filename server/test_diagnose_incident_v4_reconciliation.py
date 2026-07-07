from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from diagnose_incident_v4_reconciliation import diagnose


class DiagnoseIncidentV4ReconciliationTests(unittest.TestCase):
    def test_classifies_response_family_qa_and_crop_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "job"
            incident = root / "incident"
            evidence = root / "evidence"
            for path in (job, incident, evidence):
                path.mkdir()
            claims = []
            for field_id in ("exact", "severity", "rejected", "other-crop"):
                claims.append(
                    {
                        "incident_id": f"incident-{field_id}",
                        "timeline_bucket": "2025-01-06",
                        "field_id": field_id,
                        "crop_instance_id": f"crop-{field_id}",
                        "response_class": "medium_decline",
                        "fresh_response_evidence": True,
                        "knowledge_time": "2025-01-10",
                    }
                )
            pd.DataFrame(claims).to_parquet(
                incident / "incident_membership.parquet", index=False
            )
            pd.DataFrame(
                [
                    {
                        "incident_id": row["incident_id"],
                        "timeline_bucket": row["timeline_bucket"],
                        "fresh_decline_field_count": 1,
                    }
                    for row in claims
                ]
            ).to_parquet(incident / "incident_weekly_state.parquet", index=False)
            acquisitions = [
                _acquisition("exact", "crop-exact", "medium_decline", True, True),
                _acquisition("severity", "crop-severity", "severe_decline", True, True),
                _acquisition("rejected", "crop-rejected", "not_evaluable", False, False),
                _acquisition("other-crop", "crop-different", "severe_decline", True, True),
            ]
            pd.DataFrame(acquisitions).to_parquet(
                evidence / "field_s2_acquisition_v4.parquet", index=False
            )
            (evidence / "manifest.json").write_text(
                json.dumps(
                    {
                        "reconciliation": {
                            "s2_acquisitions": {
                                "candidate_acquisition_count": 4,
                                "published_acquisition_count": 4,
                                "excluded_without_causal_crop_count": 0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (job / "state.json").write_text(
                json.dumps(
                    {
                        "config": {"incident_dir": str(incident)},
                        "paths": {"evidence_dir": str(evidence)},
                    }
                ),
                encoding="utf-8",
            )

            result = diagnose(job)

            summary = {
                row["classification"]: row["field_claim_count"]
                for row in result["classification_summary"]
            }
            self.assertEqual(summary["exact_response"], 1)
            self.assertEqual(summary["same_decline_family_different_severity"], 1)
            self.assertEqual(summary["rejected_by_v4_qa"], 1)
            self.assertEqual(summary["decline_owned_by_different_crop"], 1)
            self.assertEqual(result["weekly_membership_count_mismatches"], [])


def _acquisition(
    field_id: str,
    crop_instance_id: str,
    response_class: str,
    usable: bool,
    new_response: bool,
) -> dict[str, object]:
    return {
        "field_id": field_id,
        "crop_instance_id": crop_instance_id,
        "spectral_source_date": "2025-01-08",
        "knowledge_time": "2025-01-08",
        "acquisition_status": "usable" if usable else "rejected_cloud",
        "spectral_usable": usable,
        "new_response_evidence": new_response,
        "response_class": response_class,
    }


if __name__ == "__main__":
    unittest.main()
