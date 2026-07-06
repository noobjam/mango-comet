from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from story_monitor.incident_policy_v4 import (
    AVAILABILITY_MODES,
    CONTROLLED_STAGE_BUCKETS,
    DEFAULT_INCIDENT_POLICY_V4_PATH,
    HAZARD_FAMILIES,
    load_incident_policy_v4,
)


class IncidentPolicyV4Tests(unittest.TestCase):
    def test_default_policy_is_frozen_hashed_and_contract_aligned(self) -> None:
        policy = load_incident_policy_v4()

        self.assertEqual(policy.availability_modes, AVAILABILITY_MODES)
        self.assertEqual(policy.hazard_families, HAZARD_FAMILIES)
        self.assertEqual(policy.stage_buckets, CONTROLLED_STAGE_BUCKETS)
        self.assertEqual(
            policy.source_sha256,
            hashlib.sha256(DEFAULT_INCIDENT_POLICY_V4_PATH.read_bytes()).hexdigest(),
        )
        self.assertIn("UNCALIBRATED", policy.calibration_status.upper())
        self.assertIn("UNCALIBRATED", policy.warning.upper())
        self.assertEqual(policy.stage_bucket_for("Grain filling"), "fruiting_or_grain_fill")
        self.assertEqual(
            policy.stage_bucket_for("fruiting_or_grain_fill"),
            "fruiting_or_grain_fill",
        )
        self.assertEqual(policy.stage_bucket_for("invented phase"), "unknown")
        self.assertEqual(policy.validate_availability_mode(" Strict "), "strict")
        with self.assertRaises(ValueError):
            policy.validate_availability_mode("best-effort")
        with self.assertRaises(FrozenInstanceError):
            policy.version = "changed"  # type: ignore[misc]

    def test_policy_rejects_calibrated_claim_and_contract_drift(self) -> None:
        mutations = (
            (
                "calibrated",
                lambda payload: payload.update(
                    calibration_status="CALIBRATED", warning="Validated policy"
                ),
                "uncalibrated",
            ),
            (
                "availability modes",
                lambda payload: payload.update(availability_modes=["reconstructed", "strict"]),
                "frozen contract",
            ),
            (
                "hazards",
                lambda payload: payload.update(hazard_families=["heat"]),
                "frozen contract",
            ),
            (
                "pressure thresholds",
                lambda payload: payload["pressure_rank_thresholds"].update(high=4.0),
                "out of order",
            ),
            (
                "reference window",
                lambda payload: payload["reference_window_days"].update(
                    minimum=22, maximum=21
                ),
                "out of order",
            ),
            (
                "spectral direction",
                lambda payload: payload["spectral_change_thresholds"].update(
                    severe_ndvi_delta=-0.01
                ),
                "out of order",
            ),
            (
                "missing stage bucket aliases",
                lambda payload: payload["stage_aliases"].pop("flowering"),
                "every frozen stage bucket",
            ),
        )
        for label, mutate, error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(DEFAULT_INCIDENT_POLICY_V4_PATH.read_text())
                mutate(payload)
                path = Path(directory) / "policy.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    load_incident_policy_v4(path)


if __name__ == "__main__":
    unittest.main()
