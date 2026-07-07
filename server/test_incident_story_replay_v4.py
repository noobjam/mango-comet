from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from run_incident_story_replay_v4 import (
    _validated_replay,
    _validated_source_adapter,
    _validated_viewer,
)
from story_monitor.incident_policy_v3 import load_incident_policy_v3
from story_monitor.incident_policy_v4 import load_incident_policy_v4
from story_monitor.incident_release_v4 import CORRECTION_POLICY
from story_monitor.incident_story_replay_v4 import (
    CHECKPOINT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _replay_partition_ids,
    build_incident_story_replay_v4,
)
from story_monitor.incident_viewer_v4 import export_incident_viewer_v4


class IncidentStoryReplayV4Tests(unittest.TestCase):
    def test_partition_worklist_includes_orphan_pressure_and_s2_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ledger, partition in (("crop", 7), ("pressure", 11), ("s2", 13)):
                (root / ledger / f"replay_partition={partition}").mkdir(
                    parents=True
                )

            self.assertEqual(_replay_partition_ids(root), [7, 11, 13])

    def test_builds_checkpointed_native_release_and_audit_crosswalk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            geometry = root / "geometry.parquet"
            audit = root / "old-v3"
            output = root / "incidents-v4-replay"
            checkpoints = root / "checkpoints"
            _write_evidence(evidence)
            _write_geometry(geometry)
            _write_audit_release(audit)
            tracker = _tracker_policy()

            result = build_incident_story_replay_v4(
                evidence,
                geometry,
                audit,
                output,
                checkpoints,
                baseline_through="2025-01-12",
                source_policy=load_incident_policy_v4(),
                tracker_policy=tracker,
                threads=1,
                replay_partitions=2,
            )

            self.assertEqual(result["status"], "complete")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertTrue(
                manifest["semantics"]["lifecycle_state_recomputed_from_v4"]
            )
            self.assertFalse(manifest["semantics"]["old_incident_ids_seed_new_ids"])
            self.assertEqual(
                manifest["validation"]["membership_counter_mismatch_count"], 0
            )
            self.assertGreater(
                manifest["validation"]["row_counts"]["incident_weekly_state"], 0
            )
            crosswalk = pd.read_parquet(
                output / "old_to_new_incident_crosswalk.parquet"
            )
            self.assertIn("overlap", set(crosswalk["match_status"]))
            memberships = pd.read_parquet(output / "incident_membership.parquet")
            self.assertFalse(
                memberships.duplicated(
                    ["incident_id", "timeline_bucket", "field_id"]
                ).any()
            )
            weekly = pd.read_parquet(output / "incident_weekly_state.parquet")
            self.assertFalse(weekly["knowledge_time_inferred"].any())
            member_max = memberships.groupby(
                ["incident_id", "timeline_bucket"]
            )["knowledge_time"].max()
            weekly_bound = weekly.set_index(
                ["incident_id", "timeline_bucket"]
            )["knowledge_time"]
            self.assertTrue(
                (
                    pd.to_datetime(
                        weekly_bound.loc[member_max.index], utc=True
                    )
                    >= pd.to_datetime(member_max, utc=True)
                ).all()
            )
            for index in range(1, 8):
                checkpoint = next(checkpoints.glob(f"{index:02d}_*"))
                payload = json.loads((checkpoint / "manifest.json").read_text())
                self.assertEqual(
                    payload["schema_version"], CHECKPOINT_SCHEMA_VERSION
                )
                self.assertEqual(payload["run"]["status"], "complete")

            viewer = root / "viewer-v4-native"
            viewer_result = export_incident_viewer_v4(
                output,
                evidence,
                checkpoints / "01_context" / "source_generation",
                viewer,
                threads=1,
                native_replay=True,
            )
            self.assertEqual(viewer_result["status"], "complete")
            viewer_manifest = json.loads((viewer / "manifest.json").read_text())
            self.assertTrue(viewer_manifest["run"]["native_replay"])
            self.assertEqual(
                viewer_manifest["validation"][
                    "lifecycle_reconciliation_contradiction_count"
                ],
                0,
            )
            runner_state = {
                "paths": {
                    "incident_dir": str(output),
                    "checkpoint_dir": str(checkpoints),
                },
                "config": {
                    "evidence_dir": str(evidence),
                    "geometry_parquet": str(geometry),
                    "audit_incident_dir": str(audit),
                    "baseline_through": "2025-01-12",
                },
            }
            self.assertEqual(
                _validated_replay(output, runner_state)["status"], "valid"
            )
            source_adapter = _validated_source_adapter(runner_state)
            self.assertEqual(
                _validated_viewer(viewer, runner_state, source_adapter)["status"],
                "valid",
            )

            second_output = root / "incidents-v4-replay-copy"
            second = build_incident_story_replay_v4(
                evidence,
                geometry,
                audit,
                second_output,
                checkpoints,
                baseline_through="2025-01-12",
                source_policy=load_incident_policy_v4(),
                tracker_policy=tracker,
                threads=1,
                replay_partitions=2,
            )
            self.assertEqual(result["generation_id"], second["generation_id"])
            pd.testing.assert_frame_equal(
                pd.read_parquet(output / "incident_weekly_state.parquet"),
                pd.read_parquet(second_output / "incident_weekly_state.parquet"),
            )

            shuffled_evidence = root / "evidence-shuffled"
            _shuffle_evidence(evidence, shuffled_evidence)
            shuffled_output = root / "incidents-v4-replay-shuffled"
            build_incident_story_replay_v4(
                shuffled_evidence,
                geometry,
                audit,
                shuffled_output,
                root / "shuffled-checkpoints",
                baseline_through="2025-01-12",
                source_policy=load_incident_policy_v4(),
                tracker_policy=tracker,
                threads=1,
                replay_partitions=2,
            )
            for name, keys in (
                ("weekly_components.parquet", ["component_id"]),
                (
                    "incident_weekly_state.parquet",
                    ["incident_id", "timeline_bucket"],
                ),
                (
                    "incident_membership.parquet",
                    ["incident_id", "timeline_bucket", "field_id"],
                ),
            ):
                expected = pd.read_parquet(output / name).sort_values(
                    keys, kind="mergesort"
                ).reset_index(drop=True)
                actual = pd.read_parquet(shuffled_output / name).sort_values(
                    keys, kind="mergesort"
                ).reset_index(drop=True)
                pd.testing.assert_frame_equal(expected, actual)


def _tracker_policy():
    return replace(
        load_incident_policy_v3(),
        minimum_source_field_centroid_coverage=1.0,
        minimum_source_crop_instance_week_centroid_coverage=1.0,
        minimum_known_stage_coverage=1.0,
        minimum_known_stage_coverage_per_supported_crop=1.0,
        minimum_stage_coverage_crop_instance_weeks=1,
        minimum_evaluable_fields=1,
        minimum_active_fields=1,
        severe_override_min_fields=1,
        severe_override_min_fresh_response_fields=1,
        minimum_crop_monitored_instances=1,
        minimum_crop_evaluable_instances=1,
        quiet_close_weeks=1,
    )


def _write_evidence(root: Path) -> None:
    root.mkdir()
    source_policy = load_incident_policy_v4()
    dates = pd.date_range("2024-12-30", "2026-01-18", freq="D")
    crop_rows = []
    pressure_rows = []
    s2_rows = []
    for field_index in range(3):
        field_id = f"field-{field_index}"
        crop_id = f"crop-{field_index}"
        for day in dates:
            iso = day.date().isoformat()
            crop_rows.append(
                {
                    "field_id": field_id,
                    "crop_instance_id": crop_id,
                    "observation_date": iso,
                    "knowledge_time": f"{iso}T08:00:00Z",
                    "crop_name": "maize",
                    "crop_season": "2025-26",
                    "crop_stage_raw": "silking",
                    "stage_family_raw": "flowering",
                    "stage_bucket": "flowering",
                    "stage_effective_date": iso,
                    "crop_instance_start_date": "2024-12-30",
                    "availability_mode": "strict",
                    "policy_version": source_policy.version,
                    "policy_sha256": source_policy.source_sha256,
                }
            )
            for hazard in source_policy.hazard_families:
                rank = 0
                if hazard == "heat":
                    if day in pd.DatetimeIndex(
                        ["2024-12-30", "2024-12-31", "2025-01-06", "2025-01-07"]
                    ):
                        rank = 2
                    elif day >= pd.Timestamp("2026-01-05"):
                        rank = 4
                pressure_rows.append(
                    {
                        "record_id": f"p-{field_id}-{iso}-{hazard}",
                        "field_id": field_id,
                        "crop_instance_id": crop_id,
                        "observation_date": iso,
                        "pressure_observation_date": iso,
                        "knowledge_time": f"{iso}T09:00:00Z",
                        "hazard_family": hazard,
                        "pressure_observed": True,
                        "pressure_rank": rank,
                        "pressure_band": {
                            0: "NONE", 2: "LOW-MED", 4: "HIGH"
                        }[rank],
                        "availability_mode": "strict",
                        "policy_version": source_policy.version,
                        "policy_sha256": source_policy.source_sha256,
                    }
                )
        reference_id = f"reference-{field_id}"
        s2_rows.extend(
            [
                {
                    "acquisition_id": reference_id,
                    "field_id": field_id,
                    "crop_instance_id": crop_id,
                    "spectral_source_date": "2024-12-30",
                    "knowledge_time": "2024-12-30T10:00:00Z",
                    "acquisition_attempted": True,
                    "spectral_usable": True,
                    "new_response_evidence": False,
                    "response_class": "insufficient_reference",
                    "reference_acquisition_id": None,
                    "valid_pixel_fraction": 0.9,
                    "availability_mode": "strict",
                    "policy_version": source_policy.version,
                    "policy_sha256": source_policy.source_sha256,
                },
                {
                    "acquisition_id": f"decline-{field_id}",
                    "field_id": field_id,
                    "crop_instance_id": crop_id,
                    "spectral_source_date": "2026-01-05",
                    "knowledge_time": "2026-01-05T10:00:00Z",
                    "crop_assignment_effective_date": "2026-01-05",
                    "crop_assignment_available_at": "2026-01-05T08:00:00Z",
                    "acquisition_attempted": True,
                    "spectral_usable": True,
                    "new_response_evidence": True,
                    "response_class": "severe_decline",
                    "reference_acquisition_id": reference_id,
                    "valid_pixel_fraction": 0.9,
                    "availability_mode": "strict",
                    "policy_version": source_policy.version,
                    "policy_sha256": source_policy.source_sha256,
                },
            ]
        )
    frames = {
        "crop": pd.DataFrame(crop_rows),
        "pressure": pd.DataFrame(pressure_rows),
        "s2": pd.DataFrame(s2_rows),
    }
    filenames = {
        "crop": "crop_day_context_v4.parquet",
        "pressure": "field_day_pressure_v4.parquet",
        "s2": "field_s2_acquisition_v4.parquet",
    }
    artifacts = {}
    for label, frame in frames.items():
        path = root / filenames[label]
        frame.to_parquet(path, index=False)
        artifacts[label] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "row_count": len(frame),
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": "crop-impact-incident-evidence-v4/1",
        "run": {
            "status": "complete",
            "immutable": True,
            "source_generation_id": "source-generation-1",
            "release_as_of": "2026-01-18",
            "as_of_date": "2026-01-18",
            "released_at": "2026-01-18T23:00:00Z",
        },
        "correction_policy": CORRECTION_POLICY,
        "availability": {"mode": "strict"},
        "policy": {
            "version": source_policy.version,
            "sha256": source_policy.source_sha256,
        },
        "inputs": {},
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _write_geometry(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "field_id": f"field-{index}",
                "centroid_lon": 30.001 + index * 0.001,
                "centroid_lat": -1.001 + index * 0.001,
                "geometry_text": (
                    f"POLYGON (({30.0 + index * 0.01} -1.01, "
                    f"{30.005 + index * 0.01} -1.01, "
                    f"{30.005 + index * 0.01} -1.005, "
                    f"{30.0 + index * 0.01} -1.005, "
                    f"{30.0 + index * 0.01} -1.01))"
                ),
                "district": "district-a",
                "sector": "sector-a",
                "cell": "cell-a",
                "village": "village-a",
            }
            for index in range(3)
        ]
    ).to_parquet(path, index=False)


def _shuffle_evidence(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    for index, (label, filename) in enumerate(
        (
            ("crop", "crop_day_context_v4.parquet"),
            ("pressure", "field_day_pressure_v4.parquet"),
            ("s2", "field_s2_acquisition_v4.parquet"),
        ),
        start=1,
    ):
        path = destination / filename
        frame = pd.read_parquet(path).sample(
            frac=1, random_state=index
        ).reset_index(drop=True)
        frame.to_parquet(path, index=False)
        manifest["artifacts"][label] = {
            "name": filename,
            "size_bytes": path.stat().st_size,
            "row_count": len(frame),
            "sha256": _sha256(path),
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _write_audit_release(root: Path) -> None:
    root.mkdir()
    rows = []
    for week in ("2026-01-05", "2026-01-12"):
        for index in range(3):
            rows.append(
                {
                    "incident_id": "old-incident",
                    "timeline_bucket": week,
                    "field_id": f"field-{index}",
                }
            )
    pd.DataFrame(rows).to_parquet(root / "incident_membership.parquet", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"run": {"status": "complete", "immutable": True}}) + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
