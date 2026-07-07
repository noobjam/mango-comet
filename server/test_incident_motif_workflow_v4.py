from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from story_monitor.incident_motif_workflow_v4 import (
    _require_reviewed_calibration_labels,
    build_diagnostic_motif_release_v4,
    evaluate_prefix_release_v4,
    fit_reviewed_prefix_release_v4,
    materialize_causal_incident_evidence_v4,
    score_live_prefix_release_v4,
)
from story_monitor.incident_motifs_v4 import (
    MotifDiscoveryConfig,
    PrefixCalibrationConfig,
)
from story_monitor.incident_viewer_v4 import (
    LIFECYCLE_RECONCILIATION_SCHEMA_VERSION,
    SCHEMA_VERSION as VIEWER_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    incident_dir = root / "incident-v3"
    incident_dir.mkdir()
    incidents = [
        ("train-1", "2025-01-06", "2025-01-09"),
        ("train-2", "2025-01-13", "2025-01-16"),
        ("cal-1", "2025-02-03", "2025-02-06"),
        ("hold-1", "2025-03-03", "2025-03-06"),
    ]
    weekly: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    s2_rows: list[dict[str, object]] = []
    for incident_id, start, end in incidents:
        start_time = pd.Timestamp(start, tz="UTC")
        end_time = pd.Timestamp(end, tz="UTC")
        weekly.extend(
            [
                {
                    "incident_id": incident_id,
                    "knowledge_time": start_time,
                    "current_state": "ACTIVE",
                    "footprint_area_km2": 2.0,
                    "footprint_carried_forward": False,
                },
                {
                    "incident_id": incident_id,
                    "knowledge_time": end_time,
                    "current_state": "CLOSED_RECOVERED",
                    "footprint_area_km2": 3.0,
                    "footprint_carried_forward": False,
                },
            ]
        )
        windows.append(
            {
                "incident_id": incident_id,
                "exposure_id": "exposure-" + incident_id,
                "crop_name": "Maize",
                "hazard_family": "Heat",
                "terminal_state": "CLOSED_RECOVERED",
                "right_censored": False,
                "confirmed_time": start_time,
                "first_available_at": start_time,
                "feature_available_at": end_time,
            }
        )
        field_id = "field-" + incident_id
        crop_instance_id = "crop-" + incident_id
        membership.append(
            {
                "incident_id": incident_id,
                "field_id": field_id,
                "crop_instance_id": crop_instance_id,
                "hazard_family": "Heat",
                "timeline_bucket": start_time,
                "knowledge_time": start_time,
                "stage_bucket": "vegetative",
                "membership_role": "impact_lag",
            }
        )
        s2_rows.append(
            {
                "acquisition_id": "acquisition-" + incident_id,
                "field_id": field_id,
                "crop_instance_id": crop_instance_id,
                "acquisition_date": start_time,
                "known_date": start_time + pd.Timedelta(days=1),
                "spectral_source_date": start_time,
                "knowledge_time": start_time + pd.Timedelta(days=1),
                "acquisition_attempted": True,
                "spectral_usable": True,
                "valid_pixel_fraction": 1.0,
                "response_class": "medium_decline",
                "new_response_evidence": False,
                "ndvi_delta": -0.2,
                "ndmi_delta": -0.1,
                "psri_delta": 0.1,
            }
        )
        for day in pd.date_range(start_time, end_time, freq="D"):
            daily.append(
                {
                    "field_id": field_id,
                    "crop_instance_id": crop_instance_id,
                    "hazard_family": "Heat",
                    "observation_date": day,
                    "knowledge_time": day,
                    "pressure_observed": True,
                    "weather_intensity": 1.0,
                    "pressure_active": False,
                    "severe_pressure": False,
                    "affected_count": 2,
                    "monitored_count": 10,
                    "evaluable_count": 10,
                    "footprint_area_km2": 2.0,
                    "footprint_carried_forward": False,
                    "stage_bucket": "vegetative",
                }
            )
    pd.DataFrame(weekly).to_parquet(
        incident_dir / "incident_weekly_state.parquet", index=False
    )
    pd.DataFrame(windows).to_parquet(
        incident_dir / "incident_windows.parquet", index=False
    )
    pd.DataFrame(membership).to_parquet(
        incident_dir / "incident_membership.parquet", index=False
    )
    pd.DataFrame(
        columns=["parent_incident_id", "child_incident_id"]
    ).to_parquet(incident_dir / "incident_lineage.parquet", index=False)
    (incident_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "generation_id": "incident-v3-test",
                    "source_generation_id": "source-generation-test",
                },
                "semantics": {"archetype_is_optional_not_identity": True},
            }
        ),
        encoding="utf-8",
    )
    evidence_dir = root / "evidence-v4"
    evidence_dir.mkdir()
    policy_hash = "a" * 64
    for row in daily:
        row.update(
            availability_mode="reconstructed",
            policy_version="test-policy-v4",
            policy_sha256=policy_hash,
        )
    for row in s2_rows:
        row.update(
            availability_mode="reconstructed",
            policy_version="test-policy-v4",
            policy_sha256=policy_hash,
            reference_acquisition_id=None,
        )
    daily_path = evidence_dir / "field_day_pressure_v4.parquet"
    pd.DataFrame(daily).to_parquet(daily_path, index=False)
    membership_path = evidence_dir / "field_s2_acquisition_v4.parquet"
    pd.DataFrame(s2_rows).to_parquet(membership_path, index=False)
    crop_path = evidence_dir / "crop_day_context_v4.parquet"
    pd.DataFrame(
        [{
            "field_id": "context-field", "crop_instance_id": "context-crop",
            "observation_date": pd.Timestamp("2025-01-01"),
            "stage_effective_date": pd.Timestamp("2025-01-01"),
            "knowledge_time": pd.Timestamp("2025-01-01"),
            "availability_mode": "reconstructed",
            "policy_version": "test-policy-v4", "policy_sha256": policy_hash,
        }]
    ).to_parquet(crop_path, index=False)
    artifacts = {}
    for label, path in (("crop", crop_path), ("pressure", daily_path), ("s2", membership_path)):
        artifacts[label] = {
            "name": path.name, "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": len(pd.read_parquet(path)),
        }
    (evidence_dir / "manifest.json").write_text(
        json.dumps({
            "run": {"status": "complete", "immutable": True,
                    "source_generation_id": "source-generation-test",
                    "release_as_of": "2025-03-31",
                    "released_at": "2025-04-01T00:00:00Z"},
            "correction_policy": {
                "mode": "append_only_no_revisions",
                "late_corrections_supported": False,
                "require_explicit_revision_supersession": True,
                "failure_mode": (
                    "Reject changes to published natural keys until a future explicit "
                    "revision/supersession contract is implemented."
                ),
            },
            "availability": {"mode": "reconstructed"},
            "policy": {"version": "test-policy-v4", "sha256": policy_hash},
            "artifacts": artifacts,
        }), encoding="utf-8",
    )
    viewer_dir = root / "viewer-v4"
    viewer_dir.mkdir()
    checkpoints = pd.DataFrame(weekly).copy()
    checkpoints["story_week"] = pd.to_datetime(checkpoints["knowledge_time"])
    checkpoints["source_checkpoint_knowledge_time"] = checkpoints["knowledge_time"]
    checkpoints["story_known_time"] = checkpoints[
        "source_checkpoint_knowledge_time"
    ] + pd.Timedelta(hours=12)
    checkpoints["knowledge_time"] = checkpoints["story_known_time"]
    checkpoints["story_known_date"] = checkpoints["story_known_time"].dt.date
    checkpoints["contributing_evidence_knowledge_time"] = pd.Series(
        pd.NaT, index=checkpoints.index, dtype="datetime64[ns, UTC]"
    )
    checkpoints["checkpoint_bound_mode"] = "reconstructed"
    checkpoints["source_checkpoint_knowledge_time_inferred"] = False
    checkpoints["story_known_time_raised"] = True
    checkpoints["knowledge_bound_complete"] = True
    checkpoint_path = viewer_dir / "story_checkpoints_v4.parquet"
    checkpoints.to_parquet(checkpoint_path, index=False)
    lifecycle_path = viewer_dir / "lifecycle_reconciliation_v4.parquet"
    pd.DataFrame(
        {
            "schema_version": LIFECYCLE_RECONCILIATION_SCHEMA_VERSION,
            "incident_id": checkpoints["incident_id"].astype(str),
            "story_week": checkpoints["story_week"],
            "contradiction_count": 0,
            "positive_claim_reconciliation_complete": True,
            "source_state_preserved": True,
            "lifecycle_state_recomputed": False,
            "component_absence_replayed": False,
            "full_lifecycle_replay_supported": False,
            "lifecycle_causal_claim_supported": False,
        }
    ).to_parquet(lifecycle_path, index=False)
    checkpoint_artifact = {
        "sha256": _sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "row_count": len(checkpoints),
    }
    lifecycle_artifact = {
        "sha256": _sha256(lifecycle_path),
        "size_bytes": lifecycle_path.stat().st_size,
        "row_count": len(checkpoints),
    }
    viewer_artifacts = {
        checkpoint_path.name: checkpoint_artifact,
        lifecycle_path.name: lifecycle_artifact,
    }
    content_hash = hashlib.sha256(
        json.dumps(
            {
                name: artifact["sha256"]
                for name, artifact in viewer_artifacts.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (viewer_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": VIEWER_SCHEMA_VERSION,
                "mode": "crop_incident_v4_dual_clock",
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "viewer_ready": True,
                },
                "source": {
                    "incident_manifest_sha256": _sha256(
                        incident_dir / "manifest.json"
                    ),
                    "evidence_manifest_sha256": _sha256(
                        evidence_dir / "manifest.json"
                    ),
                    "bundle_content_sha256": content_hash,
                },
                "semantics": {
                    "lifecycle_reconciliation_schema_version": (
                        LIFECYCLE_RECONCILIATION_SCHEMA_VERSION
                    ),
                    "lifecycle_state_recomputed_from_v4": False,
                    "lifecycle_causal_ownership_claimed": False,
                    "component_absence_replayed_from_v4": False,
                },
                "outputs": {
                    "story_checkpoints": checkpoint_path.name,
                    "lifecycle_reconciliation": lifecycle_path.name,
                },
                "artifacts": viewer_artifacts,
            }
        ),
        encoding="utf-8",
    )
    return incident_dir, daily_path, membership_path, viewer_dir


def _refresh_viewer_manifest_bindings(
    viewer_dir: Path, incident_dir: Path, evidence_dir: Path
) -> None:
    path = viewer_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source"]["incident_manifest_sha256"] = _sha256(
        incident_dir / "manifest.json"
    )
    manifest["source"]["evidence_manifest_sha256"] = _sha256(
        evidence_dir / "manifest.json"
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")


class IncidentMotifWorkflowV4Tests(unittest.TestCase):
    def test_path_adapter_carries_monday_weather_then_stops_after_close(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership_path = root / "membership.parquet"
            pressure_path = root / "pressure.parquet"
            s2_path = root / "s2.parquet"
            windows_path = root / "windows.parquet"
            lineage_path = root / "lineage.parquet"
            checkpoints_path = root / "checkpoints.parquet"
            pd.DataFrame(
                [
                    {
                        "incident_id": "parent",
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "hazard_family": "heat",
                        "timeline_bucket": "2025-01-06",
                        "knowledge_time": "2025-01-10",
                        "membership_role": "impact_lag",
                    }
                ]
            ).to_parquet(membership_path, index=False)
            pd.DataFrame(
                [
                    {
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "hazard_family": "heat",
                        "observation_date": day,
                        "knowledge_time": day,
                        "pressure_observed": True,
                        "weather_intensity": 1.0,
                        "pressure_rank": 3,
                    }
                    for day in ("2025-01-13", "2025-01-20")
                ]
            ).to_parquet(pressure_path, index=False)
            pd.DataFrame(
                [
                    {
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "acquisition_id": "s2-1",
                        "spectral_source_date": "2025-01-06",
                        "knowledge_time": "2025-01-10",
                        "acquisition_attempted": False,
                        "spectral_usable": False,
                    }
                ]
            ).to_parquet(s2_path, index=False)
            pd.DataFrame(
                [
                    {
                        "incident_id": "parent",
                        "terminal_state": "CLOSED_RECOVERED",
                        "right_censored": False,
                        "closed_week": "2025-01-13",
                        "feature_available_at": "2025-01-17",
                    }
                ]
            ).to_parquet(windows_path, index=False)
            pd.DataFrame(
                columns=["parent_incident_id", "child_incident_id", "lineage_type"]
            ).to_parquet(lineage_path, index=False)
            pd.DataFrame(
                [
                    {
                        "incident_id": "parent",
                        "story_known_time": "2025-01-10",
                        "current_state": "ACTIVE",
                    },
                    {
                        "incident_id": "parent",
                        "story_known_time": "2025-01-17",
                        "current_state": "CLOSED_RECOVERED",
                    },
                ]
            ).to_parquet(checkpoints_path, index=False)
            daily_output = root / "joined-daily.parquet"
            s2_output = root / "joined-s2.parquet"
            materialize_causal_incident_evidence_v4(
                membership_path,
                pressure_path,
                s2_path,
                daily_output,
                s2_output,
                incident_windows_path=windows_path,
                incident_lineage_path=lineage_path,
                incident_checkpoints_path=checkpoints_path,
                threads=1,
                memory_limit="256MB",
            )
            joined = pd.read_parquet(daily_output)
            self.assertEqual(
                list(pd.to_datetime(joined["timeline_date"], utc=True)),
                [pd.Timestamp("2025-01-13T00:00:00Z")],
            )

    def test_path_adapter_filters_effective_week_before_late_weather_asof(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership_path = root / "membership.parquet"
            pressure_path = root / "pressure.parquet"
            s2_path = root / "s2.parquet"
            pd.DataFrame(
                [
                    {
                        "incident_id": "parent",
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "hazard_family": "heat",
                        "timeline_bucket": "2025-01-06",
                        "knowledge_time": "2025-01-10",
                        "membership_role": "impact_lag",
                    },
                    {
                        "incident_id": "child",
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "hazard_family": "heat",
                        "timeline_bucket": "2025-01-20",
                        "knowledge_time": "2025-01-20",
                        "membership_role": "impact_lag",
                    },
                ]
            ).to_parquet(membership_path, index=False)
            pd.DataFrame(
                [
                    {
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "hazard_family": "heat",
                        "observation_date": "2025-01-15",
                        # This row arrived after the future-effective child
                        # membership was already known.  The Jan-15 observation
                        # still belongs to the latest eligible Jan-06 owner.
                        "knowledge_time": "2025-02-01",
                        "pressure_observed": True,
                        "weather_intensity": 1.0,
                        "pressure_rank": 3,
                    }
                ]
            ).to_parquet(pressure_path, index=False)
            pd.DataFrame(
                [
                    {
                        "field_id": "field-1",
                        "crop_instance_id": "crop-1",
                        "acquisition_id": "s2-1",
                        "spectral_source_date": "2025-01-06",
                        "knowledge_time": "2025-01-10",
                        "acquisition_attempted": False,
                        "spectral_usable": False,
                    }
                ]
            ).to_parquet(s2_path, index=False)

            daily_output = root / "joined-daily.parquet"
            s2_output = root / "joined-s2.parquet"
            materialize_causal_incident_evidence_v4(
                membership_path,
                pressure_path,
                s2_path,
                daily_output,
                s2_output,
                threads=1,
                memory_limit="256MB",
            )

            joined = pd.read_parquet(daily_output)
            self.assertEqual(list(joined["incident_id"]), ["parent"])
            self.assertEqual(
                list(pd.to_datetime(joined["timeline_date"], utc=True)),
                [pd.Timestamp("2025-01-15T00:00:00Z")],
            )

    def test_calibration_dispositions_include_explicit_novel_incidents(self) -> None:
        review_hash = "a" * 64
        split = pd.DataFrame(
            [
                {"incident_id": "cal-known", "temporal_split": "calibration", "eligible": True},
                {"incident_id": "cal-novel", "temporal_split": "calibration", "eligible": True},
            ]
        )
        overlay = pd.DataFrame(
            [
                {
                    "model_version": "model-v1",
                    "review_status": "approved",
                    "reviewed_motif_id": "motif-a",
                    "review_version": "review-v1",
                }
            ]
        )
        labels = pd.DataFrame(
            [
                {
                    "incident_id": "cal-known",
                    "model_version": "model-v1",
                    "reviewed_motif_id": "motif-a",
                    "review_status": "approved",
                    "review_version": "review-v1",
                    "review_overlay_sha256": review_hash,
                },
                {
                    "incident_id": "cal-novel",
                    "model_version": "model-v1",
                    "reviewed_motif_id": pd.NA,
                    "review_status": "novel_unassigned",
                    "review_version": "review-v1",
                    "review_overlay_sha256": review_hash,
                },
            ]
        )

        _require_reviewed_calibration_labels(
            labels, split, overlay, review_overlay_sha256=review_hash
        )

        invalid = labels.copy()
        invalid.loc[invalid["incident_id"].eq("cal-novel"), "reviewed_motif_id"] = "motif-a"
        with self.assertRaisesRegex(ValueError, "novel dispositions require null"):
            _require_reviewed_calibration_labels(
                invalid, split, overlay, review_overlay_sha256=review_hash
            )

    def test_rejected_s2_response_cannot_change_persistent_impact(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, pressure_path, s2_path, _viewer_dir = _fixture(root)
            membership_path = incident_dir / "incident_membership.parquet"
            membership = pd.read_parquet(membership_path)
            membership["membership_role"] = "pressure_core"
            membership["fresh_response_evidence"] = False
            membership.to_parquet(membership_path, index=False)
            s2 = pd.read_parquet(s2_path)
            s2["spectral_usable"] = False
            s2["new_response_evidence"] = True
            s2.to_parquet(s2_path, index=False)
            daily_output = root / "joined-daily.parquet"
            s2_output = root / "joined-s2.parquet"
            materialize_causal_incident_evidence_v4(
                membership_path, pressure_path, s2_path,
                daily_output, s2_output, threads=1, memory_limit="256MB",
            )
            joined = pd.read_parquet(daily_output)
            self.assertEqual(int(joined["affected_count"].sum()), 0)

    def test_path_adapter_rolls_back_partial_pair_install(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, pressure_path, s2_path, _viewer_dir = _fixture(root)
            daily_output = root / "joined-daily.parquet"
            s2_output = root / "joined-s2.parquet"
            real_replace = __import__("os").replace

            def fail_second_replace(source: object, destination: object) -> None:
                if Path(destination).resolve() == s2_output.resolve():
                    raise OSError("simulated second-install failure")
                real_replace(source, destination)

            with patch(
                "story_monitor.incident_motif_workflow_v4.os.replace",
                side_effect=fail_second_replace,
            ), self.assertRaisesRegex(OSError, "second-install"):
                materialize_causal_incident_evidence_v4(
                    incident_dir / "incident_membership.parquet",
                    pressure_path,
                    s2_path,
                    daily_output,
                    s2_output,
                    threads=1,
                    memory_limit="256MB",
                )
            self.assertFalse(daily_output.exists())
            self.assertFalse(s2_output.exists())
            self.assertFalse(
                any(root.glob(".*.incomplete.json")),
                "handled install failure must clean its transaction marker",
            )

    def test_full_diagnostic_review_calibration_and_holdout_replay(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, daily_path, membership_path, viewer_dir = _fixture(root)
            raw_pressure = pd.read_parquet(daily_path)
            raw_pressure = pd.concat(
                [
                    raw_pressure,
                    raw_pressure.iloc[[0]].assign(
                        field_id="field-not-owned-by-any-incident",
                        crop_instance_id="crop-not-owned-by-any-incident",
                    ),
                ],
                ignore_index=True,
            )
            raw_pressure.to_parquet(daily_path, index=False)
            evidence_manifest_path = daily_path.parent / "manifest.json"
            evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
            evidence_manifest["artifacts"]["pressure"].update(
                sha256=_sha256(daily_path), size_bytes=daily_path.stat().st_size,
                row_count=len(raw_pressure),
            )
            evidence_manifest_path.write_text(
                json.dumps(evidence_manifest), encoding="utf-8"
            )
            _refresh_viewer_manifest_bindings(
                viewer_dir, incident_dir, daily_path.parent
            )
            raw_v3_weekly = incident_dir / "incident_weekly_state.parquet"
            raw_v3_weekly.unlink()
            raw_pressure_count = len(raw_pressure)
            source_hashes = {
                path: _sha256(path)
                for path in [
                    incident_dir / "incident_windows.parquet",
                    incident_dir / "incident_membership.parquet",
                    incident_dir / "incident_lineage.parquet",
                    incident_dir / "manifest.json",
                    viewer_dir / "story_checkpoints_v4.parquet",
                    viewer_dir / "manifest.json",
                    daily_path,
                    membership_path,
                ]
            }
            prefix_config = PrefixCalibrationConfig(
                weather_day_horizons=(2,),
                s2_acquisition_horizons=(0,),
                minimum_training_support=2,
                minimum_calibration_support=1,
                minimum_weather_observed_days=2,
            )
            discovery_dir = root / "motif-discovery"
            with self.assertRaises(ValueError):
                build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    membership_path,
                    incident_dir / "nested-output",
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=daily_path.parent / "manifest.json",
                    viewer_dir=viewer_dir,
                    config=MotifDiscoveryConfig(min_cluster_size=2, min_samples=1),
                    prefix_config=prefix_config,
                )
            real_read_parquet = pd.read_parquet
            pandas_reads: list[Path] = []

            def guarded_read_parquet(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
                resolved = Path(path).expanduser().resolve()
                if resolved in {daily_path.resolve(), membership_path.resolve()}:
                    raise AssertionError("raw field ledger reached pandas.read_parquet")
                pandas_reads.append(resolved)
                return real_read_parquet(path, *args, **kwargs)

            with patch(
                "story_monitor.incident_motif_workflow_v4.pd.read_parquet",
                side_effect=guarded_read_parquet,
            ), patch(
                "story_monitor.incident_motifs_v4._fit_hdbscan",
                return_value=(np.asarray([0, 0]), np.ones(2)),
            ):
                built = build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    membership_path,
                    discovery_dir,
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=daily_path.parent / "manifest.json",
                    viewer_dir=viewer_dir,
                    config=MotifDiscoveryConfig(min_cluster_size=2, min_samples=1),
                    prefix_config=prefix_config,
                    threads=2,
                    memory_limit="256MB",
                    temp_dir=root / "duckdb-spill",
                )
            self.assertFalse(built["map_publication_supported"])
            self.assertEqual(built["training_story_count"], 2)
            manifest = json.loads(
                (discovery_dir / "model_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["map_publication_supported"])
            self.assertEqual(manifest["prefix_model_status"], "blocked_pending_review")
            self.assertEqual(
                manifest["release_binding"]["source_generation_id"],
                "source-generation-test",
            )
            self.assertEqual(
                manifest["release_binding"]["story_checkpoints"]["row_count"], 8
            )
            adapter = manifest["evidence_path_adapter"]
            self.assertEqual(adapter["engine"], "duckdb")
            self.assertFalse(adapter["raw_field_ledgers_loaded_into_pandas"])
            self.assertEqual(adapter["threads"], 2)
            self.assertEqual(adapter["memory_limit"], "256MB")
            incident_daily = pd.read_parquet(
                discovery_dir / "incident_daily_pressure.parquet"
            )
            self.assertLess(len(incident_daily), raw_pressure_count)
            self.assertFalse(incident_daily["pressure_active"].any())
            self.assertEqual(
                int(incident_daily["affected_count"].sum()), 16,
                "impact_lag membership must persist between sparse acquisitions",
            )
            completed_features = pd.read_parquet(
                discovery_dir / "completed_story_features.parquet"
            )
            eligibility = pd.read_parquet(
                discovery_dir / "eligibility_ledger.parquet"
            ).set_index("incident_id")
            self.assertEqual(
                pd.Timestamp(eligibility.loc["train-1", "first_available_at"]).hour,
                12,
                "eligibility must use viewer story_known_time, not raw V3 knowledge_time",
            )
            self.assertEqual(
                pd.Timestamp(eligibility.loc["train-1", "feature_available_at"]).hour,
                12,
            )
            self.assertTrue(completed_features["s2_echo_age_max"].gt(0).all())
            self.assertTrue(
                completed_features["s2_usable_acquisition_count"].eq(1).all()
            )
            self.assertTrue(completed_features["maximum_observed_area_km2"].eq(3.0).all())
            self.assertTrue(
                completed_features["stage_distribution_json"].str.contains(
                    "latest_stage_only", regex=False
                ).all()
            )
            self.assertFalse(
                any(path.name == "incident_daily_pressure.parquet" for path in pandas_reads),
                "full incident/day evidence must stay in DuckDB/on disk",
            )
            self.assertFalse(
                any(path.name == "incident_s2_acquisitions.parquet" for path in pandas_reads),
                "full incident S2 evidence must stay in DuckDB/on disk",
            )
            self.assertFalse(raw_v3_weekly.exists())
            self.assertTrue(all(_sha256(path) == digest for path, digest in source_hashes.items()))
            with self.assertRaises(FileExistsError):
                build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    membership_path,
                    discovery_dir,
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=daily_path.parent / "manifest.json",
                    viewer_dir=viewer_dir,
                )

            review = pd.read_parquet(
                discovery_dir / "review_overlay_template.parquet"
            )
            review["review_status"] = "approved"
            review["reviewed_motif_id"] = "reviewed:steady-heat"
            review["review_version"] = "review-v1"
            review_path = root / "approved-review.parquet"
            review.to_parquet(review_path, index=False)
            calibration_labels_path = root / "reviewed-calibration-labels.parquet"
            pd.DataFrame(
                [
                    {
                        "incident_id": "cal-1",
                        "model_version": review.iloc[0]["model_version"],
                        "reviewed_motif_id": "reviewed:steady-heat",
                        "review_status": "approved",
                        "review_version": "review-v1",
                        "review_overlay_sha256": _sha256(review_path),
                    }
                ]
            ).to_parquet(calibration_labels_path, index=False)
            with self.assertRaisesRegex(ValueError, "maturity horizons"):
                fit_reviewed_prefix_release_v4(
                    discovery_dir, review_path, calibration_labels_path,
                    root / "bad-horizon-prefix-model",
                    config=PrefixCalibrationConfig(
                        weather_day_horizons=(3,),
                        s2_acquisition_horizons=(0,),
                        minimum_training_support=2,
                        minimum_calibration_support=1,
                    ),
                )
            prefix_artifact = discovery_dir / "causal_prefix_features.parquet"
            original_prefix_bytes = prefix_artifact.read_bytes()
            tampered = pd.read_parquet(prefix_artifact)
            tampered.loc[tampered.index[0], "duration_days"] += 1
            tampered.to_parquet(prefix_artifact, index=False)
            with self.assertRaisesRegex(ValueError, "producing manifest"):
                fit_reviewed_prefix_release_v4(
                    discovery_dir, review_path, calibration_labels_path,
                    root / "tampered-prefix-model", config=prefix_config,
                )
            prefix_artifact.write_bytes(original_prefix_bytes)
            prefix_dir = root / "prefix-model"
            fitted = fit_reviewed_prefix_release_v4(
                discovery_dir,
                review_path,
                calibration_labels_path,
                prefix_dir,
                config=prefix_config,
            )
            self.assertFalse(fitted["map_publication_supported"])
            self.assertGreater(fitted["prototype_count"], 0)

            prefix_manifest_path = prefix_dir / "prefix_manifest.json"
            prefix_manifest = json.loads(
                prefix_manifest_path.read_text(encoding="utf-8")
            )
            live_before = root / "live-score-before"
            live_result = score_live_prefix_release_v4(
                incident_dir,
                daily_path.parent,
                viewer_dir,
                prefix_dir,
                live_before,
                as_of="2025-01-08T23:00:00Z",
                threads=1,
                memory_limit="256MB",
            )
            self.assertEqual(live_result["active_incident_count"], 1)
            live_assignments = pd.read_parquet(
                live_before / "live_prefix_assignments.parquet"
            )
            self.assertEqual(set(live_assignments["incident_id"]), {"train-1"})
            self.assertEqual(
                set(live_assignments["prefix_manifest_sha256"]),
                {_sha256(prefix_manifest_path)},
            )
            self.assertTrue(
                {
                    "incident_manifest_sha256",
                    "evidence_manifest_sha256",
                    "viewer_manifest_sha256",
                    "source_generation_id",
                }
                <= set(live_assignments.columns)
            )
            live_manifest = json.loads(
                (live_before / "live_score_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                live_manifest["append_contract"]["mode"],
                "immutable_as_of_delta",
            )
            before_release_hash = _sha256(live_before / "live_score_manifest.json")
            with self.assertRaises(FileExistsError):
                score_live_prefix_release_v4(
                    incident_dir,
                    daily_path.parent,
                    viewer_dir,
                    prefix_dir,
                    live_before,
                    as_of="2025-01-08T23:00:00Z",
                )

            # A later source release may contain arbitrarily different future
            # evidence.  The already-published as-of delta remains immutable,
            # and rescoring the same cutoff produces identical causal features.
            future_pressure = pd.read_parquet(daily_path)
            future_mask = (
                future_pressure["field_id"].eq("field-train-1")
                & pd.to_datetime(future_pressure["observation_date"], utc=True).eq(
                    pd.Timestamp("2025-01-09T00:00:00Z")
                )
            )
            self.assertTrue(future_mask.any())
            future_pressure.loc[future_mask, "weather_intensity"] = 999.0
            future_pressure.to_parquet(daily_path, index=False)
            evidence_manifest_path = daily_path.parent / "manifest.json"
            changed_evidence_manifest = json.loads(
                evidence_manifest_path.read_text(encoding="utf-8")
            )
            changed_evidence_manifest["artifacts"]["pressure"].update(
                sha256=_sha256(daily_path),
                size_bytes=daily_path.stat().st_size,
                row_count=len(future_pressure),
            )
            evidence_manifest_path.write_text(
                json.dumps(changed_evidence_manifest), encoding="utf-8"
            )
            _refresh_viewer_manifest_bindings(
                viewer_dir, incident_dir, daily_path.parent
            )
            live_after = root / "live-score-after"
            score_live_prefix_release_v4(
                incident_dir,
                daily_path.parent,
                viewer_dir,
                prefix_dir,
                live_after,
                as_of="2025-01-08T23:00:00Z",
                threads=1,
                memory_limit="256MB",
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(live_before / "live_prefix_features.parquet"),
                pd.read_parquet(live_after / "live_prefix_features.parquet"),
            )
            self.assertEqual(
                _sha256(live_before / "live_score_manifest.json"),
                before_release_hash,
            )

            final_labels_path = root / "sealed-holdout-labels.parquet"
            pd.DataFrame(
                [
                    {
                        "incident_id": "hold-1",
                        "final_assignment_status": "accepted",
                        "reviewed_motif_id": "reviewed:steady-heat",
                        "discovery_model_version": manifest["model_version"],
                        "review_version": "review-v1",
                        "review_overlay_sha256": _sha256(review_path),
                        "prefix_model_version": prefix_manifest["model_version"],
                        "prefix_manifest_sha256": _sha256(prefix_manifest_path),
                    }
                ]
            ).to_parquet(final_labels_path, index=False)
            bad_labels_path = root / "mismatched-holdout-labels.parquet"
            pd.read_parquet(final_labels_path).assign(
                prefix_model_version="wrong-prefix-model"
            ).to_parquet(bad_labels_path, index=False)
            bad_evaluation_dir = root / "bad-evaluation"
            with self.assertRaisesRegex(ValueError, "prefix_model_version"):
                evaluate_prefix_release_v4(
                    discovery_dir,
                    prefix_dir,
                    bad_labels_path,
                    bad_evaluation_dir,
                )
            self.assertFalse(bad_evaluation_dir.exists())
            evaluation_dir = root / "evaluation"
            evaluated = evaluate_prefix_release_v4(
                discovery_dir,
                prefix_dir,
                final_labels_path,
                evaluation_dir,
            )
            self.assertTrue(evaluated["hard_gates_passed"])
            self.assertFalse(evaluated["map_publication_supported"])
            assignments = pd.read_parquet(
                evaluation_dir / "prefix_replay_assignments.parquet"
            )
            holdout = assignments[assignments["incident_id"].eq("hold-1")]
            self.assertIn("pending", set(holdout["assignment_status"]))
            self.assertIn(
                "tentative_crop_evidence_supported", set(holdout["assignment_status"])
            )
            self.assertEqual(
                holdout.sort_values("prefix_as_of_time").iloc[-1]["assignment_status"],
                "tentative_crop_evidence_supported",
                holdout.to_string(index=False),
            )

    def test_rejects_mutable_or_incomplete_incident_release(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, daily_path, membership_path, viewer_dir = _fixture(root)
            manifest_path = incident_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run"]["immutable"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "immutable complete"):
                build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    membership_path,
                    output,
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=daily_path.parent / "manifest.json",
                    viewer_dir=viewer_dir,
                )
            self.assertFalse(output.exists())

    def test_rejects_mixed_incident_and_evidence_generations(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, daily_path, s2_path, viewer_dir = _fixture(root)
            evidence_manifest_path = daily_path.parent / "manifest.json"
            evidence_manifest = json.loads(
                evidence_manifest_path.read_text(encoding="utf-8")
            )
            evidence_manifest["run"]["source_generation_id"] = "other-generation"
            evidence_manifest_path.write_text(
                json.dumps(evidence_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source_generation_id"):
                build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    s2_path,
                    root / "must-not-exist",
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=evidence_manifest_path,
                    viewer_dir=viewer_dir,
                )

    def test_rejects_viewer_bound_to_another_incident_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            incident_dir, daily_path, s2_path, viewer_dir = _fixture(root)
            viewer_manifest_path = viewer_dir / "manifest.json"
            viewer_manifest = json.loads(
                viewer_manifest_path.read_text(encoding="utf-8")
            )
            viewer_manifest["source"]["incident_manifest_sha256"] = "0" * 64
            viewer_manifest_path.write_text(
                json.dumps(viewer_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incident_manifest_sha256"):
                build_diagnostic_motif_release_v4(
                    incident_dir,
                    daily_path,
                    s2_path,
                    root / "must-not-exist",
                    train_through="2025-01-31",
                    calibration_through="2025-02-28",
                    evaluation_through="2025-03-31",
                    evidence_manifest_path=daily_path.parent / "manifest.json",
                    viewer_dir=viewer_dir,
                )

    def test_rejects_checkpoint_inventory_hash_size_and_row_count_mismatch(
        self,
    ) -> None:
        cases = (
            ("sha256", "0" * 64, "artifact hash mismatch"),
            ("size_bytes", 0, "artifact size mismatch"),
            ("row_count", 999, "parquet row_count mismatch"),
        )
        for field, replacement, message in cases:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                incident_dir, daily_path, s2_path, viewer_dir = _fixture(root)
                viewer_manifest_path = viewer_dir / "manifest.json"
                viewer_manifest = json.loads(
                    viewer_manifest_path.read_text(encoding="utf-8")
                )
                viewer_manifest["artifacts"]["story_checkpoints_v4.parquet"][
                    field
                ] = replacement
                viewer_manifest_path.write_text(
                    json.dumps(viewer_manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_diagnostic_motif_release_v4(
                        incident_dir,
                        daily_path,
                        s2_path,
                        root / "must-not-exist",
                        train_through="2025-01-31",
                        calibration_through="2025-02-28",
                        evaluation_through="2025-03-31",
                        evidence_manifest_path=daily_path.parent / "manifest.json",
                        viewer_dir=viewer_dir,
                    )


if __name__ == "__main__":
    unittest.main()
