from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from story_monitor.incident_archetypes_v3 import (
    FEATURE_GROUPS,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    fit_robust_feature_schema,
    transform_finite_feature_matrix,
)
from story_monitor.incident_archetype_discovery_v3 import (
    IncidentArchetypeDiscoveryConfig,
    train_incident_archetypes_v3,
)
from weekly_story_monitor import build_parser, main as weekly_story_main


TRAINING_THROUGH = "2025-01-31"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_row(
    incident_id: str,
    crop_name: str,
    hazard_family: str,
    first_week: str,
    last_week: str,
    variant: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "incident_id": incident_id,
        "exposure_id": f"exposure-{incident_id}",
        "crop_name": crop_name,
        "hazard_family": hazard_family,
        "stratification_key": f"{crop_name}::{hazard_family}",
        "first_evidence_week": first_week,
        "last_evidence_week": last_week,
        "final_state": "CLOSED_RECOVERED",
    }
    for index, name in enumerate(MODEL_FEATURE_COLUMNS):
        row[name] = variant + (index % 4) * 0.01
    row["observed_week_count"] = 2.0
    row["maximum_area_km2"] = 4.0 + variant
    return row


def _source_frames(*, future_variant: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stories = [
        ("mh-0", "maize", "heat", "2025-01-01", "2025-01-08", 0.10),
        ("mh-1", "maize", "heat", "2025-01-01", "2025-01-08", 0.11),
        ("mh-noise", "maize", "heat", "2025-01-01", "2025-01-08", 0.90),
        ("bf-0", "beans", "ponding_flooding", "2025-01-01", "2025-01-08", 0.20),
        ("bf-1", "beans", "ponding_flooding", "2025-01-01", "2025-01-08", 0.21),
        ("bf-noise", "beans", "ponding_flooding", "2025-01-01", "2025-01-08", 0.95),
        ("bf-embargo", "beans", "ponding_flooding", "2025-01-22", "2025-02-05", 0.40),
        ("mh-future", "maize", "heat", "2025-02-05", "2025-02-12", future_variant),
    ]
    features = pd.DataFrame(_feature_row(*story) for story in stories)
    weekly_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    for incident_id, crop, hazard, first_week, last_week, _ in stories:
        for index, week in enumerate((first_week, last_week)):
            weekly_rows.append(
                {
                    "incident_id": incident_id,
                    "exposure_id": f"exposure-{incident_id}",
                    "timeline_bucket": week,
                    "crop_name": crop,
                    "hazard_family": hazard,
                    "active_count": 2 if index == 0 else 0,
                    "severe_count": 0,
                    "affected_count": 2 if index == 0 else 1,
                    "monitored_count": 5,
                    "evaluable_count": 5,
                    "current_state": "ACTIVE" if index == 0 else "CLOSED_RECOVERED",
                    "footprint_area_km2": 4.0 + index,
                    "hazard_intensity": 1.0 - index * 0.2,
                }
            )
            membership_rows.append(
                {
                    "incident_id": incident_id,
                    "timeline_bucket": week,
                    "crop_instance_id": f"crop-{incident_id}",
                    "field_id": f"field-{incident_id}",
                    "episode_id": f"episode-{incident_id}",
                    "membership_role": "pressure_core" if index == 0 else "recovering",
                    "stage_bucket": "vegetative" if index == 0 else "flowering",
                    "response_class": "recovery" if index else "none",
                    "event_state": "RECOVERING" if index else "ACTIVE",
                }
            )
    return features, pd.DataFrame(weekly_rows), pd.DataFrame(membership_rows)


def _write_source(
    root: Path,
    *,
    future_variant: float = 0.3,
    duplicate_feature: bool = False,
    lineage_totals: int | None = None,
    censored_incident_id: str | None = None,
) -> None:
    root.mkdir()
    features, weekly, membership = _source_frames(future_variant=future_variant)
    if duplicate_feature:
        features = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    if lineage_totals is not None:
        weekly["split_count"] = lineage_totals
        weekly["merge_count"] = lineage_totals
    if censored_incident_id is not None:
        features.loc[
            features["incident_id"] == censored_incident_id, "final_state"
        ] = "CLOSED_DATA_CENSORED"
        last_index = weekly[
            weekly["incident_id"] == censored_incident_id
        ]["timeline_bucket"].idxmax()
        weekly.loc[last_index, "current_state"] = "CLOSED_DATA_CENSORED"
    frames = {
        "completed_incident_features.parquet": features,
        "incident_weekly_state.parquet": weekly,
        "incident_membership.parquet": membership,
    }
    for name, frame in frames.items():
        frame.to_parquet(root / name, index=False)
    artifacts = {
        name: {"sha256": _sha256(root / name), "size_bytes": (root / name).stat().st_size}
        for name in frames
    }
    manifest = {
        "schema_version": "crop-impact-incident-generation-v3/1",
        "run": {
            "status": "complete",
            "immutable": True,
            "generation_id": "incident-v3-test-source",
            "baseline_through": "2024-12-31",
            "source_as_of_date": "2025-03-01",
        },
        "policy": {
            "version": "incident-policy-v3-test",
            "sha256": "a" * 64,
            "effective_sha256": "b" * 64,
            "calibration_status": "uncalibrated",
        },
        "semantics": {
            "primary_story_identity": "crop_impact_incident_id",
            "archetype_is_optional_not_identity": True,
        },
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fake_hdbscan(
    matrix: np.ndarray,
    config: IncidentArchetypeDiscoveryConfig,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    if backend != "cpu" or len(matrix) != 3 or config.min_cluster_size != 2:
        raise AssertionError("discovery did not fit one three-story model per stratum")
    return np.asarray([0, 0, -1]), np.asarray([0.98, 0.96, 0.0])


class IncidentArchetypeDiscoveryV3Tests(unittest.TestCase):
    def _config(self, *, engine: str = "cpu") -> IncidentArchetypeDiscoveryConfig:
        return IncidentArchetypeDiscoveryConfig(
            min_cluster_size=2,
            min_samples=1,
            radius_quantile=0.95,
            engine=engine,
        )

    def test_training_is_one_row_per_story_stratified_and_preserves_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incidents"
            model = root / "model"
            _write_source(source)
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=_fake_hdbscan,
            ) as fitted:
                manifest = train_incident_archetypes_v3(
                    source,
                    model,
                    training_through=TRAINING_THROUGH,
                    config=self._config(),
                )

            self.assertEqual(fitted.call_count, 2)
            assignments = pd.read_parquet(model / "completed_assignments.parquet")
            self.assertEqual(len(assignments), 6)
            self.assertTrue(assignments["incident_id"].is_unique)
            self.assertEqual(
                set(assignments["incident_id"]),
                {"mh-0", "mh-1", "mh-noise", "bf-0", "bf-1", "bf-noise"},
            )
            noise = assignments[assignments["assignment_status"] == "noise"]
            self.assertEqual(set(noise["incident_id"]), {"mh-noise", "bf-noise"})
            self.assertTrue(noise["archetype_id"].isna().all())

            catalog = pd.read_parquet(model / "archetype_catalog.parquet")
            self.assertEqual(len(catalog), 2)
            self.assertEqual(
                set(zip(catalog["crop_name"], catalog["hazard_family"])),
                {("maize", "heat"), ("beans", "ponding_flooding")},
            )
            self.assertEqual(set(catalog["status"]), {"diagnostic_unreviewed"})
            self.assertTrue(catalog["label"].str.contains("Diagnostic unreviewed").all())

            prototypes = pd.read_parquet(model / "prototypes.parquet")
            self.assertEqual(set(prototypes["status"]), {"diagnostic_unreviewed"})
            self.assertTrue((prototypes["radius"] > 0).all())
            self.assertFalse((model / "prefix_feature_schema.json").exists())
            self.assertFalse((model / "prefix_prototypes.parquet").exists())

            schema = json.loads((model / "feature_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["strata_columns"], ["crop_name", "hazard_family"])
            prohibited = (
                "timeline", "evidence_week", "latitude", "longitude",
                "center_lat", "center_lon", "district", "admin", "location",
            )
            self.assertFalse(
                any(token in name.lower() for name in schema["feature_names"] for token in prohibited)
            )
            self.assertEqual(
                schema["feature_weighting"]["method"],
                "equal_l2_energy_per_semantic_family",
            )
            for group in schema["feature_weighting"]["groups"].values():
                self.assertAlmostEqual(
                    len(group["features"]) * group["per_feature_weight"] ** 2,
                    1.0,
                )
            self.assertEqual(manifest["training_story_count"], 6)
            self.assertEqual(manifest["excluded_after_cutoff_count"], 2)
            self.assertEqual(manifest["holdout_story_count"], 1)
            self.assertEqual(manifest["embargo_story_count"], 1)
            self.assertTrue(manifest["semantics"]["incident_identity_preserved"])
            self.assertEqual(manifest["engine_used"], "cpu")
            self.assertEqual(
                manifest["prefix_prototypes"]["status"], "blocked_pending_review"
            )
            self.assertTrue(
                manifest["prefix_prototypes"]["review_overlay_required"]
            )

    def test_semantic_family_weights_match_actual_matrix_energy(self) -> None:
        base = {"crop_name": "maize", "hazard_family": "heat"}
        base.update({name: 0.0 for name in MODEL_FEATURE_COLUMNS})
        high = dict(base)
        high.update({name: 1.0 for name in MODEL_FEATURE_COLUMNS})
        schema = fit_robust_feature_schema(pd.DataFrame([base, high]))
        stage_probe = dict(base)
        stage_probe.update({name: 1.0 for name in FEATURE_GROUPS["crop_stage"]})
        temporal_probe = dict(base)
        temporal_probe.update({name: 1.0 for name in FEATURE_GROUPS["temporal_shape"]})
        matrix = transform_finite_feature_matrix(
            pd.DataFrame([base, stage_probe, temporal_probe]), schema
        )
        stage_distance = float(np.linalg.norm(matrix[1] - matrix[0]))
        temporal_distance = float(np.linalg.norm(matrix[2] - matrix[0]))
        self.assertAlmostEqual(stage_distance, temporal_distance, places=6)
        self.assertAlmostEqual(stage_distance, 2.0, places=6)

    def test_duplicate_story_rows_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incidents"
            model = root / "model"
            _write_source(source, duplicate_feature=True)
            with self.assertRaisesRegex(ValueError, "one row per incident_id"):
                train_incident_archetypes_v3(
                    source,
                    model,
                    training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
            self.assertFalse(model.exists())

    def test_censored_and_merged_fragments_are_not_discovery_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incidents"
            model = root / "model"
            _write_source(source, censored_incident_id="mh-noise")

            def fit(values, config, backend):
                self.assertIn(len(values), {2, 3})
                return np.zeros(len(values), dtype=np.int64), np.ones(len(values))

            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=fit,
            ):
                manifest = train_incident_archetypes_v3(
                    source,
                    model,
                    training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
            assignments = pd.read_parquet(model / "completed_assignments.parquet")
            self.assertNotIn("mh-noise", set(assignments["incident_id"]))
            self.assertEqual(manifest["source_terminal_story_count"], 8)
            self.assertEqual(manifest["ineligible_censored_story_count"], 1)
            self.assertEqual(
                manifest["ineligible_censored_by_state"],
                {"CLOSED_DATA_CENSORED": 1},
            )
            self.assertFalse(
                manifest["semantics"]
                ["censored_or_merged_fragments_used_for_discovery"]
            )

    def test_future_completed_rows_cannot_change_past_cutoff_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a, source_b = root / "source-a", root / "source-b"
            model_a, model_b = root / "model-a", root / "model-b"
            _write_source(source_a, future_variant=0.3)
            _write_source(source_b, future_variant=9999.0)
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=_fake_hdbscan,
            ):
                first = train_incident_archetypes_v3(
                    source_a, model_a, training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
                second = train_incident_archetypes_v3(
                    source_b, model_b, training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
            self.assertEqual(first["model_version"], second["model_version"])
            first_schema = json.loads((model_a / "feature_schema.json").read_text())
            second_schema = json.loads((model_b / "feature_schema.json").read_text())
            self.assertEqual(first_schema, second_schema)
            first_ids = pd.read_parquet(model_a / "archetype_catalog.parquet")["archetype_id"]
            second_ids = pd.read_parquet(model_b / "archetype_catalog.parquet")["archetype_id"]
            self.assertEqual(first_ids.tolist(), second_ids.tolist())
            self.assertNotEqual(
                first["source"]["artifacts"]["completed_incident_features.parquet"]["sha256"],
                second["source"]["artifacts"]["completed_incident_features.parquet"]["sha256"],
            )

    def test_deterministic_ids_repeat_for_identical_training_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incidents"
            _write_source(source)
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=_fake_hdbscan,
            ):
                first = train_incident_archetypes_v3(
                    source, root / "model-a", training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
                second = train_incident_archetypes_v3(
                    source, root / "model-b", training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
            self.assertEqual(first["model_version"], second["model_version"])
            left = pd.read_parquet(root / "model-a" / "archetype_catalog.parquet")
            right = pd.read_parquet(root / "model-b" / "archetype_catalog.parquet")
            self.assertEqual(left["archetype_id"].tolist(), right["archetype_id"].tolist())

    def test_unreviewed_prefixes_are_blocked_even_if_lineage_totals_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a, source_b = root / "source-a", root / "source-b"
            model_a, model_b = root / "model-a", root / "model-b"
            _write_source(source_a, lineage_totals=0)
            _write_source(source_b, lineage_totals=999)
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=_fake_hdbscan,
            ):
                first = train_incident_archetypes_v3(
                    source_a, model_a, training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
                second = train_incident_archetypes_v3(
                    source_b, model_b, training_through=TRAINING_THROUGH,
                    config=self._config(),
                )
            self.assertEqual(first["model_version"], second["model_version"])
            self.assertEqual(
                first["prefix_prototypes"]["status"], "blocked_pending_review"
            )
            self.assertEqual(
                second["prefix_prototypes"]["status"], "blocked_pending_review"
            )
            self.assertFalse((model_a / "prefix_prototypes.parquet").exists())
            self.assertFalse((model_b / "prefix_prototypes.parquet").exists())

    def test_gpu_request_fails_closed_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / "incidents", root / "model"
            _write_source(source)
            with patch(
                "story_monitor.incident_archetype_discovery_v3._gpu_dependencies",
                side_effect=RuntimeError("GPU discovery requires RAPIDS"),
            ), patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan"
            ) as fitted:
                with self.assertRaisesRegex(RuntimeError, "GPU discovery requires RAPIDS"):
                    train_incident_archetypes_v3(
                        source, model, training_through=TRAINING_THROUGH,
                        config=self._config(engine="gpu"),
                    )
            fitted.assert_not_called()
            self.assertFalse(model.exists())

    def test_all_noise_or_insufficient_strata_are_refused_without_partial_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / "incidents", root / "model"
            _write_source(source)
            all_noise = lambda matrix, config, backend: (  # noqa: E731
                np.full(len(matrix), -1), np.zeros(len(matrix))
            )
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan",
                side_effect=all_noise,
            ):
                with self.assertRaisesRegex(ValueError, "no supported archetypes"):
                    train_incident_archetypes_v3(
                        source, model, training_through=TRAINING_THROUGH,
                        config=self._config(),
                    )
            self.assertFalse(model.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / "incidents", root / "model"
            _write_source(source)
            insufficient = IncidentArchetypeDiscoveryConfig(
                min_cluster_size=4,
                min_samples=1,
                radius_quantile=0.95,
                engine="cpu",
            )
            with patch(
                "story_monitor.incident_archetype_discovery_v3._fit_hdbscan"
            ) as fitted:
                with self.assertRaisesRegex(ValueError, "no supported archetypes"):
                    train_incident_archetypes_v3(
                        source, model, training_through=TRAINING_THROUGH,
                        config=insufficient,
                    )
            fitted.assert_not_called()
            self.assertFalse(model.exists())

    def test_cutoff_before_v3_terminal_cohort_is_refused_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / "incidents", root / "model"
            _write_source(source)
            with self.assertRaisesRegex(ValueError, "after the incident baseline"):
                train_incident_archetypes_v3(
                    source,
                    model,
                    training_through="2024-12-31",
                    config=self._config(),
                )
            self.assertFalse(model.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / "incidents", root / "model"
            _write_source(source)
            with self.assertRaisesRegex(ValueError, "on or before the source as-of"):
                train_incident_archetypes_v3(
                    source,
                    model,
                    training_through="2025-03-02",
                    config=self._config(),
                )
            self.assertFalse(model.exists())

    def test_cli_registers_and_dispatches_training_command(self) -> None:
        parsed = build_parser().parse_args(
            [
                "train-incident-archetypes-v3",
                "--incident-dir", "/tmp/incidents",
                "--model-dir", "/tmp/model",
                "--training-through", TRAINING_THROUGH,
                "--engine", "gpu",
                "--min-cluster-size", "12",
                "--min-samples", "3",
            ]
        )
        self.assertEqual(parsed.command, "train-incident-archetypes-v3")
        self.assertEqual(parsed.engine, "gpu")

        argv = [
            "weekly_story_monitor.py",
            "train-incident-archetypes-v3",
            "--incident-dir", "/tmp/incidents",
            "--model-dir", "/tmp/model",
            "--training-through", TRAINING_THROUGH,
        ]
        with patch.object(sys, "argv", argv), patch(
            "story_monitor.incident_archetype_discovery_v3.train_incident_archetypes_v3",
            return_value={"status": "complete", "model_version": "test-model"},
        ) as train, redirect_stdout(StringIO()) as output:
            weekly_story_main()
        self.assertEqual(train.call_count, 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["model"]["model_version"], "test-model")


if __name__ == "__main__":
    unittest.main()
