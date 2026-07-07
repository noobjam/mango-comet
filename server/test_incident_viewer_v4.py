from __future__ import annotations

from datetime import date, timedelta
import hashlib
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

import duckdb
import pandas as pd

from story_map_server import (
    ResourceNotFoundError,
    Settings,
    StoryMapStore,
    make_handler,
)
from story_monitor.incident_viewer_v4 import (
    LIFECYCLE_RECONCILIATION_SCHEMA_VERSION,
    _write_daily_grid,
    export_incident_viewer_v4,
    validate_viewer_directory,
)
from story_monitor.incident_release_v4 import CORRECTION_POLICY
from test_incident_viewer_v3 import _write_incident, _write_source


class IncidentViewerV4Tests(unittest.TestCase):
    def test_dual_clock_export_preserves_hazards_attempts_and_as_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _add_causal_regression_cases(incident)
            _write_evidence(evidence)

            result = export_incident_viewer_v4(
                incident,
                evidence,
                source,
                output,
                threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )

            self.assertEqual(result["mode"], "crop_incident_v4_dual_clock")
            self.assertEqual(validate_viewer_directory(output)["status"], "valid")
            lifecycle = pd.read_parquet(
                output / "lifecycle_reconciliation_v4.parquet"
            )
            self.assertEqual(
                set(lifecycle["schema_version"]),
                {LIFECYCLE_RECONCILIATION_SCHEMA_VERSION},
            )
            self.assertTrue(lifecycle["positive_claim_reconciliation_complete"].all())
            self.assertFalse(lifecycle["lifecycle_state_recomputed"].any())
            self.assertFalse(lifecycle["lifecycle_causal_claim_supported"].any())
            quiet = lifecycle[
                lifecycle["story_week"].astype(str).eq("2025-01-13")
            ].iloc[0]
            self.assertEqual(
                quiet["quiet_evidence_status"],
                "observed_quiet_for_attributed_members",
            )
            pressure_observations = pd.read_parquet(
                output / "pressure_observations_v4.parquet"
            )
            self.assertEqual(
                result["pressure_observation_count"], len(pressure_observations)
            )
            attempts = pd.read_parquet(output / "s2_attempts_v4.parquet")
            updates = pd.read_parquet(output / "s2_updates_v4.parquet")
            self.assertEqual(
                attempts["marker_type"].tolist(),
                ["acquisition", "acquisition", "rejected", "acquisition"],
            )
            self.assertEqual(len(updates), 3)
            self.assertTrue(updates["is_new_acquisition"].all())

            pressure = pd.read_parquet(output / "daily_pressure_grid_v4.parquet")
            first = pressure[pressure["calendar_date"].astype(str).eq("2025-01-06")]
            self.assertEqual(set(first["hazard_family"]), {"drought", "heat"})

            states = pd.read_parquet(output / "field_day_state_v4.parquet")
            stale = states[
                states["calendar_date"].astype(str).eq("2025-01-20")
                & states["field_id"].eq("field-1")
            ]
            self.assertTrue((stale["spectral_source_date"].astype(str) == "2025-01-05").all())
            self.assertTrue((stale["evidence_freshness"] == "stale").all())

            store = StoryMapStore(
                Settings(
                    run_dir=output,
                    static_dir=root,
                    host="127.0.0.1",
                    port=8877,
                    raster_tiles="",
                    raster_attribution="",
                    default_feature_limit=5000,
                    max_feature_limit=20000,
                    log_level="INFO",
                )
            )
            timeline = store.v4_timeline()
            self.assertEqual(timeline["clock"], "daily_as_of")
            rejected_day = next(
                row for row in timeline["days"]
                if row["calendar_date"] == "2025-01-08"
            )
            self.assertEqual(rejected_day["rejected_s2_attempt_count"], 1)
            self.assertEqual(rejected_day["new_s2_field_count"], 0)

            frame = store.v4_frame(
                calendar_date="2025-01-06", bbox=None, limit=5000
            )
            self.assertTrue(frame["field_overview"]["meta"]["complete"])
            self.assertEqual(frame["meta"]["accounted_field_count"], 2)
            self.assertEqual(frame["meta"]["source_field_count"], 2)
            self.assertFalse(frame["meta"]["unmappable_warning"])
            self.assertEqual(
                {feature["properties"]["hazard_family"] for feature in frame["pressure"]["features"]},
                {"drought", "heat"},
            )
            detail_frame = store.v4_frame(
                calendar_date="2025-01-06",
                bbox=(29.9, -2.1, 30.3, -1.8),
                limit=5000,
            )
            self.assertEqual(len(detail_frame["fields"]["features"]), 2)
            self.assertFalse(detail_frame["fields"]["meta"]["truncated"])
            compact = store.v4_frame_state(
                calendar_date="2025-01-06",
                bbox=(29.9, -2.1, 30.3, -1.8),
                limit=5000,
            )
            self.assertEqual(len(compact["rows"]), 2)
            self.assertTrue(compact["geometry_version"])
            self.assertFalse(compact["meta"]["truncated"])
            self.assertTrue(all("geometry_geojson" not in row for row in compact["rows"]))
            compact_field_one = next(
                row for row in compact["rows"] if row["field_id"] == "field-1"
            )
            self.assertEqual(compact_field_one["active_hazards"], "drought,heat")

            active_filtered = store.v4_frame(
                calendar_date="2025-01-20",
                bbox=None,
                limit=5000,
                filters={"incident_state": "ACTIVE"},
            )
            recovering_filtered = store.v4_frame(
                calendar_date="2025-01-20",
                bbox=None,
                limit=5000,
                filters={"incident_state": "RECOVERING"},
            )
            self.assertEqual(active_filtered["story_footprints"]["features"], [])
            self.assertEqual(
                len(recovering_filtered["story_footprints"]["features"]), 1
            )

            field_detail = store.v4_field_detail(
                "field-1",
                as_of_date="2025-01-10",
                crop_instance_id="crop-1",
                lookback_days=30,
                history_limit=5000,
            )
            self.assertEqual(field_detail["field_id"], "field-1")
            self.assertEqual(
                field_detail["current_state_scope"],
                "explicit_crop_latest_known_state",
            )
            self.assertEqual(
                {row["hazard_family"] for row in field_detail["daily_pressure"]},
                {"drought", "heat"},
            )
            heat_pressure = [
                row for row in field_detail["daily_pressure"]
                if row["hazard_family"] == "heat"
            ]
            drought_pressure = [
                row for row in field_detail["daily_pressure"]
                if row["hazard_family"] == "drought"
            ]
            self.assertTrue(all(
                row["attributed_incident_ids"] == "incident-1"
                for row in heat_pressure
            ))
            self.assertTrue(all(
                row["attributed_incident_ids"] is None
                for row in drought_pressure
            ))
            self.assertEqual(
                [row["marker_type"] for row in field_detail["s2_attempts"]],
                ["acquisition", "rejected"],
            )
            self.assertEqual(field_detail["story_checkpoints"], [])
            field_detail_after_week = store.v4_field_detail(
                "field-1",
                as_of_date="2025-01-12",
                crop_instance_id="crop-1",
                lookback_days=30,
                history_limit=5000,
            )
            self.assertEqual(len(field_detail_after_week["story_checkpoints"]), 1)
            self.assertEqual(
                field_detail_after_week["story_checkpoints"][0]["incident_state"],
                "ACTIVE",
            )
            self.assertEqual(
                field_detail_after_week["story_checkpoints"][0]["stage_bucket"],
                "vegetative",
            )
            bounded_field = store.v4_field_detail(
                "field-1",
                as_of_date="2025-01-10",
                crop_instance_id="crop-1",
                lookback_days=30,
                history_limit=1,
            )
            self.assertTrue(bounded_field["history"]["any_truncated"])
            self.assertTrue(all(
                len(bounded_field[key]) <= 1
                for key in ("daily_pressure", "s2_attempts", "story_checkpoints")
            ))
            detail = store.v4_incident_detail(
                "incident-1", as_of_date="2025-01-12"
            )
            self.assertTrue(detail["weekly_state"])
            self.assertTrue(all(
                str(row["story_known_date"])[:10] <= "2025-01-12"
                for row in detail["weekly_state"]
            ))
            self.assertEqual(detail["window"]["terminal_state"], "ACTIVE")
            self.assertIsNone(detail["window"].get("closed_week"))
            self.assertEqual(detail["window"]["relapse_count"], 0)
            self.assertEqual(detail["window"]["merge_count"], 1)
            self.assertEqual(detail["window"]["split_count"], 0)
            self.assertEqual(
                [row["parent_incident_id"] for row in detail["lineage"]["incoming"]],
                ["incident-past"],
            )
            self.assertEqual(detail["lineage"]["outgoing"], [])
            self.assertEqual(
                {row["field_id"] for row in detail["s2_updates"]}, {"field-1"}
            )
            self.assertTrue(all(
                int(row["pressure_field_count"]) == 1
                for row in detail["daily_pressure"]
            ))
            self.assertEqual(
                {row["hazard_family"] for row in detail["daily_pressure"]},
                {"heat"},
            )
            with self.assertRaises(ResourceNotFoundError):
                store.v4_incident_detail(
                    "incident-1", as_of_date="2025-01-05"
                )

            bounded = store.v4_incident_detail(
                "incident-1",
                as_of_date="2025-01-24",
                lookback_days=30,
                history_limit=1,
            )
            self.assertEqual(bounded["history"]["lookback_days"], 30)
            self.assertEqual(bounded["history"]["history_limit_per_collection"], 1)
            self.assertTrue(bounded["history"]["any_truncated"])
            for key in (
                "weekly_state", "stage_summary", "footprints", "daily_pressure",
                "s2_updates", "s2_attempts",
            ):
                self.assertLessEqual(len(bounded[key]), 1)

            recent = store.v4_incident_detail(
                "incident-1",
                as_of_date="2025-01-24",
                lookback_days=2,
                history_limit=5000,
            )
            self.assertEqual(recent["history"]["window_start"], "2025-01-23")
            self.assertTrue(
                recent["history"]["current_checkpoint_outside_lookback"]
            )
            self.assertEqual(len(recent["weekly_state"]), 1)
            self.assertEqual(recent["daily_pressure"], [])
            self.assertEqual(recent["s2_updates"], [])
            self.assertEqual(recent["s2_attempts"], [])

            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(store, store.settings)
            )
            thread = Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                timeline_http = _http_json(httpd.server_port, "/api/v4/timeline")
                frame_http = _http_json(
                    httpd.server_port,
                    "/api/v4/frame/2025-01-06?bbox=29.9,-2.1,30.3,-1.8",
                )
                state_http = _http_json(
                    httpd.server_port,
                    "/api/v4/frame-state/2025-01-06"
                    "?bbox=29.9,-2.1,30.3,-1.8&limit=1",
                )
                field_http = _http_json(
                    httpd.server_port,
                    "/api/v4/field/field-1?as_of=2025-01-10"
                    "&crop_instance_id=crop-1&lookback_days=30&history_limit=5",
                )
                incident_http = _http_json(
                    httpd.server_port,
                    "/api/v4/incident/incident-1?as_of=2025-01-24"
                    "&lookback_days=30&history_limit=1",
                )
                incident_http_wide = _http_json(
                    httpd.server_port,
                    "/api/v4/incident/incident-1?as_of=2025-01-24"
                    "&lookback_days=30&history_limit=5",
                )
                self.assertEqual(timeline_http["clock"], "daily_as_of")
                self.assertEqual(frame_http["calendar_date"], "2025-01-06")
                self.assertEqual(len(state_http["rows"]), 1)
                self.assertTrue(state_http["meta"]["truncated"])
                self.assertEqual(field_http["field_id"], "field-1")
                self.assertEqual(
                    field_http["history"]["history_limit_per_collection"], 5
                )
                self.assertEqual(incident_http["as_of_date"], "2025-01-24")
                self.assertEqual(
                    incident_http["history"]["history_limit_per_collection"], 1
                )
                self.assertTrue(incident_http["history"]["any_truncated"])
                self.assertEqual(
                    incident_http_wide["history"]["history_limit_per_collection"],
                    5,
                )
                self.assertGreater(
                    len(incident_http_wide["weekly_state"]),
                    len(incident_http["weekly_state"]),
                )
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=3)

    def test_source_and_knowledge_clocks_are_preserved_without_daily_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _write_evidence(evidence)

            crop_path = evidence / "crop_day_context_v4.parquet"
            crop = pd.read_parquet(crop_path)
            crop["knowledge_time"] = pd.to_datetime(crop["knowledge_time"])
            late_crop = crop["field_id"].eq("field-1") & crop[
                "observation_date"
            ].astype(str).isin({"2025-01-06", "2025-01-07"})
            crop.loc[late_crop, "knowledge_time"] = pd.Timestamp(
                "2025-01-08 12:00:00"
            )
            crop.to_parquet(crop_path, index=False)

            pressure_path = evidence / "field_day_pressure_v4.parquet"
            pressure = pd.read_parquet(pressure_path)
            pressure["knowledge_time"] = pd.to_datetime(pressure["knowledge_time"])
            pressure["weather_available_at"] = pd.to_datetime(
                pressure["weather_available_at"]
            )
            late_pressure = pressure["field_id"].eq("field-1") & pressure[
                "hazard_family"
            ].eq("heat") & pressure["observation_date"].astype(str).isin(
                {"2025-01-06", "2025-01-07"}
            )
            pressure.loc[late_pressure, "knowledge_time"] = pd.Timestamp(
                "2025-01-08 12:00:00"
            )
            pressure.loc[late_pressure, "weather_available_at"] = pd.Timestamp(
                "2025-01-08 12:00:00"
            )
            pressure.loc[
                pressure["field_id"].eq("field-1")
                & pressure["hazard_family"].eq("heat")
                & pressure["observation_date"].astype(str).eq("2025-01-06"),
                ["pressure_rank", "pressure_score"],
            ] = [4, 0.99]
            pressure.to_parquet(pressure_path, index=False)

            s2_path = evidence / "field_s2_acquisition_v4.parquet"
            s2 = pd.read_parquet(s2_path)
            s2["knowledge_time"] = pd.to_datetime(s2["knowledge_time"])
            s2["spectral_source_date"] = pd.to_datetime(s2["spectral_source_date"])
            s2["reference_source_date"] = pd.to_datetime(
                s2["reference_source_date"]
            )
            first = s2["acquisition_id"].eq("s2-2025-01-05-field-1")
            s2.loc[first, "knowledge_time"] = pd.Timestamp("2025-01-06 12:00:00")
            extra = s2.loc[first].iloc[0].copy()
            extra["acquisition_id"] = "s2-2025-01-04-field-1"
            extra["spectral_source_date"] = pd.Timestamp("2025-01-04")
            extra["reference_source_date"] = pd.Timestamp("2024-12-24")
            extra["response_class"] = "recovery"
            s2 = pd.concat([s2, pd.DataFrame([extra])], ignore_index=True)
            s2.to_parquet(s2_path, index=False)
            _refresh_evidence_artifacts(evidence)

            export_incident_viewer_v4(
                incident,
                evidence,
                source,
                output,
                threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )

            states = pd.read_parquet(output / "field_day_state_v4.parquet")
            field_one = states[states["field_id"].eq("field-1")]
            self.assertTrue(
                {"2025-01-06", "2025-01-07"}
                <= set(field_one["calendar_date"].astype(str))
            )
            observed = field_one[field_one["pressure_observed"]]
            self.assertTrue(
                (
                    observed["calendar_date"].astype(str)
                    == observed["pressure_observation_date"].astype(str)
                ).all()
            )
            jan_six = field_one[
                field_one["calendar_date"].astype(str).eq("2025-01-06")
            ]
            self.assertNotIn("heat", set(jan_six["hazard_family"]))
            self.assertTrue(
                (jan_six["spectral_source_date"].astype(str) == "2025-01-05").all()
            )
            self.assertTrue(
                (jan_six["s2_acquisition_id"] == "s2-2025-01-05-field-1").all()
            )
            self.assertTrue((jan_six["s2_knowledge_time"].dt.hour == 12).all())

            ledger = pd.read_parquet(output / "pressure_observations_v4.parquet")
            late = ledger[
                ledger["field_id"].eq("field-1")
                & ledger["hazard_family"].eq("heat")
                & ledger["pressure_effective_date"].astype(str).isin(
                    {"2025-01-06", "2025-01-07"}
                )
            ]
            self.assertEqual(len(late), 2)
            self.assertTrue(
                (late["pressure_knowledge_time"].dt.date == date(2025, 1, 8)).all()
            )
            store = _store(output, root)
            before_known = store.v4_field_detail(
                "field-1", as_of_date="2025-01-07", crop_instance_id="crop-1"
            )
            self.assertFalse(any(
                row["hazard_family"] == "heat"
                for row in before_known["daily_pressure"]
            ))
            after_known = store.v4_field_detail(
                "field-1", as_of_date="2025-01-08", crop_instance_id="crop-1"
            )
            revealed_late = [
                row for row in after_known["daily_pressure"]
                if row["hazard_family"] == "heat"
                and str(row["pressure_effective_date"])[:10]
                    in {"2025-01-06", "2025-01-07"}
            ]
            self.assertEqual(len(revealed_late), 2)
            self.assertTrue(all(
                str(row["pressure_knowledge_time"])[:10] == "2025-01-08"
                for row in revealed_late
            ))

    def test_country_completeness_accounts_for_unmappable_fields_as_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source, include_second_geometry=False)
            _write_incident(incident, source)
            _write_evidence(evidence)
            export_incident_viewer_v4(
                incident, evidence, source, output, threads=1,
                min_valid_geometry_coverage=0.5,
                min_frame_geometry_coverage=0.5,
            )

            frame = _store(output, root).v4_frame(
                calendar_date="2025-01-06", bbox=None, limit=1
            )
            meta = frame["meta"]
            self.assertTrue(meta["complete_country_representation"])
            self.assertEqual(meta["source_field_count"], 2)
            self.assertEqual(meta["represented_field_count"], 1)
            self.assertEqual(meta["unmappable_field_count"], 1)
            self.assertEqual(meta["accounted_field_count"], 2)
            self.assertTrue(meta["unmappable_warning"])
            self.assertEqual(meta["warnings"][0]["code"], "unmappable_fields")
            self.assertFalse(meta["country_representation_truncated"])

            manifest_path = output / "manifest.json"
            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = json.loads(json.dumps(original_manifest))
            manifest["artifacts"]["daily_timeline_v4.parquet"]["size_bytes"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                validate_viewer_directory(output)
            manifest = json.loads(json.dumps(original_manifest))
            manifest["artifacts"]["daily_timeline_v4.parquet"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_viewer_directory(output)
            manifest = json.loads(json.dumps(original_manifest))
            manifest["artifacts"]["daily_timeline_v4.parquet"]["row_count"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row_count mismatch"):
                validate_viewer_directory(output)
            with self.assertRaisesRegex(ValueError, "row_count mismatch"):
                _store(output, root)

    def test_current_state_defaults_to_crop_active_on_as_of_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _write_evidence(evidence)
            crop_path = evidence / "crop_day_context_v4.parquet"
            crop = pd.read_parquet(crop_path)
            retired = crop[
                crop["field_id"].eq("field-1")
                & crop["observation_date"].astype(str).eq("2025-01-06")
            ].iloc[0].copy()
            retired["crop_instance_id"] = "crop-retired"
            retired["crop_name"] = "barley"
            pd.concat([crop, pd.DataFrame([retired])], ignore_index=True).to_parquet(
                crop_path, index=False
            )
            _refresh_evidence_artifacts(evidence)
            export_incident_viewer_v4(
                incident, evidence, source, output, threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )
            store = _store(output, root)

            default = store.v4_field_detail("field-1", as_of_date="2025-01-10")
            self.assertEqual(default["current_state_scope"], "crops_active_on_as_of_date")
            self.assertEqual(
                {row["crop_instance_id"] for row in default["current_state"]},
                {"crop-1"},
            )
            explicit = store.v4_field_detail(
                "field-1", as_of_date="2025-01-10",
                crop_instance_id="crop-retired",
            )
            self.assertEqual(
                explicit["current_state_scope"], "explicit_crop_latest_known_state"
            )
            self.assertEqual(
                {row["crop_instance_id"] for row in explicit["current_state"]},
                {"crop-retired"},
            )

    def test_sunday_stage_evidence_known_monday_delays_story_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _write_evidence(evidence)
            crop_path = evidence / "crop_day_context_v4.parquet"
            crop = pd.read_parquet(crop_path)
            crop["knowledge_time"] = pd.to_datetime(crop["knowledge_time"])
            sunday = (
                crop["field_id"].eq("field-1")
                & crop["crop_instance_id"].eq("crop-1")
                & crop["observation_date"].astype(str).eq("2025-01-12")
            )
            crop.loc[sunday, "knowledge_time"] = pd.Timestamp(
                "2025-01-13 08:30:00"
            )
            crop.to_parquet(crop_path, index=False)
            pressure_path = evidence / "field_day_pressure_v4.parquet"
            pressure = pd.read_parquet(pressure_path)
            pressure["knowledge_time"] = pd.to_datetime(pressure["knowledge_time"])
            pressure["weather_available_at"] = pd.to_datetime(
                pressure["weather_available_at"]
            )
            sunday_pressure = (
                pressure["field_id"].eq("field-1")
                & pressure["crop_instance_id"].eq("crop-1")
                & pressure["hazard_family"].eq("heat")
                & pressure["observation_date"].astype(str).eq("2025-01-12")
            )
            pressure.loc[sunday_pressure, "knowledge_time"] = pd.Timestamp(
                "2025-01-13 09:15:00"
            )
            pressure.loc[sunday_pressure, "weather_available_at"] = pd.Timestamp(
                "2025-01-13 09:15:00"
            )
            pressure.to_parquet(pressure_path, index=False)
            s2_path = evidence / "field_s2_acquisition_v4.parquet"
            s2 = pd.read_parquet(s2_path)
            s2["knowledge_time"] = pd.to_datetime(s2["knowledge_time"])
            s2["spectral_source_date"] = pd.to_datetime(s2["spectral_source_date"])
            recovery = s2["acquisition_id"].eq("s2-2025-01-23-field-1")
            s2.loc[recovery, "spectral_source_date"] = pd.Timestamp("2025-01-12")
            s2.loc[recovery, "knowledge_time"] = pd.Timestamp(
                "2025-01-13 10:45:00"
            )
            s2.to_parquet(s2_path, index=False)
            _refresh_evidence_artifacts(evidence)

            export_incident_viewer_v4(
                incident, evidence, source, output, threads=1,
                min_valid_geometry_coverage=1.0,
                min_frame_geometry_coverage=1.0,
            )
            checkpoints = pd.read_parquet(output / "story_checkpoints_v4.parquet")
            first = checkpoints[
                checkpoints["story_week"].astype(str).eq("2025-01-06")
            ].iloc[0]
            self.assertEqual(
                str(first["source_checkpoint_knowledge_time"]),
                "2025-01-06 00:00:00",
            )
            self.assertEqual(
                str(first["crop_context_knowledge_time"]),
                "2025-01-13 08:30:00",
            )
            self.assertEqual(
                str(first["pressure_knowledge_time"]), "2025-01-13 09:15:00"
            )
            self.assertEqual(
                str(first["s2_response_knowledge_time"]), "2025-01-13 10:45:00"
            )
            self.assertEqual(
                str(first["story_known_time"]), "2025-01-13 10:45:00"
            )
            self.assertEqual(str(first["story_known_date"]), "2025-01-13")
            self.assertTrue(first["story_known_time_raised"])
            self.assertEqual(first["checkpoint_bound_mode"], "reconstructed")

            store = _store(output, root)
            sunday_frame = store.v4_frame(
                calendar_date="2025-01-12", bbox=None, limit=5000
            )
            monday_frame = store.v4_frame(
                calendar_date="2025-01-13", bbox=None, limit=5000
            )
            self.assertEqual(sunday_frame["story_footprints"]["features"], [])
            self.assertEqual(len(monday_frame["story_footprints"]["features"]), 1)
            self.assertTrue(
                monday_frame["clocks"]["latest_story_known_time"].startswith(
                    "2025-01-13 10:45:00"
                )
            )
            with self.assertRaises(ResourceNotFoundError):
                store.v4_incident_detail("incident-1", as_of_date="2025-01-12")
            monday_detail = store.v4_incident_detail(
                "incident-1", as_of_date="2025-01-13"
            )
            self.assertEqual(
                str(monday_detail["weekly_state"][0]["story_known_time"])[:19],
                "2025-01-13 10:45:00",
            )

    def test_strict_checkpoint_clock_rejects_inferred_underbound_or_unattributed(self) -> None:
        for scenario, expected in (
            ("inferred", "may not use inferred"),
            ("underbound", "below contributing evidence"),
            ("unattributed", "attribution is incomplete"),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                incident = root / "incident"
                evidence = root / "evidence"
                output = root / "viewer-v4"
                _write_source(source)
                _write_incident(incident, source)
                _write_evidence(evidence)
                _set_evidence_availability_mode(evidence, "strict")
                weekly_path = incident / "incident_weekly_state.parquet"
                weekly = pd.read_parquet(weekly_path)
                changed = [weekly_path]
                if scenario == "inferred":
                    weekly["knowledge_time"] = "2025-01-31 12:00:00"
                    weekly["knowledge_time_inferred"] = True
                elif scenario == "unattributed":
                    weekly["knowledge_time"] = "2025-01-31 12:00:00"
                    membership_path = incident / "incident_membership.parquet"
                    membership = pd.read_parquet(membership_path)
                    membership = membership[
                        ~membership["timeline_bucket"].astype(str).eq("2025-01-06")
                    ]
                    membership.to_parquet(membership_path, index=False)
                    changed.append(membership_path)
                weekly.to_parquet(weekly_path, index=False)
                _refresh_artifact_hashes(incident, *changed)

                with self.assertRaisesRegex(ValueError, expected):
                    export_incident_viewer_v4(
                        incident, evidence, source, output, threads=1,
                        min_valid_geometry_coverage=1.0,
                        min_frame_geometry_coverage=1.0,
                    )
                self.assertFalse(output.exists())

    def test_lifecycle_reconciliation_rejects_response_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _write_evidence(evidence)

            weekly_path = incident / "incident_weekly_state.parquet"
            weekly = pd.read_parquet(weekly_path)
            second_week = weekly["timeline_bucket"].astype(str).eq("2025-01-13")
            weekly.loc[second_week, "fresh_recovery_field_count"] = 1
            weekly.to_parquet(weekly_path, index=False)
            membership_path = incident / "incident_membership.parquet"
            membership = pd.read_parquet(membership_path)
            recovery_member = membership["timeline_bucket"].astype(str).eq("2025-01-13")
            membership.loc[recovery_member, "fresh_response_evidence"] = True
            membership.loc[recovery_member, "response_class"] = "recovery"
            membership.to_parquet(membership_path, index=False)
            _refresh_artifact_hashes(incident, weekly_path, membership_path)

            s2_path = evidence / "field_s2_acquisition_v4.parquet"
            s2 = pd.read_parquet(s2_path)
            candidate = s2["acquisition_id"].eq("s2-2025-01-23-field-1")
            s2.loc[candidate, "spectral_source_date"] = "2025-01-13"
            s2.loc[candidate, "knowledge_time"] = "2025-01-13"
            s2.loc[candidate, "response_class"] = "severe_decline"
            s2.to_parquet(s2_path, index=False)
            _refresh_evidence_artifacts(evidence)

            with self.assertRaisesRegex(
                ValueError, "unsupported_fresh_recovery_claim"
            ):
                export_incident_viewer_v4(
                    incident, evidence, source, output, threads=1,
                    min_valid_geometry_coverage=1.0,
                    min_frame_geometry_coverage=1.0,
                )
            self.assertFalse(output.exists())

    def test_lifecycle_reconciliation_distinguishes_quiet_from_missing_weather(self) -> None:
        for missing_weather, expected in (
            (False, "observed_quiet_for_attributed_members"),
            (True, "missing_weather"),
        ):
            with self.subTest(missing_weather=missing_weather), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                incident = root / "incident"
                evidence = root / "evidence"
                output = root / "viewer-v4"
                _write_source(source)
                _write_incident(incident, source)
                _write_evidence(evidence)
                if missing_weather:
                    pressure_path = evidence / "field_day_pressure_v4.parquet"
                    pressure = pd.read_parquet(pressure_path)
                    dates = pd.to_datetime(pressure["observation_date"])
                    missing = (
                        pressure["field_id"].eq("field-1")
                        & pressure["hazard_family"].eq("heat")
                        & dates.between("2025-01-13", "2025-01-19")
                    )
                    pressure.loc[missing, "pressure_observed"] = False
                    pressure.loc[missing, "pressure_active"] = False
                    pressure.to_parquet(pressure_path, index=False)
                    _refresh_evidence_artifacts(evidence)

                export_incident_viewer_v4(
                    incident, evidence, source, output, threads=1,
                    min_valid_geometry_coverage=1.0,
                    min_frame_geometry_coverage=1.0,
                )
                lifecycle = pd.read_parquet(
                    output / "lifecycle_reconciliation_v4.parquet"
                )
                second = lifecycle[
                    lifecycle["story_week"].astype(str).eq("2025-01-13")
                ].iloc[0]
                self.assertEqual(second["quiet_evidence_status"], expected)
                self.assertFalse(second["component_absence_replayed"])

    def test_daily_grid_uses_modal_current_stage_not_latest_s2_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory, duckdb.connect(":memory:") as con:
            day = pd.Timestamp("2025-01-06")
            rows = []
            geometry = []
            for index in range(4):
                majority = index < 3
                field_id = f"field-{index}"
                rows.append(
                    {
                        "calendar_date": day,
                        "field_id": field_id,
                        "crop_instance_id": f"crop-{index}",
                        "monitored": True,
                        "evaluable": True,
                        "crop_context_observed": True,
                        "stage_source_date": day,
                        "crop_observation_date": day,
                        "s2_knowledge_date": (
                            pd.Timestamp("2025-01-01") if majority
                            else pd.Timestamp("2025-01-06")
                        ),
                        "hazard_family": "heat",
                        "crop_name": "maize" if majority else "beans",
                        "stage_bucket": "vegetative" if majority else "flowering",
                        "pressure_observed": True,
                        "risk_rank": 1,
                        "crop_impact_active": False,
                        "response_class": "no_material_change",
                        "spectral_source_date": pd.Timestamp("2025-01-01"),
                        "evidence_freshness": "fresh",
                    }
                )
                geometry.append(
                    {"field_id": field_id, "centroid_lon": 30.1, "centroid_lat": -1.9}
                )
            con.register("field_day_state_input", pd.DataFrame(rows))
            con.register("geometry_input", pd.DataFrame(geometry))
            con.execute(
                "CREATE VIEW field_day_state_v4 AS SELECT * FROM field_day_state_input"
            )
            con.execute("CREATE VIEW geometry_v4 AS SELECT * FROM geometry_input")
            output = Path(directory) / "grid.parquet"
            _write_daily_grid(con, output, cell_degrees=1.0)
            grid = pd.read_parquet(output)
            self.assertEqual(len(grid), 1)
            self.assertEqual(grid.iloc[0]["dominant_crop_name"], "maize")
            self.assertEqual(grid.iloc[0]["dominant_stage_bucket"], "vegetative")

    def test_missing_required_positive_story_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            incident = root / "incident"
            evidence = root / "evidence"
            output = root / "viewer-v4"
            _write_source(source)
            _write_incident(incident, source)
            _write_evidence(evidence)
            for filename in (
                "field_day_pressure_v4.parquet",
                "field_s2_acquisition_v4.parquet",
            ):
                path = evidence / filename
                pd.read_parquet(path).iloc[0:0].to_parquet(path, index=False)
            _refresh_evidence_artifacts(evidence)

            with self.assertRaisesRegex(
                ValueError, "unsupported_pressure_core_claim"
            ):
                export_incident_viewer_v4(
                    incident,
                    evidence,
                    source,
                    output,
                    threads=1,
                    min_valid_geometry_coverage=1.0,
                    min_frame_geometry_coverage=1.0,
                )
            self.assertFalse(output.exists())


def _write_evidence(root: Path) -> None:
    root.mkdir()
    days = [date(2025, 1, 6) + timedelta(days=offset) for offset in range(20)]
    crop_rows = []
    pressure_rows = []
    for day in days:
        for field, crop_instance, crop, stage in (
            ("field-1", "crop-1", "maize", "flowering"),
            ("field-2", "crop-2", "beans", "vegetative"),
        ):
            active_pressure = (
                field == "field-1" and day < date(2025, 1, 13)
            )
            crop_rows.append(
                {
                    "observation_date": day,
                    "knowledge_time": day,
                    "field_id": field,
                    "crop_instance_id": crop_instance,
                    "crop_name": crop,
                    "crop_season": "2025A",
                    "stage_bucket": stage,
                    "stage_effective_date": day,
                    "crop_context_observed": True,
                }
            )
            pressure_rows.append(
                {
                    "observation_date": day,
                    "knowledge_time": day,
                    "weather_available_at": day,
                    "field_id": field,
                    "crop_instance_id": crop_instance,
                    "hazard_family": "heat",
                    "pressure_score": 0.8 if active_pressure else 0.2,
                    "pressure_rank": 3 if active_pressure else 1,
                    "pressure_band": "MED-HIGH" if active_pressure else "LOW",
                    "pressure_observed": True,
                    "pressure_active": active_pressure,
                }
            )
        if day == date(2025, 1, 6):
            pressure_rows.append(
                {
                    "observation_date": day,
                    "knowledge_time": day,
                    "weather_available_at": day,
                    "field_id": "field-1",
                    "crop_instance_id": "crop-1",
                    "hazard_family": "drought",
                    "pressure_score": 0.6,
                    "pressure_rank": 2,
                    "pressure_band": "LOW-MED",
                    "pressure_observed": True,
                    "pressure_active": True,
                }
            )
    pd.DataFrame(crop_rows).to_parquet(root / "crop_day_context_v4.parquet", index=False)
    pd.DataFrame(pressure_rows).to_parquet(root / "field_day_pressure_v4.parquet", index=False)
    pd.DataFrame(
        [
            {
                "field_id": "field-1", "crop_instance_id": "crop-1",
                "acquisition_id": "s2-2025-01-05-field-1",
                "knowledge_time": "2025-01-06", "spectral_source_date": "2025-01-05",
                "acquisition_attempted": True,
                "spectral_usable": True, "spectral_echo_days": 1,
                "acquisition_status": "usable",
                "s2_field_quality_flag": "pass",
                "spectral_freshness": "fresh", "reference_source_date": "2024-12-25",
                "response_class": "medium_decline", "new_response_evidence": True,
                "crop_name": "maize", "stage_bucket": "flowering",
            },
            {
                "field_id": "field-1", "crop_instance_id": "crop-1",
                "acquisition_id": "echo-2025-01-07-field-1",
                "knowledge_time": "2025-01-07", "spectral_source_date": "2025-01-05",
                "acquisition_attempted": False,
                "spectral_usable": True, "spectral_echo_days": 2,
                "acquisition_status": "echo",
                "response_class": "no_new_acquisition", "new_response_evidence": False,
            },
            {
                "field_id": "field-2", "crop_instance_id": "crop-2",
                "acquisition_id": "s2-2025-01-06-field-2",
                "knowledge_time": "2025-01-07", "spectral_source_date": "2025-01-06",
                "acquisition_attempted": True,
                "spectral_usable": True, "spectral_echo_days": 1,
                "acquisition_status": "usable",
                "s2_field_quality_flag": "pass",
                "spectral_freshness": "fresh", "reference_source_date": "2024-12-25",
                "response_class": "medium_decline", "new_response_evidence": True,
                "crop_name": "beans", "stage_bucket": "vegetative",
            },
            {
                "field_id": "field-1", "crop_instance_id": "crop-1",
                "acquisition_id": "s2-2025-01-08-field-1",
                "knowledge_time": "2025-01-08", "spectral_source_date": "2025-01-08",
                "acquisition_attempted": True,
                "spectral_usable": False, "acquisition_status": "rejected",
                "s2_field_quality_flag": "cloud_or_quality_mask",
                "response_class": "spectral_missing", "new_response_evidence": False,
            },
            {
                "field_id": "field-1", "crop_instance_id": "crop-1",
                "acquisition_id": "s2-2025-01-23-field-1",
                "knowledge_time": "2025-01-24", "spectral_source_date": "2025-01-23",
                "acquisition_attempted": True,
                "spectral_usable": True, "spectral_echo_days": 1,
                "acquisition_status": "usable",
                "s2_field_quality_flag": "pass",
                "spectral_freshness": "fresh", "reference_source_date": "2025-01-05",
                "response_class": "recovery", "new_response_evidence": True,
                "crop_name": "maize", "stage_bucket": "flowering",
            },
        ]
    ).to_parquet(root / "field_s2_acquisition_v4.parquet", index=False)
    policy_version = "v4-test"
    policy_sha256 = "b" * 64
    for filename in (
        "crop_day_context_v4.parquet",
        "field_day_pressure_v4.parquet",
        "field_s2_acquisition_v4.parquet",
    ):
        path = root / filename
        frame = pd.read_parquet(path)
        if filename == "field_s2_acquisition_v4.parquet":
            frame = frame[~frame["acquisition_id"].astype(str).str.startswith("echo-")]
            frame["reference_acquisition_id"] = pd.Series(
                pd.NA, index=frame.index, dtype="string"
            )
            has_reference = frame["acquisition_id"].eq("s2-2025-01-23-field-1")
            frame.loc[~has_reference, "new_response_evidence"] = False
            frame.loc[
                has_reference, "reference_acquisition_id"
            ] = "s2-2025-01-05-field-1"
        frame["policy_version"] = policy_version
        frame["policy_sha256"] = policy_sha256
        frame["availability_mode"] = "reconstructed"
        frame.to_parquet(path, index=False)
    files = {
        "crop": "crop_day_context_v4.parquet",
        "pressure": "field_day_pressure_v4.parquet",
        "s2": "field_s2_acquisition_v4.parquet",
    }
    artifacts = {}
    for label, filename in files.items():
        path = root / filename
        artifacts[label] = {
            "name": filename,
            "size_bytes": path.stat().st_size,
            "row_count": len(pd.read_parquet(path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": "complete",
                    "immutable": True,
                    "source_generation_id": "generation-test",
                    "release_as_of": "2025-01-31",
                    "as_of_date": "2025-01-31",
                    "released_at": "2025-01-31T23:59:59Z",
                },
                "correction_policy": CORRECTION_POLICY,
                "availability": {"mode": "reconstructed"},
                "policy": {"version": policy_version, "sha256": policy_sha256},
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _add_causal_regression_cases(root: Path) -> None:
    membership_path = root / "incident_membership.parquet"
    membership = pd.read_parquet(membership_path)
    future_member = membership["field_id"].eq("field-2")
    membership.loc[future_member, "timeline_bucket"] = "2025-01-13"
    membership.loc[future_member, "knowledge_time"] = "2025-01-13"
    membership.to_parquet(membership_path, index=False)

    windows_path = root / "incident_windows.parquet"
    windows = pd.read_parquet(windows_path)
    windows.loc[:, "closed_week"] = "2025-01-20"
    windows.loc[:, "terminal_state"] = "CLOSED_RECOVERED"
    windows.loc[:, "relapse_count"] = 9
    windows.to_parquet(windows_path, index=False)

    lineage_path = root / "incident_lineage.parquet"
    pd.DataFrame(
        [
            {
                "timeline_bucket": "2025-01-06",
                "knowledge_time": "2025-01-06",
                "lineage_type": "merge",
                "parent_incident_id": "incident-past",
                "child_incident_id": "incident-1",
            },
            {
                "timeline_bucket": "2025-01-20",
                "knowledge_time": "2025-01-20",
                "lineage_type": "split",
                "parent_incident_id": "incident-1",
                "child_incident_id": "incident-future",
            },
        ]
    ).to_parquet(lineage_path, index=False)
    _refresh_artifact_hashes(root, membership_path, windows_path, lineage_path)


def _refresh_artifact_hashes(root: Path, *paths: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in paths:
        manifest["artifacts"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh_evidence_artifacts(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for label, filename in {
        "crop": "crop_day_context_v4.parquet",
        "pressure": "field_day_pressure_v4.parquet",
        "s2": "field_s2_acquisition_v4.parquet",
    }.items():
        path = root / filename
        manifest["artifacts"][label] = {
            "name": filename,
            "size_bytes": path.stat().st_size,
            "row_count": len(pd.read_parquet(path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _set_evidence_availability_mode(root: Path, mode: str) -> None:
    for filename in (
        "crop_day_context_v4.parquet",
        "field_day_pressure_v4.parquet",
        "field_s2_acquisition_v4.parquet",
    ):
        path = root / filename
        frame = pd.read_parquet(path)
        frame["availability_mode"] = mode
        frame.to_parquet(path, index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["availability"]["mode"] = mode
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_evidence_artifacts(root)


def _store(run_dir: Path, static_dir: Path) -> StoryMapStore:
    return StoryMapStore(
        Settings(
            run_dir=run_dir,
            static_dir=static_dir,
            host="127.0.0.1",
            port=8877,
            raster_tiles="",
            raster_attribution="",
            default_feature_limit=5000,
            max_feature_limit=20000,
            log_level="INFO",
        )
    )


def _http_json(port: int, path: str) -> dict:
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise AssertionError(
                f"GET {path} returned HTTP {response.status}: {body!r}"
            )
        return json.loads(body)
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
