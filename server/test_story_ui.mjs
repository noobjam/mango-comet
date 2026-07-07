import assert from "node:assert/strict";
import test from "node:test";

import { ApiClient } from "./static/api.js";
import { FrameDataLoader, GEOMETRY_BATCH_SIZE, mergeFrameState } from "./static/frame-data.js";
import { computeActivityStats } from "./static/activity.js";
import { EvolutionController } from "./static/evolution-controller.js";
import {
  FootprintCollectionCache,
  loadFootprintCollection,
} from "./static/footprint-cache.js";
import { shouldLoadHistory } from "./static/history.js";
import {
  INCIDENT_TRUTH_LABEL,
  adjacentBuckets,
  assertCompleteFootprintCollection,
  coincidentIncidentCandidates,
  fieldViewportCoverage,
  footprintRoleCollection,
  footprintVisualModel,
  incidentDetailModel,
  incidentHitCandidates,
  isCropIncident,
  isCropIncidentV3,
  isCropIncidentV4,
  nextIncidentCandidate,
  normalizeFootprintCollection,
  shouldLoadEvolution,
  v3LayerModel,
  v4LayerModel,
} from "./static/incident-v3.js";
import {
  footprintHistoryVisualModel,
  incidentFootprintHistory,
  incidentSelectionTransition,
  incidentStoryArc,
} from "./static/incident-story.js";
import {
  lifecycleModel,
  prefixTrajectoryModel,
} from "./static/inspector.js";
import {
  cropStoryTrajectoryModel,
  fieldEvidenceRibbonModel,
} from "./static/crop-story-trajectory-model.js";
import { INCIDENT_FACETS, INCIDENT_MODEL_STATUS } from "./static/filters.js";
import { buildEvolutionModel } from "./static/map-evolution.js";
import { MapView } from "./static/map-view.js";
import { alphaForState, lineColorFor } from "./static/palette.js";
import { activityCount, timelineDateLabel } from "./static/timeline.js";

test("compact state merges cached static geometry with dynamic properties", () => {
  const geometry = new Map([["A", {
    type: "Feature",
    geometry: { type: "Point", coordinates: [30, -1] },
    properties: { field_id: "A", district: "D", max_risk_band: "OLD" },
  }]]);
  const frame = mergeFrameState({
    geometry_version: "v1",
    rows: [{ field_id: "A", max_risk_band: "HIGH" }, { field_id: "missing" }],
    meta: { state_count: 2 },
  }, geometry, new Set(["missing"]));
  assert.equal(frame.features.length, 1);
  assert.equal(frame.features[0].properties.district, "D");
  assert.equal(frame.features[0].properties.max_risk_band, "HIGH");
  assert.equal(frame.meta.geometry_missing_count, 1);
});

test("geometry is requested in bounded batches and reused across dates", async () => {
  const fieldIds = Array.from({ length: GEOMETRY_BATCH_SIZE + 1 }, (_, index) => `field-${index}`);
  const posts = [];
  const api = {
    abortPrefix() {},
    preload() {},
    async get() {
      return { geometry_version: "v1", rows: fieldIds.map((field_id) => ({ field_id })), meta: {} };
    },
    async post(_url, body) {
      posts.push(body.field_ids);
      return {
        geometry_version: "v1",
        type: "FeatureCollection",
        features: body.field_ids.map((field_id) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [30, -1] },
          properties: { field_id },
        })),
        meta: { missing_field_ids: [] },
      };
    },
  };
  const loader = new FrameDataLoader(api);
  const first = await loader.load({ bucket: "2025-01-01", legacyUrl: "/legacy" });
  const second = await loader.load({ bucket: "2025-01-08", legacyUrl: "/legacy" });
  assert.equal(first.features.length, fieldIds.length);
  assert.equal(second.features.length, fieldIds.length);
  assert.deepEqual(posts.map((batch) => batch.length), [GEOMETRY_BATCH_SIZE, 1]);
});

test("unsupported compact endpoints fall back to the legacy frame", async () => {
  const legacy = { type: "FeatureCollection", features: [], meta: { transport: "legacy" } };
  const api = {
    abortPrefix() {},
    preload() {},
    async get(url) {
      if (url.startsWith("/api/frame-state/")) throw Object.assign(new Error("not found"), { status: 404 });
      return legacy;
    },
  };
  const loader = new FrameDataLoader(api);
  assert.equal(await loader.load({ bucket: "2025-01-01", legacyUrl: "/legacy" }), legacy);
  assert.equal(loader.compactSupported, false);
});

test("V4 compact state path hydrates cached geometry without using the polygon frame", async () => {
  const gets = [];
  const api = {
    abortPrefix() {},
    preload() {},
    async get(url) {
      gets.push(url);
      return {
        geometry_version: "v4-geometry",
        rows: [{ field_id: "field-1", timeline_bucket: "2025-01-06" }],
        meta: {},
      };
    },
    async post() {
      return {
        geometry_version: "v4-geometry",
        features: [{
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [] },
          properties: { field_id: "field-1" },
        }],
        meta: { missing_field_ids: [] },
      };
    },
  };
  const loader = new FrameDataLoader(api);
  const frame = await loader.load({
    bucket: "2025-01-06",
    query: { bbox: "29,-2,30,-1" },
    statePath: "/api/v4/frame-state",
    legacyUrl: "/api/v4/frame/2025-01-06",
    selectLegacy: (payload) => payload.fields,
  });
  assert.match(gets[0], /^\/api\/v4\/frame-state\/2025-01-06\?/);
  assert.equal(gets.includes("/api/v4/frame/2025-01-06"), false);
  assert.equal(frame.features.length, 1);
  assert.equal(frame.meta.transport, "compact-state-plus-geometry");
});

test("stale geometry responses cannot repopulate a newer-version cache", () => {
  const loader = new FrameDataLoader({});
  loader.geometryVersion = "new";
  assert.throws(() => loader.rememberGeometry({
    geometry_version: "old",
    features: [{
      type: "Feature",
      geometry: { type: "Point", coordinates: [0, 0] },
      properties: { field_id: "A" },
    }],
    meta: {},
  }, "old"), /Geometry version changed/);
  assert.equal(loader.geometry.size, 0);
});

test("activity-center trail breaks on weak overlap and missing weeks", () => {
  const model = buildEvolutionModel({ buckets: [
    { timeline_bucket: "2025-01-01", center_lon: 30, center_lat: -1, field_count: 10 },
    { timeline_bucket: "2025-01-08", center_lon: 30.1, center_lat: -1, field_count: 11, jaccard_overlap: 0.6, p90_dispersion_km: 12.5 },
    { timeline_bucket: "2025-01-15", center_lon: 30.2, center_lat: -1, field_count: 8, jaccard_overlap: 0.1 },
    { timeline_bucket: "2025-01-29", center_lon: 30.3, center_lat: -1, field_count: 9, jaccard_overlap: 0.9 },
  ] }, "2025-01-29");
  assert.equal(model.points.length, 4);
  assert.equal(model.segments.length, 1);
  assert.equal(model.segments[0].end.bucket, "2025-01-08");
  assert.equal(model.points[1].dispersionP90, 12.5);
  assert.equal(model.dots.length, 11);
});

test("field lifecycle normalizes event spans and selected week", () => {
  const model = lifecycleModel([
    { event_start_date: "2025-01-01", active_end_date: "2025-01-05", max_risk_band: "MEDIUM" },
    { event_start_date: "2025-01-10", active_end_date: "2025-01-20", max_risk_band: "HIGH" },
  ], "2025-01-12");
  assert.equal(model.items.length, 2);
  assert.equal(model.items[0].selected, false);
  assert.equal(model.items[1].selected, true);
  assert.ok(model.items[1].startPercent > model.items[0].startPercent);
});

test("open lifecycle extends to the generation cutoff, beyond last pressure", () => {
  const model = lifecycleModel([
    {
      event_start_date: "2025-01-01", active_end_date: "2025-01-05",
      as_of_date: "2025-01-20", right_censored: true,
    },
  ], "2025-01-12");
  assert.equal(new Date(model.end).toISOString().slice(0, 10), "2025-01-20");
  assert.equal(model.items[0].selected, true);
  assert.ok(model.items[0].pressurePercent < 100);
});

test("causal prefix trajectory preserves gaps and selected week", () => {
  const states = prefixTrajectoryModel([
    { timeline_bucket: "2025-01-01", event_state: "ACTIVE" },
    { timeline_bucket: "2025-01-08", event_state: "QUIET_PENDING" },
    { timeline_bucket: "2025-01-29", event_state: "CLOSED_RECOVERED" },
  ], "2025-01-08");
  assert.equal(states[0].gapBefore, false);
  assert.equal(states[1].selected, true);
  assert.equal(states[2].gapBefore, true);
});

test("V4 trajectory gives every hazard a distinct lane and distinguishes low from missing", () => {
  const model = fieldEvidenceRibbonModel({
    as_of_date: "2025-01-03",
    history: { window_start: "2025-01-01", window_end: "2025-01-03" },
    lanes: {
      daily_pressure: [
        { calendar_date: "2025-01-01", hazard_family: "heat", risk_rank: 1, pressure_observed: true },
        { calendar_date: "2025-01-03", hazard_family: "heat", risk_rank: 3, pressure_observed: true },
        { calendar_date: "2025-01-01", hazard_family: "drought", risk_rank: 2, pressure_observed: true },
        { calendar_date: "2025-01-01", hazard_family: "damaging_wind", risk_rank: 2, pressure_observed: true },
        { calendar_date: "2025-01-01", hazard_family: "flooding", risk_rank: 2, pressure_observed: true },
        { calendar_date: "2025-01-01", hazard_family: "ponding", risk_rank: 2, pressure_observed: true },
      ],
    },
  }, "2025-01-03");
  assert.equal(model.hazards.length, 5);
  assert.equal(new Set(model.lanes.pressure.map((lane) => lane.hazard)).size, 5);
  const heat = model.lanes.pressure.find((lane) => lane.hazard === "heat");
  assert.deepEqual(heat.cells.map((cell) => cell.state), [
    "observed-low", "missing", "elevated",
  ]);
});

test("V4 trajectory separates S2 source and knowledge clocks and step-held freshness", () => {
  const model = cropStoryTrajectoryModel({
    as_of_date: "2025-02-10",
    history: { window_start: "2025-01-01", window_end: "2025-02-10" },
    s2_attempts: [
      {
        field_id: "f1", crop_instance_id: "c1", spectral_source_date: "2025-01-02",
        knowledge_date: "2025-01-03", marker_type: "acquisition",
        spectral_usable: true, response_class: "medium_decline",
      },
      {
        field_id: "f2", crop_instance_id: "c2", spectral_source_date: "2025-01-08",
        knowledge_date: "2025-01-09", marker_type: "rejected", spectral_usable: false,
      },
      {
        field_id: "f3", crop_instance_id: "c3", spectral_source_date: "2025-01-15",
        knowledge_date: "2025-01-16", marker_type: "acquisition",
        spectral_usable: true, response_class: "recovery",
      },
      {
        field_id: "f4", crop_instance_id: "c4", spectral_source_date: "2025-01-22",
        knowledge_date: "2025-01-23", marker_type: "acquisition",
        spectral_usable: true, response_class: "no_change",
      },
    ],
  }, "2025-02-10");
  assert.ok(model.lanes.s2.every((event) => event.sourceX <= event.knowledgeX));
  assert.deepEqual(new Set(model.lanes.s2.map((event) => event.responseKind)), new Set([
    "decline", "rejected", "recovery", "no-change",
  ]));
  assert.deepEqual(new Set(model.lanes.s2Holds.map((hold) => hold.freshness)), new Set([
    "fresh", "aging", "stale",
  ]));
});

test("V4 trajectory uses release freshness policy and rejected S2 cannot change crop stage", () => {
  const base = {
    as_of_date: "2025-01-20",
    history: { window_start: "2025-01-01", window_end: "2025-01-20" },
    s2_attempts: [
      {
        field_id: "f1", crop_instance_id: "c1", spectral_source_date: "2025-01-01",
        knowledge_date: "2025-01-02", marker_type: "acquisition",
        spectral_usable: true, response_class: "no_change", stage_bucket: "vegetative",
      },
      {
        field_id: "f2", crop_instance_id: "c2", spectral_source_date: "2025-01-05",
        knowledge_date: "2025-01-06", marker_type: "rejected",
        spectral_usable: false, response_class: "no_change", stage_bucket: "poison_stage",
      },
    ],
  };
  const defaults = cropStoryTrajectoryModel(base);
  const configured = cropStoryTrajectoryModel({
    ...base,
    freshness_policy: { fresh_max_days: 2, aging_max_days: 4 },
  });
  const defaultFresh = defaults.lanes.s2Holds.find((hold) => hold.freshness === "fresh");
  const configuredFresh = configured.lanes.s2Holds.find((hold) => hold.freshness === "fresh");
  assert.ok(configuredFresh.width < defaultFresh.width);
  assert.ok(configured.lanes.stage.some((segment) => segment.stage === "vegetative"));
  assert.ok(!configured.lanes.stage.some((segment) => segment.stage === "poison_stage"));
});

test("V4 trajectory keeps the full crop-stage band and one labelled row per story", () => {
  const model = cropStoryTrajectoryModel({
    as_of_date: "2025-01-31",
    history: { window_start: "2025-01-01", window_end: "2025-01-31" },
    story_checkpoints: [
      { incident_id: "incident-maize", crop_name: "maize", hazard_family: "heat", story_week: "2025-01-01", story_known_date: "2025-01-03", incident_state: "ACTIVE", stage_bucket: "vegetative" },
      { incident_id: "incident-maize", crop_name: "maize", hazard_family: "heat", story_week: "2025-01-08", story_known_date: "2025-01-10", incident_state: "RECOVERING", stage_bucket: "flowering" },
      { incident_id: "incident-maize", crop_name: "maize", hazard_family: "heat", story_week: "2025-01-15", story_known_date: "2025-01-17", incident_state: "CLOSED_RECOVERED", stage_bucket: "mature" },
      { incident_id: "incident-beans", crop_name: "beans", hazard_family: "drought", story_week: "2025-01-05", story_known_date: "2025-01-07", incident_state: "CANDIDATE", stage_bucket: "emergence" },
    ],
  }, "2025-01-31");
  assert.equal(model.lanes.stories.length, 2);
  assert.equal(new Set(model.lanes.stories.map((story) => story.label)).size, 2);
  assert.equal(model.lanes.stage[0].stage, "unknown");
  assert.ok(model.lanes.stage.some((segment) => segment.stage === "vegetative"));
  assert.ok(model.lanes.stage.some((segment) => segment.stage === "flowering"));
  const maize = model.lanes.stories.find((story) => story.incidentId === "incident-maize");
  assert.deepEqual(new Set(maize.milestones.map((item) => item.kind)), new Set([
    "start", "recovery", "closed",
  ]));
  const finalStage = model.lanes.stage.at(-1);
  assert.ok(Math.abs(finalStage.startX + finalStage.width - 100) < 0.01);
});

test("V4 incident S2 aggregation bounds the DOM model without losing counts", () => {
  const start = Date.parse("2020-01-01T00:00:00Z");
  const attempts = Array.from({ length: 1000 }, (_, index) => {
    const source = new Date(start + index * 86400000).toISOString().slice(0, 10);
    const known = new Date(start + (index + 1) * 86400000).toISOString().slice(0, 10);
    const responses = ["medium_decline", "severe_decline", "recovery", "no_change"];
    return {
      field_id: `field-${index}`,
      crop_instance_id: `crop-${index}`,
      spectral_source_date: source,
      knowledge_date: known,
      marker_type: index % 11 === 0 ? "rejected" : "acquisition",
      spectral_usable: index % 11 !== 0,
      response_class: responses[index % responses.length],
    };
  });
  const model = cropStoryTrajectoryModel({
    as_of_date: "2022-12-31",
    history: { window_start: "2020-01-01", window_end: "2022-12-31" },
    s2_attempts: attempts,
  });
  assert.equal(model.s2Aggregated, true);
  assert.ok(model.lanes.s2.length <= 240);
  assert.ok(model.lanes.s2Holds.length <= 240);
  assert.equal(model.lanes.s2.reduce((sum, event) => sum + event.count, 0), 1000);
  assert.ok(model.lanes.s2.every((event) => event.aggregateRange === true));
  assert.deepEqual(model.ticks.map((tick) => tick.x), [0, 25, 50, 75, 100]);
});

test("V4 stage and story trajectory visuals stay bounded while retaining endpoints", () => {
  const start = Date.parse("2020-01-01T00:00:00Z");
  const checkpoints = Array.from({ length: 1000 }, (_, index) => ({
    incident_id: `incident-${index % 100}`,
    crop_name: "maize",
    hazard_family: "heat",
    story_week: new Date(start + index * 86400000).toISOString().slice(0, 10),
    story_known_date: new Date(start + index * 86400000).toISOString().slice(0, 10),
    incident_state: index % 2 ? "ACTIVE" : "RECOVERING",
    stage_bucket: index % 3 ? "vegetative" : "flowering",
  }));
  const model = cropStoryTrajectoryModel({
    as_of_date: "2022-12-31",
    history: { window_start: "2020-01-01", window_end: "2022-12-31" },
    story_checkpoints: checkpoints,
  });
  assert.ok(model.lanes.stories.length <= 64);
  assert.ok(model.lanes.stage.length <= 120);
  assert.equal(model.lanes.stage[0].startX, 0);
  assert.ok(model.lanes.stage.every((segment, index, values) => (
    index === values.length - 1
      ? Math.abs(segment.startX + segment.width - 100) < 0.0001
      : Math.abs(segment.startX + segment.width - values[index + 1].startX) < 0.0001
  )));
  assert.ok(Math.abs(model.lanes.stage.reduce(
    (sum, segment) => sum + segment.width, 0,
  ) - 100) < 0.0001);
  assert.ok(model.lanes.stories.reduce(
    (sum, story) => sum + story.milestones.length, 0,
  ) <= 256);
  assert.ok(model.lanes.stories.every((story) => story.blocks.length > 0));
  assert.equal(model.counts.storySourceLanes, 100);
  assert.equal(model.counts.storyLanes, 64);
  assert.equal(model.counts.storyLanesOmitted, 36);
});

test("history is fetched only for an enabled filtered comparison", () => {
  const manifest = { server: { optimized_geometry: true } };
  assert.equal(shouldLoadHistory(manifest, { motif_family: "heat" }, false), false);
  assert.equal(shouldLoadHistory(manifest, {}, true), false);
  assert.equal(shouldLoadHistory(manifest, { motif_family: "heat" }, true), true);
  assert.equal(shouldLoadHistory({ server: { optimized_geometry: false } }, { motif_family: "heat" }, true), false);
});

test("closed field states are not counted as open now", () => {
  const stats = computeActivityStats({
    features: [
      { properties: { field_id: "open", event_state: "ACTIVE" } },
      { properties: { field_id: "closed", event_state: "CLOSED_RECOVERED" } },
    ],
    meta: { timeline_bucket: "2025-01-08" },
  }, { features: [], meta: {} });
  assert.equal(stats.affected, 1);
});

test("incident story activity keeps membership-free lifecycle weeks active", () => {
  assert.equal(activityCount({
    field_count: 0, story_cluster_count: 2, activity_unit: "incident_stories",
  }), 2);
  assert.equal(activityCount({ field_count: 0, incident_count: 1, activity_count: 1 }), 1);
  assert.equal(activityCount({ field_count: 3 }), 3);
  assert.equal(activityCount({ field_count: 3, story_cluster_count: 1 }), 3);
});

test("diagnostic archetype states are visibly distinct", () => {
  assert.ok(alphaForState({ archetype_display_state: "pending_anchor" }, 188) < 100);
  assert.deepEqual(lineColorFor({ archetype_display_state: "novel_unassigned" }), [251, 146, 60, 245]);
  assert.notDeepEqual(
    lineColorFor({ archetype_display_state: "accepted" }),
    lineColorFor({ archetype_display_state: "pending_anchor" }),
  );
});

test("evolution requests are coalesced while a filter query is in flight", async () => {
  let resolveRequest;
  let calls = 0;
  const api = {
    get() {
      calls += 1;
      return new Promise((resolve) => { resolveRequest = resolve; });
    },
  };
  const controller = new EvolutionController(api, {
    section: {}, summary: {}, status: {},
  });
  const first = controller.load({ hazard_signature: "heat" });
  const second = controller.load({ hazard_signature: "heat" });
  await Promise.resolve();
  assert.equal(calls, 1);
  resolveRequest({ points: [{ timeline_bucket: "2025-01-01" }] });
  assert.deepEqual(await first, await second);
  assert.equal(calls, 1);
});

test("crop incident V3 is detected at either manifest mode location", () => {
  assert.equal(isCropIncidentV3({ mode: "crop_incident_v3" }), true);
  assert.equal(isCropIncidentV3({ run: { mode: "crop_incident_v3" } }), true);
  assert.equal(isCropIncidentV3({ mode: "legacy" }), false);
  assert.equal(shouldLoadEvolution({ mode: "crop_incident_v3" }), false);
  assert.equal(shouldLoadEvolution({ mode: "legacy" }), true);
});

test("dual-clock V4 mode uses daily labels and audited country representation", () => {
  const manifest = { mode: "crop_incident_v4_dual_clock" };
  assert.equal(isCropIncidentV4(manifest), true);
  assert.equal(isCropIncident(manifest), true);
  assert.equal(isCropIncidentV3(manifest), false);
  assert.equal(shouldLoadEvolution(manifest), false);
  assert.match(timelineDateLabel("daily", "2026-05-17"), /^As of /);
  assert.match(timelineDateLabel("weekly", "2026-05-11"), /^Week of /);
  const overview = v4LayerModel(8);
  assert.equal(overview.fieldOverview.visible, true);
  assert.equal(overview.fieldOverview.completeness, "api-audited");
  assert.equal(overview.fields.visible, false);
  assert.equal(v4LayerModel(12).fields.visible, true);
  assert.equal(overview.pressure.clock, "daily");
  assert.equal(overview.cropImpact.clock, "s2-step-held");
  assert.equal(overview.story.clock, "weekly-knowledge-gated");
});

test("V4 controls expose hazard filtering, model status, and field cap truth", () => {
  assert.ok(INCIDENT_FACETS.some(([, key]) => key === "hazard_family"));
  assert.match(INCIDENT_MODEL_STATUS, /operational crop stories/i);
  assert.match(INCIDENT_MODEL_STATUS, /motifs are not published/i);
  const capped = fieldViewportCoverage({
    features: [{}, {}, {}],
    meta: { feature_count: 3, source_field_count: 12, truncated: true },
  });
  assert.deepEqual(capped, {
    shown: 3,
    source: 12,
    truncated: true,
    label: "showing 3 of 12 viewport fields · capped",
  });
  assert.equal(fieldViewportCoverage({
    features: [{}, {}], meta: { source_field_count: 2 },
  }).truncated, false);
});

test("V3 layer model keeps exact complete footprints primary and hides movement trails", () => {
  const overview = v3LayerModel(8);
  assert.equal(overview.primaryLayer, "exact-complete-incident-footprints");
  assert.deepEqual(overview.footprints, { visible: true, exact: true, complete: true });
  assert.equal(overview.fields.visible, false);
  assert.equal(overview.evolution.visible, false);
  assert.equal(v3LayerModel(12).fields.visible, true);

  const geometry = { type: "Polygon", coordinates: [[[30, -1], [31, -1], [30, -1]]] };
  const pressureGeometry = { type: "Polygon", coordinates: [[[30, -1], [30.5, -1], [30, -1]]] };
  const normalized = normalizeFootprintCollection({
    rows: [{ incident_id: "i-1", geometry, pressure_geometry: pressureGeometry }],
  });
  assert.equal(normalized.features[0].geometry, geometry);
  assert.equal(footprintRoleCollection(normalized, "pressure").features[0].geometry, pressureGeometry);
});

test("V3 overview refuses capped or incomplete footprint payloads", () => {
  const complete = {
    features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [] }, properties: {} }],
    meta: {
      complete: true,
      truncated: false,
      feature_cap_applied: false,
      low_zoom_footprints_dropped: false,
      feature_count: 1,
      footprint_geometry_method: "exact_union_of_grid_rectangles",
    },
  };
  assert.equal(assertCompleteFootprintCollection(complete), complete);
  assert.throws(
    () => assertCompleteFootprintCollection({ ...complete, meta: { ...complete.meta, truncated: true } }),
    /incomplete/i,
  );
  assert.throws(
    () => assertCompleteFootprintCollection({ ...complete, meta: { ...complete.meta, feature_count: 2 } }),
    /incomplete/i,
  );
});

test("V3 footprint styles distinguish live, carried, quiet, and recovering evidence", () => {
  assert.equal(footprintVisualModel({ incident_state: "ACTIVE", pressure_core_field_count: 2 }).key, "pressure");
  assert.equal(footprintVisualModel({ incident_state: "CONFIRMED", pressure_core_field_count: 0, impact_lag_field_count: 1 }).key, "impact");
  assert.match(
    footprintVisualModel({ incident_state: "CONFIRMED", pressure_core_field_count: 0, impact_lag_field_count: 1 }).label,
    /without current pressure/i,
  );
  assert.equal(footprintVisualModel({ incident_state: "CANDIDATE", pressure_core_field_count: 0, watch_frontier_field_count: 2 }).key, "watch");
  assert.equal(footprintVisualModel({ footprint_carried_forward: true }).key, "carried");
  assert.equal(footprintVisualModel({ incident_state: "QUIET_PENDING" }).key, "quiet");
  assert.equal(footprintVisualModel({ incident_state: "RECOVERING" }).key, "recovering");
  assert.match(INCIDENT_TRUTH_LABEL, /not physical movement/i);
});

test("V3 timeline prefetch targets both adjacent weeks", () => {
  const buckets = [
    { timeline_bucket: "2026-01-05" },
    { timeline_bucket: "2026-01-12" },
    { timeline_bucket: "2026-01-19" },
  ];
  assert.deepEqual(
    adjacentBuckets(buckets, 1).map((row) => row.timeline_bucket),
    ["2026-01-05", "2026-01-19"],
  );
  assert.deepEqual(
    adjacentBuckets(buckets, 0).map((row) => row.timeline_bucket),
    ["2026-01-12"],
  );
});

test("V4 footprint and story histories never cross the selected knowledge day", () => {
  const geometry = { type: "Polygon", coordinates: [[[30, -1], [31, -1], [30, -1]]] };
  const footprints = [
    { incident_id: "i-1", story_week: "2026-01-05", story_known_date: "2026-01-11", geometry },
    { incident_id: "i-1", story_week: "2026-01-12", story_known_date: "2026-01-18", geometry },
  ];
  const beforeSecond = incidentFootprintHistory(footprints, "2026-01-17", "i-1");
  assert.equal(beforeSecond.current.properties.story_week, "2026-01-05");
  assert.equal(beforeSecond.collection.features.length, 0);
  const afterSecond = incidentFootprintHistory(footprints, "2026-01-18", "i-1");
  assert.equal(afterSecond.current.properties.story_week, "2026-01-12");
  assert.equal(afterSecond.collection.features.length, 1);

  const arc = incidentStoryArc([
    { timeline_bucket: "2026-01-05", story_known_date: "2026-01-11", incident_state: "ACTIVE" },
    { timeline_bucket: "2026-01-12", story_known_date: "2026-01-18", incident_state: "RECOVERING" },
  ], [], "2026-01-17");
  assert.equal(arc.length, 1);
  assert.equal(arc[0].knownDate, "2026-01-11");
});

test("selected crop incident persists while the timeline week changes", () => {
  assert.deepEqual(incidentSelectionTransition({
    incidentMode: true,
    selectedIncidentId: "incident-1",
    selectionBucket: "2026-01-05",
    nextBucket: "2026-01-12",
  }), {
    changed: true,
    preserveIncident: true,
    clearSelection: false,
    nextSelectionBucket: "2026-01-12",
  });
  assert.equal(incidentSelectionTransition({
    incidentMode: false,
    selectedIncidentId: "incident-1",
    selectionBucket: "2026-01-05",
    nextBucket: "2026-01-12",
  }).clearSelection, true);
});

test("normalized countrywide footprints are reused across bbox-only field pans", async () => {
  let loads = 0;
  let normalizations = 0;
  const cache = new FootprintCollectionCache({
    normalize(payload) {
      normalizations += 1;
      return { ...payload, normalized: true };
    },
  });
  const key = "/api/incident-footprints/2026-01-12?crop_name=maize";
  const loader = async () => {
    loads += 1;
    return { features: [] };
  };
  const first = await cache.load(key, loader);
  const second = await cache.load(key, loader);
  assert.equal(loads, 1);
  assert.equal(normalizations, 1);
  assert.equal(first, second);
});

test("countrywide footprint cache evicts by estimated bytes", async () => {
  let loads = 0;
  const cache = new FootprintCollectionCache({
    limit: 10,
    maxBytes: 10,
    estimateBytes: (value) => value.bytes,
  });
  const first = await cache.load("first", async () => ({ bytes: 6 }));
  await cache.load("second", async () => ({ bytes: 6 }));
  const reloaded = await cache.load("first", async () => {
    loads += 1;
    return { bytes: 6, reload: true };
  });
  assert.equal(first.bytes, 6);
  assert.equal(loads, 1);
  assert.equal(reloaded.reload, true);
  assert.ok(cache.sizeBytes <= cache.maxBytes);
});

test("fallback map source does not reset an unchanged footprint collection", () => {
  let updates = 0;
  const source = { setData() { updates += 1; } };
  const view = new MapView({});
  view.map = { getSource() { return source; } };
  const footprints = { type: "FeatureCollection", features: [] };
  view.setSource("incident-footprints-fallback", footprints, "incident_id");
  view.setSource("incident-footprints-fallback", footprints, "incident_id");
  assert.equal(updates, 1);
});

test("switching incidents clears the previous onion-skin history immediately", () => {
  const view = new MapView({});
  view.selectedIncidentId = "incident-a";
  view.selectedIncidentHistory = {
    type: "FeatureCollection", features: [{ properties: { incident_id: "incident-a" } }],
  };
  view.selectedIncidentCurrentFootprint = { properties: { incident_id: "incident-a" } };
  view.setSelectedIncident("incident-b");
  assert.equal(view.selectedIncidentHistory.features.length, 0);
  assert.equal(view.selectedIncidentCurrentFootprint, null);
});

test("Shift-clicking a field keeps the underlying incident selectable at field zoom", () => {
  const previousDeck = globalThis.deck;
  globalThis.deck = { GeoJsonLayer: class { constructor(properties) { Object.assign(this, properties); } } };
  try {
    const selectedFields = [];
    const selectedIncidents = [];
    const incident = {
      type: "Feature",
      properties: { incident_id: "incident-a", crop_name: "maize" },
    };
    const view = new MapView({
      incidentMode: true,
      onSelect(properties) { selectedFields.push(properties.field_id); },
      onSelectIncident(properties) { selectedIncidents.push(properties.incident_id); },
    });
    view.footprints = { type: "FeatureCollection", features: [incident] };
    view.incidentDeckHitFeatures = () => [incident];
    const layer = view.fieldDeckLayer("incident-field-drilldown");
    layer.onClick({
      object: { properties: { field_id: "field-a" } },
      srcEvent: { shiftKey: true },
    });
    assert.deepEqual(selectedIncidents, ["incident-a"]);
    assert.deepEqual(selectedFields, []);
    layer.onClick({
      object: { properties: { field_id: "field-a" } },
      srcEvent: { shiftKey: false },
    });
    assert.deepEqual(selectedFields, ["field-a"]);
  } finally {
    globalThis.deck = previousDeck;
  }
});

test("incident story arc is causal and shows stage, pressure, impact, unresolved, and area", () => {
  const arc = incidentStoryArc([
    {
      timeline_bucket: "2026-01-05", incident_state: "ACTIVE",
      pressure_core_field_count: 4, impact_lag_field_count: 1,
      unresolved_carried_field_count: 2, footprint_area_km2: 3.25,
    },
    {
      timeline_bucket: "2026-01-12", incident_state: "RECOVERING",
      pressure_core_field_count: 0, impact_lag_field_count: 3,
      unresolved_carried_field_count: 1, footprint_area_km2: 2.5,
    },
    { timeline_bucket: "2026-01-19", incident_state: "CLOSED_RECOVERED" },
  ], [
    {
      timeline_bucket: "2026-01-05", stage_bucket: "maturity_or_harvest",
      affected_crop_instance_count: 0,
    },
    {
      timeline_bucket: "2026-01-05", stage_bucket: "vegetative",
      affected_crop_instance_count: 3,
    },
    {
      timeline_bucket: "2026-01-12", stage_bucket: "flowering",
      affected_crop_instance_count: 2,
    },
  ], "2026-01-12");
  assert.equal(arc.length, 2);
  assert.equal(arc[0].stage, "vegetative");
  assert.equal(arc[0].pressure, 4);
  assert.equal(arc[0].areaKm2, 3.25);
  assert.equal(arc[1].stage, "flowering");
  assert.equal(arc[1].impact, 3);
  assert.equal(arc[1].unresolved, 1);
  assert.equal(arc[1].selected, true);
});

test("selected incident history uses prior exact polygons only and never implies movement", () => {
  const polygon = (offset) => ({
    type: "Polygon",
    coordinates: [[[offset, 0], [offset + 1, 0], [offset, 1], [offset, 0]]],
  });
  const history = incidentFootprintHistory([
    { timeline_bucket: "2026-01-05", incident_id: "i-1", geometry: polygon(0) },
    { timeline_bucket: "2026-01-12", incident_id: "i-1", geometry: polygon(1) },
    { timeline_bucket: "2026-01-19", incident_id: "i-1", geometry: polygon(2) },
    { timeline_bucket: "2026-01-05", incident_id: "other", geometry: polygon(3) },
  ], "2026-01-12", "i-1");
  assert.deepEqual(history.current.geometry, polygon(1));
  assert.equal(history.current.properties.timeline_bucket, "2026-01-12");
  assert.equal(history.prior.properties.timeline_bucket, "2026-01-05");
  assert.equal(history.collection.features.length, 1);
  assert.equal(history.collection.features[0].properties.age_band, "recent");
  assert.equal(history.collection.meta.is_physical_movement, false);
  assert.ok(footprintHistoryVisualModel(
    history.collection.features[0].properties,
  ).lineAlpha < 255);
  const recent = footprintHistoryVisualModel({ age_band: "recent" });
  const middle = footprintHistoryVisualModel({ age_band: "middle" });
  const old = footprintHistoryVisualModel({ age_band: "old" });
  assert.ok(recent.fillAlpha > middle.fillAlpha);
  assert.ok(middle.fillAlpha > old.fillAlpha);
  assert.ok(old.fillAlpha > 0);
});

test("co-located crop incidents remain discoverable in deterministic cycle order", () => {
  const collection = {
    features: [
      { properties: { timeline_bucket: "2026-01-12", coincident_group_id: "same-shape", incident_id: "i-beans", crop_name: "beans", coincident_incident_index: 0 } },
      { properties: { timeline_bucket: "2026-01-12", coincident_group_id: "same-shape", incident_id: "i-maize", crop_name: "maize", coincident_incident_index: 1 } },
      { properties: { timeline_bucket: "2026-01-19", coincident_group_id: "same-shape", incident_id: "later", crop_name: "maize" } },
    ],
  };
  assert.deepEqual(
    coincidentIncidentCandidates(collection, collection.features[1].properties)
      .map((feature) => feature.properties.incident_id),
    ["i-beans", "i-maize"],
  );
});

test("nested non-identical hit footprints are all reachable across repeated clicks", () => {
  const polygon = (size) => ({
    type: "Polygon",
    coordinates: [[[0, 0], [size, 0], [size, size], [0, size], [0, 0]]],
  });
  const feature = (incidentId, groupId, index, size) => ({
    type: "Feature",
    geometry: polygon(size),
    properties: {
      timeline_bucket: "2026-01-12",
      incident_id: incidentId,
      crop_name: incidentId,
      coincident_group_id: groupId,
      coincident_incident_index: index,
    },
  });
  const outer = feature("outer", "outer-shape", 0, 10);
  const nested = feature("nested", "nested-shape", 0, 4);
  const exactA = feature("exact-a", "exact-shape", 0, 2);
  const exactB = feature("exact-b", "exact-shape", 1, 2);
  const collection = { features: [outer, nested, exactA, exactB] };
  const hits = [outer, nested, exactA, outer];
  const candidates = incidentHitCandidates(collection, hits, outer.properties);
  const reversed = incidentHitCandidates(
    collection,
    [...hits].reverse(),
    outer.properties,
  );
  const orderedIds = candidates.map((item) => item.properties.incident_id);
  assert.deepEqual(
    reversed.map((item) => item.properties.incident_id),
    orderedIds,
  );
  assert.equal(new Set(orderedIds).size, 4);
  assert.equal(
    Math.abs(orderedIds.indexOf("exact-a") - orderedIds.indexOf("exact-b")),
    1,
  );

  let selected = "";
  const reached = [];
  for (let click = 0; click < candidates.length; click += 1) {
    const next = nextIncidentCandidate(candidates, selected, outer.properties);
    selected = next.properties.incident_id;
    reached.push(selected);
  }
  assert.deepEqual(new Set(reached), new Set(["outer", "nested", "exact-a", "exact-b"]));

  const selectedByMap = [];
  const view = new MapView({
    onSelectIncident(properties) {
      selectedByMap.push(properties.incident_id);
      view.selectedIncidentId = properties.incident_id;
    },
  });
  view.footprints = collection;
  for (let click = 0; click < candidates.length; click += 1) {
    view.selectCoincidentIncident(outer.properties, hits);
  }
  assert.deepEqual(
    new Set(selectedByMap),
    new Set(["outer", "nested", "exact-a", "exact-b"]),
  );
});

test("footprint prefetch coalesces without retaining payloads in the generic API cache", async () => {
  const originalFetch = globalThis.fetch;
  let release;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    await new Promise((resolve) => { release = resolve; });
    return {
      ok: true,
      async json() { return { complete: true }; },
    };
  };
  try {
    const client = new ApiClient();
    const collections = new FootprintCollectionCache();
    const url = "/api/incident-footprints/2026-01-12";
    const prefetched = loadFootprintCollection(client, collections, url, {
      channel: "incident-footprints:prefetch:2026-01-12",
    });
    await Promise.resolve();
    const current = loadFootprintCollection(client, collections, url, {
      channel: "incident-footprints:current",
    });
    release();
    assert.deepEqual(await prefetched, { complete: true });
    assert.deepEqual(await current, { complete: true });
    assert.equal(calls, 1);
    assert.deepEqual(
      [...client.cache.keys()].filter((key) => key.includes("/api/incident-footprints/")),
      [],
    );
    assert.equal(collections.values.has(url), true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("generic API cache evicts decoded payloads by bytes as well as count", () => {
  const client = new ApiClient({ cacheLimit: 10, cacheByteLimit: 80 });
  client.remember("/first", { value: "a".repeat(35) });
  client.remember("/second", { value: "b".repeat(35) });
  assert.equal(client.cache.has("/first"), false);
  assert.equal(client.cache.has("/second"), true);
  assert.ok(client.cacheBytesUsed <= 80);
  client.remember("/oversized", { value: "x".repeat(200) });
  assert.equal(client.cache.has("/oversized"), false);
  client.forget("/second");
  assert.equal(client.cacheBytesUsed, 0);
});

test("A to B to A scrubbing never reuses an already-aborted request", async () => {
  const originalFetch = globalThis.fetch;
  const pending = [];
  globalThis.fetch = (url, { signal }) => new Promise((resolve, reject) => {
    const request = { url, resolve };
    pending.push(request);
    signal.addEventListener("abort", () => {
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    }, { once: true });
  });
  try {
    const client = new ApiClient();
    const firstA = client.get("/A", { channel: "frame" }).catch((error) => error);
    await Promise.resolve();
    const requestB = client.get("/B", { channel: "frame" }).catch((error) => error);
    await Promise.resolve();
    const secondA = client.get("/A", { channel: "frame" });
    await Promise.resolve();

    assert.deepEqual(pending.map((request) => request.url), ["/A", "/B", "/A"]);
    pending[2].resolve({ ok: true, async json() { return { bucket: "A-new" }; } });
    assert.deepEqual(await secondA, { bucket: "A-new" });
    assert.equal((await firstA).name, "AbortError");
    assert.equal((await requestB).name, "AbortError");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("incident detail model exposes stage, lifecycle, counts, lineage, and gaps", () => {
  const detail = incidentDetailModel({
    window: { incident_id: "incident-1", relapse_count: 2, data_gap_count: 1 },
    weekly_state: [{
      timeline_bucket: "2026-01-12", crop_name: "maize", incident_state: "RECOVERING",
      stage_distribution: '{"flowering":0.75,"vegetative":0.25}',
      monitored_count: 20, evaluable_count: 18, affected_count: 7, severe_count: 2,
      fresh_decline_field_count: 1, fresh_recovery_field_count: 4,
      coverage_adequate: false, global_crop_week_unmappable_instance_count: 5,
    }],
    footprints: [
      {
        timeline_bucket: "2026-01-05", incident_id: "incident-1",
        geometry: { type: "Polygon", coordinates: [] },
      },
      {
        timeline_bucket: "2026-01-12", incident_id: "incident-1",
        geometry: { type: "Polygon", coordinates: [] },
      },
    ],
    lineage: {
      incoming: [{ lineage_type: "split", lineage_id: "split-1" }],
      outgoing: [{ lineage_type: "merge", lineage_id: "merge-1" }],
    },
  }, { timeline_bucket: "2026-01-12" });
  assert.equal(detail.crop, "maize");
  assert.equal(detail.lifecycle, "RECOVERING");
  assert.equal(detail.dominantStage, "flowering");
  assert.equal(detail.counts.monitored, 20);
  assert.equal(detail.counts.freshRecovery, 4);
  assert.equal(detail.counts.split, 1);
  assert.equal(detail.counts.merge, 1);
  assert.equal(detail.counts.relapse, 2);
  assert.equal(detail.evidence.coverageAdequate, false);
  assert.equal(detail.evidence.dataGapCount, 1);
  assert.equal(detail.evidence.globalCropWeekUnmappableInstanceCount, 5);
  assert.equal(detail.currentFootprint.properties.timeline_bucket, "2026-01-12");
  assert.equal(detail.priorFootprint.properties.timeline_bucket, "2026-01-05");
  assert.equal(detail.footprintHistory.features.length, 1);
});

test("historical incident detail never borrows lifecycle dates from the future", () => {
  const detail = incidentDetailModel({
    window: {
      incident_id: "incident-1",
      first_evidence_week: "2026-01-05",
      confirmed_week: "2026-01-12",
      pressure_off_week: "2026-02-02",
      recovered_week: "2026-02-16",
      closed_week: "2026-02-23",
    },
    weekly_state: [{
      timeline_bucket: "2026-01-12",
      first_evidence_week: "2026-01-05",
      confirmed_week: "2026-01-12",
      pressure_off_week: null,
      recovered_week: null,
      closed_week: null,
    }],
  }, { timeline_bucket: "2026-01-12" });
  assert.equal(detail.lifecycleDates.confirmed, "2026-01-12");
  assert.equal(detail.lifecycleDates.pressureOff, undefined);
  assert.equal(detail.lifecycleDates.recovered, undefined);
  assert.equal(detail.lifecycleDates.closed, undefined);
});

test("continuous selection uses the last causal state when the selected week has no row", () => {
  const detail = incidentDetailModel({
    window: {
      incident_id: "incident-1",
      terminal_state: "CLOSED_RECOVERED",
      closed_week: "2026-02-23",
    },
    weekly_state: [{
      timeline_bucket: "2026-01-12",
      incident_id: "incident-1",
      incident_state: "ACTIVE",
      closed_week: null,
    }],
  }, { incident_id: "incident-1", timeline_bucket: "2026-01-19" });
  assert.equal(detail.observedThisWeek, false);
  assert.equal(detail.lastObservedBucket, "2026-01-12");
  assert.equal(detail.lifecycle, "ACTIVE");
  assert.equal(detail.lifecycleDates.closed, undefined);
});
