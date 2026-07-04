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
  footprintRoleCollection,
  footprintVisualModel,
  incidentDetailModel,
  incidentHitCandidates,
  isCropIncidentV3,
  nextIncidentCandidate,
  normalizeFootprintCollection,
  shouldLoadEvolution,
  v3LayerModel,
} from "./static/incident-v3.js";
import {
  footprintHistoryVisualModel,
  incidentFootprintHistory,
  incidentSelectionTransition,
  incidentStoryArc,
} from "./static/incident-story.js";
import { lifecycleModel, prefixTrajectoryModel } from "./static/inspector.js";
import { buildEvolutionModel } from "./static/map-evolution.js";
import { MapView } from "./static/map-view.js";
import { alphaForState, lineColorFor } from "./static/palette.js";
import { activityCount } from "./static/timeline.js";

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
