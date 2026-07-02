import assert from "node:assert/strict";
import test from "node:test";

import { FrameDataLoader, GEOMETRY_BATCH_SIZE, mergeFrameState } from "./static/frame-data.js";
import { computeActivityStats } from "./static/activity.js";
import { EvolutionController } from "./static/evolution-controller.js";
import { shouldLoadHistory } from "./static/history.js";
import { lifecycleModel, prefixTrajectoryModel } from "./static/inspector.js";
import { buildEvolutionModel } from "./static/map-evolution.js";

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
