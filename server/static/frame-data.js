import { HttpError, isUnsupportedError, withQuery } from "./api.js";

export const GEOMETRY_BATCH_SIZE = 2000;
export const MAX_GEOMETRY_CACHE = 25000;
const GEOMETRY_REQUEST_CONCURRENCY = 4;

export class FrameDataLoader {
  constructor(api) {
    this.api = api;
    this.compactSupported = null;
    this.geometryVersion = "";
    this.geometry = new Map();
    this.missingGeometry = new Set();
  }

  async load({ bucket, query = {}, legacyUrl }) {
    this.api.abortPrefix("frame-data:");
    if (this.compactSupported !== false) {
      try {
        const frame = await this.loadCompact(bucket, query);
        this.compactSupported = true;
        return frame;
      } catch (error) {
        if (!isUnsupportedError(error)) throw error;
        this.compactSupported = false;
        this.resetGeometry();
      }
    }
    return this.api.get(legacyUrl, { channel: "frame-data:legacy", cache: true });
  }

  preload({ bucket, query = {}, legacyUrl }) {
    if (this.compactSupported === false) {
      this.api.preload(legacyUrl);
      return;
    }
    this.api.preload(withQuery(`/api/frame-state/${encodeURIComponent(bucket)}`, query));
  }

  async loadCompact(bucket, query) {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const stateUrl = withQuery(
        `/api/frame-state/${encodeURIComponent(bucket)}`,
        attempt ? { ...query, geometry_retry: Date.now() } : query
      );
      const state = await this.api.get(stateUrl, {
        channel: "frame-data:state",
        cache: attempt === 0,
      });
      try {
        return await this.hydrate(state);
      } catch (error) {
        if (Number(error?.status) !== 409 || attempt > 0) throw error;
        this.api.forget?.(stateUrl);
        this.resetGeometry();
      }
    }
    throw new Error("Could not resolve a consistent geometry version.");
  }

  async hydrate(state) {
    const rows = Array.isArray(state?.rows) ? state.rows : [];
    const geometryVersion = String(state?.geometry_version || "");
    if (!geometryVersion) throw new Error("Compact frame response has no geometry_version.");
    if (this.geometryVersion && this.geometryVersion !== geometryVersion) this.resetGeometry();
    this.geometryVersion = geometryVersion;

    const fieldIds = unique(rows.map((row) => String(row?.field_id || "")).filter(Boolean));
    const missing = fieldIds.filter((fieldId) => !this.geometry.has(fieldId) && !this.missingGeometry.has(fieldId));
    const batches = chunk(missing, GEOMETRY_BATCH_SIZE);
    for (let start = 0; start < batches.length; start += GEOMETRY_REQUEST_CONCURRENCY) {
      const group = batches.slice(start, start + GEOMETRY_REQUEST_CONCURRENCY);
      const payloads = await Promise.all(group.map((field_ids, index) => this.api.post(
        "/api/geometry",
        { geometry_version: geometryVersion, field_ids },
        { channel: `frame-data:geometry:${start + index}` }
      )));
      for (const payload of payloads) this.rememberGeometry(payload, geometryVersion);
    }
    const frame = mergeFrameState(state, this.geometry, this.missingGeometry);
    this.trimGeometry(new Set(fieldIds));
    return frame;
  }

  rememberGeometry(payload, expectedVersion) {
    const actualVersion = String(payload?.geometry_version || "");
    if (actualVersion !== expectedVersion || this.geometryVersion !== expectedVersion) {
      throw new HttpError(409, "Geometry version changed", "Frame state and geometry do not match.");
    }
    for (const feature of payload?.features || []) {
      const fieldId = String(feature?.properties?.field_id || "");
      if (fieldId && feature?.geometry) {
        this.geometry.delete(fieldId);
        this.geometry.set(fieldId, feature);
      }
    }
    for (const fieldId of payload?.meta?.missing_field_ids || []) {
      this.missingGeometry.add(String(fieldId));
    }
  }

  trimGeometry(protectedIds) {
    for (const fieldId of this.geometry.keys()) {
      if (this.geometry.size <= MAX_GEOMETRY_CACHE) break;
      if (!protectedIds.has(fieldId)) this.geometry.delete(fieldId);
    }
    while (this.geometry.size > MAX_GEOMETRY_CACHE) this.geometry.delete(this.geometry.keys().next().value);
  }

  resetGeometry() {
    this.geometryVersion = "";
    this.geometry.clear();
    this.missingGeometry.clear();
  }
}

export function mergeFrameState(state, geometry, missingGeometry = new Set()) {
  const rows = Array.isArray(state?.rows) ? state.rows : [];
  const features = [];
  for (const row of rows) {
    const fieldId = String(row?.field_id || "");
    const staticFeature = geometry.get(fieldId);
    if (!staticFeature?.geometry) continue;
    features.push({
      type: "Feature",
      geometry: staticFeature.geometry,
      properties: { ...(staticFeature.properties || {}), ...row },
    });
  }
  const meta = {
    ...(state?.meta || {}),
    feature_count: features.length,
    state_count: Number(state?.meta?.state_count ?? rows.length),
    geometry_missing_count: rows.filter((row) => missingGeometry.has(String(row?.field_id || ""))).length,
    geometry_version: state?.geometry_version,
    transport: "compact-state-plus-geometry",
  };
  return { type: "FeatureCollection", features, meta };
}

function unique(values) {
  return [...new Set(values)];
}

function chunk(values, size) {
  const result = [];
  for (let index = 0; index < values.length; index += size) result.push(values.slice(index, index + size));
  return result;
}
