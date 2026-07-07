import { incidentFootprintHistory, incidentStoryArc } from "./incident-story.js";

export const CROP_INCIDENT_MODE = "crop_incident_v3";
export const CROP_INCIDENT_V4_MODE = "crop_incident_v4_dual_clock";
export const INCIDENT_FIELD_MIN_ZOOM = 11;
export const INCIDENT_TRUTH_LABEL = "Footprint evolution, not physical movement.";

export function isCropIncidentV3(manifest = {}) {
  return String(manifest.mode || manifest.run?.mode || "") === CROP_INCIDENT_MODE;
}

export function isCropIncidentV4(manifest = {}) {
  return String(manifest.mode || manifest.run?.mode || "") === CROP_INCIDENT_V4_MODE;
}

export function isCropIncident(manifest = {}) {
  return isCropIncidentV3(manifest) || isCropIncidentV4(manifest);
}

export function shouldLoadEvolution(manifest = {}) {
  return !isCropIncident(manifest);
}

export function v4LayerModel(zoom = 0) {
  return {
    primaryLayer: "daily-pressure-crop-story",
    fieldOverview: {
      visible: Number(zoom) < INCIDENT_FIELD_MIN_ZOOM,
      completeness: "api-audited",
    },
    pressure: { visible: true, clock: "daily" },
    cropImpact: { visible: true, clock: "s2-step-held" },
    story: { visible: true, clock: "weekly-knowledge-gated" },
    fields: { visible: Number(zoom) >= INCIDENT_FIELD_MIN_ZOOM, role: "drilldown-detail" },
    evolution: { visible: false },
  };
}

export function v3LayerModel(zoom = 0) {
  return {
    primaryLayer: "exact-complete-incident-footprints",
    footprints: { visible: true, exact: true, complete: true },
    fields: { visible: Number(zoom) >= INCIDENT_FIELD_MIN_ZOOM, role: "drilldown-detail" },
    evolution: { visible: false },
  };
}

export function fieldViewportCoverage(collection = {}) {
  const meta = collection.meta || {};
  const shown = Math.max(0, Number(
    meta.feature_count ?? collection.features?.length ?? meta.state_count ?? 0,
  ));
  const source = Math.max(shown, Number(
    meta.source_row_count ?? meta.source_field_count ?? shown,
  ));
  const truncated = truthy(meta.truncated) || truthy(meta.limit_hit) || shown < source;
  return {
    shown,
    source,
    truncated,
    label: truncated
      ? `showing ${shown.toLocaleString()} of ${source.toLocaleString()} viewport fields · capped`
      : `${shown.toLocaleString()} exact viewport fields`,
  };
}

export function normalizeFootprintCollection(payload = {}) {
  const source = Array.isArray(payload.features)
    ? payload.features
    : Array.isArray(payload.footprints)
      ? payload.footprints
      : Array.isArray(payload.rows) ? payload.rows : [];
  const features = source.map((item) => {
    if (item?.type === "Feature") return item;
    const geometry = parseGeometry(item?.geometry || item?.geometry_geojson);
    const properties = { ...(item || {}) };
    delete properties.geometry;
    delete properties.geometry_geojson;
    return { type: "Feature", geometry, properties };
  });
  for (const feature of features) {
    if (!feature?.geometry || !["Polygon", "MultiPolygon"].includes(feature.geometry.type)) {
      throw new Error("Incident footprint response contains a non-polygon geometry.");
    }
  }
  return {
    type: "FeatureCollection",
    features,
    meta: payload.meta || {},
  };
}

export function assertCompleteFootprintCollection(collection = {}) {
  const meta = collection.meta || {};
  const expected = Number(meta.feature_count);
  const actual = Array.isArray(collection.features) ? collection.features.length : 0;
  const incomplete = meta.complete !== true
    || truthy(meta.truncated)
    || truthy(meta.feature_cap_applied)
    || truthy(meta.low_zoom_footprints_dropped)
    || meta.footprint_geometry_method !== "exact_union_of_grid_rectangles"
    || (Number.isFinite(expected) && expected !== actual);
  if (incomplete) {
    throw new Error("Incident footprint response is incomplete; refusing to present it as the overview.");
  }
  return collection;
}

export function footprintVisualModel(properties = {}) {
  const state = String(properties.incident_state || properties.lifecycle_state || "").toUpperCase();
  const carried = truthy(properties.footprint_carried_forward);
  const pressureCount = numeric(first(
    properties.pressure_core_field_count,
    properties.pressure_core_count,
    properties.pressure_cell_count,
  ), 0);
  const impactCount = numeric(first(
    properties.unresolved_carried_field_count,
    properties.impact_lag_field_count,
    properties.impact_lag_count,
    properties.impact_cell_count,
  ), 0);
  const watchCount = numeric(first(
    properties.watch_frontier_field_count,
    properties.watch_frontier_count,
    properties.watch_cell_count,
  ), 0);
  const gap = truthy(properties.is_data_gap)
    || (properties.coverage_adequate !== undefined && !truthy(properties.coverage_adequate))
    || ["DATA_GAP", "INSUFFICIENT_EVIDENCE", "CLOSED_DATA_CENSORED"].includes(state);
  const recovering = state.includes("RECOVER");
  const quiet = gap
    || state.includes("QUIET")
    || state.includes("UNRESOLVED")
    || state === "WATCH"
    || state === "CANDIDATE"
    || state === "MERGED_INTO"
    || state.startsWith("CLOSED_");
  if (carried) {
    return { key: "carried", label: "Carried-forward extent", fillAlpha: 42, lineAlpha: 185, lineWidth: 1.5, dash: [2, 3] };
  }
  if (recovering) {
    return { key: "recovering", label: "Recovering evidence", fillAlpha: 78, lineAlpha: 225, lineWidth: 1.8, dash: [7, 3] };
  }
  if (pressureCount <= 0 && impactCount > 0) {
    return { key: "impact", label: "Crop-impact evidence without current pressure", fillAlpha: 92, lineAlpha: 230, lineWidth: 1.9, dash: [5, 2] };
  }
  if (pressureCount <= 0 && watchCount > 0) {
    return { key: "watch", label: "Watch-frontier evidence", fillAlpha: 58, lineAlpha: 205, lineWidth: 1.5, dash: [3, 3] };
  }
  if (quiet) {
    const label = gap ? "Evidence gap / not evaluable"
      : state.includes("UNRESOLVED") ? "Unresolved response evidence"
        : state.startsWith("CLOSED_") ? "Closed incident footprint"
          : "Quiet / watch";
    return { key: "quiet", label, fillAlpha: 56, lineAlpha: 180, lineWidth: 1.4, dash: [1, 3] };
  }
  return { key: "pressure", label: "Current pressure evidence", fillAlpha: 156, lineAlpha: 245, lineWidth: 2.1, dash: [1000, 1] };
}

export function footprintLegendEntries() {
  return [
    { key: "pressure", label: "Current pressure evidence" },
    { key: "impact", label: "Impact evidence, no current pressure" },
    { key: "watch", label: "Watch-frontier evidence" },
    { key: "recovering", label: "Recovering evidence" },
    { key: "quiet", label: "Quiet / evidence gap" },
    { key: "carried", label: "Carried-forward extent" },
  ];
}

export function footprintRoleCollection(collection = {}, role = "pressure") {
  const key = `${role}_geometry`;
  return {
    type: "FeatureCollection",
    features: (collection.features || []).flatMap((feature) => {
      const geometry = parseGeometry(feature.properties?.[key]);
      if (!geometry || !["Polygon", "MultiPolygon"].includes(geometry.type)) return [];
      return [{ ...feature, geometry }];
    }),
    meta: collection.meta || {},
  };
}

export function adjacentBuckets(buckets = [], index = 0) {
  return [buckets[index - 1], buckets[index + 1]].filter(Boolean);
}

export function coincidentIncidentCandidates(collection = {}, properties = {}) {
  const groupId = String(properties.coincident_group_id || "");
  const bucket = String(properties.timeline_bucket || "").slice(0, 10);
  if (!groupId || !bucket) return [];
  return (collection.features || []).filter((feature) => {
    const candidate = feature.properties || {};
    return String(candidate.coincident_group_id || "") === groupId
      && String(candidate.timeline_bucket || "").slice(0, 10) === bucket;
  }).sort((left, right) => (
    numeric(left.properties?.coincident_incident_index, 0)
      - numeric(right.properties?.coincident_incident_index, 0)
    || String(left.properties?.crop_name || "").localeCompare(
      String(right.properties?.crop_name || ""),
    )
    || String(left.properties?.incident_id || "").localeCompare(
      String(right.properties?.incident_id || ""),
    )
  ));
}

export function incidentHitCandidates(
  collection = {},
  hitFeatures = [],
  anchorProperties = {},
) {
  const hits = Array.isArray(hitFeatures) ? hitFeatures.slice() : [];
  if (anchorProperties?.incident_id) hits.push({ properties: anchorProperties });
  const expanded = [];
  for (const hit of hits) {
    const properties = hit?.properties || hit || {};
    const coincident = coincidentIncidentCandidates(collection, properties);
    if (coincident.length) {
      expanded.push(...coincident);
      continue;
    }
    const incidentId = String(properties.incident_id || "");
    const bucket = String(properties.timeline_bucket || "").slice(0, 10);
    const source = (collection.features || []).find((feature) => (
      String(feature.properties?.incident_id || "") === incidentId
      && (!bucket || String(feature.properties?.timeline_bucket || "").slice(0, 10) === bucket)
    ));
    expanded.push(source || (hit?.properties ? hit : { properties }));
  }
  const unique = new Map();
  for (const feature of expanded) {
    const incidentId = String(feature?.properties?.incident_id || "");
    if (incidentId && !unique.has(incidentId)) unique.set(incidentId, feature);
  }
  return [...unique.values()].sort(incidentCandidateOrder);
}

export function nextIncidentCandidate(
  candidates = [],
  selectedIncidentId = "",
  anchorProperties = {},
) {
  if (!candidates.length) return null;
  const selected = candidates.findIndex(
    (feature) => String(feature.properties?.incident_id || "")
      === String(selectedIncidentId || ""),
  );
  if (selected >= 0) return candidates[(selected + 1) % candidates.length];
  const anchorId = String(anchorProperties?.incident_id || "");
  return candidates.find(
    (feature) => String(feature.properties?.incident_id || "") === anchorId,
  ) || candidates[0];
}

function incidentCandidateOrder(left, right) {
  const a = left.properties || {};
  const b = right.properties || {};
  const aId = String(a.incident_id || "");
  const bId = String(b.incident_id || "");
  const aGroup = String(a.coincident_group_id || `incident:${aId}`);
  const bGroup = String(b.coincident_group_id || `incident:${bId}`);
  return String(a.timeline_bucket || "").localeCompare(String(b.timeline_bucket || ""))
    || aGroup.localeCompare(bGroup)
    || numeric(a.coincident_incident_index, 0) - numeric(b.coincident_incident_index, 0)
    || String(a.crop_name || "").localeCompare(String(b.crop_name || ""))
    || aId.localeCompare(bId);
}

export function incidentDetailModel(payload = {}, footprint = {}) {
  const window = payload.window || payload.incident || {};
  const weeklyRows = rows(payload.weekly_state || payload.weekly_states);
  const stageRows = rows(payload.stage_summary || payload.stage_summaries);
  const selectedBucket = String(footprint.timeline_bucket || payload.timeline_bucket || "").slice(0, 10);
  const exactWeekly = selectedRow(weeklyRows, selectedBucket);
  const hasSelectedWeek = Object.keys(exactWeekly).length > 0;
  const weekly = hasSelectedWeek
    ? exactWeekly
    : latestCausalRow(weeklyRows, selectedBucket);
  const allowWindowFallback = !selectedBucket;
  const selectedStages = stageRows.filter(
    (row) => String(row.timeline_bucket || "").slice(0, 10) === selectedBucket,
  );
  const current = { ...footprint, ...weekly };
  const stageFallback = distributionFromStageRows(selectedStages);
  const distribution = stageDistribution(
    current.stage_distribution || weekly.stage_distribution,
    current.stage_bucket || current.dominant_stage,
    stageFallback,
  );
  const lineage = payload.lineage || {};
  const lineageRows = [
    ...rows(lineage.incoming),
    ...rows(lineage.outgoing),
    ...(Array.isArray(lineage) ? lineage : []),
  ];
  const incidentId = first(current.incident_id, window.incident_id, payload.incident_id);
  const storyArc = incidentStoryArc(weeklyRows, stageRows, selectedBucket);
  const footprintHistory = incidentFootprintHistory(
    payload.footprints,
    selectedBucket,
    incidentId,
  );
  const splitCount = countValue(current, window, lineage, "split_count")
    ?? lineageRows.filter((row) => row.lineage_type === "split").length;
  const mergeCount = countValue(current, window, lineage, "merge_count")
    ?? lineageRows.filter((row) => row.lineage_type === "merge").length;
  const dataGapCount = numeric(first(current.data_gap_count, window.data_gap_count), 0);
  const coverageAdequate = current.coverage_adequate === undefined
    ? !truthy(current.is_data_gap)
    : truthy(current.coverage_adequate) && !truthy(current.is_data_gap);
  return {
    incidentId,
    crop: first(current.crop_name, current.crop_name_normalized, window.crop_name, "Unknown crop"),
    hazard: first(current.hazard_family, window.hazard_family, "Unknown exposure"),
    lifecycle: first(
      current.incident_state,
      current.lifecycle_state,
      allowWindowFallback ? window.terminal_state : null,
      hasSelectedWeek ? "Unknown" : "Not observed this week",
    ),
    lifecycleDates: {
      firstEvidence: first(current.first_evidence_week, allowWindowFallback ? window.first_evidence_week : null),
      confirmed: first(current.confirmed_week, allowWindowFallback ? window.confirmed_week : null),
      pressureOff: first(current.pressure_off_week, allowWindowFallback ? window.pressure_off_week : null),
      recovered: first(current.recovered_week, allowWindowFallback ? window.recovered_week : null),
      closed: first(current.closed_week, allowWindowFallback ? window.closed_week : null),
    },
    timelineBucket: first(selectedBucket, current.timeline_bucket),
    observedThisWeek: hasSelectedWeek,
    lastObservedBucket: weekly.timeline_bucket,
    dominantStage: first(current.dominant_stage, current.stage_bucket, dominantStage(distribution), "Unknown"),
    stageDistribution: distribution,
    counts: {
      monitored: numeric(first(current.monitored_count, current.monitored_field_count, current.monitored_crop_instance_count)),
      evaluable: numeric(first(current.evaluable_count, current.evaluable_field_count, current.evaluable_crop_instance_count)),
      affected: numeric(first(current.affected_count, current.affected_field_count, current.pressure_core_field_count)),
      severe: numeric(first(current.severe_count, current.severe_field_count, current.severe_crop_instance_count)),
      freshDecline: numeric(first(current.fresh_decline_field_count, current.fresh_decline_count)),
      freshRecovery: numeric(first(current.fresh_recovery_field_count, current.fresh_recovery_count)),
      unresolved: numeric(first(current.unresolved_carried_field_count, current.unresolved_count)),
      recovered: numeric(first(current.recovered_field_count, current.recovered_count)),
      split: numeric(splitCount, 0),
      merge: numeric(mergeCount, 0),
      relapse: numeric(first(current.relapse_count, window.relapse_count), 0),
    },
    evidence: {
      coverageAdequate,
      dataGapCount,
      coverageMissingCellCount: numeric(current.coverage_missing_cell_count, 0),
      globalCropWeekUnmappableInstanceCount: numeric(first(
        current.global_crop_week_unmappable_instance_count,
        window.global_crop_week_unmappable_instance_count,
        current.missing_geometry_count,
        window.missing_geometry_count,
      ), 0),
      carriedForward: truthy(current.footprint_carried_forward),
      rightCensored: truthy(first(current.right_censored, window.right_censored)),
    },
    coincident: {
      count: numeric(current.coincident_incident_count, 1),
      crops: stringList(current.coincident_crop_names_json),
    },
    window,
    weeklyRows,
    storyArc,
    footprintHistory: footprintHistory.collection,
    currentFootprint: footprintHistory.current,
    priorFootprint: footprintHistory.prior,
    lineage,
  };
}

function selectedRow(values, bucket) {
  if (bucket) {
    return values.find(
      (row) => String(row.timeline_bucket || "").slice(0, 10) === bucket,
    ) || {};
  }
  return values.at(-1) || {};
}

function latestCausalRow(values, bucket) {
  if (!bucket) return values.at(-1) || {};
  return values
    .filter((row) => String(row.timeline_bucket || "").slice(0, 10) <= bucket)
    .sort((left, right) => String(left.timeline_bucket || "").localeCompare(
      String(right.timeline_bucket || ""),
    ))
    .at(-1) || {};
}

function rows(value) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.rows)) return value.rows;
  if (Array.isArray(value?.items)) return value.items;
  return value && typeof value === "object" && !Array.isArray(value) ? [value] : [];
}

function stageDistribution(value, fallback, fallbackDistribution = []) {
  let parsed = value;
  if (typeof value === "string") {
    try { parsed = JSON.parse(value); } catch { parsed = {}; }
  }
  const entries = Object.entries(parsed || {})
    .map(([stage, share]) => ({ stage, share: Math.max(0, Number(share) || 0) }))
    .filter((item) => item.stage && item.share > 0)
    .sort((left, right) => right.share - left.share || left.stage.localeCompare(right.stage));
  if (!entries.length) {
    if (fallbackDistribution.length) return fallbackDistribution;
    return fallback ? [{ stage: String(fallback), share: 1 }] : [];
  }
  const total = entries.reduce((sum, item) => sum + item.share, 0);
  if (total > 0 && Math.abs(total - 1) > 0.001) {
    for (const item of entries) item.share /= total;
  }
  return entries;
}

function distributionFromStageRows(stageRows) {
  const values = {};
  for (const row of stageRows) {
    if (
      row.affected_crop_instance_count !== undefined
      && numeric(row.affected_crop_instance_count, 0) <= 0
    ) continue;
    const stage = String(row.stage_bucket || row.crop_stage || "");
    if (!stage) continue;
    values[stage] = numeric(first(
      row.stage_share,
      row.share,
      row.affected_crop_instance_count,
      row.evaluable_crop_instance_count,
      row.monitored_crop_instance_count,
      row.field_count,
      row.count,
      1,
    ), 1);
  }
  return stageDistribution(values, null);
}

function dominantStage(distribution) {
  return distribution[0]?.stage;
}

function countValue(...args) {
  const key = args.pop();
  const value = args.map((item) => item?.[key]).find((item) => item !== undefined && item !== null);
  return value === undefined ? null : numeric(value, 0);
}

function parseGeometry(value) {
  if (value && typeof value === "object") return value;
  if (typeof value === "string") {
    try { return JSON.parse(value); } catch { return null; }
  }
  return null;
}

function first(...values) {
  return values.find((value) => value !== undefined && value !== null && String(value) !== "");
}

function numeric(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function truthy(value) {
  return value === true || value === 1 || ["true", "1", "yes"].includes(String(value).toLowerCase());
}

function stringList(value) {
  if (Array.isArray(value)) return value.map(String);
  if (typeof value !== "string" || !value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}
