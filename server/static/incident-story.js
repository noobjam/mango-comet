const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export function incidentSelectionTransition({
  incidentMode = false,
  selectedIncidentId = "",
  selectionBucket = "",
  nextBucket = "",
} = {}) {
  const changed = Boolean(selectionBucket && selectionBucket !== nextBucket);
  const preserveIncident = changed && Boolean(incidentMode && selectedIncidentId);
  return {
    changed,
    preserveIncident,
    clearSelection: changed && !preserveIncident,
    nextSelectionBucket: preserveIncident ? nextBucket : changed ? "" : selectionBucket,
  };
}

export function incidentStoryArc(weeklyValue = [], stageValue = [], selectedBucket = "") {
  const cutoff = validBucket(selectedBucket) ? String(selectedBucket).slice(0, 10) : "";
  const stagesByWeek = groupByBucket(rows(stageValue));
  const eligible = rows(weeklyValue)
    .map((row) => ({ ...row, bucket: bucketOf(row), knownDate: knownDateOf(row) }))
    .filter((row) => validBucket(row.bucket)
      && (!cutoff || (row.knownDate || row.bucket) <= cutoff))
    .sort((left, right) => left.bucket.localeCompare(right.bucket))
    .map((row, index, values) => {
      const stage = dominantAffectedStage(stagesByWeek.get(row.bucket) || []);
      return {
        bucket: row.bucket,
        knownDate: row.knownDate || row.bucket,
        selected: Boolean(cutoff && (
          (row.knownDate && index === values.length - 1)
          || (!row.knownDate && row.bucket === cutoff)
        )),
        lifecycle: first(row.incident_state, row.lifecycle_state, row.current_state, "Unknown"),
        stage: first(
          stage,
          row.dominant_stage,
          row.stage_bucket,
          dominantDistributionStage(row.stage_distribution),
          "Unknown",
        ),
        pressure: numberValue(first(
          row.pressure_core_field_count,
          row.pressure_core_count,
          row.pressure_cell_count,
        ), 0),
        impact: numberValue(first(
          row.impact_lag_field_count,
          row.impact_lag_count,
          row.impact_cell_count,
        ), 0),
        unresolved: numberValue(first(
          row.unresolved_carried_field_count,
          row.unresolved_count,
        ), 0),
        areaKm2: numberValue(row.footprint_area_km2),
        carriedForward: truthy(row.footprint_carried_forward),
      };
    });
  return eligible;
}

export function incidentFootprintHistory(
  footprintValue = [], selectedBucket = "", incidentId = "",
) {
  const requested = validBucket(selectedBucket) ? String(selectedBucket).slice(0, 10) : "";
  const candidates = rows(footprintValue)
    .map((row) => footprintFeature(row, incidentId))
    .filter(Boolean)
    .sort((left, right) => (
      left.properties.timeline_bucket.localeCompare(right.properties.timeline_bucket)
    ));
  const selectedId = String(incidentId || candidates[0]?.properties?.incident_id || "");
  const matching = candidates.filter(
    (feature) => String(feature.properties.incident_id || "") === selectedId,
  );
  const cutoff = requested || matching.at(-1)?.properties?.timeline_bucket || "";
  const knowledgeClock = matching.some(
    (feature) => validBucket(feature.properties.story_known_date),
  );
  const eligible = matching.filter(
    (feature) => !cutoff || (
      knowledgeClock
        ? feature.properties.story_known_date <= cutoff
        : feature.properties.timeline_bucket <= cutoff
    ),
  );
  const current = knowledgeClock
    ? eligible.at(-1) || null
    : eligible.find((feature) => feature.properties.timeline_bucket === cutoff) || null;
  const priorRows = eligible.filter(
    (feature) => feature !== current,
  );
  const historyFeatures = priorRows.map((feature) => {
    const ageWeeks = weekDistance(
      knowledgeClock ? feature.properties.story_known_date : feature.properties.timeline_bucket,
      cutoff,
    );
    return {
      ...feature,
      properties: {
        ...feature.properties,
        age_index: ageWeeks,
        age_band: ageBand(ageWeeks),
        is_current: false,
        geometry_role: "exact_weekly_footprint",
        is_physical_movement: false,
      },
    };
  });
  return {
    current,
    prior: historyFeatures.at(-1) || null,
    collection: {
      type: "FeatureCollection",
      features: historyFeatures,
      meta: {
        incident_id: selectedId,
        selected_bucket: cutoff,
        current_available: Boolean(current),
        prior_count: historyFeatures.length,
        exact: true,
        is_physical_movement: false,
      },
    },
  };
}

export function footprintHistoryVisualModel(properties = {}) {
  const band = String(properties.age_band || "old");
  if (band === "recent") return { lineAlpha: 205, lineWidth: 2.1, dash: [7, 3] };
  if (band === "middle") return { lineAlpha: 145, lineWidth: 1.5, dash: [4, 4] };
  return { lineAlpha: 85, lineWidth: 1.0, dash: [2, 5] };
}

function footprintFeature(row, fallbackIncidentId) {
  const geometry = parseGeometry(row?.geometry || row?.geometry_geojson);
  if (!geometry || !["Polygon", "MultiPolygon"].includes(geometry.type)) return null;
  const bucket = bucketOf(row);
  if (!validBucket(bucket)) return null;
  const properties = { ...(row?.properties || row || {}) };
  delete properties.geometry;
  delete properties.geometry_geojson;
  properties.timeline_bucket = bucket;
  const knownDate = knownDateOf(properties);
  if (knownDate) properties.story_known_date = knownDate;
  properties.incident_id = String(
    properties.incident_id || fallbackIncidentId || "",
  );
  properties.geometry_role = "exact_weekly_footprint";
  properties.is_physical_movement = false;
  return { type: "Feature", geometry, properties };
}

function groupByBucket(values) {
  const grouped = new Map();
  for (const row of values) {
    const bucket = bucketOf(row);
    if (!validBucket(bucket)) continue;
    if (!grouped.has(bucket)) grouped.set(bucket, []);
    grouped.get(bucket).push(row);
  }
  return grouped;
}

function dominantAffectedStage(values) {
  const candidates = values.filter((row) => (
    row.affected_crop_instance_count === undefined
      || numberValue(row.affected_crop_instance_count, 0) > 0
  ));
  return candidates.sort((left, right) => (
    numberValue(right.affected_crop_instance_count, 0)
      - numberValue(left.affected_crop_instance_count, 0)
    || numberValue(right.pressure_core_crop_instance_count, 0)
      - numberValue(left.pressure_core_crop_instance_count, 0)
    || String(left.stage_bucket || "").localeCompare(String(right.stage_bucket || ""))
  ))[0]?.stage_bucket;
}

function dominantDistributionStage(value) {
  let parsed = value;
  if (typeof value === "string") {
    try { parsed = JSON.parse(value); } catch { parsed = null; }
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
  return Object.entries(parsed).sort((left, right) => (
    numberValue(right[1], 0) - numberValue(left[1], 0)
    || String(left[0]).localeCompare(String(right[0]))
  ))[0]?.[0];
}

function ageBand(ageWeeks) {
  if (ageWeeks <= 2) return "recent";
  if (ageWeeks <= 6) return "middle";
  return "old";
}

function weekDistance(bucket, cutoff) {
  const start = Date.parse(`${bucket}T00:00:00Z`);
  const end = Date.parse(`${cutoff}T00:00:00Z`);
  return Math.max(1, Math.round((end - start) / WEEK_MS));
}

function bucketOf(row = {}) {
  return String(
    row.story_week || row.timeline_bucket
      || row.properties?.story_week || row.properties?.timeline_bucket || "",
  ).slice(0, 10);
}

function knownDateOf(row = {}) {
  const value = row.story_known_date || row.knowledge_time
    || row.properties?.story_known_date || row.properties?.knowledge_time || "";
  const day = String(value).slice(0, 10);
  return validBucket(day) ? day : "";
}

function validBucket(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(value || "").slice(0, 10));
}

function parseGeometry(value) {
  if (value && typeof value === "object") return value;
  if (typeof value !== "string") return null;
  try { return JSON.parse(value); } catch { return null; }
}

function rows(value) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.rows)) return value.rows;
  if (Array.isArray(value?.items)) return value.items;
  return value && typeof value === "object" ? [value] : [];
}

function first(...values) {
  return values.find(
    (value) => value !== undefined && value !== null && String(value) !== "",
  );
}

function numberValue(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function truthy(value) {
  return value === true || value === 1
    || ["true", "1", "yes"].includes(String(value).toLowerCase());
}
