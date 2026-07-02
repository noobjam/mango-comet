export const MIN_TRAIL_OVERLAP = 0.15;

export function buildEvolutionModel(payload, currentBucket, { tailLength = 12 } = {}) {
  const points = evolutionRows(payload)
    .map(normalizePoint)
    .filter((point) => point.bucket && Number.isFinite(point.lon) && Number.isFinite(point.lat))
    .sort((left, right) => left.bucket.localeCompare(right.bucket))
    .filter((point) => !currentBucket || point.bucket <= currentBucket)
    .slice(-tailLength);
  const segments = [];
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const consecutive = dayGap(start.bucket, end.bucket) === 7;
    const enoughOverlap = Number.isFinite(end.overlap) && end.overlap >= MIN_TRAIL_OVERLAP;
    const allowed = end.segmentAllowed !== false && consecutive && enoughOverlap;
    if (allowed) segments.push({ start, end, overlap: end.overlap });
  }
  return {
    points,
    segments,
    dots: segments.flatMap(segmentDots),
    current: points.find((point) => !currentBucket || point.bucket === currentBucket) || null,
  };
}

export function evolutionSummary(model) {
  const point = model?.current;
  if (!point) return "No activity-center summary is available for this week.";
  const fields = point.fieldCount === null ? "Unknown field count" : `${point.fieldCount.toLocaleString()} open fields`;
  const spread = Number.isFinite(point.dispersionP90)
    ? `90% within ${formatNumber(point.dispersionP90)} km`
    : "dispersion unavailable";
  const overlap = Number.isFinite(point.overlap)
    ? `${Math.round(point.overlap * 100)}% overlap with the prior week`
    : "prior-week overlap unavailable";
  return `${fields} · ${spread} · ${overlap}. Aggregate activity center, not physical movement.`;
}

export function evolutionDeckLayers(deckApi, model, base) {
  if (!model.points.length) return [];
  const layers = [];
  if (Number.isFinite(model.current?.dispersionP90) && model.current.dispersionP90 > 0) {
    layers.push(new deckApi.ScatterplotLayer({
      id: "activity-center-dispersion",
      data: [model.current],
      getPosition: (point) => [point.lon, point.lat],
      radiusUnits: "meters",
      getRadius: (point) => point.dispersionP90 * 1000,
      getFillColor: [base[0], base[1], base[2], 18],
      getLineColor: [base[0], base[1], base[2], 105],
      lineWidthUnits: "pixels",
      getLineWidth: 1,
      filled: true,
      stroked: true,
      pickable: false,
    }));
  }
  if (model.dots.length) layers.push(new deckApi.ScatterplotLayer({
    id: "activity-center-trail-dots",
    data: model.dots,
    getPosition: (point) => point.position,
    radiusUnits: "pixels",
    getRadius: 1.8,
    getFillColor: (point) => [base[0], base[1], base[2], Math.round(70 + point.overlap * 150)],
    pickable: false,
  }));
  layers.push(new deckApi.ScatterplotLayer({
    id: "activity-center-history",
    data: model.points.filter((point) => point !== model.current),
    getPosition: (point) => [point.lon, point.lat],
    radiusUnits: "pixels",
    getRadius: 4,
    getLineColor: [base[0], base[1], base[2], 175],
    getLineWidth: 1.5,
    filled: false,
    stroked: true,
    pickable: false,
  }));
  if (model.current) layers.push(new deckApi.ScatterplotLayer({
    id: "activity-center-current",
    data: [model.current],
    getPosition: (point) => [point.lon, point.lat],
    radiusUnits: "pixels",
    getRadius: 8,
    getLineColor: [255, 255, 255, 245],
    getLineWidth: 2.5,
    filled: false,
    stroked: true,
    pickable: false,
  }));
  return layers;
}

export function evolutionGeoJson(model) {
  const dots = model.dots.map((point) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: point.position },
    properties: { kind: "trail-dot" },
  }));
  const centers = model.points.map((point) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [point.lon, point.lat] },
    properties: { kind: point === model.current ? "current-center" : "prior-center" },
  }));
  return { type: "FeatureCollection", features: [...dots, ...centers] };
}

function evolutionRows(payload) {
  if (Array.isArray(payload?.buckets)) return payload.buckets;
  if (Array.isArray(payload?.points)) return payload.points;
  if (Array.isArray(payload?.evolution)) return payload.evolution;
  return [];
}

function normalizePoint(row = {}) {
  const center = row.activity_center || row.center || {};
  return {
    bucket: String(row.timeline_bucket || row.observation_week || row.bucket || ""),
    lon: Number(row.center_lon ?? row.activity_center_lon ?? center.lon),
    lat: Number(row.center_lat ?? row.activity_center_lat ?? center.lat),
    fieldCount: optionalNumber(row.field_count ?? row.active_field_count),
    overlap: optionalNumber(
      row.field_overlap_jaccard ?? row.jaccard_overlap ?? row.overlap_jaccard ?? row.overlap_previous?.jaccard
    ),
    dispersionP50: optionalNumber(row.dispersion_p50_km ?? row.p50_dispersion_km ?? row.dispersion?.p50_km),
    dispersionP90: optionalNumber(row.dispersion_p90_km ?? row.p90_dispersion_km ?? row.dispersion?.p90_km),
    segmentAllowed: row.trail_segment_allowed,
  };
}

function segmentDots(segment) {
  const count = 12;
  return Array.from({ length: count - 1 }, (_, index) => {
    const fraction = (index + 1) / count;
    return {
      position: [
        segment.start.lon + (segment.end.lon - segment.start.lon) * fraction,
        segment.start.lat + (segment.end.lat - segment.start.lat) * fraction,
      ],
      overlap: segment.overlap,
    };
  });
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function dayGap(first, second) {
  const start = Date.parse(`${String(first).slice(0, 10)}T00:00:00Z`);
  const end = Date.parse(`${String(second).slice(0, 10)}T00:00:00Z`);
  return Number.isFinite(start) && Number.isFinite(end) ? Math.round((end - start) / 86400000) : null;
}

function formatNumber(value) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
}
