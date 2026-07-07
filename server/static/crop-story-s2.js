import {
  DAY_MS,
  dateValue,
  groupBy,
  s2KnowledgeDate,
  s2SourceDate,
  truthy,
} from "./crop-story-time.js";

const MAX_S2_VISUALS = 240;

export function s2LaneModels(sourceRows, start, end, position, freshnessPolicy = {}) {
  const normalized = sourceRows.map((row) => normalizeS2(row, position)).filter(Boolean);
  const markers = aggregateMarkers(normalized);
  const freshDays = policyDays(freshnessPolicy.fresh_max_days, 7);
  const agingDays = Math.max(
    freshDays,
    policyDays(freshnessPolicy.aging_max_days, 14),
  );
  return {
    events: markers.events,
    holds: aggregateHolds(normalized, start, end, position, freshDays, agingDays),
    sourceCount: normalized.reduce((sum, event) => sum + event.count, 0),
    aggregated: markers.aggregated,
  };
}

function normalizeS2(row = {}, position) {
  const sourceDate = s2SourceDate(row);
  const knowledgeDate = s2KnowledgeDate(row);
  const sourceX = position(sourceDate || knowledgeDate);
  const knowledgeX = position(knowledgeDate);
  if (sourceX === null || knowledgeX === null) return null;
  const markerType = String(row.marker_type || row.acquisition_status || "").toLowerCase();
  const rejected = markerType === "rejected"
    || (row.spectral_usable !== undefined && row.spectral_usable !== null
      && !truthy(row.spectral_usable));
  const response = String(row.response_class || "no_change").toLowerCase();
  return {
    ...row,
    sourceDate: String(sourceDate || knowledgeDate).slice(0, 10),
    knowledgeDate: String(knowledgeDate || "").slice(0, 10),
    sourceX,
    knowledgeX,
    rejected,
    usable: !rejected,
    response,
    responseKind: responseKind(response, rejected),
    freshness: String(
      row.evidence_freshness || row.source_freshness || row.spectral_freshness || "fresh",
    ).toLowerCase(),
    count: Math.max(1, Number(row.aggregate_count || row.field_count || 1)),
    entityKey: `${row.field_id || "aggregate"}:${row.crop_instance_id || "crop"}`,
  };
}

function aggregateMarkers(values) {
  const exact = mergeVisuals(values, (event) => [
    event.sourceDate, event.knowledgeDate, event.responseKind, event.rejected,
    event.freshness,
  ].join("|"));
  if (exact.length <= MAX_S2_VISUALS) {
    return { events: exact, aggregated: exact.length < values.length };
  }
  const category = (event) => [
    event.responseKind, event.rejected, event.freshness,
  ].join("|");
  const categoryCount = new Set(exact.map(category)).size;
  const binsPerCategory = Math.max(1, Math.floor(MAX_S2_VISUALS / categoryCount));
  const bucketed = mergeVisuals(exact, (event) => [
    Math.min(binsPerCategory - 1, Math.floor((event.knowledgeX / 100) * binsPerCategory)),
    category(event),
  ].join("|"), true).map((event) => ({ ...event, aggregateRange: true }));
  return { events: bucketed, aggregated: true };
}

function mergeVisuals(values, keyOf, mergeRange = false) {
  const grouped = new Map();
  for (const value of values) {
    const key = keyOf(value);
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        ...value,
        count: value.count,
        _weightedKnowledgeX: value.knowledgeX * value.count,
      });
      continue;
    }
    existing.count += value.count;
    existing._weightedKnowledgeX += value.knowledgeX * value.count;
    existing.sourceX = Math.min(existing.sourceX, value.sourceX);
    existing.sourceDate = [existing.sourceDate, value.sourceDate].filter(Boolean).sort()[0];
    if (mergeRange) {
      existing.knowledgeDate = [existing.knowledgeDate, value.knowledgeDate]
        .filter(Boolean).sort().at(-1);
    }
  }
  return [...grouped.values()].map((value) => {
    const result = { ...value, knowledgeX: value._weightedKnowledgeX / value.count };
    delete result._weightedKnowledgeX;
    return result;
  }).sort((left, right) => left.knowledgeX - right.knowledgeX
    || left.responseKind.localeCompare(right.responseKind));
}

function aggregateHolds(values, start, end, position, freshDays, agingDays) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
  const byEntity = groupBy(values.filter((event) => event.usable), (event) => event.entityKey);
  const holds = [];
  for (const events of byEntity.values()) {
    events.sort((left, right) => dateValue(left.knowledgeDate) - dateValue(right.knowledgeDate));
    events.forEach((event, index) => {
      const known = Math.max(start, dateValue(event.knowledgeDate));
      const stop = Math.min(
        end + DAY_MS,
        index + 1 < events.length ? dateValue(events[index + 1].knowledgeDate) : end + DAY_MS,
      );
      const source = dateValue(event.sourceDate);
      if (!Number.isFinite(known) || !Number.isFinite(stop) || stop <= known) return;
      const ranges = [
        ["fresh", known, Math.min(stop, source + (freshDays + 1) * DAY_MS)],
        ["aging", Math.max(known, source + (freshDays + 1) * DAY_MS),
          Math.min(stop, source + (agingDays + 1) * DAY_MS)],
        ["stale", Math.max(known, source + (agingDays + 1) * DAY_MS), stop],
      ];
      for (const [freshness, lower, upper] of ranges) {
        if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) continue;
        const startX = position(lower);
        const endX = position(Math.min(end, upper));
        if (startX === null || endX === null) continue;
        holds.push({
          startX,
          width: Math.max(0.25, endX - startX),
          freshness,
          responseKind: event.responseKind,
          count: event.count,
        });
      }
    });
  }
  const exact = mergeHoldSegments(holds, false);
  return exact.length <= MAX_S2_VISUALS ? exact : mergeHoldSegments(exact, true);
}

function policyDays(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function mergeHoldSegments(values, bucketed) {
  const grouped = new Map();
  for (const value of values) {
    const startBin = bucketed
      ? Math.min(11, Math.floor(value.startX * 0.12)) : value.startX.toFixed(3);
    const key = bucketed
      ? `${startBin}|${value.freshness}|${value.responseKind}`
      : `${startBin}|${value.width.toFixed(3)}|${value.freshness}|${value.responseKind}`;
    const existing = grouped.get(key);
    if (!existing) grouped.set(key, { ...value });
    else {
      const endX = Math.max(existing.startX + existing.width, value.startX + value.width);
      existing.startX = Math.min(existing.startX, value.startX);
      existing.width = endX - existing.startX;
      existing.count += value.count;
    }
  }
  return [...grouped.values()].sort((left, right) => left.startX - right.startX);
}

function responseKind(response, rejected) {
  if (rejected) return "rejected";
  if (response.includes("recover")) return "recovery";
  if (response.includes("severe") && response.includes("decline")) return "severe-decline";
  if (response.includes("decline")) return "decline";
  return "no-change";
}
