import {
  dateValue,
  groupBy,
  pretty,
  shortId,
  storyKnowledgeDate,
  storySourceDate,
  truthy,
} from "./crop-story-time.js";

const MAX_STAGE_SEGMENTS = 120;
const MAX_STORY_LANES = 64;
const MAX_STORY_VISUALS = 240;

export function stageBandModel(storyRows, s2Rows, currentRows, position) {
  const events = [
    ...storyRows.map((row) => ({
      date: storyKnowledgeDate(row),
      stage: row.stage_bucket || dominantStage(row.stage_distribution),
      priority: 3,
    })),
    ...s2Rows.filter(usableS2Stage).map((row) => ({
      date: row.knowledge_date || row.knowledge_time,
      stage: row.stage_bucket,
      priority: 2,
    })),
    ...currentRows.map((row) => ({
      date: row.crop_knowledge_time || row.crop_observation_date
        || row.stage_source_date || row.calendar_date,
      stage: row.stage_bucket,
      priority: 1,
    })),
  ].filter((event) => event.stage && position(event.date) !== null)
    .sort((left, right) => String(left.date).localeCompare(String(right.date))
      || left.priority - right.priority);
  const byDate = new Map(events.map((event) => [String(event.date).slice(0, 10), event]));
  const ordered = [...byDate.values()].sort(
    (left, right) => String(left.date).localeCompare(String(right.date)),
  );
  if (!ordered.length) return [{ stage: "unknown", startX: 0, width: 100, unknown: true }];
  const segments = [];
  const firstX = position(ordered[0].date);
  if (firstX > 0) {
    segments.push({ stage: "unknown", startX: 0, width: firstX, unknown: true });
  }
  ordered.forEach((event, index) => {
    const startX = position(event.date);
    const nextX = index + 1 < ordered.length ? position(ordered[index + 1].date) : 100;
    const segment = {
      stage: String(event.stage || "unknown"),
      startX: Math.min(99.6, startX),
      width: Math.max(0.4, nextX - startX),
      unknown: String(event.stage || "").toLowerCase() === "unknown",
    };
    const previous = segments.at(-1);
    if (previous && previous.stage === segment.stage
      && Math.abs(previous.startX + previous.width - segment.startX) < 0.01) {
      previous.width = segment.startX + segment.width - previous.startX;
    } else segments.push(segment);
  });
  return boundedTransitions(segments, MAX_STAGE_SEGMENTS);
}

export function storyLaneModels(sourceRows, payload, position) {
  const fallbackId = String(payload.incident_id || payload.window?.incident_id || "story");
  const normalized = sourceRows.map((row) => {
    const knowledgeX = position(storyKnowledgeDate(row));
    const sourceX = position(storySourceDate(row));
    if (knowledgeX === null || sourceX === null) return null;
    return {
      ...row,
      incidentId: String(row.incident_id || fallbackId),
      storyWeek: String(storySourceDate(row) || "").slice(0, 10),
      knowledgeDate: String(storyKnowledgeDate(row) || "").slice(0, 10),
      sourceX,
      knowledgeX,
      lifecycle: String(row.incident_state || row.current_state || "unknown"),
      stage: String(row.stage_bucket || dominantStage(row.stage_distribution) || "unknown"),
    };
  }).filter(Boolean);
  const grouped = groupBy(normalized, (row) => row.incidentId);
  const recentGroups = [...grouped.values()].sort((left, right) => (
    Math.max(...right.map((row) => dateValue(row.knowledgeDate)))
      - Math.max(...left.map((row) => dateValue(row.knowledgeDate)))
  )).slice(0, MAX_STORY_LANES);
  const perLaneLimit = Math.max(2, Math.floor(MAX_STORY_VISUALS / recentGroups.length));
  return recentGroups.map((values) => storyLane(values, payload, perLaneLimit))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function storyLane(values, payload, limit) {
  values.sort((left, right) => left.knowledgeX - right.knowledgeX);
  values = boundedRows(values, limit);
  const first = values[0];
  const blocks = [];
  if (first.knowledgeX > 0) {
    blocks.push({ startX: 0, width: first.knowledgeX, lifecycle: "not_known", stage: "unknown" });
  }
  values.forEach((value, index) => {
    const nextX = index + 1 < values.length ? values[index + 1].knowledgeX : 100;
    blocks.push({
      startX: value.knowledgeX,
      width: Math.max(0.45, nextX - value.knowledgeX),
      lifecycle: value.lifecycle,
      stage: value.stage,
      rightCensored: truthy(value.right_censored),
    });
  });
  const milestones = values.map((value, index) => ({
    kind: milestoneKind(value.lifecycle, index),
    x: value.knowledgeX,
    sourceX: value.sourceX,
    label: `${value.storyWeek} story week → ${value.knowledgeDate} known · ${pretty(value.lifecycle)} · ${pretty(value.stage)}`,
  }));
  deduplicateMilestoneKinds(milestones);
  const crop = first.crop_name || payload.window?.crop_name || "Crop";
  const hazard = first.hazard_family || payload.window?.hazard_family || "hazard";
  return {
    incidentId: first.incidentId,
    label: `${pretty(crop)} · ${pretty(hazard)} · ${shortId(first.incidentId)}`,
    blocks,
    milestones,
  };
}

function usableS2Stage(row = {}) {
  if (row.spectral_usable !== undefined && row.spectral_usable !== null) {
    return truthy(row.spectral_usable);
  }
  const status = String(
    row.marker_type || row.acquisition_status || "",
  ).toLowerCase();
  return ["usable", "accepted", "acquisition"].includes(status);
}

function boundedRows(values, limit) {
  const transitions = values.filter((value, index) => (
    index === 0 || index === values.length - 1
      || value.lifecycle !== values[index - 1].lifecycle
      || value.stage !== values[index - 1].stage
  ));
  if (transitions.length <= limit) return transitions;
  return sampleEndpoints(transitions, limit);
}

function boundedTransitions(values, limit) {
  if (values.length <= limit) return values;
  const sampled = sampleEndpoints(values, limit);
  return sampled.map((value, index) => {
    const nextStart = index + 1 < sampled.length
      ? sampled[index + 1].startX : 100;
    return { ...value, width: Math.max(0, nextStart - value.startX) };
  });
}

function sampleEndpoints(values, limit) {
  if (limit <= 1) return values.slice(-1);
  const indexes = new Set([0, values.length - 1]);
  for (let index = 1; index < limit - 1; index += 1) {
    indexes.add(Math.round((index * (values.length - 1)) / (limit - 1)));
  }
  return [...indexes].sort((left, right) => left - right)
    .map((index) => values[index]);
}

function milestoneKind(lifecycle, index) {
  const state = String(lifecycle || "").toLowerCase();
  if (index === 0) return "start";
  if (state.includes("closed") || state.includes("merged")) return "closed";
  if (state.includes("recover")) return "recovery";
  if (state.includes("quiet")) return "pressure-off";
  return "checkpoint";
}

function deduplicateMilestoneKinds(milestones) {
  const used = new Set();
  for (const milestone of milestones) {
    if (milestone.kind === "checkpoint") continue;
    if (used.has(milestone.kind)) milestone.kind = "checkpoint";
    used.add(milestone.kind);
  }
}

function dominantStage(value) {
  let parsed = value;
  if (typeof value === "string") {
    try { parsed = JSON.parse(value); } catch { return null; }
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  return Object.entries(parsed).sort(
    (left, right) => Number(right[1]) - Number(left[1]),
  )[0]?.[0];
}
