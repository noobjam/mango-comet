import { stageBandModel, storyLaneModels } from "./crop-story-lifecycle.js";
import { pressureLaneModels } from "./crop-story-pressure.js";
import { s2LaneModels } from "./crop-story-s2.js";
import {
  DAY_MS,
  dateValue,
  pressureDate,
  rows,
  s2KnowledgeDate,
  s2SourceDate,
  storyKnowledgeDate,
  storySourceDate,
  timePosition,
  timeTicks,
} from "./crop-story-time.js";

export function fieldEvidenceRibbonModel(payload = {}, selectedDate = "") {
  return cropStoryTrajectoryModel(payload, selectedDate);
}

export function cropStoryTrajectoryModel(payload = {}, selectedDate = "") {
  const lanes = payload.lanes || {};
  const pressureRows = firstRows(lanes.daily_pressure, payload.daily_pressure);
  const s2Rows = firstRows(lanes.s2_attempts, payload.s2_attempts, payload.s2_updates);
  const storyRows = firstRows(
    lanes.story_checkpoints,
    payload.story_checkpoints,
    payload.weekly_state,
  );
  const currentRows = rows(payload.current_state);
  const candidateDates = [
    payload.history?.window_start,
    payload.history?.window_end,
    payload.as_of_date,
    selectedDate,
    ...pressureRows.map(pressureDate),
    ...s2Rows.flatMap((row) => [s2SourceDate(row), s2KnowledgeDate(row)]),
    ...storyRows.flatMap((row) => [storySourceDate(row), storyKnowledgeDate(row)]),
  ].map(dateValue).filter(Number.isFinite);
  const explicitStart = dateValue(payload.history?.window_start);
  const explicitEnd = dateValue(payload.history?.window_end || payload.as_of_date);
  const start = Number.isFinite(explicitStart)
    ? explicitStart : candidateDates.length ? Math.min(...candidateDates) : NaN;
  const rawEnd = Number.isFinite(explicitEnd)
    ? explicitEnd : candidateDates.length ? Math.max(...candidateDates) : NaN;
  const end = Number.isFinite(start) && Number.isFinite(rawEnd)
    ? Math.max(rawEnd, start + DAY_MS) : NaN;
  const position = timePosition(start, end);
  const hazards = [...new Set([
    ...pressureRows.map((row) => String(row.hazard_family || "").toLowerCase()),
    ...storyRows.map((row) => String(row.hazard_family || "").toLowerCase()),
    ...currentRows.map((row) => String(
      row.highest_pressure_hazard || row.hazard_family || "",
    ).toLowerCase()),
  ].filter((value) => value && value !== "none"))].sort();
  const pressure = pressureLaneModels(pressureRows, hazards, start, end);
  const s2 = s2LaneModels(
    s2Rows, start, end, position, payload.freshness_policy || {},
  );
  const stage = stageBandModel(storyRows, s2Rows, currentRows, position);
  const stories = storyLaneModels(storyRows, payload, position);
  const storySourceLanes = new Set(storyRows.map((row) => (
    String(row.incident_id || payload.incident_id || payload.window?.incident_id || "story")
  ))).size;
  return {
    start,
    end,
    selectedX: position(selectedDate || payload.as_of_date),
    ticks: timeTicks(start, end),
    hazards,
    lanes: { pressure, s2: s2.events, s2Holds: s2.holds, stage, stories },
    counts: {
      pressureRows: pressureRows.length,
      s2Rows: s2.sourceCount,
      s2Visuals: s2.events.length,
      storyRows: storyRows.length,
      storySourceLanes,
      storyLanes: stories.length,
      storyLanesOmitted: Math.max(0, storySourceLanes - stories.length),
    },
    storyAggregated: storyRows.length > stories.reduce(
      (sum, story) => sum + story.milestones.length, 0,
    ) || storySourceLanes > stories.length,
    s2Aggregated: s2.aggregated,
    truncated: payload.history?.truncated || {},
    hasEvidence: Boolean(
      pressureRows.length || s2Rows.length || storyRows.length || currentRows.length,
    ),
  };
}

function firstRows(...values) {
  for (const value of values) {
    const candidate = rows(value);
    if (candidate.length) return candidate;
  }
  return [];
}
