import { cropStoryTrajectoryModel } from "./crop-story-trajectory-model.js";
import {
  compactCount,
  cssToken,
  pretty,
  shortDay,
} from "./crop-story-time.js";

export {
  cropStoryTrajectoryModel,
  fieldEvidenceRibbonModel,
} from "./crop-story-trajectory-model.js";

export function cropStoryTrajectory(payload, selectedDate, title = "Crop-story trajectory") {
  const model = cropStoryTrajectoryModel(payload, selectedDate);
  const wrapper = document.createElement("section");
  wrapper.className = "crop-story-trajectory";
  wrapper.setAttribute("aria-label", "Aligned causal crop-story trajectory");
  const heading = document.createElement("div");
  heading.className = "trajectory-heading";
  const s2Summary = model.s2Aggregated
    ? `${model.counts.s2Rows.toLocaleString()} S2 attempts · aggregated display`
    : `${model.counts.s2Rows.toLocaleString()} S2 attempts`;
  const storySummary = model.counts.storyLanesOmitted
    ? `${model.counts.storyLanes.toLocaleString()} of ${model.counts.storySourceLanes.toLocaleString()} story rows shown`
    : model.storyAggregated ? "story transitions condensed" : "";
  heading.append(
    node("strong", title),
    node("span", [s2Summary, storySummary].filter(Boolean).join(" · "), "muted"),
  );
  wrapper.appendChild(heading);
  if (!model.hasEvidence) {
    wrapper.appendChild(node("span", "No causally available evidence in this window.", "muted"));
    return wrapper;
  }
  for (const laneModel of model.lanes.pressure) {
    const lane = trajectoryLane(`Pressure · ${pretty(laneModel.hazard)}`, "pressure");
    addSelectedCursor(lane.track, model.selectedX);
    lane.track.style.setProperty("--hazard-color", laneModel.color);
    for (const cell of laneModel.cells) {
      const marker = node(
        "span", "", `trajectory-pressure-cell is-${cell.state} risk-${Math.min(4, cell.riskRank)}`,
      );
      placeSpan(marker, cell.startX, cell.width);
      marker.title = pressureCellLabel(laneModel.hazard, cell);
      marker.setAttribute("aria-label", marker.title);
      marker.setAttribute("role", "listitem");
      lane.track.appendChild(marker);
    }
    wrapper.appendChild(lane.element);
  }
  wrapper.appendChild(s2Lane(model));
  wrapper.appendChild(stageLane(model));
  for (const story of model.lanes.stories) wrapper.appendChild(storyLane(story, model));
  if (!model.lanes.stories.length) {
    const lane = trajectoryLane("Crop story", "story");
    lane.track.appendChild(node("span", "No weekly checkpoint known yet", "trajectory-empty"));
    wrapper.appendChild(lane.element);
  }
  wrapper.appendChild(trajectoryAxis(model.ticks));
  wrapper.appendChild(node(
    "span",
    "Shared linear time · pressure gaps are explicit · S2 is step-held and ages · story state changes only when known",
    "truth-note",
  ));
  return wrapper;
}

function s2Lane(model) {
  const lane = trajectoryLane("Sentinel-2 crop evidence", "s2");
  addSelectedCursor(lane.track, model.selectedX);
  for (const hold of model.lanes.s2Holds) {
    const segment = node("span", "", `trajectory-s2-hold is-${cssToken(hold.freshness)}`);
    placeSpan(segment, hold.startX, hold.width);
    segment.title = `${hold.count.toLocaleString()} held crop-evidence state${hold.count === 1 ? "" : "s"} · ${hold.freshness}`;
    lane.track.appendChild(segment);
  }
  for (const event of model.lanes.s2) {
    if (!event.aggregateRange) {
      const sourceX = Math.min(event.sourceX, event.knowledgeX);
      const knowledgeX = Math.max(event.sourceX, event.knowledgeX);
      const clock = node("span", "", "trajectory-clock-span is-s2");
      placeSpan(clock, sourceX, Math.max(0.25, knowledgeX - sourceX));
      lane.track.appendChild(clock);
      const source = node("span", "", "trajectory-s2-source");
      source.style.setProperty("--x", `${event.sourceX}%`);
      lane.track.appendChild(source);
    }
    const marker = node(
      "span",
      event.count > 1 ? compactCount(event.count) : "",
      `trajectory-s2-marker is-${event.responseKind}${event.rejected ? " is-rejected" : ""}`,
    );
    marker.style.setProperty("--x", `${event.knowledgeX}%`);
    marker.title = event.aggregateRange
      ? `${event.count.toLocaleString()} grouped attempts · individual source-to-known lags retained in data · ${event.rejected ? "rejected" : pretty(event.response)}`
      : `${event.sourceDate} source → ${event.knowledgeDate} known · ${event.rejected ? "rejected" : pretty(event.response)} · ${event.count.toLocaleString()} attempt${event.count === 1 ? "" : "s"}`;
    marker.setAttribute("aria-label", marker.title);
    marker.setAttribute("role", "listitem");
    lane.track.appendChild(marker);
  }
  if (!model.lanes.s2.length) lane.track.appendChild(node("span", "No S2 attempts", "trajectory-empty"));
  return lane.element;
}

function stageLane(model) {
  const lane = trajectoryLane("Crop stage", "stage");
  addSelectedCursor(lane.track, model.selectedX);
  for (const segment of model.lanes.stage) {
    const value = node(
      "span",
      segment.unknown ? "Unknown" : pretty(segment.stage),
      `trajectory-stage-segment${segment.unknown ? " is-unknown" : ""}`,
    );
    placeSpan(value, segment.startX, segment.width);
    value.title = segment.unknown ? "Crop stage not yet known" : pretty(segment.stage);
    lane.track.appendChild(value);
  }
  return lane.element;
}

function storyLane(story, model) {
  const lane = trajectoryLane(story.label, "story");
  addSelectedCursor(lane.track, model.selectedX);
  for (const block of story.blocks) {
    const value = node(
      "span",
      block.lifecycle === "not_known" ? "Prelude" : pretty(block.lifecycle),
      `trajectory-story-block state-${cssToken(block.lifecycle)}${block.rightCensored ? " is-open" : ""}`,
    );
    placeSpan(value, block.startX, block.width);
    value.title = block.lifecycle === "not_known"
      ? "Evidence exists, but this story was not known yet"
      : `${pretty(block.lifecycle)} · ${pretty(block.stage)}`;
    lane.track.appendChild(value);
  }
  for (const milestone of story.milestones) {
    const connector = node("span", "", "trajectory-clock-span is-story");
    placeSpan(
      connector,
      Math.min(milestone.sourceX, milestone.x),
      Math.max(0.25, Math.abs(milestone.x - milestone.sourceX)),
    );
    lane.track.appendChild(connector);
    const marker = node("span", "", `trajectory-story-milestone is-${milestone.kind}`);
    marker.style.setProperty("--x", `${milestone.x}%`);
    marker.title = milestone.label;
    marker.setAttribute("aria-label", milestone.label);
    lane.track.appendChild(marker);
  }
  return lane.element;
}

function trajectoryLane(label, kind) {
  const element = document.createElement("div");
  element.className = `trajectory-evidence-lane is-${kind}`;
  element.appendChild(node("span", label, "trajectory-evidence-label"));
  const track = document.createElement("div");
  track.className = `trajectory-evidence-track is-${kind}`;
  track.setAttribute("role", "list");
  element.appendChild(track);
  return { element, track };
}

function trajectoryAxis(ticks) {
  const axis = document.createElement("div");
  axis.className = "trajectory-axis";
  axis.appendChild(node("span", "Time", "trajectory-evidence-label"));
  const track = node("div", "", "trajectory-axis-track");
  for (const tick of ticks) {
    const value = node("time", shortDay(tick.time), "trajectory-axis-tick");
    value.style.setProperty("--x", `${tick.x}%`);
    track.appendChild(value);
  }
  axis.appendChild(track);
  return axis;
}

function addSelectedCursor(track, selectedX) {
  if (selectedX === null) return;
  const cursor = node("span", "", "trajectory-selected-day");
  cursor.style.setProperty("--x", `${selectedX}%`);
  track.appendChild(cursor);
}

function placeSpan(element, startX, width) {
  element.style.setProperty("--x", `${Math.max(0, startX)}%`);
  element.style.setProperty("--width", `${Math.max(0.2, Math.min(width, 100 - startX))}%`);
}

function pressureCellLabel(hazard, cell) {
  const period = cell.startDate === cell.endDate
    ? cell.startDate : `${cell.startDate} to ${cell.endDate}`;
  if (cell.state === "missing") return `${period} · ${pretty(hazard)} · weather pressure unavailable`;
  const coverage = cell.monitoredCount > 1
    ? ` · ${cell.observedCount.toLocaleString()} of ${cell.monitoredCount.toLocaleString()} fields observed`
    : "";
  const state = cell.state === "observed-low"
    ? "observed low pressure" : `risk rank ${cell.riskRank}`;
  return `${period} · ${pretty(hazard)} · ${state}${coverage}`;
}

function node(tag, text, className = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = String(text || "");
  return element;
}
