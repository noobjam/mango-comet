import { INCIDENT_TRUTH_LABEL, incidentDetailModel } from "./incident-v3.js";

export class Inspector {
  constructor() {
    this.tooltip = document.getElementById("hoverTooltip");
    this.selection = document.getElementById("selectionDetails");
    this.selection.setAttribute("aria-live", "polite");
    this.selection.tabIndex = -1;
  }

  showHover(properties, point) {
    if (!properties || !point) {
      this.tooltip.hidden = true;
      return;
    }
    const incident = properties.incident_id && !properties.field_id;
    this.tooltip.replaceChildren(...(incident ? [
      node("strong", properties.crop_name || properties.crop_name_normalized || "Crop incident"),
      node("div", [properties.incident_state, properties.hazard_family].filter(Boolean).join(" · ")),
      node("div", "Exact monitored footprint", "muted"),
    ] : [
      node("strong", properties.field_id || "Field"),
      node("div", readableStory(properties)),
      node("div", locationText(properties), "muted"),
    ]));
    this.tooltip.hidden = false;
    const width = this.tooltip.offsetWidth || 240;
    const height = this.tooltip.offsetHeight || 70;
    const parent = this.tooltip.parentElement.getBoundingClientRect();
    const left = Math.min(parent.width - width - 10, Math.max(10, Number(point.x || 0) + 14));
    const top = Math.min(parent.height - height - 10, Math.max(10, Number(point.y || 0) + 14));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
  }

  loading(properties) {
    this.selection.className = "detail";
    this.selection.replaceChildren(...fieldHeader(properties), node("span", "Loading event windows…", "muted"));
  }

  loadingIncident(properties) {
    this.selection.className = "detail";
    this.selection.replaceChildren(
      ...incidentHeader(properties),
      node("span", "Loading crop-incident evidence…", "muted"),
    );
  }

  showIncident(properties, payload = {}) {
    const model = incidentDetailModel(payload, properties);
    const children = incidentHeader({ ...properties, incident_id: model.incidentId });
    children.push(
      detailPair("Crop", pretty(model.crop)),
      ...(model.coincident.count > 1 ? [
        detailPair(
          "Co-located crop stories",
          `${count(model.coincident.count)} · ${model.coincident.crops.map(pretty).join(", ")} · click again to cycle`,
        ),
      ] : []),
      detailPair("Lifecycle", pretty(model.lifecycle)),
      selectedWeekEvidenceView(model),
      lifecycleDatesView(model.lifecycleDates),
      storyArcView(model.storyArc),
      footprintContextView(model),
      detailPair("Dominant dynamic stage", pretty(model.dominantStage)),
      stageDistributionView(model.stageDistribution),
      incidentMetrics(model.counts),
      detailPair("Fresh evidence", `${count(model.counts.freshDecline)} decline · ${count(model.counts.freshRecovery)} recovery`),
      detailPair("Carry / recovery", `${count(model.counts.unresolved)} unresolved carried · ${count(model.counts.recovered)} recovered`),
      detailPair("Lineage / recurrence", `${count(model.counts.split)} split · ${count(model.counts.merge)} merge · ${count(model.counts.relapse)} relapse`),
      evidenceView(model.evidence),
      node("p", INCIDENT_TRUTH_LABEL, "truth-note incident-truth"),
    );
    this.selection.replaceChildren(...children);
  }

  showIncidentError(properties, error) {
    this.selection.className = "detail";
    this.selection.replaceChildren(
      ...incidentHeader(properties),
      node("span", `Could not load incident evidence: ${error.message}`, "muted"),
      node("p", INCIDENT_TRUTH_LABEL, "truth-note incident-truth"),
    );
  }

  showSelection(properties, events = [], trajectory = [], errors = {}) {
    this.selection.className = "detail";
    const children = fieldHeader(properties);
    if (trajectory.length) {
      children.push(
        node("strong", `${trajectory.length.toLocaleString()} causal weekly prefix states`),
        prefixTrajectoryRibbon(trajectory, properties.timeline_bucket),
      );
    } else if (errors.trajectoryError) {
      children.push(node("span", `Weekly prefix states unavailable: ${errors.trajectoryError.message}`, "muted"));
    }
    if (!events.length) {
      children.push(node(
        "span",
        errors.eventsError ? `Event windows unavailable: ${errors.eventsError.message}` : "No event windows were found for this field.",
        "muted",
      ));
    } else {
      const heading = node("strong", `${events.length.toLocaleString()} recent event windows`);
      children.push(heading, lifecycleRibbon(events, properties.timeline_bucket));
      for (const event of events.slice(0, 12)) {
        const row = document.createElement("div");
        row.className = "detail-row";
        const end = event.event_end_date || (event.right_censored ? event.as_of_date : null) || event.active_end_date || "?";
        const dates = node("span", `${event.event_start_date || "?"} → ${end}${event.right_censored ? " (open)" : ""}`);
        const evidence = node("strong", [event.max_risk_band, event.hazard_signature].filter(Boolean).join(" · "));
        row.append(dates, evidence);
        children.push(row);
      }
    }
    this.selection.replaceChildren(...children);
  }

  showError(properties, error) {
    this.selection.className = "detail";
    this.selection.replaceChildren(...fieldHeader(properties), node("span", `Could not load event windows: ${error.message}`, "muted"));
  }

  clear(message = "Select a field to inspect its event windows.") {
    this.selection.className = "detail muted";
    this.selection.textContent = message;
  }
}

function incidentHeader(properties = {}) {
  const title = node(
    "strong",
    properties.crop_name || properties.crop_name_normalized || "Selected crop incident",
  );
  const context = node(
    "div",
    [properties.incident_state, properties.hazard_family].filter(Boolean).map(pretty).join(" · "),
  );
  const id = document.createElement("code");
  id.textContent = properties.incident_id || properties.story_cluster_id || "No incident ID";
  return [title, context, id];
}

function stageDistributionView(distribution = []) {
  const wrapper = document.createElement("section");
  wrapper.className = "stage-distribution";
  wrapper.setAttribute("aria-label", "Dynamic crop-stage distribution");
  wrapper.appendChild(node("strong", "Dynamic stage distribution"));
  if (!distribution.length) {
    wrapper.appendChild(node("span", "Stage evidence unavailable", "muted"));
    return wrapper;
  }
  for (const item of distribution) {
    const row = document.createElement("div");
    row.className = "stage-row";
    const label = node("span", pretty(item.stage));
    const meter = document.createElement("span");
    meter.className = "stage-meter";
    meter.style.setProperty("--stage-share", `${Math.min(100, item.share * 100)}%`);
    meter.setAttribute("aria-label", `${pretty(item.stage)} ${Math.round(item.share * 100)} percent`);
    row.append(label, meter, node("span", `${Math.round(item.share * 100)}%`, "stage-share"));
    wrapper.appendChild(row);
  }
  return wrapper;
}

function incidentMetrics(counts = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "incident-metrics";
  for (const [label, value] of [
    ["Monitored", counts.monitored],
    ["Evaluable", counts.evaluable],
    ["Affected", counts.affected],
    ["Severe", counts.severe],
  ]) {
    const item = document.createElement("div");
    item.append(node("strong", count(value)), node("span", label));
    wrapper.appendChild(item);
  }
  return wrapper;
}

function evidenceView(evidence = {}) {
  const gaps = [];
  if (!evidence.coverageAdequate) gaps.push("coverage not adequate this week");
  if (evidence.dataGapCount) gaps.push(`${count(evidence.dataGapCount)} lifecycle data gaps`);
  if (evidence.coverageMissingCellCount) {
    gaps.push(`${count(evidence.coverageMissingCellCount)} coverage cells missing`);
  }
  if (evidence.globalCropWeekUnmappableInstanceCount) {
    gaps.push(
      `${count(evidence.globalCropWeekUnmappableInstanceCount)} national crop-week rows without map geometry`,
    );
  }
  if (evidence.carriedForward) gaps.push("footprint carried forward; no fresh extent inferred");
  if (evidence.rightCensored) gaps.push("still open at the observation cutoff");
  return detailPair("Evidence gaps", gaps.length ? gaps.join(" · ") : "No flagged gap in the selected evidence summary");
}

function lifecycleDatesView(dates = {}) {
  const values = [
    ["first evidence", dates.firstEvidence],
    ["confirmed", dates.confirmed],
    ["pressure off", dates.pressureOff],
    ["recovered", dates.recovered],
    ["closed", dates.closed],
  ].filter(([, value]) => value);
  return detailPair(
    "Lifecycle dates",
    values.length
      ? values.map(([label, value]) => `${label}: ${String(value).slice(0, 10)}`).join(" · ")
      : "No dated transition reported",
  );
}

function selectedWeekEvidenceView(model = {}) {
  if (model.observedThisWeek) {
    return detailPair("Selected week evidence", "Incident row observed");
  }
  const prior = model.lastObservedBucket
    ? ` · showing last observed ${String(model.lastObservedBucket).slice(0, 10)}`
    : "";
  return detailPair("Selected week evidence", `No incident row this week${prior}`);
}

function storyArcView(weeks = []) {
  const wrapper = document.createElement("section");
  wrapper.className = "incident-story-arc";
  wrapper.setAttribute("aria-label", "Week-by-week crop incident story");
  const heading = document.createElement("div");
  heading.className = "story-arc-heading";
  heading.append(
    node("strong", "Week-by-week crop story"),
    node("span", `${weeks.length.toLocaleString()} observed weeks`, "muted"),
  );
  wrapper.appendChild(heading);
  if (!weeks.length) {
    wrapper.appendChild(node("span", "No incident evidence at or before this week.", "muted"));
    return wrapper;
  }
  const list = document.createElement("div");
  list.className = "story-arc-weeks";
  for (const week of weeks) {
    const row = document.createElement("div");
    row.className = [
      "story-arc-week",
      week.selected ? "is-selected" : "",
      week.carriedForward ? "is-carried" : "",
    ].filter(Boolean).join(" ");
    row.setAttribute(
      "aria-label",
      `${week.bucket}, ${pretty(week.lifecycle)}, ${pretty(week.stage)}, `
        + `${count(week.pressure)} pressure, ${count(week.impact)} impact, `
        + `${count(week.unresolved)} unresolved, ${areaLabel(week.areaKm2)}`,
    );
    row.append(
      node("time", week.bucket.slice(5), "story-arc-date"),
      node("span", pretty(week.lifecycle), "story-arc-state"),
      node("span", pretty(week.stage), "story-arc-stage"),
      node(
        "span",
        `P ${count(week.pressure)} · I ${count(week.impact)} · U ${count(week.unresolved)}`,
        "story-arc-counts",
      ),
      node("span", areaLabel(week.areaKm2), "story-arc-area"),
    );
    list.appendChild(row);
  }
  wrapper.appendChild(list);
  return wrapper;
}

function footprintContextView(model = {}) {
  const current = model.currentFootprint?.properties?.timeline_bucket;
  const prior = model.priorFootprint?.properties?.timeline_bucket;
  const values = [];
  if (current) values.push(`current ${String(current).slice(0, 10)}`);
  if (prior) values.push(`prior ${String(prior).slice(0, 10)}`);
  return detailPair(
    "Exact footprint outlines",
    values.length ? values.join(" · ") : "No exact outline at or before this week",
  );
}

function areaLabel(value) {
  const area = Number(value);
  return Number.isFinite(area) ? `${area.toLocaleString(undefined, { maximumFractionDigits: 2 })} km²` : "area —";
}

function detailPair(label, value) {
  const row = document.createElement("div");
  row.className = "detail-row incident-detail-row";
  row.append(node("span", label), node("strong", value));
  return row;
}

function count(value) {
  return value === null || value === undefined ? "—" : Number(value).toLocaleString();
}

function pretty(value) {
  return String(value || "Unknown").replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

export function lifecycleModel(events = [], selectedBucket = "") {
  const valid = events.map((event) => ({
    event,
    start: dateValue(event.event_start_date),
    end: dateValue(
      event.event_end_date
      || (event.right_censored ? event.as_of_date : null)
      || event.active_end_date
      || event.event_start_date
    ),
    pressureEnd: dateValue(event.active_end_date || event.event_start_date),
  })).filter((item) => Number.isFinite(item.start) && Number.isFinite(item.end));
  if (!valid.length) return { items: [], start: null, end: null };
  const start = Math.min(...valid.map((item) => item.start));
  const end = Math.max(...valid.map((item) => item.end), start + 86400000);
  const selected = dateValue(selectedBucket);
  const range = end - start;
  return {
    start,
    end,
    items: valid.sort((left, right) => left.start - right.start).map(({ event, start: itemStart, end: itemEnd, pressureEnd }) => ({
      event,
      startPercent: ((itemStart - start) / range) * 100,
      widthPercent: Math.max(2, ((Math.max(itemEnd, itemStart) - itemStart + 86400000) / range) * 100),
      pressurePercent: Math.max(
        2,
        Math.min(100, ((Math.max(pressureEnd, itemStart) - itemStart + 86400000) / (Math.max(itemEnd, itemStart) - itemStart + 86400000)) * 100),
      ),
      selected: Number.isFinite(selected) && selected >= itemStart && selected <= itemEnd,
    })),
  };
}

export function prefixTrajectoryModel(states = [], selectedBucket = "") {
  const selected = String(selectedBucket || "").slice(0, 10);
  const ordered = states
    .map((state) => ({
      ...state,
      bucket: String(state.timeline_bucket || "").slice(0, 10),
      laneKey: String(state.event_id || `${state.hazard_signature || "event"}:unknown`),
    }))
    .filter((state) => /^\d{4}-\d{2}-\d{2}$/.test(state.bucket))
    .sort((left, right) => left.laneKey.localeCompare(right.laneKey) || left.bucket.localeCompare(right.bucket));
  return ordered.map((state, index) => ({
    ...state,
    selected: state.bucket === selected,
    gapBefore: index > 0
      && state.laneKey === ordered[index - 1].laneKey
      && dateValue(state.bucket) - dateValue(ordered[index - 1].bucket) > 8 * 86400000,
  }));
}

function prefixTrajectoryRibbon(states, selectedBucket) {
  const model = prefixTrajectoryModel(states, selectedBucket);
  const wrapper = document.createElement("div");
  wrapper.className = "prefix-trajectory";
  wrapper.setAttribute("role", "group");
  wrapper.setAttribute("aria-label", "Causal weekly field story states");
  const lanes = groupBy(model, (state) => state.laneKey);
  for (const lane of lanes.values()) {
    const laneElement = document.createElement("div");
    laneElement.className = "trajectory-lane";
    const identity = lane[0];
    laneElement.appendChild(node(
      "span",
      `${identity.hazard_signature || "Event"} · ${shortId(identity.event_id)}`,
      "trajectory-lane-label",
    ));
    const track = document.createElement("div");
    track.className = "prefix-track";
    track.setAttribute("role", "list");
    for (const state of lane) {
      if (state.gapBefore) {
        const gap = node("span", "", "prefix-gap");
        gap.setAttribute("aria-label", "Missing weekly observations");
        track.appendChild(gap);
      }
      const item = document.createElement("span");
      const lifecycle = String(state.event_state || "unknown").toLowerCase().replace(/[^a-z]+/g, "-");
      const archetype = String(state.archetype_display_state || "").toLowerCase();
      const archetypeClass = archetype === "pending_anchor" ? "pending"
        : archetype === "accepted" ? "accepted"
        : archetype === "novel_unassigned" ? "novel"
        : archetype === "calibration_training" ? "calibration"
        : archetype ? "ineligible" : "";
      item.className = `prefix-state state-${lifecycle}${archetypeClass ? ` archetype-${archetypeClass}` : ""}${state.selected ? " is-selected" : ""}${state.right_censored ? " is-open" : ""}`;
      item.setAttribute("role", "listitem");
      item.tabIndex = 0;
      const label = [
        state.bucket, state.event_state, state.current_risk_band || state.max_risk_band,
        state.hazard_signature, state.daily_response_class, state.archetype_display_state,
        state.anchor_status, state.assignment_reason,
        state.right_censored ? "open at this cutoff" : "closed",
      ].filter(Boolean).join(", ");
      item.setAttribute("aria-label", label);
      item.title = label;
      track.appendChild(item);
    }
    const extents = document.createElement("div");
    extents.className = "lifecycle-extents";
    extents.append(node("span", lane[0]?.bucket || ""), node("span", lane.at(-1)?.bucket || ""));
    laneElement.append(track, extents);
    wrapper.appendChild(laneElement);
  }
  wrapper.append(node("span", "One lane per concurrent event · gaps are not interpolated", "truth-note"));
  return wrapper;
}

function lifecycleRibbon(events, selectedBucket) {
  const model = lifecycleModel(events, selectedBucket);
  const wrapper = document.createElement("div");
  wrapper.className = "lifecycle-ribbon";
  wrapper.setAttribute("role", "group");
  wrapper.setAttribute("aria-label", "Retrospective field event lifecycle");
  if (!model.items.length) {
    wrapper.appendChild(node("span", "Lifecycle dates unavailable", "muted"));
    return wrapper;
  }
  const lanes = document.createElement("div");
  lanes.className = "lifecycle-lanes";
  lanes.setAttribute("role", "list");
  for (const item of model.items) {
    const event = item.event;
    const lane = document.createElement("div");
    lane.className = "lifecycle-lane";
    lane.appendChild(node("span", `${event.hazard_signature || "Event"} · ${shortId(event.event_id)}`, "trajectory-lane-label"));
    const track = document.createElement("div");
    track.className = "lifecycle-lane-track";
    const bar = document.createElement("span");
    const rawRisk = String(event.max_risk_band || "none").toLowerCase();
    const risk = { "low-med": "medium", "med-high": "high" }[rawRisk]
      || (["low", "medium", "high", "severe"].includes(rawRisk) ? rawRisk : "none");
    bar.className = `lifecycle-event risk-${risk}${item.selected ? " is-selected" : ""}${event.right_censored ? " is-open" : ""}`;
    bar.style.setProperty("--event-start", `${item.startPercent}%`);
    bar.style.setProperty("--event-width", `${Math.min(item.widthPercent, 100 - item.startPercent)}%`);
    bar.setAttribute("role", "listitem");
    bar.tabIndex = 0;
    const pressure = document.createElement("span");
    pressure.className = "lifecycle-pressure";
    pressure.style.width = `${item.pressurePercent}%`;
    bar.appendChild(pressure);
    const lifecycleEnd = event.event_end_date || (event.right_censored ? event.as_of_date : null) || event.active_end_date || "?";
    const label = [
      `${event.event_start_date || "?"} to ${lifecycleEnd}`,
      `pressure through ${event.active_end_date || "?"}`,
      event.max_risk_band,
      event.hazard_signature,
      event.close_reason,
      event.right_censored ? "open at generation cutoff" : "closed",
    ].filter(Boolean).join(", ");
    bar.setAttribute("aria-label", label);
    bar.title = label;
    track.appendChild(bar);
    lane.appendChild(track);
    lanes.appendChild(lane);
  }
  const extents = document.createElement("div");
  extents.className = "lifecycle-extents";
  extents.append(node("span", shortDate(model.start)), node("span", shortDate(model.end)));
  wrapper.append(lanes, extents, node("span", "Monitored lifecycle outline · solid fill ends with observed pressure · open outlines are dashed", "truth-note"));
  return wrapper;
}

function fieldHeader(properties = {}) {
  const title = node("strong", properties.field_id || "Selected field");
  const story = node("div", readableStory(properties));
  const place = node("div", locationText(properties), "muted");
  const id = document.createElement("code");
  id.textContent = properties.story_cluster_id || properties.motif_id || "No story ID";
  return [title, story, id, place];
}

function readableStory(properties = {}) {
  return [
    properties.motif_family,
    properties.archetype_display_state,
    properties.current_risk_band || properties.max_risk_band,
    properties.hazard_signature,
    properties.response_signature,
  ].filter(Boolean).join(" · ") || "No evidence label";
}

function locationText(properties = {}) {
  return [properties.district, properties.sector, properties.cell, properties.village].filter(Boolean).join(" / ");
}

function node(tag, text, className = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = String(text || "");
  return element;
}

function dateValue(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
  return match ? Date.parse(`${match[1]}-${match[2]}-${match[3]}T00:00:00Z`) : NaN;
}

function shortDate(timestamp) {
  return new Intl.DateTimeFormat(undefined, { month: "short", year: "2-digit", timeZone: "UTC" })
    .format(new Date(timestamp));
}

function shortId(value) {
  const text = String(value || "unknown");
  return text.split(":").at(-1).slice(-8);
}

function groupBy(values, key) {
  const groups = new Map();
  for (const value of values) {
    const id = key(value);
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(value);
  }
  return groups;
}
