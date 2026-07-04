import { ApiClient, isAbortError, isUnsupportedError, withQuery } from "./api.js";
import { computeActivityStats } from "./activity.js";
import { EvolutionController } from "./evolution-controller.js";
import { FilterController } from "./filters.js";
import { FrameDataLoader } from "./frame-data.js";
import {
  FootprintCollectionCache,
  loadFootprintCollection,
} from "./footprint-cache.js";
import { historySupported, loadHistory, shouldLoadHistory, unavailableHistory } from "./history.js";
import { incidentSelectionTransition } from "./incident-story.js";
import {
  adjacentBuckets,
  assertCompleteFootprintCollection,
  footprintLegendEntries,
  incidentDetailModel,
  isCropIncidentV3,
  normalizeFootprintCollection,
  shouldLoadEvolution,
} from "./incident-v3.js";
import { Inspector } from "./inspector.js";
import { MapView } from "./map-view.js";
import { legendEntries } from "./palette.js";
import { TimelineController } from "./timeline.js";
const api = new ApiClient({ cacheLimit: 24 });
const frameData = new FrameDataLoader(api);
const footprintCollections = new FootprintCollectionCache({
  limit: 16,
  maxBytes: 134_217_728,
  normalize: (payload) => assertCompleteFootprintCollection(
    normalizeFootprintCollection(payload),
  ),
});
const inspector = new Inspector();
const EMPTY_FRAME = { type: "FeatureCollection", features: [], meta: {} };
const ui = elementMap([
  "runSummary", "loadingStatus", "loadingStatusText", "colorMode", "showHistory", "legend",
  "affectedCount", "enteringCount", "persistingCount", "inactiveCount", "activityScope",
  "panelToggle", "panelClose", "explorerPanel", "familyColorOption", "historyLabel",
  "evolutionSection", "evolutionSummary", "evolutionStatus", "runMode", "explorerMode",
  "historyControl", "historyLegendNote", "stateLegendNote", "incidentLegendNote",
  "selectionTitle", "mapHelp", "activityTitle", "riskColorOption", "colorModeLabel",
]);
const evolutionController = new EvolutionController(api, {
  section: ui.evolutionSection,
  summary: ui.evolutionSummary,
  status: ui.evolutionStatus,
});
const state = {
  manifest: null, map: null, filters: null, timeline: null, frame: null, trail: null,
  incidentMode: false, footprints: null,
  generation: 0,
  filterGeneration: 0,
  selectionGeneration: 0,
  selectionBucket: "",
  selectedIncidentId: "",
  selectedIncidentDetail: null,
  selectedIncidentContext: null,
  viewportTimer: null,
  lastRequestKey: "",
  evolution: null,
};

boot().catch((error) => fail(error, "The story explorer could not start"));

async function boot() {
  setStatus("Loading story evidence", "loading");
  const [config, manifest, timelineData, motifData] = await Promise.all([
    api.get("/api/config", { channel: "boot-config", cache: true }),
    api.get("/api/manifest", { channel: "boot-manifest", cache: true }),
    api.get("/api/timeline", { channel: "boot-timeline", cache: true }),
    api.get("/api/motifs?limit=500", { channel: "motifs" }),
  ]);
  state.manifest = manifest;
  state.incidentMode = isCropIncidentV3(manifest);
  const activityData = state.incidentMode ? timelineData : await getActivity({});
  if (manifest.server?.optimized_geometry === false) frameData.compactSupported = false;
  state.filters = new FilterController({
    onChange: filtersChanged,
    onSearch: searchStories,
    incidentMode: state.incidentMode,
  });
  state.filters.setData(motifData);
  if (motifData.taxonomy?.source === "hazard_signature_fallback") {
    ui.familyColorOption.textContent = "Hazard family (proxy)";
  }
  state.timeline = new TimelineController({ onChange: () => loadFrame({ force: true }) });
  const initialActivity = activityData?.buckets?.length || activityData?.activity?.length
    ? activityData
    : timelineData;
  state.timeline.setBuckets(timelineData, initialActivity);
  state.map = new MapView({
    container: "map",
    config,
    bounds: manifest.server?.bounds,
    incidentMode: state.incidentMode,
    onReady: () => loadFrame({ force: true }),
    onHover: (properties, point) => inspector.showHover(properties, point),
    onSelect: selectField,
    onSelectIncident: selectIncident,
    onViewportChange: viewportChanged,
  }).mount();
  bindUi();
  configureModeUi();
  updateHistoryAvailability({});
  renderSummary();
  renderLegend();
}

function bindUi() {
  ui.colorMode.addEventListener("change", () => {
    state.map.setColorMode(ui.colorMode.value);
    renderLegend();
  });
  ui.showHistory.addEventListener("change", () => {
    if (state.incidentMode) return;
    const hasFilters = Object.keys(state.filters.filters()).length > 0;
    const visible = historySupported(state.manifest) && hasFilters && ui.showHistory.checked;
    state.map.setHistoryVisible(visible);
    if (!visible) api.abort("trail");
    loadFrame({ force: true });
  });
  ui.panelToggle.addEventListener("click", () => togglePanel(true));
  ui.panelClose.addEventListener("click", () => togglePanel(false));
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      inspector.showHover(null, null);
      togglePanel(false);
    }
  });
}

async function filtersChanged(filters) {
  const generation = ++state.filterGeneration;
  state.generation += 1;
  state.selectionGeneration += 1;
  state.selectionBucket = "";
  state.selectedIncidentId = "";
  state.selectedIncidentDetail = null;
  state.selectedIncidentContext = null;
  state.lastRequestKey = "";
  frameData.api.abortPrefix("frame-data:");
  api.abortPrefix("prefetch:");
  api.abortPrefix("incident-footprints:");
  api.abort("incident-detail");
  api.abort("trail");
  api.abort("field-events");
  api.abort("field-trajectory");
  state.map?.setSelectedField("");
  state.map?.setSelectedIncident("");
  inspector.clear(state.incidentMode
    ? "Select an exact crop-incident footprint to inspect its evidence."
    : "Select a field matching the new filters to inspect its event lanes.");
  state.evolution = null;
  state.map?.setEvolution(null, "");
  state.timeline.stop();
  setStatus(state.incidentMode ? "Updating crop incidents" : "Updating retrospective", "loading");
  if (shouldLoadEvolution(state.manifest)) refreshEvolution(filters, generation);
  try {
    const activity = await getActivity(filters);
    if (generation !== state.filterGeneration) return;
    state.timeline.setActivity(activity);
    const latest = state.timeline.latestActiveIndex();
    if (latest >= 0) state.timeline.select(latest);
    await loadFrame({ force: true });
  } catch (error) {
    if (generation === state.filterGeneration && !isAbortError(error)) {
      fail(error, "Could not apply story filters");
    }
  }
}

async function searchStories(query) {
  try {
    const payload = await api.get(withQuery("/api/motifs", { q: query, limit: 500 }), { channel: "motifs" });
    state.filters.setData(payload);
  } catch (error) {
    if (!isAbortError(error)) fail(error, "Story search failed");
  }
}

async function getActivity(filters, { fallback = null } = {}) {
  const url = withQuery("/api/activity", filters);
  try {
    return await api.get(url, { channel: "activity", cache: true });
  } catch (error) {
    if (isAbortError(error)) throw error;
    if (fallback) return fallback;
    throw error;
  }
}

async function loadFrame({ force = false } = {}) {
  if (!state.map?.ready || !state.timeline) return;
  const bucket = state.timeline.currentBucket();
  if (!bucket) return;
  const selectionTransition = incidentSelectionTransition({
    incidentMode: state.incidentMode,
    selectedIncidentId: state.selectedIncidentId,
    selectionBucket: state.selectionBucket,
    nextBucket: bucket.timeline_bucket,
  });
  if (selectionTransition.changed) {
    if (selectionTransition.preserveIncident) {
      state.selectionBucket = selectionTransition.nextSelectionBucket;
      state.selectedIncidentContext = null;
      api.abort("field-events");
      api.abort("field-trajectory");
      state.map.setSelectedField("");
    } else {
      state.selectionGeneration += 1;
      state.selectionBucket = selectionTransition.nextSelectionBucket;
      api.abort("incident-detail");
      api.abort("field-events");
      api.abort("field-trajectory");
      state.map.setSelectedField("");
      state.map.setSelectedIncident("");
      inspector.clear(state.incidentMode
        ? "Select a crop-incident footprint to inspect this week’s evidence."
        : "Select a field to inspect this week’s concurrent event lanes.");
    }
  }
  const filters = state.filters.filters();
  updateHistoryAvailability(filters);
  const bbox = state.map.boundsString();
  const wantsFieldFrame = !state.incidentMode || state.map.incidentFieldsVisible();
  const query = { ...filters, bbox };
  const frameUrl = withQuery(`/api/frame/${encodeURIComponent(bucket.timeline_bucket)}`, query);
  const trailUrl = withQuery("/api/trail", { ...query, bucket: bucket.timeline_bucket, lookback: 5 });
  const footprintUrl = withQuery(
    `/api/incident-footprints/${encodeURIComponent(bucket.timeline_bucket)}`,
    filters,
  );
  const wantsHistory = !state.incidentMode
    && shouldLoadHistory(state.manifest, filters, ui.showHistory.checked);
  const requestKey = state.incidentMode
    ? `${wantsFieldFrame ? frameUrl : "overview-no-fields"}|footprints:${footprintUrl}`
    : `${frameUrl}|history:${wantsHistory}`;
  if (!force && requestKey === state.lastRequestKey) return;
  state.lastRequestKey = requestKey;
  const generation = ++state.generation;
  const requestedIndex = state.timeline.pendingIndex;
  setStatus(`Loading ${bucket.timeline_bucket}`, "loading");
  try {
    const [frame, trail, footprintPayload] = await Promise.all([
      wantsFieldFrame
        ? frameData.load({ bucket: bucket.timeline_bucket, query, legacyUrl: frameUrl })
        : Promise.resolve({
            ...EMPTY_FRAME,
            meta: { timeline_bucket: bucket.timeline_bucket, overview_only: true },
          }),
      wantsHistory
        ? loadHistory(state.manifest, () => api.get(trailUrl, { channel: "trail", cache: true }))
        : Promise.resolve(unavailableHistory()),
      state.incidentMode
        ? loadFootprintCollection(
            api,
            footprintCollections,
            footprintUrl,
            { channel: "incident-footprints:current" },
          )
        : Promise.resolve(null),
    ]);
    if (generation !== state.generation || requestedIndex !== state.timeline.pendingIndex) return;
    const footprints = state.incidentMode ? footprintPayload : null;
    if (state.incidentMode) state.filters.setIncidentFeatures(footprints.features);
    state.frame = frame;
    state.trail = trail;
    state.footprints = footprints;
    state.map.setData(
      frame,
      trail,
      state.incidentMode ? null : state.evolution,
      bucket.timeline_bucket,
      footprints,
    );
    refreshSelectedIncidentContext();
    renderActivityStats(frame, trail);
    if (shouldLoadEvolution(state.manifest)) {
      evolutionController.render(filters, state.evolution, bucket.timeline_bucket);
      refreshEvolution(filters, state.filterGeneration);
    }
    renderLegend();
    const count = Number(frame.meta?.feature_count ?? frame.features?.length ?? 0);
    const stories = Number(frame.meta?.motif_count ?? frame.meta?.story_cluster_count ?? 0);
    const footprintCount = Number(
      footprints?.meta?.feature_count ?? footprints?.features?.length ?? 0,
    );
    const capNotes = [];
    if (frame.meta?.truncated) capNotes.push("fields capped");
    if (!state.incidentMode && trail.meta?.truncated) capNotes.push("history capped");
    const capNote = capNotes.length ? ` · ${capNotes.join(" + ")} to protect rendering` : "";
    state.timeline.commit(
      requestedIndex,
      state.incidentMode
        ? `${footprintCount.toLocaleString()} complete incident footprints · ${count.toLocaleString()} drilldown fields${capNote}`
        : `${count.toLocaleString()} visible fields · ${stories.toLocaleString()} ${state.manifest.run?.diagnostic_preview ? "diagnostic archetype states" : state.manifest.run?.motif_count ? "motifs" : "exact signatures"}${capNote}`,
    );
    const visibleCount = state.incidentMode ? footprintCount : count;
    setStatus(visibleCount ? "Map is current" : "No matching evidence", "ready");
    prefetchAdjacent(filters, bbox, wantsFieldFrame);
  } catch (error) {
    if (generation !== state.generation || isAbortError(error)) return;
    state.lastRequestKey = "";
    fail(error, "Could not load this week");
  }
}

function updateHistoryAvailability(filters) {
  if (state.incidentMode) {
    ui.showHistory.checked = false;
    ui.showHistory.disabled = true;
    state.map.setHistoryVisible(false);
    return;
  }
  const supported = historySupported(state.manifest);
  const available = supported && Object.keys(filters).length > 0;
  ui.showHistory.disabled = !available;
  ui.historyLabel.textContent = !supported
    ? "History unavailable for raw geometry"
    : available ? "Recent history" : "Filter to compare weeks";
  state.map.setHistoryVisible(available && ui.showHistory.checked);
}

function viewportChanged() {
  window.clearTimeout(state.viewportTimer);
  state.viewportTimer = window.setTimeout(() => loadFrame(), 180);
}

async function selectIncident(properties) {
  const incidentId = String(properties?.incident_id || properties?.story_cluster_id || "");
  if (!incidentId) return;
  api.abort("field-events");
  api.abort("field-trajectory");
  state.map.setSelectedField("");
  state.map.setSelectedIncident(incidentId);
  state.selectedIncidentId = incidentId;
  state.selectedIncidentDetail = null;
  state.selectedIncidentContext = null;
  state.selectionBucket = String(properties.timeline_bucket || "");
  const selectionGeneration = ++state.selectionGeneration;
  inspector.loadingIncident(properties);
  togglePanel(true);
  try {
    const payload = await api.get(`/api/incident/${encodeURIComponent(incidentId)}`, {
      channel: "incident-detail",
      cache: true,
    });
    if (selectionGeneration !== state.selectionGeneration) return;
    state.selectedIncidentDetail = payload;
    refreshSelectedIncidentContext();
  } catch (error) {
    if (!isAbortError(error)) inspector.showIncidentError(properties, error);
  }
}

function refreshSelectedIncidentContext() {
  if (!state.selectedIncidentId || !state.selectedIncidentDetail) return;
  const bucket = String(
    state.timeline?.currentBucket()?.timeline_bucket || state.selectionBucket || "",
  ).slice(0, 10);
  const contextKey = `${state.selectedIncidentId}:${bucket}`;
  let context = state.selectedIncidentContext;
  if (!context || context.key !== contextKey) {
    const currentFeature = (state.footprints?.features || []).find((feature) => (
      String(feature.properties?.incident_id || "") === state.selectedIncidentId
    ));
    const properties = currentFeature?.properties || {
      incident_id: state.selectedIncidentId,
      timeline_bucket: bucket,
    };
    const model = incidentDetailModel(state.selectedIncidentDetail, properties);
    context = { key: contextKey, model, properties };
    state.selectedIncidentContext = context;
  }
  state.map?.setSelectedIncidentStory(
    context.model.footprintHistory,
    context.model.currentFootprint,
  );
  inspector.showIncident({
    ...context.properties,
    incident_id: context.model.incidentId,
    crop_name: context.model.crop,
    hazard_family: context.model.hazard,
    incident_state: context.model.lifecycle,
    timeline_bucket: bucket,
  }, state.selectedIncidentDetail);
}

async function selectField(properties) {
  if (!properties?.field_id) return;
  api.abort("incident-detail");
  state.map.setSelectedField(properties.field_id);
  if (!state.incidentMode) state.map.setSelectedIncident("");
  state.selectionBucket = String(properties.timeline_bucket || "");
  const selectionGeneration = ++state.selectionGeneration;
  const selectionBucket = state.selectionBucket;
  inspector.loading(properties);
  togglePanel(true);
  try {
    const fieldId = encodeURIComponent(properties.field_id);
    const [eventsResult, trajectoryResult] = await Promise.allSettled([
      api.get(`/api/field/${fieldId}/events?limit=12`, { channel: "field-events" }),
      api.get(`/api/field/${fieldId}/trajectory?limit=250`, { channel: "field-trajectory" })
        .catch((error) => isUnsupportedError(error) ? { states: [] } : Promise.reject(error)),
    ]);
    if (
      selectionGeneration !== state.selectionGeneration
      || selectionBucket !== state.selectionBucket
    ) return;
    if (eventsResult.status === "rejected" && trajectoryResult.status === "rejected") {
      throw eventsResult.reason;
    }
    const payload = eventsResult.status === "fulfilled" ? eventsResult.value : { events: [] };
    const trajectory = trajectoryResult.status === "fulfilled" ? trajectoryResult.value : { states: [] };
    inspector.showSelection(properties, payload.events || [], trajectory.states || [], {
      eventsError: eventsResult.status === "rejected" ? eventsResult.reason : null,
      trajectoryError: trajectoryResult.status === "rejected" ? trajectoryResult.reason : null,
    });
  } catch (error) {
    if (!isAbortError(error)) inspector.showError(properties, error);
  }
}

async function refreshEvolution(filters, filterGeneration) {
  if (!shouldLoadEvolution(state.manifest)) return;
  try {
    const pending = evolutionController.load(filters);
    evolutionController.render(filters, state.evolution, state.timeline?.currentBucket()?.timeline_bucket);
    const payload = await pending;
    if (filterGeneration !== state.filterGeneration) return;
    state.evolution = payload;
    const bucket = state.timeline?.currentBucket()?.timeline_bucket || "";
    state.map?.setEvolution(payload, bucket);
    evolutionController.render(filters, payload, bucket);
  } catch (error) {
    if (!isAbortError(error)) console.warn("Could not load aggregate evolution", error);
  }
}

function renderActivityStats(frame, trail) {
  if (state.incidentMode && !state.map?.incidentFieldsVisible()) {
    for (const element of [
      ui.affectedCount, ui.enteringCount, ui.persistingCount, ui.inactiveCount,
    ]) element.textContent = "—";
    ui.activityScope.textContent = "Zoom in for field counts";
    return;
  }
  const stats = computeActivityStats(frame, trail);
  ui.affectedCount.textContent = formatCount(stats.affected);
  ui.enteringCount.textContent = formatCount(stats.entering);
  ui.persistingCount.textContent = formatCount(stats.persisting);
  ui.inactiveCount.textContent = formatCount(stats.inactive);
  ui.activityScope.textContent = frame.meta?.bbox || frame.meta?.bbox_applied
    ? "Current viewport"
    : "All mapped fields";
}

function renderLegend() {
  const familyEntries = legendEntries(
    state.incidentMode ? "family" : ui.colorMode.value,
    state.incidentMode ? state.footprints?.features || [] : state.frame?.features || [],
  );
  const entries = state.incidentMode
    ? [...familyEntries, ...footprintLegendEntries().map((entry) => ({ ...entry, stateStyle: true }))]
    : familyEntries;
  ui.legend.replaceChildren(...entries.map((entry) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.className = entry.stateStyle
      ? `legend-swatch footprint-swatch is-${entry.key}`
      : "legend-swatch";
    if (entry.color) swatch.style.background = entry.color;
    item.append(swatch, document.createTextNode(entry.label));
    return item;
  }));
}

function renderSummary() {
  const run = state.manifest.run || {};
  const geometry = state.manifest.map_geometry || {};
  if (state.incidentMode) {
    const incidentCount = Number(
      run.incident_count || state.manifest.validation?.incident_count || 0,
    );
    ui.runSummary.textContent = [
      incidentCount ? `${incidentCount.toLocaleString()} crop-impact incidents` : "Crop-impact incident monitoring",
      "exact complete weekly footprints",
      "field evidence on zoom",
    ].join(" · ");
    ui.runMode.textContent = run.map_publication_approved
      ? "Crop incidents V3"
      : "Crop incidents V3 · diagnostic";
    ui.runMode.classList.toggle("is-causal", true);
    ui.runMode.classList.toggle("is-diagnostic", !run.map_publication_approved);
    ui.explorerMode.textContent = "Crop-impact incident monitor";
    return;
  }
  const diagnosticPreview = run.diagnostic_preview === true;
  const motifCount = Number(run.motif_count || 0);
  const archetypeCount = Number(run.archetype_count || motifCount);
  const taxonomySummary = diagnosticPreview
    ? `${archetypeCount.toLocaleString()} event archetypes (diagnostic)`
    : motifCount
    ? `${motifCount.toLocaleString()} discovered motifs (unreviewed)`
    : `${Number(run.story_cluster_count || 0).toLocaleString()} exact signatures`;
  ui.runSummary.textContent = [
    `${Number(run.event_count || 0).toLocaleString()} event windows`,
    taxonomySummary,
    `${Number(geometry.mappable_event_field_count || geometry.mappable_selected_field_count || 0).toLocaleString()} mapped fields`,
  ].join(" · ");
  const causalSnapshot = Boolean(run.generation_id);
  ui.runMode.textContent = diagnosticPreview ? "Diagnostic preview · not approved" : causalSnapshot ? "Causal snapshot" : "Retrospective";
  ui.runMode.classList.toggle("is-causal", causalSnapshot);
  ui.runMode.classList.toggle("is-diagnostic", diagnosticPreview);
  ui.explorerMode.textContent = diagnosticPreview ? "Unreviewed archetype preview" : causalSnapshot ? "As-of story snapshot" : "Retrospective explorer";
}

function prefetchAdjacent(filters, bbox, wantsFieldFrame = true) {
  const targets = state.incidentMode
    ? adjacentBuckets(state.timeline.buckets, state.timeline.pendingIndex)
    : [state.timeline.buckets[state.timeline.pendingIndex + 1]].filter(Boolean);
  const query = { ...filters, bbox };
  for (const target of targets) {
    if (wantsFieldFrame) {
      frameData.preload({
        bucket: target.timeline_bucket,
        query,
        legacyUrl: withQuery(`/api/frame/${encodeURIComponent(target.timeline_bucket)}`, query),
      });
    }
    if (state.incidentMode) {
      const footprintUrl = withQuery(
        `/api/incident-footprints/${encodeURIComponent(target.timeline_bucket)}`,
        filters,
      );
      loadFootprintCollection(
        api,
        footprintCollections,
        footprintUrl,
        { channel: `incident-footprints:prefetch:${target.timeline_bucket}` },
      ).catch(() => {});
    }
  }
}

function configureModeUi() {
  if (!state.incidentMode) return;
  ui.historyControl.hidden = true;
  ui.historyLegendNote.hidden = true;
  ui.stateLegendNote.hidden = true;
  ui.incidentLegendNote.hidden = false;
  ui.evolutionSection.hidden = true;
  ui.selectionTitle.textContent = "Crop-incident evidence";
  ui.activityTitle.textContent = "Visible field evidence";
  ui.familyColorOption.textContent = "Exposure family";
  ui.riskColorOption.textContent = "Field risk (zoomed detail)";
  ui.colorModeLabel.textContent = "Color overview / fields";
  ui.mapHelp.textContent = "Exact complete crop-incident footprints are the primary overview. Fields appear after zooming in for evidence drilldown. Footprint evolution, not physical movement.";
  inspector.clear("Select an exact crop-incident footprint to inspect its evidence.");
}

function setStatus(message, kind) {
  ui.loadingStatus.classList.toggle("is-loading", kind === "loading");
  ui.loadingStatus.classList.toggle("is-error", kind === "error");
  ui.loadingStatusText.textContent = message;
}

function fail(error, context) {
  console.error(context, error);
  setStatus(`${context}: ${error.message}`, "error");
}

function togglePanel(open) {
  ui.explorerPanel.classList.toggle("is-open", open);
  ui.panelToggle.setAttribute("aria-expanded", String(open));
}
function formatCount(value) { return value === null || value === undefined ? "—" : Number(value).toLocaleString(); }
function elementMap(ids) { return Object.fromEntries(ids.map((id) => [id, document.getElementById(id)])); }
