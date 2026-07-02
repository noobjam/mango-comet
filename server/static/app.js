import { ApiClient, isAbortError, isUnsupportedError, withQuery } from "./api.js";
import { computeActivityStats } from "./activity.js";
import { EvolutionController } from "./evolution-controller.js";
import { FilterController } from "./filters.js";
import { FrameDataLoader } from "./frame-data.js";
import { historySupported, loadHistory, shouldLoadHistory, unavailableHistory } from "./history.js";
import { Inspector } from "./inspector.js";
import { MapView } from "./map-view.js";
import { legendEntries } from "./palette.js";
import { TimelineController } from "./timeline.js";
const api = new ApiClient({ cacheLimit: 24 });
const frameData = new FrameDataLoader(api);
const inspector = new Inspector();
const ui = elementMap([
  "runSummary", "loadingStatus", "loadingStatusText", "colorMode", "showHistory", "legend",
  "affectedCount", "enteringCount", "persistingCount", "inactiveCount", "activityScope",
  "panelToggle", "panelClose", "explorerPanel", "familyColorOption", "historyLabel",
  "evolutionSection", "evolutionSummary", "evolutionStatus", "runMode", "explorerMode",
]);
const evolutionController = new EvolutionController(api, {
  section: ui.evolutionSection,
  summary: ui.evolutionSummary,
  status: ui.evolutionStatus,
});
const state = {
  manifest: null, map: null, filters: null, timeline: null, frame: null, trail: null,
  generation: 0,
  filterGeneration: 0,
  selectionGeneration: 0,
  selectionBucket: "",
  viewportTimer: null,
  lastRequestKey: "",
  evolution: null,
};

boot().catch((error) => fail(error, "The story explorer could not start"));

async function boot() {
  setStatus("Loading story evidence", "loading");
  const [config, manifest, timelineData, motifData, activityData] = await Promise.all([
    api.get("/api/config", { channel: "boot-config", cache: true }),
    api.get("/api/manifest", { channel: "boot-manifest", cache: true }),
    api.get("/api/timeline", { channel: "boot-timeline", cache: true }),
    api.get("/api/motifs?limit=500", { channel: "motifs" }),
    getActivity({}),
  ]);
  state.manifest = manifest;
  if (manifest.server?.optimized_geometry === false) frameData.compactSupported = false;
  state.filters = new FilterController({ onChange: filtersChanged, onSearch: searchStories });
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
    onReady: () => loadFrame({ force: true }),
    onHover: (properties, point) => inspector.showHover(properties, point),
    onSelect: selectField,
    onViewportChange: viewportChanged,
  }).mount();
  bindUi();
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
  state.lastRequestKey = "";
  frameData.api.abortPrefix("frame-data:");
  api.abort("trail");
  api.abort("field-events");
  api.abort("field-trajectory");
  state.map?.setSelectedField("");
  inspector.clear("Select a field matching the new filters to inspect its event lanes.");
  state.evolution = null;
  state.map?.setEvolution(null, "");
  state.timeline.stop();
  setStatus("Updating retrospective", "loading");
  refreshEvolution(filters, generation);
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
  const filters = state.filters.filters();
  updateHistoryAvailability(filters);
  const bbox = state.map.boundsString();
  const query = { ...filters, bbox };
  const frameUrl = withQuery(`/api/frame/${encodeURIComponent(bucket.timeline_bucket)}`, query);
  const trailUrl = withQuery("/api/trail", { ...query, bucket: bucket.timeline_bucket, lookback: 5 });
  const wantsHistory = shouldLoadHistory(state.manifest, filters, ui.showHistory.checked);
  const requestKey = `${frameUrl}|history:${wantsHistory}`;
  if (!force && requestKey === state.lastRequestKey) return;
  state.lastRequestKey = requestKey;
  const generation = ++state.generation;
  const requestedIndex = state.timeline.pendingIndex;
  setStatus(`Loading ${bucket.timeline_bucket}`, "loading");
  try {
    const [frame, trail] = await Promise.all([
      frameData.load({ bucket: bucket.timeline_bucket, query, legacyUrl: frameUrl }),
      wantsHistory
        ? loadHistory(state.manifest, () => api.get(trailUrl, { channel: "trail", cache: true }))
        : Promise.resolve(unavailableHistory()),
    ]);
    if (generation !== state.generation || requestedIndex !== state.timeline.pendingIndex) return;
    state.frame = frame;
    state.trail = trail;
    state.map.setData(frame, trail, state.evolution, bucket.timeline_bucket);
    renderActivityStats(frame, trail);
    evolutionController.render(filters, state.evolution, bucket.timeline_bucket);
    refreshEvolution(filters, state.filterGeneration);
    renderLegend();
    const count = Number(frame.meta?.feature_count ?? frame.features?.length ?? 0);
    const stories = Number(frame.meta?.motif_count ?? frame.meta?.story_cluster_count ?? 0);
    const capNotes = [];
    if (frame.meta?.truncated) capNotes.push("fields capped");
    if (trail.meta?.truncated) capNotes.push("history capped");
    const capNote = capNotes.length ? ` · ${capNotes.join(" + ")} to protect rendering` : "";
    state.timeline.commit(
      requestedIndex,
      `${count.toLocaleString()} visible fields · ${stories.toLocaleString()} ${state.manifest.run?.motif_count ? "motifs" : "exact signatures"}${capNote}`
    );
    if (state.selectionBucket && state.selectionBucket !== bucket.timeline_bucket) {
      state.selectionGeneration += 1;
      state.selectionBucket = "";
      state.map.setSelectedField("");
      inspector.clear("Select a field to inspect this week’s concurrent event lanes.");
    }
    setStatus(count ? "Map is current" : "No matching fields", "ready");
    prefetchAdjacent(filters, bbox);
  } catch (error) {
    if (generation !== state.generation || isAbortError(error)) return;
    state.lastRequestKey = "";
    fail(error, "Could not load this week");
  }
}

function updateHistoryAvailability(filters) {
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

async function selectField(properties) {
  if (!properties?.field_id) return;
  state.map.setSelectedField(properties.field_id);
  state.selectionBucket = String(properties.timeline_bucket || "");
  const selectionGeneration = ++state.selectionGeneration;
  inspector.loading(properties);
  togglePanel(true);
  try {
    const fieldId = encodeURIComponent(properties.field_id);
    const [eventsResult, trajectoryResult] = await Promise.allSettled([
      api.get(`/api/field/${fieldId}/events?limit=12`, { channel: "field-events" }),
      api.get(`/api/field/${fieldId}/trajectory?limit=250`, { channel: "field-trajectory" })
        .catch((error) => isUnsupportedError(error) ? { states: [] } : Promise.reject(error)),
    ]);
    if (selectionGeneration !== state.selectionGeneration) return;
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
  const entries = legendEntries(ui.colorMode.value, state.frame?.features || []);
  ui.legend.replaceChildren(...entries.map((entry) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.className = "legend-swatch";
    swatch.style.background = entry.color;
    item.append(swatch, document.createTextNode(entry.label));
    return item;
  }));
}

function renderSummary() {
  const run = state.manifest.run || {};
  const geometry = state.manifest.map_geometry || {};
  const motifCount = Number(run.motif_count || 0);
  const taxonomySummary = motifCount
    ? `${motifCount.toLocaleString()} discovered motifs (unreviewed)`
    : `${Number(run.story_cluster_count || 0).toLocaleString()} exact signatures`;
  ui.runSummary.textContent = [
    `${Number(run.event_count || 0).toLocaleString()} event windows`,
    taxonomySummary,
    `${Number(geometry.mappable_event_field_count || geometry.mappable_selected_field_count || 0).toLocaleString()} mapped fields`,
  ].join(" · ");
  const causalSnapshot = Boolean(run.generation_id);
  ui.runMode.textContent = causalSnapshot ? "Causal snapshot" : "Retrospective";
  ui.runMode.classList.toggle("is-causal", causalSnapshot);
  ui.explorerMode.textContent = causalSnapshot ? "As-of story snapshot" : "Retrospective explorer";
}

function prefetchAdjacent(filters, bbox) {
  const next = state.timeline.buckets[state.timeline.pendingIndex + 1];
  if (!next) return;
  const query = { ...filters, bbox };
  frameData.preload({
    bucket: next.timeline_bucket,
    query,
    legacyUrl: withQuery(`/api/frame/${encodeURIComponent(next.timeline_bucket)}`, query),
  });
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
