const FACETS = [
  ["familyFilter", "motif_family", "All families"],
  ["riskFilter", "current_risk_band", "All current risks"],
  ["hazardFilter", "hazard_signature", "All hazards"],
  ["responseFilter", "response_signature", "All responses"],
];
const INCIDENT_FACETS = [
  ["cropFilter", "crop_name", "All crops"],
  ["stageFilter", "stage_bucket", "All crop stages"],
  ["lifecycleFilter", "incident_state", "All lifecycle states"],
];

export class FilterController {
  constructor({ onChange, onSearch, incidentMode = false }) {
    this.onChange = onChange;
    this.onSearch = onSearch;
    this.incidentMode = Boolean(incidentMode);
    this.mode = this.incidentMode ? "incident" : "situation";
    this.exactStories = [];
    this.incidentFacetValues = new Map(INCIDENT_FACETS.map(([, key]) => [key, new Set()]));
    this.searchTimer = null;
    this.elements = Object.fromEntries([
      "situationMode", "exactMode", "situationFilters", "exactFilters", "storySearch",
      "storySelect", "familyFilter", "riskFilter", "hazardFilter", "responseFilter",
      "clearFilters", "storySummary", "familyFilterLabel", "modeSwitch",
      "incidentFilters", "cropFilter", "stageFilter", "lifecycleFilter",
    ].map((id) => [id, document.getElementById(id)]));
    this.bind();
  }

  bind() {
    const e = this.elements;
    e.situationMode.addEventListener("click", () => this.setMode("situation"));
    e.exactMode.addEventListener("click", () => this.setMode("exact"));
    e.storySelect.addEventListener("change", () => this.changed());
    for (const [id] of FACETS) e[id].addEventListener("change", () => this.changed());
    for (const [id] of INCIDENT_FACETS) e[id].addEventListener("change", () => this.changed());
    e.storySearch.addEventListener("input", () => {
      window.clearTimeout(this.searchTimer);
      this.searchTimer = window.setTimeout(() => this.onSearch?.(e.storySearch.value.trim()), 260);
    });
    e.clearFilters.addEventListener("click", () => this.clear());
    this.setIncidentMode(this.incidentMode);
  }

  setIncidentMode(enabled) {
    this.incidentMode = Boolean(enabled);
    if (this.incidentMode) this.mode = "incident";
    this.elements.modeSwitch.hidden = this.incidentMode;
    this.elements.incidentFilters.hidden = !this.incidentMode;
    this.elements.situationFilters.hidden = this.incidentMode || this.mode === "exact";
    this.elements.exactFilters.hidden = this.incidentMode || this.mode !== "exact";
  }

  setData(payload = {}) {
    const selected = this.elements.storySelect.value;
    const prior = this.selectedStory();
    const candidates = payload.exact_stories?.length
      ? payload.exact_stories
      : (payload.motifs || []).filter((row) => row.story_cluster_id);
    this.exactStories = candidates.slice();
    this.elements.familyFilterLabel.textContent = payload.taxonomy?.source === "hazard_signature_fallback"
      ? "Hazard family (proxy)"
      : "Motif family";
    if (prior && !this.exactStories.some((row) => row.story_cluster_id === prior.story_cluster_id)) {
      this.exactStories.unshift(prior);
    }
    this.fillStories(selected);
    for (const [id, key, emptyLabel] of FACETS) {
      this.fillFacet(this.elements[id], payload.facets?.[key] || [], key, emptyLabel);
    }
    const rows = payload.incidents || payload.exact_stories || payload.motifs || [];
    for (const [id, key, emptyLabel] of INCIDENT_FACETS) {
      const values = payload.facets?.[key] || derivedFacet(rows, key);
      this.rememberIncidentFacet(key, values);
      this.fillFacet(
        this.elements[id],
        [...this.incidentFacetValues.get(key)].sort(),
        key,
        emptyLabel,
      );
    }
    this.renderSummary();
  }

  setIncidentFeatures(features = []) {
    if (!this.incidentMode) return;
    const rows = features.map((feature) => feature?.properties || {});
    for (const [id, key, emptyLabel] of INCIDENT_FACETS) {
      this.rememberIncidentFacet(key, derivedFacet(rows, key));
      this.fillFacet(
        this.elements[id],
        [...this.incidentFacetValues.get(key)].sort(),
        key,
        emptyLabel,
      );
    }
    this.renderSummary();
  }

  rememberIncidentFacet(key, values) {
    const target = this.incidentFacetValues.get(key);
    for (const row of values || []) {
      const value = typeof row === "string" ? row : row?.[key] ?? row?.value;
      if (value) target.add(String(value));
    }
  }

  fillStories(selected) {
    const select = this.elements.storySelect;
    select.replaceChildren(new Option("Choose a story", ""));
    for (const story of this.exactStories) {
      const id = String(story.story_cluster_id || "");
      if (!id) continue;
      const shortId = id.split(":").at(-1).slice(0, 7);
      const label = `${story.short_label || readableStory(story) || "Exact story"} · ${shortId}`;
      const count = Number(story.event_count || 0);
      select.appendChild(new Option(count ? `${label} · ${count.toLocaleString()} events` : label, id));
    }
    select.value = Array.from(select.options).some((option) => option.value === selected) ? selected : "";
  }

  fillFacet(select, rows, key, emptyLabel) {
    const selected = select.value;
    select.replaceChildren(new Option(emptyLabel, ""));
    for (const row of rows) {
      const value = typeof row === "string" ? row : row[key] ?? row.value;
      if (!value) continue;
      const count = typeof row === "object" ? Number(row.event_count || row.field_count || row.count || 0) : 0;
      const label = prettyValue(value);
      select.appendChild(new Option(count ? `${label} · ${count.toLocaleString()}` : label, String(value)));
    }
    if (selected && !Array.from(select.options).some((option) => option.value === selected)) {
      select.appendChild(new Option(prettyValue(selected), selected));
    }
    select.value = Array.from(select.options).some((option) => option.value === selected) ? selected : "";
  }

  setMode(mode, { notify = true } = {}) {
    if (this.incidentMode) return;
    this.mode = mode === "exact" ? "exact" : "situation";
    const exact = this.mode === "exact";
    if (exact) this.clearFacets();
    else this.elements.storySelect.value = "";
    this.elements.exactMode.classList.toggle("is-active", exact);
    this.elements.situationMode.classList.toggle("is-active", !exact);
    this.elements.exactMode.setAttribute("aria-pressed", String(exact));
    this.elements.situationMode.setAttribute("aria-pressed", String(!exact));
    this.elements.exactFilters.hidden = !exact;
    this.elements.situationFilters.hidden = exact;
    this.renderSummary();
    if (notify) this.onChange?.(this.filters());
  }

  clear() {
    this.elements.storySearch.value = "";
    this.elements.storySelect.value = "";
    this.clearFacets();
    this.renderSummary();
    if (!this.incidentMode) this.onSearch?.("");
    this.onChange?.(this.filters());
  }

  clearFacets() {
    for (const [id] of FACETS) this.elements[id].value = "";
    for (const [id] of INCIDENT_FACETS) this.elements[id].value = "";
  }

  changed() {
    this.renderSummary();
    this.onChange?.(this.filters());
  }

  filters() {
    if (this.incidentMode) {
      const result = {};
      for (const [id, key] of INCIDENT_FACETS) {
        if (this.elements[id].value) result[key] = this.elements[id].value;
      }
      return result;
    }
    if (this.mode === "exact") {
      const storyId = this.elements.storySelect.value;
      return storyId ? { story_cluster_id: storyId } : {};
    }
    const result = {};
    for (const [id, key] of FACETS) {
      if (this.elements[id].value) result[key] = this.elements[id].value;
    }
    return result;
  }

  selectedStory() {
    const id = this.elements.storySelect.value;
    return this.exactStories.find((story) => String(story.story_cluster_id) === id) || null;
  }

  renderSummary() {
    const card = this.elements.storySummary;
    const story = this.mode === "exact" ? this.selectedStory() : null;
    const filters = this.filters();
    card.replaceChildren();
    const kicker = element(
      "div",
      "story-card-kicker",
      this.incidentMode ? "Crop-impact incidents" : this.mode === "exact" ? "Exact story" : "Similar situations",
    );
    let title = "Every story state in the selected week";
    let text = "Choose shared evidence or an exact story to narrow the retrospective.";
    const chips = [];
    if (this.incidentMode) {
      title = Object.keys(filters).length
        ? Object.values(filters).map(prettyValue).join(" · ")
        : "All crop incidents in the selected week";
      text = "Exact monitored footprints are primary; zoom in for contributing field evidence.";
    } else if (this.mode === "exact" && !story) {
      title = "Choose one strict temporal signature";
      text = "Exact-story mode preserves sequence identity instead of broadening it to similar evidence.";
    } else if (story) {
      title = story.short_label || readableStory(story) || story.story_cluster_id;
      text = `${Number(story.event_count || 0).toLocaleString()} events · ${Number(story.field_count || 0).toLocaleString()} fields. Exact identity is preserved.`;
      chips.push(story.stage_signature, story.response_signature);
    } else if (Object.keys(filters).length) {
      title = Object.values(filters).join(" · ");
      text = "An aggregate of stories sharing this evidence; it is not one moving object or one exact sequence.";
    }
    card.append(kicker, element("h3", "", title), element("p", "", text));
    const values = [...chips, ...Object.values(filters)].filter(Boolean).slice(0, 5);
    if (values.length) {
      const row = element("div", "chip-row", "");
      for (const value of values) row.appendChild(element("span", "chip", value));
      card.appendChild(row);
    }
  }
}

function derivedFacet(rows, key) {
  return [...new Set(rows.map((row) => (
    key === "incident_state" ? row?.incident_state || row?.terminal_state : row?.[key]
  )).filter(Boolean))]
    .sort((left, right) => String(left).localeCompare(String(right)));
}

function readableStory(story) {
  return [story.max_risk_band, story.hazard_signature, story.response_signature]
    .filter(Boolean)
    .map(prettyValue)
    .join(" · ");
}

function prettyValue(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replaceAll("+", " + ")
    .replace(/\s+/g, " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = String(text || "");
  return node;
}
