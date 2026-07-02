import { isAbortError, isUnsupportedError, withQuery } from "./api.js";
import { buildEvolutionModel, evolutionSummary } from "./map-evolution.js";

export class EvolutionController {
  constructor(api, { section, summary, status }) {
    this.api = api;
    this.section = section;
    this.summary = summary;
    this.status = status;
    this.supported = null;
    this.desired = null;
    this.loadedKey = "";
    this.payload = null;
    this.error = null;
    this.inflight = null;
  }

  async load(filters) {
    const entries = Object.entries(filters || {}).sort(([left], [right]) => left.localeCompare(right));
    const key = JSON.stringify(entries);
    if (!entries.length || this.supported === false) {
      this.desired = null;
      this.loadedKey = "";
      this.payload = null;
      this.error = null;
      return null;
    }
    this.desired = { key, filters: Object.fromEntries(entries) };
    if (this.loadedKey === key) return this.payload;
    if (!this.inflight) {
      this.inflight = this.drain().finally(() => { this.inflight = null; });
    }
    await this.inflight;
    return this.loadedKey === key ? this.payload : null;
  }

  async drain() {
    while (this.desired && this.desired.key !== this.loadedKey && this.supported !== false) {
      const request = this.desired;
      let payload = null;
      let error = null;
      try {
        payload = await this.api.get(withQuery("/api/evolution", request.filters), {
          channel: "evolution:serialized",
          cache: true,
        });
        this.supported = true;
      } catch (caught) {
        if (isAbortError(caught)) continue;
        if (isUnsupportedError(caught)) this.supported = false;
        else {
          error = caught;
          console.warn("Activity-center evolution is temporarily unavailable", caught);
        }
      }
      if (this.desired?.key === request.key) {
        this.loadedKey = request.key;
        this.payload = payload;
        this.error = error;
      }
    }
  }

  render(filters, payload, bucket) {
    const selected = Object.keys(filters).length > 0;
    this.section.hidden = !selected;
    if (!selected) return;
    if (this.supported === false) {
      this.status.textContent = "Unavailable";
      this.summary.textContent = "This run does not publish compact activity-center and overlap summaries yet.";
      return;
    }
    if (this.error) {
      this.status.textContent = "Query error";
      this.summary.textContent = "The aggregate evolution query failed; the current field map is still valid.";
      return;
    }
    if (this.desired && this.desired.key !== this.loadedKey) {
      this.status.textContent = "Loading";
      this.summary.textContent = "Computing the filtered aggregate across all mapped fields…";
      return;
    }
    const model = buildEvolutionModel(payload, bucket);
    this.status.textContent = model.current ? "Aggregate" : "No data";
    this.summary.textContent = evolutionSummary(model);
  }
}
