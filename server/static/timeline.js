export class TimelineController {
  constructor({ onChange }) {
    this.onChange = onChange;
    this.buckets = [];
    this.activity = new Map();
    this.pendingIndex = 0;
    this.committedIndex = -1;
    this.playing = false;
    this.timer = null;
    this.inputTimer = null;
    this.reducedMotion = Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
    this.elements = Object.fromEntries([
      "timelineSlider", "timelineLabel", "timelineHistogram", "timelineStart", "timelineEnd",
      "frameStats", "playTimeline", "playbackSpeed", "previousBucket", "nextBucket",
    ].map((id) => [id, document.getElementById(id)]));
    this.bind();
    if (this.reducedMotion) {
      this.elements.playTimeline.title = "Reduced-motion mode: playback uses discrete, slower frame changes.";
    }
  }

  bind() {
    const e = this.elements;
    e.timelineSlider.addEventListener("input", () => {
      this.stop();
      this.select(Number(e.timelineSlider.value));
      window.clearTimeout(this.inputTimer);
      this.inputTimer = window.setTimeout(() => this.onChange?.(this.pendingIndex), 80);
    });
    e.previousBucket.addEventListener("click", () => this.step(-1));
    e.nextBucket.addEventListener("click", () => this.step(1));
    e.playTimeline.addEventListener("click", () => this.toggle());
    e.playbackSpeed.addEventListener("change", () => {
      if (this.playing) this.schedule();
    });
    e.timelineHistogram.addEventListener("click", (event) => {
      const bar = event.target.closest("[data-index]");
      if (!bar) return;
      this.stop();
      this.select(Number(bar.dataset.index));
      this.onChange?.(this.pendingIndex);
    });
  }

  setBuckets(timelinePayload = {}, activityPayload = null) {
    this.buckets = (timelinePayload.buckets || timelinePayload.timeline || []).map((row) => ({
      ...row,
      timeline_bucket: String(row.timeline_bucket || row.bucket || row.date || ""),
    })).filter((row) => row.timeline_bucket);
    if (!this.buckets.length) throw new Error("No timeline buckets are available.");
    const e = this.elements;
    e.timelineSlider.min = "0";
    e.timelineSlider.max = String(this.buckets.length - 1);
    e.timelineStart.textContent = formatShort(this.buckets[0].timeline_bucket);
    e.timelineEnd.textContent = formatShort(this.buckets.at(-1).timeline_bucket);
    this.setActivity(activityPayload || timelinePayload);
    this.select(this.buckets.length - 1);
  }

  setActivity(payload = {}) {
    const rows = payload.buckets || payload.activity || payload.timeline || [];
    this.activity = new Map(rows.map((row) => [
      String(row.timeline_bucket || row.bucket || row.date || ""),
      row,
    ]));
    this.renderHistogram();
  }

  select(index) {
    this.pendingIndex = clamp(index, 0, Math.max(0, this.buckets.length - 1));
    const bucket = this.currentBucket();
    this.elements.timelineSlider.value = String(this.pendingIndex);
    this.elements.timelineSlider.setAttribute("aria-valuetext", bucket ? `Week of ${formatLong(bucket.timeline_bucket)}` : "No week");
    this.renderHistogram();
  }

  commit(index, frameText) {
    if (index !== this.pendingIndex) return;
    this.committedIndex = index;
    const bucket = this.buckets[index];
    this.elements.timelineLabel.textContent = bucket ? `Week of ${formatLong(bucket.timeline_bucket)}` : "No week selected";
    this.elements.frameStats.textContent = frameText || "No visible activity";
    this.renderHistogram();
  }

  currentBucket() {
    return this.buckets[this.pendingIndex] || null;
  }

  latestActiveIndex() {
    for (let index = this.buckets.length - 1; index >= 0; index -= 1) {
      const row = this.activity.get(this.buckets[index].timeline_bucket);
      if (row && activityCount(row) > 0) return index;
    }
    return this.buckets.length - 1;
  }

  renderHistogram() {
    const histogram = this.elements.timelineHistogram;
    if (!histogram || !this.buckets.length) return;
    const counts = this.buckets.map((bucket) => activityCount(this.activity.get(bucket.timeline_bucket)));
    const maximum = Math.max(1, ...counts);
    const fragment = document.createDocumentFragment();
    counts.forEach((count, index) => {
      const activityRow = this.activity.get(this.buckets[index].timeline_bucket) || {};
      const bar = document.createElement("span");
      bar.className = "timeline-bar";
      if (!count) bar.classList.add("is-gap");
      if (index === this.pendingIndex) bar.classList.add("is-current");
      bar.dataset.index = String(index);
      bar.style.setProperty("--bar-height", count ? `${Math.max(4, Math.round((count / maximum) * 25))}px` : "2px");
      const unit = isIncidentActivity(activityRow) ? "incident stories" : "affected fields";
      bar.title = `${formatLong(this.buckets[index].timeline_bucket)} · ${count.toLocaleString()} ${unit}`;
      fragment.appendChild(bar);
    });
    histogram.replaceChildren(fragment);
  }

  step(direction) {
    this.stop();
    const next = (this.pendingIndex + direction + this.buckets.length) % this.buckets.length;
    this.select(next);
    this.onChange?.(next);
  }

  toggle() {
    if (this.playing) this.stop();
    else {
      this.playing = true;
      this.elements.playTimeline.textContent = "Pause";
      this.elements.playTimeline.setAttribute("aria-pressed", "true");
      this.schedule();
    }
  }

  stop() {
    this.playing = false;
    window.clearTimeout(this.timer);
    this.timer = null;
    this.elements.playTimeline.textContent = "Play";
    this.elements.playTimeline.setAttribute("aria-pressed", "false");
  }

  schedule(delay = Number(this.elements.playbackSpeed.value || 1000)) {
    window.clearTimeout(this.timer);
    if (!this.playing) return;
    const effectiveDelay = this.reducedMotion ? Math.max(1600, delay) : delay;
    this.timer = window.setTimeout(async () => {
      const next = (this.pendingIndex + 1) % this.buckets.length;
      this.select(next);
      try {
        await this.onChange?.(next);
      } finally {
        if (this.playing) this.schedule();
      }
    }, effectiveDelay);
  }
}

export function activityCount(row = {}) {
  if (isIncidentActivity(row)) {
    return Number(row?.activity_count ?? row?.incident_count ?? row?.story_cluster_count ?? 0);
  }
  return Number(
    row?.field_count
      ?? row?.affected_field_count
      ?? row?.feature_count
      ?? row?.event_count
      ?? row?.incident_count
      ?? row?.story_cluster_count
      ?? 0,
  );
}

function isIncidentActivity(row = {}) {
  return row?.activity_unit === "incident_stories" || row?.activity_count !== undefined;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : minimum));
}

function formatLong(value) {
  const date = parseDate(value);
  return date ? new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(date) : String(value);
}

function formatShort(value) {
  const date = parseDate(value);
  return date ? new Intl.DateTimeFormat(undefined, { month: "short", year: "2-digit", timeZone: "UTC" }).format(date) : String(value);
}

function parseDate(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
  return match ? new Date(`${match[1]}-${match[2]}-${match[3]}T00:00:00Z`) : null;
}
