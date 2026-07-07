export const DAY_MS = 24 * 60 * 60 * 1000;

export function dateValue(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})/);
  return match ? Date.parse(`${match[1]}-${match[2]}-${match[3]}T00:00:00Z`) : NaN;
}

export function timePosition(start, end) {
  return (value) => {
    const timestamp = typeof value === "number" ? value : dateValue(value);
    if (!Number.isFinite(timestamp) || !Number.isFinite(start) || !Number.isFinite(end)) {
      return null;
    }
    return Math.max(0, Math.min(100, ((timestamp - start) / (end - start)) * 100));
  };
}

export function timeTicks(start, end) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
  return Array.from({ length: 5 }, (_, index) => ({
    x: index * 25,
    time: start + ((end - start) * index) / 4,
  }));
}

export function pressureDate(row = {}) {
  return row.calendar_date || row.pressure_effective_date || row.pressure_observation_date;
}

export function s2SourceDate(row = {}) {
  return row.spectral_source_date || row.source_date || row.knowledge_date;
}

export function s2KnowledgeDate(row = {}) {
  return row.knowledge_date || row.available_date || row.knowledge_time;
}

export function storySourceDate(row = {}) {
  return row.story_week || row.timeline_bucket;
}

export function storyKnowledgeDate(row = {}) {
  return row.story_known_date || row.knowledge_date || row.knowledge_time
    || row.story_week || row.timeline_bucket;
}

export function isoDay(timestamp) {
  return new Date(timestamp).toISOString().slice(0, 10);
}

export function shortDay(timestamp) {
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit", month: "short", year: "2-digit", timeZone: "UTC",
  }).format(new Date(timestamp));
}

export function rows(value) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.rows)) return value.rows;
  if (Array.isArray(value?.items)) return value.items;
  return value && typeof value === "object" ? [value] : [];
}

export function groupBy(values, key) {
  const groups = new Map();
  for (const value of values) {
    const id = key(value);
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(value);
  }
  return groups;
}

export function truthy(value) {
  return value === true || value === 1
    || ["true", "1", "yes"].includes(String(value).toLowerCase());
}

export function pretty(value) {
  return String(value || "Unknown").replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}

export function cssToken(value) {
  return String(value || "unknown").toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

export function compactCount(value) {
  const count = Number(value || 0);
  if (count >= 1000) return `${Math.round(count / 100) / 10}k`;
  return String(count);
}

export function shortId(value) {
  return String(value || "unknown").split(":").at(-1).slice(-8);
}
