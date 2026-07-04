const FAMILY_SPECS = {
  heat: { label: "Heat", color: "#e69f00" },
  wind: { label: "Wind", color: "#56b4e9" },
  wet: { label: "Wet / flooding", color: "#0072b2" },
  drought: { label: "Dry / drought", color: "#f0d95c" },
  stress: { label: "Stress response", color: "#cc79a7" },
  recovery: { label: "Recovery / vigor", color: "#009e73" },
  mixed: { label: "Mixed exposure", color: "#d55e00" },
  other: { label: "Other situation", color: "#9caea6" },
};

const RISK_SPECS = {
  UNKNOWN: { label: "Unknown / data gap", color: "#708078" },
  HIGH: { label: "High", color: "#e76f51" },
  "MED-HIGH": { label: "Medium–high", color: "#ef9b45" },
  "LOW-MED": { label: "Low–medium", color: "#f2c14e" },
  LOW: { label: "Low", color: "#2a9d8f" },
  NONE: { label: "None", color: "#8b9d95" },
};

export function familyKey(properties = {}) {
  const explicit = String(properties.motif_family || "").toLowerCase();
  const text = `${explicit} ${properties.hazard_family || ""} ${properties.hazard_signature || ""} ${properties.response_signature || ""}`.toLowerCase();
  const matches = new Set();
  if (has(text, ["heat", "hot", "temperature"])) matches.add("heat");
  if (has(text, ["wind", "storm"])) matches.add("wind");
  if (has(text, ["pond", "flood", "wet", "rain", "water", "precip"])) matches.add("wet");
  if (has(text, ["drought", "dry", "arid"])) matches.add("drought");
  if (has(text, ["stress", "senescence", "decline"])) matches.add("stress");
  if (has(text, ["recover", "vigor", "greenness"])) matches.add("recovery");
  if (explicit && FAMILY_SPECS[explicit]) return explicit;
  if (matches.size > 1) return "mixed";
  return matches.values().next().value || "other";
}

export function colorFor(properties, mode = "family", alpha = 210) {
  const spec = mode === "risk" ? riskSpec(properties.current_risk_band || properties.max_risk_band) : FAMILY_SPECS[familyKey(properties)];
  return hexToRgba(spec.color, alpha);
}

export function alphaForState(properties = {}, openAlpha = 188) {
  let alpha = openAlpha;
  const state = String(properties.event_state || "").toUpperCase();
  if (state === "DATA_GAP") alpha = Math.min(alpha, 96);
  if (state.startsWith("CLOSED_")) alpha = Math.min(alpha, 72);
  const archetypeState = String(properties.archetype_display_state || "").toLowerCase();
  if (archetypeState === "pending_anchor") return Math.min(alpha, 82);
  if (archetypeState === "novel_unassigned") return Math.min(alpha, 132);
  if (archetypeState && archetypeState !== "accepted") return Math.min(alpha, 104);
  return alpha;
}

export function lineColorFor(properties = {}) {
  const state = String(properties.archetype_display_state || "").toLowerCase();
  if (state === "pending_anchor") return [148, 163, 184, 220];
  if (state === "novel_unassigned") return [251, 146, 60, 245];
  if (state && state !== "accepted") return [203, 213, 225, 235];
  return [5, 20, 15, 210];
}

export function isOpenStory(properties = {}) {
  const state = String(properties.event_state || "").toUpperCase();
  return !state || ["WATCH", "ACTIVE", "SEVERE", "QUIET_PENDING", "RECOVERING", "DATA_GAP"].includes(state);
}

export function colorHexFor(properties, mode = "family") {
  return mode === "risk" ? riskSpec(properties.current_risk_band || properties.max_risk_band).color : FAMILY_SPECS[familyKey(properties)].color;
}

export function legendEntries(mode, features = []) {
  if (mode === "risk") {
    return ["UNKNOWN", "NONE", "LOW", "LOW-MED", "MED-HIGH", "HIGH"].map((key) => ({ key, ...RISK_SPECS[key] }));
  }
  const present = new Set(features.map((feature) => familyKey(feature.properties || {})));
  const keys = Object.keys(FAMILY_SPECS).filter((key) => present.has(key));
  const visible = keys.length ? keys : ["heat", "wind", "wet", "drought", "mixed", "other"];
  return visible.slice(0, 7).map((key) => ({ key, ...FAMILY_SPECS[key] }));
}

export function applyVisualProperties(collection, mode, history = false) {
  const features = (collection?.features || []).map((feature) => {
    const properties = { ...(feature.properties || {}) };
    const age = Math.max(0, Number(properties.age_index || 0));
    properties.__story_color = colorHexFor(properties, mode);
    properties.__story_line_color = rgbaToHex(lineColorFor(properties));
    properties.__story_opacity = history
      ? Math.max(0.06, 0.24 - age * 0.035)
      : alphaForState(properties, 184) / 255;
    return { ...feature, properties };
  });
  return { type: "FeatureCollection", features, meta: collection?.meta || {} };
}

function rgbaToHex(values) {
  return `#${values.slice(0, 3).map((value) => Number(value).toString(16).padStart(2, "0")).join("")}`;
}

function riskSpec(value) {
  return RISK_SPECS[String(value || "NONE").toUpperCase()] || RISK_SPECS.NONE;
}

function has(text, words) {
  return words.some((word) => text.includes(word));
}

function hexToRgba(hex, alpha) {
  const value = String(hex).replace("#", "");
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
    alpha,
  ];
}
