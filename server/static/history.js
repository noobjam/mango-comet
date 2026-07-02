const UNAVAILABLE_REASON = "optimized_geometry_required";

export function historySupported(manifest) {
  return manifest?.server?.optimized_geometry !== false;
}

export function unavailableHistory() {
  return {
    type: "FeatureCollection",
    features: [],
    meta: {
      history_available: false,
      transition_counts_available: false,
      truncated: false,
      unavailable_reason: UNAVAILABLE_REASON,
    },
  };
}

export async function loadHistory(manifest, fetchHistory) {
  if (!historySupported(manifest)) return unavailableHistory();
  return fetchHistory();
}
