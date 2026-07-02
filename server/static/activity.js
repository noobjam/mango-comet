import { isOpenStory } from "./palette.js";

export function computeActivityStats(frame, trail) {
  const currentFeatures = frame?.features || [];
  const stateAware = currentFeatures.some((feature) => feature.properties?.event_state);
  const currentIds = idsFrom(stateAware ? currentFeatures.filter((feature) => isOpenStory(feature.properties)) : currentFeatures);
  const meta = trail?.meta || {};
  const lifecycleAwareTransitions = meta.transition_scope === "open_previous_bucket";
  if (
    (!stateAware || lifecycleAwareTransitions)
    &&
    meta.transition_counts_available !== false
    && [meta.new_current_field_count, meta.persisting_field_count, meta.departed_field_count].every(isCount)
  ) {
    return {
      affected: currentIds.size,
      entering: Number(meta.new_current_field_count),
      persisting: Number(meta.persisting_field_count),
      inactive: Number(meta.departed_field_count),
    };
  }
  const currentBucket = String(frame?.meta?.timeline_bucket || "");
  const prior = (trail?.features || []).filter((feature) =>
    isPrior(feature, currentBucket) && (!stateAware || isOpenStory(feature.properties))
  );
  const ageValues = prior.map((feature) => Number(feature.properties?.age_index)).filter((value) => value > 0);
  const closestAge = ageValues.length ? Math.min(...ageValues) : null;
  let previous = [];
  if (closestAge !== null) {
    previous = prior.filter((feature) => Number(feature.properties?.age_index) === closestAge);
  } else {
    const buckets = [...new Set(prior.map((feature) => String(feature.properties?.timeline_bucket || "")).filter(Boolean))].sort();
    const latest = buckets.at(-1);
    previous = latest ? prior.filter((feature) => String(feature.properties?.timeline_bucket) === latest) : [];
  }
  const previousIds = idsFrom(previous);
  const recentIds = idsFrom(prior);
  const historyAvailable = previousIds.size > 0 || Number(trail?.meta?.bucket_count || 0) > 1;
  return {
    affected: currentIds.size,
    entering: historyAvailable ? differenceSize(currentIds, previousIds) : null,
    persisting: historyAvailable ? intersectionSize(currentIds, previousIds) : null,
    inactive: historyAvailable ? differenceSize(recentIds, currentIds) : null,
  };
}

function isCount(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value));
}

function isPrior(feature, currentBucket) {
  const properties = feature.properties || {};
  return Number(properties.age_index || 0) > 0 || (currentBucket && String(properties.timeline_bucket || "") !== currentBucket);
}

function idsFrom(features) {
  return new Set(features.map((feature) => String(feature.properties?.field_id || "")).filter(Boolean));
}

function differenceSize(left, right) {
  let count = 0;
  for (const value of left) if (!right.has(value)) count += 1;
  return count;
}

function intersectionSize(left, right) {
  let count = 0;
  for (const value of left) if (right.has(value)) count += 1;
  return count;
}
