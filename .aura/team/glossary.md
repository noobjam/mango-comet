# Story-monitoring glossary

- **Crop instance**: one field/crop/season growing cycle, identified by `crop_instance_id`.
- **Field episode**: one crop instance's continuous hazard lifecycle, identified by `episode_id` (`event_id` compatibility alias).
- **Weekly component**: one same-hazard connected set of significant fixed-grid
  cells in one reporting week, identified by `component_id`.
- **Exposure incident**: local same-hazard weekly components linked through time, identified by `exposure_id`.
- **Crop-impact story**: one exposure incident's impact on one crop, identified by stable `incident_id`.
- **Story onset**: the first causally confirmed week with an exact footprint and
  adequate monitored/evaluable crop evidence; WATCH or low-coverage evidence
  alone is not onset.
- **Causal/as-of**: computed only from evidence available through the published
  week. This is a leakage guarantee, not a causal-effect claim.
- **Evidence role**: the exact weekly footprint contribution of a cell or member
  (`pressure`, `impact`, or `watch`), retained without interpolation.
- **Exact footprint history**: the week-by-week union of published significant
  grid-cell rectangles. It is not a route, centroid trajectory, or propagation
  estimate.
- **Data gap**: a week without enough current evaluable evidence; lifecycle
  clocks freeze rather than inventing improvement or deterioration.
- **Data-censored closure**: a terminal monitoring boundary caused by extended
  missing evidence, distinct from recovery or crop outcome.
- **Unresolved response**: observed crop-response evidence that has not yet
  received later recovery evidence; it can transfer across deterministic
  split/merge lineage.
- **Recurrence**: a new confirmed segment after a prior terminal segment; it is
  represented explicitly rather than mutating the closed history.
- **Usable day**: an event day with observed pressure evidence eligible for causal feature construction.
- **Anchor**: the single cutoff date used to represent an event for archetype discovery or assignment.
- **Archetype**: an optional model-versioned pattern learned from completed crop-impact stories; never story identity.
- **Family**: a small, human-reviewed parent category used for map color and aggregation.
- **Lifecycle**: the evolving operational phase of an episode or story; not identity.
- **Pending**: an event that has not accumulated enough causal evidence for assignment.
- **Novel unassigned**: an eligible event outside the model's acceptance radius or assignment margin.
- **Review overlay**: immutable human curation keyed by `(model_version, archetype_id)`.
