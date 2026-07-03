# Story-monitoring glossary

- **Event / story**: one continuous field-level episode, identified by `event_id`.
- **Usable day**: an event day with observed pressure evidence eligible for causal feature construction.
- **Anchor**: the single cutoff date used to represent an event for archetype discovery or assignment.
- **Archetype / motif**: a stable, model-versioned pattern assigned once to an eligible event.
- **Family**: a small, human-reviewed parent category used for map color and aggregation.
- **Lifecycle**: the evolving operational phase (`WATCH`, `ACTIVE`, `SEVERE`, recovery/closure states); not identity.
- **Pending**: an event that has not accumulated enough causal evidence for assignment.
- **Novel unassigned**: an eligible event outside the model's acceptance radius or assignment margin.
- **Review overlay**: immutable human curation keyed by `(model_version, archetype_id)`.
