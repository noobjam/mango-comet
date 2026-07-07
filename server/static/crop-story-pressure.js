import {
  DAY_MS,
  dateValue,
  isoDay,
  pressureDate,
  truthy,
} from "./crop-story-time.js";

const MAX_PRESSURE_CELLS_PER_LANE = 366;

export function pressureLaneModels(sourceRows, hazards, start, end) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
  const totalDays = Math.max(2, Math.floor((end - start) / DAY_MS) + 1);
  const cellCount = Math.min(totalDays, MAX_PRESSURE_CELLS_PER_LANE);
  return hazards.map((hazard) => {
    const values = sourceRows.map(normalizePressure)
      .filter((row) => row && row.hazard === hazard);
    const cells = [];
    for (let index = 0; index < cellCount; index += 1) {
      const firstDay = Math.floor((index * totalDays) / cellCount);
      const nextDay = Math.max(
        firstDay + 1,
        Math.floor(((index + 1) * totalDays) / cellCount),
      );
      const firstTime = start + firstDay * DAY_MS;
      const nextTime = Math.min(end + DAY_MS, start + nextDay * DAY_MS);
      const matching = values.filter((row) => row.time >= firstTime && row.time < nextTime);
      const observed = matching.filter((row) => row.observed);
      const observedCount = observed.reduce((sum, row) => sum + row.observedCount, 0);
      const monitoredCount = matching.reduce((sum, row) => sum + row.monitoredCount, 0);
      const riskRank = Math.max(0, ...observed.map((row) => row.riskRank));
      const active = observed.some((row) => row.active);
      const state = !observed.length || observedCount <= 0 ? "missing"
        : monitoredCount > observedCount ? "partial"
          : riskRank <= 1 && !active ? "observed-low" : "elevated";
      cells.push({
        startX: (firstDay / totalDays) * 100,
        width: ((nextDay - firstDay) / totalDays) * 100,
        state,
        riskRank,
        active,
        observedCount,
        monitoredCount,
        startDate: isoDay(firstTime),
        endDate: isoDay(Math.max(firstTime, nextTime - DAY_MS)),
      });
    }
    return { hazard, color: hazardColor(hazard), cells };
  });
}

function normalizePressure(row = {}) {
  const time = dateValue(pressureDate(row));
  if (!Number.isFinite(time)) return null;
  const aggregate = row.pressure_field_count !== undefined;
  const observedCount = aggregate
    ? Math.max(0, Number(row.pressure_field_count || 0))
    : truthy(row.pressure_observed === undefined ? true : row.pressure_observed) ? 1 : 0;
  const monitoredCount = aggregate
    ? Math.max(observedCount, Number(row.monitored_field_count || observedCount))
    : 1;
  return {
    time,
    hazard: String(row.hazard_family || "unknown").toLowerCase(),
    riskRank: Number(row.risk_rank ?? row.max_risk_rank ?? 0),
    active: truthy(row.pressure_active)
      || Number(row.risk_rank ?? row.max_risk_rank ?? 0) > 1,
    observed: observedCount > 0,
    observedCount,
    monitoredCount,
  };
}

function hazardColor(value) {
  const key = String(value || "").toLowerCase();
  if (key.includes("heat")) return "#f07b68";
  if (key.includes("drought") || key.includes("dry")) return "#e2ae45";
  if (key.includes("flood") || key.includes("pond")) return "#59a9d8";
  if (key.includes("wind")) return "#a98bd4";
  return "#73e2b4";
}
