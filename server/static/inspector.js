export class Inspector {
  constructor() {
    this.tooltip = document.getElementById("hoverTooltip");
    this.selection = document.getElementById("selectionDetails");
  }

  showHover(properties, point) {
    if (!properties || !point) {
      this.tooltip.hidden = true;
      return;
    }
    this.tooltip.replaceChildren(
      node("strong", properties.field_id || "Field"),
      node("div", readableStory(properties)),
      node("div", locationText(properties), "muted")
    );
    this.tooltip.hidden = false;
    const width = this.tooltip.offsetWidth || 240;
    const height = this.tooltip.offsetHeight || 70;
    const parent = this.tooltip.parentElement.getBoundingClientRect();
    const left = Math.min(parent.width - width - 10, Math.max(10, Number(point.x || 0) + 14));
    const top = Math.min(parent.height - height - 10, Math.max(10, Number(point.y || 0) + 14));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${top}px`;
  }

  loading(properties) {
    this.selection.className = "detail";
    this.selection.replaceChildren(...fieldHeader(properties), node("span", "Loading event windows…", "muted"));
  }

  showSelection(properties, events = []) {
    this.selection.className = "detail";
    const children = fieldHeader(properties);
    if (!events.length) {
      children.push(node("span", "No event windows were found for this field.", "muted"));
    } else {
      const heading = node("strong", `${events.length.toLocaleString()} recent event windows`);
      children.push(heading);
      for (const event of events.slice(0, 12)) {
        const row = document.createElement("div");
        row.className = "detail-row";
        const dates = node("span", `${event.event_start_date || "?"} → ${event.active_end_date || "?"}`);
        const evidence = node("strong", [event.max_risk_band, event.hazard_signature].filter(Boolean).join(" · "));
        row.append(dates, evidence);
        children.push(row);
      }
    }
    this.selection.replaceChildren(...children);
  }

  showError(properties, error) {
    this.selection.className = "detail";
    this.selection.replaceChildren(...fieldHeader(properties), node("span", `Could not load event windows: ${error.message}`, "muted"));
  }
}

function fieldHeader(properties = {}) {
  const title = node("strong", properties.field_id || "Selected field");
  const story = node("div", readableStory(properties));
  const place = node("div", locationText(properties), "muted");
  const id = document.createElement("code");
  id.textContent = properties.story_cluster_id || properties.motif_id || "No story ID";
  return [title, story, id, place];
}

function readableStory(properties = {}) {
  return [
    properties.motif_family,
    properties.max_risk_band,
    properties.hazard_signature,
    properties.response_signature,
  ].filter(Boolean).join(" · ") || "No evidence label";
}

function locationText(properties = {}) {
  return [properties.district, properties.sector, properties.cell, properties.village].filter(Boolean).join(" / ");
}

function node(tag, text, className = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = String(text || "");
  return element;
}
