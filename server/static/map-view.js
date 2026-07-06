import {
  alphaForState,
  applyVisualProperties,
  colorFor,
  colorHexFor,
  lineColorFor,
} from "./palette.js";
import { buildEvolutionModel, evolutionDeckLayers, evolutionGeoJson } from "./map-evolution.js";
import {
  footprintRoleCollection,
  footprintVisualModel,
  incidentHitCandidates,
  nextIncidentCandidate,
  v3LayerModel,
  v4LayerModel,
} from "./incident-v3.js";
import { footprintHistoryVisualModel } from "./incident-story.js";

const EMPTY = { type: "FeatureCollection", features: [], meta: {} };

export class MapView {
  constructor({
    container, config, bounds, incidentMode = false, v4Mode = false,
    onReady, onHover, onSelect, onSelectIncident, onViewportChange,
  }) {
    this.container = container;
    this.config = config;
    this.bounds = bounds;
    this.onReady = onReady;
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.onSelectIncident = onSelectIncident;
    this.onViewportChange = onViewportChange;
    this.frame = EMPTY;
    this.footprints = EMPTY;
    this.selectedIncidentHistory = EMPTY;
    this.selectedIncidentCurrentFootprint = null;
    this.trail = EMPTY;
    this.colorMode = "family";
    this.showHistory = true;
    this.selectedFieldId = "";
    this.selectedIncidentId = "";
    this.incidentMode = Boolean(incidentMode);
    this.v4Mode = Boolean(v4Mode);
    this.v4Frame = null;
    this.evolution = null;
    this.evolutionBucket = "";
    this.overlay = null;
    this.ready = false;
    this.fallbackEventsBound = false;
    this.sourceData = new Map();
    this.fallbackFootprintInput = null;
    this.fallbackFootprintOutput = EMPTY;
    this.fallbackHistoryInput = null;
    this.fallbackHistoryOutput = EMPTY;
  }

  mount() {
    if (!window.maplibregl) throw new Error("MapLibre failed to load.");
    const center = this.bounds
      ? [(this.bounds.minLon + this.bounds.maxLon) / 2, (this.bounds.minLat + this.bounds.maxLat) / 2]
      : [30.35, -1.2];
    this.map = new maplibregl.Map({
      container: this.container,
      style: this.style(),
      center,
      zoom: 10,
      pitch: 0,
      bearing: 0,
      hash: true,
      attributionControl: true,
    });
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    this.map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");
    this.map.on("load", () => this.handleLoad());
    this.map.on("moveend", () => {
      if (this.ready) this.onViewportChange?.(this.boundsString());
    });
    return this;
  }

  style() {
    return {
      version: 8,
      sources: {
        satellite: {
          type: "raster",
          tiles: this.config.raster.tiles,
          tileSize: this.config.raster.tileSize || 256,
          attribution: this.config.raster.attribution || "",
        },
      },
      layers: [
        {
          id: "satellite",
          type: "raster",
          source: "satellite",
          paint: {
            "raster-opacity": 0.7,
            "raster-saturation": -0.42,
            "raster-contrast": 0.1,
            "raster-brightness-max": 0.74,
          },
        },
      ],
    };
  }

  handleLoad() {
    if (this.bounds) {
      const wide = window.innerWidth > 820;
      this.map.fitBounds(
        [[this.bounds.minLon, this.bounds.minLat], [this.bounds.maxLon, this.bounds.maxLat]],
        { padding: { top: 54, right: wide ? 430 : 54, bottom: 54, left: 54 }, duration: 0 }
      );
    }
    if (window.deck?.MapboxOverlay) {
      try {
        this.overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
        this.map.addControl(this.overlay);
      } catch {
        this.overlay = null;
      }
    }
    this.ready = true;
    this.render();
    this.onReady?.();
  }

  setData(
    frame,
    trail,
    evolution = this.evolution,
    evolutionBucket = this.evolutionBucket,
    footprints = this.footprints,
  ) {
    this.frame = frame || EMPTY;
    this.trail = trail || EMPTY;
    this.footprints = footprints || EMPTY;
    this.evolution = evolution || null;
    this.evolutionBucket = String(evolutionBucket || "");
    this.render();
  }

  setV4Data(payload, bucket = "") {
    this.v4Frame = payload || null;
    this.frame = payload?.fields || EMPTY;
    this.footprints = payload?.story_footprints || EMPTY;
    this.evolutionBucket = String(bucket || payload?.calendar_date || "");
    this.render();
  }

  setColorMode(mode) {
    this.colorMode = mode === "risk" ? "risk" : "family";
    this.render();
  }

  setHistoryVisible(visible) {
    this.showHistory = Boolean(visible);
    this.render();
  }

  setSelectedField(fieldId) {
    this.selectedFieldId = String(fieldId || "");
    this.render();
  }

  setSelectedIncident(incidentId) {
    this.selectedIncidentId = String(incidentId || "");
    if (!this.selectedIncidentId) {
      this.selectedIncidentHistory = EMPTY;
      this.selectedIncidentCurrentFootprint = null;
    }
    this.render();
  }

  setSelectedIncidentStory(collection, currentFootprint = null) {
    const history = collection || EMPTY;
    const current = currentFootprint || null;
    if (
      this.selectedIncidentHistory === history
      && this.selectedIncidentCurrentFootprint === current
    ) return;
    this.selectedIncidentHistory = history;
    this.selectedIncidentCurrentFootprint = current;
    this.render();
  }

  setEvolution(payload, bucket) {
    this.evolution = this.incidentMode ? null : payload || null;
    this.evolutionBucket = String(bucket || "");
    this.render();
  }

  boundsString() {
    if (!this.map) return "";
    const bounds = this.map.getBounds();
    return [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]
      .map((value) => Number(value).toFixed(6))
      .join(",");
  }

  incidentFieldsVisible() {
    if (!this.incidentMode) return true;
    const model = this.v4Mode ? v4LayerModel(this.map?.getZoom?.() || 0)
      : v3LayerModel(this.map?.getZoom?.() || 0);
    return model.fields.visible;
  }

  render() {
    if (!this.ready) return;
    if (this.overlay) {
      this.overlay.setProps({ layers: this.deckLayers() });
      return;
    }
    this.renderFallback();
  }

  deckLayers() {
    if (this.incidentMode) return this.incidentDeckLayers();
    const layers = [];
    const history = this.priorTrail();
    if (this.showHistory && history.features.length) {
      layers.push(new deck.GeoJsonLayer({
        id: "story-history",
        data: history,
        pickable: false,
        filled: true,
        stroked: false,
        getFillColor: (feature) => {
          const age = Math.max(1, Number(feature.properties?.age_index || 1));
          return colorFor(feature.properties, this.colorMode, Math.max(20, 82 - age * 11));
        },
        updateTriggers: { getFillColor: [this.colorMode, this.trail.meta?.timeline_bucket] },
      }));
    }
    layers.push(this.fieldDeckLayer());
    layers.push(...this.evolutionLayers());
    return layers;
  }

  incidentDeckLayers() {
    const zoom = this.map?.getZoom?.() || 0;
    const model = this.v4Mode ? v4LayerModel(zoom) : v3LayerModel(zoom);
    const dashExtension = deck.PathStyleExtension
      ? [new deck.PathStyleExtension({ dash: true })]
      : [];
    const layers = [];
    if (this.v4Mode && model.fieldOverview.visible) {
      layers.push(new deck.GeoJsonLayer({
        id: "v4-complete-field-overview",
        data: this.v4Frame?.field_overview || EMPTY,
        pickable: false,
        filled: true,
        stroked: false,
        getFillColor: (feature) => {
          const count = Math.max(0, Number(feature.properties?.represented_field_count || 0));
          return [69, 117, 102, Math.min(105, 18 + Math.round(Math.log2(count + 1) * 12))];
        },
      }));
    }
    if (this.v4Mode) {
      const pressure = this.v4Frame?.pressure || EMPTY;
      layers.push(new deck.GeoJsonLayer({
        id: "v4-daily-pressure-fill",
        data: pressure,
        pickable: false,
        filled: true,
        stroked: false,
        getFillColor: (feature) => colorFor(
          feature.properties,
          "family",
          Math.min(72, 18 + Number(feature.properties?.max_risk_rank || 0) * 14),
        ),
      }));
      const pressureHazards = [...new Set(
        (pressure.features || [])
          .map((feature) => String(feature.properties?.hazard_family || ""))
          .filter(Boolean),
      )].sort();
      pressureHazards.forEach((hazard, index) => {
        const features = (pressure.features || []).filter(
          (feature) => String(feature.properties?.hazard_family || "") === hazard,
        );
        layers.push(new deck.GeoJsonLayer({
          id: `v4-daily-pressure-lane-${hazard}`,
          data: { type: "FeatureCollection", features },
          pickable: false,
          filled: false,
          stroked: true,
          getLineColor: (feature) => colorFor(feature.properties, "family", 230),
          getLineWidth: 2 + (pressureHazards.length - index) * 2,
          lineWidthUnits: "pixels",
          lineWidthMinPixels: 2,
        }));
      });
      layers.push(new deck.GeoJsonLayer({
        id: "v4-s2-crop-impact",
        data: this.v4Frame?.crop_impact || EMPTY,
        pickable: false,
        filled: true,
        stroked: true,
        getFillColor: (feature) => {
          const decline = Number(feature.properties?.decline_field_count || 0);
          const recovery = Number(feature.properties?.recovery_field_count || 0);
          const stale = Number(feature.properties?.stale_evidence_field_count || 0);
          const evidence = Math.max(1, Number(feature.properties?.crop_evidence_field_count || 0));
          const alpha = stale >= evidence ? 45 : 105;
          return decline >= recovery ? [204, 121, 167, alpha] : [0, 158, 115, alpha];
        },
        getLineColor: [235, 225, 233, 180],
        getLineWidth: 0.8,
        lineWidthUnits: "pixels",
      }));
    }
    layers.push(new deck.GeoJsonLayer({
      id: "incident-exact-complete-footprints",
      data: this.footprints,
      pickable: true,
      filled: true,
      stroked: true,
      lineWidthUnits: "pixels",
      extensions: dashExtension,
      getFillColor: (feature) => {
        const visual = footprintVisualModel(feature.properties);
        return colorFor(feature.properties, "family", visual.fillAlpha);
      },
      getLineColor: (feature) => String(feature.properties?.incident_id) === this.selectedIncidentId
        ? [255, 255, 255, 255]
        : colorFor(feature.properties, "family", footprintVisualModel(feature.properties).lineAlpha),
      getLineWidth: (feature) => String(feature.properties?.incident_id) === this.selectedIncidentId
        ? 3.5 + this.coincidentOutlineExtra(feature.properties)
        : footprintVisualModel(feature.properties).lineWidth
          + this.coincidentOutlineExtra(feature.properties),
      getDashArray: (feature) => footprintVisualModel(feature.properties).dash,
      dashJustified: true,
      lineWidthMinPixels: 1,
      updateTriggers: {
        getLineColor: [this.selectedIncidentId],
        getLineWidth: [this.selectedIncidentId],
      },
      onHover: (info) => this.onHover?.(
        info.object?.properties || null,
        { x: info.x, y: info.y },
      ),
      onClick: (info) => this.selectCoincidentIncident(
        info.object?.properties || null,
        this.incidentDeckHitFeatures(info),
      ),
    }));
    if (this.selectedIncidentHistory.features?.length) {
      layers.push(new deck.GeoJsonLayer({
        id: "incident-selected-exact-history",
        data: this.selectedIncidentHistory,
        pickable: false,
        filled: false,
        stroked: true,
        lineWidthUnits: "pixels",
        extensions: dashExtension,
        getLineColor: (feature) => colorFor(
          feature.properties,
          "family",
          footprintHistoryVisualModel(feature.properties).lineAlpha,
        ),
        getLineWidth: (feature) => footprintHistoryVisualModel(
          feature.properties,
        ).lineWidth,
        getDashArray: (feature) => footprintHistoryVisualModel(
          feature.properties,
        ).dash,
        dashJustified: true,
      }));
    }
    const selectedFootprints = this.selectedIncidentFootprints();
    for (const [role, fill, line] of [
      ["watch", [242, 193, 78, 22], [242, 193, 78, 180]],
      ["impact", [204, 121, 167, 62], [204, 121, 167, 210]],
      ["pressure", [240, 122, 104, 72], [240, 122, 104, 230]],
    ]) {
      const data = footprintRoleCollection(selectedFootprints, role);
      if (!data.features.length) continue;
      layers.push(new deck.GeoJsonLayer({
        id: `incident-${role}-evidence-cells`,
        data,
        pickable: false,
        filled: true,
        stroked: true,
        getFillColor: fill,
        getLineColor: line,
        getLineWidth: 0.8,
        lineWidthUnits: "pixels",
      }));
    }
    if (model.fields.visible) layers.push(this.fieldDeckLayer("incident-field-drilldown"));
    return layers;
  }

  fieldDeckLayer(id = "story-current-fields") {
    return new deck.GeoJsonLayer({
      id,
      data: this.frame,
      pickable: true,
      filled: true,
      stroked: true,
      lineWidthUnits: "pixels",
      getFillColor: (feature) => colorFor(
        feature.properties,
        this.colorMode,
        alphaForState(feature.properties, this.incidentMode ? 156 : 188),
      ),
      getLineColor: (feature) => String(feature.properties?.field_id) === this.selectedFieldId
        ? [255, 255, 255, 255]
        : lineColorFor(feature.properties),
      getLineWidth: (feature) => {
        if (String(feature.properties?.field_id) === this.selectedFieldId) return 3;
        const risk = String(
          feature.properties?.current_risk_band || feature.properties?.max_risk_band || "",
        ).toUpperCase();
        if (this.colorMode === "risk" && ["HIGH", "SEVERE"].includes(risk)) return 1.8;
        return this.colorMode === "risk" && risk === "MEDIUM" ? 1.1 : 0.7;
      },
      lineWidthMinPixels: 0.7,
      updateTriggers: {
        getFillColor: [this.colorMode, this.incidentMode],
        getLineColor: [this.selectedFieldId],
        getLineWidth: [this.selectedFieldId, this.colorMode],
      },
      onHover: (info) => this.onHover?.(
        info.object?.properties || null,
        { x: info.x, y: info.y },
      ),
      onClick: (info) => this.onSelect?.(info.object?.properties || null),
    });
  }

  evolutionLayers() {
    const model = buildEvolutionModel(this.evolution, this.evolutionBucket);
    const base = colorFor(this.frame.features?.[0]?.properties || {}, this.colorMode, 220);
    return evolutionDeckLayers(deck, model, base);
  }

  priorTrail() {
    const current = String(this.frame.meta?.timeline_bucket || "");
    return {
      type: "FeatureCollection",
      features: (this.trail.features || []).filter((feature) => {
        const properties = feature.properties || {};
        return Number(properties.age_index || 0) > 0 || (current && String(properties.timeline_bucket) !== current);
      }),
      meta: this.trail.meta || {},
    };
  }

  renderFallback() {
    if (this.incidentMode) {
      this.renderIncidentFallback();
      return;
    }
    const current = applyVisualProperties(this.frame, this.colorMode, false);
    const history = this.showHistory ? applyVisualProperties(this.priorTrail(), this.colorMode, true) : EMPTY;
    this.setSource("incident-footprints-fallback", EMPTY, "incident_id");
    this.setSource("incident-selected-history-fallback", EMPTY, "history_id");
    for (const role of ["watch", "impact", "pressure"]) {
      this.setSource(`incident-${role}-fallback`, EMPTY, "incident_id");
    }
    this.setSource("story-history-fallback", history);
    this.setSource("story-current-fallback", current);
    this.setSource("activity-center-fallback", this.evolutionGeoJson());
    this.ensureFallbackLayers();
    this.map.setLayoutProperty("incident-footprints-fill", "visibility", "none");
    this.map.setLayoutProperty("incident-footprints-pressure-line", "visibility", "none");
    this.map.setLayoutProperty("incident-footprints-recovering-line", "visibility", "none");
    this.map.setLayoutProperty("incident-footprints-quiet-line", "visibility", "none");
    this.map.setLayoutProperty("incident-footprints-carried-line", "visibility", "none");
    this.map.setLayoutProperty("incident-footprints-selected-line", "visibility", "none");
    for (const band of ["recent", "middle", "old"]) {
      this.map.setLayoutProperty(`incident-history-${band}-line`, "visibility", "none");
    }
    for (const role of ["watch", "impact", "pressure"]) {
      this.map.setLayoutProperty(`incident-${role}-fill`, "visibility", "none");
    }
    this.map.setLayoutProperty("story-history-fill", "visibility", this.showHistory ? "visible" : "none");
    this.map.setPaintProperty("story-current-line", "line-width", [
      "case",
      ["==", ["get", "field_id"], this.selectedFieldId], 3,
      ["all", ["==", this.colorMode, "risk"], ["in", ["coalesce", ["get", "current_risk_band"], ["get", "max_risk_band"]], ["literal", ["HIGH"]]]], 1.8,
      ["all", ["==", this.colorMode, "risk"], ["in", ["coalesce", ["get", "current_risk_band"], ["get", "max_risk_band"]], ["literal", ["LOW-MED", "MED-HIGH"]]]], 1.1,
      0.7,
    ]);
    this.map.setPaintProperty("story-current-line", "line-color", [
      "case", ["==", ["get", "field_id"], this.selectedFieldId], "#ffffff", ["get", "__story_line_color"],
    ]);
  }

  renderIncidentFallback() {
    const current = applyVisualProperties(this.frame, this.colorMode, false);
    if (this.fallbackFootprintInput !== this.footprints) {
      this.fallbackFootprintInput = this.footprints;
      this.fallbackFootprintOutput = footprintFallbackCollection(this.footprints);
    }
    if (this.fallbackHistoryInput !== this.selectedIncidentHistory) {
      this.fallbackHistoryInput = this.selectedIncidentHistory;
      this.fallbackHistoryOutput = footprintHistoryFallbackCollection(
        this.selectedIncidentHistory,
      );
    }
    const footprints = this.fallbackFootprintOutput;
    this.setSource("incident-footprints-fallback", footprints, "incident_id");
    this.setSource(
      "incident-selected-history-fallback",
      this.fallbackHistoryOutput,
      "history_id",
    );
    for (const role of ["watch", "impact", "pressure"]) {
      this.setSource(
        `incident-${role}-fallback`,
        footprintRoleCollection(this.selectedIncidentFootprints(), role),
        "incident_id",
      );
    }
    this.setSource("story-history-fallback", EMPTY);
    this.setSource("story-current-fallback", current, "field_id");
    this.setSource("activity-center-fallback", EMPTY);
    this.ensureFallbackLayers();
    for (const id of [
      "incident-footprints-fill",
      "incident-footprints-pressure-line",
      "incident-footprints-recovering-line",
      "incident-footprints-quiet-line",
      "incident-footprints-carried-line",
      "incident-footprints-selected-line",
      "incident-history-recent-line",
      "incident-history-middle-line",
      "incident-history-old-line",
      "incident-watch-fill",
      "incident-impact-fill",
      "incident-pressure-fill",
      "story-current-fill",
      "story-current-line",
    ]) this.map.setLayoutProperty(id, "visibility", "visible");
    this.map.setLayoutProperty("story-history-fill", "visibility", "none");
    this.map.setLayoutProperty("activity-center-dots", "visibility", "none");
    this.map.setFilter("incident-footprints-selected-line", [
      "==", ["get", "incident_id"], this.selectedIncidentId,
    ]);
    this.map.setPaintProperty("story-current-line", "line-color", [
      "case",
      ["==", ["get", "field_id"], this.selectedFieldId],
      "#ffffff",
      ["get", "__story_line_color"],
    ]);
  }

  evolutionGeoJson() {
    return evolutionGeoJson(buildEvolutionModel(this.evolution, this.evolutionBucket));
  }

  setSource(id, data, promoteId = "field_id") {
    const source = this.map.getSource(id);
    if (source) {
      if (this.sourceData.get(id) === data) return;
      source.setData(data);
    } else {
      this.map.addSource(id, { type: "geojson", data, promoteId });
    }
    this.sourceData.set(id, data);
  }

  ensureFallbackLayers() {
    if (!this.map.getLayer("incident-footprints-fill")) this.map.addLayer({
      id: "incident-footprints-fill", type: "fill", source: "incident-footprints-fallback",
      paint: {
        "fill-color": ["get", "__footprint_color"],
        "fill-opacity": ["get", "__footprint_opacity"],
      },
    });
    for (const [id, style, dash] of [
      ["incident-footprints-pressure-line", "pressure", null],
      ["incident-footprints-recovering-line", "recovering", [7, 3]],
      ["incident-footprints-quiet-line", "quiet", [1, 3]],
      ["incident-footprints-carried-line", "carried", [2, 3]],
    ]) {
      if (!this.map.getLayer(id)) {
        const paint = {
          "line-color": ["get", "__footprint_color"],
          "line-opacity": ["get", "__footprint_line_opacity"],
          "line-width": ["get", "__footprint_line_width"],
        };
        if (dash) paint["line-dasharray"] = dash;
        this.map.addLayer({
          id, type: "line", source: "incident-footprints-fallback",
          filter: ["==", ["get", "__footprint_style"], style],
          paint,
        });
      }
    }
    for (const [role, color, opacity] of [
      ["watch", "#f2c14e", 0.1],
      ["impact", "#cc79a7", 0.24],
      ["pressure", "#f07a68", 0.28],
    ]) {
      const id = `incident-${role}-fill`;
      if (!this.map.getLayer(id)) this.map.addLayer({
        id, type: "fill", source: `incident-${role}-fallback`,
        paint: {
          "fill-color": color,
          "fill-opacity": opacity,
          "fill-outline-color": color,
        },
      });
    }
    if (!this.map.getLayer("incident-footprints-selected-line")) this.map.addLayer({
      id: "incident-footprints-selected-line", type: "line", source: "incident-footprints-fallback",
      filter: ["==", ["get", "incident_id"], this.selectedIncidentId],
      paint: { "line-color": "#ffffff", "line-opacity": 1, "line-width": 3.5 },
    });
    for (const [band, dash] of [
      ["recent", [7, 3]],
      ["middle", [4, 4]],
      ["old", [2, 5]],
    ]) {
      const id = `incident-history-${band}-line`;
      if (!this.map.getLayer(id)) this.map.addLayer({
        id,
        type: "line",
        source: "incident-selected-history-fallback",
        filter: ["==", ["get", "age_band"], band],
        paint: {
          "line-color": ["get", "__history_color"],
          "line-opacity": ["get", "__history_opacity"],
          "line-width": ["get", "__history_width"],
          "line-dasharray": dash,
        },
      });
    }
    if (!this.map.getLayer("story-history-fill")) this.map.addLayer({
      id: "story-history-fill", type: "fill", source: "story-history-fallback",
      paint: { "fill-color": ["get", "__story_color"], "fill-opacity": ["get", "__story_opacity"] },
    });
    if (!this.map.getLayer("story-current-fill")) this.map.addLayer({
      id: "story-current-fill", type: "fill", source: "story-current-fallback",
      minzoom: this.incidentMode ? 11 : 0,
      paint: { "fill-color": ["get", "__story_color"], "fill-opacity": ["get", "__story_opacity"] },
    });
    if (!this.map.getLayer("story-current-line")) this.map.addLayer({
      id: "story-current-line", type: "line", source: "story-current-fallback",
      minzoom: this.incidentMode ? 11 : 0,
      paint: { "line-color": ["case", ["==", ["get", "field_id"], this.selectedFieldId], "#ffffff", ["get", "__story_line_color"]], "line-opacity": 0.85, "line-width": 0.7 },
    });
    if (!this.map.getLayer("activity-center-dots")) this.map.addLayer({
      id: "activity-center-dots", type: "circle", source: "activity-center-fallback",
      paint: {
        "circle-radius": ["match", ["get", "kind"], "current-center", 8, "prior-center", 4, 2],
        "circle-color": "rgba(0,0,0,0)",
        "circle-stroke-color": ["match", ["get", "kind"], "current-center", "#ffffff", "#73e2b4"],
        "circle-stroke-width": ["match", ["get", "kind"], "current-center", 2.5, 1.2],
      },
    });
    if (this.fallbackEventsBound) return;
    this.fallbackEventsBound = true;
    this.map.on("mousemove", "story-current-fill", (event) => {
      this.map.getCanvas().style.cursor = "pointer";
      this.onHover?.(event.features?.[0]?.properties || null, event.point);
    });
    this.map.on("mouseleave", "story-current-fill", () => {
      this.map.getCanvas().style.cursor = "";
      this.onHover?.(null, null);
    });
    this.map.on("click", "story-current-fill", (event) => this.onSelect?.(event.features?.[0]?.properties || null));
    this.map.on("mousemove", "incident-footprints-fill", (event) => {
      if (this.fieldFeatureAt(event.point)) return;
      this.map.getCanvas().style.cursor = "pointer";
      this.onHover?.(event.features?.[0]?.properties || null, event.point);
    });
    this.map.on("mouseleave", "incident-footprints-fill", () => {
      this.map.getCanvas().style.cursor = "";
      this.onHover?.(null, null);
    });
    this.map.on("click", "incident-footprints-fill", (event) => {
      if (this.fieldFeatureAt(event.point)) return;
      this.selectCoincidentIncident(
        event.features?.[0]?.properties || null,
        event.features || [],
      );
    });
  }

  fieldFeatureAt(point) {
    if (!this.incidentMode || Number(this.map?.getZoom?.() || 0) < 11) return false;
    return Boolean(this.map.queryRenderedFeatures?.(point, {
      layers: ["story-current-fill"],
    })?.length);
  }

  coincidentOutlineExtra(properties = {}) {
    const count = Math.max(1, Number(properties.coincident_incident_count || 1));
    const index = Math.max(0, Number(properties.coincident_incident_index || 0));
    return Math.max(0, count - index - 1) * 1.15;
  }

  selectedIncidentFootprints() {
    const selected = (this.footprints.features || []).filter(
      (feature) => String(feature.properties?.incident_id || "")
        === this.selectedIncidentId,
    );
    const detail = this.selectedIncidentCurrentFootprint;
    if (detail) {
      const detailBucket = String(detail.properties?.timeline_bucket || "").slice(0, 10);
      const currentBucket = String(this.evolutionBucket || "").slice(0, 10);
      if (detailBucket === currentBucket) {
        const base = selected[0] || detail;
        return {
          type: "FeatureCollection",
          features: [{
            ...base,
            properties: {
              ...(base.properties || {}),
              ...(detail.properties || {}),
            },
          }],
          meta: this.footprints.meta || {},
        };
      }
    }
    return {
      type: "FeatureCollection",
      features: selected,
      meta: this.footprints.meta || {},
    };
  }

  incidentDeckHitFeatures(info = {}) {
    const hits = Array.isArray(info.objects)
      ? info.objects
        .map((item) => item?.object || item)
        .filter((feature) => feature?.properties?.incident_id)
      : [];
    const picker = this.overlay?.pickMultipleObjects
      ? this.overlay.pickMultipleObjects.bind(this.overlay)
      : this.overlay?._deck?.pickMultipleObjects
        ? this.overlay._deck.pickMultipleObjects.bind(this.overlay._deck)
        : null;
    if (picker && Number.isFinite(info.x) && Number.isFinite(info.y)) {
      try {
        const picked = picker({
          x: info.x,
          y: info.y,
          radius: 1,
          depth: 100,
          layerIds: ["incident-exact-complete-footprints"],
        });
        for (const item of picked || []) {
          if (item?.object?.properties?.incident_id) hits.push(item.object);
        }
      } catch {
        // The primary picked object remains selectable on older deck.gl builds.
      }
    }
    if (info.object?.properties?.incident_id) hits.push(info.object);
    return hits;
  }

  selectCoincidentIncident(properties, hitFeatures = []) {
    if (!properties) return;
    const candidates = incidentHitCandidates(
      this.footprints,
      hitFeatures,
      properties,
    );
    const next = nextIncidentCandidate(
      candidates,
      this.selectedIncidentId,
      properties,
    );
    this.onSelectIncident?.(next?.properties || properties);
  }
}

function footprintFallbackCollection(collection = EMPTY) {
  return {
    type: "FeatureCollection",
    features: (collection.features || []).map((feature) => {
      const visual = footprintVisualModel(feature.properties);
      return {
        ...feature,
        properties: {
          ...(feature.properties || {}),
          __footprint_color: colorHexFor(feature.properties, "family"),
          __footprint_opacity: visual.fillAlpha / 255,
          __footprint_line_opacity: visual.lineAlpha / 255,
          __footprint_line_width: visual.lineWidth
            + Math.max(
              0,
              Number(feature.properties?.coincident_incident_count || 1)
                - Number(feature.properties?.coincident_incident_index || 0)
                - 1,
            ) * 1.15,
          __footprint_style: visual.key,
        },
      };
    }),
    meta: collection.meta || {},
  };
}

function footprintHistoryFallbackCollection(collection = EMPTY) {
  return {
    type: "FeatureCollection",
    features: (collection.features || []).map((feature) => {
      const visual = footprintHistoryVisualModel(feature.properties);
      const properties = feature.properties || {};
      return {
        ...feature,
        properties: {
          ...properties,
          history_id: `${properties.incident_id || "incident"}:${properties.timeline_bucket || "week"}`,
          __history_color: colorHexFor(properties, "family"),
          __history_opacity: visual.lineAlpha / 255,
          __history_width: visual.lineWidth,
        },
      };
    }),
    meta: collection.meta || {},
  };
}
