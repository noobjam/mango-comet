import { alphaForState, applyVisualProperties, colorFor } from "./palette.js";
import { buildEvolutionModel, evolutionDeckLayers, evolutionGeoJson } from "./map-evolution.js";

const EMPTY = { type: "FeatureCollection", features: [], meta: {} };

export class MapView {
  constructor({ container, config, bounds, onReady, onHover, onSelect, onViewportChange }) {
    this.container = container;
    this.config = config;
    this.bounds = bounds;
    this.onReady = onReady;
    this.onHover = onHover;
    this.onSelect = onSelect;
    this.onViewportChange = onViewportChange;
    this.frame = EMPTY;
    this.trail = EMPTY;
    this.colorMode = "family";
    this.showHistory = true;
    this.selectedFieldId = "";
    this.evolution = null;
    this.evolutionBucket = "";
    this.overlay = null;
    this.ready = false;
    this.fallbackEventsBound = false;
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

  setData(frame, trail, evolution = this.evolution, evolutionBucket = this.evolutionBucket) {
    this.frame = frame || EMPTY;
    this.trail = trail || EMPTY;
    this.evolution = evolution || null;
    this.evolutionBucket = String(evolutionBucket || "");
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

  setEvolution(payload, bucket) {
    this.evolution = payload || null;
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

  render() {
    if (!this.ready) return;
    if (this.overlay) {
      this.overlay.setProps({ layers: this.deckLayers() });
      return;
    }
    this.renderFallback();
  }

  deckLayers() {
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
    layers.push(new deck.GeoJsonLayer({
      id: "story-current-fields",
      data: this.frame,
      pickable: true,
      filled: true,
      stroked: true,
      lineWidthUnits: "pixels",
      getFillColor: (feature) => colorFor(
        feature.properties,
        this.colorMode,
        alphaForState(feature.properties, 188),
      ),
      getLineColor: (feature) => String(feature.properties?.field_id) === this.selectedFieldId
        ? [255, 255, 255, 255]
        : [5, 20, 15, 210],
      getLineWidth: (feature) => {
        if (String(feature.properties?.field_id) === this.selectedFieldId) return 3;
        const risk = String(feature.properties?.current_risk_band || feature.properties?.max_risk_band || "").toUpperCase();
        if (this.colorMode === "risk" && ["HIGH", "SEVERE"].includes(risk)) return 1.8;
        return this.colorMode === "risk" && risk === "MEDIUM" ? 1.1 : 0.7;
      },
      lineWidthMinPixels: 0.7,
      updateTriggers: {
        getFillColor: [this.colorMode],
        getLineColor: [this.selectedFieldId],
        getLineWidth: [this.selectedFieldId, this.colorMode],
      },
      onHover: (info) => this.onHover?.(info.object?.properties || null, { x: info.x, y: info.y }),
      onClick: (info) => this.onSelect?.(info.object?.properties || null),
    }));
    layers.push(...this.evolutionLayers());
    return layers;
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
    const current = applyVisualProperties(this.frame, this.colorMode, false);
    const history = this.showHistory ? applyVisualProperties(this.priorTrail(), this.colorMode, true) : EMPTY;
    this.setSource("story-history-fallback", history);
    this.setSource("story-current-fallback", current);
    this.setSource("activity-center-fallback", this.evolutionGeoJson());
    this.ensureFallbackLayers();
    this.map.setLayoutProperty("story-history-fill", "visibility", this.showHistory ? "visible" : "none");
    this.map.setPaintProperty("story-current-line", "line-width", [
      "case",
      ["==", ["get", "field_id"], this.selectedFieldId], 3,
      ["all", ["==", this.colorMode, "risk"], ["in", ["coalesce", ["get", "current_risk_band"], ["get", "max_risk_band"]], ["literal", ["HIGH"]]]], 1.8,
      ["all", ["==", this.colorMode, "risk"], ["in", ["coalesce", ["get", "current_risk_band"], ["get", "max_risk_band"]], ["literal", ["LOW-MED", "MED-HIGH"]]]], 1.1,
      0.7,
    ]);
    this.map.setPaintProperty("story-current-line", "line-color", [
      "case", ["==", ["get", "field_id"], this.selectedFieldId], "#ffffff", "#05140f",
    ]);
  }

  evolutionGeoJson() {
    return evolutionGeoJson(buildEvolutionModel(this.evolution, this.evolutionBucket));
  }

  setSource(id, data) {
    const source = this.map.getSource(id);
    if (source) source.setData(data);
    else this.map.addSource(id, { type: "geojson", data, promoteId: "field_id" });
  }

  ensureFallbackLayers() {
    if (!this.map.getLayer("story-history-fill")) this.map.addLayer({
      id: "story-history-fill", type: "fill", source: "story-history-fallback",
      paint: { "fill-color": ["get", "__story_color"], "fill-opacity": ["get", "__story_opacity"] },
    });
    if (!this.map.getLayer("story-current-fill")) this.map.addLayer({
      id: "story-current-fill", type: "fill", source: "story-current-fallback",
      paint: { "fill-color": ["get", "__story_color"], "fill-opacity": ["get", "__story_opacity"] },
    });
    if (!this.map.getLayer("story-current-line")) this.map.addLayer({
      id: "story-current-line", type: "line", source: "story-current-fallback",
      paint: { "line-color": ["case", ["==", ["get", "field_id"], this.selectedFieldId], "#ffffff", "#05140f"], "line-opacity": 0.85, "line-width": 0.7 },
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
  }
}
