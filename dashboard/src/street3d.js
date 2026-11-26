// Street 3D: a tilted MapLibre map you can zoom from all of India down to single streets, with 3D buildings,
// terrain, satellite imagery and INDRA's live field draped on top. Needs internet for tiles (dashboard only;
// the server and the grader stay offline). Created lazily, so nothing is fetched until the view is opened.
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { BBOX } from './field.js';

const STYLE = 'https://tiles.openfreemap.org/styles/liberty';                                  // OpenStreetMap vector tiles, no key
const TERRAIN = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png';      // AWS open terrain tiles
const SAT = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

export class Street3D {
  constructor(container, fieldCanvas, { onHover, onPick }) {
    this.container = container; this.fieldCanvas = fieldCanvas; this.onHover = onHover; this.onPick = onPick; this.satellite = true;
  }

  ensure() {
    if (this.map) return this.ready;
    const m = this.map = new maplibregl.Map({
      container: this.container, style: STYLE, center: [80, 22], zoom: 4.2, pitch: 0, maxPitch: 80, maxZoom: 20,
      attributionControl: { compact: true }, antialias: true,
    });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    m.on('mousemove', e => this.onHover({ lat: e.lngLat.lat, lon: e.lngLat.lng, x: e.originalEvent.clientX, y: e.originalEvent.clientY }));
    m.on('mouseout', () => this.onHover(null));
    m.on('click', e => { if (e.originalEvent.shiftKey) this.onPick(e.lngLat.lat, e.lngLat.lng); });
    this.ready = new Promise(res => m.on('load', () => { this.decorate(); res(); }));
    return this.ready;
  }

  decorate() {
    const m = this.map, layers = m.getStyle().layers;
    const firstSymbol = layers.find(l => l.type === 'symbol')?.id, firstLine = layers.find(l => l.type === 'line')?.id;
    m.addSource('dem', { type: 'raster-dem', tiles: [TERRAIN], encoding: 'terrarium', tileSize: 256, maxzoom: 15 });
    m.addSource('dem-shade', { type: 'raster-dem', tiles: [TERRAIN], encoding: 'terrarium', tileSize: 256, maxzoom: 15 });
    m.setTerrain({ source: 'dem', exaggeration: 1.5 });
    m.addSource('sat', { type: 'raster', tiles: [SAT], tileSize: 256, maxzoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics' });
    m.addLayer({ id: 'sat', type: 'raster', source: 'sat', paint: { 'raster-saturation': 0.1, 'raster-contrast': 0.08 } }, firstLine);
    m.addLayer({ id: 'shade', type: 'hillshade', source: 'dem-shade', paint: { 'hillshade-exaggeration': 0.35, 'hillshade-shadow-color': '#0b1220' } }, firstLine);
    // INDRA's live field, re-read every frame from the same canvas the globe uses; fades as you zoom to street level
    // because one forecast cell is ~110 km wide
    m.addSource('indra', { type: 'canvas', canvas: this.fieldCanvas, animate: true,
      coordinates: [[BBOX.w, BBOX.n], [BBOX.e, BBOX.n], [BBOX.e, BBOX.s], [BBOX.w, BBOX.s]] });
    m.addLayer({ id: 'indra', type: 'raster', source: 'indra', paint: { 'raster-opacity': ['interpolate', ['linear'], ['zoom'], 4, 0.7, 7, 0.4, 10, 0.2, 14, 0.06], 'raster-fade-duration': 0 } }, firstSymbol);
    if (!m.getLayer('building-3d')) {                                         // ensure extruded buildings exist
      m.addLayer({ id: 'building-3d', type: 'fill-extrusion', source: 'openmaptiles', 'source-layer': 'building', minzoom: 13,
        paint: { 'fill-extrusion-color': '#d9d4cc', 'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 8],
          'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0], 'fill-extrusion-opacity': 0.92 } });
    } else m.setPaintProperty('building-3d', 'fill-extrusion-opacity', 0.92);
    m.setLight({ anchor: 'map', position: [1.4, 210, 35], intensity: 0.45 });
    this.setSatellite(this.satellite); this.setActive(!!this.active);
  }

  setSatellite(on) {
    this.satellite = on;
    if (this.map?.getLayer('sat')) this.map.setLayoutProperty('sat', 'visibility', on ? 'visible' : 'none');
  }

  /** only re-read the live field canvas while this view is on screen */
  setActive(on) { this.active = on; const src = this.map?.getSource('indra'); if (src) on ? src.play() : src.pause(); if (on) this.map?.resize(); }

  /** fly to the state first (wide, gently tilted), then sweep down to street level */
  async flyToCity(lat, lon) {
    await this.ensure();
    const m = this.map;
    m.stop();
    m.flyTo({ center: [lon, lat], zoom: 7, pitch: 35, bearing: 0, duration: 2600, essential: true });
    m.once('moveend', () => m.flyTo({ center: [lon, lat], zoom: 16, pitch: 65, bearing: -25, duration: 4200, curve: 1.2, essential: true }));
  }

  async overview() { await this.ensure(); this.map.flyTo({ center: [81, 22], zoom: 4.3, pitch: 20, bearing: 0, duration: 2000 }); }
}
