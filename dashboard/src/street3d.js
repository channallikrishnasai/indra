// Street 3D: a tilted MapLibre map you can zoom from all of India down to single streets, with 3D buildings,
// terrain, satellite imagery and INDRA's live field draped on top. Needs internet for tiles (dashboard only;
// the server and the grader stay offline). Created lazily, so nothing is fetched until the view is opened.
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { BBOX, CELLS, NLON, LAT0, LON0 } from './field.js';

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
    m.addSource('state', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    m.addLayer({ id: 'state-fill', type: 'fill', source: 'state', paint: { 'fill-color': '#ffd166', 'fill-opacity': 0.07 } }, firstSymbol);
    m.addLayer({ id: 'state-line', type: 'line', source: 'state', paint: { 'line-color': '#ffd166', 'line-width': 2.4, 'line-opacity': 0.9 } }, firstSymbol);
    this.initRain();
    m.setLight({ anchor: 'map', position: [1.4, 210, 35], intensity: 0.45 });
    this.setSatellite(this.satellite); this.setActive(!!this.active);
  }

  setSatellite(on) {
    this.satellite = on;
    if (this.map?.getLayer('sat')) this.map.setLayoutProperty('sat', 'visibility', on ? 'visible' : 'none');
  }

  /** fly to one state's own bounds (its own street view), outlined in gold */
  async flyToState(name, feature, bbox) {
    await this.ensure();
    const m = this.map;
    m.getSource('state').setData({ type: 'FeatureCollection', features: feature ? [feature] : [] });
    m.stop(); m.fitBounds(bbox, { padding: { top: 70, bottom: 110, left: 60, right: 380 }, pitch: 55, bearing: -18, maxZoom: 9.5, duration: 3600, essential: true });
  }

  // ---- rain: streaks anchored to the grid cells where the forecast has rain, falling over the 3D streets
  initRain() {
    this.rainCv = Object.assign(document.createElement('canvas'), { className: 'rainfx' });
    this.container.appendChild(this.rainCv);
    this.rp = new Float32Array(1400 * 4);                      // per drop: lat, lon, phase (0..1), speed
    this.rainT = performance.now();
  }
  /** day = [6*CELLS] physical for the current forecast day; only precipitation and wind are used */
  setWeather(day) {
    this.day = day;
    const tp = day.subarray(CELLS, 2 * CELLS), cdf = new Float32Array(CELLS); let s = 0;
    for (let k = 0; k < CELLS; k++) { s += tp[k] > 0.5 ? tp[k] : 0; cdf[k] = s; }
    this.rainCdf = s > 0 ? cdf : null; this.rainTot = s;
    if (this.rainCdf) for (let p = 0; p < this.rp.length; p += 4) this.anchor(p);
  }
  anchor(p) {
    const x = Math.random() * this.rainTot; let lo = 0, hi = CELLS - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (this.rainCdf[m] < x) lo = m + 1; else hi = m; }
    this.rp[p] = LAT0 + Math.floor(lo / NLON) + Math.random() - 0.5; this.rp[p + 1] = LON0 + lo % NLON + Math.random() - 0.5;
    this.rp[p + 2] = Math.random(); this.rp[p + 3] = 0.9 + Math.random() * 0.5;
  }
  drawRain(now) {
    const c = this.rainCv, m = this.map, W = this.container.clientWidth, H = this.container.clientHeight;
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    const g = c.getContext('2d'); g.clearRect(0, 0, W, H);
    if (!this.rainCdf || m.getZoom() < 7) return;
    const dt = Math.min(0.1, (now - this.rainT) / 1000); this.rainT = now;
    const tu = this.day.subarray(2 * CELLS, 3 * CELLS), tv = this.day.subarray(3 * CELLS, 4 * CELLS), tp = this.day.subarray(CELLS, 2 * CELLS);
    const wiggle = Math.min(1, Math.max(0, m.getZoom() - 6) / 3);             // streaks grow in as you zoom in
    g.lineCap = 'round'; g.lineWidth = 1.1;
    for (let p = 0; p < this.rp.length; p += 4) {
      this.rp[p + 2] = (this.rp[p + 2] + dt * this.rp[p + 3] * 0.55) % 1;      // falling
      const xy = m.project([this.rp[p + 1], this.rp[p + 0]]);
      const y = (xy.y + this.rp[p + 2] * H) % H, x = xy.x;
      if (x < -20 || x > W + 20) continue;
      const k = Math.floor(this.rp[p] - LAT0 + 0.5) * NLON + Math.floor(this.rp[p + 1] - LON0 + 0.5), kk = Math.min(CELLS - 1, Math.max(0, k));
      const rain = tp[kk], int = Math.min(1, rain / 20);
      if (rain < 0.5) { this.anchor(p); continue; }                             // re-spawn where it is raining now
      const len = (8 + 14 * int) * wiggle + 2, slant = -(tu[kk] * 0.35);
      g.strokeStyle = `rgba(200,220,255,${0.18 + 0.55 * int})`;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x + slant * len * 0.4, y + len); g.stroke();
    }
  }

  /** only re-read the live field canvas while this view is on screen */
  setActive(on) {
    this.active = on; const src = this.map?.getSource('indra'); if (src) on ? src.play() : src.pause(); if (on) this.map?.resize();
    if (on && !this.rainLoop && this.map) { const tick = now => { if (!this.active) { this.rainLoop = null; return; } this.drawRain(now); this.rainLoop = requestAnimationFrame(tick); }; this.rainLoop = requestAnimationFrame(tick); }
  }

  /** fly to the state first (wide, gently tilted), then sweep down to street level */
  async flyToCity(lat, lon) {
    await this.ensure();
    const m = this.map;
    m.stop();
    m.flyTo({ center: [lon, lat], zoom: 7, pitch: 35, bearing: 0, duration: 2600, essential: true });
    m.once('moveend', () => m.flyTo({ center: [lon, lat], zoom: 16, pitch: 65, bearing: -25, duration: 4200, curve: 1.2, essential: true }));
  }

  /** JPEG of the current frame; read inside a render event because the WebGL buffer is cleared afterwards */
  snapshot() {
    const m = this.map; if (!m) return Promise.resolve(null);
    return new Promise(res => { m.once('render', () => res(m.getCanvas().toDataURL('image/jpeg', 0.85))); m.triggerRepaint(); });
  }

  async overview() { await this.ensure(); this.map.flyTo({ center: [81, 22], zoom: 4.3, pitch: 20, bearing: 0, duration: 2000 }); }
}
