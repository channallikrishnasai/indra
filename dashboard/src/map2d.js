// 2D world map (equirectangular so the data grid maps with a plain affine draw). Pan/zoom via d3-zoom.
import { geoEquirectangular, geoPath, geoGraticule10 } from 'd3-geo';
import { zoom as d3zoom, zoomIdentity } from 'd3-zoom';
import { select } from 'd3-selection';
import { mesh } from 'topojson-client';
import { BBOX, LAT0, LON0, bilinear, CELLS } from './field.js';

export const CITIES = [['New Delhi', 28.6, 77.2], ['Mumbai', 19.08, 72.88], ['Kolkata', 22.57, 88.36], ['Chennai', 13.08, 80.27],
  ['Bengaluru', 12.97, 77.59], ['Hyderabad', 17.39, 78.49], ['Ahmedabad', 23.02, 72.57], ['Jaipur', 26.91, 75.79],
  ['Lucknow', 26.85, 80.95], ['Bhubaneswar', 20.3, 85.82], ['Guwahati', 26.14, 91.74], ['Thiruvananthapuram', 8.52, 76.94]];
const SEAS = [['ARABIAN SEA', 15, 64.5], ['BAY OF BENGAL', 15.5, 88.5], ['INDIAN OCEAN', 3, 80]];

export class Map2D {
  constructor(root, { onHover, onPick, onSelect, fieldCanvas, cloudCanvas }) {
    this.c = Object.fromEntries(['mapBase', 'mapField', 'mapWind', 'mapUi'].map(id => [id, document.getElementById(id)]));
    this.ctx = Object.fromEntries(Object.entries(this.c).map(([k, v]) => [k, v.getContext('2d')]));
    this.fieldCanvas = fieldCanvas; this.cloudCanvas = cloudCanvas; this.onHover = onHover; this.onPick = onPick; this.onSelect = onSelect;
    this.t = zoomIdentity; this.active = false; this.dirty = true;
    this.img = new Image(); this.img.src = '/tex/earth-blue-marble.jpg'; this.img.onload = () => this.drawBase();
    this.N = 1800; this.P = new Float32Array(this.N * 3);
    for (let k = 0; k < this.N; k++) this.respawn(k, true);

    this.zoom = d3zoom().scaleExtent([1, 40]).on('zoom', e => { this.t = e.transform; this.drawBase(); this.dirty = true; this.ctx.mapWind.clearRect(0, 0, 1e5, 1e5); });
    select(this.c.mapUi).call(this.zoom).on('dblclick.zoom', null);
    this.c.mapUi.addEventListener('pointermove', e => { const [lat, lon] = this.invert(e.offsetX, e.offsetY); this.onHover({ lat, lon, x: e.clientX, y: e.clientY }); });
    this.c.mapUi.addEventListener('pointerleave', () => this.onHover(null));
    let down = null;
    this.c.mapUi.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    this.c.mapUi.addEventListener('click', e => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;          // a drag, not a click
      const [lat, lon] = this.invert(e.offsetX, e.offsetY);
      e.shiftKey ? this.onPick(lat, lon) : this.onSelect(lat, lon);
    });
    addEventListener('resize', () => this.resize()); this.resize();
  }

  async loadBorders() {
    const topo = await (await fetch('/countries-50m.json')).json();
    this.borders = mesh(topo, topo.objects.countries, (a, b) => a !== b);
    this.coast = mesh(topo, topo.objects.countries, (a, b) => a === b);
    this.drawBase();
  }

  resize() {
    const dpr = Math.min(devicePixelRatio, 2); this.dpr = dpr; this.w = innerWidth; this.h = innerHeight;
    for (const c of Object.values(this.c)) { c.width = this.w * dpr; c.height = this.h * dpr; c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0); }
    this.W0 = Math.max(this.w, this.h * 2);                 // world width at zoom 1 (fills the screen)
    this.zoom.translateExtent([[0, (this.h - this.W0 / 2) / 2], [this.W0, (this.h + this.W0 / 2) / 2]]).extent([[0, 0], [this.w, this.h]]);
    this.drawBase(); this.dirty = true;
  }

  // world -> screen
  x(lon) { return this.t.applyX((lon + 180) / 360 * this.W0); }
  y(lat) { return this.t.applyY((this.h - this.W0 / 2) / 2 + (90 - lat) / 180 * this.W0 / 2); }
  invert(px, py) {
    const wx = this.t.invertX(px), wy = this.t.invertY(py);
    return [90 - (wy - (this.h - this.W0 / 2) / 2) / (this.W0 / 2) * 180, wx / this.W0 * 360 - 180];
  }
  projection() {
    const k = this.t.k, s = this.W0 * k / (2 * Math.PI);
    return geoEquirectangular().scale(s).translate([this.x(0), this.y(0)]).precision(0.3);
  }

  drawBase() {
    if (!this.w) return;
    const g = this.ctx.mapBase, light = this.theme === 'light'; g.clearRect(0, 0, this.w, this.h);
    if (this.img.complete && this.img.naturalWidth) {
      g.drawImage(this.img, this.x(-180), this.y(90), this.x(180) - this.x(-180), this.y(-90) - this.y(90));
      g.fillStyle = light ? 'rgba(235,242,250,0.30)' : 'rgba(2,6,14,0.22)'; g.fillRect(0, 0, this.w, this.h);
    }
    // borders/coastlines go on their own layer, drawn ABOVE the data so geography always reads through
    const b = this.bordersCanvas ??= document.createElement('canvas');
    b.width = this.w * this.dpr; b.height = this.h * this.dpr; const bg = b.getContext('2d'); bg.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const path = geoPath(this.projection(), bg), ink = light ? '20,35,60' : '225,240,255';
    bg.beginPath(); path(geoGraticule10()); bg.strokeStyle = `rgba(${ink},0.07)`; bg.lineWidth = 1; bg.stroke();
    if (this.borders && this.showBorders !== false) {
      bg.beginPath(); path(this.coast); bg.strokeStyle = `rgba(${ink},0.55)`; bg.lineWidth = 0.9; bg.stroke();
      bg.beginPath(); path(this.borders); bg.strokeStyle = `rgba(${ink},0.5)`; bg.lineWidth = 0.8; bg.stroke();
    }
    this.dirty = true;
  }
  setTheme(t) { this.theme = t; this.drawBase(); }

  setWind(day) { this.u = day ? day.subarray(2 * CELLS, 3 * CELLS) : null; this.v = day ? day.subarray(3 * CELLS, 4 * CELLS) : null; }
  respawn(k, rnd) { this.P[k * 3] = BBOX.s + Math.random() * 32; this.P[k * 3 + 1] = BBOX.w + Math.random() * 36; this.P[k * 3 + 2] = rnd ? Math.random() * 80 : 0; }
  setShock(s) { this.shock = s; this.dirty = true; }
  setOutline(f) { this.outline = f; this.dirty = true; }
  flyTo(lat, lon, k = 4.2) {
    const tx = this.w / 2 - (lon + 180) / 360 * this.W0 * k, ty = this.h / 2 - ((this.h - this.W0 / 2) / 2 + (90 - lat) / 180 * this.W0 / 2) * k;
    const from = this.t, to = zoomIdentity.translate(tx, ty).scale(k), t0 = performance.now(), dur = 1800;
    const sel = select(this.c.mapUi);
    const step = now => {
      const p = Math.min(1, (now - t0) / dur), e = p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      const kk = from.k * Math.pow(to.k / from.k, e);
      this.zoom.transform(sel, zoomIdentity.translate(from.x + (to.x - from.x) * e, from.y + (to.y - from.y) * e).scale(kk));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  frame(dt, showWind, showClouds, fieldDirty) {
    if (!this.active) return;
    const x0 = this.x(BBOX.w), y0 = this.y(BBOX.n), w = this.x(BBOX.e) - x0, h = this.y(BBOX.s) - y0;
    if (this.dirty || fieldDirty) {
      const f = this.ctx.mapField; f.clearRect(0, 0, this.w, this.h); f.imageSmoothingEnabled = true; f.imageSmoothingQuality = 'high';
      f.drawImage(this.fieldCanvas, x0, y0, w, h);
      if (this.bordersCanvas) f.drawImage(this.bordersCanvas, 0, 0, this.w, this.h);
      if (showClouds) {
        const off = 0.35 * this.t.k;                                    // cloud shadow, offset away from the sun
        f.save(); f.globalAlpha = 0.35; f.filter = 'brightness(0) blur(2px)'; f.drawImage(this.cloudCanvas, x0 + off, y0 + off * 1.4, w, h); f.restore();
        f.drawImage(this.cloudCanvas, x0, y0, w, h);
      }
      this.dirty = false;
    }
    // wind trails
    const g = this.ctx.mapWind;
    g.globalCompositeOperation = 'destination-in'; g.fillStyle = 'rgba(0,0,0,0.94)'; g.fillRect(0, 0, this.w, this.h);
    g.globalCompositeOperation = 'source-over';
    if (showWind && this.u) {
      const s = Math.min(dt, 0.05) * 0.5, P = this.P; g.lineWidth = 1.5; g.strokeStyle = 'rgba(235,248,255,0.85)'; g.beginPath();
      for (let k = 0; k < this.N; k++) {
        const lat = P[k * 3], lon = P[k * 3 + 1], gi = lat - LAT0, gj = lon - LON0;
        const u = bilinear(this.u, gi, gj), v = bilinear(this.v, gi, gj);
        const nlat = lat + v * s, nlon = lon + u * s / Math.cos(lat * Math.PI / 180);
        g.moveTo(this.x(lon), this.y(lat)); g.lineTo(this.x(nlon), this.y(nlat));
        P[k * 3] = nlat; P[k * 3 + 1] = nlon; P[k * 3 + 2] += 1;
        if (P[k * 3 + 2] > 90 || nlat < BBOX.s || nlat > BBOX.n || nlon < BBOX.w || nlon > BBOX.e) this.respawn(k, false);
      }
      g.stroke();
    }
    // UI overlay: domain box, labels, cities, shock
    const u = this.ctx.mapUi; u.clearRect(0, 0, this.w, this.h);
    if (this.outline) { u.beginPath(); geoPath(this.projection(), u)(this.outline); u.strokeStyle = '#ffd166'; u.lineWidth = 2.2; u.setLineDash([]); u.stroke(); }
    u.setLineDash([6, 5]); u.strokeStyle = 'rgba(70,211,255,0.6)'; u.lineWidth = 1; u.strokeRect(x0, y0, w, h); u.setLineDash([]);
    u.font = '600 11px Inter'; u.textAlign = 'center';
    if (this.t.k > 2 && this.showCities !== false) {
      u.fillStyle = this.theme === 'light' ? 'rgba(20,40,80,0.5)' : 'rgba(200,225,255,0.45)'; u.font = '600 10px Inter';
      for (const [n, la, lo] of SEAS) { u.save(); u.letterSpacing = '3px'; u.fillText(n, this.x(lo), this.y(la)); u.restore(); }
      u.font = '500 11px Inter';
      for (const [n, la, lo] of CITIES) {
        const px = this.x(lo), py = this.y(la);
        u.fillStyle = '#fff'; u.beginPath(); u.arc(px, py, 2.6, 0, 7); u.fill();
        u.fillStyle = 'rgba(255,255,255,0.85)'; u.shadowColor = '#000'; u.shadowBlur = 4; u.fillText(n, px, py - 7); u.shadowBlur = 0;
      }
    }
    if (this.shock) {
      const s = this.shock, p = performance.now() / 1600 % 1, rx = s.radius / 360 * this.W0 * this.t.k / Math.cos(s.lat * Math.PI / 180), ry = s.radius / 180 * this.W0 / 2 * this.t.k;
      u.strokeStyle = '#ff7a45'; u.lineWidth = 2; u.beginPath(); u.ellipse(this.x(s.lon), this.y(s.lat), rx, ry, 0, 0, 7); u.stroke();
      u.strokeStyle = `rgba(255,61,119,${1 - p})`; u.beginPath(); u.ellipse(this.x(s.lon), this.y(s.lat), rx * (0.2 + p * 1.1), ry * (0.2 + p * 1.1), 0, 0, 7); u.stroke();
    }
  }
}
