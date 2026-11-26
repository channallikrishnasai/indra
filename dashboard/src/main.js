import { NC, NLAT, NLON, CELLS, LAYERS, LAT0, LON0, BBOX, TEX_W, TEX_H, CLOUD_W, CLOUD_H, idx, layerGrid, lerpDay, paintField, paintClouds, paintLegend, bilinear, inDomain } from './field.js';
import { Globe, ll2v } from './globe.js';
import { Map2D, CITIES } from './map2d.js';
import { impacts, landMeanSeries, propagation } from './impact.js';
import { generateAlerts, diffAlerts, ADVICE, SEVERITIES } from './alerts.js';
import { loadStates, stateAt, stateStats, stateFeature, stateMask, bboxOf } from './state.js';

// ?api=http://host:port overrides; 127.0.0.1 avoids Windows' slow IPv6-first "localhost" lookup
const API = new URLSearchParams(location.search).get('api') || `http://${!location.hostname || location.hostname === 'localhost' ? '127.0.0.1' : location.hostname}:8000`;
// dashboard-only data helper (tools/window_server.py) for custom start dates; ?data=http://host:port overrides
const DATA_API = new URLSearchParams(location.search).get('data') || 'http://127.0.0.1:8100';
const STEPS = 10, DAY = 6 * CELLS, STD = [13.46, 7.55, 2.9, 2.43, 619, 20.49];
const $ = id => document.getElementById(id);
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

const COMMIT = typeof __COMMIT__ === 'string' ? __COMMIT__ : 'dev';   // set by vite.config.js at build time
const S = {
  view: 'globe', mode: 'prime', layer: 't2m', layers: new Set(['t2m']), day: 0, playing: false, speed: 1,
  opts: { wind: true, clouds: true, rain: true, borders: true, cities: true },
  windows: [], win: null, init: null, truth: null, land: null, clim: null,
  prime: null, multi: null, spread: null, shock: null, preset: 'bob', lastMs: 0, nullDiff: null,
  risk: { prime: null, multi: null }, hasRisk: true, tour: null,
};

// ---------------------------------------------------------------- API
async function post(path, body) {
  const r = await fetch(API + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });
  if (!r.ok) throw new Error(`${path} ${r.status}: ${(await r.text()).slice(0, 160)}`);
  return r.json();
}
const nest = (flat, ...shape) => {
  const rec = (off, d) => d === shape.length - 1 ? Array.from(flat.subarray(off, off + shape[d])) : Array.from({ length: shape[d] }, (_, k) => rec(off + k * shape.slice(d + 1).reduce((a, b) => a * b, 1), d + 1));
  return rec(0, 0);
};
const flatten = nested => Float32Array.from(nested.flat(Infinity));

async function simulate(initFlat, pertFlat) {
  const t0 = performance.now();
  const res = await post('/simulate', { init_states: nest(initFlat, 3, NC, NLAT, NLON), doy_sincos: S.win.doy_sincos, steps: STEPS, perturbation: pertFlat ? nest(pertFlat, NC, NLAT, NLON) : null });
  const flat = flatten(res.rollout); S.lastMs = res.inference_ms; S.roundTrip = performance.now() - t0;
  const last = initFlat.slice(2 * DAY, 3 * DAY);
  if (pertFlat) for (let k = 0; k < DAY; k++) last[k] += pertFlat[k];
  return [last, ...Array.from({ length: STEPS }, (_, d) => flat.slice(d * DAY, (d + 1) * DAY))];
}
/** INDRA-Risk probabilities for a rollout -> [day0..day10] arrays of [2*CELLS] (day 0 copies day 1) */
async function riskFor(days) {
  if (!S.hasRisk) return null;
  try {
    const flat = new Float32Array(STEPS * DAY); days.slice(1).forEach((d, k) => flat.set(d, k * DAY));
    const r = await post('/risk', { rollout: nest(flat, STEPS, NC, NLAT, NLON), doy_sincos: S.win.doy_sincos });
    const p = flatten(r.risk), out = Array.from({ length: STEPS }, (_, d) => p.slice(d * 2 * CELLS, (d + 1) * 2 * CELLS));
    return [out[0], ...out];
  } catch { S.hasRisk = false; return null; }
}

// ---------------------------------------------------------------- shocks
const PRESETS = {
  bob:    { name: 'Warm Bay of Bengal', sub: '+3 K, +15% moisture', box: [8, 20, 82, 94], parts: [['t2m', 3], ['tcwv', 15]] },
  stall:  { name: 'Monsoon stall', sub: '−40% Arabian Sea wind', box: [8, 20, 65, 75], parts: [['wind', -40]] },
  heat:   { name: 'Heat dome', sub: '+4 K over NW India', lat: 27, lon: 74, radius: 5, parts: [['t2m', 4]] },
  drought:{ name: 'Dry Western Ghats', sub: '−40% moisture', lat: 14, lon: 75, radius: 3.5, parts: [['tcwv', -40]] },
  cyclone:{ name: 'Cyclone seed', sub: '−8 hPa, +20% moisture', lat: 15, lon: 88, radius: 3, parts: [['msl', -8], ['tcwv', 20]] },
  custom: { name: 'Custom shock', sub: 'use the controls below', custom: true },
};
const VARS = {
  t2m: { min: -6, max: 6, step: 0.1, unit: ' K', def: 3 }, tcwv: { min: -60, max: 60, step: 1, unit: '%', def: 20 },
  wind: { min: -80, max: 80, step: 1, unit: '%', def: -40 }, tp: { min: -40, max: 80, step: 1, unit: ' mm/d', def: 30 },
  msl: { min: -15, max: 15, step: 0.5, unit: ' hPa', def: -6 },
};
function currentShock() {
  const p = PRESETS[S.preset];
  if (!p.custom) {
    if (p.box) { const [s, n, w, e] = p.box; return { box: p.box, lat: (s + n) / 2, lon: (w + e) / 2, radius: (n - s) / 2, parts: p.parts, name: p.name }; }
    return { lat: p.lat, lon: p.lon, radius: p.radius, parts: p.parts, name: p.name };
  }
  return { lat: +$('shockLat').value, lon: +$('shockLon').value, radius: +$('shockRad').value, parts: [[$('shockVar').value, +$('shockMag').value]], name: 'Custom shock' };
}
/** physical delta [6,32,36] added to the last input day (same convention as the judge) */
function buildPerturbation(sh, last) {
  const d = new Float32Array(DAY); let touched = 0;
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const lat = LAT0 + i, lon = LON0 + j, k = idx(i, j); let w;
    if (sh.box) w = lat >= sh.box[0] && lat <= sh.box[1] && lon >= sh.box[2] && lon <= sh.box[3] ? 1 : 0;
    else {
      const r = Math.hypot(lat - sh.lat, (lon - sh.lon) * Math.cos(lat * Math.PI / 180)) / sh.radius;
      w = r >= 1 ? 0 : r < 0.65 ? 1 : 0.5 * (1 + Math.cos(Math.PI * (r - 0.65) / 0.35));
    }
    if (!w) continue; touched++;
    for (const [v, m] of sh.parts) {
      if (v === 't2m') d[k] += m * w;
      else if (v === 'tp') d[CELLS + k] += m * w;
      else if (v === 'msl') d[4 * CELLS + k] += m * 100 * w;
      else if (v === 'tcwv') d[5 * CELLS + k] += m / 100 * last[5 * CELLS + k] * w;
      else if (v === 'wind') { d[2 * CELLS + k] += m / 100 * last[2 * CELLS + k] * w; d[3 * CELLS + k] += m / 100 * last[3 * CELLS + k] * w; }
    }
  }
  return { delta: d, touched };
}

// ---------------------------------------------------------------- views
const fieldCanvas = Object.assign(document.createElement('canvas'), { width: TEX_W, height: TEX_H });
const cloudCanvas = Object.assign(document.createElement('canvas'), { width: CLOUD_W, height: CLOUD_H });
const tooltip = $('tooltip');
const globe = new Globe($('globe'), { onHover: hover, onPick: pick, onSelect: selectState });
const map = new Map2D($('map2d'), { onHover: hover, onPick: pick, onSelect: selectState, fieldCanvas, cloudCanvas });
// MapLibre (~800 kB) is only downloaded the first time Street 3D is opened
let streetImpl = null;
const street = {
  get map() { return streetImpl?.map; },
  async load() {
    if (!streetImpl) { const { Street3D } = await import('./street3d.js'); streetImpl = new Street3D($('map3d'), fieldCanvas, { onHover: hover, onPick: pick }); streetImpl.setSatellite($('optSat').checked); }
    return streetImpl;
  },
  async ensure() { await (await this.load()).ensure(); },
  setActive(on) { streetImpl?.setActive(on); },
  setSatellite(on) { streetImpl?.setSatellite(on); },
  async flyToCity(lat, lon) { (await this.load()).flyToCity(lat, lon); },
  async overview() { (await this.load()).overview(); },
  snapshot() { return streetImpl ? streetImpl.snapshot() : Promise.resolve(null); },
  async flyToState(name) { (await this.load()).flyToState(name, stateFeature(name), bboxOf(name)); },
  setWeather(day) { streetImpl?.setWeather(day); },
};

function pick(lat, lon) {
  if (!inDomain(lat, lon)) return toast(`(${lat.toFixed(1)}°, ${lon.toFixed(1)}°) is outside INDRA's simulated domain (6–38°N, 65–101°E).`);
  S.preset = 'custom'; $('shockLat').value = lat.toFixed(1); $('shockLon').value = lon.toFixed(1); syncShockUI(); toast('Shock centre moved. Press “Run multiverse”.', 'info');
}

const src = () => S.mode === 'multi' ? (S.multi || S.prime) : S.mode === 'truth' ? S.truth : S.prime;
function dayAt(arr, t) { if (!arr) return null; const k = Math.min(Math.floor(t), STEPS - 1), f = t - k; return f < 1e-3 ? arr[k] : lerpDay(arr[k], arr[Math.min(k + 1, STEPS)], f); }
function gridFor(which, t) {
  const L = LAYERS[S.layer];
  if (L.ml) {
    const r = dayAt(S.risk[which], t); if (!r) return null;
    const g = new Float32Array(CELLS); for (let k = 0; k < CELLS; k++) g[k] = r[L.risk * CELLS + k] * 100; return g;
  }
  const d = dayAt(which === 'multi' ? S.multi : which === 'truth' ? S.truth : S.prime, t); return d && layerGrid(d, S.layer);
}

let fieldDirty = true, lastPaint = -1, cloudT = 0;
const layerTmp = Object.assign(document.createElement('canvas'), { width: TEX_W, height: TEX_H });   // scratch layer for overlays
function repaint() {
  const t = S.day, L = LAYERS[S.layer]; let grid, weatherDay;
  if (S.mode === 'delta' && S.multi) {
    const a = gridFor('multi', t), b = gridFor('prime', t); grid = a && b && a.map((v, k) => v - b[k]); weatherDay = dayAt(S.multi, t);
  } else if (S.mode === 'chance' && S.chance) {
    grid = S.chance[Math.round(t)]; weatherDay = dayAt(S.prime, t);
  } else if (S.mode === 'spread' && S.spread) {
    grid = S.spread[Math.round(t)]; weatherDay = dayAt(S.prime, t);
  } else {
    weatherDay = dayAt(src(), t);
    grid = L.ml ? gridFor(S.mode === 'multi' && S.multi ? 'multi' : 'prime', t) : weatherDay && layerGrid(weatherDay, S.layer);
  }
  if (!weatherDay) return;
  const ctx = fieldCanvas.getContext('2d'); ctx.clearRect(0, 0, TEX_W, TEX_H);
  const overlay = !L.ml && S.mode !== 'delta' && S.mode !== 'spread' && S.mode !== 'chance' && S.layers.size > 1 && weatherDay;
  if (overlay) {                                                  // several weather layers at once: composite each one
    for (const k of S.layers) { layerTmp.getContext('2d').clearRect(0, 0, TEX_W, TEX_H); paintField(layerTmp.getContext('2d'), layerGrid(weatherDay, k), { layer: k, mode: 'abs' }); ctx.drawImage(layerTmp, 0, 0); }
  } else if (grid) paintField(ctx, grid, { layer: S.mode === 'chance' ? 'hrc' : S.layer, mode: S.mode === 'delta' ? 'delta' : S.mode === 'spread' ? 'spread' : 'abs' });
  const cc = cloudCanvas.getContext('2d'); cc.clearRect(0, 0, CLOUD_W, CLOUD_H); if (S.opts.clouds && S.mode !== 'chance') paintClouds(cc, weatherDay, cloudT);   // clouds would hide the chance cells
  for (const [dst, from] of [[globe.fieldCanvas, fieldCanvas], [globe.cloudCanvas, cloudCanvas]]) { const g = dst.getContext('2d'); g.clearRect(0, 0, dst.width, dst.height); g.drawImage(from, 0, 0); }
  globe.field.tex.needsUpdate = true; globe.clouds.tex.needsUpdate = true;
  globe.setDay(weatherDay); map.setWind(weatherDay); street.setWeather(weatherDay); S.curGrid = grid; S.curDay = weatherDay;
  if (S.win) { const dt = new Date(S.win.date); dt.setDate(dt.getDate() + Math.round(t)); globe.setDate(dt); }
  fieldDirty = true;
}

function updateTimelineLabels() {
  if (!S.win) return;
  const d = Math.round(S.day), date = new Date(S.win.date); date.setDate(date.getDate() + d);
  $('dayLabel').textContent = d === 0 ? 'Day 0 · initial state' : `Day +${d}`;
  renderCity();
  $('dateLabel').textContent = date.toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }) + (S.mode !== 'prime' && S.mode !== 'truth' && S.shock ? ` · ${S.shock.name}` : '');
}

function hover(h) {
  if (!h || !inDomain(h.lat, h.lon) || !S.curDay) { tooltip.hidden = true; return; }
  const gi = h.lat - LAT0, gj = h.lon - LON0, rows = [], day = S.curDay;
  for (const [key, L] of Object.entries(LAYERS)) {
    if (L.ml) continue;
    const g = new Float32Array(CELLS); for (let k = 0; k < CELLS; k++) g[k] = L.value(day, k);
    rows.push(`<tr><td>${L.name}</td><td>${bilinear(g, gi, gj).toFixed(1)} ${L.unit}</td></tr>`);
  }
  const r = dayAt(S.risk[S.mode === 'multi' || S.mode === 'delta' ? 'multi' : 'prime'] || S.risk.prime, S.day);
  if (r) for (const [n, c] of [['Flood risk (ML)', 0], ['Heat risk (ML)', 1]]) rows.push(`<tr><td>${n}</td><td>${(bilinear(r.subarray(c * CELLS, (c + 1) * CELLS), gi, gj) * 100).toFixed(0)} %</td></tr>`);
  const extra = S.mode === 'chance' && S.curGrid ? `<tr><td>Heavy-rain chance</td><td>${bilinear(S.curGrid, gi, gj).toFixed(0)} % of runs</td></tr>` : (S.mode === 'delta' || S.mode === 'spread') && S.curGrid ? `<tr><td>${S.mode === 'delta' ? 'Δ shock' : 'Ensemble σ'}</td><td>${bilinear(S.curGrid, gi, gj).toFixed(2)} ${LAYERS[S.layer].unit}</td></tr>` : '';
  tooltip.innerHTML = `<b>${h.lat.toFixed(1)}°N ${h.lon.toFixed(1)}°E</b><table>${rows.join('')}${extra}</table>`;
  tooltip.hidden = false; tooltip.style.left = Math.min(h.x + 16, innerWidth - 200) + 'px'; tooltip.style.top = Math.min(h.y + 16, innerHeight - 220) + 'px';
}

// ---------------------------------------------------------------- right panel
const fmt = (v, d = 1) => Number.isFinite(v) ? v.toLocaleString('en-IN', { maximumFractionDigits: d, minimumFractionDigits: d }) : '–';
function card(k, v, unit, color, deltaTxt, pct, wide, tip) {
  return `<div class="card${wide ? ' wide' : ''}" style="--c:${color}"${tip ? ` title="${tip}"` : ''}><div class="k">${k}</div><div class="v">${v} <small>${unit}</small></div>${deltaTxt || ''}${pct != null ? `<div class="bar"><i style="width:${Math.max(2, Math.min(100, pct))}%"></i></div>` : ''}</div>`;
}
function dTxt(a, b, unit, d = 1) {
  if (b == null) return '';
  const x = b - a; if (Math.abs(x) < 10 ** -d / 2) return `<div class="d">multiverse: no change</div>`;
  return `<div class="d ${x > 0 ? 'up' : 'down'}">multiverse ${x > 0 ? '+' : ''}${fmt(x, d)}${unit}</div>`;
}
function renderImpacts() {
  if (!S.prime || !S.clim || !S.land) return;
  const P = impacts(S.prime, S.clim, S.land), M = S.multi ? impacts(S.multi, S.clim, S.land) : null, ens = S.spreadSummary;
  $('cards').innerHTML = [
    card('Heatwave exposure', fmt(P.heatPeak), '% land', '#ff5a3d', dTxt(P.heatPeak, M?.heatPeak, ' pts'), P.heatPeak),
    card('Flood risk area', fmt(P.floodArea / 1000, 0), 'k km²', '#5b8cff', dTxt(P.floodArea / 1000, M && M.floodArea / 1000, 'k km²', 0), P.floodArea / 20000),
    card('Crop water stress', fmt(P.cropStress), '%', '#f5b342', dTxt(P.cropStress, M?.cropStress, ' pts'), P.cropStress),
    card('Rain vs normal', (P.rainDeparture > 0 ? '+' : '') + fmt(P.rainDeparture, 0), '%', '#46d3ff', dTxt(P.rainDeparture, M?.rainDeparture, ' pts', 0), 50 + P.rainDeparture / 2),
    card('Wind power 100 m', fmt(P.windPower, 0), 'W/m²', '#3ddc97', dTxt(P.windPower, M?.windPower, ' W/m²', 0), P.windPower / 4),
    card('Peak daily rain', fmt(P.peakRain, 0), 'mm/day', '#b23cff', dTxt(P.peakRain, M?.peakRain, ' mm', 0), P.peakRain / 3),
    card('Humid heat', fmt(P.humidHeat), '% land', '#ff8a3d', dTxt(P.humidHeat, M?.humidHeat, ' pts'), P.humidHeat, false, 'Peak-day share of land with humidex >= 40 (great discomfort). Daily-mean temperature with dew point estimated from column moisture: an approximate screening index.'),
    card('Monsoon progress', `${fmt(P.rainTotal, 0)} / ${fmt(P.rainNormal, 0)}`, 'mm', '#2fb8ff', dTxt(P.rainTotal, M?.rainTotal, ' mm', 0) || `<div class="d">10-day rain vs normal</div>`, P.rainNormal > 0 ? P.rainTotal / P.rainNormal * 100 : 0, false, 'Cumulative 10-day land-mean rain against the normal for these dates.'),
    card('Ensemble confidence', ens ? fmt(ens.conf, 0) : '—', ens ? '%' : '', '#9c8fe0', ens ? `<div class="d">day-10 σ ${fmt(ens.sigma, 2)} ${LAYERS[ens.layer].unit} (${LAYERS[ens.layer].name.toLowerCase()})</div>` : `<div class="d">run the 5-member ensemble</div>`, ens?.conf, true, 'Agreement among the 5 perturbed runs (spread), not a measure of accuracy.'),
  ].join('');
  renderRisk(); renderChart(); renderCity(); renderState(); renderTicker();
}

function renderRisk() {
  const R = S.risk[S.multi && S.mode !== 'prime' && S.mode !== 'truth' ? 'multi' : 'prime'];
  if (!R || !S.land) { $('riskBars').innerHTML = `<span class="muted small" style="grid-column:1/-1">${S.hasRisk ? 'computing…' : 'risk model not loaded (train_risk.py)'}</span>`; return; }
  const row = (name, c, col) => {
    let h = `<span>${name}</span>`;
    for (let d = 1; d <= STEPS; d++) {
      let mx = 0; for (let k = 0; k < CELLS; k++) if (S.land[k] > 0.5) mx = Math.max(mx, R[d][c * CELLS + k]);
      h += `<b style="background:color-mix(in srgb, ${col} ${Math.round(15 + mx * 85)}%, transparent)" title="Day +${d}: peak probability ${Math.round(mx * 100)}%">${Math.round(mx * 100)}</b>`;
    }
    return h;
  };
  $('riskBars').innerHTML = row('Flood', 0, '#7b3dff') + row('Heat', 1, '#ff6a1a') + `<span></span>` + Array.from({ length: STEPS }, (_, d) => `<em>+${d + 1}</em>`).join('');
}

function line(pts, color, dash) { return `<polyline fill="none" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ''} stroke-linejoin="round" points="${pts}"/>`; }
function renderChart() {
  if (!S.land || !S.prime) return;
  const L = LAYERS[S.layer].ml ? LAYERS.t2m : LAYERS[S.layer], f = (a, k) => L.value(a, k);
  const light = document.documentElement.dataset.theme === 'light';
  const series = [['Prime', light ? '#0a86d8' : '#46d3ff', S.prime], ['Multiverse', '#ff7a45', S.multi], ['Observed', light ? '#0e1a2c' : '#e6eefb', S.truth]].filter(s => s[2]).map(([n, c, d]) => [n, c, landMeanSeries(d, S.land, f)]);
  const all = series.flatMap(s => s[2]); let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.15 || 1; lo -= pad; hi += pad;
  const X = d => 30 + d / STEPS * 284, Y = v => 8 + (1 - (v - lo) / (hi - lo)) * 120;
  let g = '';
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; g += `<line x1="30" x2="314" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(140,170,220,.15)"/><text x="2" y="${Y(v) + 3}" fill="#8ea0bb" font-size="9">${v.toFixed(1)}</text>`; }
  g += `<line x1="${X(S.day)}" x2="${X(S.day)}" y1="6" y2="130" stroke="#8ea0bb" stroke-dasharray="2 3"/>`;
  for (let d = 0; d <= STEPS; d += 2) g += `<text x="${X(d)}" y="144" fill="#8ea0bb" font-size="9" text-anchor="middle">${d}</text>`;
  for (const [n, c, s] of series) g += line(s.map((v, d) => `${X(d)},${Y(v)}`).join(' '), c, n === 'Observed');
  $('chart').innerHTML = g;
  $('chartTitle').textContent = `All-India land mean · ${L.name} (${L.unit})`;
  $('chartLegend').innerHTML = series.map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('');
}
function renderPropagation() {
  if (!S.multi || !S.shock) return;
  const p = propagation(S.prime, S.multi, S.shock), maxE = Math.max(...p.map(x => x.energy)) || 1, maxR = Math.max(...p.map(x => x.radius), 500);
  const X = d => 30 + (d - 0.5) / STEPS * 284;
  let g = '';
  p.forEach((x, d) => { const h = x.energy / maxE * 80; g += `<rect x="${X(d + 1) - 9}" y="${92 - h}" width="18" height="${h}" rx="3" fill="rgba(255,122,69,.4)"/>`; });
  g += line(p.map((x, d) => `${X(d + 1)},${92 - x.radius / maxR * 80}`).join(' '), '#46d3ff');
  for (let d = 1; d <= STEPS; d++) g += `<text x="${X(d)}" y="106" fill="#8ea0bb" font-size="9" text-anchor="middle">${d}</text>`;
  $('prop').innerHTML = g;
  $('propNote').innerHTML = `<span style="color:#ff7a45">■</span> response energy &nbsp; <span style="color:#46d3ff">—</span> mean distance from shock: ${fmt(p[0].radius, 0)} km on day 1 → ${fmt(p[p.length - 1].radius, 0)} km on day 10. Day-1 energy near the shock: <b>${fmt(p[0].nearFrac * 100, 0)}%</b>.`;
}
function renderTelemetry() {
  const arr = S.multi || S.prime; if (!arr) return;
  let nan = 0, minRain = Infinity, maxT = -Infinity, minT = Infinity;
  for (let d = 1; d <= STEPS; d++) for (let k = 0; k < CELLS; k++) {
    const t = arr[d][k], r = arr[d][CELLS + k]; if (!Number.isFinite(t) || !Number.isFinite(r)) nan++;
    minRain = Math.min(minRain, r); maxT = Math.max(maxT, t); minT = Math.min(minT, t);
  }
  const rows = {
    'Model': S.modelName || '–', 'ML risk head': S.hasRisk ? 'INDRA-Risk (CNN MOS)' : 'not loaded',
    'Inference (10 days)': `${fmt(S.lastMs, 0)} ms`, 'Round trip': `${fmt(S.roundTrip, 0)} ms`,
    'NaN / Inf': nan ? `${nan} ✗` : '0 ✓', 'Min rain': `${fmt(minRain, 2)} mm ${minRain >= 0 ? '✓' : '✗'}`,
    't2m range': `${fmt(minT - 273.15)}…${fmt(maxT - 273.15)} °C`,
    'Zero-shock Δ': S.nullDiff == null ? '–' : `${S.nullDiff.toExponential(1)} ${S.nullDiff < 1e-3 ? '✓' : '✗'}`,
  };
  $('tele').innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ---------------------------------------------------------------- actions
function setBusy(on, msg) { $('runBtn').disabled = $('ensBtn').disabled = on; if (msg) status(msg, on ? '' : 'ok'); }
function status(msg, cls = '') { $('status').className = 'status ' + cls; $('status').querySelector('span').textContent = msg; }
let toastT; function toast(msg, kind = '') { const t = $('toast'); t.textContent = msg; t.className = kind; t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 5200); }

async function loadWindow(i) {
  S.win = S.windows[i]; $('windowSel').value = i; const W = S.winData, per = 13 * DAY, base = i * per;
  S.init = W.slice(base, base + 3 * DAY);
  S.truth = Array.from({ length: STEPS + 1 }, (_, d) => W.slice(base + (2 + d) * DAY, base + (3 + d) * DAY));
  $('windowDesc').textContent = `${S.win.desc}. Last input day ${S.win.date}; the twin rolls forward 10 days.`;
  S.multi = null; S.spread = null; S.spreadSummary = null; S.nullDiff = null; S.shock = null; S.risk = { prime: null, multi: null };
  $('modeSeg').querySelector('[data-mode=spread]')?.remove(); S.chance = null; globe.setShock(null); map.setShock(null);
  S.day = 0; $('dayRange').value = 0;
  setBusy(true, 'simulating Timeline Prime…');
  try {
    const [prime, clim] = await Promise.all([simulate(S.init, null), post('/climatology', { doy_sincos: S.win.doy_sincos, steps: STEPS })]);
    S.prime = prime; S.clim = clim.clim.map(c => Float32Array.from(c.flat(Infinity)));
    const zero = await simulate(S.init, new Float32Array(DAY));            // live determinism check: zero shock = no change
    S.nullDiff = Math.max(...zero.slice(1).map((d, k) => d.reduce((m, v, j) => Math.max(m, Math.abs(v - S.prime[k + 1][j]) / STD[Math.floor(j / CELLS)]), 0)));
    setBusy(false, `${S.modelName} · ready`); runInfo();
  } catch (e) { setBusy(false); status('server unreachable', 'err'); toast(`Cannot reach ${API}: ${e.message}. Start it with “python serve.py --port 8000”.`); offline(); return; }
  if (S.mode !== 'truth') setMode('prime');
  repaint(); renderImpacts(); renderTelemetry(); updateTimelineLabels();
  $('prop').innerHTML = ''; $('propNote').textContent = 'Run a multiverse shock to see how far it spreads each day.';
  computeAlerts();
  S.risk.prime = await riskFor(S.prime); renderRisk(); renderTelemetry(); computeAlerts(); if (LAYERS[S.layer].ml) repaint();
}

// several shocks: their physical changes are added together (a linear approximation; one /simulate call)
// ---- causal reach: where the shock's response has got to by each day (the model moves information 7 cells a day at most)
function computeReach() {
  if (!S.multi || !S.prime || !S.shock) return null;
  const sd = [13.46, 7.55, 2.9, 2.43, 619, 20.49], perDay = [];
  let gmax = 0;
  for (let d = 1; d <= STEPS; d++) {
    const e = new Float32Array(CELLS);
    for (let k = 0; k < CELLS; k++) { let s = 0; for (let c = 0; c < 6; c++) { const z = (S.multi[d][c * CELLS + k] - S.prime[d][c * CELLS + k]) / sd[c]; s += z * z; } e[k] = Math.sqrt(s); gmax = Math.max(gmax, e[k]); }
    perDay.push(e);
  }
  const thr = Math.max(0.05, 0.05 * gmax), cellDay = new Uint8Array(CELLS).fill(99);
  perDay.forEach((e, d) => { for (let k = 0; k < CELLS; k++) if (cellDay[k] === 99 && e[k] > thr) cellDay[k] = d + 1; });
  const cellLat = k => LAT0 + Math.floor(k / NLON), cellLon = k => LON0 + k % NLON;
  const summary = {};
  for (let day = 1; day <= STEPS; day++) {
    const states = new Set(), cities = [];
    for (let k = 0; k < CELLS; k++) if (cellDay[k] <= day && k % 2 === 0) { const s = stateAt(cellLat(k), cellLon(k)); if (s) states.add(s); }
    for (const [name, la, lo] of CITIES) { const k = Math.round(la - LAT0) * NLON + Math.round(lo - LON0); if (cellDay[k] <= day) cities.push(name); }
    summary[day] = { states: [...states].sort(), cities, cells: [...cellDay].filter(v => v <= day).length };
  }
  return { cellDay, lat: S.shock.lat, lon: S.shock.lon, r0: S.shock.radius || 0, summary, name: S.shock.name };
}
function renderReach(R) {
  if (!R) { $('reachSec').hidden = true; return; }
  $('reachSec').hidden = false; map.setReach(R); map.reachT = +$('reachDay').value;
  const d = Math.max(1, Math.round(+$('reachDay').value)), s = R.summary[Math.min(10, d)];
  $('reachOut').innerHTML = `<b>By day ${d}:</b> the response reaches <b>${s.cells}</b> grid cells` +
    `<br><b>States:</b> ${s.states.join(', ') || 'none yet'}<br><b>Cities:</b> ${s.cities.join(', ') || 'none yet'}` +
    `<br><span class="small">Dashed rings: the furthest it could have gone at 7 cells a day from the shock box.</span>`;
}
let reachPlayT = null;
function playReach() {
  if (reachPlayT) { clearInterval(reachPlayT); reachPlayT = null; $('reachPlay').textContent = '▶ Play'; return; }
  let t = 0; $('reachPlay').textContent = '■ Stop';
  reachPlayT = setInterval(() => { t += 0.1; $('reachDay').value = t; renderReach(S.reach); if (t >= 10) { clearInterval(reachPlayT); reachPlayT = null; $('reachPlay').textContent = '▶ Play'; } }, 90);
}

// ---- live briefing: what happened, how risky, and pins on the three strongest changes
const RISK_COL = { Low: '#4c8bf5', Moderate: '#e8b923', High: '#f08c2e', Critical: '#e5484d' };
let pinEls = [];
function showBrief(html) { $('brief').innerHTML = `<button class="x bx" aria-label="Close briefing">×</button>${html}`; $('brief').hidden = false; $('pins').hidden = false; }
function briefing(sh) { showBrief(`<b>${sh.name}</b><span>Simulating the next 10 days…</span><div class="prog"><i></i></div>`); }
function computeBrief() {
  if (!S.multi || !S.shock || !S.land) return null;
  const cells = [];
  for (let k = 0; k < CELLS; k++) {
    if (S.land[k] <= 0.5) continue;
    let dT = 0, dR = 0;
    for (let d = 1; d <= STEPS; d++) { dT += S.multi[d][k] - S.prime[d][k]; dR += S.multi[d][CELLS + k] - S.prime[d][CELLS + k]; }
    dT /= STEPS; dR /= STEPS;
    if (Math.abs(dT) > 0.3 || Math.abs(dR) > 2) cells.push({ k, dT, dR, mag: Math.abs(dT) / 0.3 + Math.abs(dR) / 2 });
  }
  const lat = k => LAT0 + Math.floor(k / NLON), lon = k => LON0 + k % NLON;
  const near = [...cells].sort((a, b) => b.mag - a.mag).slice(0, 3);
  const states = {};
  for (const c of cells.filter((_, i) => i % 3 === 0)) { const s = stateAt(lat(c.k), lon(c.k)) || 'open sea / neighbours'; states[s] = (states[s] || 0) + 1; }
  const top = Object.entries(states).sort((a, b) => b[1] - a[1]).slice(0, 4).map(x => x[0]);
  const fresh = (S.alertsMulti || []).filter(a => a.status === 'new' || a.status === 'worse');
  const crit = fresh.filter(a => a.severity === 'Critical').length, high = fresh.filter(a => a.severity === 'High').length;
  const risk = crit ? 'Critical' : high >= 10 ? 'High' : high || fresh.length ? 'Moderate' : 'Low';
  const meanT = cells.length ? cells.reduce((s, c) => s + c.dT, 0) / cells.length : 0, meanR = cells.length ? cells.reduce((s, c) => s + c.dR, 0) / cells.length : 0;
  const area = cells.length * 111.2 * 111.2 / 1000;   // rough: one cell ~ 111 x 111 km
  const reason = { Low: 'small change, few new warnings', Moderate: 'new warnings in places that had none', High: 'many new High warnings: act on the forecast', Critical: 'Critical warnings: serious danger' }[risk];
  return { name: S.shock.name, cells: cells.length, area, meanT, meanR, top, risk, reason, high, crit, fresh: fresh.length, pins: near.map(c => ({ lat: lat(c.k), lon: lon(c.k), label: stateAt(lat(c.k), lon(c.k)) || 'Bay / sea' })) };
}
function renderBrief(b) {
  if (!b) return showBrief('<b>No change detected</b><span>The shock barely moved the forecast over this area.</span>');
  const sign = v => (v > 0 ? '+' : '') + v.toFixed(1);
  showBrief(`<div class="bhead"><b>${b.name}</b><span class="risk" style="--rc:${RISK_COL[b.risk]}">${b.risk} risk</span></div>
    <p><b>What happened:</b> over 10 days, temperature changes by <b>${sign(b.meanT)} °C</b> and rain by <b>${sign(b.meanR)} mm/day</b> on average across about <b>${b.cells} grid cells</b> (roughly ${Math.round(b.area / 1000 * 1000)} thousand km²).</p>
    <p><b>Most affected:</b> ${b.top.join(', ') || 'none'}.</p>
    <p><b>How risky:</b> ${b.reason}. New or worse warnings: ${b.fresh} (${b.high} High, ${b.crit} Critical).</p>
    <p class="small">Pins mark the three strongest changes. Averages are over the land cells that changed; this is a research forecast, not an official warning.</p>`);
  pinEls.forEach(e => e.remove()); pinEls = [];
  for (const pnt of b.pins) { const el = document.createElement('div'); el.className = 'pin'; el.innerHTML = `<i></i><span>${pnt.label}</span>`; $('pins').appendChild(el); pinEls.push(el); el.pts = pnt; }
}
// keep pins on the place they mark, hidden when they face away from the globe
function updatePins() {
  if (!pinEls.length || (S.view !== 'globe' && S.view !== 'map')) { pinEls.forEach(e => (e.style.display = 'none')); return; }
  const W = innerWidth, H = innerHeight;
  for (const e of pinEls) {
    const { lat, lon } = e.pts; let x, y, show = true;
    if (S.view === 'map') { x = map.x(lon); y = map.y(lat); }
    else {
      const v = ll2v(lat, lon, 1.02), n = v.clone().normalize(), front = n.dot(globe.camera.position.clone().normalize()) > 0.15;
      const q = v.clone().project(globe.camera); x = (q.x + 1) / 2 * W; y = (1 - q.y) / 2 * H; show = front;
    }
    e.style.display = show ? '' : 'none'; e.style.transform = `translate(${x | 0}px, ${y | 0}px)`;
  }
}
async function runMultiverse() {
  const shocks = S.shockList.length ? S.shockList : [currentShock()], last = S.init.subarray(2 * DAY, 3 * DAY);
  const delta = new Float32Array(DAY); let touched = 0;
  for (const sh of shocks) { const r = buildPerturbation(sh, last); touched += r.touched; r.delta.forEach((v, k) => { delta[k] += v; }); }
  const sh = { ...shocks[0], name: shocks.map(x => x.name).join(' + ') };
  if (!touched) return warnShock(true);
  setBusy(true, `branching universe: ${sh.name}…`);
  briefing(sh);
  try { S.multi = await simulate(S.init, delta); S.shock = sh; S.brief = computeBrief(); renderBrief(S.brief); S.reach = computeReach(); renderReach(S.reach); } catch (e) { toast(e.message); setBusy(false); return; }
  setBusy(false, `${S.modelName} · ready`); runInfo();
  globe.setShock(sh); map.setShock(sh);
  if (S.view === 'globe' && !S.tour) globe.flyTo(sh.lat, sh.lon, 2.3, 1.6);
  setMode('delta'); S.day = 1; $('dayRange').value = 1;
  repaint(); renderImpacts(); renderPropagation(); renderTelemetry(); updateTimelineLabels(); play(true);
  computeAlerts();
  S.risk.multi = await riskFor(S.multi); renderRisk(); computeAlerts();
}

async function runEnsemble() {
  setBusy(true, 'running 5-member ensemble…');
  const sig = [0.4, 0.6, 0.35, 0.35, 40, 1.2]; const members = [];
  try {
    for (let m = 0; m < 5; m++) {
      let s = 1234 + m * 977; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 2;
      const coarse = Array.from({ length: NC * 9 * 10 }, rnd), d = new Float32Array(DAY);
      for (let c = 0; c < NC; c++) for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
        const gi = i / 31 * 8, gj = j / 35 * 9, i0 = Math.floor(gi), j0 = Math.floor(gj), fi = gi - i0, fj = gj - j0, b = c * 90;
        const at = (a, bb) => coarse[b + Math.min(8, a) * 10 + Math.min(9, bb)];
        const v = (at(i0, j0) * (1 - fj) + at(i0, j0 + 1) * fj) * (1 - fi) + (at(i0 + 1, j0) * (1 - fj) + at(i0 + 1, j0 + 1) * fj) * fi;
        d[c * CELLS + idx(i, j)] = v * sig[c] * (c === 1 ? Math.min(1, S.init[2 * DAY + CELLS + idx(i, j)] / 5) : 1);
      }
      members.push(await simulate(S.init, d));
    }
  } catch (e) { toast(e.message); setBusy(false); return; }
  const lay = LAYERS[S.layer].ml ? 't2m' : S.layer;
  S.spread = Array.from({ length: STEPS + 1 }, (_, d) => {
    const gs = members.map(m => layerGrid(m[d], lay)), out = new Float32Array(CELLS);
    for (let k = 0; k < CELLS; k++) { const mu = gs.reduce((a, g) => a + g[k], 0) / gs.length; out[k] = Math.sqrt(gs.reduce((a, g) => a + (g[k] - mu) ** 2, 0) / (gs.length - 1)); }
    return out;
  });
  S.chance = Array.from({ length: STEPS + 1 }, (_, d) => {           // % of members with rain >= 64.5 mm/day (IMD heavy)
    const out = new Float32Array(CELLS); for (const m of members) for (let k = 0; k < CELLS; k++) if (m[d][CELLS + k] >= 64.5) out[k] += 100 / members.length;
    return out;
  });
  if (!$('modeSeg').querySelector('[data-mode=chance]')) { const b = document.createElement('button'); b.dataset.mode = 'chance'; b.textContent = 'Rain chance'; b.title = 'Share of the 5 perturbed runs with rain at or above 64.5 mm/day (IMD heavy). A spread among perturbed runs, not a calibrated probability.'; $('modeSeg').append(b); }
  const sigma = S.spread[STEPS].reduce((a, v) => a + v, 0) / CELLS;
  S.spreadSummary = { sigma, conf: 100 * Math.max(0, 1 - sigma / LAYERS[lay].dRange), layer: lay };
  if (lay !== S.layer) setLayer(lay);
  if (!$('modeSeg').querySelector('[data-mode=spread]')) { const b = document.createElement('button'); b.dataset.mode = 'spread'; b.textContent = 'Spread σ'; b.title = 'Ensemble spread: standard deviation among the 5 perturbed runs'; $('modeSeg').append(b); }
  setBusy(false, `${S.modelName} · ready`); const want = S.wantChance ? 'chance' : 'spread'; S.wantChance = false; setMode(want); repaint(); renderImpacts();
}

// ---------------------------------------------------------------- view state
function setMode(m) {
  if ((m === 'multi' || m === 'delta') && !S.multi) { toast('Run a multiverse shock first (left panel).', 'info'); m = 'prime'; }
  if (m === 'chance' && !S.chance) { S.wantChance = true; return runEnsemble(); }   // first click runs the ensemble, then shows the chance map
  if (m === 'spread' && !S.spread) m = 'prime';
  if (m === 'truth' && LAYERS[S.layer].ml) { toast('ML risk is a forecast product — showing observed temperature.', 'info'); setLayer('t2m', true); }
  S.mode = m; for (const b of $('modeSeg').children) b.classList.toggle('on', b.dataset.mode === m);
  const [lo, hi, unit] = paintLegend($('legend'), m === 'chance' ? 'hrc' : S.layer, m === 'delta' ? 'delta' : m === 'spread' ? 'spread' : 'abs');
  if (m === 'chance') $('layerNote').textContent = 'Heavy-rain chance: share of the 5 perturbed runs at or above 64.5 mm/day. It is a spread among perturbed runs, not a calibrated probability.';
  $('legLo').textContent = lo; $('legHi').textContent = hi; $('legUnit').textContent = unit;
  repaint(); updateTimelineLabels(); renderRisk(); renderAlerts();
}
// weather layers can be overlaid (a set); ML risk layers are always shown alone
function setLayer(k, quiet) {
  if (LAYERS[k].ml && !S.hasRisk) return toast('The ML risk model is not loaded on the server (run train_risk.py).');
  S.layer = k; S.layers = new Set(LAYERS[k].ml ? [] : [k]); syncChips();
  if (S.mode === 'spread') { S.spread = null; S.spreadSummary = null; S.mode = 'prime'; } if (S.mode === 'chance') S.mode = 'prime';
  if (!quiet) { setMode(S.mode === 'truth' && LAYERS[k].ml ? 'prime' : S.mode); renderChart(); }
}
// click on a weather chip: add or remove it from the overlay (at least one stays on)
function toggleLayer(k) {
  if (LAYERS[k].ml || !S.layers.size || (S.layers.has(k) && S.layers.size === 1)) return setLayer(k);
  if (S.layers.has(k)) { S.layers.delete(k); if (S.layer === k) S.layer = [...S.layers].at(-1); }
  else { S.layers.add(k); S.layer = k; }
  syncChips();
  if (S.mode === 'spread') { S.spread = null; S.spreadSummary = null; S.mode = 'prime'; } if (S.mode === 'chance') S.mode = 'prime';
  setMode(S.mode); renderChart();
}
function syncChips() {
  const chance = S.mode === 'chance';   // the chance map replaces the layer, so no weather chip stays lit
  for (const b of $('layerChips').children) b.classList.toggle('on', !chance && (b.dataset.l === S.layer || (!LAYERS[b.dataset.l]?.ml && S.layers.has(b.dataset.l))));
  const names = [...S.layers].map(k => LAYERS[k].name);
  $('layerNote').textContent = LAYERS[S.layer].ml ? `INDRA-Risk: probability that ${S.layer === 'flood' ? 'daily rain exceeds 64.5 mm (IMD heavy)' : 'daily-mean heat stress occurs'}, learned from INDRA's own forecasts vs what happened.`
    : names.length > 1 ? `Overlay: ${names.join(' + ')}. The legend shows ${LAYERS[S.layer].name}.` : '';
}
// "All layers": step through the five weather layers, one every 4 s; any single layer click stops it
const CYCLE = ['t2m', 'tp', 'wind', 'msl', 'tcwv'];
let cycleT = null;
function stopCycle() { clearInterval(cycleT); cycleT = null; $('layerChips').querySelector('.allchip').textContent = 'All layers ▶'; $('layerChips').querySelector('.allchip').classList.remove('on'); }
function cycleLayers() {
  if (cycleT) return stopCycle();
  let i = Math.max(0, CYCLE.indexOf(S.layer));
  const step = () => { setLayer(CYCLE[i]); i = (i + 1) % CYCLE.length; };
  $('layerChips').querySelector('.allchip').textContent = 'Stop ■'; $('layerChips').querySelector('.allchip').classList.add('on');
  step(); cycleT = setInterval(step, 4000);
}
function setView(v) {
  S.view = v; document.body.classList.toggle('view-map', v === 'map'); document.body.classList.toggle('view-street', v === 'street');
  globe.active = v === 'globe'; map.active = v === 'map'; street.setActive(v === 'street');
  if (v === 'street' && !street.map) street.ensure().then(() => {
    street.setActive(S.view === 'street');
    if (S.state) return street.flyToState(S.state);                   // a chosen state keeps its own view
    const c = CITIES[$('citySel').value]; c ? street.flyToCity(c[1], c[2]) : street.overview();
  });
  for (const b of $('viewSeg').children) b.classList.toggle('on', b.dataset.view === v);
  if (v === 'map' && !map.flown) { map.flown = true; map.flyTo(21, 82, 4.2); }
  fieldDirty = true;
}
function setLayout(l) {
  document.body.classList.remove('layout-full', 'layout-focus', 'layout-cinema'); document.body.classList.add('layout-' + l);
  for (const b of $('layoutSeg').children) b.classList.toggle('on', b.dataset.layout === l);
  globe.autoRotate = l === 'cinema' ? true : $('optRotate').checked; S.layout = l;
}
function setTheme(t, ev) {
  const apply = () => {
    document.documentElement.dataset.theme = t; store.set('indra-theme', t);
    globe.setTheme(t); map.setTheme(t); globe.frame(0, frameOpts()); renderChart();
  };
  if (!document.startViewTransition || matchMedia('(prefers-reduced-motion: reduce)').matches) return apply();
  const x = ev?.clientX ?? innerWidth - 60, y = ev?.clientY ?? 40, r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  document.startViewTransition(apply).ready.then(() => document.documentElement.animate(
    { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
    { duration: 850, easing: 'cubic-bezier(.65,0,.35,1)', pseudoElement: '::view-transition-new(root)' }));
}
function syncShockUI() {
  for (const b of $('presets').children) b.classList.toggle('on', b.dataset.p === S.preset);
  const v = VARS[$('shockVar').value], r = $('shockMag');
  Object.assign(r, { min: v.min, max: v.max, step: v.step });
  $('magOut').textContent = `${+r.value > 0 ? '+' : ''}${r.value}${v.unit}`; $('radOut').textContent = `${$('shockRad').value}°`;
  const sh = currentShock(); globe.setShock(sh); map.setShock(sh); warnShock(false);
}
function warnShock(force) {
  const sh = currentShock(), w = $('shockWarn');
  const cl = Math.max(BBOX.s, Math.min(BBOX.n, sh.lat)), co = Math.max(BBOX.w, Math.min(BBOX.e, sh.lon));
  const touches = sh.box || Math.hypot(sh.lat - cl, sh.lon - co) < sh.radius;
  w.hidden = touches; w.textContent = 'This shock lies outside the simulated domain (6–38°N, 65–101°E), so the twin cannot propagate it.';
  if (force && !touches) toast(w.textContent);
}
// the shock list: add the current settings, remove by clicking
function renderShockList() {
  $('shockList').innerHTML = S.shockList.length ? S.shockList.map((s, i) => `<button class="chip" data-i="${i}" title="remove">${s.name} ×</button>`).join('') : '<span class="hint">No extra shocks. Add one to combine them.</span>';
}
function play(on) { S.playing = on ?? !S.playing; $('playIcon').setAttribute('d', S.playing ? 'M6 4h4v16H6zM14 4h4v16h-4z' : 'M7 4v16l13-8z'); if (S.playing && S.day >= STEPS - 0.01) S.day = 0; }
function stepDay(d) { play(false); S.day = Math.max(0, Math.min(STEPS, Math.round(S.day) + d)); $('dayRange').value = S.day; repaint(); updateTimelineLabels(); renderChart(); }
const frameOpts = () => ({ ...S.opts });

// ---------------------------------------------------------------- states: click a state, see its stats, open its own 3D street view
function selectState(lat, lon) {
  const name = stateAt(lat, lon);
  if (!name) return toast('Click inside India to open a state (6–38°N, 65–101°E).', 'info');
  chooseState(name);
}
function chooseState(name) {
  S.state = name; $('stateSel').value = name;
  map.setOutline(stateFeature(name));
  if (streetImpl) streetImpl.map?.getSource('state')?.setData({ type: 'FeatureCollection', features: [stateFeature(name)] });
  $('stateSec').hidden = false; $('stateSec').classList.remove('pulse'); void $('stateSec').offsetWidth; $('stateSec').classList.add('pulse');
  renderState();
}
function renderState() {
  if (!S.state || !S.prime || !S.clim) return;
  const multi = S.multi && (S.mode === 'multi' || S.mode === 'delta');
  const P = stateStats(S.state, S.prime, S.clim, S.risk.prime, S.alertsPrime);
  const M = multi ? stateStats(S.state, S.multi, S.clim, S.risk.multi, S.alertsMulti) : null;
  const d = (a, b, dig) => M ? `<div class="d">${multi ? 'universe' : 'multiverse'} ${b - a > 0 ? '+' : ''}${(b - a).toFixed(dig)}</div>` : '';
  $('stateTitle').innerHTML = `<b>${S.state}</b> · ${P.cells} grid cells`;
  $('stateCards').innerHTML = [
    card('Mean temperature', P.meanT.toFixed(1), '°C', '#ff7a45', M && d(P.meanT, M.meanT, 1)),
    card('Mean rain', P.meanR.toFixed(1), 'mm/day', '#46d3ff', M && d(P.meanR, M.meanR, 1)),
    card('Mean wind', P.meanW.toFixed(1), 'm/s', '#3ddc97', M && d(P.meanW, M.meanW, 1)),
    card('Heatwave cells', P.heatPct.toFixed(0), '% peak', '#ff5a3d', M && d(P.heatPct, M.heatPct, 0)),
    card('Heavy-rain cells', P.floodPct.toFixed(0), '% peak', '#5b8cff', M && d(P.floodPct, M.floodPct, 0)),
    card('Alerts', P.alerts, `${P.high} High+`, '#e5484d', M && `<div class="d">${multi ? 'universe' : 'multiverse'} ${M.alerts - P.alerts > 0 ? '+' : ''}${M.alerts - P.alerts}</div>`),
    card('ML flood chance', P.floodRisk == null ? '–' : P.floodRisk.toFixed(0), '% max', '#7b3dff', null),
    card('ML heat chance', P.heatRisk == null ? '–' : P.heatRisk.toFixed(0), '% max', '#ff6a1a', null),
  ].join('');
}
// tipping point: the smallest warming of the Bay of Bengal (temperature only) that brings heavy rain to the chosen state
async function tippingPoint() {
  const name = S.state || 'Odisha'; if (!S.state) chooseState(name);
  if (!S.init) return toast('Wait for the forecast to load first.', 'info');
  const mask = stateMask(name), BOX = [8, 20, 82, 94], T = 64.5, last = S.init.subarray(2 * DAY, 3 * DAY);
  const peak = out => { let m = 0; for (let d = 1; d <= STEPS; d++) for (let k = 0; k < CELLS; k++) if (mask[k] && out[d][CELLS + k] > m) m = out[d][CELLS + k]; return m; };
  // warming + moisture rising 7% per K (Clausius-Clapeyron): warm air holds more water vapour, which is what makes heavy rain
  const rainAt = async dK => { if (!dK) return peak(S.prime); const { delta } = buildPerturbation({ box: BOX, parts: [['t2m', dK], ['tcwv', 7 * dK]] }, last); return peak(await simulate(S.init, delta)); };
  setBusy(true, `searching the tipping point for ${name}…`);
  const out = $('tipOut'); out.hidden = false; let msg;
  try {
    let lo = 0, hi = 8;
    if (await rainAt(0) >= T) msg = `${name} already has heavy rain (≥ ${T} mm/day) in this forecast; no warming needed.`;
    else if (await rainAt(hi) < T) msg = `${name} does not reach heavy rain even with +${hi} K over the Bay of Bengal (with moisture rising 7% per K).`;
    else {
      for (let i = 0; i < 9; i++) { const mid = (lo + hi) / 2; if (await rainAt(mid) >= T) hi = mid; else lo = mid; }
      msg = `Tipping point: about <b>+${hi.toFixed(2)} K</b> over the Bay of Bengal (with moisture rising 7% per K) brings heavy rain (≥ ${T} mm/day) to ${name} within 10 days.`;
    }
  } catch (e) { msg = `Could not run the search: ${e.message}`; }
  setBusy(false, `${S.modelName} · ready`);
  out.innerHTML = msg;
}
function openStateStreet() {
  if (!S.state) return toast('Click a state first.', 'info');
  setView('street'); street.flyToState(S.state);
}

// ---------------------------------------------------------------- city forecast (Prime next to Multiverse)
const CITY_VARS = [['t', 'Temp', '°C', 1, '#ff7a45'], ['r', 'Rain', 'mm', 1, '#46d3ff'], ['w', 'Wind', 'm/s', 1, '#3ddc97']];
function cityValues(days, c) {
  if (!days) return null;
  if (!c) {                                                   // All India: area-weighted land mean
    const [t, r, w] = [(a, k) => a[k] - 273.15, (a, k) => a[CELLS + k], (a, k) => Math.hypot(a[2 * CELLS + k], a[3 * CELLS + k])].map(f => landMeanSeries(days, S.land, f));
    return days.map((_, d) => ({ t: t[d], r: r[d], w: w[d] }));
  }
  const gi = c[1] - LAT0, gj = c[2] - LON0, at = (a, ch) => bilinear(a.subarray(ch * CELLS, (ch + 1) * CELLS), gi, gj);
  return days.map(a => ({ t: at(a, 0) - 273.15, r: Math.max(0, at(a, 1)), w: Math.hypot(at(a, 2), at(a, 3)) }));
}
function renderCity() {
  if (!S.prime || !S.land) return;
  const c = CITIES[$('citySel').value], P = cityValues(S.prime, c), M = S.multi && cityValues(S.multi, c), d = Math.max(1, Math.round(S.day));
  const delta = (a, b, dig) => { if (b == null) return ''; const x = b - a; return Math.abs(x) < 10 ** -dig / 2 ? '' : `<span class="${x > 0 ? 'up' : 'down'}">${x > 0 ? '+' : ''}${x.toFixed(dig)}</span>`; };
  $('cityCards').innerHTML = CITY_VARS.map(([k, n, u, dig, col]) => card(`${n} d+${d}`, P[d][k].toFixed(dig), u, col,
    M ? `<div class="d">Δ ${delta(P[d][k], M[d][k], dig) || '0'}</div>` : '')).join('');
  $('cityTab').innerHTML = `<tr><th>Day</th>${CITY_VARS.map(([, n, u]) => `<th>${n} ${u}</th>`).join('')}</tr>` +
    P.slice(1).map((p, i) => `<tr class="${i + 1 === d ? 'cur' : ''}"><td>+${i + 1}</td>${CITY_VARS.map(([k, , , dig]) => `<td>${p[k].toFixed(dig)} ${M ? delta(p[k], M[i + 1][k], dig) : ''}</td>`).join('')}</tr>`).join('');
}

// ---------------------------------------------------------------- alerts & warnings (policy co-pilot)
const SEV_COL = { Critical: '#e5484d', High: '#f08c2e', Moderate: '#e8b923', Low: '#4c8bf5' };
const sevOn = new Set(SEVERITIES);
function computeAlerts() {
  if (!S.prime || !S.clim || !S.land) return;
  S.alertsPrime = generateAlerts(S.prime, S.clim, S.land, S.risk.prime);
  S.alertsMulti = S.multi ? diffAlerts(S.alertsPrime, generateAlerts(S.multi, S.clim, S.land, S.risk.multi)) : null;
  renderAlerts();
}
function renderAlerts() {
  if (!S.alertsPrime) return;
  renderTicker();                                                  // the ticker shows the alert count
  const multi = S.alertsMulti && (S.mode === 'multi' || S.mode === 'delta'), all = multi ? S.alertsMulti : S.alertsPrime;
  const urgent = all.filter(a => a.severity === 'Critical' || a.severity === 'High').length, b = $('alertsBadge');
  b.hidden = !all.length; b.textContent = all.length; b.classList.toggle('calm', !urgent);
  $('alertsSrc').textContent = multi ? `· ${S.shock.name} universe` : '· Timeline Prime';
  $('fStatusWrap').hidden = !multi; if (!multi) $('fStatus').value = '';
  const states = [...new Set(all.map(a => a.region))].sort(), cur = $('fState').value;
  $('fState').innerHTML = `<option value="">All (${states.length})</option>` + states.map(r => `<option${r === cur ? ' selected' : ''}>${r}</option>`).join('');
  const f = a => (!$('fState').value || a.region === $('fState').value) && (!$('fType').value || a.type === $('fType').value) && (!$('fStatus').value || a.status === $('fStatus').value);
  $('sevChips').innerHTML = SEVERITIES.map(v => `<button data-s="${v}" class="${sevOn.has(v) ? '' : 'off'}" style="--sc:${SEV_COL[v]}" aria-pressed="${sevOn.has(v)}"><b>${all.filter(a => a.severity === v && f(a)).length}</b><span>${v}</span></button>`).join('');
  $('sevChips').onclick = e => { const v = e.target.closest('button')?.dataset.s; if (!v) return; sevOn.has(v) ? sevOn.delete(v) : sevOn.add(v); renderAlerts(); };
  const shown = all.filter(a => sevOn.has(a.severity) && f(a)).slice(0, 80);
  $('alertList').innerHTML = shown.length ? shown.map((a, n) => {
    const what = a.type === 'cyclone' ? 'Cyclone risk' : a.type === 'rain' ? (a.severity === 'Low' ? 'Heavy-rain watch' : 'Heavy rain') : (a.severity === 'Low' ? 'Heat watch' : 'Heat stress');
    const val = a.type === 'cyclone' ? `low-pressure core, wind up to ${a.value.toFixed(0)} m/s` : a.type === 'rain' ? `peak ${Math.round(a.value)} mm/day` : `daily mean up to ${a.value.toFixed(1)} °C`;
    const adv = ADVICE[a.type];
    return `<li data-n="${n}" style="--sc:${SEV_COL[a.severity]}"><div class="top"><span class="sev">${a.severity.toUpperCase()}</span><b>${what}</b>${a.status === 'new' ? '<span class="tag">NEW in multiverse</span>' : a.status === 'worse' ? '<span class="tag">MORE SEVERE</span>' : ''}</div>
      <div class="meta">Day +${a.day} · ${a.region} · ${val} · ${a.cells} cell${a.cells > 1 ? 's' : ''} (~${(a.cells * 12).toLocaleString('en-IN')}k km²)</div>
      <details onclick="event.stopPropagation()"><summary>Dos &amp; don'ts</summary><ul>${adv.do.map(t => `<li>✔ ${t}</li>`).join('')}${adv.dont.map(t => `<li>✘ ${t}</li>`).join('')}</ul></details></li>`;
  }).join('') : `<li class="empty">No alerts match. ${all.length ? 'Try other filters.' : 'This universe has no heavy-rain or heat alerts.'}</li>`;
  $('alertList').onclick = e => {
    const a = shown[e.target.closest('li[data-n]')?.dataset.n]; if (!a) return;
    setLayer(a.type === 'rain' ? 'tp' : a.type === 'cyclone' ? 'msl' : 't2m'); stepDay(a.day - Math.round(S.day));
    S.view === 'globe' ? globe.flyTo(a.lat, a.lon, 2.0, 1.4) : map.flyTo(a.lat, a.lon, 7);
  };
}
function toggleAlerts(on) {
  const open = on ?? !document.body.classList.contains('alerts-open');
  document.body.classList.toggle('alerts-open', open); $('alertsBtn').classList.toggle('on', open); $('alertsBtn').setAttribute('aria-expanded', open);
}

// guided tour: every event, fly in, show the caption, play its 10 days, move on (any click on Tour stops it)
// count a number up from 0 (tour captions); respects reduced motion
function countUp(el, to, dec = 0) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return (el.textContent = to.toFixed(dec));
  const t0 = performance.now();
  const step = now => { const p = Math.min(1, (now - t0) / 900), e = 1 - Math.pow(1 - p, 3); el.textContent = (to * e).toFixed(dec); if (p < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function captionFor(w, id) {
  const alerts = (S.alertsPrime || []).length, peakRain = S.prime ? Math.max(...S.prime.slice(1).map(a => Math.max(...a.subarray(CELLS, 2 * CELLS)))) : 0;
  $('caption').innerHTML = `<b>${w.title}</b><span>${w.desc} · ${w.date}</span>
    <div class="nums"><div><span data-n="alerts"></span><small>alerts</small></div><div><span data-n="rain"></span><small>peak rain mm</small></div></div>
    <div class="prog"><i></i></div>`;
  $('caption').classList.remove('swap'); void $('caption').offsetWidth; $('caption').classList.add('swap');
  countUp($('caption').querySelector('[data-n=alerts]'), alerts); countUp($('caption').querySelector('[data-n=rain]'), peakRain);
}

async function tour() {
  if (S.tour) { S.tour = null; $('tourBtn').classList.remove('on'); $('caption').hidden = true; setLayout('full'); return; }
  const id = S.tour = {}; S.tourLog = []; $('tourBtn').classList.add('on'); setLayout('cinema');
  for (let i = 0; i < S.windows.length && S.tour === id; i++) {
    const w = S.windows[i]; $('caption').hidden = false; captionFor(w, id);
    setLayer(/flood|monsoon|Kerala|Konkan/i.test(w.title) ? 'tp' : /heat/i.test(w.title) ? 't2m' : /Cyclone/i.test(w.title) ? 'wind' : 't2m');
    await loadWindow(i); if (S.tour !== id) break;
    captionFor(w, id);                                                    // numbers now come from this event's forecast
    S.tourLog.push({ title: w.title, date: w.date, alerts: S.alertsPrime.length, peakRain: Math.max(...S.prime.slice(1).map(a => Math.max(...a.subarray(CELLS, 2 * CELLS)))), meanT: landMeanSeries(S.prime.slice(1), S.land, (a, k) => a[k] - 273.15).reduce((x, y) => x + y, 0) / 10 });
    if (S.view === 'globe') globe.flyTo(21 + (Math.random() - 0.5) * 6, 82 + (Math.random() - 0.5) * 10, 2.1 + Math.random() * 0.4, 3);
    setMode('prime'); S.day = 0; play(true);
    await new Promise(r => { const t = setInterval(() => {
      $('caption').querySelector('.prog i').style.width = `${Math.min(100, S.day / STEPS * 100)}%`;
      if (!S.playing || S.tour !== id) { clearInterval(t); r(); }
    }, 200); });
    await new Promise(r => setTimeout(r, 900));
  }
  if (S.tour === id) tourEnd(id);
}

// the tour ends on a summary card with the report and replay buttons
function tourEnd(id) {
  S.tour = null; $('tourBtn').classList.remove('on'); setLayout('full');
  $('caption').innerHTML = `<b>Tour complete</b><span>${S.tourLog.length} events · every number below is from the run you just watched</span>
    <div class="nums"><div><span>${S.tourLog.reduce((a, e) => a + e.alerts, 0)}</span><small>alerts in total</small></div><div><span>${Math.max(...S.tourLog.map(e => e.peakRain)).toFixed(0)}</span><small>peak rain mm</small></div></div>
    <div class="actions"><button class="primary slim" id="tourPdf">Download tour report (PDF)</button><button class="ghost slim" id="tourAgain">Replay</button><button class="ghost slim" id="tourClose">Close</button></div>`;
  $('caption').classList.remove('swap'); void $('caption').offsetWidth; $('caption').classList.add('swap');
  $('tourPdf').onclick = () => downloadReport(S.tourLog);
  $('tourAgain').onclick = () => { $('caption').hidden = true; tour(); };
  $('tourClose').onclick = () => { $('caption').hidden = true; };
}

// live data ticker: city values and the headline numbers, scrolling above the timeline (duplicated for a seamless loop)
function renderTicker() {
  if (!S.prime || !S.clim) return;
  const d = Math.max(1, Math.round(S.day)), P = S.alertsPrime || [], items = [
    `<b>${S.win?.title || 'Forecast'}</b>`, `Day +${d}`,
    `Alerts <b>${P.length}</b> (${P.filter(a => a.severity === 'High' || a.severity === 'Critical').length} High+)`,
    ...CITIES.slice(0, 9).map((c, i) => { const v = cityValues(S.prime, c)[d]; return `${c[0]} <b>${v.t.toFixed(1)}°C</b> <i>${v.r.toFixed(1)} mm</i>`; }),
  ];
  if (S.multi) { const M = impacts(S.multi, S.clim, S.land), Pp = impacts(S.prime, S.clim, S.land); items.push(`${S.shock.name}: heatwave <b>${Pp.heatPeak.toFixed(1)}%→${M.heatPeak.toFixed(1)}%</b>`); }
  items.push(`Engine <b>${S.lastMs ? S.lastMs.toFixed(0) : '–'} ms</b>`, `Zero-shock <b>${S.nullDiff == null ? '–' : S.nullDiff < 1e-3 ? 'passes' : 'check'}</b>`);
  const one = items.map(x => `<span>${x}</span>`).join('<span class="sep">◆</span>');
  $('tickerTrack').innerHTML = one + '<span class="sep">◆</span>' + one;
}

/** current view as an image (globe, 2D map or Street 3D) -> { data: dataURL, w, h } */
async function viewImage(type = 'image/jpeg') {
  let c;
  if (S.view === 'globe') { globe.frame(0, frameOpts()); c = globe.renderer.domElement; }
  else if (S.view === 'street') {
    const data = await street.snapshot(); if (!data) return null;
    const cv = street.map.getCanvas(); return { data, w: cv.width, h: cv.height };
  } else {
    c = document.createElement('canvas'); const m = map.c.mapBase; c.width = m.width; c.height = m.height; const g = c.getContext('2d');
    g.fillStyle = getComputedStyle(document.body).getPropertyValue('--mapbg') || '#02050b'; g.fillRect(0, 0, c.width, c.height);   // JPEG has no alpha
    for (const k of ['mapBase', 'mapField', 'mapWind', 'mapUi']) g.drawImage(map.c[k], 0, 0);
  }
  return { data: c.toDataURL(type, 0.86), w: c.width, h: c.height };
}
async function snapshot() {
  const img = await viewImage('image/png'); if (!img) return toast('Nothing to capture yet.', 'info');
  const a = Object.assign(document.createElement('a'), { href: img.data, download: `indra-${S.win?.date || 'view'}-day${Math.round(S.day)}-${S.mode}.png` }); a.click();
  toast('Snapshot saved.', 'info');
}

const MODE_NAME = { prime: 'Timeline Prime', multi: 'Multiverse', delta: 'Delta shock-impact', truth: 'Observed', spread: 'Ensemble spread', chance: 'Heavy-rain chance (ensemble)' };
async function downloadReport(tourLog = null) {
  if (!S.prime || !S.clim || !S.land) return toast('Wait for the simulation to finish first.', 'info');
  $('reportBtn').disabled = true; toast('Building the scenario report…', 'info');
  try {
    const [{ makeReport }, image] = await Promise.all([import('./report.js'), viewImage()]);
    const P = impacts(S.prime, S.clim, S.land), M = S.multi ? impacts(S.multi, S.clim, S.land) : null;
    const k = (o, f) => o && f(o);
    const c = CITIES[$('citySel').value], cp = cityValues(S.prime, c), cm = S.multi && cityValues(S.multi, c);
    const alerts = (S.multi && S.alertsMulti) || S.alertsPrime || [];
    const counts = Object.fromEntries(SEVERITIES.map(v => [v, alerts.filter(a => a.severity === v).length]));
    const prop = S.multi && S.shock ? propagation(S.prime, S.multi, S.shock) : null;
    const t = (a, kk) => a[kk] - 273.15;
    const name = makeReport({
      win: S.win, shock: S.multi ? S.shock : null, mode: MODE_NAME[S.mode] || S.mode, day: Math.round(S.day), image,
      impact: [['Heatwave exposure', P.heatPeak, M?.heatPeak, '% land', 1], ['Flood-risk area', P.floodArea / 1000, k(M, m => m.floodArea / 1000), 'k km2', 0],
        ['Crop water stress', P.cropStress, M?.cropStress, '%', 1], ['Rain vs normal', P.rainDeparture, M?.rainDeparture, '%', 0],
        ['Wind power at 100 m', P.windPower, M?.windPower, 'W/m2', 0], ['Peak daily rain', P.peakRain, M?.peakRain, 'mm', 0]],
      chart: { name: 'Temperature', unit: 'C', series: [['Prime', [70, 160, 230], landMeanSeries(S.prime, S.land, t), false],
        ...(S.multi ? [['Multiverse', [255, 122, 69], landMeanSeries(S.multi, S.land, t), false]] : []), ['Observed', [14, 26, 44], landMeanSeries(S.truth, S.land, t), true]] },
      city: { name: c ? c[0] : 'All India (land average)', rows: cp.slice(1).map((v, i) => [i + 1, v.t, v.r, v.w, cm && cm[i + 1].t - v.t, cm && cm[i + 1].r - v.r, cm && cm[i + 1].w - v.w]) },
      alerts: { counts, list: alerts }, prop: prop && { near: prop[0].nearFrac, r1: prop[0].radius, r10: prop[prop.length - 1].radius },
      tele: { model: S.modelName, ms: S.lastMs, zero: S.nullDiff },
      commit: COMMIT, lastRun: S.lastRun,
      tour: tourLog?.length ? tourLog : null,
      state: S.state ? { name: S.state, ...stateStats(S.state, S.prime, S.clim, S.risk.prime, S.alertsPrime) } : null,
    });
    toast(`Saved ${name}`, 'info');
  } catch (e) { toast(`Report failed: ${e.message}`); }
  finally { $('reportBtn').disabled = false; }
}

function toggleAbout(on) {
  const a = $('about'); a.hidden = !(on ?? a.hidden);
  if (a.hidden) return;
  // count the headline numbers up from 0 each time the panel opens
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches, t0 = performance.now();
  const els = [...a.querySelectorAll('[data-count]')];
  const tick = now => {
    const p = reduce ? 1 : Math.min(1, (now - t0 - 300) / 1100), e = 1 - Math.pow(1 - Math.max(0, p), 3);
    for (const el of els) el.textContent = (+el.dataset.count * e).toFixed(+el.dataset.dec);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- custom start date (3 consecutive real days)
const fmtDay = d => d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
const addDays = (iso, n) => { const d = new Date(iso + 'T00:00:00'); d.setDate(d.getDate() + n); return d; };
function customHint() {
  const v = $('customDate').value; if (!v) return;
  $('customHint').innerHTML = `Uses <b>${fmtDay(addDays(v, -2))}</b>, <b>${fmtDay(addDays(v, -1))}</b> and <b>${fmtDay(addDays(v, 0))}</b> → forecasts <b>${fmtDay(addDays(v, 1))} – ${fmtDay(addDays(v, 10))}</b>.`;
}
async function initCustomDate() {
  try {
    const r = await (await fetch(DATA_API + '/range')).json();
    Object.assign($('customDate'), { min: r.first, max: r.last, value: r.last });
    customHint();
  } catch {
    $('customDate').disabled = $('customGo').disabled = true;
    $('customHint').innerHTML = 'Start the data helper to pick any date: <b>python tools/window_server.py</b>';
  }
}
async function runCustomDate() {
  const date = $('customDate').value; if (!date) return;
  const existing = S.windows.findIndex(w => w.id === 'custom-' + date);
  if (existing >= 0) return loadWindow(existing);
  $('customGo').disabled = true;
  try {
    const res = await fetch(`${DATA_API}/window?date=${date}`);
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
    const r = await res.json(), bin = atob(r.b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const block = new Float32Array(bytes.buffer), all = new Float32Array(S.winData.length + block.length);
    all.set(S.winData); all.set(block, S.winData.length); S.winData = all;
    S.windows.push({ id: 'custom-' + date, title: `Your dates · ${fmtDay(addDays(date, -2))} – ${fmtDay(addDays(date, 0))}`,
      desc: `Your chosen start: 3 real days ending ${fmtDay(addDays(date, 0))}`, date, doy: r.doy, doy_sincos: r.doy_sincos });
    const i = S.windows.length - 1;
    $('windowSel').insertAdjacentHTML('beforeend', `<option value="${i}">${S.windows[i].title}</option>`);
    await loadWindow(i);
  } catch (e) { toast(`Could not load ${date}: ${e.message}`); }
  finally { $('customGo').disabled = false; }
}

function initUI() {
  $('windowSel').innerHTML = S.windows.map((w, i) => `<option value="${i}">${w.title} · ${w.date}</option>`).join('');
  $('windowSel').onchange = e => loadWindow(+e.target.value);
  $('layerChips').innerHTML = `<button data-l="all" class="allchip">All layers ▶</button>` + Object.entries(LAYERS).filter(([, L]) => !L.ens).map(([k, L]) => `<button data-l="${k}" class="${k === S.layer ? 'on' : ''}">${L.name}</button>`).join('');
  $('layerChips').onclick = e => { const k = e.target.dataset.l; if (!k) return; if (k === 'all') return cycleLayers(); stopCycle(); toggleLayer(k); };
  syncChips();
  $('presets').innerHTML = Object.entries(PRESETS).map(([k, p]) => `<button data-p="${k}">${p.name}<small>${p.sub}</small></button>`).join('');
  $('presets').onclick = e => { const b = e.target.closest('button'); if (!b) return; S.preset = b.dataset.p; const p = PRESETS[S.preset]; if (!p.custom && !p.box) { $('shockLat').value = p.lat; $('shockLon').value = p.lon; $('shockRad').value = p.radius; $('shockVar').value = p.parts[0][0]; $('shockMag').value = p.parts[0][1]; } syncShockUI(); };
  $('shockVar').onchange = () => { $('shockMag').value = VARS[$('shockVar').value].def; S.preset = 'custom'; syncShockUI(); };
  for (const id of ['shockMag', 'shockRad', 'shockLat', 'shockLon']) $(id).oninput = () => { S.preset = 'custom'; syncShockUI(); };
  S.shockList = []; renderShockList();
  $('addShock').onclick = () => { S.shockList.push(currentShock()); renderShockList(); toast(`${S.shockList.length} shock(s) combined. Press Run multiverse.`, 'info'); };
  $('shockList').onclick = e => { const i = e.target.dataset.i; if (i !== undefined) { S.shockList.splice(+i, 1); renderShockList(); } };
  $('runBtn').onclick = runMultiverse; $('ensBtn').onclick = runEnsemble; $('resetBtn').onclick = resetBaseline;
  $('reloadBtn').onclick = () => location.reload();
  $('auditBtn').onclick = () => showAudit('shock');
  $('valBtn').onclick = async () => (await import('./validation.js')).openValidation();
  $('whatifBtn').onclick = () => showAudit('warm1');
  $('cmpBtn').onclick = async () => {
    const { openCompare } = await import('./compare.js');
    openCompare({ S, simulate, buildPerturbation, DAY, toast, makePdf: async r => (await import('./report.js')).makeCompareReport({ ...r, commit: COMMIT, lastRun: S.lastRun, model: S.modelName }) });
  };
  $('brief').onclick = e => { if (e.target.closest('.bx')) { $('brief').hidden = true; $('pins').hidden = true; } };
  $('modeSeg').onclick = e => e.target.dataset.mode && setMode(e.target.dataset.mode);
  $('viewSeg').onclick = e => e.target.dataset.view && setView(e.target.dataset.view);
  $('layoutSeg').onclick = e => e.target.dataset.layout && setLayout(e.target.dataset.layout);
  $('speedSeg').onclick = e => { const s = e.target.dataset.speed; if (!s) return; S.speed = +s; for (const b of $('speedSeg').children) b.classList.toggle('on', b === e.target); };
  const menu = $('viewMenu'), vb = $('viewBtn');
  vb.onclick = e => { e.stopPropagation(); menu.hidden = !menu.hidden; vb.setAttribute('aria-expanded', !menu.hidden); };
  menu.onclick = e => e.stopPropagation(); addEventListener('click', () => { menu.hidden = true; vb.setAttribute('aria-expanded', false); });
  const optMap = { optWind: 'wind', optClouds: 'clouds', optRain: 'rain', optBorders: 'borders', optCities: 'cities' };
  for (const [id, key] of Object.entries(optMap)) $(id).onchange = e => {
    S.opts[key] = e.target.checked;
    if (key === 'borders') { map.showBorders = e.target.checked; map.drawBase(); }
    if (key === 'cities') map.showCities = e.target.checked;
    if (key === 'clouds') repaint();
    fieldDirty = true;
  };
  $('optRotate').onchange = e => { globe.autoRotate = e.target.checked || S.layout === 'cinema'; };
  $('reachDay').oninput = () => { if (reachPlayT) playReach(); renderReach(S.reach); }; $('reachPlay').onclick = playReach;
  $('cloudBtn').onclick = () => { const c = $('optClouds'); c.checked = !c.checked; c.dispatchEvent(new Event('change')); $('cloudBtn').setAttribute('aria-pressed', c.checked); $('cloudBtn').classList.toggle('on', c.checked); };
  $('themeBtn').onclick = e => setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light', e);
  $('tourBtn').onclick = tour; $('snapBtn').onclick = snapshot; $('reportBtn').onclick = downloadReport;
  $('aboutBtn').onclick = () => toggleAbout(); $('aboutClose').onclick = () => toggleAbout(false); $('about').onclick = e => { if (e.target.id === 'about') toggleAbout(false); };
  $('citySel').innerHTML = '<option value="">All India (land average)</option>' + CITIES.map((c, i) => `<option value="${i}">${c[0]}</option>`).join('');
  $('citySel').onchange = () => { const c = CITIES[$('citySel').value]; renderCity(); if (!c) return S.view === 'street' && street.overview();
    S.view === 'street' ? street.flyToCity(c[1], c[2]) : S.view === 'globe' ? globe.flyTo(c[1], c[2], 1.9, 1.4) : map.flyTo(c[1], c[2], 8); };
  $('cityGo').onclick = () => { if (!$('citySel').value) { $('citySel').value = '1'; renderCity(); }                // default: Mumbai
    const c = CITIES[$('citySel').value], fresh = !street.map; setView('street'); if (!fresh) street.flyToCity(c[1], c[2]); };
  $('optSat').onchange = e => street.setSatellite(e.target.checked);
  $('stateSel').onchange = e => e.target.value && chooseState(e.target.value);
  $('stateGo').onclick = openStateStreet; $('tipBtn').onclick = tippingPoint;
  $('customDate').oninput = customHint; $('customGo').onclick = runCustomDate; initCustomDate();
  $('alertsBtn').onclick = () => toggleAlerts(); $('alertsClose').onclick = () => toggleAlerts(false);
  for (const id of ['fState', 'fType', 'fStatus']) $(id).onchange = renderAlerts;
  $('playBtn').onclick = () => play();
  $('dayRange').oninput = e => { S.day = +e.target.value; play(false); repaint(); updateTimelineLabels(); renderChart(); };
  $('ticks').innerHTML = Array.from({ length: STEPS + 1 }, (_, d) => `<span>${d === 0 ? 'init' : '+' + d}</span>`).join('');
  $('help').onclick = () => $('help').hidden = true;
  // depth: panel cards tilt toward the pointer with a soft glare
  document.addEventListener('pointermove', e => {
    const c = e.target.closest?.('.card'); if (!c) return;
    const r = c.getBoundingClientRect(), px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
    c.style.setProperty('--ry', `${(px - 0.5) * 9}deg`); c.style.setProperty('--rx', `${(0.5 - py) * 7}deg`);
    c.style.setProperty('--mx', `${px * 100}%`); c.style.setProperty('--my', `${py * 100}%`); c.style.setProperty('--glare', 1);
  });
  document.addEventListener('pointerout', e => { const c = e.target.closest?.('.card'); if (c && !c.contains(e.relatedTarget)) { c.style.setProperty('--rx', '0deg'); c.style.setProperty('--ry', '0deg'); c.style.setProperty('--glare', 0); } });
  // timeline: click a day tick to jump there
  $('ticks').onclick = e => { const i = [...$('ticks').children].indexOf(e.target.closest('span')); if (i < 0) return; play(false); S.day = i; $('dayRange').value = i; repaint(); updateTimelineLabels(); renderChart(); [...$('ticks').children].forEach((t, k) => t.classList.toggle('on', k === i)); };
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.ctrlKey || e.metaKey) return;
    const k = e.key;
    if (e.code === 'Space') { e.preventDefault(); play(); }
    else if (k === 'ArrowRight') stepDay(1); else if (k === 'ArrowLeft') stepDay(-1);
    else if (k === 'g') setView('globe'); else if (k === 'm') setView('map'); else if (k === 'z') setView('street');
    else if (k === 'd') setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
    else if (k === 'f') setLayout(S.layout === 'focus' ? 'full' : 'focus'); else if (k === 'c') setLayout(S.layout === 'cinema' ? 'full' : 'cinema');
    else if (k === 'a') toggleAlerts(); else if (k === 'r') downloadReport(); else if (k === 'i') toggleAbout(); else if (k === 't') tour(); else if (k === 's') snapshot(); else if (k === 'v') vb.click();
    else if (k === '?') $('help').hidden = !$('help').hidden; else if (k === 'Escape') { $('help').hidden = true; menu.hidden = true; toggleAbout(false); }
    else if ('1234'.includes(k)) setMode(['prime', 'multi', 'delta', 'truth'][+k - 1]);
  });
  setTheme(document.documentElement.dataset.theme || 'dark');
  setLayout('full'); syncShockUI(); setMode('prime');
}

let tPrev = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - tPrev) / 1000); tPrev = now; cloudT += dt;
  if (S.playing && S.prime) {
    S.day = Math.min(STEPS, S.day + dt * 0.8 * S.speed); $('dayRange').value = S.day;
    const q = Math.floor(S.day * 20);
    if (q !== lastPaint) { lastPaint = q; repaint(); updateTimelineLabels(); if (q % 4 === 0) renderChart(); }
    if (S.day >= STEPS) play(false);
  }
  globe.frame(dt, frameOpts());
  map.frame(dt, S.opts.wind, S.opts.clouds, fieldDirty); fieldDirty = false;
  updatePins();
  requestAnimationFrame(loop);
}

// header line: start date and time of the last run, always with the not-official note
function runInfo() {
  S.lastRun = new Date();
  $('runInfo').textContent = `Start ${S.win.date} · last run ${S.lastRun.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}`;
}
// server down: say so in the panels instead of leaving them blank
function offline() {
  const msg = `<p class="offline">The INDRA engine is not running at ${API}.<br>Start it with <code>python serve.py --port 8000</code>, then press Reload (top bar).</p>`;
  for (const id of ['cards', 'cityCards', 'riskBars', 'tele']) $(id).innerHTML = msg;
}
// state audit: affected and unaffected states for the current shock, or for a uniform +1 K over India
async function showAudit(kind) {
  if (!S.prime || !S.clim || !S.land) return toast('Wait for the forecast to load first.', 'info');
  const { auditRows, auditHtml } = await import('./audit.js');
  let scen, title, note;
  if (kind === 'shock') {
    if (!S.multi) return toast('Run a shock first (left panel), then open the audit.', 'info');
    scen = S.multi; title = `State audit: ${S.shock.name}`;
    note = 'Each state compared with Timeline Prime over the same 10 days.';
  } else {
    setBusy(true, 'running +1 K over all India…');
    try { scen = await simulate(S.init, buildPerturbation({ box: [6, 38, 65, 101], parts: [['t2m', 1]] }, S.init.subarray(2 * DAY, 3 * DAY)).delta); }
    catch (e) { setBusy(false); return toast(e.message); }
    setBusy(false, `${S.modelName} · ready`);
    title = 'What if +1 K everywhere';
    note = 'Every day of the 10-day run warmed by 1 K over all of India, same start. Rain and wind respond through the model.';
  }
  const rows = await auditRows(S.prime, scen, S.clim);
  $('audit').innerHTML = `<button class="x" id="auditClose" aria-label="Close audit">×</button>` + auditHtml(rows, title, note);
  $('audit').hidden = false;
  $('auditClose').onclick = () => ($('audit').hidden = true);
}

// clear every shock and go back to Timeline Prime
function resetBaseline() {
  if (!S.prime) return;
  S.shockList = []; renderShockList(); S.multi = null; S.shock = null; S.reach = null; S.risk.multi = null; S.alertsMulti = null; S.brief = null;
  $('brief').hidden = true; $('pins').hidden = true; pinEls.forEach(e => e.remove()); pinEls = []; $('reachSec').hidden = true; map.setReach(null);
  globe.setShock(null); map.setShock(null);
  $('prop').innerHTML = ''; $('propNote').textContent = 'Run a multiverse shock to see how far it spreads each day.';
  setMode('prime'); renderImpacts(); renderTelemetry(); renderRisk(); computeAlerts(); toast('Back to the baseline (Timeline Prime).', 'info');
}

async function boot() {
  const msg = t => $('loaderMsg').textContent = t;
  msg('Loading Earth & borders…');
  await Promise.all([globe.loadBorders(), map.loadBorders(), loadStates()]);
  $('stateSel').innerHTML = '<option value="">Select a state…</option>' + [...(await loadStates())].map(f => f.properties.name).sort().map(n => `<option>${n}</option>`).join('');
  msg('Loading initial-condition windows…');
  const res = await fetch('/windows.json');
  // demo initial states are generated locally (not shipped): python tools/make_windows.py
  if (!res.ok) { msg('No demo windows found. Run: python tools/make_windows.py --data data/india_climate_train.nc, then reload.'); return; }
  const meta = await res.json();
  S.windows = meta.windows; S.winData = new Float32Array(await (await fetch('/windows.bin')).arrayBuffer());
  initUI(); requestAnimationFrame(loop);
  msg('Connecting to the INDRA engine…');
  try {
    const h = await post('/health'); S.modelName = h.model_name;
    const st = await post('/static'); S.land = Float32Array.from(st.land.flat());
    status(`${h.model_name} · ready`, 'ok');
  } catch (e) { status('server offline', 'err'); toast(`INDRA engine not reachable at ${API}. Run “python serve.py --port 8000”, then reload.`); offline(); }
  $('modelVer').textContent = (S.modelName || 'INDRA-v1').split(' (')[0];
  document.querySelectorAll('.srcline').forEach(e => (e.textContent = `Source: kit data 2005–2021 · ${S.modelName || 'INDRA-v1'} model · commit ${COMMIT}`));
  await Promise.race([globe.ready, new Promise(r => setTimeout(r, 4000))]);
  $('loader').classList.add('done'); globe.flyTo(21, 82, 2.5);
  if (S.modelName) await loadWindow(0);
}
boot();
