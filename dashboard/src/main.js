import { NC, NLAT, NLON, CELLS, LAYERS, LAT0, LON0, BBOX, TEX_W, TEX_H, CLOUD_W, CLOUD_H, idx, layerGrid, lerpDay, paintField, paintClouds, paintLegend, bilinear, inDomain } from './field.js';
import { Globe } from './globe.js';
import { Map2D, CITIES } from './map2d.js';
import { Street3D } from './street3d.js';
import { impacts, landMeanSeries, propagation } from './impact.js';
import { generateAlerts, diffAlerts, ADVICE, SEVERITIES } from './alerts.js';

// ?api=http://host:port overrides; 127.0.0.1 avoids Windows' slow IPv6-first "localhost" lookup
const API = new URLSearchParams(location.search).get('api') || `http://${!location.hostname || location.hostname === 'localhost' ? '127.0.0.1' : location.hostname}:8000`;
const STEPS = 10, DAY = 6 * CELLS, STD = [13.46, 7.55, 2.9, 2.43, 619, 20.49];
const $ = id => document.getElementById(id);
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

const S = {
  view: 'globe', mode: 'prime', layer: 't2m', day: 0, playing: false, speed: 1,
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
const globe = new Globe($('globe'), { onHover: hover, onPick: pick });
const map = new Map2D($('map2d'), { onHover: hover, onPick: pick, fieldCanvas, cloudCanvas });
const street = new Street3D($('map3d'), fieldCanvas, { onHover: hover, onPick: pick });

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
function repaint() {
  const t = S.day, L = LAYERS[S.layer]; let grid, weatherDay;
  if (S.mode === 'delta' && S.multi) {
    const a = gridFor('multi', t), b = gridFor('prime', t); grid = a && b && a.map((v, k) => v - b[k]); weatherDay = dayAt(S.multi, t);
  } else if (S.mode === 'spread' && S.spread) {
    grid = S.spread[Math.round(t)]; weatherDay = dayAt(S.prime, t);
  } else {
    weatherDay = dayAt(src(), t);
    grid = L.ml ? gridFor(S.mode === 'multi' && S.multi ? 'multi' : 'prime', t) : weatherDay && layerGrid(weatherDay, S.layer);
  }
  if (!weatherDay) return;
  const ctx = fieldCanvas.getContext('2d'); ctx.clearRect(0, 0, TEX_W, TEX_H);
  if (grid) paintField(ctx, grid, { layer: S.layer, mode: S.mode === 'delta' ? 'delta' : S.mode === 'spread' ? 'spread' : 'abs' });
  const cc = cloudCanvas.getContext('2d'); cc.clearRect(0, 0, CLOUD_W, CLOUD_H); if (S.opts.clouds) paintClouds(cc, weatherDay, cloudT);
  for (const [dst, from] of [[globe.fieldCanvas, fieldCanvas], [globe.cloudCanvas, cloudCanvas]]) { const g = dst.getContext('2d'); g.clearRect(0, 0, dst.width, dst.height); g.drawImage(from, 0, 0); }
  globe.field.tex.needsUpdate = true; globe.clouds.tex.needsUpdate = true;
  globe.setDay(weatherDay); map.setWind(weatherDay); S.curGrid = grid; S.curDay = weatherDay;
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
  const extra = (S.mode === 'delta' || S.mode === 'spread') && S.curGrid ? `<tr><td>${S.mode === 'delta' ? 'Δ shock' : 'Ensemble σ'}</td><td>${bilinear(S.curGrid, gi, gj).toFixed(2)} ${LAYERS[S.layer].unit}</td></tr>` : '';
  tooltip.innerHTML = `<b>${h.lat.toFixed(1)}°N ${h.lon.toFixed(1)}°E</b><table>${rows.join('')}${extra}</table>`;
  tooltip.hidden = false; tooltip.style.left = Math.min(h.x + 16, innerWidth - 200) + 'px'; tooltip.style.top = Math.min(h.y + 16, innerHeight - 220) + 'px';
}

// ---------------------------------------------------------------- right panel
const fmt = (v, d = 1) => Number.isFinite(v) ? v.toLocaleString('en-IN', { maximumFractionDigits: d, minimumFractionDigits: d }) : '–';
function card(k, v, unit, color, deltaTxt, pct, wide) {
  return `<div class="card${wide ? ' wide' : ''}" style="--c:${color}"><div class="k">${k}</div><div class="v">${v} <small>${unit}</small></div>${deltaTxt || ''}${pct != null ? `<div class="bar"><i style="width:${Math.max(2, Math.min(100, pct))}%"></i></div>` : ''}</div>`;
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
    card('Wind power @100 m', fmt(P.windPower, 0), 'W/m²', '#3ddc97', dTxt(P.windPower, M?.windPower, ' W/m²', 0), P.windPower / 4),
    card('Peak daily rain', fmt(P.peakRain, 0), 'mm', '#b23cff', dTxt(P.peakRain, M?.peakRain, ' mm', 0), P.peakRain / 3),
    card('Ensemble confidence', ens ? fmt(ens.conf, 0) : '—', ens ? '%' : '', '#9c8fe0', ens ? `<div class="d">day-10 σ ${fmt(ens.sigma, 2)} ${LAYERS[ens.layer].unit} (${LAYERS[ens.layer].name.toLowerCase()})</div>` : `<div class="d">run the 5-member ensemble</div>`, ens?.conf, true),
  ].join('');
  renderRisk(); renderChart(); renderCity();
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
  $('modeSeg').querySelector('[data-mode=spread]')?.remove(); globe.setShock(null); map.setShock(null);
  S.day = 0; $('dayRange').value = 0;
  setBusy(true, 'simulating Timeline Prime…');
  try {
    const [prime, clim] = await Promise.all([simulate(S.init, null), post('/climatology', { doy_sincos: S.win.doy_sincos, steps: STEPS })]);
    S.prime = prime; S.clim = clim.clim.map(c => Float32Array.from(c.flat(Infinity)));
    const zero = await simulate(S.init, new Float32Array(DAY));            // live determinism check: zero shock = no change
    S.nullDiff = Math.max(...zero.slice(1).map((d, k) => d.reduce((m, v, j) => Math.max(m, Math.abs(v - S.prime[k + 1][j]) / STD[Math.floor(j / CELLS)]), 0)));
    setBusy(false, `${S.modelName} · ready`);
  } catch (e) { setBusy(false); status('server unreachable', 'err'); toast(`Cannot reach ${API}: ${e.message}. Start it with “python serve.py --port 8000”.`); return; }
  if (S.mode !== 'truth') setMode('prime');
  repaint(); renderImpacts(); renderTelemetry(); updateTimelineLabels();
  $('prop').innerHTML = ''; $('propNote').textContent = 'Run a multiverse shock to see how far it spreads each day.';
  computeAlerts();
  S.risk.prime = await riskFor(S.prime); renderRisk(); renderTelemetry(); computeAlerts(); if (LAYERS[S.layer].ml) repaint();
}

async function runMultiverse() {
  const sh = currentShock(), last = S.init.subarray(2 * DAY, 3 * DAY), { delta, touched } = buildPerturbation(sh, last);
  if (!touched) return warnShock(true);
  setBusy(true, `branching universe: ${sh.name}…`);
  try { S.multi = await simulate(S.init, delta); S.shock = sh; } catch (e) { toast(e.message); setBusy(false); return; }
  setBusy(false, `${S.modelName} · ready`);
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
  const sigma = S.spread[STEPS].reduce((a, v) => a + v, 0) / CELLS;
  S.spreadSummary = { sigma, conf: 100 * Math.max(0, 1 - sigma / LAYERS[lay].dRange), layer: lay };
  if (lay !== S.layer) setLayer(lay);
  if (!$('modeSeg').querySelector('[data-mode=spread]')) { const b = document.createElement('button'); b.dataset.mode = 'spread'; b.textContent = 'Ensemble σ'; $('modeSeg').append(b); }
  setBusy(false, `${S.modelName} · ready`); setMode('spread'); repaint(); renderImpacts();
}

// ---------------------------------------------------------------- view state
function setMode(m) {
  if ((m === 'multi' || m === 'delta') && !S.multi) { toast('Run a multiverse shock first (left panel).', 'info'); m = 'prime'; }
  if (m === 'spread' && !S.spread) m = 'prime';
  if (m === 'truth' && LAYERS[S.layer].ml) { toast('ML risk is a forecast product — showing observed temperature.', 'info'); setLayer('t2m', true); }
  S.mode = m; for (const b of $('modeSeg').children) b.classList.toggle('on', b.dataset.mode === m);
  const [lo, hi, unit] = paintLegend($('legend'), S.layer, m === 'delta' ? 'delta' : m === 'spread' ? 'spread' : 'abs');
  $('legLo').textContent = lo; $('legHi').textContent = hi; $('legUnit').textContent = unit;
  repaint(); updateTimelineLabels(); renderRisk(); renderAlerts();
}
function setLayer(k, quiet) {
  if (LAYERS[k].ml && !S.hasRisk) return toast('The ML risk model is not loaded on the server (run train_risk.py).');
  S.layer = k; for (const b of $('layerChips').children) b.classList.toggle('on', b.dataset.l === k);
  $('layerNote').textContent = LAYERS[k].ml ? `INDRA-Risk: probability that ${k === 'flood' ? 'daily rain exceeds 64.5 mm (IMD heavy)' : 'daily-mean heat stress occurs'}, learned from INDRA's own forecasts vs what happened.` : '';
  if (S.mode === 'spread') { S.spread = null; S.spreadSummary = null; S.mode = 'prime'; }
  if (!quiet) { setMode(S.mode === 'truth' && LAYERS[k].ml ? 'prime' : S.mode); renderChart(); }
}
function setView(v) {
  S.view = v; document.body.classList.toggle('view-map', v === 'map'); document.body.classList.toggle('view-street', v === 'street');
  globe.active = v === 'globe'; map.active = v === 'map'; street.setActive(v === 'street');
  if (v === 'street' && !street.map) street.ensure().then(() => { street.setActive(S.view === 'street'); const c = CITIES[$('citySel').value]; c ? street.flyToCity(c[1], c[2]) : street.overview(); });
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
function play(on) { S.playing = on ?? !S.playing; $('playIcon').setAttribute('d', S.playing ? 'M6 4h4v16H6zM14 4h4v16h-4z' : 'M7 4v16l13-8z'); if (S.playing && S.day >= STEPS - 0.01) S.day = 0; }
function stepDay(d) { play(false); S.day = Math.max(0, Math.min(STEPS, Math.round(S.day) + d)); $('dayRange').value = S.day; repaint(); updateTimelineLabels(); renderChart(); }
const frameOpts = () => ({ ...S.opts });

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
  $('cityCards').innerHTML = CITY_VARS.map(([k, n, u, dig, col]) => card(`${n} · day +${d}`, P[d][k].toFixed(dig), u, col,
    M ? `<div class="d">multiverse ${delta(P[d][k], M[d][k], dig) || 'no change'}</div>` : '')).join('');
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
    const what = a.type === 'rain' ? (a.severity === 'Low' ? 'Heavy-rain watch' : 'Heavy rain') : (a.severity === 'Low' ? 'Heat watch' : 'Heat stress');
    const val = a.type === 'rain' ? `peak ${Math.round(a.value)} mm/day` : `daily mean up to ${a.value.toFixed(1)} °C`;
    const adv = ADVICE[a.type];
    return `<li data-n="${n}" style="--sc:${SEV_COL[a.severity]}"><div class="top"><span class="sev">${a.severity.toUpperCase()}</span><b>${what}</b>${a.status === 'new' ? '<span class="tag">NEW in multiverse</span>' : a.status === 'worse' ? '<span class="tag">MORE SEVERE</span>' : ''}</div>
      <div class="meta">Day +${a.day} · ${a.region} · ${val} · ${a.cells} cell${a.cells > 1 ? 's' : ''} (~${(a.cells * 12).toLocaleString('en-IN')}k km²)</div>
      <details onclick="event.stopPropagation()"><summary>Dos &amp; don'ts</summary><ul>${adv.do.map(t => `<li>✔ ${t}</li>`).join('')}${adv.dont.map(t => `<li>✘ ${t}</li>`).join('')}</ul></details></li>`;
  }).join('') : `<li class="empty">No alerts match. ${all.length ? 'Try other filters.' : 'This universe has no heavy-rain or heat alerts.'}</li>`;
  $('alertList').onclick = e => {
    const a = shown[e.target.closest('li[data-n]')?.dataset.n]; if (!a) return;
    setLayer(a.type === 'rain' ? 'tp' : 't2m'); stepDay(a.day - Math.round(S.day));
    S.view === 'globe' ? globe.flyTo(a.lat, a.lon, 2.0, 1.4) : map.flyTo(a.lat, a.lon, 7);
  };
}
function toggleAlerts(on) {
  const open = on ?? !document.body.classList.contains('alerts-open');
  document.body.classList.toggle('alerts-open', open); $('alertsBtn').classList.toggle('on', open); $('alertsBtn').setAttribute('aria-expanded', open);
}

// guided tour: every event, fly in, show the caption, play its 10 days, move on (any click on Tour stops it)
async function tour() {
  if (S.tour) { S.tour = null; $('tourBtn').classList.remove('on'); $('caption').hidden = true; setLayout('full'); return; }
  const id = S.tour = {}; $('tourBtn').classList.add('on'); setLayout('cinema');
  for (let i = 0; i < S.windows.length && S.tour === id; i++) {
    const w = S.windows[i]; $('caption').innerHTML = `<b>${w.title}</b><span>${w.desc} · ${w.date}</span>`; $('caption').hidden = false;
    setLayer(/flood|monsoon|Kerala|Konkan/i.test(w.title) ? 'tp' : /heat/i.test(w.title) ? 't2m' : /Cyclone/i.test(w.title) ? 'wind' : 't2m');
    await loadWindow(i); if (S.tour !== id) break;
    if (S.view === 'globe') globe.flyTo(21 + (Math.random() - 0.5) * 6, 82 + (Math.random() - 0.5) * 10, 2.1 + Math.random() * 0.4, 3);
    setMode('prime'); S.day = 0; play(true);
    await new Promise(r => { const t = setInterval(() => { if (!S.playing || S.tour !== id) { clearInterval(t); r(); } }, 200); });
    await new Promise(r => setTimeout(r, 900));
  }
  if (S.tour === id) tour();
}

function snapshot() {
  let url;
  if (S.view === 'globe') { globe.frame(0, frameOpts()); url = globe.snapshot(); }
  else {
    const c = document.createElement('canvas'), m = map.c.mapBase; c.width = m.width; c.height = m.height; const g = c.getContext('2d');
    for (const k of ['mapBase', 'mapField', 'mapWind', 'mapUi']) g.drawImage(map.c[k], 0, 0); url = c.toDataURL('image/png');
  }
  const a = Object.assign(document.createElement('a'), { href: url, download: `indra-${S.win?.date || 'view'}-day${Math.round(S.day)}-${S.mode}.png` }); a.click();
  toast('Snapshot saved.', 'info');
}

function initUI() {
  $('windowSel').innerHTML = S.windows.map((w, i) => `<option value="${i}">${w.title} · ${w.date}</option>`).join('');
  $('windowSel').onchange = e => loadWindow(+e.target.value);
  $('layerChips').innerHTML = Object.entries(LAYERS).map(([k, L]) => `<button data-l="${k}" class="${k === S.layer ? 'on' : ''}${L.ml ? ' ml' : ''}">${L.name}</button>`).join('');
  $('layerChips').onclick = e => { const k = e.target.dataset.l; if (k) setLayer(k); };
  $('presets').innerHTML = Object.entries(PRESETS).map(([k, p]) => `<button data-p="${k}">${p.name}<small>${p.sub}</small></button>`).join('');
  $('presets').onclick = e => { const b = e.target.closest('button'); if (!b) return; S.preset = b.dataset.p; const p = PRESETS[S.preset]; if (!p.custom && !p.box) { $('shockLat').value = p.lat; $('shockLon').value = p.lon; $('shockRad').value = p.radius; $('shockVar').value = p.parts[0][0]; $('shockMag').value = p.parts[0][1]; } syncShockUI(); };
  $('shockVar').onchange = () => { $('shockMag').value = VARS[$('shockVar').value].def; S.preset = 'custom'; syncShockUI(); };
  for (const id of ['shockMag', 'shockRad', 'shockLat', 'shockLon']) $(id).oninput = () => { S.preset = 'custom'; syncShockUI(); };
  $('runBtn').onclick = runMultiverse; $('ensBtn').onclick = runEnsemble;
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
  $('themeBtn').onclick = e => setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light', e);
  $('tourBtn').onclick = tour; $('snapBtn').onclick = snapshot;
  $('citySel').innerHTML = '<option value="">All India (land average)</option>' + CITIES.map((c, i) => `<option value="${i}">${c[0]}</option>`).join('');
  $('citySel').onchange = () => { const c = CITIES[$('citySel').value]; renderCity(); if (!c) return S.view === 'street' && street.overview();
    S.view === 'street' ? street.flyToCity(c[1], c[2]) : S.view === 'globe' ? globe.flyTo(c[1], c[2], 1.9, 1.4) : map.flyTo(c[1], c[2], 8); };
  $('cityGo').onclick = () => { if (!$('citySel').value) { $('citySel').value = '1'; renderCity(); }                // default: Mumbai
    const c = CITIES[$('citySel').value], fresh = !street.map; setView('street'); if (!fresh) street.flyToCity(c[1], c[2]); };
  $('optSat').onchange = e => street.setSatellite(e.target.checked);
  $('alertsBtn').onclick = () => toggleAlerts(); $('alertsClose').onclick = () => toggleAlerts(false);
  for (const id of ['fState', 'fType', 'fStatus']) $(id).onchange = renderAlerts;
  $('playBtn').onclick = () => play();
  $('dayRange').oninput = e => { S.day = +e.target.value; play(false); repaint(); updateTimelineLabels(); renderChart(); };
  $('ticks').innerHTML = Array.from({ length: STEPS + 1 }, (_, d) => `<span>${d === 0 ? 'init' : '+' + d}</span>`).join('');
  $('help').onclick = () => $('help').hidden = true;
  addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.ctrlKey || e.metaKey) return;
    const k = e.key;
    if (e.code === 'Space') { e.preventDefault(); play(); }
    else if (k === 'ArrowRight') stepDay(1); else if (k === 'ArrowLeft') stepDay(-1);
    else if (k === 'g') setView('globe'); else if (k === 'm') setView('map'); else if (k === 'z') setView('street');
    else if (k === 'd') setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
    else if (k === 'f') setLayout(S.layout === 'focus' ? 'full' : 'focus'); else if (k === 'c') setLayout(S.layout === 'cinema' ? 'full' : 'cinema');
    else if (k === 'a') toggleAlerts(); else if (k === 't') tour(); else if (k === 's') snapshot(); else if (k === 'v') vb.click();
    else if (k === '?') $('help').hidden = !$('help').hidden; else if (k === 'Escape') { $('help').hidden = true; menu.hidden = true; }
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
  requestAnimationFrame(loop);
}

async function boot() {
  const msg = t => $('loaderMsg').textContent = t;
  msg('Loading Earth & borders…');
  await Promise.all([globe.loadBorders(), map.loadBorders()]);
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
  } catch (e) { status('server offline', 'err'); toast(`INDRA engine not reachable at ${API}. Run “python serve.py --port 8000”, then reload.`); }
  await Promise.race([globe.ready, new Promise(r => setTimeout(r, 4000))]);
  $('loader').classList.add('done'); globe.flyTo(21, 82, 2.5);
  if (S.modelName) await loadWindow(0);
}
boot();
