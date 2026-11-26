// Scenario comparison: the same start date under several shocks, side by side as light 2D canvas panels
// (no WebGL, so it stays fast on demo hardware). Each scenario is one live /simulate call.
import { BBOX, LAYERS, TEX_W, TEX_H, layerGrid, paintField, paintLegend } from './field.js';
import { impacts } from './impact.js';
import { generateAlerts } from './alerts.js';
import { loadStates } from './state.js';

export const SCENARIOS = [
  { key: 'base', name: 'No shock', sub: 'Timeline Prime (reference)', shock: null },
  { key: 'heat', name: 'Strong heat shock', sub: '+6 K over north and central India', shock: { lat: 25, lon: 78, radius: 7, parts: [['t2m', 6]] } },
  { key: 'stall', name: 'Monsoon stall', sub: '−40% Arabian Sea wind', shock: { box: [8, 20, 65, 75], lat: 14, lon: 70, radius: 6, parts: [['wind', -40]] } },
  { key: 'dry', name: 'Rain deficit', sub: '−40% moisture, −30 mm/day rain, central India', shock: { lat: 21, lon: 78, radius: 7, parts: [['tcwv', -40], ['tp', -30]] } },
];
export const MORE = [
  { key: 'bob', name: 'Warm Bay of Bengal', sub: '+3 K, +15% moisture', shock: { box: [8, 20, 82, 94], lat: 14, lon: 88, radius: 6, parts: [['t2m', 3], ['tcwv', 15]] } },
  { key: 'cyclone', name: 'Cyclone seed', sub: '−8 hPa, +20% moisture, Bay of Bengal', shock: { lat: 15, lon: 88, radius: 3, parts: [['msl', -8], ['tcwv', 20]] } },
  { key: 'ghats', name: 'Dry Western Ghats', sub: '−40% moisture', shock: { lat: 14, lon: 75, radius: 3.5, parts: [['tcwv', -40]] } },
  { key: 'cool', name: 'Cool Arabian Sea', sub: '−3 K over the Arabian Sea', shock: { box: [8, 22, 62, 72], lat: 15, lon: 67, radius: 7, parts: [['t2m', -3]] } },
];
const LAYER_OPTS = ['tp', 't2m', 'wind', 'msl', 'tcwv'];
const W = 360, H = 320;                                     // panel canvas, same 36:32 aspect as the grid
const X = lon => (lon - BBOX.w) / (BBOX.e - BBOX.w) * W, Y = lat => (BBOX.n - lat) / (BBOX.n - BBOX.s) * H;
const $ = id => document.getElementById(id);

let C = null, outline = null;
const tmp = Object.assign(document.createElement('canvas'), { width: TEX_W, height: TEX_H });

/** ctx = { S, simulate, buildPerturbation, DAY, toast, makePdf } */
export async function openCompare(ctx) {
  const { S } = ctx;
  if (!S.prime || !S.clim || !S.land) return ctx.toast('Wait for the forecast to load first.', 'info');
  if (!outline) outline = await statePath();
  C = { ctx, runs: new Map(), day: 3, layer: 'tp', delta: false, more: false, timer: null, win: S.win.date };
  C.runs.set('base', stats(S.prime));
  shell(); $('compare').hidden = false; render();
  await run(SCENARIOS);
}

export function closeCompare() { if (!C) return; clearInterval(C.timer); $('compare').hidden = true; }

async function run(list) {
  const { S, simulate, buildPerturbation, DAY } = C.ctx, last = S.init.subarray(2 * DAY, 3 * DAY);
  for (const sc of list) {
    if (C.runs.has(sc.key)) continue;
    try {
      const days = await simulate(S.init, buildPerturbation(sc.shock, last).delta);
      if (!C || $('compare').hidden) return;
      C.runs.set(sc.key, stats(days)); render();
    } catch (e) { C.ctx.toast(`Scenario “${sc.name}” failed: ${e.message}`); }
  }
}

function stats(days) {
  const { S } = C.ctx, imp = impacts(days, S.clim, S.land), al = generateAlerts(days, S.clim, S.land, null);
  return { days, imp, alerts: al.length, high: al.filter(a => a.severity === 'High' || a.severity === 'Critical').length };
}

const visible = () => (C.more ? [...SCENARIOS, ...MORE] : SCENARIOS);

function shell() {
  $('compare').innerHTML = `
    <div class="cmphead">
      <h2>Compare scenarios <span class="muted small">start ${C.win} · same initial state, different shocks</span></h2>
      <label>Layer <select id="cmpLayer">${LAYER_OPTS.map(k => `<option value="${k}">${LAYERS[k].name} (${LAYERS[k].unit})</option>`).join('')}</select></label>
      <label class="sw"><input type="checkbox" id="cmpDelta" /><i></i>Change vs no shock</label>
      <label class="cmpday">Day <input type="range" id="cmpDay" min="1" max="10" step="1" value="${C.day}" aria-label="Forecast day" /><b id="cmpDayOut">+${C.day}</b></label>
      <button class="ghost slim" id="cmpPlay">▶ Play</button>
      <span class="cmpleg"><span id="cmpLo"></span><canvas id="cmpLegend" width="120" height="8"></canvas><span id="cmpHi"></span></span>
      <button class="ghost slim" id="cmpMore">See more scenarios</button>
      <button class="primary slim" id="cmpPdf">Comparison PDF</button>
      <button class="x" id="cmpClose" aria-label="Close comparison">×</button>
    </div>
    <div class="cmpgrid" id="cmpGrid"></div>
    <p class="srcline">Each panel is a live /simulate run from the same 3 input days. Alerts use IMD rain thresholds (64.5 / 115.6 / 204.5 mm/day) and the heat rule; heatwave days = days with more than 5% of land under heat stress. Not an official IMD forecast.</p>`;
  $('cmpLayer').onchange = e => { C.layer = e.target.value; render(); };
  $('cmpDelta').onchange = e => { C.delta = e.target.checked; render(); };
  $('cmpDay').oninput = e => { C.day = +e.target.value; render(); };
  $('cmpPlay').onclick = () => {
    if (C.timer) { clearInterval(C.timer); C.timer = null; $('cmpPlay').textContent = '▶ Play'; return; }
    $('cmpPlay').textContent = '■ Stop';
    C.timer = setInterval(() => { C.day = C.day % 10 + 1; $('cmpDay').value = C.day; render(); }, 700);
  };
  $('cmpMore').onclick = async () => {
    C.more = !C.more; $('cmpMore').textContent = C.more ? 'Show fewer' : 'See more scenarios'; render();
    if (C.more) await run(MORE);
  };
  $('cmpPdf').onclick = () => pdf();
  $('cmpClose').onclick = closeCompare;
}

function render() {
  const grid = $('cmpGrid'), list = visible(), base = C.runs.get('base');
  $('cmpDayOut').textContent = `+${C.day}`;
  const [lo, hi, unit] = paintLegend($('cmpLegend'), C.layer, C.delta ? 'delta' : 'abs');
  $('cmpLo').textContent = lo; $('cmpHi').textContent = `${hi} ${unit}`;
  if (grid.children.length !== list.length) {
    grid.innerHTML = list.map(sc => `<article class="cpanel" data-k="${sc.key}"><header><b>${sc.name}</b><span>${sc.sub}</span></header>
      <canvas width="${W}" height="${H}"></canvas><div class="cstats"></div></article>`).join('');
  }
  for (const sc of list) {
    const el = grid.querySelector(`[data-k="${sc.key}"]`), r = C.runs.get(sc.key);
    draw(el.querySelector('canvas'), sc, r, base);
    el.querySelector('.cstats').innerHTML = r ? chips(r, sc.key === 'base' ? null : base) : '<span class="muted small">simulating…</span>';
  }
}

function chips(r, base) {
  const d = (v, b, dig, u) => (b == null ? '' : `<em class="${v - b > 0 ? 'up' : v - b < 0 ? 'down' : ''}">${v - b > 0 ? '+' : ''}${(v - b).toFixed(dig)}${u}</em>`);
  const I = r.imp, B = base?.imp;
  return [
    ['Alerts', `${r.alerts}`, `${r.high} High+`, d(r.alerts, base?.alerts, 0, '')],
    ['Heatwave days', `${I.heatDays}`, 'of 10', d(I.heatDays, B?.heatDays, 0, '')],
    ['Flood area', `${(I.floodArea / 1000).toFixed(0)}`, 'k km²', d(I.floodArea / 1000, B && B.floodArea / 1000, 0, '')],
    ['Crop stress', `${I.cropStress.toFixed(1)}`, '%', d(I.cropStress, B?.cropStress, 1, ' pts')],
  ].map(([k, v, u, dd]) => `<div><span>${k}</span><b>${v}</b> <small>${u}</small> ${dd}</div>`).join('');
}

function draw(cv, sc, r, base) {
  const g = cv.getContext('2d');
  g.fillStyle = '#0a1424'; g.fillRect(0, 0, W, H);
  g.fillStyle = '#16263d'; g.fill(outline);
  if (r) {
    let grid = layerGrid(r.days[C.day], C.layer), mode = 'abs';
    if (C.delta) { const b = layerGrid(base.days[C.day], C.layer); grid = grid.map((v, k) => v - b[k]); mode = 'delta'; }
    const t = tmp.getContext('2d'); t.clearRect(0, 0, TEX_W, TEX_H); paintField(t, grid, { layer: C.layer, mode });
    g.imageSmoothingEnabled = true; g.drawImage(tmp, 0, 0, W, H);
  } else { g.fillStyle = '#8aa0c0'; g.font = '13px Inter, sans-serif'; g.fillText('simulating…', W / 2 - 36, H / 2); }
  g.strokeStyle = 'rgba(220,232,250,.55)'; g.lineWidth = 0.7; g.stroke(outline);
  const s = sc.shock; if (!s) return;
  g.strokeStyle = '#ffd166'; g.lineWidth = 1.6; g.setLineDash([5, 4]); g.beginPath();
  if (s.box) g.rect(X(s.box[2]), Y(s.box[1]), X(s.box[3]) - X(s.box[2]), Y(s.box[0]) - Y(s.box[1]));
  else g.ellipse(X(s.lon), Y(s.lat), s.radius / (BBOX.e - BBOX.w) * W / Math.cos(s.lat * Math.PI / 180), s.radius / (BBOX.n - BBOX.s) * H, 0, 0, 2 * Math.PI);
  g.stroke(); g.setLineDash([]);
}

// state outlines in panel pixels, built once
async function statePath() {
  const p = new Path2D();
  for (const f of await loadStates()) {
    const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const poly of polys) for (const ring of poly) ring.forEach(([lon, lat], i) => (i ? p.lineTo(X(lon), Y(lat)) : p.moveTo(X(lon), Y(lat))));
  }
  return p;
}

async function pdf() {
  const list = visible().filter(sc => C.runs.has(sc.key));
  if (list.length < 2) return C.ctx.toast('Wait for the scenarios to finish first.', 'info');
  $('cmpPdf').disabled = true;
  try {
    const grid = $('cmpGrid'), base = C.runs.get('base');
    await C.ctx.makePdf({
      win: C.ctx.S.win, day: C.day, layer: `${LAYERS[C.layer].name}${C.delta ? ' (change vs no shock)' : ''}`, unit: LAYERS[C.layer].unit,
      scenarios: list.map(sc => {
        const r = C.runs.get(sc.key);
        return { name: sc.name, sub: sc.sub, image: grid.querySelector(`[data-k="${sc.key}"] canvas`).toDataURL('image/jpeg', 0.9),
          alerts: r.alerts, high: r.high, heatDays: r.imp.heatDays, flood: r.imp.floodArea / 1000, crop: r.imp.cropStress,
          peakRain: r.imp.peakRain, rainDep: r.imp.rainDeparture, base: sc.key === 'base' };
      }),
      baseRef: { alerts: base.alerts, heatDays: base.imp.heatDays, flood: base.imp.floodArea / 1000, crop: base.imp.cropStress },
    });
  } catch (e) { C.ctx.toast(`PDF failed: ${e.message}`); }
  $('cmpPdf').disabled = false;
}
