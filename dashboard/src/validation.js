// Rolling-origin validation panel. Numbers come from public/validation/<year>.json (exported offline by
// rolling_export.py). Hover a chart to read a day; click it to open the full-screen view with details.
// The year replay steps through the recorded daily results at your speed, and pauses on demand.
const YEARS = [2019, 2020, 2021];
// 2019 and 2020: the recipe retrained on earlier years only. 2021: the shipped model (it used 2021 to choose some settings).
const LABEL = { 2019: 'retrained recipe (trained to 2018)', 2020: 'retrained recipe (trained to 2019)', 2021: 'shipped model (2021 used for some settings)' };
const $ = id => document.getElementById(id);
let DATA = [], timer = null;

export async function openValidation() {
  const el = $('validation'); el.hidden = false;
  DATA = await Promise.all(YEARS.map(y => fetch(`/validation/${y}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null)));
  if (DATA.every(g => !g)) {
    el.innerHTML = `${closeBtn()}<h2>Rolling-origin validation</h2><p class="offline">The validation files are missing. Generate them with <code>python rolling_export.py --year 2019 --weights weights/ro2019.pt --out dashboard/public/validation/2019.json</code> (and 2020, 2021).</p>`;
    $('valClose').onclick = closeValidation; return;
  }
  el.innerHTML = `${closeBtn()}
    <h2>Rolling-origin validation <span class="muted small">the recipe retrained on earlier years only; validation, not hidden-test results</span></h2>
    <table class="audit"><tr><th>Year</th><th>Skill (mean)</th><th>Heavy-rain CSI d1</th><th>d4</th><th>d8</th></tr>
      ${YEARS.map((y, i) => `<tr><td>${y} <span class="muted small">${LABEL[y]}</span></td>${row(DATA[i])}</tr>`).join('')}</table>
    <p class="muted small">Skill = RMSE skill vs the better of persistence or climatology (1 perfect, 0 no better). CSI = heavy-rain (≥ 10 mm/day) hit score (1 perfect), the judge's definition.</p>
    <div class="vgrid">${DATA.map((g, i) => (g ? card(g, YEARS[i]) : `<div class="vchart"><b>${YEARS[i]}</b><p class="muted small">not exported</p></div>`)).join('')}</div>
    <div class="vreplay">
      <button class="ghost slim" id="vPlay">▶ Replay year</button>
      <select id="vYear" aria-label="Year to replay">${YEARS.map((y, i) => (DATA[i] ? `<option value="${i}">${y}</option>` : '')).join('')}</select>
      <input type="range" id="vScrub" min="0" max="0" value="0" aria-label="Day in the replay" />
      <span id="vReadout" class="muted small">Pick a year and press play. Pause any time, then drag to a day.</span>
    </div>
    <div class="vfoot muted small">Land-mean rain, day 1 of each window: forecast (orange) vs observed (blue). Source: kit data 2005–2021 · INDRA-v1 recipe. Not an official IMD forecast.</div>
    <div id="vDetail" class="vdetail" hidden></div>`;
  $('valClose').onclick = closeValidation;
  el.onclick = e => { if (e.target === el) closeValidation(); };
  el.querySelectorAll('.vchart svg').forEach((svg, k) => bindChart(svg, DATA.filter(Boolean)[k]));
  el.querySelectorAll('.vchart').forEach((box, k) => box.onclick = () => openDetail(DATA.filter(Boolean)[k]));
  $('vPlay').onclick = togglePlay; $('vYear').onchange = () => { stopPlay(); setReplay(); }; $('vScrub').oninput = e => { stopPlay(); showDay(+e.target.value); };
  setReplay();
}

export function closeValidation() { stopPlay(); const el = $('validation'); if (el) el.hidden = true; }
const closeBtn = () => `<button class="x" id="valClose" aria-label="Close validation">×</button>`;
const row = g => g ? `<td>${g.skill_mean.toFixed(3)}</td><td>${g.leads.d1.csi.toFixed(2)}</td><td>${g.leads.d4.csi.toFixed(2)}</td><td>${g.leads.d8.csi.toFixed(2)}</td>` : '<td colspan="4">—</td>';

function card(g, year) {
  return `<div class="vchart" data-year="${year}" title="Click for full-screen detail; hover the chart to read a day">
    <b>${year}</b> <span class="muted small">${LABEL[year]} · ${g.windows} windows · skill ${g.skill_mean.toFixed(2)}</span>
    <svg viewBox="0 0 300 124" role="img" aria-label="Forecast vs observed land-mean rain ${year}">${paths(g.series, 300, 110)}</svg>
    <div class="vread muted small">hover a day</div></div>`;
}

const extent = s => Math.max(10, ...s.map(p => Math.max(p.pred_rain, p.obs_rain))) * 1.05;
function paths(s, W, H, mx = extent(s)) {
  const X = i => (i / Math.max(1, s.length - 1)) * W, Y = v => H - (v / mx) * H;
  const line = k => s.map((p, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(p[k]).toFixed(1)}`).join(' ');
  return `<path d="${line('obs_rain')}" fill="none" stroke="#46d3ff" stroke-width="1.4" />
    <path d="${line('pred_rain')}" fill="none" stroke="#ff7a45" stroke-width="1.4" />
    <line class="vcross" x1="0" x2="0" y1="0" y2="${H}" stroke="#fff" stroke-opacity=".5" stroke-dasharray="3 3" visibility="hidden" />
    <text x="0" y="${H + 12}" font-size="9" fill="currentColor" opacity=".6">${s[0].date}</text>
    <text x="${W}" y="${H + 12}" font-size="9" fill="currentColor" opacity=".6" text-anchor="end">${s[s.length - 1].date}</text>
    <text x="${W}" y="9" font-size="9" fill="currentColor" opacity=".6" text-anchor="end">max ${mx.toFixed(0)} mm/day</text>`;
}

/** hover: a vertical marker and the forecast and observed values for that window */
function bindChart(svg, g) {
  const box = svg.closest('.vchart'), read = box.querySelector('.vread'), cross = svg.querySelector('.vcross');
  svg.onpointermove = e => {
    const r = svg.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const i = Math.round(f * (g.series.length - 1)), p = g.series[i];
    cross.setAttribute('x1', (f * 300).toFixed(1)); cross.setAttribute('x2', (f * 300).toFixed(1)); cross.setAttribute('visibility', 'visible');
    read.textContent = `${p.date} · forecast ${p.pred_rain.toFixed(1)} mm · observed ${p.obs_rain.toFixed(1)} mm · ${p.pred_t.toFixed(1)} / ${p.obs_t.toFixed(1)} °C`;
  };
  svg.onpointerleave = () => { cross.setAttribute('visibility', 'hidden'); read.textContent = 'hover a day'; };
}

function stats(s) {
  const n = s.length, mean = a => a.reduce((x, y) => x + y, 0) / n;
  const P = s.map(p => p.pred_rain), O = s.map(p => p.obs_rain);
  const mp = mean(P), mo = mean(O);
  const cov = mean(P.map((p, i) => (p - mp) * (O[i] - mo)));
  const sd = a => Math.sqrt(mean(a.map(x => (x - mean(a)) ** 2)));
  const r = cov / Math.max(sd(P) * sd(O), 1e-9);
  const mae = mean(P.map((p, i) => Math.abs(p - O[i])));
  const bias = mean(P.map((p, i) => p - O[i]));
  return { n, mp, mo, r, mae, bias, maxP: Math.max(...P), maxO: Math.max(...O) };
}

function openDetail(g) {
  const year = YEARS[DATA.indexOf(g)];
  const st = stats(g.series);
  const W = 860, H = 300;
  const rows = g.series.map(p => `<tr><td>${p.date}</td><td>${p.pred_rain.toFixed(2)}</td><td>${p.obs_rain.toFixed(2)}</td><td>${(p.pred_rain - p.obs_rain).toFixed(2)}</td><td>${p.pred_t.toFixed(1)}</td><td>${p.obs_t.toFixed(1)}</td></tr>`).join('');
  const d = $('vDetail'); d.hidden = false;
  d.innerHTML = `<div class="vfull">
    <button class="x" id="vdClose" aria-label="Close detail">×</button>
    <h3>${year} · full detail <span class="muted small">${g.windows} windows, skill mean ${g.skill_mean.toFixed(3)}</span></h3>
    <div class="vbig"><svg viewBox="0 0 ${W} ${H + 16}" id="vBig">${paths(g.series, W, H)}<line class="vcross" x1="0" x2="0" y1="0" y2="${H}" stroke="#fff" stroke-opacity=".5" stroke-dasharray="3 3" visibility="hidden"/></svg><div id="vBigRead" class="muted small">hover the chart for a day</div></div>
    <div class="vstats">
      <div><span>Mean forecast</span><b>${st.mp.toFixed(2)} mm/day</b></div><div><span>Mean observed</span><b>${st.mo.toFixed(2)} mm/day</b></div>
      <div><span>Bias</span><b>${st.bias.toFixed(2)} mm/day</b></div><div><span>MAE</span><b>${st.mae.toFixed(2)} mm/day</b></div>
      <div><span>Correlation (daily)</span><b>${st.r.toFixed(2)}</b></div><div><span>Peak forecast / observed</span><b>${st.maxP.toFixed(0)} / ${st.maxO.toFixed(0)} mm</b></div>
    </div>
    <details><summary>All ${g.series.length} daily values</summary><div class="vtable"><table class="audit"><tr><th>Date</th><th>Forecast mm</th><th>Observed mm</th><th>Error mm</th><th>Forecast °C</th><th>Observed °C</th></tr>${rows}</table></div></details>
  </div>`;
  $('vdClose').onclick = () => (d.hidden = true);
  const svg = $('vBig'), read = $('vBigRead'), cross = svg.querySelector('.vcross');
  svg.onpointermove = e => {
    const r = svg.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const i = Math.round(f * (g.series.length - 1)), p = g.series[i];
    cross.setAttribute('x1', (f * W).toFixed(1)); cross.setAttribute('x2', (f * W).toFixed(1)); cross.setAttribute('visibility', 'visible');
    read.textContent = `${p.date} · forecast ${p.pred_rain.toFixed(2)} mm/day · observed ${p.obs_rain.toFixed(2)} mm/day · error ${(p.pred_rain - p.obs_rain).toFixed(2)} · temperature ${p.pred_t.toFixed(1)} / ${p.obs_t.toFixed(1)} °C`;
  };
}

/** replay: step through the recorded daily results of the chosen year; pause and scrub at any time */
function current() { const sel = $('vYear'); return DATA.filter(Boolean)[+sel.value]; }
function setReplay() {
  const g = current(); if (!g) return;
  $('vScrub').max = g.series.length - 1; $('vScrub').value = 0; showDay(0);
}
function showDay(i) {
  const g = current(), p = g.series[i]; if (!p) return;
  $('vScrub').value = i;
  const hit = p.pred_rain >= 10 ? (p.obs_rain >= 10 ? 'hit' : 'false alarm') : (p.obs_rain >= 10 ? 'miss' : 'quiet');
  $('vReadout').textContent = `Window ${i + 1}/${g.series.length} · start ${p.date} · forecast ${p.pred_rain.toFixed(1)} mm vs observed ${p.obs_rain.toFixed(1)} mm (land mean, day 1) · ${hit} at 10 mm/day`;
  document.querySelectorAll('#validation .vchart').forEach(b => (b.dataset.cur = ''));
}
function togglePlay() { if (timer) stopPlay(); else startPlay(); }
function startPlay() {
  const g = current(); if (!g) return;
  $('vPlay').textContent = '❚❚ Pause';
  timer = setInterval(() => { const i = +$('vScrub').value + 1; if (i >= g.series.length) { stopPlay(); return; } showDay(i); }, 180);
}
function stopPlay() { clearInterval(timer); timer = null; const b = $('vPlay'); if (b) b.textContent = '▶ Replay year'; }
