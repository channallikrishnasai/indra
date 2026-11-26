// Rolling-origin validation panel: skill and heavy-rain CSI for 2019, 2020 and 2021, each from a model trained on
// the years before it, with the forecast-vs-observed land-mean rain for every year. The numbers are exported offline
// by rolling_export.py (public/validation/<year>.json); the dashboard only reads and draws them.
const YEARS = [2019, 2020, 2021];
const $ = id => document.getElementById(id);

export async function openValidation() {
  const el = $('validation'); el.hidden = false;
  el.innerHTML = `<button class="x" id="valClose" aria-label="Close validation">×</button><h2>Rolling-origin validation <span class="muted small">the recipe retrained on earlier years only; validation, not hidden-test results</span></h2><p class="muted small">Loading…</p>`;
  $('valClose').onclick = () => (el.hidden = true);
  const got = await Promise.all(YEARS.map(y => fetch(`/validation/${y}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null)));
  if (got.every(g => !g)) {
    el.innerHTML = `<button class="x" id="valClose" aria-label="Close validation">×</button><h2>Rolling-origin validation</h2>
      <p class="offline">The validation files are missing. Generate them with <code>python rolling_export.py --year 2019 --weights weights/ro2019.pt --out dashboard/public/validation/2019.json</code> (and 2020, 2021).</p>`;
    $('valClose').onclick = () => (el.hidden = true); return;
  }
  const row = g => g ? `<td>${g.skill_mean.toFixed(3)}</td><td>${g.leads.d1.csi.toFixed(2)}</td><td>${g.leads.d4.csi.toFixed(2)}</td><td>${g.leads.d8.csi.toFixed(2)}</td>` : '<td colspan="4">—</td>';
  const charts = got.map((g, i) => g ? chart(g, YEARS[i]) : `<div class="vchart"><b>${YEARS[i]}</b><p class="muted small">not exported</p></div>`).join('');
  el.innerHTML = `<button class="x" id="valClose" aria-label="Close validation">×</button>
    <h2>Rolling-origin validation <span class="muted small">2021 was used to choose some settings; 2019 and 2020 were not</span></h2>
    <table class="audit"><tr><th>Year</th><th>Skill (mean)</th><th>Heavy-rain CSI d1</th><th>d4</th><th>d8</th></tr>
      ${YEARS.map((y, i) => `<tr><td>${y}</td>${row(got[i])}</tr>`).join('')}</table>
    <p class="muted small">Skill = RMSE skill vs the better of persistence or climatology (1 is perfect, 0 is no better). CSI = heavy-rain (≥ 10 mm/day) hit score (1 is perfect). Same metric as the judge's scorer.</p>
    <div class="vgrid">${charts}</div>
    <p class="muted small">Land-mean rain, day 1 of each window: forecast (orange) vs observed (blue). Source: kit data 2005–2021 · INDRA-v1 model. Not an official IMD forecast.</p>`;
  $('valClose').onclick = () => (el.hidden = true);
}

function chart(g, year) {
  const W = 300, H = 110, pts = g.series;
  if (!pts.length) return `<div class="vchart"><b>${year}</b><p class="muted small">no windows</p></div>`;
  const mx = Math.max(10, ...pts.map(p => Math.max(p.pred_rain, p.obs_rain))) * 1.05;
  const X = i => (i / Math.max(1, pts.length - 1)) * W, Y = v => H - (v / mx) * H;
  const line = k => pts.map((p, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(p[k]).toFixed(1)}`).join(' ');
  return `<div class="vchart"><b>${year}</b> <span class="muted small">${g.windows} windows · skill ${g.skill_mean.toFixed(2)}</span>
    <svg viewBox="0 0 ${W} ${H + 14}" role="img" aria-label="Forecast vs observed land-mean rain ${year}">
      <path d="${line('obs_rain')}" fill="none" stroke="#46d3ff" stroke-width="1.4" />
      <path d="${line('pred_rain')}" fill="none" stroke="#ff7a45" stroke-width="1.4" />
      <text x="0" y="${H + 12}" font-size="9" fill="currentColor" opacity=".6">${pts[0].date}</text>
      <text x="${W}" y="${H + 12}" font-size="9" fill="currentColor" opacity=".6" text-anchor="end">${pts[pts.length - 1].date}</text>
      <text x="${W}" y="9" font-size="9" fill="currentColor" opacity=".6" text-anchor="end">max ${mx.toFixed(0)} mm/day</text>
    </svg></div>`;
}
