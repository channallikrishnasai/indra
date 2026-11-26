// Ensemble audit log: a 3D-animated explainer that opens on the Rain chance and Spread σ views.
// Every number comes from the live ensemble (S.chance, S.spread) and the validation files; nothing is pre-written.
import { CELLS, NLAT, NLON, LAT0, LAYERS, idx } from './field.js';
import { loadStates, stateMask } from './state.js';

const $ = id => document.getElementById(id);
const AREA = Array.from({ length: NLAT }, (_, i) => 111.2 * 111.2 * Math.cos((LAT0 + i) * Math.PI / 180));   // km² per cell
const fmtKm = a => (a >= 1e5 ? `${(a / 1e5).toFixed(1)} lakh km²` : `${Math.round(a / 1000)}k km²`);
const list = xs => (xs.length > 1 ? `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}` : xs[0] || '');

/** per-state summary of a per-day grid list: max over days (and the day it happens) or mean on one day */
async function byState(grids, land, { reduce = 'max', day = null } = {}) {
  const feats = await loadStates(), out = [];
  for (const f of feats) {
    const name = f.properties.name, m = stateMask(name);
    let best = 0, bestDay = 0, s = 0, n = 0;
    for (let d = 1; d < grids.length; d++) {
      if (day != null && d !== day) continue;
      const g = grids[d];
      for (let k = 0; k < CELLS; k++) {
        if (!m[k] || land[k] <= 0.5) continue;
        if (g[k] > best) { best = g[k]; bestDay = d; }
        s += g[k]; n++;
      }
    }
    if (n) out.push({ name, value: reduce === 'max' ? best : s / n, day: bestDay });
  }
  return out.sort((a, b) => b.value - a.value);
}

function chanceFindings(S) {
  const days = [];
  for (let d = 1; d < S.chance.length; d++) {
    const g = S.chance[d]; let likely = 0, possible = 0, aLikely = 0, mx = 0;
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const k = idx(i, j); if (S.land[k] <= 0.5) continue;
      mx = Math.max(mx, g[k]);
      if (g[k] >= 60) { likely++; aLikely += AREA[i]; } else if (g[k] >= 20) possible++;
    }
    days.push({ d, likely, possible, aLikely, mx });
  }
  const peak = days.reduce((a, b) => (b.likely > a.likely || (b.likely === a.likely && b.possible > a.possible) ? b : a), days[0]);
  return { days, peak };
}

function spreadFindings(S) {
  const mean = d => { let s = 0, n = 0; for (let k = 0; k < CELLS; k++) if (S.land[k] > 0.5) { s += S.spread[d][k]; n++; } return n ? s / n : 0; };
  const last = S.spread.length - 1;
  return { d1: mean(1), d5: mean(Math.min(5, last)), d10: mean(last), last };
}

const row = (i, tag, title, body, tone = '') => `<li class="lg ${tone}" style="--i:${i}"><span class="tag">${tag}</span><div><b>${title}</b><p>${body}</p></div></li>`;

export async function openEnsAudit(kind, S) {
  const el = $('ensAudit'), hasEns = !!(S.chance && S.spread);
  const date = S.win?.date ?? '';
  let i = 0, rows = [];
  rows.push(row(i++, 'START', `Start: 3 real days ending ${date}`, 'The twin begins from observed weather (the kit data), the same 3 days the main forecast uses.'));
  if (!hasEns) {
    rows.push(row(i++, 'WAIT', 'No ensemble yet', 'Press <b>Rain chance</b> or <b>Run 5-member ensemble</b> and this log fills with the real numbers from that run.', 'warn'));
  } else {
    const sig = S.spreadSummary, lay = LAYERS[sig.layer];
    rows.push(row(i++, 'NUDGE', '5 copies, each nudged a little',
      'Each copy gets small random changes on the last input day: about 0.4 K of temperature, 0.6 mm/day of rain (only where it already rains), 0.35 m/s of wind, 40 Pa of pressure and 1.2 kg/m² of moisture, smooth over a few hundred km. These are the size of real measurement uncertainty.'));
    rows.push(row(i++, 'RUN', 'Each copy rolls 10 days', 'INDRA runs each copy forward exactly like the main forecast (5 live /simulate calls). Nothing is pre-computed.'));
    const C = chanceFindings(S), P = C.peak;
    const top = (await byState(S.chance, S.land)).filter(s => s.value >= 20).slice(0, 6);
    rows.push(row(i++, 'COUNT', 'Heavy-rain chance per cell',
      `For each cell and day we count how many of the 5 copies reach 64.5 mm/day (IMD "heavy rain"). 1 of 5 = 20%, 3 of 5 = 60%, 5 of 5 = 100%. ` +
      `Peak: day +${P.d}, ${P.likely} land cells at 60% or more (${fmtKm(P.aLikely)}), ${P.possible} more at 20–40%. Highest anywhere: ${Math.round(Math.max(...C.days.map(x => x.mx)))}%.`, kind === 'chance' ? 'hot' : ''));
    rows.push(row(i++, 'SPREAD', `Spread σ: how far the copies disagree (${lay.name.toLowerCase()})`,
      (() => { const F = spreadFindings(S), r = F.d10 / Math.max(F.d1, 1e-6);
        const trend = r >= 1.2 ? `the small nudges grow ${r.toFixed(1)}× over 10 days, so the later days are less certain`
          : r <= 0.85 ? 'the differences shrink over 10 days: the model pulls the copies back together'
          : 'the differences stay about the same size over 10 days';
        return `σ is the standard deviation across the 5 copies. Over land it averages ${F.d1.toFixed(2)} ${lay.unit} on day 1 and ${F.d10.toFixed(2)} ${lay.unit} on day ${F.last}: ${trend}. Agreement score ${sig.conf.toFixed(0)}% (agreement, not accuracy).`; })(), kind === 'spread' ? 'hot' : ''));
    // what we might face
    const likely = top.filter(s => s.value >= 60), possible = top.filter(s => s.value < 60);
    let face;
    if (likely.length) face = `Heavy rain is <b>likely</b> (3 or more of 5 copies) in ${list(likely.map(s => `${s.name} (day +${s.day})`))}. On the ground that means waterlogging in towns, flash floods in low-lying areas and small rivers, and landslide risk on hill slopes. Keep drains clear and follow district advisories.`;
    else if (possible.length) face = `Heavy rain is <b>possible</b> but not agreed (1–2 of 5 copies) in ${list(possible.map(s => `${s.name} (day +${s.day})`))}. Watch these areas; the signal can grow or vanish in the next run.`;
    else face = 'No land cell reaches heavy rain in any copy. Heavy rain is unlikely in this 10-day window from this start.';
    const unsure = (await byState(S.spread, S.land, { reduce: 'mean', day: S.spread.length - 1 })).slice(0, 3).map(s => s.name);
    face += ` The forecast is least certain over ${list(unsure)} by day 10: plan there with a margin.`;
    rows.push(row(i++, 'FACE', 'What changes we might face', face, likely.length ? 'alert' : ''));
  }
  rows.push(row(i++, 'NOTE', 'Not an official forecast', 'This is a spread among perturbed runs of one model, not a calibrated probability and not an IMD warning.', 'mute'));

  const val = await Promise.all([2019, 2020, 2021].map(y => fetch(`/validation/${y}.json`).then(r => (r.ok ? r.json() : null)).catch(() => null)));
  const vline = val.map((v, k) => (v ? `<li><b>${2019 + k}</b> skill ${v.skill_mean.toFixed(3)} · heavy-rain CSI day 8 ${v.leads.d8.csi.toFixed(2)}</li>` : '')).join('');

  el.innerHTML = `
    <div class="ea-stage" id="eaStage">
      <button class="x" id="eaClose" aria-label="Close ensemble audit">×</button>
      <header class="ea-head">
        <div class="orbit" aria-hidden="true"><i class="core"></i>${[0, 1, 2, 3, 4].map(n => `<i class="m" style="--n:${n}"></i>`).join('')}</div>
        <div><span class="kick">Ensemble audit log · ${kind === 'chance' ? 'Rain chance' : 'Spread σ'}</span><h2>What just happened, step by step</h2></div>
      </header>
      <ol class="ea-log">${rows.join('')}</ol>
      <div class="ea-cards">
        <section class="ea-card" style="--i:${i++}"><h3>What is an ensemble?</h3>
          <p>Weather is chaotic: tiny differences in today's state grow into big differences a week later. One forecast cannot show that. An <b>ensemble</b> runs the same model several times from slightly different starts. Where the copies agree, the forecast is firmer; where they split, it is uncertain. Operational weather centres run dozens of copies (ECMWF runs 51); INDRA runs 5 to stay fast in the browser.</p></section>
        <section class="ea-card" style="--i:${i++}"><h3>Why validation on 3 years, not 13?</h3>
          <p>An honest test year needs a model that has <b>never seen it</b>: we retrain INDRA on only the years before it, then score that year. Each retrain takes about an hour of CPU, so 13 years means 13 retrains (15+ hours). Early years would also train on just 3–5 years of data, testing a weaker model than ours. We ran the three most recent: 2019 (trained to 2018), 2020 (to 2019) and 2021 (the shipped model, which saw 2021 when settings were chosen, so it is optimistic).</p>
          <ul class="ea-val">${vline}</ul>
          <p class="mute">More years can be added with the same tool (rolling_export.py) given the CPU time.</p></section>
      </div>
    </div>`;
  el.hidden = false; el.classList.remove('in'); void el.offsetWidth; el.classList.add('in');
  $('eaClose').onclick = () => closeEnsAudit();
  const stage = $('eaStage');                              // gentle 3D tilt toward the pointer
  el.onpointermove = e => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
    stage.style.transform = `rotateY(${x * 6}deg) rotateX(${-y * 5}deg)`;
  };
  el.onpointerleave = () => (stage.style.transform = '');
}

export function closeEnsAudit() { const el = $('ensAudit'); if (el) el.hidden = true; }
