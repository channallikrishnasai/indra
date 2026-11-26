// State audit: which states a shock changes and which it leaves alone, from the same per-state statistics the
// State panel uses (stateStats). Also used for a uniform +1 K "what if" run over all of India.
import { loadStates, stateStats } from './state.js';

// a state counts as affected when any of these moves past its threshold (vs the no-shock run)
const T = { meanT: 0.3, meanR: 0.5, heatPct: 5, floodPct: 5 };

/** rows for every state: values for the baseline and the scenario, and the changes */
export async function auditRows(base, scen, clim) {
  const feats = await loadStates(), rows = [];
  for (const f of feats) {
    const name = f.properties.name, a = stateStats(name, base, clim, null, null), b = stateStats(name, scen, clim, null, null);
    if (!a.cells) continue;                                   // no grid cells inside (tiny or outside the grid)
    const d = { name, cells: a.cells, meanT: b.meanT - a.meanT, meanR: b.meanR - a.meanR, heatPct: b.heatPct - a.heatPct, floodPct: b.floodPct - a.floodPct };
    d.affected = Math.abs(d.meanT) >= T.meanT || Math.abs(d.meanR) >= T.meanR || Math.abs(d.heatPct) >= T.heatPct || Math.abs(d.floodPct) >= T.floodPct;
    d.score = Math.max(Math.abs(d.meanT) / T.meanT, Math.abs(d.meanR) / T.meanR, Math.abs(d.heatPct) / T.heatPct, Math.abs(d.floodPct) / T.floodPct);
    d.baseT = a.meanT; d.baseR = a.meanR;
    rows.push(d);
  }
  return rows.sort((x, y) => y.score - x.score);
}

/** HTML table: affected first, then the unaffected list. */
export function auditHtml(rows, title, note) {
  const f = (v, dig, u) => `${v > 0 ? '+' : ''}${v.toFixed(dig)}${u}`;
  const hit = rows.filter(r => r.affected), quiet = rows.filter(r => !r.affected);
  const line = r => `<tr class="${r.affected ? 'hit' : ''}"><td>${r.name}</td><td>${r.baseT.toFixed(1)} °C</td><td class="${r.meanT > 0 ? 'up' : r.meanT < 0 ? 'down' : ''}">${f(r.meanT, 2, ' K')}</td>` +
    `<td>${r.baseR.toFixed(1)} mm/d</td><td class="${r.meanR > 0 ? 'up' : r.meanR < 0 ? 'down' : ''}">${f(r.meanR, 2, ' mm/d')}</td><td>${f(r.heatPct, 1, ' pts')}</td><td>${f(r.floodPct, 1, ' pts')}</td></tr>`;
  const head = `<tr><th>State</th><th>Baseline T</th><th>Δ T</th><th>Baseline rain</th><th>Δ rain</th><th>Δ heat cells</th><th>Δ heavy-rain cells</th></tr>`;
  return `<h2>${title} <span class="muted small">${hit.length} affected · ${quiet.length} unaffected</span></h2>
    <p class="muted small">${note}</p>
    <h3>Affected states (${hit.length})</h3>
    ${hit.length ? `<table class="audit">${head}${hit.map(line).join('')}</table>` : '<p class="muted small">No state moves past its threshold.</p>'}
    <h3>Unaffected states (${quiet.length})</h3>
    ${quiet.length ? `<p class="muted small">${quiet.map(r => r.name).join(' · ')}</p>` : '<p class="muted small">Every state is affected.</p>'}
    <p class="muted small">Affected means any change of at least 0.3 K mean temperature, 0.5 mm/day mean rain, 5 points of heat-stress cells or 5 points of heavy-rain cells. Source: kit data 2005–2021 · INDRA-v1 model. Not an official IMD forecast.</p>`;
}
