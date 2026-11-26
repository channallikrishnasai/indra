// India states: which state a point is in, each state's grid mask, and the stats shown in the State panel.
// Boundaries: geoBoundaries ADM1 (CC BY 2.5 IN), simplified, bundled as public/india-states.geojson.
import { geoContains } from 'd3-geo';
import { CELLS, NLAT, NLON, LAT0, LON0, idx } from './field.js';

let feats = null, masks = null;
export async function loadStates() {
  if (!feats) feats = (await (await fetch('/india-states.geojson')).json()).features;
  return feats;
}
const AREA = Array.from({ length: NLAT }, (_, i) => Math.cos((LAT0 + i) * Math.PI / 180));   // weights for land means
export const stateAt = (lat, lon) => feats?.find(f => geoContains(f, [lon, lat]))?.properties.name ?? null;
export const stateFeature = name => feats?.find(f => f.properties.name === name);

/** per-state grid mask (cells whose centre lies inside the state), computed once */
export function stateMask(name) {
  masks ??= new Map();
  if (!masks.has(name)) {
    const f = stateFeature(name), m = new Uint8Array(CELLS);
    if (f) for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) m[idx(i, j)] = geoContains(f, [LON0 + j, LAT0 + i]) ? 1 : 0;
    masks.set(name, m);
  }
  return masks.get(name);
}
export const bboxOf = name => {
  const f = stateFeature(name); if (!f) return null;
  const pts = f.geometry.type === 'Polygon' ? f.geometry.coordinates.flat() : f.geometry.coordinates.flat(2);
  const lons = pts.map(p => p[0]), lats = pts.map(p => p[1]);
  return [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]];   // [[w,s],[e,n]]
};

/** stats for one state from the current forecast (days[0] = initial state, days[1..] = forecast) */
export function stateStats(name, days, clim, risk, alerts) {
  const m = stateMask(name); let n = 0, t = 0, r = 0, w = 0, heat = 0, flood = 0, cells = 0;
  for (let d = 1; d < days.length; d++) {
    const a = days[d]; let hot = 0, wet = 0, c = 0;
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const k = idx(i, j); if (!m[k]) continue;
      const aw = AREA[i]; t += (a[k] - 273.15) * aw; r += a[CELLS + k] * aw; w += Math.hypot(a[2 * CELLS + k], a[3 * CELLS + k]) * aw; n += aw;
      if (a[k] - 273.15 >= 35 || (a[k] - clim[d - 1][k] >= 4.5 && a[k] - 273.15 >= 30)) hot++;
      if (a[CELLS + k] >= 64.5) wet++;
      c++;
    }
    heat = Math.max(heat, c ? hot / c * 100 : 0); flood = Math.max(flood, c ? wet / c * 100 : 0); cells = c;
  }
  const mine = (alerts || []).filter(x => stateAt(x.lat, x.lon) === name);
  const ml = (k) => risk ? Math.max(...risk.slice(1).map(rd => { let mx = 0; for (let q = 0; q < CELLS; q++) if (m[q]) mx = Math.max(mx, rd[k * CELLS + q]); return mx; })) * 100 : null;
  return {
    cells, meanT: n ? t / n : NaN, meanR: n ? r / n : NaN, meanW: n ? w / n : NaN,
    heatPct: heat, floodPct: flood, alerts: mine.length, high: mine.filter(x => x.severity === 'High' || x.severity === 'Critical').length,
    floodRisk: ml(0), heatRisk: ml(1),
  };
}
