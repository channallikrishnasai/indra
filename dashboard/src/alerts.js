// Alerts & warnings generated from a live INDRA rollout (+ optional INDRA-Risk probabilities).
// Pure functions, no DOM: generateAlerts(days, clim, land, risk) -> [{type, day, region, severity, value, cells, lat, lon}]
import { CELLS, NLAT, NLON, LAT0, LON0, idx } from './field.js';

export const SEVERITIES = ['Critical', 'High', 'Moderate', 'Low'];

// shortcut: nearest-centre labelling (no offline state-boundary file); cells near a border may get the neighbour's name
const REGIONS = [
  ['Jammu & Kashmir', 33.8, 75.3], ['Ladakh', 34.2, 77.6], ['Himachal Pradesh', 31.9, 77.2], ['Punjab', 30.9, 75.4], ['Uttarakhand', 30.1, 79.2],
  ['Haryana', 29.1, 76.3], ['Delhi', 28.6, 77.2], ['Rajasthan', 26.6, 73.8], ['Uttar Pradesh', 26.9, 80.9], ['Bihar', 25.6, 85.6],
  ['Sikkim', 27.5, 88.5], ['Arunachal Pradesh', 28.1, 94.4], ['Assam', 26.2, 92.8], ['Meghalaya', 25.5, 91.3], ['Nagaland', 26.1, 94.5],
  ['Manipur', 24.8, 93.9], ['Mizoram', 23.3, 92.8], ['Tripura', 23.8, 91.6], ['West Bengal', 23.4, 87.9], ['Jharkhand', 23.6, 85.5],
  ['Odisha', 20.5, 84.4], ['Chhattisgarh', 21.3, 81.9], ['Madhya Pradesh', 23.5, 78.0], ['Gujarat', 22.5, 71.5], ['Maharashtra', 19.4, 75.7],
  ['Goa', 15.4, 74.0], ['Telangana', 17.9, 79.1], ['Andhra Pradesh', 15.7, 79.7], ['Karnataka', 14.6, 76.1], ['Tamil Nadu', 11.0, 78.4],
  ['Kerala', 10.3, 76.4], ['Pakistan', 29.5, 69.5], ['Nepal', 28.2, 84.2], ['Bhutan', 27.4, 90.4], ['Bangladesh', 23.8, 90.3],
  ['Myanmar', 21.5, 95.9], ['Sri Lanka', 7.8, 80.7], ['Tibet (China)', 31.5, 88.0],
];
export function regionOf(lat, lon) {
  let best = REGIONS[0], d = Infinity;
  for (const r of REGIONS) { const x = Math.hypot(lat - r[1], (lon - r[2]) * Math.cos(lat * Math.PI / 180)); if (x < d) { d = x; best = r; } }
  return best[0];
}

export const ADVICE = {
  rain: {
    do: ['Move to higher ground if you live in a low-lying or flood-prone area', 'Keep a torch, medicines, documents and drinking water ready',
      'Follow district and NDMA/SDMA advisories', 'Clear drains around your home'],
    dont: ['Do not walk or drive through flooded roads or underpasses', 'Do not touch fallen power lines or wet electrical fittings',
      'Do not shelter near rivers, slopes or weak walls'],
  },
  heat: {
    do: ['Drink water often, even if not thirsty; use ORS or lassi', 'Stay indoors 12–4 pm; wear light, loose, cotton clothes',
      'Check on elderly people, children and outdoor workers', 'Keep livestock in shade with water'],
    dont: ['Do not do heavy outdoor work in the afternoon', 'Do not leave children or pets in parked vehicles',
      'Avoid alcohol, tea, coffee and very sugary drinks'],
  },
};

const rainSeverity = r => r >= 204.5 ? 'Critical' : r >= 115.6 ? 'High' : r >= 64.5 ? 'Moderate' : null;   // IMD extremely heavy / very heavy / heavy
// heat uses the dashboard's heat-stress rule (daily mean >= 35 C, or >= 4.5 K above normal while >= 30 C), graded by strength
function heatSeverity(tC, anom) {
  if (!(tC >= 35 || (anom >= 4.5 && tC >= 30))) return null;
  return tC >= 40 || anom >= 7 ? 'Critical' : tC >= 37.5 || anom >= 6 ? 'High' : 'Moderate';
}

/** days: [day0..dayN] Float32Array(6*CELLS) physical; clim: [day1..dayN]; land: Float32Array(CELLS); risk: [day0..dayN] Float32Array(2*CELLS) or null */
export function generateAlerts(days, clim, land, risk) {
  const groups = new Map();
  const add = (type, d, k, sev, value) => {
    const i = Math.floor(k / NLON), j = k % NLON, lat = LAT0 + i, lon = LON0 + j, region = regionOf(lat, lon), key = `${type}|${d}|${region}`;
    const g = groups.get(key) ?? { type, day: d, region, severity: 'Low', value: -Infinity, cells: 0, lat, lon };
    g.cells++;
    if (SEVERITIES.indexOf(sev) < SEVERITIES.indexOf(g.severity)) g.severity = sev;
    if (value > g.value) { g.value = value; g.lat = lat; g.lon = lon; }
    groups.set(key, g);
  };
  for (let d = 1; d < days.length; d++) {
    const a = days[d], c = clim[d - 1], r = risk?.[d];
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const k = idx(i, j); if (land[k] <= 0.5) continue;
      const rain = a[CELLS + k], tC = a[k] - 273.15, anom = a[k] - c[k];
      const rs = rainSeverity(rain);
      if (rs) add('rain', d, k, rs, rain);
      else if (r && r[k] >= 0.3) add('rain', d, k, 'Low', rain);                    // INDRA-Risk watch: >= 30% chance of heavy rain
      const hs = heatSeverity(tC, anom);
      if (hs) add('heat', d, k, hs, tC);
      else if (r && r[CELLS + k] >= 0.5) add('heat', d, k, 'Low', tC);              // INDRA-Risk watch: >= 50% chance of heat stress
    }
  }
  return [...groups.values()].sort((x, y) => SEVERITIES.indexOf(x.severity) - SEVERITIES.indexOf(y.severity) || x.day - y.day || y.cells - x.cells);
}

/** mark alerts that exist in `multi` but not in `prime` (same type, day, region) or got more severe */
export function diffAlerts(prime, multi) {
  const p = new Map(prime.map(a => [`${a.type}|${a.day}|${a.region}`, a]));
  return multi.map(a => {
    const b = p.get(`${a.type}|${a.day}|${a.region}`);
    return { ...a, status: !b ? 'new' : SEVERITIES.indexOf(a.severity) < SEVERITIES.indexOf(b.severity) ? 'worse' : 'same' };
  });
}
