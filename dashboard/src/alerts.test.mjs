// node dashboard/src/alerts.test.mjs
import assert from 'node:assert/strict';
import { CELLS, NLON, LAT0, LON0, idx } from './field.js';
import { generateAlerts, diffAlerts, regionOf } from './alerts.js';

const cell = (lat, lon) => idx(Math.round(lat - LAT0), Math.round(lon - LON0));
const day = () => { const a = new Float32Array(6 * CELLS); a.fill(273.15 + 25, 0, CELLS); return a; };   // 25 C, no rain
const land = new Float32Array(CELLS).fill(1), clim = [day(), day()].map(d => d.subarray(0, CELLS));
const days = [day(), day(), day()];
const odisha = cell(20.5, 84.5), rajasthan = cell(26.5, 73.5);
days[1][CELLS + odisha] = 130;            // very heavy rain on day 1
days[2][rajasthan] = 273.15 + 41;         // extreme heat on day 2

const alerts = generateAlerts(days, clim, land, null);
assert.equal(alerts.length, 2);
assert.deepEqual(alerts.map(a => [a.type, a.day, a.region, a.severity]), [['heat', 2, 'Rajasthan', 'Critical'], ['rain', 1, 'Odisha', 'High']]);   // most severe first
assert.equal(regionOf(13.0, 80.2), 'Tamil Nadu');

// ocean cells never alert
const sea = new Float32Array(CELLS); assert.equal(generateAlerts(days, clim, sea, null).length, 0);

// a shock that adds a new heavy-rain region shows up as "new"
const multi = days.map(d => d.slice()); multi[1][CELLS + cell(22.5, 88.0)] = 80;
const diff = diffAlerts(alerts, generateAlerts(multi, clim, land, null));
assert.equal(diff.filter(a => a.status === 'new').length, 1);
assert.equal(diff.find(a => a.status === 'new').severity, 'Moderate');
// cyclone flag: a 12 hPa low with 18 m/s winds off Odisha -> High; humidex sanity
{
  const d0 = day(), st = day(); for (const x of [d0, st]) x.fill(101000, 4 * CELLS, 5 * CELLS);
  const ci = Math.round(19.5 - LAT0), cj = Math.round(86.5 - LON0);
  for (let di = -3; di <= 3; di++) for (let dj = -3; dj <= 3; dj++) {
    const r = Math.max(Math.abs(di), Math.abs(dj)), k = idx(ci + di, cj + dj);
    st[4 * CELLS + k] = 101000 - (r === 0 ? 1200 : r === 1 ? 800 : r === 2 ? 400 : 0); if (r === 1) st[2 * CELLS + k] = 18;
  }
  const cy = generateAlerts([d0, st], [d0.subarray(0, CELLS)], sea, null);
  assert.equal(cy.length, 1); assert.equal(cy[0].type, 'cyclone'); assert.equal(cy[0].severity, 'High');
  assert.equal(generateAlerts([d0, d0], [d0.subarray(0, CELLS)], sea, null).length, 0);
}
const { humidex, dewFromTcwv } = await import('./impact.js');
assert.ok(Math.abs(humidex(30, 25) - 42.3) < 0.5);            // published humidex table: 30 C, dew point 25 C -> about 42
assert.ok(dewFromTcwv(50) > 20 && dewFromTcwv(50) < 27);       // humid monsoon column -> dew point in the low-to-mid 20s C
console.log('alerts self-check passed');
