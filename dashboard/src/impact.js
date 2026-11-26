// Impact extension: turns a simulated rollout into decisions people care about. Everything here runs on the
// live /simulate output (days[1..n]) plus the model's own climatology, never on pre-rendered results.
import { CELLS, NLAT, NLON, LAT0, idx } from './field.js';

/** Humidex (Canadian MSC) from air temperature (C) and dew point (C). */
export function humidex(tC, tdC) {
  const e = 6.11 * Math.exp(5417.753 * (1 / 273.16 - 1 / (tdC + 273.15)));
  return tC + 0.5555 * (e - 10);
}
/** Dew point (C) estimated from total column water vapour (kg/m2) with Reitan (1963): ln w[cm] = -0.981 + 0.0341 Td[F].
 * shortcut: column-to-surface relation is approximate (no surface humidity in the data); fine for a screening index. */
export function dewFromTcwv(tcwv) {
  const tdF = (Math.log(Math.max(tcwv, 0.5) / 10) + 0.981) / 0.0341;
  return (tdF - 32) / 1.8;
}

const AREA = Array.from({ length: NLAT }, (_, i) => 111.2 * 111.2 * Math.cos((LAT0 + i) * Math.PI / 180)); // km² per cell

/** land: Float32Array CELLS (1 = land). clim: array of day arrays aligned with days[1..n]. */
export function impacts(days, clim, land) {
  const n = days.length - 1; let landArea = 0;
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) if (land[idx(i, j)] > 0.5) landArea += AREA[i];
  let heatPeak = 0, heatDays = 0, floodArea = 0, peakRain = 0, wind = 0, rainSum = 0, climSum = 0, stress = 0, humidPeak = 0;
  const flooded = new Uint8Array(CELLS), rain10 = new Float32Array(CELLS), crain10 = new Float32Array(CELLS), tmean = new Float32Array(CELLS);
  for (let d = 1; d <= n; d++) {
    const a = days[d], c = clim[d - 1]; let hot = 0, muggy = 0;
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const k = idx(i, j); if (land[k] <= 0.5) continue;
      const t = a[k] - 273.15, anom = a[k] - c[k], r = a[CELLS + k];
      // daily-MEAN heat stress proxy: >= 35 C mean, or >= 4.5 K above normal while already >= 30 C (IMD-style departure)
      if (t >= 35 || (anom >= 4.5 && t >= 30)) hot += AREA[i];
      if (humidex(t, Math.min(t, dewFromTcwv(a[5 * CELLS + k]))) >= 40) muggy += AREA[i];   // humidex 40: great discomfort
      if (r >= 64.5) flooded[k] = 1;                     // IMD "heavy rain" threshold
      peakRain = Math.max(peakRain, r); rain10[k] += r; crain10[k] += c[CELLS + k]; tmean[k] += t / n;
      const sp100 = Math.hypot(a[2 * CELLS + k], a[3 * CELLS + k]) * Math.pow(10, 1 / 7);   // 10 m -> 100 m hub (1/7 power law)
      wind += 0.5 * 1.2 * sp100 ** 3 * AREA[i];
    }
    heatPeak = Math.max(heatPeak, hot / landArea); if (hot / landArea > 0.05) heatDays++; humidPeak = Math.max(humidPeak, muggy / landArea);
  }
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const k = idx(i, j); if (land[k] <= 0.5) continue;
    if (flooded[k]) floodArea += AREA[i];
    rainSum += rain10[k] * AREA[i]; climSum += crain10[k] * AREA[i];
    // crop water stress: rain deficit vs normal, amplified by heat (evaporative demand), only where crops expect rain
    if (crain10[k] > 5) stress += Math.max(0, Math.min(1, 1 - rain10[k] / crain10[k])) * Math.min(1.5, Math.max(0.6, (tmean[k] - 20) / 10)) * AREA[i];
  }
  return {
    heatPeak: heatPeak * 100, heatDays, floodArea, peakRain,
    windPower: wind / (landArea * n),                             // W/m² mean power density at 100 m
    rainDeparture: climSum > 0 ? (rainSum / climSum - 1) * 100 : 0,
    rainMean: rainSum / landArea,
    cropStress: Math.min(100, stress / landArea * 100),
    humidHeat: humidPeak * 100,                                    // % land, peak day, humidex >= 40
    rainTotal: rainSum / landArea, rainNormal: climSum / landArea,  // 10-day land-mean rain vs normal (mm): monsoon progress
  };
}

/** all-India land mean of a scalar layer per day */
export function landMeanSeries(days, land, valueFn) {
  return days.map(a => { let s = 0, w = 0; for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) { const k = idx(i, j); if (land[k] > 0.5) { s += valueFn(a, k) * AREA[i]; w += AREA[i]; } } return s / w; });
}

const STD = [13.46, 7.55, 2.90, 2.43, 619.0, 20.49];
/** Response of the multiverse vs prime per day: total energy (std units) and energy-weighted distance from the shock (km) */
export function propagation(prime, multi, shock) {
  const out = [];
  for (let d = 1; d < prime.length; d++) {
    let e = 0, ed = 0, near = 0;
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
      const k = idx(i, j); let ek = 0;
      for (let c = 0; c < 6; c++) { const z = (multi[d][c * CELLS + k] - prime[d][c * CELLS + k]) / STD[c]; ek += z * z; }
      const lat = LAT0 + i, lon = 65.5 + j;
      const dist = Math.hypot(lat - shock.lat, (lon - shock.lon) * Math.cos(lat * Math.PI / 180)) * 111.2;
      e += ek; ed += ek * dist; if (dist <= (shock.radius + 3) * 111.2) near += ek;
    }
    out.push({ energy: e, radius: e > 0 ? ed / e : 0, nearFrac: e > 0 ? near / e : 0 });
  }
  return out;
}
