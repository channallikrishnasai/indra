// Grid, layers, colour maps and the field -> texture rasteriser shared by the globe and the 2D map.
export const NC = 6, NLAT = 32, NLON = 36, CELLS = NLAT * NLON;
export const LAT0 = 6.5, LON0 = 65.5;                 // cell centres
export const BBOX = { s: 6, n: 38, w: 65, e: 101 };   // cell edges
export const CH = { t2m: 0, tp: 1, u10: 2, v10: 3, msl: 4, tcwv: 5 };

const stops = {
  turbo: ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3b', '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0402'],
  rain: ['#00000000', '#9be7ff', '#41b6ff', '#2a6cff', '#5b3cff', '#b23cff', '#ff3dd2', '#ffffff'],
  wind: ['#0b1b3a', '#1f4e8c', '#1f8fb3', '#3ccfa0', '#b8f05a', '#ffe14d', '#ff8a3d', '#ff3d6b'],
  pres: ['#2c0a5e', '#3a3fbd', '#3f8fd9', '#9fd6e8', '#f2f2f2', '#f6c28b', '#ec7d4f', '#c8333e', '#6e0f2f'],
  moist: ['#5c3a12', '#a3702e', '#d8b46a', '#e8e2c6', '#8fd1c2', '#3aa6a1', '#1d6f8c', '#123f73'],
  div: ['#0a3d91', '#2f7fd6', '#86c3f0', '#ffffff', '#ffb38a', '#f2603a', '#a8102b'],
  risk: ['#00000000', '#5ec8ff80', '#3d7bffcc', '#7b3dff', '#d23dff', '#ff3d9a'],
  heat: ['#00000000', '#ffe06680', '#ffb02ecc', '#ff6a1a', '#e3261b', '#8f0a2a'],
  spread: ['#08121f', '#21406b', '#4d76b8', '#9c8fe0', '#f0a6d8', '#fff2f9'],
};

function hex(h) { const n = parseInt(h.slice(1, 7), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255, h.length > 7 ? parseInt(h.slice(7, 9), 16) : 255]; }
function lut(name) {
  const s = stops[name].map(hex), out = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (s.length - 1), k = Math.min(Math.floor(t), s.length - 2), f = t - k;
    for (let c = 0; c < 4; c++) out[i * 4 + c] = s[k][c] + (s[k + 1][c] - s[k][c]) * f;
  }
  return out;
}
export const LUTS = Object.fromEntries(Object.keys(stops).map(k => [k, lut(k)]));

// value(arr, idx) reads one cell of a [6*32*36] day array in display units
export const LAYERS = {
  t2m:  { name: 'Temperature', unit: '°C', cmap: 'turbo', lo: -5, hi: 42, dRange: 3, value: (a, i) => a[i] - 273.15 },
  tp:   { name: 'Rainfall', unit: 'mm/day', cmap: 'rain', lo: 0, hi: 60, dRange: 20, gamma: 0.55, alphaRamp: 1.5, value: (a, i) => a[CELLS + i] },
  wind: { name: 'Wind', unit: 'm/s', cmap: 'wind', lo: 0, hi: 12, dRange: 3, value: (a, i) => Math.hypot(a[2 * CELLS + i], a[3 * CELLS + i]) },
  msl:  { name: 'Pressure', unit: 'hPa', cmap: 'pres', lo: 996, hi: 1020, dRange: 3, value: (a, i) => a[4 * CELLS + i] / 100 },
  tcwv: { name: 'Moisture', unit: 'kg/m²', cmap: 'moist', lo: 5, hi: 65, dRange: 8, value: (a, i) => a[5 * CELLS + i] },
  // INDRA-Risk ML layers: probabilities (%) from the learned post-processor, served by POST /risk
  flood: { name: 'Flood risk', unit: '%', cmap: 'risk', lo: 0, hi: 50, dRange: 25, alphaRamp: 2, risk: 0, ml: true },
  heat:  { name: 'Heat risk', unit: '%', cmap: 'heat', lo: 0, hi: 80, dRange: 40, alphaRamp: 4, risk: 1, ml: true },
};

export const idx = (i, j) => i * NLON + j;

/** bilinear sample of a scalar grid (Float32Array CELLS) at fractional grid coords */
export function bilinear(g, gi, gj) {
  gi = Math.max(0, Math.min(NLAT - 1.001, gi)); gj = Math.max(0, Math.min(NLON - 1.001, gj));
  const i = gi | 0, j = gj | 0, fi = gi - i, fj = gj - j;
  const a = g[idx(i, j)], b = g[idx(i, j + 1)], c = g[idx(i + 1, j)], d = g[idx(i + 1, j + 1)];
  return (a * (1 - fj) + b * fj) * (1 - fi) + (c * (1 - fj) + d * fj) * fi;
}
export const inDomain = (lat, lon) => lat >= BBOX.s && lat <= BBOX.n && lon >= BBOX.w && lon <= BBOX.e;
export const gridCoords = (lat, lon) => [lat - LAT0, lon - LON0];

/** scalar grid of a layer for one day array */
export function layerGrid(day, layer) {
  const L = LAYERS[layer], g = new Float32Array(CELLS);
  for (let k = 0; k < CELLS; k++) g[k] = L.value(day, k);
  return g;
}

export const lerpDay = (a, b, f) => { const o = new Float32Array(a.length); for (let k = 0; k < a.length; k++) o[k] = a[k] + (b[k] - a[k]) * f; return o; };

// Rasterise into an RGBA canvas. Row 0 = north edge (38N), column 0 = west edge (65E).
export const TEX_W = 216, TEX_H = 192;
export function paintField(ctx, grid, { layer, mode }) {
  const L = LAYERS[layer], img = ctx.createImageData(TEX_W, TEX_H), px = img.data;
  const delta = mode === 'delta', spread = mode === 'spread';
  const table = LUTS[delta ? 'div' : spread ? 'spread' : L.cmap];
  const lo = delta ? -L.dRange : spread ? 0 : L.lo, hi = delta ? L.dRange : spread ? L.dRange / 2 : L.hi;
  for (let y = 0; y < TEX_H; y++) {
    const lat = BBOX.n - (y + 0.5) / TEX_H * 32;
    for (let x = 0; x < TEX_W; x++) {
      const lon = BBOX.w + (x + 0.5) / TEX_W * 36;
      const v = bilinear(grid, lat - LAT0, lon - LON0);
      let t = (v - lo) / (hi - lo); t = Math.max(0, Math.min(1, t));
      if (L.gamma && !delta && !spread) t = Math.pow(t, L.gamma);
      const c = (t * 255) | 0, o = (y * TEX_W + x) * 4;
      // soft fade at the domain edge so the field melts into the planet
      const edge = Math.min(lat - BBOX.s, BBOX.n - lat, lon - BBOX.w, BBOX.e - lon);
      let a = Math.pow(Math.min(1, edge / 2.4), 1.5) * 0.62;   // translucent: geography stays readable
      if (delta) a *= Math.min(1, Math.abs(v) / (L.dRange * 0.18));
      else if (L.alphaRamp && !spread) a *= table[c * 4 + 3] / 255 * Math.min(1, v / L.alphaRamp);
      px[o] = table[c * 4]; px[o + 1] = table[c * 4 + 1]; px[o + 2] = table[c * 4 + 2]; px[o + 3] = a * 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Clouds: thick, grey-bottomed where it rains, thin white haze where the column is moist. Texture comes from a
// tileable fractal noise with domain warping (billowy edges) that drifts slowly so the sky feels alive.
export const CLOUD_W = 432, CLOUD_H = 384;
const NS = 256, NOISE = (() => {
  const n = new Float32Array(NS * NS); let s = 11;
  const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let oct = 0, amp = 0.5, f = 4; oct < 6; oct++, amp *= 0.52, f *= 2) {
    const g = Float32Array.from({ length: f * f }, rnd);
    for (let y = 0; y < NS; y++) for (let x = 0; x < NS; x++) {
      const gx = x / NS * f, gy = y / NS * f, i = gx | 0, j = gy | 0, fx = gx - i, fy = gy - j;
      const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), i1 = (i + 1) % f, j1 = (j + 1) % f;
      n[y * NS + x] += ((g[j * f + i] * (1 - sx) + g[j * f + i1] * sx) * (1 - sy) + (g[j1 * f + i] * (1 - sx) + g[j1 * f + i1] * sx) * sy) * amp;
    }
  }
  return n;
})();
const noise = (x, y) => {
  x = ((x % NS) + NS) % NS; y = ((y % NS) + NS) % NS;
  const i = x | 0, j = y | 0, fx = x - i, fy = y - j, i1 = (i + 1) % NS, j1 = (j + 1) % NS;
  return (NOISE[j * NS + i] * (1 - fx) + NOISE[j * NS + i1] * fx) * (1 - fy) + (NOISE[j1 * NS + i] * (1 - fx) + NOISE[j1 * NS + i1] * fx) * fy;
};
export function paintClouds(ctx, day, t = 0) {
  const tp = day.subarray(CELLS, 2 * CELLS), tw = day.subarray(5 * CELLS, 6 * CELLS);
  const img = ctx.createImageData(CLOUD_W, CLOUD_H), px = img.data, drift = t * 6;
  for (let y = 0; y < CLOUD_H; y++) {
    const lat = BBOX.n - (y + 0.5) / CLOUD_H * 32;
    for (let x = 0; x < CLOUD_W; x++) {
      const lon = BBOX.w + (x + 0.5) / CLOUD_W * 36, gi = lat - LAT0, gj = lon - LON0;
      const r = bilinear(tp, gi, gj), w = bilinear(tw, gi, gj);
      const wx = noise(x * 0.9 + 37, y * 0.9) * 40, wy = noise(x * 0.9, y * 0.9 + 91) * 40;      // domain warp
      const nz = noise(x * 1.3 + wx + drift, y * 1.3 + wy);
      const cover = Math.min(1, Math.max(0, (r - 0.3) / 14)) + Math.min(0.45, Math.max(0, (w - 32) / 55));
      let a = Math.max(0, Math.min(1, (nz - (1.05 - cover)) * 3.2));                                // coverage threshold
      const edge = Math.min(lat - BBOX.s, BBOX.n - lat, lon - BBOX.w, BBOX.e - lon);
      a *= Math.pow(Math.min(1, edge / 2.4), 1.5);
      const thick = Math.min(1, r / 40), sh = 252 - thick * 70 - (1 - nz) * 25;                 // storm cores turn grey
      const o = (y * CLOUD_W + x) * 4; px[o] = sh; px[o + 1] = sh + 2; px[o + 2] = sh + 6; px[o + 3] = a * 245;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function paintLegend(canvas, layer, mode) {
  const L = LAYERS[layer], ctx = canvas.getContext('2d'), w = canvas.width;
  const table = LUTS[mode === 'delta' ? 'div' : mode === 'spread' ? 'spread' : L.cmap];
  for (let x = 0; x < w; x++) { const c = Math.round(x / (w - 1) * 255) * 4; ctx.fillStyle = `rgb(${table[c]},${table[c + 1]},${table[c + 2]})`; ctx.fillRect(x, 0, 1, canvas.height); }
  if (mode === 'delta') return [`−${L.dRange}`, `+${L.dRange}`, `Δ ${L.unit}`];
  if (mode === 'spread') return ['0', `${L.dRange / 2}`, `σ ${L.unit}`];
  return [`${L.lo}`, `${L.hi}`, L.unit];
}
