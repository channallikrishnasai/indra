// Scenario report: a designed PDF of the current run (and, after a tour, of the whole tour), built in the browser.
// jsPDF is loaded only when this module is. Helvetica only: text is cleaned to plain ASCII first.
import { jsPDF } from 'jspdf';

const ascii = t => String(t).replace(/Δ/g, 'd').replace(/≥/g, '>=').replace(/≤/g, '<=').replace(/→/g, '->').replace(/[−–—]/g, '-')
  .replace(/σ/g, 'sd').replace(/²/g, '2').replace(/×/g, 'x').replace(/°/g, ' deg').replace(/…/g, '...').replace(/[’‘]/g, "'").replace(/[“”]/g, '"');
const C = {
  ink: [14, 26, 44], mut: [96, 112, 138], acc: [10, 111, 184], acc2: [124, 92, 255], line: [222, 230, 240], zebra: [246, 249, 253],
  night: [7, 17, 31], paper: [255, 255, 255], warm: [255, 122, 69], cool: [70, 160, 230],
};
const SEV = { Critical: [229, 72, 77], High: [240, 140, 46], Moderate: [214, 170, 20], Low: [76, 139, 245] };
const W = 210, H = 297, M = 16, CW = W - 2 * M;

/** r = { win, shock, mode, day, image, impact, chart, city, alerts, prop, tele, state?, tour? } (assembled in main.js downloadReport) */
export function makeReport(r) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4', compress: true });
  let y = 0, page = 1;
  const T = (t, x, yy, o = {}) => {
    const { size = 9.5, color = C.ink, bold = false, align = 'left', maxW } = o;
    doc.setFont('helvetica', bold ? 'bold' : 'normal'); doc.setFontSize(size); doc.setTextColor(...color);
    const lines = maxW ? doc.splitTextToSize(ascii(t), maxW) : [ascii(t)];
    doc.text(lines, x, yy, { align, lineHeightFactor: 1.35 });
    return lines.length * size * 0.353 * 1.35;                       // height used, in mm
  };
  // a paragraph that starts a new page instead of running into the footer
  const para = (t, o = {}) => { const h = doc.setFontSize(o.size || 9.5) && doc.splitTextToSize(ascii(t), CW).length * (o.size || 9.5) * 0.353 * 1.35 + 2; room(h); y += T(t, M, y, { size: 9.5, maxW: CW, ...o }) + 2; };
  const rule = (yy, color = C.line, w = 0.25) => { doc.setDrawColor(...color); doc.setLineWidth(w); doc.line(M, yy, W - M, yy); };
  const newPage = () => { doc.addPage(); page++; y = 24; };
  const room = h => { if (y + h > H - 18) newPage(); };
  const section = title => {
    room(18); y += 4;
    doc.setFillColor(...C.acc); doc.roundedRect(M, y - 4.2, 2.2, 6.6, 1, 1, 'F');
    T(title, M + 5, y, { size: 12.5, bold: true }); y += 6; rule(y); y += 5;
  };
  const pill = (x, yy, label, col, w = 22) => {
    doc.setFillColor(...col); doc.roundedRect(x, yy - 3.6, w, 5.6, 1.6, 1.6, 'F');
    T(label, x + w / 2, yy + 0.2, { size: 7.6, bold: true, color: C.paper, align: 'center' });
  };
  // table with a dark header, zebra rows, right-aligned numbers; a cell may be {t, c (colour), b (bold)}
  const table = (head, rows, widths, size = 8.6) => {
    const rowH = 5.8; room(rowH * 2);
    doc.setFillColor(...C.night); doc.rect(M, y - 4, CW, rowH, 'F');
    let x = M; head.forEach((h, i) => { T(h, i ? x + widths[i] - 2 : x + 2, y, { size: 8, bold: true, color: C.paper, align: i ? 'right' : 'left' }); x += widths[i]; });
    y += rowH - 1;
    rows.forEach((row, ri) => {
      room(rowH); if (ri % 2) { doc.setFillColor(...C.zebra); doc.rect(M, y - 4, CW, rowH, 'F'); }
      let xx = M; row.forEach((c, i) => {
        const cell = typeof c === 'object' ? c : { t: c };
        T(cell.t, i ? xx + widths[i] - 2 : xx + 2, y, { size, bold: !!cell.b, color: cell.c || C.ink, align: i ? 'right' : 'left' }); xx += widths[i];
      });
      y += rowH;
    });
    rule(y - 3.4, C.line, 0.2); y += 3;
  };
  const footer = () => {
    for (let p = 1; p <= page; p++) {
      doc.setPage(p);
      if (p > 1) rule(H - 13, C.line, 0.2);
      T(`INDRA · Team Omnix · commit ${r.commit || 'dev'} · Not an official IMD forecast`, M, H - 8.5, { size: 7.4, color: C.mut });
      T(`page ${p} of ${page}`, W - M, H - 8.5, { size: 7.4, color: C.mut, align: 'right' });
    }
  };

  // ---------------- cover: dark band, event, snapshot, key numbers
  doc.setFillColor(...C.night); doc.rect(0, 0, W, 78, 'F');
  doc.setFillColor(...C.acc2); doc.rect(0, 78, W, 1.2, 'F');
  T('INDRA', M, 24, { size: 30, bold: true, color: [232, 239, 251] });
  T('Climate digital twin of India  ·  Scenario report', M, 33, { size: 11, color: [150, 172, 205] });
  T(`Team Omnix · DevForge Track 2 · ${r.tele?.model || 'INDRA-v1'} · generated ${new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}`, M, 40, { size: 8.6, color: [120, 146, 182] });
  T(`${r.tour ? '' : `Start date ${r.win.date} · `}last run ${r.lastRun ? r.lastRun.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }) : '-'} · NOT AN OFFICIAL IMD FORECAST`, M, 46, { size: 8.2, bold: true, color: [255, 190, 110] });
  T(r.tour ? `Guided tour: ${r.tour.length} events` : r.win.title, M, 58, { size: 16, bold: true, color: [240, 245, 255] });
  T(r.tour ? 'Every event, each from its own start date' : `Last input day ${r.win.date}  ·  ${r.win.desc}`, M, 66, { size: 9, color: [160, 182, 214], maxW: CW });
  y = 88;
  if (r.image) {
    const h = Math.min(92, CW * r.image.h / r.image.w), w = Math.min(CW, h * r.image.w / r.image.h);
    doc.setFillColor(...C.line); doc.roundedRect(M - 1, y - 1, w + 2, h + 2, 3, 3, 'F');
    doc.addImage(r.image.data, 'JPEG', M, y, w, h, undefined, 'FAST'); y += h + 7;
  }
  const kpi = (x, label, value, unit, note, col) => {
    doc.setFillColor(...C.paper); doc.setDrawColor(...C.line); doc.roundedRect(x, y, 42.5, 26, 3, 3, 'FD');
    doc.setFillColor(...col); doc.roundedRect(x, y + 3, 1.2, 20, 0.6, 0.6, 'F');
    T(label, x + 5, y + 6.5, { size: 7.4, color: C.mut, bold: true });
    T(`${value} ${unit}`.trim(), x + 5, y + 15, { size: 13.5, bold: true });
    if (note) T(note, x + 5, y + 21.5, { size: 7.4, color: col, maxW: 36 });
  };
  const find = p => r.impact.find(i => i[0].startsWith(p));
  const hi = find('Heatwave'), pk = find('Peak daily'), cs = find('Crop');
  const tot = r.tour ? r.tour.reduce((a, e) => a + e.alerts, 0) : 0;
  const dlt = (i, unit) => (r.shock && i[2] != null ? `${i[2] - i[1] > 0 ? '+' : ''}${(i[2] - i[1]).toFixed(1)} ${unit} with shock` : null);
  if (r.tour) {
    const peak = Math.max(...r.tour.map(e => e.peakRain)), tm = r.tour.reduce((a, e) => a + e.meanT, 0) / r.tour.length;
    kpi(M, 'EVENTS', `${r.tour.length}`, '', '2005-2021 start dates', C.acc);
    kpi(M + 45, 'ALERTS, ALL EVENTS', `${tot}`, '', 'heavy rain and heat, summed', C.acc2);
    kpi(M + 90, 'PEAK RAIN, ANY EVENT', peak.toFixed(0), 'mm', 'largest daily total', C.cool);
    kpi(M + 135, 'MEAN TEMPERATURE', tm.toFixed(1), 'C', 'average over the events', C.warm);
  } else {
    kpi(M, 'HEATWAVE EXPOSURE', hi[1].toFixed(1), '%', dlt(hi, 'pts') || 'share of land, Prime', C.warm);
    kpi(M + 45, 'ALERTS', `${r.alerts.list.length}`, '', `${r.alerts.counts.High || 0} High · ${r.alerts.counts.Moderate || 0} Moderate`, C.acc2);
    kpi(M + 90, 'PEAK DAILY RAIN', pk[1].toFixed(0), 'mm', dlt(pk, 'mm') || '10-day maximum', C.cool);
    kpi(M + 135, 'CROP WATER STRESS', cs[1].toFixed(1), '%', dlt(cs, 'pts') || 'rain deficit x heat index', C.ink);
  }
  y += 34;
  T(`View: ${r.mode} universe, day +${r.day}${r.shock ? `  ·  Shock: ${r.shock.name} (${r.shock.lat.toFixed(1)}N, ${r.shock.lon.toFixed(1)}E, radius ${r.shock.radius.toFixed(1)} deg)` : '  ·  no shock applied'}`, M, y, { size: 8.4, color: C.mut, maxW: CW });
  y += 8; T('Every number comes from live POST /simulate output, except the Observed line (recorded weather from the kit data).', M, y, { size: 7.8, color: C.mut, maxW: CW });
  newPage();

  // ---------------- impact
  section(r.tour ? 'Impact: the last event shown on the tour' : 'Impact over the 10-day forecast');
  table(['Indicator', 'Prime', r.shock ? 'Multiverse' : '', r.shock ? 'Change' : ''], r.impact.map(([label, p, m, u, dig]) => {
    const d = m == null ? '' : `${m - p > 0 ? '+' : ''}${(m - p).toFixed(Math.max(dig, 1))} ${u.includes('%') ? 'pts' : u}`;
    return [label, `${p.toFixed(dig)} ${u}`, m == null ? '' : `${m.toFixed(dig)} ${u}`, { t: d, c: m != null && m - p > 0.0001 ? C.warm : m != null && m - p < -0.0001 ? C.cool : C.mut }];
  }), [70, 34, 34, 40]);

  // ---------------- chart
  section(`All-India land mean: ${r.chart.name} (${r.chart.unit})`);
  room(68);
  const cx = M + 13, cw = CW - 18, ch = 46, top = y + 2, all = r.chart.series.flatMap(s => s[2]);
  let lo = Math.min(...all), hiV = Math.max(...all); const pad = (hiV - lo) * 0.15 || 1; lo -= pad; hiV += pad;
  const X = d => cx + d / 10 * cw, Yv = v => top + (1 - (v - lo) / (hiV - lo)) * ch;
  for (let k = 0; k <= 4; k++) { const v = lo + (hiV - lo) * k / 4; doc.setDrawColor(...C.line); doc.setLineWidth(0.15); doc.line(cx, Yv(v), cx + cw, Yv(v)); T(v.toFixed(1), cx - 2, Yv(v) + 1, { size: 6.8, color: C.mut, align: 'right' }); }
  for (let d = 0; d <= 10; d += 2) T(d ? `+${d}` : 'init', X(d), top + ch + 4, { size: 6.8, color: C.mut, align: 'center' });
  doc.setDrawColor(...C.line); doc.setLineWidth(0.15); doc.line(X(r.day), top, X(r.day), top + ch);
  r.chart.series.forEach(([name, col, vals, dashed], si) => {
    doc.setDrawColor(...col); doc.setLineWidth(dashed ? 0.5 : 0.9); doc.setLineDashPattern(dashed ? [1.4, 1] : [], 0);
    for (let d = 1; d < vals.length; d++) doc.line(X(d - 1), Yv(vals[d - 1]), X(d), Yv(vals[d]));
    doc.setLineDashPattern([], 0);
    doc.setFillColor(...col); doc.circle(X(vals.length - 1), Yv(vals[vals.length - 1]), 0.9, 'F');
    doc.setDrawColor(...col); doc.setLineWidth(0.9); doc.line(cx + si * 46, top + ch + 9, cx + si * 46 + 7, top + ch + 9); T(name, cx + si * 46 + 9, top + ch + 10, { size: 7.8, color: C.mut });
  });
  y = top + ch + 16;

  // ---------------- city
  if (r.city) {
    section(`City forecast: ${r.city.name}`);
    const f = (v, d, dig) => `${v.toFixed(dig)}${d == null || Math.abs(d) < 0.05 ? '' : `  (${d > 0 ? '+' : ''}${d.toFixed(dig)})`}`;
    table(['Day', 'Temp (C)', 'Rain (mm/day)', 'Wind (m/s)'], r.city.rows.map(([day, t, rr, w, dt, dr, dw]) =>
      [`+${day}`, f(t, dt, 1), f(rr, dr, 1), f(w, dw, 1)]), [30, 50, 50, 50]);
  }

  // ---------------- state (when a state was chosen)
  if (r.state) {
    const s = r.state;
    section(`State: ${s.name} (${s.cells} grid cells)`);
    table(['Measure', 'Value', 'Note'], [
      ['Mean temperature', `${s.meanT.toFixed(1)} C`, 'area-weighted, 10 days'],
      ['Mean rain', `${s.meanR.toFixed(1)} mm/day`, 'area-weighted, 10 days'],
      ['Mean wind', `${s.meanW.toFixed(1)} m/s`, '10 m wind speed'],
      ['Heatwave cells (peak)', `${s.heatPct.toFixed(0)} %`, 'share of state cells in a heat day'],
      ['Heavy-rain cells (peak)', `${s.floodPct.toFixed(0)} %`, 'share with >= 64.5 mm/day'],
      ['Alerts in this state', `${s.alerts}`, `${s.high} High or Critical`],
      ['ML flood chance (max)', s.floodRisk == null ? '-' : `${s.floodRisk.toFixed(0)} %`, 'INDRA-Risk, heavy rain'],
      ['ML heat chance (max)', s.heatRisk == null ? '-' : `${s.heatRisk.toFixed(0)} %`, 'INDRA-Risk, heat stress'],
    ], [62, 42, 74]);
  }

  // ---------------- alerts
  const al = r.alerts.list.length;
  section(`Alerts and warnings (${al} in this universe)`);
  room(12); let px = M;
  for (const sv of ['Critical', 'High', 'Moderate', 'Low']) { pill(px, y, `${sv}  ${r.alerts.counts[sv] || 0}`, SEV[sv], 40); px += 44; }
  y += 8;
  if (!al) { T('No heavy-rain or heat alerts in this universe.', M, y, { color: C.mut }); y += 6; }
  for (const a of r.alerts.list.slice(0, 26)) {
    room(7.5); pill(M, y, a.severity, SEV[a.severity], 17);
    const what = a.type === 'cyclone' ? 'Cyclone risk' : a.type === 'rain' ? (a.severity === 'Low' ? 'Heavy-rain watch' : 'Heavy rain') : (a.severity === 'Low' ? 'Heat watch' : 'Heat stress');
    const val = a.type === 'cyclone' ? `low-pressure core, wind up to ${a.value.toFixed(0)} m/s` : a.type === 'rain' ? `peak ${Math.round(a.value)} mm/day` : `daily mean up to ${a.value.toFixed(1)} C`;
    T(`${what}  ·  day +${a.day}  ·  ${a.region}  ·  ${val}  ·  ${a.cells} cells${a.status === 'new' ? '  ·  NEW in multiverse' : a.status === 'worse' ? '  ·  MORE SEVERE' : ''}`, M + 20, y + 0.6, { size: 8.2, maxW: CW - 20 });
    y += 6.2;
  }
  if (al > 26) { T(`+ ${al - 26} more in the dashboard`, M + 20, y + 1, { size: 7.8, color: C.mut }); y += 6; }
  y += T('Rules: heavy rain = IMD >= 64.5 mm/day (High >= 115.6, Critical >= 204.5). Heat = daily mean >= 35 C or >= 4.5 K above normal. Low = INDRA-Risk watch.', M, y + 2, { size: 7.4, color: C.mut, maxW: CW }) + 4;

  // ---------------- tour summary
  if (r.tour) {
    section('Tour summary: every event');
    table(['Event', 'Date', 'Alerts', 'Peak rain', 'Mean temp'], r.tour.map(e => [e.title, e.date, `${e.alerts}`, `${e.peakRain.toFixed(0)} mm`, `${e.meanT.toFixed(1)} C`]), [66, 28, 22, 30, 32]);
  }

  // ---------------- engine, method, limits
  section('Shock causality, engine and method');
  if (r.prop) para(`Day-1 response energy near the shock: ${Math.round(r.prop.near * 100)}%. The response moves from ${Math.round(r.prop.r1)} km (day 1) to ${Math.round(r.prop.r10)} km (day 10) from the shock.`);
  para(`Model: ${r.tele.model}. 10-day inference ${Math.round(r.tele.ms)} ms on CPU. Zero-shock check: ${r.tele.zero == null ? 'n/a' : r.tele.zero < 1e-3 ? 'passes (identical output)' : 'FAILS'}.`);
  para('Method. A residual CNN (534,726 weights) predicts one day at a time on a 32 x 36 grid (1 degree, about 110 km). It was trained on 2005-2020 daily data and tested on 2021. Calibration shrinks far-day anomalies towards the climate. Shocks are added to the last observed day and the full difference is carried forward.', { size: 8.6 });
  para('Limits. Heavy-rain skill fades after day 1; one cell is about 110 km, so there is no street-level weather; alerts name the nearest state centre; crop stress is a simple index, not agency-tuned. Research prototype, not an official forecast.', { size: 8.6, color: C.mut });

  footer();
  const name = r.tour ? `INDRA-tour-report-${new Date().toISOString().slice(0, 10)}.pdf`
    : `INDRA-report-${r.win.date}-${r.shock ? r.shock.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') : 'prime'}.pdf`;
  doc.save(name);
  return name;
}
