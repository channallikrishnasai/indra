// Scenario report: a PDF of the current run, built in the browser (jsPDF is loaded only when this module is).
import { jsPDF } from 'jspdf';

// jsPDF's built-in Helvetica has no Δ, ≥, →, − or σ: map them to plain text
const clean = t => String(t).replace(/Δ/g, 'd').replace(/≥/g, '>=').replace(/→/g, '->').replace(/[−–]/g, '-').replace(/σ/g, 'sd').replace(/²/g, '2').replace(/×/g, 'x');
const INK = [14, 26, 44], MUT = [90, 108, 134], ACC = [10, 111, 184], LINE = [201, 214, 230];
const SEV = { Critical: [229, 72, 77], High: [240, 140, 46], Moderate: [214, 170, 20], Low: [76, 139, 245] };

/** r = { win, shock, mode, day, image:{data,w,h}|null, impact:[[label,prime,multi,unit]], chart:{name,unit,series:[[name,[r,g,b],values,dashed]]},
 *        city:{name, rows:[[day,t,r,w,dt,dr,dw]]}|null, alerts:{counts, list}, prop:{near,r1,r10}|null, tele:{model,ms,zero} } */
export function makeReport(r) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' }), W = 210, M = 14, CW = W - 2 * M;
  let y = 0;
  const text = (t, x, yy, { size = 9.5, color = INK, bold = false, align = 'left', maxW } = {}) => {
    doc.setFont('helvetica', bold ? 'bold' : 'normal'); doc.setFontSize(size); doc.setTextColor(...color);
    const lines = maxW ? doc.splitTextToSize(clean(t), maxW) : [clean(t)];
    doc.text(lines, x, yy, { align }); return lines.length * size * 0.42;
  };
  const need = h => { if (y + h > 284) { doc.addPage(); y = 16; } };
  const h2 = t => { need(14); y += 4; text(t, M, y, { size: 11.5, bold: true, color: ACC }); y += 2; doc.setDrawColor(...LINE); doc.line(M, y, W - M, y); y += 5; };
  const row = (cells, widths, { bold = false, color = INK, size = 8.8 } = {}) => {
    need(6); let x = M; cells.forEach((c, i) => { text(c, i ? x + widths[i] - 1 : x, y, { size, bold, color, align: i ? 'right' : 'left' }); x += widths[i]; }); y += 5;
  };

  // header
  doc.setFillColor(7, 17, 31); doc.rect(0, 0, W, 24, 'F');
  text('INDRA scenario report', M, 11, { size: 16, bold: true, color: [232, 239, 251] });
  text('Climate digital twin of India · Team Omnix', M, 17.5, { size: 9, color: [142, 160, 187] });
  text(new Date().toLocaleString('en-IN'), W - M, 17.5, { size: 8.5, color: [142, 160, 187], align: 'right' });
  y = 32;

  // scenario
  text(`${r.win.title} · last input day ${r.win.date}`, M, y, { size: 12, bold: true }); y += 5.5;
  y += text(r.win.desc + '.', M, y, { color: MUT, maxW: CW });
  const sh = r.shock ? `Shock: ${r.shock.name} at ${r.shock.lat.toFixed(1)}N ${r.shock.lon.toFixed(1)}E, radius ${r.shock.radius.toFixed(1)} deg (` +
    r.shock.parts.map(([v, m]) => `${v} ${m > 0 ? '+' : ''}${m}${v === 't2m' ? ' K' : v === 'msl' ? ' hPa' : v === 'tp' ? ' mm/day' : '%'}`).join(', ') + ')' : 'No shock applied: Timeline Prime only.';
  y += 1.5; y += text(sh, M, y, { bold: !!r.shock, maxW: CW });
  y += text(`View when exported: ${r.mode} universe, day +${r.day}. Every number below comes from live POST /simulate output, except the Observed line (recorded weather).`, M, y + 1, { size: 8.5, color: MUT, maxW: CW }) + 2;

  if (r.image) {
    const h = CW * r.image.h / r.image.w; need(h + 4);
    doc.addImage(r.image.data, 'JPEG', M, y, CW, h); y += h + 3;
  }

  // impact
  h2('Impact (10-day forecast)');
  const iw = [70, 36, 36, 40];
  row(['Indicator', 'Prime', r.shock ? 'Multiverse' : '', r.shock ? 'Change' : ''], iw, { bold: true, color: MUT, size: 8.3 });
  for (const [label, p, m, u, dig] of r.impact) {
    const cu = u.includes('%') ? 'pts' : u, cd = Math.max(dig, 1);                      // % changes are percentage points
    const ch = m == null ? '' : `${m - p > 0 ? '+' : ''}${(m - p).toFixed(cd)} ${cu}`;
    row([label, `${p.toFixed(dig)} ${u}`, m == null ? '' : `${m.toFixed(dig)} ${u}`, ch], iw);
  }

  // chart
  h2(`All-India land mean · ${r.chart.name} (${r.chart.unit})`);
  need(52);
  const cx = M + 12, cw = CW - 16, ch = 40, top = y, all = r.chart.series.flatMap(s => s[2]);
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.15 || 1; lo -= pad; hi += pad;
  const X = d => cx + d / 10 * cw, Y = v => top + (1 - (v - lo) / (hi - lo)) * ch;
  doc.setLineWidth(0.15); doc.setDrawColor(...LINE);
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; doc.line(cx, Y(v), cx + cw, Y(v)); text(v.toFixed(1), cx - 2, Y(v) + 1, { size: 7, color: MUT, align: 'right' }); }
  for (let d = 0; d <= 10; d += 2) text(d ? `+${d}` : 'init', X(d), top + ch + 4, { size: 7, color: MUT, align: 'center' });
  doc.setLineWidth(0.6);
  r.chart.series.forEach(([name, col, vals, dashed], i) => {
    doc.setDrawColor(...col); doc.setLineDashPattern(dashed ? [1.2, 1] : [], 0);
    for (let d = 1; d < vals.length; d++) doc.line(X(d - 1), Y(vals[d - 1]), X(d), Y(vals[d]));
    doc.setLineDashPattern([], 0); doc.line(cx + i * 42, top + ch + 8, cx + i * 42 + 6, top + ch + 8);
    text(name, cx + i * 42 + 8, top + ch + 9, { size: 7.8, color: MUT });
  });
  y = top + ch + 14;

  // city
  if (r.city) {
    need(14 + 5 * (r.city.rows.length + 1));                            // keep the heading with its table
    h2(`City forecast · ${r.city.name}${r.shock ? ' (Prime, multiverse change in brackets)' : ''}`);
    const cw4 = [24, 52, 52, 54];
    row(['Day', 'Temp C', 'Rain mm', 'Wind m/s'], cw4, { bold: true, color: MUT, size: 8.3 });
    const f = (v, d, dig) => `${v.toFixed(dig)}${d == null || Math.abs(d) < 0.05 ? '' : ` (${d > 0 ? '+' : ''}${d.toFixed(dig)})`}`;
    for (const [day, t, rr, w, dt, dr, dw] of r.city.rows) row([`+${day}`, f(t, dt, 1), f(rr, dr, 1), f(w, dw, 1)], cw4, { size: 8.4 });
  }

  // alerts
  h2(`Alerts & warnings (${r.alerts.list.length} in this universe)`);
  need(8); let x = M;
  for (const s of ['Critical', 'High', 'Moderate', 'Low']) {
    doc.setFillColor(...SEV[s]); doc.roundedRect(x, y - 3.6, 42, 6, 1.2, 1.2, 'F');
    text(`${s}: ${r.alerts.counts[s] || 0}`, x + 21, y + 0.6, { size: 8.6, bold: true, color: [255, 255, 255], align: 'center' }); x += 45;
  }
  y += 7;
  if (!r.alerts.list.length) { text('No heavy-rain or heat alerts in this universe.', M, y, { color: MUT }); y += 5; }
  for (const a of r.alerts.list.slice(0, 14)) {
    need(5.5); doc.setFillColor(...SEV[a.severity]); doc.circle(M + 1.3, y - 1.1, 1.2, 'F');
    const what = a.type === 'rain' ? (a.severity === 'Low' ? 'Heavy-rain watch' : 'Heavy rain') : (a.severity === 'Low' ? 'Heat watch' : 'Heat stress');
    const val = a.type === 'rain' ? `peak ${Math.round(a.value)} mm/day` : `daily mean up to ${a.value.toFixed(1)} C`;
    text(`${a.severity} · ${what} · day +${a.day} · ${a.region} · ${val} · ${a.cells} cells${a.status === 'new' ? ' · NEW in multiverse' : a.status === 'worse' ? ' · MORE SEVERE' : ''}`, M + 4, y, { size: 8.4 }); y += 4.8;
  }
  if (r.alerts.list.length > 14) { text(`+ ${r.alerts.list.length - 14} more in the dashboard.`, M + 4, y, { size: 8, color: MUT }); y += 5; }
  text('Rules: heavy rain = IMD >= 64.5 mm/day (>= 115.6 High, >= 204.5 Critical); heat = daily mean >= 35 C or >= 4.5 K above normal; Low = INDRA-Risk watch.', M, y + 1, { size: 7.6, color: MUT, maxW: CW }); y += 8;

  // shock + engine
  h2('Shock causality and engine');
  if (r.prop) y += text(`Day-1 response energy within the shock region: ${Math.round(r.prop.near * 100)}%. Mean distance of the response from the shock: ${Math.round(r.prop.r1)} km on day 1 -> ${Math.round(r.prop.r10)} km on day 10.`, M, y, { maxW: CW }) + 1;
  y += text(`Model ${r.tele.model}. 10-day inference ${Math.round(r.tele.ms)} ms on CPU. Zero-shock check: ${r.tele.zero == null ? 'n/a' : r.tele.zero < 1e-3 ? 'passes (identical output)' : 'FAILS'}.`, M, y, { maxW: CW }) + 3;
  y += text('Research prototype on a 1 degree (~110 km) grid trained on 2005-2020 daily data; not an official forecast. Held-out 2021 skill and limits: see "About the model" in the dashboard.', M, y, { size: 7.8, color: MUT, maxW: CW });

  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p++) { doc.setPage(p); text(`INDRA · Team Omnix · page ${p}/${pages}`, W / 2, 292, { size: 7.5, color: MUT, align: 'center' }); }
  const name = `INDRA-report-${r.win.date}-${r.shock ? r.shock.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') : 'prime'}.pdf`;
  doc.save(name);
  return name;
}
