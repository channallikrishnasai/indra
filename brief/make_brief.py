"""Builds brief/INDRA_brief.pdf: exactly 2 pages, A4.   python brief/make_brief.py
Page 1: model design, drift and blur, results, failure modes.  Page 2: track requirements, visualisation, impact use case.
Every number is taken from a run recorded in the repo (evaluate.py output, dashboard/public/validation/*.json, docs/ charts)."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepInFrame)

HERE = Path(__file__).parent; ROOT = HERE.parent
F = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("A", str(F / "arial.ttf"))); pdfmetrics.registerFont(TTFont("AB", str(F / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("AI", str(F / "ariali.ttf")))
pdfmetrics.registerFontFamily("A", normal="A", bold="AB", italic="AI", boldItalic="AB")

C = lambda h: colors.HexColor(h)
INK, MUT, ACC, VIO, WARM, GOOD = C("#0e1a2c"), C("#5a6c86"), C("#0a6fb8"), C("#6a4cf0"), C("#e0622f"), C("#18a06e")
NIGHT, PAPER, LINE, TINT = C("#07111f"), C("#ffffff"), C("#d3deec"), C("#f1f6fc")

body = ParagraphStyle("b", fontName="A", fontSize=8.9, leading=11.2, textColor=INK, spaceAfter=2.2)
small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.4, textColor=MUT, spaceAfter=1.5)
cap = ParagraphStyle("cap", parent=small, fontName="AI")
h2 = ParagraphStyle("h2", fontName="AB", fontSize=10.6, leading=13, textColor=ACC, spaceBefore=5, spaceAfter=2.5, keepWithNext=1)
cell = ParagraphStyle("c", parent=body, fontSize=8, leading=9.8, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="AB")
kpiv = ParagraphStyle("kv", fontName="AB", fontSize=11, leading=13, textColor=INK)
kpil = ParagraphStyle("kl", fontName="A", fontSize=7.3, leading=8.8, textColor=MUT)
P = lambda t, s=body: Paragraph(t, s)
B = lambda items, s=body: [P("•&nbsp;&nbsp;" + t, s) for t in items]
W = 182 * mm


def h(title):
    return P(title, h2)


def table(rows, widths, extra=None):
    t = Table([[P(str(c), cellb if (i == 0 or j == 0) and i == 0 else cell) for j, c in enumerate(r)] for i, r in enumerate(rows)], colWidths=widths)
    style = [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACC),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)] + (extra or [])
    t.setStyle(TableStyle(style))
    return t


def kpi_row(items):
    """items: (value, label, colour). Cards with a coloured top rule."""
    cells = [[P(v, kpiv)] for v, _, _ in items]
    labels = [[P(l, kpil)] for _, l, _ in items]
    t = Table([[Table([[c[0]], [lb[0]]], colWidths=[W / len(items) - 6 * mm], style=[("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)])
                for c, lb in zip(cells, labels)]], colWidths=[W / len(items)] * len(items))
    style = [("BOX", (0, 0), (-1, -1), 0.4, LINE), ("BACKGROUND", (0, 0), (-1, -1), PAPER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]
    for k, (_, _, col) in enumerate(items):
        style.append(("LINEABOVE", (k, 0), (k, 0), 2.2, col))
    t.setStyle(TableStyle(style))
    return t


def callout(text, col=ACC):
    t = Table([[P(text, body)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 3, col),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def arch():
    d = Drawing(W, 27 * mm)
    boxes = [("Dashboard", "3D globe · 2D map · audit", 0, ACC), ("serve.py", "POST /simulate · /risk", 37.5, ACC),
             ("INDRA stepper", "local CNN · 7 cells/day", 75, VIO), ("Calibration", "shrink · rain map", 112.5, VIO),
             ("Impact + INDRA-Risk", "heat · flood · crops · wind", 150, WARM)]
    for t, s, x, col in boxes:
        d.add(Rect(x * mm, 7 * mm, 32 * mm, 16 * mm, rx=6, ry=6, fillColor=PAPER, strokeColor=col, strokeWidth=1))
        d.add(String((x + 16) * mm, 16.8 * mm, t, fontName="AB", fontSize=8.2, fillColor=INK, textAnchor="middle"))
        d.add(String((x + 16) * mm, 11 * mm, s, fontName="A", fontSize=6.8, fillColor=MUT, textAnchor="middle"))
    for x in (32, 69.5, 107, 144.5):
        d.add(Line(x * mm, 15 * mm, (x + 4.9) * mm, 15 * mm, strokeColor=ACC, strokeWidth=1.1))
        d.add(Polygon([(x + 5.5) * mm, 15 * mm, (x + 4.1) * mm, 16 * mm, (x + 4.1) * mm, 14 * mm], fillColor=ACC, strokeColor=ACC))
    d.add(String(W / 2, 1.5 * mm, "init_states [3,6,32,36] + shock  →  10-day rollout, physical units (0.1–0.2 s on CPU)  →  live maps, charts, impact cards",
                 fontName="AI", fontSize=7.4, fillColor=MUT, textAnchor="middle"))
    return d


def header_footer(canv, doc):
    canv.saveState()
    canv.setFillColor(NIGHT); canv.rect(0, A4[1] - 16 * mm, A4[0], 16 * mm, stroke=0, fill=1)
    canv.setFillColor(ACC); canv.rect(0, A4[1] - 16.9 * mm, A4[0], 0.9 * mm, stroke=0, fill=1)
    canv.setFillColor(PAPER); canv.setFont("AB", 10.5)
    canv.drawString(13 * mm, A4[1] - 10.2 * mm, "INDRA  ·  climate digital twin of India")
    canv.setFont("A", 7.4); canv.setFillColor(C("#9fb4d4"))
    canv.drawRightString(A4[0] - 13 * mm, A4[1] - 10.2 * mm, "Team Omnix · Yugaantar DevForge · Track 2 “Multiverse of Monsoon”")
    canv.setFillColor(MUT); canv.setFont("A", 6.8)
    canv.drawString(13 * mm, 7 * mm, "Research prototype · not an official IMD forecast · github.com/channallikrishnasai/indra · brief 27 Nov 2026")
    canv.drawRightString(A4[0] - 13 * mm, 7 * mm, f"page {doc.page} of 2")
    canv.restoreState()


# ---------------------------------------------------------------- numbers (from the repo's own outputs)
val = {y: json.load(open(ROOT / f"dashboard/public/validation/{y}.json", encoding="utf-8")) for y in (2019, 2020, 2021)}

# ---------------------------------------------------------------- page 1
p1 = []
p1 += [P("<font size=21><b>INDRA: a climate digital twin of India</b></font>", ParagraphStyle("t", fontName="AB", fontSize=21, leading=24, textColor=INK)),
       P("Five futures of the monsoon from one causal model. Simulate the next 10 days from the last 3 observed days, then branch alternate universes from a local shock.", ParagraphStyle("st", parent=body, textColor=MUT, fontSize=9.6, leading=12.5)),
       Spacer(1, 5)]
p1 += [callout("<b>USP.</b> INDRA is <b>causal by construction</b>: 0.53 M parameters, a receptive field of 7 grid cells per day, so a shock "
               "cannot travel faster than about 7° a day. It stays local on day one and spreads afterwards, and a zero shock changes nothing. "
               "Calibration and a rain map keep long rollouts sharp and on climate, and every forecast drives a live dashboard, not a pre-rendered result.")]
p1 += [Spacer(1, 6), kpi_row([
    ("0.278 · 0.111 · 0.016", "RMSE skill, days 1·4·8 (2021)", ACC),
    ("0.53 · 0.31 · 0.28", "Heavy-rain CSI at 10 mm/day, days 1·4·8", VIO),
    ("43.7 / 45", "Estimated automatic score (2021, full run)", GOOD),
    ("0.139 · 0.137 · 0.133", "Validation skill mean, 2019 · 2020 · 2021", WARM),
])]
p1 += [h("1 · Model design")]
p1 += B([
    "<b>Input:</b> the last three days × six channels (t2m, tp, u10, v10, msl, tcwv; rain as log(1+rain)), the target day’s climatology, land/sea mask, orography, day-of-year and lat/lon.",
    "<b>Network:</b> a 3×3 stem, three residual blocks (two 3×3 convolutions each, GroupNorm, GELU) and a 1×1 head. It predicts the <i>tendency</i>, added to the last state.",
    "<b>Hard physical limits on every output:</b> rain ≥ 0, moisture ≥ 0, t2m within −40 … +60 °C. An all-zero shock reproduces the no-shock run exactly.",
    "<b>Training (CPU, about 1.5 h):</b> single-step teacher forcing, then a 2 → 4 → 6-step autoregressive rollout curriculum, latitude-weighted MSE. Years 2005–2020 train; <b>2021 is held out</b>.",
])
p1 += [Spacer(1, 3), arch(), Spacer(1, 1)]
p1 += [h("2 · How we prevent drift and blur")]
p1 += B([
    "<b>Rollout training</b> teaches the model to correct its own errors: validation day-8 error fell from 0.590 after single-step training to 0.496 after the 6-step curriculum (normalised).",
    "<b>Per-lead calibration:</b> each lead is shrunk towards climate, α fitted by least squares on training years only. Rain is fitted in mm/day, so heavy rain survives. "
    "A rain quantile map (half strength) lifts the heavy-rain CSI score from 6.7 to 13.6 points of 15 (commit d8dd318); the full 2021 run now gives 13.7.",
    "<b>Blur check:</b> day-8 spatial variance is 0.68–1.01 of the real field for every channel (judge band 0.40–2.5), and all four causality checks pass for the warm-ocean shock.",
    "<b>Multiverse</b> = calib(F(x)) + [F(x + shock) − F(x)]: calibration never mutes a shock, and the shock’s own response is kept whole.",
])
p1 += [h("3 · Results on held-out 2021 (355 windows, the organisers’ monsoon_r1 scoring code)")]
rows = [["Lead", "t2m", "tp (rain)", "u10", "v10", "msl", "tcwv", "Mean skill", "Heavy-rain CSI, 10 mm/day (INDRA / persist / clim)"],
        ["Day 1", ".205", ".276", ".257", ".304", ".328", ".297", "<b>.278</b>", "<b>.53</b> / .43 / .26"],
        ["Day 4", ".141", ".035", ".109", ".099", ".105", ".174", "<b>.111</b>", "<b>.31</b> / .21 / .26"],
        ["Day 8", ".044", "<font color='#e0622f'><b>−.042</b></font>", ".014", ".013", ".027", ".040", "<b>.016</b>", "<b>.28</b> / .20 / .26"]]
p1 += [table(rows, [14 * mm, 13 * mm, 17 * mm, 13 * mm, 13 * mm, 13 * mm, 13 * mm, 19 * mm, 67 * mm],
             [("ALIGN", (1, 0), (7, -1), "RIGHT")]),
       Spacer(1, 2),
       P("Skill = 1 − RMSE / RMSE of the better of repeat-yesterday and climatology, averaged per window. The official pooled aggregation gives 0.139; both clear the 0.12 reference. "
         "CSI is pooled over all windows and cells; the judge’s 15 points use the mean of the per-lead CSI skill (kit commit 689c879, monsoon_r1/scoring.py).", small)]
p1 += [h("4 · Failure modes and limits (stated plainly)")]
p1 += B([
    "<b>Day-8 rain is worse than the better reference:</b> skill −0.042 on 2021. Rain at day 8 is marginal for CSI (0.28 vs 0.26 climatology, 0.20 persistence), not failing. Do not treat all ten leads as equally reliable.",
    "<b>Heavy rain at IMD 64.5 mm/day (not scored):</b> CSI 0.11 / 0.01 / 0.02 at days 1 / 4 / 8 on 2021, against persistence 0.09 / 0.00 / 0.00 and climatology 0.",
    "<b>Monsoon-stall shock stays near its box:</b> on 15 July 2021, response energy falls from 259 (day 1) to 117 (day 10); the share inside the box falls from 0.62 to 0.29. It fails the judge’s “propagates” check (about 1.9 of 15 causality points).",
    "<b>Resolution and scope:</b> one cell is about 110 km (1° grid). No street-level weather. India only.",
    "<b>Validation, not a hidden test:</b> 2019 (model trained to 2018) and 2020 (trained to 2019) use the same recipe; 2021 is the shipped model, which used 2021 to pick some settings, so its score is optimistic. We cannot test the 2023–24 hidden windows.",
    "<b>Other:</b> alerts and impact indices are rule-based research thresholds, not agency-tuned. A leap-year day-of-year offset is reported and not corrected. The 7%/K moisture rule in the tipping-point tool is a fixed assumption.",
])
p1 += [Spacer(1, 2), Image(str(ROOT / "docs" / "chart-csi-2021.png"), width=118 * mm, height=118 * mm * 3.6 / 7.2)]

# ---------------------------------------------------------------- page 2
p2 = []
p2 += [h("5 · Track 2 requirements and where they are met")]
req = [["Requirement (track brief)", "Evidence in this repo"],
       ["Stable rollouts, 10 steps or more, no NaN or Inf, rain never negative", "Validator 6/6 (10-step calls, dry runs, base64 and list formats); hard limits in net.py; every 2021 window passes validity"],
       ["Real weather skill: beat repeat-yesterday and climatology, including heavy rain", "Table in section 3: day 4 and day 8 beat both on the skill score (day 8 rain is the stated exception)"],
       ["Causal shocks: local first, then outward; a zero shock changes nothing", "Warm-ocean shock passes all four checks; monsoon-stall is documented as incomplete; zero shock 0.0 difference"],
       ["Real impact: dashboard and impact module driven by our own simulation", "Dashboard calls /simulate live; impact cards, state audit, validation panel, ensemble audit"],
       ["Offline serve.py, POST /health, POST /simulate, physical units, under 5 s", "serve.py on port 8000; 0.06–0.11 s per 10-step call on this CPU; no weather APIs, no data archives"],
       ["Repo: training code, serve.py, weights, dashboard separate from server, README", "train.py, calibrate*.py, serve.py at root; weights/indra.pt tracked; dashboard/ separate; README"],
       ["Repo scan: no hardcoded dates or archives; real training code", "Repo scan 15/15 on a fresh clone; the date-based helper tools are dashboard-only and optional"]]
p2 += [table(req, [72 * mm, 110 * mm]), Spacer(1, 2)]
p2 += [h("6 · Visualisation: the dashboard (live from POST /simulate, nothing pre-rendered)")]
p2 += [Image(str(HERE / "img" / "dashboard_delta.jpg"), width=112 * mm, height=112 * mm * 1800 / 3200)]
p2 += B([
    "<b>3D globe</b> lit by the real sun for the date, with lit clouds, wind, rain and lightning from the simulation; <b>2D map</b> with borders above the data.",
    "<b>Universes:</b> Timeline Prime, Multiverse, Δ shock-impact, Ensemble σ and a heavy-rain chance map (five perturbed copies, labelled as a spread, not a probability).",
    "<b>Analysis tools:</b> causal reach map, tipping-point search, state audit (affected and unaffected states), +1 K what-if, alerts with IMD thresholds, and a validation panel with hover and full-screen detail.",
])
p2 += [Spacer(1, 2), Image(str(ROOT / "docs" / "chart-validation.png"), width=112 * mm, height=112 * mm * 3.4 / 7.2)]
p2 += [P("Validation panel: 2019 and 2020 are the recipe retrained on earlier years only (rolling-origin); 2021 is the shipped model. Source: rolling_export.py outputs.", cap)]
p2 += [h("7 · Impact use case: what a shock means for people")]
p2 += [P("Each universe becomes decisions, computed live from the rollout and the model’s own climatology. Real results, monsoon-onset window (10-day totals, Prime → Multiverse):")]
imp = [["Shock", "Heatwave exposure", "Crop water stress", "Rain vs normal", "Peak rain"],
       ["Heat dome +4 K, NW India", "<b>2.6 → 9.5 %</b>", "19.1 → 17.6 %", "−21 → −20 %", "21 → 21 mm"],
       ["Warm Bay of Bengal (+3 K, +15 % moisture)", "2.6 → 2.5 %", "19.1 → 18.0 %", "<b>−21 → −18 %</b>", "<b>21 → 26 mm</b>"],
       ["Monsoon stall (−40 % Arabian Sea wind)", "2.6 → 2.7 %", "19.1 → 18.4 %", "−21 → −20 %", "21 → 21 mm"]]
p2 += [table(imp, [62 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm], [("ALIGN", (1, 0), (-1, -1), "RIGHT")]), Spacer(1, 2)]
p2 += B([
    "<b>Who it helps:</b> health departments (heatwave exposure), NDMA and SDMA (flood-risk area at IMD heavy rain ≥ 64.5 mm/day), kharif planning (crop water stress = rain deficit × heat), grid operators (wind power at 100 m).",
    "<b>Ensemble:</b> five perturbed copies show where heavy rain is agreed. On 20 July 2021, five of five copies agree in five land cells on day +3. An agreement score is not accuracy.",
    "<b>INDRA-Risk</b> (a small CNN trained on INDRA’s own forecasts against what happened) gives heavy-rain and heat probabilities; held-out 2021 ROC-AUC 0.90 (heavy rain) and 0.99 (heat).",
])
p2 += [Spacer(1, 2), Image(str(ROOT / "docs" / "chart-monsoon-stall.png"), width=112 * mm, height=112 * mm * 3.4 / 7.2)]
p2 += [h("8 · Integrity and reproducibility")]
p2 += B([
    "<b>No lookup:</b> the model reads only the three input days, the calendar position and the static fields; no weather archive or API is used at inference. The 366-day climatology is an average of 2005–2020, smoothed over 15 days, stored as a model input.",
    "<b>Run:</b> <i>python serve.py --port 8000</i>, then <i>python -m monsoon_r1.schema_validator http://127.0.0.1:8000</i> (6/6). Train in about 1.5 h on CPU from the kit data (train.py, calibrate.py, calibrate_rain.py, train_risk.py).",
    "<b>Honesty:</b> we report the day-8 rain skill, the monsoon-stall limit and the 2021-based choices as they are; the hidden 2023–24 score is unknown to us and is not claimed here.",
])

# ---------------------------------------------------------------- assemble: each page is a fixed frame that shrinks to fit
out = HERE / "INDRA_brief.pdf"
frame_h = A4[1] - 22 * mm - 14 * mm
story = [KeepInFrame(W, frame_h, p1, mode="shrink"), PageBreak(), KeepInFrame(W, frame_h, p2, mode="shrink")]
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=22 * mm, bottomMargin=14 * mm,
                  title="INDRA: climate digital twin of India", author="Team Omnix").build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print("wrote", out)
