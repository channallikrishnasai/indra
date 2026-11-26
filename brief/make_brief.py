"""Builds brief/INDRA_brief.pdf (2 pages, A4).   python brief/make_brief.py
Screenshots in brief/img/ come from the live dashboard (Monsoon onset 2021-06-02, Warm Bay of Bengal, Δ view, day +4)."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

HERE = Path(__file__).parent
F = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("A", str(F / "arial.ttf"))); pdfmetrics.registerFont(TTFont("AB", str(F / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("AI", str(F / "ariali.ttf")))
pdfmetrics.registerFontFamily("A", normal="A", bold="AB", italic="AI", boldItalic="AB")

INK, ACC, MUT, LINE = colors.HexColor("#0e1a2c"), colors.HexColor("#0a6fb8"), colors.HexColor("#5a6c86"), colors.HexColor("#c9d6e6")
body = ParagraphStyle("b", fontName="A", fontSize=8.6, leading=10.4, textColor=INK, spaceAfter=1.6)
small = ParagraphStyle("s", parent=body, fontSize=7.4, leading=9.0, textColor=MUT, spaceAfter=1)
h1 = ParagraphStyle("h1", fontName="AB", fontSize=17, leading=20, textColor=INK)
sub = ParagraphStyle("sub", parent=body, textColor=MUT, fontSize=9.2, leading=11.5)
h2 = ParagraphStyle("h2", fontName="AB", fontSize=9.6, leading=11.5, textColor=ACC, spaceBefore=3, spaceAfter=1.5, keepWithNext=1)
cell = ParagraphStyle("c", parent=body, fontSize=7.8, leading=9.4, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="AB")
cap = ParagraphStyle("cap", parent=small, fontName="AI")
P = lambda t, s=body: Paragraph(t, s)
B = lambda items, s=body: [P("• " + t, s) for t in items]


def table(rows, widths, st=None):
    t = Table([[P(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)], colWidths=widths)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, ACC), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)] + (st or [])))
    return t


def arch():
    d = Drawing(182 * mm, 25 * mm)
    boxes = [("Dashboard", "3D globe · 2D map", 0), ("serve.py", "POST /simulate · /risk", 37.5), ("INDRA stepper", "local CNN, 7 cells/day", 75),
             ("Calibration +", "full shock response", 112.5), ("Impact + INDRA-Risk", "heat · flood · crops", 150)]
    for t, s, x in boxes:
        d.add(Rect(x * mm, 7 * mm, 32 * mm, 15 * mm, rx=6, ry=6, fillColor=colors.HexColor("#eef5fc"), strokeColor=ACC, strokeWidth=0.8))
        d.add(String((x + 16) * mm, 16.3 * mm, t, fontName="AB", fontSize=8.6, fillColor=INK, textAnchor="middle"))
        d.add(String((x + 16) * mm, 10.8 * mm, s, fontName="A", fontSize=7.2, fillColor=MUT, textAnchor="middle"))
    for x in (32, 69.5, 107, 144.5):
        d.add(Line(x * mm, 14.5 * mm, (x + 4.9) * mm, 14.5 * mm, strokeColor=ACC, strokeWidth=1))
        d.add(Polygon([(x + 5.5) * mm, 14.5 * mm, (x + 4.1) * mm, 15.5 * mm, (x + 4.1) * mm, 13.5 * mm], fillColor=ACC, strokeColor=ACC))
    d.add(String(91 * mm, 1.5 * mm, "init_states [3,6,32,36] + shock  →  10-day rollout in physical units (0.1–0.2 s on CPU)  →  live maps, charts and impact cards",
                 fontName="AI", fontSize=7.8, fillColor=MUT, textAnchor="middle"))
    return d


s = []
# ------------------------------------------------------------------ page 1: model, drift/blur, results, failure modes, architecture
s += [P("INDRA: a climate digital twin of India", h1),
      P("<b>Team Omnix</b> · Yugaantar DevForge · Track 2 “Multiverse of Monsoon” · github.com/channallikrishnasai/indra", sub), Spacer(1, 3),
      P("<b>USP.</b> A 0.53 M-parameter neural stepper that is <b>causal by construction</b>: it sees 7 grid cells per day, so a shock cannot travel faster "
        "than ~7°/day. It stays local on day 1 and spreads afterwards. A learned calibration keeps long rollouts sharp and on climate. "
        "Every forecast drives a live 3D/2D dashboard and a second ML model (INDRA-Risk) for flood and heat risk.")]
s += [P("1 · Model design", h2)] + B([
    "<b>Inputs:</b> last 3 days (6 channels; rain as log(1+rain)) given as the last state plus two daily tendencies, the target day’s "
    "climatology, land/sea mask, orography, day-of-year and lat/lon.",
    "<b>Network:</b> 3×3 stem → 3 residual blocks (two 3×3 convs, GroupNorm, GELU) → 1×1 head. It predicts the <i>tendency</i>, added to the last state.",
    "<b>Physics limits</b> on every output: rain ≥ 0, moisture ≥ 0, t2m within −40…+60 °C. An all-zero shock takes the exact no-shock path.",
    "<b>Training</b> (CPU, ~1.5 h): single-step, then 2 → 4 → 6-step autoregressive rollouts. Lat-weighted MSE. 2005–2020 train, <b>2021 held out</b>.",
])
s += [P("2 · How we prevent drift and blur", h2)] + B([
    "<b>Rollout training</b> teaches the model to fix its own errors: held-out day-8 error fell 0.590 → 0.496 (normalised).",
    "<b>Per-lead calibration:</b> each lead is clim + α<sub>k</sub>·(forecast − clim), with α fitted by least squares on training years. "
    "Rain is fitted in mm/day so heavy rain survives. The recursion itself is never damped.",
    "<b>Multiverse</b> = calib(F(x)) + [F(x + shock) − F(x)], so calibration never mutes a shock.",
])
s += [P("3 · Results on held-out 2021 (355 windows, organisers’ monsoon_r1.scoring)", h2),
      table([["Lead", "t2m", "tp", "u10", "v10", "msl", "tcwv", "Mean skill", "Heavy-rain CSI at 10 mm/day, the scored threshold (INDRA / persist / clim)"],
             ["Day 1", ".205", ".276", ".257", ".304", ".328", ".297", "<b>.278</b>", "<b>.53</b> / .43 / .26"],
             ["Day 4", ".141", ".035", ".109", ".099", ".105", ".174", "<b>.111</b>", "<b>.31</b> / .21 / .26"],
             ["Day 8", ".044", "-.042", ".014", ".013", ".027", ".040", "<b>.016</b>", "<b>.28</b> / .20 / .26"]],
            [13 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 19 * mm, 66 * mm]),
      Spacer(1, 1.5),
      P("Skill = 1 − RMSE / RMSE of the better of repeat-yesterday and climatology, averaged per window (evaluate.py). The pooled aggregation used by the official scorer gives 0.139; both clear the 0.12 reference. Positive on every channel and lead except rain at day 8, where it is −0.042: rain at day 8 is worse than the better reference. Day-8 variance ratio "
        "0.67–1.01 (judge band 0.40–2.5). Warm-Bay shock passes all four causality checks (99% of day-1 response near the box). The zero shock is "
        "bit-identical. Schema validator 6/6, 0.09–0.16 s per 10-day call. Estimated automatic score ≈ 82 / 85 (full 2021 set).", small)]
s += [P("4 · Failure modes and limits", h2)] + B([
    "<b>Heavy rain, scored threshold (10 mm/day):</b> CSI is 0.53 / 0.31 / 0.28 at days 1 / 4 / 8 against 0.20 for persistence and 0.26 for climatology at day 8. Day 8 is marginal, not failing: a 0.02 margin over climatology and 0.08 over persistence. The rain quantile map is half strength; full strength reaches 15 of 15 points, but worsens RMSE.",
    "<b>Heavy rain, IMD 64.5 mm/day (not scored):</b> CSI is 0.11 / 0.01 / 0.02 at days 1 / 4 / 8 on 2021, against persistence 0.09 / 0.00 / 0.00 and climatology 0 (climatology never reaches the threshold). Above day 1 the model is close to zero skill at this threshold.",
    "<b>Rolling-origin validation</b> (the recipe retrained, not the shipped model; each year scored by a model trained only on earlier years): 2019 (trained through 2018) skill mean 0.139, day-8 CSI 0.31; 2020 (trained through 2019) skill mean 0.137, day-8 CSI 0.29. 2021 (the shipped model, which used 2021 to pick some settings) skill mean 0.133, day-8 CSI 0.28. These are validation numbers, not hidden-test results.",
    "<b>Monsoon-stall shock spreads slowly and stays near its box:</b> on the 15 July 2021 start, response energy falls from 259 (day 1) to 117 (day 10), and the share inside the box falls from 0.62 to 0.29. It fails the propagation check.",
    "<b>Resolution:</b> one cell is about 110 km (1° grid). There is no street-level weather, and the model covers India only.",
    "<b>Alerts and impact indices are rule-based research thresholds</b> (IMD-style rain bands, a heat rule, a screening cyclone flag), not agency-tuned and not official warnings.",
    "<b>2021-based choices:</b> some settings (the rain-map strength, the calibration shrinkage) were looked at with 2021 in view, so the 2021 score is optimistic. The 2019 and 2020 validation numbers are the fairer check.",
    "<b>Day-8 rain</b> is sharper (blur ratio 0.67) but slightly less accurate than repeat-yesterday. <b>Domain-bound:</b> remote drivers (ENSO, the Atlantic) enter only via the grid state.",
])
s += [P("5 · System architecture", h2), arch()]
pass  # one flow, two pages

# ------------------------------------------------------------------ page 2: visualisation + impact
s += [P("6 · Visualisation: the dashboard (live from POST /simulate, nothing pre-rendered)", h2),
      Image(str(HERE / "img" / "dashboard_delta.jpg"), width=120 * mm, height=120 * mm * 1800 / 3200),
      P("Monsoon onset 2021-06-02, “Warm Bay of Bengal” shock (+3 K, +15% moisture), Δ shock-impact view at day +4: the anomaly stays on its box. "
        "Right: impact cards with the multiverse difference and the INDRA-Risk 10-day outlook.", cap)]
s += B([
    "<b>3D globe</b> lit by the real sun for the date (city lights at night, ocean glint, relief, atmosphere), with lit clouds, wind, rain and lightning from the simulation. "
    "<b>2D map</b> with borders drawn above the data.",
    "<b>Universes:</b> Prime, Multiverse, Δ, Observed (verification) and Ensemble σ. <b>Shocks:</b> presets or any variable/size/place (Shift+click). "
    "Plus a causality chart, live telemetry, light/dark theme, layouts and a guided tour.",
])
impact = [P("7 · Impact use case: what a shock means for people", h2),
          P("Each universe becomes decisions, computed live from the rollout and the model’s climatology. "
            "<b>Real results</b>, Monsoon-onset window (10-day totals, Prime → Multiverse):"),
          table([["Shock", "Heatwave exposure", "Crop water stress", "Rain vs normal", "Peak rain"],
                 ["Heat dome +4 K, NW", "<b>2.6 → 9.5 %</b>", "19.1 → 17.6 %", "−21 → −20 %", "21 → 21 mm"],
                 ["Warm Bay of Bengal", "2.6 → 2.5 %", "19.1 → 18.0 %", "<b>−21 → −18 %</b>", "<b>21 → 26 mm</b>"],
                 ["Monsoon stall −40% wind", "2.6 → 2.7 %", "19.1 → 18.4 %", "−21 → −20 %", "21 → 21 mm"]],
                [50 * mm, 34 * mm, 34 * mm, 32 * mm, 32 * mm]),
          Spacer(1, 2),
          P("Indicators: heatwave = % land with daily-mean t2m ≥ 35 °C or ≥ 4.5 K above normal; flood-risk area = land with IMD heavy rain ≥ 64.5 mm/day; "
            "crop stress = rain deficit vs normal × heat; plus wind power at 100 m and ensemble confidence. "
            "Users: health departments, NDMA/SDMA, kharif planning, grid operators.", small),
          P("<b>INDRA-Risk</b> (second ML model, a small CNN trained on INDRA’s own forecasts vs reality) turns each day into "
            "probabilities of heavy rain and heat stress. Held-out 2021 ROC-AUC: <b>0.90 heavy rain, 0.99 heat</b>.")]
s += impact
s += [P("8 · Integrity notes", h2)] + [P(t, small) for t in [
    "• The 366-day climatology in weights/indra.pt (and the rain climatology used for calibration) is an average of the training years 2005–2020, "
    "smoothed over 15 days, used as a model input. It is not stored weather.",
    "• The dashboard’s demo starting states are generated locally from the kit data at demo time and are not in the repo; the server never reads them.",
    "• The dashboard needs <i>npm install</i> once with internet, and its optional Street 3D view loads map tiles (OpenFreeMap, AWS terrain, Esri imagery) while running; everything else runs offline, and the grader runs only serve.py. No weather APIs anywhere.",
]]

s += [Spacer(1, 4), Image(str(HERE / "img" / "brag-poster.jpg"), width=120 * mm, height=120 * mm * 1080 / 1920),
      P("Launch video (20 s, 1920x1080): brag.mp4 in the repo, poster frame shown. Made with the Hyperframes pipeline; every number in it comes from the runs above.", small)]

out = HERE / "INDRA_brief.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=9 * mm, bottomMargin=8 * mm,
                  title="INDRA: climate digital twin of India", author="Team Omnix").build(s)
print("wrote", out)
