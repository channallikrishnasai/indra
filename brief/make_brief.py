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
body = ParagraphStyle("b", fontName="A", fontSize=10, leading=12.6, textColor=INK, spaceAfter=2.5)
small = ParagraphStyle("s", parent=body, fontSize=8.2, leading=10.2, textColor=MUT, spaceAfter=1.5)
h1 = ParagraphStyle("h1", fontName="AB", fontSize=17, leading=20, textColor=INK)
sub = ParagraphStyle("sub", parent=body, textColor=MUT, fontSize=9.2, leading=11.5)
h2 = ParagraphStyle("h2", fontName="AB", fontSize=11, leading=13, textColor=ACC, spaceBefore=5, spaceAfter=2)
cell = ParagraphStyle("c", parent=body, fontSize=8.8, leading=10.6, spaceAfter=0)
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
      table([["Lead", "t2m", "tp", "u10", "v10", "msl", "tcwv", "Mean skill", "CSI ≥ 10 mm/day (INDRA / persist / clim)"],
             ["Day 1", ".206", ".265", ".255", ".302", ".325", ".296", "<b>.275</b>", "<b>.50</b> / .43 / .26"],
             ["Day 4", ".143", ".056", ".107", ".093", ".099", ".173", "<b>.112</b>", "<b>.26</b> / .21 / .26"],
             ["Day 8", ".050", ".019", ".018", ".014", ".024", ".039", "<b>.027</b>", "<b>.25</b> / .20 / .26"]],
            [13 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 19 * mm, 66 * mm]),
      Spacer(1, 1.5),
      P("Skill = 1 − RMSE / RMSE of the better of repeat-yesterday and climatology: positive on every channel and lead. Day-8 variance ratio "
        "0.43–1.01 (judge band 0.40–2.5). Warm-Bay shock passes all four causality checks (99% of day-1 response near the box). The zero shock is "
        "bit-identical. Schema validator 6/6, 0.09–0.16 s per 10-day call. Estimated automatic score ≈ 74 / 85.", small)]
s += [P("4 · Failure modes and limits", h2)] + B([
    "<b>Heavy rain fades after day 1:</b> days 4 and 8 beat persistence but only tie climatology. Keeping more rain detail scored worse in our tests.",
    "<b>Monsoon-stall shock does not spread</b> by days 4–8 on our test window: its response stays near the Arabian Sea box and fades.",
    "<b>Day-8 rain detail</b> (0.43) sits just above the 0.40 blur floor. <b>Domain-bound:</b> remote drivers (ENSO, the Atlantic) enter only via the grid state.",
])
s += [P("5 · System architecture", h2), arch()]
s.append(PageBreak())

# ------------------------------------------------------------------ page 2: visualisation + impact
s += [P("6 · Visualisation: the dashboard (live from POST /simulate, nothing pre-rendered)", h2),
      Image(str(HERE / "img" / "dashboard_delta.jpg"), width=182 * mm, height=182 * mm * 1800 / 3200),
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

out = HERE / "INDRA_brief.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm, bottomMargin=10 * mm,
                  title="INDRA: climate digital twin of India", author="Team Omnix").build(s)
print("wrote", out)
