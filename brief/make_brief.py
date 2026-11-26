"""Builds brief/INDRA_brief.pdf (2 pages, A4).   python brief/make_brief.py"""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

F = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("A", str(F / "arial.ttf"))); pdfmetrics.registerFont(TTFont("AB", str(F / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("AI", str(F / "ariali.ttf")))
pdfmetrics.registerFontFamily("A", normal="A", bold="AB", italic="AI", boldItalic="AB")

INK, ACC, MUT, LINE = colors.HexColor("#0e1a2c"), colors.HexColor("#0a6fb8"), colors.HexColor("#5a6c86"), colors.HexColor("#c9d6e6")
body = ParagraphStyle("b", fontName="A", fontSize=8.6, leading=11.2, textColor=INK, spaceAfter=3)
small = ParagraphStyle("s", parent=body, fontSize=7.6, leading=9.6, textColor=MUT)
h1 = ParagraphStyle("h1", fontName="AB", fontSize=17, leading=20, textColor=INK)
sub = ParagraphStyle("sub", parent=body, textColor=MUT, fontSize=8.8)
h2 = ParagraphStyle("h2", fontName="AB", fontSize=9.6, leading=12, textColor=ACC, spaceBefore=5, spaceAfter=2)
cell = ParagraphStyle("c", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="AB")
P = lambda t, s=body: Paragraph(t, s)
B = lambda items: [P("• " + t) for t in items]


def table(rows, widths, head=True):
    t = Table([[P(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)], colWidths=widths)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, ACC), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def arch():
    d = Drawing(180 * mm, 30 * mm)
    boxes = [("Dashboard", "3D globe · 2D map", 0), ("serve.py", "POST /simulate · /risk", 37), ("INDRA stepper", "local CNN, 7 cells/day", 74),
             ("Calibration +", "full shock response", 111), ("Impact + INDRA-Risk", "heat · flood · crops", 148)]
    for t, s, x in boxes:
        d.add(Rect(x * mm, 8 * mm, 31 * mm, 15 * mm, rx=6, ry=6, fillColor=colors.HexColor("#eef5fc"), strokeColor=ACC, strokeWidth=0.8))
        d.add(String((x + 15.5) * mm, 17.5 * mm, t, fontName="AB", fontSize=8, fillColor=INK, textAnchor="middle"))
        d.add(String((x + 15.5) * mm, 12 * mm, s, fontName="A", fontSize=6.6, fillColor=MUT, textAnchor="middle"))
    for x in (31, 68, 105, 142):
        d.add(Line(x * mm, 15.5 * mm, (x + 5.4) * mm, 15.5 * mm, strokeColor=ACC, strokeWidth=1))
        d.add(Polygon([(x + 6) * mm, 15.5 * mm, (x + 4.6) * mm, 16.5 * mm, (x + 4.6) * mm, 14.5 * mm], fillColor=ACC, strokeColor=ACC))
    d.add(String(90 * mm, 2.5 * mm, "init_states [3,6,32,36] + shock  →  10-day rollout in physical units (0.1–0.2 s on CPU)  →  live maps, charts and impact cards",
                 fontName="AI", fontSize=7, fillColor=MUT, textAnchor="middle"))
    return d


s = []
# ------------------------------------------------------------------ page 1: model, drift/blur, failure modes
s += [P("INDRA: a climate digital twin of India", h1),
      P("Yugaantar DevForge · Track 2 “Multiverse of Monsoon” · github.com/channallikrishnasai/indra", sub), Spacer(1, 4),
      P("<b>USP.</b> A 0.53 M-parameter neural stepper that is <b>causal by construction</b>: its receptive field is 7 cells per day, so a shock "
        "physically cannot travel faster than ~7°/day. It stays local on day 1 and spreads afterwards. A learned calibration keeps long "
        "rollouts sharp and on climate. Every forecast drives a live 3D/2D dashboard and a second ML model (INDRA-Risk) for flood and heat risk.")]
s += [P("1 · Model design", h2)] + B([
    "<b>Inputs per step:</b> the last 3 days (6 channels, rain as log(1+rain), normalised), given as the last state plus two daily tendencies; "
    "the climatology of the target day; land/sea mask and orography; day-of-year (sin, cos); latitude/longitude.",
    "<b>Network:</b> 3×3 stem → 3 residual blocks (two 3×3 convolutions each, GroupNorm, GELU) → 1×1 head. Edge-replicate padding. "
    "It predicts the <i>tendency</i>, which is added to the last state.",
    "<b>Hard physics limits</b> on every output: rain ≥ 0, moisture ≥ 0, t2m within −40…+60 °C, bounded wind and pressure. "
    "Fully deterministic, so an all-zero shock returns the exact no-shock rollout.",
    "<b>Training</b> (CPU, ~1.5 h): 10 epochs single-step, then an autoregressive curriculum of 2 → 4 → 6-step rollouts. "
    "Lat-weighted MSE, rain weighted ×1.5. 2005–2020 for training; <b>2021 held out</b>.",
])
s += [P("2 · How we prevent drift and blur", h2)] + B([
    "<b>Rollout training</b> teaches the model to correct its own errors: held-out day-8 error fell from 0.590 to 0.496 (normalised) during the curriculum.",
    "<b>Per-lead calibration:</b> each reported lead is clim + α<sub>k</sub>·(forecast − clim), with α fitted by least squares on training years only. "
    "Unpredictable anomalies shrink smoothly with lead instead of drifting. The recursion itself is never damped.",
    "<b>Rain is calibrated in mm/day</b> against the arithmetic-mean rain climatology. Shrinking in log space would pull towards the geometric mean and erase heavy rain.",
    "<b>Multiverse = calibrated Prime + full dynamical response:</b> calib(F(x)) + [F(x + shock) − F(x)]. Calibration never mutes a shock.",
    "<b>Anti-blur:</b> the tendency + climatology design keeps spatial detail. Day-8 variance ratio is 0.43–1.01 across channels (the judge band is 0.40–2.5).",
])
s += [P("3 · Results on held-out 2021 (355 windows, organisers’ monsoon_r1.scoring)", h2),
      table([["Lead", "t2m", "tp", "u10", "v10", "msl", "tcwv", "Mean skill", "Heavy-rain CSI (INDRA / persist / clim)"],
             ["Day 1", ".206", ".265", ".255", ".302", ".325", ".296", "<b>.275</b>", "<b>.50</b> / .43 / .26"],
             ["Day 4", ".143", ".056", ".107", ".093", ".099", ".173", "<b>.112</b>", "<b>.26</b> / .21 / .26"],
             ["Day 8", ".050", ".019", ".018", ".014", ".024", ".039", "<b>.027</b>", "<b>.25</b> / .20 / .26"]],
            [13 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 12 * mm, 18 * mm, 60 * mm]),
      Spacer(1, 2),
      P("Skill = 1 − RMSE / RMSE of the better of repeat-yesterday and climatology (positive on every channel and lead). "
        "Shocks: warm Bay of Bengal passes all four checks (99% of day-1 response near the box). Monsoon stall is local and stable. "
        "Zero shock is bit-identical. Official schema validator: 6/6 pass, 0.09–0.16 s per 10-day call. "
        "Estimated automatic score ≈ 74 / 85 (RMSE 30/30, CSI 5.9/15, anti-blur 10/10, causality 13.1/15, repo 15/15).", small)]
s += [P("4 · Failure modes and limits (honest)", h2)] + B([
    "<b>Heavy-rain skill fades after day 1:</b> at days 4 and 8 INDRA beats persistence but only ties climatology. Calibration trades extreme-rain sharpness for RMSE.",
    "<b>Monsoon-stall shock does not spread</b> outward by days 4–8 on our test window: its response stays near the Arabian Sea box and fades there.",
    "<b>Day-8 rain detail</b> (variance ratio 0.43) sits just above the 0.40 blur floor. A wetter regime than 2021 could push it under.",
    "<b>Domain-bound:</b> the physics core sees only the 32 × 36 grid, so remote drivers (e.g. ENSO, the Atlantic) enter only through the boundary state.",
    "<b>Deterministic:</b> the ensemble uses initial-condition noise only, so its spread is a lower bound on uncertainty.",
])
s.append(PageBreak())

# ------------------------------------------------------------------ page 2: visualisation + impact
s += [P("5 · System architecture", h2), arch()]
s += [P("6 · Visualisation: the dashboard", h2),
      P("A Vite + three.js + d3-geo front end, separate from the server. Every pixel comes from a live <b>POST /simulate</b> call; nothing is pre-rendered.")]
s += B([
    "<b>3D globe</b> lit by the real sun for the simulated date: city lights on the night side, ocean sun glint, relief shading, a scattering atmosphere. "
    "INDRA’s fields are draped over the domain, with lit clouds and shadows derived from simulated rain and moisture, wind particles, falling rain and lightning in heavy-rain cells.",
    "<b>2D world map</b> (equirectangular, pan/zoom) with a translucent field, borders drawn above the data, city labels and cloud shadows.",
    "<b>Universes:</b> Timeline Prime, Multiverse, Δ shock impact, Observed (verification against what happened) and Ensemble σ. "
    "A smooth 10-day timeline interpolates between days.",
    "<b>Shock builder:</b> presets (the two judge shocks, heat dome, dry Western Ghats, cyclone seed) or any variable/magnitude/radius. "
    "Shift+click moves the shock; shocks outside the domain are refused with an explanation.",
    "<b>Causality chart:</b> response energy and mean spread distance per day, plus the share of day-1 energy near the shock.",
    "<b>Telemetry:</b> inference time, NaN check, rain ≥ 0, temperature range and a live zero-shock determinism check.",
    "<b>Presentation:</b> light/dark theme with a circular reveal, Full / Focus / Cinema layouts, a guided tour of 9 historic events, and PNG snapshots.",
])
s += [P("7 · Impact use case: what a shock means for people", h2),
      P("The impact module turns each simulated universe into decisions, computed live from the rollout and the model’s own climatology. "
        "Every card shows Prime and the Multiverse difference side by side:")]
s.append(table([["Indicator", "How it is computed", "Who uses it"],
                ["Heatwave exposure", "% of land with daily-mean t2m ≥ 35 °C, or ≥ 4.5 K above normal while ≥ 30 °C (IMD-style departure)", "Health departments, labour advisories"],
                ["Flood-risk area", "Land area (km²) with any day of IMD heavy rain ≥ 64.5 mm/day", "Disaster management (NDMA/SDMA)"],
                ["Crop water stress", "10-day rain deficit vs normal, amplified by heat (evaporative demand), where crops expect rain", "Kharif sowing and irrigation planning"],
                ["Rain vs normal", "Area-weighted 10-day rain departure, IMD-style %", "Agriculture, reservoirs"],
                ["Wind power", "Mean power density at 100 m hub (1/7 power law from 10 m)", "Grid operators"],
                ["Ensemble confidence", "Spread of 5 perturbed members at day 10", "Anyone acting on the forecast"]],
               [30 * mm, 98 * mm, 52 * mm]))
s += [Spacer(1, 3), P("<b>INDRA-Risk (second ML model).</b> A small CNN post-processor (model-output statistics) that turns INDRA’s forecast "
                     "(state, anomaly, terrain, lead, season) into calibrated probabilities of IMD heavy rain and heat stress. It is trained on INDRA’s own "
                     "forecasts versus what really happened, so it knows INDRA’s error grows with lead. Held-out 2021 ROC-AUC: <b>heavy rain 0.90, heat stress 0.99</b>. "
                     "It drives two map layers and a 10-day risk outlook strip.")]
s += [P("<b>Live demo flow:</b> pick an event (e.g. monsoon onset), apply a shock such as +3 K over the Bay of Bengal, and run the multiverse. "
        "The Δ view shows where the anomaly sits on day 1 and where it has moved by day 10. Each impact card states how the multiverse differs from Prime, "
        "and the propagation chart shows the spread distance day by day. Judges can apply any new shock on the spot; results come back in well under a second.")]
s += [P("8 · Integrity notes", h2)] + [P(t, small) for t in [
    "• The 366-day climatology inside weights/indra.pt (and the rain climatology used for calibration) is an average of the training years "
    "2005–2020, smoothed over 15 days, and used as a model input. It is not stored weather.",
    "• The dashboard’s demo starting states are generated locally from the kit data at demo time (tools/make_windows.py) and are not shipped in the repo. "
    "The server never reads them.",
    "• The dashboard needs <i>npm install</i> once with internet, and its web font falls back to system fonts offline. Neither affects the offline grader, which runs only serve.py.",
    "• No external weather APIs anywhere. Inference uses only the request and the bundled weights.",
]]

out = Path(__file__).with_name("INDRA_brief.pdf")
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=11 * mm,
                  title="INDRA: climate digital twin of India", author="channallikrishnasai").build(s)
print("wrote", out)
