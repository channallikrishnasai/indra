# INDRA: a climate digital twin of India

*Multiverse of Monsoon, Yugaantar DevForge, Track 2*

INDRA rolls India's daily weather forward from the last 3 days (Timeline Prime). It also branches alternate universes from a local shock, such as a warmer Bay of Bengal or a stalled monsoon wind (Multiverse). A live 3D globe and 2D world-map dashboard drives everything, with an impact module for heatwave, flood, crop stress, wind energy and ensemble uncertainty.

```
repo/
├── serve.py              inference server (contract: POST /health, POST /simulate)
├── indra_model/net.py    the model: local residual neural stepper + physical constraints
├── indra_model/risk.py   INDRA-Risk: learned heavy-rain / heat-stress probabilities on top of a forecast
├── train.py              training pipeline (single-step, then autoregressive rollout curriculum)
├── calibrate.py          per-lead anomaly calibration (least-squares shrinkage towards climate)
├── train_risk.py         trains INDRA-Risk on INDRA's own calibrated forecasts vs observed outcomes
├── evaluate.py           offline scoring with the organisers' monsoon_r1.scoring
├── weights/indra.pt      trained weights (+ normalisation, static fields, climatology, calibration)
├── weights/risk.pt       INDRA-Risk weights (optional: /simulate never depends on it)
├── tools/make_windows.py builds the dashboard's demo initial states locally (git-ignored output)
└── dashboard/            Vite + three.js + d3-geo front end (separate from the server)
```

## Run (offline, CPU)

```bash
pip install torch numpy fastapi uvicorn          # all available in the judge image
python serve.py --port 8000                      # listens on 0.0.0.0, ready in a few seconds
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

## Dashboard

The dashboard's demo starting states are **not shipped** in this repo (no archive of past weather). Generate them from the kit data once, before the demo. The server never reads them:

```bash
python tools/make_windows.py --data data/india_climate_train.nc   # writes dashboard/public/windows.* (git-ignored)
cd dashboard
npm install
npm run dev          # http://localhost:5173 (server expected on :8000)
# another port:      http://localhost:5173/?api=http://127.0.0.1:8765
```

* **Realistic 3D globe:** sunlight for the simulated date, city lights on the night side, ocean sun glint, relief shading, a scattering atmosphere, lit clouds that cast shadows, animated wind, falling rain where INDRA forecasts it, and lightning in heavy-rain cells.
* **Light / dark theme** with a circular-reveal transition (`d`). **Layouts:** Full, Focus and Cinema (`f` / `c`). **Guided tour** of every event (`t`). **Snapshot** to PNG (`s`). Playback speed. Press `?` for all shortcuts.
* **ML risk layers** (Flood risk, Heat risk) and a 10-day **ML risk outlook** strip from INDRA-Risk (`POST /risk`).
* **3D Globe / 2D Map** toggle (keys `g` / `m`). The whole planet is shown with a NASA Blue Marble texture, country borders and an atmosphere. INDRA's live fields are draped over its domain (6–38°N, 65–101°E).
* **Layers:** temperature, rainfall, wind, pressure and moisture, plus animated wind particles and a cloud layer derived from simulated rain and moisture.
* **Universes:** Timeline Prime, Multiverse, Δ Shock impact, Observed (verification against what really happened), and Ensemble σ.
* **Shocks:** presets (the two judge shocks, heat dome, dry Western Ghats, cyclone seed) or a custom shock of any variable, magnitude and radius. Shift+click on the globe or map moves the shock centre. A shock outside the simulated domain is refused with an explanation.
* **Impact module:** computed in the browser from the live `/simulate` output and the model's climatology:
  heatwave exposure, flood-risk area (IMD heavy rain ≥ 64.5 mm/day), crop water stress, rain departure from normal, wind power at 100 m, peak rain, and 5-member ensemble confidence.
* **Shock propagation chart:** response energy and mean spread distance per day, plus the share of day-1 energy near the shock.
* **Engine telemetry:** inference time, NaN check, rain ≥ 0, temperature range, and a live zero-shock determinism check.

## Results (held-out 2021, never used for training or calibration)

Scored with the organisers' own `monsoon_r1.scoring` on all 355 ten-day windows of 2021 (`python evaluate.py --year 2021 --every 1`).

**Forecast skill** = 1 − RMSE_INDRA / RMSE of the better baseline (repeat-yesterday or climatology), lat-weighted. Positive means INDRA beats both.

| Lead | t2m | tp | u10 | v10 | msl | tcwv | **Mean** | vs repeat-yesterday | vs climatology |
|---|---|---|---|---|---|---|---|---|---|
| Day 1 | 0.206 | 0.265 | 0.255 | 0.302 | 0.325 | 0.296 | **0.275** | 0.275 | 0.491 |
| Day 4 | 0.143 | 0.056 | 0.107 | 0.093 | 0.099 | 0.173 | **0.112** | 0.280 | 0.112 |
| Day 8 | 0.050 | 0.019 | 0.018 | 0.014 | 0.024 | 0.039 | **0.027** | 0.277 | 0.027 |

INDRA beats both baselines on every channel at every lead. The mean skill over days 1/4/8 is 0.138, above the 0.12 needed for full RMSE points.

**Heavy-rain hits** (CSI at 10 mm/day; 1 = perfect, 0 = no hits):

| Lead | INDRA | Repeat-yesterday | Climatology |
|---|---|---|---|
| Day 1 | **0.50** | 0.43 | 0.26 |
| Day 4 | **0.26** | 0.21 | 0.26 |
| Day 8 | **0.25** | 0.20 | 0.26 |

Clear skill at day 1. At days 4 and 8 INDRA beats repeat-yesterday but only ties climatology.

**No blur** (day-8 spatial variance ÷ real variance; the judge wants 0.40–2.5):
t2m 1.01 · tp 0.43 · u10 0.74 · v10 0.76 · msl 0.88 · tcwv 0.90. All pass; rain is closest to the 0.40 floor.

**Shocks** (2021-07-15 window, judge definitions):

| Shock | Responds | Local on day 1 (≥ 60% near box) | Spreads by day 4–8 | Stable (< 5 σ) |
|---|---|---|---|---|
| Warm Bay of Bengal (+3 K, +15% moisture) | yes | yes (99%) | yes | yes |
| Monsoon stall (−40% Arabian Sea wind) | yes | yes (99%) | **no** | yes |
| Zero shock | output bit-identical to no shock | | | |

**Official schema validator** (`python -m monsoon_r1.schema_validator`): all 6 checks pass (JSON lists and base64; no shock, warm ocean, zeros). Each 10-day call takes 0.09–0.16 s on CPU against the 5 s limit.

**Repo scan** (`monsoon_r1.repo_scan` on the pushed repo): 15 / 15, no flags.

**INDRA-Risk** (held-out 2021 ROC-AUC, where 0.5 = guessing and 1 = perfect): heavy rain ≥ 64.5 mm/day **0.90**, heat stress **0.99**.

**Estimated automatic score:** forecast 35.9 / 45 (RMSE 30.0, heavy rain 5.9) · anti-blur 10 / 10 · causality 13.1 / 15 · repo 15 / 15 → **≈ 74 / 85** before the brief. These numbers come from 2021; the judge uses unseen 2023–24 windows.

**Known weaknesses:** heavy-rain skill fades after day 1. The monsoon-stall response stays near the Arabian Sea instead of spreading outward by days 4–8. Rain detail at day 8 (0.43) is just above the 0.40 blur floor.

## Model design (summary)

* **Input:** 3 days × 6 channels (rain in log1p space, normalised), as the last state plus two tendencies. Also the climatology of the target day, land/sea mask, orography, day of year and lat/lon.
* **Network:** a 3×3 stem, 3 residual blocks and a 1×1 head (0.53 M params). The receptive field is a **7-cell radius per day**, so a shock cannot move faster than ~7°/day. It stays local on day 1 and spreads outward afterwards (causality by construction).
* **Output:** the tendency added to the last state, then hard physical limits: rain ≥ 0, moisture ≥ 0, t2m within −40…+60 °C, bounded winds and pressure.
* **Training:** single-step teacher forcing, then an autoregressive rollout curriculum (2 → 4 → 6 steps) with latitude-weighted MSE. The model learns to correct its own drift. Year 2021 is held out.
* **Calibration:** each reported lead is `clim + α_k (forecast − clim)`, with α fitted by least squares on training years. Rain is fitted in mm/day against the arithmetic-mean climatology so heavy rain survives. Long leads relax to climate instead of drifting or blurring.
* **Multiverse:** `calibrated(F(x)) + [F(x + shock) − F(x)]`. Calibration never mutes a shock's dynamical response, and a zero shock is bit-identical.
* **INDRA-Risk (second ML model):** a small CNN that post-processes INDRA's forecast (state, anomaly vs climate, terrain, lead time, season) into calibrated probabilities of IMD heavy rain (≥ 64.5 mm/day) and heat stress. It is trained on INDRA's own forecasts vs reality (MOS), Platt-calibrated, and scored on held-out 2021 with AUC and Brier skill.

## Train

```bash
# data from the kit release: india_climate_train.nc + stats.json into data/
python train.py --data data/india_climate_train.nc --out weights/indra.pt --bs 64 --lr 3e-3 --epochs1 10 --rollout 2:2,4:2,6:1
python calibrate.py --weights weights/indra.pt          # fit on 2015-2020 (training years only)
python train_risk.py --indra weights/indra.pt --out weights/risk.pt
python tools/make_windows.py --data data/india_climate_train.nc
PYTHONPATH=path/to/yug-monsoon-kit python evaluate.py --year 2021
```

Training takes about an hour on an 8-thread CPU.

## Limits (honest)

* The physics core covers India's 32 × 36 grid only. The globe shows the whole world for context, but shocks far away (e.g. the Atlantic) cannot be simulated, and the UI says so.
* The heatwave index is a daily-mean proxy because the data has no daily maximum temperature.
* The ensemble uses initial-condition noise on a deterministic model, so the spread is a lower bound.
