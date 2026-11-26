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
├── tools/make_windows.py builds the dashboard's demo initial states
└── dashboard/            Vite + three.js + d3-geo front end (separate from the server)
```

## Run (offline, CPU)

```bash
pip install torch numpy fastapi uvicorn          # all available in the judge image
python serve.py --port 8000                      # listens on 0.0.0.0, ready in a few seconds
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

## Dashboard

```bash
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
