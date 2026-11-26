"""INDRA inference server (offline).   python serve.py --port 8000

POST /health    -> {"status": "ready", "model_name": ...}
POST /simulate  -> contract endpoint: {"init_states":[3,6,32,36], "doy_sincos":[s,c], "steps":n, "perturbation":[6,32,36]|null}
POST /climatology -> {"doy_sincos":[s,c], "steps":n} -> model climatology [n,6,32,36] (physical) for the dashboard's impact module
POST /static    -> land/sea mask + orography [2,32,36] for the dashboard
"""
import argparse, base64, json, time
from pathlib import Path
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from indra_model.net import load_indra, NC, NLAT, NLON, doy_from_sincos, to_physical

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)

_ap = argparse.ArgumentParser(); _ap.add_argument("--port", type=int, default=8000)
_ap.add_argument("--weights", default=str(ROOT / "weights" / "indra.pt"))
ARGS, _ = _ap.parse_known_args()
MODEL = load_indra(ARGS.weights)
MODEL.simulate(np.full((3, NC, NLAT, NLON), 1.0, "f4") * np.array([295, 2, 1, 1, 101000, 30], "f4")[None, :, None, None], 0.0, 2)  # warm-up
MODEL_NAME = "INDRA-v1 (residual CNN)"

# INDRA-Risk (optional, dashboard only): learned heavy-rain / heat-stress probabilities on top of a rollout
RISK = None
if (ROOT / "weights" / "risk.pt").exists():
    from indra_model.risk import RiskNet, predict_risk
    RISK = RiskNet(); RISK.load_state_dict(torch.load(ROOT / "weights" / "risk.pt", map_location="cpu", weights_only=False)["state"]); RISK.eval()

app = FastAPI(title="INDRA climate digital twin")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _arr(x, shape):
    if isinstance(x, dict):
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if a.shape != shape:
        raise HTTPException(422, f"expected shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise HTTPException(422, "non-finite values")
    return a


def simulate(init_states, doy_sincos, steps, perturbation):
    """The perturbation is a physical delta on init_states[-1]; INDRA adds it (clamping rain/moisture at 0) internally."""
    out = MODEL.simulate(init_states, doy_from_sincos(*doy_sincos), steps, perturbation)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter(); body = await request.json()
    init = _arr(body["init_states"], (3, NC, NLAT, NLON))
    pert = None if body.get("perturbation") is None else _arr(body["perturbation"], (NC, NLAT, NLON))
    steps = int(body.get("steps", 10))
    if not 1 <= steps <= 60:
        raise HTTPException(422, "steps must be 1..60")
    out = simulate(init, [float(v) for v in body["doy_sincos"]], steps, pert)
    # json.dumps directly: FastAPI's default encoder walks all 69k floats in Python and costs seconds
    rollout = json.dumps(np.round(out.astype("f8"), 4).tolist())
    ms = (time.perf_counter() - t0) * 1000.0
    return Response(content='{"rollout": ' + rollout + ', "inference_ms": ' + f"{ms:.2f}" + "}", media_type="application/json")


@app.post("/climatology")
async def climatology_endpoint(request: Request):
    """{"doy_sincos": [s, c], "steps": n} -> the model's climatology for forecast days 1..n, [n,6,32,36] physical."""
    body = await request.json(); d = doy_from_sincos(*[float(v) for v in body["doy_sincos"]])
    steps = max(1, min(60, int(body.get("steps", 10))))
    days = [int(round(d + k)) % 366 for k in range(1, steps + 1)]
    c = to_physical(MODEL.clim[days].numpy())
    if MODEL.clim_tp is not None:     # the log-space clim gives a geometric-mean rain (too low); use the true mean
        c[:, 1] = MODEL.clim_tp[days].numpy()
    return Response(content=json.dumps({"clim": np.round(c.astype("f8"), 3).tolist()}), media_type="application/json")


@app.post("/risk")
async def risk_endpoint(request: Request):
    """{"rollout": [steps,6,32,36] physical, "doy_sincos": [s,c]} -> {"risk": [steps,2,32,36]} probabilities."""
    if RISK is None:
        raise HTTPException(404, "risk model not trained")
    body = await request.json(); r = np.asarray(body["rollout"], dtype="f4")
    if r.ndim != 4 or r.shape[1:] != (NC, NLAT, NLON) or not np.isfinite(r).all():
        raise HTTPException(422, "rollout must be finite [steps,6,32,36]")
    p = predict_risk(MODEL, RISK, r, doy_from_sincos(*[float(v) for v in body["doy_sincos"]]))
    return Response(content=json.dumps({"risk": np.round(p.astype("f8"), 4).tolist()}), media_type="application/json")


@app.post("/static")
def static_endpoint():
    s = MODEL.static[0, 0].numpy()                      # normalised land-sea mask; midpoint of its range = 50% land
    land = (s > (s.min() + s.max()) / 2).astype("f4")
    return {"land": land.tolist(), "lat": [6.5 + i for i in range(NLAT)], "lon": [65.5 + j for j in range(NLON)]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="warning")
