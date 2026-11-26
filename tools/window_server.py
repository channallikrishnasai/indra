"""Dashboard-only helper: serves 3 consecutive real days (plus the 10 observed days after them) for ANY date in the
kit data, so the dashboard can start a forecast from a date you pick. It is NOT the model server: serve.py never reads
weather data, and this file is not used by the grader. Needs the local, git-ignored data/india_climate_train.nc.

    python tools/window_server.py --data data/india_climate_train.nc --port 8100
GET /range                -> {"first": "YYYY-MM-DD", "last": "YYYY-MM-DD"}   (valid last-input days)
GET /window?date=YYYY-MM-DD -> {"date", "doy", "doy_sincos", "shape": [13,6,32,36], "b64": float32 little-endian}
"""
import argparse, base64, datetime as dt
import numpy as np
import uvicorn
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--port", type=int, default=8100)
ARGS = ap.parse_args()
ds = xr.open_dataset(ARGS.data)
DATA = ds["data"].values.astype("<f4")                                   # [N,6,32,36] physical units, loaded once
DAYS = ds["time"].values.astype("datetime64[D]")
FIRST, LAST = DAYS[2], DAYS[-11]                                          # need 2 days before and 10 observed days after

app = FastAPI(title="INDRA dashboard window server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/range")
def range_():
    return {"first": str(FIRST), "last": str(LAST)}


@app.get("/window")
def window(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    try:
        d = np.datetime64(dt.date.fromisoformat(date), "D")
    except ValueError:
        raise HTTPException(422, "date must be a real calendar date, YYYY-MM-DD")
    if not FIRST <= d <= LAST:
        raise HTTPException(404, f"date must be between {FIRST} and {LAST} (3 input days + 10 observed days must exist)")
    i = int(np.searchsorted(DAYS, d))
    if DAYS[i] != d or (DAYS[i - 2:i + 11] != d + np.arange(-2, 11)).any():
        raise HTTPException(404, "the data has a gap around this date")
    block = np.ascontiguousarray(DATA[i - 2:i + 11])                      # 3 inputs + 10 observed
    doy = (d - d.astype("datetime64[Y]")).astype(int) + 1                 # same convention as tools/make_windows.py
    return {"date": date, "doy": int(doy), "doy_sincos": [float(np.sin(2 * np.pi * doy / 365)), float(np.cos(2 * np.pi * doy / 365))],
            "shape": list(block.shape), "b64": base64.b64encode(block.tobytes()).decode("ascii")}


if __name__ == "__main__":
    print(f"window server: {FIRST} .. {LAST} on port {ARGS.port}")
    uvicorn.run(app, host="127.0.0.1", port=ARGS.port, log_level="warning")
