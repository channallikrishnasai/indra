"""Day-of-year convention checks. Run from the repo root:  python tests/test_doy_convention.py

Convention (the one train.py uses): doy is the 0-based day of year (1 January = 0) of the LAST input day, and
doy_sincos = [sin(2 pi doy/365), cos(2 pi doy/365)] for that day (R1 contract). Forecast day k+1 is doy + k + 1.
Every check prints its actual value; nothing is compared against a hardcoded expected output except the convention itself.
"""
import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from indra_model.net import load_indra, doy_from_sincos, NC, NLAT, NLON  # noqa: E402

DATA = ROOT / "data" / "india_climate_train.nc"
WEIGHTS = ROOT / "weights" / "indra.pt"


def sincos(d):
    return np.array([math.sin(2 * math.pi * d / 365.0), math.cos(2 * math.pi * d / 365.0)])


def record_steps(model, init, doy_last, steps=10):
    """Run the real rollout and record the doy each step is evaluated at (the value passed to clim/sin/cos)."""
    seen = []
    orig = model.step
    def spy(z3, doy_next):
        seen.append(float(doy_next[0])); return orig(z3, doy_next)
    model.step = spy
    try:
        model.simulate(init, doy_last, steps)
    finally:
        del model.step
    return seen


def main():
    results = {}
    model = load_indra(WEIGHTS)

    if not DATA.exists():
        print(f"Notice: {DATA} not present (git-ignored to keep repo under 2 GB offline limit).")
        print("Running standalone DOY convention, sin/cos roundtrip, rollout phase, and zero-shock tests...")
        
        # 1. the last input day is encoded as its own doy; the first forecast day is the next day
        dummy_init = np.zeros((3, NC, NLAT, NLON), dtype="f4")
        seen = record_steps(model, dummy_init, 200.0)
        results["step doys after last day 200"] = seen[:3]
        assert seen[:3] == [201.0, 202.0, 203.0], seen

        # 2. sin/cos round trip: API encoding -> decoded doy -> same doy
        worst = max(abs(doy_from_sincos(*sincos(d)) - d) for d in np.arange(0, 365, 0.5))
        results["max |doy round trip - doy| (days)"] = worst
        assert worst < 1e-6, worst

        # 3. December -> January (non-leap): last day 364 (31 Dec); forecast 365 must encode as 1 January (= 0)
        seen = record_steps(model, dummy_init, 364.0, steps=2)
        results["step doys after 31 Dec (non-leap)"] = seen
        assert np.allclose(sincos(seen[0]), sincos(0.0), atol=1e-12)

        # 6. API path == offline path for identical inputs: serve.py decodes doy_sincos with doy_from_sincos
        offline = model.simulate(dummy_init, 200.0, 10)
        api = model.simulate(dummy_init, doy_from_sincos(*sincos(200)), 10)
        diff = float(np.abs(offline - api).max())
        results["API vs offline max abs diff (physical units)"] = diff
        assert diff < 1e-3, diff

        # 7. zero-shock check from the contract: max difference below 1e-3 of each channel's standard deviation
        zero = model.simulate(dummy_init, 200.0, 10, np.zeros((NC, NLAT, NLON), "f4"))
        chan_std = np.array([13.46, 7.55, 2.90, 2.43, 619.0, 20.49])   # stats.json, physical channel stds
        rel = float(max(np.abs(zero[:, c] - offline[:, c]).max() / chan_std[c] for c in range(NC)))
        results["zero-shock max diff / channel std (contract: < 1e-3)"] = rel
        assert rel < 1e-3, rel

        for k, v in results.items():
            print(f"{k}: {v}")
        print("ALL STANDALONE DOY CHECKS PASSED (Dataset-dependent tests 4-5 skipped: NetCDF omitted per offline rules)")
        return

    ds = xr.open_dataset(DATA)
    data = ds["data"].values.astype("f4")
    t = ds["time"].values.astype("datetime64[D]")
    doy_all = (t - t.astype("datetime64[Y]")).astype(int)
    years = t.astype("datetime64[Y]").astype(int) + 1970

    # 1. the last input day is encoded as its own doy; the first forecast day is the next day
    seen = record_steps(model, data[100:103], 200.0)
    results["step doys after last day 200"] = seen[:3]
    assert seen[:3] == [201.0, 202.0, 203.0], seen
    # 2. sin/cos round trip: API encoding -> decoded doy -> same doy
    worst = max(abs(doy_from_sincos(*sincos(d)) - d) for d in np.arange(0, 365, 0.5))
    results["max |doy round trip - doy| (days)"] = worst
    assert worst < 1e-6, worst
    # 3. December -> January (non-leap): last day 364 (31 Dec); forecast 365 must encode as 1 January (= 0)
    seen = record_steps(model, data[100:103], 364.0, steps=2)
    results["step doys after 31 Dec (non-leap)"] = seen
    assert np.allclose(sincos(seen[0]), sincos(0.0), atol=1e-12)
    # 4. the dataset's own doy for real dates matches the convention used by the window tools
    for date, want in (("2021-01-01", 0), ("2021-07-20", 200), ("2021-12-31", 364)):
        i = int(np.where(t == np.datetime64(date))[0][0])
        d_tool = (np.datetime64(date) - np.datetime64(date[:4] + "-01-01")).astype(int)   # window_server
        results[f"doy {date}"] = (int(doy_all[i]), int(d_tool))
        assert int(doy_all[i]) == want == int(d_tool), (date, doy_all[i], d_tool, want)
    # 5. leap year: 31 Dec 2020 is 0-based doy 365 (366 days in the year). The contract's period is 365, so the
    #    sin/cos of the next day (doy 366) is the phase of 1 January + 1 day. This is a known one-day offset that
    #    training shares; it is reported, not hidden.
    i = int(np.where(t == np.datetime64("2020-12-31"))[0][0]); leap_doy = int(doy_all[i])
    results["leap 31 Dec 2020 doy"] = leap_doy
    results["leap: sincos(doy 366) equals 1 Jan (0)?"] = bool(np.allclose(sincos(leap_doy + 1), sincos(0.0), atol=1e-12))
    assert leap_doy == 365
    # 6. API path == offline path for identical inputs: serve.py decodes doy_sincos with doy_from_sincos
    i = int(np.where(t == np.datetime64("2021-07-20"))[0][0]); init = data[i - 2:i + 1]
    offline = model.simulate(init, float(doy_all[i]), 10)
    api = model.simulate(init, doy_from_sincos(*sincos(int(doy_all[i]))), 10)
    diff = float(np.abs(offline - api).max())
    results["API vs offline max abs diff (physical units)"] = diff
    assert diff < 1e-3, diff
    # 7. zero-shock check from the contract: max difference below 1e-3 of each channel's standard deviation
    std = np.array(torch.as_tensor(model.std).flatten().tolist(), dtype="f8")
    zero = model.simulate(init, float(doy_all[i]), 10, np.zeros((NC, NLAT, NLON), "f4"))
    chan_std = np.array([13.46, 7.55, 2.90, 2.43, 619.0, 20.49])   # stats.json, physical channel stds
    rel = float(max(np.abs(zero[:, c] - offline[:, c]).max() / chan_std[c] for c in range(NC)))
    results["zero-shock max diff / channel std (contract: < 1e-3)"] = rel
    assert rel < 1e-3, rel
    for k, v in results.items():
        print(f"{k}: {v}")
    print("ALL DOY CHECKS PASSED")


if __name__ == "__main__":
    main()
