"""Export one year's rolling-origin validation as JSON for the dashboard's validation panel.

For every 10-day window that starts in YEAR (every N-th), run the model that was trained on years before YEAR
(passed as --weights), and record: RMSE skill vs the better of persistence / climatology (days 1, 4, 8), heavy-rain
CSI (>= 10 mm/day, as in the judge), and the land-mean rain and temperature series at lead 1 (forecast vs observed).

    PYTHONPATH=path/to/kit python rolling_export.py --year 2020 --weights weights/ro2020.pt --out dashboard/public/validation/2020.json

Same metric code as evaluate.py (monsoon_r1.scoring), so the numbers are comparable.
"""
import argparse, json
import numpy as np
import torch
import xarray as xr
from monsoon_r1 import scoring as S

from indra_model.net import load_indra, NC, NLAT, NLON, LAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_climate_train.nc"); ap.add_argument("--weights", required=True)
    ap.add_argument("--year", type=int, required=True); ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    torch.set_num_threads(4)
    ds = xr.open_dataset(a.data); data = ds["data"].values.astype("f4"); t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970; doy = (t - t.astype("datetime64[Y]")).astype(int)
    train = years < a.year
    clim = np.zeros((366, NC, NLAT, NLON)); cnt = np.zeros(366)
    np.add.at(clim, doy[train], data[train]); np.add.at(cnt, doy[train], 1); clim /= np.maximum(cnt, 1)[:, None, None, None]
    clim = np.mean([np.roll(clim, o, 0) for o in range(-7, 8)], axis=0)
    model = load_indra(a.weights)
    idx = [i for i in np.where(years == a.year)[0] if i >= 2 and i + 10 < len(t) and years[i + 10] == a.year][::a.every]
    leads = [1, 4, 8]; rm = {L: [] for L in leads}; rp = {L: [] for L in leads}; rc = {L: [] for L in leads}
    hits = {k: {L: [[], []] for L in leads} for k in ("m", "p", "c")}; series = []
    for i in idx:
        out = model.simulate(data[i - 2:i + 1], float(doy[i]), 10)   # 0-based doy of the LAST input day
        for L in leads:
            tr = data[i + L]; c = clim[(doy[i] + L) % 366]
            rm[L].append(S.rmse_lat(out[L - 1], tr)); rp[L].append(S.rmse_lat(data[i], tr)); rc[L].append(S.rmse_lat(c, tr))
            for k, p in (("m", out[L - 1]), ("p", data[i]), ("c", c)):
                hits[k][L][0].append(p[1]); hits[k][L][1].append(tr[1])
        series.append({"date": str(t[i + 1]), "pred_rain": float(out[0][1].mean()), "obs_rain": float(data[i + 1][1].mean()),
                       "pred_t": float(out[0][0].mean() - 273.15), "obs_t": float(data[i + 1][0].mean() - 273.15)})
    res = {"year": a.year, "windows": len(idx), "leads": {}}
    for L in leads:
        m, p, c = np.mean(rm[L], 0), np.mean(rp[L], 0), np.mean(rc[L], 0)
        ss = float(np.nanmean(1 - m / np.minimum(p, c)))
        cs = {k: S.csi(np.stack(hits[k][L][0]), np.stack(hits[k][L][1])) for k in hits}
        res["leads"][f"d{L}"] = {"skill": ss, "csi": cs["m"], "csi_persist": cs["p"], "csi_clim": cs["c"]}
    res["skill_mean"] = float(np.mean([res["leads"][f"d{L}"]["skill"] for L in leads]))
    res["series"] = series
    with open(a.out, "w", encoding="utf-8") as f: json.dump(res, f)
    print(f"{a.year}: {len(idx)} windows, skill mean {res['skill_mean']:.3f}, d8 CSI {res['leads']['d8']['csi']:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
