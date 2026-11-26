"""Score INDRA on held-out data with the organisers' own scoring code (monsoon_r1.scoring), no server needed.

    PYTHONPATH=path/to/yug-monsoon-kit python evaluate.py --data data/india_climate_train.nc --year 2021

Reports RMSE skill vs the better of persistence / climatology at days 1, 4, 8, heavy-rain CSI skill,
day-8 variance ratio (anti-blur) and the two shock causality tests + the null shock.
"""
import argparse, json
import numpy as np
import torch
import xarray as xr
from monsoon_r1 import scoring as S

from indra_model.net import load_indra, NC, NLAT, NLON


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--weights", default="weights/indra.pt"); ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--every", type=int, default=7); a = ap.parse_args()
    torch.set_num_threads(4)
    ds = xr.open_dataset(a.data); data = ds["data"].values.astype("f4"); t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970; doy = (t - t.astype("datetime64[Y]")).astype(int)
    train = years < a.year
    # physical climatology from training years only (what "climatology" means for the judge)
    clim = np.zeros((366, NC, NLAT, NLON)); cnt = np.zeros(366)
    np.add.at(clim, doy[train], data[train]); np.add.at(cnt, doy[train], 1); clim /= np.maximum(cnt, 1)[:, None, None, None]
    clim = np.mean([np.roll(clim, o, 0) for o in range(-7, 8)], axis=0)
    std = np.array(json.load(open("data/stats.json"))["std"], "f4")
    model = load_indra(a.weights)

    idx = [i for i in np.where(years == a.year)[0] if i >= 2 and i + 10 < len(t) and years[i + 10] == a.year][::a.every]
    leads = [1, 4, 8]; rm = {L: [] for L in leads}; rp = {L: [] for L in leads}; rc = {L: [] for L in leads}
    hits = {k: {L: [[], []] for L in leads} for k in ("m", "p", "c")}; ratios = []
    for i in idx:
        out = model.simulate(data[i - 2:i + 1], float(doy[i]), 10)   # 0-based doy of the LAST input day (training convention)
        for L in leads:
            tr = data[i + L]; c = clim[(doy[i] + L) % 366]
            rm[L].append(S.rmse_lat(out[L - 1], tr)); rp[L].append(S.rmse_lat(data[i], tr)); rc[L].append(S.rmse_lat(c, tr))
            for k, p in (("m", out[L - 1]), ("p", data[i]), ("c", c)):
                hits[k][L][0].append(p[1]); hits[k][L][1].append(tr[1])
        ratios.append(S.variance_ratio(out[7], data[i + 8]))
        assert not S.validity(out, 10), S.validity(out, 10)
    print(f"{len(idx)} windows in {a.year}\n")
    print("lead  " + "  ".join(f"{c:>7s}" for c in ("t2m", "tp", "u10", "v10", "msl", "tcwv")) + "   SS(mean)  vsPersist  vsClim   CSI m/p/c")
    sss = []; csss = []
    for L in leads:
        m, p, c = np.mean(rm[L], 0), np.mean(rp[L], 0), np.mean(rc[L], 0); ss = 1 - m / np.minimum(p, c); sss.append(np.nanmean(ss))
        cs = {k: S.csi(np.stack(hits[k][L][0]), np.stack(hits[k][L][1])) for k in hits}; csss.append(S.csi_skill(cs["m"], cs["p"], cs["c"]))
        print(f"d{L}    " + "  ".join(f"{v:7.3f}" for v in ss) + f"   {np.nanmean(ss):7.3f}   {np.nanmean(1 - m / p):8.3f}  {np.nanmean(1 - m / c):7.3f}   {cs['m']:.2f}/{cs['p']:.2f}/{cs['c']:.2f}")
    fp = S.forecast_points(sss, csss)
    r = np.array(ratios); print("\nday-8 variance ratio per channel:", np.round(np.nanmean(r, 0), 2), " anti-blur pts", round(S.anti_blur_points(r), 2))

    # shocks on one monsoon window
    i = int(np.where(t == np.datetime64(f"{a.year}-07-15"))[0][0]); init = data[i - 2:i + 1]; d = float(doy[i])
    base = model.simulate(init, d, 10); res = []
    for name, box in (("warm_ocean", S.BOX_BAY_OF_BENGAL), ("monsoon_stall", S.BOX_ARABIAN_SEA)):
        out = model.simulate(init, d, 10, S.make_shock(name, init[-1], std))
        c = S.causality_checks(base, out, box, std); res.append(c)
        print(f"{name:14s} nonzero {c['nonzero']} localized {c['localized']} ({c['near_frac_day1']:.2f}) propagates {c['propagates']} stable {c['stable']} far {c['far_frac_max']:.3f}")
    null_ok = S.null_consistent(base, model.simulate(init, d, 10, np.zeros((NC, NLAT, NLON), 'f4')), std)
    cp = S.causality_points(res, null_ok)
    print(f"null shock consistent: {null_ok}")
    print(f"\nESTIMATED: forecast {fp['points']:.1f}/45 (rmse {fp['rmse_pts']:.1f}, csi {fp['csi_pts']:.1f})  anti-blur {S.anti_blur_points(r):.1f}/10  causality {cp:.1f}/15")


if __name__ == "__main__":
    main()
