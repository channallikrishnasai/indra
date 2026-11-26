"""Fit INDRA's per-lead anomaly calibration on TRAINING years and store it in the checkpoint.

For each lead k and channel c:  alpha[k,c] = <p', t'> / <p', p'>  with p' = forecast - climatology, t' = truth - climatology
(lat-weighted). This is the least-squares optimal shrinkage of the forecast anomaly: near 1 at day 1, decaying as
skill fades, so long leads relax towards climate instead of drifting. Rain is fitted in physical space (mm/day)
against the arithmetic-mean rain climatology, which keeps heavy rain alive for the 10 mm/day CSI.
    python calibrate.py --data data/india_climate_train.nc --weights weights/indra.pt --years 2015-2020
"""
import argparse
import numpy as np
import torch
import xarray as xr

from indra_model.net import load_indra, to_model_space, NC, LAT

LEADS = 14


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--weights", default="weights/indra.pt"); ap.add_argument("--years", default="2015-2020")
    ap.add_argument("--val-from", type=int, default=2021); ap.add_argument("--every", type=int, default=3); a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-")); assert y1 < a.val_from, "calibrate on training years only"
    torch.set_num_threads(8)
    m = load_indra(a.weights); m.alpha = None; m.clim_tp = None
    ds = xr.open_dataset(a.data); t = ds["time"].values.astype("datetime64[D]"); phys = ds["data"].values.astype("f4")
    years = t.astype("datetime64[Y]").astype(int) + 1970; doy = (t - t.astype("datetime64[Y]")).astype(int)

    tr = years < a.val_from                                           # arithmetic rain climatology, training years
    ct = np.zeros((366,) + phys.shape[2:]); cnt = np.zeros(366)
    np.add.at(ct, doy[tr], phys[tr, 1]); np.add.at(cnt, doy[tr], 1); ct /= np.maximum(cnt, 1)[:, None, None]
    ct[365] = ct[364] if cnt[365] == 0 else ct[365]
    ct = np.mean([np.roll(ct, o, 0) for o in range(-7, 8)], axis=0).astype("f4")
    ctt = torch.from_numpy(ct)

    Z = m.norm(torch.from_numpy(to_model_space(phys))); P = torch.from_numpy(phys[:, 1])
    idx = np.array([i for i in np.where((years >= y0) & (years <= y1))[0] if i >= 2 and i + LEADS < len(t)])[::a.every]
    w = torch.as_tensor(np.cos(np.deg2rad(LAT)), dtype=torch.float32).view(1, 1, -1, 1)
    num = torch.zeros(LEADS, NC); den = torch.zeros(LEADS, NC); nr = torch.zeros(LEADS); dr = torch.zeros(LEADS)
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = torch.as_tensor(idx[s:s + 64]); d = torch.as_tensor(doy[idx[s:s + 64]], dtype=torch.float32)
            out = m.rollout_z(torch.stack([Z[b - 2], Z[b - 1], Z[b]], 1), d, LEADS)
            for k in range(LEADS):
                c = m.clim_at(d + k + 1); p = out[:, k] - c; q = Z[b + k + 1] - c
                num[k] += (p * q * w).sum(dim=(0, 2, 3)); den[k] += (p * p * w).sum(dim=(0, 2, 3))
                rain = torch.expm1((out[:, k, 1] * m.std[0, 1] + m.mean[0, 1]).clamp_min(0))
                cr = ctt[(torch.as_tensor(doy[idx[s:s + 64]]) + k + 1) % 366]
                pr = rain - cr; qr = P[b + k + 1] - cr
                nr[k] += (pr * qr * w[0]).sum(); dr[k] += (pr * pr * w[0]).sum()
    alpha = (num / den.clamp_min(1e-6)).clamp(0, 1); alpha_tp = (nr / dr.clamp_min(1e-6)).clamp(0, 1)
    print("alpha per lead (rows) x channel t2m tp u10 v10 msl tcwv:\n", np.round(alpha.numpy(), 2))
    print("rain alpha (physical):", np.round(alpha_tp.numpy(), 2))
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    ck.update(alpha=alpha, alpha_tp=alpha_tp, clim_tp=torch.from_numpy(ct))  # tensors: the server loads with weights_only=True
    torch.save(ck, a.weights); print("saved calibration into", a.weights)


if __name__ == "__main__":
    main()
