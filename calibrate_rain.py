"""Rain quantile mapping (bias correction), fitted on TRAINING years only.

For each lead k = 1..10, the calibrated model's rain distribution is matched to the observed rain distribution over the
same training years (2015-2020): the stored tables are the quantiles of both. At run time, INDRA maps its predicted rain
through these tables, so it produces about as many heavy-rain days as really happen. Only the rain channel is touched.
    python calibrate_rain.py --weights weights/indra.pt --years 2015-2020
"""
import argparse
import numpy as np
import torch
import xarray as xr
from indra_model.net import load_indra, to_model_space

QUANT = np.linspace(0, 1, 201)                  # 200 bins; the top tail is what matters for heavy rain


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--weights", default="weights/indra.pt"); ap.add_argument("--years", default="2015-2020")
    ap.add_argument("--val-from", type=int, default=2021); ap.add_argument("--every", type=int, default=3); a = ap.parse_args()
    y0, y1 = map(int, a.years.split("-")); assert y1 < a.val_from, "fit on training years only"
    torch.set_num_threads(8)
    m = load_indra(a.weights); m.qmap_pred = m.qmap_obs = None
    ds = xr.open_dataset(a.data); t = ds["time"].values.astype("datetime64[D]"); phys = ds["data"].values.astype("f4")
    years = t.astype("datetime64[Y]").astype(int) + 1970; doy = (t - t.astype("datetime64[Y]")).astype(int).astype("f4")
    Z = m.norm(torch.from_numpy(to_model_space(phys)))
    idx = np.array([i for i in np.where((years >= y0) & (years <= y1))[0] if i >= 2 and i + 10 < len(t)])[::a.every]
    pred = [[] for _ in range(10)]; obs = [[] for _ in range(10)]
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = torch.as_tensor(idx[s:s + 64]); d = torch.as_tensor(doy[idx[s:s + 64]])
            out = m.calibrated(m.rollout_z(torch.stack([Z[b - 2], Z[b - 1], Z[b]], 1), d, 10), d)     # what /simulate returns
            rain = torch.expm1((out[:, :, 1] * m.std[0, 1] + m.mean[0, 1]).clamp_min(0)).numpy()
            for k in range(10):
                pred[k].append(rain[:, k].ravel()); obs[k].append(phys[(idx[s:s + 64] + k + 1), 1].ravel())
    qp = np.stack([np.quantile(np.concatenate(p), QUANT) for p in pred]); qo = np.stack([np.quantile(np.concatenate(o), QUANT) for o in obs])
    print("day-1 predicted vs observed rain quantiles (90, 99, 99.9%):", np.round(np.quantile(np.concatenate(pred[0]), [.9, .99, .999]), 1),
          np.round(np.quantile(np.concatenate(obs[0]), [.9, .99, .999]), 1))
    ck = torch.load(a.weights, map_location="cpu", weights_only=True); ck.update(qmap_pred=torch.from_numpy(qp).float(), qmap_obs=torch.from_numpy(qo).float())
    torch.save(ck, a.weights); print("saved rain quantile maps into", a.weights)


if __name__ == "__main__":
    main()
