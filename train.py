"""Train INDRA on india_climate_train.nc (CPU is enough: ~1-2 min/epoch on 8 threads).

    python train.py --data data/india_climate_train.nc --out weights/indra.pt

Curriculum: single-step teacher forcing, then autoregressive rollout fine-tuning (2 -> 4 -> 6 steps) so the
model learns to correct its own errors instead of drifting or blurring. Years >= --val-from are held out.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from indra_model.net import Indra, to_model_space, NC, LAT


def load(path):
    ds = xr.open_dataset(path)
    data = ds["data"].values.astype("f4")                      # [N,6,32,36] physical
    static = ds["static"].values.astype("f4")
    t = ds["time"].values.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    doy = ((t - t.astype("datetime64[Y]")).astype(int)).astype("f4")   # 0-based day of year
    return data, static, years, doy


def climatology(z, doy, smooth=15):
    """[N,6,H,W] transformed fields -> [366,6,H,W] smoothed calendar-day mean (0-based doy index)."""
    idx = np.minimum(doy.astype(int), 365)
    clim = np.zeros((366,) + z.shape[1:], "f8"); cnt = np.zeros(366)
    np.add.at(clim, idx, z); np.add.at(cnt, idx, 1)
    clim /= np.maximum(cnt, 1)[:, None, None, None]
    clim[365] = np.where(cnt[365] > 0, clim[365], clim[364])
    h = smooth // 2
    return np.mean([np.roll(clim, o, axis=0) for o in range(-h, h + 1)], axis=0).astype("f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--out", default="weights/indra.pt")
    ap.add_argument("--val-from", type=int, default=2021)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--epochs1", type=int, default=14)
    ap.add_argument("--rollout", default="2:3,4:3,6:2", help="steps:epochs curriculum after single-step training")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from this checkpoint (fine-tune)")
    ap.add_argument("--rain-w", type=float, default=1.5, help="loss weight on rain")
    ap.add_argument("--heavy-w", type=float, default=0.0, help="extra loss weight on target cells with rain >= 10 mm/day (real observations)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data, static, years, doy = load(a.data)
    z = to_model_space(data)
    tr = years < a.val_from
    mean = z[tr].mean(axis=(0, 2, 3)); std = z[tr].std(axis=(0, 2, 3))
    clim = climatology(z[tr], doy[tr])
    model = Indra(mean, std, static, clim, width=a.width)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True)["state"])
    print(f"params {sum(p.numel() for p in model.net.parameters())/1e6:.2f}M  train days {tr.sum()}  val days {(~tr).sum()}")

    Z = model.norm(torch.from_numpy(z))                         # [N,6,H,W] normalised
    D = torch.from_numpy(doy)
    w = torch.as_tensor(np.cos(np.deg2rad(LAT)), dtype=torch.float32); w = (w / w.mean()).view(1, 1, -1, 1)
    chw = torch.tensor([1.0, a.rain_w, 1.0, 1.0, 1.0, 1.0]).view(1, NC, 1, 1)   # rain matters twice: RMSE + CSI

    def starts(mask, K):
        ok = np.where(mask)[0]; ok = ok[(ok >= 2) & (ok + K < len(Z))]
        return ok[mask[ok + K]]                                 # whole window inside the split

    def batch(idx, K):
        idx = torch.as_tensor(idx)
        x = torch.stack([Z[idx - 2], Z[idx - 1], Z[idx]], 1)
        y = torch.stack([Z[idx + k] for k in range(1, K + 1)], 1)
        return x, y, D[idx]

    def loss_fn(pred, y):
        base = ((pred - y) ** 2) * w[:, None] * chw[:, None]
        if a.heavy_w:                                  # heavier loss on real heavy-rain days (rain channel is log1p, normalised)
            r = y[:, :, 1] * model.std[0, 1] + model.mean[0, 1]
            base = base * (1 + a.heavy_w * (r > math.log1p(10.0)).float()[:, :, None])
        return base.mean()

    @torch.no_grad()
    def val(K=8):
        model.eval(); vi = starts(~tr, K)[::3]; tot = torch.zeros(K)
        for s in range(0, len(vi), 64):
            x, y, d = batch(vi[s:s + 64], K)
            p = model.rollout_z(x, d, K)
            tot += (((p - y) ** 2) * w[:, None]).mean(dim=(0, 2, 3, 4)) * len(x[:, 0])
        model.train(); return (tot / len(vi)).sqrt()

    opt = torch.optim.AdamW(model.net.parameters(), lr=a.lr, weight_decay=1e-4)
    phases = [(1, a.epochs1, a.lr)] + [(int(k), int(e), a.lr * 0.15) for k, e in (p.split(":") for p in a.rollout.split(","))]
    for K, epochs, lr in phases:
        idx_all = starts(tr, K); steps_total = epochs * math.ceil(len(idx_all) / a.bs); it = 0
        for e in range(epochs):
            t0 = time.time(); perm = np.random.permutation(idx_all); run = 0.0
            for s in range(0, len(perm), a.bs):
                for g in opt.param_groups:                       # cosine schedule per phase
                    g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * it / steps_total)); it += 1
                x, y, d = batch(perm[s:s + a.bs], K)
                pred = model.rollout_z(x, d, K)
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step(); run += loss.item()
            v = val()
            print(f"K={K} ep {e+1}/{epochs} loss {run/math.ceil(len(perm)/a.bs):.4f} val-rmse(norm) d1 {v[0]:.3f} d4 {v[3]:.3f} d8 {v[7]:.3f}  {time.time()-t0:.0f}s", flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "width": a.width, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}, a.out)  # tensors, not numpy pickles
    meta = {"model": "INDRA v1", "val_from": a.val_from, "width": a.width, "params": sum(p.numel() for p in model.net.parameters())}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
