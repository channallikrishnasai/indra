"""Train INDRA-Risk (heavy-rain and heat-stress probabilities) on INDRA's own rollouts vs observed outcomes.

    python train_risk.py --data data/india_climate_train.nc --indra weights/indra.pt --out weights/risk.pt

Run AFTER calibrate.py. Training pairs: for windows in training years, run INDRA 10 days ahead; input = INDRA's forecast day k (+ lead k),
target = what actually happened on that day. Held-out year (2021) is scored with ROC-AUC and Brier skill.
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from indra_model.net import load_indra, to_model_space
from indra_model.risk import RiskNet, risk_features, event_labels, EVENTS

LEADS = 10


def auc(p, y):
    o = np.argsort(p); y = y[o]; n1 = y.sum(); n0 = len(y) - n1
    return float((np.cumsum(1 - y)[y == 1]).sum() / max(n1 * n0, 1)) if n1 and n0 else float("nan")


def build(indra, Z, phys, doy, idx, clim_t2m):
    X, Y = [], []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = torch.as_tensor(idx[s:s + 64]); d = torch.as_tensor(doy[idx[s:s + 64]], dtype=torch.float32)
            out = indra.calibrated(indra.rollout_z(torch.stack([Z[b - 2], Z[b - 1], Z[b]], 1), d, LEADS), d)   # what /simulate returns
            for k in range(LEADS):
                lead = torch.full((len(b),), float(k + 1))
                X.append(risk_features(indra, out[:, k], d + k + 1, lead))
                tgt = torch.from_numpy(phys[(b + k + 1).numpy()])
                Y.append(event_labels(tgt, torch.from_numpy(clim_t2m[(doy[idx[s:s + 64]] + k + 1) % 366])))
    return torch.cat(X), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--indra", default="weights/indra.pt"); ap.add_argument("--out", default="weights/risk.pt")
    ap.add_argument("--val-from", type=int, default=2021); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--every", type=int, default=2); a = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(8)
    indra = load_indra(a.indra)
    ds = xr.open_dataset(a.data); t = ds["time"].values.astype("datetime64[D]"); phys = ds["data"].values.astype("f4")
    years = t.astype("datetime64[Y]").astype(int) + 1970; doy = (t - t.astype("datetime64[Y]")).astype(int)
    tr = years < a.val_from
    ct = np.zeros((366, 32, 36)); cnt = np.zeros(366)                       # t2m climatology, training years
    np.add.at(ct, doy[tr], phys[tr, 0]); np.add.at(cnt, doy[tr], 1); ct /= np.maximum(cnt, 1)[:, None, None]
    ct[365] = ct[364]; ct = np.mean([np.roll(ct, o, 0) for o in range(-7, 8)], axis=0).astype("f4")
    Z = indra.norm(torch.from_numpy(to_model_space(phys)))
    ok = lambda m: np.array([i for i in np.where(m)[0] if i >= 2 and i + LEADS < len(t) and m[i + LEADS]])
    t0 = time.time()
    Xtr, Ytr = build(indra, Z, phys, doy, ok(tr)[::a.every], ct)
    Xva, Yva = build(indra, Z, phys, doy, ok(~tr)[::3], ct)
    print(f"pairs train {len(Xtr)} val {len(Xva)}  base rates {Ytr.mean(dim=(0, 2, 3)).numpy().round(4)}  ({time.time()-t0:.0f}s)")

    pos = Ytr.mean(dim=(0, 2, 3)); pw = ((1 - pos) / pos.clamp_min(1e-4)).clamp(max=50).sqrt().view(1, -1, 1, 1)
    net = RiskNet(); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for e in range(a.epochs):
        perm = torch.randperm(len(Xtr)); run = 0.0
        for s in range(0, len(perm), 128):
            j = perm[s:s + 128]
            loss = F.binary_cross_entropy_with_logits(net(Xtr[j]), Ytr[j], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item() * len(j)
        print(f"epoch {e+1} loss {run/len(Xtr):.4f}", flush=True)

    # pos_weight shifts probabilities up: recalibrate with one temperature/bias per event (Platt) on training pairs
    with torch.no_grad():
        lt = net(Xtr[::4]); yt = Ytr[::4]
    ab = torch.zeros(len(EVENTS), 2, requires_grad=True); o2 = torch.optim.LBFGS([ab], max_iter=100)
    def closure():
        o2.zero_grad(); l = F.binary_cross_entropy_with_logits(lt * ab[:, 0].exp().view(1, -1, 1, 1) + ab[:, 1].view(1, -1, 1, 1), yt); l.backward(); return l
    o2.step(closure)
    with torch.no_grad():
        net.out.weight.mul_(ab[:, 0].exp().view(-1, 1, 1, 1)); net.out.bias.mul_(ab[:, 0].exp()).add_(ab[:, 1])
        p = torch.sigmoid(net(Xva)).numpy(); y = Yva.numpy()
    for c, name in enumerate(EVENTS):
        pc, yc = p[:, c].ravel(), y[:, c].ravel(); clim_p = yc.mean()
        bss = 1 - ((pc - yc) ** 2).mean() / max(((clim_p - yc) ** 2).mean(), 1e-9)
        print(f"{name:12s} held-out {a.val_from}: AUC {auc(pc, yc):.3f}  Brier skill {bss:.3f}  base rate {clim_p:.4f}")
    torch.save({"state": net.state_dict()}, a.out); print("saved", a.out)


if __name__ == "__main__":
    main()
