"""INDRA core: a local, residual, physics-constrained neural stepper for India's daily weather.

One step maps (day t-2, t-1, t) -> day t+1. Design choices that matter for the benchmark:
  * Purely local convolutions (receptive field 7 cells per day) -> a shock can only move ~7 cells/day,
    so perturbations stay local on day 1 and spread outward with time (causality by construction).
  * Residual (tendency) prediction in normalised space + rain in log1p space -> stable, sharp rollouts.
  * Climatology of the target day, static land/sea + orography and lat/lon are inputs -> the net learns
    anomaly dynamics instead of memorising the mean state; long rollouts relax towards a realistic climate.
  * Hard physical constraints on the output: rain >= 0, moisture >= 0, t2m inside [-40, +60] C.
  * Fully deterministic -> a zero perturbation reproduces the unperturbed rollout exactly.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC, NLAT, NLON = 6, 32, 36
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)
IN_CH = 3 * NC + NC + 2 + 2 + 2   # state + 2 tendencies, target climatology, static, doy, lat/lon


def to_model_space(x: np.ndarray) -> np.ndarray:
    """Physical -> transformed space (rain becomes log1p(rain)); works on [..., 6, H, W]."""
    y = np.array(x, dtype="f4", copy=True)
    y[..., 1, :, :] = np.log1p(np.maximum(y[..., 1, :, :], 0.0))
    return y


def to_physical(y: np.ndarray) -> np.ndarray:
    x = np.array(y, dtype="f4", copy=True)
    x[..., 1, :, :] = np.expm1(np.maximum(x[..., 1, :, :], 0.0))
    return x


def doy_from_sincos(s: float, c: float) -> float:
    """Inverse of [sin(2 pi d/365), cos(2 pi d/365)] -> d in [0, 365)."""
    return (math.atan2(s, c) / (2 * math.pi) * 365.0) % 365.0


DYN_CH = 2   # feats=True models: moisture convergence and low-pressure anomaly, from the state every step


def add_static_feats(static: np.ndarray) -> np.ndarray:
    """[2,H,W] (land/sea mask, orography m) -> [6,H,W]: + terrain slope, aspect (sin, cos), distance to coast (cells).
    Computed from the static fields only, so it is the same for every request."""
    from scipy.ndimage import distance_transform_edt
    land = static[0] > 0.5; oro = static[1].astype("f8")
    gy, gx = np.gradient(oro); ang = np.arctan2(gy, gx)
    dist = distance_transform_edt(land) if land.any() and (~land).any() else np.zeros_like(oro)
    return np.concatenate([static, np.hypot(gx, gy)[None], np.sin(ang)[None], np.cos(ang)[None], dist[None]]).astype("f4")


class ResBlock(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(w, w, 3, padding=1, padding_mode="replicate")
        self.n = nn.GroupNorm(8, w)

    def forward(self, x):
        return x + self.c2(F.gelu(self.c1(F.gelu(self.n(x)))))


class IndraNet(nn.Module):
    """Stem 3x3 + 3 residual blocks (2 x 3x3 each) + 1x1 head = receptive-field radius 7 cells per step."""

    def __init__(self, width: int = 96, blocks: int = 3, in_ch: int = IN_CH):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.head = nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, NC, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, feats):
        return self.head(self.blocks(self.stem(feats)))


class Indra(nn.Module):
    """Wraps IndraNet with normalisation buffers, static fields and climatology so rollout() is self-contained."""

    def __init__(self, mean, std, static, clim, width=96, blocks=3, feats=False):
        super().__init__()
        self.feats = feats                       # True: 4 extra physical features (terrain, coast, convergence, low)
        if feats:
            static = add_static_feats(np.asarray(static, dtype="f4"))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, NC, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, NC, 1, 1))
        st = torch.as_tensor(static, dtype=torch.float32)
        st = (st - st.mean(dim=(1, 2), keepdim=True)) / (st.std(dim=(1, 2), keepdim=True) + 1e-6)
        self.register_buffer("static", st[None])                                  # [1,2 or 6,H,W]
        in_ch = 3 * NC + NC + st.shape[0] + 2 + 2 + (DYN_CH if feats else 0)
        self.net = IndraNet(width, blocks, in_ch)
        self.register_buffer("clim", torch.as_tensor(clim, dtype=torch.float32))   # [366,6,H,W] transformed space
        la = torch.as_tensor((LAT - LAT.mean()) / LAT.std(), dtype=torch.float32)
        lo = torch.as_tensor((LON - LON.mean()) / LON.std(), dtype=torch.float32)
        self.register_buffer("latlon", torch.stack(torch.meshgrid(la, lo, indexing="ij"))[None])

    def norm(self, x):
        return (x - self.mean) / self.std

    def clim_at(self, doy: torch.Tensor) -> torch.Tensor:
        """doy float [B] (0..365) -> linearly interpolated normalised climatology [B,6,H,W]."""
        d0 = torch.floor(doy).long() % 366; d1 = (d0 + 1) % 366; f = (doy - torch.floor(doy)).view(-1, 1, 1, 1)
        return self.norm(self.clim[d0] * (1 - f) + self.clim[d1] * f)

    def step(self, z3: torch.Tensor, doy_next: torch.Tensor) -> torch.Tensor:
        """z3 [B,3,6,H,W] normalised transformed states (oldest first) -> next normalised state [B,6,H,W]."""
        B = z3.shape[0]
        x0, x1, x2 = z3[:, 0], z3[:, 1], z3[:, 2]
        ang = 2 * math.pi * doy_next / 365.0
        doyp = torch.stack([torch.sin(ang), torch.cos(ang)], 1).view(B, 2, 1, 1).expand(B, 2, NLAT, NLON)
        extra = [self.dyn_feats(x2)] if self.feats else []
        f = torch.cat([x2, x2 - x1, x1 - x0, self.clim_at(doy_next), self.static.expand(B, -1, -1, -1), *extra,
                       doyp, self.latlon.expand(B, -1, -1, -1)], 1)
        nxt = x2 + self.net(f)
        return self.constrain(nxt)

    def dyn_feats(self, x2):
        """Two features from the last normalised state [B,6,H,W]: moisture convergence (-div(u,v) x tcwv) and the
        low-pressure anomaly (msl minus its 7x7 mean). Central differences, replicate edges. Normalised units."""
        u, v, msl, q = x2[:, 2:3], x2[:, 3:4], x2[:, 4:5], x2[:, 5:6]
        up = F.pad(u, (1, 1, 0, 0), mode="replicate"); dudx = (up[..., 2:] - up[..., :-2]) / 2
        vp = F.pad(v, (0, 0, 1, 1), mode="replicate"); dvdy = (vp[:, :, 2:] - vp[:, :, :-2]) / 2
        conv = -(dudx + dvdy) * q
        anom = msl - F.avg_pool2d(F.pad(msl, (3, 3, 3, 3), mode="replicate"), 7, stride=1)
        return torch.cat([conv, anom], 1)

    def constrain(self, z):
        """Hard physical limits, expressed in normalised space (rain/moisture >= 0, -40 C <= t2m <= +60 C)."""
        lo = (torch.tensor([233.5, 0.0, -60.0, -60.0, 90000.0, 0.0], device=z.device).view(1, NC, 1, 1) - self.mean) / self.std
        hi = (torch.tensor([332.5, 7.0, 60.0, 60.0, 110000.0, 90.0], device=z.device).view(1, NC, 1, 1) - self.mean) / self.std
        return torch.maximum(torch.minimum(z, hi), lo)

    def rollout_z(self, z3, doy_last, steps):
        outs = []
        for k in range(steps):
            nxt = self.step(z3, doy_last + k + 1)
            outs.append(nxt)
            z3 = torch.cat([z3[:, 1:], nxt[:, None]], 1)
        return torch.stack(outs, 1)

    alpha = None     # [L,6] calibrated anomaly damping per lead (set by calibrate.py), None = raw model
    clim_tp = None   # [366,H,W] arithmetic-mean rain climatology (mm/day) for physical-space rain calibration
    alpha_tp = None  # [L] rain damping per lead, physical space
    qmix = 1.0                    # strength of the rain mapping
    qmap_pred = qmap_obs = None  # [10,201] rain quantiles of the model / of observations (calibrate_rain.py)

    def map_rain(self, x):
        """Quantile-map the rain channel of a physical rollout [steps,6,H,W] (no-op when not fitted)."""
        if self.qmap_pred is None:
            return x
        x = x.copy()
        for k in range(x.shape[0]):
            qp, qo = self.qmap_pred[min(k, 9)].numpy(), self.qmap_obs[min(k, 9)].numpy()
            mapped = np.interp(np.maximum(x[k, 1], 0.0), qp, qo)   # np.interp clamps above the top quantile
            x[k, 1] = (1 - self.qmix) * x[k, 1] + self.qmix * mapped     # qmix = 1: full mapping, 0.5: halfway
        return x

    def calibrated(self, out_z, doy_last):
        """Report clim + alpha_k * (state - clim) per lead k. The recursion itself stays undamped, so the dynamics
        (and shock propagation) are untouched; only the reported field shrinks unpredictable anomalies."""
        if self.alpha is None:
            return out_z
        res = []
        for k in range(out_z.shape[1]):
            c = self.clim_at(doy_last + k + 1)
            a = self.alpha[min(k, self.alpha.shape[0] - 1)].view(1, NC, 1, 1)
            z = c + a * (out_z[:, k] - c)
            if self.clim_tp is not None:
                # rain is shrunk in PHYSICAL space towards the arithmetic-mean rain climatology: shrinking in log space
                # would pull towards the geometric mean and wipe out heavy rain (bad for the 10 mm/day CSI)
                d = doy_last + k + 1; d0 = torch.floor(d).long() % 366; f = (d - torch.floor(d)).view(-1, 1, 1)
                ct = self.clim_tp[d0] * (1 - f) + self.clim_tp[(d0 + 1) % 366] * f
                raw = torch.expm1((out_z[:, k, 1] * self.std[0, 1] + self.mean[0, 1]).clamp_min(0))
                tp = (ct + self.alpha_tp[min(k, len(self.alpha_tp) - 1)] * (raw - ct)).clamp_min(0)
                z = torch.cat([z[:, :1], ((torch.log1p(tp) - self.mean[0, 1]) / self.std[0, 1])[:, None], z[:, 2:]], 1)
            res.append(self.constrain(z))
        return torch.stack(res, 1)

    def _z(self, x):
        return self.norm(torch.from_numpy(to_model_space(x)).flatten(0, 1)).view(x.shape[0], 3, NC, NLAT, NLON)

    @torch.no_grad()
    def simulate(self, init_phys: np.ndarray, doy_last: float, steps: int, perturbation: np.ndarray | None = None) -> np.ndarray:
        """init_phys [3,6,H,W] physical, perturbation [6,H,W] physical delta on the last day (or None)
        -> rollout [steps,6,H,W] physical.

        Multiverse = calibrated Prime forecast + the FULL dynamical response of the shock:
            out = calib(F(x)) + (F(x + d) - F(x))
        so skill-calibration never mutes a shock, and a zero shock is bit-identical to Prime."""
        doy = torch.tensor([float(doy_last)])
        if perturbation is not None and not np.any(perturbation):
            perturbation = None       # zero shock -> exactly the Prime path (no batch/unit round trip)
        phys = lambda z: to_physical((z * self.std[None] + self.mean[None]).numpy())[0]
        if perturbation is None:
            return self.map_rain(phys(self.calibrated(self.rollout_z(self._z(init_phys[None]), doy, steps), doy)))
        shocked = init_phys.copy(); shocked[-1] = shocked[-1] + perturbation
        shocked[-1, 1] = np.maximum(shocked[-1, 1], 0.0); shocked[-1, 5] = np.maximum(shocked[-1, 5], 0.0)
        raw = self.rollout_z(self._z(np.stack([init_phys, shocked])), doy.repeat(2), steps)
        # compose in PHYSICAL units: adding a log-rain response onto a different base would rescale it
        out = self.map_rain(phys(self.calibrated(raw[:1], doy))) + (phys(raw[1:]) - phys(raw[:1]))
        z = self.norm(torch.from_numpy(to_model_space(out)))
        return phys(self.constrain(z)[None])


def load_indra(path) -> Indra:
    """Rebuild INDRA from a checkpoint written by train.py (+ alpha from calibrate.py, if present)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    feats = ck["state"]["static"].shape[1] == 6          # feature models store 6 static channels
    m = Indra(ck["mean"], ck["std"], np.zeros((2, NLAT, NLON), "f4"), np.zeros((366, NC, NLAT, NLON), "f4"), width=ck["width"], feats=feats)
    m.load_state_dict(ck["state"]); m.eval()
    if "alpha" in ck:
        m.alpha = torch.as_tensor(ck["alpha"], dtype=torch.float32)
    if "qmap_pred" in ck:
        m.qmap_pred = ck["qmap_pred"].float(); m.qmap_obs = ck["qmap_obs"].float(); m.qmix = float(ck.get("qmix", 1.0))
    if "clim_tp" in ck:
        m.clim_tp = torch.as_tensor(ck["clim_tp"], dtype=torch.float32)
        m.alpha_tp = torch.as_tensor(ck["alpha_tp"], dtype=torch.float32)
    return m
