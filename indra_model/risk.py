"""INDRA-Risk: a learned post-processor (model output statistics) that turns an INDRA forecast day into calibrated
event probabilities per cell:
    0  P(heavy rain >= 64.5 mm/day, IMD "heavy")      -> flood risk
    1  P(heat stress: daily-mean t2m >= 35 C, or >= 4.5 K above normal while >= 30 C)
It is trained on INDRA's OWN rollouts vs what really happened, with lead time as an input, so the probabilities
account for INDRA's growing error with lead (sharper at day 1, more cautious at day 8).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import NC, NLAT, NLON, to_model_space

EVENTS = ("heavy_rain", "heat_stress")
RISK_IN = 2 * NC + 2 + 3   # forecast state + anomaly vs climatology, static, (lead, sin doy, cos doy)


def event_labels(phys_day, clim_t2m):
    """phys_day [B,6,H,W] physical, clim_t2m [B,H,W] K -> [B,2,H,W] {0,1}."""
    t = phys_day[:, 0] - 273.15
    heavy = phys_day[:, 1] >= 64.5
    heat = (t >= 35) | ((phys_day[:, 0] - clim_t2m >= 4.5) & (t >= 30))
    return torch.stack([heavy, heat], 1).float()


class RiskNet(nn.Module):
    def __init__(self, w=48):
        super().__init__()
        self.c1 = nn.Conv2d(RISK_IN, w, 3, padding=1, padding_mode="replicate")
        self.c2 = nn.Conv2d(w, w, 3, padding=1, padding_mode="replicate")
        self.out = nn.Conv2d(w, len(EVENTS), 1)

    def forward(self, x):
        return self.out(F.gelu(self.c2(F.gelu(self.c1(x)))))    # logits


def risk_features(indra, z_day, doy, lead):
    """z_day [B,6,H,W] normalised INDRA state, doy [B] float, lead [B] float -> [B,RISK_IN,H,W]."""
    B = z_day.shape[0]
    ang = 2 * math.pi * doy / 365.0
    planes = torch.stack([lead / 10.0, torch.sin(ang), torch.cos(ang)], 1).view(B, 3, 1, 1).expand(B, 3, NLAT, NLON)
    return torch.cat([z_day, z_day - indra.clim_at(doy), indra.static.expand(B, -1, -1, -1), planes], 1)


@torch.no_grad()
def predict_risk(indra, risk, rollout_phys, doy_last):
    """rollout_phys [steps,6,H,W] numpy physical -> probabilities [steps,2,H,W] numpy."""
    x = torch.from_numpy(to_model_space(rollout_phys))
    z = indra.norm(x)
    steps = z.shape[0]
    lead = torch.arange(1, steps + 1, dtype=torch.float32)
    doy = float(doy_last) + lead
    return torch.sigmoid(risk(risk_features(indra, z, doy, lead))).numpy()
