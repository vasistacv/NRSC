"""
losses.py
=========
v4-restored — the configuration that gave best correlation (0.51) and SEDI (0.79).
With softened over-prediction penalty from v5 (keeps FAR manageable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config


class TweedieExLoss(nn.Module):
    def __init__(self, p=1.5, p90_thresh=35.0, p95_thresh=48.97, p99_thresh=86.17):
        super().__init__()
        self.p = p
        self.p90_thresh = p90_thresh
        self.p95_thresh = p95_thresh
        self.p99_thresh = p99_thresh

    def forward(self, mu, y):
        p  = self.p
        mu = mu.clamp(min=1e-6)
        y  = y.clamp(min=0.0)

        term1 = torch.pow(y + 1e-8, 2 - p) / ((1 - p) * (2 - p))
        term2 = y * torch.pow(mu, 1 - p) / (1 - p)
        term3 = torch.pow(mu, 2 - p) / (2 - p)
        dev   = 2.0 * (term1 - term2 + term3)

        scale = torch.ones_like(dev)

        # Under-prediction penalties
        p90_under = (y >= self.p90_thresh) & (y < self.p95_thresh) & (mu < y)
        scale[p90_under] = config.PENALTY_P90_UNDER

        p95_under = (y >= self.p95_thresh) & (y < self.p99_thresh) & (mu < y)
        scale[p95_under] = config.PENALTY_P95_UNDER

        p99_under = (y >= self.p99_thresh) & (mu < y)
        scale[p99_under] = config.PENALTY_P99_UNDER

        # Over-prediction: dry false alarm
        fa_dry = (y < config.DRY_THRESHOLD) & (mu > 3.0)
        scale[fa_dry] = config.PENALTY_FALSE_ALARM

        # Over-prediction: threshold-crossing at P90 (soft)
        fa_p90 = (y < self.p90_thresh) & (mu >= self.p90_thresh)
        scale[fa_p90] = config.PENALTY_FALSE_ALARM * 0.8

        return (dev * scale).mean()


def quantile_loss(pred, target, q):
    err = target - pred
    return torch.mean(torch.max(q * err, (q - 1) * err))


class QuantileBlendLoss(nn.Module):
    def __init__(self, p90_thresh=35.0):
        super().__init__()

    def forward(self, pred, target):
        rainy_mask = target >= config.DRY_THRESHOLD
        if rainy_mask.sum() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p, t = pred[rainy_mask], target[rainy_mask]
        return 0.5 * quantile_loss(p, t, 0.90) + 0.5 * quantile_loss(p, t, 0.95)


class CombinedLoss(nn.Module):
    def __init__(self, p90_thresh=35.0, p95_thresh=48.97, p99_thresh=86.17):
        super().__init__()
        self.tweedie = TweedieExLoss(
            p=config.TWEEDIE_P,
            p90_thresh=p90_thresh, p95_thresh=p95_thresh, p99_thresh=p99_thresh,
        )
        self.quantile = QuantileBlendLoss(p90_thresh=p90_thresh)
        self.alpha = config.QUANTILE_BLEND_WEIGHT

    def forward(self, occ_logit, intensity, pred, target):
        l_tweedie  = self.tweedie(pred, target)
        l_quantile = self.quantile(pred, target)

        # Small MSE on rainy days to boost correlation
        rainy = target >= config.DRY_THRESHOLD
        if rainy.sum() > 2:
            l_mse = F.mse_loss(pred[rainy], target[rainy]) * 0.005
        else:
            l_mse = torch.tensor(0.0, device=pred.device)

        total = l_tweedie + self.alpha * l_quantile + l_mse

        components = {
            "tweedie":  l_tweedie.item(),
            "quantile": l_quantile.item(),
            "occ_bce":  l_mse.item(),
            "total":    total.item(),
        }
        return total, components
