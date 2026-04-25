"""
losses.py — Improved Loss for SpatialNet Dual-Head
====================================================
Key improvements over baseline:
1. Proper BCE loss for occurrence head (not faked)
2. Focal-weighted Tweedie for intensity (focus on hard examples)
3. Huber loss on rainy days for better correlation
4. Gradient-friendly formulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class DualHeadLoss(nn.Module):
    """
    Joint loss for occurrence + intensity dual-head model.
    
    L_total = α * L_occurrence + β * L_intensity + γ * L_correlation
    
    L_occurrence: BCE with class-weighted focal modulation
    L_intensity:  Tweedie + asymmetric penalty for extremes
    L_correlation: Huber loss on rainy days (boosts R without MSE sensitivity to outliers)
    """
    def __init__(self, p90_thresh=35.0, p95_thresh=48.97, p99_thresh=86.17):
        super().__init__()
        self.p90 = p90_thresh
        self.p95 = p95_thresh
        self.p99 = p99_thresh
        self.tweedie_p = config.TWEEDIE_P
        
        # Loss weights
        self.alpha = 0.3    # occurrence weight
        self.beta  = 1.0    # intensity weight  
        self.gamma = 0.15   # correlation (Huber) weight
    
    def forward(self, occ_logit, intensity, pred, target):
        # ── 1. Occurrence loss (proper BCE) ──
        rain_label = (target >= config.DRY_THRESHOLD).float()
        
        # Focal-style weighting: harder examples get more weight
        occ_prob = torch.sigmoid(occ_logit)
        # Weight dry days less (they're easy), rainy days more
        pos_weight = torch.where(rain_label == 1, 
                                 torch.tensor(2.0, device=target.device),
                                 torch.tensor(1.0, device=target.device))
        l_occ = F.binary_cross_entropy_with_logits(
            occ_logit, rain_label, weight=pos_weight, reduction='mean')
        
        # ── 2. Intensity loss (Tweedie + asymmetric extreme penalty) ──
        p = self.tweedie_p
        mu = intensity.clamp(min=1e-6)
        y  = target.clamp(min=0.0)
        
        term1 = torch.pow(y + 1e-8, 2 - p) / ((1 - p) * (2 - p))
        term2 = y * torch.pow(mu, 1 - p) / (1 - p)
        term3 = torch.pow(mu, 2 - p) / (2 - p)
        dev = 2.0 * (term1 - term2 + term3)
        
        # Asymmetric penalty weights
        scale = torch.ones_like(dev)
        
        # Under-prediction of extreme events — heavily penalize
        p90_under = (y >= self.p90) & (y < self.p95) & (mu < y)
        scale[p90_under] = config.PENALTY_P90_UNDER
        
        p95_under = (y >= self.p95) & (y < self.p99) & (mu < y)
        scale[p95_under] = config.PENALTY_P95_UNDER
        
        p99_under = (y >= self.p99) & (mu < y)
        scale[p99_under] = config.PENALTY_P99_UNDER
        
        # Over-prediction: false alarm for dry days
        fa_dry = (y < config.DRY_THRESHOLD) & (mu > 3.0)
        scale[fa_dry] = config.PENALTY_FALSE_ALARM
        
        # Over-prediction: crossing P90 threshold falsely
        fa_p90 = (y < self.p90) & (mu >= self.p90)
        scale[fa_p90] = config.PENALTY_FALSE_ALARM * 0.8
        
        l_intensity = (dev * scale).mean()
        
        # ── 3. Correlation loss (Huber on rainy days) ──
        rainy = target >= config.DRY_THRESHOLD
        if rainy.sum() > 2:
            l_corr = F.huber_loss(intensity[rainy], target[rainy], 
                                   reduction='mean', delta=15.0)
        else:
            l_corr = torch.tensor(0.0, device=target.device)
        
        # ── Combine ──
        total = self.alpha * l_occ + self.beta * l_intensity + self.gamma * l_corr
        
        components = {
            "occ_bce":   l_occ.item(),
            "tweedie":   l_intensity.item(),
            "huber":     l_corr.item(),
            "total":     total.item(),
        }
        return total, components


class CombinedLoss(DualHeadLoss):
    """Alias for compatibility with train_dual_eval.py interface."""
    pass
