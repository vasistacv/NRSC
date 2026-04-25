"""
model.py — SpatialNet: Proper CNN for Rainfall Downscaling
==========================================================
Unlike SmallNet (which flattens the 9×9 patch, losing spatial structure),
SpatialNet uses Conv2D layers to extract spatial features like:
  - Moisture gradients (TCWV spatial patterns)
  - Convergence zones (wind field patterns)
  - CAPE spatial distribution
  - Multi-level humidity transitions

Architecture:
  CNN branch:  Conv → SE → ResBlock → SE → Conv → AvgPool+MaxPool → 192-d
  Tab branch:  MLP 24→64→32
  Fusion:      224→96→48 → dual heads (occurrence + intensity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ──────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learn channel importance."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4), bias=False), nn.SiLU(inplace=True),
            nn.Linear(max(ch // reduction, 4), ch, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class ResBlock(nn.Module):
    """Residual block with two convolutions + SE attention."""
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.se    = SEBlock(ch)
        self.drop  = nn.Dropout2d(dropout)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


# ──────────────────────────────────────────────────────────────────────────────
# SPATIALNET — PROPER CNN FOR 9×9 PATCHES
# ──────────────────────────────────────────────────────────────────────────────

class SpatialNet(nn.Module):
    """
    CNN-based rainfall downscaling model that preserves spatial structure.
    
    Key differences from SmallNet:
    1. Uses Conv2D to learn spatial features (gradients, convergence, etc.)
    2. SE blocks for channel attention (which weather variables matter?)
    3. Residual connections for better gradient flow
    4. Dual pooling (avg + max) for richer spatial summary
    5. Proper dual-head: occurrence (classification) + intensity (regression)
    """
    def __init__(self, n_channels=19, n_tabular=24, window_size=9):
        super().__init__()
        
        # ── CNN spatial branch ──
        self.cnn = nn.Sequential(
            # Block 1: 19 → 48 channels
            nn.Conv2d(n_channels, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            SEBlock(48),
            nn.Dropout2d(0.10),
            
            # Block 2: Residual block (preserves 48 channels)
            ResBlock(48, dropout=0.10),
            
            # Block 3: 48 → 96 channels
            nn.Conv2d(48, 96, 3, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            SEBlock(96),
            nn.Dropout2d(0.10),
            
            # Block 4: Another residual block
            ResBlock(96, dropout=0.10),
        )
        
        # Dual pooling: avg + max for richer features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        cnn_out = 96 * 2  # 192 from avg+max pooling
        
        # ── Tabular branch ──
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tabular, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(),
        )
        tab_out = 32
        
        # ── Fusion ──
        fusion_in = cnn_out + tab_out  # 192 + 32 = 224
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 96), nn.BatchNorm1d(96), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(96, 48), nn.BatchNorm1d(48), nn.SiLU(), nn.Dropout(0.15),
        )
        
        # ── Dual heads ──
        self.occ_head = nn.Linear(48, 1)           # Occurrence logit (train with BCE)
        self.int_head = nn.Sequential(
            nn.Linear(48, 1), nn.Softplus()         # Intensity (always ≥ 0)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, patch, tabular):
        # CNN spatial features
        cnn_feat = self.cnn(patch)
        avg_feat = self.avg_pool(cnn_feat).flatten(1)  # (B, 96)
        max_feat = self.max_pool(cnn_feat).flatten(1)  # (B, 96)
        spatial = torch.cat([avg_feat, max_feat], dim=1)  # (B, 192)
        
        # Tabular features
        tab_feat = self.tab_mlp(tabular)  # (B, 32)
        
        # Fuse
        fused = self.fusion(torch.cat([spatial, tab_feat], dim=1))  # (B, 48)
        
        # Dual heads
        occ_logit = self.occ_head(fused).squeeze(-1)       # (B,) — rain probability logit
        intensity = self.int_head(fused).squeeze(-1)        # (B,) — predicted mm
        
        # Final prediction = intensity (gated by occurrence during eval)
        pred = intensity
        
        return occ_logit, intensity, pred
    
    def predict(self, patch, tabular):
        """Inference: gate intensity by occurrence probability."""
        occ_logit, intensity, _ = self.forward(patch, tabular)
        occ_prob = torch.sigmoid(occ_logit)
        # Soft gating: scale intensity by occurrence probability
        return intensity * occ_prob
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SpatialNet trainable parameters: {total:,}")
        return total


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def build_model(window_size=9, n_channels=19, n_tabular=24):
    """Build SpatialNet model."""
    model = SpatialNet(n_channels=n_channels, n_tabular=n_tabular, window_size=window_size)
    model.count_params()
    return model
