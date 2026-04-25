"""
model.py
========
Optimized architectures for extreme rainfall downscaling over Telangana.

Three models available:
  1. SmallNet     — Proven MLP baseline (128→64→32), best for tabular features
  2. PatchMLP     — Flattened patch + tabular, no CNN overhead
  3. DualPathNet  — CNN spatial + MLP tabular with lightweight fusion

With only ~6000 training samples, simpler models generalize better.
The SmallNet architecture previously achieved the best results on this dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ──────────────────────────────────────────────────────────────────────────────
# 1. SmallNet — Proven best performer for this dataset size
# ──────────────────────────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    """
    Compact MLP — prevents overfitting on ~6K samples.
    Uses flattened patch + tabular features as combined input.
    Previously achieved best CSI/SEDI scores on this exact dataset.
    """
    def __init__(self, n_in):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.act = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.d1  = nn.Dropout(0.25)
        self.d2  = nn.Dropout(0.20)
        self.sp  = nn.Softplus()

    def forward(self, patch, tabular):
        # Flatten patch and concatenate with tabular
        B = patch.size(0)
        flat_patch = patch.view(B, -1)
        x = torch.cat([flat_patch, tabular], dim=-1)

        h = self.d1(self.act(self.bn1(self.fc1(x))))
        h = self.d2(self.act(self.bn2(self.fc2(h))))
        h = self.act(self.fc3(h))
        pred = self.sp(self.out(h)).squeeze(-1)

        # For compatibility with dual-head loss, create a dummy occ_logit
        # occ_logit > 0 when pred > 0.5mm (approx)
        occ_logit = (pred - 0.5) * 5.0
        return occ_logit, pred, pred

    def predict(self, patch, tabular):
        _, _, pred = self.forward(patch, tabular)
        return pred

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SmallNet trainable parameters: {total:,}")
        return total


# ──────────────────────────────────────────────────────────────────────────────
# 2. PatchMLP — Slightly larger, with residual connections
# ──────────────────────────────────────────────────────────────────────────────

class PatchMLP(nn.Module):
    """
    Flatten patch + tabular → deeper MLP with skip connections.
    Good middle ground between SmallNet and DualPathNet.
    """
    def __init__(self, n_in):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.30)
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.20)
        )
        # Skip projection
        self.skip = nn.Linear(256, 64)
        self.out = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def forward(self, patch, tabular):
        B = patch.size(0)
        x = torch.cat([patch.view(B, -1), tabular], dim=-1)

        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2) + self.skip(h1)  # residual
        pred = self.out(h3).squeeze(-1)

        occ_logit = (pred - 0.5) * 5.0
        return occ_logit, pred, pred

    def predict(self, patch, tabular):
        _, _, pred = self.forward(patch, tabular)
        return pred

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PatchMLP trainable parameters: {total:,}")
        return total


# ──────────────────────────────────────────────────────────────────────────────
# 3. DualPathNet — CNN + MLP (kept for comparison but may overfit)
# ──────────────────────────────────────────────────────────────────────────────

class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False), nn.SiLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class DualPathNet(nn.Module):
    def __init__(self, n_channels=19, n_tabular=24, window_size=3,
                 base_ch=32, cnn_depth=2, cnn_dropout=0.15,
                 mlp_hidden=(128, 64), mlp_dropout=0.20,
                 fusion_dim=64, fusion_dropout=0.15):
        super().__init__()
        # Lightweight CNN
        self.cnn = nn.Sequential(
            ConvBNSiLU(n_channels, base_ch),
            SEBlock(base_ch),
            nn.Dropout2d(cnn_dropout),
            ConvBNSiLU(base_ch, base_ch * 2, kernel=1, padding=0),
            SEBlock(base_ch * 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        cnn_out = base_ch * 2

        # MLP for tabular
        self.mlp = nn.Sequential(
            nn.Linear(n_tabular, mlp_hidden[0]), nn.BatchNorm1d(mlp_hidden[0]),
            nn.SiLU(), nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]), nn.BatchNorm1d(mlp_hidden[1]),
            nn.SiLU(), nn.Dropout(mlp_dropout)
        )

        # Fusion
        fusion_in = cnn_out + mlp_hidden[-1]
        self.head = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim), nn.BatchNorm1d(fusion_dim),
            nn.SiLU(), nn.Dropout(fusion_dropout),
            nn.Linear(fusion_dim, 1), nn.Softplus()
        )

    def forward(self, patch, tabular):
        cnn_emb = self.cnn(patch)
        mlp_emb = self.mlp(tabular)
        fused = torch.cat([cnn_emb, mlp_emb], dim=-1)
        pred = self.head(fused).squeeze(-1)
        occ_logit = (pred - 0.5) * 5.0
        return occ_logit, pred, pred

    def predict(self, patch, tabular):
        _, _, pred = self.forward(patch, tabular)
        return pred

    def count_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DualPathNet trainable parameters: {total:,}")
        return total


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def build_model(window_size=3, n_channels=19, n_tabular=24):
    """Build model based on config.MODEL_TYPE."""
    model_type = getattr(config, 'MODEL_TYPE', 'SmallNet')

    if model_type == "SmallNet":
        n_in = n_channels * window_size * window_size + n_tabular
        model = SmallNet(n_in)
    elif model_type == "PatchMLP":
        n_in = n_channels * window_size * window_size + n_tabular
        model = PatchMLP(n_in)
    elif model_type == "DualPathNet":
        model = DualPathNet(
            n_channels=n_channels, n_tabular=n_tabular,
            window_size=window_size,
            base_ch=config.CNN_BASE_CHANNELS,
            cnn_depth=config.CNN_DEPTH,
            cnn_dropout=config.CNN_DROPOUT,
            mlp_hidden=config.MLP_HIDDEN[:2],
            mlp_dropout=config.MLP_DROPOUT,
            fusion_dim=config.FUSION_HIDDEN,
            fusion_dropout=config.FUSION_DROPOUT
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

    model.count_params()
    return model
