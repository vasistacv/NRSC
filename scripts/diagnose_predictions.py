"""Quick diagnostic: check prediction Distribution vs actual."""
import sys
sys.path.insert(0, r"D:\NEW_NRSC\extracted_files")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path

import config
from dataset import get_dataloaders
from model import build_model
from metrics import collect_predictions

device = "cpu"

print("Loading data...")
train_loader, val_loader, test_loader, norm, thresholds = get_dataloaders(window_size=3)

# Load best model
out_dir = config.OUTPUT_DIR / "window_3"
import glob
ckpts = sorted(out_dir.glob("ckpt_*.pt"))
if ckpts:
    best = ckpts[-1]
    print(f"Loading: {best.name}")
    
    n_channels = 19
    n_tabular = 24
    model = build_model(window_size=3, n_channels=n_channels, n_tabular=n_tabular)
    ckpt = torch.load(str(best), map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    
    preds, targets = collect_predictions(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"  PREDICTION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  Targets (actual rainfall):")
    print(f"    min={targets.min():.1f}  max={targets.max():.1f}  mean={targets.mean():.1f}")
    print(f"    P50={np.median(targets):.1f}  P90={np.percentile(targets, 90):.1f}  P95={np.percentile(targets, 95):.1f}")
    
    print(f"\n  Predictions:")
    print(f"    min={preds.min():.1f}  max={preds.max():.1f}  mean={preds.mean():.1f}")
    print(f"    P50={np.median(preds):.1f}  P90={np.percentile(preds, 90):.1f}  P95={np.percentile(preds, 95):.1f}")
    
    # Rainy days only
    rainy = targets >= 0.1
    print(f"\n  Predictions on RAINY days (n={rainy.sum()}):")
    print(f"    min={preds[rainy].min():.1f}  max={preds[rainy].max():.1f}  mean={preds[rainy].mean():.1f}")
    print(f"    P90={np.percentile(preds[rainy], 90):.1f}  P95={np.percentile(preds[rainy], 95):.1f}")
    
    # Extreme days
    extreme = targets >= 35.0
    print(f"\n  Predictions on P90+ days (actual>=35mm, n={extreme.sum()}):")
    if extreme.sum() > 0:
        print(f"    min={preds[extreme].min():.1f}  max={preds[extreme].max():.1f}  mean={preds[extreme].mean():.1f}")
        print(f"    Actual mean: {targets[extreme].mean():.1f}")
        print(f"    Pred/Actual ratio: {preds[extreme].mean()/targets[extreme].mean():.2f}")
    
    # What threshold would give best CSI?
    print(f"\n  CSI optimization scan:")
    from metrics import contingency, csi, far, pod
    for t in [5, 10, 15, 17.5, 20, 25, 30, 35, 40]:
        ct = contingency(preds, targets, t)
        print(f"    thresh={t:5.1f}mm  CSI={csi(ct):.4f}  POD={pod(ct):.4f}  FAR={far(ct):.4f}  H={ct['H']:3d}  M={ct['M']:3d}  FA={ct['FA']:3d}")
