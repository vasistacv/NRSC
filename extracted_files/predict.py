"""
predict.py
==========
Run inference with a trained DualPathNet checkpoint on new ECMWF data.

Usage:
    python predict.py --checkpoint path/to/ckpt.pt --year 2024 --month 7
    python predict.py --checkpoint path/to/ckpt.pt --csv output_preds.csv

Outputs:
  - Predictions for all ground-truth station locations for the specified period
  - CSV with columns: Date, Station, Lat, Lon, Pred_mm, Obs_mm (if available)
  - Prints summary metrics if observations are available
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

import config
from dataset  import RainfallDataBuilder, Normaliser, RainfallDataset
from model    import build_model, DualPathNet
from metrics  import evaluate, print_metrics
from torch.utils.data import DataLoader


def load_checkpoint(ckpt_path: str, device: str) -> DualPathNet:
    """Load a saved DualPathNet checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Infer window size from checkpoint path name
    window_size = 3
    if "window_5" in ckpt_path or "window5" in ckpt_path:
        window_size = 5

    # Build model with same architecture
    model = build_model(window_size=window_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    score   = ckpt.get("score", "?")
    metrics = ckpt.get("metrics", {})
    print(f"Loaded checkpoint: epoch={epoch}  score={score:.4f}")
    if metrics:
        csi_p90 = metrics.get("val_CSI_p90", "?")
        print(f"  Val CSI_P90 = {csi_p90}")

    return model, window_size


def predict_period(
    model: DualPathNet,
    window_size: int,
    years: list,
    norm_path: str,
    device: str,
    output_csv: str = None
):
    """Generate predictions for a list of years."""
    builder = RainfallDataBuilder(window_size=window_size)
    patches, tabular, targets = builder.build(years)

    # Load normaliser fitted on training data
    norm = Normaliser()
    norm.load(Path(norm_path))
    patches_n = norm.transform_patches(patches)
    tabular_n = norm.transform_tabular(tabular)

    dataset = RainfallDataset(patches_n, tabular_n, targets)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for patch, tab, _ in loader:
            patch = patch.to(device)
            tab   = tab.to(device)
            pred  = model.predict(patch, tab)
            all_preds.append(pred.cpu().numpy())

    preds = np.concatenate(all_preds)

    # Compute metrics
    # Get thresholds from training data reference
    rainy = targets[targets >= config.DRY_THRESHOLD]
    thresholds = {
        "p90": float(np.percentile(rainy, 90)),
        "p95": float(np.percentile(rainy, 95)),
        "p99": float(np.percentile(rainy, 99)),
    }

    metrics = evaluate(preds, targets, thresholds, prefix="pred")
    print_metrics(metrics, title=f"Prediction metrics (years={years})")

    # Save CSV if requested
    if output_csv:
        # Re-build GT rows to get station metadata
        gt = pd.read_csv(config.GROUND_TRUTH, parse_dates=["Date"])
        gt = gt.dropna(subset=["Rainfall_mm"])
        gt = gt[gt["Date"].dt.month.isin(config.MONSOON_MONTHS)]
        gt = gt[gt["Date"].dt.year.isin(years)]
        gt = gt.reset_index(drop=True)

        if len(gt) == len(preds):
            gt["Pred_mm"] = preds
            gt.to_csv(output_csv, index=False)
            print(f"Predictions saved to: {output_csv}")
        else:
            print(f"[WARN] Row count mismatch: gt={len(gt)}  preds={len(preds)}. CSV not saved.")

    return preds, targets, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--norm",       required=True, help="Path to normaliser.npz")
    parser.add_argument("--years",      nargs="+", type=int, default=[2024])
    parser.add_argument("--csv",        default=None)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, window_size = load_checkpoint(args.checkpoint, device)

    predict_period(
        model=model,
        window_size=window_size,
        years=args.years,
        norm_path=args.norm,
        device=device,
        output_csv=args.csv,
    )
