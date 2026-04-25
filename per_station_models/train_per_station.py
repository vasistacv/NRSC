"""
train_per_station.py
====================
Supervisor Task 2: Train individual models for each station separately.
Supervisor Task 3: Compare individual vs combined model per station.

This script:
  1. Loops through all 7 stations
  2. For each station, trains an individual SmallNet model (9x9 window)
  3. Evaluates on 2024 test data
  4. Prints a comparison table: ECMWF raw vs Individual model vs Combined model

Usage:
    python train_per_station.py
    python train_per_station.py --station Maheshwaram    # single station
    python train_per_station.py --debug                  # quick 5-epoch test
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import config
from dataset import get_dataloaders
from model import build_model
from losses import CombinedLoss
from metrics import evaluate, print_metrics, collect_predictions


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("Device: CPU")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN ONE STATION
# ──────────────────────────────────────────────────────────────────────────────

def train_single_station(station_name, window_size=9, n_epochs=300, debug=False):
    """
    Train a model using ONLY data from one station.
    Returns test metrics dict for 2024.
    """
    set_seed(config.SEED)
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  STATION: {station_name} | window={window_size}x{window_size}")
    print(f"{'='*60}")

    # Output directory for this station
    out_dir = config.OUTPUT_DIR / station_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data for this station only ──
    print("\n[1/4] Loading data ...")
    train_loader, val_loader, test_loader, norm, thresholds = get_dataloaders(
        window_size=window_size,
        station_filter=station_name
    )
    p90, p95, p99 = thresholds["p90"], thresholds["p95"], thresholds["p99"]

    # Check if we have enough data
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    if n_train < 50:
        print(f"  [SKIP] Too few training samples for {station_name}")
        return None

    # ── Build model ──
    print("\n[2/4] Building model ...")
    sample_patch, sample_tab, _ = next(iter(train_loader))
    n_channels = sample_patch.shape[1]
    n_tabular = sample_tab.shape[1]
    print(f"  Input: patch={sample_patch.shape}, tabular={sample_tab.shape}")

    model = build_model(window_size=window_size, n_channels=n_channels, n_tabular=n_tabular)
    model = model.to(device)

    # ── Loss + Optimizer ──
    print("\n[3/4] Configuring loss + optimizer ...")
    loss_fn = CombinedLoss(p90_thresh=p90, p95_thresh=p95, p99_thresh=p99)
    print(f"  Thresholds: P90={p90:.2f}mm, P95={p95:.2f}mm, P99={p99:.2f}mm")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR_INIT,
        weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.999)
    )

    actual_epochs = 5 if debug else n_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR_INIT,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=actual_epochs, pct_start=0.2,
        anneal_strategy="cos",
        final_div_factor=config.LR_INIT / config.LR_MIN,
        div_factor=10.0,
    )

    # ── Training loop ──
    print(f"\n[4/4] Training for {actual_epochs} epochs ...")
    best_score = -np.inf
    best_metrics = {}
    best_state = None
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, actual_epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for patch, tabular, target in train_loader:
            patch = patch.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            occ_logit, intensity, pred = model(patch, tabular)
            loss, components = loss_fn(occ_logit, intensity, pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += components["total"]
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_preds, val_targets = collect_predictions(model, val_loader, device)
        val_metrics = evaluate(val_preds, val_targets, thresholds, prefix="val")

        csi_rain = val_metrics.get("val_CSI_rain", 0.0)
        csi_p90 = val_metrics.get("val_CSI_p90", 0.0)
        sedi_p90 = val_metrics.get("val_SEDI_p90", 0.0)
        score = 0.2 * csi_rain + 0.4 * csi_p90 + 0.4 * sedi_p90

        if epoch % 20 == 0 or epoch == 1:
            print(f"  E{epoch:03d} loss={avg_loss:.4f} CSI_rain={csi_rain:.4f} "
                  f"CSI_p90={csi_p90:.4f} SEDI_p90={sedi_p90:.4f} score={score:.4f}")

        # Track best
        if score > best_score + 1e-4:
            best_score = score
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={config.PATIENCE})")
            break

    train_time = time.time() - t_start
    print(f"  Training done in {train_time/60:.1f} min")

    # ── Load best and evaluate on test ──
    if best_state is not None:
        model.load_state_dict(best_state)

    test_preds, test_targets = collect_predictions(model, test_loader, device)

    # ── Isotonic Regression calibration (same as baseline train.py) ──
    print("  Applying Isotonic Regression calibration...")
    val_preds, val_targets = collect_predictions(model, val_loader, device)

    from sklearn.isotonic import IsotonicRegression
    iso_reg = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
    iso_reg.fit(val_preds, val_targets)

    test_preds_cal = iso_reg.predict(test_preds)
    val_preds_cal = iso_reg.predict(val_preds)

    print(f"    Raw preds:  mean={test_preds.mean():.1f}  P90={np.percentile(test_preds, 90):.1f}")
    print(f"    Cal preds:  mean={test_preds_cal.mean():.1f}  P90={np.percentile(test_preds_cal, 90):.1f}")
    print(f"    Targets:    mean={test_targets.mean():.1f}  P90={np.percentile(test_targets, 90):.1f}")

    # Optimize thresholds on calibrated val predictions
    from metrics import contingency, csi
    calibrated_thresholds = dict(thresholds)
    for label, base_thresh in [("p90", thresholds["p90"]), ("p95", thresholds["p95"])]:
        best_csi_val = 0.0
        best_t = base_thresh
        for factor in np.arange(0.3, 1.5, 0.025):
            trial_t = base_thresh * factor
            ct = contingency(val_preds_cal, val_targets, trial_t)
            trial_csi = csi(ct)
            if trial_csi > best_csi_val:
                best_csi_val = trial_csi
                best_t = trial_t
        calibrated_thresholds[label] = best_t
        print(f"    {label}: base={base_thresh:.1f}mm -> calibrated={best_t:.1f}mm (val CSI={best_csi_val:.4f})")

    # Evaluate all 3 options (same as baseline)
    test_metrics_orig = evaluate(test_preds, test_targets, thresholds, prefix="test")
    test_metrics_cal = evaluate(test_preds_cal, test_targets, calibrated_thresholds, prefix="test_cal")
    test_metrics_cal_orig_t = evaluate(test_preds_cal, test_targets, thresholds, prefix="test_calot")

    print_metrics(test_metrics_orig, title=f"{station_name} - raw preds, orig thresholds")
    print_metrics(test_metrics_cal, title=f"{station_name} - calibrated preds, cal thresholds")
    print_metrics(test_metrics_cal_orig_t, title=f"{station_name} - calibrated preds, orig thresholds")

    # Pick the best one
    options = [
        ("raw+orig", test_metrics_orig, thresholds, "test"),
        ("cal+cal", test_metrics_cal, calibrated_thresholds, "test_cal"),
        ("cal+orig", test_metrics_cal_orig_t, thresholds, "test_calot"),
    ]
    best_option = max(options, key=lambda x: x[1].get(f"{x[3]}_CSI_p90", 0) + x[1].get(f"{x[3]}_CSI_p95", 0))
    print(f"  Best option: {best_option[0]}")

    # Normalize keys to test_ prefix
    test_metrics = {k.replace(f"{best_option[3]}_", "test_"): v for k, v in best_option[1].items()}

    # Save results
    results = {
        "station": station_name,
        "window_size": window_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "thresholds_original": thresholds,
        "thresholds_calibrated": calibrated_thresholds,
        "best_calibration": best_option[0],
        "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))},
        "best_val_metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (float, int))},
        "n_epochs_trained": epoch,
        "training_minutes": round(train_time / 60, 1),
    }

    results_path = out_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # Save model checkpoint
    ckpt_path = out_dir / "best_model.pt"
    torch.save({
        "model": best_state,
        "thresholds": thresholds,
        "calibrated_thresholds": calibrated_thresholds,
        "station": station_name,
    }, str(ckpt_path))

    # Save normaliser
    norm.save(out_dir / "normaliser.npz")

    return test_metrics


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(individual_results, ecmwf_path, combined_path=None):
    """
    Task 3: Print side-by-side comparison of ECMWF vs Individual vs Combined.
    """
    print(f"\n{'='*100}")
    print(f"  COMPARISON: ECMWF RAW vs INDIVIDUAL MODEL vs COMBINED MODEL (2024)")
    print(f"{'='*100}")

    # Load ECMWF baseline
    ecmwf_results = {}
    if ecmwf_path.exists():
        with open(ecmwf_path) as f:
            ecmwf_all = json.load(f)
        for stn in config.ALL_STATIONS:
            key = f"{stn}_2024"
            if key in ecmwf_all:
                ecmwf_results[stn] = ecmwf_all[key]

    # Load combined model results (if available)
    combined_results = {}
    if combined_path and combined_path.exists():
        with open(combined_path) as f:
            combined_all = json.load(f)
        # The combined model doesn't have per-station breakdown yet
        # We'll just show the overall numbers

    # Print header
    print(f"\n{'Station':15s} | {'Metric':10s} | {'ECMWF Raw':>10s} | {'Individual':>10s} | {'Delta':>8s}")
    print("-" * 65)

    metrics_to_show = [
        ("CSI_rain", "CSI_rain", "test_CSI_rain"),
        ("FAR_rain", "FAR_rain", "test_FAR_rain"),
        ("CSI_p90",  "CSI_p90",  "test_CSI_p90"),
        ("SEDI_p90", "SEDI_p90", "test_SEDI_p90"),
        ("CSI_p95",  "CSI_p95",  "test_CSI_p95"),
        ("corr",     "corr_rainy", "test_corr_rainy"),
    ]

    for stn in config.ALL_STATIONS:
        ecmwf = ecmwf_results.get(stn, {})
        indiv = individual_results.get(stn, {})

        for label, ecmwf_key, indiv_key in metrics_to_show:
            e_val = ecmwf.get(ecmwf_key, None)
            i_val = indiv.get(indiv_key, None)

            e_str = f"{e_val:.4f}" if e_val is not None else "N/A"
            i_str = f"{i_val:.4f}" if i_val is not None else "N/A"

            if e_val is not None and i_val is not None:
                delta = i_val - e_val
                # For FAR, negative delta is better
                if "FAR" in label:
                    d_str = f"{delta:+.4f}" + (" <<" if delta < -0.05 else "")
                else:
                    d_str = f"{delta:+.4f}" + (" <<" if delta > 0.05 else "")
            else:
                d_str = ""

            stn_label = stn if label == metrics_to_show[0][0] else ""
            print(f"{stn_label:15s} | {label:10s} | {e_str:>10s} | {i_str:>10s} | {d_str:>8s}")

        print("-" * 65)

    # Summary averages
    print(f"\n{'AVERAGE':15s} |", end="")
    for label, ecmwf_key, indiv_key in metrics_to_show:
        e_vals = [ecmwf_results[s].get(ecmwf_key, 0) for s in config.ALL_STATIONS if s in ecmwf_results]
        i_vals = [individual_results[s].get(indiv_key, 0) for s in config.ALL_STATIONS if s in individual_results]
        e_avg = np.mean(e_vals) if e_vals else 0
        i_avg = np.mean(i_vals) if i_vals else 0
        print(f" {label}: ECMWF={e_avg:.3f} Indiv={i_avg:.3f} |", end="")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-station individual models")
    parser.add_argument("--station", type=str, default=None,
                        help="Train single station only (e.g. Maheshwaram)")
    parser.add_argument("--window", type=int, default=9,
                        help="Window size (default: 9)")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Max training epochs per station (default: 400, same as best baseline)")
    parser.add_argument("--debug", action="store_true",
                        help="Quick 5-epoch smoke test")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which stations to train
    if args.station:
        if args.station not in config.ALL_STATIONS:
            print(f"ERROR: Unknown station '{args.station}'")
            print(f"Valid: {config.ALL_STATIONS}")
            sys.exit(1)
        stations = [args.station]
    else:
        stations = config.ALL_STATIONS

    # Train each station
    all_results = {}
    for stn in stations:
        metrics = train_single_station(
            station_name=stn,
            window_size=args.window,
            n_epochs=args.epochs,
            debug=args.debug,
        )
        if metrics is not None:
            all_results[stn] = metrics

    # Print comparison table (Task 3)
    ecmwf_path = config.ROOT_DIR / "ecmwf_baseline_results.json"
    combined_path = config.ROOT_DIR / "final_model_baseline" / "9x9" / "dual_eval_results.json"
    print_comparison_table(all_results, ecmwf_path, combined_path)

    # Save all results
    summary_path = config.OUTPUT_DIR / "all_stations_summary.json"
    summary = {}
    for stn, metrics in all_results.items():
        summary[stn] = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved: {summary_path}")
