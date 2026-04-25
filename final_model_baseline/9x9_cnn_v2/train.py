"""
train.py — SpatialNet CNN Training Pipeline
=============================================
Trains the SpatialNet CNN with:
  - Cosine annealing with warm restarts
  - Dual-head loss (occurrence BCE + intensity Tweedie + Huber)
  - Weighted sampling for extreme events
  - Best checkpoint saving based on validation CSI_rain
"""

import sys, json, warnings, time, gc
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import RainfallDataBuilder, RainfallDataset, Normaliser, build_weighted_sampler
from model import build_model
from losses import DualHeadLoss
from metrics import evaluate, print_metrics, collect_predictions


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        patch, tabular, target = [b.to(device, non_blocking=True) for b in batch]
        
        occ_logit, intensity, pred = model(patch, tabular)
        loss, _ = criterion(occ_logit, intensity, pred, target)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        patch, tabular, target = [b.to(device, non_blocking=True) for b in batch]
        occ_logit, intensity, pred = model(patch, tabular)
        loss, _ = criterion(occ_logit, intensity, pred, target)
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    set_seed(config.SEED)
    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  SpatialNet CNN Training Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load data ──
    print("[1/4] Loading data...")
    builder = RainfallDataBuilder(window_size=config.DEFAULT_WINDOW)
    
    tr_patches, tr_tabular, tr_targets = builder.build(config.TRAIN_YEARS)
    vl_patches, vl_tabular, vl_targets = builder.build(config.VAL_YEARS)
    te_patches, te_tabular, te_targets = builder.build(config.TEST_YEARS)
    
    # Normalise
    norm = Normaliser()
    norm.fit(tr_patches, tr_tabular)
    
    out_dir = config.OUTPUT_DIR / f"window_{config.DEFAULT_WINDOW}"
    out_dir.mkdir(parents=True, exist_ok=True)
    norm.save(out_dir / "normaliser.npz")
    
    tr_p = norm.transform_patches(tr_patches)
    tr_t = norm.transform_tabular(tr_tabular)
    vl_p = norm.transform_patches(vl_patches)
    vl_t = norm.transform_tabular(vl_tabular)
    te_p = norm.transform_patches(te_patches)
    te_t = norm.transform_tabular(te_tabular)
    
    tr_ds = RainfallDataset(tr_p, tr_t, tr_targets)
    vl_ds = RainfallDataset(vl_p, vl_t, vl_targets)
    te_ds = RainfallDataset(te_p, te_t, te_targets)
    
    sampler, p90, p95, p99 = build_weighted_sampler(tr_targets)
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    
    train_loader = DataLoader(tr_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(vl_ds, batch_size=config.BATCH_SIZE * 2,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)
    test_loader = DataLoader(te_ds, batch_size=config.BATCH_SIZE * 2,
                             shuffle=False, num_workers=config.NUM_WORKERS,
                             pin_memory=True)
    
    print(f"\n  Train: {len(tr_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(vl_ds)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(te_ds)} samples, {len(test_loader)} batches")
    print(f"  P90={p90:.1f}mm  P95={p95:.1f}mm  P99={p99:.1f}mm")
    
    # ── Build model ──
    print(f"\n[2/4] Building SpatialNet (window={config.DEFAULT_WINDOW})...")
    model = build_model(window_size=config.DEFAULT_WINDOW, n_channels=19, n_tabular=24)
    model = model.to(device)
    
    # ── Loss, optimizer, scheduler ──
    criterion = DualHeadLoss(p90_thresh=p90, p95_thresh=p95, p99_thresh=p99)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR_INIT,
        weight_decay=config.WEIGHT_DECAY)
    
    # Cosine annealing with warm restarts (T_0=50 epochs, restart every 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=config.LR_MIN)
    
    # ── Training loop ──
    print(f"\n[3/4] Training for {config.NUM_EPOCHS} epochs...")
    best_val_csi = -1
    best_epoch = 0
    patience_counter = 0
    history = []
    saved_checkpoints = []
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, config.GRAD_CLIP)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Evaluate on val set
        preds_vl, targets_vl = collect_predictions(model, val_loader, device)
        val_metrics = evaluate(preds_vl, targets_vl, thresholds)
        
        val_csi = val_metrics.get("CSI_rain", 0)
        val_csi_p90 = val_metrics.get("CSI_p90", 0)
        val_far = val_metrics.get("FAR_rain", 1)
        val_sedi = val_metrics.get("SEDI_rain", 0)
        
        # Composite score for checkpoint selection
        score = val_csi + val_csi_p90 * 2.0 + val_sedi * 0.5 - val_far * 1.0
        
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        
        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_CSI_rain": val_csi, "val_CSI_p90": val_csi_p90,
            "val_FAR_rain": val_far, "val_SEDI_rain": val_sedi,
            "score": score, "lr": lr_now,
        }
        history.append(record)
        
        if epoch % config.LOG_INTERVAL == 0 or epoch <= 5:
            print(f"  Epoch {epoch:4d}/{config.NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"CSI: {val_csi:.4f} | P90: {val_csi_p90:.4f} | "
                  f"FAR: {val_far:.4f} | SEDI: {val_sedi:.4f} | "
                  f"LR: {lr_now:.2e} | {elapsed:.1f}s")
        
        # Save best checkpoint
        if score > best_val_csi:
            best_val_csi = score
            best_epoch = epoch
            patience_counter = 0
            
            ckpt_path = out_dir / f"ckpt_epoch{epoch:04d}_csi{val_csi:.4f}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "score": score,
            }, str(ckpt_path))
            saved_checkpoints.append(ckpt_path)
            
            # Keep only top N checkpoints
            if len(saved_checkpoints) > config.SAVE_BEST_N:
                old = saved_checkpoints.pop(0)
                if old.exists():
                    old.unlink()
            
            if epoch % config.LOG_INTERVAL != 0:
                print(f"  * Epoch {epoch:4d} | NEW BEST score={score:.4f} | "
                      f"CSI={val_csi:.4f} | CSI_P90={val_csi_p90:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={config.PATIENCE})")
            break
    
    # ── Final evaluation ──
    print(f"\n[4/4] Final evaluation (best epoch={best_epoch})...")
    
    # Load best checkpoint
    best_ckpt = saved_checkpoints[-1]
    ckpt = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Test set evaluation
    preds_te, targets_te = collect_predictions(model, test_loader, device)
    test_metrics = evaluate(preds_te, targets_te, thresholds)
    print_metrics(test_metrics, title="TEST SET RESULTS (2024)")
    
    # Correlation
    rainy = targets_te >= config.DRY_THRESHOLD
    if rainy.sum() > 2:
        corr = float(np.corrcoef(preds_te[rainy], targets_te[rainy])[0, 1])
        print(f"  corr_rainy = {corr:.4f}")
    
    # Save history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save test results
    test_results = {
        "best_epoch": best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))},
        "thresholds": thresholds,
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n  All outputs saved to: {out_dir}")
    print(f"  Best checkpoint: {best_ckpt.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
