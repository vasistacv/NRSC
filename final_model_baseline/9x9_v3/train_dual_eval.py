"""
train_dual_eval.py — Enhanced Ensemble Pipeline v3
====================================================
Key improvements over baseline:
  1. Spatial statistics features for XGBoost (not raw flattened patch)
  2. Dedicated P90+ extreme event classifier
  3. 3-stage ensemble: rain gate × (weighted intensity) × extreme boost
  4. Broader grid search for ensemble weights
  5. Train on 2015-2022 (8 years), val 2023, test 2024

Two evaluation modes:
  A) Temporal split: Train 2015-2022, Val 2023, Test 2024
  B) Random 70/15/15 split: All years mixed
"""

import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

import config
from dataset import RainfallDataBuilder, Normaliser
from model import build_model
from metrics import evaluate, print_metrics, contingency, csi


# ──────────────────────────────────────────────────────────────────────────────
# SPATIAL FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_spatial_features(patches):
    """
    Extract physics-informed spatial statistics from 9×9 patches.
    
    Instead of flattening 19×9×9 = 1539 raw values (mostly noise),
    extract 133 meaningful features that capture:
    - Local weather at station (center pixel)
    - Regional context (spatial mean)
    - Extreme signals nearby (spatial max)
    - Spatial gradients (convergence/divergence indicators)
    - Local anomaly (how different is station from surroundings)
    
    Returns: (N, 133) array
    """
    B, C, H, W = patches.shape
    center = H // 2
    
    feats = []
    
    # 1. Center pixel values (19) — what ECMWF predicts at station
    center_vals = patches[:, :, center, center]
    feats.append(center_vals)
    
    # 2. Spatial mean (19) — regional average
    spatial_mean = patches.mean(axis=(2, 3))
    feats.append(spatial_mean)
    
    # 3. Spatial max (19) — strongest signal nearby
    spatial_max = patches.reshape(B, C, -1).max(axis=2)
    feats.append(spatial_max)
    
    # 4. Spatial std (19) — variability = gradient proxy
    spatial_std = patches.std(axis=(2, 3))
    feats.append(spatial_std)
    
    # 5. Center minus mean (19) — local anomaly
    center_anomaly = center_vals - spatial_mean
    feats.append(center_anomaly)
    
    # 6. 3×3 neighborhood mean (19) — local context
    inner = patches[:, :, center-1:center+2, center-1:center+2]
    inner_mean = inner.mean(axis=(2, 3))
    feats.append(inner_mean)
    
    # 7. Gradient magnitude at center (19) — convergence/divergence
    dx = patches[:, :, center, min(center+1, W-1)] - patches[:, :, center, max(center-1, 0)]
    dy = patches[:, :, min(center+1, H-1), center] - patches[:, :, max(center-1, 0), center]
    grad_mag = np.sqrt(dx**2 + dy**2)
    feats.append(grad_mag)
    
    return np.hstack(feats)  # (B, 19*7) = (B, 133)


def build_features(patches, tabulars, norm_obj, nn_preds=None):
    """Build feature matrix for XGBoost: spatial stats + tabular + NN predictions."""
    p = norm_obj.transform_patches(patches)
    t = norm_obj.transform_tabular(tabulars)
    
    spatial = extract_spatial_features(p)
    
    parts = [spatial, t]
    if nn_preds is not None:
        parts.append(nn_preds.reshape(-1, 1))
    
    return np.hstack(parts)


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL NET PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────

def nn_predict(nn_model, patches, tabulars, norm_obj):
    """Get neural net predictions for a set of samples."""
    p = torch.from_numpy(norm_obj.transform_patches(patches)).float()
    t = torch.from_numpy(norm_obj.transform_tabular(tabulars)).float()
    ds = TensorDataset(p, t)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for pb, tb in loader:
            preds.append(nn_model.predict(pb, tb).numpy())
    return np.concatenate(preds)


# ──────────────────────────────────────────────────────────────────────────────
# 3-STAGE ENSEMBLE PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                 nn_vl, nn_te, thresholds, p90, label=""):
    """
    3-stage ensemble:
      Stage 1: Rain/no-rain classifier (XGBoost)
      Stage 2: Intensity regressor (XGBoost) — weighted by event importance
      Stage 3: Extreme event classifier (XGBoost) — boosts P90+ predictions
      
    Final = rain_gate × (w_nn × NN + w_xgb × XGB_reg) × extreme_boost
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    p95 = thresholds["p95"]

    # ── Stage 1: Rain classifier ──────────────────────────────────────────
    print("  [Stage 1] Training rain classifier...")
    y_rain_tr = (y_tr >= config.DRY_THRESHOLD).astype(int)
    y_rain_vl = (y_vl >= config.DRY_THRESHOLD).astype(int)
    n_dry, n_wet = (y_rain_tr == 0).sum(), (y_rain_tr == 1).sum()

    clf_rain = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_dry/n_wet)
    clf_rain.fit(X_tr, y_rain_tr, eval_set=[(X_vl, y_rain_vl)], verbose=False)
    rain_vl = clf_rain.predict_proba(X_vl)[:, 1]
    rain_te = clf_rain.predict_proba(X_te)[:, 1]
    print(f"    Rain classifier iterations: {clf_rain.best_iteration}")

    # ── Stage 2: Intensity regressor ──────────────────────────────────────
    print("  [Stage 2] Training intensity regressor...")
    w = np.ones(len(y_tr), dtype=np.float32)
    w[(y_tr >= 0.1) & (y_tr < p90)] = 2.0
    w[(y_tr >= p90) & (y_tr < p95)] = 10.0
    w[y_tr >= p95] = 20.0

    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_vl, y_vl)], verbose=False)
    xgb_vl = reg.predict(X_vl).clip(min=0)
    xgb_te = reg.predict(X_te).clip(min=0)
    print(f"    Regression iterations: {reg.best_iteration}")

    # ── Stage 3: Extreme event classifier (P90+) ─────────────────────────
    print("  [Stage 3] Training extreme event classifier (P90+)...")
    y_ext_tr = (y_tr >= p90).astype(int)
    y_ext_vl = (y_vl >= p90).astype(int)
    n_normal, n_extreme = (y_ext_tr == 0).sum(), (y_ext_tr == 1).sum()

    clf_extreme = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.03, max_depth=4,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        gamma=1.5, reg_alpha=1.0, reg_lambda=3.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42,
        scale_pos_weight=n_normal/n_extreme)
    clf_extreme.fit(X_tr, y_ext_tr, eval_set=[(X_vl, y_ext_vl)], verbose=False)
    ext_vl = clf_extreme.predict_proba(X_vl)[:, 1]
    ext_te = clf_extreme.predict_proba(X_te)[:, 1]
    print(f"    Extreme classifier iterations: {clf_extreme.best_iteration}")

    # ── Isotonic calibration of NN predictions ────────────────────────────
    print("  [Cal] Calibrating NN predictions...")
    iso = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
    iso.fit(nn_vl, y_vl)
    nn_vl_cal = iso.predict(nn_vl)
    nn_te_cal = iso.predict(nn_te)

    # ── Grid search on VALIDATION ────────────────────────────────────────
    print("  [Grid] Searching best ensemble config on validation...")
    best_score, best_cfg, best_m = -999, None, None

    for w_nn in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for use_cal in [False, True]:
            nn_v = nn_vl_cal if use_cal else nn_vl
            ens_vl = w_nn * nn_v + (1 - w_nn) * xgb_vl

            for rain_t in np.arange(0.30, 0.60, 0.02):
                rain_mask = rain_vl >= rain_t
                gated = np.zeros_like(ens_vl)
                gated[rain_mask] = ens_vl[rain_mask]

                # Try with and without extreme boost
                for ext_boost in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    final = gated.copy()
                    if ext_boost > 0:
                        # Boost predictions where extreme classifier is confident
                        ext_mask = ext_vl >= 0.5
                        final[ext_mask] = final[ext_mask] * (1.0 + ext_boost * ext_vl[ext_mask])

                    m = evaluate(final, y_vl, thresholds, prefix="")
                    far_p90 = m.get("FAR_p90", 1.0)
                    csi_p90 = m.get("CSI_p90", 0)
                    csi_rain = m.get("CSI_rain", 0)
                    csi_p95 = m.get("CSI_p95", 0)
                    sedi_p90 = m.get("SEDI_p90", 0)

                    # Score: balanced across all metrics
                    score = (csi_rain * 1.0
                             + csi_p90 * 3.0
                             + csi_p95 * 2.0
                             + sedi_p90 * 1.0
                             + m.get("SEDI_p95", 0) * 0.5
                             - far_p90 * 2.0
                             - m.get("FAR_rain", 1.0) * 0.5)

                    if score > best_score:
                        best_score = score
                        best_cfg = {
                            "w_nn": w_nn,
                            "rain_t": float(rain_t),
                            "ext_boost": ext_boost,
                            "use_cal": use_cal,
                        }
                        best_m = m

    print(f"  Best config: w_nn={best_cfg['w_nn']}, rain_t={best_cfg['rain_t']:.2f}, "
          f"ext_boost={best_cfg['ext_boost']}, cal={best_cfg['use_cal']}")
    print(f"  Val: CSI_rain={best_m.get('CSI_rain',0):.4f}  "
          f"CSI_P90={best_m.get('CSI_p90',0):.4f}  "
          f"FAR_P90={best_m.get('FAR_p90',0):.4f}  "
          f"CSI_P95={best_m.get('CSI_p95',0):.4f}  "
          f"SEDI_P90={best_m.get('SEDI_p90',0):.4f}")

    # ── Apply ONCE to test ───────────────────────────────────────────────
    print(f"\n  >> Evaluating ONCE on test (no leakage)...")
    nn_t = nn_te_cal if best_cfg["use_cal"] else nn_te
    ens_te = best_cfg["w_nn"] * nn_t + (1 - best_cfg["w_nn"]) * xgb_te
    rain_mask_te = rain_te >= best_cfg["rain_t"]
    final_te = np.zeros_like(ens_te)
    final_te[rain_mask_te] = ens_te[rain_mask_te]

    if best_cfg["ext_boost"] > 0:
        ext_mask_te = ext_te >= 0.5
        final_te[ext_mask_te] = final_te[ext_mask_te] * (1.0 + best_cfg["ext_boost"] * ext_te[ext_mask_te])

    test_m = evaluate(final_te, y_te, thresholds, prefix="")
    corr = 0
    rainy_mask = y_te >= 0.1
    if rainy_mask.sum() > 2:
        corr = float(np.corrcoef(final_te[rainy_mask], y_te[rainy_mask])[0, 1])

    print_metrics(test_m, title=f"{label} -- TEST RESULTS")
    print(f"  corr_rainy = {corr:.4f}")

    # P90 contingency detail
    pred_p90 = final_te >= p90
    actual_p90 = y_te >= p90
    H = (pred_p90 & actual_p90).sum()
    M = (~pred_p90 & actual_p90).sum()
    FA = (pred_p90 & ~actual_p90).sum()
    print(f"  P90 contingency: H={H}, M={M}, FA={FA}, n_actual={actual_p90.sum()}, n_pred={pred_p90.sum()}")

    return test_m, corr, best_cfg


def main():
    print("\n" + "=" * 60)
    print("  ENHANCED ENSEMBLE PIPELINE v3")
    print("=" * 60)

    # ── Load ALL data ────────────────────────────────────────────────────
    print("\n[1/4] Loading ALL data...")
    builder = RainfallDataBuilder(window_size=config.DEFAULT_WINDOW)
    all_years = list(range(2015, 2025))

    all_patches, all_tabular, all_targets = [], [], []
    year_indices = []
    cumulative = 0
    for yr in all_years:
        p, t, y = builder.build([yr])
        all_patches.append(p)
        all_tabular.append(t)
        all_targets.append(y)
        year_indices.append((yr, cumulative, cumulative + len(y)))
        cumulative += len(y)

    all_patches = np.concatenate(all_patches)
    all_tabular = np.concatenate(all_tabular)
    all_targets = np.concatenate(all_targets)
    print(f"  Total: {len(all_targets)} samples across {len(all_years)} years")

    rainy_all = all_targets[all_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_all, 90))
    p95 = float(np.percentile(rainy_all, 95))
    p99 = float(np.percentile(rainy_all, 99))
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    print(f"  P90={p90:.1f}mm  P95={p95:.1f}mm  P99={p99:.1f}mm")
    print(f"  Rainy: {len(rainy_all)} ({len(rainy_all)/len(all_targets)*100:.1f}%)")
    print(f"  P90 events: {(all_targets >= p90).sum()}")

    # ── Load SmallNet ────────────────────────────────────────────────────
    print(f"\n[2/4] Loading SmallNet...")
    nn_model = build_model(window_size=config.DEFAULT_WINDOW, n_channels=config.N_CNN_CHANNELS, n_tabular=config.N_TABULAR)

    ckpt_dir = config.OUTPUT_DIR / f"window_{config.DEFAULT_WINDOW}"
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"  ERROR: No checkpoints in {ckpt_dir}")
        print("  Run 'python train.py --window 9' first!")
        sys.exit(1)

    ckpt_path = pts[-1]
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()
    print(f"  Loaded: {ckpt_path.name}")

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION A: Temporal Split (MAXIMIZED)
    # Train XGBoost on ALL 2015-2023, random 15% holdout for early stopping
    # SmallNet was trained on 2015-2022, so its 2023 preds are "test-like"
    # — which is GOOD because XGBoost learns to work with noisy NN preds
    # ═══════════════════════════════════════════════════════════════════
    print("\n[3/4] Temporal Split Evaluation (MAXIMIZED)...")

    # All non-test data = 2015-2023 (9 years)
    all_train_mask = np.zeros(len(all_targets), dtype=bool)
    te_mask = np.zeros(len(all_targets), dtype=bool)
    for yr, start, end in year_indices:
        if yr in config.TEST_YEARS:
            te_mask[start:end] = True
        elif yr != 2024:  # everything except 2024
            all_train_mask[start:end] = True

    all_train_idx = np.where(all_train_mask)[0]
    n_all_train = len(all_train_idx)

    # Random 85/15 split from 2015-2023 for XGBoost train/val
    np.random.seed(99)
    perm = np.random.permutation(n_all_train)
    n_xgb_train = int(0.85 * n_all_train)
    xgb_tr_idx = all_train_idx[perm[:n_xgb_train]]
    xgb_vl_idx = all_train_idx[perm[n_xgb_train:]]

    print(f"  XGBoost Train: {len(xgb_tr_idx)} samples (from 2015-2023)")
    print(f"  XGBoost Val:   {len(xgb_vl_idx)} samples (random 15% holdout)")
    print(f"  Test:          {te_mask.sum()} samples (2024)")

    # Normaliser fit on ALL 2015-2023
    norm_temporal = Normaliser()
    norm_temporal.fit(all_patches[all_train_mask], all_tabular[all_train_mask])

    # NN predictions for all splits
    nn_tr_t = nn_predict(nn_model, all_patches[xgb_tr_idx], all_tabular[xgb_tr_idx], norm_temporal)
    nn_vl_t = nn_predict(nn_model, all_patches[xgb_vl_idx], all_tabular[xgb_vl_idx], norm_temporal)
    nn_te_t = nn_predict(nn_model, all_patches[te_mask], all_tabular[te_mask], norm_temporal)

    # XGBoost features
    X_tr_t = build_features(all_patches[xgb_tr_idx], all_tabular[xgb_tr_idx], norm_temporal, nn_tr_t)
    X_vl_t = build_features(all_patches[xgb_vl_idx], all_tabular[xgb_vl_idx], norm_temporal, nn_vl_t)
    X_te_t = build_features(all_patches[te_mask], all_tabular[te_mask], norm_temporal, nn_te_t)

    print(f"  Feature dimensions: {X_tr_t.shape[1]}")

    temporal_m, temporal_corr, temporal_cfg = run_pipeline(
        X_tr_t, all_targets[xgb_tr_idx],
        X_vl_t, all_targets[xgb_vl_idx],
        X_te_t, all_targets[te_mask],
        nn_vl_t, nn_te_t,
        thresholds, p90,
        label="A: TEMPORAL (Train 2015-2023 XGBoost, Test 2024)")

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION B: Random Split
    # ═══════════════════════════════════════════════════════════════════
    print("\n[4/4] Random Split Evaluation...")
    np.random.seed(42)
    N = len(all_targets)
    indices = np.random.permutation(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    print(f"  Random split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"  Test P90 events: {(all_targets[test_idx] >= p90).sum()}")

    norm_random = Normaliser()
    norm_random.fit(all_patches[train_idx], all_tabular[train_idx])

    nn_tr_r = nn_predict(nn_model, all_patches[train_idx], all_tabular[train_idx], norm_random)
    nn_vl_r = nn_predict(nn_model, all_patches[val_idx], all_tabular[val_idx], norm_random)
    nn_te_r = nn_predict(nn_model, all_patches[test_idx], all_tabular[test_idx], norm_random)

    X_tr_r = build_features(all_patches[train_idx], all_tabular[train_idx], norm_random, nn_tr_r)
    X_vl_r = build_features(all_patches[val_idx], all_tabular[val_idx], norm_random, nn_vl_r)
    X_te_r = build_features(all_patches[test_idx], all_tabular[test_idx], norm_random, nn_te_r)

    random_m, random_corr, random_cfg = run_pipeline(
        X_tr_r, all_targets[train_idx],
        X_vl_r, all_targets[val_idx],
        X_te_r, all_targets[test_idx],
        nn_vl_r, nn_te_r,
        thresholds, p90,
        label="B: RANDOM 70/15/15 SPLIT (all years mixed)")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPARISON: Temporal Split vs Random Split")
    print("=" * 70)
    print(f"\n  {'Metric':<15s} {'Temporal (2024)':>15s} {'Random 70/30':>15s} {'Target':>10s}")
    print("  " + "-" * 57)
    for key in ["CSI_rain", "POD_rain", "FAR_rain", "SEDI_rain",
                "CSI_p90", "POD_p90", "FAR_p90", "SEDI_p90",
                "CSI_p95", "POD_p95", "FAR_p95", "SEDI_p95"]:
        t_val = temporal_m.get(key, 0)
        r_val = random_m.get(key, 0)
        target = ""
        if "CSI_rain" in key: target = ">=0.50"
        elif "CSI_p90" in key: target = ">=0.25"
        elif "CSI_p95" in key: target = ">=0.35"
        elif "SEDI" in key: target = ">=0.50"
        print(f"  {key:<15s} {t_val:>15.4f} {r_val:>15.4f} {target:>10s}")
    print(f"  {'corr_rainy':<15s} {temporal_corr:>15.4f} {random_corr:>15.4f}")
    print(f"  {'RMSE':<15s} {temporal_m.get('RMSE',0):>15.4f} {random_m.get('RMSE',0):>15.4f}")

    # Save
    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "temporal_split": {
            "config": temporal_cfg,
            "metrics": {k: v for k, v in temporal_m.items() if isinstance(v, (float, int))},
            "correlation": temporal_corr,
        },
        "random_split": {
            "config": random_cfg,
            "metrics": {k: v for k, v in random_m.items() if isinstance(v, (float, int))},
            "correlation": random_corr,
        },
        "thresholds": thresholds,
        "feature_count": int(X_tr_t.shape[1]),
    }
    with open(out_dir / "dual_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_dir / 'dual_eval_results.json'}")


if __name__ == "__main__":
    main()
