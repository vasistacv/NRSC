"""
train_dual_eval.py — Dual Evaluation Pipeline
===============================================
Two evaluation modes:
  A) Temporal split: Train 2015-2021, Val 2022-2023, Test 2024 (realistic)
  B) Random 70/30 split: All years mixed (more stable, larger test set)

Both use: Rain classifier gate + SmallNet/XGBoost ensemble
All hyperparameters chosen on VALIDATION, evaluated ONCE on test.
"""

import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import config
from dataset import RainfallDataBuilder, Normaliser
from model import build_model
from metrics import evaluate, print_metrics, contingency, csi


def run_pipeline(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                 nn_vl, nn_te, thresholds, p90, label=""):
    """Run full pipeline: train classifiers + regression, grid search on val, eval on test."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    p95 = thresholds["p95"]

    # ── Rain classifier ──────────────────────────────────────────────────
    print("  Training rain classifier...")
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

    best_rain_t, best_rain_csi = 0.5, 0
    for t in np.arange(0.2, 0.8, 0.01):
        ct = contingency((rain_vl >= t).astype(float) * 999, y_vl, config.DRY_THRESHOLD)
        c = csi(ct)
        if c > best_rain_csi:
            best_rain_csi, best_rain_t = c, t
    print(f"  Rain threshold: {best_rain_t:.2f} (val CSI={best_rain_csi:.4f})")

    # ── XGBoost Regression ───────────────────────────────────────────────
    print("  Training XGBoost regression...")
    w = np.ones(len(y_tr), dtype=np.float32)
    w[(y_tr >= 0.1) & (y_tr < p90)] = 2.0
    w[(y_tr >= p90) & (y_tr < p95)] = 8.0
    w[y_tr >= p95] = 15.0

    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_vl, y_vl)], verbose=False)
    xgb_vl = reg.predict(X_vl).clip(min=0)
    xgb_te = reg.predict(X_te).clip(min=0)
    print(f"  Regression iter: {reg.best_iteration}")

    # ── Grid search on VALIDATION ────────────────────────────────────────
    print("  Grid search on validation...")
    best_score, best_cfg, best_m = -999, None, None

    for w_nn in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]:
        ens_vl = w_nn * nn_vl + (1 - w_nn) * xgb_vl

        for rain_t in np.arange(0.35, 0.55, 0.02):
            rain_mask = rain_vl >= rain_t
            final = np.zeros_like(ens_vl)
            final[rain_mask] = ens_vl[rain_mask]

            m = evaluate(final, y_vl, thresholds, prefix="")
            far_p90 = m.get("FAR_p90", 1.0)
            csi_p90 = m.get("CSI_p90", 0)
            csi_rain = m.get("CSI_rain", 0)

            # Score: maximize CSI_P90 while penalizing FAR heavily
            score = (csi_rain + csi_p90 * 3.0 + m.get("CSI_p95", 0) * 1.0
                     + m.get("SEDI_rain", 0) * 0.3 - far_p90 * 2.0)

            if score > best_score:
                best_score = score
                best_cfg = {"w_nn": w_nn, "rain_t": float(rain_t)}
                best_m = m

    print(f"  Best config: w_nn={best_cfg['w_nn']}, rain_t={best_cfg['rain_t']:.2f}")
    print(f"  Val: CSI_rain={best_m.get('CSI_rain',0):.4f}  "
          f"CSI_P90={best_m.get('CSI_p90',0):.4f}  "
          f"FAR_P90={best_m.get('FAR_p90',0):.4f}  "
          f"CSI_P95={best_m.get('CSI_p95',0):.4f}")

    # ── Apply ONCE to test ───────────────────────────────────────────────
    print("\n  >> Evaluating ONCE on test (no leakage)...")
    ens_te = best_cfg["w_nn"] * nn_te + (1 - best_cfg["w_nn"]) * xgb_te
    rain_mask_te = rain_te >= best_cfg["rain_t"]
    final_te = np.zeros_like(ens_te)
    final_te[rain_mask_te] = ens_te[rain_mask_te]

    test_m = evaluate(final_te, y_te, thresholds, prefix="")
    corr = np.corrcoef(final_te[y_te >= 0.1], y_te[y_te >= 0.1])[0, 1] if (y_te >= 0.1).sum() > 2 else 0

    print_metrics(test_m, title=f"{label} — TEST RESULTS")
    print(f"  corr_rainy = {corr:.4f}")

    # P90 contingency
    pred_p90 = final_te >= p90
    actual_p90 = y_te >= p90
    H = (pred_p90 & actual_p90).sum()
    M = (~pred_p90 & actual_p90).sum()
    FA = (pred_p90 & ~actual_p90).sum()
    print(f"  P90 contingency: H={H}, M={M}, FA={FA}, n_actual={actual_p90.sum()}, n_pred={pred_p90.sum()}")

    return test_m, corr, best_cfg


def main():
    print("\n" + "=" * 60)
    print("  DUAL EVALUATION PIPELINE")
    print("=" * 60)

    # ── Load ALL data ────────────────────────────────────────────────────
    print("\n[1/3] Loading ALL data...")
    builder = RainfallDataBuilder(window_size=config.DEFAULT_WINDOW)
    all_years = list(range(2015, 2025))  # 2015-2024

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
    print(f"\n[2/3] Loading SmallNet (window={config.DEFAULT_WINDOW})...")
    nn_model = build_model(window_size=config.DEFAULT_WINDOW, n_channels=19, n_tabular=24)
    
    # Needs to match the best checkpoint inside experiment_outputs/window_{size}/
    ckpt_dir = config.OUTPUT_DIR / f"window_{config.DEFAULT_WINDOW}"
    
    # Simple logic to find the best checkpoint
    import glob
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"ERROR: No .pt checkpoints found in {ckpt_dir} for window_size={config.DEFAULT_WINDOW}.")
        print("You must run 'python extracted_files/train.py' first to train the neural network!")
        sys.exit(1)
        
    ckpt_path = pts[-1]  # The trainer saves them sequentially, last one is fine
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()

    # ── Normalise ALL data once ──────────────────────────────────────────
    # For temporal split: fit normaliser on train years only
    # For random split: fit normaliser on the 70% train split

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION A: Temporal Split (2015-2021 / 2022-2023 / 2024)
    # ═══════════════════════════════════════════════════════════════════
    print("\n[3/3] Running evaluations...")

    # Split by year
    tr_mask = np.zeros(len(all_targets), dtype=bool)
    vl_mask = np.zeros(len(all_targets), dtype=bool)
    te_mask = np.zeros(len(all_targets), dtype=bool)
    for yr, start, end in year_indices:
        if yr in config.TRAIN_YEARS:
            tr_mask[start:end] = True
        elif yr in config.VAL_YEARS:
            vl_mask[start:end] = True
        elif yr in config.TEST_YEARS:
            te_mask[start:end] = True

    norm_temporal = Normaliser()
    norm_temporal.fit(all_patches[tr_mask], all_tabular[tr_mask])

    def flatten(patches, tabulars, norm_obj):
        p = norm_obj.transform_patches(patches)
        t = norm_obj.transform_tabular(tabulars)
        return np.hstack([p.reshape(p.shape[0], -1), t])

    def nn_predict(patches, tabulars, norm_obj):
        p = torch.from_numpy(norm_obj.transform_patches(patches)).float()
        t = torch.from_numpy(norm_obj.transform_tabular(tabulars)).float()
        ds = TensorDataset(p, t)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for pb, tb in loader:
                preds.append(nn_model.predict(pb, tb).numpy())
        return np.concatenate(preds)

    X_tr_t = flatten(all_patches[tr_mask], all_tabular[tr_mask], norm_temporal)
    X_vl_t = flatten(all_patches[vl_mask], all_tabular[vl_mask], norm_temporal)
    X_te_t = flatten(all_patches[te_mask], all_tabular[te_mask], norm_temporal)
    nn_vl_t = nn_predict(all_patches[vl_mask], all_tabular[vl_mask], norm_temporal)
    nn_te_t = nn_predict(all_patches[te_mask], all_tabular[te_mask], norm_temporal)

    temporal_m, temporal_corr, temporal_cfg = run_pipeline(
        X_tr_t, all_targets[tr_mask],
        X_vl_t, all_targets[vl_mask],
        X_te_t, all_targets[te_mask],
        nn_vl_t, nn_te_t,
        thresholds, p90,
        label="A: TEMPORAL SPLIT (Train 2015-2021, Val 2022-2023, Test 2024)")

    # ═══════════════════════════════════════════════════════════════════
    # EVALUATION B: Random 70/30 Split (all years mixed)
    # ═══════════════════════════════════════════════════════════════════
    np.random.seed(42)
    N = len(all_targets)
    indices = np.random.permutation(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)  # 70% train, 15% val, 15% test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    print(f"\n  Random split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"  Test P90 events: {(all_targets[test_idx] >= p90).sum()}")

    norm_random = Normaliser()
    norm_random.fit(all_patches[train_idx], all_tabular[train_idx])

    X_tr_r = flatten(all_patches[train_idx], all_tabular[train_idx], norm_random)
    X_vl_r = flatten(all_patches[val_idx], all_tabular[val_idx], norm_random)
    X_te_r = flatten(all_patches[test_idx], all_tabular[test_idx], norm_random)
    nn_vl_r = nn_predict(all_patches[val_idx], all_tabular[val_idx], norm_random)
    nn_te_r = nn_predict(all_patches[test_idx], all_tabular[test_idx], norm_random)

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
        if "CSI_rain" in key: target = "≥0.35"
        elif "CSI_p90" in key: target = "≥0.25"
        elif "CSI_p95" in key: target = "≥0.35"
        elif "SEDI" in key: target = "≥0.50"
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
    }
    with open(out_dir / "dual_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_dir / 'dual_eval_results.json'}")


if __name__ == "__main__":
    main()
