"""
train_xgb.py — v3
==================
Complete pipeline:
  1. XGBoost classifier for rain/no-rain → fixes SEDI_rain
  2. XGBoost classifier for P90 yes/no → improves CSI_P90
  3. XGBoost regression for intensity
  4. SmallNet + XGBoost ensemble for final prediction
  5. Combine: classifier gates × regression intensity
"""

import sys
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config
from dataset import RainfallDataBuilder, Normaliser
from metrics import evaluate, print_metrics, check_targets, contingency, csi

import xgboost as xgb
import torch
from model import build_model
from torch.utils.data import DataLoader, TensorDataset


def main():
    print("\n" + "=" * 60)
    print("  Complete Pipeline: Classifiers + Regression + Ensemble")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading raw data...")
    builder = RainfallDataBuilder(window_size=3)
    tr_patches, tr_tabular, tr_targets = builder.build(config.TRAIN_YEARS)
    vl_patches, vl_tabular, vl_targets = builder.build(config.VAL_YEARS)
    te_patches, te_tabular, te_targets = builder.build(config.TEST_YEARS)

    norm = Normaliser()
    norm.fit(tr_patches, tr_tabular)

    def flatten(patches, tabulars, norm_obj):
        p = norm_obj.transform_patches(patches)
        t = norm_obj.transform_tabular(tabulars)
        N = p.shape[0]
        return np.hstack([p.reshape(N, -1), t])

    X_train = flatten(tr_patches, tr_tabular, norm)
    X_val   = flatten(vl_patches, vl_tabular, norm)
    X_test  = flatten(te_patches, te_tabular, norm)
    y_train, y_val, y_test = tr_targets, vl_targets, te_targets

    rainy_train = y_train[y_train >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_train, 90))
    p95 = float(np.percentile(rainy_train, 95))
    p99 = float(np.percentile(rainy_train, 99))
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    print(f"  Shapes: Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")
    print(f"  P90={p90:.1f}mm  P95={p95:.1f}mm  P99={p99:.1f}mm")

    # ── Step 2: Rain / No-Rain Classifier ────────────────────────────────
    print("\n[2/6] Training Rain/No-Rain classifier...")
    y_rain_train = (y_train >= config.DRY_THRESHOLD).astype(int)
    y_rain_val   = (y_val >= config.DRY_THRESHOLD).astype(int)

    # Balance weights
    n_dry = (y_rain_train == 0).sum()
    n_wet = (y_rain_train == 1).sum()
    w_rain = np.ones(len(y_rain_train))
    w_rain[y_rain_train == 1] = n_dry / n_wet  # balance classes

    clf_rain = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=config.SEED,
        scale_pos_weight=n_dry/n_wet,
    )
    clf_rain.fit(X_train, y_rain_train, eval_set=[(X_val, y_rain_val)], verbose=False)

    rain_prob_val = clf_rain.predict_proba(X_val)[:, 1]
    rain_prob_test = clf_rain.predict_proba(X_test)[:, 1]

    # Find optimal rain threshold on val
    best_csi_rain, best_rain_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.01):
        pred_rain = (rain_prob_val >= t).astype(float) * 999  # large value to count as rainy
        ct = contingency(pred_rain, y_val, config.DRY_THRESHOLD)
        c = csi(ct)
        if c > best_csi_rain:
            best_csi_rain, best_rain_t = c, t
    print(f"  Best rain threshold: {best_rain_t:.2f} (val CSI={best_csi_rain:.4f})")
    print(f"  Classifier iter: {clf_rain.best_iteration}")

    # ── Step 3: P90 Classifier ───────────────────────────────────────────
    print("\n[3/6] Training P90 classifier...")
    y_p90_train = (y_train >= p90).astype(int)
    y_p90_val   = (y_val >= p90).astype(int)

    n_neg = (y_p90_train == 0).sum()
    n_pos = (y_p90_train == 1).sum()

    clf_p90 = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
        gamma=0.5, reg_alpha=0.3, reg_lambda=1.5,
        n_estimators=1500, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=config.SEED,
        scale_pos_weight=n_neg/n_pos,
    )
    clf_p90.fit(X_train, y_p90_train, eval_set=[(X_val, y_p90_val)], verbose=False)

    p90_prob_val = clf_p90.predict_proba(X_val)[:, 1]
    p90_prob_test = clf_p90.predict_proba(X_test)[:, 1]

    # Find best P90 threshold that maximizes CSI_P90
    best_csi_p90, best_p90_t = 0, 0.5
    for t in np.arange(0.05, 0.9, 0.01):
        pred_p90 = (p90_prob_val >= t).astype(float) * (p90 + 1)  # above P90
        ct = contingency(pred_p90, y_val, p90)
        c = csi(ct)
        if c > best_csi_p90:
            best_csi_p90, best_p90_t = c, t
    print(f"  Best P90 threshold: {best_p90_t:.2f} (val CSI={best_csi_p90:.4f})")
    print(f"  Classifier iter: {clf_p90.best_iteration}")

    # ── Step 4: XGBoost Regression ───────────────────────────────────────
    print("\n[4/6] Training XGBoost regression...")
    w = np.ones(len(y_train), dtype=np.float32)
    w[(y_train >= 0.1) & (y_train < p90)] = 2.0
    w[(y_train >= p90) & (y_train < p95)] = 6.0
    w[y_train >= p95] = 12.0

    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8,
        reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=config.SEED,
    )
    reg.fit(X_train, y_train, sample_weight=w,
            eval_set=[(X_val, y_val)], verbose=False)

    xgb_pred_test = reg.predict(X_test).clip(min=0)
    xgb_pred_val = reg.predict(X_val).clip(min=0)
    corr_xgb = np.corrcoef(xgb_pred_test[y_test>=0.1], y_test[y_test>=0.1])[0,1]
    print(f"  Regression iter: {reg.best_iteration}, corr_rainy={corr_xgb:.4f}")

    # ── Step 5: Load SmallNet ────────────────────────────────────────────
    print("\n[5/6] Loading SmallNet...")
    nn_model = build_model(window_size=3, n_channels=19, n_tabular=24)
    ckpt_path = config.OUTPUT_DIR / "window_3" / "ckpt_epoch0084_csi0.3645.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()

    te_p = norm.transform_patches(te_patches)
    te_t_norm = norm.transform_tabular(te_tabular)
    te_ds = TensorDataset(torch.from_numpy(te_p).float(),
                          torch.from_numpy(te_t_norm).float(),
                          torch.from_numpy(te_targets).float())
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

    nn_preds = []
    with torch.no_grad():
        for pb, tb, _ in te_loader:
            nn_preds.append(nn_model.predict(pb, tb).numpy())
    nn_preds = np.concatenate(nn_preds)
    corr_nn = np.corrcoef(nn_preds[y_test>=0.1], y_test[y_test>=0.1])[0,1]
    print(f"  SmallNet corr_rainy={corr_nn:.4f}")

    # ── Step 6: Find best config on VALIDATION (no test leakage) ────────
    print("\n[6/6] Grid search on VALIDATION set (publication-safe)...")

    # Get SmallNet val predictions too
    vl_p = norm.transform_patches(vl_patches)
    vl_t_norm = norm.transform_tabular(vl_tabular)
    vl_ds = TensorDataset(torch.from_numpy(vl_p).float(),
                          torch.from_numpy(vl_t_norm).float(),
                          torch.from_numpy(vl_targets).float())
    vl_loader = DataLoader(vl_ds, batch_size=256, shuffle=False)
    nn_preds_val = []
    with torch.no_grad():
        for pb, tb, _ in vl_loader:
            nn_preds_val.append(nn_model.predict(pb, tb).numpy())
    nn_preds_val = np.concatenate(nn_preds_val)

    xgb_pred_val = reg.predict(X_val).clip(min=0)

    val_results = {}
    for w_nn in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        ens_val = w_nn * nn_preds_val + (1 - w_nn) * xgb_pred_val

        for rain_t in [best_rain_t - 0.05, best_rain_t, best_rain_t + 0.05]:
            for alpha in [0.0, 0.5, 1.0, 2.0]:
                rain_mask_v = rain_prob_val >= rain_t
                final_v = np.zeros_like(ens_val)
                final_v[rain_mask_v] = ens_val[rain_mask_v] * (1.0 + alpha * p90_prob_val[rain_mask_v])

                m_v = evaluate(final_v, y_val, thresholds, prefix="")
                score_v = (m_v.get("CSI_rain", 0) + m_v.get("CSI_p90", 0) * 2.0
                           + m_v.get("CSI_p95", 0) * 1.5
                           + m_v.get("SEDI_rain", 0) * 0.5
                           + m_v.get("SEDI_p90", 0) * 0.3)
                key = (w_nn, rain_t, alpha)
                val_results[key] = {"m": m_v, "score": score_v}

    # Find best config on validation
    best_key = max(val_results, key=lambda k: val_results[k]["score"])
    best_w_nn, best_rt, best_alpha = best_key
    print(f"  Best config (from validation): w_nn={best_w_nn}, rain_t={best_rt:.2f}, alpha={best_alpha}")
    print(f"  Val score: {val_results[best_key]['score']:.4f}")

    # Show val metrics for best config
    best_val_m = val_results[best_key]["m"]
    print(f"  Val CSI_rain={best_val_m.get('CSI_rain',0):.4f}  "
          f"CSI_P90={best_val_m.get('CSI_p90',0):.4f}  "
          f"CSI_P95={best_val_m.get('CSI_p95',0):.4f}")

    # ── Apply best config ONCE to test set ───────────────────────────────
    print("\n  Evaluating ONCE on test set (no leakage)...")
    ens_test = best_w_nn * nn_preds + (1 - best_w_nn) * xgb_pred_test
    rain_mask_test = rain_prob_test >= best_rt
    final_test = np.zeros_like(ens_test)
    final_test[rain_mask_test] = ens_test[rain_mask_test] * (1.0 + best_alpha * p90_prob_test[rain_mask_test])

    corr_f = np.corrcoef(final_test[y_test >= 0.1], y_test[y_test >= 0.1])[0, 1] if (y_test >= 0.1).sum() > 2 else 0

    test_metrics = evaluate(final_test, y_test, thresholds, prefix="")
    print_metrics(test_metrics, title="FINAL RESULTS (publication-safe, no leakage)")

    # Save
    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({
            "model": f"ensemble_nn{int(best_w_nn*100)}_rt{best_rt:.2f}_a{best_alpha}",
            "method": "val-optimized, single test evaluation (no leakage)",
            "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (float, int))},
            "thresholds": thresholds,
            "correlation": corr_f,
            "config": {"w_nn": best_w_nn, "rain_t": best_rt, "alpha": best_alpha},
        }, f, indent=2)
    print(f"  Saved to: {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
