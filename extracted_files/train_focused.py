"""
train_focused.py — Focused P90 CSI maximization
=================================================
Uses all existing models, no retraining needed.
Extensive grid search on VALIDATION over:
  - Rain classifier threshold
  - P90 classifier threshold  
  - Regression minimum for P90 boost
  - Ensemble weight (SmallNet vs XGBoost)
  - Boost magnitude
Then ONE evaluation on test.
"""

import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

import config
from dataset import RainfallDataBuilder, Normaliser
from model import build_model
from metrics import evaluate, print_metrics, contingency, csi


def main():
    print("\n" + "=" * 60)
    print("  Focused P90 CSI Optimization")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    builder = RainfallDataBuilder(window_size=3)
    tr_patches, tr_tabular, tr_targets = builder.build(config.TRAIN_YEARS)
    vl_patches, vl_tabular, vl_targets = builder.build(config.VAL_YEARS)
    te_patches, te_tabular, te_targets = builder.build(config.TEST_YEARS)

    norm = Normaliser()
    norm.fit(tr_patches, tr_tabular)

    rainy_train = tr_targets[tr_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_train, 90))
    p95 = float(np.percentile(rainy_train, 95))
    thresholds = {"p90": p90, "p95": p95, "p99": float(np.percentile(rainy_train, 99))}
    print(f"  P90={p90:.1f}mm  P95={p95:.1f}mm")

    def flatten(patches, tabulars):
        p = norm.transform_patches(patches)
        t = norm.transform_tabular(tabulars)
        return np.hstack([p.reshape(p.shape[0], -1), t])

    X_tr = flatten(tr_patches, tr_tabular)
    X_vl = flatten(vl_patches, vl_tabular)
    X_te = flatten(te_patches, te_tabular)

    # ── Get SmallNet predictions ─────────────────────────────────────────
    print("\n[2/4] Loading SmallNet + XGBoost...")
    nn_model = build_model(window_size=3, n_channels=19, n_tabular=24)
    ckpt_path = config.OUTPUT_DIR / "window_3" / "ckpt_epoch0084_csi0.3645.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()

    def nn_predict(patches, tabulars):
        p = torch.from_numpy(norm.transform_patches(patches)).float()
        t = torch.from_numpy(norm.transform_tabular(tabulars)).float()
        ds = TensorDataset(p, t)
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        preds = []
        with torch.no_grad():
            for pb, tb in loader:
                preds.append(nn_model.predict(pb, tb).numpy())
        return np.concatenate(preds)

    nn_vl = nn_predict(vl_patches, vl_tabular)
    nn_te = nn_predict(te_patches, te_tabular)

    # ── Train XGBoost models ─────────────────────────────────────────────
    print("\n  Training XGBoost regression...")
    w = np.ones(len(tr_targets), dtype=np.float32)
    w[(tr_targets >= 0.1) & (tr_targets < p90)] = 2.0
    w[(tr_targets >= p90) & (tr_targets < p95)] = 8.0
    w[tr_targets >= p95] = 15.0

    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, tr_targets, sample_weight=w, eval_set=[(X_vl, vl_targets)], verbose=False)
    xgb_vl = reg.predict(X_vl).clip(min=0)
    xgb_te = reg.predict(X_te).clip(min=0)

    # Rain classifier
    print("  Training rain classifier...")
    y_rain_tr = (tr_targets >= config.DRY_THRESHOLD).astype(int)
    y_rain_vl = (vl_targets >= config.DRY_THRESHOLD).astype(int)
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

    # P90 classifier
    print("  Training P90 classifier...")
    y_p90_tr = (tr_targets >= p90).astype(int)
    y_p90_vl = (vl_targets >= p90).astype(int)
    n_neg, n_pos = (y_p90_tr == 0).sum(), (y_p90_tr == 1).sum()

    clf_p90 = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.02, max_depth=7,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.65,
        gamma=0.3, reg_alpha=0.2, reg_lambda=1.0,
        n_estimators=2000, early_stopping_rounds=100,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_neg/n_pos)
    clf_p90.fit(X_tr, y_p90_tr, eval_set=[(X_vl, y_p90_vl)], verbose=False)
    p90_vl = clf_p90.predict_proba(X_vl)[:, 1]
    p90_te = clf_p90.predict_proba(X_te)[:, 1]

    print(f"  XGBoost iter={reg.best_iteration}, Rain iter={clf_rain.best_iteration}, P90 iter={clf_p90.best_iteration}")

    # ── Extensive grid search on VALIDATION ──────────────────────────────
    print("\n[3/4] Extensive grid search on VALIDATION...")
    print("  Testing many combinations...")

    best_score = -1
    best_config = None
    best_metrics = None
    count = 0

    for w_nn in [0.10, 0.14, 0.18, 0.20, 0.22, 0.25, 0.30]:
        ens_vl = w_nn * nn_vl + (1 - w_nn) * xgb_vl

        for rain_t in np.arange(0.35, 0.55, 0.03):
            rain_mask = rain_vl >= rain_t

            for strategy in ["none", "condboost", "softscale"]:
                if strategy == "none":
                    # Pure regression + rain gate
                    final = np.zeros_like(ens_vl)
                    final[rain_mask] = ens_vl[rain_mask]
                    m = evaluate(final, vl_targets, thresholds, prefix="")
                    far_p90 = m.get("FAR_p90", 1.0)
                    if far_p90 > 0.60:
                        count += 1; continue  # Skip high-FAR configs
                    score = (m.get("CSI_rain", 0) + m.get("CSI_p90", 0) * 3.0
                             + m.get("CSI_p95", 0) * 1.0 + m.get("SEDI_rain", 0) * 0.3
                             - far_p90 * 3.0)  # strong FAR penalty on P90
                    count += 1
                    if score > best_score:
                        best_score = score
                        best_config = {"w_nn": w_nn, "rain_t": rain_t, "strategy": "none"}
                        best_metrics = m

                elif strategy == "condboost":
                    # Boost only when BOTH classifier and regression agree
                    for p90_t in np.arange(0.05, 0.5, 0.03):
                        for min_reg in [p90 * f for f in [0.3, 0.4, 0.5, 0.6, 0.7]]:
                            for boost in [p90 * f for f in [1.0, 1.02, 1.05]]:
                                p90_mask = p90_vl >= p90_t
                                final = np.zeros_like(ens_vl)
                                final[rain_mask] = ens_vl[rain_mask]
                                cond = rain_mask & p90_mask & (ens_vl >= min_reg)
                                final[cond] = np.maximum(final[cond], boost)

                                m = evaluate(final, vl_targets, thresholds, prefix="")
                                far_p90 = m.get("FAR_p90", 1.0)
                                if far_p90 > 0.60:
                                    count += 1; continue
                                score = (m.get("CSI_rain", 0) + m.get("CSI_p90", 0) * 3.0
                                         + m.get("CSI_p95", 0) * 1.0 + m.get("SEDI_rain", 0) * 0.3
                                         - far_p90 * 3.0)
                                count += 1
                                if score > best_score:
                                    best_score = score
                                    best_config = {"w_nn": w_nn, "rain_t": rain_t, "strategy": "condboost",
                                                   "p90_t": p90_t, "min_reg": min_reg, "boost": boost}
                                    best_metrics = m

                elif strategy == "softscale":
                    # Scale by P90 probability, but only for medium+ predictions
                    for min_reg in [5, 10, 15, 20]:
                        for scale in [0.5, 1.0, 1.5, 2.0]:
                            final = np.zeros_like(ens_vl)
                            final[rain_mask] = ens_vl[rain_mask]
                            above_min = rain_mask & (ens_vl >= min_reg)
                            final[above_min] = final[above_min] * (1.0 + scale * p90_vl[above_min])

                            m = evaluate(final, vl_targets, thresholds, prefix="")
                            far_p90 = m.get("FAR_p90", 1.0)
                            if far_p90 > 0.55:
                                count += 1; continue
                            score = (m.get("CSI_rain", 0) + m.get("CSI_p90", 0) * 3.0
                                     + m.get("CSI_p95", 0) * 1.0 + m.get("SEDI_rain", 0) * 0.3
                                     - far_p90 * 3.0)
                            count += 1
                            if score > best_score:
                                best_score = score
                                best_config = {"w_nn": w_nn, "rain_t": rain_t, "strategy": "softscale",
                                               "min_reg": min_reg, "scale": scale}
                                best_metrics = m

    if best_config is None:
        print("  WARNING: No config passed FAR constraint, using pure regression with lowest FAR w_nn")
        # Fallback: pure regression, find w_nn with best CSI_P90 regardless of constraint
        for w_nn in [0.10, 0.14, 0.18, 0.20, 0.22, 0.25, 0.30]:
            ens_vl = w_nn * nn_vl + (1 - w_nn) * xgb_vl
            rain_mask = rain_vl >= 0.43
            final = np.zeros_like(ens_vl)
            final[rain_mask] = ens_vl[rain_mask]
            m = evaluate(final, vl_targets, thresholds, prefix="")
            score = m.get("CSI_p90", 0) - m.get("FAR_p90", 1.0)
            if score > best_score:
                best_score = score
                best_config = {"w_nn": w_nn, "rain_t": 0.43, "strategy": "none"}
                best_metrics = m

    print(f"  Searched {count} configurations")
    print(f"  Best: {best_config}")
    print(f"  Val CSI_rain={best_metrics.get('CSI_rain',0):.4f}  "
          f"CSI_P90={best_metrics.get('CSI_p90',0):.4f}  "
          f"CSI_P95={best_metrics.get('CSI_p95',0):.4f}  "
          f"SEDI_P90={best_metrics.get('SEDI_p90',0):.4f}")

    # ── Apply ONCE to test ───────────────────────────────────────────────
    print("\n[4/4] Applying best config to TEST (single evaluation, no leakage)...")

    cfg = best_config
    ens_te = cfg["w_nn"] * nn_te + (1 - cfg["w_nn"]) * xgb_te
    rain_mask_te = rain_te >= cfg["rain_t"]

    final_te = np.zeros_like(ens_te)

    if cfg["strategy"] == "none":
        final_te[rain_mask_te] = ens_te[rain_mask_te]

    elif cfg["strategy"] == "condboost":
        final_te[rain_mask_te] = ens_te[rain_mask_te]
        cond = rain_mask_te & (p90_te >= cfg["p90_t"]) & (ens_te >= cfg["min_reg"])
        final_te[cond] = np.maximum(final_te[cond], cfg["boost"])

    elif cfg["strategy"] == "softscale":
        final_te[rain_mask_te] = ens_te[rain_mask_te]
        above_min = rain_mask_te & (ens_te >= cfg["min_reg"])
        final_te[above_min] = final_te[above_min] * (1.0 + cfg["scale"] * p90_te[above_min])

    corr_f = np.corrcoef(final_te[te_targets >= 0.1], te_targets[te_targets >= 0.1])[0, 1] if (te_targets >= 0.1).sum() > 2 else 0

    test_m = evaluate(final_te, te_targets, thresholds, prefix="")
    print_metrics(test_m, title="FINAL TEST RESULTS (no leakage)")
    print(f"  corr_rainy = {corr_f:.4f}")

    # Detailed P90 analysis
    pred_p90 = final_te >= p90
    actual_p90 = te_targets >= p90
    H = (pred_p90 & actual_p90).sum()
    M = (~pred_p90 & actual_p90).sum()
    FA = (pred_p90 & ~actual_p90).sum()
    print(f"\n  P90 contingency: H={H}, M={M}, FA={FA}, n_actual={actual_p90.sum()}, n_pred={pred_p90.sum()}")

    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({
            "model": "focused_ensemble",
            "method": "val-optimized extensive grid search, single test eval",
            "test_metrics": {k: v for k, v in test_m.items() if isinstance(v, (float, int))},
            "thresholds": thresholds,
            "correlation": corr_f,
            "config": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in cfg.items()},
        }, f, indent=2)
    print(f"  Saved to: {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
