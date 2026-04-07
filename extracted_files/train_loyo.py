"""
train_loyo.py — Leave-One-Year-Out Cross-Validation
====================================================
For each year Y (2015-2024):
  Train on ALL other years except Y and val_year
  Val = one held-out year (for threshold tuning)
  Test = year Y
  
Collect predictions for ALL years, evaluate ONCE.
This gives the most robust and representative metric.
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


def main():
    print("\n" + "=" * 60)
    print("  LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print("=" * 60)

    # ── Load ALL data per year ───────────────────────────────────────────
    print("\n[1/3] Loading ALL data by year...")
    builder = RainfallDataBuilder(window_size=3)
    all_years = list(range(2015, 2025))

    year_data = {}
    for yr in all_years:
        p, t, y = builder.build([yr])
        year_data[yr] = {"patches": p, "tabular": t, "targets": y}
        print(f"  Year {yr}: {len(y)} samples, {(y >= 0.1).sum()} rainy")

    # Compute thresholds from ALL data
    all_targets = np.concatenate([year_data[yr]["targets"] for yr in all_years])
    rainy_all = all_targets[all_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_all, 90))
    p95 = float(np.percentile(rainy_all, 95))
    p99 = float(np.percentile(rainy_all, 99))
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    print(f"\n  Overall: {len(all_targets)} samples, P90={p90:.1f}mm, P95={p95:.1f}mm")
    print(f"  Total P90 events: {(all_targets >= p90).sum()}")

    # ── Load SmallNet ────────────────────────────────────────────────────
    print("\n[2/3] Loading SmallNet...")
    nn_model = build_model(window_size=3, n_channels=19, n_tabular=24)
    ckpt_path = config.OUTPUT_DIR / "window_3" / "ckpt_epoch0084_csi0.3645.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()

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

    def flatten(patches, tabulars, norm_obj):
        p = norm_obj.transform_patches(patches)
        t = norm_obj.transform_tabular(tabulars)
        return np.hstack([p.reshape(p.shape[0], -1), t])

    # ── LOYO Cross-Validation ────────────────────────────────────────────
    print("\n[3/3] LOYO Cross-Validation...")
    all_preds = []
    all_truths = []
    per_year_metrics = {}

    for test_yr in all_years:
        # Val year: previous year (wrap around for 2015)
        val_yr = test_yr - 1 if test_yr > 2015 else 2024
        train_yrs = [y for y in all_years if y != test_yr and y != val_yr]

        print(f"\n  Fold: Test={test_yr}, Val={val_yr}, Train={train_yrs}")

        # Merge train/val/test data
        tr_p = np.concatenate([year_data[y]["patches"] for y in train_yrs])
        tr_t = np.concatenate([year_data[y]["tabular"] for y in train_yrs])
        tr_y = np.concatenate([year_data[y]["targets"] for y in train_yrs])
        vl_p = year_data[val_yr]["patches"]
        vl_t = year_data[val_yr]["tabular"]
        vl_y = year_data[val_yr]["targets"]
        te_p = year_data[test_yr]["patches"]
        te_t = year_data[test_yr]["tabular"]
        te_y = year_data[test_yr]["targets"]

        # Normalize
        norm = Normaliser()
        norm.fit(tr_p, tr_t)
        X_tr = flatten(tr_p, tr_t, norm)
        X_vl = flatten(vl_p, vl_t, norm)
        X_te = flatten(te_p, te_t, norm)
        nn_vl = nn_predict(vl_p, vl_t, norm)
        nn_te = nn_predict(te_p, te_t, norm)

        # Train rain classifier
        y_rain_tr = (tr_y >= config.DRY_THRESHOLD).astype(int)
        y_rain_vl = (vl_y >= config.DRY_THRESHOLD).astype(int)
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

        # Find best rain threshold on val
        best_rt, best_rc = 0.5, 0
        for t in np.arange(0.25, 0.7, 0.01):
            ct = contingency((rain_vl >= t).astype(float) * 999, vl_y, config.DRY_THRESHOLD)
            c = csi(ct)
            if c > best_rc:
                best_rc, best_rt = c, t

        # Train XGBoost regression
        w = np.ones(len(tr_y), dtype=np.float32)
        w[(tr_y >= 0.1) & (tr_y < p90)] = 2.0
        w[(tr_y >= p90) & (tr_y < p95)] = 8.0
        w[tr_y >= p95] = 15.0

        reg = xgb.XGBRegressor(
            objective="reg:squarederror", learning_rate=0.03,
            max_depth=6, min_child_weight=8, subsample=0.8,
            colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
            n_estimators=2000, early_stopping_rounds=80,
            verbosity=0, n_jobs=-1, random_state=42)
        reg.fit(X_tr, tr_y, sample_weight=w, eval_set=[(X_vl, vl_y)], verbose=False)
        xgb_vl = reg.predict(X_vl).clip(min=0)
        xgb_te = reg.predict(X_te).clip(min=0)

        # Grid search on val: w_nn + rain_t
        best_score, best_wnn, best_rain_t = -999, 0.20, best_rt
        for w_nn in [0.10, 0.15, 0.20, 0.25, 0.30]:
            ens_vl = w_nn * nn_vl + (1 - w_nn) * xgb_vl
            for rt in np.arange(max(0.25, best_rt - 0.08), best_rt + 0.08, 0.02):
                rain_mask = rain_vl >= rt
                final = np.zeros_like(ens_vl)
                final[rain_mask] = ens_vl[rain_mask]
                m = evaluate(final, vl_y, thresholds, prefix="")
                far = m.get("FAR_p90", 1.0)
                score = (m.get("CSI_rain", 0) + m.get("CSI_p90", 0) * 3.0
                         + m.get("CSI_p95", 0) * 1.0 + m.get("SEDI_rain", 0) * 0.3
                         - far * 2.0)
                if score > best_score:
                    best_score = score
                    best_wnn = w_nn
                    best_rain_t = rt

        # Apply best config to test year
        ens_te = best_wnn * nn_te + (1 - best_wnn) * xgb_te
        rain_mask_te = rain_te >= best_rain_t
        final_te = np.zeros_like(ens_te)
        final_te[rain_mask_te] = ens_te[rain_mask_te]

        # Per-year metrics
        m_yr = evaluate(final_te, te_y, thresholds, prefix="")
        p90_H = ((final_te >= p90) & (te_y >= p90)).sum()
        p90_FA = ((final_te >= p90) & (te_y < p90)).sum()
        p90_M = ((final_te < p90) & (te_y >= p90)).sum()
        per_year_metrics[test_yr] = {
            "CSI_rain": m_yr.get("CSI_rain", 0),
            "CSI_p90": m_yr.get("CSI_p90", 0),
            "FAR_p90": m_yr.get("FAR_p90", 0),
            "SEDI_p90": m_yr.get("SEDI_p90", 0),
            "H": int(p90_H), "M": int(p90_M), "FA": int(p90_FA),
            "config": {"w_nn": best_wnn, "rain_t": float(best_rain_t)},
        }
        print(f"    w_nn={best_wnn}, rain_t={best_rain_t:.2f} → "
              f"CSI_rain={m_yr.get('CSI_rain',0):.3f}  "
              f"CSI_P90={m_yr.get('CSI_p90',0):.3f}  "
              f"H={p90_H} FA={p90_FA}")

        all_preds.append(final_te)
        all_truths.append(te_y)

    # ── AGGREGATE RESULTS ────────────────────────────────────────────────
    all_preds = np.concatenate(all_preds)
    all_truths = np.concatenate(all_truths)

    print("\n" + "=" * 60)
    agg_m = evaluate(all_preds, all_truths, thresholds, prefix="")
    corr = np.corrcoef(all_preds[all_truths >= 0.1], all_truths[all_truths >= 0.1])[0, 1]
    print_metrics(agg_m, title="LOYO AGGREGATE RESULTS (all years)")
    print(f"  corr_rainy = {corr:.4f}")

    # P90 contingency
    pred_p90 = all_preds >= p90
    actual_p90 = all_truths >= p90
    H = (pred_p90 & actual_p90).sum()
    M = (~pred_p90 & actual_p90).sum()
    FA = (pred_p90 & ~actual_p90).sum()
    print(f"  P90 contingency: H={H}, M={M}, FA={FA}, n_actual={actual_p90.sum()}")

    # Per-year table
    print("\n  Per-year breakdown:")
    print(f"  {'Year':<6s} {'CSI_rain':>9s} {'CSI_P90':>8s} {'FAR_P90':>8s} {'SEDI_P90':>9s} {'H':>3s} {'M':>3s} {'FA':>3s}")
    print("  " + "-" * 55)
    for yr in all_years:
        d = per_year_metrics[yr]
        print(f"  {yr:<6d} {d['CSI_rain']:>9.4f} {d['CSI_p90']:>8.4f} {d['FAR_p90']:>8.4f} {d['SEDI_p90']:>9.4f} {d['H']:>3d} {d['M']:>3d} {d['FA']:>3d}")

    # Averages
    avg_csi_rain = np.mean([per_year_metrics[yr]["CSI_rain"] for yr in all_years])
    avg_csi_p90 = np.mean([per_year_metrics[yr]["CSI_p90"] for yr in all_years])
    print(f"\n  Year-average CSI_rain = {avg_csi_rain:.4f}")
    print(f"  Year-average CSI_P90  = {avg_csi_p90:.4f}")

    # Save
    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "Leave-One-Year-Out Cross-Validation",
        "aggregate_metrics": {k: v for k, v in agg_m.items() if isinstance(v, (float, int))},
        "correlation": corr,
        "thresholds": thresholds,
        "per_year": {str(yr): per_year_metrics[yr] for yr in all_years},
    }
    with open(out_dir / "loyo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_dir / 'loyo_results.json'}")


if __name__ == "__main__":
    main()
