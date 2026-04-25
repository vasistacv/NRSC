"""
eval_all_temporal.py — Multi-Directional Temporal Evaluation
=============================================================
Tests the model's true temporal generalization in ALL directions:

  A) Forward:  Train 2015-2023, Test 2024 (standard)
  B) Reverse:  Train 2018-2024, Test 2015-2017 (backward prediction)
  C) Middle:   Train 2015-2017 + 2022-2024, Test 2018-2021 (gap prediction)
  D) LOYO:     Leave-One-Year-Out for each year 2015-2024 (ultimate robustness test)

If the model has TRUE skill, ALL of these should show good results.
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
# ECMWF RAW BASELINE
# ──────────────────────────────────────────────────────────────────────────────

def ecmwf_baseline(patches, targets, thresholds, p90):
    """
    Compute raw ECMWF skill: tp center pixel.
    Channel 0 = tp, ALREADY in mm (converted during patch build in dataset.py).
    Center = (4,4) for 9x9 patch.
    """
    center = patches.shape[2] // 2
    ecmwf_mm = patches[:, 0, center, center].copy()  # already in mm
    ecmwf_mm = ecmwf_mm.clip(min=0)

    m = evaluate(ecmwf_mm, targets, thresholds, prefix="")
    corr = 0
    rainy = targets >= 0.1
    if rainy.sum() > 2:
        corr = float(np.corrcoef(ecmwf_mm[rainy], targets[rainy])[0, 1])

    return {
        "CSI_rain": m.get("CSI_rain", 0),
        "POD_rain": m.get("POD_rain", 0),
        "FAR_rain": m.get("FAR_rain", 0),
        "CSI_p90": m.get("CSI_p90", 0),
        "POD_p90": m.get("POD_p90", 0),
        "FAR_p90": m.get("FAR_p90", 0),
        "SEDI_p90": m.get("SEDI_p90", 0),
        "CSI_p95": m.get("CSI_p95", 0),
        "POD_p95": m.get("POD_p95", 0),
        "FAR_p95": m.get("FAR_p95", 0),
        "SEDI_p95": m.get("SEDI_p95", 0),
        "RMSE": m.get("RMSE", 0),
        "MAE": m.get("MAE", 0),
        "corr_rainy": corr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# REUSE: spatial features + NN predict from train_dual_eval
# ──────────────────────────────────────────────────────────────────────────────

def extract_spatial_features(patches):
    B, C, H, W = patches.shape
    center = H // 2
    feats = []
    feats.append(patches[:, :, center, center])
    feats.append(patches.mean(axis=(2, 3)))
    feats.append(patches.reshape(B, C, -1).max(axis=2))
    feats.append(patches.std(axis=(2, 3)))
    feats.append(patches[:, :, center, center] - patches.mean(axis=(2, 3)))
    inner = patches[:, :, center-1:center+2, center-1:center+2]
    feats.append(inner.mean(axis=(2, 3)))
    dx = patches[:, :, center, min(center+1, W-1)] - patches[:, :, center, max(center-1, 0)]
    dy = patches[:, :, min(center+1, H-1), center] - patches[:, :, max(center-1, 0), center]
    feats.append(np.sqrt(dx**2 + dy**2))
    return np.hstack(feats)


def build_features(patches, tabulars, norm_obj, nn_preds=None):
    p = norm_obj.transform_patches(patches)
    t = norm_obj.transform_tabular(tabulars)
    spatial = extract_spatial_features(p)
    parts = [spatial, t]
    if nn_preds is not None:
        parts.append(nn_preds.reshape(-1, 1))
    return np.hstack(parts)


def nn_predict(nn_model, patches, tabulars, norm_obj):
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
# ENSEMBLE PIPELINE (same 3-stage as train_dual_eval)
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                 nn_vl, nn_te, thresholds, p90, label=""):
    """3-stage ensemble: rain gate + intensity + extreme boost."""
    print(f"\n  {'='*55}")
    print(f"  {label}")
    print(f"  {'='*55}")

    p95 = thresholds["p95"]

    # Stage 1: Rain classifier
    y_rain_tr = (y_tr >= 0.1).astype(int)
    y_rain_vl = (y_vl >= 0.1).astype(int)
    n_dry, n_wet = (y_rain_tr == 0).sum(), max((y_rain_tr == 1).sum(), 1)
    clf_rain = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_dry/n_wet)
    clf_rain.fit(X_tr, y_rain_tr, eval_set=[(X_vl, y_rain_vl)], verbose=False)
    rain_vl = clf_rain.predict_proba(X_vl)[:, 1]
    rain_te = clf_rain.predict_proba(X_te)[:, 1]

    # Stage 2: Intensity regressor
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

    # Stage 3: Extreme classifier
    y_ext_tr = (y_tr >= p90).astype(int)
    y_ext_vl = (y_vl >= p90).astype(int)
    n_normal, n_extreme = (y_ext_tr == 0).sum(), max((y_ext_tr == 1).sum(), 1)
    clf_ext = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.03, max_depth=4,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        gamma=1.5, reg_alpha=1.0, reg_lambda=3.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_normal/n_extreme)
    clf_ext.fit(X_tr, y_ext_tr, eval_set=[(X_vl, y_ext_vl)], verbose=False)
    ext_vl = clf_ext.predict_proba(X_vl)[:, 1]
    ext_te = clf_ext.predict_proba(X_te)[:, 1]

    # Isotonic calibration
    iso = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
    iso.fit(nn_vl, y_vl)
    nn_vl_cal = iso.predict(nn_vl)
    nn_te_cal = iso.predict(nn_te)

    # Grid search on validation
    best_score, best_cfg, best_m = -999, None, None
    for w_nn in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for use_cal in [False, True]:
            nn_v = nn_vl_cal if use_cal else nn_vl
            ens_vl = w_nn * nn_v + (1 - w_nn) * xgb_vl
            for rain_t in np.arange(0.30, 0.60, 0.02):
                rain_mask = rain_vl >= rain_t
                gated = np.zeros_like(ens_vl)
                gated[rain_mask] = ens_vl[rain_mask]
                for ext_boost in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    final = gated.copy()
                    if ext_boost > 0:
                        ext_mask = ext_vl >= 0.5
                        final[ext_mask] = final[ext_mask] * (1.0 + ext_boost * ext_vl[ext_mask])
                    m = evaluate(final, y_vl, thresholds, prefix="")
                    score = (m.get("CSI_rain", 0) * 1.0
                             + m.get("CSI_p90", 0) * 3.0
                             + m.get("CSI_p95", 0) * 2.0
                             + m.get("SEDI_p90", 0) * 1.0
                             + m.get("SEDI_p95", 0) * 0.5
                             - m.get("FAR_p90", 1.0) * 2.0
                             - m.get("FAR_rain", 1.0) * 0.5)
                    if score > best_score:
                        best_score = score
                        best_cfg = {"w_nn": w_nn, "rain_t": float(rain_t),
                                    "ext_boost": ext_boost, "use_cal": use_cal}
                        best_m = m

    # Apply to test
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
    rainy = y_te >= 0.1
    if rainy.sum() > 2:
        corr = float(np.corrcoef(final_te[rainy], y_te[rainy])[0, 1])

    # Print key metrics
    print(f"    Config: w_nn={best_cfg['w_nn']}, rain_t={best_cfg['rain_t']:.2f}, "
          f"ext={best_cfg['ext_boost']}, cal={best_cfg['use_cal']}")
    print(f"    CSI_rain={test_m.get('CSI_rain',0):.4f}  FAR_rain={test_m.get('FAR_rain',0):.4f}")
    print(f"    CSI_p90 ={test_m.get('CSI_p90',0):.4f}  FAR_p90 ={test_m.get('FAR_p90',0):.4f}  "
          f"SEDI_p90={test_m.get('SEDI_p90',0):.4f}")
    print(f"    CSI_p95 ={test_m.get('CSI_p95',0):.4f}  FAR_p95 ={test_m.get('FAR_p95',0):.4f}  "
          f"SEDI_p95={test_m.get('SEDI_p95',0):.4f}")
    print(f"    corr_rainy={corr:.4f}  RMSE={test_m.get('RMSE',0):.2f}")

    return test_m, corr, best_cfg


def run_temporal_test(nn_model, all_patches, all_tabular, all_targets,
                      year_indices, train_years, test_years, thresholds, p90, label):
    """Run a single temporal test + ECMWF baseline."""
    tr_mask = np.zeros(len(all_targets), dtype=bool)
    te_mask = np.zeros(len(all_targets), dtype=bool)
    for yr, start, end in year_indices:
        if yr in train_years:
            tr_mask[start:end] = True
        elif yr in test_years:
            te_mask[start:end] = True

    tr_idx = np.where(tr_mask)[0]
    np.random.seed(99)
    perm = np.random.permutation(len(tr_idx))
    n_tr = int(0.85 * len(tr_idx))
    xgb_tr = tr_idx[perm[:n_tr]]
    xgb_vl = tr_idx[perm[n_tr:]]

    norm = Normaliser()
    norm.fit(all_patches[tr_mask], all_tabular[tr_mask])

    nn_tr = nn_predict(nn_model, all_patches[xgb_tr], all_tabular[xgb_tr], norm)
    nn_vl = nn_predict(nn_model, all_patches[xgb_vl], all_tabular[xgb_vl], norm)
    nn_te = nn_predict(nn_model, all_patches[te_mask], all_tabular[te_mask], norm)

    X_tr = build_features(all_patches[xgb_tr], all_tabular[xgb_tr], norm, nn_tr)
    X_vl = build_features(all_patches[xgb_vl], all_tabular[xgb_vl], norm, nn_vl)
    X_te = build_features(all_patches[te_mask], all_tabular[te_mask], norm, nn_te)

    # ECMWF raw baseline for this test set
    ecmwf_m = ecmwf_baseline(all_patches[te_mask], all_targets[te_mask], thresholds, p90)

    n_test_p90 = (all_targets[te_mask] >= p90).sum()
    print(f"  Train: {len(xgb_tr)} | Val: {len(xgb_vl)} | Test: {te_mask.sum()} "
          f"| Test P90 events: {n_test_p90}")
    print(f"  ECMWF raw:  CSI_rain={ecmwf_m['CSI_rain']:.4f}  CSI_p90={ecmwf_m['CSI_p90']:.4f}  "
          f"RMSE={ecmwf_m['RMSE']:.2f}  corr={ecmwf_m['corr_rainy']:.4f}")

    m, c, cfg = run_ensemble(X_tr, all_targets[xgb_tr], X_vl, all_targets[xgb_vl],
                              X_te, all_targets[te_mask], nn_vl, nn_te,
                              thresholds, p90, label)
    return m, c, cfg, ecmwf_m


def main():
    print("\n" + "=" * 60)
    print("  MULTI-DIRECTIONAL TEMPORAL EVALUATION")
    print("  Testing TRUE model skill across ALL time periods")
    print("=" * 60)

    # Load all data
    print("\n[1/2] Loading ALL data...")
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

    rainy_all = all_targets[all_targets >= 0.1]
    p90 = float(np.percentile(rainy_all, 90))
    p95 = float(np.percentile(rainy_all, 95))
    p99 = float(np.percentile(rainy_all, 99))
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    print(f"  Total: {len(all_targets)} | P90={p90:.1f}mm | P95={p95:.1f}mm")

    # Load SmallNet
    print("\n[2/2] Loading SmallNet...")
    nn_model = build_model(window_size=config.DEFAULT_WINDOW, n_channels=19, n_tabular=24)
    ckpt_dir = config.OUTPUT_DIR / f"window_{config.DEFAULT_WINDOW}"
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"  ERROR: No checkpoints in {ckpt_dir}"); sys.exit(1)
    ckpt = torch.load(str(pts[-1]), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()
    print(f"  Loaded: {pts[-1].name}")

    results = {}

    # ═══════════════════════════════════════════════════════════════════
    # A) FORWARD: Train 2015-2023, Test 2024
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  A) FORWARD: Train 2015-2023 -> Test 2024")
    print("=" * 60)
    m, c, cfg, ecmwf_fwd = run_temporal_test(
        nn_model, all_patches, all_tabular, all_targets, year_indices,
        train_years=list(range(2015, 2024)), test_years=[2024],
        thresholds=thresholds, p90=p90, label="FORWARD 2015-2023 -> 2024")
    results["forward_2024"] = {"metrics": {k: v for k, v in m.items() if isinstance(v, (float, int))},
                                "corr": c, "config": cfg, "ecmwf": ecmwf_fwd}

    # ═══════════════════════════════════════════════════════════════════
    # B) REVERSE: Train 2018-2024, Test 2015-2017
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  B) REVERSE: Train 2018-2024 -> Test 2015-2017")
    print("=" * 60)
    m, c, cfg, ecmwf_rev = run_temporal_test(
        nn_model, all_patches, all_tabular, all_targets, year_indices,
        train_years=list(range(2018, 2025)), test_years=[2015, 2016, 2017],
        thresholds=thresholds, p90=p90, label="REVERSE 2018-2024 -> 2015-2017")
    results["reverse_2015_2017"] = {"metrics": {k: v for k, v in m.items() if isinstance(v, (float, int))},
                                     "corr": c, "config": cfg, "ecmwf": ecmwf_rev}

    # ═══════════════════════════════════════════════════════════════════
    # C) MIDDLE-OUT: Train edges, Test middle
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  C) MIDDLE-OUT: Train 2015-2017+2022-2024 -> Test 2018-2021")
    print("=" * 60)
    m, c, cfg, ecmwf_mid = run_temporal_test(
        nn_model, all_patches, all_tabular, all_targets, year_indices,
        train_years=[2015, 2016, 2017, 2022, 2023, 2024],
        test_years=[2018, 2019, 2020, 2021],
        thresholds=thresholds, p90=p90, label="MIDDLE-OUT edges -> 2018-2021")
    results["middle_2018_2021"] = {"metrics": {k: v for k, v in m.items() if isinstance(v, (float, int))},
                                    "corr": c, "config": cfg, "ecmwf": ecmwf_mid}

    # ═══════════════════════════════════════════════════════════════════
    # D) LOYO: Leave-One-Year-Out (each year as test)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  D) LEAVE-ONE-YEAR-OUT (LOYO)")
    print("=" * 60)
    loyo_results = {}
    loyo_ecmwf = {}
    for test_yr in all_years:
        train_yrs = [y for y in all_years if y != test_yr]
        m, c, cfg, ecmwf_m = run_temporal_test(
            nn_model, all_patches, all_tabular, all_targets, year_indices,
            train_years=train_yrs, test_years=[test_yr],
            thresholds=thresholds, p90=p90, label=f"LOYO test={test_yr}")
        loyo_results[str(test_yr)] = {
            "CSI_rain": m.get("CSI_rain", 0),
            "CSI_p90": m.get("CSI_p90", 0),
            "FAR_p90": m.get("FAR_p90", 0),
            "SEDI_p90": m.get("SEDI_p90", 0),
            "CSI_p95": m.get("CSI_p95", 0),
            "SEDI_p95": m.get("SEDI_p95", 0),
            "corr_rainy": c,
            "RMSE": m.get("RMSE", 0),
        }
        loyo_ecmwf[str(test_yr)] = ecmwf_m
    results["loyo"] = loyo_results
    results["loyo_ecmwf"] = loyo_ecmwf

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE: MODEL vs ECMWF
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 115)
    print("  LOYO SUMMARY — Model vs ECMWF Raw Baseline")
    print("=" * 115)
    print(f"  {'Year':<6s} | {'--- MODEL ---':^50s} | {'--- ECMWF RAW ---':^50s}")
    print(f"  {'':6s} | {'CSI_r':>7s} {'CSI_90':>7s} {'FAR_90':>7s} {'SEDI_90':>8s} {'CSI_95':>7s} {'corr':>7s} {'RMSE':>7s}"
          f" | {'CSI_r':>7s} {'CSI_90':>7s} {'FAR_90':>7s} {'SEDI_90':>8s} {'CSI_95':>7s} {'corr':>7s} {'RMSE':>7s}")
    print("  " + "-" * 110)

    for yr in all_years:
        r = loyo_results[str(yr)]
        e = loyo_ecmwf[str(yr)]
        print(f"  {yr:<6d}"
              f" | {r['CSI_rain']:>7.3f} {r['CSI_p90']:>7.3f} {r['FAR_p90']:>7.3f} {r['SEDI_p90']:>8.3f}"
              f" {r['CSI_p95']:>7.3f} {r['corr_rainy']:>7.3f} {r.get('RMSE',0):>7.1f}"
              f" | {e['CSI_rain']:>7.3f} {e['CSI_p90']:>7.3f} {e['FAR_p90']:>7.3f} {e['SEDI_p90']:>8.3f}"
              f" {e['CSI_p95']:>7.3f} {e['corr_rainy']:>7.3f} {e.get('RMSE',0):>7.1f}")

    print("  " + "-" * 110)
    # Means
    def loyo_mean(d, key):
        return np.mean([d[str(y)][key] for y in all_years])

    print(f"  {'MEAN':<6s}"
          f" | {loyo_mean(loyo_results,'CSI_rain'):>7.3f} {loyo_mean(loyo_results,'CSI_p90'):>7.3f}"
          f" {loyo_mean(loyo_results,'FAR_p90'):>7.3f} {loyo_mean(loyo_results,'SEDI_p90'):>8.3f}"
          f" {loyo_mean(loyo_results,'CSI_p95'):>7.3f} {loyo_mean(loyo_results,'corr_rainy'):>7.3f}"
          f" {loyo_mean(loyo_results,'RMSE'):>7.1f}"
          f" | {loyo_mean(loyo_ecmwf,'CSI_rain'):>7.3f} {loyo_mean(loyo_ecmwf,'CSI_p90'):>7.3f}"
          f" {loyo_mean(loyo_ecmwf,'FAR_p90'):>7.3f} {loyo_mean(loyo_ecmwf,'SEDI_p90'):>8.3f}"
          f" {loyo_mean(loyo_ecmwf,'CSI_p95'):>7.3f} {loyo_mean(loyo_ecmwf,'corr_rainy'):>7.3f}"
          f" {loyo_mean(loyo_ecmwf,'RMSE'):>7.1f}")

    # Save
    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multi_temporal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_dir / 'multi_temporal_results.json'}")


if __name__ == "__main__":
    main()
