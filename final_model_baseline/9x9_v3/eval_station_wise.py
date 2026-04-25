"""
eval_station_wise.py — COMPLETE Station-Wise Evaluation
=========================================================
ALL metrics, ALL temporal splits, ALL stations, Model vs ECMWF.

Evaluations:
  A) Forward:   Train 2015-2023, Test 2024
  B) Reverse:   Train 2018-2024, Test 2015-2017
  C) Middle:    Train 2015-2017+2022-2024, Test 2018-2021
  D) LOYO:      Leave-One-Year-Out (10 years)

Each broken down by station with full metrics for Model AND ECMWF.
"""

import sys, json, warnings, numpy as np, pandas as pd
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
from metrics import evaluate


# ── helpers ─────────────────────────────────────────────────────────────────

def extract_spatial_features(patches):
    B, C, H, W = patches.shape
    c = H // 2
    feats = [patches[:, :, c, c], patches.mean(axis=(2,3)),
             patches.reshape(B,C,-1).max(axis=2), patches.std(axis=(2,3)),
             patches[:, :, c, c] - patches.mean(axis=(2,3)),
             patches[:, :, c-1:c+2, c-1:c+2].mean(axis=(2,3))]
    dx = patches[:,:,c,min(c+1,W-1)] - patches[:,:,c,max(c-1,0)]
    dy = patches[:,:,min(c+1,H-1),c] - patches[:,:,max(c-1,0),c]
    feats.append(np.sqrt(dx**2 + dy**2))
    return np.hstack(feats)

def build_features(patches, tabulars, norm, nn_preds=None):
    p, t = norm.transform_patches(patches), norm.transform_tabular(tabulars)
    parts = [extract_spatial_features(p), t]
    if nn_preds is not None: parts.append(nn_preds.reshape(-1,1))
    return np.hstack(parts)

def nn_predict(model, patches, tabulars, norm):
    p = torch.from_numpy(norm.transform_patches(patches)).float()
    t = torch.from_numpy(norm.transform_tabular(tabulars)).float()
    loader = DataLoader(TensorDataset(p, t), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for pb, tb in loader:
            preds.append(model.predict(pb, tb).numpy())
    return np.concatenate(preds)


ALL_METRIC_KEYS = [
    "CSI_rain", "POD_rain", "FAR_rain", "SEDI_rain",
    "CSI_p90", "POD_p90", "FAR_p90", "SEDI_p90",
    "CSI_p95", "POD_p95", "FAR_p95", "SEDI_p95",
    "RMSE", "MAE", "corr_rainy",
]

def full_metrics(preds, targets, thresholds, p90):
    m = evaluate(preds, targets, thresholds, prefix="")
    corr = 0.0
    rainy = targets >= 0.1
    if rainy.sum() > 2:
        c = np.corrcoef(preds[rainy], targets[rainy])[0, 1]
        corr = float(c) if not np.isnan(c) else 0.0
    out = {}
    for k in ALL_METRIC_KEYS:
        if k == "corr_rainy":
            out[k] = corr
        else:
            out[k] = m.get(k, 0.0)
    out["n_samples"] = len(targets)
    out["n_rainy"] = int(rainy.sum())
    out["n_p90"] = int((targets >= p90).sum())
    out["n_p95"] = int((targets >= thresholds["p95"]).sum())
    return out


# ── 3-stage ensemble ───────────────────────────────────────────────────────

def train_and_predict(X_tr, y_tr, X_vl, y_vl, X_te, nn_vl, nn_te, p90, p95, thresholds):
    """Train ensemble, grid search, return test predictions."""
    # Stage 1: Rain
    yr = (y_tr >= 0.1).astype(int)
    yv = (y_vl >= 0.1).astype(int)
    spw = max((yr==0).sum(),1) / max((yr==1).sum(),1)
    clf_rain = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=spw)
    clf_rain.fit(X_tr, yr, eval_set=[(X_vl, yv)], verbose=False)

    # Stage 2: Regression
    w = np.ones(len(y_tr), dtype=np.float32)
    w[(y_tr>=0.1)&(y_tr<p90)] = 2.0
    w[(y_tr>=p90)&(y_tr<p95)] = 10.0
    w[y_tr>=p95] = 20.0
    reg = xgb.XGBRegressor(
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_vl, y_vl)], verbose=False)

    # Stage 3: Extreme
    ye = (y_tr >= p90).astype(int)
    yev = (y_vl >= p90).astype(int)
    spw2 = max((ye==0).sum(),1) / max((ye==1).sum(),1)
    clf_ext = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.03, max_depth=4,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        gamma=1.5, reg_alpha=1.0, reg_lambda=3.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=spw2)
    clf_ext.fit(X_tr, ye, eval_set=[(X_vl, yev)], verbose=False)

    # Isotonic
    iso = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
    iso.fit(nn_vl, y_vl)

    # Grid search
    rain_vl = clf_rain.predict_proba(X_vl)[:,1]
    xgb_vl = reg.predict(X_vl).clip(min=0)
    ext_vl = clf_ext.predict_proba(X_vl)[:,1]
    nn_vl_cal = iso.predict(nn_vl)

    best_score, best_cfg = -999, None
    for w_nn in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for use_cal in [False, True]:
            nv = nn_vl_cal if use_cal else nn_vl
            ev = w_nn * nv + (1-w_nn) * xgb_vl
            for rain_t in np.arange(0.30, 0.60, 0.03):
                rm = rain_vl >= rain_t
                gated = np.zeros_like(ev); gated[rm] = ev[rm]
                for eb in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    f = gated.copy()
                    if eb > 0:
                        em = ext_vl >= 0.5
                        f[em] = f[em] * (1.0 + eb * ext_vl[em])
                    m = evaluate(f, y_vl, thresholds, prefix="")
                    s = (m.get("CSI_rain",0)*1 + m.get("CSI_p90",0)*3
                         + m.get("CSI_p95",0)*2 + m.get("SEDI_p90",0)*1
                         - m.get("FAR_p90",1)*2 - m.get("FAR_rain",1)*0.5)
                    if s > best_score:
                        best_score = s
                        best_cfg = {"w_nn":w_nn, "rain_t":float(rain_t),
                                    "ext_boost":eb, "use_cal":use_cal}

    # Apply to test
    rain_te = clf_rain.predict_proba(X_te)[:,1]
    xgb_te = reg.predict(X_te).clip(min=0)
    ext_te = clf_ext.predict_proba(X_te)[:,1]
    nn_te_use = iso.predict(nn_te) if best_cfg["use_cal"] else nn_te
    ens = best_cfg["w_nn"] * nn_te_use + (1-best_cfg["w_nn"]) * xgb_te
    rm = rain_te >= best_cfg["rain_t"]
    final = np.zeros_like(ens); final[rm] = ens[rm]
    if best_cfg["ext_boost"] > 0:
        em = ext_te >= 0.5
        final[em] = final[em] * (1.0 + best_cfg["ext_boost"] * ext_te[em])
    return final


# ── station assignment ─────────────────────────────────────────────────────

def assign_stations(all_targets, all_years_arr):
    gt = pd.read_csv(config.GROUND_TRUTH, parse_dates=["Date"])
    gt = gt.dropna(subset=["Rainfall_mm"])
    gt = gt[gt["Date"].dt.month.isin(config.MONSOON_MONTHS)]
    gt["year"] = gt["Date"].dt.year

    stations = []
    idx = 0
    for year in range(2015, 2025):
        n = (all_years_arr == year).sum()
        gt_year = gt[gt["year"] == year]
        potential = []
        for month in config.MONSOON_MONTHS:
            for _, row in gt_year[gt_year["Date"].dt.month == month].iterrows():
                potential.append((row["Station"], float(row["Rainfall_mm"])))
        matched = 0; pot_idx = 0
        while matched < n and pot_idx < len(potential):
            stn, rain = potential[pot_idx]
            if abs(all_targets[idx + matched] - rain) < 0.01:
                stations.append(stn); matched += 1
            pot_idx += 1
        while matched < n:
            stations.append("Unknown"); matched += 1
        idx += n
    return np.array(stations)


# ── run one temporal split ─────────────────────────────────────────────────

def run_split(nn_model, patches, tabular, targets, years_arr, stations,
              train_years, test_years, thresholds, p90, p95, unique_stns, label):
    """Run one temporal split, return per-station metrics for model & ECMWF."""
    tr_mask = np.isin(years_arr, train_years)
    te_mask = np.isin(years_arr, test_years)

    tr_idx = np.where(tr_mask)[0]
    np.random.seed(99)
    perm = np.random.permutation(len(tr_idx))
    n_tr = int(0.85 * len(tr_idx))
    xgb_tr, xgb_vl = tr_idx[perm[:n_tr]], tr_idx[perm[n_tr:]]

    norm = Normaliser()
    norm.fit(patches[tr_mask], tabular[tr_mask])

    nn_tr = nn_predict(nn_model, patches[xgb_tr], tabular[xgb_tr], norm)
    nn_vl = nn_predict(nn_model, patches[xgb_vl], tabular[xgb_vl], norm)
    nn_te = nn_predict(nn_model, patches[te_mask], tabular[te_mask], norm)

    X_tr = build_features(patches[xgb_tr], tabular[xgb_tr], norm, nn_tr)
    X_vl = build_features(patches[xgb_vl], tabular[xgb_vl], norm, nn_vl)
    X_te = build_features(patches[te_mask], tabular[te_mask], norm, nn_te)

    print(f"    Train: {len(xgb_tr)} | Val: {len(xgb_vl)} | Test: {te_mask.sum()}")

    model_preds = train_and_predict(X_tr, targets[xgb_tr], X_vl, targets[xgb_vl],
                                     X_te, nn_vl, nn_te, p90, p95, thresholds)

    center = patches.shape[2] // 2
    ecmwf_preds = patches[te_mask, 0, center, center].copy().clip(min=0)

    te_targets = targets[te_mask]
    te_stations = stations[te_mask]

    # Overall
    result = {
        "ALL": {
            "model": full_metrics(model_preds, te_targets, thresholds, p90),
            "ecmwf": full_metrics(ecmwf_preds, te_targets, thresholds, p90),
        }
    }

    # Per station
    for stn in unique_stns:
        sm = te_stations == stn
        if sm.sum() == 0:
            continue
        result[stn] = {
            "model": full_metrics(model_preds[sm], te_targets[sm], thresholds, p90),
            "ecmwf": full_metrics(ecmwf_preds[sm], te_targets[sm], thresholds, p90),
        }
    return result


def print_station_table(result, unique_stns, label):
    """Print full metrics table for one split."""
    print(f"\n{'='*130}")
    print(f"  {label}")
    print(f"{'='*130}")

    # Header
    print(f"\n  {'Station':>15s} | {'CSI_r':>6s} {'POD_r':>6s} {'FAR_r':>6s}"
          f" | {'CSI_90':>6s} {'POD_90':>6s} {'FAR_90':>6s} {'SEDI_90':>7s}"
          f" | {'CSI_95':>6s} {'POD_95':>6s} {'FAR_95':>6s} {'SEDI_95':>7s}"
          f" | {'RMSE':>6s} {'MAE':>6s} {'corr':>6s}"
          f" | {'n':>4s} {'P90':>3s} {'P95':>3s}")

    def print_row(name, m):
        print(f"  {name:>15s}"
              f" | {m['CSI_rain']:>6.3f} {m['POD_rain']:>6.3f} {m['FAR_rain']:>6.3f}"
              f" | {m['CSI_p90']:>6.3f} {m['POD_p90']:>6.3f} {m['FAR_p90']:>6.3f} {m['SEDI_p90']:>7.3f}"
              f" | {m['CSI_p95']:>6.3f} {m['POD_p95']:>6.3f} {m['FAR_p95']:>6.3f} {m['SEDI_p95']:>7.3f}"
              f" | {m['RMSE']:>6.1f} {m['MAE']:>6.1f} {m['corr_rainy']:>6.3f}"
              f" | {m['n_samples']:>4d} {m['n_p90']:>3d} {m.get('n_p95',0):>3d}")

    # MODEL
    print(f"\n  --- MODEL ---")
    print("  " + "-" * 125)
    for stn in ["ALL"] + list(unique_stns):
        if stn in result:
            print_row(stn, result[stn]["model"])

    # ECMWF
    print(f"\n  --- ECMWF RAW ---")
    print("  " + "-" * 125)
    for stn in ["ALL"] + list(unique_stns):
        if stn in result:
            print_row(stn, result[stn]["ecmwf"])


def main():
    print("\n" + "=" * 60)
    print("  COMPLETE STATION-WISE EVALUATION")
    print("  ALL metrics · ALL splits · Model vs ECMWF")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading ALL data...")
    builder = RainfallDataBuilder(window_size=config.DEFAULT_WINDOW)
    all_years = list(range(2015, 2025))
    all_p, all_t, all_y = [], [], []
    years_arr = []
    for yr in all_years:
        p, t, y = builder.build([yr])
        all_p.append(p); all_t.append(t); all_y.append(y)
        years_arr.extend([yr] * len(y))
    patches = np.concatenate(all_p)
    tabular = np.concatenate(all_t)
    targets = np.concatenate(all_y)
    years_arr = np.array(years_arr)

    rainy = targets[targets >= 0.1]
    p90 = float(np.percentile(rainy, 90))
    p95 = float(np.percentile(rainy, 95))
    thresholds = {"p90": p90, "p95": p95, "p99": float(np.percentile(rainy, 99))}
    print(f"  Total: {len(targets)} | P90={p90:.1f}mm | P95={p95:.1f}mm")

    # Stations
    print("\n[2/3] Assigning stations...")
    stations = assign_stations(targets, years_arr)
    unique_stns = sorted(set(stations) - {"Unknown"})
    for s in unique_stns:
        print(f"    {s:20s}: {(stations==s).sum()} samples")

    # NN
    print("\n[3/3] Loading SmallNet...")
    nn_model = build_model(window_size=config.DEFAULT_WINDOW, n_channels=19, n_tabular=24)
    ckpt_dir = config.OUTPUT_DIR / f"window_{config.DEFAULT_WINDOW}"
    pts = sorted(ckpt_dir.glob("*.pt"))
    ckpt = torch.load(str(pts[-1]), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"]); nn_model.eval()
    print(f"  Loaded: {pts[-1].name}")

    all_results = {}

    # ═══ A) Forward: Train 2015-2023, Test 2024 ═══
    print("\n  A) FORWARD: Train 2015-2023 -> Test 2024")
    r = run_split(nn_model, patches, tabular, targets, years_arr, stations,
                  list(range(2015,2024)), [2024], thresholds, p90, p95, unique_stns,
                  "A) FORWARD 2015-2023 -> 2024")
    print_station_table(r, unique_stns, "A) FORWARD: Train 2015-2023 -> Test 2024")
    all_results["forward_2024"] = r

    # ═══ B) Reverse: Train 2018-2024, Test 2015-2017 ═══
    print("\n  B) REVERSE: Train 2018-2024 -> Test 2015-2017")
    r = run_split(nn_model, patches, tabular, targets, years_arr, stations,
                  list(range(2018,2025)), [2015,2016,2017], thresholds, p90, p95, unique_stns,
                  "B) REVERSE 2018-2024 -> 2015-2017")
    print_station_table(r, unique_stns, "B) REVERSE: Train 2018-2024 -> Test 2015-2017")
    all_results["reverse_2015_2017"] = r

    # ═══ C) Middle-Out ═══
    print("\n  C) MIDDLE-OUT: Train 2015-2017+2022-2024 -> Test 2018-2021")
    r = run_split(nn_model, patches, tabular, targets, years_arr, stations,
                  [2015,2016,2017,2022,2023,2024], [2018,2019,2020,2021],
                  thresholds, p90, p95, unique_stns,
                  "C) MIDDLE-OUT -> 2018-2021")
    print_station_table(r, unique_stns, "C) MIDDLE-OUT: Train edges -> Test 2018-2021")
    all_results["middle_2018_2021"] = r

    # ═══ D) LOYO per station ═══
    print("\n  D) LOYO — Leave-One-Year-Out")
    loyo = {}
    for test_yr in all_years:
        print(f"    LOYO test={test_yr}...")
        train_yrs = [y for y in all_years if y != test_yr]
        r = run_split(nn_model, patches, tabular, targets, years_arr, stations,
                      train_yrs, [test_yr], thresholds, p90, p95, unique_stns,
                      f"LOYO test={test_yr}")
        loyo[test_yr] = r

    all_results["loyo"] = {}

    # LOYO summary per station (mean across years)
    print(f"\n{'='*130}")
    print(f"  D) LOYO MEAN — Station-wise (averaged across 10 years)")
    print(f"{'='*130}")

    print(f"\n  --- MODEL (LOYO MEAN) ---")
    print(f"  {'Station':>15s} | {'CSI_r':>6s} {'POD_r':>6s} {'FAR_r':>6s}"
          f" | {'CSI_90':>6s} {'POD_90':>6s} {'FAR_90':>6s} {'SEDI_90':>7s}"
          f" | {'CSI_95':>6s} {'POD_95':>6s} {'FAR_95':>6s} {'SEDI_95':>7s}"
          f" | {'RMSE':>6s} {'MAE':>6s} {'corr':>6s}")
    print("  " + "-" * 125)

    for stn in ["ALL"] + list(unique_stns):
        vals = {k: [] for k in ALL_METRIC_KEYS}
        for yr in all_years:
            if stn in loyo[yr]:
                for k in ALL_METRIC_KEYS:
                    vals[k].append(loyo[yr][stn]["model"][k])
        if not vals["CSI_rain"]:
            continue
        mm = {k: np.mean(v) for k, v in vals.items()}
        print(f"  {stn:>15s}"
              f" | {mm['CSI_rain']:>6.3f} {mm['POD_rain']:>6.3f} {mm['FAR_rain']:>6.3f}"
              f" | {mm['CSI_p90']:>6.3f} {mm['POD_p90']:>6.3f} {mm['FAR_p90']:>6.3f} {mm['SEDI_p90']:>7.3f}"
              f" | {mm['CSI_p95']:>6.3f} {mm['POD_p95']:>6.3f} {mm['FAR_p95']:>6.3f} {mm['SEDI_p95']:>7.3f}"
              f" | {mm['RMSE']:>6.1f} {mm['MAE']:>6.1f} {mm['corr_rainy']:>6.3f}")
        all_results["loyo"][f"model_{stn}"] = mm

    print(f"\n  --- ECMWF RAW (LOYO MEAN) ---")
    print(f"  {'Station':>15s} | {'CSI_r':>6s} {'POD_r':>6s} {'FAR_r':>6s}"
          f" | {'CSI_90':>6s} {'POD_90':>6s} {'FAR_90':>6s} {'SEDI_90':>7s}"
          f" | {'CSI_95':>6s} {'POD_95':>6s} {'FAR_95':>6s} {'SEDI_95':>7s}"
          f" | {'RMSE':>6s} {'MAE':>6s} {'corr':>6s}")
    print("  " + "-" * 125)

    for stn in ["ALL"] + list(unique_stns):
        vals = {k: [] for k in ALL_METRIC_KEYS}
        for yr in all_years:
            if stn in loyo[yr]:
                for k in ALL_METRIC_KEYS:
                    vals[k].append(loyo[yr][stn]["ecmwf"][k])
        if not vals["CSI_rain"]:
            continue
        em = {k: np.mean(v) for k, v in vals.items()}
        print(f"  {stn:>15s}"
              f" | {em['CSI_rain']:>6.3f} {em['POD_rain']:>6.3f} {em['FAR_rain']:>6.3f}"
              f" | {em['CSI_p90']:>6.3f} {em['POD_p90']:>6.3f} {em['FAR_p90']:>6.3f} {em['SEDI_p90']:>7.3f}"
              f" | {em['CSI_p95']:>6.3f} {em['POD_p95']:>6.3f} {em['FAR_p95']:>6.3f} {em['SEDI_p95']:>7.3f}"
              f" | {em['RMSE']:>6.1f} {em['MAE']:>6.1f} {em['corr_rainy']:>6.3f}")
        all_results["loyo"][f"ecmwf_{stn}"] = em

    # Save
    out_dir = config.OUTPUT_DIR / "final_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out_dir / "station_wise_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n  Saved to: {out_dir / 'station_wise_full_results.json'}")


if __name__ == "__main__":
    main()
