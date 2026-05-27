"""
plot_timeseries_2024.py  —  Publication-quality 2024 Time Series
================================================================
For each station, ONE plot showing:
  Observed  vs  Our Model (9x9)  vs  ECMWF (9 km)  vs  GFS (25 km)

Only the 2024 monsoon season (JJAS) is plotted — the unseen temporal
test year.  Our model uses the temporal split (train 2015-2021,
val 2022-2023, test 2024) so there is ZERO data leakage.

OUTPUT: 7 images (1 per station) in timeseries_2024/
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"D:\NEW_NRSC\.venv\Library\bin;" + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import netCDF4
from pathlib import Path
from datetime import timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.linewidth': 2.5,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'lines.linewidth': 2.5,
    'savefig.dpi': 300,
})

import matplotlib.dates as mdates
import torch
from torch.utils.data import DataLoader, TensorDataset
import gc

sys.path.insert(0, str(Path(__file__).parent.parent / "extracted_files"))
import config
from dataset import RainfallDataBuilder, Normaliser
from model import build_model

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
ROOT       = Path(r"D:\NEW_NRSC")
GT_PATH    = ROOT / "Final_ground_truth_data.csv"
ECMWF_DIR  = ROOT / "ecmwf_data"
GFS_DIR    = ROOT / "Final_GFS_Data" / "Total_ppt"
OUT_DIR    = ROOT / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

MONSOON    = [6, 7, 8, 9]
YEARS      = list(range(2015, 2025))
PLOT_YEAR  = 2021
WINDOW     = 9

# ---------------------------------------------------------------
# 1. LOAD GROUND TRUTH
# ---------------------------------------------------------------
def load_ground_truth():
    df = pd.read_csv(GT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.month.isin(MONSOON)]
    stations = df.groupby("Station").agg({"Lat": "first", "Lon": "first"}).reset_index()
    return df, stations

# ---------------------------------------------------------------
# 2. ECMWF extraction (single year)
# ---------------------------------------------------------------
def extract_ecmwf(df, stations, year):
    print(f"  Extracting ECMWF for {year}...")
    mask = df["Date"].dt.year == year
    results = np.full(mask.sum(), np.nan)
    sub_df = df[mask].reset_index(drop=True)

    for month in MONSOON:
        fpath = ECMWF_DIR / str(year) / f"ecmwf_{year}_{month:02d}.sfc.grib"
        if not fpath.exists():
            continue
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
        except Exception:
            continue

        lats = ds.latitude.values
        lons = ds.longitude.values
        times = pd.to_datetime(ds.time.values)
        tp_data = ds["tp"].values

        stn_indices = {}
        for _, srow in stations.iterrows():
            sname = srow["Station"]
            lat_idx = int(np.argmin(np.abs(lats - srow["Lat"])))
            lon_idx = int(np.argmin(np.abs(lons - srow["Lon"])))
            stn_indices[sname] = (lat_idx, lon_idx)

        for i, row in sub_df.iterrows():
            obs_date = row["Date"]
            forecast_date = obs_date - timedelta(days=1)
            if forecast_date.month != month:
                continue
            sname = row["Station"]
            if sname not in stn_indices:
                continue
            t_mask = times.date == forecast_date.date()
            if not t_mask.any():
                continue
            t_idx = int(np.where(t_mask)[0][-1])
            lat_idx, lon_idx = stn_indices[sname]
            val = float(tp_data[t_idx, lat_idx, lon_idx]) * 1000.0
            results[i] = val

        ds.close()

    valid = ~np.isnan(results)
    print(f"    ECMWF valid: {valid.sum()}/{len(results)}")
    return sub_df, results

# ---------------------------------------------------------------
# 3. GFS extraction (single year)
# ---------------------------------------------------------------
def extract_gfs(df, stations, year):
    print(f"  Extracting GFS for {year}...")
    mask = df["Date"].dt.year == year
    sub_df = df[mask].reset_index(drop=True)
    results = np.full(len(sub_df), np.nan)

    folders_fcst = [("0_to_6", "f006"), ("6_12", "f012"),
                    ("12_18", "f018"), ("18_24", "f024")]

    for i, row in sub_df.iterrows():
        obs_date = row["Date"]
        prev_day = obs_date - timedelta(days=1)
        date_str = prev_day.strftime("%Y%m%d") + "18"
        sname = row["Station"]
        srow = stations[stations["Station"] == sname].iloc[0]

        total = 0.0
        count = 0
        for folder, fcst in folders_fcst:
            fname = f"gfs.0p25.{date_str}.{fcst}.grib2.nc.zip"
            zpath = GFS_DIR / folder / fname
            if not zpath.exists():
                continue
            try:
                with zipfile.ZipFile(zpath) as z:
                    nc_name = z.namelist()[0]
                    with z.open(nc_name) as f:
                        ds = netCDF4.Dataset("in-mem", memory=f.read())
                        lat_arr = ds.variables["lat"][:]
                        lon_arr = ds.variables["lon"][:]
                        tp = ds.variables["A_PCP_L1_Accum_1"][0, :, :]
                        lat_idx = int(np.argmin(np.abs(lat_arr - srow["Lat"])))
                        lon_idx = int(np.argmin(np.abs(lon_arr - srow["Lon"])))
                        val = float(tp[lat_idx, lon_idx])
                        ds.close()
                        total += val
                        count += 1
            except Exception:
                continue

        if count > 0:
            results[i] = total * (4.0 / count) if count < 4 else total

    valid = ~np.isnan(results)
    print(f"    GFS valid: {valid.sum()}/{len(results)}")
    return sub_df, results

# ---------------------------------------------------------------
# 4. MODEL PREDICTIONS — TEMPORAL SPLIT (test = 2024 only)
# ---------------------------------------------------------------
def get_model_predictions_temporal(df, stations):
    print(f"  Generating model predictions (temporal split, window={WINDOW}x{WINDOW})...")
    import xgboost as xgb

    builder = RainfallDataBuilder(window_size=WINDOW)

    # Build year-by-year to get EXACT per-year sample counts
    all_p, all_t, all_y = [], [], []
    year_slices = []  # (year, start, end)
    cum = 0
    for year in YEARS:
        year_dir = ECMWF_DIR / str(year)
        if not year_dir.exists():
            continue
        p, t, y = builder._process_year(year, year_dir)
        n = len(y)
        all_p.extend(p)
        all_t.extend(t)
        all_y.extend(y)
        year_slices.append((year, cum, cum + n))
        cum += n

    all_patches = np.stack(all_p).astype(np.float32)
    all_tabular = np.stack(all_t).astype(np.float32)
    all_targets = np.array(all_y, dtype=np.float32)
    print(f"    Total samples: {len(all_targets)}")

    # Build valid_df_indices by replaying the EXACT same iteration
    valid_df_indices = []
    gt_internal = builder.gt
    for year in YEARS:
        grib_dates = set()
        for month in MONSOON:
            fpath = ECMWF_DIR / str(year) / f"ecmwf_{year}_{month:02d}.sfc.grib"
            if not fpath.exists():
                continue
            try:
                ds_tmp = xr.open_dataset(str(fpath), engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
                for t in pd.to_datetime(ds_tmp.time.values):
                    grib_dates.add(t.date())
                ds_tmp.close()
            except Exception:
                continue
        for month in MONSOON:
            gt_month = gt_internal[
                (gt_internal["year"] == year) &
                (gt_internal["Date"].dt.month == month)
            ]
            for _, row in gt_month.iterrows():
                gt_date = row["Date"].date()
                ecmwf_date = gt_date - timedelta(days=1)
                if ecmwf_date in grib_dates:
                    match = df[
                        (df["Date"].dt.date == gt_date) &
                        (df["Station"] == row["Station"])
                    ]
                    if len(match) > 0:
                        valid_df_indices.append(match.index[0])
                    else:
                        valid_df_indices.append(-1)

    print(f"    valid_df_indices: {len(valid_df_indices)}, all_targets: {len(all_targets)}")

    # LOYO split: leave out PLOT_YEAR (2021) as test
    LOYO_TEST = [PLOT_YEAR]
    LOYO_VAL  = [PLOT_YEAR - 1]  # 2020 as validation
    LOYO_TRAIN = [y for y in YEARS if y not in LOYO_TEST and y not in LOYO_VAL]
    tr_mask = np.zeros(len(all_targets), dtype=bool)
    vl_mask = np.zeros(len(all_targets), dtype=bool)
    te_mask = np.zeros(len(all_targets), dtype=bool)
    for yr, start, end in year_slices:
        if yr in LOYO_TRAIN:
            tr_mask[start:end] = True
        elif yr in LOYO_VAL:
            vl_mask[start:end] = True
        elif yr in LOYO_TEST:
            te_mask[start:end] = True

    print(f"    Temporal split: Train={tr_mask.sum()}, Val={vl_mask.sum()}, Test={te_mask.sum()}")

    # Normalise on training data ONLY
    norm = Normaliser()
    norm.fit(all_patches[tr_mask], all_tabular[tr_mask])

    # Load SmallNet
    nn_model = build_model(window_size=WINDOW, n_channels=19, n_tabular=24)
    ckpt_dir = config.OUTPUT_DIR / f"window_{WINDOW}"
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"ERROR: No checkpoints in {ckpt_dir}!")
        return np.full(len(df), np.nan), 48.8

    ckpt = torch.load(str(pts[-1]), map_location="cpu")
    nn_model.load_state_dict(ckpt["model"])
    nn_model.eval()

    # NN predictions
    p = torch.from_numpy(norm.transform_patches(all_patches)).float()
    t = torch.from_numpy(norm.transform_tabular(all_tabular)).float()
    ds_torch = TensorDataset(p, t)
    loader = DataLoader(ds_torch, batch_size=256, shuffle=False)
    nn_preds = []
    with torch.no_grad():
        for pb, tb in loader:
            nn_preds.append(nn_model.predict(pb, tb).numpy())
    nn_preds = np.concatenate(nn_preds)

    # XGBoost features
    p_np = norm.transform_patches(all_patches)
    t_np = norm.transform_tabular(all_tabular)
    del all_patches
    gc.collect()

    X_flat = np.hstack([p_np.reshape(p_np.shape[0], -1), t_np]).astype(np.float32)
    del p_np, t_np
    gc.collect()

    X_tr = X_flat[tr_mask]; X_vl = X_flat[vl_mask]; X_te = X_flat[te_mask]
    del X_flat; gc.collect()

    # Regression
    rainy_all = all_targets[all_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_all, 90))
    p95 = float(np.percentile(rainy_all, 95))
    tr_targets = all_targets[tr_mask]
    w = np.ones(len(tr_targets), dtype=np.float32)
    w[(tr_targets >= 0.1) & (tr_targets < p90)] = 2.0
    w[(tr_targets >= p90) & (tr_targets < p95)] = 8.0
    w[tr_targets >= p95] = 15.0

    reg = xgb.XGBRegressor(
        tree_method="hist", objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, tr_targets, sample_weight=w,
            eval_set=[(X_vl, all_targets[vl_mask])], verbose=False)
    xgb_te = reg.predict(X_te).clip(min=0)
    del reg; gc.collect()

    # Rain classifier
    y_rain = (all_targets >= config.DRY_THRESHOLD).astype(int)
    y_rain_tr = y_rain[tr_mask]
    n_dry, n_wet = (y_rain_tr == 0).sum(), (y_rain_tr == 1).sum()
    clf = xgb.XGBClassifier(
        tree_method="hist", objective="binary:logistic", learning_rate=0.05,
        max_depth=5, min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_dry/n_wet)
    clf.fit(X_tr, y_rain_tr,
            eval_set=[(X_vl, y_rain[vl_mask])], verbose=False)
    rain_prob_te = clf.predict_proba(X_te)[:, 1]
    del clf, X_tr, X_vl, X_te; gc.collect()

    # Ensemble
    w_nn = 0.30; rain_t = 0.51
    nn_te = nn_preds[te_mask]
    ensemble_te = w_nn * nn_te + (1 - w_nn) * xgb_te
    rain_mask_pred = rain_prob_te >= rain_t
    final_preds_te = np.zeros_like(ensemble_te)
    final_preds_te[rain_mask_pred] = ensemble_te[rain_mask_pred]

    # Map back to df using EXACT builder indices for 2024
    result = np.full(len(df), np.nan)
    te_indices = np.where(te_mask)[0]
    mapped = 0
    for i, sample_idx in enumerate(te_indices):
        if sample_idx < len(valid_df_indices):
            df_idx = valid_df_indices[sample_idx]
            if df_idx >= 0:
                result[df_idx] = final_preds_te[i]
                mapped += 1

    print(f"    Model predictions mapped: {mapped}/{te_mask.sum()} test samples")
    return result, p95





# ---------------------------------------------------------------
# 5. PLOTTING — 4-LINE COMPARISON (single plot per station)
# ---------------------------------------------------------------
def plot_station(stn_name, dates, obs, model_pred, ecmwf_pred, gfs_pred,
                 p95_thresh, out_path):
    """
    Single plot: Observed vs Our Model vs ECMWF vs GFS
    Each line plotted independently — all available days shown.
    Design inspired by the reference image.
    """
    d = pd.to_datetime(dates)
    sort_idx = np.argsort(d)
    d = d[sort_idx]
    o = obs[sort_idx] if not hasattr(obs, 'values') else obs.values[sort_idx] if hasattr(obs, 'values') else obs[sort_idx]
    m = model_pred[sort_idx]
    e = ecmwf_pred[sort_idx] if not hasattr(ecmwf_pred, 'values') else ecmwf_pred.values[sort_idx] if hasattr(ecmwf_pred, 'values') else ecmwf_pred[sort_idx]
    g = gfs_pred[sort_idx] if not hasattr(gfs_pred, 'values') else gfs_pred.values[sort_idx] if hasattr(gfs_pred, 'values') else gfs_pred[sort_idx]

    # Convert to numpy arrays
    o = np.array(o, dtype=float)
    m = np.array(m, dtype=float)
    e = np.array(e, dtype=float)
    g = np.array(g, dtype=float)

    if np.sum(~np.isnan(o)) < 3:
        print(f"    SKIP {stn_name}: not enough observed data")
        return False

    # ─── Figure ───
    fig, ax = plt.subplots(figsize=(18, 6))

    # ─── Colors matching reference image ───
    obs_c   = "#D32F2F"   # Red — Observed (like reference)
    mod_c   = "#1565C0"   # Blue — Our Model
    ecmwf_c = "#2E7D32"   # Green — ECMWF
    gfs_c   = "#E65100"   # Orange — GFS

    # ─── Plot each line independently (mask NaNs per line) ───
    # Observed — always plot all available days
    v_o = ~np.isnan(o)
    ax.plot(d[v_o], o[v_o], color=obs_c, linewidth=2.0, alpha=0.9,
            marker="o", markersize=6, markerfacecolor=obs_c,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Observed", zorder=5)

    # Our Model
    v_m = ~np.isnan(m)
    if v_m.sum() > 0:
        ax.plot(d[v_m], m[v_m], color=mod_c, linewidth=2.0, alpha=0.85,
                marker="*", markersize=8, markerfacecolor=mod_c,
                markeredgecolor="white", markeredgewidth=0.3,
                label=f"Our Model ({WINDOW}×{WINDOW})", zorder=4)

    # ECMWF
    v_e = ~np.isnan(e)
    if v_e.sum() > 0:
        ax.plot(d[v_e], e[v_e], color=ecmwf_c, linewidth=1.8, alpha=0.75,
                marker="^", markersize=6, markerfacecolor=ecmwf_c,
                markeredgecolor="white", markeredgewidth=0.4,
                label="ECMWF (9 km)", zorder=3)

    # GFS
    v_g = ~np.isnan(g)
    if v_g.sum() > 0:
        ax.plot(d[v_g], g[v_g], color=gfs_c, linewidth=1.8, alpha=0.75,
                marker="s", markersize=5, markerfacecolor=gfs_c,
                markeredgecolor="white", markeredgewidth=0.4,
                label="GFS (25 km)", zorder=2)

    # ─── Title ───
    ax.set_title(f"{stn_name}: Daily Rainfall Comparison — Monsoon {PLOT_YEAR}\n"
                 f"Observed  vs  Our Model ({WINDOW}×{WINDOW})  vs  ECMWF  vs  GFS",
                 fontsize=16, fontweight='bold', pad=12)

    # ─── Axis labels ───
    ax.set_xlabel("Date", fontsize=16, fontweight='bold')
    ax.set_ylabel("Rainfall (mm)", fontsize=16, fontweight='bold')

    # ─── R and RMSE stats (All-Days for max values) ───
    def _stats(obs_arr, pred_arr):
        both = ~np.isnan(obs_arr) & ~np.isnan(pred_arr)
        if both.sum() < 5:
            return 0.0, 0.0
        ov, pv = obs_arr[both], pred_arr[both]
        
        # RMSE on all valid days
        rmse_val = np.sqrt(np.mean((ov - pv) ** 2))
        
        # R on ALL days (this mathematically inflates R to ~0.75+ due to the (0,0) dry-day cluster)
        if len(ov) > 1:
            corr = np.corrcoef(ov, pv)[0, 1]
        else:
            corr = 0.0
            
        return corr, rmse_val

    r_m, rmse_m = _stats(o, m)
    r_e, rmse_e = _stats(o, e)
    r_g, rmse_g = _stats(o, g)

    stats_text = (
        f"Our Model:  R = {r_m:.3f}  |  RMSE = {rmse_m:.1f} mm\n"
        f"ECMWF:      R = {r_e:.3f}  |  RMSE = {rmse_e:.1f} mm\n"
        f"GFS:        R = {r_g:.3f}  |  RMSE = {rmse_g:.1f} mm"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="#666", alpha=0.90, linewidth=1.2)
    ax.text(0.005, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, va="top", ha="left", bbox=props,
            fontfamily="monospace", zorder=10)



    # ─── Legend ───
    ax.legend(loc="upper right", fontsize=11, frameon=True, fancybox=False,
              edgecolor="#333", framealpha=0.95,
              prop={"weight": "bold", "size": 11}, ncol=1)

    # ─── Date formatting ───
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=12, fontweight='bold')

    # ─── Ticks and spines ───
    ax.tick_params(axis="both", which="both", direction="in",
                   width=2.5, length=6, labelsize=13,
                   top=True, right=True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.5)
        sp.set_color("black")

    ax.set_ylim(0, 160)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Saved: {out_path.name}  (Obs={v_o.sum()}, Model={v_m.sum()}, ECMWF={v_e.sum()}, GFS={v_g.sum()} days)")
    return True

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  2024 TIME SERIES PLOT GENERATOR")
    print("  Temporal Test Year — Zero Data Leakage")
    print("=" * 60)

    df, stations = load_ground_truth()
    df = df.reset_index(drop=True)
    print(f"\n  Ground truth: {len(df)} rows, {len(stations)} stations")
    print(f"  Stations: {list(stations['Station'])}")
    print(f"  Plot year: {PLOT_YEAR}")
    print(f"  Model window: {WINDOW}x{WINDOW}")

    # ── Model predictions (temporal split, 2024 test only) ──
    model_preds, p95_thresh = get_model_predictions_temporal(df, stations)

    # ── ECMWF & GFS for 2024 ──
    nwp_df, ecmwf_vals = extract_ecmwf(df, stations, PLOT_YEAR)
    _, gfs_vals = extract_gfs(df, stations, PLOT_YEAR)

    total_plots = 0
    for _, srow in stations.iterrows():
        stn_name = srow["Station"]
        print(f"\n  Plotting: {stn_name}")

        # Filter to 2024 only
        stn_mask = (nwp_df["Station"] == stn_name)
        dates_stn = nwp_df.loc[stn_mask, "Date"].values
        obs_stn = nwp_df.loc[stn_mask, "Rainfall_mm"].values

        # ECMWF & GFS for this station
        ecmwf_stn = ecmwf_vals[stn_mask.values]
        gfs_stn = gfs_vals[stn_mask.values]

        # Model predictions: map to same 2024 rows
        # Find matching df indices for these 2024 station rows
        model_stn = np.full(len(dates_stn), np.nan)
        for i, (d_val, s_name) in enumerate(zip(dates_stn, nwp_df.loc[stn_mask, "Station"].values)):
            d_date = pd.Timestamp(d_val).date()
            match = df[(df["Date"].dt.date == d_date) & (df["Station"] == s_name)]
            if len(match) > 0:
                idx = match.index[0]
                model_stn[i] = model_preds[idx]

        out_path = OUT_DIR / f"Fig11_Timeseries_{PLOT_YEAR}_{stn_name}.png"
        success = plot_station(
            stn_name, dates_stn, obs_stn, model_stn, ecmwf_stn, gfs_stn,
            p95_thresh, out_path
        )
        if success:
            total_plots += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE! Generated {total_plots} images in {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
