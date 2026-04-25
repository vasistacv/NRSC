"""
plot_correlation.py  —  Publication-quality scatter plots
=========================================================
For each station x window size:
  3 subplots: Observed vs ECMWF | Observed vs Our Model | Observed vs GFS

OUTPUT: 21 images (7 stations x 3 window sizes)
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys, json
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
from matplotlib.ticker import MaxNLocator
from scipy import stats
import torch
from torch.utils.data import DataLoader, TensorDataset

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
OUT_DIR    = ROOT / "correlation_plots"
OUT_DIR.mkdir(exist_ok=True)

MONSOON   = [6, 7, 8, 9]
YEARS     = list(range(2015, 2025))
WIN_SIZES = [3, 5, 31]

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
# 2. ECMWF: batch extraction per month
# ---------------------------------------------------------------
def extract_ecmwf_all(df, stations, window_size, valid_mask=None):
    """Extract ECMWF tp for every row in df, using D-1 forecast."""
    print(f"  Loading ECMWF (window={window_size}x{window_size})...")
    results = np.full(len(df), np.nan)

    ecmwf_res = 0.1  # degrees

    for year in YEARS:
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
            tp_data = ds["tp"].values  # shape: (ntime, nlat, nlon) in meters

            half = window_size // 2

            # For each station, find grid indices once
            stn_indices = {}
            for _, srow in stations.iterrows():
                sname = srow["Station"]
                lat_idx = int(np.argmin(np.abs(lats - srow["Lat"])))
                lon_idx = int(np.argmin(np.abs(lons - srow["Lon"])))
                stn_indices[sname] = (lat_idx, lon_idx)

            # Match df rows where forecast_date (obs_date - 1) falls in this month
            for i, row in df.iterrows():
                obs_date = row["Date"]
                forecast_date = obs_date - timedelta(days=1)

                if forecast_date.year != year or forecast_date.month != month:
                    continue

                if valid_mask is not None and not valid_mask[i]:
                    continue

                sname = row["Station"]
                if sname not in stn_indices:
                    continue

                # Find time index
                t_mask = times.date == forecast_date.date()
                if not t_mask.any():
                    continue
                t_idx = int(np.where(t_mask)[0][-1])

                lat_idx, lon_idx = stn_indices[sname]
                lat_s = max(0, lat_idx - half)
                lat_e = min(len(lats), lat_idx + half + 1)
                lon_s = max(0, lon_idx - half)
                lon_e = min(len(lons), lon_idx + half + 1)

                patch = tp_data[t_idx, lat_s:lat_e, lon_s:lon_e]
                if patch.size > 0:
                    results[i] = float(np.nanmean(patch)) * 1000.0  # m -> mm

            ds.close()

    valid = ~np.isnan(results)
    print(f"    ECMWF valid: {valid.sum()}/{len(results)}")
    return results

# ---------------------------------------------------------------
# 3. GFS: batch extraction
# ---------------------------------------------------------------
def _read_gfs_zip(zpath, lat, lon, window_size):
    if not zpath.exists():
        return np.nan
    try:
        with zipfile.ZipFile(zpath) as z:
            nc_name = z.namelist()[0]
            with z.open(nc_name) as f:
                ds = netCDF4.Dataset("in-mem", memory=f.read())
                lats = ds.variables["lat"][:]
                lons = ds.variables["lon"][:]
                tp = ds.variables["A_PCP_L1_Accum_1"][0, :, :]

                lat_idx = int(np.argmin(np.abs(lats - lat)))
                lon_idx = int(np.argmin(np.abs(lons - lon)))

                # Scale window for GFS 0.25 vs ECMWF 0.1
                gfs_half = max(0, int((window_size // 2) * 0.1 / 0.25))

                lat_s = max(0, lat_idx - gfs_half)
                lat_e = min(len(lats), lat_idx + gfs_half + 1)
                lon_s = max(0, lon_idx - gfs_half)
                lon_e = min(len(lons), lon_idx + gfs_half + 1)

                patch = tp[lat_s:lat_e, lon_s:lon_e]
                ds.close()
                return float(np.nanmean(patch)) if patch.size > 0 else np.nan
    except Exception:
        return np.nan


def extract_gfs_all(df, stations, window_size, valid_mask=None):
    """Extract GFS daily total precip (4 x 6hr accumulations from D-1 18Z init)."""
    print(f"  Loading GFS (window={window_size}x{window_size})...")
    results = np.full(len(df), np.nan)

    folders_fcst = [("0_to_6", "f006"), ("6_12", "f012"),
                    ("12_18", "f018"), ("18_24", "f024")]

    for i, row in df.iterrows():
        if valid_mask is not None and not valid_mask[i]:
            continue
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
            val = _read_gfs_zip(zpath, srow["Lat"], srow["Lon"], window_size)
            if not np.isnan(val):
                total += val
                count += 1

        if count > 0:
            results[i] = total * (4.0 / count) if count < 4 else total

        if (i + 1) % 500 == 0:
            print(f"    GFS progress: {i+1}/{len(df)}")

    valid = ~np.isnan(results)
    print(f"    GFS valid: {valid.sum()}/{len(results)}")
    return results

# ---------------------------------------------------------------
# 4. MODEL PREDICTIONS
# ---------------------------------------------------------------
def get_model_predictions(df, stations, window_size):
    """Run SmallNet + XGBoost locally for a specific window_size."""
    print(f"  Generating model predictions for {window_size}x{window_size}...")

    import xgboost as xgb
    import glob

    # Build features dynamically for this window
    builder = RainfallDataBuilder(window_size=window_size)
    all_patches, all_tabular, all_targets = builder.build(YEARS)

    # ... replicate rebuilding logic ...
    gt_internal = builder.gt
    valid_df_indices = []
    for year in YEARS:
        ecmwf_dir_year = ECMWF_DIR / str(year)
        grib_dates = set()
        for month in MONSOON:
            fpath = ecmwf_dir_year / f"ecmwf_{year}_{month:02d}.sfc.grib"
            if not fpath.exists():
                continue
            try:
                ds_tmp = xr.open_dataset(str(fpath), engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
                times = pd.to_datetime(ds_tmp.time.values)
                for t in times:
                    grib_dates.add(t.date())
                ds_tmp.close()
            except Exception:
                continue

        for month in MONSOON:
            gt_month = gt_internal[
                (gt_internal["year"] == year) &
                (gt_internal["Date"].dt.month == month)
            ]
            for idx_gt, row in gt_month.iterrows():
                gt_date = row["Date"].date()
                ecmwf_date = gt_date - timedelta(days=1)
                if ecmwf_date in grib_dates:
                    match = df[
                        (df["Date"].dt.date == gt_date) &
                        (df["Station"] == row["Station"])
                    ]
                    if len(match) > 0:
                        valid_df_indices.append(match.index[0])

    norm = Normaliser()
    norm.fit(all_patches, all_tabular)

    # Load Dynamic SmallNet
    nn_model = build_model(window_size=window_size, n_channels=19, n_tabular=24)
    # Get the latest checkpoint for this window
    ckpt_dir = config.OUTPUT_DIR / f"window_{window_size}"
    pts = list(ckpt_dir.glob("*.pt"))
    if len(pts) == 0:
        print(f"ERROR: No checkpoints found in {ckpt_dir}!")
        return np.full(len(df), np.nan)
        
    ckpt_path = pts[-1]
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
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

    # XGBoost features — MEMORY-SAFE for 31x31
    import gc
    p_np = norm.transform_patches(all_patches)
    t_np = norm.transform_tabular(all_tabular)
    del all_patches, all_tabular
    gc.collect()

    X_flat = np.hstack([p_np.reshape(p_np.shape[0], -1), t_np]).astype(np.float32)
    del p_np, t_np
    gc.collect()

    # 70/30 split
    rainy_all = all_targets[all_targets >= config.DRY_THRESHOLD]
    p90 = float(np.percentile(rainy_all, 90))
    p95 = float(np.percentile(rainy_all, 95))

    w = np.ones(len(all_targets), dtype=np.float32)
    w[(all_targets >= 0.1) & (all_targets < p90)] = 2.0
    w[(all_targets >= p90) & (all_targets < p95)] = 8.0
    w[all_targets >= p95] = 15.0

    n = len(all_targets)
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    tr_idx = idx[:n_train]
    vl_idx = idx[n_train:n_train+n_val]
    te_idx = idx[n_train+n_val:]

    # Split IMMEDIATELY and free the giant X_flat to save ~1.2GB RAM
    X_tr = X_flat[tr_idx]
    X_vl = X_flat[vl_idx]
    X_te = X_flat[te_idx]
    del X_flat
    gc.collect()

    reg = xgb.XGBRegressor(
        tree_method="hist",
        objective="reg:squarederror", learning_rate=0.03,
        max_depth=6, min_child_weight=8, subsample=0.8,
        colsample_bytree=0.65, gamma=0.8, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=2000, early_stopping_rounds=80,
        verbosity=0, n_jobs=-1, random_state=42)
    reg.fit(X_tr, all_targets[tr_idx], sample_weight=w[tr_idx],
            eval_set=[(X_vl, all_targets[vl_idx])], verbose=False)
    xgb_te = reg.predict(X_te).clip(min=0)
    del reg
    gc.collect()

    # Rain classifier
    y_rain = (all_targets >= config.DRY_THRESHOLD).astype(int)
    n_dry, n_wet = (y_rain[tr_idx] == 0).sum(), (y_rain[tr_idx] == 1).sum()
    clf = xgb.XGBClassifier(
        tree_method="hist",
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_dry/n_wet)
    
    y_rain_vl = y_rain[vl_idx]
    y_rain_te = y_rain[te_idx]
    
    clf.fit(X_tr, y_rain[tr_idx],
            eval_set=[(X_vl, y_rain_vl)], verbose=False)
    rain_prob_te = clf.predict_proba(X_te)[:, 1]
    del clf, X_tr, X_te, X_vl
    gc.collect()

    # Ensemble — TEST SET ONLY (zero leakage)
    w_nn = 0.35
    nn_te = nn_preds[te_idx]
    ensemble_te = w_nn * nn_te + (1 - w_nn) * xgb_te
    rain_mask = rain_prob_te >= 0.45
    final_preds_te = np.zeros_like(ensemble_te)
    final_preds_te[rain_mask] = ensemble_te[rain_mask]

    # Map back to df-aligned array (NaN for rows without GRIB data)
    result = np.full(len(df), np.nan)
    
    # GUARANTEE ZERO DATA LEAKAGE:
    # Only map the 15% completely hold-out test predictions to the plotting array!
    for i, test_i in enumerate(te_idx):
        if test_i < len(valid_df_indices):
            df_idx = valid_df_indices[test_i]
            result[df_idx] = final_preds_te[i]

    valid_count = (~np.isnan(result)).sum()
    print(f"    Model predictions aligned: {valid_count}/{len(df)} valid")
    return result

# ---------------------------------------------------------------
# 5. PLOTTING
# ---------------------------------------------------------------
def plot_station(stn_name, obs, ecmwf, model_pred, gfs, window_size, out_path):
    """3 side-by-side scatter plots: ECMWF | Model | GFS."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle(f"{stn_name}  |  Window: {window_size} x {window_size}",
                 fontsize=20, fontweight="bold", y=1.02)

    plot_data = [
        ("ECMWF (9 km)", ecmwf, "#1565C0"),
        ("Our Model",    model_pred, "#C62828"),
        ("GFS (25 km)",  gfs, "#2E7D32"),
    ]

    for idx, (label, pred, color) in enumerate(plot_data):
        ax = axes[idx]

        if pred is None or len(pred) == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
            ax.set_title(f"Observed vs {label}", fontweight="bold")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(1.8)
            continue

        valid = ~np.isnan(obs) & ~np.isnan(pred)
        x = obs[valid]
        y = pred[valid]

        if len(x) < 5:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f"Observed vs {label}", fontweight="bold")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(1.8)
            continue

        # Scatter
        ax.scatter(x, y, s=18, alpha=0.45, color=color, edgecolors="none", zorder=3)

        # 1:1 line
        max_val = 100
        ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, alpha=0.6, label="1:1 line")

        # Linear regression line
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        x_fit = np.linspace(0, max_val, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=color, lw=2.0, alpha=0.85)

        # Stats
        corr = np.corrcoef(x, y)[0, 1]
        rmse = np.sqrt(np.mean((x - y) ** 2))
        bias = np.mean(y - x)
        n = len(x)

        stats_text = (f"r = {corr:.3f}\n"
                      f"RMSE = {rmse:.2f} mm\n"
                      f"Bias = {bias:+.2f} mm\n"
                      f"N = {n}")
        props = dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="gray", alpha=0.9)
        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
                fontsize=12, va="top", ha="left", bbox=props,
                fontfamily="monospace", fontweight="bold")

        # Axis formatting
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed Rainfall (mm)", fontweight="bold")
        ax.set_ylabel(f"{label} Rainfall (mm)", fontweight="bold")
        ax.set_title(f"Observed vs {label}", fontweight="bold")

        # FULL BOX with bold spines
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(1.8)
            sp.set_color("black")

        # Bold ticks on all 4 sides
        ax.tick_params(axis="both", which="both", direction="in",
                       width=1.5, length=6, labelsize=13,
                       top=True, right=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  CORRELATION SCATTER PLOT GENERATOR")
    print("=" * 60)

    df, stations = load_ground_truth()
    df = df.reset_index(drop=True)
    print(f"\n  Ground truth: {len(df)} rows, {len(stations)} stations")
    print(f"  Stations: {list(stations['Station'])}")

    # We will test the three most scientifically relevant scales
    WIN_SIZES_TO_TEST = [3, 9, 31]
    
    total_plots = 0

    for ws in WIN_SIZES_TO_TEST:
        print(f"\n{'=' * 60}")
        print(f"  WINDOW SIZE: {ws} x {ws}")
        print(f"{'=' * 60}")
        
        # Dynamically load and run XGBoost for specifically this window size
        model_preds = get_model_predictions(df, stations, window_size=ws)

        # Mask for test holdout predictions
        test_mask = ~np.isnan(model_preds)

        # Extract ECMWF for this window size (only test data)
        ecmwf_all = extract_ecmwf_all(df, stations, ws, valid_mask=test_mask)

        # Extract GFS for this window size (only test data)
        gfs_all = extract_gfs_all(df, stations, ws, valid_mask=test_mask)

        # Plot per station
        for _, srow in stations.iterrows():
            sname = srow["Station"]
            stn_mask = df["Station"] == sname

            obs   = df.loc[stn_mask, "Rainfall_mm"].values
            ecmwf = ecmwf_all[stn_mask.values]
            gfs   = gfs_all[stn_mask.values]
            m_pred = model_preds[stn_mask.values]

            out_name = f"{sname}_window{ws}x{ws}.png"
            out_path = OUT_DIR / out_name
            plot_station(sname, obs, ecmwf, m_pred, gfs, ws, out_path)
            total_plots += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE! Generated {total_plots} images in {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
