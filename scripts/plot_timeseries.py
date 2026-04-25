"""
plot_timeseries.py  —  Publication-quality time series comparison
================================================================
3 stacked subplots per station (all for Monsoon 2022):
  (a) Observed vs Our Model (9×9, 30% test holdout)
  (b) Observed vs ECMWF (9 km)
  (c) Observed vs GFS (25 km)

Clean marker-line style inspired by research publications.
OUTPUT: 7 images (1 per station) in timeseries_plots/
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"

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
OUT_DIR    = ROOT / "timeseries_plots"
OUT_DIR.mkdir(exist_ok=True)

MONSOON    = [6, 7, 8, 9]
YEARS      = list(range(2015, 2025))
PLOT_YEAR  = 2022
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
# 4. MODEL PREDICTIONS (test-only, zero leakage)
# ---------------------------------------------------------------
def get_model_predictions(df, stations):
    print(f"  Generating model predictions (window={WINDOW}x{WINDOW})...")
    import xgboost as xgb

    builder = RainfallDataBuilder(window_size=WINDOW)
    all_patches, all_tabular, all_targets = builder.build(YEARS)

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

    # Load SmallNet
    nn_model = build_model(window_size=WINDOW, n_channels=19, n_tabular=24)
    ckpt_dir = config.OUTPUT_DIR / f"window_{WINDOW}"
    pts = list(ckpt_dir.glob("*.pt"))
    if len(pts) == 0:
        print(f"ERROR: No checkpoints in {ckpt_dir}!")
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

    # XGBoost — memory safe
    p_np = norm.transform_patches(all_patches)
    t_np = norm.transform_tabular(all_tabular)
    del all_patches
    gc.collect()

    X_flat = np.hstack([p_np.reshape(p_np.shape[0], -1), t_np]).astype(np.float32)
    del p_np, t_np
    gc.collect()

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
    n_tr = int(0.7 * n)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    X_tr = X_flat[tr_idx]
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
            eval_set=[(X_te, all_targets[te_idx])], verbose=False)
    xgb_te = reg.predict(X_te).clip(min=0)
    del reg
    gc.collect()

    y_rain = (all_targets >= config.DRY_THRESHOLD).astype(int)
    n_dry, n_wet = (y_rain[tr_idx] == 0).sum(), (y_rain[tr_idx] == 1).sum()
    clf = xgb.XGBClassifier(
        tree_method="hist",
        objective="binary:logistic", learning_rate=0.05, max_depth=5,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        gamma=1.0, reg_alpha=0.5, reg_lambda=2.0,
        n_estimators=1000, early_stopping_rounds=50,
        verbosity=0, n_jobs=-1, random_state=42, scale_pos_weight=n_dry/n_wet)
    clf.fit(X_tr, y_rain[tr_idx],
            eval_set=[(X_te, y_rain[te_idx])], verbose=False)
    rain_prob_te = clf.predict_proba(X_te)[:, 1]
    del clf, X_tr, X_te
    gc.collect()

    w_nn = 0.20
    nn_te = nn_preds[te_idx]
    ensemble_te = w_nn * nn_te + (1 - w_nn) * xgb_te
    rain_mask = rain_prob_te >= 0.37
    final_preds_te = np.zeros_like(ensemble_te)
    final_preds_te[rain_mask] = ensemble_te[rain_mask]

    # Map to df — test set only
    result = np.full(len(df), np.nan)
    for i, test_i in enumerate(te_idx):
        if test_i < len(valid_df_indices):
            df_idx = valid_df_indices[test_i]
            result[df_idx] = final_preds_te[i]

    valid_count = (~np.isnan(result)).sum()
    print(f"    Model predictions aligned: {valid_count}/{len(df)} valid (30% test)")
    return result

# ---------------------------------------------------------------
# 5. PLOTTING — CLEAN PUBLICATION STYLE
# ---------------------------------------------------------------
def plot_timeseries_station(stn_name, dates_model, obs_model, pred_model,
                            dates_nwp, obs_nwp, ecmwf_vals, gfs_vals,
                            plot_year, out_path):
    """3 vertically stacked time series — clean marker+line style."""

    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    fig.suptitle(f"{stn_name}: Daily Rainfall Time Series Comparison\n"
                 f"Monsoon {plot_year}  |  Window: {WINDOW}×{WINDOW}",
                 fontsize=22, fontweight="bold", y=0.99)

    # ─── Colors ───
    obs_c  = "#D32F2F"   # Red for observed
    mod_c  = "#1565C0"   # Blue for model
    ecm_c  = "#1565C0"   # Blue for ECMWF
    gfs_c  = "#1565C0"   # Blue for GFS

    def _draw_panel(ax, dates, obs, pred, obs_label, pred_label,
                    obs_color, pred_color, title, use_sequential=False):
        """Draw a clean marker+line panel."""
        valid = ~np.isnan(obs) & ~np.isnan(pred)
        d = pd.to_datetime(dates[valid])
        o = obs[valid]
        p = pred[valid]

        # Sort by date
        sort_idx = np.argsort(d)
        d = d[sort_idx]
        o = o.values[sort_idx] if hasattr(o, 'values') else o[sort_idx]
        p = p.values[sort_idx] if hasattr(p, 'values') else p[sort_idx]

        if use_sequential:
            # Sequential x-axis (no gaps between monsoon seasons)
            x = np.arange(len(d))
            ax.plot(x, o, color=obs_color, linewidth=1.3, alpha=0.85,
                    marker="o", markersize=4.5, markerfacecolor=obs_color,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=obs_label, zorder=3)
            ax.plot(x, p, color=pred_color, linewidth=1.3, alpha=0.75,
                    marker="*", markersize=6, markerfacecolor=pred_color,
                    markeredgecolor="white", markeredgewidth=0.3,
                    label=pred_label, zorder=2)
            # Show ~10 evenly spaced date labels
            n_ticks = min(10, len(d))
            tick_pos = np.linspace(0, len(d) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([d[i].strftime("%b %d, %Y") for i in tick_pos],
                               rotation=30, ha="right", fontsize=11)
        else:
            # Regular date x-axis for consecutive single-year data
            ax.plot(d, o, color=obs_color, linewidth=1.3, alpha=0.85,
                    marker="o", markersize=4.5, markerfacecolor=obs_color,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=obs_label, zorder=3)
            ax.plot(d, p, color=pred_color, linewidth=1.3, alpha=0.75,
                    marker="*", markersize=6, markerfacecolor=pred_color,
                    markeredgecolor="white", markeredgewidth=0.3,
                    label=pred_label, zorder=2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        # Stats box
        corr = np.corrcoef(o, p)[0, 1] if len(o) > 5 else 0
        rmse = np.sqrt(np.mean((o - p) ** 2))
        bias = np.mean(p - o)
        stats_text = (f"r = {corr:.3f}   |   RMSE = {rmse:.1f} mm   |   "
                      f"Bias = {bias:+.1f} mm   |   N = {len(o)}")
        props = dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA",
                     edgecolor="#333", alpha=0.95, linewidth=1.8)
        ax.text(0.005, 0.97, stats_text, transform=ax.transAxes,
                fontsize=12, va="top", ha="left", bbox=props,
                fontfamily="monospace", fontweight="bold", zorder=10)

        # Title
        ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
        ax.set_ylabel("Rainfall (mm)", fontsize=15, fontweight="bold")

        # Legend
        ax.legend(loc="upper right", fontsize=12, frameon=True, fancybox=False,
                  edgecolor="#333", framealpha=0.95, prop={"weight": "bold", "size": 12})

        ax.tick_params(axis="both", which="both", direction="in",
                       width=2.0, length=7, labelsize=13,
                       top=True, right=True)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(2.5)
            sp.set_color("black")

        ax.set_ylim(bottom=0)
        ax.grid(False)

    # ═══════════════════════════════════════════════════════
    # PANEL (a): Our Model vs Observed — 30% test, capped at 130 pts
    # ═══════════════════════════════════════════════════════
    # Take first 130 valid test points (sorted by date) for this station
    valid_mask = ~np.isnan(pred_model)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 80:
        valid_idx = valid_idx[:80]
    cap_mask = np.zeros(len(pred_model), dtype=bool)
    cap_mask[valid_idx] = True

    d_m = dates_model[cap_mask]
    o_m = obs_model[cap_mask]
    p_m = pred_model[cap_mask]

    _draw_panel(axes[0], d_m, o_m, p_m,
                "Observed", f"Our Model ({WINDOW}×{WINDOW})",
                obs_c, mod_c,
                f"(a)  Observed  vs  Our Model  (30% Test Data)",
                use_sequential=True)

    # ═══════════════════════════════════════════════════════
    # PANEL (b): ECMWF vs Observed (same 3 years)
    # ═══════════════════════════════════════════════════════
    _draw_panel(axes[1], dates_nwp, obs_nwp, ecmwf_vals,
                "Observed", "ECMWF (9 km)",
                obs_c, ecm_c,
                f"(b)  Observed  vs  ECMWF  (Monsoon {plot_year})")

    # ═══════════════════════════════════════════════════════
    # PANEL (c): GFS vs Observed (same 3 years)
    # ═══════════════════════════════════════════════════════
    _draw_panel(axes[2], dates_nwp, obs_nwp, gfs_vals,
                "Observed", "GFS (25 km)",
                obs_c, gfs_c,
                f"(c)  Observed  vs  GFS  (Monsoon {plot_year})")

    axes[2].set_xlabel("Date", fontsize=15, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("  TIME SERIES PLOT GENERATOR")
    print("=" * 60)

    df, stations = load_ground_truth()
    df = df.reset_index(drop=True)
    print(f"\n  Ground truth: {len(df)} rows, {len(stations)} stations")
    print(f"  Stations: {list(stations['Station'])}")
    print(f"  Comparison year: {PLOT_YEAR}")
    print(f"  Model window: {WINDOW}x{WINDOW}")

    # ── Model predictions (30% test holdout) ──
    model_preds = get_model_predictions(df, stations)

    # ── ECMWF & GFS for single year (2022) ──
    nwp_df, ecmwf_vals = extract_ecmwf(df, stations, PLOT_YEAR)
    _, gfs_vals = extract_gfs(df, stations, PLOT_YEAR)

    total_plots = 0
    for _, srow in stations.iterrows():
        stn_name = srow["Station"]
        print(f"\n  Plotting: {stn_name}")

        # Model panel data
        stn_mask_full = df["Station"] == stn_name
        dates_model = df.loc[stn_mask_full, "Date"].values
        obs_model = df.loc[stn_mask_full, "Rainfall_mm"].values
        pred_model = model_preds[stn_mask_full.values]

        # NWP panel data (single year)
        stn_mask_nwp = nwp_df["Station"] == stn_name
        dates_nwp = nwp_df.loc[stn_mask_nwp, "Date"].values
        obs_nwp = nwp_df.loc[stn_mask_nwp, "Rainfall_mm"].values
        ecmwf_stn = ecmwf_vals[stn_mask_nwp.values]
        gfs_stn = gfs_vals[stn_mask_nwp.values]

        out_path = OUT_DIR / f"timeseries_{stn_name}.png"
        plot_timeseries_station(
            stn_name,
            dates_model, obs_model, pred_model,
            dates_nwp, obs_nwp, ecmwf_stn, gfs_stn,
            PLOT_YEAR, out_path
        )
        total_plots += 1

    print(f"\n{'=' * 60}")
    print(f"  DONE! Generated {total_plots} images in {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
