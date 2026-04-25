"""
compare_all_metrics_2024.py
===========================
Compute CSI, POD, FAR, SEDI, R, RMSE for Our Model vs ECMWF vs GFS
on 2024 monsoon data (the temporal test year).
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
os.environ["PATH"] = r"D:\NEW_NRSC\.venv\Library\bin;" + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
import xarray as xr
import zipfile, netCDF4
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "extracted_files"))

ROOT      = Path(r"D:\NEW_NRSC")
GT_PATH   = ROOT / "Final_ground_truth_data.csv"
ECMWF_DIR = ROOT / "ecmwf_data"
GFS_DIR   = ROOT / "Final_GFS_Data" / "Total_ppt"
MONSOON   = [6, 7, 8, 9]

# ── Load ground truth for 2024 ──
df = pd.read_csv(GT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"].dt.month.isin(MONSOON)]
df_2024 = df[df["Date"].dt.year == 2024].reset_index(drop=True)
stations = df.groupby("Station").agg({"Lat": "first", "Lon": "first"}).reset_index()

print(f"2024 monsoon rows: {len(df_2024)}")
print(f"Stations: {list(stations['Station'])}")

# ── Extract ECMWF ──
print("\nExtracting ECMWF...")
ecmwf_vals = np.full(len(df_2024), np.nan)
for month in MONSOON:
    fpath = ECMWF_DIR / "2024" / f"ecmwf_2024_{month:02d}.sfc.grib"
    if not fpath.exists(): continue
    try:
        ds = xr.open_dataset(str(fpath), engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
    except: continue
    lats, lons = ds.latitude.values, ds.longitude.values
    times = pd.to_datetime(ds.time.values)
    tp_data = ds["tp"].values
    stn_idx = {}
    for _, sr in stations.iterrows():
        stn_idx[sr["Station"]] = (int(np.argmin(np.abs(lats - sr["Lat"]))),
                                   int(np.argmin(np.abs(lons - sr["Lon"]))))
    for i, row in df_2024.iterrows():
        fd = row["Date"] - timedelta(days=1)
        if fd.month != month: continue
        sn = row["Station"]
        if sn not in stn_idx: continue
        tm = times.date == fd.date()
        if not tm.any(): continue
        ti = int(np.where(tm)[0][-1])
        li, lo = stn_idx[sn]
        ecmwf_vals[i] = float(tp_data[ti, li, lo]) * 1000.0
    ds.close()
print(f"  ECMWF valid: {(~np.isnan(ecmwf_vals)).sum()}/{len(df_2024)}")

# ── Extract GFS ──
print("Extracting GFS...")
gfs_vals = np.full(len(df_2024), np.nan)
folders_fcst = [("0_to_6", "f006"), ("6_12", "f012"),
                ("12_18", "f018"), ("18_24", "f024")]
for i, row in df_2024.iterrows():
    prev = row["Date"] - timedelta(days=1)
    ds_str = prev.strftime("%Y%m%d") + "18"
    sr = stations[stations["Station"] == row["Station"]].iloc[0]
    total, count = 0.0, 0
    for folder, fcst in folders_fcst:
        zp = GFS_DIR / folder / f"gfs.0p25.{ds_str}.{fcst}.grib2.nc.zip"
        if not zp.exists(): continue
        try:
            with zipfile.ZipFile(zp) as z:
                with z.open(z.namelist()[0]) as f:
                    nc = netCDF4.Dataset("in-mem", memory=f.read())
                    la = nc.variables["lat"][:]
                    lo = nc.variables["lon"][:]
                    tp = nc.variables["A_PCP_L1_Accum_1"][0,:,:]
                    li = int(np.argmin(np.abs(la - sr["Lat"])))
                    loi = int(np.argmin(np.abs(lo - sr["Lon"])))
                    total += float(tp[li, loi])
                    count += 1
                    nc.close()
        except: continue
    if count > 0:
        gfs_vals[i] = total * (4.0/count) if count < 4 else total
print(f"  GFS valid: {(~np.isnan(gfs_vals)).sum()}/{len(df_2024)}")

# ── Observed ──
obs = df_2024["Rainfall_mm"].values.astype(float)

# ── Thresholds (from training data, same as model) ──
df_train = df[df["Date"].dt.year.isin(range(2015, 2024))]
rainy_train = df_train["Rainfall_mm"].values
rainy_train = rainy_train[rainy_train >= 0.1]
p90 = float(np.percentile(rainy_train, 90))
p95 = float(np.percentile(rainy_train, 95))
print(f"\nThresholds: P90={p90:.1f} mm, P95={p95:.1f} mm")

# ── Metrics computation ──
def compute_metrics(pred, obs_arr, name):
    # Only use common valid days
    valid = ~np.isnan(pred) & ~np.isnan(obs_arr)
    p = pred[valid]
    o = obs_arr[valid]
    n = len(o)
    
    # Continuous
    r_all = float(np.corrcoef(o, p)[0, 1])
    rainy_mask = o >= 0.1
    r_rainy = float(np.corrcoef(o[rainy_mask], p[rainy_mask])[0, 1]) if rainy_mask.sum() > 5 else 0
    rmse_val = float(np.sqrt(np.mean((o - p) ** 2)))
    mae_val = float(np.mean(np.abs(o - p)))
    
    results = {"name": name, "N": n, "R_all": r_all, "R_rainy": r_rainy,
               "RMSE": rmse_val, "MAE": mae_val}
    
    # Categorical at each threshold
    for label, thresh in [("rain", 0.1), ("P90", p90), ("P95", p95)]:
        pe = p >= thresh
        oe = o >= thresh
        H = int((pe & oe).sum())
        M = int((~pe & oe).sum())
        FA = int((pe & ~oe).sum())
        CN = int((~pe & ~oe).sum())
        
        csi = H / (H + M + FA) if (H + M + FA) > 0 else 0
        pod = H / (H + M) if (H + M) > 0 else 0
        far_val = FA / (H + FA) if (H + FA) > 0 else 0
        fbi = (H + FA) / (H + M) if (H + M) > 0 else 0
        
        eps = 1e-7
        hr = np.clip(H / (H + M + eps), eps, 1 - eps)
        fr = np.clip(FA / (FA + CN + eps), eps, 1 - eps)
        num = np.log(fr) - np.log(hr) - np.log(1 - fr) + np.log(1 - hr)
        den = np.log(fr) + np.log(hr) + np.log(1 - fr) + np.log(1 - hr)
        sedi_val = float(num / den) if abs(den) > eps else 0
        
        hss_num = 2.0 * (H * CN - M * FA)
        hss_den = (H + M) * (M + CN) + (H + FA) * (FA + CN)
        hss_val = hss_num / hss_den if hss_den > 0 else 0
        
        results[f"CSI_{label}"] = round(csi, 4)
        results[f"POD_{label}"] = round(pod, 4)
        results[f"FAR_{label}"] = round(far_val, 4)
        results[f"FBI_{label}"] = round(fbi, 4)
        results[f"HSS_{label}"] = round(hss_val, 4)
        results[f"SEDI_{label}"] = round(sedi_val, 4)
        results[f"H_{label}"] = H
        results[f"M_{label}"] = M
        results[f"FA_{label}"] = FA
        results[f"n_obs_{label}"] = H + M
    
    return results

ecmwf_metrics = compute_metrics(ecmwf_vals, obs, "ECMWF (9 km)")
gfs_metrics = compute_metrics(gfs_vals, obs, "GFS (25 km)")

# Load our model's metrics from JSON
with open(ROOT / "final_model_baseline" / "9x9" / "dual_eval_results.json") as f:
    model_json = json.load(f)
model_metrics = model_json["temporal_split"]["metrics"]
model_metrics["name"] = "Our Model (9×9)"
model_metrics["R_all"] = "N/A"
model_metrics["R_rainy"] = model_metrics["corr_rainy"]

# ── Print comparison ──
print("\n" + "=" * 80)
print("  FULL METRIC COMPARISON — 2024 MONSOON (TEMPORAL TEST YEAR)")
print("=" * 80)

all_m = [model_metrics, ecmwf_metrics, gfs_metrics]
names = ["Our Model (9×9)", "ECMWF (9 km)", "GFS (25 km)"]

# Continuous
print(f"\n{'CONTINUOUS':^80}")
print(f"{'Metric':<20} {'Our Model':>15} {'ECMWF':>15} {'GFS':>15}")
print("-" * 65)
for k in ["R_rainy", "RMSE", "MAE"]:
    vals = []
    for m in all_m:
        v = m.get(k, "N/A")
        vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
    print(f"  {k:<18} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

# Categorical
for cat in ["rain", "P90", "P95"]:
    cat_key = cat
    print(f"\n{'CATEGORICAL — ' + cat.upper():^80}")
    print(f"{'Metric':<20} {'Our Model':>15} {'ECMWF':>15} {'GFS':>15}")
    print("-" * 65)
    for metric in ["CSI", "POD", "FAR", "FBI", "HSS", "SEDI", "H", "M", "FA", "n_obs"]:
        key = f"{metric}_{cat_key}"
        vals = []
        for m in all_m:
            v = m.get(key, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"  {metric:<18} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

# Save to JSON
output = {
    "thresholds": {"P90": p90, "P95": p95},
    "our_model": model_metrics,
    "ecmwf": ecmwf_metrics,
    "gfs": gfs_metrics
}
out_path = ROOT / "timeseries_2024" / "full_comparison_2024.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {out_path}")
print("=" * 80)
