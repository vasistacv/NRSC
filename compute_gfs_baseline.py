"""
compute_gfs_baseline.py
========================
Compute GFS baseline metrics in the EXACT same format as ecmwf_baseline_results.json.
Reads GFS zip/nc files, extracts daily rainfall per station, computes CSI/POD/FAR/SEDI.
Saves to gfs_baseline_results.json
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import zipfile
import netCDF4
from pathlib import Path
from datetime import timedelta

ROOT      = Path(r"D:\NEW_NRSC")
GT_PATH   = ROOT / "Final_ground_truth_data.csv"
GFS_DIR   = ROOT / "Final_GFS_Data" / "Total_ppt"
OUT_PATH  = ROOT / "gfs_baseline_results.json"
MONSOON   = [6, 7, 8, 9]
DRY_THR   = 0.1

# ── Load ground truth ──
print("Loading ground truth...")
df = pd.read_csv(GT_PATH, parse_dates=["Date"])
df = df.dropna(subset=["Rainfall_mm"])
df = df[df["Date"].dt.month.isin(MONSOON)]
df["year"] = df["Date"].dt.year

stations = df.groupby("Station").agg({"Lat": "first", "Lon": "first"}).reset_index()
print(f"  {len(df)} rows, {len(stations)} stations")

# Compute thresholds from rainy days
rainy = df[df["Rainfall_mm"] >= DRY_THR]["Rainfall_mm"].values
P90 = float(np.percentile(rainy, 90))
P95 = float(np.percentile(rainy, 95))
print(f"  P90={P90:.2f} mm, P95={P95:.2f} mm")

# ── GFS extraction ──
folders_fcst = [("0_to_6", "f006"), ("6_12", "f012"),
                ("12_18", "f018"), ("18_24", "f024")]

def extract_gfs_for_row(row, srow):
    obs_date = row["Date"]
    prev_day = obs_date - timedelta(days=1)
    date_str = prev_day.strftime("%Y%m%d") + "18"

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
        return total * (4.0 / count) if count < 4 else total
    return np.nan


def compute_csi(H, M, FA):
    denom = H + M + FA
    return H / denom if denom > 0 else 0.0

def compute_pod(H, M):
    return H / (H + M) if (H + M) > 0 else 0.0

def compute_far(H, FA):
    return FA / (H + FA) if (H + FA) > 0 else 0.0

def compute_fbi(H, M, FA):
    return (H + FA) / (H + M) if (H + M) > 0 else 0.0

def compute_hss(H, M, FA, N):
    expected = ((H + M) * (H + FA) + (N - H - M - FA) * (N - H - FA)) / N if N > 0 else 0
    return (H - expected) / (N - expected) if (N - expected) != 0 else 0.0

def compute_sedi(H, M, FA, N):
    pod = H / (H + M) if (H + M) > 0 else 0.0
    CN = N - H - M - FA
    pofd = FA / (FA + CN) if (FA + CN) > 0 else 0.0
    if pod <= 0 or pod >= 1 or pofd <= 0 or pofd >= 1:
        return 0.0
    import math
    num = (math.log(pofd) - math.log(pod) - math.log(1 - pofd) + math.log(1 - pod))
    den = (math.log(pofd) + math.log(pod) + math.log(1 - pofd) + math.log(1 - pod))
    return num / den if den != 0 else 0.0


def compute_metrics(obs, pred, p90, p95):
    """Compute all categorical metrics for rain, p90, p95."""
    obs = np.array(obs)
    pred = np.array(pred)

    # Remove NaN pairs
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs = obs[valid]
    pred = pred[valid]
    N = len(obs)

    results = {}

    for label, threshold in [("rain", DRY_THR), ("p90", p90), ("p95", p95)]:
        o_bin = obs >= threshold
        p_bin = pred >= threshold

        H  = int(np.sum(o_bin & p_bin))
        M  = int(np.sum(o_bin & ~p_bin))
        FA = int(np.sum(~o_bin & p_bin))

        results[f"CSI_{label}"]  = round(compute_csi(H, M, FA), 4)
        results[f"POD_{label}"]  = round(compute_pod(H, M), 4)
        results[f"FAR_{label}"]  = round(compute_far(H, FA), 4)
        results[f"FBI_{label}"]  = round(compute_fbi(H, M, FA), 4)
        results[f"HSS_{label}"]  = round(compute_hss(H, M, FA, N), 4)
        results[f"SEDI_{label}"] = round(compute_sedi(H, M, FA, N), 4)
        results[f"H_{label}"]    = H
        results[f"M_{label}"]    = M
        results[f"FA_{label}"]   = FA
        results[f"n_obs_{label}"] = int(o_bin.sum())

    # Correlation and RMSE on rainy days
    rainy_mask = obs >= DRY_THR
    if rainy_mask.sum() > 5:
        corr = float(np.corrcoef(obs[rainy_mask], pred[rainy_mask])[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    results["corr_rainy"] = round(corr, 4)
    results["RMSE"] = round(rmse, 4)
    results["n_samples"] = N

    return results


# ── Main loop: per station × per year ──
all_results = {}
years = list(range(2015, 2025))

for _, srow in stations.iterrows():
    stn = srow["Station"]
    print(f"\n{'='*50}")
    print(f"Station: {stn}")

    for year in years:
        sub = df[(df["Station"] == stn) & (df["year"] == year)].reset_index(drop=True)
        if len(sub) == 0:
            continue

        print(f"  Year {year}: {len(sub)} days...", end=" ")

        obs_vals = sub["Rainfall_mm"].values
        gfs_vals = np.full(len(sub), np.nan)

        for i, (_, row) in enumerate(sub.iterrows()):
            gfs_vals[i] = extract_gfs_for_row(row, srow)

        valid = ~np.isnan(gfs_vals)
        print(f"GFS valid: {valid.sum()}/{len(sub)}")

        if valid.sum() < 5:
            continue

        metrics = compute_metrics(obs_vals[valid], gfs_vals[valid], P90, P95)
        key = f"{stn}_{year}"
        all_results[key] = metrics

# ── Save ──
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE! Saved GFS baseline results -> {OUT_PATH}")
print(f"  {len(all_results)} station-year entries")
print(f"{'='*60}")
