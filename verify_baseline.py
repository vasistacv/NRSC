"""
Verify ECMWF tp units are correctly converted to mm.
Check the distribution of ECMWF point-extracted values vs ground truth.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import timedelta

ROOT_DIR = Path(r"D:\NEW_NRSC")
ECMWF_DIR = ROOT_DIR / "ecmwf_data"

# Load one SFC file and check raw tp values
print("=" * 60)
print("  ECMWF tp UNIT VERIFICATION")
print("=" * 60)

# 1. Show raw GRIB metadata
sfc_path = ECMWF_DIR / "2021" / "ecmwf_2021_07.sfc.grib"
ds = xr.open_dataset(
    str(sfc_path), engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
    indexpath=None
)
for dim in list(ds.dims):
    if dim not in ("latitude", "longitude", "time"):
        ds = ds.isel({dim: 0})

print(f"\nFile: {sfc_path.name}")
print(f"tp units from GRIB: {ds['tp'].attrs.get('units', 'UNKNOWN')}")
print(f"tp GRIB_paramId: {ds['tp'].attrs.get('GRIB_paramId', 'UNKNOWN')}")

# 2. Extract raw tp at a station location (Maheshwaram: 17.1342, 78.4334)
lat_arr = ds["latitude"].values
lon_arr = ds["longitude"].values
lat_idx = int(np.argmin(np.abs(lat_arr - 17.1342)))
lon_idx = int(np.argmin(np.abs(lon_arr - 78.4334)))

print(f"\nNearest grid point to Maheshwaram (17.1342, 78.4334):")
print(f"  lat[{lat_idx}] = {lat_arr[lat_idx]}, lon[{lon_idx}] = {lon_arr[lon_idx]}")

tp_raw = ds["tp"].values  # shape: (time, lat, lon)
if tp_raw.ndim > 3:
    tp_raw = tp_raw.squeeze()

tp_at_station_raw = tp_raw[:, lat_idx, lon_idx]  # in GRIB units (metres)
tp_at_station_mm = tp_at_station_raw * 1000.0     # converted to mm

print(f"\n--- Raw ECMWF tp at Maheshwaram for July 2021 ---")
print(f"  Raw (metres): min={tp_at_station_raw.min():.6f}, max={tp_at_station_raw.max():.6f}, mean={tp_at_station_raw.mean():.6f}")
print(f"  In mm (*1000): min={tp_at_station_mm.min():.2f}, max={tp_at_station_mm.max():.2f}, mean={tp_at_station_mm.mean():.2f}")

# 3. Compare with ground truth for same month/station
gt = pd.read_csv(ROOT_DIR / "Final_ground_truth_data.csv", parse_dates=["Date"])
gt_mahesh_jul21 = gt[(gt["Station"]=="Maheshwaram") & 
                      (gt["Date"].dt.year==2021) & 
                      (gt["Date"].dt.month==7)]
gt_vals = gt_mahesh_jul21["Rainfall_mm"].dropna().values

print(f"\n--- Ground Truth: Maheshwaram July 2021 ---")
print(f"  GT (mm): min={gt_vals.min():.2f}, max={gt_vals.max():.2f}, mean={gt_vals.mean():.2f}")
print(f"  GT days with rain>=0.1mm: {(gt_vals >= 0.1).sum()} / {len(gt_vals)}")

# 4. Day-by-day comparison (with D-1 offset)
print(f"\n--- Day-by-day comparison (D-1 aligned) ---")
print(f"{'GT Date':12s} | {'GT (mm)':>8s} | {'ECMWF raw(m)':>12s} | {'ECMWF (mm)':>10s} | {'Match?':>6s}")
print("-" * 65)

time_vals = pd.DatetimeIndex(ds.time.values)
ecmwf_by_date = {}
for ti, t in enumerate(time_vals):
    ecmwf_by_date[t.date()] = float(tp_at_station_raw[ti])

count = 0
for _, row in gt_mahesh_jul21.iterrows():
    gt_date = row["Date"].date()
    ecmwf_date = gt_date - timedelta(days=1)  # D-1
    
    if ecmwf_date in ecmwf_by_date:
        raw_val = ecmwf_by_date[ecmwf_date]
        mm_val = raw_val * 1000.0
        gt_val = row["Rainfall_mm"]
        
        # "Match" = both agree on rain/no-rain
        both_rain = (mm_val >= 0.1) and (gt_val >= 0.1)
        both_dry = (mm_val < 0.1) and (gt_val < 0.1)
        match = "YES" if (both_rain or both_dry) else "NO"
        
        if count < 15:  # Show first 15 days
            print(f"{str(gt_date):12s} | {gt_val:8.2f} | {raw_val:12.6f} | {mm_val:10.2f} | {match:>6s}")
        count += 1

print(f"... ({count} total matched days)")

# 5. Check: Does ECMWF ever predict >= 35mm (P90) at this point?
all_ecmwf_mm = np.array([v * 1000.0 for v in ecmwf_by_date.values()])
print(f"\n--- ECMWF tp distribution at Maheshwaram (July 2021) ---")
print(f"  Total days: {len(all_ecmwf_mm)}")
print(f"  Days >= 0.1mm: {(all_ecmwf_mm >= 0.1).sum()}")
print(f"  Days >= 10mm:  {(all_ecmwf_mm >= 10.0).sum()}")
print(f"  Days >= 35mm (P90): {(all_ecmwf_mm >= 35.0).sum()}")
print(f"  Days >= 49mm (P95): {(all_ecmwf_mm >= 49.0).sum()}")
print(f"  P50: {np.percentile(all_ecmwf_mm, 50):.2f} mm")
print(f"  P90: {np.percentile(all_ecmwf_mm, 90):.2f} mm")
print(f"  P99: {np.percentile(all_ecmwf_mm, 99):.2f} mm")
print(f"  Max: {all_ecmwf_mm.max():.2f} mm")

ds.close()
