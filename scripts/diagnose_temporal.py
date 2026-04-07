"""
diagnose_temporal.py
====================
Deep analysis of the temporal alignment between ECMWF GRIB and Ground Truth.

ECMWF download config:
  time  = "12:00:00"   (init time = 12 UTC)
  step  = "24"         (24h forecast lead)
  date  = "2015-06-01/to/2015-06-30"

So: init 2015-06-01 12UTC + 24h step => valid at 2015-06-02 12UTC
    init 2015-06-30 12UTC + 24h step => valid at 2015-07-01 12UTC

IMD ground truth:
  Rainfall_mm for date D = accumulated rainfall 08:30 IST Day D to 08:30 IST Day D+1
  (standard IMD convention)

IST = UTC + 5:30

So: GT date "2015-06-03" = rain accumulated from 2015-06-03 03:00 UTC 
                                                to 2015-06-04 03:00 UTC

Best ECMWF match for GT date "2015-06-03":
  ECMWF init on 2015-06-02 at 12:00 UTC, step=24h => valid 2015-06-03 12UTC
  This forecast "sees" the atmosphere on the evening of June 2nd
  and predicts 24h of weather through June 3rd 12UTC.
  The GT accumulation period (June 3 03UTC to June 4 03UTC) overlaps heavily
  with this forecast.

Therefore: GT Day D should match ECMWF Day D-1 (init date).

This script verifies GRIB time axes to confirm.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, r"D:\NEW_NRSC\extracted_files")

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

ECMWF_DIR = Path(r"D:\NEW_NRSC\ecmwf_data")
GT_PATH   = Path(r"D:\NEW_NRSC\Final_ground_truth_data.csv")

print("=" * 70)
print("  TEMPORAL ALIGNMENT DIAGNOSTIC")
print("=" * 70)

# ── 1. Examine GRIB time axes ──────────────────────────────────────────────
print("\n[1] ECMWF GRIB Time Structure")
print("-" * 50)

test_file = ECMWF_DIR / "2015" / "ecmwf_2015_06.sfc.grib"
if test_file.exists():
    ds = xr.open_dataset(str(test_file), engine="cfgrib",
                         backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
                         indexpath=None)
    print(f"  File: {test_file.name}")
    print(f"  Dimensions: {dict(ds.dims)}")
    
    if "time" in ds.coords:
        times = pd.DatetimeIndex(ds.time.values)
        print(f"  time coord (init times):")
        print(f"    First: {times[0]}")
        print(f"    Last:  {times[-1]}")
        print(f"    Count: {len(times)}")
        print(f"    Sample: {times[:5].tolist()}")
    
    if "valid_time" in ds.coords:
        vtimes = pd.DatetimeIndex(np.atleast_1d(ds.valid_time.values))
        print(f"  valid_time coord:")
        print(f"    First: {vtimes[0]}")
        print(f"    Last:  {vtimes[-1]}")
        print(f"    Count: {len(vtimes)}")
        print(f"    Sample: {vtimes[:5].tolist()}")
    
    if "step" in ds.coords:
        print(f"  step: {ds.step.values}")
    
    # Print all coordinate info
    print(f"\n  All coordinates:")
    for c in ds.coords:
        val = ds.coords[c].values
        if hasattr(val, '__len__') and len(np.atleast_1d(val)) > 1:
            arr = np.atleast_1d(val)
            print(f"    {c}: [{arr[0]} ... {arr[-1]}] (n={len(arr)})")
        else:
            print(f"    {c}: {val}")
    ds.close()

# ── 2. Examine GT date range ───────────────────────────────────────────────
print("\n[2] Ground Truth Date Structure")
print("-" * 50)

gt = pd.read_csv(GT_PATH, parse_dates=["Date"])
print(f"  Columns: {list(gt.columns)}")
print(f"  Date range: {gt['Date'].min()} to {gt['Date'].max()}")
print(f"  Total rows: {len(gt)}")
print(f"  Stations: {sorted(gt['Station'].unique())}")
print(f"\n  June 2015 dates (first 10):")
gt_june = gt[(gt['Date'].dt.year == 2015) & (gt['Date'].dt.month == 6)]
for d in sorted(gt_june['Date'].unique())[:10]:
    n_stations = len(gt_june[gt_june['Date'] == d])
    rain_vals = gt_june[gt_june['Date'] == d]['Rainfall_mm'].values
    print(f"    {d.strftime('%Y-%m-%d')} — {n_stations} stations, "
          f"rain range [{rain_vals.min():.1f}, {rain_vals.max():.1f}]")

# ── 3. Alignment Analysis ──────────────────────────────────────────────────
print("\n[3] TEMPORAL ALIGNMENT ANALYSIS")
print("-" * 50)

print("""
  ECMWF Download Configuration:
    type    = fc (forecast)
    time    = 12:00:00 UTC (initialization)
    step    = 24 hours
    date    = YYYY-MM-01 to YYYY-MM-{ndays}

  This means each GRIB "time" entry is an INIT DATE at 12:00 UTC.
  The VALID TIME = init_date + 24h = next day at 12:00 UTC.

  IMD Rainfall Convention:
    Date D = rainfall accumulated 08:30 IST Day D to 08:30 IST Day D+1
    08:30 IST = 03:00 UTC

  ALIGNMENT TABLE:
  ┌─────────────────────────────┬───────────────────────────────┐
  │ ECMWF GRIB "time" (init)   │ Forecast valid period         │
  ├─────────────────────────────┼───────────────────────────────┤
  │ 2015-06-01 12:00 UTC       │ → 2015-06-02 12:00 UTC        │
  │ 2015-06-02 12:00 UTC       │ → 2015-06-03 12:00 UTC        │
  │ 2015-06-03 12:00 UTC       │ → 2015-06-04 12:00 UTC        │
  └─────────────────────────────┴───────────────────────────────┘

  ┌─────────────────────────────┬───────────────────────────────┐
  │ GT "Date"                   │ Accumulation period           │
  ├─────────────────────────────┼───────────────────────────────┤
  │ 2015-06-02                  │ 06-02 03:00Z → 06-03 03:00Z  │
  │ 2015-06-03                  │ 06-03 03:00Z → 06-04 03:00Z  │
  └─────────────────────────────┴───────────────────────────────┘

  CONCLUSION:
    For GT Date D, the best ECMWF match is init date D-1.
    
    Example: GT "2015-06-03" (rain from 03Z Jun3 to 03Z Jun4)
             → ECMWF init "2015-06-02" 12Z (valid at 03Z Jun3 12Z)
             The 24h forecast starting 12Z Jun2 covers the bulk of
             the GT accumulation window.

    CURRENT BUG: dataset.py matches GT date D with ECMWF date D
    (same day), which is WRONG. The ECMWF forecast for init date D
    is actually predicting weather for Day D+1, NOT Day D.
    This means the model has been training on FUTURE weather data
    that doesn't correspond to the observed rainfall!
""")

# ── 4. Verify with actual GRIB data ────────────────────────────────────────
print("[4] Verifying GRIB time entries")
print("-" * 50)

if test_file.exists():
    ds = xr.open_dataset(str(test_file), engine="cfgrib",
                         backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
                         indexpath=None)
    
    if "time" in ds.dims:
        grib_dates = pd.DatetimeIndex(ds.time.values).date
        gt_dates = sorted(gt_june['Date'].dt.date.unique())
        
        print(f"  GRIB init dates (June 2015): {len(grib_dates)} entries")
        print(f"  GT dates (June 2015): {len(gt_dates)} entries")
        
        # Check overlap
        grib_set = set(grib_dates)
        gt_set = set(gt_dates)
        overlap = grib_set & gt_set
        
        print(f"  Direct date overlap: {len(overlap)} days")
        
        # Check D-1 alignment
        from datetime import timedelta
        gt_minus1 = {d - timedelta(days=1) for d in gt_dates}
        overlap_shifted = grib_set & gt_minus1
        print(f"  D-1 shifted overlap: {len(overlap_shifted)} days")
        
        grib_list = sorted(grib_dates)
        print(f"\n  First 5 GRIB init dates: {grib_list[:5]}")
        print(f"  First 5 GT dates:        {gt_dates[:5]}")
        print(f"\n  → GRIB date {grib_list[0]} predicts weather valid at "
              f"{grib_list[0] + timedelta(days=1)}")
        print(f"  → This should align with GT date {grib_list[0] + timedelta(days=1)}")
    
    ds.close()

print("\n" + "=" * 70)
print("  DIAGNOSIS COMPLETE")
print("=" * 70)
