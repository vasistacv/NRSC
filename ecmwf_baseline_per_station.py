"""
ecmwf_baseline_per_station.py
=============================
Task 1 from supervisor: "See how much skill we can extract from ECMWF."

Computes raw ECMWF tp (total precipitation) skill against ground truth
for EACH station INDIVIDUALLY and for EACH YEAR.

This establishes the absolute baseline floor that your ML model must beat.

Output: JSON report + printed table.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import timedelta

# ── Config ──────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(r"D:\NEW_NRSC")
ECMWF_DIR    = ROOT_DIR / "ecmwf_data"
GT_PATH      = ROOT_DIR / "Final_ground_truth_data.csv"
OUTPUT_PATH  = ROOT_DIR / "ecmwf_baseline_results.json"

SFC_VARS = ["tp"]
MONSOON_MONTHS = [6, 7, 8, 9]
DRY_THRESHOLD = 0.1  # mm

ALL_YEARS = list(range(2015, 2025))
TRAIN_YEARS = list(range(2015, 2022))
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024]


# ── Metric Functions ────────────────────────────────────────────────────────

def contingency(pred, obs, threshold):
    """Binary contingency table."""
    pred_yes = pred >= threshold
    obs_yes  = obs >= threshold
    H  = int(np.sum(pred_yes & obs_yes))
    M  = int(np.sum(~pred_yes & obs_yes))
    FA = int(np.sum(pred_yes & ~obs_yes))
    CN = int(np.sum(~pred_yes & ~obs_yes))
    return H, M, FA, CN

def compute_metrics(pred, obs, threshold, label=""):
    H, M, FA, CN = contingency(pred, obs, threshold)
    n = H + M + FA + CN
    
    pod  = H / (H + M) if (H + M) > 0 else 0.0
    far  = FA / (H + FA) if (H + FA) > 0 else 0.0
    csi  = H / (H + M + FA) if (H + M + FA) > 0 else 0.0
    fbi  = (H + FA) / (H + M) if (H + M) > 0 else 0.0
    
    # HSS
    expected = ((H+M)*(H+FA) + (CN+M)*(CN+FA)) / n if n > 0 else 0
    hss = (H + CN - expected) / (n - expected) if (n - expected) != 0 else 0.0
    
    # SEDI
    hr = H / (H + M) if (H + M) > 0 else 0.0
    far_rate = FA / (FA + CN) if (FA + CN) > 0 else 0.0
    if hr > 0 and hr < 1 and far_rate > 0 and far_rate < 1:
        num = np.log(far_rate) - np.log(hr) - np.log(1-far_rate) + np.log(1-hr)
        den = np.log(far_rate) + np.log(hr) + np.log(1-far_rate) + np.log(1-hr)
        sedi = num / den if den != 0 else 0.0
    else:
        sedi = 0.0
    
    return {
        f"CSI_{label}": round(csi, 4),
        f"POD_{label}": round(pod, 4),
        f"FAR_{label}": round(far, 4),
        f"FBI_{label}": round(fbi, 4),
        f"HSS_{label}": round(hss, 4),
        f"SEDI_{label}": round(sedi, 4),
        f"H_{label}": H, f"M_{label}": M, f"FA_{label}": FA,
        f"n_obs_{label}": H + M,
    }


# ── GRIB Reader ─────────────────────────────────────────────────────────────

def load_ecmwf_tp_for_year(year):
    """Load daily ECMWF total precipitation for a year, return {date: (lat_arr, lon_arr, tp_field_mm)}."""
    year_dir = ECMWF_DIR / str(year)
    if not year_dir.exists():
        print(f"  [WARN] Year dir missing: {year_dir}")
        return {}
    
    date_fields = {}
    
    for month in MONSOON_MONTHS:
        sfc_path = year_dir / f"ecmwf_{year}_{month:02d}.sfc.grib"
        if not sfc_path.exists():
            continue
        
        try:
            ds = xr.open_dataset(
                str(sfc_path), engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
                indexpath=None
            )
            # Collapse non-spatial/time dims
            for dim in list(ds.dims):
                if dim not in ("latitude", "longitude", "time"):
                    ds = ds.isel({dim: 0})
            
            lat_arr = ds["latitude"].values
            lon_arr = ds["longitude"].values
            
            if "time" in ds.dims and ds.dims["time"] > 1:
                time_vals = pd.DatetimeIndex(ds.time.values)
                for ti, t in enumerate(time_vals):
                    # tp is in metres, convert to mm
                    tp_field = ds["tp"].isel(time=ti).values.astype(np.float32) * 1000.0
                    date_fields[t.date()] = (lat_arr, lon_arr, tp_field)
            else:
                t = pd.Timestamp(ds.time.values)
                tp_field = ds["tp"].values.astype(np.float32) * 1000.0
                if tp_field.ndim > 2:
                    tp_field = tp_field.squeeze()
                date_fields[t.date()] = (lat_arr, lon_arr, tp_field)
            
            ds.close()
        except Exception as e:
            print(f"  [WARN] Error reading {sfc_path.name}: {e}")
    
    # Also try to load the month before June (May) for D-1 alignment on June 1
    may_path = year_dir / f"ecmwf_{year}_05.sfc.grib"
    if may_path.exists():
        try:
            ds = xr.open_dataset(
                str(may_path), engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
                indexpath=None
            )
            for dim in list(ds.dims):
                if dim not in ("latitude", "longitude", "time"):
                    ds = ds.isel({dim: 0})
            lat_arr = ds["latitude"].values
            lon_arr = ds["longitude"].values
            if "time" in ds.dims and ds.dims["time"] > 1:
                time_vals = pd.DatetimeIndex(ds.time.values)
                for ti, t in enumerate(time_vals):
                    tp_field = ds["tp"].isel(time=ti).values.astype(np.float32) * 1000.0
                    date_fields[t.date()] = (lat_arr, lon_arr, tp_field)
            ds.close()
        except:
            pass
    
    return date_fields


def extract_point_value(lat, lon, lat_arr, lon_arr, field):
    """Extract nearest-grid-point value."""
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
    return float(field[lat_idx, lon_idx])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ECMWF RAW SKILL BASELINE - PER STATION, PER YEAR")
    print("=" * 70)
    
    # Load ground truth
    gt = pd.read_csv(GT_PATH, parse_dates=["Date"])
    gt = gt.dropna(subset=["Rainfall_mm"])
    gt = gt[gt["Date"].dt.month.isin(MONSOON_MONTHS)]
    gt["year"] = gt["Date"].dt.year
    
    stations = gt["Station"].unique()
    print(f"\nStations: {list(stations)}")
    print(f"GT rows: {len(gt):,}")
    
    # Compute P90/P95 thresholds from training years only
    train_gt = gt[gt["year"].isin(TRAIN_YEARS)]
    rainy_train = train_gt[train_gt["Rainfall_mm"] >= DRY_THRESHOLD]["Rainfall_mm"]
    p90 = float(np.percentile(rainy_train, 90))
    p95 = float(np.percentile(rainy_train, 95))
    print(f"Thresholds (from train): P90={p90:.2f} mm, P95={p95:.2f} mm")
    
    # Process year by year
    results = {}
    
    for year in ALL_YEARS:
        print(f"\n{'-'*50}")
        print(f"  Year {year}")
        print(f"{'-'*50}")
        
        ecmwf_data = load_ecmwf_tp_for_year(year)
        print(f"  ECMWF dates loaded: {len(ecmwf_data)}")
        
        gt_year = gt[gt["year"] == year]
        
        for station in stations:
            gt_station = gt_year[gt_year["Station"] == station]
            if len(gt_station) == 0:
                continue
            
            preds = []
            obs = []
            
            for _, row in gt_station.iterrows():
                gt_date = row["Date"].date()
                lat, lon = row["Lat"], row["Lon"]
                rain_mm = float(row["Rainfall_mm"])
                
                # D-1 alignment: GT date D -> ECMWF init date D-1
                ecmwf_date = gt_date - timedelta(days=1)
                
                if ecmwf_date not in ecmwf_data:
                    continue
                
                lat_arr, lon_arr, tp_field = ecmwf_data[ecmwf_date]
                ecmwf_pred = extract_point_value(lat, lon, lat_arr, lon_arr, tp_field)
                
                preds.append(ecmwf_pred)
                obs.append(rain_mm)
            
            if len(preds) < 10:
                continue
            
            preds = np.array(preds)
            obs = np.array(obs)
            
            # Compute metrics
            metrics_rain = compute_metrics(preds, obs, DRY_THRESHOLD, "rain")
            metrics_p90  = compute_metrics(preds, obs, p90, "p90")
            metrics_p95  = compute_metrics(preds, obs, p95, "p95")
            
            # Correlation on rainy days
            rainy_mask = obs >= DRY_THRESHOLD
            corr = float(np.corrcoef(preds[rainy_mask], obs[rainy_mask])[0, 1]) if rainy_mask.sum() > 5 else 0.0
            rmse = float(np.sqrt(np.mean((preds - obs) ** 2)))
            
            all_metrics = {**metrics_rain, **metrics_p90, **metrics_p95,
                           "corr_rainy": round(corr, 4), "RMSE": round(rmse, 4),
                           "n_samples": len(preds)}
            
            key = f"{station}_{year}"
            results[key] = all_metrics
            
            print(f"  {station:15s} | CSI_rain={metrics_rain['CSI_rain']:.3f} "
                  f"FAR_rain={metrics_rain['FAR_rain']:.3f} "
                  f"CSI_p90={metrics_p90['CSI_p90']:.3f} "
                  f"SEDI_p90={metrics_p90['SEDI_p90']:.3f} "
                  f"corr={corr:.3f}")
    
    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  SUMMARY: RAW ECMWF SKILL BY STATION (TEST YEAR 2024)")
    print(f"{'='*90}")
    print(f"{'Station':15s} | {'CSI_rain':>8} | {'FAR_rain':>8} | {'CSI_p90':>7} | {'SEDI_p90':>8} | {'CSI_p95':>7} | {'SEDI_p95':>8} | {'Corr':>5}")
    print("-" * 90)
    
    for station in stations:
        key = f"{station}_2024"
        if key not in results:
            continue
        m = results[key]
        print(f"{station:15s} | {m['CSI_rain']:8.4f} | {m['FAR_rain']:8.4f} | "
              f"{m['CSI_p90']:7.4f} | {m['SEDI_p90']:8.4f} | "
              f"{m['CSI_p95']:7.4f} | {m['SEDI_p95']:8.4f} | {m['corr_rainy']:5.3f}")
    
    # ── Year-over-year for all stations combined ─────────────────────────────
    print(f"\n{'='*90}")
    print(f"  YEAR-OVER-YEAR: ALL STATIONS COMBINED")
    print(f"{'='*90}")
    print(f"{'Year':6s} | {'CSI_rain':>8} | {'FAR_rain':>8} | {'CSI_p90':>7} | {'SEDI_p90':>8} | {'CSI_p95':>7} | {'Corr':>5}")
    print("-" * 70)
    
    for year in ALL_YEARS:
        # Aggregate all stations for this year
        year_preds = []
        year_obs = []
        for station in stations:
            key = f"{station}_{year}"
            if key not in results:
                continue
            # We need the raw data — recompute from the stored metrics is tricky
            # Just print the average
        
        # Average across stations
        year_metrics = [results.get(f"{s}_{year}") for s in stations if f"{s}_{year}" in results]
        if not year_metrics:
            continue
        avg_csi_rain = np.mean([m["CSI_rain"] for m in year_metrics])
        avg_far_rain = np.mean([m["FAR_rain"] for m in year_metrics])
        avg_csi_p90  = np.mean([m["CSI_p90"] for m in year_metrics])
        avg_sedi_p90 = np.mean([m["SEDI_p90"] for m in year_metrics])
        avg_csi_p95  = np.mean([m["CSI_p95"] for m in year_metrics])
        avg_corr     = np.mean([m["corr_rainy"] for m in year_metrics])
        
        print(f"{year:6d} | {avg_csi_rain:8.4f} | {avg_far_rain:8.4f} | "
              f"{avg_csi_p90:7.4f} | {avg_sedi_p90:8.4f} | {avg_csi_p95:7.4f} | {avg_corr:5.3f}")
    
    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
