"""Analyze year-by-year LOYO performance + data distribution."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Year-by-year LOYO performance
print("="*130)
print("  YEAR-BY-YEAR LOYO PERFORMANCE")
print("="*130)

d = json.load(open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json"))
loyo = d["loyo_per_year"]

print("\n  MODEL:")
hdr = f"  {'Year':>6} | {'CSI_r':>6} {'POD_r':>6} {'FAR_r':>6} | {'CSI90':>6} {'POD90':>6} {'FAR90':>6} {'SEDI90':>7} | {'CSI95':>6} {'POD95':>6} {'SEDI95':>7} | {'RMSE':>6} {'corr':>6} | {'n':>4} {'P90':>3} {'P95':>3}"
print(hdr)
print("  " + "-"*125)
for yr in sorted(loyo.keys(), key=int):
    m = loyo[yr]["ALL"]["model"]
    print(f"  {yr:>6} | {m['CSI_rain']:>6.3f} {m['POD_rain']:>6.3f} {m['FAR_rain']:>6.3f}"
          f" | {m['CSI_p90']:>6.3f} {m['POD_p90']:>6.3f} {m['FAR_p90']:>6.3f} {m['SEDI_p90']:>7.3f}"
          f" | {m['CSI_p95']:>6.3f} {m['POD_p95']:>6.3f} {m['SEDI_p95']:>7.3f}"
          f" | {m['RMSE']:>6.1f} {m['corr_rainy']:>6.3f}"
          f" | {m['n']:>4d} {m['n_p90']:>3d} {m['n_p95']:>3d}")

print("\n  ECMWF RAW:")
print(hdr)
print("  " + "-"*125)
for yr in sorted(loyo.keys(), key=int):
    m = loyo[yr]["ALL"]["ecmwf"]
    print(f"  {yr:>6} | {m['CSI_rain']:>6.3f} {m['POD_rain']:>6.3f} {m['FAR_rain']:>6.3f}"
          f" | {m['CSI_p90']:>6.3f} {m['POD_p90']:>6.3f} {m['FAR_p90']:>6.3f} {m['SEDI_p90']:>7.3f}"
          f" | {m['CSI_p95']:>6.3f} {m['POD_p95']:>6.3f} {m['SEDI_p95']:>7.3f}"
          f" | {m['RMSE']:>6.1f} {m['corr_rainy']:>6.3f}"
          f" | {m['n']:>4d} {m['n_p90']:>3d} {m['n_p95']:>3d}")

# 2. Ground truth distribution analysis per year
print("\n\n" + "="*130)
print("  GROUND TRUTH RAINFALL DISTRIBUTION BY YEAR")
print("="*130)

gt = pd.read_csv(r"D:\NEW_NRSC\Final_ground_truth_data.csv", parse_dates=["Date"])
gt = gt.dropna(subset=["Rainfall_mm"])
gt = gt[gt["Date"].dt.month.isin([6,7,8,9])]
gt["year"] = gt["Date"].dt.year

print(f"\n  {'Year':>6} | {'N':>5} | {'Mean':>6} {'Median':>6} {'Std':>6} | {'P50':>6} {'P75':>6} {'P90':>6} {'P95':>6} {'P99':>6} {'Max':>6} | {'Rain%':>6} {'P90+':>4} {'P95+':>4} {'P99+':>4} | {'Dry%':>5}")
print("  " + "-"*125)

for yr in range(2015, 2025):
    ydf = gt[gt["year"]==yr]["Rainfall_mm"]
    n = len(ydf)
    rainy = ydf[ydf >= 0.1]
    pct_rain = 100*len(rainy)/n if n > 0 else 0
    pct_dry = 100 - pct_rain
    
    if len(rainy) > 0:
        p90 = np.percentile(rainy, 90)
        p95 = np.percentile(rainy, 95)
    else:
        p90 = p95 = 0
    
    n_p90 = int((ydf >= 35.0).sum())  # approx P90 threshold
    n_p95 = int((ydf >= 49.0).sum())
    n_p99 = int((ydf >= 86.0).sum())
    
    print(f"  {yr:>6} | {n:>5} | {ydf.mean():>6.1f} {ydf.median():>6.1f} {ydf.std():>6.1f}"
          f" | {np.percentile(ydf,50):>6.1f} {np.percentile(ydf,75):>6.1f} {np.percentile(ydf,90):>6.1f}"
          f" {np.percentile(ydf,95):>6.1f} {np.percentile(ydf,99):>6.1f} {ydf.max():>6.1f}"
          f" | {pct_rain:>5.1f}% {n_p90:>4d} {n_p95:>4d} {n_p99:>4d}"
          f" | {pct_dry:>4.1f}%")

# 3. Monthly distribution per year
print("\n\n" + "="*130)
print("  MONTHLY MEAN RAINFALL (mm) PER YEAR")
print("="*130)
print(f"\n  {'Year':>6} | {'Jun':>8} {'Jul':>8} {'Aug':>8} {'Sep':>8} | {'JJAS_mean':>9}")
print("  " + "-"*65)
for yr in range(2015, 2025):
    ydf = gt[gt["year"]==yr]
    vals = []
    for m in [6,7,8,9]:
        mdf = ydf[ydf["Date"].dt.month == m]["Rainfall_mm"]
        vals.append(mdf.mean() if len(mdf)>0 else 0)
    print(f"  {yr:>6} | {vals[0]:>8.1f} {vals[1]:>8.1f} {vals[2]:>8.1f} {vals[3]:>8.1f} | {np.mean(vals):>9.1f}")

# 4. Station counts per year
print("\n\n" + "="*130)
print("  SAMPLES PER YEAR PER STATION")
print("="*130)
stations = sorted(gt["Station"].unique())
hdr2 = f"  {'Year':>6} |"
for s in stations:
    hdr2 += f" {s[:8]:>8}"
hdr2 += f" | {'Total':>6}"
print(f"\n{hdr2}")
print("  " + "-"*100)
for yr in range(2015, 2025):
    ydf = gt[gt["year"]==yr]
    row = f"  {yr:>6} |"
    for s in stations:
        n = len(ydf[ydf["Station"]==s])
        row += f" {n:>8d}"
    row += f" | {len(ydf):>6d}"
    print(row)

# 5. Extreme event rate per year
print("\n\n" + "="*130)
print("  EXTREME EVENT RATE (%) PER YEAR")
print("="*130)
print(f"\n  {'Year':>6} | {'>=0.1mm':>7} {'>=35mm':>7} {'>=49mm':>7} {'>=86mm':>7} | {'Ratio P90/Rain':>14} {'Ratio P95/Rain':>14}")
print("  " + "-"*90)
for yr in range(2015, 2025):
    ydf = gt[gt["year"]==yr]["Rainfall_mm"]
    n = len(ydf)
    n_rain = int((ydf>=0.1).sum())
    n_p90 = int((ydf>=35.0).sum())
    n_p95 = int((ydf>=49.0).sum())
    n_p99 = int((ydf>=86.0).sum())
    r_p90 = n_p90/n_rain*100 if n_rain > 0 else 0
    r_p95 = n_p95/n_rain*100 if n_rain > 0 else 0
    print(f"  {yr:>6} | {n_rain:>5}/{n:<5} {n_p90:>5}/{n:<5} {n_p95:>5}/{n:<5} {n_p99:>5}/{n:<5}"
          f" | {r_p90:>12.1f}% {r_p95:>12.1f}%")
