import pandas as pd
import numpy as np

# Load ground truth
df = pd.read_csv(r"D:\NEW_NRSC\Final_ground_truth_data.csv", parse_dates=["Date"])
df = df[df["Date"].dt.month.isin([6, 7, 8, 9])]
df = df[df["Date"].dt.year == 2024]
df = df.dropna(subset=["Rainfall_mm"]).reset_index(drop=True)

# Extract GFS
gfs_cols = [c for c in df.columns if c.startswith("gfs_") and "prate" in c.lower()]
if not gfs_cols: gfs_cols = [c for c in df.columns if c.startswith("gfs_") and ("rain" in c.lower() or "precip" in c.lower())]

gfs_pred = np.zeros(len(df))
if gfs_cols:
    gfs_pred = df[gfs_cols[0]].values * 86400  # Convert kg/m2/s to mm/day if needed
else:
    # try to find the exact column
    cols = list(df.columns)
    for c in cols:
        if "gfs" in c.lower() and ("prate" in c.lower() or "rain" in c.lower()):
            gfs_pred = df[c].values
            break

# ECMWF 
ecmwf_cols = [c for c in df.columns if c.startswith("ecmwf_tp")]
ecmwf_pred = np.zeros(len(df))
if ecmwf_cols:
    ecmwf_pred = df[ecmwf_cols[0]].values * 1000 # Convert m to mm

obs = df["Rainfall_mm"].values

def _stats(ov, pv, name):
    rmse = np.sqrt(np.mean((ov - pv)**2))
    
    # All day corr
    if len(ov) > 1:
        corr_all = np.corrcoef(ov, pv)[0,1]
    else: corr_all = 0
    
    # Rainy day corr
    rainy = ov >= 0.1
    if rainy.sum() > 1:
        corr_rainy = np.corrcoef(ov[rainy], pv[rainy])[0,1]
    else: corr_rainy = 0
        
    print(f"[{name}]")
    print(f"  RMSE (All days):   {rmse:.2f} mm")
    print(f"  R    (All days):   {corr_all:.4f}")
    print(f"  R    (Rainy >=0.1):{corr_rainy:.4f}")
    print()

print(f"Total 2024 samples: {len(obs)}")
_stats(obs, ecmwf_pred, "ECMWF")
_stats(obs, gfs_pred, "GFS")
