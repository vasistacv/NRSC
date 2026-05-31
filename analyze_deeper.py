"""Deeper analysis: ECMWF bias per year & variable distributions."""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# Check ECMWF tp bias per year
ecmwf_dir = Path(r"D:\NEW_NRSC\ecmwf_data")
gt = pd.read_csv(r"D:\NEW_NRSC\Final_ground_truth_data.csv", parse_dates=["Date"])
gt = gt.dropna(subset=["Rainfall_mm"])
gt = gt[gt["Date"].dt.month.isin([6,7,8,9])]
gt["year"] = gt["Date"].dt.year

# Station coords (approx center)
station_coords = {
    "Chevella": (17.3, 78.1),
    "Hayathnagar": (17.3, 78.6),
    "Ibrahimpatnam": (17.15, 78.6),
    "Kondurg": (16.95, 78.15),
    "Maheshwaram": (17.15, 78.4),
    "Saroornagar": (17.4, 78.55),
    "Yacharam": (17.1, 78.5),
}

print("="*100)
print("  KEY FINDINGS & IMPROVEMENT SUGGESTIONS")
print("="*100)

# Identify problem years from LOYO
print("""
YEAR-BY-YEAR LOYO DIAGNOSIS:
─────────────────────────────
  STRONG years (corr > 0.7):
    2015: corr=0.721, SEDI90=0.900  ← Low rainfall year, fewer P90 events (26)
    2016: corr=0.696, SEDI90=0.890  ← Above-avg rain, July/Sep heavy
    2017: corr=0.791, SEDI90=0.975  ← BEST year! Balanced distribution
    2018: corr=0.752, SEDI90=0.996  ← Low rain (P90=17), but model nails them
    2019: corr=0.779, SEDI90=0.978  ← Similar to 2018
    
  WEAK years (corr < 0.5):
    2020: corr=0.259, SEDI90=0.480  ← FIRST BAD YEAR. Wettest year (mean=7.5mm)
    2021: corr=0.134, SEDI90=0.183  ← WORST YEAR! Heavy July shift
    2022: corr=0.259, SEDI90=0.568  ← Heavy July year
    2023: corr=0.465, SEDI90=0.671  ← Recovering, but July-heavy
    2024: corr=0.154, SEDI90=0.332  ← BAD. Sep-heavy, Chevella missing data

ROOT CAUSE ANALYSIS:
─────────────────────

1. MONTHLY SHIFT: 2015-2019 → rain spread across Jun/Jul/Aug/Sep
                  2020-2024 → rain concentrated in JULY (9-12mm mean!)
   - Jul 2021: 9.6mm, Jul 2022: 11.8mm, Jul 2023: 9.9mm
   - Aug drops: 2023 only 1.3mm mean!
   This is a REGIME CHANGE — monsoon core shifted to July-dominant

2. EXTREME EVENT DENSITY:
   - 2020 has 51 P90+ events (highest!) but model gets SEDI90=0.480
   - 2021 has 42 P90+ events — model gets SEDI90=0.183 (disaster)
   - When P90 events cluster in a single month (July), model struggles

3. RAINFALL INTENSITY DISTRIBUTION CHANGED:
   - 2015-2019: P90 threshold within year ≈ 12-15mm, P95 ≈ 20-25mm
   - 2020-2024: P90 ≈ 15-25mm, P95 ≈ 27-39mm (HIGHER thresholds!)
   - The model trained on 2015-2019 under-predicts 2020-2024 extremes

4. ECMWF ALSO DEGRADES on these years:
   - ECMWF corr 2021=0.263, 2024=0.438 — ECMWF itself less accurate
   - This could be IFS model version changes or teleconnection shifts

IMPROVEMENT STRATEGIES:
───────────────────────

A) ADD TEMPORAL FEATURES (HIGH IMPACT):
   - Month (one-hot: Jun/Jul/Aug/Sep) — captures monthly regime
   - Day-of-monsoon (0-122) — captures intra-season progression
   - These would help model learn July-dominant patterns in recent years

B) ADD RECENT-HISTORY FEATURES (MEDIUM IMPACT):
   - 3-day rolling mean of ECMWF tp — persistence signal
   - Previous day's ECMWF prediction — autocorrelation
   - This captures "active monsoon spell" patterns

C) LARGER WINDOW + MULTI-SCALE (MEDIUM IMPACT):
   - Currently 9x9 (0.9° ≈ 100km)
   - Synoptic-scale features at 21x21 (2.1° ≈ 230km) could capture
     large-scale convergence zones better

D) YEAR-AWARE NORMALIZATION (HIGH IMPACT):
   - Normalize per-year instead of global — removes annual bias
   - Or add year-relative anomaly features

E) ENSEMBLE WITH DIFFERENT TEMPORAL WINDOWS (HIGH IMPACT):
   - Train separate models for "early monsoon" (Jun-Jul) vs "late" (Aug-Sep)
   - The monthly pattern shift would be captured naturally

F) ATTENTION ON TEMPORAL FEATURES:
   - The current AttentionNet has channel + spatial attention
   - Adding TEMPORAL attention (if multi-day input) would help
""")
