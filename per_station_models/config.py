"""
config.py — Per-Station Model Configuration
=============================================
Copied from extracted_files/config.py and adapted for station-wise training.
Each station gets its own model trained only on that station's data.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(r"D:\NEW_NRSC")
ECMWF_DIR       = ROOT_DIR / "ecmwf_data"
GROUND_TRUTH    = ROOT_DIR / "Final_ground_truth_data.csv"
OUTPUT_DIR      = ROOT_DIR / "per_station_models" / "outputs"

# ─────────────────────────────────────────────
# ECMWF GRIB SETTINGS
# ─────────────────────────────────────────────
SFC_VARS = ["tp", "tcwv", "cape", "d2m"]
PL_VARS  = ["r", "w", "vo", "u", "v"]
PRESSURE_LEVELS = [850, 500, 200]
GRID_RES = 0.1
MONSOON_MONTHS = [6, 7, 8, 9]

# ─────────────────────────────────────────────
# PATCH (WINDOW) SETTINGS
# ─────────────────────────────────────────────
WINDOW_SIZES = [9]
DEFAULT_WINDOW = 9

# ─────────────────────────────────────────────
# DATA SPLIT
# ─────────────────────────────────────────────
TRAIN_YEARS = list(range(2015, 2022))
VAL_YEARS   = [2022, 2023]
TEST_YEARS  = [2024]

# ─────────────────────────────────────────────
# RAINFALL THRESHOLDS  (mm)
# ─────────────────────────────────────────────
DRY_THRESHOLD  = 0.1
P90_APPROX     = 35.0
P95_APPROX     = 48.97
P99_APPROX     = 86.17

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
MODEL_TYPE = "SmallNet"

CNN_BASE_CHANNELS   = 32
CNN_DEPTH           = 2
CNN_DROPOUT         = 0.15
MLP_HIDDEN          = [128, 64]
MLP_DROPOUT         = 0.20
FUSION_HIDDEN       = 64
FUSION_DROPOUT      = 0.15

# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────
TWEEDIE_P          = 1.5
PENALTY_P90_UNDER  = 4.0
PENALTY_P95_UNDER  = 8.0
PENALTY_P99_UNDER  = 15.0
PENALTY_FALSE_ALARM = 8.0
QUANTILE_BLEND_WEIGHT = 0.20

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE         = 64          # Same as best baseline
NUM_EPOCHS         = 400         # Same as best baseline
LR_INIT            = 5e-4
LR_MIN             = 1e-6
WEIGHT_DECAY       = 1e-3

# Weighted sampler — v4 values (same as best baseline)
OVERSAMPLE_P90     = 5.0
OVERSAMPLE_P95     = 8.0
OVERSAMPLE_P99     = 15.0
OVERSAMPLE_RAIN    = 1.5

# Early stopping
PATIENCE           = 80          # Same as best baseline

# Gradient clipping
GRAD_CLIP          = 1.0

# Random seed
SEED               = 42

# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────
TARGET_CSI_RAIN    = 0.50
TARGET_CSI_P90     = 0.50
TARGET_CSI_P95     = 0.50
TARGET_SEDI        = 0.60

# ─────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────
DEVICE = "cuda"
NUM_WORKERS = 0

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_INTERVAL  = 10
SAVE_BEST_N   = 3

# ─────────────────────────────────────────────
# STATIONS
# ─────────────────────────────────────────────
ALL_STATIONS = [
    "Chevella", "Hayathnagar", "Ibrahimpatnam",
    "Kondurg", "Maheshwaram", "Saroornagar", "Yacharam"
]
