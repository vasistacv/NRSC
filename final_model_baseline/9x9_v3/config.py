"""
config.py — v3 Enhanced SmallNet
================================
Same proven SmallNet architecture.
Key change: train on 2015-2022 (8 years, includes former val year 2022).
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(r"D:\NEW_NRSC")
ECMWF_DIR       = ROOT_DIR / "ecmwf_data"
GROUND_TRUTH    = ROOT_DIR / "Final_ground_truth_data.csv"
OUTPUT_DIR      = ROOT_DIR / "final_model_baseline" / "9x9_v3"

# ─────────────────────────────────────────────
# ECMWF GRIB SETTINGS
# ─────────────────────────────────────────────
SFC_VARS = ["tp", "tcwv", "cape", "d2m"]

# Variable reduction based on correlation + physics analysis:
#   Removed: vo (corr +0.04), u (corr +0.03), v (corr -0.11)
#   Selective levels: r@850,500 (drop 200), w@500 only (W850=0.00)
PL_VAR_LEVELS = {
    "r": [850, 500],   # RH 500 (~0.42), RH 850 (+0.16). Drop RH 200 (+0.09)
    "w": [500],         # W 500 (~0.48 HIGHEST). Drop W 850 (0.00 ZERO)
}
# For backward compatibility
PL_VARS = list(PL_VAR_LEVELS.keys())
PRESSURE_LEVELS = [850, 500, 200]  # master list (filtered per var)

# Total CNN channels: 4 surface + 2 (r) + 1 (w) = 7
N_CNN_CHANNELS = 7
# Tabular features: 13 (reduced from 24)
N_TABULAR = 13
GRID_RES = 0.1
MONSOON_MONTHS = [6, 7, 8, 9]

# ─────────────────────────────────────────────
# PATCH (WINDOW) SETTINGS
# ─────────────────────────────────────────────
WINDOW_SIZES = [3, 5, 9]
DEFAULT_WINDOW = 9

# ─────────────────────────────────────────────
# DATA SPLIT
# ─────────────────────────────────────────────
TRAIN_YEARS = list(range(2015, 2020))  # 2015-2019 (5 years)
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022, 2023, 2024]

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
MODEL_TYPE = "AttentionNet"

CNN_BASE_CHANNELS   = 48
CNN_DEPTH           = 2
CNN_DROPOUT         = 0.30
MLP_HIDDEN          = [128, 64]
MLP_DROPOUT         = 0.40
FUSION_HIDDEN       = 64
FUSION_DROPOUT      = 0.15

# ─────────────────────────────────────────────
# LOSS FUNCTION — aggressive FAR suppression
# ─────────────────────────────────────────────
TWEEDIE_P          = 1.5

# Aggressive penalties for underpredicting extreme events
PENALTY_P90_UNDER  = 10.0
PENALTY_P95_UNDER  = 20.0
PENALTY_P99_UNDER  = 30.0
PENALTY_FALSE_ALARM = 8.0

QUANTILE_BLEND_WEIGHT = 0.20

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE         = 48
NUM_EPOCHS         = 300       # attention converges faster
LR_INIT            = 3e-4
LR_MIN             = 1e-6
WEIGHT_DECAY       = 5e-3

# Weighted sampler — aggressive extreme oversampling
OVERSAMPLE_P90     = 10.0
OVERSAMPLE_P95     = 20.0
OVERSAMPLE_P99     = 40.0
OVERSAMPLE_RAIN    = 1.5
# dry days = weight 1.0

# Early stopping
PATIENCE           = 60

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
LOG_INTERVAL  = 20
SAVE_BEST_N   = 3
