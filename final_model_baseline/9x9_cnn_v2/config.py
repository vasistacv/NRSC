"""
config.py — SpatialNet v2 Configuration
=========================================
Tuned for the new CNN architecture.
Key changes from baseline:
  - MODEL_TYPE = "SpatialNet"
  - Larger batch size (CNN benefits from larger batches for BN stability)
  - Lower LR with cosine annealing
  - Adjusted oversampling weights
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR        = Path(r"D:\NEW_NRSC")
ECMWF_DIR       = ROOT_DIR / "ecmwf_data"
GROUND_TRUTH    = ROOT_DIR / "Final_ground_truth_data.csv"
OUTPUT_DIR      = ROOT_DIR / "experiment_outputs_cnn_v2"

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
MODEL_TYPE = "SpatialNet"

CNN_BASE_CHANNELS   = 48
CNN_DEPTH           = 2
CNN_DROPOUT         = 0.10
MLP_HIDDEN          = [64, 32]
MLP_DROPOUT         = 0.15
FUSION_HIDDEN       = 96
FUSION_DROPOUT      = 0.20

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
BATCH_SIZE         = 96          # Larger for BN stability with CNN
NUM_EPOCHS         = 500         # More epochs — CNN needs longer
LR_INIT            = 3e-4        # Slightly lower for CNN
LR_MIN             = 1e-6
WEIGHT_DECAY        = 5e-4       # Less regularization (CNN has implicit reg from structure)

# Weighted sampler
OVERSAMPLE_P90     = 6.0
OVERSAMPLE_P95     = 10.0
OVERSAMPLE_P99     = 18.0
OVERSAMPLE_RAIN    = 1.5

# Early stopping
PATIENCE           = 100         # More patience for CNN convergence

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
SAVE_BEST_N   = 5
