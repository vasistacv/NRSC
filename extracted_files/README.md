# Telangana Extreme Rainfall Downscaling
## DualPathNet — CNN + MLP Fusion for ECMWF IFS Data

---

## Architecture Overview

```
Patch (19 ch, 3×3 or 5×5)           Tabular features (24)
        │                                     │
  ┌─────▼──────┐                    ┌─────────▼────────┐
  │ CNN Branch │                    │   MLP Branch     │
  │ Stem Conv  │                    │  256 → 128 → 64  │
  │ ResBlock×4 │                    │  BN + SiLU + DO  │
  │ SE-gating  │                    └─────────┬────────┘
  │ AvgPool    │                              │
  │ Linear→64  │                              │
  └─────┬──────┘                              │
        └──────────────┬──────────────────────┘
                 concat (128)
                       │
              ┌────────▼────────┐
              │  Fusion Head    │
              │  128 → 64       │
              └──────┬──────────┘
                ┌────┴─────┐
         occ_logit      intensity
         sigmoid()      Softplus()
                └────×───┘
                 prediction (mm)
```

**Why DualPathNet?**
- CNN captures mesoscale spatial organisation (convergence, gradients)
- MLP captures large-scale thermodynamics (CAPE, TCWV, shear)
- Dual-head separates occurrence (classification) from intensity (regression)
- SE-gating learns which channels matter most per spatial context

---

## File Structure

```
rainfall_downscaling/
├── config.py       — All paths, hyperparameters, thresholds
├── dataset.py      — GRIB reader, patch builder, normaliser, DataLoader
├── model.py        — DualPathNet architecture
├── losses.py       — TweedieExLoss + QuantileBlend + OccurrenceBCE
├── metrics.py      — CSI, FAR, POD, SEDI for all categories
├── train.py        — Full training loop + window comparison
├── predict.py      — Inference on new data
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
conda create -n rainfall python=3.11
conda activate rainfall
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Install eccodes (required for cfgrib on Windows)
```bash
conda install -c conda-forge eccodes
```

### 3. Edit config.py
```python
ROOT_DIR     = Path(r"D:\NEW_NRSC")
ECMWF_DIR    = ROOT_DIR / "ecmwf_data"
GROUND_TRUTH = ROOT_DIR / "Final_ground_truth_data.csv"
OUTPUT_DIR   = ROOT_DIR / "experiment_outputs"
```

---

## Running

### Quick smoke test (5 epochs)
```bash
cd D:\NEW_NRSC
python train.py --window 3 --debug
```

### Train single window
```bash
python train.py --window 3    # 3×3 patch (~33 km)
python train.py --window 5    # 5×5 patch (~55 km)
```

### Train both windows and compare (recommended)
```bash
python train.py
```

### Inference on held-out data
```bash
python predict.py \
  --checkpoint experiment_outputs/window_3/ckpt_best.pt \
  --norm       experiment_outputs/window_3/normaliser.npz \
  --years 2024 \
  --csv    predictions_2024.csv
```

---

## Key Design Decisions

### Why 3×3 instead of 31×31?
- Extreme rainfall in Telangana is convective — sub-100 km scale
- 31×31 at 0.1° = 3.1° = ~340 km — far too large, averages out extremes
- 3×3 at 0.1° = 0.3° = ~33 km — tight local signal
- 5×5 = ~55 km — slightly broader mesoscale

### Why weighted sampling?
- 61.4% dry days, only 6% P90+ events
- Without resampling: model ignores extremes
- P99+ events weighted 40× — ensures every batch sees extreme examples

### Why dual-head (occurrence × intensity)?
- Pure regression: model hedges toward safe mean → kills CSI
- Dual-head: explicit classification keeps FAR controlled
  while intensity regression focuses on magnitude of rainy events

### Why TweedieExLoss with p=1.5?
- Tweedie p=1.5 handles zero-inflated + heavy-tailed distribution
- Asymmetric penalties: P90 under-prediction costs 10×, P99 costs 40×
- Plus false-alarm penalty (3×) to keep FAR low

### Why OneCycleLR?
- Small dataset (~7K training samples) benefits from warm-up + annealing
- Reaches good optima faster than cosine annealing alone

---

## Expected Performance (based on architecture design)

| Metric       | Target  | Realistic range (monsoon DL) |
|-------------|---------|------------------------------|
| CSI_rain    | > 0.50  | 0.45 – 0.60                  |
| CSI_P90     | > 0.50  | 0.40 – 0.55                  |
| CSI_P95     | > 0.50  | 0.30 – 0.45 (very hard)      |
| SEDI_P90    | > 0.60  | 0.55 – 0.70                  |
| FAR_rain    | < 0.30  | 0.25 – 0.40                  |

Note: CSI_P95 > 0.50 rivals operational NWP systems. Achieving it
with ~7K training samples would be exceptional. Focus first on
CSI_P90 and SEDI_P90 as primary benchmarks.

---

## Troubleshooting

**GRIB load error: "Dimensions {'step'} do not exist"**
→ Already handled — dataset.py collapses all non-spatial dims with isel(0)

**Windows pickling error in DataLoader**
→ Set `NUM_WORKERS = 0` in config.py

**CUDA out of memory**
→ Reduce BATCH_SIZE to 64 in config.py

**No samples built**
→ Check ECMWF_DIR path, ensure year subdirectories exist
→ Run: python -c "from dataset import RainfallDataBuilder; b=RainfallDataBuilder(); b.build([2021])"

