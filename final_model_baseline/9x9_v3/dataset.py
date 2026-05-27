"""
dataset.py
==========
ECMWF GRIB reader + Ground Truth alignment for rainfall downscaling.

CRITICAL TEMPORAL ALIGNMENT:
  ECMWF: init_time=12:00 UTC, step=24h => valid at D+1 12:00 UTC
  IMD GT: Date D = rainfall accumulated 08:30 IST Day D to 08:30 IST Day D+1
  
  Therefore: GT Day D => use ECMWF init Day D-1
  
  This means the GRIB field from init 2015-06-02 12UTC is the forecast
  that covers the period including GT date 2015-06-03.

Channel layout (19 channels):
  [0]  tp        SFC  total precipitation
  [1]  tcwv      SFC  total column water vapour
  [2]  cape      SFC  convective available potential energy
  [3]  d2m       SFC  2m dewpoint temperature
  [4-6]   r   @ 850/500/200 hPa  relative humidity
  [7-9]   w   @ 850/500/200 hPa  vertical velocity
  [10-12]  vo @ 850/500/200 hPa  vorticity
  [13-15]  u  @ 850/500/200 hPa  U-wind
  [16-18]  v  @ 850/500/200 hPa  V-wind
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import timedelta

import xarray as xr

import config


# ──────────────────────────────────────────────────────────────────────────────
# GRIB READER — PRESERVES TIME DIMENSION
# ──────────────────────────────────────────────────────────────────────────────

def _open_sfc_grib(path: Path) -> xr.Dataset:
    """Open a surface GRIB file, KEEPING the time dimension intact."""
    datasets = []
    for var in config.SFC_VARS:
        try:
            filter_name = "2d" if var == "d2m" else var
            ds = xr.open_dataset(
                str(path),
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": filter_name}},
                indexpath=None
            )
            # Collapse ONLY step/number dims, KEEP time
            for dim in list(ds.dims):
                if dim not in ("latitude", "longitude", "time"):
                    ds = ds.isel({dim: 0})
            
            # Rename the data variable to our standard name
            data_vars = list(ds.data_vars)
            if data_vars and data_vars[0] != var:
                ds = ds.rename({data_vars[0]: var})
            
            datasets.append(ds[[var]])
        except Exception as e:
            print(f"  [WARN] SFC var {var} failed in {path.name}: {e}")
    
    if not datasets:
        raise RuntimeError(f"Could not load any SFC variables from {path}")
    return xr.merge(datasets)


def _open_pl_grib(path: Path) -> xr.Dataset:
    """Open a pressure-level GRIB file, KEEPING the time dimension intact."""
    datasets = []
    for var in config.PL_VARS:
        try:
            ds = xr.open_dataset(
                str(path),
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": var}},
                indexpath=None
            )
            # Keep only desired pressure levels
            if "isobaricInhPa" in ds.dims:
                levels_available = ds["isobaricInhPa"].values
                levels_to_use = [l for l in config.PRESSURE_LEVELS if l in levels_available]
                if not levels_to_use:
                    continue
                ds = ds.sel(isobaricInhPa=levels_to_use)
            
            # Collapse ONLY step/number, KEEP time and isobaricInhPa
            for dim in list(ds.dims):
                if dim not in ("latitude", "longitude", "time", "isobaricInhPa"):
                    ds = ds.isel({dim: 0})
            
            datasets.append(ds[[var]])
        except Exception as e:
            print(f"  [WARN] PL var {var} failed in {path.name}: {e}")
    
    if not datasets:
        raise RuntimeError(f"Could not load any PL variables from {path}")
    return xr.merge(datasets)


def _parse_month_from_filename(filename: str) -> int:
    parts = filename.replace(".sfc.grib", "").replace(".pl.grib", "").split("_")
    return int(parts[-1])


# ──────────────────────────────────────────────────────────────────────────────
# GRID UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _lat_lon_to_grid_idx(lat, lon, lat_arr, lon_arr):
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))
    return lat_idx, lon_idx


def _extract_patch(field, lat_idx, lon_idx, half):
    H, W = field.shape
    r0, r1 = lat_idx - half, lat_idx + half + 1
    c0, c1 = lon_idx - half, lon_idx + half + 1
    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
        return None
    return field[r0:r1, c0:c1].copy()


# ──────────────────────────────────────────────────────────────────────────────
# TABULAR FEATURE DERIVATION
# ──────────────────────────────────────────────────────────────────────────────

def derive_tabular_features(sfc_patch, pl_data):
    """Compute physics-informed scalar indices from patch-centre values."""
    mid = sfc_patch.shape[-1] // 2

    tp   = float(sfc_patch[0, mid, mid])
    tcwv = float(sfc_patch[1, mid, mid])
    cape = float(sfc_patch[2, mid, mid])
    d2m  = float(sfc_patch[3, mid, mid])

    tp_mm = tp * 1000.0

    r850  = pl_data.get("r_850",  50.0)
    r500  = pl_data.get("r_500",  50.0)
    r200  = pl_data.get("r_200",  20.0)
    w850  = pl_data.get("w_850",   0.0)
    w500  = pl_data.get("w_500",   0.0)
    u850  = pl_data.get("u_850",   0.0)
    u200  = pl_data.get("u_200",   0.0)
    v850  = pl_data.get("v_850",   0.0)
    v200  = pl_data.get("v_200",   0.0)
    vo850 = pl_data.get("vo_850",  0.0)

    ws850      = np.sqrt(u850**2 + v850**2)
    ws200      = np.sqrt(u200**2 + v200**2)
    shear_mag  = np.sqrt((u200-u850)**2 + (v200-v850)**2)
    tp_log     = np.log1p(max(tp_mm, 0.0))
    cape_uplift = cape * max(-w850, 0.0) / 1000.0
    d2m_dev    = d2m - 295.0
    vort_rh    = vo850 * r850 * 1e4
    rh_diff    = r850 - r500
    cape_tcwv  = cape * tcwv / 1e5
    tp_cape    = tp_mm * cape / 1e3

    feats = np.array([
        tp_mm, tcwv, cape, d2m,
        r850, r500, r200,
        w850, w500,
        u850, u200, v850, v200,
        vo850,
        ws850, ws200, shear_mag,
        tp_log, cape_uplift, d2m_dev,
        vort_rh, rh_diff,
        cape_tcwv, tp_cape,
    ], dtype=np.float32)

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# DATA BUILDER — WITH CORRECT D-1 TEMPORAL ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

class RainfallDataBuilder:
    """
    Scans ecmwf_data/ year-by-year, aligns daily GRIB fields with ground truth
    using the D-1 lag: for GT date D, use ECMWF init date D-1.
    
    Only monsoon months (JJAS) are processed.
    """

    def __init__(self, window_size: int = 3):
        self.window = window_size
        self.half   = window_size // 2

        print(f"Loading ground truth from {config.GROUND_TRUTH} ...")
        self.gt = pd.read_csv(config.GROUND_TRUTH, parse_dates=["Date"])
        self.gt = self.gt.dropna(subset=["Rainfall_mm"])
        self.gt = self.gt[self.gt["Date"].dt.month.isin(config.MONSOON_MONTHS)]
        self.gt["year"] = self.gt["Date"].dt.year
        print(f"  -> {len(self.gt):,} valid ground-truth rows after monsoon filter")

    def build(self, years: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_patches, all_tabular, all_targets = [], [], []

        for year in years:
            year_dir = config.ECMWF_DIR / str(year)
            if not year_dir.exists():
                print(f"  [WARN] Year directory not found: {year_dir}")
                continue

            print(f"\nProcessing year {year} ...")
            p, t, y = self._process_year(year, year_dir)
            all_patches.extend(p)
            all_tabular.extend(t)
            all_targets.extend(y)
            print(f"  -> {len(y)} samples from {year}")

        if not all_patches:
            raise RuntimeError("No samples built! Check ECMWF directory and GRIB files.")

        patches = np.stack(all_patches, axis=0).astype(np.float32)
        tabular = np.stack(all_tabular, axis=0).astype(np.float32)
        targets = np.array(all_targets, dtype=np.float32)

        print(f"\nTotal samples: {len(targets):,}")
        print(f"  Patches shape : {patches.shape}")
        print(f"  Tabular shape : {tabular.shape}")
        print(f"  Rain>=0.1mm   : {(targets >= 0.1).sum()} ({100*(targets>=0.1).mean():.1f}%)")
        if (targets >= 0.1).sum() > 10:
            print(f"  P90 approx    : {np.percentile(targets[targets>=0.1], 90):.1f} mm")
            print(f"  P95 approx    : {np.percentile(targets[targets>=0.1], 95):.1f} mm")

        return patches, tabular, targets

    def _process_year(self, year: int, year_dir: Path):
        """Process all monsoon months in one year directory with D-1 alignment."""
        patches_y, tabular_y, targets_y = [], [], []

        # Load ALL GRIB files for this year, build a date-indexed cache
        sfc_files = sorted(year_dir.glob("*.sfc.grib"))
        pl_files  = sorted(year_dir.glob("*.pl.grib"))

        sfc_map = {_parse_month_from_filename(f.name): f for f in sfc_files}
        pl_map  = {_parse_month_from_filename(f.name): f for f in pl_files}

        # Also check previous year for May 31 -> June 1 boundary
        prev_year_dir = config.ECMWF_DIR / str(year - 1)

        # Build a date -> (sfc_ds, pl_ds, time_idx) lookup
        # We need months 5-9 of ECMWF data to cover JJAS GT with D-1 shift
        ecmwf_months_needed = set()
        for m in config.MONSOON_MONTHS:
            ecmwf_months_needed.add(m)       # same month
            ecmwf_months_needed.add(m - 1)   # previous month (for 1st of month)

        # Load GRIB datasets for needed months
        grib_cache = {}  # {date_obj: (sfc_ds_at_time, pl_ds_at_time)}
        
        for month in sorted(ecmwf_months_needed):
            if month < 1:
                continue
            
            # Determine which year-dir and file to use
            if month in sfc_map and month in pl_map:
                sfc_path, pl_path = sfc_map[month], pl_map[month]
            elif month == 5 and prev_year_dir.exists():
                # May might be in previous year's folder...skip if not available
                continue
            else:
                continue

            try:
                sfc_ds = _open_sfc_grib(sfc_path)
                pl_ds  = _open_pl_grib(pl_path)
            except Exception as e:
                print(f"    GRIB load error for month {month}: {e}")
                continue

            # Index by date
            if "time" in sfc_ds.dims and sfc_ds.dims["time"] > 1:
                time_vals = pd.DatetimeIndex(sfc_ds.time.values)
                for ti, t in enumerate(time_vals):
                    date_key = t.date()
                    grib_cache[date_key] = (sfc_ds, pl_ds, ti)
            else:
                # Single time step — use the time coordinate if available
                if "time" in sfc_ds.coords:
                    t = pd.Timestamp(sfc_ds.time.values)
                    grib_cache[t.date()] = (sfc_ds, pl_ds, None)

        if not grib_cache:
            print(f"    [WARN] No GRIB dates cached for year {year}")
            return [], [], []

        grib_dates = sorted(grib_cache.keys())
        print(f"    GRIB cache: {len(grib_dates)} dates "
              f"({grib_dates[0]} to {grib_dates[-1]})")

        # Now process GT rows with D-1 alignment
        for month in config.MONSOON_MONTHS:
            gt_month = self.gt[
                (self.gt["year"] == year) &
                (self.gt["Date"].dt.month == month)
            ]

            if len(gt_month) == 0:
                continue

            print(f"    Month {month:02d} ...", end=" ")

            count = 0
            skipped_no_grib = 0

            for _, row in gt_month.iterrows():
                gt_date = row["Date"].date()
                lat, lon = row["Lat"], row["Lon"]
                rain_mm  = float(row["Rainfall_mm"])

                # D-1 ALIGNMENT: For GT date D, use ECMWF init date D-1
                ecmwf_date = gt_date - timedelta(days=1)

                if ecmwf_date not in grib_cache:
                    skipped_no_grib += 1
                    continue

                sfc_ds, pl_ds, time_idx = grib_cache[ecmwf_date]

                # Select the correct time step from the GRIB
                if time_idx is not None:
                    sfc_slice = sfc_ds.isel(time=time_idx)
                    pl_slice  = pl_ds.isel(time=time_idx)
                else:
                    sfc_slice = sfc_ds
                    pl_slice  = pl_ds

                lat_arr = sfc_slice["latitude"].values
                lon_arr = sfc_slice["longitude"].values
                lat_idx, lon_idx = _lat_lon_to_grid_idx(lat, lon, lat_arr, lon_arr)

                # Build 19-channel patch
                patch = self._build_patch(sfc_slice, pl_slice, lat_idx, lon_idx)
                if patch is None:
                    continue

                # Build tabular features
                pl_scalars = self._pl_scalars(pl_slice, lat_idx, lon_idx)
                tab = derive_tabular_features(patch, pl_scalars)

                patches_y.append(patch)
                tabular_y.append(tab)
                targets_y.append(rain_mm)
                count += 1

            msg = f"{count} samples"
            if skipped_no_grib > 0:
                msg += f" (skipped {skipped_no_grib} — no D-1 GRIB)"
            print(msg)

        return patches_y, tabular_y, targets_y

    def _build_patch(self, sfc_ds, pl_ds, lat_idx, lon_idx):
        """Build (19, W, W) patch from a single-timestep GRIB slice."""
        channels = []

        # Surface channels (4)
        for var in config.SFC_VARS:
            if var not in sfc_ds:
                channels.append(np.zeros((self.window, self.window), dtype=np.float32))
                continue
            field = sfc_ds[var].values.astype(np.float32)
            if field.ndim > 2:
                field = field.squeeze()
            if field.ndim != 2:
                channels.append(np.zeros((self.window, self.window), dtype=np.float32))
                continue
            if var == "tp":
                field = field * 1000.0
            patch = _extract_patch(field, lat_idx, lon_idx, self.half)
            if patch is None:
                return None
            channels.append(patch)

        # Pressure-level channels (5 vars x 3 levels = 15)
        for var in config.PL_VARS:
            if var not in pl_ds:
                for _ in config.PRESSURE_LEVELS:
                    channels.append(np.zeros((self.window, self.window), dtype=np.float32))
                continue
            for level in config.PRESSURE_LEVELS:
                try:
                    field = pl_ds[var].sel(isobaricInhPa=level).values.astype(np.float32)
                    if field.ndim > 2:
                        field = field.squeeze()
                    if field.ndim != 2:
                        channels.append(np.zeros((self.window, self.window), dtype=np.float32))
                        continue
                    patch = _extract_patch(field, lat_idx, lon_idx, self.half)
                    if patch is None:
                        return None
                    channels.append(patch)
                except Exception:
                    channels.append(np.zeros((self.window, self.window), dtype=np.float32))

        if len(channels) == 0:
            return None

        return np.stack(channels, axis=0)

    def _pl_scalars(self, pl_ds, lat_idx, lon_idx):
        """Extract scalar values at station grid cell for tabular features."""
        scalars = {}
        for var in config.PL_VARS:
            if var not in pl_ds:
                continue
            for level in config.PRESSURE_LEVELS:
                try:
                    field = pl_ds[var].sel(isobaricInhPa=level).values
                    if field.ndim > 2:
                        field = field.squeeze()
                    scalars[f"{var}_{int(level)}"] = float(field[lat_idx, lon_idx])
                except Exception:
                    scalars[f"{var}_{int(level)}"] = 0.0
        return scalars


# ──────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ──────────────────────────────────────────────────────────────────────────────

class Normaliser:
    def __init__(self):
        self.patch_mean = None
        self.patch_std  = None
        self.tab_mean   = None
        self.tab_std    = None
        self.fitted     = False

    def fit(self, patches, tabular):
        self.patch_mean = patches.mean(axis=(0, 2, 3), keepdims=False)[:, None, None]
        self.patch_std  = patches.std(axis=(0, 2, 3),  keepdims=False)[:, None, None] + 1e-8
        self.tab_mean   = tabular.mean(axis=0)
        self.tab_std    = tabular.std(axis=0) + 1e-8
        self.fitted     = True

    def transform_patches(self, patches):
        assert self.fitted
        return (patches - self.patch_mean) / self.patch_std

    def transform_tabular(self, tabular):
        assert self.fitted
        return (tabular - self.tab_mean) / self.tab_std

    def save(self, path):
        np.savez(str(path),
                 patch_mean=self.patch_mean, patch_std=self.patch_std,
                 tab_mean=self.tab_mean, tab_std=self.tab_std)

    def load(self, path):
        d = np.load(str(path))
        self.patch_mean = d["patch_mean"]
        self.patch_std  = d["patch_std"]
        self.tab_mean   = d["tab_mean"]
        self.tab_std    = d["tab_std"]
        self.fitted     = True


# ──────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────────────

class RainfallDataset(Dataset):
    def __init__(self, patches, tabular, targets):
        self.patches = torch.from_numpy(patches).float()
        self.tabular = torch.from_numpy(tabular).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.patches[idx], self.tabular[idx], self.targets[idx]


# ──────────────────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER
# ──────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(targets):
    rainy = targets[targets >= config.DRY_THRESHOLD]
    p90 = np.percentile(rainy, 90)
    p95 = np.percentile(rainy, 95)
    p99 = np.percentile(rainy, 99)

    print(f"\nWeighted sampler thresholds (from training data):")
    print(f"  P90 = {p90:.2f} mm   (oversample x{config.OVERSAMPLE_P90})")
    print(f"  P95 = {p95:.2f} mm   (oversample x{config.OVERSAMPLE_P95})")
    print(f"  P99 = {p99:.2f} mm   (oversample x{config.OVERSAMPLE_P99})")

    weights = np.ones(len(targets), dtype=np.float32)
    weights[(targets >= config.DRY_THRESHOLD) & (targets < p90)] = config.OVERSAMPLE_RAIN
    weights[(targets >= p90) & (targets < p95)] = config.OVERSAMPLE_P90
    weights[(targets >= p95) & (targets < p99)] = config.OVERSAMPLE_P95
    weights[targets >= p99] = config.OVERSAMPLE_P99

    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(weights),
        replacement=True
    )
    return sampler, float(p90), float(p95), float(p99)


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL NOISE AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

# Which years to augment and how many noisy copies per original sample
AUGMENT_YEARS = [2018, 2019, 2020]
AUGMENT_COPIES = 3           # number of noisy copies per sample
PATCH_NOISE_SCALE = 0.03     # 3% of per-channel std
TABULAR_NOISE_SCALE = 0.03   # 3% of per-feature std


def temporal_augment(patches, tabular, targets, years_arr,
                     augment_years=None, n_copies=None,
                     patch_noise=None, tab_noise=None, seed=77):
    """
    Create noise-jittered copies of samples from specified years.

    Args:
        patches:  (N, C, H, W) float32
        tabular:  (N, F) float32
        targets:  (N,) float32
        years_arr: (N,) int — year label for each sample
        augment_years: list of years to augment (default: AUGMENT_YEARS)
        n_copies: number of noisy copies per sample (default: AUGMENT_COPIES)
        patch_noise: noise scale for patches (default: PATCH_NOISE_SCALE)
        tab_noise: noise scale for tabular (default: TABULAR_NOISE_SCALE)
        seed: random seed for reproducibility

    Returns:
        aug_patches, aug_tabular, aug_targets, aug_years — augmented arrays
        (original data + synthetic copies concatenated)
    """
    if augment_years is None:
        augment_years = AUGMENT_YEARS
    if n_copies is None:
        n_copies = AUGMENT_COPIES
    if patch_noise is None:
        patch_noise = PATCH_NOISE_SCALE
    if tab_noise is None:
        tab_noise = TABULAR_NOISE_SCALE

    rng = np.random.RandomState(seed)

    # Find indices of samples from target years
    mask = np.isin(years_arr, augment_years)
    src_idx = np.where(mask)[0]

    if len(src_idx) == 0:
        print(f"  [AUG] No samples found for years {augment_years}, skipping.")
        return patches, tabular, targets, years_arr

    # Compute per-channel std for patches and per-feature std for tabular
    # using ONLY the source samples to keep noise proportional to their scale
    p_std = patches[src_idx].std(axis=(0, 2, 3), keepdims=True)[:, :, np.newaxis, np.newaxis]
    # Reshape to (1, C, 1, 1) for broadcasting
    p_std = patches[src_idx].reshape(len(src_idx), patches.shape[1], -1).std(axis=(0, 2))
    p_std = p_std[np.newaxis, :, np.newaxis, np.newaxis]  # (1, C, 1, 1)

    t_std = tabular[src_idx].std(axis=0, keepdims=True)   # (1, F)

    new_patches, new_tabular, new_targets, new_years = [], [], [], []

    for _ in range(n_copies):
        # Add small Gaussian noise scaled to feature variability
        noisy_p = patches[src_idx] + rng.randn(*patches[src_idx].shape).astype(np.float32) * p_std * patch_noise
        noisy_t = tabular[src_idx] + rng.randn(*tabular[src_idx].shape).astype(np.float32) * t_std * tab_noise

        # Ensure tp (channel 0) stays non-negative
        noisy_p[:, 0] = np.maximum(noisy_p[:, 0], 0.0)

        new_patches.append(noisy_p)
        new_tabular.append(noisy_t)
        new_targets.append(targets[src_idx].copy())  # targets unchanged
        new_years.append(years_arr[src_idx].copy())

    # Concatenate original + augmented
    aug_patches = np.concatenate([patches] + new_patches, axis=0)
    aug_tabular = np.concatenate([tabular] + new_tabular, axis=0)
    aug_targets = np.concatenate([targets] + new_targets, axis=0)
    aug_years = np.concatenate([years_arr] + new_years, axis=0)

    n_new = len(src_idx) * n_copies
    print(f"\n  [AUG] Temporal noise augmentation:")
    print(f"    Source years: {augment_years}")
    print(f"    Source samples: {len(src_idx)}")
    print(f"    Copies per sample: {n_copies} (noise: patch={patch_noise:.1%}, tab={tab_noise:.1%})")
    print(f"    New synthetic samples: {n_new}")
    print(f"    Total after augmentation: {len(aug_targets)} ({len(targets)} original + {n_new} synthetic)")

    return aug_patches, aug_tabular, aug_targets, aug_years


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE LOADER
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(window_size: int = 3, use_augmentation: bool = True):
    builder = RainfallDataBuilder(window_size=window_size)

    # Build per-year for training so we can track year labels
    print("\n=== Building TRAIN data (per-year) ===")
    tr_patches_list, tr_tabular_list, tr_targets_list, tr_years_list = [], [], [], []
    for yr in config.TRAIN_YEARS:
        p, t, y = builder.build([yr])
        tr_patches_list.append(p)
        tr_tabular_list.append(t)
        tr_targets_list.append(y)
        tr_years_list.append(np.full(len(y), yr, dtype=np.int32))

    tr_patches = np.concatenate(tr_patches_list)
    tr_tabular = np.concatenate(tr_tabular_list)
    tr_targets = np.concatenate(tr_targets_list)
    tr_years = np.concatenate(tr_years_list)
    print(f"  Original train total: {len(tr_targets)} samples")

    # Apply temporal noise augmentation on training data
    if use_augmentation:
        tr_patches, tr_tabular, tr_targets, tr_years = temporal_augment(
            tr_patches, tr_tabular, tr_targets, tr_years)

    print("\n=== Building VAL data ===")
    vl_patches, vl_tabular, vl_targets = builder.build(config.VAL_YEARS)

    print("\n=== Building TEST data ===")
    te_patches, te_tabular, te_targets = builder.build(config.TEST_YEARS)

    # Fit normaliser on training data only (including augmented)
    norm = Normaliser()
    norm.fit(tr_patches, tr_tabular)

    out_dir = config.OUTPUT_DIR / f"window_{window_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    norm.save(out_dir / "normaliser.npz")

    # Normalise all splits
    tr_patches_n = norm.transform_patches(tr_patches)
    tr_tabular_n = norm.transform_tabular(tr_tabular)
    vl_patches_n = norm.transform_patches(vl_patches)
    vl_tabular_n = norm.transform_tabular(vl_tabular)
    te_patches_n = norm.transform_patches(te_patches)
    te_tabular_n = norm.transform_tabular(te_tabular)

    tr_ds = RainfallDataset(tr_patches_n, tr_tabular_n, tr_targets)
    vl_ds = RainfallDataset(vl_patches_n, vl_tabular_n, vl_targets)
    te_ds = RainfallDataset(te_patches_n, te_tabular_n, te_targets)

    sampler, p90, p95, p99 = build_weighted_sampler(tr_targets)

    train_loader = DataLoader(tr_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=config.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(vl_ds, batch_size=config.BATCH_SIZE * 2,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(te_ds, batch_size=config.BATCH_SIZE * 2,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)

    thresholds = {"p90": p90, "p95": p95, "p99": p99}

    print(f"\nDataLoader sizes:")
    print(f"  Train batches : {len(train_loader)} ({'with' if use_augmentation else 'without'} augmentation)")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, norm, thresholds
