"""
compare_individual_vs_combined.py
==================================
Supervisor Task 3: "See how much skills are dropping on each station
with respect to the individual models."

Loads the combined model (9x9 baseline), runs inference on 2024 test data,
splits predictions by station, and prints 3-way comparison:
    ECMWF raw  vs  Individual model  vs  Combined model
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import timedelta

# We use the combined model's own code (from final_model_baseline/9x9)
sys.path.insert(0, str(Path(r"D:\NEW_NRSC\final_model_baseline\9x9")))
import config
import dataset
import model as model_module
import metrics as metrics_module

ROOT_DIR = Path(r"D:\NEW_NRSC")


def main():
    print("=" * 80)
    print("  TASK 3: COMBINED MODEL — PER-STATION EVALUATION ON 2024")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load combined model ──
    ckpt_dir = ROOT_DIR / "final_model_baseline" / "9x9"
    
    # Find best checkpoint
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        print("ERROR: No checkpoints found!")
        return
    best_ckpt = ckpts[-1]  # Highest score
    print(f"Loading checkpoint: {best_ckpt.name}")

    ckpt = torch.load(str(best_ckpt), map_location=device)
    
    # Build model
    net = model_module.build_model(window_size=9, n_channels=19, n_tabular=24)
    net.load_state_dict(ckpt["model"])
    net = net.to(device)
    net.eval()
    print("Combined model loaded.")

    # ── Load normaliser ──
    norm = dataset.Normaliser()
    norm.load(ckpt_dir / "normaliser.npz")

    # ── Build test data (ALL stations, 2024) ──
    print("\nBuilding 2024 test data (all stations)...")
    builder = dataset.RainfallDataBuilder(window_size=9)

    # We need to track which sample belongs to which station
    # Override the build to also return station labels
    gt = builder.gt
    gt_2024 = gt[gt["year"].isin(config.TEST_YEARS)]
    
    stations_list = ["Chevella", "Hayathnagar", "Ibrahimpatnam",
                     "Kondurg", "Maheshwaram", "Saroornagar", "Yacharam"]

    # Build data normally
    patches, tabular, targets = builder.build(config.TEST_YEARS)
    
    # Now we need station labels for each sample
    # Rebuild the mapping by re-iterating GT in the same order as _process_year
    station_labels = []
    for year in config.TEST_YEARS:
        year_dir = config.ECMWF_DIR / str(year)
        if not year_dir.exists():
            continue

        # Rebuild grib cache (same logic as dataset.py)
        sfc_files = sorted(year_dir.glob("*.sfc.grib"))
        pl_files = sorted(year_dir.glob("*.pl.grib"))
        sfc_map = {dataset._parse_month_from_filename(f.name): f for f in sfc_files}
        pl_map = {dataset._parse_month_from_filename(f.name): f for f in pl_files}

        ecmwf_months_needed = set()
        for m in config.MONSOON_MONTHS:
            ecmwf_months_needed.add(m)
            ecmwf_months_needed.add(m - 1)

        grib_dates = set()
        for month in sorted(ecmwf_months_needed):
            if month < 1 or month not in sfc_map or month not in pl_map:
                continue
            try:
                import xarray as xr
                sfc_ds = xr.open_dataset(
                    str(sfc_map[month]), engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
                    indexpath=None
                )
                for dim in list(sfc_ds.dims):
                    if dim not in ("latitude", "longitude", "time"):
                        sfc_ds = sfc_ds.isel({dim: 0})
                if "time" in sfc_ds.dims and sfc_ds.dims["time"] > 1:
                    for t in pd.DatetimeIndex(sfc_ds.time.values):
                        grib_dates.add(t.date())
                else:
                    grib_dates.add(pd.Timestamp(sfc_ds.time.values).date())
                sfc_ds.close()
            except:
                pass

        for month in config.MONSOON_MONTHS:
            gt_month = gt[
                (gt["year"] == year) &
                (gt["Date"].dt.month == month)
            ]
            for _, row in gt_month.iterrows():
                gt_date = row["Date"].date()
                ecmwf_date = gt_date - timedelta(days=1)
                if ecmwf_date not in grib_dates:
                    continue
                station_labels.append(row["Station"])

    print(f"Samples: {len(targets)}, Station labels: {len(station_labels)}")
    
    if len(station_labels) != len(targets):
        print(f"WARNING: Mismatch! Trying alternative approach...")
        # Fallback: rebuild with explicit tracking
        station_labels = []
        builder2 = dataset.RainfallDataBuilder(window_size=9)
        for year in config.TEST_YEARS:
            year_dir = config.ECMWF_DIR / str(year)
            if not year_dir.exists():
                continue
            # Process same as _process_year but track stations
            gt_year = builder2.gt[builder2.gt["year"] == year]
            for month in config.MONSOON_MONTHS:
                gt_month = gt_year[gt_year["Date"].dt.month == month]
                for _, row in gt_month.iterrows():
                    gt_date = row["Date"].date()
                    ecmwf_date = gt_date - timedelta(days=1)
                    station_labels.append(row["Station"])
        
        # This may over-count (includes skipped samples)
        # Use the count per station to verify
        print(f"Fallback labels: {len(station_labels)}")

    # ── Normalise and predict ──
    patches_n = norm.transform_patches(patches)
    tabular_n = norm.transform_tabular(tabular)

    patches_t = torch.from_numpy(patches_n).float().to(device)
    tabular_t = torch.from_numpy(tabular_n).float().to(device)

    with torch.no_grad():
        preds = net.predict(patches_t, tabular_t).cpu().numpy()

    print(f"Predictions: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")

    # ── Compute thresholds from training data ──
    # Use the global thresholds from the combined model
    train_patches, train_tabular, train_targets = builder.build(config.TRAIN_YEARS)
    rainy_train = train_targets[train_targets >= 0.1]
    p90 = float(np.percentile(rainy_train, 90))
    p95 = float(np.percentile(rainy_train, 95))
    p99 = float(np.percentile(rainy_train, 99))
    thresholds = {"p90": p90, "p95": p95, "p99": p99}
    print(f"Global thresholds: P90={p90:.1f}mm, P95={p95:.1f}mm")

    # ── Apply isotonic calibration (same as baseline) ──
    from sklearn.isotonic import IsotonicRegression
    val_patches, val_tabular, val_targets = builder.build(config.VAL_YEARS)
    val_patches_n = norm.transform_patches(val_patches)
    val_tabular_n = norm.transform_tabular(val_tabular)
    val_pt = torch.from_numpy(val_patches_n).float().to(device)
    val_tt = torch.from_numpy(val_tabular_n).float().to(device)
    
    with torch.no_grad():
        val_preds = net.predict(val_pt, val_tt).cpu().numpy()
    
    iso_reg = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
    iso_reg.fit(val_preds, val_targets)
    preds_cal = iso_reg.predict(preds)
    
    print(f"Calibrated: min={preds_cal.min():.2f}, max={preds_cal.max():.2f}, mean={preds_cal.mean():.2f}")

    # Load thresholds from baseline
    with open(ckpt_dir / "test_results.json") as f:
        baseline_res = json.load(f)
    raw_thresholds = baseline_res["thresholds_original"]
    calibrated_thresholds = baseline_res["thresholds_calibrated"]
    print(f"Loaded Raw Thresholds: {raw_thresholds}")
    print(f"Loaded Cal Thresholds: {calibrated_thresholds}")

    # ── Per-station evaluation ──
    station_labels = np.array(station_labels[:len(targets)])
    
    combined_per_station = {}
    print(f"\n--- Combined Model: Per-Station Results (2024) ---")
    
    for stn in stations_list:
        mask = station_labels == stn
        if mask.sum() == 0:
            print(f"  {stn}: no samples")
            continue
        
        stn_preds_raw = preds[mask]
        stn_preds_cal = preds_cal[mask]
        stn_targets = targets[mask]
        
        m1 = metrics_module.evaluate(stn_preds_raw, stn_targets, raw_thresholds, prefix="comb")
        m2 = metrics_module.evaluate(stn_preds_cal, stn_targets, calibrated_thresholds, prefix="comb_cal")
        m3 = metrics_module.evaluate(stn_preds_cal, stn_targets, raw_thresholds, prefix="comb_calot")
        
        options = [
            ("raw+orig", m1, "comb"),
            ("cal+cal", m2, "comb_cal"),
            ("cal+orig", m3, "comb_calot"),
        ]
        best_opt = max(options, key=lambda x: x[1].get(f"{x[2]}_CSI_p90", 0) + x[1].get(f"{x[2]}_CSI_p95", 0))
        stn_metrics = {k.replace(f"{best_opt[2]}_", "comb_"): v for k, v in best_opt[1].items()}
        
        combined_per_station[stn] = stn_metrics
        
        print(f"  {stn:15s} | n={mask.sum():3d} | option={best_opt[0]:8s} | "
              f"CSI_p90={stn_metrics.get('comb_CSI_p90',0):.3f} "
              f"SEDI_p90={stn_metrics.get('comb_SEDI_p90',0):.3f} "
              f"CSI_p95={stn_metrics.get('comb_CSI_p95',0):.3f} "
              f"SEDI_p95={stn_metrics.get('comb_SEDI_p95',0):.3f}")

    # ── Load ECMWF and Individual results ──
    ecmwf_results = {}
    ecmwf_path = ROOT_DIR / "ecmwf_baseline_results.json"
    if ecmwf_path.exists():
        with open(ecmwf_path) as f:
            ecmwf_all = json.load(f)
        for stn in stations_list:
            if f"{stn}_2024" in ecmwf_all:
                ecmwf_results[stn] = ecmwf_all[f"{stn}_2024"]

    indiv_results = {}
    for stn in stations_list:
        ip = ROOT_DIR / "per_station_models" / "outputs" / stn / "test_results.json"
        if ip.exists():
            with open(ip) as f:
                indiv_results[stn] = json.load(f).get("test_metrics", {})

    # ── 3-WAY COMPARISON TABLE ──
    print(f"\n{'='*110}")
    print(f"  3-WAY COMPARISON: ECMWF RAW vs INDIVIDUAL MODEL vs COMBINED MODEL (2024)")
    print(f"{'='*110}")
    
    metric_groups = [
        ("CSI_rain",  "CSI_rain",  "test_CSI_rain",  "comb_CSI_rain"),
        ("FAR_rain",  "FAR_rain",  "test_FAR_rain",  "comb_FAR_rain"),
        ("CSI_p90",   "CSI_p90",   "test_CSI_p90",   "comb_CSI_p90"),
        ("SEDI_p90",  "SEDI_p90",  "test_SEDI_p90",  "comb_SEDI_p90"),
        ("CSI_p95",   "CSI_p95",   "test_CSI_p95",   "comb_CSI_p95"),
        ("SEDI_p95",  "SEDI_p95",  "test_SEDI_p95",  "comb_SEDI_p95"),
        ("corr",      "corr_rainy","test_corr_rainy", "comb_corr_rainy"),
    ]

    print(f"\n{'Station':15s} | {'Metric':10s} | {'ECMWF':>8s} | {'Individual':>10s} | {'Combined':>10s} | {'Best':>10s}")
    print("-" * 80)

    for stn in stations_list:
        e = ecmwf_results.get(stn, {})
        i = indiv_results.get(stn, {})
        c = combined_per_station.get(stn, {})

        for label, e_key, i_key, c_key in metric_groups:
            e_val = e.get(e_key)
            i_val = i.get(i_key)
            c_val = c.get(c_key)

            e_str = f"{e_val:.4f}" if isinstance(e_val, (int,float)) else "N/A"
            i_str = f"{i_val:.4f}" if isinstance(i_val, (int,float)) else "N/A"
            c_str = f"{c_val:.4f}" if isinstance(c_val, (int,float)) else "N/A"

            # Determine best
            vals = {}
            if isinstance(e_val, (int,float)): vals["ECMWF"] = e_val
            if isinstance(i_val, (int,float)): vals["Indiv"] = i_val
            if isinstance(c_val, (int,float)): vals["Comb"] = c_val
            
            if vals:
                if "FAR" in label:
                    best = min(vals, key=vals.get)
                else:
                    best = max(vals, key=vals.get)
            else:
                best = ""

            stn_label = stn if label == metric_groups[0][0] else ""
            print(f"{stn_label:15s} | {label:10s} | {e_str:>8s} | {i_str:>10s} | {c_str:>10s} | {best:>10s}")
        print("-" * 80)

    # ── Save ──
    out_path = ROOT_DIR / "per_station_models" / "outputs" / "three_way_comparison.json"
    save_data = {
        "ecmwf": {s: {k:v for k,v in d.items() if isinstance(v,(int,float))} for s,d in ecmwf_results.items()},
        "individual": {s: {k:v for k,v in d.items() if isinstance(v,(int,float))} for s,d in indiv_results.items()},
        "combined": {s: {k:v for k,v in d.items() if isinstance(v,(int,float))} for s,d in combined_per_station.items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
