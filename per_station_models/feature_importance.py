"""
feature_importance_loyo.py
===========================
Leave-One-Year-Out (LOYO) feature importance analysis.

Loads ALL data once, then for each eval year, evaluates which channels
consistently help vs hurt. Only features positive across ALL years are kept.

This prevents overfitting to any single test year.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(r"D:\NEW_NRSC\final_model_baseline\9x9")))
import config
import dataset
import model as model_module
import metrics as metrics_module

ROOT_DIR = Path(r"D:\NEW_NRSC")

CHANNEL_NAMES = [
    "tp (sfc)", "tcwv (sfc)", "cape (sfc)", "d2m (sfc)",
    "r_850", "r_500", "r_200",
    "w_850", "w_500", "w_200",
    "vo_850", "vo_500", "vo_200",
    "u_850", "u_500", "u_200",
    "v_850", "v_500", "v_200",
]

TABULAR_NAMES = [
    "tp_mm", "tcwv", "cape", "d2m",
    "r_850", "r_500", "r_200",
    "w_850", "w_500",
    "u_850", "u_200", "v_850", "v_200",
    "vo_850",
    "ws_850", "ws_200", "shear_mag",
    "tp_log", "cape_uplift", "d2m_dev",
    "vort_rh", "rh_diff",
    "cape_tcwv", "tp_cape",
]


def compute_score(preds, targets, thresholds):
    m = metrics_module.evaluate(preds, targets, thresholds, prefix="x")
    csi_rain = m.get("x_CSI_rain", 0)
    csi_p90 = m.get("x_CSI_p90", 0)
    sedi_p90 = m.get("x_SEDI_p90", 0)
    corr = m.get("x_corr_rainy", 0)
    if np.isnan(corr): corr = 0
    return csi_rain + csi_p90 + sedi_p90 + max(corr, 0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    ckpt_dir = ROOT_DIR / "final_model_baseline" / "9x9"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
    best_ckpt = ckpts[-1]
    print(f"Checkpoint: {best_ckpt.name}")

    ckpt = torch.load(str(best_ckpt), map_location=device)
    net = model_module.build_model(window_size=9, n_channels=19, n_tabular=24)
    net.load_state_dict(ckpt["model"])
    net = net.to(device)
    net.eval()

    # ── Load ALL data ONCE (2015-2024), track year boundaries ──
    print("\nLoading ALL years (2015-2024) in one pass...")
    builder = dataset.RainfallDataBuilder(window_size=9)

    year_data = {}
    for year in range(2015, 2025):
        print(f"\n--- Year {year} ---")
        try:
            p, t, y = builder.build([year])
            year_data[year] = (p, t, y)
            print(f"  -> {len(y)} samples loaded")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print(f"\nAll data loaded. Years available: {sorted(year_data.keys())}")

    # ── Eval years: we test on 2022, 2023, 2024 ──
    eval_years = [2022, 2023, 2024]
    n_repeats = 3

    ch_importance_by_year = {}
    tab_importance_by_year = {}

    for test_year in eval_years:
        if test_year not in year_data:
            print(f"\nSkipping {test_year} (no data)")
            continue

        print(f"\n{'='*60}")
        print(f"  EVALUATING TEST YEAR {test_year}")
        print(f"{'='*60}")

        # Split: train = all except test_year
        train_patches_list, train_tab_list, train_target_list = [], [], []
        for y in sorted(year_data.keys()):
            if y == test_year:
                continue
            p, t, tgt = year_data[y]
            train_patches_list.append(p)
            train_tab_list.append(t)
            train_target_list.append(tgt)

        train_patches = np.concatenate(train_patches_list, axis=0)
        train_tabular = np.concatenate(train_tab_list, axis=0)
        train_targets = np.concatenate(train_target_list, axis=0)

        test_patches, test_tabular, test_targets = year_data[test_year]

        # Compute thresholds from train
        rainy = train_targets[train_targets >= 0.1]
        thresholds = {
            "p90": float(np.percentile(rainy, 90)),
            "p95": float(np.percentile(rainy, 95)),
            "p99": float(np.percentile(rainy, 99)),
        }
        print(f"  Thresholds: P90={thresholds['p90']:.1f}mm P95={thresholds['p95']:.1f}mm")
        print(f"  Train: {len(train_targets)} samples, Test: {len(test_targets)} samples")

        # Normalise on train
        norm = dataset.Normaliser()
        norm.fit(train_patches, train_tabular)
        patches_n = norm.transform_patches(test_patches)
        tabular_n = norm.transform_tabular(test_tabular)

        patches_t = torch.from_numpy(patches_n).float().to(device)
        tabular_t = torch.from_numpy(tabular_n).float().to(device)

        # Baseline
        with torch.no_grad():
            base_preds = net.predict(patches_t, tabular_t).cpu().numpy()
        base_score = compute_score(base_preds, test_targets, thresholds)
        print(f"  Baseline score: {base_score:.4f}")

        # ── Channel importance ──
        print(f"  Computing channel importance (19 channels x {n_repeats} repeats)...")
        ch_imp = {}
        for ch_idx in range(19):
            drops = []
            for _ in range(n_repeats):
                pp = patches_n.copy()
                perm = np.random.permutation(len(pp))
                pp[:, ch_idx, :, :] = pp[perm, ch_idx, :, :]
                pt = torch.from_numpy(pp).float().to(device)
                with torch.no_grad():
                    p = net.predict(pt, tabular_t).cpu().numpy()
                drops.append(base_score - compute_score(p, test_targets, thresholds))
            ch_imp[CHANNEL_NAMES[ch_idx]] = float(np.mean(drops))
        ch_importance_by_year[test_year] = ch_imp

        # ── Tabular importance ──
        print(f"  Computing tabular importance (24 features x {n_repeats} repeats)...")
        tab_imp = {}
        for fi in range(24):
            drops = []
            for _ in range(n_repeats):
                tp = tabular_n.copy()
                perm = np.random.permutation(len(tp))
                tp[:, fi] = tp[perm, fi]
                tt = torch.from_numpy(tp).float().to(device)
                with torch.no_grad():
                    p = net.predict(patches_t, tt).cpu().numpy()
                drops.append(base_score - compute_score(p, test_targets, thresholds))
            name = TABULAR_NAMES[fi] if fi < len(TABULAR_NAMES) else f"feat_{fi}"
            tab_imp[name] = float(np.mean(drops))
        tab_importance_by_year[test_year] = tab_imp

        # Print year summary
        sorted_ch = sorted(ch_imp.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 channels for {test_year}:")
        for name, drop in sorted_ch[:5]:
            print(f"    {name:15s}: {drop:+.4f}")
        print(f"  Bottom 3 (hurting):")
        for name, drop in sorted_ch[-3:]:
            print(f"    {name:15s}: {drop:+.4f}")

    # ── AGGREGATE across all eval years ──
    print(f"\n{'='*70}")
    print(f"  AGGREGATE CHANNEL IMPORTANCE (averaged over years {eval_years})")
    print(f"{'='*70}")

    avg_ch = defaultdict(list)
    for year, imp in ch_importance_by_year.items():
        for name, drop in imp.items():
            avg_ch[name].append(drop)

    avg_ch_final = {name: (np.mean(drops), np.std(drops), np.min(drops))
                    for name, drops in avg_ch.items()}
    sorted_avg_ch = sorted(avg_ch_final.items(), key=lambda x: x[1][0], reverse=True)

    print(f"\n{'Rank':>4s} {'Channel':15s} {'Mean Drop':>10s} {'Std':>8s} {'Min':>8s} {'Consistent?':>12s} {'Verdict':>15s}")
    print("-" * 80)
    for rank, (name, (mean, std, mn)) in enumerate(sorted_avg_ch, 1):
        all_positive = mn > 0
        consistent = "YES" if all_positive else "NO"

        if mean > 0.01 and all_positive:
            verdict = "KEEP"
        elif mean > 0.01 and not all_positive:
            verdict = "KEEP *unstable*"
        elif mean > -0.01:
            verdict = "NEUTRAL"
        else:
            verdict = "REMOVE"

        print(f"{rank:4d} {name:15s} {mean:+10.4f} {std:8.4f} {mn:+8.4f} {consistent:>12s} {verdict:>15s}")

    # ── AGGREGATE TABULAR ──
    print(f"\n{'='*70}")
    print(f"  AGGREGATE TABULAR IMPORTANCE (averaged over years {eval_years})")
    print(f"{'='*70}")

    avg_tab = defaultdict(list)
    for year, imp in tab_importance_by_year.items():
        for name, drop in imp.items():
            avg_tab[name].append(drop)

    avg_tab_final = {name: (np.mean(drops), np.std(drops), np.min(drops))
                     for name, drops in avg_tab.items()}
    sorted_avg_tab = sorted(avg_tab_final.items(), key=lambda x: x[1][0], reverse=True)

    print(f"\n{'Rank':>4s} {'Feature':15s} {'Mean Drop':>10s} {'Std':>8s} {'Min':>8s} {'Consistent?':>12s} {'Verdict':>15s}")
    print("-" * 80)
    for rank, (name, (mean, std, mn)) in enumerate(sorted_avg_tab, 1):
        all_positive = mn > 0
        consistent = "YES" if all_positive else "NO"
        if mean > 0.005 and all_positive:
            verdict = "KEEP"
        elif mean > -0.005:
            verdict = "NEUTRAL"
        else:
            verdict = "REMOVE"
        print(f"{rank:4d} {name:15s} {mean:+10.4f} {std:8.4f} {mn:+8.4f} {consistent:>12s} {verdict:>15s}")

    # ── Per-year detail table ──
    print(f"\n{'='*70}")
    print(f"  PER-YEAR CHANNEL BREAKDOWN")
    print(f"{'='*70}")
    print(f"\n{'Channel':15s}", end="")
    for y in eval_years:
        print(f" | {y:>8d}", end="")
    print(f" | {'Mean':>8s}")
    print("-" * (15 + 11 * (len(eval_years) + 1)))
    for name, (mean, _, _) in sorted_avg_ch:
        print(f"{name:15s}", end="")
        for y in eval_years:
            val = ch_importance_by_year.get(y, {}).get(name, 0)
            print(f" | {val:+8.4f}", end="")
        print(f" | {mean:+8.4f}")

    # Save
    results = {
        "eval_years": eval_years,
        "channel_importance_by_year": {str(y): d for y, d in ch_importance_by_year.items()},
        "tabular_importance_by_year": {str(y): d for y, d in tab_importance_by_year.items()},
        "avg_channel_importance": {n: {"mean": float(m), "std": float(s), "min": float(mn)}
                                    for n, (m, s, mn) in sorted_avg_ch},
        "avg_tabular_importance": {n: {"mean": float(m), "std": float(s), "min": float(mn)}
                                    for n, (m, s, mn) in sorted_avg_tab},
    }
    out_path = ROOT_DIR / "per_station_models" / "outputs" / "feature_importance_loyo.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
