"""
metrics.py
==========
Evaluation metrics for extreme precipitation downscaling.

Metrics implemented:
  - Critical Success Index (CSI)        — primary metric
  - Probability of Detection (POD)      — recall
  - False Alarm Ratio (FAR)             — precision complement
  - Symmetric Extremal Dependence Index (SEDI)
  - Frequency Bias (FBI)
  - Heidke Skill Score (HSS)
  - RMSE and MAE (continuous)
  - Pearson r on rainy days

All categorical metrics computed for three thresholds:
  "rain"  : ≥ 0.1 mm
  "p90"   : ≥ P90 (computed from data)
  "p95"   : ≥ P95
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# CONTINGENCY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def contingency(pred_mm: np.ndarray, obs_mm: np.ndarray, threshold: float) -> Dict:
    """
    Compute a 2×2 contingency table for the given rainfall threshold.

    Returns dict with keys: hits, misses, false_alarms, correct_negatives
    and their rates (H, M, FA, CN).
    """
    pred_event = pred_mm >= threshold
    obs_event  = obs_mm  >= threshold

    hits      = int(( pred_event &  obs_event).sum())
    misses    = int((~pred_event &  obs_event).sum())
    false_alarms = int(( pred_event & ~obs_event).sum())
    correct_neg  = int((~pred_event & ~obs_event).sum())

    total = hits + misses + false_alarms + correct_neg

    return {
        "H":   hits,
        "M":   misses,
        "FA":  false_alarms,
        "CN":  correct_neg,
        "n":   total,
        "n_obs":  hits + misses,
        "n_pred": hits + false_alarms,
    }


# ──────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ──────────────────────────────────────────────────────────────────────────────

def csi(ct: Dict) -> float:
    """Critical Success Index = H / (H + M + FA). Range [0,1], 1=perfect."""
    denom = ct["H"] + ct["M"] + ct["FA"]
    return ct["H"] / denom if denom > 0 else 0.0


def pod(ct: Dict) -> float:
    """Probability of Detection = H / (H + M). Recall of events."""
    denom = ct["H"] + ct["M"]
    return ct["H"] / denom if denom > 0 else 0.0


def far(ct: Dict) -> float:
    """False Alarm Ratio = FA / (H + FA). Fraction of alarms that were wrong."""
    denom = ct["H"] + ct["FA"]
    return ct["FA"] / denom if denom > 0 else 0.0


def fbi(ct: Dict) -> float:
    """Frequency Bias = (H+FA)/(H+M). 1=unbiased, >1=over-forecast."""
    denom = ct["H"] + ct["M"]
    return (ct["H"] + ct["FA"]) / denom if denom > 0 else 0.0


def hss(ct: Dict) -> float:
    """Heidke Skill Score. 0=no skill, 1=perfect, negative=worse than random."""
    n   = ct["n"]
    H   = ct["H"]
    M   = ct["M"]
    FA  = ct["FA"]
    CN  = ct["CN"]
    num  = 2.0 * (H * CN - M * FA)
    denom = ((H + M) * (M + CN) + (H + FA) * (FA + CN))
    return num / denom if denom > 0 else 0.0


def sedi(ct: Dict) -> float:
    """
    Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).
    Range [-1, 1], 1=perfect. Robust for rare events.
    SEDI = [log(F) - log(H_rate) - log(1-F) + log(1-H_rate)] /
           [log(F) + log(H_rate) + log(1-F) + log(1-H_rate)]

    where H_rate = H/(H+M)  and F = FA/(FA+CN)
    """
    eps = 1e-7
    H   = ct["H"]
    M   = ct["M"]
    FA  = ct["FA"]
    CN  = ct["CN"]

    hit_rate  = H  / (H  + M  + eps)
    false_rate = FA / (FA + CN + eps)

    # Clamp to avoid log(0)
    hit_rate   = np.clip(hit_rate,   eps, 1 - eps)
    false_rate = np.clip(false_rate, eps, 1 - eps)

    log_F    = np.log(false_rate)
    log_H    = np.log(hit_rate)
    log_1mF  = np.log(1 - false_rate)
    log_1mH  = np.log(1 - hit_rate)

    num   = log_F - log_H - log_1mF + log_1mH
    denom = log_F + log_H + log_1mF + log_1mH

    return float(num / denom) if abs(denom) > eps else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CONTINUOUS METRICS
# ──────────────────────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def corr_rainy(pred: np.ndarray, obs: np.ndarray, threshold: float = 0.1) -> float:
    mask = obs >= threshold
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(pred[mask], obs[mask])[0, 1])


# ──────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    pred_mm:    np.ndarray,
    obs_mm:     np.ndarray,
    thresholds: Dict,       # {"p90": float, "p95": float, "p99": float}
    prefix:     str = ""
) -> Dict:
    """
    Compute all metrics for all categories.

    Returns a flat dict: {prefix_metric_category: value}
    """
    results = {}
    p = prefix + "_" if prefix else ""

    rain_thresh = 0.1
    p90 = thresholds["p90"]
    p95 = thresholds["p95"]

    for label, thresh in [("rain", rain_thresh), ("p90", p90), ("p95", p95)]:
        ct = contingency(pred_mm, obs_mm, thresh)
        results[f"{p}CSI_{label}"]  = round(csi(ct),  4)
        results[f"{p}POD_{label}"]  = round(pod(ct),  4)
        results[f"{p}FAR_{label}"]  = round(far(ct),  4)
        results[f"{p}FBI_{label}"]  = round(fbi(ct),  4)
        results[f"{p}HSS_{label}"]  = round(hss(ct),  4)
        results[f"{p}SEDI_{label}"] = round(sedi(ct), 4)

        # Print contingency counts for debugging
        results[f"{p}H_{label}"]   = ct["H"]
        results[f"{p}M_{label}"]   = ct["M"]
        results[f"{p}FA_{label}"]  = ct["FA"]
        results[f"{p}n_obs_{label}"] = ct["n_obs"]

    # Continuous
    results[f"{p}RMSE"]       = round(rmse(pred_mm, obs_mm), 4)
    results[f"{p}MAE"]        = round(mae(pred_mm,  obs_mm), 4)
    results[f"{p}corr_rainy"] = round(corr_rainy(pred_mm, obs_mm), 4)

    return results


def print_metrics(results: Dict, title: str = "Evaluation"):
    """Pretty-print a metrics dict."""
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print(f"{'-'*50}")

    categories = ["rain", "p90", "p95"]
    metrics    = ["CSI", "POD", "FAR", "SEDI", "HSS"]

    # Detect prefix
    prefix = ""
    for k in results:
        if "CSI_rain" in k:
            prefix = k.replace("CSI_rain", "")
            break

    for cat in categories:
        n_obs = results.get(f"{prefix}n_obs_{cat}", "?")
        print(f"\n  [{cat.upper()}]  (n_obs = {n_obs})")
        for m in metrics:
            val = results.get(f"{prefix}{m}_{cat}", "N/A")
            # Mark if target is met
            flag = ""
            if m == "CSI"  and isinstance(val, float) and val > 0.50: flag = " OK"
            if m == "SEDI" and isinstance(val, float) and val > 0.60: flag = " OK"
            if m == "FAR"  and isinstance(val, float) and val < 0.30: flag = " OK"
            print(f"    {m:6s}: {val:.4f}{flag}")

    print(f"\n  [CONTINUOUS]")
    for k in ["RMSE", "MAE", "corr_rainy"]:
        val = results.get(f"{prefix}{k}", "N/A")
        print(f"    {k:12s}: {val}")
    print(f"{'-'*50}\n")


def check_targets(results: Dict, prefix: str = "") -> bool:
    """Returns True if ALL target thresholds from config are met."""
    import config
    p = prefix + "_" if prefix else ""
    checks = [
        results.get(f"{p}CSI_rain", 0) > config.TARGET_CSI_RAIN,
        results.get(f"{p}CSI_p90",  0) > config.TARGET_CSI_P90,
        results.get(f"{p}CSI_p95",  0) > config.TARGET_CSI_P95,
        results.get(f"{p}SEDI_rain", 0) > config.TARGET_SEDI,
        results.get(f"{p}SEDI_p90",  0) > config.TARGET_SEDI,
        results.get(f"{p}SEDI_p95",  0) > config.TARGET_SEDI,
    ]
    return all(checks)


# ──────────────────────────────────────────────────────────────────────────────
# COLLECT PREDICTIONS FROM DATALOADER
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Run model on a DataLoader and return (preds_mm, targets_mm)."""
    model.eval()
    preds_list   = []
    targets_list = []

    for batch in loader:
        patch, tabular, target = batch
        patch   = patch.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)

        pred = model.predict(patch, tabular)
        preds_list.append(pred.cpu().numpy())
        targets_list.append(target.numpy())

    return np.concatenate(preds_list), np.concatenate(targets_list)
