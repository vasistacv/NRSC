"""Full comparison: 19ch (old) vs 7ch Run1 vs 7ch Run2 (current) vs ECMWF — ALL metrics, ALL splits."""
import json, subprocess, sys

# Load current 7ch Run2 results
with open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json") as f:
    run2 = json.load(f)

# Load old 19ch results from git main
old_json = subprocess.check_output(
    ["git", "show", "main:final_model_baseline/9x9_v3/final_ensemble/station_final_results.json"],
    cwd=r"D:\NEW_NRSC", stderr=subprocess.DEVNULL
).decode("utf-8")
old19 = json.loads(old_json)

# 7ch Run1 numbers (from the first 7ch evaluation — manually recorded)
run1_loyo = {
    "CSI_rain": 0.490, "POD_rain": 0.800, "FAR_rain": 0.441, "SEDI_rain": 0.545,
    "CSI_p90": 0.258, "POD_p90": 0.634, "FAR_p90": 0.709, "SEDI_p90": 0.697,
    "CSI_p95": 0.199, "POD_p95": 0.597, "FAR_p95": 0.779, "SEDI_p95": 0.678,
    "RMSE": 16.203, "MAE": 7.772, "corr_rainy": 0.501
}
run1_temporal = {
    "CSI_rain": 0.506, "POD_rain": 0.836, "FAR_rain": 0.438, "SEDI_rain": 0.535,
    "CSI_p90": 0.137, "POD_p90": 0.361, "FAR_p90": 0.820, "SEDI_p90": 0.470,
    "CSI_p95": 0.093, "POD_p95": 0.268, "FAR_p95": 0.876, "SEDI_p95": 0.440,
    "RMSE": 19.946, "MAE": 9.290, "corr_rainy": 0.226
}
run1_reverse = {
    "CSI_rain": 0.465, "POD_rain": 0.798, "FAR_rain": 0.473, "SEDI_rain": 0.549,
    "CSI_p90": 0.343, "POD_p90": 0.915, "FAR_p90": 0.646, "SEDI_p90": 0.942,
    "CSI_p95": 0.320, "POD_p95": 0.923, "FAR_p95": 0.671, "SEDI_p95": 0.959,
    "RMSE": 14.923, "MAE": 7.042, "corr_rainy": 0.710
}
run1_random = {
    "CSI_rain": 0.510, "POD_rain": 0.845, "FAR_rain": 0.437, "SEDI_rain": 0.605,
    "CSI_p90": 0.308, "POD_p90": 0.717, "FAR_p90": 0.649, "SEDI_p90": 0.835,
    "CSI_p95": 0.258, "POD_p95": 0.727, "FAR_p95": 0.714, "SEDI_p95": 0.862,
    "RMSE": 14.011, "MAE": 6.648, "corr_rainy": 0.696
}

MK = ["CSI_rain","POD_rain","FAR_rain","SEDI_rain",
      "CSI_p90","POD_p90","FAR_p90","SEDI_p90",
      "CSI_p95","POD_p95","FAR_p95","SEDI_p95",
      "RMSE","MAE","corr_rainy"]

def print_split(label, m19, e19, m_r1, m_r2, e_r2):
    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")
    print(f"  {'Metric':<15} | {'19ch Model':>12} | {'19ch ECMWF':>12} | {'7ch Run1':>12} | {'7ch Run2':>12} | {'ECMWF Run2':>12} | {'Best':>8}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for k in MK:
        v19 = m19.get(k, 0.0)
        ve19 = e19.get(k, 0.0)
        vr1 = m_r1.get(k, 0.0)
        vr2 = m_r2.get(k, 0.0)
        ver2 = e_r2.get(k, 0.0)
        
        low_better = "FAR" in k or k in ("RMSE", "MAE")
        models = [("19ch", v19), ("7chR1", vr1), ("7chR2", vr2)]
        if low_better:
            best_name = min(models, key=lambda x: x[1])[0]
        else:
            best_name = max(models, key=lambda x: x[1])[0]
        
        marker = f"  {best_name}"
        
        fmt = ".1f" if k in ("RMSE","MAE") else ".3f"
        print(f"  {k:<15} | {v19:>12.{fmt[1:]}}"
              f" | {ve19:>12.{fmt[1:]}}"
              f" | {vr1:>12.{fmt[1:]}}"
              f" | {vr2:>12.{fmt[1:]}}"
              f" | {ver2:>12.{fmt[1:]}}"
              f" | {marker:>8}")

# A) Temporal
print_split(
    "A) TEMPORAL (Train 2015-2020 -> Test 2021-2024)",
    old19["temporal"]["ALL"]["model"],
    old19["temporal"]["ALL"]["ecmwf"],
    run1_temporal,
    run2["temporal"]["ALL"]["model"],
    run2["temporal"]["ALL"]["ecmwf"]
)

# B) Reverse
print_split(
    "B) REVERSE (Train 2018-2024 -> Test 2015-2017)",
    old19["reverse"]["ALL"]["model"],
    old19["reverse"]["ALL"]["ecmwf"],
    run1_reverse,
    run2["reverse"]["ALL"]["model"],
    run2["reverse"]["ALL"]["ecmwf"]
)

# C) Random
print_split(
    "C) RANDOM (70/15/15 Mixed)",
    old19["random"]["ALL"]["model"],
    old19["random"]["ALL"]["ecmwf"],
    run1_random,
    run2["random"]["ALL"]["model"],
    run2["random"]["ALL"]["ecmwf"]
)

# D) LOYO Mean
print_split(
    "D) LOYO MEAN (10-fold Leave-One-Year-Out)",
    old19["loyo_model_mean"]["ALL"],
    old19["loyo_ecmwf_mean"]["ALL"],
    run1_loyo,
    run2["loyo_model_mean"]["ALL"],
    run2["loyo_ecmwf_mean"]["ALL"]
)

# Summary table
print(f"\n\n{'='*120}")
print("  SUMMARY: Win count per model (across all 4 splits x 15 metrics = 60 comparisons)")
print(f"{'='*120}")

wins = {"19ch": 0, "7chR1": 0, "7chR2": 0}

splits = [
    ("Temporal", old19["temporal"]["ALL"]["model"], run1_temporal, run2["temporal"]["ALL"]["model"]),
    ("Reverse",  old19["reverse"]["ALL"]["model"],  run1_reverse,  run2["reverse"]["ALL"]["model"]),
    ("Random",   old19["random"]["ALL"]["model"],   run1_random,   run2["random"]["ALL"]["model"]),
    ("LOYO",     old19["loyo_model_mean"]["ALL"],   run1_loyo,     run2["loyo_model_mean"]["ALL"]),
]

split_wins = {}
for sname, m19, mr1, mr2 in splits:
    sw = {"19ch": 0, "7chR1": 0, "7chR2": 0}
    for k in MK:
        v19 = m19.get(k, 0.0)
        vr1 = mr1.get(k, 0.0)
        vr2 = mr2.get(k, 0.0)
        low_better = "FAR" in k or k in ("RMSE", "MAE")
        models = [("19ch", v19), ("7chR1", vr1), ("7chR2", vr2)]
        if low_better:
            best_name = min(models, key=lambda x: x[1])[0]
        else:
            best_name = max(models, key=lambda x: x[1])[0]
        wins[best_name] += 1
        sw[best_name] += 1
    split_wins[sname] = sw

print(f"\n  {'Split':<12} | {'19ch':>6} | {'7ch R1':>8} | {'7ch R2':>8} | Total")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+------")
for sname in ["Temporal", "Reverse", "Random", "LOYO"]:
    sw = split_wins[sname]
    total = sum(sw.values())
    print(f"  {sname:<12} | {sw['19ch']:>6} | {sw['7chR1']:>8} | {sw['7chR2']:>8} | {total:>5}")

print(f"  {'TOTAL':<12} | {wins['19ch']:>6} | {wins['7chR1']:>8} | {wins['7chR2']:>8} | {sum(wins.values()):>5}")
print(f"\n  WINNER: {'7ch Run2' if wins['7chR2'] >= max(wins.values()) else '7ch Run1' if wins['7chR1'] >= max(wins.values()) else '19ch'}")

# vs ECMWF improvement for 7ch Run2 LOYO
print(f"\n\n{'='*120}")
print("  7ch Run2 (BEST MODEL) vs ECMWF RAW — LOYO Mean Improvement")
print(f"{'='*120}")
m = run2["loyo_model_mean"]["ALL"]
e = run2["loyo_ecmwf_mean"]["ALL"]
for k in MK:
    vm = m.get(k, 0.0)
    ve = e.get(k, 0.0)
    if ve != 0:
        pct = (vm - ve) / abs(ve) * 100
    else:
        pct = float('inf') if vm > 0 else 0
    low_better = "FAR" in k or k in ("RMSE", "MAE")
    if low_better:
        direction = "better" if pct < 0 else "worse"
    else:
        direction = "better" if pct > 0 else "worse"
    pct_str = f"{pct:+.1f}%" if abs(pct) < 10000 else "MASSIVE"
    print(f"  {k:<15}: Model={vm:>8.3f}  ECMWF={ve:>8.3f}  ({pct_str} {direction})")
