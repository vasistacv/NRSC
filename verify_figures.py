"""Verify ALL hardcoded values in figure scripts match the actual JSON results."""
import json, sys

with open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json") as f:
    d = json.load(f)

m = d["loyo_model_mean"]["ALL"]
e = d["loyo_ecmwf_mean"]["ALL"]

errors = []
warnings = []

def check(script, desc, expected, actual, tol=0.002):
    if abs(expected - actual) > tol:
        errors.append(f"  ERROR {script}: {desc} = {expected} but JSON has {actual:.4f} (diff={expected-actual:+.4f})")
    elif abs(expected - actual) > 0.0005:
        warnings.append(f"  WARN  {script}: {desc} = {expected} vs JSON {actual:.4f} (minor rounding)")

print("="*80)
print("  VERIFICATION: Hardcoded values vs station_final_results.json")
print("="*80)

# ============================================================
# fig2_performance_bars.py — LOYO Mean
# ============================================================
print("\n--- fig2_performance_bars.py ---")
# model_rain = [CSI, POD, SEDI]
check("fig2", "model_rain CSI",  0.488, m["CSI_rain"])
check("fig2", "model_rain POD",  0.790, m["POD_rain"])
check("fig2", "model_rain SEDI", 0.545, m["SEDI_rain"])
# model_p90
check("fig2", "model_p90 CSI",  0.330, m["CSI_p90"])
check("fig2", "model_p90 POD",  0.607, m["POD_p90"])
check("fig2", "model_p90 SEDI", 0.692, m["SEDI_p90"])
# model_p95
check("fig2", "model_p95 CSI",  0.221, m["CSI_p95"])
check("fig2", "model_p95 POD",  0.581, m["POD_p95"])
check("fig2", "model_p95 SEDI", 0.577, m["SEDI_p95"])
# ecmwf (unchanged)
check("fig2", "ecmwf_rain CSI", 0.415, e["CSI_rain"])
check("fig2", "ecmwf_rain POD", 0.972, e["POD_rain"])
check("fig2", "ecmwf_rain SEDI",0.339, e["SEDI_rain"])
check("fig2", "ecmwf_p90 CSI",  0.082, e["CSI_p90"])
check("fig2", "ecmwf_p90 POD",  0.104, e["POD_p90"])
check("fig2", "ecmwf_p90 SEDI", 0.227, e["SEDI_p90"])
check("fig2", "ecmwf_p95 CSI",  0.056, e["CSI_p95"])
check("fig2", "ecmwf_p95 POD",  0.069, e["POD_p95"])
check("fig2", "ecmwf_p95 SEDI", 0.006, e["SEDI_p95"])
print("  Checked 18 values")

# ============================================================
# fig4_radar_chart.py — LOYO Mean
# ============================================================
print("\n--- fig4_radar_chart.py ---")
# model_vals = [CSI Rain, POD Rain, SEDI Rain, CSI P90, CSI P95, Correlation]
check("fig4", "CSI Rain",    0.488, m["CSI_rain"])
check("fig4", "POD Rain",    0.790, m["POD_rain"])
check("fig4", "SEDI Rain",   0.545, m["SEDI_rain"])
check("fig4", "CSI P90",     0.330, m["CSI_p90"])
check("fig4", "CSI P95",     0.221, m["CSI_p95"])
check("fig4", "Correlation", 0.543, m["corr_rainy"])
print("  Checked 6 values")

# ============================================================
# fig5_split_comparison.py — Per-split CSI
# ============================================================
print("\n--- fig5_split_comparison.py ---")
splits_model = {
    "temporal": d["temporal"]["ALL"]["model"],
    "reverse":  d["reverse"]["ALL"]["model"],
    "random":   d["random"]["ALL"]["model"],
}
# Model CSI P90: [Temporal, Reverse, Random, LOYO]
check("fig5", "Temporal CSI_p90", 0.119, splits_model["temporal"]["CSI_p90"])
check("fig5", "Reverse CSI_p90",  0.497, splits_model["reverse"]["CSI_p90"])
check("fig5", "Random CSI_p90",   0.417, splits_model["random"]["CSI_p90"])
check("fig5", "LOYO CSI_p90",     0.330, m["CSI_p90"])
# Model CSI P95
check("fig5", "Temporal CSI_p95", 0.102, splits_model["temporal"]["CSI_p95"])
check("fig5", "Reverse CSI_p95",  0.3835, splits_model["reverse"]["CSI_p95"])
check("fig5", "Random CSI_p95",   0.319, splits_model["random"]["CSI_p95"])
check("fig5", "LOYO CSI_p95",     0.221, m["CSI_p95"])
print("  Checked 8 values")

# ============================================================
# fig9_table_figure.py — Per-split all metrics
# ============================================================
print("\n--- fig9_table_figure.py ---")
# model = {split: [CSI_r, SEDI_r, CSI_90, CSI_95, RMSE, Corr]}
fig9_data = {
    "Temporal": [0.4956, 0.508, 0.1189, 0.1022, 15.975, 0.255],
    "Reverse":  [0.4669, 0.556, 0.4972, 0.3835, 13.832, 0.785],
    "Random":   [0.5055, 0.596, 0.4167, 0.3191, 12.129, 0.733],
    "LOOCV":    [0.488, 0.545, 0.330, 0.221, 14.142, 0.543],
}
json_map = {"Temporal": "temporal", "Reverse": "reverse", "Random": "random"}
keys = ["CSI_rain", "SEDI_rain", "CSI_p90", "CSI_p95", "RMSE", "corr_rainy"]
for split_name, vals in fig9_data.items():
    if split_name == "LOOCV":
        src = m
    else:
        src = d[json_map[split_name]]["ALL"]["model"]
    for i, k in enumerate(keys):
        check("fig9", f"{split_name} {k}", vals[i], src[k], tol=0.5 if k=="RMSE" else 0.002)
print("  Checked 24 values")

# ============================================================
# fig3_station_heatmap.py — Per-station CSI
# ============================================================
print("\n--- fig3_station_heatmap.py ---")
stations = ["Chevella", "Hayathnagar", "Ibrahimpatnam", "Kondurg", "Maheshwaram", "Saroornagar", "Yacharam"]
fig3_model = [
    [0.505, 0.336, 0.140],
    [0.467, 0.373, 0.282],
    [0.469, 0.391, 0.179],
    [0.513, 0.263, 0.108],
    [0.528, 0.317, 0.237],
    [0.492, 0.296, 0.269],
    [0.436, 0.253, 0.182],
]
for i, stn in enumerate(stations):
    sm = d["loyo_model_mean"][stn]
    check("fig3", f"{stn} CSI_rain", fig3_model[i][0], sm["CSI_rain"])
    check("fig3", f"{stn} CSI_p90",  fig3_model[i][1], sm["CSI_p90"])
    check("fig3", f"{stn} CSI_p95",  fig3_model[i][2], sm["CSI_p95"])
print("  Checked 21 values")

# ============================================================
# fig7_boxplot.py — Per-station SEDI/CSI
# ============================================================
print("\n--- fig7_boxplot.py ---")
fig7_csi_p90 = [0.336, 0.373, 0.391, 0.263, 0.317, 0.296, 0.253]
fig7_csi_p95 = [0.140, 0.282, 0.179, 0.108, 0.237, 0.269, 0.182]
fig7_sedi_p90 = [0.491, 0.649, 0.658, 0.358, 0.639, 0.661, 0.468]
fig7_sedi_p95 = [0.125, 0.275, -0.001, -0.015, 0.322, 0.274, 0.019]
for i, stn in enumerate(stations):
    sm = d["loyo_model_mean"][stn]
    check("fig7", f"{stn} CSI_p90",  fig7_csi_p90[i],  sm["CSI_p90"])
    check("fig7", f"{stn} CSI_p95",  fig7_csi_p95[i],  sm["CSI_p95"])
    check("fig7", f"{stn} SEDI_p90", fig7_sedi_p90[i], sm["SEDI_p90"])
    check("fig7", f"{stn} SEDI_p95", fig7_sedi_p95[i], sm["SEDI_p95"])
print("  Checked 28 values")

# ============================================================
# fig6_spatial_map.py — Per-station CSI P95
# ============================================================
print("\n--- fig6_spatial_map.py ---")
fig6_model = {"Chevella": 0.140, "Hayathnagar": 0.282, "Ibrahimpatnam": 0.179,
              "Kondurg": 0.108, "Maheshwaram": 0.237, "Saroornagar": 0.269, "Yacharam": 0.182}
for stn, val in fig6_model.items():
    sm = d["loyo_model_mean"][stn]
    check("fig6", f"{stn} CSI_p95", val, sm["CSI_p95"])
print("  Checked 7 values")

# ============================================================
# fig10_improvement.py — % improvement
# ============================================================
print("\n--- fig10_improvement.py ---")
fig10 = [
    ("POD Rain",   0.790, m["POD_rain"]),
    ("CSI Rain",   0.488, m["CSI_rain"]),
    ("SEDI Rain",  0.545, m["SEDI_rain"]),
    ("Correlation",0.543, m["corr_rainy"]),
    ("CSI P90",    0.330, m["CSI_p90"]),
    ("SEDI P90",   0.692, m["SEDI_p90"]),
    ("CSI P95",    0.221, m["CSI_p95"]),
    ("SEDI P95",   0.577, m["SEDI_p95"]),
]
for name, hardcoded, actual in fig10:
    check("fig10", f"{name} model_val", hardcoded, actual)
print("  Checked 8 values")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}")
total = 18+6+8+24+21+28+7+8
print(f"  TOTAL VALUES CHECKED: {total}")
if errors:
    print(f"\n  ERRORS ({len(errors)}):")
    for e in errors:
        print(e)
else:
    print(f"\n  ✅ NO ERRORS — ALL {total} VALUES MATCH!")
if warnings:
    print(f"\n  WARNINGS ({len(warnings)}) — minor rounding:")
    for w in warnings:
        print(w)
print(f"{'='*80}")
