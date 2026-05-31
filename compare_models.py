"""Compare all 3 models side by side."""
print("LOYO MEAN COMPARISON - ALL STATIONS")
print("="*90)
hdr = f"{'Metric':<15} | {'19ch (old)':>12} | {'7ch Run1':>12} | {'7ch Run2':>12} | {'ECMWF':>12}"
print(hdr)
print("-"*90)
data = [
    ("CSI Rain",    0.478, 0.490, 0.488, 0.415),
    ("POD Rain",    0.773, 0.800, 0.790, 0.972),
    ("FAR Rain",    0.443, 0.441, 0.438, 0.580),
    ("CSI P90",     0.217, 0.258, 0.330, 0.082),
    ("POD P90",     0.475, 0.634, 0.607, 0.104),
    ("FAR P90",     0.717, 0.709, 0.608, 0.649),
    ("SEDI P90",    0.592, 0.697, 0.692, 0.227),
    ("CSI P95",     0.207, 0.199, 0.221, 0.056),
    ("POD P95",     0.492, 0.597, 0.581, 0.069),
    ("SEDI P95",    0.473, 0.678, 0.577, 0.006),
    ("RMSE",        14.3,  16.2,  14.1,  12.9),
    ("MAE",         7.2,   7.8,   6.8,   6.3),
    ("Correlation", 0.449, 0.501, 0.543, 0.245),
]
for name, v19, v7a, v7b, ecmwf in data:
    low_better = "FAR" in name or "RMSE" in name or "MAE" in name
    best = min(v19, v7a, v7b) if low_better else max(v19, v7a, v7b)
    m19 = " <-BEST" if v19 == best else ""
    m7a = " <-BEST" if v7a == best else ""
    m7b = " <-BEST" if v7b == best else ""
    print(f"  {name:<13} | {v19:>10.3f}{m19:>8} | {v7a:>10.3f}{m7a:>8} | {v7b:>10.3f}{m7b:>8} | {ecmwf:>10.3f}")

print()
print("7ch Run2 vs 19ch improvement:")
imp = [
    ("CSI P90",     0.217, 0.330),
    ("FAR P90",     0.717, 0.608),
    ("SEDI P90",    0.592, 0.692),
    ("CSI P95",     0.207, 0.221),
    ("SEDI P95",    0.473, 0.577),
    ("RMSE",        14.3,  14.1),
    ("Correlation", 0.449, 0.543),
]
for name, old, new in imp:
    pct = (new - old) / abs(old) * 100
    arrow = "better" if ("FAR" in name or "RMSE" in name) == (pct < 0) else ("better" if pct > 0 else "worse")
    print(f"  {name:<13}: {old:.3f} -> {new:.3f}  ({pct:+.1f}% {arrow})")
