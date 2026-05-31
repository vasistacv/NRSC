"""
Figure 8: Year-wise LOYO Performance — CSI Rain + CSI P90
Two panels showing Model vs ECMWF across 10 years (2015–2024).
"""

import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import defaultdict

# ── Load data ────────────────────────────────────────────────────────────────
with open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json") as f:
    results = json.load(f)

years = list(range(2015, 2025))

# ── Model & ECMWF per-year CSI from loyo_per_year ───────────────────────────
model_csi_rain = []
model_csi_p90 = []
ecmwf_csi_rain = []
ecmwf_csi_p90 = []

for y in years:
    yd = results["loyo_per_year"].get(str(y), {})
    m = yd.get("ALL", {}).get("model", {})
    e = yd.get("ALL", {}).get("ecmwf", {})
    model_csi_rain.append(m.get("CSI_rain", 0.0))
    model_csi_p90.append(m.get("CSI_p90", 0.0))
    ecmwf_csi_rain.append(e.get("CSI_rain", 0.0))
    ecmwf_csi_p90.append(e.get("CSI_p90", 0.0))

# ── GFS per-year CSI ─────────────────────────────────────────────────────────
with open(r"D:\NEW_NRSC\gfs_baseline_results.json") as f:
    gfs_raw = json.load(f)

gfs_yearly_rain = defaultdict(lambda: {"H": 0, "M": 0, "FA": 0})
gfs_yearly_p90  = defaultdict(lambda: {"H": 0, "M": 0, "FA": 0})

for key, data in gfs_raw.items():
    parts = key.rsplit("_", 1)
    if len(parts) != 2:
        continue
    try:
        year = int(parts[1])
    except ValueError:
        continue
    if year not in years:
        continue

    gfs_yearly_rain[year]["H"]  += data.get("H_rain", 0)
    gfs_yearly_rain[year]["M"]  += data.get("M_rain", 0)
    gfs_yearly_rain[year]["FA"] += data.get("FA_rain", 0)

    gfs_yearly_p90[year]["H"]  += data.get("H_p90", 0)
    gfs_yearly_p90[year]["M"]  += data.get("M_p90", 0)
    gfs_yearly_p90[year]["FA"] += data.get("FA_p90", 0)

def compute_csi(h, m, fa):
    denom = h + m + fa
    return h / denom if denom > 0 else 0.0

gfs_csi_rain = [compute_csi(gfs_yearly_rain[y]["H"], gfs_yearly_rain[y]["M"], gfs_yearly_rain[y]["FA"]) for y in years]
gfs_csi_p90  = [compute_csi(gfs_yearly_p90[y]["H"], gfs_yearly_p90[y]["M"], gfs_yearly_p90[y]["FA"]) for y in years]

model_overall_rain = results["loyo_model_mean"]["ALL"]["CSI_rain"]
model_overall_p90  = results["loyo_model_mean"]["ALL"]["CSI_p90"]

print("Model CSI_rain per year:", [f"{v:.3f}" for v in model_csi_rain])
print("ECMWF CSI_rain per year:", [f"{v:.3f}" for v in ecmwf_csi_rain])
print("Model CSI_p90 per year:", [f"{v:.3f}" for v in model_csi_p90])
print("ECMWF CSI_p90 per year:", [f"{v:.3f}" for v in ecmwf_csi_p90])

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLOR_MODEL = "#3575B2"
COLOR_ECMWF = "#E8873D"
COLOR_GFS   = "#2E7D32"

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

panels = [
    ("(a) CSI Rain \u2013 Year-wise LOYO", model_csi_rain, ecmwf_csi_rain, gfs_csi_rain, model_overall_rain),
    ("(b) CSI P90 \u2013 Year-wise LOYO",  model_csi_p90,  ecmwf_csi_p90,  gfs_csi_p90,  model_overall_p90),
]

x = np.array(years)

for ax, (title, m_vals, e_vals, g_vals, m_mean) in zip(axes, panels):
    m_vals = np.array(m_vals)
    e_vals = np.array(e_vals)
    g_vals = np.array(g_vals)

    ax.plot(x, m_vals, color=COLOR_MODEL, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.0,
            linewidth=2.0, label="Model", zorder=4)

    ax.plot(x, e_vals, color=COLOR_ECMWF, marker="s", markersize=7,
            markeredgecolor="white", markeredgewidth=1.0,
            linewidth=2.0, label="ECMWF (9 km)", zorder=4)

    ax.plot(x, g_vals, color=COLOR_GFS, marker="^", markersize=7,
            markeredgecolor="white", markeredgewidth=1.0,
            linewidth=2.0, label="GFS (25 km)", zorder=4)

    ax.fill_between(x, m_vals, e_vals,
                    where=(m_vals >= e_vals),
                    color=COLOR_MODEL, alpha=0.15,
                    interpolate=True, zorder=2)
    ax.fill_between(x, m_vals, e_vals,
                    where=(m_vals < e_vals),
                    color=COLOR_ECMWF, alpha=0.15,
                    interpolate=True, zorder=2)

    e_mean = np.mean(e_vals)
    g_mean = np.mean(g_vals)
    ax.axhline(m_mean, color=COLOR_MODEL, linestyle="--", linewidth=1.2,
               alpha=0.6, zorder=3, label=f"Model mean ({m_mean:.3f})")
    ax.axhline(e_mean, color=COLOR_ECMWF, linestyle="--", linewidth=1.2,
               alpha=0.6, zorder=3, label=f"ECMWF mean ({e_mean:.3f})")
    ax.axhline(g_mean, color=COLOR_GFS, linestyle="-.", linewidth=1.2,
               alpha=0.6, zorder=3, label=f"GFS mean ({g_mean:.3f})")

    ax.set_xlabel("Year")
    ax.set_ylabel("CSI")
    ax.set_title(title, pad=10)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")

    all_vals = np.concatenate([m_vals, e_vals, g_vals])
    y_max = max(all_vals) * 1.3 if max(all_vals) > 0 else 0.05
    ax.set_ylim(-0.005, y_max)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5,
                  color="#cccccc", zorder=0)
    ax.xaxis.grid(True, which="major", linestyle=":", linewidth=0.4,
                  color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.5)

    ax.legend(frameon=True, framealpha=0.9, edgecolor="#bbbbbb",
              fontsize=9, loc="upper left")

fig.tight_layout(pad=2.5)

out_path = r"D:\NEW_NRSC\paper_figures\Fig8_LOYO_Yearly.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved -> {out_path}")
