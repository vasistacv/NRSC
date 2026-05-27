"""
Figure 5: Split Comparison – Grouped Bar Chart
CSI at P90 and P95 thresholds across 4 evaluation splits.
Nature-style publication figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
splits = ["Temporal", "Reverse\nTemporal", "Random", "LOOCV"]

# (Model, ECMWF, GFS) per split
csi_p90 = {
    "Model":  [0.128, 0.267, 0.260, 0.217],
    "ECMWF":  [0.131, 0.057, 0.145, 0.082],
    "GFS":    [0.083, 0.083, 0.083, 0.083],
}
csi_p95 = {
    "Model":  [0.101, 0.279, 0.250, 0.207],
    "ECMWF":  [0.087, 0.015, 0.037, 0.056],
    "GFS":    [0.059, 0.059, 0.059, 0.059],
}

panels = [
    ("(a) CSI P90 – All Stations", csi_p90),
    ("(b) CSI P95 – All Stations", csi_p95),
]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize":   10.5,
    "ytick.labelsize":   10,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

COLOR_MODEL = "#3575B2"   # steel blue
COLOR_ECMWF = "#E8873D"   # warm orange
COLOR_GFS   = "#2E7D32"   # forest green
BAR_WIDTH    = 0.25
EDGE_LW      = 0.7

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

x = np.arange(len(splits))

for ax, (title, data) in zip(axes, panels):
    model_vals = data["Model"]
    ecmwf_vals = data["ECMWF"]
    gfs_vals   = data["GFS"]

    bars_m = ax.bar(
        x - BAR_WIDTH, model_vals, BAR_WIDTH,
        color=COLOR_MODEL, edgecolor="black", linewidth=EDGE_LW,
        label="Model", zorder=3,
    )
    bars_e = ax.bar(
        x, ecmwf_vals, BAR_WIDTH,
        color=COLOR_ECMWF, edgecolor="black", linewidth=EDGE_LW,
        label="ECMWF (9 km)", hatch="///", zorder=3,
    )
    bars_g = ax.bar(
        x + BAR_WIDTH, gfs_vals, BAR_WIDTH,
        color=COLOR_GFS, edgecolor="black", linewidth=EDGE_LW,
        label="GFS (25 km)", hatch="\\\\\\", zorder=3,
    )

    # Value labels
    for bar in bars_m:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.006,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9, color="#222222",
        )
    for bar in bars_e:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.006,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9, color="#222222",
        )
    for bar in bars_g:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.006,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=9, color="#222222",
        )

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("CSI")
    ax.set_title(title, pad=10)
    ax.set_ylim(0, max(max(model_vals), max(ecmwf_vals), max(gfs_vals)) * 1.25)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))

    # Light horizontal grid behind bars
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5,
                  color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Bold box - all spines visible
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.5)

    # Legend
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#bbbbbb",
              fontsize=11, loc="upper right")

fig.tight_layout(pad=2.5)

out_path = r"D:\NEW_NRSC\paper_figures\Fig5_Split_Comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {out_path}")
