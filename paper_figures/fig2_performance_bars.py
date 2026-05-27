"""
Fig 2 – Performance Comparison: Model vs ECMWF
Grouped bar chart (1×3 sub-panels) for Nature-style publication.
Data source: LOOCV 10-Year Average, ALL stations.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
metrics = ["CSI", "POD", "SEDI"]

model_rain  = [0.478, 0.773, 0.517]
ecmwf_rain  = [0.415, 0.972, 0.339]

model_p90   = [0.217, 0.475, 0.592]
ecmwf_p90   = [0.082, 0.104, 0.227]

model_p95   = [0.207, 0.492, 0.473]
ecmwf_p95   = [0.056, 0.069, 0.006]

panels = [
    {"title": "Rain Detection",  "model": model_rain, "ecmwf": ecmwf_rain, "label": "a"},
    {"title": "P90 Extreme",     "model": model_p90,  "ecmwf": ecmwf_p90,  "label": "b"},
    {"title": "P95 Severe",      "model": model_p95,  "ecmwf": ecmwf_p95,  "label": "c"},
]

# ── Colours & style ────────────────────────────────────────────────────────
COLOR_MODEL = "#1565C0"
COLOR_ECMWF = "#E65100"
HATCH_ECMWF = "///"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize":   11,
    "ytick.labelsize":   10,
    "legend.fontsize":   11,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.subplots_adjust(wspace=0.08)

bar_width = 0.32
x = np.arange(len(metrics))

for ax, panel in zip(axes, panels):
    m_vals = panel["model"]
    e_vals = panel["ecmwf"]

    bars_m = ax.bar(
        x - bar_width / 2, m_vals, bar_width,
        color=COLOR_MODEL, edgecolor="black", linewidth=0.6,
        label="Model", zorder=3,
    )
    bars_e = ax.bar(
        x + bar_width / 2, e_vals, bar_width,
        color=COLOR_ECMWF, edgecolor="black", linewidth=0.6,
        hatch=HATCH_ECMWF, label="ECMWF", zorder=3,
    )

    # Value labels
    for bar in bars_m:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.015,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=10, color=COLOR_MODEL,
        )
    for bar in bars_e:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.015,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=10, color=COLOR_ECMWF,
        )

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    # Gridlines
    ax.yaxis.grid(True, which="major", linestyle="-",  linewidth=0.4, color="#CCCCCC", zorder=0)
    ax.yaxis.grid(True, which="minor", linestyle=":",  linewidth=0.3, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)

    # Panel label & title
    ax.set_title(f"({panel['label']})  {panel['title']}", pad=10)

    # Bold box - all spines visible
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.5)

# Y-axis label only on first panel
axes[0].set_ylabel("Score")

# ── Shared legend ───────────────────────────────────────────────────────────
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center", ncol=2,
    frameon=True, edgecolor="black", fancybox=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=10,
)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = r"D:\NEW_NRSC\paper_figures\Fig2_Performance_Comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] Saved: {out_path}")
