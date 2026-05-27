"""
Figure 10: Percentage Improvement of Model over ECMWF
Nature-style horizontal bar chart
LOOCV 10-Year Mean, ALL Stations
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
metrics = [
    ("POD Rain",    -20.5,  0.773, 0.972),
    ("CSI Rain",    +15.2,  0.478, 0.415),
    ("SEDI Rain",   +52.5,  0.517, 0.339),
    ("Correlation", +83.3,  0.449, 0.245),
    ("CSI P90",    +164.6,  0.217, 0.082),
    ("SEDI P90",   +160.8,  0.592, 0.227),
    ("CSI P95",    +269.6,  0.207, 0.056),
    ("SEDI P95",   +300.0,  0.473, 0.006),   # capped at 300 % (actual 7783 %)
]

# Sort by magnitude (ascending so largest bar is on top)
metrics.sort(key=lambda x: x[1])

labels       = [m[0] for m in metrics]
improvements = [m[1] for m in metrics]
model_vals   = [m[2] for m in metrics]
ecmwf_vals   = [m[3] for m in metrics]

# ── Colours ───────────────────────────────────────────────────────────────────
colors = ['#c0392b' if v < 0 else '#27ae60' for v in improvements]

# ── Figure setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
})

fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, improvements, height=0.62, color=colors,
               edgecolor='white', linewidth=0.5, zorder=3)

# ── Zero line ─────────────────────────────────────────────────────────────────
ax.axvline(0, color='#2c3e50', linewidth=1.0, zorder=4)

# ── Bar-end labels ────────────────────────────────────────────────────────────
for i, (val, model_v, ecmwf_v, metric_name) in enumerate(
        zip(improvements, model_vals, ecmwf_vals, labels)):
    # Determine sign symbol
    sign = '+' if val > 0 else ''
    # Special annotation for SEDI P95 (capped)
    if metric_name == "SEDI P95":
        disp_text = f"{sign}{val:.0f}%  (actual +7783%)"
    else:
        disp_text = f"{sign}{val:.1f}%"

    detail_text = f"Model {model_v:.3f} vs ECMWF {ecmwf_v:.3f}"

    # Place percentage label at bar end
    offset = 4 if val >= 0 else -4
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, i, disp_text,
            va='center', ha=ha, fontsize=10,
            color=colors[i], zorder=5)

    # Place detail text (model vs ecmwf) inside the bar or opposite side
    if abs(val) > 80:
        # Inside bar
        inner_offset = -6 if val >= 0 else 6
        inner_ha = 'right' if val >= 0 else 'left'
        ax.text(val + inner_offset, i, detail_text,
                va='center', ha=inner_ha, fontsize=10,
                color='white', fontstyle='italic', zorder=5)
    else:
        # Below the bar label
        ax.text(val + offset, i - 0.28, detail_text,
                va='center', ha=ha, fontsize=10,
                color='#555555', fontstyle='italic', zorder=5)

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Improvement over ECMWF  (%)', fontsize=10, labelpad=10)
ax.set_xlim(-60, 370)

# Light grid on x only
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.grid(axis='x', which='major', linestyle='--', linewidth=0.4,
        color='#b0b0b0', zorder=1)
ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.25,
        color='#d0d0d0', zorder=1)

# Bold box - all spines visible
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(2.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    'Percentage Improvement of Hybrid Model over ECMWF\n'
    'LOOCV 10-Year Mean  ·  All Stations',
    fontsize=16, pad=16, linespacing=1.45
)

# ── Legend patch ──────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', edgecolor='none', label='Model outperforms ECMWF'),
    Patch(facecolor='#c0392b', edgecolor='none', label='ECMWF outperforms Model'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          framealpha=0.9, edgecolor='#cccccc', fontsize=11,
          handlelength=1.2, handleheight=1.0)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = r'D:\NEW_NRSC\paper_figures\Fig10_Improvement_Summary.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Saved → {out_path}")
