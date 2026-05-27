"""
Figure 10: Percentage Improvement of Model over ECMWF & GFS
Nature-style horizontal bar chart
LOOCV 10-Year Mean, ALL Stations
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# (metric, %_vs_ecmwf, model_val, ecmwf_val, %_vs_gfs, gfs_val)
metrics = [
    ("POD Rain",    -20.5, 0.773, 0.972,   -19.0, 0.954),
    ("CSI Rain",    +15.2, 0.478, 0.415,   +14.1, 0.419),
    ("SEDI Rain",   +52.5, 0.517, 0.339,   +89.4, 0.273),
    ("Correlation", +83.3, 0.449, 0.245,  +121.2, 0.203),
    ("CSI P90",    +164.6, 0.217, 0.082,  +155.3, 0.085),
    ("SEDI P90",   +160.8, 0.592, 0.227,  +174.1, 0.216),
    ("CSI P95",    +269.6, 0.207, 0.056,  +283.3, 0.054),
    ("SEDI P95",   +300.0, 0.473, 0.006,  +300.0, 0.112),   # ECMWF capped (actual 7783%), GFS capped (actual 322%)
]

# Sort by ECMWF improvement magnitude (ascending so largest bar is on top)
metrics.sort(key=lambda x: x[1])

labels       = [m[0] for m in metrics]
imp_ecmwf    = [m[1] for m in metrics]
model_vals   = [m[2] for m in metrics]
ecmwf_vals   = [m[3] for m in metrics]
imp_gfs      = [m[4] for m in metrics]
gfs_vals     = [m[5] for m in metrics]

# ── Colours ───────────────────────────────────────────────────────────────────
colors_e = ['#c0392b' if v < 0 else '#27ae60' for v in imp_ecmwf]
colors_g = ['#c0392b' if v < 0 else '#1565C0' for v in imp_gfs]

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

fig, ax = plt.subplots(figsize=(14, 9))

y_pos = np.arange(len(labels))
bar_h = 0.38

# ── ECMWF improvement bars (upper) ───────────────────────────────────────────
bars_e = ax.barh(y_pos + bar_h/2, imp_ecmwf, height=bar_h, color=colors_e,
               edgecolor='white', linewidth=0.5, zorder=3)

# ── GFS improvement bars (lower) ─────────────────────────────────────────────
bars_g = ax.barh(y_pos - bar_h/2, imp_gfs, height=bar_h, color=colors_g,
               edgecolor='white', linewidth=0.5, zorder=3)

# ── Zero line ─────────────────────────────────────────────────────────────────
ax.axvline(0, color='#2c3e50', linewidth=1.0, zorder=4)

# ── ECMWF bar labels ─────────────────────────────────────────────────────────
for i, (val, model_v, ecmwf_v, metric_name) in enumerate(
        zip(imp_ecmwf, model_vals, ecmwf_vals, labels)):
    sign = '+' if val > 0 else ''
    if metric_name == "SEDI P95":
        disp_text = f"{sign}{val:.0f}%  (actual +7783%)"
    else:
        disp_text = f"{sign}{val:.1f}%"

    detail_text = f"Model {model_v:.3f} vs ECMWF {ecmwf_v:.3f}"

    # Percentage label at bar end
    offset = 4 if val >= 0 else -4
    ha = 'left' if val >= 0 else 'right'

    if abs(val) > 120:
        # Large bar: detail inside (white italic), percentage outside
        ax.text(val + offset, i + bar_h/2, disp_text,
                va='center', ha=ha, fontsize=10,
                color=colors_e[i], fontweight='bold', zorder=5)
        ax.text(val - 6 if val >= 0 else val + 6, i + bar_h/2, detail_text,
                va='center', ha='right' if val >= 0 else 'left', fontsize=9.5,
                color='white', fontstyle='italic', zorder=5)
    elif abs(val) > 40:
        # Medium bar: detail inside, percentage outside
        ax.text(val + offset, i + bar_h/2, disp_text,
                va='center', ha=ha, fontsize=10,
                color=colors_e[i], fontweight='bold', zorder=5)
        ax.text(val - 4 if val >= 0 else val + 4, i + bar_h/2, detail_text,
                va='center', ha='right' if val >= 0 else 'left', fontsize=8.5,
                color='white', fontstyle='italic', zorder=5)
    else:
        # Small bar: only percentage, detail text to the right after percentage
        combined = f"{disp_text}   {detail_text}"
        ax.text(val + offset, i + bar_h/2, combined,
                va='center', ha=ha, fontsize=9,
                color=colors_e[i], zorder=5)

# ── GFS bar labels ───────────────────────────────────────────────────────────
for i, (val, model_v, gfs_v, metric_name) in enumerate(
        zip(imp_gfs, model_vals, gfs_vals, labels)):
    sign = '+' if val > 0 else ''
    disp_text = f"{sign}{val:.1f}%"

    detail_text = f"Model {model_v:.3f} vs GFS {gfs_v:.3f}"

    # Percentage label at bar end
    offset = 4 if val >= 0 else -4
    ha = 'left' if val >= 0 else 'right'

    if abs(val) > 120:
        # Large bar: detail inside (white italic), percentage outside
        ax.text(val + offset, i - bar_h/2, disp_text,
                va='center', ha=ha, fontsize=10,
                color=colors_g[i], fontweight='bold', zorder=5)
        ax.text(val - 6 if val >= 0 else val + 6, i - bar_h/2, detail_text,
                va='center', ha='right' if val >= 0 else 'left', fontsize=9.5,
                color='white', fontstyle='italic', zorder=5)
    elif abs(val) > 40:
        # Medium bar: detail inside, percentage outside
        ax.text(val + offset, i - bar_h/2, disp_text,
                va='center', ha=ha, fontsize=10,
                color=colors_g[i], fontweight='bold', zorder=5)
        ax.text(val - 4 if val >= 0 else val + 4, i - bar_h/2, detail_text,
                va='center', ha='right' if val >= 0 else 'left', fontsize=8.5,
                color='white', fontstyle='italic', zorder=5)
    else:
        # Small bar: only percentage, detail text to the right after percentage
        combined = f"{disp_text}   {detail_text}"
        ax.text(val + offset, i - bar_h/2, combined,
                va='center', ha=ha, fontsize=9,
                color=colors_g[i], zorder=5)

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Improvement  (%)', fontsize=10, labelpad=10)
ax.set_xlim(-60, 400)

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
    'Percentage Improvement of Hybrid Model over ECMWF & GFS\n'
    'LOOCV 10-Year Mean  ·  All Stations',
    fontsize=16, pad=16, linespacing=1.45
)

# ── Legend patch ──────────────────────────────────────────────────────────────
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', edgecolor='none', label='Model outperforms ECMWF'),
    Patch(facecolor='#1565C0', edgecolor='none', label='Model outperforms GFS'),
    Patch(facecolor='#c0392b', edgecolor='none', label='Baseline outperforms Model'),
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
