"""
Figure 3: Station-wise LOOCV CSI Heatmap
Three panels: (a) Raw Model CSI values, (b) % Improvement over ECMWF, (c) % Improvement over GFS
Nature-quality formatting with serif fonts, 300 DPI output.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data ─────────────────────────────────────────────────────────────────────
stations = [
    "Chevella", "Hayathnagar", "Ibrahimpatnam",
    "Kondurg", "Maheshwaram", "Saroornagar", "Yacharam",
]
thresholds = ["CSI_rain", "CSI_p90", "CSI_p95"]
threshold_labels = ["CSI (Rain/No-Rain)", "CSI (≥P90)", "CSI (≥P95)"]

# Model CSI values
model = np.array([
    [0.487, 0.173, 0.092],   # Chevella
    [0.456, 0.237, 0.166],   # Hayathnagar
    [0.461, 0.286, 0.293],   # Ibrahimpatnam
    [0.490, 0.145, 0.061],   # Kondurg
    [0.527, 0.257, 0.273],   # Maheshwaram
    [0.484, 0.182, 0.150],   # Saroornagar
    [0.432, 0.192, 0.140],   # Yacharam
])

# ECMWF CSI values
ecmwf = np.array([
    [0.426, 0.014, 0.000],
    [0.398, 0.063, 0.103],
    [0.397, 0.108, 0.100],
    [0.420, 0.064, 0.033],
    [0.452, 0.166, 0.108],
    [0.428, 0.108, 0.033],
    [0.380, 0.091, 0.050],
])

# GFS CSI values (25 km)
gfs = np.array([
    [0.430, 0.015, 0.000],
    [0.404, 0.071, 0.083],
    [0.398, 0.101, 0.051],
    [0.425, 0.113, 0.094],
    [0.463, 0.131, 0.088],
    [0.430, 0.100, 0.043],
    [0.390, 0.056, 0.038],
])

# % Improvement over ECMWF
with np.errstate(divide="ignore", invalid="ignore"):
    pct_improve_ecmwf = np.where(
        ecmwf == 0,
        np.where(model == 0, 0.0, 100.0),
        ((model - ecmwf) / ecmwf) * 100.0,
    )

# % Improvement over GFS
with np.errstate(divide="ignore", invalid="ignore"):
    pct_improve_gfs = np.where(
        gfs == 0,
        np.where(model == 0, 0.0, 100.0),
        ((model - gfs) / gfs) * 100.0,
    )

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
})

# ── Figure ───────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3,
    figsize=(24, 6),
    gridspec_kw={"wspace": 0.55},
)

# ── Helper: draw one heatmap ─────────────────────────────────────────────────
def draw_heatmap(ax, data, cmap, vmin, vmax, fmt, label, title, cbar_fmt=None):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Ticks
    ax.set_xticks(np.arange(len(threshold_labels)))
    ax.set_yticks(np.arange(len(stations)))
    ax.set_xticklabels(threshold_labels, fontsize=12)
    ax.set_yticklabels(stations, fontsize=12)

    # Move x-tick labels to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate x labels slightly
    plt.setp(ax.get_xticklabels(), rotation=25, ha="left",
             rotation_mode="anchor")

    # Cell annotations
    for i in range(len(stations)):
        for j in range(len(threshold_labels)):
            val = data[i, j]
            # Choose text colour for contrast
            norm_val = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            text_color = "white" if norm_val > 0.72 or norm_val < 0.28 else "black"
            # For the improvement panel, also darken text on deep greens
            if cmap == "RdYlGn" and norm_val > 0.78:
                text_color = "white"
            elif cmap == "RdYlGn" and norm_val < 0.35:
                text_color = "white"
            else:
                text_color = "black"

            ann = fmt.format(val)
            ax.text(j, i, ann, ha="center", va="center",
                    fontsize=10, color=text_color)

    # Grid-like cell borders
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color("#333333")

    # Minor ticks for grid lines between cells
    ax.set_xticks(np.arange(len(threshold_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(stations) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#333333", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)

    # Title
    ax.set_title(title, fontsize=16, pad=38, loc="left")

    # Colour-bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(label, fontsize=12)
    if cbar_fmt:
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(cbar_fmt))

    return im

# ── Panel (a): Raw Model CSI ────────────────────────────────────────────────
draw_heatmap(
    ax_a, model,
    cmap="Blues",
    vmin=0.0, vmax=0.55,
    fmt="{:.3f}",
    label="CSI",
    title="(a) Bias-Corrected Model CSI",
)

# ── Panel (b): % Improvement over ECMWF ─────────────────────────────────────
abs_max_e = np.max(np.abs(pct_improve_ecmwf))
sym_lim_e = min(abs_max_e * 1.15, 1200)

draw_heatmap(
    ax_b, pct_improve_ecmwf,
    cmap="RdYlGn",
    vmin=-sym_lim_e * 0.15, vmax=sym_lim_e,
    fmt="{:.0f}%",
    label="Improvement over ECMWF (%)",
    title="(b) % Improvement over ECMWF",
    cbar_fmt="%g%%",
)

# ── Panel (c): % Improvement over GFS ────────────────────────────────────────
abs_max_g = np.max(np.abs(pct_improve_gfs))
sym_lim_g = min(abs_max_g * 1.15, 1200)

draw_heatmap(
    ax_c, pct_improve_gfs,
    cmap="RdYlGn",
    vmin=-sym_lim_g * 0.15, vmax=sym_lim_g,
    fmt="{:.0f}%",
    label="Improvement over GFS (%)",
    title="(c) % Improvement over GFS",
    cbar_fmt="%g%%",
)

# ── Final adjustments ────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 3 · Station-wise LOOCV CSI Performance",
    fontsize=16, y=0.02,
    fontstyle="italic", color="#444444",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = r"D:\NEW_NRSC\paper_figures\Fig3_Station_Heatmap.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
