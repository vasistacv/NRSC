"""
Figure 4 – Radar/Spider Chart: Model vs ECMWF vs GFS LOOCV 10-Year Average (All Stations)
Nature-style publication figure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ── Data ────────────────────────────────────────────────────────────────
categories = ["CSI Rain", "POD Rain", "SEDI Rain", "CSI P90", "CSI P95", "Correlation"]
model_vals  = [0.478, 0.773, 0.517, 0.217, 0.207, 0.449]
ecmwf_vals  = [0.415, 0.972, 0.339, 0.082, 0.056, 0.245]
gfs_vals    = [0.420, 0.955, 0.273, 0.083, 0.059, 0.203]

N = len(categories)

# ── Angles (close the polygon) ─────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]                       # close loop

model_vals += model_vals[:1]
ecmwf_vals += ecmwf_vals[:1]
gfs_vals   += gfs_vals[:1]

# ── Colours ─────────────────────────────────────────────────────────────
CLR_MODEL = "#1565C0"
CLR_ECMWF = "#E65100"
CLR_GFS   = "#2E7D32"

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

# ── Global font defaults (serif, Nature style) ─────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
})

# ── Grid & spine styling ───────────────────────────────────────────────
ax.set_facecolor("white")
ax.spines['polar'].set_linewidth(2.5)
ax.spines['polar'].set_visible(True)

# Radial grid
ax.set_rlabel_position(30)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(
    ["0.2", "0.4", "0.6", "0.8", "1.0"],
    fontsize=12, color="#555555", fontfamily="serif",
)
ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.6, linestyle="--")

# Angular grid
ax.xaxis.grid(True, color="#BBBBBB", linewidth=0.6, linestyle="-")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(
    categories,
    fontsize=13, fontweight="bold", fontfamily="serif", color="#222222",
)
# Push category labels away from outer circle
ax.tick_params(axis="x", pad=18)

# ── Plot Model ──────────────────────────────────────────────────────────
ax.plot(
    angles, model_vals,
    color=CLR_MODEL, linewidth=2.4, linestyle="-",
    label="Model", zorder=4,
)
ax.fill(angles, model_vals, color=CLR_MODEL, alpha=0.25, zorder=3)
# value markers
ax.scatter(
    angles[:-1], model_vals[:-1],
    s=38, color=CLR_MODEL, edgecolors="white", linewidths=0.8, zorder=5,
)

# ── Plot ECMWF ─────────────────────────────────────────────────────────
ax.plot(
    angles, ecmwf_vals,
    color=CLR_ECMWF, linewidth=2.4, linestyle="--",
    label="ECMWF", zorder=4,
)
ax.fill(angles, ecmwf_vals, color=CLR_ECMWF, alpha=0.15, zorder=2)
ax.scatter(
    angles[:-1], ecmwf_vals[:-1],
    s=38, color=CLR_ECMWF, edgecolors="white", linewidths=0.8, zorder=5,
)

# ── Plot GFS ───────────────────────────────────────────────────────────
ax.plot(
    angles, gfs_vals,
    color=CLR_GFS, linewidth=2.4, linestyle="-.",
    label="GFS (25 km)", zorder=3,
)
ax.fill(angles, gfs_vals, color=CLR_GFS, alpha=0.10, zorder=1)
ax.scatter(
    angles[:-1], gfs_vals[:-1],
    s=38, color=CLR_GFS, edgecolors="white", linewidths=0.8, zorder=5,
)

# ── Value annotations (Per-axis manual offsets to prevent overlap) ──────
# Format: (model_radial_offset, ecmwf_radial_offset)
# Positive = outward, negative = inward
# Low-value axes need MUCH bigger offsets since labels bunch near center
manual_offsets = {
    "CSI Rain":    ( 0.12, -0.14),
    "POD Rain":    (-0.14,  0.12),   # ECMWF is higher here
    "SEDI Rain":   ( 0.12, -0.14),
    "CSI P90":     ( 0.18, -0.18),   # both values low, need big spread
    "CSI P95":     ( 0.18, -0.18),   # both values low
    "Correlation": ( 0.18, -0.18),   # both values low-mid
}

for i in range(N):
    angle = angles[i]
    m_val = model_vals[i]
    e_val = ecmwf_vals[i]
    cat = categories[i]
    m_off, e_off = manual_offsets.get(cat, (0.14, -0.14))

    # Model label
    ax.text(
        angle, m_val + m_off,
        f"{m_val:.2f}",
        ha="center", va="center",
        fontsize=8, fontweight="bold", fontfamily="serif", color=CLR_MODEL,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor=CLR_MODEL, alpha=0.92, linewidth=0.6),
        zorder=7,
    )
    # ECMWF label
    ax.text(
        angle, e_val + e_off,
        f"{e_val:.2f}",
        ha="center", va="center",
        fontsize=8, fontweight="bold", fontfamily="serif",
        color=CLR_ECMWF,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor=CLR_ECMWF, alpha=0.92, linewidth=0.6),
        zorder=7,
    )

    # GFS label
    g_val = gfs_vals[i]
    # Place GFS label further out if close to ECMWF
    g_off = e_off - 0.16 if abs(g_val - e_val) < 0.05 else e_off + 0.16
    ax.text(
        angle, g_val + g_off,
        f"{g_val:.2f}",
        ha="center", va="center",
        fontsize=8, fontweight="bold", fontfamily="serif",
        color=CLR_GFS,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor=CLR_GFS, alpha=0.92, linewidth=0.6),
        zorder=7,
    )

# ── Legend ──────────────────────────────────────────────────────────────
legend = ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.18, 1.12),
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=False,
    edgecolor="#CCCCCC",
    facecolor="white",
    framealpha=0.95,
    prop={"family": "serif", "size": 12},
)
legend.get_frame().set_linewidth(0.8)

# ── Title ───────────────────────────────────────────────────────────────
ax.set_title(
    "LOOCV 10-Year Average Performance\n(All Stations)",
    fontsize=10, fontfamily="serif",
    color="#222222", pad=28,
)

# ── Export ──────────────────────────────────────────────────────────────
out = r"D:\NEW_NRSC\paper_figures\Fig4_Radar_Comparison.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[✓] Saved → {out}")
