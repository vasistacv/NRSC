#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 6 — Spatial Performance Map of CSI_p95 (LOOCV)
Nature-quality figure showing station-level model improvement over ECMWF
on a Telangana district boundary map.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from pathlib import Path

# ──────────────────────────────────────────────────────────
# 1.  Global rcParams — Nature style
# ──────────────────────────────────────────────────────────
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
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "legend.fontsize":   10,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ──────────────────────────────────────────────────────────
# 2.  Station data
# ──────────────────────────────────────────────────────────
stations = {
    "Chevella":       {"lat": 17.3067, "lon": 78.1353, "model": 0.140, "ecmwf": 0.000, "imp": 100},
    "Hayathnagar":    {"lat": 17.3230, "lon": 78.5540, "model": 0.282, "ecmwf": 0.103, "imp": 174},
    "Ibrahimpatnam":  {"lat": 17.1017, "lon": 78.6294, "model": 0.179, "ecmwf": 0.100, "imp": 79},
    "Kondurg":        {"lat": 17.0992, "lon": 78.0369, "model": 0.108, "ecmwf": 0.033, "imp": 227},
    "Maheshwaram":    {"lat": 17.1342, "lon": 78.4334, "model": 0.237, "ecmwf": 0.108, "imp": 119},
    "Saroornagar":    {"lat": 17.3394, "lon": 78.5556, "model": 0.269, "ecmwf": 0.033, "imp": 715},
    "Yacharam":       {"lat": 17.0449, "lon": 78.6643, "model": 0.182, "ecmwf": 0.050, "imp": 264},
}

names = list(stations.keys())
lats  = np.array([stations[n]["lat"] for n in names])
lons  = np.array([stations[n]["lon"] for n in names])
model = np.array([stations[n]["model"] for n in names])
imp   = np.array([stations[n]["imp"] for n in names])

# ──────────────────────────────────────────────────────────
# 3.  Load Telangana shapefile
# ──────────────────────────────────────────────────────────
shp_path = Path(
    r"D:\NEW_NRSC\telangana_shp\20250818__Telangana_Shape_Files_District_Mandal"
    r"\TS_District_Boundary_33\TS_District_Boundary_33_FINAL.shp"
)
ts = gpd.read_file(shp_path).to_crs(epsg=4326)

# ──────────────────────────────────────────────────────────
# 4.  Colour / size scaling
# ──────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn
norm = Normalize(vmin=0, vmax=400)

# Circle sizes proportional to Model CSI — scaled for visual clarity
size_min, size_max = 120, 650
model_norm = (model - model.min()) / (model.max() - model.min() + 1e-9)
sizes = size_min + model_norm * (size_max - size_min)

# ──────────────────────────────────────────────────────────
# 5.  Build figure
# ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

# --- Telangana district boundaries -----------------------
try:
    ts_valid = ts[~ts.geometry.is_empty & ts.geometry.notnull()]
    ts_valid.plot(ax=ax, facecolor="#f5f5f0", edgecolor="#999999",
            linewidth=0.6, alpha=0.55)
except Exception as e:
    print(f"  Warning: shapefile plot failed ({e}), drawing without boundaries")
    ax.set_xlim(77.4, 79.2)
    ax.set_ylim(16.8, 17.8)

# --- Station scatter -------------------------------------
sc = ax.scatter(
    lons, lats, s=sizes, c=imp, cmap=cmap, norm=norm,
    edgecolors="k", linewidths=0.8, zorder=5, alpha=0.92,
)

# --- Station labels with halo ----------------------------
text_pe = [pe.withStroke(linewidth=2.5, foreground="white")]

# Manual nudges for readability (dx, dy in data coords) - spread out to avoid overlap
nudge = {
    "Chevella":      (-0.18,  0.04),
    "Hayathnagar":   ( 0.08,  0.06),
    "Ibrahimpatnam": ( 0.10, -0.06),
    "Kondurg":       (-0.18, -0.04),
    "Maheshwaram":   (-0.18, -0.06),
    "Saroornagar":   ( 0.08, -0.02),
    "Yacharam":      ( 0.10,  0.04),
}

for name, lat, lon, m, im in zip(names, lats, lons, model, imp):
    dx, dy = nudge.get(name, (0.03, 0.015))
    label = f"{name}\nCSI={m:.3f}  \u0394={im}%"
    ax.annotate(
        label, xy=(lon, lat), xytext=(lon + dx, lat + dy),
        fontsize=7, fontweight="bold",
        ha="left", va="center",
        path_effects=text_pe,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999",
                  alpha=0.9, linewidth=0.5),
        arrowprops=dict(arrowstyle="-", color="#444444",
                        lw=0.6, connectionstyle="arc3,rad=0.15"),
        zorder=6,
    )

# ──────────────────────────────────────────────────────────
# 6.  Colorbar
# ──────────────────────────────────────────────────────────
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.55)
cbar.set_label("Improvement over ECMWF (%)", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=10, width=0.5, length=3)
cbar.outline.set_linewidth(0.6)

# ──────────────────────────────────────────────────────────
# 7.  Size legend (Model CSI_p95)
# ──────────────────────────────────────────────────────────
legend_vals = [0.12, 0.20, 0.28]
legend_sizes = [
    size_min + ((v - model.min()) / (model.max() - model.min() + 1e-9))
    * (size_max - size_min) for v in legend_vals
]
legend_handles = [
    ax.scatter([], [], s=sz, c="gray", edgecolors="k",
               linewidths=0.6, alpha=0.75,
               label=f"CSI$_{{p95}}$ = {v:.2f}")
    for v, sz in zip(legend_vals, legend_sizes)
]
leg = ax.legend(
    handles=legend_handles, title="Model CSI$_{p95}$",
    loc="lower left", frameon=True, framealpha=0.92,
    edgecolor="#cccccc", fancybox=False, borderpad=0.8,
    title_fontsize=11, fontsize=10,
)
leg.get_frame().set_linewidth(0.5)

# ──────────────────────────────────────────────────────────
# 8.  Axis cosmetics
# ──────────────────────────────────────────────────────────
# Zoom to study region with some padding
pad = 0.15
ax.set_xlim(lons.min() - pad - 0.25, lons.max() + pad + 0.25)
ax.set_ylim(lats.min() - pad - 0.1, lats.max() + pad + 0.15)

ax.set_xlabel("Longitude (°E)", fontsize=10, labelpad=6)
ax.set_ylabel("Latitude (°N)", fontsize=10, labelpad=6)
ax.set_title(
    "Spatial Distribution of CSI$_{p95}$ Improvement (LOOCV)",
    fontsize=10, pad=12,
)
ax.set_aspect("equal")

# Light grid for geographic context
ax.grid(True, linestyle=":", linewidth=0.35, color="#bbbbbb", alpha=0.6)
ax.tick_params(which="both", top=True, right=True)

# Panel label
ax.text(
    0.02, 0.97, "(f)", transform=ax.transAxes,
    fontsize=10, va="top", ha="left",
)

# ──────────────────────────────────────────────────────────
# 9.  Save
# ──────────────────────────────────────────────────────────
out = Path(r"D:\NEW_NRSC\paper_figures\Fig6_Spatial_Performance.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, facecolor="white")
plt.close(fig)
print(f"✓ Saved  →  {out}  ({out.stat().st_size / 1024:.0f} KB)")
