#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 – Study Area Map (Nature-quality)
Panel (a): India with neighboring countries context, Telangana highlighted.
Panel (b): Telangana districts + 7 station locations (zoomed).
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Load shapefiles ──
india = gpd.read_file(r"D:\NEW_NRSC\Shapefile_India\STATE_BDY_UPD.shp").to_crs(epsg=4326)
ts_districts = gpd.read_file(
    r"D:\NEW_NRSC\telangana_shp\20250818__Telangana_Shape_Files_District_Mandal"
    r"\TS_District_Boundary_33\TS_District_Boundary_33_FINAL.shp"
).to_crs(epsg=4326)

telangana = india[india["STATE"] == "TELANGANA"]
other_states = india[india["STATE"] != "TELANGANA"]

# ── Station data ──
stations = {
    "Chevella":       (17.3067, 78.1353),
    "Hayathnagar":    (17.3230, 78.5540),
    "Ibrahimpatnam":  (17.1017, 78.6294),
    "Kondurg":        (17.0992, 78.0369),
    "Maheshwaram":    (17.1342, 78.4334),
    "Saroornagar":    (17.3394, 78.5556),
    "Yacharam":       (17.0449, 78.6643),
}

# Better label offsets to avoid overlap
label_offsets = {
    "Chevella":       (-0.50, 0.08),
    "Hayathnagar":    (0.12,  0.12),
    "Ibrahimpatnam":  (0.12, -0.12),
    "Kondurg":        (-0.55, -0.08),
    "Maheshwaram":    (-0.60, -0.10),
    "Saroornagar":    (0.12,  -0.04),
    "Yacharam":       (0.12, -0.10),
}

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 7),
    gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.12},
)

# ── Panel (a) – India (professional, colorful) ──
# Give different regions subtle colors for visual interest
# North India
north = ["JAMMU & KASHMIR", "LADAKH", "HIMACHAL PRADESH", "PUNJAB", "HARYANA",
         "UTTARAKHAND", "UTTAR PRADESH", "DELHI"]
south = ["KARNATAKA", "TAMIL NADU", "KERALA", "ANDHRA PRADESH", "GOA"]
east = ["WEST BENGAL", "ODISHA", "BIHAR", "JHARKHAND", "SIKKIM",
        "ASSAM", "MEGHALAYA", "TRIPURA", "MIZORAM", "MANIPUR", "NAGALAND", "ARUNACHAL PRADESH"]
west = ["RAJASTHAN", "GUJARAT", "MAHARASHTRA", "MADHYA PRADESH", "CHHATTISGARH"]

region_colors = {
    "north": "#E8F0FE",  # light blue
    "south": "#FFF3E0",  # light orange
    "east": "#E8F5E9",   # light green
    "west": "#FDE8E8",   # light pink
}

for _, row in other_states.iterrows():
    state = row["STATE"]
    if state in north:
        color = region_colors["north"]
    elif state in south:
        color = region_colors["south"]
    elif state in east:
        color = region_colors["east"]
    elif state in west:
        color = region_colors["west"]
    else:
        color = "#F5F5F5"
    gpd.GeoDataFrame([row]).plot(ax=ax1, color=color, edgecolor="#BDBDBD", linewidth=0.4, zorder=1)

# Telangana filled with strong color
telangana.plot(ax=ax1, color="#00897B", edgecolor="#004D40", linewidth=0.8, zorder=2)

# Red bounding box
ts_bounds = telangana.total_bounds
pad = 0.8
rect = mpatches.FancyBboxPatch(
    (ts_bounds[0] - pad, ts_bounds[1] - pad),
    (ts_bounds[2] - ts_bounds[0]) + 2 * pad,
    (ts_bounds[3] - ts_bounds[1]) + 2 * pad,
    boxstyle="round,pad=0.15",
    linewidth=2.5, edgecolor="#D32F2F", facecolor="none", zorder=3,
)
ax1.add_patch(rect)

ax1.annotate(
    "Study Area", fontsize=11, fontstyle="italic", fontweight="bold", color="#D32F2F",
    xy=(ts_bounds[2] + pad * 0.5, (ts_bounds[1] + ts_bounds[3]) / 2),
    xytext=(ts_bounds[2] + 4, ts_bounds[1] - 3),
    arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5, connectionstyle="arc3,rad=-0.2"),
    zorder=4,
)

ax1.set_title("India", fontsize=18, fontweight="bold", pad=12)
ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold", va="top")
ax1.set_xticks([])
ax1.set_yticks([])
for sp in ax1.spines.values():
    sp.set_linewidth(2.5)
    sp.set_color("black")
    sp.set_visible(True)
ax1.set_facecolor("#F0F8FF")  # very light sky blue for ocean feel

# ── Panel (b) – Telangana (ZOOMED IN) ──
# Color districts with a light gradient for visual interest
np.random.seed(42)
n_districts = len(ts_districts)
pastel_blues = plt.cm.Blues(np.linspace(0.08, 0.25, n_districts))
for i, (_, district) in enumerate(ts_districts.iterrows()):
    gpd.GeoDataFrame([district]).plot(ax=ax2, color=pastel_blues[i],
                                       edgecolor="#37474F", linewidth=0.6, zorder=1)

# Zoom to Telangana bounds
ts_full_bounds = ts_districts.total_bounds
x_margin = (ts_full_bounds[2] - ts_full_bounds[0]) * 0.05
y_margin = (ts_full_bounds[3] - ts_full_bounds[1]) * 0.05
ax2.set_xlim(ts_full_bounds[0] - x_margin, ts_full_bounds[2] + x_margin)
ax2.set_ylim(ts_full_bounds[1] - y_margin, ts_full_bounds[3] + y_margin)

# Plot station markers
lats = [c[0] for c in stations.values()]
lons = [c[1] for c in stations.values()]
ax2.scatter(lons, lats, marker="*", s=400, c="#D32F2F",
            edgecolors="#B71C1C", linewidths=0.7, zorder=4)

for name, (lat, lon) in stations.items():
    dx, dy = label_offsets[name]
    ax2.annotate(
        name, xy=(lon, lat), xytext=(lon + dx, lat + dy),
        fontsize=8, fontweight="bold", color="#212121",
        arrowprops=dict(arrowstyle="-", color="#757575", lw=0.8, shrinkA=0, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#9E9E9E",
                  linewidth=0.6, alpha=0.92),
        zorder=5,
    )

legend_elements = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#D32F2F",
           markeredgecolor="#B71C1C", markersize=16, label="Station", linewidth=0),
]
ax2.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.95,
           edgecolor="#666", fontsize=11, handletextpad=0.4, borderpad=0.5)

ax2.set_title("Telangana \u2013 Study Region", fontsize=18, fontweight="bold", pad=12)
ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="top")

ax2.tick_params(axis="both", labelsize=11, width=2.0, length=5)
ax2.set_xlabel("Longitude (\u00b0E)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Latitude (\u00b0N)", fontsize=14, fontweight="bold")
for sp in ax2.spines.values():
    sp.set_linewidth(2.5)
    sp.set_color("black")
    sp.set_visible(True)
ax2.set_facecolor("white")

fig.patch.set_facecolor("white")
plt.tight_layout()
out_path = r"D:\NEW_NRSC\paper_figures\Fig1_Study_Area_Map.png"
fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"[OK] Figure saved -> {out_path}")
