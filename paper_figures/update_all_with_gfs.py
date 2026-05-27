"""
update_all_figures_with_gfs.py
===============================
Regenerate ALL comparison figures with three-way comparison:
  Model (blue) vs ECMWF (orange) vs GFS (green)

Figures updated:
  - Fig2: Performance Bars (renamed from Fig3 heatmap numbering)
  - Fig3: Station Heatmap
  - Fig4: Radar Chart
  - Fig5: Split Comparison (GFS not applicable here - only LOOCV)
  - Fig8: LOYO Yearly
  - Fig9: Comprehensive Table
  - Fig10: Improvement Summary
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"D:\NEW_NRSC\.venv\Library\bin;" + os.environ.get("PATH", "")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.linewidth': 2.5,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'xtick.major.width': 2.5, 'ytick.major.width': 2.5,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'savefig.dpi': 300,
})

OUT = Path(r"D:\NEW_NRSC\paper_figures")

# Colors
CLR_MODEL = "#1565C0"
CLR_ECMWF = "#E65100"
CLR_GFS   = "#2E7D32"

# ── Load data ──
with open(r"D:\NEW_NRSC\experiment_outputs\final_ensemble\loyo_results.json") as f:
    loyo = json.load(f)
with open(r"D:\NEW_NRSC\ecmwf_baseline_results.json") as f:
    ecmwf = json.load(f)
with open(r"D:\NEW_NRSC\gfs_baseline_results.json") as f:
    gfs = json.load(f)

# ── Aggregate helpers ──
def pool_csi(data, label):
    H = sum(v.get(f"H_{label}", 0) for v in data.values())
    M = sum(v.get(f"M_{label}", 0) for v in data.values())
    FA = sum(v.get(f"FA_{label}", 0) for v in data.values())
    return H / (H + M + FA) if (H + M + FA) > 0 else 0

def pool_pod(data, label):
    H = sum(v.get(f"H_{label}", 0) for v in data.values())
    M = sum(v.get(f"M_{label}", 0) for v in data.values())
    return H / (H + M) if (H + M) > 0 else 0

def avg_metric(data, key):
    vals = [v[key] for v in data.values() if key in v]
    return np.mean(vals) if vals else 0

def station_pool(data, stn, label):
    stn_data = {k: v for k, v in data.items() if k.startswith(stn + "_")}
    return pool_csi(stn_data, label)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3: STATION HEATMAP — 3 panels (Model, ECMWF improvement, GFS improvement)
# ═══════════════════════════════════════════════════════════════════════════
def gen_fig3():
    print("Generating Fig3 (Station Heatmap with GFS)...")
    stations = ["Chevella", "Hayathnagar", "Ibrahimpatnam", "Kondurg",
                "Maheshwaram", "Saroornagar", "Yacharam"]
    metrics = ["CSI Rain", "CSI P90", "CSI P95"]

    # Model CSI from LOYO per_station
    loyo_ps = loyo.get("per_station", {})
    model_vals = np.zeros((7, 3))
    ecmwf_vals = np.zeros((7, 3))
    gfs_vals   = np.zeros((7, 3))

    for i, stn in enumerate(stations):
        # Model
        stn_key = stn
        stn_d = loyo_ps.get(stn_key, {})
        model_vals[i, 0] = stn_d.get("CSI_rain", 0)
        model_vals[i, 1] = stn_d.get("CSI_p90", 0)
        model_vals[i, 2] = stn_d.get("CSI_p95", 0)
        # ECMWF
        for j, label in enumerate(["rain", "p90", "p95"]):
            ecmwf_vals[i, j] = station_pool(ecmwf, stn, label)
            gfs_vals[i, j]   = station_pool(gfs, stn, label)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    import seaborn as sns

    # (a) Model CSI
    sns.heatmap(model_vals, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=metrics, yticklabels=stations,
                linewidths=1, linecolor='white', vmin=0, vmax=0.55,
                cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title("(a) Model CSI Values", fontsize=15, fontweight='bold')

    # (b) % Improvement over ECMWF
    imp_ecmwf = np.where(ecmwf_vals > 0,
                         ((model_vals - ecmwf_vals) / ecmwf_vals) * 100, 0)
    # For zero ECMWF, show 100% if model > 0
    for i in range(7):
        for j in range(3):
            if ecmwf_vals[i, j] == 0 and model_vals[i, j] > 0:
                imp_ecmwf[i, j] = 100.0

    annot_ecmwf = np.array([[f"{v:.0f}%" for v in row] for row in imp_ecmwf])
    sns.heatmap(imp_ecmwf, annot=annot_ecmwf, fmt="", cmap="RdYlGn",
                xticklabels=metrics, yticklabels=stations,
                linewidths=1, linecolor='white', center=0,
                cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title("(b) Improvement over ECMWF (%)", fontsize=15, fontweight='bold')

    # (c) % Improvement over GFS
    imp_gfs = np.where(gfs_vals > 0,
                       ((model_vals - gfs_vals) / gfs_vals) * 100, 0)
    for i in range(7):
        for j in range(3):
            if gfs_vals[i, j] == 0 and model_vals[i, j] > 0:
                imp_gfs[i, j] = 100.0

    annot_gfs = np.array([[f"{v:.0f}%" for v in row] for row in imp_gfs])
    sns.heatmap(imp_gfs, annot=annot_gfs, fmt="", cmap="RdYlGn",
                xticklabels=metrics, yticklabels=stations,
                linewidths=1, linecolor='white', center=0,
                cbar_kws={"shrink": 0.8}, ax=axes[2])
    axes[2].set_title("(c) Improvement over GFS (%)", fontsize=15, fontweight='bold')

    for a in axes:
        a.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "Fig3_Station_Heatmap.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print("  -> Fig3_Station_Heatmap.png DONE")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4: RADAR CHART — Model vs ECMWF vs GFS
# ═══════════════════════════════════════════════════════════════════════════
def gen_fig4():
    print("Generating Fig4 (Radar with GFS)...")

    categories = ["CSI Rain", "POD Rain", "SEDI Rain", "CSI P90", "CSI P95", "Correlation"]

    # Model
    m_agg = loyo["aggregate_metrics"]
    model = [m_agg["CSI_rain"], m_agg["POD_rain"], m_agg["SEDI_rain"],
             m_agg["CSI_p90"], m_agg["CSI_p95"], m_agg["corr_rainy"]]

    # ECMWF pooled
    ec = [pool_csi(ecmwf, "rain"), pool_pod(ecmwf, "rain"), avg_metric(ecmwf, "SEDI_rain"),
          pool_csi(ecmwf, "p90"), pool_csi(ecmwf, "p95"), avg_metric(ecmwf, "corr_rainy")]

    # GFS pooled
    gf = [pool_csi(gfs, "rain"), pool_pod(gfs, "rain"), avg_metric(gfs, "SEDI_rain"),
          pool_csi(gfs, "p90"), pool_csi(gfs, "p95"), avg_metric(gfs, "corr_rainy")]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    model_v = model + model[:1]
    ecmwf_v = ec + ec[:1]
    gfs_v   = gf + gf[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines['polar'].set_linewidth(2.5)

    ax.set_rlabel_position(30)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=11, color="#555")
    ax.yaxis.grid(True, color="#CCC", linewidth=0.6, linestyle="--")
    ax.xaxis.grid(True, color="#BBB", linewidth=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold", color="#222")
    ax.tick_params(axis="x", pad=20)

    # Plot
    ax.plot(angles, model_v, color=CLR_MODEL, lw=2.4, label="Our Model", zorder=4)
    ax.fill(angles, model_v, color=CLR_MODEL, alpha=0.20, zorder=3)
    ax.scatter(angles[:-1], model_v[:-1], s=40, color=CLR_MODEL, edgecolors="white", lw=0.8, zorder=5)

    ax.plot(angles, ecmwf_v, color=CLR_ECMWF, lw=2.4, linestyle="--", label="ECMWF (9 km)", zorder=4)
    ax.fill(angles, ecmwf_v, color=CLR_ECMWF, alpha=0.12, zorder=2)
    ax.scatter(angles[:-1], ecmwf_v[:-1], s=40, color=CLR_ECMWF, edgecolors="white", lw=0.8, zorder=5)

    ax.plot(angles, gfs_v, color=CLR_GFS, lw=2.4, linestyle="-.", label="GFS (25 km)", zorder=4)
    ax.fill(angles, gfs_v, color=CLR_GFS, alpha=0.10, zorder=1)
    ax.scatter(angles[:-1], gfs_v[:-1], s=40, color=CLR_GFS, edgecolors="white", lw=0.8, zorder=5)

    # Value labels with bbox
    for i in range(N):
        a = angles[i]
        for val, color, off in [(model_v[i], CLR_MODEL, 0.14),
                                 (ecmwf_v[i], CLR_ECMWF, -0.14),
                                 (gfs_v[i], CLR_GFS, -0.28 if gfs_v[i] < ecmwf_v[i] else 0.28)]:
            # Skip if too close to another value
            ax.text(a, val + off, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor=color, alpha=0.9, linewidth=0.4),
                    zorder=7)

    ax.set_title("LOOCV 10-Year Average Performance\n(All Stations)",
                 fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12),
              fontsize=12, frameon=True, edgecolor="#333")

    plt.tight_layout()
    fig.savefig(OUT / "Fig4_Radar_Comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print("  -> Fig4_Radar_Comparison.png DONE")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 9: COMPREHENSIVE TABLE — Model vs ECMWF vs GFS
# ═══════════════════════════════════════════════════════════════════════════
def gen_fig9():
    print("Generating Fig9 (Comprehensive Table with GFS)...")
    m_agg = loyo["aggregate_metrics"]

    rows = []
    metrics_list = [
        ("CSI Rain",   "CSI_rain"),
        ("POD Rain",   "POD_rain"),
        ("FAR Rain",   "FAR_rain"),
        ("SEDI Rain",  "SEDI_rain"),
        ("CSI P90",    "CSI_p90"),
        ("POD P90",    "POD_p90"),
        ("SEDI P90",   "SEDI_p90"),
        ("CSI P95",    "CSI_p95"),
        ("POD P95",    "POD_p95"),
        ("SEDI P95",   "SEDI_p95"),
        ("Corr (rainy)", "corr_rainy"),
        ("RMSE (mm)",  "RMSE"),
    ]

    for label, key in metrics_list:
        m_val = m_agg.get(key, 0)
        e_val = avg_metric(ecmwf, key)
        g_val = avg_metric(gfs, key)
        rows.append([label, f"{m_val:.3f}", f"{e_val:.3f}", f"{g_val:.3f}"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    col_labels = ["Metric", "Our Model", "ECMWF (9 km)", "GFS (25 km)"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Header styling
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor('#1A237E')
        cell.set_text_props(color='white', fontweight='bold', fontsize=13)

    # Row styling
    for i in range(1, len(rows) + 1):
        for j in range(4):
            cell = table[i, j]
            cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#DDD')
            if j == 0:
                cell.set_text_props(fontweight='bold')
            elif j == 1:  # Model column - highlight best
                cell.set_text_props(color=CLR_MODEL, fontweight='bold')

    ax.set_title("Comprehensive Metric Comparison — LOOCV 10-Year Average",
                 fontsize=16, fontweight='bold', pad=20, y=1.02)

    plt.tight_layout()
    fig.savefig(OUT / "Fig9_Comprehensive_Table.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print("  -> Fig9_Comprehensive_Table.png DONE")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 10: IMPROVEMENT SUMMARY BAR CHART — vs ECMWF and vs GFS
# ═══════════════════════════════════════════════════════════════════════════
def gen_fig10():
    print("Generating Fig10 (Improvement Summary with GFS)...")
    m_agg = loyo["aggregate_metrics"]

    metrics_list = [
        ("CSI\nRain", "CSI_rain"),
        ("POD\nRain", "POD_rain"),
        ("SEDI\nRain", "SEDI_rain"),
        ("CSI\nP90", "CSI_p90"),
        ("CSI\nP95", "CSI_p95"),
        ("Corr", "corr_rainy"),
    ]

    labels = [m[0] for m in metrics_list]
    model_v = [m_agg.get(m[1], 0) for m in metrics_list]
    ecmwf_v = [avg_metric(ecmwf, m[1]) for m in metrics_list]
    gfs_v   = [avg_metric(gfs, m[1]) for m in metrics_list]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - w, model_v, w, label="Our Model", color=CLR_MODEL,
                   edgecolor="white", linewidth=1.2, zorder=3)
    bars2 = ax.bar(x, ecmwf_v, w, label="ECMWF (9 km)", color=CLR_ECMWF,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.85)
    bars3 = ax.bar(x + w, gfs_v, w, label="GFS (25 km)", color=CLR_GFS,
                   edgecolor="white", linewidth=1.2, zorder=3, alpha=0.85)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=15, fontweight='bold')
    ax.set_title("Performance Comparison — Our Model vs ECMWF vs GFS\n(LOOCV 10-Year Average, All Stations)",
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right', frameon=True, edgecolor='#333')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for sp in ax.spines.values():
        sp.set_linewidth(2.5)

    plt.tight_layout()
    fig.savefig(OUT / "Fig10_Improvement_Summary.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print("  -> Fig10_Improvement_Summary.png DONE")


# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    gen_fig3()
    gen_fig4()
    gen_fig9()
    gen_fig10()
    print("\n[OK] All figures updated with GFS comparison!")
