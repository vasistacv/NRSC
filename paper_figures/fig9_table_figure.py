"""
Figure 9: Comprehensive Comparison Table — Model vs ECMWF vs GFS
Nature-style table with professional formatting and clear color coding.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'font.weight': 'bold',
})

# ── Data ──
metrics = ["CSI_r", "SEDI_r", "CSI_90", "CSI_95", "RMSE", "Corr"]
metric_labels = ["CSI\n(Rainy)", "SEDI\n(Rainy)", "CSI\n(\u2265P90)", "CSI\n(\u2265P95)", "RMSE\n(mm/day)", "Correlation"]

strategies = ["Temporal", "Reverse", "Random", "LOOCV"]

model = {
    "Temporal": [0.498, 0.518, 0.128, 0.101, 15.7, 0.227],
    "Reverse":  [0.466, 0.551, 0.267, 0.279, 14.1, 0.600],
    "Random":   [0.521, 0.628, 0.260, 0.250, 12.1, 0.614],
    "LOOCV":    [0.478, 0.517, 0.217, 0.207, 14.3, 0.449],
}

ecmwf = {
    "Temporal": [0.440, 0.331, 0.131, 0.087, 13.0, 0.350],
    "Reverse":  [0.393, 0.358, 0.057, 0.015, 14.5, 0.105],
    "Random":   [0.411, 0.350, 0.145, 0.037, 12.2, 0.309],
    "LOOCV":    [0.415, 0.339, 0.082, 0.056, 12.9, 0.245],
}

gfs = {
    "Temporal": [0.419, 0.273, 0.085, 0.054, 15.0, 0.203],
    "Reverse":  [0.419, 0.273, 0.085, 0.054, 15.0, 0.203],
    "Random":   [0.419, 0.273, 0.085, 0.054, 15.0, 0.203],
    "LOOCV":    [0.419, 0.273, 0.085, 0.054, 15.0, 0.203],
}

higher_better = [True, True, True, True, False, True]

# ── Colors ──
MODEL_WIN  = "#C8E6C9"   # soft green
ECMWF_WIN  = "#FFCDD2"   # soft red
TIE_BG     = "#F5F5F5"
HEADER_BG  = "#1A237E"   # deep indigo
HEADER_FG  = "#FFFFFF"
MODEL_ROW  = "#E3F2FD"   # light blue for model row label
ECMWF_ROW  = "#FFF3E0"   # light amber for ecmwf row label
GFS_ROW    = "#E8F5E9"   # light green for gfs row label

# ── Build table data ──
row_labels = []
cell_text = []
cell_colors = []

for strat in strategies:
    m_vals = model[strat]
    e_vals = ecmwf[strat]
    g_vals = gfs[strat]

    # Find best value for each metric among 3 sources
    all_sources = [m_vals, e_vals, g_vals]

    # Model row
    row_labels.append(f"{strat}\nModel")
    m_row_text = []
    m_row_colors = []
    for j, (mv, ev, gv, hb) in enumerate(zip(m_vals, e_vals, g_vals, higher_better)):
        m_row_text.append(f"{mv:.3f}" if mv < 1 else f"{mv:.1f}")
        if hb:
            is_best = mv >= ev and mv >= gv
        else:
            is_best = mv <= ev and mv <= gv
        m_row_colors.append(MODEL_WIN if is_best else TIE_BG)
    cell_text.append(m_row_text)
    cell_colors.append(m_row_colors)

    # ECMWF row
    row_labels.append(f"{strat}\nECMWF")
    e_row_text = []
    e_row_colors = []
    for j, (mv, ev, gv, hb) in enumerate(zip(m_vals, e_vals, g_vals, higher_better)):
        e_row_text.append(f"{ev:.3f}" if ev < 1 else f"{ev:.1f}")
        if hb:
            is_best = ev >= mv and ev >= gv
        else:
            is_best = ev <= mv and ev <= gv
        e_row_colors.append(MODEL_WIN if is_best else TIE_BG)
    cell_text.append(e_row_text)
    cell_colors.append(e_row_colors)

    # GFS row
    row_labels.append(f"{strat}\nGFS")
    g_row_text = []
    g_row_colors = []
    for j, (mv, ev, gv, hb) in enumerate(zip(m_vals, e_vals, g_vals, higher_better)):
        g_row_text.append(f"{gv:.3f}" if gv < 1 else f"{gv:.1f}")
        if hb:
            is_best = gv >= mv and gv >= ev
        else:
            is_best = gv <= mv and gv <= ev
        g_row_colors.append(MODEL_WIN if is_best else TIE_BG)
    cell_text.append(g_row_text)
    cell_colors.append(g_row_colors)

n_rows = len(cell_text)
n_cols = len(metric_labels)

# ── Figure ──
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")

# Title
fig.suptitle(
    "Comprehensive Performance: Model vs ECMWF vs GFS\nAcross All Validation Strategies",
    fontsize=16, fontweight="bold", fontfamily="serif", y=0.95,
)

# Table
table = ax.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=metric_labels,
    cellColours=cell_colors,
    rowColours=[MODEL_ROW if i % 3 == 0 else (ECMWF_ROW if i % 3 == 1 else GFS_ROW) for i in range(n_rows)],
    colColours=[HEADER_BG] * n_cols,
    cellLoc="center",
    rowLoc="center",
    loc="center",
    bbox=[0.10, 0.12, 0.88, 0.75],
)

table.auto_set_font_size(False)
table.set_fontsize(11)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#90A4AE")
    cell.set_linewidth(1.2)
    text = cell.get_text()
    text.set_fontfamily("serif")

    if row == 0:
        # Column headers
        cell.set_text_props(color=HEADER_FG, fontsize=11, fontweight="bold")
        cell.set_facecolor(HEADER_BG)
        cell.set_height(0.10)
    elif col == -1:
        # Row labels
        mod_idx = (row - 1) % 3  # 0=Model, 1=ECMWF, 2=GFS
        if mod_idx == 0:
            cell.set_facecolor(MODEL_ROW)
            cell.set_text_props(color="#0D47A1", fontsize=10, fontweight="bold")
        elif mod_idx == 1:
            cell.set_facecolor(ECMWF_ROW)
            cell.set_text_props(color="#E65100", fontsize=10, fontweight="bold")
        else:
            cell.set_facecolor(GFS_ROW)
            cell.set_text_props(color="#2E7D32", fontsize=10, fontweight="bold")
    else:
        # Data cells
        cell.set_text_props(fontsize=11, fontweight="bold")

# Strategy separator lines
for (row, col), cell in table.get_celld().items():
    if row >= 1 and (row - 1) % 3 == 0 and row > 1:
        cell.set_edgecolor("#546E7A")
        cell.set_linewidth(2.0)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MODEL_WIN, edgecolor="#666", label='Winner (higher = better)'),
    Patch(facecolor=ECMWF_WIN, edgecolor="#666", label='Loser'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           frameon=True, edgecolor="#999", fontsize=11, bbox_to_anchor=(0.5, 0.04))

fig.text(0.50, 0.01,
         "Note: For RMSE, lower is better; for all others, higher is better.",
         fontsize=10, fontfamily="serif", ha="center", color="#5D6D7E", fontstyle="italic")

out = r"D:\NEW_NRSC\paper_figures\Fig9_Comprehensive_Table.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {out}")
