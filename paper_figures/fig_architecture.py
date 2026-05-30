"""
fig_architecture.py — Final Architecture (MAXIMUM FONTS)
==========================================================
• Canvas 32x16 for extra room
• Very large fonts (20-28pt)
• v row fixed inside Panel A
• White formula, Classifier labels
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withStroke
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
})

fig, ax = plt.subplots(figsize=(32, 16))
ax.set_xlim(0, 32)
ax.set_ylim(0, 16)
ax.axis('off')

# Subtle bg
grad = np.ones((100, 100, 4))
for i in range(100):
    t = i / 100.0
    grad[i, :, :3] = 0.975 - 0.008 * t
    grad[i, :, 3] = 1.0
ax.imshow(grad, extent=[0, 32, 0, 16], aspect='auto', zorder=0)

# Colors
C_CNN1, C_CNN2 = '#5C9BD5', '#4472C4'
C_ATTN1, C_ATTN2 = '#7E57C2', '#5E35B1'
C_CONCAT = '#FFA726'
C_XGB_BG = '#E8F5E9'
C_XGB1, C_XGB2, C_XGB3 = '#43A047', '#388E3C', '#2E7D32'
C_CAL, C_OUT = '#FB8C00', '#E53935'
C_ARR = '#37474F'

# ── Helpers ──────────────────────────────────────────────────────────────
def pbox(x, y, w, h, fc, text, fs=18, tc='white'):
    r, g, b = mcolors.to_rgb(fc)
    ec = (max(r - 0.15, 0), max(g - 0.15, 0), max(b - 0.15, 0))
    bx = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                        fc=fc, ec=ec, lw=1.5, zorder=5)
    ax.add_patch(bx)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc, zorder=6,
            linespacing=1.2)

def block3d(x, y, w, h, fc, text, fs=15, tc='white'):
    r, g, b = mcolors.to_rgb(fc)
    sk = 0.1 * h
    front = plt.Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         fc=fc, ec=(max(r - 0.12, 0), max(g - 0.12, 0), max(b - 0.12, 0)),
                         lw=1.5, zorder=5)
    ax.add_patch(front)
    top = plt.Polygon([[x, y + h], [x + w, y + h],
                        [x + w + sk, y + h + sk], [x + sk, y + h + sk]],
                       fc=(min(r + 0.12, 1), min(g + 0.12, 1), min(b + 0.12, 1)),
                       ec=(max(r - 0.08, 0), max(g - 0.08, 0), max(b - 0.08, 0)),
                       lw=0.8, zorder=5)
    ax.add_patch(top)
    side = plt.Polygon([[x + w, y], [x + w + sk, y + sk],
                         [x + w + sk, y + h + sk], [x + w, y + h]],
                        fc=(max(r - 0.1, 0), max(g - 0.1, 0), max(b - 0.1, 0)),
                        ec=(max(r - 0.15, 0), max(g - 0.15, 0), max(b - 0.15, 0)),
                        lw=0.8, zorder=5)
    ax.add_patch(side)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=tc, zorder=6,
            rotation=90, linespacing=1.1)

def pdiamond(cx, cy, s, fc, text, fs=15, tc='white'):
    r, g, b = mcolors.to_rgb(fc)
    d = plt.Polygon([[cx, cy + s], [cx + s, cy], [cx, cy - s], [cx - s, cy]],
                     fc=fc, ec=(max(r - 0.15, 0), max(g - 0.15, 0), max(b - 0.15, 0)),
                     lw=2.5, zorder=7)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=8, linespacing=1.15)

def parr(x1, y1, x2, y2, c=C_ARR, lw=3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                mutation_scale=20))

# ═══════════════════════════════════════════════════════════════════════════
# BOLD BLACK BORDER (lw=10)
# ═══════════════════════════════════════════════════════════════════════════
fill = FancyBboxPatch((0.3, 0.3), 31.4, 15.2, boxstyle="round,pad=0.25",
                       fc='#FAFCFF', ec='none', lw=0, zorder=1)
ax.add_patch(fill)

border = FancyBboxPatch((0.3, 0.3), 31.4, 15.2, boxstyle="round,pad=0.25",
                         fc='none', ec='black', lw=10, zorder=10)
ax.add_patch(border)

# ═══════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════
ax.text(16, 15.05,
        'Hybrid AttentionNet-XGBoost Framework for Statistical Downscaling',
        ha='center', va='center', fontsize=28, fontweight='bold',
        color='#1A237E', fontfamily='serif',
        path_effects=[withStroke(linewidth=3, foreground='white')],
        zorder=11)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL A: 19 Atmospheric Channels  (y: 8.0 → 14.3)
# height = 6.3, enough for all items
# ═══════════════════════════════════════════════════════════════════════════
panel_a = FancyBboxPatch((0.7, 8.0), 8.0, 6.3, boxstyle="round,pad=0.15",
                          fc='#F5F8FF', ec='#1565C0', lw=2.5, zorder=2)
ax.add_patch(panel_a)

ax.text(4.7, 13.85, '9 Atmospheric Variables', ha='center', fontsize=23,
        fontweight='bold', color='#0D47A1')
ax.text(4.7, 13.35, '(4 Sfc + 5 PL × 3 Heights = 19 Channels, 9×9 Grid)', ha='center',
        fontsize=14, color='#546E7A', fontstyle='italic')

ax.text(1.1, 12.75, 'Surface Variables (4)', fontsize=20, fontweight='bold',
        color='#1565C0')
sfc = [('tp', 'Total Precipitation'), ('tcwv', 'Total Column Water Vapour'),
       ('cape', 'Conv. Available PE'), ('d2m', '2m Dewpoint Temp.')]
for i, (c, d) in enumerate(sfc):
    yy = 12.25 - i * 0.45
    ax.text(1.3, yy, c, fontsize=18, fontweight='bold', color='#1565C0',
            fontfamily='monospace')
    ax.text(3.2, yy, d, fontsize=17, color='#37474F')

ax.text(1.1, 10.3, 'Pressure-Level (15)', fontsize=20, fontweight='bold',
        color='#5C6BC0')
ax.text(1.1, 9.85, '@ 850/500/200 hPa (×3 each)', fontsize=15,
        color='#78909C', fontstyle='italic')
pl = [('r', 'Rel. Humidity'), ('w', 'Vert. Velocity'),
      ('vo', 'Vorticity'), ('u', 'U-wind'), ('v', 'V-wind')]
for i, (c, d) in enumerate(pl):
    yy = 9.45 - i * 0.35
    ax.text(1.3, yy, c, fontsize=18, fontweight='bold', color='#5C6BC0',
            fontfamily='monospace')
    ax.text(3.2, yy, f'{d} (×3)', fontsize=17, color='#37474F')
# Last item "v" at 9.45 - 4*0.35 = 8.05 → above panel bottom 8.0 ✓

# ═══════════════════════════════════════════════════════════════════════════
# PANEL B: 24 Physics Features  (y: 0.7 → 7.5)
# gap = 8.0 - 7.5 = 0.5
# ═══════════════════════════════════════════════════════════════════════════
panel_b = FancyBboxPatch((0.7, 0.7), 8.0, 6.8, boxstyle="round,pad=0.15",
                          fc='#F5FFF5', ec='#2E7D32', lw=2.5, zorder=2)
ax.add_patch(panel_b)

ax.text(4.7, 7.1, '24 Physics-Derived Scalar Features', ha='center',
        fontsize=23, fontweight='bold', color='#1B5E20')

col1 = ['tp (mm)', 'tcwv', 'cape', 'd2m', 'r_850', 'r_500', 'r_200',
        'w_850', 'w_500', 'u_850', 'u_200', 'v_850']
for i, v in enumerate(col1):
    ax.text(1.0, 6.55 - i * 0.47, f'{i + 1:2d}. {v}', fontsize=17,
            color='#37474F', fontfamily='monospace')
# Last: 6.55 - 11*0.47 = 1.38 → above 0.7 ✓

col2 = ['v_200', 'vo_850', 'ws_850', 'ws_200', 'shear_mag', 'tp_log',
        'cape_uplift', 'd2m_dev', 'vort_rh', 'rh_diff', 'cape_tcwv',
        'tp_cape']
for i, v in enumerate(col2):
    ax.text(5.0, 6.55 - i * 0.47, f'{i + 13:2d}. {v}', fontsize=17,
            color='#37474F', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE FLOW
# ═══════════════════════════════════════════════════════════════════════════

# ── Input boxes ──────────────────────────────────────────────────────────
pbox(9.5, 9.2, 2.4, 2.6, '#00897B',
     'ECMWF\nGrid Patch\n9×9×19', fs=17)

pbox(9.5, 4.2, 2.4, 2.4, '#2E7D32',
     '24 Physics\nFeatures', fs=17)

# Arrows from panels
parr(8.7, 10.5, 9.5, 10.5, c='#00897B')
parr(8.7, 5.4, 9.5, 5.4, c='#2E7D32')

# ── CNN Backbone ─────────────────────────────────────────────────────────
ax.text(15.5, 14.2, 'CNN Backbone', fontsize=26, fontweight='bold',
        color='#1A237E', ha='center',
        path_effects=[withStroke(linewidth=3, foreground='white')],
        zorder=3)

parr(11.9, 10.5, 12.7, 10.5, c='#00897B', lw=3)

block3d(12.8, 8.5, 1.0, 3.8, C_CNN1, 'Conv\n19→48', fs=15)
block3d(14.1, 8.5, 1.0, 3.8, C_CNN2, 'Conv\n48→96', fs=15)
block3d(15.4, 8.5, 1.0, 3.8, C_ATTN1, 'Channel\nAttn', fs=14)
block3d(16.7, 8.5, 1.0, 3.8, C_ATTN2, 'Spatial\nAttn', fs=14)

# ── Tabular MLP ──────────────────────────────────────────────────────────
ax.text(14.5, 7.8, 'Tabular MLP', fontsize=26, fontweight='bold',
        color='#1B5E20', ha='center',
        path_effects=[withStroke(linewidth=3, foreground='white')],
        zorder=3)

parr(11.9, 5.4, 12.7, 5.4, c='#2E7D32', lw=3)

pbox(12.8, 3.8, 3.0, 2.8, '#388E3C',
     '24→128→64\nBN + SiLU\nDropout', fs=18)

# ── Concat Diamond ───────────────────────────────────────────────────────
pdiamond(19.5, 7.8, 1.0, C_CONCAT, 'Concat\n160', fs=15)

parr(17.9, 10.3, 19.5, 8.8, c='#5C6BC0', lw=3)
parr(15.8, 5.2, 19.5, 6.8, c='#388E3C', lw=3)

# ── XGBoost 3-Stage Ensemble ────────────────────────────────────────────
xgb_bg = FancyBboxPatch((21.0, 3.0), 9.5, 9.5,
                         boxstyle="round,pad=0.25",
                         fc=C_XGB_BG, ec='#43A047', lw=2.5, zorder=2)
ax.add_patch(xgb_bg)

ax.text(25.75, 12.0, 'XGBoost 3-Stage Ensemble', fontsize=24,
        fontweight='bold', color='#1B5E20', ha='center',
        path_effects=[withStroke(linewidth=3, foreground='white')],
        zorder=3)

parr(20.5, 7.8, 21.5, 7.8, c=C_ARR, lw=3.5)

# Stage 1 — Rain Gate CLASSIFIER
pbox(21.5, 6.2, 2.8, 4.2, C_XGB1,
     'Rain Gate\n─────\nBinary\nClassifier', fs=18)
parr(24.3, 8.3, 24.6, 8.3, c='#2E7D32', lw=2.5)

# Stage 2 — Intensity Regressor
pbox(24.6, 6.2, 2.8, 4.2, C_XGB2,
     'Intensity\nRegressor\n─────\nWeighted\nSamples', fs=18)
parr(27.4, 8.3, 27.7, 8.3, c='#2E7D32', lw=2.5)

# Stage 3 — Extreme CLASSIFIER
pbox(27.7, 6.2, 2.5, 4.2, C_XGB3,
     'Extreme\nClassifier\n─────\nP90+\nDetection', fs=17)

# Arrow down
parr(25.75, 6.2, 25.75, 5.3, c=C_ARR, lw=3.5)

# Calibration
pbox(21.8, 3.3, 3.6, 1.8, C_CAL,
     'Calibration & Blending\nIsotonic + Grid Search',
     fs=16, tc='#3E2723')

parr(25.4, 4.2, 26.0, 4.2, c=C_ARR, lw=3.5)

# Output
pbox(26.0, 3.3, 3.5, 1.8, C_OUT,
     'Rainfall Prediction\n(mm/day)', fs=19)

# ═══════════════════════════════════════════════════════════════════════════
# FORMULA — WHITE text on dark blue
# ═══════════════════════════════════════════════════════════════════════════
formula_box = FancyBboxPatch((9.5, 0.7), 21.0, 1.2,
                              boxstyle="round,pad=0.12",
                              fc='#1A237E', ec='#283593', lw=2, zorder=5)
ax.add_patch(formula_box)
ax.text(20.0, 1.3,
        r'$\mathbf{Final \;=\; RainGate \;\times\; '
        r'[\,w_{nn}\!\cdot\!NN_{cal} \;+\; '
        r'(1 - w_{nn})\!\cdot\!XGB_{reg}\,] '
        r'\;\times\; ExtremeBoost}$',
        ha='center', va='center', fontsize=22, color='white',
        fontfamily='serif', zorder=6)

# ── Save ──────────────────────────────────────────────────────────────────
out = r"D:\NEW_NRSC\paper_figures\Fig_Architecture.png"
plt.tight_layout(pad=0.2)
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"[OK] Saved -> {out}")
