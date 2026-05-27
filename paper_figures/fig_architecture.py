"""
fig_architecture.py
====================
Publication-quality Neural Network Architecture Diagram.
AttentionNet: CNN (Channel + Spatial Attention) + Tabular MLP → Fusion → Prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
})

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(-0.5, 20)
ax.set_ylim(-1, 10.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colors ──
C_INPUT    = '#E3F2FD'  # light blue
C_CNN      = '#1565C0'  # dark blue
C_ATTN     = '#7B1FA2'  # purple
C_MLP      = '#00897B'  # teal
C_FUSION   = '#E65100'  # orange
C_OUTPUT   = '#C62828'  # red
C_ARROW    = '#455A64'
C_HEADER   = '#0D47A1'
C_TAB_IN   = '#E8F5E9'  # light green

def draw_box(x, y, w, h, color, text, fontsize=10, textcolor='white',
             alpha=0.95, edgecolor='#333', lw=1.5, bold=True):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=textcolor, zorder=4)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))

def draw_curved_arrow(x1, y1, x2, y2, color=C_ARROW, rad=0.3, lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"))

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(10, 10.2, 'AttentionNet — Dual-Path Architecture for Statistical Downscaling',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color=C_HEADER, fontfamily='serif')
ax.text(10, 9.7, 'CNN Backbone with Channel & Spatial Attention + Tabular MLP → Fusion Head',
        ha='center', va='center', fontsize=12, fontweight='normal',
        color='#555', fontfamily='serif')

# ═══════════════════════════════════════════════════════════════
# PATH LABELS
# ═══════════════════════════════════════════════════════════════
ax.text(5.5, 8.95, 'CNN Path (Spatial Features)', ha='center', fontsize=13,
        fontweight='bold', color=C_CNN, fontstyle='italic')
ax.text(5.5, 3.95, 'Tabular Path (Physics Features)', ha='center', fontsize=13,
        fontweight='bold', color=C_MLP, fontstyle='italic')

# ═══════════════════════════════════════════════════════════════
# INPUT BOXES
# ═══════════════════════════════════════════════════════════════

# CNN Input
draw_box(0, 7.2, 2.8, 1.4, C_INPUT, 'ECMWF Grid Patch\n9×9 × 19 channels\n(tp, tcwv, cape, d2m,\nr, w, vo, u, v\n@ 850/500/200 hPa)',
         fontsize=8.5, textcolor='#1A1A1A', edgecolor=C_CNN, lw=2)

# Tabular Input
draw_box(0, 2.2, 2.8, 1.4, C_TAB_IN, 'Tabular Features\n24 scalars\n(wind shear, CAPE×TCWV,\nRH diff, vorticity×RH,\ntp_log, d2m_dev, ...)',
         fontsize=8.5, textcolor='#1A1A1A', edgecolor=C_MLP, lw=2)

# ═══════════════════════════════════════════════════════════════
# CNN PATH (top)
# ═══════════════════════════════════════════════════════════════

# Conv Block 1
draw_box(3.8, 7.5, 2.0, 0.9, C_CNN,
         'Conv2d 3×3\n19 → 48 ch\nBN + SiLU', fontsize=9)
draw_arrow(2.8, 7.9, 3.8, 7.9, color=C_CNN)

# Conv Block 2
draw_box(6.5, 7.5, 2.0, 0.9, C_CNN,
         'Conv2d 3×3\n48 → 96 ch\nBN + SiLU', fontsize=9)
draw_arrow(5.8, 7.9, 6.5, 7.9, color=C_CNN)

# Channel Attention
draw_box(9.2, 7.5, 2.0, 0.9, C_ATTN,
         'Channel\nAttention\n(SE Block)', fontsize=9)
draw_arrow(8.5, 7.9, 9.2, 7.9, color=C_ATTN)

# Spatial Attention
draw_box(11.9, 7.5, 2.0, 0.9, C_ATTN,
         'Spatial\nAttention\n(CBAM)', fontsize=9)
draw_arrow(11.2, 7.9, 11.9, 7.9, color=C_ATTN)

# AdaptiveAvgPool
draw_box(14.6, 7.5, 1.8, 0.9, C_CNN,
         'Adaptive\nAvgPool2d\n→ (B, 96)', fontsize=9, alpha=0.8)
draw_arrow(13.9, 7.9, 14.6, 7.9, color=C_CNN)

# ═══════════════════════════════════════════════════════════════
# TABULAR PATH (bottom)
# ═══════════════════════════════════════════════════════════════

# MLP Block 1
draw_box(3.8, 2.5, 2.0, 0.9, C_MLP,
         'Linear\n24 → 128\nBN + SiLU\nDrop(0.40)', fontsize=8.5)
draw_arrow(2.8, 2.9, 3.8, 2.9, color=C_MLP)

# MLP Block 2
draw_box(6.5, 2.5, 2.0, 0.9, C_MLP,
         'Linear\n128 → 64\nBN + SiLU\nDrop(0.30)', fontsize=8.5)
draw_arrow(5.8, 2.9, 6.5, 2.9, color=C_MLP)

# Output embedding
draw_box(9.2, 2.5, 1.8, 0.9, C_MLP,
         'Tab\nEmbedding\n→ (B, 64)', fontsize=9, alpha=0.8)
draw_arrow(8.5, 2.9, 9.2, 2.9, color=C_MLP)

# ═══════════════════════════════════════════════════════════════
# FUSION
# ═══════════════════════════════════════════════════════════════

# Concatenation
draw_box(14.6, 4.7, 1.8, 1.2, C_FUSION,
         'Concat\n96 + 64\n= 160', fontsize=10)

# Arrows from CNN and Tab paths to Concat
draw_curved_arrow(15.5, 7.5, 15.5, 5.9, color=C_CNN, rad=0.0)
draw_curved_arrow(11.0, 2.95, 14.6, 5.0, color=C_MLP, rad=-0.2)

# Fusion Head
draw_box(17.0, 4.9, 1.8, 0.8, C_FUSION,
         'Linear 160→64\nBN + SiLU\nDrop(0.15)', fontsize=9)
draw_arrow(16.4, 5.3, 17.0, 5.3, color=C_FUSION)

# ═══════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════
draw_box(17.0, 3.3, 1.8, 0.8, C_OUTPUT,
         'Linear 64→1\nSoftplus\n→ Rainfall (mm)', fontsize=9)
draw_arrow(17.9, 4.9, 17.9, 4.1, color=C_OUTPUT)

# ═══════════════════════════════════════════════════════════════
# ANNOTATIONS: Key Design Choices
# ═══════════════════════════════════════════════════════════════
annot_x = 0.3
annot_y = 0.7

annotations = [
    "• Weighted Loss: P90 penalty ×10, P95 ×20, P99 ×30 — forces extreme event detection",
    "• Weighted Oversampling: P90 ×10, P95 ×20, P99 ×40 — addresses class imbalance",
    "• Channel Attention learns which atmospheric variables matter most",
    "• Spatial Attention learns that center pixels (station location) matter more than edges",
    "• Trainable parameters: ~211K  |  LOYO-CV across 10 monsoon seasons (2015–2024)",
]

for i, txt in enumerate(annotations):
    ax.text(annot_x, 0.7 - i * 0.35, txt, fontsize=9, color='#333',
            fontfamily='serif', va='center')

# ── Divider line ──
ax.plot([0.2, 19.8], [1.5, 1.5], color='#DDD', linewidth=1, linestyle='--', zorder=1)

# ── Save ──
out = r"D:\NEW_NRSC\paper_figures\Fig_Architecture.png"
plt.tight_layout()
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"[OK] Saved -> {out}")
