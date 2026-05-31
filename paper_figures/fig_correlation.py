"""
fig_correlation_heatmap.py
===========================
Publication-quality Feature Correlation Heatmap.
Shows correlation between all 13 features (derived from 7 atmospheric channels) and observed rainfall.
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PATH"] = r"D:\NEW_NRSC\.venv\Library\bin;" + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
})

sys.path.insert(0, str(Path(r"D:\NEW_NRSC\final_model_baseline\9x9_v3")))
import config
from dataset import RainfallDataBuilder

# ── Feature names ──
FEATURE_NAMES = [
    'ECMWF TP (mm)', 'TCWV', 'CAPE', 'D2M',
    'RH 850', 'RH 500',
    'W 500',
    'TP (log)', 'CAPE×Uplift', 'D2M Dev',
    'RH Diff',
    'CAPE×TCWV', 'TP×CAPE',
]

# ── Build data for ALL years ──
print("Building data for correlation analysis...")
builder = RainfallDataBuilder(window_size=9)
patches, tabular, targets = builder.build(list(range(2015, 2025)))

print(f"  Tabular shape: {tabular.shape}")
print(f"  Targets shape: {targets.shape}")

# ── Build DataFrame ──
n_feats = tabular.shape[1]
# Use provided names or generic ones
names = FEATURE_NAMES[:n_feats]
if len(names) < n_feats:
    names += [f'Feature_{i}' for i in range(len(names), n_feats)]

df = pd.DataFrame(tabular, columns=names)
df['Observed\nRainfall (mm)'] = targets

# ── Compute correlation matrix ──
corr = df.corr()

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                          gridspec_kw={'width_ratios': [3, 1]})

# Panel (a): Full correlation matrix
ax1 = axes[0]
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
            annot=False, ax=ax1)
ax1.set_title('(a) Feature Correlation Matrix (7 Channels → 13 Features)', fontsize=16, fontweight='bold', pad=12)
ax1.tick_params(axis='both', labelsize=9)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)

# Panel (b): Correlation with Rainfall (sorted bar chart)
ax2 = axes[1]
rainfall_corr = corr['Observed\nRainfall (mm)'].drop('Observed\nRainfall (mm)').sort_values()

colors = ['#C62828' if v < 0 else '#1565C0' for v in rainfall_corr.values]
bars = ax2.barh(range(len(rainfall_corr)), rainfall_corr.values, color=colors,
                edgecolor='white', linewidth=0.5, height=0.7)

ax2.set_yticks(range(len(rainfall_corr)))
ax2.set_yticklabels(rainfall_corr.index, fontsize=9)
ax2.set_xlabel('Correlation with Rainfall', fontsize=13, fontweight='bold')
ax2.set_title('(b) Feature–Rainfall\nCorrelation', fontsize=16, fontweight='bold', pad=12)
ax2.axvline(x=0, color='#333', linewidth=1, linestyle='-')
ax2.set_xlim(-0.5, 0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (v, name) in enumerate(zip(rainfall_corr.values, rainfall_corr.index)):
    ax2.text(v + 0.02 if v >= 0 else v - 0.02,
             i, f'{v:.2f}', va='center',
             ha='left' if v >= 0 else 'right',
             fontsize=8, fontweight='bold',
             color='#1565C0' if v >= 0 else '#C62828')

for sp in ax2.spines.values():
    sp.set_linewidth(2)

plt.tight_layout(w_pad=3)

out = r"D:\NEW_NRSC\paper_figures\Fig_Correlation_Heatmap.png"
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"\n[OK] Saved -> {out}")
