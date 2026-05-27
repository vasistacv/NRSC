"""
Fig 7: Station-wise LOOCV Performance Distribution
Nature-style 2x2 box plots comparing Model vs ECMWF across CSI and SEDI metrics.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
model_csi_p90  = [0.173, 0.237, 0.286, 0.145, 0.257, 0.182, 0.192]
ecmwf_csi_p90  = [0.014, 0.063, 0.108, 0.064, 0.166, 0.108, 0.091]

model_csi_p95  = [0.092, 0.166, 0.293, 0.061, 0.273, 0.150, 0.140]
ecmwf_csi_p95  = [0.000, 0.103, 0.100, 0.033, 0.108, 0.033, 0.050]

model_sedi_p90  = [0.265, 0.522, 0.630, 0.001, 0.652, 0.529, 0.320]
ecmwf_sedi_p90  = [-0.338, -0.082, 0.148, -0.018, 0.422, -0.010, 0.075]

model_sedi_p95  = [0.007, 0.146, 0.316, -0.305, 0.339, 0.084, -0.128]
ecmwf_sedi_p95  = [-0.342, 0.008, -0.242, -0.210, 0.007, -0.255, -0.239]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    'xtick.labelsize':    12,
    'ytick.labelsize':    11,
    'legend.fontsize':    11,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':       42,   # editable text in PDFs
    'ps.fonttype':        42,
})

COLOR_MODEL = '#2166AC'   # steel blue
COLOR_ECMWF = '#E66101'   # burnt orange
FILL_MODEL  = '#9ECAE1'   # light blue fill
FILL_ECMWF  = '#FDCDAC'   # light orange fill

panels = [
    ('CSI (P90)',  model_csi_p90,  ecmwf_csi_p90),
    ('CSI (P95)',  model_csi_p95,  ecmwf_csi_p95),
    ('SEDI (P90)', model_sedi_p90, ecmwf_sedi_p90),
    ('SEDI (P95)', model_sedi_p95, ecmwf_sedi_p95),
]
labels_panel = ['(a)', '(b)', '(c)', '(d)']

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (ax, (title, model_data, ecmwf_data)) in enumerate(zip(axes, panels)):

    positions = [1, 2]

    # --- box‑plot styling ---------------------------------------------------
    box_kw = dict(
        widths=0.45,
        patch_artist=True,
        showfliers=False,          # we'll overlay raw points instead
        whiskerprops=dict(linewidth=1.0, color='#333333'),
        capprops=dict(linewidth=1.0, color='#333333'),
        medianprops=dict(linewidth=1.8, color='#333333', zorder=5),
        boxprops=dict(linewidth=1.0),
    )

    bp_model = ax.boxplot(
        [model_data], positions=[1], **box_kw
    )
    bp_ecmwf = ax.boxplot(
        [ecmwf_data], positions=[2], **box_kw
    )

    # Fill colours
    for patch in bp_model['boxes']:
        patch.set_facecolor(FILL_MODEL)
        patch.set_edgecolor(COLOR_MODEL)
    for patch in bp_ecmwf['boxes']:
        patch.set_facecolor(FILL_ECMWF)
        patch.set_edgecolor(COLOR_ECMWF)

    # Whisker / cap colours
    for element in ['whiskers', 'caps']:
        for line in bp_model[element]:
            line.set_color(COLOR_MODEL)
        for line in bp_ecmwf[element]:
            line.set_color(COLOR_ECMWF)

    # --- scatter individual stations (jittered) ----------------------------
    np.random.seed(42)
    jitter_strength = 0.06
    jitter_m = np.random.uniform(-jitter_strength, jitter_strength, len(model_data))
    jitter_e = np.random.uniform(-jitter_strength, jitter_strength, len(ecmwf_data))

    ax.scatter(
        np.ones(len(model_data)) + jitter_m, model_data,
        s=36, color=COLOR_MODEL, edgecolors='white', linewidths=0.6,
        zorder=6, alpha=0.85, label='Proposed Model'
    )
    ax.scatter(
        np.full(len(ecmwf_data), 2) + jitter_e, ecmwf_data,
        s=36, color=COLOR_ECMWF, edgecolors='white', linewidths=0.6,
        zorder=6, alpha=0.85, label='ECMWF'
    )

    # --- mean marker (diamond) ---------------------------------------------
    mean_m = np.mean(model_data)
    mean_e = np.mean(ecmwf_data)
    ax.scatter([1], [mean_m], marker='D', s=50, color='white',
               edgecolors=COLOR_MODEL, linewidths=1.2, zorder=7)
    ax.scatter([2], [mean_e], marker='D', s=50, color='white',
               edgecolors=COLOR_ECMWF, linewidths=1.2, zorder=7)

    # --- axis formatting ---------------------------------------------------
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Proposed\nModel', 'ECMWF'])
    ax.set_xlim(0.4, 2.6)

    # y‑axis: metric name
    metric_name = title.split(' ')[0]   # CSI or SEDI
    ax.set_ylabel(metric_name)

    # Panel label
    ax.set_title(f'{labels_panel[idx]}  {title}', loc='left',
                 fontsize=16, pad=8)

    # Light horizontal grid
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, color='#888888')
    ax.set_axisbelow(True)

    # Bold box - all spines visible
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(2.5)

    # zero-line for SEDI panels
    if 'SEDI' in title:
        ax.axhline(0, color='#999999', linewidth=0.6, linestyle='-', zorder=1)

    # Legend only in first panel
    if idx == 0:
        ax.legend(loc='upper right', frameon=True, framealpha=0.9,
                  edgecolor='#cccccc', fancybox=False)

plt.tight_layout(w_pad=3.0, h_pad=3.0)

out_path = r'D:\NEW_NRSC\paper_figures\Fig7_Station_Distribution.png'
fig.savefig(out_path, dpi=300, facecolor='white')
plt.close(fig)
print(f'Saved -> {out_path}')
