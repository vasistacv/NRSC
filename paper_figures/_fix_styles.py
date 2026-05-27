import re, glob

# Files to fix (NOT fig1 which is already rewritten, NOT fig11 already fixed)
files = [
    'D:/NEW_NRSC/paper_figures/fig2_performance_bars.py',
    'D:/NEW_NRSC/paper_figures/fig3_station_heatmap.py',
    'D:/NEW_NRSC/paper_figures/fig4_radar_chart.py',
    'D:/NEW_NRSC/paper_figures/fig5_split_comparison.py',
    'D:/NEW_NRSC/paper_figures/fig6_spatial_map.py',
    'D:/NEW_NRSC/paper_figures/fig7_boxplot.py',
    'D:/NEW_NRSC/paper_figures/fig8_loyo_yearly.py',
    'D:/NEW_NRSC/paper_figures/fig9_table_figure.py',
    'D:/NEW_NRSC/paper_figures/fig10_improvement.py',
]

for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            c = f.read()
    except:
        print(f'SKIP {fpath}')
        continue

    # 1. Fix rcParams values
    c = re.sub(r'"font\.size":\s*\d+', '"font.size": 14', c)
    c = re.sub(r"'font\.size':\s*\d+", "'font.size': 14", c)
    c = re.sub(r'"axes\.linewidth":\s*[\d.]+', '"axes.linewidth": 2.5', c)
    c = re.sub(r"'axes\.linewidth':\s*[\d.]+", "'axes.linewidth': 2.5", c)
    c = re.sub(r'"xtick\.major\.width":\s*[\d.]+', '"xtick.major.width": 2.5', c)
    c = re.sub(r"'xtick\.major\.width':\s*[\d.]+", "'xtick.major.width': 2.5", c)
    c = re.sub(r'"ytick\.major\.width":\s*[\d.]+', '"ytick.major.width": 2.5', c)
    c = re.sub(r"'ytick\.major\.width':\s*[\d.]+", "'ytick.major.width': 2.5", c)
    c = re.sub(r'"axes\.labelsize":\s*\d+', '"axes.labelsize": 16', c)
    c = re.sub(r"'axes\.labelsize':\s*\d+", "'axes.labelsize': 16", c)
    c = re.sub(r'"axes\.titlesize":\s*\d+', '"axes.titlesize": 18', c)
    c = re.sub(r"'axes\.titlesize':\s*\d+", "'axes.titlesize': 18", c)

    # 2. Fix fontsize=20 everywhere to appropriate sizes
    # ax.text (annotations in plots) -> 10
    c = re.sub(r'(ax\w*\.text\([^)]*?)fontsize=20', r'\g<1>fontsize=10', c)
    c = re.sub(r'(ax\w*\.text\([^)]*?)fontsize=18', r'\g<1>fontsize=10', c)
    
    # set_xticklabels / set_yticklabels -> 12
    c = re.sub(r'(set_[xy]ticklabels\([^)]*?)fontsize=20', r'\g<1>fontsize=12', c)
    c = re.sub(r'(set_[xy]ticklabels\([^)]*?)fontsize=18', r'\g<1>fontsize=12', c)
    
    # set_title -> 16
    c = re.sub(r'(\.set_title\([^)]*?)fontsize=20', r'\g<1>fontsize=16', c)
    c = re.sub(r'(\.set_title\([^)]*?)fontsize=22', r'\g<1>fontsize=16', c)
    
    # set_xlabel / set_ylabel -> 15
    c = re.sub(r'(set_[xy]label\([^)]*?)fontsize=20', r'\g<1>fontsize=15', c)
    
    # legend fontsize -> 11
    c = re.sub(r'(\.legend\([^)]*?)fontsize=20', r'\g<1>fontsize=11', c)
    
    # set_label (colorbar) -> 12
    c = re.sub(r'(\.set_label\([^)]*?)fontsize=20', r'\g<1>fontsize=12', c)
    
    # annotate -> 9
    c = re.sub(r'(\.annotate\([^)]*?)fontsize=20', r'\g<1>fontsize=9', c)
    
    # fig.text -> 11
    c = re.sub(r'(fig\.text\([^)]*?)fontsize=20', r'\g<1>fontsize=11', c)
    
    # suptitle -> 16
    c = re.sub(r'(\.suptitle\([^)]*?)fontsize=20', r'\g<1>fontsize=16', c)
    
    # title_fontsize=20 -> 12
    c = c.replace('title_fontsize=20', 'title_fontsize=12')
    
    # Remaining fontsize=20 -> 10
    c = c.replace('fontsize=20', 'fontsize=10')

    # 3. Remove spine.set_visible(False) lines
    c = re.sub(r".*spine.*set_visible\(False\).*\n", "", c)
    
    # 4. Fix spine linewidths to 2.5
    c = re.sub(r'(spine\.set_linewidth\()[\d.]+\)', r'\g<1>2.5)', c)
    c = re.sub(r"(sp\.set_linewidth\()[\d.]+\)", r"\g<1>2.5)", c)
    c = c.replace('set_linewidth(3.0)', 'set_linewidth(2.5)')
    c = c.replace('set_linewidth(4.0)', 'set_linewidth(2.5)')

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)
    print(f'Fixed {fpath}')

print('\nAll 9 scripts fixed!')
