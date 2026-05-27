import re, glob

# Add bold spine enforcement to all figure scripts
files = glob.glob('D:/NEW_NRSC/paper_figures/fig*.py')

for fpath in files:
    if '_fix' in fpath or '_style' in fpath:
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()

    # Check if there's a "# Remove top" or "# Remove top/right" comment with empty code after
    c = re.sub(r'# Remove top.*spines\s*\n\s*\n', 
               '# Bold box - all spines visible\n    for sp in ax.spines.values():\n        sp.set_visible(True)\n        sp.set_linewidth(2.5)\n\n', c)
    c = re.sub(r'# Remove top \u0026 right spines\s*\n\s*\n', 
               '# Bold box - all spines visible\n    for sp in ax.spines.values():\n        sp.set_visible(True)\n        sp.set_linewidth(2.5)\n\n', c)

    # For the radar chart (fig4), ensure the outer circle spine is thick
    if 'radar' in fpath.lower() or 'fig4' in fpath.lower():
        if 'ax.spines' not in c:
            c = c.replace("ax.set_facecolor(\"white\")", 
                          "ax.set_facecolor(\"white\")\nax.spines['polar'].set_linewidth(2.5)\nax.spines['polar'].set_visible(True)")

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)
    print(f'Spines fixed: {fpath}')

print('Done!')
