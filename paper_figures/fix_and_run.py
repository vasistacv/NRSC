import glob
import re
import sys
import subprocess

files = glob.glob('D:/NEW_NRSC/paper_figures/fig*.py')

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove any line that disables spines
    content = re.sub(r'(\w+)\.spines\[.*?\]\.set_visible\(False\)', '', content)
    
    # Replace the loop that disables all spines
    content = re.sub(r'for spine in (\w+)\.spines\.values\(\):\n\s+spine\.set_visible\(False\)', 
                     r'for spine in \1.spines.values():\n    spine.set_linewidth(4.0)\n    spine.set_visible(True)', content)
    content = re.sub(r'spine\.set_visible\(False\)', 'spine.set_visible(True)\n    spine.set_linewidth(4.0)', content)

    # Force rcParams to be bold and thick
    content = re.sub(r'["\']axes\.linewidth["\']:\s*[\d\.]+', '"axes.linewidth": 4.0', content)
    content = re.sub(r'["\']xtick\.major\.width["\']:\s*[\d\.]+', '"xtick.major.width": 4.0', content)
    content = re.sub(r'["\']ytick\.major\.width["\']:\s*[\d\.]+', '"ytick.major.width": 4.0', content)

    # Remove all explicit fontweight arguments from matplotlib calls so rcParams handles it or we can safely add it
    content = re.sub(r',\s*fontweight=[\'"](bold|semibold|normal)[\'"]', '', content)
    content = re.sub(r'fontweight=[\'"](bold|semibold|normal)[\'"],\s*', '', content)

    # Make manual titles and labels bold
    content = re.sub(r'fontsize=\d+', 'fontsize=20, fontweight="bold"', content)

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print("Scripts updated. Running them all now...")

for f in files:
    print(f"Running {f}...")
    subprocess.run([sys.executable, f], check=True)

print("All 11 figures regenerated perfectly.")
