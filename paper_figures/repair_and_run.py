import glob
import re
import sys
import subprocess

for f in glob.glob('D:/NEW_NRSC/paper_figures/fig*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove all fontweight keyword arguments to avoid duplicates and rely on rcParams
    content = re.sub(r',\s*fontweight=[\'"][a-zA-Z]+[\'"]', '', content)
    content = re.sub(r'fontweight=[\'"][a-zA-Z]+[\'"]\s*,?\s*', '', content)

    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print("Running scripts...")
for f in glob.glob('D:/NEW_NRSC/paper_figures/fig*.py'):
    print(f'Running {f}...')
    subprocess.run([sys.executable, f], check=True)
print('Done!')
