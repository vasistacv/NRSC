import glob
import sys
import subprocess

print("Running scripts...")
for f in sorted(glob.glob('D:/NEW_NRSC/paper_figures/fig*.py')):
    print(f'Running {f}...')
    try:
        subprocess.run([sys.executable, f], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {f}: {e}")

print('Done!')
