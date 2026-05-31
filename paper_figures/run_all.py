"""Run ALL figure scripts (no shortcuts) then create PPT."""
import subprocess, sys, time

scripts = [
    ("fig_architecture.py", 60),
    ("fig_correlation.py", 300),
    ("fig1_study_area.py", 60),
    ("fig2_performance_bars.py", 60),
    ("fig3_station_heatmap.py", 60),
    ("fig4_radar_chart.py", 60),
    ("fig5_split_comparison.py", 60),
    ("fig6_spatial_map.py", 120),
    ("fig7_boxplot.py", 60),
    ("fig8_loyo_yearly.py", 60),
    ("fig9_table_figure.py", 60),
    ("fig10_improvement.py", 60),
    ("fig11_timeseries.py", 600),
    ("create_ppt.py", 60),
]

failed = []
for s, timeout in scripts:
    print(f"\n{'='*60}")
    print(f"  Running {s} (timeout={timeout}s) ...")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, s],
            cwd=r"D:\NEW_NRSC\paper_figures",
            capture_output=True, text=True, timeout=timeout
        )
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"  FAILED ({dt:.0f}s): {r.stderr[-500:]}")
            failed.append(s)
        else:
            print(f"  OK ({dt:.0f}s): {r.stdout.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s!")
        failed.append(s)

print(f"\n\n{'='*60}")
if failed:
    print(f"  FAILED SCRIPTS: {failed}")
else:
    print("  ALL SCRIPTS COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
