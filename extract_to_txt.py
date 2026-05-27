import json

json_path = r'D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json'
out_txt = r'D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\results_for_sir_v2.txt'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def format_station(station, metrics_dict, is_loyo=False, label_override=None):
    s = f"    Station: {label_override if label_override else station}\n"
    s += "    " + "-"*130 + "\n"
    s += "    {:<8} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}\n".format(
        "Source", "CSI_r", "POD_r", "FAR_r", "SEDI_r", "CSI_90", "POD_90", "FAR_90", "SEDI_90", "CSI_95", "POD_95", "FAR_95", "SEDI_95", "RMSE", "MAE", "Corr"
    )
    s += "    " + "-"*130 + "\n"
    
    if is_loyo:
        sources = ['model', 'ecmwf']
        d_srcs = {'model': data['loyo_model_mean'].get(station, {}), 'ecmwf': data['loyo_ecmwf_mean'].get(station, {})}
    else:
        sources = ['model', 'ecmwf']
        d_srcs = metrics_dict

    for src in sources:
        m = d_srcs.get(src, {})
        if not m: continue
        s += "    {:<8} | {:6.3f} {:6.3f} {:6.3f} {:6.3f} | {:6.3f} {:6.3f} {:6.3f} {:6.3f} | {:6.3f} {:6.3f} {:6.3f} {:6.3f} | {:6.1f} {:6.1f} {:6.3f}\n".format(
            src.upper(),
            m.get('CSI_rain', 0), m.get('POD_rain', 0), m.get('FAR_rain', 0), m.get('SEDI_rain', 0),
            m.get('CSI_p90', 0), m.get('POD_p90', 0), m.get('FAR_p90', 0), m.get('SEDI_p90', 0),
            m.get('CSI_p95', 0), m.get('POD_p95', 0), m.get('FAR_p95', 0), m.get('SEDI_p95', 0),
            m.get('RMSE', 0), m.get('MAE', 0), m.get('corr_rainy', 0)
        )
    s += "\n"
    return s

with open(out_txt, 'w', encoding='utf-8') as f:
    f.write("="*138 + "\n")
    f.write(" "*48 + "FINAL FULL EVALUATION RESULTS FOR SUPERVISOR\n")
    f.write("="*138 + "\n\n")

    # NEW: Grand Summary
    if 'loyo_model_mean' in data and 'ALL' in data['loyo_model_mean']:
        f.write("------------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("GRAND SUMMARY: OVERALL 10-YEAR AVERAGE FOR ALL STATIONS COMBINED (LOOCV)\n")
        f.write("------------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write(format_station('ALL', None, is_loyo=True, label_override="GLOBAL MEAN (All 10 Years + All Stations)"))
        f.write("\n\n")

    # 1. Temporal Split
    if 'temporal' in data:
        f.write("1. TEMPORAL SPLIT (Train 2015-2020, Test 2021-2024)\n")
        f.write("="*138 + "\n")
        stations = sorted(list(data['temporal'].keys()))
        if 'ALL' in stations: stations.insert(0, stations.pop(stations.index('ALL')))
        for st in stations:
            f.write(format_station(st, data['temporal'][st]))

    # 2. Reverse Split
    if 'reverse' in data:
        f.write("2. REVERSE TEMPORAL SPLIT (Train 2021-2024, Test 2015-2020)\n")
        f.write("="*138 + "\n")
        stations = sorted(list(data['reverse'].keys()))
        if 'ALL' in stations: stations.insert(0, stations.pop(stations.index('ALL')))
        for st in stations:
            f.write(format_station(st, data['reverse'][st]))

    # 3. Random Split
    if 'random' in data:
        f.write("3. RANDOM SPLIT (80% Train, 20% Test)\n")
        f.write("="*138 + "\n")
        stations = sorted(list(data['random'].keys()))
        if 'ALL' in stations: stations.insert(0, stations.pop(stations.index('ALL')))
        for st in stations:
            f.write(format_station(st, data['random'][st]))

    # 4. LOOCV (LOYO Mean)
    if 'loyo_model_mean' in data:
        f.write("4. LOOCV / LOYO MEAN (Average Performance over all 10 Years broken down by Station)\n")
        f.write("="*138 + "\n")
        stations = sorted(list(data['loyo_model_mean'].keys()))
        # Remove 'ALL' from this list since it's now explicitly in the Grand Summary
        if 'ALL' in stations: stations.remove('ALL')
        for st in stations:
            f.write(format_station(st, None, is_loyo=True))

print(f"Successfully updated text file at: {out_txt}")
