import json
d = json.load(open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json"))

m = d["loyo_model_mean"]["ALL"]
e = d["loyo_ecmwf_mean"]["ALL"]
print("=== LOYO MEAN MODEL ===")
for k in ["CSI_rain","POD_rain","FAR_rain","SEDI_rain","CSI_p90","POD_p90","FAR_p90","SEDI_p90","CSI_p95","POD_p95","FAR_p95","SEDI_p95","RMSE","MAE","corr_rainy"]:
    print(f"  {k}: {m[k]}")
print("\n=== LOYO MEAN ECMWF ===")
for k in ["CSI_rain","POD_rain","FAR_rain","SEDI_rain","CSI_p90","POD_p90","FAR_p90","SEDI_p90","CSI_p95","POD_p95","FAR_p95","SEDI_p95","RMSE","MAE","corr_rainy"]:
    print(f"  {k}: {e[k]}")

for split in ["temporal","reverse","random"]:
    sm = d[split]["ALL"]["model"]
    se = d[split]["ALL"]["ecmwf"]
    print(f"\n=== {split.upper()} MODEL ===")
    for k in ["CSI_rain","CSI_p90","CSI_p95","POD_p90","SEDI_p90","SEDI_p95","RMSE","corr_rainy"]:
        print(f"  {k}: {sm[k]}")
    print(f"=== {split.upper()} ECMWF ===")
    for k in ["CSI_rain","CSI_p90","CSI_p95"]:
        print(f"  {k}: {se[k]}")

# Improvement calculations for fig10
print("\n=== IMPROVEMENT vs ECMWF (LOYO) ===")
for k in ["CSI_rain","SEDI_rain","CSI_p90","POD_p90","SEDI_p90","CSI_p95","POD_p95","SEDI_p95","corr_rainy"]:
    vm = m[k]; ve = e[k]
    if ve != 0:
        pct = (vm - ve) / abs(ve) * 100
    else:
        pct = 99999
    print(f"  {k}: model={vm:.4f} ecmwf={ve:.4f} improvement={pct:+.1f}%")

# Per-station
print("\n=== LOYO PER STATION MODEL ===")
for stn in ["Chevella","Hayathnagar","Ibrahimpatnam","Kondurg","Maheshwaram","Saroornagar","Yacharam"]:
    sm = d["loyo_model_mean"][stn]
    se = d["loyo_ecmwf_mean"][stn]
    print(f"  {stn}: CSI_r={sm['CSI_rain']:.3f} CSI90={sm['CSI_p90']:.3f} POD90={sm['POD_p90']:.3f} SEDI90={sm['SEDI_p90']:.3f} CSI95={sm['CSI_p95']:.3f} RMSE={sm['RMSE']:.1f} corr={sm['corr_rainy']:.3f}")
    print(f"    ecmwf: CSI_r={se['CSI_rain']:.3f} CSI90={se['CSI_p90']:.3f} SEDI90={se['SEDI_p90']:.3f} corr={se['corr_rainy']:.3f}")
