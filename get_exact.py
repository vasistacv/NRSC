import json
d=json.load(open(r"D:\NEW_NRSC\final_model_baseline\9x9_v3\final_ensemble\station_final_results.json"))
m=d["loyo_model_mean"]["ALL"]
t=d["temporal"]["ALL"]["model"]
r=d["reverse"]["ALL"]["model"]
rn=d["random"]["ALL"]["model"]
print("EXACT VALUES:")
print(f"  fig5 Reverse CSI_p95:  {r['CSI_p95']}")
print(f"  fig9 Temporal RMSE:    {t['RMSE']}")
print(f"  fig9 Temporal corr:    {t['corr_rainy']}")
print(f"  fig9 Reverse CSI_p95:  {r['CSI_p95']}")
print(f"  fig9 Reverse RMSE:     {r['RMSE']}")
print(f"  fig9 Reverse corr:     {r['corr_rainy']}")
print(f"  fig9 Random RMSE:      {rn['RMSE']}")
print(f"  fig9 Random corr:      {rn['corr_rainy']}")
print(f"  fig9 LOYO RMSE:        {m['RMSE']}")
print(f"  fig9 LOYO corr:        {m['corr_rainy']}")
