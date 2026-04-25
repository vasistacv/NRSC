"""
eval_station_final.py — FINAL Station-Wise Evaluation
=======================================================
4 evaluations × 7 stations × Model vs ECMWF × ALL metrics

  A) Temporal:  Train 2015-2023, Test 2024
  B) Reverse:   Train 2018-2024, Test 2015-2017
  C) Random:    70/15/15 all years mixed
  D) LOYO:      Leave-One-Year-Out (10 years)
"""

import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

import config
from dataset import RainfallDataBuilder, Normaliser
from model import build_model
from metrics import evaluate

# ── helpers ─────────────────────────────────────────────────────────────

def extract_spatial_features(patches):
    B, C, H, W = patches.shape
    c = H // 2
    feats = [patches[:,:,c,c], patches.mean(axis=(2,3)),
             patches.reshape(B,C,-1).max(axis=2), patches.std(axis=(2,3)),
             patches[:,:,c,c] - patches.mean(axis=(2,3)),
             patches[:,:,c-1:c+2,c-1:c+2].mean(axis=(2,3))]
    dx = patches[:,:,c,min(c+1,W-1)] - patches[:,:,c,max(c-1,0)]
    dy = patches[:,:,min(c+1,H-1),c] - patches[:,:,max(c-1,0),c]
    feats.append(np.sqrt(dx**2 + dy**2))
    return np.hstack(feats)

def build_feat(patches, tab, norm, nn=None):
    p, t = norm.transform_patches(patches), norm.transform_tabular(tab)
    parts = [extract_spatial_features(p), t]
    if nn is not None: parts.append(nn.reshape(-1,1))
    return np.hstack(parts)

def nn_pred(model, patches, tab, norm):
    p = torch.from_numpy(norm.transform_patches(patches)).float()
    t = torch.from_numpy(norm.transform_tabular(tab)).float()
    loader = DataLoader(TensorDataset(p, t), batch_size=256, shuffle=False)
    out = []
    with torch.no_grad():
        for pb, tb in loader:
            out.append(model.predict(pb, tb).numpy())
    return np.concatenate(out)

MK = ["CSI_rain","POD_rain","FAR_rain","SEDI_rain",
      "CSI_p90","POD_p90","FAR_p90","SEDI_p90",
      "CSI_p95","POD_p95","FAR_p95","SEDI_p95",
      "RMSE","MAE","corr_rainy"]

def metrics(preds, tgt, th, p90):
    m = evaluate(preds, tgt, th, prefix="")
    corr = 0.0
    r = tgt >= 0.1
    if r.sum() > 2:
        c = np.corrcoef(preds[r], tgt[r])[0,1]
        corr = float(c) if not np.isnan(c) else 0.0
    o = {k: (corr if k=="corr_rainy" else m.get(k,0.0)) for k in MK}
    o["n"] = len(tgt); o["n_p90"] = int((tgt>=p90).sum())
    o["n_p95"] = int((tgt>=th["p95"]).sum())
    return o

# ── ensemble ────────────────────────────────────────────────────────────

def run_ensemble(X_tr, y_tr, X_vl, y_vl, X_te, nn_vl, nn_te, p90, p95, th):
    # Rain clf
    yr=(y_tr>=0.1).astype(int); yv=(y_vl>=0.1).astype(int)
    spw=max((yr==0).sum(),1)/max((yr==1).sum(),1)
    cr = xgb.XGBClassifier(objective="binary:logistic",learning_rate=0.05,max_depth=5,
        min_child_weight=10,subsample=0.8,colsample_bytree=0.7,gamma=1.0,reg_alpha=0.5,
        reg_lambda=2.0,n_estimators=1000,early_stopping_rounds=50,verbosity=0,n_jobs=-1,
        random_state=42,scale_pos_weight=spw)
    cr.fit(X_tr,yr,eval_set=[(X_vl,yv)],verbose=False)
    # Regression
    w=np.ones(len(y_tr),dtype=np.float32)
    w[(y_tr>=0.1)&(y_tr<p90)]=2; w[(y_tr>=p90)&(y_tr<p95)]=10; w[y_tr>=p95]=20
    rg = xgb.XGBRegressor(objective="reg:squarederror",learning_rate=0.03,max_depth=6,
        min_child_weight=8,subsample=0.8,colsample_bytree=0.65,gamma=0.8,reg_alpha=0.5,
        reg_lambda=2.0,n_estimators=2000,early_stopping_rounds=80,verbosity=0,n_jobs=-1,
        random_state=42)
    rg.fit(X_tr,y_tr,sample_weight=w,eval_set=[(X_vl,y_vl)],verbose=False)
    # Extreme clf
    ye=(y_tr>=p90).astype(int); yev=(y_vl>=p90).astype(int)
    spw2=max((ye==0).sum(),1)/max((ye==1).sum(),1)
    ce = xgb.XGBClassifier(objective="binary:logistic",learning_rate=0.03,max_depth=4,
        min_child_weight=5,subsample=0.8,colsample_bytree=0.7,gamma=1.5,reg_alpha=1.0,
        reg_lambda=3.0,n_estimators=1000,early_stopping_rounds=50,verbosity=0,n_jobs=-1,
        random_state=42,scale_pos_weight=spw2)
    ce.fit(X_tr,ye,eval_set=[(X_vl,yev)],verbose=False)
    # Isotonic
    iso = IsotonicRegression(y_min=0.0,out_of_bounds='clip')
    iso.fit(nn_vl, y_vl)
    # Grid search
    rv=cr.predict_proba(X_vl)[:,1]; xv=rg.predict(X_vl).clip(min=0)
    ev=ce.predict_proba(X_vl)[:,1]; nc=iso.predict(nn_vl)
    best_s,best_c=-999,None
    for wn in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        for uc in [False,True]:
            nv=nc if uc else nn_vl; en=wn*nv+(1-wn)*xv
            for rt in np.arange(0.30,0.60,0.03):
                rm=rv>=rt; g=np.zeros_like(en); g[rm]=en[rm]
                for eb in [0,0.3,0.5,0.7,1.0]:
                    f=g.copy()
                    if eb>0:
                        em=ev>=0.5; f[em]=f[em]*(1+eb*ev[em])
                    m=evaluate(f,y_vl,th,prefix="")
                    s=(m.get("CSI_rain",0)*1+m.get("CSI_p90",0)*3+m.get("CSI_p95",0)*2
                       +m.get("SEDI_p90",0)*1-m.get("FAR_p90",1)*2-m.get("FAR_rain",1)*0.5)
                    if s>best_s: best_s=s; best_c={"wn":wn,"rt":float(rt),"eb":eb,"uc":uc}
    # Apply test
    rte=cr.predict_proba(X_te)[:,1]; xte=rg.predict(X_te).clip(min=0)
    ete=ce.predict_proba(X_te)[:,1]
    nte=iso.predict(nn_te) if best_c["uc"] else nn_te
    en=best_c["wn"]*nte+(1-best_c["wn"])*xte
    rm=rte>=best_c["rt"]; f=np.zeros_like(en); f[rm]=en[rm]
    if best_c["eb"]>0:
        em=ete>=0.5; f[em]=f[em]*(1+best_c["eb"]*ete[em])
    return f

# ── station assignment ─────────────────────────────────────────────────

def assign_stations(targets, years_arr):
    gt=pd.read_csv(config.GROUND_TRUTH,parse_dates=["Date"])
    gt=gt.dropna(subset=["Rainfall_mm"])
    gt=gt[gt["Date"].dt.month.isin(config.MONSOON_MONTHS)]
    gt["year"]=gt["Date"].dt.year
    stations=[]; idx=0
    for year in range(2015,2025):
        n=(years_arr==year).sum()
        pot=[]
        for month in config.MONSOON_MONTHS:
            gm=gt[(gt["year"]==year)&(gt["Date"].dt.month==month)]
            for _,row in gm.iterrows():
                pot.append((row["Station"],float(row["Rainfall_mm"])))
        matched=0; pi=0
        while matched<n and pi<len(pot):
            stn,rain=pot[pi]
            if abs(targets[idx+matched]-rain)<0.01:
                stations.append(stn); matched+=1
            pi+=1
        while matched<n: stations.append("Unknown"); matched+=1
        idx+=n
    return np.array(stations)

# ── split runner ───────────────────────────────────────────────────────

def run_split(nn_model, P, T, Y, yrs, stns, tr_idx, vl_idx, te_idx, th, p90, p95, ustn, label):
    norm=Normaliser(); norm.fit(P[tr_idx],T[tr_idx])
    ntr=nn_pred(nn_model,P[tr_idx],T[tr_idx],norm)
    # 85/15 from train for XGB
    np.random.seed(99); perm=np.random.permutation(len(tr_idx))
    nt=int(0.85*len(tr_idx))
    xtr,xvl=tr_idx[perm[:nt]],tr_idx[perm[nt:]]
    nntr=nn_pred(nn_model,P[xtr],T[xtr],norm)
    nnvl=nn_pred(nn_model,P[xvl],T[xvl],norm)
    nnte=nn_pred(nn_model,P[te_idx],T[te_idx],norm)
    Xtr=build_feat(P[xtr],T[xtr],norm,nntr)
    Xvl=build_feat(P[xvl],T[xvl],norm,nnvl)
    Xte=build_feat(P[te_idx],T[te_idx],norm,nnte)
    print(f"  {label}: train={len(xtr)} val={len(xvl)} test={len(te_idx)}")
    mpred=run_ensemble(Xtr,Y[xtr],Xvl,Y[xvl],Xte,nnvl,nnte,p90,p95,th)
    ctr=P.shape[2]//2
    epred=P[te_idx,0,ctr,ctr].copy().clip(min=0)
    tgt=Y[te_idx]; tstn=stns[te_idx]
    res={"ALL":{"model":metrics(mpred,tgt,th,p90),"ecmwf":metrics(epred,tgt,th,p90)}}
    for s in ustn:
        sm=tstn==s
        if sm.sum()==0: continue
        res[s]={"model":metrics(mpred[sm],tgt[sm],th,p90),
                "ecmwf":metrics(epred[sm],tgt[sm],th,p90)}
    return res

def print_table(res, ustn, label):
    print(f"\n{'='*140}")
    print(f"  {label}")
    print(f"{'='*140}")
    hdr=(f"  {'Station':>15s} | {'CSI_r':>6s} {'POD_r':>6s} {'FAR_r':>6s}"
         f" | {'CSI_90':>6s} {'POD_90':>6s} {'FAR_90':>6s} {'SEDI90':>7s}"
         f" | {'CSI_95':>6s} {'POD_95':>6s} {'FAR_95':>6s} {'SEDI95':>7s}"
         f" | {'RMSE':>6s} {'MAE':>6s} {'corr':>6s} | {'n':>4s} {'P90':>3s} {'P95':>3s}")
    def row(name,m):
        print(f"  {name:>15s}"
              f" | {m['CSI_rain']:>6.3f} {m['POD_rain']:>6.3f} {m['FAR_rain']:>6.3f}"
              f" | {m['CSI_p90']:>6.3f} {m['POD_p90']:>6.3f} {m['FAR_p90']:>6.3f} {m['SEDI_p90']:>7.3f}"
              f" | {m['CSI_p95']:>6.3f} {m['POD_p95']:>6.3f} {m['FAR_p95']:>6.3f} {m['SEDI_p95']:>7.3f}"
              f" | {m['RMSE']:>6.1f} {m['MAE']:>6.1f} {m['corr_rainy']:>6.3f}"
              f" | {m['n']:>4d} {m['n_p90']:>3d} {m['n_p95']:>3d}")
    for src in ["MODEL","ECMWF RAW"]:
        k="model" if src=="MODEL" else "ecmwf"
        print(f"\n  --- {src} ---")
        print(hdr); print("  "+"-"*135)
        for s in ["ALL"]+list(ustn):
            if s in res: row(s, res[s][k])

def main():
    print("\n"+"="*60)
    print("  FINAL STATION-WISE EVALUATION — ALL SPLITS")
    print("="*60)

    # Load
    print("\n[1/3] Loading data...")
    builder=RainfallDataBuilder(window_size=config.DEFAULT_WINDOW)
    all_yrs=list(range(2015,2025))
    aP,aT,aY=[],[],[]
    yrs_arr=[]
    for yr in all_yrs:
        p,t,y=builder.build([yr])
        aP.append(p); aT.append(t); aY.append(y)
        yrs_arr.extend([yr]*len(y))
    P=np.concatenate(aP); T=np.concatenate(aT); Y=np.concatenate(aY)
    yrs_arr=np.array(yrs_arr)
    rainy=Y[Y>=0.1]
    p90=float(np.percentile(rainy,90)); p95=float(np.percentile(rainy,95))
    th={"p90":p90,"p95":p95,"p99":float(np.percentile(rainy,99))}
    print(f"  Total: {len(Y)} | P90={p90:.1f}mm | P95={p95:.1f}mm")

    print("\n[2/3] Stations...")
    stns=assign_stations(Y,yrs_arr)
    ustn=sorted(set(stns)-{"Unknown"})
    for s in ustn: print(f"    {s:20s}: {(stns==s).sum()}")

    print("\n[3/3] SmallNet...")
    nn=build_model(window_size=config.DEFAULT_WINDOW,n_channels=19,n_tabular=24)
    pts=sorted((config.OUTPUT_DIR/f"window_{config.DEFAULT_WINDOW}").glob("*.pt"))
    ck=torch.load(str(pts[-1]),map_location="cpu")
    nn.load_state_dict(ck["model"]); nn.eval()
    print(f"  Loaded: {pts[-1].name}")

    ALL={}

    # A) Temporal: Train 2015-2023, Test 2024
    print("\n  === A) TEMPORAL: 2015-2023 -> 2024 ===")
    tr=np.where(np.isin(yrs_arr,list(range(2015,2024))))[0]
    te=np.where(yrs_arr==2024)[0]
    r=run_split(nn,P,T,Y,yrs_arr,stns,tr,tr,te,th,p90,p95,ustn,"TEMPORAL")
    print_table(r,ustn,"A) TEMPORAL: Train 2015-2023 -> Test 2024")
    ALL["temporal"]=r

    # B) Reverse: Train 2018-2024, Test 2015-2017
    print("\n  === B) REVERSE: 2018-2024 -> 2015-2017 ===")
    tr=np.where(np.isin(yrs_arr,list(range(2018,2025))))[0]
    te=np.where(np.isin(yrs_arr,[2015,2016,2017]))[0]
    r=run_split(nn,P,T,Y,yrs_arr,stns,tr,tr,te,th,p90,p95,ustn,"REVERSE")
    print_table(r,ustn,"B) REVERSE: Train 2018-2024 -> Test 2015-2017")
    ALL["reverse"]=r

    # C) Random 70/15/15
    print("\n  === C) RANDOM 70/15/15 ===")
    np.random.seed(42); perm=np.random.permutation(len(Y))
    n_tr=int(0.70*len(Y)); n_vl=int(0.15*len(Y))
    tr_i=perm[:n_tr]; vl_i=perm[n_tr:n_tr+n_vl]; te_i=perm[n_tr+n_vl:]
    r=run_split(nn,P,T,Y,yrs_arr,stns,tr_i,tr_i,te_i,th,p90,p95,ustn,"RANDOM")
    print_table(r,ustn,"C) RANDOM 70/15/15 SPLIT")
    ALL["random"]=r

    # D) LOYO — per year with full station tables
    print("\n  === D) LOYO ===")
    loyo_model={s:{k:[] for k in MK} for s in ["ALL"]+list(ustn)}
    loyo_ecmwf={s:{k:[] for k in MK} for s in ["ALL"]+list(ustn)}
    loyo_all_years = {}
    for tyr in all_yrs:
        print(f"    LOYO test={tyr}...")
        tr=np.where(yrs_arr!=tyr)[0]
        te=np.where(yrs_arr==tyr)[0]
        r=run_split(nn,P,T,Y,yrs_arr,stns,tr,tr,te,th,p90,p95,ustn,f"LOYO-{tyr}")
        loyo_all_years[tyr] = r
        # Print full table for each year
        print_table(r, ustn, f"D) LOYO test={tyr} (train on {[y for y in all_yrs if y!=tyr]})")
        for s in ["ALL"]+list(ustn):
            if s in r:
                for k in MK:
                    loyo_model[s][k].append(r[s]["model"][k])
                    loyo_ecmwf[s][k].append(r[s]["ecmwf"][k])

    # LOYO mean table
    print(f"\n{'='*140}")
    print(f"  D-MEAN) LOYO MEAN (averaged across 10 years)")
    print(f"{'='*140}")
    hdr=(f"  {'Station':>15s} | {'CSI_r':>6s} {'POD_r':>6s} {'FAR_r':>6s}"
         f" | {'CSI_90':>6s} {'POD_90':>6s} {'FAR_90':>6s} {'SEDI90':>7s}"
         f" | {'CSI_95':>6s} {'POD_95':>6s} {'FAR_95':>6s} {'SEDI95':>7s}"
         f" | {'RMSE':>6s} {'MAE':>6s} {'corr':>6s}")
    for src,data in [("MODEL",loyo_model),("ECMWF RAW",loyo_ecmwf)]:
        print(f"\n  --- {src} ---")
        print(hdr); print("  "+"-"*135)
        for s in ["ALL"]+list(ustn):
            if not data[s]["CSI_rain"]: continue
            mm={k:np.mean(data[s][k]) for k in MK}
            print(f"  {s:>15s}"
                  f" | {mm['CSI_rain']:>6.3f} {mm['POD_rain']:>6.3f} {mm['FAR_rain']:>6.3f}"
                  f" | {mm['CSI_p90']:>6.3f} {mm['POD_p90']:>6.3f} {mm['FAR_p90']:>6.3f} {mm['SEDI_p90']:>7.3f}"
                  f" | {mm['CSI_p95']:>6.3f} {mm['POD_p95']:>6.3f} {mm['FAR_p95']:>6.3f} {mm['SEDI_p95']:>7.3f}"
                  f" | {mm['RMSE']:>6.1f} {mm['MAE']:>6.1f} {mm['corr_rainy']:>6.3f}")

    ALL["loyo_per_year"] = loyo_all_years
    ALL["loyo_model_mean"]={s:{k:float(np.mean(loyo_model[s][k])) for k in MK}
                       for s in ["ALL"]+list(ustn) if loyo_model[s]["CSI_rain"]}
    ALL["loyo_ecmwf_mean"]={s:{k:float(np.mean(loyo_ecmwf[s][k])) for k in MK}
                       for s in ["ALL"]+list(ustn) if loyo_ecmwf[s]["CSI_rain"]}

    out=config.OUTPUT_DIR/"final_ensemble"
    out.mkdir(parents=True,exist_ok=True)
    def conv(o):
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,np.ndarray):return o.tolist()
        return o
    with open(out/"station_final_results.json","w") as f:
        json.dump(ALL,f,indent=2,default=conv)
    print(f"\n  Saved: {out/'station_final_results.json'}")

if __name__=="__main__":
    main()
