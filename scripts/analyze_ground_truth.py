"""Deep Analysis of Ground Truth Rainfall Data - writes to file"""
import pandas as pd
import numpy as np
import sys

outpath = r'd:\NEW_NRSC\scripts\gt_analysis_output.txt'
with open(outpath, 'w') as f:
    sys.stdout = f
    
    df = pd.read_csv(r'd:\NEW_NRSC\Final_ground_truth_data.csv')
    print('='*80)
    print('GROUND TRUTH DATA: COMPREHENSIVE ANALYSIS')
    print('='*80)
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'Dtypes:\n{df.dtypes}')
    print(f'\nNull counts:\n{df.isnull().sum()}')
    print(f'\nUnique Stations: {list(df.Station.unique())}  ({df.Station.nunique()} total)')
    print(f'Unique Sources: {list(df.Source.unique())}')
    print(f'Date range: {df.Date.min()} to {df.Date.max()}')

    df['Date'] = pd.to_datetime(df['Date'])
    print(f'\nYear range: {df.Date.dt.year.min()} to {df.Date.dt.year.max()}')
    print(f'Month coverage: {sorted(df.Date.dt.month.unique())}')

    print(f'\nRecords per year:')
    for yr, cnt in df.groupby(df.Date.dt.year).size().items():
        print(f'  {yr}: {cnt}')

    print(f'\nRecords per station:')
    for stn, cnt in df.groupby('Station').size().items():
        print(f'  {stn}: {cnt}')

    print(f'\nStation coordinates:')
    for stn in df.Station.unique():
        sub = df[df.Station==stn]
        print(f'  {stn}: Lat={sub.Lat.iloc[0]:.4f}, Lon={sub.Lon.iloc[0]:.4f}')

    print(f'\n{"="*80}')
    print('RAINFALL STATISTICS')
    print('='*80)
    print(df['Rainfall_mm'].describe())
    print(f'\nZero rainfall: {(df.Rainfall_mm==0).sum()}/{len(df)} = {(df.Rainfall_mm==0).mean()*100:.1f}%')
    print(f'Non-zero rainfall: {(df.Rainfall_mm>0).sum()} ({(df.Rainfall_mm>0).mean()*100:.1f}%)')

    for p in [50, 75, 90, 95, 99]:
        val = df.Rainfall_mm.quantile(p/100)
        nz = df.loc[df.Rainfall_mm>0, 'Rainfall_mm']
        val_nz = nz.quantile(p/100)
        print(f'  P{p}: {val:.1f} mm (all) | {val_nz:.1f} mm (non-zero)')

    print(f'\nMax rainfall: {df.Rainfall_mm.max()} mm')
    idx = df.Rainfall_mm.idxmax()
    print(f'Max record: Date={df.loc[idx,"Date"]}, Station={df.loc[idx,"Station"]}, Value={df.loc[idx,"Rainfall_mm"]}')

    print(f'\nTop 20 extreme rainfall events:')
    top20 = df.nlargest(20, 'Rainfall_mm')
    for _, row in top20.iterrows():
        print(f'  {row.Date.strftime("%Y-%m-%d")} | {row.Station:15s} | {row.Rainfall_mm:.1f} mm')

    print(f'\nExtreme event distribution:')
    for thr in [0.1, 2.5, 7.5, 35.5, 64.5, 115.6, 204.5]:
        cnt = (df.Rainfall_mm >= thr).sum()
        print(f'  >={thr}mm: {cnt} events ({cnt/len(df)*100:.3f}%) [IMD category threshold]')

    print(f'\nIMD rainfall categories:')
    cats = {
        'No rain (0)': (df.Rainfall_mm == 0).sum(),
        'Trace (0.1-2.4)': ((df.Rainfall_mm >= 0.1) & (df.Rainfall_mm < 2.5)).sum(),
        'Light (2.5-7.4)': ((df.Rainfall_mm >= 2.5) & (df.Rainfall_mm < 7.5)).sum(),
        'Moderate (7.5-35.4)': ((df.Rainfall_mm >= 7.5) & (df.Rainfall_mm < 35.5)).sum(),
        'Rather heavy (35.5-64.4)': ((df.Rainfall_mm >= 35.5) & (df.Rainfall_mm < 64.5)).sum(),
        'Heavy (64.5-115.5)': ((df.Rainfall_mm >= 64.5) & (df.Rainfall_mm < 115.6)).sum(),
        'Very heavy (115.6-204.4)': ((df.Rainfall_mm >= 115.6) & (df.Rainfall_mm < 204.5)).sum(),
        'Extremely heavy (>=204.5)': (df.Rainfall_mm >= 204.5).sum(),
    }
    for cat, cnt in cats.items():
        print(f'  {cat}: {cnt} ({cnt/len(df)*100:.2f}%)')

    print(f'\nPer-station stats:')
    for stn in sorted(df.Station.unique()):
        sub = df[df.Station==stn]
        print(f'  {stn}: mean={sub.Rainfall_mm.mean():.2f}, std={sub.Rainfall_mm.std():.2f}, max={sub.Rainfall_mm.max():.1f}, P90={sub.Rainfall_mm.quantile(0.9):.1f}, P95={sub.Rainfall_mm.quantile(0.95):.1f}, P99={sub.Rainfall_mm.quantile(0.99):.1f}, zero%={((sub.Rainfall_mm==0).mean()*100):.1f}%')

    print(f'\nPer-year stats:')
    for yr in sorted(df.Date.dt.year.unique()):
        sub = df[df.Date.dt.year==yr]
        print(f'  {yr}: mean={sub.Rainfall_mm.mean():.2f}, sum={sub.Rainfall_mm.sum():.0f}, max={sub.Rainfall_mm.max():.1f}, rain_events={(sub.Rainfall_mm>0).sum()}, zero%={((sub.Rainfall_mm==0).mean()*100):.1f}%')

    print(f'\nPer-month stats:')
    for m in sorted(df.Date.dt.month.unique()):
        sub = df[df.Date.dt.month==m]
        print(f'  Month {m}: mean={sub.Rainfall_mm.mean():.2f}, max={sub.Rainfall_mm.max():.1f}, zero%={((sub.Rainfall_mm==0).mean()*100):.1f}%, skewness={sub.Rainfall_mm.skew():.2f}')

    print(f'\nMissing station-year combinations:')
    found_missing = False
    for yr in sorted(df.Date.dt.year.unique()):
        for stn in sorted(df.Station.unique()):
            sub = df[(df.Date.dt.year==yr) & (df.Station==stn)]
            if len(sub) == 0:
                print(f'  MISSING: {yr} - {stn}')
                found_missing = True
    if not found_missing:
        print('  None missing at year level')

    print(f'\nDays per station per year:')
    pivot = df.groupby([df.Date.dt.year, 'Station']).size().unstack(fill_value=0)
    print(pivot.to_string())

    print(f'\nDistribution characteristics:')
    print(f'  Skewness (all): {df.Rainfall_mm.skew():.2f}')
    print(f'  Kurtosis (all): {df.Rainfall_mm.kurtosis():.2f}')
    nz = df.loc[df.Rainfall_mm>0, 'Rainfall_mm']
    print(f'  Skewness (non-zero): {nz.skew():.2f}')
    print(f'  Kurtosis (non-zero): {nz.kurtosis():.2f}')

    print(f'\nSpatial extent of stations:')
    print(f'  Lat range: {df.Lat.min():.4f} to {df.Lat.max():.4f} ({(df.Lat.max()-df.Lat.min())*111:.1f} km N-S)')
    print(f'  Lon range: {df.Lon.min():.4f} to {df.Lon.max():.4f} ({(df.Lon.max()-df.Lon.min())*111*np.cos(np.radians(17.2)):.1f} km E-W)')

    print(f'\nInter-station daily rainfall correlation:')
    piv = df.pivot_table(index='Date', columns='Station', values='Rainfall_mm')
    corr = piv.corr()
    print(corr.to_string(float_format='%.3f'))

    print(f'\nConsecutive dry day analysis:')
    for stn in sorted(df.Station.unique()):
        sub = df[df.Station==stn].sort_values('Date')
        dry = (sub.Rainfall_mm == 0).astype(int)
        groups = dry.diff().ne(0).cumsum()
        dry_spells = dry.groupby(groups).sum()
        dry_spells = dry_spells[dry_spells > 0]
        if len(dry_spells) > 0:
            print(f'  {stn}: max_dry_spell={dry_spells.max()} days, mean_dry_spell={dry_spells.mean():.1f}, n_spells={len(dry_spells)}')

    # Monthly mean per station
    print(f'\nMonthly mean rainfall per station:')
    monthly_stn = df.groupby([df.Date.dt.month, 'Station'])['Rainfall_mm'].mean().unstack()
    print(monthly_stn.to_string(float_format='%.2f'))

    print('\nGROUND TRUTH ANALYSIS COMPLETE')

sys.stdout = sys.__stdout__
print(f'Output written to {outpath}')
