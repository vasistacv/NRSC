"""Deep Analysis of ECMWF GRIB Data - Surface and Pressure Level Fields"""
import sys
import os
import numpy as np

outpath = r'd:\NEW_NRSC\scripts\ecmwf_analysis_output.txt'

with open(outpath, 'w') as f:
    sys.stdout = f
    
    base = r'd:\NEW_NRSC\ecmwf_data'
    
    print('='*80)
    print('ECMWF GRIB DATA: COMPREHENSIVE ANALYSIS')
    print('='*80)
    
    # === DEEP DIVE: Surface File ===
    print('\n' + '='*80)
    print('DEEP DIVE: Surface File (2015-06)')
    print('='*80)
    
    import cfgrib
    
    sfc_path = os.path.join(base, '2015', 'ecmwf_2015_06.sfc.grib')
    datasets_sfc = cfgrib.open_datasets(sfc_path)
    print(f'\nNumber of sub-datasets in SFC: {len(datasets_sfc)}')
    
    for i, ds in enumerate(datasets_sfc):
        print(f'\n--- SFC Sub-dataset {i} ---')
        print(f'Dimensions: {dict(ds.dims)}')
        
        for c in ds.coords:
            coord = ds.coords[c]
            try:
                size = coord.size
                if size <= 10:
                    print(f'  Coord {c}: dtype={coord.dtype}, size={size}, values={coord.values}')
                else:
                    print(f'  Coord {c}: dtype={coord.dtype}, size={size}, first5={coord.values[:5]}, last5={coord.values[-5:]}')
            except:
                print(f'  Coord {c}: dtype={coord.dtype}, value={coord.values}')
        
        for v in ds.data_vars:
            var = ds[v]
            print(f'  Var: {v}')
            print(f'    long_name: {var.attrs.get("long_name", "N/A")}')
            print(f'    units: {var.attrs.get("units", "N/A")}')
            print(f'    GRIB_paramId: {var.attrs.get("GRIB_paramId", "N/A")}')
            print(f'    GRIB_shortName: {var.attrs.get("GRIB_shortName", "N/A")}')
            print(f'    GRIB_typeOfLevel: {var.attrs.get("GRIB_typeOfLevel", "N/A")}')
            print(f'    shape: {var.shape}')
            vals = var.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                print(f'    min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}')
        
        print(f'  Attrs: {dict(ds.attrs)}')
        ds.close()
    
    # === DEEP DIVE: Pressure Level File ===
    print('\n' + '='*80)
    print('DEEP DIVE: Pressure Level File (2015-06)')
    print('='*80)
    
    pl_path = os.path.join(base, '2015', 'ecmwf_2015_06.pl.grib')
    datasets_pl = cfgrib.open_datasets(pl_path)
    print(f'\nNumber of sub-datasets in PL: {len(datasets_pl)}')
    
    for i, ds in enumerate(datasets_pl):
        print(f'\n--- PL Sub-dataset {i} ---')
        print(f'Dimensions: {dict(ds.dims)}')
        
        for c in ds.coords:
            coord = ds.coords[c]
            try:
                size = coord.size
                if size <= 10:
                    print(f'  Coord {c}: dtype={coord.dtype}, size={size}, values={coord.values}')
                else:
                    print(f'  Coord {c}: dtype={coord.dtype}, size={size}, first5={coord.values[:5]}, last5={coord.values[-5:]}')
            except:
                print(f'  Coord {c}: dtype={coord.dtype}, value={coord.values}')
        
        for v in ds.data_vars:
            var = ds[v]
            print(f'  Var: {v}')
            print(f'    long_name: {var.attrs.get("long_name", "N/A")}')
            print(f'    units: {var.attrs.get("units", "N/A")}')
            print(f'    GRIB_paramId: {var.attrs.get("GRIB_paramId", "N/A")}')
            print(f'    GRIB_shortName: {var.attrs.get("GRIB_shortName", "N/A")}')
            print(f'    GRIB_typeOfLevel: {var.attrs.get("GRIB_typeOfLevel", "N/A")}')
            print(f'    shape: {var.shape}')
            vals = var.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                print(f'    min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}')
        
        ds.close()
    
    # === FILE INVENTORY ===
    print('\n' + '='*80)
    print('FILE INVENTORY: ALL YEARS')
    print('='*80)
    
    total_size = 0
    file_count = 0
    for yr in range(2015, 2025):
        yr_dir = os.path.join(base, str(yr))
        yr_size = 0
        yr_files = 0
        for mo in ['06', '07', '08', '09']:
            sfc = os.path.join(yr_dir, f'ecmwf_{yr}_{mo}.sfc.grib')
            pl = os.path.join(yr_dir, f'ecmwf_{yr}_{mo}.pl.grib')
            sfc_ok = os.path.exists(sfc)
            pl_ok = os.path.exists(pl)
            sfc_size = os.path.getsize(sfc) if sfc_ok else 0
            pl_size = os.path.getsize(pl) if pl_ok else 0
            status = 'OK' if (sfc_ok and pl_ok) else 'MISSING!'
            print(f'  {yr}-{mo}: SFC={sfc_size/(1024*1024):.1f}MB PL={pl_size/(1024*1024):.1f}MB [{status}]')
            yr_size += sfc_size + pl_size
            yr_files += (1 if sfc_ok else 0) + (1 if pl_ok else 0)
        
        print(f'  {yr} TOTAL: {yr_size/(1024*1024):.1f} MB, {yr_files} GRIB files')
        total_size += yr_size
        file_count += yr_files
    
    print(f'\nGRAND TOTAL: {total_size/(1024*1024*1024):.2f} GB, {file_count} GRIB files')
    
    # === Spot check time axes across years ===
    print('\n' + '='*80)
    print('TEMPORAL AXIS CHECK ACROSS ALL YEARS')
    print('='*80)
    
    for yr in range(2015, 2025):
        for mo in ['06', '07', '08', '09']:
            fpath = os.path.join(base, str(yr), f'ecmwf_{yr}_{mo}.sfc.grib')
            if os.path.exists(fpath):
                try:
                    dsets = cfgrib.open_datasets(fpath)
                    ds = dsets[0]
                    times = ds.coords['time'].values if 'time' in ds.coords else None
                    vt = ds.coords['valid_time'].values if 'valid_time' in ds.coords else None
                    step_val = ds.coords['step'].values if 'step' in ds.coords else None
                    
                    n_times = times.size if times is not None else 0
                    first_t = str(times.flat[0])[:19] if times is not None and times.size > 0 else 'N/A'
                    last_t = str(times.flat[-1])[:19] if times is not None and times.size > 1 else first_t
                    first_vt = str(vt.flat[0])[:19] if vt is not None and vt.size > 0 else 'N/A'
                    last_vt = str(vt.flat[-1])[:19] if vt is not None and vt.size > 1 else first_vt
                    
                    print(f'  {yr}-{mo}: n_times={n_times}, init={first_t}..{last_t}, valid={first_vt}..{last_vt}, step={step_val}')
                    for d in dsets:
                        d.close()
                except Exception as e:
                    print(f'  {yr}-{mo}: ERROR - {e}')
    
    print('\nECMWF ANALYSIS COMPLETE')

sys.stdout = sys.__stdout__
print(f'Output written to {outpath}')
