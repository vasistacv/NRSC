"""Explore both GFS and ECMWF data structures."""
import zipfile, netCDF4, cfgrib, xarray as xr
import numpy as np
from pathlib import Path

# ── GFS ──
print("=" * 50)
print("GFS STRUCTURE")
print("=" * 50)
zpath = r"D:\NEW_NRSC\Final_GFS_Data\Total_ppt\0_to_6\gfs.0p25.2015060118.f006.grib2.nc.zip"
with zipfile.ZipFile(zpath) as z:
    nc_name = z.namelist()[0]
    with z.open(nc_name) as f:
        ds = netCDF4.Dataset("in-mem", memory=f.read())
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        tp = ds.variables["A_PCP_L1_Accum_1"][:]
        print(f"Precip var: A_PCP_L1_Accum_1, units=kg m-2 (=mm)")
        print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}, step={np.diff(lats[:3])}")
        print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}, step={np.diff(lons[:3])}")
        print(f"Grid shape: {tp.shape}")
        print(f"Precip stats: min={tp.min():.2f}, max={tp.max():.2f}, mean={tp.mean():.2f}")
        ds.close()

# ── ECMWF ──
print("\n" + "=" * 50)
print("ECMWF STRUCTURE")
print("=" * 50)
ecmwf_dir = Path(r"D:\NEW_NRSC\ecmwf_data")
# Find a sample surface file
sfc_files = sorted(ecmwf_dir.glob("*sfc*.grib"))
if sfc_files:
    f = sfc_files[0]
    print(f"Sample file: {f.name}")
    ds = xr.open_dataset(str(f), engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coords: {list(ds.coords)}")
    print(f"tp units: {ds['tp'].attrs.get('units', 'N/A')}")
    print(f"tp stats: min={float(ds['tp'].min()):.6f}, max={float(ds['tp'].max()):.6f}")
    lats = ds.latitude.values
    lons = ds.longitude.values
    print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}, step={np.diff(lats[:3])}")
    print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}, step={np.diff(lons[:3])}")
    print(f"Grid shape: lat={len(lats)}, lon={len(lons)}")
else:
    print("No sfc grib files found.")

# ── Check ECMWF tp units (meters vs mm) ──
print("\nECMWF tp is in METERS. To convert to mm, multiply by 1000.")
print("GFS A_PCP_L1_Accum_1 is in kg/m2 which equals mm directly.")
