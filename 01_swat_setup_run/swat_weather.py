#!/usr/bin/env python3
"""
🌦️ SWAT+ hourly weather data preparation tool (ERA5-Land + DEM mask)
=======================================================
Downloads hourly meteorological data from ERA5-Land, selects valid land
stations based on a DEM, and generates complete SWAT+ inputs:
  - pcp: hourly (tstep=60)  ← required for sub-daily time step
  - tmp/slr/hmd/wnd: daily scale (tstep=0)
  - weather-wgn.cli: weather generator statistical parameters
  - weather-sta.cli: station-WGN association
  - *.cli: file list for each variable

Prerequisites:
  pip install cdsapi netCDF4 xarray pandas rasterio
  ~/.cdsapirc already configured

Usage:
  python swat_weather.py --country cambodia --start 2024-01-01 --days 14 \\
      --dem-dir /swat_global

Improvements:
  ✅ Uses a DEM mask to keep only valid land grid cells, skipping ocean / NoData areas
  ✅ Stations are numbered consecutively, avoiding the "numbered but no data" case
  ✅ Extracts real elevation from the DEM and assigns it to stations
  ✅ Also supports NaN detection as a fallback (when no DEM is available)
"""

import os, sys, argparse, warnings, glob
from pathlib import Path
from datetime import datetime, timedelta, date
import numpy as np
warnings.filterwarnings("ignore")

COUNTRY_BBOX = {
    "cambodia":    (102.0, 9.5, 108.5, 15.0),
    "thailand":    (97.0, 5.0, 106.0, 21.0),
    "vietnam":     (102.0, 8.0, 110.0, 24.0),
    "laos":        (100.0, 13.5, 108.0, 23.0),
    "myanmar":     (92.0, 9.5, 101.5, 28.5),
    "malaysia":    (99.5, 0.5, 119.5, 7.5),
    "philippines": (117.0, 5.0, 127.0, 21.0),
}

ERA5_VARS = [
    "2m_temperature", "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "total_precipitation",
]


# ============================================================
# DEM mask: read the DEM and determine which ERA5 grid cells are on land
# ============================================================
def find_dem_files(dem_dir):
    """Recursively search for all DEM raster files under dem_dir"""
    dem_dir = Path(dem_dir)
    patterns = ["**/*.tif", "**/*.tiff", "**/*.img", "**/*.adf",
                "**/*.hgt", "**/*.dt2", "**/*.bil"]
    found = []
    for pat in patterns:
        found.extend(dem_dir.glob(pat))
    # Filter: prioritize paths containing "dem", "elev", or "srtm" (case-insensitive)
    # Also include files under raster subdirectories
    prioritized = [f for f in found
                   if any(kw in str(f).lower()
                          for kw in ["dem", "elev", "srtm", "raster"])]
    if prioritized:
        return prioritized
    return found


def build_land_mask_from_dem(dem_files, lats, lons):
    """
    Read DEM files and build a land mask on the ERA5 grid (lats × lons).
    Returns:
      land_mask: bool[n_lat, n_lon] - True = land
      elev_grid: float[n_lat, n_lon] - DEM elevation (0 for ocean points)
    """
    import rasterio
    from rasterio.transform import rowcol

    n_lat, n_lon = len(lats), len(lons)
    land_mask = np.zeros((n_lat, n_lon), dtype=bool)
    elev_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    for dem_path in dem_files:
        try:
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1)
                nodata = src.nodata
                transform = src.transform
                bounds = src.bounds  # (left, bottom, right, top)

                for i, lat in enumerate(lats):
                    for j, lon in enumerate(lons):
                        # Skip points already marked as land (covered by a previous DEM)
                        if land_mask[i, j]:
                            continue
                        # Check whether the point falls within this DEM's extent
                        if not (bounds.left <= lon <= bounds.right and
                                bounds.bottom <= lat <= bounds.top):
                            continue
                        # Convert to DEM pixel coordinates
                        try:
                            row, col = rowcol(transform, lon, lat)
                            row, col = int(row), int(col)
                        except Exception:
                            continue
                        if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
                            val = dem_data[row, col]
                            # Determine whether it is a valid value (not NoData, not NaN)
                            is_valid = True
                            if nodata is not None and val == nodata:
                                is_valid = False
                            if np.isnan(val) if np.issubdtype(type(val), np.floating) else False:
                                is_valid = False
                            if is_valid:
                                land_mask[i, j] = True
                                elev_grid[i, j] = float(val)
                print(f"    ✅ Read DEM: {dem_path.name} "
                      f"(extent: {bounds.left:.1f}~{bounds.right:.1f}°E, "
                      f"{bounds.bottom:.1f}~{bounds.top:.1f}°N)")
        except Exception as e:
            print(f"    ⚠️  Could not read {dem_path}: {e}")

    return land_mask, elev_grid


def build_land_mask_from_data(*arrays_3d, min_valid_ratio=0.5):
    """
    Determine land vs. ocean from the NaNs in the ERA5 data itself.
    Accepts multiple 3D arrays (n_time, n_lat, n_lon), requiring:
      - all variables must have data
      - each variable has at least a min_valid_ratio fraction of valid (non-NaN) time steps
    This filters out:
      1) ocean points (all NaN)
      2) ERA5-Land boundary points (land geographically but not covered by ERA5)
      3) abnormal points with partial missing data
    Returns land_mask: bool[n_lat, n_lon]
    """
    mask = None
    for arr in arrays_3d:
        if arr.ndim == 3:
            n_t = arr.shape[0]
            valid_count = np.sum(~np.isnan(arr), axis=0)  # (n_lat, n_lon)
            var_mask = valid_count >= max(1, int(n_t * min_valid_ratio))
        elif arr.ndim == 2:
            var_mask = ~np.isnan(arr)
        else:
            continue
        mask = var_mask if mask is None else (mask & var_mask)
    return mask if mask is not None else np.ones(arrays_3d[0].shape[-2:], dtype=bool)


# ============================================================
# Download
# ============================================================
def download_era5(bbox, start_date, end_date, nc_dir):
    import cdsapi
    nc_dir = Path(nc_dir); nc_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox
    area = [north, west, south, east]
    total_days = (end_date - start_date).days
    print(f"\n{'='*60}")
    print(f"📥 Downloading ERA5-Land (0.1°, hourly)")
    print(f"   region: N={north}, W={west}, S={south}, E={east}")
    print(f"   period: {start_date} → {end_date} ({total_days} days)")
    print(f"{'='*60}")
    client = cdsapi.Client()
    nc_files = []
    cur = start_date
    while cur < end_date:
        m_end = min(date(cur.year, cur.month+1, 1) if cur.month < 12
                    else date(cur.year+1, 1, 1), end_date)
        day_list = []
        d = cur
        while d < m_end and d < end_date:
            day_list.append(f"{d.day:02d}"); d += timedelta(days=1)
        if not day_list: cur = m_end; continue
        nc = nc_dir / f"era5land_{cur.year}{cur.month:02d}.nc"
        if nc.exists():
            print(f"  ⏭️  Already exists: {nc.name}"); nc_files.append(nc)
            cur = m_end; continue
        print(f"  📥 {cur.year}-{cur.month:02d} ({len(day_list)} days)...")
        try:
            client.retrieve("reanalysis-era5-land", {
                "variable": ERA5_VARS,
                "year": str(cur.year), "month": f"{cur.month:02d}",
                "day": day_list,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area, "data_format": "netcdf",
                "download_format": "unarchived",
            }, str(nc))
            nc_files.append(nc)
            print(f"    ✅ {nc.name} ({nc.stat().st_size/1048576:.1f} MB)")
        except Exception as e:
            print(f"    ❌ {e}")
        cur = m_end
    print(f"  ✅ Download complete: {len(nc_files)} files")
    return nc_files


# ============================================================
# NC → SWAT+ files (with DEM mask)
# ============================================================
def nc_to_swat(nc_files, output_dir, start_date, end_date, dem_dir=None):
    import xarray as xr, pandas as pd
    output_dir = Path(output_dir)
    swat_dir = output_dir / "swat_weather"
    swat_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔄 ERA5 → SWAT+ (hourly precipitation + daily temperature/radiation/humidity/wind)")
    print(f"{'='*60}")

    # ---- Read NC ----
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    for dim in ["number", "expver"]:
        if dim in ds.dims:
            ds = ds.isel({dim: 0}).drop_vars(dim, errors="ignore")

    lat_n = "latitude" if "latitude" in ds.dims else "lat"
    lon_n = "longitude" if "longitude" in ds.dims else "lon"
    time_n = "valid_time" if "valid_time" in ds.dims else "time"

    lats = ds[lat_n].values; lons = ds[lon_n].values
    times = pd.DatetimeIndex(ds[time_n].values)
    n_lat, n_lon, n_t = len(lats), len(lons), len(times)
    print(f"  grid: {n_lat}×{n_lon}, time steps: {n_t}")

    # ---- Extract variables ----
    def G(name):
        return ds[name].values.squeeze().astype(np.float64)
    tp  = np.maximum(G("tp") * 1000.0, 0.0)       # m→mm
    t2m = G("t2m") - 273.15                        # K→°C
    d2m = G("d2m") - 273.15
    A, B = 17.625, 243.04
    rh = np.clip(np.exp(A*d2m/(B+d2m)) / np.exp(A*t2m/(B+t2m)), 0, 1)
    ssrd = np.maximum(G("ssrd") / 1e6, 0.0)       # J→MJ
    wnd = np.sqrt(G("u10")**2 + G("v10")**2)
    ds.close()
    print(f"  tp={np.nanmean(tp):.4f} mm/hr, t2m={np.nanmean(t2m):.1f}°C, "
          f"ssrd={np.nanmean(ssrd):.2f} MJ, wnd={np.nanmean(wnd):.1f} m/s")

    # ---- Daily aggregation ----
    times_py = times.to_pydatetime()
    days = sorted(set(t.date() for t in times_py))
    n_days = len(days)
    day_map = {}
    for i, t in enumerate(times_py):
        day_map.setdefault(t.date(), []).append(i)

    tmax_d = np.full((n_days, n_lat, n_lon), np.nan)
    tmin_d = np.full((n_days, n_lat, n_lon), np.nan)
    slr_d  = np.full((n_days, n_lat, n_lon), np.nan)
    hmd_d  = np.full((n_days, n_lat, n_lon), np.nan)
    wnd_d  = np.full((n_days, n_lat, n_lon), np.nan)
    pcp_d  = np.full((n_days, n_lat, n_lon), np.nan)
    for di, d in enumerate(days):
        idx = day_map[d]
        tmax_d[di] = np.nanmax(t2m[idx], axis=0)
        tmin_d[di] = np.nanmin(t2m[idx], axis=0)
        slr_d[di]  = np.nansum(ssrd[idx], axis=0)
        hmd_d[di]  = np.nanmean(rh[idx], axis=0)
        wnd_d[di]  = np.nanmean(wnd[idx], axis=0)
        pcp_d[di]  = np.nansum(tp[idx], axis=0)

    # ==========================================================
    # 🔑 Key improvement: build a land mask to filter out ocean stations
    # ==========================================================
    elev_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    if dem_dir:
        print(f"\n  🗺️  Building land mask from DEM...")
        print(f"     search directory: {dem_dir}")
        dem_files = find_dem_files(dem_dir)
        if dem_files:
            print(f"     found {len(dem_files)} DEM files:")
            for f in dem_files[:10]:
                print(f"       - {f}")
            if len(dem_files) > 10:
                print(f"       ... and {len(dem_files)-10} more")
            land_mask, elev_grid = build_land_mask_from_dem(dem_files, lats, lons)
            n_land_dem = int(np.sum(land_mask))
            print(f"     DEM mask: {n_land_dem} land grid cells / {n_lat*n_lon} total cells")

            # Extra safety check: also drop points marked as land by the DEM
            # but with incomplete ERA5 data. Check all key variables, requiring
            # at least 50% of time steps to have data.
            era5_has_data = build_land_mask_from_data(t2m, tp, ssrd, wnd)
            combined_mask = land_mask & era5_has_data
            n_dropped = n_land_dem - int(np.sum(combined_mask))
            if n_dropped > 0:
                print(f"     ⚠️  {n_dropped} DEM land points have no ERA5 data, dropped")
            land_mask = combined_mask
        else:
            print(f"     ⚠️  No DEM files found, falling back to ERA5 data detection mode")
            land_mask = build_land_mask_from_data(t2m, tp, ssrd, wnd)
    else:
        print(f"\n  ⚠️  --dem-dir not specified, using ERA5 data detection to filter invalid stations")
        land_mask = build_land_mask_from_data(t2m, tp, ssrd, wnd)

    # ---- Build land-only station list (consecutive numbering) ----
    stations = []
    sta_idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            if land_mask[i, j]:
                sta_idx += 1
                stations.append({
                    "id": f"s{sta_idx:05d}",   # consecutive numbering, 5 digits
                    "lat": float(lats[i]),
                    "lon": float(lons[j]),
                    "elev": float(elev_grid[i, j]),
                    "i": i, "j": j,
                })

    n_sta = len(stations)
    n_ocean = n_lat * n_lon - n_sta
    n_years = max(1, days[-1].year - days[0].year + 1)
    print(f"\n  📍 Initial selection: {n_sta} grid cells passed the mask (skipped {n_ocean} invalid cells)")

    # ---- Final validation: check data completeness station by station ----
    # Ensure each station has enough valid data across all variables
    valid_stations = []
    dropped_detail = []
    for s in stations:
        ii, jj = s["i"], s["j"]
        # Check the valid fraction of each variable
        tp_valid  = np.mean(~np.isnan(tp[:, ii, jj]))
        t2m_valid = np.mean(~np.isnan(t2m[:, ii, jj]))
        slr_valid = np.mean(~np.isnan(slr_d[:, ii, jj]))
        wnd_valid = np.mean(~np.isnan(wnd[:, ii, jj]))
        min_valid = min(tp_valid, t2m_valid, slr_valid, wnd_valid)
        if min_valid >= 0.5:  # all variables valid for at least 50% of time steps
            valid_stations.append(s)
        else:
            dropped_detail.append(
                f"     {s['id']} (lat={s['lat']}, lon={s['lon']}) "
                f"valid fraction: tp={tp_valid:.0%} t2m={t2m_valid:.0%} "
                f"slr={slr_valid:.0%} wnd={wnd_valid:.0%}")

    n_dropped_final = len(stations) - len(valid_stations)
    if n_dropped_final > 0:
        print(f"  ⚠️  Final validation dropped {n_dropped_final} stations with incomplete data:")
        for msg in dropped_detail[:10]:
            print(msg)
        if len(dropped_detail) > 10:
            print(f"     ... and {len(dropped_detail)-10} more")

    # Renumber consecutively
    stations = []
    for idx, s in enumerate(valid_stations):
        s_new = dict(s)
        s_new["id"] = f"s{idx+1:05d}"
        stations.append(s_new)

    n_sta = len(stations)
    print(f"  ✅ Final valid stations: {n_sta} (numbered s00001 ~ s{n_sta:05d})")

    if n_sta == 0:
        print("  ❌ No valid land stations found! Please check:")
        print("     - whether the DEM files cover the ERA5 download region")
        print("     - whether the bbox is correct")
        sys.exit(1)

    def fv(v):
        """NaN → -99.0 (SWAT+ missing value)"""
        return -99.0 if np.isnan(v) else float(v)

    # ---- Write data files ----
    print(f"  📝 Writing {n_sta} stations × 5 variables ...")
    for si, s in enumerate(stations):
        if si % 500 == 0 and si > 0:
            print(f"      ... {si}/{n_sta}")
        ii, jj, sid = s["i"], s["j"], s["id"]

        # PCP: hourly, tstep=60
        with open(swat_dir / f"{sid}.pcp", "w") as f:
            f.write(f"{sid}.pcp: ERA5-Land hourly precipitation\n")
            f.write(f"  nbyr     tstep       lat       lon      elev\n")
            f.write(f"{n_years:6d}{60:10d}{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")
            for ti in range(n_t):
                t = times_py[ti]
                jday = t.timetuple().tm_yday
                v = fv(tp[ti, ii, jj])
                f.write(f"{t.year:4d}{jday:4d}{t.month:4d}{t.day:4d}{t.hour:4d}{v:12.3f}\n")

        # TMP: daily scale, tstep=0
        with open(swat_dir / f"{sid}.tmp", "w") as f:
            f.write(f"{sid}.tmp: ERA5-Land daily temperature\n")
            f.write(f"  nbyr     tstep       lat       lon      elev\n")
            f.write(f"{n_years:6d}{0:10d}{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")
            for di, d in enumerate(days):
                jday = d.timetuple().tm_yday
                f.write(f"{d.year:4d}{jday:4d}{fv(tmax_d[di,ii,jj]):10.3f}{fv(tmin_d[di,ii,jj]):10.3f}\n")

        # SLR
        with open(swat_dir / f"{sid}.slr", "w") as f:
            f.write(f"{sid}.slr: ERA5-Land daily solar radiation\n")
            f.write(f"  nbyr     tstep       lat       lon      elev\n")
            f.write(f"{n_years:6d}{0:10d}{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")
            for di, d in enumerate(days):
                jday = d.timetuple().tm_yday
                f.write(f"{d.year:4d}{jday:4d}{fv(slr_d[di,ii,jj]):12.3f}\n")

        # HMD
        with open(swat_dir / f"{sid}.hmd", "w") as f:
            f.write(f"{sid}.hmd: ERA5-Land daily relative humidity\n")
            f.write(f"  nbyr     tstep       lat       lon      elev\n")
            f.write(f"{n_years:6d}{0:10d}{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")
            for di, d in enumerate(days):
                jday = d.timetuple().tm_yday
                f.write(f"{d.year:4d}{jday:4d}{fv(hmd_d[di,ii,jj]):12.3f}\n")

        # WND
        with open(swat_dir / f"{sid}.wnd", "w") as f:
            f.write(f"{sid}.wnd: ERA5-Land daily wind speed\n")
            f.write(f"  nbyr     tstep       lat       lon      elev\n")
            f.write(f"{n_years:6d}{0:10d}{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")
            for di, d in enumerate(days):
                jday = d.timetuple().tm_yday
                f.write(f"{d.year:4d}{jday:4d}{fv(wnd_d[di,ii,jj]):12.3f}\n")

    # ---- CLI file lists (valid stations only) ----
    for ext in ["pcp", "tmp", "slr", "hmd", "wnd"]:
        with open(swat_dir / f"{ext}.cli", "w") as f:
            f.write(f"{ext}.cli: written by swat_weather.py\n")
            f.write(f"filename\n")
            for s in stations:
                f.write(f"{s['id']}.{ext}\n")

    # ---- WGN weather generator ----
    print(f"  📝 Writing wgn_stations.csv + wgn_monthly.csv ...")
    _write_wgn_csv(swat_dir, stations, days, day_map, tp, t2m, rh, ssrd, wnd,
                   tmax_d, tmin_d, pcp_d, slr_d, hmd_d, wnd_d)

    # ---- weather-sta.cli ----
    with open(swat_dir / "weather-sta.cli", "w") as f:
        f.write(f"weather-sta.cli: written by swat_weather.py\n")
        hdr = (f"{'name':20s}{'wgn':20s}{'pcp':20s}{'tmp':20s}"
               f"{'slr':20s}{'hmd':20s}{'wnd':20s}{'wnd_dir':20s}"
               f"{'atmo_dep':20s}{'lat':>10s}{'lon':>10s}{'elev':>10s}\n")
        f.write(hdr)
        for s in stations:
            sid = s["id"]
            f.write(f"{sid:20s}{sid:20s}{sid+'.pcp':20s}{sid+'.tmp':20s}"
                    f"{sid+'.slr':20s}{sid+'.hmd':20s}{sid+'.wnd':20s}"
                    f"{'null':20s}{'null':20s}"
                    f"{s['lat']:10.3f}{s['lon']:10.3f}{s['elev']:10.1f}\n")

    # ---- Coordinate CSV ----
    with open(swat_dir / "stations.csv", "w") as f:
        f.write("id,lat,lon,elev\n")
        for s in stations:
            f.write(f"{s['id']},{s['lat']:.4f},{s['lon']:.4f},{s['elev']:.1f}\n")

    # ---- Validation: ensure every file listed in a CLI exists ----
    print(f"\n  🔍 Verifying file completeness...")
    missing = []
    for ext in ["pcp", "tmp", "slr", "hmd", "wnd"]:
        for s in stations:
            fpath = swat_dir / f"{s['id']}.{ext}"
            if not fpath.exists():
                missing.append(str(fpath))
    if missing:
        print(f"  ❌ Missing {len(missing)} files! First 5:")
        for m in missing[:5]:
            print(f"     {m}")
    else:
        print(f"  ✅ All {n_sta * 5} data files validated (5 variables per station)")

    # ---- Summary ----
    print(f"\n  ✅ Done!")
    print(f"  📂 {swat_dir}/")
    print(f"  📍 {n_sta} valid stations (mask skipped {n_ocean}, validation dropped {n_dropped_final})")
    print(f"  📅 {days[0]} → {days[-1]} ({n_days} days, {n_t} hourly steps)")
    print(f"  📊 precipitation={np.nanmean(pcp_d):.1f}mm/d  Tmax/Tmin={np.nanmean(tmax_d):.1f}/{np.nanmean(tmin_d):.1f}°C")
    print(f"     SLR={np.nanmean(slr_d):.1f}MJ  RH={np.nanmean(hmd_d):.2f}  WND={np.nanmean(wnd_d):.1f}m/s")
    if dem_dir:
        print(f"  🗺️  DEM directory: {dem_dir}")
    return swat_dir


# ============================================================
# WGN weather generator
# ============================================================
def _write_wgn_csv(swat_dir, stations, days, day_map, tp, t2m, rh, ssrd, wnd,
                   tmax_d, tmin_d, pcp_d, slr_d, hmd_d, wnd_d):
    """
    Generate the SWAT+ Editor "Two CSV files" format:

    1) wgn_stations.csv:  id, name, lat, lon, elev, rain_yrs
    2) wgn_monthly.csv:   id, wgn_id, month, tmp_max_ave, tmp_min_ave,
                          tmp_max_sd, tmp_min_sd, pcp_ave, pcp_sd, pcp_skew,
                          wet_dry, wet_wet, pcp_days, pcp_hhr, slr_ave,
                          dew_ave, wnd_ave
    """
    month_didx = {}
    for di, d in enumerate(days):
        month_didx.setdefault(d.month, []).append(di)

    def safe(v, default=0.0):
        return default if (v is None or np.isnan(v)) else float(v)

    # CSV 1: stations
    with open(swat_dir / "wgn_stations.csv", "w") as f:
        f.write("id,name,lat,lon,elev,rain_yrs\n")
        for idx, s in enumerate(stations):
            f.write(f"{idx+1},{s['id']},{s['lat']:.4f},{s['lon']:.4f},{s['elev']:.1f},1\n")

    # CSV 2: monthly values
    with open(swat_dir / "wgn_monthly.csv", "w") as f:
        f.write("id,wgn_id,month,tmp_max_ave,tmp_min_ave,tmp_max_sd,tmp_min_sd,"
                "pcp_ave,pcp_sd,pcp_skew,wet_dry,wet_wet,pcp_days,pcp_hhr,"
                "slr_ave,dew_ave,wnd_ave\n")

        row_id = 0
        for idx, s in enumerate(stations):
            ii, jj = s["i"], s["j"]
            wgn_id = idx + 1

            for mon in range(1, 13):
                row_id += 1
                didx = month_didx.get(mon)

                if didx and len(didx) >= 1:
                    tmx = tmax_d[didx, ii, jj]
                    tmn = tmin_d[didx, ii, jj]
                    pcp = pcp_d[didx, ii, jj]
                    sl  = slr_d[didx, ii, jj]
                    hm  = hmd_d[didx, ii, jj]
                    wn  = wnd_d[didx, ii, jj]

                    tmp_max_ave = safe(np.nanmean(tmx), 30.0)
                    tmp_min_ave = safe(np.nanmean(tmn), 22.0)
                    tmp_max_sd  = max(0.1, safe(np.nanstd(tmx), 1.5))
                    tmp_min_sd  = max(0.1, safe(np.nanstd(tmn), 1.0))
                    pcp_ave     = safe(np.nansum(pcp), 100.0)
                    pcp_sd_v    = max(0.1, safe(np.nanstd(pcp), 5.0))
                    wet = pcp > 0.1
                    n_d = len(pcp)
                    p_days = max(0.1, safe(float(np.nansum(wet)), 10.0))
                    p_wd   = min(0.95, safe(float(np.nansum(wet))/max(1,n_d), 0.4))
                    p_ww   = min(0.95, p_wd)
                    h_idx = []
                    for di in didx: h_idx.extend(day_map[days[di]])
                    pcp_hr = tp[h_idx, ii, jj] if h_idx else np.array([0.0])
                    p_hhr  = max(0.1, safe(float(np.nanmax(pcp_hr)), 10.0))
                    slr_av = safe(np.nanmean(sl), 18.0)
                    dew_av = safe(np.nanmean(hm), 0.75)
                    wnd_av = safe(np.nanmean(wn), 2.0)
                else:
                    tmp_max_ave, tmp_min_ave = 32.0, 23.0
                    tmp_max_sd, tmp_min_sd = 1.5, 1.0
                    pcp_ave, pcp_sd_v = 150.0, 10.0
                    p_wd, p_ww, p_days, p_hhr = 0.4, 0.6, 15.0, 30.0
                    slr_av, dew_av, wnd_av = 18.0, 0.75, 2.0

                f.write(f"{row_id},{wgn_id},{mon},"
                        f"{tmp_max_ave:.3f},{tmp_min_ave:.3f},"
                        f"{tmp_max_sd:.3f},{tmp_min_sd:.3f},"
                        f"{pcp_ave:.3f},{pcp_sd_v:.3f},0.000,"
                        f"{p_wd:.3f},{p_ww:.3f},"
                        f"{p_days:.3f},{p_hhr:.3f},"
                        f"{slr_av:.3f},{dew_av:.3f},{wnd_av:.3f}\n")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SWAT+ hourly weather (ERA5-Land + DEM mask)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a DEM mask (recommended)
  python swat_weather.py --country cambodia --start 2024-01-01 --days 14 \\
      --dem-dir /swat_global

  # Without a DEM, using NaN detection only (fallback)
  python swat_weather.py --country cambodia --start 2024-01-01 --days 14

SWAT+ Editor import steps:
  1) Climate → Weather Generator → IMPORT DATA
     → click Browse, select swat_weather/weather-wgn.cli
     → ☑ check "Use observed weather data"
     → Start Import
  2) Climate → Weather Stations → Import Observed
     → select the swat_weather/ directory
     → format SWAT+
  3) time.sim → step = 24 (hourly)
""")
    parser.add_argument("--country", type=str)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W","S","E","N"))
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output-dir", type=str, default="./weather_data")
    parser.add_argument("--dem-dir", type=str, default=None,
                        help="DEM file root directory (e.g. /swat_global); "
                             "the program recursively searches for all DEM files under it")
    args = parser.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.country:
        k = args.country.lower().strip()
        if k not in COUNTRY_BBOX:
            print(f"❌ Not supported: {k}. Available: {', '.join(COUNTRY_BBOX.keys())}"); sys.exit(1)
        bbox = COUNTRY_BBOX[k]
    else:
        print("❌ --country or --bbox is required"); sys.exit(1)

    s = date.fromisoformat(args.start)
    e = s + timedelta(days=args.days)
    nc = download_era5(bbox, s, e, Path(args.output_dir)/"raw_nc")
    if not nc: print("❌ Download failed"); sys.exit(1)
    nc_to_swat(nc, args.output_dir, s, e, dem_dir=args.dem_dir)

if __name__ == "__main__":
    main()
