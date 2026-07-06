#!/usr/bin/env python3
"""
Future scenario pipeline: validate → convert weather → run SWAT+ → extract features.

Processes 81 regions × 15 scenarios (5 GCMs × 3 SSPs) for ISIMIP3b 2025-2040.
Warmup: 2025-2026 (nyskip=2), useful output: 2027-2040.

Usage:
  python3 run_future_scenarios.py                        # full pipeline
  python3 run_future_scenarios.py --step validate        # only validate NC data
  python3 run_future_scenarios.py --step convert         # only convert weather
  python3 run_future_scenarios.py --step run             # only run SWAT+ + extract
  python3 run_future_scenarios.py --workers 6            # parallel SWAT+ runs
  python3 run_future_scenarios.py --region brazil/brazil_norte --gcm gfdl-esm4 --ssp ssp126
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("/data")
SWAT_DIR = DATA_DIR / "swat_global"
REGIONAL_NC_DIR = DATA_DIR / "isimip3b_regional"
GLOBAL_NC_DIR = DATA_DIR / "isimip3b_global"
WORK_DIR = Path("/swat_tmp")

ALL_GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
ALL_SSPS = ["ssp126", "ssp370", "ssp585"]
ALL_VARS = ["pr", "tas", "rsds", "hurs", "sfcwind"]

# Time configuration
YRC_START = 2025
YRC_END = 2040
NYSKIP = 2
NBYR = YRC_END - YRC_START + 1  # 16
OUTPUT_START = YRC_START + NYSKIP  # 2027

# Feature engineering (from run_workflow.py)
DAILY_FIELDS = ['flo_in', 'flo_stor', 'precip', 'evap']
LSUNIT_WB_FIELDS = ['surq_gen', 'latq', 'sw_ave', 'perc', 'et', 'pet']
AQUIFER_FIELDS = ['flo', 'stor', 'rchrg', 'dep_wt']
LAG_HOURS = [1, 3, 6, 12, 24]
PRECIP_ROLL_HOURS = [6, 12, 24, 72, 168]


# ============================================================
# Step 0: Validation
# ============================================================
def validate_regional_nc():
    """Validate all pre-extracted regional NC files."""
    log.info("=" * 60)
    log.info("Step 0: Validating pre-extracted regional NC files")
    log.info("=" * 60)

    import netCDF4 as nc

    errors = []
    warnings = []
    n_checked = 0

    regions = discover_regions()
    log.info(f"Checking {len(regions)} regions × {len(ALL_GCMS)} GCMs × {len(ALL_SSPS)} SSPs × {len(ALL_VARS)} vars × 4 periods")

    expected_periods = ["2021_2025", "2026_2030", "2031_2035", "2036_2040"]

    for r in regions:
        continent, region = r["continent"], r["region"]
        for gcm in ALL_GCMS:
            for ssp in ALL_SSPS:
                nc_dir = REGIONAL_NC_DIR / continent / region / gcm / ssp
                if not nc_dir.exists():
                    errors.append(f"MISSING DIR: {nc_dir}")
                    continue

                for var in ALL_VARS:
                    for period in expected_periods:
                        pattern = f"{gcm}*_{ssp}_{var}_*_{period}.nc"
                        matches = list(nc_dir.glob(pattern))
                        if not matches:
                            errors.append(f"MISSING: {continent}/{region}/{gcm}/{ssp}/{var}/{period}")
                            continue

                        f = matches[0]
                        if f.stat().st_size == 0:
                            errors.append(f"EMPTY: {f}")
                            continue

                        # Spot-check: open every 10th file
                        n_checked += 1
                        if n_checked % 10 == 0:
                            try:
                                ds = nc.Dataset(str(f), "r")
                                if var not in ds.variables:
                                    errors.append(f"NO VAR '{var}': {f}")
                                else:
                                    data = ds[var]
                                    # Check a small sample
                                    sample = data[0, :, :]
                                    if np.all(np.isnan(sample)):
                                        warnings.append(f"ALL NaN first timestep: {f}")

                                    # Value range checks
                                    if var == "tas":
                                        vmin, vmax = np.nanmin(sample), np.nanmax(sample)
                                        if vmin < 180 or vmax > 340:
                                            errors.append(f"tas OUT OF RANGE [{vmin:.1f}, {vmax:.1f}]: {f}")
                                    elif var == "pr":
                                        vmin = np.nanmin(sample)
                                        if vmin < 0:
                                            errors.append(f"pr NEGATIVE [{vmin}]: {f}")
                                    elif var == "hurs":
                                        vmin, vmax = np.nanmin(sample), np.nanmax(sample)
                                        if vmin < 0 or vmax > 150:
                                            errors.append(f"hurs OUT OF RANGE [{vmin:.1f}, {vmax:.1f}]: {f}")

                                ds.close()
                            except Exception as e:
                                errors.append(f"CORRUPT: {f}: {e}")

    log.info(f"\nValidation complete: checked {n_checked} files in detail")
    if errors:
        log.error(f"ERRORS ({len(errors)}):")
        for e in errors[:20]:
            log.error(f"  {e}")
        if len(errors) > 20:
            log.error(f"  ... and {len(errors) - 20} more")
        return False
    if warnings:
        log.warning(f"Warnings ({len(warnings)}):")
        for w in warnings[:10]:
            log.warning(f"  {w}")
    log.info("✓ All NC files validated successfully")
    return True


def validate_txtinout_scenarios():
    """Validate all 1215 scenario TxtInOut directories."""
    log.info("\nValidating TxtInOut scenario directories...")
    errors = []
    regions = discover_regions()

    for r in regions:
        continent, region = r["continent"], r["region"]
        for gcm in ALL_GCMS:
            for ssp in ALL_SSPS:
                txtinout = SWAT_DIR / continent / region / region / "Scenarios" / "isimip3b" / gcm / ssp / "TxtInOut"
                if not txtinout.exists():
                    errors.append(f"MISSING: {txtinout}")
                    continue
                if not (txtinout / "file.cio").exists():
                    errors.append(f"NO file.cio: {txtinout}")

    if errors:
        log.error(f"TxtInOut errors ({len(errors)}):")
        for e in errors[:10]:
            log.error(f"  {e}")
        return False
    log.info(f"✓ All 1215 TxtInOut directories valid")
    return True


# ============================================================
# Step 1: Delete global NC files
# ============================================================
def delete_global_nc():
    """Delete global NC files to free ~1TB."""
    if not GLOBAL_NC_DIR.exists():
        log.info("Global NC directory already deleted")
        return
    size_gb = sum(f.stat().st_size for f in GLOBAL_NC_DIR.rglob("*") if f.is_file()) / 1e9
    log.info(f"Deleting {GLOBAL_NC_DIR} ({size_gb:.0f} GB)...")
    shutil.rmtree(GLOBAL_NC_DIR)
    log.info("✓ Global NC files deleted")


# ============================================================
# Step 2: Convert NC → SWAT+ weather (inject into existing stations)
# ============================================================
def parse_station_name_coords(name):
    """Parse lat/lon from station name like s31500n108900e."""
    import re
    m = re.match(r's(\d+)([ns])(\d+)([ew])', name)
    if not m:
        return None, None
    lat = int(m.group(1)) / 1000.0 * (1 if m.group(2) == 'n' else -1)
    lon = int(m.group(3)) / 1000.0 * (1 if m.group(4) == 'e' else -1)
    return lat, lon


def parse_weather_sta_cli(txtinout):
    """Read weather-sta.cli and return list of station dicts with name, pcp_prefix, lat, lon."""
    cli_file = txtinout / "weather-sta.cli"
    stations = []
    with open(cli_file) as f:
        lines = f.readlines()
    if len(lines) < 3:
        return stations
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        name = parts[0]
        pcp_prefix = parts[2].replace('.pcp', '') if parts[2] != 'null' else None
        tmp_prefix = parts[3].replace('.tmp', '') if parts[3] != 'null' else None
        slr_prefix = parts[4].replace('.slr', '') if parts[4] != 'null' else None
        hmd_prefix = parts[5].replace('.hmd', '') if parts[5] != 'null' else None
        wnd_prefix = parts[6].replace('.wnd', '') if parts[6] != 'null' else None
        lat, lon = parse_station_name_coords(name)
        if lat is None:
            continue
        stations.append({
            'name': name, 'lat': lat, 'lon': lon,
            'pcp': pcp_prefix, 'tmp': tmp_prefix,
            'slr': slr_prefix, 'hmd': hmd_prefix, 'wnd': wnd_prefix,
        })
    return stations




# ============================================================
# Step 3: Modify time.sim and print.prt
# ============================================================
def setup_simulation_config(txtinout):
    """Modify time.sim and print.prt for future scenario."""
    # time.sim: preserve header lines, only update line 3 (data line)
    time_sim = txtinout / "time.sim"
    with open(time_sim) as f:
        tlines = f.readlines()
    tlines[2] = f"       0    {YRC_START}       0    {YRC_END}      24\n"
    with open(time_sim, "w") as f:
        f.writelines(tlines)

    # print.prt: set nyskip and enable channel_sd daily output
    prt_file = txtinout / "print.prt"
    if not prt_file.exists():
        return

    with open(prt_file) as f:
        lines = f.readlines()

    # Line 3 (index 2) has nyskip as first field
    parts = lines[2].split()
    parts[0] = str(NYSKIP)
    lines[2] = "   ".join(parts) + "\n"

    # Enable daily output for channel_sd, lsunit_wb, aquifer
    targets = {'lsunit_wb', 'channel_sd', 'aquifer'}
    for i, line in enumerate(lines):
        p = line.split()
        if len(p) >= 5 and p[0] in targets:
            if p[1] != 'y':
                p[1] = 'y'
                lines[i] = "  ".join(p) + "\n"

    with open(prt_file, "w") as f:
        f.writelines(lines)


# ============================================================
# Step 4: Run SWAT+ and extract features
# ============================================================
def inject_weather_to_workdir(work_txtinout, nc_dir, gcm, ssp, stations):
    """Inject ISIMIP3b weather into work directory (optimized with grid-point caching)."""
    import xarray as xr

    periods = ["2021_2025", "2026_2030", "2031_2035", "2036_2040"]
    import netCDF4 as _nc4
    for var in ALL_VARS:
        for period in periods:
            pattern = f"{gcm}*_{ssp}_{var}_*_{period}.nc"
            matches = list(nc_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Missing NC: {nc_dir}/{pattern}")
            with _nc4.Dataset(str(matches[0]), "r") as _ds:
                _lons = _ds["lon"][:]
                if len(set(_lons.tolist())) != len(_lons):
                    raise ValueError(f"DUP_LON in {matches[0].name}")

    var_datasets = {}
    for var in ALL_VARS:
        ds_list = []
        for period in periods:
            pattern = f"{gcm}*_{ssp}_{var}_*_{period}.nc"
            matches = list(nc_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Missing NC: {nc_dir}/{pattern}")
            ds_list.append(xr.open_dataset(matches[0], mask_and_scale=False))
        combined = xr.concat(ds_list, dim='time').sel(
            time=slice(f"{YRC_START}-01-01", f"{YRC_END}-12-31T23:59:59"))
        var_datasets[var] = combined

    sample_ds = var_datasets['pr']
    isimip_lats = sample_ds.lat.values
    isimip_lons = sample_ds.lon.values
    if len(isimip_lats) > 1 and isimip_lats[0] > isimip_lats[-1]:
        isimip_lats = isimip_lats[::-1]
        for v in var_datasets:
            var_datasets[v] = var_datasets[v].sortby('lat')

    times = pd.DatetimeIndex(sample_ds.time.values)
    n_days = len(times) // 24
    days = pd.date_range(f"{YRC_START}-01-01", periods=n_days, freq='D')
    nlat, nlon = len(isimip_lats), len(isimip_lons)

    pr_data = var_datasets['pr']['pr'].values.astype(np.float64)
    tas_data = var_datasets['tas']['tas'].values.astype(np.float64)
    rsds_data = var_datasets['rsds']['rsds'].values.astype(np.float64)
    hurs_data = var_datasets['hurs']['hurs'].values.astype(np.float64)
    wind_data = var_datasets['sfcwind']['sfcwind'].values.astype(np.float64)

    # Replace NC fill values (ocean/boundary cells have ~9.97e+36) with NaN
    FILL_THRESH = 1e10
    pr_data[pr_data > FILL_THRESH] = np.nan
    tas_data[np.abs(tas_data) > FILL_THRESH] = np.nan
    rsds_data[rsds_data > FILL_THRESH] = np.nan
    hurs_data[hurs_data > FILL_THRESH] = np.nan
    wind_data[wind_data > FILL_THRESH] = np.nan

    # Unit checks
    pr_units = var_datasets['pr']['pr'].attrs.get('units', '')
    if 'kg' in pr_units or 's-1' in pr_units or 's^-1' in pr_units:
        pr_data = pr_data * 3600.0
    np.clip(pr_data, 0, None, out=pr_data)

    if np.nanmean(tas_data[:24]) > 100:
        tas_data = tas_data - 273.15

    if np.nanmean(hurs_data[:24]) > 5:
        hurs_data = hurs_data / 100.0
    np.clip(hurs_data, 0, 1, out=hurs_data)
    np.clip(wind_data, 0, None, out=wind_data)

    # Daily aggregates (suppress warnings for ocean grid cells with all-NaN)
    pr_hourly = pr_data.reshape(n_days, 24, nlat, nlon)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tmax_d = np.nanmax(tas_data.reshape(n_days, 24, nlat, nlon), axis=1)
        tmin_d = np.nanmin(tas_data.reshape(n_days, 24, nlat, nlon), axis=1)
        slr_d = np.nanmean(rsds_data.reshape(n_days, 24, nlat, nlon), axis=1) * 0.0864
        hmd_d = np.nanmean(hurs_data.reshape(n_days, 24, nlat, nlon), axis=1)
        wnd_d = np.nanmean(wind_data.reshape(n_days, 24, nlat, nlon), axis=1)

    del pr_data, tas_data, rsds_data, hurs_data, wind_data
    for ds in var_datasets.values():
        ds.close()
    del var_datasets
    gc.collect()

    # Pre-compute date info arrays
    yrs = np.array([d.year for d in days], dtype=np.int32)
    jdays = np.array([d.timetuple().tm_yday for d in days], dtype=np.int32)
    mons = np.array([d.month for d in days], dtype=np.int32)
    ddays = np.array([d.day for d in days], dtype=np.int32)

    # Build land mask from first timestep of temperature data
    land_mask = np.isfinite(tmax_d[0])

    # Find nearest LAND grid indices for each station
    sta_lats = np.array([s['lat'] for s in stations])
    sta_lons = np.array([s['lon'] for s in stations])
    lat_idx = np.empty(len(stations), dtype=np.int32)
    lon_idx = np.empty(len(stations), dtype=np.int32)
    n_remapped = 0
    for si in range(len(stations)):
        ii = np.argmin(np.abs(isimip_lats - sta_lats[si]))
        jj = np.argmin(np.abs(isimip_lons - sta_lons[si]))
        if land_mask[ii, jj]:
            lat_idx[si], lon_idx[si] = ii, jj
        else:
            dist2 = (isimip_lats[:, None] - sta_lats[si]) ** 2 + \
                     (isimip_lons[None, :] - sta_lons[si]) ** 2
            dist2[~land_mask] = np.inf
            lat_idx[si], lon_idx[si] = np.unravel_index(np.argmin(dist2), (nlat, nlon))
            n_remapped += 1
    if n_remapped:
        log.info(f"  Remapped {n_remapped} stations from ocean to nearest land grid cell")

    # Group stations by grid point to cache data strings
    from collections import defaultdict
    grid_groups = defaultdict(list)
    for si in range(len(stations)):
        grid_groups[(lat_idx[si], lon_idx[si])].append(si)

    n_years = NBYR

    # Process by grid point: build data once, write to all stations at that point
    for (ii, jj), sta_indices in grid_groups.items():
        # Build pcp data lines for this grid point
        pcp_lines = []
        for di in range(n_days):
            yr, jd, mon, day = yrs[di], jdays[di], mons[di], ddays[di]
            for hr in range(24):
                v = pr_hourly[di, hr, ii, jj]
                if np.isnan(v):
                    v = 0.0
                pcp_lines.append(f"{yr:4d}{jd:4d}{mon:4d}{day:4d}{hr:4d}{v:12.3f}\n")
        pcp_body = "".join(pcp_lines)

        # Build daily data lines
        tmp_lines = []
        slr_lines = []
        hmd_lines = []
        wnd_lines = []
        for di in range(n_days):
            yr, jd = yrs[di], jdays[di]
            tmx = tmax_d[di, ii, jj]
            tmn = tmin_d[di, ii, jj]
            sl = slr_d[di, ii, jj]
            hm = hmd_d[di, ii, jj]
            wn = wnd_d[di, ii, jj]
            if np.isnan(tmx): tmx = 0.0
            if np.isnan(tmn): tmn = 0.0
            if np.isnan(sl): sl = 0.0
            if np.isnan(hm): hm = 0.0
            if np.isnan(wn): wn = 0.0
            tmp_lines.append(f"{yr:4d}{jd:4d}{tmx:12.3f}{tmn:12.3f}\n")
            slr_lines.append(f"{yr:4d}{jd:4d}{sl:12.3f}\n")
            hmd_lines.append(f"{yr:4d}{jd:4d}{hm:12.3f}\n")
            wnd_lines.append(f"{yr:4d}{jd:4d}{wn:12.3f}\n")

        tmp_body = "".join(tmp_lines)
        slr_body = "".join(slr_lines)
        hmd_body = "".join(hmd_lines)
        wnd_body = "".join(wnd_lines)

        # Write for each station at this grid point
        for si in sta_indices:
            s = stations[si]
            lat, lon = s['lat'], s['lon']

            if s['pcp']:
                with open(work_txtinout / f"{s['pcp']}.pcp", 'w') as f:
                    f.write(f"{s['pcp']}.pcp: ISIMIP3b {gcm} {ssp}\n")
                    f.write(f"  nbyr     tstep       lat       lon      elev\n")
                    f.write(f"{n_years:6d}{60:10d}{lat:10.3f}{lon:10.3f}{0.0:10.1f}\n")
                    f.write(pcp_body)

            if s['tmp']:
                with open(work_txtinout / f"{s['tmp']}.tmp", 'w') as f:
                    f.write(f"{s['tmp']}.tmp: ISIMIP3b {gcm} {ssp}\n")
                    f.write(f"  nbyr     tstep       lat       lon      elev\n")
                    f.write(f"{n_years:6d}{0:10d}{lat:10.3f}{lon:10.3f}{0.0:10.1f}\n")
                    f.write(tmp_body)

            if s['slr']:
                with open(work_txtinout / f"{s['slr']}.slr", 'w') as f:
                    f.write(f"{s['slr']}.slr: ISIMIP3b {gcm} {ssp}\n")
                    f.write(f"  nbyr     tstep       lat       lon      elev\n")
                    f.write(f"{n_years:6d}{0:10d}{lat:10.3f}{lon:10.3f}{0.0:10.1f}\n")
                    f.write(slr_body)

            if s['hmd']:
                with open(work_txtinout / f"{s['hmd']}.hmd", 'w') as f:
                    f.write(f"{s['hmd']}.hmd: ISIMIP3b {gcm} {ssp}\n")
                    f.write(f"  nbyr     tstep       lat       lon      elev\n")
                    f.write(f"{n_years:6d}{0:10d}{lat:10.3f}{lon:10.3f}{0.0:10.1f}\n")
                    f.write(hmd_body)

            if s['wnd']:
                with open(work_txtinout / f"{s['wnd']}.wnd", 'w') as f:
                    f.write(f"{s['wnd']}.wnd: ISIMIP3b {gcm} {ssp}\n")
                    f.write(f"  nbyr     tstep       lat       lon      elev\n")
                    f.write(f"{n_years:6d}{0:10d}{lat:10.3f}{lon:10.3f}{0.0:10.1f}\n")
                    f.write(wnd_body)

    del pr_hourly, tmax_d, tmin_d, slr_d, hmd_d, wnd_d
    gc.collect()
    return len(stations)


_nc_semaphore = None

def _init_worker(sem):
    global _nc_semaphore
    _nc_semaphore = sem

def run_one_scenario(continent, region, gcm, ssp):
    """
    Complete pipeline for one scenario:
    copy config → inject weather → run SWAT+ → extract features → save → cleanup.
    """
    scenario_id = f"{continent}/{region}/{gcm}/{ssp}"
    txtinout_src = SWAT_DIR / continent / region / region / "Scenarios" / "isimip3b" / gcm / ssp / "TxtInOut"
    nc_dir = REGIONAL_NC_DIR / continent / region / gcm / ssp
    output_dir = SWAT_DIR / continent / region / region / "Scenarios" / "isimip3b" / gcm / ssp / "datasets"
    output_file = output_dir / "ml_features.parquet"

    if output_file.exists() and output_file.stat().st_size > 0:
        return f"SKIP (done): {scenario_id}"

    if not txtinout_src.exists():
        return f"ERROR (no TxtInOut): {scenario_id}"
    if not nc_dir.exists():
        return f"ERROR (no NC dir): {scenario_id}"

    work_txtinout = WORK_DIR / f"{continent}_{region}_{gcm}_{ssp}" / "TxtInOut"

    try:
        # Copy config files only (skip old weather files)
        if work_txtinout.exists():
            shutil.rmtree(work_txtinout, ignore_errors=True)
        work_txtinout.mkdir(parents=True, exist_ok=True)

        weather_exts = {'.pcp', '.tmp', '.slr', '.hmd', '.wnd'}
        for f in txtinout_src.iterdir():
            if f.is_file() and f.suffix not in weather_exts:
                shutil.copy2(f, work_txtinout / f.name)

        # Parse stations from weather-sta.cli
        stations = parse_weather_sta_cli(work_txtinout)
        if not stations:
            return f"ERROR (no stations): {scenario_id}"

        # Inject ISIMIP3b weather (semaphore limits concurrent NC loading to avoid OOM)
        if _nc_semaphore:
            _nc_semaphore.acquire()
        try:
            n_sta = inject_weather_to_workdir(work_txtinout, nc_dir, gcm, ssp, stations)
        finally:
            if _nc_semaphore:
                _nc_semaphore.release()

        # Setup time.sim and print.prt
        setup_simulation_config(work_txtinout)

        # Run SWAT+
        swat_exe = Path.home() / "bin" / "swatplus"
        proc = subprocess.run(
            [str(swat_exe)], cwd=str(work_txtinout),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        if proc.returncode == -6:
            swat_exe2 = Path.home() / "bin" / "swatplus_mine"
            proc = subprocess.run(
                [str(swat_exe2)], cwd=str(work_txtinout),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            if proc.returncode != 0:
                err = (proc.stderr or "")[-500:]
                return f"ERROR (swatplus_mine rc={proc.returncode}): {scenario_id}\n{err}"
        elif proc.returncode != 0:
            err = (proc.stderr or "")[-500:]
            return f"ERROR (swatplus rc={proc.returncode}): {scenario_id}\n{err}"

        # Extract features
        result = extract_features(work_txtinout, output_dir, scenario_id)
        return result

    except Exception as e:
        import traceback
        return f"ERROR ({e}): {scenario_id}\n{traceback.format_exc()[-300:]}"
    finally:
        work_parent = work_txtinout.parent
        if work_parent.exists():
            shutil.rmtree(work_parent, ignore_errors=True)


def parse_chandeg_con(txtinout):
    """Parse chandeg.con to get channel_id → {wst, lat, lon} mapping."""
    chandeg = txtinout / "chandeg.con"
    channels = {}
    if not chandeg.exists():
        return channels
    with open(chandeg) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 9:
            gis_id = int(parts[2])
            wst = parts[8]
            lat = float(parts[4]) if len(parts) > 4 else 0.0
            lon = float(parts[5]) if len(parts) > 5 else 0.0
            channels[gis_id] = {'wst': wst, 'lat': lat, 'lon': lon}
    return channels


def get_target_channels(region_path, chan_info):
    """Get target channels from station_channel_result.csv (same as training).
    Falls back to hydro_plants.csv nearest-neighbor if result CSV missing."""
    result_csv = region_path / "hydro" / "station_channel_result.csv"
    if result_csv.exists():
        try:
            df = pd.read_csv(result_csv, encoding='utf-8-sig')
            if 'swatplus_lcha' in df.columns:
                channels = sorted(df['swatplus_lcha'].dropna().astype(int).unique().tolist())
                valid = [ch for ch in channels if ch in chan_info]
                if valid:
                    return valid
        except Exception:
            pass

    hydro_csv = region_path / "hydro" / "hydro_plants.csv"
    if not hydro_csv.exists():
        return list(chan_info.keys())

    try:
        plants = pd.read_csv(hydro_csv, encoding='utf-8-sig')
    except Exception:
        return list(chan_info.keys())

    if 'lat_unified' not in plants.columns or 'lon_unified' not in plants.columns:
        return list(chan_info.keys())

    plants = plants.dropna(subset=['lat_unified', 'lon_unified'])
    if len(plants) == 0:
        return list(chan_info.keys())

    ch_ids = np.array(list(chan_info.keys()))
    ch_lats = np.array([chan_info[c]['lat'] for c in ch_ids])
    ch_lons = np.array([chan_info[c]['lon'] for c in ch_ids])

    target = set()
    for _, plant in plants.iterrows():
        plat, plon = plant['lat_unified'], plant['lon_unified']
        dist2 = (ch_lats - plat) ** 2 + (ch_lons - plon) ** 2
        nearest_idx = np.argmin(dist2)
        target.add(int(ch_ids[nearest_idx]))

    return sorted(target)


def extract_features(txtinout, output_dir, scenario_id):
    """Extract ML features from SWAT+ output for target channels only."""
    parts = scenario_id.split("/")
    continent, region = parts[0], parts[1]
    region_path = SWAT_DIR / continent / region

    # Get channel→info mapping from chandeg.con
    chan_info = parse_chandeg_con(txtinout)
    if not chan_info:
        return f"ERROR (no channels in chandeg.con): {scenario_id}"

    # Get target channels from hydro_plants.csv (not all channels!)
    target_channels = get_target_channels(region_path, chan_info)
    chan_wst = {ch: chan_info[ch]['wst'] for ch in target_channels if ch in chan_info}
    log.info(f"  {scenario_id}: {len(target_channels)} target channels (of {len(chan_info)} total)")

    # Read SWAT+ output files
    swat_subday_file = txtinout / "channel_sd_subday.txt"
    swat_day_file = txtinout / "channel_sd_day.txt"

    if not swat_subday_file.exists():
        return f"ERROR (no subday output): {scenario_id}"

    # Delete unused large files to free disk
    for unused in ['sd_chanbud_day.txt', 'channel_sdmorph_day.txt']:
        f = txtinout / unused
        if f.exists():
            f.unlink()

    # Prefilter large channel files (stream-based, <1GB memory)
    prefilter_channel_file(swat_subday_file, target_channels)
    prefilter_channel_file(swat_day_file, target_channels)

    # Read subday (target channels only, chunked)
    subday_subset = read_subday(swat_subday_file, target_channels)
    if subday_subset is None or len(subday_subset) == 0:
        return f"ERROR (no subday data): {scenario_id}"

    swat_day_subset = read_day(swat_day_file, target_channels, DAILY_FIELDS)

    # Read weather by unique wst
    wst_mapping = read_weather_sta_cli(txtinout)
    unique_wsts = set(chan_wst.values())
    pcp_cache = {}
    tmp_cache = {}
    for wst in unique_wsts:
        if wst in wst_mapping and wst_mapping[wst].get('pcp'):
            pcp_df = read_pcp_hourly(txtinout, wst_mapping[wst]['pcp'])
            if pcp_df is not None:
                pcp_cache[wst] = pcp_df
        if wst in wst_mapping and wst_mapping[wst].get('tmp'):
            tmp_df = read_tmp_daily(txtinout, wst_mapping[wst]['tmp'])
            if tmp_df is not None:
                tmp_cache[wst] = tmp_df

    # Read lsunit_wb and aquifer (simplified: use all available)
    lsunit_wb_file = txtinout / "lsunit_wb_day.txt"
    aquifer_file = txtinout / "aquifer_day.txt"
    channel_mapping = build_channel_mapping(region_path, region, target_channels, txtinout)

    all_lsu_ids = []
    all_aqu_ids = []
    for ch, m in channel_mapping.items():
        all_lsu_ids.extend(m.get('lsu_ids', []))
        all_aqu_ids.extend(m.get('aquifer_ids', []))
    all_lsu_ids = list(set(all_lsu_ids))
    all_aqu_ids = list(set(all_aqu_ids))

    lsunit_wb_data = None
    if all_lsu_ids and lsunit_wb_file.exists():
        prefilter_swat_output_file(lsunit_wb_file, all_lsu_ids, 'rtu')
        lsunit_wb_data = read_lsunit_wb(lsunit_wb_file, all_lsu_ids, LSUNIT_WB_FIELDS)
    aquifer_data = None
    if all_aqu_ids and aquifer_file.exists():
        prefilter_swat_output_file(aquifer_file, all_aqu_ids, 'aqu')
        aquifer_data = read_aquifer(aquifer_file, all_aqu_ids, AQUIFER_FIELDS)

    # Filter to output period only
    output_start_dt = pd.Timestamp(f"{OUTPUT_START}-01-01")
    subday_subset = subday_subset[subday_subset['datetime'] >= output_start_dt].copy()

    # Build features for all channels
    all_features = []
    for ch_id, wst in chan_wst.items():
        result = build_features_no_grfr(
            ch_id, ch_id, wst, subday_subset, swat_day_subset,
            pcp_cache, tmp_cache, lsunit_wb_data, aquifer_data, channel_mapping
        )
        if result is not None and len(result) > 0:
            all_features.append(result)

    if not all_features:
        return f"ERROR (no features built): {scenario_id}"

    dataset = pd.concat(all_features, ignore_index=True)

    feature_cols = (
        ['flo_out', 'precip_mm']
        + [f'flo_out_lag{lag}h' for lag in LAG_HOURS]
        + [f'precip_sum_{w}h' for w in PRECIP_ROLL_HOURS]
        + DAILY_FIELDS + ['tmp_ave']
        + [f'wb_{f}' for f in LSUNIT_WB_FIELDS]
        + [f'aqu_{f}' for f in AQUIFER_FIELDS]
        + ['sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']
    )
    meta_cols = ['datetime', 'date', 'hour', 'channel_id']

    existing = [c for c in feature_cols if c in dataset.columns]
    save_cols = [c for c in meta_cols if c in dataset.columns] + existing

    dataset = dataset.dropna(subset=existing).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ml_features.parquet"
    dataset[save_cols].to_parquet(output_file, index=False)

    n_channels = dataset['channel_id'].nunique() if 'channel_id' in dataset.columns else 0
    sz_mb = output_file.stat().st_size / 1e6
    del dataset, all_features, subday_subset, swat_day_subset, pcp_cache, tmp_cache
    gc.collect()

    return f"OK ({n_channels} channels, {sz_mb:.1f} MB): {scenario_id}"


# ============================================================
# Helper functions (adapted from run_workflow.py)
# ============================================================
def discover_regions():
    """Find all regions."""
    regions = []
    for continent_dir in sorted(SWAT_DIR.iterdir()):
        if not continent_dir.is_dir():
            continue
        for region_dir in sorted(continent_dir.iterdir()):
            if not region_dir.is_dir():
                continue
            dem_path = region_dir / "rasters" / "dem.tif"
            if dem_path.exists():
                regions.append({
                    "continent": continent_dir.name,
                    "region": region_dir.name,
                })
    return regions


def prefilter_channel_file(filepath, target_channels):
    """Stream-filter large channel file, keeping only target channels."""
    if not filepath.exists():
        return

    target_set = set(str(ch) for ch in target_channels)

    with open(filepath, 'r') as f:
        header_lines = [f.readline() for _ in range(3)]

    col_names = header_lines[1].split()
    try:
        gis_id_idx = col_names.index('gis_id')
    except ValueError:
        return

    tmp_path = filepath.with_suffix('.tmp_filtered')
    n_kept = 0

    with open(filepath, 'r') as fin, open(tmp_path, 'w') as fout:
        for line in header_lines:
            fout.write(line)
        for _ in range(3):
            fin.readline()
        for line in fin:
            parts = line.split()
            if len(parts) > gis_id_idx and parts[gis_id_idx] in target_set:
                fout.write(line)
                n_kept += 1

    if n_kept == 0:
        tmp_path.unlink(missing_ok=True)
        return

    shutil.move(str(tmp_path), str(filepath))


def prefilter_swat_output_file(filepath, target_ids, name_prefix):
    """Stream-filter SWAT+ output file by name column."""
    if not filepath.exists():
        return

    target_ids = set(target_ids)
    prefix_len = len(name_prefix)

    with open(filepath, 'r') as f:
        header_lines = [f.readline() for _ in range(3)]

    tmp_path = filepath.with_suffix('.tmp_filtered')
    n_after = 0

    with open(filepath, 'r') as fin, open(tmp_path, 'w') as fout:
        for h in header_lines:
            fout.write(h)
        for _ in range(3):
            fin.readline()
        for line in fin:
            parts = line.split()
            if len(parts) < 7:
                continue
            name = parts[6]
            if not name.startswith(name_prefix):
                continue
            suffix = name[prefix_len:]
            if suffix.isdigit() and int(suffix) in target_ids:
                fout.write(line)
                n_after += 1

    if n_after == 0:
        tmp_path.unlink(missing_ok=True)
        return
    shutil.move(str(tmp_path), str(filepath))


def find_qswat_database(region_path: Path, area_name: str):
    """Find the QSWAT+ sqlite database file under the DatabaseBackups directory."""
    db_dir = region_path / area_name / "DatabaseBackups"
    if not db_dir.exists():
        # Try other possible paths
        for candidate in region_path.rglob("DatabaseBackups"):
            db_dir = candidate
            break

    if not db_dir.exists():
        print(f"⚠️  DatabaseBackups directory not found")
        return None

    # Find the most recent .sqlite file
    sqlite_files = sorted(db_dir.glob("*.sqlite"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not sqlite_files:
        print(f"⚠️  No .sqlite file found in DatabaseBackups")
        return None

    db_path = sqlite_files[0]
    print(f"  Database: {db_path.name}")
    return db_path


def build_channel_mapping(region_path: Path, area_name: str, target_channels: list, txtinout: Path):
    """
    Build from the QSWAT+ database:
      channel_id → {'lsu_ids': [...], 'aquifer_ids': [...]}
    Unmatched channels are matched to the nearest neighbor by lat/lon via chandeg.con / rout_unit.con.
    """
    import sqlite3

    db_path = find_qswat_database(region_path, area_name)
    if db_path is None:
        return {}

    conn = sqlite3.connect(str(db_path))
    lsus = pd.read_sql("SELECT id, channel, subbasin FROM gis_lsus", conn)
    aquifers = pd.read_sql("SELECT id, subbasin FROM gis_aquifers", conn)
    conn.close()

    mapping = {}
    mapped_channels = []
    unmapped_channels = []

    for ch in target_channels:
        lsu_rows = lsus[lsus['channel'] == ch]
        if len(lsu_rows) == 0:
            unmapped_channels.append(ch)
            continue

        lsu_ids = lsu_rows['id'].tolist()
        sub_ids = lsu_rows['subbasin'].unique().tolist()
        aqu_ids = aquifers[aquifers['subbasin'].isin(sub_ids)]['id'].tolist()

        mapping[ch] = {
            'lsu_ids': lsu_ids,
            'aquifer_ids': aqu_ids,
            'method': 'database',
        }
        mapped_channels.append(ch)

    # --- Unmatched channels: nearest-neighbor match by lat/lon ---
    if unmapped_channels:
        print(f"  Attempting nearest-neighbor lat/lon match for {len(unmapped_channels)} unmapped channels...")

        # Read chandeg.con
        chandeg_file = txtinout / "chandeg.con"
        rtu_file = txtinout / "rout_unit.con"

        if chandeg_file.exists() and rtu_file.exists():
            # Read rout_unit.con - parse line by line, not via read_csv
            rtu_records = []
            rtu_aqu_map = {}
            with open(rtu_file, 'r') as f:
                f.readline()  # skip header comment line
                f.readline()  # skip column-name line
                for line in f:
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        name = parts[1]
                        lat = float(parts[4])
                        lon = float(parts[5])
                    except (ValueError, IndexError):
                        continue
                    if not name.startswith('rtu'):
                        continue
                    lsu_id = int(name[3:])
                    rtu_records.append({'lsu_id': lsu_id, 'lat': lat, 'lon': lon})
                    # Find the 'aqu' keyword
                    for j, v in enumerate(parts):
                        if v == 'aqu' and j + 1 < len(parts):
                            try:
                                rtu_aqu_map[lsu_id] = int(parts[j + 1])
                            except ValueError:
                                pass
                            break

            rtu_df = pd.DataFrame(rtu_records)

            # Read chandeg.con - parse line by line
            cha_records = []
            with open(chandeg_file, 'r') as f:
                f.readline()  # skip header comment line
                f.readline()  # skip column-name line
                for line in f:
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        ch_id = int(parts[0])
                        lat = float(parts[4])
                        lon = float(parts[5])
                    except (ValueError, IndexError):
                        continue
                    cha_records.append({'id': ch_id, 'lat': lat, 'lon': lon})

            cha_df = pd.DataFrame(cha_records)

            rtu_lats = rtu_df['lat'].values
            rtu_lons = rtu_df['lon'].values
            rtu_lsu_ids = rtu_df['lsu_id'].values

            for ch in unmapped_channels:
                cha_row = cha_df[cha_df['id'] == ch]
                if len(cha_row) == 0:
                    print(f"    Channel {ch}: not found in chandeg.con either, skipping")
                    continue

                ch_lat = cha_row['lat'].values[0]
                ch_lon = cha_row['lon'].values[0]

                # Euclidean distance (lat/lon approximation)
                dists = np.sqrt((rtu_lats - ch_lat)**2 + (rtu_lons - ch_lon)**2)
                nearest_idx = np.argmin(dists)
                nearest_lsu_id = int(rtu_lsu_ids[nearest_idx])
                nearest_dist = dists[nearest_idx]

                # Find the aquifer for this LSU via the database
                lsu_row = lsus[lsus['id'] == nearest_lsu_id]
                if len(lsu_row) > 0:
                    sub_ids = lsu_row['subbasin'].unique().tolist()
                    aqu_ids = aquifers[aquifers['subbasin'].isin(sub_ids)]['id'].tolist()
                else:
                    aqu_ids = []

                lsu_ids = [nearest_lsu_id]
                mapping[ch] = {
                    'lsu_ids': lsu_ids,
                    'aquifer_ids': aqu_ids,
                    'method': f'nearest(dist={nearest_dist:.4f})',
                }
                mapped_channels.append(ch)
                print(f"    Channel {ch} ({ch_lat:.3f},{ch_lon:.3f}) → "
                      f"nearest LSU {lsu_ids} Aquifer {aqu_ids} (dist={nearest_dist:.4f}°)")

            del cha_df, rtu_df
        else:
            print(f"    ❌ chandeg.con or rout_unit.con does not exist, cannot do nearest-neighbor match")

    # Print summary
    still_unmapped = [ch for ch in target_channels if ch not in mapping]
    print(f"\n  Mapping result: {len(mapping)}/{len(target_channels)} channels")
    for ch in target_channels:
        if ch in mapping:
            m = mapping[ch]
            print(f"    Channel {ch} → LSU {m['lsu_ids']} → Aquifer {m['aquifer_ids']}  [{m['method']}]")
    if still_unmapped:
        print(f"  ❌ Still unmapped: {still_unmapped}")

    return mapping


def read_subday(filepath, target_channels):
    """Read channel_sd_subday.txt in chunks to avoid OOM on large files."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None

    target_set = set(target_channels)
    usecols = ['yr', 'mon', 'day', 'tstep', 'gis_id', 'flo_out']
    chunks = []

    for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c',
                             usecols=usecols, chunksize=2_000_000):
        filtered = chunk[chunk['gis_id'].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    tmin, tmax = df['tstep'].min(), df['tstep'].max()
    if tmin == 1 and tmax == 24:
        df['hour'] = df['tstep'] - 1
    elif tmin == 0 and tmax == 23:
        df['hour'] = df['tstep']
    else:
        df['hour'] = df['tstep'] - 1

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    df = df[['datetime', 'date', 'hour', 'gis_id', 'flo_out']].copy()
    df['flo_out'] = pd.to_numeric(df['flo_out'], errors='coerce') / 3600.0
    return df


def read_day(filepath, target_channels, daily_fields):
    """Read channel_sd_day.txt in chunks to avoid OOM on large files."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None

    target_set = set(target_channels)
    # First pass: determine available columns from header
    with open(filepath, 'r') as f:
        f.readline()  # skip line 0
        header_line = f.readline()  # line 1 = column names
    all_cols = header_line.split()
    usecols = ['yr', 'mon', 'day', 'gis_id'] + [c for c in daily_fields if c in all_cols]
    chunks = []

    for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c',
                             usecols=usecols, chunksize=2_000_000):
        filtered = chunk[chunk['gis_id'].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    cols_keep = ['date', 'gis_id'] + [c for c in daily_fields if c in df.columns]
    return df[cols_keep].copy()


def read_weather_sta_cli(txtinout_dir):
    """Parse weather-sta.cli to get station→file mapping with lat/lon."""
    cli_file = txtinout_dir / "weather-sta.cli"
    mapping = {}
    if not cli_file.exists():
        return mapping

    with open(cli_file) as f:
        lines = f.readlines()

    if len(lines) < 3:
        return mapping

    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 7:
            name = parts[0]
            pcp = parts[2].replace('.pcp', '') if parts[2] != 'null' else None
            tmp = parts[3].replace('.tmp', '') if parts[3] != 'null' else None
            lat = float(parts[-3]) if len(parts) >= 11 else None
            lon = float(parts[-2]) if len(parts) >= 11 else None
            mapping[name] = {'pcp': pcp, 'tmp': tmp, 'lat': lat, 'lon': lon}
    return mapping


def parse_old_wst_coords(wst_name):
    """Parse lat/lon from old station name like s31500n108900e or s100n49700w."""
    import re
    m = re.match(r's(\d+)n(\d+)([ew])', wst_name)
    if not m:
        return None, None
    lat_raw, lon_raw, hemisphere = int(m.group(1)), int(m.group(2)), m.group(3)
    lat = lat_raw / 1000.0
    lon = lon_raw / 1000.0
    if hemisphere == 'w':
        lon = -lon
    return lat, lon


def remap_wst_to_new_stations(unique_wsts, wst_mapping):
    """Remap old station names to nearest new stations when names don't match."""
    remap = {}
    stations_with_coords = [(name, info) for name, info in wst_mapping.items()
                            if info.get('lat') is not None]
    if not stations_with_coords:
        return remap

    new_lats = np.array([info['lat'] for _, info in stations_with_coords])
    new_lons = np.array([info['lon'] for _, info in stations_with_coords])
    new_names = [name for name, _ in stations_with_coords]

    for wst in unique_wsts:
        if wst in wst_mapping:
            remap[wst] = wst
            continue
        lat, lon = parse_old_wst_coords(wst)
        if lat is None:
            continue
        dist = (new_lats - lat)**2 + (new_lons - lon)**2
        nearest_idx = np.argmin(dist)
        remap[wst] = new_names[nearest_idx]
    return remap


def read_pcp_hourly(txtinout_dir, prefix):
    """Read hourly .pcp file."""
    pcp_file = txtinout_dir / f"{prefix}.pcp"
    if not pcp_file.exists():
        return None

    with open(pcp_file) as f:
        f.readline()  # header
        f.readline()  # col names
        meta = f.readline().split()
        tstep = int(meta[1])

    if tstep != 60:
        return None

    rows = []
    with open(pcp_file) as f:
        for _ in range(3):
            f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                yr, jd, mon, day, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                val = float(parts[5])
                rows.append((yr, mon, day, hr, val))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=['yr', 'mon', 'day', 'hr', 'precip_mm'])
    df['datetime'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'})) + pd.to_timedelta(df['hr'], unit='h')
    return df[['datetime', 'precip_mm']]


def read_tmp_daily(txtinout_dir, prefix):
    """Read daily .tmp file, compute average temperature."""
    tmp_file = txtinout_dir / f"{prefix}.tmp"
    if not tmp_file.exists():
        return None

    rows = []
    with open(tmp_file) as f:
        for _ in range(3):
            f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                yr, jd = int(parts[0]), int(parts[1])
                tmax, tmin = float(parts[2]), float(parts[3])
                rows.append((yr, jd, (tmax + tmin) / 2.0))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=['yr', 'jd', 'tmp_ave'])
    df['date'] = pd.to_datetime(df['yr'] * 1000 + df['jd'], format='%Y%j')
    return df[['date', 'tmp_ave']]


def read_lsunit_wb(filepath, target_lsu_ids, fields):
    """Read lsunit_wb_day.txt in chunks to avoid OOM."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None

    target_set = set(target_lsu_ids)
    chunks = []

    for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c',
                             chunksize=2_000_000):
        chunk['lsu_id'] = chunk['name'].str.replace('rtu', '').astype(int)
        filtered = chunk[chunk['lsu_id'].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    cols = ['date', 'lsu_id'] + [f for f in fields if f in df.columns]
    return df[cols].copy()


def read_aquifer(filepath, target_aqu_ids, fields):
    """Read aquifer_day.txt in chunks to avoid OOM."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None

    target_set = set(target_aqu_ids)
    chunks = []

    for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c',
                             chunksize=2_000_000):
        chunk['aqu_id'] = chunk['name'].str.replace('aqu', '').astype(int)
        filtered = chunk[chunk['aqu_id'].isin(target_set)]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    cols = ['date', 'aqu_id'] + [f for f in fields if f in df.columns]
    return df[cols].copy()


def build_features_no_grfr(ch_id, comid, wst, subday_subset, swat_day_subset,
                            pcp_cache, tmp_cache, lsunit_wb_data, aquifer_data, channel_mapping):
    """Build hourly features without GRFR target (for inference only)."""
    sub = subday_subset[subday_subset['gis_id'] == ch_id].copy()
    sub = sub.sort_values('datetime').reset_index(drop=True)
    if len(sub) == 0:
        return None

    # Hourly precipitation
    if wst in pcp_cache:
        sub = pd.merge(sub, pcp_cache[wst][['datetime', 'precip_mm']], on='datetime', how='left')
    else:
        sub['precip_mm'] = np.nan

    # Daily channel fields
    if swat_day_subset is not None:
        daily = swat_day_subset[swat_day_subset['gis_id'] == ch_id].drop(columns=['gis_id'], errors='ignore')
        sub = pd.merge(sub, daily, on='date', how='left')

    # Daily temperature
    if wst in tmp_cache:
        sub = pd.merge(sub, tmp_cache[wst], on='date', how='left')
    else:
        sub['tmp_ave'] = np.nan

    # LSUnit water balance
    if lsunit_wb_data is not None and ch_id in channel_mapping:
        lsu_ids = channel_mapping[ch_id].get('lsu_ids', [])
        if lsu_ids:
            lsu_sub = lsunit_wb_data[lsunit_wb_data['lsu_id'].isin(lsu_ids)]
            if len(lsu_sub) > 0:
                lsu_daily = lsu_sub.drop(columns=['lsu_id']).groupby('date').mean().reset_index()
                rename_cols = {c: f'wb_{c}' for c in lsu_daily.columns if c != 'date'}
                lsu_daily = lsu_daily.rename(columns=rename_cols)
                sub = pd.merge(sub, lsu_daily, on='date', how='left')

    # Aquifer
    if aquifer_data is not None and ch_id in channel_mapping:
        aqu_ids = channel_mapping[ch_id].get('aquifer_ids', [])
        if aqu_ids:
            aqu_sub = aquifer_data[aquifer_data['aqu_id'].isin(aqu_ids)]
            if len(aqu_sub) > 0:
                aqu_daily = aqu_sub.drop(columns=['aqu_id']).groupby('date').mean().reset_index()
                rename_cols = {c: f'aqu_{c}' for c in aqu_daily.columns if c != 'date'}
                aqu_daily = aqu_daily.rename(columns=rename_cols)
                sub = pd.merge(sub, aqu_daily, on='date', how='left')

    # Lag features
    for lag in LAG_HOURS:
        sub[f'flo_out_lag{lag}h'] = sub['flo_out'].shift(lag)

    # Precipitation rolling sums
    for w in PRECIP_ROLL_HOURS:
        sub[f'precip_sum_{w}h'] = sub['precip_mm'].rolling(w, min_periods=1).sum()

    # Time features
    sub['sin_hour'] = np.sin(2 * np.pi * sub['hour'] / 24.0)
    sub['cos_hour'] = np.cos(2 * np.pi * sub['hour'] / 24.0)
    doy = sub['datetime'].dt.dayofyear
    sub['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    sub['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)

    sub['comid'] = comid
    sub['channel_id'] = ch_id
    return sub


# ============================================================
# Main orchestrator
# ============================================================
def run_convert_all(regions, workers=4):
    """No-op: weather injection now happens inside run_one_scenario."""
    log.info("Weather injection is integrated into run step (no separate convert needed)")
    return True


def run_swat_all(regions, workers=6, nc_concurrent=3):
    """Run SWAT+ and extract features for all scenarios."""
    log.info("=" * 60)
    log.info("Step 3: Running SWAT+ and extracting features")
    log.info("=" * 60)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r in regions:
        for gcm in ALL_GCMS:
            for ssp in ALL_SSPS:
                tasks.append((r["continent"], r["region"], gcm, ssp))

    log.info(f"Total SWAT+ tasks: {len(tasks)}")

    done = 0
    ok = 0
    errors = 0
    skipped = 0

    NC_CONCURRENT = nc_concurrent
    mgr = Manager()
    nc_sem = mgr.Semaphore(NC_CONCURRENT)
    log.info(f"  NC semaphore: max {NC_CONCURRENT} concurrent NC loads, {workers} total workers")

    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1, initializer=_init_worker, initargs=(nc_sem,)) as executor:
        futures = {executor.submit(run_one_scenario, *t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if "OK" in result:
                ok += 1
            elif "SKIP" in result:
                skipped += 1
            else:
                errors += 1
                log.error(result)

            if done % 10 == 0:
                log.info(f"  Progress: {done}/{len(tasks)} (ok={ok}, skip={skipped}, err={errors})")

    log.info(f"\nSWAT+ complete: {ok} ok, {skipped} skipped, {errors} errors / {done} total")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Future scenario pipeline")
    parser.add_argument("--step", choices=["validate", "run", "all"], default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--nc-concurrent", type=int, default=3, help="max concurrent NC loads")
    parser.add_argument("--region", type=str, default=None, help="e.g. europe_west/luxembourg")
    parser.add_argument("--gcm", type=str, default=None)
    parser.add_argument("--ssp", type=str, default=None)
    parser.add_argument("--continents", type=str, default=None, help="comma-separated, e.g. china,europe_west")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ISIMIP3b Future Scenario Pipeline")
    log.info(f"Period: {YRC_START}-{YRC_END}, warmup={NYSKIP}yr, output from {OUTPUT_START}")
    log.info(f"Workers: {args.workers}")
    log.info("=" * 60)

    if args.region:
        parts = args.region.split("/")
        regions = [{"continent": parts[0], "region": parts[1]}]
    else:
        regions = discover_regions()

    if args.continents:
        allowed = set(args.continents.split(","))
        regions = [r for r in regions if r["continent"] in allowed]
    log.info(f"Regions: {len(regions)}")

    if args.gcm:
        global ALL_GCMS
        ALL_GCMS = [args.gcm]
    if args.ssp:
        global ALL_SSPS
        ALL_SSPS = [args.ssp]

    t_start = time.time()

    if args.step in ("validate", "all"):
        if not validate_regional_nc():
            log.error("Validation failed!")
            sys.exit(1)
        if not validate_txtinout_scenarios():
            log.error("TxtInOut validation failed!")
            sys.exit(1)

    if args.step in ("run", "all"):
        n_errors = run_swat_all(regions, workers=args.workers, nc_concurrent=args.nc_concurrent)

    elapsed = time.time() - t_start
    log.info(f"\nPipeline complete in {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
