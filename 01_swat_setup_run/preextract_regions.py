#!/usr/bin/env python3
"""
Pre-extract regional subsets from global ISIMIP3b NC files.

Reads each global NC file once into memory (~33 GB), slices out all 81 regions'
bbox subsets, saves as small per-region NC files. This eliminates the 2100x read
amplification caused by the [1, 280, 720] chunk layout when extracting small regions.

Output: /data/isimip3b_regional/{continent}/{region}/{gcm}/{scenario}/{filename}.nc

Usage:
  python3 preextract_regions.py                          # all files, 2 workers
  python3 preextract_regions.py --workers 3              # 3 parallel workers
  python3 preextract_regions.py --gcm gfdl-esm4 --var pr # specific GCM/variable
  python3 preextract_regions.py --dry-run                # show what would be done
"""

import argparse
import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GLOBAL_DIR = Path("/data/isimip3b_global")
REGIONAL_DIR = Path("/data/isimip3b_regional")
SWAT_DIR = Path("/data/swat_global")
BUFFER_DEG = 0.3

ALL_GCMS = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
ALL_SCENARIOS = ["ssp126", "ssp370", "ssp585"]
ALL_VARS = ["pr", "tas", "rsds", "hurs", "sfcwind"]


def discover_regions():
    """Find all regions with DEM files."""
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
                    "dem": str(dem_path),
                })
    return regions


def get_dem_bbox(dem_path, buffer_deg=BUFFER_DEG):
    """Extract WGS84 bbox from DEM with buffer."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(dem_path) as src:
        b = src.bounds
        if src.crs and src.crs.to_epsg() != 4326:
            west, south, east, north = transform_bounds(
                src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        else:
            west, south, east, north = b.left, b.bottom, b.right, b.top
    return (west - buffer_deg, south - buffer_deg,
            east + buffer_deg, north + buffer_deg)


def bbox_to_indices(bbox, lats, lons):
    """Convert bbox to lat/lon index ranges. lats may be descending."""
    west, south, east, north = bbox
    lat_ascending = lats[0] < lats[-1]

    if lat_ascending:
        lat_mask = (lats >= south) & (lats <= north)
    else:
        lat_mask = (lats >= south) & (lats <= north)

    lon_mask = (lons >= west) & (lons <= east)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return None

    return (lat_idx[0], lat_idx[-1] + 1, lon_idx[0], lon_idx[-1] + 1)


def extract_one_file(args):
    """Process one global NC file: extract all regions and save."""
    nc_path, regions_with_indices, var_name = args
    import netCDF4 as nc

    fname = os.path.basename(nc_path)
    parts = fname.split("_")
    gcm = parts[0]
    scenario = [p for p in parts if p.startswith("ssp") or p == "historical"][0]

    # Pre-check: skip loading if all regions already extracted
    all_done = True
    for r in regions_with_indices:
        if r["indices"] is None:
            continue
        out_path = REGIONAL_DIR / r["continent"] / r["region"] / gcm / scenario / fname
        if not out_path.exists() or out_path.stat().st_size == 0:
            all_done = False
            break
    if all_done:
        log.info(f"  Skip {fname}: all regions already extracted")
        return fname, 0, 0

    t0 = time.time()
    log.info(f"  Loading {fname} ...")

    try:
        ds = nc.Dataset(nc_path, "r")
    except Exception as e:
        log.error(f"  Failed to open {nc_path}: {e}")
        return fname, 0, 0

    lats = ds["lat"][:]
    lons = ds["lon"][:]
    times = ds["time"]

    data = ds[var_name][:]  # full array into RAM
    load_time = time.time() - t0
    log.info(f"  Loaded {fname}: shape={data.shape}, {load_time:.0f}s")

    n_saved = 0
    n_skipped = 0

    for r in regions_with_indices:
        continent = r["continent"]
        region = r["region"]
        idx = r["indices"]
        if idx is None:
            continue

        lat_s, lat_e, lon_s, lon_e = idx

        out_dir = REGIONAL_DIR / continent / region / gcm / scenario
        out_path = out_dir / fname

        if out_path.exists() and out_path.stat().st_size > 0:
            n_skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        sub_data = data[:, lat_s:lat_e, lon_s:lon_e]
        sub_lats = lats[lat_s:lat_e]
        sub_lons = lons[lon_s:lon_e]

        n_time, n_lat, n_lon = sub_data.shape
        chunk_time = min(720, n_time)

        try:
            out_ds = nc.Dataset(str(out_path), "w", format="NETCDF4")
            out_ds.createDimension("time", n_time)
            out_ds.createDimension("lat", n_lat)
            out_ds.createDimension("lon", n_lon)

            t_var = out_ds.createVariable("time", times.datatype, ("time",))
            t_var[:] = times[:]
            for attr in times.ncattrs():
                t_var.setncattr(attr, times.getncattr(attr))

            lat_var = out_ds.createVariable("lat", "f8", ("lat",))
            lat_var[:] = sub_lats
            lat_var.units = "degrees_north"

            lon_var = out_ds.createVariable("lon", "f8", ("lon",))
            lon_var[:] = sub_lons
            lon_var.units = "degrees_east"

            v = out_ds.createVariable(
                var_name, "f4", ("time", "lat", "lon"),
                chunksizes=(chunk_time, n_lat, n_lon),
                zlib=True, complevel=1, shuffle=True,
            )
            v[:] = sub_data

            src_var = ds[var_name]
            for attr in src_var.ncattrs():
                if attr != "_FillValue":
                    v.setncattr(attr, src_var.getncattr(attr))

            out_ds.close()
            n_saved += 1
        except Exception as e:
            log.error(f"  Error writing {out_path}: {e}")
            if out_path.exists():
                out_path.unlink()

    ds.close()
    del data
    elapsed = time.time() - t0
    log.info(f"  Done {fname}: saved={n_saved}, skipped={n_skipped}, {elapsed:.0f}s")
    return fname, n_saved, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gcm", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--var", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gcms = [args.gcm] if args.gcm else ALL_GCMS
    scenarios = [args.scenario] if args.scenario else ALL_SCENARIOS
    variables = [args.var] if args.var else ALL_VARS

    log.info("=== ISIMIP3b Regional Pre-extraction ===")
    log.info(f"GCMs: {gcms}")
    log.info(f"Scenarios: {scenarios}")
    log.info(f"Variables: {variables}")
    log.info(f"Workers: {args.workers}")

    log.info("Discovering regions ...")
    regions = discover_regions()
    log.info(f"Found {len(regions)} regions")

    log.info("Computing bboxes from DEMs ...")
    for r in regions:
        r["bbox"] = get_dem_bbox(r["dem"])

    nc_files = []
    for gcm in gcms:
        for scenario in scenarios:
            gcm_dir = GLOBAL_DIR / gcm / scenario
            if not gcm_dir.exists():
                log.warning(f"  Missing: {gcm_dir}")
                continue
            for f in sorted(gcm_dir.glob("*.nc")):
                var_name = None
                for v in variables:
                    if f"_{v}_" in f.name:
                        var_name = v
                        break
                if var_name:
                    nc_files.append((str(f), var_name))

    log.info(f"Total NC files to process: {len(nc_files)}")

    if args.dry_run:
        for f, v in nc_files[:10]:
            log.info(f"  {os.path.basename(f)} (var={v})")
        if len(nc_files) > 10:
            log.info(f"  ... and {len(nc_files) - 10} more")
        return

    sample_ds = None
    for f, v in nc_files:
        try:
            import netCDF4 as nc
            sample_ds = nc.Dataset(f, "r")
            sample_lats = sample_ds["lat"][:]
            sample_lons = sample_ds["lon"][:]
            sample_ds.close()
            break
        except Exception:
            continue

    if sample_lats is None:
        log.error("Cannot read any NC file to get coordinate grid")
        return

    for r in regions:
        r["indices"] = bbox_to_indices(r["bbox"], sample_lats, sample_lons)
        idx = r["indices"]
        if idx:
            n_lat = idx[1] - idx[0]
            n_lon = idx[3] - idx[2]
            log.info(f"  {r['continent']}/{r['region']}: {n_lat}x{n_lon} grid points")
        else:
            log.warning(f"  {r['continent']}/{r['region']}: NO grid points in bbox!")

    tasks = []
    for nc_path, var_name in nc_files:
        tasks.append((nc_path, regions, var_name))

    t_start = time.time()

    if args.workers <= 1:
        results = []
        for task in tasks:
            results.append(extract_one_file(task))
    else:
        with Pool(processes=args.workers) as pool:
            results = list(pool.imap_unordered(extract_one_file, tasks, chunksize=1))

    total_time = time.time() - t_start
    total_saved = sum(r[1] for r in results)
    log.info(f"\n{'='*60}")
    log.info(f"Pre-extraction complete!")
    log.info(f"Files processed: {len(results)}")
    log.info(f"Regional files saved: {total_saved}")
    log.info(f"Total time: {total_time/3600:.1f} hours")
    log.info(f"Output: {REGIONAL_DIR}")


if __name__ == "__main__":
    main()
