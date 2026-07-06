#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWAT+ Input Data Preparation Tool v4
=====================================

v4 New features:
  - DEM resampling (dem_resolution): 90m/250m/500m etc.
  - Outputs in subfolders: rasters/, vectors/, figures/, intermediate/
  - Each step auto-saves a PNG to figures/

Output directory structure:
  swat_output/
  ├── rasters/          <- QSWAT+ rasters (dem.tif, landuse.tif, soil.tif)
  ├── vectors/          <- QSWAT+ vectors (rivers.shp, basins.shp, ...)
  ├── figures/          <- Visualization PNGs
  ├── intermediate/     <- Intermediate files (can be deleted)
  └── raw_downloads/    <- Raw downloads (SWAT official data)

DEM resolution (dem_resolution):
  None = keep original resolution (default)
  90   = 90m  (recommended for country scale)
  250  = 250m (large countries / quick test)
  500  = 500m (very large regions)
"""

import os, sys, zipfile, warnings, argparse, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests, rasterio, rasterio.crs
from pathlib import Path
from pyproj import CRS
from shapely.geometry import box, mapping, Point
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds

warnings.filterwarnings("ignore")

# =============================================================================
# Logging verbosity control
# =============================================================================
# Set to False to suppress detailed per-step messages (batch mode).
# Even when VERBOSE=False, key milestones and errors are always printed.
VERBOSE = True

def _log(msg, force=False, **kwargs):
    """Print msg only when VERBOSE or force is True."""
    if VERBOSE or force:
        print(msg, **kwargs)

def _log_step(title, force=True):
    """Print a step header (always shown)."""
    print(f"\n  >> {title}")

# =============================================================================
# Constants
# =============================================================================
SWAT_URLS = {
    "landuse": {
        "af": {"orig": "https://swat.tamu.edu/media/116396/af_landuse.zip",
               "resamp": "https://swat.tamu.edu/media/116397/af_landuse_newres.zip"},
        "ap": {"orig": "https://swat.tamu.edu/media/116398/ap_landuse.zip",
               "resamp": "https://swat.tamu.edu/media/116399/ap_landuse_newres.zip"},
        "ea": {"orig": "https://swat.tamu.edu/media/116400/ea_landuse.zip",
               "resamp": "https://swat.tamu.edu/media/116401/ea_landuse_newres.zip"},
        "na": {"orig": "https://swat.tamu.edu/media/116402/na_landuse.zip",
               "resamp": "https://swat.tamu.edu/media/116403/na_landuse_newres.zip"},
        "sa": {"orig": "https://swat.tamu.edu/media/116404/sa_landuse.zip",
               "resamp": "https://swat.tamu.edu/media/116405/sa_landuse_newres.zip"},
    },
    "soil": {
        "af": "https://swat.tamu.edu/media/116406/af_soil.zip",
        "ap": "https://swat.tamu.edu/media/116407/ap_soil.zip",
        "ea": "https://swat.tamu.edu/media/116408/ea_soil.zip",
        "na": "https://swat.tamu.edu/media/116409/na_soil.zip",
        "sa": "https://swat.tamu.edu/media/116410/sa_soil.zip",
    },
}

RIVER_DETAIL_MAP = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}

# =============================================================================
# Output directory management
# =============================================================================
class OutputDirs:
    def __init__(self, base_dir):
        self.base = Path(base_dir)
        self.rasters = self.base / "rasters"
        self.vectors = self.base / "vectors"
        self.figures = self.base / "figures"
        self.intermediate = self.base / "intermediate"
        self.raw_downloads = self.base / "raw_downloads"

    def create_all(self):
        for d in [self.rasters, self.vectors, self.figures,
                  self.intermediate, self.raw_downloads]:
            d.mkdir(parents=True, exist_ok=True)
        return self


def save_figure(fig, fig_path, dpi=150):
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"  [Fig] Saved: {fig_path}")


# =============================================================================
# Utility functions
# =============================================================================
def download_and_extract(url, zip_path, extract_dir):
    zip_path, extract_dir = Path(zip_path), Path(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        _log(f"  [Skip] Already exists: {extract_dir.name}"); return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        _log(f"  [Download] {zip_path.name} ...")
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dl = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk); dl += len(chunk)
                if total: _log(f"\r    {dl/total*100:.0f}% ({dl//1048576}MB/{total//1048576}MB)", end="")
        print()
    _log(f"  [Extract] {zip_path.name}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z: z.extractall(extract_dir)
    _log(f"  [OK] Done")


def get_country_boundary(country_name):
    print(f"[Country] Getting boundary: {country_name}")
    try:
        world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    except Exception:
        try: world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            import geodatasets; world = gpd.read_file(geodatasets.data.naturalearth.land110)
    name_col = None
    for c in ["name","NAME","ADMIN","NAME_EN","admin","name_en","NAME_LONG"]:
        if c in world.columns: name_col = c; break
    if name_col is None: raise ValueError(f"Country name column not found. Available: {list(world.columns)}")
    country_gdf = world[world[name_col].str.contains(country_name, case=False, na=False)]
    if len(country_gdf) == 0:
        raise ValueError(f"Country not found: '{country_name}'\nAvailable: {sorted(world[name_col].dropna().unique()[:50])}")
    _log(f"  [OK] Found: {country_name}")
    return country_gdf.copy()


def get_study_area(country=None, outlet_lon=None, outlet_lat=None, bbox=None, hydrobasins_shp=None):
    if outlet_lon is not None and outlet_lat is not None:
        print(f"[Study area] Using outlet: ({outlet_lon}, {outlet_lat})")
        outlet = Point(outlet_lon, outlet_lat)
        if isinstance(hydrobasins_shp, list):
            basins_all = pd.concat([gpd.read_file(f) for f in hydrobasins_shp], ignore_index=True)
        else:
            basins_all = gpd.read_file(hydrobasins_shp)
        containing = basins_all[basins_all.contains(outlet)]
        if len(containing) == 0:
            basins_all["dist"] = basins_all.geometry.distance(outlet)
            containing = basins_all.nsmallest(1, "dist")
            print("  [Warning] Outlet not within any basin, using nearest")
        study_area = containing.copy()
    elif bbox is not None:
        print(f"[Study area] Using bounding box: {bbox}")
        study_area = gpd.GeoDataFrame({"geometry": [box(*bbox)]}, crs="EPSG:4326")
    elif country is not None:
        study_area = get_country_boundary(country)
    else:
        raise ValueError("Must specify one of: country, outlet_lon/lat, or bbox")
    if study_area.crs is None: study_area = study_area.set_crs("EPSG:4326")
    else: study_area = study_area.to_crs("EPSG:4326")
    return study_area


# =============================================================================
# Step functions
# =============================================================================
def step_clip_basins(study_area, hydrobasins_shp, min_area_km2=50, dirs=None):
    print(f"\n{'='*50}\n[Step] Clip basins (GRFR)")
    
    # 1. Compute the bounding box (keep the existing optimization)
    bounds = study_area.to_crs("EPSG:4326").total_bounds
    buf = 0.5 
    read_bbox = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
    print(f"  [Optimization] Using bbox filter: {read_bbox}")

    print(f"  [Load] {hydrobasins_shp}")
    if isinstance(hydrobasins_shp, list):
        dfs = []
        for f in hydrobasins_shp:
            try:
                chunk = gpd.read_file(f, bbox=read_bbox)
                if not chunk.empty:
                    dfs.append(chunk)
            except Exception as e:
                print(f"  [Warning] Failed to read {f} with bbox: {e}")
        if not dfs:
            raise ValueError("No basin data found within the study area bbox!")
        basins_all = pd.concat(dfs, ignore_index=True)
        print(f"  [Load] Merged {len(dfs)} files (bbox filtered)")
    else:
        basins_all = gpd.read_file(hydrobasins_shp, bbox=read_bbox)
    # GRFR shapefiles may lack embedded CRS - default to WGS84
    if not isinstance(basins_all, gpd.GeoDataFrame):
        basins_all = gpd.GeoDataFrame(basins_all, geometry="geometry")
    if basins_all.crs is None:
        basins_all = basins_all.set_crs("EPSG:4326")

    print(f"  Basins in bbox: {len(basins_all)}")
    if len(basins_all) == 0:
        raise ValueError("No basins found in the study area bounding box!")

    # --- Start of core optimization ---
    print(f"  [Optimization] Filtering via Spatial Join (sjoin)...")

    # A. Align coordinate reference systems
    if basins_all.crs != study_area.crs:
        study_area_filter = study_area.to_crs(basins_all.crs)
    else:
        study_area_filter = study_area.copy()

    # B. Geometry simplification (key step!)
    # Natural Earth 10m is too detailed and can stall the computation.
    # 0.005 degrees is about 500 m, precise enough for basin filtering while
    # removing more than 90% of the vertices.
    study_area_filter.geometry = study_area_filter.geometry.simplify(tolerance=0.005, preserve_topology=True)

    # C. Use sjoin instead of intersects
    # sjoin uses an R-tree index, over 100x faster than checking intersects one by one
    basins = gpd.sjoin(basins_all, study_area_filter, how="inner", predicate="intersects")

    # D. Deduplicate & clean up
    # If a basin spans two provinces, sjoin produces two rows, so deduplicate
    basins = basins[~basins.index.duplicated()].copy()
    if "index_right" in basins.columns:
        basins = basins.drop(columns=["index_right"])
    # --- End of core optimization ---

    print(f"  Intersecting study area: {len(basins)}")
    
    # GRFR area column may be 'unitarea', 'area', or HydroBasins 'SUB_AREA'
    area_col_basin = next((c for c in ["unitarea", "area", "SUB_AREA", "AREA_KM2"]
                           if c in basins.columns), None)
    if area_col_basin and min_area_km2 > 0:
        basins = basins[basins[area_col_basin] >= min_area_km2]
        print(f"  After area >= {min_area_km2} km² filter: {len(basins)}")
        
    if dirs:
        fig, ax = plt.subplots(figsize=(10, 8))
        basins.plot(ax=ax, edgecolor="blue", facecolor="lightblue", alpha=0.5, linewidth=0.5)
        study_area.boundary.plot(ax=ax, edgecolor="red", linewidth=2, linestyle="--", label="Study area boundary")
        ax.set_title(f"Sub-basins: {len(basins)}", fontsize=14); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); save_figure(fig, dirs.figures / "01_basins.png")
    return basins


def step_clip_rivers(basins, hydrorivers_shp, river_detail=3, hydrolakes_shp=None, dirs=None):
    print(f"\n{'='*50}\n[Step] Clip rivers (GRFR)")
    if hydrorivers_shp is None:
        _log(f"  [Warning] No river shapefile(s) provided"); return None

    min_order = RIVER_DETAIL_MAP.get(river_detail, 4)
    _log(f"  River detail level: {river_detail} (stream order >= {min_order})")

    bounds = basins.to_crs("EPSG:4326").total_bounds; buf = 0.1
    read_bbox = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)

    _log(f"  Loading GRFR rivers (bbox pre-filter)...")
    # Support both single path and list of paths (GRFR pfaf tiles)
    shp_list = hydrorivers_shp if isinstance(hydrorivers_shp, list) else [hydrorivers_shp]
    frames = []
    for shp in shp_list:
        if not Path(shp).exists():
            continue
        try:
            chunk = gpd.read_file(shp, bbox=read_bbox)
            if len(chunk) > 0:
                frames.append(chunk)
        except Exception as e:
            _log(f"  [Warning] Failed to read {shp}: {e}")
    if not frames:
        _log(f"  [Warning] No GRFR river data found for this region"); return None
    rivers = pd.concat(frames, ignore_index=True)
    if not isinstance(rivers, gpd.GeoDataFrame):
        rivers = gpd.GeoDataFrame(rivers, geometry="geometry")
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    _log(f"  River segments in bbox: {len(rivers)}")

    # GRFR uses 'order' or 'strmOrder' for stream order; fall back gracefully
    order_col = next((c for c in rivers.columns
                      if c.lower() in ("order", "strmorder", "ord_stra", "streamorde", "strahler")),
                     None)
    # 1. Apply attribute filter first (drop small streams early)
    if order_col:
        rivers = rivers[rivers[order_col] >= min_order].copy()
        _log(f"  After stream order >= {min_order} filter: {len(rivers)}")
    
    if len(rivers) == 0:
        return rivers

    # 2. Geometry simplification optimization (key step!)
    # Create a simplified copy of the basins for spatial queries, tolerance 0.005 degrees (~500m)
    _log(f"  [Optimization] Simplifying basins for spatial query...")
    basins_simple = basins.to_crs("EPSG:4326").copy()
    basins_simple.geometry = basins_simple.geometry.simplify(0.005, preserve_topology=True)

    rivers = rivers.to_crs("EPSG:4326")

    # Run sjoin with the simplified basins, which is extremely fast
    joined = gpd.sjoin(rivers, basins_simple, how="inner", predicate="intersects")
    rivers = rivers.loc[joined.index.unique()].copy()
    _log(f"  River segments within basins: {len(rivers)}")

    if dirs and len(rivers) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        # For plotting, still draw the original high-resolution basins
        basins.to_crs("EPSG:4326").plot(ax=ax, edgecolor="gray", facecolor="lightyellow", alpha=0.4, linewidth=0.3)
        if order_col and order_col in rivers.columns:
            for ord_val in sorted(rivers[order_col].unique()):
                subset = rivers[rivers[order_col] == ord_val]
                subset.plot(ax=ax, color="steelblue", linewidth=max(0.2, ord_val*0.3))
        else:
            rivers.plot(ax=ax, color="steelblue", linewidth=0.5)
        ax.set_title(f"River network: {len(rivers)} segments (detail={river_detail}, order>={min_order})", fontsize=14)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        save_figure(fig, dirs.figures / "02_rivers.png")
    return rivers


def step_download_swat_data(continent, dirs, use_original_landuse=False):
    # Normalize to a list
    continents = continent if isinstance(continent, list) else [continent]
    print(f"\n{'='*50}\n[Step] Download SWAT data (continent: {continents})")
    raw_dir = dirs.raw_downloads
    for cont in continents:
        lu_key = "orig" if use_original_landuse else "resamp"
        lu_url = SWAT_URLS["landuse"][cont][lu_key]
        print(f"\n  Land use [{cont}] ({'~400m' if use_original_landuse else '~800m'}):")
        download_and_extract(lu_url, raw_dir / f"{cont}_landuse.zip", raw_dir / f"{cont}_landuse")
        print(f"\n  Soil [{cont}] (FAO Soil Map):")
        download_and_extract(SWAT_URLS["soil"][cont], raw_dir / f"{cont}_soil.zip", raw_dir / f"{cont}_soil")
    print(f"\n  Downloaded files:")
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(".zip"):
            _log(f"    {p.relative_to(raw_dir)}  ({p.stat().st_size/1048576:.1f} MB)")


def resample_dem(src_path, dst_path, target_resolution_m):
    """Resample DEM to the specified resolution (metres)."""
    _log(f"  [DEM] Resampling -> {target_resolution_m}m")
    with rasterio.open(src_path) as src:
        src_res_x = abs(src.transform.a)
        if src.crs and src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2
            m_per_deg = 111320 * np.cos(np.radians(center_lat))
            current_res_m = src_res_x * m_per_deg
            new_res = target_resolution_m / m_per_deg
        else:
            current_res_m = src_res_x
            new_res = target_resolution_m
        _log(f"    Current resolution: ~{current_res_m:.1f}m, target: {target_resolution_m}m")
        if current_res_m >= target_resolution_m * 0.9:
            _log(f"    [Skip] Current resolution already >= target, skipping")
            shutil.copy2(src_path, dst_path); return dst_path
        new_w = max(1, int((src.bounds.right - src.bounds.left) / new_res))
        new_h = max(1, int((src.bounds.top - src.bounds.bottom) / new_res))
        new_tf = from_bounds(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, new_w, new_h)
        data = src.read(out_shape=(src.count, new_h, new_w), resampling=Resampling.bilinear)
        meta = src.meta.copy()
        meta.update(width=new_w, height=new_h, transform=new_tf, compress="lzw")
        with rasterio.open(dst_path, "w", **meta) as dst: dst.write(data)
        ratio = (src.width * src.height) / (new_w * new_h)
        _log(f"    {src.width}x{src.height} -> {new_w}x{new_h} (downscale {ratio:.1f}x)")
    return dst_path


def step_prepare_dem(basins, srtm_dir, dirs, dem_resolution=None):
    """
    Mosaic and clip DEM tiles.
    For areas crossing 60°N: use SRTM (≤60°N) + Copernicus (>60°N) and merge.
    For areas entirely ≤60°N: use SRTM only.
    For areas entirely >60°N: use Copernicus only.
    """
    print(f"\n{'='*50}\n[Step] Prepare DEM")
    if dem_resolution: _log(f"  Target resolution: {dem_resolution}m")
    dem_path = dirs.intermediate / "dem_wgs84.tif"
    
    bounds = basins.to_crs("EPSG:4326").total_bounds  # (minx, miny, maxx, maxy)
    buf = 0.2
    clip_bounds = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
    
    min_lat = bounds[1]  # southernmost latitude of the study area
    max_lat = bounds[3]  # northernmost latitude of the study area

    _log(f"  Study area latitude range: {min_lat:.2f}°N to {max_lat:.2f}°N")

    dem_parts = []  # paths to the individual DEM parts
    
    # ========== Part 1: SRTM for ≤60°N ==========
    if min_lat < 60.0:
        _log(f"  [SRTM] Processing area ≤60°N...")
        srtm_files = list(Path(srtm_dir).rglob("*.tif"))
        _log(f"    Found {len(srtm_files)} SRTM tiles total")
        
        # SRTM only covers up to 60°N, so clip the bounds accordingly
        srtm_clip_bounds = (clip_bounds[0], clip_bounds[1], clip_bounds[2], min(clip_bounds[3], 60.0))
        
        relevant = []
        for f in srtm_files:
            with rasterio.open(f) as src:
                tb = src.bounds
                if tb.right > srtm_clip_bounds[0] and tb.left < srtm_clip_bounds[2] and \
                   tb.top > srtm_clip_bounds[1] and tb.bottom < srtm_clip_bounds[3]:
                    relevant.append(f)
        _log(f"    Overlapping study area (≤60°N): {len(relevant)} tiles")
        
        if len(relevant) > 0:
            datasets = [rasterio.open(p) for p in relevant]
            mosaic, out_transform = merge(datasets)
            out_meta = datasets[0].meta.copy()
            out_meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=out_transform, nodata=-32768)
            for ds in datasets: ds.close()
            
            srtm_mosaic = dirs.intermediate / "_tmp_srtm_mosaic.tif"
            with rasterio.open(srtm_mosaic, "w", **out_meta) as dst: 
                dst.write(mosaic)
            dem_parts.append(srtm_mosaic)
            _log(f"    [OK] SRTM mosaic created")
        else:
            _log(f"    [Warning] No SRTM tiles found for ≤60°N area")
    
    # ========== Part 2: Copernicus for >60°N ==========
    if max_lat > 60.0:
        _log(f"  [Copernicus] Processing area >60°N...")
        # Copernicus only downloads the portion above 60°N (when SRTM covers the part below)
        # If the entire region is above 60°N, download all of it
        if max_lat > 60.0:
            _log(f"  [Copernicus] Processing area >60°N...")
            if min_lat >= 60.0:
                cop_bounds = clip_bounds
            else:
                # [Change here] Use 59.0 instead of 60.0 to create an overlap zone with SRTM
                cop_bounds = (clip_bounds[0], 59.0, clip_bounds[2], clip_bounds[3])
        
        cop_dem = _download_copernicus_dem_tiles(cop_bounds, dirs)
        if cop_dem is not None:
            dem_parts.append(cop_dem)
            _log(f"    [OK] Copernicus DEM created")
        else:
            _log(f"    [Warning] Failed to download Copernicus DEM for >60°N area")
    
    # ========== Merge all parts ==========
    if len(dem_parts) == 0:
        print("  [Error] No DEM data available!")
        return None
    elif len(dem_parts) == 1:
        tmp_merged = dem_parts[0]
    else:
        _log(f"  [Merge] Merging {len(dem_parts)} DEM sources...")
        datasets = [rasterio.open(p) for p in dem_parts]
        
        # Pass nodata to ensure transparent-background merging
        mosaic, out_transform = merge(datasets, nodata=-32768)
        
        out_meta = datasets[0].meta.copy()
        out_meta.update(
            height=mosaic.shape[1], 
            width=mosaic.shape[2], 
            transform=out_transform, 
            nodata=-32768, 
            compress="lzw",
            dtype=rasterio.float32,
            BIGTIFF="YES"  # also enable BigTIFF support here
        )
        mosaic = mosaic.astype(np.float32)

        for ds in datasets: 
            ds.close()
        
        tmp_merged = dirs.intermediate / "_tmp_dem_merged.tif"
        with rasterio.open(tmp_merged, "w", **out_meta) as dst:
            dst.write(mosaic)
    
    # ========== Clip to basin boundary ==========
    _log(f"  [Clip] Clipping to basin boundaries...")
    basins_4326 = basins.to_crs("EPSG:4326")
    basins_dissolved = basins_4326.dissolve().buffer(buf)
    clip_geoms = [mapping(g) for g in basins_dissolved]
    
    with rasterio.open(tmp_merged) as src:
        clipped, clipped_tf = rio_mask(src, clip_geoms, crop=True, nodata=-32768, all_touched=True)
        clip_meta = src.meta.copy()
        clip_meta.update(height=clipped.shape[1], width=clipped.shape[2], transform=clipped_tf, nodata=-32768, compress="lzw")
    
    with rasterio.open(dem_path, "w", **clip_meta) as dst:
        dst.write(clipped)
    
    # Clean up temporary files
    for p in dem_parts:
        p.unlink(missing_ok=True)
    if len(dem_parts) > 1:
        tmp_merged.unlink(missing_ok=True)

    # Optional resampling
    if dem_resolution:
        resampled_path = dirs.intermediate / "dem_wgs84_resampled.tif"
        resample_dem(dem_path, resampled_path, dem_resolution)
        dem_path = resampled_path

    size_mb = dem_path.stat().st_size / 1048576
    with rasterio.open(dem_path) as src:
        _log(f"  [OK] DEM: {dem_path.name} ({src.width}x{src.height}, {size_mb:.1f} MB)")
    return dem_path


def _download_copernicus_dem_tiles(bounds, dirs):
    """
    Download Copernicus DEM GLO-30 from AWS Open Data.
    No API key required, no rate limits.
    
    bounds: (minx, miny, maxx, maxy) in WGS84
    Returns: Path to merged DEM mosaic (not clipped)
    """
    import math
    
    cop_dem_path = dirs.intermediate / "_tmp_cop_dem.tif"
    
    west, south, east, north = bounds
    print(f"    Download bounds: W={west:.2f}, S={south:.2f}, E={east:.2f}, N={north:.2f}")
    
    tile_dir = dirs.intermediate / "_cop_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    
    def is_valid_tif(filepath):
        try:
            with rasterio.open(filepath) as src:
                _ = src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            return True
        except:
            return False
    
    def get_cop30_aws_url(lat, lon):
        """
        Build the AWS URL for a Copernicus GLO-30 tile.
        File structure on AWS:
        s3://copernicus-dem-30m/Copernicus_DSM_COG_10_Nxx_00_Exxx_00_DEM/Copernicus_DSM_COG_10_Nxx_00_Exxx_00_DEM.tif
        lat/lon are the coordinates of the tile's lower-left corner
        """
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        lat_abs = abs(lat)
        lon_abs = abs(lon)
        
        # The directory name and the file name are identical
        tile_name = f"Copernicus_DSM_COG_10_{lat_prefix}{lat_abs:02d}_00_{lon_prefix}{lon_abs:03d}_00_DEM"
        # The full URL includes the subdirectory
        url = f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{tile_name}/{tile_name}.tif"
        return url, f"{tile_name}.tif"
    
    tile_files = []
    failed_tiles = []
    skipped_tiles = []
    
    # Compute the range of tiles to download (1x1 degree each)
    lat_min = math.floor(south)
    lat_max = math.ceil(north)
    lon_min = math.floor(west)
    lon_max = math.ceil(east)
    
    total_tiles = (lat_max - lat_min) * (lon_max - lon_min)
    print(f"    Need to download up to {total_tiles} tiles (1x1 degree each)")
    
    tile_idx = 0
    for lat in range(lat_min, lat_max):
        for lon in range(lon_min, lon_max):
            tile_idx += 1
            url, tile_name = get_cop30_aws_url(lat, lon)
            tile_file = tile_dir / tile_name
            
            # Check for an already-existing file
            if tile_file.exists():
                if is_valid_tif(tile_file):
                    print(f"    [{tile_idx}/{total_tiles}] [Skip] Already exists: {tile_name}")
                    tile_files.append(tile_file)
                    continue
                else:
                    tile_file.unlink()
            
            print(f"    [{tile_idx}/{total_tiles}] [Download] {tile_name}")
            
            try:
                resp = requests.get(url, stream=True, timeout=300)
                
                if resp.status_code == 404:
                    # This area may have no data (e.g. ocean areas)
                    print(f"      [Skip] No data (404 - ocean)")
                    skipped_tiles.append(tile_name)
                    continue
                elif resp.status_code == 403:
                    print(f"      [Error] Access denied (403)")
                    failed_tiles.append((tile_name, "403 Forbidden"))
                    continue
                
                resp.raise_for_status()
                
                # Download the file
                total_bytes = 0
                with open(tile_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        total_bytes += len(chunk)
                
                if is_valid_tif(tile_file):
                    tile_files.append(tile_file)
                    print(f"      [OK] {total_bytes / 1024 / 1024:.2f} MB")
                else:
                    print(f"      [Error] Invalid GeoTIFF")
                    tile_file.unlink()
                    failed_tiles.append((tile_name, "Invalid GeoTIFF"))
                    
            except requests.exceptions.RequestException as e:
                print(f"      [Error] Request failed: {e}")
                failed_tiles.append((tile_name, str(e)))
            except Exception as e:
                print(f"      [Error] {e}")
                failed_tiles.append((tile_name, str(e)))
    
    # Print statistics
    print(f"    [Summary] Downloaded: {len(tile_files)}, Skipped (ocean): {len(skipped_tiles)}, Failed: {len(failed_tiles)}")
    
    if len(tile_files) == 0:
        print("    [Error] No valid DEM tiles downloaded!")
        return None
    
    # Merge all tiles
    print(f"    [Merge] Merging {len(tile_files)} tiles...")
    datasets = [rasterio.open(p) for p in tile_files]
    mosaic, out_transform = merge(datasets)

    # [Key change]: cast the array to float32 and normalize all possible NoData values to -32768
    mosaic = mosaic.astype(np.float32)
    cop_nodata = datasets[0].nodata
    if cop_nodata is not None:
        mosaic[mosaic == cop_nodata] = -32768
    mosaic[mosaic <= -32767] = -32768
    mosaic[np.isnan(mosaic)] = -32768
    
    out_meta = datasets[0].meta.copy()
    out_meta.update(
        height=mosaic.shape[1], 
        width=mosaic.shape[2], 
        transform=out_transform,
        nodata=-32768,
        dtype=rasterio.float32,
        compress="lzw",
        BIGTIFF="YES"  # [Key change]: break past the 4GB limit to prevent southern data from being truncated!
    )
    for ds in datasets:
        ds.close()
    
    with rasterio.open(cop_dem_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    
    print(f"    [OK] Copernicus DEM mosaic: {cop_dem_path}")
    return cop_dem_path

def step_reproject_rasters(basins, dirs, continent, target_epsg, dem_wgs_path=None):
    """Clip and reproject all rasters to target UTM CRS."""
    continents = continent if isinstance(continent, list) else [continent]
    print(f"\n{'='*50}\n[Step] Clip & reproject rasters -> EPSG:{target_epsg}")
    dst_crs = CRS.from_epsg(target_epsg)
    dst_crs_rio = rasterio.crs.CRS.from_wkt(dst_crs.to_wkt())
    std_4326_rio = rasterio.crs.CRS.from_wkt(CRS.from_epsg(4326).to_wkt())
    buf = 0.2
    clip_bounds = basins.to_crs("EPSG:4326").total_bounds
    # Clip to actual basin boundary (dissolve + buffer) instead of bbox
    basins_4326 = basins.to_crs("EPSG:4326")
    basins_dissolved = basins_4326.dissolve().buffer(buf)
    clip_geojson = [mapping(g) for g in basins_dissolved]
    tlon = (clip_bounds[0]-buf, clip_bounds[2]+buf); tlat = (clip_bounds[1]-buf, clip_bounds[3]+buf)

    def find_covering(directory):
        directory = Path(directory); all_r = []
        for pat in ["*.tif","*.tiff","**/hdr.adf","*.bil","*.img"]: all_r.extend(directory.rglob(pat))
        covering = []
        for f in sorted(all_r):
            with rasterio.open(f) as s: b = s.bounds
            if (b.right>tlon[0] and b.left<tlon[1] and b.top>tlat[0] and b.bottom<tlat[1]) or \
               (b.top>tlon[0] and b.bottom<tlon[1] and b.right>tlat[0] and b.left<tlat[1]):
                covering.append(f)
        _log(f"    Found {len(covering)}/{len(all_r)} tiles covering study area"); return covering

    def merge_and_clip(raster_paths, out_name, nodata_val=-9999):
        if not raster_paths: return None
        _log(f"    Preparing {out_name}: {len(raster_paths)} tiles")
        std_4326 = std_4326_rio; fixed = []
        for rp in raster_paths:
            with rasterio.open(rp) as src:
                src_crs = src.crs
                # SWAT official rasters sometimes lack embedded CRS but are always WGS84
                if src_crs is None:
                    _log(f"      [Warning] {Path(rp).name} has no CRS, defaulting to WGS84")
                    src_crs = std_4326
                    # Write temporary copy with CRS embedded
                    tmp = dirs.intermediate / f"_tmp_fixcrs_{out_name}_{len(fixed)}.tif"
                    meta = src.meta.copy()
                    meta.update(crs=std_4326, driver="GTiff")
                    with rasterio.open(tmp, "w", **meta) as d:
                        d.write(src.read())
                    fixed.append(tmp)
                elif src_crs == std_4326: fixed.append(rp)
                else:
                    tf,w,h = calculate_default_transform(src_crs, std_4326, src.width, src.height, *src.bounds)
                    data = src.read(); meta = src.meta.copy()
                    dst_data = np.zeros((meta["count"],h,w), dtype=data.dtype)
                    reproject(source=data, destination=dst_data, src_transform=meta["transform"], src_crs=src_crs, dst_transform=tf, dst_crs=std_4326, resampling=Resampling.nearest)
                    tmp = dirs.intermediate / f"_tmp_fix_{out_name}_{len(fixed)}.tif"
                    meta.update(crs=std_4326, transform=tf, width=w, height=h, driver="GTiff")
                    with rasterio.open(tmp,"w",**meta) as d: d.write(dst_data)
                    fixed.append(tmp)
        if len(fixed)==1: mosaic_path = fixed[0]
        else:
            dsets = [rasterio.open(p) for p in fixed]; md, mt = merge(dsets)
            meta = dsets[0].meta.copy(); meta.update(height=md.shape[1], width=md.shape[2], transform=mt, driver="GTiff")
            if meta.get("crs") is None: meta["crs"] = std_4326  # safety: ensure CRS set
            for d in dsets: d.close()
            mosaic_path = dirs.intermediate / f"_tmp_mosaic_{out_name}.tif"
            with rasterio.open(mosaic_path,"w",**meta) as d: d.write(md)
        with rasterio.open(mosaic_path) as src:
            try:
                cl, ct = rio_mask(src, clip_geojson, crop=True, nodata=src.nodata or nodata_val, all_touched=True)
                meta = src.meta.copy(); meta.update(height=cl.shape[1], width=cl.shape[2], transform=ct, nodata=src.nodata or nodata_val, driver="GTiff", compress="lzw")
                if meta.get("crs") is None: meta["crs"] = std_4326
            except Exception as e: _log(f"      [Warning] Clip failed: {e}"); return mosaic_path
        result = dirs.intermediate / f"{out_name}_wgs84.tif"
        with rasterio.open(result,"w",**meta) as d: d.write(cl)
        _log(f"      After clip: {cl.shape[2]}x{cl.shape[1]}")
        for p in dirs.intermediate.glob(f"_tmp_*_{out_name}*"):
            if p != result: p.unlink(missing_ok=True)
        return result

    def to_utm(src_path, out_name, resamp):
        final = dirs.rasters / f"{out_name}.tif"
        with rasterio.open(src_path) as src:
            src_crs_val = src.crs or std_4326_rio  # default WGS84 if no CRS
            tf,w,h = calculate_default_transform(src_crs_val, dst_crs_rio, src.width, src.height, *src.bounds)
            kw = src.meta.copy(); kw.update(crs=dst_crs_rio, transform=tf, width=w, height=h, compress="lzw", driver="GTiff")
            with rasterio.open(final,"w",**kw) as dst:
                for i in range(1, src.count+1):
                    reproject(source=rasterio.band(src,i), destination=rasterio.band(dst,i), src_transform=src.transform, src_crs=src_crs_val, dst_transform=tf, dst_crs=dst_crs_rio, resampling=resamp)
        with rasterio.open(final) as c: _log(f"    [OK] -> {final.name} ({c.width}x{c.height}, EPSG:{target_epsg})")
        return final

    raw = dirs.raw_downloads
    # DEM
    if dem_wgs_path and Path(dem_wgs_path).exists():
        print(f"\n  DEM:"); to_utm(dem_wgs_path, "dem", Resampling.bilinear)
    else:
        fallback = dirs.intermediate / "dem_wgs84.tif"
        if fallback.exists(): print(f"\n  DEM:"); to_utm(fallback, "dem", Resampling.bilinear)
        else: print("  [Error] dem_wgs84.tif not found!")
    # Landuse
    print(f"\n  LANDUSE:")
    lu = []
    for cont in continents:
        lu += find_covering(raw / f"{cont}_landuse")
    if lu:
        lc = merge_and_clip(lu, "landuse")
        if lc: to_utm(lc, "landuse", Resampling.nearest)
    else: print("    [Error] No land use tiles covering study area!")
    
    # Soil
    print(f"\n  SOIL:")
    sl = []
    for cont in continents:
        sl += find_covering(raw / f"{cont}_soil")
    if sl:
        sc = merge_and_clip(sl, "soil")
        if sc: to_utm(sc, "soil", Resampling.nearest)
    else: print("    [Error] No soil tiles covering study area!")


def step_fill_nodata_gaps(dirs):
    """
    Ensure that wherever DEM has valid data, soil and landuse also have values.
    Fill NoData holes in soil.tif / landuse.tif using nearest-neighbour imputation.

    Rationale:
      DEM extent = SWAT+ modelling extent, but the official SWAT soil/landuse
      rasters may have NoData holes near coasts/islands/borders.  These gaps
      cause QSWAT+ to abort.  scipy.ndimage.distance_transform_edt with
      return_indices finds the nearest valid pixel for each NoData cell.
    """
    print(f"\n{'='*50}\n[Step] Fill soil/landuse NoData gaps (align with DEM extent)")

    from scipy.ndimage import distance_transform_edt

    dem_path = dirs.rasters / "dem.tif"
    if not dem_path.exists():
        print("  [Warning] dem.tif not found, skipping"); return

    # Read DEM valid-pixel mask
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_nd = dem_src.nodata
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        dem_h, dem_w = dem_data.shape

    if dem_nd is not None:
        dem_valid = (dem_data != dem_nd) & ~np.isnan(dem_data.astype(float))
    else:
        dem_valid = ~np.isnan(dem_data.astype(float))

    dem_valid_count = dem_valid.sum()
    _log(f"  DEM valid pixels: {dem_valid_count:,} / {dem_h * dem_w:,}")

    # Fill soil and landuse
    for rname in ["soil.tif", "landuse.tif"]:
        rpath = dirs.rasters / rname
        if not rpath.exists():
            _log(f"  [Warning] {rname} not found, skipping")
            continue

        with rasterio.open(rpath) as src:
            data = src.read(1)
            nd = src.nodata
            meta = src.meta.copy()
            src_h, src_w = data.shape
            src_transform = src.transform
            src_crs = src.crs

        # Identify NoData pixels
        if nd is not None:
            is_nodata = (data == nd)
            if np.issubdtype(data.dtype, np.floating):
                is_nodata = is_nodata | np.isnan(data)
        else:
            if np.issubdtype(data.dtype, np.floating):
                is_nodata = np.isnan(data)
            else:
                is_nodata = (data == 0)

        total_nodata = is_nodata.sum()
        total_valid = (~is_nodata).sum()
        print(f"\n  {rname}: {src_w}x{src_h}, valid={total_valid:,}, nodata={total_nodata:,}")

        if total_nodata == 0:
            _log(f"    [Skip] No holes found"); continue
        if total_valid == 0:
            _log(f"    [Error] Entirely NoData, cannot fill!"); continue

        # Align DEM mask to soil/landuse grid
        if (src_h == dem_h and src_w == dem_w and
            str(src_crs) == str(dem_crs) and
            src_transform.almost_equals(dem_transform, precision=1e-6)):
            dem_mask_aligned = dem_valid
        else:
            from rasterio.warp import reproject, Resampling as _Resamp
            dem_mask_float = dem_valid.astype(np.float32)
            aligned = np.zeros((src_h, src_w), dtype=np.float32)
            reproject(
                source=dem_mask_float, destination=aligned,
                src_transform=dem_transform, src_crs=dem_crs,
                dst_transform=src_transform, dst_crs=src_crs,
                resampling=_Resamp.nearest,
            )
            dem_mask_aligned = aligned > 0.5

        # Count holes within DEM extent
        gaps_in_dem = is_nodata & dem_mask_aligned
        n_gaps = gaps_in_dem.sum()
        print(f"    Gaps within DEM extent: {n_gaps:,} pixels ({n_gaps / max(dem_mask_aligned.sum(), 1) * 100:.2f}%)")

        if n_gaps == 0:
            print(f"    [Skip] No gaps within DEM extent"); continue

        # Nearest-neighbour fill
        _, nearest_indices = distance_transform_edt(
            is_nodata, return_distances=True, return_indices=True)

        filled = data.copy()
        filled[is_nodata] = data[nearest_indices[0][is_nodata],
                                  nearest_indices[1][is_nodata]]

        # Apply fill only within DEM extent
        result = data.copy()
        result[gaps_in_dem] = filled[gaps_in_dem]

        print(f"    [OK] Filled {n_gaps:,} NoData pixels")

        # Write back
        tmp_path = rpath.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(result, 1)
        tmp_path.replace(rpath)
        _log(f"    [Save] Written: {rpath.name}")

    print(f"\n  [OK] soil/landuse NoData fill complete!")


def step_clip_lakes(basins, hydrolakes_shp, min_lake_area_km2=1.0, dirs=None):
    print(f"\n{'='*50}\n[Step] Clip lakes (HydroLAKES)")
    if not hydrolakes_shp or not Path(hydrolakes_shp).exists():
        _log(f"  [Warning] HydroLAKES file not found: {hydrolakes_shp}")
        return None

    bounds = basins.to_crs("EPSG:4326").total_bounds
    buf = 0.1
    bbox = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
    
    _log(f"  [Load] {hydrolakes_shp}")
    lakes_all = gpd.read_file(hydrolakes_shp, bbox=bbox)
    lakes_all = lakes_all.to_crs("EPSG:4326")
    _log(f"  Lakes in bbox: {len(lakes_all)}")

    if len(lakes_all) == 0:
        return None

    # --- Optimization 1: apply attribute filter first (pre-filter) ---
    # Drop small lakes and reservoirs before the expensive spatial operations

    # Area filter
    area_col = next((c for c in ["Lake_area", "LAKE_AREA", "lake_area", "Area_km2", "AREA"] 
                     if c in lakes_all.columns), None)
    
    if area_col and min_lake_area_km2 > 0:
        n_before = len(lakes_all)
        lakes_all = lakes_all[lakes_all[area_col] >= min_lake_area_km2].copy()
        _log(f"  [Optimization] Pre-filter area >= {min_lake_area_km2}: {n_before} -> {len(lakes_all)}")

    # Type filter (Remove reservoirs type=2)
    type_col = next((c for c in ["Lake_type", "LAKE_TYPE", "lake_type"] 
                     if c in lakes_all.columns), None)
    if type_col:
        n_before = len(lakes_all)
        lakes_all = lakes_all[lakes_all[type_col] != 2].copy()
        _log(f"  [Optimization] Pre-filter remove reservoirs: {n_before} -> {len(lakes_all)}")

    if len(lakes_all) == 0:
        _log(f"  [Warning] No lakes remain after pre-filtering")
        return None

    # --- Optimization 2: geometry simplification (spatial join with simplified basins) ---
    _log(f"  [Optimization] Spatial join with simplified basins...")
    basins_simple = basins.to_crs("EPSG:4326").copy()
    basins_simple.geometry = basins_simple.geometry.simplify(0.005, preserve_topology=True)
    
    joined = gpd.sjoin(lakes_all, basins_simple, how="inner", predicate="intersects")
    lakes = lakes_all.loc[joined.index.unique()].copy()
    _log(f"  Lakes within basins: {len(lakes)}")

    # Print size breakdown (Reporting)
    if area_col:
        areas = lakes[area_col]
        n_small = (areas < 10).sum()
        n_medium = ((areas >= 10) & (areas < 100)).sum()
        n_large = (areas >= 100).sum()
        _log(f"  Lake size breakdown: <10 km²={n_small}, 10-100 km²={n_medium}, >=100 km²={n_large}")

    # Save figure
    if dirs and len(lakes) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        basins.to_crs("EPSG:4326").plot(ax=ax, edgecolor="gray", facecolor="lightyellow", alpha=0.4, linewidth=0.3)
        lakes.plot(ax=ax, facecolor="cyan", edgecolor="darkblue", alpha=0.6, linewidth=0.5)
        ax.set_title(f"Lakes: {len(lakes)} (>= {min_lake_area_km2} km²)", fontsize=14)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        save_figure(fig, dirs.figures / "02c_lakes.png")

    return lakes


def step_clip_pour_points(basins, pour_points_shp, rivers=None,
                          max_river_dist_km=2.0, cluster_dist_km=2.0, dirs=None):
    print(f"\n{'='*50}\n[Step] Clip pour points (HydroBasins)")
    if not Path(pour_points_shp).exists():
        _log(f"  [Warning] Pour-points file not found: {pour_points_shp}"); return None

    bounds = basins.to_crs("EPSG:4326").total_bounds; buf = 0.2
    read_bbox = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
    _log(f"  [Load] {pour_points_shp} (bbox pre-clip)")
    pour_all = gpd.read_file(pour_points_shp, bbox=read_bbox)
    _log(f"  Pour points in region: {len(pour_all)}")

    if len(pour_all) == 0:
        _log(f"  [Warning] No pour points found in region"); return None

    # Method 1: match by HYBAS_ID (Fastest)
    if "HYBAS_ID" in basins.columns and "HYBAS_ID" in pour_all.columns:
        basin_ids = set(basins["HYBAS_ID"].values)
        pour = pour_all[pour_all["HYBAS_ID"].isin(basin_ids)].copy()
        _log(f"  Matched by HYBAS_ID: {len(pour)} pour points")
    else:
        # Method 2: spatial join (Fallback - Optimized)
        _log(f"  [Optimization] ID match failed, using Spatial Join with simplified basins...")
        pour_all = pour_all.to_crs("EPSG:4326")
        
        # Optimization: use simplified geometries for the containment test
        basins_simple = basins.to_crs("EPSG:4326").copy()
        basins_simple.geometry = basins_simple.geometry.simplify(0.005, preserve_topology=True)
        
        joined = gpd.sjoin(pour_all, basins_simple, how="inner", predicate="within")
        pour = pour_all.loc[joined.index.unique()].copy()
        _log(f"  Spatial join: {len(pour)} pour points")

    # ---- River distance filter ----
    centroid = basins.to_crs("EPSG:4326").unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    utm_epsg = (32600 + utm_zone) if centroid.y >= 0 else (32700 + utm_zone)

    if rivers is not None and len(rivers) > 0:
        from shapely import STRtree
        rivers_utm = rivers.to_crs(epsg=utm_epsg)
        river_geoms = rivers_utm.geometry.values
        # Building the STRtree is fast; this is already a good approach
        tree = STRtree(river_geoms)

        pour_utm = pour.to_crs(epsg=utm_epsg)
        dists = []
        # This loop would be slow with many points, but pour points are usually limited, so no extra optimization needed for now
        for pt in pour_utm.geometry:
            idx = tree.nearest(pt)
            dists.append(pt.distance(river_geoms[idx]))
        pour["_river_dist_m"] = dists

        if max_river_dist_km and max_river_dist_km > 0:
            _log(f"  [Filter] Removing pour points > {max_river_dist_km} km from river...")
            n_before = len(pour)
            max_dist_m = max_river_dist_km * 1000
            near_mask = pour["_river_dist_m"] <= max_dist_m
            n_removed = (~near_mask).sum()
            pour = pour.loc[near_mask].copy()
            _log(f"    River distance filter: {n_before} -> {len(pour)} (removed {n_removed})")

    # ---- Cluster deduplication ----
    if len(pour) > 1 and cluster_dist_km and cluster_dist_km > 0:
        _log(f"  [Cluster] Dedup within {cluster_dist_km} km, keep nearest to river...")
        n_before = len(pour)
        cluster_dist_m = cluster_dist_km * 1000

        pour_utm = pour.to_crs(epsg=utm_epsg).copy()
        coords = np.array([(g.x, g.y) for g in pour_utm.geometry])

        if "_river_dist_m" in pour.columns:
            order = pour["_river_dist_m"].values.argsort()
        else:
            order = np.arange(len(pour))

        keep_mask = np.ones(len(pour), dtype=bool)
        kept_coords = []

        for idx in order:
            if not keep_mask[idx]:
                continue
            pt = coords[idx]
            too_close = False
            for kc in kept_coords:
                dist = np.sqrt((pt[0] - kc[0])**2 + (pt[1] - kc[1])**2)
                if dist < cluster_dist_m:
                    too_close = True
                    break
            if too_close:
                keep_mask[idx] = False
            else:
                kept_coords.append(pt)

        pour = pour.iloc[keep_mask].copy()
        n_merged = n_before - len(pour)
        if n_merged > 0:
            _log(f"    Cluster dedup: {n_before} -> {len(pour)} (merged {n_merged} nearby points)")
        else:
            _log(f"    No nearby duplicate points")

    if "_river_dist_m" in pour.columns:
        pour = pour.drop(columns=["_river_dist_m"])

    if "NEXT_DOWN" in pour.columns:
        basin_ids = set(basins["HYBAS_ID"].values) if "HYBAS_ID" in basins.columns else set()
        pour["is_outlet"] = pour["NEXT_DOWN"].apply(lambda x: x == 0 or x not in basin_ids)
        _log(f"  Watershed outlets: {pour['is_outlet'].sum()}")

    pour_qswat = pour.copy()
    pour_qswat["ID"] = range(len(pour_qswat))
    pour_qswat["INLET"] = 0
    pour_qswat["RES"] = 0
    pour_qswat["PTSOURCE"] = 0
    pour["_qswat"] = pour_qswat[["ID", "INLET", "RES", "PTSOURCE"]].values.tolist()

    if dirs and len(pour) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        basins.to_crs("EPSG:4326").plot(ax=ax, edgecolor="gray", facecolor="lightyellow", alpha=0.4, linewidth=0.3)
        if rivers is not None and len(rivers) > 0:
            rivers.to_crs("EPSG:4326").plot(ax=ax, color="steelblue", linewidth=0.3, alpha=0.4)
        pour.to_crs("EPSG:4326").plot(ax=ax, color="red", markersize=15)
        ax.set_title(f"Pour points: {len(pour)}", fontsize=14)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        save_figure(fig, dirs.figures / "03_pour_points.png")

    return pour, pour_qswat


def step_save_vectors(basins, rivers, study_area, dirs, target_epsg,
                      outlet_lon=None, outlet_lat=None,
                      pour_points=None, pour_qswat=None, lakes=None):
    print(f"\n{'='*50}\n[Step] Save vector data")
    dst_crs = CRS.from_epsg(target_epsg); vec = dirs.vectors
    # basins.to_crs("EPSG:4326").to_file(vec / "basins_wgs84.shp")
    # basins.to_crs(dst_crs).to_file(vec / "basins.shp")
    # _log(f"  [OK] basins.shp ({len(basins)} sub-basins)")
    if rivers is not None and len(rivers) > 0:
        rivers.to_crs("EPSG:4326").to_file(vec / "rivers_wgs84.shp")
        rivers.to_crs(dst_crs).to_file(vec / "rivers.shp")
        _log(f"  [OK] rivers.shp ({len(rivers)} river segments)")
    study_area.to_file(vec / "study_area.shp"); _log(f"  [OK] study_area.shp")

    # Lakes -> for QSWAT+ Add Lakes
    # QSWAT+ requires RES field: 1=reservoir, 2=pond
    # HydroLAKES Lake_type: 1=lake, 2=reservoir, 3=regulated lake
    # Mapping: reservoir/regulated -> RES=1, natural lake -> RES=2 (pond)
    if lakes is not None and len(lakes) > 0:
        lakes_out = lakes.copy()
        if "Lake_type" in lakes_out.columns:
            lakes_out["RES"] = lakes_out["Lake_type"].apply(
                lambda t: 1 if t in [2, 3] else 2  # reservoir/regulated->1, lake->2
            )
        else:
            lakes_out["RES"] = 1  # default to reservoir
        lakes_out.to_crs("EPSG:4326").to_file(vec / "lakes_wgs84.shp")
        lakes_out.to_crs(dst_crs).to_file(vec / "lakes.shp")
        n_res = (lakes_out["RES"] == 1).sum()
        n_pond = (lakes_out["RES"] == 2).sum()
        _log(f"  [OK] lakes.shp ({len(lakes_out)} lakes: {n_res} reservoir + {n_pond} pond)")

    # Pour points (all, with original HydroBasins attributes)
    if pour_points is not None and len(pour_points) > 0:
        # Save original version (all HydroBasins attributes)
        cols_save = [c for c in pour_points.columns if c != "_qswat"]
        pour_points[cols_save].to_crs("EPSG:4326").to_file(vec / "pour_points_wgs84.shp")
        pour_points[cols_save].to_crs(dst_crs).to_file(vec / "pour_points.shp")
        _log(f"  [OK] pour_points.shp ({len(pour_points)} pour points, with original attributes)")

    # QSWAT+-compatible format (ID, INLET, RES, PTSOURCE, geometry)
    if pour_qswat is not None and len(pour_qswat) > 0:
        # All pour points -> QSWAT+ format
        qswat_cols = ["ID", "INLET", "RES", "PTSOURCE", "geometry"]
        gdf_all = pour_qswat[qswat_cols].copy()
        gdf_all.to_crs(dst_crs).to_file(vec / "outlets_qswat.shp")
        gdf_all.to_crs("EPSG:4326").to_file(vec / "outlets_qswat_wgs84.shp")
        _log(f"  [OK] outlets_qswat.shp ({len(gdf_all)} points, QSWAT+ compatible)")

        # Watershed outlets only -> QSWAT+ format
        if "is_outlet" in pour_points.columns:
            outlet_mask = pour_points["is_outlet"].values
            gdf_main = pour_qswat.loc[outlet_mask, qswat_cols].copy()
            gdf_main["ID"] = range(len(gdf_main))  # re-number IDs
            if len(gdf_main) > 0:
                gdf_main.to_crs(dst_crs).to_file(vec / "outlets_main_qswat.shp")
                gdf_main.to_crs("EPSG:4326").to_file(vec / "outlets_main_qswat_wgs84.shp")
                _log(f"  [OK] outlets_main_qswat.shp ({len(gdf_main)} watershed outlets, QSWAT+ compatible)")

    # Manually specified outlet -> QSWAT+ format
    if outlet_lon is not None and outlet_lat is not None:
        manual = gpd.GeoDataFrame(
            {"ID": [0], "INLET": [0], "RES": [0], "PTSOURCE": [0],
             "geometry": [Point(outlet_lon, outlet_lat)]},
            crs="EPSG:4326"
        )
        manual.to_crs(dst_crs).to_file(vec / "outlet_manual.shp")
        _log(f"  [OK] outlet_manual.shp (manually specified, QSWAT+ compatible)")


def step_save_raster_figures(basins, rivers, dirs, target_epsg):
    """Save DEM / land-use / soil visualization figures."""
    print(f"\n{'='*50}\n[Step] Save raster visualization figures")
    info = [("dem.tif","DEM Elevation (m)","terrain"), ("landuse.tif","Land Use (USGS GLCC)","Set3"), ("soil.tif","Soil (FAO)","tab20")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (fn, title, cmap) in zip(axes, info):
        fp = dirs.rasters / fn
        if fp.exists():
            with rasterio.open(fp) as src:
                data = src.read(1, masked=True)
                im = ax.imshow(data, cmap=cmap, extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
                plt.colorbar(im, ax=ax, shrink=0.7)
                ax.set_title(f"{title}\n{src.width}x{src.height}, EPSG:{src.crs.to_epsg()}")
                _log(f"  [OK] {fn}: {src.width}x{src.height}, value range=[{float(data.min()):.0f}, {float(data.max()):.0f}]")
        else:
            ax.set_title(f"{title}\nNot found"); ax.text(0.5, 0.5, "File not found", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"SWAT+ Input Rasters (EPSG:{target_epsg})", fontsize=16, y=1.02)
    fig.tight_layout(); save_figure(fig, dirs.figures / "04_rasters_overview.png")

    # DEM + basins + rivers overlay
    dem_fp = dirs.rasters / "dem.tif"
    if dem_fp.exists():
        fig, ax = plt.subplots(figsize=(12, 10))
        with rasterio.open(dem_fp) as src:
            data = src.read(1, masked=True)
            im = ax.imshow(data, cmap="terrain", extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        dst_crs = CRS.from_epsg(target_epsg)
        basins.to_crs(dst_crs).boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, alpha=0.7)
        if rivers is not None and len(rivers) > 0:
            rivers.to_crs(dst_crs).plot(ax=ax, color="blue", linewidth=0.3, alpha=0.5)
        plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)
        ax.set_title("DEM + Basins + Rivers", fontsize=14)
        fig.tight_layout(); save_figure(fig, dirs.figures / "05_dem_overlay.png")


def step_verify(dirs, target_epsg):
    print(f"\n{'='*60}\n  Output Verification\n{'='*60}\n  Target projection: EPSG:{target_epsg}\n")
    all_ok = True
    def _get_epsg(crs_obj):
        """Extract EPSG from a rasterio or geopandas CRS object."""
        if crs_obj is None: return None
        try: return crs_obj.to_epsg()
        except Exception: pass
        try: return CRS.from_wkt(crs_obj.to_wkt()).to_epsg()
        except Exception: pass
        return None
    for name in ["dem.tif","landuse.tif","soil.tif"]:
        path = dirs.rasters / name
        if not path.exists(): print(f"  [Error] rasters/{name}: not found!"); all_ok = False; continue
        with rasterio.open(path) as src:
            epsg = _get_epsg(src.crs); ok = epsg == target_epsg
            _log(f"  {'[OK]' if ok else '[ERR]'} rasters/{name:15s} | {src.width:5d}x{src.height:<5d} | EPSG:{epsg} | {path.stat().st_size/1048576:.1f} MB")
            if not ok: all_ok = False
    for name in ["basins.shp","rivers.shp","lakes.shp","pour_points.shp","outlets_qswat.shp","outlets_main_qswat.shp"]:
        path = dirs.vectors / name
        if not path.exists(): _log(f"  [Skip] vectors/{name}: not found (optional)"); continue
        gdf = gpd.read_file(path); epsg = _get_epsg(gdf.crs); ok = epsg == target_epsg
        _log(f"  {'[OK]' if ok else '[ERR]'} vectors/{name:15s} | {len(gdf):5d} features   | EPSG:{epsg}")
        if not ok: all_ok = False
    figs = sorted(dirs.figures.glob("*.png"))
    if figs:
        print(f"\n  Figures ({len(figs)}):")
        for f in figs: _log(f"     figures/{f.name}  ({f.stat().st_size/1024:.0f} KB)")
    print(f"\n  {'[OK] All checks passed!' if all_ok else '[Warning] Issues found -- check [Error] lines'}")
    return all_ok


# =============================================================================
# Main function
# =============================================================================
def prepare_swat_data(
    country="Cambodia", outlet_lon=None, outlet_lat=None, bbox=None,
    study_area_shp=None,
    hydrobasins_shp=None,
    hydrorivers_shp=None,
    hydrolakes_shp="/home/cfeng/hydro/source_data/atlas/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp",
    hydrobasins_dir="/home/cfeng/hydro/source_data/atlas",
    grfr_dir="/home/cfeng/hydro/source_data/grfr",
    pour_points_shp=None,
    river_detail=3, basin_level=12, dem_resolution=None,
    continent="ea", output_dir="./swat_output", target_epsg=32648,
    srtm_dir="/home/cfeng/hydro/dem/srtm90m/",
    use_original_landuse=False, min_basin_area_km2=50,
    max_river_dist_km=2.0, cluster_dist_km=2.0,
    min_lake_area_km2=1.0,
):
    """
    One-click SWAT+ input data preparation.

    Parameters
    ----------
    country : str              Country name (English)
    study_area_shp : str       Custom study-area shapefile path (overrides country/bbox)
    river_detail : int 1-5     River detail level (1=main rivers only, 5=all)
    basin_level : int 1-12     HydroBasins level
    dem_resolution : int       Target DEM resolution (m), None=original. Recommend: 90/250/500
    continent : str            SWAT continent code: ea/af/ap/na/sa
    target_epsg : int          Target projection EPSG (UTM)
    max_river_dist_km : float  Max pour-point to river distance (km); farther removed. 0/None=no filter
    cluster_dist_km : float    Cluster dedup distance (km); keep nearest to river per cluster. 0/None=no dedup
    """
    assert river_detail in range(1,6), f"river_detail must be 1-5"
    assert 1 <= basin_level <= 12, f"basin_level must be 1-12"
    _conts = continent if isinstance(continent, list) else [continent]
    for _c in _conts:
        assert _c in SWAT_URLS["soil"], f"Unknown continent: {_c}"

    dirs = OutputDirs(output_dir).create_all()

    # Auto-locate GRFR basin shapefiles (cat_pfaf_*) across all pfaf subdirs
    if hydrobasins_shp is None:
        grfr_path = Path(grfr_dir)
        candidates_found = []
        if grfr_path.exists():
            for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
                for cand in sorted(pfaf_dir.glob("cat_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
                    candidates_found.append(str(cand))
        if candidates_found:
            hydrobasins_shp = candidates_found  # list spanning all pfaf tiles
            _log(f"  [GRFR] Found {len(candidates_found)} basin shapefiles in {grfr_dir}")
        else:
            raise FileNotFoundError(
                f"GRFR basin shapefiles (cat_pfaf_*) not found in: {grfr_dir}\n"
                f"Expected subdirs: pfaf_N_MERIT_Hydro_v07_Basins_v01__extract/")

    # Auto-locate GRFR river shapefiles (riv_pfaf_*) - used as hydrorivers_shp
    if hydrorivers_shp is None:
        grfr_path = Path(grfr_dir)
        riv_candidates = []
        if grfr_path.exists():
            for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
                for cand in sorted(pfaf_dir.glob("riv_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
                    riv_candidates.append(str(cand))
        if riv_candidates:
            hydrorivers_shp = riv_candidates  # list spanning all pfaf tiles
            _log(f"  [GRFR] Found {len(riv_candidates)} river shapefiles in {grfr_dir}")
        else:
            print(f"  [Warning] GRFR river shapefiles (riv_pfaf_*) not found in: {grfr_dir}")

    # GRFR does not use pour-point shapefiles; pour_points_shp stays None (step already skipped)

    print("="*60)
    print("SWAT+ Input Data Preparation Tool v4")
    print("="*60)
    _log(f"  Country/region:  {country or 'Custom'}")
    _log(f"  River detail:    {river_detail} (Strahler order >= {RIVER_DETAIL_MAP[river_detail]})")
    _log(f"  Basin level:     HydroBasins Level {basin_level}")
    _log(f"  DEM resample:    {f'{dem_resolution}m' if dem_resolution else 'keep original'}")
    _log(f"  Continent:       {continent}")
    _log(f"  Target EPSG:     {target_epsg}")
    _log(f"  Output dir:      {output_dir}/")
    _log(f"  Pour points:     {pour_points_shp or 'not found'}")
    _log(f"    ├── rasters/       <- QSWAT+ rasters")
    _log(f"    ├── vectors/       <- QSWAT+ vectors")
    _log(f"    ├── figures/       <- Visualization PNGs")
    _log(f"    ├── intermediate/  <- Intermediate files")
    _log(f"    └── raw_downloads/ <- Raw downloads")
    print("="*60)

    # Step 1: Study area
    if study_area_shp and Path(study_area_shp).exists():
        print(f"[Study area] Using custom shapefile: {study_area_shp}")
        study_area = gpd.read_file(study_area_shp)
        if study_area.crs is None: study_area = study_area.set_crs("EPSG:4326")
        else: study_area = study_area.to_crs("EPSG:4326")
    else:
        study_area = get_study_area(country=country, outlet_lon=outlet_lon, outlet_lat=outlet_lat, bbox=bbox, hydrobasins_shp=hydrobasins_shp)
    fig, ax = plt.subplots(figsize=(8, 6))
    study_area.plot(ax=ax, edgecolor="red", facecolor="lightyellow", linewidth=2)
    ax.set_title(f"Study area: {country or 'Custom'}", fontsize=14); ax.grid(True, alpha=0.3); fig.tight_layout()
    save_figure(fig, dirs.figures / "00_study_area.png")

    # Steps 2-3: Basins & rivers
    basins = step_clip_basins(study_area, hydrobasins_shp, min_area_km2=min_basin_area_km2, dirs=dirs)
    rivers = step_clip_rivers(basins, hydrorivers_shp, river_detail=river_detail, hydrolakes_shp=hydrolakes_shp, dirs=dirs)

    # Step 2b: Lakes
    lakes = step_clip_lakes(basins, hydrolakes_shp, min_lake_area_km2=min_lake_area_km2, dirs=dirs)

    # Step: Clip pour points
    pour_points = None
    pour_qswat = None
    # if pour_points_shp:
    #     result = step_clip_pour_points(basins, pour_points_shp, rivers=rivers,
    #                                    max_river_dist_km=max_river_dist_km,
    #                                    cluster_dist_km=cluster_dist_km, dirs=dirs)
    #     if result is not None:
    #         pour_points, pour_qswat = result
    print("  [Step] Clip pour points: SKIPPED (User request)")
    # Step 4: Download
    step_download_swat_data(continent, dirs, use_original_landuse=use_original_landuse)

    # Step 5: DEM (with optional resampling)
    dem_wgs_path = step_prepare_dem(basins, srtm_dir, dirs, dem_resolution=dem_resolution)

    # Step 6: Reproject
    step_reproject_rasters(basins, dirs, continent, target_epsg, dem_wgs_path=dem_wgs_path)

    # Step 6b: Fill soil/landuse NoData gaps (align with DEM extent)
    step_fill_nodata_gaps(dirs)

    # Step 7: Vectors
    step_save_vectors(basins, rivers, study_area, dirs, target_epsg,
                      outlet_lon=outlet_lon, outlet_lat=outlet_lat,
                      pour_points=pour_points, pour_qswat=pour_qswat, lakes=lakes)

    # Step 8: Figures
    step_save_raster_figures(basins, rivers, dirs, target_epsg)

    # Step 9: Verify
    all_ok = step_verify(dirs, target_epsg)

    print(f"\n{'='*60}\nDone! Use in QSWAT+:\n{'='*60}")
    print(f"""
  Step 1 - Delineate Watershed:
    DEM:      -> {dirs.rasters}/dem.tif
    Burn in:  [x] -> {dirs.vectors}/rivers.shp

  Step 1 - Add Lakes (optional):
    Lakes:    -> {dirs.vectors}/lakes.shp

  Step 2 - Create HRUs:
    Landuse:  -> {dirs.rasters}/landuse.tif   (table: global_landuses)
    Soil:     -> {dirs.rasters}/soil.tif      (table: global_soils / global_usersoil)

  Outlet files (QSWAT+ compatible, fields: ID/INLET/RES/PTSOURCE):
    All outlets:       -> {dirs.vectors}/outlets_qswat.shp
    Watershed outlets: -> {dirs.vectors}/outlets_main_qswat.shp
    Original attrs:    -> {dirs.vectors}/pour_points.shp

  Figures: -> {dirs.figures}/
""")
    return {"study_area": study_area, "basins": basins, "rivers": rivers,
            "lakes": lakes, "pour_points": pour_points, "dirs": dirs, "all_ok": all_ok}


# =============================================================================
# Phase 1.5: Generate QSWAT+ "Use Existing Watershed" input files
# =============================================================================
# Based on Phase 1 (prepare_swat_data) outputs, generates QSWAT+ Step 1 files.
#
# Workflow:
#   step1 = prepare_swat_data(country="Cambodia", ...)
#   step2 = generate_qswat_watershed(step1, outlet=(106.96783, 11.11035))
#   step2 = generate_qswat_watershed(step1, outlet="all")
# =============================================================================

def _guess_utm_epsg(lon, lat):
    """Infer UTM EPSG from lon/lat."""
    zone = int((lon + 180) / 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _get_id_col(basins_df):
    """Return the basin ID column name (COMID for GRFR, HYBAS_ID for HydroBasins)."""
    for c in ["COMID", "HYBAS_ID", "comid", "BasinID"]:
        if c in basins_df.columns:
            return c
    raise KeyError(f"Cannot find basin ID column in: {list(basins_df.columns)}")

def _get_nextdown_col(basins_df):
    """Return the downstream-link column name."""
    for c in ["NextDownID", "NEXT_DOWN", "nextdownid", "DSLINKNO"]:
        if c in basins_df.columns:
            return c
    raise KeyError(f"Cannot find next-downstream column in: {list(basins_df.columns)}")

def _get_upstream_basins(basins_df, outlet_hybas_id):
    """Collect all upstream basin IDs from an outlet (iterative, uses stack).
    Works with both GRFR (COMID/NextDownID) and HydroBasins (HYBAS_ID/NEXT_DOWN)."""
    id_col = _get_id_col(basins_df)
    nd_col = _get_nextdown_col(basins_df)
    id_set = {outlet_hybas_id}
    children_map = {}
    for hid, nd in zip(basins_df[id_col], basins_df[nd_col]):
        children_map.setdefault(nd, []).append(hid)
    stack = [outlet_hybas_id]
    while stack:
        cur = stack.pop()
        for child in children_map.get(cur, []):
            if child not in id_set:
                id_set.add(child)
                stack.append(child)
    return id_set


def _find_country_outlets(basins):
    """
    Find all outlet basins in the study area (water flows out of boundary).
    Works with both GRFR (COMID/NextDownID) and HydroBasins (HYBAS_ID/NEXT_DOWN).
    """
    id_col = _get_id_col(basins)
    nd_col = _get_nextdown_col(basins)
    basin_ids = set(basins[id_col].values)
    mask = basins[nd_col].apply(lambda nd: nd == 0 or nd not in basin_ids)
    return basins[mask].copy()


def _build_qswat_channels(ws_basins, rivers, hybas_to_poly, target_epsg):
    """
    Build the QSWAT+ channel shapefile for a single watershed.

    QSWAT+ requires the full TauDEM StreamNet output fields:
      LINKNO, DSLINKNO, USLINKNO1, USLINKNO2, DSNODEID,
      Order, Length, Magnitude, DS_Cont_Ar, Drop, Slope,
      Straight_L, US_Cont_Ar, WSNO, DOUT_END, DOUT_START, DOUT_MID,
      BasinNo
    """
    from shapely.geometry import LineString

    id_col = _get_id_col(ws_basins)
    nd_col = _get_nextdown_col(ws_basins)
    hybas_next = dict(zip(ws_basins[id_col], ws_basins[nd_col]))
    poly_to_hybas = {v: k for k, v in hybas_to_poly.items()}

    def _ds_linkno(poly_id):
        hid = poly_to_hybas.get(poly_id)
        nd = hybas_next.get(hid, 0)
        if nd == 0 or nd not in hybas_to_poly:
            return -1
        return hybas_to_poly[nd]

    # Build upstream map: ds_link -> [us_link1, us_link2, ...]
    all_pids = sorted(ws_basins["PolygonId"].values)
    ds_map = {pid: _ds_linkno(pid) for pid in all_pids}
    upstream_map = {}  # ds_pid -> list of upstream pids
    for pid, ds in ds_map.items():
        if ds != -1:
            upstream_map.setdefault(ds, []).append(pid)

    rows = []
    covered = set()

    # Extract real river channels from clipped rivers
    if rivers is not None and len(rivers) > 0:
        basins_4326 = ws_basins.to_crs("EPSG:4326")
        rivers_4326 = rivers.to_crs("EPSG:4326")
        joined = gpd.sjoin(rivers_4326, basins_4326[[id_col, "PolygonId", "geometry"]],
                           how="inner", predicate="intersects")
        if len(joined) > 0:
            dissolved = joined.dissolve(by="PolygonId").reset_index()
            for _, row in dissolved.iterrows():
                pid = int(row["PolygonId"])
                rows.append({"PolygonId": pid, "geometry": row.geometry})
                covered.add(pid)

    # Fill missing sub-basins with synthetic stub channels
    missing = set(all_pids) - covered
    for pid in missing:
        basin_row = ws_basins[ws_basins["PolygonId"] == pid].iloc[0]
        centroid = basin_row.geometry.centroid
        dx = 0.001
        line = LineString([(centroid.x - dx, centroid.y),
                           (centroid.x + dx, centroid.y)])
        rows.append({"PolygonId": pid, "geometry": line})

    if len(missing) > 0:
        _log(f"    Channels: {len(covered)} real + {len(missing)} synthetic")

    # Reproject to target CRS
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(target_epsg)

    # Build complete attribute table
    full_rows = []
    for _, row in gdf.iterrows():
        pid = int(row["PolygonId"])
        geom = row.geometry
        ds = ds_map[pid]
        us_list = upstream_map.get(pid, [])
        us1 = us_list[0] if len(us_list) >= 1 else -1
        us2 = us_list[1] if len(us_list) >= 2 else -1

        # Compute Length (projected, metres)
        length = geom.length if geom.length > 0 else 1.0
        # Straight_L
        coords = list(geom.coords) if hasattr(geom, 'coords') else []
        if len(coords) >= 2:
            from math import sqrt
            straight_l = sqrt((coords[-1][0]-coords[0][0])**2 + (coords[-1][1]-coords[0][1])**2)
        else:
            straight_l = length

        full_rows.append({
            "LINKNO":     pid,
            "DSLINKNO":   ds,
            "USLINKNO1":  us1,
            "USLINKNO2":  us2,
            "DSNODEID":   -1,
            "Order":      1,           # Strahler order placeholder
            "Length":      round(length, 2),
            "Magnitude":  1,
            "DS_Cont_Ar": 0.0,
            "Drop":       0.0,
            "Slope":      0.001,       # placeholder
            "Straight_L": round(straight_l, 2),
            "US_Cont_Ar": 0.0,
            "WSNO":       pid,
            "DOUT_END":   0.0,
            "DOUT_START": 0.0,
            "DOUT_MID":   0.0,
            "BasinNo":    pid,         # required by QSWAT+
            "geometry":   geom,
        })

    result = gpd.GeoDataFrame(full_rows, crs=gdf.crs)
    int_cols = ["LINKNO", "DSLINKNO", "USLINKNO1", "USLINKNO2", "DSNODEID",
                "Order", "Magnitude", "WSNO", "BasinNo"]
    for col in int_cols:
        result[col] = result[col].astype(int)
    float_cols = ["Length", "DS_Cont_Ar", "Drop", "Slope", "Straight_L",
                  "US_Cont_Ar", "DOUT_END", "DOUT_START", "DOUT_MID"]
    for col in float_cols:
        result[col] = result[col].astype(float)
    return result


def _generate_one_watershed(ws_basins, outlet_hybas_id, rivers, pour_points,
                            target_epsg, output_dir, watershed_name):
    """
    Generate the 4 QSWAT+ 'Use Existing Watershed' files for one outlet.
    ws_basins is already filtered to this watershed's sub-basins.

    Returns dict or None.
    """
    from shapely.ops import unary_union

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wname = watershed_name

    if len(ws_basins) == 0:
        _log(f"  [Warning] [{wname}] No sub-basins, skipping")
        return None

    # ---- Assign sequential PolygonId ----
    _id_col = _get_id_col(ws_basins)
    if "PFAF_ID" in ws_basins.columns:
        ws_basins = ws_basins.sort_values("PFAF_ID").reset_index(drop=True)
    else:
        ws_basins = ws_basins.reset_index(drop=True)
    ws_basins["PolygonId"] = range(1, len(ws_basins) + 1)
    hybas_to_poly = dict(zip(ws_basins[_id_col], ws_basins["PolygonId"]))
    outlet_poly_id = hybas_to_poly[outlet_hybas_id]

    # ---- 1. demsubbasins.shp ----
    sub_out = ws_basins[["PolygonId", "geometry"]].copy().to_crs(target_epsg)
    sub_path = output_dir / "demsubbasins.shp"
    sub_out.to_file(sub_path)

    # ---- 2. demwshed.shp ----
    merged = unary_union(ws_basins.to_crs("EPSG:4326").geometry)
    wshed_out = gpd.GeoDataFrame(
        {"PolygonId": [0]}, geometry=[merged], crs="EPSG:4326"
    ).to_crs(target_epsg)
    wshed_path = output_dir / "demwshed.shp"
    wshed_out.to_file(wshed_path)

    # ---- 3. demchannel.shp ----
    channels = _build_qswat_channels(ws_basins, rivers, hybas_to_poly, target_epsg)
    ch_path = output_dir / "demchannel.shp"
    channels.to_file(ch_path)

    # ---- 4. outlets.shp ----
    outlet_geom = None
    _pp_id_col = _get_id_col(pour_points) if pour_points is not None and len(pour_points) > 0 else "HYBAS_ID"
    if pour_points is not None and len(pour_points) > 0 and _pp_id_col in pour_points.columns:
        match = pour_points[pour_points[_pp_id_col] == outlet_hybas_id]
        if len(match) > 0:
            outlet_geom = match.iloc[0].geometry
    if outlet_geom is None:
        outlet_row = ws_basins[ws_basins[_get_id_col(ws_basins)] == outlet_hybas_id].iloc[0]
        outlet_geom = outlet_row.geometry.centroid

    outlets_out = gpd.GeoDataFrame(
        {"ID": [0], "INLET": [0], "RES": [0], "PTSOURCE": [0]},
        geometry=[outlet_geom], crs="EPSG:4326"
    ).to_crs(target_epsg)
    out_path = output_dir / "outlets.shp"
    outlets_out.to_file(out_path)

    _log(f"  [OK] [{wname}] {len(ws_basins)} sub-basins, {len(channels)} channels -> {output_dir}/")

    return {
        "subbasins": sub_path,
        "watershed": wshed_path,
        "channels":  ch_path,
        "outlets":   out_path,
        "n_subbasins": len(ws_basins),
        "outlet_poly_id": outlet_poly_id,
        "outlet_hybas_id": outlet_hybas_id,
        "target_epsg": target_epsg,
        "basins_gdf": ws_basins,
        "watershed_name": wname,
    }


def generate_qswat_watershed(
    phase1_result,
    outlet=None,
    target_epsg=None,
    output_dir=None,
    min_watershed_basins=2,
    hydrorivers_shp=None,
):
    """
    Phase 1.5: Generate QSWAT+ 'Use Existing Watershed' files from Phase 1 results.

    Parameters
    ----------
    phase1_result : dict
        Return value of prepare_swat_data(); contains:
          basins, rivers, pour_points, dirs, study_area
    outlet : tuple, str, int, or None
        How to select the outlet:
          (lon, lat)        - specify coordinates; nearest outlet used
          "all"             - all outlets in the study area, one directory each
          int (HYBAS_ID)    - directly specify the outlet basin HYBAS_ID
          None              - equivalent to "all"
    target_epsg : int or None
        Target projection (auto-inferred from Phase 1 if not given)
    output_dir : str or None
        Root output directory (defaults to <phase1_dir>/qswat_watersheds/)
    min_watershed_basins : int
        When outlet='all', watersheds with fewer sub-basins than this are skipped

    Returns
    -------
    Single outlet : dict
    Multiple outlets : list of dict

    Jupyter usage:
    ─────────────
    from swat_data_prep import prepare_swat_data, generate_qswat_watershed

    # Step 1: Prepare data
    step1 = prepare_swat_data(country="Cambodia", basin_level=7, dem_resolution=250)

    # Step 2a: Generate QSWAT+ files for a specific outlet
    step2 = generate_qswat_watershed(step1, outlet=(106.96783, 11.11035))

    # Step 2b: Generate for all outlets
    step2 = generate_qswat_watershed(step1, outlet="all")

    # Step 2c: Specify by HYBAS_ID
    step2 = generate_qswat_watershed(step1, outlet=4070816790)
    """
    basins = phase1_result["basins"]
    rivers = phase1_result.get("rivers")
    pour_points = phase1_result.get("pour_points")
    dirs = phase1_result.get("dirs")

    # Default output directory
    if output_dir is None:
        if dirs:
            output_dir = dirs.base / "qswat_watersheds"
        else:
            output_dir = Path("./qswat_watersheds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default projection
    if target_epsg is None:
        centroid = basins.to_crs("EPSG:4326").unary_union.centroid
        target_epsg = _guess_utm_epsg(centroid.x, centroid.y)

    # Use Phase 1 rivers if available; otherwise reload from hydrorivers_shp
    if rivers is None and hydrorivers_shp and Path(hydrorivers_shp).exists():
        print("  [Load] HydroRIVERS (not provided by Phase 1)...")
        bounds = basins.to_crs("EPSG:4326").total_bounds
        buf = 0.2
        bbox = (bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
        rivers = gpd.read_file(hydrorivers_shp, bbox=bbox)

    print("=" * 60)
    print("Phase 1.5: Generating QSWAT+ 'Use Existing Watershed' files")
    _log(f"   Sub-basins: {len(basins)},  projection: EPSG:{target_epsg}")
    _log(f"   Output: {output_dir}/")
    print("=" * 60)

    # ================================================================
    # Determine outlet(s)
    # ================================================================
    all_outlets = _find_country_outlets(basins)
    print(f"\n  Total outlets in study area: {len(all_outlets)}")

    if outlet is None or (isinstance(outlet, str) and outlet.lower() == "all"):
        # ---- All outlets ----
        mode = "all"
        outlet_list = []
        for _, row in all_outlets.iterrows():
            _bid_col = _get_id_col(all_outlets_4326)
            hid = int(row[_bid_col])
            upstream = _get_upstream_basins(basins, hid)
            outlet_list.append((hid, len(upstream)))
        # Sort by descending sub-basin count
        outlet_list.sort(key=lambda x: -x[1])
        _log(f"  Mode: all outlets ({len(outlet_list)} points)")

    elif isinstance(outlet, (int, np.integer)):
        # ---- Directly specified HYBAS_ID ----
        mode = "single"
        outlet_list = [(int(outlet), None)]
        _log(f"  Mode: specified HYBAS_ID = {outlet}")

    elif isinstance(outlet, (tuple, list)) and len(outlet) == 2:
        # ---- Coordinate specified; find nearest outlet ----
        mode = "single"
        lon, lat = float(outlet[0]), float(outlet[1])
        pt = Point(lon, lat)

        # Check whether the point falls within any outlet basin
        containing = all_outlets[all_outlets.to_crs("EPSG:4326").contains(pt)]
        if len(containing) > 0:
            hid = int(containing.iloc[0][_get_id_col(containing)])
        else:
            # Find nearest outlet basin
            all_outlets_4326 = all_outlets.to_crs("EPSG:4326")
            all_outlets_4326["_dist"] = all_outlets_4326.geometry.centroid.distance(pt)
            hid = int(all_outlets_4326.nsmallest(1, "_dist").iloc[0][_get_id_col(all_outlets_4326)])

        outlet_list = [(hid, None)]
        _log(f"  Mode: coordinate ({lon}, {lat}) -> outlet HYBAS_ID = {hid}")

    else:
        raise ValueError(f"Invalid outlet parameter: {outlet}. "
                         f"Valid: (lon,lat), 'all', HYBAS_ID(int), None")

    # ================================================================
    # Generate watersheds
    # ================================================================
    results = []
    skipped = 0

    for i, (outlet_hid, n_pre) in enumerate(outlet_list):
        # Trace upstream basins
        upstream_ids = _get_upstream_basins(basins, outlet_hid)
        n_sub = len(upstream_ids)

        if mode == "all" and n_sub < min_watershed_basins:
            skipped += 1
            continue

        _bid = _get_id_col(basins)
        ws_basins = basins[basins[_bid].isin(upstream_ids)].copy()
        ws_name = f"ws_{i+1:03d}_{n_sub}sub" if mode == "all" else f"ws_{n_sub}sub"
        ws_dir = output_dir / ws_name if mode == "all" else output_dir

        print(f"\n  [{i+1}/{len(outlet_list)}] {ws_name}: "
              f"outlet={outlet_hid}, sub-basins={n_sub}")

        result = _generate_one_watershed(
            ws_basins=ws_basins,
            outlet_hybas_id=outlet_hid,
            rivers=rivers,
            pour_points=pour_points,
            target_epsg=target_epsg,
            output_dir=str(ws_dir),
            watershed_name=ws_name,
        )
        if result:
            results.append(result)

    # ================================================================
    # Summary
    # ================================================================
    if mode == "all":
        print(f"\n{'='*60}")
        _log(f"  Summary: {len(results)} watersheds, {skipped} skipped "
              f"(< {min_watershed_basins} sub-basins)")
        print(f"{'='*60}")

        if results:
            # Overview figure
            fig, ax = plt.subplots(figsize=(12, 10))
            phase1_result["study_area"].to_crs("EPSG:4326").boundary.plot(
                ax=ax, edgecolor="black", linewidth=2, linestyle="--", label="Study area boundary")

            import matplotlib.cm as cm
            colors = cm.Set3(np.linspace(0, 1, max(len(results), 1)))
            for j, r in enumerate(results):
                gdf = r["basins_gdf"].to_crs("EPSG:4326")
                gdf.plot(ax=ax, facecolor=colors[j % len(colors)],
                         edgecolor="gray", linewidth=0.3, alpha=0.6,
                         label=f"{r['watershed_name']} ({r['n_subbasins']} sub-basins)")
            ax.set_title(f"Independent watersheds: {len(results)}", fontsize=14)
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.3); fig.tight_layout()
            save_figure(fig, output_dir / "00_watersheds_overview.png")

            # CSV
            summary = [{
                "watershed": r["watershed_name"],
                "n_subbasins": r["n_subbasins"],
                "outlet_HYBAS_ID": r["outlet_hybas_id"],
                "outlet_PolygonId": r["outlet_poly_id"],
                "EPSG": r["target_epsg"],
            } for r in results]
            df_summary = pd.DataFrame(summary)
            df_summary.to_csv(output_dir / "watersheds_summary.csv", index=False)
            print(f"\n  Summary table:")
            print(df_summary.to_string(index=False))

    # Print usage instructions
    print(f"""
{'='*60}
  Done! In QSWAT+:
{'='*60}
  1. Create new project, load Phase 1 DEM:
     -> {phase1_result['dirs'].rasters if dirs else '???'}/dem.tif

  2. Select 'Use existing watershed' tab:
     Subbasins:  -> <watershed_dir>/demsubbasins.shp
     Watershed:  -> <watershed_dir>/demwshed.shp
     Channels:   -> <watershed_dir>/demchannel.shp
     Outlets:    -> <watershed_dir>/outlets.shp

  3. Click Run -> Step 1 complete (no TauDEM needed!)

  4. Step 2 (Create HRUs):
     Landuse:  -> {phase1_result['dirs'].rasters if dirs else '???'}/landuse.tif
     Soil:     -> {phase1_result['dirs'].rasters if dirs else '???'}/soil.tif

  5. Step 3 (Edit Inputs & Run SWAT+)
""")

    if mode == "single":
        return results[0] if results else None
    return results


# =============================================================================
# Command-line entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SWAT+ Input Data Preparation Tool v5", formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  # Phase 1 only
  python swat_data_prep.py --country Cambodia --dem-resolution 250

  # Phase 1 + Phase 1.5 (specific outlet -> QSWAT+ files)
  python swat_data_prep.py --country Cambodia --qswat-outlet 106.96783 11.11035

  # Phase 1 + Phase 1.5 (all outlets)
  python swat_data_prep.py --country Cambodia --qswat-outlet all
    """)
    g1 = parser.add_argument_group("Study area")
    g1.add_argument("--country", type=str, default=None)
    g1.add_argument("--outlet", type=float, nargs=2, metavar=("LON","LAT"))
    g1.add_argument("--bbox", type=float, nargs=4, metavar=("W","S","E","N"))
    g2 = parser.add_argument_group("Detail level & DEM")
    g2.add_argument("--river-detail", type=int, default=3, choices=[1,2,3,4,5])
    g2.add_argument("--basin-level", type=int, default=12, choices=range(1,13), metavar="1-12")
    g2.add_argument("--dem-resolution", type=int, default=None, metavar="METERS")
    g3 = parser.add_argument_group("Data paths")
    g3.add_argument("--hydrobasins-shp", type=str, default=None)
    g3.add_argument("--hydrorivers-shp", type=str, default=None, help="Path to river shapefile (auto-detected from --grfr-dir if not set)")
    g3.add_argument("--hydrolakes-shp", type=str, default="/home/cfeng/hydro/source_data/atlas/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp")
    g3.add_argument("--hydrobasins-dir", type=str, default="/home/cfeng/hydro/source_data/atlas")
    g3.add_argument("--pour-points-shp", type=str, default=None)
    g3.add_argument("--srtm-dir", type=str, default="/home/cfeng/hydro/dem/srtm90m/")
    g4 = parser.add_argument_group("Other")
    g4.add_argument("--continent", type=str, default="ea", choices=["ea","af","ap","na","sa"])
    g4.add_argument("--output-dir", type=str, default="./swat_output")
    g4.add_argument("--target-epsg", type=int, default=32648)
    g4.add_argument("--original-landuse", action="store_true")
    g4.add_argument("--min-basin-area", type=float, default=0.01)
    g4.add_argument("--max-river-dist", type=float, default=2.0, help="Max pour-point to river distance km (0=no filter, default 2.0)")
    g4.add_argument("--cluster-dist", type=float, default=2.0, help="Cluster dedup distance km (0=no dedup, default 2.0)")
    g4.add_argument("--min-lake-area", type=float, default=1.0, help="Min lake area km2 (0=all, default 1.0)")
    g5 = parser.add_argument_group("Phase 1.5: QSWAT+ watershed generation (optional)")
    g5.add_argument("--qswat-outlet", type=str, nargs="+", default=None,
                    help="'all' or 'LON LAT'; if set, Phase 1.5 runs automatically after Phase 1")
    g5.add_argument("--min-watershed-basins", type=int, default=2)

    args = parser.parse_args()
    o_lon = o_lat = None
    if args.outlet:
        o_lon, o_lat = args.outlet
    country = args.country
    if country is None and o_lon is None and args.bbox is None:
        country = "Cambodia"

    # ---- Phase 1 ----
    step1 = prepare_swat_data(
        country=country, outlet_lon=o_lon, outlet_lat=o_lat, bbox=args.bbox,
        hydrobasins_shp=args.hydrobasins_shp, hydrorivers_shp=args.hydrorivers_shp,
        hydrolakes_shp=args.hydrolakes_shp, hydrobasins_dir=args.hydrobasins_dir,
        pour_points_shp=args.pour_points_shp,
        river_detail=args.river_detail, basin_level=args.basin_level,
        dem_resolution=args.dem_resolution, continent=args.continent,
        output_dir=args.output_dir, target_epsg=args.target_epsg,
        srtm_dir=args.srtm_dir, use_original_landuse=args.original_landuse,
        min_basin_area_km2=args.min_basin_area,
        max_river_dist_km=args.max_river_dist,
        cluster_dist_km=args.cluster_dist,
        min_lake_area_km2=args.min_lake_area,
    )

    # ---- Phase 1.5 (optional) ----
    if args.qswat_outlet:
        if args.qswat_outlet[0].lower() == "all":
            outlet_arg = "all"
        else:
            outlet_arg = (float(args.qswat_outlet[0]), float(args.qswat_outlet[1]))
        generate_qswat_watershed(
            step1,
            outlet=outlet_arg,
            target_epsg=args.target_epsg,
            min_watershed_basins=args.min_watershed_basins,
        )

if __name__ == "__main__":
    main()
