#!/usr/bin/env python3
"""
SWAT+ Multi-Country Batch Data Preparation Tool
================================================
Large countries use precise administrative boundaries (not bbox):
  1. USA    -> BA (Balancing Authority) subregions (read directly from shapefile)
  2. China  -> 7 grid regions (Natural Earth Admin-1 province boundaries)
  3. India  -> 5 grid regions (Admin-1 state boundaries)
  4. Brazil -> SIN 4 subsystems (Admin-1 state boundaries)
  5. Canada -> Provincial electricity jurisdictions (Admin-1)

Usage:
    python batch_countries.py --base-dir ./swat_global --dry-run
    python batch_countries.py --base-dir ./swat_global --only "china_csg"
    python batch_countries.py --base-dir ./swat_global --only "CAISO"
"""

import sys, os, time, shutil, argparse, json, traceback, warnings, math
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# Matplotlib is only imported when needed (inside the plot function) so that
# the rest of the pipeline still works on headless servers without a display.
# Set the backend here so the import inside the function honours it.
import matplotlib
matplotlib.use("Agg")   # non-interactive PNG backend

os.environ["PROJ_LIB"] = "/home/cfeng/.conda/envs/pybkb/lib/python3.11/site-packages/pyproj/proj_dir/share/proj"

# ============================================================================
# HYDROPOWER PLANT TABLES -- edit file paths here
# ============================================================================
# Table 1: GHR hydropower plant list
#   Expected columns: id_unified, project_name_unified, year_unified,
#                     type_unified, ID, country, name, capacity_mw,
#                     lat_unified, lon_unified
HYDRO_TABLE_PATH = "/home/cfeng/hydro/output_data/hydro_locations/step9_snapping_grfr.xlsx"

# Only plants whose snap_quality is in this set will be included
HYDRO_QUALITY_FILTER = {"A", "B"}

# ============================================================================
# GRFR DATA CONFIG
# ============================================================================
# Folder containing GRFR shapefiles.
# Expected structure:
#   GRFR_DIR/pfaf_N_MERIT_Hydro_v07_Basins_v01__extract/
#       cat_pfaf_N_MERIT_Hydro_v07_Basins_v01.shp   <- basins
#       riv_pfaf_N_MERIT_Hydro_v07_Basins_v01.shp   <- rivers
GRFR_DIR = "/home/cfeng/hydro/source_data/grfr"

# STO plants within this distance (km) are merged into one reservoir point
STO_MERGE_KM = 0.1

MAINLAND_BBOX = {
    "Norway":   (-2.0, 57.0, 31.0, 71.5),
    "Portugal": (-9.6, 36.8, -6.0, 42.2),
    "France":   (-5.5, 41.0, 10.0, 51.5),
    "Ecuador":  (-81.0, -5.5, -75.0,  1.5),
}
# ============================================================================
# Utility functions
# ============================================================================

def auto_utm_epsg(lon, lat):
    """Compute UTM EPSG code from longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


# ============================================================================
# Admin-1 boundary data (Natural Earth 10m)
# ============================================================================
_ADMIN1_CACHE = None

ADMIN1_URL  = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
ADMIN1_LOCAL = None


def load_admin1(local_path=None):
    global _ADMIN1_CACHE
    if _ADMIN1_CACHE is not None:
        return _ADMIN1_CACHE

    src = local_path or ADMIN1_LOCAL
    if src and Path(src).exists():
        print(f"  [Local] Loading Admin-1: {src}")
        _ADMIN1_CACHE = gpd.read_file(src)
    else:
        print(f"  [Download] Fetching Natural Earth Admin-1 (~30 MB)...")
        _ADMIN1_CACHE = gpd.read_file(ADMIN1_URL)
        cache_dir = Path("../swat_global/_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "ne_10m_admin_1.shp"
        if not cache_path.exists():
            _ADMIN1_CACHE.to_file(cache_path)
            print(f"  [Cache] Saved to: {cache_path}")
    return _ADMIN1_CACHE


# Natural Earth occasionally renames countries; map config names -> NE names.
# All aliases are tried in order until a match is found.
COUNTRY_NAME_ALIASES = {
    "Czech Republic":        ["Czechia", "Czech Republic"],
    "Bosnia and Herzegovina":["Bosnia and Herzegovina", "Bosnia & Herzegovina",
                              "Bosnia-Herzegovina"],
    "North Macedonia":       ["North Macedonia", "Macedonia"],
    "Timor-Leste":           ["Timor-Leste", "East Timor"],
}


def build_study_area_from_admin1(admin1_gdf, country_name, province_names, output_shp):
    """Extract and dissolve Admin-1 features by province name list."""
    output_shp = Path(output_shp)
    if output_shp.exists():
        return output_shp

    # Try all known aliases for this country name
    aliases = COUNTRY_NAME_ALIASES.get(country_name, [country_name])
    country_gdf = gpd.GeoDataFrame()
    for alias in aliases:
        mask = admin1_gdf["admin"].str.lower() == alias.lower()
        if mask.any():
            country_gdf = admin1_gdf[mask]
            break
    if len(country_gdf) == 0:
        # Last resort: substring search across all aliases
        for alias in aliases:
            mask = admin1_gdf["admin"].str.contains(alias, case=False, na=False)
            if mask.any():
                country_gdf = admin1_gdf[mask]
                break
    if len(country_gdf) == 0:
        raise ValueError(f"Country '{country_name}' not found in Admin-1 data "
                         f"(tried: {aliases})")

    name_cols = [c for c in ["name", "name_en", "name_local", "gn_name", "name_alt"]
                 if c in country_gdf.columns]

    matched_idx = set()
    unmatched   = []
    for prov in province_names:
        found      = False
        prov_lower = prov.lower().strip()
        for col in name_cols:
            mask = country_gdf[col].str.lower().str.strip() == prov_lower
            if mask.any():
                matched_idx.update(country_gdf[mask].index.tolist())
                found = True; break
        if not found:
            for col in name_cols:
                mask = country_gdf[col].str.contains(prov, case=False, na=False)
                if mask.any():
                    matched_idx.update(country_gdf[mask].index.tolist())
                    found = True; break
        if not found:
            unmatched.append(prov)

    if unmatched:
        avail = sorted(country_gdf["name"].dropna().unique())
        print(f"    [Warning] Unmatched provinces: {unmatched}")
        print(f"    Available ({len(avail)}): {avail[:20]}...")
    if not matched_idx:
        raise ValueError(f"No provinces matched: {province_names}")

    selected = country_gdf.loc[list(matched_idx)]
    print(f"    Matched {len(selected)}/{len(province_names)} provinces/states")

    dissolved = selected.dissolve().to_crs("EPSG:4326")
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    dissolved[["geometry"]].to_file(output_shp)
    return output_shp


def build_study_area_from_shapefile(shp_path, attr_col, attr_value, output_shp):
    """Extract and dissolve shapefile features by attribute value."""
    output_shp = Path(output_shp)
    if output_shp.exists():
        return output_shp

    gdf      = gpd.read_file(shp_path)
    selected = gdf[gdf[attr_col] == attr_value]
    if len(selected) == 0:
        raise ValueError(f"No feature with {attr_col}='{attr_value}' in {shp_path}")

    dissolved = selected.dissolve().to_crs("EPSG:4326")
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    dissolved[["geometry"]].to_file(output_shp)
    return output_shp

def clip_to_mainland(gdf, country_name):
    if country_name not in MAINLAND_BBOX:
        return gdf
    from shapely.geometry import box as _box
    clip_geom = _box(*MAINLAND_BBOX[country_name])
    gdf = gdf.to_crs("EPSG:4326").copy()
    clipped = gdf.geometry.intersection(clip_geom)
    valid = ~clipped.is_empty
    gdf = gdf[valid].copy()
    gdf.geometry = clipped[valid]
    print(f"  [Mainland] Clipped '{country_name}' to mainland bbox")
    return gdf

# ============================================================================
# Hydropower plant helpers
# ============================================================================

_HYDRO_MERGED_CACHE = None


def load_hydro_merged():
    """
    Load the single unified hydropower table and filter by quality.
    Returns a GeoDataFrame (EPSG:4326), or None if file is missing.
    """
    global _HYDRO_MERGED_CACHE
    if _HYDRO_MERGED_CACHE is not None:
        return _HYDRO_MERGED_CACHE

    p = Path(HYDRO_TABLE_PATH)
    if not p.exists():
        print(f"  [Warning] HYDRO_TABLE_PATH not found: {p}  (hydro features skipped)")
        return None

    print(f"  [Hydro] Loading unified table: {p}")
    # Read all columns as strings to prevent IDs from being truncated
    df = pd.read_excel(p, dtype=str)

    # 1. Normalize the ID column name (in case the merged header is ID, not id_unified)
    if "id_unified" not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": "id_unified"})

    # 2. Quality Filter
    if "snap_quality" in df.columns:
        before = len(df)
        df = df[df["snap_quality"].isin(HYDRO_QUALITY_FILTER)]
        print(f"  [Hydro] Quality filter ({'/'.join(sorted(HYDRO_QUALITY_FILTER))}): "
              f"{before} -> {len(df)} plants retained")
    else:
        print("  [Warning] 'snap_quality' column not found. Skipping quality filter (keeping all).")

    # 3. Numeric Coercion
    # Ensure coordinates, capacity, and river IDs are in numeric format
    # GRFR columns: GRFR_COMID (river segment), GRFR_BASIN_COMID (basin outlet)
    cols_to_numeric = ["lat_unified", "lon_unified", "capacity_mw",
                       "GRFR_COMID", "GRFR_BASIN_COMID",
                       "final_HYRIV_ID"]  # keep for backward compat

    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Remove invalid coordinates
    df = df.dropna(subset=["lat_unified", "lon_unified"])

    if len(df) == 0:
        print("  [Warning] No valid plants loaded after filtering.")
        return None

    # 5. Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.lon_unified, df.lat_unified)],
        crs="EPSG:4326",
    )

    _HYDRO_MERGED_CACHE = gdf
    return gdf


def get_country_names_for_task(task):
    """Return country name(s) associated with the task for attribute-based filtering."""
    if task["type"] == "country":
        return [task["country"]] if task["country"] else []
    elif task["type"] == "admin":
        return [task["admin_country"]] if task["admin_country"] else []
    elif task["type"] == "ba_shapefile":
        return ["United States of America", "United States", "USA"]
    return []


def filter_hydro_for_task(hydro_gdf, task, study_shp=None):
    """
    Return the subset of hydro_gdf relevant to this task.
    Spatial filter is preferred (when study_shp is available);
    falls back to matching the 'country' column by name.
    """
    if hydro_gdf is None or len(hydro_gdf) == 0:
        return None

    # Spatial filter using study area polygon
    if study_shp and Path(study_shp).exists():
        try:
            boundary = gpd.read_file(study_shp).to_crs("EPSG:4326")
            subset   = hydro_gdf[hydro_gdf.geometry.within(boundary.union_all())]
            return subset if len(subset) > 0 else None
        except Exception as e:
            print(f"    [Warning] Spatial hydro filter failed: {e}")

    # Attribute (country name) fallback
    country_names = get_country_names_for_task(task)
    if not country_names:
        return None

    country_col = "country" if "country" in hydro_gdf.columns else None
    if country_col is None:
        return None

    subset = hydro_gdf[hydro_gdf[country_col].isin(country_names)]
    return subset if len(subset) > 0 else None


def write_hydro_table(subset, output_dir, hybas_lv_series=None):
    """Write a CSV table of ALL quality-filtered plants for this region.

    All plant types (STO, ROR, Canal, PS, ...) are included.
    For STO plants that were merged in write_inlets_outlets_shp, the
    'merged_plants' column lists the names of other plants in the same
    5-km cluster (empty for singletons and non-STO types).

    Parameters
    ----------
    hybas_lv_series : pd.Series or None
        If provided, a Series indexed like subset whose values are the
        GRFR basin COMID for each plant (from GRFR_BASIN_COMID).
        Written as column  grfr_basin_comid  in the CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = subset.drop(columns="geometry", errors="ignore").copy()

    # Add HydroBasins level-N basin ID if mapping was provided
    if hybas_lv_series is not None:
        col_name = "grfr_basin_comid"
        df[col_name] = hybas_lv_series.values if hasattr(hybas_lv_series, "values") else list(hybas_lv_series)

    # Annotate STO merge clusters so the CSV reflects which plants were merged
    type_col = "type_unified" if "type_unified" in df.columns else None
    if type_col:
        sto_mask = df[type_col].str.upper() == "STO"
        if sto_mask.any():
            sto_clustered = cluster_sto_plants(
                df[sto_mask].copy().reset_index(drop=True), merge_dist_km=STO_MERGE_KM)
            # Map back by original index positions
            sto_indices = df.index[sto_mask].tolist()
            df.loc[sto_indices, "sto_group"]     = sto_clustered["sto_group"].values
            df.loc[sto_indices, "merged_plants"]  = sto_clustered["merged_plants"].values
        # Non-STO rows get empty strings
        df["sto_group"]    = df["sto_group"].fillna("").astype(str).replace("nan", "") if "sto_group" in df.columns else ""
        df["merged_plants"] = df["merged_plants"].fillna("").astype(str) if "merged_plants" in df.columns else ""

    if type_col and "GRFR_order" in df.columns:  # adjust the column name to match your data
        df["closed_loop_ps"] = (
            (df[type_col].str.upper() == "PS") &
            (pd.to_numeric(df["GRFR_order"], errors="coerce") == 1)
        ).map({True: "Yes", False: ""})
    out_csv = output_dir / "hydro_plants.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    if type_col:
        counts    = df[type_col].str.upper().value_counts().to_dict()
        breakdown = "  ".join(f"{t}:{n}" for t, n in sorted(counts.items()))
        n_merged  = int((df.get("merged_plants", pd.Series(dtype=str)) != "").sum())
        extra     = f"  |  {n_merged} STO merged" if n_merged else ""
        hybas_note = "  |  grfr_basin_comid written" if hybas_lv_series is not None else ""
        print(f"    [Hydro] Plant table saved: {out_csv}  "
              f"({len(df)} plants | {breakdown}{extra}{hybas_note})")
    else:
        print(f"    [Hydro] Plant table saved: {out_csv}  ({len(df)} plants)")
    return out_csv


# ---------------------------------------------------------------------------
# Utility: Haversine distance (km) between two WGS-84 lon/lat points
# ---------------------------------------------------------------------------
def _haversine_km(lon1, lat1, lon2, lat2):
    import math
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat / 2) ** 2 + \
           math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
           math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def cluster_sto_plants(sto_df, merge_dist_km=5.0):
    """
    Cluster STO plants within merge_dist_km of each other (single-linkage).
    Returns a copy of sto_df with two new columns:

      sto_group      (int) : cluster id; plants in the same cluster share one id
      merged_plants  (str) : semicolon-separated names of OTHER plants in the
                             same cluster (empty string if singleton)

    The representative for each cluster is the plant with the highest capacity.
    """
    if sto_df is None or len(sto_df) == 0:
        return sto_df

    df = sto_df.copy().reset_index(drop=True)
    n  = len(df)

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    lons = df["lon_unified"].astype(float).tolist()
    lats = df["lat_unified"].astype(float).tolist()
    for i in range(n):
        for j in range(i + 1, n):
            if _haversine_km(lons[i], lats[i], lons[j], lats[j]) < merge_dist_km:
                union(i, j)

    root_to_gid = {}
    gid_counter = 0
    groups = []
    for i in range(n):
        r = find(i)
        if r not in root_to_gid:
            root_to_gid[r] = gid_counter
            gid_counter += 1
        groups.append(root_to_gid[r])
    df["sto_group"] = groups

    name_col = "name" if "name" in df.columns else \
               "project_name_unified" if "project_name_unified" in df.columns else None

    merged_col = [""] * n
    for gid in df["sto_group"].unique():
        members = df[df["sto_group"] == gid]
        if len(members) == 1:
            continue
        all_names = [
            str(r[name_col]) if name_col else f"plant_{idx}"
            for idx, r in members.iterrows()
        ]
        for pos, (idx, _) in enumerate(members.iterrows()):
            others = [nm for p2, nm in enumerate(all_names) if p2 != pos]
            merged_col[df.index.get_loc(idx)] = "; ".join(others)
    df["merged_plants"] = merged_col
    return df


def get_grfr_basin_pour_points(subset, grfr_dir=None):
    """
    Return GRFR basin outlet pour-points for each plant's GRFR_BASIN_COMID.

    Always returns a TUPLE: (pour_gdf, plant_series)
      pour_gdf     : GeoDataFrame of pour points (centroid of matched GRFR basin)
                     Columns: [COMID, POUR_LAT, POUR_LON, geometry]  or None
      plant_series : pd.Series indexed like subset, value = GRFR_BASIN_COMID
                     for each plant, or None

    Lookup path:
      For each plant, read GRFR_BASIN_COMID -> find the matching cat_pfaf basin
      polygon -> use its centroid as the pour point.
    """
    import pandas as _pd

    if grfr_dir is None:
        grfr_dir = GRFR_DIR
    grfr_path = Path(grfr_dir)
    if not grfr_path.exists():
        return None, None
    if subset is None or len(subset) == 0:
        return None, None

    basin_col = "GRFR_BASIN_COMID"
    if basin_col not in subset.columns:
        print(f"    [GRFR] Column '{basin_col}' not found in hydro table - skipping pour points")
        return None, None

    basin_ids = set(
        int(float(v)) for v in subset[basin_col].dropna()
        if str(v).strip() not in ("", "nan")
    )
    if not basin_ids:
        return None, None

    bbox = _bbox_from_subset(subset)

    # Scan all pfaf subdirs for cat_pfaf basins
    cat_frames = []
    for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
        for cat_shp in sorted(pfaf_dir.glob("cat_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
            try:
                chunk = gpd.read_file(str(cat_shp), bbox=bbox)
                if len(chunk) > 0:
                    cat_frames.append(chunk)
            except Exception:
                pass

    if not cat_frames:
        print(f"    [GRFR] No GRFR basin data found for bbox {bbox}")
        return None, None

    cat_all = pd.concat(cat_frames, ignore_index=True)
    if not isinstance(cat_all, gpd.GeoDataFrame):
        cat_all = gpd.GeoDataFrame(cat_all, geometry="geometry")
    if getattr(cat_all, 'crs', None) is None:
        cat_all = cat_all.set_crs("EPSG:4326")
    id_col = next((c for c in ["COMID", "comid", "HYBAS_ID"] if c in cat_all.columns), None)
    if id_col is None:
        print(f"    [GRFR] No COMID column in GRFR basin data")
        return None, None

    cat_all[id_col] = pd.to_numeric(cat_all[id_col], errors="coerce")
    matched = cat_all[cat_all[id_col].isin(basin_ids)].drop_duplicates(id_col).copy()
    matched = gpd.GeoDataFrame(matched, crs=matched.crs if hasattr(matched, 'crs') else "EPSG:4326").to_crs("EPSG:4326")

    if len(matched) == 0:
        return None, None

    # Use centroid as pour point
    matched["POUR_LAT"] = matched.geometry.centroid.y
    matched["POUR_LON"] = matched.geometry.centroid.x
    matched["geometry"] = matched.geometry.centroid
    matched = matched.rename(columns={id_col: "COMID"})

    # per-plant series: subset.index -> GRFR_BASIN_COMID
    plant_series = subset[basin_col].apply(
        lambda x: int(float(x)) if _pd.notna(x) and str(x).strip() not in ("", "nan") else None)

    print(f"    [GRFR] Pour points: {len(matched)} basin centroids from {len(basin_ids)} COMIDs")
    return matched[["COMID", "POUR_LAT", "POUR_LON", "geometry"]].to_crs("EPSG:4326"), plant_series


# Keep old name as alias (called from process_hydro_for_task)
def get_hybas_pour_points(subset, hybas_dir=None, hybas_level=None):
    """Alias for get_grfr_basin_pour_points (GRFR replacement for HydroBasins)."""
    return get_grfr_basin_pour_points(subset, grfr_dir=hybas_dir or GRFR_DIR)
def write_inlets_outlets_shp(subset, task_output_dir, output_dir, target_epsg=4326,
                             _precomputed_hybas=None):
    """
    Create a QSWAT+-compatible inlets/outlets shapefile containing:

      ① STO (reservoir) plants  →  RES=1, INLET=0, PTSOURCE=0
           STO plants within 5 km of each other are merged into one point
           (representative = highest capacity; others noted in the CSV).

      ② ROR / CANAL / PS plants →  RES=0, INLET=0, PTSOURCE=0
           Treated as plain pour-points so QSWAT+ sub-divides the channel
           at each non-reservoir hydropower location.

      ③ GRFR basin outlet pour-points  →  RES=0, INLET=0, PTSOURCE=0
           One per occupied GRFR basin (≤ number of plants).

    Existing pour-points from prepare_swat_data are NOT included.
    Add the main watershed outlet manually via "Draw inlets/outlets" in QSWAT+.

    QSWAT+ REQUIRED fields (exact names):
      ID (int), RES (int), INLET (int), PTSOURCE (int)

    Extra informational fields (ignored by QSWAT+):
      NAME, CAP_MW, YEAR, QUAL, TYPE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subset is None or len(subset) == 0:
        print("    [Hydro] No plants - inlets_outlets.shp skipped.")
        return None

    type_col = "type_unified" if "type_unified" in subset.columns else None
    name_col = "name" if "name" in subset.columns else \
               "project_name_unified" if "project_name_unified" in subset.columns else None
    cap_col  = "capacity_mw" if "capacity_mw" in subset.columns else None

    def _get_name(row, fallback):
        v = row[name_col] if name_col and name_col in row.index else fallback
        return str(v)[:40]

    def _get_year(row):
        yr = row.get("year_unified", None) if hasattr(row, "get") else getattr(row, "year_unified", None)
        try:
            return int(float(yr)) if yr and str(yr).strip() not in ("", "nan") else -9999
        except (ValueError, TypeError):
            return -9999

    rows    = []
    next_id = 1
    # ── ① STO reservoir points (merge clusters within 5 km) ────────────────
    n_sto_raw = 0
    n_sto_kept = 0
    if type_col:
        sto_df = subset[subset[type_col].str.upper() == "STO"].copy()
    else:
        sto_df = subset.iloc[0:0]

    n_sto_raw = len(sto_df)
    if n_sto_raw > 0:
        sto_cl = cluster_sto_plants(sto_df, merge_dist_km=STO_MERGE_KM)
        for gid in sto_cl["sto_group"].unique():
            members = sto_cl[sto_cl["sto_group"] == gid]
            # Representative = highest capacity
            if cap_col and cap_col in members.columns:
                rep = members.loc[members[cap_col].astype(float).idxmax()]
            else:
                rep = members.iloc[0]
            rows.append({
                "ID":       next_id,
                "RES":      1,
                "INLET":    0,
                "PTSOURCE": 0,
                "NAME":     _get_name(rep, f"RES_{next_id}"),
                "CAP_MW":   float(rep[cap_col]) if cap_col else -9999.0,
                "YEAR":     _get_year(rep),
                "QUAL":     str(rep.get("snap_quality", "") if hasattr(rep, "get") else ""),
                "TYPE":     "STO",
                "geometry": Point(float(rep["lon_unified"]), float(rep["lat_unified"])),
            })
            next_id   += 1
            n_sto_kept += 1
        n_merged = n_sto_raw - n_sto_kept
        if n_merged:
            print(f"    [Hydro] STO: {n_sto_raw} plants → {n_sto_kept} points "
                  f"({n_merged} merged within {STO_MERGE_KM} km)")
        else:
            print(f"    [Hydro] STO: {n_sto_kept} reservoir points")

    # ── ② ROR / CANAL / PS → pour-points (RES=0) ──────────────────────────
    NON_STO = {"ROR", "CANAL", "PS"}
    n_pour  = 0
    if type_col:
        non_sto = subset[subset[type_col].str.upper().isin(NON_STO)].copy()
    else:
        non_sto = subset.iloc[0:0]

    for _, row in non_sto.iterrows():
        ptype = str(row.get(type_col, "ROR")).upper() if type_col else "ROR"
        if ptype == "PS":
            order_val = row.get("GRFR_order", None)  # or the actual column name in your table
            if order_val is not None and int(float(order_val)) == 1:
                continue  # do not add to the shapefile
        rows.append({
            "ID":       next_id,
            "RES":      0,
            "INLET":    0,
            "PTSOURCE": 0,
            "NAME":     _get_name(row, f"PP_{next_id}"),
            "CAP_MW":   float(row[cap_col]) if cap_col and row[cap_col] is not None else -9999.0,
            "YEAR":     _get_year(row),
            "QUAL":     str(row.get("snap_quality", "")),
            "TYPE":     ptype,
            "geometry": Point(float(row["lon_unified"]), float(row["lat_unified"])),
        })
        next_id += 1
        n_pour  += 1
    if n_pour:
        print(f"    [Hydro] Pour-points: {n_pour} (ROR/CANAL/PS)")

    # ── ③ HydroBasins level-N outlet pour-points (downstream monitoring) ──
    n_hybas = 0
    hybas_pts = _precomputed_hybas  # already computed in process_hydro_for_task
    if hybas_pts is not None and len(hybas_pts) > 0:
        for row in hybas_pts.itertuples():
            comid_val = int(row.COMID) if hasattr(row, "COMID") else next_id
            rows.append({
                "ID":       next_id,
                "RES":      0,
                "INLET":    0,
                "PTSOURCE": 0,
                "NAME":     f"GRFR_{comid_val}",
                "CAP_MW":   -9999.0,
                "YEAR":     -9999,
                "QUAL":     "",
                "TYPE":     "GRFR_BASIN",
                "HYBAS_ID": str(comid_val),
                "geometry": row.geometry,
            })
            next_id  += 1
            n_hybas  += 1
        print(f"    [Hydro] GRFR basin pour-points: {n_hybas}")

    if not rows:
        print("    [Hydro] No points to write - inlets_outlets.shp skipped.")
        return None

    # ── ④ Deduplicate: remove points within 5 km of each other ───────────
# If two points are within 5 km, keep the one with RES=1 (reservoir).
# If both are same type, keep the one with higher CAP_MW (or first if equal).
    DEDUP_KM = 5.0
    if len(rows) > 1:
        kept = []
        dropped = set()
        for i, a in enumerate(rows):
            if i in dropped:
                continue
            for j, b in enumerate(rows):
                if j <= i or j in dropped:
                    continue
                dist = _haversine_km(
                    a["geometry"].x, a["geometry"].y,
                    b["geometry"].x, b["geometry"].y,
                )
                if dist < DEDUP_KM:
                    # Decide which to drop
                    if a["RES"] == b["RES"]:
                        # Same type: drop the lower capacity one
                        drop_idx = j if (a["CAP_MW"] or -9999) >= (b["CAP_MW"] or -9999) else i
                    else:
                        # Different type: drop the non-reservoir (RES=0)
                        drop_idx = j if a["RES"] == 1 else i
                    dropped.add(drop_idx)
            if i not in dropped:
                kept.append(a)
        if len(dropped) > 0:
            print(f"    [Hydro] Dedup: removed {len(dropped)} points within {DEDUP_KM} km of another")
        rows = kept

    combined = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    if target_epsg and int(target_epsg) != 4326:
        combined = combined.to_crs(f"EPSG:{target_epsg}")

    out_shp = output_dir / "inlets_outlets.shp"
    combined.to_file(out_shp)
    print(f"    [Hydro] inlets_outlets.shp -> {out_shp}")
    print(f"            STO: {n_sto_kept}  |  ROR/CANAL/PS: {n_pour}"
          f"  |  GRFR basin: {n_hybas}  |  EPSG:{target_epsg}")

    # ── ⑤ Extra: all-outlets version (STO reservoir -> outlet, RES=0 for all) ─
    # A copy of inlets_outlets.shp where every point is treated as a plain
    # outlet (INLET=0, RES=0, PTSOURCE=0), including STO reservoir plants.
    combined_all_outlets = combined.copy()
    combined_all_outlets["RES"] = 0
    out_shp_all = output_dir / "inlets_outlets_all_outlets.shp"
    combined_all_outlets.to_file(out_shp_all)
    print(f"    [Hydro] inlets_outlets_all_outlets.shp -> {out_shp_all}")
    print(f"            (All {len(combined_all_outlets)} points set to RES=0 / plain outlet)")

    return out_shp


def find_river_shp(task_output_dir):
    """
    Search common locations inside the task output directory for a river
    shapefile produced by prepare_swat_data / swat_data_prep.

    Typical candidates (adjust if your pipeline uses different names):
      <output>/vectors/rivers.shp
      <output>/rivers/rivers.shp
      <output>/hydrorivers_clipped.shp
      <output>/hydrorivers.shp
      <output>/rasters/../vectors/*.shp  (any .shp with "river" in name)
    """
    root = Path(task_output_dir)
    # Explicit priority list
    candidates = [
        root / "vectors"  / "rivers.shp",
        root / "rivers"   / "rivers.shp",
        root / "hydrorivers_clipped.shp",
        root / "hydrorivers.shp",
        root / "vectors"  / "hydrorivers.shp",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Generic scan: first .shp whose stem contains "river"
    for p in sorted(root.rglob("*.shp")):
        if "river" in p.stem.lower():
            return p

    return None


# ============================================================================
# Upstream river tracing via GRFR COMID -> NextDownID topology
# ============================================================================

# GRFR river data directory (pfaf_*_MERIT_Hydro_v07_Basins_v01__extract/)
GRFR_DIR = "/home/cfeng/hydro/source_data/grfr"


def _load_grfr_topology_bbox(bbox, buf=15.0):
    """
    Load GRFR topology (COMID -> NextDownID) for a buffered bbox.
    Scans all pfaf_* subdirs and reads riv_pfaf_* shapefiles.
    Returns (topo_dict, children_dict) or (None, None).
    """
    grfr_path = Path(GRFR_DIR)
    if not grfr_path.exists():
        print(f"  [Warning] GRFR_DIR not found: {grfr_path}")
        return None, None

    read_bbox = (bbox[0] - buf, bbox[1] - buf, bbox[2] + buf, bbox[3] + buf)
    frames = []
    for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
        for riv_shp in sorted(pfaf_dir.glob("riv_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
            try:
                chunk = gpd.read_file(str(riv_shp), bbox=read_bbox)
                if len(chunk) > 0:
                    frames.append(chunk)
            except Exception:
                pass
    if not frames:
        return None, None

    rivers = pd.concat(frames, ignore_index=True)
    if not isinstance(rivers, gpd.GeoDataFrame):
        rivers = gpd.GeoDataFrame(rivers, geometry="geometry")
    if getattr(rivers, 'crs', None) is None:
        rivers = rivers.set_crs("EPSG:4326")

    # Detect ID column: COMID (GRFR) or HYRIV_ID (HydroRIVERS)
    id_col  = next((c for c in ["COMID", "HYRIV_ID", "comid"] if c in rivers.columns), None)
    nd_col  = next((c for c in ["NextDownID", "NEXT_DOWN", "nextdownid"] if c in rivers.columns), None)
    if id_col is None or nd_col is None:
        print(f"  [Warning] GRFR river topology columns not found: {list(rivers.columns)}")
        return None, None

    topo = {}
    children = {}
    for _, row in rivers.iterrows():
        hid = int(row[id_col])
        nd  = int(row[nd_col])
        if hid:
            topo[hid] = nd
            children.setdefault(nd, []).append(hid)

    return topo, children


# Keep old name as alias for backwards-compat calls inside this file
def _load_hydrorivers_topology_bbox(bbox, buf=15.0):
    return _load_grfr_topology_bbox(bbox, buf=buf)


def trace_upstream_rivers(start_hyriv_ids, topo, children):
    """
    Given starting HYRIV_IDs (where plants sit), trace ALL upstream river
    segments using the NEXT_DOWN topology (iterative BFS on children map).

    Returns set of all upstream HYRIV_IDs (including starting segments).
    """
    if not topo or not children:
        return set(start_hyriv_ids)

    upstream = set()
    stack = list(start_hyriv_ids)
    while stack:
        cur = stack.pop()
        if cur in upstream or cur == 0:
            continue
        upstream.add(cur)
        for child in children.get(cur, []):
            if child not in upstream:
                stack.append(child)
    return upstream


def get_upstream_l12_basins(subset, hydrobasins_dir=None, hydrorivers_shp=None):
    """
    For each plant with a valid GRFR_COMID, trace upstream rivers and
    find all GRFR cat_pfaf basins that those rivers pass through.

    Returns (expanded_basins_gdf, upstream_comids) or (None, set()).
    """
    if subset is None or len(subset) == 0:
        return None, set()

    comid_col = "GRFR_COMID"
    if comid_col not in subset.columns:
        print(f"  [Upstream] Column '{comid_col}' not found - skipping")
        return None, set()

    # Collect starting COMIDs
    start_ids = set()
    for val in subset[comid_col].dropna():
        try:
            start_ids.add(int(float(val)))
        except (ValueError, TypeError):
            pass

    if not start_ids:
        print(f"  [Upstream] No valid {comid_col} values")
        return None, set()

    print(f"  [Upstream] {len(start_ids)} plant river segments to trace")

    # Compute plant bbox
    if all(c in subset.columns for c in ["lon_unified", "lat_unified"]):
        lons = pd.to_numeric(subset["lon_unified"], errors="coerce")
        lats = pd.to_numeric(subset["lat_unified"], errors="coerce")
        bbox = (lons.min(), lats.min(), lons.max(), lats.max())
    elif "geometry" in subset.columns:
        bbox = tuple(subset.to_crs("EPSG:4326").total_bounds)
    else:
        return None, set()

    # Load GRFR topology (generous buffer - upstream can be far)
    topo, children = _load_grfr_topology_bbox(bbox, buf=15.0)
    if topo is None:
        return None, set()

    # Trace upstream
    all_upstream = trace_upstream_rivers(start_ids, topo, children)
    print(f"  [Upstream] {len(all_upstream):,} upstream segments traced")

    # Load upstream GRFR river geometries to determine their spatial extent
    grfr_path = Path(GRFR_DIR)
    up_bbox_load = (bbox[0]-15, bbox[1]-15, bbox[2]+15, bbox[3]+15)
    riv_frames = []
    for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
        for riv_shp in sorted(pfaf_dir.glob("riv_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
            try:
                chunk = gpd.read_file(str(riv_shp), bbox=up_bbox_load)
                if len(chunk) > 0:
                    riv_frames.append(chunk)
            except Exception:
                pass
    if not riv_frames:
        return None, all_upstream

    rivers_gdf = pd.concat(riv_frames, ignore_index=True)
    if not isinstance(rivers_gdf, gpd.GeoDataFrame):
        rivers_gdf = gpd.GeoDataFrame(rivers_gdf, geometry="geometry")
    if getattr(rivers_gdf, 'crs', None) is None:
        rivers_gdf = rivers_gdf.set_crs("EPSG:4326")
    # Detect COMID column
    riv_id_col = next((c for c in ["COMID", "HYRIV_ID", "comid"] if c in rivers_gdf.columns), None)
    if riv_id_col is None:
        print("  [Upstream] Cannot find COMID column in GRFR river data")
        return None, all_upstream
    upstream_rivers = gpd.GeoDataFrame(
        rivers_gdf[rivers_gdf[riv_id_col].isin(all_upstream)],
        crs=rivers_gdf.crs if hasattr(rivers_gdf, 'crs') else "EPSG:4326"
    ).to_crs("EPSG:4326")

    if len(upstream_rivers) == 0:
        return None, all_upstream

    up_bounds = upstream_rivers.total_bounds
    up_bbox = (up_bounds[0]-0.5, up_bounds[1]-0.5,
               up_bounds[2]+0.5, up_bounds[3]+0.5)

    # Load GRFR cat_pfaf basins covering the upstream rivers
    cat_frames = []
    for pfaf_dir in sorted(grfr_path.glob("pfaf_*_MERIT_Hydro_v07_Basins_v01__extract")):
        for cat_shp in sorted(pfaf_dir.glob("cat_pfaf_*_MERIT_Hydro_v07_Basins_v01.shp")):
            try:
                chunk = gpd.read_file(str(cat_shp), bbox=up_bbox)
                if len(chunk) > 0:
                    cat_frames.append(chunk)
            except Exception:
                pass

    if not cat_frames:
        return None, all_upstream

    # Detect basin ID column
    cat_all = pd.concat(cat_frames, ignore_index=True)
    if not isinstance(cat_all, gpd.GeoDataFrame):
        cat_all = gpd.GeoDataFrame(cat_all, geometry="geometry")
    if getattr(cat_all, 'crs', None) is None:
        cat_all = cat_all.set_crs("EPSG:4326")
    cat_id_col = next((c for c in ["COMID", "HYBAS_ID", "comid"] if c in cat_all.columns), None)
    if cat_id_col:
        cat_all = cat_all.drop_duplicates(cat_id_col)
    cat_all = gpd.GeoDataFrame(cat_all, crs="EPSG:4326")

    # Spatial join: which GRFR basins intersect upstream rivers
    joined = gpd.sjoin(
        cat_all[([cat_id_col, "geometry"] if cat_id_col else ["geometry"])],
        upstream_rivers[["geometry"]],
        how="inner", predicate="intersects"
    )
    if cat_id_col:
        expanded_ids = joined[cat_id_col].unique()
        expanded = cat_all[cat_all[cat_id_col].isin(expanded_ids)].copy()
    else:
        expanded = cat_all.loc[joined.index.unique()].copy()

    print(f"  [Upstream] {len(expanded):,} GRFR basins cover upstream watersheds")
    return expanded, all_upstream


def expand_study_area_with_upstream(study_area, subset, hydrobasins_dir=None,
                                     hydrorivers_shp=None):
    """
    Expand the study area to include all L12 basins that contain upstream
    rivers of plants (traced via GRFR_COMID).

    Returns (expanded_study_area, area_before_km2, area_after_km2).
    """
    from shapely.ops import unary_union

    sa_moll = study_area.to_crs("ESRI:54009")
    area_before = float(sa_moll.geometry.area.sum()) / 1e6

    expanded_basins, _ = get_upstream_l12_basins(
        subset, hydrobasins_dir=hydrobasins_dir,
        hydrorivers_shp=hydrorivers_shp)

    if expanded_basins is None or len(expanded_basins) == 0:
        print(f"  [Expand] No upstream basins to add - area unchanged "
              f"({area_before:,.0f} km²)")
        return study_area, area_before, area_before

    merged_geom = unary_union(
        list(study_area.to_crs("EPSG:4326").geometry) +
        list(expanded_basins.to_crs("EPSG:4326").geometry)
    )
    expanded_gdf = gpd.GeoDataFrame(
        {"geometry": [merged_geom]}, crs="EPSG:4326")

    exp_moll = expanded_gdf.to_crs("ESRI:54009")
    area_after = float(exp_moll.geometry.area.sum()) / 1e6

    delta = area_after - area_before
    pct = (area_after / max(area_before, 1) - 1) * 100
    print(f"  [Expand] Study area: {area_before:,.0f} → {area_after:,.0f} km² "
          f"(+{delta:,.0f} km², +{pct:.1f}%)")

    return expanded_gdf, area_before, area_after


def plot_rivers_and_plants(subset, task_output_dir, output_dir,
                           task_label="", study_shp=None):
    """
    Generate a PNG map showing rivers and hydropower plant locations.

    Layout:
      • Study-area boundary  - thin dark outline, light grey fill
      • Rivers               - blue lines, width scaled by stream order
                               (uses 'ORD_STRA' / 'order' / 'STRAHLER' column if present)
      • ROR / Canal plants   - circles  (o), size ~ sqrt(capacity_mw)
      • STO plants           - triangles (^), size ~ sqrt(capacity_mw)
      • Colour               - capacity_mw, log-scaled colourbar
      • Quality A vs B       - solid vs dashed edge

    The PNG is saved to  <output_dir>/rivers_and_plants.png
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("    [Plot] matplotlib not available - skipping plot.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "rivers_and_plants.png"

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_aspect("equal")

    # ------------------------------------------------------------------
    # 1. Study-area boundary
    # ------------------------------------------------------------------
    if study_shp and Path(study_shp).exists():
        try:
            boundary = gpd.read_file(study_shp).to_crs("EPSG:4326")
            boundary.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#333333",
                          linewidth=0.8, zorder=1)
        except Exception as e:
            print(f"    [Plot] Could not draw boundary: {e}")

    # ------------------------------------------------------------------
    # 2. Rivers
    # ------------------------------------------------------------------
    river_shp = find_river_shp(task_output_dir)
    if river_shp:
        try:
            rivers = gpd.read_file(river_shp).to_crs("EPSG:4326")
            # Detect stream-order column
            order_col = next(
                (c for c in rivers.columns
                 if c.upper() in ("ORDER", "STRMORDER", "ORD_STRA", "STRAHLER",
                                  "STREAM_ORD", "ORD_FLOW", "STREAMORDE")),
                None)
            if order_col:
                max_ord  = rivers[order_col].max()
                for ord_val, grp in rivers.groupby(order_col):
                    lw = 0.3 + 2.2 * (ord_val / max_ord) ** 1.5
                    alpha = 0.4 + 0.5 * (ord_val / max_ord)
                    grp.plot(ax=ax, color="#3a86ff", linewidth=lw,
                             alpha=alpha, zorder=2)
            else:
                rivers.plot(ax=ax, color="#3a86ff", linewidth=0.5,
                            alpha=0.6, zorder=2)
            print(f"    [Plot] Rivers loaded: {river_shp.name}  ({len(rivers)} segments)")
        except Exception as e:
            print(f"    [Plot] Could not draw rivers: {e}")
    else:
        print(f"    [Plot] No river shapefile found in {task_output_dir} - rivers omitted.")

    # ------------------------------------------------------------------
    # 3. Hydropower plants
    # ------------------------------------------------------------------
    if subset is None or len(subset) == 0:
        ax.set_title(f"{task_label}\n(no quality-filtered plants)", fontsize=11)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"    [Plot] Saved (no plants): {out_png}")
        return out_png

    import numpy as np

    # Capacity colour mapping (log scale)
    cap_col = "capacity_mw" if "capacity_mw" in subset.columns else None
    cap_vals = subset[cap_col].fillna(1.0).clip(lower=0.1) if cap_col else None

    if cap_vals is not None:
        norm      = mcolors.LogNorm(vmin=cap_vals.min(), vmax=cap_vals.max())
        cmap      = cm.plasma
        colors    = [cmap(norm(v)) for v in cap_vals]
        # Point size: sqrt(MW) scaled
        sizes     = (np.sqrt(cap_vals.clip(lower=1)) * 18).clip(20, 600)
    else:
        colors    = ["#ff6b35"] * len(subset)
        sizes     = [60] * len(subset)
        norm, cmap = None, None

    # Plant-type marker & colour scheme
    #   STO  -> triangle up   ^   (reservoir)
    #   PS   -> diamond       D   (pumped storage)
    #   ROR  -> circle        o   (run-of-river)
    #   Canal-> square        s
    #   other-> circle        o
    TYPE_MARKER = {"STO": "^", "PS": "D", "CANAL": "s", "ROR": "o"}
    TYPE_ZORDER = {"STO": 6,   "PS": 7,   "CANAL": 4,   "ROR": 5}

    type_col = "type_unified" if "type_unified" in subset.columns else None
    plant_types = subset[type_col].str.upper() if type_col else \
                  pd.Series(["ROR"] * len(subset), index=subset.index)

    xs    = subset.geometry.x.values
    ys    = subset.geometry.y.values
    c_arr = colors if isinstance(colors, list) else list(colors)
    s_arr = list(sizes)
    qual  = subset["snap_quality"].values if "snap_quality" in subset.columns else \
            ["A"] * len(subset)

    for x, y, c, s, ptype, q in zip(xs, ys, c_arr, s_arr, plant_types, qual):
        marker    = TYPE_MARKER.get(ptype, "o")
        zo        = TYPE_ZORDER.get(ptype, 5)
        # PS gets a slightly larger base size to stand out
        s_final   = s * 1.3 if ptype == "PS" else s
        edgecolor = "white" if str(q).upper() == "A" else "#ffcc00"
        lw        = 0.6    if str(q).upper() == "A" else 1.4
        ax.scatter(x, y, c=[c], s=s_final, marker=marker,
                   edgecolors=edgecolor, linewidths=lw,
                   zorder=zo, alpha=0.88)

    # ------------------------------------------------------------------
    # 4. Colorbar
    # ------------------------------------------------------------------
    if norm is not None and cmap is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Capacity (MW)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # 5. Legend
    # ------------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor="#3a86ff", alpha=0.7, label="Rivers"),
        ax.scatter([], [], marker="^", c="grey", s=80,
                   edgecolors="white", linewidths=0.6, label="STO (reservoir)"),
        ax.scatter([], [], marker="D", c="grey", s=80,
                   edgecolors="white", linewidths=0.6, label="PS (pumped storage)"),
        ax.scatter([], [], marker="o", c="grey", s=80,
                   edgecolors="white", linewidths=0.6, label="ROR (run-of-river)"),
        ax.scatter([], [], marker="s", c="grey", s=80,
                   edgecolors="white", linewidths=0.6, label="Canal"),
        ax.scatter([], [], marker="o", c="grey", s=80,
                   edgecolors="#ffcc00", linewidths=1.4, label="Quality B edge"),
        mpatches.Patch(facecolor="#f0f0f0", edgecolor="#333333", label="Study area"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              framealpha=0.85, edgecolor="#cccccc")

    # ------------------------------------------------------------------
    # 5b. HydroBasins pour points overlay (if inlets_outlets.shp exists)
    # ------------------------------------------------------------------
    _io_shp = Path(task_output_dir) / "inlets_outlets.shp" if task_output_dir else None
    if _io_shp and _io_shp.exists():
        try:
            _io = gpd.read_file(_io_shp).to_crs("EPSG:4326")
            _hybas = _io[_io.get("TYPE", pd.Series(dtype=str)).str.startswith("HYBAS", na=False)]
            if len(_hybas) > 0:
                _hx = _hybas.geometry.x.values
                _hy = _hybas.geometry.y.values
                ax.scatter(_hx, _hy, marker="*", c="#ff3300", s=180,
                           edgecolors="white", linewidths=0.5, zorder=9,
                           label=f"GRFR basin outlet ({len(_hybas)})")
                # Annotate basin ID
                for _pt in _hybas.itertuples():
                    _hid = str(getattr(_pt, "HYBAS_ID", "")).replace("nan","")
                    if _hid:
                        ax.annotate(_hid[-6:],  # last 6 digits keep it short
                                    xy=(_pt.geometry.x, _pt.geometry.y),
                                    xytext=(3, 3), textcoords="offset points",
                                    fontsize=5, color="#cc0000", zorder=10)
                legend_handles.append(
                    ax.scatter([], [], marker="*", c="#ff3300", s=120,
                               edgecolors="white", linewidths=0.5,
                               label=f"GRFR basin outlet"))
                ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
                          framealpha=0.85, edgecolor="#cccccc")
        except Exception as _e:
            print(f"    [Plot] Could not draw HYBAS pour points: {_e}")

    # ------------------------------------------------------------------
    # 6. Labels for large plants (capacity > 500 MW)
    # ------------------------------------------------------------------
    if cap_col:
        name_col = "name" if "name" in subset.columns else \
                   "project_name_unified" if "project_name_unified" in subset.columns else None
        label_mask = subset[cap_col].fillna(0) >= 500
        for row in subset[label_mask].itertuples():
            label_text = str(getattr(row, name_col, ""))[:20] if name_col else ""
            if label_text:
                ax.annotate(label_text,
                            xy=(row.geometry.x, row.geometry.y),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=6, color="#222222",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      alpha=0.6, ec="none"),
                            zorder=8)

    # ------------------------------------------------------------------
    # 7. Finishing touches -- count each type for the title
    # ------------------------------------------------------------------
    type_counts = plant_types.value_counts().to_dict()
    parts = []
    for t in ["STO", "PS", "ROR", "CANAL"]:
        if type_counts.get(t, 0) > 0:
            parts.append(f"{t}: {type_counts[t]}")
    # catch any unexpected types
    for t, n in sorted(type_counts.items()):
        if t not in ("STO", "PS", "ROR", "CANAL"):
            parts.append(f"{t}: {n}")
    count_str = "  |  ".join(parts) + f"  |  total: {len(subset)}"
    ax.set_title(
        f"{task_label}\n"
        f"Rivers & Hydropower Plants  ({count_str})",
        fontsize=11, pad=10)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude",  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#aaaaaa")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    [Plot] Map saved: {out_png}")
    return out_png


def process_hydro_for_task(task, output_dir, study_shp=None, task_output_dir=None):
    """Load, filter, write hydro plant table + reservoir shapefile + overview map."""
    hydro_gdf = load_hydro_merged()
    if hydro_gdf is None:
        return

    subset = filter_hydro_for_task(hydro_gdf, task, study_shp=study_shp)
    if subset is None or len(subset) == 0:
        print(f"    [Hydro] No quality-filtered plants found for this region.")
        # Still try to draw a rivers-only map
        if task_output_dir:
            plot_rivers_and_plants(None, task_output_dir, output_dir,
                                   task_label=task.get("label", task["name"]),
                                   study_shp=study_shp)
        return

    # HydroBasins pour points - disabled (too close to plants)
    _hybas_pts = None
    _hybas_series = None

    write_hydro_table(subset, output_dir, hybas_lv_series=_hybas_series)
    write_inlets_outlets_shp(
        subset,
        task_output_dir=task_output_dir or str(Path(output_dir).parent),
        output_dir=output_dir,
        target_epsg=task.get("target_epsg", 4326),
        _precomputed_hybas=_hybas_pts,
    )

    # Generate rivers + plants overview map
    plot_rivers_and_plants(
        subset,
        task_output_dir=task_output_dir or str(Path(output_dir).parent),
        output_dir=output_dir,
        task_label=task.get("label", task["name"]),
        study_shp=study_shp,
    )


# ============================================================================
# Country / region definitions
# ============================================================================

SOUTHEAST_ASIA_MAINLAND = {
    "continent": "ea",
    "countries": [
        "Singapore", "Thailand",
        "Vietnam", "Myanmar", "Cambodia", "Laos",
    ]
}

# Philippines/Indonesia/Timor-Leste/Brunei fall under SWAT "ap" (Australia/Pacific)
# SWAT ea (Europe/Asia) soil/land-use rasters do not cover these island nations
SOUTHEAST_ASIA_MARITIME = {
    "continent": "ap",
    "countries": [
        "Indonesia", "Philippines",
    ]
}

LATIN_AMERICA = {
    "continent": "sa",
    "countries": ["Argentina", "Chile", "Colombia", 
                  "Ecuador", "Peru", "Bolivia"],
}

EUROPE_SUBREGIONS = {
    "continent": "ea",
    "subregions": {
        "europe_north":   ["Norway", "Sweden", "Finland", "Denmark"],
        "europe_west":    ["United Kingdom", "France", "Netherlands", "Belgium",
                          "Luxembourg", "Ireland"],
        "europe_central": ["Germany", "Austria", "Switzerland", "Poland",
                          "Czechia", "Hungary", "Slovakia", "Slovenia"],
        "europe_south":   ["Italy", "Spain", "Portugal", "Greece", "Croatia",
                          "Romania", "Bulgaria", "Serbia", "Montenegro",
                          "North Macedonia", "Albania"],
        "europe_baltic":  ["Estonia", "Latvia", "Lithuania"],
    },
}


# ============================================================================
# USA -- BA (Balancing Authority) subregions (read directly from shapefile)
# ============================================================================
USA_BA_SHP  = "/home/cfeng/GOES/BASHP/subregion_regions.shp"
USA_BA_ATTR = "Subregion"

# Hydro background & EPSG (optional; auto-computed for unlisted regions)
USA_BA_INFO = {
    "CAISO":              ("California ISO | Sierra Nevada hydro",             32611),
    "ERCOT":              ("Texas independent grid | minimal hydro",           32614),
    "FRCC":               ("Florida | no major hydro",                         32617),
    "ISONE":              ("New England ISO | Connecticut River small hydro",  32619),
    "MISO_Central":       ("MISO Central | Missouri/Ohio River",               32615),
    "MISO_North":         ("MISO North | Upper Mississippi",                   32615),
    "MISO_South":         ("MISO South | Lower Mississippi/Gulf",              32615),
    "NYISO":              ("New York ISO | Niagara Falls ~2.6 GW",             32618),
    "NorthernGrid_East":  ("NorthernGrid East | Missouri headwaters",          32612),
    "NorthernGrid_South": ("NorthernGrid South | Snake/Salmon River",          32611),
    "NorthernGrid_West":  ("NorthernGrid West / BPA | Columbia River ~31 GW", 32610),
    "PJM_East":           ("PJM East | world's largest data-center cluster",   32618),
    "PJM_West":           ("PJM West | Ohio River Valley",                     32617),
    "SERTP":              ("Southeast / TVA | Tennessee River ~30 dams",       32617),
    "SPP_North":          ("SPP North | Missouri/Platte River",                32614),
    "SPP_South":          ("SPP South | Arkansas/Red River",                   32615),
    "WestConnect_North":  ("WestConnect North | Upper Colorado River",         32612),
    "WestConnect_South":  ("WestConnect South | Colorado River/Hoover Dam",   32612),
}

USA_NON_CONUS = {}


# ============================================================================
# China -- 7 grid regions (Admin-1)
# ============================================================================
CHINA_REGIONS = {
    "continent": "ea",
    "regions": {
        "china_csg": (
            "Southern Grid (Guangdong/Guangxi/Yunnan/Guizhou/Hainan | Pearl/Lancang)", "China",
            ["Guangdong", "Guangxi", "Yunnan", "Guizhou", "Hainan"], 32648),
        "china_sw": (
            "Southwest Grid (Sichuan/Chongqing/Xizang | Jinsha/Yalong/Dadu)", "China",
            ["Sichuan", "Chongqing", "Xizang"], 32646),
        "china_cc": (
            "Central Grid (Henan/Hubei/Hunan/Jiangxi | Middle Yangtze/Three Gorges)", "China",
            ["Henan", "Hubei", "Hunan", "Jiangxi"], 32649),
        "china_ec": (
            "East Grid (Shanghai/Jiangsu/Zhejiang/Anhui/Fujian/Shandong | Lower Yangtze)", "China",
            ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Shandong"], 32651),
        "china_nc": (
            "North Grid (Beijing/Tianjin/Hebei/Shanxi/Nei Mongol | Haihe/Yellow R.)", "China",
            ["Beijing", "Tianjin", "Hebei", "Shanxi", "Nei Mongol"], 32650),
        "china_nw": (
            "Northwest Grid (Shaanxi/Gansu/Ningxia/Qinghai/Xinjiang | Upper Yellow R.)", "China",
            ["Shaanxi", "Gansu", "Ningxia Hui", "Qinghai", "Xinjiang Uygur"], 32645),
        "china_ne": (
            "Northeast Grid (Heilongjiang/Jilin/Liaoning | Songhua/Yalu)", "China",
            ["Heilongjiang", "Jilin", "Liaoning"], 32652),
        # Hong Kong/Macau/Taiwan are NOT admin-1 provinces of China in Natural Earth;
        # Hong Kong is handled as a standalone country task (see OTHER_COUNTRIES).
    },
}

# ============================================================================
# India -- 5 grid regions
# ============================================================================
INDIA_REGIONS = {
    "continent": "ea",
    "regions": {
        "india_north": (
            "Northern Grid (Upper Indus/Ganges)", "India",
            ["Jammu and Kashmir", "Himachal Pradesh", "Punjab", "Haryana",
             "Delhi", "Uttar Pradesh", "Uttarakhand", "Rajasthan", "Chandigarh",
             "Ladakh"], 32644),
        "india_east": (
            "Eastern Grid (Damodar/Lower Ganges)", "India",
            ["West Bengal", "Jharkhand", "Bihar", "Odisha", "Sikkim"], 32645),
        "india_northeast": (
            "Northeast Grid (Brahmaputra)", "India",
            ["Assam", "Meghalaya", "Mizoram", "Nagaland", "Tripura",
             "Manipur", "Arunachal Pradesh"], 32646),
        "india_west": (
            "Western Grid (Narmada/Tapti)", "India",
            ["Gujarat", "Maharashtra", "Goa", "Madhya Pradesh", "Chhattisgarh",
             "Dadra and Nagar Haveli and Daman and Diu"], 32643),
        "india_south": (
            "Southern Grid (Krishna/Kaveri/Godavari)", "India",
            ["Karnataka", "Kerala", "Tamil Nadu", "Andhra Pradesh", "Telangana",
             "Puducherry", "Lakshadweep", "Andaman and Nicobar"], 32644),
    },
}

# ============================================================================
# Brazil -- SIN 4 subsystems
# ============================================================================
BRAZIL_REGIONS = {
    "continent": "sa",
    "regions": {
        "brazil_norte": (
            "Norte subsystem (Amazon/Xingu/Tocantins)", "Brazil",
            ["Amazonas", "Pará", "Tocantins", "Amapá", "Roraima", "Rondônia", "Acre"], 32621),
        "brazil_nordeste": (
            "Nordeste subsystem (Sao Francisco)", "Brazil",
            ["Maranhão", "Piauí", "Ceará", "Rio Grande do Norte", "Paraíba",
             "Pernambuco", "Alagoas", "Sergipe", "Bahia"], 32624),
        "brazil_sudeste": (
            "SE-CO subsystem (Upper Parana)", "Brazil",
            ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Espírito Santo",
             "Goiás", "Mato Grosso do Sul", "Mato Grosso", "Distrito Federal"], 32623),
        "brazil_sul": (
            "Sul subsystem (Lower Parana/Itaipu)", "Brazil",
            ["Paraná", "Santa Catarina", "Rio Grande do Sul"], 32622),
    },
}

# ============================================================================
# Canada
# ============================================================================
CANADA_REGIONS = {
    "continent": "na",
    "regions": {
        "canada_bc":       ("BC (BC Hydro | Columbia/Peace River)", "Canada",
                           ["British Columbia"], 32610),
        "canada_prairies": ("Prairies (Nelson River/South Sask.)", "Canada",
                           ["Alberta", "Saskatchewan", "Manitoba"], 32612),
        "canada_quebec":   ("Quebec (Hydro-Quebec | La Grande/Churchill Falls)", "Canada",
                           ["Québec"], 32619),
        "canada_ontario":  ("Ontario (OPG | Niagara/Ottawa River)", "Canada",
                           ["Ontario"], 32617),
        "canada_atlantic": ("Atlantic (Churchill Falls)", "Canada",
                           ["New Brunswick", "Nova Scotia", "Prince Edward Island",
                            "Newfoundland and Labrador"], 32620)
    },
}

OTHER_COUNTRIES = {
    "russia":    ("Russia",                  "ea", 32637),
    "turkey":    ("Turkey",                  "ea", 32636),
    # Natural Earth Admin-0 sovereign name for HK SAR
    "hong_kong": ("Hong Kong",        "ea", 32650),
}

EUROPE_EPSG = {
    "Norway": 32632, "Sweden": 32633, "Finland": 32635, "Denmark": 32632,
    "United Kingdom": 32630, "France": 32631, "Netherlands": 32631,
    "Belgium": 32631, "Luxembourg": 32631, "Ireland": 32629,
    "Germany": 32632, "Austria": 32633, "Switzerland": 32632, "Poland": 32634,
    "Czechia": 32633, "Czech Republic": 32633,  # both kept for safety
    "Hungary": 32634, "Slovakia": 32634, "Slovenia": 32633,
    "Italy": 32632, "Spain": 32630, "Portugal": 32629, "Greece": 32634,
    "Croatia": 32633, "Romania": 32635, "Bulgaria": 32635,
    "Serbia": 32634, "Montenegro": 32634, "North Macedonia": 32634,
    "Albania": 32634, "Bosnia and Herzegovina": 32633,
    "Ukraine": 32636, "Moldova": 32635, "Belarus": 32635,
    "Estonia": 32635, "Latvia": 32634, "Lithuania": 32634,
    "Malta": 32633, "Cyprus": 32636, "Kosovo": 32634,
}


# ============================================================================
# Task list builder
# ============================================================================
def _build_usa_ba_tasks():
    """Auto-generate US CONUS tasks from BA shapefile."""
    tasks    = []
    shp_path = Path(USA_BA_SHP)

    if not shp_path.exists():
        print(f"  [Warning] BA shapefile not found: {shp_path}  (US CONUS will be skipped)")
        return tasks

    gdf      = gpd.read_file(shp_path)
    attr_col = USA_BA_ATTR

    if attr_col not in gdf.columns:
        candidates = [c for c in gdf.columns if 'subregion' in c.lower() or 'region' in c.lower()]
        if candidates:
            attr_col = candidates[0]
            print(f"  [Warning] Column '{USA_BA_ATTR}' not found; using '{attr_col}'")
        else:
            print(f"  [Error] Shapefile columns: {list(gdf.columns)} -- cannot identify region field")
            return tasks

    subregions = sorted(gdf[attr_col].dropna().unique())
    print(f"  [BA] Loaded {len(subregions)} subregions from shapefile")

    for sr_name in subregions:
        if sr_name in USA_BA_INFO:
            info_label, epsg = USA_BA_INFO[sr_name]
        else:
            sr_geom  = gdf[gdf[attr_col] == sr_name].dissolve()
            centroid = sr_geom.to_crs("EPSG:4326").geometry.centroid.iloc[0]
            epsg     = auto_utm_epsg(centroid.x, centroid.y)
            info_label = sr_name

        safe_name = sr_name.lower().replace(" ", "_").replace("-", "_")
        tasks.append({
            "name": f"usa_{safe_name}",
            "folder": f"usa/usa_{safe_name}",
            "type": "ba_shapefile",
            "ba_shp": str(shp_path), "ba_attr": attr_col, "ba_value": sr_name,
            "country": None, "admin_country": None, "admin_provs": None,
            "continent": "na", "target_epsg": epsg,
            "label": f"USA-BA - {sr_name} ({info_label})",
        })

    # Alaska & Hawaii
    for key, (label, admin_country, provs, epsg) in USA_NON_CONUS.items():
        tasks.append({
            "name": key, "folder": f"usa/{key}",
            "type": "admin", "country": None,
            "admin_country": admin_country, "admin_provs": provs,
            "ba_shp": None, "ba_attr": None, "ba_value": None,
            "continent": "na", "target_epsg": epsg,
            "label": f"USA - {label}",
        })
    return tasks


def build_task_list():
    tasks  = []
    _empty = {"ba_shp": None, "ba_attr": None, "ba_value": None,
              "admin_country": None, "admin_provs": None}

    # --- Southeast Asia mainland (SWAT continent = ea) ---
    sea_epsg = {"Singapore": 32648, "Malaysia": 32648, "Thailand": 32647,
                "Vietnam": 32648, "Myanmar": 32646, "Cambodia": 32648, "Laos": 32648}
    for c in SOUTHEAST_ASIA_MAINLAND["countries"]:
        tasks.append({
            "name":   c.lower().replace(" ", "_").replace("-", "_"),
            "folder": f"southeast_asia/{c.lower().replace(' ', '_').replace('-', '_')}",
            "type": "country", "country": c, **_empty,
            "continent": "ea", "target_epsg": sea_epsg.get(c, 32648),
            "label": c,
        })

    # --- Southeast Asia maritime (SWAT continent = ap) ---
    # See roughly lines 1593-1601
    sea_maritime_epsg = {"Indonesia": 32748, "Philippines": 32651,
                        "Brunei": 32650, "Timor-Leste": 32652}
    # Countries that need dual-continent coverage
    DUAL_CONTINENT = {"Philippines": ["ap", "ea"]}
    for c in SOUTHEAST_ASIA_MARITIME["countries"]:
        cont = DUAL_CONTINENT.get(c, "ap")  # default "ap"; Philippines returns ["ap", "ea"]
        tasks.append({
            "name":   c.lower().replace(" ", "_").replace("-", "_"),
            "folder": f"southeast_asia/{c.lower().replace(' ', '_').replace('-', '_')}",
            "type": "country", "country": c, **_empty,
            "continent": cont, "target_epsg": sea_maritime_epsg.get(c, 32648),
            "label": c,
        })

    # Malaysia: split into West (ea) and East (ap)
    tasks.append({
        "name": "malaysia_west", "folder": "southeast_asia/malaysia_west",
        "type": "admin", "country": None, **_empty,
        "admin_country": "Malaysia", "admin_provs": [
            "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
            "Pahang", "Perak", "Perlis", "Pulau Pinang", "Selangor",
            "Terengganu", "Kuala Lumpur", "Putrajaya"],
        "continent": "ea", "target_epsg": 32648, "label": "Malaysia West (Peninsular)",
    })
    tasks.append({
        "name": "malaysia_east", "folder": "southeast_asia/malaysia_east",
        "type": "admin", "country": None, **_empty,
        "admin_country": "Malaysia", "admin_provs": ["Sabah", "Sarawak", "Labuan"],
        "continent": "ap", "target_epsg": 32650, "label": "Malaysia East (Sabah/Sarawak)",
    })

    # --- Latin America ---
    epsg_latam = {"Argentina": 32720, "Chile": 32719, "Colombia": 32618}
    for c in LATIN_AMERICA["countries"]:
        tasks.append({
            "name": c.lower(), "folder": f"latin_america/{c.lower()}",
            "type": "country", "country": c, **_empty,
            "continent": "sa", "target_epsg": epsg_latam.get(c, 32720),
            "label": c,
        })

    # --- Europe ---
    for subregion, countries in EUROPE_SUBREGIONS["subregions"].items():
        for c in countries:
            tasks.append({
                "name":   c.lower().replace(" ", "_"),
                "folder": f"{subregion}/{c.lower().replace(' ', '_')}",
                "type": "country", "country": c, **_empty,
                "continent": "ea", "target_epsg": EUROPE_EPSG.get(c, 32632),
                "label": f"{c} ({subregion})",
            })

    # --- USA (BA shapefile + AK/HI) ---
    tasks.extend(_build_usa_ba_tasks())

    # --- China / Canada / India / Brazil (Admin-1) ---
    admin_configs = [
        ("china",  CHINA_REGIONS),  ("canada", CANADA_REGIONS),
        ("india",  INDIA_REGIONS),  ("brazil", BRAZIL_REGIONS),
    ]
    for country_key, config in admin_configs:
        continent = config["continent"]
        for key, (label, admin_country, provs, epsg) in config["regions"].items():
            tasks.append({
                "name": key, "folder": f"{country_key}/{key}",
                "type": "admin", "country": None,
                "admin_country": admin_country, "admin_provs": provs,
                "ba_shp": None, "ba_attr": None, "ba_value": None,
                "continent": continent, "target_epsg": epsg,
                "label": f"{country_key.upper()} - {label}",
            })

    # --- Russia / Turkey ---
    for key, (cname, cont, epsg) in OTHER_COUNTRIES.items():
        tasks.append({
            "name": key, "folder": f"other/{key}",
            "type": "country", "country": cname, **_empty,
            "continent": cont, "target_epsg": epsg,
            "label": cname,
        })

    return tasks


# ============================================================================
# Adaptive DEM resolution
# ============================================================================

# Approximate area (km²) for country-type tasks.
# Only countries defined here get area-based scaling; others use base resolution.
_COUNTRY_AREA_KM2 = {
    # Southeast Asia mainland
    "singapore": 730,        "thailand": 513_000,   "vietnam": 331_000,
    "myanmar": 677_000,      "cambodia": 181_000,   "laos": 237_000,
    # Southeast Asia maritime
    "indonesia": 1_905_000,  "philippines": 300_000,
    # Latin America
    "argentina": 2_780_000,  "chile": 756_000,      "colombia": 1_142_000,
    "venezuela": 916_000,    "ecuador": 284_000,    "peru": 1_285_000,
    "bolivia": 1_099_000,
    # Europe (all comfortably below 1M km², stay at base)
    "norway": 385_000,   "sweden": 450_000,  "finland": 338_000,
    "france": 551_000,   "spain": 506_000,   "germany": 357_000,
    "ukraine": 604_000,  "poland": 313_000,  "italy": 301_000,
    "turkey": 784_000,
    # Other
    "russia": 17_098_000,  "hong_kong": 1_100,
}


def compute_area_km2(shp_path):
    """Return area of a shapefile in km² (Mollweide equal-area projection)."""
    try:
        gdf = gpd.read_file(shp_path).to_crs("ESRI:54009")
        return float(gdf.geometry.area.sum()) / 1e6
    except Exception as e:
        print(f"    [Area] Could not compute area from {shp_path}: {e}")
        return None


def adaptive_dem_resolution(area_km2, base_resolution=500):
    """
    Return an appropriate DEM resolution (metres) for the given study area.

    Thresholds are conservative: the step-up only happens when the region is
    large enough that the *base* resolution would produce >~4 M DEM cells.
    A region the size of Thailand (~513 000 km²) stays at 500 m (~2 M cells).

    Breakpoints (at base=500 m):
      area ≤  1 000 000 km²  →  500 m   (~4.0 M cells max)
      area ≤  2 500 000 km²  →  750 m   (~4.4 M cells max)
      area ≤  5 000 000 km²  → 1000 m   (~5.0 M cells max)
      area ≤ 10 000 000 km²  → 1500 m   (~4.4 M cells max)
      area > 10 000 000 km²  → 2000 m
    """
    if area_km2 is None:
        return base_resolution

    STEPS = [
        (1_000_000,  base_resolution),
        (2_500_000,  max(base_resolution, 750)),
        (5_000_000,  max(base_resolution, 1000)),
        (10_000_000, max(base_resolution, 1500)),
    ]
    for max_area, res in STEPS:
        if area_km2 <= max_area:
            return res
    return max(base_resolution, 2000)


def get_task_area_km2(task, study_shp=None):
    """
    Estimate study-area size in km² for a task.
    Priority: (1) study_shp computed precisely; (2) lookup table by task name.
    """
    if study_shp and Path(study_shp).exists():
        area = compute_area_km2(study_shp)
        if area is not None:
            return area
    return _COUNTRY_AREA_KM2.get(task["name"], None)


# ============================================================================
# Cleanup & execution
# ============================================================================
def cleanup_intermediate(output_dir):
    output_dir = Path(output_dir)
    for subdir in ["intermediate", "raw_downloads"]:
        d = output_dir / subdir
        if d.exists():
            shutil.rmtree(d)


def run_batch(base_dir, only=None, skip_existing=False, dry_run=False,
              dem_resolution=500, river_detail=4, basin_level=12,
              max_river_dist_km=2.0, cluster_dist_km=10.0,
              min_lake_area_km2=10, min_basin_area_km2=50,
              admin1_path=None):

    tasks = build_task_list()

    if only:
        only_set = {s.strip().lower().replace(" ", "_").replace("-", "_")
                    for s in only.split(",")}
        tasks = [t for t in tasks
                 if t["name"] in only_set
                 or any(k in t["folder"].lower() for k in only_set)
                 or any(k in t["label"].lower() for k in only_set)]

    base_dir = Path(base_dir)

    if skip_existing:
        tasks = [t for t in tasks
                 if not (base_dir / t["folder"] / "rasters" / "dem.tif").exists()]

    total      = len(tasks)
    task_types = {}
    for t in tasks:
        task_types[t["type"]] = task_types.get(t["type"], 0) + 1

    print("=" * 70)
    print(f"SWAT+ Multi-Country Batch Data Preparation")
    print(f"   Total tasks : {total}  |  Types: {task_types}")
    print(f"   Output dir  : {base_dir}/")
    print(f"   DEM: {dem_resolution} m  |  Rivers: detail={river_detail}  |  Basins: level={basin_level}")
    print(f"   Hydro Table: {HYDRO_TABLE_PATH}")
    print("=" * 70)

    if dry_run:
        for i, t in enumerate(tasks):
            icons = {"country": "[country]", "admin": "[admin]", "ba_shapefile": "[BA]"}
            icon  = icons.get(t["type"], "[?]")
            print(f"  [{i+1:3d}/{total}] {icon} {t['folder']:40s}  {t['label']}")
            if t["type"] == "admin" and t["admin_provs"]:
                print(f"           |-- {t['admin_country']}: {t['admin_provs']}")
            elif t["type"] == "ba_shapefile":
                print(f"           |-- BA: {t['ba_attr']}='{t['ba_value']}'")
        print(f"\n  --dry-run mode, nothing executed.  {total} tasks total.")
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from swat_data_prep import prepare_swat_data
    import swat_data_prep
    swat_data_prep.VERBOSE = False  # suppress detailed per-step messages in batch mode

    admin1_gdf = None
    if any(t["type"] == "admin" for t in tasks):
        print(f"\n[Admin1] Loading administrative boundary data...")
        admin1_gdf = load_admin1(admin1_path)
        print(f"   Records: {len(admin1_gdf)}, Countries: {admin1_gdf['admin'].nunique()}")

    # Pre-load hydro tables once (result is cached internally)
    print(f"\n[Hydro] Pre-loading hydropower tables...")
    load_hydro_merged()

    results   = {"success": [], "failed": [], "skipped": []}
    start_all = time.time()

    for i, task in enumerate(tasks):
        output_dir = base_dir / task["folder"]
        elapsed    = time.time() - start_all
        eta_str    = str(timedelta(seconds=int(elapsed / i * (total - i)))) if i > 0 else "calculating..."

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{total}] {task['label']}  |  ETA: {eta_str}")
        print(f"{'='*70}")

        try:
            study_shp = None
            hydro_filter_shp = None  # <--- [new] original-boundary variable used specifically for filtering plants

            # 1. Build the original study area (BA or Admin)
            if task["type"] == "ba_shapefile":
                study_shp = output_dir / "_study_area_ba.shp"
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [BA] Region: {task['ba_value']}")
                build_study_area_from_shapefile(
                    task["ba_shp"], task["ba_attr"], task["ba_value"], study_shp)
                hydro_filter_shp = study_shp  # <--- record the original boundary

            elif task["type"] == "admin":
                study_shp = output_dir / "_study_area_admin.shp"
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [Admin] {task['admin_country']} -> {len(task['admin_provs'])} provinces/states")
                build_study_area_from_admin1(
                    admin1_gdf, task["admin_country"], task["admin_provs"], study_shp)
                hydro_filter_shp = study_shp  # <--- record the original boundary

            # ── Upstream expansion: expand study area to include full ────
            #    upstream watersheds of all hydropower plants in this region
            hydro_gdf = load_hydro_merged()
            hydro_subset = None
            if hydro_gdf is not None:
                # Preliminary filter: use spatial filtering if study_shp exists, otherwise filter by country name
                hydro_subset = filter_hydro_for_task(
                    hydro_gdf, task,
                    study_shp=str(study_shp) if study_shp else None)

            if hydro_subset is not None and len(hydro_subset) > 0 and \
               "GRFR_COMID" in hydro_subset.columns:
                
                # Load original study area for expansion
                if study_shp and Path(study_shp).exists():
                    orig_sa = gpd.read_file(study_shp).to_crs("EPSG:4326")
                elif task["type"] == "country" and task["country"]:
                    from swat_data_prep import get_country_boundary
                    orig_sa = get_country_boundary(task["country"]).to_crs("EPSG:4326")
                    orig_sa = clip_to_mainland(orig_sa, task["country"])
                    
                    # [key change] For Country tasks, save the original boundary before expansion for filtering
                    if hydro_filter_shp is None:
                        target_shp = output_dir / "_study_area_target.shp"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        orig_sa.to_file(target_shp)
                        hydro_filter_shp = target_shp
                else:
                    orig_sa = None

                if orig_sa is not None:
                    expanded_sa, area_before, area_after = \
                        expand_study_area_with_upstream(
                            orig_sa, hydro_subset,
                            hydrobasins_dir=GRFR_DIR,
                            hydrorivers_shp=None)

                    # Save expanded study area as the new study_shp
                    expanded_shp = output_dir / "_study_area_expanded.shp"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    expanded_sa.to_file(expanded_shp)
                    
                    # Update study_shp for DEM processing and hydrological modeling (needs the larger extent)
                    study_shp = expanded_shp
                    print(f"  [Expand] Saved expanded study area: {expanded_shp}")

            # ── Adaptive DEM resolution ──────────────────────────────────────
            # Note: here we use study_shp (possibly the expanded one) to compute the resolution, which is correct,
            # because the DEM must cover the entire modeling region.
            area_km2 = get_task_area_km2(
                task, study_shp=str(study_shp) if study_shp else None)
            task_res = adaptive_dem_resolution(area_km2, base_resolution=dem_resolution)
            if area_km2 is not None:
                note = f"  (base {dem_resolution} m)" if task_res != dem_resolution else ""
                print(f"  [DEM] Area: {area_km2:,.0f} km²  →  resolution: {task_res} m{note}")
            else:
                print(f"  [DEM] Area unknown, using base resolution: {task_res} m")

            kwargs = dict(
                dem_resolution=task_res, river_detail=river_detail,
                grfr_dir=GRFR_DIR, continent=task["continent"],
                target_epsg=task["target_epsg"], output_dir=str(output_dir),
                max_river_dist_km=max_river_dist_km, cluster_dist_km=cluster_dist_km,
                min_lake_area_km2=min_lake_area_km2, min_basin_area_km2=min_basin_area_km2,
            )

            # Always use study_area_shp (may be expanded) for model generation
            if study_shp and Path(study_shp).exists():
                kwargs.update(study_area_shp=str(study_shp), country=None, bbox=None)
            elif task["type"] == "country":
                cname = task["country"]
                if cname in MAINLAND_BBOX:
                    from swat_data_prep import get_country_boundary
                    _gdf = clip_to_mainland(get_country_boundary(cname).to_crs("EPSG:4326"), cname)
                    _shp = output_dir / "_study_area_mainland.shp"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    _gdf.to_file(_shp)
                    kwargs.update(study_area_shp=str(_shp), country=None, bbox=None)
                else:
                    kwargs.update(country=cname, bbox=None, study_area_shp=None)
            else:
                kwargs.update(study_area_shp=str(study_shp) if study_shp else None,
                              country=None, bbox=None)

            prepare_swat_data(**kwargs)
            cleanup_intermediate(output_dir)

            # Write hydropower plant table + QSWAT+ reservoir shapefile + overview map
            hydro_out = output_dir / "hydro"
            
            # [key change] Use hydro_filter_shp (original boundary) instead of study_shp (expanded boundary)
            # so the generated inlets_outlets.shp only contains plants within the original study area
            process_hydro_for_task(
                task, hydro_out,
                study_shp=str(hydro_filter_shp) if hydro_filter_shp else None,
                task_output_dir=str(output_dir),
            )

            results["success"].append(task["label"])
            print(f"\n  [OK] {task['label']} completed!")

        except Exception as e:
            results["failed"].append((task["label"], str(e)))
            print(f"\n  [FAIL] {task['label']}: {e}")
            traceback.print_exc()

    elapsed_total = time.time() - start_all
    print(f"\n{'='*70}")
    print(f"Batch processing complete!  Total time: {timedelta(seconds=int(elapsed_total))}")
    print(f"   Succeeded: {len(results['success'])}, Failed: {len(results['failed'])}")
    print(f"{'='*70}")

    if results["failed"]:
        print(f"\n  Failed tasks:")
        for name, err in results["failed"]:
            print(f"    - {name}: {err[:80]}")

    log_path = base_dir / "batch_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":      datetime.now().isoformat(),
            "total_time_sec": int(elapsed_total),
            "success":        results["success"],
            "failed":         [{"name": n, "error": e} for n, e in results["failed"]],
        }, f, ensure_ascii=False, indent=2)
    print(f"  Log: {log_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SWAT+ Multi-Country Batch Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_countries.py --base-dir ./swat_global --dry-run
  python batch_countries.py --base-dir ./swat_global --only "caiso"
  python batch_countries.py --base-dir ./swat_global --only "northerngrid_west"
  python batch_countries.py --base-dir ./swat_global --only "china_csg"
  python batch_countries.py --base-dir ./swat_global --skip-existing
        """)
    parser.add_argument("--base-dir",       type=str,   default="./swat_global")
    parser.add_argument("--only",           type=str,   default=None)
    parser.add_argument("--skip-existing",  action="store_true")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--admin1-path",    type=str,   default=None)
    parser.add_argument("--dem-resolution", type=int,   default=500)
    parser.add_argument("--river-detail",   type=int,   default=5)
    parser.add_argument("--basin-level",    type=int,   default=12)
    parser.add_argument("--max-river-dist", type=float, default=1.0)
    parser.add_argument("--cluster-dist",   type=float, default=20.0)
    parser.add_argument("--min-lake-area",  type=float, default=500)
    parser.add_argument("--min-basin-area", type=float, default=0.01)

    args = parser.parse_args()
    run_batch(
        base_dir=args.base_dir, only=args.only,
        skip_existing=args.skip_existing, dry_run=args.dry_run,
        dem_resolution=args.dem_resolution, river_detail=args.river_detail,
        basin_level=args.basin_level, max_river_dist_km=args.max_river_dist,
        cluster_dist_km=args.cluster_dist, min_lake_area_km2=args.min_lake_area,
        min_basin_area_km2=args.min_basin_area, admin1_path=args.admin1_path,
    )
