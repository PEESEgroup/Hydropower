#!/usr/bin/env python3
"""
SWAT+ × GRFR ML data preparation automation pipeline
=====================================
Features:
  1. Enter the specified region folder (/datasets/swat_global/<sub_region>)
  2. Run swatplus in TxtInOut
  3. Prepare hourly training data (includes channel-file prefiltering to save memory)
  4. Randomly sample months to split Train/Val/Test, save to the region's datasets subfolder

Usage:
  python run_workflow.py --region brazil/brazil_norte
  python run_workflow.py --region europe_west/france --skip-swat
  python run_workflow.py --region brazil/brazil_norte --val-months 6 --test-months 6
  python run_workflow.py --region brazil/brazil_norte --seed 42
"""

import argparse
import gc
import json
import subprocess
import sys
import shutil
from pathlib import Path

# Lazy import of scientific libraries (so --help has no dependencies)
np = pd = xr = PchipInterpolator = None

def _lazy_imports():
    global np, pd, xr, PchipInterpolator
    if np is not None:
        return
    import numpy as _np
    import pandas as _pd
    import xarray as _xr
    from scipy.interpolate import PchipInterpolator as _Pchip
    import warnings
    warnings.filterwarnings('ignore')
    np, pd, xr, PchipInterpolator = _np, _pd, _xr, _Pchip


# ============================================================
# Global constants
# ============================================================
BASE_DIR = Path("/datasets/swat_global")
GRFR_DIR = Path("/datasets/GRFR")

# Time configuration
DATA_START  = "2009-01-01"
DATA_END    = "2019-12-31"
GRFR_YEARS  = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]

# Dataset split: random month sampling
# Default 2015-2019 totals 60 months:
#   randomly sample 6 months → val
#   then randomly sample 6 more months → test
#   remaining 48 months → train
VAL_MONTHS  = 6    # how many months to randomly sample for validation
TEST_MONTHS = 6    # how many months to randomly sample for test, the rest for train
SPLIT_SEED  = 42   # random seed for reproducibility

# Feature configuration
DAILY_FIELDS = ['flo_in', 'flo_stor', 'precip', 'evap']
LSUNIT_WB_FIELDS = ['surq_gen', 'latq', 'sw_ave', 'perc', 'et', 'pet']
AQUIFER_FIELDS = ['flo', 'stor', 'rchrg', 'dep_wt']
LAG_HOURS = [1, 3, 6, 12, 24]
PRECIP_ROLL_HOURS = [6, 12, 24, 72, 168]
FLOW_THRESHOLD = 0.1  # m³/s
GRFR_PFAF_OVERRIDE = None
SPLIT_MODE = 'contiguous'
KEEP_SWAT_OUTPUT = False

# ============================================================
# Path derivation
# ============================================================
def resolve_paths(region: str):
    """Derive all paths from the subfolder."""
    region_path = BASE_DIR / region
    area_name = Path(region).name

    # TxtInOut: <region>/<area_name>/Scenarios/Default/TxtInOut
    txtinout = region_path / area_name / "Scenarios" / "Default" / "TxtInOut"
    river_table = region_path / "hydro" / "station_channel_result.csv"
    output_dir = region_path / "datasets"

    paths = {
        'region_path': region_path,
        'area_name': area_name,
        'txtinout': txtinout,
        'river_table': river_table,
        'output_dir': output_dir,
        'swat_day_file': txtinout / "channel_sd_day.txt",
        'swat_subday_file': txtinout / "channel_sd_subday.txt",
        'lsunit_wb_file': txtinout / "lsunit_wb_day.txt",
        'aquifer_file': txtinout / "aquifer_day.txt",
    }

    print("=" * 60)
    print("Path resolution")
    print("=" * 60)
    for k, v in paths.items():
        exists = "✅" if isinstance(v, Path) and v.exists() else ("❌" if isinstance(v, Path) else "")
        print(f"  {k:20s}: {v}  {exists}")
    print()
    return paths


# ============================================================
# GRFR pfaf: zero-pad the first digit of COMID
# ============================================================
def detect_grfr_pfaf(comids: list) -> str:
    """
    First COMID digit = Pfafstetter Level 1 continent code; zero-padding it gives the pfaf number.
    6xxxxxxx → "06" (South America), 2xxxxxxx → "02" (Europe), etc.
    """
    first_comid = str(comids[0])
    pfaf = f"0{first_comid[0]}"
    test_file = GRFR_DIR / f"output_pfaf_{pfaf}_{GRFR_YEARS[0]}.nc"

    if test_file.exists():
        print(f"✅ GRFR pfaf = {pfaf} (COMID {first_comid})")
        return pfaf

    raise FileNotFoundError(
        f"GRFR file does not exist: {test_file}\nPlease check the COMID or manually specify --grfr-pfaf"
    )


# ============================================================
# Step 0: edit print.prt to ensure required outputs are enabled
# ============================================================
def modify_print_prt(txtinout: Path):
    """Edit print.prt to ensure daily output for lsunit_wb, channel_sd, aquifer is y"""
    prt_file = txtinout / "print.prt"
    if not prt_file.exists():
        print(f"⚠️  print.prt does not exist: {prt_file}")
        return

    print("--- Check/edit print.prt ---")
    with open(prt_file, 'r') as f:
        lines = f.readlines()

    # Objects that need daily output enabled
    targets = {'lsunit_wb', 'channel_sd', 'aquifer'}
    modified = False

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 5 and parts[0] in targets:
            if parts[1] != 'y':
                # Reformat: object name left-aligned in 23 chars, with enough trailing spaces to keep columns aligned
                lines[i] = f"{parts[0]:<23} y           {parts[2]:<11} {parts[3]:<11} {parts[4]}\n"
                print(f"  ✅ {parts[0]}: daily → y (alignment preserved)")
                modified = True
            else:
                print(f"  ✅ {parts[0]}: daily already enabled")

    if modified:
        with open(prt_file, 'w') as f:
            f.writelines(lines)
        print("  print.prt updated with formatting preserved")
    else:
        print("  print.prt needs no changes")
    print()


# ============================================================
# Step 1: run SWAT+
# ============================================================
def run_swatplus(txtinout: Path):
    print("=" * 60)
    print("Step 1: run SWAT+")
    print("=" * 60)

    if not txtinout.exists():
        print(f"❌ TxtInOut directory does not exist: {txtinout}")
        sys.exit(1)

    exe_names = ["swatplus", "swatplus+", "swat_plus", "swat+"]
    exe_found = None
    for name in exe_names:
        local_exe = txtinout / name
        if local_exe.exists():
            exe_found = str(local_exe)
            break
        result = subprocess.run(["which", name], capture_output=True, text=True)
        if result.returncode == 0:
            exe_found = result.stdout.strip()
            break

    if exe_found is None:
        exe_found = "swatplus"
        print(f"⚠️  Executable not found, trying: {exe_found}")

    print(f"Working directory: {txtinout}")
    print(f"Executable: {exe_found}")
    print("-" * 60)

    proc = subprocess.run([exe_found], cwd=str(txtinout), capture_output=False)

    if proc.returncode == -6:
        print(f"⚠️  SWAT+ return code -6, switching to swatplus_mine and retrying...")
        print("-" * 60)
        fallback_exe = "swatplus_mine"
        # Prefer the local copy in the TxtInOut directory
        local_fallback = txtinout / fallback_exe
        if local_fallback.exists():
            fallback_exe = str(local_fallback)
        print(f"Executable: {fallback_exe}")
        proc = subprocess.run([fallback_exe], cwd=str(txtinout), capture_output=False)
        if proc.returncode != 0:
            print(f"⚠️  swatplus_mine exit code: {proc.returncode}")
        else:
            print("✅ swatplus_mine finished")
    elif proc.returncode != 0:
        print(f"⚠️  SWAT+ exit code: {proc.returncode}")
    else:
        print("✅ SWAT+ finished")
    print()


# ============================================================
# Prefilter channel file
# ============================================================
def prefilter_channel_file(filepath: Path, target_channels: list):
    """Extract only the target reaches from a large channel txt, saving in place.

    Streaming line-by-line read; memory use equals only the size of the filtered
    result and is independent of the original file size. Even a 174GB file finishes
    in a few minutes, with peak memory < 1GB.
    """
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  {filepath.name}: {file_size_mb:.0f} MB", end=" → ")
    sys.stdout.flush()

    target_set = set(str(ch) for ch in target_channels)

    # Read the first 3 header lines and locate the gis_id column
    with open(filepath, 'r') as f:
        header_lines = [f.readline() for _ in range(3)]

    # Line 2 (index 1) is the column-name row
    col_names = header_lines[1].split()
    try:
        gis_id_idx = col_names.index('gis_id')
    except ValueError:
        print(f"⚠️  gis_id column not found, column names: {col_names[:10]}")
        return

    # Streaming filter: read line by line, write only matching rows
    tmp_path = filepath.with_suffix('.tmp_filtered')
    n_total = 0
    n_kept = 0

    with open(filepath, 'r') as fin, open(tmp_path, 'w') as fout:
        # Write the 3 header lines
        for line in header_lines:
            fout.write(line)

        # Skip the header already read
        for _ in range(3):
            fin.readline()

        # Process line by line
        for line in fin:
            n_total += 1
            parts = line.split()
            if len(parts) > gis_id_idx:
                gis_id = parts[gis_id_idx]
                if gis_id in target_set:
                    fout.write(line)
                    n_kept += 1

            # Progress (print every 10 million lines)
            if n_total % 10_000_000 == 0:
                print(f"\r  {filepath.name}: {file_size_mb:.0f} MB → "
                      f"scanned {n_total:,} rows, kept {n_kept:,}", end="")
                sys.stdout.flush()

    if n_kept == 0:
        print(f"\n  ⚠️  Filter result is 0 rows, skipping (protecting original file!)")
        tmp_path.unlink(missing_ok=True)
        return

    shutil.move(str(tmp_path), str(filepath))
    new_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\r  {filepath.name}: {file_size_mb:.0f} MB → "
          f"{new_mb:.0f} MB ({n_total:,} → {n_kept:,} rows)")
    sys.stdout.flush()


# ============================================================
# Look up channel → LSU / Aquifer mapping from the QSWAT+ database
# ============================================================
def find_qswat_database(region_path: Path, area_name: str):
    """Look for the QSWAT+ sqlite database file in the DatabaseBackups directory"""
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
        print(f"⚠️  No .sqlite file in DatabaseBackups")
        return None

    db_path = sqlite_files[0]
    print(f"  Database: {db_path.name}")
    return db_path


def build_channel_mapping(region_path: Path, area_name: str, target_channels: list, txtinout: Path):
    """
    Build from the QSWAT+ database:
      channel_id → {'lsu_ids': [...], 'aquifer_ids': [...]}
    Unmatched channels are matched to the nearest by lat/lon via chandeg.con / rout_unit.con.
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

    # --- Unmatched channels: nearest lat/lon matching ---
    if unmapped_channels:
        print(f"  Attempting nearest lat/lon match for {len(unmapped_channels)} unmapped reaches...")

        # Read chandeg.con
        chandeg_file = txtinout / "chandeg.con"
        rtu_file = txtinout / "rout_unit.con"

        if chandeg_file.exists() and rtu_file.exists():
            # Read rout_unit.con - parse line by line, no read_csv
            rtu_records = []
            rtu_aqu_map = {}
            with open(rtu_file, 'r') as f:
                f.readline()  # skip title/comment line
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
                    # Find the aqu keyword
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
                f.readline()  # skip title/comment line
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

                # Use the database to find the aquifer for this LSU
                lsu_row = lsus[lsus['id'] == nearest_lsu_id]
                if len(lsu_row) > 0:
                    sub_ids = lsu_row['subbasin'].unique().tolist()
                    aqu_ids = aquifers[aquifers['subbasin'].isin(sub_ids)]['id'].tolist()
                else:
                    aqu_ids = []

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
            print(f"    ❌ chandeg.con or rout_unit.con does not exist, cannot do nearest matching")

    # Print summary
    still_unmapped = [ch for ch in target_channels if ch not in mapping]
    print(f"\n  Mapping result: {len(mapping)}/{len(target_channels)} reaches")
    for ch in target_channels:
        if ch in mapping:
            m = mapping[ch]
            print(f"    Channel {ch} → LSU {m['lsu_ids']} → Aquifer {m['aquifer_ids']}  [{m['method']}]")
    if still_unmapped:
        print(f"  ❌ Still unmapped: {still_unmapped}")

    return mapping


# ============================================================
# Prefilter lsunit_wb / aquifer files (similar to the channel file)
# ============================================================
def prefilter_swat_output_file(filepath: Path, target_ids: list, name_prefix: str):
    """
    Filter a large SWAT+ output txt by the name column.
    Keep only rows whose name has the format {prefix}{digits only} (e.g. aqu0010, rtu00010).
    Streaming write; rows are not accumulated in memory.
    """
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return

    target_ids = set(target_ids)
    prefix_len = len(name_prefix)

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  {filepath.name}: {file_size_mb:.0f} MB", end=" → ")
    sys.stdout.flush()

    with open(filepath, 'r') as f:
        header_lines = [f.readline() for _ in range(3)]

    tmp_path = filepath.with_suffix('.tmp_filtered')
    n_before = 0
    n_after = 0

    with open(filepath, 'r') as fin, open(tmp_path, 'w') as fout:
        for h in header_lines:
            fout.write(h)
        for _ in range(3):
            fin.readline()
        for line in fin:
            n_before += 1
            parts = line.split()
            if len(parts) < 7:
                continue
            name = parts[6]
            if not name.startswith(name_prefix):
                continue
            suffix = name[prefix_len:]
            if not suffix.isdigit():
                continue
            parsed_id = int(suffix)
            if parsed_id in target_ids:
                fout.write(line)
                n_after += 1

    if n_after == 0:
        print(f"⚠️  Filter result is 0 rows, skipping (protecting original file!)")
        tmp_path.unlink(missing_ok=True)
        return

    shutil.move(str(tmp_path), str(filepath))
    new_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"{new_mb:.0f} MB ({n_before:,} → {n_after:,} rows)")


# ============================================================
# Read lsunit_wb_day.txt
# ============================================================
def read_lsunit_wb(filepath, target_lsu_ids, fields):
    """Read lsunit_wb_day.txt, matching by id parsed from the name column (rtuXXXXX)"""
    print("Reading lsunit_wb_day.txt ...")
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return None

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.0f} MB")

    target_set = set(target_lsu_ids)

    if file_size_mb > 5000:
        print(f"  ⚠️  File too large, using chunked read...")
        chunks = []
        for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+',
                                  engine='c', chunksize=2_000_000):
            chunk = chunk[chunk['name'].str.startswith('rtu')].copy()
            if len(chunk) == 0:
                continue
            chunk['lsu_id'] = chunk['name'].astype(str).str[3:].astype(int)
            filtered = chunk[chunk['lsu_id'].isin(target_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            print(f"  ⚠️  No matching data")
            return None
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
    else:
        df = pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c')
        df = df[df['name'].str.startswith('rtu')].copy()
        df['lsu_id'] = df['name'].astype(str).str[3:].astype(int)

    print(f"  Shape: {df.shape}")

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))

    subset = df[df['lsu_id'].isin(target_lsu_ids)].copy()
    keep_cols = ['date', 'lsu_id'] + [f for f in fields if f in df.columns]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        print(f"  ⚠️  lsunit_wb missing columns: {missing}")
    subset = subset[keep_cols].copy()
    for col in fields:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')

    print(f"  After filter: {len(subset):,} rows, {subset['lsu_id'].nunique()} LSU")
    del df; gc.collect()
    return subset


# ============================================================
# Read aquifer_day.txt
# ============================================================
def read_aquifer(filepath, target_aqu_ids, fields):
    """Read aquifer_day.txt, matching by id parsed from the name column (aquXXXXX)"""
    print("Reading aquifer_day.txt ...")
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return None

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.0f} MB")

    target_set = set(target_aqu_ids)

    if file_size_mb > 5000:
        print(f"  ⚠️  File too large, using chunked read...")
        chunks = []
        for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+',
                                  engine='c', chunksize=2_000_000):
            chunk = chunk[chunk['name'].str.startswith('aqu')].copy()
            if len(chunk) == 0:
                continue
            chunk['aqu_id'] = chunk['name'].astype(str).str[3:].astype(int)
            filtered = chunk[chunk['aqu_id'].isin(target_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            print(f"  ⚠️  No matching data")
            return None
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
    else:
        df = pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c')
        df = df[df['name'].str.startswith('aqu')].copy()
        df['aqu_id'] = df['name'].astype(str).str[3:].astype(int)

    print(f"  Shape: {df.shape}")

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))

    subset = df[df['aqu_id'].isin(target_aqu_ids)].copy()
    keep_cols = ['date', 'aqu_id'] + [f for f in fields if f in df.columns]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        print(f"  ⚠️  aquifer missing columns: {missing}")
    subset = subset[keep_cols].copy()
    for col in fields:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')

    print(f"  After filter: {len(subset):,} rows, {subset['aqu_id'].nunique()} Aquifers")
    del df; gc.collect()
    return subset


# ============================================================
# Data reading functions
# ============================================================
def read_subday(filepath, target_channels):
    print("Reading channel_sd_subday.txt ...")
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return None

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.0f} MB")

    if file_size_mb > 5000:
        # File is still very large (>5GB), meaning prefiltering may have failed; use chunked read
        print(f"  ⚠️  File too large, using chunked read...")
        target_set = set(target_channels)
        chunks = []
        for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+',
                                  engine='c', chunksize=2_000_000):
            filtered = chunk[chunk['gis_id'].isin(target_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            print(f"  ⚠️  No matching data")
            return None
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
    else:
        df = pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c')

    print(f"  Shape: {df.shape}")

    tmin, tmax = df['tstep'].min(), df['tstep'].max()
    if tmin == 1 and tmax == 24:
        df['hour'] = df['tstep'] - 1
    elif tmin == 0 and tmax == 23:
        df['hour'] = df['tstep']
    else:
        raise ValueError(f"Unexpected tstep range: {tmin}~{tmax}")

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')

    subset = df[df['gis_id'].isin(target_channels)].copy()
    subset = subset[['datetime', 'date', 'hour', 'gis_id', 'flo_out']].copy()
    subset['flo_out'] = pd.to_numeric(subset['flo_out'], errors='coerce') / 3600.0
    print(f"  After filter: {len(subset):,} rows, {subset['gis_id'].nunique()} reaches")
    del df; gc.collect()
    return subset


def read_day(filepath, target_channels, daily_fields):
    print("Reading channel_sd_day.txt ...")
    if not filepath.exists():
        print(f"  ❌ File does not exist: {filepath}")
        return None

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.0f} MB")

    if file_size_mb > 5000:
        print(f"  ⚠️  File too large, using chunked read...")
        target_set = set(target_channels)
        chunks = []
        for chunk in pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+',
                                  engine='c', chunksize=2_000_000):
            filtered = chunk[chunk['gis_id'].isin(target_set)]
            if len(filtered) > 0:
                chunks.append(filtered)
        if not chunks:
            print(f"  ⚠️  No matching data")
            return None
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
    else:
        df = pd.read_csv(filepath, skiprows=[0, 2], sep=r'\s+', engine='c')

    print(f"  Shape: {df.shape}")

    df['date'] = pd.to_datetime(df[['yr', 'mon', 'day']].rename(
        columns={'yr': 'year', 'mon': 'month', 'day': 'day'}))
    subset = df[df['gis_id'].isin(target_channels)].copy()
    keep = ['date', 'gis_id'] + daily_fields
    subset = subset[keep].copy()
    for col in daily_fields:
        subset[col] = pd.to_numeric(subset[col], errors='coerce')
    print(f"  After filter: {len(subset):,} rows")
    del df; gc.collect()
    return subset


def read_weather_sta_cli(txtinout_dir):
    df = pd.read_csv(txtinout_dir / "weather-sta.cli", skiprows=1, sep=r'\s+', engine='c')
    print(f"weather-sta.cli: {len(df)} stations")
    mapping = {}
    for _, row in df.iterrows():
        wst = row['name'].strip()
        pcp = row['pcp'].replace('.pcp', '') if pd.notna(row['pcp']) else None
        tmp = row['tmp'].replace('.tmp', '') if pd.notna(row['tmp']) else None
        mapping[wst] = {'pcp': pcp, 'tmp': tmp}
    return mapping


def read_pcp_hourly(txtinout_dir, prefix):
    fp = txtinout_dir / f"{prefix}.pcp"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, skiprows=3, sep=r'\s+', header=None, engine='c')
    df.columns = ['yr', 'jday', 'mon', 'day', 'tstep', 'precip_mm'][:len(df.columns)]
    df['hour'] = df['tstep'].astype(int)
    df['date'] = pd.to_datetime(
        df['yr'].astype(int).astype(str) + df['jday'].astype(int).astype(str).str.zfill(3),
        format='%Y%j')
    df['datetime'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
    return df[['datetime', 'date', 'hour', 'precip_mm']].copy()


def read_tmp_daily(txtinout_dir, prefix):
    fp = txtinout_dir / f"{prefix}.tmp"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, skiprows=3, sep=r'\s+', header=None, engine='c')
    df.columns = ['year', 'jday', 'tmp_max', 'tmp_min'][:len(df.columns)]
    df['tmp_ave'] = (df['tmp_max'] + df['tmp_min']) / 2.0
    df['date'] = pd.to_datetime(
        df['year'].astype(int).astype(str) + df['jday'].astype(int).astype(str).str.zfill(3),
        format='%Y%j')
    return df[['date', 'tmp_ave']].copy()


# ============================================================
# GRFR interpolation (PCHIP 3h → 1h)
# ============================================================
def interpolate_grfr_to_hourly(qout_3h_df):
    qout_3h_df = qout_3h_df.sort_values('time').reset_index(drop=True)
    t_start = qout_3h_df['time'].iloc[0]
    x_3h = (qout_3h_df['time'] - t_start).dt.total_seconds() / 3600.0
    y_3h = qout_3h_df['Qout'].values

    pchip = PchipInterpolator(x_3h.values, y_3h)
    x_1h = np.arange(x_3h.iloc[0], x_3h.iloc[-1] + 2.5, 1.0)
    y_1h = np.maximum(pchip(x_1h), 0.0)

    dt_1h = [t_start + pd.Timedelta(hours=float(h)) for h in x_1h]
    result = pd.DataFrame({'datetime': dt_1h, 'grfr_Qout': y_1h})
    result['is_observed'] = result['datetime'].isin(set(qout_3h_df['time'].values)).astype(int)
    return result


# ============================================================
# Build feature matrix per river
# ============================================================
def build_hourly_features(row, subday_subset, swat_day_subset,
                          pcp_cache, tmp_cache, grfr_cache,
                          lsunit_wb_data=None, aquifer_data=None, channel_mapping=None):
    comid = row['GRFR_COMID']
    ch_id = row['swatplus_lcha']
    wst = row['swatplus_wst'].strip()

    # A. Hourly flo_out
    sub = subday_subset[subday_subset['gis_id'] == ch_id].copy()
    sub = sub.sort_values('datetime').reset_index(drop=True)
    if len(sub) == 0:
        return None

    # B. Hourly precipitation
    if wst in pcp_cache:
        sub = pd.merge(sub, pcp_cache[wst][['datetime', 'precip_mm']], on='datetime', how='left')
    else:
        sub['precip_mm'] = np.nan

    # C. Daily fields
    daily = swat_day_subset[swat_day_subset['gis_id'] == ch_id].drop(columns=['gis_id'])
    sub = pd.merge(sub, daily, on='date', how='left')

    # D. Daily temperature
    if wst in tmp_cache:
        sub = pd.merge(sub, tmp_cache[wst], on='date', how='left')
    else:
        sub['tmp_ave'] = np.nan

    # D2. Daily lsunit_wb (water balance) - aggregate over the LSUs mapped to the channel
    if lsunit_wb_data is not None and channel_mapping and ch_id in channel_mapping:
        lsu_ids = channel_mapping[ch_id]['lsu_ids']
        lsu_sub = lsunit_wb_data[lsunit_wb_data['lsu_id'].isin(lsu_ids)].copy()
        if len(lsu_sub) > 0:
            lsu_daily = lsu_sub.drop(columns=['lsu_id']).groupby('date').mean().reset_index()
            rename_cols = {c: f'wb_{c}' for c in lsu_daily.columns if c != 'date'}
            lsu_daily = lsu_daily.rename(columns=rename_cols)
            sub = pd.merge(sub, lsu_daily, on='date', how='left')

    # D3. Daily aquifer (groundwater)
    if aquifer_data is not None and channel_mapping and ch_id in channel_mapping:
        aqu_ids = channel_mapping[ch_id]['aquifer_ids']
        aqu_sub = aquifer_data[aquifer_data['aqu_id'].isin(aqu_ids)].copy()
        if len(aqu_sub) > 0:
            aqu_daily = aqu_sub.drop(columns=['aqu_id']).groupby('date').mean().reset_index()
            rename_cols = {c: f'aqu_{c}' for c in aqu_daily.columns if c != 'date'}
            aqu_daily = aqu_daily.rename(columns=rename_cols)
            sub = pd.merge(sub, aqu_daily, on='date', how='left')

    # E. flo_out lags (computed on the full continuous series, no breaks)
    for lag in LAG_HOURS:
        sub[f'flo_out_lag{lag}h'] = sub['flo_out'].shift(lag)

    # F. Rolling precipitation sums
    for w in PRECIP_ROLL_HOURS:
        sub[f'precip_sum_{w}h'] = sub['precip_mm'].rolling(w, min_periods=1).sum()

    # G. Time features
    sub['sin_hour'] = np.sin(2 * np.pi * sub['hour'] / 24.0)
    sub['cos_hour'] = np.cos(2 * np.pi * sub['hour'] / 24.0)
    doy = sub['datetime'].dt.dayofyear
    sub['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    sub['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)

    # H. GRFR target
    if comid not in grfr_cache:
        return None
    merged = pd.merge(sub, grfr_cache[comid], on='datetime', how='inner')
    merged['comid'] = comid
    merged['channel_id'] = ch_id
    return merged


# ============================================================
# Random month sampling split
# ============================================================
def random_month_split(dataset, val_months, test_months, seed=42, mode='discrete'):
    """
    Sample val/test from all months:
      mode='discrete': randomly sample discrete months (original logic)
      mode='contiguous': randomly choose a start point and take a contiguous block of months
    """
    rng = np.random.default_rng(seed)

    dataset['ym'] = dataset['datetime'].dt.to_period('M')
    all_months = sorted(dataset['ym'].unique().tolist())
    total = len(all_months)

    assert val_months + test_months < total, \
        f"val({val_months}) + test({test_months}) = {val_months+test_months} >= total months({total})"

    if mode == 'discrete':
        # Shuffle randomly, then sample
        indices = rng.permutation(total)
        val_indices = sorted(indices[:val_months])
        test_indices = sorted(indices[val_months:val_months + test_months])

    elif mode == 'contiguous':
        # Randomly choose two non-overlapping contiguous blocks
        # First choose the start of the val block
        max_start_val = total - val_months - test_months
        val_start = int(rng.integers(0, max_start_val + 1))
        val_indices = list(range(val_start, val_start + val_months))

        # Choose the start of the test block from the remaining indices
        remaining = [i for i in range(total) if i not in val_indices]
        # Find all contiguous segments in remaining of length >= test_months
        contiguous_starts = []
        for i in range(len(remaining) - test_months + 1):
            if remaining[i + test_months - 1] - remaining[i] == test_months - 1:
                contiguous_starts.append(i)

        if len(contiguous_starts) == 0:
            raise ValueError("Cannot find a long enough contiguous block for test in the remaining months")

        pick = int(rng.integers(0, len(contiguous_starts)))
        test_indices = remaining[contiguous_starts[pick]:contiguous_starts[pick] + test_months]

    else:
        raise ValueError(f"Unknown mode: {mode}, options: 'discrete', 'contiguous'")

    val_set = set(all_months[i] for i in val_indices)
    test_set = set(all_months[i] for i in test_indices)

    print(f"Total months: {total}  (seed: {seed}, mode: {mode})")
    print(f"  Train: {total - len(val_set) - len(test_set):2d} months")
    print(f"  Val:   {len(val_set):2d} months | {sorted(val_set)}")
    print(f"  Test:  {len(test_set):2d} months | {sorted(test_set)}")

    def assign_split(ym):
        if ym in val_set:
            return 'val'
        elif ym in test_set:
            return 'test'
        else:
            return 'train'

    dataset['split'] = dataset['ym'].map(assign_split)
    dataset.drop(columns=['ym'], inplace=True)
    return dataset


# ============================================================
# Main data preparation workflow
# ============================================================
def prepare_data(paths: dict):
    _lazy_imports()
    print("=" * 60)
    print("Step 2: data preparation")
    print("=" * 60)

    # --- Read the river lookup table ---
    print("\n--- Read the river lookup table ---")
    river_df = pd.read_csv(paths['river_table'])

    # ================= New: filter out unmatched stations/rivers =================
    # Check whether GRFR_COMID or swatplus_lcha is a missing value (NaN)
    invalid_mask = river_df['GRFR_COMID'].isna() | river_df['swatplus_lcha'].isna()

    if invalid_mask.any():
        skipped_count = invalid_mask.sum()
        skipped_df = river_df[invalid_mask]

        print(f"\n⚠️: Found {skipped_count} stations not matched to a GRFR_COMID or SWAT+ reach; they will be skipped!")
        print("Information on skipped stations:")
        # Print the skipped rows so you can verify which stations they are in the log
        print(skipped_df.to_string(index=False))

        # Filter out invalid rows, keeping only properly matched data to continue
        river_df = river_df[~invalid_mask].copy()
    # ==============================================================

    # Keep the original conversion logic (after filtering, so the rest can safely convert to int)
    river_df['GRFR_COMID'] = river_df['GRFR_COMID'].astype(int)
    river_df['swatplus_lcha'] = river_df['swatplus_lcha'].astype(int)
    print(f"Valid rivers: {len(river_df)}")
    print(river_df[['GRFR_COMID', 'swatplus_lcha', 'swatplus_wst']].head(10))

    target_channels = river_df['swatplus_lcha'].unique().tolist()
    target_comids = river_df['GRFR_COMID'].unique().tolist()

    # --- Prefilter channel files ---
    print("\n--- Prefilter channel files (extract target reaches) ---")
    print(f"Target reaches ({len(target_channels)}): {target_channels[:5]}...")
    for fkey in ['swat_subday_file', 'swat_day_file']:
        prefilter_channel_file(paths[fkey], target_channels)

    # --- Look up channel → LSU / Aquifer mapping ---
    print("\n--- Look up Channel → LSU / Aquifer mapping ---")
    channel_mapping = build_channel_mapping(
        paths['region_path'], paths['area_name'], target_channels, paths['txtinout'])

    # Collect all needed LSU and Aquifer ids
    all_lsu_ids = []
    all_aqu_ids = []
    for ch, m in channel_mapping.items():
        all_lsu_ids.extend(m['lsu_ids'])
        all_aqu_ids.extend(m['aquifer_ids'])
    all_lsu_ids = list(set(all_lsu_ids))
    all_aqu_ids = list(set(all_aqu_ids))
    print(f"  Needed LSU: {len(all_lsu_ids)}")
    print(f"  Needed Aquifer: {len(all_aqu_ids)}")

    # Write the mapping result back to the river lookup table
    print("\n--- Write the mapping result to the river lookup table ---")
    river_df['lsu_ids'] = river_df['swatplus_lcha'].map(
        lambda ch: str(channel_mapping[ch]['lsu_ids']) if ch in channel_mapping else '')
    river_df['aquifer_ids'] = river_df['swatplus_lcha'].map(
        lambda ch: str(channel_mapping[ch]['aquifer_ids']) if ch in channel_mapping else '')
    river_df.to_csv(paths['river_table'], index=False)
    print(f"  Updated: {paths['river_table']}")

    # --- Prefilter lsunit_wb and aquifer files ---
    print("\n--- Prefilter lsunit_wb / aquifer files ---")
    if KEEP_SWAT_OUTPUT:
        print("  Skipping prefilter overwrite (--keep-swat-output)")
    else:
        if all_lsu_ids:
            prefilter_swat_output_file(paths['lsunit_wb_file'], all_lsu_ids, 'rtu')
        if all_aqu_ids:
            prefilter_swat_output_file(paths['aquifer_file'], all_aqu_ids, 'aqu')

    # Delete unused large files (can be kept with --keep-swat-output)
    if not KEEP_SWAT_OUTPUT:
        for unused in ['sd_chanbud_day.txt', 'channel_sdmorph_day.txt']:
            f = paths['txtinout'] / unused
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                f.unlink()
                print(f"  🗑️ Deleted {unused} ({size_mb:.0f} MB)")
    else:
        print("  Keeping all SWAT+ output files (--keep-swat-output)")

    # --- Read subday ---
    print("\n--- Read channel_sd_subday.txt ---")
    subday_subset = read_subday(paths['swat_subday_file'], target_channels)

    # --- Read day ---
    print("\n--- Read channel_sd_day.txt ---")
    swat_day_subset = read_day(paths['swat_day_file'], target_channels, DAILY_FIELDS)

    # --- Weather station data ---
    print("\n--- Read weather station data ---")
    wst_mapping = read_weather_sta_cli(paths['txtinout'])
    unique_wsts = river_df['swatplus_wst'].str.strip().unique()

    print("\nReading hourly precipitation .pcp ...")
    pcp_cache = {}
    for wst in unique_wsts:
        if wst not in wst_mapping or wst_mapping[wst]['pcp'] is None:
            continue
        pcp_df = read_pcp_hourly(paths['txtinout'], wst_mapping[wst]['pcp'])
        if pcp_df is not None:
            pcp_cache[wst] = pcp_df
            print(f"  {wst} → {wst_mapping[wst]['pcp']}.pcp: {len(pcp_df):,} hours")
    print(f"Succeeded: {len(pcp_cache)} / {len(unique_wsts)}")

    print("\nReading daily temperature .tmp ...")
    tmp_cache = {}
    for wst in unique_wsts:
        if wst not in wst_mapping or wst_mapping[wst]['tmp'] is None:
            continue
        tmp_df = read_tmp_daily(paths['txtinout'], wst_mapping[wst]['tmp'])
        if tmp_df is not None:
            tmp_cache[wst] = tmp_df
            print(f"  {wst} → {wst_mapping[wst]['tmp']}.tmp: {len(tmp_df)} days")
    print(f"Succeeded: {len(tmp_cache)} / {len(unique_wsts)}")

    # --- Read lsunit_wb ---
    print("\n--- Read lsunit_wb_day.txt ---")
    lsunit_wb_data = None
    if all_lsu_ids:
        lsunit_wb_data = read_lsunit_wb(paths['lsunit_wb_file'], all_lsu_ids, LSUNIT_WB_FIELDS)
    else:
        print("  Skipped: no LSU mapping")

    # --- Read aquifer ---
    print("\n--- Read aquifer_day.txt ---")
    aquifer_data = None
    if all_aqu_ids:
        aquifer_data = read_aquifer(paths['aquifer_file'], all_aqu_ids, AQUIFER_FIELDS)
    else:
        print("  Skipped: no Aquifer mapping")

    # --- GRFR ---
    print("\n--- Read GRFR and interpolate to 1h ---")
    if GRFR_PFAF_OVERRIDE:
        grfr_pfaf = GRFR_PFAF_OVERRIDE
        print(f"Using manually specified pfaf = {grfr_pfaf}")
    else:
        grfr_pfaf = detect_grfr_pfaf(target_comids)

    grfr_cache = {}
    for year in GRFR_YEARS:
        nc_file = GRFR_DIR / f"output_pfaf_{grfr_pfaf}_{year}.nc"
        print(f"Reading: {nc_file}")
        if not nc_file.exists():
            print("  NOT FOUND, skip"); continue

        ds = xr.open_dataset(nc_file)
        available_rivids = ds['rivid'].values
        for comid in target_comids:
            if comid not in available_rivids:
                continue
            df_3h = ds['Qout'].sel(rivid=comid).to_dataframe().reset_index()
            df_3h['time'] = pd.to_datetime(df_3h['time'])
            df_1h = interpolate_grfr_to_hourly(df_3h)
            if comid not in grfr_cache:
                grfr_cache[comid] = df_1h
            else:
                grfr_cache[comid] = pd.concat([grfr_cache[comid], df_1h], ignore_index=True)
        ds.close()
        n_found = sum(1 for c in target_comids if c in available_rivids)
        print(f"  done ({n_found} COMIDs)")

    for comid in grfr_cache:
        grfr_cache[comid] = grfr_cache[comid].sort_values('datetime').reset_index(drop=True)
    print(f"\nSuccessfully interpolated: {len(grfr_cache)} COMIDs")

    # --- Build features (on the full continuous series, then split) ---
    print("\n--- Build hourly feature matrix per river ---")
    all_features = []
    for idx, row in river_df.iterrows():
        comid, ch_id = row['GRFR_COMID'], row['swatplus_lcha']
        print(f"[{idx+1}/{len(river_df)}] COMID={comid}, Ch={ch_id} ...", end=' ')
        result = build_hourly_features(row, subday_subset, swat_day_subset,
                                       pcp_cache, tmp_cache, grfr_cache,
                                       lsunit_wb_data, aquifer_data, channel_mapping)
        if result is not None and len(result) > 0:
            all_features.append(result)
            print(f"{len(result):,} hours")
        else:
            print("SKIPPED")

    dataset = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal data: {len(dataset):,} rows × {len(dataset.columns)} columns")
    print(f"Rivers: {dataset['comid'].nunique()}")

    del subday_subset, swat_day_subset, pcp_cache, tmp_cache, grfr_cache
    del lsunit_wb_data, aquifer_data
    gc.collect()

    # --- Remove low-flow reaches ---
    print("\n--- Remove low-flow reaches ---")
    max_flow = dataset.groupby('comid')['grfr_Qout'].max()
    small = max_flow[max_flow < FLOW_THRESHOLD].index.tolist()
    keep = max_flow[max_flow >= FLOW_THRESHOLD].index.tolist()
    print(f"Max flow < {FLOW_THRESHOLD} m³/s: {len(small)} reaches")
    for c in small:
        print(f"  COMID {c}: max = {max_flow[c]:.1f} m³/s ← removed")
    n_before = len(dataset)
    dataset = dataset[dataset['comid'].isin(keep)].reset_index(drop=True)
    print(f"Kept {len(keep)} reaches, {n_before:,} → {len(dataset):,} rows")

    # --- Define columns ---
    feature_cols = (
        ['flo_out', 'precip_mm']
        + [f'flo_out_lag{lag}h' for lag in LAG_HOURS]
        + [f'precip_sum_{w}h' for w in PRECIP_ROLL_HOURS]
        + DAILY_FIELDS + ['tmp_ave']
        + [f'wb_{f}' for f in LSUNIT_WB_FIELDS]
        + [f'aqu_{f}' for f in AQUIFER_FIELDS]
        + ['sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']
    )
    target_col = 'grfr_Qout'
    meta_cols = ['datetime', 'date', 'hour', 'comid', 'channel_id', 'is_observed']

    existing = [c for c in feature_cols if c in dataset.columns]
    missing = [c for c in feature_cols if c not in dataset.columns]
    if missing:
        print(f"⚠️  Missing columns: {missing}")
        feature_cols = existing
    print(f"Number of features: {len(feature_cols)}")

    # --- Drop NaN ---
    n_before = len(dataset)
    dataset = dataset.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"Dropped NaN: {n_before:,} → {len(dataset):,}")

    # --- Random month sampling split ---
    print("\n--- Split dataset (random month sampling) ---")
    print(f"val={VAL_MONTHS} months, test={TEST_MONTHS} months (random sampling)")
    dataset = random_month_split(dataset, VAL_MONTHS, TEST_MONTHS, seed=SPLIT_SEED, mode=SPLIT_MODE)

    for name in ['train', 'val', 'test']:
        sub = dataset[dataset['split'] == name]
        if len(sub) > 0:
            obs_pct = sub['is_observed'].mean() * 100
            print(f"  {name:5s}: {len(sub):>8,} rows | "
                  f"{sub['comid'].nunique()} rivers | observed={obs_pct:.1f}%")

    # --- Save ---
    print("\n--- Save ---")
    output_dir = paths['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cols = meta_cols + feature_cols + [target_col, 'split']
    output_file = output_dir / "ml_dataset_hourly.parquet"
    dataset[save_cols].to_parquet(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Total rows: {len(dataset):,}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Metadata
    val_months_list = sorted(dataset[dataset['split'] == 'val']['datetime'].dt.to_period('M').unique().tolist())
    test_months_list = sorted(dataset[dataset['split'] == 'test']['datetime'].dt.to_period('M').unique().tolist())

    meta = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'meta_cols': meta_cols,
        'lag_hours': LAG_HOURS,
        'precip_roll_hours': PRECIP_ROLL_HOURS,
        'data_range': [DATA_START, DATA_END],
        'split_method': 'random_months',
        'split_seed': SPLIT_SEED,
        'val_months': [str(m) for m in val_months_list],
        'test_months': [str(m) for m in test_months_list],
        'temporal_resolution': '1h',
        'grfr_interpolation': 'pchip_3h_to_1h',
        'grfr_pfaf': grfr_pfaf,
        'region': str(paths['region_path']),
        'flow_threshold': FLOW_THRESHOLD,
        'n_rivers': dataset['comid'].nunique(),
    }
    meta_file = output_dir / "ml_dataset_hourly_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved: {meta_file}")

# Validation
    print("\n--- Data validation ---")
    check = pd.read_parquet(output_file)
    print(check.groupby('split').size())
    print(f"NaN check (total): {check[feature_cols].isna().sum().sum()}")

    # Check feature completeness per river
    print("\n--- Per-river feature completeness check ---")
    all_ok = True
    for comid in sorted(check['comid'].unique()):
        river = check[check['comid'] == comid]
        nan_cols = []
        for col in feature_cols:
            nan_pct = river[col].isna().mean()
            if nan_pct > 0:
                nan_cols.append((col, nan_pct))
        if nan_cols:
            all_ok = False
            print(f"  ⚠️  COMID {comid} ({len(river):,} rows):")
            for col, pct in nan_cols:
                print(f"      {col}: {pct:.1%} NaN")
        else:
            print(f"  ✅ COMID {comid} ({len(river):,} rows): all features complete")

    if all_ok:
        print("  All river features complete ✅")

    print("\n✅ Data preparation complete!")


# ============================================================
# Main program
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SWAT+ × GRFR ML data preparation automation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_workflow.py --region brazil/brazil_norte
  python run_workflow.py --region europe_west/france --skip-swat
  python run_workflow.py --region brazil/brazil_norte --val-months 6 --test-months 6
  python run_workflow.py --region brazil/brazil_norte --seed 123
  python run_workflow.py --region europe_west/france --grfr-pfaf 02
        """
    )
    parser.add_argument('--region', required=True,
                        help='subfolder path, e.g. brazil/brazil_norte, europe_west/france')
    parser.add_argument('--skip-swat', action='store_true',
                        help='skip the SWAT+ run')
    parser.add_argument('--skip-data', action='store_true',
                        help='skip data preparation')
    parser.add_argument('--val-months', type=int, default=VAL_MONTHS,
                        help=f'how many months to randomly sample for val (default: {VAL_MONTHS})')
    parser.add_argument('--test-months', type=int, default=TEST_MONTHS,
                        help=f'how many months to randomly sample for test (default: {TEST_MONTHS})')
    parser.add_argument('--seed', type=int, default=SPLIT_SEED,
                        help=f'random split seed for reproducibility (default: {SPLIT_SEED})')
    parser.add_argument('--flow-threshold', type=float, default=FLOW_THRESHOLD,
                        help=f'low-flow removal threshold m³/s (default: {FLOW_THRESHOLD})')
    parser.add_argument('--grfr-pfaf', default=None,
                        help='manually specify the GRFR pfaf number (e.g. 06, 02)')
    parser.add_argument('--split-mode', choices=['discrete', 'contiguous'], default='discrete',
                        help='month sampling mode: discrete=random discrete, contiguous=random contiguous block (default: discrete)')
    parser.add_argument('--keep-swat-output', action='store_true',
                        help='keep all SWAT+ output files, do not delete unused large files')

    args = parser.parse_args()

    # Update module-level configuration
    import run_workflow as _self
    _self.VAL_MONTHS = args.val_months
    _self.TEST_MONTHS = args.test_months
    _self.SPLIT_SEED = args.seed
    _self.FLOW_THRESHOLD = args.flow_threshold
    _self.GRFR_PFAF_OVERRIDE = args.grfr_pfaf
    _self.SPLIT_MODE = args.split_mode
    _self.KEEP_SWAT_OUTPUT = args.keep_swat_output

    paths = resolve_paths(args.region)

    if not args.skip_swat:
        modify_print_prt(paths['txtinout'])
        run_swatplus(paths['txtinout'])
    else:
        print("⏩ Skipping SWAT+ run\n")

    if not args.skip_data:
        prepare_data(paths)
    else:
        print("⏩ Skipping data preparation\n")

    print("\n" + "=" * 60)
    print("🎉 Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
