#!/usr/bin/env python3
"""
SWAT+ simulation - server-side run script (merged version, supports daily/hourly)
============================================================
Merges all functionality of the original run_swat_server.py and fix_wst.py.

Workflow:
  1. Inject weather files + configure time.sim / print.prt / file.cio
  2. Automatically fix wst matching in .con files (original fix_wst.py)
  3. Run the SWAT+ simulation

Usage:
  # Daily simulation (default step=0, merge old + new stations)
  python run_swat_server.py \
      --txtinout ./TxtInOut \
      --weather-dir ./swat_weather \
      --start 2024-01-01 --end 2024-01-14 --nyskip 0

  # Hourly simulation (step=24)
  python run_swat_server.py \
      --txtinout ./TxtInOut \
      --weather-dir ./swat_weather \
      --start 2024-01-01 --end 2024-01-14 --nyskip 0 --step 24

  # Use new stations only, discard old CFSR stations
  python run_swat_server.py \
      --txtinout ./TxtInOut \
      --weather-dir ./swat_weather \
      --start 2024-01-01 --end 2024-01-14 --station-mode new-only

  # Inject + fix only, do not run (manual inspection)
  python run_swat_server.py \
      --txtinout ./TxtInOut \
      --weather-dir ./swat_weather \
      --start 2024-01-01 --end 2024-01-14 --nyskip 0 --no-run

  # Run fix_wst only (dry-run mode to preview changes)
  python run_swat_server.py \
      --txtinout ./TxtInOut --fix-wst-only --dry-run

  # Skip the fix_wst step
  python run_swat_server.py \
      --txtinout ./TxtInOut \
      --weather-dir ./swat_weather \
      --start 2024-01-01 --end 2024-01-14 --skip-fix-wst

Environment:
  pip install cdsapi netCDF4 xarray pandas numpy
  SWAT+ executable (rev60.5.7+) must be on PATH or specified with --swat-exe
"""

import os
import sys
import shutil
import argparse
import subprocess
import re
from pathlib import Path
from datetime import date, timedelta
from typing import Optional, List, Dict

import numpy as np


# ============================================================
# 1. Configure time.sim (hourly step)
# ============================================================
def configure_time_sim(
    txtinout: Path,
    start_date: date,
    end_date: date,
    step: int = 24,
    warmup_years: int = 0,
):
    time_sim = txtinout / "time.sim"
    jday_start = start_date.timetuple().tm_yday
    jday_end = end_date.timetuple().tm_yday

    content = (
        f"time.sim: written by run_swat_server.py\n"
        f"{'day_start':>10s}{'yrc_start':>10s}{'day_end':>10s}{'yrc_end':>10s}"
        f"{'step':>10s}  \n"
        f"{0:10d}{start_date.year:10d}"
        f"{jday_end:10d}{end_date.year:10d}"
        f"{step:10d}  \n"
    )

    backup_and_write(time_sim, content)
    print(f"  ✅ time.sim: {start_date} → {end_date}, step={step} ({'daily' if step == 0 else 'hourly'})")


# ============================================================
# 2. Configure print.prt (output control)
# ============================================================
def configure_print_prt(txtinout: Path, nyskip: int = 0, only_channel_sd: bool = True):
    """
    Configure the print.prt file.

    Args:
        txtinout: path to the TxtInOut directory
        nyskip: number of years to skip
        only_channel_sd: if True, print only the daily channel_sd output
    """
    prt_file = txtinout / "print.prt"
    if not prt_file.exists():
        print(f"  ⚠️  print.prt does not exist, skipping")
        return

    lines = prt_file.read_text().splitlines()
    if len(lines) < 3:
        print(f"  ⚠️  print.prt has an unexpected format, skipping")
        return

    # Modify line 3 (nyskip and other parameters)
    parts = lines[2].split()
    if len(parts) >= 1:
        parts[0] = str(nyskip)
    lines[2] = "".join(f"{p:<12s}" for p in parts)

    # Find the position of the "objects" line, then modify the output objects that follow
    if only_channel_sd:
        objects_line_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("objects"):
                objects_line_idx = i
                break

        if objects_line_idx is not None:
            # Starting from the line after objects, modify the output settings for all objects
            for i in range(objects_line_idx + 1, len(lines)):
                parts = lines[i].split()
                if len(parts) >= 5:
                    obj_name = parts[0]
                    if obj_name == "channel_sd":
                        # channel_sd: daily=y, rest=n
                        lines[i] = f"{obj_name:<20s}{'y':>12s}{'n':>12s}{'n':>12s}{'n':>12s}"
                    else:
                        # other objects: all n
                        lines[i] = f"{obj_name:<20s}{'n':>12s}{'n':>12s}{'n':>12s}{'n':>12s}"

    backup_and_write(prt_file, "\n".join(lines) + "\n")
    print(f"  ✅ print.prt: nyskip={nyskip}" + (", only channel_sd daily" if only_channel_sd else ""))


# ============================================================
# 3a. Reformat weather-sta.cli to Windows SWAT+ Editor format
# ============================================================
def reformat_weather_sta_cli(filepath: Path):
    """
    Reformat weather-sta.cli to the Windows SWAT+ Editor column-width format.
    The first 9 columns (name~atmo_dep) use fixed widths; extra columns
    (lat, lon, elev, etc.) are kept as-is.
    """
    BASE_COLS = ["name", "wgn", "pcp", "tmp", "slr", "hmd", "wnd", "pet", "atmo_dep"]

    lines = filepath.read_text(errors='replace').splitlines()
    if len(lines) < 2:
        return

    # Read the original header, identify extra columns (lat, lon, elev, etc.)
    orig_header = lines[1].split()
    extra_cols = orig_header[9:] if len(orig_header) > 9 else []

    def fmt_base(vals):
        """Format the first 9 columns."""
        s = f"{vals[0]:<27s}{vals[1]:>7s}"
        for v in vals[2:8]:
            s += f"{v:>27s}"
        s += f"{vals[8]:>18s}"
        return s

    def fmt_extra(vals):
        """Format the extra columns (lat, lon, elev, etc.)."""
        return "".join(f"{v:>16s}" for v in vals)

    with open(filepath, "w", newline="") as f:
        # Line 1: comment kept as-is
        f.write(lines[0].rstrip() + "\r\n")
        # Line 2: column header (first 9 columns + extra columns)
        header_line = fmt_base(BASE_COLS)
        if extra_cols:
            header_line += fmt_extra(extra_cols)
        f.write(header_line + "  \r\n")
        # Data rows
        for line in lines[2:]:
            parts = line.split()
            if not parts:
                continue
            # Pad to 9 columns
            while len(parts) < 9:
                parts.append("null")
            data_line = fmt_base(parts[:9])
            # Keep extra columns (lat, lon, elev, etc.)
            if len(parts) > 9:
                data_line += fmt_extra(parts[9:])
            f.write(data_line + "  \r\n")


# ============================================================
# 3. Copy weather files + merge weather-sta.cli
# ============================================================
def latlon_to_station_name(lat, lon):
    lat_code = int(round(abs(lat) * 1000))
    lon_code = int(round(abs(lon) * 1000))
    lat_dir = "n" if lat >= 0 else "s"
    lon_dir = "e" if lon >= 0 else "w"
    return f"s{lat_code}{lat_dir}{lon_code}{lon_dir}"


def merge_weather_sta_cli(txtinout: Path, weather_dir: Path, station_mode: str = "merge"):
    """
    Merge weather-sta.cli: keep the existing CFSR WGN stations + add ERA5-Land stations.

    station_mode:
      - "merge"    : keep the existing old stations (CFSR) + add new stations (ERA5-Land) (default)
      - "new-only" : use only the new stations (ERA5-Land), discard old stations

    1. Read the existing weather-sta.cli in TxtInOut (CFSR WGN stations)
    2. Read the newly generated weather-sta.cli in weather_dir (ERA5-Land stations)
    3. Rename ERA5-Land station names from sequential numbers to coordinate names (s15000n102000e)
    4. (merge mode) Remap all fields (wgn/pcp/tmp etc.) of CFSR stations to the nearest ERA5 station
    5. Merge both groups and write back to TxtInOut

    Returns: name_map = {old sequential name: new coordinate name}
    """
    BASE_COLS = ["name", "wgn", "pcp", "tmp", "slr", "hmd", "wnd", "pet", "atmo_dep"]

    def fmt_line(vals):
        """Windows Editor column-width format."""
        s = f"{vals[0]:<27s}{vals[1]:>7s}"
        for v in vals[2:8]:
            s += f"{v:>27s}"
        s += f"{vals[8]:>18s}  "
        return s

    # ---- Read existing stations (from TxtInOut, may be CFSR WGN stations or full Windows version) ----
    existing_stations = []
    existing_path = txtinout / "weather-sta.cli"
    if existing_path.exists():
        lines = existing_path.read_text(errors='replace').splitlines()
        orig_header = lines[1].split() if len(lines) > 1 else []
        for line in lines[2:]:
            parts = line.split()
            if not parts:
                continue
            row = {}
            for i, h in enumerate(orig_header):
                if i < len(parts):
                    row[h] = parts[i]
            existing_stations.append(row)
        print(f"  📋 Existing stations: {len(existing_stations)}")

    print(f"  📋 Station mode: {'merge (keep old + new stations)' if station_mode == 'merge' else 'new only (discard old stations)'}")

    # new-only mode: clear old stations, use only new ERA5-Land stations
    if station_mode == "new-only":
        existing_stations = []

    existing_names = {s["name"] for s in existing_stations}

    # ---- Read new ERA5-Land stations ----
    new_sta_path = weather_dir / "weather-sta.cli"
    if not new_sta_path.exists():
        print(f"  ⚠️  no weather-sta.cli in weather_dir")
        return {}

    new_lines = new_sta_path.read_text(errors='replace').splitlines()
    new_header = new_lines[1].split() if len(new_lines) > 1 else []
    lat_idx = new_header.index("lat") if "lat" in new_header else None
    lon_idx = new_header.index("lon") if "lon" in new_header else None

    name_map = {}  # old sequential name → new coordinate name
    era5_stations = []

    for line in new_lines[2:]:
        parts = line.split()
        if not parts:
            continue
        old_name = parts[0]

        lat = lon = None
        if lat_idx is not None and lon_idx is not None:
            try:
                lat = float(parts[lat_idx])
                lon = float(parts[lon_idx])
            except (IndexError, ValueError):
                pass
        if lat is None:
            print(f"  ⚠️  station {old_name} has no lat/lon, skipping")
            continue

        new_name = latlon_to_station_name(lat, lon)

        # If a station with the same name already exists (CFSR), update its fields with ERA5 data
        if new_name in existing_names:
            for es in existing_stations:
                if es["name"] == new_name:
                    for h in ["pcp", "tmp", "slr", "hmd", "wnd"]:
                        ci = new_header.index(h) if h in new_header else None
                        if ci is not None and ci < len(parts):
                            es[h] = parts[ci]
                    # also update wgn
                    wi = new_header.index("wgn") if "wgn" in new_header else None
                    if wi is not None and wi < len(parts):
                        es["wgn"] = parts[wi]
                    break
            name_map[old_name] = new_name
            continue

        # Create a new ERA5 station
        row = {"name": new_name}
        for i, h in enumerate(new_header):
            if i < len(parts) and h != "name":
                row[h] = parts[i]
        row["wgn"] = row.get("wgn", old_name)
        era5_stations.append(row)
        name_map[old_name] = new_name

    print(f"  📋 ERA5-Land stations: {len(era5_stations)} (sequential number → coordinate name)")

    # ---- Separate the pure CFSR stations among the existing ones (coordinate names not in ERA5) ----
    era5_new_names = {s["name"] for s in era5_stations}
    cfsr_stations = [s for s in existing_stations if s["name"] not in era5_new_names
                     and s["name"] not in name_map.values()]

    # ---- CFSR stations: unconditionally remap wgn + data files to the nearest ERA5 station ----
    if cfsr_stations and era5_stations:
        era5_lats, era5_lons = [], []
        for sta in era5_stations:
            m = re.match(r's(\d+)([ns])(\d+)([ew])', sta["name"])
            if m:
                era5_lats.append(int(m.group(1)) / 1000.0 * (1 if m.group(2) == 'n' else -1))
                era5_lons.append(int(m.group(3)) / 1000.0 * (1 if m.group(4) == 'e' else -1))
            else:
                era5_lats.append(0.0)
                era5_lons.append(0.0)
        era5_coords = np.column_stack([era5_lats, era5_lons])

        n_remapped = 0
        for sta in cfsr_stations:
            m = re.match(r's(\d+)([ns])(\d+)([ew])', sta["name"])
            if m:
                clat = int(m.group(1)) / 1000.0 * (1 if m.group(2) == 'n' else -1)
                clon = int(m.group(3)) / 1000.0 * (1 if m.group(4) == 'e' else -1)
                dist = (era5_coords[:, 0] - clat) ** 2 + (era5_coords[:, 1] - clon) ** 2
                idx = int(np.argmin(dist))
                nearest = era5_stations[idx]
                # Unconditionally overwrite wgn and all data-file fields
                sta["wgn"] = nearest.get("wgn", sta.get("wgn", "null"))
                for h in ["pcp", "tmp", "slr", "hmd", "wnd"]:
                    sta[h] = nearest.get(h, "null")
                n_remapped += 1
        print(f"  📋 CFSR station wgn + data files remapped: {n_remapped} stations")

    # ---- Merge and sort alphabetically by station name (matches Windows Editor) ----
    all_stations = existing_stations + era5_stations
    all_stations.sort(key=lambda s: s.get("name", ""))

    with open(existing_path, "w", newline="") as f:
        f.write("weather-sta.cli: written by run_swat_server.py (merged)\r\n")
        f.write(fmt_line(BASE_COLS) + "\r\n")
        for sta in all_stations:
            vals = []
            for c in BASE_COLS:
                v = sta.get(c, "null")
                vals.append(v if v else "null")
            f.write(fmt_line(vals) + "\r\n")

    reformat_weather_sta_cli(existing_path)

    n_cfsr = len(cfsr_stations)
    n_era5 = len(era5_stations)
    n_overlap = len(existing_stations) - n_cfsr
    total = len(all_stations)
    print(f"  ✅ weather-sta.cli merge complete: {total} stations")
    if station_mode == "new-only":
        print(f"     new stations only (ERA5-Land): {n_era5}")
    else:
        print(f"     CFSR WGN: {n_cfsr} | ERA5 (new): {n_era5} | ERA5 (existing, updated): {n_overlap}")

    return name_map


def inject_weather_files(txtinout: Path, weather_dir: Path, station_mode: str = "merge"):
    """
    Inject weather files into TxtInOut (merge mode).
    1. Copy .pcp/.tmp/.slr/.hmd/.wnd data files (without renaming)
    2. Merge weather-sta.cli (keep or discard original CFSR stations per station_mode)
    3. Return name_map for later use
    """
    weather_dir = Path(weather_dir)
    if not weather_dir.exists():
        raise FileNotFoundError(f"weather directory does not exist: {weather_dir}")

    exts = ["pcp", "tmp", "slr", "hmd", "wnd"]
    counts = {e: 0 for e in exts}

    print(f"\n📂 Injecting weather files: {weather_dir} → {txtinout}")

    for ext in exts:
        cli = weather_dir / f"{ext}.cli"
        if cli.exists():
            shutil.copy2(cli, txtinout / f"{ext}.cli")
            print(f"  ✅ {ext}.cli")

    # Merge weather-sta.cli (station_mode decides whether to keep old stations)
    name_map = merge_weather_sta_cli(txtinout, weather_dir, station_mode=station_mode)

    for ext in exts:
        for f in sorted(weather_dir.glob(f"*.{ext}")):
            shutil.copy2(f, txtinout / f.name)
            counts[ext] += 1

    total = sum(counts.values())
    print(f"  ✅ Copied {total} station data files in total:")
    for ext in exts:
        print(f"     .{ext}: {counts[ext]} files")

    for wgn_file in ["wgn_stations.csv", "wgn_monthly.csv"]:
        src = weather_dir / wgn_file
        if src.exists():
            shutil.copy2(src, txtinout / wgn_file)
            print(f"  ✅ {wgn_file}")

    return name_map


# ============================================================
# 4. Generate pcp.cli / tmp.cli / slr.cli / hmd.cli / wnd.cli
# ============================================================
def create_climate_cli_files(txtinout: Path, start_date: date, end_date: date):
    wsta = txtinout / "weather-sta.cli"
    if not wsta.exists():
        print(f"  ⚠️  weather-sta.cli does not exist, skipping .cli generation")
        return

    lines = wsta.read_text().splitlines()
    header = lines[1].split() if len(lines) > 1 else []

    ext_cols = {}
    for ext in ["pcp", "tmp", "slr", "hmd", "wnd"]:
        for i, h in enumerate(header):
            if h == ext:
                ext_cols[ext] = i
                break

    files_by_ext = {ext: [] for ext in ["pcp", "tmp", "slr", "hmd", "wnd"]}
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        for ext, col_idx in ext_cols.items():
            if col_idx < len(parts):
                fname = parts[col_idx]
                if fname != "null" and fname.endswith(f".{ext}"):
                    files_by_ext[ext].append(fname)

    print(f"\n📋 Generating .cli index files...")
    for ext, flist in files_by_ext.items():
        if not flist:
            continue

        seen = set()
        unique = []
        for fn in flist:
            if fn not in seen:
                seen.add(fn)
                if (txtinout / fn).exists():
                    unique.append(fn)

        if not unique:
            print(f"  ⚠️  {ext}.cli: no valid data files")
            continue

        unique.sort()  # Sort alphabetically by file name, matching Windows Editor

        cli_path = txtinout / f"{ext}.cli"
        with open(cli_path, "w") as f:
            f.write(f"{ext}.cli: written by run_swat_server.py\n")
            f.write(f"filename\n")
            for fn in unique:
                f.write(f"{fn}\n")
        print(f"  ✅ {ext}.cli ({len(unique)} files)")

    update_file_cio_climate(txtinout)


def update_file_cio_climate(txtinout: Path):
    cio = txtinout / "file.cio"
    if not cio.exists():
        print(f"  ⚠️  file.cio does not exist!")
        return

    text = cio.read_text()
    lines = text.splitlines()

    cli_names = {ext: f"{ext}.cli" if (txtinout / f"{ext}.cli").exists() else "null"
                 for ext in ["pcp", "tmp", "slr", "hmd", "wnd"]}

    new_climate = (f"climate           weather-sta.cli   weather-wgn.cli   null              "
                   f"{cli_names['pcp']:18s}{cli_names['tmp']:18s}"
                   f"{cli_names['slr']:18s}{cli_names['hmd']:18s}"
                   f"{cli_names['wnd']:18s}null              ")

    new_lines = []
    changed = False
    for line in lines:
        if line.strip().startswith("climate"):
            new_lines.append(new_climate)
            changed = True
        else:
            new_lines.append(line)

    if changed:
        backup_and_write(cio, "\n".join(new_lines) + "\n")
        print(f"  ✅ file.cio climate line updated:")
        print(f"     {new_climate.strip()}")
    else:
        print(f"  ⚠️  climate line not found in file.cio")


# ============================================================
# 4b. Check file.cio
# ============================================================
def verify_file_cio(txtinout: Path):
    cio = txtinout / "file.cio"
    if not cio.exists():
        print(f"  ⚠️  file.cio does not exist!")
        return False

    text = cio.read_text()
    if "weather-sta.cli" in text:
        print(f"  ✅ file.cio references weather-sta.cli")
        return True
    else:
        print(f"  ⚠️  file.cio does not reference weather-sta.cli, please check the climate section manually")
        return False


# ============================================================
# 5. Update weather-wgn.cli (weather generator association)
# ============================================================
def update_weather_wgn(txtinout: Path, weather_dir: Path):
    import csv

    stations_csv = weather_dir / "wgn_stations.csv"
    monthly_csv = weather_dir / "wgn_monthly.csv"

    if not stations_csv.exists() or not monthly_csv.exists():
        print(f"  ℹ️  WGN CSV does not exist, skipping weather-wgn.cli generation")
        return

    stations = []
    with open(stations_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append(row)

    monthly = {}
    with open(monthly_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wgn_id = int(row["wgn_id"])
            monthly.setdefault(wgn_id, []).append(row)

    stations.sort(key=lambda s: s["name"])  # Alphabetical by station name, matching Windows Editor

    wgn_cli = txtinout / "weather-wgn.cli"
    WGN_COLS = ["tmp_max_ave","tmp_min_ave","tmp_max_sd","tmp_min_sd",
                "pcp_ave","pcp_sd","pcp_skew","wet_dry","wet_wet",
                "pcp_days","pcp_hhr","slr_ave","dew_ave","wnd_ave"]
    col_header = "".join(f"  {c:>12s}" for c in WGN_COLS)

    with open(wgn_cli, "w", newline="") as f:
        f.write("weather-wgn.cli: written by run_swat_server.py\r\n")

        for sta in stations:
            wgn_id = int(sta["id"])
            name = sta["name"]
            lat = float(sta["lat"])
            lon = float(sta["lon"])
            elev = float(sta["elev"])
            rain_yrs = int(sta["rain_yrs"])

            f.write(f"{name:20s}{lat:14.5f}{lon:14.5f}{elev:14.5f}{rain_yrs:10d}  \r\n")
            f.write(col_header + "\r\n")

            months = monthly.get(wgn_id, [])
            months.sort(key=lambda r: int(r["month"]))
            for m in months:
                row = (
                    f"  {float(m['tmp_max_ave']):12.5f}"
                    f"  {float(m['tmp_min_ave']):12.5f}"
                    f"  {float(m['tmp_max_sd']):12.5f}"
                    f"  {float(m['tmp_min_sd']):12.5f}"
                    f"  {float(m['pcp_ave']):12.5f}"
                    f"  {float(m['pcp_sd']):12.5f}"
                    f"  {float(m['pcp_skew']):12.5f}"
                    f"  {float(m['wet_dry']):12.5f}"
                    f"  {float(m['wet_wet']):12.5f}"
                    f"  {float(m['pcp_days']):12.5f}"
                    f"  {float(m['pcp_hhr']):12.5f}"
                    f"  {float(m['slr_ave']):12.5f}"
                    f"  {float(m['dew_ave']):12.5f}"
                    f"  {float(m['wnd_ave']):12.5f}\r\n"
                )
                f.write(row)

    print(f"  ✅ weather-wgn.cli ({len(stations)} stations)")


# ============================================================
# 6. Run SWAT+
# ============================================================
def run_swat(txtinout: Path, swat_exe: str = "swatplus", timeout: int = 7200):
    exe_path = shutil.which(swat_exe)
    if exe_path is None:
        common_paths = [
            Path(swat_exe),
            txtinout / "swatplus",
            txtinout / "rev60.5.7_64rel.exe",
            Path.home() / "swatplus" / "swatplus",
            Path("/usr/local/bin/swatplus"),
        ]
        for p in common_paths:
            if p.exists() and os.access(p, os.X_OK):
                exe_path = str(p)
                break

    if exe_path is None:
        print(f"\n❌ SWAT+ executable not found: {swat_exe}")
        print(f"   Please specify the path with --swat-exe, or put swatplus on PATH")
        return False

    print(f"\n{'='*60}")
    print(f"🚀 Running SWAT+ simulation")
    print(f"   Executable: {exe_path}")
    print(f"   Working directory: {txtinout}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [exe_path],
            cwd=str(txtinout),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        log_file = txtinout / "swat_run.log"
        with open(log_file, "w") as f:
            f.write("===== STDOUT =====\n")
            f.write(result.stdout)
            f.write("\n===== STDERR =====\n")
            f.write(result.stderr)

        if result.returncode == 0:
            print(f"  ✅ SWAT+ ran successfully! (return code 0)")
            print(f"  📄 Log: {log_file}")
            stdout_lines = result.stdout.strip().split("\n")
            if len(stdout_lines) > 5:
                print(f"\n  --- last 5 lines of output ---")
                for line in stdout_lines[-5:]:
                    print(f"  {line}")
            return True
        else:
            print(f"  ❌ SWAT+ run failed! (return code {result.returncode})")
            print(f"  📄 Log: {log_file}")
            stderr_lines = result.stderr.strip().split("\n") if result.stderr else []
            stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
            error_lines = stderr_lines or stdout_lines[-20:]
            for line in error_lines[:20]:
                print(f"  ⚠️  {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ❌ SWAT+ run timed out ({timeout}s)")
        return False
    except Exception as e:
        print(f"  ❌ run error: {e}")
        return False


# ============================================================
# 7. Parse outputs
# ============================================================
def parse_outputs(txtinout: Path):
    print(f"\n{'='*60}")
    print(f"📊 Output file summary")
    print(f"{'='*60}")

    output_patterns = [
        ("channel_sd_*.txt",      "channel (sub-daily)"),
        ("channel_sdmorph_*.txt", "channel morphology (sub-daily)"),
        ("channel_day.txt",       "channel (daily)"),
        ("channel_yr.txt",        "channel (yearly)"),
        ("aquifer_*.txt",         "aquifer"),
        ("basin_wb_*.txt",        "basin water balance"),
        ("basin_nb_*.txt",        "basin nitrogen balance"),
        ("basin_pw_*.txt",        "basin plant-water"),
        ("lsunit_wb_*.txt",       "landscape unit water"),
        ("hru_wb_*.txt",          "HRU water"),
        ("deposition_*.txt",      "deposition"),
    ]

    found = []
    for pattern, desc in output_patterns:
        files = list(txtinout.glob(pattern))
        if files:
            total_mb = sum(f.stat().st_size for f in files) / 1048576
            found.append((desc, len(files), total_mb))
            print(f"  📄 {desc}: {len(files)} files, {total_mb:.1f} MB")

    if not found:
        txt_files = [f for f in txtinout.glob("*.txt")
                     if f.stat().st_size > 0 and f.name not in [
                         "file.cio", "time.sim", "print.prt"]]
        if txt_files:
            print(f"  📄 {len(txt_files)} output .txt files in total")
            for f in sorted(txt_files)[:15]:
                sz = f.stat().st_size / 1024
                print(f"     {f.name}: {sz:.1f} KB")
            if len(txt_files) > 15:
                print(f"     ... and {len(txt_files)-15} more files")
        else:
            print(f"  ⚠️  no output files found")

    basin_wb = list(txtinout.glob("basin_wb_*.txt"))
    if basin_wb:
        try:
            _summarize_basin_wb(basin_wb[0])
        except Exception as e:
            print(f"  ⚠️  failed to parse basin_wb: {e}")


def _summarize_basin_wb(filepath: Path):
    lines = filepath.read_text().strip().split("\n")
    if len(lines) < 3:
        return
    header = lines[0].split()
    last = lines[-1].split()
    if len(header) == len(last):
        print(f"\n  📊 basin_wb summary (last row):")
        for h, v in zip(header[:8], last[:8]):
            print(f"     {h}: {v}")


# ============================================================
# Utility functions
# ============================================================
def backup_and_write(filepath: Path, content: str):
    if filepath.exists():
        bak = filepath.with_suffix(filepath.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(filepath, bak)
    filepath.write_text(content)


def validate_txtinout(txtinout: Path) -> bool:
    required = ["file.cio"]
    missing = [f for f in required if not (txtinout / f).exists()]
    if missing:
        print(f"  ❌ TxtInOut directory is missing key files: {', '.join(missing)}")
        return False
    return True


# ============================================================
# PySWAT approach (optional)
# ============================================================
def run_with_pyswat(txtinout: Path, start_date: date, end_date: date, step: int = 0):
    try:
        from pySWATPlus import TxtinoutReader, FileEditor
        print(f"\n  ℹ️  pySWATPlus detected, running via Python API...")
        reader = TxtinoutReader(str(txtinout))
        time_editor = FileEditor(str(txtinout / "time.sim"))
        time_editor.set_value("day_start", start_date.timetuple().tm_yday)
        time_editor.set_value("yr_start", start_date.year)
        time_editor.set_value("day_end", end_date.timetuple().tm_yday)
        time_editor.set_value("yr_end", end_date.year)
        time_editor.set_value("step", step)
        time_editor.save()
        print(f"  ✅ pySWATPlus: time.sim configured (step={step})")
        return True
    except ImportError:
        pass

    try:
        from pyswat import SWAT
        print(f"\n  ℹ️  pyswat detected, running via Python API...")
        model = SWAT(str(txtinout))
        model.set_time(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            time_step="daily" if step == 0 else "hourly",
        )
        model.run()
        print(f"  ✅ pyswat: simulation complete")
        return True
    except ImportError:
        pass

    return False


# ============================================================
# ============================================================
#  fix_wst functionality (all logic from the original fix_wst.py)
# ============================================================
# ============================================================

def parse_weather_sta(txtinout):
    """Parse station coordinates from weather-sta.cli."""
    wsta = txtinout / "weather-sta.cli"
    if not wsta.exists():
        raise FileNotFoundError(f"{wsta}")
    lines = wsta.read_text().splitlines()
    header = lines[1].split()
    lat_idx = header.index("lat") if "lat" in header else None
    lon_idx = header.index("lon") if "lon" in header else None
    stations = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        lat = lon = None
        if lat_idx is not None and lon_idx is not None:
            try:
                lat, lon = float(parts[lat_idx]), float(parts[lon_idx])
            except:
                pass
        if lat is None:
            try:
                lat, lon = float(parts[-3]), float(parts[-2])
            except:
                pass
        if lat is None:
            m = re.match(r's(\d+)([ns])(\d+)([ew])', name)
            if m:
                lat = int(m.group(1)) / 1000.0 * (1 if m.group(2) == 'n' else -1)
                lon = int(m.group(3)) / 1000.0 * (1 if m.group(4) == 'e' else -1)
        if lat is not None:
            stations.append({"name": name, "lat": lat, "lon": lon})
    return stations


def build_nearest_finder(stations):
    """Build a nearest-station finder."""
    coords = np.array([(s["lat"], s["lon"]) for s in stations])
    names = [s["name"] for s in stations]

    def find_nearest(lat, lon):
        dist = (coords[:, 0] - lat) ** 2 + (coords[:, 1] - lon) ** 2
        return names[np.argmin(dist)]

    return find_nearest


def replace_nth_token(line, n, new_value):
    """Replace the nth token in the line."""
    tokens = list(re.finditer(r'\S+', line))
    if n >= len(tokens):
        return line
    m = tokens[n]
    old_len = m.end() - m.start()
    if len(new_value) <= old_len:
        padded = new_value + ' ' * (old_len - len(new_value))
    else:
        padded = new_value
    return line[:m.start()] + padded + line[m.end():]


def fix_con_file(filepath, find_nearest, dry_run=False):
    """Fix the wst column in a single .con file."""
    text = filepath.read_text()
    lines = text.splitlines()
    if len(lines) < 3:
        return 0
    header = lines[1].split()
    try:
        wst_idx = header.index("wst")
        lat_idx = header.index("lat")
        lon_idx = header.index("lon")
    except ValueError:
        return 0

    n_changed = 0
    new_lines = [lines[0], lines[1]]
    for line in lines[2:]:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        if len(parts) <= max(wst_idx, lat_idx, lon_idx):
            new_lines.append(line)
            continue
        try:
            lat, lon = float(parts[lat_idx]), float(parts[lon_idx])
        except ValueError:
            new_lines.append(line)
            continue
        old_wst = parts[wst_idx]
        new_wst = find_nearest(lat, lon)
        if old_wst != new_wst:
            new_lines.append(replace_nth_token(line, wst_idx, new_wst))
            n_changed += 1
        else:
            new_lines.append(line)

    if n_changed > 0 and not dry_run:
        bak = filepath.with_suffix(filepath.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(filepath, bak)
        filepath.write_text("\n".join(new_lines) + "\n")
    return n_changed


def check_wgn_refs(txtinout):
    """Check whether the wgn references in weather-sta.cli exist in weather-wgn.cli."""
    wsta = txtinout / "weather-sta.cli"
    wgn = txtinout / "weather-wgn.cli"
    if not wsta.exists() or not wgn.exists():
        print("  file missing")
        return
    wgn_names = set()
    for line in wgn.read_text().splitlines()[2:]:
        parts = line.split()
        if not parts:
            continue
        try:
            float(parts[0])
        except ValueError:
            wgn_names.add(parts[0])
    header = wsta.read_text().splitlines()[1].split()
    wgn_idx = header.index("wgn") if "wgn" in header else 1
    missing, total = [], 0
    for line in wsta.read_text().splitlines()[2:]:
        parts = line.split()
        if not parts or len(parts) <= wgn_idx:
            continue
        ref = parts[wgn_idx]
        if ref == "null":
            continue
        total += 1
        if ref not in wgn_names:
            missing.append(ref)
    if missing:
        print(f"  {len(missing)}/{total} wgn refs MISSING")
        for m in sorted(set(missing))[:10]:
            print(f"    X {m}")
    else:
        print(f"  OK ({total} refs)")


def check_data_file_refs(txtinout):
    """Check whether the data files referenced by weather-sta.cli exist."""
    wsta = txtinout / "weather-sta.cli"
    if not wsta.exists():
        return
    missing, checked = [], set()
    for line in wsta.read_text().splitlines()[2:]:
        for t in line.split():
            if t == "null":
                continue
            if any(t.endswith(e) for e in [".pcp", ".tmp", ".slr", ".hmd", ".wnd"]):
                if t not in checked:
                    checked.add(t)
                    if not (txtinout / t).exists():
                        missing.append(t)
    if missing:
        print(f"  {len(missing)} data files MISSING")
        for m in sorted(missing)[:10]:
            print(f"    X {m}")
    else:
        print(f"  OK ({len(checked)} files)")


def check_wgn_consistency(txtinout, station_names):
    """Check consistency of stations in weather-wgn.cli with weather-sta.cli."""
    wgn = txtinout / "weather-wgn.cli"
    if not wgn.exists():
        print("  not found")
        return
    wgn_names = set()
    for line in wgn.read_text().splitlines()[2:]:
        parts = line.split()
        if not parts:
            continue
        try:
            float(parts[0])
        except ValueError:
            wgn_names.add(parts[0])
    missing = wgn_names - station_names
    if missing:
        print(f"  {len(missing)} wgn stations NOT in weather-sta.cli")
    else:
        print(f"  OK ({len(wgn_names)} stations)")


def run_fix_wst(txtinout: Path, dry_run: bool = False):
    """
    Run the full fix_wst workflow:
      - Parse stations from weather-sta.cli
      - Fix the wst column in all .con files
      - Check wgn references, data-file references, and wgn consistency
    """
    print(f"\n{'='*60}")
    print(f"🔧 fix_wst: automatically match .con wst columns")
    print(f"   Directory: {txtinout}")
    print(f"   Mode: {'DRY-RUN' if dry_run else 'WRITE'}")
    print(f"{'='*60}")

    # Parse stations
    try:
        stations = parse_weather_sta(txtinout)
    except FileNotFoundError:
        print(f"  ⚠️  weather-sta.cli does not exist, skipping fix_wst")
        return 0

    print(f"\n  weather-sta.cli: {len(stations)} stations")
    if not stations:
        print("  ERROR: no stations")
        return 0
    lats = [s["lat"] for s in stations]
    lons = [s["lon"] for s in stations]
    print(f"  lat [{min(lats):.2f}, {max(lats):.2f}]  lon [{min(lons):.2f}, {max(lons):.2f}]")
    print(f"  e.g. {stations[0]['name']} ({stations[0]['lat']}, {stations[0]['lon']})")

    find_nearest = build_nearest_finder(stations)
    station_names = {s["name"] for s in stations}

    # Fix .con files
    con_files = sorted(txtinout.glob("*.con"))
    print(f"\n  [Layer 1] fix .con wst ({len(con_files)} files)")
    total = 0
    for cf in con_files:
        n = fix_con_file(cf, find_nearest, dry_run=dry_run)
        if n > 0:
            print(f"    {cf.name}: {n} lines {'will change' if dry_run else 'changed'}")
            total += n
        else:
            print(f"    {cf.name}: ok")

    # Check references
    print(f"\n  [Layer 2] weather-sta wgn -> weather-wgn")
    check_wgn_refs(txtinout)
    print(f"\n  [Layer 2] weather-sta -> data files")
    check_data_file_refs(txtinout)
    print(f"\n  [Layer 3] weather-wgn consistency")
    check_wgn_consistency(txtinout, station_names)

    print(f"\n  fix_wst result: ", end="")
    if dry_run:
        print(f"DRY-RUN, {total} lines to change")
    else:
        print(f"DONE, {total} lines changed (.bak saved)")

    return total


# ============================================================
# Main workflow
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SWAT+ simulation - merged version (inject + fix_wst + run, supports daily/hourly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Daily simulation (default step=0, merge old + new stations)
  python run_swat_server.py \\
      --txtinout ./TxtInOut \\
      --weather-dir ./swat_weather \\
      --start 2024-01-01 --end 2024-01-14

  # Hourly simulation (step=24)
  python run_swat_server.py \\
      --txtinout ./TxtInOut \\
      --weather-dir ./swat_weather \\
      --start 2024-01-01 --end 2024-01-14 --step 24

  # Use new stations only (discard old CFSR stations)
  python run_swat_server.py \\
      --txtinout ./TxtInOut \\
      --weather-dir ./swat_weather \\
      --start 2024-01-01 --end 2024-01-14 --station-mode new-only

  # Inject + fix only, do not run
  python run_swat_server.py \\
      --txtinout ./TxtInOut \\
      --weather-dir ./swat_weather \\
      --start 2024-01-01 --end 2024-01-14 --no-run

  # Run fix_wst only (no weather or date arguments needed)
  python run_swat_server.py --txtinout ./TxtInOut --fix-wst-only
  python run_swat_server.py --txtinout ./TxtInOut --fix-wst-only --dry-run
""",
    )

    # Required arguments
    parser.add_argument(
        "--txtinout", type=str, required=True,
        help="path to the SWAT+ TxtInOut project directory",
    )

    # fix_wst-only mode
    grp_fix = parser.add_argument_group("fix_wst options")
    grp_fix.add_argument(
        "--fix-wst-only", action="store_true",
        help="run fix_wst only (do not inject weather, do not run the model)",
    )
    grp_fix.add_argument(
        "--skip-fix-wst", action="store_true",
        help="skip the fix_wst step",
    )
    grp_fix.add_argument(
        "--dry-run", action="store_true",
        help="fix_wst dry-run mode (report only, do not modify files)",
    )

    # Weather source
    grp_wx = parser.add_argument_group("weather data source (choose one)")
    grp_wx.add_argument(
        "--weather-dir", type=str, default=None,
        help="an already-generated swat_weather directory (produced by swat_weather.py)",
    )
    grp_wx.add_argument(
        "--station-mode", type=str, default="merge",
        choices=["merge", "new-only"],
        help="station merge mode: merge=keep old + new stations (default), new-only=use new stations only (discard old stations)",
    )
    grp_wx.add_argument(
        "--country", type=str, default=None,
        help="country name; will call swat_weather.py to download and generate",
    )
    grp_wx.add_argument(
        "--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
        help="custom bounding box",
    )

    # Time
    grp_time = parser.add_argument_group("simulation time")
    grp_time.add_argument("--start", type=str, default=None, help="start date YYYY-MM-DD")
    grp_time.add_argument("--end", type=str, default=None, help="end date YYYY-MM-DD")
    grp_time.add_argument("--days", type=int, default=None, help="number of simulation days (alternative to --end)")
    grp_time.add_argument(
        "--step", type=int, default=0,
        help="time step: 0=daily (default), 24=hourly",
    )

    # Run options
    grp_run = parser.add_argument_group("run options")
    grp_run.add_argument(
        "--swat-exe", type=str, default="swatplus",
        help="path to the SWAT+ executable (default 'swatplus')",
    )
    grp_run.add_argument(
        "--no-run", action="store_true",
        help="configure only, do not run the model",
    )
    grp_run.add_argument(
        "--timeout", type=int, default=7200,
        help="run timeout in seconds (default 7200)",
    )
    grp_run.add_argument(
        "--nyskip", type=int, default=0,
        help="nyskip in print.prt (default 0)",
    )
    grp_run.add_argument(
        "--weather-output-dir", type=str, default="./weather_data",
        help="output directory for weather data download/generation (used in --country mode)",
    )

    args = parser.parse_args()

    txtinout = Path(args.txtinout).resolve()

    # ========================================
    # Mode 1: --fix-wst-only (fix wst only)
    # ========================================
    if args.fix_wst_only:
        if not txtinout.exists():
            print(f"❌ TxtInOut directory does not exist: {txtinout}")
            sys.exit(1)
        run_fix_wst(txtinout, dry_run=args.dry_run)
        return

    # ========================================
    # Mode 2: full workflow (requires --start)
    # ========================================
    if not args.start:
        print("❌ --start must be specified (unless using --fix-wst-only mode)")
        sys.exit(1)

    start_date = date.fromisoformat(args.start)
    if args.end:
        end_date = date.fromisoformat(args.end)
    elif args.days:
        end_date = start_date + timedelta(days=args.days)
    else:
        print("❌ --end or --days must be specified")
        sys.exit(1)

    step_label = "daily (step=0)" if args.step == 0 else f"hourly (step={args.step})"

    print(f"\n{'='*60}")
    print(f"🖥️  SWAT+ simulation (merged version)")
    print(f"{'='*60}")
    print(f"  TxtInOut:  {txtinout}")
    print(f"  Period:    {start_date} → {end_date} ({(end_date - start_date).days} days)")
    print(f"  Time step: {step_label}")
    print(f"  fix_wst:   {'skip' if args.skip_fix_wst else ('dry-run' if args.dry_run else 'enabled')}")
    print(f"  Station mode: {'merge (old + new stations)' if args.station_mode == 'merge' else 'new stations only'}")

    # ---- Check TxtInOut ----
    if not txtinout.exists():
        print(f"\n❌ TxtInOut directory does not exist: {txtinout}")
        sys.exit(1)

    if not validate_txtinout(txtinout):
        sys.exit(1)

    # ---- Weather data ----
    weather_dir = None

    if args.weather_dir:
        weather_dir = Path(args.weather_dir).resolve()
        if not weather_dir.exists():
            print(f"\n❌ weather directory does not exist: {weather_dir}")
            sys.exit(1)
        print(f"  Weather source: {weather_dir}")

    elif args.country or args.bbox:
        print(f"\n📥 Calling swat_weather.py to generate weather data...")
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        try:
            import swat_weather
        except ImportError:
            sys.path.insert(0, "/mnt/user-data/uploads")
            try:
                import swat_weather
            except ImportError:
                print(f"❌ swat_weather.py not found")
                sys.exit(1)

        if args.bbox:
            bbox = tuple(args.bbox)
        else:
            k = args.country.lower().strip()
            if k not in swat_weather.COUNTRY_BBOX:
                print(f"❌ not supported: {k}. Available: {', '.join(swat_weather.COUNTRY_BBOX.keys())}")
                sys.exit(1)
            bbox = swat_weather.COUNTRY_BBOX[k]

        wx_out = Path(args.weather_output_dir)
        nc_files = swat_weather.download_era5(bbox, start_date, end_date, wx_out / "raw_nc")
        if not nc_files:
            print("❌ ERA5 download failed")
            sys.exit(1)

        weather_dir = swat_weather.nc_to_swat(nc_files, str(wx_out), start_date, end_date)
        weather_dir = Path(weather_dir)
        print(f"  Weather source: {weather_dir}")
    else:
        print(f"\n  ⚠️  no weather source specified, assuming weather files already exist in TxtInOut")

    # ---- Step 1: Inject weather files (merge mode) ----
    name_map = {}
    if weather_dir:
        name_map = inject_weather_files(txtinout, weather_dir, station_mode=args.station_mode)
        update_weather_wgn(txtinout, weather_dir)

    # ---- Step 2: Generate .cli index + update file.cio climate line ----
    create_climate_cli_files(txtinout, start_date, end_date)

    # ---- Step 3: Configure time.sim ----
    configure_time_sim(txtinout, start_date, end_date, step=args.step)

    # ---- Step 4: Configure print.prt ----
    configure_print_prt(txtinout, nyskip=args.nyskip)

    # ---- Step 5: Check file.cio ----
    verify_file_cio(txtinout)

    # ---- Step 6: fix_wst (automatically fix wst in .con) ----
    if not args.skip_fix_wst:
        run_fix_wst(txtinout, dry_run=args.dry_run)
    else:
        print(f"\n  ℹ️  --skip-fix-wst: skipping wst fix")

    # ---- Step 7: Run ----
    if args.no_run:
        print(f"\n  ℹ️  --no-run mode: configuration complete, please inspect manually then run:")
        print(f"      cd {txtinout} && {args.swat_exe}")
        return

    # In dry-run mode the model is also not run
    if args.dry_run:
        print(f"\n  ℹ️  --dry-run mode: configuration complete (.con files not modified), not running the model")
        return

    pyswat_ok = run_with_pyswat(txtinout, start_date, end_date, step=args.step)
    if not pyswat_ok:
        success = run_swat(txtinout, args.swat_exe, args.timeout)
        if not success:
            sys.exit(1)

    # ---- Parse outputs ----
    parse_outputs(txtinout)

    print(f"\n{'='*60}")
    print(f"🎉 Done! Output files are in: {txtinout}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
