#!/usr/bin/env python3
"""
Script to batch-download weather data and inject it into SWAT models.
Iterates over all regions under /datasets/swat_global and runs, in order:
  1. Download weather (swat_weather_download.py)
  2. Inject files (run_swat_server.py)

Enhanced features:
  - Automatically detects which regions have not yet completed step 1 (download + format conversion)
  - Automatically retries incomplete regions until all are done or the maximum number of retry rounds is reached
  - Checks file completeness under the swat_format directory (.cli files and station data files)
"""

import os
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# ============== Configuration parameters ==============
BASE_DIR = "/datasets/swat_global"
START_DATE = "2007-01-01"
END_DATE = "2019-12-31"
GRID_SPACING = "0.2"
CHUNK_DAYS = "20"
SOURCE = "gee"
NYSKIP = "2"
STEP = "24"
WORKER = "10"

# Retry configuration
MAX_RETRY_ROUNDS = 5          # Maximum number of retry rounds
RETRY_WAIT_SECONDS = 30       # Seconds to wait before each retry round (buffer for the GEE API)
INITIAL_WORKERS = 10
# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swat_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Main log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"main_{timestamp}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Completeness check: determine whether a region's step 1 is already done
# =============================================================================
REQUIRED_CLI_FILES = ["pcp.cli", "tmp.cli", "slr.cli", "hmd.cli", "wnd.cli",
                      "weather-sta.cli"]
REQUIRED_EXTENSIONS = [".pcp", ".tmp", ".slr", ".hmd", ".wnd"]

def hard_cleanup():
    """
    Last-resort cleanup line: use system commands to forcibly kill all related
    leftover processes. Prevents OOM from leaving unrecoverable zombie/orphan
    processes in the Python process pool.
    """
    logger.info("  🧹 Performing last-resort physical cleanup, reclaiming memory...")
    import subprocess
    import time
    
    try:
        subprocess.run(["pkill", "-9", "-f", "swat_weather_download.py"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "run_swat_server.py"], stderr=subprocess.DEVNULL)
        time.sleep(2)
    except Exception as e:
        logger.warning(f"  Exception during cleanup: {e}")


def _cleanup_incomplete_region(region_info: dict):
    """
    Clean up incomplete files left behind in a region after a process was killed
    (-9/OOM).

    Key point: delete the final merged cache files under raw_nc
    (era5land_gee_*.nc / era5_pc_*.nc); otherwise the download script will see
    the cache on retry and skip directly, so it can never be repaired.
    Also delete the incomplete swat_format directory to prevent it from being
    mistakenly judged as complete.
    """
    region_path = Path(region_info["dem"]).parent.parent
    tag = f"{region_info['continent']}/{region_info['region']}"
    
    # 1. Clean up the final merged cache in raw_nc (chunk files are kept and can be reused on retry)
    nc_dir = region_path / "weather" / "raw_nc"
    if nc_dir.exists():
        # Final cache file name format: era5land_gee_YYYYMM_YYYYMM.nc or era5_pc_YYYYMM_YYYYMM.nc
        # chunk file name format: era5land_gee_chunk*.nc
        for f in nc_dir.glob("era5*.nc"):
            if "chunk" not in f.name:
                logger.info(f"  🗑️  Deleting incomplete cache: {f.name}")
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"  Failed to delete {f.name}: {e}")

    # 2. Clean up the incomplete swat_format directory
    swat_dir = Path(region_info["weather_dir"])
    if swat_dir.exists():
        is_ok, _ = check_swat_format_complete(region_info)
        if not is_ok:
            import shutil
            logger.info(f"  🗑️  Deleting incomplete swat_format directory")
            try:
                shutil.rmtree(swat_dir)
            except Exception as e:
                logger.warning(f"  Failed to delete: {e}")

def check_swat_format_complete(region_info: dict) -> tuple[bool, str]:
    """
    Check whether a region's weather/swat_format directory contains complete
    SWAT-format files.

    Checks:
      1. Whether the swat_format directory exists
      2. Whether all required .cli files are present
      3. Whether all station data files listed in each .cli file exist
      4. Whether the station data files are non-empty
      5. Whether the number of stations listed in pcp.cli is > 0

    Returns (is_complete, reason)
    """
    weather_dir = Path(region_info["weather_dir"])

    # 1. Does the directory exist?
    if not weather_dir.exists():
        return False, "swat_format directory does not exist"

    # 2. Do all .cli files exist?
    for cli_name in REQUIRED_CLI_FILES:
        cli_path = weather_dir / cli_name
        if not cli_path.exists():
            return False, f"Missing {cli_name}"
        if cli_path.stat().st_size == 0:
            return False, f"{cli_name} is an empty file"

    # 3. Read pcp.cli and parse the station list
    pcp_cli = weather_dir / "pcp.cli"
    try:
        with open(pcp_cli, "r") as f:
            lines = f.readlines()
        # First line is a comment, second line is "filename", then station file names
        station_files = [l.strip() for l in lines[2:] if l.strip()]
    except Exception as e:
        return False, f"Failed to read pcp.cli: {e}"

    if len(station_files) == 0:
        return False, "No stations listed in pcp.cli"

    # 4. Check that all variable files for each station exist and are non-empty
    missing_count = 0
    empty_count = 0
    for pcp_file in station_files:
        station_id = pcp_file.replace(".pcp", "")
        for ext in REQUIRED_EXTENSIONS:
            data_file = weather_dir / f"{station_id}{ext}"
            if not data_file.exists():
                missing_count += 1
            elif data_file.stat().st_size == 0:
                empty_count += 1
    
    if missing_count > 0:
        return False, f"Missing {missing_count} station data files"
    if empty_count > 0:
        return False, f"{empty_count} station data files are empty"

    # 5. Spot-check whether the line count of the first station file is reasonable
    # For 2013-2019 (7 years), daily data is about 2557 lines (including 3 header lines); hourly data is more
    first_station = station_files[0].replace(".pcp", "")
    tmp_file = weather_dir / f"{first_station}.tmp"
    try:
        with open(tmp_file, "r") as f:
            line_count = sum(1 for _ in f)
        # 3 header lines + at least 365 days of data
        if line_count < 368:
            return False, f"Too few station data lines ({line_count} lines); data may be incomplete"
    except Exception as e:
        return False, f"Failed to read station file: {e}"

    return True, f"Complete ({len(station_files)} stations, {line_count - 3} days of data)"


def check_nc_files_exist(region_info: dict) -> tuple[bool, str]:
    """
    Check whether the raw_nc directory contains NC files that were downloaded
    but may not have been converted.
    If so, the download succeeded but the conversion may have failed, and you can
    reconvert with --convert-only.
    """
    region_path = Path(region_info["dem"]).parent.parent
    nc_dir = region_path / "weather" / "raw_nc"

    if not nc_dir.exists():
        return False, "raw_nc directory does not exist"

    nc_files = list(nc_dir.glob("era5*.nc"))
    if not nc_files:
        return False, "No NC files in the raw_nc directory"

    # Simple size check
    total_size = sum(f.stat().st_size for f in nc_files)
    if total_size < 1024:  # < 1KB, possibly a corrupted file
        return False, f"NC file total size is abnormal ({total_size} bytes)"

    return True, f"Found {len(nc_files)} NC files ({total_size / 1048576:.1f} MB)"


def check_inject_complete(region_info: dict) -> tuple[bool, str]:
    """
    Check whether a region's step 2 (injection) is already done.
    Criterion: whether .pcp files exist in the TxtInOut directory (injected from swat_format).
    """
    txtinout = Path(region_info["txtinout"])

    if not txtinout.exists():
        return False, "TxtInOut directory does not exist"

    injected_pcp = list(txtinout.glob("*.pcp"))
    if not injected_pcp:
        return False, "No .pcp files in TxtInOut (not injected)"

    return True, f"Injected ({len(injected_pcp)} stations)"


def check_txtinout_complete(region_info: dict) -> tuple[bool, str]:
    """
    Check whether the weather files in a region's TxtInOut directory have been
    fully injected.

    Checks:
      1. Whether the TxtInOut directory exists
      2. Whether all required .cli files are present
      3. Whether all station data files listed in each .cli file exist in TxtInOut
      4. Whether the station data files are non-empty
      5. Spot-check whether the station data line count is reasonable

    If everything passes, it means injection previously succeeded and there is no
    need to download + inject again.
    Returns (is_complete, reason)
    """
    txtinout = Path(region_info["txtinout"])

    # 1. Does the TxtInOut directory exist?
    if not txtinout.exists():
        return False, "TxtInOut directory does not exist"

    # 2. Do all .cli files exist?
    for cli_name in REQUIRED_CLI_FILES:
        cli_path = txtinout / cli_name
        if not cli_path.exists():
            return False, f"Missing {cli_name} in TxtInOut"
        if cli_path.stat().st_size == 0:
            return False, f"{cli_name} is an empty file in TxtInOut"

    # 3. Read pcp.cli and parse the station list
    pcp_cli = txtinout / "pcp.cli"
    try:
        with open(pcp_cli, "r") as f:
            lines = f.readlines()
        station_files = [l.strip() for l in lines[2:] if l.strip()]
    except Exception as e:
        return False, f"Failed to read TxtInOut/pcp.cli: {e}"

    if len(station_files) == 0:
        return False, "No stations listed in TxtInOut/pcp.cli"

    # 4. Check that all variable files for each station exist and are non-empty
    missing_count = 0
    empty_count = 0
    for pcp_file in station_files:
        station_id = pcp_file.replace(".pcp", "")
        for ext in REQUIRED_EXTENSIONS:
            data_file = txtinout / f"{station_id}{ext}"
            if not data_file.exists():
                missing_count += 1
            elif data_file.stat().st_size == 0:
                empty_count += 1

    if missing_count > 0:
        return False, f"Missing {missing_count} station data files in TxtInOut"
    if empty_count > 0:
        return False, f"{empty_count} station data files are empty in TxtInOut"

    # 5. Spot-check whether the line count of the first station file is reasonable
    first_station = station_files[0].replace(".pcp", "")
    tmp_file = txtinout / f"{first_station}.tmp"
    try:
        with open(tmp_file, "r") as f:
            line_count = sum(1 for _ in f)
        if line_count < 368:
            return False, f"Too few TxtInOut station data lines ({line_count} lines); may be incomplete"
    except Exception as e:
        return False, f"Failed to read TxtInOut station file: {e}"

    return True, f"TxtInOut is complete ({len(station_files)} stations, {line_count - 3} days of data)"


def cleanup_swat_format(region_info: dict) -> bool:
    """
    Delete the weather/swat_format directory to free up disk space.
    Only call after confirming that TxtInOut injection is complete.
    Returns True if deletion succeeded or the directory does not exist.
    """
    import shutil
    swat_dir = Path(region_info["weather_dir"])
    tag = f"{region_info['continent']}/{region_info['region']}"

    if not swat_dir.exists():
        return True

    try:
        # Compute size for logging
        total_size = sum(f.stat().st_size for f in swat_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        shutil.rmtree(swat_dir)
        logger.info(f"  🗑️  Deleted swat_format directory (freed {size_mb:.1f} MB): {swat_dir}")
        return True
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to delete swat_format directory ({tag}): {e}")
        return False


def discover_regions(base_dir: str) -> list[dict]:
    """
    Iterate over base_dir and find all regions containing rasters/dem.tif.
    Directory structure: base_dir / <continent> / <region> / rasters / dem.tif
    """
    regions = []
    base = Path(base_dir)
    if not base.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return regions

    for continent_dir in sorted(base.iterdir()):
        if not continent_dir.is_dir():
            continue
        for region_dir in sorted(continent_dir.iterdir()):
            if not region_dir.is_dir():
                continue
            dem_path = region_dir / "rasters" / "dem.tif"
            region_name = region_dir.name
            txtinout_path = (
                region_dir / region_name / "Scenarios" / "Default" / "TxtInOut"
            )
            weather_dir = region_dir / "weather" / "swat_format"

            regions.append(
                {
                    "continent": continent_dir.name,
                    "region": region_name,
                    "dem": str(dem_path),
                    "txtinout": str(txtinout_path),
                    "weather_dir": str(weather_dir),
                    "has_dem": dem_path.exists(),
                    "has_txtinout": txtinout_path.exists(),
                }
            )
    return regions


def kill_process_group(pgid: int):
    """Terminate the entire process group to ensure all child processes (workers) are cleaned up"""
    import signal
    try:
        os.killpg(pgid, signal.SIGTERM)
        logger.info(f"  Sent SIGTERM to process group {pgid}")
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"  SIGTERM failed: {e}, trying SIGKILL...")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass


def wait_and_cleanup(process: subprocess.Popen, timeout: int, log_file) -> int:
    """Wait for the process to finish; on timeout, forcibly terminate the entire process group."""
    import signal
    import time

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"  Timeout ({timeout}s), forcibly terminating process group...")
        kill_process_group(os.getpgid(process.pid))
        process.wait(timeout=30)

    try:
        pgid = process.pid
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    return process.returncode


def run_command(cmd: list[str], log_path: str, description: str) -> bool:
    """Run a command and save the output to a log file. Returns True on success."""
    logger.info(f"  Running: {' '.join(cmd)}")
    logger.info(f"  Log: {log_path}")

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"# {description}\n")
            log_file.write(f"# Command: {' '.join(cmd)}\n")
            log_file.write(f"# Start time: {datetime.now().isoformat()}\n\n")
            log_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()

            returncode = wait_and_cleanup(
                process, timeout=3600 * 12, log_file=log_file
            )

            log_file.write(f"\n# End time: {datetime.now().isoformat()}\n")
            log_file.write(f"# Return code: {returncode}\n")

        if returncode != 0:
            logger.error(f"  Failed (return code={returncode}), see: {log_path}")
            return False
        else:
            logger.info(f"  Completed successfully")
            return True

    except Exception as e:
        logger.error(f"  Exception: {e}")
        try:
            kill_process_group(os.getpgid(process.pid))
        except Exception:
            pass
        return False


def step1_download(region_info: dict, worker_count: int) -> bool:
    """Step 1: Download weather data"""
    tag = f"{region_info['continent']}/{region_info['region']}"
    log_path = os.path.join(
        LOG_DIR,
        f"download_{region_info['continent']}_{region_info['region']}_{timestamp}.log",
    )

    cmd = [
        "python", "-u", "swat_weather_download.py",
        "--dem", region_info["dem"],
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--grid-spacing", GRID_SPACING,
        "--chunk-days", CHUNK_DAYS,
        "--hourly-pcp",
        "--delete-raw",
        "--source", SOURCE,
        "--workers", str(worker_count),  # ⚠️ Dynamically pass in the current worker count, using the plural flag
    ]

    return run_command(cmd, log_path, f"Download weather - {tag} (Workers: {worker_count})")


def step2_inject(region_info: dict) -> bool:
    """Step 2: Inject weather files into the SWAT model"""
    tag = f"{region_info['continent']}/{region_info['region']}"
    log_path = os.path.join(
        LOG_DIR,
        f"inject_{region_info['continent']}_{region_info['region']}_{timestamp}.log",
    )

    cmd = [
        "python", "-u", "run_swat_server.py",
        "--txtinout", region_info["txtinout"],
        "--weather-dir", region_info["weather_dir"],
        "--start", START_DATE,
        "--end", END_DATE,
        "--nyskip", NYSKIP,
        "--no-run",
        "--station-mode", "new-only",
        "--step", STEP,
    ]

    return run_command(cmd, log_path, f"Inject weather - {tag}")


def scan_incomplete_regions(regions: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Scan all regions and classify them as:
      - complete: step 1 already done
      - incomplete: step 1 not done, needs re-download
      - skipped: missing prerequisites such as DEM, cannot be processed
    """
    complete = []
    incomplete = []
    skipped = []
    
    for region in regions:
        tag = f"{region['continent']}/{region['region']}"
        
        if not region["has_dem"]:
            skipped.append(region)
            continue
        
        is_ok, reason = check_swat_format_complete(region)
        if is_ok:
            complete.append(region)
            logger.info(f"  ✅ {tag}: {reason}")
        else:
            incomplete.append(region)
            has_nc, nc_info = check_nc_files_exist(region)
            nc_hint = f" (💡 {nc_info})" if has_nc else ""
            logger.info(f"  ❌ {tag}: {reason}{nc_hint}")
    
    return complete, incomplete, skipped


def main():
    import time as _time
    
    logger.info("=" * 60)
    logger.info("SWAT weather batch processing (vertical pipeline: steps 1+2 run in sequence per region)")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Date range: {START_DATE} ~ {END_DATE}")
    logger.info(f"Maximum retry rounds: {MAX_RETRY_ROUNDS}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info("=" * 60)

    # Discover all regions
    regions = discover_regions(BASE_DIR)
    if not regions:
        logger.error("No regions found; please check the directory structure")
        sys.exit(1)

    logger.info(f"\nFound {len(regions)} regions in total\n")

    # Result statistics
    results = {
        "download_ok": [],
        "download_fail": [],
        "inject_ok": [],
        "inject_fail": [],
        "skipped": [],
        "txtinout_already_complete": [],  # Regions where TxtInOut is already complete, skipping download+injection
    }

    # ================================================================
    # Region iteration loop: each region independently runs step 1 (with retries) + step 2
    # ================================================================
    for i, region in enumerate(regions, 1):
        tag = f"{region['continent']}/{region['region']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"▶️ Processing region [{i}/{len(regions)}]: {tag}")
        logger.info(f"{'='*60}")

        # 0. Check the DEM prerequisite
        if not region["has_dem"]:
            logger.warning(f"  ⏭️ Skipping {tag} (no DEM file)")
            results["skipped"].append(tag)
            continue

        # ==========================================
        # Pre-check: whether TxtInOut is already fully injected
        # If complete, skip download+injection and clean up swat_format directly
        # ==========================================
        txtinout_ok, txtinout_reason = check_txtinout_complete(region)
        if txtinout_ok:
            logger.info(f"  ✅ [Skip] TxtInOut already complete, no download or injection needed: {txtinout_reason}")
            results["download_ok"].append(tag)
            results["inject_ok"].append(tag)
            results["txtinout_already_complete"].append(tag)
            # Clean up swat_format to free disk space
            cleanup_swat_format(region)
            continue

        logger.info(f"  🔍 TxtInOut incomplete: {txtinout_reason}, download+injection pipeline needed")

        # ==========================================
        # Step 1: Download and format conversion (with independent retry mechanism)
        # ==========================================
        step1_ok, reason = check_swat_format_complete(region)

        if step1_ok:
            logger.info(f"  ✅ [Step 1] Already done: {reason}")
            results["download_ok"].append(tag)
        else:
            logger.info(f"  🔶 [Step 1] Not done, preparing to download ({reason})")

            # Retry loop for a single region
            for retry_round in range(1, MAX_RETRY_ROUNDS + 1):
                current_workers = max(1, INITIAL_WORKERS // (2 ** (retry_round - 1)))

                if retry_round > 1:
                    logger.info(f"  ⏳ Waiting {RETRY_WAIT_SECONDS} seconds before retrying...")
                    _time.sleep(RETRY_WAIT_SECONDS)

                logger.info(f"  🔄 Download round {retry_round}/{MAX_RETRY_ROUNDS} (Workers: {current_workers})")

                # Run the download
                step1_download(region, worker_count=current_workers)

                # Re-verify file completeness
                step1_ok, reason = check_swat_format_complete(region)
                if step1_ok:
                    logger.info(f"  ✅ [Step 1] Download and conversion succeeded: {reason}")
                    results["download_ok"].append(tag)
                    break
                else:
                    logger.warning(f"  ❌ [Step 1] This round failed: {reason}")
                    _cleanup_incomplete_region(region)
                    hard_cleanup()

            # If it still fails after the maximum retries, give up on this region's subsequent steps
            if not step1_ok:
                logger.error(f"  ❌ [Step 1] Still failing after {MAX_RETRY_ROUNDS} retry rounds, skipping this region")
                results["download_fail"].append(tag)
                continue  # Skip step 2 and move on to the next region

        # ==========================================
        # Step 2: Inject weather files into the model
        # ==========================================
        if not region["has_txtinout"]:
            logger.warning(f"  ⏭️ Skipping [Step 2] (TxtInOut does not exist)")
            results["skipped"].append(f"{tag} (no TxtInOut)")
            continue

        # The check_inject_complete check was removed; injection is always run directly
        logger.info(f"  🔶 [Step 2] Starting forced injection...")
        inject_ok = step2_inject(region)

        if inject_ok:
            logger.info(f"  ✅ [Step 2] Injection succeeded")
            results["inject_ok"].append(tag)
            # After successful injection, verify TxtInOut completeness, then delete swat_format to free disk
            verify_ok, verify_reason = check_txtinout_complete(region)
            if verify_ok:
                logger.info(f"  ✅ [Verify] Post-injection TxtInOut completeness confirmed: {verify_reason}")
                cleanup_swat_format(region)
            else:
                logger.warning(f"  ⚠️  [Verify] Post-injection TxtInOut completeness check failed: {verify_reason}, keeping swat_format")
        else:
            logger.error(f"  ❌ [Step 2] Injection failed")
            results["inject_fail"].append(tag)

        # After each region is processed, perform a thorough memory and process cleanup
        hard_cleanup()

    # ================================================================
    # Summary report
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("Summary report")
    logger.info(f"{'='*60}")
    logger.info(f"Total regions:     {len(regions)}")
    logger.info(f"TxtInOut already complete (skipped): {len(results['txtinout_already_complete'])}")
    logger.info(f"Download succeeded:     {len(results['download_ok'])}")
    logger.info(f"Download failed:     {len(results['download_fail'])}")
    logger.info(f"Injection succeeded:     {len(results['inject_ok'])}")
    logger.info(f"Injection failed:     {len(results['inject_fail'])}")
    logger.info(f"Skipped:         {len(results['skipped'])}")

    if results["download_fail"]:
        logger.info("\nRegions with failed downloads (reached max retry limit):")
        for t in results["download_fail"]:
            logger.info(f"  - {t}")
    if results["inject_fail"]:
        logger.info("\nRegions with failed injection:")
        for t in results["inject_fail"]:
            logger.info(f"  - {t}")
    if results["skipped"]:
        logger.info("\nSkipped regions:")
        for t in results["skipped"]:
            logger.info(f"  - {t}")

    # Save the summary to a file
    summary_path = os.path.join(LOG_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"SWAT batch processing summary - {datetime.now().isoformat()}\n")
        f.write(f"Maximum retry rounds setting: {MAX_RETRY_ROUNDS}\n")
        f.write(f"Total regions: {len(regions)}\n")
        f.write(f"TxtInOut already complete (skipped download+injection): {len(results['txtinout_already_complete'])}\n")
        f.write(f"Download succeeded: {len(results['download_ok'])}\n")
        f.write(f"Download failed: {len(results['download_fail'])}\n")
        f.write(f"Injection succeeded: {len(results['inject_ok'])}\n")
        f.write(f"Injection failed: {len(results['inject_fail'])}\n")
        f.write(f"Skipped: {len(results['skipped'])}\n\n")

        for key, label in [
            ("download_fail", "Download failed"),
            ("inject_fail", "Injection failed"),
            ("skipped", "Skipped"),
        ]:
            if results[key]:
                f.write(f"\n{label}:\n")
                for t in results[key]:
                    f.write(f"  {t}\n")

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
