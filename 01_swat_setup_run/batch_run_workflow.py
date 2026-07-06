#!/usr/bin/env python3
"""
Script for batch-running run_workflow.py.
Iterates over all regions under /datasets/swat_global and runs in turn:
  python run_workflow.py --region <continent>/<region> --split-mode contiguous

Features:
  - Automatically discovers all regions
  - Saves each region's output to its own log file
  - Records global progress in the main log
  - Supports skipping regions that already completed successfully (resume)
  - Summary report
"""

import os
import subprocess
import sys
import signal
import logging
import time
from pathlib import Path
from datetime import datetime

# ============== Configuration parameters ==============
BASE_DIR = "/datasets/swat_global"
SPLIT_MODE = "contiguous"
TIMEOUT = 3600 * 24          # Max runtime per region (seconds), default 24 hours
SKIP_COMPLETED = True         # Whether to skip regions already completed (based on marker file)
MARKER_FILENAME = ".workflow_done"  # Marker file name for successful completion

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_logs")
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
# Region discovery
# =============================================================================
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

            regions.append(
                {
                    "continent": continent_dir.name,
                    "region": region_name,
                    "region_tag": f"{continent_dir.name}/{region_name}",
                    "region_dir": str(region_dir),
                    "dem": str(dem_path),
                    "txtinout": str(txtinout_path),
                    "has_dem": dem_path.exists(),
                    "has_txtinout": txtinout_path.exists(),
                }
            )
    return regions


# =============================================================================
# Process management
# =============================================================================
def kill_process_group(pgid: int):
    """Terminate the entire process group"""
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


def wait_and_cleanup(process: subprocess.Popen, timeout: int) -> int:
    """Wait for the process to finish; on timeout, force-kill the entire process group."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"  Timed out ({timeout}s), force-killing process group...")
        try:
            kill_process_group(os.getpgid(process.pid))
        except Exception:
            pass
        process.wait(timeout=30)

    # Ensure the child process group is thoroughly cleaned up
    try:
        pgid = process.pid
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    return process.returncode


# =============================================================================
# Command execution
# =============================================================================
def run_command(cmd: list[str], log_path: str, description: str) -> bool:
    """Run a command, printing output to both the terminal and the log file. Returns True on success."""
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

            returncode = wait_and_cleanup(process, timeout=TIMEOUT)

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


# =============================================================================
# Completion marker management (resume)
# =============================================================================
def is_workflow_done(region_info: dict) -> bool:
    """Check whether the workflow has already completed successfully for this region"""
    marker = Path(region_info["region_dir"]) / MARKER_FILENAME
    return marker.exists()


def mark_workflow_done(region_info: dict):
    """Write the completion marker file"""
    marker = Path(region_info["region_dir"]) / MARKER_FILENAME
    try:
        marker.write_text(
            f"completed_at: {datetime.now().isoformat()}\n"
            f"split_mode: {SPLIT_MODE}\n"
        )
    except Exception as e:
        logger.warning(f"  Failed to write marker file: {e}")


# =============================================================================
# Main function
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("SWAT batch workflow run")
    logger.info(f"Base directory:   {BASE_DIR}")
    logger.info(f"split-mode: {SPLIT_MODE}")
    logger.info(f"Timeout:   {TIMEOUT}s")
    logger.info(f"Skip completed: {SKIP_COMPLETED}")
    logger.info(f"Log directory:   {LOG_DIR}")
    logger.info("=" * 60)

    # Discover all regions
    regions = discover_regions(BASE_DIR)
    if not regions:
        logger.error("No regions found, please check the directory structure")
        sys.exit(1)

    logger.info(f"\nDiscovered {len(regions)} regions in total\n")

    # Result statistics
    results = {
        "success": [],
        "failed": [],
        "skipped_no_dem": [],
        "skipped_no_txtinout": [],
        "skipped_already_done": [],
    }

    start_time = time.time()

    for i, region in enumerate(regions, 1):
        tag = region["region_tag"]
        logger.info(f"\n{'='*60}")
        logger.info(f"▶️ [{i}/{len(regions)}] {tag}")
        logger.info(f"{'='*60}")

        # Precondition checks
        if not region["has_dem"]:
            logger.warning(f"  ⏭️ Skipped (no DEM file)")
            results["skipped_no_dem"].append(tag)
            continue

        if not region["has_txtinout"]:
            logger.warning(f"  ⏭️ Skipped (no TxtInOut directory)")
            results["skipped_no_txtinout"].append(tag)
            continue

        # Resume: skip regions that have already completed
        if SKIP_COMPLETED and is_workflow_done(region):
            logger.info(f"  ✅ Already completed, skipping")
            results["skipped_already_done"].append(tag)
            continue

        # Build the command
        cmd = [
            "python", "-u", "run_workflow.py",
            "--region", tag,
            "--split-mode", SPLIT_MODE,
        ]

        # Log path: workflow_logs/workflow_<continent>_<region>_<timestamp>.log
        log_path = os.path.join(
            LOG_DIR,
            f"workflow_{region['continent']}_{region['region']}_{timestamp}.log",
        )

        # Execute
        ok = run_command(cmd, log_path, f"Workflow - {tag}")

        if ok:
            logger.info(f"  ✅ Workflow succeeded: {tag}")
            results["success"].append(tag)
            mark_workflow_done(region)
        else:
            logger.error(f"  ❌ Workflow failed: {tag}")
            results["failed"].append(tag)

    # ================================================================
    # Summary report
    # ================================================================
    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    logger.info(f"\n{'='*60}")
    logger.info("Summary report")
    logger.info(f"{'='*60}")
    logger.info(f"Total regions:         {len(regions)}")
    logger.info(f"Succeeded:             {len(results['success'])}")
    logger.info(f"Failed:             {len(results['failed'])}")
    logger.info(f"Skipped (completed):     {len(results['skipped_already_done'])}")
    logger.info(f"Skipped (no DEM):      {len(results['skipped_no_dem'])}")
    logger.info(f"Skipped (no TxtInOut): {len(results['skipped_no_txtinout'])}")
    logger.info(f"Total elapsed:           {elapsed_str}")

    if results["failed"]:
        logger.info("\nFailed regions:")
        for t in results["failed"]:
            logger.info(f"  - {t}")

    # Save the summary file
    summary_path = os.path.join(LOG_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"SWAT Workflow batch processing summary - {datetime.now().isoformat()}\n")
        f.write(f"split-mode: {SPLIT_MODE}\n")
        f.write(f"Total elapsed: {elapsed_str}\n\n")
        f.write(f"Total regions:           {len(regions)}\n")
        f.write(f"Succeeded:             {len(results['success'])}\n")
        f.write(f"Failed:             {len(results['failed'])}\n")
        f.write(f"Skipped (completed):     {len(results['skipped_already_done'])}\n")
        f.write(f"Skipped (no DEM):      {len(results['skipped_no_dem'])}\n")
        f.write(f"Skipped (no TxtInOut): {len(results['skipped_no_txtinout'])}\n")

        for key, label in [
            ("success", "Succeeded"),
            ("failed", "Failed"),
            ("skipped_already_done", "Skipped (completed)"),
            ("skipped_no_dem", "Skipped (no DEM)"),
            ("skipped_no_txtinout", "Skipped (no TxtInOut)"),
        ]:
            if results[key]:
                f.write(f"\n{label}:\n")
                for t in results[key]:
                    f.write(f"  {t}\n")

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
