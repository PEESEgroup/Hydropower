#!/usr/bin/env python3
"""
Batch-run ML training (09_lstm_transformer_prob.py)
==============================================

Iterate over all regions under /datasets/swat_global, run training for each in turn, and save logs.

Usage:
  python batch_train.py                          # run all regions
  python batch_train.py --region brazil/brazil_norte  # run only one
  python batch_train.py --skip-completed          # skip already completed ones
  python batch_train.py --dry-run                 # only list the regions to run
"""

import os
import sys
import subprocess
import argparse
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== Configuration ==============
BASE_DIR = "/datasets/swat_global"

# List of training scripts: executed in order
TRAIN_SCRIPTS = {
    "lstm_transformer": {
        "script": "09_lstm_transformer_prob.py",
        "marker": ".train_lstm_transformer_done",
        "out_dir": "LSTM_Transformer_prob",
    },
    "transformer": {
        "script": "09_transformer_prob.py",
        "marker": ".train_transformer_done",
        "out_dir": "Transformer_prob",
    },
}

TIMEOUT = 3600 * 48          # max training time per region per model (seconds), default 48 hours

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_logs")
os.makedirs(LOG_DIR, exist_ok=True)
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
            region_name = region_dir.name
            dataset_file = region_dir / "datasets" / "ml_dataset_hourly.parquet"
            regions.append({
                "tag": f"{continent_dir.name}/{region_name}",
                "region_path": str(region_dir),
                "has_dataset": dataset_file.exists(),
                "region_dir": region_dir,
            })
    return regions


# =============================================================================
# Completion markers
# =============================================================================
def is_train_done(region: dict, marker_filename: str) -> bool:
    marker = region["region_dir"] / marker_filename
    return marker.exists()


def mark_train_done(region: dict, marker_filename: str, elapsed: float, script: str):
    marker = region["region_dir"] / marker_filename
    try:
        marker.write_text(
            f"completed_at: {datetime.now().isoformat()}\n"
            f"elapsed_seconds: {elapsed:.0f}\n"
            f"script: {script}\n"
        )
    except Exception as e:
        logger.warning(f"  Failed to write marker file: {e}")


# =============================================================================
# Run training
# =============================================================================
def run_training(region: dict, log_path: str, script: str, gpus: str = "0,1", workers: int = 2) -> tuple[bool, float]:
    """Run the training script; returns (success, elapsed_seconds)"""
    cmd = [
        "python", "-u", script,
        "--region", region["region_path"],
        "--gpus", gpus,
        "--workers", str(workers),
    ]

    logger.info(f"  Executing: {' '.join(cmd)}")
    logger.info(f"  Log: {log_path}")

    start = time.time()

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"# Region: {region['tag']}\n")
            log_file.write(f"# Command: {' '.join(cmd)}\n")
            log_file.write(f"# Start: {datetime.now().isoformat()}\n\n")
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

            process.wait(timeout=TIMEOUT)
            returncode = process.returncode

            elapsed = time.time() - start
            log_file.write(f"\n# End: {datetime.now().isoformat()}\n")
            log_file.write(f"# Elapsed: {elapsed:.0f}s\n")
            log_file.write(f"# Return code: {returncode}\n")

        if returncode != 0:
            logger.error(f"  Failed (return code={returncode}), see: {log_path}")
            return False, elapsed
        else:
            logger.info(f"  Completed successfully")
            return True, elapsed

    except subprocess.TimeoutExpired:
        logger.error(f"  Timed out ({TIMEOUT}s)")
        try:
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(5)
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass
        return False, time.time() - start

    except Exception as e:
        logger.error(f"  Exception: {e}")
        return False, time.time() - start


# =============================================================================
# Main function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch-run ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    global TIMEOUT
    parser.add_argument("--region", type=str, default=None,
                        help="Train only the specified region")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(TRAIN_SCRIPTS.keys()),
                        help="Train only the specified model (default: run both)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already completed regions (based on marker files)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list the regions to run, do not execute")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help=f"Timeout in seconds per region per model (default: {TIMEOUT})")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="GPU IDs passed to the training script (default: 0,1)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel threads passed to the training script (default: 2)")
    parser.add_argument("--parallel-regions", type=int, default=1,
                        help="Number of regions to train simultaneously (default: 1). GPUs are split evenly across slots")

    args = parser.parse_args()


    TIMEOUT = args.timeout

    # Determine which models to run
    if args.model:
        models_to_run = {args.model: TRAIN_SCRIPTS[args.model]}
    else:
        models_to_run = TRAIN_SCRIPTS

    logger.info("=" * 60)
    logger.info("Batch ML training")
    logger.info(f"Base directory:   {args.base_dir}")
    logger.info(f"Training models:  {', '.join(models_to_run.keys())}")
    logger.info(f"GPU:        {args.gpus}")
    logger.info(f"Workers:    {args.workers}")
    logger.info(f"Parallel regions: {args.parallel_regions}")
    logger.info(f"Timeout:    {TIMEOUT}s")
    logger.info(f"Skip completed:   {args.skip_completed}")
    logger.info(f"Log directory:    {LOG_DIR}")
    logger.info("=" * 60)

    # Discover regions
    regions = discover_regions(args.base_dir)
    if args.region:
        regions = [r for r in regions if r["tag"] == args.region]

    if not regions:
        logger.error("No regions found")
        sys.exit(1)

    # Filter: keep only regions that have a dataset
    regions_with_data = [r for r in regions if r["has_dataset"]]
    regions_no_data = [r for r in regions if not r["has_dataset"]]

    logger.info(f"\nFound {len(regions)} regions in total")
    logger.info(f"  With dataset: {len(regions_with_data)}")
    logger.info(f"  Without dataset: {len(regions_no_data)}")

    if regions_no_data:
        logger.info(f"\nRegions without a dataset (skipped):")
        for r in regions_no_data:
            logger.info(f"  - {r['tag']}")

    # Statistics
    results = {
        "success": [],
        "failed": [],
        "skipped_done": [],
        "skipped_no_data": [r["tag"] for r in regions_no_data],
    }

    if args.dry_run:
        logger.info(f"\n[DRY-RUN] Regions to be trained:")
        for i, r in enumerate(regions_with_data, 1):
            status_parts = []
            for mname, mcfg in models_to_run.items():
                done = "✅" if is_train_done(r, mcfg["marker"]) else "⬜"
                status_parts.append(f"{mname}:{done}")
            logger.info(f"  [{i}] {r['tag']}  {' '.join(status_parts)}")
        logger.info("Done (dry-run, not executed)")
        return

    start_all = time.time()

    # ---- GPU grouping ----
    all_gpus = [int(g) for g in args.gpus.split(',')]
    n_parallel = min(args.parallel_regions, len(all_gpus))
    if n_parallel < 1:
        n_parallel = 1

    gpu_slots = [[] for _ in range(n_parallel)]
    for idx, g in enumerate(all_gpus):
        gpu_slots[idx % n_parallel].append(g)
    workers_per_slot = max(1, args.workers // n_parallel)

    if n_parallel > 1:
        logger.info(f"\nParallel mode: {n_parallel} slots")
        for s, gs in enumerate(gpu_slots):
            logger.info(f"  Slot {s}: GPUs {gs}, workers={workers_per_slot}")

    # ---- Build the list of items to train (skip completed) ----
    pending = []
    for i, region in enumerate(regions_with_data, 1):
        tag = region["tag"]
        for model_name, model_cfg in models_to_run.items():
            if args.skip_completed and is_train_done(region, model_cfg["marker"]):
                logger.info(f"  ✅ {tag}/{model_name} already completed, skipping")
                results["skipped_done"].append(f"{tag}/{model_name}")
            else:
                pending.append((region, model_name, model_cfg))

    logger.info(f"\nTo train: {len(pending)} region-models, skipped: {len(results['skipped_done'])}")

    # ---- Estimate the number of rivers for each pending item (for load balancing) ----
    def _count_rivers(region):
        ds = Path(region["region_path"]) / "datasets" / "ml_dataset_hourly.parquet"
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(str(ds), columns=["comid"])
            return len(set(t.column("comid").to_pylist()))
        except Exception:
            return 50  # fallback

    results_lock = threading.Lock()
    completed_count = [0]

    def _train_region(region, model_name, model_cfg, slot_gpus, slot_workers):
        tag = region["tag"]
        script = model_cfg["script"]
        marker = model_cfg["marker"]
        gpu_str = ','.join(str(g) for g in slot_gpus)
        safe_name = tag.replace("/", "_")
        log_path = os.path.join(LOG_DIR, f"train_{safe_name}_{model_name}_{timestamp}.log")

        logger.info(f"  🚀 {tag}/{model_name} [GPUs {gpu_str}, w={slot_workers}]")
        ok, elapsed = run_training(region, log_path, script, gpu_str, slot_workers)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        with results_lock:
            completed_count[0] += 1
            progress = f"[{completed_count[0]}/{len(pending)}]"
            if ok:
                logger.info(f"  ✅ {progress} {tag}/{model_name} succeeded ({elapsed_str})")
                results["success"].append((f"{tag}/{model_name}", elapsed_str))
                mark_train_done(region, marker, elapsed, script)
            else:
                logger.error(f"  ❌ {progress} {tag}/{model_name} failed ({elapsed_str})")
                results["failed"].append((f"{tag}/{model_name}", elapsed_str))

    if n_parallel <= 1:
        # ---- Serial mode (backward compatible) ----
        for region, model_name, model_cfg in pending:
            _train_region(region, model_name, model_cfg, all_gpus, args.workers)
    else:
        # ---- Parallel mode: greedy bin-packing, balance across N queues by river count ----
        river_counts = [_count_rivers(item[0]) for item in pending]
        sorted_idx = sorted(range(len(pending)), key=lambda i: river_counts[i], reverse=True)
        queues = [[] for _ in range(n_parallel)]
        slot_load = [0] * n_parallel
        for idx in sorted_idx:
            lightest = min(range(n_parallel), key=lambda s: slot_load[s])
            queues[lightest].append(pending[idx])
            slot_load[lightest] += river_counts[idx]
        for s in range(n_parallel):
            logger.info(f"  Slot {s}: {len(queues[s])} regions, ~{slot_load[s]} rivers")

        def _process_queue(slot_id):
            for region, model_name, model_cfg in queues[slot_id]:
                _train_region(region, model_name, model_cfg,
                              gpu_slots[slot_id], workers_per_slot)

        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(_process_queue, s): s
                       for s in range(n_parallel)}
            for fut in as_completed(futures):
                slot = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"  Slot {slot} exception: {e}")

    # Summary
    total_elapsed = time.time() - start_all
    total_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))

    logger.info(f"\n{'='*60}")
    logger.info("Summary report")
    logger.info(f"{'='*60}")
    logger.info(f"Total elapsed:      {total_str}")
    logger.info(f"Succeeded:          {len(results['success'])}")
    logger.info(f"Failed:             {len(results['failed'])}")
    logger.info(f"Skipped(completed): {len(results['skipped_done'])}")
    logger.info(f"Skipped(no data):   {len(results['skipped_no_data'])}")

    if results["success"]:
        logger.info(f"\nSucceeded regions:")
        for tag, t in results["success"]:
            logger.info(f"  ✅ {tag} ({t})")

    if results["failed"]:
        logger.info(f"\nFailed regions:")
        for tag, t in results["failed"]:
            logger.info(f"  ❌ {tag} ({t})")

    # Save summary
    summary_path = os.path.join(LOG_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Batch ML training summary - {datetime.now().isoformat()}\n")
        f.write(f"Total elapsed: {total_str}\n\n")
        f.write(f"Succeeded: {len(results['success'])}\n")
        for tag, t in results["success"]:
            f.write(f"  {tag} ({t})\n")
        f.write(f"\nFailed: {len(results['failed'])}\n")
        for tag, t in results["failed"]:
            f.write(f"  {tag} ({t})\n")
        f.write(f"\nSkipped(completed): {len(results['skipped_done'])}\n")
        for tag in results["skipped_done"]:
            f.write(f"  {tag}\n")
        f.write(f"\nSkipped(no data): {len(results['skipped_no_data'])}\n")
        for tag in results["skipped_no_data"]:
            f.write(f"  {tag}\n")

    logger.info(f"\nSummary: {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
