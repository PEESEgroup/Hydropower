"""F_metrics_driver.py - run F1/F2/F3/F4 impact metrics across the whole matrix.

Discovers every output dir that has a bal_hourly_arrays.npz (the shared base all
F programs need) under /datasets/swat_global/**/revub_output/<gcm>/<ssp>[/<tag>],
parses (region_path, gcm, ssp, tag) from the path, and runs each F program by
importing its module once per worker and calling run() - no per-dir subprocess
import overhead.

Each F program writes its own parquet into the same output dir:
  F1 -> impact_sediment_metrics.parquet
  F2 -> impact_ecology_metrics.parquet
  F3 -> impact_supply_metrics.parquet     (skipped if no supply_segments.csv)
  F4 -> impact_carbon_metrics.parquet     (skipped if no emission_factors.csv)

A manifest of per-(dir,program) status is written to F_metrics_manifest.csv.

Env:
  REVUB_F_WORKERS  process pool size (default 24)
  REVUB_F_TAGS     comma list to restrict tags, e.g. "flat" or "flat,typical"
                   (default: all tags found, including the untagged "" set)
  REVUB_F_PROGRAMS comma list of F programs to run (default "F1,F2,F3,F4")
  REVUB_F_ROOT     dataset root (default /datasets/swat_global)
"""
import glob
import os
import sys
import time
import traceback
from multiprocessing import Pool

import pandas as pd

ROOT = os.environ.get('REVUB_F_ROOT', '/datasets/swat_global')
WORKERS = int(os.environ.get('REVUB_F_WORKERS', '24'))
# Default to the 'flat' tag only: that is the canonical full A-C-D run carrying
# the data-center scenarios (dc_scenario_A/B/C + dc_hydro_summary). The 'typical'
# tag is A-C only (no DC) and the untagged set is a superseded older D run.
# Pass REVUB_F_TAGS="flat,typical" or "" (all) to override.
TAGS = os.environ.get('REVUB_F_TAGS', 'flat').strip()
TAG_FILTER = set(t.strip() for t in TAGS.split(',')) if TAGS else None
PROGRAMS = [p.strip() for p in
            os.environ.get('REVUB_F_PROGRAMS', 'F1,F2,F3,F4').split(',') if p.strip()]
REGION_SUBSTR = os.environ.get('REVUB_F_REGION', '').strip()  # path substring filter (test/resume)
# REVUB_F_REGIONS: comma list of EXACT region names to restrict to (e.g. E exporters)
_RLIST = os.environ.get('REVUB_F_REGIONS', '').strip()
REGION_LIST = set(x.strip() for x in _RLIST.split(',') if x.strip()) if _RLIST else None

# program key -> (module name, run-callable name, output parquet)
_PROG = {
    'F1': ('F1_sediment_metrics', 'run', 'impact_sediment_metrics.parquet'),
    'F2': ('F2_ecology_metrics', 'run', 'impact_ecology_metrics.parquet'),
    'F3': ('F3_supply_metrics', 'run', 'impact_supply_metrics.parquet'),
    'F4': ('F4_carbon_metrics', 'run', 'impact_carbon_metrics.parquet'),
}


def discover():
    """Return list of (region_path, gcm, ssp, tag) from every base npz."""
    pat = os.path.join(ROOT, '*', '*', 'revub_output', '*', '*',
                       'bal_hourly_arrays.npz')
    pat_tag = os.path.join(ROOT, '*', '*', 'revub_output', '*', '*', '*',
                           'bal_hourly_arrays.npz')
    jobs = []
    for p in sorted(set(glob.glob(pat)) | set(glob.glob(pat_tag))):
        region_path, rest = p.split('/revub_output/')
        parts = rest.split('/')          # [gcm, ssp, (tag,) bal_hourly_arrays.npz]
        parts = parts[:-1]               # drop filename
        if len(parts) == 2:
            gcm, ssp, tag = parts[0], parts[1], ''
        elif len(parts) == 3:
            gcm, ssp, tag = parts
        else:
            continue
        if TAG_FILTER is not None and tag not in TAG_FILTER:
            continue
        if REGION_SUBSTR and REGION_SUBSTR not in region_path:
            continue
        if REGION_LIST is not None and os.path.basename(region_path) not in REGION_LIST:
            continue
        jobs.append((region_path, gcm, ssp, tag))
    return jobs


# modules imported once per worker process
_MODS = {}


def _worker_init():
    import importlib
    for key in PROGRAMS:
        mod_name = _PROG[key][0]
        _MODS[key] = importlib.import_module(mod_name)


def _run_one(job):
    region_path, gcm, ssp, tag = job
    out_dir = os.path.join(region_path, 'revub_output', gcm, ssp)
    if tag:
        out_dir = os.path.join(out_dir, tag)
    rows = []
    for key in PROGRAMS:
        mod, fn_name, parquet = _MODS[key], _PROG[key][1], _PROG[key][2]
        t0 = time.time()
        try:
            getattr(mod, fn_name)(region_path, gcm, ssp, tag)
            wrote = os.path.exists(os.path.join(out_dir, parquet))
            status = 'ok' if wrote else 'skipped'
            err = ''
        except Exception:
            status = 'error'
            err = traceback.format_exc().splitlines()[-1][:200]
        rows.append({
            'region': os.path.basename(region_path), 'gcm': gcm, 'ssp': ssp,
            'tag': tag, 'program': key, 'status': status,
            'sec': round(time.time() - t0, 1), 'error': err,
        })
    return rows


def main():
    jobs = discover()
    print(f'[F-metrics] {len(jobs)} output dirs x {len(PROGRAMS)} programs '
          f'= {len(jobs) * len(PROGRAMS)} runs; workers={WORKERS}; '
          f'tags={"ALL" if TAG_FILTER is None else sorted(TAG_FILTER)}', flush=True)
    t0 = time.time()
    manifest = []
    done = 0
    with Pool(WORKERS, initializer=_worker_init) as pool:
        for rows in pool.imap_unordered(_run_one, jobs, chunksize=1):
            manifest.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                ok = sum(1 for r in manifest if r['status'] == 'ok')
                er = sum(1 for r in manifest if r['status'] == 'error')
                print(f'  {done}/{len(jobs)} dirs | runs ok={ok} err={er} '
                      f'| {time.time() - t0:.0f}s', flush=True)
    mf = pd.DataFrame(manifest)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'F_metrics_manifest.csv')
    mf.to_csv(out, index=False)
    print(f'[F-metrics] DONE in {time.time() - t0:.0f}s -> {out}', flush=True)
    if not mf.empty:
        print(mf.groupby(['program', 'status']).size().to_string(), flush=True)


if __name__ == '__main__':
    main()
