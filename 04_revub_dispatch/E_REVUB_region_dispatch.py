#!/usr/bin/env python3
"""Run one source region against an E-generated cross-region DC load profile."""
import os
import shutil
import sys
import tempfile
import time
import traceback


if len(sys.argv) != 7:
    raise SystemExit(
        'usage: E_REVUB_region_dispatch.py REGION_PATH SCENARIO GCM SSP '
        'PROFILE_NPZ OUTPUT_DIR')

region_path, scenario, gcm, ssp, profile_npz, output_dir = sys.argv[1:]
if scenario not in ('flat', 'typical'):
    raise ValueError(f'unknown scenario {scenario!r}')

base = os.path.dirname(os.path.abspath(__file__))
scratch = tempfile.mkdtemp(prefix='revub_e_region_')
os.makedirs(output_dir, exist_ok=True)

os.environ['REVUB_REGION_PATH'] = region_path
os.environ['REVUB_GCM'] = gcm
os.environ['REVUB_SSP'] = ssp
os.environ['REVUB_LOAD_TYPE'] = 'flat' if scenario == 'flat' else 'auto'
os.environ['REVUB_LOAD_FRAC'] = 'auto'
os.environ['REVUB_CALIBRATION_ONLY'] = '0'
os.environ['REVUB_PS_MODE'] = 'cascade'
os.environ['REVUB_N_PS_ELCC'] = '100'
os.environ.setdefault('NUMBA_NUM_THREADS', os.environ.get('REVUB_NUMBA_THREADS', '2'))
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('REVUB_NUMBA_THREADS', '2'))
os.environ['REVUB_OUTPUT_DIR_OVERRIDE'] = scratch
os.environ['REVUB_D_OUTPUT_DIR_OVERRIDE'] = output_dir
os.environ['REVUB_E_DC_PROFILE_OVERRIDE'] = profile_npz
os.environ['REVUB_D_SUMMARY_ONLY'] = '1'

stages = [
    ('A', 'A_REVUB_initialise_v2.py'),
    ('B', 'B_REVUB_main_code_v2.py'),
    ('C', 'C_REVUB_PS_dispatch.py'),
    ('E-region-D', 'D_REVUB_DC_dispatch.py'),
]
namespace = {'__name__': '__main__'}
t0 = time.time()

try:
    for name, filename in stages:
        stage_t0 = time.time()
        print(f'----- {name} ({filename}) -----', flush=True)
        try:
            with open(os.path.join(base, filename)) as handle:
                exec(compile(handle.read(), filename, 'exec'), namespace)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
            print(f'----- {name} skipped with sys.exit(0) -----', flush=True)
        print(f'----- {name} done in {time.time() - stage_t0:.0f}s -----', flush=True)
    summary = os.path.join(output_dir, 'dc_hydro_summary.csv')
    if not os.path.exists(summary):
        raise RuntimeError(f'E regional dispatch did not create {summary}')
    with open(os.path.join(output_dir, '_e_region_complete.txt'), 'w') as handle:
        handle.write(
            f'region={os.path.basename(region_path)} scenario={scenario} '
            f'gcm={gcm} ssp={ssp} profile={profile_npz} '
            f'elapsed={time.time() - t0:.0f}s\n')
    print(f'STATUS: OK elapsed={time.time() - t0:.0f}s', flush=True)
except Exception:
    traceback.print_exc()
    print(f'STATUS: FAIL elapsed={time.time() - t0:.0f}s', flush=True)
    raise
finally:
    shutil.rmtree(scratch, ignore_errors=True)
