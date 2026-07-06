"""F_stage_E_redispatch.py - stage E2 cross-region redispatch hourly flows into
each exporter region's revub_output so F._load_e_scenarios can pick them up.

E2 writes per-source redispatch caches under
  E_output_v2/<bloc>/<gcm>/<ssp>/flat/<grid_tag>/redispatch_cache/<region>/<hash>/dc_scenario_{A,B}_hourly.npz
F looks for them under
  <region_path>/revub_output/<gcm>/<ssp>/flat/redispatch_cache/<region>/<hash>/

A region may carry SEVERAL hash dirs - one per redispatch ITERATION. Only the
CONVERGED (final) iteration is physically meaningful; its npz is the freshest.
We therefore stage, per (bloc, gcm, ssp, region), only the latest-mtime hash.

Canonical topology = the vintage-2040 grid (grid_tag 'year_2040_scale_1', i.e.
al0). The all-lines sensitivity (…_alllines) is intentionally NOT staged.

Usage:  python F_stage_E_redispatch.py [grid_tag]
"""
import glob
import os
import shutil
import sys

os.environ.setdefault('REVUB_E_ALLOW_LEGACY', '1')
import E_REVUB_cross_region as e1

GRID_TAG = sys.argv[1] if len(sys.argv) > 1 else 'year_2040_scale_1'
E_ROOT = os.path.join(e1.CODE, 'E_output_v2')


def main():
    pattern = os.path.join(E_ROOT, '*', '*', '*', 'flat', GRID_TAG,
                           'redispatch_cache', '*', '*')
    # group hash dirs by (gcm, ssp, region); keep the freshest (converged) one
    best = {}   # (gcm, ssp, region) -> (mtime, hash_dir)
    for hdir in glob.glob(pattern):
        if not os.path.isdir(hdir):
            continue
        a = os.path.join(hdir, 'dc_scenario_A_hourly.npz')
        if not os.path.exists(a):
            continue
        parts = hdir.split(os.sep)
        i = parts.index('E_output_v2')
        gcm, ssp = parts[i + 2], parts[i + 3]
        region = parts[-2]
        mt = os.path.getmtime(a)
        key = (gcm, ssp, region)
        if key not in best or mt > best[key][0]:
            best[key] = (mt, hdir)

    staged = 0
    missing_dir = []
    exporters = set()
    for (gcm, ssp, region), (_, hdir) in sorted(best.items()):
        rp = e1.region_dir(region)
        if rp is None:
            missing_dir.append(region)
            continue
        h = os.path.basename(hdir)
        dst = os.path.join(rp, 'revub_output', gcm, ssp, 'flat',
                           'redispatch_cache', region, h)
        os.makedirs(dst, exist_ok=True)
        for npz in glob.glob(os.path.join(hdir, 'dc_scenario_*_hourly.npz')):
            shutil.copy2(npz, os.path.join(dst, os.path.basename(npz)))
        staged += 1
        exporters.add(region)

    print(f'[stage-E] grid_tag={GRID_TAG}')
    print(f'[stage-E] staged {staged} (gcm,ssp,region) redispatch caches')
    print(f'[stage-E] {len(exporters)} distinct exporter regions: {sorted(exporters)}')
    if missing_dir:
        print(f'[stage-E] WARNING no region_dir for: {sorted(set(missing_dir))}')
    # emit exporter list for the targeted F re-run
    with open(os.path.join(e1.CODE, 'batch', 'E_exporters.txt'), 'w') as fh:
        fh.write(','.join(sorted(exporters)))
    print(f'[stage-E] exporter list -> batch/E_exporters.txt')


if __name__ == '__main__':
    main()
