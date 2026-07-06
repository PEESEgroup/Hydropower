"""
F3_prepare_routing.py - build river SEGMENTS and their controlling upstream-dam
set U(X), from RiverATLAS (replaces F3_prepare_dependency.py). See the
F3 cascade-and-shared-downstream revision notes §2 / §4.

Each dam walks DOWNSTREAM via NEXT_DOWN, registering itself onto every reach it
controls until the NEXT DAM (re-regulation) - terminal dams capped at a length
window. Reaches are grouped by their controlling-dam set U(X) → one SEGMENT per
distinct U. Cascades: A stops at B. Confluences: a post-junction reach gets both
upstream dams → no double-count.

Output: <hydro_dir>/supply_segments.csv
  seg_id, dams (|-joined names), n_dams, n_reaches, dep_length_km,
  demand_Mm3_yr, A_outlet_skm, A_dams_skm   (A_outlet/A_dams = E2 area-ratio)

Run once: python F3_prepare_routing.py
"""

import os
import sys
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import pyogrio

GDB = os.environ.get('REVUB_RIVERATLAS_GDB', '/datasets/swat_global/RiverATLAS_v10.gdb')
NIR_M_PER_YR = 1.0
EPS_AREA = 0.01                  # stop walk when dam controls <1% of local drainage (perturbation
                                 # attenuated) - replaces the arbitrary fixed-km window (issue #11)
MAX_REACHES = 5000              # runaway guard only


def load_network():
    # ire_pc_cse = irrigated-area % of the LOCAL reach sub-catchment (pairs with CATCH_SKM);
    # NOT ire_pc_use which is averaged over the whole UPSTREAM watershed (wrong support).
    cols = ['HYRIV_ID', 'NEXT_DOWN', 'LENGTH_KM', 'CATCH_SKM', 'UPLAND_SKM', 'ire_pc_cse']
    df = pyogrio.read_dataframe(GDB, layer='RiverATLAS_v10', columns=cols, read_geometry=False)
    df['HYRIV_ID'] = df['HYRIV_ID'].astype('int64')
    df['NEXT_DOWN'] = df['NEXT_DOWN'].astype('int64')
    print(f'[routing] loaded {len(df):,} reaches')
    return df.set_index('HYRIV_ID')


_HYRIV_COLS = ['snap_HYRIV_ID', 'final_HYRIV_ID', 'matched_HYRIV_ID']


def _dam_reaches(df):
    hid = np.full(len(df), np.nan)
    for c in _HYRIV_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce').values
            hid = np.where(np.isnan(hid), v, hid)
    return hid


def process_region(hydro_dir, net):
    csv = os.path.join(hydro_dir, 'station_channel_result.csv')
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, low_memory=False)
    names = df['name_unified'].astype(str).values
    hid = _dam_reaches(df)
    # HYRIV_ID -> LIST of dam names (issue #7): two dams on the same reach (e.g. Argentina
    # ALVAREZ CONDARCO & CACHEUTA (NUEVA)) must both be kept, not overwrite each other.
    dam_of = defaultdict(list)
    for i, h in enumerate(hid):
        if np.isfinite(h):
            dam_of[int(h)].append(names[i])
    dam_set = set(dam_of)

    # reach -> set of controlling dam HYRIVs (NOT names: name is non-unique - two distinct 'Funil'
    # dams in brazil_sudeste would otherwise MERGE into one segment and be scored against the wrong
    # dam's flow). Walk downstream until the next dam (re-regulation) or perturbation attenuates
    # (<EPS_AREA of local drainage). No fixed km window (issue #11) - MAX_REACHES is a runaway guard.
    reach_dams = {}
    for start, dnames in dam_of.items():
        A0 = float(net.loc[start]['UPLAND_SKM']) if start in net.index else np.nan
        h = start
        n = 0
        while h in net.index and n < MAX_REACHES:
            reach_dams.setdefault(h, set()).add(start)      # controlling HYRIV (unique), not name
            n += 1
            row = net.loc[h]
            nd = int(row['NEXT_DOWN'])
            if nd == 0 or nd in dam_set:          # mouth/sink or next dam → stop
                break
            if nd in net.index and np.isfinite(A0) and A0 > 0:
                if A0 / float(net.loc[nd]['UPLAND_SKM']) < EPS_AREA:   # attenuated → stop
                    break
            h = nd

    # group reaches by controlling-HYRIV set → segments
    segs = {}
    for h, hset in reach_dams.items():
        key = tuple(sorted(hset))
        s = segs.setdefault(key, {'n_reaches': 0, 'len': 0.0, 'irr_km2': 0.0, 'A_outlet': 0.0})
        row = net.loc[h]
        s['n_reaches'] += 1
        s['len'] += float(row['LENGTH_KM'])
        s['irr_km2'] += float(row['ire_pc_cse']) / 100.0 * float(row['CATCH_SKM'])
        s['A_outlet'] = max(s['A_outlet'], float(row['UPLAND_SKM']))

    rows = []
    for i, (key, s) in enumerate(segs.items()):
        # key = controlling HYRIVs (unique). A_dams sums their (unique) upstream areas; names for
        # display; dam_hyrivs lets F3_supply map each controlling dam to its npz station by HYRIV.
        A_dams = sum(float(net.loc[hh]['UPLAND_SKM']) for hh in key if hh in net.index)
        dam_names = '|'.join(nm for hh in key for nm in dam_of.get(hh, []))
        dam_hyrivs = '|'.join(str(hh) for hh in key)
        rows.append({'seg_id': i, 'dams': dam_names, 'dam_hyrivs': dam_hyrivs, 'n_dams': len(key),
                     'n_reaches': s['n_reaches'], 'dep_length_km': round(s['len'], 1),
                     'demand_Mm3_yr': s['irr_km2'] * 1e6 * NIR_M_PER_YR / 1e6,
                     'A_outlet_skm': s['A_outlet'], 'A_dams_skm': A_dams})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(hydro_dir, 'supply_segments.csv'), index=False)
    nz = (out['demand_Mm3_yr'] > 0).sum()
    return len(out), int(nz), int((out['n_dams'] > 1).sum())


def main():
    if not os.path.exists(GDB):
        sys.exit(f'RiverATLAS GDB not found: {GDB}')
    net = load_network()
    paths = sorted(glob.glob('/datasets/swat_global/**/bal_hourly_arrays.npz', recursive=True)) \
        + sorted(glob.glob('/tmp/revub_integration/**/bal_hourly_arrays.npz', recursive=True))
    for r in sorted({p.split('/revub_output')[0] for p in paths}):
        res = process_region(os.path.join(r, 'hydro'), net)
        if res:
            nseg, nz, nconf = res
            print(f'  {os.path.basename(r):16s} {nseg} segments ({nz} w/ irrigation, '
                  f'{nconf} multi-dam/confluence)')


if __name__ == '__main__':
    main()
