"""
F2_prepare_slope.py - extract REAL channel slope per station from RiverATLAS
(HydroATLAS v1.0) and write <hydro_dir>/channel_slope.csv for F2's Manning VRR.

RiverATLAS attribute `sgr_dk_rav` = stream gradient, reach average, in dm/km
(decimetres per km). slope_m_per_m = sgr_dk_rav * 1e-4. Joined to stations by
`snap_HYRIV_ID` (fallback final_/matched_HYRIV_ID) - 100% populated.

Run once after downloading + unzipping RiverATLAS_v10.gdb:
  python F2_prepare_slope.py            # processes all bal-ready regions
Env REVUB_RIVERATLAS_GDB overrides the GDB path.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import pyogrio

GDB = os.environ.get('REVUB_RIVERATLAS_GDB',
                     '/datasets/swat_global/RiverATLAS_v10.gdb')
SGR_TO_M_PER_M = 1e-4          # dm/km → m/m


def load_riveratlas_slope():
    layers = pyogrio.list_layers(GDB)
    layer = layers[0][0] if len(layers) else None
    print(f'[slope] RiverATLAS layer: {layer}')
    info = pyogrio.read_info(GDB, layer=layer)
    fields = list(info['fields'])
    hid = next((f for f in fields if f.upper() == 'HYRIV_ID'), None)
    sgr = next((f for f in fields if f.lower() == 'sgr_dk_rav'), None)
    if not (hid and sgr):
        raise KeyError(f'HYRIV_ID/sgr_dk_rav not in fields: {fields[:30]}')
    df = pyogrio.read_dataframe(GDB, layer=layer, columns=[hid, sgr],
                                read_geometry=False)
    df = df.rename(columns={hid: 'HYRIV_ID', sgr: 'sgr_dk_rav'})
    print(f'[slope] read {len(df):,} reaches')
    return dict(zip(df['HYRIV_ID'].astype('int64'), df['sgr_dk_rav'].astype(float)))


_HYRIV_COLS = ['snap_HYRIV_ID', 'final_HYRIV_ID', 'matched_HYRIV_ID']


def process_region(hydro_dir, slope_lookup):
    csv = os.path.join(hydro_dir, 'station_channel_result.csv')
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, low_memory=False)
    hid = np.full(len(df), np.nan)
    for c in _HYRIV_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce').values
            hid = np.where(np.isnan(hid), v, hid)
    sgr = np.array([slope_lookup.get(int(h)) if np.isfinite(h) else None for h in hid],
                   dtype=float)
    slope = sgr * SGR_TO_M_PER_M
    out = pd.DataFrame({'name_unified': df['name_unified'].astype(str),
                        'HYRIV_ID': hid, 'sgr_dk_rav': sgr, 'slope_m_per_m': slope})
    out.to_csv(os.path.join(hydro_dir, 'channel_slope.csv'), index=False)
    n = np.isfinite(slope).sum()
    return n, len(out), float(np.nanmedian(slope))


def main():
    if not os.path.exists(GDB):
        sys.exit(f'RiverATLAS GDB not found: {GDB} (unzip RiverATLAS_v10.gdb.zip first)')
    lookup = load_riveratlas_slope()
    paths = sorted(glob.glob('/datasets/swat_global/**/bal_hourly_arrays.npz', recursive=True)) \
        + sorted(glob.glob('/tmp/revub_integration/**/bal_hourly_arrays.npz', recursive=True))
    regions = sorted({p.split('/revub_output')[0] for p in paths})
    for r in regions:
        res = process_region(os.path.join(r, 'hydro'), lookup)
        if res:
            n, tot, med = res
            print(f'  {os.path.basename(r):16s} {n}/{tot} slopes  median={med:.2e} m/m')


if __name__ == '__main__':
    main()
