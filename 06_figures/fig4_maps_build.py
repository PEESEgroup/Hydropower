"""Build native-network (HydroRIVERS/RiverATLAS) reach geometries coloured by the DC vs conventional
river-impact effect, for the Fig 4 world maps.

F routes on RiverATLAS_v10 (HYRIV_ID + NEXT_DOWN), NOT GRFR - so we replicate F3_prepare_routing's
DOWNSTREAM walk here to find, for every DC-serving dam/segment, the exact reaches it controls, and
paint those reaches with the dam's/segment's DC_A - CONV effect. This matches ALL dams (GRFR matched
only 11-54 %). Three metrics -> three reach gpkgs:
  reversals_per_year_hourly (ecology), STCI (sediment), gap_months_per_year (supply).

Ecology/sediment effects are per DAM (station_idx -> HYRIV via station_channel_result.csv); the dam's
effect is assigned to every reach it controls (overlap -> keep the largest |effect|). Supply effects
are per SEGMENT (a controlling-dam set); we re-run F3's exact grouping to recover each seg_id's reach
set and paint them. Output: data/reach_<stem>.gpkg (geometry + eff) + a manifest line per metric.

Run (local geopandas env):  /home/cfeng/.conda/envs/pybkb/bin/python fig4_maps_build.py
"""
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import pyogrio
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
GDB = os.environ.get('REVUB_RIVERATLAS_GDB', '/datasets/swat_global/RiverATLAS_v10.gdb')
ROOT = '/datasets/swat_global'
EPS_AREA = 0.01            # stop walk when dam controls <1% of local drainage  (F3 issue #11)
MAX_REACHES = 5000         # runaway guard
_HYRIV_COLS = ['snap_HYRIV_ID', 'final_HYRIV_ID', 'matched_HYRIV_ID']   # coalesce order = F3

# (stem, source csv, metric filter or None)   effect column is always DC_A - CONV
METRICS = [('reversals', 'fig4_eco.csv', 'reversals_per_year_hourly'),
           ('stci',       'fig4_sed.csv', None),
           ('gap',        'fig4_sup.csv', 'gap_months_per_year')]


def load_network():
    cols = ['HYRIV_ID', 'NEXT_DOWN', 'UPLAND_SKM']
    df = pyogrio.read_dataframe(GDB, layer='RiverATLAS_v10', columns=cols, read_geometry=False)
    hyriv = df['HYRIV_ID'].astype('int64').to_numpy()
    nd = df['NEXT_DOWN'].astype('int64').to_numpy()
    up = df['UPLAND_SKM'].astype('float64').to_numpy()
    pos = {int(h): i for i, h in enumerate(hyriv)}
    print('[net] %s reaches' % f'{len(hyriv):,}')
    return pos, nd, up


def dam_hyriv_by_station(csv):
    """station_idx -> controlling HYRIV (coalesce snap>final>matched, exactly like F3._dam_reaches)."""
    df = pd.read_csv(csv, low_memory=False)
    hid = np.full(len(df), np.nan)
    for c in _HYRIV_COLS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce').to_numpy()
            hid = np.where(np.isnan(hid), v, hid)
    out = {}
    for i, h in enumerate(hid):
        if np.isfinite(h):
            out[int(df['station_idx'].iloc[i])] = int(h)
    names = {int(df['station_idx'].iloc[i]): str(df['name_unified'].iloc[i]) for i in range(len(df))}
    return out, names, hid


def walk(start, pos, nd, up, dam_set):
    """Reaches controlled by dam `start`, walking NEXT_DOWN until next dam / mouth / attenuation."""
    out = []
    i0 = pos.get(start)
    if i0 is None:
        return out
    A0 = up[i0]
    h = start; n = 0
    while n < MAX_REACHES:
        i = pos.get(h)
        if i is None:
            break
        out.append(h); n += 1
        ndn = int(nd[i])
        if ndn == 0 or ndn in dam_set:
            break
        j = pos.get(ndn)
        if j is not None and np.isfinite(A0) and A0 > 0 and up[j] > 0 and A0 / up[j] < EPS_AREA:
            break
        h = ndn
    return out


def region_dirs():
    out = {}
    for p in glob.glob(os.path.join(ROOT, '*', '*', 'hydro', 'station_channel_result.csv')):
        out[os.path.basename(os.path.dirname(os.path.dirname(p)))] = os.path.dirname(p)
    return out


def build_dam_metric(eff_by_rs, pos, nd, up, rdirs):
    """eff_by_rs: {(region, station_idx): eff}.  -> {hyriv: eff} keeping largest |eff| on overlap."""
    reach = {}
    by_region = defaultdict(dict)
    for (r, s), e in eff_by_rs.items():
        by_region[r][s] = e
    for r, smap in by_region.items():
        hydro = rdirs.get(r)
        if not hydro:
            print('  [skip] no hydro dir for', r); continue
        st2h, _, hid = dam_hyriv_by_station(os.path.join(hydro, 'station_channel_result.csv'))
        dam_set = {int(h) for h in hid if np.isfinite(h)}
        for s, e in smap.items():
            start = st2h.get(s)
            if start is None or not np.isfinite(e):
                continue
            for h in walk(start, pos, nd, up, dam_set):
                if h not in reach or abs(e) > abs(reach[h]):
                    reach[h] = float(e)
    return reach


def build_supply(eff_by_rs, pos, nd, up, rdirs):
    """Supply effect is per SEGMENT. Re-run F3's grouping to recover seg_id -> reach set, paint it."""
    reach = {}
    by_region = defaultdict(dict)
    for (r, seg), e in eff_by_rs.items():
        by_region[r][int(seg)] = e
    for r, smap in by_region.items():
        hydro = rdirs.get(r)
        if not hydro:
            print('  [skip] no hydro dir for', r); continue
        df = pd.read_csv(os.path.join(hydro, 'station_channel_result.csv'), low_memory=False)
        hid = np.full(len(df), np.nan)
        for c in _HYRIV_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors='coerce').to_numpy()
                hid = np.where(np.isnan(hid), v, hid)
        names = df['name_unified'].astype(str).to_numpy()
        dam_of = defaultdict(list)
        for i, h in enumerate(hid):
            if np.isfinite(h):
                dam_of[int(h)].append(names[i])
        dam_set = set(dam_of)
        reach_dams = {}
        for start in dam_of:                                   # same insertion order as F3
            for h in walk(start, pos, nd, up, dam_set):
                reach_dams.setdefault(h, set()).add(start)
        seg_reaches = {}                                       # seg_id -> [hyriv]  (enumerate == F3 seg_id)
        segs = {}
        for h, hset in reach_dams.items():
            key = tuple(sorted(hset))
            segs.setdefault(key, []).append(h)
        for i, (key, hs) in enumerate(segs.items()):
            seg_reaches[i] = hs
        for seg, e in smap.items():
            if not np.isfinite(e):
                continue
            for h in seg_reaches.get(seg, []):
                if h not in reach or abs(e) > abs(reach[h]):
                    reach[h] = float(e)
    return reach


def main():
    pos, nd, up = load_network()
    rdirs = region_dirs()
    print('[regions] %d hydro dirs' % len(rdirs))

    reach_maps = {}            # stem -> {hyriv: eff}
    for stem, src, metric in METRICS:
        df = pd.read_csv(os.path.join(DATA, src))
        if metric is not None and 'metric' in df.columns:
            df = df[df.metric == metric].copy()
        df['eff'] = df['DC_A'] - df['CONV']
        if stem == 'gap':
            eff = {(r, seg): e for r, seg, e in zip(df.region, df.seg, df.eff)}
            rm = build_supply(eff, pos, nd, up, rdirs)
        else:
            eff = {(r, s): e for r, s, e in zip(df.region, df.station_idx, df.eff)}
            rm = build_dam_metric(eff, pos, nd, up, rdirs)
        reach_maps[stem] = rm
        print('[%s] %d affected reaches' % (stem, len(rm)))

    union = sorted(set().union(*[set(m) for m in reach_maps.values()]))
    print('[geom] reading geometry for %s reaches ...' % f'{len(union):,}')
    uset = set(union)
    g = pyogrio.read_dataframe(GDB, layer='RiverATLAS_v10', columns=['HYRIV_ID'], use_arrow=True)
    g['HYRIV_ID'] = g['HYRIV_ID'].astype('int64')
    g = g[g['HYRIV_ID'].isin(uset)].copy()
    geom = dict(zip(g['HYRIV_ID'].to_numpy(), g.geometry.to_numpy()))
    print('[geom] matched %d / %d' % (len(geom), len(union)))

    for stem, rm in reach_maps.items():
        hs = [h for h in rm if h in geom]
        gdf = gpd.GeoDataFrame({'HYRIV_ID': hs, 'eff': [rm[h] for h in hs]},
                               geometry=[geom[h] for h in hs], crs='EPSG:4326')
        out = os.path.join(DATA, 'reach_%s.gpkg' % stem)
        gdf.to_file(out, driver='GPKG')
        print('saved %s | %d reaches | eff p[5,50,95]=%s' %
              (out, len(gdf), np.round(np.nanpercentile(gdf.eff, [5, 50, 95]), 3)))


if __name__ == '__main__':
    main()
