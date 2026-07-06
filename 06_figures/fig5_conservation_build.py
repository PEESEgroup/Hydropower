"""Conservation-overlay build for Fig 5: intersect the DC impact reaches with protected areas
(RiverATLAS pac, WDPA-derived, on disk) and threatened freshwater-fish ranges (IUCN Red List
freshwater HydroBASIN level-8 tables), GLOBALLY, sign-split (harmful eff>0 vs beneficial eff<0),
for BOTH baselines (left = DC-CONV, right = DC-BAL) and BOTH hydropeaking metrics.

Inputs (all local):
  $REVUB_FIG_DATA/reach_{reversals,reversals_bal,drawdown,drawdown_bal}.gpkg   (HYRIV_ID, eff, geom)
  RiverATLAS_v10.gdb : pac_pc_cse, pac_pc_use  (merge on HYRIV_ID)
  $REVUB_FIG_DATA/regions_4326.gpkg            (region, bloc polygons)
  HydroBASINS lev08 polygons (na,sa,eu,as,au)  +  data_external/iucn/threatened_by_hybas08.csv

Outputs (into $REVUB_FIG_DATA):
  reach_cons_<metric>_<base>.gpkg   per-reach enriched (eff, pac_pc_cse, in_pa, region, bloc,
                                    hybas08, n_threatened, n_native, length_km, geom)  -> for the MAP
  cons_region_summary.csv           per (metric, base, region) sign-split exposure metrics  -> panels
  cons_bloc_summary.csv             per (metric, base, bloc)  sign-split exposure metrics  -> panels

Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_conservation_build.py
"""
import os, glob
import numpy as np
import pandas as pd
import pyogrio
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
GDB = os.environ.get('REVUB_RIVERATLAS_GDB', '/datasets/swat_global/RiverATLAS_v10.gdb')
ATL = '/home/cfeng/hydro/source_data/atlas'
REGIONS = os.path.join(HERE, 'data', 'regions_4326.gpkg')   # combo-independent geometry lives in data/
IUCN = os.path.join(HERE, 'data_external', 'iucn', 'threatened_by_hybas08.csv')
PA_THRESH = 50.0          # pac_pc_cse >= 50% local catchment protected => reach "in protected area"
EQ_AREA = 'EPSG:6933'     # World Cylindrical Equal Area for length km

STEMS = {'reversals': ('reversals', 'CONV'), 'reversals_bal': ('reversals', 'BAL'),
         'drawdown': ('drawdown', 'CONV'),   'drawdown_bal': ('drawdown', 'BAL')}


def _attach_static(union_ids, sample_geom_gdf):
    """Build a per-reach static attribute frame for the union of HYRIV_IDs:
    pac (RiverATLAS), region/bloc (regions_4326), hybas08 + threatened (IUCN), length km."""
    base = sample_geom_gdf[['HYRIV_ID', 'geometry']].drop_duplicates('HYRIV_ID').copy()
    base = base[base.HYRIV_ID.isin(union_ids)].reset_index(drop=True)

    # pac from RiverATLAS (read once)
    print('  reading RiverATLAS pac ...', flush=True)
    ra = pyogrio.read_dataframe(GDB, layer='RiverATLAS_v10',
                                columns=['HYRIV_ID', 'pac_pc_cse', 'pac_pc_use'], read_geometry=False)
    ra['HYRIV_ID'] = ra['HYRIV_ID'].astype('int64')
    base = base.merge(ra, on='HYRIV_ID', how='left')

    pts = gpd.GeoDataFrame(base[['HYRIV_ID']].copy(),
                           geometry=base.geometry.representative_point(), crs=base.crs)

    # region + bloc
    print('  spatial join region/bloc ...', flush=True)
    reg = gpd.read_file(REGIONS)
    rj = gpd.sjoin(pts, reg[['region', 'bloc', 'geometry']], how='left', predicate='within')
    rj = rj[~rj.index.duplicated(keep='first')]
    miss = rj.region.isna()
    if miss.any():                                   # coastal reaches: nearest region
        nn = gpd.sjoin_nearest(pts[miss.values], reg[['region', 'bloc', 'geometry']], how='left')
        nn = nn[~nn.index.duplicated(keep='first')]
        rj.loc[miss, 'region'] = nn['region']; rj.loc[miss, 'bloc'] = nn['bloc']
    base['region'] = rj.region.values; base['bloc'] = rj.bloc.values

    # lev08 basin -> threatened
    print('  spatial join lev08 basin ...', flush=True)
    polys = []
    for c in ['na', 'sa', 'eu', 'as', 'au']:
        fp = f'{ATL}/hybas_{c}/hybas_{c}_lev08_v1c.shp'
        if os.path.exists(fp):
            polys.append(gpd.read_file(fp, columns=['HYBAS_ID'])[['HYBAS_ID', 'geometry']])
    hb = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=polys[0].crs)
    bj = gpd.sjoin(pts, hb, how='left', predicate='within')
    bj = bj[~bj.index.duplicated(keep='first')]
    base['hybas08'] = bj.HYBAS_ID.astype('Int64').astype(str).values
    look = pd.read_csv(IUCN, dtype={'hybas_id': str}).rename(columns={'hybas_id': 'hybas08'})
    base = base.merge(look, on='hybas08', how='left')
    base['n_threatened'] = base.n_threatened.fillna(0).astype(int)
    base['n_native'] = base.n_native.fillna(0).astype(int)

    # equal-area length
    print('  computing equal-area length ...', flush=True)
    ln = gpd.GeoDataFrame(base[['HYRIV_ID']].copy(), geometry=base.geometry, crs=base.crs).to_crs(EQ_AREA)
    base['length_km'] = ln.length.values / 1000.0
    base['in_pa'] = (base.pac_pc_cse.fillna(0) >= PA_THRESH)
    return base


def main():
    layers = {s: os.path.join(DATA, f'reach_{s}.gpkg') for s in STEMS if os.path.exists(os.path.join(DATA, f'reach_{s}.gpkg'))}
    print('layers found:', list(layers))
    gdfs = {s: gpd.read_file(p) for s, p in layers.items()}
    union_ids = set().union(*[set(g.HYRIV_ID.astype('int64')) for g in gdfs.values()])
    print('union reaches:', len(union_ids))
    any_gdf = pd.concat([g[['HYRIV_ID', 'geometry']] for g in gdfs.values()], ignore_index=True)
    any_gdf = gpd.GeoDataFrame(any_gdf, crs=list(gdfs.values())[0].crs)
    static = _attach_static(union_ids, any_gdf)
    stat_cols = ['HYRIV_ID', 'pac_pc_cse', 'pac_pc_use', 'in_pa', 'region', 'bloc',
                 'hybas08', 'n_threatened', 'n_native', 'length_km']

    reg_rows, bloc_rows = [], []
    for s, g in gdfs.items():
        metric, base = STEMS[s]
        d = g[['HYRIV_ID', 'eff']].copy()
        d['HYRIV_ID'] = d.HYRIV_ID.astype('int64')
        d = d.merge(static[stat_cols], on='HYRIV_ID', how='left')
        d['sign'] = np.where(d.eff > 0, 'harmful', np.where(d.eff < 0, 'beneficial', 'neutral'))
        # per-reach enriched gpkg (for the map): attach geometry back
        out = g[['HYRIV_ID', 'eff', 'geometry']].merge(static[stat_cols], on='HYRIV_ID', how='left')
        gpd.GeoDataFrame(out, crs=g.crs).to_file(os.path.join(DATA, f'reach_cons_{s}.gpkg'), driver='GPKG')

        def agg(df, key):
            for kval, sub in df.groupby(key):
                for sgn in ['harmful', 'beneficial']:
                    ss = sub[sub.sign == sgn]
                    L = ss.length_km.sum()
                    Lpa = ss.loc[ss.in_pa, 'length_km'].sum()
                    Lfish = ss.loc[ss.n_threatened > 0, 'length_km'].sum()
                    yield {key: kval, 'metric': metric, 'baseline': base, 'sign': sgn,
                           'n_reach': len(ss), 'length_km': L,
                           'len_in_pa_km': Lpa, 'pa_share_pct': 100 * Lpa / L if L else 0,
                           'len_in_fish_km': Lfish, 'fish_share_pct': 100 * Lfish / L if L else 0,
                           'n_threatened_sp': int(ss.loc[ss.n_threatened > 0, 'n_threatened'].sum()),
                           'wexp_pa': float((ss.eff.abs() * ss.in_pa * ss.length_km).sum())}
        reg_rows += list(agg(d, 'region'))
        bloc_rows += list(agg(d, 'bloc'))
        print(f'  [{s}] {metric}/{base}: {len(d)} reaches, '
              f'harmful {int((d.sign=="harmful").sum())} / beneficial {int((d.sign=="beneficial").sum())}')

    pd.DataFrame(reg_rows).to_csv(os.path.join(DATA, 'cons_region_summary.csv'), index=False)
    pd.DataFrame(bloc_rows).to_csv(os.path.join(DATA, 'cons_bloc_summary.csv'), index=False)
    print('saved cons_region_summary.csv + cons_bloc_summary.csv + reach_cons_*.gpkg')


if __name__ == '__main__':
    main()
