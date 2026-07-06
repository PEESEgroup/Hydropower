"""Build the ecology impact-reach layers for the conservation-overlay figure (Fig 5).

Reuses fig4_maps_build's RiverATLAS NEXT_DOWN walk to paint, for every DC-serving dam, the
reaches it controls with the per-dam effect. Unlike fig4_maps_build (which only does
eff = DC_A - CONV for reversals), this emits BOTH baselines for BOTH hydropeaking metrics:

  left  column of the figure = DC - CONV (relative to conventional hydropower operation)
  right column of the figure = DC - BAL  (relative to typical balancing-grid operation)

Outputs into $REVUB_FIG_DATA (default data_nofuture_gfdl/):
  reach_reversals.gpkg       (DC-CONV, flow reversals)  -- already exists, reused; rebuilt only if missing
  reach_reversals_bal.gpkg   (DC-BAL,  flow reversals)
  reach_drawdown.gpkg        (DC-CONV, lethal drawdown VRR_exceed_h_13cmh)
  reach_drawdown_bal.gpkg    (DC-BAL,  lethal drawdown)

Run (geopandas env):  REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_reach_build.py
"""
import os
import numpy as np
import pandas as pd
import pyogrio
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
os.environ.setdefault('REVUB_FIG_DATA', DATA)        # so fig4_maps_build.DATA matches
import fig4_maps_build as F                          # noqa: E402  (functions: load_network, build_dam_metric, region_dirs, GDB)

# (stem, metric, baseline column)
SPECS = [
    ('reversals',     'reversals_per_year_hourly', 'CONV'),
    ('reversals_bal', 'reversals_per_year_hourly', 'BAL_C'),
    ('drawdown',      'VRR_exceed_h_13cmh',        'CONV'),
    ('drawdown_bal',  'VRR_exceed_h_13cmh',        'BAL_C'),
]


def main():
    eco = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    todo = [s for s in SPECS if not os.path.exists(os.path.join(DATA, 'reach_%s.gpkg' % s[0]))]
    if not todo:
        print('all reach layers already present, nothing to do'); return
    print('building:', [s[0] for s in todo])

    pos, nd, up = F.load_network()
    rdirs = F.region_dirs()
    print('[regions] %d hydro dirs' % len(rdirs))

    reach_maps = {}
    for stem, metric, base in todo:
        df = eco[eco.metric == metric].copy()
        df['eff'] = df['DC_A'] - df[base]
        eff = {(r, s): e for r, s, e in zip(df.region, df.station_idx, df.eff)}
        reach_maps[stem] = F.build_dam_metric(eff, pos, nd, up, rdirs)
        print('[%s] %d affected reaches (eff = DC_A - %s)' % (stem, len(reach_maps[stem]), base))

    union = sorted(set().union(*[set(m) for m in reach_maps.values()]))
    print('[geom] reading geometry for %s reaches ...' % f'{len(union):,}')
    g = pyogrio.read_dataframe(F.GDB, layer='RiverATLAS_v10', columns=['HYRIV_ID'], use_arrow=True)
    g['HYRIV_ID'] = g['HYRIV_ID'].astype('int64')
    g = g[g['HYRIV_ID'].isin(set(union))].copy()
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
