"""Fig 5 IUCN view: global THREATENED freshwater-fish richness per HydroBASINS level-8 basin
(count of CR/EN/VU species, IUCN Red List freshwater HydroBASIN tables), with the DC-affected
rivers overlaid (dark lines) so you can see where the affected network sits relative to the
endangered-fish hotspots. China-worldview basemap; Equal-Earth.

Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_map_fish.py
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import revub_style as S
from revub_worldview import load_basemap
from fig1a_coverage_map import latlon_bounds, apply_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
ATL = '/home/cfeng/hydro/source_data/atlas'
IUCN = os.path.join(HERE, 'data_external', 'iucn', 'threatened_by_hybas08.csv')
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'; CMAP = plt.get_cmap('YlOrRd')


def main():
    world = load_basemap().to_crs(CRS)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')

    print('loading lev08 basins + threatened richness ...', flush=True)
    polys = []
    for c in ['na', 'sa', 'eu', 'as', 'au']:
        fp = f'{ATL}/hybas_{c}/hybas_{c}_lev08_v1c.shp'
        if os.path.exists(fp):
            polys.append(gpd.read_file(fp, columns=['HYBAS_ID'])[['HYBAS_ID', 'geometry']])
    hb = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=polys[0].crs)
    hb['hybas_id'] = hb.HYBAS_ID.astype('int64').astype(str)
    look = pd.read_csv(IUCN, dtype={'hybas_id': str})
    hb = hb.merge(look[['hybas_id', 'n_threatened']], on='hybas_id', how='left')
    hb['n_threatened'] = hb.n_threatened.fillna(0)
    hb = hb[hb.n_threatened > 0].to_crs(CRS)               # only basins with threatened fish
    vmax = float(np.nanpercentile(hb.n_threatened, 97))
    norm = Normalize(0, vmax)

    g = gpd.read_file(os.path.join(DATA, 'reach_cons_reversals.gpkg')).to_crs(CRS)
    g = g[g.eff != 0]

    fig, ax = S.fig_mm(180, 92)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.95, bottom=0.07)
    world.plot(ax=ax, color=LAND, ec='none', zorder=0)
    hb.plot(ax=ax, column='n_threatened', cmap=CMAP, norm=norm, ec='none', alpha=0.9,
            zorder=1, rasterized=True)
    g.plot(ax=ax, color='#1b2a4a', linewidth=0.32, alpha=0.85, zorder=3, rasterized=True)  # affected rivers
    apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title('IUCN threatened freshwater-fish richness per basin '
                 '(CR/EN/VU count)   -   dark lines = DC-affected rivers', fontsize=8, pad=3)
    cax = fig.add_axes([0.34, 0.06, 0.32, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', extend='max')
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    fig.text(0.34, 0.10, 'threatened fish species per basin', fontsize=6.5, ha='left', va='bottom', color='0.3')
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig5_map_fish')); plt.close(fig)
    print('saved fig5_map_fish | basins shown=%d vmax=%.0f' % (len(hb), vmax))


if __name__ == '__main__':
    main()
