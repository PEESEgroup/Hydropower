"""Fig 5 combined view (ONE map per baseline, drawn SEPARATELY, SAME size as the Fig-4 ecology maps).
  background = IUCN threatened freshwater-fish richness per HydroBASINS level-8 basin,
               DISCRETE 1..6+ classes (warm, truncated so the densest stay medium orange-red)
  faint lines = full river network (context, thin grey)
  bold lines  = DC-affected reaches, PiYG reversed: PINK = harmful, GREEN = beneficial
Map = 157.5 x 74.25 mm (matches fig4_maps_points); colorbars/legend drawn SEPARATELY per map.
Outputs: fig5_map_fish_sign_{conv,bal}(.png/.svg) + ..._legend.

Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_map_fish_sign.py [reversals|drawdown]
"""
import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

import revub_style as S
from revub_worldview import load_basemap
from revub_rivers import load_rivers_world
from fig1a_coverage_map import latlon_bounds, apply_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
ATL = '/home/cfeng/hydro/source_data/atlas'
IUCN = os.path.join(HERE, 'data_external', 'iucn', 'threatened_by_hybas08.csv')
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'
W, Ht = 157.5, 74.25                                              # == fig4_maps_points map size
RICH6 = ListedColormap(plt.get_cmap('YlOrRd')(np.linspace(0.12, 0.72, 6)))   # 6 discrete classes 1..6+
RNORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], 6)
SIGN = plt.get_cmap('PiYG_r')                                    # harmful->pink, beneficial->green
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'reversals'
UNIT = {'reversals': 'flow reversals (per yr)', 'drawdown': 'lethal drawdown (h/yr>13cm/h)'}[METRIC]
SHORT = {'reversals': 'flow reversals', 'drawdown': 'lethal drawdown'}[METRIC]
PANELS = [('%s' % METRIC, 'DC − CONV', 'vs conventional', 'conv'),
          ('%s_bal' % METRIC, 'DC − BAL', 'vs typical / balancing grid', 'bal')]


def legend(tag, V):
    # compact two-part key: discrete richness swatches (left) + sign colorbar (right)
    lf = plt.figure(figsize=(90 * S.MM, 12 * S.MM))
    sax = lf.add_axes([0.02, 0.36, 0.30, 0.30]); sax.set_axis_off()
    for i in range(6):
        sax.add_patch(Rectangle((i, 0), 1, 1, color=RICH6(i / 5), ec='white', lw=0.3))
        sax.text(i + 0.5, -0.45, '6+' if i == 5 else str(i + 1), fontsize=4.5, ha='center', va='top')
    sax.set_xlim(-0.2, 6.2); sax.set_ylim(-1.2, 1.3)
    sax.text(0.0, 1.7, 'Threatened fish species / basin', fontsize=5, ha='left', color='0.25')
    cax = lf.add_axes([0.60, 0.44, 0.30, 0.16])
    cb = lf.colorbar(ScalarMappable(norm=TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V), cmap=SIGN), cax=cax,
                     orientation='horizontal', extend='both')
    m = max(round(0.8 * V / 100) * 100, 1)                       # 3 clean ticks: -m, 0, m
    cb.set_ticks([-m, 0, m])
    cb.ax.tick_params(labelsize=5, width=0.25, length=1.5, pad=1); cb.outline.set_linewidth(0.25)
    lf.text(0.60, 0.78, 'Data-centre effect on %s' % SHORT, fontsize=5, ha='left', color='0.25')
    lf.text(0.585, 0.52, 'beneficial', fontsize=4.5, ha='right', va='center', color='#4d9221')
    lf.text(0.915, 0.52, 'harmful', fontsize=4.5, ha='left', va='center', color='#c51b7d')
    S.save(lf, os.path.join(OUT, 'fig5_map_fish_sign_%s_%s_legend' % (METRIC, tag))); plt.close(lf)


def main():
    world = load_basemap().to_crs(CRS)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')
    ctx = load_rivers_world().to_crs(CRS); ctx = ctx.cx[box[0]:box[2], box[1]:box[3]]

    print('loading lev08 richness ...', flush=True)
    polys = []
    for c in ['na', 'sa', 'eu', 'as', 'au']:
        fp = f'{ATL}/hybas_{c}/hybas_{c}_lev08_v1c.shp'
        if os.path.exists(fp):
            polys.append(gpd.read_file(fp, columns=['HYBAS_ID'])[['HYBAS_ID', 'geometry']])
    hb = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=polys[0].crs)
    hb['hybas_id'] = hb.HYBAS_ID.astype('int64').astype(str)
    look = pd.read_csv(IUCN, dtype={'hybas_id': str})
    hb = hb.merge(look[['hybas_id', 'n_threatened']], on='hybas_id', how='left')
    hb = hb[hb.n_threatened.fillna(0) > 0].copy()
    hb['nt6'] = hb.n_threatened.clip(1, 6)
    hb = hb.to_crs(CRS)

    os.makedirs(OUT, exist_ok=True)
    for stem, title, sub_t, tag in PANELS:
        g = gpd.read_file(os.path.join(DATA, 'reach_cons_%s.gpkg' % stem)).to_crs(CRS)
        g = g[g.eff != 0]
        V = float(np.nanpercentile(np.abs(g.eff), 92)) or 1.0
        snorm = TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V)

        fig, ax = S.fig_mm(W, Ht)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        world.plot(ax=ax, color=LAND, ec='none', zorder=0)
        hb.plot(ax=ax, column='nt6', cmap=RICH6, norm=RNORM, ec='none', alpha=0.82, zorder=1, rasterized=True)
        ctx.plot(ax=ax, color='0.55', linewidth=0.12, alpha=0.33, zorder=2, rasterized=True)
        g.plot(ax=ax, column='eff', cmap=SIGN, norm=snorm, linewidth=0.85, alpha=0.97, zorder=3, rasterized=True)
        apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
        S.save(fig, os.path.join(OUT, 'fig5_map_fish_sign_%s_%s' % (METRIC, tag))); plt.close(fig)
        legend(tag, V)
        print('saved fig5_map_fish_sign_%s_%s (+legend) | V=%.3g' % (METRIC, tag, V))


if __name__ == '__main__':
    main()
