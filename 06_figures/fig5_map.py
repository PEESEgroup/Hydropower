"""Fig 5 conservation map: the DC impact REACHES (HydroRIVERS lines) coloured by the DC effect
(blue beneficial, red harmful), with reaches inside PROTECTED AREAS drawn bold on top. Two panels:
left = DC - CONV (vs conventional hydropower), right = DC - BAL (vs typical balancing-grid operation).
This shows WHERE the flow-regime change lands and whether it helps or harms protected rivers.

Metric selectable (default reversals). China-worldview basemap; Equal-Earth.
Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_map.py [reversals|drawdown]
"""
import os, sys
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

import revub_style as S
from revub_worldview import load_basemap
from fig1a_coverage_map import latlon_bounds, apply_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'; CMAP = plt.get_cmap('RdBu_r')
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'reversals'
PANELS = [('%s' % METRIC, 'vs conventional', 'DC − CONV'),
          ('%s_bal' % METRIC, 'vs typical (balancing grid)', 'DC − BAL')]
TITLE = {'reversals': 'Change in flow reversals (per year)',
         'drawdown': 'Change in lethal drawdown (h/yr > 13 cm/h)'}[METRIC]


def main():
    world = load_basemap().to_crs(CRS)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')

    # common diverging scale from the CONV layer (robust p92 of |eff|)
    g0 = gpd.read_file(os.path.join(DATA, 'reach_cons_%s.gpkg' % METRIC))
    V = float(np.nanpercentile(np.abs(g0.eff[g0.eff != 0]), 92)) or 1.0
    norm = TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V)

    fig, axes = plt.subplots(1, 2, figsize=(180 * S.MM, 78 * S.MM))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.04, wspace=0.02)
    for ax, (stem, sub, short) in zip(axes, PANELS):
        g = gpd.read_file(os.path.join(DATA, 'reach_cons_%s.gpkg' % stem)).to_crs(CRS)
        g = g[g.eff != 0]
        world.plot(ax=ax, color=LAND, ec='none', zorder=0)
        base = g[~g.in_pa]; pa = g[g.in_pa]
        base.plot(ax=ax, column='eff', cmap=CMAP, norm=norm, linewidth=0.35,
                  alpha=0.7, zorder=2, rasterized=True)
        pa.plot(ax=ax, color='white', linewidth=2.2, alpha=0.9, zorder=3, rasterized=True)   # casing
        pa.plot(ax=ax, column='eff', cmap=CMAP, norm=norm, linewidth=1.3,
                alpha=1.0, zorder=4, rasterized=True)
        apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
        ax.set_title(short + '   (%s)' % sub, fontsize=7.5, pad=2)
    fig.suptitle(TITLE + '   -   line colour = DC effect; bold = inside protected area',
                 fontsize=8, y=0.99)

    # shared colorbar
    cax = fig.add_axes([0.30, 0.05, 0.40, 0.022])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    fig.text(0.295, 0.062, 'beneficial', fontsize=6.5, ha='right', va='center', color='#2e6e8e')
    fig.text(0.705, 0.062, 'harmful', fontsize=6.5, ha='left', va='center', color='#b03030')

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig5_map_%s' % METRIC)); plt.close(fig)
    print('saved fig5_map_%s | V=%.3g | PA reaches CONV=%d BAL=%d'
          % (METRIC, V, int(g0.in_pa.sum()),
             int(gpd.read_file(os.path.join(DATA, 'reach_cons_%s_bal.gpkg' % METRIC)).in_pa.sum())))


if __name__ == '__main__':
    main()
