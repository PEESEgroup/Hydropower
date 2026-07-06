"""Fig 4 maps - DC vs conventional river impact, drawn on the NATIVE river network.

One world map per indicator (ecology flow-reversals / sediment STCI / supply irrigation-gap). For each
DC-serving dam/segment we walk RiverATLAS (HydroRIVERS) downstream via NEXT_DOWN - exactly as
F3_prepare_routing does - and colour every reach it controls by the DC_A - CONV effect (diverging
RdBu_r, warm = harmful, cool = beneficial, robust +/-p92). Built by fig4_maps_build.py into
data/reach_<stem>.gpkg (native topology -> all dams matched, unlike GRFR's 11-54 %). China-worldview
basemap; South America's southern tip is kept. Map = 3/4 A4 wide x 1/4 A4 tall; colorbar drawn
SEPARATELY. Arial 7 pt. Run: python fig4_maps.py [stem]
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

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); OUT = os.path.join(HERE, 'out')
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'; CMAP = plt.get_cmap('RdBu_r')
W, Ht = 157.5, 74.25
MAPS = [('reach_reversals.gpkg', 'reversals', 'Change in flow reversals (per year)'),
        ('reach_stci.gpkg',      'sediment',  'Change in sediment-transport index'),
        ('reach_gap.gpkg',       'irrigationgap', 'Change in irrigation gap (months/yr)')]


def one(world, box, gpkg, stem, title):
    g = gpd.read_file(os.path.join(DATA, gpkg)).to_crs(CRS)
    g = g[np.isfinite(g.eff)]
    g = g.iloc[np.argsort(np.abs(g.eff.to_numpy()))]            # strongest reaches drawn on top
    V = float(np.nanpercentile(np.abs(g.eff), 92)) or 1.0
    norm = TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V)
    fig, ax = S.fig_mm(W, Ht)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    world.plot(ax=ax, color=LAND, ec='none', zorder=0)
    g.plot(ax=ax, column='eff', cmap=CMAP, norm=norm, linewidth=0.45, alpha=0.95, zorder=3,
           rasterized=True)
    apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
    S.save(fig, os.path.join(OUT, 'fig4_map_' + stem)); plt.close(fig)
    lf = plt.figure(figsize=(70 * S.MM, 16 * S.MM)); cax = lf.add_axes([0.08, 0.5, 0.84, 0.12])
    cb = lf.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', extend='both')
    cb.set_label(title + '   (DC vs conventional)', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    lf.text(0.08, 0.80, 'beneficial', fontsize=6.5, ha='left', color='#2e6e8e')
    lf.text(0.92, 0.80, 'harmful', fontsize=6.5, ha='right', color='#b03030')
    S.save(lf, os.path.join(OUT, 'fig4_map_%s_legend' % stem)); plt.close(lf)
    print('saved fig4_map_%s | %d reaches | V=%.3g' % (stem, len(g), V))


def main():
    world = load_basemap().to_crs(CRS)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')
    for gpkg, stem, title in [m for m in MAPS if (len(sys.argv) < 2 or m[1] == sys.argv[1])]:
        one(world, box, gpkg, stem, title)


if __name__ == '__main__':
    main()
