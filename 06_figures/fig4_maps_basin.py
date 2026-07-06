"""Fig 4 maps (basin-fill) - DC vs conventional river impact aggregated to HydroBASINS polygons.

Each DC-serving dam is point-in-polygon joined to its HydroBASINS sub-basin (level LEVEL); the basin is
filled with the MEAN DC_A - CONV effect of the dams it contains (diverging RdBu_r, warm = harmful,
cool = beneficial, robust +/-p92). Basins with no DC dam stay on the plain land basemap. China-
worldview basemap; South America's southern tip kept. Map = 3/4 A4 wide x 1/4 A4 tall; colorbar drawn
SEPARATELY. Arial 7 pt. Run: python fig4_maps_basin.py [stem] [level]
"""
import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

import revub_style as S
from revub_worldview import load_basemap
from revub_rivers import load_rivers_world, RIVER_BLUE
from fig1a_coverage_map import latlon_bounds, apply_bounds

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data'); OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
ATLAS = '/home/cfeng/hydro/source_data/atlas'; REGS = ['eu', 'as', 'au', 'na', 'sa']
DEFAULT_LEVEL = 5                                         # grey tessellation grid AND colour fills share this level (aligned cells)
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'; CMAP = plt.get_cmap('RdBu_r')
W, Ht = 157.5, 74.25
# ecology (reversals, lethal drawdown) is drawn as POINTS (at-a-station) by fig4_maps_points.py;
# basin-fill is for the catchment-integrated metrics only - sediment (STCI) and supply (irrigation gap).
MAPS = [('map_stci.csv', 'sediment',      'Change in sediment-transport index'),
        ('map_gap.csv',  'irrigationgap', 'Change in irrigation gap (months/yr)')]


def load_basins(level):
    parts = []
    for r in REGS:
        p = '%s/hybas_%s/hybas_%s_lev%02d_v1c.shp' % (ATLAS, r, r, level)
        if os.path.exists(p):
            parts.append(gpd.read_file(p, columns=['HYBAS_ID'])[['HYBAS_ID', 'geometry']])
    b = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(b, crs='EPSG:4326')


def one(world, ctx, basins, box, csv, stem, title, level, how):
    d = pd.read_csv(os.path.join(DATA, csv)).dropna(subset=['lat', 'lon', 'eff']).copy()   # ALL DC-serving dams (incl 0)
    pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs='EPSG:4326')
    j = gpd.sjoin(pts, basins, how='inner', predicate='within')
    g = j.groupby('HYBAS_ID')['eff']
    if how == 'maxabs':                                        # worst-case dam (largest |effect|, signed)
        agg = j.loc[g.apply(lambda s: s.abs().idxmax()).values].set_index('HYBAS_ID')['eff']
    else:
        agg = g.mean()
    # in-scope = basins that contain a DC-serving dam; zero-change ones are KEPT and drawn white.
    col = basins.merge(agg.rename('val'), on='HYBAS_ID', how='inner').to_crs(CRS)
    V = float(np.nanpercentile(np.abs(col['val']), 92)) or 1.0
    norm = TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V)
    fig, ax = S.fig_mm(W, Ht)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    world.plot(ax=ax, color=LAND, ec='none', zorder=0)
    ctx.plot(ax=ax, color=RIVER_BLUE, linewidth=0.12, alpha=0.30, zorder=1, rasterized=True)   # faint GRFR rivers
    col.plot(ax=ax, column='val', cmap=CMAP, norm=norm, ec='0.5', linewidth=0.15, alpha=0.97, zorder=2)  # grey edge so white (0) basins show
    apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
    S.save(fig, os.path.join(OUT, 'fig4_map_' + stem)); plt.close(fig)
    lf = plt.figure(figsize=(82 * S.MM, 26 * S.MM)); cax = lf.add_axes([0.08, 0.60, 0.84, 0.12])
    cb = lf.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', extend='both')
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    lf.text(0.08, 0.95, 'beneficial', fontsize=6.5, ha='left', va='top', color='#2e6e8e')
    lf.text(0.92, 0.95, 'harmful', fontsize=6.5, ha='right', va='top', color='#b03030')
    sub = 'basin worst-case' if how == 'maxabs' else 'basin mean'
    lf.text(0.50, 0.16, title + '   (DC vs conventional, %s)' % sub, fontsize=7, ha='center', va='center')
    S.save(lf, os.path.join(OUT, 'fig4_map_%s_legend' % stem)); plt.close(lf)
    print('saved fig4_map_%s | lev%02d | %d basins | V=%.3g' % (stem, level, len(col), V))


def main():
    stem_sel = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
    level = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else DEFAULT_LEVEL
    how = 'maxabs' if 'maxabs' in sys.argv else 'mean'          # default MEAN (faithful for neutral sediment)
    world = load_basemap().to_crs(CRS)
    rivers = load_rivers_world().to_crs(CRS)
    basins = load_basins(level)                               # fill level (point-in-polygon + colour)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')
    ctx = rivers.cx[box[0]:box[2], box[1]:box[3]]
    for csv, stem, title in [m for m in MAPS if (stem_sel is None or m[1] == stem_sel)]:
        one(world, ctx, basins, box, csv, stem, title, level, how)


if __name__ == '__main__':
    main()
