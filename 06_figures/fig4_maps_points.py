"""Fig 4 maps (points) - ECOLOGY metrics, DC vs conventional, one small mark per DC-serving dam.

Ecology metrics (flow reversals, lethal drawdown) are at-a-station / channel quantities - the dam is
where the hydropeaking originates - so each is drawn as ONE small point at the dam: colour = DC_A -
CONV effect (diverging RdBu_r, warm = harmful, cool = beneficial, robust +/-p92), size ~ |effect|.
Faint GRFR big-river network for geographic context (NO national borders). China-worldview basemap;
South America's southern tip kept. Map = 3/4 A4 wide x 1/4 A4 tall; colorbar + size key drawn
SEPARATELY. Arial 7 pt. Run: python fig4_maps_points.py [stem]
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
CRS = S.EQUAL_EARTH; LAND = '#eef0f2'; CMAP = plt.get_cmap('RdBu_r')
W, Ht = 157.5, 74.25
SMIN, SMAX = 1.4, 11.0                                   # point area (pt^2): small .. moderate (keep small)
MAPS = [('map_rev.csv',      'reversals', 'Change in flow reversals (per year)'),
        ('map_drawdown.csv', 'drawdown',  'Change in lethal drawdown (h/yr > 13 cm/h)')]


def sizes(a, V):
    return SMIN + (SMAX - SMIN) * np.sqrt(np.clip(np.abs(a) / V, 0, 1.5) / 1.5)


def one(world, ctx, box, csv, stem, title):
    d = pd.read_csv(os.path.join(DATA, csv)).dropna(subset=['lat', 'lon', 'eff'])
    d = d[np.abs(d.eff) > 1e-9].copy()                   # affected dams only
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs='EPSG:4326').to_crs(CRS)
    V = float(np.nanpercentile(np.abs(d.eff), 92)) or 1.0
    norm = TwoSlopeNorm(vmin=-V, vcenter=0, vmax=V)
    order = np.argsort(np.abs(g['eff'].to_numpy()))      # strongest on top
    fig, ax = S.fig_mm(W, Ht)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    world.plot(ax=ax, color=LAND, ec='none', zorder=0)
    ctx.plot(ax=ax, color=RIVER_BLUE, linewidth=0.12, alpha=0.35, zorder=1, rasterized=True)   # faint GRFR rivers
    ax.scatter(g.geometry.x.to_numpy()[order], g.geometry.y.to_numpy()[order],
               c=d['eff'].to_numpy()[order], cmap=CMAP, norm=norm,
               s=sizes(d['eff'].to_numpy()[order], V), lw=0, alpha=0.85, zorder=3)
    apply_bounds(ax, box, 0.0, 0.0); ax.set_aspect('equal'); ax.set_axis_off()
    S.save(fig, os.path.join(OUT, 'fig4_map_' + stem)); plt.close(fig)

    # separate legend: top row = colorbar (left) + size key (right); bottom row = centered title
    lf = plt.figure(figsize=(82 * S.MM, 30 * S.MM))
    cax = lf.add_axes([0.06, 0.66, 0.60, 0.11])
    cb = lf.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', extend='both')
    cb.set_ticks(cb.get_ticks())
    cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4)
    lf.text(0.06, 0.95, 'beneficial', fontsize=6.5, ha='left', va='top', color='#2e6e8e')
    lf.text(0.66, 0.95, 'harmful', fontsize=6.5, ha='right', va='top', color='#b03030')
    lf.text(0.50, 0.12, title + '   (DC vs conventional)', fontsize=7, ha='center', va='center')
    sax = lf.add_axes([0.76, 0.30, 0.22, 0.50]); sax.set_axis_off()
    keys = np.array([0.25, 0.75, 1.5]) * V
    sax.scatter(np.zeros(3), np.arange(3), s=sizes(keys, V), color='0.45', lw=0)
    for i, k in enumerate(keys):
        sax.text(0.6, i, '%.3g' % k, fontsize=6, va='center', ha='left')
    sax.set_xlim(-0.6, 3.2); sax.set_ylim(-0.7, 3.0)
    sax.text(0.0, 2.65, '|effect|', fontsize=6, ha='center', color='0.3')
    S.save(lf, os.path.join(OUT, 'fig4_map_%s_legend' % stem)); plt.close(lf)
    print('saved fig4_map_%s | n=%d V=%.3g' % (stem, len(d), V))


def main():
    world = load_basemap().to_crs(CRS)
    rivers = load_rivers_world().to_crs(CRS)
    box = apply_bounds(plt.figure().add_subplot(), latlon_bounds(-150, 150, -56, 74, CRS), 0.0, 0.0)
    plt.close('all')
    ctx = rivers.cx[box[0]:box[2], box[1]:box[3]]
    for csv, stem, title in [m for m in MAPS if (len(sys.argv) < 2 or m[1] == sys.argv[1])]:
        one(world, ctx, box, csv, stem, title)


if __name__ == '__main__':
    main()
