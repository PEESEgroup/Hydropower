"""Fig 1a - global firm-coverage (program-D scenario A, independent) hero map + insets.

Region choropleth coloured by program-D scenario-A (independent stations, no PS, no cascade)
firm coverage = min(supply_A / peak DC demand, 100%) - numerically identical to program-E E0
isolated (met_isolated == supply_A, capped at demand); cov_isolated_pct carries that number.
DC demand drawn as vertical bars (height proportional). Robinson projection. National borders
only for countries that contain a data region; all others stay borderless soft grey.
Colormap via env REVUB_CMAP (default YlGnBu). Output stem via env REVUB_SUFFIX.
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle

import revub_style as S
from revub_geo import load_regions
from revub_worldview import load_basemap, apply_china_worldview
from revub_rivers import load_rivers_world, load_rivers_inset, lw_for, RIVER_BLUE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
NE = os.path.join(HERE, 'gis', 'ne_110m_admin_0_countries.shp')

BUBBLE_FC, BUBBLE_EC = 'white', '0.15'   # DC-demand bubble fill / edge
NOT_ASSESSED = '#efefef'   # countries outside the study (no model)
NO_DC = '#b9b9b9'          # modelled region but no data centre assigned


def load_coverage():
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, 'cov_*.csv')))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['scenario'] == 'A'].drop_duplicates('region')
    return df[['region', 'demand_mw', 'cov_isolated_pct']]


def draw_layer(ax, world, g, border_idx, cmap, norm):
    """Basemap grey (z0) -> opaque coverage choropleth (z1) -> [rivers z2 added separately,
    thin threads over the fill] -> crisp region + national boundaries (z3, above rivers)."""
    world.plot(ax=ax, color=NOT_ASSESSED, ec='none', zorder=0)
    g.plot(ax=ax, column='cov_isolated_pct', cmap=cmap, norm=norm,
           ec='none', zorder=1, missing_kwds={'color': NO_DC, 'ec': 'none'})
    g.boundary.plot(ax=ax, color='0.4', lw=0.45, zorder=3)               # region dividers
    world.iloc[border_idx].boundary.plot(ax=ax, color='0.4', lw=0.45, zorder=3)    # national borders (SAME tone, uniform)
    ax.set_axis_off()
    ax.set_aspect('equal')


def set_view(ax, gdf_sub, padx=0.10, pady=0.12):
    xmin, ymin, xmax, ymax = gdf_sub.total_bounds
    dx, dy = (xmax - xmin), (ymax - ymin)
    ax.set_xlim(xmin - dx * padx, xmax + dx * padx)
    ax.set_ylim(ymin - dy * pady, ymax + dy * pady)
    return (xmin, ymin, xmax, ymax)


def truncate_cmap(name, lo=0.0, hi=0.6, n=256):
    """Use only [lo, hi] of a colormap so the max value isn't the very-dark end."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f'{name}_{lo:.2f}_{hi:.2f}', base(np.linspace(lo, hi, n)))


def latlon_bounds(lon0, lon1, lat0, lat1, crs, n=40):
    """Projected bounds of a lon/lat window (captures Robinson curvature)."""
    lon = np.linspace(lon0, lon1, n)
    lat = np.linspace(lat0, lat1, n)
    LON, LAT = np.meshgrid(lon, lat)
    gs = gpd.GeoSeries(gpd.points_from_xy(LON.ravel(), LAT.ravel()),
                       crs='EPSG:4326').to_crs(crs)
    return tuple(gs.total_bounds)


def apply_bounds(ax, b, padx=0.06, pady=0.08):
    dx, dy = b[2] - b[0], b[3] - b[1]
    out = (b[0] - dx * padx, b[1] - dy * pady, b[2] + dx * padx, b[3] + dy * pady)
    ax.set_xlim(out[0], out[2])
    ax.set_ylim(out[1], out[3])
    return out


def draw_rivers(ax, rivers, bbox, alpha=0.75):
    """Clip rivers to the panel bbox (Robinson) and draw with width ~ drainage area."""
    x0, y0, x1, y1 = bbox
    sub = rivers.cx[x0:x1, y0:y1]
    if len(sub) == 0:
        return
    sub.plot(ax=ax, color=RIVER_BLUE, linewidth=lw_for(sub['order'].to_numpy()),
             alpha=alpha, zorder=2, rasterized=True)   # z2: over fill, under borders; rasterized -> small SVG


def draw_bubbles(ax, g, dmax, smax=320.0):
    """DC-demand proportional bubbles (area ~ demand), constant point-size across panels."""
    has = g[g['demand_mw'].notna()]
    pts = has.representative_point()
    s = smax * has['demand_mw'].to_numpy() / dmax
    ax.scatter(pts.x, pts.y, s=s, facecolor=BUBBLE_FC, alpha=0.55,
               ec=BUBBLE_EC, lw=0.4, zorder=5)
    return smax


def main():
    cov = load_coverage()
    w = cov['demand_mw']
    glob_e0 = 100 * (w * cov['cov_isolated_pct'] / 100).sum() / w.sum()
    print('coverage regions: %d | global E0 (demand-wt) = %.1f%% | median = %.1f%%'
          % (len(cov), glob_e0, cov['cov_isolated_pct'].median()))

    gdf = apply_china_worldview(load_regions()).merge(cov, on='region', how='left')  # EPSG:4326
    g = gdf.to_crs(S.EQUAL_EARTH)                                # Robinson (world hero map)
    world_ll = load_basemap()                                    # China-viewpoint basemap (4326)
    world = world_ll.to_crs(S.EQUAL_EARTH)                       # same row order as world_ll

    dreps = gpd.GeoDataFrame(
        geometry=g[g['cov_isolated_pct'].notna()].representative_point(), crs=g.crs)
    j = gpd.sjoin(dreps, world[['geometry']], predicate='within', how='left')
    border_idx = sorted(set(j['index_right'].dropna().astype(int)))

    cmap = truncate_cmap(os.environ.get('REVUB_CMAP', 'YlGn'), 0.0, 0.65)   # green land -> blue rivers read as water
    norm = Normalize(0, 100)
    dmax = float(g['demand_mw'].max())
    riv_world = load_rivers_world()
    riv_inset = load_rivers_inset()
    riv_inset_ll = riv_inset.to_crs('EPSG:4326')   # insets are drawn in straight (PlateCarree)

    fig = plt.figure(figsize=(210 * S.MM, 123.75 * S.MM))   # 1 A4 wide x 5/12 A4 tall
    ax = fig.add_axes([0.00, 0.34, 0.74, 0.65])                # world map BIGGER (top-left)
    ax_u = fig.add_axes([0.025, 0.04, 0.31, 0.275])            # North America (bottom-left)
    ax_i = fig.add_axes([0.355, 0.04, 0.16, 0.275])            # India (bottom mid-left)
    ax_c = fig.add_axes([0.525, 0.04, 0.215, 0.275])           # China (bottom mid-right)
    ax_e = fig.add_axes([0.76, 0.55, 0.235, 0.42])             # Europe smaller (right, top)
    ax_s = fig.add_axes([0.76, 0.06, 0.235, 0.42])             # Southeast Asia smaller (right, bot)

    draw_layer(ax, world, g, border_idx, cmap, norm)
    # crop far-Pacific but keep mainland Alaska whole on the NW edge
    wbox = apply_bounds(ax, latlon_bounds(-169, 150, -56, 83, g.crs), 0.0, 0.0)
    draw_rivers(ax, riv_world, wbox, alpha=0.85)
    draw_bubbles(ax, g, dmax)

    # Insets are drawn in straight (PlateCarree) projection: EPSG:4326 geometry + aspect 1/cos(lat).
    # Europe uses a tight core-Europe lon/lat window (data bbox was too wide -> inset too small).
    views = [
        (ax_e, (-10.0, 35.0, 31.0, 71.0), 'Europe', 0.0, 0.0),
        (ax_u, tuple(gdf[gdf['bloc'] == 'north_america'].total_bounds),
         'North America', 0.04, 0.06),
        (ax_s, tuple(gdf[gdf['bloc'] == 'southeast_asia'].total_bounds),
         'Southeast Asia', 0.06, 0.08),
        (ax_i, tuple(gdf[gdf['bloc'] == 'india'].total_bounds), 'India', 0.06, 0.10),
        (ax_c, tuple(gdf[gdf['bloc'] == 'china'].total_bounds), 'China', 0.05, 0.08),
    ]
    boxes_ll = []
    for axi, b, name, px, py in views:
        draw_layer(axi, world_ll, gdf, border_idx, cmap, norm)
        bb = apply_bounds(axi, b, px, py)                              # padded lon/lat window
        axi.set_aspect(1.0 / np.cos(np.radians((bb[1] + bb[3]) / 2)))  # straight (PlateCarree)
        boxes_ll.append(bb)
        draw_rivers(axi, riv_inset_ll, bb, alpha=0.85)
        draw_bubbles(axi, gdf, dmax)
        for sp in axi.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color('0.4')
        axi.set_axis_on(); axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(name, fontsize=7, pad=2)

    for bb in boxes_ll:                          # locator rectangles, lon/lat window -> Robinson box
        rb = latlon_bounds(bb[0], bb[2], bb[1], bb[3], g.crs)
        ax.add_patch(Rectangle((rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
                     fill=False, ec='0.35', lw=0.4, zorder=6))

    # Singapore is a ~0.03 deg^2 speck (1.9 GW DC demand, 0% firm coverage) - invisible on the
    # SE-Asia inset. Magnify it in a small corner box on a HIGH-RES coastline: GADM 4.1 for
    # Singapore itself (710 vertices vs Natural Earth's 40) and NE 10m for the neighbours, each
    # land tile coloured by its own firm coverage to match the main choropleth.
    ne10 = gpd.read_file(os.path.join(HERE, 'gis', 'ne_10m_admin_0_countries_chn.shp'))[['ADMIN', 'geometry']]
    sgp = gpd.read_file(os.path.join(HERE, 'gis', 'gadm41_SGP_0.json'))
    covmap = cov.set_index('region')['cov_isolated_pct']
    def _ccol(region):                          # coverage colour for a region (grey if no data)
        v = covmap.get(region)
        return cmap(norm(v)) if v == v else NO_DC
    sg_dem = float(cov.loc[cov['region'] == 'singapore', 'demand_mw'].iloc[0])
    sg_win = (103.52, 1.15, 104.15, 1.52)        # lon0, lat0, lon1, lat1: Singapore + Johor + Batam
    axz = ax_s.inset_axes([0.66, 0.66, 0.33, 0.33], zorder=7)
    ne10.plot(ax=axz, color=NOT_ASSESSED, ec='0.6', lw=0.25, zorder=0)            # context land
    ne10[ne10['ADMIN'] == 'Malaysia'].plot(ax=axz, color=_ccol('malaysia_west'), ec='0.6', lw=0.25, zorder=1)
    ne10[ne10['ADMIN'] == 'Indonesia'].plot(ax=axz, color=_ccol('indonesia'), ec='0.6', lw=0.25, zorder=1)
    sgp.plot(ax=axz, color=_ccol('singapore'), ec='0.3', lw=0.5, zorder=3)        # high-res Singapore by coverage
    axz.scatter([103.82], [1.30], s=320.0 * sg_dem / dmax, facecolor=BUBBLE_FC,   # DC-demand bubble
                alpha=0.7, ec=BUBBLE_EC, lw=0.4, zorder=4)
    axz.set_xlim(sg_win[0], sg_win[2]); axz.set_ylim(sg_win[1], sg_win[3])
    axz.set_aspect(1.0 / np.cos(np.radians((sg_win[1] + sg_win[3]) / 2)))
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color('0.4')
    axz.set_title('Singapore', fontsize=6, pad=1.5)
    ax_s.indicate_inset_zoom(axz, edgecolor='0.35', lw=0.4, alpha=0.9)

    # Legend (colorbar + no-data swatch + DC-demand bubble sizes) is rendered SEPARATELY
    # by fig_legends.py -> out/fig1a_legend.* for manual compositing.

    os.makedirs(OUT, exist_ok=True)
    stem = 'fig1a_coverage' + os.environ.get('REVUB_SUFFIX', '')
    S.save(fig, os.path.join(OUT, stem))
    print('borders for %d data countries | saved: %s.svg (+.png)'
          % (len(border_idx), os.path.join(OUT, stem)))


if __name__ == '__main__':
    main()
