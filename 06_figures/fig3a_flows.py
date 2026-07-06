"""Fig 3a - cross-region interconnection: net trade (choropleth) + corridor flows (arrows).

Interconnected (E1) scenario, scenario A. Each region is shaded by its NET firm trade
(net_export_mw, colorcet CET-CBTD1 tritanopic cyan/white/salmon): salmon = net importer (pulls
firm power in), cyan = net exporter (sends firm power out). Each transmission corridor that
carries power is a curved arrow between region centroids pointing toward the net importer; width
~ sqrt(used_mw). delivered/used is firm power (not an hourly mean). PlateCarree (straight)
projection; per-continent facets sized to their own extent (Europe enlarged, India shrunk).

Two map variants via REVUB_FIG3A_MODE:
  'small' (default) - FINAL figure: no labels, no titles (composited/labelled by hand), A4 x 1/3.
  'big'             - large LABELLED reference (all corridor hubs named) to guide hand-labelling.
The legend (net-trade scale WITH magnitude + corridor-width key) is always written SEPARATELY to
out/fig3a_flows_legend.* for manual placement. All text Arial 7 pt; transparent; no title.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from adjustText import adjust_text
import colorcet as cc

import revub_style as S
from revub_geo import load_regions
from revub_worldview import load_basemap, apply_china_worldview
from fig1b_voronoi import _disp_name

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')

COR_THRESH = 50.0
M = 8000.0            # net-export half-range (MW) for the diverging choropleth
C_LINE = '#08519c'
NOT_ASSESSED = '#eef0f2'
NEUTRAL = '#f7f7f7'
NAMES = {'china': 'China', 'europe': 'Europe', 'north_america': 'North America',
         'india': 'India', 'southeast_asia': 'Southeast Asia'}
MODE = os.environ.get('REVUB_FIG3A_MODE', 'small')   # 'small' (final, unlabelled) | 'big' (labelled)
# explicit (lon0, lon1, lat0, lat1) windows for blocs whose total_bounds is blown up by far
# overseas territories - Europe only (NL Caribbean -68°, ES Canaries -18°). Core window still
# spans UK/Ireland (-12) to Turkey (45, keeps the Bulgaria-Turkey corridor) and Greece->N.Cape.
EXTENT_OVERRIDE = {'europe': (-12.0, 45.0, 34.0, 72.0)}

# per-continent panel rectangles. SMALL = final strip: Europe = big tall left column; North
# America big top-right; China(top)/SE-Asia(bottom) middle; India under N. America. Sized to each
# continent's need (Europe & N. America largest), not a rigid grid. BIG = labelled reference.
SMALL_RECTS = {'europe': [0.0, 0.01, 0.31, 0.93],
               'china': [0.32, 0.52, 0.30, 0.42], 'southeast_asia': [0.32, 0.05, 0.30, 0.44],
               'north_america': [0.61, 0.44, 0.39, 0.50], 'india': [0.63, 0.03, 0.32, 0.40]}
BIG_RECTS = {'china': [0.005, 0.52, 0.33, 0.46], 'europe': [0.345, 0.52, 0.32, 0.46],
             'north_america': [0.675, 0.52, 0.32, 0.46], 'india': [0.04, 0.02, 0.26, 0.46],
             'southeast_asia': [0.34, 0.02, 0.32, 0.46]}


def lw_for(mw, pmax):
    # thinner overall and the very-thick end compressed (was 0.5 + 4.2*sqrt -> max 4.7)
    return 0.3 + 2.0 * np.sqrt(np.maximum(mw, 0) / pmax)


def main():
    cor = pd.read_csv(os.path.join(DATA, 'fig3a_corridors.csv'))
    cor = cor[cor['used_mw'] >= COR_THRESH].copy()
    tr = pd.read_csv(os.path.join(DATA, 'fig3a_trade.csv'))[['region', 'net_export_mw']]

    proj = os.environ.get('REVUB_FIG3A_PROJ', 'pc')   # 'pc' (straight) or 'robin'
    CRS = S.EQUAL_EARTH if proj == 'robin' else 'EPSG:4326'
    g = apply_china_worldview(load_regions()).merge(tr, on='region', how='left').to_crs(CRS)
    bloc = g.set_index('region')['bloc'].to_dict()
    net = g.set_index('region')['net_export_mw'].fillna(0).to_dict()
    cen = g.set_index('region').geometry.representative_point()
    cx, cy = cen.x.to_dict(), cen.y.to_dict()
    cor = cor[cor['region_a'].isin(cx) & cor['region_b'].isin(cx)]
    cor['bloc'] = cor['region_a'].map(bloc)
    pmax = float(cor['used_mw'].max())
    cor = cor.sort_values('used_mw')
    world = load_basemap().to_crs(CRS).reset_index(drop=True)   # China-viewpoint basemap
    cmap = cc.cm['CET_CBTD1'].reversed()   # cyan/white/salmon, reversed -> importer=salmon
    # net trade is heavy-tailed (median ~0.5 GW, max ~12 GW) -> a linear scale leaves most regions
    # near-white. Map a SIGNED-SQRT of net trade so the bulk deepens while the tail keeps its order.
    g['net_t'] = np.sign(g['net_export_mw'].fillna(0.0)) * np.sqrt(np.abs(g['net_export_mw'].fillna(0.0)))
    MT = np.sqrt(M)
    norm = TwoSlopeNorm(vmin=-MT, vcenter=0.0, vmax=MT)

    def draw_map(ax, gsub, corsub, head):
        world.plot(ax=ax, color=NOT_ASSESSED, ec='none', zorder=0)
        gsub.plot(ax=ax, column='net_t', cmap=cmap, norm=norm, ec='0.55', lw=0.25,
                  zorder=1, missing_kwds={'color': NEUTRAL, 'ec': '0.7', 'lw': 0.2})
        for _, r in corsub.iterrows():
            a, b = r['region_a'], r['region_b']
            if net.get(a, 0) < net.get(b, 0):      # arrow points toward the net importer
                a, b = b, a
            ax.add_patch(FancyArrowPatch((cx[a], cy[a]), (cx[b], cy[b]), arrowstyle='-|>',
                         connectionstyle='arc3,rad=0.16', mutation_scale=head,
                         lw=lw_for(r['used_mw'], pmax), color=C_LINE, alpha=0.85,
                         capstyle='round', zorder=3))

    def set_view(ax, gsub, bbox=None, padx=0.05, pady=0.07):
        # frame the FULL region: explicit core window if given (Europe), else the bloc's full
        # geometry bounds (every polygon) so each panel shows the whole region uncropped
        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
        else:
            xmin, ymin, xmax, ymax = gsub.total_bounds
        dx, dy = (xmax - xmin) or 1, (ymax - ymin) or 1
        ax.set_xlim(xmin - dx * padx, xmax + dx * padx)
        ax.set_ylim(ymin - dy * pady, ymax + dy * pady)
        ax.set_aspect(1.0 / np.cos(np.radians((ymin + ymax) / 2)) if proj == 'pc' else 'equal')
        ax.set_axis_off()

    def sgp_inset(ax):   # mimic Fig1a: magnify Singapore (a ~0.03 deg^2 net importer) in a corner box
        ne10 = gpd.read_file(os.path.join(HERE, 'gis', 'ne_10m_admin_0_countries_chn.shp'))[['ADMIN', 'geometry']]
        sgp = gpd.read_file(os.path.join(HERE, 'gis', 'gadm41_SGP_0.json'))
        ntmap = g.set_index('region')['net_t']
        def _ncol(region):
            v = ntmap.get(region)
            return cmap(norm(v)) if v == v else NEUTRAL
        win = (103.52, 1.15, 104.15, 1.52)        # Singapore + Johor + Batam
        axz = ax.inset_axes([0.60, 0.62, 0.37, 0.37], zorder=7)
        ne10.plot(ax=axz, color=NOT_ASSESSED, ec='0.6', lw=0.25, zorder=0)
        ne10[ne10['ADMIN'] == 'Malaysia'].plot(ax=axz, color=_ncol('malaysia_west'), ec='0.6', lw=0.25, zorder=1)
        ne10[ne10['ADMIN'] == 'Indonesia'].plot(ax=axz, color=_ncol('indonesia'), ec='0.6', lw=0.25, zorder=1)
        sgp.plot(ax=axz, color=_ncol('singapore'), ec='0.3', lw=0.5, zorder=3)
        axz.set_xlim(win[0], win[2]); axz.set_ylim(win[1], win[3])
        axz.set_aspect(1.0 / np.cos(np.radians((win[1] + win[3]) / 2)))
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color('0.4')
        axz.set_title('Singapore', fontsize=6, pad=1.5)
        ax.indicate_inset_zoom(axz, edgecolor='0.35', lw=0.4, alpha=0.9)

    def add_labels(ax, corsub):   # label EVERY corridor hub (big mode only); de-overlapped, clipped
        nodes = list(set(corsub['region_a']) | set(corsub['region_b']))
        texts = [ax.text(cx[n], cy[n], _disp_name(n).replace('\n', ' '), fontsize=5.5,
                 color='0.05', ha='center', va='center', zorder=6, clip_on=True,
                 path_effects=[pe.withStroke(linewidth=1.1, foreground='white')]) for n in nodes]
        if texts:
            adjust_text(texts, ax=ax, expand=(1.05, 1.2), force_text=(0.3, 0.5),
                        arrowprops=dict(arrowstyle='-', lw=0.3, color='0.55', clip_on=True))

    os.makedirs(OUT, exist_ok=True)
    # 'panels' mode: each continent as its OWN file at the SAME size it has in the small layout
    # (figure = its SMALL_RECTS slot + a thin title strip), for hand compositing. Plus the legend.
    if MODE == 'panels':
        TITLE_MM = 4.5
        for b, rect in SMALL_RECTS.items():
            w_mm, h_mm = rect[2] * 210, rect[3] * 99
            fig = plt.figure(figsize=(w_mm * S.MM, (h_mm + TITLE_MM) * S.MM))
            ax = fig.add_axes([0.0, 0.0, 1.0, h_mm / (h_mm + TITLE_MM)])
            cs, gs = cor[cor['bloc'] == b], g[g['bloc'] == b]
            draw_map(ax, gs, cs, 5)
            set_view(ax, gs, EXTENT_OVERRIDE.get(b))
            if b == 'southeast_asia':
                sgp_inset(ax)
            ax.set_title(NAMES[b], fontsize=7, pad=2)
            S.save(fig, os.path.join(OUT, 'fig3a_panel_%s' % b))
            plt.close(fig)
        save_legend(cmap, norm, pmax)
        print('panels: saved 5 fig3a_panel_*.svg(+png) + fig3a_flows_legend')
        return

    rects = BIG_RECTS if MODE == 'big' else SMALL_RECTS
    figsize = (210 * S.MM, 230 * S.MM) if MODE == 'big' else (210 * S.MM, 99 * S.MM)
    head = 6 if MODE == 'big' else 5            # arrowhead size (mutation_scale): smaller now
    fig = plt.figure(figsize=figsize)
    for b, rect in rects.items():
        ax = fig.add_axes(rect)
        cs = cor[cor['bloc'] == b]
        gs = g[g['bloc'] == b]
        draw_map(ax, gs, cs, head)
        set_view(ax, gs, EXTENT_OVERRIDE.get(b))
        ax.set_title(NAMES[b], fontsize=7, pad=2)   # panel name kept in both modes
        if MODE == 'big':                            # labelled reference also names every hub
            add_labels(ax, cs)
        if b == 'southeast_asia':                    # Singapore magnifier (mimics Fig1a)
            sgp_inset(ax)

    os.makedirs(OUT, exist_ok=True)
    stem = 'fig3a_flows_%s_%s' % (MODE, 'pc' if proj == 'pc' else 'robin')
    S.save(fig, os.path.join(OUT, stem))
    save_legend(cmap, norm, pmax)
    print('mode=%s proj=%s corridors=%d pmax=%.0f saved %s (+legend)'
          % (MODE, proj, len(cor), pmax, stem))


def save_legend(cmap, norm, pmax):
    """Standalone legend (placed by hand): net-trade colour scale WITH magnitude + corridor key."""
    fig = plt.figure(figsize=(90 * S.MM, 38 * S.MM))
    # net-trade colorbar (smaller) with numeric (GW) ticks and importer/exporter end labels
    cax = fig.add_axes([0.10, 0.66, 0.44, 0.055])
    _tg = [-8, -4, -1, 0, 1, 4, 8]   # GW tick values, placed at their signed-sqrt positions
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                      extend='both', ticks=[np.sign(v) * np.sqrt(abs(v) * 1000.0) for v in _tg])
    cb.ax.set_xticklabels(['%+d' % v if v else '0' for v in _tg])
    cb.set_label('Net firm trade (GW)', fontsize=7, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6.5)
    fig.text(0.10, 0.86, 'net importer', fontsize=6, ha='left', color='0.2')
    fig.text(0.54, 0.86, 'net exporter', fontsize=6, ha='right', color='0.2')
    # corridor-width reference (firm power carried)
    axw = fig.add_axes([0.10, 0.04, 0.84, 0.42]); axw.set_axis_off()
    h = [mlines.Line2D([], [], color=C_LINE, lw=lw_for(v * 1000, pmax), label='%d GW' % v)
         for v in (1, 4, 7)]
    axw.legend(handles=h, loc='center left', frameon=False, fontsize=7, ncol=3,
               title='Corridor flow (arrow toward importer)', title_fontsize=7,
               columnspacing=1.6, handlelength=2.2)
    S.save(fig, os.path.join(OUT, 'fig3a_flows_legend'))


if __name__ == '__main__':
    main()
