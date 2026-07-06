"""SI figure: subnational grid regions of the six subdivided countries, with geographic names.

Four panels (United States, China, India, Brazil) showing the grid subregions filled with
distinct colours and labelled by the geographic name used in the study (revub_names). Accompanies
the naming table in SI Section 1. China-worldview geometry; output PDF to the SI figures folder.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from revub_geo import load_regions
from revub_worldview import load_basemap, apply_china_worldview
from revub_names import region_label
try:
    from adjustText import adjust_text
    HAVE_ADJ = True
except Exception:
    HAVE_ADJ = False

OUT = '/home/cfeng/myswat/si/sifigures'
LAND = '#f3f1ec'
HALO = [pe.withStroke(linewidth=1.3, foreground='white')]
PAL = [plt.get_cmap('tab20b')(i / 20) for i in range(20)] + [plt.get_cmap('tab20c')(i / 20) for i in range(20)]
GROUPS = [('United States', lambda g: g.bloc == 'north_america' and False)]  # placeholder, replaced below


def sel_country(g, prefix):
    return g[g.region.str.startswith(prefix)].copy()


def panel(ax, sub, title, fs=7.0):
    ax.set_rasterization_zorder(4)  # rasterize basemap/fills, keep labels+dots vector
    world = load_basemap()
    sub = sub.sort_values('region').reset_index(drop=True)
    xmin, ymin, xmax, ymax = sub.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    px, py = 0.10, 0.12
    x0, x1, y0, y1 = xmin - dx * px, xmax + dx * px, ymin - dy * py, ymax + dy * py
    world.plot(ax=ax, color=LAND, ec='0.78', lw=0.3, zorder=0)
    sub.plot(ax=ax, color=PAL[:len(sub)], ec='white', lw=0.6, alpha=0.85, zorder=1)
    sub.boundary.plot(ax=ax, color='0.35', lw=0.4, zorder=2)
    texts = []
    for (_, r), p in zip(sub.iterrows(), sub.representative_point()):
        ax.plot(p.x, p.y, 'o', ms=1.6, color='0.2', zorder=5)
        texts.append(ax.text(p.x, p.y, region_label(r.region), fontsize=fs, ha='center',
                             va='center', color='0.10', zorder=6, path_effects=HALO))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.radians((y0 + y1) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('0.5')
    ax.set_title(title, fontsize=fs + 2, pad=3)
    if HAVE_ADJ and len(texts) > 1:
        try:
            adjust_text(texts, ax=ax, expand=(1.3, 1.6), force_text=(0.4, 0.7),
                        arrowprops=dict(arrowstyle='-', color='0.45', lw=0.4))
        except Exception:
            pass


def main():
    g = apply_china_worldview(load_regions())
    fig = plt.figure(figsize=(9.2, 7.6))
    ax_us = fig.add_axes([0.02, 0.50, 0.96, 0.47])
    ax_cn = fig.add_axes([0.02, 0.02, 0.31, 0.45])
    ax_in = fig.add_axes([0.35, 0.02, 0.30, 0.45])
    ax_br = fig.add_axes([0.67, 0.02, 0.31, 0.45])
    panel(ax_us, sel_country(g, 'usa_'), 'United States')
    cn = g[g.region.str.startswith('china_') | (g.region == 'hong_kong')].copy()
    panel(ax_cn, cn, 'China')
    panel(ax_in, sel_country(g, 'india_'), 'India')
    panel(ax_br, sel_country(g, 'brazil_'), 'Brazil')
    out = os.path.join(OUT, 'grid_subregions')
    fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white', dpi=200)
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out + '.pdf')


if __name__ == '__main__':
    main()
