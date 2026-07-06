"""Fig 1a - subregion NAME KEY (reference only, NOT a paper panel).

Large standalone reference maps that label every sub-national region for the four multi-region
groups whose Fig-1a insets are too small to letter in place: North America (US grids + Canada),
China, India, Brazil. Each group is drawn as its OWN big file so the labels have room. Every
subregion is filled a distinct colour, outlined, and labelled with its canonical geographic name
(revub_names); US subregions carry a ' (US)' tag. Labels are de-overlapped with adjustText (leader
lines) when available. Same China-worldview geometry as the real Fig 1a, so positions match.
Purpose: the user hand-places these names onto the Fig-1a insets.

Also writes a 4-panel overview (fig1a_subregion_key_overview).
Run: python fig1a_subregion_key.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import revub_style as S
from revub_geo import load_regions
from revub_worldview import load_basemap, apply_china_worldview
from revub_names import region_label, region_label_us

try:
    from adjustText import adjust_text
    HAVE_ADJUST = True
except Exception:
    HAVE_ADJUST = False

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
LAND = '#f3f1ec'          # neutral surrounding land
LABEL_FS = 5              # match the Fig 1b large-reference label font (fontsize=5, color 0.2)
LABEL_COL = '0.2'
HALO = [pe.withStroke(linewidth=1.2, foreground='white')]
# 40 distinct, non-grey fills (tab20b + tab20c)
PALETTE = [plt.get_cmap('tab20b')(i / 20) for i in range(20)] + \
          [plt.get_cmap('tab20c')(i / 20) for i in range(20)]

GROUPS = [   # (key, title, selector, label_fn, figsize, pad)
    ('north_america', 'North America (US grids + Canada)',
     lambda g: g.bloc == 'north_america', region_label_us, (16, 10), (0.12, 0.14)),
    ('china', 'China', lambda g: g.bloc == 'china', region_label, (12, 10), (0.10, 0.12)),
    ('india', 'India', lambda g: g.bloc == 'india', region_label, (9, 11), (0.12, 0.14)),
    ('brazil', 'Brazil', lambda g: g.region.str.startswith('brazil_'), region_label, (10, 11), (0.12, 0.14)),
]


def draw_panel(ax, world, g, sel, title, label_fn, pad=(0.10, 0.12), titlesize=12):
    sub = g[sel(g)].copy().sort_values('region').reset_index(drop=True)
    xmin, ymin, xmax, ymax = sub.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    px, py = pad
    x0, x1 = xmin - dx * px, xmax + dx * px
    y0, y1 = ymin - dy * py, ymax + dy * py

    world.plot(ax=ax, color=LAND, ec='0.78', lw=0.3, zorder=0)            # context land + borders
    sub.plot(ax=ax, color=PALETTE[:len(sub)], ec='white', lw=0.7, alpha=0.85, zorder=1)
    sub.boundary.plot(ax=ax, color='0.35', lw=0.5, zorder=2)

    texts = []
    for (_, r), p in zip(sub.iterrows(), sub.representative_point()):
        ax.plot(p.x, p.y, marker='o', ms=1.6, color='0.2', zorder=5)     # anchor dot at region
        texts.append(ax.text(p.x, p.y, label_fn(r.region), fontsize=LABEL_FS, ha='center',
                             va='center', color=LABEL_COL, zorder=6, path_effects=HALO))

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.radians((y0 + y1) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('0.5')
    ax.set_title(title, fontsize=titlesize, pad=4)

    if HAVE_ADJUST and len(texts) > 1:
        try:
            adjust_text(texts, ax=ax, expand=(1.3, 1.6), force_text=(0.4, 0.7),
                        force_static=(0.2, 0.4),
                        arrowprops=dict(arrowstyle='-', color='0.45', lw=0.4))
        except Exception as e:
            print('  adjustText skipped for %s (%s)' % (title, e))


def main():
    g = apply_china_worldview(load_regions())
    world = load_basemap()
    os.makedirs(OUT, exist_ok=True)

    # ---- one big standalone file per group ----
    for key, title, sel, lab, figsize, pad in GROUPS:
        fig, ax = plt.subplots(figsize=figsize)
        draw_panel(ax, world, g, sel, title, lab, pad=pad, titlesize=9)
        out = os.path.join(OUT, 'fig1a_subregion_key_' + key)
        fig.savefig(out + '.png', dpi=200, bbox_inches='tight', facecolor='white')
        fig.savefig(out + '.svg', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('saved', os.path.basename(out))

    # ---- combined 4-panel overview ----
    fig = plt.figure(figsize=(16, 13))
    ax_na = fig.add_axes([0.02, 0.52, 0.96, 0.44])
    ax_cn = fig.add_axes([0.02, 0.03, 0.31, 0.44])
    ax_in = fig.add_axes([0.35, 0.03, 0.30, 0.44])
    ax_br = fig.add_axes([0.67, 0.03, 0.31, 0.44])
    axd = {'north_america': ax_na, 'china': ax_cn, 'india': ax_in, 'brazil': ax_br}
    for key, title, sel, lab, _, pad in GROUPS:
        draw_panel(axd[key], world, g, sel, title, lab, pad=pad, titlesize=10)
    out = os.path.join(OUT, 'fig1a_subregion_key_overview')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(out + '.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved overview  | adjustText=%s' % HAVE_ADJUST)


if __name__ == '__main__':
    main()
