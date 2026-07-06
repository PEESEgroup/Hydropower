"""Fig 1b (scatter version) - one point per region: DC demand vs hydropower firm capacity.

x = DC demand capacity (MW, 2040 peak); y = hydropower firm capacity = flat-load ELCC
(reference combo set by $REVUB_FIG_DATA; headline = gfdl-esm4/ssp370, existing hydro).
Combines the two former Voronoi panels into one. Log-log; 1:1 line
separates hydro surplus (above) from deficit (below). Coloured by interconnection bloc
(S.BLOC_COLOR - same palette as Fig3c). All text Arial 7 pt; PNG 300 ppi; no title.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import revub_style as S
from fig1b_voronoi import _region_table, INCOME_COL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
YFLOOR = 1.0   # regions with ~0 local hydro floored here so they show on the log axis


def main():
    t = _region_table()
    t = t[t['demand_mw'] > 0].copy()
    t['x'] = t['demand_mw'].clip(lower=YFLOOR)
    t['y'] = t['flat_elcc_mw'].fillna(0).clip(lower=YFLOOR)
    hi = max(t['x'].max(), t['y'].max()) * 1.4

    large = bool(os.environ.get('REVUB_SCATTER_LARGE'))   # huge all-labelled reference version
    fig, ax = S.fig_mm(290, 250) if large else S.fig_mm(110, 104)
    fig.subplots_adjust(left=0.09 if large else 0.13, right=0.93 if large else 0.86,
                        top=0.98, bottom=0.12 if large else 0.22)
    ax.plot([YFLOOR, hi], [YFLOOR, hi], color='0.45', ls='--', lw=0.6, zorder=1)
    ax.text(hi * 0.9, hi * 0.9, '1:1', color='0.45', fontsize=6, ha='right', va='bottom', rotation=45)
    for b in S.BLOC_ORDER:
        m = t['bloc'] == b
        if m.any():
            ax.scatter(t['x'][m], t['y'][m], s=20, c=S.BLOC_COLOR[b], ec='white', lw=0.3,
                       alpha=0.9, zorder=3, label=b.replace('_', ' ').title())

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(YFLOOR, hi)
    ax.set_ylim(0.7, hi)
    ax.set_xlabel('DC demand capacity (MW)')
    ax.set_ylabel('Hydropower firm capacity\n(flat-load ELCC, MW)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # only the LARGE reference version is labelled (all points); the small version is
    # left unlabelled - the user places region names manually in Adobe.
    if large:
        for _, r in t.iterrows():
            ax.annotate(r['country'], (r['x'], r['y']), fontsize=5,
                        color='0.2', xytext=(2.5, 2.5), textcoords='offset points',
                        annotation_clip=False)

    ax.legend(title='Region', frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=3, fontsize=7, title_fontsize=7, handletextpad=0.3, columnspacing=1.2)

    os.makedirs(OUT, exist_ok=True)
    stem = 'fig1b_scatter_large' if large else 'fig1b_scatter'
    S.save(fig, os.path.join(OUT, stem))
    print('saved', os.path.join(OUT, stem + '.svg'), '(+.png)  n=%d' % len(t))


if __name__ == '__main__':
    main()
