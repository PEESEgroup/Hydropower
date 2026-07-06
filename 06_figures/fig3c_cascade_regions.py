"""Fig 3c (cascade panel) - cascade-coordination firm gain per region (Fig3b idiom).

One row per region where within-basin cascade coordination adds firm capacity for the data centre
(program-D supply_C - supply_A, vs the independent baseline, as % of DC demand, median>10%); dot = median across
the 5 GCM x 3 SSP grid, whisker = min-max across those 15 scenarios, marker area ~ DC demand.
Grouped by bloc, names on the left, bloc on the right. Unlike pumped storage, cascade is a large,
broad lever and China is a major beneficiary (China Southwest +62% of demand). Arial 7 pt; no title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import revub_style as S
from revub_names import region_label
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
BLOC_NAME = {'china': 'China', 'india': 'India', 'southeast_asia': 'SE Asia',
             'south_america': 'S. America', 'europe': 'Europe', 'north_america': 'N. America'}
ORDER = ['china', 'india', 'southeast_asia', 'south_america', 'europe', 'north_america']
THRESH = 10.0   # % of demand; show regions where cascade adds a sizeable share


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = pd.read_csv(os.path.join(DATA, 'dc_supply_D.csv'))
    d['bloc'] = d.region.map(r2b)
    d = d[d.dc_mean_mw > 0].copy()
    d['ca'] = 100 * (d.supply_C_mw - d.supply_A_mw) / d.dc_mean_mw   # cascade vs independent baseline (C - A)
    g = d.groupby('region').agg(bloc=('bloc', 'first'), med=('ca', 'median'),
                                lo=('ca', 'min'), hi=('ca', 'max'), dem=('dc_mean_mw', 'median'))
    g = g[g.med > THRESH]
    dmax = g.dem.max()

    rows, bspan, y = [], {}, 0.0
    GAP = 0.6
    for b in ORDER:
        sub = g[g.bloc == b].sort_values('med', ascending=False)
        if not len(sub):
            continue
        y0 = y
        for reg, r in sub.iterrows():
            rows.append((y, reg, r)); y += 1
        bspan[b] = (y0, y - 1); y += GAP
    ymax = y - GAP

    fig, ax = S.fig_mm(70, 99)                         # 1/3 A4 wide x 1/3 A4 tall
    fig.subplots_adjust(left=0.40, right=0.80, top=0.985, bottom=0.105)
    yt = ax.get_yaxis_transform()
    for yy, reg, r in rows:
        col = S.BLOC_COLOR.get(r.bloc, '0.5')
        ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=0.8, alpha=0.5, zorder=2)
        ax.scatter([r.med], [yy], s=5 + 26 * np.sqrt(r.dem / dmax), color=col,
                   ec='white', lw=0.4, zorder=4)
        ax.text(-0.02, yy, region_label(reg), transform=yt, ha='right', va='center', fontsize=5.5)

    for b in ORDER:
        if b not in bspan:
            continue
        y0, y1 = bspan[b]
        ax.plot([1.02, 1.02], [y0 - 0.3, y1 + 0.3], transform=yt, color='0.6', lw=0.8,
                clip_on=False, solid_capstyle='round')
        ax.text(1.04, (y0 + y1) / 2, BLOC_NAME[b], transform=yt, ha='left', va='center',
                fontsize=5.5, color='0.1', fontweight='bold', clip_on=False)

    ax.set_ylim(ymax + 0.2, -0.7)
    ax.set_xscale('log')
    ax.set_xlim(5, 300)
    ax.set_xticks([10, 100]); ax.set_xticklabels(['10', '100'])
    ax.set_xlabel('Cascade gain\n(% of DC demand, log)', fontsize=7)
    ax.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_cascade_regions'))
    print('saved fig3c_cascade_regions | %d regions (incl. China)' % len(rows))


if __name__ == '__main__':
    main()
