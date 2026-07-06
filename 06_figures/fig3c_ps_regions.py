"""Fig 3c (PS panel) - pumped-storage firm gain per region (Fig3b idiom), China excluded.

One row per region where pumped storage adds firm capacity for the data centre (program-D
supply_B - supply_A, as % of the region's DC demand); dot = median across the 5 GCM x 3 SSP grid,
whisker = min-max across those 15 scenarios, marker area ~ DC demand. Grouped by bloc, names on the
left, bloc on the right (as in Fig3b). China is omitted: it is hydro-rich, so PS absorbs surplus
rather than firming the DC (negligible ELCC gain). Arial 7 pt; transparent; no figure title.
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
BLOC_NAME = {'india': 'India', 'southeast_asia': 'Southeast Asia', 'europe': 'Europe',
             'north_america': 'North America'}
ORDER = ['europe', 'north_america', 'southeast_asia', 'india']   # no China (PS negligible there)
THRESH = 0.05   # % of demand; show regions where PS does something


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = pd.read_csv(os.path.join(DATA, 'dc_supply_D.csv'))
    d['bloc'] = d.region.map(r2b)
    d = d[(d.dc_mean_mw > 0) & (d.bloc != 'china')].copy()
    d['ps'] = 100 * (d.supply_B_mw - d.supply_A_mw) / d.dc_mean_mw
    g = d.groupby('region').agg(bloc=('bloc', 'first'), med=('ps', 'median'),
                                lo=('ps', 'min'), hi=('ps', 'max'), dem=('dc_mean_mw', 'median'))
    g = g[g.med > THRESH]
    dmax = g.dem.max()

    # layout: region rows grouped by bloc, gaps between blocs
    rows, bspan, y = [], {}, 0.0
    GAP = 1.3
    for b in ORDER:
        sub = g[g.bloc == b].sort_values('med', ascending=False)
        if not len(sub):
            continue
        y0 = y
        for reg, r in sub.iterrows():
            rows.append((y, reg, r)); y += 1
        bspan[b] = (y0, y - 1); y += GAP
    ymax = y - GAP

    fig, ax = S.fig_mm(70, 148.5)                     # 1/3 A4 wide x 1/2 A4 tall
    fig.subplots_adjust(left=0.40, right=0.84, top=0.975, bottom=0.085)
    yt = ax.get_yaxis_transform()
    for yy, reg, r in rows:
        col = S.BLOC_COLOR.get(r.bloc, '0.5')
        ax.plot([max(r.lo, 0.035), max(r.hi, 0.035)], [yy, yy], color=col, lw=1.0, alpha=0.5, zorder=2)
        ax.scatter([max(r.med, 0.035)], [yy], s=9 + 48 * np.sqrt(r.dem / dmax), color=col,
                   ec='white', lw=0.5, zorder=4)
        ax.text(-0.02, yy, region_label(reg), transform=yt, ha='right', va='center', fontsize=6.5)

    for b in ORDER:
        if b not in bspan:
            continue
        y0, y1 = bspan[b]
        ax.plot([1.02, 1.02], [y0 - 0.3, y1 + 0.3], transform=yt, color='0.6', lw=0.8,
                clip_on=False, solid_capstyle='round')
        ax.text(1.07, (y0 + y1) / 2, BLOC_NAME[b], transform=yt, ha='center', va='center',
                rotation=90, fontsize=6.5, color='0.1', fontweight='bold')

    ax.set_ylim(ymax + 0.2, -0.7)        # extra bottom room so the axis never overlaps a marker
    ax.set_xscale('log')
    ax.set_xlim(0.035, 70)               # pad ends so edge markers/whiskers aren't clipped
    ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(['0.1', '1', '10'])
    ax.set_xlabel('Pumped-storage gain\n(% of DC demand, log)', fontsize=7)
    ax.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_ps_regions'))
    print('saved fig3c_ps_regions | %d regions (China excluded)' % len(rows))


if __name__ == '__main__':
    main()
