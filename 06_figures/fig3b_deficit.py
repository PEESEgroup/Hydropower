"""Fig 3b - firm coverage that cross-region interconnection adds per region (across scenarios).

One row per region with an isolated deficit: dot = median firm coverage of DC demand that
interconnection adds (interconnected minus isolated, percentage points of demand) across the
5 GCM x 3 SSP grid; whisker = min-max across those 15 scenarios; marker area ~ DC demand. Grouped
by interconnection bloc, names on the left, bloc on the right. China & India deficits are fully
closed; North America's giants only partly. Scenario A. Arial 7 pt; transparent; no figure title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

import revub_style as S
from revub_names import region_label
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
BLOC_NAME = {'china': 'China', 'india': 'India', 'southeast_asia': 'SE Asia',
             'europe': 'Europe', 'north_america': 'N. America'}
ORDER = ['china', 'india', 'southeast_asia', 'europe', 'north_america']
THRESH = 1.0   # percentage points; show regions where interconnection adds coverage


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = pd.read_csv(os.path.join(DATA, 'cov_scenarios.csv'))
    d['bloc'] = d.region.map(r2b)
    d['gain'] = d.cov_interconnected_pct - d.cov_isolated_pct
    g = d.groupby('region').agg(bloc=('bloc', 'first'), med=('gain', 'median'),
                                lo=('gain', 'min'), hi=('gain', 'max'), dem=('demand_mw', 'median'))
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

    ax.set_ylim(ymax + 0.2, -0.7)                 # extra bottom room so the axis never overlaps a marker
    ax.set_xlim(-3, 105)                           # pad both ends so edge markers aren't clipped
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Firm coverage added by interconnection\n(percentage points of DC demand)', fontsize=7)
    ax.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3b_deficit'))
    save_region_legend(dmax)
    print('saved fig3b_deficit | %d regions, dot+range over 15 scenarios' % len(rows))


def save_region_legend(dmax):
    """Standalone legend for Fig3b (placed by hand): interconnection bloc colours + marker
    semantics. Dot = median over the 15 GCM x SSP scenarios; line = min-max range; marker size
    follows the figure's own scale s = 5 + 26*sqrt(demand / dmax), keyed at 1 / 10 / 40 GW."""
    fig = plt.figure(figsize=(120 * S.MM, 22 * S.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    blocs = [('china', 'China'), ('india', 'India'), ('southeast_asia', 'Southeast Asia'),
             ('south_america', 'South America'), ('europe', 'Europe'), ('north_america', 'North America')]
    for i, (b, nm) in enumerate(blocs):
        x = 0.02 + 0.165 * i
        ax.scatter([x], [0.78], s=34, color=S.BLOC_COLOR[b], ec='white', lw=0.5, clip_on=False)
        ax.text(x + 0.016, 0.78, nm, va='center', ha='left', fontsize=6.5)
    # median dot + 15-scenario range line
    ax.scatter([0.03], [0.34], s=16, color='0.45', ec='white', lw=0.5, clip_on=False)
    ax.text(0.05, 0.34, 'median', va='center', fontsize=6.5)
    ax.plot([0.19, 0.26], [0.34, 0.34], color='0.45', lw=1.0, clip_on=False)
    ax.text(0.275, 0.34, '15-scenario range', va='center', fontsize=6.5)
    # quantitative size key - marker AREA set by DC demand, same formula as the figure
    ax.text(0.54, 0.34, 'DC demand (GW)', va='center', ha='left', fontsize=6.5)
    for x, gw in zip([0.78, 0.87, 0.96], [1, 10, 40]):
        ax.scatter([x], [0.40], s=5 + 26 * np.sqrt(gw * 1000.0 / dmax),
                   color='0.45', ec='white', lw=0.5, clip_on=False)
        ax.text(x, 0.06, '%d' % gw, va='center', ha='center', fontsize=6)
    S.save(fig, os.path.join(OUT, 'fig3_region_legend'))


if __name__ == '__main__':
    main()
