"""Fig 2b - DC-side cooling-load sensitivity to warming (by bloc, with region spread).

Same layout idiom as Fig2c: per trading bloc, a DC-demand-weighted mean cooling-load %change
bar (ssp126->ssp585, 5-GCM median; RdBu_r, red = more cooling = adverse, matching Fig2a), a
peak (99th-pct) diamond showing amplification, and grey dots for each region's cooling change
(the spread). HONESTY: total DC load barely moves (median +0.4%) because IT load dominates.
No outer frame, no title; Arial 7 pt; transparent background.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import revub_style as S
from revub_geo import load_regions
from fig1a_coverage_map import load_coverage
from fig1c_seasonal import _disp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
M = 5.0   # colour half-range (%)
NAMES = {'china': 'China', 'europe': 'Europe', 'india': 'India',
         'north_america': 'North\nAmerica', 'south_america': 'South\nAmerica',
         'southeast_asia': 'Southeast\nAsia'}
ORDER = ['south_america', 'north_america', 'india', 'china', 'southeast_asia', 'europe']


def main():
    df = pd.read_csv(os.path.join(DATA, 'fig2b_cooling.csv'))
    cov = load_coverage()[['region', 'demand_mw']]
    bloc = load_regions()[['region', 'bloc']]
    m = df.merge(cov, on='region').merge(bloc, on='region')   # inner join -> only Fig1a regions
    rows = []
    for b, sub in m.groupby('bloc'):
        w = sub['demand_mw'].clip(lower=1e-6)
        rows.append([b, np.average(sub['cool_mean_pct'], weights=w),
                     np.average(sub['cool_p99_pct'], weights=w)])
    bdf = (pd.DataFrame(rows, columns=['bloc', 'cool_mean', 'cool_p99'])
           .set_index('bloc').reindex(ORDER).reset_index())   # same order as Fig2c
    n = len(bdf)

    C_DC = '#e6843d'   # match Fig2c "DC demand" orange - cooling IS the DC-side signal
    rng = np.random.default_rng(0)

    fig, ax = S.fig_mm(105, 99)
    fig.subplots_adjust(left=0.21, right=0.96, top=0.93, bottom=0.155)
    for i, row in enumerate(bdf.itertuples()):
        sub = m[m['bloc'] == row.bloc]
        # MEAN band (upper): region means (grey dots) with the bloc mean as a lollipop sitting
        # AMONG them (thin stem from 0 = magnitude, orange diamond = mean of the dots)
        ax.scatter(sub['cool_mean_pct'], i - 0.16 + rng.uniform(-0.09, 0.09, len(sub)),
                   s=11, c='0.55', alpha=0.6, lw=0, zorder=3)
        ax.plot([0, row.cool_mean], [i - 0.16, i - 0.16], color=C_DC, lw=1.4,
                zorder=2, solid_capstyle='round')
        ax.plot(row.cool_mean, i - 0.16, marker='D', ms=5, mfc=C_DC, mec='white',
                mew=0.5, zorder=5)
        # PEAK band (lower): region 99th-pct (open dots) + bloc-mean peak (black diamond)
        ax.scatter(sub['cool_p99_pct'], i + 0.20 + rng.uniform(-0.09, 0.09, len(sub)),
                   s=12, facecolor='none', edgecolor='0.5', lw=0.5, zorder=3)
        ax.plot(row.cool_p99, i + 0.20, marker='D', ms=5, mfc='0.12', mec='white',
                mew=0.5, zorder=5)
    ax.axvline(0, color='0.1', lw=0.5, zorder=1)
    ax.set_xlim(-2, 9)
    off = m[m['cool_p99_pct'] > 9].sort_values('cool_p99_pct', ascending=False)
    if len(off):
        names = ', '.join('%s %+.0f%%' % (_disp(rr.region), rr.cool_p99_pct)
                          for rr in off.itertuples())
        ax.text(0.97, 0.52, 'Peak off scale (>+9%):\n' + names.replace(', ', ',\n'),
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color='0.45')
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAMES.get(b, b) for b in bdf['bloc']], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel('Cooling-load change, SSP1-2.6 → SSP5-8.5\n(%, 5-GCM median)', fontsize=7)
    ax.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [mlines.Line2D([], [], marker='D', ls='-', color=C_DC, mfc=C_DC, mec='white',
                             ms=5, label='Mean cooling'),
               mlines.Line2D([], [], marker='o', ls='', mfc='0.55', mec='none', ms=3.5,
                             label='Region (mean)'),
               mlines.Line2D([], [], marker='D', ls='', mfc='0.12', mec='white', ms=4,
                             label='99th-pct peak'),
               mlines.Line2D([], [], marker='o', ls='', mfc='none', mec='0.5', ms=3.5,
                             label='Region (99th-pct)')]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper right',
              handletextpad=0.4, borderpad=0.3, labelspacing=0.3)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig2b_cooling'))
    print('saved fig2b_cooling.svg(+png) | %d regions, %d blocs' % (len(m), n))


if __name__ == '__main__':
    main()
