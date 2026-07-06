"""Fig 2c - attribution of the adequacy change to the hydro vs DC side (per region group).

Adequacy ratio = hydro supply potential / total DC demand, so its log-change decomposes
EXACTLY into a hydro term (= flat-load ELCC %change, panel a) plus a DC term (= -total DC
load %change, panel b). For each major region group, DC-demand-weighted hydro and DC bars +
a net marker, with each individual region's net change as a grey dot. Same row order and
colours as Fig2b. No outer frame; Arial 7 pt; transparent background.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import revub_style as S
from fig1c_seasonal import _disp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')

C_HYDRO, C_DC = '#2c7fb8', '#e6843d'   # supply (blue) / demand (orange)
NAMES = {'china': 'China', 'europe': 'Europe', 'india': 'India',
         'north_america': 'North\nAmerica', 'south_america': 'South\nAmerica',
         'southeast_asia': 'Southeast\nAsia'}
ORDER = ['south_america', 'north_america', 'india', 'china', 'southeast_asia', 'europe']


def main():
    r = pd.read_csv(os.path.join(DATA, 'fig2c_attribution.csv')).set_index('bloc').reindex(ORDER)
    reg = pd.read_csv(os.path.join(DATA, 'fig2c_region.csv'))
    rng = np.random.default_rng(0)
    n = len(r)

    fig, ax = S.fig_mm(105, 99)
    fig.subplots_adjust(left=0.21, right=0.96, top=0.93, bottom=0.18)
    for i, (b, row) in enumerate(r.iterrows()):
        # decomposition band (upper): hydro + DC contribution bars
        ax.barh(i - 0.26, row['hydro_eff'], height=0.15, color=C_HYDRO, ec='none', zorder=2)
        ax.barh(i - 0.09, row['dc_eff'], height=0.15, color=C_DC, ec='none', zorder=2)
        # net band (lower): each region's net (dots) on the SAME row as the bloc-mean net (diamond)
        sub = reg[reg['bloc'] == b]
        ax.scatter(sub['net'], i + 0.20 + rng.uniform(-0.08, 0.08, len(sub)), s=10,
                   c='0.5', alpha=0.6, lw=0, zorder=3)
        ax.plot(row['net'], i + 0.20, marker='D', ms=4.5, mfc='0.12', mec='white',
                mew=0.4, zorder=4)
    ax.axvline(0, color='0.1', lw=0.5, zorder=1)
    ax.set_xlim(-10, 10)
    off = reg[reg['net'].abs() > 10].sort_values('net', ascending=False)
    if len(off):
        names = ', '.join('%s %+.0f%%' % (_disp(rr.region), rr.net) for rr in off.itertuples())
        ax.text(0.97, 0.52, 'Off scale (>+10%):\n' + names.replace(', ', ',\n'),
                transform=ax.transAxes, ha='right', va='top', fontsize=7, color='0.45')
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAMES.get(b, b) for b in r.index], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel('Contribution to adequacy change, SSP1-2.6 → SSP5-8.5\n(%, demand-weighted)',
                  fontsize=7)
    ax.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [mpatches.Patch(fc=C_HYDRO, label='Hydro supply'),
               mpatches.Patch(fc=C_DC, label='DC demand'),
               plt.Line2D([], [], marker='D', ls='', mfc='0.12', mec='white', ms=4,
                          label='Net (mean)'),
               plt.Line2D([], [], marker='o', ls='', mfc='0.5', mec='none', ms=3.5,
                          label='Region (net)')]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper right',
              handletextpad=0.4, borderpad=0.3, labelspacing=0.3)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig2c_attribution'))
    print('saved fig2c_attribution.svg(+png)')


if __name__ == '__main__':
    main()
