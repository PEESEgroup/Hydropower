"""Fig 4 prototype B - per-region dumbbell (down-ramp severity), DC vs two baselines.

Regions are ROWS (so they never overlap). Per bloc: filled diamond = median DC vs conventional,
open diamond = median DC vs typical grid, thin bar = IQR across that bloc's DC-serving dams (affected
only). Centered at 0; right = DC adds hydropeaking, left = DC gentler. Rows sorted by the vs-CONV
effect. Arial 7 pt; no title.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); OUT = os.path.join(HERE, 'out')
METRIC = 'RR_down_p95_pctmean'
XLABEL = 'Change in down-ramp severity (% of mean flow)'
NAMES = {'china': 'China', 'india': 'India', 'southeast_asia': 'SE Asia',
         'south_america': 'S. America', 'europe': 'Europe', 'north_america': 'N. America'}


def stat(v):
    v = v[np.abs(v) > 1e-9]
    if len(v) < 3:
        return None
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    d = d[d.metric == METRIC].copy()
    d['bloc'] = d.region.map(r2b)
    d['e_conv'] = d.DC_A - d.CONV
    d['e_typ'] = d.DC_A - d.BAL_C

    rows = []
    for b in NAMES:
        sc = stat(d.loc[d.bloc == b, 'e_conv'].to_numpy())
        st = stat(d.loc[d.bloc == b, 'e_typ'].to_numpy())
        if sc:
            rows.append((b, sc, st))
    rows.sort(key=lambda r: r[1][0])                       # sort by vs-CONV median

    fig, ax = S.fig_mm(95, 52)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.27)
    ax.axvline(0, ls='--', color='0.55', lw=0.6, zorder=1)

    for y, (b, sc, st) in enumerate(rows):
        col = S.BLOC_COLOR[b]
        # vs conventional (filled), upper sub-row
        ax.plot([sc[1], sc[2]], [y + 0.16, y + 0.16], color=col, lw=2.2, alpha=0.45,
                solid_capstyle='round', zorder=2)
        ax.scatter([sc[0]], [y + 0.16], marker='D', s=24, color=col, ec='white', lw=0.6, zorder=4)
        # vs typical grid (open), lower sub-row
        if st:
            ax.plot([st[1], st[2]], [y - 0.16, y - 0.16], color=col, lw=2.2, alpha=0.30,
                    solid_capstyle='round', zorder=2)
            ax.scatter([st[0]], [y - 0.16], marker='D', s=24, facecolor='white', ec=col, lw=1.1, zorder=4)
        ax.text(-0.02, y, NAMES[b], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7)

    ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_yticks([]); ax.set_xlim(-6, 14)
    ax.set_xlabel(XLABEL + '\n' + r'$\leftarrow$ gentler        more hydropeaking $\rightarrow$', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=6.5, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    h = [mlines.Line2D([], [], ls='', marker='D', mfc='0.35', mec='white', mew=0.5, ms=4.5, label='vs conventional'),
         mlines.Line2D([], [], ls='', marker='D', mfc='white', mec='0.35', mew=1.0, ms=4.5, label='vs typical grid')]
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6,
              ncol=2, handletextpad=0.3, columnspacing=1.4)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig4_proto_b'))
    print('saved fig4_proto_b')


if __name__ == '__main__':
    main()
