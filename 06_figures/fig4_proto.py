"""Fig 4 prototype - one ecology panel (down-ramp severity), DC vs two baselines.

Per baseline lane (top = vs conventional, bottom = vs typical grid): a pooled density ridge on top,
and BELOW it the per-DC-dam points split into ONE ROW PER REGION (coloured by bloc) with a median
tick per row - so points separate and regional differences read row-by-row. Centered at 0; right =
DC adds hydropeaking, left = gentler. Affected (non-zero) dams only. Arial 7 pt; no title.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); OUT = os.path.join(HERE, 'out')
METRIC = 'RR_down_p95_pctmean'
XLABEL = 'Change in down-ramp severity (% of mean flow)'
BLOCS = [('north_america', 'N. America'), ('europe', 'Europe'), ('south_america', 'S. America'),
         ('southeast_asia', 'SE Asia'), ('india', 'India'), ('china', 'China')]
RH = 0.115          # per-region row height
rng = np.random.default_rng(0)


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    d = d[d.metric == METRIC].copy()
    d['bloc'] = d.region.map(r2b)
    d['e_conv'] = d.DC_A - d.CONV
    d['e_typ'] = d.DC_A - d.BAL_C

    fig, ax = S.fig_mm(95, 88)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.93, bottom=0.16)
    ax.axvline(0, ls='--', color='0.55', lw=0.6, zorder=1)

    # two lane bases; each lane = density ridge (above base) + 6 region rows (below base)
    for base, col, lab in [(1.05, 'e_conv', 'vs conventional'), (0.0, 'e_typ', 'vs typical grid')]:
        v = d.loc[np.abs(d[col]) > 1e-9, col].to_numpy()
        if len(v) > 5:                                          # pooled density ridge
            kde = gaussian_kde(v); xs = np.linspace(v.min(), v.max(), 200)
            dens = 0.30 * kde(xs) / kde(xs).max()
            ax.fill_between(xs, base, base + dens, color='0.86', lw=0, zorder=2)
            ax.plot(xs, base + dens, color='0.62', lw=0.4, zorder=2)
        ax.text(-0.02, base + 0.14, lab, transform=ax.get_yaxis_transform(), ha='right',
                va='center', fontsize=7, fontweight='bold')
        for i, (b, nm) in enumerate(BLOCS):                    # one row of points per region
            y = base - 0.07 - i * RH
            bv = d.loc[(d.bloc == b) & (np.abs(d[col]) > 1e-9), col].to_numpy()
            if not len(bv):
                continue
            ax.scatter(bv, y + rng.uniform(-0.035, 0.035, len(bv)), s=2.6,
                       color=S.BLOC_COLOR[b], alpha=0.5, lw=0, zorder=3)
            ax.plot([np.median(bv)] * 2, [y - 0.045, y + 0.045], color=S.BLOC_COLOR[b],
                    lw=1.4, zorder=4)
            ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=5.3, color='0.35')

    ax.set_ylim(-0.85, 1.50); ax.set_yticks([]); ax.set_xlim(-12, 28)
    ax.set_xlabel(XLABEL + '\n' + r'$\leftarrow$ gentler        more hydropeaking $\rightarrow$', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=6.5, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig4_proto'))
    print('saved fig4_proto')


if __name__ == '__main__':
    main()
