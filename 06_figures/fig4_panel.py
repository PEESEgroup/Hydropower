"""Fig 4 - ONE figure per river-impact indicator (DC no-PS vs two baselines).

Per indicator: stacked lanes (top = vs conventional, bottom = vs typical grid), each its OWN x-axis.
Each lane = a pooled density ridge (category colour: ecology light red / sediment light blue /
supply green) + one row of per-DC-unit points per region (coloured by bloc) + per-region median tick
(median of the AFFECTED dams) with that median value in the right margin. A single centered two-line caption under the bottom axis gives
the metric and the direction ('<- gentler (beneficial)   more hydropeaking (harmful) ->'). All text
7 pt. Run: python fig4_panel.py [metric]  (default: all).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data'); OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
ECO, SED, SUP = '#f4a7a3', '#a9c8e8', '#a3d39c'
BLOCS = [('north_america', 'North America'), ('europe', 'Europe'), ('south_america', 'South America'),
         ('southeast_asia', 'Southeast Asia'), ('india', 'India'), ('china', 'China')]
NR = len(BLOCS); FS = 7                                 # single font size everywhere
rng = np.random.default_rng(0)

CFG = {
    'reversals_per_year_hourly': dict(src='fig4_eco.csv', color=ECO, lanes=['conv'], stem='reversals',
        label='Change in flow reversals (per year)', left=('fewer', 'beneficial'), right=('more', 'harmful')),
    'RR_down_p95_pctmean': dict(src='fig4_eco.csv', color=ECO, lanes=['conv', 'typ'], stem='downramp',
        label='Change in down-ramp severity (% of mean flow)', left=('gentler', 'beneficial'), right=('more hydropeaking', 'harmful')),
    'VRR_exceed_h_13cmh': dict(src='fig4_eco.csv', color=ECO, lanes=['conv', 'typ'], stem='drawdown',
        label='Change in lethal drawdown (hours/yr > 13 cm/h)', left=('fewer', 'beneficial'), right=('more', 'harmful')),
    'STCI': dict(src='fig4_sed.csv', color=SED, lanes=['conv'], stem='sediment',
        label='Change in sediment-transport index', left=('less transport', None), right=('more transport', None)),
    'gap_months_per_year': dict(src='fig4_sup.csv', color=SUP, lanes=['conv', 'typ'], stem='irrigationgap',
        label='Change in irrigation gap (months/yr)', left=('smaller gap', 'beneficial'), right=('larger gap', 'harmful')),
    'SI': dict(src='fig4_sup.csv', color=SUP, lanes=['conv', 'typ'], stem='sustainability',
        label='Change in supply sustainability index', left=('lower', 'harmful'), right=('higher', 'beneficial')),
}


def fmt(x):
    a = abs(x)
    return ('%+.0f' % x) if a >= 100 else ('%+.1f' % x) if a >= 1 else ('%+.2g' % x)


def lane(ax, d, col, color, sub):
    d = d.assign(**{col: d[col].replace([np.inf, -np.inf], np.nan)})
    d = d[np.isfinite(d[col])]
    aff = d[np.abs(d[col]) > 1e-9]; v = aff[col].to_numpy()
    if len(v) > 5:
        kde = gaussian_kde(v); xs = np.linspace(v.min(), v.max(), 200)
        ax.fill_between(xs, NR - 0.05, NR - 0.05 + 0.65 * kde(xs) / kde(xs).max(), color=color, alpha=0.9, lw=0, zorder=2)
    for i, (b, nm) in enumerate(BLOCS):
        y = NR - 1 - i
        bv = aff.loc[aff.bloc == b, col].to_numpy()
        if len(bv):
            med = np.median(bv)                              # median of affected dams (robust; matches the cloud)
            ax.scatter(bv, y + rng.uniform(-0.28, 0.28, len(bv)), s=2.4, color=S.BLOC_COLOR[b], alpha=0.5, lw=0, zorder=3)
            ax.plot([med] * 2, [y - 0.36, y + 0.36], color=S.BLOC_COLOR[b], lw=1.3, zorder=4)
            ax.text(1.02, y, fmt(med), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS, color=S.BLOC_COLOR[b])
        ax.text(-0.02, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.3')
    ax.axvline(0, ls='--', color='0.55', lw=0.6, zorder=1)
    lo, hi = (np.percentile(v, [1, 99]) if len(v) else (-1, 1)); pad = 0.10 * (hi - lo + 1e-9)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.55, NR + 0.62); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=FS, width=0.4, length=2, pad=1)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)


def caption(c):
    def side(word, kind, left):
        tag = (' (%s)' % kind) if kind else ''
        return ('← ' + word + tag) if left else (word + tag + ' →')
    (lw, lk), (rw, rk) = c['left'], c['right']
    return c['label'] + '\n' + side(lw, lk, True) + '          ' + side(rw, rk, False)


def draw(metric):
    c = CFG[metric]; r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    df = pd.read_csv(os.path.join(DATA, c['src']))
    d = df[df.metric == metric].copy() if 'metric' in df.columns else df.copy()
    d['bloc'] = d.region.map(r2b); d['conv'] = d.DC_A - d.CONV; d['typ'] = d.DC_A - d.BAL_C
    two = len(c['lanes']) == 2
    fig, ax0 = S.fig_mm(105, 68 if two else 46); ax0.set_axis_off()
    if two:
        axes = [fig.add_axes([0.27, 0.62, 0.55, 0.31]), fig.add_axes([0.27, 0.21, 0.55, 0.31])]
        subs = ['conventional', 'typical grid']
    else:
        axes = [fig.add_axes([0.27, 0.40, 0.55, 0.54])]; subs = ['conventional']
    for ax, col, sub in zip(axes, c['lanes'], subs):
        lane(ax, d, col, c['color'], sub)
    axes[-1].set_xlabel(caption(c), fontsize=FS, color='0.2', labelpad=5, linespacing=1.5)
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig4_' + c['stem']))
    print('saved fig4_' + c['stem']); plt.close(fig)


if __name__ == '__main__':
    for m in ([sys.argv[1]] if len(sys.argv) > 1 else list(CFG)):
        draw(m)
