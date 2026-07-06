"""Fig 4 strip plots - DC (no PS) vs TYPICAL GRID, one small-multiple per indicator.

Companion to the DC-vs-conventional maps: here each indicator shows ONLY the DC vs typical-grid
comparison (DC_A - BAL_C). Single lane: a pooled density ridge (category colour - ecology light red /
sediment light blue / supply green) + one row of per-DC-unit points per region (coloured by bloc) +
per-region median tick. The right-margin label carries TWO numbers per region: (1) the ABSOLUTE median
change in native units (median over the region's DC-redispatched dams of DC_A - BAL_C), and below it
(2) the RELATIVE change = median(DC_A - BAL_C) / |median(DC_A - CONV)| x 100% (|denom| so its sign
matches the absolute) - the incremental DC
effect, i.e. what fraction of the dam's total change-from-conventional is specifically due to the DC
differing from a typical flexible grid (small => DC ~ typical grid; negative => DC gentler than the
typical grid). Only the four metrics that also have world maps are drawn (reversals, lethal drawdown,
sediment STCI, irrigation gap). Centered caption = metric + beneficial/harmful direction. 1/4 A4 wide
x 1/4 A4 tall; Arial 7 pt; no title. Run: python fig4_strip_typ.py [metric]
"""
import os, sys, textwrap
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
NR = len(BLOCS); FS = 7
W, Ht = 52.5, 74.25
rng = np.random.default_rng(0)
MAP_METRICS = ['reversals_per_year_hourly', 'VRR_exceed_h_13cmh', 'STCI', 'gap_months_per_year']  # only the metrics with world maps

CFG = {
    'reversals_per_year_hourly': dict(src='fig4_eco.csv', color=ECO, stem='reversals',
        label='Change in flow reversals (per year)', left=('fewer', 'beneficial'), right=('more', 'harmful')),
    'RR_down_p95_pctmean': dict(src='fig4_eco.csv', color=ECO, stem='downramp',
        label='Change in down-ramp severity (% of mean flow)', left=('gentler', 'beneficial'), right=('more hydropeaking', 'harmful')),
    'VRR_exceed_h_13cmh': dict(src='fig4_eco.csv', color=ECO, stem='drawdown',
        label='Change in lethal drawdown (hours/yr > 13 cm/h)', left=('fewer', 'beneficial'), right=('more', 'harmful')),
    'STCI': dict(src='fig4_sed.csv', color=SED, stem='sediment',
        label='Change in sediment-transport index', left=('less transport', None), right=('more transport', None)),
    'gap_months_per_year': dict(src='fig4_sup.csv', color=SUP, stem='irrigationgap',
        label='Change in irrigation gap (months/yr)', left=('smaller gap', 'beneficial'), right=('larger gap', 'harmful')),
    'SI': dict(src='fig4_sup.csv', color=SUP, stem='sustainability',
        label='Change in supply sustainability index', left=('lower', 'harmful'), right=('higher', 'beneficial')),
}


def absfmt(x):                                      # absolute-median label (native units)
    a = abs(x)
    if a >= 100:
        return '%+.0f' % x
    if a >= 1:
        return '%+.1f' % x
    if a >= 0.01:
        return '%+.2g' % x
    return '0' if a == 0 else '%+.0e' % x


def ratiofmt(x):                                    # incremental ratio (DC vs typ)/(DC vs CONV), percent
    if x is None or not np.isfinite(x):
        return ''
    if abs(x) < 0.05:
        return '(0%)'
    return ('(%+.0f%%)' % x) if abs(x) >= 10 else ('(%+.1f%%)' % x)


def lane(ax, d, color):
    # population = DC-redispatched dams (those the DC actually changes vs conventional)
    d = d[np.isfinite(d['typ']) & np.isfinite(d['conv'])]
    aff = d[np.abs(d['conv']) > 1e-9]; v = aff['typ'].to_numpy()           # plot the increment (DC vs typical)
    if len(v) > 5:
        kde = gaussian_kde(v); xs = np.linspace(v.min(), v.max(), 200)
        ax.fill_between(xs, NR - 0.05, NR - 0.05 + 0.65 * kde(xs) / kde(xs).max(), color=color, alpha=0.9, lw=0, zorder=2)
    for i, (b, nm) in enumerate(BLOCS):
        y = NR - 1 - i
        sub = aff[aff.bloc == b]; bv = sub['typ'].to_numpy()
        if len(bv):
            med = np.median(bv)
            ax.scatter(bv, y + rng.uniform(-0.28, 0.28, len(bv)), s=2.0, color=S.BLOC_COLOR[b], alpha=0.5, lw=0, zorder=3)
            ax.plot([med] * 2, [y - 0.34, y + 0.34], color=S.BLOC_COLOR[b], lw=1.3, zorder=4)
            mc = np.median(sub['conv'].to_numpy())                          # median total change (DC vs conventional)
            ratio = (med / abs(mc) * 100.0) if abs(mc) > 1e-9 else None      # incremental DC fraction; |denom| so sign matches the absolute
            ax.text(1.04, y + 0.20, absfmt(med), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS, color=S.BLOC_COLOR[b])
            ax.text(1.04, y - 0.24, ratiofmt(ratio), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS - 1.5, color=S.BLOC_COLOR[b])
        ax.text(-0.03, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.3')
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
    lab = '\n'.join(textwrap.wrap(c['label'], 34))
    return lab + '\n' + side(lw, lk, True) + '\n' + side(rw, rk, False)


def draw(metric):
    c = CFG[metric]; r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    df = pd.read_csv(os.path.join(DATA, c['src']))
    d = df[df.metric == metric].copy() if 'metric' in df.columns else df.copy()
    d['bloc'] = d.region.map(r2b)
    d['typ'] = d.DC_A - d.BAL_C            # increment: DC vs typical grid
    d['conv'] = d.DC_A - d.CONV            # total: DC vs conventional
    fig, ax0 = S.fig_mm(W, Ht); ax0.set_axis_off()
    ax = fig.add_axes([0.40, 0.255, 0.40, 0.66])
    lane(ax, d, c['color'])
    ax.set_xlabel(caption(c), fontsize=FS, color='0.2', labelpad=4, linespacing=1.4)
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig4_typ_' + c['stem']))
    print('saved fig4_typ_' + c['stem']); plt.close(fig)


if __name__ == '__main__':
    for m in ([sys.argv[1]] if len(sys.argv) > 1 else MAP_METRICS):
        draw(m)
