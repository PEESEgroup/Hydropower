"""Fig 4 - river footprint of hydro-powered data centres, DC (no PS) vs two baselines.

One block per metric; within a block the baselines sit SIDE BY SIDE (left = vs conventional, right =
vs typical grid), each with its OWN x-axis so the near-zero 'vs typical' column stays legible. Each
column: a pooled density ridge (category colour - ecology light red, sediment light blue, supply
green) + one row of per-DC-unit points per region (coloured by bloc) + per-region median tick + the
per-region MEAN at the right. Beneficial / harmful sides labelled. Metrics where DC ≈ typical are
shown vs conventional only (single wide column). Arial 7 pt; no title.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data'); OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
ECO, SED, SUP = '#f4a7a3', '#a9c8e8', '#a3d39c'
BLOCS = [('north_america', 'N. America'), ('europe', 'Europe'), ('south_america', 'S. America'),
         ('southeast_asia', 'SE Asia'), ('india', 'India'), ('china', 'China')]
NR = len(BLOCS)
rng = np.random.default_rng(0)
PANELS = [
    ('reversals_per_year_hourly', 'fig4_eco.csv', ECO, ['conv'],         'Change in flow reversals (per year)', 'left'),
    ('RR_down_p95_pctmean',       'fig4_eco.csv', ECO, ['conv', 'typ'],  'Change in down-ramp severity (% of mean flow)', 'left'),
    ('VRR_exceed_h_13cmh',        'fig4_eco.csv', ECO, ['conv', 'typ'],  'Change in lethal drawdown (h/yr > 13 cm/h)', 'left'),
    ('STCI',                      'fig4_sed.csv', SED, ['conv'],         'Change in sediment-transport index', None),
    ('gap_months_per_year',       'fig4_sup.csv', SUP, ['conv', 'typ'],  'Change in irrigation gap (months/yr)', 'left'),
    ('SI',                        'fig4_sup.csv', SUP, ['conv', 'typ'],  'Change in supply sustainability index', 'right'),
]


def get(df, metric):
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    d = df[df.metric == metric].copy() if 'metric' in df.columns else df.copy()
    d['bloc'] = d.region.map(r2b)
    d['conv'] = d.DC_A - d.CONV
    d['typ'] = d.DC_A - d.BAL_C
    return d


def draw(ax, d, lane, color, good, show_names, sublabel):
    aff = d[np.abs(d[lane]) > 1e-9]; v = aff[lane].to_numpy()
    if len(v) > 5:
        kde = gaussian_kde(v); xs = np.linspace(v.min(), v.max(), 200)
        dens = 0.85 * kde(xs) / kde(xs).max()
        ax.fill_between(xs, NR - 0.05, NR - 0.05 + dens, color=color, alpha=0.9, lw=0, zorder=2)
    for i, (b, nm) in enumerate(BLOCS):
        y = NR - 1 - i
        bv = aff.loc[aff.bloc == b, lane].to_numpy(); allv = d.loc[d.bloc == b, lane].to_numpy()
        if len(bv):
            ax.scatter(bv, y + rng.uniform(-0.30, 0.30, len(bv)), s=1.8, color=S.BLOC_COLOR[b], alpha=0.5, lw=0, zorder=3)
            ax.plot([np.median(bv)] * 2, [y - 0.38, y + 0.38], color=S.BLOC_COLOR[b], lw=1.2, zorder=4)
        if len(allv):
            ax.text(0.995, y, '%+.2g' % np.nanmean(allv), transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=4.6, color=S.BLOC_COLOR[b], zorder=6)
        if show_names:
            ax.text(-0.04, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=5.6, color='0.3')
    ax.axvline(0, ls='--', color='0.55', lw=0.6, zorder=1)
    lo, hi = (np.percentile(v, [1, 99]) if len(v) else (-1, 1)); pad = 0.12 * (hi - lo + 1e-9)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.95, NR + 0.95); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=5.6, width=0.4, length=2, pad=1)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.text(0.5, NR + 0.5, sublabel, transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=5.6, color='0.3')
    if good is not None:
        ax.text(0.0 if good == 'left' else 1.0, -0.30, 'beneficial', transform=ax.transAxes,
                ha='left' if good == 'left' else 'right', va='top', fontsize=4.6, color='#2e8b3d')
        ax.text(1.0 if good == 'left' else 0.0, -0.30, 'harmful', transform=ax.transAxes,
                ha='right' if good == 'left' else 'left', va='top', fontsize=4.6, color='#c0392b')


def main():
    srcs = {f: pd.read_csv(os.path.join(DATA, f)) for f in {p[1] for p in PANELS}}
    n = len(PANELS)
    fig = plt.figure(figsize=(125 * S.MM, 165 * S.MM))
    top, bot = 0.955, 0.045
    bh = (top - bot) / n
    for k, (metric, src, color, lns, xlab, good) in enumerate(PANELS):
        yb = top - (k + 1) * bh + 0.052          # axes bottom within the block
        h = bh * 0.56
        d = get(srcs[src], metric)
        if len(lns) == 1:
            cols = [('conv', 0.165, 0.62, True, 'vs conventional')]
        else:
            cols = [('conv', 0.165, 0.355, True, 'vs conventional'),
                    ('typ', 0.605, 0.355, False, 'vs typical grid')]
        for lane, x0, w, names, sub in cols:
            ax = fig.add_axes([x0, yb, w, h])
            draw(ax, d, lane, color, good, names, sub)
        fig.text((0.165 + 0.62) / 2 if len(lns) == 1 else 0.5, yb - 0.028, xlab, ha='center', fontsize=6.3)

    h = [mlines.Line2D([], [], ls='', marker='o', mfc=S.BLOC_COLOR[b], mec='none', ms=4, label=nm) for b, nm in BLOCS]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=6,
               ncol=6, handletextpad=0.2, columnspacing=0.9)
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig4_full'))
    print('saved fig4_full')


if __name__ == '__main__':
    main()
