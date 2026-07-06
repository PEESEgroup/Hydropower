"""Fig 3c-ii - firm-capacity gain from each lever, across every region and scenario (beeswarm).

Each dot is one region under one scenario (pumped storage & cascade: 5 GCM x 3 SSP = 15 scenarios
x ~40 regions; cross-region interconnection: per region, reference run). x = firm capacity added,
as % of that region's DC demand (log scale); dots coloured by bloc. The global means are small,
but region by region the picture is rich: pumped storage firms up to ~75% of demand in some
regions and is positive across scenarios; interconnection ranges all the way to 100%. Medians and
the share of region-scenarios with a positive gain are marked per lever. Arial 7 pt; no title.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
BLOC_ORDER = ['china', 'india', 'southeast_asia', 'south_america', 'europe', 'north_america']
BLOC_NAME = {'china': 'China', 'india': 'India', 'southeast_asia': 'Southeast Asia',
             'south_america': 'South America', 'europe': 'Europe', 'north_america': 'North America'}
FLOOR = 0.1   # % ; smallest gain shown on the log axis


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    # PS & cascade per region (mean over the 15 GCM x SSP) - canonical program-D output
    d = pd.read_csv(os.path.join(DATA, 'dc_supply_D.csv'))
    d = d[d.dc_mean_mw > 0].copy()
    d['PS'] = 100 * (d.supply_B_mw - d.supply_A_mw) / d.dc_mean_mw       # PS vs independent (B - A)
    d['Cascade'] = 100 * (d.supply_C_mw - d.supply_A_mw) / d.dc_mean_mw  # cascade vs independent (C - A)
    ps = d.groupby('region').agg(val=('PS', 'mean'), dem=('dc_mean_mw', 'mean'))
    ca = d.groupby('region').agg(val=('Cascade', 'mean'), dem=('dc_mean_mw', 'mean'))
    # interconnection per region (mean over 15 scenarios)
    cov = pd.read_csv(os.path.join(DATA, 'cov_scenarios.csv'))
    cov['gain'] = cov.cov_interconnected_pct - cov.cov_isolated_pct
    it = cov.groupby('region').agg(val=('gain', 'mean'), dem=('demand_mw', 'mean'))
    for t in (ps, ca, it):
        t['bloc'] = [r2b.get(r) for r in t.index]

    def gwt(t):   # demand-weighted global gain (%)
        return float((t.val * t.dem).sum() / t.dem.sum())

    levers = [('Pumped storage', ps), ('Cascade', ca), ('Cross-region\ninterconnection', it)]
    rng = np.random.default_rng(0)
    fig, ax = S.fig_mm(70, 33)                         # 1/3 A4 wide x 1/9 A4 tall
    fig.subplots_adjust(left=0.30, right=0.965, top=0.97, bottom=0.26)
    ypos = [2, 1, 0]
    def fmt(x):
        return ('%.0f' % x) if x >= 1 else ('%.1f' % x)
    for y, (name, t) in zip(ypos, levers):
        v = t['val'].to_numpy(); b = t['bloc'].to_numpy()
        pos = v > FLOOR
        xx = np.clip(v[pos], FLOOR, 100)
        yy = y + rng.uniform(-0.28, 0.28, len(xx))
        cols = [S.BLOC_COLOR.get(bb, '0.5') for bb in b[pos]]
        ax.scatter(xx, yy, s=10, c=cols, alpha=0.45, lw=0, zorder=2)       # region dots
        # per-bloc demand-weighted averages: opaque diamonds coloured by bloc (no number labels -
        # only the global average is annotated)
        for bloc in BLOC_ORDER:
            m = t['bloc'] == bloc
            if m.sum() == 0:
                continue
            bav = float((t['val'][m] * t['dem'][m]).sum() / t['dem'][m].sum())
            if bav >= FLOOR:                      # skip negligible bloc means (axis-edge clutter)
                ax.scatter([bav], [y], marker='D', s=30, color=S.BLOC_COLOR[bloc],
                           ec='white', lw=0.5, zorder=5)
        gw = max(gwt(t), FLOOR)                                  # global (demand-weighted) gain
        ax.plot([gw, gw], [y - 0.5, y + 0.5], color='0.12', lw=1.6, zorder=6)
        ax.text(gw, y - 0.55, fmt(gw) + '%', ha='center', va='top', fontsize=6,
                color='0.12', fontweight='bold', zorder=7)
        ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha='right',
                va='center', fontsize=6.5)

    ax.set_xscale('log')
    ax.set_xlim(FLOOR, 100)
    ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.set_ylim(-0.95, 2.95)
    ax.set_yticks([])
    ax.set_xlabel('Firm added (% of DC demand, log)', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=6.5, width=0.4, length=2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_benefit'))
    save_legend()
    print('saved fig3c_benefit (beeswarm) + legend')


def save_legend():
    """Separate legend: diamond = bloc demand-weighted mean (by colour); black line = global."""
    fig = plt.figure(figsize=(70 * S.MM, 14 * S.MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    h = [mlines.Line2D([], [], ls='', marker='D', mfc=S.BLOC_COLOR[b], mec='white', mew=0.5,
                       ms=4.5, label=BLOC_NAME[b]) for b in BLOC_ORDER]
    h.append(mlines.Line2D([], [], color='0.12', lw=1.6, label='Global'))
    ax.legend(handles=h, frameon=False, fontsize=6, loc='center', ncol=4,
              handletextpad=0.3, columnspacing=1.0, labelspacing=0.5)
    S.save(fig, os.path.join(OUT, 'fig3c_benefit_legend'))


if __name__ == '__main__':
    main()
