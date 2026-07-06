"""Fig 3c-i - pumped storage by SEASON (when it firms) vs by hour (its diurnal rhythm), per region.

Two heatmaps, each on ONE COMMON colour scale across rows (not per-row), so colour reflects true
magnitude and small-PS regions stay faint. Each row = a bloc (its regions summed). TOP = intraday
RHYTHM: each hour's deviation from that row's daily-mean net power (absolute net is ~constant
pumping, which hides the cycle). BOTTOM = seasonal: absolute net power by month. Net = generation -
pumping; magenta = charging (pumping), green = discharging. Annotated per row: the peak-to-peak net
power swing (MW) within a day (top) and across the year (bottom) - the diurnal swing is a small
fraction of the seasonal, i.e. pumped storage is mainly a seasonal lever.

Source = program-D **scenario B** (station+PS one-to-one, P_PS_gen_B - P_PS_pump_B), reference run
mri-esm2-0/ssp370, canonical data/ps_monthly.csv & ps_diurnal.csv. Scenario B is the PS the Fig3c
beeswarm lever (supply_B - supply_A) actually measures, and - unlike scenario C - it is NOT
contaminated by the cascade re-dispatch (which makes china_sw pump ~2.3 TWh/yr at ~0.6% return).
China is therefore INCLUDED here and reads physically (pumps in the summer flood season, discharges
in late winter/spring). Each bloc's seasonal amplitude (MW) is annotated. Arial 7 pt; no title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import sys

import revub_style as S
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_interconnect_map import TRADING_BLOCS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
NAMES = {'china': 'China', 'india': 'India', 'southeast_asia': 'Southeast Asia',
         'europe': 'Europe', 'north_america': 'North America', 'south_america': 'South America'}
# ordered by seasonal amplitude (descending) in scenario B; China included (B-scenario PS is physical).
# south_america last: it has NO pumped storage (gen=pump=0 in all 8 regions), shown as a 0 row.
ORDER = ['europe', 'china', 'southeast_asia', 'north_america', 'india', 'south_america']
MONTHS = list('JFMAMJJASOND')


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    di = pd.read_csv(os.path.join(DATA, 'ps_diurnal.csv')); di['bloc'] = di.region.map(r2b)
    mo = pd.read_csv(os.path.join(DATA, 'ps_monthly.csv')); mo['bloc'] = mo.region.map(r2b)
    mo['net_mw'] = (mo.gen_mwh_day - mo.pump_mwh_day) / 24.0
    dih = di.groupby(['bloc', 'hour']).net_mw.sum().unstack().reindex(ORDER)        # bloc x 24
    mom = mo.groupby(['bloc', 'month']).net_mw.sum().unstack().reindex(ORDER)       # bloc x 12
    # Both panels on ONE COMMON scale across rows (NOT per-row): colour is magnitude-honest, so
    # small-PS regions stay faint. INTRADAY = diurnal RHYTHM (each hour's deviation from that row's
    # daily-mean net power; absolute net is ~constant pumping, which hides the cycle). SEASONAL =
    # absolute net by month. Per-row peak-to-peak swing (MW) annotated for both (comparable metric).
    dih_dev = dih.sub(dih.mean(axis=1), axis=0)
    swing_i = dih.max(axis=1) - dih.min(axis=1)          # diurnal peak-to-peak swing (MW)
    swing_s = mom.max(axis=1) - mom.min(axis=1)          # seasonal peak-to-peak swing (MW)
    Hd = (dih_dev / float(dih_dev.abs().max().max())).to_numpy()
    Hm = (mom / float(mom.abs().max().max())).to_numpy()

    cmap = plt.get_cmap('PiYG')   # colorbrewer PiYG (pink/green) - distinct from the blue/red maps
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    n = len(ORDER)
    fig = plt.figure(figsize=(70 * S.MM, 66 * S.MM))   # 1/3 A4 wide x 2/9 A4 tall
    axI = fig.add_axes([0.34, 0.70, 0.54, 0.27])    # intraday (top, 24 cols)
    axS = fig.add_axes([0.34, 0.31, 0.54, 0.27])    # seasonal (bottom, 12 cols)

    axI.imshow(Hd, aspect='auto', cmap=cmap, norm=norm, extent=[0, 24, n, 0])
    axI.set_xticks([0.5, 6, 12, 18, 23.5]); axI.set_xticklabels(['0', '6', '12', '18', '24'], fontsize=6.5)
    axI.set_xlabel('Hour of day', fontsize=7)
    axI.set_yticks(np.arange(n) + 0.5)
    axI.set_yticklabels([NAMES[b] for b in ORDER], fontsize=6.5)
    for i, b in enumerate(ORDER):        # peak-to-peak diurnal swing (MW) per row
        axI.text(24.5, i + 0.5, '%.0f MW' % swing_i[b], va='center', ha='left', fontsize=5.5, color='0.4')

    axS.imshow(Hm, aspect='auto', cmap=cmap, norm=norm, extent=[0, 12, n, 0])
    axS.set_xticks(np.arange(12) + 0.5); axS.set_xticklabels(MONTHS, fontsize=5.5)
    axS.set_xlabel('Month', fontsize=7)
    axS.set_yticks(np.arange(n) + 0.5)
    axS.set_yticklabels([NAMES[b] for b in ORDER], fontsize=6.5)
    # annotate seasonal peak-to-peak swing (MW) per row
    for i, b in enumerate(ORDER):
        axS.text(12.3, i + 0.5, '%.0f MW' % swing_s[b], va='center', ha='left', fontsize=5.5, color='0.4')
    for ax in (axI, axS):
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('0.6')

    cax = fig.add_axes([0.36, 0.175, 0.38, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                      ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(['charge', '0', 'discharge'])
    cb.set_label('Net pumped-storage power', fontsize=6)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6); cb.outline.set_linewidth(0.4)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_ps_heatmap'))
    print('saved fig3c_ps_heatmap (scenario B, China incl.) | seasonal swing(MW):', swing_s.round(0).to_dict())


if __name__ == '__main__':
    main()
