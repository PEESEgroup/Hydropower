"""Fig 3c-i (SCENARIO-C backup, superseded) - pumped storage charge/discharge by hour & month.

Backup of the original scenario-C version (program-D dc_scenario_C_hourly.npz). Superseded by
fig3c_ps_heatmap.py, which uses scenario B: scenario C contaminates china_sw with a cascade
re-dispatch artifact (pumps ~2.3 TWh/yr, returns 0.6%), so the PS lever (supply_B - supply_A) is
better depicted by scenario B. Kept for the record. Reads ps_*_C.csv. Arial 7 pt; no title.
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
ORDER = ['north_america', 'europe', 'southeast_asia', 'india']   # China excluded (C-scenario artifact)
MONTHS = list('JFMAMJJASOND')


def main():
    r2b = {r: b for b, rs in TRADING_BLOCS.items() for r in rs}
    di = pd.read_csv(os.path.join(DATA, 'ps_diurnal_C.csv')); di['bloc'] = di.region.map(r2b)
    mo = pd.read_csv(os.path.join(DATA, 'ps_monthly_C.csv')); mo['bloc'] = mo.region.map(r2b)
    mo['net_mw'] = (mo.gen_mwh_day - mo.pump_mwh_day) / 24.0
    dih = di.groupby(['bloc', 'hour']).net_mw.sum().unstack().reindex(ORDER)
    mom = mo.groupby(['bloc', 'month']).net_mw.sum().unstack().reindex(ORDER)
    amp = mom.abs().max(axis=1)
    Hd = dih.div(amp, axis=0).to_numpy()
    Hm = mom.div(amp, axis=0).to_numpy()

    cmap = plt.get_cmap('PiYG')
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    n = len(ORDER)
    fig = plt.figure(figsize=(70 * S.MM, 66 * S.MM))
    axI = fig.add_axes([0.34, 0.70, 0.54, 0.27])
    axS = fig.add_axes([0.34, 0.31, 0.54, 0.27])

    axI.imshow(Hd, aspect='auto', cmap=cmap, norm=norm, extent=[0, 24, n, 0])
    axI.set_xticks([0.5, 6, 12, 18, 23.5]); axI.set_xticklabels(['0', '6', '12', '18', '24'], fontsize=6.5)
    axI.set_xlabel('Hour of day', fontsize=7)
    axI.set_yticks(np.arange(n) + 0.5)
    axI.set_yticklabels([NAMES[b] for b in ORDER], fontsize=6.5)

    axS.imshow(Hm, aspect='auto', cmap=cmap, norm=norm, extent=[0, 12, n, 0])
    axS.set_xticks(np.arange(12) + 0.5); axS.set_xticklabels(MONTHS, fontsize=5.5)
    axS.set_xlabel('Month', fontsize=7)
    axS.set_yticks(np.arange(n) + 0.5)
    axS.set_yticklabels([NAMES[b] for b in ORDER], fontsize=6.5)
    for i, b in enumerate(ORDER):
        axS.text(12.3, i + 0.5, '%.0f MW' % amp[b], va='center', ha='left', fontsize=5.5, color='0.4')
    for ax in (axI, axS):
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('0.6')

    cax = fig.add_axes([0.36, 0.175, 0.38, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                      ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(['charge', '0', 'discharge'])
    cb.set_label('Net PS power (per-bloc norm.)', fontsize=6)
    cb.ax.tick_params(width=0.4, length=2, labelsize=6); cb.outline.set_linewidth(0.4)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_ps_heatmap_C'))
    print('saved fig3c_ps_heatmap_C (scenario-C backup) | seasonal amp(MW):', amp.round(0).to_dict())


if __name__ == '__main__':
    main()
