"""Fig 3c - the three firm-power amplifiers: pumped storage, cascade, cross-region interconnection.

Left: pumped storage is a SEASONAL lever - it delivers its firm power in each region's scarce /
peak season (winter in China, summer in the US), not within the day (intraday cycling is < a few %
of the seasonal swing). Monthly PS generation, normalised to each region's peak month.
Right: how much DC-facing firm capacity each lever adds globally (program D supply_A->B = PS,
B->C = cascade; program E E0->E1 = interconnection). Interconnection is the dominant amplifier;
the local levers are modest. PS/cascade show the 5-GCM x 3-SSP spread; interconnection is the
reference run (E2 ran A/B only). Arial 7 pt; transparent; no title.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import revub_style as S
from revub_names import region_label

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')

C_PS, C_CASC, C_ITC = '#756bb1', '#41ab5d', '#08519c'   # storage / cascade / interconnection
MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
# representative PS regions, distinct scarce-season peaks (winter / summer / autumn)
REP = [('china_ec', '#d6604d'), ('usa_caiso', '#4393c3'), ('italy', '#1a9850')]


def main():
    # ---- left: PS seasonal generation shape ----
    ps = pd.read_csv(os.path.join(DATA, 'ps_monthly.csv'))

    # ---- right: three-lever global DC-firm gain (GW) ----
    d = pd.read_csv(os.path.join(DATA, 'dc_supply_canon.csv'))
    g = d.groupby(['gcm', 'ssp'])[['supply_A_mw', 'supply_B_mw', 'supply_C_mw']].sum()
    ps_gw = (g.supply_B_mw - g.supply_A_mw) / 1000
    ca_gw = (g.supply_C_mw - g.supply_B_mw) / 1000
    cov = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DATA, 'cov_*.csv'))])
    cov = cov[cov['scenario'] == 'A']
    itc_gw = (cov['met_interconnected_mw'].sum() - cov['met_isolated_mw'].sum()) / 1000

    fig = plt.figure(figsize=(180 * S.MM, 74 * S.MM))
    axL = fig.add_axes([0.065, 0.21, 0.40, 0.60])
    axR = fig.add_axes([0.605, 0.21, 0.375, 0.60])

    # LEFT --------------------------------------------------------------
    for reg, col in REP:
        sub = ps[ps['region'] == reg].sort_values('month')
        if not len(sub):
            continue
        gen = sub['gen_mwh_day'].to_numpy()
        shape = gen / gen.max() if gen.max() > 0 else gen
        axL.plot(range(1, 13), shape, '-o', color=col, lw=1.4, ms=3, mec='white', mew=0.4,
                 label='%s (%.0f GWh yr$^{-1}$)' % (region_label(reg), sub['gen_mwh_day'].sum() * 365 / 1000))
    axL.set_xticks(range(1, 13))
    axL.set_xticklabels(MONTHS, fontsize=7)
    axL.set_xlim(0.5, 12.5)
    axL.set_ylim(0, 1.08)
    axL.set_ylabel('PS firm generation\n(fraction of peak month)', fontsize=7)
    axL.set_xlabel('Month', fontsize=7)
    axL.tick_params(labelsize=7, width=0.4, length=2)
    axL.legend(frameon=False, fontsize=6, loc='upper center', handlelength=1.4,
               handletextpad=0.4, labelspacing=0.25)
    axL.text(0.0, 1.06, 'Pumped storage: a seasonal lever',
             transform=axL.transAxes, fontsize=7, fontweight='bold', color='0.15')
    for sp in axL.spines.values():
        sp.set_visible(False)
    axL.axhline(0, color='0.1', lw=0.5)

    # RIGHT -------------------------------------------------------------
    y_ps, y_ca, y_it = 2, 1, 0
    # pumped storage: solid bar + 15-scenario whisker
    axR.barh(y_ps, ps_gw.median(), color=C_PS, ec='none', height=0.55, zorder=2)
    axR.plot([ps_gw.min(), ps_gw.max()], [y_ps, y_ps], color='0.25', lw=0.9, zorder=3)
    for xx in (ps_gw.min(), ps_gw.max()):
        axR.plot([xx, xx], [y_ps - 0.13, y_ps + 0.13], color='0.25', lw=0.9, zorder=3)
    axR.text(ps_gw.max() * 1.4, y_ps, '%.2f GW' % ps_gw.median(), va='center', ha='left',
             fontsize=6.5, color=C_PS, fontweight='bold')
    # cascade: uncertain band (cache ~0.3 to audited reference ~9.5 GW), value pending
    axR.fill_betweenx([y_ca - 0.27, y_ca + 0.27], 0.3, 9.5, color=C_CASC, alpha=0.35, zorder=2)
    for xx in (0.3, 9.5):
        axR.plot([xx, xx], [y_ca - 0.27, y_ca + 0.27], color=C_CASC, lw=0.8, zorder=3)
    axR.text(9.5 * 1.4, y_ca, '0.3 to 9.5 GW*', va='center', ha='left', fontsize=6.5,
             color=C_CASC, fontweight='bold')
    # interconnection: solid bar (reference run)
    axR.barh(y_it, itc_gw, color=C_ITC, ec='none', height=0.55, zorder=2)
    axR.text(itc_gw * 1.18, y_it, '%.0f GW' % itc_gw, va='center', ha='left',
             fontsize=6.5, color=C_ITC, fontweight='bold')
    axR.set_yticks([y_ps, y_ca, y_it])
    axR.set_yticklabels(['Pumped\nstorage', 'Cascade', 'Cross-region\ninterconnection'], fontsize=7)
    axR.set_ylim(-0.6, 2.6)
    axR.set_xscale('log')
    axR.set_xlim(0.05, 250)
    axR.set_xticks([0.1, 1, 10, 100])
    axR.set_xticklabels(['0.1', '1', '10', '100'])
    axR.tick_params(axis='x', labelsize=7, width=0.4, length=2)
    axR.tick_params(axis='y', length=0)
    axR.set_xlabel('DC firm capacity added, global (GW, log scale)', fontsize=7)
    axR.text(0.0, 1.06, 'Interconnection is the dominant amplifier',
             transform=axR.transAxes, fontsize=7, fontweight='bold', color='0.15')
    axR.text(1.0, -0.30, '*cascade reference value; per-scenario source pending',
             transform=axR.transAxes, ha='right', va='top', fontsize=5.5, color='0.5')
    for sp in axR.spines.values():
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig3c_levers'))
    print('saved fig3c_levers | PS=%.2f cascade=%.2f-%.2f interconnection=%.1f GW'
          % (ps_gw.median(), ca_gw.median(), 9.5, itc_gw))


if __name__ == '__main__':
    main()
