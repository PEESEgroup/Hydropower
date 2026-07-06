"""Fig 1b - 24h diurnal profiles for 4 archetype regions.

Hydropower firm output stacked by component (run-of-river P_r, storage P_s, flexible/PS P_f;
PS pumping = negative P_f drawn below zero) vs the near-flat 24/7 DC load. Shows the firm
mechanism: surplus (Norway) easily covers the flat DC load; cascade+PS (China SW) shows the
pumping dip; deficit regions (Malaysia W, China NC) fall far below the DC platform.
All text Arial 7 pt (project spec).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import revub_style as S
from fig1a_coverage_map import load_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
DC_NPZ = '/home/cfeng/hydro/dc_load_simulation/results/region_hourly_mri-esm2-0_ssp370_2040.npz'

REGIONS = [('norway', 'Norway', 'surplus'),
           ('china_sw', 'China SW', 'cascade + pumped storage'),
           ('malaysia_west', 'Malaysia (West)', 'deficit'),
           ('china_nc', 'China NC', 'interconnection-reliant')]

C_R, C_S, C_FG, C_FP = '#0072B2', '#009E73', '#E69F00', '#CC79A7'  # ror/storage/flex-gen/pumping


def hydro_diurnal():
    df = pd.read_csv(os.path.join(DATA, 'fig1b_hydro.csv'))
    out = {}
    for r in df['region'].unique():
        sub = df[df['region'] == r].set_index('comp')
        cols = [f'h{i}' for i in range(24)]
        out[r] = {c: sub.loc[c, cols].to_numpy(float) for c in ['P_s_C', 'P_f_C', 'P_r_C']}
    return out


def dc_diurnal():
    # Prefer the bundled diurnal CSV (region, h0..h23 in MW). Fall back to the raw
    # hourly DC-load simulation (DC_NPZ, not redistributed) to rebuild it if absent.
    csv = os.path.join(DATA, 'fig1b_dc_diurnal.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv).set_index('region')
        cols = [f'h{i}' for i in range(24)]
        return {r: df.loc[r, cols].to_numpy(float) for r in df.index}
    z = np.load(DC_NPZ)
    out = {}
    for r, _, _ in REGIONS:
        k = f'{r}__P_total_kw'
        if k not in z:
            print('MISSING DC load:', k); continue
        out[r] = z[k].reshape(365, 24).mean(axis=0) / 1000.0   # MW, average diurnal
    return out


def main():
    H = hydro_diurnal()
    D = dc_diurnal()
    cov = load_coverage().set_index('region')['cov_isolated_pct']
    hours = np.arange(24)

    fig, axes = plt.subplots(2, 2, figsize=(120 * S.MM, 100 * S.MM))
    axes = axes.ravel()

    for ax, (r, name, role) in zip(axes, REGIONS):
        pr, ps, pf = H[r]['P_r_C'], H[r]['P_s_C'], H[r]['P_f_C']
        pf_pos, pf_neg = np.clip(pf, 0, None), np.clip(pf, None, 0)
        ax.stackplot(hours, pr, ps, pf_pos, colors=[C_R, C_S, C_FG],
                     labels=['Run-of-river (Pr)', 'Storage (Ps)', 'Flexible/PS gen (Pf)'])
        if (pf_neg < 0).any():
            ax.fill_between(hours, 0, pf_neg, color=C_FP, label='PS pumping (Pf<0)')
        dc = D.get(r)
        if dc is not None:
            ax.plot(hours, dc, color='k', lw=1.2, label='DC load')
        ax.axhline(0, color='0.5', lw=0.4)
        ax.set_xlim(0, 23)
        ax.set_title(f'{name} - {role}  (E0 {cov.get(r, np.nan):.0f}%)')
        ax.margins(y=0.12)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel('Hour of day')
    for ax in axes[0::2]:
        ax.set_ylabel('Power (MW)')

    h, lab = axes[0].get_legend_handles_labels()
    # ensure pumping handle (only present in some panels) is included
    for ax in axes:
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            if b not in lab:
                lab.append(b); h.append(a)
    fig.legend(h, lab, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig1b_diurnal'))
    print('saved', os.path.join(OUT, 'fig1b_diurnal.svg'), '(+.png)')


if __name__ == '__main__':
    main()
