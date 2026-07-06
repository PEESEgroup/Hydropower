"""Fig 5 conservation panels (quantitative companion to fig5_map).

Panel e: per-bloc diverging bars of DC-affected river length, beneficial (blue, left) vs harmful
(red, right); the darker inner bar is the portion in THREATENED freshwater-fish habitat; the % at
each bar end is the share inside PROTECTED AREAS. Two columns: left = DC-CONV, right = DC-BAL.

Panel f: per-region scatter. x = % affected length inside protected areas; y = net sign
(harmful − beneficial length share, >0 harmful); size = affected length; colour = threatened
species exposed. One marker per region; the lower-left/blue = protected & beneficial (good news).

Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_panels.py [reversals|drawdown]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import revub_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'reversals'
BEN, HAR = '#2e6e8e', '#b03030'; BEN_D, HAR_D = '#16384a', '#5e1414'
BLOCS = ['north_america', 'europe', 'china', 'india', 'southeast_asia', 'south_america']
LAB = {'north_america': 'North America', 'europe': 'Europe', 'china': 'China', 'india': 'India',
       'southeast_asia': 'Southeast Asia', 'south_america': 'South America'}


def panel_e():
    b = pd.read_csv(os.path.join(DATA, 'cons_bloc_summary.csv'))
    b = b[b.metric == METRIC]
    fig, axes = plt.subplots(1, 2, figsize=(180 * S.MM, 66 * S.MM), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.86, bottom=0.16, wspace=0.08)
    xmax = b.groupby(['baseline', 'bloc', 'sign']).length_km.sum().max() / 1000 * 1.18
    for ax, (base, sub_t) in zip(axes, [('CONV', 'DC − CONV  (vs conventional)'),
                                        ('BAL', 'DC − BAL  (vs typical)')]):
        s = b[b.baseline == base]
        for i, bl in enumerate(BLOCS):
            h = s[(s.bloc == bl) & (s.sign == 'harmful')]
            be = s[(s.bloc == bl) & (s.sign == 'beneficial')]
            Lh, Lb = h.length_km.sum() / 1000, be.length_km.sum() / 1000
            Fh, Fb = h.len_in_fish_km.sum() / 1000, be.len_in_fish_km.sum() / 1000
            ax.barh(i, Lh, color=HAR, height=0.7, zorder=2)
            ax.barh(i, -Lb, color=BEN, height=0.7, zorder=2)
            ax.barh(i, Fh, color=HAR_D, height=0.34, zorder=3)
            ax.barh(i, -Fb, color=BEN_D, height=0.34, zorder=3)
            pah = h.pa_share_pct.mean() if len(h) else 0
            pab = be.pa_share_pct.mean() if len(be) else 0
            if Lh > 0:
                ax.text(Lh + xmax * 0.01, i, '%.0f%% PA' % pah, fontsize=5, va='center', ha='left', color=HAR)
            if Lb > 0:
                ax.text(-Lb - xmax * 0.01, i, '%.0f%% PA' % pab, fontsize=5, va='center', ha='right', color=BEN)
        ax.axvline(0, color='0.2', lw=0.6, zorder=4)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(-0.6, 5.6)
        ax.set_title(sub_t, fontsize=7.5, pad=3)
        ax.set_xlabel('affected river length (1000 km)', fontsize=7)
        ax.tick_params(labelsize=6.5)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([t for t in ax.get_xticks() if abs(t) <= xmax])
        ax.set_xticklabels(['%g' % abs(t) for t in ax.get_xticks()])
    axes[0].set_yticks(range(len(BLOCS))); axes[0].set_yticklabels([LAB[b_] for b_ in BLOCS], fontsize=7)
    fig.text(0.5, 0.965, 'Conservation exposure of DC flow-regime change (%s)' %
             ('flow reversals' if METRIC == 'reversals' else 'lethal drawdown'),
             ha='center', fontsize=8)
    fig.text(0.30, 0.025, '◄ beneficial', ha='center', fontsize=6.5, color=BEN)
    fig.text(0.70, 0.025, 'harmful ►', ha='center', fontsize=6.5, color=HAR)
    fig.text(0.985, 0.50, 'dark inner bar = in threatened-fish habitat', rotation=90,
             ha='right', va='center', fontsize=5.5, color='0.35')
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig5_e_%s' % METRIC)); plt.close(fig)
    print('saved fig5_e_%s' % METRIC)


def panel_f():
    r = pd.read_csv(os.path.join(DATA, 'cons_region_summary.csv'))
    r = r[r.metric == METRIC]
    fig, axes = plt.subplots(1, 2, figsize=(180 * S.MM, 74 * S.MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.88, bottom=0.13, wspace=0.06)
    # region -> bloc colour
    reg2bloc = (pd.read_csv(os.path.join(DATA, 'cons_bloc_summary.csv')))  # not needed; derive below
    for ax, (base, sub_t) in zip(axes, [('CONV', 'DC − CONV'), ('BAL', 'DC − BAL')]):
        s = r[r.baseline == base]
        rows = []
        for reg, g in s.groupby('region'):
            Lh = g[g.sign == 'harmful'].length_km.sum()
            Lb = g[g.sign == 'beneficial'].length_km.sum()
            L = Lh + Lb
            if L <= 0:
                continue
            pa_len = g.len_in_pa_km.sum()
            nthr = g.n_threatened_sp.sum()
            rows.append(dict(region=reg, L=L, pa_share=100 * pa_len / L,
                             net=100 * (Lh - Lb) / L, nthr=nthr))
        d = pd.DataFrame(rows)
        sc = ax.scatter(d.pa_share, d.net, s=6 + 60 * np.sqrt(d.L / d.L.max()),
                        c=np.log10(d.nthr + 1), cmap='viridis', alpha=0.85, lw=0.3, ec='white', zorder=3)
        ax.axhline(0, color='0.4', lw=0.6, ls='--')
        ax.set_xlabel('% affected length inside protected areas', fontsize=7)
        ax.set_title(sub_t, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('net effect  (harmful − beneficial, %)', fontsize=7)
    axes[0].text(0.02, 0.96, 'harmful ▲', transform=axes[0].transAxes, fontsize=6, va='top', color=HAR)
    axes[0].text(0.02, 0.04, 'beneficial ▼', transform=axes[0].transAxes, fontsize=6, va='bottom', color=BEN)
    cax = fig.add_axes([0.915, 0.2, 0.013, 0.6])
    cb = fig.colorbar(sc, cax=cax); cb.set_label('threatened fish exposed (log)', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.5, 0.965, 'Per-region conservation exposure (%s); each point = a region' %
             ('flow reversals' if METRIC == 'reversals' else 'lethal drawdown'),
             ha='center', fontsize=8)
    S.save(fig, os.path.join(OUT, 'fig5_f_%s' % METRIC)); plt.close(fig)
    print('saved fig5_f_%s' % METRIC)


if __name__ == '__main__':
    panel_e()
    panel_f()
