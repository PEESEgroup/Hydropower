"""Fig 6 - synthesis: carbon opportunity vs (small, mixed) river cost + energy autonomy + equity.

Six information-rich panels (each region shown as a point within its income group; explanatory prose
lives in the CAPTION, not on the panels). Data: equity_master.csv + fig4_eco.csv + climate_equity.csv.
Reference mri/ssp370/flat. Run: python fig6_panels.py [A B C D E G]   (default: all). See FIG6_design.md.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import revub_style as S
import revub_names as N

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data'); OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
FS = 7
ORDER = ['developing', 'emerging', 'developed']
LAB = {'developing': 'Developing', 'emerging': 'Emerging', 'developed': 'Developed'}
COL = {'developing': '#c1666b', 'emerging': '#e8a04c', 'developed': '#4d7ea8'}
GREEN, GREY, WARM = '#3a9d5d', '#b6bcc4', '#cf5b56'
rng = np.random.default_rng(1)


def prep():
    em = pd.read_csv(os.path.join(DATA, 'equity_master.csv'))
    eco = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    r = eco[eco.metric == 'reversals_per_year_hourly'].copy()
    r['typ'] = r.DC_A - r.BAL_C; r['conv'] = r.DC_A - r.CONV
    rg = r.groupby('region').agg(rev_typ_med=('typ', 'median'), rev_conv_med=('conv', 'median')).reset_index()
    cl = pd.read_csv(os.path.join(DATA, 'climate_equity.csv'))[['region', 'crossgcm_cv_pct', 'cov_pctchg_126_585']]
    d = em.merge(rg, on='region', how='left').merge(cl, on='region', how='left')
    d['carbon_Mt'] = d.net_carbon_avoided / 1e6
    d['dom_share'] = 100 * np.minimum(d.supply_C_mw, d.dc_mean_mw) / d.dc_mean_mw
    # True operating-margin grid emission factor (gCO2eq/kWh), backed out from the carbon identity:
    # net_avoided = E_hyd*(EF_grid - EF_hyd) and dc_carbon_intensity = (hydro + grid-shortfall) blend,
    # which algebraically give EF_grid = dc_carbon_intensity + net_carbon_avoided*1000/(dc_mean_mw*8760).
    # Validated: reproduces the China MEE OM range (596-1047) and the US eGRID non-baseload factors exactly.
    d['ef_grid'] = d.dc_carbon_intensity + d.net_carbon_avoided * 1000.0 / (d.dc_mean_mw * 8760.0)
    return d


def finish(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig6_' + stem)); plt.close(fig)
    print('saved fig6_' + stem)


def rlabel(r):                                          # proper geographic name; flag US subregions
    s = N.region_label(r)
    return (s + ' (US)') if str(r).startswith('usa_') else s


def strip(ax, d, col, gsum, sumfmt, *, logx=False, xlim=None, sizecol=None, vline=None,
          shade=None, labels=None, jitter=0.24):
    """One row of region points per income group; group summary in the right margin."""
    smax = np.sqrt(d[sizecol].clip(lower=0)).max() if sizecol else 1.0
    for i, g in enumerate(ORDER):
        sub = d[d.income_group == g].dropna(subset=[col]); v = sub[col].to_numpy(); y = i
        ss = (6 + 95 * np.sqrt(sub[sizecol].clip(lower=0)).to_numpy() / smax) if sizecol else 13
        ax.scatter(v, y + rng.uniform(-jitter, jitter, len(v)), s=ss, color=COL[g], alpha=0.55, lw=0.25, ec='white', zorder=3)
        ax.text(-0.02, y, LAB[g], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.2')
        ax.text(1.02, y, sumfmt(gsum(sub)), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS, fontweight='bold', color=COL[g])
    if shade:
        ax.axvspan(shade[0], shade[1], color=shade[2], alpha=0.06, zorder=0)
    if vline is not None:
        ax.axvline(vline, ls='--', color='0.5', lw=0.7, zorder=1)
    if logx:
        ax.set_xscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=FS - 1.5, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    if labels:
        for region, dx, dy in labels:
            row = d[d.region == region]
            if len(row):
                xv = row[col].values[0]; yv = ORDER.index(row.income_group.values[0])
                ax.annotate(rlabel(region), (xv, yv), xytext=(dx, dy), textcoords='offset points',
                            fontsize=FS - 2.3, color='0.3', ha='center',
                            arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4))


# ----------------------------------------------------------------------------- A: river reframe
def panelA(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.30, right=0.80, top=0.95, bottom=0.15)
    for i, g in enumerate(ORDER):
        sub = d[d.income_group == g].dropna(subset=['rev_typ_med']); v = sub.rev_typ_med.to_numpy(); y = i
        ax.scatter(v, y + rng.uniform(-0.20, 0.20, len(v)), s=12, color=COL[g], alpha=0.55, lw=0, zorder=3)
        med = np.median(v); ax.plot([med, med], [y - 0.34, y + 0.34], color=COL[g], lw=2.4, zorder=4, solid_capstyle='round')
        ax.text(1.02, y + 0.16, '≈0' if abs(med) < 0.5 else '%+.0f' % med, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS, color=COL[g], fontweight='bold')
        ax.text(1.02, y - 0.18, '%.0f%% gentler' % (100 * (v < 0).mean()), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS - 1.7, color='0.45')
        ax.text(-0.02, y, LAB[g], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.2')
    ax.axvline(0, ls='--', color='0.5', lw=0.7, zorder=1); ax.axvspan(-260, 0, color=GREEN, alpha=0.06, zorder=0)
    ax.set_xlim(-260, 120); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=FS - 1, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.text(0.02, 0.97, '← gentler than the grid', transform=ax.transAxes, fontsize=FS - 1.7, color=GREEN, ha='left', va='top')
    ax.set_xlabel('DC-marginal flow reversals (/yr)', fontsize=FS - 0.5, color='0.25', labelpad=3)
    finish(fig, 'a')


# ----------------------------------------------------------------------------- B: carbon (stacked by region, coloured by grid intensity)
def panelB(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.21, right=0.78, top=0.78, bottom=0.13)
    glob = d.carbon_Mt.sum()
    norm = mcolors.LogNorm(vmin=15, vmax=1200); cmap = plt.get_cmap('YlOrRd')
    for i, g in enumerate(ORDER):
        sub = d[d.income_group == g].sort_values('carbon_Mt', ascending=False)
        y = len(ORDER) - 1 - i; left = 0.0
        for _, row in sub.iterrows():
            w = row.carbon_Mt
            if not (np.isfinite(w) and w > 0):
                continue
            ci = row.dc_carbon_intensity
            c = cmap(norm(ci)) if np.isfinite(ci) and ci > 0 else '0.82'
            ax.barh(y, w, left=left, color=c, ec='white', lw=0.4, height=0.6, zorder=3)
            left += w
        big = sub.iloc[0]
        ax.annotate(rlabel(big.region), (big.carbon_Mt / 2, y + 0.31), xytext=(0, 8),
                    textcoords='offset points', fontsize=FS - 2.3, color='0.35', ha='center', arrowprops=dict(arrowstyle='-', color='0.65', lw=0.4))
        ax.text(1.02, y, '%.0f Mt\n%.0f%%' % (left, 100 * left / glob), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS - 1, fontweight='bold', color='0.3', linespacing=0.95)
        ax.text(-0.02, y, LAB[g], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.2')
    ax.set_xlim(0, 172); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_yticks([])
    ax.set_xlabel('net carbon avoided (Mt CO$_2$eq/yr)', fontsize=FS - 1)
    ax.tick_params(axis='x', labelsize=FS - 1.5, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    cax = fig.add_axes([0.21, 0.93, 0.42, 0.032])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_label_position('top'); cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label('DC carbon intensity (g CO$_2$eq/kWh)', fontsize=FS - 2.2, labelpad=2)
    cb.ax.tick_params(labelsize=FS - 2.5, width=0.3, length=1.5); cb.outline.set_linewidth(0.3)
    finish(fig, 'b')


# ----------------------------------------------------------------------------- C: energy autonomy (bar thickness = DC demand)
def panelC(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.22, right=0.74, top=0.86, bottom=0.14)
    dems = {g: d[d.income_group == g].dc_mean_mw.sum() / 1e3 for g in ORDER}
    hmax = np.sqrt(max(dems.values()))
    for i, g in enumerate(ORDER):
        sub = d[d.income_group == g]
        dem = dems[g]; dom = np.minimum(sub.supply_C_mw, sub.dc_mean_mw).sum() / sub.dc_mean_mw.sum() * 100
        y = len(ORDER) - 1 - i; h = 0.16 + 0.66 * np.sqrt(dem) / hmax
        ax.barh(y, dom, height=h, color=GREEN, zorder=3)
        ax.barh(y, 100 - dom, left=dom, height=h, color=GREY, zorder=3)
        ax.text(dom - 2, y, '%.0f%%' % dom, ha='right', va='center', fontsize=FS, fontweight='bold', color='white')
        if 100 - dom > 9:
            ax.text(dom + (100 - dom) / 2, y, '%.0f%%' % (100 - dom), ha='center', va='center', fontsize=FS - 1.5, color='white')
        ax.text(-0.03, y, LAB[g], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.2')
        ax.text(1.02, y, '%.0f GW\ndemand' % dem, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=FS - 1.7, color='0.4', linespacing=0.95)
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, len(ORDER) - 0.3); ax.set_yticks([])
    ax.set_xlabel('% of demand firmable by domestic hydro', fontsize=FS - 1, labelpad=3)
    ax.tick_params(axis='x', labelsize=FS - 1, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.scatter([], [], marker='s', s=22, color=GREEN, label='domestic clean firm (hydro)')
    ax.scatter([], [], marker='s', s=22, color=GREY, label='needs import / fossil firming')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=FS - 1.7, frameon=False, ncol=2, handletextpad=0.3, columnspacing=1.0)
    finish(fig, 'c')


# ----------------------------------------------------------------------------- D: siting paradox
LBL_D = {'india_south': 'India South\n(top developing site)', 'china_ec': 'China East\n(short, dirty)',
         'sweden': 'Sweden\n(surplus, clean grid)'}


def panelD(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.15, right=0.82, top=0.95, bottom=0.16)
    s = d.dropna(subset=['balance_ratio', 'ef_grid', 'net_carbon_avoided']).copy()
    s = s[(s.ef_grid > 0) & (s.balance_ratio > 0)]
    sz = 8 + 130 * np.sqrt(s.net_carbon_avoided.clip(lower=0) / s.net_carbon_avoided.clip(lower=0).max())
    ax.axvspan(1, 6, color=GREEN, alpha=0.05, zorder=0); ax.axhline(100, color='0.7', lw=0.5, ls=':', zorder=1); ax.axvline(1, color='0.7', lw=0.5, ls=':', zorder=1)
    for g in ORDER:
        m = s.income_group == g
        ax.scatter(s.balance_ratio[m], s.ef_grid[m], s=sz[m], color=COL[g], alpha=0.62, lw=0.3, ec='white', zorder=3, label=LAB[g])
    for r, t in LBL_D.items():
        row = s[s.region == r]
        if len(row):
            ax.annotate(t, (row.balance_ratio.values[0], row.ef_grid.values[0]), fontsize=FS - 2.3, color='0.3', ha='center',
                        xytext=(0, 16 if r != 'china_ec' else -20), textcoords='offset points', arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.18, 6); ax.set_ylim(40, 2500)
    ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_yticks([50, 100, 500, 1000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: '%g' % v)); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: '%g' % v))
    ax.tick_params(labelsize=FS - 1.5, width=0.4, length=2)
    ax.set_xlabel('hydro surplus → (firm/demand)', fontsize=FS - 1); ax.set_ylabel('marginal grid intensity (g CO$_2$eq/kWh)', fontsize=FS - 1)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.4)
    ax.text(0.98, 0.96, 'surplus + dirty grid\n= ideal DC site', transform=ax.transAxes, ha='right', va='top', fontsize=FS - 2, color=GREEN)
    hdl = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=COL[g], markeredgecolor='white', markeredgewidth=0.3, markersize=4.5, label=LAB[g]) for g in ORDER]
    leg = ax.legend(handles=hdl, loc='lower left', fontsize=FS - 2, frameon=True, handletextpad=0.4, bbox_to_anchor=(0.0, 0.0), labelspacing=0.7, borderpad=0.4)
    leg.get_frame().set_facecolor('white'); leg.get_frame().set_alpha(0.85); leg.get_frame().set_edgecolor('0.8'); leg.get_frame().set_linewidth(0.4); leg.set_zorder(6)
    finish(fig, 'd')


# ----------------------------------------------------------------------------- E: grid displacement (paired bars)
def panelE(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.22, right=0.78, top=0.93, bottom=0.16)
    sx = lambda v: np.sqrt(np.maximum(v, 0.0))
    for i, g in enumerate(ORDER):
        sub = d[d.income_group == g]
        toDC = sub.supply_C_mw.sum() / 1e3; firm = toDC + sub.grid_resid_C_mw.sum() / 1e3
        dem = sub.dc_mean_mw.sum() / 1e3; y = len(ORDER) - 1 - i
        ax.barh(y, sx(firm), color=GREY, height=0.5, zorder=2)
        ax.barh(y, sx(toDC), color=COL[g], height=0.5, zorder=3)
        ax.scatter(sx(dem), y, marker='D', s=28, color='0.12', zorder=5, ec='white', lw=0.5)
        ax.text(sx(toDC) / 2, y, '%.0f%%' % (100 * toDC / firm), ha='center', va='center', fontsize=FS - 1.5, color='white', fontweight='bold', zorder=6)
        ax.text(-0.02, y, LAB[g], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS, color='0.2')
        ax.text(sx(firm), y + 0.33, '%.0f GW' % firm, ha='center', va='bottom', fontsize=FS - 2, color='0.5')
    ax.set_xlim(0, 15.5); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_yticks([])
    ax.set_xticks(sx(np.array([0, 25, 100, 200]))); ax.set_xticklabels([0, 25, 100, 200], fontsize=FS - 1.5)
    ax.set_xlabel('firm hydropower (GW, √-scaled)', fontsize=FS - 1)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.scatter([], [], marker='s', s=22, color='0.5', label='left to grid')
    ax.scatter([], [], marker='D', s=22, color='0.12', label='DC demand')
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, -0.02), fontsize=FS - 1.7, frameon=False, ncol=1, handletextpad=0.2, labelspacing=0.3)
    finish(fig, 'e')


# ----------------------------------------------------------------------------- G: climate exposure
def panelG(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.21, right=0.80, top=0.92, bottom=0.16)
    strip(ax, d, 'crossgcm_cv_pct', lambda s: s.crossgcm_cv_pct.median(), lambda x: '%.1f%%' % x,
          logx=True, xlim=(1.3, 80), labels=[('china_nc', -2, 15), ('usa_miso_north', 0, -16)])
    ax.set_xlabel('cross-model firm-supply CV (%, log)', fontsize=FS - 1)
    ax.text(1.02, len(ORDER) - 0.55, 'group\nmedian', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=FS - 2.3, color='0.45')
    ax.text(0.98, 0.04, 'more climate-uncertain →', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS - 1.7, color=WARM)
    finish(fig, 'g')


# ----------------------------------------------------------------------------- F: cross-region transmission
def panelH(d):
    fig, ax = S.fig_mm(70, 74.25); fig.subplots_adjust(left=0.36, right=0.83, top=0.80, bottom=0.16)
    e = d.dropna(subset=['net_export_mw']).copy(); e['gw'] = e.net_export_mw / 1e3
    top = pd.concat([e.nlargest(7, 'gw'), e.nsmallest(6, 'gw')]).drop_duplicates('region').sort_values('gw')
    for y, (_, r) in enumerate(top.iterrows()):
        ax.barh(y, r.gw, color=COL[r.income_group], height=0.72, zorder=3)
        ax.text(-0.02, y, rlabel(r.region), transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS - 1.5, color='0.25')
        ax.text(r.gw + (0.25 if r.gw >= 0 else -0.25), y, '%+.1f' % r.gw, ha='left' if r.gw >= 0 else 'right', va='center', fontsize=FS - 2, color=COL[r.income_group])
    ax.axvline(0, color='0.4', lw=0.6, zorder=2)
    ax.set_xlim(-14, 12); ax.set_ylim(-0.7, len(top) - 0.3); ax.set_yticks([])
    ax.set_xlabel('← imports (depends)        net firm-power export (GW)        supplies others →', fontsize=FS - 1.7)
    ax.tick_params(axis='x', labelsize=FS - 1.5, width=0.4, length=2)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    for g in ORDER:
        ax.scatter([], [], marker='s', s=20, color=COL[g], label=LAB[g])
    ax.legend(loc='lower right', bbox_to_anchor=(1.04, 0.02), fontsize=FS - 2, frameon=False, labelspacing=0.2, handletextpad=0.2)
    finish(fig, 'f')


PANELS = {'a': panelA, 'b': panelB, 'c': panelC, 'd': panelD, 'e': panelE, 'g': panelG}

if __name__ == '__main__':
    d = prep()
    sel = [a for a in sys.argv[1:] if a in PANELS] or list(PANELS)
    for k in sel:
        PANELS[k](d)
