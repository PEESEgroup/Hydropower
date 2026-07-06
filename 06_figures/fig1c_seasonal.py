"""Fig 1c - multi-year seasonal firm-coverage heatmap (deficit regions only).

Every modelled year 2027-2040 is shown explicitly (NOT time-averaged) to demonstrate
that adequacy was explored across many hydrological years. Monthly firm coverage =
clip(100 * monthly-mean hydro firm output / monthly-mean DATA-CENTER load, 0, 100).
Numerator: each region's dc_scenario_A_hourly.npz (P_s+P_f+P_r, all stations) - scenario A,
independent stations, NO cascade/PS coordination (the Fig1 baseline), per year.
Denominator: the ACTUAL data-center load profile (region_hourly_<gcm>_<ssp>_<year>.npz,
P_total), aggregated to monthly means PER YEAR - NOT a flat load: cooling drives a real
seasonal swing (e.g. china_ec ~29%, germany ~30% month-to-month) and a diurnal cycle.
Both come from the same climate year, so coverage is climate-consistent. Built into
data/monthly_cov_years.csv by hydro_firm_monthly.csv (remote) / monthly DC load (local).

Only regions that CANNOT fully supply (min monthly coverage < 99.5% in some year-month)
are shown; perfectly-covered regions carry no seasonal signal. Rows = regions grouped by
trading bloc (north->south within bloc); columns = 14 years x 12 months (year-major barcode),
thin white dividers between years. Same YlGnBu coverage colormap as Fig1a.
All text Arial 7 pt; PNG 300 ppi; no title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import revub_style as S
from revub_geo import load_regions
from revub_names import region_label, region_label_us
from fig1a_coverage_map import truncate_cmap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')

YEARS = list(range(2027, 2041))   # 14 modelled years
MONTHS = 12
DEFICIT_THRESH = 90.0    # region kept only if its worst month falls below this - genuine seasonal
                         # deficits; drops the ~29 near-fully-covered "all-blue" marginal regions
PEAK_THRESH = 5.0        # ...AND its best month reaches >= this (drop always-zero 'no-signal'
                         # regions; lowered 20->5 to show more partial-coverage regions)
def _disp(r):
    # canonical no-acronym geographic name (shared by fig2a/2b/2c via this import) - NO (US)
    # tag here, to keep Fig 2's dense inline labels short
    return region_label(r)


def _ylabel(r):
    # Fig 1c y-axis row labels: US subregions get a ' (US)' tag
    return region_label_us(r)


def main():
    df = pd.read_csv(os.path.join(DATA, 'monthly_cov_years.csv'))
    mcols = [f'm{i}' for i in range(1, 13)]

    # keep regions that are a deficit (worst month < 99.5%) AND reach >= PEAK_THRESH in their
    # best month - i.e. partial, seasonally-variable coverage. Always-near-0 regions (no signal)
    # and always-saturated regions are dropped.
    grp = df.groupby('region')[mcols]
    worst, peak = grp.min().min(axis=1), grp.max().max(axis=1)
    keep = worst[(worst < DEFICIT_THRESH) & (peak >= PEAK_THRESH)].index
    df = df[df['region'].isin(keep)].copy()

    g = load_regions()[['region', 'bloc', 'geometry']].copy()
    g['lat'] = g.geometry.representative_point().y
    meta = g.set_index('region')[['bloc', 'lat']]

    regions = pd.DataFrame({'region': sorted(keep)})
    regions = regions.merge(meta, left_on='region', right_index=True, how='left')
    regions['bord'] = regions['bloc'].map({b: i for i, b in enumerate(S.BLOC_ORDER)}).fillna(99)
    regions = regions.sort_values(['bord', 'lat'], ascending=[True, False]).reset_index(drop=True)
    order = regions['region'].tolist()
    n = len(order)
    print('deficit regions shown: %d (of 75), grouped by bloc' % n)

    # build (n_regions, 14*12) barcode: columns year-major (2027 m1..m12, 2028 m1..m12, ...)
    cube = (df.pivot_table(index='region', columns='year', values=mcols)
              .reindex(order))                                   # MultiIndex cols (mX, year)
    mat = np.full((n, len(YEARS) * MONTHS), np.nan)
    for j, yr in enumerate(YEARS):
        block = np.column_stack([cube[(mc, yr)].to_numpy(float) for mc in mcols])
        mat[:, j * MONTHS:(j + 1) * MONTHS] = block

    cmap = truncate_cmap('GnBu', 0.15, 0.6)    # middle of colorbrewer GnBu - lighter, navy end dropped
    norm = Normalize(0, 100)
    ncol = len(YEARS) * MONTHS

    fig, ax = S.fig_mm(118, 112)
    fig.subplots_adjust(left=0.26, right=0.90, top=0.965, bottom=0.12)
    ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
              extent=(0, ncol, n - 0.5, -0.5))
    ax.set_xlim(0, ncol + ncol * 0.018)
    ax.set_ylim(n - 0.5, -0.5)

    # year dividers + year labels (centred under each 12-month block)
    for j in range(1, len(YEARS)):
        ax.axvline(j * MONTHS, color='white', lw=0.4)
    ax.set_xticks([(j + 0.5) * MONTHS for j in range(len(YEARS))])
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=7, rotation=90)
    ax.tick_params(axis='x', length=0, pad=1.5)
    ax.set_xlabel('Year  (each block = Jan-Dec)', fontsize=7)

    ax.set_yticks(range(n))
    ax.set_yticklabels([_ylabel(r) for r in order], fontsize=7, linespacing=0.9)
    ax.tick_params(axis='y', length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # bloc grouping: white dividers + right-side colour bar + bloc name
    barx = ncol + ncol * 0.004
    start = 0
    for b in S.BLOC_ORDER:
        idx = [i for i, r in enumerate(order) if regions.loc[i, 'bloc'] == b]
        if not idx:
            continue
        k = len(idx)
        if start > 0:
            ax.axhline(start - 0.5, color='white', lw=0.8)
        ax.add_patch(Rectangle((barx, start - 0.5), ncol * 0.016, k,
                     facecolor=S.BLOC_COLOR[b], ec='white', lw=0.3, clip_on=False))
        if k >= 2:   # single-row blocs (India, SE Asia) skip the label; region name identifies them
            bshort = {'southeast_asia': 'SE Asia', 'north_america': 'N. America'}.get(
                b, b.replace('_', ' ').title())
            ax.text(barx + ncol * 0.030, start + k / 2 - 0.5, bshort,
                    rotation=90, ha='left', va='center', color=S.BLOC_COLOR[b],
                    fontweight='bold', fontsize=6.5)
        start += k

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig1c_seasonal'))
    save_legend(cmap, norm)
    print('saved', os.path.join(OUT, 'fig1c_seasonal.svg'), '(+ legend)')


def save_legend(cmap, norm):
    """Standalone colorbar (placed by hand), to match the other figures' separate legends."""
    fig = plt.figure(figsize=(60 * S.MM, 16 * S.MM))
    cax = fig.add_axes([0.06, 0.55, 0.88, 0.18])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                      ticks=[0, 25, 50, 75, 100])
    cb.set_label('Firm coverage of DC demand (%)', fontsize=7)
    cb.ax.xaxis.set_label_position('top')
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=7)
    S.save(fig, os.path.join(OUT, 'fig1c_legend'))


if __name__ == '__main__':
    main()
