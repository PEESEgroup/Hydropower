"""Fig 2a - hydro-side climate sensitivity ("supply potential").

%change of hydropower flat-load ELCC ("supply potential") from ssp126 to ssp585, median across
5 GCMs, per region (env REVUB_FIG2A_METRIC switches to supply_C or generation). Diverging RdBu:
decline = red (drier under high warming), increase = blue (wetter). Centred at 0 with TwoSlopeNorm
+/-10% (extremes clipped, colorbar triangles). Dots mark regions where >=4/5 GCMs agree on sign
(robust). The most extreme decline/increase regions are labelled (radial leaders). Bottom-left
inset: per-GCM supply-weighted global change, each with an error bar (s.e. of the mean by default,
or s.d. across regions via REVUB_FIG2A_ERR=sd) so a model is a distribution, not a lone number.
Regions without data are grey. Robinson projection. All text Arial 7 pt; PNG 300 ppi; no title.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

import matplotlib.patheffects as pe
from adjustText import adjust_text
import revub_style as S
from revub_geo import load_regions
from revub_worldview import load_basemap, apply_china_worldview
from fig1a_coverage_map import NE, latlon_bounds, apply_bounds, NOT_ASSESSED, load_coverage
from fig1c_seasonal import _disp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')

MISSING = '#dddddd'   # region modelled elsewhere but no climate-sensitivity value
M = 10.0              # TwoSlopeNorm half-range (%)
AGREE = 4             # >= this many of 5 GCMs same sign -> robust dot
LABEL_THRESH = 5.0    # label every region whose |median %change| >= this

# selectable hydro metric (env REVUB_FIG2A_METRIC): csv, colorbar label, output stem
METRICS = {
    'supply': ('fig2a_hydro_sens.csv', 'Hydro firm supply (supply_C)', 'fig2a_supply'),
    'elcc':   ('fig2a_elcc_sens.csv',  'Hydropower supply potential',  'fig2a_elcc'),
    'gen':    ('fig2a_gen_sens.csv',   'Hydro generation (annual)',    'fig2a_gen'),
}
METRIC = os.environ.get('REVUB_FIG2A_METRIC', 'elcc')   # flat-load ELCC = "supply potential"
CSV, MLABEL, STEM = METRICS[METRIC]


def main():
    sens = pd.read_csv(os.path.join(DATA, CSV))
    sens['pct_med'] = pd.to_numeric(sens['pct_med'], errors='coerce')

    cov_regions = set(load_coverage()['region'])   # only regions actually shown in Fig1a
    g = (apply_china_worldview(load_regions()).merge(sens, on='region', how='left')
         .to_crs(S.EQUAL_EARTH))
    world = load_basemap().to_crs(S.EQUAL_EARTH).reset_index(drop=True)   # China-viewpoint
    has = g[g['pct_med'].notna() & g['region'].isin(cov_regions)].copy()
    miss = g[g['pct_med'].isna()]
    print('regions with sensitivity: %d | min/med/max = %.1f/%.1f/%.1f'
          % (len(has), has['pct_med'].min(), has['pct_med'].median(), has['pct_med'].max()))

    cmap = plt.get_cmap('RdBu')
    norm = TwoSlopeNorm(vmin=-M, vcenter=0.0, vmax=M)

    fig = plt.figure(figsize=(210 * S.MM, 99 * S.MM))   # A4 wide x 1/3 A4 tall
    ax = fig.add_axes([0.0, 0.16, 1.0, 0.82])           # raised to clear the bottom colorbar
    world.plot(ax=ax, color=NOT_ASSESSED, ec='none', zorder=0)
    miss.plot(ax=ax, color=MISSING, ec='0.6', lw=0.15, zorder=1)
    has.plot(ax=ax, column='pct_med', cmap=cmap, norm=norm, ec='0.45', lw=0.2, zorder=2)
    apply_bounds(ax, latlon_bounds(-150, 178, -56, 74, g.crs), 0.0, 0.0)   # -56 shows all of S. America
    ax.set_axis_off()
    ax.set_aspect('equal')

    # robust-agreement dots (>=4/5 GCMs same sign)
    rob = has[has[['n_pos', 'n_neg']].max(axis=1) >= AGREE]
    rpts = rob.representative_point()
    ax.scatter(rpts.x, rpts.y, s=1.4, c='0.12', lw=0, zorder=4)

    # label every region whose |median %change| >= LABEL_THRESH; adjustText repels labels so
    # they don't overlap each other and stay inside the axes, with thin leader lines
    ext = has[has['pct_med'].abs() >= LABEL_THRESH]
    print('labelled (|change|>=%g%%): %d regions' % (LABEL_THRESH, len(ext)))
    halo = [pe.withStroke(linewidth=1.2, foreground='white')]
    texts = []
    for _, r in ext.iterrows():
        p = r.geometry.representative_point()
        texts.append(ax.text(p.x, p.y, '%s %+.0f%%' % (_disp(r['region']), r['pct_med']),
                             fontsize=7, color='0.05', ha='center', va='center',
                             zorder=6, path_effects=halo))
    adjust_text(texts, ax=ax, expand=(1.25, 1.5), force_text=(0.4, 0.6),
                arrowprops=dict(arrowstyle='-', lw=0.4, color='0.5', shrinkA=1, shrinkB=2))

    # colorbar (horizontal, bottom), triangles for clipped extremes
    cax = fig.add_axes([0.52, 0.06, 0.30, 0.024])   # bottom-right, clear of South America
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                      extend='both', ticks=[-10, -5, 0, 5, 10])
    cb.set_label('%s: change SSP1-2.6 → SSP5-8.5 (%%, 5-GCM median)' % MLABEL, fontsize=7)
    cb.ax.xaxis.set_label_position('top')   # label above the bar (it would clip off the bottom)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=7)

    # bottom-left inset: per-GCM bars (supply-weighted global change) WITH an error bar each,
    # so a model is a distribution over regions, not a lone number. Error type via env
    # REVUB_FIG2A_ERR: 'sd' (regional spread, default) or 'se' (precision of the mean).
    short = {'gfdl-esm4': 'GFDL', 'ipsl-cm6a-lr': 'IPSL', 'mpi-esm1-2-hr': 'MPI',
             'mri-esm2-0': 'MRI', 'ukesm1-0-ll': 'UKESM'}
    PGM = pd.read_csv(os.path.join(DATA, 'fig2a_per_gcm_err.csv'))
    PGM['se'] = PGM['wsd'] / np.sqrt(PGM['n'])
    PGM = PGM.sort_values('wmean').reset_index(drop=True)
    errtype = os.environ.get('REVUB_FIG2A_ERR', 'se')   # s.e. of the supply-weighted mean
    yerr = PGM['wsd'] if errtype == 'sd' else PGM['se']
    lbl = '± s.d. across regions' if errtype == 'sd' else '± s.e. of mean'

    axg = fig.add_axes([0.045, 0.15, 0.125, 0.34])   # over the empty South Pacific (no land/labels)
    xs = np.arange(len(PGM))
    axg.bar(xs, PGM['wmean'], yerr=yerr, color=[cmap(norm(v)) for v in PGM['wmean']],
            ec='none', width=0.72, error_kw=dict(lw=0.7, capsize=2, ecolor='0.25'))
    axg.axhline(0, color='0.1', lw=0.5)
    axg.set_xticks(xs)
    axg.set_xticklabels([short.get(g, g) for g in PGM['gcm']], fontsize=6, rotation=90)
    axg.tick_params(width=0.4, length=2, labelsize=7)
    axg.set_ylabel('Δ supply potential\n(%, global)', fontsize=7, labelpad=1)
    axg.set_title('Per-GCM (%s)' % lbl, fontsize=7, pad=2)
    for sp in axg.spines.values():   # no outer frame on bar charts
        sp.set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, STEM))
    print('metric=%s robust(>=%d/5): %d | saved %s.svg(+png)' % (METRIC, AGREE, len(rob), STEM))


if __name__ == '__main__':
    main()
