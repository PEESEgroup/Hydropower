"""Fig 5 indicator panel: the headline conservation metric, drawn as a centred diverging
"tug-of-war" of THREATENED freshwater-fish species (IUCN CR/EN/VU) whose habitat is exposed
to the DC flow-regime change. Per baseline (DC-CONV, DC-BAL): green (left) = species whose
basins see ONLY beneficial change, pink (right) = ONLY harmful, grey (centre) = both.
Colours match the maps (PiYG). Run: REVUB_FIG_DATA=$PWD/data_nofuture_gfdl /opt/conda/bin/python fig5_metric.py [reversals|drawdown]
"""
import os, sys
import pandas as pd
import geopandas as gpd
import revub_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data_nofuture_gfdl')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
IUCN_LONG = os.path.join(HERE, 'data_external', 'iucn', 'threatened_long_hybas08.csv')
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'reversals'
GREEN, PINK, BOTH = '#4d9221', '#c51b7d', '#bdbdbd'
SHORT = {'reversals': 'flow reversals', 'drawdown': 'lethal drawdown'}[METRIC]
ROWS = [('%s' % METRIC, 'DC − CONV', 'vs conventional'),
        ('%s_bal' % METRIC, 'DC − BAL', 'vs typical')]


def sets(stem, sp_by_basin):
    d = gpd.read_file(os.path.join(DATA, 'reach_cons_%s.gpkg' % stem), ignore_geometry=True)
    d = d[d.eff != 0].copy(); d['hybas08'] = d.hybas08.astype(str)
    def sp(b):
        s = set()
        for hb in b: s |= sp_by_basin.get(hb, set())
        return s
    B = sp(set(d[d.eff < 0].hybas08)); H = sp(set(d[d.eff > 0].hybas08))
    return B, H


def main():
    long = pd.read_csv(IUCN_LONG, dtype={'hybas_id': str})
    sp_by_basin = long.groupby('hybas_id').sci_name.apply(set).to_dict()

    fig, ax = S.fig_mm(72, 74.25)                                # tall panel (fig4_typ height); no on-figure title
    fig.subplots_adjust(left=0.22, right=0.84, top=0.95, bottom=0.20)
    xloc = {0: 0.0, 1: 1.0}; w = 0.52
    top, bot = 0, 0
    for i, (stem, lab, sub) in enumerate(ROWS):
        B, H = sets(stem, sp_by_basin)
        ob, oh, bo = len(B - H), len(H - B), len(B & H)
        x = xloc[i]
        ax.bar(x, bo, bottom=-bo / 2, width=w, color=BOTH, zorder=2)
        ax.bar(x, oh, bottom=bo / 2, width=w, color=PINK, zorder=2)              # harmful up
        ax.bar(x, ob, bottom=-bo / 2 - ob, width=w, color=GREEN, zorder=2)       # beneficial down
        ax.text(x, bo / 2 + oh + 6, '%d' % oh, color=PINK, fontsize=7.5, ha='center', va='bottom', weight='bold')
        ax.text(x, -bo / 2 - ob - 6, '%d' % ob, color=GREEN, fontsize=7.5, ha='center', va='top', weight='bold')
        if bo:
            ax.text(x, 0, '%d' % bo, color='0.2', fontsize=5.5, ha='center', va='center')
        top = max(top, bo / 2 + oh); bot = max(bot, bo / 2 + ob)
    ax.axhline(0, color='0.3', lw=0.6, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['data-centre vs\nconventional',
                        'data-centre vs\nbalancing grid'], fontsize=5.5)
    ax.set_xlim(-0.62, 2.05); ax.set_ylim(-(bot + 40), top + 40)
    for sp_ in ('top', 'right', 'bottom'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_yticks([t for t in ax.get_yticks() if -(bot + 40) <= t <= top + 40])
    ax.set_yticklabels(['%d' % abs(t) for t in ax.get_yticks()])
    ax.set_ylabel('threatened freshwater-fish species (IUCN CR/EN/VU)', fontsize=6.5)
    ax.annotate('only\nharmful', (1.5, top * 0.55), fontsize=6, color=PINK, ha='left', va='center', weight='bold')
    ax.annotate('both', (1.5, 0), fontsize=6, color='0.45', ha='left', va='center')
    ax.annotate('only\nbeneficial', (1.5, -bot * 0.55), fontsize=6, color=GREEN, ha='left', va='center', weight='bold')
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, 'fig5_metric_%s' % METRIC))
    print('saved fig5_metric_%s' % METRIC)


if __name__ == '__main__':
    main()
