"""Fig 1b - NESTED Voronoi capacity treemaps: DC demand & hydropower firm capacity.

Two-level additively-weighted Voronoi (custom; no Python lib exists):
  level 1 = interconnection bloc (Europe / SE Asia / Latin America / N.America get a parent
            cell that is then subdivided; China / India are single-country cells),
  level 2 = countries within each bloc cell.
Cells are coloured by World-Bank income group (developed / emerging / developing) so the
development divide reads at a glance. Country units (regions aggregated to country via NE).
Small countries within a bloc are merged into 'Other <bloc>'. All text Arial 7 pt.
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPoly
from matplotlib.colors import to_rgb

import revub_style as S
from revub_geo import load_regions, US_STATE_LABEL
from revub_worldview import load_basemap
from revub_names import region_label, region_label_us
from voronoi_treemap import voronoi_treemap, circle_boundary, poly_area, poly_centroid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
OUT = os.environ.get('REVUB_FIG_OUT') or os.path.join(HERE, 'out')
NE = os.path.join(HERE, 'gis', 'ne_110m_admin_0_countries.shp')
THR = 0.0004         # sub-region drawn individually if share>=0.04%; else merged to 'Other'

INCOME_COL = {'developed': '#2171b5', 'emerging': '#41ab5d', 'developing': '#e6550d'}
OTHER_COL = '#cfcfcf'
# each income group = one colour FAMILY; sub-regions within get distinct shades
FAMILY = {'developed': 'Blues', 'emerging': 'Greens', 'developing': 'Oranges', 'other': 'Greys'}


def _shades(items):
    """Per cell: a distinct shade within its income group's colour family (darkest=largest)."""
    groups = {}
    for i, (name, inc, v) in enumerate(items):
        groups.setdefault(inc, []).append((v, i))
    colors = [None] * len(items)
    for inc, lst in groups.items():
        cm = plt.get_cmap(FAMILY.get(inc, 'Greys'))
        lst.sort(key=lambda t: -t[0])
        pos = np.linspace(0.80, 0.42, len(lst)) if len(lst) > 1 else [0.62]
        for p, (v, i) in zip(pos, lst):
            colors[i] = cm(float(p))
    return colors
CC_RENAME = {'United States of America': 'USA', 'United Kingdom': 'UK',
             'Bosnia and Herzegovina': 'Bosnia', 'Republic of Serbia': 'Serbia',
             'North Macedonia': 'N.Macedonia', 'Dominican Republic': 'Dominican Rep.'}
ACR = {'ercot': 'ERCOT', 'caiso': 'CAISO', 'frcc': 'FRCC', 'spp': 'SPP', 'miso': 'MISO',
       'pjm': 'PJM', 'sertp': 'SERTP', 'csg': 'CSG', 'nyiso': 'NYISO', 'isone': 'ISO-NE',
       'northerngrid': 'NorthernGrid', 'westconnect': 'WestConnect',
       'ec': 'East', 'nc': 'North', 'sw': 'Southwest', 'cc': 'Central', 'ne': 'Northeast',
       'nw': 'Northwest',
       'sudeste': 'Southeast', 'nordeste': 'Northeast', 'sul': 'South', 'norte': 'North'}


def _disp_name(r):
    """Canonical no-acronym geographic label (shared via revub_names)."""
    return region_label(r)


def _region_table():
    """Per SWAT region (sub-national for big countries): bloc, income, DC demand, hydro firm."""
    g = load_regions()[['region', 'bloc']].copy()
    inc = pd.read_csv(os.path.join(DATA, 'equity_master.csv'))[['region', 'income_group']]
    fr = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, 'cov_*.csv')))]
    dc = pd.concat(fr, ignore_index=True)
    dc = dc[dc['scenario'] == 'A'].drop_duplicates('region')[['region', 'demand_mw']]
    # hydro = FLAT-LOAD ELCC (program C cascade-opt, bal-ELCC fallback) - NOT D's DC-capped supply_C
    fe = pd.read_csv(os.path.join(DATA, 'hydro_flat_elcc.csv'))[['region', 'flat_elcc_mw']]
    m = (g.merge(inc, on='region', how='left')
         .merge(dc, on='region', how='left').merge(fe, on='region', how='left'))
    mode = m.groupby('bloc')['income_group'].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else 'developing')
    m['income'] = m['income_group'].fillna(mode).fillna('developing')
    m['country'] = m['region'].map(region_label_us)   # US subregions tagged ' (US)' for the labelled scatter
    return m[['country', 'bloc', 'income', 'demand_mw', 'flat_elcc_mw']]


def _txt_color(hexc):
    r, g, b = to_rgb(hexc)
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.58 else 'black'


def _country_table():
    g = load_regions()
    world = load_basemap()[['NAME', 'geometry']]   # China-viewpoint country attribution
    reps = gpd.GeoDataFrame(g[['region', 'bloc']],
                            geometry=g.representative_point(), crs=g.crs)
    j = gpd.sjoin(reps, world, predicate='within', how='left')
    j = j[~j.index.duplicated(keep='first')]
    j['country'] = [CC_RENAME.get(n, n) if isinstance(n, str) else r.replace('_', ' ').title()
                    for n, r in zip(j['NAME'], j['region'])]

    fr = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, 'cov_*.csv')))]
    dc = pd.concat(fr, ignore_index=True)
    dc = dc[dc['scenario'] == 'A'].drop_duplicates('region')[['region', 'demand_mw']]
    hy = pd.read_csv(os.path.join(DATA, 'hydro_cap.csv'))[['region', 'supply_C_mw']]
    inc = pd.read_csv(os.path.join(DATA, 'equity_master.csv'))[['region', 'income_group']]

    m = (j[['region', 'bloc', 'country']]
         .merge(dc, on='region', how='left')
         .merge(hy, on='region', how='left')
         .merge(inc, on='region', how='left'))
    agg = m.groupby('country').agg(
        bloc=('bloc', lambda s: s.mode().iloc[0]),
        income=('income_group', lambda s: s.dropna().mode().iloc[0] if s.notna().any() else 'developing'),
        demand_mw=('demand_mw', 'sum'),
        supply_C_mw=('supply_C_mw', 'sum')).reset_index()
    return agg


def _rows(tbl, col):
    return [(c, b, inc, v) for c, b, inc, v in
            zip(tbl['country'], tbl['bloc'], tbl['income'], tbl[col]) if v and v > 0]


def _collapse_by_bloc(rows, total):
    """Per bloc: keep countries with share>=THR; merge the rest into 'Other <bloc>'."""
    out = {}
    for c, b, inc, v in rows:
        out.setdefault(b, {'big': [], 'small': 0.0})
        if v / total >= THR:
            out[b]['big'].append((c, inc, v))
        else:
            out[b]['small'] += v
    bloc_cells = {}
    for b, d in out.items():
        items = sorted(d['big'], key=lambda t: -t[2])
        if d['small'] > 0:
            items.append((f'Other {b.replace("_", " ").title()}', 'other', d['small']))
        bloc_cells[b] = items
    return bloc_cells


def draw_panel(ax, rows, title):
    """Flat single-circle Voronoi over all country cells (small ones merged per bloc)."""
    total = sum(v for *_, v in rows)
    bloc_cells = _collapse_by_bloc(rows, total)
    items = []
    for b in S.BLOC_ORDER:
        items += bloc_cells.get(b, [])
    vals = np.array([v for *_, v in items])
    bnd = circle_boundary(R=1.0, nv=120)
    polys, _, _, _, _ = voronoi_treemap(vals, bnd, iters=240, gain=1.0, tol=0.02)

    small_lbls = []
    for (name, inc, v), poly in zip(items, polys):
        if len(poly) < 3:
            continue
        col = OTHER_COL if inc == 'other' else INCOME_COL.get(inc, '#cccccc')
        ax.add_patch(MplPoly(poly, closed=True, facecolor=col, ec='white', lw=0.6, zorder=2))
        cen = poly_centroid(poly)
        rcell = np.sqrt(poly_area(poly) / np.pi)
        lab = f'{name}\n{100 * v / total:.1f}%'
        if rcell > 0.115:
            ax.text(cen[0], cen[1], lab, ha='center', va='center',
                    color=_txt_color(col), linespacing=0.9, zorder=4)
        else:
            small_lbls.append((cen, lab))

    small_lbls.sort(key=lambda s: np.arctan2(s[0][1], s[0][0]))
    for cen, lab in small_lbls:
        ang = np.arctan2(cen[1], cen[0])
        lx, ly = 1.16 * np.cos(ang), 1.16 * np.sin(ang)
        ha = 'left' if lx >= 0 else 'right'
        ax.plot([cen[0], lx], [cen[1], ly], color='0.55', lw=0.3, zorder=1)
        ax.text(lx, ly, lab, ha=ha, va='center', linespacing=0.9, zorder=4)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.28, 1.28)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)


def _label_cells(ax, items, polys, total, scale=1.0, colors=None):
    """Draw + label every cell; in-cell if big enough else collect for leader lines."""
    small = []
    for i, ((name, inc, v), cp) in enumerate(zip(items, polys)):
        if len(cp) < 3:
            continue
        col = colors[i] if colors is not None else \
            (OTHER_COL if inc == 'other' else INCOME_COL.get(inc, '#cccccc'))
        ax.add_patch(MplPoly(cp, closed=True, facecolor=col, ec='white', lw=0.5, zorder=2))
        cen = poly_centroid(cp)
        rcell = np.sqrt(poly_area(cp) / np.pi)
        lab = f'{name}\n{100 * v / total:.1f}%'
        if rcell > 0.16 * scale:
            ax.text(cen[0], cen[1], lab, ha='center', va='center', fontsize=6,
                    color=_txt_color(col), linespacing=0.85, zorder=4)
        else:
            small.append((cen, lab))
    return small


def draw_drilldown(rows, title, outname):
    """Big overview (6 bloc cells) + 6 small per-bloc circles (countries), leader-connected."""
    total = sum(v for *_, v in rows)
    bloc_cells = _collapse_by_bloc(rows, total)
    blocs = [b for b in S.BLOC_ORDER if b in bloc_cells]
    bloc_tot = np.array([sum(v for *_, v in bloc_cells[b]) for b in blocs])

    R_big, Dx, Dy = 1.0, 4.1, 2.55   # elliptical ring fills the 105x74 (landscape) panel
    big = circle_boundary(R=R_big, nv=120)
    bpolys, _, _, _, _ = voronoi_treemap(bloc_tot, big, iters=200, gain=1.0, tol=0.02)
    ncount = {b: len(bloc_cells[b]) for b in blocs}
    nmax = max(ncount.values())

    fig = plt.figure(figsize=(105 * S.MM, 74.25 * S.MM))   # 1/2 A4 wide x 1/4 A4 tall
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect('equal')
    ax.axis('off')

    # central overview: one cell per bloc, coloured by bloc, labelled with region name + share
    for b, bp in zip(blocs, bpolys):
        ax.add_patch(MplPoly(bp, closed=True, facecolor=S.BLOC_COLOR[b], ec='white', lw=0.9, zorder=2))
        cen = poly_centroid(bp)
        ax.text(cen[0], cen[1], f'{b.replace("_", " ").title()}\n{100 * bloc_tot[blocs.index(b)] / total:.0f}%',
                ha='center', va='center', color=_txt_color(S.BLOC_COLOR[b]), fontsize=6,
                fontweight='bold', linespacing=0.9, zorder=4)

    # evenly-spaced angular slots (ordered by bloc cell angle, no overlap), then rotate the
    # whole ring by the circular-mean offset so each small circle lines up with its bloc cell
    n = len(blocs)
    cell_ang = sorted([(np.arctan2(poly_centroid(bp)[1], poly_centroid(bp)[0]), b)
                       for b, bp in zip(blocs, bpolys)])
    even = np.array([2 * np.pi * k / n for k in range(n)])
    d = np.array([a for a, _ in cell_ang]) - even
    off = np.arctan2(np.sin(d).mean(), np.cos(d).mean())
    slot = {b: even[k] + off for k, (_, b) in enumerate(cell_ang)}

    for b, bp in zip(blocs, bpolys):
        r_s = 0.5 + 0.62 * np.sqrt(ncount[b] / nmax)
        cen = poly_centroid(bp)
        sa = slot[b]
        sc = np.array([Dx * np.cos(sa), Dy * np.sin(sa)])
        u = sc / np.hypot(sc[0], sc[1])
        edge = sc - r_s * u
        ax.plot([cen[0], edge[0]], [cen[1], edge[1]], color=S.BLOC_COLOR[b], lw=0.8, zorder=1)
        items = bloc_cells[b]
        cvals = np.array([v for *_, v in items])
        sb = circle_boundary(R=r_s, cx=sc[0], cy=sc[1], nv=96)
        cpolys = [sb] if len(items) == 1 else \
            voronoi_treemap(cvals, sb, iters=200, gain=1.0, tol=0.03)[0]
        small = _label_cells(ax, items, cpolys, total, scale=r_s, colors=_shades(items))
        for c2, lab in small:                       # leader labels inside/near the small circle
            a2 = np.arctan2(c2[1] - sc[1], c2[0] - sc[0])
            lx, ly = sc[0] + 1.12 * r_s * np.cos(a2), sc[1] + 1.12 * r_s * np.sin(a2)
            ha = 'left' if lx >= sc[0] else 'right'
            ax.plot([c2[0], lx], [c2[1], ly], color='0.6', lw=0.25, zorder=1)
            ax.text(lx, ly, lab, ha=ha, va='center', fontsize=6, linespacing=0.85, zorder=4)
        ax.add_patch(mpatches.Circle(sc, r_s, fill=False, ec=S.BLOC_COLOR[b], lw=1.0, zorder=3))

    ax.set_xlim(-(Dx + 1.25), Dx + 1.25)
    ax.set_ylim(-(Dy + 1.25), Dy + 1.25)
    # income-group legend rendered separately by fig_legends.py -> out/fig1b_legend.*
    os.makedirs(OUT, exist_ok=True)
    S.save(fig, os.path.join(OUT, outname))
    print('saved', os.path.join(OUT, outname + '.svg'), '(+.png)')


def main():
    tbl = _region_table()
    draw_drilldown(_rows(tbl, 'demand_mw'),
                   'Data-centre demand capacity (2040 peak)', 'fig1b_dc_drill')
    draw_drilldown(_rows(tbl, 'flat_elcc_mw'),
                   'Hydropower firm capacity, flat-load ELCC (mri-esm2-0/ssp370)', 'fig1b_hydro_drill')


if __name__ == '__main__':
    main()
