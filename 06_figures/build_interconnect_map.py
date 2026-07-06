#!/usr/bin/env python3
"""Validate the interconnector catalogue, rebuild corridor edges, and draw maps.

The master table is the only hand-maintained transmission input.  This script
derives ``interconnect_edges.csv`` from existing, inter-region assets so E and
the maps cannot silently drift apart.
"""
import glob
import os
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, 'E_output')
MASTER_CSV = os.path.join(OUT, 'interconnectors_master.csv')
EDGES_CSV = os.path.join(OUT, 'interconnect_edges.csv')
ALL_EDGES_CSV = os.path.join(OUT, 'interconnect_edges_planned.csv')

TRADING_BLOCS = {
    'china': ['china_cc', 'china_csg', 'china_ec', 'china_nc', 'china_ne',
              'china_nw', 'china_sw', 'hong_kong'],
    'india': ['india_east', 'india_north', 'india_northeast', 'india_south', 'india_west'],
    'southeast_asia': ['cambodia', 'indonesia', 'laos', 'malaysia_east',
                       'malaysia_west', 'myanmar', 'philippines', 'thailand',
                       'vietnam', 'singapore'],
    'south_america': ['argentina', 'bolivia', 'chile', 'colombia', 'ecuador',
                      'peru', 'brazil_nordeste', 'brazil_norte',
                      'brazil_sudeste', 'brazil_sul'],
    'europe': ['austria', 'germany', 'hungary', 'poland', 'slovakia', 'slovenia',
               'switzerland', 'turkey', 'latvia', 'lithuania', 'finland',
               'norway', 'sweden', 'albania', 'bulgaria', 'croatia', 'greece',
               'italy', 'north_macedonia', 'portugal', 'romania', 'serbia',
               'spain', 'belgium', 'france', 'ireland', 'luxembourg',
               'united_kingdom', 'denmark', 'netherlands', 'czechia', 'estonia'],
    'north_america': ['usa_caiso', 'usa_ercot', 'usa_isone', 'usa_miso_central',
                      'usa_miso_north', 'usa_miso_south', 'usa_nyiso',
                      'usa_pjm_east', 'usa_pjm_west', 'usa_spp_north',
                      'usa_spp_south', 'usa_sertp', 'usa_frcc',
                      'usa_northerngrid_east', 'usa_northerngrid_south',
                      'usa_northerngrid_west', 'usa_westconnect_north',
                      'usa_westconnect_south', 'canada_atlantic', 'canada_bc',
                      'canada_ontario', 'canada_prairies', 'canada_quebec'],
}

# Catalogue transit countries do not all have a dedicated SWAT study-area
# shapefile. These country-center points are used only to make their corridors
# visible on the diagnostic map; E uses the catalogue graph, not map geometry.
FALLBACK_CENTERS = {
    'uruguay': (-56.0, -32.8),
    'venezuela': (-66.6, 7.0),
    'montenegro': (19.25, 42.75),
}


def find_shp(region):
    for suffix in ('_study_area_admin.shp', '_study_area_target.shp', '_study_area_ba.shp'):
        for root in ('/datasets/swat_global', '/home/cfeng/myswat/swat_global'):
            matches = glob.glob(f'{root}/*/{region}/{suffix}')
            if matches:
                return matches[0]
    for root in ('/datasets/swat_global', '/home/cfeng/myswat/swat_global'):
        matches = glob.glob(f'{root}/*/{region}/vectors/study_area.shp')
        if matches:
            return matches[0]
    return None


def edge_color(types):
    return '#C62828' if types == 'HVDC' else ('#1565C0' if types == 'AC' else '#6A1B9A')


def validate_master(master):
    required = {'bloc', 'region_a', 'region_b', 'line_name', 'capacity_mw', 'type',
                'status', 'source_url'}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f'Master table is missing columns: {sorted(missing)}')
    if master['line_name'].duplicated().any():
        names = master.loc[master['line_name'].duplicated(False), 'line_name'].tolist()
        raise ValueError(f'Duplicate line_name values: {names}')
    if master['source_url'].fillna('').str.strip().eq('').any():
        raise ValueError('Every master-table row must have a source_url')
    if (~master['type'].isin(['AC', 'HVDC', 'MIXED'])).any():
        raise ValueError('type must be AC, HVDC, or MIXED')
    if (~master['status'].isin(['existing', 'under_construction', 'planned', 'inactive'])).any():
        raise ValueError('Unexpected status in master table')
    existing = master[master['status'].eq('existing')]
    bad_existing = existing[existing['capacity_mw'].isna() | (existing['capacity_mw'] <= 0)]
    if len(bad_existing):
        raise ValueError('Existing lines require positive capacity:\n' +
                         bad_existing[['line_name', 'capacity_mw']].to_string(index=False))


def rebuild_edges(master):
    rows = master[
        master['status'].eq('existing')
        & master['capacity_mw'].notna()
        & master['capacity_mw'].gt(0)
        & master['region_a'].ne(master['region_b'])
    ].copy()
    if 'intra_region' in rows:
        intra = rows['intra_region'].astype(str).str.lower().isin(['true', '1', 'yes'])
        rows = rows[~intra]
    pairs = np.sort(rows[['region_a', 'region_b']].astype(str).values, axis=1)
    rows['region_a'] = pairs[:, 0]
    rows['region_b'] = pairs[:, 1]
    rows['ac_mw'] = np.where(rows['type'].eq('AC'), rows['capacity_mw'],
                             np.where(rows['type'].eq('MIXED'), rows['capacity_mw'] / 2, 0.0))
    rows['hvdc_mw'] = np.where(rows['type'].eq('HVDC'), rows['capacity_mw'],
                               np.where(rows['type'].eq('MIXED'), rows['capacity_mw'] / 2, 0.0))
    edges = rows.groupby(['bloc', 'region_a', 'region_b'], as_index=False).agg(
        total_mw=('capacity_mw', 'sum'),
        ac_mw=('ac_mw', 'sum'),
        hvdc_mw=('hvdc_mw', 'sum'),
        types=('type', lambda s: '+'.join(sorted(set(s)))),
        source_count=('line_name', 'size'),
    )
    edges = edges.sort_values(['bloc', 'region_a', 'region_b']).reset_index(drop=True)
    edges.to_csv(EDGES_CSV, index=False)
    all_rows = master[master['region_a'].ne(master['region_b'])].copy()
    if 'intra_region' in all_rows:
        intra = all_rows['intra_region'].astype(str).str.lower().isin(['true', '1', 'yes'])
        all_rows = all_rows[~intra]
    all_pairs = np.sort(all_rows[['region_a', 'region_b']].astype(str).values, axis=1)
    all_rows['region_a'] = all_pairs[:, 0]
    all_rows['region_b'] = all_pairs[:, 1]
    all_rows['existing_mw'] = np.where(
        all_rows['status'].eq('existing'), all_rows['capacity_mw'].fillna(0.0), 0.0)
    all_rows['future_mw'] = np.where(
        all_rows['status'].isin(['under_construction', 'planned']),
        all_rows['capacity_mw'].fillna(0.0), 0.0)
    all_rows['inactive_mw'] = np.where(
        all_rows['status'].eq('inactive'), all_rows['capacity_mw'].fillna(0.0), 0.0)
    all_rows['unknown_future'] = (
        all_rows['status'].isin(['under_construction', 'planned'])
        & all_rows['capacity_mw'].isna())
    all_edges = all_rows.groupby(['bloc', 'region_a', 'region_b'], as_index=False).agg(
        n_lines=('line_name', 'size'),
        existing_mw=('existing_mw', 'sum'),
        future_mw=('future_mw', 'sum'),
        inactive_mw=('inactive_mw', 'sum'),
        unknown_future_lines=('unknown_future', 'sum'),
        types=('type', lambda s: '+'.join(sorted(set(s)))),
    )
    all_edges['total_known_mw'] = all_edges['existing_mw'] + all_edges['future_mw']
    all_edges.sort_values(['bloc', 'region_a', 'region_b']).to_csv(ALL_EDGES_CSV, index=False)
    return edges


def draw_bloc(bloc, master):
    bloc_name = bloc.upper()
    bm = master[master['bloc'].eq(bloc_name) & master['region_a'].ne(master['region_b'])].copy()
    if len(bm) == 0:
        print(f'{bloc}: no catalogue rows, skip')
        return
    bm['cap'] = pd.to_numeric(bm['capacity_mw'], errors='coerce')
    bm['pair'] = bm.apply(lambda r: tuple(sorted((r.region_a, r.region_b))), axis=1)
    bm['future'] = bm['status'].isin(['under_construction', 'planned'])

    # Include catalogue-only transit nodes (for example Uruguay) as well as demand regions.
    regions = sorted(set(TRADING_BLOCS[bloc]) | set(bm['region_a']) | set(bm['region_b']))
    geoms = {}
    for region in regions:
        shp = find_shp(region)
        if not shp:
            continue
        try:
            geo = gpd.read_file(shp).to_crs('EPSG:4326')
            geoms[region] = (geo.geometry.union_all() if hasattr(geo.geometry, 'union_all')
                             else geo.geometry.unary_union)
        except Exception as exc:
            print(f'  [map warning] {region}: {exc}')
    if len(geoms) < 2:
        print(f'{bloc}: fewer than two geometries, skip')
        return

    gdf = gpd.GeoDataFrame({'region': list(geoms)}, geometry=list(geoms.values()), crs='EPSG:4326')
    gdf['geometry'] = gdf.geometry.simplify(0.05)
    centers = {row.region: (row.geometry.centroid.x, row.geometry.centroid.y)
               for _, row in gdf.iterrows()}
    fallback_nodes = sorted((set(regions) - set(centers)) & set(FALLBACK_CENTERS))
    centers.update({region: FALLBACK_CENTERS[region] for region in fallback_nodes})
    existing = bm[bm['status'].eq('existing')].groupby('pair').agg(
        mw=('cap', 'sum'), types=('type', lambda s: '+'.join(sorted(set(s))))).reset_index()
    future = bm[bm['future']].groupby('pair').agg(
        mw=('cap', 'sum'), types=('type', lambda s: '+'.join(sorted(set(s)))),
        under_construction=('status', lambda s: 'under_construction' in set(s)),
        unknown_capacity=('cap', lambda s: bool(s.isna().any()))).reset_index()
    inactive = bm[bm['status'].eq('inactive')].groupby('pair').agg(
        mw=('cap', 'sum'), types=('type', lambda s: '+'.join(sorted(set(s))))).reset_index()
    known = bm['cap'].dropna()
    max_mw = max(float(known.max()) if len(known) else 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(13, 10))
    gdf.plot(ax=ax, facecolor='#EFEFEF', edgecolor='#AAAAAA', linewidth=0.4)
    plotted_existing = 0
    plotted_future = 0
    plotted_inactive = 0
    for _, edge in existing.iterrows():
        ra, rb = edge['pair']
        if ra not in centers or rb not in centers:
            continue
        width = 0.8 + 4.0 * np.log1p(edge.mw) / np.log1p(max_mw)
        ax.plot([centers[ra][0], centers[rb][0]], [centers[ra][1], centers[rb][1]],
                color=edge_color(edge.types), lw=width, alpha=0.75, zorder=2,
                solid_capstyle='round')
        plotted_existing += 1
    for _, edge in future.iterrows():
        ra, rb = edge['pair']
        if ra not in centers or rb not in centers:
            continue
        width = (1.2 if edge.unknown_capacity else
                 0.8 + 4.0 * np.log1p(edge.mw) / np.log1p(max_mw))
        color = '#EF6C00' if edge.under_construction else '#777777'
        ax.plot([centers[ra][0], centers[rb][0]], [centers[ra][1], centers[rb][1]],
                color=color, lw=width, ls=(0, (5, 3)), alpha=0.9, zorder=3)
        if edge.unknown_capacity:
            mid_x = (centers[ra][0] + centers[rb][0]) / 2
            mid_y = (centers[ra][1] + centers[rb][1]) / 2
            ax.annotate('capacity unknown', (mid_x, mid_y), fontsize=6, color=color)
        plotted_future += 1
    for _, edge in inactive.iterrows():
        ra, rb = edge['pair']
        if ra not in centers or rb not in centers:
            continue
        ax.plot([centers[ra][0], centers[rb][0]], [centers[ra][1], centers[rb][1]],
                color='#222222', lw=1.0, ls=':', alpha=0.8, zorder=2)
        plotted_inactive += 1
    for region, (x, y) in centers.items():
        ax.plot(x, y, 'o', color='black', ms=4, zorder=4)
        label = region.replace(f'{bloc}_', '').replace('usa_', '').replace('canada_', 'ca_')
        ax.annotate(label, (x, y), fontsize=7, zorder=5)
    missing_geoms = sorted(set(regions) - set(centers))
    title_extra = f' | missing map geometry: {len(missing_geoms)}' if missing_geoms else ''
    ax.set_title(
        f'{bloc}: transmission interconnections\n'
        f'SOLID existing (red HVDC / blue AC / purple mixed); '
        f'DASHED future (orange construction / grey planned); DOTTED inactive\n'
        f'{plotted_existing} existing + {plotted_future} future + '
        f'{plotted_inactive} inactive corridors plotted{title_extra}',
        fontsize=10)
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    plt.tight_layout()
    png = os.path.join(OUT, f'interconnect_{bloc}.png')
    plt.savefig(png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'{bloc}: {plotted_existing} existing + {plotted_future} future + '
          f'{plotted_inactive} inactive -> {png}')
    if missing_geoms:
        print(f'  no geometry: {", ".join(missing_geoms)}')
    if fallback_nodes:
        print(f'  point fallback used: {", ".join(fallback_nodes)}')


def main():
    master = pd.read_csv(MASTER_CSV)
    master['capacity_mw'] = pd.to_numeric(master['capacity_mw'], errors='coerce')
    validate_master(master)
    edges = rebuild_edges(master)
    unknown = master[master['capacity_mw'].isna()]
    print(f'Validated {len(master)} source rows; rebuilt {len(edges)} existing corridors')
    if len(unknown):
        print('Future assets with unknown capacity (excluded from E capacity):')
        print(unknown[['bloc', 'line_name', 'status', 'expected_year']].to_string(index=False))
    for bloc in TRADING_BLOCS:
        draw_bloc(bloc, master)
    print('Done:', EDGES_CSV, 'and', ALL_EDGES_CSV)


if __name__ == '__main__':
    main()
