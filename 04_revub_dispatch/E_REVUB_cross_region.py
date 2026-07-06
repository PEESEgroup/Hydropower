#!/usr/bin/env python3
"""Cross-region hydro-to-DC dispatch based on D's three dispatch scenarios.

For each D scenario this program reports:
  E0: isolated dispatch within each SWAT region;
  E1: local-first dispatch followed by capacity-constrained inter-region trade.

D scenario A = independent operation without pumped storage.
D scenario B = matched operation including pumped storage.
D scenario C = cascade-coordinated operation.

The cross-region allocator is greedy Best-Fit Decreasing (BFD): DC loads are
processed from largest to smallest and each remaining load is assigned to the
source/path whose deliverable capacity is the tightest fit. Corridor available
transfer capacity and transmission losses are enforced on every path.
"""
import heapq
import hashlib
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
BLOC = os.environ.get('REVUB_BLOC', 'china').strip().lower()
GCM = os.environ.get('REVUB_GCM', 'mri-esm2-0').strip()
SSP = os.environ.get('REVUB_SSP', 'ssp370').strip()
SCENARIO_TAG = os.environ.get('REVUB_SCENARIO_TAG', 'flat').strip()
DEMAND_YEAR = int(os.environ.get('REVUB_DC_YEAR', '2033'))
# When set, include every catalogued interconnector regardless of commissioning
# year/status (existing + planned + under_construction + undated). Use to study an
# unconstrained "all corridors built" upper bound; default keeps the year-vintage filter.
INCLUDE_ALL_LINES = os.environ.get('REVUB_E_ALL_LINES', '0') == '1'
DEMAND_SCALE = float(os.environ.get('REVUB_E_DEMAND_SCALE', '1.0'))
ALLOW_LEGACY = os.environ.get('REVUB_E_ALLOW_LEGACY', '0') == '1'
REDISPATCH = os.environ.get('REVUB_E_REDISPATCH', '1') == '1'
REDISPATCH_MAX_ITER = int(os.environ.get('REVUB_E_MAX_ITER', '8'))
REDISPATCH_TOL_MW = float(os.environ.get('REVUB_E_TOL_MW', '0.1'))
REDISPATCH_WORKERS = int(os.environ.get('REVUB_E_WORKERS', '2'))

SWAT_ROOT = os.environ.get('REVUB_SWAT_ROOT', '/datasets/swat_global')
DC_MAPPING = os.environ.get(
    'REVUB_DC_MAPPING', '/home/cfeng/hydro/dc_load_simulation/dc_to_swat_region.csv')
DC_RESULTS = os.environ.get(
    'REVUB_DC_RESULTS', '/home/cfeng/hydro/dc_load_simulation/results')
EDGES_CSV = os.path.join(CODE, 'E_output', 'interconnect_edges.csv')
MASTER_CSV = os.path.join(CODE, 'E_output', 'interconnectors_master.csv')

# These are explicit model assumptions because the source catalogue contains
# nameplate capacity but no reservation schedule or line length. They can be
# replaced per run without editing code.
AC_ATC_FACTOR = float(os.environ.get('REVUB_E_AC_ATC_FACTOR', '0.80'))
HVDC_ATC_FACTOR = float(os.environ.get('REVUB_E_HVDC_ATC_FACTOR', '0.90'))
AC_LOSS_RATE = float(os.environ.get('REVUB_E_AC_LOSS_RATE', '0.03'))
HVDC_LOSS_RATE = float(os.environ.get('REVUB_E_HVDC_LOSS_RATE', '0.02'))
EPS = 1e-6
REGIONAL_MODEL_FILES = (
    'A_REVUB_initialise_v2.py',
    'B_REVUB_main_code_v2.py',
    'C_REVUB_PS_dispatch.py',
    'D_REVUB_DC_dispatch.py',
    'E_REVUB_region_dispatch.py',
)

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

SCENARIOS = {
    'A': ('supply_A_mw', 'independent_no_ps'),
    'B': ('supply_B_mw', 'matched_with_ps'),
    'C': ('supply_C_mw', 'cascade_coordinated'),
}


def validate_settings():
    if BLOC not in TRADING_BLOCS:
        raise ValueError(f'Unknown REVUB_BLOC={BLOC!r}; choose from {sorted(TRADING_BLOCS)}')
    for name, value in [('AC_ATC_FACTOR', AC_ATC_FACTOR),
                        ('HVDC_ATC_FACTOR', HVDC_ATC_FACTOR)]:
        if not 0 < value <= 1:
            raise ValueError(f'{name} must be in (0, 1], got {value}')
    for name, value in [('AC_LOSS_RATE', AC_LOSS_RATE),
                        ('HVDC_LOSS_RATE', HVDC_LOSS_RATE)]:
        if not 0 <= value < 1:
            raise ValueError(f'{name} must be in [0, 1), got {value}')
    if DEMAND_SCALE <= 0:
        raise ValueError('REVUB_E_DEMAND_SCALE must be positive')
    if REDISPATCH_MAX_ITER < 1:
        raise ValueError('REVUB_E_MAX_ITER must be at least 1')
    if REDISPATCH_TOL_MW <= 0:
        raise ValueError('REVUB_E_TOL_MW must be positive')
    if REDISPATCH_WORKERS < 1:
        raise ValueError('REVUB_E_WORKERS must be at least 1')


def region_dir(region):
    if not os.path.isdir(SWAT_ROOT):
        return None
    # A region name can exist under more than one continent directory (e.g. an
    # empty placeholder beside the real one - turkey lives in other/ with a stub
    # in europe_central/). Prefer the directory that actually has the hydro
    # input dataset; fall back to the first match for data-less DC regions.
    matches = []
    for continent in sorted(os.listdir(SWAT_ROOT)):
        path = os.path.join(SWAT_ROOT, continent, region)
        if os.path.isdir(path):
            matches.append(path)
    for path in matches:
        if os.path.exists(os.path.join(path, 'datasets', 'ml_dataset_hourly.parquet')):
            return path
    return matches[0] if matches else None


def d_output_file(region, filename):
    directory = region_dir(region)
    if directory is None:
        return None
    base = os.path.join(directory, 'revub_output', GCM, SSP)
    tagged = os.path.join(base, SCENARIO_TAG, filename)
    if os.path.exists(tagged):
        return tagged
    legacy = os.path.join(base, filename)
    if ALLOW_LEGACY and os.path.exists(legacy):
        print(f'  [legacy D output] {region}: {legacy}')
        return legacy
    return None


def load_d_scenario_capacity(regions):
    capacities = {code: {} for code in SCENARIOS}
    missing = []
    loaded = []
    for region in regions:
        summary = d_output_file(region, 'dc_hydro_summary.csv')
        if summary is None:
            missing.append(region)
            continue
        frame = pd.read_csv(summary)
        if len(frame) != 1:
            raise ValueError(f'{summary} must contain exactly one region row')
        row = frame.iloc[0]
        for column, expected in [('region', region), ('gcm', GCM), ('ssp', SSP)]:
            if column in frame.columns and str(row[column]).strip() != expected:
                raise ValueError(
                    f'{summary} has {column}={row[column]!r}; expected {expected!r}')
        absent = [column for column, _ in SCENARIOS.values() if column not in frame.columns]
        if absent:
            raise ValueError(f'{summary} is missing D scenario fields: {absent}')
        for code, (column, _) in SCENARIOS.items():
            value = pd.to_numeric(row[column], errors='coerce')
            capacities[code][region] = float(value) if np.isfinite(value) and value > 0 else 0.0
        loaded.append(region)
    if not loaded and not REDISPATCH:
        suffix = ' (legacy fallback is disabled)' if not ALLOW_LEGACY else ''
        raise RuntimeError(
            f'No D dc_hydro_summary.csv files found for {BLOC}/{GCM}/{SSP}/'
            f'{SCENARIO_TAG}{suffix}. Run D for the selected scenario first.')
    return capacities, loaded, missing


def missing_exporter_candidates(regions):
    """Find missing-D regions that contain hydro capacity and could export."""
    rows = []
    for region in regions:
        directory = region_dir(region)
        hydro_file = (os.path.join(directory, 'hydro', 'hydro_plants_merge.csv')
                      if directory else None)
        if not hydro_file or not os.path.exists(hydro_file):
            continue
        frame = pd.read_csv(hydro_file)
        if 'capacity_mw' not in frame.columns:
            continue
        capacity = pd.to_numeric(frame['capacity_mw'], errors='coerce')
        nameplate = float(capacity[capacity > 0].sum())
        if nameplate > EPS:
            rows.append({
                'region': region,
                'hydro_plant_count': int((capacity > 0).sum()),
                'hydro_nameplate_mw': nameplate,
                'd_summary_available': 0,
                'excluded_from_e_source_capacity': 1,
                'reason': ('D currently creates source capacity only from hydro matched '
                           'to local DC demand; this region has no usable D summary'),
            })
    return rows


def load_capacity_upper_bounds(regions):
    """Nameplate upper bounds used only to seed E's monotone redispatch iteration."""
    bounds = {code: {region: 0.0 for region in regions} for code in SCENARIOS}
    rows = []
    for region in regions:
        directory = region_dir(region)
        station_file = (os.path.join(directory, 'hydro', 'station_channel_result.csv')
                        if directory else None)
        if not station_file or not os.path.exists(station_file):
            rows.append({'region': region, 'non_ps_nameplate_mw': 0.0,
                         'ps_nameplate_mw': 0.0,
                         'station_input_available': 0})
            continue
        frame = pd.read_csv(station_file)
        if 'capacity_mw' not in frame.columns:
            raise ValueError(f'{station_file} lacks capacity_mw')
        if 'merge_status' in frame.columns:
            frame = frame[~frame['merge_status'].astype(str).eq('dropped')]
        capacity = pd.to_numeric(frame['capacity_mw'], errors='coerce')
        active = capacity.notna() & capacity.gt(0)
        if 'GRFR_COMID' in frame.columns:
            active &= pd.to_numeric(frame['GRFR_COMID'], errors='coerce').notna()
        frame = frame[active].copy()
        capacity = pd.to_numeric(frame['capacity_mw'], errors='coerce')
        station_type = (frame['type_unified'].astype(str)
                        if 'type_unified' in frame.columns
                        else pd.Series('STO', index=frame.index))
        non_ps = float(capacity[~station_type.eq('PS')].sum())
        ps = float(capacity[station_type.eq('PS')].sum())
        bounds['A'][region] = non_ps
        bounds['B'][region] = non_ps + ps
        bounds['C'][region] = non_ps + ps
        rows.append({'region': region, 'non_ps_nameplate_mw': non_ps,
                     'ps_nameplate_mw': ps,
                     'station_input_available': 1})
    return bounds, rows


def load_demand(regions):
    if not os.path.exists(DC_MAPPING):
        raise FileNotFoundError(f'DC mapping not found: {DC_MAPPING}')
    summary_path = os.path.join(DC_RESULTS, f'{GCM}_{SSP}_{DEMAND_YEAR}_summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'DC demand summary not found: {summary_path}')
    mapping = pd.read_csv(DC_MAPPING)
    summary = pd.read_csv(summary_path)
    needed_map = {'Datacenter ID', 'swat_region'}
    if not needed_map.issubset(mapping.columns):
        raise ValueError(f'{DC_MAPPING} lacks {sorted(needed_map - set(mapping.columns))}')
    needed_summary = {'Datacenter ID', 'p_total_peak_kw'}
    if not needed_summary.issubset(summary.columns):
        raise ValueError(f'{summary_path} lacks {sorted(needed_summary - set(summary.columns))}')
    individual_peak = {
        str(row['Datacenter ID']): float(row['p_total_peak_kw']) / 1e3 * DEMAND_SCALE
        for _, row in summary.iterrows()
        if np.isfinite(pd.to_numeric(row['p_total_peak_kw'], errors='coerce'))
    }
    raw_dcs = []
    for _, row in mapping[mapping['swat_region'].isin(regions)].iterrows():
        dc_id = str(row['Datacenter ID'])
        value = individual_peak.get(dc_id, np.nan)
        if np.isfinite(value) and value > EPS:
            raw_dcs.append({'dc_id': dc_id, 'region': str(row['swat_region']),
                            'individual_peak_mw': float(value)})
    if not raw_dcs:
        raise RuntimeError(f'No positive DC demand found for bloc={BLOC} in {summary_path}')

    # D evaluates hydro against the hourly aggregate regional load curve. Summing
    # each data center's own peak would assume all non-coincident peaks occur at
    # once and is therefore inconsistent with D. Preserve individual DC shares,
    # but scale them to the region's GLOBAL coincident peak (max over 2027-2040)
    # - D normalizes its DC load by dc_peak = max(_dc_maxes over all years), so
    # the binding magnitude is the peak-year (buildout) load, not a single year.
    dcs = []
    coincident_peaks = {}
    demand_years = list(range(2027, 2041))
    target_regions = sorted({row['region'] for row in raw_dcs})
    region_peaks = {region: 0.0 for region in target_regions}
    for year in demand_years:
        hourly_path = os.path.join(
            DC_RESULTS, f'region_hourly_{GCM}_{SSP}_{year}.npz')
        if not os.path.exists(hourly_path):
            raise FileNotFoundError(
                f'Regional hourly DC profile not found: {hourly_path}; '
                'E cannot derive a D-consistent global coincident peak')
        with np.load(hourly_path, allow_pickle=True) as hourly:
            for region in target_regions:
                key = f'{region}__P_total_kw'
                if key not in hourly:
                    raise KeyError(f'{hourly_path} lacks {key}')
                values = np.asarray(hourly[key], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0 or np.max(finite) <= 0:
                    continue
                region_peaks[region] = max(
                    region_peaks[region],
                    float(np.max(finite)) / 1e3 * DEMAND_SCALE)
    for region in target_regions:
        if region_peaks[region] <= 0:
            raise ValueError(f'{region}: no positive finite DC demand over 2027-2040')
    for region in target_regions:
        region_peak = region_peaks[region]
        coincident_peaks[region] = region_peak
        members = [row for row in raw_dcs if row['region'] == region]
        weight_total = sum(row['individual_peak_mw'] for row in members)
        if weight_total <= EPS:
            raise ValueError(f'Individual DC peak weights sum to zero for {region}')
        for row in members:
            dcs.append({
                'dc_id': row['dc_id'],
                'region': region,
                'demand_mw': region_peak * row['individual_peak_mw'] / weight_total,
                'individual_peak_mw': row['individual_peak_mw'],
            })
    return dcs, summary_path, coincident_peaks


def load_dc_shapes(demand_regions):
    """Load the 2027-2040 DC growth path, normalized to each region's GLOBAL
    (max-over-years) peak - so the shape GROWS year over year exactly like D's
    L_norm_DC (D normalizes per-year DC load by the global dc_peak). Combined with
    allocation magnitudes at the same global peak, E's redispatch override
    reconstructs the actual growing 2027-2040 DC load, consistent with D.
    """
    years = list(range(2027, 2041))
    max_hours = 8784

    # Pass 1: per-region GLOBAL peak across all years (matches D's dc_peak).
    per_year = {}  # year -> {region -> hourly MW-shaped (length per leap)}
    gpeak = {region: 0.0 for region in demand_regions}
    for year in years:
        path = os.path.join(DC_RESULTS, f'region_hourly_{GCM}_{SSP}_{year}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Cross-region redispatch needs {path}')
        nhrs = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
        per_year[year] = {}
        with np.load(path, allow_pickle=True) as data:
            for region in demand_regions:
                key = f'{region}__P_total_kw'
                if key not in data:
                    raise KeyError(f'{path} lacks {key}')
                raw = np.asarray(data[key], dtype=np.float64)
                if raw.size >= nhrs:
                    values = raw[:nhrs]
                else:
                    # Match D EXACTLY. D fills a leap year's extra hours with
                    # np.resize (cyclic wrap from the start of the series), keeping
                    # hours 0..8759 unshifted - NOT a Feb-29 insertion that would
                    # shift Mar-Dec by 24h. Using the identical method keeps E's
                    # redispatch DC load byte-identical to D's baseline L_norm_DC.
                    values = np.resize(raw, nhrs)
                per_year[year][region] = values
                gpeak[region] = max(gpeak[region], float(np.nanmax(values)))
    for region in demand_regions:
        if not np.isfinite(gpeak[region]) or gpeak[region] <= 0:
            raise ValueError(f'{region}: invalid global DC peak {gpeak[region]}')

    # Pass 2: normalize every year by the GLOBAL peak -> shapes grow toward 1.0 at
    # the peak year and are < 1 in earlier (smaller-buildout) years.
    shapes = {region: np.zeros((max_hours, len(years)), dtype=np.float64)
              for region in demand_regions}
    for y_idx, year in enumerate(years):
        nhrs = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
        for region in demand_regions:
            shapes[region][:nhrs, y_idx] = np.nan_to_num(
                per_year[year][region][:nhrs] / gpeak[region],
                nan=0.0, posinf=0.0, neginf=0.0)
    return shapes, years


def load_graph():
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(
            f'{MASTER_CSV} not found; run build_interconnect_map.py to rebuild it')
    master = pd.read_csv(MASTER_CSV)
    required = {'bloc', 'region_a', 'region_b', 'capacity_mw', 'type', 'status',
                'expected_year', 'line_name'}
    if not required.issubset(master.columns):
        raise ValueError(f'{MASTER_CSV} lacks {sorted(required - set(master.columns))}')
    master['capacity_mw'] = pd.to_numeric(master['capacity_mw'], errors='coerce')
    expected_year = pd.to_numeric(master['expected_year'], errors='coerce')
    operational = master['status'].eq('existing')
    committed_by_year = (
        master['status'].isin(['under_construction', 'planned'])
        & expected_year.notna()
        & expected_year.le(DEMAND_YEAR)
    )
    # REVUB_E_ALL_LINES=1 lifts the year-vintage filter: every catalogued corridor
    # (existing/planned/under_construction, dated or not) is treated as in service.
    include_corridor = pd.Series(True, index=master.index) if INCLUDE_ALL_LINES \
        else (operational | committed_by_year)
    rows = master[
        master['bloc'].astype(str).str.upper().eq(BLOC.upper())
        & include_corridor
        & master['capacity_mw'].notna()
        & master['capacity_mw'].gt(0)
        & master['region_a'].ne(master['region_b'])
    ].copy()
    if 'intra_region' in rows.columns:
        intra = rows['intra_region'].astype(str).str.lower().isin(['true', '1', 'yes'])
        rows = rows[~intra]
    if rows.empty:
        raise RuntimeError(f'No transmission assets available for bloc={BLOC}, year={DEMAND_YEAR}')
    rows['included_as_future'] = ~rows['status'].eq('existing')
    pairs = np.sort(rows[['region_a', 'region_b']].astype(str).values, axis=1)
    rows['region_a'] = pairs[:, 0]
    rows['region_b'] = pairs[:, 1]
    rows['ac_mw'] = np.where(
        rows['type'].eq('AC'), rows['capacity_mw'],
        np.where(rows['type'].eq('MIXED'), rows['capacity_mw'] / 2, 0.0))
    rows['hvdc_mw'] = np.where(
        rows['type'].eq('HVDC'), rows['capacity_mw'],
        np.where(rows['type'].eq('MIXED'), rows['capacity_mw'] / 2, 0.0))
    frame = rows.groupby(['region_a', 'region_b'], as_index=False).agg(
        total_mw=('capacity_mw', 'sum'),
        ac_mw=('ac_mw', 'sum'),
        hvdc_mw=('hvdc_mw', 'sum'),
        source_count=('line_name', 'size'),
        future_source_count=('included_as_future', 'sum'),
    )
    required = {'region_a', 'region_b', 'total_mw'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Year-filtered transmission table lacks {sorted(required - set(frame.columns))}')
    adjacency = {}
    template = {}
    for _, row in frame.iterrows():
        ra, rb = str(row['region_a']), str(row['region_b'])
        nominal = float(pd.to_numeric(row['total_mw'], errors='coerce'))
        if ra == rb or not np.isfinite(nominal) or nominal <= EPS:
            continue
        ac_mw = float(pd.to_numeric(row.get('ac_mw', 0.0), errors='coerce'))
        hvdc_mw = float(pd.to_numeric(row.get('hvdc_mw', 0.0), errors='coerce'))
        ac_mw = ac_mw if np.isfinite(ac_mw) and ac_mw > 0 else 0.0
        hvdc_mw = hvdc_mw if np.isfinite(hvdc_mw) and hvdc_mw > 0 else 0.0
        if ac_mw + hvdc_mw <= EPS:
            types = str(row.get('types', row.get('type', 'AC')))
            if types == 'HVDC':
                hvdc_mw = nominal
            elif types == 'AC':
                ac_mw = nominal
            else:
                ac_mw = nominal / 2
                hvdc_mw = nominal / 2
        assumed_available = ac_mw * AC_ATC_FACTOR + hvdc_mw * HVDC_ATC_FACTOR
        weighted_loss = ((ac_mw * AC_ATC_FACTOR * AC_LOSS_RATE
                          + hvdc_mw * HVDC_ATC_FACTOR * HVDC_LOSS_RATE)
                         / assumed_available)
        available = assumed_available
        if 'available_mw' in frame.columns:
            explicit = pd.to_numeric(row.get('available_mw'), errors='coerce')
            if np.isfinite(explicit) and explicit >= 0:
                available = min(float(explicit), nominal)
        if available <= EPS:
            continue
        if 'loss_rate' in frame.columns:
            explicit_loss = pd.to_numeric(row.get('loss_rate'), errors='coerce')
            if np.isfinite(explicit_loss) and 0 <= explicit_loss < 1:
                weighted_loss = float(explicit_loss)
        key = tuple(sorted((ra, rb)))
        template[key] = {
            'region_a': key[0], 'region_b': key[1], 'nominal_mw': nominal,
            'available_mw': available, 'remaining_mw': available,
            'efficiency': 1.0 - weighted_loss, 'loss_rate': weighted_loss,
            'ac_mw': ac_mw, 'hvdc_mw': hvdc_mw,
            'source_count': int(row.get('source_count', 0)),
            'future_source_count': int(row.get('future_source_count', 0)),
        }
        adjacency.setdefault(ra, set()).add(rb)
        adjacency.setdefault(rb, set()).add(ra)
    if not template:
        raise RuntimeError(f'No existing transmission corridors found for bloc={BLOC}')
    grid_stats = {
        'asset_rows': int(len(rows)),
        'existing_asset_rows': int(rows['status'].eq('existing').sum()),
        'future_asset_rows': int(rows['included_as_future'].sum()),
    }
    return adjacency, template, grid_stats


def fresh_edges(template):
    return {key: dict(value) for key, value in template.items()}


def shortest_feasible_path(start, goal, adjacency, edges):
    """Fewest-hop feasible path, with lower cumulative loss as the tie-breaker."""
    if start == goal:
        return [start]
    heap = [(0, 0.0, start, [start])]
    best = {}
    while heap:
        hops, loss_cost, node, path = heapq.heappop(heap)
        score = (hops, loss_cost)
        if node in best and best[node] <= score:
            continue
        best[node] = score
        if node == goal:
            return path
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in path:
                continue
            edge = edges.get(tuple(sorted((node, neighbor))))
            if edge is None or edge['remaining_mw'] <= EPS:
                continue
            heapq.heappush(
                heap,
                (hops + 1, loss_cost - math.log(edge['efficiency']),
                 neighbor, path + [neighbor]))
    return None


def path_capacity(path, edges, source_remaining):
    """Return maximum source injection and delivered MW for an oriented path."""
    if len(path) < 2:
        return source_remaining, source_remaining, 1.0
    cumulative_efficiency = 1.0
    max_send = source_remaining
    for ra, rb in zip(path[:-1], path[1:]):
        edge = edges[tuple(sorted((ra, rb)))]
        max_send = min(max_send, edge['remaining_mw'] / cumulative_efficiency)
        cumulative_efficiency *= edge['efficiency']
    return max_send, max_send * cumulative_efficiency, cumulative_efficiency


def deduct_path(path, edges, source_send):
    cumulative_efficiency = 1.0
    for ra, rb in zip(path[:-1], path[1:]):
        edge = edges[tuple(sorted((ra, rb)))]
        edge_flow = source_send * cumulative_efficiency
        edge['remaining_mw'] = max(0.0, edge['remaining_mw'] - edge_flow)
        cumulative_efficiency *= edge['efficiency']


def best_fit_candidate(dc_region, unmet, source_remaining, adjacency, edges):
    candidates = []
    for source_region, remaining in source_remaining.items():
        if source_region == dc_region or remaining <= EPS:
            continue
        path = shortest_feasible_path(source_region, dc_region, adjacency, edges)
        if path is None:
            continue
        max_send, max_delivered, efficiency = path_capacity(path, edges, remaining)
        if max_delivered <= EPS:
            continue
        # True BFD choice: the smallest source/path that can finish this load;
        # if none can, use the largest partial fit and continue.
        if max_delivered + EPS >= unmet:
            fit_key = (0, max_delivered - unmet)
        else:
            fit_key = (1, unmet - max_delivered)
        candidates.append((fit_key, len(path) - 1, -efficiency, source_region,
                           path, max_send, max_delivered, efficiency))
    return min(candidates) if candidates else None


def allocation_row(code, scope, dc, source_region, sent, delivered, path):
    return {
        'scenario': code,
        'scenario_name': SCENARIOS[code][1],
        'scope': scope,
        'dc_id': dc['dc_id'],
        'dc_region': dc['region'],
        'src_region': source_region,
        'sent_mw': sent,
        'delivered_mw': delivered,
        'allocated_mw': delivered,
        'loss_mw': sent - delivered,
        'hops': len(path) - 1,
        'path': ' > '.join(path),
        'cross_region': int(source_region != dc['region']),
    }


def run_dispatch(code, capacity_by_region, dcs, adjacency, edge_template, allow_import,
                 local_cap_by_region=None):
    source_remaining = {region: max(0.0, float(capacity_by_region.get(region, 0.0)))
                        for region in TRADING_BLOCS[BLOC]}
    edges = fresh_edges(edge_template)
    allocations = []
    scope = 'interconnected' if allow_import else 'isolated'
    ordered_dcs = sorted(dcs, key=lambda row: (-row['demand_mw'], row['dc_id']))
    unmet_by_dc = {dc['dc_id']: dc['demand_mw'] for dc in ordered_dcs}

    # Optional separate cap on LOCAL serving (E-v2: local supply is the D-verified
    # supply vs the region's own DC curve, while exports are seeded/verified
    # separately). None -> original behavior (single shared cap).
    local_cap_left = (None if local_cap_by_region is None else
                      {region: max(0.0, float(local_cap_by_region.get(region, 0.0)))
                       for region in TRADING_BLOCS[BLOC]})

    # Phase 1 reserves each region's capacity for all of its own DCs before any
    # export is allowed. Within a region, larger DCs are processed first.
    for dc in ordered_dcs:
        unmet = unmet_by_dc[dc['dc_id']]
        local = source_remaining.get(dc['region'], 0.0)
        if local_cap_left is not None:
            local = min(local, local_cap_left.get(dc['region'], 0.0))
        if local > EPS:
            delivered = min(unmet, local)
            source_remaining[dc['region']] -= delivered
            if local_cap_left is not None:
                local_cap_left[dc['region']] -= delivered
            unmet -= delivered
            allocations.append(allocation_row(
                code, scope, dc, dc['region'], delivered, delivered, [dc['region']]))
        unmet_by_dc[dc['dc_id']] = unmet

    # Phase 2 applies BFD only to deficits left after every local reservation.
    if allow_import:
        for dc in ordered_dcs:
            unmet = unmet_by_dc[dc['dc_id']]
            while unmet > EPS:
                candidate = best_fit_candidate(
                    dc['region'], unmet, source_remaining, adjacency, edges)
                if candidate is None:
                    break
                _, _, _, source_region, path, max_send, max_delivered, efficiency = candidate
                delivered = min(unmet, max_delivered)
                sent = delivered / efficiency
                sent = min(sent, max_send, source_remaining[source_region])
                delivered = sent * efficiency
                if delivered <= EPS:
                    break
                source_remaining[source_region] -= sent
                deduct_path(path, edges, sent)
                unmet -= delivered
                allocations.append(allocation_row(
                    code, scope, dc, source_region, sent, delivered, path))
                unmet_by_dc[dc['dc_id']] = unmet
    columns = ['scenario', 'scenario_name', 'scope', 'dc_id', 'dc_region',
               'src_region', 'sent_mw', 'delivered_mw', 'allocated_mw',
               'loss_mw', 'hops', 'path', 'cross_region']
    return pd.DataFrame(allocations, columns=columns), edges


def source_target_profiles(allocations, dc_shapes, years):
    """Build each source region's mixed local/remote DC target at the sending end."""
    profiles = {}
    weights = []
    if allocations.empty:
        return profiles, weights
    grouped = allocations.groupby(['src_region', 'dc_region'], as_index=False)['sent_mw'].sum()
    for _, row in grouped.iterrows():
        source = str(row['src_region'])
        sink = str(row['dc_region'])
        sent = float(row['sent_mw'])
        if sent <= EPS:
            continue
        if sink not in dc_shapes:
            raise KeyError(f'No hourly DC shape for destination region {sink}')
        profiles.setdefault(source, np.zeros((8784, len(years)), dtype=np.float64))
        profiles[source] += sent * dc_shapes[sink]
        weights.append({'src_region': source, 'dc_region': sink, 'sent_mw': sent})
    return profiles, weights


def profile_cache_key(region, profile):
    digest = hashlib.sha256()
    digest.update(region.encode('utf-8'))
    digest.update(GCM.encode('utf-8'))
    digest.update(SSP.encode('utf-8'))
    digest.update(SCENARIO_TAG.encode('utf-8'))
    for filename in REGIONAL_MODEL_FILES:
        path = os.path.join(CODE, filename)
        digest.update(filename.encode('utf-8'))
        with open(path, 'rb') as handle:
            digest.update(handle.read())
    directory = region_dir(region)
    if directory is None:
        raise FileNotFoundError(f'No SWAT region directory for E source {region}')
    inference_base = os.environ.get(
        'REVUB_INFERENCE_BASE', '/datasets/swat_global/inference_output')
    relative_region = os.path.relpath(directory, '/datasets/swat_global')
    input_paths = [
        os.path.join(directory, 'hydro', 'station_channel_result.csv'),
        os.environ.get('REVUB_TOPO_FILE',
                       os.path.join(directory, 'hydro', 'hydro_topology.csv')),
        os.path.join(directory, 'hydro', 'ps_relocation.csv'),
        os.path.join(directory, 'datasets', 'ml_dataset_hourly.parquet'),
        os.path.join(directory, 'datasets', 'load_profile_hourly.parquet'),
        os.path.join(inference_base, relative_region, GCM, SSP,
                     'corrected_all.parquet'),
    ]
    for path in input_paths:
        digest.update(path.encode('utf-8'))
        if os.path.exists(path):
            stat = os.stat(path)
            digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode('ascii'))
        else:
            digest.update(b'missing')
    excluded = {
        'REVUB_OUTPUT_DIR_OVERRIDE', 'REVUB_D_OUTPUT_DIR_OVERRIDE',
        'REVUB_E_DC_PROFILE_OVERRIDE', 'REVUB_REGION_PATH',
        'REVUB_BLOC', 'REVUB_GCM', 'REVUB_SSP', 'REVUB_SCENARIO_TAG',
        'REVUB_NUMBA_THREADS', 'REVUB_D_SUMMARY_ONLY',
        'REVUB_LOAD_TYPE', 'REVUB_LOAD_FRAC', 'REVUB_CALIBRATION_ONLY',
        'REVUB_PS_MODE', 'REVUB_N_PS_ELCC',
    }
    for name, value in sorted(os.environ.items()):
        if (name.startswith('REVUB_') and not name.startswith('REVUB_E_')
                and name not in excluded):
            digest.update(name.encode('utf-8'))
            digest.update(str(value).encode('utf-8'))
    digest.update(np.ascontiguousarray(profile).tobytes())
    return digest.hexdigest()[:16]


def run_region_redispatch(code, region, profile, years, weights, out_dir):
    # One regional run produces all A/B/C summary columns. Share it whenever
    # the actual source profile is identical across E scenarios.
    key = profile_cache_key(region, profile)
    cache_dir = os.path.join(out_dir, 'redispatch_cache', region, key)
    os.makedirs(cache_dir, exist_ok=True)
    profile_path = os.path.join(cache_dir, 'source_dc_target.npz')
    summary_path = os.path.join(cache_dir, 'dc_hydro_summary.csv')
    complete_path = os.path.join(cache_dir, '_e_region_complete.txt')
    pd.DataFrame(
        [row for row in weights if row['src_region'] == region],
        columns=['src_region', 'dc_region', 'sent_mw'],
    ).to_csv(os.path.join(cache_dir, f'source_dc_weights_{code}.csv'), index=False)
    if not (os.path.exists(summary_path) and os.path.exists(complete_path)):
        np.savez_compressed(
            profile_path, P_target_mw=profile,
            simulation_years=np.asarray(years, dtype=int))
        directory = region_dir(region)
        if directory is None:
            raise FileNotFoundError(f'No SWAT region directory for E source {region}')
        load_scenario = SCENARIO_TAG if SCENARIO_TAG in ('flat', 'typical') else 'flat'
        command = [
            sys.executable, os.path.join(CODE, 'E_REVUB_region_dispatch.py'),
            directory, load_scenario, GCM, SSP, profile_path, cache_dir,
        ]
        env = dict(os.environ)
        env.setdefault('REVUB_NUMBA_THREADS', '2')
        log_path = os.path.join(cache_dir, 'worker.log')
        with open(log_path, 'w') as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f'E regional redispatch failed for {code}/{region}; see {log_path}')
    frame = pd.read_csv(summary_path)
    if len(frame) != 1:
        raise ValueError(f'{summary_path} must contain one row')
    column = SCENARIOS[code][0]
    value = pd.to_numeric(frame.iloc[0].get(column), errors='coerce')
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{summary_path}:{column} is invalid: {value}')
    return {
        'region': region,
        'scenario': code,
        'target_peak_mw': float(np.nanmax(profile)),
        'redispatched_supply_mw': float(value),
        'cache_dir': cache_dir,
    }


def iterative_redispatch(code, initial_capacity, dcs, adjacency, edge_template,
                         dc_shapes, years, out_dir, allow_import):
    """Monotonically tighten source bounds until every allocated DC portfolio is feasible."""
    capacity = {region: max(0.0, float(initial_capacity.get(region, 0.0)))
                for region in TRADING_BLOCS[BLOC]}
    iteration_rows = []
    final_alloc = None
    final_edges = None
    converged = False
    for iteration in range(1, REDISPATCH_MAX_ITER + 1):
        allocation, final_edges = run_dispatch(
            code, capacity, dcs, adjacency, edge_template, allow_import)
        profiles, weights = source_target_profiles(allocation, dc_shapes, years)
        jobs = []
        with ThreadPoolExecutor(max_workers=REDISPATCH_WORKERS) as executor:
            for region, profile in profiles.items():
                if np.nanmax(profile) <= EPS:
                    continue
                jobs.append(executor.submit(
                    run_region_redispatch, code, region, profile, years, weights, out_dir))
            results = [future.result() for future in as_completed(jobs)]

        max_reduction = 0.0
        for result in results:
            region = result['region']
            old = capacity[region]
            target = result['target_peak_mw']
            realized = result['redispatched_supply_mw']
            # A trial below the source bound only proves feasibility up to the trial target.
            # Tighten the bound solely when the regional model cannot meet that target.
            new = min(old, realized) if realized + REDISPATCH_TOL_MW < target else old
            capacity[region] = max(0.0, new)
            reduction = old - capacity[region]
            max_reduction = max(max_reduction, reduction)
            iteration_rows.append({
                'scenario': code,
                'scope': 'interconnected' if allow_import else 'isolated',
                'iteration': iteration,
                'region': region,
                'target_peak_mw': target,
                'redispatched_supply_mw': realized,
                'capacity_before_mw': old,
                'capacity_after_mw': capacity[region],
                'capacity_reduction_mw': reduction,
                'cache_dir': result['cache_dir'],
            })
        final_alloc = allocation
        scope = 'interconnected' if allow_import else 'isolated'
        print(f'    E redispatch {code} {scope} iteration {iteration}: '
              f'{len(results)} source regions, max bound reduction={max_reduction:.2f} MW')
        if max_reduction <= REDISPATCH_TOL_MW:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f'E redispatch {code} ({"interconnected" if allow_import else "isolated"}) '
            f'did not converge after {REDISPATCH_MAX_ITER} iterations; '
            'refusing to report an allocation that has not been verified by the regional model')

    # If the final iteration tightened a bound, regenerate the network allocation once
    # with the accepted capacities so reported flows cannot exceed the final source bound.
    final_alloc, final_edges = run_dispatch(
        code, capacity, dcs, adjacency, edge_template, allow_import)
    return final_alloc, final_edges, capacity, iteration_rows


def met_by_region(allocations):
    if allocations.empty:
        return {}
    return allocations.groupby('dc_region')['delivered_mw'].sum().to_dict()


def make_trade_rows(code, allocations, regions):
    trade = {region: {'self_supply_mw': 0.0, 'export_sent_mw': 0.0,
                      'import_delivered_mw': 0.0, 'transmission_loss_mw': 0.0}
             for region in regions}
    for _, row in allocations.iterrows():
        source, sink = row['src_region'], row['dc_region']
        if source == sink:
            trade[source]['self_supply_mw'] += row['delivered_mw']
        else:
            trade[source]['export_sent_mw'] += row['sent_mw']
            trade[sink]['import_delivered_mw'] += row['delivered_mw']
            trade[source]['transmission_loss_mw'] += row['loss_mw']
    rows = []
    for region, values in trade.items():
        if not any(value > EPS for value in values.values()):
            continue
        rows.append({'scenario': code, 'scenario_name': SCENARIOS[code][1],
                     'region': region, **values,
                     'net_export_mw': (values['export_sent_mw']
                                       - values['import_delivered_mw'])})
    return rows


def make_rerun_rows(code, allocations, redispatched):
    """Report regions affected by trade and whether E already recomputed them."""
    cross = allocations[allocations['cross_region'].eq(1)]
    if cross.empty:
        return []
    rows = []
    path_nodes = set()
    for path in cross['path']:
        path_nodes.update(str(path).split(' > '))
    affected = sorted(set(cross['src_region']) | set(cross['dc_region']) | path_nodes)
    for region in affected:
        exported = cross.loc[cross['src_region'].eq(region), 'sent_mw'].sum()
        imported = cross.loc[cross['dc_region'].eq(region), 'delivered_mw'].sum()
        transit = any(
            region in str(path).split(' > ')[1:-1]
            for path in cross['path'])
        role = []
        if exported > EPS:
            role.append('exporter')
        if imported > EPS:
            role.append('importer')
        if transit:
            role.append('transit')
        rows.append({
            'scenario': code,
            'scenario_name': SCENARIOS[code][1],
            'region': region,
            'role': '+'.join(role),
            'export_sent_mw': exported,
            'import_delivered_mw': imported,
            'd_hydro_rerun_required': int(exported > EPS and not redispatched),
            'regional_balance_recompute_required': int(
                (exported > EPS or imported > EPS) and not redispatched),
            'e_network_recompute_required': int(transit),
            'joint_optimization_required': int(
                (exported > EPS or imported > EPS) and not redispatched),
            'current_D_accepts_cross_region_profile': int(redispatched),
            'reason': (
                'E supplied the allocated local/remote DC portfolio to the source-region '
                'dispatch; residual hydro serves the source regional grid'
                if redispatched else
                'trade changes the source hydro target profile; rerun D/E jointly before '
                'treating this allocation as final'),
        })
    return rows


def corridor_rows(code, final_edges):
    rows = []
    for edge in final_edges.values():
        used = edge['available_mw'] - edge['remaining_mw']
        rows.append({
            'scenario': code, 'scenario_name': SCENARIOS[code][1],
            'region_a': edge['region_a'], 'region_b': edge['region_b'],
            'nominal_mw': edge['nominal_mw'],
            'available_mw': edge['available_mw'],
            'used_mw': used,
            'remaining_mw': edge['remaining_mw'],
            'utilization_pct': 100 * used / edge['available_mw'],
            'loss_rate': edge['loss_rate'],
        })
    return rows


def main():
    validate_settings()
    regions = TRADING_BLOCS[BLOC]
    print(f'\n{"=" * 72}\nE: D-based cross-region dispatch | bloc={BLOC}\n{"=" * 72}')
    dcs, demand_path, coincident_peaks = load_demand(regions)
    capacities, loaded_regions, missing_regions = load_d_scenario_capacity(regions)
    upper_bounds, source_input_rows = load_capacity_upper_bounds(regions)
    missing_required = sorted(set(coincident_peaks) - set(loaded_regions))
    if missing_required and not REDISPATCH:
        raise RuntimeError(
            'D output is missing for regions that have DC demand: '
            f'{", ".join(missing_required)}. Refusing a partial bloc dispatch.')
    missing_hydro = missing_exporter_candidates(missing_regions)
    if REDISPATCH:
        for row in missing_hydro:
            row['excluded_from_e_source_capacity'] = 0
            row['reason'] = ('No D summary is required: E seeds this source from the station '
                             'inventory and verifies it by regional redispatch when it receives '
                             'a local or cross-region DC allocation')
    source_input_complete = all(
        row['station_input_available'] for row in source_input_rows)
    scenario_violations = []
    for region in loaded_regions:
        a = capacities['A'].get(region, 0.0)
        b = capacities['B'].get(region, 0.0)
        c = capacities['C'].get(region, 0.0)
        if b + EPS < a or c + EPS < b:
            scenario_violations.append({
                'region': region,
                'supply_A_mw': a,
                'supply_B_mw': b,
                'supply_C_mw': c,
                'expected_invariant': 'supply_C >= supply_B >= supply_A',
            })
    adjacency, edge_template, grid_stats = load_graph()
    total_demand = sum(dc['demand_mw'] for dc in dcs)
    transit_nodes = sorted(set(adjacency) - set(regions))
    print(f'  D summaries: {len(loaded_regions)} regions; missing: {len(missing_regions)}')
    print(f'  DC demand: {len(dcs)} centers, {total_demand:.1f} MW ({demand_path})')
    print(f'  Transmission: {len(edge_template)} corridors, '
          f'{len(transit_nodes)} transit-only nodes')
    if INCLUDE_ALL_LINES:
        print(f'  Grid: ALL corridors included (REVUB_E_ALL_LINES=1, year filter OFF) - '
              f'{grid_stats["existing_asset_rows"]} existing + {grid_stats["future_asset_rows"]} '
              f'planned/under-construction, regardless of commissioning year')
    else:
        print(f'  Grid vintage {DEMAND_YEAR}: {grid_stats["existing_asset_rows"]} existing + '
              f'{grid_stats["future_asset_rows"]} planned/under-construction assets in service by year')
    if transit_nodes:
        print(f'  Transit nodes: {", ".join(transit_nodes)}')
    if missing_hydro:
        candidate_mw = sum(row['hydro_nameplate_mw'] for row in missing_hydro)
        names = ', '.join(row['region'] for row in missing_hydro)
        if REDISPATCH:
            print(f'  E source calculation: {len(missing_hydro)} regions without D summaries '
                  f'contain {candidate_mw:.1f} MW hydro nameplate and are available to E when '
                  f'the allocator assigns them demand: {names}')
        else:
            print(f'  SEVERE INPUT WARNING: {len(missing_hydro)} missing-D regions contain '
                  f'{candidate_mw:.1f} MW hydro nameplate and are excluded as exporters: {names}')
    if scenario_violations:
        names = ', '.join(row['region'] for row in scenario_violations)
        print(f'  D BASELINE WARNING: scenario monotonicity is violated in '
              f'{len(scenario_violations)} regions: {names}. E1 will use fresh regional '
              'redispatch results.' if REDISPATCH else
              f'  SEVERE INPUT WARNING: D scenario monotonicity is violated in '
              f'{len(scenario_violations)} regions: {names}.')
    print(f'  ATC factors: AC={AC_ATC_FACTOR:.2f}, HVDC={HVDC_ATC_FACTOR:.2f}; '
          f'loss/corridor: AC={AC_LOSS_RATE:.1%}, HVDC={HVDC_LOSS_RATE:.1%}')

    scale_tag = format(DEMAND_SCALE, 'g').replace('.', 'p')
    grid_tag = f'year_{DEMAND_YEAR}_scale_{scale_tag}'
    if INCLUDE_ALL_LINES:
        grid_tag += '_alllines'
    out_dir = os.path.join(CODE, 'E_output', BLOC, GCM, SSP, SCENARIO_TAG, grid_tag)
    os.makedirs(out_dir, exist_ok=True)
    dc_shapes, shape_years = (load_dc_shapes(coincident_peaks)
                              if REDISPATCH else ({}, []))
    if REDISPATCH:
        print(f'  Regional redispatch: enabled, max_iter={REDISPATCH_MAX_ITER}, '
              f'tolerance={REDISPATCH_TOL_MW:g} MW, workers={REDISPATCH_WORKERS}')

    all_e0 = []
    all_e1 = []
    coverage_rows = []
    trade_rows = []
    utilization_rows = []
    summary_rows = []
    rerun_rows = []
    redispatch_rows = []
    source_bound_rows = []
    demand_by_region = {}
    for dc in dcs:
        demand_by_region[dc['region']] = demand_by_region.get(dc['region'], 0.0) + dc['demand_mw']

    for code in SCENARIOS:
        if REDISPATCH:
            e0, _, isolated_capacity, isolated_iterations = iterative_redispatch(
                code, upper_bounds[code], dcs, adjacency, edge_template,
                dc_shapes, shape_years, out_dir, False)
            e1, final_edges, final_capacity, connected_iterations = iterative_redispatch(
                code, upper_bounds[code], dcs, adjacency, edge_template,
                dc_shapes, shape_years, out_dir, True)
            redispatch_rows.extend(isolated_iterations)
            redispatch_rows.extend(connected_iterations)
        else:
            e0, _ = run_dispatch(
                code, capacities[code], dcs, adjacency, edge_template, False)
            e1, final_edges = run_dispatch(
                code, capacities[code], dcs, adjacency, edge_template, True)
            isolated_capacity = {
                region: capacities[code].get(region, 0.0) for region in regions}
            final_capacity = {
                region: capacities[code].get(region, 0.0) for region in regions}
        all_e0.append(e0)
        all_e1.append(e1)
        e0.to_csv(os.path.join(out_dir, f'bloc_allocation_{code}_E0.csv'), index=False)
        e1.to_csv(os.path.join(out_dir, f'bloc_allocation_{code}_E1.csv'), index=False)
        met0, met1 = met_by_region(e0), met_by_region(e1)
        for region, demand_mw in sorted(demand_by_region.items()):
            coverage_rows.append({
                'scenario': code, 'scenario_name': SCENARIOS[code][1],
                'region': region, 'demand_mw': demand_mw,
                'met_isolated_mw': met0.get(region, 0.0),
                'met_interconnected_mw': met1.get(region, 0.0),
                'cov_isolated_pct': 100 * met0.get(region, 0.0) / demand_mw,
                'cov_interconnected_pct': 100 * met1.get(region, 0.0) / demand_mw,
            })
        trade_rows.extend(make_trade_rows(code, e1, regions))
        rerun_rows.extend(make_rerun_rows(code, e1, REDISPATCH))
        utilization_rows.extend(corridor_rows(code, final_edges))
        sent_by_source = (e1.groupby('src_region')['sent_mw'].sum().to_dict()
                          if not e1.empty else {})
        for region in regions:
            source_bound_rows.append({
                'scenario': code,
                'scenario_name': SCENARIOS[code][1],
                'region': region,
                'd_baseline_capacity_mw': capacities[code].get(region, 0.0),
                'initial_upper_bound_mw': upper_bounds[code].get(region, 0.0),
                'isolated_final_search_bound_mw': isolated_capacity.get(region, 0.0),
                'final_redispatch_bound_mw': final_capacity.get(region, 0.0),
                'allocated_source_sent_mw': sent_by_source.get(region, 0.0),
            })
        met_e0 = e0['delivered_mw'].sum()
        met_e1 = e1['delivered_mw'].sum()
        cross = e1[e1['cross_region'].eq(1)]
        summary_rows.append({
            'scenario': code, 'scenario_name': SCENARIOS[code][1],
            'source_capacity_mw': sum(final_capacity.values()),
            'd_baseline_source_capacity_mw': sum(capacities[code].values()),
            'initial_source_upper_bound_mw': sum(upper_bounds[code].values()),
            'allocated_source_sent_mw': e1['sent_mw'].sum(),
            'demand_mw': total_demand,
            'met_isolated_mw': met_e0,
            'met_interconnected_mw': met_e1,
            'interconnected_coverage_pct': 100 * met_e1 / total_demand,
            'cross_region_sent_mw': cross['sent_mw'].sum(),
            'cross_region_delivered_mw': cross['delivered_mw'].sum(),
            'transmission_loss_mw': cross['loss_mw'].sum(),
            'unmet_interconnected_mw': total_demand - met_e1,
            'complete_exporter_input_coverage': int(
                source_input_complete if REDISPATCH else not missing_hydro),
            'legacy_d_baseline_monotonicity_ok': int(not scenario_violations),
            'regional_redispatch_performed': int(REDISPATCH),
        })
        print(f'  {code} {SCENARIOS[code][1]:24s}: E0={met_e0:10.1f} MW, '
              f'E1={met_e1:10.1f} MW, loss={cross["loss_mw"].sum():8.1f} MW')

    combined_e0 = pd.concat(all_e0, ignore_index=True)
    combined_e1 = pd.concat(all_e1, ignore_index=True)
    combined_e0.to_csv(os.path.join(out_dir, 'bloc_allocation_E0.csv'), index=False)
    combined_e1.to_csv(os.path.join(out_dir, 'bloc_allocation_E1.csv'), index=False)
    pd.DataFrame(coverage_rows).to_csv(
        os.path.join(out_dir, 'bloc_coverage_compare.csv'), index=False)
    pd.DataFrame(trade_rows).to_csv(
        os.path.join(out_dir, 'bloc_region_trade.csv'), index=False)
    pd.DataFrame(utilization_rows).to_csv(
        os.path.join(out_dir, 'bloc_corridor_utilization.csv'), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, 'bloc_scenario_summary.csv'), index=False)
    pd.DataFrame(redispatch_rows, columns=[
        'scenario', 'scope', 'iteration', 'region', 'target_peak_mw',
        'redispatched_supply_mw', 'capacity_before_mw', 'capacity_after_mw',
        'capacity_reduction_mw', 'cache_dir',
    ]).to_csv(os.path.join(out_dir, 'redispatch_iterations.csv'), index=False)
    pd.DataFrame(source_bound_rows).to_csv(
        os.path.join(out_dir, 'e_source_capacity_bounds.csv'), index=False)
    pd.DataFrame(source_input_rows).to_csv(
        os.path.join(out_dir, 'e_source_station_inventory.csv'), index=False)
    pd.DataFrame(missing_hydro, columns=[
        'region', 'hydro_plant_count', 'hydro_nameplate_mw',
        'd_summary_available', 'excluded_from_e_source_capacity', 'reason',
    ]).to_csv(os.path.join(out_dir, 'missing_d_exporter_candidates.csv'), index=False)
    pd.DataFrame(scenario_violations, columns=[
        'region', 'supply_A_mw', 'supply_B_mw', 'supply_C_mw',
        'expected_invariant',
    ]).to_csv(os.path.join(out_dir, 'd_input_scenario_violations.csv'), index=False)
    pd.DataFrame(rerun_rows, columns=[
        'scenario', 'scenario_name', 'region', 'role', 'export_sent_mw',
        'import_delivered_mw', 'd_hydro_rerun_required',
        'regional_balance_recompute_required', 'e_network_recompute_required',
        'joint_optimization_required',
        'current_D_accepts_cross_region_profile', 'reason',
    ]).to_csv(os.path.join(out_dir, 'cross_region_rerun_requirements.csv'), index=False)
    pd.DataFrame([{
        'region': region,
        'coincident_peak_mw': peak,
        'sum_individual_peaks_mw': sum(
            dc['individual_peak_mw'] for dc in dcs if dc['region'] == region),
    } for region, peak in sorted(coincident_peaks.items())]).to_csv(
        os.path.join(out_dir, 'bloc_demand_basis.csv'), index=False)
    pd.DataFrame([{
        'bloc': BLOC, 'gcm': GCM, 'ssp': SSP, 'scenario_tag': SCENARIO_TAG,
        'demand_year': DEMAND_YEAR, 'demand_scale': DEMAND_SCALE,
        'ac_atc_factor': AC_ATC_FACTOR, 'hvdc_atc_factor': HVDC_ATC_FACTOR,
        'ac_loss_rate_per_corridor': AC_LOSS_RATE,
        'hvdc_loss_rate_per_corridor': HVDC_LOSS_RATE,
        'legacy_d_output_allowed': ALLOW_LEGACY,
        'regional_redispatch_enabled': int(REDISPATCH),
        'redispatch_max_iterations': REDISPATCH_MAX_ITER,
        'redispatch_tolerance_mw': REDISPATCH_TOL_MW,
        'redispatch_workers': REDISPATCH_WORKERS,
        'complete_exporter_input_coverage': int(
            source_input_complete if REDISPATCH else not missing_hydro),
        'legacy_d_baseline_monotonicity_ok': int(not scenario_violations),
        'grid_vintage_year': DEMAND_YEAR,
        'include_all_lines_year_filter_off': int(INCLUDE_ALL_LINES),
        'existing_transmission_assets': grid_stats['existing_asset_rows'],
        'future_transmission_assets_included': grid_stats['future_asset_rows'],
    }]).to_csv(os.path.join(out_dir, 'transmission_model_config.csv'), index=False)
    if rerun_rows and not REDISPATCH:
        print('  WARNING: cross-region trades are provisional. Exporting regions require '
              'D reruns with export-adjusted hourly targets; importing regions require balance '
              'recalculation. A globally coordinated result requires joint D/E iteration, and '
              'the current D cannot accept those cross-region profiles.')
    elif rerun_rows:
        print('  Cross-region source dispatch was recomputed with the allocated destination '
              'DC curves; unallocated hydro remains assigned to each source region grid load.')
    print(f'  Saved results: {out_dir}')
    print(f'{"=" * 72}')


if __name__ == '__main__':
    main()
