#!/usr/bin/env python3
"""
Generate load_profile_hourly.parquet for all 82 REVUB subregions.

Each parquet has columns: [year, hour_of_year, L_norm]
  - year: 2027..2040
  - hour_of_year: 0..8759 (or 8783 for leap years)
  - L_norm: normalized load (0..~1), P99-normalized

Data sources:
  - Brazil: ONS CURVA_CARGA CSV (local time BRT/AMT)
  - USA: EIA-930 Balance CSV (Local Time at End of Hour)
  - Europe: OPSD 60-min CSV (CET/CEST local time)
  - UK: OPSD GB_GBN + GB_NIR columns (CET/CEST)
  - Canada Ontario: IESO CSV (local EST)
  - Canada Quebec: CCEI API CSV (local)
  - Canada Atlantic: CCEI NB+NL CSV (local)
  - Canada BC: BC Hydro XLS (local PST)
  - Canada Prairies: AESO XLSX (local MST)
  - China: Zenodo provincial hourly CSV (local CST)
  - India: ICED Yearly Demand Profile XLSX (local IST)
  - SE Asia Philippines: NGCP Hourly Demand XLSX (local PHT)
  - SE Asia Malaysia: GSO SystemDemand XML (local MYT)
  - SE Asia Vietnam: GitHub ElectricityLoads XLSX (local ICT)
  - SE Asia Thailand: Zenodo system CSV (local ICT)
  - SE Asia Indonesia/Cambodia/Laos/Myanmar: digitized from figures + Thai seasonal
  - Latin America Colombia: XM API JSON (local COT)
  - Latin America Argentina: CAMMESA XLSX (local ART)
  - Latin America Peru: COES XLSX (local PET)
  - Latin America Ecuador: Zenodo/REVUB data_load XLSX (local ECT)
  - Turkey: Kaggle CSV (local TRT)
  - Proxies: Albania/N.Macedonia (from Greece), Bolivia (from Peru), Chile (from Argentina)

Usage:
  python generate_load_profiles.py                # local time (default, reproduces original)
  python generate_load_profiles.py --utc          # convert to UTC via np.roll
  python generate_load_profiles.py --dry-run      # show what would be generated
"""

import argparse
import glob
import json
import os
import shutil
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ──
LOAD_PROFILES_DIR = '/home/cfeng/myswat/load_profiles'
OUTPUT_BASE = '/datasets/swat_global'

# ── UTC offsets (hours) for each subregion ──
# Used only when --utc is specified: np.roll(L_norm, +offset) shifts local→UTC
UTC_OFFSETS = {
    # Brazil
    'brazil_norte': -4, 'brazil_nordeste': -3, 'brazil_sudeste': -3, 'brazil_sul': -3,
    # USA (standard time; DST not modeled - original behavior)
    'usa_caiso': -8, 'usa_ercot': -6, 'usa_isone': -5, 'usa_nyiso': -5,
    'usa_miso_central': -6, 'usa_miso_north': -6, 'usa_miso_south': -6,
    'usa_pjm_east': -5, 'usa_pjm_west': -5,
    'usa_spp_north': -6, 'usa_spp_south': -6,
    'usa_sertp': -5,
    'usa_northerngrid_east': -7, 'usa_northerngrid_south': -7, 'usa_northerngrid_west': -8,
    'usa_westconnect_north': -7, 'usa_westconnect_south': -7,
    # Europe (standard time CET=+1; some countries differ)
    'austria': 1, 'belgium': 1, 'bulgaria': 2, 'croatia': 1,
    'finland': 2, 'france': 1, 'germany': 1, 'greece': 2,
    'hungary': 1, 'ireland': 0, 'italy': 1, 'latvia': 2,
    'lithuania': 2, 'luxembourg': 1, 'norway': 1, 'poland': 1,
    'portugal': 0, 'romania': 2, 'serbia': 1, 'slovakia': 1,
    'slovenia': 1, 'spain': 1, 'sweden': 1, 'switzerland': 1,
    'united_kingdom': 0,
    'albania': 1, 'north_macedonia': 1,
    # Turkey
    'turkey': 3,
    # Canada
    'canada_atlantic': -4, 'canada_bc': -8, 'canada_ontario': -5,
    'canada_prairies': -7, 'canada_quebec': -5,
    # China
    'china_nc': 8, 'china_ne': 8, 'china_ec': 8, 'china_cc': 8,
    'china_nw': 8, 'china_sw': 8, 'china_csg': 8,
    # India
    'india_east': 5.5, 'india_north': 5.5, 'india_northeast': 5.5,
    'india_south': 5.5, 'india_west': 5.5,
    # Southeast Asia
    'cambodia': 7, 'indonesia': 7, 'laos': 7,
    'malaysia_east': 8, 'malaysia_west': 8,
    'myanmar': 6.5, 'philippines': 8, 'thailand': 7, 'vietnam': 7,
    # Latin America
    'argentina': -3, 'bolivia': -4, 'chile': -4,
    'colombia': -5, 'ecuador': -5, 'peru': -5,
}

# ── Region → parent folder mapping ──
REGION_PARENT = {
    'brazil_norte': 'brazil', 'brazil_nordeste': 'brazil',
    'brazil_sudeste': 'brazil', 'brazil_sul': 'brazil',
    'usa_caiso': 'usa', 'usa_ercot': 'usa', 'usa_isone': 'usa', 'usa_nyiso': 'usa',
    'usa_miso_central': 'usa', 'usa_miso_north': 'usa', 'usa_miso_south': 'usa',
    'usa_pjm_east': 'usa', 'usa_pjm_west': 'usa',
    'usa_spp_north': 'usa', 'usa_spp_south': 'usa', 'usa_sertp': 'usa',
    'usa_northerngrid_east': 'usa', 'usa_northerngrid_south': 'usa',
    'usa_northerngrid_west': 'usa',
    'usa_westconnect_north': 'usa', 'usa_westconnect_south': 'usa',
    'austria': 'europe_central', 'germany': 'europe_central', 'hungary': 'europe_central',
    'poland': 'europe_central', 'slovakia': 'europe_central', 'slovenia': 'europe_central',
    'switzerland': 'europe_central',
    'finland': 'europe_north', 'norway': 'europe_north', 'sweden': 'europe_north',
    'bulgaria': 'europe_south', 'croatia': 'europe_south', 'greece': 'europe_south',
    'italy': 'europe_south', 'portugal': 'europe_south', 'romania': 'europe_south',
    'serbia': 'europe_south', 'spain': 'europe_south',
    'albania': 'europe_south', 'north_macedonia': 'europe_south',
    'belgium': 'europe_west', 'france': 'europe_west', 'ireland': 'europe_west',
    'luxembourg': 'europe_west', 'united_kingdom': 'europe_west',
    'latvia': 'europe_baltic', 'lithuania': 'europe_baltic',
    'turkey': 'other',
    'canada_atlantic': 'canada', 'canada_bc': 'canada', 'canada_ontario': 'canada',
    'canada_prairies': 'canada', 'canada_quebec': 'canada',
    'china_nc': 'china', 'china_ne': 'china', 'china_ec': 'china',
    'china_cc': 'china', 'china_nw': 'china', 'china_sw': 'china', 'china_csg': 'china',
    'india_east': 'india', 'india_north': 'india', 'india_northeast': 'india',
    'india_south': 'india', 'india_west': 'india',
    'cambodia': 'southeast_asia', 'indonesia': 'southeast_asia', 'laos': 'southeast_asia',
    'malaysia_east': 'southeast_asia', 'malaysia_west': 'southeast_asia',
    'myanmar': 'southeast_asia', 'philippines': 'southeast_asia',
    'thailand': 'southeast_asia', 'vietnam': 'southeast_asia',
    'argentina': 'latin_america', 'bolivia': 'latin_america', 'chile': 'latin_america',
    'colombia': 'latin_america', 'ecuador': 'latin_america', 'peru': 'latin_america',
}


def out_dir(subregion):
    """Return output directory path for a subregion."""
    parent = REGION_PARENT[subregion]
    return f'{OUTPUT_BASE}/{parent}/{subregion}/datasets'


# ════════════════════════════════════════════════════════════════════
# Helper: save_lnorm  (the core pattern used by most data sources)
# ════════════════════════════════════════════════════════════════════

def save_lnorm(df, out_directory, utc_shift=0):
    """
    From hourly load dataframe with columns ['dt', 'load_mw']
    -> P99-normalized L_norm parquet for years 2027-2040.

    Normalization: L_norm = load_mw / P99(load_mw)
    Pattern: group by (day_of_year, hour_of_day), average across years.
    UTC shift: if non-zero, np.roll the annual L_norm array by that many hours.

    Returns: (p99, L_norm_mean, n_hours)
    """
    os.makedirs(out_directory, exist_ok=True)
    df = df.dropna(subset=['dt', 'load_mw']).copy()
    p99 = df['load_mw'].quantile(0.99)
    df['L_norm'] = df['load_mw'] / p99
    df['hod'] = df['dt'].dt.hour
    df['doy'] = df['dt'].dt.dayofyear
    pattern = df.groupby(['doy', 'hod'])['L_norm'].mean().reset_index()

    rows = []
    for year in range(2027, 2041):
        n = 8784 if year % 4 == 0 else 8760
        year_vals = []
        for h in range(n):
            doy = h // 24 + 1
            hod = h % 24
            m = pattern[(pattern['doy'] == min(doy, 365)) & (pattern['hod'] == hod)]
            ln = m['L_norm'].values[0] if len(m) > 0 else df['L_norm'].mean()
            year_vals.append(ln)

        if utc_shift != 0:
            # Roll: positive offset means local is ahead of UTC,
            # so to convert local->UTC we shift by -offset
            shift = -int(round(utc_shift))
            year_vals = list(np.roll(year_vals, shift))

        for h, ln in enumerate(year_vals):
            rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

    pd.DataFrame(rows).to_parquet(f'{out_directory}/load_profile_hourly.parquet', index=False)
    return p99, df['L_norm'].mean(), len(df)


def make_from_shape(hourly_24, seasonal_12, out_directory, utc_shift=0):
    """
    Generate load_profile_hourly.parquet from a 24-hour template + 12-month
    seasonal factors. Used for image-digitized profiles (Indonesia, Cambodia,
    Laos, Myanmar).

    Normalization: L_norm = (hourly_shape * seasonal) / P99 of the result.
    """
    os.makedirs(out_directory, exist_ok=True)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    rows = []
    for year in range(2027, 2041):
        n_hrs = 8784 if year % 4 == 0 else 8760
        day = 0
        year_vals = []
        for m in range(12):
            md = month_days[m] + (1 if m == 1 and year % 4 == 0 else 0)
            sf = seasonal_12[m]
            for d in range(md):
                for h in range(24):
                    hoy = day * 24 + h
                    if hoy < n_hrs:
                        year_vals.append(hourly_24[h] * sf)
                day += 1

        # P99 normalize
        p99 = np.percentile(year_vals, 99)
        year_vals = [v / p99 for v in year_vals]

        if utc_shift != 0:
            shift = -int(round(utc_shift))
            year_vals = list(np.roll(year_vals, shift))

        for h, ln in enumerate(year_vals):
            rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

    pd.DataFrame(rows).to_parquet(f'{out_directory}/load_profile_hourly.parquet', index=False)


# ════════════════════════════════════════════════════════════════════
# Data source processors
# ════════════════════════════════════════════════════════════════════

def process_brazil_ons(utc=False):
    """
    Brazil ONS CURVA_CARGA CSVs -> 4 subregions.
    Data: semicolon-separated, columns din_instante, nom_subsistema, val_cargaenergiahomwmed
    Time: local (BRT=UTC-3 for SUL/SUDESTE/NORDESTE; AMT=UTC-4 for NORTE)
    Normalization: P99 per subsystem.
    """
    print("=== Brazil (ONS) ===")
    subsystem_map = {
        'SUL': 'brazil_sul',
        'SUDESTE': 'brazil_sudeste',
        'NORDESTE': 'brazil_nordeste',
        'NORTE': 'brazil_norte',
    }

    dfs = []
    for year in range(2020, 2025):
        f = f'{LOAD_PROFILES_DIR}/brazil/CURVA_CARGA_{year}.csv'
        if os.path.exists(f):
            dfs.append(pd.read_csv(f, sep=';'))
    if not dfs:
        print("  WARNING: No Brazil ONS files found"); return

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['dt'] = pd.to_datetime(all_data['din_instante'])
    all_data['load_mw'] = all_data['val_cargaenergiahomwmed']

    for subsys, subregion in subsystem_map.items():
        sd = all_data[all_data['nom_subsistema'] == subsys][['dt', 'load_mw']].dropna()
        shift = UTC_OFFSETS[subregion] if utc else 0
        n = save_lnorm(sd, out_dir(subregion), utc_shift=shift)
        print(f"  {subregion}: {n[2]} hrs, P99={n[0]:.0f} MW")


def process_usa_eia(utc=False):
    """
    USA EIA-930 Balance CSVs -> 17 subregions.
    Data: 'Demand (MW)', 'Balancing Authority', 'Local Time at End of Hour'
    Time: local (varies by BA). For multi-BA subregions, demands are summed.
    Normalization: P99 per subregion.
    """
    print("\n=== USA (EIA-930) ===")
    mapping = {
        'usa_caiso': ['CISO'],
        'usa_ercot': ['ERCO'],
        'usa_isone': ['ISNE'],
        'usa_nyiso': ['NYIS'],
        'usa_miso_central': ['MISO'],
        'usa_miso_north': ['MISO'],
        'usa_miso_south': ['MISO'],
        'usa_pjm_east': ['PJM'],
        'usa_pjm_west': ['PJM'],
        'usa_spp_north': ['SWPP'],
        'usa_spp_south': ['SWPP'],
        'usa_sertp': ['SOCO', 'TVA', 'DUK', 'SC', 'SCEG'],
        'usa_northerngrid_east': ['NWMT', 'WAUW'],
        'usa_northerngrid_south': ['PACE', 'PACW'],
        'usa_northerngrid_west': ['BPAT', 'AVA'],
        'usa_westconnect_north': ['WACM', 'PSCO'],
        'usa_westconnect_south': ['WALC', 'SRP', 'AZPS'],
    }

    files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/usa/EIA930_BALANCE_*.csv'))
    if not files:
        print("  WARNING: No EIA files found"); return

    eia = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    eia['demand_mw'] = pd.to_numeric(eia['Demand (MW)'], errors='coerce')
    eia['ba'] = eia['Balancing Authority']
    eia['dt'] = pd.to_datetime(eia['Local Time at End of Hour'], format='mixed', errors='coerce')

    for sub, bas in mapping.items():
        sd = eia[eia['ba'].isin(bas)].dropna(subset=['dt', 'demand_mw'])
        if len(bas) > 1:
            hourly = sd.groupby('dt')['demand_mw'].sum().reset_index()
        else:
            hourly = sd[['dt', 'demand_mw']].drop_duplicates('dt')
        hourly.columns = ['dt', 'load_mw']
        hourly = hourly[hourly['load_mw'] > 100]

        shift = UTC_OFFSETS[sub] if utc else 0
        n = save_lnorm(hourly, out_dir(sub), utc_shift=shift)
        print(f"  {sub}: {n[2]} hrs, P99={n[0]:.0f} MW")


def process_europe_opsd(utc=False):
    """
    Europe OPSD 60-min CSV -> 24 countries.
    Data: cet_cest_timestamp, {CC}_load_actual_entsoe_transparency
    Time: CET/CEST (local time, extracted from timestamp string).
    Note: the cet_cest_timestamp has mixed UTC offsets (+0100/+0200) due to DST.
    We extract local hour directly from the string to avoid DST ambiguity.
    Normalization: P99 per country.
    """
    print("\n=== Europe (OPSD) ===")
    opsd_file = f'{LOAD_PROFILES_DIR}/europe/opsd_60min.csv'
    if not os.path.exists(opsd_file):
        print("  WARNING: OPSD file not found"); return

    opsd = pd.read_csv(opsd_file, low_memory=False)
    opsd['local_hour'] = opsd['cet_cest_timestamp'].str[11:13].astype(int)
    opsd['local_date'] = pd.to_datetime(opsd['cet_cest_timestamp'].str[:10], errors='coerce')
    opsd['doy'] = opsd['local_date'].dt.dayofyear

    eu_map = {
        'austria': 'AT', 'belgium': 'BE', 'bulgaria': 'BG', 'croatia': 'HR',
        'finland': 'FI', 'france': 'FR', 'germany': 'DE', 'greece': 'GR',
        'hungary': 'HU', 'ireland': 'IE', 'italy': 'IT', 'latvia': 'LV',
        'lithuania': 'LT', 'luxembourg': 'LU', 'norway': 'NO', 'poland': 'PL',
        'portugal': 'PT', 'romania': 'RO', 'serbia': 'RS', 'slovakia': 'SK',
        'slovenia': 'SI', 'spain': 'ES', 'sweden': 'SE', 'switzerland': 'CH',
    }

    for country, code in eu_map.items():
        col = f'{code}_load_actual_entsoe_transparency'
        if col not in opsd.columns:
            continue
        sub = opsd[['local_date', 'local_hour', 'doy', col]].copy()
        sub['load_mw'] = pd.to_numeric(sub[col], errors='coerce')
        sub = sub.dropna(subset=['local_date', 'load_mw'])
        sub = sub[sub['load_mw'] > 100]
        if len(sub) < 1000:
            continue

        p99 = sub['load_mw'].quantile(0.99)
        sub['L_norm'] = sub['load_mw'] / p99
        pattern = sub.groupby(['doy', 'local_hour'])['L_norm'].mean().reset_index()

        utc_shift = UTC_OFFSETS[country] if utc else 0

        rows = []
        for year in range(2027, 2041):
            n = 8784 if year % 4 == 0 else 8760
            year_vals = []
            for h in range(n):
                doy = h // 24 + 1
                hod = h % 24
                m = pattern[(pattern['doy'] == min(doy, 365)) & (pattern['local_hour'] == hod)]
                ln = m['L_norm'].values[0] if len(m) > 0 else sub['L_norm'].mean()
                year_vals.append(ln)

            if utc_shift != 0:
                shift = -int(round(utc_shift))
                year_vals = list(np.roll(year_vals, shift))

            for h, ln in enumerate(year_vals):
                rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

        od = out_dir(country)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame(rows).to_parquet(f'{od}/load_profile_hourly.parquet', index=False)
        print(f"  {country}: P99={p99:.0f} MW")

    del opsd  # free memory


def process_uk_opsd(utc=False):
    """
    UK from OPSD: GB_GBN + GB_NIR columns summed.
    Time: CET/CEST (same as other OPSD). UK is actually UTC+0 (GMT/BST),
    but OPSD stores everything in CET/CEST. We extract local_hour from the
    CET/CEST timestamp, same as other European countries.
    Normalization: P99.
    """
    print("\n=== United Kingdom (OPSD) ===")
    opsd_file = f'{LOAD_PROFILES_DIR}/europe/opsd_60min.csv'
    if not os.path.exists(opsd_file):
        print("  WARNING: OPSD file not found"); return

    opsd = pd.read_csv(opsd_file, low_memory=False)
    opsd['local_hour'] = opsd['cet_cest_timestamp'].str[11:13].astype(int)
    opsd['local_date'] = pd.to_datetime(opsd['cet_cest_timestamp'].str[:10], errors='coerce')
    opsd['doy'] = opsd['local_date'].dt.dayofyear

    col1 = 'GB_GBN_load_actual_entsoe_transparency'
    col2 = 'GB_NIR_load_actual_entsoe_transparency'
    if col1 not in opsd.columns:
        print("  WARNING: GB columns not found"); return

    sub = opsd[['local_date', 'local_hour', 'doy']].copy()
    sub['load_mw'] = (
        pd.to_numeric(opsd[col1], errors='coerce').fillna(0) +
        pd.to_numeric(opsd.get(col2, pd.Series(0)), errors='coerce').fillna(0)
    )
    sub = sub[sub['load_mw'] > 1000].dropna(subset=['local_date'])

    p99 = sub['load_mw'].quantile(0.99)
    sub['L_norm'] = sub['load_mw'] / p99
    pattern = sub.groupby(['doy', 'local_hour'])['L_norm'].mean().reset_index()

    utc_shift = UTC_OFFSETS['united_kingdom'] if utc else 0

    rows = []
    for year in range(2027, 2041):
        n = 8784 if year % 4 == 0 else 8760
        year_vals = []
        for h in range(n):
            doy = h // 24 + 1
            hod = h % 24
            m = pattern[(pattern['doy'] == min(doy, 365)) & (pattern['local_hour'] == hod)]
            ln = m['L_norm'].values[0] if len(m) > 0 else sub['L_norm'].mean()
            year_vals.append(ln)

        if utc_shift != 0:
            shift = -int(round(utc_shift))
            year_vals = list(np.roll(year_vals, shift))

        for h, ln in enumerate(year_vals):
            rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

    od = out_dir('united_kingdom')
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f'{od}/load_profile_hourly.parquet', index=False)
    print(f"  united_kingdom: P99={p99:.0f} MW")

    del opsd


def process_canada_ontario(utc=False):
    """
    Canada Ontario: IESO CSV files.
    Data: Date, Hour (1-24), Ontario Demand
    Time: local (EST/EDT)
    Normalization: P99.
    """
    print("\n=== Canada Ontario (IESO) ===")
    dfs = []
    for year in [2022, 2023, 2024]:
        f = f'{LOAD_PROFILES_DIR}/canada/ontario_{year}.csv'
        if os.path.exists(f):
            d = pd.read_csv(f, skiprows=3)
            d['dt'] = pd.to_datetime(d['Date']) + pd.to_timedelta(d['Hour'] - 1, unit='h')
            d['load_mw'] = pd.to_numeric(d['Ontario Demand'], errors='coerce')
            dfs.append(d[['dt', 'load_mw']].dropna())
    if not dfs:
        print("  WARNING: No IESO files found"); return
    ont = pd.concat(dfs)
    shift = UTC_OFFSETS['canada_ontario'] if utc else 0
    p, m, n = save_lnorm(ont, out_dir('canada_ontario'), utc_shift=shift)
    print(f"  canada_ontario: IESO REAL, P99={p:.0f} MW, {n} hrs")


def process_canada_quebec(utc=False):
    """
    Canada Quebec: CCEI API CSV (QC_demand.csv).
    Data: DATETIME_LOCAL, OBS_VALUE
    Time: local
    Normalization: P99.
    """
    print("\n=== Canada Quebec (CCEI) ===")
    f = f'{LOAD_PROFILES_DIR}/canada/QC_demand.csv'
    if not os.path.exists(f):
        print("  WARNING: QC_demand.csv not found"); return
    d = pd.read_csv(f)
    d['dt'] = pd.to_datetime(d['DATETIME_LOCAL'], errors='coerce')
    d['load_mw'] = pd.to_numeric(d['OBS_VALUE'], errors='coerce')
    d = d.dropna(subset=['dt', 'load_mw'])
    d = d[d['load_mw'] > 10]
    shift = UTC_OFFSETS['canada_quebec'] if utc else 0
    p, m, n = save_lnorm(d[['dt', 'load_mw']], out_dir('canada_quebec'), utc_shift=shift)
    print(f"  canada_quebec: CCEI REAL, P99={p:.0f} MW, {n} hrs")


def process_canada_atlantic(utc=False):
    """
    Canada Atlantic: CCEI NB + NL CSVs aggregated.
    Data: DATETIME_LOCAL, OBS_VALUE
    Time: local
    Normalization: P99.
    """
    print("\n=== Canada Atlantic (CCEI NB+NL) ===")
    at_dfs = []
    for prov in ['NB', 'NL']:
        f = f'{LOAD_PROFILES_DIR}/canada/{prov}_demand.csv'
        if os.path.exists(f) and os.path.getsize(f) > 1000:
            d = pd.read_csv(f)
            d['dt'] = pd.to_datetime(d['DATETIME_LOCAL'], errors='coerce')
            d['load_mw'] = pd.to_numeric(d['OBS_VALUE'], errors='coerce')
            at_dfs.append(d[['dt', 'load_mw']].dropna())
    if not at_dfs:
        print("  WARNING: No NB/NL files found"); return
    combined = pd.concat(at_dfs)
    hourly = combined.groupby('dt')['load_mw'].sum().reset_index()
    hourly.columns = ['dt', 'load_mw']
    hourly = hourly[hourly['load_mw'] > 100]
    shift = UTC_OFFSETS['canada_atlantic'] if utc else 0
    p, m, n = save_lnorm(hourly, out_dir('canada_atlantic'), utc_shift=shift)
    print(f"  canada_atlantic: CCEI REAL (NB+NL), P99={p:.0f} MW, {n} hrs")


def process_canada_bc(utc=False):
    """
    Canada BC: BC Hydro XLS files.
    Data: date, hour (1-24), load_mw
    Time: local (PST)
    Normalization: P99.
    """
    print("\n=== Canada BC (BC Hydro) ===")
    bc_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/canada/bc_hydro/*.xls'))
    if not bc_files:
        print("  WARNING: No BC Hydro files found"); return
    dfs = []
    for f in bc_files:
        d = pd.read_excel(f, skiprows=3)
        d.columns = ['date', 'hour', 'load_mw']
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d['hour'] = pd.to_numeric(d['hour'], errors='coerce')
        d['load_mw'] = pd.to_numeric(d['load_mw'], errors='coerce')
        d = d.dropna()
        d['dt'] = d['date'] + pd.to_timedelta(d['hour'] - 1, unit='h')
        dfs.append(d[['dt', 'load_mw']])
    bc = pd.concat(dfs).sort_values('dt').drop_duplicates('dt')
    bc = bc[bc['load_mw'] > 1000]
    shift = UTC_OFFSETS['canada_bc'] if utc else 0
    p, m, n = save_lnorm(bc, out_dir('canada_bc'), utc_shift=shift)
    print(f"  canada_bc: BC Hydro REAL, P99={p:.0f} MW, {n} hrs")


def process_canada_prairies(utc=False):
    """
    Canada Prairies: AESO (Alberta) XLSX files.
    Data: DT_MST, area columns summed
    Time: local (MST)
    Normalization: P99.
    """
    print("\n=== Canada Prairies (AESO) ===")
    aeso_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/canada/aeso/*.xlsx'))
    if not aeso_files:
        print("  WARNING: No AESO files found"); return
    dfs = []
    for f in aeso_files:
        d = pd.read_excel(f)
        d['dt'] = pd.to_datetime(d['DT_MST'], errors='coerce')
        area_cols = [c for c in d.columns if c != 'DT_MST' and d[c].dtype in ['float64', 'int64']]
        d['load_mw'] = d[area_cols].sum(axis=1)
        dfs.append(d[['dt', 'load_mw']].dropna())
    aeso = pd.concat(dfs).sort_values('dt').drop_duplicates('dt')
    aeso = aeso[aeso['load_mw'] > 1000]
    shift = UTC_OFFSETS['canada_prairies'] if utc else 0
    p, m, n = save_lnorm(aeso, out_dir('canada_prairies'), utc_shift=shift)
    print(f"  canada_prairies: AESO REAL, P99={p:.0f} MW, {n} hrs")


def process_china_zenodo(utc=False):
    """
    China: Zenodo provincial hourly electric power load CSV.
    Data: semicolon-separated, columns = province codes (BJ, TJ, ...), rows = hours (8760)
    Time: local (CST = UTC+8), all provinces same timezone
    Normalization: P99 per grid region (sum of provinces).
    Province -> grid mapping:
      china_nc: BJ,TJ,HB,SX,SD
      china_ne: LN,JL,HL,NM
      china_ec: SH,JS,ZJ,AH,FJ,JX
      china_cc: HA,HN  (JX assigned to EC to avoid duplication)
      china_nw: SN,GS,QH,NX,XJ,XZ
      china_sw: CQ,SC,GZ,YN
      china_csg: GD,GX,HI
    """
    print("\n=== China (Zenodo) ===")
    f = f'{LOAD_PROFILES_DIR}/china/Appendix 1_Hourly electric power load final.csv'
    if not os.path.exists(f):
        print("  WARNING: China Zenodo CSV not found"); return

    df = pd.read_csv(f, sep=';')
    grid_map = {
        'china_nc': ['BJ', 'TJ', 'HB', 'SX', 'SD'],
        'china_ne': ['LN', 'JL', 'HL', 'NM'],
        'china_ec': ['SH', 'JS', 'ZJ', 'AH', 'FJ', 'JX'],
        'china_cc': ['HA', 'HN'],
        'china_nw': ['SN', 'GS', 'QH', 'NX', 'XJ', 'XZ'],
        'china_sw': ['CQ', 'SC', 'GZ', 'YN'],
        'china_csg': ['GD', 'GX', 'HI'],
    }

    for region, provs in grid_map.items():
        valid = [p for p in provs if p in df.columns]
        if not valid:
            print(f"  {region}: NO matching provinces!"); continue

        vals = df[valid].sum(axis=1)
        p99 = vals.quantile(0.99)
        ln = vals / p99

        utc_shift = UTC_OFFSETS[region] if utc else 0

        rows = []
        for year in range(2027, 2041):
            n = 8784 if year % 4 == 0 else 8760
            year_vals = []
            for h in range(n):
                year_vals.append(float(ln.iloc[h % len(ln)]))

            if utc_shift != 0:
                shift = -int(round(utc_shift))
                year_vals = list(np.roll(year_vals, shift))

            for h, lv in enumerate(year_vals):
                rows.append({'year': year, 'hour_of_year': h, 'L_norm': lv})

        od = out_dir(region)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame(rows).to_parquet(f'{od}/load_profile_hourly.parquet', index=False)
        print(f"  {region} ({','.join(valid)}): P99={p99:.0f} MWh")


def process_india_iced(utc=False):
    """
    India: ICED Yearly Demand Profile XLSX files.
    Data: Region, Hourly Demand Met (in MW) - sequential rows, 24 per day
    Time: local (IST = UTC+5:30)
    Normalization: P99 per subregion.
    Region name mapping:
      'Eastern' (not 'North') -> india_east
      'North-Eastern' -> india_northeast
      'Northen'/'Northern' -> india_north
      'Southern' -> india_south
      'Western' -> india_west
    """
    print("\n=== India (ICED) ===")
    iced_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/india/ICED/Yearly Demand Profile*.xlsx'))
    if not iced_files:
        print("  WARNING: No ICED files found"); return

    all_data = []
    for f in iced_files:
        all_data.append(pd.read_excel(f, sheet_name='Yearly Demand Profile'))
    df = pd.concat(all_data, ignore_index=True).dropna(subset=['Region'])

    region_map_in = {}
    for r in df['Region'].unique():
        rl = r.lower()
        if 'eastern' in rl and 'north' not in rl:
            region_map_in[r] = 'india_east'
        elif 'north-eastern' in rl:
            region_map_in[r] = 'india_northeast'
        elif 'northen' in rl or 'northern' in rl:
            region_map_in[r] = 'india_north'
        elif 'southern' in rl:
            region_map_in[r] = 'india_south'
        elif 'western' in rl:
            region_map_in[r] = 'india_west'

    df['subregion'] = df['Region'].map(region_map_in)
    df['demand_mw'] = pd.to_numeric(df['Hourly Demand Met (in MW)'], errors='coerce')
    df = df.dropna(subset=['subregion', 'demand_mw'])
    df = df[df['demand_mw'] > 100]

    for sub in ['india_east', 'india_north', 'india_northeast', 'india_south', 'india_west']:
        sd = df[df['subregion'] == sub].reset_index(drop=True)
        sd['hod'] = sd.index % 24
        sd['doy'] = sd.index // 24 % 365 + 1

        p99 = sd['demand_mw'].quantile(0.99)
        sd['L_norm'] = sd['demand_mw'] / p99
        pattern = sd.groupby(['doy', 'hod'])['L_norm'].mean().reset_index()

        utc_shift = UTC_OFFSETS[sub] if utc else 0

        rows = []
        for year in range(2027, 2041):
            n = 8784 if year % 4 == 0 else 8760
            year_vals = []
            for h in range(n):
                doy = h // 24 + 1
                hod = h % 24
                m = pattern[(pattern['doy'] == min(doy, 365)) & (pattern['hod'] == hod)]
                ln = m['L_norm'].values[0] if len(m) > 0 else sd['L_norm'].mean()
                year_vals.append(ln)

            if utc_shift != 0:
                shift = -int(round(utc_shift))
                year_vals = list(np.roll(year_vals, shift))

            for h, ln in enumerate(year_vals):
                rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

        od = out_dir(sub)
        os.makedirs(od, exist_ok=True)
        pd.DataFrame(rows).to_parquet(f'{od}/load_profile_hourly.parquet', index=False)
        print(f"  {sub}: P99={p99:.0f} MW, {len(sd)} hrs")


def process_se_asia_real(utc=False):
    """
    Southeast Asia real data sources:
      - Philippines: NGCP Hourly Demand XLSX (Luzon grid, cols 1-24)
      - Malaysia: GSO SystemDemand XML (10-min -> hourly resample)
      - Vietnam: GitHub ElectricityLoads XLSX (province columns summed)
      - Thailand: Zenodo system CSV (5-min -> hourly resample)
    Time: all local (PHT=+8, MYT=+8, ICT=+7, ICT=+7)
    Normalization: P99.
    """
    print("\n=== Southeast Asia (real data) ===")
    base = LOAD_PROFILES_DIR

    # Philippines
    f = f'{base}/southeast_asia/phi/Hourly Demand.xlsx'
    if os.path.exists(f):
        df2 = pd.read_excel(f, header=None)
        rows_list = []
        for _, row in df2.iterrows():
            try:
                dt = pd.to_datetime(row.iloc[0])
                for h in range(24):
                    val = float(row.iloc[1 + h]) if 1 + h < len(row) else np.nan
                    if val > 100:
                        rows_list.append({'dt': dt + pd.Timedelta(hours=h), 'load_mw': val})
            except:
                pass
        if rows_list:
            shift = UTC_OFFSETS['philippines'] if utc else 0
            p, m, n = save_lnorm(pd.DataFrame(rows_list), out_dir('philippines'), utc_shift=shift)
            print(f"  philippines: NGCP REAL, P99={p:.0f} MW, {n} hrs")
    else:
        print(f"  philippines: file not found at {f}")

    # Malaysia GSO XML
    mala_rows = []
    for f in sorted(glob.glob(f'{base}/southeast_asia/mala/SystemDemand*.xml')):
        tree = ET.parse(f)
        root = tree.getroot()
        for elem in root.findall('.//DataTable'):
            dt_e = elem.find('DT')
            mw_e = elem.find('MW')
            if dt_e is not None and mw_e is not None:
                try:
                    mala_rows.append({'dt': pd.to_datetime(dt_e.text), 'load_mw': float(mw_e.text)})
                except:
                    pass
    if mala_rows:
        mala = pd.DataFrame(mala_rows).set_index('dt').resample('h')['load_mw'].mean().reset_index()
        mala.columns = ['dt', 'load_mw']
        shift_w = UTC_OFFSETS['malaysia_west'] if utc else 0
        p, m, n = save_lnorm(mala, out_dir('malaysia_west'), utc_shift=shift_w)
        print(f"  malaysia_west: GSO REAL, P99={p:.0f} MW, {n} hrs")
        # Copy west -> east
        os.makedirs(out_dir('malaysia_east'), exist_ok=True)
        shutil.copy2(
            f"{out_dir('malaysia_west')}/load_profile_hourly.parquet",
            f"{out_dir('malaysia_east')}/load_profile_hourly.parquet"
        )
        print(f"  malaysia_east: copied from west")
    else:
        print("  malaysia: no XML files found")

    # Vietnam
    f = f'{base}/southeast_asia/vietnam/ElectricityLoads.xlsx'
    if os.path.exists(f):
        vn = pd.read_excel(f)
        vn['dt'] = pd.to_datetime(vn.iloc[:, 0], errors='coerce')
        num_cols = [c for c in vn.columns[1:] if vn[c].dtype in ['float64', 'int64']]
        vn['load_mw'] = vn[num_cols].sum(axis=1)
        vn = vn[['dt', 'load_mw']].dropna()
        vn = vn[vn['load_mw'] > 1000]
        shift = UTC_OFFSETS['vietnam'] if utc else 0
        p, m, n = save_lnorm(vn, out_dir('vietnam'), utc_shift=shift)
        print(f"  vietnam: GitHub REAL, P99={p:.0f} MW, {n} hrs")
    else:
        print(f"  vietnam: file not found at {f}")

    # Thailand Zenodo
    import zipfile
    zf = f'{base}/southeast_asia/thai/17109911.zip'
    if os.path.exists(zf):
        with zipfile.ZipFile(zf) as z:
            csvs = [n for n in z.namelist() if n.endswith('.csv')]
            dfs = []
            for n in csvs:
                z.extract(n, f'{base}/southeast_asia/thai/')
                d = pd.read_csv(f'{base}/southeast_asia/thai/{n}')
                d['dt'] = pd.to_datetime(d['datetime'], format='mixed', errors='coerce')
                dcols = [c for c in d.columns if 'demand' in c.lower()]
                d['load_mw'] = d[dcols].sum(axis=1)
                dfs.append(d[['dt', 'load_mw']].dropna())
        thai = pd.concat(dfs).set_index('dt').resample('h')['load_mw'].mean().reset_index()
        thai.columns = ['dt', 'load_mw']
        thai = thai[thai['load_mw'] > 5000]
        shift = UTC_OFFSETS['thailand'] if utc else 0
        p, m, n = save_lnorm(thai, out_dir('thailand'), utc_shift=shift)
        print(f"  thailand: Zenodo REAL, P99={p:.0f} MW, {n} hrs")
    else:
        # Try pre-extracted CSVs
        thai_csvs = sorted(glob.glob(f'{base}/southeast_asia/thai/system_20*.csv'))
        if thai_csvs:
            dfs = []
            for f in thai_csvs:
                d = pd.read_csv(f)
                d['dt'] = pd.to_datetime(d['datetime'], format='mixed', errors='coerce')
                dcols = [c for c in d.columns if 'demand' in c.lower()]
                d['load_mw'] = d[dcols].sum(axis=1)
                dfs.append(d[['dt', 'load_mw']].dropna())
            thai = pd.concat(dfs).set_index('dt').resample('h')['load_mw'].mean().reset_index()
            thai.columns = ['dt', 'load_mw']
            thai = thai[thai['load_mw'] > 5000]
            shift = UTC_OFFSETS['thailand'] if utc else 0
            p, m, n = save_lnorm(thai, out_dir('thailand'), utc_shift=shift)
            print(f"  thailand: Zenodo REAL (pre-extracted), P99={p:.0f} MW, {n} hrs")
        else:
            print("  thailand: no data found")


def process_se_asia_image(utc=False):
    """
    SE Asia countries with digitized load shapes from published figures:
      - Indonesia: Fig 5, Java-Bali daily shape, flat seasonal (equatorial)
      - Cambodia: Fig 20a, daily + Thai seasonal
      - Laos: Fig 20b, daily + Thai seasonal
      - Myanmar: Fig 20c, daily + Thai seasonal

    Daily shapes are normalized to max=1. Seasonal factors are borrowed from
    Thailand's real data pattern.

    Normalization: P99 of (daily_shape * seasonal_factor).
    """
    print("\n=== Southeast Asia (image-digitized) ===")

    # Daily shapes from published figures
    indo_daily = np.array([
        0.78, 0.76, 0.75, 0.74, 0.74, 0.76, 0.78, 0.81, 0.87, 0.90, 0.91, 0.92,
        0.91, 0.88, 0.90, 0.91, 0.92, 0.94, 0.98, 1.00, 0.97, 0.93, 0.88, 0.82
    ])
    camb_daily = np.array([
        0.53, 0.52, 0.51, 0.51, 0.52, 0.55, 0.62, 0.71, 0.80, 0.87, 0.91, 0.93,
        0.89, 0.84, 0.82, 0.84, 0.89, 0.93, 0.98, 1.00, 0.96, 0.89, 0.76, 0.62
    ])
    laos_daily = np.array([
        0.44, 0.42, 0.40, 0.40, 0.42, 0.44, 0.52, 0.60, 0.68, 0.74, 0.78, 0.80,
        0.76, 0.72, 0.70, 0.72, 0.76, 0.84, 0.96, 1.00, 0.94, 0.84, 0.68, 0.54
    ])
    myan_daily = np.array([
        0.47, 0.45, 0.43, 0.43, 0.45, 0.49, 0.57, 0.66, 0.74, 0.79, 0.81, 0.81,
        0.77, 0.72, 0.70, 0.72, 0.77, 0.85, 0.96, 1.00, 0.94, 0.85, 0.70, 0.55
    ])

    # Extract Thai seasonal from its real-data parquet
    thai_pq_path = f'{OUTPUT_BASE}/southeast_asia/thailand/datasets/load_profile_hourly.parquet'
    if os.path.exists(thai_pq_path):
        thai_pq = pd.read_parquet(thai_pq_path)
        t27 = thai_pq[thai_pq['year'] == 2027].copy()
        t27['month'] = np.clip((t27['hour_of_year'] // 730), 0, 11)
        thai_monthly = t27.groupby('month')['L_norm'].mean()
        thai_seasonal = (thai_monthly / thai_monthly.mean()).values
    else:
        # Fallback: mild tropical seasonal (from original Block 4)
        print("  WARNING: Thailand parquet not found, using fallback seasonal")
        thai_seasonal = np.array([1.0, 1.0, 1.02, 1.05, 1.05, 1.02, 1.0, 1.0, 1.0, 1.0, 0.98, 0.95])

    for name, shape in [
        ('indonesia', indo_daily),
        ('cambodia', camb_daily),
        ('laos', laos_daily),
        ('myanmar', myan_daily),
    ]:
        shift = UTC_OFFSETS[name] if utc else 0
        make_from_shape(shape, thai_seasonal, out_dir(name), utc_shift=shift)
        print(f"  {name}: image+thai seasonal")


def process_colombia_xm(utc=False):
    """
    Colombia: XM API JSON files (previously downloaded).
    Data: JSON with Items[].HourlyEntities[].Values.Hour01..Hour24
    Time: local (COT = UTC-5)
    Normalization: P99.
    """
    print("\n=== Colombia (XM API) ===")
    col_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/latin_america/colombia_2*.json'))
    if not col_files:
        print("  WARNING: No Colombia JSON files found"); return

    all_d = []
    for f in col_files:
        try:
            d = json.load(open(f))
            for item in d.get('Items', []):
                date = item['Date']
                for ent in item.get('HourlyEntities', []):
                    vals = ent.get('Values', {})
                    for h in range(1, 25):
                        key = f'Hour{h:02d}'
                        if key in vals:
                            dt = pd.to_datetime(date) + pd.Timedelta(hours=h - 1)
                            all_d.append({'dt': dt, 'load_mw': float(vals[key]) / 1000})  # kWh->MWh
        except Exception:
            pass

    if not all_d:
        print("  WARNING: No valid Colombia data"); return

    df = pd.DataFrame(all_d).sort_values('dt').drop_duplicates('dt')
    df = df[df['load_mw'] > 1000]
    shift = UTC_OFFSETS['colombia'] if utc else 0
    p, m, n = save_lnorm(df, out_dir('colombia'), utc_shift=shift)
    print(f"  colombia: XM REAL, P99={p:.0f} MW, {n} hrs")


def process_argentina_cammesa(utc=False):
    """
    Argentina: CAMMESA hourly demand XLSX.
    Data: header=None, skiprows=6, columns [year,date,month,day,holiday,daytype,datetime,hour,partial,ror,total]
    Time: local (ART = UTC-3)
    Normalization: P99.
    """
    print("\n=== Argentina (CAMMESA) ===")
    f_ar = glob.glob(f'{LOAD_PROFILES_DIR}/latin_america_raw/argentina/*.xlsx')
    if not f_ar:
        # Try alternate path
        f_ar = glob.glob(f'{LOAD_PROFILES_DIR}/latin_america/argentina/*.xlsx')
    if not f_ar:
        print("  WARNING: No CAMMESA files found"); return

    d = pd.read_excel(f_ar[0], header=None, skiprows=6)
    d.columns = ['year', 'date', 'month', 'day', 'holiday', 'daytype', 'datetime',
                 'hour', 'partial', 'ror', 'total'][:d.shape[1]]
    # NOTE: the 'datetime' column holds the per-DAY date but is DATE-ONLY (all 00:00:00); the
    # actual hour-of-day lives in the separate 'hour' column (1-24). (The 'date' column is sparse
    # - only ~40 unique values - so it must NOT be used.) Build the true hourly timestamp from
    # datetime + (hour-1); otherwise all 24 hourly rows collapse to midnight and the diurnal
    # cycle is averaged away.
    _hr = pd.to_numeric(d['hour'], errors='coerce')
    d['dt'] = pd.to_datetime(d['datetime'], errors='coerce') + pd.to_timedelta(_hr - 1, unit='h')
    d['load_mw'] = pd.to_numeric(d['total'], errors='coerce')
    d = d.dropna(subset=['dt', 'load_mw'])
    d = d[d['load_mw'] > 1000]
    shift = UTC_OFFSETS['argentina'] if utc else 0
    p, m, n = save_lnorm(d[['dt', 'load_mw']], out_dir('argentina'), utc_shift=shift)
    print(f"  argentina: CAMMESA REAL, P99={p:.0f} MW, {n} hrs")


def process_peru_coes(utc=False):
    """
    Peru: COES half-hourly XLSX files.
    Data: header=None, skiprows=3, columns [fecha, ejecutado, prog_diaria, prog_semanal]
    Time: local (PET = UTC-5)
    Normalization: P99. Half-hourly resampled to hourly.
    """
    print("\n=== Peru (COES) ===")
    peru_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/latin_america_raw/peru/DemandaCOES*.xlsx'))
    if not peru_files:
        peru_files = sorted(glob.glob(f'{LOAD_PROFILES_DIR}/latin_america/peru/DemandaCOES*.xlsx'))
    if not peru_files:
        print("  WARNING: No COES files found"); return

    dfs = []
    for f in peru_files:
        d = pd.read_excel(f, header=None, skiprows=3)
        d.columns = ['fecha', 'ejecutado', 'prog_d', 'prog_s']
        d['dt'] = pd.to_datetime(d['fecha'], format='mixed', errors='coerce')
        d['load_mw'] = pd.to_numeric(d['ejecutado'], errors='coerce')
        dfs.append(d[['dt', 'load_mw']].dropna())

    peru = pd.concat(dfs).sort_values('dt').drop_duplicates('dt')
    peru = peru.set_index('dt').resample('h')['load_mw'].mean().reset_index()
    peru.columns = ['dt', 'load_mw']
    peru = peru[peru['load_mw'] > 1000]
    shift = UTC_OFFSETS['peru'] if utc else 0
    p, m, n = save_lnorm(peru, out_dir('peru'), utc_shift=shift)
    print(f"  peru: COES REAL, P99={p:.0f} MW, {n} hrs")


def process_ecuador_zenodo(utc=False):
    """
    Ecuador: Zenodo/REVUB data_load.xlsx.
    Data: rows = hours (8760), columns = years. Values are already normalized L_norm.
    Time: local (ECT = UTC-5)
    Normalization: already normalized in source; average across year columns.
    """
    print("\n=== Ecuador (Zenodo) ===")
    f_ec = glob.glob(f'{LOAD_PROFILES_DIR}/latin_america_raw/ecaudor/data_load.xlsx')
    if not f_ec:
        f_ec = glob.glob(f'{LOAD_PROFILES_DIR}/latin_america/ecaudor/data_load.xlsx')
    if not f_ec:
        print("  WARNING: No Ecuador data_load.xlsx found"); return

    xl = pd.ExcelFile(f_ec[0])
    df_ec = xl.parse(xl.sheet_names[0])
    num_cols = df_ec.select_dtypes(include=[np.number]).columns
    avg_load = df_ec[num_cols].mean(axis=1).values[:8760]

    utc_shift = UTC_OFFSETS['ecuador'] if utc else 0

    rows = []
    for year in range(2027, 2041):
        n = 8784 if year % 4 == 0 else 8760
        year_vals = [avg_load[h % 8760] for h in range(n)]

        if utc_shift != 0:
            shift = -int(round(utc_shift))
            year_vals = list(np.roll(year_vals, shift))

        for h, ln in enumerate(year_vals):
            rows.append({'year': year, 'hour_of_year': h, 'L_norm': ln})

    od = out_dir('ecuador')
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f'{od}/load_profile_hourly.parquet', index=False)
    print(f"  ecuador: Zenodo/REVUB, L_mean={np.mean(avg_load):.3f}")


def process_turkey_kaggle(utc=False):
    """
    Turkey: Kaggle power generation and consumption CSV.
    Data: Date_Time (dd.mm.yyyy HH:MM), Consumption (MWh)
    Time: local (TRT = UTC+3)
    Normalization: P99.
    Output: written to both other/turkey AND europe_central/turkey.
    """
    print("\n=== Turkey (Kaggle) ===")
    f = f'{LOAD_PROFILES_DIR}/turkey/power Generation and consumption.csv'
    if not os.path.exists(f):
        print("  WARNING: Turkey Kaggle CSV not found"); return

    d = pd.read_csv(f)
    d['dt'] = pd.to_datetime(d['Date_Time'], format='%d.%m.%Y %H:%M', errors='coerce')
    d['load_mw'] = pd.to_numeric(d['Consumption (MWh)'], errors='coerce')
    d = d.dropna(subset=['dt', 'load_mw'])
    d = d[d['load_mw'] > 5000]

    shift = UTC_OFFSETS['turkey'] if utc else 0
    p, m, n = save_lnorm(d[['dt', 'load_mw']], out_dir('turkey'), utc_shift=shift)
    print(f"  turkey (other/): P99={p:.0f} MW, {n} hrs")

    # Also copy to europe_central/turkey
    alt_dir = f'{OUTPUT_BASE}/europe_central/turkey/datasets'
    os.makedirs(alt_dir, exist_ok=True)
    shutil.copy2(
        f"{out_dir('turkey')}/load_profile_hourly.parquet",
        f"{alt_dir}/load_profile_hourly.parquet"
    )
    print(f"  turkey (europe_central/): copied")


def process_proxies(utc=False):
    """
    Proxy regions: copy load profile from a similar region.
      - Albania -> Greece
      - North Macedonia -> Greece
      - Bolivia -> Peru
      - Chile -> Argentina (similar latitude, temperate)
    """
    print("\n=== Proxies ===")
    proxy_map = {
        'albania': 'greece',
        'north_macedonia': 'greece',
        'bolivia': 'peru',
        'chile': 'argentina',
    }

    for target, source in proxy_map.items():
        src_path = f"{out_dir(source)}/load_profile_hourly.parquet"
        dst_directory = out_dir(target)

        if not os.path.exists(src_path):
            print(f"  {target}: source {source} not available yet, skipping")
            continue

        if utc and UTC_OFFSETS.get(target) != UTC_OFFSETS.get(source):
            # Need to adjust timezone: read source, roll, re-save
            src_df = pd.read_parquet(src_path)
            src_offset = UTC_OFFSETS[source]
            tgt_offset = UTC_OFFSETS[target]
            delta = int(round(tgt_offset - src_offset))  # hours difference

            if delta != 0:
                rows = []
                for year in src_df['year'].unique():
                    yr_data = src_df[src_df['year'] == year].sort_values('hour_of_year')
                    vals = yr_data['L_norm'].values
                    # The source was already rolled to UTC; target has different offset
                    # But actually for proxies, the source parquet is already in the
                    # correct time basis (UTC or local). We just need to adjust if
                    # timezones differ and we're in UTC mode.
                    vals = np.roll(vals, -delta)
                    for h, v in enumerate(vals):
                        rows.append({'year': year, 'hour_of_year': h, 'L_norm': v})
                os.makedirs(dst_directory, exist_ok=True)
                pd.DataFrame(rows).to_parquet(
                    f'{dst_directory}/load_profile_hourly.parquet', index=False)
                print(f"  {target}: proxy from {source} (tz-adjusted {delta:+d}h)")
                continue

        os.makedirs(dst_directory, exist_ok=True)
        shutil.copy2(src_path, f'{dst_directory}/load_profile_hourly.parquet')
        print(f"  {target}: proxy from {source}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate load_profile_hourly.parquet for all 82 REVUB subregions.')
    parser.add_argument('--utc', action='store_true',
                        help='Output in UTC (shift local time by -UTC_offset via np.roll)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be generated without writing')
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN: would generate load profiles for these regions:")
        for sub in sorted(REGION_PARENT.keys()):
            parent = REGION_PARENT[sub]
            offset = UTC_OFFSETS.get(sub, 0)
            print(f"  {parent}/{sub}  (UTC{offset:+.1f})")
        print(f"\nTotal: {len(REGION_PARENT)} subregion paths")
        existing = glob.glob(f'{OUTPUT_BASE}/*/*/datasets/load_profile_hourly.parquet')
        print(f"Existing on disk: {len(existing)}")
        return

    mode = "UTC" if args.utc else "local time"
    print(f"Generating load profiles in {mode} mode\n")

    # Process all data sources in dependency order
    # (Thailand must come before SE Asia image-based countries)
    process_brazil_ons(utc=args.utc)
    process_usa_eia(utc=args.utc)
    process_europe_opsd(utc=args.utc)
    process_uk_opsd(utc=args.utc)
    process_canada_ontario(utc=args.utc)
    process_canada_quebec(utc=args.utc)
    process_canada_atlantic(utc=args.utc)
    process_canada_bc(utc=args.utc)
    process_canada_prairies(utc=args.utc)
    process_china_zenodo(utc=args.utc)
    process_india_iced(utc=args.utc)
    process_se_asia_real(utc=args.utc)      # Thailand must run before image-based
    process_se_asia_image(utc=args.utc)      # Uses Thailand's seasonal pattern
    process_colombia_xm(utc=args.utc)
    process_argentina_cammesa(utc=args.utc)
    process_peru_coes(utc=args.utc)
    process_ecuador_zenodo(utc=args.utc)
    process_turkey_kaggle(utc=args.utc)
    process_proxies(utc=args.utc)            # Must run last (depends on others)

    # Summary
    total = len(glob.glob(f'{OUTPUT_BASE}/*/*/datasets/load_profile_hourly.parquet'))
    print(f"\n{'='*60}")
    print(f"TOTAL load profiles on disk: {total}")
    print(f"Mode: {mode}")


if __name__ == '__main__':
    main()
