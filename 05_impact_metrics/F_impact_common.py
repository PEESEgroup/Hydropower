"""
F_impact_common.py - shared loader / preprocessing / output for the
hydropower dispatch-impact metric programs (F1 sediment, F2 ecology, F3 supply).

See hydropower_dispatch_impact_methodology.md §1 (scenario inventory), §2 (preprocessing),
§6 (output schema). Every metric compares a dispatch flow series against the
SAME station's natural flow Q_in_nat on the [t, y, HPP] hourly grid.

Path / env conventions mirror A-E:
    REVUB_REGION_PATH   full region dir, e.g. /datasets/swat_global/brazil/brazil_nordeste
    REVUB_GCM           default mri-esm2-0
    REVUB_SSP           default ssp370
    REVUB_SCENARIO_TAG  optional sub-dir under revub_output/<GCM>/<SSP> (e.g. flat)
Output dir = <REGION_PATH>/revub_output/<GCM>/<SSP>[/<tag>], same as B/C/D.
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration resolution (env or explicit args)
# ---------------------------------------------------------------------------

def resolve_config(region_path=None, gcm=None, ssp=None, tag=None):
    """Resolve region/gcm/ssp/tag → dict with output_dir and hydro_dir.

    Falls back to env vars (REVUB_REGION_PATH / REVUB_GCM / REVUB_SSP /
    REVUB_SCENARIO_TAG), matching the rest of the pipeline.
    """
    region_path = region_path or os.environ.get('REVUB_REGION_PATH')
    if not region_path:
        raise ValueError('region_path not given and REVUB_REGION_PATH unset')
    gcm = gcm or os.environ.get('REVUB_GCM', 'mri-esm2-0')
    ssp = ssp or os.environ.get('REVUB_SSP', 'ssp370')
    if tag is None:
        tag = os.environ.get('REVUB_SCENARIO_TAG', '').strip()
    out = os.path.join(region_path, 'revub_output', gcm, ssp)
    if tag:
        out = os.path.join(out, tag)
    return {
        'region_path': region_path,
        'region_name': os.path.basename(region_path.rstrip('/')),
        'gcm': gcm, 'ssp': ssp, 'tag': tag,
        'output_dir': out,
        'hydro_dir': os.path.join(region_path, 'hydro'),
    }


# ---------------------------------------------------------------------------
# Scenario inventory (hydropower_dispatch_impact_methodology.md §1)
#   key, file, npz_key.  'nat' is the shared natural baseline.
# ---------------------------------------------------------------------------
_SCENARIO_SPEC = [
    ('nat',    'bal_hourly_arrays.npz', 'Q_in_nat'),       # baseline
    ('CONV',   'bal_hourly_arrays.npz', 'Q_CONV_out'),
    ('BAL_A',  'bal_indep_arrays.npz',  'Q_BAL_out_indep'),
    ('BAL_C',  'bal_hourly_arrays.npz', 'Q_BAL_out'),
    ('PS_C',   'scenario_C_hourly.npz', 'Q_BAL_out_C'),
    ('DC_A',   'dc_scenario_A_hourly.npz', 'Q_out_A'),
    ('DC_B',   'dc_scenario_B_hourly.npz', 'Q_out_B'),
    ('DC_C',   'dc_scenario_C_hourly.npz', 'Q_out_C'),
]


def load_scenarios(output_dir, include_e=True):
    """Load every available dispatch flow series for one bloc.

    Returns (scenarios, meta):
      scenarios : dict[str -> ndarray[t, y, HPP]]   (m3/s)  always contains 'nat'
      meta      : dict with HPP_name, hrs_byyear, simulation_years, n_T, n_Y, n_HPP
    Missing optional scenarios are simply absent from the dict.
    """
    base = os.path.join(output_dir, 'bal_hourly_arrays.npz')
    if not os.path.exists(base):
        raise FileNotFoundError(
            f'{base} not found - run the A-C pipeline for this bloc first')

    scenarios = {}
    cache = {}

    def _load(fname):
        if fname not in cache:
            p = os.path.join(output_dir, fname)
            cache[fname] = np.load(p, allow_pickle=True) if os.path.exists(p) else None
        return cache[fname]

    for key, fname, npz_key in _SCENARIO_SPEC:
        d = _load(fname)
        if d is not None and npz_key in d.files:
            scenarios[key] = np.asarray(d[npz_key], dtype=np.float64)

    if 'nat' not in scenarios:
        raise KeyError(f'Q_in_nat missing in {base}')

    bal = _load('bal_hourly_arrays.npz')
    meta = {
        'HPP_name': [str(s) for s in bal['HPP_name']],
        'hrs_byyear': np.asarray(bal['hrs_byyear'], dtype=float),
        'simulation_years': np.asarray(bal['simulation_years'], dtype=int),
    }
    nat = scenarios['nat']
    meta['n_T'], meta['n_Y'], meta['n_HPP'] = nat.shape

    # E cross-region re-dispatch (redispatch_cache/<region>/<hash>/dc_scenario_*_hourly.npz)
    if include_e:
        scenarios.update(_load_e_scenarios(output_dir, meta))

    return scenarios, meta


def _load_e_scenarios(output_dir, meta):
    """Scan for E worker hourly outputs. Key form: E_<region>_<hash8>_<X>.

    E re-dispatches a source region to a cross-region DC target; the same
    source region may appear under multiple <hash> (different targets).
    Returns {} if none found or shapes mismatch.
    """
    out = {}
    shape = (meta['n_T'], meta['n_Y'], meta['n_HPP'])
    pattern = os.path.join(output_dir, 'redispatch_cache', '*', '*',
                           'dc_scenario_*_hourly.npz')
    for path in sorted(glob.glob(pattern)):
        parts = path.split(os.sep)
        try:
            region = parts[-3]
            h = parts[-2][:8]
        except IndexError:
            continue
        scen_letter = os.path.basename(path).split('_')[2]  # A/B/C
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        npz_key = f'Q_out_{scen_letter}'
        if npz_key in d.files:
            arr = np.asarray(d[npz_key], dtype=np.float64)
            if arr.shape == shape:
                out[f'E_{region}_{h}_{scen_letter}'] = arr
    return out


# ---------------------------------------------------------------------------
# Static station attributes (aligned to the npz HPP_name order)
# ---------------------------------------------------------------------------
_STATIC_COLS = ['name_unified', 'lat_unified', 'lon_unified', 'final_head_m',
                'GRFR_uparea', 'GRFR_order', 'GRFR_lengthkm', 'elev_min', 'elev_max',
                'max_area_km2', 'max_vol_km3', 'river', 'coeff_a', 'coeff_b', 'coeff_c', 'coeff_d',
                'snap_HYRIV_ID']


def load_static(hydro_dir, hpp_names, output_dir=None):
    """Return a DataFrame of static attributes, one row per hpp_name (same order).

    `name_unified` is NOT unique (e.g. two distinct 'Funil' dams in brazil_sudeste,
    'SALUDA' in usa_sertp). A name-only join would collapse both npz positions onto
    the first CSV row, silently mis-assigning volume/area/coords. So when the
    order-preserving `region_station_profile.csv` (written by A in exact npz HPP order,
    carrying per-position name+lat) is available, we join on the composite key
    (name_unified, round(lat,3)), which is unique. Falls back to the legacy name-only
    join if the profile is absent or its order does not match.
    """
    csv = os.path.join(hydro_dir, 'station_channel_result.csv')
    df = pd.read_csv(csv, low_memory=False)
    cols = [c for c in _STATIC_COLS if c in df.columns]
    df = df[cols].copy()
    df['name_unified'] = df['name_unified'].astype(str)
    names = [str(n) for n in hpp_names]

    prof_lat = None
    if output_dir:
        prof = os.path.join(output_dir, 'region_station_profile.csv')
        if os.path.exists(prof):
            pf = pd.read_csv(prof)
            if 'name' in pf.columns and 'lat' in pf.columns and \
                    list(pf['name'].astype(str)) == names:
                prof_lat = pf['lat'].to_numpy()

    if prof_lat is not None and 'lat_unified' in df.columns:
        df['_k'] = list(zip(df['name_unified'], df['lat_unified'].round(3)))
        df = df.drop_duplicates('_k', keep='first').set_index('_k')
        keys = list(zip(names, np.round(prof_lat, 3)))
        out = df.reindex(keys).reset_index(drop=True)
        out['name_unified'] = names                  # guarantee correct per-position name
        return out

    # fallback: legacy name-only join (assumes unique names)
    d2 = df.drop_duplicates('name_unified', keep='first').set_index('name_unified')
    return d2.reindex(names).reset_index().rename(columns={'index': 'name_unified'})


# ---------------------------------------------------------------------------
# Preprocessing helpers (§2)
# ---------------------------------------------------------------------------

def valid_series(arr3d, y, hpp, hrs_byyear):
    """One station-year flow series, truncated to valid hours (drops the
    0.0-padded tail of non-leap years). NaNs preserved for downstream masking."""
    n = int(hrs_byyear[y])
    return arr3d[:n, y, hpp]


_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def month_index(n_hours, leap):
    """Vector of month number (1..12) for each hour 0..n_hours-1 (t=0 = Jan 1 00:00)."""
    days = _MONTH_DAYS.copy()
    if leap:
        days[1] = 29
    hours = days * 24
    return np.repeat(np.arange(1, 13), hours)[:n_hours]


# Northern-hemisphere month→season; flipped for the southern hemisphere.
_SEASON_N = {12: 'winter', 1: 'winter', 2: 'winter',
             3: 'spring', 4: 'spring', 5: 'spring',
             6: 'summer', 7: 'summer', 8: 'summer',
             9: 'autumn', 10: 'autumn', 11: 'autumn'}
_FLIP = {'winter': 'summer', 'summer': 'winter', 'spring': 'autumn', 'autumn': 'spring'}


def season_of_month(month, lat):
    """Hemisphere-aware season label for a calendar month."""
    s = _SEASON_N[int(month)]
    return _FLIP[s] if (lat is not None and np.isfinite(lat) and lat < 0) else s


def season_index(n_hours, leap, lat):
    """Per-hour season label array, hemisphere-aware."""
    mon = month_index(n_hours, leap)
    base = np.array([_SEASON_N[m] for m in mon], dtype=object)
    if lat is not None and np.isfinite(lat) and lat < 0:
        base = np.array([_FLIP[s] for s in base], dtype=object)
    return base


def is_leap_hours(n_hours):
    return n_hours >= 8784


# ---------------------------------------------------------------------------
# Output (§6) - one tidy long table per topic
# ---------------------------------------------------------------------------
LONG_COLUMNS = ['region', 'gcm', 'ssp', 'scenario_tag', 'station', 'station_idx',
                'river', 'lat', 'lon', 'topic', 'scenario', 'metric', 'param',
                'value_nat', 'value_reg', 'delta', 'ratio', 'year_agg', 'notes']


def make_row(cfg, static_row, station_idx, topic, scenario, metric, param,
             value_nat=np.nan, value_reg=np.nan, year_agg='mean', notes=''):
    delta = value_reg - value_nat if (np.isfinite(value_nat) and np.isfinite(value_reg)) else np.nan
    ratio = (value_reg / value_nat) if (np.isfinite(value_nat) and np.isfinite(value_reg)
                                        and value_nat != 0) else np.nan
    return {
        'region': cfg['region_name'], 'gcm': cfg['gcm'], 'ssp': cfg['ssp'],
        'scenario_tag': cfg['tag'], 'station': static_row.get('name_unified'),
        'station_idx': station_idx, 'river': static_row.get('river'),
        'lat': static_row.get('lat_unified'), 'lon': static_row.get('lon_unified'),
        'topic': topic, 'scenario': scenario, 'metric': metric, 'param': param,
        'value_nat': value_nat, 'value_reg': value_reg, 'delta': delta, 'ratio': ratio,
        'year_agg': year_agg, 'notes': notes,
    }


def write_long_table(rows, topic, output_dir):
    """Write impact_<topic>_metrics.parquet (+ summary csv) and return the DataFrame."""
    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    os.makedirs(output_dir, exist_ok=True)
    pq = os.path.join(output_dir, f'impact_{topic}_metrics.parquet')
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pq = os.path.join(output_dir, f'impact_{topic}_metrics.csv')
        df.to_csv(pq, index=False)
    summary = os.path.join(output_dir, f'impact_{topic}_summary.csv')
    df.to_csv(summary, index=False)
    return df, pq
