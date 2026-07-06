# -*- coding: utf-8 -*-
"""
REVUB initialise - rewritten for SWAT+ / GRFR / ML pipeline data
Reads station_channel_result.csv, hydro_topology.csv, ml_dataset_hourly.parquet,
and inference_output corrected_all.parquet instead of the original Excel inputs.

Usage:
    Set REGION_PATH, INFERENCE_PATH, GCM, SSP below (or via command-line),
    then run this script followed by B_REVUB_main_code.py.
"""

import numpy as np
import pandas as pd
import sys
import os

# =============================================================================
# ★★★ USER CONFIGURATION ★★★
# =============================================================================
# These can be overridden by command-line args or a wrapper script.
# Example region: /datasets/swat_global/europe_central/austria
REGION_PATH = os.environ.get('REVUB_REGION_PATH', '/datasets/swat_global/europe_central/austria')
INFERENCE_BASE = os.environ.get('REVUB_INFERENCE_BASE', '/datasets/swat_global/inference_output')
GCM = os.environ.get('REVUB_GCM', 'mri-esm2-0')
SSP = os.environ.get('REVUB_SSP', 'ssp370')

# Derive paths
HYDRO_DIR = os.path.join(REGION_PATH, 'hydro')
DATASETS_DIR = os.path.join(REGION_PATH, 'datasets')

# Infer the inference_output sub-path from REGION_PATH
# e.g. /datasets/swat_global/europe_central/austria -> europe_central/austria
_rel = os.path.relpath(REGION_PATH, '/datasets/swat_global')
INFERENCE_PATH = os.path.join(INFERENCE_BASE, _rel, GCM, SSP, 'corrected_all.parquet')

print(f'[REVUB init] Region: {REGION_PATH}')
print(f'[REVUB init] Inference: {INFERENCE_PATH}')
print(f'[REVUB init] GCM={GCM}, SSP={SSP}')


# %% pre.0) Read station data

station_csv = os.path.join(HYDRO_DIR, 'station_channel_result.csv')
df_stations = pd.read_csv(station_csv)

# Filter to active stations (STO, PS, ROR) that have been kept (merge_status != dropped)
if 'merge_status' in df_stations.columns:
    df_stations = df_stations[df_stations['merge_status'] != 'dropped'].copy()

# Filter to stations with valid capacity
df_stations = df_stations[df_stations['capacity_mw'].notna() & (df_stations['capacity_mw'] > 0)].copy()

# Filter to stations with valid GRFR_COMID (needed for flow data)
df_stations = df_stations[df_stations['GRFR_COMID'].notna()].copy()
df_stations['GRFR_COMID'] = df_stations['GRFR_COMID'].astype(int)

df_stations = df_stations.reset_index(drop=True)
HPP_number = len(df_stations)

if HPP_number == 0:
    print('[REVUB init] ERROR: No valid hydropower stations found. Exiting.')
    sys.exit(1)

print(f'[REVUB init] Found {HPP_number} hydropower stations')
print(f'[REVUB init] Types: {df_stations["type_unified"].value_counts().to_dict()}')


# %% pre.0b) Correct GRanD/HydroLAKES mismatch
# When spatial lake matching picked a tiny lake instead of the actual reservoir,
# the bathymetry (max_area_km2, max_vol_km3, coefficients) is wrong.
# Detect by comparing GRanD values (res_area_km2) with used values (max_area_km2).
# Fix: override with GRanD data and recompute power-law coefficients from endpoints.

MISMATCH_RATIO = 0.1  # if used_area / grand_area < this, it's a mismatch
FALLBACK_B_GLOBAL = 1.81
FALLBACK_D_GLOBAL = 2.91

if 'res_area_km2' in df_stations.columns and 'max_area_km2' in df_stations.columns:
    n_corrected = 0
    for i in range(len(df_stations)):
        if df_stations.iloc[i]['type_unified'] not in ('STO', 'PS'):
            continue
        grand_area = pd.to_numeric(df_stations.iloc[i].get('res_area_km2'), errors='coerce')
        used_area = pd.to_numeric(df_stations.iloc[i].get('max_area_km2'), errors='coerce')
        grand_vol = pd.to_numeric(df_stations.iloc[i].get('res_vol_km3'), errors='coerce')

        if not (pd.notna(grand_area) and grand_area > 0 and pd.notna(used_area) and used_area > 0):
            continue
        if used_area / grand_area >= MISMATCH_RATIO:
            continue

        # Mismatch detected - GRanD area is much larger than used area
        dam_height = pd.to_numeric(df_stations.iloc[i].get('dam_height_m'), errors='coerce')
        head_m = pd.to_numeric(df_stations.iloc[i].get('final_head_m'), errors='coerce')
        h_for_bathy = dam_height if pd.notna(dam_height) and dam_height > 0 else (
            head_m if pd.notna(head_m) and head_m > 0 else np.nan)

        if pd.isna(grand_vol) or grand_vol <= 0 or pd.isna(h_for_bathy) or h_for_bathy <= 0:
            continue

        # Override max values with GRanD
        df_stations.at[i, 'max_area_km2'] = grand_area
        df_stations.at[i, 'max_vol_km3'] = grand_vol
        df_stations.at[i, 'max_head_m'] = h_for_bathy

        # Recompute power-law coefficients from endpoints: A = a*h^b, V = c*h^d
        a_new = grand_area / (h_for_bathy ** FALLBACK_B_GLOBAL)
        c_new = grand_vol / (h_for_bathy ** FALLBACK_D_GLOBAL)
        df_stations.at[i, 'coeff_a'] = a_new
        df_stations.at[i, 'coeff_b'] = FALLBACK_B_GLOBAL
        df_stations.at[i, 'coeff_c'] = c_new
        df_stations.at[i, 'coeff_d'] = FALLBACK_D_GLOBAL

        # Also fix ps values (70% of max)
        df_stations.at[i, 'ps_area_km2'] = grand_area * 0.70
        df_stations.at[i, 'ps_vol_km3'] = grand_vol * 0.70

        df_stations.at[i, 'data_source'] = 'GRanD_corrected'

        used_vol = pd.to_numeric(df_stations.iloc[i].get('max_vol_km3'), errors='coerce')
        print(f'  FIX: {df_stations.iloc[i]["name_unified"][:40]:40s} | '
              f'area: {used_area:.3f} -> {grand_area:.1f} km² | '
              f'vol: {used_vol:.6f} -> {grand_vol:.3f} km³')
        n_corrected += 1

    if n_corrected > 0:
        print(f'[REVUB init] Corrected {n_corrected} stations with GRanD reservoir data')


# %% pre.1) Time-related parameters

year_start = 2027
year_end = 2040
simulation_years = list(range(year_start, year_end + 1))

hrs_day = 24
months_yr = 12
secs_hr = 3600
mins_hr = 60

days_year = np.zeros(shape=(months_yr, len(simulation_years)))
hrs_byyear = np.zeros(shape=len(simulation_years))

for y in range(len(simulation_years)):
    yr = simulation_years[y]
    is_leap = (yr % 4 == 0 and yr % 100 != 0) or (yr % 400 == 0)
    if is_leap:
        days_year[:, y] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days_year[:, y] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hrs_byyear[y] = sum(days_year[:, y]) * hrs_day

positions = np.zeros(shape=(len(days_year) + 1, len(simulation_years)))
positions[0, :] = 0
for y in range(len(simulation_years)):
    for n in range(len(days_year)):
        positions[n + 1, y] = hrs_day * days_year[n, y] + positions[n, y]


# %% pre.2) General model parameters

calibration_only = int(os.environ.get('REVUB_CALIBRATION_ONLY', '1'))
option_storage = 0  # global default; per-station PS override below

rho = 1000.0
g = 9.81
p_exceedance = 90
T_fill_thres = 1.0
LOEE_allowed = 0.0
prevent_droughts_increase = 1.0

N_ELCC = 100
X_max = 3
psi_min_threshold = 0.01


# %% pre.3) Static parameters - read from station_channel_result.csv

HPP_name = df_stations['name_unified'].values.astype(str)

# --- Type & active status ---
type_unified = df_stations['type_unified'].values.astype(str)
HPP_active = np.ones(HPP_number, dtype=float)
HPP_active_save = np.ones(HPP_number, dtype=float)

# --- Core parameters from CSV ---
def _get_col(col, default=np.nan):
    """Safely get a column, filling NaN with default."""
    if col in df_stations.columns:
        vals = pd.to_numeric(df_stations[col], errors='coerce').values
        return np.where(np.isnan(vals), default, vals)
    return np.full(HPP_number, default)

V_max = _get_col('max_vol_km3', 0) * 1e9          # km³ → m³
V_max_cumul = V_max.copy()
A_max = _get_col('max_area_km2', 0) * 1e6          # km² → m²
A_max_cumul = A_max.copy()
h_max = _get_col('final_head_m', 0)                 # m
P_r_turb = _get_col('capacity_mw', 0)               # MW

# ROR and Canal stations: force V_max = 0, A_max = 0 (no reservoir)
ror_mask = (type_unified == 'ROR') | (type_unified == 'Canal')
V_max[ror_mask] = 0.0
V_max_cumul[ror_mask] = 0.0
A_max[ror_mask] = 0.0
A_max_cumul[ror_mask] = 0.0

# --- Turbine efficiency by head ---
# Ref: Brekke (2015) Hydraulic Turbines, NTNU; IEA Hydropower Annex
eta_turb = np.where(h_max > 200, 0.87,
           np.where(h_max > 40,  0.88,
                                 0.86))

# --- Q_max_turb: back-calculate from P = η × ρ × g × Q × h / 1e6 ---
# Avoid division by zero for stations with h_max = 0
h_safe = np.where(h_max > 0, h_max, 1.0)
Q_max_turb = np.where(
    h_max > 0,
    P_r_turb * 1e6 / (eta_turb * rho * g * h_safe),
    0.0
)

# --- Turbine count (simplified to 1) ---
no_turbines = np.ones(HPP_number)

# --- min_load_turbine ---
# Ref: Brekke (2015) Hydraulic Turbines, NTNU
min_load_turbine = np.full(HPP_number, 0.25)

# --- f_opt: ps_vol / max_vol, default 0.60 ---
# Ref: Giuliani & Castelletti (2021) WRR; Draper & Lund (2004) ASCE JWRPM
ps_vol = _get_col('ps_vol_km3', np.nan) * 1e9
with np.errstate(divide='ignore', invalid='ignore'):
    f_opt = np.where(
        (V_max > 0) & np.isfinite(ps_vol) & (ps_vol > 0),
        ps_vol / V_max,
        0.60
    )
# Clip: 0.3 (extreme flood-control) to 0.84 (must stay below f_spill=0.85)
f_opt = np.clip(f_opt, 0.3, 0.84)

# --- f_spill: 0.85 ---
# Ref: USACE EM 1110-2-1420 (2018)
f_spill = np.full(HPP_number, 0.85)

# --- f_stop / f_restart ---
# Ref: NIH Roorkee - Determination of Reservoir Storage Capacity
f_stop = np.full(HPP_number, 0.20)
f_stop_cumul = f_stop.copy()
f_restart = np.full(HPP_number, 0.25)
f_restart_cumul = f_restart.copy()

# --- f_initial_frac: use f_opt ---
f_initial_frac = f_opt.copy()

# --- d_min: 0.10 (Tennant minimum ecological flow) ---
# Ref: Tennant (1976) Fisheries 1(4):6-10
d_min = np.full(HPP_number, 0.10)

# --- alpha: 2 (conservative S-curve shape) ---
# Ref: Giuliani & Castelletti (2021) WRR
alpha = np.full(HPP_number, 2.0)

# --- gamma_hydro: NaN (let B code auto-calculate) ---
gamma_hydro = np.full(HPP_number, np.nan)

# --- mu: 0.10 (10% spill safety margin) ---
# Ref: FERC Engineering Guidelines Ch.II (2015)
mu = np.full(HPP_number, 0.10)

# --- Ramp rates ---
# Ref: NREL TP-5500-38153; ESIG Hydropower in a Flexible Grid
dP_ramp_turb = np.full(HPP_number, 0.20)  # 20% rated/min
dP_ramp_pump = np.full(HPP_number, 0.25)  # 25% rated/min

# --- Station latitude (for evaporation estimation and filtering) ---
station_lats = _get_col('lat_unified', 0)

# --- VRE parameters (disabled) ---
c_solar_relative = np.zeros(HPP_number)
c_wind_relative = np.zeros(HPP_number)

# --- Calibration years: use full range ---
year_calibration_start = np.full(HPP_number, float(year_start))
year_calibration_end = np.full(HPP_number, float(year_end))

# --- STOR parameters (for PS stations) ---
V_lower_max = np.zeros(HPP_number)
V_lower_initial_frac = np.full(HPP_number, 0.5)
P_r_pump = np.zeros(HPP_number)
Q_max_pump = np.zeros(HPP_number)
# Ref: Rehman et al. (2021) IOP Prog.Energy 3:032001; DOE PSH Tech Assessment
eta_pump = np.full(HPP_number, 0.90)

# For PS stations, set pump parameters
ps_mask = (type_unified == 'PS')
P_r_pump[ps_mask] = P_r_turb[ps_mask]  # reversible turbine-pump
Q_max_pump[ps_mask] = np.where(
    h_max[ps_mask] > 0,
    P_r_pump[ps_mask] * 1e6 * eta_pump[ps_mask] / (rho * g * h_max[ps_mask]),
    0.0
)

# --- BAL/STOR search parameters (defaults, not used when calibration_only=1) ---
f_init_BAL_start = np.full(HPP_number, 0.0)
f_init_BAL_step = np.full(HPP_number, 0.1)
f_init_BAL_end = np.full(HPP_number, 3.0)
N_refine_BAL = np.full(HPP_number, 2.0)

f_init_STOR_start = np.full(HPP_number, 0.0)
f_init_STOR_step = np.full(HPP_number, 0.1)
f_init_STOR_end = np.full(HPP_number, 3.0)
N_refine_STOR = np.full(HPP_number, 2.0)

f_size = np.full(HPP_number, 90.0)

# --- f_reg: compute from historical flow data ---
# Will be calculated below after loading historical flow


# %% pre.3b) Cascade topology - topological sort for upstream-first simulation
# Q_in_nat from GRFR/inference is fully-routed natural flow at each station.
# For cascade: Q_in[d] = Q_in_nat[d] - sum(Q_in_nat[u]) + sum(Q_out[u])
# i.e., replace upstream natural contribution with upstream regulated outflow.
#
# Only upstream stations with meaningful regulation capacity participate in
# cascade flow routing (cascade_upstream_stations column from hydro_topology.csv).
# Ref: Hanasaki et al. (2006) J. Hydrol. 327:22-41 - C/I ratio criterion
# Ref: Lehner et al. (2011) Front. Ecol. Environ. 9:494-502 - DOR index

HPP_cascade_upstream = np.full(HPP_number, 'nan', dtype=object)
HPP_cascade_downstream = np.full(HPP_number, 'nan', dtype=object)
f_cascade_upstream = np.full(HPP_number, np.nan)

# direct_upstream_indices[i] = list of HPP indices that are CASCADE-ACTIVE upstream of station i
direct_upstream_indices = [[] for _ in range(HPP_number)]
# downstream_index[i] = HPP index of the direct downstream station (-1 if OUTLET)
downstream_index = np.full(HPP_number, -1, dtype=int)

topo_csv = os.environ.get('REVUB_TOPO_FILE', os.path.join(HYDRO_DIR, 'hydro_topology.csv'))
if os.path.exists(topo_csv):
    df_topo = pd.read_csv(topo_csv)

    # Build name → HPP index lookup (only for active/filtered stations)
    _name_to_idx = {}
    for i in range(HPP_number):
        _name_to_idx[HPP_name[i]] = i

    # Parse topology - use cascade_upstream_stations (filtered by f_reg)
    # instead of direct_upstream_stations (all physical connections)
    _cascade_col = 'cascade_upstream_stations'
    if _cascade_col not in df_topo.columns:
        _cascade_col = 'direct_upstream_stations'
        print('[REVUB init] WARNING: no cascade_upstream_stations column, falling back to direct_upstream_stations')

    direct_upstream_tau = [[] for _ in range(HPP_number)]

    for _, trow in df_topo.iterrows():
        sname = trow['station_name']
        if sname not in _name_to_idx:
            continue
        idx = _name_to_idx[sname]

        # downstream
        ds_name = trow.get('downstream_station', 'OUTLET')
        if pd.notna(ds_name) and ds_name != 'OUTLET' and ds_name in _name_to_idx:
            downstream_index[idx] = _name_to_idx[ds_name]

        # cascade upstream: parse "StationA(ch123,hops=4,f_reg=0.5200,qr=0.24,tau=3); StationB(...)"
        # Note: station names may contain parentheses, so split on "(ch" not "("
        us_str = trow.get(_cascade_col, 'NONE')
        if pd.notna(us_str) and us_str != 'NONE':
            for part in str(us_str).split(';'):
                part = part.strip()
                ch_pos = part.find('(ch')
                us_name = part[:ch_pos].strip() if ch_pos > 0 else part.split('(')[0].strip()
                if us_name in _name_to_idx:
                    direct_upstream_indices[idx].append(_name_to_idx[us_name])
                    tau_val = 0
                    if 'tau=' in part:
                        try:
                            tau_val = int(part.split('tau=')[1].rstrip(')'))
                        except ValueError:
                            tau_val = 0
                    direct_upstream_tau[idx].append(tau_val)

    # Topological sort (Kahn's algorithm) - upstream stations first
    in_degree = np.array([len(direct_upstream_indices[i]) for i in range(HPP_number)])
    queue = [i for i in range(HPP_number) if in_degree[i] == 0]
    reorder = []
    while queue:
        node = queue.pop(0)
        reorder.append(node)
        for i in range(HPP_number):
            if node in direct_upstream_indices[i]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
    if len(reorder) != HPP_number:
        missing = set(range(HPP_number)) - set(reorder)
        print(f'[REVUB init] WARNING: topology has cycles, appending {len(missing)} stations')
        reorder.extend(sorted(missing))
    reorder = np.array(reorder)

    n_with_upstream = sum(1 for i in range(HPP_number) if len(direct_upstream_indices[i]) > 0)
    print(f'[REVUB init] Topology loaded: {n_with_upstream} stations have cascade-active upstream')
    print(f'[REVUB init] Simulation order (topological): {[HPP_name[i] for i in reorder[:5]]}...')
else:
    print('[REVUB init] No hydro_topology.csv found - all stations independent')
    reorder = np.arange(HPP_number)
    direct_upstream_tau = [[] for _ in range(HPP_number)]

# Apply reordering to ALL per-station arrays
HPP_name = HPP_name[reorder]
type_unified = type_unified[reorder]
HPP_active_save = HPP_active_save[reorder]
HPP_active = HPP_active[reorder]
V_max = V_max[reorder]
V_max_cumul = V_max_cumul[reorder]
A_max = A_max[reorder]
A_max_cumul = A_max_cumul[reorder]
h_max = h_max[reorder]
P_r_turb = P_r_turb[reorder]
eta_turb = eta_turb[reorder]
Q_max_turb = Q_max_turb[reorder]
no_turbines = no_turbines[reorder]
min_load_turbine = min_load_turbine[reorder]
f_opt = f_opt[reorder]
f_spill = f_spill[reorder]
f_stop = f_stop[reorder]
f_stop_cumul = f_stop_cumul[reorder]
f_restart = f_restart[reorder]
f_restart_cumul = f_restart_cumul[reorder]
f_initial_frac = f_initial_frac[reorder]
d_min = d_min[reorder]
alpha = alpha[reorder]
gamma_hydro = gamma_hydro[reorder]
mu = mu[reorder]
dP_ramp_turb = dP_ramp_turb[reorder]
dP_ramp_pump = dP_ramp_pump[reorder]
c_solar_relative = c_solar_relative[reorder]
c_wind_relative = c_wind_relative[reorder]
year_calibration_start = year_calibration_start[reorder]
year_calibration_end = year_calibration_end[reorder]
V_lower_max = V_lower_max[reorder]
V_lower_initial_frac = V_lower_initial_frac[reorder]
P_r_pump = P_r_pump[reorder]
Q_max_pump = Q_max_pump[reorder]
eta_pump = eta_pump[reorder]
HPP_cascade_upstream = HPP_cascade_upstream[reorder]
HPP_cascade_downstream = HPP_cascade_downstream[reorder]
f_cascade_upstream = f_cascade_upstream[reorder]
f_init_BAL_start = f_init_BAL_start[reorder]
f_init_BAL_step = f_init_BAL_step[reorder]
f_init_BAL_end = f_init_BAL_end[reorder]
N_refine_BAL = N_refine_BAL[reorder]
f_init_STOR_start = f_init_STOR_start[reorder]
f_init_STOR_step = f_init_STOR_step[reorder]
f_init_STOR_end = f_init_STOR_end[reorder]
N_refine_STOR = N_refine_STOR[reorder]
f_size = f_size[reorder]
station_lats = station_lats[reorder]

# Also reorder the station dataframe to keep GRFR_COMID aligned
df_stations = df_stations.iloc[reorder].reset_index(drop=True)

# Remap topology indices from old→new ordering
_old2new = {old_i: new_i for new_i, old_i in enumerate(reorder)}
direct_upstream_indices_new = [[] for _ in range(HPP_number)]
direct_upstream_tau_new = [[] for _ in range(HPP_number)]
downstream_index_new = np.full(HPP_number, -1, dtype=int)
for old_i in range(HPP_number):
    new_i = _old2new[old_i]
    direct_upstream_indices_new[new_i] = [_old2new[u] for u in direct_upstream_indices[old_i]]
    direct_upstream_tau_new[new_i] = list(direct_upstream_tau[old_i])
    if downstream_index[old_i] >= 0:
        downstream_index_new[new_i] = _old2new[downstream_index[old_i]]
direct_upstream_indices = direct_upstream_indices_new
direct_upstream_tau = direct_upstream_tau_new
downstream_index = downstream_index_new

# Recount
HPP_number = len(HPP_name)
HPP_number_run = int(np.sum(HPP_active_save))

# Print cascade chains
n_cascade = sum(1 for i in range(HPP_number) if len(direct_upstream_indices[i]) > 0)
print(f'[REVUB init] {n_cascade} stations have upstream dependencies (cascade enabled)')

# Generate cascade topology diagram
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _trace_chain_from(start, up_idx, ds_idx, n):
        visited = set()
        chain = []
        def _up(h):
            if h in visited: return
            visited.add(h)
            for u in up_idx[h]: _up(u)
            chain.append(h)
        _up(start)
        cur = ds_idx[start]
        while 0 <= cur < n and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            for u in up_idx[cur]:
                if u not in visited: _up(u)
            cur = ds_idx[cur]
        return chain

    chains_all = []
    seen = set()
    for i in range(HPP_number):
        if i in seen or len(direct_upstream_indices[i]) == 0: continue
        root = i
        while True:
            parents = [j for j in range(HPP_number)
                       if root in direct_upstream_indices[j]]
            if not parents: break
            root = parents[0]
        ch = _trace_chain_from(root, direct_upstream_indices, downstream_index, HPP_number)
        if len(ch) >= 2 and not any(h in seen for h in ch):
            chains_all.append(ch)
            seen.update(ch)

    if chains_all:
        n_chains = len(chains_all)
        max_len = max(len(c) for c in chains_all)
        fig_h = max(6, 1.2 * sum(len(c) for c in chains_all))
        fig, ax = plt.subplots(figsize=(14, min(fig_h, 60)))
        ax.axis('off')
        region_name = os.path.basename(REGION_PATH)
        ax.set_title(f'Cascade Topology: {region_name} ({n_chains} chains, {sum(len(c) for c in chains_all)} stations)',
                     fontsize=12, fontweight='bold')

        y_pos = 0
        for ci, ch in enumerate(chains_all):
            ds_groups = {}
            for hpp in ch:
                ds = downstream_index[hpp]
                if ds >= 0: ds_groups.setdefault(ds, []).append(hpp)
            _node_pos = {}
            for si, hpp in enumerate(ch):
                _node_pos[hpp] = (4.0, y_pos - si)
            for ds, parents in ds_groups.items():
                if len(parents) > 1:
                    for pi2, p in enumerate(parents):
                        x_off = -2.0 + pi2 * 4.0 / max(1, len(parents)-1)
                        _, yy = _node_pos[p]
                        _node_pos[p] = (4.0 + x_off, yy)

            for hpp in ch:
                ds = downstream_index[hpp]
                if ds >= 0 and ds in _node_pos:
                    x0, y0 = _node_pos[hpp]; x1, y1 = _node_pos[ds]
                    ax.annotate('', xy=(x1, y1+0.3), xytext=(x0, y0-0.3),
                                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))

            for hpp in ch:
                x, yy = _node_pos[hpp]
                typ = type_unified[hpp] if hpp < len(type_unified) else '?'
                colors = {'PS': ('#E91E63','#880E4F'), 'STO': ('#42A5F5','#1565C0')}
                fc, ec = colors.get(typ, ('#A5D6A7','#2E7D32'))
                cap = P_r_turb[hpp] if hpp < len(P_r_turb) else 0
                lbl = f"{HPP_name[hpp][:28]}\n[{typ}] {cap:.0f}MW"
                bbox = dict(boxstyle='round,pad=0.25', facecolor=fc, edgecolor=ec, lw=1.5, alpha=0.85)
                ax.text(x, yy, lbl, ha='center', va='center', fontsize=6.5, fontweight='bold', bbox=bbox)

            y_pos -= len(ch) + 1.5

        ax.set_xlim(-1, 9)
        ax.set_ylim(y_pos - 1, 2)
        topo_png = os.path.join(REGION_PATH, 'revub_output', 'cascade_topology.png')
        os.makedirs(os.path.dirname(topo_png), exist_ok=True)
        plt.tight_layout()
        plt.savefig(topo_png, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'[REVUB init] Cascade topology diagram saved: {topo_png}')
except Exception as e:
    print(f'[REVUB init] WARNING: could not generate topology diagram: {e}')


# %% pre.3c) Compute f_reg from historical flow data

f_reg = np.full(HPP_number, np.nan)
Q_hist_avg = np.full(HPP_number, np.nan)

ml_parquet = os.path.join(DATASETS_DIR, 'ml_dataset_hourly.parquet')
if os.path.exists(ml_parquet):
    print(f'[REVUB init] Loading historical flow data (using GRFR Qout)...')
    # Use grfr_Qout (GRFR routed flow) instead of flo_out (SWAT+ simulated flow)
    # because SWAT+ flo_out can be severely underestimated at certain comids
    df_hist = pd.read_parquet(ml_parquet, columns=['comid', 'grfr_Qout'])
    comid_means = df_hist.groupby('comid')['grfr_Qout'].mean()
    del df_hist  # free memory

    for n in range(HPP_number):
        comid = df_stations.iloc[n]['GRFR_COMID']
        if comid in comid_means.index:
            Q_avg = comid_means[comid]
            Q_hist_avg[n] = Q_avg
            if V_max[n] > 0 and Q_avg > 0:
                f_reg[n] = V_max[n] / (Q_avg * 365.25 * 24 * 3600)
            elif type_unified[n] == 'ROR':
                f_reg[n] = 0.0
            else:
                f_reg[n] = 0.0
        else:
            print(f'  WARNING: No historical flow for {HPP_name[n]} (COMID={comid})')
            f_reg[n] = 0.0

    print(f'[REVUB init] f_reg computed: min={np.nanmin(f_reg):.4f}, '
          f'max={np.nanmax(f_reg):.4f}, median={np.nanmedian(f_reg):.4f}')
else:
    print(f'[REVUB init] WARNING: {ml_parquet} not found, setting f_reg from V_max estimates')
    f_reg = np.zeros(HPP_number)

# ROR/Canal stations: force f_reg = 0
f_reg[(type_unified == 'ROR') | (type_unified == 'Canal')] = 0.0


# %% pre.3d) Filter out extreme flow-capacity mismatch stations
# Stations where P_theoretical / P_rated < 0.1 (using GRFR flow) are excluded
# because the GRFR snap likely landed on a wrong (smaller) river segment,
# or the head estimate is severely wrong. PS stations are exempt (they don't
# rely on natural inflow).

# IRENA: typical CF 30-80%; <10% implies data/matching error, not real operation
ratio_threshold = 0.1

# NOTE: the former runtime "rescue" block (which overrode final_head_m with the
# raw head_m when SWOT/JRC underestimated the penstock head) has been removed.
# Net head is now determined correctly upstream in 008_final_water_head_selection.py
# (design head > back-calc > SWOT-for-RoR-only, plus web-verified MANUAL_CORRECTIONS),
# so h_max already IS the net head and no downstream patching is needed.

P_theoretical = np.where(
    (Q_hist_avg > 0) & (h_max > 0),
    eta_turb * rho * g * Q_hist_avg * h_max / 1e6,
    0.0
)
with np.errstate(divide='ignore', invalid='ignore'):
    flow_cap_ratio = np.where(P_r_turb > 0, P_theoretical / P_r_turb, np.nan)

exclude_mask = (flow_cap_ratio < ratio_threshold) & (type_unified != 'PS') & np.isfinite(flow_cap_ratio)

# COUNTERFACTUAL (REVUB_EXCLUDE_FUTURE=1): also remove future (commission year > year_start)
# dams, reusing the chain-contraction machinery below. NaN year -> kept (don't drop for missing
# metadata). Removes ALL future stations incl. PS = "as-if-never-built" counterfactual.
# OFF by default -> baseline runs are unchanged/reproducible.
if os.environ.get('REVUB_EXCLUDE_FUTURE', '0') == '1':
    _future_year = pd.to_numeric(df_stations['year_unified'], errors='coerce').values
    if _future_year.shape[0] != exclude_mask.shape[0]:
        raise RuntimeError(f'EXCLUDE_FUTURE alignment error: year {_future_year.shape} vs mask {exclude_mask.shape}')
    _future_mask = _future_year > year_start          # year_start=2027; NaN comparison -> False (kept)
    _n_new = int(np.sum(_future_mask & ~exclude_mask))
    exclude_mask = exclude_mask | _future_mask
    print(f'[REVUB init] REVUB_EXCLUDE_FUTURE=1: removing {int(_future_mask.sum())} future (year>{year_start}) '
          f'stations ({_n_new} beyond flow-cap exclusion)')

n_excluded = int(np.sum(exclude_mask))

if n_excluded > 0:
    print(f'[REVUB init] Excluding {n_excluded} non-PS stations with flow-capacity ratio < {ratio_threshold}:')
    for n in np.where(exclude_mask)[0]:
        print(f'  SKIP: {HPP_name[n][:50]:50s} | {type_unified[n]:5s} | '
              f'{P_r_turb[n]:7.1f} MW | Q_grfr={Q_hist_avg[n]:8.1f} m3/s | ratio={flow_cap_ratio[n]:.4f}')

    keep_mask = ~exclude_mask
    keep_idx = np.where(keep_mask)[0]

    # Preserve the complete graph before per-station arrays and indices are filtered.
    _upstream_before_excl = [list(v) for v in direct_upstream_indices]
    _tau_before_excl = [list(v) for v in direct_upstream_tau]
    _downstream_before_excl = np.asarray(downstream_index, dtype=int).copy()
    _old_station_count = len(keep_mask)

    # Re-filter all per-station arrays
    HPP_name = HPP_name[keep_idx]
    type_unified = type_unified[keep_idx]
    HPP_active = HPP_active[keep_idx]
    HPP_active_save = HPP_active_save[keep_idx]
    V_max = V_max[keep_idx]
    V_max_cumul = V_max_cumul[keep_idx]
    A_max = A_max[keep_idx]
    A_max_cumul = A_max_cumul[keep_idx]
    h_max = h_max[keep_idx]
    P_r_turb = P_r_turb[keep_idx]
    eta_turb = eta_turb[keep_idx]
    Q_max_turb = Q_max_turb[keep_idx]
    no_turbines = no_turbines[keep_idx]
    min_load_turbine = min_load_turbine[keep_idx]
    f_opt = f_opt[keep_idx]
    f_spill = f_spill[keep_idx]
    f_stop = f_stop[keep_idx]
    f_stop_cumul = f_stop_cumul[keep_idx]
    f_restart = f_restart[keep_idx]
    f_restart_cumul = f_restart_cumul[keep_idx]
    f_initial_frac = f_initial_frac[keep_idx]
    d_min = d_min[keep_idx]
    alpha = alpha[keep_idx]
    gamma_hydro = gamma_hydro[keep_idx]
    mu = mu[keep_idx]
    dP_ramp_turb = dP_ramp_turb[keep_idx]
    dP_ramp_pump = dP_ramp_pump[keep_idx]
    c_solar_relative = c_solar_relative[keep_idx]
    c_wind_relative = c_wind_relative[keep_idx]
    year_calibration_start = year_calibration_start[keep_idx]
    year_calibration_end = year_calibration_end[keep_idx]
    V_lower_max = V_lower_max[keep_idx]
    V_lower_initial_frac = V_lower_initial_frac[keep_idx]
    P_r_pump = P_r_pump[keep_idx]
    Q_max_pump = Q_max_pump[keep_idx]
    eta_pump = eta_pump[keep_idx]
    HPP_cascade_upstream = HPP_cascade_upstream[keep_idx]
    HPP_cascade_downstream = HPP_cascade_downstream[keep_idx]
    f_cascade_upstream = f_cascade_upstream[keep_idx]
    f_init_BAL_start = f_init_BAL_start[keep_idx]
    f_init_BAL_step = f_init_BAL_step[keep_idx]
    f_init_BAL_end = f_init_BAL_end[keep_idx]
    N_refine_BAL = N_refine_BAL[keep_idx]
    f_init_STOR_start = f_init_STOR_start[keep_idx]
    f_init_STOR_step = f_init_STOR_step[keep_idx]
    f_init_STOR_end = f_init_STOR_end[keep_idx]
    N_refine_STOR = N_refine_STOR[keep_idx]
    f_size = f_size[keep_idx]
    f_reg = f_reg[keep_idx]
    Q_hist_avg = Q_hist_avg[keep_idx]
    station_lats = station_lats[keep_idx]
    df_stations = df_stations.iloc[keep_idx].reset_index(drop=True)

    HPP_number = len(HPP_name)
    HPP_number_run = int(np.sum(HPP_active_save))

    # ── Remap topology indices after station exclusion ──
    # Bypassing an excluded intermediate station must reconnect the two retained
    # sides of the river; simply dropping either edge would split the cascade.
    _old2new_excl = {old_i: new_i for new_i, old_i in enumerate(keep_idx)}
    _retained_old = set(_old2new_excl)

    def _next_retained_downstream(old_i):
        cur = int(_downstream_before_excl[old_i])
        seen = {old_i}
        while 0 <= cur < _old_station_count and cur not in seen:
            if cur in _retained_old:
                return cur
            seen.add(cur)
            cur = int(_downstream_before_excl[cur])
        return -1

    def _old_edge_tau(old_u, old_v):
        try:
            pos = _upstream_before_excl[old_v].index(old_u)
        except (ValueError, IndexError):
            return 0
        return int(_tau_before_excl[old_v][pos]) if pos < len(_tau_before_excl[old_v]) else 0

    def _path_tau(old_u, old_v):
        total = 0
        cur = old_u
        seen = set()
        while cur != old_v and 0 <= cur < _old_station_count and cur not in seen:
            seen.add(cur)
            nxt = int(_downstream_before_excl[cur])
            if not (0 <= nxt < _old_station_count):
                return total
            total += _old_edge_tau(cur, nxt)
            cur = nxt
        return total

    downstream_index_excl = np.full(HPP_number, -1, dtype=int)
    _reconnected_paths = 0
    for old_i, new_i in _old2new_excl.items():
        target_old = _next_retained_downstream(old_i)
        if target_old >= 0:
            downstream_index_excl[new_i] = _old2new_excl[target_old]
            if int(_downstream_before_excl[old_i]) != target_old:
                _reconnected_paths += 1

    # Collect unique retained-source -> retained-target upstream edges. Edges
    # entering an excluded target continue to that target's next retained node.
    _edge_tau_excl = {}
    for old_target in range(_old_station_count):
        for j, old_source in enumerate(_upstream_before_excl[old_target]):
            if old_source not in _retained_old:
                continue
            retained_target = (old_target if old_target in _retained_old
                               else _next_retained_downstream(old_target))
            if retained_target < 0 or retained_target == old_source:
                continue
            tau = int(_tau_before_excl[old_target][j]) if j < len(_tau_before_excl[old_target]) else 0
            if retained_target != old_target:
                tau += _path_tau(old_target, retained_target)
            key = (_old2new_excl[old_source], _old2new_excl[retained_target])
            _edge_tau_excl[key] = min(_edge_tau_excl.get(key, tau), tau)

    direct_upstream_indices_excl = [[] for _ in range(HPP_number)]
    direct_upstream_tau_excl = [[] for _ in range(HPP_number)]
    for (new_source, new_target), tau in sorted(_edge_tau_excl.items(), key=lambda item: item[0][1::-1]):
        direct_upstream_indices_excl[new_target].append(new_source)
        direct_upstream_tau_excl[new_target].append(tau)
    direct_upstream_indices = direct_upstream_indices_excl
    direct_upstream_tau = direct_upstream_tau_excl
    downstream_index = downstream_index_excl

    # Reconnected active edges can impose a new ordering constraint. Re-sort
    # every station-aligned array so simulation loops remain upstream-first.
    _post_in_degree = np.array([len(v) for v in direct_upstream_indices])
    _post_queue = [i for i in range(HPP_number) if _post_in_degree[i] == 0]
    _post_order = []
    while _post_queue:
        node = _post_queue.pop(0)
        _post_order.append(node)
        for target in range(HPP_number):
            if node in direct_upstream_indices[target]:
                _post_in_degree[target] -= 1
                if _post_in_degree[target] == 0:
                    _post_queue.append(target)
    if len(_post_order) != HPP_number:
        missing = set(range(HPP_number)) - set(_post_order)
        print(f'[REVUB init] WARNING: filtered topology has cycles; appending {len(missing)} stations')
        _post_order.extend(sorted(missing))
    _post_order = np.asarray(_post_order, dtype=int)

    if not np.array_equal(_post_order, np.arange(HPP_number)):
        _station_array_names = (
            'HPP_name', 'type_unified', 'HPP_active', 'HPP_active_save',
            'V_max', 'V_max_cumul', 'A_max', 'A_max_cumul', 'h_max',
            'P_r_turb', 'eta_turb', 'Q_max_turb', 'no_turbines',
            'min_load_turbine', 'f_opt', 'f_spill', 'f_stop',
            'f_stop_cumul', 'f_restart', 'f_restart_cumul',
            'f_initial_frac', 'd_min', 'alpha', 'gamma_hydro', 'mu',
            'dP_ramp_turb', 'dP_ramp_pump', 'c_solar_relative',
            'c_wind_relative', 'year_calibration_start',
            'year_calibration_end', 'V_lower_max', 'V_lower_initial_frac',
            'P_r_pump', 'Q_max_pump', 'eta_pump', 'HPP_cascade_upstream',
            'HPP_cascade_downstream', 'f_cascade_upstream',
            'f_init_BAL_start', 'f_init_BAL_step', 'f_init_BAL_end',
            'N_refine_BAL', 'f_init_STOR_start', 'f_init_STOR_step',
            'f_init_STOR_end', 'N_refine_STOR', 'f_size', 'f_reg',
            'Q_hist_avg', 'station_lats')
        for _array_name in _station_array_names:
            globals()[_array_name] = globals()[_array_name][_post_order]
        df_stations = df_stations.iloc[_post_order].reset_index(drop=True)

        _post_old2new = {old_i: new_i for new_i, old_i in enumerate(_post_order)}
        direct_upstream_indices = [
            [_post_old2new[u] for u in direct_upstream_indices[old_i]]
            for old_i in _post_order]
        direct_upstream_tau = [list(direct_upstream_tau[old_i]) for old_i in _post_order]
        downstream_index = np.array([
            _post_old2new[downstream_index[old_i]] if downstream_index[old_i] >= 0 else -1
            for old_i in _post_order], dtype=int)

    print(f'[REVUB init] After filtering: {HPP_number} stations remain; '
          f'{_reconnected_paths} downstream paths reconnected')


# %% pre.4) Time series - load future flow from inference_output

print(f'[REVUB init] Loading future flow data from inference_output...')

Q_in_nat_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
Q_in_nat_lateral_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
precipitation_flux_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
Q_out_stable_env_irr_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
Q_env_total_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))

# --- Evaporation: estimate from latitude-based climate zone ---
# Net reservoir evaporation (evap - precip on water surface) by climate zone:
#   Tropical (|lat| <= 15°):    ~1200 mm/yr net  (high evap, high precip partially offsets)
#   Subtropical (15° < |lat| <= 30°): ~1500 mm/yr net (high evap, less precip)
#   Warm temperate (30° < |lat| <= 45°): ~600 mm/yr net
#   Cool temperate (45° < |lat| <= 55°): ~300 mm/yr net
#   Boreal (|lat| > 55°):      ~150 mm/yr net
# Ref: Zhao & Gao (2019) Environ. Res. Lett. 14, 124062 - global reservoir evaporation
# Ref: Pokhrel et al. (2021) Nature Rev. Earth & Environ. 2, 579-594
# Note: precipitation on the reservoir surface is already implicitly included in the
# SWAT+ runoff, so we only apply net evaporation (E - P) here.
# Convert: mm/yr → kg/m²/s:  1 mm = 1 kg/m², so mm/yr / (365.25*24*3600)
evaporation_flux_hourly = np.zeros(shape=(int(np.max(positions)), len(simulation_years), HPP_number))

for n in range(HPP_number):
    abs_lat = abs(station_lats[n])
    if abs_lat <= 15:
        net_evap_mm_yr = 1200.0
    elif abs_lat <= 30:
        net_evap_mm_yr = 1500.0
    elif abs_lat <= 45:
        net_evap_mm_yr = 600.0
    elif abs_lat <= 55:
        net_evap_mm_yr = 300.0
    else:
        net_evap_mm_yr = 150.0
    evap_flux = net_evap_mm_yr / (365.25 * 24 * 3600)  # kg/m²/s
    evaporation_flux_hourly[:, :, n] = evap_flux

print(f'[REVUB init] Evaporation estimated by latitude for {HPP_number} stations')

# VRE dummy arrays (disabled)
CF_solar_hourly = np.ones(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
CF_wind_hourly = np.ones(shape=(int(np.max(positions)), len(simulation_years), HPP_number))
L_norm = np.ones(shape=(int(np.max(positions)), len(simulation_years), HPP_number))

# VRE corrector: 0 = no VRE
c_VRE_corrector = np.zeros(HPP_number)

# Apply PS river relocation: swap GRFR_COMID for relocated PS stations
_ps_reloc_path = os.path.join(HYDRO_DIR, 'ps_relocation.csv')
if os.path.exists(_ps_reloc_path):
    _reloc = pd.read_csv(_ps_reloc_path)
    _n_reloc = 0
    for _, _rr in _reloc.iterrows():
        if pd.isna(_rr.get('new_comid')): continue
        _mask = df_stations['name_unified'] == _rr['name']
        if _mask.any():
            _old = df_stations.loc[_mask, 'GRFR_COMID'].values[0]
            df_stations.loc[_mask, 'GRFR_COMID'] = int(_rr['new_comid'])
            if pd.notna(_rr.get('new_lat')) and 'lat_unified' in df_stations.columns:
                df_stations.loc[_mask, 'lat_unified'] = _rr['new_lat']
                df_stations.loc[_mask, 'lon_unified'] = _rr['new_lon']
            _n_reloc += 1
    if _n_reloc > 0:
        print(f'[REVUB init] PS river relocation: {_n_reloc} stations updated with new GRFR_COMID')

if os.path.exists(INFERENCE_PATH):
    df_flow = pd.read_parquet(INFERENCE_PATH)
    df_flow['datetime'] = pd.to_datetime(df_flow['datetime'])
    df_flow['year'] = df_flow['datetime'].dt.year

    # Get unique COMIDs we need (includes relocated PS COMIDs)
    needed_comids = set(df_stations['GRFR_COMID'].values)
    df_flow = df_flow[df_flow['comid'].isin(needed_comids)].copy()

    print(f'[REVUB init] Flow data: {len(df_flow)} rows, '
          f'{df_flow["comid"].nunique()} matching stations')

    for n in range(HPP_number):
        comid = df_stations.iloc[n]['GRFR_COMID']
        df_stn = df_flow[df_flow['comid'] == comid].copy()

        if len(df_stn) == 0:
            print(f'  WARNING: No flow data for {HPP_name[n]} (COMID={comid})')
            continue

        df_stn = df_stn.sort_values('datetime')

        for y_idx, yr in enumerate(simulation_years):
            df_yr = df_stn[df_stn['year'] == yr]
            if len(df_yr) == 0:
                continue

            n_hrs = int(hrs_byyear[y_idx])
            # Place each flow at its TRUE hour-of-year (timestamp-aligned), NOT by record position -
            # otherwise a gap (e.g. missing first 2 days) shifts the whole year's flow forward in time.
            hoy = ((df_yr['datetime'].dt.dayofyear - 1) * 24 + df_yr['datetime'].dt.hour).values.astype(int)
            flow_vals = df_yr['flo_corrected'].values
            m = (hoy >= 0) & (hoy < n_hrs)
            col = np.full(n_hrs, np.nan)
            col[hoy[m]] = flow_vals[m]
            # Fill any unfilled (missing-timestamp) hours with the year mean of available data.
            if np.any(np.isnan(col)) and np.any(~np.isnan(col)):
                col[np.isnan(col)] = np.nanmean(col)
            Q_in_nat_hourly[:n_hrs, y_idx, n] = np.nan_to_num(col)

    # Replace any negative flows with 0
    Q_in_nat_hourly[~np.isfinite(Q_in_nat_hourly)] = 0.0
    Q_in_nat_hourly[Q_in_nat_hourly < 0] = 0.0

    # Tennant minimum-flow target: 10% (d_min) of the long-term mean flow, not 10% of each hour.
    # The stable-release portion is computed later from the actual cascade-adjusted RoR pass-through.
    for n in range(HPP_number):
        _q_ref = Q_hist_avg[n] if np.isfinite(Q_hist_avg[n]) and Q_hist_avg[n] > 0 else np.nanmean(Q_in_nat_hourly[:, :, n])
        if not np.isfinite(_q_ref) or _q_ref < 0: _q_ref = 0.0
        for y_idx in range(len(simulation_years)):
            Q_env_total_hourly[:int(hrs_byyear[y_idx]), y_idx, n] = d_min[n] * _q_ref

    del df_flow
    print(f'[REVUB init] Flow data loaded successfully; environmental release targets initialized')
else:
    print(f'[REVUB init] ERROR: Inference file not found: {INFERENCE_PATH}')
    print(f'[REVUB init] Q_in_nat_hourly will be all zeros!')


# %% pre.5) Simulation accuracy (BAL/STOR defaults, not used when calibration_only=1)
# Already set above in pre.3


# %% pre.6) Bathymetry - power-law coefficients per station
# A(h) = a * h^b  (km²),  V(h) = c * h^d  (km³)
# B code uses the inverse:  h(V) = (V_m3 / (c * 1e9))^(1/d)
#                            A(V) = a * h^b * 1e6  (m²)

print(f'[REVUB init] Generating bathymetry coefficients...')

FALLBACK_B = 1.81
FALLBACK_D = 2.91

bathy_a = np.full(HPP_number, np.nan)
bathy_b = np.full(HPP_number, np.nan)
bathy_c = np.full(HPP_number, np.nan)
bathy_d = np.full(HPP_number, np.nan)

n_bathy_rescaled = 0
for n in range(HPP_number):
    if V_max[n] == 0:
        continue

    ca = pd.to_numeric(df_stations.iloc[n].get('coeff_a', np.nan), errors='coerce')
    cb = pd.to_numeric(df_stations.iloc[n].get('coeff_b', np.nan), errors='coerce')
    cc = pd.to_numeric(df_stations.iloc[n].get('coeff_c', np.nan), errors='coerce')
    cd = pd.to_numeric(df_stations.iloc[n].get('coeff_d', np.nan), errors='coerce')

    if pd.notna(ca) and pd.notna(cb) and pd.notna(cc) and pd.notna(cd) and ca > 0 and cc > 0:
        bathy_b[n] = cb
        bathy_d[n] = cd
        # Rescale coeff_c and coeff_a to anchor at h_max (= final_head_m). The CSV
        # coeff_c/coeff_a were computed by 009 from head_m, which can differ from
        # final_head_m; re-anchoring keeps the V(h)/A(h) curves consistent with the
        # net head actually used by REVUB.
        h_end = h_max[n] if h_max[n] > 0 else 1.0
        V_max_km3 = V_max[n] / 1e9
        A_max_km2 = A_max[n] / 1e6
        cc_new = V_max_km3 / (h_end ** cd)
        ca_new = A_max_km2 / (h_end ** cb) if A_max_km2 > 0 else 0.0
        if abs(cc_new - cc) / max(cc, 1e-30) > 0.01:
            n_bathy_rescaled += 1
        bathy_c[n] = cc_new
        bathy_a[n] = ca_new
    else:
        h_end = h_max[n] if h_max[n] > 0 else 1.0
        bathy_b[n] = FALLBACK_B
        bathy_d[n] = FALLBACK_D
        bathy_c[n] = (V_max[n] / 1e9) / (h_end ** FALLBACK_D)
        bathy_a[n] = (A_max[n] / 1e6) / (h_end ** FALLBACK_B) if A_max[n] > 0 else 0.0
        if V_max[n] > 0:
            print(f'  INFO: Using fallback power-law (b={FALLBACK_B}, d={FALLBACK_D}) for {HPP_name[n]}')

if n_bathy_rescaled > 0:
    print(f'[REVUB init] Rescaled bathymetry coefficients for {n_bathy_rescaled} stations (anchored to h_max)')

print(f'[REVUB init] Bathymetry coefficients set for {HPP_number} stations')


# %% pre.7) Summary

print(f'\n{"="*60}')
print(f'REVUB Initialisation Summary')
print(f'{"="*60}')
print(f'Region:          {REGION_PATH}')
print(f'GCM / SSP:       {GCM} / {SSP}')
print(f'Simulation:      {year_start}-{year_end} ({len(simulation_years)} years)')
print(f'Stations:        {HPP_number} total')
print(f'  STO:           {int(np.sum(type_unified == "STO"))}')
print(f'  ROR:           {int(np.sum(type_unified == "ROR"))}')
print(f'  PS:            {int(np.sum(type_unified == "PS"))}')
print(f'Mode:            {"CONV only" if calibration_only else "CONV + BAL + STOR"}')
print(f'{"="*60}\n')

# List stations
for n in range(HPP_number):
    cat = 'A(large)' if f_reg[n] >= 1 else ('B(small)' if f_reg[n] > 0 else 'RoR')
    print(f'  [{n:3d}] {HPP_name[n][:50]:50s} | {type_unified[n]:3s} | '
          f'{P_r_turb[n]:7.1f} MW | h={h_max[n]:6.1f}m | '
          f'f_reg={f_reg[n]:.4f} ({cat})')
