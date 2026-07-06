#!/usr/bin/env python3
"""
D_REVUB_DC_dispatch.py - Hydro-powered Data Center assessment (3 scenarios)

Reuses B+C logic directly (no reimplementation). Each station gets its own
L_norm based on whether it's matched to a DC or serving the grid.

Scenario A: Independent station dispatch, no PS
Scenario B: Independent + PS per-pair
Scenario C: Chain-level PS + closed-loop cascade propagation

Run AFTER A + B + C files (exec'd in same scope).
"""

import os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

DC_RESULTS = '/home/cfeng/hydro/dc_load_simulation/results'
DC_MAPPING = '/home/cfeng/hydro/dc_load_simulation/dc_to_swat_region.csv'
E_PROFILE_OVERRIDE = os.environ.get('REVUB_E_DC_PROFILE_OVERRIDE', '').strip()
D_SUMMARY_ONLY = os.environ.get('REVUB_D_SUMMARY_ONLY', '0') == '1'

print(f'\n{"="*60}')
print(f'D: Hydro → Data Center Assessment (3 Scenarios)')
print(f'{"="*60}')

import pandas as pd
from math import radians, sin, cos, sqrt, atan2

dc_gcm = os.environ.get('REVUB_DC_GCM', GCM)
dc_ssp = os.environ.get('REVUB_DC_SSP', SSP)
region_name = os.path.basename(REGION_PATH)
_scenario_tag = os.environ.get('REVUB_SCENARIO_TAG', '').strip()
_d_output_override = os.environ.get('REVUB_D_OUTPUT_DIR_OVERRIDE', '').strip()
OUTPUT_DIR_D = (_d_output_override if _d_output_override else
                os.path.join(REGION_PATH, 'revub_output', GCM, SSP))
if _scenario_tag and not _d_output_override:
    OUTPUT_DIR_D = os.path.join(OUTPUT_DIR_D, _scenario_tag)
os.makedirs(OUTPUT_DIR_D, exist_ok=True)

# L_norm_grid = the TYPICAL regional load curve. Energy NOT serving the data center (the 1-frac
# residual of matched stations + stations not matched to any DC) serves the regional grid, which
# has the TYPICAL demand shape - so it must be valued against the typical curve, NOT the flat curve
# used upstream for the matching reference. Load the typical profile directly here (decoupled from
# REVUB_LOAD_TYPE=flat, which only sets the matching-reference ELCC_indep). Fall back to L_norm if
# no typical profile exists for this region.
_grid_lp_path = os.path.join(DATASETS_DIR, 'load_profile_hourly.parquet')
if os.path.exists(_grid_lp_path):
    _grid_lp = pd.read_parquet(_grid_lp_path)
    L_norm_grid = np.full([_T, _Y], np.nan)
    for _yg in range(_Y):
        _yd = _grid_lp[_grid_lp['year'] == simulation_years[_yg]]['L_norm'].values
        _ng = min(int(hrs_byyear[_yg]), len(_yd))
        if _ng > 0:
            L_norm_grid[:_ng, _yg] = _yd[:_ng]
    print(f'  L_norm_grid: typical regional curve ({_grid_lp_path}, mean={np.nanmean(L_norm_grid):.3f})')
else:
    L_norm_grid = L_norm.copy()
    print(f'  L_norm_grid: no typical profile - falling back to upstream L_norm (flat)')

print(f'  Region: {region_name}, Hydro: {GCM}/{SSP}, DC: {dc_gcm}/{dc_ssp}')

# ═══════════════════════════════════════════════════════════
# D.0) Build L_norm_DC from 14 years of DC hourly data
# ═══════════════════════════════════════════════════════════

dc_key = f'{region_name}__P_total_kw'
L_norm_DC = np.full([_T, _Y], np.nan)
P_DC_MW_all = []
has_dc = True

if E_PROFILE_OVERRIDE:
    if not os.path.exists(E_PROFILE_OVERRIDE):
        raise FileNotFoundError(f'E cross-region DC profile not found: {E_PROFILE_OVERRIDE}')
    with np.load(E_PROFILE_OVERRIDE, allow_pickle=True) as _e_profile:
        if 'P_target_mw' not in _e_profile:
            raise KeyError(f'{E_PROFILE_OVERRIDE} lacks P_target_mw')
        _target = np.asarray(_e_profile['P_target_mw'], dtype=np.float64)
        _years = (np.asarray(_e_profile['simulation_years'], dtype=int).tolist()
                  if 'simulation_years' in _e_profile else list(simulation_years))
    if _target.ndim != 2:
        raise ValueError(f'{E_PROFILE_OVERRIDE}: P_target_mw must be [hour, year]')
    if _years != list(simulation_years) or _target.shape[1] != _Y:
        raise ValueError(
            f'{E_PROFILE_OVERRIDE}: years/shape do not match {simulation_years[0]}-'
            f'{simulation_years[-1]}')
    for y_idx in range(_Y):
        nhrs_y = int(hrs_byyear[y_idx])
        if _target.shape[0] < nhrs_y:
            raise ValueError(
                f'{E_PROFILE_OVERRIDE}: only {_target.shape[0]} hours for '
                f'{simulation_years[y_idx]} ({nhrs_y} required)')
        P_DC_MW_all.append(_target[:nhrs_y, y_idx].copy())
    print(f'  E override profile: {E_PROFILE_OVERRIDE}')
else:
    for y_idx in range(_Y):
        yr = simulation_years[y_idx]
        nhrs_y = int(hrs_byyear[y_idx])
        dc_path = os.path.join(DC_RESULTS, f'region_hourly_{dc_gcm}_{dc_ssp}_{yr}.npz')
        if not os.path.exists(dc_path):
            has_dc = False; break
        dc_data = np.load(dc_path, allow_pickle=True)
        if dc_key not in dc_data:
            has_dc = False; break
        p_mw = dc_data[dc_key].astype(np.float64) / 1e3
        n_use = min(nhrs_y, len(p_mw))
        P_DC_MW_all.append(p_mw[:n_use])

if has_dc:
    # Fix #1: guard against a degenerate/empty DC load series. A zero or NaN peak would make
    # L_norm_DC = src/peak = NaN everywhere, silently collapsing ALL station ELCC to 0 while
    # still writing a "successful" zero-supply result. Fail-loud instead.
    _dc_maxes = [np.max(p) for p in P_DC_MW_all if len(p) > 0]
    dc_peak = max(_dc_maxes) if _dc_maxes else 0.0
    if not np.isfinite(dc_peak) or dc_peak <= 0:
        print(f'  DC load for {region_name} is zero/invalid (peak={dc_peak}); treating as no DC.')
        has_dc = False

if not has_dc:
    print(f'  No DC data for {region_name}. Skipping.')
else:
    for y_idx in range(_Y):
        nh = int(hrs_byyear[y_idx])
        src = P_DC_MW_all[y_idx]
        # Cover the FULL year length (leap years = 8784h); wrap the profile cyclically if the
        # DC series is shorter, so no trailing NaN hours dilute the outage fraction in the kernels.
        if len(src) >= nh:
            L_norm_DC[:nh, y_idx] = src[:nh] / dc_peak
        else:
            L_norm_DC[:nh, y_idx] = np.resize(src, nh) / dc_peak
    dc_mean = np.mean(np.concatenate(P_DC_MW_all))

    if 'station_lons' not in dir():
        _lc = 'lon_unified' if 'lon_unified' in df_stations.columns else 'lon'
        # Fix #4: refuse to silently zero-fill longitudes (would place every station at lon=0 and
        # corrupt haversine matching + the E-stage coordinate export). Fail-loud instead.
        if _lc not in df_stations.columns:
            raise KeyError(f'station table for {region_name} has no lon_unified/lon column - '
                           f'cannot place stations (refusing to silently use lon=0)')
        station_lons = df_stations[_lc].values.astype(float)

    if 'station_lats' not in dir():
        _latc = 'lat_unified' if 'lat_unified' in df_stations.columns else 'lat'
        # BUG1: mirror the station_lons guard - never silently zero-fill latitudes (would place every
        # station at lat=0 and corrupt haversine matching + the E-stage coordinate export). Fail-loud.
        if _latc not in df_stations.columns:
            raise KeyError(f'station table for {region_name} has no lat_unified/lat column - '
                           f'cannot place stations (refusing to silently use lat=0)')
        station_lats = df_stations[_latc].values.astype(float)

    if E_PROFILE_OVERRIDE:
        _virtual_id = f'E_CROSS_REGION_{region_name}'
        _virtual_lat = float(np.nanmedian(station_lats))
        _virtual_lon = float(np.nanmedian(station_lons))
        dc_in_region = pd.DataFrame([{
            'Datacenter ID': _virtual_id,
            'p_total_peak_kw': dc_peak * 1e3,
        }])
        dc_coords = pd.DataFrame([{
            'Datacenter ID': _virtual_id,
            'Latitude': _virtual_lat,
            'Longitude': _virtual_lon,
        }]).set_index('Datacenter ID')
        n_dc = 1
    else:
        dc_mapping_df = pd.read_csv(DC_MAPPING)
        region_dc_ids = set(
            dc_mapping_df[dc_mapping_df['swat_region'] == region_name]['Datacenter ID'])
        dc_coords = dc_mapping_df[
            dc_mapping_df['swat_region'] == region_name].set_index('Datacenter ID')
        mid_year = simulation_years[_Y // 2]
        dc_detail_path = os.path.join(
            DC_RESULTS, f'{dc_gcm}_{dc_ssp}_{mid_year}_summary.csv')
        dc_detail = pd.read_csv(dc_detail_path) if os.path.exists(dc_detail_path) else None
        # Fix #2: a missing demand summary silently yields n_dc=0 → zero supply that looks successful.
        if dc_detail is None:
            print(f'  WARNING: DC demand summary missing ({dc_detail_path}); n_dc=0 → supply will be 0. '
                  f'Result INCOMPLETE for {region_name}.')
        dc_in_region = (dc_detail[dc_detail['Datacenter ID'].isin(region_dc_ids)].copy()
                        if dc_detail is not None else pd.DataFrame())
        n_dc = len(dc_in_region)

    print(f'  DC: {n_dc} centers, peak={dc_peak:.0f}MW, mean={dc_mean:.0f}MW, '
          f'flatness={np.std(np.concatenate(P_DC_MW_all))/dc_mean:.3f}')

    # ═══════════════════════════════════════════════════════════
    # D.0b) Greedy matching (flat ELCC reference) → determine per-station L_norm
    # ═══════════════════════════════════════════════════════════

    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371; dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _greedy_match_flat(dc_df, elcc_arr):
        """Greedy nearest-first matching using flat ELCC. Returns set of matched station indices."""
        units = []
        for i in range(HPP_number):
            if elcc_arr[i] > 0.1 and type_unified[i] != 'PS':
                units.append({'idx': i, 'elcc': elcc_arr[i], 'remaining': elcc_arr[i],
                              'lat': station_lats[i], 'lon': station_lons[i]})
        matched = set()
        alloc = []
        for _, dc in dc_df.iterrows():
            dc_id = dc['Datacenter ID']
            demand = dc['p_total_peak_kw'] / 1e3; remaining = demand
            dc_lat = dc_coords.loc[dc_id, 'Latitude'] if dc_id in dc_coords.index else 0
            dc_lon = dc_coords.loc[dc_id, 'Longitude'] if dc_id in dc_coords.index else 0
            dists = [(i, _haversine(dc_lat, dc_lon, u['lat'], u['lon']))
                     for i, u in enumerate(units) if u['remaining'] > 0.01]
            dists.sort(key=lambda x: x[1])
            for ui, dist in dists:
                if remaining < 0.01: break
                give = min(remaining, units[ui]['remaining'])
                units[ui]['remaining'] -= give; remaining -= give
                matched.add(units[ui]['idx'])
                alloc.append({'dc_id': dc_id, 'demand_mw': demand,
                              'hpp_idx': units[ui]['idx'],
                              'unit_name': HPP_name[units[ui]['idx']][:30],
                              'allocated_mw': give, 'distance_km': dist})
        return matched, pd.DataFrame(alloc)

    # Use flat ELCC_indep for matching
    flat_elcc = ELCC_indep if 'ELCC_indep' in dir() else ELCC_opt
    matched_stations, alloc_flat = _greedy_match_flat(dc_in_region, flat_elcc)
    # Per-station DC demand assigned by matching (demand-side)
    dc_alloc = alloc_flat.groupby('hpp_idx')['allocated_mw'].sum().to_dict() if len(alloc_flat) > 0 else {}
    # Split each station: frac of its capacity serves DC (DC-curve ELCC), rest serves grid (grid-curve ELCC).
    # frac = DC demand assigned / station's grid-ELCC capacity (the matching reference); naturally in [0,1].
    frac_dc = {i: min(dc_alloc[i] / flat_elcc[i], 1.0) for i in dc_alloc if flat_elcc[i] > 0}
    print(f'  Matched {len(matched_stations)} stations to {n_dc} DCs '
          f'(total DC demand allocated={sum(dc_alloc.values()):.0f} MW)')

    def _resid_grid_elcc(P_s, P_f, P_r, idx):
        """Residual (1-frac) of a matched station's DC-curve output, re-valued as ELCC vs the grid load template."""
        f = frac_dc.get(idx, 0.0)
        resid = (1.0 - f) * (np.nan_to_num(P_s[:, :, idx]) + np.nan_to_num(P_f[:, :, idx]) + np.nan_to_num(P_r[:, :, idx]))
        ratio = resid / L_norm_grid
        rv = ratio[np.isfinite(ratio) & (L_norm_grid > 0)]
        return max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0

    def _grid_portfolio_elcc(P_s, P_f, P_r, ps_gen=None, ps_pump=None, ps_grid_share=None,
                             station_grid_share=None):
        """Firm capacity of the aggregate grid-facing portfolio, including PS net power."""
        portfolio = np.zeros((_T, _Y))
        for idx in range(HPP_number):
            if type_unified[idx] == 'PS':
                continue
            share = station_grid_share[idx] if station_grid_share is not None else 1.0 - frac_dc.get(idx, 0.0)
            portfolio += share * (np.nan_to_num(P_s[:, :, idx])
                                  + np.nan_to_num(P_f[:, :, idx])
                                  + np.nan_to_num(P_r[:, :, idx]))
        if ps_gen is not None and ps_pump is not None:
            if ps_grid_share is None: ps_grid_share = np.ones(ps_gen.shape[2])
            portfolio += np.nansum((np.nan_to_num(ps_gen) - np.nan_to_num(ps_pump))
                                   * np.asarray(ps_grid_share)[None, None, :], axis=2)
        ratio = portfolio / L_norm_grid
        rv = ratio[np.isfinite(ratio) & (L_norm_grid > 0)]
        return max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0

    # ═══════════════════════════════════════════════════════════
    # Core: cascade Q_in helper
    # ═══════════════════════════════════════════════════════════

    def _cascade_qin(HPP, Q_out_arr):
        return _cascade_inflow(HPP, Q_out_arr)

    # ═══════════════════════════════════════════════════════════
    # Core: BAL dispatch for all stations (mirrors B file BAL loop)
    # ═══════════════════════════════════════════════════════════

    def _run_bal_all(matched_set):
        """Run BAL dispatch for all stations. Matched stations use L_norm_DC, others use L_norm_grid."""
        ELCC = np.zeros(HPP_number)
        Q_out = np.zeros_like(Q_BAL_out_hourly)
        V = np.zeros_like(V_BAL_hourly)
        P_s = np.zeros_like(P_BAL_hydro_stable_hourly)
        P_f = np.zeros_like(P_BAL_hydro_flexible_hourly)
        P_r = np.zeros_like(P_BAL_hydro_RoR_hourly)

        for HPP in range(HPP_number):
            q_in = _cascade_qin(HPP, Q_out)
            l_norm_hpp = L_norm_DC if HPP in matched_set else L_norm_grid
            l_mean = np.nanmean(l_norm_hpp)

            if type_unified[HPP] == 'PS':
                Q_out[:, :, HPP] = q_in; continue

            if HPP_category[HPP] == 'RoR':
                Q_out[:, :, HPP] = q_in
                P_r[:, :, HPP] = np.fmin(
                    np.fmin(q_in, Q_max_turb[HPP]) * eta_turb[HPP] * rho * g * h_max[HPP] / 1e6,
                    P_r_turb[HPP])
                ratio = P_r[:, :, HPP] / l_norm_hpp
                rv = ratio[np.isfinite(ratio) & (l_norm_hpp > 0)]
                ELCC[HPP] = max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0
                continue

            # STO: binary search - same params as B file
            c_or = C_OR_opt[HPP] if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[HPP]) else min(0.30, max(0.05, 1.0 - d_min[HPP]))
            P_avg = np.nanmean(P_CONV_hydro_stable_hourly[:, :, HPP] + P_CONV_hydro_RoR_hourly[:, :, HPP])
            # BUG3: l_mean (nanmean over an all-NaN L_norm) and hence elcc_hi can be NaN; a NaN hi makes
            # the binary-search interval test (hi-lo<0.1) never True → 200 iters of fail-open garbage.
            elcc_hi = P_avg / l_mean if np.isfinite(l_mean) and l_mean > 0 else P_avg
            if not np.isfinite(elcc_hi) or elcc_hi <= 0:
                ELCC[HPP] = 0.0; continue
            if HPP_category[HPP] == 'A':
                Qfr = q_in.copy(); Qrr = np.zeros_like(Qfr)
            else:
                Qfr = f_reg[HPP] * q_in.copy(); Qrr = q_in - Qfr

            lo, hi, best = 0.0, elcc_hi, 0.0
            for _ in range(200):
                if hi - lo < 0.1: break
                mid = (lo + hi) / 2
                res = _bal_core(_Y, hrs_byyear, c_or, mid, Qfr.copy(), Qrr.copy(),
                    Q_CONV_stable_hourly[:, :, HPP], _stable_env_release(HPP, Qrr),
                    precipitation_flux_hourly[:, :, HPP], evaporation_flux_hourly[:, :, HPP],
                    l_norm_hpp, V_CONV_hourly[0, 0, HPP], V_max_cumul[HPP], Q_max_turb[HPP],
                    P_r_turb[HPP], eta_turb[HPP], rho, g, secs_hr,
                    f_cascade_downstream[HPP], bathy_c[HPP], bathy_d[HPP],
                    bathy_a[HPP], bathy_b[HPP],
                    f_spill[HPP], f_stop_cumul[HPP], f_restart_cumul[HPP], mu[HPP],
                    HPP_category[HPP] == 'B', _T)
                if res[-1] <= OUTAGE_MAX:
                    best = mid; lo = mid
                    Q_out[:, :, HPP] = res[6]; V[:, :, HPP] = res[0]
                    P_s[:, :, HPP] = res[7]; P_f[:, :, HPP] = res[8]; P_r[:, :, HPP] = res[9]
                else:
                    hi = mid
            ELCC[HPP] = best

        return ELCC, Q_out, V, P_s, P_f, P_r

    # ═══════════════════════════════════════════════════════════
    # D.1) SCENARIO A: Independent dispatch, no PS
    # ═══════════════════════════════════════════════════════════

    print(f'\n  Scenario A: Independent, no PS')
    t_a = time.time()
    ELCC_A, Q_out_A, V_A, P_s_A, P_f_A, P_r_A = _run_bal_all(matched_stations)
    # supply = DC-share (ELCC_A x frac, DC curve); grid = residual (1-frac) output re-valued vs grid curve
    supply_A = sum(ELCC_A[i] * frac_dc.get(i, 0.0)
                   for i in range(HPP_number) if i in matched_stations and np.isfinite(ELCC_A[i]))
    # grid = TOTAL regional-grid supply: matched stations' (1-frac) residual + ALL non-matched
    # stations' full output, valued against the typical grid curve (not just matched residual).
    grid_A = _grid_portfolio_elcc(P_s_A, P_f_A, P_r_A)
    print(f'    Scenario A: supply→DC={supply_A:.0f} MW, residual→grid={grid_A:.0f} MW ({time.time()-t_a:.0f}s)')

    # ═══════════════════════════════════════════════════════════
    # D.2) SCENARIO B: Independent + PS per-pair
    # ═══════════════════════════════════════════════════════════

    print(f'  Scenario B: Independent + PS per-pair')
    t_b = time.time()

    ELCC_B = ELCC_A.copy()
    Q_out_B = Q_out_A.copy(); V_B = V_A.copy()
    P_s_B = P_s_A.copy(); P_f_B = P_f_A.copy(); P_r_B = P_r_A.copy()

    # C exits before defining its pairing structures when a region has no PS.
    # Treat that valid case as an empty optional PS fleet throughout D.
    if 'ps_paired' not in dir():
        ps_paired = []
    if 'ps_anchor' not in dir():
        ps_anchor = {}
    n_ps = len(ps_paired)
    P_PS_gen_B = np.zeros((_T, _Y, max(n_ps, 1)))
    P_PS_pump_B = np.zeros((_T, _Y, max(n_ps, 1)))
    V_PS_upper_B = np.zeros((_T + 1, _Y, max(n_ps, 1)))
    PS_grid_share_B = np.ones(max(n_ps, 1))

    def _ndown_topo(h):
        n = 0; cur = downstream_index[h]; seen = 0
        while 0 <= cur < HPP_number and seen <= HPP_number:
            n += 1; cur = downstream_index[cur]; seen += 1
        return n

    _ps_net_flow_B = {}
    _paired_anchor_B = {ps_anchor[ps] for ps in ps_paired} if n_ps > 0 else set()

    def _affected_downstream_B(*sources):
        affected = set()
        for source in sources:
            ds = downstream_index[source]
            seen = 0
            while 0 <= ds < HPP_number and seen <= HPP_number:
                affected.add(ds)
                ds = downstream_index[ds]
                seen += 1
        return sorted(affected, key=_ndown_topo, reverse=True)

    def _propagate_group_B(anchor, ps_list):
        """Route a complete anchor group without overwriting another coupled group."""
        for ds in _affected_downstream_B(anchor, *ps_list):
            # Paired anchors are updated by their joint STO+PS solve in the current or next sweep.
            if ds in _paired_anchor_B:
                continue
            q_in_ds = _cascade_qin(ds, Q_out_B)
            l_norm_ds = L_norm_DC if ds in matched_stations else L_norm_grid
            if type_unified[ds] == 'PS':
                ps_net = _ps_net_flow_B.get(ds)
                Q_out_B[:, :, ds] = (np.maximum(q_in_ds + ps_net, 0.0)
                                     if ps_net is not None else q_in_ds)
            elif HPP_category[ds] == 'RoR':
                Q_out_B[:, :, ds] = q_in_ds
                P_r_B[:, :, ds] = np.fmin(
                    np.fmin(q_in_ds, Q_max_turb[ds]) * eta_turb[ds] * rho * g * h_max[ds] / 1e6,
                    P_r_turb[ds])
            else:
                c_or_ds = (C_OR_opt[ds] if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[ds])
                           else min(0.30, max(0.05, 1.0 - d_min[ds])))
                if HPP_category[ds] == 'A':
                    Qfr_ds = q_in_ds.copy(); Qrr_ds = np.zeros_like(Qfr_ds)
                else:
                    Qfr_ds = f_reg[ds] * q_in_ds.copy(); Qrr_ds = q_in_ds - Qfr_ds
                _Pavg_ds = np.nanmean(P_CONV_hydro_stable_hourly[:, :, ds]
                                      + P_CONV_hydro_RoR_hourly[:, :, ds])
                _lmean_ds = np.nanmean(l_norm_ds)
                _hi_ds = (_Pavg_ds / _lmean_ds
                          if np.isfinite(_lmean_ds) and _lmean_ds > 0 else _Pavg_ds)
                _hi_ds = max(_hi_ds, ELCC_A[ds] if np.isfinite(ELCC_A[ds]) else 0.0)
                _lo_d, _best_d, _best_rd = 0.0, 0.0, None
                for _ in range(200):
                    if _hi_ds - _lo_d < 0.1:
                        break
                    _mid_d = (_lo_d + _hi_ds) / 2
                    res_ds = _bal_core(_Y, hrs_byyear, c_or_ds, _mid_d,
                        Qfr_ds.copy(), Qrr_ds.copy(),
                        Q_CONV_stable_hourly[:, :, ds], _stable_env_release(ds, Qrr_ds),
                        precipitation_flux_hourly[:, :, ds], evaporation_flux_hourly[:, :, ds],
                        l_norm_ds, V_CONV_hourly[0, 0, ds], V_max_cumul[ds], Q_max_turb[ds],
                        P_r_turb[ds], eta_turb[ds], rho, g, secs_hr,
                        f_cascade_downstream[ds], bathy_c[ds], bathy_d[ds],
                        bathy_a[ds], bathy_b[ds], f_spill[ds], f_stop_cumul[ds],
                        f_restart_cumul[ds], mu[ds], HPP_category[ds] == 'B', _T)
                    if res_ds[-1] <= OUTAGE_MAX:
                        _best_d = _mid_d; _lo_d = _mid_d; _best_rd = res_ds
                    else:
                        _hi_ds = _mid_d
                if ds in matched_stations or not np.isfinite(ELCC_B[ds]):
                    ELCC_B[ds] = _best_d
                if _best_rd is not None:
                    Q_out_B[:, :, ds] = _best_rd[6]; V_B[:, :, ds] = _best_rd[0]
                    P_s_B[:, :, ds] = _best_rd[7]; P_f_B[:, :, ds] = _best_rd[8]
                    P_r_B[:, :, ds] = _best_rd[9]

    def _ps_operating_params_B(ps):
        v_up = V_max[ps]
        if (not np.isfinite(v_up) or v_up <= 0) and h_max[ps] > 0 and P_r_turb[ps] > 0:
            v_up = (P_r_turb[ps] * 1e6 * 8 * 3600
                    / (eta_turb[ps] * rho * g * h_max[ps]))
        if not np.isfinite(v_up) or v_up <= 0:
            v_up = 1e6

        abs_lat = (abs(float(df_stations.iloc[ps].get('lat_unified', 30)))
                   if ps < len(df_stations) else abs(station_lats[ps]))
        if abs_lat <= 15: evap_mm = 1200.0
        elif abs_lat <= 30: evap_mm = 1500.0
        elif abs_lat <= 45: evap_mm = 600.0
        elif abs_lat <= 55: evap_mm = 300.0
        else: evap_mm = 150.0
        ps_area = (float(df_stations.iloc[ps].get('ps_area_km2', 0)) * 1e6
                   if ps < len(df_stations) else 0.5e6)
        if not np.isfinite(ps_area) or ps_area <= 0:
            ps_area = 0.5e6
        evap_l = evap_mm * 1e-3 / (365.25 * 24 * 3600) * ps_area
        return v_up, evap_l

    def _eval_anchor_group_B(anchor, p_indices, trial_elcc, q_in_anchor, q_in_ps):
        """Solve one anchor once, then dispatch all attached PS against one residual load."""
        l_norm_anchor = L_norm_DC if anchor in matched_stations else L_norm_grid
        c_or_a = (C_OR_opt[anchor]
                  if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[anchor]) else 0.20)
        if HPP_category[anchor] == 'A':
            Qfr = q_in_anchor.copy(); Qrr = np.zeros_like(Qfr)
        else:
            Qfr = f_reg[anchor] * q_in_anchor.copy(); Qrr = q_in_anchor - Qfr

        anchor_result = _bal_core(_Y, hrs_byyear, c_or_a, trial_elcc,
            Qfr.copy(), Qrr.copy(), Q_CONV_stable_hourly[:, :, anchor],
            _stable_env_release(anchor, Qrr), precipitation_flux_hourly[:, :, anchor],
            evaporation_flux_hourly[:, :, anchor], l_norm_anchor,
            V_CONV_hourly[0, 0, anchor], V_max_cumul[anchor], Q_max_turb[anchor],
            P_r_turb[anchor], eta_turb[anchor], rho, g, secs_hr,
            f_cascade_downstream[anchor], bathy_c[anchor], bathy_d[anchor],
            bathy_a[anchor], bathy_b[anchor], f_spill[anchor],
            f_stop_cumul[anchor], f_restart_cumul[anchor], mu[anchor],
            HPP_category[anchor] == 'B', _T)

        combined_power = (np.nan_to_num(anchor_result[7])
                          + np.nan_to_num(anchor_result[8])
                          + np.nan_to_num(anchor_result[9]))
        ps_results = {}
        for p_idx in sorted(p_indices, key=lambda pi: _ndown_topo(ps_paired[pi]), reverse=True):
            ps = ps_paired[p_idx]
            v_up, evap_l = _ps_operating_params_B(ps)
            Pg, Pp, Vu, _ = _chain_ps_elcc(
                _Y, hrs_byyear, _T, secs_hr, rho, g,
                combined_power, l_norm_anchor, trial_elcc,
                P_r_turb[ps], Q_max_turb[ps], eta_turb[ps],
                P_r_pump[ps], Q_max_pump[ps], eta_pump[ps],
                v_up, h_max[ps], 0.5 * v_up, q_in_ps[ps], evap_l)
            combined_power += np.nan_to_num(Pg) - np.nan_to_num(Pp)
            if h_max[ps] > 0 and eta_turb[ps] > 0 and eta_pump[ps] > 0:
                Qg = np.nan_to_num(Pg) * 1e6 / (eta_turb[ps] * rho * g * h_max[ps])
                Qp = (np.nan_to_num(Pp) * eta_pump[ps] * 1e6
                      / (rho * g * h_max[ps]))
            else:
                Qg = np.zeros_like(Pg); Qp = np.zeros_like(Pp)
            ps_results[p_idx] = (Pg, Pp, Vu, Qg - Qp)

        n_outage = 0
        n_total = 0
        for y in range(_Y):
            nhrs = int(hrs_byyear[y])
            valid = np.isfinite(l_norm_anchor[:nhrs, y])
            target = trial_elcc * l_norm_anchor[:nhrs, y]
            n_outage += int(np.sum((combined_power[:nhrs, y] < target - 1e-6) & valid))
            n_total += int(np.sum(valid))
        return anchor_result, ps_results, n_outage / max(n_total, 1)

    if n_ps > 0:
        # Multiple PS may share one anchor. They must be solved as one coupled group so the anchor
        # dispatch and its residual load are not duplicated independently for every PS.
        _ps_groups_B = {}
        for p_idx, ps in enumerate(ps_paired):
            _ps_groups_B.setdefault(ps_anchor[ps], []).append(p_idx)
        _group_order_B = sorted(_ps_groups_B, key=_ndown_topo, reverse=True)
        _b_max_sweeps = int(os.environ.get('REVUB_DC_B_GS_MAX_ITER', '200'))
        _b_flow_tol = float(os.environ.get('REVUB_DC_B_GS_TOL_Q', '0.01'))
        _b_converged = False
        for _b_sweep in range(_b_max_sweeps):
            _group_inflow_used_B = {}
            for anchor in _group_order_B:
                p_indices = _ps_groups_B[anchor]
                ps_list = [ps_paired[p_idx] for p_idx in p_indices]
                for p_idx in p_indices:
                    PS_grid_share_B[p_idx] = 1.0 - frac_dc.get(anchor, 0.0)
                q_in_anchor = _cascade_qin(anchor, Q_out_B)
                q_in_ps = {ps: _cascade_qin(ps, Q_out_B) for ps in ps_list}
                _group_inflow_used_B[anchor] = (
                    q_in_anchor.copy(), {ps: q.copy() for ps, q in q_in_ps.items()})

                l_norm_anchor = L_norm_DC if anchor in matched_stations else L_norm_grid
                l_mean_a = np.nanmean(l_norm_anchor)
                group_capacity = P_r_turb[anchor] + sum(P_r_turb[ps] for ps in ps_list)
                elcc_hi_b = max(group_capacity / max(l_mean_a, 0.01),
                                ELCC_A[anchor] if np.isfinite(ELCC_A[anchor]) else 0.0)
                lo_b, hi_b, best_b = 0.0, elcc_hi_b, 0.0
                best_res_b = _eval_anchor_group_B(anchor, p_indices, 0.0, q_in_anchor, q_in_ps)
                for _ in range(200):
                    if hi_b - lo_b < 0.1:
                        break
                    mid_b = (lo_b + hi_b) / 2
                    res_b = _eval_anchor_group_B(anchor, p_indices, mid_b, q_in_anchor, q_in_ps)
                    if res_b[-1] <= OUTAGE_MAX:
                        best_b = mid_b; best_res_b = res_b; lo_b = mid_b
                    else:
                        hi_b = mid_b

                ELCC_B[anchor] = best_b
                if best_res_b is not None:
                    anchor_res_b, ps_results_b, _ = best_res_b
                    V_bf, _, _, _, _, _, Qo_bf, Ps_bf, Pf_bf, Pr_bf, _, _ = anchor_res_b
                    Q_out_B[:, :, anchor] = Qo_bf; V_B[:, :, anchor] = V_bf
                    P_s_B[:, :, anchor] = Ps_bf; P_f_B[:, :, anchor] = Pf_bf
                    P_r_B[:, :, anchor] = Pr_bf
                    for p_idx, (Pg_b, Pp_b, Vu_b, net_q_b) in ps_results_b.items():
                        ps = ps_paired[p_idx]
                        P_PS_gen_B[:, :, p_idx] = Pg_b
                        P_PS_pump_B[:, :, p_idx] = Pp_b
                        V_PS_upper_B[:, :, p_idx] = Vu_b
                        _ps_net_flow_B[ps] = net_q_b
                        Q_out_B[:, :, ps] = np.maximum(q_in_ps[ps] + net_q_b, 0.0)

                _propagate_group_B(anchor, ps_list)

            # Re-evaluate every group after the complete sweep. This catches later groups changing an
            # earlier group's inflow (possible when PS and anchor topological directions differ).
            _b_max_change = 0.0
            for anchor in _group_order_B:
                _used_qa, _used_qps = _group_inflow_used_B[anchor]
                _b_max_change = max(
                    _b_max_change,
                    float(np.nanmax(np.abs(_cascade_qin(anchor, Q_out_B) - _used_qa))))
                for ps, _used_qp in _used_qps.items():
                    _b_max_change = max(
                        _b_max_change,
                        float(np.nanmax(np.abs(_cascade_qin(ps, Q_out_B) - _used_qp))))
            if _b_max_change <= _b_flow_tol:
                _b_converged = True
                break
        if not _b_converged:
            print(f'    WARNING: Scenario B PS/cascade coupling did not converge after '
                  f'{_b_max_sweeps} sweeps (max inflow change={_b_max_change:.3g} m3/s); '
                  f'falling back to Scenario A')
            ELCC_B = ELCC_A.copy()
            Q_out_B = Q_out_A.copy(); V_B = V_A.copy()
            P_s_B = P_s_A.copy(); P_f_B = P_f_A.copy(); P_r_B = P_r_A.copy()
            P_PS_gen_B.fill(0.0); P_PS_pump_B.fill(0.0); V_PS_upper_B.fill(0.0)
            _ps_net_flow_B.clear()

    # PS is optional independently for each disconnected river component. Select the complete A or B
    # dispatch per component, preserving all within-cascade flow dependencies while retaining useful PS
    # operation elsewhere in the region.
    _b_parent = list(range(HPP_number))
    def _b_find(x):
        while _b_parent[x] != x:
            _b_parent[x] = _b_parent[_b_parent[x]]; x = _b_parent[x]
        return x
    def _b_union(a, b):
        ra, rb = _b_find(a), _b_find(b)
        if ra != rb: _b_parent[ra] = rb
    for _h in range(HPP_number):
        _ds = downstream_index[_h]
        if 0 <= _ds < HPP_number: _b_union(_h, _ds)
    # An anchor and its PS are one operational unit even when the river graph
    # data does not place them in the same connected component.
    for _ps in ps_paired:
        _b_union(_ps, ps_anchor[_ps])
    _b_components = {}
    for _h in range(HPP_number):
        _b_components.setdefault(_b_find(_h), []).append(_h)
    for _members in _b_components.values():
        _a_supply = sum(ELCC_A[h] * frac_dc.get(h, 0.0) for h in _members
                        if h in matched_stations and np.isfinite(ELCC_A[h]))
        _b_supply = sum(ELCC_B[h] * frac_dc.get(h, 0.0) for h in _members
                        if h in matched_stations and np.isfinite(ELCC_B[h]))
        if _b_supply < _a_supply:
            for h in _members:
                ELCC_B[h] = ELCC_A[h]
                Q_out_B[:, :, h] = Q_out_A[:, :, h]; V_B[:, :, h] = V_A[:, :, h]
                P_s_B[:, :, h] = P_s_A[:, :, h]; P_f_B[:, :, h] = P_f_A[:, :, h]
                P_r_B[:, :, h] = P_r_A[:, :, h]
            _member_set = set(_members)
            for _pi, _ps in enumerate(ps_paired):
                if _ps in _member_set:
                    P_PS_gen_B[:, :, _pi] = 0.0; P_PS_pump_B[:, :, _pi] = 0.0
                    V_PS_upper_B[:, :, _pi] = 0.0
    supply_B = sum(ELCC_B[i] * frac_dc.get(i, 0.0)
                   for i in range(HPP_number) if i in matched_stations and np.isfinite(ELCC_B[i]))
    grid_B = _grid_portfolio_elcc(P_s_B, P_f_B, P_r_B, P_PS_gen_B, P_PS_pump_B, PS_grid_share_B)
    print(f'    Scenario B: supply→DC={supply_B:.0f} MW (+{supply_B-supply_A:.0f}), '
          f'residual→grid={grid_B:.0f} MW, {time.time()-t_b:.0f}s')

    # ═══════════════════════════════════════════════════════════
    # D.3) SCENARIO C: Chain PS + closed-loop, mixed L_norm
    # ═══════════════════════════════════════════════════════════

    print(f'  Scenario C: Chain PS + closed-loop cascade')
    t_c = time.time()

    # PER-STATION scenario C (consistent with A/B): produce ELCC_C[i] for every matched station,
    # then supply_C = Σ ELCC_C[i]*frac_dc[i]. Start from B (per-pair PS) so non-chained stations
    # inherit B and chained stations are then refined by PS + cascade coordination.
    # Base scenario C hourly arrays on B (independent + per-pair PS): non-chained matched stations
    # and grid stations keep their B dispatch; chained members are overwritten below with the actual
    # coordinated combined-firm dispatch so dc_scenario_C_hourly.npz reflects scenario C (not A).
    ELCC_C = ELCC_B.copy()
    Q_out_C = Q_out_B.copy(); V_C = V_B.copy()
    P_s_C = P_s_B.copy(); P_f_C = P_f_B.copy(); P_r_C = P_r_B.copy()
    P_PS_gen_C = P_PS_gen_B.copy(); P_PS_pump_C = P_PS_pump_B.copy()
    PS_grid_share_C = PS_grid_share_B.copy()
    station_grid_share_C = np.array([1.0 - frac_dc.get(i, 0.0) for i in range(HPP_number)])
    for _pi, _ps in enumerate(ps_paired):
        P_f_C[:, :, _ps] = P_PS_gen_C[:, :, _pi] - P_PS_pump_C[:, :, _pi]
    chained_stations = set()

    def _rd_station_C(h, l_norm_hpp):
        """Binary-search _bal_core for station h against l_norm_hpp using its CURRENT cascade
        inflow (reflecting PS + upstream re-dispatch in Q_out_C). Mirrors _run_bal_all exactly.
        Returns (elcc, res) where res is the best _bal_core tuple (or None)."""
        q_in = _cascade_qin(h, Q_out_C)
        l_mean = np.nanmean(l_norm_hpp)
        if HPP_category[h] == 'RoR':
            qcap = np.fmin(np.fmin(q_in, Q_max_turb[h]) * eta_turb[h] * rho * g * h_max[h] / 1e6, P_r_turb[h])
            ratio = qcap / l_norm_hpp
            rv = ratio[np.isfinite(ratio) & (l_norm_hpp > 0)]
            elcc = max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0
            return elcc, ('RoR', q_in, qcap)
        c_or = C_OR_opt[h] if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[h]) else min(0.30, max(0.05, 1.0 - d_min[h]))
        P_avg = np.nanmean(P_CONV_hydro_stable_hourly[:, :, h] + P_CONV_hydro_RoR_hourly[:, :, h])
        elcc_hi = P_avg / l_mean if np.isfinite(l_mean) and l_mean > 0 else P_avg
        if not np.isfinite(elcc_hi) or elcc_hi <= 0:
            return 0.0, None
        if HPP_category[h] == 'A':
            Qfr = q_in.copy(); Qrr = np.zeros_like(Qfr)
        else:
            Qfr = f_reg[h] * q_in.copy(); Qrr = q_in - Qfr
        lo, hi, best, best_res = 0.0, elcc_hi, 0.0, None
        for _ in range(200):
            if hi - lo < 0.1: break
            mid = (lo + hi) / 2
            res = _bal_core(_Y, hrs_byyear, c_or, mid, Qfr.copy(), Qrr.copy(),
                Q_CONV_stable_hourly[:, :, h], _stable_env_release(h, Qrr),
                precipitation_flux_hourly[:, :, h], evaporation_flux_hourly[:, :, h],
                l_norm_hpp, V_CONV_hourly[0, 0, h], V_max_cumul[h], Q_max_turb[h],
                P_r_turb[h], eta_turb[h], rho, g, secs_hr,
                f_cascade_downstream[h], bathy_c[h], bathy_d[h], bathy_a[h], bathy_b[h],
                f_spill[h], f_stop_cumul[h], f_restart_cumul[h], mu[h],
                HPP_category[h] == 'B', _T)
            if res[-1] <= OUTAGE_MAX:
                best = mid; lo = mid; best_res = res
            else:
                hi = mid
        return best, best_res

    # ---- dispatch one station to an arbitrary hourly TARGET array (elcc=1 ⇒ Lh = tgt) ----
    def _dispatch_to_target(h, q_in, tgt):
        c_or = C_OR_opt[h] if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[h]) else min(0.30, max(0.05, 1.0 - d_min[h]))
        if HPP_category[h] == 'A':
            Qfr = q_in.copy(); Qrr = np.zeros_like(Qfr)
        else:
            Qfr = f_reg[h] * q_in.copy(); Qrr = q_in - Qfr
        return _bal_core(_Y, hrs_byyear, c_or, 1.0, Qfr.copy(), Qrr.copy(),
            Q_CONV_stable_hourly[:, :, h], _stable_env_release(h, Qrr),
            precipitation_flux_hourly[:, :, h], evaporation_flux_hourly[:, :, h],
            tgt, V_CONV_hourly[0, 0, h], V_max_cumul[h], Q_max_turb[h],
            P_r_turb[h], eta_turb[h], rho, g, secs_hr,
            f_cascade_downstream[h], bathy_c[h], bathy_d[h], bathy_a[h], bathy_b[h],
            f_spill[h], f_stop_cumul[h], f_restart_cumul[h], mu[h],
            HPP_category[h] == 'B', _T)

    # ---- build cascade chains = connected components of the river graph (incl. pure river
    #      cascades, not only PS-anchored ones) ----
    _parent = list(range(HPP_number))
    def _find(x):
        while _parent[x] != x:
            _parent[x] = _parent[_parent[x]]; x = _parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb: _parent[ra] = rb
    for _h in range(HPP_number):
        _ds0 = downstream_index[_h]
        if 0 <= _ds0 < HPP_number: _union(_h, _ds0)
    _components = {}
    for _h in range(HPP_number):
        _components.setdefault(_find(_h), []).append(_h)
    if os.environ.get('REVUB_D_SKIP_SCENARIO_C', '0') == '1':
        # E-v2 fast path: skip chain combined-firm coordination entirely. With no chains the
        # loop below never runs, chain_supply_C stays 0 and supply_C falls back to
        # Σ ELCC_B·frac == supply_B. Reported numbers (supply_C, grid_C) are exactly B.
        # (The C hourly arrays are B copies except P_f_C at PS columns, which carries the
        #  PS net-gen overlay from above - so the C hourly npz is not written under this flag.)
        _components = {}
        print('    Scenario C: SKIPPED (REVUB_D_SKIP_SCENARIO_C=1) -> C = B', flush=True)

    _ps_paired_set = set(ps_paired) if 'ps_paired' in dir() else set()
    _LDC = np.nan_to_num(L_norm_DC)
    _Lgrid = np.nan_to_num(L_norm_grid)
    _DCmask = np.isfinite(L_norm_DC)
    _DCn_tot = int(_DCmask.sum())

    def _chain_combined_firm(redispatch_topo, matched_in, ps_in):
        """Combined firm of the matched members' SUMMED output vs L_norm_DC under coordinated
        dispatch. Matched storage members re-time their flexible releases toward the COMBINED
        deficit (Gauss-Seidel down the cascade, routed by the model's own _cascade_qin so the
        tau/year-boundary handling matches A/B exactly); non-matched intermediates BELOW a matched
        member route the re-timed water at their own (typical) grid dispatch; PS fills the residual.
        Pure-upstream / tributary stations are frozen at scenario A (re-dispatching them vs typical
        reproduces A exactly, so omitting them is exact - not an approximation - and much faster).
        Binary-search the max firm C with outage ≤ OUTAGE_MAX. Returns combined firm (MW)."""
        matched_set = set(matched_in)
        ps_par = {}
        for ps in ps_in:
            if not (np.isfinite(V_max[ps]) and V_max[ps] > 0 and h_max[ps] > 0):
                continue
            abs_lat = abs(float(df_stations.iloc[ps].get('lat_unified', 30))) if ps < len(df_stations) else abs(station_lats[ps])
            if abs_lat <= 15: evap_mm = 1200.0
            elif abs_lat <= 30: evap_mm = 1500.0
            elif abs_lat <= 45: evap_mm = 600.0
            elif abs_lat <= 55: evap_mm = 300.0
            else: evap_mm = 150.0
            ps_area = float(df_stations.iloc[ps].get('ps_area_km2', 0)) * 1e6 if ps < len(df_stations) else 0.5e6
            if not np.isfinite(ps_area) or ps_area <= 0: ps_area = 0.5e6
            evap_l = evap_mm * 1e-3 / (365.25 * 24 * 3600) * ps_area
            ps_par[ps] = (V_max[ps], evap_l)

        def _eval(C, capture=False):
            # frozen (non-redispatched) members keep their scenario-A outflow; _cascade_qin reads it.
            Q_out_local = Q_out_A.copy()
            Ps_ = {h: np.nan_to_num(P_s_A[:, :, h]).copy() for h in matched_in}
            Pf_ = {h: np.nan_to_num(P_f_A[:, :, h]).copy() for h in matched_in}
            Pr_ = {h: np.nan_to_num(P_r_A[:, :, h]).copy() for h in matched_in}
            V_ = {h: np.nan_to_num(V_A[:, :, h]).copy() for h in matched_in}
            total = np.zeros((_T, _Y))
            for h in matched_in: total += Ps_[h] + Pf_[h] + Pr_[h]
            ps_gen = {ps: np.zeros((_T, _Y)) for ps in ps_in if ps in ps_par}
            ps_pump = {ps: np.zeros((_T, _Y)) for ps in ps_in if ps in ps_par}
            ps_net_sum = np.zeros((_T, _Y))

            D = C * _LDC
            max_gs = int(os.environ.get('REVUB_DC_GS_MAX_ITER', '200'))
            conv_tol_mw = float(os.environ.get('REVUB_DC_GS_TOL_MW', '0.1'))
            conv_tol_q = float(os.environ.get('REVUB_DC_GS_TOL_Q', '0.01'))
            conv_tol_v_frac = float(os.environ.get('REVUB_DC_GS_TOL_V_FRAC', '1e-6'))
            converged = False
            ps_vu = {}
            Vstate = {h: np.nan_to_num(V_A[:, :, h]).copy() for h in redispatch_topo
                      if type_unified[h] != 'PS' and HPP_category[h] != 'RoR'}
            for _gs in range(max_gs):
                max_power_change = 0.0
                max_flow_change = 0.0
                max_storage_frac_change = 0.0
                for h in redispatch_topo:
                    if type_unified[h] == 'PS':
                        # PS dispatched AT its cascade position: it fills the residual combined deficit
                        # and writes its NET river flow (Q_gen - Q_pump) into Q_out_local, so downstream
                        # stations and downstream PS see the water impact (water conservation - the same
                        # river water can't be re-extracted by multiple PS).
                        if h in ps_par:
                            v_up, evap_l = ps_par[h]
                            q_river = _cascade_qin(h, Q_out_local)
                            old_net = ps_gen[h] - ps_pump[h]
                            base = total + (ps_net_sum - old_net)
                            Pg, Pp, Vu, _o = _chain_ps_elcc(_Y, hrs_byyear, _T, secs_hr, rho, g,
                                base, L_norm_DC, C,
                                P_r_turb[h], Q_max_turb[h], eta_turb[h],
                                P_r_pump[h], Q_max_pump[h], eta_pump[h],
                                v_up, h_max[h], v_up * 0.5, q_river, evap_l)
                            Pg = np.nan_to_num(Pg); Pp = np.nan_to_num(Pp)
                            Vu = np.nan_to_num(Vu)
                            if h in ps_vu:
                                max_storage_frac_change = max(max_storage_frac_change,
                                    float(np.nanmax(np.abs(Vu - ps_vu[h]))) / max(v_up, 1.0))
                            else:
                                max_storage_frac_change = max(max_storage_frac_change, 1.0)
                            ps_vu[h] = Vu
                            new_net = Pg - Pp
                            max_power_change = max(max_power_change,
                                float(np.nanmax(np.abs(Pg - ps_gen[h]))),
                                float(np.nanmax(np.abs(Pp - ps_pump[h]))))
                            ps_net_sum = ps_net_sum + (new_net - old_net)
                            ps_gen[h] = Pg; ps_pump[h] = Pp
                            Q_gen_h = Pg * 1e6 / (eta_turb[h] * rho * g * h_max[h])
                            Q_pump_h = Pp * eta_pump[h] * 1e6 / (rho * g * h_max[h])
                            _new_qout = np.maximum(q_river + Q_gen_h - Q_pump_h, 0.0)
                            max_flow_change = max(max_flow_change,
                                float(np.nanmax(np.abs(_new_qout - Q_out_local[:, :, h]))))
                            Q_out_local[:, :, h] = _new_qout
                        else:
                            _new_qout = _cascade_qin(h, Q_out_local)
                            max_flow_change = max(max_flow_change,
                                float(np.nanmax(np.abs(_new_qout - Q_out_local[:, :, h]))))
                            Q_out_local[:, :, h] = _new_qout
                        continue
                    q_in = _cascade_qin(h, Q_out_local)
                    if HPP_category[h] == 'RoR':
                        max_flow_change = max(max_flow_change,
                            float(np.nanmax(np.abs(q_in - Q_out_local[:, :, h]))))
                        Q_out_local[:, :, h] = q_in
                        if h in matched_set:
                            nPr = np.nan_to_num(np.fmin(np.fmin(q_in, Q_max_turb[h]) * eta_turb[h] * rho * g * h_max[h] / 1e6, P_r_turb[h]))
                            old = Ps_[h] + Pf_[h] + Pr_[h]
                            Pr_[h] = nPr; Ps_[h] = np.zeros((_T, _Y)); Pf_[h] = np.zeros((_T, _Y))
                            total = total + (nPr - old)
                            max_power_change = max(max_power_change,
                                float(np.nanmax(np.abs(nPr - old))))
                        continue
                    if h in matched_set:
                        old = Ps_[h] + Pf_[h] + Pr_[h]
                        tgt = np.maximum(0.0, D - (total - old) - ps_net_sum)
                    else:
                        tgt = (ELCC_A[h] if np.isfinite(ELCC_A[h]) else 0.0) * _Lgrid
                    res = _dispatch_to_target(h, q_in, tgt)
                    if h in Vstate:
                        _Vnew = np.nan_to_num(res[0])
                        max_storage_frac_change = max(max_storage_frac_change,
                            float(np.nanmax(np.abs(_Vnew - Vstate[h]))) / max(V_max_cumul[h], 1.0))
                        Vstate[h] = _Vnew
                    if h in matched_set:
                        nPs, nPf, nPr = np.nan_to_num(res[7]), np.nan_to_num(res[8]), np.nan_to_num(res[9])
                        new = nPs + nPf + nPr
                        Ps_[h], Pf_[h], Pr_[h] = nPs, nPf, nPr
                        V_[h] = np.nan_to_num(res[0])
                        total = total + (new - old)
                        max_power_change = max(max_power_change,
                            float(np.nanmax(np.abs(new - old))))
                    max_flow_change = max(max_flow_change,
                        float(np.nanmax(np.abs(res[6] - Q_out_local[:, :, h]))))
                    Q_out_local[:, :, h] = res[6]
                if (max_power_change <= conv_tol_mw and max_flow_change <= conv_tol_q
                        and max_storage_frac_change <= conv_tol_v_frac):
                    converged = True
                    break

            P_comb = total + ps_net_sum
            n_out = int(np.sum((P_comb < D - 1e-6) & _DCmask))
            outage = n_out / max(_DCn_tot, 1)
            if capture:
                return outage, {'Ps': Ps_, 'Pf': Pf_, 'Pr': Pr_, 'V': V_, 'Qo': Q_out_local,
                                'ps_gen': ps_gen, 'ps_pump': ps_pump,
                                'redis': list(redispatch_topo)}, converged
            return outage, converged

        hi = sum(P_r_turb[h] for h in matched_in) + sum(P_r_turb[h] for h in ps_in if h in ps_par)
        if not np.isfinite(hi) or hi <= 0:
            return 0.0, None
        lo, best = 0.0, 0.0
        for _ in range(40):
            if hi - lo < 0.5: break
            mid = (lo + hi) / 2
            _out, _conv = _eval(mid)
            if _conv and _out <= OUTAGE_MAX:
                best = mid; lo = mid
            else:
                hi = mid
        # re-run the winning level once to capture its coordinated hourly dispatch for write-back.
        _o, disp, _conv = _eval(best, capture=True)
        return (best, disp) if _conv else (0.0, None)

    # iterate cascade components; coordinate those with ≥2 matched members or 1 matched + PS
    chain_supply_C = 0.0
    chained_matched = set()
    for _root, _members in _components.items():
        _matched_in = [h for h in _members if h in matched_stations]
        _ps_in = [h for h in _members if type_unified[h] == 'PS' and h in _ps_paired_set]
        if len(_matched_in) < 2 and not (len(_matched_in) == 1 and _ps_in):
            continue
        chained_stations.update(_members)
        def _ndown(h):
            n = 0; cur = downstream_index[h]; seen = 0
            while 0 <= cur < HPP_number and seen <= HPP_number:
                n += 1; cur = downstream_index[cur]; seen += 1
            return n
        # Re-dispatch set = matched ∪ PS ∪ stations strictly downstream of a matched/PS member
        # (these route the re-timed water). Pure-upstream / tributary non-matched stations are
        # frozen at scenario A - re-dispatching them vs typical reproduces A exactly, so this is
        # exact, not an approximation, and avoids dispatching the whole connected component.
        _comp_set = set(_members)
        _redis = set(_matched_in) | set(_ps_in)
        for _m in list(_matched_in) + list(_ps_in):
            _cur = downstream_index[_m]; _seen = 0
            while 0 <= _cur < HPP_number and _cur in _comp_set and _seen <= HPP_number:
                _redis.add(_cur); _cur = downstream_index[_cur]; _seen += 1
        _redis_topo = sorted(_redis, key=_ndown, reverse=True)
        _firm, _disp = _chain_combined_firm(_redis_topo, _matched_in, _ps_in)
        _alloc = sum(dc_alloc.get(h, 0.0) for h in _matched_in)
        _flat = sum(flat_elcc[h] for h in _matched_in if flat_elcc[h] > 0)
        _frac = min(_alloc / _flat, 1.0) if _flat > 0 else 0.0
        _b_supply = sum(ELCC_B[h] * frac_dc.get(h, 0.0) for h in _matched_in if np.isfinite(ELCC_B[h]))
        _firm_supply = _firm * _frac
        _use_chain = _disp is not None and _firm_supply >= _b_supply

        # Arrays start from B. Overwrite them only when the coordinated chain result is the option
        # selected for the reported C supply; otherwise retain the complete, reproducible B dispatch.
        if _use_chain:
            for h in _matched_in:
                station_grid_share_C[h] = 1.0 - _frac
            for _ps in _ps_in:
                P_f_C[:, :, _ps] = 0.0
                if _ps in set(ps_paired):
                    _pi = list(ps_paired).index(_ps)
                    P_PS_gen_C[:, :, _pi] = 0.0
                    P_PS_pump_C[:, :, _pi] = 0.0
                    PS_grid_share_C[_pi] = 1.0 - _frac
            for h in _matched_in:
                P_s_C[:, :, h] = _disp['Ps'][h]; P_f_C[:, :, h] = _disp['Pf'][h]
                P_r_C[:, :, h] = _disp['Pr'][h]; V_C[:, :, h] = _disp['V'][h]
            for h in _redis_topo:
                Q_out_C[:, :, h] = _disp['Qo'][:, :, h]
            for _ps, _pg in _disp['ps_gen'].items():
                _pp = _disp['ps_pump'].get(_ps, np.zeros_like(_pg))
                P_f_C[:, :, _ps] = _pg - _pp
                if _ps in set(ps_paired):
                    _pi = list(ps_paired).index(_ps)
                    P_PS_gen_C[:, :, _pi] = _pg
                    P_PS_pump_C[:, :, _pi] = _pp
        _c_supply = _firm_supply if _use_chain else _b_supply
        chain_supply_C += _c_supply
        for h in _matched_in: chained_matched.add(h)
        print(f'    chain {[HPP_name[h][:12] for h in _matched_in]}: combFirm={_firm:.0f} '
              f'x frac{_frac:.2f}={_firm*_frac:.0f} vs Bsupply={_b_supply:.0f} -> {_c_supply:.0f}', flush=True)

    # supply_C = Σ chains (combined-firm coordinated, B-floored at supply level)
    #          + Σ non-chained matched stations (per-station B, same as supply_B).
    # Each term ≥ its supply_B counterpart ⇒ supply_C ≥ supply_B guaranteed.
    supply_C = chain_supply_C + sum(
        ELCC_B[i] * frac_dc.get(i, 0.0)
        for i in matched_stations if i not in chained_matched and np.isfinite(ELCC_B[i]))
    # grid residual: value every matched station's independent+PS (B) residual vs the grid template.
    # (Chained stations' coordinated DC dispatch is firm-pooled, not tracked per-station here, so the
    #  residual-to-grid is reported on the B dispatch - a conservative, consistent reference.)
    if os.environ.get('REVUB_D_SKIP_SCENARIO_C', '0') == '1':
        grid_C = grid_B  # C arrays are exact B copies when coordination is skipped
    else:
        grid_C = _grid_portfolio_elcc(P_s_C, P_f_C, P_r_C, P_PS_gen_C, P_PS_pump_C,
                                      PS_grid_share_C, station_grid_share_C)
    print(f'    Scenario C: supply→DC={supply_C:.0f} MW (+{supply_C-supply_A:.0f}), '
          f'residual→grid={grid_C:.0f} MW, {time.time()-t_c:.0f}s')

    # ═══════════════════════════════════════════════════════════
    # D.4) Final matching + save + plot
    # ═══════════════════════════════════════════════════════════

    total_d = dc_in_region['p_total_peak_kw'].sum() / 1e3 if len(dc_in_region) > 0 else 0

    pd.DataFrame([{
        'region': region_name, 'gcm': dc_gcm, 'ssp': dc_ssp,
        'n_dc': n_dc, 'dc_peak_mw': dc_peak, 'dc_mean_mw': dc_mean,
        'e_cross_region_profile': int(bool(E_PROFILE_OVERRIDE)),
        'supply_A_mw': supply_A, 'supply_B_mw': supply_B, 'supply_C_mw': supply_C,
        'grid_resid_A_mw': grid_A, 'grid_resid_B_mw': grid_B, 'grid_resid_C_mw': grid_C,
    }]).to_csv(os.path.join(OUTPUT_DIR_D, 'dc_hydro_summary.csv'), index=False)

    alloc_flat.to_csv(os.path.join(OUTPUT_DIR_D, 'dc_hydro_allocation.csv'), index=False)

    # Region station profile for cross-region trading (E file): region-neutral ELCC + correct coords.
    _elcc_ref = ELCC_indep if 'ELCC_indep' in dir() else ELCC_opt
    pd.DataFrame({
        'region': region_name,
        'hpp_idx': list(range(HPP_number)),
        'name': [HPP_name[i] for i in range(HPP_number)],
        'lat': [station_lats[i] for i in range(HPP_number)],
        'lon': [station_lons[i] for i in range(HPP_number)],
        'type': [type_unified[i] for i in range(HPP_number)],
        'elcc_indep_mw': [_elcc_ref[i] for i in range(HPP_number)],
        'elcc_A_mw': [ELCC_A[i] for i in range(HPP_number)],
        'elcc_B_mw': [ELCC_B[i] for i in range(HPP_number)],
    }).to_csv(os.path.join(OUTPUT_DIR_D, 'region_station_profile.csv'), index=False)

    if not D_SUMMARY_ONLY:
        np.savez_compressed(os.path.join(OUTPUT_DIR_D, 'dc_scenario_A_hourly.npz'),
            ELCC_A=ELCC_A, Q_out_A=Q_out_A, V_A=V_A,
            P_s_A=P_s_A, P_f_A=P_f_A, P_r_A=P_r_A, L_norm_DC=L_norm_DC)
        np.savez_compressed(os.path.join(OUTPUT_DIR_D, 'dc_scenario_B_hourly.npz'),
            ELCC_B=ELCC_B, Q_out_B=Q_out_B, V_B=V_B,
            P_s_B=P_s_B, P_f_B=P_f_B, P_r_B=P_r_B,
            P_PS_gen_B=P_PS_gen_B, P_PS_pump_B=P_PS_pump_B, V_PS_upper_B=V_PS_upper_B)
        if os.environ.get('REVUB_D_SKIP_SCENARIO_C', '0') == '1':
            # C was skipped (C == B at the summary level); its hourly arrays are B
            # copies with a PS overlay in P_f_C - writing them would be misleading.
            print(f'\n  Saved: summary, allocation, A/B hourly npz (C skipped)')
        else:
            np.savez_compressed(os.path.join(OUTPUT_DIR_D, 'dc_scenario_C_hourly.npz'),
                Q_out_C=Q_out_C, V_C=V_C, P_s_C=P_s_C, P_f_C=P_f_C, P_r_C=P_r_C,
                P_PS_gen_C=P_PS_gen_C, P_PS_pump_C=P_PS_pump_C)
            print(f'\n  Saved: summary, allocation, A/B/C hourly npz')
    else:
        print(f'\n  Saved: summary and allocation (E summary-only mode)')

    if D_SUMMARY_ONLY:
        print('  Plot skipped (E summary-only mode)')
    else:
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Hydro→DC: {region_name} ({dc_gcm}/{dc_ssp})\nMixed L_norm: DC-matched={len(matched_stations)} stations', fontsize=12)
            sup_vals = [supply_A, supply_B, supply_C]
            bars = axes[0,0].bar(['DC peak','A','B','C'], [dc_peak]+sup_vals, color=['orange','#90CAF9','#42A5F5','#1565C0'])
            for b,v in zip(bars,[dc_peak]+sup_vals): axes[0,0].text(b.get_x()+b.get_width()/2,v+5,f'{v:.0f}',ha='center',fontsize=9)
            axes[0,0].set_ylabel('MW'); axes[0,0].set_title('DC-share firm-capacity POTENTIAL (ELCC x frac) vs Demand')
            grid_vals = [grid_A, grid_B, grid_C]
            bars2 = axes[0,1].bar(['A','B','C'], grid_vals, color=['#A5D6A7','#66BB6A','#388E3C'])
            for b,v in zip(bars2,grid_vals): axes[0,1].text(b.get_x()+b.get_width()/2,v+5,f'{v:.0f}',ha='center',fontsize=9)
            axes[0,1].set_ylabel('MW'); axes[0,1].set_title('Residual → Grid (vs grid load curve)')
            p_yr = P_DC_MW_all[_Y//2]; ndays=len(p_yr)//24; d=np.arange(ndays)
            da=lambda x: np.array([np.mean(x[i*24:(i+1)*24]) for i in range(ndays)])
            axes[1,0].fill_between(d,0,da(p_yr),alpha=0.3,color='orange',label='DC load')
            axes[1,0].axhline(supply_A,color='#90CAF9',lw=1,label=f'A={supply_A:.0f}')
            axes[1,0].axhline(supply_C,color='#1565C0',lw=1.5,label=f'C={supply_C:.0f}')
            axes[1,0].set_xlabel('Day');axes[1,0].set_ylabel('MW');axes[1,0].legend(fontsize=7)
            axes[1,0].set_title('DC Load vs Supply')
            axes[1,1].text(0.5,0.5,f'DC: {n_dc} centers, {dc_peak:.0f} MW peak / {total_d:.0f} MW total\n{len(matched_stations)} stations matched\n\nSupply potential: {supply_A:.0f} / {supply_B:.0f} / {supply_C:.0f} MW\n(DC-curve firm cap, may exceed demand)\nResidual→grid: {grid_A:.0f} / {grid_B:.0f} / {grid_C:.0f} MW',
                transform=axes[1,1].transAxes,ha='center',va='center',fontsize=11)
            axes[1,1].set_title('Summary')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_D,'dc_hydro_3scenarios.png'),dpi=130,bbox_inches='tight');plt.close(fig)
            print(f'  Plot saved')
        except Exception as e:
            print(f'  Plot failed: {e}')

    print(f'\n{"="*60}')
    print(f'  DC demand: {total_d:.0f} MW total, {dc_peak:.0f} MW peak ({n_dc} centers)')
    print(f'  NOTE: supply = DC-curve firm-capacity POTENTIAL of matched stations (can exceed DC demand)')
    print(f'  {"":14s}{"supply(pot)":>14s}{"residual→grid":>16s}')
    print(f'  A (indep):    {supply_A:10.0f}MW{grid_A:14.0f}MW')
    print(f'  B (+PS pair): {supply_B:10.0f}MW (+{supply_B-supply_A:.0f}){grid_B:11.0f}MW')
    print(f'  C (chain+PS): {supply_C:10.0f}MW (+{supply_C-supply_A:.0f}){grid_C:11.0f}MW')
    print(f'{"="*60}')
