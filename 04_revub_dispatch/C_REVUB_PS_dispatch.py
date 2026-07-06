# -*- coding: utf-8 -*-
"""
REVUB Pumped Storage (PS) dispatch module.

Run AFTER B_REVUB_main_code_v2.py completes (via exec()).  All variables from
A_REVUB_initialise_v2.py and B_REVUB_main_code_v2.py are assumed to be in
scope, including HPP_number, type_unified, HPP_category, ELCC_opt, df_stations,
L_norm, cascade topology arrays, and all BAL results.

Workflow
--------
1. Identify PS stations (type_unified == 'PS').
2. Match each PS to an "anchor" STO reservoir using one of four modes:
       nearest      - haversine distance to nearest non-PS station
       nearest_sto  - haversine distance to nearest STO (category A/B,
                      type_unified != 'PS')
       specified    - user-provided pairs via REVUB_PS_PAIRS env var
       cascade      - nearest STO in the same cascade chain
3. Run a combined BAL + PS dispatch kernel (_ps_bal_core, Numba) that
   co-optimises turbine release and pump-back volumes.
4. Search for an aggregate ELCC that satisfies outage constraints when PS
   capacity is added to the paired STO.
5. Write results to parquet.

Environment variables
---------------------
    REVUB_PS_MODE   : off | nearest | nearest_sto | specified | cascade
                      (default: off)
    REVUB_PS_PAIRS  : semicolon-delimited "PS_name:STO_name" pairs
                      (only used when mode == specified)
"""

import numpy as np
import os
import time
import warnings

# ─── guard: A + B must have run ────────────────────────────────────────────
assert 'HPP_number' in dir(), 'Run A_REVUB_initialise_v2.py and B_REVUB_main_code_v2.py first'
assert 'ELCC_opt' in dir(), 'Run B_REVUB_main_code_v2.py (with BAL) before PS dispatch'

# =============================================================================
# PS.0) Configuration
# =============================================================================

ps_mode = os.environ.get('REVUB_PS_MODE', 'off').strip().lower()
ps_pairs_env = os.environ.get('REVUB_PS_PAIRS', '').strip()

print(f'\n{"="*60}')
print(f'REVUB PS Dispatch - mode={ps_mode}')
print(f'{"="*60}\n')

# =============================================================================
# PS.1) Identify PS stations
# =============================================================================

ps_indices = [i for i in range(HPP_number) if type_unified[i] == 'PS']
n_ps = len(ps_indices)

if ps_mode == 'off' or n_ps == 0:
    if n_ps == 0:
        print('[PS] No pumped-storage stations found - skipping PS dispatch.')
    else:
        print('[PS] REVUB_PS_MODE=off - skipping PS dispatch.')
    # Ensure downstream scripts see an empty result flag
    ps_dispatch_ran = False
else:
    ps_dispatch_ran = True
    print(f'[PS] Found {n_ps} PS station(s):')
    for idx in ps_indices:
        print(f'     {HPP_name[idx]:50s}  P_turb={P_r_turb[idx]:.1f} MW  '
              f'P_pump={P_r_pump[idx]:.1f} MW  V_upper={V_max[idx]/1e6:.1f} Mm3')

    # =================================================================
    # PS.2) Haversine helper
    # =================================================================

    def _haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance in km between two points (decimal degrees)."""
        R = 6371.0  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2.0) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
             * np.sin(dlon / 2.0) ** 2)
        return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # Determine which lat/lon columns are available in df_stations
    if 'lat_unified' in df_stations.columns:
        _lat_col, _lon_col = 'lat_unified', 'lon_unified'
    elif 'lat' in df_stations.columns:
        _lat_col, _lon_col = 'lat', 'lon'
    elif 'latitude' in df_stations.columns:
        _lat_col, _lon_col = 'latitude', 'longitude'
    else:
        raise ValueError('[PS] Cannot find lat/lon columns in df_stations. '
                         'Expected lat_unified, lat, or latitude.')

    station_lat = df_stations[_lat_col].values.astype(float)
    station_lon = df_stations[_lon_col].values.astype(float)

    # =================================================================
    # PS.3) Build PS -> anchor STO mapping
    # =================================================================
    #   ps_anchor[ps_idx] = HPP index of the paired STO reservoir
    #   Each PS must be paired to exactly one non-PS STO station.

    ps_anchor = {}  # ps HPP index -> anchor HPP index

    # ── helper: indices of all non-PS STO stations ──
    sto_indices = [i for i in range(HPP_number)
                   if type_unified[i] != 'PS'
                   and HPP_category[i] in ('A', 'B')]

    # ── helper: indices of all non-PS stations (STO or RoR) ──
    non_ps_indices = [i for i in range(HPP_number) if type_unified[i] != 'PS']

    if ps_mode == 'nearest':
        # ── Pair each PS with the nearest non-PS station (any type) ──
        if len(non_ps_indices) == 0:
            raise ValueError('[PS] nearest mode: no non-PS stations available for pairing.')
        for ps in ps_indices:
            best_dist = np.inf
            best_idx = -1
            for j in non_ps_indices:
                d = _haversine_km(station_lat[ps], station_lon[ps],
                                  station_lat[j], station_lon[j])
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            ps_anchor[ps] = best_idx
            print(f'  [nearest] {HPP_name[ps]} -> {HPP_name[best_idx]}  '
                  f'({best_dist:.1f} km)')

    elif ps_mode == 'nearest_sto':
        # ── Pair each PS with the nearest STO station (category A/B, non-PS) ──
        if len(sto_indices) == 0:
            raise ValueError('[PS] nearest_sto mode: no STO (non-PS) stations '
                             'available for pairing.')
        for ps in ps_indices:
            best_dist = np.inf
            best_idx = -1
            for j in sto_indices:
                d = _haversine_km(station_lat[ps], station_lon[ps],
                                  station_lat[j], station_lon[j])
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            ps_anchor[ps] = best_idx
            print(f'  [nearest_sto] {HPP_name[ps]} -> {HPP_name[best_idx]}  '
                  f'({best_dist:.1f} km)')

    elif ps_mode == 'specified':
        # ── Parse REVUB_PS_PAIRS  ("PS1:STO1;PS2:STO2") ──
        if not ps_pairs_env:
            raise ValueError('[PS] specified mode requires REVUB_PS_PAIRS env var '
                             '(format: "PSname1:STOname1;PSname2:STOname2")')
        _name_to_idx = {HPP_name[i]: i for i in range(HPP_number)}
        for pair_str in ps_pairs_env.split(';'):
            pair_str = pair_str.strip()
            if not pair_str:
                continue
            parts = pair_str.split(':')
            if len(parts) != 2:
                raise ValueError(f'[PS] Malformed pair "{pair_str}". '
                                 f'Expected "PS_name:STO_name".')
            ps_name, sto_name = parts[0].strip(), parts[1].strip()
            if ps_name not in _name_to_idx:
                raise ValueError(f'[PS] PS station "{ps_name}" not found in HPP_name.')
            if sto_name not in _name_to_idx:
                raise ValueError(f'[PS] STO station "{sto_name}" not found in HPP_name.')
            ps_i = _name_to_idx[ps_name]
            sto_i = _name_to_idx[sto_name]
            if type_unified[ps_i] != 'PS':
                warnings.warn(f'[PS] "{ps_name}" has type_unified={type_unified[ps_i]}, '
                              f'expected PS. Proceeding anyway.')
            ps_anchor[ps_i] = sto_i
            print(f'  [specified] {ps_name} -> {sto_name}')

        # Verify that every PS station got a pair
        for ps in ps_indices:
            if ps not in ps_anchor:
                warnings.warn(f'[PS] PS station "{HPP_name[ps]}" has no pair in '
                              f'REVUB_PS_PAIRS - it will be skipped.')

    elif ps_mode == 'cascade':
        # ── Pair each PS with the nearest STO in the same cascade chain ──
        # Walk upstream and downstream from each PS station, collecting STO
        # stations reachable in the cascade topology, then pick the closest
        # by haversine.

        def _cascade_reachable_sto(start_idx):
            """BFS/DFS through cascade topology to find all STO stations
            reachable from start_idx (excluding PS stations)."""
            visited = set()
            queue = [start_idx]
            reachable = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                # Check if this node is a STO (non-PS)
                if node != start_idx and type_unified[node] != 'PS' \
                        and HPP_category[node] in ('A', 'B'):
                    reachable.append(node)
                # Expand upstream
                for u in direct_upstream_indices[node]:
                    if u not in visited:
                        queue.append(u)
                # Expand downstream
                ds = downstream_index[node]
                if ds >= 0 and ds not in visited:
                    queue.append(ds)
            return reachable

        for ps in ps_indices:
            candidates = _cascade_reachable_sto(ps)
            if not candidates:
                # Fall back to nearest STO globally
                warnings.warn(f'[PS] cascade mode: no STO in cascade chain for '
                              f'"{HPP_name[ps]}". Falling back to nearest_sto.')
                best_dist = np.inf
                best_idx = -1
                for j in sto_indices:
                    d = _haversine_km(station_lat[ps], station_lon[ps],
                                      station_lat[j], station_lon[j])
                    if d < best_dist:
                        best_dist = d
                        best_idx = j
                if best_idx >= 0:
                    ps_anchor[ps] = best_idx
                    print(f'  [cascade->fallback] {HPP_name[ps]} -> '
                          f'{HPP_name[best_idx]}  ({best_dist:.1f} km)')
            else:
                best_dist = np.inf
                best_idx = -1
                for j in candidates:
                    d = _haversine_km(station_lat[ps], station_lon[ps],
                                      station_lat[j], station_lon[j])
                    if d < best_dist:
                        best_dist = d
                        best_idx = j
                ps_anchor[ps] = best_idx
                print(f'  [cascade] {HPP_name[ps]} -> {HPP_name[best_idx]}  '
                      f'({best_dist:.1f} km)')
    else:
        raise ValueError(f'[PS] Unknown REVUB_PS_MODE: "{ps_mode}". '
                         f'Expected: off, nearest, nearest_sto, specified, cascade.')

    # Filter to PS stations that were successfully paired
    ps_paired = [ps for ps in ps_indices if ps in ps_anchor]
    n_paired = len(ps_paired)
    if n_paired == 0:
        print('[PS] WARNING: No PS stations could be paired. Skipping dispatch.')
        ps_dispatch_ran = False

if ps_dispatch_ran:
    print(f'\n[PS] {n_paired} PS station(s) paired. Starting dispatch.\n')
    t_ps_start = time.time()

    # =================================================================
    # PS.4) Numba kernel - placeholder
    # =================================================================
    # This will be filled in with the actual pump-back + turbine dispatch
    # logic.  For now it returns zero arrays so that the outer loop can
    # be tested end-to-end.

    from numba import njit

    @njit
    def _ps_bal_core(
        # Dimensions
        nY,             # int: number of simulation years
        hrs_by,         # float64[nY]: hours per year
        _T,             # int: max hours in a year (8784)
        _secs,          # float64: seconds per hour (3600)
        _rho,           # float64: water density (kg/m3)
        _g,             # float64: gravitational acceleration (m/s2)
        # STO anchor parameters
        C_OR,           # float64: curtailment-of-release fraction
        elcc,           # float64: ELCC to serve (MW)
        Q_frac,         # float64[_T, nY]: regulated inflow fraction (STO)
        Q_ror,          # float64[_T, nY]: RoR inflow fraction (STO)
        Q_conv_stable,  # float64[_T, nY]: CONV stable outflow (STO)
        Q_env,          # float64[_T, nY]: environmental flow (STO)
        precip,         # float64[_T, nY]: precipitation flux (STO)
        evap,           # float64[_T, nY]: evaporation flux (STO)
        L_norm_arr,     # float64[_T, nY]: normalised load profile
        V_init_sto,     # float64: initial volume upper reservoir (m3)
        V_max_sto,      # float64: max volume upper reservoir (m3)
        Q_turb_max_sto, # float64: max turbine flow STO (m3/s)
        P_rated_sto,    # float64: rated turbine power STO (MW)
        eta_turb_sto,   # float64: turbine efficiency STO
        f_cascade_ds,   # float64: cascade downstream fraction STO
        bc, bd, ba, bb, # float64: bathymetry coefficients STO
        f_spill_v,      # float64: spill threshold (frac of V_max)
        f_stop_v,       # float64: stop threshold (frac of V_max)
        f_restart_v,    # float64: restart threshold (frac of V_max)
        _mu,            # float64: mu parameter STO
        is_B,           # bool: True if STO is category B
        # PS parameters
        P_rated_pump,   # float64: rated pump power (MW)
        Q_pump_max,     # float64: max pump flow (m3/s)
        eta_pump_ps,    # float64: pump efficiency
        V_upper_max_ps, # float64: max volume of upper reservoir (m3)
        h_max_ps,       # float64: PS head (m)
        P_rated_turb_ps,# float64: PS turbine rated power (MW)
        Q_turb_max_ps,  # float64: PS max turbine flow (m3/s)
        eta_turb_ps,    # float64: PS turbine efficiency
        evap_loss_ps,   # float64: net evaporation loss rate (m3/s)
        Q_in_ps_river,  # float64[_T, nY]: river flow at PS station (pump constraint)
    ):
        """
        Combined BAL + PS dispatch kernel.

        STO dispatch is identical to _bal_core(). After STO produces
        P_stable + P_RoR + P_flexible, PS covers the remaining deficit
        (generating from upper reservoir) or stores surplus (pumping
        from river to upper reservoir).
        """
        V = np.full((_T + 1, nY), np.nan)
        A = np.full((_T + 1, nY), np.nan)
        h = np.full((_T + 1, nY), np.nan)
        Qs = np.zeros((_T, nY)); Qf = np.zeros((_T, nY))
        Qsp = np.zeros((_T, nY)); Qo = np.zeros((_T, nY))
        Ps = np.zeros((_T, nY)); Pf = np.zeros((_T, nY)); Pr = np.zeros((_T, nY))
        curt = np.ones((_T, nY))
        P_ps_turb = np.zeros((_T, nY))
        P_ps_pump = np.zeros((_T, nY))
        V_ps_upper = np.full((_T + 1, nY), np.nan)
        n_outage = 0; n_total = 0

        for y in range(nY):
            nhrs = int(hrs_by[y])
            # ── STO init (identical to _bal_core) ──
            if y == 0:
                V[0, y] = V_init_sto
            else:
                for k in range(_T, -1, -1):
                    if not np.isnan(V[k, y - 1]):
                        V[0, y] = V[k, y - 1]; break
            vl = f_cascade_ds * V[0, y]
            V_km3 = vl / 1e9
            h[0, y] = (V_km3 / bc) ** (1.0 / bd) if V_km3 > 0 and bc > 0 and bd > 0 else 0.0
            A[0, y] = ba * (h[0, y] ** bb) * 1e6 if h[0, y] > 0 and ba > 0 else 0.0
            # ── PS upper reservoir: year 0 half-full, carry over after ──
            if y == 0:
                V_ps_upper[0, y] = 0.5 * V_upper_max_ps
            else:
                for k in range(_T, -1, -1):
                    if not np.isnan(V_ps_upper[k, y - 1]):
                        V_ps_upper[0, y] = V_ps_upper[k, y - 1]; break

            for n in range(nhrs):
                # ═══════════════════════════════════════════
                # STEP 1: STO dispatch (identical to _bal_core)
                # ═══════════════════════════════════════════
                qs = max((1.0 - C_OR) * Q_conv_stable[n, y] * curt[n, y], Q_env[n, y])
                Qs[n, y] = qs
                ps_sto = min(qs, Q_turb_max_sto) * eta_turb_sto * _rho * _g * h[n, y] / 1e6
                Ps[n, y] = ps_sto
                qra = min(Q_ror[n, y], max(0.0, Q_turb_max_sto - qs))
                pr = max(0.0, min(qra * eta_turb_sto * _rho * _g * h[n, y] / 1e6,
                                  P_rated_sto - ps_sto))
                Pr[n, y] = pr
                Lh = elcc * L_norm_arr[n, y] if not np.isnan(L_norm_arr[n, y]) else 0.0
                deficit = Lh - ps_sto - pr
                pf = 0.0
                if deficit > 0.0:
                    qpf = max(0.0, Q_turb_max_sto - qs - qra) * curt[n, y]
                    ppf = qpf * eta_turb_sto * _rho * _g * h[n, y] / 1e6
                    pf = min(deficit, ppf)
                Pf[n, y] = pf
                qf = pf * 1e6 / (eta_turb_sto * _rho * _g * h[n, y]) if h[n, y] > 0 else 0.0
                Qf[n, y] = qf

                # STO spill
                vfrac = V[n, y] / V_max_sto if V_max_sto > 0 else 0.0
                sp = 0.0
                if vfrac >= f_spill_v:
                    sp = max(0.0, (Q_frac[n, y] + (precip[n, y] - evap[n, y]) * A[n, y] / _rho)
                             * (1.0 + _mu) - qs - qf)
                Qsp[n, y] = sp
                Qo[n, y] = qs + qf + sp + Q_ror[n, y]

                # STO water balance
                vnew = V[n, y] + (Q_frac[n, y] - qs - qf - sp
                                  + (precip[n, y] - evap[n, y]) * A[n, y] / _rho) * _secs
                if vnew < 0:
                    over = (-vnew) / _secs
                    sr = min(sp, over); over -= sr; Qsp[n, y] -= sr; sp -= sr
                    fr = min(qf, over); over -= fr; Qf[n, y] -= fr; qf -= fr
                    pf = qf * eta_turb_sto * _rho * _g * h[n, y] / 1e6 if h[n, y] > 0 else 0.0
                    Pf[n, y] = pf
                    # Q_env is a release target, not guaranteed water. If storage and inflow cannot
                    # supply it, curtail stable release to the physically available amount.
                    if over > 0.0:
                        qsr = min(qs, over)
                        if qsr > 0.0:
                            qs -= qsr; over -= qsr; Qs[n, y] = qs
                            Ps[n, y] = min(qs, Q_turb_max_sto) * eta_turb_sto * _rho * _g * h[n, y] / 1e6
                            ps_sto = Ps[n, y]  # keep the local in sync - STEP 2 PS gap/outage uses ps_sto
                    Qo[n, y] = qs + qf + sp + Q_ror[n, y]
                    vnew = 0.0
                if vnew > V_max_sto:
                    vex = vnew - V_max_sto
                    Qsp[n, y] += vex / _secs; Qo[n, y] += vex / _secs
                    vnew = V_max_sto
                V[n + 1, y] = vnew
                vl2 = f_cascade_ds * vnew; Vk2 = vl2 / 1e9
                h[n + 1, y] = (Vk2 / bc) ** (1.0 / bd) if Vk2 > 0 and bc > 0 and bd > 0 else 0.0
                A[n + 1, y] = ba * (h[n + 1, y] ** bb) * 1e6 if h[n + 1, y] > 0 and ba > 0 else 0.0

                # Proportional curtailment (identical to _bal_core)
                vfrac_new = vnew / V_max_sto if V_max_sto > 0 else 1.0
                if f_restart_v > f_stop_v:
                    nv = max(0.0, min(1.0, (vfrac_new - f_stop_v) / (f_restart_v - f_stop_v)))
                else:
                    nv = 0.0 if vfrac_new < f_stop_v else 1.0
                if n < nhrs - 1: curt[n + 1, y] = nv
                elif y < nY - 1: curt[0, y + 1] = nv

                # ═══════════════════════════════════════════
                # STEP 2: PS dispatch
                # ═══════════════════════════════════════════
                P_sto_total = ps_sto + pr + pf
                gap = Lh - P_sto_total  # >0 = deficit, <0 = excess
                p_gen = 0.0; p_pump = 0.0

                if gap > 0.0 and h_max_ps > 0.0:
                    # PS generates: release water from upper reservoir → river
                    _vu = V_ps_upper[n, y]
                    if not np.isfinite(_vu): _vu = 0.0
                    q_avail = min(Q_turb_max_ps, max(0.0, (_vu - evap_loss_ps * _secs) / _secs))
                    q_gen = max(q_avail, 0.0)
                    p_max = min(q_gen * eta_turb_ps * _rho * _g * h_max_ps / 1e6,
                                P_rated_turb_ps)
                    p_gen = min(gap, max(0.0, p_max))
                    q_gen_actual = p_gen * 1e6 / (eta_turb_ps * _rho * _g * h_max_ps) if p_gen > 0 else 0.0
                    V_ps_upper[n + 1, y] = V_ps_upper[n, y] - q_gen_actual * _secs - evap_loss_ps * _secs

                elif gap < 0.0 and h_max_ps > 0.0:
                    # PS pumps: take water from river → store in upper reservoir
                    # Constrained by: pump capacity, upper reservoir space, AND river flow
                    excess = -gap
                    _vu2 = V_ps_upper[n, y]
                    if not np.isfinite(_vu2): _vu2 = 0.0
                    q_upper_space = (V_upper_max_ps - _vu2) / _secs
                    _qr = Q_in_ps_river[n, y]
                    if not np.isfinite(_qr) or _qr < 0.0: _qr = 0.0
                    q_river_avail = _qr * 0.9  # leave 10% for environmental flow
                    q_pump = min(Q_pump_max, max(q_upper_space, 0.0), max(q_river_avail, 0.0))
                    p_max = min(q_pump * _rho * _g * h_max_ps / (eta_pump_ps * 1e6),
                                P_rated_pump)
                    p_pump = min(excess, max(0.0, p_max))
                    q_pump_actual = p_pump * eta_pump_ps * 1e6 / (_rho * _g * h_max_ps) if p_pump > 0 else 0.0
                    V_ps_upper[n + 1, y] = V_ps_upper[n, y] + q_pump_actual * _secs - evap_loss_ps * _secs

                else:
                    V_ps_upper[n + 1, y] = V_ps_upper[n, y] - evap_loss_ps * _secs

                # Clamp upper reservoir
                vu = V_ps_upper[n + 1, y]
                if vu < 0: vu = 0.0
                if vu > V_upper_max_ps: vu = V_upper_max_ps
                V_ps_upper[n + 1, y] = vu

                P_ps_turb[n, y] = p_gen
                P_ps_pump[n, y] = p_pump

                # ═══════════════════════════════════════════
                # STEP 3: Outage - output-based (not curt-based)
                # ═══════════════════════════════════════════
                # PS fills deficits whenever STO output < load, not just
                # when STO is curtailed. So outage = total output < load.
                n_total += 1
                total_output = P_sto_total + p_gen
                if not (total_output >= Lh - 1e-6):
                    n_outage += 1

        outage = n_outage / max(n_total, 1)
        return (V, A, h, Qs, Qf, Qsp, Qo, Ps, Pf, Pr, curt,
                P_ps_turb, P_ps_pump, V_ps_upper, outage)

    # ── Compile Numba kernel with dummy data ──
    print('  Compiling PS Numba kernel...', end=' ', flush=True)
    _dummy_ps = _ps_bal_core(
        1, np.array([24.0]), 8784, 3600.0, 1000.0, 9.81,
        0.3, 100.0,
        np.ones((8784, 1)), np.ones((8784, 1)), np.ones((8784, 1)),
        np.zeros((8784, 1)), np.zeros((8784, 1)), np.zeros((8784, 1)),
        np.ones((8784, 1)),
        1e9, 2e9, 1000.0, 100.0, 0.88,
        1.0, 0.1, 2.0, 0.1, 1.5,
        0.85, 0.20, 0.25, 0.10, True,
        100.0, 50.0, 0.90, 5e6, 200.0, 100.0, 50.0, 0.88, 0.0, np.ones((8784, 1)),
    )
    print('done')

    # =================================================================
    # PS.4b) Fix PS river: trace downstream to find adequate flow
    # =================================================================
    # Open-loop PS takes water from a nearby large river, not a tiny
    # tributary. If the GRFR-snapped location is too small, trace
    # downstream until finding a river with sufficient flow.

    # PS river source = PS station itself (relocation done in build_hydro_topology)
    ps_river_source = {ps: ps for ps in ps_paired}

    # =================================================================
    # PS.5) Run PS dispatch for each paired PS station
    # =================================================================

    # ── Preallocate result arrays ──
    P_PS_turb_hourly = np.zeros((_T, _Y, n_paired))
    P_PS_pump_hourly = np.zeros((_T, _Y, n_paired))
    V_PS_upper_hourly = np.zeros((_T + 1, _Y, n_paired))
    # STO output from combined simulation (differs from B's BAL at different ELCC)
    P_STO_stable_combined = np.zeros((_T, _Y, n_paired))
    P_STO_flex_combined = np.zeros((_T, _Y, n_paired))
    P_STO_ror_combined = np.zeros((_T, _Y, n_paired))
    V_STO_combined = np.zeros((_T + 1, _Y, n_paired))

    ELCC_PS = np.full(n_paired, np.nan)        # combined ELCC (STO + PS)
    ELCC_solo = np.full(n_paired, np.nan)      # solo ELCC of anchor STO
    outage_PS = np.zeros(n_paired)
    ps_paired_names = []
    anchor_names = []

    N_PS_ELCC = int(os.environ.get('REVUB_N_PS_ELCC', '100'))

    for p_idx, ps in enumerate(ps_paired):
        anchor = ps_anchor[ps]
        ps_paired_names.append(HPP_name[ps])
        anchor_names.append(HPP_name[anchor])

        solo_elcc_curt = ELCC_opt[anchor] if np.isfinite(ELCC_opt[anchor]) else 0.0

        # PS upper reservoir volume
        v_upper_ps = V_max[ps]
        if (not np.isfinite(v_upper_ps) or v_upper_ps <= 0) and h_max[ps] > 0 and P_r_turb[ps] > 0:
            v_upper_ps = P_r_turb[ps] * 1e6 * 8 * 3600 / (eta_turb[ps] * rho * g * h_max[ps])
        if not np.isfinite(v_upper_ps) or v_upper_ps <= 0:
            v_upper_ps = 1e6
        storage_hrs = v_upper_ps * rho * g * h_max[ps] * eta_turb[ps] / (P_r_turb[ps] * 1e6 * 3600) if P_r_turb[ps] > 0 and h_max[ps] > 0 else 0

        # PS net evaporation per reservoir (m³/s)
        # Ref: Zhao & Gao (2019) Environ. Res. Lett. 14:124062
        # Ref: Pokhrel et al. (2021) Nature Rev. Earth Environ. 2:579-594
        # Net evap (evap - precip) by latitude band, applied to each reservoir
        abs_lat_ps = abs(float(df_stations.iloc[ps].get('lat_unified', 30)))
        if abs_lat_ps <= 15: net_evap_mm_yr = 1200.0
        elif abs_lat_ps <= 30: net_evap_mm_yr = 1500.0
        elif abs_lat_ps <= 45: net_evap_mm_yr = 600.0
        elif abs_lat_ps <= 55: net_evap_mm_yr = 300.0
        else: net_evap_mm_yr = 150.0
        # PS reservoir area: use ps_area_km2 if available, else estimate
        ps_area_m2 = float(df_stations.iloc[ps].get('ps_area_km2', 0)) * 1e6
        if not np.isfinite(ps_area_m2) or ps_area_m2 <= 0:
            ps_area_m2 = 0.5e6  # 0.5 km² default for PS (small reservoir)
        evap_loss_ps = net_evap_mm_yr * 1e-3 / (365.25 * 24 * 3600) * ps_area_m2  # m³/s per reservoir

        # ── Prepare STO anchor inputs early (needed for solo search) ──
        C_OR_anchor = C_OR_opt[anchor] if 'C_OR_opt' in dir() and np.isfinite(C_OR_opt[anchor]) else 0.20
        if HPP_category[anchor] == 'A':
            Qfr = Q_in_BAL_cascade[:, :, anchor].copy()
            Qrr = np.zeros_like(Qfr)
        else:
            Qfr = f_reg[anchor] * Q_in_BAL_cascade[:, :, anchor].copy()
            Qrr = Q_in_BAL_cascade[:, :, anchor] - Qfr

        # Compute solo output-based ELCC (binary search, no PS)
        def _solo_outage(et):
            r = _ps_bal_core(
                _Y, hrs_byyear, _T, secs_hr, rho, g,
                C_OR_anchor, et, Qfr.copy(), Qrr.copy(),
                Q_CONV_stable_hourly[:, :, anchor], _stable_env_release(anchor, Qrr),
                precipitation_flux_hourly[:, :, anchor], evaporation_flux_hourly[:, :, anchor],
                L_norm, V_CONV_hourly[0, 0, anchor], V_max_cumul[anchor], Q_max_turb[anchor],
                P_r_turb[anchor], eta_turb[anchor],
                f_cascade_downstream[anchor], bathy_c[anchor], bathy_d[anchor],
                bathy_a[anchor], bathy_b[anchor],
                f_spill[anchor], f_stop_cumul[anchor], f_restart_cumul[anchor], mu[anchor],
                HPP_category[anchor] == 'B',
                0.0, 0.0, 0.90, 1e6, 100.0, 0.0, 0.0, 0.88, 0.0, np.zeros((_T, _Y)))
            return r[-1]
        solo_elcc = 0.0; lo_se, hi_se = 0.0, solo_elcc_curt
        for _ in range(200):
            if hi_se - lo_se < 0.1: break
            mid_se = (lo_se + hi_se) / 2
            if _solo_outage(mid_se) <= OUTAGE_MAX:
                solo_elcc = mid_se; lo_se = mid_se
            else:
                hi_se = mid_se
        ELCC_solo[p_idx] = solo_elcc

        print(f'PS {p_idx+1}/{n_paired}: {HPP_name[ps][:35]} <-> {HPP_name[anchor][:35]}  '
              f'(solo_output={solo_elcc:.0f}MW, solo_curt={solo_elcc_curt:.0f}MW, PS={P_r_turb[ps]:.0f}MW, storage={storage_hrs:.0f}h)')

        # ── ELCC search: linear scan from high to low ──
        # PS can't add more than its rated power continuously, and is limited
        # by storage. Reasonable ceiling: 2× the curt-based solo ELCC.
        elcc_hi = (P_r_turb[anchor] + P_r_turb[ps]) / max(L_norm_mean, 0.01)
        best_elcc = solo_elcc
        best_outage = 1.0

        def _pair_outage(et):
            r = _ps_bal_core(
                _Y, hrs_byyear, _T, secs_hr, rho, g,
                C_OR_anchor, et, Qfr.copy(), Qrr.copy(),
                Q_CONV_stable_hourly[:,:,anchor], _stable_env_release(anchor, Qrr),
                precipitation_flux_hourly[:,:,anchor], evaporation_flux_hourly[:,:,anchor],
                L_norm, V_CONV_hourly[0,0,anchor], V_max_cumul[anchor], Q_max_turb[anchor],
                P_r_turb[anchor], eta_turb[anchor],
                f_cascade_downstream[anchor], bathy_c[anchor], bathy_d[anchor],
                bathy_a[anchor], bathy_b[anchor],
                f_spill[anchor], f_stop_cumul[anchor], f_restart_cumul[anchor], mu[anchor],
                HPP_category[anchor]=='B',
                P_r_pump[ps], Q_max_pump[ps], eta_pump[ps], v_upper_ps,
                h_max[ps], P_r_turb[ps], Q_max_turb[ps], eta_turb[ps],
                evap_loss_ps, Q_in_BAL_cascade[:,:,ps])
            return r[-1]
        lo_p, hi_p = 0.0, elcc_hi
        for _ in range(200):
            if hi_p - lo_p < 0.1: break
            mid_p = (lo_p + hi_p) / 2
            out_p = _pair_outage(mid_p)
            if out_p <= OUTAGE_MAX:
                best_elcc = mid_p; best_outage = out_p; lo_p = mid_p
            else:
                hi_p = mid_p

        ELCC_PS[p_idx] = best_elcc
        outage_PS[p_idx] = best_outage

        # ── Final run with best ELCC to populate result arrays ──
        result_final = _ps_bal_core(
            _Y, hrs_byyear, _T, secs_hr, rho, g,
            C_OR_anchor, best_elcc,
            Qfr.copy(), Qrr.copy(),
            Q_CONV_stable_hourly[:, :, anchor],
            _stable_env_release(anchor, Qrr),
            precipitation_flux_hourly[:, :, anchor],
            evaporation_flux_hourly[:, :, anchor],
            L_norm,
            V_CONV_hourly[0, 0, anchor],
            V_max_cumul[anchor],
            Q_max_turb[anchor],
            P_r_turb[anchor],
            eta_turb[anchor],
            f_cascade_downstream[anchor],
            bathy_c[anchor], bathy_d[anchor],
            bathy_a[anchor], bathy_b[anchor],
            f_spill[anchor],
            f_stop_cumul[anchor],
            f_restart_cumul[anchor],
            mu[anchor],
            HPP_category[anchor] == 'B',
            P_r_pump[ps],
            Q_max_pump[ps],
            eta_pump[ps],
            v_upper_ps,
            h_max[ps],
            P_r_turb[ps],
            Q_max_turb[ps],
            eta_turb[ps],
            evap_loss_ps,
            Q_in_BAL_cascade[:, :, ps],
        )
        (V_f, A_f, h_f, Qs_f, Qf_f, Qsp_f, Qo_f, Ps_f, Pf_f, Pr_f, curt_f,
         Ppt_f, Ppp_f, Vpu_f, out_f) = result_final

        P_PS_turb_hourly[:, :, p_idx] = Ppt_f
        P_PS_pump_hourly[:, :, p_idx] = Ppp_f
        V_PS_upper_hourly[:, :, p_idx] = Vpu_f
        P_STO_stable_combined[:, :, p_idx] = Ps_f
        P_STO_flex_combined[:, :, p_idx] = Pf_f
        P_STO_ror_combined[:, :, p_idx] = Pr_f
        V_STO_combined[:, :, p_idx] = V_f

        elcc_gain = best_elcc - solo_elcc
        print(f'  -> combined ELCC={best_elcc:.0f} MW  '
              f'(+{elcc_gain:.0f} MW from PS, '
              f'{elcc_gain / max(P_r_turb[ps], 1e-6) * 100:.0f}% of PS capacity)  '
              f'outage={100 * best_outage:.1f}%')

    # =================================================================
    # PS.5b) Open-loop flow impact: update Q_BAL_out and propagate
    # =================================================================
    # PS changes river flow: Q_out = Q_in + Q_gen - Q_pump
    # Q_gen returns water to river (upper→lower→river)
    # Q_pump removes water from river (river→lower→upper)

    # Snapshot the CLEAN (no-PS) BAL dispatch BEFORE PS.5b writes per-pair PS flow into the global
    # hourly arrays - so the chain-level no-PS baseline (and the chain optimization's P_chain) isn't
    # contaminated by per-pair PS (which would inflate the baseline and double-count PS on the chain).
    P_BAL_stable_clean = P_BAL_hydro_stable_hourly.copy()
    P_BAL_flex_clean = P_BAL_hydro_flexible_hourly.copy()
    P_BAL_ror_clean = P_BAL_hydro_RoR_hourly.copy()

    print(f'\n  Updating river flow for open-loop PS...')
    for p_idx, ps in enumerate(ps_paired):
        # Compute Q_gen and Q_pump in m³/s from power
        h_ps_val = h_max[ps]
        if h_ps_val <= 0: continue
        Q_gen_hourly = P_PS_turb_hourly[:, :, p_idx] * 1e6 / (eta_turb[ps] * rho * g * h_ps_val)
        Q_pump_hourly = P_PS_pump_hourly[:, :, p_idx] * eta_pump[ps] * 1e6 / (rho * g * h_ps_val)

        # Update PS station's Q_BAL_out: pass-through + net PS flow
        Q_BAL_out_hourly[:, :, ps] = Q_in_BAL_cascade[:, :, ps] + Q_gen_hourly - Q_pump_hourly
        Q_BAL_out_hourly[:, :, ps] = np.maximum(Q_BAL_out_hourly[:, :, ps], 0.0)

        net_q = np.nanmean(Q_gen_hourly - Q_pump_hourly)
        print(f'    {HPP_name[ps][:40]}: mean ΔQ = {net_q:+.2f} m³/s '
              f'(gen={np.nanmean(Q_gen_hourly):.2f}, pump={np.nanmean(Q_pump_hourly):.2f})')

    # Propagate updated PS flow to downstream stations
    # Re-compute Q_in_BAL_cascade for stations downstream of any PS
    ps_set = set(ps_paired)
    affected_downstream = set()
    for ps in ps_paired:
        ds = downstream_index[ps]
        while ds >= 0 and ds not in affected_downstream:
            affected_downstream.add(ds)
            ds = downstream_index[ds] if ds < HPP_number else -1

    if affected_downstream:
        print(f'  Propagating to {len(affected_downstream)} downstream stations...')
        # Process in topological order (station indices are already topo-sorted by A file)
        for HPP in range(HPP_number):
            if HPP not in affected_downstream:
                continue
            # Re-compute cascade for ALL affected downstream (not just direct PS neighbors)
            Q_in_updated = _cascade_inflow(HPP, Q_BAL_out_hourly)

            delta = np.nanmax(np.abs(Q_in_updated - Q_in_BAL_cascade[:, :, HPP]))
            Q_in_BAL_cascade[:, :, HPP] = Q_in_updated

            if delta < 0.01:
                continue

            # Re-run BAL for this downstream station with updated Q_in
            if type_unified[HPP] == 'PS':
                # A downstream PS must RE-APPLY its own net flow (Q_gen - Q_pump) on top of the updated
                # inflow - not just pass through - otherwise an upstream PS's propagation overwrites this
                # PS's own pumping/generation and breaks water conservation for chained PS.
                if HPP in ps_set and h_max[HPP] > 0:
                    _pj = list(ps_paired).index(HPP)
                    _Qg = P_PS_turb_hourly[:, :, _pj] * 1e6 / (eta_turb[HPP] * rho * g * h_max[HPP])
                    _Qp = P_PS_pump_hourly[:, :, _pj] * eta_pump[HPP] * 1e6 / (rho * g * h_max[HPP])
                    Q_BAL_out_hourly[:, :, HPP] = np.maximum(Q_in_updated + _Qg - _Qp, 0.0)
                else:
                    Q_BAL_out_hourly[:, :, HPP] = Q_in_updated
                print(f'    {HPP_name[HPP][:40]}: ΔQ_in = {delta:+.2f} m³/s [PS net-flow re-applied]')
            elif HPP_category[HPP] == 'RoR':
                Q_BAL_out_hourly[:, :, HPP] = Q_in_updated
                P_BAL_hydro_RoR_hourly[:, :, HPP] = np.fmin(
                    np.fmin(Q_in_updated, Q_max_turb[HPP]) * eta_turb[HPP] * rho * g * h_max[HPP] / 1e6,
                    P_r_turb[HPP])
                print(f'    {HPP_name[HPP][:40]}: ΔQ_in = {delta:+.2f} m³/s [RoR re-run]')
            elif HPP_category[HPP] in ('A', 'B'):
                # Re-run ELCC search + BAL for STO with updated cascade inflow
                c_or_ds = min(0.30, max(0.05, 1.0 - d_min[HPP]))
                old_elcc = ELCC_opt[HPP]
                if HPP_category[HPP] == 'A':
                    Qfr_ds = Q_in_updated.copy()
                    Qrr_ds = np.zeros_like(Qfr_ds)
                else:
                    Qfr_ds = f_reg[HPP] * Q_in_updated.copy()
                    Qrr_ds = Q_in_updated - Qfr_ds
                # Re-search ELCC with updated flow. The upper bound must reflect the PS-UPDATED inflow
                # (which can exceed the pre-PS average), else a higher feasible ELCC gets truncated and
                # the cascade gain is under-estimated.
                P_avg_ds = np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP] + P_CONV_hydro_RoR_hourly[:,:,HPP])
                P_avg_upd = np.nanmean(np.fmin(Q_in_updated, Q_max_turb[HPP]) * eta_turb[HPP] * rho * g * h_max[HPP] / 1e6)
                _hi_base = max(P_avg_ds, P_avg_upd)
                elcc_hi_ds = _hi_base / L_norm_mean if L_norm_mean > 0 else max(old_elcc, _hi_base)
                new_elcc = 0.0
                for et in np.linspace(elcc_hi_ds, 0, N_ELCC + 1):
                    res_t = _bal_core(
                        _Y, hrs_byyear, c_or_ds, et, Qfr_ds.copy(), Qrr_ds.copy(),
                        Q_CONV_stable_hourly[:,:,HPP], _stable_env_release(HPP, Qrr_ds),
                        precipitation_flux_hourly[:,:,HPP], evaporation_flux_hourly[:,:,HPP],
                        L_norm, V_CONV_hourly[0,0,HPP], V_max_cumul[HPP], Q_max_turb[HPP],
                        P_r_turb[HPP], eta_turb[HPP], rho, g, secs_hr,
                        f_cascade_downstream[HPP], bathy_c[HPP], bathy_d[HPP],
                        bathy_a[HPP], bathy_b[HPP],
                        f_spill[HPP], f_stop_cumul[HPP], f_restart_cumul[HPP], mu[HPP],
                        HPP_category[HPP] == 'B', _T)
                    if res_t[-1] <= OUTAGE_MAX:
                        new_elcc = et; break
                ELCC_opt[HPP] = new_elcc
                # Final run at new ELCC
                res_ds = _bal_core(
                    _Y, hrs_byyear, c_or_ds, new_elcc, Qfr_ds.copy(), Qrr_ds.copy(),
                    Q_CONV_stable_hourly[:,:,HPP], _stable_env_release(HPP, Qrr_ds),
                    precipitation_flux_hourly[:,:,HPP], evaporation_flux_hourly[:,:,HPP],
                    L_norm, V_CONV_hourly[0,0,HPP], V_max_cumul[HPP], Q_max_turb[HPP],
                    P_r_turb[HPP], eta_turb[HPP], rho, g, secs_hr,
                    f_cascade_downstream[HPP], bathy_c[HPP], bathy_d[HPP],
                    bathy_a[HPP], bathy_b[HPP],
                    f_spill[HPP], f_stop_cumul[HPP], f_restart_cumul[HPP], mu[HPP],
                    HPP_category[HPP] == 'B', _T)
                V_ds, A_ds, h_ds, Qs_ds, Qf_ds, Qsp_ds, Qo_ds, Ps_ds, Pf_ds, Pr_ds, curt_ds, out_ds = res_ds
                Q_BAL_out_hourly[:,:,HPP] = Qo_ds
                Q_BAL_stable_hourly[:,:,HPP] = Qs_ds
                Q_BAL_flexible_hourly[:,:,HPP] = Qf_ds
                Q_BAL_spill_hourly[:,:,HPP] = Qsp_ds
                V_BAL_hourly[:,:,HPP] = V_ds
                P_BAL_hydro_stable_hourly[:,:,HPP] = Ps_ds
                P_BAL_hydro_flexible_hourly[:,:,HPP] = Pf_ds
                P_BAL_hydro_RoR_hourly[:,:,HPP] = Pr_ds
                elcc_change = new_elcc - old_elcc
                print(f'    {HPP_name[HPP][:40]}: ΔQ_in={delta:+.2f} ELCC {old_elcc:.0f}→{new_elcc:.0f} ({elcc_change:+.0f}MW)')

    # =================================================================
    # PS.5c) Chain-level PS ELCC - PS serves entire cascade chain
    # =================================================================
    # Instead of pairing PS with one anchor STO, PS fills the deficit
    # of the ENTIRE chain. This is physically correct: PS is on the grid
    # and can compensate for any station's shortfall.
    #
    # P_chain(t) = Σ P_STO_i(t) for all stations in chain
    # gap(t) = chain_ELCC × L_norm(t) - P_chain(t)
    # PS fills gap → chain outage ≤ 5%

    @njit
    def _chain_ps_elcc(nY, hrs_by, _T, _secs, _rho, _g,
                       P_chain, L_norm_arr, elcc,
                       P_rated_gen, Q_turb_max_ps, eta_gen,
                       P_rated_pump, Q_pump_max_ps, eta_pump,
                       V_upper_max, h_ps, V_upper_init,
                       Q_river_ps, evap_loss):
        V_upper = np.full((_T+1, nY), np.nan)
        P_gen = np.zeros((_T, nY)); P_pump = np.zeros((_T, nY))
        n_outage = 0; n_total = 0
        for y in range(nY):
            nhrs = int(hrs_by[y])
            if y == 0:
                V_upper[0, y] = V_upper_init
            else:
                V_upper[0, y] = V_upper[int(hrs_by[y-1]), y-1]
            for t in range(nhrs):
                Lh = elcc * L_norm_arr[t, y] if not np.isnan(L_norm_arr[t, y]) else 0.0
                gap = Lh - P_chain[t, y]
                p_gen = 0.0; p_pump = 0.0
                evap_vol = evap_loss * _secs
                if gap > 0 and h_ps > 0:
                    _vuc = V_upper[t, y]
                    if not np.isfinite(_vuc): _vuc = 0.0
                    q_avail = min(Q_turb_max_ps, max(0.0, (_vuc - evap_vol) / _secs))
                    p_max = min(max(q_avail, 0.0) * eta_gen * _rho * _g * h_ps / 1e6, P_rated_gen)
                    p_gen = min(gap, max(0.0, p_max))
                    q_act = p_gen * 1e6 / (eta_gen * _rho * _g * h_ps) if p_gen > 0 else 0.0
                    V_upper[t+1, y] = V_upper[t, y] - q_act * _secs - evap_loss * _secs
                elif gap < 0 and h_ps > 0:
                    q_space = max((V_upper_max - V_upper[t, y]) / _secs, 0.0)
                    _qrc = Q_river_ps[t, y]
                    if not np.isfinite(_qrc) or _qrc < 0.0: _qrc = 0.0
                    q_river = _qrc * 0.9
                    q_pump = min(Q_pump_max_ps, q_space, max(q_river, 0.0))
                    p_max = min(q_pump * _rho * _g * h_ps / (eta_pump * 1e6), P_rated_pump)
                    p_pump = min(-gap, max(0.0, p_max))
                    q_act = p_pump * eta_pump * 1e6 / (_rho * _g * h_ps) if p_pump > 0 else 0.0
                    V_upper[t+1, y] = V_upper[t, y] + q_act * _secs - evap_loss * _secs
                else:
                    V_upper[t+1, y] = V_upper[t, y] - evap_loss * _secs
                vu = V_upper[t+1, y]
                if vu < 0: vu = 0.0
                if vu > V_upper_max: vu = V_upper_max
                V_upper[t+1, y] = vu
                P_gen[t, y] = p_gen; P_pump[t, y] = p_pump
                n_total += 1
                if P_chain[t, y] + p_gen < Lh - 1e-6:
                    n_outage += 1
        outage = n_outage / max(n_total, 1)
        return P_gen, P_pump, V_upper, outage

    # Compile
    _chain_ps_elcc(1, np.array([24.0]), 8784, 3600.0, 1000.0, 9.81,
                   np.ones((8784,1)), np.ones((8784,1)), 10.0,
                   100.0, 100.0, 0.88, 100.0, 100.0, 0.90,
                   1e9, 100.0, 5e8, np.ones((8784,1)), 0.0)

    # Group PS by cascade chain
    def _get_chain(start):
        c = []; v = set()
        def _up(h):
            if h in v: return
            v.add(h)
            for u in direct_upstream_indices[h]: _up(u)
            c.append(h)
        _up(start)
        ds = downstream_index[start]
        while 0 <= ds < HPP_number and ds not in v:
            v.add(ds); c.append(ds)
            for u in direct_upstream_indices[ds]:
                if u not in v: _up(u)
            ds = downstream_index[ds]
        return c

    # Build list of chains to optimize
    chain_jobs = []
    seen_ps = set()
    for p_idx, ps in enumerate(ps_paired):
        if ps in seen_ps: continue
        chain = _get_chain(ps)
        ps_in_chain = [h for h in chain if type_unified[h] == 'PS' and h in set(ps_paired)]
        sto_in_chain = [h for h in chain if type_unified[h] != 'PS']
        seen_ps.update(ps_in_chain)
        if len(sto_in_chain) > 0:
            chain_jobs.append({'chain': chain, 'sto': sto_in_chain, 'ps_in': ps_in_chain})

    print(f'\n  Chain-level PS ELCC ({len(chain_jobs)} chains):')

    def _process_one_chain(cj):
        """Process a single chain: compute solo + PS ELCC with closed-loop convergence."""
        chain = cj['chain']; sto_in_chain = cj['sto']; ps_in_chain = cj['ps_in']

        if len(sto_in_chain) == 0: return None

        # Sum chain power output from the CLEAN (pre-PS) BAL dispatch - the chain optimization applies
        # ps_main on top, so using the PS.5b-modified arrays would double-count PS and contaminate the
        # no-PS baseline (img1 #4).
        P_chain = np.zeros((_T, _Y))
        for h in sto_in_chain:
            P_chain += np.nan_to_num(P_BAL_stable_clean[:,:,h]) \
                     + np.nan_to_num(P_BAL_flex_clean[:,:,h]) \
                     + np.nan_to_num(P_BAL_ror_clean[:,:,h])

        solo_total = sum(ELCC_opt[h] for h in chain if np.isfinite(ELCC_opt[h]))

        # Use the largest PS in chain for dispatch
        ps_main = max(ps_in_chain, key=lambda h: P_r_turb[h])
        v_upper_ps = V_max[ps_main] if V_max[ps_main] > 0 else 1e6
        storage_hrs = v_upper_ps * rho * g * h_max[ps_main] / (P_r_turb[ps_main] * 1e6 * 3600) if P_r_turb[ps_main] > 0 else 0

        lat_ps = station_lats[ps_main]
        evap_mm_yr = 1200 if abs(lat_ps) <= 15 else (800 if abs(lat_ps) <= 30 else 400)
        A_upper_est = v_upper_ps / 20.0 if v_upper_ps > 0 else 0.0
        evap_loss_ps = (evap_mm_yr / 1000.0) * A_upper_est / (365.25 * 24 * 3600)

        chain_names = ' → '.join(f"{HPP_name[h][:12]}{'★' if type_unified[h]=='PS' else ''}" for h in chain)
        print(f'    Chain: {chain_names}')
        print(f'    Σ individual ELCC = {solo_total:.0f} MW, PS = {HPP_name[ps_main][:25]} ({P_r_turb[ps_main]:.0f}MW, {storage_hrs:.0f}h)')

        # Compute chain-level baseline ELCC WITHOUT PS (coarse+fine)
        lo_s, hi_s = 0.0, solo_total + 50
        chain_solo_elcc = 0.0
        for _ in range(200):
            if hi_s - lo_s < 1.0: break
            mid_s = (lo_s + hi_s) / 2
            _, _, _, out_s = _chain_ps_elcc(
                _Y, hrs_byyear, _T, secs_hr, rho, g,
                P_chain, L_norm, mid_s,
                0.0, 0.0, 0.88, 0.0, 0.0, 0.90,
                1.0, 1.0, 0.0,
                np.zeros((_T, _Y)), 0.0)
            if out_s <= OUTAGE_MAX:
                chain_solo_elcc = mid_s; lo_s = mid_s
            else:
                hi_s = mid_s
        # Fine phase for solo
        lo_sf = max(chain_solo_elcc - 1.0, 0.0); hi_sf = chain_solo_elcc + 1.0
        for _ in range(200):
            if hi_sf - lo_sf < 0.1: break
            mid_sf = (lo_sf + hi_sf) / 2
            _, _, _, out_sf = _chain_ps_elcc(
                _Y, hrs_byyear, _T, secs_hr, rho, g,
                P_chain, L_norm, mid_sf,
                0.0, 0.0, 0.88, 0.0, 0.0, 0.90,
                1.0, 1.0, 0.0,
                np.zeros((_T, _Y)), 0.0)
            if out_sf <= OUTAGE_MAX:
                chain_solo_elcc = mid_sf; lo_sf = mid_sf
            else:
                hi_sf = mid_sf
        print(f'    Chain ELCC (no PS) = {chain_solo_elcc:.1f} MW  (vs Σindividual={solo_total:.0f})')

        # Search chain-level ELCC with cascade feedback
        # For each trial: PS dispatch → update river flow → re-run downstream BAL → recompute P_chain
        # Skip PS with negligible storage (<1 hour at rated capacity)
        # --- Multi-PS chain optimization (severe #1): dispatch ALL PS in the chain (not just the
        #     largest), propagate every PS's net river flow through the cascade, re-run downstream STO,
        #     and capture the per-station dispatch for write-back (#2). Runs on the grid curve L_norm. ---
        def _ndc(h):
            n = 0; cur = downstream_index[h]; seen = 0
            while 0 <= cur < HPP_number and seen <= HPP_number:
                n += 1; cur = downstream_index[cur]; seen += 1
            return n
        ps_list = sorted([h for h in ps_in_chain if h_max[h] > 0 and V_max[h] > 0
                          and (V_max[h] * rho * g * h_max[h] / (P_r_turb[h] * 1e6 * 3600) if P_r_turb[h] > 0 else 0) >= 1.0],
                         key=_ndc, reverse=True)
        if not ps_list:
            print(f'    Skipped: no PS with usable storage (>=1h)')
            return {
                'chain': chain, 'solo': solo_total, 'combined': chain_solo_elcc,
                'gain': 0, 'ps': ps_main, 'P_gen': np.zeros((_T, _Y)),
                'P_pump': np.zeros((_T, _Y)), 'V_upper': np.zeros((_T+1, _Y)),
                'chain_names': chain_names, 'chain_solo_elcc': chain_solo_elcc,
                'out_f': 0, 'conv_iters': 0, 'conv_status': 'skipped',
                'ps_main_name': HPP_name[ps_main], 'ps_cap': 0.0,
                'disp': {}, 'ps_disp': {},
            }
        def _ps_param(ps):
            v_up = V_max[ps] if V_max[ps] > 0 else 1e6
            _lat = station_lats[ps]
            _emm = 1200 if abs(_lat) <= 15 else (800 if abs(_lat) <= 30 else 400)
            _aup = v_up / 20.0 if v_up > 0 else 0.0
            _evp = (_emm / 1000.0) * _aup / (365.25 * 24 * 3600)
            return v_up, _evp
        ps_param = {ps: _ps_param(ps) for ps in ps_list}
        sto_topo = sorted([h for h in chain if type_unified[h] != 'PS'], key=_ndc, reverse=True)
        chain_order = sorted(chain, key=_ndc, reverse=True)
        elcc_hi = chain_solo_elcc + sum(P_r_turb[ps] for ps in ps_list)
        best_chain_elcc = 0.0
        # Non-PS stations DOWNSTREAM of any PS - only these have their inflow changed by PS dispatch
        # and need re-running. All other chain STO keep their clean BAL dispatch (so the combined
        # output equals the clean baseline when PS don't operate → consistent with chain_solo_elcc).
        _chain_set = set(chain)
        ds_of_ps = set()
        for _ps in ps_list:
            _cur = downstream_index[_ps]; _sn = 0
            while 0 <= _cur < HPP_number and _cur in _chain_set and _sn <= HPP_number:
                if type_unified[_cur] != 'PS': ds_of_ps.add(_cur)
                _cur = downstream_index[_cur]; _sn += 1

        def _qin_C(h, Qout):
            return _cascade_inflow(h, Qout)

        def _clean_out(h):
            return (np.nan_to_num(P_BAL_stable_clean[:, :, h]) + np.nan_to_num(P_BAL_flex_clean[:, :, h])
                    + np.nan_to_num(P_BAL_ror_clean[:, :, h]))

        MAX_ITER = int(os.environ.get('REVUB_CHAIN_GS_MAX_ITER', '200'))
        CONV_TOL_MW = float(os.environ.get('REVUB_CHAIN_GS_TOL_MW', '0.1'))
        CONV_TOL_Q = float(os.environ.get('REVUB_CHAIN_GS_TOL_Q', '0.01'))
        CONV_TOL_V_FRAC = float(os.environ.get('REVUB_CHAIN_GS_TOL_V_FRAC', '1e-3'))
        _Lg = np.nan_to_num(L_norm); _Lmask = np.isfinite(L_norm); _Lntot = int(_Lmask.sum())

        def _eval_chain_elcc(trial_elcc, capture=False, use_ps=True):
            """Multi-PS chain combined-firm: STO members re-run on the cascade flow toward their own
            ELCC_opt; ALL PS dispatch (topo order) to fill the residual combined deficit and write their
            net river flow downstream (Gauss-Seidel). Returns (ps_gen, ps_pump, ps_vu, comb, outage,
            iters, status, per-station disp)."""
            Q_out_local = Q_BAL_out_hourly.copy()
            Psto = {h: _clean_out(h) for h in sto_topo}
            ps_gen = {ps: np.zeros((_T, _Y)) for ps in ps_list}
            ps_pump = {ps: np.zeros((_T, _Y)) for ps in ps_list}
            ps_vu = {}
            Vsto = {h: np.nan_to_num(V_BAL_hourly[:, :, h]).copy() for h in sto_topo}
            disp = {}
            conv_iters = 0; conv_status = 'converged'
            for _it in range(MAX_ITER):
                comb = np.zeros((_T, _Y))
                for h in sto_topo: comb = comb + Psto[h]
                for ps in ps_list: comb = comb + ps_gen[ps] - ps_pump[ps]
                max_power_change = 0.0
                max_flow_change = 0.0
                max_storage_frac_change = 0.0
                for h in chain_order:
                    if type_unified[h] == 'PS':
                        if (not use_ps) or (h not in ps_param):
                            _new_qout = _qin_C(h, Q_out_local)
                            max_flow_change = max(max_flow_change,
                                float(np.nanmax(np.abs(_new_qout - Q_out_local[:, :, h]))))
                            Q_out_local[:, :, h] = _new_qout
                            continue
                        v_up, evp = ps_param[h]
                        q_river = _qin_C(h, Q_out_local)
                        old_net = ps_gen[h] - ps_pump[h]
                        base = comb - old_net
                        Pg, Pp, Vu, _o = _chain_ps_elcc(_Y, hrs_byyear, _T, secs_hr, rho, g,
                            base, L_norm, trial_elcc,
                            P_r_turb[h], Q_max_turb[h], eta_turb[h],
                            P_r_pump[h], Q_max_pump[h], eta_pump[h],
                            v_up, h_max[h], v_up * 0.5, q_river, evp)
                        Pg = np.nan_to_num(Pg); Pp = np.nan_to_num(Pp)
                        Vu = np.nan_to_num(Vu)
                        if h in ps_vu:
                            max_storage_frac_change = max(max_storage_frac_change,
                                float(np.nanmax(np.abs(Vu - ps_vu[h]))) / max(v_up, 1.0))
                        else:
                            max_storage_frac_change = max(max_storage_frac_change, 1.0)
                        new_net = Pg - Pp
                        max_power_change = max(max_power_change,
                            float(np.nanmax(np.abs(Pg - ps_gen[h]))),
                            float(np.nanmax(np.abs(Pp - ps_pump[h]))))
                        comb = comb - old_net + new_net
                        ps_gen[h] = Pg; ps_pump[h] = Pp; ps_vu[h] = Vu
                        _Qg = Pg * 1e6 / (eta_turb[h] * rho * g * h_max[h])
                        _Qp = Pp * eta_pump[h] * 1e6 / (rho * g * h_max[h])
                        _new_qout = np.maximum(q_river + _Qg - _Qp, 0.0)
                        max_flow_change = max(max_flow_change,
                            float(np.nanmax(np.abs(_new_qout - Q_out_local[:, :, h]))))
                        Q_out_local[:, :, h] = _new_qout
                        continue
                    if h not in ds_of_ps:
                        continue  # not downstream of any PS → keep clean BAL dispatch (already in comb)
                    q_in = _qin_C(h, Q_out_local)
                    if HPP_category[h] == 'RoR':
                        Pr = np.nan_to_num(np.fmin(np.fmin(q_in, Q_max_turb[h]) * eta_turb[h] * rho * g * h_max[h] / 1e6, P_r_turb[h]))
                        max_flow_change = max(max_flow_change,
                            float(np.nanmax(np.abs(q_in - Q_out_local[:, :, h]))))
                        Q_out_local[:, :, h] = q_in
                        if h in Psto:
                            max_power_change = max(max_power_change,
                                float(np.nanmax(np.abs(Pr - Psto[h]))))
                            comb = comb - Psto[h] + Pr; Psto[h] = Pr
                        if capture: disp[h] = (np.zeros((_T, _Y)), np.zeros((_T, _Y)), Pr, q_in.copy(), np.zeros((_T+1, _Y)))
                        continue
                    c_or = min(0.30, max(0.05, 1.0 - d_min[h]))
                    if HPP_category[h] == 'A':
                        Qfr = q_in.copy(); Qrr = np.zeros_like(Qfr)
                    else:
                        Qfr = f_reg[h] * q_in.copy(); Qrr = q_in - Qfr
                    _e = ELCC_opt[h] if np.isfinite(ELCC_opt[h]) else 0.0
                    r = _bal_core(_Y, hrs_byyear, c_or, _e, Qfr, Qrr,
                        Q_CONV_stable_hourly[:, :, h], _stable_env_release(h, Qrr),
                        precipitation_flux_hourly[:, :, h], evaporation_flux_hourly[:, :, h],
                        L_norm, V_CONV_hourly[0, 0, h], V_max_cumul[h], Q_max_turb[h],
                        P_r_turb[h], eta_turb[h], rho, g, secs_hr,
                        f_cascade_downstream[h], bathy_c[h], bathy_d[h], bathy_a[h], bathy_b[h],
                        f_spill[h], f_stop_cumul[h], f_restart_cumul[h], mu[h],
                        HPP_category[h] == 'B', _T)
                    Ps, Pf, Pr = np.nan_to_num(r[7]), np.nan_to_num(r[8]), np.nan_to_num(r[9])
                    _Vnew = np.nan_to_num(r[0])
                    max_storage_frac_change = max(max_storage_frac_change,
                        float(np.nanmax(np.abs(_Vnew - Vsto[h]))) / max(V_max_cumul[h], 1.0))
                    Vsto[h] = _Vnew
                    newout = Ps + Pf + Pr
                    if h in Psto:
                        max_power_change = max(max_power_change,
                            float(np.nanmax(np.abs(newout - Psto[h]))))
                        comb = comb - Psto[h] + newout; Psto[h] = newout
                    max_flow_change = max(max_flow_change,
                        float(np.nanmax(np.abs(r[6] - Q_out_local[:, :, h]))))
                    Q_out_local[:, :, h] = r[6]
                    if capture: disp[h] = (Ps, Pf, Pr, r[6], r[0])
                conv_iters = _it + 1
                if (max_power_change <= CONV_TOL_MW and max_flow_change <= CONV_TOL_Q
                        and max_storage_frac_change <= CONV_TOL_V_FRAC):
                    break
            else:
                conv_status = 'NOT converged'
            comb = np.zeros((_T, _Y))
            for h in sto_topo: comb = comb + Psto[h]
            for ps in ps_list: comb = comb + ps_gen[ps] - ps_pump[ps]
            n_out = int(np.sum((comb < trial_elcc * _Lg - 1e-6) & _Lmask))
            outage = n_out / max(_Lntot, 1)
            return ps_gen, ps_pump, ps_vu, comb, outage, conv_iters, conv_status, disp, Q_out_local

        # Recompute the no-PS chain baseline with the SAME _eval model, so the floor below is consistent
        # with the captured dispatch (the legacy _chain_ps_elcc baseline can be optimistic relative to
        # this model and would otherwise report an ELCC whose dispatch does not actually meet it).
        _lo0, _hi0 = 0.0, max(chain_solo_elcc, solo_total) + 50.0
        _solo0 = 0.0
        for _ in range(80):
            if _hi0 - _lo0 < 0.1: break
            _m0 = (_lo0 + _hi0) / 2
            _r0 = _eval_chain_elcc(_m0, use_ps=False)
            if _r0[4] <= OUTAGE_MAX and 'NOT converged' not in _r0[6]:
                _solo0 = _m0; _lo0 = _m0
            else:
                _hi0 = _m0
        chain_solo_elcc = _solo0
        elcc_hi = chain_solo_elcc + sum(P_r_turb[ps] for ps in ps_list)

        # Binary search (chain ELCC) with the multi-PS closed-loop, rejecting non-converged solutions.
        lo_c, hi_c = 0.0, elcc_hi
        for _ in range(200):
            if hi_c - lo_c < 0.1: break
            mid_c = (lo_c + hi_c) / 2
            _r = _eval_chain_elcc(mid_c)
            out = _r[4]; conv_st = _r[6]
            if out <= OUTAGE_MAX and 'NOT converged' not in conv_st:
                best_chain_elcc = mid_c; lo_c = mid_c
            else:
                hi_c = mid_c

        # PS can always choose not to operate → chain ELCC with PS ≥ without PS
        best_chain_elcc = max(best_chain_elcc, chain_solo_elcc)
        chain_gain = best_chain_elcc - chain_solo_elcc
        # Final run at best ELCC, capturing the full per-station dispatch for write-back.
        ps_gen_f, ps_pump_f, ps_vu_f, P_comb_f, out_f, _conv_iters, _conv_status, disp_f, Qout_f = _eval_chain_elcc(best_chain_elcc, capture=True)
        if out_f > OUTAGE_MAX or 'NOT converged' in _conv_status:
            _fallback = _eval_chain_elcc(chain_solo_elcc, capture=True, use_ps=False)
            ps_gen_f, ps_pump_f, ps_vu_f, P_comb_f, out_f, _conv_iters, _conv_status, disp_f, Qout_f = _fallback
            if out_f > OUTAGE_MAX or 'NOT converged' in _conv_status:
                print(f'    WARN: validated no-PS fallback did not converge for chain; skipping chain result')
                return None
            best_chain_elcc = chain_solo_elcc
            chain_gain = 0.0
        _Pg_sum = np.zeros((_T, _Y)); _Pp_sum = np.zeros((_T, _Y))
        for ps in ps_list:
            _Pg_sum = _Pg_sum + np.nan_to_num(ps_gen_f.get(ps, 0.0))
            _Pp_sum = _Pp_sum + np.nan_to_num(ps_pump_f.get(ps, 0.0))

        result = {
            'chain': chain, 'solo': solo_total, 'combined': best_chain_elcc,
            'gain': chain_gain, 'ps': ps_main, 'P_gen': _Pg_sum, 'P_pump': _Pp_sum,
            'V_upper': ps_vu_f.get(ps_main, np.zeros((_T+1, _Y))),
            'chain_names': chain_names, 'chain_solo_elcc': chain_solo_elcc,
            'out_f': out_f, 'conv_iters': _conv_iters, 'conv_status': _conv_status,
            'ps_main_name': HPP_name[ps_main], 'ps_cap': sum(P_r_turb[ps] for ps in ps_list),
            'disp': disp_f, 'Q_out_chain': Qout_f,
            'Q_in_chain': {h: _qin_C(h, Qout_f) for h in chain},
            'ps_disp': {ps: (np.nan_to_num(ps_gen_f.get(ps, 0.0)), np.nan_to_num(ps_pump_f.get(ps, 0.0)),
                            ps_vu_f.get(ps, None)) for ps in ps_list},
        }
        return result

    # ── Run chains in parallel ──
    from concurrent.futures import ThreadPoolExecutor
    n_workers = max(1, min(len(chain_jobs), max(4, os.cpu_count() or 4)))
    chain_results = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_chain, cj): cj for cj in chain_jobs}
        for fut in futures:
            try:
                res = fut.result()
            except Exception as ex:
                print(f'    WARN: chain failed: {ex}')
                continue
            if res is None: continue
            # Print result
            print(f'    Chain: {res["chain_names"]}')
            print(f'    Chain ELCC (with PS) = {res["combined"]:.1f} MW  (no-PS={res["chain_solo_elcc"]:.1f}, '
                  f'+{res["gain"]:.1f} from PS, {res["gain"]/max(res["ps_cap"],1)*100:.0f}% of PS cap)  outage={res["out_f"]*100:.2f}%')
            print(f'    PS gen={np.nanmean(res["P_gen"]):.1f}MW pump={np.nanmean(res["P_pump"]):.1f}MW  '
                  f'convergence: {res["conv_status"]} in {res["conv_iters"]} iters')
            chain_results[tuple(sorted(res['chain']))] = res

    total_chain_gain = sum(r['gain'] for r in chain_results.values())
    print(f'\n  Total chain-level PS gain: {total_chain_gain:.0f} MW')

    # Write the FULL chain-coordinated dispatch back into the global hourly arrays so the saved
    # scenario_C_hourly.npz / ps_hourly_arrays.npz reproduce the chain ELCC reported in chain_elcc.csv
    # (#2): per-station STO power + outflow from the chain optimization, and EVERY PS's gen/pump/V_upper
    # + net river flow (not just ps_main).
    for _key, _res in chain_results.items():
        _disp = _res.get('disp', {})
        # chain STO NOT re-dispatched (not downstream of a PS) → restore their clean BAL power, so the
        # saved scenario reflects "STO at BAL + PS coordination" consistently (no per-pair PS.5b residue).
        for _h in _res['chain']:
            if type_unified[_h] != 'PS' and _h not in _disp:
                P_BAL_hydro_stable_hourly[:, :, _h] = P_BAL_stable_clean[:, :, _h]
                P_BAL_hydro_flexible_hourly[:, :, _h] = P_BAL_flex_clean[:, :, _h]
                P_BAL_hydro_RoR_hourly[:, :, _h] = P_BAL_ror_clean[:, :, _h]
        for _h, _d in _disp.items():
            _Ps, _Pf, _Pr, _Qo, _V = _d
            P_BAL_hydro_stable_hourly[:, :, _h] = _Ps
            P_BAL_hydro_flexible_hourly[:, :, _h] = _Pf
            P_BAL_hydro_RoR_hourly[:, :, _h] = _Pr
            Q_BAL_out_hourly[:, :, _h] = _Qo
            V_BAL_hourly[:, :, _h] = _V
        for _h, _Qi in _res.get('Q_in_chain', {}).items():
            Q_in_BAL_cascade[:, :, _h] = _Qi
        for _ps, _pd in _res.get('ps_disp', {}).items():
            _Pg, _Pp, _Vu = _pd
            if _ps in set(ps_paired):
                _pj = list(ps_paired).index(_ps)
                P_PS_turb_hourly[:, :, _pj] = np.nan_to_num(_Pg)
                P_PS_pump_hourly[:, :, _pj] = np.nan_to_num(_Pp)
                if _Vu is not None:
                    V_PS_upper_hourly[:, :, _pj] = np.nan_to_num(_Vu)
            # Persist the PS outflow ACTUALLY dispatched in the chain optimization (coordinated inflow),
            # not a recompute from the stale pre-chain baseline - correct for chained (PS-below-PS) cases.
            _Qoc = _res.get('Q_out_chain')
            if _Qoc is not None:
                Q_BAL_out_hourly[:, :, _ps] = _Qoc[:, :, _ps]

    # =================================================================
    # PS.6) Summary and output
    # =================================================================

    import pandas as pd

    _scenario_tag = os.environ.get('REVUB_SCENARIO_TAG', '').strip()
    _output_override = os.environ.get('REVUB_OUTPUT_DIR_OVERRIDE', '').strip()
    OUTPUT_DIR = (_output_override if _output_override else
                  os.path.join(REGION_PATH, 'revub_output', GCM, SSP))
    if _scenario_tag and not _output_override:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, _scenario_tag)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows_ps = []
    for p_idx in range(n_paired):
        ps = ps_paired[p_idx]
        anchor = ps_anchor[ps]
        for y in range(_Y):
            nhrs = int(hrs_byyear[y])
            E_ps_turb = np.nansum(P_PS_turb_hourly[:nhrs, y, p_idx])
            E_ps_pump = np.nansum(P_PS_pump_hourly[:nhrs, y, p_idx])
            rows_ps.append({
                'ps_station': HPP_name[ps],
                'anchor_station': HPP_name[anchor],
                'year': simulation_years[y],
                'P_turb_rated_MW': P_r_turb[ps],
                'P_pump_rated_MW': P_r_pump[ps],
                'ELCC_solo_MW': ELCC_solo[p_idx],
                'ELCC_combined_MW': ELCC_PS[p_idx],
                'ELCC_gain_MW': ELCC_PS[p_idx] - ELCC_solo[p_idx],
                'E_turb_MWh': E_ps_turb,
                'E_pump_MWh': E_ps_pump,
                'roundtrip_eff': E_ps_turb / max(E_ps_pump, 1e-9),
                'outage_frac': outage_PS[p_idx],
            })

    df_ps = pd.DataFrame(rows_ps)
    ps_path = os.path.join(OUTPUT_DIR, 'ps_yearly_summary.parquet')
    df_ps.to_parquet(ps_path, index=False)

    # ── Save hourly PS arrays ──
    ps_hourly_path = os.path.join(OUTPUT_DIR, 'ps_hourly_arrays.npz')
    np.savez_compressed(
        ps_hourly_path,
        ps_station_names=np.array(ps_paired_names, dtype=object),
        anchor_station_names=np.array(anchor_names, dtype=object),
        P_PS_turb_hourly=P_PS_turb_hourly,
        P_PS_pump_hourly=P_PS_pump_hourly,
        V_PS_upper_hourly=V_PS_upper_hourly,
        P_STO_stable_combined=P_STO_stable_combined,
        P_STO_flex_combined=P_STO_flex_combined,
        P_STO_ror_combined=P_STO_ror_combined,
        V_STO_combined=V_STO_combined,
        ELCC_PS=ELCC_PS,
        ELCC_solo=ELCC_solo,
    )

    t_ps_elapsed = time.time() - t_ps_start

    print(f'\n{"="*60}')
    print(f'PS Dispatch complete in {t_ps_elapsed:.1f}s')
    print(f'  Paired stations: {n_paired}')
    total_elcc_gain = np.nansum(ELCC_PS - ELCC_solo)
    total_ps_cap = sum(P_r_turb[ps] for ps in ps_paired)
    print(f'  Total PS capacity: {total_ps_cap:.0f} MW')
    print(f'  Total ELCC gain:   {total_elcc_gain:.0f} MW')
    # ── Save station ELCC (all stations, for D file) ──
    station_elcc_rows = []
    for i in range(HPP_number):
        elcc_indep_val = ELCC_indep[i] if 'ELCC_indep' in dir() and np.isfinite(ELCC_indep[i]) else ELCC_opt[i]
        station_elcc_rows.append({
            'station_name': HPP_name[i],
            'type': type_unified[i],
            'capacity_mw': P_r_turb[i],
            'elcc_indep_mw': elcc_indep_val if np.isfinite(elcc_indep_val) else 0.0,
            'elcc_cascopt_mw': ELCC_opt[i] if np.isfinite(ELCC_opt[i]) else 0.0,
            'lat': station_lats[i],
            'lon': station_lons[i] if 'station_lons' in dir() else np.nan,
        })
    pd.DataFrame(station_elcc_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'station_elcc.csv'), index=False)

    # ── Save chain ELCC (for D file) ──
    chain_elcc_rows = []
    for key, res in chain_results.items():
        chain_elcc_rows.append({
            'chain_stations': ' → '.join(HPP_name[h][:20] for h in res['chain']),
            'n_stations': len(res['chain']),
            'chain_elcc_nops_mw': res.get('chain_solo_elcc', res.get('solo', 0)),
            'chain_elcc_ps_mw': res['combined'],
            'ps_gain_mw': res['gain'],
            'ps_station': HPP_name[res['ps']] if res.get('ps') is not None else '',
        })
    pd.DataFrame(chain_elcc_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'chain_elcc.csv'), index=False)

    # ── Save PS pair ELCC (individual pairing, for non-cascade scenarios) ──
    ps_pair_rows = []
    for p_idx in range(n_paired):
        ps = ps_paired[p_idx]
        anchor = ps_anchor[ps]
        ps_pair_rows.append({
            'ps_station': HPP_name[ps],
            'anchor_station': HPP_name[anchor],
            'solo_elcc_mw': ELCC_solo[p_idx],
            'combined_elcc_mw': ELCC_PS[p_idx],
            'ps_gain_mw': ELCC_PS[p_idx] - ELCC_solo[p_idx],
        })
    pd.DataFrame(ps_pair_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'ps_pair_elcc.csv'), index=False)

    # ── Save Scenario C hourly: cascade-optimized STO + PS flow/volume ──
    # Q_BAL_out_hourly and V_BAL_hourly now contain the cascade-opt + PS-propagated values
    scenC_path = os.path.join(OUTPUT_DIR, 'scenario_C_hourly.npz')
    np.savez_compressed(scenC_path,
        Q_BAL_out_C=Q_BAL_out_hourly,
        V_BAL_C=V_BAL_hourly,
        P_BAL_stable_C=P_BAL_hydro_stable_hourly,
        P_BAL_flex_C=P_BAL_hydro_flexible_hourly,
        P_BAL_ror_C=P_BAL_hydro_RoR_hourly,
        Q_in_BAL_cascade_C=Q_in_BAL_cascade,
        ELCC_C=ELCC_opt,
    )

    print(f'  Results saved to:  {OUTPUT_DIR}')
    print(f'    ps_yearly_summary.parquet  ({len(df_ps)} rows)')
    print(f'    ps_hourly_arrays.npz')
    print(f'    scenario_C_hourly.npz      (cascade-opt + PS hourly)')
    print(f'    station_elcc.csv  ({len(station_elcc_rows)} stations)')
    print(f'    chain_elcc.csv  ({len(chain_elcc_rows)} chains)')
    print(f'    ps_pair_elcc.csv  ({len(ps_pair_rows)} pairs)')
    print(f'{"="*60}')
