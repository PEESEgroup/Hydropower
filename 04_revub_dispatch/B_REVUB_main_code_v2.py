# -*- coding: utf-8 -*-
"""
REVUB core simulation - rewritten for SWAT+/GRFR pipeline (CONV-only mode).

Requires A_REVUB_initialise_v2.py to have been run first (exec or %run).
Outputs per-station hourly power, flow, volume, head, and yearly summaries
as parquet files in REGION_PATH/revub_output/.

Based on REVUB by Sebastian Sterl, VUB-HYDR (2019-present).
"""

import numpy as np
import os
import time
import warnings


def _bathy_h_from_V(V_m3, c, d):
    """h = (V_km3 / c)^(1/d), where V_km3 = V_m3 / 1e9"""
    V_km3 = V_m3 / 1e9
    if not (V_km3 > 0) or not (c > 0) or not (d > 0):
        return 0.0
    return (V_km3 / c) ** (1.0 / d)


def _bathy_A_from_h(h, a, b):
    """A_m2 = a * h^b * 1e6"""
    if not (h > 0) or not (a > 0) or not np.isfinite(b):
        return 0.0
    return a * (h ** b) * 1e6


def _stable_env_release(HPP, q_ror):
    """Reservoir release needed after actual unregulated pass-through covers part of Q_env."""
    return np.maximum(Q_env_total_hourly[:, :, HPP] - q_ror, 0.0)

# ─── sanity check: A file must have run ───
assert 'HPP_number' in dir(), 'Run A_REVUB_initialise_v2.py first'

print(f'\n{"="*60}')
print(f'REVUB CONV Simulation - {HPP_number} stations')
print(f'{"="*60}\n')

# =============================================================================
# REVUB.1) Preallocate
# =============================================================================

_T = int(np.max(positions))   # max hours in a year (8784)
_Y = len(simulation_years)


def _delay_station_series(series_3d, station, tau):
    """Delay a station series while preserving the existing cyclic year boundary."""
    delayed = series_3d[:, :, station].copy()
    if tau <= 0:
        return delayed
    delayed = np.roll(delayed, shift=tau, axis=0)
    for y in range(_Y):
        y_prev = _Y - 1 if y == 0 else y - 1
        hrs_prev = int(hrs_byyear[y_prev])
        t0 = max(0, hrs_prev - tau)
        n_avail = hrs_prev - t0
        delayed[:n_avail, y] = series_3d[t0:hrs_prev, y_prev, station]
    return delayed


def _delay_2d_series(series_2d, tau):
    """Two-dimensional counterpart used by local cascade trial simulations."""
    delayed = series_2d.copy()
    if tau <= 0:
        return delayed
    delayed = np.roll(delayed, shift=tau, axis=0)
    for y in range(_Y):
        y_prev = _Y - 1 if y == 0 else y - 1
        hrs_prev = int(hrs_byyear[y_prev])
        t0 = max(0, hrs_prev - tau)
        n_avail = hrs_prev - t0
        delayed[:n_avail, y] = series_2d[t0:hrs_prev, y_prev]
    return delayed


def _cascade_inflow(HPP, Q_out_arr, upstream_subset=None, return_lateral=False):
    """Route upstream releases without discarding them when inferred lateral flow is negative."""
    lateral = Q_in_nat_hourly[:, :, HPP].copy()
    routed_release = np.zeros_like(lateral)
    for i, upstream in enumerate(direct_upstream_indices[HPP]):
        if upstream_subset is not None and upstream not in upstream_subset:
            continue
        tau = direct_upstream_tau[HPP][i] if i < len(direct_upstream_tau[HPP]) else 0
        lateral -= _delay_station_series(Q_in_nat_hourly, upstream, tau)
        routed_release += _delay_station_series(Q_out_arr, upstream, tau)
    inflow = np.maximum(lateral, 0.0) + routed_release
    return (inflow, lateral) if return_lateral else inflow

# ── inflow split ──
Q_in_frac_hourly = np.full([_T, _Y, HPP_number], np.nan)
Q_in_frac_store  = np.full([_T, _Y, HPP_number], np.nan)
Q_in_RoR_hourly  = np.full([_T, _Y, HPP_number], np.nan)
Q_in_RoR_store   = np.full([_T, _Y, HPP_number], np.nan)

# ── HPP classification ──
HPP_category = ['' for _ in range(HPP_number)]

# ── cascade ──
f_cascade_downstream = np.ones(HPP_number)

# ── outflow rule parameters ──
tau_fill = np.full(HPP_number, np.nan)
phi      = np.full(HPP_number, np.nan)
kappa    = np.full(HPP_number, np.nan)

# ── CONV outflow ──
Q_CONV_stable_hourly = np.full([_T, _Y, HPP_number], np.nan)
Q_CONV_spill_hourly  = np.full([_T, _Y, HPP_number], np.nan)
Q_CONV_out_hourly    = np.full([_T, _Y, HPP_number], np.nan)

# ── CONV volume / area / head ──
V_CONV_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)
A_CONV_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)
h_CONV_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)

# ── CONV power ──
P_CONV_hydro_stable_hourly = np.full([_T, _Y, HPP_number], np.nan)
P_CONV_hydro_RoR_hourly    = np.full([_T, _Y, HPP_number], np.nan)

# ── yearly energy (MWh/yr) ──
E_hydro_CONV_stable_yearly = np.zeros((_Y, HPP_number))
E_hydro_CONV_RoR_yearly    = np.zeros((_Y, HPP_number))
E_hydro_CONV_yearly        = np.zeros((_Y, HPP_number))

# ── monthly energy (GWh/month) ──
E_hydro_CONV_stable_bymonth = np.zeros((months_yr, _Y, HPP_number))
E_hydro_CONV_RoR_bymonth    = np.zeros((months_yr, _Y, HPP_number))
E_hydro_CONV_total_bymonth  = np.zeros((months_yr, _Y, HPP_number))

# ── monthly outflow / inflow / head ──
Q_CONV_out_monthly  = np.zeros((months_yr, _Y, HPP_number))
Q_in_nat_monthly    = np.zeros((months_yr, _Y, HPP_number))
h_CONV_bymonth      = np.zeros((months_yr, _Y, HPP_number))

# ── curtailment (drought protection) ──
hydro_CONV_curtailment_factor_hourly = np.full([_T, _Y, HPP_number], np.nan)

# ── capacity factor ──
CF_hydro_CONV_yearly = np.full([_Y, HPP_number], np.nan)

# ── guaranteed power ──
P_CONV_total_guaranteed = np.full(HPP_number, np.nan)

# ── outage / overflow fractions ──
fraction_outage_CONV  = np.zeros(HPP_number)
fraction_overflow_CONV = np.zeros(HPP_number)


# =============================================================================
# REVUB.2) Classify HPPs & compute outflow-curve parameters
# =============================================================================

for HPP in range(HPP_number):

    if np.isnan(f_reg[HPP]):
        f_reg[HPP] = (V_max_cumul[HPP] / (min(np.sum(days_year, 0)) * hrs_day * secs_hr * T_fill_thres)) / \
                      np.nanmean(Q_in_nat_hourly[:, int(year_calibration_start[HPP] - year_start):
                                                    int(year_calibration_end[HPP] - year_start + 1), HPP])

    if np.isnan(d_min[HPP]):
        if f_reg[HPP] > 0:
            d_min[HPP] = np.clip(
                (min_load_turbine[HPP] * Q_max_turb[HPP] / int(no_turbines[HPP])
                 - (1 - f_reg[HPP]) * np.nanmin(Q_in_nat_hourly[:, :, HPP]))
                / (np.nanmean(Q_in_nat_hourly[:,
                    int(year_calibration_start[HPP] - year_start):
                    int(year_calibration_end[HPP] - year_start + 1), HPP]) * f_reg[HPP]),
                0, 1)
        else:
            d_min[HPP] = 0

    # classify by f_reg
    # f_reg < 0.001 = negligible storage relative to flow → treat as ROR
    # This prevents S-curve gamma explosion for tiny-reservoir stations
    if f_reg[HPP] >= 1:
        HPP_category[HPP] = 'A'
    elif f_reg[HPP] >= 0.001:
        HPP_category[HPP] = 'B'
    else:
        HPP_category[HPP] = 'RoR'


# =============================================================================
# REVUB.3) CONV simulation
# =============================================================================

t_sim_start = time.time()

# Q_in_nat_hourly stores the ORIGINAL natural flow (never modified).
# Q_in_cascade_hourly stores the CASCADE-ADJUSTED inflow for each station.
Q_in_cascade_hourly = Q_in_nat_hourly.copy()

for HPP in range(HPP_number):

    print(f'HPP {HPP + 1}/{HPP_number}: {HPP_name[HPP]}', end=' ')

    # ── Apply cascade: replace upstream natural flow with upstream regulated outflow ──
    upstream_list = direct_upstream_indices[HPP]
    upstream_tau_list = direct_upstream_tau[HPP]
    if len(upstream_list) > 0:
        us_names = [HPP_name[u] for u in upstream_list]
        us_taus = [str(upstream_tau_list[i]) if i < len(upstream_tau_list) else '0' for i in range(len(upstream_list))]
        print(f'(upstream: {",".join(us_names)} tau={",".join(us_taus)}h) ', end='')
        Q_in_cascade_hourly[:, :, HPP], lateral_flow = _cascade_inflow(
            HPP, Q_CONV_out_hourly, return_lateral=True)
        # Diagnose negative lateral flow (downstream Q_nat < sum of upstream Q_nat)
        neg_mask = lateral_flow < 0
        n_neg = np.sum(neg_mask & np.isfinite(lateral_flow))
        n_fin = np.sum(np.isfinite(lateral_flow))
        if n_neg > 0 and n_fin > 0:
            frac_neg = n_neg / n_fin * 100
            mean_deficit = np.nanmean(lateral_flow[neg_mask])
            print(f'\n  WARNING: negative lateral flow {frac_neg:.1f}% of hours '
                  f'(mean deficit={mean_deficit:.1f} m³/s), lateral component clamped to 0')

    # ── Flow splitting from cascade-adjusted inflow ──
    if HPP_category[HPP] == 'B':
        Q_in_frac_hourly[:, :, HPP] = f_reg[HPP] * Q_in_cascade_hourly[:, :, HPP]
    elif HPP_category[HPP] == 'A':
        Q_in_frac_hourly[:, :, HPP] = Q_in_cascade_hourly[:, :, HPP]
    else:
        Q_in_frac_hourly[:, :, HPP] = 0.0
    Q_in_RoR_hourly[:, :, HPP] = Q_in_cascade_hourly[:, :, HPP] - Q_in_frac_hourly[:, :, HPP]
    Q_out_stable_env_irr_hourly[:, :, HPP] = _stable_env_release(HPP, Q_in_RoR_hourly[:, :, HPP])

    # ── RoR shortcut ──
    if HPP_category[HPP] == 'RoR':
        Q_CONV_out_hourly[:, :, HPP] = Q_in_cascade_hourly[:, :, HPP]
        P_CONV_hydro_RoR_hourly[:, :, HPP] = np.fmin(
            np.fmin(Q_in_cascade_hourly[:, :, HPP], Q_max_turb[HPP])
            * eta_turb[HPP] * rho * g * h_max[HPP] / 1e6,
            P_r_turb[HPP])
        P_CONV_hydro_stable_hourly[:, :, HPP] = 0.0
        hydro_CONV_curtailment_factor_hourly[:, :, HPP] = 1.0
        print('[RoR]')
        continue

    # ── PS shortcut: pass-through, no power in CONV ──
    if type_unified[HPP] == 'PS':
        Q_CONV_out_hourly[:, :, HPP] = Q_in_cascade_hourly[:, :, HPP]
        P_CONV_hydro_stable_hourly[:, :, HPP] = 0.0
        P_CONV_hydro_RoR_hourly[:, :, HPP] = 0.0
        V_CONV_hourly[:, :, HPP] = 0.0
        h_CONV_hourly[:, :, HPP] = 0.0
        A_CONV_hourly[:, :, HPP] = 0.0
        hydro_CONV_curtailment_factor_hourly[:, :, HPP] = 1.0
        print('[PS - pass-through]')
        continue

    # ── reservoir simulation: outflow curve params from cascade-adjusted flow ──
    Q_frac_cal = Q_in_frac_hourly[:,
                    int(year_calibration_start[HPP] - year_start):
                    int(year_calibration_end[HPP] - year_start + 1), HPP]
    Q_in_nat_av = np.nanmean(Q_frac_cal)

    if Q_in_nat_av > 0:
        tau_fill[HPP] = (Q_in_nat_av *
                         (min(np.sum(days_year, 0)) * hrs_day * secs_hr) / V_max_cumul[HPP]) ** (-1)
        phi[HPP] = alpha[HPP] * np.sqrt(tau_fill[HPP])
        kappa[HPP] = (np.exp(1 - d_min[HPP]) - 1) / (f_opt[HPP] ** phi[HPP])

        if np.isnan(gamma_hydro[HPP]):
            target_Q = (d_min[HPP] + np.log(kappa[HPP] * f_opt[HPP] ** phi[HPP] + 1)) * Q_in_nat_av
            if target_Q > 0:
                gamma_hydro[HPP] = np.log(Q_max_turb[HPP] / target_Q) / \
                                   ((f_spill[HPP] - f_opt[HPP]) ** 2)
            else:
                gamma_hydro[HPP] = 0.0
    else:
        tau_fill[HPP] = np.inf
        phi[HPP] = 0.0
        kappa[HPP] = 0.0
        if np.isnan(gamma_hydro[HPP]):
            gamma_hydro[HPP] = 0.0

    hydro_CONV_curtailment_factor_hourly[:, :, HPP] = 1

    cat_label = 'A(large)' if HPP_category[HPP] == 'A' else 'B(small)'
    print(f'[{cat_label}, f_reg={min(f_reg[HPP], 1):.2f}]')

    for y in range(_Y):
        hrs_year = range(int(hrs_byyear[y]))

        # ── initial conditions ──
        if y == 0:
            V_CONV_hourly[0, y, HPP] = V_max_cumul[HPP] * f_initial_frac[HPP]
            V_lookup = f_cascade_downstream[HPP] * V_CONV_hourly[0, y, HPP]
            h_CONV_hourly[0, y, HPP] = _bathy_h_from_V(V_lookup, bathy_c[HPP], bathy_d[HPP])
            A_CONV_hourly[0, y, HPP] = _bathy_A_from_h(h_CONV_hourly[0, y, HPP], bathy_a[HPP], bathy_b[HPP])
        else:
            temp = V_CONV_hourly[:, y - 1, HPP]
            V_CONV_hourly[0, y, HPP] = temp[np.isfinite(temp)][-1]
            temp = A_CONV_hourly[:, y - 1, HPP]
            A_CONV_hourly[0, y, HPP] = temp[np.isfinite(temp)][-1]
            temp = h_CONV_hourly[:, y - 1, HPP]
            h_CONV_hourly[0, y, HPP] = temp[np.isfinite(temp)][-1]

        # ── hourly loop ──
        for n in hrs_year:
            v_frac = V_CONV_hourly[n, y, HPP] / V_max_cumul[HPP]

            # outflow rule (S-curve)
            if v_frac < f_opt[HPP]:
                Q_CONV_stable_hourly[n, y, HPP] = max(
                    (d_min[HPP] + np.log(kappa[HPP] * v_frac ** phi[HPP] + 1)) * Q_in_nat_av,
                    Q_out_stable_env_irr_hourly[n, y, HPP])
                Q_CONV_spill_hourly[n, y, HPP] = 0.0

            elif v_frac < f_spill[HPP]:
                Q_CONV_stable_hourly[n, y, HPP] = max(
                    np.exp(gamma_hydro[HPP] * (v_frac - f_opt[HPP]) ** 2) * Q_in_nat_av,
                    Q_out_stable_env_irr_hourly[n, y, HPP])
                Q_CONV_spill_hourly[n, y, HPP] = 0.0

            else:
                Q_CONV_stable_hourly[n, y, HPP] = max(
                    np.exp(gamma_hydro[HPP] * (v_frac - f_opt[HPP]) ** 2) * Q_in_nat_av,
                    Q_out_stable_env_irr_hourly[n, y, HPP])
                Q_CONV_spill_hourly[n, y, HPP] = max(0.0,
                    (Q_in_frac_hourly[n, y, HPP]
                     + (precipitation_flux_hourly[n, y, HPP] - evaporation_flux_hourly[n, y, HPP])
                       * A_CONV_hourly[n, y, HPP] / rho)
                    * (1 + mu[HPP]) - Q_CONV_stable_hourly[n, y, HPP])

            # Drought protection only curtails operation above the ecological target.
            _qenv = Q_out_stable_env_irr_hourly[n, y, HPP]
            Q_CONV_stable_hourly[n, y, HPP] = _qenv + hydro_CONV_curtailment_factor_hourly[n, y, HPP] * max(
                Q_CONV_stable_hourly[n, y, HPP] - _qenv, 0.0)

            # total outflow
            Q_CONV_out_hourly[n, y, HPP] = (Q_CONV_stable_hourly[n, y, HPP]
                                            + Q_CONV_spill_hourly[n, y, HPP]
                                            + Q_in_RoR_hourly[n, y, HPP])

            # power generation
            Q_pot_turb = min(Q_CONV_stable_hourly[n, y, HPP], Q_max_turb[HPP])
            P_CONV_hydro_stable_hourly[n, y, HPP] = (
                Q_pot_turb * eta_turb[HPP] * rho * g * h_CONV_hourly[n, y, HPP] / 1e6)

            Q_RoR_avail = min(Q_in_RoR_hourly[n, y, HPP],
                              max(0, Q_max_turb[HPP] - Q_CONV_stable_hourly[n, y, HPP]))
            P_CONV_hydro_RoR_hourly[n, y, HPP] = max(0.0, min(
                Q_RoR_avail * eta_turb[HPP] * rho * g * h_CONV_hourly[n, y, HPP] / 1e6,
                P_r_turb[HPP] - P_CONV_hydro_stable_hourly[n, y, HPP]))

            # water balance
            V_CONV_hourly[n + 1, y, HPP] = V_CONV_hourly[n, y, HPP] + (
                Q_in_frac_hourly[n, y, HPP]
                - Q_CONV_stable_hourly[n, y, HPP]
                - Q_CONV_spill_hourly[n, y, HPP]
                + (precipitation_flux_hourly[n, y, HPP] - evaporation_flux_hourly[n, y, HPP])
                  * A_CONV_hourly[n, y, HPP] / rho
            ) * secs_hr

            # Clamp volume while preserving the water balance. If the provisional
            # release exceeds available storage, curtail spill first and then the
            # stable release; power must be recomputed from the water actually released.
            if V_CONV_hourly[n + 1, y, HPP] < 0:
                Q_short = -V_CONV_hourly[n + 1, y, HPP] / secs_hr
                Q_spill_cut = min(Q_CONV_spill_hourly[n, y, HPP], Q_short)
                Q_CONV_spill_hourly[n, y, HPP] -= Q_spill_cut
                Q_short -= Q_spill_cut

                Q_stable_cut = min(Q_CONV_stable_hourly[n, y, HPP], Q_short)
                Q_CONV_stable_hourly[n, y, HPP] -= Q_stable_cut

                Q_pot_turb = min(Q_CONV_stable_hourly[n, y, HPP], Q_max_turb[HPP])
                P_CONV_hydro_stable_hourly[n, y, HPP] = (
                    Q_pot_turb * eta_turb[HPP] * rho * g * h_CONV_hourly[n, y, HPP] / 1e6)
                Q_RoR_avail = min(
                    Q_in_RoR_hourly[n, y, HPP],
                    max(0, Q_max_turb[HPP] - Q_CONV_stable_hourly[n, y, HPP]))
                P_CONV_hydro_RoR_hourly[n, y, HPP] = max(0.0, min(
                    Q_RoR_avail * eta_turb[HPP] * rho * g * h_CONV_hourly[n, y, HPP] / 1e6,
                    P_r_turb[HPP] - P_CONV_hydro_stable_hourly[n, y, HPP]))
                Q_CONV_out_hourly[n, y, HPP] = (
                    Q_CONV_stable_hourly[n, y, HPP]
                    + Q_CONV_spill_hourly[n, y, HPP]
                    + Q_in_RoR_hourly[n, y, HPP])
                V_CONV_hourly[n + 1, y, HPP] = 0.0
            if V_CONV_hourly[n + 1, y, HPP] > V_max_cumul[HPP]:
                V_excess = V_CONV_hourly[n + 1, y, HPP] - V_max_cumul[HPP]
                Q_CONV_spill_hourly[n, y, HPP] += V_excess / secs_hr
                Q_CONV_out_hourly[n, y, HPP] += V_excess / secs_hr
                V_CONV_hourly[n + 1, y, HPP] = V_max_cumul[HPP]

            # update head & area from bathymetry (analytical power-law)
            V_lookup = f_cascade_downstream[HPP] * V_CONV_hourly[n + 1, y, HPP]
            h_CONV_hourly[n + 1, y, HPP] = _bathy_h_from_V(V_lookup, bathy_c[HPP], bathy_d[HPP])
            A_CONV_hourly[n + 1, y, HPP] = _bathy_A_from_h(h_CONV_hourly[n + 1, y, HPP], bathy_a[HPP], bathy_b[HPP])

            # ── drought protection ──
            # Q_ror->Q_frac transfer removed (proportional curtailment in BAL handles this)
            if V_CONV_hourly[n + 1, y, HPP] < f_stop_cumul[HPP] * V_max_cumul[HPP]:
                if n < len(hrs_year) - 1:
                    hydro_CONV_curtailment_factor_hourly[n + 1, y, HPP] = 0
                elif y < _Y - 1:
                    hydro_CONV_curtailment_factor_hourly[0, y + 1, HPP] = 0

            if hydro_CONV_curtailment_factor_hourly[n, y, HPP] == 0:
                if V_CONV_hourly[n + 1, y, HPP] > f_restart_cumul[HPP] * V_max_cumul[HPP]:
                    next_val = 1
                else:
                    next_val = 0
                if n < len(hrs_year) - 1:
                    hydro_CONV_curtailment_factor_hourly[n + 1, y, HPP] = next_val
                elif y < _Y - 1:
                    hydro_CONV_curtailment_factor_hourly[0, y + 1, HPP] = next_val

        # yearly energy totals
        E_hydro_CONV_stable_yearly[y, HPP] = np.nansum(P_CONV_hydro_stable_hourly[hrs_year, y, HPP])
        E_hydro_CONV_RoR_yearly[y, HPP] = np.nansum(P_CONV_hydro_RoR_hourly[hrs_year, y, HPP])
        E_hydro_CONV_yearly[y, HPP] = E_hydro_CONV_stable_yearly[y, HPP] + E_hydro_CONV_RoR_yearly[y, HPP]

    # ── outage/overflow stats ──
    n_valid = np.sum(~np.isnan(Q_CONV_out_hourly[:, :, HPP]))
    if n_valid > 0:
        n_zeros = np.size(hydro_CONV_curtailment_factor_hourly[:, :, HPP]) - \
                  np.count_nonzero(hydro_CONV_curtailment_factor_hourly[:, :, HPP])
        fraction_outage_CONV[HPP] = n_zeros / n_valid
    q_mean_in = np.nanmean(Q_in_cascade_hourly[:, :, HPP])
    if q_mean_in > 0:
        fraction_overflow_CONV[HPP] = np.nanmean(Q_CONV_spill_hourly[:, :, HPP]) / q_mean_in

    if fraction_outage_CONV[HPP] > 0:
        print(f'  WARNING: drought curtailment {100 * fraction_outage_CONV[HPP]:.2f}%')
    if fraction_overflow_CONV[HPP] > 0.01:
        print(f'  Note: avg spill = {100 * fraction_overflow_CONV[HPP]:.2f}% of inflow')

elapsed = time.time() - t_sim_start
print(f'\nCONV simulation done in {elapsed:.1f}s')


# =============================================================================
# REVUB.3b) BAL simulation - load-following dispatch
# =============================================================================

_bal_mode = os.environ.get('REVUB_LOAD_FRAC', '0')
bal_elcc_auto = (_bal_mode == 'auto')
bal_load_frac = 0.0 if bal_elcc_auto else float(_bal_mode)
bal_enabled = (calibration_only == 0) and (bal_elcc_auto or bal_load_frac > 0)

if bal_enabled:
    print(f'\n{"="*60}')
    print(f'BAL Simulation - {"ELCC auto-search" if bal_elcc_auto else f"load_frac={bal_load_frac}"}')
    print(f'{"="*60}\n')

    # ── Load profile: flat (firm power) / synthetic / external file ──
    load_profile_path = os.path.join(DATASETS_DIR, 'load_profile_hourly.parquet')
    load_type = os.environ.get('REVUB_LOAD_TYPE', 'auto')  # flat | synthetic | auto (auto = external if exists, else flat)

    if load_type == 'flat':
        L_norm = np.ones([_T, _Y])
        for y in range(_Y):
            L_norm[int(hrs_byyear[y]):, y] = np.nan
        print(f'  L_norm: flat (firm power mode)')
    elif load_type == 'synthetic':
        L_norm = np.full([_T, _Y], np.nan)
        for y in range(_Y):
            for n in range(int(hrs_byyear[y])):
                hod = n % 24
                if hod < 6: L_norm[n, y] = 0.55
                elif hod < 9: L_norm[n, y] = 0.55 + (hod - 6) * 0.15
                elif hod < 14: L_norm[n, y] = 1.0
                elif hod < 17: L_norm[n, y] = 0.85
                elif hod < 21: L_norm[n, y] = 1.0
                else: L_norm[n, y] = 1.0 - (hod - 21) * 0.15
        print(f'  L_norm: synthetic (Brazilian double-peak, mean={np.nanmean(L_norm):.3f})')
    elif os.path.exists(load_profile_path):
        import pandas as pd
        _lp = pd.read_parquet(load_profile_path)
        L_norm = np.full([_T, _Y], np.nan)
        for y in range(_Y):
            yr_data = _lp[_lp['year'] == simulation_years[y]]['L_norm'].values
            n = min(len(yr_data), int(hrs_byyear[y]))
            L_norm[:n, y] = yr_data[:n]
        load_type = 'external'
        print(f'  L_norm: external file ({load_profile_path})')
    else:
        L_norm = np.ones([_T, _Y])
        for y in range(_Y):
            L_norm[int(hrs_byyear[y]):, y] = np.nan
        load_type = 'flat'
        print(f'  L_norm: flat (no external file, fallback)')

    L_norm_mean = np.nanmean(L_norm)
    if not np.isfinite(L_norm_mean) or L_norm_mean <= 0:
        print(f'  WARNING: L_norm_mean={L_norm_mean}, falling back to 1.0')
        L_norm_mean = 1.0

    # ── BAL preallocate ──
    Q_BAL_stable_hourly = np.full([_T, _Y, HPP_number], np.nan)
    Q_BAL_flexible_hourly = np.full([_T, _Y, HPP_number], np.nan)
    Q_BAL_spill_hourly = np.full([_T, _Y, HPP_number], np.nan)
    Q_BAL_out_hourly = np.full([_T, _Y, HPP_number], np.nan)
    V_BAL_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)
    A_BAL_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)
    h_BAL_hourly = np.full([_T + 1, _Y, HPP_number], np.nan)
    P_BAL_hydro_stable_hourly = np.full([_T, _Y, HPP_number], np.nan)
    P_BAL_hydro_flexible_hourly = np.full([_T, _Y, HPP_number], np.nan)
    P_BAL_hydro_RoR_hourly = np.full([_T, _Y, HPP_number], np.nan)
    P_BAL_ramp_restr_hourly = np.full([_T, _Y, HPP_number], np.nan)
    hydro_BAL_curtailment_factor_hourly = np.full([_T, _Y, HPP_number], np.nan)
    E_hydro_BAL_stable_yearly = np.zeros((_Y, HPP_number))
    E_hydro_BAL_flexible_yearly = np.zeros((_Y, HPP_number))
    E_hydro_BAL_RoR_yearly = np.zeros((_Y, HPP_number))
    E_hydro_BAL_yearly = np.zeros((_Y, HPP_number))
    CF_hydro_BAL_yearly = np.full([_Y, HPP_number], np.nan)
    psi_BAL = np.full(HPP_number, np.nan)
    C_OR_opt = np.full(HPP_number, np.nan)
    ELCC_opt = np.full(HPP_number, np.nan)
    fraction_outage_BAL = np.zeros(HPP_number)

    # One-tenth rule (1 day in 10 years) = 0.1 day/yr x 24 h / 8760 h = 0.000274.
    # This is the criterion used for all published runs; the env var only overrides it.
    OUTAGE_MAX = float(os.environ.get('REVUB_OUTAGE_MAX', '0.000274'))
    N_ELCC = 40
    Q_in_BAL_cascade = Q_in_nat_hourly.copy()
    t_bal_start = time.time()

    from numba import njit

    @njit
    def _bal_core(nY, hrs_by, C_OR, elcc, Q_frac, Q_ror, Q_conv_stable, Q_env,
                  precip, evap, L_norm_arr, V_conv_init, V_max, Q_turb_max, P_rated,
                  eta, _rho, _g, _secs, f_cascade_ds, bc, bd, ba, bb,
                  f_spill_v, f_stop_v, f_restart_v, _mu, is_B, _T):
        V = np.full((_T+1, nY), np.nan); A = np.full((_T+1, nY), np.nan); h = np.full((_T+1, nY), np.nan)
        Qs = np.zeros((_T, nY)); Qf = np.zeros((_T, nY)); Qsp = np.zeros((_T, nY)); Qo = np.zeros((_T, nY))
        Ps = np.zeros((_T, nY)); Pf = np.zeros((_T, nY)); Pr = np.zeros((_T, nY))
        curt = np.ones((_T, nY)); n_outage = 0; n_total = 0
        for y in range(nY):
            nhrs = int(hrs_by[y])
            if y == 0:
                V[0,y] = V_conv_init
            else:
                for k in range(_T, -1, -1):
                    if not np.isnan(V[k, y-1]):
                        V[0,y] = V[k, y-1]; break
            vl = f_cascade_ds * V[0,y]
            V_km3 = vl / 1e9
            h[0,y] = (V_km3 / bc) ** (1.0/bd) if V_km3 > 0 and bc > 0 and bd > 0 else 0.0
            A[0,y] = ba * (h[0,y] ** bb) * 1e6 if h[0,y] > 0 and ba > 0 else 0.0
            for n in range(nhrs):
                qs = max((1.0 - C_OR) * Q_conv_stable[n,y] * curt[n,y], Q_env[n,y])
                Qs[n,y] = qs
                ps = min(qs, Q_turb_max) * eta * _rho * _g * h[n,y] / 1e6
                Ps[n,y] = ps
                qra = min(Q_ror[n,y], max(0.0, Q_turb_max - qs))
                pr = max(0.0, min(qra * eta * _rho * _g * h[n,y] / 1e6, P_rated - ps))
                Pr[n,y] = pr
                Lh = elcc * L_norm_arr[n,y] if not np.isnan(L_norm_arr[n,y]) else 0.0
                Pd = ps + pr - Lh
                if Pd < 0.0:
                    qpf = max(0.0, Q_turb_max - qs - qra) * curt[n,y]
                    ppf = qpf * eta * _rho * _g * h[n,y] / 1e6
                    Pf[n,y] = min(abs(Pd), ppf)
                qf = Pf[n,y] / (eta * _rho * _g * h[n,y]) * 1e6 if h[n,y] > 0 else 0.0
                Qf[n,y] = qf
                vfrac = V[n,y] / V_max if V_max > 0 else 0.0
                sp = 0.0
                if vfrac >= f_spill_v:
                    sp = max(0.0, (Q_frac[n,y] + (precip[n,y]-evap[n,y])*A[n,y]/_rho) * (1.0+_mu) - qs - qf)
                Qsp[n,y] = sp
                Qo[n,y] = qs + qf + sp + Q_ror[n,y]
                vnew = V[n,y] + (Q_frac[n,y] - qs - qf - sp + (precip[n,y]-evap[n,y])*A[n,y]/_rho) * _secs
                if vnew < 0:
                    over = (-vnew) / _secs
                    sr = min(sp, over); over -= sr; Qsp[n,y] -= sr; sp -= sr
                    fr = min(qf, over); over -= fr; Qf[n,y] -= fr; qf -= fr
                    Pf[n,y] = qf * eta * _rho * _g * h[n,y] / 1e6 if h[n,y] > 0 else 0.0
                    # If storage still cannot cover the stable release, curtail it all the way to
                    # the physically available amount. Q_env is a target, not a source of water.
                    if over > 0.0:
                        qsr = min(qs, over)
                        if qsr > 0.0:
                            qs -= qsr; over -= qsr; Qs[n,y] = qs
                            Ps[n,y] = min(qs, Q_turb_max) * eta * _rho * _g * h[n,y] / 1e6
                    Qo[n,y] = qs + qf + sp + Q_ror[n,y]
                    vnew = 0.0
                if vnew > V_max:
                    vex = vnew - V_max; Qsp[n,y] += vex/_secs; Qo[n,y] += vex/_secs; vnew = V_max
                V[n+1,y] = vnew
                vl2 = f_cascade_ds * vnew; Vk2 = vl2/1e9
                h[n+1,y] = (Vk2/bc)**(1.0/bd) if Vk2 > 0 and bc > 0 and bd > 0 else 0.0
                A[n+1,y] = ba * (h[n+1,y]**bb) * 1e6 if h[n+1,y] > 0 and ba > 0 else 0.0
                # Proportional curtailment: linear ramp between f_stop and f_restart
                # (replaces original binary 0/1 switch; see Helseth et al. 2022,
                #  rule-curve theory in Loucks & van Beek)
                vfrac_new = vnew / V_max if V_max > 0 else 1.0
                if f_restart_v > f_stop_v:
                    nv = max(0.0, min(1.0, (vfrac_new - f_stop_v) / (f_restart_v - f_stop_v)))
                else:
                    nv = 0.0 if vfrac_new < f_stop_v else 1.0
                if n < nhrs-1: curt[n+1,y] = nv
                elif y < nY-1: curt[0,y+1] = nv
                n_total += 1
                if not (Ps[n,y] + Pr[n,y] + Pf[n,y] >= Lh - 1e-6): n_outage += 1
        outage = n_outage / max(n_total, 1)
        return V, A, h, Qs, Qf, Qsp, Qo, Ps, Pf, Pr, curt, outage

    print('  Compiling Numba kernel...', end=' ', flush=True)
    _dummy = _bal_core(1, np.array([24.0]), 0.3, 100.0,
        np.ones((8784,1)), np.ones((8784,1)), np.ones((8784,1)), np.zeros((8784,1)),
        np.zeros((8784,1)), np.zeros((8784,1)), np.ones((8784,1)),
        1e9, 2e9, 1000.0, 100.0, 0.88, 1000.0, 9.81, 3600.0,
        1.0, 0.1, 2.0, 0.1, 1.5, 0.85, 0.20, 0.25, 0.10, True, 8784)
    print('done')

    def _run_bal_station(HPP, C_OR, elcc, Q_cascade, write_global):
        """Run BAL for one station. Returns (ψ, outage)."""
        if HPP_category[HPP] == 'A':
            Qfr = Q_cascade[:,:,HPP].copy(); Qrr = np.zeros_like(Qfr)
        else:
            Qfr = f_reg[HPP] * Q_cascade[:,:,HPP].copy()
            Qrr = Q_cascade[:,:,HPP] - Qfr
        V, A, h, Qs, Qf, Qsp, Qo, Ps, Pf, Pr, curt, outage = _bal_core(
            _Y, hrs_byyear, C_OR, elcc, Qfr, Qrr,
            Q_CONV_stable_hourly[:,:,HPP], _stable_env_release(HPP, Qrr),
            precipitation_flux_hourly[:,:,HPP], evaporation_flux_hourly[:,:,HPP],
            L_norm, V_CONV_hourly[0,0,HPP], V_max_cumul[HPP], Q_max_turb[HPP],
            P_r_turb[HPP], eta_turb[HPP], rho, g, secs_hr,
            f_cascade_downstream[HPP], bathy_c[HPP], bathy_d[HPP], bathy_a[HPP], bathy_b[HPP],
            f_spill[HPP], f_stop_cumul[HPP], f_restart_cumul[HPP], mu[HPP],
            HPP_category[HPP] == 'B', _T)
        vc = V_CONV_hourly[:-1,:,HPP]; vb = V[:-1,:]
        m = np.isfinite(vc) & np.isfinite(vb)
        _mvc = np.mean(vc[m]) if m.any() else np.nan
        trial_psi = np.mean(np.abs(vb[m]-vc[m])) / _mvc if (m.any() and np.isfinite(_mvc) and _mvc > 0) else np.inf
        if write_global:
            Q_BAL_stable_hourly[:,:,HPP] = Qs; Q_BAL_flexible_hourly[:,:,HPP] = Qf
            Q_BAL_spill_hourly[:,:,HPP] = Qsp; Q_BAL_out_hourly[:,:,HPP] = Qo
            V_BAL_hourly[:,:,HPP] = V[:,:]; A_BAL_hourly[:,:,HPP] = A[:,:]; h_BAL_hourly[:,:,HPP] = h[:,:]
            P_BAL_hydro_stable_hourly[:,:,HPP] = Ps; P_BAL_hydro_flexible_hourly[:,:,HPP] = Pf; P_BAL_hydro_RoR_hourly[:,:,HPP] = Pr
            hydro_BAL_curtailment_factor_hourly[:,:,HPP] = curt
            for y in range(_Y):
                hrs_yr = range(int(hrs_byyear[y]))
                E_hydro_BAL_stable_yearly[y,HPP] = np.nansum(Ps[hrs_yr,y])
                E_hydro_BAL_flexible_yearly[y,HPP] = np.nansum(Pf[hrs_yr,y])
                E_hydro_BAL_RoR_yearly[y,HPP] = np.nansum(Pr[hrs_yr,y])
                E_hydro_BAL_yearly[y,HPP] = E_hydro_BAL_stable_yearly[y,HPP]+E_hydro_BAL_flexible_yearly[y,HPP]+E_hydro_BAL_RoR_yearly[y,HPP]
        return trial_psi, outage

    for HPP in range(HPP_number):
        print(f'BAL {HPP+1}/{HPP_number}: {HPP_name[HPP]}', end=' ')
        upstream_list = direct_upstream_indices[HPP]
        if len(upstream_list) > 0:
            print(f'(upstream: {",".join(HPP_name[u] for u in upstream_list)}) ', end='')
            Q_in_BAL_cascade[:, :, HPP] = _cascade_inflow(HPP, Q_BAL_out_hourly)
        if HPP_category[HPP] == 'RoR':
            Q_BAL_out_hourly[:,:,HPP] = Q_in_BAL_cascade[:,:,HPP]
            P_BAL_hydro_RoR_hourly[:,:,HPP] = np.fmin(np.fmin(Q_in_BAL_cascade[:,:,HPP], Q_max_turb[HPP])*eta_turb[HPP]*rho*g*h_max[HPP]/1e6, P_r_turb[HPP])
            P_BAL_hydro_stable_hourly[:,:,HPP] = 0.0; P_BAL_hydro_flexible_hourly[:,:,HPP] = 0.0
            Q_BAL_stable_hourly[:,:,HPP] = 0.0; Q_BAL_flexible_hourly[:,:,HPP] = 0.0; Q_BAL_spill_hourly[:,:,HPP] = 0.0
            V_BAL_hourly[:,:,HPP] = 0.0; hydro_BAL_curtailment_factor_hourly[:,:,HPP] = 1.0
            for y in range(_Y):
                E_hydro_BAL_RoR_yearly[y,HPP] = np.nansum(P_BAL_hydro_RoR_hourly[:int(hrs_byyear[y]),y,HPP])
                E_hydro_BAL_yearly[y,HPP] = E_hydro_BAL_RoR_yearly[y,HPP]
            ratio = P_BAL_hydro_RoR_hourly[:,:,HPP] / L_norm
            rv = ratio[np.isfinite(ratio) & (L_norm > 0)]
            ror_elcc = np.percentile(rv, OUTAGE_MAX * 100) if len(rv) > 0 else 0.0
            ELCC_opt[HPP] = max(ror_elcc, 0.0); C_OR_opt[HPP] = 0.0; psi_BAL[HPP] = 0.0
            print(f'[RoR - pass-through, ELCC={ELCC_opt[HPP]:.0f}MW]'); continue

        # ── PS: pass-through in BAL, no power (handled by C file) ──
        if type_unified[HPP] == 'PS':
            Q_BAL_out_hourly[:,:,HPP] = Q_in_BAL_cascade[:,:,HPP]
            P_BAL_hydro_stable_hourly[:,:,HPP] = 0.0; P_BAL_hydro_flexible_hourly[:,:,HPP] = 0.0
            P_BAL_hydro_RoR_hourly[:,:,HPP] = 0.0
            Q_BAL_stable_hourly[:,:,HPP] = 0.0; Q_BAL_flexible_hourly[:,:,HPP] = 0.0; Q_BAL_spill_hourly[:,:,HPP] = 0.0
            V_BAL_hourly[:,:,HPP] = 0.0; hydro_BAL_curtailment_factor_hourly[:,:,HPP] = 1.0
            ELCC_opt[HPP] = 0.0; C_OR_opt[HPP] = 0.0; psi_BAL[HPP] = 0.0
            print(f'[PS - pass-through, dispatch in C file]'); continue

        cat = 'A(large)' if HPP_category[HPP]=='A' else 'B(small)'
        C_OR_default = min(0.30, max(0.05, 1.0 - d_min[HPP]))

        if bal_elcc_auto:
            P_avg = np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP] + P_CONV_hydro_RoR_hourly[:,:,HPP])
            elcc_hi = P_avg / L_norm_mean
            # Binary search: outage is monotonically non-decreasing in ELCC
            lo, hi = 0.0, elcc_hi
            best_elcc = 0.0; best_psi = 0.0; best_outage = 0.0
            for _ in range(200):  # ~0.001 MW precision
                if hi - lo < 0.1: break
                mid = (lo + hi) / 2
                psi_t, out_t = _run_bal_station(HPP, C_OR_default, mid, Q_in_BAL_cascade, write_global=False)
                if out_t <= OUTAGE_MAX:
                    best_elcc = mid; best_psi = psi_t; best_outage = out_t
                    lo = mid
                else:
                    hi = mid
        else:
            best_elcc = bal_load_frac * P_r_turb[HPP]
            best_psi, best_outage = _run_bal_station(HPP, C_OR_default, best_elcc, Q_in_BAL_cascade, write_global=False)

        ELCC_opt[HPP] = best_elcc; C_OR_opt[HPP] = C_OR_default; psi_BAL[HPP] = best_psi
        _run_bal_station(HPP, C_OR_default, best_elcc, Q_in_BAL_cascade, write_global=True)
        fraction_outage_BAL[HPP] = best_outage
        cf = np.mean(E_hydro_BAL_yearly[:,HPP])/(P_r_turb[HPP]*np.mean(hrs_byyear)) if P_r_turb[HPP]>0 else 0
        print(f'[{cat}, C_OR={C_OR_default:.2f}, ψ={best_psi:.4f}, ELCC={best_elcc:.0f}MW, {best_elcc/P_r_turb[HPP]*100:.0f}%cap]  CF={cf:.3f}  outage={100*best_outage:.1f}%')

    print(f'\nBAL simulation done in {time.time()-t_bal_start:.1f}s')

    # ── Snapshot Scenario A: independent station dispatch (before cascade opt) ──
    ELCC_indep = ELCC_opt.copy()
    Q_BAL_out_indep = Q_BAL_out_hourly.copy()
    V_BAL_indep = V_BAL_hourly.copy()
    P_BAL_stable_indep = P_BAL_hydro_stable_hourly.copy()
    P_BAL_flex_indep = P_BAL_hydro_flexible_hourly.copy()
    P_BAL_ror_indep = P_BAL_hydro_RoR_hourly.copy()
    Q_in_BAL_cascade_indep = Q_in_BAL_cascade.copy()

    # =========================================================================
    # REVUB.3c) CASCADE ELCC OPTIMIZATION (B_1 feature)
    # =========================================================================
    # Maximize total ELCC across cascade chains by scanning head-station ELCC.
    # Triggered by env var REVUB_CASCADE_OPT=1.

    if os.environ.get('REVUB_CASCADE_OPT', '0') == '1' and bal_elcc_auto:
        from concurrent.futures import ThreadPoolExecutor

        t_opt_start = time.time()
        N_SCAN = int(os.environ.get('REVUB_CASCADE_NSCAN', '40'))

        # ── Identify cascade chains as CONNECTED COMPONENTS of the river graph (union-find) ──
        # A confluence (A→C←B) is ONE chain, not two overlapping head-traced chains that would
        # optimize and write back the shared downstream stations TWICE with order-dependent results.
        _bparent = list(range(HPP_number))
        def _bfind(x):
            while _bparent[x] != x:
                _bparent[x] = _bparent[_bparent[x]]; x = _bparent[x]
            return x
        def _bunion(a, b):
            ra, rb = _bfind(a), _bfind(b)
            if ra != rb: _bparent[ra] = rb
        for _i in range(HPP_number):
            _d = downstream_index[_i]
            if 0 <= _d < HPP_number: _bunion(_i, _d)
        def _bndown(h):
            n = 0; cur = downstream_index[h]; seen = 0
            while 0 <= cur < HPP_number and seen <= HPP_number:
                n += 1; cur = downstream_index[cur]; seen += 1
            return n
        _bcomp = {}
        for _i in range(HPP_number):
            _bcomp.setdefault(_bfind(_i), []).append(_i)
        # Each station belongs to exactly ONE component ⇒ optimized/written back exactly once.
        chains = []
        for _members in _bcomp.values():
            sto_in_chain = [i for i in _members if HPP_category[i] != 'RoR' and type_unified[i] != 'PS']
            if len(sto_in_chain) >= 2:
                ch = sorted(_members, key=_bndown, reverse=True)            # topo order (upstream first)
                sto_in_chain = sorted(sto_in_chain, key=_bndown, reverse=True)
                chains.append((ch, sto_in_chain))

        if chains:
            print(f'\n{"="*60}')
            print(f'CASCADE ELCC OPTIMIZATION - {len(chains)} chains, N_SCAN={N_SCAN}')
            print(f'{"="*60}')

        for chain_all, chain_sto in chains:
            head = chain_sto[0]
            chain_names_str = ' -> '.join(HPP_name[i] for i in chain_sto)
            print(f'\nChain: {chain_names_str}')
            print(f'  Greedy total: {sum(ELCC_opt[i] for i in chain_sto):.0f} MW')

            P_avg_head = np.nanmean(P_CONV_hydro_stable_hourly[:,:,head] + P_CONV_hydro_RoR_hourly[:,:,head])
            elcc_candidates = np.linspace(P_avg_head / L_norm_mean, 0, N_SCAN + 1)

            chain_set = set(chain_all)

            def eval_chain(elcc_head):
                """Evaluate total chain ELCC for a given head-station ELCC."""
                Q_cascade_local = Q_in_nat_hourly.copy()
                total = 0.0
                elcc_per_station = {}

                for hpp in chain_all:
                    lateral = Q_in_nat_hourly[:, :, hpp].copy()
                    routed_release = np.zeros_like(lateral)
                    us_list = direct_upstream_indices[hpp]
                    us_tau = direct_upstream_tau[hpp]
                    for idx, u in enumerate(us_list):
                        if u not in chain_set:
                            continue
                        tau = us_tau[idx] if idx < len(us_tau) else 0
                        Qou_key = elcc_per_station.get(u, {}).get('Qout')
                        if Qou_key is None:
                            continue
                        lateral -= _delay_station_series(Q_in_nat_hourly, u, tau)
                        routed_release += _delay_2d_series(Qou_key, tau)
                    Q_cascade_local[:, :, hpp] = np.maximum(lateral, 0.0) + routed_release

                    if HPP_category[hpp] == 'RoR':
                        Qout = Q_cascade_local[:,:,hpp].copy()
                        P_ror = np.fmin(np.fmin(Qout, Q_max_turb[hpp]) * eta_turb[hpp] * rho * g * h_max[hpp] / 1e6, P_r_turb[hpp])
                        ratio = P_ror / L_norm
                        rv = ratio[np.isfinite(ratio) & (L_norm > 0)]
                        ror_elcc = max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0
                        elcc_per_station[hpp] = {'elcc': ror_elcc, 'Qout': Qout}
                        total += ror_elcc
                        continue

                    if type_unified[hpp] == 'PS':
                        Qout = Q_cascade_local[:,:,hpp].copy()
                        elcc_per_station[hpp] = {'elcc': 0, 'Qout': Qout}
                        continue

                    c_or_hpp = min(0.30, max(0.05, 1.0 - d_min[hpp]))
                    if hpp == head:
                        trial_elcc = elcc_head
                    else:
                        P_avg_s = np.nanmean(P_CONV_hydro_stable_hourly[:,:,hpp] + P_CONV_hydro_RoR_hourly[:,:,hpp])
                        elcc_hi_s = P_avg_s / L_norm_mean
                        trial_elcc = 0.0
                        for et in np.linspace(elcc_hi_s, 0, N_ELCC + 1):
                            psi_t, out_t = _run_bal_station(hpp, c_or_hpp, et, Q_cascade_local, write_global=False)
                            if out_t <= OUTAGE_MAX:
                                trial_elcc = et; break

                    if HPP_category[hpp] == 'A':
                        Qfr = Q_cascade_local[:,:,hpp].copy(); Qrr = np.zeros_like(Qfr)
                    else:
                        Qfr = f_reg[hpp] * Q_cascade_local[:,:,hpp].copy()
                        Qrr = Q_cascade_local[:,:,hpp] - Qfr
                    V,A,h,Qs,Qf,Qsp,Qo,Ps,Pf,Pr,curt,outage = _bal_core(
                        _Y, hrs_byyear, c_or_hpp, trial_elcc, Qfr, Qrr,
                        Q_CONV_stable_hourly[:,:,hpp], _stable_env_release(hpp, Qrr),
                        precipitation_flux_hourly[:,:,hpp], evaporation_flux_hourly[:,:,hpp],
                        L_norm, V_CONV_hourly[0,0,hpp], V_max_cumul[hpp], Q_max_turb[hpp],
                        P_r_turb[hpp], eta_turb[hpp], rho, g, secs_hr,
                        f_cascade_downstream[hpp], bathy_c[hpp], bathy_d[hpp], bathy_a[hpp], bathy_b[hpp],
                        f_spill[hpp], f_stop_cumul[hpp], f_restart_cumul[hpp], mu[hpp],
                        HPP_category[hpp]=='B', _T)
                    if outage > OUTAGE_MAX:
                        return None, None
                    elcc_per_station[hpp] = {'elcc': trial_elcc, 'Qout': Qo}
                    total += trial_elcc

                return total, {k: v['elcc'] for k, v in elcc_per_station.items()}

            # ── Parallel scan over head-station ELCC candidates ──
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
                futures = {pool.submit(eval_chain, e): e for e in elcc_candidates}
                results = []
                for fut in futures:
                    try:
                        total, elccs = fut.result()
                        if total is not None:
                            results.append((total, elccs, futures[fut]))
                    except Exception as ex:
                        print(f'  WARN: eval_chain failed for ELCC_head={futures[fut]:.0f}: {ex}')

            if results:
                results.sort(key=lambda x: -x[0])
                best_total, best_elccs, best_head_elcc = results[0]
                greedy_total = sum(ELCC_opt[i] for i in chain_sto)

                print(f'  Optimal head ELCC: {best_head_elcc:.0f} MW')
                print(f'  Optimal total: {best_total:.0f} MW  (greedy was {greedy_total:.0f} MW, gain={best_total-greedy_total:+.0f} MW)')
                for hpp in chain_sto:
                    old_e = ELCC_opt[hpp]
                    new_e = best_elccs.get(hpp, old_e)
                    tag = ' *' if abs(new_e - old_e) > 1 else ''
                    print(f'    {HPP_name[hpp][:45]:45s}  {old_e:6.0f} -> {new_e:6.0f} MW{tag}')

                if best_total > greedy_total:
                    print(f'  Re-running chain with optimal ELCCs...')
                    Q_in_BAL_cascade_opt = Q_in_nat_hourly.copy()
                    for hpp in chain_all:
                        Q_in_BAL_cascade_opt[:, :, hpp] = _cascade_inflow(hpp, Q_BAL_out_hourly)

                        if HPP_category[hpp] == 'RoR':
                            Q_BAL_out_hourly[:,:,hpp] = Q_in_BAL_cascade_opt[:,:,hpp]
                            P_BAL_hydro_RoR_hourly[:,:,hpp] = np.fmin(np.fmin(Q_in_BAL_cascade_opt[:,:,hpp],Q_max_turb[hpp])*eta_turb[hpp]*rho*g*h_max[hpp]/1e6, P_r_turb[hpp])
                            ratio = P_BAL_hydro_RoR_hourly[:,:,hpp] / L_norm
                            rv = ratio[np.isfinite(ratio) & (L_norm > 0)]
                            ELCC_opt[hpp] = max(np.percentile(rv, OUTAGE_MAX * 100), 0.0) if len(rv) > 0 else 0.0
                            continue

                        if type_unified[hpp] == 'PS':
                            Q_BAL_out_hourly[:,:,hpp] = Q_in_BAL_cascade_opt[:,:,hpp]
                            P_BAL_hydro_stable_hourly[:,:,hpp] = 0.0
                            P_BAL_hydro_flexible_hourly[:,:,hpp] = 0.0
                            P_BAL_hydro_RoR_hourly[:,:,hpp] = 0.0
                            ELCC_opt[hpp] = 0.0
                            continue

                        new_elcc = best_elccs.get(hpp, ELCC_opt[hpp])
                        ELCC_opt[hpp] = new_elcc
                        c_or_hpp = min(0.30, max(0.05, 1.0 - d_min[hpp]))
                        psi_BAL[hpp], fraction_outage_BAL[hpp] = _run_bal_station(hpp, c_or_hpp, new_elcc, Q_in_BAL_cascade_opt, write_global=True)

                    print(f'  Chain re-run complete')

        print(f'\nCascade optimization done in {time.time()-t_opt_start:.1f}s')

else:
    bal_enabled = False


# =============================================================================
# REVUB.4) Post-processing
# =============================================================================

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)

    for HPP in range(HPP_number):

        # yearly capacity factor
        CF_hydro_CONV_yearly[:, HPP] = (
            (E_hydro_CONV_stable_yearly[:, HPP] + E_hydro_CONV_RoR_yearly[:, HPP])
            / (P_r_turb[HPP] * hrs_byyear))

        for y in range(_Y):
            hrs_year = range(int(hrs_byyear[y]))

            # RoR plants: compute yearly energy here (CONV loop was skipped)
            if HPP_category[HPP] == 'RoR':
                E_hydro_CONV_stable_yearly[y, HPP] = np.nansum(P_CONV_hydro_stable_hourly[hrs_year, y, HPP])
                E_hydro_CONV_RoR_yearly[y, HPP] = np.nansum(P_CONV_hydro_RoR_hourly[hrs_year, y, HPP])
                E_hydro_CONV_yearly[y, HPP] = E_hydro_CONV_stable_yearly[y, HPP] + E_hydro_CONV_RoR_yearly[y, HPP]
                CF_hydro_CONV_yearly[:, HPP] = (
                    (E_hydro_CONV_stable_yearly[:, HPP] + E_hydro_CONV_RoR_yearly[:, HPP])
                    / (P_r_turb[HPP] * hrs_byyear))

            for m in range(months_yr):
                s = int(positions[m, y])
                e = int(positions[m + 1, y])

                Q_in_nat_monthly[m, y, HPP] = np.nanmean(Q_in_cascade_hourly[s:e, y, HPP])
                Q_CONV_out_monthly[m, y, HPP] = np.nanmean(Q_CONV_out_hourly[s:e, y, HPP])

                E_hydro_CONV_stable_bymonth[m, y, HPP] = 1e-3 * np.nansum(P_CONV_hydro_stable_hourly[s:e, y, HPP])
                E_hydro_CONV_RoR_bymonth[m, y, HPP] = 1e-3 * np.nansum(P_CONV_hydro_RoR_hourly[s:e, y, HPP])
                E_hydro_CONV_total_bymonth[m, y, HPP] = (E_hydro_CONV_stable_bymonth[m, y, HPP]
                                                         + E_hydro_CONV_RoR_bymonth[m, y, HPP])

                h_CONV_bymonth[m, y, HPP] = np.nanmean(h_CONV_hourly[s:e, y, HPP])

        # guaranteed power (p_exceedance percentile)
        P_CONV_total_guaranteed[HPP] = np.nanpercentile(
            P_CONV_hydro_stable_hourly[:, :, HPP] + P_CONV_hydro_RoR_hourly[:, :, HPP],
            100 - p_exceedance)


# =============================================================================
# REVUB.5) Save results
# =============================================================================

import pandas as pd

_scenario_tag = os.environ.get('REVUB_SCENARIO_TAG', '').strip()
_output_override = os.environ.get('REVUB_OUTPUT_DIR_OVERRIDE', '').strip()
OUTPUT_DIR = (_output_override if _output_override else
              os.path.join(REGION_PATH, 'revub_output', GCM, SSP))
if _scenario_tag and not _output_override:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, _scenario_tag)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── per-station yearly summary ──
rows_yearly = []
for HPP in range(HPP_number):
    for y in range(_Y):
        rows_yearly.append({
            'station': HPP_name[HPP],
            'type': type_unified[HPP],
            'year': simulation_years[y],
            'capacity_mw': P_r_turb[HPP],
            'head_m': h_max[HPP],
            'f_reg': min(f_reg[HPP], 1.0),
            'category': HPP_category[HPP],
            'E_stable_MWh': E_hydro_CONV_stable_yearly[y, HPP],
            'E_RoR_MWh': E_hydro_CONV_RoR_yearly[y, HPP],
            'E_total_MWh': E_hydro_CONV_yearly[y, HPP],
            'E_total_GWh': E_hydro_CONV_yearly[y, HPP] / 1e3,
            'CF': CF_hydro_CONV_yearly[y, HPP],
            'Q_in_mean': np.nanmean(Q_in_cascade_hourly[:int(hrs_byyear[y]), y, HPP]),
            'Q_out_mean': np.nanmean(Q_CONV_out_hourly[:int(hrs_byyear[y]), y, HPP]),
            'spill_frac': fraction_overflow_CONV[HPP],
            'outage_frac': fraction_outage_CONV[HPP],
            'P_guaranteed_MW': P_CONV_total_guaranteed[HPP],
        })

df_yearly = pd.DataFrame(rows_yearly)
yearly_path = os.path.join(OUTPUT_DIR, 'conv_yearly_summary.parquet')
df_yearly.to_parquet(yearly_path, index=False)

# ── per-station monthly summary ──
rows_monthly = []
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for HPP in range(HPP_number):
    for y in range(_Y):
        for m in range(months_yr):
            rows_monthly.append({
                'station': HPP_name[HPP],
                'year': simulation_years[y],
                'month': m + 1,
                'month_name': month_names[m],
                'E_stable_GWh': E_hydro_CONV_stable_bymonth[m, y, HPP],
                'E_RoR_GWh': E_hydro_CONV_RoR_bymonth[m, y, HPP],
                'E_total_GWh': E_hydro_CONV_total_bymonth[m, y, HPP],
                'Q_in_mean': Q_in_nat_monthly[m, y, HPP],
                'Q_out_mean': Q_CONV_out_monthly[m, y, HPP],
                'h_mean': h_CONV_bymonth[m, y, HPP],
            })

df_monthly = pd.DataFrame(rows_monthly)
monthly_path = os.path.join(OUTPUT_DIR, 'conv_monthly_summary.parquet')
df_monthly.to_parquet(monthly_path, index=False)

# ── region-level aggregate ──
agg = df_yearly.groupby('year').agg(
    n_stations=('station', 'count'),
    total_capacity_MW=('capacity_mw', 'sum'),
    total_E_GWh=('E_total_GWh', 'sum'),
    mean_CF=('CF', 'mean'),
    total_P_guaranteed_MW=('P_guaranteed_MW', 'sum'),
).reset_index()

agg_path = os.path.join(OUTPUT_DIR, 'conv_region_yearly.parquet')
agg.to_parquet(agg_path, index=False)

print(f'\n{"="*60}')
print(f'Results saved to: {OUTPUT_DIR}')
print(f'  conv_yearly_summary.parquet  ({len(df_yearly)} rows)')
print(f'  conv_monthly_summary.parquet ({len(df_monthly)} rows)')
print(f'  conv_region_yearly.parquet   ({len(agg)} rows)')
print(f'{"="*60}')

# ── print region summary ──
print(f'\nRegion Summary ({GCM}/{SSP}):')
print(f'  Stations: {HPP_number}  (RoR={sum(c=="RoR" for c in HPP_category)}, '
      f'B={sum(c=="B" for c in HPP_category)}, A={sum(c=="A" for c in HPP_category)})')
print(f'  Total capacity: {np.sum(P_r_turb):.0f} MW')
print(f'  Mean annual generation: {agg["total_E_GWh"].mean():.1f} GWh')
print(f'  Mean capacity factor: {agg["mean_CF"].mean():.3f}')
print(f'  Total guaranteed power (P{p_exceedance}): {agg["total_P_guaranteed_MW"].mean():.1f} MW')
# ── BAL results (if enabled) ──
if bal_enabled:
    rows_bal = []
    for HPP in range(HPP_number):
        for y in range(_Y):
            CF_bal = (E_hydro_BAL_yearly[y, HPP] / (P_r_turb[HPP] * hrs_byyear[y])) if P_r_turb[HPP] > 0 else 0
            rows_bal.append({
                'station': HPP_name[HPP],
                'type': type_unified[HPP],
                'year': simulation_years[y],
                'capacity_mw': P_r_turb[HPP],
                'f_reg': min(f_reg[HPP], 1.0),
                'category': HPP_category[HPP],
                'C_OR': C_OR_opt[HPP],
                'ELCC_MW': ELCC_opt[HPP],
                'psi': psi_BAL[HPP],
                'E_stable_MWh': E_hydro_BAL_stable_yearly[y, HPP],
                'E_flexible_MWh': E_hydro_BAL_flexible_yearly[y, HPP],
                'E_RoR_MWh': E_hydro_BAL_RoR_yearly[y, HPP],
                'E_total_MWh': E_hydro_BAL_yearly[y, HPP],
                'E_total_GWh': E_hydro_BAL_yearly[y, HPP] / 1e3,
                'CF': CF_bal,
                'Q_in_mean': np.nanmean(Q_in_BAL_cascade[:int(hrs_byyear[y]), y, HPP]),
                'Q_out_mean': np.nanmean(Q_BAL_out_hourly[:int(hrs_byyear[y]), y, HPP]),
                'outage_frac': fraction_outage_BAL[HPP],
            })
    df_bal = pd.DataFrame(rows_bal)
    bal_path = os.path.join(OUTPUT_DIR, 'bal_yearly_summary.parquet')
    df_bal.to_parquet(bal_path, index=False)
    print(f'  bal_yearly_summary.parquet   ({len(df_bal)} rows)')

    # ── Save hourly arrays for the demo cascade chain ──
    hourly_path = os.path.join(OUTPUT_DIR, 'bal_hourly_arrays.npz')
    np.savez_compressed(hourly_path,
        HPP_name=HPP_name,
        Q_CONV_out=Q_CONV_out_hourly,
        Q_BAL_out=Q_BAL_out_hourly,
        Q_BAL_stable=Q_BAL_stable_hourly,
        Q_BAL_flexible=Q_BAL_flexible_hourly,
        V_CONV=V_CONV_hourly,
        V_BAL=V_BAL_hourly,
        P_CONV_stable=P_CONV_hydro_stable_hourly,
        P_CONV_RoR=P_CONV_hydro_RoR_hourly,
        P_BAL_stable=P_BAL_hydro_stable_hourly,
        P_BAL_flexible=P_BAL_hydro_flexible_hourly,
        P_BAL_RoR=P_BAL_hydro_RoR_hourly,
        Q_in_nat=Q_in_nat_hourly,
        Q_in_BAL_cascade=Q_in_BAL_cascade,
        L_norm=L_norm,
        simulation_years=simulation_years,
        hrs_byyear=hrs_byyear,
        ELCC_opt=ELCC_opt,
        psi_BAL=psi_BAL,
    )
    print(f'  bal_hourly_arrays.npz        (for plotting)')

    # ── Save Scenario A (independent dispatch, pre-cascade-opt) ──
    if True:
        indep_path = os.path.join(OUTPUT_DIR, 'bal_indep_arrays.npz')
        np.savez_compressed(indep_path,
            ELCC_indep=ELCC_indep,
            Q_BAL_out_indep=Q_BAL_out_indep,
            V_BAL_indep=V_BAL_indep,
            P_BAL_stable_indep=P_BAL_stable_indep,
            P_BAL_flex_indep=P_BAL_flex_indep,
            P_BAL_ror_indep=P_BAL_ror_indep,
            Q_in_BAL_cascade_indep=Q_in_BAL_cascade_indep,
        )
        print(f'  bal_indep_arrays.npz         (Scenario A: pre-cascade-opt)')

print(f'\nsimulation finished')
