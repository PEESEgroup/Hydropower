#!/usr/bin/env python3
"""
D_REVUB_ELCC_gurobi.py - Chain-level ELCC optimization via Gurobi MILP

Replaces the rule-based ELCC search (B file linear scan + C file chain PS)
with a deterministic perfect-foresight MILP that finds the globally optimal
dispatch for each cascade chain.

Requires: gurobipy, A + B files already executed (provides topology, flow data,
station parameters). Run AFTER B file.

Phase 1: Fixed-head linearization (h̄ = time-averaged head from CONV).
         Bilinear P = η·ρ·g·h·Q → linear P = η·ρ·g·h̄·Q.

Environment variables:
    GRB_LICENSE_FILE : path to Gurobi license file
    REVUB_ELCC_GUROBI_TIMELIMIT : solver time limit in seconds (default 300)
    REVUB_ELCC_GUROBI_MIPGAP : relative MIP gap tolerance (default 0.005)
"""

import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('GRB_LICENSE_FILE', '/home/cfeng/myswat/gurobi.lic')
import gurobipy as gp
from gurobipy import GRB

TIMELIMIT = float(os.environ.get('REVUB_ELCC_GUROBI_TIMELIMIT', '300'))
MIPGAP = float(os.environ.get('REVUB_ELCC_GUROBI_MIPGAP', '0.005'))

print(f'\n{"="*60}')
print(f'ELCC MILP Optimization (Gurobi) - Phase 1 Fixed Head')
print(f'{"="*60}')
print(f'  OUTAGE_MAX = {OUTAGE_MAX}')
print(f'  Time limit = {TIMELIMIT}s, MIP gap = {MIPGAP}')

# ═══════════════════════════════════════════════════════════
# D.1) Identify cascade chains with PS
# ═══════════════════════════════════════════════════════════

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

ps_indices_all = [i for i in range(HPP_number) if type_unified[i] == 'PS']
seen_chain = set()
chains_to_optimize = []

for ps in ps_indices_all:
    if ps in seen_chain: continue
    chain = _get_chain(ps)
    key = frozenset(chain)
    if key in seen_chain: continue
    seen_chain.update(chain)
    sto = [h for h in chain if type_unified[h] != 'PS' and HPP_category[h] in ('A', 'B')]
    ror = [h for h in chain if HPP_category[h] == 'RoR' and type_unified[h] != 'PS']
    ps_in = [h for h in chain if type_unified[h] == 'PS']
    if len(sto) == 0: continue
    chains_to_optimize.append({'chain': chain, 'sto': sto, 'ror': ror, 'ps': ps_in})

print(f'  Found {len(chains_to_optimize)} chains to optimize')

# ═══════════════════════════════════════════════════════════
# D.2) Build and solve MILP for each chain
# ═══════════════════════════════════════════════════════════

ELCC_gurobi = {}

for ci, cinfo in enumerate(chains_to_optimize):
    chain = cinfo['chain']
    sto_list = cinfo['sto']
    ror_list = cinfo['ror']
    ps_list = cinfo['ps']

    chain_names = ' → '.join(f"{HPP_name[h][:12]}{'★' if type_unified[h]=='PS' else ''}" for h in chain)
    print(f'\n  Chain {ci+1}/{len(chains_to_optimize)}: {chain_names}')

    # Use the largest PS
    ps_main = max(ps_list, key=lambda h: P_r_turb[h]) if ps_list else None

    # ── Time indexing: flatten all years into one sequence ──
    time_ranges = []  # (year_idx, start_flat, end_flat, nhrs)
    flat_t = 0
    for y in range(_Y):
        nhrs = int(hrs_byyear[y])
        time_ranges.append((y, flat_t, flat_t + nhrs, nhrs))
        flat_t += nhrs
    N_T = flat_t  # total hours

    # ── Fixed head (Phase 1): use time-averaged head from CONV ──
    h_bar = {}
    A_bar = {}
    for s in sto_list:
        h_vals = []
        a_vals = []
        for y in range(_Y):
            nhrs = int(hrs_byyear[y])
            hv = h_CONV_hourly[:nhrs, y, s] if 'h_CONV_hourly' in dir() else np.full(nhrs, h_max[s])
            av = A_CONV_hourly[:nhrs, y, s] if 'A_CONV_hourly' in dir() else np.zeros(nhrs)
            h_vals.append(hv[np.isfinite(hv)])
            a_vals.append(av[np.isfinite(av)])
        h_all = np.concatenate(h_vals)
        a_all = np.concatenate(a_vals)
        h_bar[s] = np.mean(h_all) if len(h_all) > 0 and np.mean(h_all) > 0 else h_max[s]
        A_bar[s] = np.mean(a_all) if len(a_all) > 0 else 0.0

    # ── Precompute flat arrays for parameters ──
    Q_nat_flat = {}
    Q_env_flat = {}
    precip_flat = {}
    evap_flat = {}
    L_norm_flat = np.zeros(N_T)

    for s in chain:
        q = np.zeros(N_T); e = np.zeros(N_T); p = np.zeros(N_T); qe = np.zeros(N_T)
        for y_idx, y_start, y_end, nhrs in time_ranges:
            q[y_start:y_end] = Q_in_nat_hourly[:nhrs, y_idx, s]
            qe[y_start:y_end] = Q_out_stable_env_irr_hourly[:nhrs, y_idx, s]
            p[y_start:y_end] = precipitation_flux_hourly[:nhrs, y_idx, s]
            e[y_start:y_end] = evaporation_flux_hourly[:nhrs, y_idx, s]
        Q_nat_flat[s] = q
        Q_env_flat[s] = qe
        precip_flat[s] = p
        evap_flat[s] = e

    for y_idx, y_start, y_end, nhrs in time_ranges:
        L_norm_flat[y_start:y_end] = L_norm[:nhrs, y_idx]

    # ── Upstream mapping for cascade ──
    # For each station s, precompute upstream contributions with tau delay
    # Q_in(s,t) = Q_nat(s,t) + Σ_u [Q_out(u, t-τ) - Q_nat(u, t-τ)]
    # We need to map (u, tau) → flat time index offsets

    # ── Build Gurobi model ──
    t_build = time.time()
    m = gp.Model(f'ELCC_chain_{ci}')
    m.Params.TimeLimit = TIMELIMIT
    m.Params.MIPGap = MIPGAP
    m.Params.OutputFlag = 0  # suppress solver output

    # ── Decision variable: ELCC ──
    sum_cap = sum(P_r_turb[s] for s in sto_list + ror_list)
    ps_cap = P_r_turb[ps_main] if ps_main else 0
    elcc = m.addVar(lb=0, ub=sum_cap + ps_cap, name='ELCC')

    # ── STO station variables ──
    Q_reg = {}      # regulated turbine flow
    Q_ror_turb = {} # RoR portion through turbine
    Q_spill = {}    # spill flow
    V = {}          # reservoir volume
    P_sto = {}      # power output

    for s in sto_list:
        fr = f_reg[s] if f_reg[s] < 1.0 else 1.0
        h_s = h_bar[s]
        eta_s = eta_turb[s]
        power_coeff = eta_s * rho * g * h_s / 1e6  # MW per m³/s

        for t in range(N_T):
            Q_reg[s, t] = m.addVar(lb=0, ub=Q_max_turb[s], name=f'Qr_{s}_{t}')
            Q_ror_turb[s, t] = m.addVar(lb=0, ub=Q_max_turb[s], name=f'Qrt_{s}_{t}')
            Q_spill[s, t] = m.addVar(lb=0, name=f'Qsp_{s}_{t}')
            P_sto[s, t] = m.addVar(lb=0, ub=P_r_turb[s], name=f'P_{s}_{t}')

        for t in range(N_T + 1):
            V[s, t] = m.addVar(lb=0, ub=V_max_cumul[s], name=f'V_{s}_{t}')

    # ── ROR station: power is determined by inflow (no decision variables for dispatch) ──
    P_ror_var = {}
    for s in ror_list:
        for t in range(N_T):
            P_ror_var[s, t] = m.addVar(lb=0, ub=P_r_turb[s], name=f'Pr_{s}_{t}')

    # ── PS variables ──
    P_gen = {}; P_pump = {}; V_ps = {}; z_gen = {}; z_pump = {}
    Q_gen_ps = {}; Q_pump_ps = {}

    if ps_main is not None:
        ps = ps_main
        v_ps_max = V_max[ps] if V_max[ps] > 0 else 1e6
        h_ps = h_max[ps]
        eta_gen_ps = eta_turb[ps]
        eta_pump_ps = eta_pump[ps]
        pc_gen = eta_gen_ps * rho * g * h_ps / 1e6
        pc_pump = rho * g * h_ps / (eta_pump_ps * 1e6)

        lat_ps = station_lats[ps]
        evap_mm = 1200 if abs(lat_ps) <= 15 else (800 if abs(lat_ps) <= 30 else 400)
        A_ps_est = v_ps_max / 20.0
        evap_ps_rate = (evap_mm / 1000.0) * A_ps_est / (365.25 * 24 * 3600)  # m³/s

        for t in range(N_T):
            P_gen[t] = m.addVar(lb=0, ub=P_r_turb[ps], name=f'Pg_{t}')
            P_pump[t] = m.addVar(lb=0, ub=P_r_pump[ps], name=f'Pp_{t}')
            Q_gen_ps[t] = m.addVar(lb=0, ub=Q_max_turb[ps], name=f'Qg_{t}')
            Q_pump_ps[t] = m.addVar(lb=0, ub=Q_max_pump[ps], name=f'Qp_{t}')
            z_gen[t] = m.addVar(vtype=GRB.BINARY, name=f'zg_{t}')
            z_pump[t] = m.addVar(vtype=GRB.BINARY, name=f'zp_{t}')

        for t in range(N_T + 1):
            V_ps[t] = m.addVar(lb=0, ub=v_ps_max, name=f'Vps_{t}')

    # ── Outage indicator ──
    delta = {}
    for t in range(N_T):
        delta[t] = m.addVar(vtype=GRB.BINARY, name=f'd_{t}')

    # ── Q_in (derived) and Q_out (derived) as expressions ──
    # Build cascade: process stations in topological order (chain is already topo-sorted)
    Q_out_expr = {}  # Q_out[s, t] as linear expressions

    m.update()

    # ── Constraints ──
    print(f'    Building constraints ({N_T} hours, {len(chain)} stations)...', end=' ', flush=True)

    for s in chain:
        fr = f_reg[s] if s in sto_list and f_reg[s] < 1.0 else (1.0 if s in sto_list else 0.0)
        h_s = h_bar.get(s, h_max[s])
        eta_s = eta_turb[s]
        power_coeff = eta_s * rho * g * h_s / 1e6

        for t in range(N_T):
            # ── Cascade inflow ──
            q_in_expr = Q_nat_flat[s][t]  # start with natural flow (constant)

            for i_u, u in enumerate(direct_upstream_indices[s]):
                tau = direct_upstream_tau[s][i_u] if i_u < len(direct_upstream_tau[s]) else 0
                t_delayed = t - tau
                if t_delayed >= 0 and u in Q_out_expr and (u, t_delayed) in Q_out_expr:
                    q_in_expr = q_in_expr - Q_nat_flat[u][t_delayed] + Q_out_expr[u, t_delayed]
                # If t_delayed < 0, use natural flow (no upstream regulation effect)

            # ── Station type dispatch ──
            if s in sto_list:
                q_ror_total = (1.0 - fr) * q_in_expr if fr < 1.0 else 0.0

                # Turbine capacity shared: Q_reg + Q_ror_turb ≤ Q_turb_max
                m.addConstr(Q_reg[s, t] + Q_ror_turb[s, t] <= Q_max_turb[s])

                # Q_ror_turb ≤ available RoR flow
                if isinstance(q_ror_total, (int, float)):
                    m.addConstr(Q_ror_turb[s, t] <= max(q_ror_total, 0))
                else:
                    m.addConstr(Q_ror_turb[s, t] <= q_ror_total)

                # Environmental flow
                m.addConstr(Q_reg[s, t] >= Q_env_flat[s][t])

                # Power = η·ρ·g·h̄·Q_turb / 1e6
                m.addConstr(P_sto[s, t] == power_coeff * (Q_reg[s, t] + Q_ror_turb[s, t]))

                # Water balance (only regulated portion enters/leaves reservoir)
                q_frac_in = fr * q_in_expr if isinstance(q_in_expr, (int, float)) else fr * q_in_expr
                net_precip = (precip_flat[s][t] - evap_flat[s][t]) * A_bar[s] / rho

                if t < N_T:
                    m.addConstr(V[s, t+1] == V[s, t]
                                + (q_frac_in - Q_reg[s, t] - Q_spill[s, t] + net_precip) * secs_hr)

                # Q_out for cascade propagation
                if isinstance(q_ror_total, (int, float)):
                    ror_bypass = q_ror_total - Q_ror_turb[s, t]
                else:
                    ror_bypass = q_ror_total - Q_ror_turb[s, t]
                Q_out_expr[s, t] = Q_reg[s, t] + Q_ror_turb[s, t] + Q_spill[s, t] + ror_bypass

            elif type_unified[s] == 'PS' and s == ps_main:
                # PS: Q_out = Q_in + Q_gen - Q_pump
                Q_out_expr[s, t] = q_in_expr + Q_gen_ps[t] - Q_pump_ps[t]

                # PS power-flow relationship (h_PS constant → linear)
                m.addConstr(P_gen[t] == pc_gen * Q_gen_ps[t])
                m.addConstr(P_pump[t] == pc_pump * Q_pump_ps[t])

                # Mutual exclusion
                m.addConstr(z_gen[t] + z_pump[t] <= 1)
                m.addConstr(Q_gen_ps[t] <= Q_max_turb[ps] * z_gen[t])
                m.addConstr(Q_pump_ps[t] <= Q_max_pump[ps] * z_pump[t])

                # River flow constraint for pumping
                if isinstance(q_in_expr, (int, float)):
                    m.addConstr(Q_pump_ps[t] <= 0.9 * max(q_in_expr, 0))
                else:
                    m.addConstr(Q_pump_ps[t] <= 0.9 * q_in_expr)

                # PS water balance
                if t < N_T:
                    m.addConstr(V_ps[t+1] == V_ps[t]
                                + (Q_pump_ps[t] - Q_gen_ps[t] - evap_ps_rate) * secs_hr)

            elif s in ror_list:
                # ROR: Q_out = Q_in, power from min(Q_in, Q_turb_max)
                Q_out_expr[s, t] = q_in_expr

                if isinstance(q_in_expr, (int, float)):
                    q_turb_ror = min(q_in_expr, Q_max_turb[s])
                    p_ror_val = min(q_turb_ror * power_coeff, P_r_turb[s])
                    m.addConstr(P_ror_var[s, t] == max(p_ror_val, 0))
                else:
                    # Q_in is an expression - use auxiliary variable
                    q_turb_aux = m.addVar(lb=0, ub=Q_max_turb[s], name=f'qt_ror_{s}_{t}')
                    m.addConstr(q_turb_aux <= q_in_expr)
                    m.addConstr(q_turb_aux <= Q_max_turb[s])
                    m.addConstr(P_ror_var[s, t] <= power_coeff * q_turb_aux)

            else:
                # Other PS (not ps_main) - pass-through
                Q_out_expr[s, t] = q_in_expr

    # ── STO initial volume and year-boundary continuity ──
    for s in sto_list:
        m.addConstr(V[s, 0] == V_CONV_hourly[0, 0, s])
        # Year boundaries: continuous volume
        for y_idx, y_start, y_end, nhrs in time_ranges:
            if y_idx > 0:
                prev_end = time_ranges[y_idx - 1][2]  # end of previous year
                m.addConstr(V[s, y_start] == V[s, prev_end])

    # ── PS initial volume and year-boundary ──
    if ps_main is not None:
        m.addConstr(V_ps[0] == v_ps_max * 0.5)
        for y_idx, y_start, y_end, nhrs in time_ranges:
            if y_idx > 0:
                prev_end = time_ranges[y_idx - 1][2]
                m.addConstr(V_ps[y_start] == V_ps[prev_end])

    # ── Terminal storage ≥ initial: forbid draining the reservoir at the end of the horizon to
    #    inflate ELCC (the firm capacity must be sustainable, not borrowed from end-of-period storage).
    for s in sto_list:
        m.addConstr(V[s, N_T] >= V_CONV_hourly[0, 0, s])
    if ps_main is not None:
        m.addConstr(V_ps[N_T] >= v_ps_max * 0.5)

    # ── Outage constraints ──
    M_big = sum_cap + ps_cap + 100  # Big-M value
    P_chain_vars = []

    for t in range(N_T):
        # P_chain(t) = Σ P_sto + Σ P_ror + P_gen_PS
        p_total_parts = []
        for s in sto_list:
            p_total_parts.append(P_sto[s, t])
        for s in ror_list:
            p_total_parts.append(P_ror_var[s, t])
        if ps_main is not None:
            p_total_parts.append(P_gen[t])

        p_chain_t = gp.quicksum(p_total_parts)
        # Net out PS pumping load: a pumping PS CONSUMES power, so it must be subtracted from the
        # chain's net supply to the load (otherwise pumping is "free" and ELCC is inflated).
        if ps_main is not None:
            p_chain_t = p_chain_t - P_pump[t]

        # Big-M: P_chain ≥ ELCC × L(t) - M × δ(t)
        L_t = L_norm_flat[t]
        if L_t > 0:
            m.addConstr(p_chain_t >= elcc * L_t - M_big * delta[t])

    # Outage budget
    m.addConstr(gp.quicksum(delta[t] for t in range(N_T)) <= OUTAGE_MAX * N_T)

    # ── Objective ──
    m.setObjective(elcc, GRB.MAXIMIZE)

    m.update()
    build_time = time.time() - t_build
    n_vars = m.NumVars
    n_constrs = m.NumConstrs
    n_binary = sum(1 for v in m.getVars() if v.VType == GRB.BINARY)
    print(f'done ({build_time:.1f}s)')
    print(f'    Variables: {n_vars:,} ({n_binary:,} binary), Constraints: {n_constrs:,}')

    # ── Solve ──
    print(f'    Solving...', flush=True)
    t_solve = time.time()
    m.optimize()
    solve_time = time.time() - t_solve

    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        opt_elcc = elcc.X
        n_outage = sum(1 for t in range(N_T) if delta[t].X > 0.5)
        outage_pct = n_outage / N_T * 100

        # Rule-based comparison
        solo_sum = sum(ELCC_opt[h] for h in chain if np.isfinite(ELCC_opt[h]))

        print(f'    ✓ MILP ELCC = {opt_elcc:.1f} MW  (Σrule-based={solo_sum:.0f} MW)')
        print(f'      Outage: {n_outage}/{N_T} hours ({outage_pct:.2f}%)')
        print(f'      Solve time: {solve_time:.1f}s, Gap: {m.MIPGap*100:.2f}%')

        if ps_main is not None:
            gen_total = sum(P_gen[t].X for t in range(N_T))
            pump_total = sum(P_pump[t].X for t in range(N_T))
            gen_hrs = sum(1 for t in range(N_T) if P_gen[t].X > 0.1)
            pump_hrs = sum(1 for t in range(N_T) if P_pump[t].X > 0.1)
            print(f'      PS gen: {gen_total/N_T:.1f}MW avg, {gen_hrs}h ({gen_hrs/N_T*100:.1f}%)')
            print(f'      PS pump: {pump_total/N_T:.1f}MW avg, {pump_hrs}h ({pump_hrs/N_T*100:.1f}%)')

        ELCC_gurobi[frozenset(chain)] = {
            'elcc': opt_elcc, 'solo_sum': solo_sum,
            'outage_hrs': n_outage, 'solve_time': solve_time,
            'gap': m.MIPGap, 'chain': chain,
        }
    else:
        print(f'    ✗ Solver status: {m.Status} (infeasible or error)')
        ELCC_gurobi[frozenset(chain)] = {'elcc': 0, 'status': m.Status}

    del m  # free memory

# ═══════════════════════════════════════════════════════════
# D.3) Summary
# ═══════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'MILP ELCC Summary')
print(f'{"="*60}')
print(f'{"Chain":50s} {"Rule":>6s} {"MILP":>6s} {"Gain":>6s} {"Time":>5s}')
total_rule = 0; total_milp = 0
for key, res in ELCC_gurobi.items():
    chain = res.get('chain', [])
    names = ' → '.join(HPP_name[h][:10] for h in chain[:4])
    if len(chain) > 4: names += '...'
    rule = res.get('solo_sum', 0)
    milp = res.get('elcc', 0)
    st = res.get('solve_time', 0)
    total_rule += rule; total_milp += milp
    print(f'  {names:48s} {rule:6.0f} {milp:6.0f} {milp-rule:+6.0f} {st:5.1f}s')

print(f'  {"TOTAL":48s} {total_rule:6.0f} {total_milp:6.0f} {total_milp-total_rule:+6.0f}')
print(f'\n  Note: MILP ELCC is a perfect-foresight upper bound.')
print(f'  Gap includes both dispatch optimization potential and value of perfect information.')
