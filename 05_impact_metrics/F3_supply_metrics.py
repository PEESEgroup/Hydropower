"""
F3_supply_metrics.py - Program 3 · Irrigation and water supply (Scenario E, evaluated per river segment)
(Hydropower_dispatch_impact_calculation_plan.md §5 ; REVUB_F3_cascade_and_shared_downstream_revision_plan.md)

Per river SEGMENT (from F3_prepare_routing.py - segment = reaches with the same
controlling upstream-dam set U), every dispatch scenario vs natural, MONTHLY:
  supply = sum of the controlling dams' releases  (E1: as-is; E2: + area-ratio lateral)
  EFR (VMF, natural regime, fixed) → available supply → RRV / WSI / water-gap
Handles cascades (dam stops at next dam) and confluences (segment summed over its
dams) with no AREA double-counting. Uses ONLY model future dam flows + static drainage
area; no historical/borrowed downstream flow.

LIMITATIONS (documented, see Codex review):
 - SUPPLY double-count across NESTED cascades (#2): each segment's supply = the FULL release
   of its controlling dams; water already withdrawn by an UPSTREAM irrigation segment is NOT
   subtracted. Where one segment's dam set ⊂ a downstream segment's set (e.g. Argentina EL TIGRE
   credited to both a 58 and a 73 Mm³/yr demand), downstream Rel/SI are optimistic / WSI-gap
   conservative. Correct progressive consumptive-withdrawal routing is future work.
 - COVERAGE (#1): segments whose controlling dams are all UNSIMULATED (dropped by A's capacity/
   GRFR filters) have no dispatch flow; they are emitted as explicit 'dropped_no_sim_dam' rows
   (not silently skipped). The `ndam=kept/total` note flags partial drops.

  E1 supply  = Σ_d Q(d)                                   (ignores lateral, conservative)
  E2 supply  = [Σ_d Q_nat(d)]·A_outlet/A_dams + Σ_d ΔQ_d  (area-ratio lateral)

CLI:  python F3_supply_metrics.py <region_path> [gcm] [ssp] [scenario_tag]
"""

import os
import sys
import numpy as np
import pandas as pd

import F_impact_common as C

TOPIC = 'supply'
_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_SEC = _DAYS * 86400.0
VMF_LOW, VMF_INT, VMF_HIGH = 0.60, 0.45, 0.30


def load_segments(hydro_dir):
    path = os.path.join(hydro_dir, 'supply_segments.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def monthly_means(arr, hpp, hrs):
    out = np.full((arr.shape[1], 12), np.nan)
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs)
        mon = C.month_index(q.size, C.is_leap_hours(int(hrs[y])))
        for m in range(1, 13):
            sel = q[mon == m]; sel = sel[np.isfinite(sel)]
            if sel.size:
                out[y, m - 1] = sel.mean()
    return out


def efr_vmf(mm):
    mmf = np.nanmean(mm, axis=0); maf = np.nanmean(mmf)
    if not np.isfinite(maf) or maf <= 0:
        return np.full(12, np.nan)
    frac = np.where(mmf <= 0.4 * maf, VMF_LOW, np.where(mmf <= 0.8 * maf, VMF_INT, VMF_HIGH))
    return frac * mmf


def _stack_sum(mats):
    """Sum dam monthly-mean grids, returning NaN (not 0) where EVERY dam is missing that
    month (issue #12a: np.nansum turned an all-missing month into a spurious 0-flow month)."""
    a = np.stack(mats, axis=0)                       # [n_dam, Y, 12]
    valid = np.isfinite(a).any(axis=0)
    return np.where(valid, np.nansum(a, axis=0), np.nan)


def reliability(mm_flow, efr, demand_Mm3_yr, leap_byyear=None):
    """RRV / WSI / gap / EFR-compliance for a monthly flow [Y,12] vs demand.

    Missing (NaN) months cannot be scored, so they are excluded from the denominator;
    gap_months/gap_volume are therefore ANNUALIZED over the observed months (×12/n_valid)
    rather than divided by raw year count (issue #12b), and data_coverage is reported so
    sparse-data inflation of reliability is visible. Month length is leap-year aware
    (issue #12c: a static Feb=28 under-counted leap-Feb volume by 3.4%)."""
    res = {}
    finite = np.isfinite(mm_flow)
    res['EFR_compliance'] = float(np.sum((mm_flow >= efr[None, :]) & finite) / max(1, np.sum(finite)))
    if not (np.isfinite(demand_Mm3_yr) and demand_Mm3_yr > 0):
        return res
    days = np.tile(_DAYS, (mm_flow.shape[0], 1)).astype(float)     # [Y,12]
    if leap_byyear is not None:
        days[np.asarray(leap_byyear, dtype=bool), 1] = 29.0        # leap February
    month_sec = days * 86400.0
    S = np.maximum(0.0, mm_flow - efr[None, :]) * month_sec        # m³/month, NaN where missing
    s = S[finite].ravel()
    n_valid = s.size
    if n_valid == 0:
        return res
    D = demand_Mm3_yr * 1e6 / 12.0
    met = s >= D; fail = ~met; deficit = np.maximum(0.0, D - s); pos = deficit[deficit > 0]
    res['Rel'] = float(np.mean(met))
    res['Vul'] = float(min(1.0, pos.mean() / D)) if pos.size else 0.0
    # Resilience (fail→recovery rate). Count transitions only between months ADJACENT IN TIME and
    # both present - the old flattened-valid array spuriously paired across NaN-dropped months
    # (issue #2: a fail in Mar 'recovers' in May if Apr is missing). Years are consecutive, so the
    # Dec→Jan cross-year pair IS a real adjacency and is kept.
    flat = S.ravel(); fin = np.isfinite(flat)
    met_g = (flat >= D) & fin; fail_g = (flat < D) & fin
    adj = fin[:-1] & fin[1:]
    nfail = int(np.sum(fail))
    res['Res'] = float(np.sum(fail_g[:-1] & met_g[1:] & adj) / nfail) if nfail else 1.0
    res['SI'] = float(max(0.0, res['Rel'] * res['Res'] * (1 - res['Vul'])) ** (1 / 3))
    res['WSI'] = float(D * n_valid / s.sum()) if s.sum() > 0 else np.inf
    res['gap_months_per_year'] = float(np.sum(fail) / n_valid * 12.0)        # annualized over observed
    res['gap_volume_Mm3_yr'] = float(deficit.sum() / 1e6 / n_valid * 12.0)
    res['data_coverage'] = float(n_valid / mm_flow.size)
    return res


def run(region_path=None, gcm=None, ssp=None, tag=None):
    cfg = C.resolve_config(region_path, gcm, ssp, tag)
    scenarios, meta = C.load_scenarios(cfg['output_dir'])
    static = C.load_static(cfg['hydro_dir'], meta['HPP_name'], cfg['output_dir'])
    segs = load_segments(cfg['hydro_dir'])
    hrs = meta['hrs_byyear']; nat = scenarios['nat']
    leap = (np.asarray(hrs, dtype=int) >= 8784)              # per-year leap flag (issue #12c)
    disp = [k for k in scenarios if k != 'nat']
    # map controlling dam -> npz station INDEX by HYRIV (issue #1: dam NAME is non-unique - two
    # 'Funil' in brazil_sudeste - so a name->idx map silently scores both against the last index).
    # static['snap_HYRIV_ID'] is position-aligned to HPP order via load_static(...output_dir).
    hyriv2idx = {}
    if 'snap_HYRIV_ID' in static.columns:
        for i in range(meta['n_HPP']):
            h = static.iloc[i].get('snap_HYRIV_ID')
            if np.isfinite(h):
                hyriv2idx[int(h)] = i
    name2idx = {n: i for i, n in enumerate(meta['HPP_name'])}   # fallback only

    if segs is None:
        print('  ERROR: no supply_segments.csv - run F3_prepare_routing.py first'); return
    irr_segs = segs[segs['demand_Mm3_yr'] > 0]
    print(f'[F3 supply] {cfg["region_name"]} {cfg["gcm"]}/{cfg["ssp"]}'
          f'{("/" + cfg["tag"]) if cfg["tag"] else ""}  {len(segs)} segments '
          f'({len(irr_segs)} w/ irrigation, {(segs["n_dams"]>1).sum()} confluence)')

    # cache monthly means per dam per scenario
    cache = {}
    def mm_of(scen, idx):
        k = (scen, idx)
        if k not in cache:
            cache[k] = monthly_means(scenarios[scen], idx, hrs)
        return cache[k]

    rows = []
    for _, seg in irr_segs.iterrows():
        # resolve controlling dams to npz indices by HYRIV (unique); fall back to name only if the
        # segment file predates the dam_hyrivs column.
        if 'dam_hyrivs' in seg and pd.notna(seg['dam_hyrivs']) and hyriv2idx:
            all_h = [int(x) for x in str(seg['dam_hyrivs']).split('|') if x not in ('', 'nan')]
            idxs = [hyriv2idx[h] for h in all_h if h in hyriv2idx]
            n_total = len(all_h)
        else:
            all_dams = str(seg['dams']).split('|')
            idxs = [name2idx[d] for d in all_dams if d in name2idx]
            n_total = len(all_dams)
        if not idxs:
            # every controlling dam is UNSIMULATED (dropped by A's capacity/GRFR filters) -> no
            # dispatch flow exists. Emit an explicit coverage row instead of silently dropping
            # (issue #1: this demand otherwise vanishes from the table with no record).
            rows.append(C.make_row(cfg, static.iloc[0].to_dict(), 0, TOPIC, 'nat',
                                   'dropped_no_sim_dam',
                                   f'seg={seg["seg_id"]};demand_Mm3_yr={seg["demand_Mm3_yr"]:.2f};'
                                   f'dams={seg["dams"][:60]}',
                                   value_nat=np.nan, value_reg=np.nan, year_agg='monthly'))
            continue
        srow = static.iloc[idxs[0]].to_dict()           # representative coords
        n_drop = n_total - len(idxs)                     # partially-unsimulated control set
        dem = seg['demand_Mm3_yr']
        A_ratio = (seg['A_outlet_skm'] / seg['A_dams_skm']
                   if seg['A_dams_skm'] > 0 else 1.0)
        mm_nat_sum = _stack_sum([mm_of('nat', i) for i in idxs])
        efr = efr_vmf(mm_nat_sum)                        # E1 EFR (natural sum regime)
        efr2 = efr_vmf(mm_nat_sum * A_ratio)
        for method, fac, ef in [('E1', 1.0, efr), ('E2', A_ratio, efr2)]:
            nat_flow = mm_nat_sum * fac
            rn = reliability(nat_flow, ef, dem, leap)
            for scen in disp:
                mm_reg_sum = _stack_sum([mm_of(scen, i) for i in idxs])
                reg_flow = nat_flow + (mm_reg_sum - mm_nat_sum)   # area-scale nat, add Δ
                rr = reliability(reg_flow, ef, dem, leap)
                note = (f'method={method};seg={seg["seg_id"]};ndam={len(idxs)}/{n_total}'
                        f'{(";dropped="+str(n_drop)) if n_drop else ""};dams={seg["dams"][:40]}')
                for kk in rr:
                    rows.append(C.make_row(cfg, srow, idxs[0], TOPIC, scen, kk, note,
                                           value_nat=rn.get(kk, np.nan), value_reg=rr[kk],
                                           year_agg='monthly'))

    df, path = C.write_long_table(rows, TOPIC, cfg['output_dir'])
    print(f'  wrote {len(df)} rows → {path}')
    for m in ['SI', 'gap_months_per_year']:
        for method in ['E1', 'E2']:
            sub = df[(df.metric == m) & (df.scenario == 'BAL_C') & (df.param.str.contains(f'method={method}'))]
            if not sub.empty:
                print(f'    {m:20s} {method} BAL_C: nat={sub["value_nat"].mean():.3f} reg={sub["value_reg"].mean():.3f}')
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0] if len(a) > 0 else None, a[1] if len(a) > 1 else None,
        a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
