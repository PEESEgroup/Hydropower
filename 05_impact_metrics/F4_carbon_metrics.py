"""
F4_carbon_metrics.py - Program Four: carbon emission impact of powering the data center
(hydropower_dispatch_impact_calculation_plan.md §6 ; REVUB_F4_carbon_emissions_plan.md)

Per region × DC scenario (A/B/C): carbon of meeting the data-center demand =
  ① hydro reservoir GHG (E_hydro→DC × reservoir EF) + ② grid backup for the
  shortfall (E_shortfall × grid non-hydro EF).
Energy from D's dc_hydro_summary.csv: E_DC = dc_mean·8760; E_hydro→DC =
min(supply_X, dc_mean)·8760 (firm hydro); E_shortfall = E_DC − E_hydro→DC.
EFs from F4_prepare_emission_factors.py (emission_factors.csv).

carbon[tCO2/yr] = E[MWh/yr] × EF[gCO2eq/kWh] / 1000
net_avoided = counterfactual all-grid carbon − actual(①+②)

DEFERRED / DOCUMENTED (Codex review):
 - PS pumping (#3): pumped-storage pump energy (P_PS_pump_B/C, in the hourly npz only) is NOT
   accounted. It is hydro-sourced (pumps from the river/upper reservoir using surplus hydro,
   NOT charged from the grid), so the real effect is a small round-trip energy LOSS, not a
   grid-EF charge. Measured immaterial: Argentina ~1.5 GWh/yr ≈ 0.12% of E_DC. Caveat only.
 - Cross-region E carbon (#9): this reads only the single-region D summary. Cross-region DC
   supply (E redispatch_cache) would need source-region reservoir EF + demand-region grid EF +
   transmission loss - not yet built, deferred consistently with F1/F2/F3's E handling.

CLI:  python F4_carbon_metrics.py <region_path> [gcm] [ssp] [scenario_tag]
"""

import os
import sys
import numpy as np
import pandas as pd

import F_impact_common as C

TOPIC = 'carbon'
HOURS_YR = 8760.0

# Reservoir-EF fallback: hydro-served regions whose DC-supplying stations have NO G-res
# per-reservoir match (n_matched=0, or matched stations all lack a finite gres EF) get a
# NaN gen-weighted reservoir EF in emission_factors.csv, which would silently drop the whole
# region's hydro carbon (and thus net_carbon_avoided / intensity) to NaN. Instead, fall back
# to the GLOBAL MEDIAN per-reservoir G-res EF (finite rows of reservoir_ef_gres.csv,
# gres_ef_gco2kwh ≈ 11.4 gCO2/kWh) so the region still carries carbon. This recovers
# china_nw (~4 Mt) and belgium. Band multipliers match F4_prepare's GWP_BAND (0.6, 2.5):
# low = GWP100→lower-envelope, high = GWP20/GWP100 time-horizon sensitivity.
_GRES_PATH = '/datasets/swat_global/_carbon_ef/reservoir_ef_gres.csv'
GWP_BAND = (0.6, 2.5)


def _global_median_reservoir_ef():
    """Global median per-reservoir G-res EF over finite gres_ef_gco2kwh rows (≈11.4)."""
    try:
        g = pd.read_csv(_GRES_PATH, low_memory=False)
        v = pd.to_numeric(g['gres_ef_gco2kwh'], errors='coerce')
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan
    except Exception:
        return np.nan


def _carbon_t(e_mwh, ef_gco2kwh):
    return e_mwh * ef_gco2kwh / 1000.0          # MWh × g/kWh /1000 = tCO2


def run(region_path=None, gcm=None, ssp=None, tag=None):
    cfg = C.resolve_config(region_path, gcm, ssp, tag)
    out_dir = cfg['output_dir']
    summ = os.path.join(out_dir, 'dc_hydro_summary.csv')
    ef_path = os.path.join(cfg['hydro_dir'], 'emission_factors.csv')
    if not os.path.exists(summ):
        print(f'  no dc_hydro_summary.csv in {out_dir} - run D first'); return
    if not os.path.exists(ef_path):
        print('  no emission_factors.csv - run F4_prepare_emission_factors.py first'); return
    s = pd.read_csv(summ).iloc[0]
    ef = pd.read_csv(ef_path).iloc[0]
    dc_mean = float(s['dc_mean_mw'])
    E_DC = dc_mean * HOURS_YR                    # MWh/yr
    hyd_bands = {'low': ef['hydro_ef_low'], 'central': ef['hydro_ef_central'], 'high': ef['hydro_ef_high']}
    grid_nh = float(ef['grid_nonhydro_ef']); grid_mg = float(ef['grid_marginal_ef'])

    # RESERVOIR-EF FALLBACK: if the gen-weighted reservoir (hydro) EF is NaN - no G-res match
    # for the DC-supplying stations (n_matched=0, e.g. china_nw / luxembourg, or matched-but-
    # unmatched-EF, e.g. belgium) - use the global median per-reservoir EF (≈11.4 gCO2/kWh)
    # so the region still carries hydro carbon instead of being dropped to NaN.
    hyd_ef_src = 'gres-weighted'
    if not np.isfinite(hyd_bands['central']):
        med = _global_median_reservoir_ef()
        if np.isfinite(med):
            hyd_bands = {'low': med * GWP_BAND[0], 'central': med, 'high': med * GWP_BAND[1]}
            hyd_ef_src = f'global-median-fallback({med:.1f})'
            print(f'  [reservoir-EF fallback] {cfg["region_name"]}: hydro EF NaN '
                  f'(n_matched={ef.get("n_matched", "?")}) -> global median {med:.1f} gCO2/kWh')

    print(f'[F4 carbon] {cfg["region_name"]}  DC mean={dc_mean:.1f}MW E_DC={E_DC/1e3:.1f}GWh/yr '
          f'| hydroEF={hyd_bands["central"]:.1f}[{hyd_ef_src}] gridNH={grid_nh:.0f}[{ef.get("grid_country","")} {ef.get("grid_tier","")}]')

    srow = {'name_unified': cfg['region_name'], 'lat_unified': np.nan, 'lon_unified': np.nan, 'river': ''}
    rows = []
    for letter in ['A', 'B', 'C']:
        col = f'supply_{letter}_mw'
        if col not in s or not np.isfinite(s[col]):
            continue
        supply = float(s[col])
        # Firm hydro energy to the DC. supply_X is the firm-capacity potential ON the DC load
        # shape (ELCC·frac on L_norm_DC), not a constant MW; the flat min(supply, dc_mean)
        # envelope captures ~98-100% of the curve-based energy (issue-#3 audit: 0-1.8% error,
        # because the DC load is near-flat). NB grid_resid_X_mw is the REGION's residual grid
        # load, NOT the DC shortfall - do not use it here.
        E_hyd = min(supply, dc_mean) * HOURS_YR          # firm hydro → DC
        E_short = max(0.0, E_DC - E_hyd)                  # grid backup
        scen = f'DC_{letter}'
        # energy split
        rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'E_hydro_to_DC_GWh', '',
                               value_nat=np.nan, value_reg=E_hyd / 1e3, year_agg='annual'))
        rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'E_grid_shortfall_GWh', '',
                               value_nat=np.nan, value_reg=E_short / 1e3, year_agg='annual'))
        rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'shortfall_frac', '',
                               value_nat=np.nan, value_reg=E_short / E_DC if E_DC else np.nan))
        # ② grid carbon (non-hydro central + marginal upper bound)
        grid_c_nh = _carbon_t(E_short, grid_nh)
        grid_c_mg = _carbon_t(E_short, grid_mg)
        rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'grid_carbon_tCO2_yr',
                               f'grid_nonhydro={grid_nh:.0f};tier={ef["grid_tier"]}',
                               value_nat=np.nan, value_reg=grid_c_nh, year_agg='annual'))
        rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'grid_carbon_tCO2_yr_marginal',
                               f'grid_marginal={grid_mg:.0f}', value_nat=np.nan, value_reg=grid_c_mg))
        # ① hydro carbon (low/central/high reservoir EF band) + total + intensity + net-avoided
        for band, hef in hyd_bands.items():
            hyd_c = _carbon_t(E_hyd, hef)
            total = hyd_c + grid_c_nh
            intensity = total * 1000.0 / E_DC if E_DC else np.nan       # gCO2eq/kWh
            allgrid = _carbon_t(E_DC, grid_nh)                          # counterfactual all-grid
            net_avoided = allgrid - total
            p = f'hydroEF={band}({hef:.0f})'
            rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'hydro_carbon_tCO2_yr', p,
                                   value_nat=np.nan, value_reg=hyd_c, year_agg='annual'))
            rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'total_carbon_tCO2_yr', p,
                                   value_nat=allgrid, value_reg=total, year_agg='annual'))
            rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'dc_carbon_intensity_gCO2_kWh', p,
                                   value_nat=grid_nh, value_reg=intensity, year_agg='annual'))
            rows.append(C.make_row(cfg, srow, 0, TOPIC, scen, 'net_carbon_avoided_tCO2_yr', p,
                                   value_nat=np.nan, value_reg=net_avoided, year_agg='annual'))

    df, path = C.write_long_table(rows, TOPIC, out_dir)
    print(f'  wrote {len(df)} rows → {path}')
    sub = df[(df.metric == 'total_carbon_tCO2_yr') & (df.param.str.contains('central'))]
    for _, r in sub.iterrows():
        print(f'    {r["scenario"]}: total={r["value_reg"]:.0f} tCO2/yr (all-grid={r["value_nat"]:.0f}, '
              f'net_avoided={r["value_nat"]-r["value_reg"]:.0f})')
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0] if len(a) > 0 else None, a[1] if len(a) > 1 else None,
        a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
