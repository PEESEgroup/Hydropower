"""Build the full DC-vs-CONV / DC-vs-typical change tables for EVERY impact metric, by category.

Reads the raw impact_<topic>_summary.csv (mri-esm2-0/ssp370/flat) across all regions, restricts to the
DC-serving units (exactly the (region, station_idx) in fig4_eco/fig4_sed and (region, seg) in
fig4_sup), collapses multi-param rows to ONE value per (unit, scenario, metric) [prefer season=total,
else mean over params/methods/b-values], then per metric reports:
  n_units, %affected, median Δ(DC−CONV) and Δ(DC−typical) over affected units, and the median
  RELATIVE change (% of the CONV / BAL_C baseline) so metrics on different scales are comparable.
Validated against the existing fig4 CSVs for the metrics they contain. Run with --validate to check.

Run: /home/cfeng/.conda/envs/pybkb/bin/python fig4_change_table.py [--validate]
"""
import os, re, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
# impact summaries are computed under the TYPICAL tag (that is where BAL_C = balancing the typical grid
# lives; CONV is load-independent so it equals flat). Synced from S3 into data/_summaries.
SUMROOT = os.path.join(DATA, '_summaries'); GCM = 'mri-esm2-0'; SSP = 'ssp370'; TAG = 'typical'
EPS = 1e-9
NAMES = {'reversals_per_year_hourly': 'flow reversals (per yr)', 'RR_down_p95_pctmean': 'down-ramp severity (% mean)',
         'RR_down_p95': 'down-ramp rate p95', 'RR_down_mean': 'down-ramp rate mean', 'RR_up_p95': 'up-ramp rate p95',
         'VRR_exceed_h_13cmh': 'lethal drawdown (h/yr >13cm/h)', 'VRR_exceed_h_10cmh': 'lethal drawdown (h/yr >10cm/h)',
         'VRR_p95_cmh': 'stage-change rate p95 (cm/h)', 'varial_area_m2': 'varial zone area (m²)',
         'varial_width_m': 'varial zone width (m)', 'HAI_mean_abs_alt': 'hydrologic alteration (HAI)',
         'HAI_group': 'HAI by group', 'IHA': 'IHA parameter', 'Ecodeficit': 'ecodeficit', 'Ecosurplus': 'ecosurplus',
         'HP1': 'high-pulse count', 'Q1.67': 'channel-forming flow Q1.67',
         'STCI': 'sediment-transport capacity (STCI)', 'Qeff': 'effective discharge', 'Qstar_annualmax': 'Q* annual max',
         'Qstar_Q5pct': 'Q* at Q5%', 'P_retention_frac': 'particulate retention frac', 'residence_time_yr': 'residence time (yr)',
         'gap_months_per_year': 'irrigation gap (months/yr)', 'gap_volume_Mm3_yr': 'irrigation gap volume (Mm³/yr)',
         'SI': 'sustainability index', 'Rel': 'reliability', 'Res': 'resilience', 'Vul': 'vulnerability',
         'WSI': 'water-supply index', 'EFR_compliance': 'env. flow compliance', 'data_coverage': 'data coverage'}


def summary_paths(topic):
    # local no-tag DC summaries (.../ssp370/impact_*.csv) - the only files that carry DC_A/B/C + CONV
    # + BAL_C together, so DC-vs-CONV and DC-vs-typical are internally consistent within one file.
    return glob.glob(os.path.join('/datasets/swat_global', '*', '*', 'revub_output', GCM, SSP,
                                  'impact_%s_summary.csv' % topic))


def load_topic(topic):
    dfs = []
    for p in summary_paths(topic):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    d = pd.concat(dfs, ignore_index=True)
    return d


def seg_from_param(param):
    m = re.search(r'seg=(\d+)', str(param))
    return int(m.group(1)) if m else -1


def collapse(d, unit_col):
    """One value_reg per (region, unit, scenario, metric): prefer season=total, else mean over rows."""
    d = d.copy()
    p = d['param'].astype(str)
    d['_istotal'] = p.str.contains('season=total')
    has_total = d.groupby(['region', unit_col, 'scenario', 'metric'])['_istotal'].transform('any')
    keep = (~has_total) | d['_istotal']
    d = d[keep]
    g = d.groupby(['region', unit_col, 'metric', 'scenario'])['value_reg'].mean().reset_index()
    return g.pivot_table(index=['region', unit_col, 'metric'], columns='scenario', values='value_reg').reset_index()


def build(topic, unit_col, dc_units):
    d = load_topic(topic)
    if unit_col == 'seg':
        d['seg'] = d['param'].map(seg_from_param)
    piv = collapse(d, unit_col)
    for c in ['CONV', 'BAL_C', 'DC_A']:                                      # guarantee scenario columns
        if c not in piv:
            piv[c] = np.nan
    piv = piv.merge(dc_units, on=['region', unit_col], how='inner')          # restrict to DC-serving units
    rows = []
    for metric, g in piv.groupby('metric'):
        g = g.dropna(subset=['CONV', 'DC_A'])
        if not len(g):
            continue
        dconv = (g['DC_A'] - g['CONV']).to_numpy()
        dtyp = (g['DC_A'] - g['BAL_C']).to_numpy() if 'BAL_C' in g else np.full(len(g), np.nan)
        base = g['CONV'].to_numpy(); baset = g['BAL_C'].to_numpy() if 'BAL_C' in g else np.full(len(g), np.nan)
        ac = np.abs(dconv) > EPS
        at = np.abs(dtyp) > EPS
        def relpct(delta, b, mask):
            m = mask & (np.abs(b) > EPS)
            return np.median(np.abs(delta[m] / b[m])) * 100 if m.any() else np.nan
        rows.append(dict(metric=metric, name=NAMES.get(metric, metric), n=len(g),
                         pct_aff_conv=100 * ac.mean(),
                         med_dCONV=np.median(dconv[ac]) if ac.any() else 0.0,
                         relpct_CONV=relpct(dconv, base, ac),
                         med_dTYP=np.median(dtyp[at]) if at.any() else 0.0,
                         relpct_TYP=relpct(dtyp, baset, at)))
    return pd.DataFrame(rows)


def fmt_table(df, title):
    df = df.sort_values('relpct_CONV', ascending=False, na_position='last')
    print('\n### %s\n' % title)
    print('| metric | n | %% affected | med Δ DC−CONV | rel%% CONV | med Δ DC−typ | rel%% typ |')
    print('|---|--:|--:|--:|--:|--:|--:|')
    for _, r in df.iterrows():
        def g(x, p=2):
            return '-' if (x is None or (isinstance(x, float) and np.isnan(x))) else (('%+.*g' % (p, x)) if abs(x) < 1e4 else '%+.0f' % x)
        print('| %s | %d | %.0f%% | %s | %s | %s | %s |' % (
            r['name'], r['n'], r['pct_aff_conv'], g(r['med_dCONV']),
            ('%.1f%%' % r['relpct_CONV']) if not np.isnan(r['relpct_CONV']) else '-',
            g(r['med_dTYP']), ('%.1f%%' % r['relpct_TYP']) if not np.isnan(r['relpct_TYP']) else '-'))


def main():
    eco_units = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))[['region', 'station_idx']].drop_duplicates()
    sed_units = pd.read_csv(os.path.join(DATA, 'fig4_sed.csv'))[['region', 'station_idx']].drop_duplicates()
    sup_units = pd.read_csv(os.path.join(DATA, 'fig4_sup.csv'))[['region', 'seg']].drop_duplicates()
    eco = build('ecology', 'station_idx', eco_units)
    sed = build('sediment', 'station_idx', sed_units)
    sup = build('supply', 'seg', sup_units)

    if '--validate' in sys.argv:
        validate(eco, sed, sup)
        return
    fmt_table(eco, 'ECOLOGY  (DC-serving dams; sorted by relative DC−CONV change)')
    fmt_table(sed, 'SEDIMENT (DC-serving dams)')
    fmt_table(sup, 'SUPPLY   (DC-affected segments)')
    for nm, df in [('eco', eco), ('sed', sed), ('sup', sup)]:
        df.to_csv(os.path.join(DATA, 'change_table_%s.csv' % nm), index=False)


def validate(eco, sed, sup):
    """Compare per-station collapsed CONV/BAL_C/DC_A against the fig4 CSVs for shared metrics."""
    print('=== VALIDATION (mean abs diff vs fig4 CSVs, shared metrics) ===')
    # rebuild per-unit (not aggregated) for a few metrics and compare
    d = load_topic('ecology'); piv = collapse(d, 'station_idx')
    f = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    for m in ['reversals_per_year_hourly', 'HAI_mean_abs_alt', 'RR_down_p95_pctmean', 'VRR_exceed_h_13cmh']:
        a = piv[piv.metric == m][['region', 'station_idx', 'CONV', 'BAL_C', 'DC_A']]
        b = f[f.metric == m][['region', 'station_idx', 'CONV', 'BAL_C', 'DC_A']]
        mg = a.merge(b, on=['region', 'station_idx'], suffixes=('_mine', '_fig'))
        for c in ['CONV', 'BAL_C', 'DC_A']:
            diff = np.abs(mg[c + '_mine'] - mg[c + '_fig'])
            print('  ecology %-26s %-6s n=%d  maxdiff=%.3g' % (m, c, len(mg), diff.max() if len(mg) else np.nan))


if __name__ == '__main__':
    main()
