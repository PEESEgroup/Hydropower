"""Three change tables (ecology / sediment / supply) from the validated fig4 CSVs.

For every metric present in fig4_eco/fig4_sed/fig4_sup (DC-serving units, 74 regions), report the
DC-vs-CONVENTIONAL (DC_A − CONV) and DC-vs-TYPICAL-grid (DC_A − BAL_C) change: % of units affected,
median change over affected units (native units), and median change as % of the baseline (so metrics
on different scales are comparable → "which moves most"). Run: python fig4_table.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data')
EPS = 1e-9
NAMES = {'reversals_per_year_hourly': 'flow reversals (per yr)', 'RR_down_p95_pctmean': 'down-ramp severity (% of mean flow)',
         'VRR_exceed_h_13cmh': 'lethal drawdown (h/yr > 13 cm/h)', 'HAI_mean_abs_alt': 'hydrologic alteration (HAI)',
         'STCI': 'sediment-transport capacity (STCI)', 'gap_months_per_year': 'irrigation gap (months/yr)',
         'SI': 'supply sustainability index'}


def rows_for(df, metric_col):
    out = []
    metrics = df[metric_col].unique() if metric_col else [None]
    for m in metrics:
        g = df if metric_col is None else df[df[metric_col] == m]
        g = g.dropna(subset=['CONV', 'DC_A', 'BAL_C'])
        if not len(g):
            continue
        dC = (g.DC_A - g.CONV).to_numpy(); dT = (g.DC_A - g.BAL_C).to_numpy()
        baseC = g.CONV.to_numpy(); baseT = g.BAL_C.to_numpy()
        aC = np.abs(dC) > EPS; aT = np.abs(dT) > EPS

        def rel(d, b, a):
            mm = a & (np.abs(b) > EPS)
            return np.median(np.abs(d[mm] / b[mm])) * 100 if mm.any() else np.nan
        out.append(dict(metric=m or 'STCI', name=NAMES.get(m or 'STCI', m or 'STCI'), n=len(g),
                        pctC=100 * aC.mean(), medC=np.median(dC[aC]) if aC.any() else 0.0, relC=rel(dC, baseC, aC),
                        pctT=100 * aT.mean(), medT=np.median(dT[aT]) if aT.any() else 0.0, relT=rel(dT, baseT, aT)))
    return pd.DataFrame(out)


def g(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    return ('%+.0f' % x) if abs(x) >= 100 else ('%+.2g' % x)


def show(df, title):
    df = df.sort_values('relC', ascending=False, na_position='last')
    print('\n### %s\n' % title)
    print('| metric | n | DC vs CONVENTIONAL |  |  | DC vs TYPICAL grid |  |  |')
    print('| | | % affected | median Δ | % of base | % affected | median Δ | % of base |')
    print('|---|--:|--:|--:|--:|--:|--:|--:|')
    for _, r in df.iterrows():
        print('| %s | %d | %.0f%% | %s | %s | %.0f%% | %s | %s |' % (
            r['name'], r.n, r.pctC, g(r.medC), ('%.1f%%' % r.relC) if not np.isnan(r.relC) else '-',
            r.pctT, g(r.medT), ('%.1f%%' % r.relT) if not np.isnan(r.relT) else '-'))


def main():
    eco = pd.read_csv(os.path.join(DATA, 'fig4_eco.csv'))
    sed = pd.read_csv(os.path.join(DATA, 'fig4_sed.csv'))
    sup = pd.read_csv(os.path.join(DATA, 'fig4_sup.csv'))
    show(rows_for(eco, 'metric'), 'ECOLOGY (DC-serving dams, 74 regions; sorted by relative DC−CONV change)')
    show(rows_for(sed, None), 'SEDIMENT (DC-serving dams, 74 regions)')
    show(rows_for(sup, 'metric'), 'SUPPLY (DC-affected segments, 74 regions)')


if __name__ == '__main__':
    main()
