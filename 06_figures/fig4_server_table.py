"""[runs on the server] All-metric change tables, by category, from the flat DC + typical summaries.

DC_A & CONV come from the FLAT tag (the canonical A-C-D run carrying the DC scenarios); BAL_C (the
typical-grid baseline) from the TYPICAL tag. Restrict to the DC-serving units listed in
dc_units_dam.csv / dc_units_seg.csv (the same set the figures use). Collapse multi-param rows to one
value per (region, unit, scenario, metric): prefer season=total, else mean. Per metric report
%affected, median Δ and median |Δ|/baseline (%) for DC-vs-CONV and DC-vs-typical.

Run on server: /home/ubuntu/miniforge3/envs/revub/bin/python fig4_server_table.py
"""
import glob, os, re
import numpy as np, pandas as pd

ROOT = '/datasets/swat_global'; GCM = 'mri-esm2-0'; SSP = 'ssp370'; EPS = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))


def paths(topic, tag):
    sub = '' if tag == '' else tag + '/'
    return glob.glob('%s/*/*/revub_output/%s/%s/%simpact_%s_summary.csv' % (ROOT, GCM, SSP, sub, topic))


def collapse(d, unit):
    d = d.copy(); p = d['param'].astype(str)
    d['_t'] = p.str.contains('season=total')
    ht = d.groupby(['region', unit, 'scenario', 'metric'])['_t'].transform('any')
    d = d[(~ht) | d['_t']]
    return (d.groupby(['region', unit, 'metric', 'scenario'])['value_reg'].mean().reset_index()
            .pivot_table(index=['region', unit, 'metric'], columns='scenario', values='value_reg').reset_index())


def load(topic, tag, unit):
    ps = paths(topic, tag)
    d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)
    if unit == 'seg':
        d['seg'] = d['param'].map(lambda s: int(re.search(r'seg=(\d+)', str(s)).group(1)) if re.search(r'seg=(\d+)', str(s)) else -1)
    return collapse(d, unit)


def build(topic, unit, units_csv):
    dc = pd.read_csv(os.path.join(HERE, units_csv))
    flat = load(topic, 'flat', unit)                     # CONV, DC_A
    typ = load(topic, 'typical', unit)                   # BAL_C (typical grid)
    flat = flat.merge(dc, on=['region', unit], how='inner')
    keep = ['region', unit, 'metric'] + [c for c in ['CONV', 'DC_A'] if c in flat]
    m = flat[keep].merge(typ[['region', unit, 'metric', 'BAL_C']], on=['region', unit, 'metric'], how='left')
    for c in ['CONV', 'DC_A', 'BAL_C']:
        if c in m:
            m[c] = pd.to_numeric(m[c], errors='coerce')
    rows = []
    for metric, g in m.groupby('metric'):
        g = g.dropna(subset=['CONV', 'DC_A'])
        if not len(g):
            continue
        dC = (g.DC_A - g.CONV).to_numpy(); bC = g.CONV.to_numpy(); aC = np.abs(dC) > EPS
        if 'BAL_C' in g:
            dT = (g.DC_A - g.BAL_C).to_numpy(); bT = g.BAL_C.to_numpy(); aT = np.abs(dT) > EPS
        else:
            dT = np.full(len(g), np.nan); bT = dT; aT = np.zeros(len(g), bool)

        def rel(d, b, a):
            mm = a & (np.abs(b) > EPS)
            return float(np.median(np.abs(d[mm] / b[mm])) * 100) if mm.any() else np.nan
        rows.append(dict(metric=metric, n=len(g), n_typ=int(np.isfinite(dT).sum()),
                         pctC=100 * aC.mean(), medC=float(np.median(dC[aC])) if aC.any() else 0.0, relC=rel(dC, bC, aC),
                         pctT=100 * aT.mean(), medT=float(np.median(dT[aT])) if aT.any() else 0.0, relT=rel(dT, bT, aT)))
    out = pd.DataFrame(rows).sort_values('relC', ascending=False, na_position='last')
    out.to_csv(os.path.join(HERE, 'srv_table_%s.csv' % topic), index=False)
    print('\n### %s (n units, n with typical)\n' % topic.upper())
    print('metric | n | n_typ | %aff_C | medΔ_CONV | rel%_CONV | %aff_T | medΔ_typ | rel%_typ')
    for _, r in out.iterrows():
        print('%-26s %4d %4d | %3.0f%% %+10.4g %7s | %3.0f%% %+10.4g %7s' % (
            r.metric, r.n, r.n_typ, r.pctC, r.medC, ('%.1f%%' % r.relC) if not np.isnan(r.relC) else '-',
            r.pctT, r.medT, ('%.1f%%' % r.relT) if not np.isnan(r.relT) else '-'))


for topic, unit, csv in [('ecology', 'station_idx', 'dc_units_dam.csv'),
                         ('sediment', 'station_idx', 'dc_units_dam.csv'),
                         ('supply', 'seg', 'dc_units_seg.csv')]:
    build(topic, unit, csv)
