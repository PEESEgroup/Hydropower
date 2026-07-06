"""
F1_sediment_metrics.py - Program 1: Sediment and nutrients
(hydropower_dispatch_impact_computation_plan.md §3)

Metrics, each computed per station for every dispatch scenario vs the same
station's natural flow:
  3.1 STCI sediment transport capacity ratio   Σ Qreg^b / Σ Qnat^b   - core dispatch-sensitive metric
      NB STCI measures dispatch-induced change in downstream transport CAPACITY, NOT the
      sediment MASS trapped behind the dam (trapping efficiency / BQART tonnage were
      intentionally dropped, see §3) - a near-total-trapping dam with little outflow-regime
      change can still score STCI≈1 (#4, by design / scope).
  3.3 effective discharge / below-dam flood-peak ratio Q*   effective discharge, flood-peak ratio
  3.5 dam-building background (residence time / Vollenweider P retention) - scenario-independent, cheap, local data only

(3.2 BQART tonnage and 3.4 stream power were removed: 3.2 only rescales 3.1's STCI by an
external sediment anchor - no new dispatch information - and both need external
relief/lithology/slope/width that isn't worth the trouble. See plan doc §3.)

The rating exponent b drives STCI. b is taken per station from a lithology-derived
table `<hydro_dir>/sediment_params.csv` (cols name_unified, b_low, b_central, b_high),
built by F1_prepare_lithology_b.py. If that table is absent, F1 falls back to the
global default sweep b ∈ {1.5, 2.0, 2.5}.

CLI:  python F1_sediment_metrics.py <region_path> [gcm] [ssp] [scenario_tag]
   or: REVUB_REGION_PATH=... python F1_sediment_metrics.py
"""

import os
import sys
import numpy as np
import pandas as pd

import F_impact_common as C

TOPIC = 'sediment'
B_DEFAULT = {'b_low': 1.5, 'b_central': 2.0, 'b_high': 2.5}   # fallback (no lithology table)
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ---------------------------------------------------------------------------
# Per-station rating exponent b (from lithology table; else global default)
# ---------------------------------------------------------------------------
def load_b_table(hydro_dir, hpp_names, lats=None):
    path = os.environ.get('REVUB_SEDIMENT_PARAMS') or \
        os.path.join(hydro_dir, 'sediment_params.csv')
    cols = ['name_unified', 'b_low', 'b_central', 'b_high', 'litho_class']
    if not os.path.exists(path):
        df = pd.DataFrame({'name_unified': hpp_names})
        for k, v in B_DEFAULT.items():
            df[k] = v
        df['litho_class'] = 'default'
        return df, False
    df = pd.read_csv(path)
    df['name_unified'] = df['name_unified'].astype(str)
    keep = [c for c in cols if c in df.columns]
    # composite (name, round(lat,3)) join so duplicate-named stations (two 'Funil') keep their own b;
    # falls back to name-only if no lat available (older sediment_params.csv).
    if lats is not None and 'lat' in df.columns:
        df['_k'] = list(zip(df['name_unified'], pd.to_numeric(df['lat'], errors='coerce').round(3)))
        df = df[keep + ['_k']].drop_duplicates('_k', keep='first').set_index('_k')
        keys = list(zip([str(n) for n in hpp_names], np.round(np.asarray(lats, float), 3)))
        df = df.reindex(keys).reset_index(drop=True)
        df['name_unified'] = [str(n) for n in hpp_names]
    else:
        df = df[keep].drop_duplicates('name_unified', keep='first').set_index('name_unified')
        df = df.reindex(hpp_names).reset_index().rename(columns={'index': 'name_unified'})
    for k, v in B_DEFAULT.items():                       # fill any gaps with default
        if k not in df:
            df[k] = v
        df[k] = pd.to_numeric(df[k], errors='coerce').fillna(v)
    return df, True


def _finite_pair(qn, qr):
    m = np.isfinite(qn) & np.isfinite(qr) & (qn >= 0) & (qr >= 0)
    return qn[m], qr[m]


# ---------------------------------------------------------------------------
# 3.1 STCI  - Σ Qreg^b / Σ Qnat^b   (per year, then mean over years)
# ---------------------------------------------------------------------------
def stci_station(nat, reg, hpp, hrs, b_values):
    """Return {label: stci_mean} for each (label, b) in b_values."""
    out = {}
    n_Y = nat.shape[1]
    for label, b in b_values:
        per_year = []
        for y in range(n_Y):
            qn = C.valid_series(nat, y, hpp, hrs)
            qr = C.valid_series(reg, y, hpp, hrs)
            qn, qr = _finite_pair(qn, qr)
            if qn.size == 0:
                continue
            denom = np.sum(qn ** b)
            if denom > 0:
                per_year.append(np.sum(qr ** b) / denom)
        out[label] = (b, float(np.mean(per_year)) if per_year else np.nan)
    return out


# ---------------------------------------------------------------------------
# 3.3 Effective discharge + flood-peak ratio Q*
# ---------------------------------------------------------------------------
def effective_discharge(series_pooled, b, n_bins=25):
    """Modal sediment-transporting discharge: argmax over bins of (freq × Q^b)."""
    q = series_pooled[series_pooled > 0]
    if q.size < n_bins:
        return np.nan
    edges = np.logspace(np.log10(q.min()), np.log10(q.max()), n_bins + 1)
    counts, _ = np.histogram(q, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    load = counts * (centers ** b)
    if not np.any(load > 0):
        return np.nan
    return float(centers[int(np.argmax(load))])


def qstar_and_qeff(nat, reg, hpp, hrs, b):
    nat_peaks, reg_peaks, nat_pool, reg_pool = [], [], [], []
    for y in range(nat.shape[1]):
        qn = C.valid_series(nat, y, hpp, hrs)
        qr = C.valid_series(reg, y, hpp, hrs)
        qn, qr = _finite_pair(qn, qr)
        if qn.size == 0:
            continue
        nat_peaks.append(np.max(qn)); reg_peaks.append(np.max(qr))
        nat_pool.append(qn); reg_pool.append(qr)
    if not nat_peaks:
        return {}
    nat_pool = np.concatenate(nat_pool); reg_pool = np.concatenate(reg_pool)
    return {
        'Qstar_annualmax': (np.mean(nat_peaks), np.mean(reg_peaks)),
        'Qstar_Q5pct': (np.percentile(nat_pool, 95), np.percentile(reg_pool, 95)),
        'Qeff': (effective_discharge(nat_pool, b), effective_discharge(reg_pool, b)),
    }


# ---------------------------------------------------------------------------
# 3.5 Build-the-dam background (scenario-independent; local data only)
#     Vollenweider P retention from reservoir residence time τ = V / Q.
# ---------------------------------------------------------------------------
def background_station(nat, hpp, hrs, static_row):
    v = np.concatenate([C.valid_series(nat, y, hpp, hrs) for y in range(nat.shape[1])])
    v = v[np.isfinite(v)]
    qbar = v.mean() if v.size else np.nan          # valid hours only (exclude 0-padded tail)
    Q_km3 = qbar * SECONDS_PER_YEAR / 1e9
    V_km3 = static_row.get('max_vol_km3', np.nan)
    tau = (V_km3 / Q_km3) if (np.isfinite(V_km3) and np.isfinite(Q_km3) and Q_km3 > 0) else np.nan
    # P RETENTION (issue #10): R = 1 - [P]_lake/[P]_in = 1 - 1/(1+sqrt(tau)) = sqrt(tau)/(1+sqrt(tau)).
    # Retention INCREASES with residence time (long-residence reservoirs trap more P). The prior
    # code used 1/(1+sqrt(tau)) = the concentration PASS-THROUGH fraction (decreasing with tau) and
    # mislabeled it as retention. (Vollenweider 1976 / Larsen & Mercier 1976 mass-balance.)
    Rp = (np.sqrt(tau) / (1.0 + np.sqrt(tau))) if np.isfinite(tau) and tau >= 0 else np.nan
    return [
        ('residence_time_yr', tau, 'V_max/Q_nat'),
        ('P_retention_frac', Rp,
         'Vollenweider/Larsen-Mercier 1976 retention R=sqrt(tau)/(1+sqrt(tau)) (increases w/ tau)'),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(region_path=None, gcm=None, ssp=None, tag=None):
    cfg = C.resolve_config(region_path, gcm, ssp, tag)
    scenarios, meta = C.load_scenarios(cfg['output_dir'])
    static = C.load_static(cfg['hydro_dir'], meta['HPP_name'], cfg['output_dir'])
    btab, has_litho = load_b_table(cfg['hydro_dir'], meta['HPP_name'], static.get('lat_unified'))
    hrs = meta['hrs_byyear']
    nat = scenarios['nat']
    disp_keys = [k for k in scenarios if k != 'nat']

    print(f'[F1 sediment] {cfg["region_name"]} {cfg["gcm"]}/{cfg["ssp"]}'
          f'{("/" + cfg["tag"]) if cfg["tag"] else ""}')
    print(f'  stations={meta["n_HPP"]}  scenarios={disp_keys}')
    print(f'  b source: {"lithology table" if has_litho else "global default 1.5/2.0/2.5"}')

    rows = []
    for hpp in range(meta['n_HPP']):
        srow = static.iloc[hpp].to_dict()
        brow = btab.iloc[hpp].to_dict()
        b_values = [('b_low', float(brow['b_low'])),
                    ('b_central', float(brow['b_central'])),
                    ('b_high', float(brow['b_high']))]
        litho = brow.get('litho_class', '')
        b_mid = float(brow['b_central'])

        # 3.5 background (scenario-independent)
        for metric, val, note in background_station(nat, hpp, hrs, srow):
            rows.append(C.make_row(cfg, srow, hpp, TOPIC, 'background', metric, '',
                                   value_nat=val, value_reg=np.nan, year_agg='pooled',
                                   notes=note))

        for scen in disp_keys:
            reg = scenarios[scen]
            # 3.1 STCI (per-station b: low/central/high)
            for label, (b, val) in stci_station(nat, reg, hpp, hrs, b_values).items():
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'STCI',
                                       f'{label} b={b:.2f} litho={litho}',
                                       value_nat=1.0, value_reg=val))
            # 3.3 effective discharge / Q*
            for name, pair in qstar_and_qeff(nat, reg, hpp, hrs, b_mid).items():
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, name,
                                       f'b={b_mid:.2f}' if name == 'Qeff' else '',
                                       value_nat=pair[0], value_reg=pair[1]))

    df, path = C.write_long_table(rows, TOPIC, cfg['output_dir'])
    print(f'  wrote {len(df)} rows → {path}')
    sub = df[(df.metric == 'STCI') & (df.param.str.startswith('b_central'))]
    if not sub.empty:
        print('  mean STCI(central b) by scenario:')
        for scen, g in sub.groupby('scenario'):
            print(f'    {scen:10s} {g["ratio"].mean():.3f}')
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0] if len(a) > 0 else None, a[1] if len(a) > 1 else None,
        a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
