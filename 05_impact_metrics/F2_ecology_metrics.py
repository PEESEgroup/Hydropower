"""
F2_ecology_metrics.py - Program 2: Ecology & Biodiversity
(hydropower_dispatch_impact_method.md §4)

Per station, every dispatch scenario vs the same station's natural flow:
  4.1 Ecodeficit / Ecosurplus ecological water deficit/surplus  🟢  FDC-based, seasonal (hemisphere-aware)
  4.2 ramping rate RR / HP1 / reversal count           🟢  pure flow, hourly ramping
  4.3 cm/h vertical down-ramp VRR                    🟡  at-a-station hydraulic-geometry depth proxy (order-of-magnitude)
  4.4 varial zone area                          🟡  wetted-width change × reach length (order-of-magnitude)
  4.5 HAI hydrologic alteration / Q1.67              🟢  IHA-lite alteration, no absolute "probability"

🟡 metrics (4.3, 4.4) use at-a-station hydraulic geometry (no downstream rating curve
exists, see plan §4.3): depth d≈0.27·Q^0.30 (Andreadis et al. 2013), width w≈7.2·Q^0.5.
Results are ORDER-OF-MAGNITUDE, flagged in notes - not true stage.

CLI:  python F2_ecology_metrics.py <region_path> [gcm] [ssp] [scenario_tag]
"""

import os
import sys
import numpy as np
import pandas as pd

import F_impact_common as C

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid; support both.
_trapz = getattr(np, 'trapezoid', None) or np.trapz

TOPIC = 'ecology'
DT_H = 1.0                       # hourly
SEASONS = ['spring', 'summer', 'autumn', 'winter']
DEPTH_A, DEPTH_F = 0.27, 0.30    # d = a·Q^f  (Andreadis et al. 2013 bankfull depth; hc=0.27,hp=0.30 per HydroMT)
WIDTH_A, WIDTH_F = 7.2, 0.50     # w = a·Q^f  (downstream hydraulic geometry)
VRR_THRESH = {'VRR_exceed_h_10cmh': 10.0, 'VRR_exceed_h_13cmh': 13.0}  # trout / salmon


def _finite(a):
    return a[np.isfinite(a)]


def _pool(arr, hpp, hrs, mask_season=None, lat=None, season=None):
    """Concatenate valid-hour series across years (optionally one season)."""
    chunks = []
    for y in range(arr.shape[1]):
        s = C.valid_series(arr, y, hpp, hrs)
        if season is not None:
            lab = C.season_index(s.size, C.is_leap_hours(int(hrs[y])), lat)
            s = s[lab == season]
        chunks.append(s)
    return _finite(np.concatenate(chunks)) if chunks else np.array([])


# ---------------------------------------------------------------------------
# 4.1 Ecodeficit / Ecosurplus  (FDC integral)
# ---------------------------------------------------------------------------
_PGRID = (np.arange(1, 501) - 0.5) / 500.0


def _fdc(values):
    v = np.sort(values)[::-1]                    # descending flow
    if v.size < 10:
        return None
    p = (np.arange(1, v.size + 1) - 0.5) / v.size  # ascending exceedance prob
    return np.interp(_PGRID, p, v)


def ecodeficit(nat_vals, reg_vals, mean_nat=None):
    fn, fr = _fdc(nat_vals), _fdc(reg_vals)
    if mean_nat is None:                              # default: per-pool mean (total)
        mean_nat = np.mean(nat_vals) if nat_vals.size else np.nan
    if fn is None or fr is None or not np.isfinite(mean_nat) or mean_nat <= 0:
        return np.nan, np.nan
    deficit = _trapz(np.maximum(fn - fr, 0.0), _PGRID) / mean_nat
    surplus = _trapz(np.maximum(fr - fn, 0.0), _PGRID) / mean_nat
    return deficit, surplus


# ---------------------------------------------------------------------------
# 4.2 ramping rate RR / HP1 / reversals
# ---------------------------------------------------------------------------
def ramping(arr, hpp, hrs):
    down, up, hp1_days, reversals_yr = [], [], [], []
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs)         # KEEP NaN - do not pre-filter (issue #8b)
        if np.sum(np.isfinite(q)) < 48:
            continue
        dq = np.diff(q) / DT_H                        # diff FIRST, on the gapped series
        dq = dq[np.isfinite(dq)]                      # then drop diffs spanning a NaN gap
        down.append(-dq[dq < 0])                 # down-ramp magnitudes (>0)
        up.append(dq[dq > 0])
        sign = np.sign(dq); sign = sign[sign != 0]
        reversals_yr.append(int(np.sum(np.diff(sign) != 0)))
        nd = q.size // 24
        if nd:
            day = q[:nd * 24].reshape(nd, 24)
            mean = np.nanmean(day, 1)
            ok = np.isfinite(mean) & (mean > 0)
            rng = np.nanmax(day, 1) - np.nanmin(day, 1)
            hp1_days.append(rng[ok] / mean[ok])
    down = np.concatenate(down) if down else np.array([])
    up = np.concatenate(up) if up else np.array([])
    # mean over valid hours only (exclude the 0.0-padded non-leap-year tail)
    vbar = np.concatenate([C.valid_series(arr, y, hpp, hrs) for y in range(arr.shape[1])])
    vbar = vbar[np.isfinite(vbar)]
    qbar = vbar.mean() if vbar.size else np.nan
    return {
        'RR_down_mean': float(down.mean()) if down.size else np.nan,
        'RR_down_p95': float(np.percentile(down, 95)) if down.size else np.nan,
        'RR_up_p95': float(np.percentile(up, 95)) if up.size else np.nan,
        'RR_down_p95_pctmean': float(np.percentile(down, 95) / qbar * 100) if (down.size and qbar > 0) else np.nan,
        'HP1': float(np.median(np.concatenate(hp1_days))) if hp1_days else np.nan,
        'reversals_per_year_hourly': float(np.mean(reversals_yr)) if reversals_yr else np.nan,
    }


# ---------------------------------------------------------------------------
# 4.3 cm/h vertical down-ramp VRR
#   depth model d(Q) = depth_coef · Q^depth_exp  [m]
#   - with REAL channel slope S (RiverATLAS sgr_dk_rav): Manning + AHG width →
#     d=K·Q^0.3, K=(n/(7.2·√S))^0.6  (n=0.035)  ← "true value" path
#   - without slope: global AHG fallback d=0.27·Q^0.30 (Andreadis 2013) ← order-of-magnitude
# ---------------------------------------------------------------------------
MANNING_N = 0.035


def depth_model(slope_m_per_m):
    """Return (coef, exp, note) for the depth-discharge relation."""
    if np.isfinite(slope_m_per_m) and slope_m_per_m > 0:
        K = (MANNING_N / (WIDTH_A * np.sqrt(slope_m_per_m))) ** 0.6
        return K, 0.3, f'Manning real-slope S={slope_m_per_m:.2e}'
    return DEPTH_A, DEPTH_F, 'global AHG d=0.27Q^0.30 (order-of-magnitude)'


def vrr_cmh(arr, hpp, hrs, depth_coef=DEPTH_A, depth_exp=DEPTH_F):
    p95, exc10, exc13, nyear = [], 0.0, 0.0, 0
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs)
        valid = np.isfinite(q)
        if valid.sum() < 48:
            continue
        # KEEP zero-flow hours (issue #8a): a real dewatering 10->0->10 IS the most severe
        # drawdown; dropping zeros erased it. Clamp negatives to 0; NaN stays NaN (0^exp=0 ok).
        q = np.where(valid, np.maximum(q, 0.0), np.nan)
        d = depth_coef * q ** depth_exp           # m
        drop = -np.diff(d) / DT_H * 100.0         # cm/h, positive = down
        drop = drop[np.isfinite(drop) & (drop > 0)]   # drop gap-spanning diffs
        if drop.size:
            p95.append(np.percentile(drop, 95))
            exc10 += np.sum(drop > 10.0); exc13 += np.sum(drop > 13.0); nyear += 1
    return {
        'VRR_p95_cmh': float(np.mean(p95)) if p95 else np.nan,
        'VRR_exceed_h_10cmh': float(exc10 / nyear) if nyear else np.nan,
        'VRR_exceed_h_13cmh': float(exc13 / nyear) if nyear else np.nan,
    }


# ---------------------------------------------------------------------------
# 4.4 varial zone (daily wetted-width change × reach length)
# ---------------------------------------------------------------------------
def varial_area(arr, hpp, hrs, length_km):
    widths = []
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs)
        q = q[np.isfinite(q) & (q >= 0)]
        nd = q.size // 24
        if not nd:
            continue
        day = q[:nd * 24].reshape(nd, 24)
        wmax = WIDTH_A * day.max(1) ** WIDTH_F
        wmin = WIDTH_A * day.min(1) ** WIDTH_F
        widths.append(wmax - wmin)
    if not widths:
        return np.nan, np.nan
    dwidth = float(np.mean(np.concatenate(widths)))          # mean daily varial width (m)
    area = dwidth * length_km * 1000.0 if np.isfinite(length_km) else np.nan
    return dwidth, area


# ---------------------------------------------------------------------------
# 4.5 Full IHA - 33 parameters in 5 groups (Richter et al. 1996), on DAILY flow
# ---------------------------------------------------------------------------
_IHA_WINDOWS = [1, 3, 7, 30, 90]
_MONTHS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# parameter → IHA group (1 magnitude-monthly, 2 extremes, 3 timing, 4 pulses, 5 rate)
_IHA_GROUP = {}
for _m in range(1, 13):
    _IHA_GROUP[f'mag_month{_m:02d}'] = 1
for _w in _IHA_WINDOWS:
    _IHA_GROUP[f'min{_w}day'] = 2; _IHA_GROUP[f'max{_w}day'] = 2
_IHA_GROUP.update({'zero_days': 2, 'baseflow_index': 2,
                   'jd_max': 3, 'jd_min': 3,
                   'lo_pulse_count': 4, 'lo_pulse_dur': 4,
                   'hi_pulse_count': 4, 'hi_pulse_dur': 4,
                   'rise_rate': 5, 'fall_rate': 5, 'reversals': 5})
_TIMING = {'jd_max', 'jd_min'}


def _q_recurrence(annual_max, ri=1.67):
    am = np.sort(_finite(np.asarray(annual_max)))
    if am.size < 5:
        return np.nan
    P = np.arange(1, am.size + 1) / (am.size + 1)             # Weibull non-exceedance
    return float(np.interp(1.0 - 1.0 / ri, P, am))


def _annual_max(arr, hpp, hrs):
    out = []
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs); q = q[np.isfinite(q)]
        if q.size:
            out.append(q.max())
    return out


def _daily_by_year(arr, hpp, hrs):
    out = []
    for y in range(arr.shape[1]):
        q = C.valid_series(arr, y, hpp, hrs)
        nd = q.size // 24
        if nd:
            out.append(np.nanmean(q[:nd * 24].reshape(nd, 24), axis=1))   # daily mean
    return out


def _month_of_day(nd):
    days = _MONTHS_NONLEAP.copy()
    if nd >= 366:
        days[1] = 29
    return np.repeat(np.arange(1, 13), days)[:nd]


def _pulse_stats(mask):
    if not mask.any():
        return 0, 0.0
    e = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    durs = np.where(e == -1)[0] - np.where(e == 1)[0]
    return int(durs.size), float(durs.mean())


def _roll_ext(d, w, fn):
    if d.size < w:
        return np.nan
    return float(fn(np.convolve(d, np.ones(w) / w, 'valid')))


def _iha_year(d, lo, hi):
    """33 IHA params for one year of daily flow d. Returns dict or None."""
    d = d[np.isfinite(d)]
    if d.size < 300:
        return None
    p = {}
    mon = _month_of_day(d.size)
    for m in range(1, 13):
        sel = d[mon == m]; p[f'mag_month{m:02d}'] = float(sel.mean()) if sel.size else np.nan
    for w in _IHA_WINDOWS:
        p[f'min{w}day'] = _roll_ext(d, w, np.min)
        p[f'max{w}day'] = _roll_ext(d, w, np.max)
    p['zero_days'] = float(np.sum(d <= 0))
    am = d.mean()
    p['baseflow_index'] = (_roll_ext(d, 7, np.min) / am) if am > 0 else np.nan
    p['jd_max'] = float(np.argmax(d) + 1)
    p['jd_min'] = float(np.argmin(d) + 1)
    p['lo_pulse_count'], p['lo_pulse_dur'] = _pulse_stats(d < lo) if np.isfinite(lo) else (np.nan, np.nan)
    p['hi_pulse_count'], p['hi_pulse_dur'] = _pulse_stats(d > hi) if np.isfinite(hi) else (np.nan, np.nan)
    dd = np.diff(d)
    p['rise_rate'] = float(dd[dd > 0].mean()) if np.any(dd > 0) else 0.0
    p['fall_rate'] = float(dd[dd < 0].mean()) if np.any(dd < 0) else 0.0
    sg = np.sign(dd); sg = sg[sg != 0]
    p['reversals'] = float(np.sum(np.diff(sg) != 0))
    return p


def _circ_mean_day(vals, period=365.0):
    """Circular mean of day-of-year (issue: a linear median of timing dates that scatter around
    New Year gives ~mid-year, e.g. days {365,1} -> 183 instead of ~0/365). Day 1 -> angle 0."""
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    ang = (v - 1.0) / period * 2 * np.pi
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * period
    return (m % period) + 1.0


def iha_33(arr, hpp, hrs, lo, hi):
    yrs = [pp for pp in (_iha_year(d, lo, hi) for d in _daily_by_year(arr, hpp, hrs)) if pp]
    if not yrs:
        return {}
    # timing params (jd_max/jd_min) aggregated CIRCULARLY across years; all others by median.
    return {k: (_circ_mean_day([y[k] for y in yrs]) if k in _TIMING
                else float(np.nanmedian([y[k] for y in yrs]))) for k in yrs[0]}


def _pulse_thresholds(arr, hpp, hrs):
    pooled = np.concatenate(_daily_by_year(arr, hpp, hrs)) if _daily_by_year(arr, hpp, hrs) else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 30:
        return np.nan, np.nan
    return float(np.percentile(pooled, 25)), float(np.percentile(pooled, 75))


def _day_shift(a, b, period=365.0):
    s = (b - a) % period
    return s - period if s > period / 2 else s          # signed circular distance


def hai_full(nat, reg, hpp, hrs):
    """Returns (nat_params, reg_params, alterations, group_scores, overall)."""
    lo, hi = _pulse_thresholds(nat, hpp, hrs)
    a = iha_33(nat, hpp, hrs, lo, hi)
    b = iha_33(reg, hpp, hrs, lo, hi)
    # BOUNDED symmetric relative change in [-1,1] (issue: a near-zero natural baseline made the
    # old (b-a)/a explode to tens, so "HAI" was unbounded and NOT a 0-1 index). Using
    # (b-a)/max(|a|,|b|) keeps every param in [-1,1] -> group/overall scores are guaranteed 0-1.
    # This also covers the zero-baseline cases (nat=0,reg!=0 -> ±1; both 0 -> 0) without dropping
    # any param (which previously left Group 5 = NaN). NB this is NOT the Richter RVA risk index
    # (fraction of years outside the natural 25-75 band) - it is a bounded alteration magnitude.
    alts = {}
    for k in a:
        if not (np.isfinite(a[k]) and np.isfinite(b[k])):
            continue
        if k in _TIMING:
            alts[k] = abs(_day_shift(a[k], b[k])) / 182.5      # 0..1 fraction-of-year shift
        else:
            denom = max(abs(a[k]), abs(b[k]), 1e-9)
            alts[k] = (b[k] - a[k]) / denom                    # bounded to [-1, 1]
    groups = {}
    for g in range(1, 6):
        vals = [abs(alts[k]) for k in alts if _IHA_GROUP.get(k) == g]
        groups[g] = float(np.mean(vals)) if vals else np.nan
    overall = float(np.mean([abs(v) for v in alts.values()])) if alts else np.nan
    return a, b, alts, groups, overall


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_slope_table(hydro_dir):
    """HYRIV_ID → slope_m_per_m (real channel slope, RiverATLAS). {} if absent.

    Keyed by HYRIV_ID, NOT name (issue #2): slope is a property of the river REACH, so
    duplicate-named stations (two 'Funil') on different reaches get their own slope, and
    two stations genuinely on the same reach correctly share it."""
    path = os.path.join(hydro_dir, 'channel_slope.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if 'slope_m_per_m' not in df.columns or 'HYRIV_ID' not in df.columns:
        return {}
    return {int(h): float(s) for h, s in zip(df['HYRIV_ID'], df['slope_m_per_m'])
            if np.isfinite(h) and np.isfinite(s)}


def run(region_path=None, gcm=None, ssp=None, tag=None):
    cfg = C.resolve_config(region_path, gcm, ssp, tag)
    scenarios, meta = C.load_scenarios(cfg['output_dir'])
    static = C.load_static(cfg['hydro_dir'], meta['HPP_name'], cfg['output_dir'])
    slope_tbl = load_slope_table(cfg['hydro_dir'])
    hrs = meta['hrs_byyear']
    nat = scenarios['nat']
    disp = [k for k in scenarios if k != 'nat']

    def _hid(i):
        h = static.iloc[i].get('snap_HYRIV_ID')
        return int(h) if np.isfinite(h) else None
    n_slope = sum(1 for i in range(meta['n_HPP']) if _hid(i) in slope_tbl)
    print(f'[F2 ecology] {cfg["region_name"]} {cfg["gcm"]}/{cfg["ssp"]}'
          f'{("/" + cfg["tag"]) if cfg["tag"] else ""}  stations={meta["n_HPP"]}  scenarios={disp}')
    print(f'  VRR depth: {n_slope}/{meta["n_HPP"]} stations with real slope (Manning), rest global AHG')

    rows = []
    for hpp in range(meta['n_HPP']):
        srow = static.iloc[hpp].to_dict()
        lat = srow.get('lat_unified')
        length_km = srow.get('GRFR_lengthkm', np.nan)
        nat_pool = {None: _pool(nat, hpp, hrs)}
        for s in SEASONS:
            nat_pool[s] = _pool(nat, hpp, hrs, lat=lat, season=s)
        ann_mean_nat = np.mean(nat_pool[None]) if nat_pool[None].size else np.nan  # annual Q̄_nat

        for scen in disp:
            reg = scenarios[scen]
            # 4.1 Ecodeficit (total + seasonal) - ALL normalized by annual Q̄_nat (Vogel 2007),
            # so seasonal values are comparable across seasons.
            for s in [None] + SEASONS:
                rp = _pool(reg, hpp, hrs, lat=lat, season=s) if s else _pool(reg, hpp, hrs)
                def_, sur_ = ecodeficit(nat_pool[s], rp, mean_nat=ann_mean_nat)
                if s is not None and nat_pool[None].size:
                    # weight by the season's DURATION fraction (issue #7): the seasonal FDC
                    # integral was divided by the ANNUAL mean without the ~0.25 season-length
                    # factor, inflating seasonal Ecodeficit ~4x (some >1). The size ratio makes
                    # the 4 seasonal deficits sum coherently to the annual value.
                    frac = nat_pool[s].size / nat_pool[None].size
                    def_ = def_ * frac if np.isfinite(def_) else def_
                    sur_ = sur_ * frac if np.isfinite(sur_) else sur_
                tag_s = s or 'total'
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'Ecodeficit',
                                       f'season={tag_s}', value_nat=0.0, value_reg=def_,
                                       year_agg='pooled'))
                if s is None:
                    rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'Ecosurplus',
                                           'season=total', value_nat=0.0, value_reg=sur_,
                                           year_agg='pooled'))
            # 4.2 ramping (nat vs reg)
            rn, rr = ramping(nat, hpp, hrs), ramping(reg, hpp, hrs)
            for k in rr:
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, k, '',
                                       value_nat=rn[k], value_reg=rr[k]))
            # 4.3 cm/h VRR (Manning with real slope if available, else global AHG)
            dcoef, dexp, dnote = depth_model(slope_tbl.get(_hid(hpp), np.nan))
            vn = vrr_cmh(nat, hpp, hrs, dcoef, dexp)
            vr = vrr_cmh(reg, hpp, hrs, dcoef, dexp)
            for k in vr:
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, k, dnote,
                                       value_nat=vn[k], value_reg=vr[k]))
            # 4.4 varial zone
            dwn, an = varial_area(nat, hpp, hrs, length_km)
            dwr, ar = varial_area(reg, hpp, hrs, length_km)
            rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'varial_width_m',
                                   'hydraulic-geom approx (order-of-magnitude)', value_nat=dwn, value_reg=dwr))
            rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'varial_area_m2',
                                   'width×reach length (order-of-magnitude)', value_nat=an, value_reg=ar))
            # 4.5 Q1.67 flood-recurrence alteration (instantaneous annual max)
            # Risk = generic |alteration| ordinal bin. NOTE: NOT McManamay's threshold -
            # deep-lit review (verified) found NO published "1.67-yr flood >0.3 = high risk"
            # rule and NO transferable flow→biodiversity formula; only the qualitative
            # Poff & Zimmerman (2010) inference that risk rises with alteration magnitude.
            q167n = _q_recurrence(_annual_max(nat, hpp, hrs))
            q167r = _q_recurrence(_annual_max(reg, hpp, hrs))
            alt167 = (q167r - q167n) / q167n if (np.isfinite(q167n) and q167n) else np.nan
            ax = abs(alt167)
            risk = ('na' if not np.isfinite(alt167) else
                    'high' if ax > 0.5 else 'moderate' if ax > 0.25 else 'low')
            rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'Q1.67',
                                   f'risk={risk}(generic |alt| bin, not a calibrated threshold)',
                                   value_nat=q167n, value_reg=q167r))
            # 4.5 full IHA-33 (Richter 1996): 33 params + 5 group scores + overall HAI
            a, b, alts, groups, overall = hai_full(nat, reg, hpp, hrs)
            for k in sorted(alts):
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'IHA',
                                       f'param={k};group={_IHA_GROUP.get(k)}',
                                       value_nat=a[k], value_reg=b[k]))
            gnames = {1: 'magnitude_monthly', 2: 'extremes', 3: 'timing', 4: 'pulses', 5: 'rate_change'}
            for g, gs in groups.items():
                rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'HAI_group',
                                       f'group={g}_{gnames[g]}', value_nat=0.0, value_reg=gs))
            rows.append(C.make_row(cfg, srow, hpp, TOPIC, scen, 'HAI_mean_abs_alt',
                                   f'n_param={len(alts)}', value_nat=0.0, value_reg=overall))

    df, path = C.write_long_table(rows, TOPIC, cfg['output_dir'])
    print(f'  wrote {len(df)} rows → {path}')
    for m in ['Ecodeficit', 'HP1', 'VRR_exceed_h_13cmh', 'HAI_mean_abs_alt']:
        sub = df[(df.metric == m) & (df.scenario == 'BAL_C')]
        if m == 'Ecodeficit':
            sub = sub[sub.param == 'season=total']
        if not sub.empty:
            col = 'value_reg'
            print(f'    {m:20s} BAL_C mean = {sub[col].mean():.3f}')
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0] if len(a) > 0 else None, a[1] if len(a) > 1 else None,
        a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
