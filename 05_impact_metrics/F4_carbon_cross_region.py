"""
F4_carbon_cross_region.py - CROSS-REGION (E) data-center carbon, from E's OWN allocation+loss output.

Each cross-region flow (a source region exports firm hydro to a DC in another region) is read
DIRECTLY from E's interconnected allocation `bloc_allocation_*E1*.csv`, which records the REAL
sent_mw / delivered_mw / loss_mw / hops / path that E computed from its transmission model
(ac_loss_rate_per_corridor=0.03, hvdc=0.02 × hops; transmission_model_config.csv). We do NOT
assume a flat loss - E already computed it (2.3-9.9% by route).

Carbon of a cross-region flow:
  source GENERATES sent_mw of hydro → SOURCE-region reservoir EF (the loss is extra source hydro
  generation, so it is charged at the source hydro EF);
  the DC RECEIVES delivered_mw; its carbon INTENSITY of received power = sent·src_EF / delivered.
  Counterfactual (avoided): the DC served delivered_mw from its OWN grid → delivered·demand_grid_EF.

Needs only ENERGY + EFs - not source hydro FLOWS - so it runs from E's energy/allocation output even
though F1/F2/F3 cross-region (which need the regulated source FLOWS - NOT saved by this E run, only
P_target_mw) cannot run here without re-running E with hourly flow output.

Enumerates EVERY E1 cross-region flow (continent × config × bloc × year × scale × scenario A/B/C ×
src → dc) → one long table. Region EFs from each region's hydro/emission_factors.csv (run
F4_prepare_emission_factors.py first); brazil_norte/bolivia have no single-region run ON THIS SERVER,
so their source EF is a flagged areal×area/(capacity·CF) estimate - replace by syncing their
bal_yearly_summary and re-running F4_prepare.

Output: /datasets/swat_global/_carbon_ef/impact_carbon_cross_region.csv
CLI:  python F4_carbon_cross_region.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd

E_ROOT = os.environ.get('REVUB_E_OUTPUT', '/home/cfeng/myswat/REVUB/1_REVUB_code/E_output')
CARBON = '/datasets/swat_global/_carbon_ef'
HOURS_YR = 8760.0
GRES_MAX = 12000.0
CF_DEFAULT = 0.5
REGION_COUNTRY = {'singapore': 'Singapore', 'bolivia': 'Bolivia', 'brazil_norte': 'Brazil'}
DEMAND_GRID_DEFAULT = {'singapore': 417.0}   # Singapore EMA grid factor ~0.4168 tCO2/MWh (gas)


def _load_gres():
    """(name, round(lat,3)) -> gres_ef (per-kWh) and -> (areal, area) for the cap-based fallback."""
    g = pd.read_csv(os.path.join(CARBON, 'reservoir_ef_gres.csv'), low_memory=False)
    ef, areal = {}, {}
    for n, la, e, ar, a in zip(g['name'], g['lat'], g['gres_ef_gco2kwh'],
                               g['areal_total_gco2e_m2yr'], g['area_km2']):
        if not np.isfinite(la):
            continue
        k = (str(n), round(float(la), 3))
        if np.isfinite(e):
            ef[k] = e
        if np.isfinite(ar) and np.isfinite(a):
            areal[k] = (float(ar), float(a))
    return ef, areal


def _src_cap(src_region):
    try:
        st = pd.read_csv(os.path.join('/datasets/swat_global', *_REGDIR.get(src_region, (src_region,)),
                                      'hydro', 'station_channel_result.csv'), low_memory=False)
        return {str(n): float(c) for n, c in zip(st['name_unified'], st['capacity_mw']) if np.isfinite(c)}
    except Exception:
        return {}


def _pairing_src_ef(gres_ef, gres_areal):
    """Map (prefix, src) -> (src hydro EF, status), from the EXPORTING stations of each redispatch
    pairing (dc_hydro_allocation + region_station_profile + the pairing's own dispatch npz for REAL
    generation). The source's export stations are the same regardless of which DC, so we key by
    (prefix, src). Aggregated allocated-MW-weighted across hashes."""
    acc = {}   # (prefix,src) -> [num, den, status]
    for pdir in glob.glob(os.path.join(E_ROOT, '**', 'redispatch_cache', '*', '*', ''), recursive=True):
        parts = pdir.rstrip('/').split('/')
        try:
            rc = parts.index('redispatch_cache')
        except ValueError:
            continue
        prefix = '/'.join(parts[:rc]); src = parts[rc + 1]
        try:
            prof = pd.read_csv(os.path.join(pdir, 'region_station_profile.csv'))
            alloc = pd.read_csv(os.path.join(pdir, 'dc_hydro_allocation.csv'))
        except Exception:
            continue
        if 'hpp_idx' not in alloc.columns or 'name' not in prof.columns:
            continue
        names = list(prof['name'].astype(str)); lats = list(prof['lat'])
        cap = _src_cap(src)
        # REAL per-station annual generation from this pairing's own dispatch npz (P_s+P_f+P_r),
        # aligned to the region_station_profile / npz HPP order. Lets us compute a REAL per-kWh EF
        # (areal·area/gen) for regions with no single-region F4 run - no capacity estimate needed.
        gen = None
        try:
            z = np.load(os.path.join(pdir, 'dc_scenario_A_hourly.npz'), allow_pickle=True)
            if {'P_s_A', 'P_f_A', 'P_r_A'}.issubset(set(z.files)):
                P = np.asarray(z['P_s_A']) + np.asarray(z['P_f_A']) + np.asarray(z['P_r_A'])
                gen = P.mean(axis=(0, 1)) * HOURS_YR / 1e3      # GWh/yr per station (npz order)
        except Exception:
            gen = None
        num = den = 0.0
        wmeth = {'gres': 0.0, 'gres-npz-gen': 0.0, 'cap-based-gen-estimate': 0.0}
        for _, r in alloc.iterrows():
            i = int(r['hpp_idx'])
            if not (0 <= i < len(names)):
                continue
            key = (names[i], round(float(lats[i]), 3)); w = float(r.get('allocated_mw', 0.0) or 0.0)
            ef = gres_ef.get(key); meth = 'gres'
            if ef is None or not np.isfinite(ef):
                ar = gres_areal.get(key)
                if ar and gen is not None and 0 <= i < len(gen) and gen[i] > 0:
                    ef = ar[0] * ar[1] / gen[i]                 # areal[g/m2/yr]·area[km2]/gen[GWh] = gCO2/kWh
                    meth = 'gres-npz-gen'
                elif ar and cap.get(names[i], 0) > 0:
                    ef = ar[0] * ar[1] * 1e6 / (cap[names[i]] * CF_DEFAULT * HOURS_YR * 1e3)
                    meth = 'cap-based-gen-estimate'
                else:
                    continue
            num += min(GRES_MAX, ef) * w; den += w; wmeth[meth] += w
        if den <= 0:
            continue
        a = acc.setdefault((prefix, src), [0.0, 0.0,
                                           {'gres': 0.0, 'gres-npz-gen': 0.0, 'cap-based-gen-estimate': 0.0}])
        a[0] += num; a[1] += den
        for m, wm in wmeth.items():
            a[2][m] += wm
    # status = the method that carries the most allocated-MW weight (not the worst-case label)
    return {k: (v[0] / v[1], max(v[2], key=v[2].get)) for k, v in acc.items() if v[1] > 0}


# region basename -> path components under /datasets/swat_global (for capacity lookup)
_REGDIR = {}
for _ef in glob.glob('/datasets/swat_global/**/hydro/station_channel_result.csv', recursive=True):
    _rel = _ef.split('/datasets/swat_global/')[1].split('/hydro/')[0]
    _REGDIR[os.path.basename(_rel)] = tuple(_rel.split('/'))


def _region_grid_ef():
    out = {}
    for ef in glob.glob('/datasets/swat_global/**/hydro/emission_factors.csv', recursive=True):
        try:
            out[os.path.basename(ef.split('/hydro/')[0])] = float(pd.read_csv(ef).iloc[0]['grid_nonhydro_ef'])
        except Exception:
            pass
    return out


def _grid_ef(dc_reg, gmap, ember):
    if dc_reg in gmap and np.isfinite(gmap[dc_reg]):
        return gmap[dc_reg]
    cc = REGION_COUNTRY.get(dc_reg, dc_reg.split('_')[0].title())
    e = ember[ember['country'] == cc]
    if not e.empty and pd.notna(e['nonhydro_gco2kwh'].iloc[0]):
        return float(e['nonhydro_gco2kwh'].iloc[0])
    return DEMAND_GRID_DEFAULT.get(dc_reg, np.nan)


def run():
    gres_ef, gres_areal = _load_gres()
    src_ef_map = _pairing_src_ef(gres_ef, gres_areal)   # (prefix,src,dc) -> (ef, status) from exporting stations
    gmap = _region_grid_ef()
    ember = pd.read_csv(os.path.join(CARBON, 'ember_country_ef.csv'))
    # Per directory, the cross-region E1 allocation exists as BOTH a combined `bloc_allocation_E1.csv`
    # (scenario col = A/B/C) AND per-scenario `bloc_allocation_{A,B,C}_E1.csv`. Reading both
    # DOUBLE-COUNTS (verified vs E's cross_region_rerun_requirements: was 2×). Use the combined file
    # if present, else the per-scenario set.
    by_dir = {}
    for f in glob.glob(os.path.join(E_ROOT, '**', 'bloc_allocation*E1*.csv'), recursive=True):
        by_dir.setdefault(os.path.dirname(f), []).append(f)
    files = []
    for dd, fs in by_dir.items():
        combined = os.path.join(dd, 'bloc_allocation_E1.csv')
        files += [combined] if combined in fs else fs
    rows = []
    for f in sorted(files):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        need = {'src_region', 'dc_region', 'sent_mw', 'delivered_mw', 'loss_mw', 'cross_region'}
        if not need.issubset(d.columns):
            continue
        cr = d[d['cross_region'] == 1]
        if cr.empty:
            continue
        parts = f.split('/')
        eo = next((i for i,p in enumerate(parts) if p.startswith('E_output')), 0)
        cont = parts[eo + 1] if eo + 1 < len(parts) else '?'
        bloc = next((p for p in parts if p.startswith('virtual_')), 'self')
        # the leaf config dir is e.g. 'year_2033_scale_1' or 'year_2033_scale_1__alllines5ties' -
        # parse year/scale/config out of it (the old startswith scan left scale='?' and overlapped).
        leaf = next((p for p in parts if p.startswith('year_')), '')
        mm = re.match(r'year_(\d+)_scale_(\d+)(?:_+(.+))?$', leaf)
        year = f'year_{mm.group(1)}' if mm else '?'
        scale = f'scale_{mm.group(2)}' if mm else '?'
        cfg = (mm.group(3) or '') if mm else ''
        prefix = '/'.join(parts[:parts.index(os.path.basename(f))])  # dir holding this bloc_allocation
        for _, r in cr.iterrows():
            src = str(r['src_region']); dc = str(r['dc_region'])
            sent = float(r['sent_mw']); deliv = float(r['delivered_mw']); loss = float(r['loss_mw'])
            if not (np.isfinite(sent) and sent > 0 and np.isfinite(deliv) and deliv > 0):
                continue
            scen = str(r.get('scenario', '?'))
            src_ef, src_status = src_ef_map.get((prefix, src), (np.nan, "missing"))
            gef = _grid_ef(dc, gmap, ember)
            E_gen = sent * HOURS_YR                              # MWh hydro generated at source
            E_deliver = deliv * HOURS_YR                         # MWh received by the DC
            hyd_c = E_gen * src_ef / 1000.0 if np.isfinite(src_ef) else np.nan   # tCO2 (loss = extra source hydro)
            intensity = hyd_c * 1000.0 / E_deliver if (np.isfinite(hyd_c) and E_deliver) else np.nan
            allgrid = E_deliver * gef / 1000.0 if np.isfinite(gef) else np.nan   # DC from its own grid
            net_avoid = (allgrid - hyd_c) if (np.isfinite(allgrid) and np.isfinite(hyd_c)) else np.nan
            rows.append(dict(
                continent=cont, config=cfg, bloc=bloc, year=year, scale=scale, scenario=scen,
                src_region=src, dc_region=dc, path=str(r.get('path', '')), hops=int(r.get('hops', 0)),
                sent_mw=round(sent, 1), delivered_mw=round(deliv, 1), loss_mw=round(loss, 1),
                loss_frac=round(loss / sent, 4) if sent else np.nan,
                src_hydro_ef=round(src_ef, 1) if np.isfinite(src_ef) else np.nan, src_ef_status=src_status,
                demand_grid_ef=round(gef, 1) if np.isfinite(gef) else np.nan,
                hydro_carbon_tCO2_yr=round(hyd_c) if np.isfinite(hyd_c) else np.nan,
                dc_intensity_gCO2_kWh=round(intensity, 1) if np.isfinite(intensity) else np.nan,
                net_avoided_tCO2_yr=round(net_avoid) if np.isfinite(net_avoid) else np.nan))
    df = pd.DataFrame(rows)
    out = os.path.join(CARBON, 'impact_carbon_cross_region.csv')
    df.to_csv(out, index=False)
    print(f'wrote {len(df)} cross-region flows -> {out}')
    if len(df):
        print(f'  loss from E (real): {df["loss_frac"].min():.3f}..{df["loss_frac"].max():.3f} '
              f'(mean {df["loss_frac"].mean():.3f}) | src_ef_status: {df["src_ef_status"].value_counts().to_dict()}')
        b = df[df.scenario == 'A'].groupby(['src_region', 'dc_region']).agg(
            sent=('sent_mw', 'max'), loss=('loss_frac', 'mean'),
            inten=('dc_intensity_gCO2_kWh', 'first'), net=('net_avoided_tCO2_yr', 'max')).reset_index().sort_values('inten')
        for _, r in b.iterrows():
            print(f"    {r['src_region']:>14s}->{r['dc_region']:<16s} sent={r['sent']:7.0f}MW loss={r['loss']:.3f} "
                  f"int={r['inten']:6.1f} net_avoid={r['net']:.0f}")
    return df


if __name__ == '__main__':
    run()
