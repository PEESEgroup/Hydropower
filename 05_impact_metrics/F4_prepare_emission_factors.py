"""
F4_prepare_emission_factors.py - REAL emission factors for the DC carbon metric (F4).
See REVUB_F4_carbon_emissions_plan.md / REVUB_data_sources_and_calculation_verification.md.

Rule: finest available, fall back to coarser.
  ① Hydro reservoir EF (gCO2eq/kWh) - PER-RESERVOIR, from the G-res model
       (F4_gres_model.py → reservoir_ef_gres.csv): the full G-res 4-pathway flux
       (CO2 diffusion + CH4 diffusion/ebullition/degassing, GWP100=34, 100-yr lifetime,
       % littoral from per-reservoir depth, climate from WorldClim, soil C from SoilGrids,
       soil P from Yang 2013) ÷ generation. The DC's hydro EF = generation-weighted mean
       over the stations actually serving the DC (DC-matched). Falls back to the prior
       zone-median-flux estimate only if the G-res table is missing.
  ② Grid backup EF (gCO2eq/kWh) - OFFICIAL operating-margin / non-hydro marginal where
       published (F4_grid_factors.resolve_grid: US eGRID subregion, China MEE regional grid,
       AR/CO/IN official OM), else Ember non-hydro/marginal per country. NB: the grid factor
       backs up a hydro SHORTFALL, so it is the marginal/non-hydro generator, not the
       hydro-diluted grid average.

Output: <hydro_dir>/emission_factors.csv (region row) + reservoir_ef.csv (per station)

CLI:  python F4_prepare_emission_factors.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

import F4_grid_factors as GF

CARBON = '/datasets/swat_global/_carbon_ef'
# low/high EF band multipliers, both literature-anchored:
#   high 2.5 = AR5 biogenic CH4 GWP20/GWP100 ratio (86/34=2.53; IPCC AR5 Table 8.7) -> a
#             time-horizon sensitivity (using GWP20 instead of GWP100).
#   low 0.6  = lower edge of the published global reservoir-GHG envelope (Deemer et al. 2016
#             BioScience 66:949: 0.8 [0.5-1.2] Pg CO2e/yr -> ~0.6-1.5x central).
GWP_BAND = (0.6, 2.5)
# EF caps (issue #10). G-res mechanistic rows are geometry-vetted (geom_implausible already
# excluded) so they pass up to GRES_MAX - a Balbina-class literature ceiling (~10,600 gCO2/kWh,
# Kemenes et al. 2011 JGR 116:G03004; Scherer & Pfister 2016 saw up to 167,129) - which only
# stops pathological 1e5-class division artifacts, NOT genuine low-power-density reservoirs.
# The coarse zone-flux FALLBACK (artifact-prone areal-flux estimate) keeps the tighter 3000 cap.
GRES_MAX = 12000.0
EF_CAP = 3000.0

# ── per-reservoir G-res EF table (preferred) ──
_GRES_PATH = os.path.join(CARBON, 'reservoir_ef_gres.csv')

# Fallback only (if G-res table missing): climate-zone MEDIAN areal flux (Deemer/Almeida).
ZONE_FLUX = {'tropical': 1500, 'subtropical': 800, 'temperate': 400, 'boreal': 200}


def _zone(lat):
    a = abs(lat)
    return ('tropical' if a < 23.5 else 'subtropical' if a < 35 else
            'temperate' if a < 55 else 'boreal')

# region basename → (Ember country, grid tier note). Subnational grids (China province,
# US eGRID subregion, India CEA) are resolved by passing the subnational key below into
# GF.resolve_grid; for those the Ember country still names the nation (China / United States
# of America / India) so resolve_grid takes its subnational/official branch.
REGION_COUNTRY = {
    # ── Latin America (kept) ──
    'brazil_norte': ('Brazil', 'country-proxy(ONS-N upgrade)'),
    'brazil_nordeste': ('Brazil', 'country-proxy(ONS-NE upgrade)'),
    'brazil_sudeste': ('Brazil', 'country-proxy(ONS-SE/CW upgrade)'),
    'brazil_sul': ('Brazil', 'country-proxy(ONS-S upgrade)'),
    'argentina': ('Argentina', 'country'), 'chile': ('Chile', 'country'),
    'colombia': ('Colombia', 'country'), 'ecuador': ('Ecuador', 'country'),
    'peru': ('Peru', 'country'), 'bolivia': ('Bolivia', 'country'),
    # ── Malaysia + Baltic (kept) ──
    'malaysia_west': ('Malaysia', 'country'), 'malaysia_east': ('Malaysia', 'country'),
    'lithuania': ('Lithuania', 'country'), 'latvia': ('Latvia', 'country'),

    # ── China: regional grid via _CN_PROV2GRID (province passed via REGION_CN_PROV) ──
    'china_nc': ('China', 'subnational(North-grid OM)'),
    'china_ne': ('China', 'subnational(Northeast-grid OM)'),
    'china_ec': ('China', 'subnational(East-grid OM)'),
    'china_cc': ('China', 'subnational(Central-grid OM)'),
    'china_nw': ('China', 'subnational(Northwest-grid OM)'),
    'china_csg': ('China', 'subnational(South-grid OM)'),
    'china_sw': ('China', 'subnational(Southwest-grid OM)'),

    # ── USA: eGRID subregion non-baseload (acronym passed via REGION_US_EGRID) ──
    'usa_caiso': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_ercot': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_isone': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_nyiso': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_pjm_east': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_pjm_west': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_miso_north': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_miso_central': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_miso_south': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_spp_north': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_spp_south': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_sertp': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_northerngrid_west': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_northerngrid_east': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_northerngrid_south': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_westconnect_north': ('United States of America', 'subnational(eGRID non-baseload)'),
    'usa_westconnect_south': ('United States of America', 'subnational(eGRID non-baseload)'),

    # ── India: CEA national combined-margin (official-country) ──
    'india_north': ('India', 'official-country(CEA CM)'),
    'india_east': ('India', 'official-country(CEA CM)'),
    'india_northeast': ('India', 'official-country(CEA CM)'),
    'india_south': ('India', 'official-country(CEA CM)'),
    'india_west': ('India', 'official-country(CEA CM)'),

    # ── EU (Ember country) ──
    'austria': ('Austria', 'country'), 'belgium': ('Belgium', 'country'),
    'croatia': ('Croatia', 'country'), 'finland': ('Finland', 'country'),
    'france': ('France', 'country'), 'germany': ('Germany', 'country'),
    'greece': ('Greece', 'country'), 'ireland': ('Ireland', 'country'),
    'italy': ('Italy', 'country'), 'luxembourg': ('Luxembourg', 'country'),
    'norway': ('Norway', 'country'), 'portugal': ('Portugal', 'country'),
    'slovakia': ('Slovakia', 'country'), 'slovenia': ('Slovenia', 'country'),
    'spain': ('Spain', 'country'), 'sweden': ('Sweden', 'country'),
    'switzerland': ('Switzerland', 'country'),
    'united_kingdom': ('United Kingdom', 'country'),

    # ── Canada (Ember country; provincial detail not in Ember table) ──
    'canada_atlantic': ('Canada', 'country'), 'canada_bc': ('Canada', 'country'),
    'canada_ontario': ('Canada', 'country'), 'canada_prairies': ('Canada', 'country'),
    'canada_quebec': ('Canada', 'country'),

    # ── Turkey + SE Asia (Ember country; watch exact Ember spellings) ──
    'turkey': ('Turkey', 'country'), 'thailand': ('Thailand', 'country'),
    'vietnam': ('Viet Nam', 'country'), 'indonesia': ('Indonesia', 'country'),
    'philippines': ('Philippines (the)', 'country'), 'cambodia': ('Cambodia', 'country'),
    'laos': ("Lao People's Democratic Republic (the)", 'country'),
    'myanmar': ('Myanmar', 'country'),

    # ── Balkans / Central Europe (Ember country) ──
    'albania': ('Albania', 'country'), 'north_macedonia': ('North Macedonia', 'country'),
    'serbia': ('Serbia', 'country'), 'bulgaria': ('Bulgaria', 'country'),
    'romania': ('Romania', 'country'), 'hungary': ('Hungary', 'country'),
    'poland': ('Poland', 'country'),
}

# region basename -> a representative Chinese province in the target MEE regional grid
# (the province is only used to hit GF._CN_PROV2GRID, which resolves to the regional OM).
# NOTE: the province values below are intentionally kept in Chinese -- they are functional
# lookup keys that must match the 'provinces' field of the official China MEE grid-emission-
# factor dataset; romanizing them would break the China subnational lookup. The pinyin in
# each comment is for readability only.
REGION_CN_PROV = {
    'china_nc': '河北',   # Hebei     - North grid
    'china_ne': '辽宁',   # Liaoning  - Northeast grid
    'china_ec': '江苏',   # Jiangsu   - East grid
    'china_cc': '河南',   # Henan     - Central grid
    'china_nw': '陕西',   # Shaanxi   - Northwest grid
    'china_csg': '广东',  # Guangdong - South grid (China Southern Grid)
    'china_sw': '四川',   # Sichuan   - Southwest grid
}

# region basename → US eGRID subregion acronym (non-baseload = avoided-emissions marginal).
# NOTE: ISO/RTO territories do NOT map 1:1 to eGRID NERC subregions; this is an approximation.
REGION_US_EGRID = {
    'usa_caiso': 'CAMX', 'usa_ercot': 'ERCT', 'usa_isone': 'NEWE', 'usa_nyiso': 'NYUP',
    'usa_pjm_east': 'RFCE', 'usa_pjm_west': 'RFCW', 'usa_miso_north': 'MROW',
    'usa_miso_central': 'SRMW', 'usa_miso_south': 'SRMV', 'usa_spp_north': 'SPNO',
    'usa_spp_south': 'SPSO', 'usa_sertp': 'SRSO', 'usa_northerngrid_west': 'NWPP',
    'usa_northerngrid_east': 'NWPP', 'usa_northerngrid_south': 'RMPA',
    'usa_westconnect_north': 'RMPA', 'usa_westconnect_south': 'AZNM',
}


def _load_gres():
    """Return (ef_dict, geom_bad_set), both keyed by the composite (name, latkey).
    `name` is NOT unique (two 'Funil', 'Miranda' Spain/Brazil, ...), so a name-only dict
    would silently keep only the last duplicate's EF (audit issue #5). Key on
    (name, round(lat,3)) - unique across the dataset. geom_bad reservoirs (mean depth
    < 1.5 m -> corrupted area) get EF=NaN with NO zone-flux fallback (would reuse bad area)."""
    if os.path.exists(_GRES_PATH):
        g = pd.read_csv(_GRES_PATH, low_memory=False)
        g['_k'] = list(zip(g['name'].astype(str), g['lat'].round(3)))
        ef = g.set_index('_k')['gres_ef_gco2kwh'].to_dict()
        bad = set(g.loc[g.get('geom_implausible', False) == True, '_k']) \
            if 'geom_implausible' in g.columns else set()
        return ef, bad
    return None, set()


def _matched_stations(out_dir):
    """(matched_names, weight_dict) for the stations that actually serve the DC.

    Uses D's AUTHORITATIVE per-DC allocation `dc_hydro_allocation.csv` (hpp_idx +
    allocated_mw), mapped through the npz `HPP_name` order - NOT the old
    Q_out_A!=Q_BAL_out heuristic, which mis-classified cascade-downstream stations as
    suppliers and missed unchanged-outflow suppliers (audit issue #4: brazil_sudeste
    +21/-6; peru/ecuador heuristic n=0 -> fell back to ALL). weight_dict[name] =
    summed allocated MW = energy actually delivered to the DC (for issue #6 weighting).
    Falls back to [] if the allocation file/npz is missing.
    """
    try:
        bal = np.load(os.path.join(out_dir, 'bal_hourly_arrays.npz'), allow_pickle=True)
        names = [str(n) for n in bal['HPP_name']]
        alloc = pd.read_csv(os.path.join(out_dir, 'dc_hydro_allocation.csv'))
    except Exception:
        return [], {}
    if 'hpp_idx' not in alloc.columns or alloc.empty:
        return [], {}
    w = {}
    for _, r in alloc.iterrows():
        i = int(r['hpp_idx'])
        if 0 <= i < len(names):
            w[names[i]] = w.get(names[i], 0.0) + float(r.get('allocated_mw', 0.0) or 0.0)
    return list(w.keys()), w


def process_region(region_path, region, ember, gres, geom_bad):
    hydro_dir = os.path.join(region_path, 'hydro')
    csv = os.path.join(hydro_dir, 'station_channel_result.csv')
    if not os.path.exists(csv):
        return None
    st = pd.read_csv(csv, low_memory=False)
    # generation per station (mean annual GWh)
    gen = {}
    for tag in ['', 'flat', 'typical']:
        p = os.path.join(region_path, 'revub_output', 'mri-esm2-0', 'ssp370', tag, 'bal_yearly_summary.parquet')
        if os.path.exists(p):
            g = pd.read_parquet(p); gen = g.groupby('station')['E_total_GWh'].mean().to_dict(); out_dir = os.path.dirname(p); break
    else:
        return None
    # per-reservoir EF - prefer G-res table (keyed by composite (name,lat)), fall back to zone-flux.
    # NB gen is name-keyed (bal_yearly_summary carries no position); duplicate-named stations
    # (two 'Funil') therefore share averaged generation - an upstream limitation, documented.
    recs = []
    for _, r in st.iterrows():
        n = str(r['name_unified']); lat = r.get('lat_unified'); area = r.get('max_area_km2')
        key = (n, round(float(lat), 3)) if np.isfinite(lat) else (n, np.nan)
        g = gen.get(n, np.nan)
        ef = np.nan; zn = ''; src = ''
        if key in geom_bad:
            src = 'geom_implausible'          # corrupted area/volume -> no EF, NO zone-flux fallback
        elif gres is not None and key in gres and np.isfinite(gres[key]):
            ef = min(GRES_MAX, float(gres[key])); src = 'gres'   # vetted -> Balbina-class ceiling only
            zn = _zone(lat) if np.isfinite(lat) else ''
        elif np.isfinite(lat) and np.isfinite(area) and np.isfinite(g) and g > 0:
            zn = _zone(lat)
            ef = min(EF_CAP, ZONE_FLUX[zn] * float(area) / float(g)); src = 'zone-flux'
        recs.append({'name_unified': n, 'lat': lat, 'zone': zn, 'area_km2': area,
                     'gen_GWh': g, 'ef_source': src,
                     'hydro_ef_gco2kwh': round(ef, 1) if np.isfinite(ef) else ef})
    rdf = pd.DataFrame(recs)
    rdf.to_csv(os.path.join(hydro_dir, 'reservoir_ef.csv'), index=False)

    # DC's hydro EF = weighted over the DC-supplying stations. Weight by ENERGY ACTUALLY
    # DELIVERED TO THE DC (allocated MW from D's dc_hydro_allocation.csv), not each station's
    # total generation (issue #6). If no allocation -> all stations, total-generation weighted.
    matched, wdict = _matched_stations(out_dir)
    sel = pd.DataFrame(); wt = ''
    if matched:
        sel = rdf[rdf['name_unified'].isin(matched) & np.isfinite(rdf['hydro_ef_gco2kwh'])].copy()
        if not sel.empty:
            sel['w'] = sel['name_unified'].map(wdict).fillna(0.0)
            if sel['w'].sum() <= 0:              # allocated MW unavailable -> total generation
                sel['w'] = sel['gen_GWh']
            wt = 'DC-allocated-MW'
    if sel.empty:                                # no DC alloc, or matched lack a valid EF -> all stations
        sel = rdf[np.isfinite(rdf['hydro_ef_gco2kwh']) & (rdf['gen_GWh'] > 0)].copy()
        sel['w'] = sel['gen_GWh']; wt = 'all-stations gen-wt(fallback)'
    sel = sel[np.isfinite(sel['w']) & (sel['w'] > 0)]
    hyd_ef = float(np.average(sel['hydro_ef_gco2kwh'], weights=sel['w'])) if not sel.empty else np.nan
    hyd_src = f'gres per-reservoir, {wt}'

    # grid backup EF - official OM/marginal where available, else Ember (F4_grid_factors).
    # Pass the subnational key so resolve_grid reaches its US eGRID / China regional branch.
    cc, tier_note = REGION_COUNTRY.get(region, (None, 'unmapped'))
    province = REGION_CN_PROV.get(region)
    egrid_sub = REGION_US_EGRID.get(region)
    gf = GF.resolve_grid(region, cc, ember, province=province, egrid_subregion=egrid_sub)
    grid_nh = gf['central']; grid_mg = gf['upper']; yr = gf['year']
    if not np.isfinite(grid_nh):                  # issue #14: never emit NaN grid EF silently
        grid_nh, grid_mg, yr = 650.0, 900.0, 0    # global non-hydro / fossil default (Ember world ~)
        gf = dict(gf, tier='global-default',
                  source=f'UNMAPPED region (cc={cc}) - global non-hydro default ~650')
        print(f'  WARNING: {region} unmapped grid (cc={cc}); using global default 650/900 gCO2/kWh')

    out = pd.DataFrame([{
        'region': region, 'n_matched': len(matched),
        'hydro_ef_central': round(hyd_ef, 1),
        'hydro_ef_low': round(hyd_ef * GWP_BAND[0], 1), 'hydro_ef_high': round(hyd_ef * GWP_BAND[1], 1),
        'hydro_source': hyd_src,
        'grid_nonhydro_ef': round(grid_nh, 1) if np.isfinite(grid_nh) else grid_nh,
        'grid_marginal_ef': round(grid_mg, 1) if np.isfinite(grid_mg) else grid_mg,
        'grid_country': cc, 'grid_year': yr, 'grid_tier': gf['tier'], 'grid_source': gf['source'],
    }])
    out.to_csv(os.path.join(hydro_dir, 'emission_factors.csv'), index=False)
    return out.iloc[0].to_dict()


def main():
    ember = pd.read_csv(os.path.join(CARBON, 'ember_country_ef.csv'))
    gres, geom_bad = _load_gres()
    print(f"[F4_prepare] G-res per-reservoir table: {'LOADED '+str(len(gres))+' reservoirs, '+str(len(geom_bad))+' geom-implausible excluded' if gres else 'MISSING → zone-flux fallback'}")
    paths = sorted(glob.glob('/datasets/swat_global/**/bal_hourly_arrays.npz', recursive=True)) \
        + sorted(glob.glob('/tmp/revub_integration/**/bal_hourly_arrays.npz', recursive=True))
    for r in sorted({p.split('/revub_output')[0] for p in paths}):
        region = os.path.basename(r)
        res = process_region(r, region, ember, gres, geom_bad)
        if res:
            print(f"  {region:16s} hydroEF={res['hydro_ef_central']:6.0f} (DC-matched n={res['n_matched']}) | "
                  f"gridBackup={res['grid_nonhydro_ef']} marg={res['grid_marginal_ef']} "
                  f"[{res['grid_country']} {res['grid_year']} {res['grid_tier']}]")


if __name__ == '__main__':
    main()
