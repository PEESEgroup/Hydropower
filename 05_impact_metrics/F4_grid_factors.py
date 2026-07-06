"""
F4_grid_factors.py - official grid CO2 factor resolver (fine → coarse fallback).

For F4 the grid factor backs up a HYDRO SHORTFALL, so the physically-correct number is the
OPERATING-MARGIN / non-hydro marginal generator - NOT the (hydro-diluted) grid average.
E.g. Brazil's official SIN average ~39 gCO2/kWh is all-grid and meaningless as a backup factor;
the relevant Brazilian backup is its thermal margin. So the resolver prefers, in order:

  central backup EF:  official OPERATING-MARGIN (OM) / eGRID non-baseload  >  Ember non-hydro  >  Ember marginal
  upper  backup EF:   official dispatch-marginal / combined-margin / fossil  >  Ember fossil marginal

Subnational resolution (USA eGRID 26 subregions, China 7 regional grids) is wired and activates
when a US/China reservoir carries a state / province / subregion tag; otherwise national fallback.

Files under /datasets/swat_global/_carbon_ef/grid_official/ (downloaded from official sources):
  egrid_subregions.csv, china_mee_grid.csv, india_cea.csv, brazil_mcti_ons.csv, latam_official.csv
Ember fallback: /datasets/swat_global/_carbon_ef/ember_country_ef.csv
"""
import os
import numpy as np
import pandas as pd

CARBON = '/datasets/swat_global/_carbon_ef'
GO = os.path.join(CARBON, 'grid_official')

# ── Official COUNTRY-level backup factors (central=OM/marginal, upper=dispatch/CM/fossil) ──
# Only countries where an OFFICIAL marginal/OM factor was actually retrieved. Grid-AVERAGE-only
# countries (Chile 238 avg, Ecuador 162 avg, Brazil 39 avg) are intentionally NOT used as backup -
# their average is hydro-diluted; we keep the (more appropriate) Ember non-hydro/marginal and record
# the official average only as a reference note.
OFFICIAL_COUNTRY = {
    # country : (central_OM, upper_marginal, year, source)
    # Convention: central = OPERATING MARGIN (the generator that actually responds when
    # hydro is short); upper = pure dispatch/fossil marginal. India kept on the same
    # convention (central = OM 961, not the BM-diluted combined margin) for consistency
    # with AR/CO; CM 736 would understate India's backup vs the others.
    'Argentina': (429.3, 580.3, 2023, 'AR SecEnergia OM/dispatch-marginal'),
    'Colombia':  (607.0, 660.0, 2024, 'CO UPME OM/combined-margin'),
    # India: now resolved from india_cea.csv (combined-margin central, OM upper) via
    # _load_india()/the India branch below. This literal entry is a hardcoded fallback only
    # (used if india_cea.csv is missing) and matches the CSV: CM 736 central, OM 961 upper.
    'India':     (736.0, 961.0, 2025, 'India CEA 2024-25 CM (fallback literal)'),
}
# Official grid-AVERAGE-only (kept as reference note, NOT used as backup central)
OFFICIAL_AVG_NOTE = {
    'Chile':   (238.4, 2023, 'CL Coordinador grid-avg(ref)'),
    'Ecuador': (161.6, 2024, 'EC ARCONEL SNI grid-avg(ref)'),
    'Brazil':  (38.5,  2023, 'BR MCTI SIN grid-avg(ref, hydro-diluted)'),
}

# China province → regional grid (2023 boundaries; for subnational resolution)
_CN_PROV2GRID = {}
# USA state → eGRID subregion is many-to-many; handled via explicit region tag instead.


def _load_china():
    p = os.path.join(GO, 'china_mee_grid.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    om = df[(df['type'] == 'OM') & (df['year'] == 2023)]
    for _, r in om.iterrows():
        for prov in str(r['provinces']).split('|'):
            _CN_PROV2GRID[prov] = (float(r['factor_gco2kwh']), r['grid_region_en'])
    # build margin (upper) per region too
    bm = df[(df['type'] == 'BM') & (df['year'] == 2023)].set_index('grid_region_en')['factor_gco2kwh'].to_dict()
    return {'OM': om.set_index('grid_region_en')['factor_gco2kwh'].to_dict(), 'BM': bm}


def _load_egrid():
    p = os.path.join(GO, 'egrid_subregions.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    # central = non-baseload (marginal proxy); upper = same (eGRID gives one marginal). avg as ref.
    return df.set_index('subregion_acronym')[['co2e_avg_gco2e_per_kwh',
                                              'co2e_nonbaseload_gco2e_per_kwh']].to_dict('index')


def _load_india():
    """India CEA national single-grid combined-margin (central) + operating-margin (upper).
    AVOIDED-EMISSIONS marginal convention: central = combined_margin (CM = 0.5*OM+0.5*BM),
    upper = operating_margin (thermal-only OM). Reads india_cea.csv; returns None if missing."""
    p = os.path.join(GO, 'india_cea.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    def _pick(ftype):
        r = df[df['factor_type'] == ftype]
        return float(r['gco2_per_kwh'].iloc[0]) if not r.empty else np.nan
    cm = _pick('combined_margin')
    om = _pick('operating_margin')
    fy = str(df['fiscal_year'].iloc[0]) if 'fiscal_year' in df.columns and not df.empty else '2024-25'
    if not np.isfinite(cm):
        return None
    return dict(central=cm, upper=om if np.isfinite(om) else cm, fy=fy)


_CHINA = None
_EGRID = None
_INDIA = None


def _ember_fallback(ember, country):
    e = ember[ember['country'] == country] if country else pd.DataFrame()
    if e.empty:
        return (np.nan, np.nan, 0)
    nh = e['nonhydro_gco2kwh'].iloc[0]
    mg = e['fossil_marginal_gco2kwh'].iloc[0]
    yr = int(e['year'].iloc[0])
    return (float(nh) if pd.notna(nh) else np.nan,
            float(mg) if pd.notna(mg) else np.nan, yr)


def resolve_grid(region, country, ember, province=None, egrid_subregion=None):
    """Return dict(central, upper, year, tier, source) - best available grid BACKUP factor.

    region          : REVUB region basename (e.g. 'argentina', 'brazil_norte')
    country         : Ember country name (from REGION_COUNTRY)
    ember           : ember_country_ef DataFrame (fallback)
    province        : Chinese province name (enables China subnational OM)
    egrid_subregion : US eGRID subregion acronym (enables US subnational non-baseload)
    """
    global _CHINA, _EGRID, _INDIA
    nh, mg, eyr = _ember_fallback(ember, country)

    # ① USA subnational (eGRID non-baseload = marginal proxy)
    if country in ('United States of America', 'United States', 'USA') and egrid_subregion:
        if _EGRID is None:
            _EGRID = _load_egrid() or {}
        row = _EGRID.get(egrid_subregion)
        if row:
            c = float(row['co2e_nonbaseload_gco2e_per_kwh'])
            return dict(central=c, upper=c, year=2023, tier='official-subnational',
                        source=f'US eGRID {egrid_subregion} non-baseload')

    # ② China subnational (regional-grid OM = marginal)
    if country == 'China' and province:
        if _CHINA is None:
            _CHINA = _load_china() or {'OM': {}, 'BM': {}}
            _ = _CN_PROV2GRID  # populated as side effect of _load_china
        hit = _CN_PROV2GRID.get(province)
        if hit:
            om, grid_en = hit
            # OM is the marginal generator (single value); central==upper for the
            # subnational marginal factor, same as eGRID non-baseload above.
            return dict(central=om, upper=om, year=2023, tier='official-subnational',
                        source=f'CN MEE {grid_en}-grid OM 2023')

    # ②b India national grid (CEA combined-margin central / OM upper)
    if country == 'India':
        if _INDIA is None:
            _INDIA = _load_india() or {}
        if _INDIA:
            return dict(central=_INDIA['central'], upper=_INDIA['upper'], year=2025,
                        tier='official-country',
                        source=f"India CEA {_INDIA.get('fy', '2024-25')} CM")
        # CSV missing -> fall through to OFFICIAL_COUNTRY['India'] literal fallback below

    # ③ official country-level OM/marginal
    if country in OFFICIAL_COUNTRY:
        c, u, yr, src = OFFICIAL_COUNTRY[country]
        return dict(central=c, upper=u, year=yr, tier='official-country', source=src)

    # ④ Ember non-hydro (+ official grid-avg reference note if any)
    note = ''
    if country in OFFICIAL_AVG_NOTE:
        av, ay, asrc = OFFICIAL_AVG_NOTE[country]
        note = f'; {asrc}={av:.0f}@{ay}'
    return dict(central=nh, upper=mg, year=eyr, tier='ember-country', source=f'Ember {eyr} non-hydro{note}')
