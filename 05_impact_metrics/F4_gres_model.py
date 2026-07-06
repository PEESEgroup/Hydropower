#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F4_gres_model.py -- Faithful per-reservoir G-res reservoir-GHG model.

Implements the post-impoundment GHG emission sub-models of the G-res Tool
(Prairie et al. 2017; as applied globally by Harrison et al. 2021) to estimate
GHG intensity in gCO2eq/kWh for the global set of ~4193 hydropower reservoirs in
this project, using locally-available input layers.

This REPLACES a crude "zone-median flux x area / generation" approximation with
the actual G-res semi-empirical equations.

COEFFICIENT SOURCE (IMPORTANT): all four pathway regressions use the **clean,
peer-reviewed** values from Soued & Prairie (2020), Biogeosciences 17:515-541,
Supplement Table S2 (which reproduces the Prairie et al. 2017 G-res equations in
plain ASCII), cross-checked against the runnable reference implementation
`reemission` (Janus et al., github.com/tomjanus/reemission). An earlier draft
extracted coefficients from the PDF-GARBLED G-res Technical Documentation and got
several wrong (CH4-diff littoral 0.5058->0.6068; CH4-degas intercept -6.6029->-5.5029;
ebullition radiance term ^0.4 -> linear); those are corrected here against the clean
source. We deliberately use ONE vintage (Prairie 2017 / Soued 2020) consistently
rather than mixing it with the Prairie-2021 recalibration (different coefficients).

Four post-impoundment emission pathways are predicted (techdoc sec 3.1):
    1. CO2 diffusion        (techdoc p.36 "CO2 Diffusive Emissions Integrated on Lifetime")
    2. CH4 diffusion        (techdoc p.37 "CH4 Diffusive Emissions Integrated on 100 yrs")
    3. CH4 ebullition       (techdoc p.38 "CH4 Bubbling Emissions")
    4. CH4 degassing        (techdoc p.39 "CH4 Degassing Emissions"; only if deep intake)

All fluxes are converted to gCO2eq/m2/yr (CH4 GWP100 = 34; CO2 biogenic but
counted; N2O not modeled; 100-yr lifetime-integrated forms used so reservoir age
cancels in the per-kWh ratio). Areal total is then divided by per-area generation
to yield gCO2eq/kWh.

METHODOLOGICAL CHOICES (see README block at bottom of validation report; documented inline):
  * Allocation: 100% to hydropower. These are single-purpose hydropower reservoirs
    in this database (GRanD "use=hydroelectricity"); no multi-use shares available,
    so the full footprint is attributed to generation (techdoc sec 4 allocation
    would only REDUCE the per-kWh value if other uses were present).
  * Net-vs-gross is MIXED (documented honestly): CH4 pathways are GROSS post-impoundment.
    CO2 uses the techdoc lifetime-integrated form, which ALREADY subtracts the 100-yr
    equilibrium "natural baseline" (the displaced-emissions correction, techdoc p.36) - so
    CO2 is effectively net-of-natural-baseline. We do NOT additionally subtract the full
    pre-impoundment landscape balance or Unrelated Anthropogenic Sources (these need
    catchment land-cover layers not available here). Net result: absolute areal fluxes are
    NOT directly comparable to Harrison 2021's bias-corrected global-SUM means
    (~1900-3100 gCO2e/m2/yr - those are retransformation-corrected basin sums, a different
    quantity); our per-reservoir medians (tens-hundreds) are the right scale for per-
    reservoir prediction. Validate by RELATIVE pattern (tropical>boreal; named-reservoir
    ordering) and the per-kWh distribution, not against that global-sum number.
  * %RiverAreaBeforeImpoundment set to 0 (unknown) -> no reduction of CO2 flux.
  * TP transfer: soil total-P (gP/m2) -> reservoir-water TP (ug/L) via a documented
    monotonic percentile map onto an oligo->eutrophic range (5-60 ug/L). TP is a
    LOW-leverage predictor (+2.5% per +10%); a constant-TP=20 variant is computed
    to confirm insensitivity (reported in validation, not written to CSV).
  * Max depth: dam-head proxy (head_m -> max_head_m -> dam_height_m), exactly as
    G-res uses dam height for max depth. Fallback MaxDepth = MeanDepth/0.46 when no
    head, or when head <= mean depth -- 0.46 = the characteristic global mean:max
    depth ratio (Neumann 1959, J.Fish.Res.Board Can. 16:923; Kalff 2002 Limnology).
  * TP soil->reservoir-water transfer: percentile-rank map onto the Nurnberg 1996 /
    Wetzel 2001 trophic TP bands (5-60 ug/L). The clean upstream method would be
    G-res's own land-cover P-export -> Vollenweider/Maavara 2015 (PNAS 112:15603)
    residence-time retention; that needs catchment land-cover P-export coefficients
    not available here. TP is LOW-leverage (+2.5%/+10%, Harrison 2021 Table 7).
  * WRT and degassing annual water flow both derived from HydroRIVERS average
    discharge (snap_DIS_AV_CMS, m3/s), available for 100% of reservoirs.
  * Degassing applied to ALL reservoirs (conservative; hydropower => typically deep
    intake, per Harrison 2021 sec 2.3). A no-degassing variant is also output.

CLI:  python F4_gres_model.py
Output: /datasets/swat_global/_carbon_ef/reservoir_ef_gres.csv
"""

import glob
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Paths / constants
# --------------------------------------------------------------------------- #
CARBON_DIR = "/datasets/swat_global/_carbon_ef"
COORDS_CSV = f"{CARBON_DIR}/all_reservoir_coords.csv"
SOC_CSV = f"{CARBON_DIR}/reservoir_soc.csv"
TP_CSV = f"{CARBON_DIR}/reservoir_totalP.csv"
CLIMATE_CSV = f"{CARBON_DIR}/reservoir_climate.csv"
STATION_GLOB = "/datasets/swat_global/**/station_channel_result.csv"
REVUB_GLOB = "/datasets/swat_global/**/revub_output/**/bal_yearly_summary.parquet"
OUT_CSV = f"{CARBON_DIR}/reservoir_ef_gres.csv"

GWP_CH4 = 34.0            # GWP100 for CH4 (techdoc Table 9, IPCC 2013)
SEC_PER_YR = 31_536_000.0  # s/yr (techdoc "Annual discharge")

# --------------------------------------------------------------------------- #
#  Geometry / climate sub-formulas (techdoc Annex III)
# --------------------------------------------------------------------------- #

def eff_temperature(monthly_T, slope):
    """Effective Temperature (degC), techdoc Annex III p.43.

    EffTemp = log10( mean_over_12months( 10^(T_month * slope) ) ) / slope
    with each monthly T floored at 4 degC before the 10^ step.
    slope = 0.05 for CO2, 0.052 for CH4.
    """
    T = np.asarray(monthly_T, dtype=float)
    if np.all(~np.isfinite(T)):
        return np.nan
    T = np.where(np.isfinite(T), T, np.nan)
    T = np.clip(T, 4.0, None)               # "If Temperature per month lower then 4C, use 4C"
    corr = np.power(10.0, T * slope)        # monthly temperature correction coefficient
    mean_corr = np.nanmean(corr)
    if not np.isfinite(mean_corr) or mean_corr <= 0:
        return np.nan
    return math.log10(mean_corr) / slope


def pct_littoral(mean_depth, max_depth):
    """% Littoral Area, techdoc Annex III p.44.

    q = MaxDepth/MeanDepth - 1                 (q-bathymetric shape)
    %Littoral = [1 - (1 - 3/MaxDepth)^q] * 100
    Clamp 3/MaxDepth <= 1; if MaxDepth <= 3 m, the whole reservoir is littoral -> 100%.

    NOTE: this is the single dominant driver of CH4 diffusion + ebullition (and,
    through diffusion, degassing). It depends entirely on the max:mean depth ratio,
    which here rests on the dam-head proxy for max depth -- the largest single
    source of uncertainty in this model (Harrison 2021 sec 3.4 efficiency analysis).
    """
    if not (np.isfinite(mean_depth) and np.isfinite(max_depth)):
        return np.nan
    if max_depth <= 3.0:
        return 100.0
    if mean_depth <= 0:
        return np.nan
    q = max_depth / mean_depth - 1.0
    if q <= 0:                               # max ~ mean (flat) -> degenerate; treat as fully littoral-ish
        return 100.0
    base = 1.0 - 3.0 / max_depth             # 3/MaxDepth < 1 guaranteed here
    val = (1.0 - base ** q) * 100.0
    return float(np.clip(val, 0.0, 100.0))


def cum_ghr(srad_monthly_kJ, ice_free_months):
    """Reservoir Cumulative Global Horizontal Radiance (kWh/m2/period), techdoc p.46.

    CumGHR = mean(daily GHR over ICE-FREE months) * (number of ice-free months).
    NB UNITS: NO x30.4 day->month factor. Soued&Prairie 2020 Table S3 reports CumGHR
    = 56.4 "kWh/m2/yr" for a tropical site (Batang Ai); that is mean_daily_radiance
    (~4.7 kWh/m2/day) x 12 ice-free months, NOT an annual integral. The reemission
    reference code (Janus, tomjanus/reemission) confirms the x30.4 G-res-techdoc factor
    is DELIBERATELY DISABLED ("results in very high CH4 emissions; set to 1.0"). The
    ebullition coefficient 0.04928 was fit to THIS definition, so applying x30.4 would
    inflate the exponent ~30x. Input srad is WorldClim kJ/m2/day -> /3600 -> kWh/m2/day.
    """
    srad = np.asarray(srad_monthly_kJ, dtype=float)
    if np.all(~np.isfinite(srad)):
        return np.nan
    srad_kWh_d = srad / 3600.0               # kJ/m2/day -> kWh/m2/day
    n = ice_free_months
    if not np.isfinite(n) or n <= 0:
        n = 12.0
    n = float(np.clip(n, 1.0, 12.0))
    # Use the n warmest months as the ice-free set (highest srad ~ warmest).
    valid = srad_kWh_d[np.isfinite(srad_kWh_d)]
    if valid.size == 0:
        return np.nan
    k = int(round(n))
    k = max(1, min(k, valid.size))
    icefree = np.sort(valid)[::-1][:k]
    mean_ghr = np.mean(icefree)              # kWh/m2/day, ice-free period
    return mean_ghr * n                      # kWh/m2/day x n_months (Soued&Prairie units; NO x30.4)


# --------------------------------------------------------------------------- #
#  Emission pathway models (techdoc Annex III, "Statistical summary")
# --------------------------------------------------------------------------- #

def co2_diffusion(eff_temp_co2, area_km2, soilC, tp_ugL, pct_river_before=0.0):
    """CO2 Diffusive Emissions Integrated on Lifetime (gCO2e/m2/yr).
    techdoc p.36.

    Annual (age-specific) flux, log10 (mg C/m2/d):
       log10(f) = 1.7892 + 0.0400*EffTempCO2 + 0.06918*log10(Area_km2)
                  + 0.0216*SoilC_kgC_m2 + 0.1472*log10(TP_ugL)
    Lifetime form (techdoc gives the explicit integrated expression): the age term
    is flux ~ age^b with b = -0.3364; the 100-yr lifetime-integrated mean WITHOUT
    the natural baseline is
       I = [100^(b+1) - 0.5^(b+1)] / [(b+1)*(100-0.5)]     (integral 0.5..100 / span)
    times the intercept flux, MINUS the 100-yr equilibrium flux
       base100 = 10^( intercept_log + b*log10(100) )
    Unit conversion: *(44/12) C->CO2, *365 annualize, /1000 mg->g, and the
    (1 - %RiverAreaBeforeImpoundment/100) net-impounded-area proration.
    """
    if not (np.isfinite(eff_temp_co2) and np.isfinite(area_km2) and area_km2 > 0):
        return np.nan
    soilC = soilC if np.isfinite(soilC) else 0.0
    tp = tp_ugL if (np.isfinite(tp_ugL) and tp_ugL > 0) else 20.0
    b = -0.3364
    intercept_log = (1.7892
                     + 0.0400 * eff_temp_co2
                     + 0.06918 * math.log10(area_km2)
                     + 0.0216 * soilC
                     + 0.1472 * math.log10(tp))
    # lifetime-integration factor (integral of age^b from 0.5 to 100, divided by span)
    I = (100.0 ** (b + 1) - 0.5 ** (b + 1)) / ((b + 1) * (100.0 - 0.5))
    river_frac = 1.0 - (pct_river_before / 100.0 if np.isfinite(pct_river_before) else 0.0)
    unit = (44.0 / 12.0) * 365.0 / 1000.0
    integrated = (10.0 ** intercept_log) * I * unit * river_frac
    base100 = (10.0 ** (intercept_log + b * math.log10(100.0))) * unit * river_frac
    flux = integrated - base100              # "without natural baseline" (displaced-emissions removed)
    return max(flux, 0.0)


def ch4_diffusion(eff_temp_ch4, pct_litt):
    """CH4 Diffusive Emissions Integrated on 100 yrs (gCO2e/m2/yr).
    techdoc p.37.

    log10(annual f) = 0.8804 - 0.0116*Age + 0.6068*log10(%Littoral/100) + 0.04828*EffTempCH4
    (Soued&Prairie 2020 Biogeosciences 17:515 SI Table S2, reproducing Prairie 2017.)
    The age term is LINEAR in Age, so the annual flux ~ 10^(-0.0116*Age) (exponential
    decay); its mean over 0..100 yr is the closed form
       J = [1 - 10^(-100*0.0116)] / (100*0.0116*ln(10))
    Unit: *365/1000 *(16/12) *34. (No natural-baseline subtraction for CH4.)
    """
    if not (np.isfinite(eff_temp_ch4) and np.isfinite(pct_litt) and pct_litt > 0):
        return np.nan
    intercept_log = (0.8804
                     + 0.6068 * math.log10(pct_litt / 100.0)
                     + 0.04828 * eff_temp_ch4)
    J = (1.0 - 10.0 ** (-100.0 * 0.0116)) / (100.0 * 0.0116 * math.log(10.0))
    flux = (10.0 ** intercept_log) * J * (365.0 / 1000.0) * (16.0 / 12.0) * GWP_CH4
    return max(flux, 0.0)


def ch4_ebullition(pct_litt, cumghr):
    """CH4 Bubbling Emissions (gCO2e/m2/yr).  techdoc p.38.

    Soued&Prairie 2020 SI Table S2 (clean ASCII, reproducing Prairie 2017):
        log10(f) = -0.98574 + 1.0075*log10(%Littoral/100) + 0.04928*CumGHR
    The cumulative-global-horizontal-radiance term is LINEAR (no exponent, no log, no
    /30.4) -- confirmed against BOTH clean sources (Soued&Prairie 2020 SI and the
    runnable reemission reference code, which codes `k3_ebull * global_radiance()`
    linearly). An earlier draft from the garbled G-res techdoc used CumGHR^0.4/^0.5;
    that was a PDF-decode artifact and is REMOVED. CumGHR in kWh/m2/day x n_months
    (see cum_ghr; ~tens for the tropics, matching Soued's 56.4). Ebullition has no age
    term (time-independent). Unit: *(16/12) *34 *365/1000.
    """
    if not (np.isfinite(pct_litt) and pct_litt > 0 and np.isfinite(cumghr) and cumghr > 0):
        return np.nan
    log_f = (-0.98574
             + 1.0075 * math.log10(pct_litt / 100.0)
             + 0.04928 * cumghr)
    flux = (10.0 ** log_f) * (16.0 / 12.0) * GWP_CH4 * 365.0 / 1000.0
    return max(flux, 0.0)


def ch4_degassing(ch4_diff_gco2e, wrt_yr, annual_flow_m3yr, res_area_m2):
    """CH4 Degassing Emissions (gCO2e/m2/yr).  techdoc p.39.

    [CH4]_diff (g CH4-C/m3) = 10^( -5.5029 + 2.2857*log10(CH4_diffusion_gCO2e_m2_yr)
                                    + 0.9866*log10(WRT_yr) )
    (Soued&Prairie 2020 SI Table S2, reproducing Prairie 2017.)
    Reservoir-wide -> areal:
        flux = [CH4]_diff * (AnnualWaterFlow_m3yr) * 0.9 * (16/12) * 34 / ReservoirArea_m2
    AnnualWaterFlow = the annual through-flow volume, taken as HydroRIVERS discharge *
    seconds/yr. 0.9 = degassing efficiency factor. (Dividing by area_m2 is equivalent to
    the reemission 1e-6 t-bookkeeping / area_km2.) Applied only when intake is below the
    thermocline; here applied to all (hydropower => typically deep intake, Harrison 2021),
    with a no-degassing variant also reported.
    """
    if not (np.isfinite(ch4_diff_gco2e) and ch4_diff_gco2e > 0
            and np.isfinite(wrt_yr) and wrt_yr > 0
            and np.isfinite(annual_flow_m3yr) and annual_flow_m3yr > 0
            and np.isfinite(res_area_m2) and res_area_m2 > 0):
        return np.nan
    log_k = (-5.5029
             + 2.2857 * math.log10(ch4_diff_gco2e)
             + 0.9866 * math.log10(wrt_yr))
    flux = (10.0 ** log_k) * annual_flow_m3yr * 0.9 * (16.0 / 12.0) * GWP_CH4 / res_area_m2
    return max(flux, 0.0)


# --------------------------------------------------------------------------- #
#  Data loading / joining
# --------------------------------------------------------------------------- #

def load_station_attrs():
    """Concatenate per-region station_channel_result.csv; keep geometry + discharge.

    Keeps lat_unified so the downstream join can key on (name, lat) - `name` alone is
    NOT unique (e.g. two distinct 'Funil' dams in brazil_sudeste; 'Santa Clara' in both
    sudeste and sul), and a name-only join silently mis-assigns geometry across them.
    """
    cols = ["name_unified", "lat_unified", "head_m", "max_head_m", "dam_height_m",
            "res_avg_depth_m", "max_area_km2", "max_vol_km3", "res_area_km2", "res_vol_km3",
            "snap_DIS_AV_CMS", "type_unified"]
    files = glob.glob(STATION_GLOB, recursive=True)
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
            frames.append(d)
        except Exception as e:
            print(f"  WARN station file {f}: {e}")
    sc = pd.concat(frames, ignore_index=True)
    sc["latkey"] = sc["lat_unified"].round(3)
    # collapse exact (name, lat) duplicates only: take the first non-null per column
    sc = sc.sort_values("name_unified").groupby(["name_unified", "latkey"], as_index=False).first()
    return sc


def load_generation():
    """Map reservoir name -> mean annual generation (GWh/yr) over the bal summaries.

    Only ~11 regions have revub_output. If a reservoir appears in multiple
    GCM/SSP/tag summaries we average across them (single climate-neutral estimate).
    Match key: parquet 'station' == coords 'name' (verified identical strings).
    """
    files = glob.glob(REVUB_GLOB, recursive=True)
    rows = []
    for f in files:
        try:
            p = pd.read_parquet(f, columns=["station", "E_total_GWh"])
            g = p.groupby("station")["E_total_GWh"].mean().reset_index()
            rows.append(g)
        except Exception as e:
            print(f"  WARN parquet {f}: {e}")
    if not rows:
        return pd.DataFrame(columns=["name", "gen_GWh"])
    allg = pd.concat(rows, ignore_index=True)
    gen = allg.groupby("station")["E_total_GWh"].mean().reset_index()
    gen.columns = ["name", "gen_GWh"]
    return gen


def tp_transfer(soil_tp_gm2_series):
    """Soil total-P (gP/m2) -> reservoir-water TP (ug/L), documented monotonic proxy.

    G-res's CO2 model wants reservoir-water TP (ug/L), not soil gP/m2. We map the
    soil-P percentile rank of each reservoir onto an oligo->eutrophic reservoir-TP
    range [5, 60] ug/L (Wetzel 2001 trophic bands: oligo <10, meso 10-30, eutro
    30-100). This is monotonic and defensible; TP is a low-leverage predictor in the
    CO2 model (+2.5% flux per +10% TP, Harrison 2021 Table 7), so the exact mapping
    barely moves results -- the constant-TP=20 variant in validation confirms this.
    """
    s = soil_tp_gm2_series.astype(float)
    valid = s[np.isfinite(s)]
    if valid.empty:
        return pd.Series(20.0, index=s.index)
    ranks = s.rank(pct=True)                 # 0..1 percentile
    tp = 5.0 + ranks * (60.0 - 5.0)
    tp = tp.fillna(20.0)
    return tp


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading inputs ...")
    coords = pd.read_csv(COORDS_CSV)
    soc = pd.read_csv(SOC_CSV)[["name", "lat", "ocs_tha"]]
    tp = pd.read_csv(TP_CSV)[["name", "lat", "totalP_value"]]
    clim = pd.read_csv(CLIMATE_CSV)
    station = load_station_attrs()
    gen = load_generation()

    # `name` is NOT unique (collisions both within a region - two 'Funil' in
    # brazil_sudeste - and across regions - 'Funil'/'Miranda' Brazil vs Spain). A
    # name-only join silently mis-assigns layer values. We therefore key every layer
    # on (name, latkey=round(lat,3)); lat distinguishes all genuine collisions.
    coords["latkey"] = coords["lat"].round(3)
    soc["latkey"] = soc["lat"].round(3)
    tp["latkey"] = tp["lat"].round(3)
    clim["latkey"] = clim["lat"].round(3)
    soc = soc.drop_duplicates(["name", "latkey"], keep="first")
    tp = tp.drop_duplicates(["name", "latkey"], keep="first")
    clim = clim.drop_duplicates(["name", "latkey"], keep="first")
    # generation parquet is name-keyed only (no lat); collisions there are an upstream
    # limitation - averaged. Kept name-only.
    gen = gen.drop_duplicates("name", keep="first")
    print(f"  coords={len(coords)}  station_rows={len(station)}  gen_stations={len(gen)}")

    # --- join layers on (name, latkey); station on (name, latkey); gen on name ---
    df = coords.merge(soc.drop(columns=["lat"]), on=["name", "latkey"], how="left")
    df = df.merge(tp.drop(columns=["lat"]), on=["name", "latkey"], how="left")
    df = df.merge(clim.drop(columns=["lat", "lon"], errors="ignore"), on=["name", "latkey"], how="left")
    station = station.rename(columns={"name_unified": "name"})
    df = df.merge(station.drop(columns=["lat_unified"], errors="ignore"),
                  on=["name", "latkey"], how="left")
    df = df.merge(gen, on="name", how="left")
    print(f"  joined rows={len(df)}")

    # --- soil carbon: ocs_tha (t/ha) -> kgC/m2  (1 t/ha = 0.1 kg/m2) ---
    df["soilC_kgC_m2"] = df["ocs_tha"] * 0.1

    # --- TP transfer ---
    df["tp_ugL"] = tp_transfer(df["totalP_value"])

    # --- geometry fallbacks ---
    # Area: coords.area_km2 -> max_area_km2 -> res_area_km2
    df["area_km2_use"] = (df["area_km2"]
                          .fillna(df.get("max_area_km2"))
                          .fillna(df.get("res_area_km2")))
    # Volume: coords.vol_km3 -> max_vol_km3 -> res_vol_km3
    df["vol_km3_use"] = (df["vol_km3"]
                         .fillna(df.get("max_vol_km3"))
                         .fillna(df.get("res_vol_km3")))

    # Mean depth: provided depth -> res_avg_depth_m -> Volume/Area*1000
    df["mean_depth_m"] = df["depth_m"].fillna(df.get("res_avg_depth_m"))
    md_from_vol = df["vol_km3_use"] / df["area_km2_use"] * 1000.0
    df["mean_depth_m"] = df["mean_depth_m"].fillna(md_from_vol)

    # Max depth: dam-head proxy (head_m -> max_head_m -> dam_height_m)
    head_proxy = (df.get("head_m")
                  .fillna(df.get("max_head_m"))
                  .fillna(df.get("dam_height_m")))
    df["max_depth_m"] = head_proxy
    # Ensure MaxDepth > MeanDepth; else fallback MaxDepth = MeanDepth / 0.46
    need_fallback = ~(df["max_depth_m"] > df["mean_depth_m"])
    df.loc[need_fallback, "max_depth_m"] = df.loc[need_fallback, "mean_depth_m"] / 0.46
    df["_maxdepth_fallback"] = need_fallback | head_proxy.isna()

    # --- climate-derived predictors ---
    tcols = [f"tavg_{m:02d}" for m in range(1, 13)]
    scols = [f"srad_{m:02d}" for m in range(1, 13)]
    have_t = all(c in df.columns for c in tcols)
    have_s = all(c in df.columns for c in scols)

    eff_co2, eff_ch4, ghr = [], [], []
    for _, r in df.iterrows():
        Tm = [r.get(c, np.nan) for c in tcols] if have_t else [np.nan] * 12
        Sm = [r.get(c, np.nan) for c in scols] if have_s else [np.nan] * 12
        eff_co2.append(eff_temperature(Tm, 0.05))
        eff_ch4.append(eff_temperature(Tm, 0.052))
        ghr.append(cum_ghr(Sm, r.get("ice_free_months", np.nan)))
    df["eff_temp_co2"] = eff_co2
    df["eff_temp_ch4"] = eff_ch4
    df["cum_ghr"] = ghr

    # --- littoral ---
    df["pct_littoral"] = [pct_littoral(md, mx)
                          for md, mx in zip(df["mean_depth_m"], df["max_depth_m"])]

    # --- WRT and annual through-flow from HydroRIVERS discharge ---
    df["annual_flow_m3yr"] = df["snap_DIS_AV_CMS"] * SEC_PER_YR
    vol_m3 = df["vol_km3_use"] * 1e9
    df["wrt_yr"] = vol_m3 / df["annual_flow_m3yr"]
    df.loc[~np.isfinite(df["wrt_yr"]), "wrt_yr"] = np.nan

    res_area_m2 = df["area_km2_use"] * 1e6

    # --- pathway fluxes (gCO2e/m2/yr) ---
    df["co2_diff_gco2_m2yr"] = [
        co2_diffusion(et, a, sc, tp_, 0.0)
        for et, a, sc, tp_ in zip(df["eff_temp_co2"], df["area_km2_use"],
                                  df["soilC_kgC_m2"], df["tp_ugL"])
    ]
    df["ch4_diff"] = [ch4_diffusion(et, pl)
                      for et, pl in zip(df["eff_temp_ch4"], df["pct_littoral"])]
    df["ch4_ebull"] = [ch4_ebullition(pl, g)
                       for pl, g in zip(df["pct_littoral"], df["cum_ghr"])]
    df["ch4_degas"] = [
        ch4_degassing(cd, w, fl, am)
        for cd, w, fl, am in zip(df["ch4_diff"], df["wrt_yr"],
                                 df["annual_flow_m3yr"], res_area_m2)
    ]

    # --- degassing STRATIFICATION GATE + literature CAP (validated against published GHG) ---
    # Degassing physically requires a stratified anoxic hypolimnion AND a below-thermocline
    # intake; G-res itself gates degassing on "intake deeper than the thermocline" (Harrison
    # et al. 2021 GBC 35:e2020GB006888; reemission code; IHA G-res docs) -- it is NOT a blanket
    # pathway. The unconstrained term blows up (1e5-1e6 gCO2e/m2/yr) for small-area / high-
    # throughflow run-of-river plants because it scales ~discharge/area.
    # GATE (run-of-river / polymictic screen): degassing = 0 when the reservoir cannot hold a
    # stable seasonal hypolimnion -- mean depth < 5 m (polymictic; Lewis 1983 CJFAS 40:1779) OR
    # residence time < 0.1 yr (~36 d; short-WRT = riverine, the Q/V-dominated regime of the
    # densimetric-Froude criterion F=320(L/D)(Q/V)>1, Winton et al. 2019 Biogeosciences 16:1657;
    # Sawakuchi et al. 2021 measured negligible degassing at run-of-river Santo Antonio). NB a
    # pure depth>thermocline(6.95*A^0.185) gate is NOT used: for large reservoirs it gives false
    # negatives (would wrongly zero Balbina, mean depth ~7 m but a real deep-intake degasser).
    # CAP: bound surviving degassing at 4000 gCO2e/m2/yr, the max MEASURED reservoir-surface-
    # normalised degassing (Tucurui; Kemenes et al. 2016 Inland Waters 6:295), so out-of-domain
    # extrapolation cannot exceed the worst real reservoir.
    DEGAS_CAP = 4000.0
    can_strat = df["mean_depth_m"].notna() & df["wrt_yr"].notna()
    riverine = can_strat & ((df["mean_depth_m"] < 5.0) | (df["wrt_yr"] < 0.1))
    df.loc[riverine, "ch4_degas"] = 0.0                          # mixed/run-of-river -> no degassing
    df["ch4_degas"] = df["ch4_degas"].clip(upper=DEGAS_CAP)      # bound at max measured (Tucurui)

    # --- areal totals ---
    # Missing ebullition/degassing (cumghr/WRT gaps) contribute 0 - minor, they are small
    # pathways. But a row missing CO2-diffusion OR CH4-diffusion is NOT a valid total (those
    # two are the bulk of the flux) - flag it and NaN the total so partial rows can't
    # masquerade as complete (audit finding: 253 CO2-only rows were silently under-totalled).
    paths = ["co2_diff_gco2_m2yr", "ch4_diff", "ch4_ebull", "ch4_degas"]
    core_ok = df["co2_diff_gco2_m2yr"].notna() & df["ch4_diff"].notna()
    df["pathways_complete"] = core_ok & df["ch4_ebull"].notna() & df["ch4_degas"].notna()
    df["areal_total_gco2e_m2yr"] = df[paths].sum(axis=1, min_count=1).where(core_ok)
    areal_no_degas = df[["co2_diff_gco2_m2yr", "ch4_diff", "ch4_ebull"]].sum(axis=1, min_count=1).where(core_ok)

    # --- per-kWh ---
    # areal (gCO2e/m2/yr) * area (m2) = gCO2e/yr ; / (gen GWh/yr * 1e9 Wh? ) ...
    # generation GWh/yr -> kWh/yr = GWh * 1e6.  gCO2e/yr / kWh/yr = gCO2e/kWh.
    annual_g = df["areal_total_gco2e_m2yr"] * res_area_m2
    annual_g_nd = areal_no_degas * res_area_m2
    gen_kWh = df["gen_GWh"] * 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        df["gres_ef_gco2kwh"] = annual_g / gen_kWh
        df["gres_ef_no_degas_gco2kwh"] = annual_g_nd / gen_kWh
    df.loc[~np.isfinite(df["gres_ef_gco2kwh"]), "gres_ef_gco2kwh"] = np.nan
    df.loc[~np.isfinite(df["gres_ef_no_degas_gco2kwh"]), "gres_ef_no_degas_gco2kwh"] = np.nan

    # --- GEOMETRY-VALIDITY GUARD (exclude impossible-geometry reservoirs) ---
    # A dammed storage/PS reservoir cannot have a mean depth below ~1.5 m. Such values come
    # from a bad area or volume input (e.g. DEM flood-fill grossly over-estimating surface area
    # -> Cardenillo: 1195 km2 / 0.05 km3 = 0.04 m "depth", giving a spurious EF of ~6900). The
    # over-estimated area inflates total emissions (areal x area), so the per-kWh EF is not
    # trustworthy. Flag and drop the EF (and areal) for these data-quality cases.
    geom_bad = df["mean_depth_m"] < 1.5
    df["geom_implausible"] = geom_bad
    df.loc[geom_bad, ["areal_total_gco2e_m2yr", "gres_ef_gco2kwh",
                      "gres_ef_no_degas_gco2kwh"]] = np.nan

    # --- constant-TP=20 sensitivity variant (validation only, not written) ---
    co2_tp20 = [co2_diffusion(et, a, sc, 20.0, 0.0)
                for et, a, sc in zip(df["eff_temp_co2"], df["area_km2_use"], df["soilC_kgC_m2"])]
    df["_co2_tp20"] = co2_tp20

    # --- write output ---
    out_cols = ["name", "region", "lat", "area_km2", "mean_depth_m", "max_depth_m",
                "pct_littoral", "eff_temp_co2", "eff_temp_ch4", "cum_ghr", "wrt_yr",
                "soilC_kgC_m2", "tp_ugL",
                "co2_diff_gco2_m2yr", "ch4_diff", "ch4_ebull", "ch4_degas",
                "pathways_complete", "geom_implausible", "areal_total_gco2e_m2yr", "gen_GWh",
                "gres_ef_gco2kwh", "gres_ef_no_degas_gco2kwh"]
    out = df.copy()
    out["area_km2"] = out["area_km2_use"]    # write the actually-used area
    out = out[out_cols]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  rows={len(out)}")

    # --- validation report ---
    validation_report(df)


def validation_report(df):
    def q(s, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return "no data"
        return {f"p{int(p*100)}": round(float(s.quantile(p)), 1) for p in ps}

    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    n = len(df)

    print("\n[1] FALLBACK / COVERAGE RATES")
    print(f"  total reservoirs: {n}")
    print(f"  max-depth fallback (no usable head): {df['_maxdepth_fallback'].sum()} "
          f"({100*df['_maxdepth_fallback'].mean():.1f}%)")
    print(f"  missing area:        {df['area_km2_use'].isna().sum()} "
          f"({100*df['area_km2_use'].isna().mean():.1f}%)")
    print(f"  missing mean depth:  {df['mean_depth_m'].isna().sum()}")
    print(f"  missing %littoral:   {df['pct_littoral'].isna().sum()}")
    print(f"  missing WRT:         {df['wrt_yr'].isna().sum()}")
    print(f"  has generation:      {df['gen_GWh'].notna().sum()} "
          f"({100*df['gen_GWh'].notna().mean():.1f}%)")
    print(f"  has areal flux:      {df['areal_total_gco2e_m2yr'].notna().sum()}")
    print(f"  has per-kWh EF:      {df['gres_ef_gco2kwh'].notna().sum()}")
    if "type_unified" in df.columns:
        noarea = df[df["area_km2_use"].isna()]
        print("  -- reservoirs lacking area (no G-res flux), by plant type:")
        print(f"     {noarea['type_unified'].value_counts().to_dict()}")
        print("     (ROR/Canal have no reservoir surface -> areal flux is physically"
              " inapplicable; only STO/PS with storage are true coverage gaps)")

    print("\n[2] AREAL FLUX (gCO2e/m2/yr) -- judge by RELATIVE pattern, not absolute level")
    print("    (Harrison ~1900-3100 are bias-corrected global SUMS, a different quantity;")
    print("     per-reservoir medians of tens-hundreds are correct here)")
    print(f"  total areal flux quantiles: {q(df['areal_total_gco2e_m2yr'])}")
    print(f"  mean areal flux: {df['areal_total_gco2e_m2yr'].mean():.0f}")
    for c, lbl in [("co2_diff_gco2_m2yr", "CO2 diff"), ("ch4_diff", "CH4 diff"),
                   ("ch4_ebull", "CH4 ebull"), ("ch4_degas", "CH4 degas")]:
        print(f"    {lbl:10s}: {q(df[c])}")

    # tropical vs boreal latitude bands
    lat = df["lat"].abs()
    trop = df[lat < 23.5]["areal_total_gco2e_m2yr"]
    bor = df[lat > 50]["areal_total_gco2e_m2yr"]
    print(f"  tropical (|lat|<23.5) median areal: {trop.median():.0f}  (n={trop.notna().sum()})")
    print(f"  boreal   (|lat|>50)   median areal: {bor.median():.0f}  (n={bor.notna().sum()})")

    print("\n[3] PER-kWh EF (gCO2/kWh) -- cf IPCC/G-res median ~20-80; Scherer&Pfister mean ~273")
    ef = pd.to_numeric(df["gres_ef_gco2kwh"], errors="coerce").dropna()
    efnd = pd.to_numeric(df["gres_ef_no_degas_gco2kwh"], errors="coerce").dropna()
    print(f"  with degassing:    n={len(ef)}  {q(ef)}  mean={ef.mean():.0f}")
    print(f"  no degassing:      n={len(efnd)}  {q(efnd)}  mean={efnd.mean():.0f}")
    if len(ef):
        print(f"  fraction > 820 (coal): {100*(ef>820).mean():.1f}%   "
              f"> 1000 (Balbina-class): {100*(ef>1000).mean():.1f}%")
        print(f"  fraction < 50: {100*(ef<50).mean():.1f}%   < 100: {100*(ef<100).mean():.1f}%")

    print("\n[4] NAMED RESERVOIR SPOT-CHECKS (areal gCO2e/m2/yr; EF gCO2/kWh)")
    for key in ["Balbina", "Tucuru", "Itaipu", "Furnas", "Petit", "Sobradinho",
                "Three Gorges", "Hoover", "Kaunas", "Kruonis"]:
        sub = df[df["name"].str.contains(key, case=False, na=False)]
        for _, r in sub.head(2).iterrows():
            print(f"  {r['name'][:45]:45s} lat={r['lat']:.1f} "
                  f"litt={r['pct_littoral']:.0f}% areal={r['areal_total_gco2e_m2yr']:.0f} "
                  f"EF={r['gres_ef_gco2kwh'] if np.isfinite(r['gres_ef_gco2kwh']) else float('nan'):.0f}")

    print("\n[5] TP SENSITIVITY (constant TP=20 ug/L vs percentile transfer)")
    a = pd.to_numeric(df["co2_diff_gco2_m2yr"], errors="coerce")
    b = pd.to_numeric(df["_co2_tp20"], errors="coerce")
    rel = ((a - b) / b).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  median relative change in CO2-diffusion flux: {100*rel.median():.1f}%  "
          f"(IQR {100*rel.quantile(.25):.1f}..{100*rel.quantile(.75):.1f}%)  "
          f"-> confirms TP is low-leverage")

    # vs crude method: aggregate the per-region hydro/reservoir_ef.csv files
    print("\n[6] VS PRIOR CRUDE METHOD (per-region hydro/reservoir_ef.csv)")
    crude_files = glob.glob("/datasets/swat_global/**/hydro/reservoir_ef.csv", recursive=True)
    cr = []
    for f in crude_files:
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("name_unified", "hydro_ef_gco2kwh"))
            cr.append(d)
        except Exception:
            pass
    if cr:
        crude = pd.concat(cr, ignore_index=True).drop_duplicates("name_unified", keep="first")
        crude = crude.rename(columns={"name_unified": "name", "hydro_ef_gco2kwh": "crude_ef"})
        cmp = df[["name", "gres_ef_gco2kwh"]].merge(crude, on="name", how="inner")
        cmp = cmp.dropna(subset=["gres_ef_gco2kwh", "crude_ef"])
        print(f"  crude files={len(crude_files)}  matched reservoirs={len(cmp)}")
        if len(cmp):
            print(f"  G-res median EF: {cmp['gres_ef_gco2kwh'].median():.1f}  "
                  f"crude median EF: {cmp['crude_ef'].median():.1f}")
            print(f"  G-res mean EF:   {cmp['gres_ef_gco2kwh'].mean():.1f}  "
                  f"crude mean EF:   {cmp['crude_ef'].mean():.1f}")
            ratio = (cmp["gres_ef_gco2kwh"] / cmp["crude_ef"]).replace([np.inf, -np.inf], np.nan).dropna()
            print(f"  G-res/crude ratio median: {ratio.median():.2f}  "
                  f"(IQR {ratio.quantile(.25):.2f}..{ratio.quantile(.75):.2f})")
    else:
        print("  no prior crude files found -- skipped")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
