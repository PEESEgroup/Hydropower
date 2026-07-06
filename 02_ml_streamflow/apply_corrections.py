#!/usr/bin/env python3
"""
Final correction program - writes all "after-the-fact verified" corrections
directly into each region's station_channel_result.csv, without rerunning the
009/010/011/batch pipeline.

Correction sources (all web-verified, with data provenance):
  A. Head   : /home/cfeng/hydro/output_data/hydro_locations/step6_final_kept.xlsx
              (net head from 008: final_head_m / final_source / final_rule /
               final_confidence, including 321 manual + web_verified_heads.json)
              matched by id_unified.
  B. Type/volume/capacity : /home/cfeng/myswat/corrections_all_merged.csv
              (5-agent web review of 392 stations; updates type_unified /
               max_vol_km3 / capacity_mw). Matched by (subregion, name_unified).

Design principles (per the user's request to "edit the final aggregation program"):
  - Only touch the final table; do not touch geometry/snapping/channel matching.
  - Back up each table before its first edit as station_channel_result.csv.bak_corr
    (idempotent: skip if it already exists).
  - Dry-run by default; only write with --apply.

Usage:
  python apply_corrections.py            # dry-run, statistics only
  python apply_corrections.py --apply    # write
"""
import argparse, glob, os, shutil
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

STEP6  = '/home/cfeng/hydro/output_data/hydro_locations/step6_final_kept.xlsx'
CORR   = '/home/cfeng/myswat/corrections_all_merged.csv'
PS_VOL = '/home/cfeng/hydro/ps_volume_corrections.json'
PS_NOT = '/home/cfeng/hydro/ps_not_ps_stations.json'
GLOB   = '/datasets/swat_global/*/*/hydro/station_channel_result.csv'
HEAD_COLS = ['final_head_m', 'final_source', 'final_rule', 'final_confidence']
TYPE_MAP = {'STORAGE': 'STO', 'RUN-OF-RIVER': 'ROR', 'PUMPED STORAGE': 'PS', 'PUMPED-STORAGE': 'PS'}
DEFAULT_PS_STORAGE_HOURS = 8.0


def load_head_map():
    s6 = pd.read_excel(STEP6)
    s6['_k'] = s6['id_unified'].astype(str).str.strip()
    s6 = s6[~s6['_k'].duplicated(keep='first')].set_index('_k')
    return {c: s6[c] for c in HEAD_COLS}


def load_corr_lookup():
    c = pd.read_csv(CORR)
    for col in ['original_type', 'corrected_type']:
        c[col] = c[col].astype(str).str.strip().str.upper().replace(TYPE_MAP)
    lut = {}
    for _, r in c.iterrows():
        lut[(str(r['subregion']).strip(), str(r['name']).strip())] = r
    return lut


def load_ps_vol():
    """Load PS upper reservoir volume corrections + misclassified PS list."""
    import json
    ps_corr = {}
    ps_not = set()
    try:
        with open(PS_VOL, encoding='utf-8') as f:
            ps_corr = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(PS_NOT, encoding='utf-8') as f:
            ps_not = {s['name'] for s in json.load(f)}
    except FileNotFoundError:
        pass
    return ps_corr, ps_not


def estimate_ps_vol(cap_mw, head_m, hours=DEFAULT_PS_STORAGE_HOURS):
    if cap_mw <= 0 or head_m <= 0: return np.nan
    return cap_mw * 1e6 * hours * 3600 / (0.88 * 1000 * 9.81 * head_m * 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write changes (default dry-run)')
    args = ap.parse_args()

    hmap = load_head_map()
    corr = load_corr_lookup()
    ps_corr, ps_not = load_ps_vol()
    print(f'step6 heads: {len(hmap["final_head_m"])} stations | corrections: {len(corr)} entries | PS vol: {len(ps_corr)} corrections, {len(ps_not)} not-PS\n')

    csvs = sorted(glob.glob(GLOB))
    n_head = n_type = n_vol = n_cap = n_ps_vol = n_ps_est = n_ps_recl = files_mod = 0
    src_after = {}

    for path in csvs:
        parts = path.split('/')
        subregion = parts[4]                      # /datasets/swat_global/<region>/<subregion>/hydro/... -> [4]=subregion
        df = pd.read_csv(path)
        if 'id_unified' not in df.columns:
            print(f'  SKIP no id_unified: {path}'); continue
        key_id = df['id_unified'].astype(str).str.strip()
        modified = False

        # ---- A. Head (by id_unified) ----
        for col in HEAD_COLS:
            mapped = key_id.map(hmap[col])
            newcol = mapped.where(mapped.notna(), df[col] if col in df.columns else np.nan)
            if col in df.columns:
                changed = (~_eq(df[col], newcol)).sum()
            else:
                changed = int(newcol.notna().sum())
            if changed:
                df[col] = newcol; n_head += int(changed); modified = True

        # ---- B. Type/volume/capacity (by subregion+name) ----
        for idx, row in df.iterrows():
            name = str(row.get('name_unified', '')).strip()
            r = corr.get((subregion, name))
            if r is None:
                continue
            # type
            if str(r['original_type']) != str(r['corrected_type']) and str(r['corrected_type']) != 'NAN':
                if str(df.at[idx, 'type_unified']) != str(r['corrected_type']):
                    df.at[idx, 'type_unified'] = r['corrected_type']; n_type += 1; modified = True
            # volume -> max_vol_km3 (>10% change, or 0<->positive)
            ov, cv = r['original_vol_km3'], r['corrected_vol_km3']
            if not pd.isna(ov) and not pd.isna(cv):
                hit = (ov == 0 and cv > 0) or (cv == 0 and ov > 0) or (ov > 0 and abs(cv - ov) / ov > 0.1)
                if hit and 'max_vol_km3' in df.columns and not _close(df.at[idx, 'max_vol_km3'], cv):
                    df.at[idx, 'max_vol_km3'] = cv; n_vol += 1; modified = True
            # capacity (>10 MW diff)
            oc, cc = r['original_cap_mw'], r['corrected_cap_mw']
            if not pd.isna(oc) and not pd.isna(cc) and abs(cc - oc) > 10:
                if not _close(df.at[idx, 'capacity_mw'], cc):
                    df.at[idx, 'capacity_mw'] = cc; n_cap += 1; modified = True

        # ---- C. PS upper reservoir volume (by name_unified) ----
        for idx, row in df.iterrows():
            name = str(row.get('name_unified', '')).strip()
            typ = str(row.get('type_unified', ''))

            # C1: Reclassify misidentified PS → STO
            if name in ps_not and typ == 'PS':
                df.at[idx, 'type_unified'] = 'STO'
                n_ps_recl += 1; modified = True; continue

            if typ != 'PS': continue

            vol_col = 'ps_vol_km3' if 'ps_vol_km3' in df.columns else 'max_vol_km3'
            head_col = 'final_head_m' if 'final_head_m' in df.columns else 'head_m'

            def _set_ps_vol(idx, vol_km3):
                """Write PS upper vol to BOTH ps_vol_km3 AND max_vol_km3."""
                if 'ps_vol_km3' in df.columns:
                    df.at[idx, 'ps_vol_km3'] = vol_km3
                if 'ps_vol_10e4m3' in df.columns:
                    df.at[idx, 'ps_vol_10e4m3'] = vol_km3 * 1e5
                if 'max_vol_km3' in df.columns:
                    df.at[idx, 'max_vol_km3'] = vol_km3
                if 'max_vol_10e4m3' in df.columns:
                    df.at[idx, 'max_vol_10e4m3'] = vol_km3 * 1e5

            # C2: Apply verified correction
            if name in ps_corr:
                new_vol = ps_corr[name]['ps_upper_vol_km3']
                if new_vol > 0:
                    _set_ps_vol(idx, new_vol)
                    n_ps_vol += 1; modified = True
                continue

            # C3: Estimate for PS without verified data
            cap = float(row.get('capacity_mw', 0) or 0)
            head = float(row.get(head_col, 0) or 0)
            cur_vol = float(row.get(vol_col, 0) or 0)
            if cap > 0 and head > 0:
                cur_h = cur_vol * 1e9 * 1000 * 9.81 * head / (cap * 1e6 * 3600) if cur_vol > 0 else 0
                if cur_h < 4 or cur_h > 200:
                    est = estimate_ps_vol(cap, head)
                    if not np.isnan(est):
                        _set_ps_vol(idx, est)
                        n_ps_est += 1; modified = True

        if 'final_source' in df.columns:
            for k, v in df['final_source'].value_counts().items():
                src_after[k] = src_after.get(k, 0) + int(v)

        if modified and args.apply:
            bak = path + '.bak_corr'
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            df.to_csv(path, index=False)
        if modified:
            files_mod += 1

    print(f'files touched  : {files_mod}/{len(csvs)}')
    print(f'head cells     : {n_head}')
    print(f'type fixes     : {n_type}')
    print(f'volume fixes   : {n_vol}')
    print(f'capacity fixes : {n_cap}')
    print(f'PS vol correct : {n_ps_vol}')
    print(f'PS vol estimate: {n_ps_est}')
    print(f'PS→STO reclass : {n_ps_recl}')
    print(f'final_source AFTER: {dict(sorted(src_after.items(), key=lambda x:-x[1]))}')
    print('\n' + ('APPLIED (backups: *.bak_corr)' if args.apply else 'DRY-RUN - rerun with --apply to write'))


def _eq(a, b):
    an = pd.to_numeric(a, errors='coerce'); bn = pd.to_numeric(b, errors='coerce')
    num_eq = np.isclose(an.fillna(-9e99), bn.fillna(-9e99), rtol=1e-4)
    str_eq = a.astype(str).values == b.astype(str).values
    both_num = an.notna().values & bn.notna().values
    return np.where(both_num, num_eq, str_eq)


def _close(x, y):
    try:
        return abs(float(x) - float(y)) < 1e-6
    except (TypeError, ValueError):
        return str(x) == str(y)


if __name__ == '__main__':
    main()
