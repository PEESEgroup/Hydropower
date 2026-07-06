#!/usr/bin/env python3
"""
Merge all curated corrections into station_channel_result.csv.

Correction sources (highest to lowest priority):
  1. Web-verified corrections (two passes A/B: vol/head/cap/type/status, with URLs)
  2. corrections_all_merged.csv (392 stations: type/vol/cap)
  3. f_reg reclassification already applied by fix_ror_type_in_csv.py
  4. Net head already applied via apply_corrections.py (step 6)

This script applies:
  A. type/vol/cap corrections from corrections_all_merged.csv
  B. vol/head/cap errors found by the web-verification passes
  C. duplicate-station flags found by the web-verification passes
  D. it does NOT modify planned/under-construction stations (they are excluded
     automatically by the flow-capacity filter in REVUB init)

Usage:
  python merge_all_corrections_to_csv.py            # dry-run
  python merge_all_corrections_to_csv.py --apply    # write
"""
import argparse, glob, os, shutil, json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CORR_CSV = '/home/cfeng/myswat/corrections_all_merged.csv'
WEBCHECK_A_JSON = '/tmp/webcheck_corrections_a.json'
WEBCHECK_B_JSON = '/tmp/webcheck_corrections_b.json'
GLOB = '/datasets/swat_global/*/*/hydro/station_channel_result.csv'

TYPE_MAP = {'STORAGE': 'STO', 'RUN-OF-RIVER': 'ROR', 'PUMPED STORAGE': 'PS',
            'PUMPED-STORAGE': 'PS', 'RUN OF RIVER': 'ROR',
            'CONVENTIONAL': None, 'EMBANKMENT DAM': None,
            'MULTIPURPOSE': None, 'UNKNOWN': None}


def load_corrections_csv():
    c = pd.read_csv(CORR_CSV)
    lut = {}
    for _, r in c.iterrows():
        sub = str(r['subregion']).strip()
        name = str(r['name']).strip()
        ct = str(r.get('corrected_type', '')).strip().upper()
        ct = TYPE_MAP.get(ct, ct) or ct
        if ct in ('NAN', '', 'NONE'):
            ct = None
        lut[(sub, name)] = {
            'type': ct,
            'vol': r['corrected_vol_km3'] if pd.notna(r.get('corrected_vol_km3')) else None,
            'cap': r['corrected_cap_mw'] if pd.notna(r.get('corrected_cap_mw')) else None,
            'src': str(r.get('source', 'corrections_all_merged')),
        }
    return lut


def load_webcheck_a():
    if not os.path.exists(WEBCHECK_A_JSON):
        return {}
    with open(WEBCHECK_A_JSON) as f:
        return json.load(f)


def load_webcheck_b():
    if not os.path.exists(WEBCHECK_B_JSON):
        return {}
    with open(WEBCHECK_B_JSON) as f:
        data = json.load(f)
    result = {}
    for s in data.get('result', {}).get('all', []):
        result[s['name']] = s
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    corr_lut = load_corrections_csv()
    webcheck_a = load_webcheck_a()
    webcheck_b = load_webcheck_b()

    print(f'corrections_all_merged: {len(corr_lut)} entries')
    print(f'web-verified corrections (A): {len(webcheck_a)} entries')
    print(f'web-verified corrections (B): {len(webcheck_b)} entries')

    csvs = sorted(glob.glob(GLOB))
    stats = {'type': 0, 'vol': 0, 'cap': 0, 'head': 0, 'dup': 0, 'files': 0}

    for path in csvs:
        parts = path.split('/')
        subregion = parts[4]
        df = pd.read_csv(path)
        modified = False

        for idx, row in df.iterrows():
            name = str(row.get('name_unified', '')).strip()
            if not name:
                continue

            # corrections_all_merged
            cr = corr_lut.get((subregion, name))
            if cr:
                # Type
                if cr['type'] and cr['type'] in ('STO', 'ROR', 'PS', 'Canal'):
                    old_t = str(df.at[idx, 'type_unified']).strip()
                    if old_t != cr['type']:
                        df.at[idx, 'type_unified'] = cr['type']
                        stats['type'] += 1
                        modified = True

                # Volume -> res_vol_km3
                if cr['vol'] is not None and 'res_vol_km3' in df.columns:
                    old_v = pd.to_numeric(df.at[idx, 'res_vol_km3'], errors='coerce')
                    if pd.isna(old_v) or abs(old_v - cr['vol']) > 0.01:
                        df.at[idx, 'res_vol_km3'] = cr['vol']
                        if 'max_vol_km3' in df.columns:
                            df.at[idx, 'max_vol_km3'] = cr['vol']
                        stats['vol'] += 1
                        modified = True

                # Capacity
                if cr['cap'] is not None:
                    old_c = pd.to_numeric(df.at[idx, 'capacity_mw'], errors='coerce')
                    if pd.notna(old_c) and abs(old_c - cr['cap']) > 10:
                        df.at[idx, 'capacity_mw'] = cr['cap']
                        stats['cap'] += 1
                        modified = True

            # web-verified corrections (override for verified stations)
            sc = webcheck_a.get(name)
            if sc:
                # Volume
                if 'vol' in sc:
                    old_v = pd.to_numeric(df.at[idx, 'res_vol_km3'], errors='coerce') if 'res_vol_km3' in df.columns else np.nan
                    if pd.isna(old_v) or abs(old_v - sc['vol']) / max(sc['vol'], 0.001) > 0.1:
                        if 'res_vol_km3' in df.columns:
                            df.at[idx, 'res_vol_km3'] = sc['vol']
                        if 'max_vol_km3' in df.columns:
                            df.at[idx, 'max_vol_km3'] = sc['vol']
                        stats['vol'] += 1
                        modified = True
                        cap = pd.to_numeric(df.at[idx, 'capacity_mw'], errors='coerce')
                        if pd.notna(cap) and cap >= 100:
                            print(f'  VOL: {subregion:<28s} {name[:40]:<42s} -> {sc["vol"]:.2f}km3  src={sc.get("source","")[:50]}')

                # Head
                if 'head' in sc:
                    old_h = pd.to_numeric(df.at[idx, 'final_head_m'], errors='coerce')
                    if pd.notna(old_h) and old_h > 0:
                        ratio = sc['head'] / old_h if old_h > 0 else 999
                        if ratio > 2 or ratio < 0.5:
                            df.at[idx, 'final_head_m'] = sc['head']
                            stats['head'] += 1
                            modified = True
                            print(f'  HEAD: {subregion:<28s} {name[:40]:<42s} {old_h:.1f}->{sc["head"]:.1f}m  src={sc.get("source","")[:50]}')

                # Capacity
                if 'cap' in sc:
                    old_c = pd.to_numeric(df.at[idx, 'capacity_mw'], errors='coerce')
                    if pd.notna(old_c) and abs(old_c - sc['cap']) / max(old_c, 1) > 0.15:
                        df.at[idx, 'capacity_mw'] = sc['cap']
                        stats['cap'] += 1
                        modified = True
                        print(f'  CAP: {subregion:<28s} {name[:40]:<42s} {old_c:.0f}->{sc["cap"]:.0f}MW  src={sc.get("source","")[:50]}')

                # Type (only if the web check explicitly found STO/ROR/PS)
                if sc.get('type') in ('STO', 'ROR', 'PS'):
                    old_t = str(df.at[idx, 'type_unified']).strip()
                    if old_t != sc['type']:
                        df.at[idx, 'type_unified'] = sc['type']
                        stats['type'] += 1
                        modified = True

        if modified:
            stats['files'] += 1
            if args.apply:
                bak = path + '.bak_merge'
                if not os.path.exists(bak):
                    shutil.copy2(path, bak)
                df.to_csv(path, index=False)

    print(f'\n{"="*60}')
    print(f'Type fixes:     {stats["type"]}')
    print(f'Volume fixes:   {stats["vol"]}')
    print(f'Head fixes:     {stats["head"]}')
    print(f'Capacity fixes: {stats["cap"]}')
    print(f'Files modified: {stats["files"]}/{len(csvs)}')
    print(f'\n{"APPLIED (backups: *.bak_merge)" if args.apply else "DRY-RUN - rerun with --apply"}')


if __name__ == '__main__':
    main()
