"""F_prep_driver.py - build the per-region prep CSVs for the whole region set.

Runs LOCALLY where RiverATLAS_v10.gdb and _carbon_ef live. The upstream prep
scripts discover regions by globbing revub_output npz (only present for a few
regions locally); we instead drive them directly over every hydro dir that has
a station_channel_result.csv, reading the heavy GDB / carbon tables exactly once.

  F2_prepare_slope         -> hydro/channel_slope.csv     (RiverATLAS slope)
  F3_prepare_routing       -> hydro/supply_segments.csv   (RiverATLAS network)
  F4_prepare_emission_factors -> hydro/emission_factors.csv (ember + G-res)

F1 sediment_params.csv (Macrostrat API) is handled separately; F1 metrics fall
back to default b exponents when it is absent.

Usage:  python F_prep_driver.py [only_missing]
        only_missing  -> skip regions that already have the product (default: all)
"""
import glob
import os
import sys
import traceback

import pandas as pd

import F2_prepare_slope as P2
import F3_prepare_routing as P3
import F4_prepare_emission_factors as P4

ROOT = '/datasets/swat_global'
ONLY_MISSING = len(sys.argv) > 1 and sys.argv[1] == 'only_missing'


def regions():
    paths = glob.glob(os.path.join(ROOT, '*', '*', 'hydro',
                                   'station_channel_result.csv'))
    return sorted({os.path.dirname(os.path.dirname(p)) for p in paths})


def main():
    regs = regions()
    print(f'[prep] {len(regs)} regions; only_missing={ONLY_MISSING}', flush=True)

    # ---- F2 slope (RiverATLAS read once) -------------------------------
    print('[prep] loading RiverATLAS slope lookup ...', flush=True)
    slope = P2.load_riveratlas_slope()
    print('[prep] loading RiverATLAS network ...', flush=True)
    net = P3.load_network()
    print('[prep] loading carbon tables ...', flush=True)
    ember = pd.read_csv(os.path.join(P4.CARBON, 'ember_country_ef.csv'))
    gres, geom_bad = P4._load_gres()

    counts = {'slope': 0, 'segments': 0, 'emission': 0, 'err': 0}
    for r in regs:
        hyd = os.path.join(r, 'hydro')
        name = os.path.basename(r)
        # F2 slope
        try:
            tgt = os.path.join(hyd, 'channel_slope.csv')
            if not (ONLY_MISSING and os.path.exists(tgt)):
                if P2.process_region(hyd, slope):
                    counts['slope'] += 1
        except Exception:
            counts['err'] += 1
            print(f'  ERR slope {name}\n{traceback.format_exc()}', flush=True)
        # F3 segments
        try:
            tgt = os.path.join(hyd, 'supply_segments.csv')
            if not (ONLY_MISSING and os.path.exists(tgt)):
                if P3.process_region(hyd, net):
                    counts['segments'] += 1
        except Exception:
            counts['err'] += 1
            print(f'  ERR segments {name}\n{traceback.format_exc()}', flush=True)
        # F4 emission factors
        try:
            tgt = os.path.join(hyd, 'emission_factors.csv')
            if not (ONLY_MISSING and os.path.exists(tgt)):
                if P4.process_region(r, name, ember, gres, geom_bad):
                    counts['emission'] += 1
        except Exception:
            counts['err'] += 1
            print(f'  ERR emission {name}\n{traceback.format_exc()}', flush=True)

    print(f'[prep] DONE slope={counts["slope"]} segments={counts["segments"]} '
          f'emission={counts["emission"]} errors={counts["err"]}', flush=True)


if __name__ == '__main__':
    main()
