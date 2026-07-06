"""
F1_prepare_lithology_b.py - assign each station a sediment rating exponent b
(low / central / high) from real point lithology, for STCI (§3.1).

Lithology source: Macrostrat point API (https://macrostrat.org/api), queried at
each dam's lat/lon - real geological-map lithology, no bulk download. Stations
outside Macrostrat coverage fall back to the default b range and are flagged.

Lithology → b mapping - GLOBAL BAND + modest endpoint nudge only.
Conclusion of a verified deep-literature review (see md §3.1): lithology is a WEAK
predictor of the rating exponent b. The strongest GLOBAL predictors are climatic/
morphologic - air temperature (inverse) and basin relief (positive), Syvitski et al.
(2000, WRR 36, 2747-2760); lithology was not even a tested predictor there. No
published transferable "lithology class → b" lookup exists (the only quantitative
coefficient, Li et al. 2025 WRR karst PLS-SEM lithology→b = -0.42, is karst-internal).
b also varies strongly WITHIN a class (karst CV 24-28%; segmented low/high-flow
exponents at 52/62 stations, Hoffmann et al. 2020). And the supply→b relation is
NON-MONOTONIC: hyper-supply loess FLATTENS the SSC-Q curve → LOW b (Zhang 2021;
Tang; PMC9330806), so "more erodible → higher b" would MISASSIGN loess.

Therefore b defaults to the literature global band 1.5 / 2.0 / 2.5 (LOAD-exponent
convention Qs=a·Q^b; matches karst mean 1.90-2.04, Li et al. 2025) for ALL stations.
Only the two DIRECTIONALLY-VERIFIED endpoints (Bywater-Reyes 2017, within-method:
sedimentary highest b, diabase lowest) get a modest ±0.2 central nudge. loess/
unconsolidated, volcanic, carbonate are NOT differentiated (non-monotonic / no data /
≈band central). STCI is reported as the low/central/high band regardless.

  clastic sedimentary           friable, supply-rich (↑ verified)  1.7 / 2.2 / 2.7
  crystalline (plutonic+metam)  resistant, supply-ltd (↓ verified) 1.3 / 1.8 / 2.3
  unconsolidated                non-monotonic (loess may be LOW)   1.5 / 2.0 / 2.5
  volcanic / volcaniclastic     no published b basis               1.5 / 2.0 / 2.5
  carbonate                     karst ≈ band central (Li 2025)     1.5 / 2.0 / 2.5
  default (no coverage)         global band                        1.5 / 2.0 / 2.5

Output: <hydro_dir>/sediment_params.csv
  name_unified, lat, lon, litho_class, litho_raw, litho_source, b_low, b_central, b_high

CLI:  python F1_prepare_lithology_b.py <region_path> [gcm] [ssp] [tag]
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd

import F_impact_common as C

B_MAP = {                                   # (b_low, b_central, b_high) - global band + endpoint nudge
    'clastic_sed':    (1.7, 2.2, 2.7),      # friable, supply-rich - verified ↑ (Bywater-Reyes 2017)
    'crystalline':    (1.3, 1.8, 2.3),      # resistant, supply-limited - verified ↓ (diabase)
    'unconsolidated': (1.5, 2.0, 2.5),      # NON-monotonic (loess may flatten → low b): no nudge
    'volcanic':       (1.5, 2.0, 2.5),      # no published b basis (basalt≠volcaniclastic): no nudge
    'carbonate':      (1.5, 2.0, 2.5),      # karst mean ≈ band central (Li 2025): no nudge
    'default':        (1.5, 2.0, 2.5),      # literature global band
}

# keyword → class, evaluated in this priority order
_RULES = [
    ('unconsolidated', ['alluv', 'unconsolidat', 'colluv', 'loess', 'evaporit',
                        'gypsum', ' salt', 'halite', 'glacial', ' till', 'eolian',
                        'aeolian', 'fluvial', 'lacustrine', 'regolith', 'soil']),
    ('carbonate',      ['limestone', 'dolomit', 'carbonate', 'chalk', 'marl', 'calcite', 'travertine']),
    ('crystalline',    ['granit', 'granodiorit', 'diorit', 'gabbro', 'tonalit', 'syenit',
                        'plutonic', 'intrusive', 'gneiss', 'schist', 'granulit', 'charnockit',
                        'quartzit', 'migmatit', 'phyllit', 'amphibolit', 'metamorphic',
                        'metased', 'metavolcan', 'slate', 'marble', 'eclogite', 'anorthosite',
                        'peridotit', 'pegmatit', 'crystalline']),
    ('volcanic',       ['basalt', 'andesit', 'rhyolit', 'dacit', 'volcanic', 'tuff', 'lava',
                        'pyroclast', 'ignimbrit', 'trachyt', 'obsidian', 'felsic', 'mafic']),
    ('clastic_sed',    ['sandstone', 'shale', 'mudstone', 'siltstone', 'conglomerat', 'clastic',
                        'greywacke', 'graywacke', 'turbidit', 'arkose', 'sedimentary', 'flysch',
                        'molasse', 'claystone', 'sand', 'clay', 'silt', 'gravel']),
]


def classify_litho(text):
    s = (text or '').lower()
    for cls, kws in _RULES:
        if any(k in s for k in kws):
            return cls
    return 'default'


def macrostrat_litho(lat, lon, retries=2, pause=0.3):
    """Return (litho_text, source) at a point, or (None, reason)."""
    url = 'https://macrostrat.org/api/v2/geologic_units/map?' + \
        urllib.parse.urlencode({'lat': lat, 'lng': lon})
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'REVUB-impact/1.0'})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            recs = data.get('success', {}).get('data', [])
            if not recs:
                return None, 'no_coverage'
            # prefer the first record with a non-empty lith description
            for rec in recs:
                lith = (rec.get('lith') or '').strip()
                if lith:
                    return lith, 'macrostrat'
            return None, 'empty_lith'
        except Exception as e:
            if attempt < retries:
                time.sleep(pause * (attempt + 1))
                continue
            return None, f'error:{type(e).__name__}'
    return None, 'error'


def run(region_path=None, gcm=None, ssp=None, tag=None):
    cfg = C.resolve_config(region_path, gcm, ssp, tag)
    _, meta = C.load_scenarios(cfg['output_dir'], include_e=False)
    names = meta['HPP_name']
    static = C.load_static(cfg['hydro_dir'], names, cfg['output_dir'])  # composite (name,lat) join
    #   so duplicate-named stations query Macrostrat at their OWN coords (not the first dam's)

    rows = []
    print(f'[F1 litho-b] {cfg["region_name"]}  {len(names)} stations  (querying Macrostrat)')
    for i, name in enumerate(names):
        lat = static.iloc[i].get('lat_unified')
        lon = static.iloc[i].get('lon_unified')
        if not (np.isfinite(lat) and np.isfinite(lon)):
            cls, raw, src = 'default', '', 'no_coords'
        else:
            raw, src = macrostrat_litho(float(lat), float(lon))
            cls = classify_litho(raw) if raw else 'default'
            time.sleep(0.25)
        bl, bc, bh = B_MAP[cls]
        rows.append({'name_unified': name, 'lat': lat, 'lon': lon,
                     'litho_class': cls, 'litho_raw': (raw or '')[:120],
                     'litho_source': src, 'b_low': bl, 'b_central': bc, 'b_high': bh})
        print(f'  {name[:32]:32s} {cls:14s} b={bc}  [{src}] {(raw or "")[:50]}')

    out = pd.DataFrame(rows)
    path = os.path.join(cfg['hydro_dir'], 'sediment_params.csv')
    out.to_csv(path, index=False)
    n_real = (out['litho_source'] == 'macrostrat').sum()
    print(f'  wrote {path}  ({n_real}/{len(out)} from Macrostrat, '
          f'{len(out) - n_real} default)')
    print('  class counts:', out['litho_class'].value_counts().to_dict())
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0] if len(a) > 0 else None, a[1] if len(a) > 1 else None,
        a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
