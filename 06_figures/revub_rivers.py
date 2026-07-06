"""River network layer for Fig 1a from GRFR / MERIT Hydro Vector (MERIT-Basins) reaches.

Source: /home/cfeng/hydro/source_data/grfr/pfaf_<1-9>_.../riv_*.shp (EPSG:4326, ~2.9M reaches).
Field 'order' = Strahler-like stream order 1-9 -> used as the river "level": higher order =
thicker line. Only HIGH-level reaches are drawn (order >= threshold). Two cached tiers,
pre-projected to the map CRS (Robinson):
  data/rivers_world_grfr.gpkg  - order >= WORLD_ORDER, for the world panel
  data/rivers_inset_grfr.gpkg  - order >= INSET_ORDER, clipped to the inset bounding boxes
"""
import os
import glob
import numpy as np
import geopandas as gpd
import pandas as pd

import revub_style as S
from revub_geo import load_regions

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data')
GRFR = sorted(glob.glob('/home/cfeng/hydro/source_data/grfr/*/riv_*.shp'))  # raw source; only used to rebuild the bundled rivers_*_grfr.gpkg if absent

WORLD_ORDER = 6     # world panel: high-level rivers only (stream order 6-9)
INSET_ORDER = 4     # insets: also show order 4-5 tributaries
RIVER_BLUE = '#3b82c4'   # medium blue: thin threads over the green fill (visible, not obscuring)

EUROPE_WIN = (-12, 33, 33, 72)   # lon/lat window for the Europe inset detail tier


def lw_for(order, base=0.04, step=0.13, lo=4):
    """Line width from stream order ('level'): higher order -> thicker. Kept thin overall."""
    return np.clip(base + step * (np.asarray(order, float) - lo), 0.05, 0.85)


def _inset_bboxes():
    """lon/lat (minx,miny,maxx,maxy) boxes for the inset regions, padded a little."""
    g = load_regions()
    boxes = {'europe': EUROPE_WIN}
    for b in ('north_america', 'india', 'china', 'southeast_asia'):
        xmin, ymin, xmax, ymax = g[g['bloc'] == b].total_bounds
        dx, dy = (xmax - xmin) * 0.08, (ymax - ymin) * 0.08
        boxes[b] = (xmin - dx, ymin - dy, xmax + dx, ymax + dy)
    return boxes


def _read_grfr(order_min, bbox=None):
    """Read all pfaf tiles with 'order' >= order_min (and optional bbox); concat in EPSG:4326."""
    where = '"order" >= %d' % order_min
    parts = []
    for p in GRFR:
        sub = gpd.read_file(p, columns=['COMID', 'order'], where=where, bbox=bbox)
        if len(sub):
            parts.append(sub)
    if not parts:
        return gpd.GeoDataFrame(columns=['COMID', 'order', 'geometry'],
                                geometry='geometry', crs='EPSG:4326')
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            geometry='geometry', crs='EPSG:4326')


def _build_world():
    out = os.path.join(DATA, 'rivers_world_grfr.gpkg')
    if os.path.exists(out):
        return out
    print('building rivers_world_grfr.gpkg (order>=%d) ...' % WORLD_ORDER)
    r = _read_grfr(WORLD_ORDER)
    r['geometry'] = r.geometry.simplify(0.02, preserve_topology=False)
    r = r[~r.geometry.is_empty & r.geometry.notna()].to_crs(S.EQUAL_EARTH)
    r.to_file(out, driver='GPKG')
    print('  wrote %d reaches' % len(r))
    return out


def _build_inset():
    out = os.path.join(DATA, 'rivers_inset_grfr.gpkg')
    if os.path.exists(out):
        return out
    print('building rivers_inset_grfr.gpkg (order>=%d, clipped to insets) ...' % INSET_ORDER)
    parts = [_read_grfr(INSET_ORDER, bbox=bb) for bb in _inset_bboxes().values()]
    r = pd.concat(parts, ignore_index=True).drop_duplicates('COMID')
    r = gpd.GeoDataFrame(r, geometry='geometry', crs='EPSG:4326')
    r['geometry'] = r.geometry.simplify(0.005, preserve_topology=False)
    r = r[~r.geometry.is_empty & r.geometry.notna()].to_crs(S.EQUAL_EARTH)
    r.to_file(out, driver='GPKG')
    print('  wrote %d reaches' % len(r))
    return out


def load_rivers_world():
    return gpd.read_file(_build_world())


def load_rivers_inset():
    return gpd.read_file(_build_inset())


if __name__ == '__main__':
    w = load_rivers_world()
    i = load_rivers_inset()
    print('world tier:', len(w), '| inset tier:', len(i),
          '| world orders:', sorted(w['order'].unique()))
