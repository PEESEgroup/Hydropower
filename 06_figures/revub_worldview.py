"""revub_worldview.py - Natural Earth China point-of-view (worldview) basemap + matching
region reassignment.

Natural Earth v5+ (Dec 2021) ships per-viewpoint admin_0 country files because the de-facto
"line of actual control" view is not appropriate in every legal jurisdiction. We use the
China viewpoint (ne_10m_admin_0_countries_chn): South Tibet (Arunachal Pradesh) and Taiwan are
part of China, Aksai Chin is China, etc. The POV variant exists only at 10m, so we simplify it
for a clean small-multiple basemap.

To keep the COLOURED REVUB regions consistent with the basemap, South Tibet is moved from
india_northeast -> china_sw using the EXACT area Natural Earth reassigns between the default and
China viewpoints, i.e. India(default) - India(china).
"""
import os
import geopandas as gpd
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
GIS = os.path.join(HERE, 'gis')
CHN = os.path.join(GIS, 'ne_10m_admin_0_countries_chn.shp')        # China viewpoint
DEFAULT10 = os.path.join(GIS, 'ne_10m_admin_0_countries.shp')      # de-facto viewpoint (for diff)

_ST = None


def _uall(gs):
    try:
        return gs.union_all()
    except AttributeError:
        return gs.unary_union


def south_tibet():
    """Land Natural Earth reassigns from India to China between the default and China viewpoints
    (= South Tibet / Arunachal, plus tiny NW slivers). Cached."""
    global _ST
    if _ST is None:
        chn = gpd.read_file(CHN)
        dfl = gpd.read_file(DEFAULT10)
        ia = _uall(dfl[dfl['ADMIN'] == 'India'].geometry)
        ic = _uall(chn[chn['ADMIN'] == 'India'].geometry)
        _ST = ia.difference(ic)
    return _ST


def load_basemap(simplify_deg=0.06):
    """China-viewpoint world basemap (EPSG:4326), Antarctica dropped, simplified, index reset.
    Drop-in for: gpd.read_file(NE).query("CONTINENT != 'Antarctica'").reset_index(drop=True)."""
    w = gpd.read_file(CHN).query("CONTINENT != 'Antarctica'").reset_index(drop=True)
    if simplify_deg:
        w['geometry'] = w.geometry.simplify(simplify_deg)
    return w


def apply_china_worldview(regions):
    """Move South Tibet from india_northeast -> china_sw so the coloured regions match the
    China-viewpoint basemap. `regions` is the load_regions() GeoDataFrame (EPSG:4326)."""
    g = regions.copy()
    st = south_tibet()
    idx = g.index[g['region'] == 'india_northeast']
    cdx = g.index[g['region'] == 'china_sw']
    if len(idx) and len(cdx):
        ine = g.at[idx[0], 'geometry']
        piece = ine.intersection(st)
        cut = ine.difference(st)
        # `difference` leaves thin slivers of india_northeast wherever the SWAT region pokes past
        # the NE-derived South-Tibet polygon (the two McMahon lines come from different sources).
        # Keep only India's main body; fold every stray sliver into China too, so no boundary line
        # is drawn across South Tibet. Buffer-close then welds hairline seams in china_sw.
        polys = [p for p in (cut.geoms if cut.geom_type == 'MultiPolygon' else [cut]) if not p.is_empty]
        polys.sort(key=lambda p: p.area, reverse=True)
        main, slivers = polys[0], polys[1:]
        piece = unary_union([piece] + slivers)
        g.at[cdx[0], 'geometry'] = g.at[cdx[0], 'geometry'].union(piece).buffer(0.03).buffer(-0.03)
        g.at[idx[0], 'geometry'] = main.buffer(0)
    return g
