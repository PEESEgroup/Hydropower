"""revub_geo.py - build & cache the 88-region master geometry (EPSG:4326).

Reuses build_interconnect_map.find_shp + TRADING_BLOCS to resolve each region key
to its local SWAT study-area shapefile, dissolves to one (multi)polygon per region,
simplifies for publication, and caches to data/regions_4326.gpkg.
Callers reproject to revub_style.EQUAL_EARTH at plot time.
"""
import os
import sys
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # so we can import build_interconnect_map
from build_interconnect_map import find_shp, TRADING_BLOCS   # noqa: E402

CACHE = os.path.join(os.environ.get('REVUB_FIG_DATA') or os.path.join(HERE, 'data'), 'regions_4326.gpkg')

# Plain-language state/area each opaque US grid-region (ISO/RTO/planning region) covers,
# derived by overlaying the region polygons with US states (dominant states by area).
# Used to clarify ISO acronyms in Fig1b/Fig1c labels.
US_STATE_LABEL = {
    'usa_caiso': 'California', 'usa_ercot': 'Texas', 'usa_frcc': 'Florida',
    'usa_nyiso': 'New York', 'usa_isone': 'New England',
    'usa_miso_north': 'MN-WI-IA', 'usa_miso_central': 'MI-IL-MO', 'usa_miso_south': 'LA-AR-MS',
    'usa_pjm_east': 'Mid-Atlantic', 'usa_pjm_west': 'N. Illinois',
    'usa_spp_north': 'NE-SD-ND', 'usa_spp_south': 'KS-OK', 'usa_sertp': 'GA-AL-NC',
    'usa_northerngrid_east': 'MT-ID', 'usa_northerngrid_south': 'NV-UT',
    'usa_northerngrid_west': 'OR-WA',
    'usa_westconnect_north': 'CO-WY', 'usa_westconnect_south': 'AZ-NM',
}


def load_regions(rebuild=False, simplify_deg=0.05, verbose=True):
    if (not rebuild) and os.path.exists(CACHE):
        return gpd.read_file(CACHE)
    rows, geoms, missing = [], [], []
    for bloc, regions in TRADING_BLOCS.items():
        for r in regions:
            shp = find_shp(r)
            if not shp:
                missing.append(r)
                continue
            g = gpd.read_file(shp)
            if g.crs is None:
                g.set_crs('EPSG:4326', inplace=True)
            g = g.to_crs('EPSG:4326')
            try:
                geom = g.geometry.union_all()
            except AttributeError:
                geom = g.geometry.unary_union
            rows.append({'region': r, 'bloc': bloc})
            geoms.append(geom)
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs='EPSG:4326')
    if simplify_deg:
        gdf['geometry'] = gdf.geometry.simplify(simplify_deg)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    gdf.to_file(CACHE, driver='GPKG')
    if verbose:
        print('regions built:', len(gdf))
        if missing:
            print('MISSING geometry (flow-only / transit):', missing)
    return gdf


if __name__ == '__main__':
    g = load_regions(rebuild=True)
    print(g.groupby('bloc').size().to_string())
