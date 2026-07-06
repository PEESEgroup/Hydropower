"""revub_names.py - canonical human-readable region labels (no acronyms), shared by all figures.

US grid regions (ISO/RTO/planning) -> plain geography (PJM East -> Mid-Atlantic, CAISO ->
California, MISO North -> Upper Midwest, ...); China/India/Brazil grid regions -> country +
compass; Malaysia East/West; everything else -> Title Case of the key. One source of truth so
every figure labels the same region the same way.
"""

# US grid regions -> plain geographic names (no ISO/RTO acronyms)
US_FULL = {'usa_caiso': 'California', 'usa_ercot': 'Texas', 'usa_frcc': 'Florida',
           'usa_nyiso': 'New York', 'usa_isone': 'New England', 'usa_pjm_east': 'Mid-Atlantic',
           'usa_pjm_west': 'Northern Illinois', 'usa_miso_north': 'Upper Midwest',
           'usa_miso_central': 'Central Midwest', 'usa_miso_south': 'Lower Mississippi',
           'usa_spp_north': 'Northern Plains', 'usa_spp_south': 'Southern Plains',
           'usa_sertp': 'Southeast US', 'usa_northerngrid_east': 'Northern Rockies',
           'usa_northerngrid_south': 'Great Basin', 'usa_northerngrid_west': 'Pacific Northwest',
           'usa_westconnect_north': 'Central Rockies', 'usa_westconnect_south': 'Desert Southwest'}
CA_FULL = {'canada_bc': 'British Columbia', 'canada_prairies': 'Prairies',
           'canada_ontario': 'Ontario', 'canada_quebec': 'Quebec', 'canada_atlantic': 'Atlantic Canada'}
CN_DIR = {'nc': 'North', 'ec': 'East', 'cc': 'Central', 'nw': 'Northwest',
          'ne': 'Northeast', 'sw': 'Southwest', 'csg': 'South'}
IN_DIR = {'north': 'North', 'northeast': 'Northeast', 'east': 'East', 'south': 'South', 'west': 'West'}
BR_DIR = {'norte': 'North', 'nordeste': 'Northeast', 'sudeste': 'Southeast', 'sul': 'South'}
MY_DIR = {'east': 'East', 'west': 'West'}


def region_label(r):
    """Canonical no-acronym label for a REVUB region key."""
    if r in US_FULL:
        return US_FULL[r]
    if r in CA_FULL:
        return CA_FULL[r]
    if r.startswith('china_'):
        d = r.split('_', 1)[1]
        return 'China ' + CN_DIR.get(d, d.capitalize())
    if r.startswith('india_'):
        d = r.split('_', 1)[1]
        return 'India ' + IN_DIR.get(d, d.capitalize())
    if r.startswith('brazil_'):
        d = r.split('_', 1)[1]
        return 'Brazil ' + BR_DIR.get(d, d.capitalize())
    if r.startswith('malaysia_'):
        d = r.split('_', 1)[1]
        return 'Malaysia ' + MY_DIR.get(d, d.capitalize())
    return r.replace('_', ' ').title()


def region_label_us(r):
    """region_label with a ' (US)' tag appended for US subregions - for auto-drawn labels
    (Fig 1b reference, Fig 1c) where the US geography (California, Texas, Pacific Northwest…)
    should be flagged as US. Matches the local rlabel() convention used in Fig 6."""
    if not str(r).startswith('usa_'):
        return region_label(r)
    s = region_label(r)
    if s.endswith(' US'):          # 'Southeast US' -> 'Southeast (US)', not 'Southeast US (US)'
        s = s[:-3]
    return s + ' (US)'
