"""revub_style.py - Nature-grade style for REVUB figures (matplotlib 3.4.3-safe).

Import at the top of every figure script. Locks fonts/sizes/colormaps/projection
per REVUB_nature_figure_style.md. SVG-first export.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np

# Register a real Arial (not installed system-wide on this box) so figures use Arial,
# not the DejaVu fallback. First existing candidate wins; registers as family 'Arial'.
_ARIAL_CANDIDATES = [
    '/home/cfeng/.conda/envs/pybkb/lib/python3.11/site-packages/geemap/data/fonts/arial.ttf',
    '/home/wenhao/.config/Ultralytics/Arial.ttf',
]
for _fp in _ARIAL_CANDIDATES:
    if os.path.exists(_fp):
        try:
            _fm.fontManager.addfont(_fp)
        except Exception:
            pass

MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE, MAXH = 89 * MM, 120 * MM, 183 * MM, 170 * MM  # inches

# PROJECT SPEC (all figures): every text element is Arial at 7 pt.
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.titlesize': 7, 'legend.title_fontsize': 7,
    'axes.linewidth': 0.5, 'lines.linewidth': 0.75,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'patch.linewidth': 0.25,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150,
})

# Map projection: Robinson (per user - matches the Crowther-2015 reference look; Equal Earth
# made Europe look too small). Robinson is not strictly equal-area, but coverage % is a
# per-region attribute (not read off map area) and DC demand is encoded by bubble area, so
# the aesthetic choice is acceptable here. lon_0=0; NE data is split at +/-180 and no data
# region crosses it, so the full-world basemap does not smear.
MAP_CRS = '+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs'
EQUAL_EARTH = MAP_CRS   # back-compat alias used by figure scripts

WONG = ['#000000', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
BLOC_ORDER = ['china', 'india', 'southeast_asia', 'south_america', 'europe', 'north_america']
BLOC_COLOR = {'china': '#0072B2', 'india': '#E69F00', 'southeast_asia': '#009E73',
              'south_america': '#D55E00', 'europe': '#56B4E9', 'north_america': '#CC79A7'}
INCOME_COLOR = {'developing': '#D55E00', 'emerging': '#E69F00', 'developed': '#0072B2'}
MISSING = '#dddddd'   # no-data grey


def fig_mm(w_mm, h_mm):
    """Figure sized to final print mm (never post-scale)."""
    return plt.subplots(figsize=(w_mm * MM, h_mm * MM))


def panel(ax, letter, x=-0.02, y=1.02):
    """7 pt bold lowercase upright panel letter, top-left (project spec: all text 7 pt)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=7,
            fontweight='bold', va='bottom', ha='left')


def save(fig, path_noext, svg=True, png=True, dpi=300):
    """SVG (vector) + PNG at 300 ppi. bbox=None -> output is EXACTLY the figsize (for
    precise A4-fraction panels); design the layout so content stays inside the canvas."""
    if svg:
        fig.savefig(path_noext + '.svg', bbox_inches=None, dpi=dpi,    # dpi sets rasterized-layer res
                    transparent=True)
    if png:
        fig.savefig(path_noext + '.png', bbox_inches=None, dpi=dpi, transparent=True)


def quantile_bins(values, k):
    """Integer class 0..k-1 per value (equal-count bins, NaN-safe)."""
    v = np.asarray(values, float)
    edges = np.nanquantile(v[~np.isnan(v)], np.linspace(0, 1, k + 1))
    edges[-1] = np.inf
    return np.clip(np.digitize(v, edges[1:-1], right=False), 0, k - 1)
