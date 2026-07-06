"""Standalone legend images for manual compositing (kept out of the data panels).

  out/fig1a_legend.*  - coverage colorbar + 'No data center' swatch + DC-demand bubble sizes
  out/fig1b_legend.*  - income-group colour families (shade = sub-region)
All text Arial 7 pt; PNG 300 ppi; exact figsize.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import revub_style as S
from fig1a_coverage_map import truncate_cmap, BUBBLE_FC, BUBBLE_EC, NO_DC, OUT
from fig1b_voronoi import FAMILY


def fig1a_legend():
    cmap = truncate_cmap('YlGn', 0.0, 0.65)
    norm = Normalize(0, 100)
    fig = plt.figure(figsize=(72 * S.MM, 72 * S.MM))

    cax = fig.add_axes([0.08, 0.91, 0.55, 0.035])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=[0, 50, 100])
    cb.set_label('Firm coverage (%)')
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)

    axn = fig.add_axes([0.08, 0.60, 0.8, 0.08])
    axn.set_axis_off()
    axn.legend(handles=[mpatches.Patch(facecolor=NO_DC, ec='0.6', lw=0.3, label='No data center')],
               frameon=False, loc='center left', handlelength=1.2, handletextpad=0.6)

    axl = fig.add_axes([0.05, 0.02, 0.9, 0.50])
    axl.set_axis_off()
    dmax, smax = 42245.0, 320.0
    handles = [mlines.Line2D([], [], marker='o', ls='', mfc=BUBBLE_FC, mec=BUBBLE_EC, alpha=0.9,
               markersize=np.sqrt(smax * v / dmax), label=f'{v:,}') for v in (40000, 5000, 500)]
    axl.legend(handles=handles, title='DC demand (MW)', frameon=False, loc='center left',
               labelspacing=2.0, borderpad=0.4, handletextpad=1.2)

    S.save(fig, os.path.join(OUT, 'fig1a_legend'))


def fig1b_legend():
    fig = plt.figure(figsize=(95 * S.MM, 22 * S.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    groups = [('Developed', 'developed'), ('Emerging', 'emerging'),
              ('Developing', 'developing'), ('Other', 'other')]
    for i, (lab, key) in enumerate(groups):
        x0 = 0.04 + i * 0.245
        cm = plt.get_cmap(FAMILY[key])
        for j, p in enumerate(np.linspace(0.40, 0.85, 5)):
            ax.add_patch(mpatches.Rectangle((x0 + j * 0.026, 0.45), 0.026, 0.26,
                         facecolor=cm(p), ec='none'))
        ax.text(x0 + 0.065, 0.34, lab, ha='center', va='top', fontsize=7)
    ax.text(0.02, 0.97, 'Income group  (shade = sub-region; darker = larger)',
            ha='left', va='top', fontsize=7)
    S.save(fig, os.path.join(OUT, 'fig1b_legend'))


if __name__ == '__main__':
    fig1a_legend()
    fig1b_legend()
    print('saved out/fig1a_legend.*  out/fig1b_legend.*')
