#!/usr/bin/env python3
"""Radar chart: existing hydropower vs alternative firm-power options for 24/7 AI data centers.

Sized and styled to match a single Fig. 6 sub-panel: 70 x 74.25 mm, Arial 7 pt, 300 dpi,
via revub_style (S.fig_mm / S.save). Reads radar_matrix.json (12 dimensions, metric-based;
see SOURCES.md / VERIFICATION.md). Existing hydropower is highlighted (filled).
Usage:  python plot_radar.py
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # nature_figures/ on path
import revub_style as S                       # locks Arial 7 pt, 300 dpi, mm sizing

DIMS = ["tech_maturity", "installed_capacity", "permitting_leadtime", "cost", "firmness",
        "grid_flexibility", "ride_through", "low_carbon", "energy_security",
        "developing_world_suitability", "siting_flexibility", "social_acceptance"]
# short labels for panel size (full names in SOURCES.md / caption)
DIM_LABEL = {
    "tech_maturity": "Maturity", "installed_capacity": "Capacity",
    "permitting_leadtime": "Permitting", "cost": "Low cost", "firmness": "Firm",
    "grid_flexibility": "Flexibility", "ride_through": "Ride-\nthrough",
    "low_carbon": "Low\ncarbon", "energy_security": "Security",
    "developing_world_suitability": "Developing", "siting_flexibility": "Siting",
    "social_acceptance": "Acceptance",
}
STYLE = [   # keyword, colour, display
    ("hydro",     "#15607a", "Existing hydropower"),
    ("nuclear",   "#c0392b", "Nuclear (incl. SMR)"),
    ("geotherm",  "#e08a1e", "Geothermal (incl. EGS)"),
    ("long",      "#2a9d8f", "Wind+solar + LDES"),
    ("batt",      "#6aa84f", "Wind+solar + 4h battery"),
    ("gas",       "#9aa0a6", "Unabated gas (CCGT)"),
]
def style_for(label):
    l = label.lower()
    for i, (k, c, d) in enumerate(STYLE):
        if k in l:
            return i, c, d
    return 99, "#bbb", label

def main():
    matrix = json.load(open(os.path.join(HERE, "radar_matrix.json")))["matrix"]
    N = len(DIMS)
    ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); ang += ang[:1]

    fig = plt.figure(figsize=(70*S.MM, 74.25*S.MM))
    ax = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(left=0.23, right=0.77, top=0.99, bottom=0.30)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_facecolor("white")
    ax.set_ylim(0, 1.0)

    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(np.linspace(0, 2*np.pi, 160), [r]*160, color="#e6e9ec", lw=0.4, zorder=0)
    for a in ang[:-1]:
        ax.plot([a, a], [0, 1.0], color="#e6e9ec", lw=0.4, zorder=0)
    ax.set_yticklabels([]); ax.set_xticks([]); ax.spines["polar"].set_visible(False)

    for a, d in zip(ang[:-1], DIMS):
        ha = "center"
        if 0.15 < a < np.pi-0.15: ha = "left"
        elif np.pi+0.15 < a < 2*np.pi-0.15: ha = "right"
        ax.text(a, 1.13, DIM_LABEL[d], fontsize=5.4, ha=ha, va="center", color="#23282d",
                linespacing=0.9)

    rows = sorted(matrix, key=lambda r: -style_for(r["technology"])[0])  # hydro on top
    handles = []
    for row in rows:
        i, col, disp = style_for(row["technology"])
        vals = [float(row[d]) for d in DIMS]; vals += vals[:1]
        hyd = i == 0
        ax.plot(ang, vals, color=col, lw=1.4 if hyd else 0.8, zorder=8 if hyd else 5,
                solid_joinstyle="round", alpha=1 if hyd else 0.9)
        if hyd:
            ax.fill(ang, vals, color=col, alpha=0.20, zorder=3)
        ax.scatter(ang[:-1], vals[:-1], s=7 if hyd else 2.5, color=col,
                   zorder=9 if hyd else 6, edgecolors="white", linewidths=0.3 if hyd else 0)
        handles.append((i, Line2D([0],[0], color=col, lw=1.4 if hyd else 0.9, label=disp)))
    handles = [h for _, h in sorted(handles, key=lambda x: x[0])]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=2, frameon=False, fontsize=5.2, handlelength=1.3,
                    columnspacing=1.0, labelspacing=0.35, handletextpad=0.4)
    leg.get_texts()[0].set_fontweight("bold")

    out = os.path.join(os.path.dirname(HERE), "out")   # nature_figures/out (with the Fig.6 panels)
    os.makedirs(out, exist_ok=True)
    S.save(fig, os.path.join(out, "fig_radar_firmpower"))    # SVG + PNG, 300 dpi, exact mm
    fig.savefig(os.path.join(out, "fig_radar_firmpower.pdf"), bbox_inches=None, transparent=True)  # vector for SI
    print("saved", os.path.join(out, "fig_radar_firmpower") + ".svg / .png / .pdf  (70 x 74.25 mm)")

if __name__ == "__main__":
    main()
