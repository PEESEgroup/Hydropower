#!/usr/bin/env bash
# Reproduce the paper's figure panels from the bundled data (../data).
#
# The paper has 25 data panels across 6 figures (Fig 3a is shown as 5 per-continent
# facets). Headline data: existing-hydropower-only ("nofuture") + GFDL-ESM4/SSP3-7.0
# reference, with the 15-GCM x SSP ensemble where a figure is inherently multi-model.
#
# Usage:   bash reproduce_figures.sh                 # uses `python` on PATH
#          PYTHON=/path/to/python bash reproduce_figures.sh
# Output:  ../figures/*.png  (only the paper panels are kept; intermediates pruned)
#
# Two panel groups are NOT regenerated here and ship pre-rendered:
#   * Fig 5a, 5c (basin-fill maps) need HydroBASINS polygons (free, hydrosheds.org;
#     run fig4_maps_basin.py after placing them under data_external/).
# Fig 4e, 4f (IUCN threatened-fish overlay) are omitted entirely for IUCN licensing.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export REVUB_FIG_DATA="$PWD/../data"
export REVUB_FIG_OUT="$PWD/../figures"
mkdir -p "$REVUB_FIG_OUT"

fail=0
run(){ echo ">> $*"; "$PY" "$@" >/dev/null 2>>"$REVUB_FIG_OUT/_errors.log" || { echo "   !! FAILED: $*"; fail=$((fail+1)); }; }
: > "$REVUB_FIG_OUT/_errors.log"

echo "== Figure 1 =="
run fig1a_coverage_map.py            # 1a  coverage map
run fig1b_scatter.py                 # 1b  firm capacity vs flat-load demand
run fig1c_seasonal.py                # 1c  monthly firm coverage

echo "== Figure 2 =="
run fig2a_hydro_sensitivity.py       # 2a  ELCC change SSP1-2.6 -> SSP5-8.5 (+5-GCM inset)
run fig2b_cooling.py                 # 2b  cooling-load change
run fig2c_attribution.py             # 2c  supply/demand attribution

echo "== Figure 3 =="
REVUB_FIG3A_MODE=panels run fig3a_flows.py   # 3a  net firm trade, per-continent facets
run fig3b_deficit.py                 # 3b  coverage added by interconnection
run fig3c_levers.py                  # 3c  firm capacity added by each lever
run fig3c_ps_heatmap.py              # 3d  net pumped-storage power (hour x month)
run fig3c_cascade_regions.py         # 3e  cascade-coordination gain

echo "== Figure 4 (ecology) + Figure 5 distributions =="
run fig4_maps_points.py              # 4a, 4c  drawdown / reversals point maps
run fig4_strip_typ.py                # 4b, 4d, 5b, 5d  distributions vs balancing grid

echo "== Figure 6 =="
run fig6_panels.py                   # 6b, 6c, 6d, 6e, 6f
run fig_radar/plot_radar.py          # 6a  firm-power radar
[ -f out/fig_radar_firmpower.png ] && cp -f out/fig_radar_firmpower.png "$REVUB_FIG_OUT/fig6a_radar.png"
rm -rf out

# ---- keep only the paper panels (prune intermediates / alternates / legends) ----
PAPER="fig1a_coverage fig1b_scatter fig1c_seasonal \
fig2a_elcc fig2b_cooling fig2c_attribution \
fig3a_panel_china fig3a_panel_europe fig3a_panel_india fig3a_panel_north_america fig3a_panel_southeast_asia \
fig3b_deficit fig3c_levers fig3c_ps_heatmap fig3c_cascade_regions \
fig4_map_drawdown fig4_typ_drawdown fig4_map_reversals fig4_typ_reversals \
fig4_map_irrigationgap fig4_typ_irrigationgap fig4_map_sediment fig4_typ_sediment \
fig6_b fig6_c fig6_d fig6_e fig6_g fig6a_radar"
cd "$REVUB_FIG_OUT"
for f in *.png *.svg; do
  [ -e "$f" ] || continue
  base="${f%.*}"
  case " $PAPER " in *" $base "*) ;; *) rm -f "$f";; esac
done

echo
echo "== DONE: $(ls "$REVUB_FIG_OUT"/*.png 2>/dev/null | wc -l) paper-panel PNGs in figures/  (failures: $fail) =="
echo "   (Fig 5a/5c ship pre-rendered - need HydroBASINS to regenerate; Fig 4e/4f omitted - IUCN)"
[ "$fail" -eq 0 ] || { echo "see figures/_errors.log"; exit 1; }
