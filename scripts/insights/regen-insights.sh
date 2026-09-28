#!/usr/bin/env bash
# Regenerate every insights lens (docs/insights/*.{json,md} + figures) from the
# frozen runs, under ONE scoring switch (D118: first-turn scoring is primary).
#
#   scripts/insights/regen-insights.sh                     # first-turn (primary)
#   VOXPARITY_SCORING_TURN=followup scripts/insights/regen-insights.sh   # sensitivity
#
# Runs from the pinned bank worktree (its items, store and runs ARE the freeze)
# and writes into this checkout. No model calls, no spend: the same-call LLM
# coder reads its cache (or the committed samecall.json) and never calls out.
# Every lens cache is rebuilt in a fresh temp dir so no result is read from a
# cache built under a different scoring mode.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
# Roots: VXP_BANK / VXP_MAIN (or BANK / MAIN), else paths.local.yaml (voxparity.paths).
BANK="${VXP_BANK:-${BANK:-$(cd "$HERE" && uv run --quiet python -m voxparity.paths bank)}}"
MAIN="${VXP_MAIN:-${MAIN:-$(cd "$HERE" && uv run --quiet python -m voxparity.paths main)}}"
INS="$HERE/docs/insights"
export VOXPARITY_SCORING_TURN="${VOXPARITY_SCORING_TURN:-first_turn}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vxp-insights-XXXXXX")"
export BANK MAIN VXP_BANK="$BANK" VXP_MAIN="$MAIN"
export VX_INSIGHT_CACHE="$TMP/human.pkl" VX_REBUILD=1
export PSYCH_CACHE="$TMP/psych.pkl"
export ROBUST_CACHE="$TMP/robust.pkl" ROBUST_REBUILD=1
export VXP_INSIGHTS_CACHE="$TMP/models_cells.json"
HUMAN="$MAIN/runs/game-20260925/human-*"
S="$HERE/scripts/insights"

py() { (cd "$BANK" && uv run --project "$HERE" --extra paper --with scipy python "$@"); }

echo "== scoring: $VOXPARITY_SCORING_TURN (caches in $TMP)"

echo "== atlas"
py "$S/atlas_data.py" --human-runs "$HUMAN" --out "$TMP/atlas_cells.json"
py "$S/atlas_build.py" --cells "$TMP/atlas_cells.json" --bank "$BANK" --out "$INS"
py "$S/atlas_render.py" --atlas "$INS/atlas.json" --out "$INS/atlas.md"
py "$S/atlas_figures.py" --atlas "$INS/atlas.json" --cells "$TMP/atlas_cells.json" \
  --out "$INS/figures"

echo "== harm"
py "$S/harm_analysis.py" --bank "$BANK" --human "$HUMAN"
py "$S/harm_report.py"

echo "== acoustic"
py "$S/acoustic_analysis.py" --bank "$BANK" --main "$MAIN"
py "$S/acoustic_figures.py"

echo "== human"
py "$S/human_insights.py"
py "$S/human_figures.py"

echo "== perception robustness (reanalysis A)"
py "$S/perception_robustness.py"
py "$S/perception_robustness_report.py"

echo "== same-call perception (no LLM calls)"
py "$S/samecall_analysis.py"

echo "== psychometrics"
py "$S/psych_report.py"

echo "== models"
py "$S/models_load.py"
py "$S/models_analysis.py"
py "$S/models_report.py"
py "$S/models_figures.py"

echo "== robustness v2 (reanalysis D)"
py "$S/robust_report.py"

echo "== players band + best-minus-players (paper v3)"
py "$S/players_band.py"

echo "== null test without masked-word cells (Appendix A.5)"
py "$S/noslot.py"

echo "== final analyses (error asymmetry, flip accuracy, admitted cells, generations)"
py "$S/final_analyses.py"

echo "== discovery (probe label, stakes, criterion, menu position, voices)"
py "$S/discovery.py"

echo "regen-insights: done ($VOXPARITY_SCORING_TURN) -> $INS"
