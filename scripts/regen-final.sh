#!/usr/bin/env bash
# Regenerate every final-matrix table, figure and RESULTS §9 from the frozen runs.
#
# Runs from the pinned bank worktree (its live store and items ARE the freeze) and
# writes into this checkout. Deterministic: fixed seeds, no timestamps in outputs.
#
#   scripts/regen-final.sh            # paths resolve via voxparity.paths (paths.local.yaml)
#   BANK=... MAIN=... REGATE=... scripts/regen-final.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATHS() { (cd "$HERE" && uv run --project "$HERE" python -m voxparity.paths "$1"); }
BANK="${BANK:-$(PATHS bank)}"                    # frozen bank + 20260915-final-* runs
MAIN="${MAIN:-$(PATHS main)}"                    # game imports, trials, review, recordings
REGATE="${REGATE:-$BANK/runs/regate/whisper-l3t-20260915.summary.json}"
OUT="$HERE/docs/results/final"
HUMAN_RUNS="$MAIN/runs/game-20260925/human-*"
TRIALS="$MAIN/runs/web-trials.jsonl"
BUNDLE="$MAIN/web/data/items.json"                 # game bundle (gitignored, lives in MAIN)

vp() { (cd "$BANK" && uv run --project "$HERE" --extra paper voxparity "$@"); }

echo "== analyze freeze (+ RESULTS §9)"
vp analyze freeze --out "$OUT" \
  --template "$OUT/section9.template.md" --results-md "$HERE/docs/RESULTS.md"

echo "== analyze human"
vp analyze human --human-runs "$HUMAN_RUNS" --trials "$TRIALS" \
  --bundle "$BUNDLE" --out "$OUT"

echo "== analyze paper (+ figures)"
vp analyze paper --human-runs "$HUMAN_RUNS" --out "$OUT"

echo "== analyze gemini-bias"
vp analyze gemini-bias --out "$OUT/gemini_bias.json"

echo "== analyze crossjudge (gpt-audio-mini = primary, gpt-audio = second judge)"
vp analyze crossjudge --human-runs "$HUMAN_RUNS" \
  --xjudge runs/crossjudge/openai-gpt-audio-mini.jsonl --out "$OUT/crossjudge.json"
vp analyze crossjudge --human-runs "$HUMAN_RUNS" \
  --xjudge runs/crossjudge/openai-gpt-audio.jsonl --out "$OUT/crossjudge-gptaudio.json"

echo "== analyze listeners (spec §12)"
vp analyze listeners --human-runs "$HUMAN_RUNS" --trials "$TRIALS" \
  --bundle "$BUNDLE" --review "$MAIN/runs/review-bhavik.json" \
  --record-meta "$MAIN/stimuli/human-recordings/metadata.jsonl" \
  --out "$OUT/listeners.json" --profile-out "$BANK/runs/listener-profile.json"

echo "== exp-analyze (closability)"
vp exp-analyze --out-dir "$HERE/docs/results/exp"

echo "== local Whisper re-gate summary"
if [ -f "$REGATE" ]; then
  cp "$REGATE" "$OUT/regate-whisper.json"
else
  echo "   (no re-gate summary at $REGATE; keeping the committed copy)"
fi

echo "regen-final: done -> $OUT"
