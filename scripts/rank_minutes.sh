#!/usr/bin/env bash
# Which minutes of your own story to play (scripts/rank_minutes.py; exploratory, forecast only): stock TRIBE v2's audio-only and full
# forecasts of WAV on your NVIDIA GPU (r2/run_story_forecast.py, R2's harness), then R5's ranking of its 60 s windows. R5 checked this
# ranking once, on Paranoia, as minutes of a whole-story scan; a scan of only the chosen minutes was not tested.
# Needs route 2's setup (scripts/judge_gpu_setup.sh: STILLHERE_TRIBE_PYTHON, STILLHERE_MODELS_DIR, with the Llama 3.2 access step), uv
# for TRIBE v2's speech-to-text (uvx whisperx large-v3, unless STILLHERE_WORDS_TSV gives a word table), and STILLHERE_SCORE_PYTHON with
# requirements-score.txt. A 6 GB GPU: STILLHERE_ASR_BATCH=4. WAV: an English story recording, at least 80 s; stages already done are kept.
#
# usage: scripts/rank_minutes.sh WAV DATA_DIR [PART]   (PART: a short name for the story; default: the file's name)
set -euo pipefail

[ $# -ge 2 ] || { sed -n '9p' "$0" | sed 's/^# //' >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
WAV=$(realpath "$1"); DATA=$(realpath -m "$2")
PART=${3:-$(basename "$WAV" | sed 's/\.[^.]*$//; s/[^A-Za-z0-9_-]/_/g' | cut -c1-40)}
[ -f "$WAV" ] || { echo "refused: no file $WAV" >&2; exit 2; }
case "$DATA/" in "$REPO"/*) echo "refused: DATA_DIR must be outside this repository" >&2; exit 2 ;; esac
MODELS=${STILLHERE_MODELS_DIR:-$DATA/models}
TRIBE_PY=${STILLHERE_TRIBE_PYTHON:-python}
SCORE_PY=${STILLHERE_SCORE_PYTHON:-python3}
F="$REPO/scripts/paranoia_files.py"

"$SCORE_PY" -c "import numpy, scipy, nibabel, nilearn, soundfile" 2>/dev/null || {
  echo "refused: $SCORE_PY lacks the scorers' packages (pip install -r requirements-score.txt, then STILLHERE_SCORE_PYTHON)" >&2; exit 2; }
[ -d "$MODELS/hub" ] || { echo "refused: no models in $MODELS/hub (run scripts/judge_gpu_setup.sh, or set STILLHERE_MODELS_DIR)" >&2; exit 2; }
mkdir -p "$DATA"
echo "== the harness (tribe_common from the R2 kernel, checked) and the atlas"
python3 "$F" common "$DATA"
python3 "$F" layout "$DATA"
python3 "$F" fetch-atlas "$DATA"
OUT="$DATA/stories/$PART"
for m in audio full; do
  if [ -f "$OUT/stage_${PART}_$m.json" ] && [ -f "$OUT/pred_stock_${PART}_$m.npz" ]; then echo "$PART $m: done earlier"; continue; fi
  echo "== $PART: stock TRIBE v2, $m"
  HF_HOME="$MODELS" HF_HUB_CACHE="$MODELS/hub" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$TRIBE_PY" "$DATA/r2/run_story_forecast.py" "$WAV" "$PART" "$m" "$OUT"
done
echo "== R5's ranking of $PART's minutes"
"$SCORE_PY" "$REPO/scripts/rank_minutes.py" rank "$OUT" --atlas "$DATA/atlas" --wav "$PART=$WAV"
