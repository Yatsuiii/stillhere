#!/usr/bin/env bash
# Paranoia (R2 and R5) end to end on your own machine (cloud review of Nov 2, 12:15 UTC, item 6; docs/RUN_ON_YOUR_GPU.md).
# Fetches the public scans (OpenNeuro ds001338, the fMRIPrep fsaverage5 derivative our scoring read), the stimuli and the atlas, each
# checked against what our run read; makes the nine stock TRIBE v2 forecasts on your GPU with r2/run_r2_forecast.py; then scores them with
# scripts/r2_paranoia.py and scripts/r5_short_scan.py under --verify-own: your forecasts are not held to our pins, the scans must equal
# ours, nothing is written into this repository, and your numbers print beside ours. --no-gpu skips the forecasts and re-scores ours
# (the release asset, checked against the pins) from the public scans.
#
# usage: scripts/reproduce_paranoia.sh [--dry-run] DATA_DIR            4a: your forecasts on your GPU (route 2's setup, plus about 2.6 GB)
#        scripts/reproduce_paranoia.sh [--dry-run] --no-gpu DATA_DIR   4b: ours re-scored; no GPU, no models, no login (about 2.9 GB)
#   DATA_DIR outside this repository. Scoring needs requirements-score.txt (checked before anything is fetched).
# environment:
#   STILLHERE_MODELS_DIR    the models folder scripts/judge_gpu_setup.sh made (holding hub/); default DATA_DIR/models
#   STILLHERE_TRIBE_PYTHON  the TRIBE environment's python (judge_gpu_setup.sh's venv); default: python
#   STILLHERE_SCORE_PYTHON  a python with requirements-score.txt installed, for scoring; default: python3
#   STILLHERE_WORDS         ours (default: our word tables as stim.tsv, no speech-to-text) or asr (TRIBE's own speech-to-text,
#                           uvx whisperx, online; then GPT-2 small surprisal is computed from your words, also online)
#   STILLHERE_RELEASE_URL   where our word tables, surprisal and forecasts are published (default: the paranoia-forecasts-v1 release)
# Prints sizes, counts, hashes, timings and the scorers' lines; never word text. Research use; not a diagnosis or medical advice.
set -euo pipefail

usage() { sed -n '9,19p' "$0" | sed 's/^# \{0,1\}//'; }
DRY=0; NO_GPU=0; DATA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --no-gpu) NO_GPU=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    *) DATA=$1 ;;
  esac
  shift
done
[ -n "$DATA" ] || { usage >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
DATA=$(realpath -m "$DATA")
case "$DATA/" in "$REPO"/*) echo "refused: DATA_DIR must be outside this repository (results/ holds our scored record)" >&2; exit 2 ;; esac
MODELS=${STILLHERE_MODELS_DIR:-$DATA/models}
TRIBE_PY=${STILLHERE_TRIBE_PYTHON:-python}
SCORE_PY=${STILLHERE_SCORE_PYTHON:-python3}
WORDS=${STILLHERE_WORDS:-ours}
case "$WORDS" in ours|asr) ;; *) echo "STILLHERE_WORDS must be ours or asr" >&2; exit 2 ;; esac
F="$REPO/scripts/paranoia_files.py"

if [ "$DRY" = 0 ]; then                       # first, before any network use: the scorers' packages
  "$SCORE_PY" -c "import numpy, scipy, nibabel, nilearn, soundfile, pandas, sklearn" 2>/dev/null || {
    echo "refused before any download: $SCORE_PY lacks the scorers' packages. python3 -m venv .venv-score &&" \
         ".venv-score/bin/pip install -r requirements-score.txt, then STILLHERE_SCORE_PYTHON=.venv-score/bin/python" >&2; exit 2; }
fi
if [ "$NO_GPU" = 1 ]; then python3 "$F" sizes "$DATA" --no-gpu; else python3 "$F" sizes "$DATA"; fi
[ "$DRY" = 1 ] && { echo "dry run: nothing fetched"; exit 0; }
mkdir -p "$DATA"

echo "== stimuli, full text and atlas"
python3 "$F" fetch-public "$DATA"

if [ "$NO_GPU" = 1 ]; then
  OWN="$DATA/ours_release"
  echo "== our forecasts, word tables and surprisal from the release (checked against the pins)"
  python3 "$F" fetch-release "$OWN" --forecasts
else
  OWN="$DATA/r2/forecast/out"
  [ -d "$MODELS/hub" ] || { echo "refused: no models in $MODELS/hub (run scripts/judge_gpu_setup.sh, or set STILLHERE_MODELS_DIR)" >&2; exit 2; }
  echo "== the harness: tribe_common from the R2 kernel (checked), the runner and the Alice harness"
  python3 "$F" common "$DATA"
  python3 "$F" layout "$DATA"
  mkdir -p "$OWN"
  if [ -f "$OWN/WORDS" ] && [ "$(cat "$OWN/WORDS")" != "$WORDS" ]; then   # forecasts from one word source scored with the other's tables
    echo "refused: $OWN holds forecasts made with STILLHERE_WORDS=$(cat "$OWN/WORDS"); use a new DATA_DIR for $WORDS" >&2; exit 2
  fi
  echo "$WORDS" > "$OWN/WORDS"
  if [ "$WORDS" = ours ]; then
    python3 "$F" fetch-release "$OWN"
    python3 "$F" seed-words "$DATA" "$OWN"
    export HF_HUB_OFFLINE=1
  fi
  echo "== nine stock forecasts (audio, full, shuffled; story1-3)"
  for s in story1 story2 story3; do
    for m in audio full shuf; do
      st="stock_${s}_${m}"
      if [ -f "$OWN/stage_$st.json" ] && [ -f "$OWN/pred_$st.npz" ]; then echo "$st: done earlier"; continue; fi
      HF_HOME="$MODELS" HF_HUB_CACHE="$MODELS/hub" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$TRIBE_PY" "$DATA/r2/run_r2_forecast.py" "$st"
    done
  done
  if [ "$WORDS" = asr ]; then
    echo "== GPT-2 small surprisal of your word tables"
    "$SCORE_PY" "$REPO/scripts/r2_paranoia.py" surprisal --root "$DATA" --verify-own "$OWN"
  fi
fi

echo "== the scans: 132 GIFTIs and 3 events files (sizes as r2/data_check.json; each must equal our download)"
"$SCORE_PY" "$REPO/scripts/r2_paranoia.py" download --root "$DATA" --verify-own "$OWN"

echo "== scoring R2 and R5 on your forecasts (nothing written into results/)"
mkdir -p "$DATA/score"
"$SCORE_PY" "$REPO/scripts/r2_paranoia.py" score --root "$DATA" --verify-own "$OWN" --out "$DATA/score"
"$SCORE_PY" "$REPO/scripts/r5_short_scan.py" "$DATA" --verify-own "$OWN" --out "$DATA/score"
echo "done: your results are in $DATA/score"
