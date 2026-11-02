#!/usr/bin/env bash
# Stages our Paranoia release asset (cloud review of Nov 2, 12:15 UTC, item 6): the nine stock forecasts, the three word tables and the
# surprisal file our R2 and R5 scoring read, each checked against the pins committed in scripts/r2_paranoia.py, with SHA256SUMS and
# README_RELEASE.md (TRIBE v2 CC BY-NC 4.0 attribution, ds001338 credit). It does not upload; the upload is a separate, deliberate step:
#   gh release create paranoia-forecasts-v1 STAGE_DIR/* --title "Paranoia forecasts (stock TRIBE v2)" --notes-file STAGE_DIR/README_RELEASE.md
# usage: scripts/make_paranoia_release.sh FORECAST_DIR STAGE_DIR [--copy]
#   FORECAST_DIR holds pred_stock_*.npz and words_story*.tsv (r2/forecast/out of the scoring root); its root's r2/surprisal.json is
#   staged too when present. STAGE_DIR gets symbolic links by default (it may sit in RAM); --copy copies.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SRC=$1; OUT=$2; shift 2
SURP=$(realpath -m "$SRC/../../surprisal.json")
if [ -f "$SURP" ]; then
  exec python3 "$(dirname "$0")/paranoia_files.py" stage-release "$SRC" "$OUT" --surprisal "$SURP" "$@"
fi
exec python3 "$(dirname "$0")/paranoia_files.py" stage-release "$SRC" "$OUT" "$@"
