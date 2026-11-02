#!/usr/bin/env bash
# The StillHere page on your own computer, with live forecasts on your own NVIDIA GPU and no API key (cloud review of Nov 2, 12:15 UTC,
# item 5; docs/RUN_ON_YOUR_GPU.md). Run scripts/judge_gpu_setup.sh first.
#
# usage: STILLHERE_TRIBE_PYTHON=... STILLHERE_MODELS_DIR=... bash scripts/run_local_gpu.sh
#   starts web/serve.py on http://127.0.0.1:${STILLHERE_PORT:-8780} with STILLHERE_LOCAL_GPU=1, and `tribe_worker.py serve` against it at
#   any hour. A fresh job token goes into a 0600 file under the state dir ($STILLHERE_STATE_DIR, default
#   ~/.local/state/stillhere-local-gpu), never /tmp, and is never printed. Ctrl-C stops both.
#   web/serve.py runs on $STILLHERE_WEB_PYTHON (default python3, with the repository's requirements.txt installed).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
PORT=${STILLHERE_PORT:-8780}
TRIBE_PY=${STILLHERE_TRIBE_PYTHON:?run scripts/judge_gpu_setup.sh first; it prints STILLHERE_TRIBE_PYTHON and STILLHERE_MODELS_DIR}
: "${STILLHERE_MODELS_DIR:?run scripts/judge_gpu_setup.sh first; it prints STILLHERE_MODELS_DIR}"
WEB_PY=${STILLHERE_WEB_PYTHON:-python3}
STATE=${STILLHERE_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/stillhere-local-gpu}
case "$STATE/" in /tmp/*|/var/tmp/*) echo "keep the state dir out of /tmp (it holds the job token)"; exit 2;; esac
mkdir -p "$STATE/worker"; chmod 0700 "$STATE"
TOKEN_FILE="$STATE/job_token"
(umask 077; "$WEB_PY" -c 'import secrets; print(secrets.token_hex(32))' > "$TOKEN_FILE")
if [ "$(stat -c %a "$TOKEN_FILE")" != 600 ]; then              # e.g. an NTFS or FAT mount that ignores permissions
  rm -f "$TOKEN_FILE"; echo "the state dir's file system ignores permissions; set STILLHERE_STATE_DIR to a folder on a Linux file system"
  exit 2
fi

cleanup() { kill "${WEB_PID:-}" "${WORKER_PID:-}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "$REPO"
STILLHERE_WORKER_TOKEN=$(cat "$TOKEN_FILE") STILLHERE_LOCAL_GPU=1 WORKER_HOURS_ET="any time (your GPU)" \
  "$WEB_PY" web/serve.py --port "$PORT" &
WEB_PID=$!
for _ in $(seq 50); do
  "$WEB_PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/api/status', timeout=1)" 2>/dev/null && break
  sleep 0.2
done

STATE_DIRECTORY="$STATE/worker" STILLHERE_JOBS_URL="http://127.0.0.1:$PORT" STILLHERE_HOURS_ET=00:00-24:00 \
  "$TRIBE_PY" worker/tribe_worker.py serve --token-file "$TOKEN_FILE" &
WORKER_PID=$!

echo "StillHere (local GPU): open http://127.0.0.1:$PORT/ once the worker logs 'engine loaded'. Ctrl-C stops both."
wait -n "$WEB_PID" "$WORKER_PID"
