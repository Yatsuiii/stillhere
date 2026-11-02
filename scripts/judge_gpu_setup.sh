#!/usr/bin/env bash
# Sets up TRIBE v2 on your own NVIDIA GPU for StillHere (cloud review of Nov 2, 12:15 UTC, item 3; docs/RUN_ON_YOUR_GPU.md).
#
# usage: bash scripts/judge_gpu_setup.sh <models_dir>
#   1. a Python 3.11 venv ($STILLHERE_TRIBE_VENV, default .venv-tribe in the repository) from requirements-tribe.txt, plus spaCy's
#      en_core_web_lg 3.8.0 (neuralset would otherwise download it on first use)
#   2. the three models at worker/link_models.py's pinned revisions, into <models_dir>/download (about 9.5 GB). Llama 3.2 3B is gated:
#      the script checks your access before downloading anything. With STILLHERE_HF_SOURCE=<an existing Hugging Face hub folder that
#      already holds those revisions> nothing is downloaded.
#   3. worker/link_models.py: <models_dir>/worker/hub holds only the files the worker loads (hard links, so <models_dir> must be on one
#      file system), then its audit.
# Everything after the download runs with HF_HUB_OFFLINE=1. Prints the environment lines the other scripts read. Never prints a token.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: bash scripts/judge_gpu_setup.sh <models_dir>"; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
MODELS=$(realpath -m "$1")
case "$MODELS/" in "$REPO"/*) echo "put <models_dir> outside the repository (about 9.5 GB)"; exit 2;; esac
mkdir -p "$MODELS"
VENV=${STILLHERE_TRIBE_VENV:-$REPO/.venv-tribe}
EN_CORE="https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
export HF_HUB_DISABLE_TELEMETRY=1

free_gb() { df -Pk "$1" | awk 'NR == 2 {print int($4 / 1048576)}'; }
need() {                                       # stop before a long download that cannot fit, rather than in the middle of it
  local dir=$1 gb=$2 what=$3 have; mkdir -p "$dir"; have=$(free_gb "$dir")
  [ "$have" -ge "$gb" ] || { echo "not enough disk for $what: $have GB free under $dir, about $gb GB needed"; exit 1; }
}
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/   GPU: /' \
  || echo "   no nvidia-smi on PATH: the forecasts need an NVIDIA GPU and driver (route 4b, --no-gpu, does not)"
[ -x "$VENV/bin/python" ] || need "$(dirname "$VENV")" 8 "the TRIBE environment (about 7 GB)"
[ -n "${STILLHERE_HF_SOURCE:-}" ] || [ -d "$MODELS/download/hub" ] || need "$MODELS" 10 "the models (9.47 GB)"   # a rerun downloads none again

echo "== 1. the TRIBE environment: $VENV"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import tribev2, spacy, en_core_web_lg" 2>/dev/null; then
  echo "   already installed"
elif command -v uv >/dev/null; then
  uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$VENV/bin/python" -r "$REPO/requirements-tribe.txt" "en_core_web_lg @ $EN_CORE"
elif command -v python3.11 >/dev/null; then
  python3.11 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$REPO/requirements-tribe.txt" "en_core_web_lg @ $EN_CORE"
else
  echo "needs Python 3.11 (python3.11 on PATH, or uv: https://docs.astral.sh/uv/)"; exit 1
fi
PY="$VENV/bin/python"
"$PY" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("   torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if ok else "none visible")
if ok:
    print(f"   GPU memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB (the worker was tested on 6 GiB)")
EOF

echo "== 2. the models (worker/link_models.py's pinned revisions)"
mapfile -t SPECS < <("$PY" -c "
import sys; sys.path.insert(0, '$REPO/worker'); import link_models as L
for repo, (rev, files) in L.MODELS.items(): print(repo, rev, *files)")
if [ -n "${STILLHERE_HF_SOURCE:-}" ]; then
  SRC=$(cd "$STILLHERE_HF_SOURCE" && pwd -P); echo "   no download: linking from $SRC"
else
  SRC="$MODELS/download/hub"; mkdir -p "$SRC"
  for spec in "${SPECS[@]}"; do
    read -r repo rev _ <<<"$spec"
    if [ "$repo" = "meta-llama/Llama-3.2-3B" ] && ! "$VENV/bin/hf" download "$repo" config.json --revision "$rev" --cache-dir "$SRC" \
        >/dev/null 2>&1; then
      echo "Llama 3.2 3B is gated: accept Meta's licence at https://huggingface.co/meta-llama/Llama-3.2-3B, log in with" \
           "'$VENV/bin/hf auth login', then run this script again."
      exit 1
    fi
  done
  for spec in "${SPECS[@]}"; do
    read -r repo rev files <<<"$spec"
    echo "   $repo @ ${rev:0:7}"
    # shellcheck disable=SC2086
    "$VENV/bin/hf" download "$repo" $files --revision "$rev" --cache-dir "$SRC" >/dev/null
  done
fi
export HF_HUB_OFFLINE=1

echo "== 3. the worker's model folder: $MODELS/worker"
"$PY" "$REPO/worker/link_models.py" "$SRC" "$MODELS/worker"
"$PY" "$REPO/worker/link_models.py" --audit "$MODELS/worker"

cat <<EOF

Done. The other scripts read:
  export STILLHERE_TRIBE_PYTHON=$PY
  export STILLHERE_MODELS_DIR=$MODELS/worker
Then: $PY worker/tribe_worker.py verify     (docs/RUN_ON_YOUR_GPU.md)
EOF
