#!/bin/sh
# Muse Code Stop hook: runs the PREREG guard for the git repo Muse is working in, if that repo has one. The guard decides warn or
# hard mode (warn-only until 2026-10-29 00:00 ET); this script passes its exit code through.
top=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[ -f "$top/stillhere/prereg_check.py" ] || exit 0
PYTHONPATH="$top" exec python3 -m stillhere.prereg_check --registry "$top/prereg_registry.json"
