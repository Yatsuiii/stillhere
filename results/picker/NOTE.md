# results/picker: dated note, Nov 1 2026 (cloud review of Oct 31, 23:15 UTC, item 3)

The files in this folder are unchanged since 9748d31; this note only says what two of them are.

- `tierD_wall_baselines.json` is the exploratory run after the scoring call, under `docs/rules/cheap_baselines_rule.md` (its own `label`
  and `rule` fields say so): `scripts/wall_baselines.py` on the same manifest, collected series and forecasts, after the single scoring
  call had finished. In `score_job.sh` it failed to start (rc 1 at 12:54:11 UTC, a relative package path after a change of directory);
  it was then run by hand with the same arguments and the path made absolute (rc 0 at 13:11:22 UTC; `d22/picker/score_job.log`). It
  decides nothing about S1-S4 or the wall.
- `tierD_picker_targets_per_clip.json` (SHA-256 ef8e8880232b12e87e1b7257ec0a1d3cc10fef728c211b5418759641a10922fb) is the scoring call's
  targets file, `tierD_picker_targets.json` (97c3215d..., in `MANIFEST.sha256`, kept outside the repository), with its `per_listener`
  values replaced by a one-line note; `manifest_sha256` and `per_clip` are copied unchanged. Made by this command, run from `d22/picker/`
  with the repository's `results/picker` as the second argument; rerun on Nov 1, it reproduces the committed bytes:

```
python3 - scored results/picker <<'PY'
import json, sys
S, D = sys.argv[1:]
t = json.load(open(f"{S}/tierD_picker_targets.json"))
json.dump({"manifest_sha256": t["manifest_sha256"], "per_clip": t["per_clip"],
           "per_listener": "kept outside the repository (per-listener values); SHA-256 of the full file in MANIFEST.sha256"},
          open(f"{D}/tierD_picker_targets_per_clip.json", "w"), indent=1)
PY
```
