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

## Dated note, Nov 1 2026 (cloud review of Nov 1, 06:05 UTC, item 16)

- `MANIFEST.sha256`'s header says "as the single scoring call wrote it". That does not hold for `tierD_wall_baselines.json`: the
  scoring call (`score_picker.py`) wrote the other files; `tierD_wall_baselines.json` was written by the hand run of
  `scripts/wall_baselines.py` that followed it. Its hash in the MANIFEST is that file's.
- The log behind "rc 0 at 13:11:22 UTC": `score_job_log_excerpt.txt` (lines 14-29 of `d22/picker/score_job.log`, unchanged except that the local work folder's path is shortened to `<work>/`). In
  `score_job.sh` the wall baselines step failed at once (rc 1, 12:54:11 UTC: `ModuleNotFoundError: numpy`), because the script changes
  directory into `stillhere-app` and then puts the analysis environment's site-packages on PYTHONPATH as a relative path that no longer
  resolves. Run by hand with the same arguments and that path made absolute, numpy imported and the script finished (rc 0, 13:11:22 UTC).
  Two lines were added by hand: the 12:54:54 UTC line that explains the failure, and the last line, the job's own 12:54:11 rc 1
  entry, which `score_job.sh`'s log() had written into `stillhere-app/score_job.log` after its change of directory and which was
  then appended to this log, after the hand run's lines.
