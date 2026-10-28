"""Run table for the Emo-FilM rule (docs/rules/emofilm_rule.md version 2; exploratory): for each of the 30 Tears of Steel runs in
ds004892, the subject, session, logged film onset (film time 0), film duration, the end of the last logged washout, the volume count,
the volume-count check against the designed scan length (stillhere.emofilm_score.volume_check) and the number of scored volumes.
Reads only public metadata saved from OpenNeuro: each run's events file (func/*_task-scan_acq-TearsOfSteel_events.tsv), its raw BOLD
JSON sidecar (RepetitionTime, SliceTiming) and the volume counts (size_t) of the MRIQC group table. No BOLD image, preprocessed series,
motion or confound file. A viewer's run is the session holding its Tears of Steel BOLD sidecar (sub-S06: ses-4; its extra ses-3 events
file has no run). Prints the table and the kept and dropped counts; with --out also writes the rows as JSON (outside this repo).
usage: python scripts/emofilm_run_table.py META_ROOT [--out FILE]   (META_ROOT: the saved ds004892 metadata tree)"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import emofilm_score as E  # noqa: E402

PRED_STARTS = np.arange(190.0)          # the forecast kernel's 1 Hz samples, release 0 to 190 s


def tsv(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_table(root: str) -> list:
    counts = {r["bids_name"]: int(float(r["size_t"])) for r in tsv(f"{root}/mriqc_group_bold.tsv") if "TearsOfSteel" in r["bids_name"]}
    rows = []
    for side in sorted(glob.glob(f"{root}/sub-*/ses-*/func/sub-*_ses-*_task-TearsOfSteel_bold.json")):
        sub, ses = re.search(r"(sub-S\d+)_(ses-\d+)_", os.path.basename(side)).groups()
        ev = [dict(r, onset=float(r["onset"]), duration=float(r["duration"]))
              for r in tsv(f"{os.path.dirname(side)}/{sub}_{ses}_task-scan_acq-TearsOfSteel_events.tsv")]
        meta = json.load(open(side)); tr = float(meta["RepetitionTime"]); so = float(np.mean(meta["SliceTiming"]))
        onset = E.film_onset(ev); end = max(r["onset"] + r["duration"] for r in ev)
        chk = E.volume_check(counts[f"{sub}_{ses}_task-TearsOfSteel_bold"])
        rows.append(dict(subject=sub, session=ses, film_onset_s=round(onset, 3),
                         film_duration_s=round(next(r["duration"] for r in ev if r["trial_type"] == "film"), 3),
                         events_end_s=round(end, 2), events_end_trs=int(np.ceil(end / tr)), n_vols=chk["n_vols"], off_by=chk["off_by"],
                         keep=chk["keep"], tr=tr, slice_offset_s=round(so, 5),
                         scored_volumes=int(len(E.film_volumes(chk["n_vols"], tr, onset, so, PRED_STARTS)[0]))))
    if len({r["subject"] for r in rows}) != len(rows):
        raise SystemExit("refused: a viewer has more than one Tears of Steel run")
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out")
    a = ap.parse_args(); rows = run_table(a.root)
    print(f"{'subject':8s} {'session':7s} {'onset_s':>8s} {'film_s':>8s} {'events_end_s':>12s} {'n_vols':>6s} {'off_by':>6s} "
          f"{'scored':>6s}  check (expected {E.EXPECTED_VOLS} +- {E.VOL_TOL})")
    for r in rows:
        print(f"{r['subject']:8s} {r['session']:7s} {r['film_onset_s']:8.3f} {r['film_duration_s']:8.3f} {r['events_end_s']:12.2f} "
              f"{r['n_vols']:6d} {r['off_by']:+6d} {r['scored_volumes']:6d}  {'keep' if r['keep'] else 'DROP'}")
    kept = [r for r in rows if r["keep"]]
    print(f"runs {len(rows)}; kept {len(kept)}; dropped {len(rows) - len(kept)}: {[r['subject'] for r in rows if not r['keep']]}")
    if a.out:
        if os.path.abspath(a.out).startswith(REPO + os.sep):
            raise SystemExit("refused: --out must be outside the repo")
        json.dump({"rule": "docs/rules/emofilm_rule.md (version 2)", "expected_vols": E.EXPECTED_VOLS, "tolerance_trs": E.VOL_TOL,
                   "runs": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
