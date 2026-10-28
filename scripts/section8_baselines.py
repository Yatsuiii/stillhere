"""Section-8 cheap baselines (docs/rules/cheap_baselines_rule.md; exploratory): TRIBE (unpartialled) vs HRF x word rate and HRF x envelope
on LPP-EN held-out section 8, with the section-8 harness (stillhere/heldout.py) on the same cached inputs as scripts/heldout_section8.py:
49 listeners, 0.03 Hz, volumes 10..154 and 10..last predicted volume, regressors at the primary lag -1.25 s. The section-8 BOLD was
first read on Oct 22-23; this rule is committed after that read. Writes OUT/section8_baselines.json (per-listener z, so OUT is outside
this repo) and results/section8_baselines.json (group numbers only).
usage: python scripts/section8_baselines.py --root ../stillhere --out ../stillhere/d22/sec8"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import cheap_baselines as CB  # noqa: E402
from stillhere import heldout as H  # noqa: E402

DEV_LAG = -1.25
SEC8_LABEL = "recomputed with code built during the event; first measured Oct 22-23"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    assert not os.path.abspath(a.out).startswith(REPO), "OUT must be outside the repo"
    R = a.root; os.makedirs(a.out, exist_ok=True)
    bold = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(f"{R}/p1/cache_en8/sub-*.npz"))}
    n = bold[sorted(bold)[0]]["LANG"].shape[1]
    sets = H.parcel_sets(f"{R}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot", f"{R}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot")
    P = {k: np.load(f"{R}/k2/out_en8/k2_out/pred_en8_{k}.npz") for k in ("full", "audio")}
    pp = {k: H.pred_parcels(d["P"], d["starts"], n, sets) for k, d in P.items()}
    last = int((float(P["full"]["starts"][-1]) + 5) // H.TR)
    on = H.word_onsets(H.textgrid_intervals(f"{R}/p1/data/lppEN_section8.TextGrid"))
    wps = np.bincount(on.astype(int), minlength=int(max(on.max(), n * H.TR)) + 40).astype(float)
    env = H.envelope_per_second(f"{R}/p1/data/task-lppEN_section_8.wav")
    full, public = {"label": CB.LABEL, "section8_label": SEC8_LABEL, "rule": "docs/rules/cheap_baselines_rule.md", "listeners": len(bold)}, {}
    for tag, j1 in (("145", 154), ("full", min(last, n - 1))):
        r = CB.section8(H, bold, pp, wps, env, n, j1, DEV_LAG)
        full[tag] = r; public[tag] = {k: v for k, v in r.items() if k not in ("LANG", "AUD")}
        for net in ("LANG", "AUD"):
            public[tag][net] = {"mean_z": r[net]["mean_z"], "tribe_vs": r[net]["tribe_vs"]}
            for k, c in r[net]["tribe_vs"].items():
                print(f"[{CB.LABEL}; {SEC8_LABEL}] {tag} TRs {net}: TRIBE z {r[net]['mean_z']['tribe']:.2f} vs {k} z {r[net]['mean_z'][k]:.2f} | "
                      f"HL {c['hl']:+.2f} [{c['hl_ci95'][0]:+.2f}, {c['hl_ci95'][1]:+.2f}], k {c['k_tribe_higher']}/{c['n']}, p {c['p_two_sided']:.2g} "
                      f"-> {c['verdict']}", flush=True)
    json.dump(full, open(f"{a.out}/section8_baselines.json", "w"), indent=1)
    json.dump({"label": CB.LABEL, "section8_label": SEC8_LABEL, "rule": "docs/rules/cheap_baselines_rule.md", "listeners": len(bold), **public},
              open(f"{REPO}/results/section8_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
