"""Ghost-line data for Paranoia story 1, as docs/rules/display_rule_ghostline.md's Paranoia section fixes it (a display, no new
statistic): every step the LPP section's, applied to R2's committed reading (scripts/r2_paranoia.py's loading, coverage, kept LANG
parcels, cleaning and forecast read). Only the forecast and the 22-listener group line are written; no listener's own series.

Reads, under DATA (scripts/reproduce_paranoia.sh's DATA_DIR): r2/bold (story 1's GIFTIs), atlas/, and the stock full forecast of story 1
from DATA/ours_release or DATA/r2/forecast/out. Every input must equal the committed hashes (results/r2_bold_SHA256.json,
results/r2_paranoia_SHA256.json, R2's FORECAST_PINS). Refuses if the rule file has no commit or has uncommitted changes, or if the
rule's chip commit cannot be resolved. Writes results/ghostline_paranoia_s1.json; prints counts, shapes and timings only.
usage: python scripts/ghostline_paranoia.py DATA"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import alice_heldout as AH  # noqa: E402
import r2_paranoia as R2  # noqa: E402
from stillhere import heldout as H  # noqa: E402

RULE = "docs/rules/display_rule_ghostline.md"
R2_RULE = "docs/rules/r2_paranoia_rule.md"
OUT = os.path.join(REPO, "results/ghostline_paranoia_s1.json")
STORY, SHORT = "story1", 120                                          # the LPP section's short view: the first 120 s of the window


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def rule_commit() -> tuple:
    if git("status", "--porcelain", "--", RULE) or "## Paranoia ghost line" not in git("show", f"HEAD:{RULE}"):
        raise SystemExit("refused: the rule's Paranoia section has no commit or has uncommitted changes")
    h, when = git("log", "-1", "--format=%h %cI", "--", RULE).split()
    return h, when


def r2_commit() -> str:
    """The last commit touching R2's rule before the first Paranoia brain file was downloaded (the scoring kernel's start)."""
    start = json.load(open(os.path.join(REPO, "results/r2_run/r2_score_run.json")))["started_utc"]
    c = git("log", "-1", f"--before={start}", "--format=%h", "--", R2_RULE)
    if not c:
        raise SystemExit("refused: no commit of R2's rule before the scoring run's start")
    return c


def caption(r2: dict) -> str:
    """R2's committed numbers only (results/r2_paranoia.json, models.stock), rounded as R2's line rounds them; nothing recomputed."""
    ve, share = r2["models"]["stock"]["volunteer_equivalent_language"], r2["models"]["stock"]["reference"]["share"]
    return (f"~{round(ve['point_k'])} volunteers' worth (at least {ve['claim_k']}). Group level. Volunteer equivalent "
            f"{ve['point_k']:.1f}, lower bound {ve['lower_k']:.2f}; {round(100 * share)}% of what the other listeners' average reaches.")


def z(x) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def lines(listener_rows: list, forecast_rows, W: int) -> dict:
    """The LPP section's steps on R2's window: each source's mean over parcels of its cleaned rows, z-scored over the window; the group
    line the mean of the listeners' lines, z-scored again; the short view cropped afterwards, never re-scaled."""
    each = [z(R2.clean(r, R2.J0, W).mean(0)) for r in listener_rows]
    return {"forecast": z(R2.clean(forecast_rows, R2.J0, W).mean(0)), "group": z(np.mean(each, 0))}


def check_inputs(data: str, fc_path: str) -> dict:
    bold = json.load(open(os.path.join(REPO, "results/r2_bold_SHA256.json")))
    read = json.load(open(os.path.join(REPO, "results/r2_paranoia_SHA256.json")))
    got = {}
    for f, h in bold.items():
        if STORY in f and f.endswith(".gii"):
            got[f] = sha(os.path.join(data, "r2/bold", f))
            if got[f] != h:
                raise SystemExit(f"refused: {f} differs from the committed hash")
    for a in ("lh", "rh"):
        f = f"{a}.Schaefer2018_400Parcels_17Networks_order.annot"
        if sha(os.path.join(data, "atlas", f)) != read[f]:
            raise SystemExit(f"refused: {f} differs from the committed hash")
    if sha(fc_path) != R2.FORECAST_PINS[f"stock_{STORY}_full"]:
        raise SystemExit("refused: the stock forecast differs from R2's pin")
    return {"bold_files": len(got), "forecast_sha256": R2.FORECAST_PINS[f"stock_{STORY}_full"]}


def main(data: str):
    t0 = time.time()
    rc, rwhen = rule_commit(); chip_commit = r2_commit()
    fc_path = next(p for p in (f"{data}/ours_release/pred_stock_{STORY}_full.npz", f"{data}/r2/forecast/out/pred_stock_{STORY}_full.npz")
                   if os.path.exists(p))
    checked = check_inputs(data, fc_path)
    annot = [f"{data}/atlas/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")]
    sets = H.parcel_sets(*annot); medial = R2.medial_wall(*annot)
    runs = {s: R2.load_run(f"{data}/r2/bold", s, STORY) for s in R2.SUBS}
    n = min(v.shape[1] for v in runs.values())
    covs = {s: R2.coverage({STORY: runs[s]}, medial) for s in R2.SUBS}
    keep = AH.kept_parcels(covs, sets, {"LANG": sorted(R2.SUBS), "AUD": sorted(R2.SUBS)}); ksets = AH.subset(sets, keep)
    d = np.load(fc_path)
    fc = R2.pred_parcels(d["P"], d["starts"], n, ksets, R2.fshift(0.0))["LANG"]
    r2 = json.load(open(os.path.join(REPO, "results/r2_paranoia.json"))); W = r2["window"]["volumes"][STORY][1]
    L = lines([AH.parcel_series(runs[s], covs[s], sets, keep)["LANG"][:, :n] for s in R2.SUBS], fc, W)
    doc = {"format": "stillhere-ghostline/1", "label": "display only; no new statistic",
           "rule": {"path": RULE, "commit": rc, "committed": rwhen},
           "dataset": "Paranoia (Finn et al. 2018, ds001338) story 1, the 22 listeners R2 scored",
           "title": "Paranoia, story 1 (outside TRIBE v2's training and evaluation sets)", "region": "R2's language parcels (LANG)",
           "n_parcels": len(ksets["LANG"]), "volumes": [R2.J0, W], "tr_s": 1.0, "time_s": [float(j) for j in range(R2.J0, W + 1)],
           "short_view": {"volumes": [R2.J0, R2.J0 + SHORT - 1], "index": [0, SHORT],
                          "note": "the LPP section's window step: the first 120 s of the scored window, cropped after z-scoring"},
           "draw_order": ["forecast", "chip", "listeners"],
           "chip": {"short": "Checked: Paranoia (outside TRIBE's sets)",
                    "long": f"rule and forecasts committed before the brain data was downloaded (R2) · {chip_commit}"},
           "lines": {"forecast": {"label": "TRIBE v2 forecast", "style": "gray dashed", "values": [round(float(v), 4) for v in L["forecast"]]},
                     "group": {"label": "22 public listeners (average)", "style": "solid", "values": [round(float(v), 4) for v in L["group"]]}},
           "caption": {"text": caption(r2), "source": "results/r2_paranoia.json models.stock", "only_beside": "this line",
                       "label": "exploratory"},
           "provenance": {**checked, "forecast_read": "R2 pred_parcels at shift 0 (story time j - 2 - 5 s)",
                          "cleaning": "R2 clean on volumes 20..W per parcel row"},
           "extracted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"{len(R2.SUBS)} listeners, {doc['n_parcels']} LANG parcels, volumes {R2.J0}-{W}, chip {chip_commit}, rule {rc}, "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.split("usage: ")[1])
    main(sys.argv[1])
