"""R6 (docs/rules/r6_alice_short_scan_rule.md; exploratory, a replication attempt of R5): R5's minute ranking and test on the Alice
Datasets, chapter 1. The Alice held-out rule's data, coverage, kept LANG parcels, onsets and cleaning (scripts/alice_heldout.py and
stillhere/heldout.py, imported); R5's PULA, picks, gauge statistic and test (scripts/r5_short_scan.py, imported). Only the windows are
restated here, because R5's are cut at R2's TR of 1 s: 30-volume windows (60 s at TR 2.0 s) of the Alice full window, volumes 10 to j1.
One scoring run, after the rule and this file are on origin/main unchanged; it refuses a second run. Writes OUT/r6_alice_short_scan.json
(per listener; OUT outside the repo) and results/r6_alice_short_scan.json (group numbers and the rule's line).
usage: python scripts/r6_alice_short_scan.py score --root ROOT --out OUT   (ROOT as for scripts/alice_heldout.py)"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import alice_heldout as AH  # noqa: E402
import r5_short_scan as R5  # noqa: E402
from stillhere import heldout as H  # noqa: E402

RULE = "docs/rules/r6_alice_short_scan_rule.md"
LABEL = "Exploratory; the Alice scans were read on Oct 30 for the held-out test"   # the cloud's ruling of Nov 3, 19:05 UTC, item 3
RESULTS = os.path.join(REPO, "results/r6_alice_short_scan.json")
WIN = int(round(60 / H.TR))                                           # 30 volumes = 60 s at TR 2.0 s
HP = 0.03
FORECAST_PINS = {"full": "0a47d825a521f643fc5b298942fde7049679b070381bc97330ffd11a25d30ae3",   # the Alice rule, section 3
                 "audio": "d747d74f16231b249e1288da14f48ed0b57e434bd356edfb2836cdc7f166859b"}
LINES = {   # rule section 5, word for word
    "pass": "Exploratory; the Alice scans were read on Oct 30 for the held-out test. A replication attempt of R5: on a second public "
            "story outside TRIBE v2's training and evaluation sets (the "
            "Alice Datasets, chapter 1; {n} listeners), listeners' language areas followed TRIBE's forecast more closely in the {M} minutes "
            "it ranked highest than in the {M} minutes it ranked lowest ({k} of {n} listeners, one-sided p = {p}). Ranking the same minutes "
            "by word rate alone: {kwr} of {n}, p = {pwr}.",
    "fail": "Exploratory; the Alice scans were read on Oct 30 for the held-out test. A replication attempt of R5: on a second public "
            "story outside TRIBE v2's training and evaluation sets (the "
            "Alice Datasets, chapter 1; {n} listeners), the minutes TRIBE's forecast ranked highest did not reliably give a closer "
            "language-area match than the minutes it ranked lowest ({k} of {n} listeners, one-sided p = {p}). R5's Paranoia result is not "
            "replicated here. We show it because we said we would.",
    "not_run": "R6: rule committed, not run ({reason}).",
}


def windows(j1: int) -> list:
    """Non-overlapping WIN-volume windows of volumes J0..j1, as index ranges into the cleaned window; a short trailing window is not used."""
    n = j1 - H.J0 + 1
    return [(s, s + WIN) for s in range(0, n - WIN + 1, WIN)]


def seconds(a: int, b: int) -> list:
    """The stimulus seconds a window's volumes span (volume j: seconds 2j and 2j + 1), as indices into a per-second series."""
    return list(range(int(H.TR * (H.J0 + a)), int(H.TR * (H.J0 + b))))


def evaluate(B: dict, X: dict, C: dict, Pf0, Pa0, wr0, wps, env, j1: int, listeners: list) -> dict:
    """R6 on prepared inputs, all cleaned on J0..j1: B {listener: LANG BOLD rows}, X {listener: full forecast rows at its onset},
    C {listener: [LANG-mean audio-only forecast, word rate, its derivative] at its onset}; Pf0, Pa0, wr0 the full and audio-only rows and
    the word-rate row at the stimulus's own time (the one ranking); wps and env per stimulus second."""
    ws = windows(j1); win = {"alice": ws}
    secs = [seconds(a, b) for a, b in ws]
    rank = {"tribe": [R5.pula(Pf0, Pa0, wr0, a, b) for a, b in ws],
            "word_rate": [float(np.mean(np.asarray(wps)[s])) for s in secs],
            "envelope": [float(np.std(np.asarray(env)[s])) for s in secs]}
    picks = {name: R5.pick(sc) for name, sc in rank.items()}
    idx = {name: {"top": R5.volumes(win, {"alice": tb[0]}), "bottom": R5.volumes(win, {"alice": tb[1]})} for name, tb in picks.items()}
    m = {s: {name: {side: R5.m_on(B[s], X[s], C[s], v) for side, v in sides.items()} for name, sides in idx.items()} for s in listeners}
    d = {name: [m[s][name]["top"] - m[s][name]["bottom"] for s in listeners] for name in rank}
    res = {name: R5.test(d[name]) for name in rank}
    res["tribe_top_vs_word_rate_top"] = R5.test([m[s]["tribe"]["top"] - m[s]["word_rate"]["top"] for s in listeners])
    res["minutes_each_side"] = len(picks["tribe"][0])
    res["windows"] = {"n": len(ws), "kept_each_end": R5.k_of(len(ws)), "volumes": [[H.J0 + a, H.J0 + b - 1] for a, b in ws],
                      "stimulus_seconds": [[s[0], s[-1] + 1] for s in secs], "tribe_top": picks["tribe"][0],
                      "tribe_bottom": picks["tribe"][1], "pula": rank["tribe"]}
    return {"group": res, "per_listener": {s: {"m": m[s], "d_tribe": m[s]["tribe"]["top"] - m[s]["tribe"]["bottom"]} for s in listeners}}


def line(g: dict) -> str:
    t, wr = g["tribe"], g["word_rate"]
    f = {"n": t["n"], "M": g["minutes_each_side"], "k": t["k_above_0"], "p": f"{t['wilcoxon_p']:.2g}"}
    if t["pass"]:
        return LINES["pass"].format(kwr=wr["k_above_0"], pwr=f"{wr['wilcoxon_p']:.2g}", **f)
    return LINES["fail"].format(**f)


# ---- the scoring run ----------------------------------------------------------------------------------------------------------------
def on_main() -> bool:
    """The rule and this scorer are on origin/main and the working copies equal it (rule section 6)."""
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)
    me = os.path.relpath(os.path.abspath(__file__), REPO)
    there = all(git("log", "-n1", "--format=%H", "origin/main", "--", p).stdout.strip() for p in (RULE, me))
    return there and git("diff", "--quiet", "origin/main", "--", RULE, me).returncode == 0


def guard(out_dir):
    if os.path.realpath(out_dir).startswith(os.path.realpath(REPO) + os.sep):
        raise SystemExit("refused: OUT must be outside the repository (per-listener values)")
    if not on_main():
        raise SystemExit("refused: the rule and this scorer must be on origin/main, unchanged, before scoring (rule section 6)")
    if os.path.exists(RESULTS):
        raise SystemExit("refused: results/r6_alice_short_scan.json exists; R6 is scored once")


def check_inputs(root) -> dict:
    """The Alice rule's pins for the stimulus, annotations and the two stock forecasts; each scored listener's derivative and events
    file against the SHA-256 the Alice download recorded (ROOT/alice/bold/SHA256.json)."""
    A = f"{root}/alice"
    need = {"stimulus": (f"{A}/stimulus/{AH.STIM}", AH.STIM_SHA), "annotations": (f"{A}/annotations/annotations.tsv", AH.ANN_SHA),
            **{f"pred_alice_{k}": (f"{A}/forecast/out/pred_alice_{k}.npz", h) for k, h in FORECAST_PINS.items()}}
    missing = sorted(k for k, (p, _) in need.items() if not os.path.isfile(p))
    if missing or not os.path.isfile(f"{A}/bold/SHA256.json"):
        raise SystemExit(f"refused: missing inputs {missing or ['alice/bold/SHA256.json']}")
    bad = sorted(k for k, (p, h) in need.items() if AH.sha(p) != h)
    rec = json.load(open(f"{A}/bold/SHA256.json"))
    files = [f"sub-{s}_task-alice_bold_preprocessed.nii.gz" for s in AH.SCORED] + [f"sub-{s}_task-alice_events.tsv" for s in AH.SCORED]
    bad += sorted(f for f in files if f not in rec or AH.sha(f"{A}/bold/{f}") != rec[f])
    if bad:
        raise SystemExit(f"refused: inputs differ from the Alice rule's pins or the recorded download: {bad}")
    return {k: h for k, (_, h) in need.items()} | {"bold_and_events": "equal to alice/bold/SHA256.json"}


def score(root, out_dir):
    guard(out_dir)
    pins = check_inputs(root)
    A = f"{root}/alice"; os.makedirs(out_dir, exist_ok=True)
    sets = H.parcel_sets(*[f"{root}/atlas/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")])
    lang = {"LANG": sets["LANG"]}
    o = {s: AH.onset_of(f"{A}/bold/sub-{s}_task-alice_events.tsv")[0] for s in AH.SCORED}
    vert, covs = {}, {}
    for s in AH.SCORED:
        vert[s] = AH.project(f"{A}/bold/sub-{s}_task-alice_bold_preprocessed.nii.gz"); covs[s] = AH.covered(vert[s])
    elig = sorted(s for s in AH.SCORED if AH.area_shares(covs[s], sets)["LANG"] >= AH.COVER)
    res = {"label": LABEL, "rule": RULE, "dataset": "OpenNeuro ds002322 v1.0.4 (CC0), chapter 1",
           "inputs_sha256": pins, "listeners_language": len(elig)}
    if len(elig) < AH.MIN_N:
        res["line"] = LINES["not_run"].format(reason="coverage")
        return _write(res, res, out_dir)
    keep = AH.kept_parcels(covs, lang, {"LANG": elig}); ksets = AH.subset(lang, keep)
    n = AH.SKIP + min(v.shape[1] for v in vert.values())
    P = {k: np.load(f"{A}/forecast/out/pred_alice_{k}.npz") for k in FORECAST_PINS}
    last = min(int((float(P["full"]["starts"][-1]) + o[s] + 5) // H.TR) for s in AH.SCORED)
    j1 = min(last, n - 1)
    on = np.array([t for _, t, _ in AH.timed_words(f"{A}/annotations/annotations.tsv")])
    wps = np.bincount(on.astype(int), minlength=int(max(on.max(), n * H.TR)) + 40).astype(float)
    env = H.envelope_per_second(f"{A}/stimulus/{AH.STIM}")
    cl = lambda M: H.clean(M, HP, H.J0, j1)
    tj = H.TR * np.arange(n)

    def regs(L):   # heldout.stage_scores' word rate and its derivative at lag L
        return (cl(H.hrf_regressor(wps, tj - L)[None]),
                cl((H.hrf_regressor(wps, tj - L + 0.5) - H.hrf_regressor(wps, tj - L - 0.5))[None]))
    pp0 = {k: H.pred_parcels(d["P"], d["starts"], n, ksets)["LANG"] for k, d in P.items()}
    B, X, C = {}, {}, {}
    for s in elig:
        pps = {k: H.pred_parcels(d["P"], d["starts"] + o[s], n, ksets)["LANG"] for k, d in P.items()}
        B[s] = cl(AH.to_raw(AH.parcel_series(vert[s], covs[s], lang, keep))["LANG"][:, :n])
        X[s] = cl(pps["full"]); C[s] = [cl(pps["audio"]).mean(0, keepdims=True), *regs(AH.DEV_LAG + o[s])]
    del vert
    ev = evaluate(B, X, C, cl(pp0["full"]), cl(pp0["audio"]), regs(AH.DEV_LAG)[0], wps, env, j1, elig)
    pub = {**res, "lang_parcels_kept": len(keep["LANG"]), "window_volumes": [H.J0, j1], **ev["group"]}
    pub["line"] = line(ev["group"])
    return _write(pub, {**pub, "onsets_s": o, "per_listener": ev["per_listener"]}, out_dir)


def _write(pub, full, out_dir):
    json.dump(full, open(f"{out_dir}/r6_alice_short_scan.json", "w"), indent=1, default=float)
    json.dump(pub, open(RESULTS, "w"), indent=1, default=float)
    print(pub["line"], flush=True)
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--root", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    score(a.root, a.out)
