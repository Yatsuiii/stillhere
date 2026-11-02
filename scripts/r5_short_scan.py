"""R5 (docs/rules/r5_short_scan_rule.md; exploratory): does stock TRIBE v2 pick the minutes of a Paranoia story where listeners' language
areas follow it most? Within each story part, non-overlapping 60-volume windows of R2's scored window are ranked by PULA, computed from
the forecasts alone; per listener, R2's LANGUAGE gauge statistic on the top third of windows minus on the bottom third (equal volumes);
PASS needs a one-sided exact Wilcoxon p < 0.05 and at least k*(n) listeners above 0. The same with windows ranked by word rate and by the
envelope's SD is reported beside it. Data, coverage, cleaning, forecasts and the gauge are R2's (scripts/r2_paranoia.py, imported); R5
reads only R2's pinned stock forecasts and word tables, so it does not depend on the fine-tuned line. Scored in R2's one scoring run.
Writes OUT/r5_short_scan.json (per listener; outside the repo) and results/r5_short_scan.json (group).
A judge's own run (--verify-own DIR; dated note of Nov 2 in the rule; scripts/reproduce_paranoia.sh): the stock forecasts and word tables
come from DIR and are hashed, not checked against R2's pins; the stimuli must still match R2's STIM_PINS and the brain data and atlas must
equal what our scoring run read (R2's check). Nothing is written into results/; the judge's numbers are printed beside ours.
usage: python scripts/r5_short_scan.py ROOT OUT_DIR   (ROOT as for r2_paranoia.py score)
       python scripts/r5_short_scan.py ROOT --verify-own DIR --out OUT_DIR"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import alice_heldout as AH  # noqa: E402
import r2_paranoia as R2  # noqa: E402
from stillhere import heldout as H  # noqa: E402

RULE = "docs/rules/r5_short_scan_rule.md"
WIN = 60                                                              # volumes per window (60 s at TR 1 s)
LINES = {   # rule section 5, word for word
    "pass": "Exploratory. On another public story (Paranoia, Finn et al. 2018; {n} listeners, outside TRIBE v2's training and evaluation "
            "sets), listeners' language areas followed TRIBE's forecast more closely in the {M} minutes it ranked highest than in the {M} "
            "minutes it ranked lowest ({k} of {n} listeners, one-sided p = {p}). Ranking the same minutes by word rate alone: {kwr} of "
            "{n}, p = {pwr}.",
    "fail": "Exploratory. On another public story (Paranoia, Finn et al. 2018; {n} listeners, outside TRIBE v2's training and evaluation "
            "sets), the minutes TRIBE's forecast ranked highest did not reliably give a closer language-area match than the minutes it "
            "ranked lowest ({k} of {n} listeners, one-sided p = {p}). We show it because we said we would.",
    "not_run": "R5: rule committed, not run ({reason}).",
}


# ---- windows and the forecast-only ranking ------------------------------------------------------------------------------------------
def windows(W: dict) -> dict:
    """{part: [(a, b), ...]}: non-overlapping WIN-volume windows of each part's scored volumes J0..W[part], as index ranges into the
    joined series (parts in W's order); a trailing window shorter than WIN is not used."""
    out, off = {}, 0
    for part, j1 in W.items():
        n = j1 - R2.J0 + 1
        out[part] = [(off + s, off + s + WIN) for s in range(0, n - WIN + 1, WIN)]
        off += n
    return out


def pula(Pf, Pa, wr, a: int, b: int) -> float:
    """The picker's PULA on one window: per parcel row, the SD of the full forecast's residual after least squares on [1, the same
    parcel's audio-only forecast, the word-rate regressor] over volumes a..b-1; the mean over parcels."""
    sds = []
    for i in range(Pf.shape[0]):
        X = np.column_stack([np.ones(b - a), Pa[i, a:b], np.asarray(wr)[0, a:b]])
        y = Pf[i, a:b]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sds.append(float(np.std(y - X @ beta)))
    return float(np.mean(sds))


def k_of(n: int) -> int:
    """Windows kept at each end of a part with n windows: floor((n + 1) / 3)."""
    return (n + 1) // 3


def pick(scores: list) -> tuple:
    """(top, bottom) window indices: the k highest and the k lowest scores, ties (to 12 digits) to the earlier window."""
    k = k_of(len(scores)); r = [round(float(s), 12) for s in scores]
    top = sorted(range(len(r)), key=lambda i: (-r[i], i))[:k]
    bottom = sorted(range(len(r)), key=lambda i: (r[i], i))[:k]
    return sorted(top), sorted(bottom)


def volumes(win: dict, chosen: dict) -> np.ndarray:
    """Joined-series volume indices of the chosen windows ({part: [window index]})."""
    return np.concatenate([np.arange(*win[p][i]) for p in win for i in chosen.get(p, [])]).astype(int)


def per_second(part_w: tuple, offset: int) -> list:
    """The story seconds a window covers (volume j at story time j - 2 s), as indices into a per-second series of that part."""
    a, b = part_w
    return [int(R2.J0 + (j - offset) - R2.ONSET) for j in range(a, b)]


# ---- the target ---------------------------------------------------------------------------------------------------------------------
def partial_fz(B, X, controls) -> float:
    """R2's LANGUAGE gauge statistic without its null: the Fisher-z mean over rows of the partial r of z-scored B rows with X rows given
    [controls, 1] (R2.gauge(...)[1])."""
    T = B.shape[1]
    C = np.vstack(list(controls) + [np.ones((1, T))]).T
    M = np.linalg.pinv(C.T @ C)
    proj = lambda A: ((A @ C) @ M) @ C.T
    rb = H.zrow(B); rb = rb - proj(rb)
    X = np.asarray(X, float)
    return float(H._fz_rows(rb, (X - proj(X))[None])[0])


def m_on(B, X, controls, idx) -> float:
    return partial_fz(B[:, idx], X[:, idx], [np.asarray(c)[:, idx] for c in controls])


def kstar(n: int) -> int:
    """The smallest k with P(X >= k) < 0.05 for X ~ Binomial(n, 0.5)."""
    return next(k for k in range(n + 1) if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n < 0.05)


def test(d) -> dict:
    d = np.asarray(d, float); n = len(d); k = int((d > 0).sum())
    p = float(stats.wilcoxon(d, alternative="greater", method="exact").pvalue)
    return {"n": n, "k_above_0": k, "kstar": kstar(n), "wilcoxon_p": p, "median": float(np.median(d)),
            "pass": bool(p < 0.05 and k >= kstar(n))}


def evaluate(Bj: dict, F0: dict, reg: dict, stories: dict, W: dict, listeners: list) -> dict:
    """R5 on R2's prepared inputs: Bj {listener: {"LANG": cleaned joined rows}}, F0 R2's joined stock forecasts at shift 0, reg R2's
    regressors at the development lag, stories {part: {"wps", "env"}} per second, W the scored window per part."""
    Pf, Pa = F0[("full", "LANG")], F0[("audio", "LANG")]
    tc = R2.lang_controls(F0, reg)
    win = windows(W); offs = dict(zip(W, np.cumsum([0] + [j1 - R2.J0 + 1 for j1 in W.values()])[:-1]))
    rank = {"tribe": {}, "word_rate": {}, "envelope": {}}
    for p, ws in win.items():
        rank["tribe"][p] = [pula(Pf, Pa, reg["wr"], a, b) for a, b in ws]
        secs = [per_second(w, offs[p]) for w in ws]
        rank["word_rate"][p] = [float(np.mean(np.asarray(stories[p]["wps"])[s])) for s in secs]
        rank["envelope"][p] = [float(np.std(np.asarray(stories[p]["env"])[s])) for s in secs]
    picks = {name: {p: pick(sc) for p, sc in r.items()} for name, r in rank.items()}
    idx = {name: {"top": volumes(win, {p: tb[0] for p, tb in pk.items()}), "bottom": volumes(win, {p: tb[1] for p, tb in pk.items()})}
           for name, pk in picks.items()}
    m = {s: {name: {side: m_on(Bj[s]["LANG"], Pf, tc, v) for side, v in sides.items()} for name, sides in idx.items()} for s in listeners}
    d = {name: [m[s][name]["top"] - m[s][name]["bottom"] for s in listeners] for name in rank}
    res = {"tribe": test(d["tribe"]), "word_rate": test(d["word_rate"]), "envelope": test(d["envelope"]),
           "tribe_top_vs_word_rate_top": test([m[s]["tribe"]["top"] - m[s]["word_rate"]["top"] for s in listeners])}
    res["minutes_each_side"] = int(len(idx["tribe"]["top"]) // WIN)
    res["windows"] = {p: {"n": len(ws), "kept_each_end": k_of(len(ws)), "tribe_top": picks["tribe"][p][0],
                          "tribe_bottom": picks["tribe"][p][1], "pula": rank["tribe"][p]} for p, ws in win.items()}
    return {"group": res, "per_listener": {s: {"m": m[s], "d_tribe": m[s]["tribe"]["top"] - m[s]["tribe"]["bottom"]} for s in listeners}}


def line(g: dict) -> str:
    t, wr = g["tribe"], g["word_rate"]
    f = {"n": t["n"], "M": g["minutes_each_side"], "k": t["k_above_0"], "p": f"{t['wilcoxon_p']:.2g}"}
    if t["pass"]:
        return LINES["pass"].format(kwr=wr["k_above_0"], pwr=f"{wr['wilcoxon_p']:.2g}", **f)
    return LINES["fail"].format(**f)


# ---- the scoring run ----------------------------------------------------------------------------------------------------------------
def check_inputs(root) -> dict:
    """R2's pins for what R5 reads: the stimuli, the word tables and the stock forecasts (full and audio for the test, shuf only for
    R2's scored window); the fine-tuned forecasts too, for that window, when R2 pinned them."""
    R = f"{root}/r2"
    need = {**{f"stimulus {s}": (f"{R}/stimuli/{s}_audio.wav", R2.STIM_PINS.get(s)) for s in R2.STORIES},
            **{f"words {s}": (f"{R}/forecast/out/words_{s}.tsv", R2.WORD_PINS.get(s)) for s in R2.STORIES},
            **{f"pred stock_{s}_{k}": (f"{R}/forecast/out/pred_stock_{s}_{k}.npz", R2.FORECAST_PINS.get(f"stock_{s}_{k}"))
               for s in R2.STORIES for k in R2.MODES},
            **{f"pred tuned_{s}_{k}": (f"{R}/forecast/out/pred_tuned_{s}_{k}.npz", R2.FORECAST_PINS[f"tuned_{s}_{k}"])
               for s in R2.STORIES for k in R2.MODES if f"tuned_{s}_{k}" in R2.FORECAST_PINS}}
    if not all(h for _, h in need.values()):
        raise SystemExit("refused: R2's pins for the stimuli, word tables or stock forecasts are not filled (R2's forecasts' commit first)")
    bad = sorted(k for k, (p, h) in need.items() if R2.sha(p) != h)
    if bad:
        raise SystemExit(f"refused: inputs differ from R2's pins: {bad}")
    return {k: h for k, (_, h) in need.items()}


def own_inputs(root, own) -> dict:
    """--verify-own: the judge's stock forecasts and word tables in OWN, hashed and recorded, never compared with R2's pins; the stimuli
    still against R2's STIM_PINS; arms R2's check that the brain data and atlas equal what our scoring run read."""
    R = f"{root}/r2"
    got = {f"stimulus {s}": R2.sha(f"{R}/stimuli/{s}_audio.wav") for s in R2.STORIES}
    bad = sorted(k for k in got if got[k] != R2.STIM_PINS.get(k.split()[1]))
    if bad:
        raise SystemExit(f"refused: the stimuli differ from ds001338's recordings (R2's STIM_PINS): {bad}")
    files = ({f"words {s}": f"{own}/words_{s}.tsv" for s in R2.STORIES}
             | {f"pred stock_{s}_{k}": f"{own}/pred_stock_{s}_{k}.npz" for s in R2.STORIES for k in R2.MODES})
    missing = sorted(k for k, p in files.items() if not os.path.isfile(p))
    if missing:
        raise SystemExit(f"refused: {own} lacks {missing}")
    got.update({k: R2.sha(p) for k, p in files.items()})
    R2.VERIFY.clear(); R2.VERIFY.update(json.load(open(R2.OURS["read"])))
    return got


def score(root, out_dir, own=None):
    """R2's loading (scripts/r2_paranoia.py score: coverage, kept parcels, the scored window, each run cleaned then joined), stock only,
    then evaluate()."""
    if own is not None:
        R2.refuse_results(out_dir)
    assert not os.path.abspath(out_dir).startswith(REPO), "OUT must be outside the repo (per-listener values)"
    R2.fshift(0.0)                                                    # refuses until R2's FORECAST_READ is set
    R2.VERIFY.clear()
    pins = check_inputs(root) if own is None else own_inputs(root, own)
    R = f"{root}/r2"; os.makedirs(out_dir, exist_ok=True)
    FO = f"{R}/forecast/out" if own is None else own                  # the forecasts and word tables
    write = _write if own is None else _write_own
    annot = [f"{root}/atlas/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")]
    if own is not None:
        for a in annot:
            R2.note(a)                                                # the judge's atlas must be ours
    sets = H.parcel_sets(*annot); medial = R2.medial_wall(*annot)
    vert, covs, table = {}, {}, {}
    for s in R2.SUBS:
        vert[s] = {st: R2.load_run(f"{R}/bold", s, st) for st in R2.STORIES}
        covs[s] = R2.coverage(vert[s], medial)
        table[s] = AH.area_shares(covs[s], sets)["LANG"] >= R2.COVER
    elig = {"LANG": sorted(s for s in R2.SUBS if table[s])}
    res = {"label": "exploratory", "rule": RULE, "dataset": R2.DATASET, "inputs_sha256": pins, "listeners_language": len(elig["LANG"])}
    if len(elig["LANG"]) < R2.MIN_N:
        res["line"] = LINES["not_run"].format(reason="coverage")
        return write(res, res, out_dir)
    keep = AH.kept_parcels(covs, {"LANG": sets["LANG"]}, {"LANG": elig["LANG"]}); ksets = AH.subset({"LANG": sets["LANG"]}, keep)
    n = {st: min(vert[s][st].shape[1] for s in R2.SUBS) for st in R2.STORIES}
    stories = {}
    for st in R2.STORIES:
        words = R2.word_table(f"{FO}/words_{st}.tsv"); on = np.array([t for _, t in words])
        n_sec = int(max(on.max(), n[st] * R2.TR)) + 40
        stories[st] = {"n": n[st], "onsets": on, "wps": np.bincount(on.astype(int), minlength=n_sec).astype(float),
                       "env": H.envelope_per_second(f"{R}/stimuli/{st}_audio.wav"), "base": {}}
    models = ["stock"] + (["tuned"] if any(k.startswith("pred tuned_") for k in pins) else [])
    rd = lambda m, st, k: np.load(f"{FO}/pred_{m}_{st}_{k}.npz")
    pp = {m: {st: {k: R2.pred_parcels(rd(m, st, k)["P"], rd(m, st, k)["starts"], n[st], ksets, 0.0) for k in R2.MODES}
              for st in R2.STORIES} for m in models}
    W = {st: min(n[st] - 1, min(R2.last_volume(pp[m][st], n[st]) for m in models)) for st in R2.STORIES}   # R2's scored window
    bold = {s: {st: AH.parcel_series(vert[s][st], covs[s], {"LANG": sets["LANG"]}, keep)["LANG"][:, :n[st]] for st in R2.STORIES}
            for s in elig["LANG"]}
    del vert
    Bj = {s: {"LANG": R2.cj({st: bold[s][st] for st in W}, W)} for s in elig["LANG"]}
    F0 = R2.joined_forecasts({st: pp["stock"][st] for st in W}, W)
    reg = R2.regressors(stories, W, R2.DEV_LAG)
    ev = evaluate(Bj, F0, reg, stories, W, elig["LANG"])
    pub = {**res, "window_volumes": {st: [R2.J0, j1] for st, j1 in W.items()}, **ev["group"]}
    pub["line"] = line(ev["group"])
    return write(pub, {**pub, "per_listener": ev["per_listener"]}, out_dir)


def _write(pub, full, out_dir):
    json.dump(full, open(f"{out_dir}/r5_short_scan.json", "w"), indent=1, default=float)
    json.dump(pub, open(f"{REPO}/results/r5_short_scan.json", "w"), indent=1, default=float)
    print(pub["line"], flush=True)
    return pub


OURS = os.path.join(REPO, "results/r5_short_scan.json")


def side_by_side(theirs, ours) -> list:
    """The judge's R5 numbers beside ours: TRIBE's k of n and p, word rate's, the minutes each side, PASS or FAIL."""
    def row(d):
        if "tribe" not in d:
            return ["not scored"] * 5
        t, w = d["tribe"], d["word_rate"]
        return [f"{t['k_above_0']} of {t['n']}", f"{t['wilcoxon_p']:.2g}", f"{w['k_above_0']} of {w['n']}, p = {w['wilcoxon_p']:.2g}",
                str(d["minutes_each_side"]), "PASS" if t["pass"] else "FAIL"]
    names = ["TRIBE: listeners above 0", "TRIBE: one-sided p", "word rate", "minutes each side", "verdict"]
    a, b = row(theirs), row(ours)
    return [f"{'':26}{'yours':>18}{'ours':>18}"] + [f"{n:26}{x:>18}{y:>18}" for n, x, y in zip(names, a, b)]


def _write_own(pub, full, out_dir):
    """--verify-own: the judge's results to OUT only (never results/), then their numbers beside ours."""
    R2.refuse_results(out_dir)
    json.dump(full, open(f"{out_dir}/r5_short_scan.json", "w"), indent=1, default=float)
    json.dump(pub, open(f"{out_dir}/r5_short_scan_public.json", "w"), indent=1, default=float)
    print(pub["line"], flush=True)
    print("\n".join(["", "Your forecasts beside ours (results/r5_short_scan.json):"] + side_by_side(pub, json.load(open(OURS)))), flush=True)
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out_dir", nargs="?", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--verify-own", default=None, metavar="DIR", help="a judge's own stock forecasts and word tables (not R2's pins)")
    a = ap.parse_args()
    out = a.out or a.out_dir
    if out is None:
        ap.error("OUT_DIR (or --out) is required")
    score(a.root, out, a.verify_own)
