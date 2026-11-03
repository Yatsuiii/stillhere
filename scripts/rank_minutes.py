"""Which minutes of a story to play: R5's ranking (docs/rules/r5_short_scan_rule.md) for a lab's own story, from stock TRIBE v2's
forecasts alone (no brain data). Forecast only, exploratory: R5 checked this ranking once, on Paranoia (17 of 22 listeners followed the
top-ranked minutes more closely than the bottom-ranked; word rate alone: 8 of 22); R6's replication on the Alice Datasets failed (12 of
22, p = 0.34). Both ranked minutes of a whole-story scan; a scan that plays only the chosen minutes was not tested.

The method is R5's own code, imported, not restated: a virtual scan of each story part at TR 1 s with the story starting 2 s in and the
first 20 volumes dropped (R2's section 5); stock TRIBE v2's full and audio-only forecasts read as R2 reads them (volume j at story time
j - 7 s) on the language parcels, each cleaned over the part's window; word rate as R2's regressor at the development lag; 60-volume
windows (R5.windows), each scored by the picker's PULA (R5.pula: the spread of the full forecast left after the audio-only forecast and
word rate), and the top and bottom k = floor((n + 1) / 3) windows per part (R5.pick). The same windows ranked by word rate are shown
beside it, as R5 reported them.

Language parcels: with no brain data, every LANG parcel of the Schaefer-400 atlas (stillhere/heldout.py parcel_sets). R5 used the LANG
parcels that every Paranoia listener's scan covered (19); `check-r5` below reproduces R5's own window scores with those.

Inputs, as scripts/rank_minutes.sh writes them (or as the Paranoia release ships them): DIR/pred_stock_<part>_{full,audio}.npz (P, starts),
DIR/words_<part>.tsv (TRIBE v2's word table), and the part's audio.
usage: python scripts/rank_minutes.py rank DIR --atlas ATLAS_DIR --wav PART=WAV [PART=WAV ...] [--out FILE]
       python scripts/rank_minutes.py check-r5 PARANOIA_DATA   (scripts/reproduce_paranoia.sh's DATA_DIR, with ours_release/)"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import alice_heldout as AH  # noqa: E402
import r2_paranoia as R2  # noqa: E402
import r5_short_scan as R5  # noqa: E402
from stillhere import heldout as H  # noqa: E402

CAVEAT = ("Forecast only, exploratory. On Paranoia (R5) 17 of 22 listeners followed the top-ranked minutes more closely than the "
          "bottom-ranked (word rate alone: 8 of 22); on a second story, the Alice Datasets (R6), it did not replicate (12 of 22, p = 0.34). "
          "Both ranked minutes of a whole-story scan; a scan that plays only these minutes was not tested. Healthy adults only. Not a "
          "diagnosis or medical advice.")


MINUTES = ("Each minute is 60 volumes of a whole-story scan at TR 1 s with the story starting 2 s in and the first 20 volumes dropped "
           "(R2's design), named by the story seconds that play during it (R5's mapping); TRIBE v2 reads the response about 5 s behind "
           "the sound, so each minute's score reflects the story from about 5 s earlier.")


def mss(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def rank(parts: dict, lang: list, n: dict = None) -> dict:
    """parts: {name: {"full": (P, starts), "audio": (P, starts), "onsets": word onsets (s), "env": per-second envelope}}, in story order.
    lang: the language parcels (vertex index arrays). n: volumes per part (default: as many as the forecasts cover).
    Returns R5's windows, scores and picks per part, each window with the story seconds it hears."""
    sets = {"LANG": list(lang)}
    pp, stories, W = {}, {}, {}
    for p, d in parts.items():
        nn = (n or {}).get(p) or int(np.asarray(d["full"][1]).max() + R2.ONSET + 5) + 2
        pp[p] = {k: R2.pred_parcels(d[k][0], d[k][1], nn, sets, R2.fshift(0.0)) for k in ("full", "audio")}
        W[p] = min(nn - 1, R2.last_volume(pp[p], nn))
        on = np.asarray(d["onsets"], float)
        n_sec = int(max(on.max(), nn * R2.TR)) + 40
        stories[p] = {"n": nn, "onsets": on, "wps": np.bincount(on.astype(int), minlength=n_sec).astype(float), "env": d["env"], "base": {}}
    short = [p for p in W if W[p] - R2.J0 + 1 < R5.WIN]
    if short:
        raise SystemExit(f"refused: {short} shorter than one {R5.WIN}-s window after the first {R2.J0} s")
    F0 = R2.joined_forecasts(pp, W)
    reg = R2.regressors(stories, W, R2.DEV_LAG)
    Pf, Pa = F0[("full", "LANG")], F0[("audio", "LANG")]
    win = R5.windows(W)
    offs = dict(zip(W, np.cumsum([0] + [j1 - R2.J0 + 1 for j1 in W.values()])[:-1]))
    out = {"parts": {}, "minutes_each_side": 0, "lang_parcels": len(sets["LANG"]), "minutes": MINUTES, "caveat": CAVEAT}
    for p, ws in win.items():
        pula = [R5.pula(Pf, Pa, reg["wr"], a, b) for a, b in ws]
        secs = [R5.per_second(w, offs[p]) for w in ws]
        wr = [float(np.mean(stories[p]["wps"][s])) for s in secs]
        top, bottom = R5.pick(pula)
        wtop, wbottom = R5.pick(wr)
        order = sorted(range(len(ws)), key=lambda i: (-round(pula[i], 12), i))
        out["parts"][p] = {"window_volumes": [R2.J0, W[p]], "kept_each_end": R5.k_of(len(ws)), "tribe_top": top, "tribe_bottom": bottom,
                           "word_rate_top": wtop, "word_rate_bottom": wbottom,
                           "windows": [{"i": i, "story_s": [s[0], s[-1] + 1], "story": f"{mss(s[0])}-{mss(s[-1] + 1)}",
                                        "pula": pula[i], "rank": order.index(i) + 1, "words_per_s": wr[i]} for i, s in enumerate(secs)]}
        out["minutes_each_side"] += len(top)
    return out


def load_parts(d: str, wavs: dict) -> dict:
    """The rank inputs from DIR's files (the Paranoia release's names), one part per --wav, in the order given."""
    parts = {}
    for p, wav in wavs.items():
        rd = lambda k: np.load(os.path.join(d, f"pred_stock_{p}_{k}.npz"))
        f, a = rd("full"), rd("audio")
        parts[p] = {"full": (f["P"], f["starts"]), "audio": (a["P"], a["starts"]),
                    "onsets": [t for _, t in R2.word_table(os.path.join(d, f"words_{p}.tsv"))], "env": H.envelope_per_second(wav)}
    return parts


def atlas_lang(atlas: str) -> list:
    return H.parcel_sets(*[os.path.join(atlas, f"{h}.Schaefer2018_400Parcels_17Networks_order.annot") for h in ("lh", "rh")])["LANG"]


def report(res: dict) -> str:
    rows = []
    for p, r in res["parts"].items():
        rows.append(f"{p}: {len(r['windows'])} minutes scored, top and bottom {r['kept_each_end']} kept")
        for w in sorted(r["windows"], key=lambda w: w["rank"]):
            tag = "TOP" if w["i"] in r["tribe_top"] else "bottom" if w["i"] in r["tribe_bottom"] else ""
            rows.append(f"  #{w['rank']:<2} story {w['story']:>11}  PULA {w['pula']:.3f}  {w['words_per_s']:.2f} words/s  {tag}")
    return "\n".join(rows + ["", res["minutes"], "", res["caveat"]])


def check_r5(data: str) -> dict:
    """R5's own window scores, recomputed through rank() from the release's forecasts and word tables with R5's 19 kept parcels and run
    lengths (both from the public scans, as R5's scorer reads them): the same picks as results/r5_short_scan.json and every score within
    1e-6 of it. Machines differ in the last digits: on our laptop R5's own scorer and this function agree exactly, and both sit up to
    4.1e-9 from the scores R5's Kaggle run wrote (Nov 3)."""
    annot = [f"{data}/atlas/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")]
    sets = H.parcel_sets(*annot); medial = R2.medial_wall(*annot)
    covs, n = {}, {}
    for s in R2.SUBS:
        runs = {st: R2.load_run(f"{data}/r2/bold", s, st) for st in R2.STORIES}
        covs[s] = R2.coverage(runs, medial)
        n = {st: min(n.get(st, 10 ** 9), runs[st].shape[1]) for st in R2.STORIES}
        if not AH.area_shares(covs[s], sets)["LANG"] >= R2.COVER:
            covs.pop(s)
    keep = AH.kept_parcels(covs, {"LANG": sets["LANG"]}, {"LANG": sorted(covs)})
    lang = AH.subset({"LANG": sets["LANG"]}, keep)["LANG"]
    res = rank(load_parts(f"{data}/ours_release", {st: f"{data}/r2/stimuli/{st}_audio.wav" for st in R2.STORIES}), lang, n)
    ours = json.load(open(os.path.join(REPO, "results/r5_short_scan.json")))["windows"]
    diff = {p: float(np.max(np.abs(np.array([w["pula"] for w in r["windows"]]) - ours[p]["pula"]))) for p, r in res["parts"].items()}
    same = {p: r["tribe_top"] == ours[p]["tribe_top"] and r["tribe_bottom"] == ours[p]["tribe_bottom"] for p, r in res["parts"].items()}
    return {"lang_parcels": len(lang), "max_abs_pula_diff": diff, "same_picks": same,
            "pass": all(same.values()) and max(diff.values()) < 1e-6}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("dir"); r.add_argument("--atlas", required=True)
    r.add_argument("--wav", nargs="+", required=True, metavar="PART=WAV"); r.add_argument("--out")
    c = sub.add_parser("check-r5"); c.add_argument("data")
    a = ap.parse_args(argv)
    if a.cmd == "check-r5":
        res = check_r5(a.data)
        print(json.dumps(res, indent=1))
        return 0 if res["pass"] else 1
    wavs = dict(w.split("=", 1) for w in a.wav)
    res = rank(load_parts(a.dir, wavs), atlas_lang(a.atlas))
    json.dump(res, open(a.out or os.path.join(a.dir, "rank_minutes.json"), "w"), indent=1)
    print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
