"""PREREG §8 held-out section 8 (LPP English): recompute the stage numbers with stillhere/heldout.py from the cached pre-event inputs.

Reads, under ROOT (the analysis folder): p1/cache_en8 and p1/cache (BOLD parcel series of the section-8 English and section-1 French
listeners), k2/out_en8/k2_out and k2/out/k2_out (stock TRIBE predictions), p1/data (TextGrid, audio, parse trees), p1/surprisal8.json
and atlas/ (Schaefer-400 annotations). Writes OUT/heldout_section8.json (per-listener z included, so OUT must be outside this repo)
and prints the summary lines. Settings are the ones HELDOUT.md fixed on Oct 23: 0.03 Hz, volumes 10-154 and 10 to the last predicted
volume, regressors at the dev envelope lag -1.25 s plus the section's own-best-lag sensitivity, circular nulls >= 15 TRs, M1 seed 11
x 20 sets, M2 seed 13 x 20 sets with 2,000 bootstrap resamples, volunteer equivalents with the D9 bootstrap.
usage: python scripts/heldout_section8.py --root ../stillhere --out ../stillhere/d22/sec8 [--parts stage,m1,m2,m3]"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillhere import heldout as H  # noqa: E402

DEV_LAG = -1.25
M1_KS, M2_KS = (1, 2, 3, 4, 6, 8, 12, 15), (1, 2, 3, 5)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cache(d):
    return {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(f"{d}/sub-*.npz"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="stage,m1,m2,m3")
    a = ap.parse_args()
    R, parts, t0 = a.root, a.parts.split(","), time.time()
    assert not os.path.abspath(a.out).startswith(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "OUT must be outside the repo"
    os.makedirs(a.out, exist_ok=True)
    f = {"tg": f"{R}/p1/data/lppEN_section8.TextGrid", "wav": f"{R}/p1/data/task-lppEN_section_8.wav", "tree": f"{R}/p1/data/lppEN_tree.csv",
         "surprisal": f"{R}/p1/surprisal8.json", "lh": f"{R}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot",
         "rh": f"{R}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot",
         **{f"en8_{k}": f"{R}/k2/out_en8/k2_out/pred_en8_{k}.npz" for k in ("full", "audio", "shuf")},
         **{f"fr_{k}": f"{R}/k2/out/k2_out/pred_fr_{k}.npz" for k in ("full", "audio")}}
    bold, frb = load_cache(f"{R}/p1/cache_en8"), load_cache(f"{R}/p1/cache")
    n, fn = bold[sorted(bold)[0]]["LANG"].shape[1], frb[sorted(frb)[0]]["LANG"].shape[1]
    sets = H.parcel_sets(f["lh"], f["rh"])
    assert [len(sets["AUD"]), len(sets["LANG"])] == [bold[sorted(bold)[0]]["AUD"].shape[0], bold[sorted(bold)[0]]["LANG"].shape[0]]
    P = {k: np.load(f[f"en8_{k}"]) for k in ("full", "audio", "shuf")}
    pp = {k: H.pred_parcels(d["P"], d["starts"], n, sets) for k, d in P.items()}
    last = int((float(P["full"]["starts"][-1]) + 5) // H.TR)
    iv = H.textgrid_intervals(f["tg"])
    on = H.word_onsets(iv)
    n_sec = int(max(on.max(), n * H.TR)) + 40
    wps = np.bincount(on.astype(int), minlength=n_sec).astype(float)
    env = H.envelope_per_second(f["wav"])
    S = json.load(open(f["surprisal"]))
    base, match = H.baseline_series(H.word_list(iv), (np.array(S["onsets"]), np.array(S["surprisal_bits"])),
                                    [l.strip() for l in open(f["tree"], encoding="utf-8") if l.strip()], n_sec)
    out = {"inputs_sha256": {k: sha(v) for k, v in f.items()}, "listeners": len(bold), "fr_listeners": len(frb), "n_volumes": n,
           "last_predicted_volume": last, "tree_match": match}
    print(f"{len(bold)} listeners, {n} volumes, predictions reach volume {last}; tree offset {match['tree_offset']} "
          f"(first-200 match {match['offset_match_first_200']:.2f}, word match {match['word_match']:.2f})", flush=True)
    if "stage" in parts:
        out["stage"] = {}
        for tag, j1 in (("145", 154), ("full", min(last, n - 1))):
            Lown, _ = H.own_best_lag([bold[s]["AUD"] for s in sorted(bold)[:16]], env, n, j1, 0.03)
            for lab, L in (("primary", DEV_LAG), ("own_lag", Lown)):
                r = H.stage_scores(bold, pp, wps, env, base, n, j1, L)
                out["stage"][f"{tag}_{lab}"] = r
                b = r["baselines"]
                print(f"[{tag} TRs, {lab}, L {L:+.2f} s] HEARING z {r['hearing']['z']:.2f} ({r['hearing']['detect_pct']:.0f}%) | envelope z "
                      f"{r['envelope']['z']:.2f} ({r['envelope']['detect_pct']:.0f}%), TRIBE > envelope p {r['envelope']['p_tribe_gt_envelope']:.2g} | "
                      f"LANGUAGE z {r['language']['z']:.2f} ({r['language']['detect_pct']:.0f}%) p {r['language']['p']:.2g} | shuffled z "
                      f"{r['shuffled']['z']:.2f}, real > shuffled p {r['shuffled']['p_real_gt_shuffled']:.2g} | baselines "
                      + ", ".join(f"{k} {v:.2f}" for k, v in b.items()) + f" | best {r['best_baseline']}, TRIBE > best p {r['p_tribe_gt_best']:.2g}"
                      f" | {time.time() - t0:.0f} s", flush=True)
    T145 = 154 - H.J0 + 1
    cl = lambda A: H.clean(A, 0.03, H.J0, 154)
    wr1 = cl(H.wordrate_1hz(on, n)[None])
    ctrl = {"AUD": H.zeros_control(T145), "LANG": [cl(pp["audio"]["LANG"]).mean(0, keepdims=True), wr1]}
    if "m1" in parts:
        ks = [k for k in M1_KS if k < len(bold) - 1] + [len(bold) - 1]
        m1 = H.isc_curve(bold, ctrl, ks, 20, 11)
        out["m1"] = {net: {str(k): v for k, v in res.items()} for net, res in m1.items()}
        for net in ("AUD", "LANG"):
            print(f"M1 {net} ISC(k) mean z / detect: " + ", ".join(f"{k}: {np.mean([z for z, _ in m1[net][k]]):.2f} / {100 * np.mean([d for _, d in m1[net][k]]):.0f}%" for k in ks)
                  + f" | {time.time() - t0:.0f} s", flush=True)
        if "stage" in parts:
            out["volunteer_equivalent"] = {}
            for net, g in (("LANG", "language"), ("AUD", "hearing")):
                curve = [float(np.mean([z for z, _ in m1[net][k]])) for k in ks]
                v = H.volunteer_equivalent(out["stage"]["145_primary"]["per_listener"][g], ks, curve)
                out["volunteer_equivalent"][g] = v
                print(f"volunteer equivalent {g}: z {v['z']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] -> point {v['point_k']:.1f}, lower {v['lower_k']:.1f} -> about {v['claim_k']}", flush=True)
    if "m2" in parts or "m3" in parts:
        FP = {k: np.load(f[f"fr_{k}"]) for k in ("full", "audio")}
    if "m2" in parts:
        fr_full = H.pred_parcels(FP["full"]["P"], FP["full"]["starts"], fn, sets)
        m2 = H.hybrid(bold, {"AUD": cl(pp["audio"]["AUD"]), "LANG": cl(pp["full"]["LANG"])}, ctrl, frb, cl(fr_full["LANG"]), M2_KS, 20, 13, 2000)
        out["m2"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in m2.items()}
        for net in ("AUD", "LANG"):
            for k in M2_KS:
                r = m2[net][k]
                print(f"M2 {net} k={k}: alone z {r['alone_z']:.2f} ({r['alone_detect_pct']:.0f}%) | + TRIBE z {r['hybrid_z']:.2f} ({r['hybrid_detect_pct']:.0f}%) | "
                      f"dz {r['dz']:+.2f} [{r['dz_ci95'][0]:+.2f}, {r['dz_ci95'][1]:+.2f}]", flush=True)
        for k in M2_KS:
            r = m2["wrong_story"][k]
            print(f"M2 wrong story k={k}: controls alone {r['alone_detect_pct']:.0f}% (z {r['alone_z']:.2f}) | + wrong TRIBE {r['hybrid_detect_pct']:.0f}% (z {r['hybrid_z']:.2f})"
                  f" | {time.time() - t0:.0f} s", flush=True)
    if "m3" in parts:
        fr_en = {k: H.pred_parcels(d["P"], d["starts"], n, sets) for k, d in FP.items()}
        own = {"AUD": cl(pp["audio"]["AUD"]), "LANG": cl(pp["full"]["LANG"])}
        wrong = {"AUD": cl(fr_en["audio"]["AUD"]), "LANG": cl(fr_en["full"]["LANG"])}
        grp = {net: cl(np.mean([frb[t][net] for t in sorted(frb)], 0)) for net in ("AUD", "LANG")}
        m3 = H.wrong_story_alone(bold, own, wrong, ctrl, grp)
        out["m3"] = m3
        for net in ("AUD", "LANG"):
            r = m3[net]
            print(f"M3 {net} (n {r['n']}): TRIBE own z {r['tribe_own_z']:.2f} ({r['tribe_own_detect_pct']:.0f}%), wrong story z {r['tribe_wrong_z']:.2f} "
                  f"({r['tribe_wrong_detect_pct']:.1f}%, {r['tribe_wrong_detected']} of {r['n']}) | ISC own z {r['isc_own_z']:.2f} ({r['isc_own_detect_pct']:.0f}%), "
                  f"wrong story z {r['isc_wrong_z']:.2f} ({r['isc_wrong_detect_pct']:.1f}%, {r['isc_wrong_detected']} of {r['n']})", flush=True)
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(f"{a.out}/heldout_section8.json", "w"), indent=1)
    print("wrote", f"{a.out}/heldout_section8.json", "|", out["seconds"], "s")


if __name__ == "__main__":
    main()
