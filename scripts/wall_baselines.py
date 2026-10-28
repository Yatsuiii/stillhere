"""Story-wall cheap baselines (docs/rules/cheap_baselines_rule.md; exploratory), run AFTER the picker's single scoring call, on the same
manifest, collected parcel series and predictions, with the frozen picker code imported read-only (nothing in it is changed or re-run).
Refuses unless the frozen scoring call has already run on this manifest (its SCORED.lock and tierD_picker_wall.json exist in OUT) and
the posted hash equals the manifest's SHA-256. Writes OUT/tierD_wall_baselines.json beside the frozen tierD_picker_wall.json:
(a) per clip and pooled, TRIBE (unpartialled) vs HRF x word rate and HRF x envelope on LANG and AUD; (b) the D19 wall with each raw
regressor in place of the T1 gauge (own minus wrong, lit p < 0.01, family and N of M rules, fake-own check), via the frozen bars.wall.
usage: python scripts/wall_baselines.py PICKER_DIR MANIFEST_DIR DATA_ROOT PRED_ROOT OUT_DIR --posted-hash HEX [--workers N]"""
import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillhere import cheap_baselines as CB  # noqa: E402

_G = {}


def _per_run(task):
    q, i = task; g = _G; r, parts, tr = g["S"][q][i]; c = g["R"][q]["clip"]
    return q, g["L"][q][i], CB.run_gauges(g["Gm"], g["Cm"], r, parts[c], g["clips"][c], tr)


def _wall(task):
    """d per raw regressor for one listener of clip q (own minus mean wrong over the other eligible families), as D19's _wall."""
    q, i, fake = task; g = _G; r, parts, tr = g["S"][q][i]; R = g["R"]; own = fake or q; skip = {R[q]["family"], R[own]["family"]}
    diffs = {k: [] for k in CB.KINDS}
    for f, qs in sorted(g["pool"].get(R[q]["block"], {}).items()):
        if f in skip:
            continue
        o, w_ = {k: [] for k in CB.KINDS}, {k: [] for k in CB.KINDS}
        for x_ in qs:
            x = CB.pair_gauges(g["Gm"], g["Cm"], r, parts[R[q]["clip"]], g["clips"][R[own]["clip"]], g["cparts"][own],
                               g["clips"][R[x_]["clip"]], g["cparts"][x_], tr)
            if x is not None:
                for k in CB.KINDS:
                    o[k].append(x[0][k]); w_[k].append(x[1][k])
        for k in CB.KINDS:
            if o[k]:
                diffs[k].append(np.mean(o[k]) - np.mean(w_[k]))
    return q, fake, {k: float(np.mean(v)) if v else None for k, v in diffs.items()}


def pmap(fn, tasks, w):
    if w <= 1:
        return [fn(t) for t in tasks]
    with multiprocessing.get_context("fork").Pool(w) as pool:
        return pool.map(fn, tasks, chunksize=4)


def main(a):
    t0 = time.time(); sys.path.insert(0, a.picker_dir)
    import clips as C, gauges as G, bars as B, score_picker as SP     # the frozen picker code, read-only
    mpath = f"{a.manifest_dir}/picker_manifest.json"; h = hashlib.sha256(open(mpath, "rb").read()).hexdigest()
    if a.posted_hash != h:
        sys.exit("refused: the posted hash is not the manifest's SHA-256")
    if not (os.path.exists(f"{a.out}/SCORED.lock") and os.path.exists(f"{a.out}/tierD_picker_wall.json")):
        sys.exit("refused: run the frozen picker scoring call first (no SCORED.lock or tierD_picker_wall.json in OUT)")
    M = json.load(open(mpath))
    if M["status"] == "next step":
        sys.exit("picker is a next step: no wall, no baselines")
    z = np.load(f"{a.picker_dir}/sets.npz"); sets = {k: [z[f"{k}_{i}"] for i in range(int(z[f"{k}_n"]))] for k in ("AUD", "LANG")}
    rows = M["rows"]; rids = [r["rid"] for r in rows]; R = {r["rid"]: r for r in rows}
    clips = {c: SP.load_clip(C.stem_of(c), a.data, a.pred, sets) for c in sorted({r["clip"] for r in rows})}
    S = {q: [SP.series(a.data, x["file"]) for x in R[q]["listeners"]] for q in rids}; L = {q: [x["listener"] for x in R[q]["listeners"]] for q in rids}
    cparts = {q: S[q][0][1][R[q]["clip"]] for q in rids}
    el = [r for r in rows if r["wall_eligible"]]; pool = {}
    for r in el:
        pool.setdefault(r["block"], {}).setdefault(r["family"], []).append(r["rid"])
    _G.update(S=S, L=L, R=R, clips=clips, cparts=cparts, pool=pool, Gm=G, Cm=C)
    # (a) per listener and per clip
    per = {}
    for q, lst, o in pmap(_per_run, [(q, i) for q in rids for i in range(len(S[q]))], a.workers):
        per.setdefault(q, {})[lst] = o
    counted = [q for q in rids if R[q]["counted"] and R[q]["wall_eligible"]]
    pooled = {}
    for net in ("LANG", "AUD"):
        vals = {k: [per[q][l][net][k] for q in counted for l in per[q]] for k in CB.KINDS}
        pooled[net] = {"mean": {k: float(np.mean(v)) if v else None for k, v in vals.items()},
                       "tribe_vs": {k: CB.compare(vals["tribe"], vals[k]) for k in ("word_rate", "envelope")}}
    per_clip = {q: {net: {k: float(np.mean([per[q][l][net][k] for l in per[q]])) for k in CB.KINDS} for net in ("LANG", "AUD")} for q in rids}
    # (b) the baseline walls
    fakes = {r["rid"]: B.fake_own([x for x in rows if x["block"] == r["block"]], r["rid"]) for r in el}
    tasks = [(q, i, None) for q in rids for i in range(len(S[q]))] + [(q, i, f) for q, f in fakes.items() if f for i in range(len(S[q]))]
    dd = {}
    for q, f, d in pmap(_wall, tasks, a.workers):
        for k, v in d.items():
            if v is not None:
                dd.setdefault((k, q, f), []).append(v)
    walls = {}
    for k in CB.KINDS:
        d_real = {q: dd.get((k, q, None), []) for q in rids}; d_fake = {q: dd.get((k, q, f), []) for q, f in fakes.items() if f}
        walls[k] = B.wall(rows, d_real, d_fake)
    res = {"label": CB.LABEL, "rule": "docs/rules/cheap_baselines_rule.md", "manifest_sha256": h, "regressors": list(CB.KINDS),
           "pooled_counted_eligible": pooled, "per_clip": per_clip, "walls_raw": walls, "seconds": int(time.time() - t0)}
    json.dump(res, open(f"{a.out}/tierD_wall_baselines.json", "w"), indent=1, default=float)
    for k, w in walls.items():
        print(f"[{CB.LABEL}] wall with {k} (raw): N {w['N']} of M {w['M']}, fake-label F {w['F']} of {w['M']}", flush=True)
    for net in ("LANG", "AUD"):
        for k, c in pooled[net]["tribe_vs"].items():
            print(f"[{CB.LABEL}] {net} TRIBE vs {k}: n {c['n']}, HL {c.get('hl')}, CI {c.get('hl_ci95')}, p {c.get('p_two_sided')} -> {c['verdict']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("picker_dir", "manifest_dir", "data", "pred", "out"):
        ap.add_argument(k)
    ap.add_argument("--posted-hash", required=True); ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    main(ap.parse_args())
