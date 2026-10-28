"""notthefall (D20) cheap baselines (docs/rules/cheap_baselines_rule.md; exploratory), run as their own private Kaggle CPU kernel
(stillhere-d20-baselines, built by scripts/make_d20_baselines.py) after the frozen scoring kernel stillhere-d20-score has finished.

Before any fetch it refuses unless the attached frozen tierD_d20_score.json exists, has status "scored" and names the posted code
MANIFEST, and the frozen code dataset (stillhere-d20-score-code) is attached unchanged. It then imports that frozen code read-only and
re-reads the public notthefall data with it: the same fail-closed allow-list (d20_keys.json), descriptor exclusions, FD QC, scored volumes
and onset clock (d20_score.run_one), for the versions the frozen run scored. Word rate comes from the onsets of the one speech-to-text pass
(D27) and the envelope from each version's audio, both in stillhere-d20-pred-asr's regressors.npz. Per listener-run: the raw gauges of
stillhere/cheap_baselines.py (TRIBE unpartialled, HRF x word rate, HRF x envelope; LANG and AUD) and, as a check that the re-read equals
the frozen run, the frozen T1. Per regressor: D20's own paired_test (intact > shortscram, and intact > longscram if the frozen run scored
it). Writes tierD_d20_baselines.json (scores and tests, no BOLD values) beside the frozen primary; the frozen output is only read."""
import datetime
import glob
import hashlib
import importlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from stillhere import cheap_baselines as CB
except ImportError:                                  # the flat Kaggle code dataset (stillhere-d20-baselines-code)
    import cheap_baselines as CB

FROZEN_MANIFEST = "92efa035b62a63d9badb5871716f86d7395cf0fecf544ba14ed3aadc3b8eab14"   # stillhere-d20-score-code as posted
FROZEN_KERNEL = "stillhere-d20-score"
T1_TOL = 1e-6                                        # re-read T1 vs frozen T1: same code and data, but BLAS order may differ by machine


class Refused(Exception):
    pass


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def frozen_result(inputs):
    """The attached frozen output: exactly one tierD_d20_score.json, written by stillhere-d20-score with status 'scored'."""
    ps = glob.glob(f"{inputs}/**/tierD_d20_score.json", recursive=True)
    if len(ps) != 1:
        raise Refused(f"{len(ps)} frozen tierD_d20_score.json attached (need 1: run only after {FROZEN_KERNEL} is COMPLETE)")
    fz = json.load(open(ps[0]))
    if fz.get("kernel") != FROZEN_KERNEL or fz.get("status") != "scored":
        raise Refused(f"frozen result has status {fz.get('status')!r}, not 'scored': no baseline")
    return ps[0], fz


def frozen_code(inputs, manifest):
    """Imports the attached frozen code dataset (the folder whose MANIFEST.sha256 hashes to `manifest`) without changing it."""
    roots = [os.path.dirname(p) for p in glob.glob(f"{inputs}/**/MANIFEST.sha256", recursive=True) if _sha(p) == manifest]
    if not roots:
        raise Refused(f"frozen code dataset with MANIFEST {manifest[:12]} not attached")
    sys.path.insert(0, roots[0])
    DS = importlib.import_module("d20_score")
    if any(os.path.dirname(os.path.abspath(m.__file__)) != os.path.abspath(roots[0]) for m in (DS, DS.G, DS.C, DS.D)):
        raise Refused("frozen modules were imported from somewhere other than the frozen code dataset")
    return roots[0], DS


def main(now=None, net=None, inputs="/kaggle/input", out="/kaggle/working/d20_baselines", manifest=FROZEN_MANIFEST, workers=4):
    t0 = time.time(); now = now or datetime.datetime.now(datetime.timezone.utc); os.makedirs(out, exist_ok=True); say_lines = []

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); say_lines.append(s)
    res = {"kernel": "stillhere-d20-baselines", "label": CB.LABEL, "rule": "docs/rules/cheap_baselines_rule.md",
           "started_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "regressors": list(CB.KINDS)}

    def finish(status):
        res.update(status=status, seconds=int(time.time() - t0), lines=say_lines)
        json.dump(res, open(f"{out}/tierD_d20_baselines.json", "w"), indent=1, default=float); return res

    def refuse(e):
        say("REFUSED:", e); return finish(f"refused: {e}")
    try:
        path, fz = frozen_result(inputs)
        if fz.get("code_manifest_sha256") != manifest:
            raise Refused("the frozen result names a different code MANIFEST")
        if any(re.search(r"stillhere-conf|ds003171", x) for x in glob.glob(f"{inputs}/**/*", recursive=True)):
            raise Refused("a forbidden input is mounted")
        root, DS = frozen_code(inputs, manifest)
    except Refused as e:
        return refuse(e)
    G, C = DS.G, DS.C
    try:
        if now < DS.parse(DS.START):
            raise DS.Refused(f"before {DS.START}")
        DS.check_manifest(root)
        asr_base, _ = DS.find_pred(inputs, DS.ASR_KERNEL, C.CFG["stock_ckpt_sha256"])
    except DS.Refused as e:
        return refuse(e)
    T1f = fz["T1_per_listener"]; scored = [v for v in DS.VERSIONS if v in T1f]
    res.update(frozen_result=os.path.relpath(path, inputs), frozen_status=fz["status"], code_manifest_sha256=manifest,
               posted_picker_manifest=fz.get("posted_picker_manifest"), scored_versions=scored)

    # re-read with the frozen code: allow-list, exclusions, QC, volumes and onset clock as the frozen run ------------------------------
    F = DS.Fetcher(json.load(open(f"{root}/d20_keys.json")), net or DS.Net(), f"{out}/_tmp")
    S = DS.sets_of(f"{root}/sets.npz")
    clips = {v: G.Clip(DS.VERSIONS[v], {k: np.load(f"{asr_base}/pred/{v}/pred_{k}.npz") for k in ("full", "audio")},
                       dict(np.load(f"{asr_base}/pred/{v}/regressors.npz")), S) for v in scored}
    items = {}
    for x in F.A.values():
        if x.get("task") in [DS.VERSIONS[v] for v in scored] and "listener" in x:
            items.setdefault((x["task"], x["listener"]), {})[x["kind"]] = x["key"]

    def one(item):
        (task, lst), ks = item
        try:
            rec, ser = DS.run_one(F, S, lst, task, ks)
        except DS.Refused:
            raise
        except Exception as e:
            return dict(listener=lst, task=task, status=f"error {type(e).__name__}"), None
        if ser is None or rec["status"] != "ok":
            return rec, None
        clip = clips[next(k for k, t in DS.VERSIONS.items() if t == task)]
        return rec, {"T1": G.listener_T1_T3(ser, rec["parts"], clip, rec["tr"])[0], **CB.run_gauges(G, C, ser, rec["parts"], clip, rec["tr"])}
    try:
        with ThreadPoolExecutor(workers) as ex:
            done = list(ex.map(one, sorted(items.items())))
    except DS.Refused as e:
        return refuse(e)
    res["bold_fetched"] = sum(1 for k in F.log if k.endswith(".func.gii"))
    by = {v: {} for v in scored}; recs = []
    for rec, g in done:
        recs.append({k: rec[k] for k in ("listener", "task", "status") if k in rec})
        if g is not None:
            by[next(k for k, t in DS.VERSIONS.items() if t == rec["task"])][rec["listener"]] = g
    res["runs"] = recs

    # the re-read must reproduce the frozen run's listeners and T1 ---------------------------------------------------------------------
    diff = [abs(by[v][x]["T1"] - T1f[v][x]) for v in scored for x in T1f[v] if x in by[v]]
    res["same_listeners_as_frozen"] = all(sorted(by[v]) == sorted(T1f[v]) for v in scored)
    res["T1_max_abs_diff_vs_frozen"] = max(diff) if diff else None
    res["matches_frozen"] = bool(res["same_listeners_as_frozen"] and diff and max(diff) <= T1_TOL)

    # D20's own test per regressor, beside the frozen TRIBE result ---------------------------------------------------------------------
    tests = {}
    for net_ in ("LANG", "AUD"):
        for k in CB.KINDS:
            sc = {v: {x: g[net_][k] for x, g in by[v].items()} for v in scored}
            t = {"intact_gt_shortscram": DS.paired_test(sc["intact"], sc["shortscram"])}
            if "longscram" in sc:
                t["intact_gt_longscram"] = DS.paired_test(sc["intact"], sc["longscram"])
            tests[f"{net_}_{k}"] = t
    res["tests"] = tests
    res["per_listener"] = {v: {x: {n_: g[n_] for n_ in ("LANG", "AUD")} for x, g in d.items()} for v, d in by.items()}
    res["frozen_primary_intact_gt_shortscram"] = fz.get("primary_intact_gt_shortscram")
    res["frozen_secondary_intact_gt_longscram"] = fz.get("secondary_intact_gt_longscram")
    fp = res["frozen_primary_intact_gt_shortscram"] or {}
    say(f"[{CB.LABEL}] frozen D20 primary (TRIBE T1): n {fp.get('n_pairs')}, p {fp.get('p')}, HL {fp.get('hl')} {fp.get('hl_ci95')} -> {fp.get('status')}")
    for name, t in tests.items():
        for comp, p in t.items():
            say(f"[{CB.LABEL}] D20 {comp} with {name}: n {p.get('n_pairs')}, k {p.get('k_intact_higher')}, p {p.get('p')}, "
                f"HL {p.get('hl')} {p.get('hl_ci95')} -> {p.get('status')}")
    say(f"[{CB.LABEL}] re-read matches the frozen run: {res['matches_frozen']} (listeners {res['same_listeners_as_frozen']}, "
        f"max |T1 - frozen T1| {res['T1_max_abs_diff_vs_frozen']})")
    return finish("scored" if res["matches_frozen"] else "scored, re-read differs from the frozen run")
