"""Face map runner (docs/rules/face_map_rule.md; exploratory). Three commands:

  maps   Forecasts only, no brain data: R = intact - faces removed and C = intact - area matched per Schaefer-400 parcel, from the
         committed intact film kernel's forecast and the two edit kernels' forecasts (each pred_full.npz, the intact kernel's format),
         averaged over the intact rule's scored film-time grid (prediction read at film time - 5 s, linear). Writes face_map_RC.json with
         both maps, each map's SHA-256 and the input forecasts' SHA-256. It must be committed and pushed before any BOLD is read.
  gate   Gate 0, stimulus only: the face-area fraction F from the final film stimuli's edits_v2.json (block-snapped SAM 3.1 face regions,
         whole frame; hash-checked against the stimuli MANIFEST d4e27d55) through the brightness code path, and
         the VIF of F against brightness, motion, envelope and word rate on the common grid; at most 5, else NOT RUN. --words takes the
         intact kernel's words.csv or the folder its output was downloaded to (Saturday's layout, words.csv under bridge/).
  score  Reads R and C from face_map_RC.json only (never recomputes them). The read guard first (scripts/emofilm_score_runner.py
         read_guard): gate 0 passed and face_map_RC.json on origin/main of RC_REPO with the local blob hash; a local copy that differs
         stops without a result. NOT RUN when gate 0 failed, the maps are not on origin/main, or the commit that put them there is later
         than Sun Nov 1 18:00 UTC. Then every cache file's SHA-256 must equal the prepare manifest's; per viewer the intact rule's scored
         volumes, exclusion and clean_full_then_crop on all 400 parcels; M_i; gate 1, B1, B2, the outcome. Group results to RESULTS;
         per-viewer maps only to CACHE.
The Emo-FilM download (pilot/emofilm/download_tos_runs.py) started Oct 29 05:45 UTC; it only downloads; nothing reads it before the
read guard allows. Prints counts, outcomes, p values, hashes and timings only (never BOLD values or per-viewer maps).
usage: python scripts/face_map_runner.py maps --intact DIR --removed DIR --matched DIR --runs RUN_TABLE --atlas DIR [--out JSON]
       python scripts/face_map_runner.py gate --edits edits_v2.json --regressors DIR --words FILE_OR_DIR --runs RUN_TABLE --out GATE_JSON
       python scripts/face_map_runner.py score --forecast DIR --edits edits_v2.json --regressors DIR --runs RUN_TABLE --atlas DIR
              --cache DIR --gate GATE_JSON --rc_repo REPO [--results JSON]"""
import argparse
import csv
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import emofilm_score as E  # noqa: E402
from stillhere import face_map as FM  # noqa: E402

NO_GPU_CUTOFF = datetime.datetime(2026, 10, 30, 20, 0, tzinfo=datetime.timezone.utc)   # the no-GPU items (code, rules, gate code)
RC_CUTOFF = datetime.datetime(2026, 11, 1, 18, 0, tzinfo=datetime.timezone.utc)        # R and C on origin/main, else NOT RUN
FILM_STIM_MANIFEST = "d4e27d55e2c462f8edd9d2373407692d03534893a3fac284a6819897f52a6f44"   # the final film stimuli (v2)
N_FRAMES = 4560                                                                   # release 0 to 190 s at 24 fps
_spec = importlib.util.spec_from_file_location("efr", os.path.join(REPO, "scripts/emofilm_score_runner.py"))
ER = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ER)
sha = ER.sha


def find_one(d, name) -> str:
    """The single file called NAME at or under D (the download layouts differ); refuses when there are none or several."""
    hits = [d] if os.path.isfile(d) and os.path.basename(d) == name else sorted(glob.glob(f"{d}/**/{name}", recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"refused: {len(hits)} files called {name} under {d} (need exactly 1)")
    return hits[0]


def map_sha(a) -> str:
    return hashlib.sha256(np.asarray(a, "<f8").tobytes()).hexdigest()


def atlas_parcels(atlas) -> dict:
    parcels = FM.all_parcels(*(f"{atlas}/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    if len(parcels["vertices"]) != 400 or int(parcels["tested"].sum()) != 374:
        raise SystemExit(f"refused: the atlas gives {int(parcels['tested'].sum())} tested parcels, not the rule's 374")
    return parcels


# ---- maps ---------------------------------------------------------------------------------------------------------------------
def cmd_maps(a):
    files = {k: find_one(getattr(a, k), "pred_full.npz") for k in ("intact", "removed", "matched")}
    P = {k: np.load(f) for k, f in files.items()}
    st = P["intact"]["starts"]
    if any(not np.array_equal(P[k]["starts"], st) or P[k]["P"].shape != P["intact"]["P"].shape for k in ("removed", "matched")):
        raise SystemExit("refused: the three forecasts do not share sample times and shape")
    parcels = atlas_parcels(a.atlas); tr = float(json.load(open(a.runs))["runs"][0]["tr"])
    R = FM.forecast_map(P["intact"]["P"], P["removed"]["P"], st, parcels, tr)
    C = FM.forecast_map(P["intact"]["P"], P["matched"]["P"], st, parcels, tr)
    grid = E.common_grid(tr)
    out = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "forecasts_only": True,
           "grid": {"film_time_s": [float(grid[0]), float(grid[-1])], "step_s": tr, "points": int(len(grid)), "prediction_read_at": "film time - 5 s, linear"},
           "parcel_names_sha256": hashlib.sha256("\n".join(parcels["names"]).encode()).hexdigest(),
           "inputs_sha256": {k: sha(f) for k, f in files.items()}, "R": R.tolist(), "C": C.tolist(), "R_sha256": map_sha(R), "C_sha256": map_sha(C),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(f"maps: R and C over 400 parcels on {len(grid)} grid points; R {out['R_sha256'][:16]}, C {out['C_sha256'][:16]}; commit and push {a.out}")
    return out


# ---- gate ---------------------------------------------------------------------------------------------------------------------
def face_px_per_frame(edits_json) -> list:
    """Per frame of release 0 to 190 s, the block-snapped SAM 3.1 face-region pixel counts (`px`, the whole frame: what viewers saw) from
    the final film stimuli's edits_v2.json, which must be listed with its hash in the MANIFEST.sha256 beside it, and that MANIFEST must
    be the stimuli's committed one (docs/rules/film_stimulus_rule.md)."""
    d = os.path.dirname(os.path.abspath(edits_json)); man = os.path.join(d, "MANIFEST.sha256")
    if not os.path.isfile(man) or sha(man) != FILM_STIM_MANIFEST:
        raise SystemExit("refused: the film stimuli MANIFEST is missing or not the committed one")
    listed = {l.split()[1]: l.split()[0] for l in open(man) if l.strip()}
    if listed.get(os.path.basename(edits_json)) != sha(edits_json):
        raise SystemExit("refused: edits_v2.json differs from the film stimuli MANIFEST")
    pf = sorted(json.load(open(edits_json))["per_frame"], key=lambda f: f["frame"])
    if [f["frame"] for f in pf] != list(range(N_FRAMES)):
        raise SystemExit(f"refused: edits_v2.json does not cover frames 0..{N_FRAMES - 1}")
    return [[x["px"] for x in f["faces"]] for f in pf]


def per_second_series(edits_json, regressors_dir, words) -> dict:
    """Per-second face, brightness, motion, envelope and word rate over release 0 to 190 s (the segment the edits and forecasts cover)."""
    face = FM.face_per_s(face_px_per_frame(edits_json), 24.0)
    rz = np.load(f"{regressors_dir}/film_regressors.npz"); n = len(rz["envelope"])
    with open(find_one(words, "words.csv")) as f:
        onsets = [float(r["start"]) for r in csv.DictReader(f)]
    return {"face": face[:n], "brightness": rz["brightness"], "motion": rz["motion"], "envelope": rz["envelope"],
            "word_rate": E.word_rate_per_s(onsets, n)}


def cmd_gate(a):
    tr = float(json.load(open(a.runs))["runs"][0]["tr"])
    g = FM.gate0(per_second_series(a.edits, a.regressors, a.words), tr)
    g.update(inputs_sha256={"edits": sha(a.edits), "regressors": sha(f"{a.regressors}/film_regressors.npz"), "words": sha(find_one(a.words, "words.csv"))})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(g, open(a.out, "w"), indent=1)
    print(f"gate 0: VIF(face) {g['vif_face']:.2f} (max {g['vif_max']}) -> {'pass' if g['pass'] else 'NOT RUN'}")


# ---- score --------------------------------------------------------------------------------------------------------------------
def not_run(a, why):
    res = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "outcome": "NOT RUN", "why": why}
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True); json.dump(res, open(a.results, "w"), indent=1)
    print(f"face map: NOT RUN ({why})")
    return res


def cmd_score(a):
    guard = ER.read_guard(a.gate, a.rc_repo)
    if not guard["ok"]:
        if guard["kind"] in ("differs", "fetch"):
            raise SystemExit(f"refused: {guard['why']}")
        return not_run(a, guard["why"])
    if datetime.datetime.fromisoformat(guard["rc_commit_utc"]) > RC_CUTOFF:
        return not_run(a, f"R and C reached origin/main at {guard['rc_commit_utc']}, after Sun Nov 1 18:00 UTC")
    rc = json.load(open(os.path.join(a.rc_repo, ER.RC_REL)))
    R, C = np.asarray(rc["R"], float), np.asarray(rc["C"], float)
    if map_sha(R) != rc["R_sha256"] or map_sha(C) != rc["C_sha256"]:
        raise SystemExit("refused: R or C does not match its recorded SHA-256")
    parcels = atlas_parcels(a.atlas)
    if hashlib.sha256("\n".join(parcels["names"]).encode()).hexdigest() != rc["parcel_names_sha256"]:
        raise SystemExit("refused: the atlas parcel order differs from the maps'")
    try:
        fc = ER.load_forecast(a.forecast)
    except SystemExit as e:
        return not_run(a, f"intact forecast: {e}")
    man = json.load(open(f"{a.cache}/prepare_manifest.json"))
    if sha(a.runs) != man["runs_table_sha256"]:
        raise SystemExit("refused: the run table differs from the one prepare used")
    bad = [k for k, v in man["runs"].items() if v["status"] == "prepared" and sha(f"{a.cache}/{v['cache']}") != v["cache_sha256"]]
    if bad:
        raise SystemExit(f"refused: {len(bad)} cache files differ from the prepare manifest")
    T = {f"{r['subject']}_{r['session']}": r for r in json.load(open(a.runs))["runs"]}
    per_s = per_second_series(a.edits, a.regressors, ER.forecast_dir(a.forecast)); tr = float(next(iter(T.values()))["tr"]); t0 = time.time()
    Ms, excluded = {}, {}
    for key, v in man["runs"].items():
        if v["status"] != "prepared":
            continue
        c = dict(np.load(f"{a.cache}/{v['cache']}")); r = T[key]
        idx, ft, ts = E.film_volumes(len(c["keep"]), tr, float(r["film_onset_s"]), float(r["slice_offset_s"]), fc["full"][1])
        ok = E.run_pass(c["fd"][idx]) if c["fd"].size else v["mriqc_fd_mean"] <= E.FD_MAX
        if not ok:
            excluded[key] = "D7" if c["fd"].size else "MRIQC fd_mean"; continue
        B, k = E.clean_full_then_crop(c["raw_ALL"], tr, c["keep"], idx)
        regs = E.baseline_regressors({b: per_s[b] for b in E.BASELINES}, ft)
        cl = lambda x: E.clean(np.atleast_2d(x), tr, k)
        Ms[key] = FM.viewer_map(B, cl(FM.face_regressor(per_s["face"], ft)), [cl(regs[b]) for b in ("brightness", "motion", "envelope", "word_rate")])
    M = np.stack([Ms[k] for k in sorted(Ms)])
    perms = FM.spin_perms(spin_centroids(parcels), parcels["hemi"])
    res = FM.evaluate(R, C, M, parcels, perms)
    res.update(rule="docs/rules/face_map_rule.md", excluded=excluded, gate0=json.load(open(a.gate)), rc={"blob": guard["rc_blob"],
               "commit_utc": guard["rc_commit_utc"], "R_sha256": rc["R_sha256"], "C_sha256": rc["C_sha256"]}, seconds=round(time.time() - t0))
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True); json.dump(res, open(a.results, "w"), indent=1, default=float)
    np.savez(f"{a.cache}/face_maps_per_viewer.npz", M=M, viewers=np.array(sorted(Ms)))
    print(f"face map: {res['outcome']} ({len(Ms)} viewers, {len(excluded)} excluded, {res['seconds']} s)")
    return res


def spin_centroids(parcels):
    from nilearn import datasets, surface
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    sph = [surface.load_surf_mesh(fs5[f"sphere_{h}"]).coordinates for h in ("left", "right")]
    return FM.centroids(sph[0], sph[1], parcels)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("maps"); [m.add_argument(f"--{k}", required=True) for k in ("intact", "removed", "matched", "runs", "atlas")]
    m.add_argument("--out", default=f"{REPO}/{ER.RC_REL}")
    g = sp.add_parser("gate"); [g.add_argument(f"--{k}", required=True) for k in ("edits", "regressors", "words", "runs", "out")]
    s = sp.add_parser("score")
    [s.add_argument(f"--{k}", required=True) for k in ("forecast", "edits", "regressors", "runs", "atlas", "cache", "gate", "rc_repo")]
    s.add_argument("--results", default=f"{REPO}/results/face_map.json")
    a = ap.parse_args()
    {"maps": cmd_maps, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
