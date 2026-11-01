"""Face map runner (docs/rules/face_map_rule.md; exploratory). Every file it writes goes to RC_REPO/results/, RC_REPO being the clone of
the private GitHub repo (stillhere-pilot-public, origin github.com/Yatsuiii/stillhere, branch main) whose origin/main the read guard
checks; the chain commits and pushes what it writes. Four commands:

  gate     Gate 0, stimulus only: the face-area fraction F from the final film stimuli's edits_v2.json (block-snapped SAM 3.1 face
           regions, whole frame; hash-checked against the stimuli MANIFEST d4e27d55) through the brightness code path, and the VIF of F
           against brightness, motion, envelope and word rate on the common grid; at most 5, else the face test is NOT RUN (exit
           code 3, and the JSON's "pass" key is false). --words takes the intact kernel's words.csv or the folder its output was
           downloaded to (then its own words.csv, not export/'s copy); refused when emofilm_score.check_words fails. Writes --out
           (outside the repo is fine).
  maps     Forecasts only, no brain data: R = intact - faces removed and C = intact - area matched per Schaefer-400 parcel, from the
           committed intact film kernel's forecast and the two edit kernels' forecasts, on the intact rule's grid (film time 10 to 185 s,
           read at 5 to 180 s, linear); plus the DESCRIPTIVE (never decides): R and C on only the grid points whose 4 s video clip has
           every in-view face controlled. Refuses unless the three kernels' own records check out (kernel_checks) and each forecast is
           one row per second (emofilm_score.load_pred). Writes RC_REPO/results/face_map_RC.json with each map's SHA-256, the inputs'
           SHA-256 and the kernel checks.
  not-run  Writes RC_REPO/results/face_map.json with outcome NOT RUN and its reason. Once it is on origin/main the face test is closed
           for good: maps, score and not-run refuse, and the read guard lets the intact film be scored alone.
  score    Refuses when the face test is closed on origin/main (NOT RUN, or already scored: scored once); needs the read guard's face
           path (gate 0 passed and face_map_RC.json on origin/main with the local blob hash) and that file's commit no later than Sun Nov
           1 18:00 UTC (otherwise: run not-run, push it, then score the intact film alone). Reads R and C from face_map_RC.json only;
           the forecast's words.csv and pred_full.npz must be the ones gate 0 and maps hashed; cache hashes as prepared; per viewer
           the intact rule's scored volumes, exclusion and clean_full_then_crop on all 400 parcels; M_i (refused, nothing dropped, on
           any constant or non-finite cleaned row or M_i, and then written UNINFORMATIVE with the viewers and parcels named, to be pushed
           at once: cloud ruling of Nov 1, 05:05 UTC, item 3); gate 1, B1, B2, the outcome; reports the descriptive and the stimulus checks
           beside it. Writes RC_REPO/results/face_map.json (group results); per-viewer maps only to CACHE.
The Emo-FilM download (pilot/emofilm/download_tos_runs.py) started Oct 29 05:45 UTC; it only downloads; nothing reads it before the
read guard allows. Prints counts, outcomes, p values, hashes and timings only (never BOLD values or per-viewer maps).
usage: python scripts/face_map_runner.py gate --edits edits_v2.json --regressors DIR --words FILE_OR_DIR --runs RUN_TABLE --out GATE_JSON
       python scripts/face_map_runner.py maps --intact DIR --removed DIR --matched DIR --edits edits_v2.json --runs RUN_TABLE --atlas DIR
              --rc_repo REPO
       python scripts/face_map_runner.py not-run --reason TEXT --rc_repo REPO
       python scripts/face_map_runner.py score --forecast DIR --edits edits_v2.json --regressors DIR --runs RUN_TABLE --atlas DIR
              --cache DIR --gate GATE_JSON --rc_repo REPO"""
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

NO_GPU_CUTOFF = datetime.datetime(2026, 10, 30, 20, 0, tzinfo=datetime.timezone.utc)   # rules and code; the gate runs Saturday instead
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


def out_path(rc_repo, rel) -> str:
    p = os.path.join(rc_repo, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def refuse_if_closed(rc_repo):
    st = ER.face_test_state(rc_repo)
    if st["closed"]:
        raise SystemExit(f"refused: the face test is closed on origin/main ({st['outcome']}: {st['why']})")


# ---- stimulus side ------------------------------------------------------------------------------------------------------------
def stimulus_manifest(edits_json) -> dict:
    """{file name: SHA-256} of the MANIFEST.sha256 beside the final film stimuli's edits_v2.json, which must be the stimuli's committed
    one (docs/rules/film_stimulus_rule.md)."""
    man = os.path.join(os.path.dirname(os.path.abspath(edits_json)), "MANIFEST.sha256")
    if not os.path.isfile(man) or sha(man) != FILM_STIM_MANIFEST:
        raise SystemExit("refused: the film stimuli MANIFEST is missing or not the committed one")
    return {l.split()[1]: l.split()[0] for l in open(man) if l.strip()}


def stimulus_frames(edits_json) -> list:
    """Per frame of release 0 to 190 s, its faces from the final film stimuli's edits_v2.json, which must match its hash in the
    committed stimuli MANIFEST beside it (stimulus_manifest)."""
    listed = stimulus_manifest(edits_json)
    if listed.get(os.path.basename(edits_json)) != sha(edits_json):
        raise SystemExit("refused: edits_v2.json differs from the film stimuli MANIFEST")
    pf = sorted(json.load(open(edits_json))["per_frame"], key=lambda f: f["frame"])
    if [f["frame"] for f in pf] != list(range(N_FRAMES)):
        raise SystemExit(f"refused: edits_v2.json does not cover frames 0..{N_FRAMES - 1}")
    return [f["faces"] for f in pf]


def words_csv(p) -> str:
    """The intact kernel's words.csv: P itself, or the forecast folder's own (emofilm_score_runner.forecast_dir: P or P/bridge), never
    the copy in its export/ (the edit kernels' input; a recursive search finds both)."""
    f = p if os.path.isfile(p) else os.path.join(ER.forecast_dir(p), "words.csv")
    if os.path.basename(f) != "words.csv" or not os.path.isfile(f):
        raise SystemExit(f"refused: no words.csv at {f}")
    return f


def per_second_series(edits_json, regressors_dir, words) -> tuple:
    """Per-second face (block-snapped px over the whole frame: what viewers saw), brightness, motion, envelope and word rate over
    release 0 to 190 s, and the words table's check (emofilm_score.check_words on the regressors' length; refused when it fails)."""
    face = FM.face_per_s([[x["px"] for x in faces] for faces in stimulus_frames(edits_json)], 24.0)
    rz = np.load(f"{regressors_dir}/film_regressors.npz"); n = len(rz["envelope"])
    with open(words_csv(words)) as f:
        rows = list(csv.DictReader(f))
    chk = ER.refused(E.check_words, rows, n); onsets = [float(r["start"]) for r in rows]
    return {"face": face[:n], "brightness": rz["brightness"], "motion": rz["motion"], "envelope": rz["envelope"],
            "word_rate": E.word_rate_per_s(onsets, n)}, chk


def cmd_gate(a):
    tr = float(json.load(open(a.runs))["runs"][0]["tr"]); per_s, words = per_second_series(a.edits, a.regressors, a.words)
    g = FM.gate0(per_s, tr)
    g.update(inputs_sha256={"edits": sha(a.edits), "regressors": sha(f"{a.regressors}/film_regressors.npz"), "words": sha(words_csv(a.words))},
             words=words)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(g, open(a.out, "w"), indent=1)
    print(f"gate 0: VIF(face) {g['vif_face']:.2f} (max {g['vif_max']}) -> {'pass' if g['pass'] else 'NOT RUN'}")
    return g


# ---- maps ---------------------------------------------------------------------------------------------------------------------
KERNEL_JSON = {"intact": "tierD_bridge_forecast.json", "removed": "tierD_film_rem_forecast.json", "matched": "tierD_film_am_forecast.json"}
VIDEO = {"intact": "seg_intact.mp4", "removed": "faces_removed_v2.mp4", "matched": "area_matched_v2.mp4"}


def kernel_checks(files: dict, edits_json) -> dict:
    """The three kernels' own records (the JSON beside each pred_full.npz), checked before R and C exist: each forecast is the stock
    checkpoint on its own cut (by name and by the stimuli MANIFEST's hash: swapped folders would flip B2's sign) with its video, predict
    and predict_audio stages ok; both edits ran on this intact run's export (an intact rerun between the edit kernels changes it) and
    added files to video-extractor caches only; one environment (pip freeze, Hugging Face pins, each JSON's downloaded revisions at the pins,
    GPU); and the intact transcript is free of the chunked speech-to-text's copies. Refuses naming every failed check; returns all."""
    J = {}
    for k, f in files.items():
        p = os.path.join(os.path.dirname(f), KERNEL_JSON[k])
        if not os.path.isfile(p):
            raise SystemExit(f"refused: no {KERNEL_JSON[k]} beside the {k} forecast {f}")
        J[k] = json.load(open(p))
    listed, I = stimulus_manifest(edits_json), J["intact"]; w = I.get("words") or {}; exp = (I.get("export") or {}).get("manifest_sha256")
    seg = {"intact": (I.get("stimulus") or {}).get("segment"), "removed": J["removed"].get("video"), "matched": J["matched"].get("video")}
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    chk = {}
    for k, j in J.items():
        pins, revs = j.get("hf_pins"), j.get("hf_revisions")
        chk[f"{k}_video"] = seg[k] == VIDEO[k]
        chk[f"{k}_segment_sha256"] = VIDEO[k] in listed and (j.get("stimulus") or {}).get("segment_sha256") == listed[VIDEO[k]]
        chk[f"{k}_ckpt_matches_pin"] = j.get("ckpt_matches_pin") is True
        chk[f"{k}_stages_ok"] = all(((j.get("stages") or {}).get(s) or {}).get("ok") is True for s in ("video", "predict", "predict_audio"))
        chk[f"{k}_hf_revisions_pinned"] = isinstance(pins, dict) and isinstance(revs, dict) and all(revs[r] == pins[r] for r in revs if r in pins)
    for k in ("removed", "matched"):
        cd, fd = J[k].get("cache_dirs_added"), J[k].get("cache_files_added_by_dir")    # exca names dirs per extractor class, so a stage
        vid = lambda xs: bool(xs) and all(str(c).startswith("neuralset.extractors.video.") for c in xs)   # that missed adds files, not dirs
        chk[f"{k}_on_intact_export"] = bool(exp) and J[k].get("intact_export_manifest_sha256") == exp
        chk[f"{k}_video_caches_only"] = isinstance(cd, list) and vid(cd)
        chk[f"{k}_files_added_video_only"] = isinstance(fd, dict) and vid(fd)
    for e in ("pip_freeze_sha256", "hf_pins", "gpu"):
        chk[f"same_{e}"] = bool(I.get(e)) and all(j.get(e) == I.get(e) for j in J.values())
    chk["intact_words_no_shift_copies"] = num(w.get("exact_shift_copies")) and w["exact_shift_copies"] == 0
    chk["intact_words_inside_segment"] = num(w.get("last_start_s")) and num(w.get("segment_seconds")) and w["last_start_s"] < w["segment_seconds"]
    bad = [c for c, ok in chk.items() if not ok]
    if bad:
        raise SystemExit(f"refused: the kernels' records do not check out: {', '.join(bad)}")
    return {"checks": chk, "segment_sha256": {k: J[k]["stimulus"]["segment_sha256"] for k in J}, "intact_export_manifest_sha256": exp,
            "cache_dirs_added": {k: J[k]["cache_dirs_added"] for k in ("removed", "matched")},
            "cache_files_added_by_dir": {k: J[k]["cache_files_added_by_dir"] for k in ("removed", "matched")}, "pip_freeze_sha256": I["pip_freeze_sha256"],
            "hf_pins": I["hf_pins"], "hf_revisions": {k: j["hf_revisions"] for k, j in J.items()}, "gpu": I["gpu"], "intact_words": w}


def cmd_maps(a):
    refuse_if_closed(a.rc_repo)
    files = {k: find_one(getattr(a, k), "pred_full.npz") for k in ("intact", "removed", "matched")}
    kc = kernel_checks(files, a.edits)
    P = {k: ER.refused(E.load_pred, f)[0] for k, f in files.items()}; st = np.arange(len(P["intact"]), dtype=float)   # load_pred: 0..n-1
    if any(P[k].shape != P["intact"].shape for k in ("removed", "matched")):
        raise SystemExit("refused: the three forecasts do not share sample times and shape")
    parcels = atlas_parcels(a.atlas); tr = float(json.load(open(a.runs))["runs"][0]["tr"]); grid = E.common_grid(tr)
    R = FM.forecast_map(P["intact"], P["removed"], st, parcels, tr)
    C = FM.forecast_map(P["intact"], P["matched"], st, parcels, tr)
    keep = FM.controlled_grid(FM.frames_fully_controlled(stimulus_frames(a.edits)), tr)
    desc = {"label": "descriptive: never decides", "definition": "grid points whose 4 s video clip (ending at film time - 5 s) has every "
            "in-view face controlled (edits_v2.json: control_px > 0 wherever px_in_view > 0)", "grid_points_kept": int(keep.sum())}
    if keep.any():
        Rd = FM.forecast_map(P["intact"], P["removed"], st, parcels, tr, keep)
        Cd = FM.forecast_map(P["intact"], P["matched"], st, parcels, tr, keep)
        desc.update(R=Rd.tolist(), C=Cd.tolist(), R_sha256=map_sha(Rd), C_sha256=map_sha(Cd))
    out = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "forecasts_only": True,
           "grid": {"film_time_s": [float(grid[0]), float(grid[-1])], "step_s": tr, "points": int(len(grid)), "prediction_read_at": "film time - 5 s, linear"},
           "parcel_names_sha256": hashlib.sha256("\n".join(parcels["names"]).encode()).hexdigest(),
           "inputs_sha256": {**{k: sha(f) for k, f in files.items()}, "edits_v2": sha(a.edits)}, "kernel_checks": kc, "R": R.tolist(), "C": C.tolist(),
           "R_sha256": map_sha(R), "C_sha256": map_sha(C), "descriptive": desc,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    p = out_path(a.rc_repo, ER.RC_REL); json.dump(out, open(p, "w"), indent=1)
    print(f"maps: R and C over 400 parcels on {len(grid)} grid points (descriptive on {int(keep.sum())}); R {out['R_sha256'][:16]}, "
          f"C {out['C_sha256'][:16]}; commit and push {ER.RC_REL} in {a.rc_repo}")
    return out


# ---- not run ------------------------------------------------------------------------------------------------------------------
def cmd_not_run(a):
    if not str(a.reason or "").strip():
        raise SystemExit("refused: NOT RUN needs a reason")
    refuse_if_closed(a.rc_repo)
    res = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "outcome": "NOT RUN", "why": a.reason.strip(),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    p = out_path(a.rc_repo, ER.FACE_REL); json.dump(res, open(p, "w"), indent=1)
    print(f"face map: NOT RUN ({res['why']}); commit and push {ER.FACE_REL} in {a.rc_repo}; the intact film is then scored alone")
    return res


# ---- score --------------------------------------------------------------------------------------------------------------------
def stimulus_checks(rc_repo) -> dict:
    """Beside the result: control coverage, eccentricity and change-energy ratio (results/film_stimulus_checks.json) and the
    close-ups without a control."""
    p = [x for x in (os.path.join(rc_repo, "results/film_stimulus_checks.json"), os.path.join(REPO, "results/film_stimulus_checks.json")) if os.path.isfile(x)]
    if not p:
        return {"found": False}
    d = json.load(open(p[0])); c = d["checks"]["control_area_within_1pct"]; r = d["reported"]
    return {"found": True, "in_view_face_px_with_control": c["in_view_face_px_with_control"], "faces_in_view": c["faces_in_view"],
            "no_control": c["no_control"], "no_control_note": "the largest close-ups, whose in-view part fits nowhere else in TRIBE's view",
            "median_eccentricity_view_px": r["median_eccentricity_view_px"], "change_energy_ratio_control_over_face": r["change_energy_ratio_control_over_face"]}


def cmd_score(a):
    refuse_if_closed(a.rc_repo)
    guard = ER.read_guard(a.gate, a.rc_repo)
    if not guard["ok"] or guard["mode"] != "face":
        raise SystemExit(f"refused: {guard['why'] or 'the face path of the read guard is not open'}")
    if datetime.datetime.fromisoformat(guard["rc_commit_utc"]) > RC_CUTOFF:
        raise SystemExit(f"refused: R and C reached origin/main at {guard['rc_commit_utc']}, after Sun Nov 1 18:00 UTC: run not-run, push it, "
                         "then score the intact film alone")
    rc = json.load(open(os.path.join(a.rc_repo, ER.RC_REL)))
    R, C = np.asarray(rc["R"], float), np.asarray(rc["C"], float)
    if map_sha(R) != rc["R_sha256"] or map_sha(C) != rc["C_sha256"]:
        raise SystemExit("refused: R or C does not match its recorded SHA-256")
    parcels = atlas_parcels(a.atlas)
    if hashlib.sha256("\n".join(parcels["names"]).encode()).hexdigest() != rc["parcel_names_sha256"]:
        raise SystemExit("refused: the atlas parcel order differs from the maps'")
    fc = ER.load_forecast(a.forecast); g0 = json.load(open(a.gate))
    if fc["sha256"]["words.csv"] != g0["inputs_sha256"]["words"]:
        raise SystemExit("refused: the forecast's words.csv differs from the one gate 0 used")
    if fc["sha256"]["pred_full.npz"] != rc["inputs_sha256"]["intact"]:
        raise SystemExit("refused: the forecast's pred_full.npz differs from the intact forecast the maps used")
    man = json.load(open(f"{a.cache}/prepare_manifest.json"))
    if sha(a.runs) != man["runs_table_sha256"]:
        raise SystemExit("refused: the run table differs from the one prepare used")
    bad = [k for k, v in man["runs"].items() if v["status"] == "prepared" and sha(f"{a.cache}/{v['cache']}") != v["cache_sha256"]]
    if bad:
        raise SystemExit(f"refused: {len(bad)} cache files differ from the prepare manifest")
    T = {f"{r['subject']}_{r['session']}": r for r in json.load(open(a.runs))["runs"]}
    per_s, _ = per_second_series(a.edits, a.regressors, ER.forecast_dir(a.forecast)); tr = float(next(iter(T.values()))["tr"]); t0 = time.time()
    Ms, excluded, flat = {}, {}, {}
    for key, v in man["runs"].items():
        if v["status"] != "prepared":
            continue
        c = dict(np.load(f"{a.cache}/{v['cache']}")); r = T[key]
        idx, ft, ts = E.film_volumes(len(c["keep"]), tr, float(r["film_onset_s"]), float(r["slice_offset_s"]), fc["full"][1])
        ok = E.run_pass(c["fd"][idx]) if c["fd"].size else v["mriqc_fd_mean"] <= E.FD_MAX
        if not ok:
            excluded[key] = "D7" if c["fd"].size else "MRIQC fd_mean"; continue
        B, k = E.clean_full_then_crop(c["raw_ALL"], tr, c["keep"], idx); flat[key] = FM.degenerate_rows(B)
        regs = E.baseline_regressors({b: per_s[b] for b in E.BASELINES}, ft)
        cl = lambda x: E.clean(np.atleast_2d(x), tr, k)
        Ms[key] = FM.viewer_map(B, cl(FM.face_regressor(per_s["face"], ft)), [cl(regs[b]) for b in ("brightness", "motion", "envelope", "word_rate")])
    M = np.stack([Ms[k] for k in sorted(Ms)]); bad = np.stack([flat[k] for k in sorted(Ms)]) | ~np.isfinite(M)
    if bad.any():                                                       # refuse, never drop: one such row fails gate 1 for good
        named = "; ".join(f"{k}: {', '.join(parcels['names'][j] for j in np.where(b)[0])}" for k, b in zip(sorted(Ms), bad) if b.any())
        msg = (f"{int(bad.any(1).sum())} of {len(M)} viewers have a constant or non-finite cleaned BOLD row or M_i, in "
               f"{int(bad.any(0).sum())} of {M.shape[1]} parcels; no viewer or parcel is dropped ({named})")
        ER.write_uninformative(a.rc_repo, f"score: {msg}", "score")    # the face test is UNINFORMATIVE (cloud ruling of Nov 1, 05:05 UTC, item 3)
        raise SystemExit(f"refused: {msg}. The face test is UNINFORMATIVE: commit and push {ER.FACE_REL} at once, then score the intact "
                         "film with prepare in intact mode (AUD, LANG, VIS)")
    perms = FM.spin_perms(spin_centroids(parcels), parcels["hemi"])
    res = FM.evaluate(R, C, M, parcels, perms)
    S = np.asarray(parcels["tested"], bool); Mbar = M.mean(0); dsc = rc.get("descriptive", {})
    if "R" in dsc:
        Rd, Cd = np.asarray(dsc["R"], float), np.asarray(dsc["C"], float)
        dsc_rep = {"label": "descriptive: never decides", "grid_points_kept": dsc["grid_points_kept"],
                   "rho_R_Mbar": FM.spearman(Rd[S], Mbar[S]), "rho_C_Mbar": FM.spearman(Cd[S], Mbar[S]),
                   "median_d": float(np.median([FM.spearman(Rd[S], m[S]) - FM.spearman(Cd[S], m[S]) for m in M]))}
    else:
        dsc_rep = {"label": "descriptive: never decides", "grid_points_kept": dsc.get("grid_points_kept", 0), "note": "no grid point qualified"}
    res.update(rule="docs/rules/face_map_rule.md", excluded=excluded, gate0=g0, descriptive=dsc_rep,
               stimulus_checks=stimulus_checks(a.rc_repo), rc={"blob": guard["rc_blob"], "commit_utc": guard["rc_commit_utc"],
               "R_sha256": rc["R_sha256"], "C_sha256": rc["C_sha256"]}, seconds=round(time.time() - t0))
    json.dump(res, open(out_path(a.rc_repo, ER.FACE_REL), "w"), indent=1, default=float)
    np.savez(f"{a.cache}/face_maps_per_viewer.npz", M=M, viewers=np.array(sorted(Ms)))
    print(f"face map: {res['outcome']} ({len(Ms)} viewers, {len(excluded)} excluded, {res['seconds']} s); commit and push {ER.FACE_REL}")
    return res


def spin_centroids(parcels):
    from nilearn import datasets, surface
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    sph = [surface.load_surf_mesh(fs5[f"sphere_{h}"]).coordinates for h in ("left", "right")]
    return FM.centroids(sph[0], sph[1], parcels)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); [g.add_argument(f"--{k}", required=True) for k in ("edits", "regressors", "words", "runs", "out")]
    m = sp.add_parser("maps"); [m.add_argument(f"--{k}", required=True) for k in ("intact", "removed", "matched", "edits", "runs", "atlas", "rc_repo")]
    n = sp.add_parser("not-run"); [n.add_argument(f"--{k}", required=True) for k in ("reason", "rc_repo")]
    s = sp.add_parser("score")
    [s.add_argument(f"--{k}", required=True) for k in ("forecast", "edits", "regressors", "runs", "atlas", "cache", "gate", "rc_repo")]
    a = ap.parse_args()
    out = {"gate": cmd_gate, "maps": cmd_maps, "not-run": cmd_not_run, "score": cmd_score}[a.cmd](a)
    if a.cmd == "gate" and not out["pass"]:
        sys.exit(3)                                                     # gate 0 failed: the chain runs not-run, not the edit kernels


if __name__ == "__main__":
    main()
