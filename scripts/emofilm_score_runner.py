"""Emo-FilM film scoring runner (docs/rules/emofilm_rule.md, version 2 with its Oct 28 additions; exploratory). Two stages, using only
the committed functions in stillhere/emofilm_score.py (and, through them, stillhere/heldout.py and stillhere/cheap_baselines.py):

  prepare  Per kept run of the run table: the image's volume count must equal the table's (else the run is dropped and reported); FD
           (Power) from the session's FEAT motion file, its rows located by acquisition order (scans.tsv) and the session runs' MRIQC
           volume counts: the file must be the session's runs concatenated (rows = sum of counts) or exactly one run whose count is the
           Tears of Steel run's and no other run's; otherwise the rule's fallback: no frame censoring and run exclusion by MRIQC fd_mean
           only. The image is projected to fsaverage5 exactly as the LPP harness (vol_to_surf, ball 3 mm, linear, the mid mesh between
           pial and white) and averaged into Schaefer-400 parcels (AUD, LANG, VIS, and all 400 for the face map). Saved per run to
           CACHE (outside the repo): the raw parcel rows over the whole run, FD and the censoring mask, plus a manifest with every
           input's and cache file's SHA-256. Refuses (drops nothing, writes no manifest) naming every run with a constant or non-finite
           AUD, LANG or VIS row (and, on the face path, any of the 400), each by viewer and parcel; on the face path it also writes the
           face test's UNINFORMATIVE with that reason (cloud ruling of Nov 1, 05:05 UTC, item 3), to be pushed at once, after which the
           read guard opens intact mode (AUD, LANG, VIS). No forecast is needed.
  score    With the forecast kernel's output (pred_full.npz, pred_audio.npz, words.csv, tierD_bridge_forecast.json): refuses unless the
           forecast files exist, each forecast is one row per second (emofilm_score.load_pred), the words table passes
           emofilm_score.check_words and every cache file's SHA-256 equals the manifest's. Per viewer: scored volumes (no lag search, the
           10 s drop), clean_full_then_crop, the D7 rule, the HEARING / LANGUAGE / VISUAL gauges and the four baselines; group medians
           with viewer bootstrap CIs, TRIBE minus each baseline, ISC(k) and the volunteer-equivalent rule. Group results go to
           results/emofilm_film_score.json; per-viewer values only to CACHE/score_per_viewer.json.

Read guard (rulings of Oct 29, 06:30 and 06:50 UTC): prepare opens no image or motion file unless, on origin/main of RC_REPO, EITHER
the face test's NOT RUN with its reason is committed (results/face_map.json; GATE_JSON is then not needed) OR gate 0 has passed and
results/face_map_RC.json is there with the local copy's blob hash. The intact film score does not otherwise depend on the face test.
The Emo-FilM download (pilot/emofilm/download_tos_runs.py) started Oct 29 05:45 UTC; it only downloads; nothing reads it before this
guard allows.
Prints counts, shapes, pass/fail and timings only (never BOLD values or per-viewer scores).
usage: python scripts/emofilm_score_runner.py prepare --data DIR --meta DIR --runs RUN_TABLE --atlas DIR --cache DIR --gate GATE_JSON
              --rc_repo REPO
       python scripts/emofilm_score_runner.py score --forecast DIR --regressors DIR --runs RUN_TABLE --atlas DIR --cache DIR [--results JSON]"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import emofilm_score as E  # noqa: E402
from stillhere import face_map as FM  # noqa: E402
from stillhere import heldout as H  # noqa: E402

TOS_SHA = "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"
NETS = ("AUD", "LANG", "VIS")
GAUGES = ("HEARING", "LANGUAGE", "VISUAL")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
RC_REL = "results/face_map_RC.json"


def refused(f, *x):
    """f(*x), its ValueError (a check of stillhere/emofilm_score.py that failed) turned into the runner's refusal."""
    try:
        return f(*x)
    except ValueError as e:
        raise SystemExit(f"refused: {e}") from None


# ---- read guard (project-thread rulings of Oct 29, 06:30 and 06:50 UTC) -------------------------------------------------------
FACE_REL = "results/face_map.json"
CRED = ("-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential")


def _git(rc_repo, *x):
    return subprocess.run(["git", "-C", rc_repo, *x], capture_output=True, text=True)


def fetch_origin(rc_repo):
    """None after a good fetch of origin main; "absent" when origin has no main branch; "fetch" when the fetch failed."""
    if _git(rc_repo, *CRED, "fetch", "-q", "origin", "main").returncode == 0:
        return None
    heads = _git(rc_repo, *CRED, "ls-remote", "--heads", "origin", "main")
    return "absent" if heads.returncode == 0 and not heads.stdout.strip() else "fetch"


def origin_json(rc_repo, rel):
    """The JSON file REL as committed on origin/main (after fetch_origin), or None when it is not there."""
    r = _git(rc_repo, "show", f"origin/main:{rel}")
    return json.loads(r.stdout) if r.returncode == 0 else None


def face_test_state(rc_repo) -> dict:
    """Whether the face test is closed on origin/main: results/face_map.json there with outcome NOT RUN and a reason (never revived), or
    with a scored outcome (scored once). Fetches first."""
    f = fetch_origin(rc_repo)
    if f == "fetch":
        raise SystemExit("refused: git fetch of origin main failed")
    fm = origin_json(rc_repo, FACE_REL) if f is None else None
    if not fm:
        return {"closed": False, "outcome": None, "why": None}
    return {"closed": True, "outcome": fm.get("outcome"), "why": fm.get("why") or fm.get("outcome")}


def read_guard(gate_json, rc_repo) -> dict:
    """Whether a BOLD file may be opened, on origin/main of RC_REPO (the private GitHub repo, after a fetch): EITHER the face test's
    NOT RUN with its reason (results/face_map.json) is there, OR gate 0 has passed and results/face_map_RC.json is there with the local
    copy's blob hash. An UNINFORMATIVE face test on origin/main (a constant or non-finite parcel row after BOLD was opened; cloud ruling
    of Nov 1, 05:05 UTC, item 3) opens it in intact mode: AUD, LANG and VIS only. Returns {ok, mode ("not_run", "intact" or "face"), kind
    (when not ok: "gate", "absent", "fetch", "differs"), why, ...}."""
    f = fetch_origin(rc_repo)
    if f == "fetch":
        return {"ok": False, "kind": "fetch", "why": "git fetch of origin main failed"}
    fm = origin_json(rc_repo, FACE_REL) if f is None else None
    if fm and fm.get("outcome") == "NOT RUN" and str(fm.get("why") or "").strip():
        return {"ok": True, "mode": "not_run", "kind": None, "why": f"face test NOT RUN on origin/main: {fm['why']}"}
    if fm and fm.get("outcome") == "UNINFORMATIVE" and str(fm.get("why") or "").strip():      # cloud ruling of Nov 1, 05:05 UTC, item 3
        return {"ok": True, "mode": "intact", "kind": None, "why": f"face test UNINFORMATIVE on origin/main: {fm['why']}"}
    if not os.path.isfile(gate_json) or json.load(open(gate_json)).get("pass") is not True:
        return {"ok": False, "kind": "gate", "why": "gate 0 (collinearity) has not passed and no NOT RUN is on origin/main"}
    if not os.path.isfile(os.path.join(rc_repo, RC_REL)):
        return {"ok": False, "kind": "absent", "why": f"{RC_REL} is not present"}
    remote = _git(rc_repo, "rev-parse", f"origin/main:{RC_REL}") if f is None else None
    if remote is None or remote.returncode:
        return {"ok": False, "kind": "absent", "why": f"{RC_REL} is not on origin/main"}
    if _git(rc_repo, "hash-object", RC_REL).stdout.strip() != remote.stdout.strip():
        return {"ok": False, "kind": "differs", "why": f"the local {RC_REL} differs from origin/main's"}
    when = _git(rc_repo, "log", "-1", "--format=%cI", "origin/main", "--", RC_REL).stdout.strip()
    return {"ok": True, "mode": "face", "kind": None, "why": None, "rc_blob": remote.stdout.strip(), "rc_commit_utc": when}


def write_uninformative(rc_repo, why, stage) -> dict:
    """RC_REPO/results/face_map.json with outcome UNINFORMATIVE and the named reason (viewer, parcel), when prepare or score refuses on a
    constant or non-finite parcel row after BOLD was opened (cloud ruling of Nov 1, 05:05 UTC, item 3). It is pushed at once; the read
    guard then lets prepare score the intact film in intact mode (AUD, LANG, VIS). Never over a face test already closed on origin/main."""
    st = face_test_state(rc_repo)
    if st["closed"]:
        raise SystemExit(f"refused: the face test is closed on origin/main ({st['outcome']}: {st['why']})")
    res = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "outcome": "UNINFORMATIVE", "why": why, "stage": stage,
           "ruling": "cloud ruling of Nov 1, 05:05 UTC, item 3", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(rc_repo, FACE_REL); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(res, open(p, "w"), indent=1)
    return res


def degenerate_named(rows: dict, names=None) -> str:
    """'AUD 1 of 7 (row 3), ALL 2 of 400 (17Networks_LH_VisCent_ExStr_1, ...)': per set, the constant or non-finite rows, named."""
    out = []
    for k, idx in rows.items():
        if len(idx[0]):
            nm = [names[i] for i in idx[0]] if k == "ALL" and names else [f"row {i}" for i in idx[0]]
            out.append(f"{k} {len(idx[0])} of {idx[1]} ({', '.join(nm)})")
    return ", ".join(out)


# ---- inputs -------------------------------------------------------------------------------------------------------------------
def session_runs(scans_tsv, mriqc: dict) -> list:
    """(bids_name, volume count) of the session's BOLD runs in acquisition order (scans.tsv acq_time; counts = MRIQC size_t)."""
    rows = sorted((r["acq_time"], r["filename"]) for r in csv.DictReader(open(scans_tsv), delimiter="\t")
                  if r["filename"].startswith("func/") and r["filename"].endswith("_bold.nii.gz"))
    names = [os.path.basename(f)[:-len(".nii.gz")] for _, f in rows]
    return [(n, int(float(mriqc[n]["size_t"]))) for n in names]


def motion_rows(n_rows: int, runs: list, tos_name: str):
    """The Tears of Steel run's rows in a session motion file, or None when they cannot be located unambiguously (rule fallback)."""
    names = [n for n, _ in runs]; counts = [c for _, c in runs]
    if tos_name not in names:
        return None
    i = names.index(tos_name)
    if n_rows == sum(counts):
        return E.run_rows(counts, i)                                   # the session's runs concatenated in acquisition order
    if n_rows == counts[i] and counts.count(counts[i]) == 1:
        return slice(0, n_rows)                                        # the file is that one run (no other run has its length)
    return None


def load_motion(path) -> np.ndarray:
    """MCFLIRT parameters (n, 6): whitespace-separated rows (rotations x, y, z in rad, then translations in mm); a text header is skipped."""
    rows = []
    for line in gzip.open(path, "rt") if str(path).endswith(".gz") else open(path):
        v = line.split()
        if not v:
            continue
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            continue
    p = np.asarray(rows, float)
    if p.ndim != 2 or p.shape[1] != 6:
        raise ValueError(f"motion file has shape {p.shape}, not (n, 6)")
    return p


def mid_meshes():
    """fsaverage5 mid meshes (pial-white midpoint coordinates, pial faces), exactly as the LPP harness built them."""
    from nilearn import datasets, surface
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5"); out = {}
    for h in ("left", "right"):
        p = surface.load_surf_mesh(fs5[f"pial_{h}"]); w = surface.load_surf_mesh(fs5[f"white_{h}"])
        out[h] = ((0.5 * (p.coordinates + w.coordinates), p.faces), None)
    return out


def parcel_index(atlas_dir) -> dict:
    lh, rh = (f"{atlas_dir}/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh"))
    sets = H.parcel_sets(lh, rh); sets["VIS"] = E.visual_sets(lh, rh)
    allp = FM.all_parcels(lh, rh)
    sets["ALL"], sets["ALL_names"] = allp["vertices"], allp["names"]              # all 400 parcels, for the face map (docs/rules/face_map_rule.md)
    return sets


# ---- prepare ------------------------------------------------------------------------------------------------------------------
def prepare_run(img_path, motion_path, row: dict, runs: list, mriqc_fd_mean: float, sets: dict, project, check=NETS) -> dict:
    """One run -> {status, n_vols, raw rows per network (p, n_vols), fd or None, keep (n_vols,), fd_source, degenerate}; degenerate:
    per set in `check`, its constant or non-finite rows (face_map.degenerate_rows), which cmd_prepare refuses on."""
    import nibabel as nib
    img = nib.load(img_path); n = int(img.shape[3])
    if n != int(row["n_vols"]):
        return {"status": f"dropped: image has {n} volumes, run table {row['n_vols']}"}
    tos = f"{row['subject']}_{row['session']}_task-TearsOfSteel_bold"
    params = load_motion(motion_path); sl = motion_rows(len(params), runs, tos)
    if sl is not None:
        fd = E.fd_power(params[sl]); keep = fd <= E.FD_MAX; src = "motion rows"
    else:
        fd = None; keep = np.ones(n, bool); src = "fallback: rows not locatable; no censoring; exclusion by MRIQC fd_mean"
    vert = np.asarray(project(img), np.float32)
    if vert.shape != (20484, n):
        raise ValueError(f"projection shape {vert.shape}")
    raw = E.parcel_means(vert, {k: sets[k] for k in NETS})
    if "ALL" in sets:
        raw["ALL"] = E.parcel_means(vert, {"ALL": sets["ALL"]})["ALL"]           # the face map's series; the intact score reads NETS only
    rows = {k: FM.degenerate_rows(raw[k]) for k in check if k in raw}
    return {"status": "prepared", "n_vols": n, "raw": raw, "fd": fd, "keep": keep, "fd_source": src, "mriqc_fd_mean": mriqc_fd_mean,
            "degenerate": {k: int(v.sum()) for k, v in rows.items()},
            "degenerate_rows": {k: (np.where(v)[0].tolist(), int(len(v))) for k, v in rows.items()}}


def cmd_prepare(a):
    guard = read_guard(a.gate, a.rc_repo)                              # before any image or motion file is opened
    if not guard["ok"]:
        raise SystemExit(f"refused: {guard['why']}; no BOLD file is opened")
    os.makedirs(a.cache, exist_ok=True); assert not os.path.abspath(a.cache).startswith(REPO), "CACHE must be outside the repo"
    T = json.load(open(a.runs)); mriqc = {r["bids_name"]: r for r in csv.DictReader(open(f"{a.meta}/mriqc_group_bold.tsv"), delimiter="\t")}
    sets = parcel_index(a.atlas); meshes = mid_meshes(); project = lambda img: E.project(img, meshes)
    man = {"label": E.LABEL, "rule": "docs/rules/emofilm_rule.md", "runs_table_sha256": sha(a.runs), "runs": {}}; t0 = time.time()
    kept = [r for r in T["runs"] if r["keep"] is True]; flat = {}
    check = NETS + ("ALL",) if guard["mode"] == "face" else NETS     # ALL is the face map's only; not_run scores the intact film alone
    for r in kept:
        sub, ses = r["subject"], r["session"]; t1 = time.time()
        img = f"{a.data}/{sub}_{ses}_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz"
        mot = f"{a.data}/{sub}_{ses}_task-mc_space-orig_desc-motion_bold.tsv.gz"
        runs = session_runs(f"{a.meta}/{sub}/{ses}/{sub}_{ses}_scans.tsv", mriqc)
        res = prepare_run(img, mot, r, runs, float(mriqc[f"{sub}_{ses}_task-TearsOfSteel_bold"]["fd_mean"]), sets, project, check)
        ent = {"status": res["status"], "image_sha256": sha(img), "motion_sha256": sha(mot)}
        if res["status"] == "prepared":
            out = f"{a.cache}/{sub}_{ses}.npz"
            np.savez(out, **{f"raw_{k}": v for k, v in res["raw"].items()}, keep=res["keep"],
                     fd=res["fd"] if res["fd"] is not None else np.zeros(0))
            ent.update(cache=os.path.basename(out), cache_sha256=sha(out), n_vols=res["n_vols"], fd_source=res["fd_source"],
                       mriqc_fd_mean=res["mriqc_fd_mean"], censored_frames=int((~res["keep"]).sum()),
                       parcels={k: int(v.shape[0]) for k, v in res["raw"].items()})
            if any(res["degenerate"].values()):
                flat[f"{sub}_{ses}"] = degenerate_named(res["degenerate_rows"], sets.get("ALL_names"))
        man["runs"][f"{sub}_{ses}"] = ent
        print(f"{sub}_{ses}: {ent['status']} ({time.time() - t1:.0f} s)", flush=True)
    if flat:                                                             # refuse, never drop: no manifest, so no score
        msg = (f"{len(flat)} of {len(kept)} viewers have constant or non-finite parcel rows (nothing is dropped): "
               + "; ".join(f"{k}: {v}" for k, v in flat.items()))
        if guard["mode"] == "face":                                      # BOLD is open: the face test is UNINFORMATIVE (item 3)
            write_uninformative(a.rc_repo, f"prepare: {msg}", "prepare")
            raise SystemExit(f"refused: {msg}. The face test is UNINFORMATIVE: commit and push {FACE_REL} at once, then rerun prepare, "
                             "which scores the intact film in intact mode (AUD, LANG, VIS)")
        raise SystemExit(f"refused: {msg}")
    json.dump(man, open(f"{a.cache}/prepare_manifest.json", "w"), indent=1)
    prepared = [v for v in man["runs"].values() if v["status"] == "prepared"]
    print(f"prepared {len(prepared)} of {len(kept)} kept runs; dropped {len(kept) - len(prepared)}; fd sources "
          f"{sorted({v['fd_source'] for v in prepared})}; censored frames total {sum(v['censored_frames'] for v in prepared)}; "
          f"parcels {prepared[0]['parcels'] if prepared else None}; {time.time() - t0:.0f} s")


# ---- score --------------------------------------------------------------------------------------------------------------------
def forecast_dir(d) -> str:
    """The folder holding the intact kernel's files: D itself, or D/bridge (the layout `kaggle kernels output` gives)."""
    return d if os.path.isfile(f"{d}/pred_full.npz") or not os.path.isfile(f"{d}/bridge/pred_full.npz") else f"{d}/bridge"


def load_forecast(d) -> dict:
    d = forecast_dir(d)
    need = ("pred_full.npz", "pred_audio.npz", "words.csv", "tierD_bridge_forecast.json")
    miss = [f for f in need if not os.path.isfile(f"{d}/{f}")]
    if miss:
        raise SystemExit(f"refused: forecast files missing: {miss}")
    t = json.load(open(f"{d}/tierD_bridge_forecast.json"))
    if t["stimulus"]["sha256"] != TOS_SHA or t.get("ckpt_matches_pin") is not True:
        raise SystemExit("refused: the forecast is not the stock checkpoint on the official release")
    full, audio = (refused(E.load_pred, f"{d}/pred_{k}.npz") for k in ("full", "audio"))     # one row per second, in order, finite
    with open(f"{d}/words.csv") as f:
        words = list(csv.DictReader(f))
    return {"full": full, "audio": audio, "onsets": [float(r["start"]) for r in words], "words": words,
            "visual": bool(t["stages"].get("video", {}).get("ok")), "sha256": {f: sha(f"{d}/{f}") for f in need}}


def score_viewer(c: dict, r: dict, fc: dict, per_s: dict, sets: dict, mriqc_fd_mean: float) -> dict:
    """One viewer with the committed functions; returns gauges, baseline gauges, grid series and the exclusion verdict."""
    tr, onset, so = float(r["tr"]), float(r["film_onset_s"]), float(r["slice_offset_s"])
    Pf, sf = fc["full"]; Pa, sa = fc["audio"]
    idx, ft, ts = E.film_volumes(len(c["keep"]), tr, onset, so, sf)
    if c["fd"].size:
        ok, why = E.run_pass(c["fd"][idx]), "D7 on scored frames"
    else:
        ok, why = mriqc_fd_mean <= E.FD_MAX, "MRIQC fd_mean only (motion rows not locatable)"
    if not ok:
        return {"excluded": why, "n_scored": int(len(idx))}
    bold, k = {}, None
    for net in NETS:
        bold[net], k = E.clean_full_then_crop(c[f"raw_{net}"], tr, c["keep"], idx)
    pa = E.parcel_means(E.tribe_at(Pa, sa, ts), {n: sets[n] for n in ("AUD", "LANG")})
    pf = E.parcel_means(E.tribe_at(Pf, sf, ts), {n: sets[n] for n in ("LANG", "VIS")})
    pred = {"audio": pa, "full": pf}
    regs = E.baseline_regressors(per_s, ft)
    g = E.viewer_scores(bold, pred, regs["word_rate"], k, tr, fc["visual"])
    b = E.baseline_gauges(bold, pred, regs, k, tr, fc["visual"])
    grid = E.common_grid(tr)
    return {"excluded": None, "n_scored": int(len(idx)), "n_kept": int(k.sum()), "gauges": g, "baselines": b,
            "grid": {net: E.to_grid(bold[net], ft, k, grid) for net in NETS}}


def isc_controls(fc: dict, sets: dict, per_s: dict, tr: float) -> dict:
    """LANGUAGE's controls on the common grid (the LANG-mean audio-only prediction and HRF x word rate), cleaned like the gauges'."""
    grid = E.common_grid(tr); Pa, sa = fc["audio"]
    lang_a = E.parcel_means(E.tribe_at(Pa, sa, grid - E.TRIBE_OFFSET), {"LANG": sets["LANG"]})["LANG"].mean(0, keepdims=True)
    wr = np.atleast_2d(H.hrf_regressor(per_s["word_rate"], grid))
    return {"HEARING": [], "LANGUAGE": [E.clean(lang_a, tr), E.clean(wr, tr)], "VISUAL": []}


def cmd_score(a):
    fc = load_forecast(a.forecast)
    man = json.load(open(f"{a.cache}/prepare_manifest.json"))
    if sha(a.runs) != man["runs_table_sha256"]:
        raise SystemExit("refused: the run table differs from the one prepare used")
    bad = [k for k, v in man["runs"].items() if v["status"] == "prepared" and sha(f"{a.cache}/{v['cache']}") != v["cache_sha256"]]
    if bad:
        raise SystemExit(f"refused: {len(bad)} cache files differ from the prepare manifest")
    T = {f"{r['subject']}_{r['session']}": r for r in json.load(open(a.runs))["runs"]}
    rz = np.load(f"{a.regressors}/film_regressors.npz"); per_s = {k: rz[k] for k in ("envelope", "brightness", "motion")}
    n_s = len(per_s["envelope"]); refused(E.check_words, fc["words"], n_s)                    # the segment is the regressors' length
    per_s["word_rate"] = E.word_rate_per_s(fc["onsets"], n_s)
    sets = parcel_index(a.atlas); t0 = time.time(); per = {}
    for key, v in man["runs"].items():
        if v["status"] != "prepared":
            continue
        c = dict(np.load(f"{a.cache}/{v['cache']}"))
        per[key] = score_viewer(c, T[key], fc, per_s, sets, v["mriqc_fd_mean"])
    scored = {k: v for k, v in per.items() if v["excluded"] is None}
    tr = float(next(iter(T.values()))["tr"])
    res = {"label": E.LABEL, "rule": "docs/rules/emofilm_rule.md", "forecast_sha256": fc["sha256"], "visual_ran": fc["visual"],
           "viewers": {"prepared": len(per), "scored": len(scored), "excluded": {k: v["excluded"] for k, v in per.items() if v["excluded"]}},
           "fd_sources": sorted({v["fd_source"] for v in man["runs"].values() if v["status"] == "prepared"}), "gauges": {}, "isc": {}}
    ctl = isc_controls(fc, sets, per_s, tr)
    for gname, net in zip(GAUGES, NETS):
        rows = [v["gauges"][gname] for v in scored.values() if isinstance(v["gauges"][gname], dict)]
        if not rows:
            res["gauges"][gname] = "not run"; continue
        zs = [x["z"] for x in rows]
        res["gauges"][gname] = {**E.group(zs), "share_detected": float(np.mean([x["detected"] for x in rows]))}
        curve = E.isc_curve({k: v["grid"][net] for k, v in scored.items()}, ctl[gname])
        res["isc"][gname] = {"curve": curve, "volunteer_equivalent": E.volunteer_equivalent(zs, curve)}
    res["baselines"] = E.compare_baselines([v["baselines"] for v in scored.values()])
    res["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
    json.dump(res, open(a.results, "w"), indent=1, default=float)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in per.items()}, open(f"{a.cache}/score_per_viewer.json", "w"),
              indent=1, default=float)
    print(f"scored {len(scored)} of {len(per)} viewers; excluded {len(per) - len(scored)}; visual {fc['visual']}; {res['seconds']} s")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); [p.add_argument(f"--{k}", required=True) for k in ("data", "meta", "runs", "atlas", "cache", "gate", "rc_repo")]
    s = sp.add_parser("score"); [s.add_argument(f"--{k}", required=True) for k in ("forecast", "regressors", "runs", "atlas", "cache")]
    s.add_argument("--results", default=f"{REPO}/results/emofilm_film_score.json")
    a = ap.parse_args()
    cmd_prepare(a) if a.cmd == "prepare" else cmd_score(a)


if __name__ == "__main__":
    main()
