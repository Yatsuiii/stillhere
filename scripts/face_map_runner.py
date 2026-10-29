"""Face map runner (docs/rules/face_map_rule.md; exploratory). Two commands:

  gate   Gate 0, stimulus only (no brain data): the face-area fraction F from the SAM 3.1 edit masks (edits.json) through the brightness
         code path, and the VIF of F against brightness, motion, envelope and word rate (film_regressors.npz and the intact kernel's
         words.csv) on the intact rule's common grid; at most 5, else NOT RUN. Writes GATE_JSON.
  score  Refuses unless gate 0 passed, it is before Fri Oct 30 20:00 UTC, the film-versions forecasts and the intact forecast exist,
         and every cache file's SHA-256 equals the prepare manifest's (scripts/emofilm_score_runner.py prepare, the same cache the intact
         scorer reads). Per viewer: the intact rule's scored volumes, exclusion and clean_full_then_crop on all 400 parcels; M_i;
         R and C from the film versions; gate 1, B1, B2, the outcome. Group results to RESULTS; the per-viewer maps only to CACHE.
         A refusal of the first three kinds writes the outcome NOT RUN; a cache mismatch stops without a result.
Prints counts, outcomes, p values and timings only (never BOLD values or per-viewer maps).
usage: python scripts/face_map_runner.py gate --edits edits.json --regressors DIR --words words.csv --runs RUN_TABLE --out GATE_JSON
       python scripts/face_map_runner.py score --versions DIR --forecast DIR --edits edits.json --regressors DIR --runs RUN_TABLE
              --atlas DIR --cache DIR --gate GATE_JSON [--results JSON] [--now ISO]"""
import argparse
import csv
import datetime
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

DEADLINE = datetime.datetime(2026, 10, 30, 20, 0, tzinfo=datetime.timezone.utc)
_spec = importlib.util.spec_from_file_location("efr", os.path.join(REPO, "scripts/emofilm_score_runner.py"))
ER = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ER)
sha = ER.sha


def per_second_series(edits_json, regressors_dir, words_csv) -> dict:
    """Per-second face, brightness, motion, envelope and word rate over the release window the edits cover."""
    ed = json.load(open(edits_json)); assert ed["window_release_s"] == [0.0, 190.0] and ed["source_sha256"] == ER.TOS_SHA
    face = FM.face_per_s([f["masked_px"] for f in ed["per_frame"]], ed["fps"])
    rz = np.load(f"{regressors_dir}/film_regressors.npz"); n = len(rz["envelope"])
    with open(words_csv) as f:
        onsets = [float(r["start"]) for r in csv.DictReader(f)]
    return {"face": face[:n], "brightness": rz["brightness"], "motion": rz["motion"], "envelope": rz["envelope"],
            "word_rate": E.word_rate_per_s(onsets, n)}


def cmd_gate(a):
    tr = float(json.load(open(a.runs))["runs"][0]["tr"])
    g = FM.gate0(per_second_series(a.edits, a.regressors, a.words), tr)
    g.update(inputs_sha256={"edits": sha(a.edits), "regressors": sha(f"{a.regressors}/film_regressors.npz"), "words": sha(a.words)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(g, open(a.out, "w"), indent=1)
    print(f"gate 0: VIF(face) {g['vif_face']:.2f} (max {g['vif_max']}) -> {'pass' if g['pass'] else 'NOT RUN'}")


def not_run(a, why):
    res = {"label": FM.LABEL, "rule": "docs/rules/face_map_rule.md", "outcome": "NOT RUN", "why": why}
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True); json.dump(res, open(a.results, "w"), indent=1)
    print(f"face map: NOT RUN ({why})")
    return res


def load_versions(d) -> dict:
    out = {}
    for v in ("intact", "faces_removed", "area_matched"):
        p = [x for x in (f"{d}/{v}/pred_full.npz", f"{d}/versions/{v}/pred_full.npz") if os.path.isfile(x)]
        if not p:
            return {}
        z = np.load(p[0]); out[v] = (z["P"], z["starts"])
    return out


def cmd_score(a):
    now = datetime.datetime.fromisoformat(a.now) if a.now else datetime.datetime.now(datetime.timezone.utc)
    if now > DEADLINE:
        return not_run(a, "not finished by Fri Oct 30 20:00 UTC")
    if not os.path.isfile(a.gate) or not json.load(open(a.gate)).get("pass"):
        return not_run(a, "gate 0 (collinearity) missing or failed")
    vers = load_versions(a.versions)
    if not vers:
        return not_run(a, "film-versions forecasts missing")
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
    parcels = FM.all_parcels(*(f"{a.atlas}/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    if len(parcels["vertices"]) != 400 or int(parcels["tested"].sum()) != 374:
        raise SystemExit(f"refused: the atlas gives {int(parcels['tested'].sum())} tested parcels, not the rule's 374")
    per_s = per_second_series(a.edits, a.regressors, f"{a.forecast}/words.csv"); tr = float(next(iter(T.values()))["tr"]); t0 = time.time()
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
    P_i, st = vers["intact"]
    R = FM.forecast_map(P_i, vers["faces_removed"][0], st, parcels, tr); C = FM.forecast_map(P_i, vers["area_matched"][0], st, parcels, tr)
    perms = FM.spin_perms(spin_centroids(parcels), parcels["hemi"])
    res = FM.evaluate(R, C, M, parcels, perms)
    res.update(rule="docs/rules/face_map_rule.md", excluded=excluded, gate0=json.load(open(a.gate)), seconds=round(time.time() - t0))
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True); json.dump(res, open(a.results, "w"), indent=1, default=float)
    np.savez(f"{a.cache}/face_maps_per_viewer.npz", M=M, viewers=np.array(sorted(Ms)), R=R, C=C)
    print(f"face map: {res['outcome']} ({len(Ms)} viewers, {len(excluded)} excluded, {res['seconds']} s)")
    return res


def spin_centroids(parcels):
    from nilearn import datasets, surface
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    sph = [surface.load_surf_mesh(fs5[f"sphere_{h}"]).coordinates for h in ("left", "right")]
    return FM.centroids(sph[0], sph[1], parcels)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); [g.add_argument(f"--{k}", required=True) for k in ("edits", "regressors", "words", "runs", "out")]
    s = sp.add_parser("score")
    [s.add_argument(f"--{k}", required=True) for k in ("versions", "forecast", "edits", "regressors", "runs", "atlas", "cache", "gate")]
    s.add_argument("--results", default=f"{REPO}/results/face_map.json"); s.add_argument("--now", default="")
    a = ap.parse_args()
    cmd_gate(a) if a.cmd == "gate" else cmd_score(a)


if __name__ == "__main__":
    main()
