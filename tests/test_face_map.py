"""The face map (stillhere/face_map.py, scripts/face_map_runner.py) on synthetic data only: no Emo-FilM file, no forecast, no SAM
mask. Parcel masks, the face regressor, the partial correlation, k*, spin permutations, every outcome, the lead check, gate 0, and the
runner's NOT RUN paths, its cache refusal and one planted end-to-end score."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nilearn")
nib = pytest.importorskip("nibabel")
from stillhere import emofilm_score as E  # noqa: E402
from stillhere import face_map as FM  # noqa: E402
from stillhere import heldout as H  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def write_annots(d: Path):
    """Two fsaverage5-sized annotations, 200 parcels each in contiguous vertex blocks; names carry the project's set patterns."""
    import nibabel.freesurfer as fs
    for h in ("lh", "rh"):
        labels = np.minimum(np.arange(10242) // 51, 199) + 1
        names = ["Background+FreeSurfer_Defined_Medial_Wall"]
        for i in range(200):
            n_aud = 4 if h == "lh" else 3                                   # 7 auditory parcels and 19 left language parcels, as Schaefer-400
            n = "SomMotB_Aud" if i < n_aud else "TempPar" if i < n_aud + 19 else "VisCent" if i < 49 else "Other"
            names.append(f"17Networks_{'LH' if h == 'lh' else 'RH'}_{n}_{i}")
        ctab = np.array([[i % 255, (3 * i) % 255, (7 * i) % 255, 0, i] for i in range(201)])
        fs.write_annot(str(d / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot"), labels, ctab, names, fill_ctab=True)
    return d


def test_all_parcels_masks(tmp_path):
    P = FM.all_parcels(*(write_annots(tmp_path) / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    assert len(P["vertices"]) == 400 and (P["hemi"] == 1).sum() == 200 and P["vertices"][200].min() >= 10242
    assert (~P["tested"]).sum() == 4 + 3 + 19 and P["tested"].sum() == 374     # AUD both hemispheres, LANG (TempPar) left only
    assert P["visual"].sum() == (49 - 23) + (49 - 22)


def test_face_regressor_and_partial_fz():
    px = [[0] * 2] * 24 + [[1280 * 534 // 2]] * 24 + [[]] * 24
    ps = FM.face_per_s(px); assert np.allclose(ps, [0.0, 0.5, 0.0])
    assert FM.face_regressor(np.zeros(10), np.arange(5.0)).shape == (5,)
    rng = np.random.default_rng(0); x = rng.standard_normal(200); ctl = rng.standard_normal(200)
    B = np.vstack([x + 0.3 * rng.standard_normal(200), ctl + 0.1 * rng.standard_normal(200)])
    z = FM.partial_fz(B, x, [ctl]); assert z[0] > 1.5 and abs(z[1]) < 0.5
    with pytest.raises(ValueError):
        FM.partial_fz(B[:1], x, [x])                              # controlling for x itself leaves nothing to correlate


def test_kstar_and_spin_perms():
    assert FM.kstar(28) == 19 and FM.kstar(20) == 15
    rng = np.random.default_rng(1); c = rng.standard_normal((400, 3)); c /= np.linalg.norm(c, axis=1, keepdims=True)
    hemi = np.repeat([0, 1], 200); perms = FM.spin_perms(c, hemi, n=20, seed=3)
    assert all(sorted(p[:200]) == list(range(200)) and sorted(p[200:]) == list(range(200, 400)) for p in perms)
    assert (FM.spin_perms(c, hemi, n=20, seed=3) == perms).all() and (perms != np.arange(400)).any()


def synth(kind, n=12, seed=5):
    rng = np.random.default_rng(seed); R = rng.standard_normal(400); C = rng.standard_normal(400)
    tested = np.ones(400, bool); tested[:26] = False; visual = np.zeros(400, bool); visual[26:76] = True
    c = rng.standard_normal((400, 3)); c /= np.linalg.norm(c, axis=1, keepdims=True)
    parcels = {"tested": tested, "visual": visual, "hemi": np.repeat([0, 1], 200)}
    base = {"pass": R, "fail": -R, "nonspecific": R + C, "noise": np.zeros(400)}[kind]
    M = np.stack([base + (0.8 if kind != "noise" else 1.0) * rng.standard_normal(400) for _ in range(n)])
    return R, C, M, parcels, FM.spin_perms(c, parcels["hemi"], n=200, seed=7)


@pytest.mark.parametrize("kind,outcome", [("pass", "PASS"), ("fail", "FAIL"), ("nonspecific", "FAIL not face-specific"), ("noise", "UNINFORMATIVE")])
def test_outcomes(kind, outcome):
    res = FM.evaluate(*synth(kind))
    assert res["outcome"] == outcome and res["parcels_tested"] == 374
    if outcome == "PASS":
        assert res["film_leads_video"] and res["caption"] is None and res["B1"]["kstar"] == 10


def test_lead_check_carried_by_visual():
    R, C, M, parcels, _ = synth("pass")
    nv = parcels["tested"] & ~parcels["visual"]
    M2 = M.copy(); M2[:, nv] = C[nv] + 0.5 * np.random.default_rng(2).standard_normal((len(M), nv.sum()))
    med, lead = FM.lead_check(R, C, M2, parcels); assert med < 0 and not lead


def test_gate0_vif():
    rng = np.random.default_rng(4); ps = {k: rng.random(190) for k in ("face", "brightness", "motion", "envelope", "word_rate")}
    assert FM.gate0(ps, 1.3)["pass"]
    ps["face"] = ps["brightness"] + 0.01 * rng.random(190)
    g = FM.gate0(ps, 1.3); assert not g["pass"] and g["vif_face"] > 5


# ---- the runner --------------------------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("fmr", ROOT / "scripts/face_map_runner.py")
RN = importlib.util.module_from_spec(spec); spec.loader.exec_module(RN)
NOHOOK = ["-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]


def git(repo, *args, env=None):
    import os
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *NOHOOK, *args], capture_output=True, text=True, check=True, env={**os.environ, **(env or {})})


def rc_repo(tmp: Path) -> Path:
    """A clone whose origin is a local bare repo, standing in for the private GitHub repo."""
    import subprocess
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(tmp / "origin.git")], check=True)
    subprocess.run(["git", "clone", "-q", str(tmp / "origin.git"), str(tmp / "rc")], check=True, capture_output=True)
    return tmp / "rc"


SEGS = {"seg_intact.mp4": "1" * 64, "faces_removed_v2.mp4": "2" * 64, "area_matched_v2.mp4": "3" * 64}    # the stimuli MANIFEST's cuts
PINS = {"facebook/tribev2": "a" * 40, "facebook/vjepa2-vitg-fpc64-256": "b" * 40}


def kernel_records(fb: Path, rem: Path, am: Path, words: dict):
    """The three kernels' own JSON records in the new kernels' format: one environment, the edits on the intact run's export adding
    only video-extractor caches; hf_revisions differ in their keys (only the intact run downloads its speech-to-text model)."""
    env = {"pip_freeze_sha256": "4" * 64, "hf_pins": PINS, "gpu": "Tesla T4"}
    stages = {s: {"ok": True} for s in ("video", "audio", "text", "predict", "predict_audio")}; exp = RN.sha(fb / "export" / "MANIFEST.sha256")
    json.dump({"stimulus": {"sha256": RN.ER.TOS_SHA, "segment": "seg_intact.mp4", "segment_sha256": SEGS["seg_intact.mp4"]}, "ckpt_matches_pin": True,
               "stages": {"events": {"ok": True}, **stages}, "export": {"manifest_sha256": exp}, **env,
               "hf_revisions": {**PINS, "openai/whisper-large-v3": "c" * 40}, "words": words}, open(fb / "tierD_bridge_forecast.json", "w"))
    for d, name, video in ((rem, "film_rem", "faces_removed_v2.mp4"), (am, "film_am", "area_matched_v2.mp4")):
        json.dump({"video": video, "stimulus": {"sha256": RN.ER.TOS_SHA, "segment": video, "segment_sha256": SEGS[video]}, "ckpt_matches_pin": True,
                   "stages": stages, "intact_export_manifest_sha256": exp, "cache_dirs_added": ["neuralset.extractors.video.HuggingFaceVideo_x1"],
                   "cache_files_added_by_dir": {"neuralset.extractors.video.HuggingFaceVideo_x1": 2160},
                   **env, "hf_revisions": {"facebook/tribev2": PINS["facebook/tribev2"]}}, open(d / f"tierD_{name}_forecast.json", "w"))


def build(tmp: Path, monkeypatch, n_view=8, n_vols=260, tr=1.3):
    """Synthetic edits_v2 (some in-view faces without a control), regressors, the intact forecast in Saturday's bridge/ layout (its
    export/ holding a second words.csv), the two edit forecasts in film_rem/ and film_am/ with the three kernels' records, run table
    and prepare cache with planted face tracking; then gate 0 and maps (into a git clone, not yet pushed)."""
    rng = np.random.default_rng(9); (tmp / "atlas").mkdir(); atlas = write_annots(tmp / "atlas")
    parcels = FM.all_parcels(*(atlas / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    d = {k: tmp / k for k in ("fc", "rem", "am", "rg", "cache")}
    for p in d.values():
        p.mkdir()
    (d["fc"] / "bridge" / "export").mkdir(parents=True); (d["rem"] / "film_rem").mkdir(); (d["am"] / "film_am").mkdir()
    faces = np.repeat(np.clip(np.convolve(rng.random(190), np.ones(5) / 5, "same") - 0.4, 0, None), 24)    # per frame, 190 s
    stim = tmp / "stim"; stim.mkdir()                                                   # the final film stimuli's edits_v2.json format
    uncontrolled = (np.arange(len(faces)) // 24) % 7 == 3                               # every 7th second: a close-up with no control
    pf = [{"frame": i, "faces": [{"px": int(f * 1280 * 534), "px_in_view": int(f * 470 * 470), "control_px": 0 if uncontrolled[i] else int(f * 470 * 470),
                                  "pieces": 1}] if f > 0 else []} for i, f in enumerate(faces)]
    json.dump({"seg": "seg_intact.mp4", "seg_sha256": "0" * 64, "view": [404, 32, 873, 501], "per_frame": pf[::-1]}, open(stim / "edits_v2.json", "w"))
    (stim / "MANIFEST.sha256").write_text("".join(f"{h}  {n}\n" for n, h in SEGS.items()) + f"{RN.sha(stim / 'edits_v2.json')}  edits_v2.json\n")
    monkeypatch.setattr(RN, "FILM_STIM_MANIFEST", RN.sha(stim / "MANIFEST.sha256"))
    np.savez(d["rg"] / "film_regressors.npz", envelope=rng.random(190), brightness=rng.random(190), motion=rng.random(190))
    t = np.arange(190.0); base = rng.standard_normal((190, 20484)).astype(np.float32); fb = d["fc"] / "bridge"
    ws = np.sort(rng.uniform(0, 189, 300))
    for p in (fb / "words.csv", fb / "export" / "words.csv"):                              # the export holds a copy for the edit kernels
        p.write_text("start,duration,text\n" + "".join(f"{s:.2f},0.3,w\n" for s in ws))
    (fb / "export" / "MANIFEST.sha256").write_text(f"{RN.sha(fb / 'export' / 'words.csv')}  words.csv\n")
    np.savez(fb / "pred_full.npz", P=base, starts=t); np.savez(fb / "pred_audio.npz", P=base, starts=t)
    rem, am = d["rem"] / "film_rem", d["am"] / "film_am"
    kernel_records(fb, rem, am, {"n": 300, "last_start_s": float(np.round(ws[-1], 2)), "segment_seconds": 190.0, "exact_shift_copies": 0,
                                 "worst_shift_s": None, "worst_shift_count": 0})
    w = rng.standard_normal(400); off = np.zeros(20484, np.float32)
    for p, v in enumerate(parcels["vertices"]):
        off[v] = w[p]
    np.savez(rem / "pred_full.npz", P=base - off, starts=t)
    np.savez(am / "pred_full.npz", P=base - rng.standard_normal(20484).astype(np.float32), starts=t)
    face_s = FM.face_per_s([[x["px"] for x in f["faces"]] for f in pf]); runs, man = [], {"runs": {}}
    for i in range(n_view):
        sub = f"sub-T{i:02d}"; onset = 94.0 + 0.1 * i; ft = np.arange(n_vols) * tr + 0.6 - onset
        F = H.hrf_regressor(face_s, ft); F = (F - F.mean()) / (F.std() + 1e-9)
        np.savez(d["cache"] / f"{sub}_ses-1.npz", raw_ALL=np.outer(w, F) + rng.standard_normal((400, n_vols)), keep=np.ones(n_vols, bool), fd=np.zeros(0))
        runs.append({"subject": sub, "session": "ses-1", "keep": True, "tr": tr, "film_onset_s": onset, "slice_offset_s": 0.6, "n_vols": n_vols})
        man["runs"][f"{sub}_ses-1"] = {"status": "prepared", "cache": f"{sub}_ses-1.npz", "cache_sha256": RN.sha(d["cache"] / f"{sub}_ses-1.npz"), "mriqc_fd_mean": 0.1}
    rt = tmp / "runs.json"; json.dump({"runs": runs}, open(rt, "w")); man["runs_table_sha256"] = RN.sha(rt)
    json.dump(man, open(d["cache"] / "prepare_manifest.json", "w"))
    repo = rc_repo(tmp); (repo / "results").mkdir()
    json.dump({"checks": {"control_area_within_1pct": {"in_view_face_px_with_control": 0.8894, "faces_in_view": 2717, "no_control": 51}},
               "reported": {"median_eccentricity_view_px": {"faces_removed": 95.1, "area_matched": 77.7}, "change_energy_ratio_control_over_face": 0.8943}},
              open(repo / "results/film_stimulus_checks.json", "w"))
    gate = tmp / "gate.json"
    RN.cmd_gate(argparse.Namespace(edits=str(stim / "edits_v2.json"), regressors=str(d["rg"]), words=str(d["fc"]), runs=str(rt), out=str(gate)))
    rc = RN.cmd_maps(argparse.Namespace(intact=str(d["fc"]), removed=str(d["rem"]), matched=str(d["am"]), edits=str(stim / "edits_v2.json"),
                                        runs=str(rt), atlas=str(atlas), rc_repo=str(repo)))
    a = argparse.Namespace(forecast=str(d["fc"]), edits=str(stim / "edits_v2.json"), regressors=str(d["rg"]), runs=str(rt), atlas=str(atlas),
                           cache=str(d["cache"]), gate=str(gate), rc_repo=str(repo))
    return a, rc, w


def push(repo: Path, rel: str, date=None):
    env = {"GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date} if date else None
    git(repo, "add", rel); git(repo, "commit", "-q", "-m", rel, env=env); git(repo, "push", "-q", "origin", "HEAD:main")


def test_maps_recover_the_planted_edit_record_hashes_and_the_descriptive(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch)
    R = np.asarray(rc["R"]); assert R.shape == (400,) and np.allclose(R, w, atol=1e-4)      # intact - (intact - w) = w per parcel
    assert rc["R_sha256"] == RN.map_sha(R) and set(rc["inputs_sha256"]) == {"intact", "removed", "matched", "edits_v2"} and rc["forecasts_only"]
    dsc = rc["descriptive"]; assert 0 < dsc["grid_points_kept"] < rc["grid"]["points"] and np.allclose(dsc["R"], w, atol=1e-4)
    assert json.load(open(a.gate))["pass"] and (Path(a.rc_repo) / "results/face_map_RC.json").is_file()
    g = json.load(open(a.gate)); fb = Path(a.forecast) / "bridge"                            # gate 0 read bridge/words.csv, not export/'s copy
    assert g["inputs_sha256"]["words"] == RN.sha(fb / "words.csv") and g["words"]["n"] == 300
    kc = rc["kernel_checks"]; assert len(kc["checks"]) >= 20 and all(kc["checks"].values())   # every kernel-record check is recorded
    assert kc["gpu"] == "Tesla T4" and kc["hf_pins"] == PINS and kc["intact_words"]["exact_shift_copies"] == 0
    assert kc["cache_files_added_by_dir"]["removed"] == {"neuralset.extractors.video.HuggingFaceVideo_x1": 2160}


def test_maps_refuse_kernel_records_that_do_not_check_out(tmp_path, monkeypatch):
    """Swapped edit folders would flip B2's sign; an intact rerun between the edit kernels, another stimulus, checkpoint, environment or
    a failed stage, a non-video cache, the chunked transcript, or a forecast that is not one row per second each refuse the maps."""
    a, rc, w = build(tmp_path, monkeypatch); rem, am = tmp_path / "rem", tmp_path / "am"
    m = argparse.Namespace(intact=a.forecast, removed=str(rem), matched=str(am), edits=a.edits, runs=a.runs, atlas=a.atlas, rc_repo=a.rc_repo)
    J = {"intact": Path(a.forecast) / "bridge/tierD_bridge_forecast.json", "removed": rem / "film_rem/tierD_film_rem_forecast.json",
         "matched": am / "film_am/tierD_film_am_forecast.json"}

    def refused(k, change, match):
        orig = J[k].read_text(); d = json.loads(orig); change(d); J[k].write_text(json.dumps(d))
        with pytest.raises(SystemExit, match=match):
            RN.cmd_maps(m)
        J[k].write_text(orig)
    refused("removed", lambda d: d.update(video="area_matched_v2.mp4"), "removed_video")
    refused("intact", lambda d: d["stimulus"].update(segment="faces_removed_v2.mp4", segment_sha256=SEGS["faces_removed_v2.mp4"]), "intact_video")
    refused("matched", lambda d: d["stimulus"].update(segment_sha256="9" * 64), "matched_segment_sha256")
    refused("matched", lambda d: d.update(ckpt_matches_pin=False), "matched_ckpt_matches_pin")
    refused("intact", lambda d: d.pop("ckpt_matches_pin"), "intact_ckpt_matches_pin")
    refused("removed", lambda d: d["stages"]["predict_audio"].update(ok=False), "removed_stages_ok")
    refused("intact", lambda d: d["stages"].pop("video"), "intact_stages_ok")
    refused("matched", lambda d: d.update(intact_export_manifest_sha256="8" * 64), "matched_on_intact_export")   # an intact rerun in between
    refused("removed", lambda d: d["cache_dirs_added"].append("neuralset.extractors.audio.Wav2VecBert_x2"), "removed_video_caches_only")
    refused("matched", lambda d: d.update(cache_dirs_added=[]), "matched_video_caches_only")
    refused("removed", lambda d: d["cache_files_added_by_dir"].update({"neuralset.extractors.audio.Wav2VecBert_x2": 99999}),
            "removed_files_added_video_only")                                                    # an audio stage that missed the cache
    refused("matched", lambda d: d.update(cache_files_added_by_dir={}), "matched_files_added_video_only")
    refused("removed", lambda d: d.pop("cache_files_added_by_dir"), "removed_files_added_video_only")
    refused("matched", lambda d: d.update(gpu="Tesla P100-PCIE-16GB"), "same_gpu")
    refused("intact", lambda d: d.update(pip_freeze_sha256="5" * 64), "same_pip_freeze_sha256")
    refused("removed", lambda d: d["hf_pins"].update({"facebook/tribev2": "d" * 40}), "same_hf_pins")
    refused("matched", lambda d: d["hf_revisions"].update({"facebook/tribev2": "e" * 40}), "matched_hf_revisions_pinned")
    refused("intact", lambda d: d["words"].update(exact_shift_copies=84), "intact_words_no_shift_copies")
    refused("intact", lambda d: d["words"].update(last_start_s=377.0), "intact_words_inside_segment")
    with pytest.raises(SystemExit, match="tierD_film_rem_forecast.json"):
        RN.cmd_maps(argparse.Namespace(**{**vars(m), "removed": str(am), "matched": str(rem)}))  # the two edit folders swapped
    p = Path(a.forecast) / "bridge/pred_full.npz"; p.rename(p.with_suffix(".bak"))                 # the old kernels' unstitched output
    np.savez(p, P=np.zeros((6, 20484), np.float32), starts=np.array([0.0, 1, 2, 1, 2, 3]))
    with pytest.raises(SystemExit, match="starts"):
        RN.cmd_maps(m)
    z = np.ones((4, 20484), np.float32); z[2, 7] = np.nan; np.savez(p, P=z, starts=np.arange(4.0))
    with pytest.raises(SystemExit, match="finite"):
        RN.cmd_maps(m)
    p.with_suffix(".bak").rename(p)
    fs = [p, rem / "film_rem/pred_full.npz", am / "film_am/pred_full.npz"]                        # all three truncated to 150 s
    for f in fs:
        z = dict(np.load(f)); f.rename(f.with_suffix(".bak")); np.savez(f, P=z["P"][:150], starts=z["starts"][:150])
    with pytest.raises(SystemExit, match="scored window"):
        RN.cmd_maps(m)
    for f in fs:
        f.with_suffix(".bak").replace(f)
    assert RN.cmd_maps(m)["R_sha256"] == rc["R_sha256"]                                            # restored: the same maps again


def test_controlled_grid_needs_the_whole_4s_clip():
    ok = np.ones(4560, bool); ok[24 * 50] = False                                         # one uncontrolled frame at 50.0 s
    keep = FM.controlled_grid(ok, 1.3); s = E.common_grid(1.3) - E.TRIBE_OFFSET
    assert not keep[(s >= 50.0) & (s < 54.0)].any() and keep[(s < 49.9) | (s >= 54.1)].all()
    assert list(FM.frames_fully_controlled([[{"px_in_view": 5, "control_px": 0}], [{"px_in_view": 0, "control_px": 0}], []])) == [False, True, True]


def test_score_face_path_scored_once_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(FM.spin_perms, "__defaults__", (100, FM.SPIN_SEED))
    a, rc, w = build(tmp_path, monkeypatch); repo = Path(a.rc_repo)
    with pytest.raises(SystemExit, match="not on origin/main"):
        RN.cmd_score(a)                                                                      # R and C are not pushed yet
    push(repo, "results/face_map_RC.json", date="2026-11-01T17:59:00+00:00")
    rcp = repo / "results/face_map_RC.json"; orig = rcp.read_text(); dd = json.loads(orig); dd["R"][0] += 1.0; rcp.write_text(json.dumps(dd))
    with pytest.raises(SystemExit, match="differs"):
        RN.cmd_score(a)
    rcp.write_text(orig)
    res = RN.cmd_score(a)
    assert res["outcome"] == "PASS" and res["viewers"] == 8 and res["rc"]["R_sha256"] == rc["R_sha256"] and '"M"' not in json.dumps(res)
    assert res["descriptive"]["label"].startswith("descriptive") and "rho_R_Mbar" in res["descriptive"]
    assert res["stimulus_checks"]["no_control"] == 51 and res["stimulus_checks"]["change_energy_ratio_control_over_face"] == 0.8943
    push(repo, "results/face_map.json")                                                       # the scored result on origin/main
    for f in (RN.cmd_score, lambda x: RN.cmd_not_run(argparse.Namespace(reason="late", rc_repo=x.rc_repo))):
        with pytest.raises(SystemExit, match="closed"):
            f(a)                                                                             # scored once: no second score, no NOT RUN after it


def test_cache_change_refused(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch); push(Path(a.rc_repo), "results/face_map_RC.json", date="2026-10-31T10:00:00+00:00")
    f = Path(a.cache) / "sub-T01_ses-1.npz"; z = dict(np.load(f)); z["keep"][0] = False; np.savez(f, **z)
    with pytest.raises(SystemExit, match="cache"):
        RN.cmd_score(a)


def test_score_refuses_words_or_intact_forecast_other_than_gate_and_maps_used(tmp_path, monkeypatch):
    monkeypatch.setattr(FM.spin_perms, "__defaults__", (100, FM.SPIN_SEED))
    a, rc, w = build(tmp_path, monkeypatch); push(Path(a.rc_repo), "results/face_map_RC.json", date="2026-10-31T10:00:00+00:00")
    fb = Path(a.forecast) / "bridge"; wc = fb / "words.csv"; orig = wc.read_text()
    wc.write_text(orig + "188.50,0.3,w\n")                                                   # a transcript other than gate 0's
    with pytest.raises(SystemExit, match="words.csv differs"):
        RN.cmd_score(a)
    wc.write_text(orig); p = fb / "pred_full.npz"; z = dict(np.load(p)); np.savez(p, P=z["P"] + 1, starts=z["starts"])   # an intact rerun after maps
    with pytest.raises(SystemExit, match="pred_full.npz differs"):
        RN.cmd_score(a)
    assert not (Path(a.rc_repo) / "results/face_map.json").exists()


def test_score_refuses_a_constant_cleaned_row_and_drops_no_viewer(tmp_path, monkeypatch):
    """One constant parcel row from one viewer would give a NaN M_i (partial_fz divides by its zero SD) and gate 1 would fail for good."""
    monkeypatch.setattr(FM.spin_perms, "__defaults__", (100, FM.SPIN_SEED))
    a, rc, w = build(tmp_path, monkeypatch); push(Path(a.rc_repo), "results/face_map_RC.json", date="2026-10-31T10:00:00+00:00")
    cache = Path(a.cache); f = cache / "sub-T03_ses-1.npz"; z = dict(np.load(f)); z["raw_ALL"][7] = 0.0; np.savez(f, **z)
    man = json.load(open(cache / "prepare_manifest.json")); man["runs"]["sub-T03_ses-1"]["cache_sha256"] = RN.sha(f)
    json.dump(man, open(cache / "prepare_manifest.json", "w"))
    with pytest.raises(SystemExit, match="1 of 8 viewers.* 1 of 400 parcels"):
        RN.cmd_score(a)
    assert not (Path(a.rc_repo) / "results/face_map.json").exists() and not (cache / "face_maps_per_viewer.npz").exists()


def test_maps_after_the_cutoff_score_refuses_and_points_to_not_run(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch)
    push(Path(a.rc_repo), "results/face_map_RC.json", date="2026-11-01T18:00:01+00:00")
    with pytest.raises(SystemExit, match="not-run"):
        RN.cmd_score(a)


def test_not_run_closes_the_face_test_and_opens_the_read_guard(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch); repo = Path(a.rc_repo)
    (repo / "results/face_map_RC.json").unlink()                                            # e.g. gate 0 failed: no edit forecasts, no maps
    assert not RN.ER.read_guard(str(tmp_path / "no_gate.json"), a.rc_repo)["ok"]
    with pytest.raises(SystemExit, match="reason"):
        RN.cmd_not_run(argparse.Namespace(reason="  ", rc_repo=a.rc_repo))
    RN.cmd_not_run(argparse.Namespace(reason="gate 0: VIF(face) above 5", rc_repo=a.rc_repo))
    assert not RN.ER.read_guard(str(tmp_path / "no_gate.json"), a.rc_repo)["ok"]            # written but not yet on origin/main
    push(repo, "results/face_map.json")
    g = RN.ER.read_guard(str(tmp_path / "no_gate.json"), a.rc_repo); assert g["ok"] and g["mode"] == "not_run"
    m = argparse.Namespace(intact=a.forecast, removed=a.forecast, matched=a.forecast, edits=a.edits, runs=a.runs, atlas=a.atlas, rc_repo=a.rc_repo)
    for f in (RN.cmd_maps, RN.cmd_score, RN.cmd_not_run):
        with pytest.raises(SystemExit, match="closed"):
            f(m if f is RN.cmd_maps else a if f is RN.cmd_score else argparse.Namespace(reason="again", rc_repo=a.rc_repo))


def test_prepare_refuses_before_opening_any_file(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch)
    p = argparse.Namespace(data=str(tmp_path / "no_such_data"), meta=str(tmp_path / "no_such_meta"), runs=a.runs, atlas=a.atlas,
                           cache=str(tmp_path / "pc"), gate=a.gate, rc_repo=a.rc_repo)
    with pytest.raises(SystemExit, match="no BOLD file is opened"):
        RN.ER.cmd_prepare(p)                                                                 # neither the maps nor a NOT RUN on origin/main
    assert not (tmp_path / "pc").exists()


def test_gate_refuses_edits_not_matching_the_stimuli_manifest(tmp_path, monkeypatch):
    a, rc, w = build(tmp_path, monkeypatch)
    g = argparse.Namespace(edits=a.edits, regressors=a.regressors, words=a.forecast, runs=a.runs, out=str(tmp_path / "g2.json"))
    cp = tmp_path / "copied"; cp.mkdir(); s = np.linspace(1.0, 60.0, 40)                    # the chunked speech-to-text's +120 s copies
    (cp / "words.csv").write_text("start,duration,text\n" + "".join(f"{x:.3f},0.3,u{i}\n" for i, x in enumerate(s))
                                  + "".join(f"{x + 120:.3f},0.3,u{i}\n" for i, x in enumerate(s[:20])))
    with pytest.raises(SystemExit, match="copies"):
        RN.cmd_gate(argparse.Namespace(**{**vars(g), "words": str(cp)}))
    monkeypatch.setattr(RN, "FILM_STIM_MANIFEST", "0" * 64)
    with pytest.raises(SystemExit, match="MANIFEST"):
        RN.cmd_gate(g)                                                                       # not the committed stimuli MANIFEST
    monkeypatch.setattr(RN, "FILM_STIM_MANIFEST", RN.sha(Path(a.edits).parent / "MANIFEST.sha256"))
    d = json.load(open(a.edits)); d["per_frame"][0]["faces"] = []; json.dump(d, open(a.edits, "w"))
    with pytest.raises(SystemExit, match="differs"):
        RN.cmd_gate(g)                                                                       # edits_v2.json changed after the MANIFEST
