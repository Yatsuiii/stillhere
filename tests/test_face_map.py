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


def build(tmp: Path, n_view=8, n_vols=260, tr=1.3):
    """Synthetic edits, regressors, words, intact forecast, film versions, run table and prepare cache with planted face tracking."""
    rng = np.random.default_rng(9); (tmp / "atlas").mkdir(); atlas = write_annots(tmp / "atlas")
    parcels = FM.all_parcels(*(atlas / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    d = {k: tmp / k for k in ("fc", "vers", "rg", "cache")}
    for p in d.values():
        p.mkdir()
    faces = np.repeat(np.clip(np.convolve(rng.random(190), np.ones(5) / 5, "same") - 0.4, 0, None), 24)    # per frame, 190 s
    json.dump({"window_release_s": [0.0, 190.0], "source_sha256": RN.ER.TOS_SHA, "fps": 24,
               "per_frame": [{"masked_px": [int(f * 1280 * 534)]} for f in faces]}, open(tmp / "edits.json", "w"))
    np.savez(d["rg"] / "film_regressors.npz", envelope=rng.random(190), brightness=rng.random(190), motion=rng.random(190))
    t = np.arange(190.0); base = rng.standard_normal((190, 20484)).astype(np.float32)
    with open(d["fc"] / "words.csv", "w") as f:
        f.write("start,duration,text\n" + "".join(f"{s:.2f},0.3,w\n" for s in np.sort(rng.uniform(0, 189, 300))))
    np.savez(d["fc"] / "pred_full.npz", P=base, starts=t); np.savez(d["fc"] / "pred_audio.npz", P=base, starts=t)
    json.dump({"stimulus": {"sha256": RN.ER.TOS_SHA}, "ckpt_matches_pin": True, "stages": {"video": {"ok": True}}}, open(d["fc"] / "tierD_bridge_forecast.json", "w"))
    w = rng.standard_normal(400); off = np.zeros(20484, np.float32)
    for p, v in enumerate(parcels["vertices"]):
        off[v] = w[p]
    for v, P in (("intact", base), ("faces_removed", base - off), ("area_matched", base - rng.standard_normal(20484).astype(np.float32))):
        (d["vers"] / v).mkdir(); np.savez(d["vers"] / v / "pred_full.npz", P=P, starts=t)
    face_s = FM.face_per_s([[int(f * 1280 * 534)] for f in faces]); runs, man = [], {"runs": {}}
    for i in range(n_view):
        sub = f"sub-T{i:02d}"; onset = 94.0 + 0.1 * i; ft = np.arange(n_vols) * tr + 0.6 - onset
        F = H.hrf_regressor(face_s, ft); F = (F - F.mean()) / (F.std() + 1e-9)
        raw = np.outer(w, F) + rng.standard_normal((400, n_vols))
        np.savez(d["cache"] / f"{sub}_ses-1.npz", raw_ALL=raw, keep=np.ones(n_vols, bool), fd=np.zeros(0))
        runs.append({"subject": sub, "session": "ses-1", "keep": True, "tr": tr, "film_onset_s": onset, "slice_offset_s": 0.6, "n_vols": n_vols})
        man["runs"][f"{sub}_ses-1"] = {"status": "prepared", "cache": f"{sub}_ses-1.npz", "cache_sha256": RN.sha(d["cache"] / f"{sub}_ses-1.npz"), "mriqc_fd_mean": 0.1}
    rt = tmp / "runs.json"; json.dump({"runs": runs}, open(rt, "w")); man["runs_table_sha256"] = RN.sha(rt)
    json.dump(man, open(d["cache"] / "prepare_manifest.json", "w"))
    gate = tmp / "gate.json"
    RN.cmd_gate(argparse.Namespace(edits=str(tmp / "edits.json"), regressors=str(d["rg"]), words=str(d["fc"] / "words.csv"), runs=str(rt), out=str(gate)))
    return argparse.Namespace(versions=str(d["vers"]), forecast=str(d["fc"]), edits=str(tmp / "edits.json"), regressors=str(d["rg"]), runs=str(rt),
                              atlas=str(atlas), cache=str(d["cache"]), gate=str(gate), results=str(tmp / "res.json"), now="2026-10-30T12:00:00+00:00")


def test_runner_planted_pass_not_run_paths_and_cache_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(FM, "N_SPIN", 100); monkeypatch.setattr(FM.spin_perms, "__defaults__", (100, FM.SPIN_SEED))
    a = build(tmp_path)
    assert json.load(open(a.gate))["pass"]
    res = RN.cmd_score(a)
    assert res["outcome"] == "PASS" and res["viewers"] == 8 and res["parcels_tested"] == 374 and "M" not in json.dumps(res)
    assert RN.cmd_score(argparse.Namespace(**{**vars(a), "now": "2026-10-30T20:00:01+00:00"}))["outcome"] == "NOT RUN"
    g = json.load(open(a.gate)); g["pass"] = False; json.dump(g, open(a.gate, "w"))
    assert RN.cmd_score(a)["outcome"] == "NOT RUN"; g["pass"] = True; json.dump(g, open(a.gate, "w"))
    assert RN.cmd_score(argparse.Namespace(**{**vars(a), "versions": str(tmp_path / "none")}))["outcome"] == "NOT RUN"
    f = Path(a.cache) / "sub-T01_ses-1.npz"; z = dict(np.load(f)); z["keep"][0] = False; np.savez(f, **z)
    with pytest.raises(SystemExit):
        RN.cmd_score(a)
