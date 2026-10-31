"""The Emo-FilM scoring runner on synthetic data only (no Emo-FilM file, no forecast): motion rows, the motion parser, prepare with a
stand-in projector, and score end to end on planted signal, including its refusals."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nilearn")
nib = pytest.importorskip("nibabel")
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("efr", ROOT / "scripts/emofilm_score_runner.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
E = R.E


def test_motion_rows_concatenated_single_and_ambiguous():
    runs = [("a_bold", 710), ("tos_bold", 607), ("c_bold", 945)]
    assert R.motion_rows(2262, runs, "tos_bold") == slice(710, 1317)          # the session's runs concatenated
    assert R.motion_rows(607, runs, "tos_bold") == slice(0, 607)             # the file is the Tears of Steel run alone
    assert R.motion_rows(946, runs, "tos_bold") is None                      # neither: the rule's fallback
    assert R.motion_rows(607, runs + [("d_bold", 607)], "tos_bold") is None  # another run has the same length: ambiguous
    assert R.motion_rows(2262, runs, "missing_bold") is None


def test_load_motion_and_session_order(tmp_path):
    p = tmp_path / "m.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write("rx ry rz tx ty tz\n" + "".join("0.01  -0.02  0.0  1.0  0.5  0.2  \n" for _ in range(4)))
    assert R.load_motion(p).shape == (4, 6)
    bad = tmp_path / "b.tsv"; bad.write_text("1 2 3\n")
    with pytest.raises(ValueError):
        R.load_motion(bad)
    s = tmp_path / "scans.tsv"
    s.write_text("filename\tacq_time\nfunc/x_task-B_bold.nii.gz\t2000-01-01T10:00:00\nfmap/x_phasediff.nii.gz\t2000-01-01T08:00:00\n"
                 "func/x_task-A_bold.nii.gz\t2000-01-01T09:00:00\n")
    assert R.session_runs(s, {"x_task-A_bold": {"size_t": "10"}, "x_task-B_bold": {"size_t": "12.0"}}) == [("x_task-A_bold", 10), ("x_task-B_bold", 12)]


V, SETS = 60, {"AUD": [np.arange(0, 10), np.arange(10, 20)], "LANG": [np.arange(20, 30), np.arange(30, 40)],
               "VIS": [np.arange(40, 50), np.arange(50, 60)]}


def test_prepare_run_counts_motion_and_parcels(tmp_path):
    n = 40; img = nib.Nifti1Image(np.zeros((2, 2, 2, n), np.float32), np.eye(4)); ip = tmp_path / "i.nii.gz"; nib.save(img, ip)
    mp = tmp_path / "m.tsv"; mp.write_text("".join("0 0 0 0 0 0\n" for _ in range(n - 1)) + "0 0 0 3.0 0 0\n")   # one jump at the end
    row = {"subject": "sub-X", "session": "ses-1", "n_vols": n}
    runs = [("sub-X_ses-1_task-TearsOfSteel_bold", n)]
    proj = lambda im: np.random.default_rng(0).standard_normal((20484, n))
    sets = {k: [v + 0 for v in vs] for k, vs in SETS.items()}
    out = R.prepare_run(ip, mp, row, runs, 0.1, sets, proj)
    assert out["status"] == "prepared" and out["fd"] is not None and out["keep"].sum() == n - 1 and out["raw"]["VIS"].shape == (2, n)
    out = R.prepare_run(ip, mp, {**row, "n_vols": n + 1}, runs, 0.1, sets, proj)
    assert out["status"].startswith("dropped")
    out = R.prepare_run(ip, mp, row, [("sub-X_ses-1_task-TearsOfSteel_bold", n + 5)], 0.1, sets, proj)
    assert out["fd"] is None and out["keep"].all() and out["fd_source"].startswith("fallback")
    def proj_bad(im):
        x = np.random.default_rng(0).standard_normal((20484, n)); x[10:20] = 7.0; x[45] = np.nan; x[60:70] = 7.0   # AUD, VIS, face-map-only
        return x
    sets_all = {**sets, "ALL": [*sets["AUD"], *sets["VIS"], np.arange(60, 70)]}
    out = R.prepare_run(ip, mp, row, runs, 0.1, sets_all, proj_bad)
    assert out["status"] == "prepared" and out["degenerate"] == {"AUD": 1, "LANG": 0, "VIS": 1}       # ALL only when asked
    assert R.prepare_run(ip, mp, row, runs, 0.1, sets_all, proj_bad, R.NETS + ("ALL",))["degenerate"]["ALL"] == 3


def test_prepare_refuses_every_degenerate_viewer_and_checks_all_parcels_on_the_face_path_only(tmp_path, monkeypatch):
    """A constant or NaN parcel row refuses prepare (nothing dropped), naming every such viewer; a parcel only the face map reads refuses
    on the face path, not on the not_run path, where the intact film is scored alone."""
    n = 40; data, meta, cache = tmp_path / "data", tmp_path / "meta", tmp_path / "cache"; data.mkdir(); runs = []
    mriqc = "bids_name\tsize_t\tfd_mean\n"
    for s in ("sub-A", "sub-B", "sub-C"):
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, n), np.float32), np.eye(4)), data / f"{s}_ses-1_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz")
        with gzip.open(data / f"{s}_ses-1_task-mc_space-orig_desc-motion_bold.tsv.gz", "wt") as f:
            f.write("0 0 0 0 0 0\n" * n)
        (meta / s / "ses-1").mkdir(parents=True)
        (meta / s / "ses-1" / f"{s}_ses-1_scans.tsv").write_text(f"filename\tacq_time\nfunc/{s}_ses-1_task-TearsOfSteel_bold.nii.gz\t2000-01-01T10:00:00\n")
        mriqc += f"{s}_ses-1_task-TearsOfSteel_bold\t{n}\t0.1\n"; runs.append({"subject": s, "session": "ses-1", "keep": True, "n_vols": n})
    (meta / "mriqc_group_bold.tsv").write_text(mriqc); rt = tmp_path / "runs.json"; json.dump({"runs": runs}, open(rt, "w"))
    bad = {"sub-A": [(slice(60, 70), 7.0)], "sub-B": [(slice(10, 20), 7.0), (55, np.nan)]}    # A: a face-map-only parcel; B: AUD and VIS

    def project(img, meshes):
        x = np.random.default_rng(1).standard_normal((20484, n))
        for sl, v in bad.get(Path(img.get_filename()).name[:5], []):
            x[sl] = v
        return x
    monkeypatch.setattr(R, "parcel_index", lambda d: {**SETS, "ALL": [*SETS["AUD"], *SETS["LANG"], *SETS["VIS"], np.arange(60, 70)]})
    monkeypatch.setattr(R, "mid_meshes", lambda: None); monkeypatch.setattr(E, "project", project)
    a = argparse.Namespace(data=str(data), meta=str(meta), runs=str(rt), atlas="unused", cache=str(cache), gate="unused", rc_repo="unused")
    for mode, match in (("face", "2 of 3 viewers.*sub-A_ses-1: ALL 1 of 7.*sub-B_ses-1: AUD 1 of 2, VIS 1 of 2, ALL 2 of 7"),
                        ("not_run", "1 of 3 viewers.*sub-B_ses-1: AUD 1 of 2, VIS 1 of 2")):
        monkeypatch.setattr(R, "read_guard", lambda g, r, mode=mode: {"ok": True, "mode": mode, "why": None})
        with pytest.raises(SystemExit, match=match) as e:
            R.cmd_prepare(a)
        assert not (cache / "prepare_manifest.json").exists() and (mode == "face" or "sub-A" not in str(e.value))
    del bad["sub-B"]; R.cmd_prepare(a)                                       # not_run: sub-A's face-map-only parcel does not stop it
    assert [v["status"] for v in json.load(open(cache / "prepare_manifest.json"))["runs"].values()] == ["prepared"] * 3


def build(tmp, signal=3.0, n_view=6, n_vols=260, tr=1.3):
    """A synthetic cache, forecast, regressors and run table: each viewer's BOLD = the forecast at its film times (+ noise)."""
    rng = np.random.default_rng(1); t = np.arange(190.0)
    shared = [np.convolve(rng.standard_normal(190), np.ones(6) / 6, "same") for _ in range(6)]                 # one smooth series per parcel
    base = np.stack([shared[v // 10] + 0.05 * rng.standard_normal(190) for v in range(V)], 1)                      # (190, V)
    fdir, cache, rg = tmp / "fc", tmp / "cache", tmp / "rg"
    for d in (fdir, cache, rg):
        d.mkdir()
    np.savez(fdir / "pred_full.npz", P=base, starts=t); np.savez(fdir / "pred_audio.npz", P=base * 0.9 + 0.1 * rng.standard_normal(base.shape), starts=t)
    with open(fdir / "words.csv", "w") as f:
        f.write("start,duration,text\n" + "".join(f"{s:.2f},0.3,w\n" for s in np.sort(rng.uniform(0, 189, 300))))
    json.dump({"stimulus": {"sha256": R.TOS_SHA}, "ckpt_matches_pin": True, "stages": {"video": {"ok": True}}}, open(fdir / "tierD_bridge_forecast.json", "w"))
    np.savez(rg / "film_regressors.npz", envelope=rng.random(190), brightness=rng.random(190), motion=rng.random(190))
    runs, man = [], {"label": "exploratory", "runs": {}}
    for i in range(n_view):
        sub = f"sub-T{i:02d}"; onset = 94.0 + 0.1 * i
        runs.append({"subject": sub, "session": "ses-1", "keep": True, "tr": tr, "film_onset_s": onset, "slice_offset_s": 0.6, "n_vols": n_vols})
        ft = np.arange(n_vols) * tr + 0.6 - onset; ts = np.clip(ft - E.TRIBE_OFFSET, 0, 189)
        sig = E.tribe_at(base, t, ts)                                                                          # (V, n_vols)
        raw = {k: np.stack([sig[v].mean(0) * signal + rng.standard_normal(n_vols) for v in SETS[k]]) for k in R.NETS}
        np.savez(cache / f"{sub}_ses-1.npz", **{f"raw_{k}": v for k, v in raw.items()}, keep=np.ones(n_vols, bool), fd=np.zeros(0))
        man["runs"][f"{sub}_ses-1"] = {"status": "prepared", "cache": f"{sub}_ses-1.npz", "cache_sha256": R.sha(cache / f"{sub}_ses-1.npz"),
                                       "fd_source": "fallback", "mriqc_fd_mean": 0.1 if i else 0.9}                    # viewer 0 moves too much
    rt = tmp / "runs.json"; json.dump({"runs": runs}, open(rt, "w"))
    man["runs_table_sha256"] = R.sha(rt); json.dump(man, open(cache / "prepare_manifest.json", "w"))
    return argparse.Namespace(forecast=str(fdir), regressors=str(rg), runs=str(rt), atlas="unused", cache=str(cache), results=str(tmp / "res.json"))


def test_score_end_to_end_planted_signal_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "parcel_index", lambda d: SETS)
    monkeypatch.setattr(E, "ISC_KS", (1, 2)); monkeypatch.setattr(E, "ISC_REPS", 2); monkeypatch.setattr(E, "NBOOT", 200)
    monkeypatch.setattr(E.isc_curve, "__defaults__", ((1, 2), 2, E.SEED)); monkeypatch.setattr(E.group, "__defaults__", (200, E.SEED))
    a = build(tmp_path)
    R.cmd_score(a); res = json.load(open(a.results))
    assert res["viewers"] == {"prepared": 6, "scored": 5, "excluded": {"sub-T00_ses-1": "MRIQC fd_mean only (motion rows not locatable)"}}
    for g in ("HEARING", "LANGUAGE", "VISUAL"):
        assert res["gauges"][g]["n"] == 5 and res["gauges"][g]["median_z"] > 2 and res["gauges"][g]["ci_excludes_0"]
    assert set(res["baselines"]["LANG"]) == set(E.BASELINES) and res["label"] == "exploratory"
    assert "z" not in json.dumps(res["viewers"])                                    # per-viewer values stay in the cache file
    cache = Path(a.cache); f = cache / "sub-T01_ses-1.npz"; z = dict(np.load(f)); z["keep"][0] = False; np.savez(f, **z)
    with pytest.raises(SystemExit):
        R.cmd_score(a)                                                              # a cache file changed after prepare
    (Path(a.forecast) / "words.csv").unlink()
    with pytest.raises(SystemExit):
        R.cmd_score(a)                                                              # no forecast, no score


def fast(monkeypatch):
    monkeypatch.setattr(R, "parcel_index", lambda d: SETS)
    monkeypatch.setattr(E.isc_curve, "__defaults__", ((1, 2), 2, E.SEED)); monkeypatch.setattr(E.group, "__defaults__", (200, E.SEED))


def test_score_refuses_an_unstitched_or_nonfinite_forecast(tmp_path, monkeypatch):
    """The old kernels saved TRIBE's overlapping predict windows end to end (starts out of order); tribe_at would read them as one series."""
    fast(monkeypatch); a = build(tmp_path); fd = Path(a.forecast); good = {f: dict(np.load(fd / f)) for f in ("pred_full.npz", "pred_audio.npz")}
    z = good["pred_full.npz"]; np.savez(fd / "pred_full.npz", P=np.r_[z["P"][:100], z["P"][90:]], starts=np.r_[np.arange(100.0), np.arange(90.0, 190.0)])
    with pytest.raises(SystemExit, match="refused: pred_full.npz.*starts"):
        R.cmd_score(a)
    np.savez(fd / "pred_full.npz", **good["pred_full.npz"]); z = good["pred_audio.npz"]; P = z["P"].copy(); P[50, 3] = np.nan
    np.savez(fd / "pred_audio.npz", P=P, starts=z["starts"])
    with pytest.raises(SystemExit, match="refused: pred_audio.npz.*finite"):
        R.cmd_score(a)
    np.savez(fd / "pred_audio.npz", **good["pred_audio.npz"]); z = good["pred_full.npz"]
    np.savez(fd / "pred_full.npz", P=z["P"][:150], starts=z["starts"][:150])                  # truncated: film_volumes would drop volumes
    with pytest.raises(SystemExit, match="refused: pred_full.npz.*scored window"):
        R.cmd_score(a)
    assert not Path(a.results).exists()


def test_score_refuses_a_words_table_with_the_chunked_copies(tmp_path, monkeypatch):
    fast(monkeypatch); a = build(tmp_path); s = np.sort(np.random.default_rng(4).uniform(0, 110, 120))
    rows = [(x, f"t{i % 40}") for i, x in enumerate(s)]; rows += [(x + 120.0, t) for x, t in rows if x < 70]    # the +120 s block
    (Path(a.forecast) / "words.csv").write_text("start,duration,text\n" + "".join(f"{x:.3f},0.3,{t}\n" for x, t in rows))
    with pytest.raises(SystemExit, match="refused: .*copies of words 120.000 s earlier"):
        R.cmd_score(a)
    (Path(a.forecast) / "words.csv").write_text("start,duration,text\n0.5,0.3,a\n377.2,0.3,b\n")
    with pytest.raises(SystemExit, match="at or past"):
        R.cmd_score(a)
    assert not Path(a.results).exists()


def test_score_without_signal_detects_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "parcel_index", lambda d: SETS)
    monkeypatch.setattr(E.isc_curve, "__defaults__", ((1, 2), 2, E.SEED)); monkeypatch.setattr(E.group, "__defaults__", (200, E.SEED))
    a = build(tmp_path, signal=0.0)
    R.cmd_score(a); res = json.load(open(a.results))
    assert all(abs(res["gauges"][g]["median_z"]) < 2 for g in ("HEARING", "LANGUAGE", "VISUAL"))
