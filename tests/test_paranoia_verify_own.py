"""--verify-own on scripts/r2_paranoia.py and scripts/r5_short_scan.py (a judge's own Paranoia forecasts; dated notes of Nov 2 in both
rules), on synthetic data only: no Paranoia brain data is read. The judge's forecasts, word tables and surprisal skip our pins only under
the flag; the stimuli and every other input must still equal ours; nothing is written into results/; the judge's numbers print beside
ours. Our own scoring path keeps refusing unpinned inputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nilearn")
pytest.importorskip("scipy")
REPO = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R2 = load("r2_paranoia", "scripts/r2_paranoia.py")
R5 = load("r5_short_scan", "scripts/r5_short_scan.py")
R5.R2 = R2                                                                         # one R2 module for both (its VERIFY state is shared)
OURS_R2 = json.loads((REPO / "results/r2_paranoia.json").read_text())
OURS_R5 = json.loads((REPO / "results/r5_short_scan.json").read_text())


def results_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO / "results").glob("r[25]_*.json")}


def made_up_root(tmp_path):
    """A root and a judge's folder holding made-up stimuli, word tables, surprisal and the nine stock forecasts (none equal our pins)."""
    root, own = tmp_path / "root", tmp_path / "own"
    (root / "r2/stimuli").mkdir(parents=True); (root / "r2/forecast/out").mkdir(parents=True); own.mkdir()
    for d in (root / "r2/forecast/out", own):
        for s in R2.STORIES:
            (d / f"words_{s}.tsv").write_text(f"text\tstart\tduration\nw\t{len(s)}\t0.2\n")
            for k in R2.MODES:
                (d / f"pred_stock_{s}_{k}.npz").write_bytes(f"{d.name} {s} {k}".encode())
        (d / "surprisal.json").write_text("{}")
    (root / "r2/surprisal.json").write_text("{}")
    for s in R2.STORIES:
        (root / f"r2/stimuli/{s}_audio.wav").write_bytes(s.encode())
    return root, own


# ---- results/ is never written ------------------------------------------------------------------------------------------------------
def test_verify_own_refuses_an_out_inside_the_repository(tmp_path):
    before = results_hashes()
    root, own = made_up_root(tmp_path)
    for out in (REPO / "results", REPO / "results/judge", REPO):
        with pytest.raises(SystemExit, match="inside the repository"):
            R2.score(str(root), str(out), str(own))
        with pytest.raises(SystemExit, match="inside the repository"):
            R5.score(str(root), str(out), str(own))
    link = tmp_path / "looks_outside"; link.symlink_to(REPO / "results")             # a path outside that resolves into results/
    with pytest.raises(SystemExit, match="inside the repository"):
        R2.refuse_results(str(link))
    assert results_hashes() == before and not (REPO / "results/judge").exists()


def test_the_judges_files_go_to_out_only_and_print_beside_ours(tmp_path, capsys):
    before = results_hashes()
    theirs = copy.deepcopy(OURS_R2); theirs["models"]["stock"]["volunteer_equivalent_language"]["point_k"] = 2.5
    R2.READ.clear(); R2.READ[str(tmp_path / "x.func.gii")] = "ab"
    R2._write_own(theirs, {**theirs, "full": {}}, str(tmp_path))
    t5 = copy.deepcopy(OURS_R5); t5["tribe"]["k_above_0"] = 15
    R5._write_own(t5, {**t5, "per_listener": {}}, str(tmp_path))
    assert {p.name for p in tmp_path.iterdir()} == {"r2_paranoia.json", "r2_paranoia_public.json", "SHA256.json", "r5_short_scan.json",
                                                   "r5_short_scan_public.json"}
    assert results_hashes() == before
    out = capsys.readouterr().out
    assert OURS_R2["lines"][0] in out and OURS_R5["line"] in out and "Your forecasts beside ours" in out


# ---- our pins: skipped only under --verify-own --------------------------------------------------------------------------------------
def test_default_path_still_refuses_unpinned_inputs(tmp_path):
    root, _ = made_up_root(tmp_path)
    with pytest.raises(SystemExit, match="inputs differ from the pins"):
        R2.check_pins(str(root))
    with pytest.raises(SystemExit, match="inputs differ from R2's pins"):
        R5.check_inputs(str(root))
    with pytest.raises(SystemExit, match="inputs differ from the pins"):
        R2.score(str(root), str(tmp_path / "out"))                                  # no flag: our pins, as before


def test_pins_are_skipped_only_under_verify_own(tmp_path, monkeypatch):
    root, own = made_up_root(tmp_path)
    monkeypatch.setattr(R2, "STIM_PINS", {s: R2.sha(root / f"r2/stimuli/{s}_audio.wav") for s in R2.STORIES})
    ours = tmp_path / "ours_read.json"; ours.write_text('{"task-story1_events.tsv": "e1"}')
    monkeypatch.setitem(R2.OURS, "read", str(ours))
    got = R2.own_inputs(str(root), str(own))
    assert got["pred stock_story2_shuf"] == R2.sha(own / "pred_stock_story2_shuf.npz") != R2.FORECAST_PINS["stock_story2_shuf"]
    assert got["words story1"] == R2.sha(own / "words_story1.tsv") and got["surprisal"] == R2.sha(own / "surprisal.json")
    assert R2.VERIFY == {"task-story1_events.tsv": "e1"}                             # note() now checks every other input against ours
    got5 = R5.own_inputs(str(root), str(own))
    assert got5["pred stock_story3_audio"] == R2.sha(own / "pred_stock_story3_audio.npz") and not any("tuned" in k for k in got5)
    with pytest.raises(SystemExit, match="inputs differ from the pins"):              # the same files without the flag
        R2.check_pins(str(root))
    (own / "pred_stock_story1_full.npz").unlink()
    with pytest.raises(SystemExit, match=r"lacks \['pred stock_story1_full'\]"):
        R2.own_inputs(str(root), str(own))
    (root / "r2/stimuli/story2_audio.wav").write_bytes(b"another recording")
    for f in (R2.own_inputs, R5.own_inputs):
        with pytest.raises(SystemExit, match=r"stimuli differ from ds001338's recordings.*stimulus story2"):
            f(str(root), str(own))
    R2.VERIFY.clear()


def test_under_verify_own_every_other_input_must_equal_ours(tmp_path):
    g = tmp_path / "sub-x_task-story1_hemi-L_space-fsaverage5_bold.func.gii"; g.write_bytes(b"gifti")
    p = tmp_path / "pred_stock_story1_full.npz"; p.write_bytes(b"the judge's forecast")
    assert R2.note(str(g)) == R2.sha(g)                                              # our own scoring: no such check (VERIFY empty)
    R2.VERIFY.update({g.name: R2.sha(g)})
    assert R2.note(str(g)) == R2.sha(g) and R2.note(str(p)) == R2.sha(p)             # equal to ours; the judge's own file is not compared
    g.write_bytes(b"another scan")
    with pytest.raises(SystemExit, match="differs from the file our scoring run read"):
        R2.note(str(g))
    R2.VERIFY.clear()


# ---- the printout -------------------------------------------------------------------------------------------------------------------
def test_side_by_side_prints_the_judges_numbers_beside_ours():
    theirs = copy.deepcopy(OURS_R2)
    ve = theirs["models"]["stock"]["volunteer_equivalent_language"]; ve.update(point_k=2.04, lower_k=0.91, claim_k=0)
    theirs["models"]["stock"]["reference"]["share"] = 0.512
    theirs["models"]["stock"]["cheap_baselines"]["AUD"]["tribe_vs"]["envelope"]["verdict"] = "not beaten"
    rows = {r[:30].strip(): r[30:].split() for r in R2.side_by_side(theirs, OURS_R2)[1:]}
    assert rows["volunteer-equivalent (point)"] == ["2.0", "3.0"] and rows["lower bound"] == ["0.91", "1.88"]
    assert rows["at least"] == ["0", "1"] and rows["share"] == ["51%", "59%"] and rows["baselines beaten"] == ["3", "of", "4", "4", "of", "4"]
    assert R2.side_by_side({"lines": []}, OURS_R2)[1].split()[-2:] == ["scored", "3.0"]   # a coverage-only run says so
    t5 = copy.deepcopy(OURS_R5); t5["tribe"].update(k_above_0=13, wilcoxon_p=0.081, **{"pass": False})
    r5 = {r[:26].strip(): r[26:] for r in R5.side_by_side(t5, OURS_R5)[1:]}
    assert r5["TRIBE: listeners above 0"].split() == ["13", "of", "22", "17", "of", "22"]
    assert r5["verdict"].split() == ["FAIL", "PASS"] and r5["TRIBE: one-sided p"].split() == ["0.081", "0.0004"]


# ---- the whole scoring step under --verify-own on a synthetic root ------------------------------------------------------------------
def test_r2_score_under_verify_own_end_to_end(tmp_path, monkeypatch, capsys):
    """score() from files to the side-by-side printout on made-up data, the judge's forecasts in their own folder and our forecast pins
    untouched (they would refuse these files): the GIFTI reader and the atlas are replaced as in tests/test_r2_paranoia.py."""
    sf = pytest.importorskip("soundfile")
    before = results_hashes()
    rng = np.random.default_rng(12)
    subs = [f"sub-{i:02d}" for i in range(12)]
    dur = {"story1": 70, "story2": 60, "story3": 50}
    sets = {"AUD": [np.arange(0, 3), np.arange(3, 6)], "LANG": [np.arange(6, 9), np.arange(9, 12), np.arange(12, 15)]}
    medial = np.zeros(20484, bool); medial[10000:10242] = True
    R, own = tmp_path / "r2", tmp_path / "judge"
    for d in ("stimuli", "bold"):
        (R / d).mkdir(parents=True)
    own.mkdir()
    surp, sig = {}, {}
    for story, s_ in dur.items():
        sf.write(R / f"stimuli/{story}_audio.wav", 0.1 * rng.standard_normal(s_ * 8000).astype("float32"), 8000)
        on = np.sort(rng.uniform(0.2, s_ - 1, 3 * s_))
        (own / f"words_{story}.tsv").write_text("text\tstart\tduration\tsequence_id\tsentence\n" + "".join(
            f"w{rng.integers(0, 30)}\t{t:.3f}\t0.2\t0\t\n" for t in on))
        surp[story] = {"onsets": on.tolist(), "surprisal_bits": rng.uniform(1, 9, len(on)).tolist()}
        (R / f"bold/task-{story}_events.tsv").write_text(f"onset\tduration\tstim_file\n2\t{s_}\t{story}_audio.wav\n")
        sig[story] = np.convolve(rng.standard_normal(s_ + 20), np.ones(5) / 5, "same")[:s_]
        for k in R2.MODES:
            P = sig[story][:, None] * (1 + 0.1 * rng.standard_normal((1, 15))) + 0.4 * rng.standard_normal((s_, 15))
            np.savez_compressed(own / f"pred_stock_{story}_{k}.npz", P=P.astype("float32"), starts=np.arange(s_))
    (own / "surprisal.json").write_text(json.dumps(surp))

    def fake_run(bold_dir, s, story):
        n = dur[story] + 2
        v = rng.standard_normal((20484, n)).astype(np.float32)
        al = R2.pred_parcels(np.tile(sig[story][:, None], (1, 15)), np.arange(dur[story]), n, {"x": [np.arange(15)]})["x"][0]
        v[:15] += np.nan_to_num(al)[None]
        return v
    atl = tmp_path / "atlas"; atl.mkdir()
    for h in ("lh", "rh"):
        (atl / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot").write_bytes(h.encode())
    ours_read = {p.name: R2.sha(p) for p in [*(R / "bold").glob("*.tsv"), *(R / "stimuli").glob("*.wav"), *atl.iterdir()]}
    (tmp_path / "ours_read.json").write_text(json.dumps(ours_read))
    monkeypatch.setitem(R2.OURS, "read", str(tmp_path / "ours_read.json"))
    monkeypatch.setattr(R2, "SUBS", tuple(subs)); monkeypatch.setattr(R2, "load_run", fake_run)
    monkeypatch.setattr(R2.H, "parcel_sets", lambda a, b: sets); monkeypatch.setattr(R2, "medial_wall", lambda a, b: medial)
    monkeypatch.setattr(R2, "M1_REPS", 3)
    monkeypatch.setattr(R2, "STIM_PINS", {s: R2.sha(R / f"stimuli/{s}_audio.wav") for s in R2.STORIES})
    out = tmp_path / "out"
    pub = R2.score(str(tmp_path), str(out), str(own))
    assert len(pub["lines"]) == 4 and "12 listeners, the whole story: 126 volumes = 2.1 minutes at TR 1 s" in pub["lines"][0]
    assert pub["inputs_sha256"]["pred stock_story1_full"] == R2.sha(own / "pred_stock_story1_full.npz")
    assert json.loads((out / "r2_paranoia_public.json").read_text())["lines"] == pub["lines"] and (out / "SHA256.json").exists()
    printed = capsys.readouterr().out
    assert "Your forecasts beside ours (results/r2_paranoia.json):" in printed and "1.88" in printed
    assert results_hashes() == before
    (atl / "lh.Schaefer2018_400Parcels_17Networks_order.annot").write_bytes(b"another atlas")
    with pytest.raises(SystemExit, match="lh.Schaefer2018_400Parcels_17Networks_order.annot differs from the file our scoring run read"):
        R2.score(str(tmp_path), str(out), str(own))
    R2.VERIFY.clear()
