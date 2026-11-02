"""scripts/r2_paranoia.py and r2/run_r2_forecast.py on synthetic data only (docs/rules/r2_paranoia_rule.md): no Paranoia brain data is
read by any test. Coverage and the medial wall, the GIFTI loader, each run cleaned on its own then joined, the timing at TR 1 s with the
story at 2 s, the leave-one-out reference share, the lines per branch, the refusals, the runner's word checks, and that with one run
every R2 function reproduces heldout.py, cheap_baselines.py and alice_heldout.py at TR 1 s exactly (to rounding: R2's gauge applies the
same projector in factored form)."""
import importlib.util
import re
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
RUN = load("run_r2_forecast", "r2/run_r2_forecast.py")
H, AH = R2.H, R2.AH
CB = AH.CB
SUBS = ["a", "b", "c", "d", "e", "f"]
N, N_SEC = 300, 330


@pytest.fixture
def h_at_tr1(monkeypatch):
    """heldout's module constants at R2's values (TR 1 s, first scored volume 20, 30-TR null minimum), for the one-run comparisons."""
    monkeypatch.setattr(H, "TR", 1.0); monkeypatch.setattr(H, "J0", 20); monkeypatch.setattr(H, "MIN", 30)


def synthetic(seed=0, n=N, shift=0.0):
    """Six listeners, 3 hearing and 4 language parcels, a shared story-locked signal plus noise; forecasts that carry it."""
    rng = np.random.default_rng(seed)
    starts = np.arange(n + 10, dtype=float)
    sig = np.convolve(rng.standard_normal(n + 30), np.ones(6) / 6, "same")[:n + 10]
    P = {k: {"P": (sig[:, None] * (1 + 0.1 * rng.standard_normal((1, 20))) + 0.5 * rng.standard_normal((n + 10, 20))).astype("float32"),
             "starts": starts} for k in ("full", "audio", "shuf")}
    sets = {"AUD": [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)], "LANG": [np.arange(9, 12), np.arange(12, 15), np.arange(15, 18), np.arange(18, 20)]}
    pp = {k: R2.pred_parcels(d["P"], d["starts"], n, sets, shift) for k, d in P.items()}
    bold = {s: {net: np.nan_to_num(pp["full"][net]) + 0.8 * rng.standard_normal(pp["full"][net].shape) for net in ("AUD", "LANG")} for s in SUBS}
    story = {"n": n, "onsets": np.sort(rng.uniform(0, n - 5, 600)), "wps": rng.poisson(2.5, N_SEC).astype(float),
             "env": np.abs(rng.standard_normal(N_SEC)), "base": {k: rng.standard_normal(N_SEC) for k in ("B_freq", "surprisal")}}
    return P, sets, pp, bold, story


def close(a, b, rtol=1e-7):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            close(a[k], b[k], rtol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, rtol)
    elif isinstance(a, (float, int, np.floating)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=rtol, atol=1e-9), (a, b)
    else:
        assert a == b, (a, b)


# ---- coverage, the medial wall, the GIFTI loader ----------------------------------------------------------------------------------------
def test_medial_wall_is_schaefers_unlabelled_vertices():
    names = [b"Background+FreeSurfer_Defined_Medial_Wall", b"17Networks_LH_VisCent_ExStr_1", b"17Networks_LH_TempPar_1"]
    lh, rh = np.array([0, 1, 2, 2, 0]), np.array([1, 0, -1, 2, 2])
    assert R2.medial_from_labels(lh, names, rh, names).tolist() == [True, False, False, False, True, False, True, True, False, False]


def test_a_vertex_is_covered_only_if_covered_in_all_three_runs_and_off_the_medial_wall():
    rng = np.random.default_rng(1)
    runs = {s: rng.standard_normal((8, 40)) for s in R2.STORIES}
    runs["story3"][2] = 5.0                                               # constant in one run
    runs["story2"][4, 7] = np.nan                                         # one missing volume in another
    medial = np.zeros(8, bool); medial[6] = True
    cov = R2.coverage(runs, medial)
    assert cov.tolist() == [True, True, False, True, False, True, False, True]
    sets = {"AUD": [np.array([0, 1])], "LANG": [np.array([2, 3, 4, 5])]}
    assert AH.area_shares(cov, sets) == {"AUD": 1.0, "LANG": 0.5}         # LANG 50%: below the 90% rule for this listener


def test_gifti_loader_joins_left_then_right_and_refuses_wrong_files(tmp_path):
    nb = pytest.importorskip("nibabel")

    def write(s, story, hemi, structure, T, offset):
        da = [nb.gifti.GiftiDataArray((np.arange(10242) + offset + 1000 * t).astype(np.float32), intent="NIFTI_INTENT_TIME_SERIES")
              for t in range(T)]
        nb.save(nb.gifti.GiftiImage(meta=nb.gifti.GiftiMetaData({"AnatomicalStructurePrimary": structure}), darrays=da),
                tmp_path / R2.gifti_name(s, story, hemi))
    write("sub-x", "story1", "L", "CortexLeft", 3, 0); write("sub-x", "story1", "R", "CortexRight", 3, 0.5)
    v = R2.load_run(tmp_path, "sub-x", "story1")
    assert v.shape == (20484, 3) and v[0, 1] == 1000.0 and v[10242, 0] == 0.5 and v[10242 + 7, 2] == 2007.5
    write("sub-y", "story1", "L", "CortexRight", 3, 0); write("sub-y", "story1", "R", "CortexLeft", 3, 0)
    with pytest.raises(SystemExit, match="CortexRight, not 10242 of CortexLeft"):
        R2.load_run(tmp_path, "sub-y", "story1")
    write("sub-z", "story1", "L", "CortexLeft", 3, 0); write("sub-z", "story1", "R", "CortexRight", 4, 0)
    with pytest.raises(SystemExit, match="differ in volume count"):
        R2.load_run(tmp_path, "sub-z", "story1")


# ---- cleaning, joining, timing ------------------------------------------------------------------------------------------------------
def test_each_run_is_cleaned_on_its_own_then_joined(h_at_tr1):
    rng = np.random.default_rng(2)
    rows = {"story1": rng.standard_normal((3, 120)).cumsum(1), "story2": rng.standard_normal((3, 90)).cumsum(1) + 50}
    W = {"story1": 110, "story2": 80}
    got = R2.cj(rows, W)
    ref = np.hstack([H.clean(rows["story1"], 0.03, 20, 110), H.clean(rows["story2"], 0.03, 20, 80)])
    assert got.shape == (3, 91 + 61) and np.array_equal(got, ref)
    assert np.allclose(got[:, :91].mean(1), 0) and np.allclose(got[:, 91:].mean(1), 0)        # each run z-scored on its own
    from nilearn import signal
    assert np.allclose(got[:, :91], signal.clean(rows["story1"][:, 20:111].T, detrend=True, high_pass=0.03, t_r=1.0,
                                                 standardize="zscore_sample").T)               # the high-pass at t_r 1 s
    joined_then_cleaned = H.clean(np.hstack([rows["story1"][:, :111], rows["story2"][:, 20:81]]), 0.03, 20, 151)
    assert not np.allclose(got[:, 20:], joined_then_cleaned)


def test_timing_volume_j_reads_story_time_j_minus_7_and_regressors_j_minus_2_minus_L():
    n = 60
    P = np.tile(np.arange(100, dtype=np.float32)[:, None], (1, 4))         # a forecast whose value is its own second
    sets = {"LANG": [np.arange(4)]}
    a = R2.pred_parcels(P, np.arange(100), n, sets)["LANG"][0]
    assert np.isnan(a[:7]).all() and np.allclose(a[7:], np.arange(7, n) - 7)    # story time j - 2 - 5
    b = R2.pred_parcels(P, np.arange(100), n, sets, shift=-1.25)["LANG"][0]
    assert np.allclose(b[7:], np.arange(7, n) - 7 + 1.25)                   # the literal section 5 reading moves it by L
    assert np.allclose(R2.story_times(5, -1.25), np.arange(5) - 2 + 1.25)
    x = np.zeros(80); x[30] = 1.0
    r0, rL = H.hrf_regressor(x, R2.story_times(80, 0.0)), H.hrf_regressor(x, R2.story_times(80, -3.0))
    assert np.argmax(rL) == np.argmax(r0) - 3                              # L < 0 reads a later story time: the peak comes 3 volumes earlier
    assert R2.H.TR == 2.0                                                  # heldout's own TR is restored after every R2 call


def test_window_ends_at_the_last_volume_every_forecast_covers():
    P, sets, *_ = synthetic(3)                                             # forecast seconds 0..309
    pp = {k: R2.pred_parcels(d["P"], d["starts"], 330, sets) for k, d in P.items()}
    assert R2.last_volume(pp, 330) == 316                                  # read at story time j - 7
    assert R2.last_volume({k: {n: A[:, :300] for n, A in v.items()} for k, v in pp.items()}, 300) == 299


def test_forecast_read_is_section8_and_refuses_when_unset(monkeypatch):
    assert R2.FORECAST_READ == "section8" and R2.fshift(-1.25) == 0.0            # the rule's dated clarification of Nov 1
    monkeypatch.setattr(R2, "FORECAST_READ", None)
    with pytest.raises(SystemExit, match="rule section 5 and section 8's practice differ"):
        R2.fshift(-1.25)
    monkeypatch.setattr(R2, "FORECAST_READ", "section8"); assert R2.fshift(-1.25) == 0.0
    monkeypatch.setattr(R2, "FORECAST_READ", "lagged"); assert R2.fshift(-1.25) == -1.25


# ---- one run: R2's functions are heldout's at TR 1 s --------------------------------------------------------------------------------
def test_gauge_equals_heldout_gauge(h_at_tr1):
    rng = np.random.default_rng(4)
    for p, T, nc in ((4, 200, 3), (7, 260, 0), (19, 150, 2)):
        B, X = rng.standard_normal((p, T)), rng.standard_normal((p, T)) + 0.3 * rng.standard_normal((1, T))
        ctl = [rng.standard_normal((1, T)) for _ in range(nc)] or H.zeros_control(T)
        assert np.allclose(R2.gauge(B, X, ctl), H.gauge(B, X, ctl), rtol=1e-9, atol=1e-12)


def test_one_run_stage_equals_heldout_stage_scores(h_at_tr1):
    P, sets, pp, bold, story = synthetic(5)
    j1, L = 290, -1.25
    W = {"story1": j1}
    ref = H.stage_scores(bold, pp, story["wps"], story["env"], story["base"], N, j1, L + R2.ONSET)     # heldout reads at tj - L
    F = R2.joined_forecasts({"story1": pp}, W)
    reg = R2.regressors({"story1": story}, W, L)
    Bj = {s: {net: R2.cj({"story1": bold[s][net]}, W) for net in ("AUD", "LANG")} for s in SUBS}
    per = {s: R2.listener_stage(s, Bj[s], F, F, reg) for s in SUBS}
    got = AH.group_stage(per, L + R2.ONSET, j1, SUBS, SUBS)
    pl, rp = got.pop("per_listener"), ref.pop("per_listener"); got.pop("n_hearing")
    close(got, ref)
    assert pl["listeners"] == rp["listeners"] and all(np.allclose(pl[k], rp[k], rtol=1e-7) for k in rp if k != "listeners")


def test_one_run_cheap_baselines_equal_section8(h_at_tr1):
    P, sets, pp, bold, story = synthetic(6)
    j1, L = 290, -1.25
    W = {"story1": j1}
    ref = CB.section8(H, bold, pp, story["wps"], story["env"], N, j1, L + R2.ONSET)
    F = R2.joined_forecasts({"story1": pp}, W)
    reg = R2.regressors({"story1": story}, W, L)
    Bj = {s: {net: R2.cj({"story1": bold[s][net]}, W) for net in ("AUD", "LANG")} for s in SUBS}
    got = AH.group_cheap({s: R2.listener_cheap(Bj[s], F, F, reg) for s in SUBS}, {"LANG": SUBS, "AUD": SUBS})
    for net in ("LANG", "AUD"):
        close(got[net]["mean_z"], ref[net]["mean_z"]); close(got[net]["tribe_vs"], ref[net]["tribe_vs"])


def test_one_run_own_best_lag_and_isc_curve_equal_alice_heldout(h_at_tr1):
    P, sets, pp, bold, story = synthetic(7)
    j1 = 290; W = {"story1": j1}
    Bj = {s: {net: R2.cj({"story1": bold[s][net]}, W) for net in ("AUD", "LANG")} for s in SUBS}
    lag, zz = R2.own_best_lag([Bj[s]["AUD"] for s in SUBS], {"story1": story}, W)
    rlag, rzz = AH.own_best_lag_onsets([bold[s]["AUD"] for s in SUBS], [R2.ONSET] * len(SUBS), story["env"], N, j1, 0.03)
    assert lag == rlag and np.allclose(zz, rzz, rtol=1e-7)
    wr1 = R2.cj({"story1": R2.wordrate_1hz(story["onsets"], N)[None]}, W)
    ctrl = {s: {"AUD": H.zeros_control(j1 - 19), "LANG": [R2.cj({"story1": pp["audio"]["LANG"]}, W).mean(0, keepdims=True), wr1]} for s in SUBS}
    assert np.allclose(wr1, H.clean(H.wordrate_1hz(story["onsets"] + R2.ONSET, N)[None], 0.03, 20, j1))   # M1's word rate at scan time
    ks, subs_by = [1, 2, 3, 5], {"AUD": SUBS, "LANG": SUBS}
    got = R2.isc_curve({s: {"story1": bold[s]} for s in SUBS}, Bj, ctrl, R2.isc_draws(subs_by, ks, 4, 11), W)
    close(got, AH.isc_curve_onsets(bold, ctrl, ks, 4, 11, subs_by, 0.03, j1))


def test_isc_draws_keep_the_generator_order_when_one_area_is_scored_alone(h_at_tr1):
    P, sets, pp, bold, story = synthetic(8)
    W = {"story1": 290}
    Bj = {s: {net: R2.cj({"story1": bold[s][net]}, W) for net in ("AUD", "LANG")} for s in SUBS}
    ctrl = {s: {"AUD": H.zeros_control(271), "LANG": H.zeros_control(271)} for s in SUBS}
    d = R2.isc_draws({"AUD": SUBS, "LANG": SUBS}, [1, 2, 5], 3, 11)
    both = R2.isc_curve({s: {"story1": bold[s]} for s in SUBS}, Bj, ctrl, d, W)
    lang = R2.isc_curve({s: {"story1": bold[s]} for s in SUBS}, Bj, ctrl, d, W, nets=("LANG",))
    assert lang["LANG"] == both["LANG"] and len(d["LANG"]["a"][5]) == 1 and len(d["LANG"]["a"][2]) == 3   # k = n - 1: one set


# ---- the reference share ---------------------------------------------------------------------------------------------------------------
def test_reference_is_the_other_listeners_average_with_languages_gauge_and_controls(h_at_tr1):
    rng = np.random.default_rng(9)
    Bj = {s: {"LANG": rng.standard_normal((4, 200))} for s in SUBS}
    ctl = [rng.standard_normal((1, 200))]
    got = R2.reference(Bj, SUBS[:4], ctl)
    assert set(got) == set(SUBS[:4])
    want = R2.gauge(Bj["b"]["LANG"], (Bj["a"]["LANG"] + Bj["c"]["LANG"] + Bj["d"]["LANG"]) / 3, ctl)[0]
    assert got["b"] == want                                                # e and f are not scored for LANGUAGE: never in the average


def test_share_is_the_models_mean_z_over_the_references_mean_z_on_the_same_listeners():
    ref = {k: v for k, v in zip("abcdefgh", (2.0, 6.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0))}
    sh = R2.share({**{k: v / 2 for k, v in ref.items()}, "z": 100.0}, ref)
    assert sh["n"] == 8 and sh["reference_mean_z"] == 4.0 and sh["model_mean_z"] == 2.0 and sh["share"] == 0.5
    assert sh["reference_wilcoxon_p"] < 0.05


def test_no_share_when_the_reference_does_not_reach_the_bar():
    """Rule section 10 (the cloud's item 8): a reference with mean z at or below 0, or one-sided Wilcoxon p at or above 0.05, gives no share;
    the line names the reference's mean z instead, in every branch."""
    neg = {k: v for k, v in zip("abcdefgh", (-1.0, 0.5, -0.2, 0.1, -0.4, 0.2, -0.3, 0.0))}
    weak = {k: v for k, v in zip("abcdefgh", (2.0, -1.9, 1.5, -1.6, 0.4, -0.3, 0.2, 0.1))}
    for ref in (neg, weak):
        sh = R2.share({k: 1.0 for k in ref}, ref)
        assert sh["share"] is None and (sh["reference_mean_z"] <= 0 or sh["reference_wilcoxon_p"] >= 0.05)
    ve = lambda point, claim: {"point_k": point, "claim_k": claim}
    a = R2.outcome("stock", 21, 1250, ve(3.6, 2), {"z": 1.9, "p": 0.0004}, None, (3, 4), zref=0.31)
    assert a[0].endswith("and at least 2. The other listeners' average did not reach the bar (z 0.31).") and "%" not in a[0]
    b = R2.outcome("tuned", 21, 1250, ve(1.4, 0), {"z": 0.61, "p": 0.012}, None, (1, 4), zref=-0.05)
    assert b[0].endswith("(about 1 at the point estimate). The other listeners' average did not reach the bar (z -0.05).")
    c = R2.outcome("stock", 21, 1250, ve(0.3, 0), {"z": 0.12, "p": 0.31}, None, (0, 4), zref=0.2)
    assert c[0].endswith("(z 0.12, p = 0.31); the other listeners' average did not reach the bar (z 0.20). We show it because we said we would.")
    assert R2.NO_SHARE in " ".join((REPO / R2.RULE).read_text().split())


# ---- lines -------------------------------------------------------------------------------------------------------------------------
def test_lines_per_branch():
    ve = lambda point, claim: {"point_k": point, "claim_k": claim}
    lang = lambda z, p: {"z": z, "p": p}
    assert R2.outcome("stock", 9) == ["Paranoia: rule committed, not run (coverage): 9 listeners had the language areas in view, fewer than 10."]
    a = R2.outcome("stock", 21, 1250, ve(3.6, 2), lang(1.9, 0.0004), 0.574, (3, 4))
    assert a[0] == ("On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; 21 listeners, the "
                    "whole story: 1,250 volumes = 20.8 minutes at TR 1 s), stock TRIBE v2's forecast was worth about 4 real volunteers, and at "
                    "least 2. It reached 57% of what the other listeners' average reaches.")
    assert a[1] == "What it had to beat: word rate and the sound envelope (beaten in 3 of 4 comparisons)."
    b = R2.outcome("tuned", 21, 1250, ve(1.4, 0), lang(0.61, 0.012), 0.236, (1, 4))
    assert b[0].startswith("Fine-tuned TRIBE v2: On another public story,") and "fine-tuned TRIBE v2's forecast tracked the listeners' language " \
        "areas (z 0.61, p = 0.012), but at its lower bound it was worth less than one real volunteer (about 1 at the point estimate). It reached " \
        "24% of what" in b[0]
    assert b[1] == "Fine-tuned TRIBE v2: What it had to beat: word rate and the sound envelope (beaten in 1 of 4 comparisons)."
    c = R2.outcome("stock", 21, 1250, ve(0.3, 0), lang(0.12, 0.31), 0.05, (0, 4))
    assert c[0].endswith("did not reach the bar in language areas (z 0.12, p = 0.31); it reached 5% of what the other listeners' average reaches. "
                         "We show it because we said we would.")
    for line in a + b + c + [R2.LINES["pair"]]:
        assert chr(0x2014) not in line and "ceiling" not in line.lower() and not re.search(r"\b(new|generalis|generaliz)", line, re.I)


def test_lines_are_the_rules_words():
    rule = " ".join((REPO / R2.RULE).read_text().split())
    for k, line in R2.LINES.items():
        assert line in rule, k


# ---- refusals ----------------------------------------------------------------------------------------------------------------------
def unpinned(monkeypatch, mod):
    """The state before the pins commit (354cab7 filled them)."""
    for k, v in (("STIM_PINS", {}), ("WORD_PINS", {}), ("SURPRISAL_PIN", ""), ("FORECAST_PINS", {}), ("RECORD_PINS", {}), ("WHISPERX", "")):
        monkeypatch.setattr(mod, k, v)


def test_download_and_scoring_refuse_before_the_pins(tmp_path, monkeypatch):
    unpinned(monkeypatch, R2)
    with pytest.raises(SystemExit, match="not pinned yet"):
        R2.download(str(tmp_path))
    monkeypatch.setattr(R2, "FORECAST_READ", "section8")
    with pytest.raises(SystemExit, match="not pinned yet"):
        R2.score(str(tmp_path), str(tmp_path / "out"))
    assert not (tmp_path / "r2/bold").exists()


def test_pins_must_cover_the_nine_stock_forecasts(tmp_path, monkeypatch):
    d = tmp_path / "r2"; (d / "stimuli").mkdir(parents=True); (d / "forecast/out").mkdir(parents=True)
    for s in R2.STORIES:
        (d / f"stimuli/{s}_audio.wav").write_bytes(s.encode()); (d / f"forecast/out/words_{s}.tsv").write_bytes(b"w" + s.encode())
    (d / "surprisal.json").write_bytes(b"{}")
    (d / "forecast/out/pred_stock_story1_full.npz").write_bytes(b"p")
    monkeypatch.setattr(R2, "STIM_PINS", {s: R2.sha(d / f"stimuli/{s}_audio.wav") for s in R2.STORIES})
    monkeypatch.setattr(R2, "WORD_PINS", {s: R2.sha(d / f"forecast/out/words_{s}.tsv") for s in R2.STORIES})
    monkeypatch.setattr(R2, "SURPRISAL_PIN", R2.sha(d / "surprisal.json"))
    monkeypatch.setattr(R2, "FORECAST_PINS", {"stock_story1_full": R2.sha(d / "forecast/out/pred_stock_story1_full.npz")})
    (d / "r2_asr.json").write_text('{"whisperx": ["whisperx-3.8.6.dist-info"]}')
    monkeypatch.setattr(R2, "RECORD_PINS", {"r2/r2_asr.json": R2.sha(d / "r2_asr.json")}); monkeypatch.setattr(R2, "WHISPERX", "3.8.6")
    with pytest.raises(SystemExit, match="pred stock_story3_shuf not pinned") as e:
        R2.check_pins(str(tmp_path))
    assert "tuned" not in str(e.value)                                            # rule section 11: the fine-tuned line is not run
    monkeypatch.setattr(R2, "WHISPERX", "3.9.0")
    with pytest.raises(SystemExit, match="whisperx"):
        R2.check_pins(str(tmp_path))
    monkeypatch.setattr(R2, "WHISPERX", "3.8.6"); (d / "r2_asr.json").write_text('{"whisperx": []}')
    with pytest.raises(SystemExit, match="record r2/r2_asr.json"):
        R2.check_pins(str(tmp_path))


def test_download_refuses_a_gifti_whose_size_differs_from_the_data_check(tmp_path, monkeypatch):
    sub = R2.SUBS[0]
    monkeypatch.setattr(R2, "check_pins", lambda root: {}); monkeypatch.setattr(R2, "SUBS", (sub,))
    monkeypatch.setattr(R2.netfetch, "get", lambda url, dst: open(dst, "wb").write(b"not the file"))
    with pytest.raises(SystemExit, match=f"refused: {sub}_task-story1_hemi-L_space-fsaverage5_bold.func.gii is 12 bytes; r2/data_check.json"):
        R2.download(str(tmp_path))


def test_a_gifti_must_match_the_downloads_hashes(tmp_path):
    g = tmp_path / "x.func.gii"; g.write_bytes(b"gifti")
    (tmp_path / "SHA256.json").write_text('{"x.func.gii": "%s"}' % R2.sha(g))
    assert R2.note(str(g)) == R2.sha(g)
    g.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from the download's SHA256.json"):
        R2.note(str(g))


def test_events_onset_must_be_two_seconds(tmp_path):
    f = tmp_path / "e.tsv"
    f.write_text("onset\tduration\tstim_file\n2\t524\tstory1_audio.wav\n")
    assert R2.story_onset(f, "story1") == 2.0
    f.write_text("onset\tduration\tstim_file\n0\t524\tstory1_audio.wav\n")
    with pytest.raises(SystemExit, match="onset 0"):
        R2.story_onset(f, "story1")


# ---- words and baselines -----------------------------------------------------------------------------------------------------------
def test_words_as_the_baselines_read_them(tmp_path):
    assert [R2.norm_word(w) for w in ("Hello,", "\u201cWell", "don't.", "--", "  IT ")] == ["hello", "well", "don't", "--", "it"]
    f = tmp_path / "w.tsv"
    f.write_text('text\tstart\tduration\tsequence_id\tsentence\n The\t0.5\t0.2\t0\t"x, y"\nthe,\t1.4\t0.2\t0\t"x, y"\ncat.\t2.2\t0.3\t1\tz\n')
    w = R2.word_table(f)
    assert w == [("the", 0.5), ("the", 1.4), ("cat", 2.2)]
    b = R2.baselines(w, {"onsets": [0.5, 1.4, 2.2], "surprisal_bits": [2.0, 1.0, 6.0]}, 5)
    assert np.isclose(b["B_freq"][0], np.log10(2 / 3) - np.mean([np.log10(2 / 3)] * 2 + [np.log10(1 / 3)]))   # frequency within the table
    assert np.isclose(b["surprisal"][2], 3.0) and set(b) == {"B_freq", "surprisal"}


# ---- the forecast runner's helpers (no GPU, no TRIBE) -------------------------------------------------------------------------------
def test_runner_stage_order():
    assert len(RUN.STAGES) == 18 and RUN.STAGES[:3] == ["stock_story1_audio", "stock_story1_full", "stock_story1_shuf"]
    assert RUN.missing_before("stock_story2_shuf", {"stock_story2_audio"}) == ["stock_story2_full"]
    assert len(RUN.missing_before("tuned_story1_audio", set(RUN.STAGES[:8]))) == 1           # every stock stage first
    assert RUN.missing_before("tuned_story3_full", set(RUN.STAGES[:9]) | {"tuned_story3_audio"}) == []
    with pytest.raises(SystemExit):
        RUN.parse("stock_story4_full")


def test_runner_flags_words_repeated_at_one_exact_shift():
    rng = np.random.default_rng(10)
    texts = [f"w{i}" for i in rng.integers(0, 40, 200)]; starts = np.sort(rng.uniform(0, 120, 200)).round(3).tolist()
    assert RUN.shift_copies(texts, starts)["exact_shift_copies"] == 0
    late = [(t, s) for t, s in zip(texts, starts) if s < 60]
    dup = texts + [t for t, _ in late], starts + [s + 60.0 for _, s in late]                 # the over-60 s duplication
    assert RUN.shift_copies(*dup)["exact_shift_copies"] >= len(late) and RUN.shift_copies(*dup)["worst_shift_s"] == 60.0


def test_runner_fulltext_counts_per_part_without_the_header(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("-" * 40 + "\n\nCitation header, four words.\n\n" + "-" * 40 + "\n\nOne two three.\n\nFour five\n\n---END PART---\n\nsix seven\n"
                 "\n---END PART---\n\n\u201ceight\u201d nine \u2014 ten\n")
    assert RUN.fulltext_counts(f) == {"story1": 5, "story2": 2, "story3": 3}


def test_runner_shuffled_table_keeps_timings_and_permutes_texts(tmp_path):
    head = "text\tstart\tduration\tsequence_id\tsentence\n"
    rows = [("a", 0.1, 0.2), ("b", 0.5, 0.2), ("c", 0.9, 0.3)]
    (tmp_path / "t.tsv").write_text(head + "".join(f"{t}\t{s}\t{d}\t0\t\n" for t, s, d in rows))
    (tmp_path / "s.tsv").write_text(head + "".join(f"{t}\t{s}\t{d}\t0\t\n" for t, (_, s, d) in zip("cab", rows)))
    assert RUN.check_shuffled(tmp_path / "s.tsv", tmp_path / "t.tsv") == {"same_onsets_and_durations": True, "texts_permuted": True, "texts_moved": 3}
    (tmp_path / "x.tsv").write_text(head + "".join(f"{t}\t{s + 1}\t{d}\t0\t\n" for t, s, d in rows))
    with pytest.raises(AssertionError):
        RUN.check_shuffled(tmp_path / "x.tsv", tmp_path / "t.tsv")


# ---- the whole scoring step on a synthetic root -------------------------------------------------------------------------------------
@pytest.mark.parametrize("read", ["section8", "lagged"])
def test_score_end_to_end_on_a_synthetic_root(tmp_path, monkeypatch, read):
    """score() from files to lines on made-up data: 12 listeners (one without language coverage in one run), three runs of 72, 62 and
    52 volumes, forecasts that carry the listeners' shared signal. The GIFTI reader and the atlas are replaced (tested above)."""
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(12)
    subs = [f"sub-{i:02d}" for i in range(12)]
    dur = {"story1": 70, "story2": 60, "story3": 50}
    sets = {"AUD": [np.arange(0, 3), np.arange(3, 6)], "LANG": [np.arange(6, 9), np.arange(9, 12), np.arange(12, 15)]}
    medial = np.zeros(20484, bool); medial[10000:10242] = True
    R = tmp_path / "r2"
    for d in ("stimuli", "forecast/out", "bold"):
        (R / d).mkdir(parents=True)
    surp, sig = {}, {}
    for story, s_ in dur.items():
        sf.write(R / f"stimuli/{story}_audio.wav", 0.1 * rng.standard_normal(s_ * 8000).astype("float32"), 8000)
        on = np.sort(rng.uniform(0.2, s_ - 1, 3 * s_))
        (R / f"forecast/out/words_{story}.tsv").write_text("text\tstart\tduration\tsequence_id\tsentence\n" + "".join(
            f"w{rng.integers(0, 30)}\t{t:.3f}\t0.2\t0\t\n" for t in on))
        surp[story] = {"onsets": on.tolist(), "surprisal_bits": rng.uniform(1, 9, len(on)).tolist()}
        (R / f"bold/task-{story}_events.tsv").write_text(f"onset\tduration\tstim_file\n2\t{s_}\t{story}_audio.wav\n")
        sig[story] = np.convolve(rng.standard_normal(s_ + 20), np.ones(5) / 5, "same")[:s_]
        for m in R2.SCORED_MODELS:
            for k in R2.MODES:
                P = sig[story][:, None] * (1 + 0.1 * rng.standard_normal((1, 15))) + 0.4 * rng.standard_normal((s_, 15))
                np.savez_compressed(R / f"forecast/out/pred_{m}_{story}_{k}.npz", P=P.astype("float32"), starts=np.arange(s_))
    (R / "surprisal.json").write_text(__import__("json").dumps(surp))

    def fake_run(bold_dir, s, story):
        n = dur[story] + 2
        v = rng.standard_normal((20484, n)).astype(np.float32)
        al = R2.pred_parcels(np.tile(sig[story][:, None], (1, 15)), np.arange(dur[story]), n, {"x": [np.arange(15)]})["x"][0]
        v[:15] += np.nan_to_num(al)[None]
        if s == "sub-03" and story == "story2":
            v[6:15] = np.nan                                               # language area out of view in one run: not scored for LANGUAGE
        return v
    monkeypatch.setattr(R2, "SUBS", tuple(subs)); monkeypatch.setattr(R2, "load_run", fake_run)
    monkeypatch.setattr(R2.H, "parcel_sets", lambda a, b: sets); monkeypatch.setattr(R2, "medial_wall", lambda a, b: medial)
    monkeypatch.setattr(R2, "M1_REPS", 3); monkeypatch.setattr(R2, "FORECAST_READ", read)
    monkeypatch.setattr(R2, "STIM_PINS", {s: R2.sha(R / f"stimuli/{s}_audio.wav") for s in R2.STORIES})
    monkeypatch.setattr(R2, "WORD_PINS", {s: R2.sha(R / f"forecast/out/words_{s}.tsv") for s in R2.STORIES})
    monkeypatch.setattr(R2, "SURPRISAL_PIN", R2.sha(R / "surprisal.json"))
    monkeypatch.setattr(R2, "FORECAST_PINS", {p.stem[5:]: R2.sha(p) for p in (R / "forecast/out").glob("pred_*.npz")})
    (R / "r2_asr.json").write_text('{"whisperx": ["whisperx-3.8.6.dist-info"]}')
    monkeypatch.setattr(R2, "RECORD_PINS", {"r2/r2_asr.json": R2.sha(R / "r2_asr.json")}); monkeypatch.setattr(R2, "WHISPERX", "3.8.6")
    atl = tmp_path / "atlas"; atl.mkdir()
    for h in ("lh", "rh"):
        (atl / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot").write_bytes(h.encode())
    written = {}
    monkeypatch.setattr(R2, "_write", lambda pub, full, out: written.update(pub=pub, full=full) or pub)
    pub = R2.score(str(tmp_path), str(tmp_path / "out"))
    assert pub["listeners"]["sub-03"]["language"] is False and pub["listeners"]["sub-03"]["hearing"] is True
    assert pub["window"] == {"volumes": {"story1": [20, 71], "story2": [20, 61], "story3": [20, 51]}, "T": 52 + 42 + 32, "minutes": 2.1}
    assert len(pub["lines"]) == 4 and pub["lines"][2] == R2.LINES["tuned_not_run"] == (
        "Fine-tuned TRIBE v2: rule committed, not run (the stock runs' Llama features were not kept).")
    assert pub["lines"][-1] == R2.LINES["pair"] and "11 listeners, the whole story: 126 volumes = 2.1 minutes at TR 1 s" in pub["lines"][0]
    assert set(pub["models"]) == {"stock"}
    for m in R2.SCORED_MODELS:
        r = pub["models"][m]["reference"]
        assert r["n"] == 11 and np.isclose(r["share"], r["model_mean_z"] / r["reference_mean_z"])
        assert pub["models"][m]["stage"]["primary"]["n"] == 11 and pub["models"][m]["stage"]["primary"]["n_hearing"] == 12
        assert set(pub["models"][m]["cheap_baselines"]) == {"LANG", "AUD"} and "per_listener" not in pub["models"][m]["stage"]["primary"]
    read = {Path(k).name for k in R2.READ}                                          # every input read is hashed for the results
    assert {"words_story1.tsv", "surprisal.json", "task-story3_events.tsv", "pred_stock_story2_shuf.npz", "story1_audio.wav"} <= read
    __import__("json").dumps(written["full"], default=float)
