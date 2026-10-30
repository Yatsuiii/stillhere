"""scripts/alice_heldout.py on synthetic data only (docs/rules/alice_heldout_rule.md section 8): no Alice BOLD is read by any test.
Raw-volume indexing, the per-listener onset shift, the coverage rule and its exclusions, the outcome lines, and that the group step
formed one listener at a time reproduces stillhere/heldout.py and stillhere/cheap_baselines.py exactly when every onset is equal."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nilearn")
pytest.importorskip("scipy")
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("alice_heldout", REPO / "scripts/alice_heldout.py")
AH = importlib.util.module_from_spec(spec); spec.loader.exec_module(AH)
H, CB = AH.H, AH.CB

SUBS = ["a", "b", "c", "d", "e", "f"]
N_VOLS, N_SEC = 200, 460


def synthetic(seed=0, onset=0.0):
    """Six listeners, 3 hearing and 4 language parcels, a shared stimulus-locked signal plus noise; forecasts that carry it."""
    rng = np.random.default_rng(seed)
    starts = np.arange(400, dtype=float)
    sig = np.convolve(rng.standard_normal(420), np.ones(6) / 6, "same")[:400]
    P = {k: {"P": (sig[:, None] * (1 + 0.1 * rng.standard_normal((1, 20))) + 0.5 * rng.standard_normal((400, 20))).astype("float32"),
             "starts": starts} for k in ("full", "audio", "shuf")}
    sets = {"AUD": [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)], "LANG": [np.arange(9, 12), np.arange(12, 15), np.arange(15, 18), np.arange(18, 20)]}
    pp = {k: H.pred_parcels(d["P"], d["starts"] + onset, N_VOLS, sets) for k, d in P.items()}
    bold = {s: {net: pp["full"][net] + 0.8 * rng.standard_normal(pp["full"][net].shape) for net in ("AUD", "LANG")} for s in SUBS}
    for s in SUBS:
        for net in bold[s]:
            bold[s][net][:, :10] = np.nan_to_num(bold[s][net][:, :10])
            bold[s][net] = np.nan_to_num(bold[s][net])
    wps = rng.poisson(2.5, N_SEC).astype(float)
    env = np.abs(rng.standard_normal(N_SEC))
    base = {k: rng.standard_normal(N_SEC) for k in ("B_freq", "B_bu", "B_td", "surprisal")}
    return P, sets, pp, bold, wps, env, base


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            close(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
    elif isinstance(a, (float, int, np.floating)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def test_derivative_volume_i_sits_at_raw_volume_i_plus_10():
    r = AH.to_raw({"LANG": np.arange(12, dtype=float).reshape(2, 6)})
    assert r["LANG"].shape == (2, 16) and np.isnan(r["LANG"][:, :10]).all()
    assert np.array_equal(r["LANG"][:, 10:], np.arange(12, dtype=float).reshape(2, 6))
    assert AH.SKIP == H.J0 == 10                                           # the first kept volume is section 8's skip point


def test_each_listener_is_shifted_by_its_own_onset():
    P, sets, *_ = synthetic()
    o = 0.8
    a = H.pred_parcels(P["full"]["P"], P["full"]["starts"] + o, N_VOLS, sets)["LANG"]
    al = H.align(P["full"]["P"], P["full"]["starts"], N_VOLS)               # volume j reads stimulus time 2j - 5 - o
    t = np.arange(N_VOLS) * H.TR - 5.0 - o
    ok = (t >= 0) & (t <= 399)
    ref = np.array([np.interp(t[ok], P["full"]["starts"], P["full"]["P"][:, v]) for v in range(20)])
    assert np.allclose(a[0][ok], ref[sets["LANG"][0]].mean(0), atol=1e-5) and al.shape == (N_VOLS, 20)
    x = np.random.default_rng(1).standard_normal(N_SEC); tj = H.TR * np.arange(N_VOLS); L = -1.25
    assert np.allclose(H.hrf_regressor(x, tj - (L + o)), H.hrf_regressor(x, tj - L - o))   # the lag carries the onset


def test_coverage_shares_eligibility_and_kept_parcels():
    rng = np.random.default_rng(2)
    sets = {"AUD": [np.arange(0, 10)], "LANG": [np.arange(10, 20), np.arange(20, 30)]}
    full = rng.standard_normal((30, 50))
    part = full.copy(); part[20:24] = 0.0                                  # 4 of LANG's 20 vertices out of view: 80%
    nan = full.copy(); nan[10:20] = np.nan; nan[20:30] = 0.0               # language area absent
    covs = {s: AH.covered(v) for s, v in (("x", full), ("y", part), ("z", nan))}
    sh = {s: AH.area_shares(c, sets) for s, c in covs.items()}
    assert sh["x"] == {"AUD": 1.0, "LANG": 1.0} and sh["y"]["LANG"] == 0.8 and sh["z"]["LANG"] == 0.0
    elig = {"LANG": [s for s in sh if sh[s]["LANG"] >= AH.COVER], "AUD": [s for s in sh if sh[s]["AUD"] >= AH.COVER]}
    assert elig == {"LANG": ["x"], "AUD": ["x", "y", "z"]}                 # 80% is below the 90% floor: excluded from language
    keep = AH.kept_parcels(covs, sets, {"LANG": ["x", "y"], "AUD": ["x"]})
    assert keep == {"AUD": [0], "LANG": [0, 1]}                            # y still has 6 of parcel 1's 10 vertices
    ser = AH.parcel_series(part, covs["y"], sets, keep)
    assert np.allclose(ser["LANG"][1], part[24:30].mean(0))                # the mean over covered vertices only


def test_group_stage_formed_one_listener_at_a_time_equals_stage_scores():
    o = 0.4
    P, sets, pp, bold, wps, env, base = synthetic(3, onset=o)
    for j1 in (154, 190):
        ref = H.stage_scores(bold, pp, wps, env, base, N_VOLS, j1, -1.25 + o)
        per = {s: H.stage_scores({s: bold[s]}, pp, wps, env, base, N_VOLS, j1, -1.25 + o) for s in SUBS}
        got = AH.group_stage(per, -1.25 + o, j1, SUBS, SUBS)
        pl = got.pop("per_listener"); rp = ref.pop("per_listener"); got.pop("n_hearing")
        close(got, ref)
        assert pl["listeners"] == rp["listeners"] and all(np.allclose(pl[k], rp[k]) for k in rp if k != "listeners")


def test_own_best_lag_and_isc_curve_with_equal_onsets_equal_heldout():
    P, sets, pp, bold, wps, env, base = synthetic(4)
    aud = [bold[s]["AUD"] for s in SUBS]
    assert AH.own_best_lag_onsets(aud, [0.0] * len(SUBS), env, N_VOLS, 154) == H.own_best_lag(aud, env, N_VOLS, 154, 0.03)
    cl = lambda X: H.clean(X, 0.03, H.J0, 154)
    ctrl = {"AUD": H.zeros_control(145), "LANG": [cl(pp["audio"]["LANG"]).mean(0, keepdims=True), cl(H.wordrate_1hz(np.arange(0, 400, 0.5), N_VOLS)[None])]}
    ks = [1, 2, 3, 5]
    close(AH.isc_curve_onsets(bold, {s: ctrl for s in SUBS}, ks, 4, 11, {"AUD": SUBS, "LANG": SUBS}), H.isc_curve(bold, ctrl, ks, 4, 11))


def test_cheap_baselines_formed_one_listener_at_a_time_equal_section8():
    P, sets, pp, bold, wps, env, base = synthetic(5)
    ref = CB.section8(H, bold, pp, wps, env, N_VOLS, 154, -1.25)
    per = {s: CB.section8(H, {s: bold[s]}, pp, wps, env, N_VOLS, 154, -1.25) for s in SUBS}
    got = AH.group_cheap(per, {"LANG": SUBS, "AUD": SUBS})
    for net in ("LANG", "AUD"):
        close(got[net]["mean_z"], ref[net]["mean_z"]); close(got[net]["tribe_vs"], ref[net]["tribe_vs"])


def test_outcome_lines():
    ve = lambda point, claim: {"point_k": point, "claim_k": claim}
    lang = lambda z, p: {"z": z, "p": p}
    assert AH.outcome(9) == ["Alice Datasets: rule committed, not run (coverage): 9 listeners had the language areas in view, fewer than 10."]
    a = AH.outcome(22, ve(4.2, 2), lang(1.5, 0.001), (6, 8))
    assert a[0].endswith("the Alice Datasets, 22 listeners), the forecast was worth about 4 real volunteers, and at least 2.")
    assert a[1] == "What it had to beat: word rate and the sound envelope (beaten in 6 of 8 comparisons)."
    b = AH.outcome(22, ve(1.4, 0), lang(0.61, 0.012), (2, 8))
    assert "tracked the listeners' language areas (z 0.61, p = 0.012), but at its lower bound it was worth less than one" in b[0]
    c = AH.outcome(22, ve(0.3, 0), lang(0.12, 0.31), (0, 8))
    assert "did not reach the bar in language areas (z 0.12, p = 0.31). We show it because we said we would." in c[0]
    for line in a + b + c:
        assert "outside TRIBE v2's training and evaluation sets" in line or line.startswith("What it had to beat")
        assert not re.search(r"\b(new|generalis|generaliz)", line, re.I) and chr(0x2014) not in line


def test_lines_are_the_rules_words():
    rule = " ".join((REPO / "docs/rules/alice_heldout_rule.md").read_text().split())
    for k, line in AH.LINES.items():
        assert line in rule, k


def test_timed_words_and_tree_baselines(tmp_path):
    f = tmp_path / "ann.tsv"
    f.write_text("Words\tWord Onset\tWord Offset\tCFG BU nodeCount\tCFG TD nodeCount\n"
                 "Alice\t20.1\t20.6\t2\t4\nwas\t20.6\t20.9\t1\t3\nquite\t\t\t1\t1\nalice\t22.4\t22.9\t3\t1\n")
    w = AH.timed_words(f)
    assert [x[:2] for x in w] == [("alice", 20.1), ("was", 20.6), ("alice", 22.4)]          # the untimed row is dropped
    b = AH.baselines(w, {"onsets": [20.1, 20.6, 22.4], "surprisal_bits": [5.0, 2.0, 3.0]}, 30)
    assert np.isclose(b["B_bu"][20], (2 - 2) + (1 - 2)) and np.isclose(b["B_bu"][22], 3 - 2)   # centred, summed into the onset second
    assert np.isclose(b["B_freq"].sum(), 0.0) and set(b) == {"B_freq", "B_bu", "B_td", "surprisal"}


def test_sub53_excluded_before_scoring():
    """Rule section 2, amended Oct 30 19:40 UTC: 22 listeners scored, 11 per protocol; the reason travels into the results file."""
    assert "53" in AH.EXCLUDED and "53" not in AH.SCORED and len(AH.SCORED) == 22
    proto = [AH.PROTOCOL[s] for s in AH.SCORED]
    assert proto.count("single-echo, 44 slices") == 11 and proto.count("three-echo (echo 2), 33 slices") == 11
    rule = " ".join((REPO / "docs/rules/alice_heldout_rule.md").read_text().split())
    assert "sub-53 is excluded: its published derivative fails the gzip CRC check" in rule and "Listeners scored: 22" in rule
