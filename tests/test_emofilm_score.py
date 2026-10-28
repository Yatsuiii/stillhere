"""Emo-FilM scorer on synthetic data only (no Emo-FilM file is read): the opening-window volume mapping, prediction sample times,
alignment sign and onset, a planted forecast recovered and a null not, censoring and the D7 run rule, FD, motion-row mapping, bootstrap
determinism, the ISC(k) curve with TRIBE placed on it, and the four baselines scored and compared with TRIBE. Needs nilearn (skipped
without it)."""
import numpy as np
import pytest

pytest.importorskip("nilearn")
from stillhere import emofilm_score as E  # noqa: E402

TR, ONSET, SLICE, NPRED, NVOL = 1.3, 93.887, 0.61625, 190, 607          # the kernel forecasts release 0 to 190 s at 1 Hz
STARTS = np.arange(NPRED, dtype=float)
rng0 = np.random.default_rng(7)


def smooth(n, v, rng):
    x = rng.standard_normal((n, v)); k = np.ones(8) / 8
    return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, x)


P_AUD, P_FULL = smooth(NPRED, 20, rng0), smooth(NPRED, 20, rng0)
SETS = {"AUD": [np.arange(0, 10), np.arange(10, 20)], "LANG": [np.arange(0, 10), np.arange(10, 20)]}


def viewer(rng, planted=True, onset=ONSET, gain=1.0):
    """Parcel rows on the scored volumes: the prediction read at the true sample times (planted) plus noise."""
    j, ft, ts = E.film_volumes(NVOL, TR, onset, SLICE, STARTS)
    sig = E.parcel_means(E.tribe_at(P_AUD, STARTS, ts), SETS)["AUD"] if planted else np.zeros((2, len(j)))
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-12)     # gain = signal-to-noise ratio
    return j, ft, ts, gain * sig + rng.standard_normal((2, len(j)))


def test_scored_volumes_map_film_time_and_tribe_offset():
    j, ft, ts = E.film_volumes(NVOL, TR, ONSET, SLICE, STARTS)
    assert ft[0] >= E.DROP_S and (j[0] - 1) * TR + SLICE - ONSET < E.DROP_S
    assert np.allclose(ts, ft - E.TRIBE_OFFSET) and ts.min() >= E.OPENING_S[0] and ts.max() <= E.OPENING_S[1]
    assert (j[-1] + 1) * TR + SLICE - ONSET - E.TRIBE_OFFSET > E.OPENING_S[1]          # the next volume's sample is past 180 s
    assert np.allclose(j * TR + SLICE - ONSET, ft) and np.all(np.diff(j) == 1)


def test_opening_window_volume_counts_over_the_onset_range():
    counts = {len(E.film_volumes(NVOL, TR, on, SLICE, STARTS)[0]) for on in np.linspace(93.5, 94.7, 25)}
    assert counts <= {134, 135}                                                    # 175 s of film time at TR 1.3 s
    short = E.film_volumes(NVOL, TR, ONSET, SLICE, np.arange(60.0))[2]
    assert short.max() <= 59                                                       # never read past the prediction
    assert E.OPENING_S[0] <= E.BRIDGE_S[0] and E.BRIDGE_S[1] <= E.OPENING_S[1]       # the forecast covers the scene used for SAM


def test_tribe_at_reads_the_prediction_at_its_sample_times():
    P = np.arange(10, dtype=float)[:, None] * np.ones((1, 3))
    assert np.allclose(E.tribe_at(P, np.arange(10.0) + 0.5, [2.5, 3.0, 9.5]), [[2, 2.5, 9]] * 3)
    assert np.allclose(E.tribe_at(P, np.arange(10.0), [0.0, 4.25]), [[0, 4.25]] * 3)


def test_alignment_sign_and_onset_matter():
    rng = np.random.default_rng(1); j, ft, ts, B = viewer(rng)
    keep = np.ones(len(j), bool); X = lambda t: E.parcel_means(E.tribe_at(P_AUD, STARTS, t), SETS)["AUD"]
    z_true = E.gauge_null(E.clean(B, TR, keep), E.clean(X(ts), TR, keep), [np.zeros((1, len(j)))])["z"]
    z_late = E.gauge_null(E.clean(B, TR, keep), E.clean(X(ts + 8.0), TR, keep), [np.zeros((1, len(j)))])["z"]
    z_sign = E.gauge_null(E.clean(B, TR, keep), E.clean(X(ts - 8.0), TR, keep), [np.zeros((1, len(j)))])["z"]
    assert z_true > 3 and z_true > z_late + 2 and z_true > z_sign + 2


def test_planted_group_recovered_and_null_not():
    zs = {}
    for planted in (True, False):
        rng = np.random.default_rng(11 if planted else 12); out = []
        for _ in range(12):
            j, ft, ts, B = viewer(rng, planted=planted, gain=0.6)
            keep = np.ones(len(j), bool); X = E.parcel_means(E.tribe_at(P_AUD, STARTS, ts), SETS)["AUD"]
            out.append(E.gauge_null(E.clean(B, TR, keep), E.clean(X, TR, keep), [np.zeros((1, len(j)))])["z"])
        zs[planted] = E.group(out, nboot=2000)
    assert zs[True]["ci_excludes_0"] and zs[True]["median_z"] > 2 and zs[True]["label"] == "exploratory"
    assert not zs[False]["ci_excludes_0"]


def test_censoring_fd_and_run_rule():
    p = np.zeros((10, 6)); p[5, 3] = 1.0; p[6, 0] = 0.01                  # 1 mm translation step, then 0.01 rad = 0.5 mm
    fd = E.fd_power(p)
    assert fd[0] == 0 and np.isclose(fd[5], 1.0) and np.isclose(fd[6], 1.0 + 0.5) and np.isclose(fd[7], 0.5)
    assert E.run_pass(np.r_[np.full(80, 0.1), np.full(20, 0.9)]) and not E.run_pass(np.r_[np.full(60, 0.1), np.full(40, 0.9)])
    A = rng0.standard_normal((3, 200)); keep = np.ones(200, bool); keep[50:60] = False
    assert E.clean(A, TR, keep).shape == (3, 190)
    assert E.run_rows([100, 607, 500], 1) == slice(100, 707)


def test_bootstrap_is_deterministic():
    z = np.random.default_rng(3).standard_normal(30)
    assert E.group(z, nboot=1000) == E.group(z, nboot=1000)


def test_viewer_scores_keys_and_visual_not_run():
    rng = np.random.default_rng(5); j, ft, ts, B = viewer(rng); keep = np.ones(len(j), bool)
    pa = E.parcel_means(E.tribe_at(P_AUD, STARTS, ts), SETS); pf = E.parcel_means(E.tribe_at(P_FULL, STARTS, ts), SETS)
    out = E.viewer_scores({"AUD": B, "LANG": B}, {"audio": pa, "full": pf}, rng.standard_normal(len(j)), keep, TR, visual=False)
    assert set(out) == {"HEARING", "LANGUAGE", "VISUAL"} and out["VISUAL"] == "not run" and out["HEARING"]["z"] > 2


def test_isc_curve_rises_with_k_and_tribe_is_placed():
    rng = np.random.default_rng(9); grid = E.common_grid(TR); shared = smooth(len(grid), 2, rng).T
    assert grid[0] == E.DROP_S and grid[-1] <= E.OPENING_S[1] + E.TRIBE_OFFSET and len(grid) == 135
    series = {f"v{i:02d}": shared + 1.5 * rng.standard_normal(shared.shape) for i in range(16)}
    curve = E.isc_curve(series, [], ks=(1, 2, 4, 8, 15), reps=2)
    assert curve["mean_z"][-1] > curve["mean_z"][0]
    ve = E.volunteer_equivalent([curve["mean_z"][2]] * 10, curve)
    assert ve["label"] == "exploratory" and isinstance(ve["claim_k"], int) and 1 <= ve["point_k"] <= 15


# ---- baselines ---------------------------------------------------------------------------------------------------------------
def test_word_rate_and_hrf_regressors():
    wr = E.word_rate_per_s([0.2, 0.7, 3.9, 179.5, 180.0, -1.0], 180)
    assert len(wr) == 180 and wr[0] == 2 and wr[3] == 1 and wr[179] == 1 and wr.sum() == 4
    per_s = {b: np.zeros(190) for b in E.BASELINES}; per_s["motion"][20] = 1.0
    ft = np.arange(10, 185, 0.1); regs = E.baseline_regressors(per_s, ft)
    assert set(regs) == set(E.BASELINES) and np.allclose(regs["word_rate"], 0)
    assert 24.0 < ft[np.argmax(regs["motion"])] < 27.0                             # SPM HRF peak about 5 s after second 20


def test_frame_stats_and_per_second():
    f = np.zeros((48, 4, 4, 3), np.uint8); f[24:] = 200
    b, m = E.frame_stats(f)
    assert b[0] == 0 and abs(b[-1] - 200) < 1e-9 and m[24] > 0 and m[1:24].sum() == 0 and m[25:].sum() == 0
    assert np.allclose(E.per_second(np.r_[np.zeros(24), np.ones(24)], 24), [0, 1])


def test_baselines_scored_like_tribe_and_compared_over_viewers():
    """VIS follows TRIBE's full forecast (TRIBE beats brightness and motion there); LANG follows word rate, which TRIBE's full forecast
    does not carry (word rate beats TRIBE there)."""
    rng = np.random.default_rng(13); per_viewer = []
    for _ in range(12):
        j, ft, ts, _B = viewer(rng); T = len(j); keep = np.ones(T, bool)
        pf = E.parcel_means(E.tribe_at(P_FULL, STARTS, ts), SETS); pa = E.parcel_means(E.tribe_at(P_AUD, STARTS, ts), SETS)
        per_s = {b: smooth(NPRED, 1, rng)[:, 0] for b in E.BASELINES}; regs = E.baseline_regressors(per_s, ft)
        zr = lambda a: (a - a.mean(-1, keepdims=True)) / a.std(-1, keepdims=True)
        bold = {"VIS": zr(pf["AUD"]) + rng.standard_normal((2, T)), "AUD": rng.standard_normal((2, T)),
                "LANG": np.tile(zr(regs["word_rate"]), (2, 1)) + rng.standard_normal((2, T))}
        pred = {"audio": {"AUD": pa["AUD"], "LANG": pa["LANG"]}, "full": {"LANG": pa["LANG"], "VIS": pf["AUD"]}}
        g = E.baseline_gauges(bold, pred, regs, keep, TR, visual=True)
        assert set(g) == {"LANG", "AUD", "VIS"} and set(g["VIS"]) == {"tribe", *E.BASELINES}
        per_viewer.append(g)
    c = E.compare_baselines(per_viewer)
    assert c["label"] == "exploratory" and c["primary"] == ["LANG:word_rate", "AUD:envelope", "VIS:brightness", "VIS:motion"]
    assert c["VIS"]["motion"]["verdict"] == "TRIBE beats" and c["VIS"]["brightness"]["verdict"] == "TRIBE beats"
    assert c["LANG"]["word_rate"]["verdict"] == "baseline beats" and c["VIS"]["motion"]["n"] == 12
    no_vis = E.baseline_gauges(bold, pred, regs, keep, TR, visual=False)
    assert no_vis["VIS"] == "not run" and E.compare_baselines([no_vis] * 3)["VIS"] == "not run"
