"""Emo-FilM scorer on synthetic data only (no Emo-FilM file is read): scored-volume mapping, alignment sign and onset, a planted
forecast recovered and a null not, censoring and the D7 run rule, FD, motion-row mapping, bootstrap determinism, and the ISC(k) curve
with TRIBE placed on it. Needs nilearn (skipped without it)."""
import numpy as np
import pytest

pytest.importorskip("nilearn")
from stillhere import emofilm_score as E  # noqa: E402

TR, ONSET, SLICE, NPRED, NVOL = 1.3, 93.887, 0.65, 589, 607
rng0 = np.random.default_rng(7)


def smooth(n, v, rng):
    x = rng.standard_normal((n, v)); k = np.ones(8) / 8
    return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, x)


P_AUD, P_FULL = smooth(NPRED, 20, rng0), smooth(NPRED, 20, rng0)
SETS = {"AUD": [np.arange(0, 10), np.arange(10, 20)], "LANG": [np.arange(0, 10), np.arange(10, 20)]}


def viewer(rng, planted=True, onset=ONSET, gain=1.0):
    """Parcel rows on the scored volumes: the prediction read at the true sample times (planted) plus noise."""
    j, ft, ts = E.film_volumes(NVOL, TR, onset, SLICE, NPRED)
    sig = E.parcel_means(E.tribe_at(P_AUD, ts), SETS)["AUD"] if planted else np.zeros((2, len(j)))
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-12)     # gain = signal-to-noise ratio
    return j, ft, ts, gain * sig + rng.standard_normal((2, len(j)))


def test_scored_volumes_map_film_time_and_tribe_offset():
    j, ft, ts = E.film_volumes(NVOL, TR, ONSET, SLICE, NPRED)
    assert ft[0] >= E.DROP_S and (j[0] - 1) * TR + SLICE - ONSET < E.DROP_S and ft[-1] <= E.CUT_S
    assert np.allclose(ts, ft - E.TRIBE_OFFSET) and ts.min() >= 0 and ts.max() <= NPRED - 1
    assert np.allclose(j * TR + SLICE - ONSET, ft)


def test_alignment_sign_and_onset_matter():
    rng = np.random.default_rng(1); j, ft, ts, B = viewer(rng)
    keep = np.ones(len(j), bool); X = lambda t: E.parcel_means(E.tribe_at(P_AUD, t), SETS)["AUD"]
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
            keep = np.ones(len(j), bool); X = E.parcel_means(E.tribe_at(P_AUD, ts), SETS)["AUD"]
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
    pa = E.parcel_means(E.tribe_at(P_AUD, ts), SETS); pf = E.parcel_means(E.tribe_at(P_FULL, ts), SETS)
    out = E.viewer_scores({"AUD": B, "LANG": B}, {"audio": pa, "full": pf}, rng.standard_normal(len(j)), keep, TR, visual=False)
    assert set(out) == {"HEARING", "LANGUAGE", "VISUAL"} and out["VISUAL"] == "not run" and out["HEARING"]["z"] > 2


def test_isc_curve_rises_with_k_and_tribe_is_placed():
    rng = np.random.default_rng(9); grid = np.arange(E.DROP_S, E.CUT_S, TR); shared = smooth(len(grid), 2, rng).T
    series = {f"v{i:02d}": shared + 1.5 * rng.standard_normal(shared.shape) for i in range(16)}
    curve = E.isc_curve(series, [], ks=(1, 2, 4, 8, 15), reps=2)
    assert curve["mean_z"][-1] > curve["mean_z"][0]
    ve = E.volunteer_equivalent([curve["mean_z"][2]] * 10, curve)
    assert ve["label"] == "exploratory" and isinstance(ve["claim_k"], int) and 1 <= ve["point_k"] <= 15
