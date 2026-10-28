"""Synthetic tests for stillhere/heldout.py (random data only; needs numpy, scipy, nilearn)."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("nilearn")
from stillhere import heldout as H  # noqa: E402


def _naive_gauge(B, X, controls):
    """The gauge written out loop by loop: lstsq residuals on [controls, 1], Pearson r per row, Fisher z mean, np.roll null."""
    T = B.shape[1]
    C = np.vstack(list(controls) + [np.ones((1, T))]).T
    res = lambda a: a - C @ np.linalg.lstsq(C, a, rcond=None)[0]

    def score(x):
        rs = []
        for p in range(B.shape[0]):
            b = res((B[p] - B[p].mean()) / B[p].std())
            q = res(x[p])
            rs.append(np.corrcoef(b, q)[0, 1])
        return np.arctanh(np.clip(rs, -0.999, 0.999)).mean()
    real = score(X)
    null = np.array([score(np.roll(X, k, 1)) for k in range(H.MIN, T - H.MIN)])
    return (real - null.mean()) / null.std(), real


def test_gauge_equals_the_loop_definition():
    rng = np.random.default_rng(0)
    B, X, ctl = rng.standard_normal((4, 80)), rng.standard_normal((4, 80)), [rng.standard_normal((1, 80)), rng.standard_normal((1, 80))]
    for c in (ctl, H.zeros_control(80)):
        z, real = H.gauge(B, X, c)
        z0, real0 = _naive_gauge(B, X, c)
        assert abs(z - z0) < 1e-8 and abs(real - real0) < 1e-10


def test_gauge_finds_a_planted_signal_and_partials_out_controls():
    rng = np.random.default_rng(1)
    T = 145
    sig = np.convolve(rng.standard_normal(T + 20), np.ones(5) / 5, "same")[:T]
    X = np.tile(sig, (6, 1))
    assert H.gauge(X + 2.0 * rng.standard_normal((6, T)), X, H.zeros_control(T))[0] > 3
    assert abs(H.gauge(rng.standard_normal((6, T)), X, H.zeros_control(T))[0]) < 3
    shared = np.convolve(rng.standard_normal(T + 20), np.ones(5) / 5, "same")[:T]
    B = np.tile(shared, (6, 1)) + 0.5 * rng.standard_normal((6, T))
    X2 = np.tile(shared, (6, 1)) + 0.5 * rng.standard_normal((6, T))
    assert H.gauge(B, X2, H.zeros_control(T))[0] > 3                       # shared component drives the plain score
    assert H.gauge(B, X2, [shared[None]])[0] < 2                           # and is removed when it is a control


def test_hrf_regressor_lag_sign():
    x = np.zeros(120)
    x[20] = 1.0
    t = np.arange(0, 100, 0.1)
    peak = t[np.argmax(H.hrf_regressor(x, t))]
    assert 24 < peak < 27                                                  # SPM HRF peaks about 5 s after the impulse
    assert abs(t[np.argmax(H.hrf_regressor(x, t - 3.0))] - (peak + 3.0)) < 0.11    # reading at t - L moves the response L later


def test_align_reads_two_j_minus_five():
    starts = np.arange(100)
    P = np.tile(starts[:, None].astype(np.float32), (1, 3))
    a = H.align(P, starts, 40)
    assert np.isnan(a[:3]).all() and np.allclose(a[3:, 0], 2.0 * np.arange(3, 40) - 5)
    assert np.isnan(H.align(P, starts, 60)[53:]).all()                    # 2j - 5 > 99 s is past the last prediction


def test_textgrid_words_and_pause_marks(tmp_path):
    rows = [("0", "1", "sp"), ("1", "1.5", "The"), ("1.5", "2", "SP"), ("2", "2.4", "<p>"), ("2.4", "3", "cat"), ("3", "3.2", "Sp"), ("3.2", "4", "")]
    body = "".join(f'        intervals [{i}]:\n            xmin = {a} \n            xmax = {b} \n            text = "{t}" \n' for i, (a, b, t) in enumerate(rows, 1))
    p = tmp_path / "x.TextGrid"
    p.write_text('File type = "ooTextFile"\nitem [1]:\n        name = "TokensAlign" \n' + body, encoding="utf-8")
    iv = H.textgrid_intervals(p)
    assert list(H.word_onsets(iv)) == [1.0, 2.4, 3.0]                     # word rate drops sp, SP and <p> exactly
    assert H.word_list(iv) == [("the", 1.0), ("cat", 2.4)]                # baseline features drop them in any case


def test_tree_leaves_depths():
    lv = H.tree_leaves(["(S (NP (DT the) (NN cat)) (VP (VBD sat)))"])
    assert [(l["w"], l["td"], l["bu"]) for l in lv] == [("the", 3, 1), ("cat", 1, 2), ("sat", 2, 3)]


def test_per_second_centres_and_fills():
    x = H.per_second([1.0, np.nan, 3.0], [0.2, 1.5, 1.7], 4)
    assert np.allclose(x, [-1.0, 1.0, 0.0, 0.0])                           # values 1, 2, 3 -> -1, 0, 1; seconds 0, 1, 1


def test_volunteer_equivalent_mapping():
    v = H.volunteer_equivalent([2.5] * 10, [1, 2, 4], [1.0, 2.0, 3.0], nboot=200)
    assert v["point_k"] == 3.0 and v["lower_k"] == 3.0 and v["claim_k"] == 3
    v = H.volunteer_equivalent([0.5] * 10, [1, 2, 4], [1.0, 2.0, 3.0], nboot=200)
    assert v["point_k"] == 0.0 and v["claim_k"] == 0
    v = H.volunteer_equivalent([9.0] * 10, [1, 2, 4], [1.0, 2.0, 3.0], nboot=200)
    assert v["point_k"] == 4.0


def _listeners(n, T, seed):
    rng = np.random.default_rng(seed)
    s = np.convolve(rng.standard_normal(T + 20), np.ones(3) / 3, "same")[:T]
    s = s / s.std()
    bold = {f"sub-{i:02d}": {net: np.tile(s, (p, 1)) + 2.0 * rng.standard_normal((p, T)) for net, p in (("AUD", 2), ("LANG", 3))} for i in range(n)}
    return bold, s


def test_isc_curve_rises_with_k_and_is_seeded():
    bold, _ = _listeners(6, 160, 2)
    ctrl = {"AUD": H.zeros_control(145), "LANG": H.zeros_control(145)}
    a = H.isc_curve(bold, ctrl, [1, 5], 3, 11)
    assert a == H.isc_curve(bold, ctrl, [1, 5], 3, 11)
    for net in ("AUD", "LANG"):
        assert np.mean([z for z, _ in a[net][5]]) > np.mean([z for z, _ in a[net][1]])


def test_hybrid_gain_and_wrong_story_check():
    bold, s = _listeners(6, 160, 3)
    wrong, _ = _listeners(5, 160, 4)                                       # another story: independent shared signal
    ctrl = {"AUD": H.zeros_control(145), "LANG": H.zeros_control(145)}
    cl = lambda A: H.clean(A, 0.03, H.J0, 154)
    tribe = {net: cl(np.tile(s, (p, 1))) for net, p in (("AUD", 2), ("LANG", 3))}
    wrong_tribe = cl(np.tile(np.random.default_rng(5).standard_normal(160), (3, 1)))
    m = H.hybrid(bold, tribe, ctrl, wrong, wrong_tribe, [1, 2], 3, 13, 200)
    for net in ("AUD", "LANG"):
        assert m[net][1]["dz"] > 0 and m[net][1]["dz_ci95"][0] > 0        # a model that carries the story helps few controls
    assert m["wrong_story"][1]["alone_z"] < m["LANG"][1]["alone_z"]       # other-story controls track this listener less
    assert m == H.hybrid(bold, tribe, ctrl, wrong, wrong_tribe, [1, 2], 3, 13, 200)
