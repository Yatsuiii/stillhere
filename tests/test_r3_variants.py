"""R3's pure parts on synthetic data (docs/rules/r3_variants_rule.md): the 1-ring operator, smoothing, averaging, the choice of k and
the adoption decision. The harness itself is section 3's and D3's, kept in the analysis tree; the run reproduces D3's and chk8's
numbers before it scores any variant."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_variants", REPO / "lpp_variants/score_variants.py")
SV = importlib.util.module_from_spec(spec); spec.loader.exec_module(SV)

# two hemispheres of 4 vertices each, two triangles per hemisphere sharing an edge (vertex 0 and 3 are not neighbours)
FACES = np.array([[0, 1, 2], [1, 2, 3]])


def op():
    return SV.ring_operator(FACES, FACES, n_lh=4)


def test_ring_operator_rows_and_hemispheres():
    S = op().toarray()
    assert S.shape == (8, 8) and np.allclose(S.sum(1), 1)
    assert not S[:4, 4:].any() and not S[4:, :4].any()               # the hemispheres stay apart
    assert S[0, 3] == 0 and S[0, 0] == pytest.approx(1 / 3) and S[1, 1] == pytest.approx(1 / 4)


def test_smooth_passes():
    S = op(); x = np.random.default_rng(0).standard_normal((8, 5))
    assert np.array_equal(SV.smooth(x, S, 0), x)
    assert np.allclose(SV.smooth(x, S, 2), S @ (S @ x))
    assert np.allclose(SV.smooth(np.ones((8, 3)), S, 3), 1)          # a constant map stays constant


def test_average_needs_the_same_seconds():
    a = {"P": np.ones((3, 2), "float32"), "starts": np.arange(3)}; b = {"P": np.full((3, 2), 3, "float32"), "starts": np.arange(3)}
    assert np.allclose(SV.average(a, b)["P"], 2)
    with pytest.raises(AssertionError, match="different seconds"):
        SV.average(a, {**b, "starts": np.arange(1, 4)})


def test_choose_k_ties_go_to_the_smaller():
    assert SV.choose_k({1: 1.70, 2: 1.75, 3: 1.75}) == 2
    assert SV.choose_k({1: 1.80, 2: 1.75, 3: 1.60}) == 1


def ids(n=28):
    return [f"sub-FR{i:03d}" for i in range(n)]


def test_decide_adopts_only_past_both_steps():
    rng = np.random.default_rng(1); base = {s: float(v) for s, v in zip(ids(), rng.standard_normal(28))}
    up = {s: v + 0.25 + 0.01 * i for i, (s, v) in enumerate(base.items())}
    assert SV.decide(1.7, 1.8, base, up)["adopted"] is True
    d = SV.decide(1.7, 1.6, base, up)                                  # French clears the bar, English did not improve
    assert d["bar_ok"] and not d["dev_ok"] and d["adopted"] is False
    small = {s: v + 0.15 for s, v in base.items()}                     # every listener higher, mean below +0.20
    d = SV.decide(1.7, 1.8, base, small)
    assert d["fr_higher"] == 28 and d["fr_p"] < 0.05 and d["adopted"] is False


def test_decide_needs_a_significant_gain():
    base = {s: 1.0 for s in ids()}
    mixed = {s: 1.0 + (2.0 if i < 4 else -0.05) for i, s in enumerate(ids())}   # a +0.24 mean carried by 4 of 28 listeners
    d = SV.decide(1.7, 1.8, base, mixed)
    assert d["fr_mean_difference"] >= 0.2 and d["fr_p"] >= 0.05 and d["adopted"] is False
    assert SV.decide(1.7, 1.8, base, dict(base))["fr_p"] == 1.0
    with pytest.raises(AssertionError, match="different French listeners"):
        SV.decide(1.7, 1.8, base, {s: 1.0 for s in ids(27)})


def test_pins_refuse_missing_or_changed(tmp_path):
    f = tmp_path / "pred.npz"; f.write_bytes(b"forecast")
    with pytest.raises(SystemExit, match="not pinned"):
        SV.check_pins(None, "laptop French forecasts")
    assert SV.check_pins({str(f): SV.sha256(f)}, "x") == {str(f): SV.sha256(f)}
    f.write_bytes(b"another forecast")
    with pytest.raises(SystemExit, match="differ from the pinned"):
        SV.check_pins({str(f): "0" * 64}, "x")
