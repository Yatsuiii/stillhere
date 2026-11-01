"""R5's scorer (scripts/r5_short_scan.py) on synthetic inputs: windows, the picker's PULA, the ranking and its ties, the gauge statistic
(equal to R2's), the binomial bar, the test, a planted listener effect, the pins and the rule's lines. No Paranoia data."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
spec = importlib.util.spec_from_file_location("r5_short_scan", REPO / "scripts/r5_short_scan.py")
R5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R5)
R2 = R5.R2


def test_windows_are_whole_60_volume_blocks_in_joined_coordinates():
    W = {"story1": 20 + 150 - 1, "story2": 20 + 70 - 1}                   # 150 and 70 scored volumes
    w = R5.windows(W)
    assert w == {"story1": [(0, 60), (60, 120)], "story2": [(150, 210)]}
    assert R5.per_second((150, 152), 150) == [18, 19]                       # volume 20 of story2 is story second 18


def test_pula_is_the_residual_sd_after_audio_and_word_rate():
    rng = np.random.default_rng(0); T = 60
    Pa = rng.standard_normal((3, T)); wr = rng.standard_normal((1, T)); e = rng.standard_normal((3, T))
    e -= np.array([np.linalg.lstsq(np.column_stack([np.ones(T), Pa[i], wr[0]]), e[i], rcond=None)[0] @ np.column_stack([np.ones(T), Pa[i], wr[0]]).T for i in range(3)])
    Pf = 2.0 + 0.5 * Pa + 0.3 * wr + e
    assert R5.pula(Pf, Pa, wr, 0, T) == pytest.approx(float(np.mean(e.std(1))), rel=1e-9)


def test_pick_keeps_a_third_at_each_end_ties_to_the_earlier_window():
    assert [R5.k_of(n) for n in (8, 7, 5, 4, 3, 2, 1)] == [3, 2, 2, 1, 1, 1, 0]
    assert R5.pick([5, 1, 9, 9, 3, 1, 7, 2]) == ([2, 3, 6], [1, 5, 7])
    assert R5.pick([1.0, 1.0, 1.0, 1.0, 1.0]) == ([0, 1], [0, 1])        # all tied: the earlier windows at both ends


def test_gauge_statistic_equals_r2s():
    rng = np.random.default_rng(1); T = 200
    B = rng.standard_normal((5, T)); X = B + rng.standard_normal((5, T)); C = [rng.standard_normal((1, T)) for _ in range(3)]
    assert R5.partial_fz(B, X, C) == pytest.approx(R2.gauge(B, X, C)[1], abs=1e-12)


def test_binomial_bar_and_the_test():
    assert (R5.kstar(22), R5.kstar(16), R5.kstar(10)) == (16, 12, 9)
    t = R5.test(np.linspace(0.1, 1.0, 22))
    assert t["pass"] and t["k_above_0"] == 22 and t["wilcoxon_p"] < 1e-5
    d = np.r_[np.full(15, 0.5), -np.full(7, 0.01)]                          # Wilcoxon passes, but 15 of 22 is under the bar
    t = R5.test(d)
    assert t["wilcoxon_p"] < 0.05 and t["k_above_0"] == 15 and not t["pass"]


def synthetic(planted: bool, n_list=12, seed=3):
    """Joined inputs over two parts (8 and 5 windows): the forecast's own residual amplitude varies by window (that is what PULA ranks);
    with planted, each listener's language rows carry that residual, so where TRIBE's PULA is high the listener follows it better."""
    rng = np.random.default_rng(seed)
    W = {"story1": 20 + 8 * 60 - 1, "story2": 20 + 5 * 60 - 1}; T = 13 * 60; p = 6
    amp = np.repeat(rng.uniform(0.2, 3.0, 13), 60)
    Pa = rng.standard_normal((p, T)); wr = rng.standard_normal((1, T)); dwr = rng.standard_normal((1, T))
    resid = rng.standard_normal((p, T)) * amp
    Pf = 0.7 * Pa + 0.2 * wr + resid
    F0 = {("full", "LANG"): Pf, ("audio", "LANG"): Pa}
    reg = {"wr": wr, "dwr": dwr}
    Bj = {f"s{i}": {"LANG": (0.6 * resid if planted else 0.0) + rng.standard_normal((p, T)) * 2.0} for i in range(n_list)}
    stories = {st: {"wps": rng.uniform(1, 4, 600), "env": rng.uniform(0, 1, 600)} for st in W}
    return Bj, F0, reg, stories, W, sorted(Bj)


def test_planted_effect_passes_and_no_effect_does_not():
    ev = R5.evaluate(*synthetic(True))
    g = ev["group"]
    assert g["tribe"]["pass"] and g["tribe"]["k_above_0"] == 12 and g["minutes_each_side"] == 5
    assert [g["windows"][p]["kept_each_end"] for p in ("story1", "story2")] == [3, 2]
    assert R5.line(g).startswith("Exploratory. On another public story (Paranoia, Finn et al. 2018; 12 listeners") and "5 minutes it ranked highest" in R5.line(g)
    nulls = [R5.evaluate(*synthetic(False, n_list=22, seed=s))["group"] for s in range(100, 160)]
    assert sum(g0["tribe"]["pass"] for g0 in nulls) <= 3                         # listeners of pure noise: at most 5% false PASS
    g0 = next(g0 for g0 in nulls if not g0["tribe"]["pass"])
    assert R5.line(g0).endswith("We show it because we said we would.")


def test_inputs_refuse_until_r2s_pins_are_filled(tmp_path, monkeypatch):
    for k, v in (("STIM_PINS", {}), ("WORD_PINS", {}), ("FORECAST_PINS", {})):     # the state before R2's pins commit (354cab7)
        monkeypatch.setattr(R2, k, v)
    with pytest.raises(SystemExit, match="not filled"):
        R5.check_inputs(str(tmp_path))


def test_lines_are_the_rules_words():
    text = re.sub(r"\n\s+", " ", (REPO / "docs/rules/r5_short_scan_rule.md").read_text())
    for key, label in (("pass", "PASS"), ("fail", "FAIL"), ("not_run", "Not run")):
        assert re.search(rf'- {label}: "(.*?)"(?=\s|$)', text).group(1) == R5.LINES[key], label
