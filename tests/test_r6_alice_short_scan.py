"""R6's scorer (scripts/r6_alice_short_scan.py) on synthetic inputs only: windows at Alice's TR, a planted listener effect, the null's
false-PASS rate, the guards and the rule's lines. No Alice data."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
spec = importlib.util.spec_from_file_location("r6_alice_short_scan", REPO / "scripts/r6_alice_short_scan.py")
R6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R6)


def test_windows_are_whole_minutes_at_tr_2():
    assert R6.WIN == 30 and R6.H.TR == 2.0 and R6.H.J0 == 10
    w = R6.windows(371)                                                       # the Alice full window, volumes 10 to 371
    assert len(w) == 12 and w[0] == (0, 30) and w[-1] == (330, 360)           # volumes 370 and 371 are not used
    assert R6.R5.k_of(len(w)) == 4
    assert R6.seconds(0, 30) == list(range(20, 80)) and R6.seconds(330, 360) == list(range(680, 740))
    assert len(R6.windows(154)) == 4                                          # the 145-volume window holds 4 minutes


def synthetic(planted: bool, n_list=22, seed=3, j1=371):
    """Inputs on volumes 10..j1: the forecast's own residual amplitude varies by window (what PULA ranks); with planted, each listener's
    language rows carry that residual, so where TRIBE's PULA is high the listener follows it better."""
    rng = np.random.default_rng(seed)
    T = j1 - 10 + 1; p = 6
    amp = np.r_[np.repeat(rng.uniform(0.2, 3.0, T // 30), 30), np.ones(T % 30)]
    Pa = rng.standard_normal((p, T)); wr = rng.standard_normal((1, T)); dwr = rng.standard_normal((1, T))
    resid = rng.standard_normal((p, T)) * amp
    Pf = 0.7 * Pa + 0.2 * wr + resid
    subs = [f"{i:02d}" for i in range(n_list)]
    B = {s: (0.6 * resid if planted else 0.0) + rng.standard_normal((p, T)) * 2.0 for s in subs}
    X = {s: Pf for s in subs}
    C = {s: [Pa.mean(0, keepdims=True), wr, dwr] for s in subs}
    return B, X, C, Pf, Pa, wr, rng.uniform(1, 4, 800), rng.uniform(0, 1, 800), j1, subs


def test_planted_effect_passes_and_no_effect_does_not():
    g = R6.evaluate(*synthetic(True))["group"]
    assert g["tribe"]["pass"] and g["tribe"]["k_above_0"] == 22 and g["minutes_each_side"] == 4
    assert g["windows"]["n"] == 12 and g["windows"]["kept_each_end"] == 4 and g["windows"]["volumes"][0] == [10, 39]
    assert g["windows"]["stimulus_seconds"][0] == [20, 80]
    assert R6.line(g).startswith(R6.LABEL + ". A replication attempt of R5: on a second public story") and "4 minutes it ranked highest" in R6.line(g)
    assert R6.LABEL == "Exploratory; the Alice scans were read on Oct 30 for the held-out test"
    nulls = [R6.evaluate(*synthetic(False, seed=s))["group"] for s in range(100, 160)]
    assert sum(g0["tribe"]["pass"] for g0 in nulls) <= 3                       # listeners of pure noise: at most 5% false PASS
    g0 = next(g0 for g0 in nulls if not g0["tribe"]["pass"])
    assert R6.line(g0).endswith("R5's Paranoia result is not replicated here. We show it because we said we would.")


def test_per_listener_values_stay_out_of_the_group_numbers():
    ev = R6.evaluate(*synthetic(True, n_list=12))
    assert set(ev["per_listener"]) == {f"{i:02d}" for i in range(12)} and "per_listener" not in ev["group"]


def test_the_guards_refuse_before_reading_anything(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="outside the repository"):
        R6.guard(str(REPO / "out"))
    monkeypatch.setattr(R6, "on_main", lambda: False)
    with pytest.raises(SystemExit, match="on origin/main"):
        R6.guard(str(tmp_path / "out"))
    monkeypatch.setattr(R6, "on_main", lambda: True)
    done = tmp_path / "r6.json"; done.write_text("{}")
    monkeypatch.setattr(R6, "RESULTS", str(done))
    with pytest.raises(SystemExit, match="scored once"):
        R6.guard(str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="missing inputs"):
        R6.check_inputs(str(tmp_path))


def test_lines_are_the_rules_words():
    text = re.sub(r"\n\s+", " ", (REPO / "docs/rules/r6_alice_short_scan_rule.md").read_text())
    for key, label in (("pass", "PASS"), ("fail", "FAIL"), ("not_run", "Not run")):
        assert re.search(rf'- {label}: "(.*?)"(?=\s|$)', text).group(1) == R6.LINES[key], label


def test_the_forecast_pins_are_the_alice_rules():
    text = (REPO / "docs/rules/alice_heldout_rule.md").read_text()
    assert all(h in text for h in R6.FORECAST_PINS.values())
