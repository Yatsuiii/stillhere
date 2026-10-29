"""The ghost-line data agree with docs/rules/display_rule_ghostline.md: the script's constants match the rule, the 5 listeners
reproduce from the seed, the line builder z-scores as stated, and results/ghostline_lpp_s8.json (once written) has the draw order,
chips, labels and short view the rule fixes, no banned words and no verdict word on the single listener."""
import datetime
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RULE = ROOT / "docs/rules/display_rule_ghostline.md"
DATA = ROOT / "results/ghostline_lpp_s8.json"
BANNED = (r"(un)?aware(ness)?", r"(un|sub)?conscious(ness|ly)?", r"hospital\w*", r"detect\w*", r"diagnos(e|ed|es|ing)",
          r"a test for patients", r"validat\w*", r"before anyone looked")
VERDICT = r"pass\w*|fail\w*|hit|miss\w*|good|bad|best|worst|typical|significant|\d"


def script():
    spec = importlib.util.spec_from_file_location("ghostline_lpp", ROOT / "scripts/ghostline_lpp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def test_script_matches_rule():
    g, rule = script(), flat(RULE.read_text())
    for sid in g.FIVE:
        assert f"`{sid}`" in rule
    assert f"default_rng({g.FIVE_SEED})" in rule
    assert f"| {g.CHIP['short']} | {g.CHIP['long']} |" in rule
    assert g.FROZEN_LINE in rule and g.LABELS["one"] in rule and g.TITLE in rule
    assert (g.J0, g.J1, g.SHORT_J1) == (10, 154, 69) and "volumes 10 to 154" in rule and "volumes 10 to 69" in rule
    assert g.ORDER == ["forecast", "chip", "listeners", "one_listener"]
    assert len(g.ALL_IDS) == 49 and list(g.ALL_IDS) == sorted(g.ALL_IDS)


def test_five_reproduce_from_seed():
    g = script()
    pick = np.random.default_rng(g.FIVE_SEED).choice(list(g.ALL_IDS), 5, replace=False)
    assert tuple(sorted(pick.tolist())) == g.FIVE


def test_median_listener_and_lines_on_synthetic_rows():
    pytest.importorskip("nilearn")
    g = script()
    ids = [f"s{i:02d}" for i in range(49)]
    z = list(np.linspace(-1, 2, 49))[::-1]
    assert g.median_listener(ids, z) == "s24"
    rng = np.random.default_rng(0)
    rows = {s: rng.standard_normal((6, 170)) for s in ids}
    ln = g.lines(rows, rng.standard_normal((6, 170)), five=tuple(ids[:5]), one=ids[3])
    for v in ln.values():
        assert v.shape == (145,) and abs(v.mean()) < 1e-9 and abs(v.std(ddof=1) - 1) < 1e-9


@pytest.mark.skipif(not DATA.exists(), reason="results/ghostline_lpp_s8.json not written yet (after Oct 29 16:00 UTC)")
def test_data_file_follows_rule():
    g, d = script(), json.loads(DATA.read_text())
    text = DATA.read_text()
    for pat in BANNED:
        assert not re.search(pat, text, flags=re.I), pat
    assert chr(0x2014) not in text
    assert d["draw_order"] == g.ORDER and d["chip"] == g.CHIP and d["title"] == g.TITLE
    assert d["frozen_line"]["text"] == g.FROZEN_LINE and text.count("volunteers") == 1
    L = d["lines"]
    assert [L[k]["label"] for k in ("forecast", "group", "five", "one")] == [g.LABELS[k] for k in ("forecast", "group", "five", "one")]
    assert L["forecast"]["style"] == "gray dashed" and L["one"]["style"] == "thin solid"
    assert not re.search(VERDICT, L["one"]["label"].replace("1 public", ""), flags=re.I)
    assert tuple(L["five"]["listeners"]) == g.FIVE and L["five"]["seed"] == g.FIVE_SEED
    assert d["volumes"] == [10, 154] and len(d["time_s"]) == 145 and d["short_view"]["index"] == [0, 60]
    for k in ("forecast", "group", "five", "one"):
        v = np.array(L[k]["values"])
        assert v.shape == (145,) and abs(v.mean()) < 0.01 and abs(v.std(ddof=1) - 1) < 0.01
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%h", "--", "docs/rules/display_rule_ghostline.md"],
                         capture_output=True, text=True).stdout.split()
    assert d["rule"]["commit"] in log
    at = datetime.datetime.fromisoformat(d["extracted_utc"])
    assert at >= datetime.datetime.fromisoformat("2026-10-29T16:00:00+00:00") and at > datetime.datetime.fromisoformat(d["rule"]["committed"])
