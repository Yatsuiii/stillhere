"""Writes the two SYNTHETIC desk fixtures (random tables, no real data): gate1_bars.gate1() run on the synthetic score table of the
Gate 1 unit tests, once with strong effects (bars pass) and once with no language drop (P1 and P3 fail). The output layout is the one
frozen.compose() builds from the real tier-D files, so the real results drop in unchanged after Gate 1.
Needs numpy and scipy (run with the d21 interpreter); reads the frozen Gate 1 code read-only and writes no bytecode next to it.
usage: python scripts/make_synthetic_gate1_fixtures.py GATE1_DIR"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gate1_bars as G                                       # noqa: E402
from test_gate1_bars import table                            # noqa: E402

from stillhere.frozen import compose                         # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
for name, kw in (("synthetic_gate1_pass", {}), ("synthetic_gate1_p1_p3_fail", {"lang_drop": 0.0, "seed": 1})):
    T, subs = table(**kw)
    res = G.gate1(T, deep_effective_n=len(subs))
    res["demo_volunteer_D14"] = G.demo_volunteer(T, subs)
    d7 = {"included": {st: subs for st in ("awake", "light", "deep", "recovery")}, "deep_effective_n": len(subs)}
    doc = compose(json.loads(json.dumps(res, default=float)), d7, trs_per_state=145, source="synthetic")
    (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(name, "| sedation line", res["sedation_stage_line"], "| P1", res["bars"]["P1"]["pass"], "| P3", res["bars"]["P3"]["pass"])
