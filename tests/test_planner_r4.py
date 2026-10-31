"""R4 with stub clients (no network): the system prompt each version sends, v2's default unchanged, the dev set kept apart from the
R4 held-out set, and the total cap stopping new runs."""
import importlib.util
from pathlib import Path

from stillhere import egress, planner

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rpr4", ROOT / "scripts/run_planner_r4_eval.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
Q = {"id": "T01", "text": "Healthy adults, English LibriVox, intact vs reversed, auditory cortex.", "class": "accept",
     "expected": {"decision": "accept", "reasons": ["none"], "spec": {"population": "healthy_adults"}}}
ACCEPT = {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "analysis": "forecast_tribe_stock",
          "brain_areas": ["auditory"], "conditions": ["intact", "reversed"], "decline_reason": "none"}


def recorder(sent):
    def client(req):
        egress.check(req); sent.append(req.system[0].text)
        return {"content": dict(ACCEPT)}
    return client


def test_each_version_sends_its_own_system_prompt():
    for version, name in (("v2", "planner_system"), ("r4", "planner_system_r4")):
        sent = []
        r = R.run_one(Q, version, 1, base=recorder(sent))
        assert sent == [(ROOT / f"stillhere/public_text/{name}.txt").read_text()]
        assert r["decision"] == "accept" and [c["codes"] for c in r["calls"]] == [[]]


def test_v2_default_is_unchanged():
    sent = []
    planner.plan(egress.PublicQuestion(text=Q["text"]), "spark-1.3", recorder(sent))
    assert sent == [(ROOT / "stillhere/public_text/planner_system.txt").read_text()]


def test_r4_prompt_changes_only_the_analysis_sentence():
    v2 = (ROOT / "stillhere/public_text/planner_system.txt").read_text()
    r4 = (ROOT / "stillhere/public_text/planner_system_r4.txt").read_text()
    old = "Choose the analysis only from the analysis enum; any other analysis declines with analysis_not_offered."
    head, tail = v2.split(old)
    assert r4.startswith(head) and r4.endswith(tail) and old not in r4


def test_dev_and_heldout_are_apart():
    R.check_sets()
    assert {q["id"][:1] for q in R.questions("r4_heldout")} == {"R"} and len(R.questions("r4_heldout")) == 62
    assert len(R.questions("dev")) == 168


def test_total_cap_stops_new_runs(monkeypatch):
    monkeypatch.setitem(R.state, "total_before", R.TOTAL_CAP - 0.001); monkeypatch.setitem(R.state, "spend", 0.002)
    monkeypatch.setitem(R.state, "skipped", 0)
    assert R.run_one(Q, "r4", 1, base=recorder([])) is None and R.state["skipped"] == 1
