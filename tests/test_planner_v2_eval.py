"""The v2 runner with stub clients (no network): a decline with filled fields is one valid call, a backend stops at its cap, and the
two agreement bars are computed over three backends."""
import importlib.util
from pathlib import Path

from stillhere import egress, planner

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rpv2", ROOT / "scripts/run_planner_v2_eval.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
Q = {"id": "T01", "text": "Plan the study in 7-year-old children.", "class": "population",
     "expected": {"decision": "decline", "reasons": ["population_not_validated"], "spec": {}}}


def filled_decline(req):
    egress.check(req)
    return {"content": {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "analysis": "forecast_tribe_stock",
                        "brain_areas": ["auditory"], "conditions": ["intact", "reversed"], "decline_reason": "population_not_validated"}}


def test_filled_decline_is_one_valid_call_and_caps_stop_a_backend(monkeypatch):
    monkeypatch.setattr(planner, "live_client", lambda timeout=60.0: filled_decline)
    r = R.run_one("heldout", Q, "spark-open-baseten", 1)
    assert r["decision"] == "decline" and r["reason"] == "population_not_validated" and not r["rule_made"]
    assert [c["codes"] for c in r["calls"]] == [[]] and r["trace"][0] == "plan:ok:normalised"
    R.spend["spark-open"] = R.CAPS["spark-open"]
    assert R.run_one("heldout", Q, "spark-open", 1) is None and R.skipped["spark-open"] == 1


def test_agreements_over_three_backends():
    def run(b, rep, reason):
        return {"backend": b, "qid": "T01", "repeat": rep, "decision": "decline", "reason": reason, "spec": {}}
    runs = [run(b, rep, "population_not_validated") for b in R.CAPS for rep in (1, 2, 3)]
    ag = R.agreements(runs)
    assert ag["meta_vs_fireworks"]["pass"] and ag["baseten_vs_fireworks"]["pass"]
    runs = [run(b, rep, "population_not_validated" if b != "spark-open-baseten" else "language_not_validated") for b in R.CAPS for rep in (1, 2, 3)]
    assert not R.agreements(runs)["baseten_vs_fireworks"]["pass"]
