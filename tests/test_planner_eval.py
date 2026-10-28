"""Planner eval scoring (stillhere/planner_eval.py) on stub cards and runs: no model call."""
from stillhere import planner, planner_eval as PE

QS = [{"id": "A", "class": "accept", "expected": {"decision": "accept", "reasons": ["none"],
                                                   "spec": {"population": "healthy_adults", "language": "english",
                                                            "conditions": ["intact", "reversed"]}}},
      {"id": "D", "class": "population", "expected": {"decision": "decline", "reasons": ["population_not_validated", "clinical_out_of_scope"],
                                                       "spec": {}}}]


def card(text, spec, declined):
    return planner.Card(text=text, spec=spec, declined=declined)


def run(qid, backend, decision, reason=None, spec=None, codes=([],), rule_made=False, tools=("forecast",), s=1.0, tok=(100, 200)):
    calls = [{"kind": "plan", "codes": list(c), "s": s, "prompt_tokens": tok[0], "completion_tokens": tok[1]} for c in codes]
    return {"qid": qid, "backend": backend, "decision": decision, "reason": reason, "spec": spec or {}, "rule_made": rule_made,
            "tools": list(tools) if decision == "accept" else [], "calls": calls}


ACC = {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "brain_areas": [],
       "conditions": ["reversed", "intact"], "decline_reason": "none"}


def test_outcome_reads_the_card():
    assert PE.outcome(card("StudySpec: ...", ACC, False))["decision"] == "accept"
    d = PE.outcome(card("Declined: population_not_validated. ...", planner.decline_spec("population_not_validated"), True))
    assert d["decision"] == "decline" and d["reason"] == "population_not_validated" and not d["rule_made"]
    r = PE.outcome(card("Declined: x. ... not from the model.", planner.decline_spec("analysis_not_offered"), True))
    assert r["rule_made"]
    assert PE.outcome(card("Planner unavailable: the model call failed (ConnectError).", {}, True))["decision"] == "unavailable"


def test_score_run_fields_reasons_and_first_answer():
    s = PE.score_run(QS[0], run("A", "x", "accept", spec=ACC))
    assert s["decision_ok"] and (s["field_hits"], s["field_total"]) == (3, 3) and s["first_valid"] and s["tool_ok"]
    wrong = dict(ACC, language="french")
    assert PE.score_run(QS[0], run("A", "x", "accept", spec=wrong))["field_hits"] == 2
    s = PE.score_run(QS[1], run("D", "x", "decline", "clinical_out_of_scope", codes=(["schema_invalid"], [])))
    assert s["decision_ok"] and s["reason_ok"] and not s["first_valid"]
    assert not PE.score_run(QS[1], run("D", "x", "decline", "language_not_validated"))["reason_ok"]
    s = PE.score_run(QS[1], run("D", "x", "accept", spec=ACC))
    assert s["false_accept"] and not s["decision_ok"]
    s = PE.score_run(QS[0], run("A", "x", "unavailable", codes=()))
    assert s["unavailable"] and not s["decision_ok"] and not s["answered"]


def test_summary_bars_agreement_and_cost():
    rs = [run("A", "spark-1.3", "accept", spec=ACC) for _ in range(3)] + \
         [run("D", "spark-1.3", "decline", "population_not_validated") for _ in range(3)] + \
         [run("A", "spark-open", "accept", spec=ACC) for _ in range(3)] + \
         [run("D", "spark-open", "accept", spec=ACC), run("D", "spark-open", "decline", "population_not_validated"),
          run("D", "spark-open", "decline", "population_not_validated", rule_made=True, codes=(["schema_invalid"], ["schema_invalid"]))]
    out = PE.summarize(rs, QS)
    m1, m2 = out["backends"]["spark-1.3"], out["backends"]["spark-open"]
    assert m1["decision_accuracy"] == 1.0 and all(v["pass"] for v in m1["bars"].values())
    assert m2["decision_accuracy"] == round(5 / 6, 4) and m2["false_accept_rate"] == round(1 / 3, 4)
    assert m2["bars"]["decision_accuracy"]["pass"] is False and m2["bars"]["rule_made_decline_rate"]["pass"] is False
    assert m2["repeat_consistency"] == 0.5 and m2["questions_below_two_of_three"] == []
    assert out["cross_backend_agreement"]["value"] == 1.0                       # modal outcome agrees on both questions
    assert m1["cost_usd"] == round(PE.cost(600, 1200), 6) and m1["prompt_tokens"] == 600


def test_spend_cap_whichever_first():
    assert PE.over_cap({"spark-open": 14.99, "spark-1.3": 24.99}) is None
    assert PE.over_cap({"spark-open": 15.0, "spark-1.3": 1.0}) == "spark-open"
    assert PE.over_cap({"spark-open": 0.1, "spark-1.3": 25.0}) == "spark-1.3"
