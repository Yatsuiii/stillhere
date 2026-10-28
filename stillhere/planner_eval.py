"""Scoring for the Planner v0 evaluation (docs/rules/planner_eval_rule.md; exploratory). Pure functions over run records, so the
scoring is unit-tested with stubs; scripts/run_planner_eval.py makes the calls. Bars, prices and caps are the rule's, set before any run."""
import statistics
from collections import Counter

PRICE_IN, PRICE_OUT = 1.25e-6, 4.25e-6                    # $ per token, both backends (rule: Spend cap)
CAPS = {"spark-open": 15.0, "spark-1.3": 25.0}
BARS = {"decision_accuracy": (">=", 0.90), "false_accept_rate": ("<=", 0.05), "decline_reason_accuracy": (">=", 0.80),
        "spec_field_accuracy": (">=", 0.90), "first_answer_validity": (">=", 0.85), "rule_made_decline_rate": ("<=", 0.05),
        "planner_unavailable_rate": ("<=", 0.02)}
AGREEMENT_BAR = 0.80
FIELDS = ("population", "language", "stimulus", "brain_areas", "conditions")
FALLBACK_NOTE = "not from the model"


def cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens or 0) * PRICE_IN + (completion_tokens or 0) * PRICE_OUT


def over_cap(spend: dict) -> str | None:
    """The first backend whose spend reached its cap (the run stops for both, whichever comes first), else None."""
    return next((b for b, cap in CAPS.items() if spend.get(b, 0.0) >= cap), None)


def outcome(card) -> dict:
    """decision (accept, decline or unavailable), reason and spec of one planner card."""
    if card.text.startswith("Planner unavailable"):
        return {"decision": "unavailable", "reason": None, "spec": {}}
    reason = (card.spec or {}).get("decline_reason")
    return {"decision": "decline" if card.declined else "accept", "reason": reason, "spec": card.spec,
            "rule_made": FALLBACK_NOTE in card.text}


def _field_hits(expected: dict, spec: dict) -> tuple:
    hits = total = 0
    for f in FIELDS:
        if f in expected:
            total += 1
            got = spec.get(f)
            hits += (sorted(got) == sorted(expected[f])) if isinstance(expected[f], list) and isinstance(got, list) else (got == expected[f])
    return hits, total


def score_run(q: dict, run: dict) -> dict:
    exp = q["expected"]; d = run["decision"]
    s = {"decision_ok": d == exp["decision"], "unavailable": d == "unavailable", "rule_made": bool(run.get("rule_made")),
         "false_accept": exp["decision"] == "decline" and d == "accept"}
    plan_codes = [c.get("codes") for c in run.get("calls", []) if c.get("kind") == "plan" and "codes" in c]
    s["answered"] = bool(plan_codes)
    s["first_valid"] = bool(plan_codes) and plan_codes[0] == []
    if exp["decision"] == "decline" and d == "decline":
        s["reason_ok"] = run["reason"] in exp["reasons"]
    if exp["decision"] == "accept" and d == "accept":
        s["field_hits"], s["field_total"] = _field_hits(exp["spec"], run.get("spec") or {})
        s["tool_ok"] = bool(run.get("tools"))
    return s


def _rate(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def modal(runs: list) -> tuple:
    """The most common (decision, reason-if-decline) over a question's repeats; ties broken by the sorted key."""
    keys = Counter((r["decision"], r["reason"] if r["decision"] == "decline" else None) for r in runs)
    return sorted(keys.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def summarize(runs: list, questions: list) -> dict:
    Q = {q["id"]: q for q in questions}; out = {"backends": {}}
    by_b = {}
    for r in runs:
        by_b.setdefault(r["backend"], []).append(r)
    for b, rs in sorted(by_b.items()):
        sc = [score_run(Q[r["qid"]], r) for r in rs]
        dec = [s for r, s in zip(rs, sc) if Q[r["qid"]]["expected"]["decision"] == "decline"]
        calls = [c for r in rs for c in r.get("calls", [])]
        lat = sorted(c["s"] for c in calls)
        m = {"runs": len(rs), "questions": len({r["qid"] for r in rs}),
             "decision_accuracy": _rate(s["decision_ok"] for s in sc),
             "false_accept_rate": _rate(s["false_accept"] for s in dec),
             "decline_reason_accuracy": _rate(s["reason_ok"] for s in sc if "reason_ok" in s),
             "spec_field_accuracy": (round(sum(s["field_hits"] for s in sc if "field_total" in s)
                                           / max(1, sum(s["field_total"] for s in sc if "field_total" in s)), 4)
                                     if any("field_total" in s for s in sc) else None),
             "first_answer_validity": _rate(s["first_valid"] for s in sc if s["answered"]),
             "rule_made_decline_rate": _rate(s["rule_made"] for s in sc),
             "planner_unavailable_rate": _rate(s["unavailable"] for s in sc),
             "tool_step_ok_rate": _rate(s["tool_ok"] for s in sc if "tool_ok" in s),
             "calls": len(calls), "latency_median_s": round(statistics.median(lat), 2) if lat else None,
             "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2) if lat else None,
             "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
             "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls)}
        m["cost_usd"] = round(cost(m["prompt_tokens"], m["completion_tokens"]), 6)
        per_q = {}
        for r in rs:
            per_q.setdefault(r["qid"], []).append(r)
        m["repeat_consistency"] = _rate(len({(r["decision"], r["reason"]) for r in v}) == 1 for v in per_q.values() if len(v) > 1)
        m["bars"] = {k: {"value": m[k], "bar": f"{op} {v}",
                         "pass": None if m[k] is None else (m[k] >= v if op == ">=" else m[k] <= v)} for k, (op, v) in BARS.items()}
        m["by_class_decision_accuracy"] = {c: _rate(s["decision_ok"] for r, s in zip(rs, sc) if Q[r["qid"]]["class"] == c)
                                           for c in sorted({q["class"] for q in questions})}
        m["questions_below_two_of_three"] = sorted(qid for qid, v in per_q.items()
                                                   if sum(score_run(Q[qid], r)["decision_ok"] and score_run(Q[qid], r).get("reason_ok", True)
                                                          for r in v) < 2)
        out["backends"][b] = m
    bs = sorted(by_b)
    if len(bs) == 2:
        qa = {b: {} for b in bs}
        for b in bs:
            for r in by_b[b]:
                qa[b].setdefault(r["qid"], []).append(r)
        both = sorted(set(qa[bs[0]]) & set(qa[bs[1]]))
        agree = [modal(qa[bs[0]][q]) == modal(qa[bs[1]][q]) for q in both]
        out["cross_backend_agreement"] = {"value": _rate(agree), "bar": f">= {AGREEMENT_BAR}", "questions": len(both),
                                          "pass": None if not agree else _rate(agree) >= AGREEMENT_BAR,
                                          "disagreeing": [q for q, a in zip(both, agree) if not a]}
    return out
