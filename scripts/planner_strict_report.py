"""The extra numbers the project thread asked for on a planner evaluation (ruling of Oct 29, 06:00 UTC, item 4), from a results file:
per backend and question set, (1) decision accuracy with a fallback (rule-made decline) or no answer (planner unavailable) counted as a
wrong decision; (2) false accepts as a count over expected-decline runs, with the exact (Clopper-Pearson) two-sided 95% interval, whose
upper end is the reported bound (also when the count is 0); (3) declines made by the model vs by the planner's fixed rules.
usage: python scripts/planner_strict_report.py RESULTS_JSON QUESTIONS_JSON [SET] [--json OUT]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scipy.stats import beta  # noqa: E402


def upper95(k: int, n: int) -> float:
    """Exact two-sided 95% upper bound for k successes out of n (one-sided 97.5%)."""
    return 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))


def report(runs: list, questions: list) -> dict:
    Q = {q["id"]: q for q in questions}; out = {}
    for b in sorted({r["backend"] for r in runs}):
        rs = [r for r in runs if r["backend"] == b and r["qid"] in Q]
        strict = sum(r["decision"] == Q[r["qid"]]["expected"]["decision"] and not r.get("rule_made") and r["decision"] != "unavailable" for r in rs)
        dec_exp = [r for r in rs if Q[r["qid"]]["expected"]["decision"] == "decline"]
        fa = sum(r["decision"] == "accept" for r in dec_exp)
        declines = [r for r in rs if r["decision"] == "decline"]
        out[b] = {"runs": len(rs), "strict_decision_accuracy": round(strict / len(rs), 4) if rs else None,
                  "false_accepts": fa, "expected_decline_runs": len(dec_exp), "false_accept_rate": round(fa / len(dec_exp), 4) if dec_exp else None,
                  "false_accept_upper95": round(upper95(fa, len(dec_exp)), 4) if dec_exp else None,
                  "declines_by_model": sum(not r.get("rule_made") for r in declines), "declines_by_rules": sum(bool(r.get("rule_made")) for r in declines),
                  "unavailable": sum(r["decision"] == "unavailable" for r in rs)}
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res, qfile = json.load(open(args[0])), json.load(open(args[1]))["questions"]
    runs = res["runs"] if len(args) < 3 else [r for r in res["runs"] if r.get("set") == args[2]]
    rep = report(runs, qfile)
    for b, m in rep.items():
        print(f"{b}: strict decision {m['strict_decision_accuracy']}; false accepts {m['false_accepts']}/{m['expected_decline_runs']} "
              f"({m['false_accept_rate']}, exact 95% upper {m['false_accept_upper95']}); declines by model {m['declines_by_model']}, "
              f"by rules {m['declines_by_rules']}; unavailable {m['unavailable']}")
    if "--json" in sys.argv:
        json.dump(rep, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
