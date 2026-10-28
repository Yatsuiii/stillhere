"""Planner evaluation, Baseten addition (docs/rules/planner_eval_baseten_rule.md; exploratory). Reuses scripts/run_planner_eval.py's own
run_one and client code unchanged, in this process only: the backend spark-open-baseten (egress id baseten/muse-spark-open-1026) is
added to planner.MODELS here and the cap table holds just its $15. Same 106 questions, three repeats, the product's settings.
BASETEN_API_KEY must be set in the environment for the run only.
  run:      python scripts/run_planner_eval_baseten.py run [--workers 3]    -> results/planner_eval_baseten.json
  compare:  python scripts/run_planner_eval_baseten.py compare              -> adds the pass-bar agreement against results/planner_eval.json
Prints counts, rates, timings and spend only."""
import argparse
import datetime
import importlib.util
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stillhere import egress, planner, planner_eval as PE  # noqa: E402

B, CAP = "spark-open-baseten", 15.0
OUT = ROOT / "results/planner_eval_baseten.json"
RELEASE = {"verdict": "two hosted versions",
           "fireworks": "accounts/fireworks/models/muse-spark-open: card names huggingface.co/meta-models/Muse-Spark-Open-816B-A42B, created "
                        "2026-10-14, precision not stated; responses carry no version string",
           "baseten": "meta-models/muse-spark-open-1026 (hackathon id): no model card in Baseten's listing and no version string in responses; "
                      "Baseten's listed meta-models/muse-spark-open (created 2026-10-14) is served at fp4 quantization",
           "checked_utc": "2026-10-28T21:43Z"}
PRICE = {"prompt_per_token": 1.25e-6, "completion_per_token": 4.25e-6, "source": "Baseten's listing for meta-models/muse-spark-open"}


def label(run: dict) -> tuple:
    """The rule's decision label: the in-menu choice for an accept, the decline type for a decline."""
    if run["decision"] == "accept":
        s = run.get("spec") or {}
        return ("accept", s.get("population"), s.get("language"), s.get("stimulus"), tuple(sorted(s.get("brain_areas") or [])),
                tuple(sorted(s.get("conditions") or [])))
    return ("decline", run.get("reason")) if run["decision"] == "decline" else ("unavailable",)


def agreement(fw: list, bt: list) -> dict:
    """Fireworks repeat agreement (pairs of Fireworks repeats per question) and Baseten-vs-Fireworks agreement (all cross pairs)."""
    by = lambda runs: {q: [label(r) for r in runs if r["qid"] == q] for q in {r["qid"] for r in runs}}
    F, T = by(fw), by(bt)
    within = [a == b for q in F for a, b in itertools.combinations(F[q], 2)]
    cross = [a == b for q in F if q in T for a in T[q] for b in F[q]]
    w, c = sum(within) / len(within), sum(cross) / len(cross)
    return {"fireworks_repeat_agreement": round(w, 4), "baseten_vs_fireworks_agreement": round(c, 4), "pairs": {"within": len(within), "cross": len(cross)},
            "bar": "cross >= within - 0.05", "pass": c >= w - 0.05}


def run(workers: int):
    spec = importlib.util.spec_from_file_location("rpe", ROOT / "scripts/run_planner_eval.py")
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    planner.MODELS = {**planner.MODELS, B: "baseten/muse-spark-open-1026"}
    PE.CAPS = {B: CAP}; R.spend.clear(); R.spend[B] = 0.0
    Q = json.load(open(ROOT / "eval/planner_questions.json"))["questions"]
    for q in Q:
        egress.PublicQuestion(text=q["text"])
    t0 = datetime.datetime.now(datetime.timezone.utc)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        runs = [r for r in ex.map(lambda a: R.run_one(*a), [(q, B, rep) for rep in (1, 2, 3) for q in Q]) if r]
    try:
        summary = PE.summarize(runs, Q)
    except Exception as e:                                            # the shared summary expects two backends; per-run records stand
        summary = {"error": type(e).__name__}
    out = {"label": "exploratory", "rule": "docs/rules/planner_eval_baseten_rule.md (stillhere-app d465108)", "backend": B,
           "model": "baseten/muse-spark-open-1026", "effort": planner.EFFORT, "release_check": RELEASE, "price": PRICE, "questions": len(Q), "repeats": 3,
           "started_utc": t0.isoformat(timespec="seconds"), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "stopped": R.stop["why"], "spend_usd": round(R.spend[B], 4), "cap_usd": CAP, "summary": summary,
           "runs": sorted(runs, key=lambda r: (r["qid"], r["repeat"]))}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"runs {len(runs)}; stopped: {R.stop['why']}; spend ${R.spend[B]:.3f}")


def compare():
    main, bt = json.load(open(ROOT / "results/planner_eval.json")), json.load(open(OUT))
    fw = [r for r in main["runs"] if r["backend"] == "spark-open"]
    bt["agreement"] = agreement(fw, bt["runs"])
    json.dump(bt, open(OUT, "w"), indent=1)
    a = bt["agreement"]
    print(f"Fireworks repeat agreement {a['fireworks_repeat_agreement']:.3f}; Baseten-vs-Fireworks {a['baseten_vs_fireworks_agreement']:.3f} "
          f"({a['pairs']}); pass {a['pass']} (reported as {RELEASE['verdict']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "compare")); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    run(a.workers) if a.cmd == "run" else compare()
