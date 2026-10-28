"""Runs the Planner v0 evaluation (docs/rules/planner_eval_rule.md; exploratory): every question in eval/planner_questions.json, three
repeats, on spark-1.3 (Meta Model API) and spark-open (Fireworks), through planner.plan / planner.run_tools and the typed egress only,
with the product's own settings. FIREWORKS_API_KEY must be set in the environment for the run only (the Meta key is found as
meta_client finds it). Records per run: question id, backend, repeat, decision, reason, spec (enum values), trace, model calls (kind,
seconds, tokens, the first answer's check codes), tool names. Stops scheduling as soon as either backend reaches its spend cap.
Writes results/planner_eval.json and prints counts, rates, timings and costs only.
usage: python scripts/run_planner_eval.py [--repeats 3] [--workers 3] [--limit N]"""
import argparse
import datetime
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx  # noqa: E402

from stillhere import egress, planner, planner_eval as PE  # noqa: E402

_tl = threading.local()
_real_post = httpx.post


def _recording_post(*a, **k):                     # meta_client calls httpx.post; this only keeps each response's usage block
    r = _real_post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


httpx.post = _recording_post
lock = threading.Lock(); spend = {b: 0.0 for b in PE.CAPS}; stop = {"why": None}


def client_for(rec: dict, base):
    def client(req):
        _tl.usage = {}; t0 = time.time(); call = {"kind": "tools" if req.tools else "plan"}
        try:
            out = base(req)
            if not req.tools:
                call["codes"] = planner.validate_answer(out.get("content"), planner.load_enums())
            return out
        except Exception as e:
            call["error"] = type(e).__name__
            raise
        finally:
            u = getattr(_tl, "usage", {}) or {}
            call.update(s=round(time.time() - t0, 2), prompt_tokens=u.get("prompt_tokens") or 0, completion_tokens=u.get("completion_tokens") or 0)
            rec["calls"].append(call)
            with lock:
                spend[rec["backend"]] += PE.cost(call["prompt_tokens"], call["completion_tokens"])
    return client


def run_one(q: dict, backend: str, rep: int) -> dict | None:
    with lock:
        if stop["why"] or PE.over_cap(spend):
            stop["why"] = stop["why"] or f"spend cap reached on {PE.over_cap(spend)}"
            return None
    rec = {"qid": q["id"], "backend": backend, "repeat": rep, "calls": [], "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    client = client_for(rec, planner.live_client())
    card = planner.plan(egress.PublicQuestion(text=q["text"]), backend, client)
    rec.update(PE.outcome(card)); rec["trace"] = card.trace; rec["tools"] = []
    if rec["decision"] == "accept":
        rec["tools"] = [r["tool"] for r in planner.run_tools(card.spec, backend, client)]
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    Q = json.load(open(ROOT / "eval/planner_questions.json"))["questions"]
    Q = Q[:a.limit] if a.limit else Q
    for q in Q:
        egress.PublicQuestion(text=q["text"])                            # every question passes the scan before any call
    t0 = datetime.datetime.now(datetime.timezone.utc); runs, done = [], [0]
    pools = {b: ThreadPoolExecutor(max_workers=a.workers) for b in planner.MODELS}
    futs = [pools[b].submit(run_one, q, b, rep) for rep in range(1, a.repeats + 1) for q in Q for b in planner.MODELS]
    for f in futs:
        r = f.result()
        if r:
            runs.append(r); done[0] += 1
            if done[0] % 60 == 0:
                print(f"{done[0]} runs; spend " + ", ".join(f"{b} ${v:.3f}" for b, v in spend.items()), flush=True)
    for p in pools.values():
        p.shutdown()
    summary = PE.summarize(runs, Q)
    out = {"label": "exploratory", "rule": "docs/rules/planner_eval_rule.md (stillhere-app 309cd84)", "questions": len(Q), "repeats": a.repeats,
           "models": planner.MODELS, "effort": planner.EFFORT, "started_utc": t0.isoformat(timespec="seconds"),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "stopped": stop["why"], "spend_usd": {b: round(v, 4) for b, v in spend.items()}, "caps_usd": PE.CAPS, "summary": summary,
           "runs": sorted(runs, key=lambda r: (r["backend"], r["qid"], r["repeat"]))}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results/planner_eval.json", "w"), indent=1)
    print(f"runs {len(runs)}; stopped: {stop['why']}; spend " + ", ".join(f"{b} ${v:.3f}" for b, v in spend.items()))
    for b, m in summary["backends"].items():
        print(f"== {b}: {m['runs']} runs, {m['calls']} calls, latency median {m['latency_median_s']} s p90 {m['latency_p90_s']} s, cost ${m['cost_usd']}")
        for k, v in m["bars"].items():
            print(f"   {k:26s} {v['value']} (bar {v['bar']}) {'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a'}")
        print(f"   tool step ok {m['tool_step_ok_rate']}; repeat consistency {m['repeat_consistency']}; below 2 of 3: {m['questions_below_two_of_three']}")
    ag = summary.get("cross_backend_agreement", {})
    print(f"cross-backend agreement {ag.get('value')} over {ag.get('questions')} questions (bar {ag.get('bar')}) "
          f"{'PASS' if ag.get('pass') else 'FAIL'}; disagreeing {ag.get('disagreeing')}")


if __name__ == "__main__":
    main()
