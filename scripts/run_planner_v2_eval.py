"""Runs the Planner v2 evaluation (docs/rules/planner_v2_eval_rule.md; exploratory): the 62 held-out requests (headline) and the 106 v0
requests, three repeats, on spark-1.3 (Meta Model API), spark-open (Fireworks) and spark-open-baseten (Baseten), through planner.plan /
planner.run_tools and the typed egress only, with the product's settings. FIREWORKS_API_KEY and BASETEN_API_KEY must be set in the
environment for the run only. Each backend stops taking runs at its own cap ($15, $15, $10). Scoring is v0's (stillhere/planner_eval.py,
bars unchanged), per backend and per question set, plus Meta-vs-Fireworks modal agreement and the Baseten-vs-Fireworks pairwise bar.
Writes results/planner_v2_eval.json and results/planner_v2_eval_summary.txt; prints counts, rates, timings and costs only.
usage: python scripts/run_planner_v2_eval.py [--repeats 3] [--workers 4] [--limit N]"""
import argparse
import datetime
import importlib.util
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

CAPS = {"spark-1.3": 15.0, "spark-open": 15.0, "spark-open-baseten": 10.0}
SETS = {"heldout": "eval/planner_questions_v2_heldout.json", "seen": "eval/planner_questions.json"}
_tl = threading.local(); _real_post = httpx.post
lock = threading.Lock(); spend = {b: 0.0 for b in CAPS}; skipped = {b: 0 for b in CAPS}


def _recording_post(*a, **k):                     # meta_client calls httpx.post; this only keeps each response's usage block
    r = _real_post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


httpx.post = _recording_post


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


def run_one(qset: str, q: dict, backend: str, rep: int) -> dict | None:
    with lock:
        if spend[backend] >= CAPS[backend]:
            skipped[backend] += 1
            return None
    rec = {"set": qset, "qid": q["id"], "backend": backend, "repeat": rep, "calls": [],
           "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    client = client_for(rec, planner.live_client())
    card = planner.plan(egress.PublicQuestion(text=q["text"]), backend, client)
    rec.update(PE.outcome(card)); rec["trace"] = card.trace; rec["tools"] = []
    if rec["decision"] == "accept":
        rec["tools"] = [r["tool"] for r in planner.run_tools(card.spec, backend, client)]
    return rec


def agreements(runs: list) -> dict:
    """Meta vs Fireworks modal agreement (v0 bar 80%) and the Baseten-vs-Fireworks pairwise bar (planner_eval_baseten_rule.md)."""
    spec = importlib.util.spec_from_file_location("rpeb", ROOT / "scripts/run_planner_eval_baseten.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    by = {b: {} for b in CAPS}
    for r in runs:
        by[r["backend"]].setdefault(r["qid"], []).append(r)
    both = sorted(set(by["spark-1.3"]) & set(by["spark-open"]))
    agree = [PE.modal(by["spark-1.3"][q]) == PE.modal(by["spark-open"][q]) for q in both]
    out = {"meta_vs_fireworks": {"value": round(sum(agree) / len(agree), 4) if agree else None, "questions": len(both), "bar": ">= 0.8",
                                 "pass": bool(agree) and sum(agree) / len(agree) >= PE.AGREEMENT_BAR,
                                 "disagreeing": [q for q, a in zip(both, agree) if not a]}}
    fw = [r for r in runs if r["backend"] == "spark-open"]; bt = [r for r in runs if r["backend"] == "spark-open-baseten"]
    out["baseten_vs_fireworks"] = M.agreement(fw, bt) if fw and bt else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    assert set(planner.MODELS) == set(CAPS), "the planner's backends and the caps differ"
    Q = {s: json.load(open(ROOT / f))["questions"] for s, f in SETS.items()}
    if a.limit:
        Q = {s: qs[:a.limit] for s, qs in Q.items()}
    for qs in Q.values():
        for q in qs:
            egress.PublicQuestion(text=q["text"])                        # every question passes the scan before any call
    t0 = datetime.datetime.now(datetime.timezone.utc); runs = []
    pools = {b: ThreadPoolExecutor(max_workers=a.workers) for b in CAPS}
    futs = [pools[b].submit(run_one, s, q, b, rep) for rep in range(1, a.repeats + 1) for s in ("heldout", "seen") for q in Q[s] for b in CAPS]
    for i, f in enumerate(futs, 1):
        r = f.result()
        if r:
            runs.append(r)
        if i % 150 == 0:
            print(f"{i}/{len(futs)} scheduled runs done; spend " + ", ".join(f"{b} ${v:.3f}" for b, v in spend.items()), flush=True)
    for p in pools.values():
        p.shutdown()
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"runs": runs, "spend_usd": spend, "skipped_runs": skipped}, open(ROOT / "results/planner_v2_eval.raw.json", "w"))   # kept even if scoring fails
    result = {"label": "exploratory", "rule": "docs/rules/planner_v2_eval_rule.md (stillhere-app 6c558d3)", "repeats": a.repeats,
              "models": planner.MODELS, "effort": planner.EFFORT, "baseten_note": "two hosted versions of Muse Spark Open, never a same-weights check",
              "started_utc": t0.isoformat(timespec="seconds"), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "spend_usd": {b: round(v, 4) for b, v in spend.items()}, "caps_usd": CAPS, "skipped_runs": skipped, "sets": {}}
    lines = []
    for s in ("heldout", "seen"):
        rs = [r for r in runs if r["set"] == s]
        summ = PE.summarize(rs, Q[s]); summ.pop("cross_backend_agreement", None)
        summ["agreement"] = agreements(rs)
        result["sets"][s] = {"questions": len(Q[s]), "summary": summ}
        lines.append(f"== {s} ({len(Q[s])} questions)")
        for b, m in summ["backends"].items():
            lines.append(f"{b}: " + ", ".join(f"{k} {v['value']} ({'pass' if v['pass'] else 'FAIL'})" for k, v in m["bars"].items())
                         + f"; cost ${m['cost_usd']:.3f}; latency median {m['latency_median_s']} s")
        ag = summ["agreement"]
        lines.append(f"Meta vs Fireworks modal agreement {ag['meta_vs_fireworks']['value']} (bar 0.8, {'pass' if ag['meta_vs_fireworks']['pass'] else 'FAIL'})")
        bt = ag["baseten_vs_fireworks"]
        if bt:
            lines.append(f"Baseten vs Fireworks {bt['baseten_vs_fireworks_agreement']} vs Fireworks repeat {bt['fireworks_repeat_agreement']} "
                         f"({'pass' if bt['pass'] else 'FAIL'}; two hosted versions)")
    result["runs"] = sorted(runs, key=lambda r: (r["set"], r["backend"], r["qid"], r["repeat"]))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(result, open(ROOT / "results/planner_v2_eval.json", "w"), indent=1)
    (ROOT / "results/planner_v2_eval.raw.json").unlink()
    (ROOT / "results/planner_v2_eval_summary.txt").write_text("Planner v2 evaluation (exploratory)\n" + "\n".join(lines) + "\n"
                                                              + f"spend {result['spend_usd']}; skipped {skipped}\n")
    print("\n".join(lines)); print(f"runs {len(runs)}; spend {result['spend_usd']}; skipped {skipped}")


if __name__ == "__main__":
    main()
