"""R4 (cloud ruling of Oct 31, 08:45 UTC; docs/rules/planner_r4_rule.md): planner first-answer validity on Muse Spark 1.3, v2's
system prompt against R4's candidate (planner_system_r4.txt), through planner.plan and the typed egress only, with the product's
settings. Plan calls only: the after-accept tool call v2's evaluation also made does not touch first-answer validity, so it is not
made here. Sets: dev (v0's 106 requests and v2's 62, the only requests R4 develops on) and r4_heldout (the 62 requests committed in
public f19badd before any prompt or schema change, for the one final measurement of both versions). Scoring is v0's
(stillhere/planner_eval.py, bars unchanged). Spend: every call is costed at the planner rule's prices and counted in
results/planner_r4_spend.json across all R4 runs; no run starts once the total reaches TOTAL_CAP ($13.00, under 10% of the Meta credit
left on Oct 31, $136.43). Writes results/planner_r4_<set>_<version>.json; prints rates and cost only.
usage: python scripts/run_planner_r4_eval.py --set dev|r4_heldout --version v2|r4 [--repeats 1] [--workers 4]"""
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

BACKEND = "spark-1.3"
TOTAL_CAP = 13.00
SETS = {"dev": ["eval/planner_questions.json", "eval/planner_questions_v2_heldout.json"], "r4_heldout": ["eval/planner_questions_r4_heldout.json"]}
SYSTEM = {"v2": "planner_system", "r4": "planner_system_r4"}
SPEND = ROOT / "results/planner_r4_spend.json"
_tl = threading.local(); _real_post = httpx.post; lock = threading.Lock()
state = {"spend": 0.0, "skipped": 0, "total_before": 0.0}


def _recording_post(*a, **k):                     # meta_client calls httpx.post; this only keeps each response's usage block
    r = _real_post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


def questions(qset: str) -> list:
    return [q for f in SETS[qset] for q in json.load(open(ROOT / f))["questions"]]


def check_sets():
    """The dev set never contains an R4 held-out request (by id and by text)."""
    dev, held = questions("dev"), questions("r4_heldout")
    assert not {q["id"] for q in dev} & {q["id"] for q in held}, "an R4 held-out id is in the dev set"
    assert not {q["text"] for q in dev} & {q["text"] for q in held}, "an R4 held-out request is in the dev set"


def client_for(rec: dict, base):
    def client(req):
        _tl.usage = {}; t0 = time.time(); call = {"kind": "plan"}
        try:
            out = base(req)
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
                state["spend"] += PE.cost(call["prompt_tokens"], call["completion_tokens"])
    return client


def run_one(q: dict, version: str, rep: int, base=None) -> dict | None:
    with lock:
        if state["total_before"] + state["spend"] >= TOTAL_CAP:
            state["skipped"] += 1
            return None
    rec = {"qid": q["id"], "backend": BACKEND, "version": version, "repeat": rep, "calls": [],
           "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    card = planner.plan(egress.PublicQuestion(text=q["text"]), BACKEND, client_for(rec, base or planner.live_client()), system=SYSTEM[version])
    rec.update(PE.outcome(card)); rec["trace"] = card.trace
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS)); ap.add_argument("--version", required=True, choices=sorted(SYSTEM))
    ap.add_argument("--repeats", type=int, default=1); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    check_sets()
    Q = questions(a.set)
    for q in Q:
        egress.PublicQuestion(text=q["text"])                            # every request passes the scan before any call
    log = json.load(open(SPEND)) if SPEND.exists() else {"cap_usd": TOTAL_CAP, "runs": []}
    state["total_before"] = sum(r["usd"] for r in log["runs"])
    httpx.post = _recording_post
    t0 = datetime.datetime.now(datetime.timezone.utc)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        runs = [r for r in pool.map(lambda x: run_one(*x), [(q, a.version, rep) for rep in range(1, a.repeats + 1) for q in Q]) if r]
    summ = PE.summarize([{**r, "set": a.set} for r in runs], Q)
    m = summ["backends"][BACKEND]
    out = {"label": "exploratory", "rule": "docs/rules/planner_r4_rule.md", "set": a.set, "version": a.version, "system": SYSTEM[a.version],
           "repeats": a.repeats, "model": planner.MODELS[BACKEND], "effort": planner.EFFORT, "plan_calls_only": True,
           "started_utc": t0.isoformat(timespec="seconds"), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "spend_usd": round(state["spend"], 4), "skipped_runs": state["skipped"], "bars": m["bars"],
           "runs": sorted(runs, key=lambda r: (r["qid"], r["repeat"]))}
    log["runs"].append({"set": a.set, "version": a.version, "repeats": a.repeats, "usd": round(state["spend"], 4), "finished_utc": out["finished_utc"]})
    json.dump(log, open(SPEND, "w"), indent=1)
    json.dump(out, open(ROOT / f"results/planner_r4_{a.set}_{a.version}.json", "w"), indent=1)
    print(f"{a.set} {a.version} x{a.repeats}: " + ", ".join(f"{k} {v['value']} ({'pass' if v['pass'] else 'FAIL'})" for k, v in m["bars"].items())
          + f"; spend ${state['spend']:.3f}; R4 total ${state['total_before'] + state['spend']:.3f} of ${TOTAL_CAP:.2f}; skipped {state['skipped']}")


if __name__ == "__main__":
    main()
