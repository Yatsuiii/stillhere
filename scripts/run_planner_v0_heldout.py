"""Planner v0 on the v2 held-out set (ruling of Oct 29, 06:00 UTC, item 4: "v0 on the same held-out set"; exploratory). Runs v0's
planner code from a git worktree (V0_ROOT: the current tree with stillhere/planner.py and its system prompt restored from the commit
before Planner v2, 67a9670) on the 62 held-out requests, three repeats, on all three backends (the Baseten backend is added in this
process only, as the v0 Baseten addition did). Same egress, settings, record format and scoring as the v2 run; v2 itself is not touched.
FIREWORKS_API_KEY and BASETEN_API_KEY in the environment for the run only. Cap $5 per backend. Writes results/planner_v0_heldout.json.
usage: python scripts/run_planner_v0_heldout.py V0_ROOT [--workers 4]"""
import argparse
import datetime
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("v0_root"); ap.add_argument("--workers", type=int, default=4); A = ap.parse_args()
sys.path.insert(0, str(Path(A.v0_root).resolve()))
import httpx  # noqa: E402

from stillhere import egress, planner, planner_eval as PE  # noqa: E402

assert Path(planner.__file__).resolve().is_relative_to(Path(A.v0_root).resolve()), "planner was not imported from the v0 worktree"
assert "analysis" not in planner.study_schema(planner.load_enums())["properties"], "this is not v0 (it has an analysis field)"
planner.MODELS = {**planner.MODELS, "spark-open-baseten": "baseten/muse-spark-open-1026"}
CAP = 5.0
_tl = threading.local(); _post = httpx.post; lock = threading.Lock(); spend = {b: 0.0 for b in planner.MODELS}


def _rec_post(*a, **k):
    r = _post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


httpx.post = _rec_post


def run_one(q, b, rep):
    with lock:
        if spend[b] >= CAP:
            return None
    rec = {"qid": q["id"], "backend": b, "repeat": rep, "set": "heldout", "calls": []}
    base = planner.live_client()

    def client(req):
        _tl.usage = {}; t0 = time.time(); call = {"kind": "tools" if req.tools else "plan"}
        try:
            out = base(req)
            if not req.tools:
                call["codes"] = planner.validate_answer(out.get("content"), planner.load_enums())
            return out
        except Exception as e:
            call["error"] = type(e).__name__; raise
        finally:
            u = _tl.usage or {}
            call.update(s=round(time.time() - t0, 2), prompt_tokens=u.get("prompt_tokens") or 0, completion_tokens=u.get("completion_tokens") or 0)
            rec["calls"].append(call)
            with lock:
                spend[b] += PE.cost(call["prompt_tokens"], call["completion_tokens"])
    card = planner.plan(egress.PublicQuestion(text=q["text"]), b, client)
    rec.update(PE.outcome(card)); rec["trace"] = card.trace; rec["tools"] = []
    if rec["decision"] == "accept":
        rec["tools"] = [r["tool"] for r in planner.run_tools(card.spec, b, client)]
    return rec


Q = json.load(open(ROOT / "eval/planner_questions_v2_heldout.json"))["questions"]
for q in Q:
    egress.PublicQuestion(text=q["text"])
t0 = datetime.datetime.now(datetime.timezone.utc)
with ThreadPoolExecutor(max_workers=A.workers * 3) as ex:
    runs = [r for r in ex.map(lambda a: run_one(*a), [(q, b, rep) for rep in (1, 2, 3) for q in Q for b in planner.MODELS]) if r]
summ = PE.summarize(runs, Q); summ.pop("cross_backend_agreement", None)
json.dump({"label": "exploratory", "what": "Planner v0 (planner.py and planner_system.txt from 67a9670) on the v2 held-out set",
           "started_utc": t0.isoformat(timespec="seconds"), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "spend_usd": {b: round(v, 4) for b, v in spend.items()}, "cap_usd": CAP, "summary": summ, "runs": runs},
          open(ROOT / "results/planner_v0_heldout.json", "w"), indent=1)
print(f"runs {len(runs)}; spend " + ", ".join(f"{b} ${v:.3f}" for b, v in spend.items()))
