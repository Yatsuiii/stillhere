"""The 3 example requests shown as chips above the live page's input box (cloud rulings of Oct 29, 18:15 IST, item 2), each run once
live on planner v2.1 with Spark 1.3 (Meta Model API) and Spark Open (Fireworks) through the typed egress. Labelled "demo, not scored":
v2.1 is not re-measured and the 62 held-out and 106 seen requests are not rerun. A chip is kept only if both backends accept it; a
declined one is replaced by the next candidate, and every try is kept in the file (runs under an earlier v2.1 prompt under
"superseded"). Keys come from the environment for the run only
(MODEL_API_KEY or Muse Code's store for spark-1.3, FIREWORKS_API_KEY). Writes results/planner_v21_examples.json.
usage: python scripts/run_planner_v21_examples.py"""
import datetime
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx  # noqa: E402

from stillhere import egress, planner, planner_eval as PE, planner_v21 as P  # noqa: E402

CANDIDATES = [
    "Plan an exploratory stock TRIBE forecast for healthy English-speaking adults: LibriVox Alice chapter 1, intact versus sentences "
    "shuffled, in the language network.",
    "Plan an exploratory stock TRIBE forecast of the hearing areas for healthy English-speaking adults: LibriVox Alice chapter 2, "
    "intact versus played backwards.",
    "Plan an exploratory stock TRIBE forecast for healthy English-speaking adults: LibriVox Alice chapter 3, intact versus sentences "
    "shuffled with seed 7, in the language network and hearing areas.",
    "Plan an exploratory stock TRIBE forecast for healthy English-speaking adults listening to LibriVox Alice chapter 2 with its "
    "sentences shuffled, compared with the intact chapter, in the language network.",
    "Plan an exploratory stock TRIBE forecast of the language network for healthy English-speaking adults: LibriVox Alice chapter 3, "
    "intact versus played backwards.",
]
BACKENDS = ("spark-1.3", "spark-open")
_tl = threading.local(); _post = httpx.post


def _rec_post(*a, **k):
    r = _post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


httpx.post = _rec_post


def run(text: str, backend: str) -> dict:
    base, calls = planner.live_client(), []

    def client(req):
        _tl.usage = {}; t0 = time.time()
        try:
            return base(req)
        finally:
            u = _tl.usage or {}
            calls.append({"s": round(time.time() - t0, 2), "prompt_tokens": u.get("prompt_tokens") or 0,
                          "completion_tokens": u.get("completion_tokens") or 0, "tools_offered": bool(req.tools)})
    card = P.plan(egress.PublicQuestion(text=text), backend, client)
    rt = getattr(card, "route", {"kind": "unavailable"})
    return {"backend": backend, "decision": "accept" if rt["kind"] in ("live", "evidence") else "decline", "route": rt,
            "spec": card.spec, "card": card.text, "trace": card.trace, "calls": calls,
            "cost_usd": round(sum(PE.cost(c["prompt_tokens"], c["completion_tokens"]) for c in calls), 5)}


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    sha = hashlib.sha256((ROOT / "stillhere/public_text/planner_v21_system.txt").read_bytes()).hexdigest()
    out = ROOT / "results/planner_v21_examples.json"
    superseded = []                                    # earlier runs under another v2.1 prompt stay in the file, with their tries
    if out.exists():
        old = json.loads(out.read_text())
        superseded = old.get("superseded", []) + [{k: old.get(k) for k in ("prompt_sha256", "started_utc", "spend_usd", "tries")}]
    tries, kept = [], []
    for text in CANDIDATES:
        if len(kept) == 3:
            break
        egress.PublicQuestion(text=text)
        runs = [run(text, b) for b in BACKENDS]
        ok = all(r["decision"] == "accept" and r["route"]["kind"] == "live" for r in runs)
        tries.append({"text": text, "kept": ok, "runs": runs})
        print(f"try {len(tries)}: " + ", ".join(f"{r['backend']} {r['decision']} ({r['route']['kind']})" for r in runs)
              + (" -> kept" if ok else " -> replaced"), flush=True)
        if ok:
            kept.append({"text": text, "runs": runs})
    spend = {b: round(sum(r["cost_usd"] for t in tries for r in t["runs"] if r["backend"] == b), 5) for b in BACKENDS}
    json.dump({"label": "demo, not scored", "badge": P.BADGE, "version": P.VERSION, "prompt_sha256": sha,
               "what": "the 3 example chips above the live page's input box: each run once on planner v2.1, kept only if both backends "
                       "accept it; every try is listed, kept or not",
               "started_utc": t0.isoformat(timespec="seconds"),
               "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "spend_usd": spend, "examples": kept, "tries": tries, "superseded": superseded},
              open(out, "w"), indent=1)
    print(f"kept {len(kept)} of {len(tries)} tries; spend " + ", ".join(f"{b} ${v:.4f}" for b, v in spend.items()))


if __name__ == "__main__":
    main()
