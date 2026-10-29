"""Live planner demo for the video and the Pilot page: the five demo requests in eval/demo_requests.json (demo, not scored) through
Planner v2 as it is (frozen) on one backend, plus the tool step on every accept. Prints each request and its card as it comes back, so
a terminal recording of this run is the demo; writes results/demo_runs[_round2]_<backend>.json (cards, decisions, reasons, specs, tool
calls, latency, tokens; never a key). --round picks the round in eval/demo_requests.json (2: the disclosed rewordings written after round 1).
--merge combines the per-backend files of both rounds into results/demo_runs.json.
Keys come from the environment for the run only (MODEL_API_KEY or Muse Code's store for spark-1.3, FIREWORKS_API_KEY, BASETEN_API_KEY).
usage: python scripts/run_demo_requests.py BACKEND [--round N] [--pause S] | --merge"""
import argparse
import datetime
import json
import sys
import textwrap
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx  # noqa: E402

from stillhere import egress, planner, planner_eval as PE  # noqa: E402

LABEL = "demo, not scored"
NAMES = {"spark-1.3": "Muse Spark 1.3 (Meta Model API)", "spark-open": "Muse Spark Open (Fireworks)",
         "spark-open-baseten": "Muse Spark Open (Baseten)"}
_tl = threading.local(); _post = httpx.post


def _rec_post(*a, **k):
    r = _post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


def say(s: str = "", indent: str = "") -> None:
    for line in s.splitlines() or [""]:
        print(textwrap.fill(line, 96, initial_indent=indent, subsequent_indent=indent + "  ") if line else "", flush=True)


def run_one(q: dict, backend: str) -> dict:
    rec = {"id": q["id"], "round": q["round"], "label": LABEL, "text": q["text"], "backend": backend, "model": planner.MODELS[backend], "calls": []}
    base = planner.live_client()

    def client(req):
        _tl.usage = {}; t0 = time.time(); call = {"kind": "tools" if req.tools else "plan"}
        try:
            out = base(req)
            if req.tools:
                call["tool_calls"] = [{"name": c.get("name"), "arguments": c.get("arguments")} for c in out.get("tool_calls", [])]
            else:
                call["codes"] = planner.validate_answer(out.get("content"), planner.load_enums())
            return out
        except Exception as e:
            call["error"] = type(e).__name__; raise
        finally:
            u = _tl.usage or {}
            call.update(s=round(time.time() - t0, 2), prompt_tokens=u.get("prompt_tokens") or 0, completion_tokens=u.get("completion_tokens") or 0)
            rec["calls"].append(call)

    t0 = time.time()
    card = planner.plan(egress.PublicQuestion(text=q["text"]), backend, client)
    rec.update(PE.outcome(card)); rec["trace"] = card.trace; rec["tools"] = []
    if rec["decision"] == "accept":
        results = planner.run_tools(card.spec, backend, client)
        rec["tools"] = [r["tool"] for r in results]
        card = planner.build_card(card.spec, results, planner.load_enums())
    rec["card"] = card.display(); rec["total_s"] = round(time.time() - t0, 2)
    rec["prompt_tokens"] = sum(c["prompt_tokens"] for c in rec["calls"]); rec["completion_tokens"] = sum(c["completion_tokens"] for c in rec["calls"])
    rec["cost_usd"] = round(PE.cost(rec["prompt_tokens"], rec["completion_tokens"]), 5)
    return rec


def out_path(backend: str, rnd: int) -> Path:
    return ROOT / f"results/demo_runs{'' if rnd == 1 else f'_round{rnd}'}_{backend}.json"


def live(backend: str, rnd: int, pause: float) -> None:
    httpx.post = _rec_post
    Q = [q for q in json.load(open(ROOT / "eval/demo_requests.json"))["requests"] if q["round"] == rnd]
    say(f"StillHere Pilot: study planner, live ({LABEL})"); say(f"Planner v2 on {NAMES[backend]}"); say()
    started = datetime.datetime.now(datetime.timezone.utc); runs = []
    for q in Q:
        say(f"Request {q['id']}: {q['text']}"); say(f"... planning on {NAMES[backend]}")
        r = run_one(q, backend); runs.append(r)
        say(r["card"], "  ")
        say(f"  decision: {r['decision']}" + (f" ({r['reason']})" if r["decision"] == "decline" else f"; tools: {', '.join(r['tools']) or 'none'}")
            + f"; {r['total_s']:.1f} s, {len(r['calls'])} call(s), {r['prompt_tokens'] + r['completion_tokens']} tokens")
        say(); time.sleep(pause)
    spend = sum(r["cost_usd"] for r in runs)
    out = {"label": LABEL, "what": f"round {rnd} of the demo requests (eval/demo_requests.json) through Planner v2 (frozen), one backend, live",
           "round": rnd, "backend": backend, "model": planner.MODELS[backend], "effort": planner.EFFORT,
           "started_utc": started.isoformat(timespec="seconds"),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "spend_usd": round(spend, 5), "price_note": "cost at the planner rule's prices (stillhere/planner_eval.py)", "runs": runs}
    json.dump(out, open(out_path(backend, rnd), "w"), indent=1)
    say(f"{len(runs)} requests; {sum(r['decision'] == 'accept' for r in runs)} planned, {sum(r['decision'] == 'decline' for r in runs)} declined; "
        f"about ${spend:.3f}")


def merge() -> None:
    parts = [json.load(open(p)) for rnd in (1, 2) for b in planner.MODELS if (p := out_path(b, rnd)).exists()]
    json.dump({"label": LABEL, "what": "the demo requests (eval/demo_requests.json) through Planner v2 (frozen), live, per backend and round",
               "requests": "eval/demo_requests.json", "rounds": json.load(open(ROOT / "eval/demo_requests.json"))["about"],
               "backends": sorted({p["backend"] for p in parts}),
               "spend_usd": {f"{p['backend']} round {p['round']}": p["spend_usd"] for p in parts}, "runs": [r for p in parts for r in p["runs"]]},
              open(ROOT / "results/demo_runs.json", "w"), indent=1)
    print(f"merged {len(parts)} files, {sum(len(p['runs']) for p in parts)} runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("backend", nargs="?"); ap.add_argument("--pause", type=float, default=0.0); ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else live(A.backend, A.round, A.pause)
