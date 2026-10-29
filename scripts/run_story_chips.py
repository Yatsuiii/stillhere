"""Moves the live page's 3 example chips to the screened stories (cloud plan of Oct 29, 19:20 UTC, 2f): each candidate runs once live on
planner v2.1 with Spark 1.3 and Spark Open (Fireworks), through Fork P's runner (scripts/run_planner_v21_examples.py run()), and is kept
only if both backends route it to the recipe it names and that recipe's forecast is in data/forecast_cache.json and its BUILD replay
passed. Labelled "demo, not scored". Every try is kept. alice_ch02 stays as the backup in Fork P's file. Keys from the environment for
the run only. Writes results/planner_v21_story_chips.json; the page reads its chips from there.
usage: python scripts/run_story_chips.py"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_planner_v21_examples as RX  # noqa: E402

from stillhere import egress, planner_v21 as P  # noqa: E402
from web import jobs  # noqa: E402

BASE = "Plan an exploratory stock TRIBE forecast"
CANDIDATES = [   # (request, the recipe it names: clip, condition, seed as normalised)
    (f"{BASE} for healthy English-speaking adults: The Black Cat, intact versus sentences shuffled, in the language network.",
     ("story_black_cat", "sentence_scramble", 20261028)),
    (f"{BASE} of the hearing areas for healthy English-speaking adults: The Tell-tale Heart, intact versus played backwards.",
     ("story_telltale_heart", "reversed", 0)),
    (f"{BASE} for healthy English-speaking adults: The Diamond Necklace, intact versus sentences shuffled with seed 20261028, in the "
     "language network and hearing areas.", ("story_diamond_necklace", "sentence_scramble", 20261028)),
    (f"{BASE} for healthy English-speaking adults listening to The Black Cat with its sentences shuffled, compared with the intact story, "
     "in the language network.", ("story_black_cat", "sentence_scramble", 20261028)),
    (f"{BASE} of the hearing areas for healthy English-speaking adults: The Diamond Necklace, intact versus played backwards.",
     ("story_diamond_necklace", "reversed", 0)),
]


def ready(want) -> bool:
    r = jobs.recipe(*want); build = json.loads((ROOT / "web/replay/build.json").read_text()).get(f"{want[0]}:{want[1]}") or {}
    return jobs.cached(jobs.cache_key(r)) is not None and build.get("pass") is not False


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    tries, kept, clips = [], [], set()
    for text, want in CANDIDATES:
        if len(kept) == 3:
            break
        if want[0] in clips or not ready(want):
            continue
        egress.PublicQuestion(text=text)
        runs = [RX.run(text, b) for b in RX.BACKENDS]
        got = [jobs.recipe(r["route"]["clip_id"], r["route"]["condition"], r["route"]["seed"]) if r["route"]["kind"] == "live" else None
               for r in runs]
        ok = all(g == jobs.recipe(*want) for g in got)
        tries.append({"text": text, "wants": dict(zip(("clip_id", "condition", "seed"), want)), "kept": ok, "runs": runs})
        print(f"try {len(tries)}: {want[0]} {want[1]}: " + ", ".join(f"{r['backend']} {r['route']['kind']} "
              f"{r['route'].get('clip_id')} {r['route'].get('condition')}" for r in runs) + (" -> kept" if ok else " -> not kept"), flush=True)
        if ok:
            kept.append({"text": text, "runs": runs}); clips.add(want[0])
    spend = {b: round(sum(r["cost_usd"] for t in tries for r in t["runs"] if r["backend"] == b), 5) for b in RX.BACKENDS}
    json.dump({"label": "demo, not scored", "badge": P.BADGE, "version": P.VERSION,
               "what": "the 3 example chips above the live page's input box, moved to the screened stories: each run once on planner "
                       "v2.1, kept only if both backends route it to the cached recipe it names; every try is listed",
               "started_utc": t0.isoformat(timespec="seconds"),
               "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "spend_usd": spend, "examples": kept, "tries": tries}, open(ROOT / "results/planner_v21_story_chips.json", "w"), indent=1)
    print(f"kept {len(kept)} of {len(tries)} tries; spend " + ", ".join(f"{b} ${v:.4f}" for b, v in spend.items()))


if __name__ == "__main__":
    main()
