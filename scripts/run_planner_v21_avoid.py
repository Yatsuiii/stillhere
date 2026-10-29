"""One recorded run of planner v2.1 with avoid_tags (cloud plan of Oct 29, 19:20 UTC, 3b): a request that asks to avoid a story note, run
once live on the default planner (Spark 1.3, Meta), through Fork P's runner (scripts/run_planner_v21_examples.py run()). Spend goes
through the web ledger's caps (web/ledger.py reserve and settle; the in-process store unless Upstash is configured). Labelled "demo,
not scored". Keys from the environment for the run only, never printed. Writes results/planner_v21_avoid_tags_run.json.
usage: python scripts/run_planner_v21_avoid.py"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_planner_v21_examples as RX  # noqa: E402

from stillhere import egress, planner_v21 as P  # noqa: E402
from web import ledger  # noqa: E402

TEXT = ("Plan an exploratory stock TRIBE forecast for healthy English-speaking adults: a LibriVox story with no violence, intact "
        "versus sentences shuffled, in the language network.")
RESERVE_USD = 0.05


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    egress.PublicQuestion(text=TEXT)
    res = ledger.reserve("meta", RESERVE_USD)
    run = RX.run(TEXT, "spark-1.3")
    ledger.settle(res, run["cost_usd"])
    enums = P.load_enums()
    rt = run["route"]
    checks = {"avoid_tags": run["spec"].get("avoid_tags"),
              "route_kind": rt["kind"],
              "clip_passes_filter": rt["kind"] == "live" and P.passes(rt["clip_id"], rt.get("avoid_tags") or [], enums["clip_tags"]),
              "clip_card_tags": enums["clip_tags"].get(rt.get("clip_id")) if rt["kind"] == "live" else None}
    json.dump({"label": "demo, not scored", "badge": P.BADGE, "version": P.VERSION,
               "prompt_sha256": hashlib.sha256((ROOT / "stillhere/public_text/planner_v21_system.txt").read_bytes()).hexdigest(),
               "what": "one live run of planner v2.1 with avoid_tags on the default planner, recorded as the replay; code filters the menu",
               "started_utc": t0.isoformat(timespec="seconds"),
               "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "text": TEXT, "spend_usd": {"spark-1.3": run["cost_usd"]}, "checks": checks, "run": run},
              open(ROOT / "results/planner_v21_avoid_tags_run.json", "w"), indent=1)
    print(f"{run['decision']} {rt['kind']} clip {rt.get('clip_id')} avoid {checks['avoid_tags']} passes {checks['clip_passes_filter']}; "
          f"trace {run['trace']}; cost ${run['cost_usd']:.5f}")


if __name__ == "__main__":
    main()
