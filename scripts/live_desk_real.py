"""Live check of the analysis desk on the REAL Gate 1 results (results/gate1_real.json): at most CAP Muse Spark calls at low reasoning
effort, allowlisted menu questions only, every request typed by the egress wrapper and logged by meta_client (tokens, no content).
Answers are recorded to replay/desk_real.jsonl, so the stage default (REPLAY) reproduces them without a call.
usage: python scripts/live_desk_real.py [CAP] [EFFORT]"""
import json
import sys
from pathlib import Path

import httpx

from stillhere import desk, meta_client
from stillhere.frozen import Frozen

ROOT = Path(__file__).resolve().parents[1]
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
EFFORT = sys.argv[2] if len(sys.argv) > 2 else "low"
calls = []


class CapReached(LookupError):
    pass


def capped(req):
    if len(calls) >= CAP:
        raise CapReached("live call cap reached")
    calls.append("route" if req.tools else "answer")
    try:
        out = meta_client.call(req, timeout=90.0, effort=EFFORT or None)
    except httpx.HTTPStatusError as e:                                        # the API's own error message, not our content
        print(f"call {len(calls)} ({calls[-1]}): HTTP {e.response.status_code} {e.response.text[:300]}")
        raise
    last = json.loads(meta_client.USAGE_LOG.read_text().splitlines()[-1])
    print(f"call {len(calls)} ({calls[-1]}): {last['seconds']} s, prompt {last['prompt_tokens']}, completion {last['completion_tokens']}, "
          f"tool calls {last['tool_calls']}")
    return out


fz = Frozen.load(ROOT / "results/gate1_real.json")
d = desk.Desk(fz, client=desk.RecordingClient(capped, ROOT / "replay/desk_real.jsonl"), mode="LIVE")
for q in ("q_language_fades", "q_lag_plus_2"):
    card = d.ask(desk.public(q))
    print(f"--- {q}: source {card.source} | badge {card.badge} | trace {card.trace}\n{card.text}")
print(f"live calls: {len(calls)} ({', '.join(calls)})")
