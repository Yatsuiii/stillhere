"""Live smoke of the analysis desk (Oct 27): at most 3 Muse Spark calls, SYNTHETIC Gate 1 fixture only, allowlisted public text only.
Prints per call the step, seconds, token counts and tool-call count, then each card's source, badge and trace codes; no request or
response content. usage: python scripts/smoke_desk.py"""
import json
from pathlib import Path

import httpx

from stillhere import desk, meta_client
from stillhere.frozen import Frozen

CAP = 3
calls = []


class CapReached(LookupError):
    pass


def capped(req):
    if len(calls) >= CAP:
        raise CapReached("live call cap reached")
    calls.append("route" if req.tools else "answer")
    try:
        out = meta_client.call(req, timeout=60.0)
    except httpx.HTTPStatusError as e:                                        # the API's own error message, not our content
        print(f"call {len(calls)} ({calls[-1]}): HTTP {e.response.status_code} {e.response.text[:300]}")
        raise
    last = json.loads(meta_client.USAGE_LOG.read_text().splitlines()[-1])
    print(f"call {len(calls)} ({calls[-1]}): {last['seconds']} s, prompt {last['prompt_tokens']}, completion {last['completion_tokens']}, "
          f"tool calls {last['tool_calls']}")
    return out


fz = Frozen.load(Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic_gate1_pass.json")
d = desk.Desk(fz, client=capped, mode="LIVE")
for q in ("q_language_fades", "q_lag_plus_2"):
    card = d.ask(desk.public(q))
    print(f"{q}: source {card.source} | badge {card.badge} | trace {card.trace}")
