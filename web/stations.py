"""The live stations of the hosted Pilot page: PLAN, BUILD, AUDIT, FORECAST, FILM, and the EVIDENCE the static page already shows.
Each live call goes through the typed egress, a spend reservation (web/ledger.py) and the wording guard (web/guard.py); when a call
must not or cannot run (no key, a cap, the session limit, an API error) the station answers with a recorded run, labelled so.
Typed text is used for the one model call it is for and is never stored, logged or put in a receipt."""
import base64
import contextlib
import datetime
import functools
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from stillhere import egress, manipulation_check as MC, meta_client, planner, planner_eval as PE, planner_v21 as P
from stillhere.story_tags import TAGS
from web import guard, jobs, ledger
from web.store import StoreError, get_store, shared as store_shared

REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "web/replay"
PROVIDER = {"spark-1.3": "meta", "spark-open": "fireworks"}
HOSTS = {"spark-1.3": (("meta", "spark-1.3"),),                            # (provider, planner backend) in the order tried: Baseten is
         "spark-open": (("fireworks", "spark-open"), ("baseten", "spark-open-baseten"))}   # Spark Open's fallback host (plan of 19:20 UTC, 1c)
KEY_ENV = {"fireworks": "FIREWORKS_API_KEY", "baseten": "BASETEN_API_KEY"}
TIMEOUT_S = {"meta": 60.0, "fireworks": 25.0, "baseten": 60.0}   # the page's plan and audit calls: a stalled Fireworks call settles as
                                                                  # api_error and falls through to Baseten (ruling of Oct 30, 07:40 UTC, e)
RESERVE = {"plan": 0.05, "audit": 0.03, "transcribe": 0.10, "sam": 0.50}   # worst case $ per call; Voice Transcribe keeps its
RESHUFFLE_SEEDS = (20261028, 7, 1, 2, 3, 4)                                # reservation (never settled), SAM settles to SAM_USD_PER_FRAME;
                                                                           # 7 is the third example chip's seed
# SAM 3.1's price per one-frame call (env SAM_USD_PER_FRAME; docs/deploy_checklist.md: a one-frame call bills a 50-frame minimum,
# $0.01): FILM settles each completed frame to it. Set, a frame reserves twice its price ($0.06 a run at $0.01), so FILM fits a share
# with little left (cloud review of Oct 31, 12:20 UTC, item 1: $0.50 a frame sent FILM to its recorded run with $1 of a share left);
# unset, each frame reserves and keeps $0.50.
SAM_MIN_USD = 0.01   # one SAM call bills a 50-frame minimum: $0.01 at Meta's $0.20 per 1,000 frames (docs/deploy_checklist.md)


def sam_price(raw):
    """SAM_USD_PER_FRAME as a price: a finite number of at least SAM_MIN_USD, else None (unset). A value set by mistake would make FILM
    reserve too little: 0, negative or not a number $0, passing spent caps (refuter's review of Oct 31, item 8); 0.0002, Meta's list
    price per frame, $0.0006 a run against a real $0.03 (second refuter's review, item D). So it counts as unset, FILM reserves and
    keeps $0.50 a frame, and the log says so."""
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        v = None
    if v is None or not math.isfinite(v) or v < SAM_MIN_USD:
        print(f"stillhere: SAM_USD_PER_FRAME is not a number of at least {SAM_MIN_USD} (one call's 50-frame minimum), so FILM "
              "reserves $0.50 a frame", file=sys.stderr, flush=True)
        return None
    return v


SAM_USD_PER_FRAME = sam_price(os.environ.get("SAM_USD_PER_FRAME"))
FILM_N = 3                                                                 # frames per FILM run
# FILM's clock (cloud review of Oct 31, 12:20 UTC, item 7): each SAM call gets 60 s, not sam.segment's 120 s default, and no frame
# starts once 150 s have passed, so the run ends with the api_error line well inside Vercel Hobby's 300 s and its finally always
# settles (a platform kill at 300 s would leave the whole reservation booked). The 60 s bounds each phase and read, not a call, so
# every SAM call also gets FILM's end, FILM_END_S after the first frame starts (sam.segment's deadline; refuter's review of Oct 31,
# item 3), and FILM waits for a call only until that end (_sam_frame: second refuter's review, item A), so the finally settles by
# about 180 s plus one frame's edit (a few seconds), whatever the stream does.
SAM_TIMEOUT_S, FILM_DEADLINE_S, FILM_END_S = 60.0, 150, 180
TRANSCRIBE_TIMEOUT_S = 60.0   # BUILD's Voice Transcribe call: the menu's clips are 40 s at most; the client's 300 s default is Vercel
                              # Hobby's whole function limit, so a stall would end in a platform 504 instead of the recorded run
FILM_FRAMES = REPO / "web/film/frames.json"
RECORDED = "recorded run"
NO_RECORDED_CHECK = "No recorded check for this clip and condition."
ND = guard.DISCLAIMER
OLD_BUG = {"audio": "assets/conditions/alice_ch01_75s_word_scramble.wav", "results": "results/conditions_manipulation_check.json",
           "commits": ("2002a8d", "64f3c99")}


class Bad(ValueError):
    """A malformed page request (400)."""


# ---- usage capture: the token counts of this thread's model calls, for the spend ledger; and each chat call's output bound -------
_tl = threading.local()
_post = httpx.post


def _max_tokens(left_usd: float, body: dict) -> int:
    """The output tokens a chat call may write so that it cannot cost more than what is left of its reservation: every prompt token
    is at least one byte of what the request sends (its messages, tools and answer schema), plus a margin for the chat template, so
    their byte count bounds the prompt's price. Below one token, no call (ValueError)."""
    sent = sum(len(json.dumps(body.get(f) or "").encode()) for f in ("messages", "tools", "response_format")) + 64 * len(body["messages"])
    n = int((left_usd - sent * PE.PRICE_IN) / PE.PRICE_OUT)
    if n < 1:
        raise ValueError("the reservation cannot cover this call")
    return n


NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError, httpx.UnsupportedProtocol)   # failures before the provider


def _recording_post(*a, **k):
    on = getattr(_tl, "on", False)
    body, budget = k.get("json"), getattr(_tl, "budget", None) if on else None
    if budget is not None and isinstance(body, dict) and "messages" in body:       # a chat call under a reservation: bound its output
        k["json"] = {**body, "max_tokens": _max_tokens(budget - chat_cost(_tl.usage), body)}
    try:
        r = _post(*a, **k)
    except NEVER_SENT:
        raise
    except Exception:
        if on:
            _tl.reached = _tl.unknown = True                   # it may have reached the provider: its cost is unknown
        raise
    if on:
        try:
            u = r.json().get("usage")
        except Exception:
            u = None
        _tl.reached = True
        if u:
            _tl.usage.append(u)
        else:
            _tl.unknown = True
    return r


httpx.post = _recording_post


@contextlib.contextmanager
def capture(budget: float = None):
    """The usage of this thread's model calls; with `budget` (the call's reservation), every chat call inside carries max_tokens so
    that all of them together cannot cost more."""
    _tl.on, _tl.usage, _tl.budget, _tl.reached, _tl.unknown = True, [], budget, False, False
    try:
        yield _tl.usage
    finally:
        _tl.on, _tl.budget = False, None


def chat_cost(usage: list) -> float:
    return sum(PE.cost(u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0) for u in usage)


def spent(usage: list, reserved: float) -> float:
    """What the last capture's chat calls settle to: their token cost, or the whole reservation when one reached the provider and
    reported no usage (it may have been billed); a call that never connected costs nothing (fourth review, item 6)."""
    return reserved if getattr(_tl, "unknown", False) else chat_cost(usage)


def reached() -> bool:
    """Whether the last capture's calls reached the provider at all (a non-chat call then keeps its reservation when it fails)."""
    return getattr(_tl, "reached", False)


def _store_safe(fallback):
    """A station whose store fails (an outage, a spent quota, a timeout) answers with `fallback` (its recorded run, called with the
    station's own arguments) and the line "Live runs are paused", logged by the failure's type only (fourth review, item 4)."""
    def deco(fn):
        @functools.wraps(fn)
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except StoreError as e:
                print(f"stillhere: store error ({e}) in {fn.__name__}", file=sys.stderr, flush=True)
                return fallback(*a, **k)
        return run
    return deco


meta_credit_ended = ledger.meta_credit_ended
META_CREDIT_END_UTC = ledger.META_CREDIT_END_UTC


def live_ok(provider: str) -> bool:
    if os.environ.get("STILLHERE_LIVE", "1") == "0":
        return False
    if os.environ.get("VERCEL") and not store_shared():        # serverless instances do not share memory: no shared ledger, no spend
        return False
    try:
        return bool(meta_client._key()) if provider == "meta" else bool(os.environ.get(KEY_ENV.get(provider, "")))
    except Exception:
        return False


def first_live(backend: str):
    """The first host for this backend whose key is set, or Refused("no_key")."""
    for prov, b in HOSTS[backend]:
        if live_ok(prov):
            return prov, b
    raise ledger.Refused("no_key")


def affordable_host(backend: str, run_id: str, usd: float) -> None:
    """Refused unless some live host for this backend has `usd` left under every cap and share (ledger.affordable, read only)."""
    first = None
    for prov, _ in HOSTS[backend]:
        if live_ok(prov):
            try:
                return ledger.affordable(run_id, prov, usd)
            except ledger.Refused as e:
                first = first or e
    raise first or ledger.Refused("no_key")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def receipt(rec: dict):
    """The receipt's id, or None when the store cannot keep it (the live answer is still shown, without its receipt link)."""
    rid = secrets.token_hex(8)
    try:
        get_store().set(f"receipt:{rid}", json.dumps({**rec, "utc": _now()}), ttl=14 * 86400)
    except StoreError as e:
        print(f"stillhere: store error ({e}) in receipt", file=sys.stderr, flush=True); return None
    return rid


def get_receipt(rid: str):
    try:
        v = get_store().get(f"receipt:{rid}") if rid and rid.isalnum() else None
    except StoreError:
        return None
    return json.loads(v) if v else None


def _load(rel: str):
    p = REPO / rel
    return json.loads(p.read_text()) if p.is_file() else None


SAVED_SHOWN = "a saved forecast is shown where one exists"          # what FORECAST shows instead of a live forecast


def why_line(code: str, shown: str = "this station shows its recorded run") -> str:
    """The page's line for a refusal code; `shown` ends the lines that say what the station shows instead (FORECAST: SAVED_SHOWN)."""
    return {"session_limit": "This session has used its 5 live runs for this hour.", "spend_cap": "The daily live budget is used up.",
            "no_key": "Live calls are off on this server.", "api_error": "The live call failed, so this station shows its recorded run.", "no_run": "Start with a plan first.",
            "station_used": "This run already used this station.", "provider_not_offered": "That model is not offered live.",
            "recorded_plan": "This plan is a recorded run, so this station shows its recording.",
            "ip_limit": f"This network address has used its {ledger.RUNS_PER_HOUR_PER_ADDRESS} live runs for this hour.",
            "station_cap": "The live runs of this station for this hour are used up across all visitors, so {shown}.",
            "not_the_plan": "This run's plan names another clip or condition, so {shown}.",
            "seed_not_offered": "This seed is not one of the offered seeds, so {shown}.",   # cloud review of Oct 31, 12:20 UTC, item 10
            "plan_declined": "This run's plan was declined, so {shown}.",
            "session_daily": "This session has used its share of the live budget until 00:00 UTC, so {shown}.",
            "ip_daily": "This network address has used its share of the live budget until 00:00 UTC, so {shown}.",
            "shared_limit": (f"Without a visitor cookie, this browser shares this network address's {ledger.RUNS_PER_HOUR} live runs an "
                             "hour, and they are used up."),
            "shared_daily": ("Without a visitor cookie, this browser shares this network address's part of the live budget, used up "
                             "until 00:00 UTC, so {shown}."),
            "worker_offline": f"The laptop worker is offline (it runs live {hours_et()}), so {{shown}}.",
            "store_down": "Live runs are paused, so {shown}.",
            "ip_station": "This network address has used its part of this station's live runs for this hour, so {shown}.",
            "slow_down": "This network address sent too many requests in a short time, so {shown}.",
            "post_credit_cap": ("The event's Meta API credit has ended and the $20 set aside after it is spent, so this station shows "
                                "its recorded run.")}.get(
        code, "Live calls are off right now.").format(shown=shown)


# ---- PLAN -----------------------------------------------------------------------------------------------------------------------
SCOREBOARD = "Planner v2 · 62 unseen requests · 2.3% vs 7.6% / 6.8% (5% bar)"


def menu() -> dict:
    """The live-menu line above the input box, the 3 example chips Fork P ran on both backends, the scoreboard with its badge, and the
    stories the screen kept off the menu (title and failed bar only, story_menu_rule.md section 6) and the stories held off it (title and
    the ruling's reason only)."""
    m = _load("stillhere/live_menu.json") or {}
    ex = _load("results/planner_v21_story_chips.json") or _load("results/planner_v21_examples.json") or {}   # stories, else Alice
    summ = m.get("screen_summary") or {}
    return {"text": P.menu_text(), "chips": [e["text"] for e in ex.get("examples", [])], "badge": P.BADGE, "scoreboard": SCOREBOARD,
            "clips": [{"id": c["id"], "title": c["title"], "seconds": c.get("dur_s")} for c in m.get("clips", [])],
            "default_seed": m.get("default_seed"), "reshuffle_seeds": list(RESHUFFLE_SEEDS),
            "failed": [{"title": f["title"], "bar": str(f["bar"])} for f in summ.get("failed", [])],
            "held": [{"title": h["title"], "reason": h["reason"]} for h in summ.get("held", [])]}   # title and reason only


# ---- the story card (docs/rules/story_menu_rule.md section 6, word for word; data in docs/story_menu_data.md) ---------------------
STORY_LEAD = "{title}, read by {reader}: one of {n} screened public stories · 1 voice heard · {k} sentences · notes: "
STORY_NONE = "none of the listed tags found"
STORY_TAIL = ". Content notes come from Muse Spark Open reading the transcript. Not a full review; listen before you use a story."


def _mss(t: float) -> str:
    s = int(t); return f"{s // 60}:{s % 60:02d}"


def story_card(clip_id: str):
    """The PLAN and BUILD card for a screened story on the live menu, or None for a clip with no screen data (alice_ch01-03): the
    rule's line with {title}, {reader}, {n}, {k} and the kept notes as "{tag} ({m:ss})" in time order, each linking to the archive.org
    file at #t={seconds}."""
    m = _load("stillhere/live_menu.json") or {}
    clip = next((c for c in m.get("clips", []) if c["id"] == clip_id), None)
    sc = (clip or {}).get("screen")
    if not sc:
        return None
    notes = sorted((x for x in sc.get("notes", []) if x.get("tag") in TAGS), key=lambda x: x["t"])
    items = [{"label": f"{x['tag']} ({_mss(x['t'])})", "href": f"{sc['archive_url']}#t={int(x['t'])}"} for x in notes]
    lead = STORY_LEAD.format(title=clip["title"], reader=clip["reader"], n=m["screen_summary"]["n_on_menu"], k=sc["sentences"])
    text = lead + (", ".join(i["label"] for i in items) if items else STORY_NONE) + STORY_TAIL
    if guard.clean(text) != text:                              # a cheap backstop (cloud ruling of Oct 30, 04:20 UTC, item 5): a title,
        return {"lead": guard.TEMPLATE, "notes": [], "none": "", "tail": "", "text": guard.TEMPLATE}   # reader or tag with a banned word
    return {"lead": lead, "notes": items, "none": None if items else STORY_NONE, "tail": STORY_TAIL, "text": text}


def card_view(card) -> dict:
    """What the page shows of a v2.1 card: its own plain-words text (no spec code shown raw), the route, and the guard on every line."""
    rt = getattr(card, "route", None) or {"kind": "unavailable"}
    unavailable = not card.spec
    view = {"decision": "unavailable" if unavailable else "decline" if rt["kind"] == "decline" else "accept",
            "lines": [guard.clean(l) for l in card.text.split("\n") if l.strip()], "route": {"kind": rt["kind"]}}
    if rt["kind"] == "live":
        view["route"].update({k: rt[k] for k in ("clip_id", "condition", "seed", "defaults")})
        view["story"] = story_card(rt["clip_id"])
    elif rt["kind"] == "evidence":                                                   # ruling (2): off the live list -> its evidence panel
        view["route"].update(panel=rt.get("panel"), line=guard.clean(rt.get("line") or ""))
    return view


def recorded_plan(why: str) -> dict:
    rp = _load("web/replay/plan.json") or {}
    runs = [{**r, "card": {**r["card"], "story": story_card(r["card"]["route"].get("clip_id"))}} if (r["card"].get("route") or {}).get(
        "kind") == "live" else r for r in rp.get("runs", [])]
    return {"mode": "recorded", "label": RECORDED, "why": why_line(why), "runs": runs}


@_store_safe(lambda *a, **k: recorded_plan("store_down"))
def plan(text: str, backend: str, sid: str, ip: str = None) -> dict:
    text = (text or "").strip()
    if not text or len(text) > 280:
        raise Bad("text")
    if backend not in PROVIDER:
        raise Bad("backend")
    try:
        q = egress.PublicQuestion(text=text)
    except ValueError:
        return {"mode": "refused", "line": "This request can't be sent to the planner. StillHere plans research studies in healthy "
                                           "volunteers with public stimuli."}
    try:
        with ledger.Claimed() as t:                              # the visitor's counts, then PLAN's shared hour (ledger: Order); any error
            first_live(backend)                                # before the paid call gives them back. first_live first, so a recorded
            run_id = ledger.start_run(sid, ip=ip, claimed=t)     # run never uses up a visitor's live runs
            addr = ledger.use(run_id, sid, "plan", claimed=t)
            affordable_host(backend, run_id, RESERVE["plan"]); ledger.station_hour("plan", claimed=t, addr=addr)
            t0, card, why, used = time.time(), None, "no_key", None
            for prov, b in HOSTS[backend]:                     # Spark Open: Fireworks, then Baseten when Fireworks is off, capped or fails
                if not live_ok(prov):
                    continue
                try:
                    res = ledger.reserve(prov, RESERVE["plan"], run_id)
                except ledger.Refused as e:
                    why = e.code; continue
                t.paid, usage = True, []
                try:
                    with capture(RESERVE["plan"]) as usage:
                        c = P.plan(q, b, planner.live_client(TIMEOUT_S[prov]))
                except Exception:
                    c = None
                usd = spent(usage, RESERVE["plan"]); ledger.settle(res, usd)
                if c is not None and c.spec:                   # an empty spec is "planner unavailable": try the next host
                    card, used = c, (prov, b); break
                why = "api_error"
            if card is None:
                if not t.paid:                                 # no host could take the call: nothing was spent
                    t.give_back()
                return recorded_plan(why)
    except ledger.Refused as e:
        return recorded_plan(e.code)
    view = card_view(card)
    get_store().set(f"runspec:{run_id}", json.dumps({"spec": card.spec or {}, "route": getattr(card, "route", None)}), ttl=ledger.RUN_TTL_S)
    rid = receipt({"station": "plan", "model": P.MODELS[used[1]], "host": used[0], "planner": P.VERSION, "spec_sha256": hashlib.sha256(
        json.dumps(card.spec or {}, sort_keys=True).encode()).hexdigest(), "seconds": round(time.time() - t0, 2), "usd": round(usd, 5)})
    return {"mode": "live", "run_id": run_id, "backend": backend, "card": view, "badge": P.BADGE, "receipt_id": rid, "nd": ND}


# ---- AUDIT ----------------------------------------------------------------------------------------------------------------------
AUDIT_SCHEMA = {"type": "object", "properties": {"notes": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                "required": ["notes"], "additionalProperties": False}


def recorded_audit(why: str, line: str = None) -> dict:
    return {**(_load("web/replay/audit.json") or {"notes": []}), "mode": "recorded", "label": f"{AUDIT_LABEL} ({RECORDED})",
            "why": line or why_line(why)}


AUDIT_LABEL = "Advisory check, not measured"


@_store_safe(lambda *a, **k: recorded_audit("store_down"))
def audit(run_id: str, sid: str) -> dict:
    """Spark Open's advisory check of the plan: never blocks anything, never measured."""
    spec = (json.loads(get_store().get(f"runspec:{run_id}") or "null") or {}).get("spec") if run_id else None
    if not spec:                                               # a recorded plan (no live run): the recorded audit
        return recorded_audit(None, "This plan is a recorded run, so its check is too.")
    if (spec.get("decline_reason") or "none") != "none":
        return {"mode": "none", "label": AUDIT_LABEL, "line": "No accepted plan to check."}
    enums = {k: [v] if isinstance(v, str) else list(v) for k, v in spec.items()
             if k in ("population", "language", "stimulus", "analysis", "brain_areas", "conditions", "clip_id", "condition") and v}
    try:
        with ledger.Claimed() as t:
            first_live("spark-open")
            addr = ledger.use(run_id, sid, "audit", claimed=t)
            affordable_host("spark-open", run_id, RESERVE["audit"]); ledger.station_hour("audit", claimed=t, addr=addr)
            t0, out, why, model = time.time(), None, "no_key", None
            for prov, b in HOSTS["spark-open"]:                # Fireworks, then Baseten
                if not live_ok(prov):
                    continue
                try:
                    res = ledger.reserve(prov, RESERVE["audit"], run_id)
                except ledger.Refused as e:
                    why = e.code; continue
                model, usage, t.paid = P.MODELS[b], [], True
                try:
                    with capture(RESERVE["audit"]) as usage:
                        out = meta_client.chat(egress.Request(model=model, system=[planner.public("audit_system")],
                                                              user=[egress.TableEnums(enums=enums)], schema_name="audit_notes",
                                                              json_schema=AUDIT_SCHEMA), timeout=TIMEOUT_S[prov])
                except Exception:
                    out, why = None, "api_error"
                usd = spent(usage, RESERVE["audit"]); ledger.settle(res, usd)
                if out is not None:
                    break
            if out is None:
                if not t.paid:
                    t.give_back()
                return recorded_audit(why)
    except ledger.Refused as e:
        return recorded_audit(e.code)
    notes = [guard.clean(n) for n in (out.get("notes") or [])[:3] if isinstance(n, str)]
    rid = receipt({"station": "audit", "model": model, "seconds": round(time.time() - t0, 2), "usd": round(usd, 5)})
    return {"mode": "live", "label": AUDIT_LABEL, "notes": notes, "receipt_id": rid}


# ---- BUILD ----------------------------------------------------------------------------------------------------------------------
def planned(run_id: str, clip_id: str, condition: str) -> None:
    """BUILD and FORECAST run only the clip and condition this run's live plan routed to, as the page asks: Refused('no_run') for a
    run with no stored plan, 'plan_declined' for a declined one, 'not_the_plan' for a plan off the live list or another clip or
    condition (a hand-made request). Read before the run's use is counted, so a refused request costs the run nothing."""
    raw = get_store().get(f"runspec:{run_id}") if run_id else None
    if not raw:
        raise ledger.Refused("no_run")
    rs = json.loads(raw); rt = rs.get("route") or {}
    if ((rs.get("spec") or {}).get("decline_reason") or "none") != "none" or rt.get("kind") == "decline":
        raise ledger.Refused("plan_declined")
    if rt.get("kind") != "live" or (rt.get("clip_id"), rt.get("condition")) != (clip_id, condition):
        raise ledger.Refused("not_the_plan")


def _recipe_mod():
    import importlib
    return importlib.import_module("worker.recipe")


def checklist(condition: str, chk: dict) -> list:
    """The Voice Transcribe checklist: each item with its number and the rule it is held to (stillhere/manipulation_check.py RULES)."""
    r, pc = chk["rule"], lambda x: f"{round(100 * x)}%"
    if condition == "reversed":
        return [{"item": "Words heard on reversed speech", "value": str(chk["n_words_heard"]), "ok": None, "rule": "no word check"}]
    items = [{"item": "Words heard (word match)", "value": pc(chk["recall"]), "ok": chk["recall"] >= r["recall_min"],
              "rule": f"at least {pc(r['recall_min'])}"}]
    if condition == "intact":
        items.append({"item": "Order kept", "value": pc(chk["order_vs_original"]), "ok": chk["order_vs_original"] >= r["order_vs_original_min"],
                      "rule": f"at least {pc(r['order_vs_original_min'])}"})
    elif condition == "sentence_scramble":
        items += [{"item": "Sentences heard in the new order", "value": pc(chk["order_vs_expected"]),
                   "ok": chk["order_vs_expected"] >= r["order_vs_expected_min"], "rule": f"at least {pc(r['order_vs_expected_min'])}"},
                  {"item": "Order changed from the original", "value": pc(chk["order_vs_expected"] - chk["order_vs_original"]),
                   "ok": chk["order_vs_expected"] - chk["order_vs_original"] >= r["expected_minus_original_min"],
                   "rule": f"new order beats the old by at least {pc(r['expected_minus_original_min'])}"}]
    elif condition == "word_scramble":
        items.append({"item": "Order changed from the original", "value": pc(chk["order_vs_original"]),
                      "ok": chk["order_vs_original"] <= r["order_vs_original_max"], "rule": f"at most {pc(r['order_vs_original_max'])} in the old order"})
    return items


def old_bug() -> dict:
    """The real broken word scramble, from the committed audio and its committed check: no model call, numbers copied."""
    c = (_load(OLD_BUG["results"]) or {})["conditions"]["word_scramble"]
    return {"mode": "recorded", "label": "the old bug, from the repository history", "condition": "word_scramble", "pass": False,
            "checklist": checklist("word_scramble", c), "audio_url": "/media/old_bug.wav?v=" + hashlib.sha256(
                (REPO / OLD_BUG["audio"]).read_bytes()).hexdigest()[:8],                    # named by content, so a day's cache is safe
            "caption": (f"Our first word scramble (commit {OLD_BUG['commits'][0]}, Oct 28) kept only {round(100 * c['recall'])}% of its "
                        f"words when Voice Transcribe read it back (commit {OLD_BUG['commits'][1]}), so it was caught before scanning.")}


def recorded_build(clip_id: str, condition: str, why: str) -> dict:
    """Only a recording of this clip and condition, never another one's."""
    got = (_load("web/replay/build.json") or {}).get(f"{clip_id}:{condition}")
    if not got:
        return {"mode": "recorded", "label": RECORDED, "why": why_line(why), "checklist": [], "line": NO_RECORDED_CHECK,
                "story": story_card(clip_id)}
    return {**got, "mode": "recorded", "label": f"{RECORDED}, seed {got.get('seed')}", "why": why_line(why), "story": story_card(clip_id)}


@_store_safe(lambda run_id, sid, clip_id, condition, seed: recorded_build(clip_id, condition, "store_down"))
def build(run_id: str, sid: str, clip_id: str, condition: str, seed: int) -> dict:
    try:
        r = jobs.recipe(clip_id, condition, seed)
    except ValueError as e:
        raise Bad(str(e))
    if condition in ("sentence_scramble", "word_scramble") and r["seed"] not in RESHUFFLE_SEEDS:
        raise Bad("seed_not_offered")
    try:
        with ledger.Claimed() as t:
            if not live_ok("meta"):
                raise ledger.Refused("no_key")
            planned(run_id, clip_id, condition)
            addr = ledger.use(run_id, sid, "build", max_uses=1 + len(RESHUFFLE_SEEDS), claimed=t)
            ledger.affordable(run_id, "meta", RESERVE["transcribe"])
            R = _recipe_mod(); idx = R.load_index()
            built, orig = R.build_condition(r, idx), R.build_condition({"clip_id": clip_id, "condition": "intact", "seed": 0}, idx)
            wav = R.wav_bytes(built.audio, built.sr)
            t0 = time.time()
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f"{clip_id}_{condition}.wav"; p.write_bytes(wav)
                try:
                    audio = egress.PublicAudio(path=str(p))     # a build that is not on the allowlist is refused: "build differs"
                except ValueError:
                    t.give_back(); return {"mode": "refused", "line": "This build differs from the listed public builds, so it was not sent."}
                ledger.station_hour("build", claimed=t, addr=addr)
                res = ledger.reserve("meta", RESERVE["transcribe"], run_id)   # no published price: the reservation stands once it answers
                t.paid = True
                try:
                    with capture():
                        out = meta_client.transcribe(audio, mode="DIARIZATION", timeout=TRANSCRIBE_TIMEOUT_S)
                except Exception:                              # never connected: not counted; reached the provider: it may be billed
                    ledger.settle(res, RESERVE["transcribe"] if reached() else 0.0)
                    return recorded_build(clip_id, condition, "api_error")
    except ledger.Refused as e:
        return recorded_build(clip_id, condition, "recorded_plan" if e.code == "no_run" else e.code)
    chk = MC.check_condition(condition, out.get("transcript") or "", [w["word"] for w in built.words], [w["word"] for w in orig.words])
    items = checklist(condition, chk)
    rid = receipt({"station": "build", "model": meta_client.ASR_MODEL, "input_sha256": hashlib.sha256(wav).hexdigest(),
                   "recipe": r, "seconds": round(time.time() - t0, 2), "usd": None, "usd_reserved": RESERVE["transcribe"]})
    return {"mode": "live", "condition": condition, "seed": r["seed"], "pass": chk["pass"], "checklist": items, "story": story_card(clip_id),
            "audio_b64": base64.b64encode(wav).decode(), "receipt_id": rid}


# ---- FORECAST -------------------------------------------------------------------------------------------------------------------
REVERSED_LINE = "0 words reach the text lane (model input, not a brain result)"
WORD_RATE_LINE = "Flat by construction: a word-rate model cannot see order."
GRAY_LINE = "No public scan of this clip. Forecast, not a scan."
SCANNED_LINE = "A public scan of this chapter exists ({dataset}, {accession}); not compared here. Forecast, not a scan."
CHIP_SHORT = "Forecast only · audio-only, laptop"


def end_line(clip_id: str) -> str:
    """The live end line for one menu clip, from its public_scan field (cloud plan of Oct 29, 19:20 UTC, 1a): "No public scan" only for
    a clip checked and found in no public story-fMRI dataset; a clip a public dataset played names that dataset."""
    clip = next((c for c in (_load("stillhere/live_menu.json") or {}).get("clips", []) if c["id"] == clip_id), None)
    if clip is None or "public_scan" not in clip:
        raise KeyError(f"{clip_id}: not on the live menu, or never checked for a public scan")
    ps = clip["public_scan"]
    return GRAY_LINE if ps is None else SCANNED_LINE.format(dataset=ps["dataset"], accession=ps["accession"])


def chip(clip_id: str) -> dict:
    return {"short": CHIP_SHORT, "long": end_line(clip_id)}


def saved_forecast(clip_id: str, condition: str, seed: int, why: str) -> dict:
    """FORECAST's fallback: the saved forecast for this clip and condition, with the line saying why no live one runs."""
    r = jobs.recipe(clip_id, condition, seed)
    return {**_forecast_base(r), "state": "offline", "why": why, "fallback": jobs.fallback(r)}


def _forecast_base(r: dict) -> dict:
    return {"chip": chip(r["clip_id"]), "end_line": end_line(r["clip_id"]), "word_rate_line": WORD_RATE_LINE,
            "reversed_line": REVERSED_LINE if r["condition"] == "reversed" else None, "hours_et": hours_et()}


@_store_safe(lambda run_id, sid, clip_id, condition, seed: saved_forecast(clip_id, condition, seed,
                                                                           why_line("store_down", shown=SAVED_SHOWN)))
def forecast(run_id: str, sid: str, clip_id: str, condition: str, seed: int) -> dict:
    try:
        r = jobs.recipe(clip_id, condition, seed)
    except ValueError as e:
        raise Bad(str(e))
    base = _forecast_base(r)
    try:
        with ledger.Claimed() as t:                              # the run's use and the hours stand only for a job that reaches the queue
            if condition in ("sentence_scramble", "word_scramble") and r["seed"] not in RESHUFFLE_SEEDS:
                raise ledger.Refused("seed_not_offered")       # BUILD's seeds only; the saved run, not a 400 (the page keeps its built state)
            planned(run_id, r["clip_id"], r["condition"])
            addr = ledger.use(run_id, sid, "forecast", max_uses=4, claimed=t)
            out = jobs.submit(r, sid, claimed=t, addr=addr)
            if out["state"] != "queued":                       # cached, busy, offline or a lost queue race: no laptop time
                t.give_back()
    except ledger.Refused as e:                                # no live run (a recorded plan), used up or past the hour: the saved run
        return {**base, "state": "offline", "why": "Cached forecast only." if e.code == "no_run" else why_line(e.code, shown=SAVED_SHOWN),
                "fallback": jobs.fallback(r)}
    if out["state"] == "offline":
        out = {**out, "why": why_line("worker_offline", shown=SAVED_SHOWN)}
    return {**base, **out}


def slowed(station: str, b: dict) -> dict:
    """A station's answer to a POST past its address's token bucket (web/app.py): its recorded run with the why line, as for any other
    refusal, made with no store command (cloud review of Oct 31, 12:20 UTC, item 6): FORECAST shows the committed saved forecast only,
    and the old bug is a recording already."""
    if station == "forecast":
        try:
            r = jobs.recipe(b.get("clip_id"), b.get("condition"), b.get("seed"))
        except ValueError as e:
            raise Bad(str(e))
        return {**_forecast_base(r), "state": "offline", "why": why_line("slow_down", shown=SAVED_SHOWN),
                "fallback": jobs.fallback(r, from_store=False)}
    if station == "build":
        return old_bug() if b.get("old_bug") else recorded_build(b.get("clip_id"), b.get("condition"), "slow_down")
    return {"plan": recorded_plan, "audit": recorded_audit, "film": recorded_film}[station]("slow_down")


def hours_et() -> str:
    return os.environ.get("WORKER_HOURS_ET", "10:00-14:00 ET, Nov 4-9")


SAVED_LINE = "A saved forecast is shown whenever one exists for your choice. New seeds run live on the laptop, {hours}."   # ruling 8


# ---- FILM -----------------------------------------------------------------------------------------------------------------------
JPEG_QUALITY = 90


def _sam_frame(sam, clip: str, q, end: float):
    """One SAM call, waited for at most until `end` (time.time() seconds): (answer or None, its error or None, whether it reached the
    provider), or TimeoutError while it still runs. httpx fixes a stream's read timeout when the body starts, so bytes trickled within
    one line each get a full read timeout and sam.segment's per-line check cannot end the call; it runs in a daemon thread, which ends
    on its own, and FILM stops waiting at its end (second refuter's review of Oct 31, item A). capture() is per thread, so it runs in
    there."""
    box = {}

    def run():
        with capture():
            try:
                box["seg"] = sam.segment(clip, q, timeout=SAM_TIMEOUT_S, deadline=end)
            except Exception as e:
                box["err"] = e
            box["hit"] = reached()
    t = threading.Thread(target=run, name="stillhere-sam", daemon=True); t.start(); t.join(max(end - time.time(), 0.0))
    if t.is_alive():
        raise TimeoutError("the SAM call is still running at FILM's end")
    return box.get("seg"), box.get("err"), box.get("hit", True)


def _b64jpg(rgb) -> str:
    """One FILM image as the page shows it: half the release frame (640x267), JPEG at quality 90 through opencv, which live FILM
    already needs. About 0.6 MB for a run's nine images where PNG sent 4.7 MB (cloud review of Oct 31, 12:20 UTC, optional item); the
    recorded run's images are JPEG too."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb[::2, ::2].copy(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("the frame could not be encoded")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


CREDIT = "Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study."


def recorded_film(why: str) -> dict:
    return {**(_load("web/replay/film.json") or {"frames": []}), "mode": "recorded", "label": RECORDED, "why": why_line(why), "credit": CREDIT}


@_store_safe(lambda *a, **k: recorded_film("store_down"))
def film(run_id: str, sid: str, concept: str, ip: str = None) -> dict:
    concept = (concept or "").strip()
    if not concept or len(concept) > 40:
        raise Bad("concept")
    try:
        q = egress.PublicQuestion(text=concept)
    except ValueError:
        return {"mode": "refused", "credit": CREDIT, "line": "This concept can't be sent."}
    try:
        if not live_ok("meta"):
            raise ledger.Refused("no_key")
        import cv2
        from stillhere import face_edit as FE
        import meta_sam_parser  # noqa: F401  (Meta's SAM License; a dependency, not included)
    except ledger.Refused as e:
        return recorded_film(e.code)
    except ImportError:
        return recorded_film("no_key")
    from web import sam
    per = RESERVE["sam"] if SAM_USD_PER_FRAME is None else 2 * SAM_USD_PER_FRAME   # priced: twice the price (item 1, above)
    try:
        with ledger.Claimed() as t:                              # every frame reserved before the first call (third review, item 7)
            if not run_id:
                run_id = ledger.start_run(sid, ip=ip, claimed=t)
            addr = ledger.use(run_id, sid, "film", claimed=t)
            ledger.affordable(run_id, "meta", FILM_N * per); ledger.station_hour("film", claimed=t, addr=addr)
            res = ledger.reserve("meta", FILM_N * per, run_id)
            t.paid = True
    except ledger.Refused as e:
        return recorded_film(e.code)
    frames, t0, priced, held = [], time.time(), [], 0.0
    try:
        for f in json.loads(FILM_FRAMES.read_text())["frames"][:FILM_N]:
            if time.time() - t0 >= FILM_DEADLINE_S:            # out of time: this frame and the rest are freed below
                return recorded_film("api_error")
            try:
                seg, err, hit = _sam_frame(sam, str(REPO / f["clip"]), q, t0 + FILM_END_S)
            except TimeoutError:                               # still running at FILM's end: it may be billed, so it stands
                held += per; return recorded_film("api_error")
            if err is not None:                                # this frame failed; later ones are not called. Never connected: not
                held += per if hit else 0.0                    # counted; reached the provider (a parse error after it answered): kept
                return recorded_film("api_error")
            if seg.get("status") == "response.completed" and SAM_USD_PER_FRAME is not None:
                priced.append(SAM_USD_PER_FRAME)
            else:                                              # no price, or an answer that did not complete (it may still be
                held += per                                    # billed): the frame's reservation stands
            try:                                               # an object only in the letterbox bars maps to an empty mask: dropped;
                frame = cv2.cvtColor(cv2.imread(str(REPO / f["release_jpg"])), cv2.COLOR_BGR2RGB)   # any other error editing a
                masks = [m for m in FE.masks_for({"objects": seg["objects"]}) if m.any()] if seg["objects"] else []   # frame gives
                if not masks:                                  # the recorded run, never a 500 (fifth review, item 4)
                    frames.append({"id": f["id"], "found": False, "line": "no mask found", "intact": _b64jpg(frame)}); continue
                img, px, offs = FE.edit(frame, masks, skip_unplaceable=True)
                shown = {v: _b64jpg(img[v]) for v in FE.VERSIONS}   # encoded inside the try: a failed encode is the recorded run too
            except Exception:
                return recorded_film("api_error")
            frames.append({"id": f["id"], "found": True, "n_masks": len(masks), "controls": sum(o is not None for o in offs), **shown})
    finally:                                                   # on every way out: each frame called at its price or reservation,
        ledger.settle(res, sum(priced) + held)                 # the frames not called freed
    rid = receipt({"station": "film", "model": "sam-3.1", "frames": [f["id"] for f in frames], "seconds": round(time.time() - t0, 2),
                   "usd": round(sum(priced), 5) if priced else None, "usd_reserved": round(held, 2)})
    return {"mode": "live", "frames": frames, "credit": CREDIT, "receipt_id": rid,
            "caption": "SAM 3.1 masked what you typed; StillHere filled it with the surrounding colour, and filled a same-size patch "
                       "elsewhere in the frame the same way as a control."}


# ---- EVIDENCE -------------------------------------------------------------------------------------------------------------------
def evidence() -> dict:
    """The static page's evidence panels (one builder, scripts/build_pilot_page.py), the ghost-line data where it exists, the order-map
    outcome, and which tile sits beside a live gray line: the notthefall pair only if D20 passed, else the LPP ghost line (ruling 3)."""
    bp, sl = guard._bp, {k: _load(v) for k, v in guard._bp.SLOTS.items()}
    html = (bp.lpp_html(_load("results/section8.json"), _load("results/section8_baselines.json"), _load("results/ghostline_lpp_s8.json"))
            + bp.paranoia_html(_load(bp.PARANOIA[0]), _load(bp.PARANOIA[1])) + bp.alice_html(_load(bp.ALICE[0]))
            + bp.wall_html(sl["story_wall"])
            + bp.d20_html(sl["d20"]) + bp.film_result_html(sl["face_map"], sl["film_score"], _load("results/ghostline_film.json")))
    d20 = bp.d20_status(sl["d20"])
    om = _load("results/order_map_gonogo.json") or {}
    return {"html": html, "lpp_ghost": _load("results/ghostline_lpp_s8.json"), "film_ghost": _load("results/ghostline_film.json"),
            "d20": d20, "pair": "d20" if d20["status"] == "pass" else "lpp",
            "pair_label": "Real listeners, different story." if d20["status"] == "pass" else "Real listeners, held-out story",
            "order_map": {"verdict": om.get("verdict"), "consequence": om.get("consequence")}, "film_status": bp.film_status(sl["face_map"]),
            "d20_chip": f'{bp.GHOST_CHIPS["d20"]["short"]}: {bp.GHOST_CHIPS["d20"]["long"]}', "prereg_url": bp.PREREG_URL,
            "prereg_note": bp.PREREG_NOTE}
