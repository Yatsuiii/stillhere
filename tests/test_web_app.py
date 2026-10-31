"""The hosted Pilot app (web/): no key ever reaches the browser; every live station falls back to its recorded run with the model APIs
blocked; the wording guard replaces banned model lines; spend caps and the 5-runs-per-hour limit hold; the jobs API follows
docs/jobs_api.md with a fake worker (token, events, result checks, expiry, cache); the evidence endings follow the rulings of Oct 29,
18:15 IST; and the page lays out at 360 px with no horizontal scroll."""
import datetime
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]


def public_files(text_only=False):
    """Every file the page serves from web/public, subfolders included (fonts/); text_only: the ones that hold wording."""
    return [f for f in sorted((REPO / "web/public").rglob("*"))
            if f.is_file() and (not text_only or f.suffix in (".html", ".js", ".css", ".txt"))]


SECRETS ={"MODEL_API_KEY": "meta-SECRET-0123456789abcdef", "FIREWORKS_API_KEY": "fw-SECRET-0123456789abcdef",
           "STILLHERE_WORKER_TOKEN": "worker-SECRET-0123456789abcdef", "UPSTASH_REDIS_REST_TOKEN": "upstash-SECRET-0123456789"}


@pytest.fixture
def web(monkeypatch, tmp_path):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.setenv("STILLHERE_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    from web import app, jobs, ledger, stations, store
    from worker import recipe as R
    real = R.load_index                                     # the worker fork's model manifest may not be committed yet
    monkeypatch.setattr(R, "load_index", lambda path=R.INDEX: {"models": {"tribe": "facebook/tribev2 (test stand-in)"}, **real(path)})
    monkeypatch.setattr(jobs, "CACHE", tmp_path / "no_cache.json")                   # tests own the precomputed cache
    store.set_store(store.MemoryStore())
    store.get_store().set("visitor:" + "a" * 32, "1")                                # call()'s default cookie, as if a page issued it
    if hasattr(app, "TokenBucket"):                                                  # its own test sets the real one
        monkeypatch.setattr(app, "BUCKET", app.TokenBucket(10 ** 9, 1e-6))
    return app, jobs, ledger, stations, store


def call(app, method, path, body=None, auth=None, cookie="sh_sid=" + "a" * 32, env=None):
    """One request; a POST carries Content-Type application/json as the page's and the worker's do; `env` adds or overrides keys."""
    data = json.dumps(body).encode() if body is not None else b""
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(data)), "wsgi.input": io.BytesIO(data),
           "HTTP_COOKIE": cookie, "wsgi.url_scheme": "http", **({"CONTENT_TYPE": "application/json"} if method == "POST" else {}),
           **(env or {})}
    if auth:
        env["HTTP_AUTHORIZATION"] = auth
    st = {}
    out = b"".join(app.app(env, lambda s, h: st.update(s=s, h=h)))
    return int(st["s"].split()[0]), dict(st["h"]), out


def planned_run(ledger, store, sid, clip_id="alice_ch02", condition="sentence_scramble", decline="none", ip=None):
    """A run as a live PLAN leaves it: the run record and its stored plan, routed to one clip and condition (or declined)."""
    run = ledger.start_run(sid, ip=ip)
    route = ({"kind": "live", "clip_id": clip_id, "condition": condition, "seed": 20261028, "defaults": []} if decline == "none"
             else {"kind": "decline", "reason": decline})
    store.get_store().set(f"runspec:{run}", json.dumps({"spec": {"population": "healthy_adults", "decline_reason": decline}, "route": route}))
    return run


def block_apis(monkeypatch, stations):
    def refuse(*a, **k):
        raise httpx.ConnectError("blocked in this test")
    monkeypatch.setattr(stations, "_post", refuse); monkeypatch.setattr(httpx, "stream", refuse)
    monkeypatch.setattr(stations.meta_client, "transcribe", refuse)


# ---- keys and fallbacks -----------------------------------------------------------------------------------------------------------
def test_no_key_reaches_the_browser(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    block_apis(monkeypatch, stations)
    seen = b""
    for p in ("/", "/app.js", "/app.css", "/pilot.css", "/receipt.html", "/receipt.js", "/api/status", "/api/evidence"):
        code, h, out = call(app, "GET", p); assert code == 200, p; seen += out + json.dumps(h).encode()
    for p, b in (("/api/plan", {"text": "Plan an exploratory stock TRIBE forecast of Alice chapter 1.", "backend": "spark-1.3"}),
                 ("/api/plan", {"text": "Plan it on Spark Open.", "backend": "spark-open"}), ("/api/audit", {"run_id": "x"}),
                 ("/api/build", {"old_bug": True}), ("/api/film", {"concept": "face"}),
                 ("/api/forecast", {"clip_id": "alice_ch01", "condition": "reversed", "seed": 0})):
        code, h, out = call(app, "POST", p, b); seen += out + json.dumps(h).encode()
    for v in SECRETS.values():
        assert v.encode() not in seen
    for f in public_files():
        assert not re.search(rb"(api[_-]?key|Bearer|SECRET|fw_[A-Za-z0-9]{8})", f.read_bytes()), f.name


def test_every_station_falls_back_to_its_recorded_run_with_the_apis_blocked(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    block_apis(monkeypatch, stations)
    _, _, out = call(app, "POST", "/api/plan", {"text": "Plan an exploratory stock TRIBE forecast of Alice chapter 1.", "backend": "spark-1.3"})
    d = json.loads(out); assert d["mode"] == "recorded" and d["label"] == "recorded run" and d["runs"]
    assert any(r["card"]["decision"] == "decline" for r in d["runs"]) and any(r["card"]["decision"] == "accept" for r in d["runs"])
    _, _, out = call(app, "POST", "/api/film", {"concept": "face"}); d = json.loads(out)
    assert d["mode"] == "recorded" and len(d["frames"]) == 3 and "CC BY 3.0" in d["credit"]
    run = planned_run(ledger, store, "a" * 32, "alice_ch01", "sentence_scramble")
    d = stations.audit(run, "a" * 32); assert d["mode"] == "recorded" and d["label"].startswith("Advisory check, not measured")
    d = stations.build(run, "a" * 32, "alice_ch01", "sentence_scramble", 20261028)
    assert d["mode"] == "recorded" and d["checklist"]
    _, _, out = call(app, "POST", "/api/forecast", {"clip_id": "alice_ch01", "condition": "reversed", "seed": 0})
    d = json.loads(out); assert d["state"] == "offline" and d["end_line"] == (
        "A public scan of this chapter exists (Alice Datasets, OpenNeuro ds002322); not compared here. Forecast, not a scan.")
    assert d["chip"]["short"] == "Forecast only · audio-only, laptop" and d["reversed_line"].startswith("0 words reach the text lane")


def test_live_off_does_not_use_a_visitors_runs(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    for _ in range(8):
        stations.plan("Plan a story test.", "spark-1.3", "b" * 32)
    ledger.start_run("b" * 32)                                                       # still allowed: recorded runs cost no live run


def test_old_bug_numbers_come_from_the_committed_check(web):
    app, jobs, ledger, stations, store = web
    d = stations.old_bug(); c = json.loads((REPO / "results/conditions_manipulation_check.json").read_text())["conditions"]["word_scramble"]
    assert d["pass"] is False and f"kept only {round(100 * c['recall'])}% of its words" in d["caption"]
    code, h, out = call(app, "GET", "/media/old_bug.wav"); assert code == 200 and out[:4] == b"RIFF"


# ---- guard, wording, limits -------------------------------------------------------------------------------------------------------
BANNED_HERE = ("aware", "conscious", "hospital", "detect", "diagnose", "a test for patients", "before anyone looked", "understanding",
               "comprehension", "validated")


def test_guard_replaces_banned_model_lines(web):
    from web import guard
    for w in BANNED_HERE:
        assert guard.clean(f"The plan {w} things.") == guard.TEMPLATE, w
    assert guard.clean("Check the clip length.") == "Check the clip length."
    assert guard.clean("a" + chr(0x2014) + "b") == guard.TEMPLATE


def test_page_files_and_recorded_runs_hold_no_banned_wording(web):
    from web import guard
    for f in public_files(text_only=True) + list((REPO / "web/replay").glob("*.json")):
        txt = f.read_text(errors="ignore")
        if f.suffix == ".json":
            txt = " ".join(str(v) for v in _strings(json.loads(txt)) if not str(v).startswith("data:"))
            txt = txt.replace("Use this on my patient to see if they understand.", "")    # a visitor's request, quoted, not our line
        assert not guard.hits(txt), (f.name, guard.hits(txt))
        assert chr(0x2014) not in txt, f.name


def _strings(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_spend_caps_and_the_session_limit(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    monkeypatch.setitem(ledger.TOTAL_CAP, "fireworks", 0.10); monkeypatch.setitem(ledger.DAILY_CAP, "fireworks", 1.0)
    r = ledger.reserve("fireworks", 0.06); ledger.settle(r, 0.01)
    ledger.reserve("fireworks", 0.06)
    with pytest.raises(ledger.Refused) as e:
        ledger.reserve("fireworks", 0.06)
    assert e.value.code == "spend_cap" and ledger.remaining()["fireworks"]["total_left_usd"] == pytest.approx(0.03, abs=1e-6)
    with pytest.raises(ledger.Refused):
        ledger.reserve("another_host", 0.01)
    t = 1_000_000.0
    for i in range(ledger.RUNS_PER_HOUR):
        ledger.start_run("c" * 32, now=t + i)
    with pytest.raises(ledger.Refused) as e:
        ledger.start_run("c" * 32, now=t + 10)
    assert e.value.code == "session_limit"
    ledger.start_run("c" * 32, now=t + 3601)
    run = ledger.start_run("d" * 32); ledger.use(run, "d" * 32, "plan")
    with pytest.raises(ledger.Refused):
        ledger.use(run, "d" * 32, "plan")
    with pytest.raises(ledger.Refused):
        ledger.use(run, "e" * 32, "audit")                                         # another session cannot use this run


def test_default_caps_follow_the_plan_of_oct_29_1920_utc(web):
    app, jobs, ledger, stations, store = web
    assert ledger.TOTAL_CAP == {"meta": 80, "fireworks": 50, "baseten": 50}
    assert ledger.DAILY_CAP == {"meta": 20, "fireworks": 12, "baseten": 12}


class _Card:
    def __init__(self, ok):
        self.spec = {"population": "healthy_adults", "decline_reason": "none", "stimulus": "librivox_audiobook"} if ok else {}
        self.text = "Plan (v2.1): healthy adults." if ok else "Planner unavailable."
        self.route = {"kind": "live", "clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 20261028, "defaults": []} if ok else None


def test_spark_open_falls_back_to_baseten_on_stubs(web, monkeypatch):
    """No real call: the planner and the chat call are stubs. Fireworks fails, or is capped, and Baseten answers."""
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-SECRET-0123456789")
    calls = []

    def fake_plan(q, backend, client):
        calls.append(backend); return _Card(backend == "spark-open-baseten")
    monkeypatch.setattr(stations.P, "plan", fake_plan)
    d = stations.plan("Plan a story test on chapter 2.", "spark-open", "f" * 32)
    assert d["mode"] == "live" and calls == ["spark-open", "spark-open-baseten"]
    assert stations.get_receipt(d["receipt_id"])["host"] == "baseten"
    calls.clear(); monkeypatch.setitem(ledger.DAILY_CAP, "fireworks", 0.0)              # Fireworks capped for the day: straight to Baseten
    d = stations.plan("Plan a story test on chapter 2.", "spark-open", "f" * 32)
    assert d["mode"] == "live" and calls == ["spark-open-baseten"]
    chats = []
    monkeypatch.setattr(stations.meta_client, "chat", lambda req, **kw: chats.append(req.model) or {"notes": ["Check the control."]})
    a = stations.audit(d["run_id"], "f" * 32)
    assert a["mode"] == "live" and chats == ["baseten/muse-spark-open-1026"] and a["notes"] == ["Check the control."]
    monkeypatch.delenv("BASETEN_API_KEY"); calls.clear()
    d = stations.plan("Plan a story test on chapter 2.", "spark-open", "f" * 32)
    assert d["mode"] == "recorded" and calls == []                                    # Fireworks capped, Baseten off: recorded run


def test_after_the_meta_credit_end_meta_stays_live_under_a_20_dollar_cap(web, monkeypatch):
    """Ruling of Oct 29, 19:40 UTC, item 3, on stubs: Meta stations stay live after the credit end; Meta spend after it has one more
    total cap of $20, then labelled recorded runs; a failed call falls back with its own label."""
    app, jobs, ledger, stations, store = web
    calls = []
    monkeypatch.setattr(stations.P, "plan", lambda q, b, c: calls.append(b) or _Card(True))
    monkeypatch.setattr(stations, "chat_cost", lambda usage: 4.0)                     # each stub plan "costs" $4
    monkeypatch.setitem(ledger.DAILY_CAP, "meta", 1000.0)                               # only the after-credit cap is under test
    monkeypatch.setenv("META_CREDIT_END_UTC", "2026-10-01T00:00:00Z")
    assert ledger.meta_credit_ended() and stations.live_ok("meta")
    modes = [stations.plan("Plan a story test on chapter 2.", "spark-1.3", f"{i:032x}")["mode"] for i in range(7)]
    assert modes == ["live"] * 5 + ["recorded"] * 2 and len(calls) == 5                  # 5 x $4 = $20, then the cap holds
    d = stations.plan("Plan a story test on chapter 2.", "spark-1.3", "9" * 32)
    assert d["why"] == ("The event's Meta API credit has ended and the $20 set aside after it is spent, so this station shows its "
                        "recorded run.") and d["label"] == "recorded run"
    assert float(store.get_store().get("spend:meta:after_credit")) == pytest.approx(20.0)
    monkeypatch.setenv("META_CREDIT_END_UTC", "2099-01-01T00:00:00Z")                 # before the end: no after-credit cap
    assert stations.plan("Plan a story test on chapter 2.", "spark-1.3", "8" * 32)["mode"] == "live"
    monkeypatch.setattr(stations.P, "plan", lambda q, b, c: _Card(False))              # a failed call: recorded, with its own line
    d = stations.plan("Plan a story test on chapter 2.", "spark-1.3", "7" * 32)
    assert d["mode"] == "recorded" and d["why"] == "The live call failed, so this station shows its recorded run."
    monkeypatch.delenv("META_CREDIT_END_UTC")
    assert ledger.META_CREDIT_END_UTC == "2026-11-05T02:00:00Z"
    assert ledger.meta_credit_ended(datetime.datetime(2026, 11, 5, 2, 0, tzinfo=datetime.timezone.utc))
    assert not ledger.meta_credit_ended(datetime.datetime(2026, 11, 5, 1, 59, tzinfo=datetime.timezone.utc))


# ---- the story card (story_menu_rule.md section 6) -------------------------------------------------------------------------------
def stub_menu():
    m = json.loads((REPO / "stillhere/live_menu.json").read_text())
    story = {"id": "story_gift_of_the_magi", "title": "The Gift of the Magi", "reader": "A Reader",
             "source": "LibriVox (public domain)", "start_s": 61.2, "dur_s": 38.4, "intact_wav_sha256": "0" * 64, "public_scan": None,
             "screen": {"archive_url": "https://archive.org/download/item_x/magi_64kb.mp3", "sentences": 7, "notes_host": "fireworks",
                        "notes": [{"tag": "death or dying", "t": 750.9, "quote": None},
                                  {"tag": "grief or loss", "t": 65.4, "quote": "she cried"},
                                  {"tag": "not a listed tag", "t": 5.0, "quote": "x"}]}}
    plain = {**story, "id": "story_plain", "title": "A Plain Story", "screen": {**story["screen"], "notes": []}}
    m["clips"] = m["clips"] + [story, plain]
    m["screen_summary"] = {"n_on_menu": 10, "failed": [{"title": "A Two-Voice Story", "bar": "1"}, {"title": "Too Quiet", "bar": "trim"}],
                           "held": [{"title": "A Held Story", "reason": "window is the reader's introduction, not the story"}],
                           "rule": "docs/rules/story_menu_rule.md"}
    return m


@pytest.fixture
def menu_stub(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    real, m = stations._load, stub_menu()
    monkeypatch.setattr(stations, "_load", lambda rel: m if rel == "stillhere/live_menu.json" else real(rel))
    return stations


def test_story_card_is_the_rules_line_word_for_word(menu_stub):
    c = menu_stub.story_card("story_gift_of_the_magi")
    assert c["text"] == ("The Gift of the Magi, read by A Reader: one of 10 screened public stories · 1 voice heard · 7 sentences · notes: "
                         "grief or loss (1:05), death or dying (12:30). Content notes come from Muse Spark Open reading the transcript. "
                         "Not a full review; listen before you use a story.")
    assert [n["href"] for n in c["notes"]] == ["https://archive.org/download/item_x/magi_64kb.mp3#t=65",
                                                "https://archive.org/download/item_x/magi_64kb.mp3#t=750"]
    assert c["text"] == c["lead"] + ", ".join(n["label"] for n in c["notes"]) + c["tail"]
    p = menu_stub.story_card("story_plain")
    assert "notes: none of the listed tags found. Content notes" in p["text"] and p["notes"] == []
    for cid in ("alice_ch01", "alice_ch02", "alice_ch03"):                          # no screen data: no card, the views are unchanged
        assert menu_stub.story_card(cid) is None


def test_story_card_reaches_plan_and_build_and_failures_are_title_and_bar_only(menu_stub, monkeypatch):
    stations = menu_stub

    class C:
        spec = {"population": "healthy_adults", "decline_reason": "none", "stimulus": "librivox_audiobook"}
        text = "Plan (v2.1): healthy adults."
        route = {"kind": "live", "clip_id": "story_gift_of_the_magi", "condition": "sentence_scramble", "seed": 20261028, "defaults": []}
    assert stations.card_view(C())["story"]["text"].startswith("The Gift of the Magi, read by A Reader")
    C.route = {**C.route, "clip_id": "alice_ch02"}
    assert stations.card_view(C())["story"] is None
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    b = stations.build(None, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)
    assert b["story"] is None
    assert stations.menu()["failed"] == [{"title": "A Two-Voice Story", "bar": "1"}, {"title": "Too Quiet", "bar": "trim"}]
    assert stations.menu()["held"] == [{"title": "A Held Story", "reason": "window is the reader's introduction, not the story"}]


def test_story_card_passes_the_guard(menu_stub, monkeypatch):
    """Item 5 of the ruling of Oct 30, 04:20 UTC: title, reader, tags and caveat go through guard.clean; a banned word anywhere in
    them turns the whole card into the guard's template line."""
    from web import guard
    stations = menu_stub
    m = stub_menu()
    for field, value in (("title", "The Consciousness of Things"), ("reader", "A Hospital Reader")):
        bad = json.loads(json.dumps(m)); bad["clips"][-2][field] = value
        monkeypatch.setattr(stations, "_load", lambda rel, b=bad: b if rel == "stillhere/live_menu.json" else None)
        c = stations.story_card("story_gift_of_the_magi")
        assert c["text"] == guard.TEMPLATE and c["lead"] == guard.TEMPLATE and c["notes"] == [] and value not in json.dumps(c)


def test_the_avoid_line_shows_word_for_word(web):
    """Item 3: planner v2.1's avoid line reaches the PLAN card exactly as the planner writes it."""
    app, jobs, ledger, stations, store = web
    from stillhere import egress, planner, planner_v21 as P
    assert P.AVOID_LINE == "The chosen story's card lists none of them. Not a full review; listen before you use a story."
    ans = {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "analysis": "forecast_tribe_stock",
           "brain_areas": ["language_network"], "conditions": ["intact", "sentence_scramble"], "decline_reason": "none",
           "clip_id": None, "condition": "sentence_scramble", "seed": None, "avoid_tags": [P.TAG_IDS["violence"]]}
    card = P.plan(planner.public("q_plan_6yo"), "spark-1.3", lambda req: egress.check(req) and {"content": ans})
    view = stations.card_view(card)
    line = next(l for l in view["lines"] if "Notes to avoid" in l)
    assert line.endswith(P.AVOID_LINE) and line == next(l for l in card.text.split("\n") if "Notes to avoid" in l)


def test_an_unexpected_error_logs_its_type_never_the_request(web, monkeypatch, capsys):
    app, jobs, ledger, stations, store = web
    secret_text = "my patient Jane Roe, scan of 12 March"
    def boom(text, backend, sid, **kw):
        raise ValueError(f"input_value={text!r}")                                      # pydantic-style message that quotes the input
    monkeypatch.setattr(stations, "plan", boom)
    code, h, out = call(app, "POST", "/api/plan", {"text": secret_text, "backend": "spark-1.3"})
    err = capsys.readouterr()
    assert code == 500 and json.loads(out) == {"error": "server_error"} and secret_text.encode() not in out
    assert "ValueError" in err.err and secret_text not in err.err + err.out


def test_on_vercel_live_calls_need_the_shared_store(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    assert stations.live_ok("meta")                                                     # local: keys set, in-memory store
    monkeypatch.setenv("VERCEL", "1")
    assert not stations.live_ok("meta") and not stations.live_ok("fireworks")          # serverless without Upstash: no spend
    monkeypatch.setattr(stations, "store_shared", lambda: True)
    assert stations.live_ok("meta")


def test_a_call_that_never_answers_is_not_counted(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    def down(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(stations.meta_client, "transcribe", down)
    run = planned_run(ledger, store, "a" * 32, "alice_ch02", "sentence_scramble")
    d = stations.build(run, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)
    assert d["mode"] == "recorded" and d["why"] == "The live call failed, so this station shows its recorded run."
    assert float(store.get_store().get("spend:meta:total")) == pytest.approx(0.0)


def test_without_any_question_list_typed_text_is_refused_before_any_call(web, monkeypatch, tmp_path):
    """On a server where STILLHERE_QUESTION_BLOCKLIST names no file, egress fails closed: PLAN and FILM refuse typed text before any
    model call; the other stations and the recorded runs are unaffected."""
    app, jobs, ledger, stations, store = web
    monkeypatch.setattr(stations.egress, "BLOCKLIST", tmp_path / "missing.txt")
    monkeypatch.setattr(stations.P, "plan", lambda *a: pytest.fail("no model call"))
    d = stations.plan("Plan a story test on The Black Cat.", "spark-1.3", "h" * 32)
    assert d["mode"] == "refused" and d["line"].startswith("This request can't be sent to the planner.")
    assert stations.film(None, "h" * 32, "face")["mode"] == "refused"
    monkeypatch.setattr(stations.egress, "BLOCKLIST", REPO / "web/server_question_blocklist.txt")      # the committed server list
    assert stations.egress.PublicQuestion(text="Plan a story test on The Black Cat.").text


def test_safe_never_appears(menu_stub):
    texts = [menu_stub.story_card(c)["text"] for c in ("story_gift_of_the_magi", "story_plain")]
    texts += [f.read_text() for f in public_files(text_only=True)] + [menu_stub.STORY_LEAD, menu_stub.STORY_TAIL, menu_stub.STORY_NONE]
    for t in texts:
        assert not re.search(r"\bsafe(ly|ty)?\b", t, re.I)


def test_story_cards_on_the_real_menu(web):
    """3a on the worker fork's screened stories: the rule's line exactly, {n} = the stories on the menu, notes in time order with
    m:ss and an archive.org #t= link, failures title and bar only, no card for the Alice clips, and "safe" nowhere."""
    app, jobs, ledger, stations, store = web
    from stillhere.story_tags import TAGS
    menu = json.loads((REPO / "stillhere/live_menu.json").read_text())
    screened = [c for c in menu["clips"] if c.get("screen")]
    if not screened:
        pytest.skip("no screened story on the live menu yet")
    n = menu["screen_summary"]["n_on_menu"]
    assert n == len(screened)
    for c in menu["clips"]:
        card = stations.story_card(c["id"])
        if not c.get("screen"):
            assert card is None; continue
        sc, notes = c["screen"], sorted(c["screen"]["notes"], key=lambda x: x["t"])
        assert all(x["tag"] in TAGS for x in notes)
        exp = (f"{c['title']}, read by {c['reader']}: one of {n} screened public stories · 1 voice heard · {sc['sentences']} sentences · "
               "notes: " + (", ".join(f"{x['tag']} ({int(x['t']) // 60}:{int(x['t']) % 60:02d})" for x in notes) or "none of the listed tags found")
               + ". Content notes come from Muse Spark Open reading the transcript. Not a full review; listen before you use a story.")
        assert card["text"] == exp
        assert [x["href"] for x in card["notes"]] == [f"{sc['archive_url']}#t={int(x['t'])}" for x in notes]
        assert sc["archive_url"].startswith("https://archive.org/download/") and sc["archive_url"].endswith(".mp3")
        assert not re.search(r"\bsafe\b", card["text"], re.I)
    assert all(set(f) == {"title", "bar"} and f["bar"] in ("1", "2", "3", "trim") for f in menu["screen_summary"]["failed"])
    assert stations.menu()["failed"] == menu["screen_summary"]["failed"]


def test_example_chips_are_on_screened_stories_with_saved_runs(web):
    """2f: the chips come from results/planner_v21_story_chips.json; both backends routed each to the recipe it names, whose
    forecast is cached and whose BUILD replay passed. alice_ch02 stays as the backup on the menu."""
    app, jobs, ledger, stations, store = web
    monkeypatch_cache = jobs.CACHE
    jobs.CACHE = REPO / "data/forecast_cache.json"
    try:
        doc = json.loads((REPO / "results/planner_v21_story_chips.json").read_text())
        assert stations.menu()["chips"] == [e["text"] for e in doc["examples"]] and len(doc["examples"]) == 3
        builds = json.loads((REPO / "web/replay/build.json").read_text())
        clips = set()
        for e in doc["examples"]:
            routes = {(r["route"]["clip_id"], r["route"]["condition"]) for r in e["runs"]}
            assert len(routes) == 1 and {r["backend"] for r in e["runs"]} == {"spark-1.3", "spark-open"}
            r = e["runs"][0]["route"]; rec = jobs.recipe(r["clip_id"], r["condition"], r["seed"])
            assert r["clip_id"].startswith("story_") and jobs.cached(jobs.cache_key(rec)) is not None
            assert builds[f"{r['clip_id']}:{r['condition']}"]["pass"] is not False
            clips.add(r["clip_id"])
        assert len(clips) == 3 and "alice_ch02" in [c["id"] for c in json.loads((REPO / "stillhere/live_menu.json").read_text())["clips"]]
    finally:
        jobs.CACHE = monkeypatch_cache


# ---- jobs API with a fake worker --------------------------------------------------------------------------------------------------
W = "Bearer " + SECRETS["STILLHERE_WORKER_TOKEN"]


def result_for(key, n=40):
    return {"cache_key": key, "tr_s": 1.0, "seconds": n, "regions": {"language": [0.1] * n, "hearing": [0.2] * n}, "word_rate": [2] * n,
            "n_words_text_lane": 0, "models": {"tribe": "facebook/tribev2"}, "input_sha256": "0" * 64, "peak_vram_gb": 3.5,
            "worker_version": "test", "gpu": "test GPU", "timings_s": {"total": 9.0}, "computed_utc": "2026-10-30T14:00:00Z"}


def test_fake_worker_round_trip(web):
    app, jobs, ledger, stations, store = web
    assert call(app, "POST", "/api/worker/claim", {}, auth="Bearer nope")[0] == 401
    assert call(app, "POST", "/api/worker/claim", {})[0] == 401
    code, _, _ = call(app, "POST", "/api/worker/claim", {}, auth=W); assert code == 204 and jobs.worker_online()
    run = planned_run(ledger, store, "a" * 32, "alice_ch02", "reversed")
    code, _, out = call(app, "POST", "/api/forecast", {"run_id": run, "clip_id": "alice_ch02", "condition": "reversed", "seed": 99})
    d = json.loads(out); assert code == 202 and d["state"] == "queued"
    code, _, out = call(app, "POST", "/api/worker/claim", {"worker": "test"}, auth=W); j = json.loads(out)["job"]
    assert j["recipe"] == {"clip_id": "alice_ch02", "condition": "reversed", "seed": 0} and j["deadline_s"] == 40
    jid = j["id"]
    for e in ({"stage": "build", "lane": None, "status": "start"}, {"stage": "text_lane", "lane": "llama-3.2", "status": "end", "detail": {"n_words": 0}},
              {"stage": "video_lane", "lane": "v-jepa-2", "status": "off"}):
        assert call(app, "POST", f"/api/worker/jobs/{jid}/event", e, auth=W)[0] == 204
    assert call(app, "POST", f"/api/worker/jobs/{jid}/event", {"stage": "telepathy", "lane": None, "status": "start"}, auth=W)[0] == 422
    bad = result_for(j["cache_key"]); bad["regions"]["language"] = [float("nan")] * 40
    assert call(app, "POST", f"/api/worker/jobs/{jid}/result", bad, auth=W)[0] == 422
    assert call(app, "POST", f"/api/worker/jobs/{jid}/result", result_for("f" * 64), auth=W)[0] == 422
    assert call(app, "POST", f"/api/worker/jobs/{jid}/result", result_for(j["cache_key"]), auth=W)[0] == 204
    assert call(app, "POST", f"/api/worker/jobs/{jid}/result", result_for(j["cache_key"]), auth=W)[0] == 409
    code, _, out = call(app, "GET", f"/api/forecast/{d['job_id']}"); s = json.loads(out)
    assert s["status"] == "done" and len(s["events"]) == 3 and s["result"]["regions"]["language"][0] == 0.1
    assert s["events"][1]["detail"] == {"n_words": 0}
    run2 = planned_run(ledger, store, "a" * 32, "alice_ch02", "reversed")
    code, _, out = call(app, "POST", "/api/forecast", {"run_id": run2, "clip_id": "alice_ch02", "condition": "reversed", "seed": 5})
    assert json.loads(out)["state"] == "cached"                                     # reversed ignores the seed: same cache key


def test_jobs_expire_and_fall_back(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    run = planned_run(ledger, store, "a" * 32, "alice_ch03", "sentence_scramble")
    _, _, out = call(app, "POST", "/api/forecast", {"run_id": run, "clip_id": "alice_ch03", "condition": "sentence_scramble", "seed": 3})
    jid = json.loads(out)["job_id"]
    _, _, out = call(app, "POST", "/api/worker/claim", {}, auth=W); j = json.loads(out)["job"]
    monkeypatch.setattr(jobs, "JOB_EXPIRY_S", -1)
    assert call(app, "POST", f"/api/worker/jobs/{j['id']}/result", result_for(j["cache_key"]), auth=W)[0] == 409
    _, _, out = call(app, "GET", f"/api/forecast/{jid}"); assert json.loads(out)["status"] == "expired" and "fallback" in json.loads(out)


@pytest.mark.parametrize("shape", ["flat", "entries"])
def test_worker_offline_serves_the_precomputed_cache(web, monkeypatch, tmp_path, shape):
    app, jobs, ledger, stations, store = web
    r = jobs.recipe("alice_ch03", "sentence_scramble", 20261028); key = jobs.cache_key(r)
    entries = {key: {**result_for(key), "recipe": r}}
    doc = {"models": {}, "default_seed": 20261028, **({"entries": entries} if shape == "entries" else entries)}
    (tmp_path / "c.json").write_text(json.dumps(doc)); monkeypatch.setattr(jobs, "CACHE", tmp_path / "c.json")
    _, _, out = call(app, "POST", "/api/forecast", {"clip_id": "alice_ch03", "condition": "sentence_scramble", "seed": 3})
    d = json.loads(out); assert d["state"] == "offline" and d["fallback"]["source"] == "cache" and "job_id" not in d
    assert d["fallback"]["computed_utc"] and d["fallback"]["recipe"]["seed"] == 20261028
    assert d["fallback"]["note"] == "No saved run for seed 3; shown: the saved run at seed 20261028."
    _, _, out = call(app, "POST", "/api/forecast", {"clip_id": "alice_ch03", "condition": "sentence_scramble", "seed": 20261028})
    d = json.loads(out); assert d["state"] == "offline" and d["fallback"]["note"] is None   # the saved run is for this very choice


def test_recorded_build_only_for_its_own_clip_and_condition(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    d = stations.build(None, "a" * 32, "alice_ch02", "reversed", 0)
    assert d["mode"] == "recorded" and d["line"] == "No recorded check for this clip and condition." and d["checklist"] == []
    d = stations.build(None, "a" * 32, "alice_ch01", "sentence_scramble", 1)
    assert d["checklist"] and d["label"] == "recorded run, seed 20261028"


def test_queued_jobs_expire_when_no_worker_is_online(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)                              # worker online: the job queues
    run = planned_run(ledger, store, "a" * 32, "alice_ch02", "sentence_scramble")
    _, _, out = call(app, "POST", "/api/forecast", {"run_id": run, "clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 2})
    jid = json.loads(out)["job_id"]
    store.get_store().delete("worker:last_seen"); monkeypatch.setattr(jobs, "JOB_EXPIRY_S", -1)
    _, _, out = call(app, "GET", f"/api/forecast/{jid}"); assert json.loads(out)["status"] == "expired"


def test_every_menu_clip_was_checked_for_a_public_scan_and_the_end_line_follows_it(web):
    app, jobs, ledger, stations, store = web
    menu = json.loads((REPO / "stillhere/live_menu.json").read_text())
    ids = [c["id"] for c in menu["clips"]]                         # fresh screened stories first (story menu rule), then the Alice clips
    alice = [i for i in ids if i.startswith("alice_")]
    assert alice == ["alice_ch02", "alice_ch01", "alice_ch03"] and ids[len(ids) - 3:] == alice      # ch02 stays the backup
    assert all(i.startswith("story_") for i in ids[:len(ids) - 3])
    for c in menu["clips"]:
        assert "public_scan" in c, c["id"]
        line = stations.end_line(c["id"])
        if c["public_scan"] is None:
            assert line == "No public scan of this clip. Forecast, not a scan."
        else:
            assert "No public scan" not in line and c["public_scan"]["accession"] in line
        _, _, out = call(app, "POST", "/api/forecast", {"clip_id": c["id"], "condition": "reversed", "seed": 0})
        d = json.loads(out); assert d["end_line"] == line and d["chip"]["long"] == line
    assert stations.end_line("alice_ch01") == ("A public scan of this chapter exists (Alice Datasets, OpenNeuro ds002322); not compared here. "
                                               "Forecast, not a scan.")
    ex = json.loads((REPO / "results/planner_v21_examples.json").read_text())
    for e in ex["examples"]:                                                          # no example chip ends on "No public scan" for ch01
        for r in e["runs"]:
            if r["route"].get("clip_id") == "alice_ch01":
                assert "No public scan" not in stations.end_line("alice_ch01")
    with pytest.raises(KeyError):
        stations.end_line("not_on_the_menu")


def test_recipes_only_from_the_live_menu(web):
    app, jobs, ledger, stations, store = web
    for clip, cond, seed in (("other_clip", "intact", 0), ("alice_ch01", "upload", 0), ("alice_ch01", "intact", -1), ("alice_ch01", "intact", "7")):
        with pytest.raises(ValueError):
            jobs.recipe(clip, cond, seed)
    with pytest.raises(stations.Bad):
        stations.build(None, "a" * 32, "alice_ch01", "sentence_scramble", 999)       # not a listed build: never transcribed


# ---- evidence endings (rulings 2, 3, 4, 5) ----------------------------------------------------------------------------------------
def test_evidence_pairs_the_gray_line_with_d20_only_on_a_pass(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    real = stations._load
    for status, pair in (("pass", "d20"), ("fail", "lpp"), (None, "lpp")):
        d20 = None if status is None else {**json.loads((REPO / "results/d20_notthefall.json").read_text())}
        if d20:
            d20["primary"] = {**d20["primary"], "status": status, "n": 12}
        monkeypatch.setattr(stations, "_load", lambda rel, d=d20: d if rel == "results/d20_notthefall.json" else real(rel))
        e = stations.evidence()
        assert e["pair"] == pair and e["pair_label"] == ("Real listeners, different story." if pair == "d20" else "Real listeners, held-out story")
        if pair == "d20":
            assert "12 listeners, 9.1 minutes = 365 TRs per version" in e["d20"]["line"]
        elif status == "fail":
            assert "notthefall" not in re.sub(r"<[^>]+>", " ", e["html"])
    for w in ("understanding", "comprehension"):
        assert w not in stations.evidence()["html"].lower()
    assert stations.evidence()["order_map"]["verdict"] == "NO-GO"


def test_a_plan_off_the_live_list_ends_on_its_evidence_panel(web):
    app, jobs, ledger, stations, store = web
    from stillhere import planner_v21 as P

    class Card:
        text, spec = "Plan (v2.1): healthy adults, English, Le Petit Prince.", {"stimulus": "le_petit_prince", "decline_reason": "none"}
        route = P.route({"stimulus": "le_petit_prince", "decline_reason": "none", "conditions": ["intact", "reversed"]})
    v = stations.card_view(Card())
    assert v["route"]["kind"] == "evidence" and v["route"]["panel"] == "lpp_section8"


# ---- 360 px --------------------------------------------------------------------------------------------------------------------
CHROME = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or shutil.which("chromium")


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
def test_page_lays_out_at_360_px(web, tmp_path, monkeypatch):
    """The live page in a 360 px iframe of a probe page served from the same origin (a test-only wrapper; the app sends no probe)."""
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    probe = (b'<!doctype html><html><body style="margin:0"><iframe id="f" src="/" style="border:0;width:360px;height:900px" '
             b'onload="setTimeout(()=>{var d=this.contentDocument.documentElement;document.body.dataset.sw=d.scrollWidth},1500)">'
             b'</iframe></body></html>')

    def wrapper(env, start):
        if env["PATH_INFO"] == "/probe.html":
            start("200 OK", [("Content-Type", "text/html")]); return [probe]
        return app.app(env, lambda s, h: start(s, [(k, v) for k, v in h if k != "Content-Security-Policy"]))   # let the probe frame it

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, wrapper, handler_class=Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={tmp_path / 'p'}",
                              "--virtual-time-budget=6000", "--window-size=520,900", "--dump-dom", f"http://127.0.0.1:{port}/probe.html"],
                             capture_output=True, text=True, timeout=120).stdout
    finally:
        srv.shutdown()
    m = re.search(r'data-sw="(\d+)"', out)
    assert m and int(m.group(1)) <= 360, out[-300:]


def test_a_hanging_fireworks_call_times_out_and_baseten_answers(web, monkeypatch):
    """Ruling of Oct 30, 07:40 UTC, e: Spark Open on Fireworks gets 25 s on the page's plan and audit calls, Baseten 60 s, Spark 1.3
    unchanged; a timeout settles as api_error and falls through to Baseten. The Fireworks stub is a real socket that accepts and never
    answers, reached by the real httpx client with the station's timeout (shortened to 0.5 s here)."""
    app, jobs, ledger, stations, store = web
    assert stations.TIMEOUT_S == {"meta": 60.0, "fireworks": 25.0, "baseten": 60.0}
    monkeypatch.setenv("BASETEN_API_KEY", "bt-SECRET-0123456789abcdef")
    plan_timeouts = []
    monkeypatch.setattr(stations.planner, "live_client", lambda timeout=60.0: plan_timeouts.append(timeout) or (lambda req: None))
    monkeypatch.setattr(stations.P, "plan", lambda q, b, client: _Card(b == "spark-open-baseten"))
    run = stations.plan("Plan a story test on chapter 2.", "spark-open", "f" * 32)["run_id"]
    stations.plan("Plan a story test on chapter 2.", "spark-1.3", "f" * 32)
    assert plan_timeouts == [25.0, 60.0, 60.0]                                        # Fireworks, then Baseten; then Spark 1.3 on Meta
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(4)                   # the kernel completes the handshake; nothing answers
    real_post, seen = httpx.post, []

    def post(url, **kw):
        seen.append((url.split("/")[2], kw["timeout"]))
        if "api.fireworks.ai" in url:
            return real_post(f"http://127.0.0.1:{srv.getsockname()[1]}/chat/completions", **kw)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"usage": {}, "choices": [
            {"message": {"content": json.dumps({"notes": ["Check the control."]})}}]})
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setitem(stations.TIMEOUT_S, "fireworks", 0.5)
    t0 = datetime.datetime.now()
    try:
        a = stations.audit(run, "f" * 32)
    finally:
        srv.close()
    assert (datetime.datetime.now() - t0).total_seconds() < 10
    assert seen == [("api.fireworks.ai", 0.5), ("inference.baseten.co", 60.0)]
    assert a["mode"] == "live" and a["notes"] == ["Check the control."]
    assert stations.get_receipt(a["receipt_id"])["model"] == "baseten/muse-spark-open-1026"


RELATIVE_DAYS = re.compile(r"\b(tonight|today|tomorrow|yesterday|this week)\b", re.I)


def test_no_page_or_card_text_names_a_relative_day(web, monkeypatch):
    """Ruling of Oct 30, 07:40 UTC, b: the page is read for days after it is written, so no text it can show says tonight, today,
    tomorrow, yesterday or this week: the static files, the evidence panels on both sides of the wall's cutoff, every reason line, every
    story card, the recorded runs and the Film station's status in each state."""
    app, jobs, ledger, stations, store = web
    bp = stations.guard._bp
    texts = [f.read_text() for f in sorted((REPO / "web/public").iterdir()) if f.suffix in (".html", ".js")]
    texts += [f.read_text() for f in sorted((REPO / "web/replay").glob("*.json"))]
    texts += [bp.build(), json.dumps(stations.evidence())]
    for now in (bp.WALL_CUTOFF_UTC - datetime.timedelta(seconds=1), bp.WALL_CUTOFF_UTC):
        texts.append(bp.wall_html(None, now=now))
    texts += [stations.why_line(c) for c in ("session_limit", "spend_cap", "no_key", "api_error", "no_run", "station_used",
                                              "provider_not_offered", "recorded_plan", "post_credit_cap", "other") + NEW_CODES]
    texts += [jobs.BUSY_LINE, jobs.FAILED_LINE, jobs.ADDRESS_BUSY_LINE]
    menu = json.loads((REPO / "stillhere/live_menu.json").read_text())
    clips = [c["id"] for c in menu.get("clips", [])]
    assert clips
    texts += [json.dumps(stations.story_card(c)) for c in clips]
    texts += [bp.film_status(fm)["text"] for fm in (None, {"outcome": "PASS"}, {"outcome": "NOT RUN", "why": "gate 0 failed"})]
    hits = sorted({m.group(0) for t in texts for m in RELATIVE_DAYS.finditer(t)})
    assert hits == []


def test_evidence_carries_the_alice_panel_after_section8(web):
    app, jobs, ledger, stations, store = web
    d = json.loads((REPO / "results/alice_heldout.json").read_text())
    h = stations.evidence()["html"]
    assert d["lines"][0] in h and d["lines"][1] in h
    assert h.index("Le Petit Prince, section 8</h3>") < h.index("The Alice Datasets, chapter 1</h3>")


# ---- hardening (Oct 31): cookie rotation, caps, atomic counts, plan checks, the queue, POST checks, caching ------------------------
NEW_CODES = ("ip_limit", "station_cap", "not_the_plan", "plan_declined", "session_daily", "ip_daily", "shared_limit", "shared_daily",
             "worker_offline", "store_down", "ip_station")
PLAN_BODY = {"text": "Plan a story test on chapter 2.", "backend": "spark-1.3"}
ROUTE = {"clip_id": "alice_ch02", "condition": "sentence_scramble"}                 # where _Card routes


def stub_plans(monkeypatch, stations):
    calls = []
    monkeypatch.setattr(stations.P, "plan", lambda q, b, c: calls.append(b) or _Card(True))
    return calls


@pytest.fixture
def live(web, monkeypatch):
    """Every paid station goes live on stubs: the planner, the audit chat, Voice Transcribe and SAM (with opencv, face_edit and
    meta-sam-parser stood in); the real egress is blocked underneath. Returns the web tuple and the count of each paid call."""
    app, jobs, ledger, stations, store = web
    import numpy as np
    import stillhere
    from web import sam
    block_apis(monkeypatch, stations)
    paid = {"plan": 0, "audit": 0, "transcribe": 0, "sam": 0}
    monkeypatch.setattr(stations.P, "plan", lambda q, b, c: paid.__setitem__("plan", paid["plan"] + 1) or _Card(True))
    monkeypatch.setattr(stations.meta_client, "chat", lambda req, **kw: paid.__setitem__("audit", paid["audit"] + 1) or {"notes": []})
    monkeypatch.setattr(stations.meta_client, "transcribe", lambda a, **k: paid.__setitem__("transcribe", paid["transcribe"] + 1) or {"transcript": ""})
    fe = types.ModuleType("stillhere.face_edit")
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imread=lambda p: np.zeros((4, 4, 3), np.uint8), cvtColor=lambda a, c: a,
                                                                  COLOR_BGR2RGB=4))
    monkeypatch.setitem(sys.modules, "meta_sam_parser", types.ModuleType("meta_sam_parser"))
    monkeypatch.setitem(sys.modules, "stillhere.face_edit", fe); monkeypatch.setattr(stillhere, "face_edit", fe, raising=False)
    monkeypatch.setattr(sam, "segment", lambda clip, q: paid.__setitem__("sam", paid["sam"] + 1) or {"objects": [], "usage": {},
                                                                                                     "status": "response.completed"})
    R, memo = stations._recipe_mod(), {}
    real = R.build_condition
    monkeypatch.setattr(R, "build_condition", lambda r, idx: memo.get(json.dumps(r, sort_keys=True)) or memo.setdefault(
        json.dumps(r, sort_keys=True), real(r, idx)))                                # the same real build, made once
    return web, paid


def J(r):
    return json.loads(r[2])


def issued(sid):
    """A cookie as if a page had issued it (ledger.issue_visitor), for tests that need many visitors from one address."""
    from web.store import get_store
    get_store().set(f"visitor:{sid}", "1"); return f"sh_sid={sid}"


def page_cookie(app, ip):
    """The cookie the page's first call (POST /api/visitor) gets from this address."""
    return call(app, "POST", "/api/visitor", {}, cookie="", env={"HTTP_X_REAL_IP": ip})[1]["Set-Cookie"].split(";")[0]


def upstash(monkeypatch, store, fail=False):
    """Upstash's REST API in process: web/smoke_local.py's stand-in (the commands the smoke test serves) behind a fake httpx.post,
    as the store. Returns the requests sent, one per round trip; `fail`: every request fails, as in an outage."""
    from web.smoke_local import RedisStub
    stub = object.__new__(RedisStub); stub.db, stub.exp, stub.seen = {}, {}, []
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append(json)
        if fail:
            raise httpx.ConnectError("store down")
        out = [{"result": stub.run(c)} for c in json] if url.endswith("/pipeline") else {"result": stub.run(json)}
        return httpx.Response(200, request=httpx.Request("POST", url), json=out)
    monkeypatch.setattr(httpx, "post", post); store.set_store(store.UpstashStore("https://up.example", "t"))
    return sent


def hours(store, station):
    """The station's live uses in the store's current hour window."""
    return len(store.get_store()._d.get(f"win:cap:{station}") or {})


def full_run(app, ck, ip):
    """One ordinary visit through the app with cookie `ck`: PLAN, BUILD seven times (the build and every reshuffle), AUDIT, FILM."""
    env = {"HTTP_X_REAL_IP": ip}
    d = J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck, env=env)); rid = d.get("run_id")
    b = [J(call(app, "POST", "/api/build", {"run_id": rid, **ROUTE, "seed": 20261028}, cookie=ck, env=env)) for _ in range(7)]
    a = J(call(app, "POST", "/api/audit", {"run_id": rid}, cookie=ck, env=env))
    f = J(call(app, "POST", "/api/film", {"run_id": rid, "concept": "face"}, cookie=ck, env=env))
    return {"plan": d["mode"], "build": [x["mode"] for x in b], "audit": a["mode"], "film": f["mode"],
            "why": [x.get("why") for x in [d, *b, a, f] if x.get("why")]}


LIVE_RUN = {"plan": "live", "build": ["live"] * 7, "audit": "live", "film": "live", "why": []}


def test_new_lines_pass_the_guard(web):
    app, jobs, ledger, stations, store = web
    from web import guard
    lines = [stations.why_line(c) for c in NEW_CODES] + [jobs.BUSY_LINE, jobs.FAILED_LINE, jobs.ADDRESS_BUSY_LINE]
    assert len(set(lines)) == len(lines) and stations.why_line("other") not in lines
    lines += [stations.why_line(c, shown=stations.SAVED_SHOWN) for c in NEW_CODES]
    assert not guard.hits(" ".join(lines)) and chr(0x2014) not in " ".join(lines) and "{" not in " ".join(lines)


def test_an_address_gets_more_runs_than_a_session_and_ipv6_counts_by_its_64(live, monkeypatch):
    """Items 1 and 3 of the second review: a session gets RUNS_PER_HOUR runs, its address (x-real-ip, else REMOTE_ADDR) more, so a
    venue's or an office's visitors behind one NAT are not held to one visitor's runs; an IPv6 address counts as its /64."""
    (app, jobs, ledger, stations, store), paid = live
    stub_plans(monkeypatch, stations)
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "plan", 1000)                 # only the visitor limits are under test
    monkeypatch.setitem(ledger.ADDRESS_SLICE, "plan", 1000)
    n, k = ledger.RUNS_PER_HOUR, ledger.RUNS_PER_HOUR_PER_ADDRESS
    assert k > n
    plan = lambda ck, env: J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=issued(ck), env=env))
    one = {"HTTP_X_REAL_IP": "192.0.2.10"}
    ds = [plan("1" * 32, one) for _ in range(n + 1)]
    assert [d["mode"] for d in ds] == ["live"] * n + ["recorded"] and ds[-1]["why"] == stations.why_line("session_limit")
    vercel = {"HTTP_X_REAL_IP": "203.0.113.7", "REMOTE_ADDR": "10.0.0.1"}
    ds = [plan(f"{i:032x}", vercel) for i in range(k + 1)]
    assert [d["mode"] for d in ds] == ["live"] * k + ["recorded"] and ds[-1]["why"] == stations.why_line("ip_limit")
    d = J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=issued("9" * 32), env=vercel))
    assert d["mode"] == "recorded" and d["why"] == stations.why_line("ip_limit")   # FILM's own run counts against the address too
    assert plan("e" * 32, {"HTTP_X_REAL_IP": "203.0.113.8", "REMOTE_ADDR": "10.0.0.1"})["mode"] == "live"
    ds = [plan(f"{i + 100:032x}", {"REMOTE_ADDR": "198.51.100.2"}) for i in range(k + 1)]   # no x-real-ip (local serve): the peer
    assert ds[-2]["mode"] == "live" and ds[-1]["why"] == stations.why_line("ip_limit")
    ds = [plan(f"{i + 200:032x}", {"HTTP_X_REAL_IP": f"2001:db8:1:2::{i + 1:x}"}) for i in range(40)]
    assert [d["mode"] for d in ds] == ["live"] * k + ["recorded"] * (40 - k)            # 40 addresses, one /64, one limit
    assert plan("f" * 32, {"HTTP_X_REAL_IP": "2001:db8:1:3::1"})["mode"] == "live"  # the next /64 is another visitor


def test_the_address_is_kept_only_as_a_salted_hash(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    ledger.start_run("a" * 32, ip="203.0.113.7")
    s = store.get_store(); dump = json.dumps({k: v for k, v in s._d.items()})
    assert "203.0.113.7" not in dump and ledger._address("203.0.113.7") in dump
    salted = ledger._address("203.0.113.7")
    monkeypatch.setenv("STILLHERE_WORKER_TOKEN", "another-secret"); assert ledger._address("203.0.113.7") != salted
    monkeypatch.delenv("STILLHERE_WORKER_TOKEN")                                     # no server secret: plain SHA-256
    assert ledger._address("203.0.113.7") == __import__("hashlib").sha256(b"203.0.113.7").hexdigest()[:16]
    assert ledger._address("2001:db8:1:2::1") == ledger._address("2001:db8:1:2:ffff::9") != ledger._address("2001:db8:1:3::1")
    assert ledger._address("::ffff:192.0.2.1") == ledger._address("192.0.2.1")


def test_one_visitor_takes_a_small_share_of_a_day(live, monkeypatch):
    """Item 1 of the second review: one cookie on one address making a full run every hour of a day keeps at most
    SESSION_DAILY_USD on the ledger; rotating cookies from one address, at most ADDRESS_DAILY_USD; each is a small share of the
    configured daily cap. A refusal says why, with the UTC time it ends."""
    (app, jobs, ledger, stations, store), paid = live
    clock = [86400.0 * 20000 + 60]
    monkeypatch.setattr(ledger, "time", types.SimpleNamespace(time=lambda: clock[0])); monkeypatch.setattr(ledger, "_day", lambda: "d1")
    meta = lambda: float(store.get_store().get("spend:meta:d1") or 0)
    ck = page_cookie(app, "203.0.113.9")
    first = full_run(app, ck, "203.0.113.9")
    assert first == LIVE_RUN
    for _ in range(23):
        clock[0] += 3600; last = full_run(app, ck, "203.0.113.9")
    assert meta() <= ledger.SESSION_DAILY_USD["meta"] + 1e-9 and last["film"] == "recorded"
    assert stations.why_line("session_daily") in last["why"] and paid["sam"] == 3
    one = meta()
    for h in range(24):                                                             # a new cookie every hour, one address
        clock[0] += 3600; last = full_run(app, page_cookie(app, "203.0.113.10"), "203.0.113.10")
    assert meta() - one <= ledger.ADDRESS_DAILY_USD["meta"] + 1e-9 and stations.why_line("ip_daily") in last["why"]
    for p in ledger.DAILY_CAP:                                                       # the shares, of whatever caps are configured
        assert ledger.SESSION_DAILY_USD[p] == pytest.approx(ledger.DAILY_CAP[p] * 0.20)
        assert ledger.ADDRESS_DAILY_USD[p] == pytest.approx(ledger.DAILY_CAP[p] * 0.20)


def test_one_full_run_leaves_every_station_live_for_the_next_visitor(live):
    """Item 2 of the second review: the station caps hold several ordinary runs, so one visitor's run (with every reshuffle) never
    switches a station off for the next one."""
    (app, jobs, ledger, stations, store), paid = live
    for i in range(3):
        ip = f"198.51.100.{i + 4}"; assert full_run(app, page_cookie(app, ip), ip) == LIVE_RUN


def test_caps_and_bounds_are_sized_from_the_configured_daily_caps(web):
    """The station caps and the visitor bounds as numbers: BUILD's hourly cap holds ten runs at a run's worst case (1 + the reshuffle
    seeds = 7 builds), every other station's at least three runs' uses; a session's and an address's daily bound (20% of the
    configured cap each, fifth review, item 1) fit one ordinary run's worst case; one address takes at most a quarter of any station's
    hour. An hour at every cap can now reserve more than a Meta day (FILM unpriced), so the daily cap is what stops many addresses."""
    app, jobs, ledger, stations, store = web
    R, caps, day, sam = stations.RESERVE, ledger.STATION_RUNS_PER_HOUR, ledger.DAILY_CAP, 3 * stations.RESERVE["sam"]
    per_run = {"plan": 1, "build": 1 + len(stations.RESHUFFLE_SEEDS), "audit": 1, "forecast": 4, "film": 1}
    assert per_run["build"] == 7 and caps["build"] >= 10 * per_run["build"], caps
    assert set(caps) == set(per_run) and all(caps[k] >= 3 * per_run[k] for k in per_run), caps
    run_meta = R["plan"] + per_run["build"] * R["transcribe"] + sam
    assert run_meta <= ledger.SESSION_DAILY_USD["meta"] == ledger.ADDRESS_DAILY_USD["meta"] == pytest.approx(0.20 * day["meta"])
    assert R["plan"] + R["audit"] <= min(ledger.SESSION_DAILY_USD["fireworks"], ledger.SESSION_DAILY_USD["baseten"])
    assert caps["plan"] * R["plan"] + caps["audit"] * R["audit"] < min(day["fireworks"], day["baseten"])
    assert caps == {"plan": 120, "build": 70, "audit": 120, "forecast": 30, "film": 10}            # fifth review, items 1 and 8
    slices = {k: ledger.address_slice(k) for k in caps}
    assert slices == {"plan": 10, "build": 17, "audit": 10, "forecast": 7, "film": 2}
    assert all(v <= caps[k] / 4 for k, v in slices.items()) and slices["build"] >= 2 * per_run["build"]


def test_each_station_has_a_global_hourly_cap_then_its_recorded_run(web, monkeypatch):
    """Every visitor together gets STATION_RUNS_PER_HOUR live uses of a station per sliding hour, then the recorded run with its why
    line (FORECAST: the saved forecast, and only jobs that would reach the laptop count); a use the hour refuses is given back."""
    app, jobs, ledger, stations, store = web
    calls = stub_plans(monkeypatch, stations)
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "plan", 2)
    ds = [stations.plan("Plan a story test on chapter 2.", "spark-1.3", f"{i:032x}") for i in range(3)]
    assert [d["mode"] for d in ds] == ["live", "live", "recorded"] and len(calls) == 2
    assert ds[2]["why"] == stations.why_line("station_cap") and ds[2]["label"] == "recorded run"
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "audit", 0)
    with pytest.raises(ledger.Refused) as e:
        ledger.station_hour("audit")
    assert e.value.code == "station_cap" and hours(store, "audit") == 0               # a refused hour is claimed back at once
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "audit", 1)
    ledger.station_hour("audit"); assert hours(store, "audit") == 1


def test_forecasts_hourly_cap_counts_only_jobs_for_the_laptop(web, monkeypatch):
    """Item 10 of the second review: a cached forecast costs no paid call and no laptop time, so it never counts toward FORECAST's
    hourly cap; a job that would reach the laptop does, and past the cap the visitor gets the saved forecast with a line that says
    so (not "recorded run")."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    r = jobs.recipe("alice_ch02", "reversed", 0); key = jobs.cache_key(r)
    store.get_store().set(f"result:{key}", json.dumps({**result_for(key), "recipe": r}))
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "forecast", 1)
    for i in range(5):
        sid = f"{i:032x}"; run = planned_run(ledger, store, sid, "alice_ch02", "reversed")
        assert stations.forecast(run, sid, "alice_ch02", "reversed", 0)["state"] == "cached"
    assert stations.forecast(planned_run(ledger, store, "a" * 32), "a" * 32, *ROUTE.values(), 2)["state"] == "queued"
    d = stations.forecast(planned_run(ledger, store, "b" * 32), "b" * 32, *ROUTE.values(), 3)
    assert d["state"] == "offline" and d["why"] == stations.why_line("station_cap", shown=stations.SAVED_SHOWN)
    assert "saved forecast" in d["why"] and "recorded run" not in d["why"] and len(store.get_store().lrange("jobs:queue")) == 1


def test_hourly_limits_hold_over_every_true_hour(web, monkeypatch):
    """Item 2 of the third review: an adversary fills the end of an hour, then tries every second for two hours. No true hour (any
    3600 s span) lets more than the limit through, and the window is no stricter than an hour: the next event passes the moment the
    first is an hour old (the old weighted buckets let 9 of 5 and 139 of 70 through, and blocked for 72 minutes)."""
    app, jobs, ledger, stations, store = web
    clock = [3600.0 * 500_000 - 1.0]
    monkeypatch.setattr(ledger, "time", types.SimpleNamespace(time=lambda: clock[0]))
    for limit in (3, 5, 20, 40, 70):
        ok, start = [], clock[0]
        for _ in range(3 * limit):
            if ledger._window(f"win:t{limit}", limit):
                ok.append(clock[0])
        while clock[0] < start + 7200:
            clock[0] += 1.0
            if ledger._window(f"win:t{limit}", limit):
                ok.append(clock[0])
        worst = max(sum(1 for u in ok if t <= u < t + 3600) for t in ok)
        assert worst == limit and ok[limit] - ok[0] == 3600, (limit, worst, ok[limit] - ok[0])
        clock[0] += 7200


def test_no_true_hour_reserves_more_than_the_station_caps_allow(live, monkeypatch):
    """Item 2 of the third review, at the configured caps: visitors from many addresses (so no per-visitor bound binds) fill the end
    of an hour, then come every 20 s for two hours. In no true hour do PLAN, BUILD and FILM reserve more Meta spend than the station
    caps allow (PLAN 120 x $0.05 + BUILD 70 x $0.10 + FILM 10 x 3 x $0.50 = $28.00, of which PLAN's $6.00 settles to its tokens).
    That is more than a configured Meta day, so with FILM unpriced the daily cap, not the hourly caps, stops many addresses."""
    (app, jobs, ledger, stations, store), paid = live
    caps, R = ledger.STATION_RUNS_PER_HOUR, stations.RESERVE
    bound = caps["plan"] * R["plan"] + caps["build"] * R["transcribe"] + caps["film"] * 3 * R["sam"]
    monkeypatch.setitem(ledger.DAILY_CAP, "meta", 1e6); monkeypatch.setitem(ledger.TOTAL_CAP, "meta", 1e6)   # only the hours bind
    clock = [3600.0 * 500_000 - 1.0]
    monkeypatch.setattr(ledger, "time", types.SimpleNamespace(time=lambda: clock[0]))
    log, real = [], ledger.reserve
    monkeypatch.setattr(ledger, "reserve", lambda p, usd, run_id=None: (real(p, usd, run_id), log.append((clock[0], p, usd)))[0])
    n = [0]

    def visitor():
        n[0] += 1; sid, ip = f"{n[0]:032x}", f"10.{n[0] // 65536}.{n[0] // 256 % 256}.{n[0] % 256}"
        run = stations.plan("Plan a story test on chapter 2.", "spark-1.3", sid, ip=ip).get("run_id")
        for _ in range(7):
            stations.build(run, sid, *ROUTE.values(), 20261028)
        stations.film(run, sid, "face", ip=ip)
    for _ in range(60):
        visitor()
    start = clock[0]
    while clock[0] < start + 7200:
        clock[0] += 20; visitor()
    meta = [(t, usd) for t, p, usd in log if p == "meta"]
    worst = max(sum(u for v, u in meta if t <= v < t + 3600) for t, _ in meta)
    assert worst == pytest.approx(bound) and worst <= bound + 1e-9


def test_the_run_counter_holds_under_concurrent_starts(web):
    """Item 3: twenty starts at once from one session get exactly RUNS_PER_HOUR runs. Every read answers 20 ms late (a network
    round trip), so a read-then-write count lets every start through; one INCR per start cannot be raced."""
    app, jobs, ledger, stations, store = web

    class SlowReads(store.MemoryStore):
        def get(self, k):
            v = super().get(k); time.sleep(0.02); return v

        def lrange(self, k, start=0, stop=-1):
            v = super().lrange(k, start, stop); time.sleep(0.02); return v
    store.set_store(SlowReads())
    gate, got = threading.Barrier(20), []

    def start():
        gate.wait()
        try:
            got.append(ledger.start_run("c" * 32))
        except ledger.Refused as e:
            got.append(e.code)
    ts = [threading.Thread(target=start) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(g != "session_limit" for g in got) == ledger.RUNS_PER_HOUR and got.count("session_limit") == 20 - ledger.RUNS_PER_HOUR


def test_the_window_is_one_sorted_set_pipeline(web, monkeypatch):
    """Item 2 of the third review: an event in a sliding hour is ZREMRANGEBYSCORE (older than the hour), ZADD, ZCARD and EXPIRE in one
    Upstash pipeline, and a refused one is claimed back with ZREM; the in-memory store keeps the same set."""
    app, jobs, ledger, stations, store = web
    sent = upstash(monkeypatch, store)
    clock = [1_000_000.0]
    monkeypatch.setattr(ledger, "time", types.SimpleNamespace(time=lambda: clock[0]))
    assert [bool(ledger._window("win:t", 2)) for _ in range(3)] == [True, True, False]
    z = sent[0]
    assert [c[0] for c in z] == ["ZREMRANGEBYSCORE", "ZADD", "ZCARD", "EXPIRE"] and z[0][1:3] == ["win:t", "-inf"]
    assert float(z[0][3]) == clock[0] - 3600 and float(z[1][2]) == clock[0] and z[3][2] == "3600"
    assert len(sent) == 4 and sent[3] == ["ZREM", "win:t", sent[2][1][3]]            # the refused third, claimed back
    clock[0] += 3599; assert not ledger._window("win:t", 2)
    clock[0] += 1; assert ledger._window("win:t", 2)                                # the first two are an hour old: room again


def test_film_settles_each_completed_frame_to_sams_price(web, monkeypatch):
    """Item 4, on stubs (opencv, face_edit, meta-sam-parser and SAM itself): each frame reserves first; a completed frame settles to
    SAM_USD_PER_FRAME; a frame whose response did not complete keeps its reservation; the receipt says which is which."""
    app, jobs, ledger, stations, store = web
    import numpy as np
    import stillhere
    from web import sam
    fe = types.ModuleType("stillhere.face_edit")
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imread=lambda p: np.zeros((4, 4, 3), np.uint8), cvtColor=lambda a, c: a,
                                                                  COLOR_BGR2RGB=4))
    monkeypatch.setitem(sys.modules, "meta_sam_parser", types.ModuleType("meta_sam_parser"))
    monkeypatch.setitem(sys.modules, "stillhere.face_edit", fe); monkeypatch.setattr(stillhere, "face_edit", fe, raising=False)
    status = iter(["response.completed", "response.completed", "response.failed"])
    monkeypatch.setattr(sam, "segment", lambda clip, q: {"objects": [], "usage": {}, "status": next(status)})
    monkeypatch.setattr(stations, "SAM_USD_PER_FRAME", 0.02, raising=False)
    monkeypatch.setattr(ledger, "STATION_RUNS_PER_HOUR", {**getattr(ledger, "STATION_RUNS_PER_HOUR", {}), "film": 2}, raising=False)
    d = stations.film(None, "a" * 32, "face")
    assert d["mode"] == "live" and len(d["frames"]) == 3
    assert float(store.get_store().get("spend:meta:total")) == pytest.approx(2 * 0.02 + 0.50)
    r = stations.get_receipt(d["receipt_id"]); assert r["usd"] == pytest.approx(0.04) and r["usd_reserved"] == pytest.approx(0.50)
    monkeypatch.setattr(stations, "SAM_USD_PER_FRAME", None); status = iter(["response.completed"] * 3)
    d = stations.film(None, "b" * 32, "face")                                        # no price set: every reservation stands
    assert d["mode"] == "live"; r = stations.get_receipt(d["receipt_id"]); assert r["usd"] is None and r["usd_reserved"] == pytest.approx(1.50)


def test_build_and_forecast_hold_to_the_runs_plan(web, monkeypatch):
    """Item 5: BUILD and FORECAST run only the clip and condition the run's live plan routed to, and nothing for a declined plan;
    a refused request costs no call and no queue entry and leaves the run its uses."""
    app, jobs, ledger, stations, store = web
    heard = []
    monkeypatch.setattr(stations.meta_client, "transcribe", lambda audio, **k: heard.append(audio) or {"transcript": ""})
    call(app, "POST", "/api/worker/claim", {}, auth=W)                              # worker online: a forecast would queue
    run = planned_run(ledger, store, "a" * 32, "alice_ch02", "sentence_scramble")
    for clip, cond in (("alice_ch03", "sentence_scramble"), ("alice_ch02", "word_scramble")):
        b, f = stations.build(run, "a" * 32, clip, cond, 20261028), stations.forecast(run, "a" * 32, clip, cond, 20261028)
        assert b["mode"] == "recorded" and b["why"] == stations.why_line("not_the_plan")
        assert f["state"] == "offline" and f["why"] == stations.why_line("not_the_plan", shown=stations.SAVED_SHOWN)
        assert "saved forecast" in f["why"] and "recorded run" not in f["why"]
    declined = planned_run(ledger, store, "a" * 32, decline="clinical_out_of_scope")
    b = stations.build(declined, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)
    f = stations.forecast(declined, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)
    assert b["why"] == stations.why_line("plan_declined") and f["why"] == stations.why_line("plan_declined", shown=stations.SAVED_SHOWN)
    assert heard == [] and store.get_store().lrange("jobs:queue") == []
    assert stations.build(run, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)["mode"] == "live" and len(heard) == 1
    assert stations.forecast(run, "a" * 32, "alice_ch02", "sentence_scramble", 20261028)["state"] == "queued"


def test_the_job_queue_holds_six_then_says_busy(web):
    """Item 6: at most 6 jobs wait; the next visitor gets "busy" with its line (and the saved run where one exists), no job."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    out = []
    for i in range(7):
        sid = f"{i:032x}"; out.append(stations.forecast(planned_run(ledger, store, sid), sid, "alice_ch02", "sentence_scramble", 2))
    assert [d["state"] for d in out] == ["queued"] * 6 + ["busy"] and out[-1]["why"] == jobs.BUSY_LINE and "job_id" not in out[-1]
    assert len(store.get_store().lrange("jobs:queue")) == 6 and "fallback" in out[-1]
    call(app, "POST", "/api/worker/claim", {}, auth=W)                              # the worker takes one: room for one more
    assert stations.forecast(planned_run(ledger, store, "f" * 32), "f" * 32, "alice_ch02", "sentence_scramble", 2)["state"] == "queued"


def test_an_over_cap_submit_leaves_no_id_in_the_queue(web, monkeypatch):
    """Item 11 of the second review: twenty forecasts at once, with every read of the queue 50 ms late, queue six jobs and leave six
    ids: a submit past the cap takes its own id back out, so the queue never reads full while it has room."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    S = store.get_store(); real = S.lrange
    monkeypatch.setattr(S, "lrange", lambda k, start=0, stop=-1: (real(k, start, stop), time.sleep(0.05))[0])
    runs = [(f"{i:032x}", planned_run(ledger, store, f"{i:032x}")) for i in range(20)]
    gate, out = threading.Barrier(20), []

    def go(sid, run):
        gate.wait(); out.append(stations.forecast(run, sid, *ROUTE.values(), 2)["state"])
    ts = [threading.Thread(target=go, args=r) for r in runs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    q = real("jobs:queue")
    assert out.count("queued") == 6 and out.count("busy") == 14 and len(q) == 6 and all(S.get(f"job:{j}") for j in q)


def test_posts_need_json_and_our_own_origin(web, monkeypatch):
    """Item 7: a POST needs Content-Type application/json (a cross-site form or no-preflight fetch cannot send it), and an Origin,
    when sent, must be this page's own: the host the request was addressed to, with the scheme the cookie's Secure flag reads, or one
    of Vercel's names for this deployment."""
    app, jobs, ledger, stations, store = web
    b, local = {"old_bug": True}, {"HTTP_HOST": "127.0.0.1:8780"}
    for ct in ("", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"):
        assert call(app, "POST", "/api/build", b, env={**local, "CONTENT_TYPE": ct})[0] == 415, ct
    assert call(app, "POST", "/api/worker/claim", {}, auth=W, env={"CONTENT_TYPE": "text/plain"})[0] == 415
    assert call(app, "POST", "/api/build", b, env={**local, "CONTENT_TYPE": "application/json; charset=utf-8"})[0] == 200
    assert call(app, "POST", "/api/build", b, env={**local, "HTTP_ORIGIN": "http://127.0.0.1:8780"})[0] == 200
    assert call(app, "POST", "/api/build", b, env={"HTTP_HOST": "localhost:8780", "HTTP_ORIGIN": "http://localhost:8780"})[0] == 200
    for o in ("https://evil.example", "null", "http://127.0.0.1:8781", "https://127.0.0.1:8780", "http://127.0.0.1:8780.evil.example"):
        code, _, out = call(app, "POST", "/api/build", b, env={**local, "HTTP_ORIGIN": o}); assert code == 403 and b"pass" not in out, o
    preview = {"HTTP_HOST": "stillhere-git-hardening-yatsu.vercel.app", "HTTP_X_FORWARDED_PROTO": "https"}
    assert call(app, "POST", "/api/build", b, env={**preview, "HTTP_ORIGIN": "https://stillhere-git-hardening-yatsu.vercel.app"})[0] == 200
    assert call(app, "POST", "/api/build", b, env={**preview, "HTTP_ORIGIN": "http://stillhere-git-hardening-yatsu.vercel.app"})[0] == 403
    spoof = {"HTTP_HOST": "stillhere.vercel.app", "HTTP_X_FORWARDED_PROTO": "https", "HTTP_X_FORWARDED_HOST": "evil.example"}
    assert call(app, "POST", "/api/build", b, env={**spoof, "HTTP_ORIGIN": "https://evil.example"})[0] == 403   # a client header
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "stillhere.vercel.app")      # production, if the function sees another Host
    assert call(app, "POST", "/api/build", b, env={**preview, "HTTP_ORIGIN": "https://stillhere.vercel.app"})[0] == 200
    assert call(app, "GET", "/api/status", env={"HTTP_ORIGIN": "https://evil.example"})[0] == 200          # GETs are not checked


def test_status_is_cached_five_seconds_and_names_no_budget(web):
    """Item 8: /api/status is the same for every visitor, so a shared cache may keep it 5 s; it carries no budget and no cookie."""
    app, jobs, ledger, stations, store = web
    code, h, out = call(app, "GET", "/api/status", cookie="")
    assert code == 200 and h["Cache-Control"] == "s-maxage=5" and "budget" not in json.loads(out) and "Set-Cookie" not in h
    code, h, _ = call(app, "GET", "/api/evidence", cookie="")                       # only a page load issues a cookie
    assert h["Cache-Control"] == "no-store" and "Set-Cookie" not in h


def test_static_files_are_cached_a_day_and_the_pages_are_not(web):
    """Static assets and media get max-age=86400 and no session cookie (a shared cache may keep them); the pages get no-cache, so a
    later deploy's page is always fetched. No static answer sets the cookie: the page's first call does (POST /api/visitor)."""
    app, jobs, ledger, stations, store = web
    for p in ("/app.js", "/app.css", "/pilot.css", "/receipt.js", "/hero.png", "/fonts/NotoSerif-Regular.woff2", "/media/old_bug.wav"):
        code, h, _ = call(app, "GET", p, cookie="")
        assert code == 200 and h["Cache-Control"] == "max-age=86400" and "Set-Cookie" not in h, p
    for p in ("/", "/receipt.html"):
        code, h, _ = call(app, "GET", p, cookie="")
        assert code == 200 and h["Cache-Control"] == "no-cache" and "Set-Cookie" not in h, p        # the page's first call sets it
    code, h, _ = call(app, "GET", "/no-such-file.txt"); assert code == 404 and "Cache-Control" not in h


def test_visitor_ids_come_from_the_pages_first_call(web, monkeypatch):
    """Items 3 and 5 of the third review and item 3 of the fourth: a page load touches no store and issues nothing, so crawlers and
    scripts that never run the page's script spend no ids. Each page's first call (POST /api/visitor) gets a fresh id (SET NX on its
    own key), at most RUNS_PER_HOUR_PER_ADDRESS an hour per address; only cookies the server issued are honored. Cookieless requests
    share the address's one session, and its refusal says so, not "this session"."""
    app, jobs, ledger, stations, store = web
    stub_plans(monkeypatch, stations)
    for _ in range(20):
        h = call(app, "GET", "/", cookie="", env={"HTTP_X_REAL_IP": "203.0.113.30"})[1]; assert "Set-Cookie" not in h
    assert [k for k in store.get_store()._d if k.startswith(("vid:", "visitor:", "win:"))] == ["visitor:" + "a" * 32]   # the fixture's
    here = {"HTTP_X_REAL_IP": "203.0.113.30"}
    a, b = page_cookie(app, "203.0.113.30"), page_cookie(app, "203.0.113.30")
    assert a != b                                                                    # two people behind one NAT: two visitors
    gate, out = threading.Barrier(20), []

    def go():
        gate.wait(); code, h, body = call(app, "POST", "/api/plan", PLAN_BODY, cookie="", env=here); out.append((json.loads(body), h))
    ts = [threading.Thread(target=go) for _ in range(20)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(d["mode"] == "live" for d, _ in out) == ledger.RUNS_PER_HOUR and not any("Set-Cookie" in h for _, h in out)
    assert {d.get("why") for d, _ in out if d["mode"] != "live"} == {stations.why_line("shared_limit")}
    assert "session" not in stations.why_line("shared_limit") and "network address" in stations.why_line("shared_limit")
    d = J(call(app, "POST", "/api/plan", PLAN_BODY, cookie="sh_sid=" + "7" * 32, env=here))      # made up, never issued: the address
    assert d["why"] == stations.why_line("shared_limit")
    assert J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=a, env=here))["mode"] == "live"   # the issued cookie: its own runs
    for _ in range(ledger.RUNS_PER_HOUR_PER_ADDRESS - 2):                             # an address mints at most 20 ids an hour
        assert "Set-Cookie" in call(app, "POST", "/api/visitor", {}, cookie="", env=here)[1]
    h = call(app, "POST", "/api/visitor", {}, cookie="", env=here)[1]; assert "Set-Cookie" not in h
    sent = upstash(monkeypatch, store)
    sid = call(app, "POST", "/api/visitor", {}, cookie="", env={"HTTP_X_REAL_IP": "192.0.2.50"})[1]["Set-Cookie"].split(";")[0][7:]
    assert [c[0] for c in sent[0]] == ["ZREMRANGEBYSCORE", "ZADD", "ZCARD", "EXPIRE"] and sent[0][1][1].startswith("win:ids:ip:")
    assert sent[1] == ["SET", f"visitor:{sid}", "1", "EX", "86400", "NX"] and len(sent) == 2   # SET NX on the new id's own key


def test_the_page_never_needs_the_store(web, monkeypatch, capsys):
    """Item 3 of the third review: with the store failing (an outage, a spent quota, a timeout) the pages and assets still load, and
    no page load reaches the store at all; the page's first call answers with no cookie (fourth review, item 4), logged by type."""
    app, jobs, ledger, stations, store = web
    sent = upstash(monkeypatch, store, fail=True)
    for path, ck in (("/", ""), ("/", "sh_sid=" + "e" * 32), ("/receipt.html", ""), ("/app.js", ""), ("/app.css", "")):
        assert call(app, "GET", path, cookie=ck)[0] == 200, path
    assert sent == []
    code, h, _ = call(app, "POST", "/api/visitor", {}, cookie="")
    assert code == 200 and "Set-Cookie" not in h and "ConnectError" in capsys.readouterr().err


def test_the_first_visit_cookie_comes_with_the_page(live):
    """The cookie from the page's first call is the one every later call carries: PLAN keeps its run for BUILD, and no other call
    hands out a second cookie."""
    (app, jobs, ledger, stations, store), paid = live
    ck = page_cookie(app, "203.0.113.50")
    code, hp, out = call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck, env={"HTTP_X_REAL_IP": "203.0.113.50"}); d = json.loads(out)
    he = call(app, "GET", "/api/evidence", cookie=ck)[1]
    assert "Set-Cookie" not in hp and "Set-Cookie" not in he and d["mode"] == "live"
    b = J(call(app, "POST", "/api/build", {"run_id": d["run_id"], **ROUTE, "seed": 20261028}, cookie=ck))
    assert b["mode"] == "live"


def test_a_refused_visitor_takes_no_shared_hour(live, monkeypatch):
    """Items 1 and 4 of the third review: every per-visitor check (its runs, the session's and the address's daily share, the spend
    caps) comes before a station's shared hour, and a refusal gives back everything the request took. A visitor with nothing left
    cannot switch a station off for others, and a station whose hour is full costs a visitor none of its runs."""
    (app, jobs, ledger, stations, store), paid = live
    monkeypatch.setattr(ledger, "_day", lambda: "d1"); S = store.get_store()
    ip = "203.0.113.77"; ck = page_cookie(app, ip)
    S.set(f"vspend:meta:{ck.split('=')[1]}:d1", str(ledger.SESSION_DAILY_USD["meta"]))      # this session's day is spent
    outs = [J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=ck, env={"HTTP_X_REAL_IP": ip})) for _ in range(3)]
    assert [o["why"] for o in outs] == [stations.why_line("session_daily")] * 3 and paid["sam"] == 0 and hours(store, "film") == 0
    ip2 = "198.51.100.200"
    assert J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=page_cookie(app, ip2), env={"HTTP_X_REAL_IP": ip2}))["mode"] == "live"
    ip3, env3 = "203.0.113.78", {"HTTP_X_REAL_IP": "203.0.113.78"}
    runs = [(c, J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=c, env=env3))["run_id"]) for c in (issued("1" * 32), issued("2" * 32))]
    S.set(f"vspend:meta:ip:{ledger._address(ip3)}:d1", str(ledger.ADDRESS_DAILY_USD["meta"]))   # now the address's day is spent
    before = (paid["plan"], paid["transcribe"], hours(store, "plan"), hours(store, "build"))
    whys = {J(call(app, "POST", "/api/build", {"run_id": rid, **ROUTE, "seed": 20261028}, cookie=c, env=env3))["why"] for c, rid in runs
            for _ in range(7)}
    whys |= {J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=issued(f"{i + 3:032x}"), env=env3))["why"] for i in range(8)}
    assert whys == {stations.why_line("ip_daily")} and (paid["plan"], paid["transcribe"], hours(store, "plan"), hours(store, "build")) == before
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "plan", hours(store, "plan"))     # PLAN's hour is full
    ip4, ck4 = "203.0.113.79", issued("9" * 32)
    ds = [J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck4, env={"HTTP_X_REAL_IP": ip4})) for _ in range(ledger.RUNS_PER_HOUR)]
    assert {d["why"] for d in ds} == {stations.why_line("station_cap")} and not S._d.get(f"win:runs:{ck4.split('=')[1]}")
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "plan", 40)                      # the hour frees up: all five runs are still there
    ds = [J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck4, env={"HTTP_X_REAL_IP": ip4}))["mode"] for _ in range(ledger.RUNS_PER_HOUR)]
    assert ds == ["live"] * ledger.RUNS_PER_HOUR


def test_forecast_gives_back_what_a_refused_job_took(web, monkeypatch):
    """Item 6 of the third review: FORECAST keeps the run's use and the station's hour only for a job that reaches the queue; past
    the hour, on the queue race, offline, busy or cached, both are given back."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "forecast", 0)
    sid = "c" * 32; run = planned_run(ledger, store, sid)
    assert {stations.forecast(run, sid, *ROUTE.values(), s)["why"] for s in (1, 2, 3, 4)} == {
        stations.why_line("station_cap", shown=stations.SAVED_SHOWN)}
    monkeypatch.setitem(ledger.STATION_RUNS_PER_HOUR, "forecast", 30)
    assert stations.forecast(run, sid, *ROUTE.values(), 2)["state"] == "queued"     # the run still had its four uses
    S = store.get_store()
    for i in range(6):
        S.rpush("jobs:queue", f"{i:016x}")
    real = S.lrange; monkeypatch.setattr(S, "lrange", lambda k, start=0, stop=-1: [])       # as if six others pushed after the check
    sid2 = "d" * 32; run2 = planned_run(ledger, store, sid2)
    took = hours(store, "forecast")
    assert stations.forecast(run2, sid2, *ROUTE.values(), 3)["state"] == "busy" and hours(store, "forecast") == took
    assert float(S.get(f"use:{run2}:forecast") or 0) == 0 and len(real("jobs:queue")) == 7


def test_film_reserves_every_frame_first_and_settles_each(live, monkeypatch):
    """Item 7 of the third review: FILM reserves all three frames before its first SAM call, so a session that cannot afford them
    pays for none; each frame then settles on its own: a frame that never answered costs nothing, and frames not reached are freed."""
    (app, jobs, ledger, stations, store), paid = live
    from web import sam
    monkeypatch.setattr(ledger, "_day", lambda: "d1"); S = store.get_store()
    ip = "203.0.113.81"; ck = page_cookie(app, ip); sid = ck.split("=")[1]
    S.set(f"vspend:meta:{sid}:d1", str(ledger.SESSION_DAILY_USD["meta"] - 1.0))         # $1.00 of the session's share left
    d = J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=ck, env={"HTTP_X_REAL_IP": ip}))
    assert d["why"] == stations.why_line("session_daily") and paid["sam"] == 0
    assert float(S.get(f"vspend:meta:{sid}:d1")) == pytest.approx(ledger.SESSION_DAILY_USD["meta"] - 1.0)
    calls = []

    def flaky(clip, q):
        calls.append(clip)
        if len(calls) == 2:
            raise httpx.ConnectError("frame 2 never answered")
        return {"objects": [], "usage": {}, "status": "response.completed"}
    monkeypatch.setattr(sam, "segment", flaky)
    ck2 = page_cookie(app, "198.51.100.81"); sid2 = ck2.split("=")[1]
    d = J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=ck2, env={"HTTP_X_REAL_IP": "198.51.100.81"}))
    assert d["why"] == stations.why_line("api_error") and len(calls) == 2
    assert float(S.get(f"vspend:meta:{sid2}:d1")) == pytest.approx(stations.RESERVE["sam"])      # frame 1 stands; 2 and 3 freed


def test_each_chat_call_is_bounded_by_its_reservation(web, monkeypatch, capsys):
    """Item 8 of the third review: every chat call made under a reservation carries max_tokens, so even a provider that bills the
    most it can (a prompt token per byte of the request, every allowed output token) stays inside it, across the two calls PLAN may
    make; a reservation too small for the prompt sends nothing. settle() never moves spend past a cap: an over-run is logged and held
    at the cap, so the next call is refused instead."""
    app, jobs, ledger, stations, store = web
    from stillhere import egress, meta_client, planner, planner_v21 as P
    sent = []

    def greedy(url, json=None, **kw):                     # a first call billed as measured, then the worst a bounded call can cost
        sent.append(json)
        u = ({"prompt_tokens": 900, "completion_tokens": 1400} if len(sent) == 1 else
             {"prompt_tokens": len(__import__("json").dumps(json["messages"]).encode()), "completion_tokens": json["max_tokens"]})
        return httpx.Response(200, request=httpx.Request("POST", url), json={"usage": u, "choices": [{"message": {"content": '{"notes": []}'}}]})
    monkeypatch.setattr(stations, "_post", greedy)
    req = egress.Request(model=P.MODELS["spark-open"], system=[planner.public("audit_system")], user=[egress.TableEnums(enums={
        "population": ["healthy_adults"]})], schema_name="audit_notes", json_schema=stations.AUDIT_SCHEMA)
    with stations.capture(stations.RESERVE["plan"]) as usage:
        meta_client.call(req); meta_client.call(req)
    assert len(sent) == 2 and all(isinstance(b["max_tokens"], int) and b["max_tokens"] > 0 for b in sent)
    assert stations.chat_cost(usage) <= stations.RESERVE["plan"] and sent[1]["max_tokens"] < sent[0]["max_tokens"]
    sent.clear()
    with stations.capture(0.0001):
        with pytest.raises(Exception):
            meta_client.call(req)
    assert sent == []
    monkeypatch.setitem(ledger.DAILY_CAP, "fireworks", 0.10)
    r = ledger.reserve("fireworks", 0.05); ledger.settle(r, 0.30)                    # a call that cost more than it reserved
    assert float(store.get_store().get(f"spend:fireworks:{ledger._day()}")) == pytest.approx(0.10)
    with pytest.raises(ledger.Refused) as e:
        ledger.reserve("fireworks", 0.01)
    assert e.value.code == "spend_cap" and "more than its reservation" in capsys.readouterr().err


def test_saved_forecasts_say_why_and_the_old_bug_audio_is_named_by_hash(web):
    """Item 9 of the third review: the worker-offline and failed or expired job fallbacks carry a why-line; the old-bug recording
    reaches the page as /media/old_bug.wav?v=<sha8>, so its day-long cache is safe."""
    app, jobs, ledger, stations, store = web
    import hashlib
    run = planned_run(ledger, store, "a" * 32)
    d = stations.forecast(run, "a" * 32, *ROUTE.values(), 2)                         # no worker online
    assert d["state"] == "offline" and d["why"] == stations.why_line("worker_offline", shown=stations.SAVED_SHOWN)
    assert stations.hours_et() in d["why"]
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    jid = stations.forecast(planned_run(ledger, store, "b" * 32), "b" * 32, *ROUTE.values(), 3)["job_id"]
    j = json.loads(call(app, "POST", "/api/worker/claim", {}, auth=W)[2])["job"]
    call(app, "POST", f"/api/worker/jobs/{j['id']}/fail", {"reason": "oom"}, auth=W)
    s = J(call(app, "GET", f"/api/forecast/{jid}")); assert s["status"] == "failed" and s["why"] == jobs.FAILED_LINE
    url = stations.old_bug()["audio_url"]; path, v = url.split("?v=")
    assert path == "/media/old_bug.wav" and hashlib.sha256(call(app, "GET", path)[2]).hexdigest()[:8] == v


def test_the_pages_name_each_asset_by_its_content_hash(web, monkeypatch, tmp_path):
    """Item 7 of the second review: the served pages and style sheets name every local asset as path?v=<first 8 hex of the SHA-256
    of what that path serves>, generated as they are served, so a deploy that changes a file changes its URL; nothing else in the
    page changes."""
    app, jobs, ledger, stations, store = web
    import hashlib
    seen = {}
    for page, f in (("/", "index.html"), ("/receipt.html", "receipt.html"), ("/app.css", "app.css")):
        text = call(app, "GET", page)[2].decode()
        refs = re.findall(r'(?:(?:href|src)="|url\()(/[^"?#)]+)\?v=([0-9a-f]{8})', text)
        assert refs and re.sub(r"\?v=[0-9a-f]{8}", "", text) == (REPO / "web/public" / f).read_text(), page
        assert not re.search(r'(?:(?:href|src)="|url\()/[^"?#)]+\.(?:css|js|png|woff2)(?=[")])', text), page
        for path, v in refs:
            assert hashlib.sha256(call(app, "GET", path)[2]).hexdigest()[:8] == v, (page, path); seen[path] = v
    assert {"/pilot.css", "/app.css", "/app.js", "/hero.png", "/receipt.js"} <= set(seen) and sum(p.startswith("/fonts/") for p in seen) == 5
    pub = tmp_path / "public"; shutil.copytree(REPO / "web/public", pub)
    (pub / "app.js").write_text((pub / "app.js").read_text() + "\n// a later deploy\n"); monkeypatch.setattr(app, "PUBLIC", pub)
    assert re.search(r'/app\.js\?v=([0-9a-f]{8})', call(app, "GET", "/")[2].decode()).group(1) != seen["/app.js"]


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
@pytest.mark.parametrize("case", ["queue_full", "own_job_failed"])
def test_the_page_shows_the_why_line_beside_a_saved_forecast(web, monkeypatch, tmp_path, case):
    """In the real page: a FORECAST that shows a saved forecast shows, beside it, the line saying why (the recorded-run note's look):
    a full queue (item 10 of the second review), and a session's own job, which the page follows (not "the worker is offline") until
    it fails (item 9 of the third review). The probe page is the served page plus one test script that starts FORECAST."""
    app, jobs, ledger, stations, store = web
    fb = {**result_for("k" * 64), "recipe": {"clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 20261028}, "note": None}
    base = lambda clip: {"chip": stations.chip(clip), "end_line": stations.end_line(clip), "word_rate_line": stations.WORD_RATE_LINE,
                         "reversed_line": None, "hours_et": stations.hours_et()}
    if case == "queue_full":
        want = jobs.BUSY_LINE
        monkeypatch.setattr(stations, "forecast", lambda run, sid, clip, cond, seed: {**base(clip), "state": "busy", "why": want, "fallback": fb})
    else:
        want = jobs.FAILED_LINE
        monkeypatch.setattr(stations, "forecast", lambda run, sid, clip, cond, seed: {**base(clip), "state": "busy", "job_id": "0" * 16})
        monkeypatch.setattr(jobs, "status", lambda jid: {"status": "failed", "recipe": fb["recipe"], "events": [], "fallback": fb, "why": want})
    probe_js = (b'addEventListener("load", () => { S.route = {kind: "live", clip_id: "alice_ch02", condition: "sentence_scramble", '
                b'seed: 2}; S.seed = 2; doForecast(); });')

    def wrapper(env, start):
        if env["PATH_INFO"] == "/probe.js":
            start("200 OK", [("Content-Type", "text/javascript")]); return [probe_js]
        if env["PATH_INFO"] == "/probe.html":
            got = {}
            body = b"".join(app.app({**env, "PATH_INFO": "/"}, lambda s, h: got.update(s=s, h=h)))
            body = body.replace(b"</body>", b'<script src="/probe.js"></script></body>')
            start(got["s"], [(k, v) for k, v in got["h"] if k != "Content-Length"]); return [body]
        return app.app(env, start)

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    srv = make_server("127.0.0.1", port, wrapper, handler_class=Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={tmp_path / 'p'}",
                              "--virtual-time-budget=6000", "--dump-dom", f"http://127.0.0.1:{port}/probe.html"],
                             capture_output=True, text=True, timeout=120).stdout
    finally:
        srv.shutdown()
    box = out[out.index('id="forecast-out"'):]
    assert "Cached forecast, computed 2026-10-30T14:00:00Z" in box, out[-400:]
    assert want.replace("'", "&#39;") in box or want in box, box[:600]


def test_a_static_error_is_a_500_not_a_crash(web, monkeypatch, capsys):
    """Item 10: an error while serving a static file is answered (500) and logged by type only, like a route's."""
    app, jobs, ledger, stations, store = web

    def broken(path):
        raise PermissionError(f"/private/{path}")
    monkeypatch.setattr(app, "static", broken)
    code, h, out = call(app, "GET", "/app.js")
    err = capsys.readouterr().err
    assert code == 500 and b"private" not in out and "PermissionError" in err and "private" not in err and "app.js" not in err


# ---- fourth review (Oct 31) ------------------------------------------------------------------------------------------------------
def test_one_address_never_switches_a_station_off_for_another(live, monkeypatch):
    """Item 1: every station gives each network address at most a quarter of its hour (address_slice; FILM 1), and FORECAST at most
    two waiting jobs, so one address with many cookies, using every station all it can, leaves every station live for another (SAM's
    price set, so FILM's slice binds before the address's daily share)."""
    (app, jobs, ledger, stations, store), paid = live
    monkeypatch.setattr(stations, "SAM_USD_PER_FRAME", 0.05)
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    ip, env = "203.0.113.60", {"HTTP_X_REAL_IP": "203.0.113.60"}
    got = {k: 0 for k in ledger.STATIONS}
    for i in range(8):
        ck = issued(f"{i + 1:032x}")
        for _ in range(ledger.RUNS_PER_HOUR):
            d = J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck, env=env)); got["plan"] += d["mode"] == "live"
            rid = d.get("run_id")
            got["build"] += sum(J(call(app, "POST", "/api/build", {"run_id": rid, **ROUTE, "seed": 20261028}, cookie=ck, env=env))["mode"]
                                == "live" for _ in range(7))
            got["audit"] += J(call(app, "POST", "/api/audit", {"run_id": rid}, cookie=ck, env=env))["mode"] == "live"
            got["forecast"] += sum(call(app, "POST", "/api/forecast", {"run_id": rid, **ROUTE, "seed": s}, cookie=ck, env=env)[0] == 202
                                   for s in (1, 2, 3, 4))
            got["film"] += J(call(app, "POST", "/api/film", {"run_id": rid, "concept": "face"}, cookie=ck, env=env))["mode"] == "live"
    assert all(got[k] <= ledger.address_slice(k) for k in got) and got["film"] == 2 and got["forecast"] <= 2, got
    them, ck = {"HTTP_X_REAL_IP": "198.51.100.61"}, page_cookie(app, "198.51.100.61")
    assert full_run(app, ck, "198.51.100.61") == LIVE_RUN
    d = J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck, env=them))
    assert call(app, "POST", "/api/forecast", {"run_id": d["run_id"], **ROUTE, "seed": 7}, cookie=ck, env=them)[0] == 202
    with pytest.raises(ledger.Refused) as e:                                        # its FILM slice (2) is spent; others' are not
        ledger.station_hour("film", addr=ledger._address(ip))
    assert e.value.code == "ip_station" and stations.why_line("ip_station").startswith("This network address has used its part")


def chrome_dump(wrapper, path, tmp_path, budget=8000):
    """The DOM headless Chrome ends with on `path`, served by `wrapper` on a free local port; budget None: in real time, dumped once
    the page has loaded."""
    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIServer
    srv = make_server("127.0.0.1", port, wrapper, server_class=type("T", (ThreadingMixIn, WSGIServer), {"daemon_threads": True}),
                      handler_class=Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        vt = [f"--virtual-time-budget={budget}"] if budget else []
        return subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={tmp_path / 'p'}", *vt,
                               "--dump-dom", f"http://127.0.0.1:{port}{path}"], capture_output=True, text=True, timeout=120).stdout
    finally:
        srv.shutdown()


def with_probe(app, probe_js):
    """The served page with one test script after app.js (/probe.html, /probe.js); everything else is the app."""
    def wrapper(env, start):
        if env["PATH_INFO"] == "/probe.js":
            start("200 OK", [("Content-Type", "text/javascript")]); return [probe_js]
        if env["PATH_INFO"] == "/probe.html":
            got = {}
            body = b"".join(app.app({**env, "PATH_INFO": "/"}, lambda s, h: got.update(s=s, h=h)))
            body = body.replace(b"</body>", b'<script src="/probe.js"></script></body>')
            start(got["s"], [(k, v) for k, v in got["h"] if k != "Content-Length"]); return [body]
        return app.app(env, start)
    return wrapper


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
def test_plan_waits_for_the_visitor_call(web, monkeypatch, tmp_path):
    """Item 2: "Plan it" stays disabled until the page's first call (POST /api/visitor) answers, so a PLAN never goes before the
    cookie and strands its run. The visitor call is held for a second; a click at once does nothing, a later click carries it."""
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    log = []
    page = with_probe(app, b'document.body.dataset.before = String(document.getElementById("plan-go").disabled);'
                           b'document.getElementById("ask").value = "Plan a story test."; document.getElementById("plan-go").click();'
                           b'setTimeout(() => document.getElementById("plan-go").click(), 4000);')

    def wrapper(env, start):
        if env["PATH_INFO"] == "/api/visitor":
            time.sleep(1.0); out = page(env, start); log.append(("visitor", None)); return out
        if env["PATH_INFO"] == "/api/plan":
            log.append(("plan", env.get("HTTP_COOKIE", "")))
        return page(env, start)
    out = chrome_dump(wrapper, "/probe.html", tmp_path)
    assert 'data-before="true"' in out, out[:300]
    assert log and log[0][0] == "visitor" and [k for k, _ in log].count("plan") == 1 and "sh_sid=" in log[-1][1], log


def saved_cache(jobs, monkeypatch, tmp_path):
    """A precomputed cache holding alice_ch02 sentence_scramble at the default seed, as data/forecast_cache.json would."""
    r = jobs.recipe("alice_ch02", "sentence_scramble", 20261028); key = jobs.cache_key(r)
    (tmp_path / "c.json").write_text(json.dumps({"entries": {key: {**result_for(key), "recipe": r}}}))
    monkeypatch.setattr(jobs, "CACHE", tmp_path / "c.json")


def test_a_store_outage_gives_recorded_runs_and_a_whole_status(live, monkeypatch, capsys, tmp_path):
    """Item 4: with the store failing on every command, the page's calls still answer: status is well formed (live off, worker
    offline, the menu), the visitor call answers with no cookie, and every station gives its recorded run (FORECAST its saved one)
    with the line "Live runs are paused"; each failure is logged by type only."""
    (app, jobs, ledger, stations, store), paid = live
    upstash(monkeypatch, store, fail=True)
    env = {"HTTP_X_REAL_IP": "203.0.113.53"}
    code, h, out = call(app, "GET", "/api/status", cookie="", env=env); st = json.loads(out)
    assert code == 200 and st["live"] == {"meta": False, "fireworks": False} and st["worker_online"] is False
    assert st["runs_per_hour"] == ledger.RUNS_PER_HOUR and st["menu"]["text"] and st["menu"]["chips"]
    code, h, _ = call(app, "POST", "/api/visitor", {}, cookie="", env=env); assert code == 200 and "Set-Cookie" not in h
    down, saved = stations.why_line("store_down"), stations.why_line("store_down", shown=stations.SAVED_SHOWN)
    for path, body in (("/api/plan", PLAN_BODY), ("/api/audit", {"run_id": "0" * 16}), ("/api/film", {"concept": "face"}),
                       ("/api/build", {"run_id": "0" * 16, **ROUTE, "seed": 20261028})):
        code, _, out = call(app, "POST", path, body, cookie="", env=env); d = json.loads(out)
        assert code == 200 and d["mode"] == "recorded" and d["why"] == down, (path, d.get("why"))
    code, _, out = call(app, "POST", "/api/forecast", {"run_id": "0" * 16, **ROUTE, "seed": 20261028}, cookie="", env=env)
    d = json.loads(out); assert code == 200 and d["state"] == "offline" and d["why"] == saved
    assert call(app, "POST", "/api/build", {"old_bug": True}, cookie="", env=env)[0] == 200 and paid == {k: 0 for k in paid}
    saved_cache(jobs, monkeypatch, tmp_path)                                         # a job's poll: final, the saved forecast, why
    code, h, out = call(app, "GET", "/api/forecast/" + "0" * 16, env={"QUERY_STRING": "clip_id=alice_ch02&condition=sentence_scramble&seed=2"})
    d = json.loads(out)
    assert code == 200 and h["Cache-Control"] == "no-store" and d["status"] == "failed" and d["why"] == saved and d["events"] == []
    assert d["fallback"]["recipe"]["seed"] == 20261028 and d["end_line"] and d["chip"]["short"]
    for q in ("", "clip_id=not_on_the_menu&condition=intact&seed=0"):                # no recipe it can trust: a final state still
        code, _, out = call(app, "GET", "/api/forecast/" + "0" * 16, env={"QUERY_STRING": q}); d = json.loads(out)
        assert code == 200 and d["status"] == "failed" and d["why"] == saved and d.get("fallback") is None and "end_line" not in d
    err = capsys.readouterr().err; assert "ConnectError" in err and "203.0.113.53" not in err and "Plan a story" not in err


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
def test_the_page_never_shows_undefined_with_the_store_down(web, monkeypatch, tmp_path):
    """Item 4 in the real page: with the store failing, the page shows its status line, menu line and chips, and no "undefined";
    a FORECAST queued just before the store failed polls once, gets a final answer (the saved forecast and why), and stops."""
    app, jobs, ledger, stations, store = web
    upstash(monkeypatch, store, fail=True); saved_cache(jobs, monkeypatch, tmp_path)
    monkeypatch.setattr(stations, "forecast", lambda run, sid, clip, cond, seed: {**stations._forecast_base(jobs.recipe(clip, cond, seed)),
                                                                                  "state": "queued", "job_id": "0" * 16})
    polls = []
    page = with_probe(app, b'addEventListener("load", () => { S.route = {kind: "live", clip_id: "alice_ch02", condition: '
                           b'"sentence_scramble", seed: 2}; S.seed = 2; doForecast(); });')

    def wrapper(env, start):
        if env["PATH_INFO"].startswith("/api/forecast/"):
            polls.append(env.get("QUERY_STRING", ""))
        return page(env, start)
    out = chrome_dump(wrapper, "/probe.html", tmp_path)
    text = re.sub(r"<[^>]+>", " ", out)
    assert "undefined" not in text and "Loading live status." not in text and "live runs per hour per visitor" in text
    assert re.search(r'id="chips">\s*<button', out), out[:400]
    box = out[out.index('id="forecast-out"'):]
    assert "Cached forecast, computed 2026-10-30T14:00:00Z" in box and "Live runs are paused" in box, box[:500]
    assert len(polls) == 1 and "clip_id=alice_ch02" in polls[0], polls


def test_a_store_error_mid_take_gives_everything_back(web, monkeypatch):
    """Item 6: a store error between the run's start and the station's hour gives back what the request took, as a refusal does."""
    app, jobs, ledger, stations, store = web
    stub_plans(monkeypatch, stations)

    class Flaky(store.MemoryStore):
        fail = True

        def zwindow(self, k, member, score, span):
            if self.fail and k.startswith("win:cap:plan"):
                raise store.StoreError("ConnectError")
            return super().zwindow(k, member, score, span)
    S = Flaky(); store.set_store(S)
    d = stations.plan("Plan a story test on chapter 2.", "spark-1.3", "a" * 32, ip="203.0.113.70")
    assert d["mode"] == "recorded" and d["why"] == stations.why_line("store_down")
    assert not S._d.get("win:runs:" + "a" * 32) and not S._d.get(f"win:runs:ip:{ledger._address('203.0.113.70')}")
    assert not [k for k in S._d if k.startswith("run:")] and all(float(S._d[k]) == 0 for k in S._d if k.startswith("use:"))


def test_a_call_that_reached_the_provider_settles_its_reservation(live, monkeypatch):
    """Item 6: a call that reached the provider and then failed on our side (a parse error after the answer) may have been billed,
    so it settles to its reservation, not $0; one that never connected settles to $0."""
    (app, jobs, ledger, stations, store), paid = live
    from web import sam
    monkeypatch.setattr(stations, "_post", lambda *a, **k: httpx.Response(200, request=httpx.Request("POST", "https://api.meta.ai/v1/x"),
                                                                          json={"id": "file-1"}))
    def answered_then_broke(*a, **k):
        httpx.post("https://api.meta.ai/v1/files", json={}); raise ValueError("line without a frame marker")
    S, day = store.get_store(), ledger._day()
    meta = lambda: float(S.get(f"spend:meta:{day}") or 0)
    monkeypatch.setattr(stations.meta_client, "transcribe", answered_then_broke)
    d = stations.build(planned_run(ledger, store, "a" * 32), "a" * 32, *ROUTE.values(), 20261028)
    assert d["why"] == stations.why_line("api_error") and meta() == pytest.approx(stations.RESERVE["transcribe"])
    monkeypatch.setattr(sam, "segment", answered_then_broke)
    d = stations.film(None, "b" * 32, "face", ip="198.51.100.71")
    assert d["why"] == stations.why_line("api_error") and meta() == pytest.approx(stations.RESERVE["transcribe"] + stations.RESERVE["sam"])
    def plan_answered_then_broke(q, b, client):
        httpx.post("https://api.meta.ai/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        raise ValueError("unparseable answer")
    monkeypatch.setattr(stations.P, "plan", plan_answered_then_broke)
    before = meta(); d = stations.plan("Plan a story test on chapter 2.", "spark-1.3", "c" * 32, ip="198.51.100.72")
    assert d["why"] == stations.why_line("api_error") and meta() - before == pytest.approx(stations.RESERVE["plan"])
    def never_connected(*a, **k):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(sam, "segment", never_connected); before = meta()
    stations.film(None, "d" * 32, "face", ip="198.51.100.73"); assert meta() == pytest.approx(before)


# ---- fifth review (Oct 31) -------------------------------------------------------------------------------------------------------
def test_three_walks_with_reshuffles_from_one_visitor_stay_live(live, monkeypatch):
    """Item 1: a session's daily share is the address's, 20% of the configured cap ($4.00 at Meta's $20), so three full walks (PLAN,
    the build and its six reshuffles, AUDIT, FILM), one an hour, stay live once SAM's measured price is set (SAM_USD_PER_FRAME;
    $0.05 a frame stands in for it here). Unset, FILM keeps its $1.50 a run and a second FILM in a day does not fit."""
    (app, jobs, ledger, stations, store), paid = live
    monkeypatch.setattr(stations, "SAM_USD_PER_FRAME", 0.05)
    clock = [86400.0 * 20000 + 60]
    monkeypatch.setattr(ledger, "time", types.SimpleNamespace(time=lambda: clock[0])); monkeypatch.setattr(ledger, "_day", lambda: "d1")
    ck = page_cookie(app, "203.0.113.9")
    for _ in range(3):
        assert full_run(app, ck, "203.0.113.9") == LIVE_RUN
        clock[0] += 3600
    assert ledger.SESSION_DAILY_SHARE == ledger.ADDRESS_DAILY_SHARE == 0.20


def test_film_stays_live_for_six_visitors_in_an_hour(live, monkeypatch):
    """Item 1: FILM's hour holds 10 for everyone and 2 for one address (with SAM's price set, so the address's daily share does not
    bind first: unpriced, two FILMs and a third's $1.50 pass an address's $4.00)."""
    (app, jobs, ledger, stations, store), paid = live
    monkeypatch.setattr(stations, "SAM_USD_PER_FRAME", 0.05)
    for i in range(6):
        ip = f"198.51.100.{i + 10}"
        d = J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=page_cookie(app, ip), env={"HTTP_X_REAL_IP": ip}))
        assert d["mode"] == "live", (i, d.get("why"))
    one = {"HTTP_X_REAL_IP": "198.51.100.40"}
    ds = [J(call(app, "POST", "/api/film", {"concept": "face"}, cookie=issued(f"{i + 1:032x}"), env=one)) for i in range(3)]
    assert [d["mode"] for d in ds] == ["live", "live", "recorded"] and ds[2]["why"] == stations.why_line("ip_station")


def test_forecast_answers_an_unoffered_seed_with_the_saved_run(web):
    """Item 2: FORECAST takes only the seeds BUILD offers for a scramble (RESHUFFLE_SEEDS); another gets the saved forecast with
    the plan line, not a 400 (the page keeps its built state across plans), and costs the run nothing."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    run = planned_run(ledger, store, "a" * 32)
    code, _, out = call(app, "POST", "/api/forecast", {"run_id": run, **ROUTE, "seed": 99})
    d = json.loads(out)
    assert code == 200 and d["state"] == "offline" and d["why"] == stations.why_line("not_the_plan", shown=stations.SAVED_SHOWN)
    assert store.get_store().lrange("jobs:queue") == [] and float(store.get_store().get(f"use:{run}:forecast") or 0) == 0
    assert call(app, "POST", "/api/forecast", {"run_id": run, **ROUTE, "seed": 7})[0] == 202


def test_an_address_holds_at_most_two_waiting_jobs_even_at_once(web, monkeypatch):
    """Item 2: an address's waiting jobs are one atomic count (added on submit, removed when the worker claims a job or it expires),
    so ten forecasts at once from one address, with every queue read 50 ms late, queue two."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    S = store.get_store(); real_lrange, real_mget = S.lrange, S.mget
    monkeypatch.setattr(S, "lrange", lambda k, start=0, stop=-1: (real_lrange(k, start, stop), time.sleep(0.05))[0])
    monkeypatch.setattr(S, "mget", lambda keys: (real_mget(keys), time.sleep(0.05))[0])
    runs = [(f"{i:032x}", planned_run(ledger, store, f"{i:032x}", ip="203.0.113.44")) for i in range(10)]
    gate, out = threading.Barrier(10), []

    def go(sid, run):
        gate.wait(); out.append(stations.forecast(run, sid, *ROUTE.values(), 2)["state"])
    ts = [threading.Thread(target=go, args=r) for r in runs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out.count("queued") == jobs.ADDRESS_WAITING == 2 and len(real_lrange("jobs:queue")) == 2
    call(app, "POST", "/api/worker/claim", {}, auth=W)                              # the worker claims one: room for one more
    more = planned_run(ledger, store, "f" * 32, ip="203.0.113.44")
    assert stations.forecast(more, "f" * 32, *ROUTE.values(), 3)["state"] == "queued"
    monkeypatch.setattr(jobs, "JOB_EXPIRY_S", -1); S.delete("worker:last_seen")      # the queued ones expire with no worker
    for jid in real_lrange("jobs:queue"):
        assert jobs.status(jid)["status"] == "expired"
    assert float(S.get(f"waiting:{ledger._address('203.0.113.44')}") or 0) == 0


def test_busy_follows_only_a_job_for_the_same_recipe(web):
    """Item 3: a session's job is followed only by a request for the same recipe; another recipe while it runs gets the saved
    forecast and a line saying the session's other forecast is still on the laptop."""
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    run = planned_run(ledger, store, "a" * 32)
    jid = stations.forecast(run, "a" * 32, *ROUTE.values(), 1)["job_id"]
    d = stations.forecast(run, "a" * 32, *ROUTE.values(), 2)
    assert d["state"] == "busy" and "job_id" not in d and d["why"] == jobs.SESSION_BUSY_LINE and "fallback" in d
    assert stations.forecast(run, "a" * 32, *ROUTE.values(), 1) == {**stations._forecast_base(jobs.recipe(*ROUTE.values(), 1)),
                                                                     "state": "busy", "job_id": jid}


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
def test_a_new_forecast_stops_the_earlier_poll(web, monkeypatch, tmp_path):
    """Item 3 in the real page: a forecast started while an earlier one is still polling owns the output; the earlier poll, when its
    job finishes, writes nothing (a generation count checked before each write)."""
    app, jobs, ledger, stations, store = web
    first = {**stations._forecast_base(jobs.recipe(*ROUTE.values(), 1)), "state": "queued", "job_id": "1" * 16}
    second = {**stations._forecast_base(jobs.recipe(*ROUTE.values(), 2)), "state": "cached",
              "result": {**result_for("k" * 64), "source": "worker"}}
    answers = iter([first, second])
    monkeypatch.setattr(stations, "forecast", lambda *a: next(answers))
    polls = []

    def status(jid):
        polls.append(jid)
        if len(polls) == 1:
            return {"status": "running", "recipe": first["chip"] and jobs.recipe(*ROUTE.values(), 1), "events": []}
        return {"status": "done", "recipe": jobs.recipe(*ROUTE.values(), 1), "events": [], "result": result_for("j" * 64)}
    monkeypatch.setattr(jobs, "status", status)
    page = with_probe(app, b'addEventListener("load", () => { S.route = {kind: "live", clip_id: "alice_ch02", condition: '
                           b'"sentence_scramble", seed: 1}; S.seed = 1; doForecast(); setTimeout(() => { S.seed = 2; doForecast(); }, 500); });')
    out = chrome_dump(page, "/probe.html", tmp_path)
    box = out[out.index('id="forecast-out"'):out.index("</section>", out.index('id="forecast-out"'))]
    assert len(polls) >= 1 and "<h3>Forecast</h3>" in box and "Forecast, live" not in box, box[:400]


def test_film_drops_a_mask_that_lies_only_in_the_letterbox(live, monkeypatch):
    """Item 4: a SAM object only in the 640x360 cut's letterbox bars maps to an empty release mask; FILM drops empty masks (none
    left: "no mask found") instead of a 500, and any error while editing a frame gives the recorded run with the API line."""
    (app, jobs, ledger, stations, store), paid = live
    import numpy as np
    import stillhere
    from web import sam
    fe = types.ModuleType("stillhere.face_edit"); fe.VERSIONS = ("intact", "faces_removed", "area_matched")
    cut_h, cut_w, act_y0, act_h, rel_h, rel_w = 360, 640, 46, 268, 534, 1280    # stillhere/face_edit.py's geometry

    def masks_for(row):                                                              # its crop to the active picture, numpy only
        out = []
        for o in row["objects"]:
            cut = np.zeros((cut_h, cut_w), bool); x0, y0, x1, y1 = o["box_xyxy"]; cut[y0:y1, x0:x1] = True
            act = cut[act_y0:act_y0 + act_h]
            out.append(act[np.arange(rel_h) * act_h // rel_h][:, np.arange(rel_w) * cut_w // rel_w])
        return out

    def edit(frame, masks, skip_unplaceable=False):                                  # as face_edit.box() does on an empty mask
        for m in masks:
            np.nonzero(m)[1].min()
        return {v: frame for v in fe.VERSIONS}, [], [None] * len(masks)
    fe.masks_for, fe.edit = masks_for, edit
    monkeypatch.setitem(sys.modules, "stillhere.face_edit", fe); monkeypatch.setattr(stillhere, "face_edit", fe, raising=False)
    bar = {"id": 1, "score": 0.9, "box_xyxy": [100, 2, 160, 40], "mask": {}}           # inside the top bar (y < 46)
    monkeypatch.setattr(sam, "segment", lambda clip, q: {"objects": [bar], "usage": {}, "status": "response.completed"})
    code, _, out = call(app, "POST", "/api/film", {"concept": "face"}, cookie=page_cookie(app, "203.0.113.91"),
                        env={"HTTP_X_REAL_IP": "203.0.113.91"})
    d = json.loads(out)
    assert code == 200 and d["mode"] == "live" and [f["found"] for f in d["frames"]] == [False] * 3
    monkeypatch.setattr(fe, "masks_for", lambda row: (_ for _ in ()).throw(RuntimeError("a mask the decoder cannot read")))
    code, _, out = call(app, "POST", "/api/film", {"concept": "face"}, cookie=page_cookie(app, "203.0.113.92"),
                        env={"HTTP_X_REAL_IP": "203.0.113.92"})
    assert code == 200 and json.loads(out)["why"] == stations.why_line("api_error")


def test_build_transcribe_ends_well_inside_the_function_limit(live, monkeypatch):
    """Item 5: BUILD's Voice Transcribe call gets 60 s, not the client's 300 s default (Vercel Hobby's whole function limit), so a
    stall ends in the recorded run, not a platform 504; the longest clip on the menu is 40 s."""
    (app, jobs, ledger, stations, store), paid = live
    seen = []
    monkeypatch.setattr(stations.meta_client, "transcribe", lambda audio, **k: seen.append(k.get("timeout")) or {"transcript": ""})
    assert stations.build(planned_run(ledger, store, "a" * 32), "a" * 32, *ROUTE.values(), 20261028)["mode"] == "live"
    assert seen == [60.0] and stations.TRANSCRIBE_TIMEOUT_S == 60.0
    assert max(c.get("dur_s") or 0 for c in json.loads((REPO / "stillhere/live_menu.json").read_text())["clips"]) <= 45


@pytest.mark.skipif(CHROME is None, reason="headless Chrome is not on this machine")
def test_a_slow_visitor_call_is_cut_off_at_8_s(web, monkeypatch, tmp_path):
    """Item 6 in the real page, in real time: the visitor call is aborted at 8 s, so its Set-Cookie arriving at 9 s never lands; a
    PLAN at 8.5 s and every later call go on as the same cookieless visitor, so nothing is stranded. The page's load is held to
    11 s by one slow test image so the clock runs."""
    app, jobs, ledger, stations, store = web
    monkeypatch.setenv("STILLHERE_LIVE", "0")
    log = []
    page = with_probe(app, b'setTimeout(() => { document.getElementById("ask").value = "Plan a story test.";'
                           b' document.getElementById("plan-go").click(); }, 8500);'
                           b'setTimeout(() => fetch("/api/evidence"), 10000);'
                           b'document.body.append(Object.assign(document.createElement("img"), {src: "/hold.png", alt: ""}));')

    def wrapper(env, start):
        path, ck = env["PATH_INFO"], env.get("HTTP_COOKIE", "")
        if path == "/hold.png":
            time.sleep(11); start("404 Not Found", [("Content-Type", "text/plain")]); return [b""]
        if path == "/api/visitor":
            time.sleep(9)
        if path in ("/api/plan", "/api/evidence"):
            log.append((path, "sh_sid=" in ck))
        return page(env, start)
    out = chrome_dump(wrapper, "/probe.html", tmp_path, budget=None)
    assert ("/api/plan", False) in log and ("/api/evidence", True) not in log, log


def test_a_flood_from_one_address_never_reaches_the_store(web, monkeypatch):
    """Item 7: a per-address token bucket in each instance stands before the visitor call and every station POST, ahead of any store
    command: 20 at once, then one every 10 s; past it, 429 and no store command. Another address is untouched."""
    app, jobs, ledger, stations, store = web
    stub_plans(monkeypatch, stations)
    now = [1000.0]
    monkeypatch.setattr(app, "BUCKET", app.TokenBucket(*app.POST_LIMIT, clock=lambda: now[0]))
    assert app.POST_LIMIT == (20, 10.0)
    sent = upstash(monkeypatch, store)
    env = {"HTTP_X_REAL_IP": "203.0.113.99"}
    codes = [call(app, "POST", "/api/plan", PLAN_BODY, cookie="", env=env)[0] for _ in range(20)]
    before = len(sent)
    late = [call(app, "POST", p, b, cookie="", env=env)[0] for p, b in (("/api/plan", PLAN_BODY), ("/api/visitor", {}),
                                                                         ("/api/build", {"old_bug": True}), ("/api/film", {"concept": "f"}))]
    assert set(codes) == {200} and late == [429] * 4 and len(sent) == before
    assert call(app, "POST", "/api/plan", PLAN_BODY, cookie="", env={"HTTP_X_REAL_IP": "198.51.100.99"})[0] == 200
    now[0] += 10.0; assert call(app, "POST", "/api/plan", PLAN_BODY, cookie="", env=env)[0] == 200
    assert call(app, "POST", "/api/worker/claim", {}, auth=W, env=env)[0] in (200, 204)   # the worker is not a visitor


def test_status_reads_the_worker_once_in_five_seconds(web, monkeypatch):
    """Item 7: /api/status reads the worker's heartbeat from the store at most once every 5 s per instance, however often it is
    asked (a cache-busting query skips the CDN, not this)."""
    app, jobs, ledger, stations, store = web
    sent = upstash(monkeypatch, store)
    now = [5000.0]; monkeypatch.setattr(jobs, "_clock", lambda: now[0], raising=False)
    for i in range(10):
        assert call(app, "GET", "/api/status", env={"QUERY_STRING": f"x={i}"})[0] == 200
    assert sum(1 for c in sent if c == ["GET", "worker:last_seen"]) == 1
    now[0] += 5.1; call(app, "GET", "/api/status")
    assert sum(1 for c in sent if c == ["GET", "worker:last_seen"]) == 2


def test_plan_and_audit_hours_hold_a_busy_hour(live, monkeypatch):
    """Item 8: PLAN's and AUDIT's hours hold 120, one address taking at most 10 of each, so six addresses with ten runs each are all
    live; PLAN and AUDIT settle to their tokens, so the daily caps still bound the money."""
    (app, jobs, ledger, stations, store), paid = live
    live_runs = 0
    for a in range(6):
        env = {"HTTP_X_REAL_IP": f"198.51.100.{a + 120}"}
        for c in range(2):
            ck = issued(f"{a * 2 + c + 1:032x}")
            for _ in range(ledger.RUNS_PER_HOUR):
                d = J(call(app, "POST", "/api/plan", PLAN_BODY, cookie=ck, env=env))
                a_ = J(call(app, "POST", "/api/audit", {"run_id": d.get("run_id")}, cookie=ck, env=env))
                live_runs += d["mode"] == "live" and a_["mode"] == "live"
    assert live_runs == 60
