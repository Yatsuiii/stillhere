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
import threading
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
SECRETS = {"MODEL_API_KEY": "meta-SECRET-0123456789abcdef", "FIREWORKS_API_KEY": "fw-SECRET-0123456789abcdef",
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
    return app, jobs, ledger, stations, store


def call(app, method, path, body=None, auth=None, cookie="sh_sid=" + "a" * 32):
    data = json.dumps(body).encode() if body is not None else b""
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(data)), "wsgi.input": io.BytesIO(data),
           "HTTP_COOKIE": cookie, "wsgi.url_scheme": "http"}
    if auth:
        env["HTTP_AUTHORIZATION"] = auth
    st = {}
    out = b"".join(app.app(env, lambda s, h: st.update(s=s, h=h)))
    return int(st["s"].split()[0]), dict(st["h"]), out


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
    for f in (REPO / "web/public").iterdir():
        assert not re.search(rb"(api[_-]?key|Bearer|SECRET|fw_[A-Za-z0-9]{8})", f.read_bytes()), f.name


def test_every_station_falls_back_to_its_recorded_run_with_the_apis_blocked(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    block_apis(monkeypatch, stations)
    _, _, out = call(app, "POST", "/api/plan", {"text": "Plan an exploratory stock TRIBE forecast of Alice chapter 1.", "backend": "spark-1.3"})
    d = json.loads(out); assert d["mode"] == "recorded" and d["label"] == "recorded run" and d["runs"]
    assert any(r["card"]["decision"] == "decline" for r in d["runs"]) and any(r["card"]["decision"] == "accept" for r in d["runs"])
    _, _, out = call(app, "POST", "/api/film", {"concept": "face"}); d = json.loads(out)
    assert d["mode"] == "recorded" and len(d["frames"]) == 3 and "CC BY 3.0" in d["credit"]
    run = ledger.start_run("a" * 32)
    store.get_store().set(f"runspec:{run}", json.dumps({"spec": {"population": "healthy_adults", "decline_reason": "none"}}))
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
    for f in list((REPO / "web/public").iterdir()) + list((REPO / "web/replay").glob("*.json")):
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
    monkeypatch.setattr(stations.meta_client, "chat", lambda req: chats.append(req.model) or {"notes": ["Check the control."]})
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
    def boom(text, backend, sid):
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
    run = ledger.start_run("a" * 32)
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
    texts += [f.read_text() for f in (REPO / "web/public").iterdir()] + [menu_stub.STORY_LEAD, menu_stub.STORY_TAIL, menu_stub.STORY_NONE]
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
    run = ledger.start_run("a" * 32)
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
    run2 = ledger.start_run("a" * 32)
    code, _, out = call(app, "POST", "/api/forecast", {"run_id": run2, "clip_id": "alice_ch02", "condition": "reversed", "seed": 5})
    assert json.loads(out)["state"] == "cached"                                     # reversed ignores the seed: same cache key


def test_jobs_expire_and_fall_back(web, monkeypatch):
    app, jobs, ledger, stations, store = web
    call(app, "POST", "/api/worker/claim", {}, auth=W)
    run = ledger.start_run("a" * 32)
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
    run = ledger.start_run("a" * 32)
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
