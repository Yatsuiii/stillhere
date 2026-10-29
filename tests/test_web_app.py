"""The hosted Pilot app (web/): no key ever reaches the browser; every live station falls back to its recorded run with the model APIs
blocked; the wording guard replaces banned model lines; spend caps and the 5-runs-per-hour limit hold; the jobs API follows
docs/jobs_api.md with a fake worker (token, events, result checks, expiry, cache); the evidence endings follow the rulings of Oct 29,
18:15 IST; and the page lays out at 360 px with no horizontal scroll."""
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
    d = json.loads(out); assert d["state"] == "offline" and d["end_line"] == "No public scan of this clip. Forecast, not a scan."
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
        ledger.reserve("baseten", 0.01)
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
        d20 = None if status is None else {"primary_intact_gt_shortscram": {"status": status, "n_pairs": 12, "k_intact_higher": 9, "p": 0.01}}
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
