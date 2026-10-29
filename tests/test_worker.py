"""The live worker (worker/, docs/worker.md): recipes and cache keys, the committed clips, the protocol end to end against the local
fake jobs server with a stand-in engine (no GPU), the sandbox unit and the install script, and the precomputed forecast cache."""
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from worker import fake_jobs_server as F  # noqa: E402
from worker import recipe as R  # noqa: E402

IDX = R.load_index()
UNIT = (REPO / "worker/stillhere-worker.service.in").read_text()
EVENT_ORDER = [("build", "start"), ("build", "end"), ("video_lane", "off"), ("text_lane", "start"), ("text_lane", "end"),
               ("audio_lane", "start"), ("audio_lane", "end"), ("predict", "start"), ("predict", "end"), ("regions", "start"),
               ("regions", "end")]


def test_recipe_schema_refuses_everything_else():
    good = {"clip_id": "alice_ch01", "condition": "word_scramble", "seed": 7}
    assert R.normalize(good, IDX) == good
    assert R.normalize({**good, "condition": "intact"}, IDX)["seed"] == 0
    assert R.normalize({**good, "condition": "reversed", "seed": 99}, IDX)["seed"] == 0
    bad = [{**good, "extra": 1}, {"clip_id": "alice_ch01", "condition": "intact"}, {**good, "clip_id": "lpp_section8"},
           {**good, "clip_id": "../confidential"}, {**good, "condition": "sentence_shuffle"}, {**good, "seed": -1}, {**good, "seed": 2**31},
           {**good, "seed": True}, {**good, "seed": 1.0}, {**good, "seed": "7"}, ["alice_ch01", "intact", 0], None]
    for b in bad:
        with pytest.raises(R.RecipeRefused):
            R.normalize(b, IDX)


def test_cache_key_is_the_documented_formula():
    rec = {"clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 20261028}
    c = R.clip_entry("alice_ch02", IDX)
    want = hashlib.sha256(json.dumps({**rec, "models": IDX["models"], "wav_sha256": c["wav_sha256"], "words_sha256": c["words_sha256"]},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert R.cache_key(rec, IDX) == want
    assert R.cache_key({**rec, "seed": 1}, IDX) != want
    assert R.cache_key({**rec, "condition": "intact", "seed": 5}, IDX) == R.cache_key({**rec, "condition": "intact", "seed": 0}, IDX)
    other = {**IDX, "models": {**IDX["models"], "precision": "4bit"}}
    assert R.cache_key(rec, other) != want


def test_clips_match_index_and_live_menu():
    menu = json.loads((REPO / "stillhere/live_menu.json").read_text())
    ids = [c["id"] for c in IDX["clips"]]
    assert ids == [f"alice_ch{i:02d}" for i in range(1, 13)]
    for m in menu["clips"]:
        assert R.clip_entry(m["id"], IDX)["wav_sha256"] == m["intact_wav_sha256"]
    for c in IDX["clips"]:
        x, sr, words = R.load_clip(c["id"], IDX)
        assert sr == 16000 and abs(len(x) / sr - 40.0) < 1e-6 and len(words) == c["n_words"]
        assert all(w["end"] > w["start"] or 0.01 < w["start"] < 39.95 for w in words)
        assert not any("\u2014" in w["word"] for w in words)
    for k in ("tribe", "w2v_bert", "llama", "precision", "v_jepa_2", "audio_lane_device", "brain_device", "word_source"):
        assert IDX["models"][k]


@pytest.mark.parametrize("cond", R.CONDITIONS)
def test_build_condition(cond):
    rec = {"clip_id": "alice_ch05", "condition": cond, "seed": 20261028}
    a, b = R.build_condition(rec, IDX), R.build_condition(rec, IDX)
    assert a.wav_sha256 == b.wav_sha256 and len(a.audio) == 640000
    assert a.wav_sha256 == hashlib.sha256(R.wav_bytes(a.audio, a.sr)).hexdigest()
    _, _, words = R.load_clip("alice_ch05", IDX)
    if cond == "intact":
        assert a.wav_sha256 == R.clip_entry("alice_ch05", IDX)["wav_sha256"] and a.words == [
            {"word": w["word"], "start": round(w["start"], 4), "end": round(w["end"], 4)} for w in words]
    if cond == "reversed":
        assert a.words == []
    else:
        assert Counter(w["word"] for w in a.words) == Counter(w["word"] for w in words)
        assert all(-0.01 <= w["start"] <= w["end"] <= 40.01 for w in a.words)
    if cond in ("sentence_scramble", "word_scramble"):
        assert a.wav_sha256 != R.build_condition({**rec, "seed": 3}, IDX).wav_sha256


def test_word_table_round_trips_through_pandas(tmp_path):
    """A word that starts with a double quote must not open a quoted field (it silently dropped half of alice_ch07's words once)."""
    pd = pytest.importorskip("pandas")
    from worker.engine import check_tsv, write_tsv
    for cid in ("alice_ch07", "alice_ch08", "alice_ch12"):
        b = R.build_condition({"clip_id": cid, "condition": "word_scramble", "seed": 7}, IDX)
        assert any(w["word"].startswith('"') for w in b.words)
        p = tmp_path / f"{cid}.tsv"; write_tsv(b.words, p); check_tsv(b.words, p)
        assert len(pd.read_csv(p, sep="\t")) == len(b.words)


def test_word_rate_and_regions():
    assert R.word_rate([{"start": 0.2}, {"start": 0.9}, {"start": 2.5}, {"start": 40.2}], 3) == [2, 0, 1]
    reg = json.loads((REPO / "worker/regions.json").read_text())
    assert len(reg["LANG"]) == 19 and len(reg["AUD"]) == 7
    assert all(0 <= i < 20484 for k in ("LANG", "AUD") for v in reg[k] for i in v)


class StubEngine:
    """Stands in for worker/engine.Engine: same events, a result of the documented shape, no torch."""
    gpu, load_s = "stub", 0.0

    def __init__(self, cache_dir):
        pass

    def load(self):
        return self

    def models(self):
        return IDX["models"]

    def cache_full(self):
        return False

    def run(self, built, workdir, emit):
        for st, status in EVENT_ORDER:
            lane = {"video_lane": "v-jepa-2", "text_lane": "llama-3.2", "audio_lane": "w2v-bert"}.get(st)
            emit(st, lane, status, {"n_words": len(built.words)} if st == "text_lane" else {})
        return {"tr_s": 1.0, "seconds": 40, "regions": {"language": [0.1] * 40, "hearing": [0.2] * 40},
                "word_rate": R.word_rate(built.words, 40), "n_words_text_lane": len(built.words), "input_sha256": built.wav_sha256,
                "models": IDX["models"], "timings_s": {"total": 0.0}}


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_protocol_end_to_end_with_fake_server(monkeypatch, tmp_path):
    import worker.engine
    from worker import tribe_worker as T
    monkeypatch.setattr(worker.engine, "Engine", StubEngine)
    good = {"clip_id": "alice_ch03", "condition": "sentence_scramble", "seed": 7}
    jobs = [{"id": "j1", "recipe": good, "cache_key": R.cache_key(good, IDX)},
            {"id": "j2", "recipe": good, "cache_key": "0" * 64},
            {"id": "j3", "recipe": {**good, "clip_id": "not_listed"}, "cache_key": "0" * 64},
            {"id": "j4", "recipe": {**good, "audio_url": "https://x"}, "cache_key": "0" * 64}]
    token = "t" * 64; tf = tmp_path / "tok"; tf.write_text(token)
    st = F.State(token, jobs); port = _port(); srv = F.serve(port, st)
    monkeypatch.setenv("STILLHERE_JOBS_URL", f"http://127.0.0.1:{port}"); monkeypatch.setattr(T, "POLL_S", 0.01)
    monkeypatch.setattr(T, "STATE", tmp_path); monkeypatch.setattr(T, "within_hours", lambda *a, **k: True)
    try:
        T.main(["serve", "--token-file", str(tf), "--max-polls", "2"])
    finally:
        srv.shutdown()
    assert st.unauthorized == 0 and set(st.results) == {"j1"}
    assert st.fails == {"j2": "cache_key_mismatch", "j3": "bad_recipe", "j4": "bad_recipe"}
    res = st.results["j1"]
    assert res["cache_key"] == jobs[0]["cache_key"] and len(res["regions"]["language"]) == 40
    assert [(e["stage"], e["status"]) for e in st.events["j1"]] == EVENT_ORDER
    assert st.events["j2"] == st.events["j3"] == []                     # refused jobs never start a stage


def test_max_polls_counts_failed_claims(monkeypatch, tmp_path):
    """A test run ends once the jobs server is gone: --max-polls counts failed claims too (Oct 29 E2E kept retrying)."""
    import worker.engine
    from worker import tribe_worker as T
    monkeypatch.setattr(worker.engine, "Engine", StubEngine)
    tf = tmp_path / "tok"; tf.write_text("t" * 64)
    monkeypatch.setenv("STILLHERE_JOBS_URL", f"http://127.0.0.1:{_port()}")     # nothing listens here
    monkeypatch.setattr(T, "POLL_S", 0.01); monkeypatch.setattr(T, "STATE", tmp_path)
    monkeypatch.setattr(T, "within_hours", lambda *a, **k: True)                    # independent of the clock (worker hours)
    slept = []; monkeypatch.setattr(T.time, "sleep", slept.append)
    T.main(["serve", "--token-file", str(tf), "--max-polls", "3"])
    assert slept == [0.01, 0.02]                                                    # backoff still doubles between failures


def test_wrong_token_gets_401_and_https_only(tmp_path):
    from worker import tribe_worker as T
    st = F.State("right" * 8, []); port = _port(); srv = F.serve(port, st)
    try:
        code, body = T.Jobs(f"http://127.0.0.1:{port}", "wrong" * 8).claim({})
    finally:
        srv.shutdown()
    assert (code, body, st.unauthorized) == (401, None, 1)
    with pytest.raises(SystemExit):
        T.Jobs("http://example.com", "x" * 40)


def test_unit_has_every_sandbox_line_of_the_ruling():
    lines = [ln.strip() for ln in UNIT.splitlines()]
    for want in ("DynamicUser=yes", "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes", "PrivateTmp=yes",
                 "TemporaryFileSystem=/run/media", "StateDirectory=stillhere-worker", "LoadCredential=job_token:/etc/stillhere-worker/job_token",
                 "Environment=HF_HUB_OFFLINE=1"):
        assert want in lines, want
    binds = [ln.split("=", 1)[1] for ln in lines if ln.startswith("BindReadOnlyPaths=")]
    assert binds == ["@MODELS@:/opt/stillhere-worker/models", "@SITE@:/opt/stillhere-worker/site-packages",
                     "@SPACY@:/opt/stillhere-worker/spacy"]
    assert not any(ln.startswith(("BindPaths=", "PrivateDevices=yes", "ReadWritePaths=")) for ln in lines)
    assert "[Install]" not in lines and "Restart=@RESTART@" in lines and "Wants=@WANTS@" in lines
    assert "Environment=STILLHERE_HOURS_ET=10:00-14:00" in lines and "EnvironmentFile=-/etc/stillhere-worker/worker.env" in lines


def test_install_script_renders_every_field_and_refuses_http():
    sh = (REPO / "worker/install_worker.sh").read_text()
    assert subprocess.run(["bash", "-n", "worker/install_worker.sh"], cwd=REPO).returncode == 0
    assert subprocess.run(["bash", "-n", "worker/uninstall_worker.sh"], cwd=REPO).returncode == 0
    fields = set(re.findall(r"@[A-Z_]+@", UNIT))
    assert fields == {"@ROLE@", "@TYPE@", "@COMMAND@", "@RESTART@", "@WANTS@", "@MODELS@", "@SITE@", "@SPACY@", "@JOBS_URL@", "@HIDDEN@"}
    assert 'render "serve" simple serve on-failure stillhere-worker-awake.service' in sh   # exit 0 at 14:00 ET stays down
    assert re.findall(r"systemctl enable[^\n]*", sh) == ["systemctl enable --now stillhere-worker.timer"]   # only the timer
    un = (REPO / "worker/uninstall_worker.sh").read_text()
    assert all(u in un for u in ("stillhere-worker.timer", "stillhere-worker-awake.service", "stillhere-worker-selftest.service"))
    assert all(f"s#{f}#" in sh for f in fields)
    assert "^https://" in sh and "SELF-TEST FAILED" in sh and "--audit" in sh


def test_timer_and_awake_units():
    timer = (REPO / "worker/stillhere-worker.timer").read_text().splitlines()
    for want in ("OnCalendar=2026-11-04..09 10:00:00 America/New_York", "Persistent=true", "Unit=stillhere-worker.service",
                 "WantedBy=timers.target"):
        assert want in timer, want
    awake = (REPO / "worker/stillhere-worker-awake.service").read_text().splitlines()
    assert "BindsTo=stillhere-worker.service" in awake
    assert any(ln.startswith("ExecStart=/usr/bin/systemd-inhibit --what=sleep:idle:handle-lid-switch") and "--mode=block" in ln for ln in awake)


def test_worker_hours(monkeypatch):
    import datetime
    from zoneinfo import ZoneInfo

    from worker import tribe_worker as T
    at = lambda h, m: datetime.datetime(2026, 11, 4, h, m, tzinfo=ZoneInfo("America/New_York"))
    assert [T.within_hours(at(*hm), "10:00-14:00") for hm in ((9, 59), (10, 0), (13, 59), (14, 0))] == [False, True, True, False]
    assert T.within_hours(at(23, 59), "00:00-24:00") and T.within_hours(at(0, 0), "00:00-24:00")
    loaded = []

    class NoLoad(StubEngine):
        def load(self):
            loaded.append(1); return self
    import worker.engine
    monkeypatch.setattr(worker.engine, "Engine", NoLoad); monkeypatch.setattr(T, "within_hours", lambda *a, **k: False)
    assert T.main(["serve"]) == 0 and loaded == []                      # outside the hours: exit 0 before loading anything


def test_serve_stops_with_exit_0_when_the_hours_end(monkeypatch, tmp_path):
    import worker.engine
    from worker import tribe_worker as T
    monkeypatch.setattr(worker.engine, "Engine", StubEngine)
    calls = iter([True, True, False])                                  # start check, first poll, then the hours are over
    monkeypatch.setattr(T, "within_hours", lambda *a, **k: next(calls))
    token = "t" * 64; tf = tmp_path / "tok"; tf.write_text(token)
    st = F.State(token, []); port = _port(); srv = F.serve(port, st)
    monkeypatch.setenv("STILLHERE_JOBS_URL", f"http://127.0.0.1:{port}"); monkeypatch.setattr(T, "POLL_S", 0.01)
    monkeypatch.setattr(T, "STATE", tmp_path)
    try:
        assert T.main(["serve", "--token-file", str(tf)]) == 0
    finally:
        srv.shutdown()
    assert st.claims == 1


def test_forecast_cache():
    p = REPO / "data/forecast_cache.json"
    if not p.exists():
        pytest.skip("cache not computed yet")
    doc = json.loads(p.read_text())
    extras = doc.get("extra_recipes", [])
    assert doc["models"] == IDX["models"] and len(doc["entries"]) == 48 + len(extras)
    assert {"clip_id": "alice_ch03", "condition": "sentence_scramble", "seed": 7} in extras    # the page's example chip 3
    seen = set()
    for key, e in doc["entries"].items():
        assert key == e["cache_key"] == R.cache_key(e["recipe"], IDX)
        seen.add((e["recipe"]["clip_id"], e["recipe"]["condition"], e["recipe"]["seed"]))
        assert e["recipe"]["seed"] in (0, IDX["default_seed"]) or e["recipe"] in extras
        assert e["worker_version"] == "2"                    # version 1 purged TRIBE's feature cache mid-process (docs/worker.md)
        assert e["recipe"]["condition"] == "reversed" or e["llama_placement"]["gpu_modules"] > 0
        for k in ("language", "hearing"):
            v = e["regions"][k]; assert 30 <= len(v) <= 60 and all(math.isfinite(x) for x in v)
        assert len(e["word_rate"]) == len(e["regions"]["language"])
        n = 0 if e["recipe"]["condition"] == "reversed" else R.clip_entry(e["recipe"]["clip_id"], IDX)["n_words"]
        assert n - 1 <= e["n_words_text_lane"] <= n          # TRIBE's own word steps may drop one (alice_ch06 word scramble: 145 of 146)
        assert e["input_sha256"] == R.build_condition(e["recipe"], IDX).wav_sha256
        assert not any(k in e for k in ("P", "vertices", "transcript", "audio"))
    assert len(seen) == 48 + len(extras)


def test_serve_exits_for_a_fresh_restart_when_the_feature_cache_is_full(monkeypatch, tmp_path):
    """The engine never prunes TRIBE's feature cache while it runs; past the limit serve returns after the job (systemd restarts it)."""
    import worker.engine
    from worker import tribe_worker as T

    class FullStub(StubEngine):
        def cache_full(self):
            return True
    monkeypatch.setattr(worker.engine, "Engine", FullStub)
    good = {"clip_id": "alice_ch04", "condition": "intact", "seed": 0}
    jobs = [{"id": f"j{i}", "recipe": good, "cache_key": R.cache_key(good, IDX)} for i in (1, 2)]
    token = "t" * 64; tf = tmp_path / "tok"; tf.write_text(token)
    st = F.State(token, jobs); port = _port(); srv = F.serve(port, st)
    monkeypatch.setenv("STILLHERE_JOBS_URL", f"http://127.0.0.1:{port}"); monkeypatch.setattr(T, "STATE", tmp_path)
    monkeypatch.setattr(T, "within_hours", lambda *a, **k: True)
    try:
        code = T.main(["serve", "--token-file", str(tf)])             # no --max-polls: only the full cache can end it
    finally:
        srv.shutdown()
    assert code == T.EXIT_RESTART != 0                                 # non-zero, so Restart=on-failure starts a fresh worker
    assert set(st.results) == {"j1"} and [j["id"] for j in st.queue] == ["j2"]
