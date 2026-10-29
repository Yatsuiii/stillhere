"""Story menu (docs/rules/story_menu_rule.md): pool, exclusion, trim, bars, window, and the screen and notes stages run on stubs, so the
egress guard, the allowlists and the spend cap are tested before any real Voice Transcribe or Spark Open call (cloud plan, Part 5)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stillhere import story_menu as SM  # noqa: E402

pytest.importorskip("pydantic")


def W(text: str, t0: float = 0.0, step: float = 0.4) -> list:
    return [{"word": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.3, 3)} for i, w in enumerate(text.split())]


def sec(n, title, play, lang="English", readers=1, url="https://archive.org/download/x/y.mp3"):
    return {"section_number": str(n), "title": title, "language": lang, "playtime": str(play), "listen_url": url,
            "readers": [{"display_name": f"R{i}"} for i in range(readers)]}


def test_pool_filters_orders_and_caps():
    b1 = {"id": "1", "title": "Vol 1", "sections": [sec(2, "Two", 300), sec(1, "One", 240), sec(3, "Long", 721), sec(4, "Pie Man", 300),
                                                    sec(5, "French", 300, lang="French"), sec(6, "Duo", 300, readers=2), sec(7, "Edge", 720)]}
    b2 = {"id": "2", "title": "Vol 2", "sections": [sec(i, f"S{i}", 400) for i in range(1, 40)]}
    b1["sections"][2]["playtime"] = "1801"; b1["sections"][6]["playtime"] = "1800"
    p = SM.pool([b1, b2])
    assert [x["title"] for x in p[:3]] == ["One", "Two", "Edge"] and len(p) == SM.POOL_SIZE


def test_public_scan_exclusion():
    assert SM.public_scan("Pie Man") == "ds002345" and SM.public_scan("pie-man!") == "ds002345"
    assert SM.public_scan("The Matchstick Seller") == "naturalstories"
    assert SM.public_scan("Le Petit Prince") == "ds003643"
    assert SM.public_scan("The Little Match Girl") == "naturalstories"          # by name (amended rule)
    assert SM.public_scan("The Toys of Peace") is None and SM.public_scan("The Black Cat") is None


def test_committed_pool():
    """Amended rule (4:00-30:00, The Little Match Girl excluded): 11 stories, Vol. 001 first, all at or under 1800 s."""
    doc = json.loads((REPO / "results/story_menu/pool.json").read_text())
    assert [p["id"] for p in doc["projects"]] == ["296", "297"] and len(doc["sections_seen"]) == 20
    assert len(doc["pool"]) == 11 and all(240 <= p["playtime_s"] <= 1800 for p in doc["pool"])
    assert [p["book_id"] for p in doc["pool"]] == ["296"] * 5 + ["297"] * 6
    assert "The Little Match Girl" not in [p["title"] for p in doc["pool"]] and "Duplicity of Hargraves" not in [p["title"] for p in doc["pool"]]
    assert not any("reader" in p for p in doc["pool"])                # reader names only for menu stories, never in the public pool


def test_trim():
    w = W("This is a LibriVox recording. Read by Someone Reader. Once upon a time. A girl walked. She stopped. "
          "The tale is told. End of the story.")
    first, last = SM.trim(w)
    assert w[first]["word"] == "Once" and w[last]["word"] == "told."
    with pytest.raises(SM.TrimFail):
        SM.trim(W("Once upon a time. A girl walked."))
    w2 = W("Read by X. One two. Three four.")                         # no closing phrase: ends at the last complete sentence
    assert w2[SM.trim(w2)[1]]["word"] == "four."


def test_bars():
    t = lambda sp, a, b: {"speaker": sp, "startMs": a, "endMs": b}
    assert SM.bar1([[t("A", 0, 9600), t("B", 9600, 10000)]])["pass"] and not SM.bar1([[t("A", 0, 9000), t("B", 9000, 10000)]])["pass"]
    assert not SM.bar1([])["pass"]
    w = W("one two three"); w[2]["start"] = w[1]["end"] + 5.1; w[2]["end"] = w[2]["start"] + 0.3
    assert not SM.bar2(w)["pass"] and SM.bar2(W("one two three"))["pass"]
    assert SM.bar3("one two three four", W("one two three four"))["pass"] and not SM.bar3("one two", W("one two three four"))["pass"]
    assert SM.chunks(10.0, 135.0) == [(10.0, 70.0), (70.0, 130.0), (130.0, 135.0)]


def test_window_is_the_earliest_valid_candidate():
    text = " ".join(f"word{i} end{i}." for i in range(40))            # 40 sentences of 2 words, 0.4 s per word
    w = W(text, t0=5.0, step=0.6)
    win = SM.window(w, 0, len(w) - 1)
    assert win["first_word"] == 0 and win["sentences"] >= 6 and 30.0 <= win["end_s"] - win["start_s"] <= 40.0
    assert SM.window(W("a b. c d."), 0, 3) is None


@pytest.fixture
def run(tmp_path, monkeypatch):
    import scripts.story_menu_run as RUN
    from stillhere import egress
    for name in ("AUDIO_ALLOWLIST", "TEXT_ALLOWLIST"):
        p = tmp_path / f"{name}.txt"; p.write_text("# test\n"); monkeypatch.setattr(RUN, name, p)
    monkeypatch.setattr(egress, "AUDIO_ALLOWLIST", RUN.AUDIO_ALLOWLIST); monkeypatch.setattr(egress, "ALLOWLIST", RUN.TEXT_ALLOWLIST)
    monkeypatch.setattr(RUN, "RESULTS", tmp_path / "results")
    return RUN


def _story(work: Path, sid="story_test", seconds=240.0):
    import numpy as np

    from stillhere import conditions as C
    d = work / sid; d.mkdir(parents=True)
    C.write_wav(d / "section.wav", 0.01 * np.sin(np.arange(int(seconds * 16000)) / 7.0), 16000)
    words = W("Read by Someone. " + " ".join(f"tale{i} goes on." for i in range(150)), t0=1.0, step=0.5)
    (d / "words.json").write_text(json.dumps({"words": words}))
    first, last = SM.trim(words)
    info = {"id": sid, "book_id": "297", "section": 1, "archive_url": "https://archive.org/download/x/test.mp3",
            "trim": {"first_word": first, "last_word": last, "start_s": words[first]["start"], "end_s": words[last]["end"]}}
    (d / "story.json").write_text(json.dumps(info))
    return words


def test_screen_on_stubs_allowlists_and_stops_at_the_cap(tmp_path, run):
    words = _story(tmp_path)
    calls = []

    def stub(audio):
        calls.append(audio.path)
        return {"transcript": "goes on", "audioDurationMs": 60000, "turns": [{"speaker": "A", "startMs": 0, "endMs": 60000}]}
    ledger = run.Ledger(tmp_path / "ledger.jsonl")
    run.screen(tmp_path, stub, ledger, cap=0.25)                       # $0.10 a call: the third call would cross $0.25
    doc = json.loads((tmp_path / "story_test/screen.json").read_text())
    assert len(calls) == 2 and doc["complete"] is False and doc["pass"] is False and doc["failed_bar"] == "cap"
    assert abs(ledger.total("vt_screen") - 0.20) < 1e-9
    listed = run.AUDIO_ALLOWLIST.read_text()
    assert all(Path(p).name.startswith("chunk") for p in calls) and listed.count("story menu screen chunk story_test") == doc["chunks"]
    assert words


def test_egress_refuses_audio_that_is_not_listed(tmp_path, run):
    from stillhere import egress
    _story(tmp_path)
    with pytest.raises(ValueError):                                  # EgressRefused, wrapped by pydantic
        egress.PublicAudio(path=str(tmp_path / "story_test/section.wav"))
    with pytest.raises(ValueError):                                  # EgressRefused, wrapped by pydantic
        egress.PublicText(text="any transcript that is not on the text allowlist")


def test_notes_on_stubs(tmp_path, run, monkeypatch):
    pytest.importorskip("stillhere.story_tags")
    words = _story(tmp_path)
    (tmp_path / "story_test/screen.json").write_text(json.dumps({"id": "story_test", "pass": True}))
    monkeypatch.setattr(run, "guard_blocks", lambda q: "conscious" in q)
    seen = []

    def stub(model, text):
        seen.append(model)
        if model.startswith("fireworks"):
            raise RuntimeError("fireworks down")
        return ({"notes": [{"tag": "violence", "quote": "tale3 goes on"}, {"tag": "violence", "quote": "not in the story at all"},
                           {"tag": "made up tag", "quote": "tale4 goes on"}, {"tag": "grief or loss", "quote": "tale9 goes on conscious"},
                           {"tag": "drowning", "quote": "tale7 goes"}],
                 "description": "One. Two."}, {"prompt_tokens": 1000, "completion_tokens": 100})
    ledger = run.Ledger(tmp_path / "ledger.jsonl")
    run.notes(tmp_path, stub, ledger)
    doc = json.loads((tmp_path / "story_test/notes.json").read_text())
    assert seen == ["fireworks/muse-spark-open"] * 2 + ["baseten/muse-spark-open-1026"] and doc["host"] == "baseten"
    tags = [(n["tag"], n["quote"]) for n in doc["notes"]]
    assert tags == [("violence", "tale3 goes on"), ("drowning", "tale7 goes")]
    t3 = next(w["start"] for w in words if w["word"] == "tale3")
    assert doc["notes"][0]["t"] == round(t3, 2)
    listed = run.TEXT_ALLOWLIST.read_text()
    assert listed.count("story menu transcript story_test") == 1 and ledger.total("spark_open") > 0


def test_guard_withholds_but_keeps_tag_and_time(tmp_path, run, monkeypatch):
    pytest.importorskip("stillhere.story_tags")
    words = W("Read by X. He was conscious of the cold. It went on.")
    monkeypatch.setattr(run, "guard_blocks", lambda q: "conscious" in q)
    kept = run.keep_notes([{"tag": "frightening scenes", "quote": "He was conscious of the cold."}], words)
    assert kept == [{"tag": "frightening scenes", "t": words[3]["start"], "quote": None}]


def test_live_stages_refuse_without_the_flag(tmp_path):
    import scripts.story_menu_run as RUN
    for stage in ("screen", "notes"):
        with pytest.raises(SystemExit):
            RUN.main([stage, str(tmp_path)])


def test_short_id():
    import scripts.story_menu_run as RUN
    assert RUN.short_id("The Little Match Girl") == "story_little_match_girl" and RUN.short_id("The Toys of Peace") == "story_toys_of_peace"


def test_checks_on_stubs_record_build_replays(tmp_path, run, monkeypatch):
    """A window clip's 4 conditions go out only as allowlisted builds; each check is recorded keyed clip:condition."""
    import hashlib

    import numpy as np

    from stillhere import conditions as C
    from worker import recipe as R
    root = tmp_path / "root"; (root / "data/live_clips").mkdir(parents=True)
    words = W(" ".join(f"tale{i} goes on." for i in range(30)), t0=0.4, step=0.4)
    sha = C.write_wav(root / "data/live_clips/story_test.wav", 0.01 * np.sin(np.arange(16000 * 37) / 5.0), 16000)
    wt = json.dumps({"words": words}); (root / "data/live_clips/story_test.words.json").write_text(wt)
    idx = {**R.load_index(), "clips": [{"id": "story_test", "wav": "data/live_clips/story_test.wav", "wav_sha256": sha, "seconds": 37.0,
                                        "words": "data/live_clips/story_test.words.json", "words_sha256": hashlib.sha256(wt.encode()).hexdigest(),
                                        "rule": "docs/rules/story_menu_rule.md"}]}
    monkeypatch.setattr(run, "BUILD_REPLAY", tmp_path / "build.json")
    lines = [f"{R.build_condition({'clip_id': 'story_test', 'condition': c, 'seed': idx['default_seed']}, idx, root).wav_sha256}  t"
             for c in R.CONDITIONS]
    run.allowlist(run.AUDIO_ALLOWLIST, lines, "test")
    stub = lambda audio: {"transcript": " ".join(w["word"] for w in words), "turns": []}
    ledger = run.Ledger(tmp_path / "ledger.jsonl")
    run.checks(tmp_path, stub, ledger, idx=idx, root=root)
    rec = json.loads((tmp_path / "build.json").read_text())
    assert sorted(rec) == sorted(f"story_test:{c}" for c in R.CONDITIONS)
    assert rec["story_test:intact"]["pass"] is True and rec["story_test:intact"]["checklist"]
    assert abs(ledger.total("vt_check") - 0.40) < 1e-9


def test_menu_refuses_until_three_windows_are_ready(tmp_path, run, monkeypatch):
    monkeypatch.setattr(run, "BUILD_REPLAY", tmp_path / "build.json")
    with pytest.raises(SystemExit):
        run.menu(tmp_path)


def test_public_write_stops_on_a_blocklisted_title_or_reader(run, monkeypatch):
    monkeypatch.setattr(run, "public_ok", lambda t: t != "Hidden Name")
    run.require_public({"id": "story_x", "title": "A Title", "reader": "Some Reader"})
    with pytest.raises(run.PublicTextHit) as e:
        run.require_public({"id": "story_x", "title": "A Title", "reader": "Hidden Name"})
    assert "Hidden Name" not in str(e.value) and "reader" in str(e.value)
