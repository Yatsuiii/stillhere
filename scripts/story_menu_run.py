"""Fresh public stories for the live menu, the I/O around stillhere/story_menu.py (docs/rules/story_menu_rule.md; exploratory, no brain
data). Bulk files go to WORK (outside every repository); small results to results/story_menu/. Prints counts, timings and costs only.

  catalogue   resolve the two collection titles through the LibriVox API (stop unless each is exactly one project), write the pool
  fetch       per pool story: the 64 kb MP3 (archive.org MD5 checked), a bit-exact 16 kHz WAV, faster-whisper word table, the trim
  screen      Voice Transcribe DIARIZATION on 60 s chunks of each trimmed story; bars 1-3; batch cap $25 (stops before a call that
              would cross it); one ledger file for every Part 2 call
  notes       Spark Open content notes for passing stories (Fireworks; Baseten only if Fireworks fails twice); quotes matched and timed
  windows     the window of each passing story; clip WAV, word table, live menu and live clip entries, allowlist lines
  checks      the Voice Transcribe manipulation check of the 4 conditions of each window, recorded as BUILD replays
  menu        once at least 3 windows have 4 cached forecasts and 4 checks: the live menu entries and the screen summary
Real model calls run only with --live; without it every call goes to the stub (tests and dry runs).
usage: python scripts/story_menu_run.py STAGE WORK [--live] [--only ID]"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stillhere import conditions as C  # noqa: E402
from stillhere import story_menu as SM  # noqa: E402

TITLES = ("Short Story Collection Vol. 001", "Short Story Collection Vol. 002")
API = "https://librivox.org/api/feed/audiobooks/"
RESULTS = REPO / "results/story_menu"
VT_RESERVE_USD = 0.10                      # Voice Transcribe publishes no price; the web ledger's worst-case reservation per call
AUDIO_ALLOWLIST = REPO / "stillhere/public_audio_allowlist.txt"
TEXT_ALLOWLIST = REPO / "stillhere/public_text_allowlist.txt"
NOTES_SYSTEM = ("You read the transcript of a short public-domain story read aloud on LibriVox. Go through the tag list one tag at a "
                "time and decide for each tag whether it applies to the story. Return one entry for every tag in the list: the tag, "
                "applies true or false, and, when it applies, up to 3 quotes copied word for word from the transcript, each at most 12 "
                "words, showing where it applies. Also return a description of the story in exactly two sentences.")
NOTES_CALLS, NOTES_TEMPERATURE, QUOTE_MAX_WORDS = 3, 0.6, 12           # rule section 4, amended Oct 29 (cloud ruling of 20:30 UTC)


UA = {"User-Agent": "StillHere-pilot/1.0 (research; public LibriVox audio)"}     # the LibriVox API refuses urllib's default agent


def _get(url: str, timeout: float = 60) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def log(msg: str):
    print(time.strftime("%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


def _om():
    spec = importlib.util.spec_from_file_location("order_map", REPO / "scripts/order_map.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def short_id(title: str) -> str:
    w = SM.norm_title(title).split()
    w = w[1:] if w and w[0] in ("the", "a", "an") else w
    return "story_" + "_".join(w[:4])


# ---- ledger (one file for every Part 2 call) -------------------------------------------------------------------------------------
class Ledger:
    def __init__(self, path: Path):
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list:
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []

    def total(self, kind: str) -> float:
        return sum(r.get("usd", 0.0) for r in self.rows() if r["kind"] == kind)

    def add(self, **row):
        with self.path.open("a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}) + "\n")


class CapReached(RuntimeError):
    pass


# ---- catalogue and pool ----------------------------------------------------------------------------------------------------------
def catalogue(work: Path, fetch_json=None) -> dict:
    get = fetch_json or (lambda url: json.loads(_get(url)))
    books = []
    for t in TITLES:
        d = get(API + "?" + urllib.parse.urlencode({"title": "^" + t, "extended": 1, "format": "json"}))
        hits = [b for b in d.get("books", []) if b["title"].strip() == t]
        if len(hits) != 1:
            raise SystemExit(f"STOP: {t!r} resolves to {len(hits)} LibriVox projects, not exactly one")
        books.append(hits[0])
    P = SM.pool(books)
    doc = {"rule": "docs/rules/story_menu_rule.md", "projects": [{"id": b["id"], "title": b["title"], "archive": b.get("url_iarchive"),
                                                                   "sections": len(b["sections"])} for b in books],
           "sections_seen": [{"book": b["title"], "section": int(s["section_number"]), "title": s["title"], "language": s.get("language"),
                              "readers": len(s.get("readers") or []), "playtime_s": int(s["playtime"]),
                              "public_scan": SM.public_scan(s["title"])} for b in books for s in b["sections"]],
           "pool": P}
    (work / "catalogue.json").write_text(json.dumps(books, indent=1)); RESULTS.mkdir(parents=True, exist_ok=True)
    (work / "pool_private.json").write_text(json.dumps(P, indent=1) + "\n")       # with reader names: laptop only
    # Public: a reader name appears only for a story that reaches the menu (and passes the blocklist check there), so the public pool
    # carries no reader names at all, and no marker of any kind.
    doc["pool"] = [{k: v for k, v in p.items() if k != "reader"} for p in P]
    (RESULTS / "pool.json").write_text(json.dumps(doc, indent=1) + "\n")
    log(f"projects {[p['id'] for p in doc['projects']]}; sections {len(doc['sections_seen'])}; pool {len(P)}")
    return doc


# ---- fetch, words, trim ----------------------------------------------------------------------------------------------------------
def fetch(work: Path, only=None):
    pool = json.loads((work / "pool_private.json").read_text()); om = _om()
    for s in pool:
        sid = short_id(s["title"])
        if only and sid != only:
            continue
        d = work / sid; d.mkdir(parents=True, exist_ok=True); mp3 = d / "section.mp3"
        item, fname = s["listen_url"].split("/download/")[1].split("/", 1)
        files = json.loads(_get(f"https://archive.org/metadata/{item}/files"))["result"]
        md5 = {f["name"]: f.get("md5") for f in files}[fname]
        if not mp3.exists() or hashlib.md5(mp3.read_bytes()).hexdigest() != md5:
            mp3.write_bytes(_get(f"https://archive.org/download/{item}/{fname}", timeout=300))
        assert hashlib.md5(mp3.read_bytes()).hexdigest() == md5, f"{sid}: MD5 differs from archive.org"
        wav = d / "section.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(wav)], check=True)
        t0 = time.time(); words = om.whisper_words(wav)
        (d / "words.json").write_text(json.dumps({"words": words}, indent=1) + "\n")
        info = {"id": sid, **s, "archive_url": f"https://archive.org/download/{item}/{fname}", "mp3_md5": md5,
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "n_words": len(words), "whisper_s": round(time.time() - t0, 1)}
        try:
            first, last = SM.trim(words)
            info.update(trim={"first_word": first, "last_word": last, "start_s": words[first]["start"], "end_s": words[last]["end"]})
        except SM.TrimFail as e:
            info.update(trim=None, failed="trim", trim_reason=str(e))
        (d / "story.json").write_text(json.dumps(info, indent=1) + "\n")
        log(f"{sid}: {len(words)} words in {info['whisper_s']} s; trim {'ok' if info['trim'] else info['trim_reason']}")


def retrim(work: Path):
    """Recomputes each pool story's trim from its saved word table (after a rule change to the trim); prints which trims changed."""
    for p in sorted(work.glob("story_*/story.json")):
        info = json.loads(p.read_text())
        if pool_ids() is not None and info["id"] not in pool_ids():
            continue
        words = json.loads((p.parent / "words.json").read_text())["words"]; old = info.get("trim")
        try:
            first, last = SM.trim(words)
            info.update(trim={"first_word": first, "last_word": last, "start_s": words[first]["start"], "end_s": words[last]["end"]})
            info.pop("failed", None); info.pop("trim_reason", None)
        except SM.TrimFail as e:
            info.update(trim=None, failed="trim", trim_reason=str(e))
        p.write_text(json.dumps(info, indent=1) + "\n")
        log(f"{info['id']}: trim {'unchanged' if info['trim'] == old else 'CHANGED'} ({'ok' if info['trim'] else info['trim_reason']})")


# ---- screen (Voice Transcribe) ---------------------------------------------------------------------------------------------------
def write_chunks(d: Path, info: dict) -> list:
    x, sr = C.read_wav(d / "section.wav"); out = []
    for k, (a, b) in enumerate(SM.chunks(info["trim"]["start_s"], info["trim"]["end_s"])):
        p = d / f"chunk{k:02d}.wav"; sha = C.write_wav(p, x[int(round(a * sr)):int(round(b * sr))], sr)
        out.append({"k": k, "start_s": round(a, 3), "end_s": round(b, 3), "path": str(p), "sha256": sha})
    return out


def allowlist(path: Path, lines: list, marker: str):
    """Appends `sha  description` lines that are not listed yet, under a marker comment."""
    have = {l.split()[0] for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}
    new = [l for l in lines if l.split()[0] not in have]
    if new:
        with path.open("a") as f:
            f.write(f"# {marker}\n" + "\n".join(new) + "\n")
    return len(new)


def pool_ids() -> set:
    """Ids of the current pool (results/story_menu/pool.json); story folders outside it are never screened or used."""
    p = RESULTS / "pool.json"
    return {short_id(s["title"]) for s in json.loads(p.read_text())["pool"]} if p.exists() else None


def screen(work: Path, transcribe, ledger: Ledger, only=None, cap=SM.SCREEN_CAP_USD):
    """transcribe(path) -> Voice Transcribe DIARIZATION JSON. Stops before a call that would take the screen past the cap."""
    from stillhere import egress
    ids = pool_ids()
    stories = sorted((json.loads(p.read_text()) for p in work.glob("story_*/story.json")), key=lambda i: (i["book_id"], i["section"]))
    for info in [i for i in stories if ids is None or i["id"] in ids]:
        sid = info["id"]; d = work / sid
        if (only and sid != only) or info.get("failed") or (d / "screen.json").exists():
            continue
        ch = write_chunks(d, info)
        allowlist(AUDIO_ALLOWLIST, [f"{c['sha256']}  story menu screen chunk {sid} {c['k']:02d} ({c['start_s']}-{c['end_s']} s of "
                                    f"{info['archive_url'].rsplit('/', 1)[1]}; LibriVox, public domain)" for c in ch], f"story menu screen: {sid}")
        res, complete = [], True
        for c in ch:
            raw = d / f"chunk{c['k']:02d}.vt.json"                  # each paid answer is saved at once and reused on a rerun
            got = json.loads(raw.read_text()) if raw.exists() else None
            if not got or got.get("sha256") != c["sha256"]:
                if ledger.total("vt_screen") + VT_RESERVE_USD > cap + 1e-9:
                    complete = False; log(f"{sid}: batch cap ${cap} reached before chunk {c['k']}"); break
                audio = egress.PublicAudio(path=c["path"])
                ledger.add(kind="vt_screen", story=sid, chunk=c["k"], usd=VT_RESERVE_USD, audio_s=round(c["end_s"] - c["start_s"], 3))
                got = {"sha256": c["sha256"], "answer": transcribe(audio)}; raw.write_text(json.dumps(got))
            out = got["answer"]
            # a turn without a speaker label still counts as speech time, under the label None (against bar 1)
            res.append({"k": c["k"], "turns": [{"speaker": t.get("speaker"), "startMs": t.get("startMs", 0), "endMs": t.get("endMs", 0)}
                                               for t in out.get("turns") or []],
                        "unlabelled_turns": sum(1 for t in out.get("turns") or [] if t.get("speaker") is None),
                        "transcript": out.get("transcript") or "", "audio_ms": out.get("audioDurationMs")})
        words = json.loads((d / "words.json").read_text())["words"][info["trim"]["first_word"]:info["trim"]["last_word"] + 1]
        b1 = SM.bar1([r["turns"] for r in res]); b2 = SM.bar2(words); b3 = SM.bar3(" ".join(r["transcript"] for r in res), words)
        failed = None if complete and b1["pass"] and b2["pass"] and b3["pass"] else (
            "cap" if not complete else "1" if not b1["pass"] else "2" if not b2["pass"] else "3")
        doc = {"id": sid, "chunks": len(ch), "called": len(res), "complete": complete, "bar1": b1, "bar2": b2, "bar3": b3,
               "pass": failed is None, "failed_bar": failed, "calls": res}
        (d / "screen.json").write_text(json.dumps(doc, indent=1) + "\n")
        log(f"{sid}: {len(res)}/{len(ch)} chunks; bar1 {b1['min_share']:.3f} bar2 {b2['longest_gap_s']} s bar3 {b3['agreement']}; "
            f"{'pass' if failed is None else 'fail ' + failed}")
        if not complete:
            break


# ---- content notes (Spark Open) --------------------------------------------------------------------------------------------------
def notes_schema() -> dict:
    """One entry per tag, every tag present: "tags" is an object with one required property per tag of the fixed list (the model
    cannot return an empty list, which it did with an array), each entry {tag, applies, quotes}."""
    from stillhere.story_tags import TAGS
    entry = lambda t: {"type": "object", "additionalProperties": False, "required": ["tag", "applies", "quotes"],
                       "properties": {"tag": {"type": "string", "enum": [t]}, "applies": {"type": "boolean"},
                                      "quotes": {"type": "array", "items": {"type": "string"}}}}
    return {"type": "object", "additionalProperties": False, "required": ["tags", "description"],
            "properties": {"tags": {"type": "object", "additionalProperties": False, "required": list(TAGS),
                                    "properties": {t: entry(t) for t in TAGS}},
                           "description": {"type": "string"}}}


def as_list(content) -> dict:
    """The answer with "tags" as a list of entries in the fixed order (accepts the keyed form the schema asks for)."""
    from stillhere.story_tags import TAGS
    if isinstance(content, dict) and isinstance(content.get("tags"), dict):
        return {**content, "tags": [content["tags"].get(t) for t in TAGS]}
    return content


def valid_answer(content) -> bool:
    """Every tag of the fixed list exactly once, applies a boolean, at most 3 quotes (strings)."""
    from stillhere.story_tags import TAGS
    content = as_list(content)
    if not isinstance(content, dict) or not isinstance(content.get("tags"), list) or not isinstance(content.get("description"), str):
        return False
    if not all(isinstance(e, dict) for e in content["tags"]):
        return False
    seen = [e.get("tag") for e in content["tags"]]
    return (sorted(seen) == sorted(TAGS) and all(isinstance(e.get("applies"), bool) and isinstance(e.get("quotes"), list)
                                                  and len(e["quotes"]) <= 3 and all(isinstance(q, str) for q in e["quotes"])
                                                  for e in content["tags"]))


def guard_blocks(quote: str) -> bool:
    """The egress blocklist (laptop-only, built from the confidential clip) or the page's wording guard."""
    from stillhere import blockterms, egress
    sys.path.insert(0, str(REPO))
    from web import guard
    return bool(blockterms.hits(quote, *egress.blocklist())) or guard.clean(quote) != quote


def verified(answer: dict, words: list) -> tuple:
    """(notes, dropped): each applying tag's quotes of at most 12 words found word for word in the story, timed at the run's start;
    dropped counts tags marked applies with no verified quote."""
    toks, owner = SM.tokens(words); out, dropped = [], 0
    for e in answer["tags"]:
        if not e["applies"]:
            continue
        hits = [(SM.match_note(q, toks, owner, words), q) for q in e["quotes"] if len(q.split()) <= QUOTE_MAX_WORDS]
        hits = [(t, q) for t, q in hits if t is not None]
        if hits:
            out += [{"tag": e["tag"], "t": round(float(t), 2), "quote": q} for t, q in hits]
        else:
            dropped += 1
    return out, dropped


def union(answers: list, words: list) -> dict:
    """The card's notes: the union across the calls of tags with at least one verified quote, each tag once, at its earliest
    verified time (quote withheld if the guard blocks it; tag and time stay)."""
    per_call, dropped, best = [], 0, {}
    for a in answers:
        notes, d = verified(a, words); dropped += d; per_call.append(sorted({n["tag"] for n in notes}))
        for n in notes:
            if n["tag"] not in best or n["t"] < best[n["tag"]]["t"]:
                best[n["tag"]] = n
    kept = [{"tag": n["tag"], "t": n["t"], "quote": None if guard_blocks(n["quote"]) else n["quote"]}
            for n in sorted(best.values(), key=lambda n: n["t"])]
    return {"notes": kept, "dropped_applies_without_verified_quote": dropped, "tags_per_call": per_call,
            "tags_from_one_call": sum(1 for t in best if sum(t in c for c in per_call) == 1)}


def keep_notes(raw: list, words: list) -> list:
    """First-commit form (a free list of notes), kept for the record of the first answers only."""
    from stillhere.story_tags import TAGS
    toks, owner = SM.tokens(words); out = []
    for n in raw:
        if n.get("tag") not in TAGS:
            continue
        t = SM.match_note(n.get("quote", ""), toks, owner, words)
        if t is None:
            continue
        out.append({"tag": n["tag"], "t": round(float(t), 2), "quote": None if guard_blocks(n["quote"]) else n["quote"]})
    return sorted(out, key=lambda n: n["t"])


def notes(work: Path, call, ledger: Ledger, only=None):
    """call(model, text, temperature) -> (content, usage). Exactly NOTES_CALLS calls per passing story on one host: Fireworks; Baseten
    for all of them only if a Fireworks call fails (an error or no valid output after 2 attempts). All answers are kept."""
    from stillhere import planner_eval as PE
    ids = pool_ids()
    for p in sorted(work.glob("story_*/screen.json")):
        scr = json.loads(p.read_text()); sid = scr["id"]; d = p.parent
        if (only and sid != only) or (ids is not None and sid not in ids) or not scr["pass"] or (d / "notes.json").exists():
            continue
        info = json.loads((d / "story.json").read_text())
        words = json.loads((d / "words.json").read_text())["words"][info["trim"]["first_word"]:info["trim"]["last_word"] + 1]
        text = " ".join(w["word"] for w in words)
        allowlist(TEXT_ALLOWLIST, [f"{hashlib.sha256(text.encode()).hexdigest()}  story menu transcript {sid} (faster-whisper, LibriVox)",
                                   f"{hashlib.sha256(NOTES_SYSTEM.encode()).hexdigest()}  story menu notes system prompt (amended)"],
                  f"story menu notes: {sid}")
        record, used = [], None
        for model in ("fireworks/muse-spark-open", "baseten/muse-spark-open-1026"):
            host = model.split("/")[0]; answers = []
            for k in range(1, NOTES_CALLS + 1):
                ok = None
                for attempt in (1, 2):
                    try:
                        content, usage = call(model, text, NOTES_TEMPERATURE)
                        usd = PE.cost(usage.get("prompt_tokens"), usage.get("completion_tokens")); good = valid_answer(content)
                        ledger.add(kind="spark_open", story=sid, host=host, call=k, attempt=attempt, temperature=NOTES_TEMPERATURE,
                                   usd=round(usd, 6), prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                                   valid=good)
                        record.append({"host": host, "call": k, "attempt": attempt, "temperature": NOTES_TEMPERATURE, "valid": good,
                                       "usage": usage, "answer": content})
                        if good:
                            ok = as_list(content); break
                    except Exception as e:
                        ledger.add(kind="spark_open", story=sid, host=host, call=k, attempt=attempt, temperature=NOTES_TEMPERATURE,
                                   usd=0.0, error=type(e).__name__)
                        record.append({"host": host, "call": k, "attempt": attempt, "temperature": NOTES_TEMPERATURE, "error": type(e).__name__})
                if ok is None:
                    break
                answers.append(ok)
            if len(answers) == NOTES_CALLS:
                used = (host, answers); break
        if not used:
            (d / "notes_failed.json").write_text(json.dumps({"id": sid, "record": record}, indent=1) + "\n")
            log(f"{sid}: no host gave {NOTES_CALLS} valid answers"); continue
        host, answers = used; u = union(answers, words)
        (d / "notes.json").write_text(json.dumps({"id": sid, "host": host, "temperature": NOTES_TEMPERATURE, "calls": NOTES_CALLS, **u,
                                                  "descriptions": [a["description"] for a in answers], "record": record,
                                                  "rule": "docs/rules/story_menu_rule.md section 4 (amended Oct 29, 20:30 UTC ruling)"},
                                                 indent=1) + "\n")
        log(f"{sid}: {host}; {len(u['notes'])} tags on the card, {u['tags_from_one_call']} from only 1 of 3 calls, "
            f"{u['dropped_applies_without_verified_quote']} applies-without-verified-quote dropped, {sum(n['quote'] is None for n in u['notes'])} quotes withheld")


def live_call(model: str, text: str, temperature: float = None):
    """One real Spark Open json_schema call through the typed egress (keys from env vars only)."""
    from stillhere import egress, meta_client
    req = egress.Request(model=model, system=[egress.PublicText(text=NOTES_SYSTEM)], user=[egress.PublicText(text=text)],
                         schema_name="content_notes", json_schema=notes_schema())
    rows_before = meta_client.USAGE_LOG.read_text().count("\n") if meta_client.USAGE_LOG.exists() else 0
    out = meta_client.call(req, timeout=300, temperature=temperature)
    rows = [json.loads(r) for r in meta_client.USAGE_LOG.read_text().splitlines()[rows_before:]]
    row = next(r for r in reversed(rows) if r.get("schema") == "content_notes" and r.get("model") == model)
    return out["content"], {"prompt_tokens": row.get("prompt_tokens"), "completion_tokens": row.get("completion_tokens")}


def live_transcribe(audio):
    from stillhere import meta_client
    return meta_client.transcribe(audio, mode="DIARIZATION")


# ---- windows, checks, menu ------------------------------------------------------------------------------------------------------
LIVE_CLIPS = REPO / "data/live_clips.json"
LIVE_MENU = REPO / "stillhere/live_menu.json"
BUILD_REPLAY = REPO / "web/replay/build.json"
RESHUFFLE_SEEDS = (20261028, 7, 1, 2, 3, 4)                   # web/stations.py RESHUFFLE_SEEDS (Reshuffle is offered only these)


class PublicTextHit(SystemExit):
    pass


def public_ok(text: str) -> bool:
    """True if the text holds no word or phrase of the laptop-only blocklist (stillhere.blockterms; fails closed if it is missing)."""
    from stillhere import blockterms, egress
    return not blockterms.hits(str(text or ""), *egress.blocklist())


def require_public(entry: dict, fields=("title", "reader")):
    """Stops before a public-facing write if a menu story's title or reader hits the blocklist; the message carries no text."""
    for f in fields:
        if not public_ok(entry.get(f, "")):
            raise PublicTextHit(f"STOP: a menu story's {f} hits the laptop blocklist (story {entry.get('id')}); nothing was written for it")


def screened(work: Path) -> list:
    """Stories in catalogue order with their screen result (None when not screened)."""
    out, ids = [], pool_ids()
    for p in sorted(work.glob("story_*/story.json")):
        info = json.loads(p.read_text()); scr = p.parent / "screen.json"
        if ids is not None and info["id"] not in ids:
            continue
        out.append((info, json.loads(scr.read_text()) if scr.exists() else None))
    return sorted(out, key=lambda x: (x[0]["book_id"], x[0]["section"]))


def windows(work: Path, only=None) -> list:
    """The window of each passing story, first MENU_SIZE in catalogue order: clip WAV and word table into data/live_clips/, the clip
    into data/live_clips.json, its intact SHA-256 and every Reshuffle recipe build onto the public-audio allowlist."""
    from worker import recipe as R
    idx = json.loads(LIVE_CLIPS.read_text()); kept = []
    for info, scr in screened(work):
        if not scr or not scr["pass"] or len(kept) == SM.MENU_SIZE:
            continue
        d = work / info["id"]; words = json.loads((d / "words.json").read_text())["words"]
        win = SM.window(words, info["trim"]["first_word"], info["trim"]["last_word"])
        (d / "window.json").write_text(json.dumps(win, indent=1) + "\n")
        if not win or (only and info["id"] != only):
            continue
        x, sr = C.read_wav(d / "section.wav"); a, b = int(round(win["start_s"] * sr)), int(round(win["end_s"] * sr))
        out = REPO / "data/live_clips"; sid = info["id"]
        wav_sha = C.write_wav(out / f"{sid}.wav", x[a:b], sr)
        cw = [{"word": w["word"].replace("\u2014", "-"), "start": round(w["start"] - win["start_s"], 3), "end": round(w["end"] - win["start_s"], 3)}
              for w in words[win["first_word"]:win["last_word"] + 1]]
        wtxt = json.dumps({"words": cw}, indent=1) + "\n"; (out / f"{sid}.words.json").write_text(wtxt)
        entry = {"id": sid, "title": info["title"], "reader": info["reader"], "licence": "public domain (LibriVox)",
                 "source": info["archive_url"], "start_s": win["start_s"], "dur_s": round(win["end_s"] - win["start_s"], 3),
                 "seconds": round((b - a) / sr, 3), "sr": sr, "wav": f"data/live_clips/{sid}.wav", "wav_sha256": wav_sha,
                 "words": f"data/live_clips/{sid}.words.json", "words_sha256": hashlib.sha256(wtxt.encode()).hexdigest(),
                 "n_words": len(cw), "sentences": win["sentences"], "rule": "docs/rules/story_menu_rule.md"}
        require_public(entry)
        idx["clips"] = [c for c in idx["clips"] if c["id"] != sid] + [entry]
        LIVE_CLIPS.write_text(json.dumps(idx, indent=1) + "\n"); kept.append(entry)
        lines = [f"{wav_sha}  story menu window {sid} ({win['start_s']}-{win['end_s']} s of {info['archive_url'].rsplit('/', 1)[1]}; LibriVox)"]
        for cond in R.CONDITIONS:
            for seed in ([0] if cond in R.SEEDLESS else RESHUFFLE_SEEDS):
                bld = R.build_condition({"clip_id": sid, "condition": cond, "seed": seed}, R.load_index())
                lines.append(f"{bld.wav_sha256}  live build {sid} {cond} seed {bld.recipe['seed']} (worker/recipe.py build_condition)")
        n = allowlist(AUDIO_ALLOWLIST, lines, f"story menu window and Reshuffle builds: {sid}")
        log(f"{sid}: window {win['start_s']}-{win['end_s']} s, {win['sentences']} sentences, {len(cw)} words; {n} allowlist lines")
    return kept


def checks(work: Path, transcribe, ledger: Ledger, only=None, idx=None, root=REPO):
    """The Voice Transcribe manipulation check of the 4 conditions of each window (default seed), recorded as BUILD replays keyed
    clip:condition in the format web/stations.py reads."""
    import tempfile

    from stillhere import egress
    from stillhere import manipulation_check as MC
    from web import stations
    from worker import recipe as R
    idx = idx or R.load_index(); rec = json.loads(BUILD_REPLAY.read_text()) if BUILD_REPLAY.exists() else {}
    for c in [c for c in idx["clips"] if c.get("rule") == "docs/rules/story_menu_rule.md" and (not only or c["id"] == only)]:
        orig = R.build_condition({"clip_id": c["id"], "condition": "intact", "seed": 0}, idx, root)
        for cond in R.CONDITIONS:
            key = f"{c['id']}:{cond}"
            if key in rec:
                continue
            bld = R.build_condition({"clip_id": c["id"], "condition": cond, "seed": idx["default_seed"]}, idx, root)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"{c['id']}_{cond}.wav"; p.write_bytes(R.wav_bytes(bld.audio, bld.sr))
                audio = egress.PublicAudio(path=str(p))
                ledger.add(kind="vt_check", story=c["id"], condition=cond, usd=VT_RESERVE_USD, audio_s=c["seconds"])
                out = transcribe(audio)
            chk = MC.check_condition(cond, out.get("transcript") or "", [w["word"] for w in bld.words], [w["word"] for w in orig.words])
            rec[key] = {"mode": "live", "condition": cond, "seed": bld.recipe["seed"], "pass": chk["pass"],
                        "checklist": stations.checklist(cond, chk), "recorded_by": "scripts/story_menu_run.py checks"}
            BUILD_REPLAY.write_text(json.dumps(rec, indent=1) + "\n")
            log(f"{key}: check {'pass' if chk['pass'] else 'fail'}")


def ready(work: Path) -> list:
    """Windows whose 4 default-seed forecasts are cached (data/forecast_cache.json) and whose 4 conditions are checked (BUILD replays)."""
    from worker import recipe as R
    idx = R.load_index(); cache = json.loads((REPO / "data/forecast_cache.json").read_text())["entries"]
    rec = json.loads(BUILD_REPLAY.read_text()) if BUILD_REPLAY.exists() else {}
    out = []
    for c in [c for c in idx["clips"] if c.get("rule") == "docs/rules/story_menu_rule.md"]:
        keys = [R.cache_key({"clip_id": c["id"], "condition": cond, "seed": 0 if cond in R.SEEDLESS else idx["default_seed"]}, idx)
                for cond in R.CONDITIONS]
        if all(k in cache for k in keys) and all(f"{c['id']}:{cond}" in rec for cond in R.CONDITIONS):
            out.append(c["id"])
    return out


def menu(work: Path, skip=()):
    """Section 7: the live menu changes only once at least 3 windows are ready. The fresh stories go first in catalogue order, then
    alice_ch02, alice_ch01, alice_ch03 (their fields kept; ch02 is the backup), then any other listed clip; failed stories are listed
    by title and bar only. Every menu story's title, reader and quotes pass the blocklist gate before the file is written."""
    ok = [i for i in ready(work) if i not in skip]
    if len(ok) < 3:
        raise SystemExit(f"menu unchanged: {len(ok)} window(s) ready, the rule needs at least 3")
    m = json.loads(LIVE_MENU.read_text()); idx = json.loads(LIVE_CLIPS.read_text()); failed = []; fresh = []
    clips = {c["id"]: c for c in idx["clips"]}
    for info, scr in screened(work):
        if info.get("failed") == "trim":
            failed.append({"title": info["title"], "bar": "trim"})
        elif scr and not scr["pass"]:
            failed.append({"title": info["title"], "bar": scr["failed_bar"] if scr["failed_bar"] in ("1", "2", "3") else "3"})
        elif info["id"] in ok:
            c = clips[info["id"]]; d = work / c["id"]
            nt = json.loads((d / "notes.json").read_text()) if (d / "notes.json").exists() else {"notes": [], "host": None}
            entry = {"id": c["id"], "title": c["title"], "short": c["title"], "reader": c["reader"], "source": c["source"], "start_s": c["start_s"],
                     "dur_s": c["dur_s"], "intact_wav_sha256": c["wav_sha256"], "public_scan": SM.public_scan(c["title"]),
                     "screen": {"archive_url": c["source"], "sentences": c["sentences"],
                                "notes": [dict(n) for n in nt["notes"]], "notes_host": nt["host"]}}
            require_public(entry)
            for n in entry["screen"]["notes"]:
                if n.get("quote") is not None and not public_ok(n["quote"]):
                    n["quote"] = None                                 # withheld, tag and time kept (rule 2c)
            fresh.append(entry)
    for f in failed:
        if not public_ok(f["title"]):
            raise PublicTextHit("STOP: a failed story's title hits the laptop blocklist; nothing was written")
    old = {c["id"]: c for c in m["clips"]}
    alice = [old[i] for i in ("alice_ch02", "alice_ch01", "alice_ch03") if i in old]
    rest = [c for c in m["clips"] if c["id"] not in {e["id"] for e in fresh} | {a["id"] for a in alice} and not c["id"].startswith("story_")]
    m["clips"] = fresh + alice + rest
    m["screen_summary"] = {"n_on_menu": len(fresh), "failed": failed, "rule": "docs/rules/story_menu_rule.md"}
    LIVE_MENU.write_text(json.dumps(m, indent=1) + "\n")
    on = {e["id"] for e in fresh}                                        # reader names only for stories on the menu
    for c in idx["clips"]:
        if c.get("rule") == "docs/rules/story_menu_rule.md" and c["id"] not in on:
            c.pop("reader", None)
    LIVE_CLIPS.write_text(json.dumps(idx, indent=1) + "\n")
    log(f"live menu: {[e['id'] for e in fresh]} then {[a['id'] for a in alice]}; {len(failed)} failed stories listed")


def summary(work: Path, ledger: Ledger) -> dict:
    """results/story_menu/screen_summary.json: per pool story its title, trim, bars and windows (no reader names, no transcripts), and
    every Part 2 call's count, audio seconds and tokens from the ledger. Voice Transcribe publishes no price and its answers carry no
    cost, so its dollars are the $0.10 reservation per call, not a charge; Spark Open dollars use stillhere/planner_eval.py's rate."""
    from stillhere import planner_eval as PE
    stories = []
    for info, scr in screened(work):
        d = work / info["id"]; win = json.loads((d / "window.json").read_text()) if (d / "window.json").exists() else None
        nt = json.loads((d / "notes.json").read_text()) if (d / "notes.json").exists() else None
        stories.append({"id": info["id"], "title": info["title"], "playtime_s": info["playtime_s"],
                        "trim": "ok" if info.get("trim") else "failed",
                        "screen": None if not scr else {"chunks": scr["chunks"], "bar1_min_share": round(scr["bar1"]["min_share"], 4),
                                                        "bar2_longest_gap_s": scr["bar2"]["longest_gap_s"], "bar3_agreement": scr["bar3"]["agreement"],
                                                        "pass": scr["pass"], "failed_bar": scr["failed_bar"]},
                        "window": win, "notes_tags": None if not nt else [n["tag"] for n in nt["notes"]]})
    rows = ledger.rows(); kinds = sorted({r["kind"] for r in rows}); cost = {}
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        cost[k] = {"rows": len(rs), "audio_s": round(sum(r.get("audio_s", 0) for r in rs), 1),
                   "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rs), "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rs),
                   "usd": round(sum(r.get("usd", 0) for r in rs), 4), "errors": sum(1 for r in rs if r.get("error"))}
    doc = {"rule": "docs/rules/story_menu_rule.md", "label": "exploratory", "stories": stories, "spend": cost,
           "price_notes": {"voice_transcribe": "no published price and no cost in its answers: usd is the $0.10-per-call reservation; the "
                                               "charge needs the billing page",
                           "spark_open": f"tokens x ${PE.PRICE_IN * 1e6:.2f} per million input, ${PE.PRICE_OUT * 1e6:.2f} per million output "
                                         "(stillhere/planner_eval.py rate)"}}
    for s in stories:
        if not public_ok(s["title"]):
            raise PublicTextHit(f"STOP: a story title hits the laptop blocklist (story {s['id']}); summary not written")
    (RESULTS / "screen_summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("work"); ap.add_argument("--live", action="store_true")
    ap.add_argument("--only"); ap.add_argument("--skip", action="append", default=[]); a = ap.parse_args(argv); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(work / "part2_ledger.jsonl")
    if a.stage == "catalogue":
        catalogue(work)
    elif a.stage == "fetch":
        fetch(work, a.only)
    elif a.stage == "retrim":
        retrim(work)
    elif a.stage == "screen":
        if not a.live:
            raise SystemExit("screen makes real Voice Transcribe calls: pass --live (tests use the stub)")
        screen(work, live_transcribe, ledger, a.only)
    elif a.stage == "windows":
        windows(work, a.only)
    elif a.stage == "checks":
        if not a.live:
            raise SystemExit("checks makes real Voice Transcribe calls: pass --live (tests use the stub)")
        checks(work, live_transcribe, ledger, a.only)
    elif a.stage == "menu":
        menu(work, a.skip)
    elif a.stage == "summary":
        print(json.dumps(summary(work, ledger)["spend"], indent=1))
    elif a.stage == "notes":
        if not a.live:
            raise SystemExit("notes makes real Spark Open calls: pass --live (tests use the stub)")
        notes(work, live_call, ledger, a.only)
    else:
        raise SystemExit(f"unknown stage {a.stage}")


if __name__ == "__main__":
    main()
