"""Fresh public stories for the live menu: the pure parts of docs/rules/story_menu_rule.md (exploratory; no brain data). Pool selection
and the public-scan exclusion (section 1), the trim (2), the three screen bars (3), content-note matching (4) and the window (5). The
I/O (catalogue, downloads, faster-whisper, Voice Transcribe, Spark Open, forecasts) lives in scripts/story_menu_run.py."""
import re

from stillhere import conditions as C
from stillhere import manipulation_check as MC

LENGTH_S = (240, 1800)                    # section 1: listed length 4:00 to 30:00 inclusive (amended Oct 29, 19:55 UTC)
POOL_SIZE = 25
START_PHRASES = ("librivox org", "librivox dot org", "libravox org", "libravox", "read by", "recorded by", "reading by")   # amended 20:13 UTC
END_PHRASES = ("end of", "this recording is in the public domain", "the end")
START_WITHIN_S, END_WITHIN_S = 90.0, 60.0
CHUNK_S = 60.0
BAR1_MIN_SHARE, BAR2_MAX_GAP_S, BAR3_MIN_AGREEMENT = 0.95, 5.0, 0.95
SCREEN_CAP_USD = 35.0                     # amended Oct 29, 19:55 UTC (was $25)
PAD_S, WINDOW_MAX_S, WINDOW_MIN_S, WINDOW_MIN_SENTENCES = 0.3, 40.0, 30.0, 6
MENU_SIZE = 10

# Section 1 exclusion: known public story-fMRI stimuli. Titles as published; each is compared lower-cased with punctuation removed, with
# and without spaces (dataset ids are written without spaces), so the check can only exclude more, never less.
EXCLUDED = {
    "ds002345": ["21styear", "The 21st Year", "bigbang", "black", "I Knew You Were Black", "bronx", "Running from the Bronx", "forgot",
                 "The Man Who Forgot Ray Bradbury", "friends", "himym", "lucy", "merlin", "milkywayoriginal", "milkywaysynonyms",
                 "milkywayvodka", "Milky Way", "notthefall", "It's Not the Fall That Gets You", "notthefallintact", "notthefalllongscram",
                 "notthefallshortscram", "pieman", "Pie Man", "piemanpni", "prettymouth", "Pretty Mouth and Green My Eyes", "santa",
                 "seinfeld", "shame", "shapesphysical", "shapessocial", "sherlock", "slumlordreach", "Slumlord",
                 "Reach for the Stars One Small Step at a Time", "tunnel", "Tunnel Under the World", "upintheair", "vinny"],
    "ds003643": ["Le Petit Prince", "The Little Prince"],
    "ds002322": ["Alice's Adventures in Wonderland", "Down the Rabbit-Hole", "Alice's Adventures in Wonderland chapter 1"],
    "ds003020": ["alternateithicatom", "avatar", "howtodraw", "legacy", "life", "myfirstdaywiththeyankees", "naked", "odetostepfather",
                 "souls", "undertheinfluence", "adollshouse", "adventuresinsayingyes", "buck", "exorcism", "eyespy", "fromboyhoodtofatherhood",
                 "hangtime", "haveyoumethimyet", "inamoment", "itsabox", "sloth", "stagefright", "swimmingwithastronauts", "thatthingonmyarm",
                 "theclosetthatateeverything", "tildeath", "wheretheressmoke", "treasureisland", "goingthelibertyway", "kiksuya"],
    "wehbe2014": ["Harry Potter and the Philosopher's Stone", "Harry Potter and the Sorcerer's Stone", "The Midnight Duel"],
    "naturalstories": ["Boar", "The Boar", "Aqua", "Matchstick Seller", "The Matchstick Seller", "King of Birds", "The King of Birds",
                       "Elvis", "Mr Sticky", "High School", "Roswell", "Tulips", "Tourettes",
                       "The Little Match Girl"],                 # excluded by name: its adapted text is in Natural Stories (amended rule)
}


def norm_title(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", str(t).lower()).split())


def public_scan(title: str):
    """The dataset whose stimulus list holds this title (section 1), or None."""
    n = norm_title(title)
    for ds, titles in EXCLUDED.items():
        for s in titles:
            m = norm_title(s)
            if n == m or n.replace(" ", "") == m.replace(" ", ""):
                return ds
    return None


def pool(books: list) -> list:
    """Sections in catalogue order (books in the rule's order, sections by section_number) that are English, list exactly one reader,
    last 4:00-12:00, and are not a known public stimulus; the first POOL_SIZE."""
    out = []
    for b in books:
        for s in sorted(b["sections"], key=lambda s: int(s["section_number"])):
            ok = (str(s.get("language", "")).strip() == "English" and len(s.get("readers") or []) == 1
                  and LENGTH_S[0] <= int(s["playtime"]) <= LENGTH_S[1] and public_scan(s["title"]) is None)
            if ok:
                out.append({"book_id": b["id"], "book": b["title"], "section": int(s["section_number"]), "title": s["title"],
                            "reader": s["readers"][0]["display_name"], "playtime_s": int(s["playtime"]), "listen_url": s["listen_url"]})
            if len(out) == POOL_SIZE:
                return out
    return out


def tokens(words: list) -> tuple:
    """Normalised tokens of a word table (manipulation_check.words per word) and, for each token, its word index."""
    toks, owner = [], []
    for i, w in enumerate(words):
        for t in MC.words(w["word"]):
            toks.append(t); owner.append(i)
    return toks, owner


def _find(toks: list, phrase: str) -> list:
    p = phrase.split()
    return [i for i in range(len(toks) - len(p) + 1) if toks[i:i + len(p)] == p]


class TrimFail(ValueError):
    pass


def trim(words: list) -> tuple:
    """(first word index, last word index) of the trimmed story (section 2), on whole sentences."""
    toks, owner = tokens(words); sents = C.sentences(words); total = words[-1]["end"]
    hits = [(i, p) for p in START_PHRASES for i in _find(toks, p) if words[owner[i]]["start"] < START_WITHIN_S]
    if not hits:
        raise TrimFail("no LibriVox preamble found")
    i, p = max(hits)                                             # the last occurrence of any phrase
    after = owner[i + len(p.split()) - 1]
    starts = [s for s in sents if s[0] > after]
    if not starts:
        raise TrimFail("no sentence after the preamble")
    first = starts[0][0]
    ends = [(i, p) for p in END_PHRASES for i in _find(toks, p) if words[owner[i]]["start"] >= total - END_WITHIN_S]
    complete = [s for s in sents if words[s[-1]]["word"].rstrip(C._CLOSERS).endswith((".", "!", "?"))]
    if ends:
        q = owner[min(ends)[0]]
        done = [s for s in complete if s[-1] < q and s[0] >= first]
    else:
        done = [s for s in complete if s[0] >= first]
    if not done:
        raise TrimFail("no complete sentence in the story")
    return first, done[-1][-1]


def chunks(t0: float, t1: float) -> list:
    """Consecutive CHUNK_S spans of [t0, t1]; the last may be shorter."""
    out, a = [], t0
    while a < t1 - 1e-9:
        out.append((a, min(a + CHUNK_S, t1))); a += CHUNK_S
    return out


def bar1(turns_per_chunk: list) -> dict:
    """One voice: in every chunk one speaker label covers >= 95% of the speech time inside Voice Transcribe's turns."""
    shares = []
    for turns in turns_per_chunk:
        tot = {}
        for t in turns:
            tot[t["speaker"]] = tot.get(t["speaker"], 0) + max(0, t["endMs"] - t["startMs"])
        s = sum(tot.values()); shares.append(max(tot.values()) / s if s else 0.0)
    return {"min_share": min(shares) if shares else 0.0, "pass": bool(shares) and min(shares) >= BAR1_MIN_SHARE}


def bar2(words: list) -> dict:
    gap = max((b["start"] - a["end"] for a, b in zip(words[:-1], words[1:])), default=0.0)
    return {"longest_gap_s": round(gap, 3), "pass": gap <= BAR2_MAX_GAP_S}


def bar3(vt_text: str, words: list) -> dict:
    fw = [t for w in words for t in MC.words(w["word"])]; vt = MC.words(vt_text)
    a = MC.lcs(vt, fw) / len(fw) if fw else 0.0
    return {"agreement": round(a, 4), "pass": a >= BAR3_MIN_AGREEMENT}


def match_note(quote: str, toks: list, owner: list, words: list):
    """Time (s, in the original file) of the first contiguous run of the quote's normalised words in the story, or None."""
    q = MC.words(quote)
    if not q:
        return None
    hit = _find(toks, " ".join(q))
    return words[owner[hit[0]]]["start"] if hit else None


def window(words: list, first: int, last: int) -> dict:
    """The earliest valid candidate (section 5): from each sentence start, add sentences while the padded clip is <= 40.0 s; valid if
    >= 30.0 s and >= 6 sentences."""
    sents = [s for s in C.sentences(words) if s[0] >= first and s[-1] <= last]
    for i in range(len(sents)):
        best = None
        for j in range(i, len(sents)):
            a, b = words[sents[i][0]]["start"] - PAD_S, words[sents[j][-1]]["end"] + PAD_S
            if b - a > WINDOW_MAX_S:
                break
            best = (j, a, b)
        if best:
            j, a, b = best
            if b - a >= WINDOW_MIN_S and j - i + 1 >= WINDOW_MIN_SENTENCES:
                return {"first_word": sents[i][0], "last_word": sents[j][-1], "start_s": round(a, 3), "end_s": round(b, 3),
                        "sentences": j - i + 1}
    return None
