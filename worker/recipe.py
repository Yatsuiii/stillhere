"""Recipes for the live TRIBE worker and the page's BUILD station (docs/jobs_api.md). A recipe is {clip_id, condition, seed} and nothing
else: clip_id names one of the committed public LibriVox clips in data/live_clips.json, condition is one of stillhere/conditions.py's
four, seed an integer (0 for intact and reversed, which ignore it). build_condition turns a recipe into audio and the word table moved
with it, with stillhere/conditions.build, so the BUILD station (Voice Transcribe check) and the worker (TRIBE forecast) read the same
samples. numpy and the standard library only, so the web backend can import it.

Word tables: each committed clip's word timings (local faster-whisper base.en, data/live_clips/<id>.words.json) moved with the pieces:
every word keeps its offset inside its word segment, and segments are laid end to end in the condition's order (the crossfade margins
shift a word by at most XFADE_S / 2). Reversed has no words, so no word reaches TRIBE's text lane."""
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from stillhere import conditions as C  # noqa: E402

INDEX = REPO / "data/live_clips.json"
CONDITIONS = C.CONDITIONS                                  # intact, sentence_scramble, word_scramble, reversed
SEEDLESS = ("intact", "reversed")
DEFAULT_SEED = C.SEED
MAX_SEED = 2**31 - 1
MAX_SECONDS = 40.0


class RecipeRefused(ValueError):
    """The recipe is outside the schema or names a clip that is not on the list. The worker posts reason bad_recipe."""


@dataclass
class Built:
    recipe: dict
    audio: np.ndarray
    sr: int
    words: list = field(default_factory=list)
    wav_sha256: str = ""


def load_index(path=INDEX) -> dict:
    return json.loads(Path(path).read_text())


def clip_entry(clip_id: str, index: dict) -> dict:
    for c in index["clips"]:
        if c["id"] == clip_id:
            return c
    raise RecipeRefused("clip_id is not on the live clip list")


def normalize(recipe: dict, index: dict) -> dict:
    """The canonical recipe, or RecipeRefused. Exactly the three keys; seed an int in [0, 2^31 - 1]; seed 0 where it is ignored."""
    if not isinstance(recipe, dict) or set(recipe) != {"clip_id", "condition", "seed"}:
        raise RecipeRefused("a recipe has exactly clip_id, condition and seed")
    cid, cond, seed = recipe["clip_id"], recipe["condition"], recipe["seed"]
    if not isinstance(cid, str):
        raise RecipeRefused("clip_id must be a string")
    clip_entry(cid, index)
    if cond not in CONDITIONS:
        raise RecipeRefused("condition is not one of " + ", ".join(CONDITIONS))
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise RecipeRefused("seed must be an integer from 0 to 2^31 - 1")
    return {"clip_id": cid, "condition": cond, "seed": 0 if cond in SEEDLESS else seed}


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def cache_key(recipe: dict, index: dict) -> str:
    """SHA-256 of canonical {clip_id, condition, seed, models, wav_sha256, words_sha256} (docs/jobs_api.md)."""
    r = normalize(recipe, index); c = clip_entry(r["clip_id"], index)
    return hashlib.sha256(canonical({**r, "models": index["models"], "wav_sha256": c["wav_sha256"],
                                     "words_sha256": c["words_sha256"]})).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_clip(clip_id: str, index: dict, root=REPO) -> tuple:
    """(x, sr, words) for a listed clip, after checking both files against the index's SHA-256s."""
    c = clip_entry(clip_id, index); wav, wj = Path(root) / c["wav"], Path(root) / c["words"]
    if _sha_file(wav) != c["wav_sha256"] or _sha_file(wj) != c["words_sha256"]:
        raise RecipeRefused("clip files do not match the live clip list")
    x, sr = C.read_wav(wav)
    if len(x) / sr > MAX_SECONDS + 1e-6:
        raise RecipeRefused("clip is longer than the 40 s cap")
    return x, sr, json.loads(wj.read_text())["words"]


def moved_words(words: list, n: int, sr: int, order: list) -> list:
    """The word table of a condition that lays word segments end to end in `order` (indices into words)."""
    cuts = C.cut_points(words, n, sr); out, pos = [], 0
    for i in order:
        a, b = cuts[i], cuts[i + 1]; d = (pos - a) / sr; w = words[i]
        out.append({"word": w["word"], "start": round(w["start"] + d, 4), "end": round(w["end"] + d, 4)})
        pos += b - a
    return out


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """The 16-bit PCM mono wav exactly as stillhere/conditions.write_wav writes it."""
    import wave
    y = np.clip(np.round(np.asarray(audio) * 32768.0), -32768, 32767).astype("<i2"); buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
    return buf.getvalue()


def build_condition(recipe: dict, index: dict = None, root=REPO) -> Built:
    index = index or load_index(); r = normalize(recipe, index)
    x, sr, words = load_clip(r["clip_id"], index, root)
    c = C.build(x, sr, words, seed=r["seed"] if r["condition"] not in SEEDLESS else DEFAULT_SEED)
    meta = c["meta"]
    if r["condition"] == "intact":
        order = list(range(len(words)))
    elif r["condition"] == "sentence_scramble":
        sents = C.sentences(words); order = [i for s in meta["sentence_order"] for i in sents[s]]
    elif r["condition"] == "word_scramble":
        order = meta["word_order"]
    else:
        order = None
    mw = [] if order is None else moved_words(words, len(x), sr, order)
    audio = np.asarray(c[r["condition"]], dtype=float)
    return Built(recipe=r, audio=audio, sr=sr, words=mw, wav_sha256=hashlib.sha256(wav_bytes(audio, sr)).hexdigest())


def word_rate(words: list, seconds: int) -> list:
    """Words per second: word onsets counted in 1 s bins [s, s + 1)."""
    out = [0] * seconds
    for w in words:
        s = int(np.floor(w["start"]))
        if 0 <= s < seconds:
            out[s] += 1
    return out


def allowlist_lines(seeds: list, index: dict = None, root=REPO) -> list:
    """public_audio_allowlist.txt lines (sha256, two spaces, description) for every listed clip x condition x seed; intact and
    reversed once each (they ignore the seed), and duplicates dropped."""
    index = index or load_index(); out, seen = [], set()
    for c in index["clips"]:
        for cond in CONDITIONS:
            for seed in ([0] if cond in SEEDLESS else seeds):
                b = build_condition({"clip_id": c["id"], "condition": cond, "seed": seed}, index, root)
                if b.wav_sha256 not in seen:
                    seen.add(b.wav_sha256)
                    out.append(f"{b.wav_sha256}  live build {c['id']} {cond} seed {b.recipe['seed']} (worker/recipe.py build_condition)")
    return out
