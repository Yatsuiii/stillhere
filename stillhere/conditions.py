"""Stimulus conditions built from one public clip and its word timings (StillHere Pilot, exploratory): intact, sentence scramble,
word scramble and reversed.

- Cuts fall at word boundaries: between two consecutive words the cut is the midpoint of the gap between them (their shared edge when
  they abut), so no cut lands inside a word. Word k's segment runs from the cut before it to the cut after it; the first segment starts
  at 0 and the last ends at the end of the clip, so the segments tile the clip.
- Sentences are runs of words up to a word ending in . ! or ? (closing quotes ignored); trailing words without one form the last
  sentence. Sentence scramble permutes whole sentences and keeps the word order inside each; word scramble permutes words.
- A join between two pieces that were not neighbours in the clip is a raised-cosine, equal-power crossfade of XFADE_S seconds. Each piece
  carries XFADE_S / 2 of its original neighbourhood on each side, and the crossfade spends exactly those margins, so the output length
  equals the input length (stated tolerance: at most one crossfade). Where a piece is followed by its own successor the audio simply
  continues, so intact equals the input sample for sample and only true splices are crossfaded.
- Reversed is the time-reversed clip (numpy.flip).
- Permutations use numpy default_rng(SEED) and are redrawn (deterministically) if they come out as the identity."""
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

SEED = 20261028
XFADE_S = 0.010
DURATION_TOLERANCE_S = XFADE_S
CONDITIONS = ("intact", "sentence_scramble", "word_scramble", "reversed")
_CLOSERS = "\"'”’)"


def cut_points(words: list, n: int, sr: int) -> list:
    """Sample indices of the cuts: 0, one between each pair of consecutive words, n."""
    inner = [int(round((a["end"] + b["start"]) / 2 * sr)) for a, b in zip(words[:-1], words[1:])]
    cuts = [0] + [min(max(c, 0), n) for c in inner] + [n]
    if any(b < a for a, b in zip(cuts[:-1], cuts[1:])):
        raise ValueError("word timings overlap or are out of order")
    return cuts


def sentences(words: list) -> list:
    """Lists of word indices, one per sentence."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        if w["word"].rstrip(_CLOSERS).endswith((".", "!", "?")):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def _permutation(n: int, rng) -> list:
    if n < 2:
        return list(range(n))
    while True:
        p = [int(i) for i in rng.permutation(n)]
        if p != list(range(n)):
            return p


def _window(m: int) -> tuple:
    """Equal-power raised-cosine fade-in and fade-out of m samples (fin**2 + fout**2 = 1)."""
    rc = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
    return np.sqrt(rc), np.sqrt(1.0 - rc)


def assemble(x: np.ndarray, spans: list, order: list, sr: int, xfade_s: float = XFADE_S) -> np.ndarray:
    """Concatenate spans[k] for k in order. Non-neighbour joins get the crossfade; neighbour joins continue the audio unchanged."""
    x = np.asarray(x, dtype=float); n = len(x); h = int(round(xfade_s * sr / 2)); L = 2 * h
    piece = lambda k: x[max(spans[k][0] - h, 0):min(spans[k][1] + h, n)]
    out = piece(order[0]).copy()
    for prev, k in zip(order[:-1], order[1:]):
        p = piece(k)
        if L == 0:
            out = np.concatenate([out, p]); continue
        if k == prev + 1:                                   # original neighbours: out ends where p's first L samples already are
            out = np.concatenate([out, p[L:]]); continue
        m = min(L, len(out), len(p)); fin, fout = _window(m)
        out = np.concatenate([out[:len(out) - m], out[len(out) - m:] * fout + p[:m] * fin, p[m:]])
    return out


def build(x: np.ndarray, sr: int, words: list, seed: int = SEED, xfade_s: float = XFADE_S) -> dict:
    """The four conditions and what the manipulation check needs (orders and the expected word sequences)."""
    x = np.asarray(x, dtype=float); n = len(x)
    cuts = cut_points(words, n, sr); spans = list(zip(cuts[:-1], cuts[1:]))
    rng = np.random.default_rng(seed)
    sents = sentences(words); s_order = _permutation(len(sents), rng)
    sentence_words = [i for s in s_order for i in sents[s]]
    w_order = _permutation(len(words), rng)
    out = {"intact": assemble(x, spans, list(range(len(words))), sr, xfade_s),
           "sentence_scramble": assemble(x, spans, sentence_words, sr, xfade_s),
           "word_scramble": assemble(x, spans, w_order, sr, xfade_s),
           "reversed": np.flip(x).copy()}
    text = [w["word"] for w in words]
    out["meta"] = {"seed": seed, "xfade_s": xfade_s, "n_words": len(words), "n_sentences": len(sents), "sentence_order": s_order,
                   "word_order": w_order, "expected_words": {"intact": text, "sentence_scramble": [text[i] for i in sentence_words],
                                                            "word_scramble": [text[i] for i in w_order]}}
    return out


def read_wav(path) -> tuple:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono wav")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768.0, w.getframerate()


def write_wav(path, x: np.ndarray, sr: int) -> str:
    y = np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_files(wav, words_json, out_dir, stem: str, seed: int = SEED) -> dict:
    """Write the four condition wavs and a manifest (orders, lengths, SHA-256s) for a public clip."""
    x, sr = read_wav(wav); words = json.loads(Path(words_json).read_text())["words"]
    c = build(x, sr, words, seed); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); files = {}
    for name in CONDITIONS:
        f = out_dir / f"{stem}_{name}.wav"; files[name] = {"file": f.name, "sha256": write_wav(f, c[name], sr), "samples": int(len(c[name]))}
    man = {"label": "exploratory", "source": Path(wav).name, "sr": sr, "samples_in": int(len(x)), "files": files, **c["meta"]}
    (out_dir / f"{stem}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man
