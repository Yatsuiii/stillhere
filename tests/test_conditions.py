"""conditions.py on synthetic tone-burst "words" with known boundaries, plus one smoke run on the public LibriVox clip when faster-whisper
word timings are on disk. The tone bursts abut and are phase-continuous (the input has no step anywhere), so every splice joins two
non-zero signals at unrelated phases: a hard cut clicks and the crossfade must not."""
from pathlib import Path

import numpy as np
import pytest

from stillhere import conditions as C

SR = 16000
REPO = Path(__file__).resolve().parents[1]


def tones(n_words=12, seed=0):
    rng = np.random.default_rng(seed); durs = rng.uniform(0.15, 0.40, n_words); freqs = rng.uniform(200, 900, n_words)
    words, f_per_sample, t = [], [], 0.0
    for i, (d, f) in enumerate(zip(durs, freqs)):
        n = int(round(d * SR)); f_per_sample.append(np.full(n, f))
        end = t + n / SR; words.append({"start": t, "end": end, "word": f"w{i}" + ("." if i % 4 == 3 else ""), "f": f}); t = end
    phase = np.cumsum(2 * np.pi * np.concatenate(f_per_sample) / SR)            # phase-continuous across word boundaries
    return 0.5 * np.sin(phase), words, max(2 * np.pi * w["f"] * 0.5 / SR for w in words)


def test_intact_equals_input_and_reversed_is_flip():
    x, words, _ = tones(); c = C.build(x, SR, words)
    assert np.array_equal(c["intact"], x) and np.array_equal(c["reversed"], np.flip(x))


def test_duration_preserved_for_every_condition():
    x, words, _ = tones(); c = C.build(x, SR, words)
    for name in C.CONDITIONS:
        assert abs(len(c[name]) - len(x)) <= round(C.DURATION_TOLERANCE_S * SR), name
        assert len(c[name]) == len(x), name                                     # exact by construction


def test_splices_land_on_word_boundaries():
    x, words, _ = tones(); c = C.build(x, SR, words); h = int(round(C.XFADE_S * SR / 2)); L = 2 * h
    cuts = C.cut_points(words, len(x), SR); spans = list(zip(cuts[:-1], cuts[1:])); out_len = 0
    for i, k in enumerate(c["meta"]["word_order"]):                             # each piece (with its margins) is placed L before the end
        a, b = spans[k]; start = 0 if i == 0 else out_len - L; real = start + (h if a > 0 else 0)
        assert np.allclose(c["word_scramble"][real + L:real + (b - a) - L], x[a + L:b - L])   # its interior is the untouched segment
        out_len = start + min(b + h, len(x)) - max(a - h, 0)
    assert out_len == len(x)
    assert cuts[1:-1] == [int(round(w["end"] * SR)) for w in words[:-1]]      # abutting words: cuts at the shared edges


def test_no_clicks_and_a_hard_cut_would_click():
    x, words, max_slope = tones(); c = C.build(x, SR, words)
    for name in ("sentence_scramble", "word_scramble"):
        assert np.max(np.abs(np.diff(c[name]))) <= 1.6 * max_slope, name                # equal-power mixing of two tones peaks near sqrt(2)
    cuts = C.cut_points(words, len(x), SR); spans = list(zip(cuts[:-1], cuts[1:]))
    hard = C.assemble(x, spans, c["meta"]["word_order"], SR, xfade_s=0.0)
    assert np.max(np.abs(np.diff(hard))) > 3 * max_slope                         # the test can see a click


def test_seeded_determinism_and_orders():
    x, words, _ = tones(); a, b = C.build(x, SR, words), C.build(x, SR, words); d = C.build(x, SR, words, seed=1)
    for name in C.CONDITIONS:
        assert np.array_equal(a[name], b[name])
    assert a["meta"]["word_order"] != d["meta"]["word_order"]
    assert a["meta"]["word_order"] != list(range(len(words))) and a["meta"]["sentence_order"] != list(range(a["meta"]["n_sentences"]))
    assert sorted(a["meta"]["expected_words"]["word_scramble"]) == sorted(w["word"] for w in words)
    flat = a["meta"]["expected_words"]["sentence_scramble"]; sents = C.sentences(words)
    assert flat == [words[i]["word"] for s in a["meta"]["sentence_order"] for i in sents[s]]


def test_sentences_split_on_terminal_punctuation():
    w = [{"word": t, "start": 0, "end": 0} for t in ("Once", "upon.", "A", "time!\"", "And", "then")]
    assert C.sentences(w) == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.skipif(not (REPO / "assets" / "librivox_alice_ch01_75s.words.json").exists(), reason="LibriVox word timings not on disk")
def test_smoke_on_the_public_librivox_clip(tmp_path):
    man = C.build_files(REPO / "assets" / "librivox_alice_ch01_75s.wav", REPO / "assets" / "librivox_alice_ch01_75s.words.json", tmp_path, "t")
    assert man["n_words"] > 200 and man["n_sentences"] >= 5
    assert all(f["samples"] == man["samples_in"] for f in man["files"].values())
    x, sr = C.read_wav(REPO / "assets" / "librivox_alice_ch01_75s.wav"); y, _ = C.read_wav(tmp_path / "t_intact.wav")
    assert np.max(np.abs(x - y)) < 1e-4                                          # 16-bit round trip of the untouched clip
