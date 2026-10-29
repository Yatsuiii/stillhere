"""scripts/order_map.py (order-map go/no-go, exploratory) on synthetic words with known sentence lengths: piece rules, the shuffle,
the timeline mapping, the kept seconds after every cut, the scale-free change measure and the 3-of-3 verdict."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("order_map", ROOT / "scripts/order_map.py")
O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
SR = 100


def words_for(sentence_lens: list, per_word: float = 0.5) -> list:
    """Abutting words; each sentence's last word ends with a full stop."""
    out, t = [], 0.0
    for L in sentence_lens:
        n = int(round(L / per_word))
        for i in range(n):
            out.append({"word": "w" + ("." if i == n - 1 else ""), "start": t, "end": t + per_word}); t += per_word
    return out


def test_long_sentences_are_the_pieces():
    w = words_for([10, 12, 8, 10]); g, kind = O.pieces(w, 40 * SR, SR)
    assert kind == "sentences" and len(g) == 4


def test_short_sentences_give_blocks_of_15_s():
    w = words_for([3, 4, 5, 4, 3, 5, 4, 4, 4, 4]); g, kind = O.pieces(w, 40 * SR, SR)
    spans = O.piece_spans(w, g, 40 * SR, SR)
    assert kind == "blocks" and len(g) == 2 and all(b - a >= 15 for a, b in spans)


def test_midpoint_fallback_when_no_15_s_split():
    w = words_for([4, 24, 6, 6]); g, kind = O.pieces(w, 40 * SR, SR)
    spans = O.piece_spans(w, g, 40 * SR, SR)
    assert kind == "two blocks at the midpoint" and [round(a) for a, _ in spans] == [0, 28]


def test_one_piece_when_nothing_splits():
    w = words_for([4, 30, 6]); g, _ = O.pieces(w, 40 * SR, SR)
    assert len(g) == 1 and O.shuffle_order(1) == [0]


def test_shuffle_is_never_identity_and_words_move_with_pieces():
    w = words_for([3, 4, 5, 4, 3, 5, 4, 4, 4, 4]); c = O.build(np.zeros(40 * SR), SR, w)
    order = c["meta"]["order"]; assert order != sorted(order)
    moved = c["words"]["sentence_shuffle"]; spans = c["meta"]["piece_spans_s"]
    first = c["meta"]["pieces"][order[0]][0]
    assert abs(moved[0]["start"] - (w[first]["start"] - spans[order[0]][0])) < 1e-9
    assert len(moved) == len(w) and c["words"]["reversed"] == []
    assert len(c["audio"]["sentence_shuffle"]) == 40 * SR


def test_kept_seconds_drop_5_s_after_every_cut():
    spans = [(0.0, 20.0), (20.0, 40.0)]; kept = O.kept_seconds(spans, [1, 0], 40)
    assert [s for s, _ in kept] == list(range(5, 20)) + list(range(25, 40))
    assert dict(kept)[5] == 25.5 and dict(kept)[25] == 5.5            # shuffled second -> original-time centre


def test_unmoved_piece_keeps_its_start():
    spans = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]; kept = O.kept_seconds(spans, [0, 2, 1], 30)
    ss = [s for s, _ in kept]
    assert 0 in ss and 10 not in ss and 15 in ss and 20 not in ss and 25 in ss


def test_unshuffle_recovers_an_order_free_forecast():
    rng = np.random.default_rng(0); I = rng.normal(size=(40, 3)); spans = [(0.0, 20.0), (20.0, 40.0)]
    S = np.concatenate([I[20:], I[:20]])                               # a forecast that only moves with the audio
    Ik, Uk, ok = O.align_pair(I, S, O.kept_seconds(spans, [1, 0], 40))
    assert len(ok) == 30 and np.allclose(O.parcel_change(Ik, Uk), 0)


def test_change_is_scale_free():
    rng = np.random.default_rng(1); Ik = rng.normal(size=(30, 4)); Uk = Ik + rng.normal(scale=0.3, size=(30, 4))
    k = np.array([1.0, 10.0, 0.01, 3.0])
    assert np.allclose(O.parcel_change(Ik, Uk), O.parcel_change(Ik * k, Uk * k))


def test_verdict_needs_three_of_three():
    assert O.verdict([True, True, True]) == "GO"
    assert O.verdict([True, True, False]) == "NO-GO" and O.verdict([True, True]) == "NO-GO"
