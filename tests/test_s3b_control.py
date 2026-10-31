"""S3b's control hook on a stand-in text extractor (scripts/s3b_control.py): one vector kept per word, the plain mean over words, every
word set to that mean with its timing unchanged, other extractors untouched, the class always restored. The real extractor is checked
on the laptop with TRIBE v2 before any S3b run (docs/rules/s3b_late_rule.md)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s3b_control", REPO / "scripts/s3b_control.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


class TimedArray:
    def __init__(self, *, frequency, start, data=None, duration=None):
        self.frequency, self.start, self.data, self.duration = frequency, start, data, duration


class Word:
    def __init__(self, text, start, duration):
        self.text, self.start, self.duration = text, start, duration


class Sentence(Word):
    pass


class FakeText:
    """Yields one static vector per event, as HuggingFaceText._get_timed_arrays does."""
    def __init__(self, vecs):
        self.vecs = vecs

    def _get_timed_arrays(self, events, start, duration):
        for e in events:
            yield TimedArray(frequency=0, start=e.start, duration=e.duration, data=self.vecs[e.text])


VECS = {"a": np.array([1.0, 0.0]), "b": np.array([3.0, 2.0]), "c": np.array([5.0, 10.0])}
WORDS = [Word("a", 0.0, 0.3), Word("b", 0.4, 0.2), Word("c", 1.0, 0.5)]


def run(ext, events):
    return list(ext._get_timed_arrays(events, 0.0, 2.0))


def test_recording_keeps_one_vector_per_word_and_changes_nothing():
    ext = FakeText(VECS)
    with S.recording(ext) as seen:
        first = run(ext, WORDS[:2]); second = run(ext, WORDS[1:])           # two windows sharing word b
    assert len(seen) == 3
    assert [t.data.tolist() for t in first + second] == [VECS[w.text].tolist() for w in WORDS[:2] + WORDS[1:]]
    assert S.clip_mean(seen).tolist() == [3.0, 4.0]                       # each word once, not weighted by the shared window


def test_word_mean_sets_every_word_and_keeps_timing():
    ext = FakeText(VECS); v = np.array([3.0, 4.0])
    with S.word_mean(ext, v) as n:
        out = run(ext, WORDS)
    assert n["words"] == 3
    assert all(t.data.tolist() == [3.0, 4.0] for t in out)
    assert [(t.start, t.duration, t.frequency) for t in out] == [(w.start, w.duration, 0) for w in WORDS]
    out[0].data[0] = 99.0
    assert out[1].data[0] == 3.0 and v[0] == 3.0                           # each word gets its own copy


def test_other_instances_untouched_and_class_restored():
    ext, other = FakeText(VECS), FakeText(VECS)
    orig = FakeText._get_timed_arrays
    with S.word_mean(ext, np.zeros(2)):
        assert [t.data.tolist() for t in run(other, WORDS)] == [VECS[w.text].tolist() for w in WORDS]
    assert FakeText._get_timed_arrays is orig
    with pytest.raises(ZeroDivisionError):
        with S.word_mean(ext, np.zeros(2)):
            1 / 0
    assert FakeText._get_timed_arrays is orig
    assert [t.data.tolist() for t in run(ext, WORDS)] == [VECS[w.text].tolist() for w in WORDS]   # hook off: the extractor as it was


def test_refusals():
    ext = FakeText(VECS)
    with pytest.raises(RuntimeError, match="Sentence"):
        with S.recording(ext):
            run(ext, [Sentence("a", 0.0, 1.0)])
    with pytest.raises(RuntimeError, match="shape"):
        with S.word_mean(ext, np.zeros(3)):
            run(ext, WORDS)
    with pytest.raises(ValueError):
        S.clip_mean({})


def test_inherited_method_is_restored_not_shadowed():
    class Child(FakeText):
        pass
    ext = Child(VECS)
    with S.word_mean(ext, np.zeros(2)):
        assert "_get_timed_arrays" in Child.__dict__
    assert "_get_timed_arrays" not in Child.__dict__ and Child._get_timed_arrays is FakeText._get_timed_arrays
