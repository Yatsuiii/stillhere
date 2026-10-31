"""S3b's exact control (PREREG section 9; amendment to D26 (c), cloud ruling of Oct 31, 21:00 UTC): each word's Llama vector replaced
by the clip-mean vector, word timing unchanged, then TRIBE v2 as usual.

Where the hook sits. TRIBE v2's text lane is neuralset's HuggingFaceText. It turns words into model input in _get_timed_arrays: one
static vector per word (from _get_data, cached per word and context, then its layer aggregation), each placed over the word's start and
duration. The 2 Hz frames are built from those after this method, so the vectors it yields are per word, before any framing. (The ruling
names get_static; HuggingFaceText overrides _get_timed_arrays and never calls get_static, so the hook is on _get_timed_arrays.)

Two context managers patch the extractor's class for one instance only and always restore it:
  recording(ext)  passes every vector through unchanged and keeps one per word (keyed by start, duration and text);
  word_mean(ext, v)  yields every word with data v (the same array for all), start and duration unchanged, and counts the words it set.
clip_mean(seen) is the plain mean over the recorded words, each word once. With neither manager active the extractor is untouched, which is
the "hook off" route the self-check compares with the scored forecasts. No torch import here."""
import contextlib

import numpy as np


def word_key(event) -> tuple:
    return (round(float(event.start), 4), round(float(event.duration), 4), str(event.text))


def _same(ta, data):
    """A copy of TimedArray ta with its data replaced (frequency, start and duration unchanged)."""
    return type(ta)(frequency=ta.frequency, start=ta.start, duration=ta.duration, data=data)


@contextlib.contextmanager
def _patched(ext, on_word):
    cls = type(ext); orig = cls._get_timed_arrays; own = "_get_timed_arrays" in cls.__dict__   # inherited: delete on exit, not shadow

    def patched(self, events, start, duration):
        if self is not ext:
            yield from orig(self, events, start, duration)
            return
        events = list(events)
        arrays = list(orig(self, events, start, duration))
        if len(arrays) != len(events):
            raise RuntimeError(f"text extractor yielded {len(arrays)} arrays for {len(events)} events")
        for event, ta in zip(events, arrays):
            if type(event).__name__ != "Word":
                raise RuntimeError(f"unexpected {type(event).__name__} event in the text lane")
            yield on_word(event, ta)

    cls._get_timed_arrays = patched
    try:
        yield
    finally:
        if own:
            cls._get_timed_arrays = orig
        else:
            del cls._get_timed_arrays


@contextlib.contextmanager
def recording(ext):
    seen = {}

    def keep(event, ta):
        seen.setdefault(word_key(event), np.array(ta.data, copy=True))
        return ta

    with _patched(ext, keep):
        yield seen


def clip_mean(seen: dict) -> np.ndarray:
    if not seen:
        raise ValueError("no word vectors recorded")
    vs = list(seen.values())
    if any(v.shape != vs[0].shape for v in vs):
        raise ValueError("word vectors differ in shape")
    return np.mean(np.stack(vs).astype("float64"), axis=0).astype(vs[0].dtype)


@contextlib.contextmanager
def word_mean(ext, v):
    v = np.asarray(v); count = {"words": 0}

    def fixed(event, ta):
        if np.shape(ta.data) != v.shape:
            raise RuntimeError(f"word vector shape {np.shape(ta.data)} differs from the clip mean's {v.shape}")
        count["words"] += 1
        return _same(ta, v.copy())

    with _patched(ext, fixed):
        yield count
