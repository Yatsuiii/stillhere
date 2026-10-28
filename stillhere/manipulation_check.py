"""Manipulation checks for the Pilot stimulus conditions (exploratory). The rules and thresholds below were committed before any Voice
Transcribe output for these conditions or for the diarisation clip was read.

Words are lower-cased with punctuation stripped. For a condition, the reference is the local faster-whisper transcript of the intact
clip; `expected` is that word list in the condition's own order (stillhere.conditions build meta).
- recall: words of the reference found in the transcript, counted as a multiset, over the reference length (are the words still there).
- order_vs_original: longest common subsequence of the transcript with the reference, over the reference length.
- order_vs_expected: the same with the condition's expected order.
Rules: intact keeps its words in order; sentence scramble keeps its words and follows the new sentence order better than the original;
word scramble keeps most words but breaks the original order; reversed has no word check (its recall is reported as intelligibility only).

Diarisation gate: on the known-turns two-reader clip, Voice Transcribe may be used as a speaker gate only if it returns exactly two
speaker labels, at least 90% of the audio inside its turns carries the label of the true reader under the best one-to-one label mapping,
and its number of speaker changes (after merging consecutive turns with the same label) is within 1 of the true number."""
import re
from collections import Counter

RULES = {"intact": {"recall_min": 0.85, "order_vs_original_min": 0.80},
         "sentence_scramble": {"recall_min": 0.80, "order_vs_expected_min": 0.70, "expected_minus_original_min": 0.10},
         "word_scramble": {"recall_min": 0.60, "order_vs_original_max": 0.50},
         "reversed": {}}
DIARIZATION = {"n_labels": 2, "agreement_min": 0.90, "changes_tolerance": 1}


def words(text) -> list:
    if isinstance(text, (list, tuple)):
        text = " ".join(text)
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'").replace("-", " "))


def recall(hyp: list, ref: list) -> float:
    return sum((Counter(hyp) & Counter(ref)).values()) / len(ref) if ref else 0.0


def lcs(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def check_condition(name: str, transcript: str, expected: list, original: list) -> dict:
    hyp, exp, ref = words(transcript), words(expected), words(original)
    m = {"recall": recall(hyp, ref), "order_vs_original": lcs(hyp, ref) / len(ref) if ref else 0.0,
         "order_vs_expected": lcs(hyp, exp) / len(exp) if exp else 0.0, "n_words_heard": len(hyp), "n_words_reference": len(ref)}
    r = RULES[name]
    if name == "intact":
        ok = m["recall"] >= r["recall_min"] and m["order_vs_original"] >= r["order_vs_original_min"]
    elif name == "sentence_scramble":
        ok = (m["recall"] >= r["recall_min"] and m["order_vs_expected"] >= r["order_vs_expected_min"]
              and m["order_vs_expected"] - m["order_vs_original"] >= r["expected_minus_original_min"])
    elif name == "word_scramble":
        ok = m["recall"] >= r["recall_min"] and m["order_vs_original"] <= r["order_vs_original_max"]
    else:
        ok = None                                               # reversed: no word check; recall is reported as intelligibility only
    return {**m, "pass": ok, "rule": r}


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def diarization(turns: list, truth: list) -> dict:
    """turns: [{speaker, startMs, endMs}] from Voice Transcribe; truth: [{reader, start_ms, end_ms}]."""
    labels = sorted({t["speaker"] for t in turns}); readers = sorted({t["reader"] for t in truth})
    ov = {(lab, r): sum(_overlap(t["startMs"], t["endMs"], u["start_ms"], u["end_ms"]) for t in turns if t["speaker"] == lab
                        for u in truth if u["reader"] == r) for lab in labels for r in readers}
    pred_ms = sum(max(0, t["endMs"] - t["startMs"]) for t in turns)
    best = 0.0
    if len(labels) == len(readers) == 2:
        best = max(ov[(labels[0], readers[0])] + ov[(labels[1], readers[1])], ov[(labels[0], readers[1])] + ov[(labels[1], readers[0])])
    agreement = best / pred_ms if pred_ms else 0.0
    merged = [t["speaker"] for i, t in enumerate(turns) if i == 0 or t["speaker"] != turns[i - 1]["speaker"]]
    changes, true_changes = len(merged) - 1, sum(1 for a, b in zip(truth[:-1], truth[1:]) if a["reader"] != b["reader"])
    ok = (len(labels) == DIARIZATION["n_labels"] and agreement >= DIARIZATION["agreement_min"]
          and abs(changes - true_changes) <= DIARIZATION["changes_tolerance"])
    return {"n_labels": len(labels), "n_turns": len(turns), "agreement": agreement, "speaker_changes": changes,
            "true_speaker_changes": true_changes, "pass": ok, "criterion": DIARIZATION}
