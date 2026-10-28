"""The manipulation-check rules and the diarisation gate criterion, fixed before any Voice Transcribe output was read, on synthetic
transcripts; and the DIARIZATION transcription path with the HTTP call stubbed (no key in the usage log, Standard model id only)."""
import json

import httpx
import pytest

from stillhere import egress, meta_client
from stillhere import manipulation_check as M

REF = "once or twice she peeped into the book her sister was reading but it had no pictures".split()


def test_criteria_are_the_committed_ones():
    assert M.DIARIZATION == {"n_labels": 2, "agreement_min": 0.90, "changes_tolerance": 1}
    assert M.RULES["intact"] == {"recall_min": 0.85, "order_vs_original_min": 0.80}
    assert M.RULES["word_scramble"] == {"recall_min": 0.60, "order_vs_original_max": 0.50} and M.RULES["reversed"] == {}


def test_condition_rules_on_synthetic_transcripts():
    assert M.check_condition("intact", " ".join(REF), REF, REF)["pass"]
    half = REF[8:] + REF[:8]                                                     # two "sentences" swapped
    s = M.check_condition("sentence_scramble", " ".join(half), half, REF)
    assert s["pass"] and s["order_vs_expected"] == 1.0 and s["order_vs_original"] < 0.9
    shuffled = [REF[i] for i in (5, 12, 0, 9, 3, 15, 7, 1, 14, 10, 4, 13, 2, 8, 11, 6)]
    assert M.check_condition("word_scramble", " ".join(shuffled), shuffled, REF)["pass"]
    assert not M.check_condition("word_scramble", " ".join(REF), shuffled, REF)["pass"]          # order not broken
    assert not M.check_condition("intact", "something else entirely", REF, REF)["pass"]
    assert M.check_condition("reversed", "", REF, REF)["pass"] is None and M.words("Didn't, Alice-like.") == ["didn't", "alice", "like"]


def test_diarization_criterion():
    truth = [{"reader": r, "start_ms": i * 10000, "end_ms": (i + 1) * 10000} for i, r in enumerate("ABABAB")]
    good = [{"speaker": s, "startMs": i * 10000 + 300, "endMs": (i + 1) * 10000 - 200} for i, s in enumerate("XYXYXY")]
    assert M.diarization(good, truth)["pass"]
    one = [{**t, "speaker": "X"} for t in good]
    assert not M.diarization(one, truth)["pass"]                                  # a single label fails
    swapped = [{**t, "speaker": s} for t, s in zip(good, "XYYXXY")]
    assert not M.diarization(swapped, truth)["pass"]                              # wrong attributions fail


def test_diarization_mode_posts_to_the_batch_endpoint_and_logs_no_text(tmp_path, monkeypatch):
    import hashlib
    clip = tmp_path / "c.wav"; clip.write_bytes(b"RIFF....WAVE"); allow = tmp_path / "allow.txt"
    allow.write_text(hashlib.sha256(clip.read_bytes()).hexdigest() + "  test\n"); monkeypatch.setattr(egress, "AUDIO_ALLOWLIST", allow)
    log = tmp_path / "usage.jsonl"; monkeypatch.setattr(meta_client, "USAGE_LOG", log); monkeypatch.setattr(meta_client, "_key", lambda: "KEY123")
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return httpx.Response(200, json={"transcript": "secret words", "audioDurationMs": 1000, "turns": [{"speaker": "A"}]},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(meta_client.httpx, "post", post)
    out = meta_client.transcribe(egress.PublicAudio(path=str(clip)), mode="DIARIZATION")
    req = json.loads(seen["files"]["request"][1])
    assert seen["url"].endswith("/asr/transcribe") and req == {"mode": "DIARIZATION", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    assert out["turns"] == [{"speaker": "A"}] and "KEY123" not in log.read_text() and "secret" not in log.read_text()
    with pytest.raises(ValueError):
        meta_client.transcribe(egress.PublicAudio(path=str(clip)), mode="PUSH_TO_TALK")
