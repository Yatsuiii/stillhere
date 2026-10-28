"""The blocklist builder with the transcriber and the common-word list stubbed: salted hashes only (no word stored), common words
never blocked on their own, names and quoted 4-word phrases blocked, file mode 600, the decrypted audio deleted even when transcription
fails, refusal outside Yatsu's home or inside the repo, only the SHA-256 returned."""
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

from stillhere import egress

spec = importlib.util.spec_from_file_location("bqb", Path(__file__).resolve().parents[1] / "scripts" / "build_question_blocklist.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
COMMON = lambda: {"the", "went", "can", "say", "language", "fades", "under", "deep", "sedation", "does", "fade"}


def test_builder_writes_salted_hashes_deletes_audio_and_returns_only_the_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    audio = tmp_path / "private.wav"; audio.write_bytes(b"RIFF fake"); out = tmp_path / ".config" / "stillhere" / "bl.txt"
    h = B.build(audio, out, transcriber=lambda p: "The Zorblax went to Quintaville.\nCan we say language fades?", common=COMMON)
    assert not audio.exists() and re.fullmatch(r"[0-9a-f]{64}", h) and h == hashlib.sha256(out.read_bytes()).hexdigest()
    assert oct(out.stat().st_mode & 0o777) == "0o600" and "zorblax" not in out.read_text().lower()
    monkeypatch.setattr(egress, "BLOCKLIST", out)
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="what did zorblax hear")
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="So can we say language fades?")                         # a quoted 4-word phrase
    assert egress.PublicQuestion(text="Does language fade under deep sedation?")             # common words stay usable


def test_frequent_words_pass_and_names_stay_blocked():
    text = "Morning, Rosa. Are you reading the paper?\nI'm sure and I said When\nwhen the garden opens"
    common = {"morning", "rosa", "are", "you", "reading", "the", "paper", "i'm", "sure", "and", "said", "when", "garden", "opens"}
    t = B.blocked_terms(text, common)
    assert "rosa" in t and "morning rosa are you" in t                                     # a name however frequent; phrases
    assert not {"reading", "paper", "morning", "when", "i'm", "said", "garden"} & t        # ordinary words on their own


def test_audio_deleted_even_if_transcription_fails_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    audio = tmp_path / "private.wav"; audio.write_bytes(b"x")
    def boom(p): raise RuntimeError("model missing")
    with pytest.raises(RuntimeError):
        B.build(audio, tmp_path / "bl.txt", transcriber=boom, common=COMMON)
    assert not audio.exists()
    outside = Path("/tmp") / "not_home.wav"
    with pytest.raises(SystemExit):
        B.build(outside, tmp_path / "bl.txt", transcriber=lambda p: "")


def test_keep_audio_leaves_the_file_and_still_refuses_outside_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    audio = tmp_path / "private.wav"; audio.write_bytes(b"x")
    B.build(audio, tmp_path / "bl.txt", transcriber=lambda p: "zorblax", keep_audio=True, common=COMMON)
    assert audio.exists()
    with pytest.raises(SystemExit):
        B.build(Path("/tmp") / "not_home.wav", tmp_path / "bl.txt", transcriber=lambda p: "", keep_audio=True)
