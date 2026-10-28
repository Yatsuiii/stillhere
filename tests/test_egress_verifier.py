"""Unit tests: the egress wrapper refuses everything except typed tier-D numbers, allowlisted public text and scanned questions, and
refuses contributor models; the verifier enforces the D18/D23 card rules. No network."""
from pathlib import Path

import pytest

from stillhere import blockterms, egress, verifier

PT = Path(egress.__file__).with_name("public_text")
SYS = egress.PublicText(text=(PT / "desk_system.txt").read_text())
SCHEMA = {"type": "object", "properties": {"line_id": {"type": "string", "enum": ["a"]}}, "required": ["line_id"], "additionalProperties": False}


def req(**kw):
    base = dict(model="muse-spark-1.3", system=[SYS], user=[egress.TierD(values={"p7_pass": True})], schema_name="t", json_schema=SCHEMA)
    base.update(kw)
    return egress.Request(**base)


def test_plain_string_and_dict_refused():
    for bad in ("tell me about the clip", {"model": "muse-spark-1.3"}, ["x"]):
        with pytest.raises(egress.EgressRefused):
            egress.check(bad)


def test_typed_request_accepted():
    assert egress.check(req()).model == "muse-spark-1.3"


@pytest.mark.parametrize("m", ["muse-spark-1.3-contributor", "muse-spark-1.2-contributor", "muse-wonder-1.0", "gpt-4o"])
def test_contributor_and_unlisted_models_refused(m):
    with pytest.raises(ValueError):
        req(model=m)


def test_unlisted_public_text_refused():
    with pytest.raises(ValueError):
        egress.PublicText(text="any text that is not on the allowlist")


def test_listed_public_text_accepted():
    assert egress.PublicText(text=(PT / "d23_opener.txt").read_text()).text.startswith("Labs study")


@pytest.mark.parametrize("bad", [{"note": "a sentence with spaces"}, {"x": "word" * 20}])
def test_tierd_enum_must_be_identifier(bad):
    with pytest.raises(ValueError):
        egress.TierD(enums=bad)


def test_tierd_rejects_strings_in_values():
    with pytest.raises(ValueError):
        egress.TierD(values={"z": "1.2"})


def test_question_fail_closed_without_blocklist(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "BLOCKLIST", tmp_path / "missing.txt")
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="Can we say language fades under deep sedation?")


def test_question_blocked_word_and_length(tmp_path, monkeypatch):
    salt = bytes(range(32)); bl = tmp_path / "bl.txt"; monkeypatch.setattr(egress, "BLOCKLIST", bl)
    bl.write_text(f"# salted\nsalt {salt.hex()}\n{blockterms.term_hash(salt, 'secretword')}\n")
    assert egress.PublicQuestion(text="Can we say language fades under deep sedation?")
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="what about SecretWord here")
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="x" * 281)


def test_question_blocklist_stores_no_word_and_refuses_malformed_files(tmp_path, monkeypatch):
    bl = tmp_path / "bl.txt"; monkeypatch.setattr(egress, "BLOCKLIST", bl)
    bl.write_text("secretword\n")                                   # plaintext words: malformed, so every question is refused
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="Can we say language fades under deep sedation?")
    salt = b"\x07" * 32; bl.write_text(f"salt {salt.hex()}\n{blockterms.term_hash(salt, 'secretword')}\n")
    assert "secretword" not in bl.read_text() and egress.PublicQuestion(text="is hearing above the rest null")
    with pytest.raises(ValueError):
        egress.PublicQuestion(text="secretword?")


def card(text, kind="group", n=0):
    subj = {"kind": "group", "n": 15} if kind == "group" else {"kind": "volunteer", "id": "vol-01"}
    return {"subject": subj, "text": verifier.HEADER.format(n=n) + " " + text, "controls_scanned": n}


def test_clean_group_card_passes():
    assert verifier.verify(card(f"Language under deep sedation: {verifier.NOT_SHOWN}. {verifier.DISCLAIMER}")) == []


def test_header_required():
    c = card("x"); c["text"] = "no header"
    assert any("header" in v for v in verifier.verify(c))


@pytest.mark.parametrize("bad", ["worth a full follow-up scan", "preserved speech processing", "the patient is aware", "any hospital",
                                 "we recommend", "a diagnosis", "works on any story", "life support"])
def test_blocked_words_caught(bad):
    assert verifier.verify(card(bad))


def test_fixed_lines_exempt():
    assert verifier.verify(card(verifier.DECLINE_CLINICAL)) == []


def test_volunteer_card_rules():
    assert verifier.verify(card(f"HEARING z 1.2. {verifier.GROUP_LEVEL}.", kind="volunteer")) == []
    assert verifier.verify(card(f"{verifier.RESPONSE_SHOWN}. {verifier.GROUP_LEVEL}.", kind="volunteer"))
    assert verifier.verify(card("HEARING z 1.2.", kind="volunteer"))


def test_group_verdict_with_header_passes():
    # the fixed header already carries "Not a diagnosis or medical advice.", which D23 requires on any card with verdict words
    assert verifier.verify(card(verifier.RESPONSE_SHOWN)) == []


def test_public_audio_only_allowlisted_clips(tmp_path, monkeypatch):
    import hashlib
    import pytest
    from stillhere import egress, meta_client
    listed, other = tmp_path / "public.wav", tmp_path / "other.wav"
    listed.write_bytes(b"public clip"); other.write_bytes(b"any other clip")
    allow = tmp_path / "allow.txt"; allow.write_text(f"{hashlib.sha256(b'public clip').hexdigest()}  test clip\n")
    monkeypatch.setattr(egress, "AUDIO_ALLOWLIST", allow)
    assert egress.PublicAudio(path=str(listed)).path == str(listed)
    with pytest.raises(ValueError):                  # pydantic wraps EgressRefused (a ValueError) in its ValidationError
        egress.PublicAudio(path=str(other))
    with pytest.raises(ValueError):
        egress.PublicAudio(path=str(tmp_path / "missing.wav"))
    with pytest.raises(egress.EgressRefused):
        meta_client.transcribe(str(listed))              # a bare path is not a PublicAudio
