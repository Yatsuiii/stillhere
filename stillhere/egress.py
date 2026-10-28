"""Typed egress: the only way anything leaves this machine for a Meta Model API call (PLAN §4, SPARK-PLAN "Data that leaves the machine").

A request may carry only three payload types:
- TierD: numbers, booleans and identifier-like enums (no prose), e.g. bar pass flags, counts, z values, run IDs, hashes.
- PublicText: text whose SHA-256 is on the allowlist (PREREG lines, D-entries, line templates). Unlisted text is refused.
- PublicAudio (Voice Transcribe only): a public clip (e.g. LibriVox) whose file SHA-256 is on the public-audio allowlist; every other
  file, including any confidential clip, is refused.
- PublicQuestion: at most 280 characters typed on the laptop, refused if any word's salted hash is on the local blocklist built from
  the confidential transcript (no word is stored). The blocklist lives outside the repo; if it is missing, every question is refused (fail closed).
Models: only the listed Standard-mode IDs; any `-contributor` ID is refused, because Meta may train on what it receives.
Tools: a request either offers tools (name, allowlisted description, string parameters whose values are identifier enums) or asks for a
strict json_schema answer, never both and never neither."""
import hashlib
import os
import re
from pathlib import Path
from typing import Union

from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, field_validator, model_validator

ALLOWED_MODELS = frozenset({"muse-spark-1.3", "muse-spark-1.2", "muse-voice-transcribe-1.0",
                            "fireworks/muse-spark-open"})               # Meta's open-weights Muse Spark, served by Fireworks (Meta models only)
ALLOWLIST = Path(__file__).with_name("public_text_allowlist.txt")
AUDIO_ALLOWLIST = Path(__file__).with_name("public_audio_allowlist.txt")
FRAME_ALLOWLIST = Path(__file__).with_name("public_frame_allowlist.txt")
FIREWORKS_MODELS = frozenset(m for m in ALLOWED_MODELS if m.startswith("fireworks/"))
VALIDATION_TABLE = Path(__file__).with_name("validation_table.json")
PLANNER_CODES = frozenset({"none", "population_not_validated", "language_not_validated", "brain_area_not_validated", "condition_not_validated",
                           "stimulus_not_validated", "analysis_not_offered", "clinical_out_of_scope", "schema_invalid", "decline_inconsistent",
                           "list_stimuli", "make_conditions", "check_conditions", "forecast"})
BLOCKLIST = Path(os.environ.get("STILLHERE_QUESTION_BLOCKLIST", Path.home() / ".config/stillhere/question_blocklist.txt"))
_ENUM = re.compile(r"^[A-Za-z0-9_.:|+\-]{1,64}$")          # identifiers, not sentences: no spaces


class EgressRefused(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TierD(_Strict):
    values: dict[str, Union[StrictBool, StrictInt, StrictFloat, None]] = {}
    enums: dict[str, str] = {}

    @field_validator("values", "enums")
    @classmethod
    def _keys(cls, d):
        for k in d:
            if not _ENUM.match(k):
                raise EgressRefused(f"tier-D key is not an identifier: {k!r}")
        return d

    @field_validator("enums")
    @classmethod
    def _enum_values(cls, d):
        for k, v in d.items():
            if not _ENUM.match(v):
                raise EgressRefused(f"tier-D enum {k!r} is not an identifier")
        return d


def _allowlisted() -> frozenset:
    if not ALLOWLIST.exists():
        return frozenset()
    return frozenset(l.split()[0] for l in ALLOWLIST.read_text().splitlines() if l.strip() and not l.startswith("#"))


class PublicText(_Strict):
    text: str

    @field_validator("text")
    @classmethod
    def _listed(cls, t):
        if hashlib.sha256(t.encode()).hexdigest() not in _allowlisted():
            raise EgressRefused("public text is not on the SHA-256 allowlist")
        return t


class PublicAudio(_Strict):
    path: str

    @field_validator("path")
    @classmethod
    def _listed(cls, p):
        f = Path(p)
        if not f.is_file():
            raise EgressRefused("audio file not found")
        listed = frozenset(l.split()[0] for l in AUDIO_ALLOWLIST.read_text().splitlines() if l.strip() and not l.startswith("#")) \
            if AUDIO_ALLOWLIST.exists() else frozenset()
        if hashlib.sha256(f.read_bytes()).hexdigest() not in listed:
            raise EgressRefused("audio is not on the public-audio SHA-256 allowlist")
        return p


class PublicFrame(_Strict):
    """A frame of a public film (e.g. Tears of Steel, CC BY 3.0) whose file SHA-256 is on the public-frame allowlist; anything else,
    including any frame or still of a confidential clip, is refused."""
    path: str

    @field_validator("path")
    @classmethod
    def _listed(cls, p):
        f = Path(p)
        if not f.is_file():
            raise EgressRefused("frame file not found")
        listed = frozenset(l.split()[0] for l in FRAME_ALLOWLIST.read_text().splitlines() if l.strip() and not l.startswith("#")) \
            if FRAME_ALLOWLIST.exists() else frozenset()
        if hashlib.sha256(f.read_bytes()).hexdigest() not in listed:
            raise EgressRefused("frame is not on the public-frame SHA-256 allowlist")
        return p


def table_vocabulary() -> frozenset:
    """Every id in the public validation table, plus the planner's fixed decline and validator codes and tool names."""
    import json as _json
    t = _json.loads(VALIDATION_TABLE.read_text())
    return frozenset(e["id"] for k, v in t.items() if isinstance(v, list) for e in v) | PLANNER_CODES


class TableEnums(_Strict):
    """Planner enums only: every value is an id from the public validation table, a planner decline/validator code or a tool name.
    No numbers and no free text, so it may go to either backend (unlike TierD, which never goes to Fireworks)."""
    enums: dict[str, list[str]]

    @field_validator("enums")
    @classmethod
    def _in_vocabulary(cls, d):
        vocab = table_vocabulary()
        for k, vals in d.items():
            if not _ENUM.match(k) or any(v not in vocab for v in vals):
                raise EgressRefused(f"table enum {k!r} carries a value outside the validation-table vocabulary")
        return d


class PublicQuestion(_Strict):
    text: str

    @field_validator("text")
    @classmethod
    def _scan(cls, t):
        if len(t) > 280:
            raise EgressRefused("question longer than 280 characters")
        salt, blocked = blocklist()
        if any(word_hash(salt, w) in blocked for w in re.findall(r"[a-z']+", t.lower())):
            raise EgressRefused("question contains a blocked word")
        return t


def word_hash(salt: bytes, word: str) -> str:
    return hashlib.sha256(salt + word.encode()).hexdigest()


def blocklist() -> tuple:
    """The laptop-only question blocklist: a 'salt <hex>' line, then one salted SHA-256 per blocked word (no word is stored).
    Missing or malformed means every question is refused (fail closed)."""
    if not BLOCKLIST.exists():
        raise EgressRefused("question blocklist missing: questions are refused until it exists")
    lines = [l.strip() for l in BLOCKLIST.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not lines or not lines[0].startswith("salt ") or not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in lines[1:]):
        raise EgressRefused("question blocklist malformed: questions are refused")
    return bytes.fromhex(lines[0].split()[1]), frozenset(lines[1:])


Part = Union[TierD, PublicText, PublicQuestion, PublicFrame, TableEnums]


class Tool(_Strict):
    """A function the model may call. Every parameter is a required string limited to identifier enum values."""
    name: str
    description: PublicText
    params: dict[str, list[str]]

    @field_validator("name")
    @classmethod
    def _name(cls, n):
        if not _ENUM.match(n):
            raise EgressRefused(f"tool name is not an identifier: {n!r}")
        return n

    @field_validator("params")
    @classmethod
    def _params(cls, d):
        for k, vals in d.items():
            if not _ENUM.match(k) or not vals or not all(_ENUM.match(v) for v in vals):
                raise EgressRefused(f"tool parameter {k!r} must be an identifier with identifier enum values")
        return d


class Request(_Strict):
    model: str
    system: list[PublicText]
    user: list[Part]
    schema_name: str = ""
    json_schema: Union[dict, None] = None
    tools: list[Tool] = []
    parallel_tool_calls: bool = True

    @model_validator(mode="after")
    def _fireworks_public_only(self):
        """Fireworks carries public text, scanned questions and public film frames only: no tier-D values (they come from ds003171)."""
        if self.model in FIREWORKS_MODELS and any(isinstance(p, TierD) for p in self.user):
            raise EgressRefused("tier-D values never go to Fireworks: public text and public film frames only")
        return self

    @model_validator(mode="after")
    def _one_answer_form(self):
        if bool(self.tools) == (self.json_schema is not None):
            raise EgressRefused("a request offers tools or asks for a json_schema answer, exactly one of the two")
        if self.json_schema is not None and not _ENUM.match(self.schema_name):
            raise EgressRefused("a json_schema answer needs an identifier schema_name")
        return self

    @field_validator("model")
    @classmethod
    def _model(cls, m):
        if m.endswith("-contributor") or m not in ALLOWED_MODELS:
            raise EgressRefused(f"model not allowed: {m}")
        return m


def check(req) -> "Request":
    """Refuse anything that is not a fully typed Request (a plain string, a dict, a list of strings, ...)."""
    if not isinstance(req, Request):
        raise EgressRefused(f"only egress.Request may leave the machine, got {type(req).__name__}")
    return req
