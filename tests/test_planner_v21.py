"""Planner v2.1 with a stub client (no network): v2 left as published; plain words for every code and id; the v2.1 schema (clip_id,
condition, seed); one plan call and no tool call; routing by stimulus, including the LibriVox-under-Le-Petit-Prince bug v2's tool loop
had; outputs for another clip refused; stated defaults; the live-menu line and the examples file in plain, allowed words."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from stillhere import egress, planner, planner_v21 as P

ROOT = Path(__file__).resolve().parent.parent
Q = planner.public("q_plan_6yo")                      # an allowlisted question; the stub decides the answer
BANNED = re.compile(r"\b(aware|awareness|conscious|consciousness|hospital|detect\w*|diagnos\w*|validated|understanding|comprehension)\b"
                    r"|a test for patients|before anyone looked|" + "\u2014", re.I)


class Fake:
    def __init__(self, *replies):
        self.replies, self.reqs = list(replies), []

    def __call__(self, req):
        egress.check(req)
        self.reqs.append(req)
        return self.replies.pop(0)


def spec(**kw):
    s = {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "analysis": "forecast_tribe_stock",
         "brain_areas": ["language_network"], "conditions": ["intact", "sentence_scramble"], "decline_reason": "none",
         "clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 7}
    s.update(kw)
    return s


def lpp(**kw):
    return spec(stimulus="le_petit_prince", clip_id=None, condition=None, seed=None, **kw)


def test_v2_is_left_as_published():
    listed = {line.split()[1]: line.split()[0] for line in (ROOT / "stillhere/public_text_allowlist.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    assert hashlib.sha256((ROOT / "stillhere/public_text/planner_system.txt").read_bytes()).hexdigest() == listed["planner_system.txt"]
    assert "clip_id" not in planner.study_schema(planner.load_enums())["properties"]
    card = planner.build_card(planner.decline_spec("condition_not_validated"), [], planner.load_enums())
    assert "Nearest validated option" in card.text            # v2's published wording; v2.1 changes it, v2 does not


def test_every_code_and_id_reads_in_plain_words():
    enums = P.load_enums()
    for code in planner.DECLINE_REASONS[1:]:
        text = P.card_text(P.decline_spec(code), P.route(P.decline_spec(code), enums), [], enums)
        assert "Nearest tested option" in text and code not in text and "_" not in text and not BANNED.search(text), text
    table = json.loads(planner.TABLE.read_text())
    for axis, entries in table.items():
        for e in entries if isinstance(entries, list) else []:
            assert "_" not in P.plain_id(e["id"]) and not BANNED.search(P.plain_id(e["id"])), e["id"]
    for c in enums["clip_id"]:
        assert "_" not in P.plain_id(c)


def test_schema_adds_clip_condition_seed():
    enums = P.load_enums()
    s = P.study_schema(enums)
    assert set(s["required"]) == set(P.FIELDS) and s["additionalProperties"] is False
    assert s["properties"]["seed"] == {"type": ["integer", "null"]}
    assert s["properties"]["clip_id"]["enum"] == sorted(enums["clip_id"]) + [None]
    assert set(enums["clip_id"]) == {c["id"] for c in P.live_menu()["clips"]}
    ok = spec()
    assert P.validate_answer(ok, enums) == []
    assert P.validate_answer(spec(clip_id="alice_ch99"), enums) == ["schema_invalid"]
    assert P.validate_answer(spec(seed=-1), enums) == ["schema_invalid"]
    assert P.validate_answer(spec(seed=True), enums) == ["schema_invalid"]
    assert P.validate_answer(spec(condition="reversed"), enums) == ["decline_inconsistent"]      # not among the spec's conditions
    assert P.validate_answer(spec(condition="intact"), enums) == ["schema_invalid"]           # the live run is the manipulated one
    assert P.validate_answer({k: v for k, v in ok.items() if k != "seed"}, enums) == ["schema_invalid"]
    egress.TableEnums(enums={"clip_id": enums["clip_id"]})                                    # clip ids may go to any backend


def test_one_plan_call_and_no_tool_call():
    fake = Fake({"content": spec()})
    card = P.plan(Q, "spark-open", fake)
    assert len(fake.reqs) == 1 and not fake.reqs[0].tools and fake.reqs[0].json_schema is not None
    assert card.route["kind"] == "live" and [s["station"] for s in card.results] == ["build", "check", "forecast"]
    assert all(s["output"]["status"] == "pending" for s in card.results)                      # no executor: pending, no stand-in


def test_v2_tool_loop_gave_le_petit_prince_plans_librivox_outputs_and_v21_does_not():
    # v2: Fork B's Baseten runs. The model's tool calls on a Le Petit Prince plan ran the LibriVox clip's tools.
    lpp_v2 = {k: v for k, v in lpp().items() if k in P.FIELDS[:7]}
    calls = {"tool_calls": [{"name": "make_conditions", "arguments": {}}, {"name": "forecast", "arguments": {}}]}
    out = planner.run_tools(lpp_v2, "spark-open-baseten", Fake(calls))
    assert out[0]["output"]["source"] == planner.LIBRIVOX_WAV.name                              # the bug, reproduced on v2
    assert "audio_sha256" in out[1]["output"]
    # v2.1: the same plan ends on Le Petit Prince's evidence panel; no LibriVox output anywhere on the card.
    fake = Fake({"content": lpp()})
    librivox_tools = {st: (lambda rt: {"clip_id": "alice_ch01", "condition": "intact", "seed": 20261028, "source": "librivox"})
                      for st in P.STATIONS}
    card = P.plan(Q, "spark-open-baseten", fake, executors=librivox_tools)
    assert card.route == {"kind": "evidence", "stimulus": "le_petit_prince", "panel": "lpp_section8", "results": "results/section8.json",
                          "line": P.live_menu()["evidence_panels"]["le_petit_prince"]["line"]}
    assert [s["station"] for s in card.results] == ["evidence"]
    assert "librivox" not in json.dumps(card.results).lower() and "LibriVox" not in card.text
    assert len(fake.reqs) == 1 and not fake.reqs[0].tools


def test_outputs_for_another_clip_condition_or_seed_are_refused():
    rt = P.route(spec())
    other = {"build": lambda r: {"clip_id": "alice_ch01", "condition": r["condition"], "seed": r["seed"]},
             "check": lambda r: {"clip_id": r["clip_id"], "condition": "reversed", "seed": r["seed"]},
             "forecast": lambda r: {"clip_id": r["clip_id"], "condition": r["condition"], "seed": 1}}
    assert [s["output"]["status"] for s in P.run_stations(rt, other)] == ["refused"] * 3
    own = {st: (lambda r: {"clip_id": r["clip_id"], "condition": r["condition"], "seed": r["seed"], "wording": "ok"}) for st in P.STATIONS}
    assert all(s["output"].get("wording") == "ok" for s in P.run_stations(rt, own))


def test_defaults_are_stated_not_silent():
    fake = Fake({"content": spec(clip_id=None, condition=None, seed=None)})
    card = P.plan(Q, "spark-1.3", fake)
    assert card.route["clip_id"] == P.live_menu()["clips"][0]["id"] and card.route["condition"] == "sentence_scramble"
    assert card.route["seed"] == P.live_menu()["default_seed"] and card.route["defaults"] == ["clip", "condition", "seed"]
    assert "no clip named" in card.text and "default seed" in card.text


def test_a_decline_nulls_the_new_fields_and_uses_plain_words():
    fake = Fake({"content": spec(decline_reason="clinical_out_of_scope")})
    card = P.plan(Q, "spark-1.3", fake)
    assert card.declined and card.spec == P.decline_spec("clinical_out_of_scope") and card.results == []
    assert card.text.startswith("Declined: questions about patients") and "Nearest tested option: people: healthy adults" in card.text


def test_two_rejections_give_the_rule_made_decline():
    bad = {"content": {"population": "healthy_adults"}}
    fake = Fake(bad, bad)
    card = P.plan(Q, "spark-1.3", fake)
    assert card.declined and "fixed rules" in card.text and len(fake.reqs) == 2 and fake.reqs[1].system[-1].text.startswith("Your previous")


def test_menu_line_and_examples_file_use_allowed_words():
    text = P.menu_text()
    assert not BANNED.search(text) and "_" not in text and "Le Petit Prince" in text
    for c in P.live_menu()["clips"]:
        assert c["short"] in text
    ex = ROOT / "results/planner_v21_examples.json"
    if not ex.exists():
        pytest.skip("examples not run yet")
    d = json.loads(ex.read_text())
    assert d["label"] == "demo, not scored" and d["badge"] == P.BADGE and len(d["examples"]) == 3
    for e in d["examples"]:
        assert not BANNED.search(e["text"])
        for r in e["runs"]:
            assert r["decision"] == "accept" and not BANNED.search(r["card"]), r
