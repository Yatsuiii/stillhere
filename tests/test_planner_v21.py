"""Planner v2.1 with a stub client (no network): v2 left as published; plain words for every code and id; the v2.1 schema (clip_id,
condition, seed); one plan call and no tool call; routing by stimulus, including the LibriVox-under-Le-Petit-Prince bug v2's tool loop
had; outputs for another clip refused; stated defaults; the live-menu line and the examples file in plain, allowed words; avoid_tags
(code filters the menu by the story cards, one re-prompt, then the rule-made decline with the nearest clip from code)."""
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
         "clip_id": "alice_ch02", "condition": "sentence_scramble", "seed": 7, "avoid_tags": []}
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
    for code in planner.DECLINE_REASONS[1:] + P.AVOID_REASONS:
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
    assert d["label"] == "demo, not scored" and d["badge"] in (P.BADGE, P.BADGE_BEFORE_TAGS) and len(d["examples"]) == 3
    for e in d["examples"]:
        assert not BANNED.search(e["text"])
        for r in e["runs"]:
            assert r["decision"] == "accept" and not BANNED.search(r["card"]), r


# ---- avoid_tags (cloud plan of Oct 29, 19:20 UTC, 3b) ------------------------------------------------------------------------
NOT_FREE = re.compile(r"\bsafe\b|free of|contains no|has no (?!screened)", re.I)      # the page never says a story is free of a note


def test_menu_line_gives_the_real_clip_lengths():
    secs = sorted(round(c["dur_s"]) for c in P.live_menu()["clips"])
    assert f"{secs[0]}-{secs[-1]} s each" in P.menu_text() or f"{secs[0]} s each" in P.menu_text()
    assert not re.search(r"; 40 s each\)", P.menu_text()) or secs[0] == 40              # no longer "40 s each" for every clip


def test_clip_tags_are_the_story_card_tags_and_unknown_without_screen_data():
    tags = P.clip_tags()
    for c in P.live_menu()["clips"]:
        sc = c.get("screen")
        if isinstance(sc, dict):
            assert tags[c["id"]] == sorted({P.TAG_IDS[n["tag"]] for n in sc["notes"]})
        else:
            assert tags[c["id"]] is None and not P.passes(c["id"], ["violence"], tags) and P.passes(c["id"], [], tags)


def test_schema_and_validator_for_avoid_tags():
    enums = P.load_enums()
    s = P.study_schema(enums)
    assert s["properties"]["avoid_tags"] == {"type": "array", "items": {"type": "string", "enum": list(P.TAG_IDS.values())}}
    assert "avoid_tags" in s["required"]
    assert P.validate_answer(spec(avoid_tags=["not a tag"]), enums) == ["schema_invalid"]
    assert P.validate_answer(spec(avoid_tags="violence"), enums) == ["schema_invalid"]
    tags = enums["clip_tags"]
    carrying = next(c for c in enums["clip_id"] if tags[c] and "violence" in tags[c])
    unscreened = next(c for c in enums["clip_id"] if tags[c] is None)
    assert P.validate_answer(spec(clip_id=carrying, avoid_tags=["violence"]), enums) == ["clip_has_avoided_tag"]
    assert P.validate_answer(spec(clip_id=unscreened, avoid_tags=["violence"]), enums) == ["clip_has_avoided_tag"]
    assert P.validate_answer(spec(clip_id=carrying, avoid_tags=[]), enums) == []          # nothing avoided: every clip passes
    assert P.validate_answer(lpp(avoid_tags=["violence"]), enums) == []                   # no live clip: the filter does not apply


def test_the_model_sees_tag_ids_only_through_table_enums():
    fake = Fake({"content": spec(clip_id=None, avoid_tags=["violence"])})
    P.plan(Q, "spark-open", fake)
    sent = {k: v for u in fake.reqs[0].user if isinstance(u, egress.TableEnums) for k, v in u.enums.items()}
    assert sent["avoid_tag"] == list(P.TAG_IDS.values())
    screened = {c for c, t in P.clip_tags().items() if t is not None}
    assert {k.split(".", 1)[1] for k in sent if k.startswith("clip_tags.")} == screened
    assert all(" " not in v for k, vals in sent.items() for v in vals)                     # identifiers, no free text


def test_default_clip_is_the_first_that_passes_and_the_card_never_says_free_of():
    enums = P.load_enums()
    want = P.passing_clips(enums, ["violence"])
    assert want and want[0] != enums["clip_id"][0] or "violence" not in (enums["clip_tags"][enums["clip_id"][0]] or ["violence"])
    card = P.plan(Q, "spark-1.3", Fake({"content": spec(clip_id=None, avoid_tags=["violence"])}))
    assert card.route["kind"] == "live" and card.route["clip_id"] == want[0] and card.route["avoid_tags"] == ["violence"]
    assert "violence" not in (enums["clip_tags"][want[0]] or [])
    assert "Notes to avoid: violence." in card.text and "not a full review" in card.text and not NOT_FREE.search(card.text)


def test_a_clip_with_an_avoided_tag_gets_one_reprompt_then_the_rule_made_decline():
    enums = P.load_enums(); tags = enums["clip_tags"]
    carrying = next(c for c in enums["clip_id"] if tags[c] and "violence" in tags[c])
    bad = {"content": spec(clip_id=carrying, avoid_tags=["violence"])}
    fake = Fake(bad, bad)
    card = P.plan(Q, "spark-1.3", fake)
    assert len(fake.reqs) == 2 and card.declined and card.trace[:2] == ["plan:rejected:clip_has_avoided_tag"] * 2
    sent = {k: v for u in fake.reqs[1].user if isinstance(u, egress.TableEnums) for k, v in u.enums.items()}
    assert sent["rejected"] == ["clip_has_avoided_tag"]
    near = P.passing_clips(enums, ["violence"])[0]
    assert card.text.startswith("Declined: the story chosen carries a note you asked to avoid. Nearest tested option: story: "
                                + P.plain_id(near)) and "fixed rules" in card.text and not NOT_FREE.search(card.text)
    ok = Fake(bad, {"content": spec(clip_id=near, avoid_tags=["violence"])})                # the re-prompt fixes it
    card = P.plan(Q, "spark-1.3", ok)
    assert card.route["kind"] == "live" and card.route["clip_id"] == near and not card.declined


def test_no_passing_clip_is_a_code_made_decline_with_no_option():
    enums = P.load_enums()
    every = sorted(set().union(*[set(t) for t in enums["clip_tags"].values() if t is not None]))
    if any(t == [] for t in enums["clip_tags"].values()):
        pytest.skip("a screened clip has no notes, so some clip always passes")
    card = P.plan(Q, "spark-1.3", Fake({"content": spec(clip_id=None, avoid_tags=every)}))
    assert card.declined and card.spec["decline_reason"] == "no_story_without_avoided_notes" and card.results == []
    assert "Nearest tested option: story: none offered" in card.text and not NOT_FREE.search(card.text)


def test_badge_names_story_tags_and_keeps_the_measured_number_as_v2s():
    assert P.BADGE == "v2.1 adds clip, condition, seed and story tags; not re-measured"
    assert "%" not in P.BADGE


def test_recorded_avoid_tags_run():
    f = ROOT / "results/planner_v21_avoid_tags_run.json"
    if not f.exists():
        pytest.skip("avoid_tags run not recorded yet")
    d = json.loads(f.read_text()); r = d["run"]
    assert d["label"] == "demo, not scored" and d["badge"] == P.BADGE and r["backend"] == "spark-1.3"
    assert r["spec"]["avoid_tags"] and r["route"]["kind"] == "live" and len(r["calls"]) == 1
    assert P.passes(r["route"]["clip_id"], r["route"]["avoid_tags"], P.clip_tags())        # still true on today's menu
    assert not BANNED.search(r["card"]) and not NOT_FREE.search(r["card"]) and "not a full review" in r["card"]
