"""Unit tests for the analysis desk with a scripted fake Spark client and the SYNTHETIC Gate 1 fixtures (no network, no real data):
licensing and the strict schema, forced tool use, number bindings, declines, the template fallback, the two injected tests of
SPARK-PLAN A (fabricated number, "detects awareness"), REPLAY by default, and the tool-call path of the Meta client."""
import copy
import json
from pathlib import Path

import httpx
import pytest

from stillhere import desk, egress, meta_client, verifier
from stillhere.frozen import Frozen

FIX = Path(__file__).with_name("fixtures")
PASS = json.loads((FIX / "synthetic_gate1_pass.json").read_text())
FAIL = json.loads((FIX / "synthetic_gate1_p1_p3_fail.json").read_text())
Q_FADE, Q_LAG = desk.public("q_language_fades"), desk.public("q_lag_plus_2")


class Fake:
    """Returns the scripted replies in order and keeps every request it was sent."""
    def __init__(self, *replies):
        self.replies, self.reqs = list(replies), []

    def __call__(self, req):
        egress.check(req)
        self.reqs.append(req)
        return self.replies.pop(0)


def run(*aids):
    return {"tool_calls": [{"name": "run_registered", "arguments": {"analysis_id": a}} for a in aids]}


def answer(lids, reason="none", near="sedation", note="ok", binds=None):
    b = binds if binds is not None else [{"line_id": lid, "slot": s, "source": k} for lid in lids for s, k in desk.bindings(lid).items()]
    return {"content": {"decline_reason": reason, "nearest_analysis": near, "line_ids": lids, "bindings": b, "note": note}}


def fz(doc=PASS, **bars):
    d = copy.deepcopy(doc)
    for k, v in bars.items():
        d["gate1"]["bars"][k]["pass"] = v
    return Frozen(d)


# ---- frozen results and licensing ---------------------------------------------------------------------------------------
def test_fixtures_are_synthetic_and_counts_only():
    f = Frozen(PASS)
    assert f.synthetic and f.included_awake == 16 and f.doc["isc_loo_reference"] == 15
    assert "sub-" not in json.dumps(PASS["included"])
    with pytest.raises(ValueError):
        Frozen({**PASS, "format": "other"})


def test_lookup_is_tier_d():
    vals = Frozen(PASS).lookup("P4")
    egress.TierD(values=vals)                                  # identifier keys, numbers/bools/None only
    assert vals["P4.L.auc_ci_lo"] <= vals["P4.L.auc"] <= vals["P4.L.auc_ci_hi"]


def test_unlicensed_line_cannot_be_decoded():
    lic_fail = desk.licensed_lines(Frozen(FAIL))
    assert "sedation_no_line" in lic_fail and "P1_fallback" in lic_fail and "P3_fallback" in lic_fail
    assert not any(lid.startswith("sedation_stage") for lid in lic_fail) and "P1_pass" not in lic_fail
    schema = desk.answer_schema(lic_fail, ["P1.k"])
    assert "sedation_stage" not in schema["properties"]["line_ids"]["items"]["enum"]
    lic_pass = desk.licensed_lines(Frozen(PASS))
    assert lic_pass[0] == "sedation_stage_waking" and "P1_fallback" not in lic_pass


def test_schema_is_strict_at_every_level():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for p in s["properties"].values():
                walk(p)
        if s.get("type") == "array":
            walk(s["items"])
    walk(desk.answer_schema(desk.licensed_lines(Frozen(PASS)), ["sedation.k", "sedation.n"]))


def test_p7_failure_reads_nothing_else():
    f = fz(P7=False)
    assert desk.licensed_lines(f) == ["P7_fallback"] and desk.template_lines(f) == ["P7_fallback"]
    c = Fake(run("sedation", "P1"), answer(["P7_fallback"]))
    card = desk.Desk(f, c).ask(Q_FADE)
    assert card.source == "spark" and "Fix preprocessing" in card.text
    assert set(c.reqs[1].json_schema["properties"]["line_ids"]["items"]["enum"]) == {"P7_fallback"}
    assert card.controls_scanned == 15                         # P7 is LOO-ISC: its reference size is stated (D18)


def test_p8_failure_renames_the_gauge():
    t = desk.render(fz(P8=False), "P1_pass")
    assert "SPEECH-LOCKED" in t and "LANGUAGE" not in t


# ---- the desk -----------------------------------------------------------------------------------------------------------
def test_happy_path_numbers_come_from_bindings():
    f, c = Frozen(PASS), Fake(run("sedation", "P1"), answer(["sedation_stage_waking"]))
    card = desk.Desk(f, c).ask(Q_FADE)
    p1 = PASS["gate1"]["bars"]["P1"]
    assert card.source == "spark" and f"in {p1['k']} of {p1['n']} volunteers" in card.text and "(145 TRs = 4.8 minutes per state)" in card.text
    assert card.text.startswith(verifier.HEADER.format(n=0)) and verifier.verify(card.verifier_view()) == []
    assert card.badge == "REPLAY · SYNTHETIC DATA"
    route, ans = c.reqs
    assert route.tools and route.json_schema is None and ans.json_schema and not ans.tools
    assert ans.user[1].values["sedation.k"] == p1["k"] and ans.user[1].enums["line.sedation_stage_waking.k"] == "sedation.k"


def test_no_tool_call_is_reprompted_once_then_template():
    c = Fake({"tool_calls": []}, {"tool_calls": []})
    card = desk.Desk(Frozen(FAIL), c).ask(Q_FADE)
    assert card.source == "template" and "TEMPLATE" in card.badge and len(c.reqs) == 2
    assert desk.public("desk_reprompt") in c.reqs[1].system
    assert desk.template_lines(Frozen(FAIL)) == ["sedation_no_line", "P1_fallback", "P3_fallback"]


def test_wrong_binding_injection_is_caught():
    wrong = [{"line_id": "sedation_stage_waking", "slot": s, "source": "sedation.n" if s == "k" else k}
             for s, k in desk.bindings("sedation_stage_waking").items()]
    c = Fake(run("sedation"), answer(["sedation_stage_waking"], binds=wrong), answer(["sedation_stage_waking"], binds=wrong))
    card = desk.Desk(Frozen(PASS), c).ask(Q_FADE)
    assert card.source == "template" and len(c.reqs) == 3
    assert c.reqs[2].user[2].enums["rejected.0"] == "wrong_binding:sedation_stage_waking.k"


def test_fabricated_number_in_note_is_caught_and_never_shown():
    c = Fake(run("sedation"), answer(["sedation_stage_waking"], note="language fell in 15 of 16"), answer(["sedation_stage_waking"]))
    card = desk.Desk(Frozen(PASS), c).ask(Q_FADE)
    assert card.source == "spark" and "language fell" not in card.text
    assert c.reqs[2].user[2].enums == {"rejected.0": "digit_in_note"}


def test_fabricated_source_is_caught():
    binds = [{"line_id": "sedation_stage_waking", "slot": s, "source": "sedation.k_typed" if s == "k" else k}
             for s, k in desk.bindings("sedation_stage_waking").items()]
    c = Fake(run("sedation"), answer(["sedation_stage_waking"], binds=binds), answer(["sedation_stage_waking"]))
    desk.Desk(Frozen(PASS), c).ask(Q_FADE)
    assert "unknown_source" in c.reqs[2].user[2].enums.values()


def test_awareness_claim_is_caught_by_the_claim_gate():
    c = Fake(run("sedation"), answer(["sedation_stage_waking"], note="StillHere detects awareness"),
             answer(["sedation_stage_waking"], note="detects awareness again"))
    card = desk.Desk(Frozen(PASS), c).ask(Q_FADE)
    assert card.source == "template" and "awareness" not in card.text
    assert c.reqs[2].user[2].enums == {"rejected.0": "blocked_word_in_note"}
    bad = verifier.verify({"text": verifier.HEADER.format(n=0) + "\nStillHere detects awareness.", "subject": {"kind": "group", "n": 16},
                           "controls_scanned": 0})
    assert bad == ["blocked word: 'awareness'"]


def test_unlicensed_line_from_spark_is_rejected():
    c = Fake(run("sedation", "P1"), answer(["sedation_stage"]), answer(["sedation_no_line"]))
    card = desk.Desk(Frozen(FAIL), c).ask(Q_FADE)
    assert c.reqs[2].user[2].enums["rejected.0"] == "unlicensed_line" and card.source == "spark"


def test_lag_request_declines_with_the_licensed_line_or_readme():
    route = {"tool_calls": [{"name": "decline_unregistered", "arguments": {"nearest_analysis": "P1"}}]}
    card = desk.Desk(Frozen(PASS), Fake(route)).ask(Q_LAG)
    assert card.source == "decline" and "Nearest registered analysis: P1 (PREREG §6 P1)." in card.text and "With zero control scans" in card.text
    card = desk.Desk(Frozen(FAIL), Fake(route)).ask(Q_LAG)
    body = card.text.split("\n", 1)[1]                       # after the fixed header
    assert "L - 1 s and L + 1 s" in body and "With zero control scans" not in body and " of 16" not in body      # timing question: P1's D24 marks
    assert verifier.verify(card.verifier_view()) == []


def test_decline_from_the_answer_step_and_clinical():
    card = desk.Desk(Frozen(PASS), Fake(run("P4"), answer([], reason="not_preregistered", near="P4"))).ask(Q_FADE)
    assert "Nearest registered analysis: P4" in card.text and "reported in the README" in card.text
    card = desk.Desk(Frozen(PASS), Fake(run("P4"), answer([], reason="not_preregistered", near="P4"))).ask(Q_LAG)
    assert "Nearest registered analysis: P1" in card.text                     # a timing question always points to P1's D24 marks
    card = desk.Desk(Frozen(PASS), Fake(run("P1"), answer([], reason="clinical_out_of_scope"))).ask(Q_FADE)
    assert verifier.DECLINE_CLINICAL in card.text and verifier.verify(card.verifier_view()) == []


def test_replay_is_the_default_and_falls_back_when_not_recorded(tmp_path):
    d = desk.Desk(Frozen(PASS), replay=tmp_path / "none.jsonl")
    assert d.mode == "REPLAY"
    card = d.ask(Q_FADE)
    assert card.source == "template" and card.trace == ["client_error:NotRecorded"]
    req = egress.Request(model=desk.MODEL, system=[desk.public("desk_route"), desk.public("desk_catalogue")], user=[Q_FADE], tools=desk.TOOLS)
    (tmp_path / "r.jsonl").write_text(json.dumps({"digest": desk.digest(req), "response": run("sedation")}) + "\n")
    assert desk.ReplayClient(tmp_path / "r.jsonl")(req) == run("sedation")


# ---- egress and client --------------------------------------------------------------------------------------------------
def test_request_offers_tools_or_schema_exactly():
    base = dict(model=desk.MODEL, system=[desk.public("desk_route")], user=[Q_FADE])
    with pytest.raises(ValueError):
        egress.Request(**base)
    with pytest.raises(ValueError):
        egress.Request(**base, tools=desk.TOOLS, schema_name="x", json_schema={"type": "object"})
    with pytest.raises(ValueError):
        egress.Tool(name="run registered", description=desk.public("tool_run_registered"), params={"a": ["x"]})
    with pytest.raises(ValueError):
        egress.Tool(name="t", description=desk.public("tool_run_registered"), params={"a": ["a sentence"]})


def test_meta_client_sends_tools_and_parses_calls(monkeypatch, tmp_path):
    sent = {}

    def post(url, json, timeout, headers):
        sent.update(json)
        calls = [{"id": "c1", "type": "function", "function": {"name": "run_registered", "arguments": '{"analysis_id": "P1"}'}}]
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": calls}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(meta_client, "USAGE_LOG", tmp_path / "u.jsonl")
    monkeypatch.setenv("MODEL_API_KEY", "test")
    req = egress.Request(model=desk.MODEL, system=[desk.public("desk_route")], user=[Q_FADE], tools=desk.TOOLS)
    assert meta_client.call(req) == {"tool_calls": [{"name": "run_registered", "arguments": {"analysis_id": "P1"}}]}
    assert "response_format" not in sent and sent["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert json.loads((tmp_path / "u.jsonl").read_text())["tool_calls"] == 1
