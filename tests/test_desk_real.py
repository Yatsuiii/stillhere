"""The analysis desk on the REAL Gate 1 results (results/gate1_real.json, built by scripts/build_desk_results.py from the frozen run's
tier-D files) with a scripted fake Spark client (no network): the sedation stage line is never licensed (P1 and P3 failed), the frozen
P1/P3 fallbacks are offered, D24 lag marks and the D24/D25 timing note are on every card, P2a and P7 name what they measure, requests
outside the catalogue decline, and every request that would leave the machine is a typed egress.Request with tier-D values only."""
import json
from pathlib import Path

import pytest

from stillhere import desk, egress, verifier
from stillhere.frozen import ANALYSES, Frozen

REAL = Path(__file__).resolve().parents[1] / "results/gate1_real.json"
Q_FADE, Q_LAG = desk.public("q_language_fades"), desk.public("q_lag_plus_2")


class Fake:
    def __init__(self, *replies):
        self.replies, self.reqs = list(replies), []

    def __call__(self, req):
        egress.check(req)                                    # anything but a typed Request is refused before "sending"
        self.reqs.append(req)
        return self.replies.pop(0)


def run(*aids):
    return {"tool_calls": [{"name": "run_registered", "arguments": {"analysis_id": a}} for a in aids]}


def answer(lids, reason="none", near="sedation", note="ok"):
    b = [{"line_id": lid, "slot": s, "source": k} for lid in lids for s, k in desk.bindings(lid).items()]
    return {"content": {"decline_reason": reason, "nearest_analysis": near, "line_ids": lids, "bindings": b, "note": note}}


@pytest.fixture(scope="module")
def fz():
    return Frozen.load(REAL)


def test_real_document_is_counts_only_with_provenance(fz):
    d = fz.doc
    assert d["source"] == "real" and not fz.synthetic and d["timing_notes"] == ["D24", "D25"]
    assert "sub-" not in json.dumps(d["included"]) and d["included"]["deep"] == d["deep_effective_n"] == 12
    assert all(len(h) == 64 for h in d["sources"].values()) and len(d["sources"]) == 4


def test_language_fades_line_is_never_licensed_on_the_real_result(fz):
    lic = desk.licensed_lines(fz)
    assert not any(lid.startswith("sedation_stage") for lid in lic)
    assert {"sedation_no_line", "P1_fallback", "P3_fallback", "P7_pass", "P8_pass", "P2a_pass"} <= set(lic)
    assert "P1_pass" not in lic and "P3_pass" not in lic
    schema = desk.answer_schema(lic, ["sedation.k"])
    assert not any(e.startswith("sedation_stage") for e in schema["properties"]["line_ids"]["items"]["enum"])
    assert desk.template_lines(fz) == ["sedation_no_line", "P1_fallback", "P3_fallback"]


def test_spark_choosing_the_stage_line_is_rejected_then_fallbacks_ship(fz):
    c = Fake(run("sedation", "P1", "P3"), answer(["sedation_stage"]), answer(["sedation_no_line", "P1_fallback", "P3_fallback"]))
    card = desk.Desk(fz, c).ask(Q_FADE)
    assert c.reqs[2].user[2].enums["rejected.0"] == "unlicensed_line" and card.source == "spark"
    assert "language fade under deep sedation" not in card.text and "hearing held" not in card.text
    assert '"TRIBE-only LANGUAGE did not reach the bar"' in card.text and "One combined gauge, no split" in card.text
    assert "in 9 of 12 volunteers (needed 10)" in card.text and "145 TRs = 4.8 minutes per state" in card.text
    assert card.text.count(desk.LAG_MARK.strip()) == 1                    # P1 is lag-sensitive (D24), P3 is not
    assert card.text.endswith(desk.TIMING_NOTE) and verifier.verify(card.verifier_view()) == []
    assert card.badge == "REPLAY"


def test_hearing_and_p7_name_what_they_measure(fz):
    p2a, p7 = desk.render(fz, "P2a_pass"), desk.render(fz, "P7_pass")
    assert p2a.startswith("P2a: bar met under Holm; TRIBE HEARING in Deep above the rest null") and desk.LAG_MARK in p2a
    assert "brain-to-brain ISC, not TRIBE" in p7 and desk.LAG_MARK not in p7


def test_requests_outside_the_catalogue_decline(fz):
    route = {"tool_calls": [{"name": "decline_unregistered", "arguments": {"nearest_analysis": "P1"}}]}
    card = desk.Desk(fz, Fake(route)).ask(Q_LAG)
    body = card.text.split("\n", 1)[1]
    assert card.source == "decline" and "Nearest registered analysis: P1" in body and "L - 1 s and L + 1 s" in body


def test_lag_decline_points_to_P1_whatever_spark_names(fz):
    route = {"tool_calls": [{"name": "decline_unregistered", "arguments": {"nearest_analysis": "P7"}}]}
    card = desk.Desk(fz, Fake(route)).ask(Q_LAG); body = card.text
    assert "Nearest registered analysis: P1" in body and "P7" not in body.split("Nearest registered analysis:")[1].split(".")[0]
    assert desk.timing_question(Q_LAG) and not desk.timing_question(Q_FADE) and verifier.verify(card.verifier_view()) == []
    assert "With zero control scans" not in body and verifier.verify(card.verifier_view()) == []


def test_every_outgoing_payload_is_typed_tier_d(fz):
    for aid in ANALYSES:
        egress.TierD(values=fz.lookup(aid))                               # identifier keys; numbers, booleans and None only
    c = Fake(run("sedation", "P1", "P3"), answer(["sedation_no_line", "P1_fallback", "P3_fallback"]))
    desk.Desk(fz, c).ask(Q_FADE)
    for req in c.reqs:
        assert isinstance(req, egress.Request) and not req.model.endswith("-contributor")
        assert all(isinstance(p, (egress.TierD, egress.PublicText, egress.PublicQuestion)) for p in req.user)
        assert all(isinstance(p, egress.PublicText) for p in req.system)
    with pytest.raises(egress.EgressRefused):
        egress.check({"model": desk.MODEL, "user": "Can we say language fades?"})


def test_recording_client_makes_a_replay_file(fz, tmp_path):
    rec = tmp_path / "r.jsonl"
    live = desk.RecordingClient(Fake(run("sedation", "P1", "P3"), answer(["sedation_no_line", "P1_fallback", "P3_fallback"])), rec)
    first = desk.Desk(fz, live, mode="LIVE").ask(Q_FADE)
    again = desk.Desk(fz, replay=rec).ask(Q_FADE)
    assert first.text == again.text and again.source == "spark" and again.badge == "REPLAY"


def test_desk_adds_the_failed_co_primary_claims_spark_leaves_out(fz):
    c = Fake(run("sedation", "P1", "P2a"), answer(["P2a_pass", "P1_fallback", "sedation_no_line"]))
    card = desk.Desk(fz, c).ask(Q_FADE)
    lines = card.text.split("\n")[1:-1]                                   # between the header and the timing note
    assert lines[0].startswith("Gate 1: no sedation stage line") and "One combined gauge, no split" in card.text
    assert [l.split(" ")[0] for l in lines] == ["Gate", "P1", "P3", "P2a:"] and "desk_added:P3_fallback" in card.trace
