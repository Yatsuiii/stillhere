"""Planner v0 with a stub client (no network, no live model): enums rebuilt from the table on every request, menu
declines under both backends, exactly-one reprompt then a deterministic decline, ordered tool calls with
parallel_tool_calls False, and no TierD in any planner request."""
import copy
import json

import httpx
import pytest

from stillhere import egress, planner

Q_6YO = planner.public("q_plan_6yo")
Q_LAG = planner.public("q_lag_plus_2")


class Fake:
    """Returns the scripted replies in order and keeps every request it was sent."""

    def __init__(self, *replies):
        self.replies, self.reqs = list(replies), []

    def __call__(self, req):
        egress.check(req)
        self.reqs.append(req)
        return self.replies.pop(0)


def accept(**kw):
    spec = {"population": "healthy_adults", "language": "english", "stimulus": "librivox_audiobook", "analysis": "forecast_tribe_stock",
            "brain_areas": ["auditory"], "conditions": ["intact", "reversed"], "decline_reason": "none"}
    spec.update(kw)
    return {"content": spec}


def decline(reason):
    return {"content": {"population": None, "language": None, "stimulus": None, "analysis": None,
                        "brain_areas": [], "conditions": [], "decline_reason": reason}}


def table_copy(tmp_path):
    t = json.loads(planner.TABLE.read_text())
    p = tmp_path / "validation_table.json"
    p.write_text(json.dumps(t))
    return p


def use_table(monkeypatch, path):
    monkeypatch.setattr(planner, "TABLE", path)
    monkeypatch.setattr(egress, "VALIDATION_TABLE", path)


# ---- enums rebuilt from the table on every request -----------------------------------------------
def test_enums_rebuilt_from_modified_table_each_request(tmp_path, monkeypatch):
    p = table_copy(tmp_path)
    use_table(monkeypatch, p)
    t = json.loads(p.read_text())
    t["stimulus"].append({"id": "new_stim_x", "evidence": "test", "status": "validated"})
    p.write_text(json.dumps(t))
    c = Fake(decline("stimulus_not_validated"))
    planner.plan(Q_LAG, "spark-1.3", c)
    schema = c.reqs[0].json_schema
    assert "new_stim_x" in schema["properties"]["stimulus"]["enum"]           # no code change needed
    assert "word_scramble" not in schema["properties"]["conditions"]["items"]["enum"]  # failed_check never appears
    assert "blender_film" not in schema["properties"]["stimulus"]["enum"]     # pending never appears
    assert "lag_search" not in str(schema)                                   # not_offered never appears
    t["stimulus"].append({"id": "new_stim_y", "evidence": "test", "status": "validated"})
    p.write_text(json.dumps(t))
    c = Fake(decline("stimulus_not_validated"))
    planner.plan(Q_LAG, "spark-1.3", c)                                      # rebuilt on the next request too
    assert "new_stim_y" in c.reqs[0].json_schema["properties"]["stimulus"]["enum"]
    assert "french" in c.reqs[0].json_schema["properties"]["language"]["enum"]  # group_level_only stays offered


def test_schema_is_one_flat_strict_object():
    schema = planner.study_schema(planner.load_enums())
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for k in ("population", "language", "stimulus", "analysis"):
        assert schema["properties"][k]["type"] == ["string", "null"]
        assert None in schema["properties"][k]["enum"]
    assert set(schema["properties"]["decline_reason"]["enum"]) == set(planner.DECLINE_REASONS)


# ---- menu declines, both backends -----------------------------------------------------------------
@pytest.mark.parametrize("backend", ["spark-1.3", "spark-open"])
def test_menu_questions_decline_with_reason_and_nearest(backend):
    c = Fake(decline("population_not_validated"))
    card = planner.plan(Q_6YO, backend, c)
    assert card.declined and card.spec["decline_reason"] == "population_not_validated"
    assert "population_not_validated" in card.text and "healthy_adults" in card.text
    assert c.reqs[0].model == planner.MODELS[backend]
    c = Fake(decline("analysis_not_offered"))
    card = planner.plan(Q_LAG, backend, c)
    assert card.declined and card.spec["decline_reason"] == "analysis_not_offered"
    assert "analysis_not_offered" in card.text and "forecast_tribe_stock" in card.text
    assert "not from the model" not in card.text                        # the model's own decline carries no fallback note


@pytest.mark.parametrize("backend", ["spark-1.3", "spark-open"])
@pytest.mark.parametrize("question,reason", [(Q_6YO, "population_not_validated"), (Q_LAG, "analysis_not_offered")])
def test_validator_rejects_stub_that_tries_to_accept(question, reason, backend):
    """A stub that tries to accept the menu questions is rejected: an invented population id fails the schema, and a
    lone intact condition fails consistency. Both re-prompt exactly once, then decline."""
    bad = {"population": "children_6yo"} if question is Q_6YO else {"conditions": ["intact"]}
    c = Fake(accept(**bad), decline(reason))
    card = planner.plan(question, backend, c)
    assert len(c.reqs) == 2
    assert card.declined and card.spec["decline_reason"] == reason
    reprompt = c.reqs[1]
    assert planner.public("planner_reprompt") in reprompt.system
    rejected = [p for p in reprompt.user if isinstance(p, egress.TableEnums)
                and "rejected" in p.enums]
    assert len(rejected) == 1 and rejected[0].enums["rejected"] != []
    assert c.reqs[0].model == c.reqs[1].model == planner.MODELS[backend]


def test_second_failure_is_a_deterministic_decline():
    c = Fake({"content": {"nope": 1}}, {"content": ["intact"]})
    card = planner.plan(Q_6YO, "spark-1.3", c)
    assert len(c.reqs) == 2 and card.declined
    assert card.spec["decline_reason"] == "population_not_validated"   # from the question, not the stub
    assert card.spec["population"] is None and card.spec["conditions"] == []
    assert card.text.endswith("not from the model.")                   # a rule-made decline is never shown as model output
    c = Fake(accept(population="children_6yo"), accept(population="children_6yo"))
    card = planner.plan(Q_LAG, "spark-open", c)                       # stubborn stub: still exactly one reprompt
    assert len(c.reqs) == 2 and card.declined and card.spec["decline_reason"] == "analysis_not_offered"


def test_failed_call_is_unavailable_not_a_decline():
    def down(req):
        raise httpx.ConnectError("offline")
    card = planner.plan(Q_6YO, "spark-1.3", down)
    assert card.declined and card.spec == {} and card.trace == ["client_error:ConnectError"]
    assert card.text.startswith("Planner unavailable") and "Declined" not in card.text


    class FailSecond(Fake):                                            # first answer rejected, then the reprompt fails
        def __call__(self, req):
            if self.reqs:
                down(req)
            return super().__call__(req)
    card = planner.plan(Q_6YO, "spark-open", FailSecond({"content": {"nope": 1}}))
    assert card.trace == ["plan:rejected:schema_invalid", "client_error:ConnectError"] and "Declined" not in card.text


# ---- tools -----------------------------------------------------------------------------------------
def test_tool_calls_run_in_order_with_parallel_false():
    spec = accept()["content"]
    c = Fake({"tool_calls": [{"name": "forecast", "arguments": {"stimulus": "librivox_audiobook"}},
                             {"name": "list_stimuli", "arguments": {"stimulus": "librivox_audiobook"}},
                             {"name": "check_conditions", "arguments": {"condition": "intact"}},
                             {"name": "make_conditions", "arguments": {"condition": "reversed"}}]})
    out = planner.run_tools(spec, "spark-1.3", c)
    assert [r["tool"] for r in out] == ["forecast", "list_stimuli", "check_conditions", "make_conditions"]
    req = c.reqs[0]
    assert req.parallel_tool_calls is False
    assert sorted(t.name for t in req.tools) == ["check_conditions", "forecast", "list_stimuli", "make_conditions"]
    assert len(c.reqs) == 1                                          # results never go back to a model
    card = planner.build_card(spec, out, planner.load_enums())
    assert not card.declined and "exploratory" in card.text
    assert "forecast, not validated" in card.text
    assert out[1]["output"]["stimuli"] == planner.load_enums()["stimulus"]
    assert out[3]["output"]["conditions"]["reversed"]["samples"] > 0
    assert out[2]["output"]["label"] == "exploratory"


def test_decline_runs_no_tools_and_no_call():
    c = Fake()
    assert planner.run_tools(decline("stimulus_not_validated")["content"], "spark-1.3", c) == []
    assert c.reqs == []


# ---- egress boundaries -----------------------------------------------------------------------------
def test_no_planner_request_carries_tier_d_and_fireworks_refuses_it():
    spec = accept()["content"]
    c = Fake(decline("population_not_validated"))
    planner.plan(Q_6YO, "spark-open", c)
    c2 = Fake({"tool_calls": []})
    planner.run_tools(spec, "spark-open", c2)
    for req in c.reqs + c2.reqs:
        assert not any(isinstance(p, egress.TierD) for p in req.user)
        assert req.model == "fireworks/muse-spark-open"
    sys_ = planner.public("planner_system")
    with pytest.raises(ValueError):
        egress.Request(model="fireworks/muse-spark-open", system=[sys_], user=[egress.TierD(values={"k": 1})],
                       schema_name="study_spec", json_schema=planner.study_schema(planner.load_enums()))


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        planner.plan(Q_LAG, "spark-9.9", Fake())


# ---- v2: decline normalisation and the analysis field -------------------------------------------------
def test_v2_decline_with_filled_fields_is_normalised_not_rejected():
    ans = accept(decline_reason="population_not_validated")              # the model declines but leaves the valid fields filled
    c = Fake(ans)
    card = planner.plan(Q_6YO, "spark-1.3", c)
    assert len(c.reqs) == 1 and card.declined and card.spec == planner.decline_spec("population_not_validated")
    assert card.trace[0] == "plan:ok:normalised" and "not from the model" not in card.text
    assert planner.validate_answer(ans["content"], planner.load_enums()) == []


def test_v2_analysis_field_comes_from_the_table_and_an_accept_needs_it(tmp_path, monkeypatch):
    schema = planner.study_schema(planner.load_enums())
    assert schema["properties"]["analysis"]["enum"] == ["forecast_tribe_stock", None] and "analysis" in schema["required"]
    no_analysis = accept(analysis=None)
    assert planner.validate_answer(no_analysis["content"], planner.load_enums()) == ["decline_inconsistent"]
    c = Fake(no_analysis, decline("analysis_not_offered"))
    card = planner.plan(Q_LAG, "spark-open", c)
    assert len(c.reqs) == 2 and card.spec["decline_reason"] == "analysis_not_offered"
    p = table_copy(tmp_path); use_table(monkeypatch, p)
    t = json.loads(p.read_text()); t["analysis"].append({"id": "new_analysis_x", "evidence": "test", "status": "validated"}); p.write_text(json.dumps(t))
    c = Fake(decline("analysis_not_offered")); planner.plan(Q_LAG, "spark-1.3", c)
    assert "new_analysis_x" in c.reqs[0].json_schema["properties"]["analysis"]["enum"]


def test_v2_baseten_backend_is_a_planner_backend():
    c = Fake(decline("population_not_validated"))
    card = planner.plan(Q_6YO, "spark-open-baseten", c)
    assert c.reqs[0].model == "baseten/muse-spark-open-1026" and card.declined
    assert not any(isinstance(p, egress.TierD) for p in c.reqs[0].user)
