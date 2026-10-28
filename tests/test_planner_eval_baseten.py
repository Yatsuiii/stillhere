"""The Baseten addition's pass-bar arithmetic, with made-up runs (no network): decision labels and the two pairwise agreements."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rpeb", ROOT / "scripts/run_planner_eval_baseten.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def acc(q, rep, areas=("auditory",)):
    return {"qid": q, "repeat": rep, "decision": "accept", "spec": {"population": "healthy_adults", "language": "english",
            "stimulus": "librivox_audiobook", "brain_areas": list(areas), "conditions": ["intact", "reversed"]}}


def dec(q, rep, reason):
    return {"qid": q, "repeat": rep, "decision": "decline", "reason": reason}


def test_labels_ignore_list_order_and_keep_decline_type():
    assert M.label(acc("a", 1, ("language", "auditory"))) == M.label(acc("a", 2, ("auditory", "language")))
    assert M.label(dec("b", 1, "population_not_validated")) != M.label(dec("b", 1, "language_not_validated"))
    assert M.label({"qid": "c", "decision": "unavailable"}) == ("unavailable",)


def test_agreement_and_bar():
    fw = [acc("a", r) for r in (1, 2, 3)] + [dec("b", 1, "x"), dec("b", 2, "x"), dec("b", 3, "y")]
    bt = [acc("a", r) for r in (1, 2, 3)] + [dec("b", r, "x") for r in (1, 2, 3)]
    g = M.agreement(fw, bt)
    assert g["fireworks_repeat_agreement"] == round(4 / 6, 4) and g["baseten_vs_fireworks_agreement"] == round(15 / 18, 4) and g["pass"]
    bt = [dec("a", r, "z") for r in (1, 2, 3)] + [dec("b", r, "x") for r in (1, 2, 3)]
    assert not M.agreement(fw, bt)["pass"]
