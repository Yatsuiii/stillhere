"""The Spark Open face-edit check's scoring, with stubs (no frames, no network): Glimmer's parser, one retry on an unparsed answer, and the
three rule rates (unparsed or error counts as yes for faces removed and as not yes elsewhere)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sofc", ROOT / "scripts/run_spark_open_face_check.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def test_glimmers_parser():
    assert [S.parse(t) for t in ("yes", "yes.", "no", "no, none", "maybe", "", "i think yes")] == ["yes", "yes", "no", "no", "unparsed", "unparsed", "yes"]


def test_one_retry_on_unparsed(monkeypatch):
    calls = []
    def ask(frame, prompt):
        calls.append(prompt); return ("unparsed" if prompt == "frame_check_question" else "no"), 10, 5
    monkeypatch.setattr(S, "ask", ask)
    row = S.one(Path("/x"), "faces_removed", {"t_release": 25.0, "name": "t025.0.jpg", "sam_faces": 2})
    assert calls == ["frame_check_question", "frame_check_retry"] and row["answer"] == "no" and row["first_answer"] == "unparsed"


def test_rates_and_bars():
    def rows(v, answers):
        return [{"version": v, "answer": a, "sam_faces": 2} for a in answers]
    rs = rows("intact", ["yes"] * 25 + ["no"] * 6) + rows("faces_removed", ["no"] * 28 + ["unparsed", "error", "yes"]) + rows("area_matched", ["yes"] * 31)
    r = S.rates(rs)
    assert round(r["intact_with_face_yes"], 3) == round(25 / 31, 3) and r["pass"]["intact"]
    assert round(r["faces_removed_no"], 3) == round(28 / 31, 3) and r["pass"]["faces_removed"]
    assert r["pass"]["area_matched"] and r["pass"]["all"]
    rs = rows("intact", ["yes"] * 31) + rows("faces_removed", ["no"] * 27 + ["unparsed"] * 4) + rows("area_matched", ["yes"] * 31)
    r = S.rates(rs)
    assert not r["pass"]["faces_removed"] and not r["pass"]["all"]                # 27/31 = 0.871 < 0.90: unparsed counts as yes
