"""Edit-check bars with made-up answers (no frames, no network): unparsed or error counts against every bar, and the no-mask set allows
at most 3 answers that are not "no"."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rec", ROOT / "scripts/run_edit_check.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


def rows(intact, removed, control, nomask):
    out = []
    for v, ans in (("intact", intact), ("faces_removed", removed), ("area_matched", control)):
        out += [{"version": v, "set": "bridge" if i < 31 else "new", "answer": a} for i, a in enumerate(ans)]
    return out + [{"version": "intact", "set": "no_mask", "answer": a} for a in nomask]


def test_pass_and_each_way_to_fail():
    ok = rows(["yes"] * 62, ["no"] * 62, ["yes"] * 62, ["no"] * 28 + ["yes"] * 3)
    assert R.rates(ok)["pass"]["all"]
    r = R.rates(rows(["yes"] * 62, ["no"] * 55 + ["unparsed"] * 7, ["yes"] * 62, ["no"] * 31))
    assert not r["pass"]["faces_removed"] and r["faces_removed_no"] == [55, 62]          # 55/62 = 0.887 < 0.90: unparsed counts against
    r = R.rates(rows(["yes"] * 62, ["no"] * 62, ["yes"] * 62, ["no"] * 27 + ["unparsed"] * 4))
    assert not r["pass"]["no_mask"] and r["no_mask_yes_or_unparsed"] == [4, 31]
    r = R.rates(rows(["yes"] * 49 + ["error"] * 13, ["no"] * 62, ["yes"] * 62, ["no"] * 31))
    assert not r["pass"]["intact"]                                                        # 49/62 = 0.79 < 0.80
    assert R.rates(rows(["yes"] * 62, ["no"] * 62, ["yes"] * 61, ["no"] * 31))["pass"]["area_matched"] is False
