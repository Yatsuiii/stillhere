"""The video's slot logic (no Chrome, no ffmpeg): an ending picks its own panel and length or leaves the scene out, unfilled numbers are
drawn as placeholders, filled numbers are escaped, results slides carry the medical disclaimer and film slides the CC BY 3.0 credit."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rv3", ROOT / "scripts/render_video_v3.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
SPEC = json.loads((ROOT / "video/scenes_v3.json").read_text())
scene = lambda sid: next(s for s in SPEC["scenes"] if s["id"] == sid)


def test_endings_lengths_omission_and_footers():
    film = scene("b_maps")
    items, secs = V.resolve(film, {"film": "PASS"})
    assert secs == 26 and any("lined up" in i.get("text", "") for i in items)
    items, secs = V.resolve(film, {"film": "FAIL"})
    assert secs == 14 and any("did not line up" in i.get("text", "") for i in items)
    assert V.resolve(scene("a_d20"), {"d20": "omit"}) is None
    items, _ = V.resolve(film, {})
    h = V.scene_html(film, items, {"N": 28}, SPEC, Path("/x"))
    assert "PLACEHOLDER" in h and "28 real viewers" in h and SPEC["nd"] in h and SPEC["credit"].split(" |")[0] in h
    assert V.fill("<b>{p}</b>", {"p": "0.01<"}) == "&lt;b&gt;0.01&lt;&lt;/b&gt;" and '<span class="ph">{k}</span>' in V.fill("{k}", {})
    for s in SPEC["scenes"]:
        r = V.resolve(s, {})
        if s.get("results"):
            assert SPEC["nd"] in V.scene_html(s, r[0], {}, SPEC, Path("/x"))


def test_length_is_about_three_minutes():
    total = lambda end: sum(V.resolve(s, end)[1] for s in SPEC["scenes"] if V.resolve(s, end)) - SPEC["crossfade_s"] * (len(SPEC["scenes"]) - 1)
    assert 160 <= total({"film": "FAIL", "d20": "pass"}) <= 185 and 165 <= total({"film": "PASS", "d20": "pass"}) <= 185
