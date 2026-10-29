"""The hero-first video's slot and placeholder logic (no Chrome, no ffmpeg): an ending picks its own panel, unfilled numbers are drawn as
placeholders, filled numbers are escaped, and every scene with film frames or maps carries the CC BY 3.0 credit."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("rv3", ROOT / "scripts/render_video_v3.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

SPEC = json.loads((Path(V.REPO) / "video/scenes_v3.json").read_text())


def test_endings_placeholders_and_credit():
    film = next(s for s in SPEC["scenes"] if s.get("slot") == "film")
    h = V.scene_html(film, {"N": 28}, {"film": "FAIL"}, SPEC, Path("/x"))
    assert "did not line up" in h and "28 real viewers" in h and '<span class="ph">{k}</span>' in h and SPEC["credit"].split(" |")[0] in h
    h = V.scene_html(film, {}, {}, SPEC, Path("/x"))
    assert "PLACEHOLDER" in h and "did not line up" not in h
    assert V.fill("<b>{p}</b>", {"p": "0.01<"}) == "&lt;b&gt;0.01&lt;&lt;/b&gt;"
    for s in SPEC["scenes"]:
        for it in s.get("items", []) + s.get("common", []):
            if it["kind"] in ("frames", "maps"):
                assert "credit" in V.scene_html(s, {}, {}, SPEC, Path("/x"))
    assert abs(sum(s["seconds"] for s in SPEC["scenes"]) - SPEC["crossfade_s"] * (len(SPEC["scenes"]) - 1) - 180) < 1
