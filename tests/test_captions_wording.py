"""Wording rules for captions, voice-over and slides (ruling of Oct 29, 06:30 UTC): none of the banned words appears, the edit is
described as a fill, and the sedation case study is not in the video."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED = ("aware", "awareness", "conscious", "consciousness", "hospital", "detect", "detects", "detected", "detection", "diagnose",
          "a test for patients")
FILES = ("docs/captions.md", "video/scenes_v3.json")


def test_no_banned_words():
    for f in FILES:
        text = (ROOT / f).read_text().lower()
        for w in BANNED:
            assert not re.search(r"\b" + re.escape(w) + r"\b", text), f"{w!r} in {f}"


def test_fill_wording_and_no_sedation_in_the_video():
    caps = (ROOT / "docs/captions.md").read_text()
    assert "SAM filled the faces it masked with the surrounding colour" in caps and "blur" not in caps.lower()
    scenes = json.dumps(json.loads((ROOT / "video/scenes_v3.json").read_text())).lower()
    assert "sedation" not in scenes and "blur" not in scenes
