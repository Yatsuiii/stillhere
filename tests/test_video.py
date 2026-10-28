"""The video skeleton (video/scenes.json, scripts/render_video.py) is checked without rendering: every quoted line is PREREG.md (with D18's
rewording of §7's first line) or PLAN.md §1 word for word; desk lines are licensed by the frozen bars; every number on a slide equals its
source JSON; no forbidden wording, subject IDs or external resources; the live egress refusals are refusals; the total length is 2:30 to
3:00. When video/skeleton.mp4 exists, its length, size and missing audio track are checked with ffprobe."""
import html
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from stillhere import desk

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO.parent / "stillhere"
FORBIDDEN = ("t" + "aken", "312", "j_max", "duration", "word count", "words per", "beats the control group", "detects awareness", "any language",
             "first test that needs no controls", "hidden aware mind", chr(0x2014))
D18_NEVER = re.compile(r"\b(aware|awareness|conscious)\b", re.I)          # PREREG D18: never what StillHere detects or a person's result


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


RV = _mod("render_video")
SPEC = json.loads((REPO / "video/scenes.json").read_text())
X = RV.context()
HTML = {s["id"]: RV.scene_html(s, X) for s in SPEC["scenes"]}
ITEMS = [(s["id"], it) for s in SPEC["scenes"] for it in s["items"]]


def test_total_length_is_two_thirty_to_three_minutes():
    assert 150 <= RV.total_seconds(SPEC) <= 180
    assert len({s["id"] for s in SPEC["scenes"]}) == len(SPEC["scenes"])


@pytest.mark.skipif(not (ANALYSIS / "PREREG.md").exists(), reason="PREREG.md and PLAN.md are not on this machine")
def test_quoted_lines_are_word_for_word():
    prereg = (ANALYSIS / "PREREG.md").read_text().replace("**", ""); plan = (ANALYSIS / "PLAN.md").read_text().replace("**", "")
    assert RV.D18_LINE in prereg                                          # D18's replacement for §7's first allowed line
    for sid, it in ITEMS:
        if it["kind"] == "prereg":
            assert it["text"] in prereg, it["text"]
        if it["kind"] == "plan":
            assert it["text"] in plan, it["text"]
            assert not it.get("note") or it["note"] in plan, it["note"]
        if it["kind"] in ("prereg", "plan"):
            assert html.escape(it["text"]) in HTML[sid]
    for sid, it in ITEMS:
        if it["kind"] == "lpp_line":
            ln = X["lines"]["lpp"][it["index"]]
            assert html.escape(RV.BP.fill(ln["quote"], X["v"])) in HTML[sid]
            assert RV.BP.fill(ln["quote"], X["v"]) in prereg


def test_desk_lines_are_licensed_and_rendered():
    lic = desk.licensed_lines(X["fz"])
    for sid, it in ITEMS:
        if it["kind"] == "desk_line":
            assert it["line_id"] in lic
            assert html.escape(desk.render(X["fz"], it["line_id"])) in HTML[sid]
    assert html.escape(desk.TIMING_NOTE) in HTML["sedation"]
    assert "sedation_stage" not in json.dumps(SPEC)                       # the sedation stage line is not licensed by Gate 1
    head = X["card"].text.split("\n")[0]
    assert html.escape(head) in HTML["desk"] and X["card"].badge in HTML["desk"]


def test_numbers_come_from_the_results_files():
    g, s8 = X["g"], X["s8"]
    assert "Included per state: " + ", ".join(f'{s.capitalize()} {g["included"][s]}' for s in RV.BP.STATES) in HTML["sedation"]
    assert f'{g["trs_per_state"]} TRs = {RV.BP.minutes(g["trs_per_state"], g["tr_s"])} minutes per state' in HTML["sedation"]
    for k in ("language_z", "shuffled_z", "surprisal_z", "hearing_z", "envelope_z"):
        assert f"<td>{s8['stage'][k]:.2f}</td>" in HTML["lpp"], k
    for name, r in X["cm"]["conditions"].items():
        assert f'<td>{name}</td><td>{r["recall"]:.3f}</td><td>{r["order_vs_expected"]:.3f}</td>' in HTML["plan"], name
    assert f'agreement {X["dz"]["agreement"]:.3f}' in HTML["plan"] and RV.BP.IN_PROGRESS in HTML["plan"]
    assert f'{X["lt"]["predicted_aud_response_strength"]:.3f}' in HTML["scans"] and html.escape(X["lt"]["label"]) in HTML["scans"]
    for k, t in RV.BP.PLACEHOLDERS.items():
        assert html.escape(t) in HTML["scans"], k


def test_section_8_label_and_group_statistics_only():
    label = X["lines"]["s8_label"]
    for sid in ("curves", "lpp"):
        assert label in HTML[sid], sid
        for b in re.split(r"</li>|</p>|</th>|</blockquote>", HTML[sid]):
            if "section 8" in b and "<h1>" not in b:
                assert label in b, (sid, re.sub(r"<[^>]+>", "", b)[-120:])
    assert all("fill-opacity" not in h for h in HTML.values())                   # no per-volunteer dots on any slide
    assert "cohort" not in json.dumps(SPEC) and "<svg" not in HTML["sedation"]


def test_egress_refusals_are_live_refusals():
    r = RV.refusals()
    assert len(r) == 3 and all(m != "NOT REFUSED" for _, m in r)
    for _, m in r:
        assert html.escape(m) in HTML["tiers"]


def test_no_forbidden_wording_ids_or_external_resources():
    for sid, h in HTML.items():
        low = h.lower()
        for w in FORBIDDEN:
            assert w not in low, (sid, w)
        assert not D18_NEVER.search(re.sub(r"<[^>]+>", " ", h)), sid
        assert not re.search(r"sub-\w+", h), sid
        assert "<script" not in low and "http://" not in low and "https://" not in low and " src=" not in low, sid


@pytest.mark.skipif(not (REPO / "video/skeleton.mp4").exists() or not shutil.which("ffprobe"), reason="no rendered video or no ffprobe")
def test_rendered_video_matches_the_scene_list():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
                          "-of", "json", str(REPO / "video/skeleton.mp4")], capture_output=True, text=True, check=True).stdout
    info = json.loads(out); st = info["streams"]
    assert [s["codec_type"] for s in st] == ["video"]                       # silent: no audio track at all
    assert (st[0]["codec_name"], st[0]["width"], st[0]["height"], st[0]["r_frame_rate"]) == ("h264", 1920, 1080, "30/1")
    assert abs(float(info["format"]["duration"]) - RV.total_seconds(SPEC)) < 0.5
