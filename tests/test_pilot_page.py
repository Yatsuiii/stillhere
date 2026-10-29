"""The Pilot page (demo/index.html, scripts/build_pilot_page.py): it is up to date; no banned wording, em dash, script or remote
resource; every results panel carries "Not a diagnosis or medical advice." and every panel with film frames the CC BY 3.0 credit; the
numbers come from their JSON; each slot shows a labelled pending panel without its file and fills from fixture files (every face-map
ending, Glimmer pass and fail, the story wall, notthefall pass and not-pass, recorded plans); the sedation tab carries only the frozen
fail format; the media match demo/media/MEDIA.json; and at phone width (390 px) headless Chrome renders it with no horizontal overflow."""
import copy
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pilot_page", REPO / "scripts/build_pilot_page.py")
BP = importlib.util.module_from_spec(spec); spec.loader.exec_module(BP)
PAGE = (REPO / "demo/index.html").read_text()
CHROME = shutil.which("google-chrome-stable") or shutil.which("google-chrome") or shutil.which("chromium")


def build_with(monkeypatch, slots: dict) -> str:
    """The page with each SLOTS file replaced: a dict is the file's content, None means the file is missing."""
    real = BP.load
    by_rel = {BP.SLOTS[k]: v for k, v in slots.items()}
    monkeypatch.setattr(BP, "load", lambda rel: copy.deepcopy(by_rel[rel]) if rel in by_rel else real(rel))
    return BP.build()


def text(page: str) -> str:
    return re.sub(r"\s+", " ", BP.text_of(page))


def panels(page: str) -> list:
    """The outermost panels' HTML: each '<div class="panel' up to its matching '</div>' (panels nested in a panel are part of it)."""
    out, end = [], -1
    for m in re.finditer(r'<div class="panel', page):
        if m.start() < end:
            continue
        depth, i = 0, m.start()
        for tag in re.finditer(r"<div\b|</div>", page[i:]):
            depth += 1 if tag.group().startswith("<div") else -1
            if depth == 0:
                end = i + tag.end(); break
        out.append(page[m.start():end])
    return out


NONE = {k: None for k in BP.SLOTS}


# ---- the page as committed ------------------------------------------------------------------------------------------------------------
def test_page_is_up_to_date():
    assert BP.build() == PAGE


def test_no_banned_wording_em_dash_scripts_or_remote_resources():
    low = text(PAGE).lower()
    for w in BP.BANNED:
        assert not re.search(r"\b" + re.escape(w) + r"\b", low), w
    assert chr(0x2014) not in PAGE and "<script" not in PAGE.lower() and "http://" not in PAGE and "https://" not in PAGE
    assert "Nearest validated option" not in PAGE                                   # planner v2's decline sentence (ruling pending)
    with pytest.raises(AssertionError):
        BP.check(PAGE.replace("</main>", "<p>It detects faces.</p></main>"))


def test_results_panels_carry_the_footer_and_film_frames_the_credit():
    results = PAGE[PAGE.index('id="check"'):PAGE.index("<footer")]
    ps = panels(results)
    assert len(ps) == 9                                                               # check, film, Glimmer, forecast, LPP, 3 slots, sedation
    for p in ps:
        assert p.rstrip().endswith(f'<p class="nd">{html.escape(BP.ND)}</p></div>'), re.sub(r"<[^>]+>", " ", p)[:120]
    for p in panels(PAGE):
        if "<img" in p:
            assert html.escape(BP.CREDIT) in p
    assert PAGE.count("<img") == 6 and html.escape(BP.CREDIT) in PAGE[PAGE.index("<footer"):]


def test_numbers_come_from_their_json():
    t = text(PAGE)
    cm = BP.load("results/conditions_manipulation_check.json")["conditions"]
    assert (f"the intact story kept {round(100 * cm['intact']['recall'])}% of its words; the sentence scramble kept "
            f"{round(100 * cm['sentence_scramble']['recall'])}% of its words in a new order.") in t
    assert "the intact story kept 99% of its words; the sentence scramble kept 98% of its words" in t           # docs/captions.md c.3
    assert "The word scramble kept only 16% of its words, so it was caught as broken before scanning and is not used." in t
    assert "Edit check: Spark Open saw a face in 54 of 62 intact frames, in 0 of 62 filled ones, and in 51 of 62 control frames." in t
    assert "SAM filled the faces it masked with the surrounding colour" in t
    assert "the forecast was worth about 5 real volunteers, and at least 2. It agreed with the classic sound test." in t
    assert f"Le Petit Prince, section 8, held out: {BP.S8_LABEL}." in t and "beaten in 8 of 8 comparisons" in t
    assert ("the default planner (Muse Spark 1.3) accepted 2.3% it should have declined (upper bound 6.5%). Private mode, Muse Spark "
            "Open, which can run on your own servers: 7.6% (Fireworks), 6.8% (Baseten).") in t
    assert "predicted in 7.87 s, peak 3.55 GB" in t and "a forecast, not validated" in t
    lc = BP.load("results/demo_live_tribe_conditions.json")["mean_aud_response_strength"]
    assert f"intact story {lc['intact']:.3f}, sentence scramble {lc['sentence_scramble']:.3f} (raw TRIBE units)" in re.sub(r" ([,.])", r"\1", t)
    assert "it says nothing about language or understanding" in t and PAGE.count("<polyline") == 8     # 4 windows x 2 versions
    assert "so the tool lines above describe that clip, not le_petit_prince." in t                     # DEMO2's Baseten card


def test_exact_upper_bound_matches_the_strict_report():
    assert round(BP.upper95(3, 132), 3) == 0.065 and BP.upper95(5, 5) == 1.0
    assert abs(BP.upper95(0, 132) - (1 - 0.025 ** (1 / 132))) < 1e-9                # closed form when k = 0


def test_sedation_tab_is_the_frozen_fail_format_only():
    sed = PAGE[PAGE.index('<details id="sedation"'):PAGE.index("</details>", PAGE.index('<details id="sedation"'))]
    for line in BP.SEDATION:
        assert f"<p>{html.escape(line)}</p>" in sed
    assert "missed by one" not in PAGE.lower()
    outside = text(PAGE.replace(sed, "")).lower()                                  # elsewhere: the tab's link and the data credit only
    assert outside.count("sedation") == 2 and "sedation case study credits" in outside and "the sedation data (ds003171)" in outside


def test_media_match_their_manifest_and_stay_small():
    meta = json.loads((REPO / "demo/media/MEDIA.json").read_text())
    for it in meta["items"]:
        b = (REPO / "demo/media" / it["file"]).read_bytes()
        assert hashlib.sha256(b).hexdigest() == it["sha256"] and len(b) == it["bytes"], it["file"]
    assert sum(it["bytes"] for it in meta["items"]) < 2_000_000
    assert set(re.findall(r'src="media/([^"]+)"', PAGE)) <= {it["file"] for it in meta["items"]}
    assert meta["frames_manifest_sha256"] == BP.load("results/edit_check_spark_open.json")["frames_manifest_sha256"]


# ---- slots ----------------------------------------------------------------------------------------------------------------------------
def test_missing_files_show_labelled_pending_slots(monkeypatch):
    page = build_with(monkeypatch, NONE); t = text(page)
    for rel in (BP.SLOTS["demo_runs"], BP.SLOTS["glimmer"], BP.SLOTS["story_wall"], BP.SLOTS["d20"]):
        assert f"Fills from {rel} when it lands." in t, rel
    assert f'Fills from {BP.SLOTS["face_map"]} and {BP.SLOTS["film_score"]} when they land.' in t
    for a in BP.ASKS:                                                                # no request file: the two typed asks
        assert html.escape(a["text"]) in page
    assert page.count('class="panel pending"') == len(BP.ASKS) + 4


RUNS = {"label": "demo, not scored", "runs": [
    {"id": "R1", "text": "Plan a story test.", "backend": "spark-1.3", "decision": "decline", "reason": "analysis_not_offered",
     "card": "Declined: analysis_not_offered. Nearest validated option: analysis forecast_tribe_stock. StillHere Pilot plans exploratory "
             "studies only from the validation table.", "total_s": 4.5, "tools": []},
    {"id": "R1", "text": "Plan a story test.", "backend": "spark-open", "decision": "decline", "reason": "analysis_not_offered",
     "nearest": "analysis forecast_tribe_stock", "card": "Declined: analysis_not_offered. Nearest validated option: analysis "
     "forecast_tribe_stock. StillHere Pilot plans exploratory studies only from the validation table.", "tools": []},
    {"id": "R1b", "text": "Plan a stock TRIBE forecast for a story test.", "backend": "spark-1.3", "decision": "accept", "reason": "none",
     "card": "StudySpec: population=healthy_adults.\nManipulation check (exploratory): {\"intact\": {\"recall\": 0.99, \"pass\": true, "
             "\"transcript\": \"zzz secret words\"}}.", "tools": ["list_stimuli", "check_conditions"], "total_s": 6.0}]}
REQS = {"requests": [{"id": "R1", "text": "Plan a story test.", "round": 1, "shows": "a decline"},
                     {"id": "R1b", "text": "Plan a stock TRIBE forecast for a story test.", "round": 2, "reworded_from": "R1",
                      "why": "round 1 was declined"}]}


def test_recorded_plans_fill_with_rounds_and_the_decline_wording_rule(monkeypatch):
    page = build_with(monkeypatch, {**NONE, "demo_requests": REQS, "demo_runs": RUNS}); t = text(page)
    assert page.count('type="radio"') == 1                                           # the rewording is shown under its request
    assert "Round 2, reworded:" in t and "round 1 was declined." in t
    assert "Declined: analysis_not_offered. StillHere Pilot plans exploratory studies only from the validation table." in t
    assert "Nearest validated option" not in page                                    # stored sentence left out
    assert "Nearest tested option: analysis forecast_tribe_stock." in t              # structured field, relabelled
    assert "intact kept 99% of its words, pass." in t and "zzz secret words" not in page
    assert "tools run: list_stimuli, check_conditions" in t and "Shows: a decline" in t


SPARK = BP.load("results/edit_check_spark_open.json")


def test_glimmer_pass_and_fail(monkeypatch):
    t = text(build_with(monkeypatch, {**NONE, "glimmer": {k: SPARK[k] for k in ("rates", "label")}}))
    assert "Edit check: Spark Open and Glimmer saw a face in 54 of 62 intact frames, in 0 of 62 filled ones, and in 51 of 62 control frames." in t
    bad = copy.deepcopy({k: SPARK[k] for k in ("rates", "label")})
    bad["rates"]["faces_removed_no"] = [40, 62]; bad["rates"]["pass"].update(faces_removed=False, all=False)
    t = text(build_with(monkeypatch, {**NONE, "glimmer": bad}))
    assert "The edit check failed (filled frames called no face: 40 of 62), so the film test did not run." in t


def test_story_wall_and_notthefall(monkeypatch):
    wall = {"wall": {"N": 7, "M": 9, "bar_pass": True}, "lines": {"line": "A pre-written line.", "guards": []}}
    d20 = {"primary_intact_gt_shortscram": {"n_pairs": 12, "k_intact_higher": 10, "p": 0.0034, "status": "pass"}}
    t = text(build_with(monkeypatch, {**NONE, "story_wall": wall, "d20": d20}))
    assert "A pre-written line." in t and "7 of 9 story families" in t and "bar met" in t
    assert ("Real listeners' brains followed the intact story more than the scrambled one where TRIBE said they would (10 of 12, "
            "p = 0.0034).") in t
    d20["primary_intact_gt_shortscram"]["status"] = "fail"
    t = text(build_with(monkeypatch, {**NONE, "d20": d20}))
    assert "Status: fail. Only a pass is shown as a result line" in t and "Real listeners' brains followed" not in t


FILM = {"viewers": {"scored": 28}, "gauges": {"VISUAL": {"median_z": 1.5, "ci95": [1.1, 1.9]}, "HEARING": "not run"},
        "isc": {"VISUAL": {"volunteer_equivalent": {"point_k": 3.6, "lower_k": 2.2, "claim_k": 2}}},
        "baselines": {"primary": ["VIS:brightness", "VIS:motion"], "VIS": {"brightness": {"verdict": "TRIBE beats"},
                                                                          "motion": {"verdict": "no difference shown"}}}}
B1 = {"k_above_0": 21, "wilcoxon_p": 0.004}
ENDINGS = {
    "PASS": ("SAM filled the faces it masked; TRIBE forecast the change; that forecast lined up with where 28 real viewers' brains tracked "
             "faces in the intact film (21 of 28 viewers, p = 0.004), and more than for the control."),
    "FAIL not face-specific": ("TRIBE's forecast lined up with where 28 real viewers' brains tracked faces (21 of 28), but no better than "
                               "its forecast for the control: not specific to faces."),
    "FAIL": ("TRIBE's forecast of the filled faces did not line up with where 28 real viewers' brains tracked faces (21 of 28 viewers, "
             "p = 0.004). We show it because we said we would."),
    "UNINFORMATIVE": "The 28 viewers' own face maps did not agree with each other, so this film cannot test the forecast."}


@pytest.mark.parametrize("outcome", list(ENDINGS))
def test_every_face_map_ending(monkeypatch, outcome):
    t = text(build_with(monkeypatch, {**NONE, "face_map": {"outcome": outcome, "viewers": 28, "B1": B1}}))
    assert ENDINGS[outcome] in t and "Data TRIBE's authors never used: 28 real viewers of the intact film." in t


def test_pass_carried_by_visual_parcels_and_not_run_with_the_intact_film(monkeypatch):
    fm = {"outcome": "PASS", "viewers": 28, "B1": B1, "caption": "carried by visual-network parcels"}
    assert ENDINGS["PASS"] + " Most of the match comes from visual-network areas." in text(build_with(monkeypatch, {**NONE, "face_map": fm}))
    fm = {"outcome": "NOT RUN", "why": "edit not confirmed by Glimmer"}
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm}))
    assert ("The face test did not run: edit not confirmed by Glimmer. It was set to run once, by rules written before any brain data "
            "was read.") in t and "fills from results/emofilm_film_score.json when it lands" in t
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm, "film_score": FILM}))
    assert "VISUAL about 4, at least 2 1.50 [1.10, 1.90]" in t
    assert "What it had to beat: brightness, motion, sound envelope and word rate (VIS vs brightness: TRIBE beats; VIS vs motion: no " \
           "difference shown)." in t and "28 real viewers of the intact film" in t


# ---- phone viewport -------------------------------------------------------------------------------------------------------------------
def render(tmp_path: Path, width: int, screenshot: Path = None) -> dict:
    """Headless Chrome lays a window out at least 500 px wide, so the page is loaded unchanged into an iframe WIDTH px wide, whose
    document's scroll width and height a probe page reads (file pages may read each other only with --allow-file-access-from-files);
    then, optionally, a screenshot of the iframe at the page's full height, cropped to WIDTH with ffmpeg."""
    shutil.copytree(REPO / "demo", tmp_path / "demo", dirs_exist_ok=True)
    probe = tmp_path / "probe.html"

    def run(h: int, *a):
        probe.write_text(f'<!doctype html><html><body style="margin:0;background:#fff"><iframe id="f" src="demo/index.html" '
                         f'style="border:0;display:block;width:{width}px;height:{h}px" onload="var d=this.contentDocument.documentElement;'
                         f'document.body.dataset.sw=d.scrollWidth;document.body.dataset.sh=d.scrollHeight"></iframe></body></html>')
        return subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--allow-file-access-from-files",
                               f"--user-data-dir={tmp_path / 'profile'}", "--force-device-scale-factor=1", "--virtual-time-budget=8000",
                               f"--window-size={max(width, 500) + 20},{h}", *a, probe.as_uri()], capture_output=True, text=True, timeout=180)
    m = re.search(r'data-sw="(\d+)" data-sh="(\d+)"', run(900, "--dump-dom").stdout)
    sw, sh = int(m.group(1)), int(m.group(2))
    if screenshot:
        raw = tmp_path / "raw.png"; run(sh, f"--screenshot={raw}")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-vf", f"crop={width}:{sh}:0:0", str(screenshot)], check=True)
    return {"scroll_width": sw, "scroll_height": sh}


@pytest.mark.skipif(CHROME is None or shutil.which("ffmpeg") is None, reason="headless Chrome or ffmpeg is not on this machine")
@pytest.mark.parametrize("width", [390, 1280])
def test_no_horizontal_overflow_at_phone_and_desktop_width(tmp_path, width):
    """Set PILOT_SHOTS=DIR to keep the screenshots (page_390.png, page_1280.png)."""
    out = Path(os.environ.get("PILOT_SHOTS") or tmp_path); out.mkdir(parents=True, exist_ok=True)
    r = render(tmp_path, width, out / f"page_{width}.png")
    assert r["scroll_width"] <= width, r
    assert (out / f"page_{width}.png").stat().st_size > 50_000
