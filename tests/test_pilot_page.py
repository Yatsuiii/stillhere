"""The Pilot page (demo/index.html, scripts/build_pilot_page.py): it is up to date; no banned wording, em dash, script or remote
resource; every results panel carries "Not a diagnosis or medical advice." and every panel with film frames the CC BY 3.0 credit; the
numbers come from their JSON; each slot shows a labelled pending panel without its file and fills from fixture files (every face-map
ending, Glimmer pass and fail, the story wall, notthefall pass and not-pass, recorded plans, shown in plain words); the sedation case
study is one link to its write-up and no sedation number is on the page; the media match demo/media/MEDIA.json; and at phone width (390 px) headless Chrome renders it with no horizontal overflow."""
import copy
import datetime
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
    assert chr(0x2014) not in PAGE and "<script" not in PAGE.lower() and "http://" not in PAGE
    assert sorted(re.findall(r"https://[^\"'\s>]+", PAGE)) == sorted([BP.SEDATION_URL, BP.PREREG_URL])   # links out: anchors only
    assert "Nearest validated option" not in PAGE and "validation table" not in PAGE and "no rule" not in text(PAGE)
    assert "before anyone looked" in BP.BANNED and "validated" in BP.BANNED
    for bad in ("<p>It detects faces.</p>", "<p>Checked before anyone looked.</p>", '<img src="https://example.org/x.png">',
                '<a href="https://example.org/">x</a>', f'<link href="{BP.REPO_URL}/x.css">'):
        with pytest.raises(AssertionError):
            BP.check(PAGE.replace("</main>", bad + "</main>"))


def test_results_panels_carry_the_footer_and_film_frames_the_credit():
    results = PAGE[PAGE.index('id="check"'):PAGE.index("<footer")]
    ps = panels(results)
    assert len(ps) == 9                                                               # check, film, Glimmer, forecast, LPP, Alice, wall, D20, film
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
    assert "predicted in 7.87 s, peak 3.55 GB" in t and "a forecast, not checked against scans" in t
    b = BP.load("results/section8_baselines.json")["145"]["LANG"]; s8 = BP.load("results/section8.json")
    assert f"In language areas it beat a simple word-rate model in {b['tribe_vs']['word_rate']['k_tribe_higher']} of 49 listeners." in t
    assert (f"the held-out language result is {s8['stage']['language_z']:.2f} z" in t and f"unpartialled, {b['mean_z']['tribe']:.2f} z "
            "over the same 145 time points" in t and "1.76 z" in t and "1.65 z" in t)
    assert "reversed speech: no word check (0 words heard)" in t
    lc = BP.load("results/demo_live_tribe_conditions.json")["mean_aud_response_strength"]
    assert f"intact story {lc['intact']:.3f}, sentence scramble {lc['sentence_scramble']:.3f} (raw TRIBE units)" in re.sub(r" ([,.])", r"\1", t)
    ghost = [BP.ghost_svg(g) for g in (BP.load("results/ghostline_lpp_s8.json"), BP.load("results/ghostline_film.json")) if g]
    assert "it says nothing about language" in t and "understanding" not in t.lower()
    assert PAGE.count("<polyline") == 8 + sum(g.count("<polyline") for g in ghost)      # the ghost lines appear once their files land
    assert "so the tool lines above describe that clip, not le_petit_prince." in t                     # DEMO2's Baseten card


def test_exact_upper_bound_matches_the_strict_report():
    assert round(BP.upper95(3, 132), 3) == 0.065 and BP.upper95(5, 5) == 1.0
    assert abs(BP.upper95(0, 132) - (1 - 0.025 ** (1 / 132))) < 1e-9                # closed form when k = 0


def test_sedation_is_one_link_to_its_write_up():
    assert PAGE.count(f'href="{BP.SEDATION_URL}"') == 1 and (REPO / "docs/writeup_sedation.md").is_file()
    t = text(PAGE).lower()                                                           # the link line and the TRIBE data credit only
    assert t.count("sedation") == 3 and "the sedation data (ds003171)" in t and "9 of 12" not in t and "missed by one" not in t


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
    assert ("Declined: analysis not offered. Nearest tested option: analysis forecast_tribe_stock. StillHere Pilot plans exploratory "
            "studies only from its table of tested options.") in t                   # stored wording, shown in plain words
    assert "Nearest validated option" not in page and "analysis_not_offered" not in t and BP.DISPLAY_NOTE in t
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
    t = text(build_with(monkeypatch, {**NONE, "story_wall": wall}))
    assert "A pre-written line." in t and "7 of 9 story families" in t and "bar met" in t
    d20 = json.loads((REPO / "results/d20_notthefall.json").read_text())             # not BP.load: build_with patched it above
    t = text(build_with(monkeypatch, {**NONE, "d20": d20}))
    assert ("With zero same-clip controls, stock TRIBE v2's language readout was stronger when the same healthy listeners heard a public "
            "story intact than when they heard its sentences shuffled (Narratives notthefall, 22 listeners, 9.1 minutes = 365 TRs per "
            "version; the order of their two runs is not public).") in t                # PREREG D20's pass line, N filled (17:35 UTC item 2)
    assert ("Each version also had its own TRIBE forecast, so a pass does not separate a change in the listeners' brains from a change in "
            "TRIBE's forecast, and the run order is not public, so it cannot be separated from an order or repeat-listening effect. Intact "
            "vs the coarser scramble (descriptive): HL 0.0215 [95% CI -0.0068, 0.0538], p = 0.0735, 21 listeners, 9.1 minutes = 365 TRs "
            "per version; did not reach p < 0.05.") in t                               # item 3 small print, word for word
    assert ("Exploratory, same test, language areas: word rate alone p = 0.61, sound envelope alone p = 0.93 (22 listeners, 9.1 minutes = "
            "365 TRs per version).") in t and "What it had to beat" not in t.split("notthefall: real")[1].split("Film face test")[0]
    assert BP.PREREG_NOTE in t and f'href="{BP.PREREG_URL}"' in build_with(monkeypatch, {**NONE, "d20": d20})
    assert not re.search(r"\b1[35] of 2[12]\b", t)                                # D23(b): no k of N for D20
    d20["baselines"]["matches_frozen"] = False
    assert "word rate alone" not in text(build_with(monkeypatch, {**NONE, "d20": d20}))
    for status in ("fail", "not run"):                                               # any other ending: the panel leaves the page
        d20["primary"]["status"] = status
        t = text(build_with(monkeypatch, {**NONE, "d20": d20}))
        assert "notthefall" not in t.lower().replace("the sedation data", "") and "stock TRIBE v2's language readout" not in t


def test_d20_summary_holds_no_k_of_n_or_per_listener_values():
    d = BP.load("results/d20_notthefall.json"); raw = json.dumps(d)
    assert "k_intact_higher" not in raw and "per_listener" not in raw and d["primary"]["status"] == "pass" and d["primary"]["n"] == 22
    assert set(d["sources"]) == {"tierD_d20_score.json", "tierD_d20_baselines.json"} and d["baselines"]["matches_frozen"] is True


FILM = {"viewers": {"scored": 28}, "gauges": {"VISUAL": {"median_z": 1.5, "ci95": [1.1, 1.9]}, "HEARING": "not run"},
        "isc": {"VISUAL": {"volunteer_equivalent": {"point_k": 3.6, "lower_k": 2.2, "claim_k": 2}}},
        "baselines": {"primary": ["VIS:brightness", "VIS:motion"], "VIS": {"brightness": {"verdict": "TRIBE beats"},
                                                                          "motion": {"verdict": "no difference shown"}}}}
B1 = {"k_above_0": 21, "kstar": 19, "wilcoxon_p": 0.004, "spin_p": 0.012}
B2 = {"k_above_0": 15, "kstar": 19, "wilcoxon_p": 0.21, "spin_p": 0.31}
ENDINGS = {        # cloud ruling of Nov 1, 05:05 UTC, item 5: fixed before the result
    "PASS": ("Exploratory. SAM filled the faces it masked; TRIBE forecast the change; that forecast lined up with where 28 real viewers' "
             "brains tracked faces in the intact film (21 of 28 viewers, p = 0.004), and more than for the control."),
    "FAIL not face-specific": ("Exploratory. TRIBE's forecast lined up with where 28 real viewers' brains tracked faces (21 of 28), but not "
                               "reliably more than its forecast for the control (B2: 15 of 28, p = 0.21, spin p = 0.31)."),
    "FAIL": ("Exploratory. TRIBE's forecast of the filled faces did not line up with where 28 real viewers' brains tracked faces: it missed "
             "the spin test (21 of 28 viewers above 0, 19 needed; Wilcoxon p = 0.004; spin p = 0.12). We show it because we said we would."),
    "UNINFORMATIVE": "Exploratory. The 28 viewers' own face maps did not agree with each other, so this film cannot test the forecast."}


@pytest.mark.parametrize("outcome", list(ENDINGS))
def test_every_face_map_ending(monkeypatch, outcome):
    b1 = dict(B1, spin_p=0.12) if outcome == "FAIL" else B1
    t = text(build_with(monkeypatch, {**NONE, "face_map": {"outcome": outcome, "viewers": 28, "B1": b1, "B2": B2}}))
    assert ENDINGS[outcome] in t and BP.FILM_FOOT.format(N=28) in t and "never used" not in t
    assert (BP.FACE_B2_NOTE in t) == (outcome in ("PASS", "FAIL not face-specific")) and "no better than" not in t


def test_fail_names_every_missed_criterion():
    assert BP.failed_criteria({"k_above_0": 12, "kstar": 19, "wilcoxon_p": 0.2, "spin_p": 0.01}) == "the Wilcoxon test and the viewer count"
    assert BP.failed_criteria({"k_above_0": 12, "kstar": 19, "wilcoxon_p": 0.2, "spin_p": 0.4}) == "all three criteria"


def test_uninformative_names_its_reason_and_pending_says_what_stops_it(monkeypatch):
    fm = {"outcome": "UNINFORMATIVE", "viewers": 28, "why": "a constant cleaned BOLD row (viewer sub-S03_ses-2, parcel 17)"}
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm}))
    assert "Exploratory. The face test is UNINFORMATIVE: a constant cleaned BOLD row (viewer sub-S03_ses-2, parcel 17)." in t
    t = text(build_with(monkeypatch, NONE))
    assert ("If the edit check fails or gate 0 stops it, it is NOT RUN; if the viewers' maps are not reliable (gate 1), it is "
            "UNINFORMATIVE.") in t


def test_film_headline_never_says_hearing_beats_the_envelope(monkeypatch):
    fs = copy.deepcopy(FILM); fs["baselines"]["primary"].append("AUD:envelope"); fs["baselines"]["AUD"] = {"envelope": {"verdict": "TRIBE beats"}}
    t = text(build_with(monkeypatch, {**NONE, "face_map": {"outcome": "NOT RUN", "why": "x"}, "film_score": fs}))
    assert "AUD vs sound envelope" not in t and "VIS vs brightness: TRIBE beats" in t


def test_ghost_lines_keep_the_draw_order():
    doc = {"time_s": [20.0, 22.0, 24.0, 26.0], "chip": {"short": "Checked: held-out LPP", "long": "held out"},
           "lines": {"forecast": {"label": "TRIBE v2 forecast", "values": [0.1, 0.3, -0.2, 0.0]},
                     "group": {"label": "49 public listeners (average)", "values": [0.0, 0.2, None, 0.1]},
                     "one": {"label": "1 public listener", "values": [0.2, -0.1, 0.0, 0.3]}}}
    svg = BP.ghost_svg(doc)
    order = [svg.index(x) for x in ('class="g-forecast"', "Checked: held-out LPP", 'class="g-group"', 'class="g-one"')]
    assert order == sorted(order) and 'stroke-dasharray="7 5"' in svg.split('class="g-group"')[0]


def test_pass_carried_by_visual_parcels_and_not_run_with_the_intact_film(monkeypatch):
    fm = {"outcome": "PASS", "viewers": 28, "B1": B1, "caption": "carried by visual-network parcels"}
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm}))
    assert ENDINGS["PASS"] + " The match was carried by visual-network parcels." in t and "Most of the match" not in t
    fm = {"outcome": "NOT RUN", "why": "edit not confirmed by Glimmer"}
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm}))
    assert ("The face test did not run: edit not confirmed by Glimmer. It was set to run once, by rules written before any brain data "
            "was read.") in t and "fills from results/emofilm_film_score.json when it lands" in t
    t = text(build_with(monkeypatch, {**NONE, "face_map": fm, "film_score": FILM}))
    assert "VISUAL about 4, at least 2 1.50 [1.10, 1.90]" in t and "Every gauge that ran" in t
    page = build_with(monkeypatch, {**NONE, "face_map": fm, "film_score": FILM})
    assert all(page.rfind("<details", 0, m.start()) > page.rfind("</details>", 0, m.start()) for m in re.finditer("<table", page))
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


def test_every_repository_path_the_page_links_to_exists_at_head():
    """Ruling of Oct 30, 07:40 UTC, a: the page's github.com links go live when the repository turns public on Nov 3, so each path
    they name must exist at HEAD: the static page, the hosted page's files and its evidence panels."""
    texts = [BP.build()] + [f.read_text() for f in sorted((REPO / "web/public").iterdir()) if f.suffix in (".html", ".js")]
    texts.append(BP.d20_html(BP.load(BP.SLOTS["d20"])))
    paths = sorted({m.group(1) for t in texts for m in re.finditer(re.escape(BP.REPO_URL) + r"/(?:blob|tree)/main/([^\"'#?<\s]+)", t)})
    assert "PREREG.md" in paths
    missing = [p for p in paths if subprocess.run(["git", "cat-file", "-e", f"HEAD:{p}"], cwd=REPO, capture_output=True).returncode]
    assert missing == []


def test_story_wall_is_pending_until_its_frozen_cutoff_then_not_run():
    """Ruling of Oct 30, 07:40 UTC, b: no relative days; after Sun Nov 1, 18:00 UTC with no result, not run. Cloud ruling of Oct 31,
    04:30 UTC: the reason shown is the one the not-run record states (any time), never "GPU time ran out" unless the record says so."""
    before = BP.wall_html(None, now=BP.WALL_CUTOFF_UTC - datetime.timedelta(minutes=1))
    after = BP.wall_html(None, now=BP.WALL_CUTOFF_UTC)
    why = "the prediction route failed its D26 self-check"
    recorded = BP.wall_html({"not_run": True, "why": why}, now=BP.WALL_CUTOFF_UTC - datetime.timedelta(hours=30))
    assert "Not scored yet; scored once, as pre-registered (D19)." in before and "pending" in before
    assert "Story wall: pre-registered, not run; no partial scoring." in after and "not run" in after
    assert f"Story wall: pre-registered, not run ({why}); no partial scoring." in recorded and "pending" not in recorded
    assert "GPU time" not in before + after + recorded
    assert BP.WALL_CUTOFF_UTC.isoformat() == "2026-11-01T18:00:00+00:00"


def test_film_station_says_why_and_reads_the_face_test_slot():
    """Ruling of Oct 30, 07:40 UTC, c: the purpose line word for word, and one status line per state of results/face_map.json."""
    page = " ".join((REPO / "web/public/index.html").read_text().split())
    assert ("Why this station: the face test in Evidence asks how TRIBE's forecast changes when the faces in a film are filled in. SAM 3.1 "
            "masked the faces in the first 190 s of Tears of Steel; StillHere filled them with the surrounding colour, and filled a same-size "
            "patch elsewhere in each frame as a control. TRIBE forecasts all three versions, and the face test compares the forecast's face "
            "map with face maps from the scans of 28 people who watched the film. Type \"face\", or another word, to make the same edit on "
            "three frames here.") in page
    assert 'id="film-status"' in page
    assert BP.film_status(None) == {"text": "Face test: not scored yet.", "href": None}
    for o in ("PASS", "FAIL", "FAIL not face-specific", "UNINFORMATIVE"):
        assert BP.film_status({"outcome": o}) == {"text": f"Face test: {o}. Exploratory.", "href": "#face-test"}
    assert BP.film_status({"outcome": "NOT RUN", "why": "gate 0 failed"}) == {
        "text": "The face test did not run: gate 0 failed. This station shows the edit it would have used.", "href": None}
    assert 'id="face-test"' in BP.film_result_html({"outcome": "NOT RUN", "why": "gate 0 failed"}, None)


def test_alice_panel_carries_both_lines_byte_for_byte(monkeypatch):
    """Cloud ruling of Oct 30, 21:05 UTC: right after section 8; the results file's two lines exactly; no dots and no chart; no file,
    no panel."""
    d = json.loads((REPO / BP.ALICE[0]).read_text())
    ps = [p for p in panels(PAGE) if "The Alice Datasets, chapter 1" in p]
    assert len(ps) == 1 and d["lines"][0] in ps[0] and d["lines"][1] in ps[0]
    assert PAGE.index("Le Petit Prince, section 8</h3>") < PAGE.index("The Alice Datasets, chapter 1</h3>")
    assert "<svg" not in ps[0] and "dots" not in ps[0] and "b-info" in ps[0] and ">exploratory<" in ps[0]
    assert "Volunteer-equivalent 1.6, lower bound 0, 22 listeners." in text(ps[0]) and BP.ALICE[1] in ps[0]
    real = BP.load
    monkeypatch.setattr(BP, "load", lambda rel: None if rel == BP.ALICE[0] else real(rel))
    assert "The Alice Datasets, chapter 1" not in BP.build()
