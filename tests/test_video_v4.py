"""The v4 cut (video/scenes_v4.json, scripts/render_video_v4.py) without rendering: the 172 s timeline and its beats, a headline of at
most 7 words that fits the frame at 88 px (49.5 px at 1080 wide), voice-over and subtitles on every scene at a speakable rate, the D23
opener word for word, the banned wording, footers, the ghost-line display rule (draw order, chips, the about-5 line only beside the LPP
ghost line), every slot ending, the masked planner sentence, numbers against their results files, the generated voice-over script, and
the old skeleton cut removed. Rulings of Oct 29 18:15 IST: D20 pass uses the PREREG pass line verbatim and a fail drops the scene and
pairs the live gray line with the LPP ghost line; film result sentences come verbatim from docs/captions.md and "brain signal rose with
on-screen face area" appears only on a PASS; the LIVE beat ends on the NO-GO time courses with the reversed variant."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("rv4", ROOT / "scripts/render_video_v4.py")
V = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(V)
SPEC = json.loads((ROOT / "video/scenes_v4.json").read_text())
SC = {s["id"]: s for s in SPEC["scenes"]}
BEATS = [(0, 10), (10, 24), (24, 42), (42, 60), (60, 92), (92, 118), (118, 150), (150, 172)]
BANNED = ("aware", "awareness", "conscious", "consciousness", "hospital", "detect", "detects", "detected", "detection", "diagnose",
          "a test for patients", "before anyone looked", "face check", "validated", "sedation", "blur", "understanding", "comprehension",
          "beats the sound envelope", "beat the sound envelope")
PREREG = ROOT.parent / "stillhere/PREREG.md"                            # the analysis repository's hashed text, outside this repository
FONT = Path("/usr/share/fonts/noto/NotoSans-Black.ttf")               # what Chrome draws the 800-weight headline with here
FILM_ENDINGS = {"PASS", "PASS_VISUAL", "FAIL_NFS", "FAIL", "UNINFORMATIVE", "NOT_RUN"}


def all_text(o, skip=("mask", "file", "kind", "tone", "key", "view", "show")) -> str:
    if isinstance(o, dict):
        return " ".join(all_text(v) for k, v in o.items() if k not in skip)
    if isinstance(o, list):
        return " ".join(all_text(v) for v in o)
    return o if isinstance(o, str) else ""


def endings_of(slot):
    return next(s["variants"] for s in SPEC["scenes"] if s.get("slot") == slot)


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_timeline_and_beats():
    s = SPEC["scenes"]
    assert s[0]["start"] == 0 and s[-1]["end"] == SPEC["total_s"] == 172
    assert all(a["end"] == b["start"] for a, b in zip(s, s[1:]))
    assert [(b["start"], b["end"]) for b in SPEC["beats"]] == BEATS
    beat = {b["id"]: b for b in SPEC["beats"]}
    for x in s:
        assert beat[x["beat"]]["start"] <= x["start"] < x["end"] <= beat[x["beat"]]["end"], x["id"]


def test_headlines_short_and_large():
    px = SPEC["headline_px"]
    assert px >= 48 and px * 1080 / 1920 >= 48 and f"font-size:{px}px" in V.CSS
    for x in SPEC["scenes"]:
        assert 1 <= len(x["headline"].split()) <= 7, x["headline"]
    if FONT.exists() and importlib.util.find_spec("PIL"):
        from PIL import ImageFont
        f = ImageFont.truetype(str(FONT), px)
        for x in SPEC["scenes"]:
            assert f.getlength(x["headline"]) - (len(x["headline"]) - 1) <= 1712, x["headline"]   # inside the 104 px margins


def test_voice_over_on_every_scene_at_a_speakable_rate():
    vo = SPEC["vo"]
    assert all(0 <= c["start"] < c["end"] <= 172 for c in vo) and all(a["end"] <= b["start"] for a, b in zip(vo, vo[1:]))
    for x in SPEC["scenes"]:
        cover = sum(max(0, min(c["end"], x["end"]) - max(c["start"], x["start"])) for c in vo)
        assert cover >= min(1.0, x["end"] - x["start"]), f"{x['id']} has no voice-over"
    for c in vo:
        for t in (c.get("variants") or {"": c["text"]}).values():
            assert V.words(t) / (c["end"] - c["start"]) <= 3.0, t
    total = sum(max(V.words(t) for t in (c.get("variants") or {"": c["text"]}).values()) for c in vo)
    assert total / 172 <= 2.6


def test_opener_word_for_word_under_the_cold_open():
    opener = (ROOT / "stillhere/public_text/d23_opener.txt").read_text().strip()
    first = [c for c in SPEC["vo"] if c["start"] < 24]
    assert first[0]["start"] < 10 and " ".join(c["text"] for c in first) == opener


def test_banned_wording_everywhere():
    texts = {"scenes": all_text(SPEC["scenes"]), "vo": all_text(SPEC["vo"]), "script": (ROOT / "docs/vo_script.md").read_text()}
    for name, t in texts.items():
        low = t.lower()
        for w in BANNED:
            assert not re.search(r"\b" + re.escape(w) + r"\b", low), f"{w!r} in {name}"
        assert chr(0x2014) not in t, f"em dash in {name}"


def test_footers_and_film_credit():
    ctx = {"ghost": V.synthetic_ghost(), "film_ghost": V.synthetic_film_ghost(), "maps": {"R": Path("/x/R.png"), "M": Path("/x/M.png")}}
    film_ok = (V.FILM / "edits_v2.json").exists()
    ctx["d20_line"] = V.d20_line(SPEC)
    for x in SPEC["scenes"]:
        options = [dict()] + ([{x["slot"]: e} for e in x["variants"]] if "slot" in x else [])
        for end in options:
            if x["id"] == "film_ghost" and not film_ok or V.resolve(x, end) is None:
                continue
            for _, items in V.resolve(x, end):
                h = V.scene_html(x, items, {}, SPEC, ctx)
                assert SPEC["nd"] in h or not x.get("results"), x["id"]
                assert SPEC["credit"].split(" |")[0] in h or not x.get("film"), x["id"]
                assert "<h1>" in h
    assert SC["cold_open"].get("research") and all(SC[i].get("film") for i in SC if SC[i]["beat"] == "film")
    assert SC["close_line"]["items"][-1]["text"] == "Research use. Healthy volunteers and public data. TRIBE v2 CC BY-NC. Not a diagnosis or medical advice."


def test_ghost_line_display_rule():
    rule = (ROOT / "docs/rules/display_rule_ghostline.md").read_text()
    chips = set(re.findall(r"^\| [^|]+ \| ([^|]+?) \|", rule, re.M)) - {"Short chip", "---"}
    frozen = "~5 volunteers' worth (at least 2). Group level."
    assert frozen in rule and V.synthetic_ghost()["frozen_line"]["text"] == frozen
    for x in SPEC["scenes"]:
        steps = V.resolve(x, {})
        prev = []
        for _, items in steps:
            for it in items:
                if it["kind"] == "ghost":
                    assert it["show"] == list(V.GHOST_ORDER[: len(it["show"])]), x["id"]
                    assert prev == it["show"][: len(prev)], f"{x['id']}: a step removed or reordered a line"
                    assert "five" not in it["show"] or it.get("frozen_line"), x["id"]
                    prev = it["show"]
                if it["kind"] == "chip":
                    assert it["text"] in chips, it["text"]
        text = all_text(x)
        if any(it["kind"] in ("livelines", "filmghost") for _, items in steps for it in items) or x["beat"] in ("live", "film"):
            assert "volunteers' worth" not in text and "frozen_line" not in json.dumps(x), x["id"]
    for x in SPEC["scenes"]:
        for items in [x.get("common", [])] + [v for v in x.get("variants", {}).values() if v]:
            for it in items:
                assert it["kind"] != "chip" or it["text"] in chips, it["text"]
    assert "Checked: held-out LPP" in chips and "Forecast only · audio-only, laptop" in chips


def test_d20_endings_keep_172_s():
    assert set(SC["trust_d20"]["variants"]) == set(SC["live_result"]["variants"]) == {"pass", "fail"}
    assert SC["trust_d20"]["variants"]["fail"] is None and SC["live_result"]["slot"] == SC["trust_d20"]["slot"] == "d20"
    for e in ("pass", "fail", None):
        tl = V.timeline(SPEC, {"d20": e} if e else {})
        assert tl[0][1] == 0 and tl[-1][2] == 172 and all(a[2] == b[1] for a, b in zip(tl, tl[1:]))
        ids = [x["id"] for x, _, _ in tl]
        assert ("trust_d20" in ids) == (e != "fail")
        if e == "fail":
            assert next(b for x, _, b in tl if x["id"] == "trust_wall") == 118
    pair = {e: V.resolve(SC["live_result"], {"d20": e})[0][1] for e in ("pass", "fail")}
    assert [i["pair"] for i in pair["pass"] if i["kind"] == "endscreen"] == ["d20"]
    assert [i["pair"] for i in pair["fail"] if i["kind"] == "endscreen"] == ["lpp"]
    ctx = {"ghost": V.synthetic_ghost(), "maps": {}, "d20_line": V.d20_line(SPEC)}
    assert "Real listeners, held-out story" in V.endscreen_html("lpp", {}, ctx)
    assert "Real listeners, different story." in V.endscreen_html("d20", {}, ctx)


def test_d20_pass_line_is_the_prereg_line():
    line = V.d20_line(SPEC)
    if not PREREG.exists():
        pytest.skip("PREREG.md lives in the analysis repository")
    d20 = re.search(r"\*\*D20 \(added.*?(?=\n- \*\*D)", PREREG.read_text(), re.S).group(0)
    m = re.search(r'Pass: "([^"]+)"', d20)
    assert m and line.replace("{d20_N} listeners", "N listeners").replace("notthefall", "`notthefall`") == m.group(1)


def test_film_sentences_verbatim_and_neutral():
    caps = flat((ROOT / "docs/captions.md").read_text())
    for e, items in SC["film_stamp"]["variants"].items():
        for it in items:
            if it["kind"] in ("text", "small"):
                assert it["text"] in caps, (e, it["text"])
    for it in SC["film_stamp"]["common"]:
        assert it["text"] in caps
    rose = "Brain signal rose with on-screen face area."
    for e, items in SC["film_ghost"]["variants"].items():
        assert (rose in all_text(items)) == (e == "PASS"), e
    assert rose not in all_text(SC["film_ghost"]["common"]) and rose.lower() not in all_text(SPEC["vo"]).lower()
    titles = {i["title"] for st in SC["film_ghost"]["steps"] for i in st["items"] if i["kind"] == "filmghost"}
    assert titles == {"Visual areas: TRIBE forecast (gray), {N} viewers (colour), intact film"}
    assert SC["film_maps"]["items"][0]["maps"][1]["title"] == "{N} real viewers: brain signal vs on-screen face area"


def test_live_beat_ends_on_the_no_go_time_courses():
    o = json.loads((ROOT / "results/order_map_gonogo.json").read_text())
    assert o["verdict"] == "NO-GO" and not any(x.get("slot") == "order" for x in SPEC["scenes"])
    assert "order map" not in all_text(SPEC["scenes"]).lower()
    run = o["clips"][V.LIVE_CLIP]["runs"]
    assert run["reversed"]["word_events_reaching_text_lane"] == 0 and "0 words" in V.lanes_html("reversed")
    assert f"{run['intact']['word_events_reaching_text_lane']} words" in V.lanes_html("intact")
    assert "0 words reach the text lane (model input, not a brain result)." in all_text(SC["live_reversed"])
    for cond in ("intact", "reversed", "order"):
        svg = V.livetc_svg(cond)
        assert "Forecasts, not scans." in svg and "#1f6fb4" not in svg                # gray forecasts only, no listener colour
    assert [i["kind"] for _, items in V.resolve(SC["live_result"], {}) for i in items][0] == "endscreen"


def test_film_ghost_line_follows_the_display_rule():
    rule = (ROOT / "docs/rules/display_rule_ghostline.md").read_text()
    assert "## Film ghost line" in rule
    prev, steps = [], SC["film_ghost"]["steps"]
    for n, st in enumerate(steps):
        fg = [i for i in st["items"] if i["kind"] == "filmghost"]; chip = [i for i in st["items"] if i["kind"] == "chip"]
        assert len(fg) == 1 and fg[0]["show"] == ["forecast", "group", "one"][: len(fg[0]["show"])]
        assert prev == fg[0]["show"][: len(prev)]; prev = fg[0]["show"]
        assert bool(chip) == (n >= 1) and (len(fg[0]["show"]) == 1 or chip)          # forecast with ticks, then chip, then group, then one
        assert all(c["text"] == "Film: rule {film_commit}" for c in chip)
    assert prev == ["forecast", "group", "one"]
    doc = V.synthetic_film_ghost()
    assert doc["lines"]["group"]["label"] == "{N} public viewers (average)" and doc["lines"]["one"]["label"] == "1 public viewer"
    t = doc["time_s"]
    assert len(t) == 135 and t[0] == 10 and abs(t[-1] - 184.2) < 1e-6
    svg = V.filmghost_svg(doc, ["forecast", "group", "one"], "x", {})
    assert "Ticks: share of the frame covered by faces, each second" in svg and "#1f6fb4" in svg
    assert "#1f6fb4" not in V.filmghost_svg(doc, ["forecast"], "x", {})
    if (V.FILM / "edits_v2.json").exists():
        ticks = V.face_ticks()
        assert ticks and all(0 <= s_ <= 185 and v > 0 for s_, v in ticks)


def test_ghost_file_swaps_in(tmp_path, monkeypatch):
    doc = {k: v for k, v in V.synthetic_ghost().items() if k != "placeholder"}
    f = tmp_path / "ghostline_lpp_s8.json"; f.write_text(json.dumps(doc))
    monkeypatch.setattr(V, "GHOST", f)
    g = V.ghost_doc()
    assert "placeholder" not in g and "PLACEHOLDER" not in V.ghost_svg(g, "short", list(V.GHOST_ORDER))
    fdoc = {k: v for k, v in V.synthetic_film_ghost().items() if k != "placeholder"}
    ff = tmp_path / "ghostline_film.json"; ff.write_text(json.dumps(fdoc))
    monkeypatch.setattr(V, "FILM_GHOST", ff)
    assert "PLACEHOLDER" not in V.filmghost_svg(V.film_ghost_doc(), ["forecast", "group", "one"], "x", {})


def test_every_slot_ending_resolves():
    assert set(endings_of("film")) == FILM_ENDINGS
    for x in SPEC["scenes"]:
        if "slot" in x:
            for e in x["variants"]:
                assert V.resolve(x, {x["slot"]: e}) or x["variants"][e] is None
            assert any(it["kind"] == "placeholder" for _, items in V.resolve(x, {}) for it in items)
    film_vo = next(c for c in SPEC["vo"] if c.get("slot") == "film")
    assert set(film_vo["variants"]) == FILM_ENDINGS
    assert "It was set to run once, by rules written before any brain data was read." in all_text(SC["film_stamp"]["variants"]["NOT_RUN"])
    assert "On the intact film, the forecast was worth about {veq_film} real viewers, and at least {veq_film_lo}." in all_text(SC["film_stamp"]["common"])
    assert V.vo_cues(SPEC, {"film": "FAIL"})[[c.get("slot") for c in SPEC["vo"]].index("film")]["text"] == film_vo["variants"]["FAIL"]


def test_planner_sentence_is_masked():
    line = "  Declined: clinical_out_of_scope. Nearest validated option: population healthy_adults.\r\n    StillHere Pilot plans"
    pat = SC["plan_decline"]["items"][0]["mask"]
    out = V.mask_text(line, pat)
    assert "validated option" not in out and len(out) == len(line) and out.startswith("  Declined: clinical_out_of_scope. ")
    cast = ROOT / SC["plan_decline"]["items"][0]["file"]
    if not importlib.util.find_spec("PIL"):
        return                                                        # render_cast needs Pillow
    if cast.exists():
        import render_cast as RC
        text = V.mask_text("".join(d for _, d in RC.events(str(cast))), pat)
        assert "validated option" not in text
    if (ROOT / SC["plan_accept"]["items"][0]["file"]).exists():
        import render_cast as RC
        assert "validated option" not in "".join(d for _, d in RC.events(str(ROOT / SC["plan_accept"]["items"][0]["file"])))


def test_numbers_match_their_results():
    b = json.loads((ROOT / "results/section8_baselines.json").read_text())["145"]["LANG"]["tribe_vs"]["word_rate"]
    assert f"{b['k_tribe_higher']} of {b['n']} listeners" in all_text(SC["trust_curve"])
    r = json.loads((ROOT / "results/edit_check_spark_open.json").read_text())["rates"]
    card = all_text(SC["film_check"]["common"])
    for k in ("intact_with_face_yes", "faces_removed_no", "area_matched_yes"):
        assert f"{r[k][0]} of {r[k][1]}" in card
    f = json.loads((ROOT / "results/film_stimulus_checks.json").read_text())["checks"]["control_area_within_1pct"]
    film = all_text(SC["film_edit"])
    assert f"{round(100 * f['in_view_face_px_with_control'])}% of face area" in film and f"the {f['no_control']} largest" in film
    m = json.loads((ROOT / "results/conditions_manipulation_check.json").read_text())["conditions"]["word_scramble"]
    bug = all_text(SC["build_bug"])
    assert f"{round(100 * float(m['recall']))}% of its words" in bug and f"{m['n_words_heard']} of its {m['n_words_reference']} words" in bug
    summ = (ROOT / "results/planner_v2_eval_summary.txt").read_text().split("== seen")[0]
    rates = [float(x) for x in re.findall(r"false_accept_rate ([0-9.]+)", summ)]
    assert [f"{100 * x:.1f}%" for x in rates] == ["2.3%", "7.6%", "6.8%"]
    assert "2.3% vs 7.6% / 6.8% (5% bar)" in all_text(SC["plan_score"])


def test_voice_over_script_is_generated():
    script = (ROOT / "docs/vo_script.md").read_text()
    assert script == V.vo_markdown(SPEC) and "39-56 s per warm worker job" in script        # the open LIVE timing question


def test_old_cut_removed():
    for f in ("video/skeleton.mp4", "video/scenes.json", "video/contact_sheet.png"):
        assert not (ROOT / f).exists(), f


def test_parcel_maps_to_vertices():
    pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
    import numpy as np
    spec = importlib.util.spec_from_file_location("rm", ROOT / "scripts/render_maps.py")
    RM = importlib.util.module_from_spec(spec); spec.loader.exec_module(RM)
    if not (RM.ATLAS / "lh.Schaefer2018_400Parcels_17Networks_order.annot").exists():
        pytest.skip("no Schaefer atlas outside the repository")
    P = RM.parcels()
    v = RM.to_vertices(np.arange(400.0), P)
    assert v.shape == (20484,) and np.isfinite(v).sum() == sum(len(x) for x in P["vertices"])
    assert np.isnan(RM.to_vertices(np.ones(400), P, tested_only=True)).sum() > np.isnan(v).sum()
    a, b = RM.synthetic(P, 1), RM.synthetic(P, 1)
    assert np.array_equal(a, b) and np.abs(a).max() == 1.0
