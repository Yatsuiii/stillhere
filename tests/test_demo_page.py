"""The StillHere Pilot page (demo/index.html) is rebuilt from JSON and checked: it is up to date; every quoted line and note is PREREG §7
word for word with its numbers filled from results/section8.json; every section-8 mention carries the label "recomputed with code built
during the event; first measured Oct 22-23"; the planner section is labelled in progress and shows no invented model output; the
manipulation checks, diarisation and live forecast numbers come from their JSON; the sedation case study carries only frozen group
statistics (counts per state and the desk's licensed lines) with no per-volunteer value or subject ID; the results file equals a rebuild
from the analysis tree (when it is on this machine); no forbidden wording, no external resources; text colours pass WCAG AA."""
import html
import importlib.util
import json
import re
from pathlib import Path

import pytest

from stillhere import desk, frozen

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO.parent / "stillhere" / "PREREG.md"
FORBIDDEN = ("t" + "aken", "312", "j_max", "duration", "word count", "words per", "beats the control group", "detects awareness",
             "any language", "first test that needs no controls", "hidden aware mind", chr(0x2014))
D18_NEVER = re.compile(r"\b(aware|awareness|conscious)\b", re.I)          # PREREG D18: never what StillHere detects or a person's result


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


BP = _mod("build_demo_page")
PAGE = (REPO / "demo/index.html").read_text()
TEXT = re.sub(r"<[^>]+>", " ", PAGE)
S8, LINES = BP.load("results/section8.json"), BP.load("demo/lines.json")
V = BP.line_values(S8, LINES)


def test_page_is_up_to_date():
    assert BP.build() == PAGE


@pytest.mark.skipif(not PREREG.exists(), reason="PREREG.md is not on this machine")
def test_every_quoted_line_is_prereg_word_for_word():
    text = PREREG.read_text().replace("**", "")
    quoted = [*LINES["pitch"], BP.fill(LINES["m3"], V)] + [BP.fill(ln["quote"], V) for ln in LINES["lpp"]] \
        + [BP.fill(ln["note"], V) for ln in LINES["lpp"] if ln["note"]]
    for q in quoted:
        assert q in text, q
        assert html.escape(q) in PAGE, q
    assert html.escape(" ".join(LINES["pitch"])) in PAGE                          # D18 first sentence + the rest of §7's first line


def test_every_section_8_mention_carries_its_label():
    label = LINES["s8_label"]
    assert S8["label"] == label
    blocks = re.split(r"</li>|</figcaption>|</th>|</td>|</p>", PAGE)
    nav = PAGE[PAGE.index('<ul class="tabs">'):PAGE.index("</ul>", PAGE.index('<ul class="tabs">'))]
    for b in blocks:
        if "section 8" in b and b not in nav and not b.rstrip().endswith("</a>"):
            assert label in b, re.sub(r"<[^>]+>", "", b)[-160:]
    assert PAGE.count(label) >= 8


def test_planner_section_is_in_progress_and_invents_no_model_output():
    plan = PAGE[PAGE.index('id="plan"'):PAGE.index('id="scans"')]
    assert BP.IN_PROGRESS in plan and "No planner model output is shown yet." in plan
    vt = BP.load("stillhere/validation_table.json")
    for k, i, st, _ in BP.planned_today(vt):
        assert f"<td>{html.escape(i)}</td><td>{html.escape(st)}</td>" in plan, i
    for ex in LINES["planner_examples"]:
        code, near = BP.expected_decline(vt, ex)
        assert html.escape(ex["request"]) in plan and f"<b>{code}</b>" in plan and html.escape(near) in plan
    assert BP.expected_decline(vt, LINES["planner_examples"][0])[0] == "population_not_validated"
    assert BP.expected_decline(vt, LINES["planner_examples"][1])[0] == "analysis_not_offered"


def test_pilot_numbers_come_from_their_json():
    cm, dz, lt = BP.load("results/conditions_manipulation_check.json"), BP.load("results/diarization_validation.json"), \
        BP.load("results/live_tribe_librivox.json")
    for name, r in cm["conditions"].items():
        assert f'<td>{name}</td><td>{r["recall"]:.3f}</td><td>{r["order_vs_expected"]:.3f}</td>' in PAGE, name
    assert cm["conditions"]["word_scramble"]["pass"] is False and '<td>word_scramble</td><td>0.163</td><td>0.118</td><td><span class="res-fail">fail' in PAGE
    assert f'agreement {dz["agreement"]:.3f}' in PAGE and f'{dz["speaker_changes"]} speaker changes against {dz["true_speaker_changes"]} true' in PAGE
    assert f'{lt["predicted_aud_response_strength"]:.3f}' in PAGE and html.escape(lt["label"]) in PAGE and html.escape(lt["wording"]) in PAGE
    assert "(exploratory)" in PAGE


def test_sedation_case_study_is_frozen_group_statistics_only():
    g = BP.load("results/gate1_real.json"); fz = frozen.Frozen(g)
    sed = PAGE[PAGE.index('id="sedation"'):PAGE.index('id="desk"')]
    assert "Included per state: " + ", ".join(f'{s.capitalize()} {g["included"][s]}' for s in BP.STATES) in sed
    for lid in desk.licensed_lines(fz):
        assert html.escape(desk.render(fz, lid)) in sed, lid                     # k of N, k*, HL with CI, p, fallback lines, lag marks
    assert html.escape(desk.TIMING_NOTE) in sed
    assert "<svg" not in sed and "fill-opacity" not in PAGE                      # no chart, no per-volunteer dots anywhere
    assert not (REPO / "results/cohort.json").exists() and not hasattr(_mod("build_demo_results"), "cohort")
    assert not re.search(r"sub-\w+", PAGE) and "demo_volunteer" not in json.dumps(g)


@pytest.mark.skipif(not PREREG.exists(), reason="the analysis tree is not on this machine")
def test_results_file_equals_a_rebuild_from_the_analysis_tree():
    BR = _mod("build_demo_results"); root = PREREG.parent
    assert BR.section8(root) == BP.load("results/section8.json")


def test_placeholders_for_the_pending_scans():
    for k, t in BP.PLACEHOLDERS.items():
        assert f'<p class="pending">{html.escape(t)}</p>' in PAGE[PAGE.index(f'id="{k}"'):], k


def test_no_forbidden_wording_ids_or_external_resources():
    low = PAGE.lower()
    for w in FORBIDDEN:
        assert w not in low, w
    assert not D18_NEVER.search(TEXT)
    assert "<script" not in low and "http://" not in low and "https://" not in low and "@import" not in low and " src=" not in low


def _lum(hexc):
    ch = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def test_text_colours_pass_wcag_aa_on_white():
    for colr in (BP.INK, BP.MUTED, *BP.COL.values()):
        assert (1.05) / (_lum(colr) + 0.05) >= 4.5, colr
