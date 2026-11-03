"""results/ghostline_paranoia_s1.json against docs/rules/display_rule_ghostline.md's Paranoia section (cloud ruling of Nov 3, 19:05
UTC, item 2): the LPP layout with the forecast and the 22-listener group only, the short view the LPP section's window step (a crop of
the z-scored window), the chip's commit resolved from R2's scoring run, and a caption made of R2's committed numbers only. The
recompute needs the public Paranoia data (STILLHERE_PARANOIA_DIR: reproduce_paranoia.sh's DATA_DIR)."""
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = json.loads((ROOT / "results/ghostline_paranoia_s1.json").read_text())
SECTION = (ROOT / "docs/rules/display_rule_ghostline.md").read_text().split("## Paranoia ghost line", 1)[1]


@pytest.fixture(scope="module")
def G():
    pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
    spec = importlib.util.spec_from_file_location("ghostline_paranoia", ROOT / "scripts/ghostline_paranoia.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_layout_is_the_lpp_files_with_the_forecast_and_the_group_only():
    assert DOC["format"] == "stillhere-ghostline/1" and DOC["label"] == "display only; no new statistic"
    assert set(DOC["lines"]) == {"forecast", "group"} and DOC["draw_order"] == ["forecast", "chip", "listeners"]
    assert DOC["lines"]["group"]["label"] == "22 public listeners (average)" and DOC["lines"]["forecast"]["label"] == "TRIBE v2 forecast"
    assert DOC["n_parcels"] == 19 and DOC["tr_s"] == 1.0 and "provenance" in DOC and "listener" not in json.dumps(DOC["provenance"])


def test_the_short_view_is_the_lpp_window_step_cropped_from_the_z_scored_window():
    r2 = json.loads((ROOT / "results/r2_paranoia.json").read_text())
    j0, w = r2["window"]["volumes"]["story1"]
    assert DOC["volumes"] == [j0, w] and DOC["time_s"] == [float(j) for j in range(j0, w + 1)]
    assert DOC["short_view"]["volumes"] == [j0, j0 + 119] and DOC["short_view"]["index"] == [0, 120]
    for k in ("forecast", "group"):
        v = np.array(DOC["lines"][k]["values"])
        assert len(v) == len(DOC["time_s"]) and abs(v.mean()) < 1e-3 and abs(v.std(ddof=1) - 1) < 1e-3     # z over the whole window
        short = v[slice(*DOC["short_view"]["index"])]
        assert abs(short.std(ddof=1) - 1) > 1e-3 or abs(short.mean()) > 1e-3                                  # the crop is not re-scaled


def test_the_chip_names_r2s_rule_as_of_the_scoring_runs_start(G):
    assert G.r2_commit() == "fc16400"
    assert DOC["chip"] == {"short": "Checked: Paranoia (outside TRIBE's sets)",
                           "long": "rule and forecasts committed before the brain data was downloaded (R2) · fc16400"}
    assert "fc16400" in SECTION and DOC["rule"]["path"] == "docs/rules/display_rule_ghostline.md"


def test_the_caption_is_r2s_committed_numbers_only(G):
    r2 = json.loads((ROOT / "results/r2_paranoia.json").read_text())
    assert DOC["caption"]["text"] == G.caption(r2) and DOC["caption"]["source"] == "results/r2_paranoia.json models.stock"
    assert DOC["caption"]["text"] in " ".join(SECTION.split())
    nums = re.findall(r"\d+(?:\.\d+)?", DOC["caption"]["text"])
    ve, share = r2["models"]["stock"]["volunteer_equivalent_language"], r2["models"]["stock"]["reference"]["share"]
    assert nums == [str(round(ve["point_k"])), str(ve["claim_k"]), f"{ve['point_k']:.1f}", f"{ve['lower_k']:.2f}", str(round(100 * share))]


def test_the_section_says_when_it_was_written_and_discloses_the_draft():
    head = SECTION.split("\n", 1)[0]
    assert "written Nov 3 2026, after R2 was scored on Nov 1" in head and "A draft of this picture was drawn on Nov 3" in SECTION
    assert "LPP section's own window step" in SECTION


@pytest.mark.skipif(not os.environ.get("STILLHERE_PARANOIA_DIR"), reason="needs the Paranoia data (STILLHERE_PARANOIA_DIR)")
def test_the_committed_file_is_what_the_script_makes(G, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", str(tmp_path / "g.json"))
    G.main(os.environ["STILLHERE_PARANOIA_DIR"])
    doc = json.loads((tmp_path / "g.json").read_text())
    for k in ("lines", "chip", "caption", "volumes", "short_view", "n_parcels", "time_s"):
        assert doc[k] == DOC[k], k


def test_the_pages_paranoia_panel_draws_the_short_view_in_the_rules_order_with_the_caption():
    spec = importlib.util.spec_from_file_location("bp_paranoia", ROOT / "scripts/build_pilot_page.py")
    BP = importlib.util.module_from_spec(spec); spec.loader.exec_module(BP)
    r2, r5 = (json.loads((ROOT / p).read_text()) for p in BP.PARANOIA[:2])
    html = BP.paranoia_html(r2, r5, DOC)
    i_f, i_c, i_g = html.index('class="g-forecast"'), html.index(DOC["chip"]["short"].replace("'", "&#x27;")), html.index('class="g-group"')
    assert i_f < i_c < i_g and 'class="g-one"' not in html and 'class="g-five"' not in html
    assert html.count("<polyline") == 2 and DOC["caption"]["text"].replace("'", "&#x27;") in html and DOC["chip"]["long"] in html
    assert BP.short_view(DOC)["time_s"] == DOC["time_s"][:120] and BP.paranoia_html(r2, r5) == BP.paranoia_html(r2, r5, None)


def test_the_video_draws_the_short_view_as_a_strip_in_the_rules_order():
    spec = importlib.util.spec_from_file_location("rv4_paranoia", ROOT / "scripts/render_video_v4.py")
    V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
    full = V.item_html({"kind": "pghost", "show": ["forecast", "chip", "group"]}, {}, {"pghost": DOC})
    assert full.count("<path") == 2 and DOC["chip"]["short"].replace("'", "&#x27;") in full and 'width="1712" height="400"' in full
    assert "seconds of the scan: 20 to 139" in full                                     # the short view, not the whole window
    first = V.item_html({"kind": "pghost", "show": ["forecast", "chip"]}, {}, {"pghost": DOC})
    assert first.count("<path") == 1 and V.item_html({"kind": "pghost", "show": ["forecast"]}, {}, {"pghost": None}) == ""
    with pytest.raises(AssertionError):
        V.item_html({"kind": "pghost", "show": ["forecast", "group"]}, {}, {"pghost": DOC})          # never skips the chip step
