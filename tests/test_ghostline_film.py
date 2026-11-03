"""results/ghostline_film.json against the film section of docs/rules/display_rule_ghostline.md, and scripts/ghostline_film.py's steps
on synthetic inputs (censored gaps, z-scoring over non-missing points, the group mean, the single-viewer pick and its ties, the ticks).
The recomputation from the analysis tree runs only where that tree is (STILLHERE_ROOT, default ../stillhere)."""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = json.loads((REPO / "results/ghostline_film.json").read_text())
ROOT = Path(os.environ.get("STILLHERE_ROOT", REPO.parent / "stillhere"))


@pytest.fixture(scope="module")
def GF():
    pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
    spec = importlib.util.spec_from_file_location("ghostline_film", REPO / "scripts/ghostline_film.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def vals(k):
    return np.array([np.nan if v is None else v for v in DOC["lines"][k]["values"]])


def test_the_file_has_the_rules_layout_grid_labels_and_chip():
    assert DOC["format"] == "stillhere-ghostline/1" and DOC["label"] == "display only; no new statistic"
    assert DOC["grid"] == {"film_s": [10.0, 184.2], "points": 135, "step_s": 1.3} and len(DOC["time_s"]) == 135
    assert np.allclose(np.diff(DOC["time_s"]), 1.3) and DOC["time_s"][0] == 10.0
    assert set(DOC["lines"]) == {"forecast", "group", "one"} and all(len(v["values"]) == 135 for v in DOC["lines"].values())
    N = DOC["N"]
    assert [DOC["lines"][k]["label"] for k in ("forecast", "group", "one")] == ["TRIBE v2 forecast", f"{N} public viewers (average)",
                                                                               "1 public viewer"]
    fc = DOC["film_rule"]["commit"]
    assert DOC["chip"] == {"short": f"Film: rule {fc}", "long": f"rule committed before any viewer's scan was read · {fc}"}
    assert DOC["draw_order"] == ["forecast", "ticks", "chip", "group", "one"]
    assert DOC["ticks_label"] == "Ticks: share of the frame covered by faces, each second"
    assert DOC["region"].startswith("the intact film rule's VISUAL parcels") and "frozen_line" not in DOC


def test_n_is_the_intact_scores_scored_count_and_the_film_commit_predates_the_first_scan_read():
    score = json.loads((REPO / "results/emofilm_film_score.json").read_text())
    assert DOC["N"] == score["viewers"]["scored"] and score["gauges"]["VISUAL"] != "not run"
    when = lambda c: subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cI", c], capture_output=True, text=True,
                                    check=True).stdout.strip()
    import datetime
    read = datetime.datetime.fromisoformat(DOC["first_scan_read_utc"].replace("Z", "+00:00"))
    for c in (DOC["film_rule"]["commit"], DOC["rule"]["film_section_commit"]):
        assert datetime.datetime.fromisoformat(when(c)) < read
    later = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h", f"--since={when(DOC['film_rule']['commit'])}",
                            f"--before={DOC['first_scan_read_utc']}", "--", "docs/rules/emofilm_rule.md"], capture_output=True,
                           text=True, check=True).stdout.split()
    assert later == [DOC["film_rule"]["commit"]]                           # the last commit touching the film rule before the read


def test_every_line_is_z_scored_over_its_own_points_and_gaps_stay_gaps():
    for k in ("forecast", "group", "one"):
        v = vals(k); ok = np.isfinite(v)
        assert abs(v[ok].mean()) < 0.01 and abs(v[ok].std(ddof=1) - 1) < 0.01, k
    assert np.isfinite(vals("forecast")).all()
    assert all(isinstance(x, (float, int)) or x is None for k in DOC["lines"] for x in DOC["lines"][k]["values"])


def test_no_viewer_series_but_the_one_line_and_its_run_only_in_provenance():
    assert set(DOC["lines"]) == {"forecast", "group", "one"} and "viewers" not in DOC
    txt = json.dumps({k: v for k, v in DOC.items() if k != "provenance"})
    assert not re.search(r"sub-S\d\d", txt) and re.fullmatch(r"sub-S\d\d_ses-\d", DOC["provenance"]["one_viewer_run"])
    banned = ("aware", "conscious", "hospital", "detect", "diagnos", "understanding", "comprehension", "validated", "ceiling")
    assert not any(b in txt.lower() for b in banned)
    assert not any(w in DOC["lines"]["one"]["label"].lower() for w in ("pass", "fail", "good", "bad", "typical", "median"))


def test_ticks_are_the_face_share_per_second_above_zero():
    t = DOC["ticks"]
    assert t and all(0 <= s <= 185 and 0 < v <= 1 for s, v in t) and [s for s, _ in t] == sorted({s for s, _ in t})


def test_censored_neighbours_leave_a_point_missing_and_kept_ones_keep_the_scorers_value(GF):
    ft = np.arange(10) * 1.3 + 9.5; k = np.ones(10, bool); k[4] = False
    grid = np.arange(10, 21, 1.3)
    gap = GF.censored_gaps(ft, k, grid)
    between = [(ft[3] < g < ft[5]) for g in grid]
    assert list(gap) == between and gap.any()
    assert not GF.censored_gaps(ft, np.ones(10, bool), grid).any()


def test_z_rows_group_mean_and_the_single_viewer_pick(GF):
    x = np.array([1.0, np.nan, 3.0, 5.0]); z = GF.zrow(x)
    assert np.isnan(z[1]) and np.allclose(z[[0, 2, 3]], (x[[0, 2, 3]] - 3) / 2)
    g = GF.group_line([[1, np.nan, 3, 0], [3, np.nan, 5, 0], [np.nan, np.nan, 4, 6]])     # means 2, missing, 4, 2: a missing value is left out
    assert np.isnan(g[1]) and np.allclose(g[[0, 2, 3]], GF.zrow(np.array([2.0, 4.0, 2.0])))
    zs = {"b": 0.5, "a": 0.5, "c": 0.1, "d": 2.0}
    assert GF.pick_one(zs) == "a"                                         # 4 viewers: the 2nd ascending; 0.5 tie goes to the earlier ID
    assert GF.pick_one({"x": 3.0, "y": 1.0, "z": 2.0}) == "z"
    assert GF.ticks([0, 0.2, 0, 0.05] + [0] * 300) == [[1, 0.2], [3, 0.05]]


@pytest.mark.skipif(not (ROOT / "pilot/emofilm/cache_face_nov1/prepare_manifest.json").exists(), reason="needs the analysis tree")
def test_the_committed_file_is_what_the_script_makes_from_the_checked_inputs(GF):
    doc = GF.build(str(ROOT))
    for k in ("lines", "ticks", "chip", "N", "time_s", "grid", "film_rule", "first_scan_read_utc", "n_parcels"):
        assert doc[k] == DOC[k], k
    assert doc["provenance"]["one_viewer_run"] == DOC["provenance"]["one_viewer_run"]
