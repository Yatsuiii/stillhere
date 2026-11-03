"""The film ghost line's data (docs/rules/display_rule_ghostline.md, film section, committed Oct 29 in 5064a9b before any Emo-FilM scan
was read): results/ghostline_film.json, display only, no new statistic. Every series comes from the intact film scorer's own functions
(stillhere/emofilm_score.py, through scripts/emofilm_score_runner.py) on the intact score's inputs, each checked first:
- the forecast: the intact kernel's files through the runner's load_forecast (stimulus and stock checkpoint pins), and their SHA-256s
  must equal the ones the score output records (results/emofilm_film_score.json);
- the viewers: the run table and every prepare cache file against prepare_manifest.json (as the score stage checks them), and the
  score's per-viewer file, whose scored viewers must be the score output's count and whose frozen VISUAL z median must equal its median;
- the face ticks: edits_v2.json against the film stimuli MANIFEST (face_map_runner.stimulus_frames), through face_map.face_per_s.

Lines, as the rule fixes them: the forecast is each VISUAL parcel's vertex mean read at grid time - 5 s (tribe_at), cleaned on the grid
(clean), then the parcel mean, z-scored over the grid; each scored viewer's VISUAL rows are made as the scorer makes them for its ISC
curve (film_volumes, clean_full_then_crop, to_grid), then the parcel mean, z-scored over that viewer's non-missing points; the group
line is the mean of the viewers' lines at each point (missing points left out), z-scored over the grid; the single viewer is the
((N - 1) // 2 + 1)-th in ascending frozen VISUAL z, ties by run ID. All z-scores use ddof 1; nothing is smoothed.
Censored frames: to_grid interpolates between kept volumes, also across a censored one; the rule says censored points are missing and
nothing is interpolated across them, so a grid point whose two neighbouring scored volumes are not both kept is left missing. Every
value kept is the scorer's own to_grid value; the counts left missing are recorded.
The chip's {film_commit}: the last commit touching docs/rules/emofilm_rule.md before the first scan was opened, which the prepare
stage's read log records; refused if that log shows a scan opened before it, or if the film section's own commit is not earlier.
Prints counts, shapes and timings only, never a viewer's values.
usage: python scripts/ghostline_film.py --root ANALYSIS_ROOT   (work/stillhere; writes results/ghostline_film.json)"""
import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import emofilm_score as E  # noqa: E402
from stillhere import face_map as FM  # noqa: E402

RULE, FILM_RULE, FILM_SECTION_COMMIT = "docs/rules/display_rule_ghostline.md", "docs/rules/emofilm_rule.md", "5064a9b"
OUT = os.path.join(REPO, "results/ghostline_film.json")
LABELS = {"forecast": "TRIBE v2 forecast", "group": "{N} public viewers (average)", "one": "1 public viewer"}
TICKS_LABEL = "Ticks: share of the frame covered by faces, each second"
CHIP = {"short": "Film: rule {film_commit}", "long": "rule committed before any viewer's scan was read · {film_commit}"}
ORDER = ["forecast", "ticks", "chip", "group", "one"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def zrow(x) -> np.ndarray:
    """z-score over the non-missing points (ddof 1); missing points stay missing."""
    x = np.asarray(x, float); ok = np.isfinite(x)
    out = np.full(x.shape, np.nan); out[ok] = (x[ok] - x[ok].mean()) / x[ok].std(ddof=1)
    return out


def censored_gaps(ft, k, grid) -> np.ndarray:
    """Grid points inside the scored volumes' span whose two neighbouring scored volumes are not both kept (a point exactly at a scored
    volume needs only that volume kept)."""
    ft, k, grid = np.asarray(ft, float), np.asarray(k, bool), np.asarray(grid, float)
    out = np.zeros(grid.shape, bool)
    for i, g in enumerate(grid):
        if g < ft[0] or g > ft[-1]:
            continue
        lo = int(np.searchsorted(ft, g, side="right")) - 1
        out[i] = not (k[lo] if np.isclose(g, ft[lo]) or lo + 1 >= len(ft) else k[lo] and k[lo + 1])
    return out


def viewer_line(raw_vis, keep_full, row: dict, starts, grid) -> tuple:
    """(line on the grid, grid points left missing for censoring): the scorer's ISC series for VISUAL (score_viewer), the parcel mean,
    z-scored over the viewer's non-missing points."""
    tr, onset, so = float(row["tr"]), float(row["film_onset_s"]), float(row["slice_offset_s"])
    idx, ft, _ = E.film_volumes(len(keep_full), tr, onset, so, starts)
    B, k = E.clean_full_then_crop(raw_vis, tr, keep_full, idx)
    m = E.to_grid(B, ft, k, grid).mean(0)
    gap = censored_gaps(ft, k, grid) & np.isfinite(m)
    m[gap] = np.nan
    return zrow(m), int(gap.sum())


def forecast_line(Pf, starts, vis_sets, tr: float, grid) -> np.ndarray:
    pf = E.parcel_means(E.tribe_at(Pf, starts, np.asarray(grid) - E.TRIBE_OFFSET), {"VIS": vis_sets})["VIS"]
    return zrow(E.clean(pf, tr).mean(0))


def group_line(lines) -> np.ndarray:
    L = np.asarray(lines, float); n = np.isfinite(L).sum(0)
    mean = np.where(n > 0, np.nansum(L, 0) / np.maximum(n, 1), np.nan)
    return zrow(mean)


def pick_one(z_by_run: dict) -> str:
    """The ((N - 1) // 2 + 1)-th scored viewer in ascending frozen VISUAL z (the lower middle when N is even), ties by run ID."""
    ranked = sorted(z_by_run.items(), key=lambda kv: (kv[1], kv[0]))
    return ranked[(len(ranked) - 1) // 2][0]


def ticks(per_s) -> list:
    """[film second, share] for every second 0 to 185 s with a share above 0."""
    return [[s, round(float(v), 4)] for s, v in enumerate(np.asarray(per_s)[:186]) if v > 0]


def r3(v) -> list:
    return [None if not np.isfinite(x) else round(float(x), 3) for x in v]


def git(*a) -> str:
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout.strip()


def commits(first_read_utc: str) -> dict:
    """The display rule's commit, the film section's, and {film_commit}, each checked against the first scan read."""
    if git("status", "--porcelain", "--", RULE, FILM_RULE):
        raise SystemExit("refused: a rule file has uncommitted changes")
    read = datetime.datetime.fromisoformat(first_read_utc.replace("Z", "+00:00"))
    when = lambda c: datetime.datetime.fromisoformat(git("log", "-1", "--format=%cI", c))
    film = git("log", "-1", "--format=%h", f"--before={first_read_utc}", "--", FILM_RULE)
    if not film or when(film) >= read or when(FILM_SECTION_COMMIT) >= read:
        raise SystemExit("refused: the film rule or the film section was not committed before the first scan was read")
    rule = git("log", "-1", "--format=%h", "--", RULE)
    return {"rule": {"path": RULE, "commit": rule, "committed": when(rule).isoformat(), "film_section_commit": FILM_SECTION_COMMIT,
                     "film_section_committed": when(FILM_SECTION_COMMIT).isoformat()},
            "film_rule": {"path": FILM_RULE, "commit": film, "committed": when(film).isoformat()}}


def first_read(log_path) -> str:
    """The prepare stage's start: the first time any Emo-FilM scan was opened (its read log's first 'prepare start' line)."""
    for line in open(log_path):
        if line.split()[1:3] == ["prepare", "start"]:
            return line.split()[0]
    raise SystemExit("refused: no prepare start in the read log")


def build(root: str) -> dict:
    ER = _load("efr", "scripts/emofilm_score_runner.py"); FR = _load("fmr", "scripts/face_map_runner.py")
    em = f"{root}/pilot/emofilm"; cache = f"{em}/cache_face_nov1"
    score = json.load(open(os.path.join(REPO, "results/emofilm_film_score.json")))
    if score["gauges"].get("VISUAL") == "not run" or not score.get("visual_ran"):
        raise SystemExit("refused: the intact film score reports VISUAL as not run, so no film ghost line is drawn")
    fc = ER.load_forecast(f"{root}/d22/picker_pred/film_out_v2/intact")
    if fc["sha256"] != score["forecast_sha256"]:
        raise SystemExit("refused: the forecast files are not the ones the intact score read")
    man = json.load(open(f"{cache}/prepare_manifest.json")); runs_path = f"{em}/tos_run_table.json"
    if ER.sha(runs_path) != man["runs_table_sha256"]:
        raise SystemExit("refused: the run table differs from the one prepare used")
    prepared = {k: v for k, v in man["runs"].items() if v["status"] == "prepared"}
    if any(ER.sha(f"{cache}/{v['cache']}") != v["cache_sha256"] for v in prepared.values()):
        raise SystemExit("refused: a cache file differs from the prepare manifest")
    per = json.load(open(f"{cache}/score_per_viewer.json"))
    z = {k: v["gauges"]["VISUAL"]["z"] for k, v in per.items() if v["excluded"] is None}
    N = int(score["viewers"]["scored"])
    if len(z) != N or not np.isclose(np.median(list(z.values())), score["gauges"]["VISUAL"]["median_z"], rtol=0, atol=1e-12):
        raise SystemExit("refused: the per-viewer file is not the intact score's")
    T = {f"{r['subject']}_{r['session']}": r for r in json.load(open(runs_path))["runs"]}
    tr = float(T[next(iter(z))]["tr"]); grid = E.common_grid(tr)
    vis = E.visual_sets(*(f"{root}/atlas/{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")))
    (Pf, sf) = fc["full"]
    lines, gaps = {}, {}
    for run in sorted(z):
        c = np.load(f"{cache}/{prepared[run]['cache']}")
        assert c["raw_VIS"].shape[0] == len(vis), "the cache's VISUAL rows are not the atlas's VISUAL parcels"
        lines[run], gaps[run] = viewer_line(c["raw_VIS"], c["keep"], T[run], sf, grid)
    one = pick_one(z)
    per_s = FM.face_per_s([[x["px"] for x in faces] for faces in FR.stimulus_frames(f"{root}/pilot/film/film_stim_v2/edits_v2.json")], 24.0)
    read = first_read(f"{em}/face_scoring_nov1.log")
    cm = commits(read); fcm = cm["film_rule"]["commit"]
    lab = {k: v.format(N=N) for k, v in LABELS.items()}
    return {"format": "stillhere-ghostline/1", "label": "display only; no new statistic", **cm,
            "first_scan_read_utc": read, "read_log": "the prepare stage's read log (pilot/emofilm/face_scoring_nov1.log)",
            "dataset": "Emo-FilM (ds004892), Tears of Steel, the intact film score's scored viewers",
            "title": "Tears of Steel, intact film (Emo-FilM)",
            "region": "the intact film rule's VISUAL parcels (emofilm_score.visual_sets: VisCent and VisPeri, both hemispheres)",
            "n_parcels": len(vis), "tr_s": tr, "N": N,
            "grid": {"film_s": [round(float(grid[0]), 1), round(float(grid[-1]), 1)], "points": len(grid), "step_s": tr},
            "time_s": [round(float(t), 1) for t in grid], "draw_order": ORDER,
            "chip": {k: v.format(film_commit=fcm) for k, v in CHIP.items()},
            "lines": {"forecast": {"label": lab["forecast"], "style": "gray dashed", "values": r3(forecast_line(Pf, sf, vis, tr, grid))},
                      "group": {"label": lab["group"], "style": "solid colour", "values": r3(group_line(list(lines.values())))},
                      "one": {"label": lab["one"], "style": "thin solid", "values": r3(lines[one])}},
            "ticks": ticks(per_s), "ticks_label": TICKS_LABEL,
            "provenance": {"one_viewer_run": one,
                           "selection": "median frozen VISUAL z of the intact score's per-viewer file: the ((N - 1) // 2 + 1)-th "
                                        "ascending, ties by run ID",
                           "forecast_sha256": fc["sha256"]["pred_full.npz"], "per_viewer_sha256": ER.sha(f"{cache}/score_per_viewer.json"),
                           "prepare_manifest_sha256": ER.sha(f"{cache}/prepare_manifest.json"),
                           "censored_points_left_missing": {"all_viewers": int(sum(gaps.values())), "viewers_with_any": sum(1 for g in gaps.values() if g),
                                                            "one": gaps[one]},
                           "disclosure": "before the read guard existed, one session motion file was opened for its shape only "
                                         "(docs/rules/face_map_rule.md, its disclosure); it is not a brain scan"},
            "extracted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    a = ap.parse_args(); t0 = time.time()
    doc = build(os.path.abspath(a.root))
    json.dump(doc, open(OUT, "w"), indent=1)
    L = doc["lines"]
    print(f"{os.path.relpath(OUT, REPO)}: N {doc['N']}, {doc['n_parcels']} VISUAL parcels, {doc['grid']['points']} grid points; "
          f"missing points: forecast {L['forecast']['values'].count(None)}, group {L['group']['values'].count(None)}, "
          f"one {L['one']['values'].count(None)}; {len(doc['ticks'])} face ticks; chip {doc['chip']['short']}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
