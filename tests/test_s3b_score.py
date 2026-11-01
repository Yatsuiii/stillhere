"""S3b's scorer checks (scripts/s3b_score.py) on stand-in part folders: pins, the notebooks' manifests, run and gate records, parts.json,
each control's provenance, shape, values and second grid, the overlay, --out, --check-only and the not-run line. No listener data and no
statistic: the scoring step itself is checked on the real files before the pins commit (docs/rules/s3b_late_rule.md, section 6)."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s3b_score", REPO / "scripts/s3b_score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

T, V = 6, 20484
OWNS = {"1": ["a1", "a2"], "2": ["b1"]}
STEMS = ["a1", "a2", "b1"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def npz(p, P, starts=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, P=P.astype("float32"), starts=np.arange(len(P)) if starts is None else starts)


def manifest(root):
    files = sorted(x for x in root.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256")
    (root / "MANIFEST.sha256").write_text("".join(f"{sha(x)}  {x.relative_to(root)}\n" for x in files))


def make_part(root, part, rng):
    gate = {"clips": {s: {"pass": True, "scored_sha256": h, "median_vertex_r": 1.0} for s, h in S.REFS.items()}, "pass": True}
    clips = {}
    for s in OWNS[part]:
        npz(root / f"PRED/{s}/pred_timing_exact.npz", rng.normal(size=(T, V)))
        npz(root / f"PRED/{s}/pred_full.npz", rng.normal(size=(T, V)))
        (root / f"PRED/{s}/provenance.json").write_text(json.dumps({
            "stem": s, "ckpt_sha256": S.STOCK_PIN, "ckpt_sha256_by_mode": {m: S.STOCK_PIN for m in ("full", "audio", "timing_exact")},
            "timing_mode": "exact (S3b run late)", "s3b_control_sha256": S.HOOK, "s3b_gate": gate}))
        clips[s] = {"timing_exact_word_events": 40, "timing_exact_words_recorded": 40, "timing_exact_hook_words": 55}
    (root / "picker_pred_out").mkdir(parents=True, exist_ok=True)
    (root / S.SUMMARY).write_text(json.dumps({"gate": gate, "clips": clips, "s3b_control_sha256": S.HOOK}))
    (root / "colab_run.json").write_text(json.dumps({
        "s3b_part": int(part), "gpu": "Tesla T4, 580.82.07, 15360 MiB", "notebook_code_cells_sha256": S.NOTEBOOKS[part],
        "kernel_sha256": S.KERNELS[part], "s3b_control_sha256": S.HOOK, "s3b_refs_sha256": S.REFS, "kernel_rc": 0}))
    manifest(root)


@pytest.fixture
def env(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"parts": [{"part": int(p), "owns": o} for p, o in OWNS.items()]}))
    dirs = {p: tmp_path / f"s3b_part{p}_out" for p in OWNS}
    for p, d in dirs.items():
        make_part(d, p, rng)
    merged = tmp_path / "merged"
    for s in STEMS:
        npz(merged / s / "pred_full.npz", rng.normal(size=(T, V))); npz(merged / s / "pred_audio.npz", rng.normal(size=(T, V)))
        (merged / s / "provenance.json").write_text(json.dumps({"stem": s, "ckpt_sha256": S.STOCK_PIN}))
    monkeypatch.setattr(S, "PARTS", str(parts)); monkeypatch.setattr(S, "PARTS_SHA", sha(parts))
    monkeypatch.setattr(S, "scored_inputs", lambda full=True: ({"rows": []}, {}, {}) if full else ({"rows": []}, None, None))
    monkeypatch.setattr(S, "counted_stems", lambda M: STEMS)
    monkeypatch.setattr(S, "compare", lambda a, b: {"pass": True, "median_vertex_r": 1.0})
    monkeypatch.setattr(S, "CONTROL_PINS", {})

    def pin():
        S.CONTROL_PINS.clear(); S.CONTROL_PINS.update(S.pins([str(d) for d in dirs.values()], STEMS, S.owners(STEMS)))

    def run(out="out", check_only=True):
        a = SimpleNamespace(merged=str(merged), s3b=[str(d) for d in dirs.values()], out=str(tmp_path / out), check_only=check_only,
                            print_pins=False)
        return S.main(a, results=str(tmp_path / "results.json"))

    pin()
    return SimpleNamespace(tmp=tmp_path, dirs=dirs, merged=merged, parts=parts, pin=pin, run=run)


def rewrite(root, rel, fn):
    p = root / rel; d = json.loads(p.read_text()); fn(d); p.write_text(json.dumps(d)); manifest(root)


def refusal(env, out="out"):
    assert env.run(out) == 2
    return json.loads((env.tmp / out / "s3b_late.json").read_text())["refused"]


def test_lines_are_the_rules_words():
    text = re.sub(r"\n\s+", " ", (REPO / "docs/rules/s3b_late_rule.md").read_text())
    for key, label in ((True, "Pass"), (False, "Fail"), ("not run", "Not run")):
        rule = re.search(rf'- {label}: "(.*?)"(?=\s|$)', text).group(1)
        assert rule == (S.NOT_RUN if key == "not run" else S.LINES[key]), label


def test_check_only_passes_on_a_complete_set_and_stops_before_any_statistic(env, monkeypatch):
    monkeypatch.setattr(S, "score", lambda *a: pytest.fail("check-only must not score"))
    monkeypatch.setattr(S, "picker", lambda: pytest.fail("check-only must not load the picker"))
    assert env.run() == 0
    chk = json.loads((env.tmp / "out/s3b_check.json").read_text())
    assert sorted(chk["clips"]) == STEMS and sorted(chk["parts"]) == ["1", "2"] and chk["fresh_full_vs_scored_misses"] == []
    assert chk["clips"]["a1"]["words"] == {"word_events": 40, "words_recorded": 40, "hook_words": 55}
    assert not (env.tmp / "results.json").exists()


def test_descriptive_comparison_names_misses_but_does_not_refuse(env, monkeypatch):
    calls = iter([{"pass": True}, {"pass": False}, {"pass": True}])
    monkeypatch.setattr(S, "compare", lambda a, b: next(calls))
    assert env.run() == 0
    assert json.loads((env.tmp / "out/s3b_check.json").read_text())["fresh_full_vs_scored_misses"] == ["a2"]


def test_out_must_be_new_and_empty(env):
    (env.tmp / "used").mkdir(); (env.tmp / "used/x").write_text("x")
    with pytest.raises(SystemExit):
        env.run("used")


def test_no_pins_refuses_and_a_scoring_run_writes_the_not_run_line(env):
    S.CONTROL_PINS.clear()
    assert env.run(check_only=False) == 2
    pub = json.loads((env.tmp / "results.json").read_text())
    assert pub["line"] == "S3b: pre-registered, not run (the control forecasts' SHA-256 are not pinned)." and pub["S3b"] is None


def test_a_changed_file_refuses_and_check_only_writes_no_result(env):
    npz(env.dirs["1"] / "PRED/a2/pred_timing_exact.npz", np.zeros((T, V)))
    assert refusal(env) == ("1 of 3 counted clips have no checked control: a2: part1/PRED/a2/pred_timing_exact.npz differs from its pin")
    assert not (env.tmp / "results.json").exists()


def test_a_file_the_notebook_manifest_disagrees_with_refuses(env):
    npz(env.dirs["2"] / "PRED/b1/pred_full.npz", np.ones((T, V))); env.pin()           # pinned, but not what the notebook listed
    assert "b1: part2/PRED/b1/pred_full.npz differs from the notebook's MANIFEST.sha256" in refusal(env)


def test_an_unpinned_file_refuses(env):
    del S.CONTROL_PINS["part2/picker_pred_out/predict_summary.json"]
    assert refusal(env) == ("1 of 3 counted clips have no checked control: part2/picker_pred_out/predict_summary.json is not pinned "
                            "(1 clips: b1)")


@pytest.mark.parametrize("rel,change,why", [
    ("colab_run.json", lambda d: d.update(notebook_code_cells_sha256="0" * 64), "notebook_code_cells_sha256"),
    ("colab_run.json", lambda d: d.update(kernel_sha256="0" * 64), "kernel_sha256"),
    ("colab_run.json", lambda d: d.update(s3b_control_sha256="0" * 64), "s3b_control_sha256"),
    ("colab_run.json", lambda d: d.update(kernel_rc=1), "kernel_rc"),
    ("colab_run.json", lambda d: d.update(gpu="NVIDIA L4"), "gpu"),
    (S.SUMMARY, lambda d: d["gate"]["clips"]["pieman"].update(scored_sha256="0" * 64), "compared pieman with another reference"),
    (S.SUMMARY, lambda d: d["gate"]["clips"]["lppFR_section8"].update(**{"pass": False}), "gate did not pass on lppFR_section8"),
    (S.SUMMARY, lambda d: d["gate"].update(**{"pass": False}), "gate did not pass"),
    (S.SUMMARY, lambda d: d.update(s3b_control_sha256="0" * 64), "names another hook"),
])
def test_run_and_gate_records_are_checked(env, rel, change, why):
    rewrite(env.dirs["2"], rel, change); env.pin()
    assert why in refusal(env)


def test_the_part_is_read_as_text(env):
    rewrite(env.dirs["1"], "colab_run.json", lambda d: d.update(s3b_part="1")); env.pin()
    assert env.run() == 0


def test_two_folders_for_one_part_refuse(env):
    rewrite(env.dirs["2"], "colab_run.json", lambda d: d.update(s3b_part=1))
    assert refusal(env).startswith("two folders for part 1")


def test_parts_json_is_pinned_disjoint_and_covering(env, monkeypatch):
    env.parts.write_text(env.parts.read_text() + " ")
    assert refusal(env, "o1") == "parts.json differs from its pinned SHA-256"
    env.parts.write_text(json.dumps({"parts": [{"part": 1, "owns": ["a1", "a2", "b1"]}, {"part": 2, "owns": ["b1"]}]}))
    monkeypatch.setattr(S, "PARTS_SHA", sha(env.parts))
    assert refusal(env, "o2") == "parts.json gives b1 to parts 1 and 2"
    env.parts.write_text(json.dumps({"parts": [{"part": 1, "owns": ["a1", "a2"]}]}))
    monkeypatch.setattr(S, "PARTS_SHA", sha(env.parts))
    assert refusal(env, "o3") == "parts.json owns no part for ['b1']"


@pytest.mark.parametrize("change", [lambda d: d.update(timing_mode="none: the S3b gate did not pass"),
                                    lambda d: d["ckpt_sha256_by_mode"].update(timing_exact="0" * 64),
                                    lambda d: d.update(s3b_gate={"pass": False}), lambda d: d.update(stem="a2")])
def test_control_provenance_is_checked(env, change):
    rewrite(env.dirs["1"], "PRED/a1/provenance.json", change); env.pin()
    assert "a1: the control's provenance" in refusal(env)


def test_words_recorded_must_equal_word_events(env):
    rewrite(env.dirs["1"], S.SUMMARY, lambda d: d["clips"]["a2"].update(timing_exact_words_recorded=39)); env.pin()
    assert "a2: words recorded or set differ from the clip's word events" in refusal(env)


@pytest.mark.parametrize("P,starts,what", [(np.full((T, V), np.nan), None, "finite"), (np.zeros((T - 1, V)), None, "shape"),
                                           (np.zeros((T, V)), np.arange(1, T + 1), "second grid")])
def test_control_values_shape_and_grid(env, P, starts, what):
    npz(env.dirs["2"] / "PRED/b1/pred_timing_exact.npz", P, starts); manifest(env.dirs["2"]); env.pin()
    assert f"b1: the control fails the {what} check" in refusal(env)


def test_a_missing_summary_entry_is_a_named_refusal(env):
    rewrite(env.dirs["2"], S.SUMMARY, lambda d: d.pop("clips")); env.pin()
    assert "b1: words recorded or set differ from the clip's word events" in refusal(env)


def test_a_missing_array_is_a_named_refusal(env):
    np.savez(env.dirs["1"] / "PRED/a1/pred_timing_exact.npz", starts=np.arange(T)); manifest(env.dirs["1"]); env.pin()
    assert "a1: KeyError" in refusal(env)


def test_a_missing_run_record_is_a_named_refusal(env):
    (env.dirs["1"] / "colab_run.json").unlink()
    assert refusal(env) == f"{env.dirs['1']}/colab_run.json unreadable (FileNotFoundError)"


def test_overlay_replaces_links_and_refuses_unpinned_controls(env):
    got = S.controls([str(d) for d in env.dirs.values()], STEMS, S.owners(STEMS), str(env.merged))["found"]
    ovl = env.tmp / "ovl"; (ovl / "a1").mkdir(parents=True)
    (ovl / "a1/pred_timing.npz").symlink_to(env.merged / "a1/pred_full.npz")          # a stale link is replaced
    S.overlay(str(env.merged), got, str(ovl))
    assert (ovl / "a1/pred_timing.npz").resolve() == (env.dirs["1"] / "PRED/a1/pred_timing_exact.npz").resolve()
    approx = env.tmp / "approx/pred_timing.npz"; npz(approx, np.zeros((T, V)))
    (ovl / "zz").mkdir(); (ovl / "zz/pred_timing.npz").symlink_to(approx)             # e.g. the dry run's approximate control
    with pytest.raises(S.Refusal, match="not a pinned file"):
        S.overlay(str(env.merged), got, str(ovl))


# ---- cloud review of Nov 1, 06:05 UTC, items 11-15 ----------------------------------------------------------------------------------
def test_every_clip_is_checked_and_the_not_run_line_counts_them(env):
    npz(env.dirs["1"] / "PRED/a2/pred_timing_exact.npz", np.zeros((T, V))); manifest(env.dirs["1"])
    rewrite(env.dirs["2"], "PRED/b1/provenance.json", lambda d: d.update(timing_mode="none")); env.pin()
    S.CONTROL_PINS["part1/PRED/a2/pred_timing_exact.npz"] = "0" * 64                       # a2's control is not its pinned file
    why = refusal(env)
    assert why.startswith("2 of 3 counted clips have no checked control: ")
    assert "a2: part1/PRED/a2/pred_timing_exact.npz differs from its pin" in why and "b1: the control's provenance" in why


def test_a_part_failure_names_every_clip_it_takes(env):
    rewrite(env.dirs["1"], S.SUMMARY, lambda d: d["gate"].update(**{"pass": False})); env.pin()
    assert refusal(env) == "2 of 3 counted clips have no checked control: part 1's gate did not pass (2 clips: a1, a2)"
    line = json.loads((env.tmp / "out/s3b_late.json").read_text())["line"]
    assert line.startswith("S3b: pre-registered, not run (2 of 3 counted clips have no checked control")


def test_a_malformed_manifest_line_is_named(env):
    with open(env.dirs["2"] / "MANIFEST.sha256", "a") as f:
        f.write("onlyonetoken\n")
    env.pin()
    assert "part2/MANIFEST.sha256 has a malformed line: 'onlyonetoken'" in refusal(env)


def test_a_bad_npz_writes_the_not_run_line_with_its_error(env):
    (env.dirs["2"] / "PRED/b1/pred_timing_exact.npz").write_bytes(b"not an npz"); manifest(env.dirs["2"]); env.pin()
    why = refusal(env)
    assert why.startswith("1 of 3 counted clips have no checked control: b1: ") and "Error" in why


def test_check_only_names_a_missing_scored_audio_or_provenance(env):
    (env.merged / "a1/pred_audio.npz").unlink(); (env.merged / "b1/provenance.json").unlink()
    why = refusal(env)
    assert "a1: no scored pred_audio.npz in" in why and "b1: no scored provenance.json in" in why


def test_the_scoring_run_happens_once(env, monkeypatch):
    monkeypatch.setattr(S, "score", lambda merged, found, ovl: ({"S3b": {"pass_": True}, "line": "scored"}, {}))
    lock = env.tmp / "S3B.lock"
    assert env.run("c1") == 0 and not lock.exists()                                     # --check-only takes no lock
    assert env.run("s1", check_only=False) == 0 and lock.exists() and (env.tmp / "results.json").exists()
    with pytest.raises(SystemExit, match="already been scored or refused once"):
        env.run("s2", check_only=False)
    (env.tmp / "results.json").unlink()
    with pytest.raises(SystemExit, match="already been scored or refused once"):
        env.run("s3", check_only=False)                                                  # the lock alone refuses


def test_check_only_and_pins_parse_only_the_manifest(env, monkeypatch):
    def only_manifest(full=True):
        assert not full, "the per-listener targets were parsed"
        return {"rows": []}, None, None
    monkeypatch.setattr(S, "scored_inputs", only_manifest)
    assert env.run() == 0
    a = SimpleNamespace(merged=str(env.merged), s3b=[str(d) for d in env.dirs.values()], out=None, check_only=False, print_pins=True)
    assert S.main(a, results=str(env.tmp / "results.json")) == 0 and not (env.tmp / "S3B.lock").exists()
