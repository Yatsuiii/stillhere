"""Cheap baselines (docs/rules/cheap_baselines_rule.md), synthetic data only: the paired comparison's verdicts and determinism; the D20
hook returns the frozen value unchanged and maps records back to listeners; the rendered D20 kernel keeps the frozen call exactly; and,
when the frozen picker code and nilearn are on this machine, an end-to-end run on the picker test's planted world: the add-on leaves the
frozen wall output byte-identical, TRIBE's raw wall lights where the planted own-story signal is, and the word-rate wall does not."""
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from stillhere import cheap_baselines as CB

REPO = Path(__file__).resolve().parents[1]
PICKER = Path(os.environ.get("STILLHERE_PICKER_DIR", REPO.parent / "stillhere" / "d22" / "picker"))


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="needs scipy")
def test_compare_verdicts_and_determinism():
    rng = np.random.default_rng(3); n = 20; base = rng.normal(0, 1, n)
    assert CB.compare(base + 1.0 + rng.normal(0, 0.2, n), base)["verdict"] == "TRIBE beats"
    assert CB.compare(base - 1.0 + rng.normal(0, 0.2, n), base)["verdict"] == "baseline beats"
    assert CB.compare(base + rng.normal(0, 1, n), base)["verdict"] == "no difference shown"
    a, b = rng.normal(0, 1, 15), rng.normal(0, 1, 15)
    assert CB.compare(a, b) == CB.compare(a, b)                                    # seeded bootstrap
    assert CB.compare([1.0], [0.0])["verdict"] == "not enough listeners"


def _render(tmp_path):
    spec = importlib.util.spec_from_file_location("mk", REPO / "scripts" / "make_d20_baselines_kernel.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    fz = tmp_path / "frozen"; fz.mkdir()
    (fz / "stillhere_d20_score.py").write_text('"""frozen stub"""\nimport glob, os, sys\nimport d20_score\nPOSTED = "@@POSTED_HASH@@"\n'
                                               'd20_score.main(POSTED, p8="pass")                 # frozen call\n')
    (fz / "kernel-metadata.json").write_text(json.dumps({"id": "x/stillhere-d20-score"}))
    return Path(mk.render(str(fz), str(tmp_path / "k"), "ab" * 32))


def test_rendered_d20_kernel_keeps_the_frozen_call(tmp_path):
    src = _render(tmp_path).read_text()
    assert src.count('d20_score.main(POSTED, p8="pass")') == 1 and 'POSTED = "' + "ab" * 32 + '"' in src and "@@POSTED_HASH@@" not in src
    assert "def run_gauges" in src and "import json" in src and "G.listener_T1_T3 = _orig_t1_t3" in src


def test_d20_hook_passes_the_frozen_value_through(tmp_path, monkeypatch):
    path = _render(tmp_path); out = tmp_path / "kaggle_out"; out.mkdir()
    G = types.ModuleType("gauges"); C = types.ModuleType("clips"); D = types.ModuleType("d20_score")
    calls = []

    def frozen_t1_t3(r, parts, clip, tr):
        calls.append(clip.stem); return (r["t1"], 0.5, 100)
    G.listener_T1_T3 = frozen_t1_t3
    D.VERSIONS = {"intact": "notthefallintact", "shortscram": "notthefallshortscram"}
    runs = [({"t1": 0.30 + i / 100}, "notthefallintact", f"sub-{i:03d}") for i in range(12)] + \
           [({"t1": 0.10 + i / 100}, "notthefallshortscram", f"sub-{i:03d}") for i in range(12)]

    def main(posted, p8):
        T1 = {"intact": {}, "shortscram": {}}
        for r, stem, lst in runs:
            v = "intact" if stem.endswith("intact") else "shortscram"
            T1[v][lst] = G.listener_T1_T3(r, None, types.SimpleNamespace(stem=stem), 1.5)[0]
        return {"status": "scored", "T1_per_listener": T1}
    D.main = main
    D.paired_test = lambda a, b: {"n_pairs": len(set(a) & set(b)), "p": 0.01, "hl": 0.1, "hl_ci95": [0.0, 0.2]}
    monkeypatch.setitem(sys.modules, "gauges", G); monkeypatch.setitem(sys.modules, "clips", C); monkeypatch.setitem(sys.modules, "d20_score", D)
    src = path.read_text().replace("/kaggle/working/d20/tierD_d20_baselines.json", str(out / "tierD_d20_baselines.json"))
    src = src.replace("g = run_gauges(G, C, r, parts, clip, tr)", "g = _fake_run(G, C, r, parts, clip, tr)")   # no real BOLD in this test
    fake_run = lambda G_, C_, r, parts, clip, tr: {net: {k: float(r["t1"]) + j for j, k in enumerate(CB.KINDS)} for net in ("LANG", "AUD")}
    exec(compile(src, str(path), "exec"), {"__name__": "__main__", "_fake_run": fake_run})
    res = json.loads((out / "tierD_d20_baselines.json").read_text())
    assert calls.count("notthefallintact") == 12 and sys.modules["gauges"].listener_T1_T3 is frozen_t1_t3      # hook removed afterwards
    assert res["frozen_status"] == "scored" and res["matched_runs"] == {"intact": 12, "shortscram": 12} and res["label"] == "exploratory"
    assert set(res["tests"]) == {f"{net}_{k}" for net in ("LANG", "AUD") for k in CB.KINDS}


@pytest.mark.skipif(not (PICKER / "test_picker.py").exists() or importlib.util.find_spec("nilearn") is None,
                    reason="needs the frozen picker code and nilearn")
def test_wall_baselines_end_to_end_on_the_planted_world(tmp_path, monkeypatch):
    sys.path[:0] = [str(PICKER), str(PICKER.parent / "d6"), str(PICKER.parent.parent / "timing")]
    import test_picker as TP
    root = TP.world(str(tmp_path / "w"), gain=0.15)
    M, h, b, w, t, args = TP.run_world(root)
    frozen = {f: hashlib.sha256((Path(root) / "out" / f).read_bytes()).hexdigest() for f in os.listdir(Path(root) / "out")}
    spec = importlib.util.spec_from_file_location("wb", REPO / "scripts" / "wall_baselines.py")
    wb = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, "wb", wb); spec.loader.exec_module(wb)   # pool pickling
    wb.main(types.SimpleNamespace(picker_dir=str(PICKER), manifest_dir=args.manifest, data=args.data, pred=args.pred, out=args.out,
                                  posted_hash=h, workers=4))
    after = {f: hashlib.sha256((Path(root) / "out" / f).read_bytes()).hexdigest() for f in frozen}
    assert after == frozen                                                                      # frozen outputs untouched
    res = json.loads((Path(root) / "out" / "tierD_wall_baselines.json").read_text())
    W = res["walls_raw"]
    assert W["tribe"]["N"] >= W["tribe"]["M"] - 1 and W["word_rate"]["N"] <= 2, {k: (v["N"], v["M"]) for k, v in W.items()}
    assert res["pooled_counted_eligible"]["LANG"]["tribe_vs"]["word_rate"]["verdict"] == "TRIBE beats"
    with pytest.raises(SystemExit):
        wb.main(types.SimpleNamespace(picker_dir=str(PICKER), manifest_dir=args.manifest, data=args.data, pred=args.pred, out=args.out,
                                      posted_hash="0" * 64, workers=1))
