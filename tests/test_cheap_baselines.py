"""Cheap baselines (docs/rules/cheap_baselines_rule.md), synthetic data only: the paired comparison's verdicts and determinism; and,
when the frozen picker code and nilearn are on this machine, an end-to-end run on the picker test's planted world: the add-on leaves the
frozen wall output byte-identical, TRIBE's raw wall lights where the planted own-story signal is, and the word-rate wall does not. The
notthefall (D20) kernel has its own tests (tests/test_d20_baselines.py)."""
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
