"""R1 on synthetic data and a toy model only (docs/rules/alice_finetuned_rule.md section 7): no Alice BOLD and no real checkpoint is
read by any test. The forecast runner's weight loading and its refusals, the forecast-equals-stock refusal (these need torch), and the
scorer's overlay, pin refusals, stock-file check, writer and paired comparison (these need scipy only, so they run in CI)."""
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
torch = None
REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def RT():
    """The tuned runner, with tribe_common (kept outside the repository) stubbed: only its weight and comparison code is tested."""
    global torch
    torch = pytest.importorskip("torch")
    saved = sys.modules.get("tribe_common")
    sys.modules["tribe_common"] = types.ModuleType("tribe_common")
    try:
        yield load("run_alice_forecast_tuned", REPO / "alice/forecast/run_alice_forecast_tuned.py")
    finally:
        if saved is None:
            sys.modules.pop("tribe_common", None)
        else:
            sys.modules["tribe_common"] = saved


@pytest.fixture(scope="module")
def AF():
    return load("alice_finetuned", REPO / "scripts/alice_finetuned.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Toy:
    def __init__(self):            # torch is imported by the RT fixture every test using Toy requests
        torch.manual_seed(0)
        self._model = torch.nn.Linear(3, 2).eval()


def ckpts(tmp_path, extra=False):
    stock = tmp_path / "best.ckpt"; stock.write_bytes(b"released weights")
    sd = {"model.weight": torch.full((2, 3), 0.5), "model.bias": torch.zeros(2)}
    if extra:
        sd["model.extra"] = torch.zeros(1)
    tuned = tmp_path / "tuned.ckpt"; torch.save({"state_dict": sd}, tuned)
    return stock, tuned


# ---- the forecast runner ----------------------------------------------------------------------------------------------------------
def test_runner_writes_its_own_folder(RT):
    assert RT.R.OUT.name == "out_tuned" and RT.STOCK_OUT.name == "out" and RT.R.WORK.name == "tmp"   # the stock work folders
    assert RT.R.load_model_cpu_brain is RT.load_tuned


def test_tuned_weights_load_strictly(RT, tmp_path, monkeypatch):
    stock, tuned = ckpts(tmp_path)
    monkeypatch.setattr(RT, "STOCK_PIN", sha(stock)); monkeypatch.setattr(RT, "TUNED_PIN", sha(tuned))
    m = Toy(); info = RT.apply_tuned(m, tuned=tuned, stock_ckpt=stock)
    assert info["tuned_tensors"] == 2 and info["tensors_changed_vs_stock"] == 2 and info["training_mode"] is False
    assert torch.equal(m._model.weight, torch.full((2, 3), 0.5))


@pytest.mark.parametrize("which", ["stock", "tuned"])
def test_pin_mismatch_refused(RT, tmp_path, monkeypatch, which):
    stock, tuned = ckpts(tmp_path)
    monkeypatch.setattr(RT, "STOCK_PIN", sha(stock) if which == "tuned" else "0" * 64)
    monkeypatch.setattr(RT, "TUNED_PIN", sha(tuned) if which == "stock" else "0" * 64)
    with pytest.raises(AssertionError, match=f"{which} checkpoint differs"):
        RT.apply_tuned(Toy(), tuned=tuned, stock_ckpt=stock)


def test_extra_key_refused(RT, tmp_path, monkeypatch):
    stock, tuned = ckpts(tmp_path, extra=True)
    monkeypatch.setattr(RT, "STOCK_PIN", sha(stock)); monkeypatch.setattr(RT, "TUNED_PIN", sha(tuned))
    with pytest.raises(RuntimeError):
        RT.apply_tuned(Toy(), tuned=tuned, stock_ckpt=stock)


def test_training_mode_refused(RT, tmp_path, monkeypatch):
    stock, tuned = ckpts(tmp_path)
    monkeypatch.setattr(RT, "STOCK_PIN", sha(stock)); monkeypatch.setattr(RT, "TUNED_PIN", sha(tuned))
    m = Toy(); m._model.train()
    with pytest.raises(AssertionError, match="eval mode"):
        RT.apply_tuned(m, tuned=tuned, stock_ckpt=stock)


def test_forecast_equal_to_stock_refused(RT, tmp_path):
    P = np.random.default_rng(0).standard_normal((10, 4)).astype("float32"); starts = np.arange(10)
    np.savez(tmp_path / "s.npz", P=P, starts=starts); np.savez(tmp_path / "same.npz", P=P, starts=starts)
    np.savez(tmp_path / "t.npz", P=P + 0.01, starts=starts); np.savez(tmp_path / "late.npz", P=P + 0.01, starts=starts + 1)
    d = RT.vs_stock(tmp_path / "t.npz", tmp_path / "s.npz")
    assert d["max_abs_diff"] == pytest.approx(0.01, rel=1e-3) and d["corr"] > 0.99 and d["stock_sha256"] == sha(tmp_path / "s.npz")
    with pytest.raises(AssertionError, match="weights did not apply"):
        RT.vs_stock(tmp_path / "same.npz", tmp_path / "s.npz")
    with pytest.raises(AssertionError, match="different seconds"):
        RT.vs_stock(tmp_path / "late.npz", tmp_path / "s.npz")


# ---- the scorer -------------------------------------------------------------------------------------------------------------------
def fake_root(tmp_path):
    root = tmp_path / "root"
    for d in ("atlas", "alice/bold", "alice/stimulus", "alice/annotations"):
        (root / d).mkdir(parents=True)
    (root / "alice/surprisal.json").write_text("{}")
    pred = tmp_path / "pred"; pred.mkdir()
    for i, k in enumerate(("full", "audio", "shuf")):
        np.savez(pred / f"pred_alice_{k}.npz", P=np.full((3, 2), i, "float32"), starts=np.arange(3))
    return root, pred


def test_pins_required_and_checked(AF, tmp_path):
    _, pred = fake_root(tmp_path)
    with pytest.raises(SystemExit, match="not pinned"):
        AF.check_pins(pred, None)
    pins = {k: sha(pred / f"pred_alice_{k}.npz") for k in AF.MODES}
    assert AF.check_pins(pred, pins) == pins
    np.savez(pred / "pred_alice_shuf.npz", P=np.zeros((3, 2), "float32"), starts=np.arange(3))
    with pytest.raises(SystemExit, match=r"\['shuf'\] differ"):
        AF.check_pins(pred, pins)


def test_overlay_links(AF, tmp_path):
    root, pred = fake_root(tmp_path); ovl = tmp_path / "ovl"
    AF.overlay(root, pred, ovl); AF.overlay(root, pred, ovl)   # a second call finds the same links
    assert os.path.realpath(ovl / "alice/forecast/out") == os.path.realpath(pred)
    for rel in AF.LINKS:
        assert os.path.realpath(ovl / rel) == os.path.realpath(root / rel)
    other = tmp_path / "other"; other.mkdir()
    with pytest.raises(AssertionError, match="links elsewhere"):
        AF.overlay(root, other, ovl)


PUB = json.load(open(REPO / "results/alice_heldout.json"))["stage"]["145_primary"]          # the stock run of record
FIRST = json.load(open(REPO / "results/alice_heldout_first_run.json"))                       # the first run: same forecast pins


def stage(listeners, lang, hearing, z=None):
    """A claim-window stage; its group z are the run of record's unless z (language, hearing) is given."""
    zl, zh = z or (PUB["language"]["z"], PUB["hearing"]["z"])
    return {"per_listener": {"listeners": listeners, "language": lang, "hearing_listeners": listeners, "hearing": hearing},
            "language": {"z": zl}, "hearing": {"z": zh}}


def stock_file(AF, path, st, pins=None):
    """A stock run output file as alice_heldout.score writes it: the pinned forecasts' SHA-256 and the claim-window stage."""
    json.dump({"inputs_sha256": {f"pred_{k}": v for k, v in (pins or AF.STOCK_PINS).items()}, "stage": {"145_primary": st}},
              open(path, "w"))
    return path


def test_stock_file_must_be_the_run_of_record(AF, tmp_path):
    st = stage(["a"], [1.0], [1.0])
    assert AF.stock_scores(stock_file(AF, tmp_path / "ok.json", st)) == st
    with pytest.raises(SystemExit, match="not the stock run"):                   # other forecasts
        AF.stock_scores(stock_file(AF, tmp_path / "other.json", st, {**AF.STOCK_PINS, "full": "0" * 64}))
    first_pins = {k: FIRST["inputs_sha256"][f"pred_{k}"] for k in AF.MODES}
    assert first_pins == AF.STOCK_PINS                                          # the first run's real pins: the same forecasts
    first = stage(["a"], [1.0], [1.0], (FIRST["stage"]["145_primary"]["language"]["z"], FIRST["stage"]["145_primary"]["hearing"]["z"]))
    with pytest.raises(SystemExit, match="not the stock run of record"):        # so its group z tell it apart
        AF.stock_scores(stock_file(AF, tmp_path / "first.json", first, first_pins))


def test_paired_comparison(AF):
    s = stage(["a", "b", "c", "d"], [1.0, 2.0, 0.5, 1.0], [2.0, 2.0, 2.0, 2.0])
    t = stage(["b", "a", "c", "e"], [2.5, 1.2, 0.4, 9.0], [2.0, 2.0, 2.0, 9.0])   # order differs; d and e are in one run only
    r = AF.paired(s, t)
    assert r["language"]["n"] == 3 and r["language"]["finetuned_higher"] == 2
    assert r["language"]["mean_difference"] == pytest.approx((0.2 + 0.5 - 0.1) / 3)
    assert r["language"]["stock_mean_z"] == pytest.approx(3.5 / 3) and r["language"]["finetuned_mean_z"] == pytest.approx(4.1 / 3)
    assert r["hearing"]["mean_difference"] == 0 and r["hearing"]["p_two_sided"] == 1.0
    from scipy import stats
    assert r["language"]["p_two_sided"] == pytest.approx(stats.wilcoxon([0.2, 0.5, -0.1], alternative="two-sided").pvalue)


def test_writer_keeps_the_stock_file(AF, tmp_path, monkeypatch):
    repo = tmp_path / "repo"; (repo / "results").mkdir(parents=True)
    recorded = json.dumps({"stock": "as recorded", "stage": {"145_primary": PUB}})      # the run of record's group z, as stock_scores reads
    (repo / "results/alice_heldout.json").write_text(recorded)
    monkeypatch.setattr(AF.AH, "REPO", str(repo))
    stock_full = stock_file(AF, tmp_path / "stock_full.json", stage(["a", "b"], [1.0, 1.0], [1.0, 1.0]))
    out = tmp_path / "out"; out.mkdir()
    pub = {"rule": "docs/rules/alice_heldout_rule.md", "lines": ["line one", "line two"], "stage": {"145_primary": {"n": 2}}}
    full = {**pub, "stage": {"145_primary": stage(["a", "b"], [1.5, 0.5], [1.0, 2.0])}}
    AF.make_writer(stock_full)(pub, full, str(out))
    assert (repo / "results/alice_heldout.json").read_text() == recorded
    res = json.load(open(repo / "results/alice_finetuned.json"))
    assert res["lines"] == ["Fine-tuned TRIBE v2: line one", "Fine-tuned TRIBE v2: line two"]
    assert res["rule"] == AF.RULE and res["model"]["name"] == "fine-tuned TRIBE v2"
    assert res["vs_stock"]["language"]["n"] == 2 and res["vs_stock"]["label"] == "descriptive, decides nothing"
    assert "per_listener" not in json.dumps(res) and "per_listener" in (out / "alice_finetuned.json").read_text()


def test_writer_not_scored_path_prefixes_once(AF, tmp_path, monkeypatch):
    repo = tmp_path / "repo"; (repo / "results").mkdir(parents=True); monkeypatch.setattr(AF.AH, "REPO", str(repo))
    line = "Alice Datasets: rule committed, not run (coverage): 9 listeners had the language areas in view, fewer than 10."
    res = {"lines": [line]}                                   # the not-scored path hands the writer one dict twice
    AF.make_writer(tmp_path / "unused.json")(res, res, str(tmp_path))
    out = json.load(open(repo / "results/alice_finetuned.json"))
    assert out["lines"] == ["Fine-tuned TRIBE v2: " + line] and "vs_stock" not in out


def test_run_scores_the_tuned_forecasts(AF, tmp_path, monkeypatch):
    """run() hands alice_heldout.score an overlay whose forecast folder is the fine-tuned one, and its writer is the R1 writer."""
    root, pred = fake_root(tmp_path); out = tmp_path / "out"; ovl = tmp_path / "ovl"
    stock_full = stock_file(AF, tmp_path / "stock_full.json", stage(["a"], [1.0], [1.0]))
    pins = {k: sha(pred / f"pred_alice_{k}.npz") for k in AF.MODES}
    seen = {}

    def fake_score(r, o):
        seen["root"], seen["out"] = r, o
        seen["full_pred"] = sha(Path(r) / "alice/forecast/out/pred_alice_full.npz")
        seen["writer"] = AF.AH._write
        return "scored"
    monkeypatch.setattr(AF.AH, "score", fake_score); monkeypatch.setattr(AF.AH, "_write", AF.AH._write)
    assert AF.run(str(root), str(out), pred_dir=str(pred), ovl=str(ovl), pins=pins, stock_full=str(stock_full)) == "scored"
    assert seen["root"] == str(ovl) and seen["out"] == str(out) and seen["full_pred"] == pins["full"]
    assert seen["writer"].__qualname__.startswith("make_writer")
    monkeypatch.setattr(AF, "FORECAST_PINS", None)
    with pytest.raises(SystemExit, match="not pinned"):
        AF.run(str(root), str(out), pred_dir=str(pred), ovl=str(ovl), stock_full=str(stock_full))
    bad_stock = stock_file(AF, tmp_path / "bad.json", stage(["a"], [1.0], [1.0]), {**AF.STOCK_PINS, "shuf": "0" * 64})
    with pytest.raises(SystemExit, match="not the stock run"):                 # refused before any scoring
        AF.run(str(root), str(out), pred_dir=str(pred), ovl=str(ovl), pins=pins, stock_full=str(bad_stock))
