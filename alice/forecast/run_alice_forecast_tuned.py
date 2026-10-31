"""R1 (cloud ruling of Oct 31, 08:45 UTC; docs/rules/alice_finetuned_rule.md): the three Alice forecasts again, with the fine-tuned
TRIBE v2 weights (tuned.ckpt, SHA-256 pinned below; frozen Oct 26, trained on LPP sections 2-7 and 9 and Narratives, never Alice).
Everything else is the stock runner, imported unchanged (run_alice_forecast.py: the same stimulus, words, events, harness, stitched
windows, placement, work folders and feature cache). The additions: the weights, loaded strictly into the released model right after
it is built, as D3's tuned kernel did (d22/d3/d3_pred_tpl.py apply_weights); a separate output folder, so the stock forecasts are
never touched; and two records. The sound extractor's cache key is the absolute path of the stage's stim.wav, so the tuned run keeps
the stock work folders (prepare_dir writes the same bytes there again; checked unchanged) and every sound and Llama feature is read
from the stock runs' cache (checked: no file added to it). Each forecast's distance from the stock forecast of the same mode is
recorded; the check that the weights applied is tensors_changed_vs_stock, on the model that predicts.
Writes pred_alice_{audio,full,shuf}.npz and stage_*.json, each with "weights", "vs_stock" and "features" records, to forecast/out_tuned.
usage: python run_alice_forecast_tuned.py STAGE   (STAGE: alice_audio | alice_full | alice_shuf)"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import run_alice_forecast as R  # noqa: E402

STOCK_PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"   # the released best.ckpt (D3)
TUNED_PIN = "96923860eb47ee8e8f65e597f2fc216992470852dc07af95112d80d32746421e"   # tuned.ckpt (D3, K4)
TUNED = R.ROOT / "d22/out_train3_real/train/tuned.ckpt"
STOCK_OUT = R.OUT
R.OUT = R.A / "forecast/out_tuned"            # R.WORK and R.CACHE stay the stock runs' (the sound cache is keyed by stim.wav's path)
WEIGHTS = {}


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def apply_tuned(m, tuned=TUNED, stock_ckpt=None) -> dict:
    """D3's apply_weights, tuned branch: the released checkpoint must match its pin, tuned.ckpt must match its pin, and its state dict
    (keys prefixed "model.") loads strictly into the built brain model; at least one tensor must change, and the model stays in eval."""
    ck = stock_ckpt or glob.glob(os.environ.get("HF_HOME", "/tmp/hf") + "/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]
    info = {"stock_ckpt_sha256": sha256(ck)}
    assert info["stock_ckpt_sha256"] == STOCK_PIN, "stock checkpoint differs from the pin"
    info["tuned_ckpt_sha256"] = sha256(tuned)
    assert info["tuned_ckpt_sha256"] == TUNED_PIN, "tuned checkpoint differs from the pin"
    sd = torch.load(tuned, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items()}
    before = {k: v.detach().cpu().clone() for k, v in m._model.state_dict().items()}
    m._model.load_state_dict(sd, strict=True)
    after = m._model.state_dict()
    info["tuned_tensors"] = len(sd)
    info["tensors_changed_vs_stock"] = int(sum(not torch.equal(before[k], after[k].detach().cpu()) for k in before))
    assert info["tensors_changed_vs_stock"] > 0, "tuned.ckpt changed no tensor"
    info["training_mode"] = bool(m._model.training)
    assert not info["training_mode"], "the brain model is not in eval mode"
    return info


_stock_loader = R.load_model_cpu_brain


def load_tuned(feats):
    m = _stock_loader(feats)
    WEIGHTS.clear(); WEIGHTS.update(apply_tuned(m))
    return m


R.load_model_cpu_brain = load_tuned


def vs_stock_arrays(P, starts, stock_npz) -> dict:
    """How far a fine-tuned forecast is from the stock one of the same mode (forecast against forecast; no brain data)."""
    b = np.load(stock_npz)
    assert np.array_equal(starts, b["starts"]), "tuned and stock forecasts cover different seconds"
    P, Q = np.asarray(P, "float64"), b["P"].astype("float64")
    d = {"max_abs_diff": float(np.abs(P - Q).max()), "corr": float(np.corrcoef(P.ravel(), Q.ravel())[0, 1]),
         "stock_sha256": sha256(stock_npz)}
    assert d["max_abs_diff"] > 0, "the fine-tuned forecast equals the stock forecast: the weights did not apply"
    return d


def vs_stock(tuned_npz, stock_npz) -> dict:
    a = np.load(tuned_npz)
    return vs_stock_arrays(a["P"], a["starts"], stock_npz)


def files(folder) -> dict:
    """{relative path: SHA-256} of every file under folder (empty if it does not exist)."""
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()} if folder.exists() else {}


def cache_files(cache) -> set:
    return {str(p.relative_to(cache)) for p in Path(cache).rglob("*") if p.is_file()} if Path(cache).exists() else set()


def main(stage):
    work_before, cache_before = files(R.WORK / stage), cache_files(R.CACHE)
    assert work_before, f"the stock work folder {R.WORK / stage} is missing: its sound features would not be found in the cache"
    R.main(stage)
    added = sorted(cache_files(R.CACHE) - cache_before)
    assert files(R.WORK / stage) == work_before, "prepare_dir wrote different bytes into the stock work folder"
    assert not added, f"{len(added)} files added to the stock feature cache: features were computed, not read"
    f = R.OUT / f"stage_{stage}.json"
    res = json.load(open(f))
    assert WEIGHTS, "the tuned loader did not run"
    res["weights"] = dict(WEIGHTS)
    res["features"] = {"sound_and_text": "read from the stock runs' feature cache (no file added)", "work_folder_unchanged": True}
    res["vs_stock"] = vs_stock(R.OUT / f"pred_{stage}.npz", STOCK_OUT / f"pred_{stage}.npz")
    json.dump(res, open(f, "w"), indent=1)
    print(json.dumps({"stage": stage, "weights": res["weights"], "vs_stock": res["vs_stock"]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
