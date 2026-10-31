"""R3 (cloud ruling of Oct 31, 08:45 UTC; docs/rules/r3_variants_rule.md): LPP French section 1 forecasts, stock and fine-tuned,
made on the laptop in one environment so the confirmation compares like with like. English section 1 is not rerun: D3's stock and
tuned forecasts (one Kaggle environment, Oct 26) are used as they are.

The harness is the Alice runner's (alice/forecast/run_alice_forecast.py, imported unchanged): k2/tribe_common.py at tribev2 af58661,
stitched 50 s windows, text lane in fp16 through accelerate, the sound extractor and TRIBE's brain model on the CPU and Llama 3.2's
layers on a fixed 3 GiB GPU budget. Words are the LPP French TextGrid, as section 3's French forecasts and D3's kernel read it
(tribe_common.textgrid_words). The fine-tuned weights load through R1's loader (alice/forecast/run_alice_forecast_tuned.py
apply_tuned: both pins, strict, eval). Features are cached in this script's own folder. The sound extractor's cache key is the path of
the stage's stim.wav, so the stock and fine-tuned stages of one mode share one work folder, and a fine-tuned stage reads every sound
and Llama feature its stock stage computed (checked: no file added to the cache, work folder unchanged; the fine-tune changed only
TRIBE's brain model). Each stock stage records its agreement with section 3's Kaggle forecast of the same mode (k2, Oct 21).
Writes pred_<stage>.npz and stage_<stage>.json to OUT; prints counts, shapes and timings, never word text.
usage: python run_lpp1_forecast.py STAGE   (STAGE: fr_stock_audio | fr_stock_full | fr_tuned_audio | fr_tuned_full; in this order)"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "alice/forecast"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import run_alice_forecast as R  # noqa: E402
import run_alice_forecast_tuned as RT  # noqa: E402

C = R.C
P1 = R.ROOT / "p1/data"
WAV, WAV_SHA = P1 / "task-lppFR_section_1.wav", "8d2f6ddb876347f3bfabf0aa26174fb320c0b29b96f72b339bb9759abd582f71"
TG, TG_SHA = P1 / "lppFR_section1.TextGrid", "1784c62a0e7bf87541c4768a9a4f579e829dae9323f1de670d725c8dfd30bc37"
K2 = R.ROOT / "k2/out/k2_out"
OUT = R.ROOT / "lpp_variants/forecast/out"; WORK = R.ROOT / "lpp_variants/forecast/tmp"
R.CACHE = R.ROOT / "lpp_variants/forecast/cache"
STAGES = {"fr_stock_audio": ("stock", "audio"), "fr_stock_full": ("stock", "full"),
          "fr_tuned_audio": ("tuned", "audio"), "fr_tuned_full": ("tuned", "full")}


def agreement(P, starts, ref_npz) -> dict:
    """Forecast against forecast: the largest absolute difference and the correlation over the seconds both cover."""
    ref = np.load(ref_npz)
    common, ia, ib = np.intersect1d(starts, ref["starts"], return_indices=True)
    a, b = P[ia].astype("float64"), ref["P"][ib].astype("float64")
    return {"ref": str(ref_npz.relative_to(R.ROOT)), "seconds_compared": int(len(common)), "max_abs_diff": float(np.abs(a - b).max()),
            "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1])}


def main(stage):
    model, mode = STAGES[stage]
    assert hashlib.sha256(WAV.read_bytes()).hexdigest() == WAV_SHA, "stimulus bytes differ from the pinned SHA-256"
    assert hashlib.sha256(TG.read_bytes()).hexdigest() == TG_SHA, "TextGrid bytes differ from the pinned SHA-256"
    OUT.mkdir(parents=True, exist_ok=True); R.CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    words = C.textgrid_words(str(TG))
    work = WORK / f"fr_{mode}"                                  # shared by the stock and fine-tuned stages of this mode
    if model == "tuned":
        work_before, cache_before = RT.files(work), RT.cache_files(R.CACHE)
        assert work_before, f"run fr_stock_{mode} first: {work} is missing"
    d, n = C.prepare_dir(work, WAV, words)
    feats = {"full": ["text", "audio", "video"], "audio": ["audio"]}[mode]
    ev = C.build_events(d, "french", with_text=("text" in feats))
    res = {"stage": stage, "model": model, "mode": mode, "n_tsv_words": n, "wav_sha256": WAV_SHA, "textgrid_sha256": TG_SHA}
    after60 = sum(1 for w in words if w[1] > 60)
    res.update(C.event_checks(ev, n, after60) if "text" in feats else {"audio_end_s": float(ev[ev.type == "Audio"].stop.max())})
    R.PLACEMENT.clear(); R.pin_llama()
    m = RT._stock_loader(feats)
    if model == "tuned":
        res["weights"] = RT.apply_tuned(m)
    starts, P, nwin = C.stitched_predict(m, ev, res["audio_end_s"])
    res.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),
                "contiguous_seconds": bool((np.diff(starts) == 1).all()), **C.gpu_stats(), "seconds": round(time.time() - t0),
                "placement": {"audio_extractor": "cpu", "brain_model": "cpu", **R.PLACEMENT}})
    if "text" in feats and not R.PLACEMENT:
        res["placement"]["text_features"] = "read from this script's feature cache"
    assert res["finite"] and res["contiguous_seconds"], "forecast not finite or not one row per second"
    if model == "stock":
        res["vs_section3_kaggle"] = agreement(P, starts, K2 / f"pred_fr_{mode}.npz")
    else:
        added = sorted(RT.cache_files(R.CACHE) - cache_before)
        assert RT.files(work) == work_before, "prepare_dir wrote different bytes into the stock stage's work folder"
        assert not added, f"{len(added)} files added to the feature cache: features were computed, not read"
        res["features"] = {"sound_and_text": "read from the stock stage's cache entries (no file added)", "work_folder_unchanged": True}
        res["vs_stock"] = RT.vs_stock_arrays(P, starts, OUT / f"pred_fr_stock_{mode}.npz")
    f = OUT / f"pred_{stage}.npz"
    np.savez_compressed(f, P=P.astype("float32"), starts=starts)
    res["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    json.dump(res, open(OUT / f"stage_{stage}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("weights",)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
