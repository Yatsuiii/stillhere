"""Alice Datasets (ds002322) chapter 1: the full-chapter stock TRIBE v2 forecasts, before any Alice BOLD is read (cloud ruling of
Oct 30, 16:10 UTC, item a). Section 8's harness unchanged: k2/tribe_common.py at tribev2 af58661, the same three runs as en8 (audio,
full, and full with the word texts shuffled across their onsets, seed 7), stitched 50 s windows, text lane in fp16 through accelerate.
Input: the dataset's own stimulus (what the listeners heard; SHA-256 pinned below) and the dataset's own word timings
(code/annotations.tsv): rows without a timing are dropped and counted; a duration below 0.01 s (one row ends before it starts, one
has zero length) is set to 0.01 s. Laptop placement only, as the live worker (worker/engine.py) does on the same 6 GB GPU: the sound
extractor and TRIBE's brain model run on the CPU, and Llama 3.2's fp16 layers get a fixed GPU budget (the worker's pinned loader,
LLAMA_GPU_GIB) with the rest on the CPU, which leaves room for the 1,024-token contexts; the split is recorded with each run.
Writes pred_alice_{audio,full,shuf}.npz and stage_*.json to OUT and prints counts, shapes and timings, never word text.
usage: python run_alice_forecast.py STAGE   (STAGE: alice_audio | alice_full | alice_shuf)"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the analysis root: k2/tribe_common.py is kept outside the repository
sys.path.insert(0, str(ROOT / "k2"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tribe_common as C  # noqa: E402

A = ROOT / "alice"
WAV = A / "stimulus/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"
WAV_SHA = "bb8003578d0667cd45fbce932c45dcfa3abee4be40da98c0a83dea80516aeefe"
ANN = A / "annotations/annotations.tsv"
OUT = A / "forecast/out"; CACHE = A / "forecast/cache"; WORK = A / "forecast/tmp"
MIN_DUR = 0.01
LLAMA_GPU_GIB = 3.0
PLACEMENT = {}


def pin_llama():
    """worker/engine.py install_llama_loader's pinned branch: accelerate splits Llama 3.2 by a fixed GPU budget, not by free memory."""
    from neuralset.extractors import text as nst
    orig = nst.HuggingFaceText._load_model

    def _load(self_, **kw):
        if self_.device != "accelerate":
            return orig(self_, **kw)
        if self_.pretrained is not True:
            raise RuntimeError("pinned loader handles the pretrained model only")
        from transformers import AutoModel
        m = AutoModel.from_pretrained(self_.model_name, device_map="auto", torch_dtype=torch.float16,
                                      max_memory={0: f"{int(LLAMA_GPU_GIB * 1024)}MiB", "cpu": "24GiB"})
        dm = getattr(m, "hf_device_map", None) or {}
        PLACEMENT.update({"llama_gpu_budget_gib": LLAMA_GPU_GIB, "gpu_modules": sum(1 for v in dm.values() if str(v) not in ("cpu", "disk")),
                          "cpu_modules": sum(1 for v in dm.values() if str(v) == "cpu")})
        return m
    nst.HuggingFaceText._load_model = _load


def words():
    rows = list(csv.DictReader(open(ANN), delimiter="\t"))
    timed = [r for r in rows if r["Word Onset"].strip() and r["Word Offset"].strip()]
    w = [(r["Words"].strip(), float(r["Word Onset"]), max(float(r["Word Offset"]) - float(r["Word Onset"]), MIN_DUR)) for r in timed]
    return w, {"rows": len(rows), "timed": len(timed), "dropped_untimed": len(rows) - len(timed),
               "set_to_min_duration": sum(1 for r in timed if float(r["Word Offset"]) - float(r["Word Onset"]) < MIN_DUR),
               "annotations_sha256": hashlib.sha256(ANN.read_bytes()).hexdigest()}


def load_model_cpu_brain(feats):
    """C.load_model's config, unchanged, plus the brain model on the CPU (a from_pretrained argument, not a config key)."""
    from tribev2.demo_utils import TribeModel
    cu = {"data.num_workers": 0, "data.features_to_use": feats, "data.text_feature.device": "accelerate",
          "data.overlap_trs_train": 50, "data.audio_feature.device": "cpu"}
    return TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE), config_update=cu, device="cpu")


def main(stage):
    assert hashlib.sha256(WAV.read_bytes()).hexdigest() == WAV_SHA, "stimulus bytes differ from the pinned SHA-256"
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    mode, shuffle = {"alice_audio": ("audio", None), "alice_full": ("full", None), "alice_shuf": ("full", 7)}[stage]
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    w, wstats = words()
    d, n = C.prepare_dir(WORK / stage, WAV, w, shuffle_seed=shuffle)
    feats = {"full": ["text", "audio", "video"], "audio": ["audio"]}[mode]
    ev = C.build_events(d, "english", with_text=("text" in feats))
    res = {"stage": stage, "mode": mode, "shuffle_seed": shuffle, "n_tsv_words": n, **wstats}
    after60 = sum(1 for x in w if x[1] > 60)
    res.update(C.event_checks(ev, n, after60) if "text" in feats else {"audio_end_s": float(ev[ev.type == "Audio"].stop.max())})
    pin_llama()
    m = load_model_cpu_brain(feats)
    starts, P, nwin = C.stitched_predict(m, ev, res["audio_end_s"])
    res.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),
                "contiguous_seconds": bool((np.diff(starts) == 1).all()), **C.gpu_stats(), "seconds": round(time.time() - t0),
                "placement": {"audio_extractor": "cpu", "brain_model": "cpu", **PLACEMENT}})
    if "text" in feats and not PLACEMENT:   # Llama not loaded: every word's features came from the infra cache an earlier pinned-loader run wrote
        res["placement"]["text_features"] = "read from the infra cache"
    f = OUT / f"pred_{stage}.npz"
    np.savez_compressed(f, P=P.astype("float32"), starts=starts)
    res["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    json.dump(res, open(OUT / f"stage_{stage}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "annotations_sha256"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
