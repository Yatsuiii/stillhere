"""TRIBE forecasts for the order-map go/no-go (docs/rules/order_map_gonogo.md; exploratory, forecast only, no brain data), on this
laptop's GPU. Stock TRIBE v2 (facebook/tribev2 best.ckpt, checked against the stock pin), text and audio lanes on, video lane off (no
video). The text lane reads each condition's word table (<wav>.tsv, written by scripts/order_map.py prepare) in place of TRIBE's own
speech-to-text, through TRIBE's own word, sentence and context transforms; reversed has no words, so no word reaches the text lane.
Every wav must be on the public-audio allowlist (egress.PublicAudio). One 40 s clip is one TRIBE window (100 TRs), predicted with
TRIBE's built-in predict. Writes RUN_DIR/<clip>/pred_<condition>.npz (P: seconds x 20484, starts) and run_<condition>.json (extractors
that ran, Llama 3.2 precision and placement, words reaching the text lane, timings, peak VRAM). Prints counts and timings only.
usage: HF_HOME=<.hf> HF_TOKEN_PATH=<token file> PYTHONPATH=.:<tribe site-packages> python3.11 scripts/order_map_run.py RUN_DIR CACHE_DIR
       [--text-device accelerate|cuda] [--only CLIP]"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stillhere import egress, live_tribe  # noqa: E402

FEATURES = ["text", "audio"]


def build_events(wav: Path, with_text: bool):
    """TRIBE's audio-to-events pipeline in the order the Kaggle runs used (words first, then chunking)."""
    import pandas as pd
    from neuralset.events.transforms import AddContextToWords, AddSentenceToWords, AddText, ChunkEvents, RemoveMissing
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio
    ev = standardize_events(pd.DataFrame([{"type": "Audio", "filepath": str(wav), "start": 0, "timeline": "default", "subject": "default"}]))
    ts = []
    if with_text:
        ts += [ExtractWordsFromAudio(language="english"), AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),
               AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""), RemoveMissing()]
    ts += [ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30)]
    for t in ts:
        ev = t(ev)
    return standardize_events(ev)


def text_lane_info(m) -> dict:
    """What the text extractor actually loaded: model, dtype, and where its layers sit."""
    ex = m.data.text_feature; mod = getattr(ex, "_model", None)
    if mod is None:
        return {"model": ex.model_name, "loaded": False}
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in mod.parameters()})
    dmap = getattr(mod, "hf_device_map", None)
    place = ({"gpu_modules": sum(1 for v in dmap.values() if str(v) not in ("cpu", "disk")),
              "cpu_modules": sum(1 for v in dmap.values() if str(v) == "cpu"), "disk_modules": sum(1 for v in dmap.values() if str(v) == "disk")}
             if dmap else {"device": str(next(mod.parameters()).device)})
    q = getattr(getattr(mod, "config", None), "quantization_config", None)
    return {"model": ex.model_name, "loaded": True, "dtypes": dtypes, "quantization": str(q.quant_method) if q else None,
            "extractor_device": ex.device, "placement": place}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("cache_dir")
    ap.add_argument("--text-device", default="accelerate"); ap.add_argument("--only", default=None); a = ap.parse_args()
    import torch
    from huggingface_hub import hf_hub_download
    from tribev2.demo_utils import TribeModel
    run = Path(a.run_dir); man = json.loads((run / "prepare_manifest.json").read_text())
    ckpt = live_tribe.verify_ckpt(hf_hub_download("facebook/tribev2", "best.ckpt"))
    t0 = time.time()
    m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(a.cache_dir),
                                   config_update={"data.num_workers": 0, "data.features_to_use": FEATURES,
                                                  "data.text_feature.device": a.text_device})
    load_s = round(time.time() - t0, 2)
    jobs = [(clip, cond, f) for clip, mc in man["clips"].items() if not a.only or clip == a.only for cond, f in mc["files"].items()]
    events, ev_s = {}, {}
    for clip, cond, f in jobs:
        wav = run / clip / f"{cond}.wav"
        egress.PublicAudio(path=str(wav))                                       # refuses anything not on the public-audio allowlist
        t1 = time.time(); events[clip, cond] = build_events(wav, with_text=f["text_lane_words"] > 0); ev_s[clip, cond] = round(time.time() - t1, 2)
    # Memory only (the 6 GB GPU): the text lane's features for every condition first, cached on disk by TRIBE's own feature cache;
    # then Llama 3.2 is released before the audio lane and the brain model run. The features are the same either way.
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    for k, ev in events.items():
        if (ev.type == "Word").any():
            m.data.text_feature.prepare(ev)
    text_info = text_lane_info(m); text_info["features_s"] = round(time.time() - t1, 2)
    text_info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    m.data.text_feature.__pydantic_private__.pop("_model", None); gc.collect(); torch.cuda.empty_cache()
    print(f"text lane: {text_info}", flush=True)
    for clip, cond, f in jobs:
        ev = events[clip, cond]
        torch.cuda.reset_peak_memory_stats(); t2 = time.time()
        P, segs = m.predict(events=ev, verbose=False)
        t3 = time.time()
        P = np.asarray(P, np.float32); starts = np.array([s.start for s in segs], float)
        if P.ndim != 2 or P.shape[1] != live_tribe.N_VERTICES or not np.isfinite(P).all():
            raise ValueError(f"{clip} {cond}: unexpected prediction {P.shape}")
        np.savez_compressed(run / clip / f"pred_{cond}.npz", P=P, starts=starts)
        info = {"clip": clip, "condition": cond, "wav_sha256": f["sha256"], "ckpt_sha256": ckpt, "features_to_use": FEATURES,
                "video_lane": "off (audio only, no video events)",
                "audio_lane": {"model": m.data.audio_feature.model_name, "device": m.data.audio_feature.device},
                "text_lane": text_info, "text_lane_device_setting": a.text_device,
                "words_in_table": f["text_lane_words"], "word_events_reaching_text_lane": int((ev.type == "Word").sum()),
                "seconds_predicted": int(P.shape[0]), "first_start_s": float(starts[0]), "model_load_s": load_s,
                "events_s": ev_s[clip, cond], "predict_s": round(t3 - t2, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "device": torch.cuda.get_device_name(0),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (run / clip / f"run_{cond}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(f"{clip} {cond}: {P.shape[0]} s predicted, words {info['word_events_reaching_text_lane']}, events {info['events_s']} s, "
              f"predict {info['predict_s']} s, peak VRAM {info['peak_vram_gb']} GB, text lane {info['text_lane'].get('dtypes')} "
              f"{info['text_lane'].get('placement')}", flush=True)


if __name__ == "__main__":
    main()
