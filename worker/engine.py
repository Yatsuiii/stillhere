"""The worker's TRIBE engine: stock TRIBE v2 loaded once, text and audio lanes on, video lane off (audio only), run on one built
recipe at a time. Same model, checkpoint pin, lanes and audio-to-events pipeline as the order-map run (scripts/order_map_run.py):
Llama 3.2 3B in float16 placed by accelerate across GPU and CPU, w2v-BERT 2.0 for sound, and the text lane reading the recipe's moved
word table (<wav>.tsv) in place of TRIBE's own speech-to-text, so no speech-to-text runs here and reversed sends no word to the text
lane. Measured on the 6 GB laptop GPU (docs/worker.md): with TRIBE's in-RAM feature cache off, its text extractor loads Llama 3.2 for
each job (about 5 s from the page cache) and releases it after the text lane, so the GPU is empty between jobs; w2v-BERT and the brain
model stay loaded, on the devices AUDIO_DEVICE and BRAIN_DEVICE name (CPU: Llama's float16 layers fill the GPU while the text lane runs).
Every result records the devices through the model manifest.

Output is region values only: per second of clip time, the mean over the frozen proof's language (LANG) and hearing (AUD) parcels
(worker/regions.json, from stillhere/heldout.py parcel_sets) of the parcel-mean forecast, raw TRIBE units. No vertex data leaves."""
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REGIONS = json.loads((HERE / "regions.json").read_text())
STOCK_CKPT_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"   # as stillhere/live_tribe.py
N_VERTICES = 20484
FEATURES = ["text", "audio"]
AUDIO_DEVICE = os.environ.get("STILLHERE_AUDIO_DEVICE", "cpu")
BRAIN_DEVICE = os.environ.get("STILLHERE_BRAIN_DEVICE", "cpu")
WORKER_VERSION = "1"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_tsv(words: list, path) -> int:
    """<wav>.tsv in the table TRIBE's ExtractWordsFromAudio reads (pandas read_csv, tab separated) instead of running its own
    speech-to-text. Written with csv quoting, so a word that starts with a double quote cannot open a quoted field and swallow the
    rows after it (unquoted, a clip with quoted dialogue silently lost half its words); check_tsv proves the round trip."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["text", "start", "duration", "sequence_id", "sentence"])
        for x in words:
            w.writerow([x["word"], f"{x['start']:.4f}", f"{x['end'] - x['start']:.4f}", 0, ""])
    return len(words)


def check_tsv(words: list, path) -> None:
    """Reads the table back exactly as TRIBE does and refuses to go on unless every word comes back unchanged and in order."""
    import pandas as pd
    t = pd.read_csv(path, sep="\t")
    if [str(v) for v in t["text"]] != [x["word"] for x in words]:
        raise ValueError("word table does not read back unchanged")


def build_events(wav: Path, with_text: bool):
    """TRIBE's audio-to-events pipeline in the order the Kaggle and order-map runs used (words first, then chunking)."""
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


def region_series(P: np.ndarray) -> dict:
    """{"language": [...], "hearing": [...]}: per second, the mean over parcels of the parcel-mean forecast (raw units)."""
    out = {}
    for name, key in (("language", "LANG"), ("hearing", "AUD")):
        s = np.mean(np.stack([P[:, v].mean(1) for v in REGIONS[key]], 1), 1)
        out[name] = [round(float(v), 6) for v in s]
    return out


class Engine:
    """Load once (load), then run(built, workdir, emit) per job. emit(stage, lane, status, detail) posts a real stage event."""

    def __init__(self, cache_dir, audio_device=AUDIO_DEVICE, brain_device=BRAIN_DEVICE):
        self.cache_dir, self.audio_device, self.brain_device = Path(cache_dir), audio_device, brain_device
        self.m = None; self.ckpt = None

    def load(self):
        import torch
        from huggingface_hub import hf_hub_download
        from tribev2.demo_utils import TribeModel
        ckpt = hf_hub_download("facebook/tribev2", "best.ckpt")
        got = sha256_file(ckpt)
        if got != STOCK_CKPT_SHA256:
            raise RuntimeError("checkpoint is not the stock TRIBE v2 pin")
        self.ckpt = got; t0 = time.time()
        self.m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(self.cache_dir), device=self.brain_device,
                                            config_update={"data.num_workers": 0, "data.features_to_use": FEATURES,
                                                           "data.text_feature.device": "accelerate",
                                                           "data.audio_feature.device": self.audio_device,
                                                           # no in-RAM feature cache: it keys audio by file path, and RAM would grow
                                                           "data.audio_feature.infra.keep_in_ram": False,
                                                           "data.text_feature.infra.keep_in_ram": False})
        self.load_s = round(time.time() - t0, 2)
        from worker import recipe as R
        self.manifest = R.load_index()["models"]
        want = {"w2v_bert": self.m.data.audio_feature.model_name, "llama": self.m.data.text_feature.model_name,
                "audio_lane_device": self.audio_device, "brain_device": self.brain_device}
        for k, v in want.items():
            if self.manifest[k].split("@")[0] != v:
                raise RuntimeError(f"model manifest differs from the loaded engine: {k}")
        if self.ckpt not in self.manifest["tribe"]:
            raise RuntimeError("model manifest differs from the loaded engine: tribe")
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return self

    def models(self) -> dict:
        """The model manifest this engine runs, exactly data/live_clips.json "models" (part of every cache key); load() refuses to
        start if they differ."""
        return self.manifest

    def run(self, built, workdir: Path, emit) -> dict:
        import torch
        from worker import recipe as R
        m = self.m; t = {}; workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        before = set(self.cache_dir.rglob("*")) if self.cache_dir.exists() else set()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time(); emit("build", None, "start", {})
        # content-addressed path: TRIBE's feature cache keys audio by file path, so one path must never hold two different builds
        wav = workdir / built.wav_sha256[:20] / "condition.wav"; wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(R.wav_bytes(built.audio, built.sr))
        n_words = write_tsv(built.words, wav.with_suffix(".tsv")) if built.words else 0
        if n_words:
            check_tsv(built.words, wav.with_suffix(".tsv"))
        ev = build_events(wav, with_text=n_words > 0)
        t["build"] = round(time.time() - t0, 2); emit("build", None, "end", {})
        emit("video_lane", "v-jepa-2", "off", {})
        t1 = time.time(); emit("text_lane", "llama-3.2", "start", {"n_words": n_words, "precision": "float16"})
        if (ev.type == "Word").any():
            m.data.text_feature.prepare(ev)
        t["text_lane"] = round(time.time() - t1, 2); emit("text_lane", "llama-3.2", "end", {"n_words": int((ev.type == "Word").sum())})
        t2 = time.time(); emit("audio_lane", "w2v-bert", "start", {"device": self.audio_device})
        m.data.audio_feature.prepare(ev)
        t["audio_lane"] = round(time.time() - t2, 2); emit("audio_lane", "w2v-bert", "end", {})
        t3 = time.time(); emit("predict", None, "start", {"device": self.brain_device})
        P, segs = m.predict(events=ev, verbose=False)
        P = np.asarray(P, np.float32); t["predict"] = round(time.time() - t3, 2); emit("predict", None, "end", {})
        if P.ndim != 2 or P.shape[1] != N_VERTICES or not np.isfinite(P).all():
            raise ValueError("unexpected prediction shape or values")
        emit("regions", None, "start", {})
        seconds = int(round(len(built.audio) / built.sr))
        reg = {k: v[:seconds] for k, v in region_series(P).items()}
        emit("regions", None, "end", {})
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        for p in sorted(set(self.cache_dir.rglob("*")) - before, key=lambda p: len(p.parts), reverse=True):  # drop this job's features
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        shutil.rmtree(workdir, ignore_errors=True); gc.collect()
        t["total"] = round(time.time() - t0, 2)
        return {"tr_s": 1.0, "seconds": len(reg["language"]), "regions": reg, "word_rate": R.word_rate(built.words, len(reg["language"])),
                "n_words_text_lane": int((ev.type == "Word").sum()), "input_sha256": built.wav_sha256, "models": self.models(),
                "gpu": self.gpu, "peak_vram_gb": peak, "timings_s": t,
                "first_start_s": float(segs[0].start) if len(segs) else 0.0, "worker_version": WORKER_VERSION,
                "label": "Forecast, not a scan: TRIBE v2 on the laptop, audio only, no brain data",
                "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
