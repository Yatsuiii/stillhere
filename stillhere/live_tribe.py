"""Live laptop TRIBE run for the demo (PLAN §7, STORY-PICKER demo beat): stock TRIBE v2, audio-only, on one short public clip.

Takes only an egress.PublicAudio (a clip whose SHA-256 is on the public-audio allowlist; any other file, including every confidential
clip, is refused before anything is read), at most MAX_SECONDS long (a 20 s clip fits the 6 GB laptop GPU; 60 s windows do not).
Verifies the checkpoint against the stock pin before use. Returns tier-D numbers only: timings, peak VRAM, seconds predicted, and
the predicted AUD response strength, defined as the mean over the Schaefer-400/17 auditory parcels of the SD over time of the
parcel-mean audio-only prediction (raw TRIBE units). That is TRIBE's predicted response strength for hearing, a forecast; it says
nothing about language or understanding (PREREG §9), and the LANGUAGE forecast is precomputed on Kaggle, never live."""
import hashlib
import time
import wave
from pathlib import Path

import numpy as np

from . import egress

LABEL = "live, audio-only, stock TRIBE v2, one public clip; a forecast, not validated"
WORDING = "TRIBE's predicted response strength (hearing, audio-only). A forecast, not validated; it says nothing about language or understanding."
UNITS = "raw TRIBE units: mean over the auditory parcels of the SD over time of the parcel-mean audio-only prediction"
STOCK_CKPT_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"     # facebook/tribev2 best.ckpt, as on Kaggle
MAX_SECONDS = 25.0
N_VERTICES = 20484                                                                        # fsaverage5, left hemisphere first
HEMI_OFFSET = 10242


def clip_seconds(path: str) -> float:
    """Duration of a PCM WAV file from its header (stdlib only)."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_ckpt(path) -> str:
    got = sha256_file(path)
    if got != STOCK_CKPT_SHA256:
        raise egress.EgressRefused(f"checkpoint is not the stock TRIBE v2 pin: {got[:12]}...")
    return got


def aud_parcels(atlas_dir) -> list:
    """Schaefer-400/17 auditory parcels (SomMotB_Aud) as vertex-index arrays in TRIBE order (left hemisphere first)."""
    import nibabel.freesurfer as fs
    out = []
    for hemi, off in (("lh", 0), ("rh", HEMI_OFFSET)):
        lab, _, names = fs.read_annot(str(Path(atlas_dir) / f"{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"))
        out += [np.where(lab == i)[0] + off for i, n in enumerate(names) if "SomMotB_Aud" in n.decode()]
    return out


def response_strength(P: np.ndarray, parcels: list) -> float:
    """Mean over parcels of the SD over time of the parcel-mean prediction (P: seconds x vertices, raw units)."""
    return float(np.mean([P[:, v].mean(1).std() for v in parcels]))


def _tribe_predict(wav: str, cache_dir: str):
    """Stock TRIBE v2, audio features only, built-in predict (a clip this short is one window). Returns (P, info)."""
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from neuralset.events.transforms import ChunkEvents
    from neuralset.events.utils import standardize_events
    from tribev2.demo_utils import TribeModel
    ckpt = verify_ckpt(hf_hub_download("facebook/tribev2", "best.ckpt"))
    ev = standardize_events(pd.DataFrame([{"type": "Audio", "filepath": wav, "start": 0, "timeline": "default", "subject": "default"}]))
    ev = standardize_events(ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30)(ev))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir,
                                       config_update={"data.num_workers": 0, "data.features_to_use": ["audio"]})
    t1 = time.time()
    P, _ = model.predict(events=ev, verbose=False)
    t2 = time.time()
    info = {"ckpt_sha256": ckpt, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
            "load_s": round(t1 - t0, 2), "predict_s": round(t2 - t1, 2)}
    return np.asarray(P, dtype=np.float32), info


def run(audio: egress.PublicAudio, atlas_dir, cache_dir, predict=None, parcels=None) -> dict:
    """One live run. predict(wav, cache_dir) -> (P, info) is injectable for tests; the default is stock TRIBE v2 on this machine."""
    if not isinstance(audio, egress.PublicAudio):
        raise egress.EgressRefused(f"only an allowlisted egress.PublicAudio is run, got {type(audio).__name__}")
    secs = clip_seconds(audio.path)
    if secs > MAX_SECONDS:
        raise egress.EgressRefused(f"clip is {secs:.1f} s; the live run takes at most {MAX_SECONDS:.0f} s")
    t0 = time.time()
    P, info = (predict or _tribe_predict)(audio.path, str(cache_dir))
    if P.ndim != 2 or P.shape[1] != N_VERTICES or not np.isfinite(P).all():
        raise ValueError(f"unexpected prediction shape or values: {P.shape}")
    parc = parcels if parcels is not None else aud_parcels(atlas_dir)
    return {"label": LABEL, "model": "stock TRIBE v2 (facebook/tribev2), audio features only", **info,
            "audio_sha256": sha256_file(audio.path), "clip_seconds": round(secs, 2), "seconds_predicted": int(P.shape[0]),
            "predicted_aud_response_strength": round(response_strength(P, parc), 6), "units": UNITS, "n_aud_parcels": len(parc),
            "wording": WORDING, "total_s": round(time.time() - t0, 2), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
