"""Live demo, not validated: stock TRIBE v2, audio-only, on this laptop's GPU, for one LibriVox story chapter (the 75 s Alice ch. 1
excerpt in assets/conditions) intact and sentence-scrambled. The live run takes at most 25 s of audio, so each condition is cut into
four back-to-back 18.75 s windows (wave module, so the bytes and their SHA-256 are the same on every run); every window must be on the
public-audio allowlist (--cut writes the windows and prints their allowlist lines). Prints one line per window; writes
results/demo_live_tribe_conditions.json: timings, peak VRAM, shapes, the predicted AUD response strength, and the per-second
auditory-parcel time course (small; derived from public audio only). Hearing only: it says nothing about language or understanding.
Run with the TRIBE environment and every cache on the Windows partition (see scripts/live_tribe_demo.py).
usage: python scripts/live_tribe_conditions_demo.py [--cut]"""
import argparse
import gc
import json
import os
import time
import wave
from pathlib import Path

from stillhere import egress, live_tribe

ROOT = Path(__file__).resolve().parent.parent
W = Path(os.environ.get("STILLHERE_WORK", "."))
CONDITIONS = ("intact", "sentence_scramble")
N_WIN, LABEL = 4, "demo, not validated"
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--cut", action="store_true")
ap.add_argument("--atlas", default=os.environ.get("STILLHERE_ATLAS", str(W / "stillhere/atlas")))
ap.add_argument("--cache", default=os.environ.get("STILLHERE_LIVE_CACHE", str(W / ".cache/stillhere_live")))
ap.add_argument("--out", default=str(ROOT / "results/demo_live_tribe_conditions.json"))
a = ap.parse_args()
WIN_DIR = Path(a.cache) / "demo_windows"


def windows(cond: str) -> list:
    """Cuts the condition into N_WIN equal windows (idempotent) and returns (path, start s, end s)."""
    src = ROOT / f"assets/conditions/alice_ch01_75s_{cond}.wav"
    with wave.open(str(src), "rb") as w:
        p, frames = w.getparams(), w.readframes(w.getnframes())
    n = p.nframes // N_WIN; WIN_DIR.mkdir(parents=True, exist_ok=True); out = []
    for i in range(N_WIN):
        dst = WIN_DIR / f"alice_ch01_75s_{cond}_w{i + 1}of{N_WIN}.wav"
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(p.nchannels); w.setsampwidth(p.sampwidth); w.setframerate(p.framerate)
            w.writeframes(frames[i * n * p.sampwidth * p.nchannels:(i + 1) * n * p.sampwidth * p.nchannels])
        out.append((dst, round(i * n / p.framerate, 2), round((i + 1) * n / p.framerate, 2)))
    return out


if a.cut:
    for cond in CONDITIONS:
        for dst, s0, s1 in windows(cond):
            print(f"{live_tribe.sha256_file(dst)}  demo window, condition {cond.replace('_', ' ')}, {s0}-{s1} s of "
                  f"assets/conditions/alice_ch01_75s_{cond}.wav (cut by scripts/live_tribe_conditions_demo.py)")
    raise SystemExit

import numpy as np  # noqa: E402
import torch  # noqa: E402

parcels = live_tribe.aud_parcels(a.atlas); runs = []; t0 = time.time()
print(f"Live TRIBE v2 (stock, audio only) on {torch.cuda.get_device_name(0)}", flush=True)
print(f"LibriVox, Alice ch. 1 (75 s), intact and sentence-scrambled, 4 windows each ({LABEL})", flush=True)
for cond in CONDITIONS:
    for dst, s0, s1 in windows(cond):
        held = {}

        def predict(wav, cache_dir):
            P, info = live_tribe._tribe_predict(wav, cache_dir); held["P"] = P
            return P, info

        r = live_tribe.run(egress.PublicAudio(path=str(dst)), a.atlas, a.cache, predict=predict)
        P = held.pop("P")
        r.update(condition=cond, window_s=[s0, s1], prediction_shape=list(P.shape),
                 aud_parcel_timecourse=np.round(np.stack([P[:, v].mean(1) for v in parcels], 1), 5).tolist())
        runs.append(r)
        print(f"  {cond:<17} {s0:>5.2f}-{s1:<5.2f} s | load {r['load_s']:>5.2f} s, predict {r['predict_s']:>5.2f} s | VRAM "
              f"{r['peak_vram_gb']} GB | {P.shape[0]} x {P.shape[1]} | AUD {r['predicted_aud_response_strength']:.4f}", flush=True)
        del P; gc.collect(); torch.cuda.empty_cache()
mean = {c: round(float(np.mean([r["predicted_aud_response_strength"] for r in runs if r["condition"] == c])), 6) for c in CONDITIONS}
out = {"label": LABEL, "what": "live laptop run, stock TRIBE v2 audio-only, one LibriVox chapter excerpt, intact and sentence-scrambled",
       "source": "assets/conditions/alice_ch01_75s_{intact,sentence_scramble}.wav (LibriVox, public domain)", "windows_per_condition": N_WIN,
       "wording": live_tribe.WORDING, "units": live_tribe.UNITS,
       "timecourse_note": "aud_parcel_timecourse: seconds x auditory parcels, parcel-mean prediction, raw TRIBE units",
       "mean_aud_response_strength": mean, "total_s": round(time.time() - t0, 2), "runs": runs}
Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
print(f"mean predicted AUD strength: intact {mean['intact']:.4f}, sentence-scrambled {mean['sentence_scramble']:.4f}", flush=True)
print(f"{len(runs)} windows in {out['total_s']:.0f} s | a forecast for hearing only, not validated", flush=True)
