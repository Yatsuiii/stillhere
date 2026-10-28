"""Live demo (PLAN §7): stock TRIBE v2, audio-only, on the allowlisted 20 s LibriVox clip, on this laptop's GPU. Prints one line of
timings and the readout; writes results/live_tribe_librivox.json (tier-D numbers only). Run it with the TRIBE environment and every
cache on the Windows partition, e.g.
  W=<your work dir holding .venvs/tribe and .hf>
  HF_HOME=$W/.hf TORCH_HOME=$W/.torch XDG_CACHE_HOME=$W/.cache PYTHONPATH=.:$W/.venvs/tribe/lib/python3.11/site-packages \\
    ~/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 scripts/live_tribe_demo.py"""
import argparse
import json
import os
from pathlib import Path

from stillhere import egress, live_tribe

W = Path(os.environ.get("STILLHERE_WORK", "."))                    # holds .venvs/tribe and the .hf model cache
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--audio", default="assets/librivox_alice_ch01_20s.wav")
ap.add_argument("--atlas", default=os.environ.get("STILLHERE_ATLAS", str(W / "stillhere/atlas")))
ap.add_argument("--cache", default=os.environ.get("STILLHERE_LIVE_CACHE", str(W / ".cache/stillhere_live")))
ap.add_argument("--out", default="results/live_tribe_librivox.json")
a = ap.parse_args()
r = live_tribe.run(egress.PublicAudio(path=a.audio), a.atlas, a.cache)
Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
print(f"live TRIBE: {r['clip_seconds']} s clip, {r['seconds_predicted']} s predicted on {r['device']} | load {r['load_s']} s, predict "
      f"{r['predict_s']} s, total {r['total_s']} s, peak VRAM {r['peak_vram_gb']} GB | predicted AUD response strength "
      f"{r['predicted_aud_response_strength']} ({r['n_aud_parcels']} parcels) | {r['label']}")
