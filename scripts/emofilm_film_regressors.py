"""Stimulus-side baselines for the Emo-FilM rule (docs/rules/emofilm_rule.md version 2; exploratory): per-second ENVELOPE, BRIGHTNESS
and MOTION of release 0 to END s of the public Tears of Steel release, and WORD RATE from the forecast kernel's words.csv when given.
No brain data is read. Frames are decoded by ffmpeg at the release's native 24 fps and size and streamed in short chunks; the audio is
decoded to 16-bit PCM at its native rate. Writes OUT/film_regressors.npz (per-second series) and OUT/film_regressors.json (shapes,
counts, input hashes); OUT must be outside this repo. Prints counts, shapes and timings only.
usage: python scripts/emofilm_film_regressors.py --movie tears_of_steel_720p.mov --out DIR [--end 190] [--words words.csv]"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import emofilm_score as E  # noqa: E402
from stillhere import heldout as H  # noqa: E402

TOS_SHA = "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"
FPS, CHUNK = 24, 24


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def frame_series(movie, end: float) -> tuple:
    """Per-frame brightness and motion over release 0 to end s, streamed CHUNK frames at a time; each chunk is scored with the previous
    chunk's last frame prepended, so motion crosses chunk borders exactly as on the whole clip."""
    w, h = [int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                            "-of", "csv=p=0", movie], capture_output=True, text=True, check=True).stdout.strip().split(",")]
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", "0", "-t", str(end), "-i", movie, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    size, bright, motion, prev = w * h * 3, [], [], None
    while True:
        buf = p.stdout.read(size * CHUNK)
        if not buf:
            break
        f = np.frombuffer(buf[: len(buf) // size * size], np.uint8).reshape(-1, h, w, 3)
        b, m = E.frame_stats(f if prev is None else np.concatenate([prev, f]))
        bright.append(b if prev is None else b[1:]); motion.append(m if prev is None else m[1:]); prev = f[-1:]
    assert p.wait() == 0, "ffmpeg failed"
    return np.concatenate(bright), np.concatenate(motion), (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--end", type=float, default=190.0); ap.add_argument("--words")
    a = ap.parse_args()
    assert not os.path.abspath(a.out).startswith(REPO), "OUT must be outside the repo"
    t0 = time.time(); assert sha256(a.movie) == TOS_SHA, "the movie is not the official Tears of Steel release"
    os.makedirs(a.out, exist_ok=True); n = int(a.end)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "0", "-t", str(a.end), "-i", a.movie, "-vn", "-c:a", "pcm_s16le", wav], check=True)
        env = H.envelope_per_second(wav)[:n]
    bright, motion, (w, h) = frame_series(a.movie, a.end)
    per_s = {"envelope": env, "brightness": E.per_second(bright, FPS)[:n], "motion": E.per_second(motion, FPS)[:n]}
    meta = {"label": "exploratory", "rule": "docs/rules/emofilm_rule.md (version 2)", "movie_sha256": TOS_SHA, "release_s": [0.0, a.end],
            "frames": int(len(bright)), "frame_size": [w, h], "fps": FPS}
    if a.words:
        with open(a.words) as f:
            on = [float(r["start"]) for r in csv.DictReader(f)]
        per_s["word_rate"] = E.word_rate_per_s(on, n); meta["words_sha256"] = sha256(a.words); meta["n_words"] = len(on)
    assert all(len(v) == n and np.all(np.isfinite(v)) for v in per_s.values()), "a series is short or not finite"
    np.savez(os.path.join(a.out, "film_regressors.npz"), **per_s)
    meta["series"] = {k: list(v.shape) for k, v in per_s.items()}; meta["seconds"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(a.out, "film_regressors.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("frames", "frame_size", "series", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
