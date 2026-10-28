"""Builds the known-turns two-reader clip for validating Voice Transcribe diarisation (StillHere Pilot, exploratory): six turns of about
10 s alternating between two credited LibriVox readers of the same public-domain recording project (Alice's Adventures in Wonderland,
chapter 1 read by Kristen McQuillin = reader A, chapter 2 read by Brad Bush = reader B; archive.org item alice_in_wonderland_librivox).
Each turn is cut at a word gap (local faster-whisper base.en timings, offline) near 10 s, consecutive within each reader's excerpt; turns are
joined with the 10 ms equal-power crossfade of stillhere.conditions. The true turn table (reader, start_ms, end_ms) is exact by
construction and written next to the wav.
usage: HF_HUB_OFFLINE=1 PYTHONPATH=.:<faster-whisper site-packages> python3.11 scripts/build_diarization_clip.py CH01_MP3 CH02_MP3"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from stillhere import conditions as C

SR = 16000
OUT = Path(__file__).resolve().parents[1] / "assets" / "diarization"
READERS = {"A": ("Kristen McQuillin", "wonderland_ch_01_64kb.mp3", 100.0), "B": ("Brad Bush", "wonderland_ch_02_64kb.mp3", 20.0)}
TURN_S, TURNS = 10.0, ("A", "B", "A", "B", "A", "B")


def excerpt(mp3: str, start: float, dur: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ss", str(start), "-t", str(dur), "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", str(f)], check=True)
        return C.read_wav(f)[0]


def word_gaps(x: np.ndarray) -> list:
    """Cut candidates (s): midpoints of the gaps between consecutive words, local faster-whisper only."""
    from faster_whisper import WhisperModel
    m = WhisperModel("Systran/faster-whisper-base.en", device="cpu", compute_type="int8", local_files_only=True)
    segs, _ = m.transcribe(x.astype(np.float32), language="en", word_timestamps=True)
    w = [(v.start, v.end) for s in segs for v in s.words]
    return [(a[1] + b[0]) / 2 for a, b in zip(w[:-1], w[1:])]


def main(ch01: str, ch02: str):
    src = {"A": ch01, "B": ch02}; pieces = {}
    for r, (name, _, start) in READERS.items():
        x = excerpt(src[r], start, TURN_S * 3 + 8); gaps = word_gaps(x); cuts = [0.0]
        for _ in range(3):
            cuts.append(min(gaps, key=lambda g: abs(g - (cuts[-1] + TURN_S))))
        pieces[r] = [x[int(round(a * SR)):int(round(b * SR))] for a, b in zip(cuts[:-1], cuts[1:])]
    seq = [pieces[r].pop(0) for r in TURNS]; h = int(round(C.XFADE_S * SR / 2)); L = 2 * h; fin, fout = C._window(L)
    out, table = seq[0].copy(), []
    table.append({"reader": TURNS[0], "start_ms": 0, "end_ms": round(len(seq[0]) / SR * 1000)})
    for r, p in zip(TURNS[1:], seq[1:]):
        out = np.concatenate([out[:-L], out[-L:] * fout + p[:L] * fin, p[L:]])     # 10 ms crossfade; the boundary is its centre
        start = len(out) - len(p); table[-1]["end_ms"] = round((start + h) / SR * 1000)
        table.append({"reader": r, "start_ms": round((start + h) / SR * 1000), "end_ms": round(len(out) / SR * 1000)})
    OUT.mkdir(parents=True, exist_ok=True)
    sha = C.write_wav(OUT / "two_readers_6turns.wav", out, SR)
    truth = {"label": "exploratory", "file": "two_readers_6turns.wav", "sha256": sha, "sr": SR, "duration_ms": round(len(out) / SR * 1000),
             "readers": {r: {"name": v[0], "source": "https://archive.org/download/alice_in_wonderland_librivox/" + v[1], "excerpt_start_s": v[2]}
                         for r, v in READERS.items()},
             "licence": "LibriVox recordings are in the public domain", "turns": table}
    (OUT / "two_readers_6turns.truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    print(sha, truth["duration_ms"], "ms;", [(x["reader"], x["start_ms"], x["end_ms"]) for x in table])


if __name__ == "__main__":
    main(*sys.argv[1:3])
