"""The live worker's clips (cloud plan v2, item 5): 40 s from 60 s of each of the 12 chapters of the LibriVox recording of Alice's
Adventures in Wonderland (archive.org item alice_in_wonderland_librivox, public domain), cut and transcribed exactly as the order-map
go/no-go did (scripts/order_map.py cut_clip and whisper_words: ffmpeg bitexact, local faster-whisper base.en word timings), so
alice_ch01-03 reproduce stillhere/live_menu.json's intact SHA-256s. Writes data/live_clips/<id>.wav (the intact condition, written by
stillhere/conditions.write_wav), data/live_clips/<id>.words.json and the clips part of data/live_clips.json; also prints each chapter's
spoken preamble (first 30 s) so the reader can be named. Every source MP3 must match archive.org's MD5 (files.json beside them).
usage: HF_HUB_OFFLINE=1 HF_HUB_CACHE=<faster-whisper cache> PYTHONPATH=.:<asr site-packages> python3.11 worker/make_clips.py SRC_DIR TMP_DIR"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stillhere import conditions as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("order_map", REPO / "scripts/order_map.py")
OM = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(OM)
OUT = REPO / "data/live_clips"
ITEM = "alice_in_wonderland_librivox"
# Readers per chapter from the LibriVox catalog (librivox.org/api/feed/audiobooks, book 200, sections 1-12).
READERS = ["Kristen McQuillin", "Brad Bush", "Roger W. Barnett", "miette", "Mark Bradford", "Raza Shah", "Kara Shallenberg",
           "Kristen McQuillin", "MarinaMechanical", "Roger W. Barnett", "R. Francis Smith", "Chris Goringe"]


def clean_words(words: list, seconds: float) -> list:
    """Two fixes to faster-whisper's table, nothing else: zero-duration words at a clip edge (start within 10 ms of 0, or within 50 ms of
    the end) are dropped, since they are not timed inside the clip (and a cut that close to an edge breaks the intact identity of
    stillhere/conditions.assemble); the em dash character in a word is written as a hyphen (the repository bans that character)."""
    out = []
    for w in words:
        if w["end"] <= w["start"] and (w["start"] <= 0.01 or w["start"] >= seconds - 0.05):
            continue
        out.append({**w, "word": w["word"].replace("\u2014", "-")})
    return out


def md5_ok(mp3: Path, files: dict) -> bool:
    return hashlib.md5(mp3.read_bytes()).hexdigest() == files[mp3.name]


def main(src: Path, tmp: Path):
    f = json.loads((src / "files.json").read_text()); f = f if isinstance(f, list) else f.get("result", f.get("files", []))
    files = {x["name"]: x.get("md5") for x in f}
    OUT.mkdir(parents=True, exist_ok=True); tmp.mkdir(parents=True, exist_ok=True); clips = []
    for ch in range(1, 13):
        cid, mp3 = f"alice_ch{ch:02d}", src / f"wonderland_ch_{ch:02d}_64kb.mp3"
        assert md5_ok(mp3, files), f"{mp3.name}: MD5 differs from archive.org"
        cut = tmp / f"{cid}_source.wav"; OM.cut_clip(mp3, cut)
        x, sr = C.read_wav(cut)
        words = clean_words(OM.whisper_words(cut), len(x) / sr)
        wav_sha = C.write_wav(OUT / f"{cid}.wav", x, sr)                     # the intact condition equals the cut sample for sample
        wtxt = json.dumps({"words": words}, indent=1) + "\n"; (OUT / f"{cid}.words.json").write_text(wtxt)
        pre = tmp / f"{cid}_preamble.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-t", "30", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(pre)],
                       check=True)
        print(f"{cid}: {len(words)} words, {len(C.sentences(words))} sentences; preamble: "
              f"{' '.join(w['word'] for w in OM.whisper_words(pre))}", flush=True)
        clips.append({"id": cid, "title": f"Alice's Adventures in Wonderland, chapter {ch}", "reader": READERS[ch - 1],
                      "licence": "public domain (LibriVox)", "source": f"https://archive.org/download/{ITEM}/{mp3.name}", "start_s": OM.START_S, "dur_s": OM.DUR_S,
                      "seconds": round(len(x) / sr, 3), "sr": sr, "wav": f"data/live_clips/{cid}.wav", "wav_sha256": wav_sha,
                      "words": f"data/live_clips/{cid}.words.json", "words_sha256": hashlib.sha256(wtxt.encode()).hexdigest(),
                      "n_words": len(words)})
    idx = REPO / "data/live_clips.json"
    cur = json.loads(idx.read_text()) if idx.exists() else {}
    cur["clips"] = clips
    idx.write_text(json.dumps(cur, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
