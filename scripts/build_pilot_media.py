"""Builds demo/media/, the small binaries the Pilot page (scripts/build_pilot_page.py) shows, once, with ffmpeg; the page builder itself
needs only the standard library and the committed files, so a clean clone builds the page without ffmpeg or anything outside the repo.

- Audio: the four LibriVox condition clips (assets/conditions/alice_ch01_75s_*.wav, 75 s), as mono 32 kbit/s MP3 (plays on phones).
- Film frames: the final v2 edit-check frames (intact, faces filled, same-size control) at two instants, chosen by a fixed rule, not by
  their answers: the bridge frame the video uses (bridge_t032.5.jpg) and the first frame of the "new" set in film order. Each source
  frame's SHA-256 must match the frames' MANIFEST.sha256, whose own SHA-256 must equal results/edit_check_spark_open.json's
  frames_manifest_sha256. Resized to 640 px wide, JPEG, metadata stripped. Tears of Steel (Blender Foundation, CC BY 3.0).
Writes demo/media/MEDIA.json: each output's SHA-256, size and source (with the source's SHA-256).
usage: python scripts/build_pilot_media.py [--frames DIR]"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo/media"
DEFAULT_FRAMES = REPO.parent / "stillhere/pilot/probes/k1e_editcheck/edit_check_frames"   # final v2 edit-check frames
CONDITIONS = ("intact", "sentence_scramble", "word_scramble", "reversed")
VERSIONS = ("intact", "faces_removed", "area_matched")
VIDEO_FRAME = "bridge_t032.5.jpg"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def frame_names(frames: Path) -> list:
    new = sorted(json.loads((frames / "frames.json").read_text())["sets"]["new"], key=lambda f: f["cut_frame"])
    return [VIDEO_FRAME, new[0]["name"]]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", default=str(DEFAULT_FRAMES)); a = ap.parse_args(argv)
    frames = Path(a.frames)
    want = json.loads((REPO / "results/edit_check_spark_open.json").read_text())["frames_manifest_sha256"]
    assert sha(frames / "MANIFEST.sha256") == want, "the frames are not the ones the edit check read"
    manifest = dict(reversed(ln.split("  ", 1)) for ln in (frames / "MANIFEST.sha256").read_text().splitlines() if ln.strip())
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    audio = [(REPO / f"assets/conditions/alice_ch01_75s_{c}.wav", f"alice_75s_{c}.mp3") for c in CONDITIONS]
    for src, name in audio:
        ffmpeg("-i", str(src), "-map_metadata", "-1", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "32k", str(OUT / name))
        items.append({"file": name, "source": str(src.relative_to(REPO)), "source_sha256": sha(src)})
    for fname in frame_names(frames):
        for v in VERSIONS:
            src = frames / v / fname; rel = f"{v}/{fname}"
            assert sha(src) == manifest[rel], f"{rel} differs from MANIFEST.sha256"
            name = f"{Path(fname).stem}_{v}.jpg"
            ffmpeg("-i", str(src), "-map_metadata", "-1", "-vf", "scale=640:-2", "-q:v", "5", str(OUT / name))
            items.append({"file": name, "source": f"edit_check_frames/{rel}", "source_sha256": manifest[rel], "frame": fname, "version": v})
    for it in items:
        p = OUT / it["file"]; it.update(sha256=sha(p), bytes=p.stat().st_size)
    meta = {"what": "media for demo/index.html, built by scripts/build_pilot_media.py", "frames_manifest_sha256": want,
            "credit": "Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study.",
            "audio_source": "LibriVox, Alice's Adventures in Wonderland, chapter 1, read by Kristen McQuillin (public domain)", "items": items}
    (OUT / "MEDIA.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(items)} files, {sum(i['bytes'] for i in items) / 1024:.0f} KB in {OUT.relative_to(REPO)}")
    return meta


if __name__ == "__main__":
    main()
