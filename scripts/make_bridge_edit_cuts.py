"""Builds the three bridge-scene cuts of Tears of Steel (Blender Foundation 2012, CC BY 3.0; release 25.00 to 40.25 s, 366 frames at
24 fps) in the versions of stillhere/face_edit.py: intact, faces removed, area matched, for the video and the forecasts with no BOLD
(docs/rules/emofilm_bridge_rule.md; exploratory). Every frame is edited at the release's 1280x534 resolution, and all three cuts are
encoded the same way (H.264 CRF 18, yuv420p) with the release's own audio for that window (AAC 192 kb/s). Writes
{intact,faces_removed,area_matched}_25p00_40p25.mp4, edits.json (per frame: faces, masked pixels, area-matched offsets) and
MANIFEST.sha256. Prints counts only.
usage: python make_bridge_edit_cuts.py RELEASE_MOV FACES_API_JSONL SAMPARSER_SITE OUT_DIR"""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

REL, FACES, PARSER, OUT = sys.argv[1:5]
sys.path[:0] = [PARSER, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from stillhere import face_edit as E  # noqa: E402

N, START_S = 366, 25.0
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(line) for line in open(FACES)]
assert len(rows) == N and [r["frame"] for r in rows] == list(range(N))
names = {v: f"{v}_25p00_40p25.mp4" for v in E.VERSIONS}
enc = {v: subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{E.REL_W}x{E.REL_H}",
                            "-r", str(E.FPS), "-i", "-", "-ss", f"{START_S}", "-t", f"{N / E.FPS}", "-i", REL, "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", f"{OUT}/{names[v]}"], stdin=subprocess.PIPE) for v in E.VERSIONS}
per_frame = []
for r, f in E.release_frames(REL, set(range(E.CUT_START_FRAME, E.CUT_START_FRAME + N))):
    c = r - E.CUT_START_FRAME
    img, px, offs = E.edit(f, E.masks_for(rows[c]))
    for v in E.VERSIONS:
        enc[v].stdin.write(img[v].tobytes())
    per_frame.append({"cut_frame": c, "release_frame": r, "sam_faces": len(px), "masked_px": px, "area_matched_offsets": offs})
for v in E.VERSIONS:
    enc[v].stdin.close()
    assert enc[v].wait() == 0, f"ffmpeg failed for {v}"
offsets = Counter(tuple(o) for f in per_frame for o in f["area_matched_offsets"])
json.dump({"source": os.path.basename(REL), "source_sha256": hashlib.sha256(open(REL, "rb").read()).hexdigest(), "window_release_s": [25.0, 40.25],
           "frames": N, "fps": E.FPS, "resolution": [E.REL_W, E.REL_H], "video": "H.264 CRF 18 yuv420p", "audio": "release audio, AAC 192 kb/s",
           "dilate_px": E.DILATE, "ring_px": E.RING, "gap_px": E.GAP, "sam": "sam-3.1, Meta Model API, concept face, bridge cut 640x360",
           "offset_counts": {str(k): n for k, n in offsets.items()}, "per_frame": per_frame}, open(f"{OUT}/edits.json", "w"), indent=1)
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:
    for p in sorted(names.values()) + ["edits.json"]:
        fh.write(f"{hashlib.sha256(open(f'{OUT}/{p}', 'rb').read()).hexdigest()}  {p}\n")
print(f"frames {len(per_frame)}; faces per frame {sorted(Counter(f['sam_faces'] for f in per_frame).items())}; "
      f"area-matched offsets {dict(offsets)}; masked px per face {min(min(f['masked_px']) for f in per_frame)}-{max(max(f['masked_px']) for f in per_frame)}")
