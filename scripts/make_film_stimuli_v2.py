"""Film stimuli v2 (docs/rules/film_stimulus_rule.md; exploratory): the one allowed re-render of the Tears of Steel face edits for the
film-rem and film-am forecasts. The intact cut is SEG (release 0 to 190 s, built with the intact kernel's own command: ffmpeg -ss 0 -t 190
-c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k); the intact kernel reads this exact file. Each frame of SEG is decoded to its
yuv420p planes, edited with stillhere/face_edit.py's edit_yuv (SAM 3.1 masks, 2x2-snapped; faces removed, or each face's in-view part moved
inside TRIBE's view as the control), and encoded losslessly (libx264 -qp 0, yuv420p) with SEG's audio stream copied unchanged. So every
pixel outside an edit decodes as in SEG, and the audio is bit-identical. Writes faces_removed_v2.mp4, area_matched_v2.mp4, regions.npz
(per frame, both edit regions, bit-packed), edits_v2.json and MANIFEST.sha256 (with SEG). Prints counts only.
usage: python make_film_stimuli_v2.py SEG FACES_API_JSONL SAMPARSER_SITE OUT_DIR"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

SEG, FACES, PARSER, OUT = sys.argv[1:5]
sys.path[:0] = [PARSER, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from stillhere import face_edit as E  # noqa: E402

W, H, N = E.REL_W, E.REL_H, 4560
FS = W * H * 3 // 2
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(line) for line in open(FACES)]
assert len(rows) == N and [r["frame"] for r in rows] == list(range(N))
dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", SEG, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)
names = {"faces_removed": "faces_removed_v2.mp4", "area_matched": "area_matched_v2.mp4"}
enc = {v: subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-r", str(E.FPS),
                            "-i", "-", "-i", SEG, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", f"{OUT}/{n}"], stdin=subprocess.PIPE) for v, n in names.items()}
regions = {v: [] for v in names}; per_frame = []
for c in range(N):
    buf = dec.stdout.read(FS)
    assert len(buf) == FS, f"segment ended at frame {c}"
    f = np.frombuffer(buf, np.uint8)
    Y, U, V = f[:W * H].reshape(H, W), f[W * H:W * H + FS // 6].reshape(H // 2, W // 2), f[W * H + FS // 6:].reshape(H // 2, W // 2)
    out, reg, stats = E.edit_yuv(Y, U, V, E.masks_for(rows[c]))
    for v in names:
        y, u, w = out[v]; enc[v].stdin.write(y.tobytes() + u.tobytes() + w.tobytes())
        regions[v].append(np.packbits(reg[v]))
    per_frame.append({"frame": c, "faces": stats})
assert dec.stdout.read(1) == b"", "segment has more frames than 4560"
dec.wait()
for v in names:
    enc[v].stdin.close(); assert enc[v].wait() == 0, f"ffmpeg failed for {v}"
np.savez_compressed(f"{OUT}/regions.npz", **{v: np.stack(r) for v, r in regions.items()}, shape=np.array([N, H, W]))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
faces = [x for f in per_frame for x in f["faces"]]
vis = [x for x in faces if x["px_in_view"]]
summary = {"frames": N, "face_instances": len(faces), "in_view": len(vis), "controls_placed": sum(1 for x in vis if x["pieces"]), "split": sum(1 for x in vis if x["pieces"] > 1),
           "in_view_face_px": sum(x["px_in_view"] for x in vis), "control_px": sum(x["control_px"] for x in vis)}
json.dump({"seg": os.path.basename(SEG), "seg_sha256": sha(SEG), "view": E.VIEW, "dilate_px": E.DILATE, "ring_px": E.RING, "gap_px": E.GAP,
           "video": "libx264 -qp 0 (lossless) yuv420p; audio stream copied from the segment", "summary": summary, "per_frame": per_frame},
          open(f"{OUT}/edits_v2.json", "w"), indent=0)
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:
    for p in [SEG] + [f"{OUT}/{n}" for n in names.values()] + [f"{OUT}/regions.npz", f"{OUT}/edits_v2.json"]:
        fh.write(f"{sha(p)}  {os.path.basename(p)}\n")
print("summary", summary, "| control px / in-view face px", round(summary["control_px"] / max(summary["in_view_face_px"], 1), 4))
