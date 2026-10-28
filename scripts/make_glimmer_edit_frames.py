"""Builds the edited-frame set for the Glimmer face-edit check (docs/rules/glimmer_face_edit_rule.md, exploratory): the bridge scene of
Tears of Steel (Blender Foundation 2012, CC BY 3.0), release t = 25.0, 25.5, ..., 40.0 s (31 frames), in the three versions of
stillhere/face_edit.py (intact, faces removed, area matched), every version written by the same JPEG encoder at the same quality.
Writes glimmer_edit_frames/{intact,faces_removed,area_matched}/, frames.json (instants, face counts, masked areas, offsets) and
MANIFEST.sha256. Prints counts only.
usage: python make_glimmer_edit_frames.py RELEASE_MOV FACES_API_JSONL SAMPARSER_SITE OUT_DIR"""
import hashlib
import json
import os
import sys

import cv2

REL, FACES, PARSER, OUT = sys.argv[1:5]
sys.path[:0] = [PARSER, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from stillhere import face_edit as E  # noqa: E402

N, QUALITY = 31, 95
rows = [json.loads(line) for line in open(FACES)]
want = {E.CUT_START_FRAME + 12 * k: 12 * k for k in range(N)}       # release frame -> cut frame (2 per second)
for v in E.VERSIONS:
    os.makedirs(f"{OUT}/{v}", exist_ok=True)
frames = []
for r, f in E.release_frames(REL, set(want)):
    c = want[r]; t = round(25.0 + c / E.FPS, 2); name = f"t{t:05.1f}.jpg"
    row = rows[c]; assert row["frame"] == c
    img, px, offs = E.edit(f, E.masks_for(row))
    for v in E.VERSIONS:
        cv2.imwrite(f"{OUT}/{v}/{name}", img[v], [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    frames.append({"name": name, "t_release": t, "release_frame": r, "sam_faces": len(px), "masked_px": px, "area_matched_offsets": offs})
src = hashlib.sha256(open(REL, "rb").read()).hexdigest()
json.dump({"source": os.path.basename(REL), "source_sha256": src, "fps": 2, "resolution": [E.REL_W, E.REL_H], "jpeg_quality": QUALITY,
           "dilate_px": E.DILATE, "ring_px": E.RING, "gap_px": E.GAP, "sam": "sam-3.1, Meta Model API, concept face, bridge cut 640x360",
           "frames": frames}, open(f"{OUT}/frames.json", "w"), indent=1)
files = sorted(f"{v}/{f['name']}" for v in E.VERSIONS for f in frames) + ["frames.json"]
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:
    for p in files:
        fh.write(f"{hashlib.sha256(open(f'{OUT}/{p}', 'rb').read()).hexdigest()}  {p}\n")
print(f"frames {len(frames)} per version; faces per frame {sorted({f['sam_faces'] for f in frames})}; "
      f"masked px per face {min(min(f['masked_px']) for f in frames)}-{max(max(f['masked_px']) for f in frames)}; files {len(files)}")
