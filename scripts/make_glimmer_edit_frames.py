"""Builds the edited-frame set for the Glimmer face-edit check (docs/rules/glimmer_face_edit_rule.md, exploratory): the bridge scene of
Tears of Steel (Blender Foundation 2012, CC BY 3.0), release t = 25.0, 25.5, ..., 40.0 s (31 frames), in three versions at the release's
own 1280x534 resolution:
- intact: the release frame;
- faces_removed: every SAM 3.1 face mask (Meta Model API run on the 640x360 bridge cut, pilot/probes/sam31/out_bridge_api), mapped to
  release pixels (nearest neighbour, then dilated by DILATE px to cover the 2x upscale), filled with the local mean colour (the mean of a
  RING px band around the mask, other faces excluded);
- area_matched: per frame, each face mask's exact shape moved to a non-face region of the same frame (first free offset in OFFSETS whose
  box, padded by GAP px, misses every face box and every earlier placement and stays inside the frame), filled the same way.
Every version is written by the same JPEG encoder at the same quality. Writes glimmer_edit_frames/{intact,faces_removed,area_matched}/,
frames.json (instants, face counts, masked areas, offsets) and MANIFEST.sha256. Prints counts only.
usage: python make_glimmer_edit_frames.py RELEASE_MOV FACES_API_JSONL SAMPARSER_SITE OUT_DIR"""
import hashlib
import json
import os
import sys

import cv2
import numpy as np

REL, FACES, PARSER, OUT = sys.argv[1:5]
sys.path.insert(0, PARSER)
import meta_sam_parser as M  # noqa: E402

FPS, T0, N = 24, 25.0, 31                      # release fps; bridge cut starts at release 25.0 s (cut frame 0 = release frame 600)
CUT_W, CUT_H, ACT_Y0, ACT_H = 640, 360, 46, 268  # cut: release letterboxed to 640x360, active picture 640x268 at y 46
REL_W, REL_H = 1280, 534
DILATE, RING, GAP, QUALITY = 2, 6, 20, 95
OFFSETS = [(0, 150), (0, -150), (150, 0), (-150, 0), (150, 150), (-150, 150), (150, -150), (-150, -150),
           (300, 0), (-300, 0), (0, 250), (0, -250), (300, 150), (-300, 150), (300, -150), (-300, -150)]
VERSIONS = ("intact", "faces_removed", "area_matched")


def masks_for(row):
    """Release-resolution boolean masks, one per SAM object, from the box-local lossless payloads."""
    out = []
    for o in row["objects"]:
        m = o["mask"]
        r = M.decode_mask_to_raster(M.SegmentationMask(encoding="lossless", payload=m["enc"] + m["payload"], width=m["W"], height=m["H"]))
        local = np.frombuffer(r, np.uint8).reshape(m["H"], m["W"])
        x0, y0 = o["box_xyxy"][:2]
        cut = np.zeros((CUT_H, CUT_W), np.uint8)
        cut[y0:y0 + m["H"], x0:x0 + m["W"]] = local[:CUT_H - y0, :CUT_W - x0]
        rel = cv2.resize(cut[ACT_Y0:ACT_Y0 + ACT_H], (REL_W, REL_H), interpolation=cv2.INTER_NEAREST)
        out.append(cv2.dilate(rel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1,) * 2)).astype(bool))
    return out


def fill(img, mask, exclude):
    """Fills mask with the mean colour of the RING px band around it (pixels in exclude are left out of the mean)."""
    band = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1,) * 2)).astype(bool) & ~mask & ~exclude
    img[mask] = img[band].mean(axis=0).round().astype(np.uint8)


def box(m, pad=0):
    ys, xs = np.nonzero(m)
    return xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad


def hits(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def place(m, occupied):
    """The first offset that keeps the shape inside the frame and its padded box clear of every face and earlier placement."""
    x0, y0, x1, y1 = box(m)
    for dx, dy in OFFSETS:
        b = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if b[0] < 0 or b[1] < 0 or b[2] >= REL_W or b[3] >= REL_H:
            continue
        pb = (b[0] - GAP, b[1] - GAP, b[2] + GAP, b[3] + GAP)
        if not any(hits(pb, t) for t in occupied):
            moved = np.zeros_like(m)
            ys, xs = np.nonzero(m); moved[ys + dy, xs + dx] = True
            return moved, (dx, dy)
    raise SystemExit("no free non-face region for a mask")


rows = [json.loads(line) for line in open(FACES)]
want = {12 * k: 600 + 12 * k for k in range(N)}                  # cut frame -> release frame (2 per second)
cap, rel, n = cv2.VideoCapture(REL), {}, 0
while n <= max(want.values()):
    ok, f = cap.read()
    if not ok:
        raise SystemExit("release ended early")
    if n in want.values():
        rel[n] = f
    n += 1
for v in VERSIONS:
    os.makedirs(f"{OUT}/{v}", exist_ok=True)
frames = []
for c, r in want.items():
    t = round(T0 + c / FPS, 2); name = f"t{t:05.1f}.jpg"
    row = rows[c]; assert row["frame"] == c
    ms = masks_for(row); faces = np.zeros((REL_H, REL_W), bool)
    for m in ms:
        faces |= m
    img = {v: rel[r].copy() for v in VERSIONS}
    for m in ms:
        fill(img["faces_removed"], m, faces & ~m)
    occupied, offs = [box(m, GAP) for m in ms], []
    for m in ms:
        moved, off = place(m, occupied); occupied.append(box(moved, GAP)); offs.append(off)
        fill(img["area_matched"], moved, faces)
    for v in VERSIONS:
        cv2.imwrite(f"{OUT}/{v}/{name}", img[v], [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    frames.append({"name": name, "t_release": t, "release_frame": r, "sam_faces": len(ms),
                   "masked_px": [int(m.sum()) for m in ms], "area_matched_offsets": offs})
src = hashlib.sha256(open(REL, "rb").read()).hexdigest()
json.dump({"source": os.path.basename(REL), "source_sha256": src, "fps": 2, "resolution": [REL_W, REL_H], "jpeg_quality": QUALITY,
           "dilate_px": DILATE, "ring_px": RING, "gap_px": GAP, "sam": "sam-3.1, Meta Model API, concept face, bridge cut 640x360",
           "frames": frames}, open(f"{OUT}/frames.json", "w"), indent=1)
files = sorted(f"{v}/{f['name']}" for v in VERSIONS for f in frames) + ["frames.json"]
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:
    for p in files:
        fh.write(f"{hashlib.sha256(open(f'{OUT}/{p}', 'rb').read()).hexdigest()}  {p}\n")
print(f"frames {len(frames)} per version; faces per frame {sorted({f['sam_faces'] for f in frames})}; "
      f"masked px per face {min(min(f['masked_px']) for f in frames)}-{max(max(f['masked_px']) for f in frames)}; files {len(files)}")
