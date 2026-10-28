"""Face edits of the Tears of Steel bridge scene (exploratory; docs/rules/emofilm_bridge_rule.md and glimmer_face_edit_rule.md), shared by
scripts/make_glimmer_edit_frames.py (2 frames per second, for the Glimmer check) and scripts/make_bridge_edit_cuts.py (every frame, for
the video and forecasts). Masks: SAM 3.1 (Meta Model API, concept "face") run on the 640x360 bridge cut, box-local lossless payloads
decoded with meta-sam-parser (Meta's SAM License; a dependency, not included here). Versions at the release's own 1280x534 resolution:
- intact: the release frame;
- faces_removed: every face mask mapped to release pixels (nearest neighbour, then dilated by DILATE px to cover the 2x upscale) and
  filled with the local mean colour (the mean of a RING px band around the mask, other faces left out of the mean);
- area_matched: each face mask's exact shape moved to a non-face region of the same frame (the first offset in OFFSETS whose box,
  padded by GAP px, misses every face box and every earlier placement and stays inside the frame), filled the same way.
Cut frame c is release frame 600 + c (the cut starts at release 25.0 s at 24 fps; checked against the release pixels)."""
import cv2
import numpy as np

FPS, CUT_START_FRAME = 24, 600
CUT_W, CUT_H, ACT_Y0, ACT_H = 640, 360, 46, 268     # the cut letterboxes the release to 640x360; active picture 640x268 at y 46
REL_W, REL_H = 1280, 534
DILATE, RING, GAP = 2, 6, 20
OFFSETS = [(0, 150), (0, -150), (150, 0), (-150, 0), (150, 150), (-150, 150), (150, -150), (-150, -150),
           (300, 0), (-300, 0), (0, 250), (0, -250), (300, 150), (-300, 150), (300, -150), (-300, -150)]
VERSIONS = ("intact", "faces_removed", "area_matched")


def masks_for(row):
    """Release-resolution boolean masks, one per SAM object of one cut frame (a row of faces_api.jsonl)."""
    import meta_sam_parser as M
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


def edit(frame, masks):
    """The three versions of one release frame, and per face its masked pixels and area-matched offset."""
    faces = np.zeros((REL_H, REL_W), bool)
    for m in masks:
        faces |= m
    img = {v: frame.copy() for v in VERSIONS}
    for m in masks:
        fill(img["faces_removed"], m, faces & ~m)
    occupied, offs = [box(m, GAP) for m in masks], []
    for m in masks:
        moved, off = place(m, occupied); occupied.append(box(moved, GAP)); offs.append(off)
        fill(img["area_matched"], moved, faces)
    return img, [int(m.sum()) for m in masks], offs


def release_frames(path, wanted):
    """Yields (release frame index, BGR frame) for the wanted indices, in order, decoding from the start."""
    cap, n, last = cv2.VideoCapture(path), 0, max(wanted)
    while n <= last:
        ok, f = cap.read()
        if not ok:
            raise SystemExit("release ended early")
        if n in wanted:
            yield n, f
        n += 1
