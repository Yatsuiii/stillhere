"""Face edits of the Tears of Steel bridge scene (exploratory; docs/rules/emofilm_bridge_rule.md and glimmer_face_edit_rule.md), shared by
scripts/make_glimmer_edit_frames.py (2 frames per second, for the Glimmer check) and scripts/make_bridge_edit_cuts.py (every frame, for
the video and forecasts). Masks: SAM 3.1 (Meta Model API, concept "face") run on the 640x360 bridge cut, box-local lossless payloads
decoded with meta-sam-parser (Meta's SAM License; a dependency, not included here). Versions at the release's own 1280x534 resolution:
- intact: the release frame;
- faces_removed: every face mask mapped to release pixels (nearest neighbour, then dilated by DILATE px to cover the 2x upscale) and
  filled with the local mean colour (the mean of a RING px band around the mask, other faces left out of the mean);
- area_matched: each face mask's exact shape moved to a non-face region of the same frame (the first offset in OFFSETS whose box,
  padded by GAP px, misses every face box and every earlier placement and stays inside the frame; for other windows than the
  bridge, then the nearest such spot on a 20 px grid), filled the same way.
Cut frame c is release frame 600 + c (the cut starts at release 25.0 s at 24 fps; checked against the release pixels)."""
from functools import lru_cache

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


@lru_cache(maxsize=None)
def _grid():
    """Fallback offsets for a shape the fixed OFFSETS cannot place: every 20 px step over the frame, nearest first."""
    return tuple(sorted(((dx, dy) for dx in range(-REL_W, REL_W + 1, 20) for dy in range(-REL_H, REL_H + 1, 20) if (dx, dy) != (0, 0)),
                  key=lambda o: (o[0] ** 2 + o[1] ** 2, o)))


def place(m, occupied, search=False):
    """The first offset that keeps the shape inside the frame and its padded box clear of every face and earlier placement; with
    search, the nearest free spot on a 20 px grid is tried after the fixed OFFSETS."""
    x0, y0, x1, y1 = box(m)
    for dx, dy in OFFSETS + (list(_grid()) if search else []):
        b = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        if b[0] < 0 or b[1] < 0 or b[2] >= REL_W or b[3] >= REL_H:
            continue
        pb = (b[0] - GAP, b[1] - GAP, b[2] + GAP, b[3] + GAP)
        if not any(hits(pb, t) for t in occupied):
            moved = np.zeros_like(m)
            ys, xs = np.nonzero(m); moved[ys + dy, xs + dx] = True
            return moved, (dx, dy)
    raise SystemExit("no free non-face region for a mask")


def edit(frame, masks, skip_unplaceable=False):
    """The three versions of one release frame, and per face its masked pixels and area-matched offset. With skip_unplaceable, a face
    whose shape fits none of the fixed offsets is searched for on a 20 px grid, and one that fits nowhere free gets no control patch
    (its offset is None) instead of stopping the run."""
    faces = np.zeros((REL_H, REL_W), bool)
    for m in masks:
        faces |= m
    img = {v: frame.copy() for v in VERSIONS}
    for m in masks:
        fill(img["faces_removed"], m, faces & ~m)
    occupied, offs = [box(m, GAP) for m in masks], []
    for m in masks:
        try:
            moved, off = place(m, occupied, search=skip_unplaceable)
        except SystemExit:
            if not skip_unplaceable:
                raise
            offs.append(None); continue
        occupied.append(box(moved, GAP)); offs.append(off)
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


# ---- film stimuli v2 (docs/rules/film_stimulus_rule.md) --------------------------------------------------------------------------------
# Edits are made on the intact segment's own decoded yuv420p planes and encoded losslessly, so every pixel outside an edit decodes
# exactly as in the intact segment. Every edit region is snapped to the 2x2 chroma grid. A face's control patch is the part of the
# face that TRIBE sees (inside VIEW), moved to the nearest free spot inside VIEW (exact pixel test, VIEW_GAP px from faces and other
# patches), cut into 2, 4 or 8 equal pieces only when it fits nowhere whole; face removal and control then change the same number of
# pixels that reach the model, except for faces whose in-view part fits nowhere even in 8 pieces (counted).
VIEW = (404, 32, 873, 501)        # inclusive x0, y0, x1, y1: the release pixels V-JEPA2's preprocessing (shortest edge 292, centre crop 256)
#                                   depends on, found by perturbing single rows and columns through the processor itself


def view_mask():
    v = np.zeros((REL_H, REL_W), bool); v[VIEW[1]:VIEW[3] + 1, VIEW[0]:VIEW[2] + 1] = True
    return v


def snap2(mask):
    """Grows a mask to whole 2x2 blocks (the chroma grid of yuv420p)."""
    m = mask.reshape(REL_H // 2, 2, REL_W // 2, 2).any(axis=(1, 3))
    return np.repeat(np.repeat(m, 2, 0), 2, 1)


VIEW_GAP, MAX_PIECES = 4, 8                # control patches keep 4 px from faces and from each other; at most 8 pieces per face


def _place_px(shape_mask, occ):
    """The nearest free top-left (in distance from where the shape is) inside VIEW, on the 2x2 grid, where the shape overlaps no
    occupied pixel; exact pixel test for every position at once (FFT overlap count). The moved mask, or None."""
    from scipy.signal import fftconvolve
    X0, Y0, X1, Y1 = VIEW
    ys, xs = np.nonzero(shape_mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    shape = shape_mask[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    o = occ[Y0:Y1 + 1, X0:X1 + 1].astype(np.float32)
    if shape.shape[0] > o.shape[0] or shape.shape[1] > o.shape[1]:
        return None
    py, px = np.nonzero(fftconvolve(o, shape[::-1, ::-1], mode="valid") < 0.5)
    keep = ((X0 + px) % 2 == 0) & ((Y0 + py) % 2 == 0)
    py, px = py[keep], px[keep]
    if not len(py):
        return None
    d = (X0 + px - x0) ** 2 + (Y0 + py - y0) ** 2
    i = np.lexsort((px, py, d))[0]
    moved = np.zeros_like(shape_mask); moved[ys + int(Y0 + py[i] - y0), xs + int(X0 + px[i] - x0)] = True
    return moved


def _pieces(m, n):
    """n pieces of a shape, cut across its longer side at 2-px-aligned lines of equal pixel count."""
    ys, xs = np.nonzero(m)
    coord = ys if (ys.max() - ys.min()) >= (xs.max() - xs.min()) else xs
    edges = [int(coord.min())] + [int(np.quantile(coord, q)) // 2 * 2 for q in np.linspace(0, 1, n + 1)[1:-1]] + [int(coord.max()) + 1]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (coord >= a) & (coord < b)
        if sel.any():
            p = np.zeros_like(m); p[ys[sel], xs[sel]] = True; out.append(p)
    return out


def place_in_view(m, occ):
    """The control for one face's in-view part: the whole shape at the nearest free spot inside VIEW; failing that, the shape cut into
    2, 4 or 8 equal pieces placed one by one. Returns (moved pixels, number of pieces), or (None, 0) if no split fits."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * VIEW_GAP + 1,) * 2)
    n = 1
    while n <= MAX_PIECES:
        trial, moved = occ.copy(), np.zeros_like(m)
        for p in _pieces(m, n):
            mv = _place_px(p, trial)
            if mv is None:
                moved = None; break
            moved |= mv; trial |= cv2.dilate(mv.astype(np.uint8), k).astype(bool)
        if moved is not None:
            return moved, n
        n *= 2
    return None, 0


def fill_yuv(Y, U, V, region, exclude):
    """Fills a 2x2-aligned region with the local mean of a RING px band around it, per plane (chroma at its own resolution)."""
    band = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1,) * 2)).astype(bool) & ~region & ~exclude
    Y[region] = np.uint8(round(float(Y[band].mean())))
    rc, bc = region[::2, ::2], band[::2, ::2] & ~region[::2, ::2]
    for P in (U, V):
        P[rc] = np.uint8(round(float(P[bc].mean())))


def edit_yuv(Y, U, V, masks):
    """faces_removed and area_matched planes for one intact frame, the union region of each edit, and per face: snapped pixels, pixels
    inside VIEW, control pixels and offset (None when the face is not in VIEW or no free spot exists)."""
    snapped = [snap2(m) for m in masks]
    faces = np.zeros((REL_H, REL_W), bool)
    for s in snapped:
        faces |= s
    out = {"faces_removed": (Y.copy(), U.copy(), V.copy()), "area_matched": (Y.copy(), U.copy(), V.copy())}
    for s in snapped:
        fill_yuv(*out["faces_removed"], s, faces & ~s)
    vm = view_mask(); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * VIEW_GAP + 1,) * 2)
    occ = cv2.dilate(faces.astype(np.uint8), k).astype(bool); ctrl = np.zeros_like(faces); stats = []
    for s in snapped:
        vis = s & vm; rec = {"px": int(s.sum()), "px_in_view": int(vis.sum()), "control_px": 0, "pieces": 0}
        if vis.any():
            moved, n = place_in_view(vis, occ)
            if moved is not None:
                occ |= cv2.dilate(moved.astype(np.uint8), k).astype(bool); ctrl |= moved
                fill_yuv(*out["area_matched"], moved, faces)
                rec.update(control_px=int(moved.sum()), pieces=n)
        stats.append(rec)
    return out, {"faces_removed": faces, "area_matched": ctrl}, stats
