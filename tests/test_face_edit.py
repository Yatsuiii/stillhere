"""stillhere/face_edit.py on synthetic frames (no film, no SAM call): faces removed are filled with one local-mean colour, the area-matched
control moves the exact mask shape to a free spot and leaves the faces untouched, and a mask with no free spot is refused."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from stillhere import face_edit as E  # noqa: E402


def frame_and_masks():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (E.REL_H, E.REL_W, 3), dtype=np.uint8)
    masks = []
    for x0 in (400, 800):                                            # two "faces" side by side, near the top like the bridge shot
        m = np.zeros((E.REL_H, E.REL_W), bool); m[100:170, x0:x0 + 60] = True; masks.append(m)
    return f, masks


def test_faces_removed_filled_and_area_matched_moved():
    f, masks = frame_and_masks()
    img, px, offs = E.edit(f, masks)
    assert px == [70 * 60, 70 * 60] and offs == [(0, 150), (0, 150)]
    assert (img["intact"] == f).all()
    for m in masks:
        assert len(np.unique(img["faces_removed"][m], axis=0)) == 1    # one local-mean colour per face
        assert (img["area_matched"][m] == f[m]).all()                  # the control leaves the faces as they are
    moved = np.zeros_like(masks[0]); moved[250:320, 400:460] = True
    assert len(np.unique(img["area_matched"][moved], axis=0)) == 1 and moved.sum() == masks[0].sum()
    outside = ~(masks[0] | masks[1])
    assert (img["faces_removed"][outside] == f[outside]).all()


def test_no_free_region_is_refused():
    m = np.zeros((E.REL_H, E.REL_W), bool); m[10:520, 10:1270] = True
    with pytest.raises(SystemExit):
        E.place(m, [E.box(m, E.GAP)])


def test_search_fallback_places_a_big_face_the_fixed_offsets_cannot():
    m = np.zeros((E.REL_H, E.REL_W), bool); m[150:450, 380:700] = True       # a close-up: every fixed offset leaves the frame or hits it
    with pytest.raises(SystemExit):
        E.place(m, [E.box(m, E.GAP)])                                          # the bridge default: no search, refused as before
    moved, off = E.place(m, [E.box(m, E.GAP)], search=True)
    assert moved.sum() == m.sum() and off not in E.OFFSETS and not (moved & m).any()
    f = np.zeros((E.REL_H, E.REL_W, 3), np.uint8)
    img, px, offs = E.edit(f, [m], skip_unplaceable=True)
    assert offs == [off] and px == [int(m.sum())]


def planes(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, (E.REL_H, E.REL_W), dtype=np.uint8), rng.integers(0, 255, (E.REL_H // 2, E.REL_W // 2), dtype=np.uint8),
            rng.integers(0, 255, (E.REL_H // 2, E.REL_W // 2), dtype=np.uint8))


def face(x0, y0, w, h):
    m = np.zeros((E.REL_H, E.REL_W), bool); m[y0:y0 + h, x0:x0 + w] = True
    return m


def test_yuv_edit_changes_nothing_outside_its_2x2_regions_and_controls_match_the_visible_face():
    Y, U, V = planes()
    masks = [face(501, 101, 51, 61), face(1001, 151, 41, 41)]          # one face in TRIBE's view (odd edges), one outside it
    out, reg, stats = E.edit_yuv(Y, U, V, masks)
    for v in ("faces_removed", "area_matched"):
        y, u, w = out[v]; r = reg[v]
        assert (r.reshape(E.REL_H // 2, 2, E.REL_W // 2, 2).all(axis=(1, 3)) == r.reshape(E.REL_H // 2, 2, E.REL_W // 2, 2).any(axis=(1, 3))).all()
        assert (y[~r] == Y[~r]).all() and (u[~r[::2, ::2]] == U[~r[::2, ::2]]).all() and (w[~r[::2, ::2]] == V[~r[::2, ::2]]).all()
    assert stats[0]["px_in_view"] == stats[0]["px"] and stats[0]["control_px"] == stats[0]["px_in_view"] and stats[0]["pieces"] == 1
    assert stats[1]["px_in_view"] == 0 and stats[1]["control_px"] == 0                 # TRIBE never sees it: no control
    assert not (reg["area_matched"] & ~E.view_mask()).any() and not (reg["area_matched"] & reg["faces_removed"]).any()


def test_a_face_too_big_to_fit_whole_is_split_or_left_without_control():
    Y, U, V = planes(1)
    big = face(404, 120, 300, 250)                                         # most of the view's width: no whole placement fits
    out, reg, stats = E.edit_yuv(Y, U, V, [big])
    s = stats[0]
    assert s["pieces"] in (0, 2, 4, 8)
    if s["pieces"]:
        assert s["control_px"] == s["px_in_view"] and not (reg["area_matched"] & ~E.view_mask()).any()
