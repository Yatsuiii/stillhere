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
