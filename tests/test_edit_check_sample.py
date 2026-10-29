"""The edit-check sampler on made-up SAM rows (no frames, no network): window and bridge exclusion, the face-size floor, one pick per
contiguous part, determinism under the fixed seed."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ecs", ROOT / "scripts/edit_check_sample.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def rows():
    out = []
    for f in range(4560):
        if f % 3 == 0:
            objs = []
        elif f % 3 == 1:
            objs = [{"box_xywh": [10, 10, 30, 30]}]          # 900 px: counts
        else:
            objs = [{"box_xywh": [10, 10, 10, 10]}]          # 100 px: below the floor
        out.append({"frame": f, "objects": objs})
    return out


def test_window_bridge_floor_and_spread():
    s = S.sample(rows())
    face, nomask = s["face_frames"], s["no_mask_frames"]
    assert len(face) == len(nomask) == 31 and len(set(face)) == 31
    assert all(240 <= f < 4320 and not 600 <= f < 966 and f % 3 == 1 for f in face)
    assert all(240 <= f < 4320 and f % 3 == 0 for f in nomask)
    assert face == sorted(face) and face[0] < 600 and face[-1] > 4000      # one per contiguous part: spread over the window
    assert S.sample(rows()) == s                                            # fixed seed: same picks every time
