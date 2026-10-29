"""Frame sample for the edit check (docs/rules/edit_check_rule.md; exploratory), chosen from SAM 3.1's output alone, before any frame is
shown to a model. Input: faces_api.jsonl of the SAM run on Tears of Steel release 0-190 s (cut frame c = release frame c, 24 fps).
- face frames: film time 10-180 s (frames 240-4319), outside the bridge window (frames 600-965, whose 31 committed frames are used as
  they are), with at least one SAM face whose box covers at least MIN_BOX cut pixels (640x360 cut; about 4x that at 1280x534);
- no-mask frames: film time 10-180 s with no SAM object at all.
Each list is sorted, split into 31 contiguous parts of near-equal size, and one frame is drawn from each part with a fixed seed, so the
picks spread over the window. Writes the sample JSON. Prints counts only. usage: python edit_check_sample.py FACES_API_JSONL OUT_JSON"""
import json
import sys

import numpy as np

N, SEED, MIN_BOX = 31, 20261029, 500
WINDOW, BRIDGE = (240, 4320), (600, 966)


def pick(frames, n=N, seed=SEED):
    """One frame from each of n contiguous near-equal parts of the sorted list, drawn with a fixed seed."""
    rng = np.random.default_rng(seed)
    return [int(rng.choice(part)) for part in np.array_split(np.array(sorted(frames)), n)]


def eligible(rows):
    face, nomask = [], []
    for r in rows:
        f = r["frame"]
        if not WINDOW[0] <= f < WINDOW[1]:
            continue
        if not r["objects"]:
            nomask.append(f)
        elif not BRIDGE[0] <= f < BRIDGE[1] and max(o["box_xywh"][2] * o["box_xywh"][3] for o in r["objects"]) >= MIN_BOX:
            face.append(f)
    return face, nomask


def sample(rows):
    face, nomask = eligible(rows)
    return {"seed": SEED, "per_set": N, "min_box_cut_px": MIN_BOX, "window_frames": list(WINDOW), "bridge_frames_excluded": list(BRIDGE),
            "eligible": {"face": len(face), "no_mask": len(nomask)},
            "face_frames": pick(face, seed=SEED), "no_mask_frames": pick(nomask, seed=SEED + 1)}


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1])]
    assert len(rows) == 4560 and [r["frame"] for r in rows] == list(range(4560))
    s = sample(rows)
    json.dump(s, open(sys.argv[2], "w"), indent=1)
    print(f"eligible face frames {s['eligible']['face']}, no-mask frames {s['eligible']['no_mask']}; picked {len(s['face_frames'])} + "
          f"{len(s['no_mask_frames'])} (film s {s['face_frames'][0] / 24:.1f}-{s['face_frames'][-1] / 24:.1f} and "
          f"{s['no_mask_frames'][0] / 24:.1f}-{s['no_mask_frames'][-1] / 24:.1f})")
