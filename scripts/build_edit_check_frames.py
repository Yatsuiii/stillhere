"""Builds the edit-check frame set (docs/rules/edit_check_rule.md) from the FINAL film-version cuts and the committed bridge frames:
- bridge: the 31 committed bridge frames (glimmer_edit_frames, MANIFEST checked), three versions, copied byte for byte;
- new: the 31 sampled face frames (eval/edit_check_sample.json), decoded from {intact,faces_removed,area_matched}_0p00_190p00.mp4;
- no_mask: the 31 sampled no-mask frames, decoded from the intact cut only.
Decoded frames are written as JPEG quality 95 at the cuts' own size. Writes <version>/<name>.jpg, frames.json (sets, names, cut frame,
film time, the cuts' MANIFEST hash) and MANIFEST.sha256. Shows no frame to any model. Prints counts only.
usage: python build_edit_check_frames.py CUTS_DIR BRIDGE_FRAMES_DIR SAMPLE_JSON OUT_DIR"""
import hashlib
import json
import os
import shutil
import sys

import cv2

CUTS, BRIDGE, SAMPLE, OUT = sys.argv[1:5]
VERSIONS = ("intact", "faces_removed", "area_matched")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def check_manifest(d):
    bad = [l.split()[1] for l in open(f"{d}/MANIFEST.sha256") if sha(f"{d}/{l.split()[1]}") != l.split()[0]]
    assert not bad, f"{len(bad)} files in {d} differ from its manifest"
    return sha(f"{d}/MANIFEST.sha256")


def decode(path, frames):
    """{cut frame: BGR image} for the wanted frames, read in order."""
    cap, out, n, last = cv2.VideoCapture(path), {}, 0, max(frames)
    while n <= last:
        ok, f = cap.read()
        assert ok, f"{os.path.basename(path)} ended at frame {n}"
        if n in frames:
            out[n] = f
        n += 1
    return out


cuts_hash, bridge_hash = check_manifest(CUTS), check_manifest(BRIDGE)
S = json.load(open(SAMPLE)); new, nomask = S["face_frames"], S["no_mask_frames"]
bridge_names = [f["name"] for f in json.load(open(f"{BRIDGE}/frames.json"))["frames"]]
for v in VERSIONS:
    os.makedirs(f"{OUT}/{v}", exist_ok=True)
    for name in bridge_names:
        shutil.copyfile(f"{BRIDGE}/{v}/{name}", f"{OUT}/{v}/bridge_{name}")
    want = set(new) | (set(nomask) if v == "intact" else set())
    for c, img in decode(f"{CUTS}/{v}_0p00_190p00.mp4", want).items():
        cv2.imwrite(f"{OUT}/{v}/f{c:04d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
meta = {"rule": "docs/rules/edit_check_rule.md", "cuts_manifest_sha256": cuts_hash, "bridge_manifest_sha256": bridge_hash,
        "sample_sha256": sha(SAMPLE),
        "sets": {"bridge": [{"name": f"bridge_{n}"} for n in bridge_names],
                 "new": [{"name": f"f{c:04d}.jpg", "cut_frame": c, "film_s": round(c / 24, 3)} for c in new],
                 "no_mask": [{"name": f"f{c:04d}.jpg", "cut_frame": c, "film_s": round(c / 24, 3)} for c in nomask]}}
json.dump(meta, open(f"{OUT}/frames.json", "w"), indent=1)
files = sorted(f"{v}/{n}" for v in VERSIONS for n in os.listdir(f"{OUT}/{v}")) + ["frames.json"]
with open(f"{OUT}/MANIFEST.sha256", "w") as fh:
    for p in files:
        fh.write(f"{sha(f'{OUT}/{p}')}  {p}\n")
print(f"bridge {len(bridge_names)} x3, new {len(new)} x3, no-mask {len(nomask)} x1; files {len(files)}")
