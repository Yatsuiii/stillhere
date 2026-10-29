"""Stimulus checks for the film-rem and film-am forecasts (docs/rules/film_stimulus_rule.md; ruling 3b of Oct 29), run before any edited
forecast exists, on the v2 cuts made by scripts/make_film_stimuli_v2.py. Every frame is decoded through TRIBE's own video reader
(moviepy VideoFileClip, as neuralset's HuggingFaceVideo reads it); TRIBE's view is V-JEPA2's own video processor (shortest edge 292,
centre crop 256), loaded from PROC_DIR (facebook/vjepa2-vitg-fpc64-256's video_preprocessor_config.json).
  1 pixels outside each edit region identical to the intact cut, every frame of both edited cuts
  2 audio identical in all three cuts (AAC packets and decoded samples)
  3 each placed control's pixel count within 1% of its face's in-view pixel count; faces with no control counted
  4 every control pixel inside TRIBE's view
  reported: median eccentricity of changed pixels in TRIBE's view (control vs face removal) and the change-energy ratio (control / face
  removal), both on every 12th frame (2 per second) after V-JEPA2's resize and crop.
Writes results/film_stimulus_checks.json; prints counts, pass/fail and the two ratios only.
usage: python film_stimulus_checks.py STIM_DIR PROC_DIR"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

STIM, PROC = sys.argv[1:3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moviepy import VideoFileClip  # noqa: E402

VIEW = (404, 32, 873, 501)
CUTS = {"intact": "seg_intact.mp4", "faces_removed": "faces_removed_v2.mp4", "area_matched": "area_matched_v2.mp4"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res = {"label": "exploratory", "rule": "docs/rules/film_stimulus_rule.md", "checks": {}, "reported": {}}

bad = [l.split()[1] for l in open(f"{STIM}/MANIFEST.sha256") if sha(f"{STIM}/{l.split()[1]}") != l.split()[0]]
assert not bad, f"{len(bad)} files differ from the manifest"
res["manifest_sha256"] = sha(f"{STIM}/MANIFEST.sha256")
edits = json.load(open(f"{STIM}/edits_v2.json"))
R = dict(np.load(f"{STIM}/regions.npz")); N, H, W = (int(x) for x in R["shape"])     # loaded once

# 2 audio
pk = {v: hashlib.sha256(subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{STIM}/{f}", "-map", "0:a", "-c", "copy", "-f", "data", "-"],
                                       capture_output=True, check=True).stdout).hexdigest() for v, f in CUTS.items()}
clips = {v: VideoFileClip(f"{STIM}/{f}") for v, f in CUTS.items()}
pcm = {v: hashlib.sha256(c.audio.to_soundarray(fps=44100).tobytes()).hexdigest() for v, c in clips.items()}
res["checks"]["audio_identical"] = {"pass": len(set(pk.values())) == 1 and len(set(pcm.values())) == 1,
                                    "aac_packets_sha256": pk["intact"][:16], "pcm_sha256": pcm["intact"][:16]}

# 1 outside pixels, 4 control inside the view, and the TRIBE-view sample
assert all(abs(c.fps - 24) < 1e-6 for c in clips.values()), "a cut is not 24 fps"
from transformers import AutoVideoProcessor  # noqa: E402
proc = AutoVideoProcessor.from_pretrained(PROC)
view = lambda img: proc(videos=[img[None]], return_tensors="pt", do_normalize=False)["pixel_values_videos"][0, 0].numpy()
inview = np.zeros((H, W), bool); inview[VIEW[1]:VIEW[3] + 1, VIEW[0]:VIEW[2] + 1] = True
out_frames = {"faces_removed": 0, "area_matched": 0}; out_px = {"faces_removed": 0, "area_matched": 0}; ctrl_outside_view = 0
ecc = {"faces_removed": [], "area_matched": []}; energy = {"faces_removed": 0.0, "area_matched": 0.0}
yy, xx = np.mgrid[0:256, 0:256]; dist = np.hypot(yy - 127.5, xx - 127.5)
iters = {v: c.iter_frames() for v, c in clips.items()}; n = 0
for i in range(N):
    fr = {v: next(it, None) for v, it in iters.items()}
    if any(f is None for f in fr.values()):
        break
    n += 1
    for v in ("faces_removed", "area_matched"):
        reg = np.unpackbits(R[v][i])[:H * W].reshape(H, W).astype(bool)
        d = (fr[v] != fr["intact"]).any(-1) & ~reg
        if d.any():
            out_frames[v] += 1; out_px[v] += int(d.sum())
        if v == "area_matched":
            ctrl_outside_view += int((reg & ~inview).sum())
    if i % 12 == 0:
        a = view(fr["intact"])
        for v in ("faces_removed", "area_matched"):
            diff = view(fr[v]) - a; ch = (np.abs(diff) > 1e-6).any(0)
            ecc[v].extend(dist[ch].tolist()); energy[v] += float((diff ** 2).sum())
res["frames_checked"] = n
res["checks"]["frame_count"] = {"pass": n == N and N == 4560, "frames": n}
res["checks"]["outside_edit_identical"] = {"pass": out_frames["faces_removed"] == 0 and out_frames["area_matched"] == 0,
                                           "frames_with_any_difference": out_frames, "pixels": out_px}
res["checks"]["control_inside_view"] = {"pass": ctrl_outside_view == 0, "control_pixels_outside_view": ctrl_outside_view}

# 3 control area
faces = [x for f in edits["per_frame"] for x in f["faces"] if x["px_in_view"]]
placed = [x for x in faces if x["pieces"]]
off = [x for x in placed if abs(x["control_px"] - x["px_in_view"]) > 0.01 * x["px_in_view"]]
res["checks"]["control_area_within_1pct"] = {"pass": not off, "placed": len(placed), "outside_1pct": len(off),
                                             "faces_in_view": len(faces), "no_control": len(faces) - len(placed),
                                             "split": sum(1 for x in placed if x["pieces"] > 1),
                                             "in_view_face_px_with_control": round(sum(x["px_in_view"] for x in placed) / sum(x["px_in_view"] for x in faces), 4)}
med = {v: float(np.median(e)) if e else None for v, e in ecc.items()}
res["reported"] = {"median_eccentricity_view_px": med, "eccentricity_ratio_control_over_face": round(med["area_matched"] / med["faces_removed"], 4),
                   "change_energy_ratio_control_over_face": round(energy["area_matched"] / energy["faces_removed"], 4),
                   "sampled_frames": len(range(0, n, 12))}
res["pass"] = all(c["pass"] for c in res["checks"].values())
os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(res, open(f"{ROOT}/results/film_stimulus_checks.json", "w"), indent=1)
for k, c in res["checks"].items():
    print(k, "PASS" if c["pass"] else "FAIL", {a: b for a, b in c.items() if a != "pass"})
print("reported", res["reported"]); print("all checks:", "PASS" if res["pass"] else "FAIL")
