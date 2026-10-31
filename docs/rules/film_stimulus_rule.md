# Film stimuli for the face-map test: what TRIBE sees, the one re-render, and the checks (exploratory)

Project-thread rulings of Oct 29, 06:00 UTC, items 2 and 3b. Committed before any edited forecast exists; the commit time is the record.
After this commit the stimuli never change. Wording: SAM filled the faces it masked with the surrounding colour (controls get exactly the same fill); Spark Open and
Glimmer run the edit check. Nothing here "detects faces".

## What TRIBE sees (measured, not assumed)

- Reader: neuralset's `HuggingFaceVideo` opens the segment with moviepy (`VideoFileClip`) and reads RGB frames at full resolution with
  `get_frame(t)`. TRIBE v2's checkpoint config: `frequency: 2.0`, `clip_duration: 4`, model `facebook/vjepa2-vitg-fpc64-256`; so at each
  0.5 s step it encodes 64 frames spread over the 4 s ending there.
- View: V-JEPA2's own video processor (`VJEPA2VideoProcessor`: resize shortest edge to 292, bilinear; centre crop 256 x 256). Found by
  perturbing single rows and columns through that processor: only release pixels x 404 to 873 and y 32 to 501 (a 470 x 470 window of the
  1280 x 534 frame) change the model's input. Faces outside it are invisible to TRIBE.

## The re-render (the one allowed, before any edited forecast)

The first film cuts (`pilot/film/film_edits`) were encoded separately from the release, lossy, with their own audio, so pixels outside
the edits and the audio could not match the intact forecast's input. Changed, for every frame alike (global settings only):

1. Source: the intact segment itself. `seg_intact.mp4` is release 0 to 190 s cut once with the intact kernel's own command
   (`ffmpeg -ss 0 -t 190 -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k`); the intact kernel now reads this exact file
   (SHA-256 asserted) from the private dataset stillhere-film-stimuli instead of cutting its own, so every cut starts from the frames
   TRIBE forecasts for the intact film.
2. Edits on the decoded yuv420p planes, every edit region grown to whole 2 x 2 blocks (the chroma grid), encoded losslessly
   (`libx264 -qp 0`), with the segment's audio stream copied unchanged. So outside an edit every decoded pixel equals the intact segment's.
3. Fill: unchanged in kind, the local mean (of a 6 px band around the region, other faces left out), per plane. No blur.
4. Control (area matched): for each face, the part TRIBE sees (inside the view) is moved to the nearest free spot inside the view
   (exact pixel test, 4 px from faces and other patches), cut into 2, 4 or 8 equal pieces only when it fits nowhere whole. A face with
   no part in the view gets no control. Masks are unchanged: the same SAM 3.1 output (Meta Model API, concept face, on the 640 x 360 cut
   of release 0 to 190 s), mapped to release pixels by nearest neighbour and grown 2 px.

Build (`scripts/make_film_stimuli_v2.py`): 4,560 frames; 3,504 face instances, 2,717 with pixels in TRIBE's view; a control was placed for
2,666 of them (40 in pieces); controls cover 88.9% of in-view face pixels. The 51 faces without a control are the largest close-ups,
whose in-view part fits nowhere else in the view even in 8 pieces. Files and hashes: `MANIFEST.sha256` of stillhere-film-stimuli
(SHA-256 d4e27d55e2c462f8edd9d2373407692d03534893a3fac284a6819897f52a6f44).

## Checks (`scripts/film_stimulus_checks.py`, every frame through TRIBE's own reader)

1. Pixels outside each edit region are identical to the intact cut, in both edited cuts.
2. The audio is identical in all three cuts (AAC packets and decoded samples).
3. Each placed control's area is within 1% of its face's in-view area (it is the same shape, so equal); faces without a control counted.
4. Every control pixel is inside TRIBE's view.
Reported: the median eccentricity of changed pixels in TRIBE's view (control vs face removal) and the change-energy ratio (control over
face removal), on every 12th frame after V-JEPA2's resize and crop. Results: `results/film_stimulus_checks.json`.

Results (all PASS): 0 pixels differ outside the edits in 4,560 frames of either edited cut; audio identical (AAC packets and decoded
samples); 2,666 placed controls all exactly their face's in-view area, 51 faces without one (11.1% of in-view face pixels); 0 control
pixels outside TRIBE's view. Reported: median eccentricity of changed pixels in TRIBE's view 77.7 px for controls vs 95.1 px for face
removal (ratio 0.817: controls sit nearer the centre); change energy, control over face removal, 0.894.

## Forecasts

- intact (stillhere-pilot-bridge-intact): the committed film kernel; it reads `seg_intact.mp4`, and after its stages exports its events
  (the transcript), the extracted audio, words.csv and its audio and text feature caches with a MANIFEST.sha256.
- film-rem and film-am (stillhere-pilot-film-rem, -am): the same kernel with only the video file changed (`faces_removed_v2.mp4`,
  `area_matched_v2.mp4`); each loads the intact export, every file hash-asserted, so the transcript and the audio and text features are
  the intact run's; only the video stage computes new features. R and C use the intact kernel's forecast. Each saves its own output.
- Saturday order: picker-pred, intact film, film-rem, film-am, the Glimmer edit check, then the picker fallback parts only if
  triggered, then scoring. Builder: `pilot/film/k_film/make_film_kernels.py` (SHA-256 18a19f742ec90039...); kernels: intact f367fac66857f559...,
  film-rem 47078e93b851d74c..., film-am 0a7ef8bc370b857e....
- Oct 31, 10:25 UTC: the builder and kernels above are superseded by public d7bba3d: builder `kernels/film/make_film_kernels.py`
  ea3126b1cfb18004..., intact 4deff1a0e1a5429e..., film-rem 29f59a53f05d29eb..., film-am 09b3bec84d77577a..., lock
  `kernels/film/film_lock.txt` 13306c91335c0774... (the film entry in `docs/frozen_code_changes.md` says what changed and why). The
  stimuli are unchanged.
- CPU checks before Saturday: film-rem's and film-am's own input code ran on Kaggle CPU (stillhere-filmcheck) against the real
  stimuli dataset and a stand-in intact export: segment SHA-256 asserted, export files hash-checked, PASS; the intact kernel's
  export code ran in a local sandbox: it exports exactly the audio and text caches (not video), manifest self-check PASS.
- Output format: film-rem and film-am save pred_full.npz and pred_audio.npz exactly as the intact kernel does (same stage code;
  only the output folder differs: film_rem/, film_am/), so R and C come from intact minus each, per parcel.
