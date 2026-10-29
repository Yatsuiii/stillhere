# Face map rule: does TRIBE's forecast of a face edit land where viewers' brains follow faces? (exploratory)

Project-thread ruling of Thu Oct 29, 06:00 UTC. Committed, with `stillhere/face_map.py`, its collinearity gate and their synthetic
tests, before any Emo-FilM brain data is read and before any film-versions forecast exists; it is pushed to the private GitHub repo
before any BOLD read. The commit time is the record. Every number it produces is labelled exploratory. Scored once, at the logged
onsets; never re-scored at another onset, window, parcel set or HRF.

Additions of the project-thread ruling of Thu Oct 29, 06:30 UTC, committed before any Emo-FilM BOLD was read: the forecast maps come
from the committed intact film kernel and two edit kernels and are computed, committed and pushed before any BOLD is read
(`face_map_runner.py maps`, `results/face_map_RC.json`); a read guard; the gate's timing; the cutoffs (sections below). Disclosure:
the Emo-FilM download (`pilot/emofilm/download_tos_runs.py`) started Oct 29 05:45 UTC; it only downloads; nothing reads it before
the read guard allows. Before the 06:00 UTC ruling, one session motion file (sub-S01 ses-4) was opened for its shape only (946 rows
x 6 values; no value printed or used).

## Question
SAM 3.1 removes the faces from the opening of Tears of Steel, with an equal-area control edit. TRIBE v2 forecasts the brain response to
each version. Does the map of where the forecast changes when faces go (R) match where real viewers' brains follow how much face is on
screen (M), and more than the map for the control edit (C)?

## What it inherits from the intact film rule (`docs/rules/emofilm_rule.md`), unchanged
The 28 kept runs; film time 0 from each run's logged onset; the scored volumes (film time 10 to 185 s, TRIBE sample 5 to 180 s, the
fixed 5 s offset, no lag search); Schaefer-400 17-network parcels on fsaverage5; the same projection, cache and cleaning
(`clean_full_then_crop` on the same cached series); the same run exclusion (D7, or MRIQC fd_mean when the motion rows cannot be
located). The intact scorer runs byte-identical; the face map then runs on the same cleaned series, over all 400 parcels.

## Forecast maps (no brain data; computed, committed and pushed before any BOLD is read)
- Sources: the committed intact film kernel's forecast (`stillhere-pilot-bridge-intact`, stock TRIBE v2, release 0 to 190 s) and two
  edit kernels, each saving its own `pred_full.npz` in the intact kernel's format: film-rem (faces removed) and film-am (area matched:
  each face's exact shape moved to a non-face spot of the same frame, 100% of face pixels matched; `scripts/make_bridge_edit_cuts.py
  --start-frame 0 --frames 4560`). The three must share sample times and shape.
- Per parcel (the mean of its vertices), the forecast read at film time - 5 s (linear between the 1 Hz samples, as the intact rule) on
  the intact rule's scored film-time grid (`emofilm_score.common_grid`: film time 10 to 185 s in steps of one TR, so the prediction is
  read at 5 to 180 s of the film), averaged over the grid.
- R = intact - faces removed. C = intact - area matched.
- `face_map_runner.py maps` writes `results/face_map_RC.json`: both maps, each map's SHA-256, the three input forecasts' SHA-256 and the
  parcel order's SHA-256. It is committed and pushed to origin/main before any BOLD is read. `score` reads R and C from that file only
  (never recomputes them), checks both SHA-256 values, and requires the local file's git blob hash to equal origin/main's.

## Read guard
No process opens an Emo-FilM BOLD file (or motion file) unless gate 0 has passed AND `results/face_map_RC.json` is on origin/main of the
private GitHub repo with the same blob hash as the local copy (`git fetch`, then compare). The check runs first in
`emofilm_score_runner.py prepare` (before any image or motion file is opened) and in `face_map_runner.py score`; otherwise they refuse.

## Face regressor F(t) and the real face maps
- Face-area fraction per release frame: the sum of that frame's SAM 3.1 face-mask pixels (`edits.json`, `masked_px`, release
  resolution: the masks the edit removed) divided by 1280 x 534. It goes through the brightness code path: mean per second
  (`emofilm_score.per_second`), then the SPM HRF regressor read at film time (`heldout.hrf_regressor`, no lag).
- M_i, per viewer i and parcel: the Fisher z of the partial correlation between the cleaned BOLD row and F, controlling brightness,
  motion, envelope and word rate (the intact rule's baseline regressors at film time); F and the controls are cleaned on the viewer's
  kept scored frames exactly as the intact baselines.

## Parcels tested
The 374 Schaefer-400 parcels outside the project's auditory set (SomMotB_Aud, both hemispheres: 7) and language set (left TempPar,
DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl: 19).

## Gate 0: collinearity (stimulus only, before any BOLD is read)
The variance inflation factor of F against brightness, motion, envelope and word rate (the five HRF regressors on the common grid) must
be at most 5; otherwise the face map is NOT RUN. It runs when the intact forecast kernel's `words.csv` exists (after Saturday's intact
forecast); `--words` takes that file or the folder the kernel's output was downloaded to.

## Gate 1: reliability
Leave one out: per viewer, Spearman rho across the 374 parcels between M_i and the mean of the other viewers' maps. One-sided exact
Wilcoxon (greater than 0) across viewers, p < 0.05; otherwise UNINFORMATIVE.

## Tests (all three criteria must hold for each)
- B1: rho_R,i = Spearman rho across the 374 parcels between R and M_i is above 0.
- B2: d_i = rho_R,i - rho_C,i is above 0 (rho_C,i the same with C).
- Criteria: (1) one-sided exact Wilcoxon across viewers, p < 0.05; (2) at least k*(N) viewers above 0, where k*(N) is the smallest k
  with P(X >= k) < 0.05 for X ~ Binomial(N, 0.5) (19 of 28); (3) spin permutation p < 0.05 on the group-mean map Mbar (the mean of the
  viewers' maps): statistic rho(R, Mbar) for B1 and rho(R, Mbar) - rho(C, Mbar) for B2, over the 374 parcels.
- Spin: each parcel's centroid is the normalised mean of its fsaverage5 sphere coordinates (nilearn's bundled fsaverage5). 1,000 uniform
  random rotations (seed 20261029); the left hemisphere is rotated and the right gets the mirrored rotation; within each hemisphere the
  rotated centroids are matched one to one to the original ones (Hungarian assignment); all 400 parcels rotate, R and C together, and
  the statistic is computed over the 374 tested parcels. p = (1 + the number of rotations at or above the observed value) / 1001.

## Outcome, the first that applies
NOT RUN (gate 0 fails, an input is missing, or R and C were not on origin/main by Sun Nov 1, 18:00 UTC: the time of the commit that put
them there), UNINFORMATIVE (gate 1 fails), FAIL (B1 fails), FAIL not face-specific (B1 passes, B2 fails), PASS.

Cutoffs: everything that needs no GPU (this rule, the code, the gate code) by Fri Oct 30, 20:00 UTC; R and C on origin/main by Sun
Nov 1, 18:00 UTC. The score itself has no Friday cutoff.

On a PASS the film leads the video only if the median of d_i computed on the tested parcels outside the visual network (VisCent and
VisPeri, both hemispheres) is also above 0; otherwise the PASS is captioned "carried by visual-network parcels".

## Wording
SAM edits the frames; Spark Open and Glimmer run the edit check. A result here is described only as TRIBE's forecast compared with
viewers' brain responses, with its outcome word from the list above.
