# Face map rule: does TRIBE's forecast of a face edit land where viewers' brains follow faces? (exploratory)

Project-thread ruling of Thu Oct 29, 06:00 UTC. Committed, with `stillhere/face_map.py`, its collinearity gate and their synthetic
tests, before any Emo-FilM brain data is read and before any film-versions forecast exists; it is pushed to the private GitHub repo
before any BOLD read. The commit time is the record. Every number it produces is labelled exploratory. Scored once, at the logged
onsets; never re-scored at another onset, window, parcel set or HRF.

Additions of the project-thread rulings of Thu Oct 29, 06:30 UTC and after (the final film stimuli v2 for F; 06:50 UTC: the Saturday
chain, the either-or read guard and `not-run`, the stimulus checks reported beside the result, the descriptive), committed before any
Emo-FilM BOLD was read: the forecast maps come from the committed intact film kernel and two edit kernels and are computed, committed
and pushed before any BOLD is read (`face_map_runner.py maps`, `results/face_map_RC.json`); a read guard; the gate's timing; the cutoffs
(sections below). Disclosure: the Emo-FilM download (`pilot/emofilm/download_tos_runs.py`) started Oct 29 05:45 UTC; it only downloads;
nothing reads it before the read guard allows. Before the 06:00 UTC ruling, one session motion file (sub-S01 ses-4) was opened for its
shape only (946 rows x 6 values; no value printed or used).

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
- Oct 31, 2026 (cloud audit of 05:30 UTC): the control scored is `docs/rules/film_stimulus_rule.md`'s v2 (88.9% in-view coverage,
  51 faces without a control).
- Per parcel (the mean of its vertices), the forecast read at film time - 5 s (linear between the 1 Hz samples, as the intact rule) on
  the intact rule's scored film-time grid, averaged over the grid. The grid is exactly the intact rule's (`emofilm_score.common_grid`):
  film time 10 to 185 s in steps of one TR (135 points at TR 1.3 s), so the prediction is read at 5 to 180 s of the film. "10-180" in
  the thread was shorthand for this grid.
- R = intact - faces removed. C = intact - area matched.
- `face_map_runner.py maps` writes `results/face_map_RC.json`: both maps, each map's SHA-256, the three input forecasts' SHA-256 and the
  parcel order's SHA-256. It is committed and pushed to origin/main before any BOLD is read. `score` reads R and C from that file only
  (never recomputes them), checks both SHA-256 values, and requires the local file's git blob hash to equal origin/main's.

## Read guard
The repo the guard checks is origin/main of the private GitHub repo `Yatsuiii/stillhere` (`https://github.com/Yatsuiii/stillhere.git`),
through the clone given as `--rc_repo` (`git fetch`, then `git show origin/main:<path>` and blob hashes). No process opens an Emo-FilM
BOLD file (or motion file) unless EITHER
- gate 0 has passed AND `results/face_map_RC.json` is on origin/main with the same blob hash as the local copy (the face path), OR
- `results/face_map.json` with outcome NOT RUN and a non-empty reason is committed and on origin/main (the NOT RUN path).

The check runs first in `emofilm_score_runner.py prepare` (before any image or motion file is opened) and in `face_map_runner.py
score`, which needs the face path; otherwise they refuse.

Once `results/face_map.json` is on origin/main the face test is closed for good: NOT RUN is never revived, and a scored outcome is
scored once. `maps`, `not-run` and `score` then refuse.

Sunday's call (project thread ruling of Oct 29, 07:30 UTC), made by Yatsu: if R and C are not on GitHub by Sun Nov 1 17:00 UTC and no
running forecast can finish and push by 17:45 UTC, push NOT RUN with the logged reason (GPU time ran out, or the failing step), then
score the intact film alone. NOT RUN is pushed automatically only when gate 0 fails; every other failure waits for this call. The
project thread checks the repository on Sat Oct 31 12:00 UTC and Sun Nov 1 15:30 UTC and pings Yatsu if anything is missing.

## Face regressor F(t) and the real face maps
- Face-area fraction per release frame: the sum of that frame's face-region pixels divided by 1280 x 534. Source: the final film
  stimuli (`docs/rules/film_stimulus_rule.md`; stillhere-film-stimuli, MANIFEST SHA-256 d4e27d55...): `edits_v2.json`, field `px` of
  each face, the block-snapped SAM 3.1 region the v2 edit removed, over the whole frame. `edits_v2.json` must match its hash in that
  MANIFEST and cover frames 0 to 4559 (release 0 to 190 s at 24 fps). It goes through the brightness code path: mean per second
  (`emofilm_score.per_second`), then the SPM HRF regressor read at film time (`heldout.hrf_regressor`, no lag).
- Why the whole frame and not TRIBE's view (`px_in_view`, release x 404 to 873, y 32 to 501): M_i measures what the viewers' brains
  follow, and the viewers saw the whole frame in the scanner. TRIBE's view enters through R and C, which are TRIBE's own forecasts of
  the edited segments. (Of the 3,504 face instances, 45.5% lie wholly inside TRIBE's view.)
- M_i, per viewer i and parcel: the Fisher z of the partial correlation between the cleaned BOLD row and F, controlling brightness,
  motion, envelope and word rate (the intact rule's baseline regressors at film time); F and the controls are cleaned on the viewer's
  kept scored frames exactly as the intact baselines.

## Parcels tested
The 374 Schaefer-400 parcels outside the project's auditory set (SomMotB_Aud, both hemispheres: 7) and language set (left TempPar,
DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl: 19).

## Gate 0: collinearity (stimulus only, before any BOLD is read)
The variance inflation factor of F against brightness, motion, envelope and word rate (the five HRF regressors on the common grid) must
be at most 5; otherwise the face map is NOT RUN. `face_map_runner.py gate` writes `--out` with key `"pass"` and exits with code 3 on a
fail.

Timing, and why: word rate comes from the intact forecast kernel's `words.csv` (the transcript TRIBE's own pipeline makes of the
film), which exists only after Saturday's intact forecast. So the gate runs on Saturday, right after that forecast, on its `words.csv`
(`--words` takes that file or the folder the kernel's output was downloaded to). It needs no GPU and runs locally.

## The Saturday chain, in order
1. picker-pred.
2. The intact film forecast.
3. Gate 0 (CPU, local), on the intact forecast's `words.csv`.
4. Only if gate 0 passes: the film-rem and film-am forecasts, then `face_map_runner.py maps`; commit and push
   `results/face_map_RC.json`. If gate 0 fails: `face_map_runner.py not-run --reason "gate 0: ..."`; commit and push
   `results/face_map.json`; no edit forecast is made for the face test.
5. The Glimmer edit check.
6. The picker fallback parts, only if triggered.
7. Scoring: `emofilm_score_runner.py prepare` (opens the BOLD only through the read guard), the intact film `score`, then, on the face
   path, `face_map_runner.py score`; commit and push its `results/face_map.json`.

Everything that is not in this chain keeps the Fri Oct 30, 20:00 UTC cutoff.

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
Nov 1, 18:00 UTC. The score itself has no Friday cutoff. If R and C are not on origin/main by Sun 18:00 UTC: run `not-run` with that
reason, commit and push `results/face_map.json`, then score the intact film alone. `score` refuses R and C committed after the cutoff
and says so. The intact film score does not otherwise depend on the face test: its scorer and outputs are unchanged, and the face
test only decides which path of the read guard opens the BOLD.

## Reported beside the result (stimulus facts; they decide nothing)
From `results/film_stimulus_checks.json` (`docs/rules/film_stimulus_rule.md`), copied into `results/face_map.json`:
- Coverage: the area-matched control covers 0.889 of the in-view face pixels.
- 51 face instances have no control: the largest close-ups, whose in-view part fits nowhere else in TRIBE's view. Their frames stay
  in the forecast maps and in every test; this is disclosed beside the result.
- Eccentricity: median eccentricity of the changed pixels in TRIBE's view, 77.7 px for the control vs 95.1 px for face removal
  (the controls sit nearer the centre).
- Change energy: control over face removal, 0.894.

## Descriptive (pre-committed; never decides)
R and C recomputed on only the grid points where every in-view face has a control: a point is kept when every frame of the 4 s clip
TRIBE encodes for it (the 4 s ending at film time - 5 s) has `control_px` > 0 for each face with `px_in_view` > 0. On the real
stimuli 51 frames fail this and 129 of the 135 grid points are kept. `maps` writes these maps (with their SHA-256 and the kept count)
in `results/face_map_RC.json`; `score` reports rho(R, Mbar), rho(C, Mbar) and the median d_i with them. Labelled descriptive; it is
not an outcome and never changes one.

On a PASS the film leads the video only if the median of d_i computed on the tested parcels outside the visual network (VisCent and
VisPeri, both hemispheres) is also above 0; otherwise the PASS is captioned "carried by visual-network parcels".

## Wording
SAM edits the frames; Spark Open and Glimmer run the edit check. A result here is described only as TRIBE's forecast compared with
viewers' brain responses, with its outcome word from the list above.
