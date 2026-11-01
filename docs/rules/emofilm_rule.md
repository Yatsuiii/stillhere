# Rule (version 2): TRIBE v2 forecast vs Emo-FilM viewers, Tears of Steel, the opening three minutes (StillHere Pilot, exploratory)

Version 2 fixed on 2026-10-28 and committed with its scoring code (`stillhere/emofilm_score.py`) and its stimulus-side baseline code
(`scripts/emofilm_film_regressors.py`) BEFORE any Emo-FilM brain data was downloaded or read. Up to this commit only public metadata had
been read: BOLD sidecars, events files, the dataset listing, the data descriptor and the `size_t` and `spacing_tr` columns of the MRIQC
group table (volume counts). No BOLD image, preprocessed series or motion file. The commit time is the record. Nothing here is revised
after the BOLD is read; a later change is reported as a change, with both results. Every number it produces is labelled "exploratory".
It is not part of the frozen PREREG.

Additions of 2026-10-28 (project thread ruling), committed separately, also before any Emo-FilM brain data was read: film time 0 from
each run's own logged onset, the run table printed from metadata, the volume-count check, and cleaning the whole run before cropping
(sections "Run check" and "Window and timing", and the cleaning line under "TRIBE gauges"). The 10 s drop stays, as in T1 and the wall.

## Why version 2
- The bridge-window amendment (`emofilm_bridge_rule.md`, commit 53d14b4) scored 11 or 12 volumes per viewer: too few to test a
  forecast. The bridge scene is no longer scored (see "Bridge scene" below).
- "No combat on screen" limits only what is shown (frames in the video, the SAM 3.1 edits), not the scored window.
- Version 1 (commit 2762956) scored the whole 588 s cut. TRIBE's video path costs about 35.5 T4 minutes per film minute (probe 1a), so
  the whole cut needs about 5.8 h of T4 time and the first three minutes about 1.9 h.
- Neither earlier window was run. Both are kept below as superseded.

## Data
- Stimulus: Tears of Steel (Blender Foundation 2012, CC BY 3.0), official release `tears_of_steel_720p.mov` from
  download.blender.org (SHA-256 efa9062d...; 1280x534 at 24 fps, 44.1 kHz stereo). The Emo-FilM scanned cut is release 0 to 588 s, so
  film time equals release time.
- Viewers: OpenNeuro ds004892 (Emo-FilM fMRI, CC0), all 30 viewers, their Tears of Steel run, minus the runs the run check drops
  (below). sub-S06 uses its ses-4 events (its BOLD run is in ses-4; the extra ses-3 events file is ignored).
- Input image: `derivatives/preprocessing/.../*_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz` (FSL FEAT: MCFLIRT motion
  correction, BET, 6 mm FWHM smoothing, grand-mean scaling, high-pass sigma 50 s; then WM, CSF and six motion regressors removed, and
  registration to MNI). No further nuisance regression is added.
- Projection: fsaverage5 mid-depth surface sampling exactly as the LPP harness (nilearn `vol_to_surf`, kind ball, radius 3 mm, linear
  interpolation, mid meshes between pial and white), then Schaefer-400 17-network parcels on fsaverage5.
- TR 1.3 s from the BOLD sidecar. Slice timing: the derivative is in MNI space, so per-vertex slice times cannot be recovered; every
  vertex uses the run's mean SliceTiming as one offset (0.616 s in every Tears of Steel sidecar).

## Run check (before scoring, from metadata only)
- Expected volume count: 607, the Tears of Steel scan length in TRs with both washouts in the data descriptor's Table 1 ("Scan duration
  (TRs) - includes washout"). Chosen because it is the designed length, fixed before any brain data, and equal to the median of the 30 runs'
  counts. The events files cannot serve as the reference: every run's last logged washout ends at 772.6 to 773.6 s (595 TRs), yet every
  run has 605 to 613 volumes (the scan continues after the last logged row), so an events-based check would drop every run.
  607 was chosen after the volume counts were seen (QC table only, no images).
- A run is dropped if its volume count (the raw run's `size_t` in the MRIQC group table) is off by more than 2 TRs (`volume_check`).
- `scripts/emofilm_run_table.py` prints, for all 30 runs, the subject, session, logged film onset, film duration, end of the last logged
  washout, volume count, the check and the scored volume count. Result before any brain data: 28 runs kept; 2 dropped, sub-S13 and
  sub-S22 (613 volumes each, +6). Kept runs: 605 to 609 volumes; film onsets 93.608 to 94.575 s over all 30.
- When a preprocessed image is opened, its number of volumes must equal that count: the logged onsets count from the raw run's first
  volume, so a derivative that lost or gained volumes cannot be aligned. A mismatch drops the run and is reported.

## Window and timing (no lag search)
- Stimulus window: release 0 to 180 s, the first three minutes of the cut.
- Film time 0 of each run is that run's own logged film onset: the onset of the single film row in its events file
  (`func/*_task-scan_acq-TearsOfSteel_events.tsv`; rows rest, film, rest), in seconds from the run's first volume. Emo-FilM runs open
  with 90 s of fixation, so film time 0 falls about 94 s into the run (`film_onset`).
- Film time of volume j: j x TR + slice offset - film onset.
- No lag search anywhere: the only offset is TRIBE's fixed 5 s, the same for every run, network and baseline.
- Scored volumes: film time at least 10 s (version 1's drop of the response to the film's start after the 90 s washout), and the TRIBE
  sample (film time - 5 s, TRIBE's offset) inside 0 to 180 s and inside the prediction. That is film time 10 to 185 s: 134 or 135
  volumes per viewer (counted from the 30 events files and sidecars: 27 viewers 135, 3 viewers 134; of the 28 kept runs, 25 and 3). The
  ruling's "about 138" is 180 s / 1.3 s; the 10 s drop (kept by the project thread, as in T1 and the wall) removes 3 or 4 of them.
- TRIBE v2 stock predictions (the only model), full (video + audio + text) and audio-only, from the private Kaggle kernel
  `stillhere-pilot-bridge-intact` (name kept from the bridge rule) on release 0 to 190 s: the 10 s margin keeps prediction samples
  through 180 s away from the segment's end. Words and their timings come from the harness's own speech-to-text pass on that segment's
  audio (`words.csv`). The prediction is read at film time - 5 s, linear between its 1 Hz sample times.
- Onset validation: the dataset's own onset check (derivatives/onset_validation, derived from brain data) is not read before
  scoring. It is read right after the film score and reported beside it; if it puts the onsets somewhere else, the film result
  carries that caveat. A run at the authors' stated onset is labelled exploratory and is never the headline.

## Motion and quality
- FD (Power): from the six MCFLIRT parameters (rotations in radians times 50 mm, translations in mm). Source: the session's FEAT motion
  file (`task-mc_..._desc-motion_bold.tsv.gz`), rows of the Tears of Steel run, located by acquisition order and volume counts of the
  session's runs (from the raw sidecars). If the rows cannot be located unambiguously, frames are not censored and run exclusion uses
  MRIQC's `fd_mean` only; that case is reported per viewer.
- Run exclusion as D7, on the scored volumes: mean FD <= 0.5 mm and at most 30% of scored frames with FD > 0.5 mm. Censoring: frames
  with FD > 0.5 mm anywhere in the run are left out of the whole-run cleaning and dropped before scoring.

## TRIBE gauges (per viewer)
- Cleaning, BOLD: detrend, Butterworth high-pass 0.03 Hz, z-score over the whole run (all its volumes, washouts included), on its
  uncensored frames (nilearn `signal.clean` with the censoring mask), then cropped to the scored volumes (`clean_full_then_crop`).
  TRIBE's predictions, the controls and the baseline regressors exist only inside the forecast window, so they get the same cleaning on
  the scored, uncensored frames. (Version 2 as first committed cleaned the BOLD on the window only; replaced before any data was read.)
- HEARING: AUD parcels (SomMotB_Aud, both hemispheres), Fisher-z mean of r(BOLD, audio-only prediction).
- LANGUAGE: LANG parcels (left TempPar, DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl), Fisher-z mean of the partial r(BOLD, full
  prediction) given the LANG-mean audio-only prediction and HRF x word rate.
- VISUAL: visual parcels (VisCent and VisPeri, both hemispheres), Fisher-z mean of r(BOLD, full prediction). The video path runs on one
  T4 (probe 1a); if the kernel's video stage fails, VISUAL is reported as "not run".
- Null: the same score over circular shifts of the regressor by 15 .. T - 16 frames; z = (real - mean) / SD; detected when real is
  above the null's 95th percentile.

## Baselines (the cheap-baselines conventions, docs/rules/cheap_baselines_rule.md)
- Four regressors, each a per-second series over stimulus seconds [k, k+1), convolved with the SPM HRF on a 0.1 s grid and read at
  film time (`heldout.hrf_regressor`; no lag, as the word rate in version 1):
  - WORD RATE: word onsets per second from the kernel's `words.csv` (the same words as LANGUAGE's control).
  - ENVELOPE: mean absolute amplitude of the channel-mean release audio in each second (`heldout.envelope_per_second`).
  - BRIGHTNESS: mean Rec. 709 luma of each release frame (native 24 fps, 1280x534, the frames TRIBE's video path samples from),
    averaged per second.
  - MOTION: mean absolute luma difference of each release frame from the previous one (0 for the first), averaged per second.
- Envelope, brightness and motion are computed from the public release by `scripts/emofilm_film_regressors.py` (no brain data).
- Per viewer, every regressor is scored like TRIBE's unpartialled prediction: the raw Fisher-z mean over parcels of r(cleaned BOLD row,
  cleaned regressor row), one baseline row tiled over the parcels, z against the same circular-shift null, on LANG (TRIBE full), AUD
  (TRIBE audio-only) and VIS (TRIBE full; only if VISUAL runs). TRIBE is unpartialled here because LANGUAGE partials word rate out of it.

## Comparison with the baselines
- Paired over viewers: d = TRIBE z - baseline z. Reported: n, k (d > 0), the Hodges-Lehmann median of d with a 95% viewer bootstrap
  interval (10,000 resamples, numpy default_rng(20261028)), two-sided Wilcoxon p (`cheap_baselines.compare`). "TRIBE beats the
  baseline" only if the HL is above 0 and the interval's lower end is above 0; "baseline beats TRIBE" only if the HL is below 0 and the
  upper end is below 0; otherwise "no difference shown".
- Primary comparisons, stated before any data is read: each network against the baseline meant to explain it: LANG vs word rate, AUD vs
  envelope, VIS vs brightness, VIS vs motion. Every other network and baseline pair is reported beside them. No correction across
  comparisons (descriptive).
- The pitch claims only what TRIBE beats. Where a baseline beats TRIBE, or no difference is shown, the write-up says so and no line
  claims otherwise.

## Group results (all exploratory)
- Per TRIBE gauge: median z over viewers with a 95% viewer bootstrap CI (10,000 resamples, `numpy.random.default_rng(20261028)`), and
  the share of viewers detected. Criterion: the group median z CI excludes 0.
- Volunteer comparison: ISC(k) for k = 1 .. 15: each viewer against the mean of k other viewers' cleaned series (20 random draws per k,
  seed 20261028), scored by the same gauge and null (LANGUAGE ISC partials the same controls). Because film onsets differ by up to
  about 1 s, each viewer's cleaned parcel series is first resampled by linear interpolation onto a common film-time grid (10 s to 185 s,
  steps of one TR: 135 points); censored frames are missing there and ignored. TRIBE's per-viewer z values are placed on that curve
  with the section-8 volunteer-equivalent rule (mean z and the lower end of its 10,000-resample viewer bootstrap CI, rounded down).
- The write-up names this rule and its commit.

## Bridge scene (release 25.00 to 40.25 s)
- Kept only for the SAM 3.1 edits (faces removed, area matched) and the video. It is not scored: its 11 or 12 volumes per viewer are
  too few to test a forecast. The edited versions are forecast-only: TRIBE's predictions for them are compared with the intact
  forecast, with no BOLD.
- The scene lies inside the forecast window (release 0 to 190 s), so the intact forecast covers it.

## Superseded (neither was run; no Emo-FilM brain data had been read)
- Version 1 (commit 2762956): scored volumes from film time 10 s to the end of the 588 s cut, TRIBE gauges and ISC only.
- Bridge-window amendment (commit 53d14b4, `emofilm_bridge_rule.md`): film time 30.00 to 45.25 s, 11 or 12 volumes per viewer, a
  wrong-segment null. Replaced because 11 or 12 volumes are too few to test a forecast.

Dated note, Nov 1 2026 (cloud ruling of Nov 1, 06:05 UTC, item 2; the text above is unchanged): the intact film's public line was written
after the result (`results/emofilm_film_score.json`, scored once, Nov 1, 05:29:59-05:31:54 UTC) from this rule's reporting conventions: it
claims only what TRIBE beat and states the misses, with the numbers copied from the file. "Exploratory. On a film clip in neither TRIBE v2's
training list nor its evaluation list (Tears of Steel, film seconds 10-185; Emo-FilM, 16 viewers), TRIBE v2's forecast followed the
viewers' hearing areas (median z 3.23) and visual areas (2.77) more closely than the sound envelope, brightness or motion did. In language
areas (0.80) it did no better than word rate. In every area it was worth less than one real viewer." Beside it: "What it had to beat: word
rate, the sound envelope, brightness and motion (beaten in 3 of 4 primary comparisons; every pair in results/emofilm_film_score.json)."
This line names hearing against the sound envelope, which the page's Oct 29 headline rule kept out; the Nov 1 ruling supersedes it for
this line.
