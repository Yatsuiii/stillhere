# Rule: TRIBE v2 forecast vs Emo-FilM viewers, Tears of Steel (StillHere Pilot, exploratory)

Fixed on 2026-10-28, committed with its scoring code (`stillhere/emofilm_score.py`) BEFORE any Emo-FilM brain data was downloaded or
read. The commit time is the record. It is not revised after the BOLD is read; anything changed later is reported as a change, with
both results. Every number it produces is labelled "exploratory". It is not part of the frozen PREREG.

## Data
- Stimulus: Tears of Steel (Blender Foundation 2012, CC BY 3.0), official release `tears_of_steel_720p.mov` from
  download.blender.org, cut to the scanned version: release time 0 to 588 s (end credits removed), per the dataset's own cut file.
- Viewers: OpenNeuro ds004892 (Emo-FilM fMRI, CC0), all 30 viewers, their Tears of Steel run. sub-S06 uses its ses-4 events (its BOLD
  run is in ses-4; the extra ses-3 events file is ignored).
- Input image: `derivatives/preprocessing/.../*_task-TearsOfSteel_space-MNI_desc-ppres_bold.nii.gz` (FSL FEAT: MCFLIRT motion
  correction, BET, 6 mm FWHM smoothing, grand-mean scaling, high-pass sigma 50 s; then WM, CSF and six motion regressors removed, and
  registration to MNI). No further nuisance regression is added.
- Projection: fsaverage5 mid-depth surface sampling exactly as the LPP harness (nilearn `vol_to_surf`, kind ball, radius 3 mm, linear
  interpolation, mid meshes between pial and white), then Schaefer-400 17-network parcels on fsaverage5.
- TR 1.3 s from the BOLD sidecar. Slice timing: the derivative is in MNI space, so per-vertex slice times cannot be recovered; every
  vertex uses the run's mean SliceTiming as one offset.

## Timing (no lag search)
- Film time of volume j: j x TR + slice offset - film onset, where the film onset is the run's own film row in its events file.
- Scored volumes: film time from 10 s (the first 10 s of the film are dropped) to the end of the 588 s cut, and only volumes whose
  TRIBE sample (film time - 5 s, TRIBE's offset) falls inside the prediction.
- TRIBE v2 stock predictions (the only model): full (video + audio + text; words and timings from the harness speech-to-text pass on
  the cut's own audio) and audio-only, 1 Hz, linear between samples, read at film time - 5 s.
- Word rate: per-second word counts from the same speech-to-text words, convolved with the SPM HRF, read at film time.

## Motion and quality
- FD (Power): from the six MCFLIRT parameters (rotations in radians times 50 mm, translations in mm). Source: the session's FEAT motion
  file (`task-mc_..._desc-motion_bold.tsv.gz`), rows of the Tears of Steel run, located by acquisition order and volume counts of the
  session's runs (from the raw sidecars). If the rows cannot be located unambiguously, frames are not censored and run exclusion uses
  MRIQC's `fd_mean` only; that case is reported per viewer.
- Run exclusion as D7: mean FD <= 0.5 mm and at most 30% of scored frames with FD > 0.5 mm. Censoring: frames with FD > 0.5 mm are
  dropped before scoring.

## Gauges (per viewer)
- Cleaning: detrend, Butterworth high-pass 0.03 Hz, z-score, on the scored, uncensored frames (the LPP stage harness).
- HEARING: AUD parcels (SomMotB_Aud, both hemispheres), Fisher-z mean of r(BOLD, audio-only prediction).
- LANGUAGE: LANG parcels (left TempPar, DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl), Fisher-z mean of the partial r(BOLD, full
  prediction) given the LANG-mean audio-only prediction and HRF x word rate.
- VISUAL: visual parcels (VisCent and VisPeri, both hemispheres), Fisher-z mean of r(BOLD, full prediction), only if the TRIBE video
  path runs on the cut (probe 1a); otherwise reported as "not run".
- Null: the same score over circular shifts of the prediction by 15 .. T - 16 frames; z = (real - mean) / SD; detected when real is
  above the null's 95th percentile.

## Group results (all exploratory)
- Per gauge: median z over viewers with a 95% viewer bootstrap CI (10,000 resamples, `numpy.random.default_rng(20261028)`), and the
  share of viewers detected.
- Volunteer comparison: ISC(k) for k = 1 .. 15: each viewer against the mean of k other viewers' cleaned series (20 random draws per k,
  seed 20261028), scored by the same gauge and null (LANGUAGE ISC partials the same controls). Because film onsets differ by up to
  about 1 s, each viewer's cleaned parcel series is first resampled by linear interpolation onto a common film-time grid (10 s to the
  end of the cut, steps of one TR); censored frames are missing there and ignored. TRIBE's per-viewer z values are placed on that
  curve with the section-8 volunteer-equivalent rule (mean z and the lower end of its 10,000-resample viewer bootstrap CI, rounded down).
- What counts: no pass or fail bar beyond "the group median z CI excludes 0" per gauge. The write-up names this rule and its commit.
