# Rule: TRIBE v2 forecast vs Emo-FilM viewers, Tears of Steel opening bridge conversation (StillHere Pilot, exploratory)

Fixed on 2026-10-28 and committed with its scoring code (`stillhere/emofilm_score.py`: `window_mask`, `window_score`,
`bridge_viewer_scores`, `frame_stats`, `per_second`, `tribe_beats`) BEFORE any Emo-FilM brain data was downloaded or read. The commit
time is the record. It amends `emofilm_rule.md` (commit 2762956) for this window only; everything not stated here follows that rule.
Every number it produces is labelled "exploratory". Nothing here is revised after the BOLD is read; a later change is reported as a
change, with both results.

## Window
- The opening bridge conversation: release time 25.00 to 40.25 s of `tears_of_steel_720p.mov` (SHA-256 efa9062d...), frames 600 to 966
  at 24 fps. The Emo-FilM cut starts at release 0, so film time equals release time.
- Source of the bounds: the Emo-FilM metadata has no scene boundaries (FilmInfo.txt lists licences only, the cut file gives trim points
  only, the data descriptor has no scene table). The bounds are the shot cuts found by ffmpeg scene detection on the official release
  (cuts at 18.42, 25.00, 40.25 s at threshold 0.25; no other cut inside the window at threshold 0.08), checked by eye at one frame per
  second: one continuous shot of two people talking on a canal bridge, one of them holding a mechanical prop; no combat, weapons or
  fighting on screen. Before it: an establishing skyline with a funding credit; after it: the "40 years later" hologram sequence.
- Scored volumes: those whose TRIBE sample (film time - 5 s) lies in 25.00 to 40.25 s, so film time 30.00 to 45.25 s: 11 or 12 volumes
  per viewer at TR 1.3 s. Scan time = film onset (the viewer's own events file) + film time; median window 118.89 to 134.14 s after run
  start (start range 118.61 to 119.58 s). Per-viewer values: `pilot/film/bridge_window.json` (outside the repo).
- This is a short window: about 12 volumes per viewer. The results are descriptive; low power is stated next to every number.

## Viewers and parcels
- All 30 ds004892 viewers, their Tears of Steel run; sub-S06 uses its ses-4 events (onset 93.898 s); the base rule's motion rules and
  run exclusion (D7) apply to the whole run.
- Parcels (Schaefer-400 17-network on fsaverage5, as the base rule): AUD (SomMotB_Aud), LANG (left TempPar, DefaultB_Temp,
  DefaultB_PFCv, DefaultB_PFCl), VISUAL (VisCent, VisPeri) when the TRIBE video path runs (probe 1a: it runs on one T4).

## Stimulus versions, in this order
1. intact: the release frames and audio as scanned.
2. faces removed: every face in the window removed with SAM 3.1 masks (concept "face"), inpainted or filled with the local mean.
3. area matched: per frame, the same masked area removed from a non-face region of the same frame (same fill).
The video and every input stay at V-JEPA2's frame sampling and resolution (64 frames per 0.5 s step at 256 px, features at 2 Hz) and the
release's own frame rate and size: no change of frame rate or resolution, because they define TRIBE's inputs. Versions 2 and 3 wait for
SAM 3.1 access (requested; Meta approves by hand). The brain data are the same for every version (the viewers saw the intact film), so
versions 2 and 3 test how much TRIBE's agreement with the viewers depends on the faces.

## Scoring (per viewer, per version)
- BOLD is cleaned over the whole run's kept scored frames (the base rule: detrend, 0.03 Hz high-pass, z-score); the window frames are
  then cut from it. Regressors are only z-scored inside the window (about 12 frames is too few to high-pass).
- TRIBE gauges as the base rule: HEARING (AUD vs audio-only prediction), LANGUAGE (LANG vs full prediction, partialling the LANG-mean
  audio-only prediction and HRF x word rate), VISUAL (VIS vs full prediction).
- Null (wrong segment): the same score with the viewer's BOLD from every other run of the same number of consecutive kept frames whose
  first film time is at least 15 s from the window's, against the same window regressor; z = (real - null mean) / null SD; detected
  when real is above the null's 95th percentile.
- Baselines, each scored as one regressor against AUD, LANG and VIS with the same cleaning and null: HRF x word rate (words from the
  harness speech-to-text pass on the forecast clip's audio), HRF x sound envelope, HRF x frame brightness (mean Rec. 709 luma per frame),
  HRF x frame motion (mean absolute luma difference from the previous frame). Brightness and motion use the release's native 24 fps
  frames at native size, the frames V-JEPA2 samples from, averaged per second; every baseline is convolved with the SPM HRF at 1 Hz and
  read at film time (as the base rule's word rate). For versions 2 and 3, brightness and motion are recomputed on that version's frames.

## Group results (all exploratory)
- Per gauge and version: median z over viewers with a 95% viewer bootstrap CI (10,000 resamples, `numpy.random.default_rng(20261028)`),
  share detected; the same for every baseline.
- Version comparison: viewer-paired Hodges-Lehmann median of the z difference (faces removed minus intact; area matched minus intact;
  faces removed minus area matched) with the same bootstrap; plus the forecast-level change (mean absolute difference of TRIBE's
  predicted VISUAL, AUD and LANG series between versions, no BOLD).
- The pitch claims only what TRIBE beats: TRIBE beats a baseline on a gauge only if its group median z is higher AND the viewer-paired
  Hodges-Lehmann median of (TRIBE z - baseline z) has a 95% bootstrap CI above 0 (`tribe_beats`). Where TRIBE does not beat a baseline,
  the write-up says so and no line claims it.
