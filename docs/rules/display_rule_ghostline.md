# Ghost-line display rule (exploratory display; committed before any real trace is drawn or extracted)

Cloud plan v2 (Oct 29, 16:20 IST), item 2. One picture repeats on the page and in the video: the ghost line. TRIBE v2's forecast is
drawn first as a dashed gray line, and real public listeners are drawn over it as solid coloured lines. This rule fixes every choice
behind that picture before any listener's series is extracted for it. It is a display rule only: it adds no statistic, and no frozen
number is recomputed or restated differently.

## What is drawn (LPP, the one ghost line with real listeners)

- Data: LPP English section 8 (ds003643), the 49 listeners the frozen held-out proof used, from the parcel series that proof already
  read (analysis folder `p1/cache_en8`, listeners `sub-EN057` to `sub-EN115`) and the stock TRIBE v2 forecast it already used
  (`k2/out_en8/k2_out/pred_en8_full.npz`). No new download and no other data.
- Region: the frozen proof's language parcels, exactly as `stillhere/heldout.py` `parcel_sets` defines them: Schaefer-400
  17-network labels in the left hemisphere whose names contain TempPar, DefaultB_Temp, DefaultB_PFCv or DefaultB_PFCl (atlas
  `atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot`). Not redefined.
- Segment: the proof's scored window, volumes 10 to 154 (145 volumes, TR 2.0 s, scan time 20 to 308 s), because it is exactly what
  the "about 5, at least 2" number was measured on. Short view (video cold open and small tiles): volumes 10 to 69, the first 120 s of
  that window, cropped from the full-window lines after cleaning and z-scoring, never re-cleaned or re-scaled.
- TRIBE lag: the frozen proof's alignment, `heldout.align`: volume j reads the forecast for stimulus time 2j - 5 s (TRIBE's own 5 s
  offset), linear between its 1 Hz samples. The proof's -1.25 s lag belongs to the word-rate and envelope regressors, which are not
  drawn.
- Cleaning: the proof's `heldout.clean` on volumes 10 to 154 for every parcel row (linear detrend, 0.03 Hz Butterworth high-pass,
  z-score with ddof 1). Forecast rows are the per-parcel vertex means from `heldout.pred_parcels`, cleaned the same way.
- One line per source: the mean over the language parcels of the cleaned rows, then z-scored over the window (ddof 1).
- Smoothing: none. Lines join the volume points; no spline, no filter beyond the proof's cleaning.
- The 49-listener line: the mean of the 49 listeners' lines, z-scored over the window. Labelled "49 public listeners (average)".
- The 5-listener line: the mean of 5 listeners' lines, z-scored over the window. The 5 were drawn from the 49 IDs sorted as strings
  (file names only, no data read) with numpy `default_rng(20261029).choice(ids, 5, replace=False)`, numpy 2.3.5, before any series
  was extracted: `sub-EN068`, `sub-EN079`, `sub-EN091`, `sub-EN113`, `sub-EN114`. They may include the single listener below.
  Labelled "5 public listeners (average)". Drawn only where the "about 5" line is on screen.
- The single listener: the listener with the median frozen LANGUAGE z of the proof's primary run (stage `145_primary`,
  `per_listener.language` in `d22/sec8/heldout_section8.json`, SHA-256 73aa5139b5057f00..., the source of `results/section8.json`):
  the 25th of the 49 in ascending order, ties broken by ID. Labelled "1 public listener", with no verdict word and no number. Its ID is
  recorded in the data file for reproducibility only and never shown.
- Y axis: z units, unlabelled numbers allowed; X axis: seconds of the section's scan. Title on screen: "Le Petit Prince, section 8
  (LPP English)". No other Le Petit Prince text (no words, transcript or captions of the story) appears on screen.

## Draw order (always, every scene and tile)

1. The gray dashed forecast, in full for the segment shown, before anything else.
2. The trust chip.
3. Solid coloured listener lines: the 49-listener line; the 5-listener line only where the "about 5" line is shown.
4. The thin 1-listener line, last.

A scene may stop after any step but never reorders them.

## Numbers on screen

Only the frozen numbers, copied from `results/section8.json` and never recomputed: "~5 volunteers' worth (at least 2). Group level."
(volunteer equivalent 5.0, lower bound 2.58, language areas). This line appears only beside the LPP ghost line. It never appears next
to a live or cached laptop forecast or any other clip.

## Trust chips (exact text)

Short chip on the line; long chip as its expansion or footnote.

| Line | Short chip | Long chip |
|---|---|---|
| LPP section 8 | Checked: held-out LPP | held out; first measured Oct 22-23; LPP is one of TRIBE v2's own evaluation sets |
| Story wall | Story wall: rule 340b74b | scoring rule committed Oct 28 · 340b74b; 41 of the 42 stories were used before then, for development or to tune a second model |
| notthefall (D20) | notthefall: rule 340b74b | scoring rule committed before these scans were read · 340b74b |
| Film (Emo-FilM) | Film: rule {film_commit} | rule committed before any viewer's scan was read · {film_commit} |
| Live or cached laptop forecast | Forecast only · audio-only, laptop | No public scan of this clip. Forecast, not a scan. |
| Live or cached laptop forecast, alice_ch01 (public_scan: the Alice Datasets) | Forecast only · audio-only, laptop | A public scan of this chapter exists (Alice Datasets, OpenNeuro ds002322); not compared here. Forecast, not a scan. |

How each chip was checked (dates in UTC; commits are in the public repository):

- LPP: section 8 was first measured Oct 22-23, before this repository began (Oct 27), so the chip makes no rule-before-scans claim.
- Story wall: the picker's scoring rule (PREREG chain head 31085486, with section 9 and D19 to D27) entered the repository in
  340b74b (Oct 28 10:02). The picker's own download pass reads the scans from Oct 29 01:01, but 41 of the 42 picker stories are
  listed in `picker_config.json` as used earlier (`touched` for development scoring, or in `tuned_training_clips`, whose scans were read
  Oct 24-25 to tune the second model); only lppFR_section8 is in neither. So the chip states the date instead of a before-read claim.
- notthefall (D20): the scoring code (MANIFEST 92efa035) entered the repository in 340b74b (Oct 28 10:02); notthefall is on the
  picker's `never` list, was held out of tuning, and its scans are read only by the stillhere-d20-score run (from Oct 29 16:00). The
  chip is used only if that run's start time, from its Kaggle log, is after 340b74b; otherwise it reads "scoring rule committed
  Oct 28 · 340b74b; scans read <date>".
- Film: {film_commit} is the last public commit touching the rule that scores the drawn result (`docs/rules/emofilm_rule.md` for the
  intact film, `docs/rules/face_map_rule.md` for the face test) before the first viewer scan was opened, as the scoring runner's read
  log records. The chip is used only if that log shows no scan opened before {film_commit}; otherwise it states both dates. One
  session motion file was opened for its shape before the read guard existed (disclosure 8); it is not a brain scan and is disclosed
  where the chip expands.
- Never "before anyone looked", and never any chip text not in this table.

## Other lines

- Story wall and notthefall tiles use only the outputs their frozen scoring runs committed on Thursday or later in that run's own files
  (`tierD_picker_wall.json`, `tierD_d20_score.json`). No new laptop-forecast-versus-scan comparison is computed for them, and no ghost
  line is drawn for them unless the frozen output itself holds the series.
- Live or cached laptop forecasts (the judge's own clip, condition and seed): the gray dashed line alone, with the chip "Forecast only ·
  audio-only, laptop" and the line "No public scan of this clip. Forecast, not a scan." when the clip's public_scan field in
  stillhere/live_menu.json is null; for alice_ch01, a public fMRI stimulus (public_scan: the Alice Datasets), the line is "A public scan
  of this chapter exists (Alice Datasets, OpenNeuro ds002322); not compared here. Forecast, not a scan." (amended under the cloud plan
  of Oct 29, 19:20 UTC, item 1a; a display line only, no statistic). No coloured line and no volunteer number. The
  notthefall pair may sit beside it as a separate tile with "Real listeners, different story."

## Film ghost line (Emo-FilM, Tears of Steel; added Oct 29 before any Emo-FilM brain data was read)

Added in a separate commit after the LPP part, before the LPP lines were extracted and before the read guard can open any Emo-FilM
scan (Saturday at the earliest). The commit time is the record. Every choice comes from `docs/rules/emofilm_rule.md` and
`docs/rules/face_map_rule.md`; nothing here is a new test, and the picture adds no number.

- Region: the intact film rule's VISUAL parcels, VisCent and VisPeri in both hemispheres (`emofilm_score.visual_sets`). Why: the
  ghost line is a time course of forecast and viewers, and VISUAL is the committed film gauge that compares TRIBE's full forecast with
  the viewers' time courses on the video path, where faces enter the model. The face test's 374 parcels define a map across parcels
  (R, C and M), not a time course, so drawing a time course on them would be a new analysis; HEARING and LANGUAGE are the audio and
  speech gauges. If the intact film score reports VISUAL as "not run", no film ghost line is drawn.
- Grid: the intact rule's common grid (`emofilm_score.common_grid`): film time 10 to 185 s in steps of one TR (1.3 s), 135 points.
  Always the whole grid; no crop or zoom.
- Forecast line: TRIBE v2 stock full forecast of the intact film (the committed intact kernel's `pred_full.npz`), each VISUAL parcel
  the mean of its vertices read at grid time - 5 s (`emofilm_score.tribe_at`, TRIBE's fixed offset, no lag search), cleaned on the
  grid with `emofilm_score.clean` (detrend, 0.03 Hz high-pass, z-score), as the rule cleans its grid controls; then the mean over the
  parcels, z-scored over the grid (ddof 1).
- Viewer lines: each scored viewer's VISUAL rows exactly as the intact scorer makes them for its ISC curve (`clean_full_then_crop`,
  then `to_grid` onto the same grid; censored or uncovered points are missing). Per viewer: the mean over the parcels, z-scored over
  that viewer's non-missing points (ddof 1). Missing points stay gaps; nothing is interpolated across them.
- Group line: the mean of the scored viewers' lines at each grid point (missing points left out), z-scored over the grid. Labelled
  "{N} public viewers (average)", N the scored count from the score output (28 only if all 28 kept runs are scored).
- Single viewer: the scored viewer with the median frozen VISUAL z in the score run's per-viewer file (`score_per_viewer.json`,
  `gauges.VISUAL.z`): the ((N - 1) // 2 + 1)-th in ascending order (the lower middle when N is even), ties broken by run ID.
  Labelled "1 public viewer", no verdict word, no number; its run ID is kept in the data file only.
- Face ticks: the face rule's F, the share of the whole 1280 x 534 frame covered by faces (`edits_v2.json` field `px`, the final
  film stimuli v2), averaged per film second, before any HRF. One tick per film second from 0 to 185 s where the share is above 0,
  its height proportional to the share, drawn under the lines on the same film-time axis without shifting. Label: "Ticks: share of
  the frame covered by faces, each second". The lines are at scan time, so a response follows its ticks; the label says nothing more.
- No smoothing, no other line. The faces-removed and same-size-control forecasts are never drawn against viewers (nobody watched
  those versions). No volunteer number beside the film ghost line unless it is the VISUAL gauge's own entry in the score output,
  copied; never another gauge's.
- Caption: by default only the ticks label above and the chip. The video beat's proposed caption "brain signal rose with on-screen
  face area" is a statement about the brain data, so this display rule does not license it; it appears only if a committed result
  states it (a ruling decides which), never from the picture.
- Draw order and chip: as above (gray dashed forecast in full, then the chip, then the group line, then the thin single viewer; the
  ticks are drawn in step 1 with the forecast, below it). Chip: short "Film: rule {film_commit}", long "rule committed before any viewer's scan
  was read · {film_commit}", where {film_commit} is the last public commit touching `docs/rules/emofilm_rule.md` before the first
  Emo-FilM scan was opened (b670da6 as of this commit), checked against the prepare stage's read log as in the chip table; this
  section's own commit must also predate that read.
- Data: `results/ghostline_film.json`, written only after the intact film score exists, from its outputs and the prepare cache; it
  holds the four series, the grid, the labels, the chip, N, and the commits of this rule and the film rule.

## Data file and checks

`scripts/ghostline_lpp.py` writes `results/ghostline_lpp_s8.json` (the four lines on the full window, the short-view index range,
times, labels, chips, the draw order, the quoted frozen line and its source, this rule's commit) and a local reference render outside
the repository. It refuses before 2026-10-29 16:00 UTC (the D22 rule forbade printing parcel series before Thu Oct 29 noon ET), if
this rule file has uncommitted changes or no commit, or if the frozen per-listener file's SHA-256 differs. It prints only counts,
shapes and timings. `tests/test_ghostline.py` checks the data file against this rule (draw order, chip text, labels, short view = crop,
z-scoring, the five IDs, no banned words, no verdict words on the single listener).
