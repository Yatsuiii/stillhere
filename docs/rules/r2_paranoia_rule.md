# R2 rule: stock and fine-tuned TRIBE v2 on Paranoia (Finn et al. 2018), the whole story (exploratory)

Cloud rulings of Oct 31, 08:45 UTC (R2), 20:15 UTC and 21:30 UTC (R2 items 1-3); Yatsu said "Run it" (Oct 31). Committed and pushed
before any Paranoia brain data is downloaded or read and before any Paranoia forecast exists; the commit time is the record. Already
on origin before it: the development records (`r2/harness_dev/`) and the data check (`r2/data_check.py`, `r2/data_check.json`), both
in commit 52424ff. Everything below is fixed by this commit; a change is a new commit that says what changed and why, made before the
step it changes runs. Order: this rule; a code commit (runner, scorer, synthetic tests); the forecasts on the laptop; a commit with the
forecasts', the word table's and the surprisal file's SHA-256 and nothing else; the cloud's go; the download on Colab; one scoring run;
the results commit. The result is shown whatever it is, labelled exploratory, one line per model.

Adopted for R2 from R3: none (`results/r3_variants.json`).

## 1. What is tested

How many real listeners TRIBE v2's language-area forecast is worth on a story outside its training and evaluation sets, and what share
of the other listeners' average it reaches, for stock TRIBE v2 (primary, D3) and fine-tuned TRIBE v2 (reported on its own line, never
replacing the stock line). Alice's rule (`docs/rules/alice_heldout_rule.md`) is the template; this section and the next list every
difference.

TRIBE v2 was trained on CNeuroMod, BoldMoments, Lebel2023 and Wen2017 and evaluated by its authors on NNDb, LPP, Narratives and HCP
(d'Ascoli et al. 2026, Table 1); the fine-tune (`docs/rules/alice_finetuned_rule.md` section 1, weights 96923860...) trained on LPP
sections 2-7 and 9 and Narratives. Paranoia (ds001338) is in none of these lists.

## 2. Data (from `r2/data_check.json`; headers and sidecars only)

- OpenNeuro ds001338 v1.0.0 (Finn, Corlett, Chen, Bandettini and Constable, Nature Communications 2018; DOI
  10.18112/openneuro.ds001338.v1.0.0), licence CC BY-NC 4.0, cited as asked. An original audio story in three parts (story1, story2,
  story3), one run each, heard by 22 listeners (all 22 in participants.tsv; the dataset documents no exclusion).
- Brain data: the fMRIPrep 22.0.2 derivative (OpenNeuroDerivatives ds001338-fmriprep, its source the v1.0.0 DOI), the fsaverage5
  GIFTI files `sub-*_task-story{1,2,3}_hemi-{L,R}_space-fsaverage5_bold.func.gii` as published: 132 files, 2.52 GB, each 10,242
  vertices (CortexLeft, CortexRight), joined left then right as TRIBE orders them. Volumes 526, 452 and 332 per story, the same in
  every listener and hemisphere; TR 1 s; not slice-time corrected (sidecars).
- Differences from Alice's template, all fixed here:
  - The projection becomes the identity: the files are already on fsaverage5. fMRIPrep sampled each listener's own cortical surface
    and resampled it to fsaverage5; Alice's volumes were projected onto the template's surface (3 mm ball).
  - No spatial smoothing in the files (Alice's derivative was smoothed 3 mm), no slice-timing correction, and none of fMRIPrep's
    confound regressors are used: section 8's cleaning is the only cleaning, as it was for LPP and Alice.
  - TR 1 s instead of 2 s, so every volume parameter is converted to seconds (section 5).
- Disclosed: on Oct 31, before this rule, a data-check agent read the raw BOLD's AFNI headers, which include BRICK_STATS (the minimum
  and maximum of each raw volume); no forecast was compared with them, and the derivative is a different file. The data check behind
  this rule read GIFTI XML headers up to the first data array and the JSON sidecars only.

## 3. Stimulus, words and the forecasts

- Stimuli: `stimuli/story{1,2,3}_audio.wav` from the raw dataset, each heard in its own run (events.tsv: onset 2 s, durations 524, 449
  and 329 s). Their SHA-256 are pinned by the forecasts' commit.
- Words: the dataset has a full text (`stimuli/story_fulltext.txt`) but no word timings, so the words come from TRIBE v2's own
  speech-to-text (`ExtractWordsFromAudio`, as tribev2 af58661 ships it), run once per story inside `tribe_common.build_events` (the
  training order that avoids the over-60 s duplication), as the film forecasts did. Checks, each recorded: no exact-shift copies, last
  word before the audio's end; descriptive only: the word count against the full text's (counts, never text). That one word table per
  story feeds every forecast, the word-rate regressor and every word baseline; its SHA-256 is pinned by the forecasts' commit.
- Forecasts, per story: audio-only, full, and full with the word texts shuffled across their onsets (seed 7), for stock TRIBE v2 and
  for fine-tuned TRIBE v2 (18 in all), on the laptop (RTX 4050) with the Alice placement (sound extractor and brain model on the CPU,
  Llama 3.2 with a fixed 3 GiB GPU budget); section 8's harness (`k2/tribe_common.py`, stitched 50 s windows, fp16 text through
  accelerate). Stock first. The fine-tuned runs load the weights as R1 did (both pins, strict, at least one tensor changed, eval mode)
  and read every sound and Llama feature from the stock runs' cache (its key is the stage's `stim.wav` path): each must add no file to
  the cache, leave the work folder unchanged, and differ from stock. A run that fails technically is fixed and rerun; no forecast is
  compared with Paranoia brain data before the one scoring run.
- Surprisal: GPT-2 small by `p1/surprisal8.py`'s method on the word table, pinned by the forecasts' commit.
- Never during Sunday's filming or a chain step that uses the laptop.

## 4. Coverage

Per listener and run, a vertex is covered if its series is finite and not constant; the medial wall (Schaefer-400's unlabelled
vertices) is excluded; a vertex counts as covered for a listener only if it is covered in all three runs. Then Alice's rule unchanged:
an area counts for a listener only if at least 90% of its vertices are covered (otherwise that listener leaves every score for that
area and the claim); a parcel is used only if every listener scored for its area has at least one covered vertex in it; a parcel's
series is the mean over its covered vertices; fewer than 10 listeners for LANGUAGE: "rule committed, not run (coverage)".

## 5. Timing and the one window (seconds; each conversion listed)

- Run r's volume j is at scan time j s; story time 0 is scan time 2 s (each run's events.tsv); TRIBE's 5 s offset is unchanged, so at
  lag L volume j reads story time j - 2 - 5 - L s of run r's forecast (linear between 1 Hz samples).
- The one primary window: the whole story. Per run, volumes from 20 s (section 8's first 10 volumes at TR 2 s = 20 s dropped; at TR 1 s,
  the first 20 volumes) to the last volume with brain data and a forecast for every listener; each run is cleaned on its own
  (`heldout.clean`: detrend, 0.03 Hz high-pass at t_r 1 s, z-score), then the three runs are joined in story order (about 1,250
  volumes, 20.8 minutes). No 145-volume or 290 s line.
- Conversions from section 8 (TR 2 s) to TR 1 s: first scored volume 10 -> 20 (20 s); circular-shift null minimum 15 TRs -> 30 TRs (30
  s), shifts on the joined series; development lag -1.25 s and the envelope's own-best-lag search range (-14 to +14 s) are already in
  seconds; the high-pass 0.03 Hz is in Hz; regressors are made at 1 Hz and sampled at each volume's story time, as in section 8; ISC
  draws (M1) use the same 30-TR shift minimum; listener counts k, repetitions (20) and seeds (11; 20261028) are not time and stay.

## 6. Scoring: section 8's gauges, Alice's per-listener group step

- Per listener: LANGUAGE z (the partial r of the 19 LANG parcels' full forecast given the LANG-mean audio-only forecast and the
  word-rate regressor and its derivative at L, against the circular-shift null) and HEARING z (the 7 AUD parcels, audio-only forecast,
  lag 0), at the development lag (primary) and at the envelope's own best lag on these data (sensitivity); the shuffled-word forecast
  as section 8's control (real above shuffled, Wilcoxon).
- Group step and claim as Alice's (`scripts/alice_heldout.py`): means, shares above section 8's bar, Wilcoxon tests, the best word
  baseline (log word frequency within the story, GPT-2 small surprisal; no parse-tree baselines, the dataset has no parses), the cheap
  baselines (word rate and the sound envelope, `docs/rules/cheap_baselines_rule.md`), M1 (ISC(k) for k in 1, 2, 3, 4, 6, 8, 12, 15 below
  n - 1, plus n - 1; 20 repetitions; seed 11) and `volunteer_equivalent` (10,000 bootstrap resamples, seed 20261028), rounded down at
  the lower 95% bound.
- The leave-one-out reference: per listener, the other listeners' average joined series (all scored for LANGUAGE but that listener)
  used as the forecast, with the same gauge, controls and null. Share = the model's mean LANGUAGE z divided by the reference's mean
  LANGUAGE z over the same listeners, as a percentage (both means reported).
- z depends on the scan's TR and length, so z is not compared across datasets with different TRs; the share is the cross-dataset
  number. Section 8's M2 and M3 are not run.

## 7. Lines (written before the result)

{model} is "stock TRIBE v2" or "fine-tuned TRIBE v2"; fine-tuned lines start "Fine-tuned TRIBE v2: ". {n} counts the listeners scored
for LANGUAGE, {T} and {M} the scored volumes and minutes (PREREG's minutes rule); numbers are copied from the output.

- Claim at least 1: "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; {n}
  listeners, the whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast was worth about {point} real volunteers, and at least {claim}. It reached
  {share}% of what the other listeners' average reaches."
- Claim 0, LANGUAGE p < 0.05: "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018;
  {n} listeners, the whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast tracked the listeners' language areas (z {z}, p = {p}), but at its
  lower bound it was worth less than one real volunteer (about {point} at the point estimate). It reached {share}% of what the other
  listeners' average reaches."
- LANGUAGE p of 0.05 or more: "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018;
  {n} listeners, the whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast did not reach the bar in language areas (z {z}, p = {p}); it reached
  {share}% of what the other listeners' average reaches. We show it because we said we would."
- Coverage: "Paranoia: rule committed, not run (coverage): {n} listeners had the language areas in view, fewer than 10."
- Not on origin/main by Mon Nov 2, 12:00 UTC: "Paranoia: rule committed, not run."
- Beside any scored line: "What it had to beat: word rate and the sound envelope (beaten in {k} of {K} comparisons)."
- Beside the pair: "z is not compared across datasets with different scan timing; the share is."

No other line describes this result. Not a diagnosis or medical advice.

## 8. Cutoff

On origin/main by Mon Nov 2, 12:00 UTC, or the page and README list "rule committed, not run".
