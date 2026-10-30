# Alice held-out rule: TRIBE v2's forecast against the Alice Datasets listeners (exploratory)

Cloud rulings of Oct 30, 16:05, 16:10 and 16:20 UTC (item E). Committed and pushed before any Alice BOLD is downloaded; the commit
time is the record. Everything below is fixed by this commit; a change is a new commit that says what changed and why, made before the
step it changes runs. The second commit, still before any BOLD, added the forecasts' and the surprisal file's SHA-256, the forecast
runner, and the SHA-256 of every cited file kept outside the repository (cloud ruling of Oct 30, 18:20 UTC). The result is shown
whatever it is, labelled exploratory.

## 1. What is tested

Section 8's claim, unchanged, on a different public audiobook, outside TRIBE v2's training and evaluation sets: how many real
listeners stock TRIBE v2's language-area forecast is worth (`stillhere/heldout.py` `volunteer_equivalent`), with the same areas, lag,
cleaning, nulls, baselines and claim rule as LPP section 8 (`HELDOUT.md`, Oct 23; `scripts/heldout_section8.py`).

TRIBE v2 was trained on CNeuroMod, BoldMoments, Lebel2023 and Wen2017 and evaluated by its authors on NNDb, LPP, Narratives and HCP
(d'Ascoli et al. 2026, Table 1). The Alice Datasets are in neither list.

## 2. Data

- OpenNeuro ds002322 v1.0.4, the Alice Datasets (Bhattasali, Brennan, Luh, Franzluebbers and Hale, LREC 2020; DOI
  10.18112/openneuro.ds002322.v1.0.4). Licence CC0. Cited as its HowToAcknowledge asks.
- Listeners: the 23 with a preprocessed derivative: sub-18, 22, 23, 24, 26, 28, 30, 31, 35, 36, 37, 39, 41, 42, 43, 44, 45, 47, 48,
  49, 50, 51, 53. The dataset already excludes 3 of the 29 scanned (1 head motion, 2 quiz performance). sub-38, sub-46 and sub-52
  have no file in `derivatives/` (no reason documented) and are not scored.
- sub-24's events.tsv names `wonderland_ch_01.mp3` instead of the stimulus below, with the same 748 s duration as every other
  listener; the README describes one stimulus for all, and the 748 s matches the slowed file, not the original-speed reading. sub-24
  is scored as hearing the stimulus below; the results file reports the mismatch.
- BOLD: `derivatives/sub-XX/sub-XX_task-alice_bold_preprocessed.nii.gz` as published (SPM8: realigned, coregistered, smoothed 3 mm,
  normalised to MNI with the ICBM template, high-pass 1/128 Hz, first 10 volumes discarded). TR 2.0 s. No other preprocessing; section
  8's cleaning applies on top, as it did for LPP.
- Protocol, reported as a column, pooled, no extra analysis: single-echo EPI, 44 slices, TE 27 ms (11: sub-18 to sub-37); three-echo
  EPI, second echo, 33 slices, TE 27.5 ms (12: sub-39 to sub-53).

## 3. Stimulus, words and the forecasts (fixed before any BOLD)

- Stimulus: `stimuli/DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav` (Kristen McQuillan's LibriVox reading of chapter 1, slowed
  20% with PSOLA, normalised to 70 dB; 743.66 s, 44.1 kHz mono), SHA-256
  bb8003578d0667cd45fbce932c45dcfa3abee4be40da98c0a83dea80516aeefe. What the listeners heard; never our menu clip.
- Words: the dataset's own timings, `code/annotations.tsv` (SHA-256
  6e25b0d85a1457a75c608f2a298dbef97172f83270eb35a82b43c92229d0e46c), as section 8 used the LPP TextGrid: 2,101 timed rows; the 31
  rows without a timing are dropped from TRIBE's input and from every word-based regressor, and counted; a duration below 0.01 s (one
  row ends before it starts, one has zero length) is set to 0.01 s. The timings were checked against the audio (first word onset
  20.12 s = first speech energy 20.12 s; the best fit is at shift 0).
- Forecasts: section 8's harness (`k2/tribe_common.py`, tribev2 af58661, the stock checkpoint, the same three runs as en8: audio-only,
  full, and full with the word texts shuffled across their onsets, seed 7; stitched 50 s windows; text lane in fp16 through
  accelerate), run by `alice/forecast/run_alice_forecast.py` on the laptop GPU (RTX 4050, 6 GB). Placement only, as the live worker
  does on the same GPU (`worker/engine.py`): the sound extractor and TRIBE's brain model run on the CPU, and Llama 3.2's layers get a
  fixed 3 GiB GPU budget with the rest on the CPU (the worker's pinned loader), recorded with each run. The runner is in the repository;
  its launcher `alice/forecast/run_all.sh` (offline flags, the tribe environment) is kept outside the repository, SHA-256
  e3574a28679a139fd1de4b0b55e027e266ac73af6ee9f53cfef7fdc46298ccdf.
- Forecasts (`alice/forecast/out/pred_alice_{audio,full,shuf}.npz`, kept outside the repository; each 744 by 20,484, finite, made
  on Oct 30 by the committed runner), SHA-256: audio d747d74f16231b249e1288da14f48ed0b57e434bd356edfb2836cdc7f166859b, full
  0a47d825a521f643fc5b298942fde7049679b070381bc97330ffd11a25d30ae3, shuffled
  7c4ba627bc0b6d801e263fa17d887f18a4028dec84fbec9850c750a4d66f41bf. The first full and shuffled processes wrote all 2,101 words'
  Llama features to the text extractor's cache with the pinned loader, then were stopped when the laptop ran out of system memory
  while loading the sound extractor; the reruns read those features from the cache, as each stage file records ("read from the
  infra cache"). Nothing else changed; the audio run reproduced its earlier output byte for byte.
- Surprisal: GPT-2 small by `p1/surprisal8.py`'s method on the timed words (`scripts/alice_heldout.py surprisal`):
  `alice/surprisal.json`, 2,101 words, kept outside the repository, SHA-256
  ad737833b56128cb1b7963b76fdfd458639922717ba05f14158b449644f80366.

## 4. Volumes, surface and coverage

- Projection: section 8's LPP projection unchanged (`p1/hearing.py` `project`: nilearn `vol_to_surf` onto fsaverage5 between the
  pial and white meshes, 3 mm ball, linear), then the areas exactly as `heldout.parcel_sets` defines them (Schaefer-400, 17 networks).
- Coverage, from each listener's projected series only, before any scoring: a vertex is covered if its series is finite with
  non-zero variance. An area counts for a listener only if at least 90% of its vertices are covered: a listener below 90% for
  LANGUAGE is excluded from every language score and from the claim; below 90% for HEARING, from every hearing score. A parcel is used
  only if every listener scored for its area has at least one covered vertex in it; a parcel's series is the mean over its covered
  vertices. Each listener's covered share per area, and the parcels kept, are reported.
- Floor: if fewer than 10 listeners remain for LANGUAGE, nothing is scored and the result is "rule committed, not run (coverage)".
- Indexing: derivative volume i is raw volume i + 10. Series are placed in raw-volume indexing (volumes 0 to 9 empty and never read),
  so section 8's skip and windows are unchanged.

## 5. Timing

- Each listener's own events.tsv onset: stimulus time 0 is scan time `onset` (0.003 to 0.837 s across listeners). That listener's
  forecast and every stimulus-time regressor are shifted by its onset; TRIBE's own 5 s offset is unchanged (raw volume j reads
  stimulus time 2j - onset - 5 s, linear between 1 Hz samples).
- Windows: section 8's two, in raw volumes. Primary, the claim window: volumes 10 to 154 (145 volumes); it starts at raw volume 10,
  which is both section 8's skip point and the first kept volume. Full length: volume 10 to the last volume with BOLD and a forecast
  for every listener.

## 6. Scoring: section 8, unchanged

- `heldout.clean` (detrend, 0.03 Hz high-pass, z-score); `stage_scores` at the development lag -1.25 s (PRIMARY) and at the envelope's
  own best lag on these data (sensitivity, as section 8); circular-shift nulls of at least 15 TRs.
- M1: the ISC(k) curve on the same listeners, k in (1, 2, 3, 4, 6, 8, 12, 15) below n - 1, plus n - 1; 20 repetitions; seed 11.
- Claim: `volunteer_equivalent` (10,000 bootstrap resamples over listeners, seed 20261028) of the primary-window LANGUAGE z, rounded
  down at the lower 95% bound, as section 8.
- One listener at a time: because the onsets differ, section 8's per-listener functions are called on each listener with its own
  shift, and the group step (means, the shares of listeners above section 8's bar, Wilcoxon tests, the best baseline, the own-best lag, the ISC draws, the cheap
  baselines' comparisons) is formed with section 8's formulas; with equal onsets it reproduces section 8's functions exactly
  (`tests/test_alice_heldout.py`).
- Baselines, defined as section 8 defined them: log word frequency within the chapter, from the timed words; parse-tree bottom-up and
  top-down node counts, labelled "parse from the dataset's annotations, not LPP's" (`CFG BU nodeCount` and `CFG TD nodeCount` in
  annotations.tsv); GPT-2 small surprisal (section 3).
- Cheap baselines (`docs/rules/cheap_baselines_rule.md`): word rate and the sound envelope, every predictor scored the same way, as
  `results/section8_baselines.json`.
- Not run: section 8's M2 (hybrid) and M3 (wrong story), because they need a second story's listener group and are not part of the
  claim.

## 7. Outcomes and their lines (written before the result)

{n} counts the listeners scored for LANGUAGE after coverage; numbers are copied from the output, never recomputed.

- Claim at least 1: "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets, {n}
  listeners), the forecast was worth about {point} real volunteers, and at least {claim}."
- Claim 0, LANGUAGE p < 0.05: "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets,
  {n} listeners), the forecast tracked the listeners' language areas (z {z}, p = {p}), but at its lower bound it was worth less than
  one real volunteer (about {point} at the point estimate)."
- LANGUAGE p of 0.05 or more: "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets,
  {n} listeners), the forecast did not reach the bar in language areas (z {z}, p = {p}). We show it because we said we would."
- Fewer than 10 listeners for LANGUAGE after coverage: "Alice Datasets: rule committed, not run (coverage): {n} listeners had the
  language areas in view, fewer than 10."
- Not on origin/main by Mon Nov 2, 12:00 UTC: "Alice Datasets: rule committed, not run."
- Beside any scored line, as section 8's page does: "What it had to beat: word rate and the sound envelope (beaten in {k} of {K}
  comparisons)."

No other line describes this result. Not a diagnosis or medical advice.

## 8. Code committed with this rule

`scripts/alice_heldout.py`, three steps: `download` (the 23 derivatives and their events.tsv, only once this commit is on origin and
the cloud has said go), `surprisal`, and `score` (one run). `score` writes the per-listener numbers outside the repository and
`results/alice_heldout.json` with the group numbers, the outcome line, the stimulus-file note, and per listener only the protocol,
onset and covered shares. `tests/test_alice_heldout.py` checks the indexing, the onset shift, the coverage rule and its exclusions,
the outcome lines, and the one-listener-at-a-time group step against section 8's functions, on synthetic data only; no Alice BOLD is
read by any test. `alice/forecast/run_alice_forecast.py`, the forecast runner, was added by the second commit.

Cited files kept outside the repository, SHA-256 pinned by the second commit:

- `k2/tribe_common.py` 5029b14736b31dd652f9f37ac0b25bc72e9cf978c4e1d6958f6847e4dd459a34
- `p1/hearing.py` 54459cc322c1505a1adcc1a73b95af72bf3ed68edc2ff651a455509e1bff73b6
- `p1/surprisal8.py` e69f6d842131483263bc54370ef5fb20acaba967e9e0f44fa420277875ad0543
- `HELDOUT.md` c88d68ead0e66f2c51f5479630478ee9a339cbce9a81b5ed04f780751588a40b
- `alice/forecast/run_all.sh` e3574a28679a139fd1de4b0b55e027e266ac73af6ee9f53cfef7fdc46298ccdf

## 9. Order and cutoff

This commit on origin; the next commit with the forecasts' and the surprisal's SHA-256; the cloud's go; the derivatives download; one
scoring run. The result is on origin/main by Mon Nov 2, 12:00 UTC, or the page and README list "rule committed, not run". The
forecasts never go ahead of the Saturday chain: they run on the laptop and finish before Sat 00:00 UTC, or they run after the chain's
local gate 0 step.
