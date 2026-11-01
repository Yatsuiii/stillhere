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

## 9. Dated amendment and clarification, Nov 1 2026 (Yatsu's decisions, before any Paranoia brain data is downloaded); sections 1-8 unchanged

Clarification, the forecast read (Yatsu, about 05:55 UTC). Section 5's sentence "at lag L volume j reads story time j - 2 - 5 - L s of run
r's forecast" is read as describing the regressors: the lag L moves only the word-rate regressor and its derivative, and every forecast is
read at story time j - 2 - 5 s. Reason: section 1 says this rule inherits section 8's gauges and Alice's template and lists every
difference, and it lists none for the forecast read; Alice, section 8 and the development scripts (`r2/harness_dev/`, dev_grid's best
forecast shift 0) all read it this way. `scripts/r2_paranoia.py` sets FORECAST_READ = "section8".

Amendment, where the steps run (section 3's "on the laptop (RTX 4050) with the Alice placement" and the order's "the download on Colab"):
- Speech-to-text. As tribev2 af58661 ships it (`uvx whisperx`, large-v3, float16, batch 16, the English align model), it runs out of GPU
  memory on the laptop's 6 GB GPU for a full story, even with the GPU otherwise idle (three tries 05:12-05:13 UTC, one at 05:19, and the
  bare command at about 05:21, out of memory after 21 s). It ran instead on a Kaggle T4 (16 GB): the kernel stillhere-r2-asr
  (`r2/kaggle_asr/`, code 61b891d0...), tribev2's own `ExtractWordsFromAudio._get_transcript_from_audio`, unchanged, each table written
  as the transform writes it; whisperx 3.8.6 (uvx's pick, recorded; the same version the laptop's cache holds). Words: 1,348, 1,215 and
  859 (`r2/kaggle_asr/r2_asr.json`, table SHA-256 1a98288a..., d3d07e7b..., db996815...). These are the stories' word tables.
- Stock forecasts. On the laptop, the full stage of story 1 ran out of RAM (the 7 GB cap set so the browser holding the Colab runs is never
  stopped; 05:51 UTC) and the laptop stages were stopped (Yatsu, 05:52 UTC). The nine stock forecasts run on Kaggle T4s instead, in two
  kernels (`r2/kaggle_forecast/`): stillhere-r2-forecast-a (audio-only and full, story1-3) and stillhere-r2-forecast-b (shuffled,
  story1-3, after part A). The stage code is `r2/run_r2_forecast.py`'s for a stock stage, its check functions embedded byte for byte;
  tribe_common as the D22 kernels ran it (build_events, load_model, apply_weights('stock') with the checkpoint pin, stitched_predict); the
  film kernels' package lock and Hugging Face pins. Placement: every model on the T4 (the text lane through accelerate), instead of the
  Alice placement (sound extractor and brain model on the CPU, Llama on a 3 GiB GPU budget); the features and the model are the same.
  Within a story the stages share one stim.wav path, so the sound features are computed once (part B loads part A's). The laptop's one
  finished stage (stock_story1_audio) is not used: all nine come from the same route.
- Fine-tuned forecasts: pending Yatsu's decision. The fine-tuned weights stay local, so that line runs on the laptop or not at all.
- Scoring: on Kaggle CPU instead of Colab (the download and one scoring run in the same kernel).

## 10. Dated amendment, Nov 1 2026 (cloud ruling of Nov 1, 06:05 UTC, items 6-10), before the forecasts' pins commit and before any Paranoia brain data is downloaded; sections 1-9 unchanged

The stock forecasts run on Kaggle T4s (section 9, c82cb3d); R5 (`docs/rules/r5_short_scan_rule.md`, b97e19d; scorer 16fee4d) reads the
same pinned stock forecasts and word tables and is scored in this rule's one scoring run.

Item 6, the forecast read (ruling). FORECAST_READ = "section8": every forecast is read at story time j - 2 - 5 s; the lag moves only the
regressors, read at story time j - 2 - L, as heldout.align, alice_heldout and the development records do. Section 5's "j - 2 - 5 - L" was
a drafting error, ruled before any forecast or brain data (section 9 read it the same way). The own-lag line: a grid of -4 to +1 s in
0.25 s steps, over the first 16 HEARING-eligible listeners in sorted order; section 5's "-14 to +14 s" is PREREG section 2's onset
check and is not used.

Item 7, what the code does (33eb10f, with this section's changes), checked against `scripts/r2_paranoia.py`:
- heldout's TR-2 constants at TR 1: clean's t_r and align's and wordrate_1hz's volume times run at 1 s; the first scored volume is 20;
  the null's shift minimum is 30 TRs.
- M1 runs on the joined whole-story window (each run cleaned on its own, then joined), not on 145 volumes; its controls are made per run
  and joined (LANGUAGE: the LANG-mean audio-only forecast and word rate at scan time, no lag; HEARING: none, so HEARING's M1 does not
  depend on the model and is computed once).
- Every gauge applies heldout.gauge's projector in factored form: the same numbers to rounding, tested against heldout.gauge.
- Per model, its own audio-only forecast is LANGUAGE's control in the stage, in the reference and in M1 (R1's precedent). The cheap
  baselines have no control: per model, its own forecast is the TRIBE series compared with word rate and the envelope (the full forecast
  for LANG, the audio-only forecast for AUD), unpartialled, as cheap_baselines.section8 scores them; for the fine-tuned line that is the
  fine-tuned forecast, a stated deviation from `docs/rules/cheap_baselines_rule.md`'s "stock".
- The reference is the mean of the other LANGUAGE-eligible listeners' cleaned joined LANG series, scored with LANGUAGE's gauge, controls
  (at the development lag) and null.
- K = 4 per model: LANG and AUD, each against word rate and the envelope, on the one window.
- Formats as Alice: round(point_k), claim_k, z to 2 decimals, p to 2 significant digits; the share as a whole percent.
- Word baselines: each word lower-cased with the punctuation speech-to-text attaches at its edges stripped; log word frequency within each
  part's own word table, and GPT-2 small surprisal run on each part's table from a fresh start (alice_heldout.gpt2_surprisal, p1/surprisal8.py's
  method).
- Coverage as Alice: a listener with less than 90% of LANG covered leaves the language scores and the claim; less than 90% of AUD, the
  hearing scores only.

Item 8, added to section 7. If the reference's mean LANGUAGE z is 0 or below, or its one-sided Wilcoxon p (z above 0 across listeners)
is 0.05 or more, no share is given, and the share clause becomes "the other listeners' average did not reach the bar (z {zref})",
{zref} the reference's mean LANGUAGE z to 2 decimals: "It reached {share}% of what the other listeners' average reaches." becomes "The
other listeners' average did not reach the bar (z {zref}).", and "; it reached {share}% ..." becomes "; the other listeners' average did
not reach the bar (z {zref})." (`share`, `with_share`, `outcome`; a synthetic test per branch.)

Item 9, speech-to-text and pins. Section 3's route changed as section 9 records: whisperx large-v3 float16 batch 16 ran out of memory on
the 6 GB RTX 4050, so tribev2 af58661's `_get_transcript_from_audio` ran unchanged on a Kaggle T4 (kernel stillhere-r2-asr, code
61b891d0..., whisperx 3.8.6 as uvx picked it; 1,348, 1,215 and 859 words; checks passed), before any forecast or brain data; those word
tables fed every stage. The forecasts' pins commit pins the stimuli, the word tables, the surprisal file, every forecast, the whisperx
version (`WHISPERX`, checked against r2_asr.json) and `RECORD_PINS`:
- tribe_common as the kernels ran it: the kernels embed the D22 kernels' copy (COMMON of d22/d3/kernels/stock/d3_pred.py, SHA-256
  8df0380c...), which is `k2/tribe_common.py` (5029b147...) plus `apply_weights`, `sha256` and the two checkpoint pins; every function the
  two share (build_events, stitched_predict, load_model, write_tsv and the rest) is identical;
- `p1/surprisal8.py` (e69f6d84...);
- the launchers and kernel code (`r2/kaggle_asr/`, `r2/kaggle_forecast/`, the watcher), `r2_asr.json` and the stage records.
The scorer refuses unless each matches. `download` refuses any GIFTI whose size differs from `r2/data_check.json` and writes
`bold/SHA256.json`; the scoring run checks every GIFTI it reads against that file and writes the SHA-256 of every input it read
(`results/r2_paranoia_SHA256.json`, committed with the results).

Item 10, disclosures.
(a) `r2/data_check.py` fetched bytes 0-65,535 of each of the 132 GIFTIs (8,650,752 bytes, 8.65 MB). Each header ends at byte 1,161
(left) or 1,162 (right), where `<Data>` begins, so each response also held encoded brain data: at about 43 KB per encoded volume (file
size over volumes, for example 22,787,492 bytes over 526), very likely all of volume 0 and part of volume 1. Only the bytes before `<Data>`
were parsed; nothing was decoded, saved or compared, and no forecast existed. Volumes 0 and 1 precede the story (onset 2 s) and no step
reads them (the first scored volume is 20). "No brain value read" in r2/data_check.py means none decoded.
(b) Differences from Alice not yet listed: one top-level events.tsv onset for all listeners (no per-subject events); events.tsv timings are
whole seconds while the stimuli last 523.94, 448.62 and 329.36 s (durations 524, 449 and 329), so the 2 s onset may be rounded by up to
half a volume (the own-lag line is the check); the runs are cleaned separately and joined, so the null's circular shifts cross run
boundaries; no 1/128 Hz filter in the derivative (Alice's SPM8 derivative has one; fMRIPrep's has none); M1 on the whole story.
(c) Cache wording, correcting section 3's "its key is the stage's stim.wav path": neuralset keys the sound features by the file's path,
offset and duration, and the Llama features by word and context. With the laptop runner each story and stage had its own folder; in the
Kaggle kernels the three stages of a story share one stim.wav path (section 9), so the sound features are computed once per story.
(d) `r2/harness_dev/README.md`'s "rerun ... on Oct 31 at 21:45 UTC" is wrong: the two output files were written at 21:29:30 and 21:30:53
UTC, before their commit 52424ff (21:31:22 UTC). dev_grid scores volumes 12-275 (TR 2 s). Full SHA-256: dev_window.py
1cfe3b21697621535a00a8e0bd4518dadd057282f1b8f5803b2d2fe31657042d, dev_grid.py
99ba6686b4c83e692845f3947298506d3050ddf338197e785471c6fb54d9db2e, dev_window.out.txt
d965e55dd97d8239a2efa4eae1cffd4f9e5eade231093e86404d44c9856abe4e, dev_grid.out.txt
cad28183f8f3d784c0d510a260bab193f0ff32d6ac2433df60e693bd676579eb. Both scripts import the analysis tree's `p1/` modules and do not run
from this repository.

## 11. Dated amendment, Nov 1 2026, 08:30 UTC (Yatsu's call; the cloud is informed), before the forecasts' pins commit and before any Paranoia brain data is downloaded; sections 1-10 unchanged

- The fine-tuned line is not run. Section 3 requires its stages to read every sound and Llama feature from the stock runs' cache. The
  stock forecasts ran on Kaggle T4s (section 9): part A exported its sound cache (Wav2VecBert, 10 files) for part B, but neither part kept
  its Llama text cache (each wrote 10 HuggingFaceText files that were not exported), so the fine-tuned stages cannot read the stock runs'
  features, and there is no laptop slot to make them before the cutoff (the fine-tuned weights stay on the laptop). Its line: "Fine-tuned
  TRIBE v2: rule committed, not run (the stock runs' Llama features were not kept)." Every other line in section 7 is written for stock
  TRIBE v2 only; the pair line ("z is not compared across datasets with different scan timing; the share is.") stays.
- The cloud's condition (a) of Nov 1, 07:15 UTC (check_pins with all 18 forecasts) becomes the 9 stock forecasts, with the word tables,
  surprisal, code and records pinned. The scored window (section 5's last volume with a forecast) comes from the stock forecasts.
- Scoring runs once on a Kaggle CPU kernel (section 9) while Yatsu is offline (11:30 UTC Nov 1 to about 03:30 UTC Nov 2); the results
  commit follows at about 03:30-04:00 UTC Nov 2, after the kernel's output is checked. The gap from scoring to the results commit is
  disclosed here; the kernel's output stays untouched on Kaggle until then. The kernel clones the repository from a git bundle at the pins
  commit (a full clone, checked not shallow, HEAD checked), reads its inputs from a private dataset checked against the pins, and runs
  download, score and R5's scorer (`docs/rules/r5_short_scan_rule.md`, which reads the stock forecasts only) once each.
