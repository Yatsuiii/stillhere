# R3 rule: two forecast variants through PREREG's variant rule (LPP section 1: English to develop, French to confirm)

Cloud ruling of Oct 31, 08:45 UTC (R3). Committed and pushed before any variant is scored on either language and before any French
fine-tuned forecast exists; the commit time is the record. A second commit, after the four French forecasts and before the one
scoring run, adds their SHA-256 (`FR_PINS` in `lpp_variants/score_variants.py`) and nothing else. Any other change is a new commit
that says what changed and why, made before the step it changes runs. R3 finishes before R2's rule is committed, and R2's rule names
the adopted variants or says none.

## 1. The rule applied

PREREG.md line 38: "Rule for any future variant: chosen on one language, it must beat the default on the other (paired one-sided
Wilcoxon p < 0.05 and +0.2 z) before use." Here: chosen on English, confirmed on French.

## 2. Data

- LPP section 1 only: English, 49 listeners (`p1/cache_en`), French, 28 listeners (`p1/cache`), the parcel series section 3 and D3
  scored. Never section 8, never sections 2-7 or 9.
- Disclosed. Section 1, in both languages, is PREREG's development set (line 101): the variants line 38 lists were tried on it, and
  D3 already compared stock and fine-tuned TRIBE v2 on English section 1 (LANGUAGE 1.7252 against 1.9283 over all 49).
  The fine-tune never trained on section 1, but its training targets were group means over sections 2-7 and 9 of 33 English and 19
  French listeners, so 19 of the 28 French listeners' other sections were in them. The 9 French listeners in no training target
  (FR001, FR005, FR006, FR014, FR016, FR022, FR023, FR024, FR025) are reported on their own, descriptively; the decision uses all 28.

## 3. Forecasts

- English: D3's (Oct 26, one Kaggle T4 environment, the k2 harness): stock full 925ac173..., stock audio 96f48a9c..., fine-tuned full
  2f22aac9..., fine-tuned audio 006072d9... (SHA-256 as D3's table lists them; the scorer refuses others).
- French: four new forecasts on the laptop in one environment, by `lpp_variants/run_lpp1_forecast.py`: stock audio, stock full,
  fine-tuned audio, fine-tuned full, in that order. It imports the Alice runner unchanged (the same harness file
  `k2/tribe_common.py`, stitched 50 s windows, fp16 text through accelerate, the laptop placement) and R1's weight loader (both pins,
  strict, eval); words from the French TextGrid (SHA-256 1784c62a...), stimulus SHA-256 8d2f6ddb.... Launched by
  `lpp_variants/run_all_fr.sh` (kept outside the repository, SHA-256 839d3d9b49ebe41811adf2184e9d5df1e44acaf4701c72e0a32325960a9192f5;
  `run_all.sh`'s settings). The sound extractor's cache key is the path of the stage's `stim.wav`, so the stock and fine-tuned
  stages of one mode share one work folder, and each fine-tuned stage must add no file to the feature cache and leave the work folder
  unchanged: the two models then read identical features. Each stock forecast records its agreement with section 3's Kaggle forecast
  of the same mode; each fine-tuned forecast must differ from the stock one. A run that fails technically is fixed and rerun; no
  French forecast is compared with brain data before the one scoring run.

- Second commit, Oct 31, after the four French forecasts and before scoring. Made by `run_all_fr.sh` on the laptop: stock audio
  11:02-11:07 UTC; stock full first started 11:07 and was stopped at 11:33 by Claude Code for low system memory (nothing written;
  the Llama features it had cached were kept, and features do not depend on the brain model), then rerun 13:11-13:17 UTC as a
  systemd unit; fine-tuned audio and full 13:17-13:18 UTC. Each 614 by 20,484, finite, one row per second, 13 windows. Stock against
  section 3's Kaggle forecasts: audio correlation 1.000000, largest difference 2.3e-6; full correlation 1.000000, largest
  difference 7.4e-4 (fp16 text lane, the size D3 saw). Fine-tuned: 105 of 108 tensors changed, eval mode, every feature read from
  the stock stages' cache entries (no file added, work folders unchanged); correlation with stock 0.806 (audio) and 0.870 (full).
  SHA-256: stock full f61ce1b83965b71f48038d8aa214d1711a7788c52b33661e5da8cd5541a3718c, stock audio
  bdadf0885afd16002df944be9bc4bb2c27a16d13e6671566ee2205f2ac4eaec4, fine-tuned full
  805f404f8a2031db872c0d4282239952952983c3ab67a15d0476a532c74154e0, fine-tuned audio
  d4d1ed6b06e19043108294758df1e1d0e07708f6a11fe0ed48954b08e3b2ea2e. No variant has been scored on either language.

## 4. The default and the harness

The stock forecast through D3's harness (`d22/d3/d3_table.py`, which is `timing/chk8.py`'s computation), unchanged: per listener,
LANGUAGE z is the partial r of the 19 LANG parcels on the full forecast given the LANG-mean audio-only forecast and the word-rate
regressor and its derivative at chk8's l_env (English -1.25 s, French -1.77 s), volumes 10-154 cleaned at 0.03 Hz, against a
circular-shift null of at least 15 TRs; HEARING z is `p1/hearing.py` `score_z` at lag 0 on the 7 AUD parcels with the audio-only
forecast. The scorer reproduces, before it scores any variant, D3's English numbers (stock 1.7252 and 1.4723, fine-tuned 1.9283 and
1.4495) and chk8's French stock LANGUAGE (1.11, from section 3's forecasts), or refuses; checked before this commit: 1.7252, 1.4723,
1.9283, 1.4495 and 1.1079. In French the default is the laptop stock forecast, made in the same environment as the fine-tuned one.

## 5. The candidates (each applied to the full and the audio-only forecast alike)

- (a) The average of the stock and fine-tuned forecasts, second by second.
- (b) The stock forecast smoothed on the surface, forecast only: k passes of `p1/develop3.py`'s 1-ring operator S = D^-1 (A + I)
  on fsaverage5's pial mesh (each vertex becomes the mean of itself and its neighbours; the hemispheres are not joined), then the
  parcel harness unchanged.
  - Not line 38's "2-pass smoothing": that variant smoothed the brain data and the forecast alike and scored vertex by vertex, on
    French, in-sample (`p1/SPEC.md` DEV3, `p1/develop3.py`). This one leaves the brain data as recorded and smooths only the forecast.
  - "Matched to the data's smoothing" cannot be measured on these data: the dataset notes list no blurring step for the LPP
    derivatives (`research/data.md`: AFNI slice timing, despiking, motion correction, ME-ICA, MNI N27, 2 mm), so the only spatial
    averaging known is the projection's 3 mm ball; the English cache holds parcel series only; and the French vertex cache belongs
    to the confirmation set. So k is chosen on English instead: k in {1, 2, 3}, the one with the highest mean English LANGUAGE z
    (ties: the smaller k).

## 6. Decision

A candidate is adopted only if both hold:
- English (development): its mean LANGUAGE z over the 49 listeners is above the default's.
- French (confirmation): per listener, its LANGUAGE z minus the default's has a mean of at least +0.20, and a paired one-sided
  Wilcoxon test (variant greater) gives p < 0.05, over the 28 listeners.

Two candidates are confirmed, each against line 38's bar as written (no correction for two). HEARING and the 9 French listeners in no
training target are reported and decide nothing. A candidate that fails either step is reported as not adopted, with its numbers.

## 7. Lines (written before the result)

- For each candidate: "R3 (rule committed {commit time}), {name}: {adopted | not adopted}. English, development: {en_gain:+.2f} z
  against the default. French, confirmation: {fr_mean_difference:+.2f} z, higher for {fr_higher} of 28 listeners, one-sided
  p = {fr_p}."
- Then: "Adopted for R2: {the adopted candidates, or none}."

## 8. Code, order and outputs

`lpp_variants/run_lpp1_forecast.py`, `lpp_variants/score_variants.py` and `tests/test_r3_variants.py` (synthetic only: the 1-ring
operator, smoothing, averaging, the choice of k, the decision and the pin refusals; they need scipy only and run in CI). The scorer
reads only `p1/cache_en` and `p1/cache` (parcel series), never `p1/vcache`. Order: this commit on origin; the four French
forecasts on the laptop (never during a chain step that uses the laptop, never during Sunday's filming); the second commit with
their pins; one scoring run (`score_variants.py --out`, from the analysis tree's `p1/`), which writes `results/r3_variants.json` and
the per-listener numbers outside the repository; the results commit; then R2's rule.


## Dated note, Oct 31, after the result (cloud ruling of Oct 31, 21:30 UTC, items 7-8; the text above is unchanged)

- Every bare "section 3" above means PREREG section 3.
- French section 1 is PREREG's development set (line 101), already scored for stock before this rule (chk8's 1.11), and 19 of its 28
  listeners' other sections were in the fine-tune's training targets; the 9 who were not are reported on their own
  (`results/r3_variants.json`, fr_holdout9). "Confirm" here means PREREG line 38's other-language check, not a test on unseen data.
- Candidate (a) is the raw average of the stock and fine-tuned forecasts, second by second (no weighting, no rescaling). Each
  forecast's spread, forecast only (`lpp_variants/forecast_sd.py`, `results/r3_forecast_sd.json`; the mean over vertices of the SD over
  time, LANG / AUD vertices): EN full: stock 0.172 / 0.139, fine-tuned 0.050 / 0.046, average 0.110 / 0.090; EN audio: stock 0.087 / 0.108, fine-tuned 0.022 / 0.037, average 0.054 / 0.072; FR full: stock 0.173 / 0.179, fine-tuned 0.048 / 0.057, average 0.109 / 0.117; FR audio: stock 0.092 / 0.138, fine-tuned 0.025 / 0.050, average 0.057 / 0.093. Descriptive: the
  fine-tuned forecasts vary about a third as much as stock, so the average mostly follows stock.
- Candidate (b) is forecast-only surface smoothing with k chosen on English (k = 3), not smoothing "matched to the data's smoothing"
  (section 5 says why that could not be measured).
