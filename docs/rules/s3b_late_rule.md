# S3b, run late: amendment to D26 (c)

Cloud ruling of Oct 31, 21:00 UTC, item 3. Order, each step on origin before the next: the picker's scored results and SCORED.lock
(`results/picker/`, commit 9748d31, 20:51 UTC); this amendment with the hook, its synthetic test and the laptop check; a second commit
with the S3b kernels' and notebooks' SHA-256 and the scorer, before any S3b forecast; the forecasts on Colab; one scoring run; the
results commit. S3b is labelled "pre-registered, run late (Oct 31-Nov 2, after S1-S3 were seen)" wherever it appears, is reported beside
S1-S3 whatever it shows, and changes no S1-S3 number or wording.

## 1. What changes in D26 (c)

D26 (c) reported S3b as "pre-registered, not run (the cache does not allow the exact control)": the cache kept from tuning holds 2 Hz
frames, in which words sharing a frame are already merged. A fresh run does not have that limit. TRIBE v2's text lane (neuralset
`HuggingFaceText`) makes one vector per word in `_get_timed_arrays` (from `_get_data`, cached per word and context, after its layer
aggregation) and places each over the word's start and duration; the frames are built from those afterwards. So the exact control is a
hook on `_get_timed_arrays`. (The ruling says `get_static`; `HuggingFaceText` overrides `_get_timed_arrays` and never calls
`get_static`, so the hook sits one method up, on the same per-word vectors.) The frame-level approximation stays descriptive only, as
D26 (c) says.

## 2. The control (section 9's definition, unchanged)

PULA_timing: the same word onsets and durations, with each word's Llama vector replaced by the clip-mean vector (the plain mean over the
clip's words, each word once), then TRIBE v2 as usual. Code: `scripts/s3b_control.py` (`recording`, `clip_mean`, `word_mean`); synthetic
test `tests/test_s3b_control.py`.

Laptop check before this commit (`scripts/s3b_hook_check.py`, stock TRIBE v2, Alice placement, first 60 s of LPP-FR section 1, the
development section; no brain data), Oct 31 20:57-20:59 UTC: 142 word events, 142 vectors recorded, mean vector (2, 3072); with the hook
on, every word-bearing text frame of every window is parallel to the clip mean (98 frames, minimum cosine 1.0000) and the forecast
differs from the plain one (correlation 0.753); with the hook off, two plain forecasts agree exactly and a plain forecast after both
hooks equals the first (max |difference| 0.0); the class is restored. Pass.

## 3. Route and gate

- The three D26 fallback part kernels (`parts.json`), on a Colab T4 each (the GPU model the scored forecasts ran on; the notebook refuses
  any other), changed only as follows: the prediction step runs modes full and audio as before and `timing_exact` (the hook above) in
  place of the approximate timing mode; the check that a clip is complete names `timing_exact`; and the five driver lines that name
  `/kaggle/` read `/content/kaggle/` (part 3's Colab variant). Feature stage, stimuli and their checks, checkpoint pin, condition 3 and
  the merge are unchanged.
- Gate, before any control forecast: in each notebook, with the hook off, the fresh full forecasts of LPP-FR section 8 and pieman are
  compared with the scored ones (`merged/`, SHA-256 in the second commit) by `pred_stage.compare` on D26's bar (median vertex r >= 0.999,
  max |difference| <= 0.05 SD), stock pin asserted. If either fails, that notebook makes no `timing_exact` forecast.
- Budget: at most 45 Colab compute units over all S3b notebooks. Cutoff Mon Nov 2, 18:00 UTC: anything not on origin by then stays
  "pre-registered, not run".

## 4. Scoring (one run, on the laptop or Colab CPU)

Per clip, PULA_timing is `gauges.predictors_at`'s PULA computed on the control forecast in place of P_full (the code path the approximate
mode used), with the scored P_audio, word rate and rows (T1's rows of a complete run, as scored). T1, the families, blocks and counted
rows are the scored ones (`results/picker/`). S3b = `bars.rank_test(F, "PULA", "T1", ("PULA_timing",))`, `bars.py` unchanged: 10,000
within-block permutations, one-sided; it passes if rho > 0 and p < 0.05. It runs only if every counted row's clip has a control forecast
that passed the gate; otherwise it is "pre-registered, not run" with the count missing.

## 5. Lines (written before any S3b forecast)

- Pass: "S3b (pre-registered, run late: Oct 31-Nov 2, after S1-S3 were seen): TRIBE's predicted response strength still ranked the
  stories with listeners' language readouts after a forecast that keeps each word's timing but not its content was partialled out
  (partial rho {rho:.2f}, one-sided p {p:.4f}, {families} story families in {blocks} blocks). It's the content, not just the timing."
- Fail: "S3b (pre-registered, run late: Oct 31-Nov 2, after S1-S3 were seen): once a forecast that keeps each word's timing but not its
  content was partialled out, TRIBE's predicted response strength no longer ranked the stories reliably (partial rho {rho:.2f},
  one-sided p {p:.4f}, {families} story families in {blocks} blocks). S1-S3 stand as reported; the content-over-timing line is not used."
- Not run: "S3b: pre-registered, not run ({reason})."
