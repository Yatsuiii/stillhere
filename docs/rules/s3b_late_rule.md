# S3b, run late: amendment to D26 (c)

Cloud ruling of Oct 31, 21:00 UTC, item 3. Order, each step on origin before the next: the picker's scored results and SCORED.lock
(`results/picker/`, commit 9748d31, 20:51 UTC); this amendment with the hook, its synthetic test and the laptop check; a second commit
with the S3b kernels' and notebooks' SHA-256 and the scorer, before any S3b forecast; the forecasts on Colab; a commit with the control
forecasts' SHA-256 (`CONTROL_PINS` in `scripts/s3b_score.py`) and nothing else; one scoring run; the results commit. S3b is labelled "pre-registered, run late (Oct 31-Nov 2, after S1-S3 were seen)" wherever it appears, is reported beside
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

## 6. Second commit (before any S3b forecast)

- Hook `scripts/s3b_control.py` 6bf3e6714f829068cf76b02f42334986dce5e3cbadb12d417d610990618aeac4; prediction step `scripts/s3b_harness.py`
  (embedded as each kernel's HARNESS_STAGE) 5d32584c3f1924552a8eae4343a62647759a0b8c3f8cf65ceb6b44aa3866a9b2. Its laptop dry run on two
  60 s stand-ins (LPP-FR section 1, development audio): without references the gate fails and no `timing_exact` is made; with matching
  references it passes and both controls are made (142 of 142 words recorded, correlation with the plain forecast 0.753, as in section 2).
- Kernels (each the part's kernel in `parts.json` with exactly two lines changed: HARNESS_STAGE and the completeness check), built by
  `d22/picker_pred/s3b/make_s3b_colab.py` (6ac203964c8eed79e4c698681787dd06de158bceb401524384cae2e0ca20a563): part 1 d2bcf8303f2cb7d376190e659ffd67dacd66a2bdce1c70851213d8444dd871e0, part 2
  be45a2b03e25150d14cbf10e9e16b4d582f5aeb92524fc29e305fb99b31d7a21, part 3 61987fb388668b0574d52e0ad0fea09f29abafffd280aca1674a259bf589a368.
  Notebook code cells (the first cell prints the hash as run): part 1 7bb8b4dcfed20a3bb2fc8716a95c8e6c8b829779a8e833d35276a6ebc12f2f8b,
  part 2 b75ad8a22d6f7d692143042037c8fa2a48a1c7dca6f1ed885d9f790eaf142517, part 3 0b027cf4ffbfa36e148db4c572a1695ef5f127e01a2de11ae667971b8248794d.
  Colab path lines changed: 32, 125, 129, 151, 157 in each, as part 3's notebook.
- Gate references (`s3b_refs.zip` 64db8a2e8a4ff9b2853644b9cd5959b1147cbe200571c5a9ea34211b103ac651): the scored pred_full.npz of
  lppFR_section8 53c77a37a42b32e2ab64346979492e59aba550351683524f7d099add8bc88163 (Kaggle part 2) and pieman
  196be7a95b71f9331fa98dc6798a16f6b5e24d4e1799e5b3f9afc9db61d1fed5 (Colab part 3), from `merged/PRED`.
- Scorer `scripts/s3b_score.py` (e58939e655b9b279b18c368c98bbbdd927bc5f8050a97742b6461ded2baed77a). Checked before this commit: given the scored approximate timing forecasts as the
  control, it reproduces the scoring call's descriptive `PULA_timing_approx_partial` exactly (rho 0.561, one-sided p 0.0012, 37 families
  in 4 blocks), and its PULA equals the scored PULA for all 39 counted rows. That approximate number was already in the scored record
  (`results/picker/tierD_picker_bars.json`); it is descriptive and decides nothing.
- Budget: the parts' expected extraction is 6.0, 5.7 and 4.8 T4-hours (16.5 in all); at about 2 Colab units per T4-hour that is about 33
  units (an estimate; the cap is 45).

## 7. Dated note, Nov 1 2026 (cloud review of Oct 31, 23:15 UTC); sections 1-6 are unchanged

The cloud's review (a418f6d, 005aa0a, 9748d31; read-only, find then refute, two refuters each) found the hook and the route sound and
allows S3b to be scored after the notebooks finish, once the following are committed, in or before the control pins commit.

(a) Why S3b is late. The scored parts were fresh extractions, so per-word vectors existed when the scored forecasts were made; S3b was
left out because no control was built, not because the data were missing. This is said wherever the late S3b is shown (the result
file's `why_late`).

(b) Beside the result: "decided after the scored record was read, including S4, the wall and the approximate frame-level version
(descriptive, partial rho 0.56, p 0.0012)" (the result file's `beside`).

(c) Times. The ruling that added the late run reached the server at 20:48:21 UTC on Oct 31 (its text says 21:00). Then 9748d31 (the
picker's scored results and SCORED.lock), committed 20:51:31 and pushed 20:51:45; the laptop check, 20:57-20:59; a418f6d, committed
21:04:45 and pushed 21:04:58; 005aa0a, committed 21:14:26 and pushed 21:14:40 (push times from GitHub's push record for the repository).

(d) "Content" in the lines means the language model's word-by-word vectors. The sound lane and the story's average vector are kept.

(e) Which runs count: the first complete run of each part with a passing gate. Every notebook run (reruns, failures, gate fails) is
listed with its outcome in the control pins commit. How the gate references reached each runtime: `s3b_refs.zip` was uploaded through
the Files panel in ten pieces (the browser tool takes at most 10 MB at a time) and joined in the Colab terminal; the joined file's
SHA-256 was 64db8a2e... before the notebook ran, and the third cell checks the two references inside it, as written.

Scorer changes, made before any S3b forecast was read (`scripts/s3b_score.py`, replacing the scorer pinned in section 6, e58939e6...;
synthetic tests `tests/test_s3b_score.py`):
- It scores into a new, empty `--out` directory. `overlay()` replaces existing links and refuses if any control in the overlay resolves to
  a file whose SHA-256 is not a pinned control, so the dry run's approximate controls can never be read.
- Per part, it checks the run record against section 6: the notebook's code cells and the kernel (each part's hashes), the hook
  6bf3e671..., the gate references (LPP-FR section 8 53c77a37..., pieman 196be7a9...), kernel exit code 0, a T4; and the gate record:
  the same hook, each reference passed, the gate passed. Any mismatch refuses.
- `CONTROL_PINS` covers every file it reads from a part's folder: the notebook's MANIFEST.sha256, the run record, the gate record, and
  per clip the control, its provenance and the fresh full forecast. Each file must also match its line in the notebook's own
  MANIFEST.sha256.
- `parts.json` must have its SHA-256 (ab9086df...); its owns lists must be disjoint and cover every counted stem; two folders for one
  part refuse; the part is read as text. A missing file or entry is a named refusal.
- `--check-only` loads and checks every control on the downloaded files (shape and second grid equal to the scored full forecast's, finite
  values, provenance: stock checkpoint, the exact control, this hook, a passing gate; words recorded equal to word events) and stops
  before any listener data or statistic. It runs first.
- Descriptive, not gating: each counted clip's fresh full forecast against the scored one by D26's comparison (`pred_stage.compare`),
  reported beside S3b, naming any clip that misses the bar.
- It prints section 5's lines word for word, checked by a test against this file. (The section 6 scorer had put the label in
  parentheses, "run late (Oct 31-Nov 2, ...)", where section 5 says "run late: Oct 31-Nov 2, ..."; fixed.) A refusal writes section 5's
  not-run line with the named reason.
- Checked before this note: given the scored approximate timing forecasts as the control (pins set in memory, never committed), the
  revised scorer reproduces the scored descriptive `PULA_timing_approx_partial` exactly (rho 0.5610, one-sided p 0.0012, 37 families in
  4 blocks), and its PULA equals the scored PULA for every counted row; 32 synthetic tests pass.
