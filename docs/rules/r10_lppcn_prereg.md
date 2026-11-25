# R10 preregistration: does the zero-volunteer story test detect each healthy listener on a new story, in Mandarin?

Raghav Sharma, 25 Nov 2026. Preregistered. Committed and pushed before any of the data it scores is downloaded and before any
forecast for it exists; the commit time on `main` is the record (OSF and AsPredicted were not reachable from where this was
written; a mirror there, if made, will point back to this commit). Everything below is fixed by this commit. A change is a new
commit that says what changed and why, made before the step it changes runs. The result is shown whatever it is.

Healthy volunteers only. Not a diagnosis or medical advice, and nothing here is about patients.

## 1. Why

On Paranoia (Finn et al. 2018; R2, `results/r2_paranoia.json`), stock TRIBE v2's forecast of the average healthy brain response,
with no volunteers scanned on that story, detected 21 of 22 listeners one at a time (LANGUAGE z > 1.645 against a circular-shift
null) over 1,250 volumes at TR 1 s. That result was exploratory: the rule was committed before the data was read, but the per-person
count was not the rule's decision. On Alice (12.5 minutes) the same count was 16 of 22. So the per-person rate is the open question,
and it has only been seen on English. R10 asks it once, as the decision, on data nobody in this project has opened.

## 2. Data

- OpenNeuro ds003643 (Le Petit Prince multilingual corpus; Li, Bhattasali, Zhang et al., Scientific Data 2022), licence CC0, the
  Mandarin part: 35 listeners (`participants.tsv`: sub-CN001 to sub-CN037, no CN012 or CN035), each with 9 preprocessed runs
  (`derivatives/sub-CN*/func/*_task-lppCN_run-{04..12}_space-MNIColin27_desc-preproc_bold.nii.gz`), one per book section, TR 2 s,
  GE Discovery MR750, Jiangsu Normal University.
- Stimuli: `stimuli/task-lppCN_section_{1..9}.wav`. Words and timings: `annotation/CN/lppCN_section{1..9}.TextGrid` (section
  lengths from the TextGrids: 566, 644, 644, 614, 586, 784, 728, 586 and 802 s).
- Sections are identified by the run's rank among the listener's 9 derivative runs (rank 1 = section 1), as for LPP English and
  French (PREREG §2), never by run number.
- What was read before this commit, all on 25 Nov: `participants.tsv`, `README`, `dataset_description.json`, the dataset's file
  list, the `xmax` (length) of each Mandarin TextGrid, and the JSON sidecar of one raw run (sub-CN001 run-04 echo-1: TR, slice
  timing, scanner). No BOLD value, no word text and no audio.
- Untouched: PREREG §2 and D1 kept all Chinese data out of tuning, D15 kept Chinese clips out of the story wall, and the tuning
  fetch code refuses any CN file (PREREG D22 Part A (e)). No Mandarin BOLD has been correlated with anything in this project.
- Where it sits for TRIBE v2: LPP is one of the held-out evaluation sets in TRIBE v2's paper (d'Ascoli et al. 2026, arXiv
  2605.04326, Table 1), where predictions were scored against group-average responses; TRIBE v2 was not trained on it. The paper does
  not say which LPP languages it scored. So R10 tests StillHere's per-person readout on a new story and language; it is not a
  claim about TRIBE's generalisation.

## 3. Forecast and scoring

- Model: stock TRIBE v2 only, the pinned checkpoint (`facebook/tribev2` revision f894e783020944dcd96e5568550afe2aa9743f9f,
  `best.ckpt`, SHA-256 9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321; PREREG D22 Part A). Its text lane is
  Llama 3.2, which is not documented as trained for Mandarin. That is part of what is tested: a fail is reported as a fail on
  Mandarin, with no cause claimed.
- Per section: the full forecast, the audio-only forecast, and the full forecast with word texts shuffled across onsets (seed 7),
  built from the dataset's TextGrid words (no speech-to-text), with section 8's harness (`k2/tribe_common.py`). Their SHA-256 are
  committed, with nothing else, before the BOLD is downloaded.
- Brain data: each MNIColin27 run projected to fsaverage5 exactly as the LPP harness does (`project` in `scripts/alice_heldout.py`:
  vol_to_surf, 3 mm ball, linear, mid meshes; left then right). Cleaning, alignment, parcels and gauges: exactly `stillhere/heldout.py`
  as it stands at this commit (the LPP section 8 scorer): Schaefer-400 LANG and AUD parcels, regressors, development lag -1.25 s,
  circular-shift null, seeds and resample counts. Per section, volumes 10 to the last volume with brain data and a forecast
  for every listener; each section cleaned on its own, then sections joined in story order.
- Per listener: LANGUAGE z (partial r of the LANG parcels with the full forecast, given the LANG-mean audio-only forecast, word rate
  and its derivative, against the circular-shift null), as in R2. **Detected** means LANGUAGE z > 1.645.
- Coverage: R2's rule (section 4 there), with "all three runs" read as "every section in the window". If fewer than 20 listeners
  have the language areas in view: "R10: preregistered, not run (coverage)".

## 4. Hypotheses, window and decision

**Primary window: sections 1 to 4** joined (about 1,190 scored volumes, 41 minutes at TR 2 s). It is chosen to match the number of
scored volumes behind the Paranoia result (1,250), because the null's spread depends on volume count, not minutes.

**Secondary window: sections 1 and 2** (about 590 volumes, 20 minutes), a length a clinical scan can afford. Same tests, own
k*(n), reported on its own line, never replacing the primary.

- **H1 (group).** Mean LANGUAGE z >= 1 and one-sided Wilcoxon signed-rank p < 0.05 across listeners (the existing pass rule,
  `heldout.py`).
- **H2 (per person, the decision).** k of n listeners detected. PASS if k >= k*(n), the smallest k with P(X >= k) <= 0.05 for
  X ~ Binomial(n, 0.70). For n = 35, k* = 30. Read: a sensitivity of 70% or less is ruled out at 5%.
- **Guard G1 (wrong story).** Each listener's primary-window BOLD is scored with the forecast for sections 5 to 8 (the next part of
  the same book: same narrator, same characters), cropped to the same length, with its own audio-only and word-rate controls. False
  alarms = listeners with z > 1.645. The guard holds if false alarms <= floor(0.07 n), the project's 7% wrong-story limit (2 of 35).
- **Guard G2 (cheap baselines).** TRIBE's per-listener LANGUAGE z beats word rate and the sound envelope (`docs/rules/
  cheap_baselines_rule.md`), one-sided Wilcoxon p < 0.05 for each.

The per-person line is used only if H1, H2, G1 and G2 all hold. Beside every result, descriptive only: the detected count per single
section (about 10 minutes each) and cumulatively over sections 1 to 9; the shuffled-word forecast's count; median z and range.

## 5. Sample size

All 35 Mandarin listeners are used; there are no others. The justification, with the numbers from `scripts/prereg_lppcn_power.py`
(`results/prereg_lppcn_power.json`, no brain data read):

- **The COGITATE rule**: each lab's sample set at 2.5 times the usual sample for the method (after Simonsohn 2015's small
  telescopes), aiming above 90% power for a within-subject effect of d >= 0.5, described by its authors as a rough estimate. Our original is one 22-listener result, so 2.5 times
  is 55. We have 35, which is 64% of that. We say so rather than stretch it.
- **H1 at d = 0.5:** power 0.89 (paired t) and 0.88 (Wilcoxon, 20,000 simulations) at n = 35, just under the 0.90 target; it would
  be 0.98 at n = 55. Paranoia's effect was far larger (mean z 4.0, listener SD about 1.6, d_z about 2.4), so H1 is not where R10 can
  fail unless Mandarin breaks the readout.
- **H2 is where the size matters.** With k* = 30 of 35, power is 0.99 if the true per-person rate is 0.95 (Paranoia's), 0.87 at
  0.90, 0.57 at 0.85 and 0.27 at 0.80. At n = 55 these would be 1.00, 0.98, 0.81 and 0.45. So R10 can confirm a Paranoia-like rate,
  and has little power to separate 0.80 from 0.70.
- **G1:** with at most 2 false alarms of 35 allowed, the guard holds with probability 0.91 if the true false-alarm rate is 3%, 0.75
  at 5% and 0.55 at 7%.
- **What 35 cannot do:** Simonsohn's d33 for n = 22 is 0.27; at n = 35 a group test has 0.46 power against it. R10 makes no
  claim about effects that small.

## 6. Lines (written before the result)

{n}, {k}, {kstar}, {fa}, {kwr} and {kenv} are copied from the output.

- PASS (H1, H2, G1, G2 hold): "Preregistered. On a new story in a new language (Le Petit Prince in Mandarin, Li et al. 2022;
  {n} healthy listeners, 41 minutes), the forecast of an average healthy brain response, with no volunteers scanned on this story,
  detected {k} of {n} listeners one at a time (needed {kstar}). On the next part of the same book it raised {fa} false alarms."
- H2 fails, others hold: "Preregistered. On Le Petit Prince in Mandarin ({n} healthy listeners, 41 minutes), the zero-volunteer
  forecast detected {k} of {n} listeners one at a time, below the preregistered {kstar}. We show it because we said we would."
- H1 fails: "Preregistered. On Le Petit Prince in Mandarin ({n} healthy listeners), the zero-volunteer forecast did not track the
  listeners' language areas as a group (z {z}, p = {p}). The Paranoia per-person result does not carry over to this story and
  language; cause not claimed."
- A guard fails: the H2 count is given with the guard result ("{fa} of {n} false alarms on the wrong section, over the limit of
  {famax}" or "did not beat word rate / the sound envelope") and no per-person line is used.
- Secondary window: the same lines with "20 minutes" and its own {kstar}, labelled secondary.
- Not run by Sun 31 Jan 2027, 23:59 UTC: "R10: preregistered, not run."

Never "understanding", "comprehension", "awareness" or any patient wording. The forecast is a model of average healthy responses;
nothing here says it stands in for a person.

## 7. Order

This rule; a code commit (runner and scorer for ds003643 CN, synthetic tests, `heldout.py` unchanged); the forecasts (Kaggle);
a commit with the forecasts', word tables' and stimuli's SHA-256 and nothing else; the BOLD download (sections 1 to 8 only); one
scoring run; the results commit.
