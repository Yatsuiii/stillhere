# R5 rule: does TRIBE v2 pick the minutes of a story where listeners' language areas follow it most? (exploratory)

Yatsu's call, Nov 1 2026, about 05:50 UTC. Committed and pushed before any Paranoia brain data is downloaded or read and before any
Paranoia forecast exists on this machine; the commit time is the record. It reuses R2's data, forecasts, cleaning and LANGUAGE gauge
(`docs/rules/r2_paranoia_rule.md`, with its dated amendments) and is scored in R2's one scoring run, after R2's pins commit. Everything
below is fixed by this commit; a change is a new commit that says what changed and why, made before the step it changes runs. The result
is shown whatever it is, labelled exploratory.

## 1. Why

For a person who can be scanned only briefly, the minutes of story played decide whether their language areas have a fair chance to show
a response. The picker (PREREG section 9) found that TRIBE's predicted response strength ranks whole stories with listeners' language
readouts (S1 and S3 passed); its impact test, S4 (a block's top stories against all its stories, in listener z), pointed the same way but
its interval crossed 0. R5 asks the short-scan version of S4 within a story, on 22 listeners whose brain data nobody here has seen: do
the minutes TRIBE ranks highest give a closer language-area match than the minutes it ranks lowest?

## 2. Windows and the forecast-only ranking (no brain data)

- Per story part (story1, story2, story3), on R2's scored window of that part (volumes 20 to the part's last scored volume; section 5 of
  R2): non-overlapping 60-volume windows (60 s at TR 1 s) from volume 20; a trailing window shorter than 60 volumes is not used.
- Each window's score is the picker's predictor PULA computed on stock TRIBE v2's forecasts as R2's LANGUAGE gauge reads them (each run
  cleaned on its own and joined, the forecast read at story time j - 2 - 5 s): per LANG parcel, the residual of the full forecast's row
  after least squares on [1, the same parcel's audio-only forecast row, the word-rate regressor at R2's development lag] within the
  window, its standard deviation; PULA = the mean over the LANG parcels R2 keeps.
- Within each part, windows are ranked by PULA; the top k and the bottom k are kept, k = floor((n + 1) / 3) for a part with n windows
  (8 windows: 3; 7: 2; 5: 2). The rest are not used. Ties (equal PULA to 12 digits) go to the earlier window.
- The same ranking with the window's word rate (mean words per second) in place of PULA, and with the sound envelope's standard deviation,
  is computed for the baseline comparison (section 4).

## 3. The target and the test

- Listeners: R2's LANGUAGE-eligible listeners (its coverage rule); fewer than 10: "R5: rule committed, not run (coverage)".
- Per listener i and window set S: m_i(S), R2's LANGUAGE gauge statistic (the Fisher-z mean over LANG parcels of the partial r of the
  cleaned joined BOLD row with the full forecast's row given the LANG-mean audio-only forecast, word rate and its derivative at the
  development lag, and a constant), computed on the volumes of S only. No circular-shift null is used: the comparison is within listener.
- d_i = m_i(TRIBE's top windows) - m_i(TRIBE's bottom windows), the same number of volumes on each side.
- PASS needs both: (1) one-sided exact Wilcoxon signed-rank test across listeners, d_i > 0, p < 0.05; (2) at least k*(n) listeners
  with d_i > 0, k*(n) the smallest k with P(X >= k) < 0.05 for X ~ Binomial(n, 0.5) (16 of 22). Otherwise FAIL.
- Exploratory. Stock TRIBE v2 only (the primary model, D3); the fine-tuned model is not part of R5.

## 4. Beside the result (descriptive; never decides)

- What it had to beat: the same d_i with windows ranked by word rate, and by the envelope's standard deviation (k of n above 0,
  one-sided Wilcoxon p for each), and TRIBE's top windows against word rate's top windows (m_i(TRIBE top) - m_i(word-rate top)).
- The minutes used per side, and the windows' PULA per part.

## 5. Lines (written before any Paranoia brain data exists here)

{n} is the listeners scored, {M} the minutes on each side, {k} and {p} from section 3, {kwr} and {pwr} the word-rate ranking's.

- PASS: "Exploratory. On another public story (Paranoia, Finn et al. 2018; {n} listeners, outside TRIBE v2's training and evaluation
  sets), listeners' language areas followed TRIBE's forecast more closely in the {M} minutes it ranked highest than in the {M} minutes
  it ranked lowest ({k} of {n} listeners, one-sided p = {p}). Ranking the same minutes by word rate alone: {kwr} of {n}, p = {pwr}."
- FAIL: "Exploratory. On another public story (Paranoia, Finn et al. 2018; {n} listeners, outside TRIBE v2's training and evaluation
  sets), the minutes TRIBE's forecast ranked highest did not reliably give a closer language-area match than the minutes it ranked
  lowest ({k} of {n} listeners, one-sided p = {p}). We show it because we said we would."
- Not run: "R5: rule committed, not run ({reason})." (coverage; or R2 not scored by its cutoff).

No other line describes this result. Not a diagnosis or medical advice. Nothing here is about patients: the listeners are healthy
volunteers, and R5 says nothing about anyone who cannot respond.

## 6. Order and cutoff

This rule; the scorer and its synthetic tests (`scripts/r5_short_scan.py`, `tests/test_r5_short_scan.py`); R2's forecasts and pins
commit (R5 reads R2's pinned stock forecasts and word tables); the download; one scoring run with R2's; the results commit. Same cutoff
as R2: on origin/main by Mon Nov 2, 12:00 UTC, or "R5: rule committed, not run".

Dated note, Nov 1 2026 (cloud review of Nov 1, 07:15 UTC, item 3; the text above is unchanged): R5 was added at about 05:50 UTC, after
the face test's result (pushed 05:32 UTC) and before any Paranoia brain data was downloaded or opened. "Whose brain data nobody here has
seen" is as R2's disclosures say (`docs/rules/r2_paranoia_rule.md`, sections 2 and 10): an earlier agent read the raw files' AFNI header
statistics, and the data check fetched the first 64 KB of each fsaverage5 GIFTI, which held encoded data never decoded; no brain value
was read.

Dated note, Nov 2 2026 (cloud review of Nov 2, 12:15 UTC, item 6; after the result, results commit 13e93c0; the text above is unchanged):
`scripts/r5_short_scan.py` gains `--verify-own DIR` (with `--out`) so anyone can run this rule's test on their own stock Paranoia
forecasts (`scripts/reproduce_paranoia.sh`). Under it, the forecasts and word tables come from DIR and are not held to R2's pins; the
stimuli, scans and atlas must equal what our scoring run read; nothing is written into `results/`; and the numbers print beside ours. It
is not part of this rule's result. Our scoring path is unchanged (`docs/frozen_code_changes.md`, entry of Nov 2).
