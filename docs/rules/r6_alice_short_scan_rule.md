# R6 rule: does R5's minute ranking hold on a second story? The Alice Datasets, chapter 1 (a replication attempt of R5)

Drafted Nov 3 2026 and approved in the cloud's ruling of Nov 3, 19:05 UTC (item 3); committed and pushed to origin/main before any
step below runs, and the commit time is the record. Label, on every line and table row that reports R6: "Exploratory; the Alice scans were read on Oct 30 for the held-out test". It is R5 (`docs/rules/r5_short_scan_rule.md`) moved, choice for choice, from Paranoia to the Alice
Datasets, with the Alice held-out rule's data, forecasts, coverage, cleaning and LANGUAGE gauge (`docs/rules/alice_heldout_rule.md`,
with its dated amendments). Everything below is fixed by that commit; a change is a new commit that says what changed and why, made
before the step it changes runs. The result is shown whatever it is, labelled exploratory.

## 1. Why

R5 found, on one story (Paranoia, 22 listeners), that the minutes stock TRIBE v2 ranks highest by its forecast alone gave listeners'
language areas a closer match than the minutes it ranks lowest (17 of 22, one-sided p = 0.0004), and word rate alone did not (8 of 22).
One story is one story. R6 asks the same question, with the same ranking and the same test, on the second public story outside TRIBE
v2's training and evaluation sets that this project has: the Alice Datasets, chapter 1, 22 listeners, whose stock forecasts were made
on Oct 30, before any Alice brain data was downloaded.

## 2. Windows and the forecast-only ranking (no brain data)

- The scored window is the Alice held-out rule's full window, the one its "full" stage scored: volumes J0 = 10 to j1 = min(the last
  volume every listener's forecast covers, n - 1), n the shortest run (`scripts/alice_heldout.py` score: 372 volumes in every scored
  listener, so j1 = 371). Not its 145-volume primary window: R5 ranks minutes across a whole story, and 145 volumes hold 4 minutes.
  The first 10 volumes are dropped as section 8 drops them (its J0), as R2 dropped its first 20 at TR 1 s.
- Window size, scaled to ch01's length (the ruling's words): R5's unit is one minute of the story, and the number of minutes follows
  the length of what is scored (Paranoia's three parts gave 8, 7 and 5). Here the same minute is 30 volumes at TR 2.0 s, and ch01's
  full window gives 12 of them; so the window is R5's minute and the count is ch01's own, not a different window length.
- Non-overlapping 30-volume windows (60 s at TR 2.0 s) from volume 10; a trailing window shorter than 30 volumes is not used (with
  j1 = 371: 12 windows, volumes 10 to 369; volumes 370 and 371 are not used).
- Each window's score is R5's PULA (`scripts/r5_short_scan.py` `pula`, imported) on stock TRIBE v2's Alice forecasts as the Alice
  rule's LANGUAGE gauge reads them, at the stimulus's own time (onset 0, one ranking for every listener): the full and audio-only
  forecasts (`pred_alice_full.npz`, `pred_alice_audio.npz`, the Alice rule's pins) read at stimulus time 2j - 5 s
  (`heldout.pred_parcels`), on the LANG parcels the Alice scoring keeps (its coverage rule), each row cleaned on J0..j1
  (`heldout.clean`, 0.03 Hz); the word-rate regressor as the Alice rule's at its development lag (-1.25 s), cleaned the same way.
  Per LANG parcel: the residual of the full forecast's row after least squares on [1, the same parcel's audio-only row, the word-rate
  regressor] within the window, its standard deviation; PULA = the mean over the kept LANG parcels.
- Windows are ranked by PULA; the top k and the bottom k are kept, k = floor((n + 1) / 3) for n windows (12 windows: 4); the rest are
  not used. Ties (equal PULA to 12 digits) go to the earlier window (R5's `pick`, imported).
- The same ranking with the window's word rate (mean words per second over the stimulus seconds the window's volumes span, volume j
  spanning seconds 2j and 2j + 1, from the timed words of `code/annotations.tsv` as the Alice rule takes them) in place of PULA, and
  with the sound envelope's standard deviation over the same seconds, is computed for the comparison (section 4).

## 3. The target and the test

- Listeners: the Alice rule's LANGUAGE-eligible listeners (its coverage rule; sub-53 excluded as it is there); fewer than 10: "R6: rule
  committed, not run (coverage)".
- Per listener i and window set S: m_i(S), the Alice rule's LANGUAGE gauge statistic without its null (R5's `m_on`, imported: the
  Fisher-z mean over the kept LANG parcels of the partial r of listener i's cleaned BOLD row with the full forecast's cleaned row, given
  the LANG-mean audio-only forecast, word rate and its derivative at the development lag, and a constant), computed on the volumes of S
  only. As the Alice rule does, listener i's forecast and regressors are read at its own audio onset (events.tsv). No circular-shift
  null is used: the comparison is within listener.
- d_i = m_i(TRIBE's top windows) - m_i(TRIBE's bottom windows), the same number of volumes on each side.
- PASS needs both: (1) one-sided exact Wilcoxon signed-rank test across listeners, d_i > 0, p < 0.05; (2) at least k*(n) listeners
  with d_i > 0, k*(n) the smallest k with P(X >= k) < 0.05 for X ~ Binomial(n, 0.5) (16 of 22). Otherwise FAIL. (R5's `test` and
  `kstar`, imported.)
- Exploratory. Stock TRIBE v2 only; the fine-tuned model is not part of R6.

## 4. Beside the result (descriptive; never decides)

- What it had to beat: the same d_i with windows ranked by word rate, and by the envelope's standard deviation (k of n above 0,
  one-sided Wilcoxon p for each), and TRIBE's top windows against word rate's top windows (m_i(TRIBE top) - m_i(word-rate top)).
- The minutes used per side, and the windows' PULA.

## 5. Lines (written before any R6 statistic exists)

{n} is the listeners scored, {M} the minutes on each side, {k} and {p} from section 3, {kwr} and {pwr} the word-rate ranking's.

- PASS: "Exploratory; the Alice scans were read on Oct 30 for the held-out test. A replication attempt of R5: on a second public story outside TRIBE v2's training and evaluation sets (the Alice
  Datasets, chapter 1; {n} listeners), listeners' language areas followed TRIBE's forecast more closely in the {M} minutes it ranked
  highest than in the {M} minutes it ranked lowest ({k} of {n} listeners, one-sided p = {p}). Ranking the same minutes by word rate
  alone: {kwr} of {n}, p = {pwr}."
- FAIL: "Exploratory; the Alice scans were read on Oct 30 for the held-out test. A replication attempt of R5: on a second public story outside TRIBE v2's training and evaluation sets (the Alice
  Datasets, chapter 1; {n} listeners), the minutes TRIBE's forecast ranked highest did not reliably give a closer language-area match
  than the minutes it ranked lowest ({k} of {n} listeners, one-sided p = {p}). R5's Paranoia result is not replicated here. We show it
  because we said we would."
- Not run: "R6: rule committed, not run ({reason})." (coverage; or not scored by the cutoff).

No other line describes this result. Not a diagnosis or medical advice. Nothing here is about patients: the listeners are healthy
volunteers, and R6 says nothing about anyone who cannot respond.

## 6. Order and cutoff

This rule; the scorer and its synthetic tests (`scripts/r6_alice_short_scan.py`, `tests/test_r6_alice_short_scan.py`), on origin/main
with it; one scoring run (`python scripts/r6_alice_short_scan.py score --root ROOT --out OUT`, OUT outside the repository), which
refuses to run unless this rule is on origin/main unchanged, refuses a second run once `results/r6_alice_short_scan.json` exists, checks
the stimulus, annotations and forecasts against the Alice rule's pins and the brain data against the hashes the Alice download recorded;
the results commit. No new forecast, no GPU, no new download. Cutoff: scored and its results committed by Wed Nov 4, 20:00 UTC, or
"R6: rule committed, not run (cutoff)".

## 7. What has been seen before this rule

- The Alice brain data (OpenNeuro ds002322, 22 scored listeners) were downloaded on Oct 30 and read in the Alice held-out rule's
  scoring runs (the first run and the corrected run, Oct 30-31), in R1's fine-tuned run (Oct 31), and in a development script of
  Nov 1 that scored other forecast sets against stock (`alice/forecast_dev/score_dev.py`, kept outside the repository; group means
  and paired differences only). All of them computed whole-window statistics: the LANGUAGE gauge on volumes 10-154 (z 1.42, the
  held-out line) and on the full window 10-371 (the "full" stage), the hearing and envelope gauges, the M1 listener curve and the
  cheap baselines. Their group numbers were seen; the held-out ones are in `results/alice_heldout.json`.
- What has never been computed on them, by any committed or uncommitted script we know of: any per-window or per-minute statistic,
  any fit of a listener on part of the window, PULA on the Alice forecasts, or any ranking of Alice's minutes. Nobody has looked at
  per-minute fits.
- This rule was drafted from design facts only: TR, run length (372 volumes), the scored windows, the listener and parcel counts
  (22 listeners eligible; 19 of 19 LANG parcels kept), the onset range (0.003 to 0.837 s) and the forecasts' shapes (744 by 20,484).
  No Alice brain value, per-listener result or listener series was read to write it.
- So R6 is not a test on unseen data in the way R2 and R5 were: the listeners' whole-window fit is known, and a strong whole-window fit
  makes no promise about which minutes fit best. That is why it is labelled a replication attempt, exploratory, and why its result is
  reported whatever it shows.
