# R7 rule: per listener, does the forecast tell the intact story from the shuffled one? (exploratory)

Raghav's call, 22 Nov 2026. Committed and pushed before the per-listener values in the two frozen D20 outputs are opened for this
question; the commit time is the record. The D20 summary (`results/d20_notthefall.json`) deliberately shows no per-listener value and
no k of N (D23(b)), and nobody working on R7 has read one. The Kaggle job logs for D20 may have printed counts, so R7 is labelled
exploratory, not pre-registered. Everything below is fixed by this commit. A change is a new commit that says what changed and why,
made before the step it changes runs. The result is shown whatever it is.

## 1. Why

"Worth about N real volunteers" measures how sharp the forecast is as a reference. A test for one person asks something narrower: is
this brain following the story? D20 answered that for the group (22 listeners, p = 0.00516). R7 asks it for each listener, with the
truth known: every listener heard the intact story and a sentence-shuffled version.

## 2. Inputs (frozen, read only)

- `tierD_d20_score.json` from the stillhere-d20-score job, SHA-256 ef488b980e308372cdadc18b89b96f8169e52afaf7f481eba036d266d74a2f38
- `tierD_d20_baselines.json` from the stillhere-d20-baselines job, SHA-256 e9c0046969d5290f2be59c628dc818ca99103e78822be25072ad7550bdbff30d

Both must match these hashes (the ones `results/d20_notthefall.json` records). On a mismatch: "R7: rule committed, not run (inputs)".
No BOLD is downloaded and nothing is re-scored. R7 only reads the per-listener values D20's frozen code already wrote.

## 3. The test

- Per listener i who heard intact and shortscram (D20's primary pairs): d_i = T1(intact) - T1(shortscram), T1 being D20's frozen
  LANGUAGE statistic for stock TRIBE v2, as `T1_per_listener` stores it.
- A listener counts as detected if d_i > 0.
- PASS needs k >= k*(n), the smallest k with P(X >= k) <= 0.05 for X ~ Binomial(n, 0.5) (16 of 22; 15 of 21). Otherwise FAIL.
- Accuracy is reported as k of n. A coin gets n / 2 on average.

## 4. Beside the result (descriptive; never decides)

- The same count for word rate and for the sound envelope, from the baselines file's per-listener LANG values.
- The same count for intact vs longscram (D20's secondary pairs), for TRIBE and both stand-ins.
- If the score file holds a real-volunteer reference per listener (other listeners' average), the same count with it. If it does not,
  the line says "no real-volunteer reference in the frozen output".
- The median d_i and its range.

## 5. Lines

- PASS: "Exploratory. On Narratives notthefall, for {k} of {n} listeners the forecast, with no volunteers scanned, matched their
  language areas better on the intact story than on the sentence-shuffled one. Word rate alone: {kwr} of {n}."
- FAIL: "Exploratory. On Narratives notthefall, the forecast told the intact story from the sentence-shuffled one for {k} of {n}
  listeners, which is not reliably better than chance (needed {kstar}). Word rate alone: {kwr} of {n}. We show it because we said we
  would."

Healthy volunteers only. Not a diagnosis or medical advice, and nothing here is about patients.
