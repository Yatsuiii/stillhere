# veq pre-registration

This file is append only. Each section is committed before the code or data it talks about, and the commit that adds it is the
timestamp. Later sections never edit earlier ones; a change of plan is a new dated entry that says what changed and why.

## 1. What the scorer must do (written 21 Nov 2026, before the scorer code)

veq answers two questions for each person in a story-listening fMRI dataset:

1. Does a brain-encoding model predict this person's brain response better than cheap stand-ins such as word rate and the sound
   envelope?
2. How many real volunteers is the model's forecast worth for this person? That is, how many other people's scans, averaged, would
   match this person as closely as the model does?

The scorer takes arrays only: per-person BOLD for a set of brain areas, one forecast (or one per person) for the same areas, and named
baseline regressors. It never downloads data and never runs a model, so it carries no model's licence.

Definitions, fixed before the code is written:

- Cleaning: per row, linear detrend, optional Butterworth high-pass, z-score.
- Score (z): the Fisher-z mean over areas of the correlation between a person's BOLD row and the forecast row, both residualised on
  any control regressors plus an intercept, turned into a z against the same score for every circular shift of the forecast by at
  least `min_shift` volumes. This is the gauge StillHere used for every published number (stillhere/heldout.py, `gauge`).
- Baselines are scored with exactly the same gauge and the same controls as the model.
- "Beyond baselines": the model's z with every baseline added to the controls. A model that only tracks word rate should score near
  zero here.
- Volunteer curve: for each person and each k, the z of the average of k other people's cleaned BOLD (random draws, fixed seed),
  scored with the same gauge and controls.
- Volunteer-equivalent: the model's z placed on the curve by linear interpolation; 0 below the first point; at or above the last
  point it is reported as the largest k and flagged as at the ceiling. Per person, and for the group mean with a bootstrap 95%
  interval over people. The claim uses the lower end, rounded down.
- Beats a baseline: the model's group mean z is higher and a one-sided Wilcoxon signed-rank test on per-person differences gives
  p < 0.05 after Holm correction across baselines.

## 2. Failure injections the scorer must catch (written before the scorer code)

Each one is a test in `tests/test_failure_injections.py`, on synthetic data with a known answer. The scorer is not done until all of
them pass, and none may be loosened afterwards without a new entry here.

- F1 Null forecast: noise unrelated to the brain data gives a group mean z within +/-1, a volunteer-equivalent of 0 at the lower
  bound, and beats no baseline.
- F2 Word rate in disguise: a forecast that is word rate plus a little noise, on data that also contains a story signal, must not beat
  the word-rate baseline, and its beyond-baselines z must be within +/-1.645 for most people.
- F3 Real signal: a forecast carrying the shared story signal beats both baselines and its beyond-baselines z passes 1.645 for most
  people.
- F4 Monotone: a cleaner forecast of the same signal is worth at least as many volunteers.
- F5 Misalignment: the real-signal forecast shifted by 20 volumes scores within +/-1.5 for the group (a timing bug must not pass).
- F6 Leak: a forecast that is literally one person's own BOLD puts that person at the ceiling and the flag says so.
- F7 Scale: multiplying the forecast by 10 or adding a constant changes no number.
- F8 Inputs: mismatched lengths, NaNs or fewer than 3 people raise an error instead of returning a number.
- F9 Determinism: the same inputs and seed give byte-identical JSON.

## 3. Amendment before freezing (21 Nov 2026): F2 failed, so "beats a baseline" got stricter

With the section 1 definition, F2 failed: a forecast that is word rate plus a little noise "beat" the word-rate baseline, because its
per-person z differed from word rate's by tiny amounts in a consistent direction and the signed-rank test called that significant.
That is the exact mistake the scorer exists to catch, so the definition changes, not the test.

New definition: the model beats a baseline only if (a) its group mean z is higher, (b) the Holm-corrected one-sided Wilcoxon p on the
per-person differences is below 0.05, and (c) its beyond-baselines z is above zero for the group (one-sided Wilcoxon p < 0.05). Part
(c) asks whether the forecast carries anything the baselines do not. The F1 to F9 tests are unchanged.

The failing run is kept: `results/dev_f2_failure.txt`.
