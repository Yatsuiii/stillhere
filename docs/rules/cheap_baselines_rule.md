# Cheap baselines beside TRIBE (StillHere Pilot rule, exploratory)

Fixed on 2026-10-28 before the brain data of the story wall (D19) and notthefall (D20) exist on this machine; for LPP-EN section 8 the
BOLD was already read on Oct 22-23, so its baseline is committed after that read. Every result here is labelled "exploratory". It sits
beside the frozen or recomputed result and never changes it; the pitch only claims what TRIBE beats.

## Regressors (all per listener-run, same scored volumes, same cleaning, same QC as the TRIBE gauge they sit beside)
- WORD RATE: HRF x per-second word onsets (the harness's own word-rate regressor, at the harness's lag).
- ENVELOPE: HRF x per-second audio envelope (the harness's own envelope regressor, at the same lag).
- TRIBE: the stock TRIBE v2 prediction. For the comparison it is scored UNPARTIALLED, because the LANGUAGE gauge partials word rate and
  audio out of TRIBE, and the same partial with word rate in TRIBE's place would remove word rate from itself.

## Gauges (the harness's raw Fisher-z mean over parcels of r(cleaned BOLD row, filtered regressor row); one regressor row tiled over parcels)
- LANG parcels: TRIBE P_full, WORD RATE, ENVELOPE.
- AUD parcels: TRIBE P_audio, WORD RATE, ENVELOPE.

## Comparison
Paired over listeners: d = TRIBE - baseline. Reported: n, k (d > 0), Hodges-Lehmann median of d with a 95% listener bootstrap interval
(10,000 resamples, numpy default_rng(20261028)), two-sided Wilcoxon p. "TRIBE beats the baseline" only if the HL is above 0 and the
interval's lower end is above 0; "baseline beats TRIBE" only if the HL is below 0 and the upper end is below 0; otherwise "no difference
shown". No correction across baselines or gauges (descriptive).

## Per check
- Section 8 (LPP-EN, 49 listeners): the section-8 harness (stillhere/heldout.py: clean at 0.03 Hz, volumes 10..154 and 10..290, regressors
  at the primary lag -1.25 s), z against the harness's circular-shift null for every gauge; label "recomputed with code built during the
  event; first measured Oct 22-23".
- Story wall (D19): after the picker's single scoring call, on the same collected parcel series, manifest rows and volumes, using the
  frozen picker functions read-only (gauges.run_times, clip.covered, clip.regs, gauges.clean, confounds, keepmask, gauges.hearing).
  (a) Per clip: mean over listeners of each raw gauge; TRIBE vs each baseline paired over all scored listeners of counted, eligible clips.
  (b) A baseline wall: D19's own-minus-wrong d per listener with each regressor in place of TRIBE (raw r, own clip's regressor vs the wrong
  families' regressors aligned from their own onsets, the same volumes as D19's pairs), lit, family and N of M rules as D19, the fake-own
  check as D19; reported for TRIBE raw, WORD RATE and ENVELOPE beside the frozen wall.
- notthefall (D20): inside the same Kaggle session as the frozen scoring call: a hook records, for every run the frozen code scores, the
  raw gauges on the same in-memory series and returns the frozen function's own value unchanged (the hashed code dataset is not modified
  and the frozen outputs are identical with and without the hook). Then, per regressor, D20's own test (intact > shortscram, and the
  secondary intact > longscram if scored) with D20's paired_test.
- Blender film (Emo-FilM): see docs/rules/emofilm_rule.md (word rate, envelope, frame brightness and motion).
