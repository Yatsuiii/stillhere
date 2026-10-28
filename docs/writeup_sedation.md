# Sedation case study (OpenNeuro ds003171): frozen group statistics

The sedation study is a case study, not a headline. The stimulus was shared privately by its authors; nothing from it (audio, words,
durations, predictions) appears here, and every number below is a group statistic. TRIBE v2 (stock) is the primary model; the
fine-tuned model is the pre-registered second model. Gate 1 used the timing set by D24 and ran before the 9 PM ET decision-table label
(D25).

## Pre-registered result (frozen bars)

No sedation stage line: two of the three co-primary bars did not pass.

- P7 (pipeline check, brain-to-brain ISC, not TRIBE): met.
- P1 (LANGUAGE Awake > Deep): not met. LANGUAGE fell in 9 of 12 volunteers (needed 10), p = 0.0034, Hodges-Lehmann median change
  +1.11 z [95% CI +0.43, +1.78]. The claim becomes "HEARING + ISC replication; TRIBE-only LANGUAGE did not reach the bar".
  Lag-sensitive (D24): it passes at L + 1 s.
- P3 (LANGUAGE falls more than HEARING): not met. Hodges-Lehmann median -2.32 z [-3.56, -0.79]: HEARING fell more than LANGUAGE.
  The claim becomes "one combined gauge, no split".
- P8 (LANGUAGE follows the real word order over shuffled words, Awake): met, 10 of 15, p = 0.018.
- P2a (TRIBE HEARING in Deep above the rest null): met, p = 0.0024.
- P5 (both gauges above the rest null and a wrong story, Awake): met for both. P6 (Recovery above Deep): met, lag-sensitive.

Units: z against each volunteer's own rest scan (z_rest), 145 TRs = 4.8 minutes per state. LANGUAGE is reported in z only. No hearing
detection percentage is reported for this dataset: against a wrong story, the HEARING gauge "detected" 5 of 15 Awake volunteers (33%),
above the 7% limit (exploratory check below).

## Exploratory, designed after seeing the Gate 1 result

Sensitivities listed in PREREG before Gate 1, with code written after the main result:

| Variant | P1 | P3 | Notes |
|---|---|---|---|
| Global signal regression | met (10 of 12, p = 0.017) | not met (-2.72) | P5-LANGUAGE and P6 not met |
| 0.02 Hz high-pass | met (10 of 12, p = 0.0017) | not met (-2.11) | P8 not met (p = 0.10) |
| Without one volunteer's Deep run | not met (8 of 11) | not met | no change |
| Fine-tuned TRIBE (pre-registered second model) | met (10 of 12, p = 0.008) | not met (-2.55) | P6 not met |
| Proportional P3 (each gauge's drop over its Awake median) | | +0.03 [-0.28, +0.99] | both gauges fall by a similar fraction |

P3 fails in every variant. P1 is borderline: it passes under some analysis choices and fails in the pre-registered one.

Word rate against TRIBE (same language parcels, volunteers, timing and rest null): in Awake, TRIBE's full prediction edged out a
word-rate-only regressor (10 of 15, p = 0.018), but word rate explains most of what the language parcels track in this clip (about 70%
of what the two explain together). Under deep sedation, TRIBE's full prediction fell in 11 of 12 volunteers (p = 0.0012).

Wrong-story check (Awake, scored against a different public story's prediction): HEARING 5 of 15 (33%), LANGUAGE 0 of 15 (0%).

What this case study supports: both readouts track the story in awake volunteers with zero same-clip controls, hearing stays above
the rest null under deep sedation, and the readouts fall under deep sedation. It does not support "language fades while hearing holds".
