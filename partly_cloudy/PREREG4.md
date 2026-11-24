# Estimator fix and final analysis plan (frozen 2026-11-24, before running)

Why: an independent re-check found the p-value estimator noisy and biased. It drew 1000 circular shifts with replacement from only 139 possible shifts, which gives p of about m/139 + .001 instead of (m+1)/140, and borderline subjects flipped with the seed. With short windows there are too few shifts to reach p < .05 at all. This change was triggered by the review, not by outcomes.

Null (everywhere from now on): 2000 phase-randomised surrogates of the subject's series (same amplitude spectrum, random phases, fixed seed 0 per test). p = (m+1)/2001 one-sided on r.
Calibration: for every person and length, run 20 tests against mismatched references (the reversed reference, circularly shifted by 20 evenly spaced offsets). The cutoff is set so that mismatched detections run at 5% over all people in the dataset. Rates are reported at that calibrated cutoff, with p < .05 alongside.
Intervals: Wilson 95% on the number of people detected.

1. ds000228 at full length (158 TRs): detection by age group. List the missed adults with exact p, pain r and p, spike count (TRs 11-168 only), and VMPFC voxel count.
2. ds000228 by length: windows of 30, 60, 90, 120 and 158 TRs, starts every 10 TRs. Per-person detection = the share of windows detected at that length's calibrated cutoff. Group mean with a bootstrap 95% CI over people.
3. IBC by length: one viewing (24 viewings) vs two viewings averaged (12 people; z-score each viewing, average, test against the averaged leave-one-out reference). Same windows, calibration done separately for one-viewing and two-viewing tests. Group spheres are primary, personal localiser ROIs secondary.
4. The split-half personal-ROI analysis is rerun with the new null, and per-subject claims are taken only from the rerun.
