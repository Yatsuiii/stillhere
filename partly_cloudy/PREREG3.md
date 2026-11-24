# Scan length and confidence intervals (frozen 2026-11-24, before running)

## Confidence intervals
Every detection rate gets a Wilson 95% interval. Group comparisons are descriptive only.

## How many minutes are needed (ds000228)
- Window lengths: 30, 60, 90, 120 and 158 TRs (1, 2, 3, 4 and 5.3 minutes of film after the credits).
- For each length L < 158, use every window start from 0 to 158-L in steps of 10 TRs. The person's ToM network series and the adult reference (leave self out) are cut to the same window.
- Test: r, 500 circular shifts within the window (shift at least 5 TRs), p < .05. Detection rate for each person and length = the share of their windows detected. The group rate is the mean over people.
- False alarms for each length: the same, against 5 mismatched references per window (the reversed reference, circularly shifted by evenly spaced offsets).
- Reported per age group, with the calibrated rate alongside.

## One vs two viewings (IBC, 12 adults)
- Reference: the IBC leave-one-subject-out mean for the same viewing direction (as in the IBC write-up, ToM group spheres, TRs 5 to end).
- One viewing: each viewing tested on its own (24 tests per length).
- Two viewings: the person's two viewings averaged (after z-scoring each), tested against the average of the two references (12 tests per length).
- Same window lengths and window scheme as above. Run lengths are capped by the IBC run length.
