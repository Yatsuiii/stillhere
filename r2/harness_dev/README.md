# R2 development records (LPP-EN section 1, PREREG's development section)

Committed before R2's rule (cloud ruling of Oct 31, 21:30 UTC, R2 item 3). Both scripts are byte-identical to the ones the cloud named
(`dev_window.py` 1cfe3b2169762153..., `dev_grid.py` 99ba6686b4c83e69...); they ran from the analysis tree's `p1/` on Oct 31 (first at
17:30-17:40 UTC, rerun for these output files on Oct 31 at 21:45 UTC with identical numbers). Forecasts are D3's English section 1
forecasts; the brain data are the 49 English listeners' parcel series of section 1, already scored in D3. They decided nothing; they are
the evidence behind R2's choices (the full-story line and the share of the leave-one-out reference).

- `dev_window.out.txt`: LANGUAGE z for stock, fine-tuned, their average, and the other 48 listeners' average used as the forecast
  (variable `loo48_ceiling`), at 145 volumes and at the full run.
- `dev_grid.out.txt`: the same at the full run across high-pass cutoffs (0.01, 0.02, 0.03 Hz) and forecast shifts (-2 to +2 s), each as a
  share of the leave-one-out reference.

The scripts' word "ceiling" is a variable name for the leave-one-out reference; public text says "X% of what the other listeners'
average reaches" and never "ceiling".
