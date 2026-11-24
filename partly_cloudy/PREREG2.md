# Follow-up checks on the Partly Cloudy per-person result (ds000228)

Frozen 2026-11-24 after the first results were seen (see RESULTS.md), before any of the checks below were run.
These are follow-ups, not blind tests. Each one is fixed here so it can't be tuned after the fact.

## A. Personal ROIs by split half (are the 6 missed adults a group-ROI problem?)
- For each ToM region, use the voxels within 15 mm of the centre of the shipped 9 mm sphere (same 2 mm grid as the BOLD). Clean with the same nuisance regression as before.
- Halves: TRs 11-89 (half 1) and 90-168 (half 2).
- Selection on half 1: correlate each voxel with the adult ToM reference (leave self out). Keep the top 50 voxels per region.
- Test on half 2 with those voxels: network r with the reference, 1000 circular shifts, p < .05. Swap the halves and repeat.
- Comparison: the shipped 9 mm sphere scored on the same half.
- Outcome: detections in both halves for each ROI type, all subjects. Report the 6 missed adults individually.
- Control: the same personal-ROI pipeline against the time-reversed reference in both selection and test. This shows how many false detections the selection step itself creates.

## B. False-alarm calibration
- The one reversed-reference test per person is a single noisy draw.
- Replace it with 20 mismatched references per person: the reversed adult reference, circularly shifted by 20 evenly spaced offsets (from 8 to 150 TRs).
- False-alarm rate per age group = the share of the 20 x n tests with p < .05.

## C. Motion and usable timepoints
- Recompute the primary per-person r after dropping every TR flagged by an ART spike regressor (scrubbing). Use the same circular-shift test on the kept TRs.
- Report detection with scrubbing per group, and whether any of the 6 missed adults flip.

## D. Alignment
- For each subject, find the lag in -3..+3 TRs that maximises r with the reference.
- Report the lag distribution per group, and the lags of the missed adults.
- This is descriptive only. The detection rule stays at lag 0, since choosing the best lag would inflate detection.
