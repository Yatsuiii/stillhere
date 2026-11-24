# IBC replication: Partly Cloudy twice per adult, with a false-belief localiser

Data: OpenNeuro ds002685 (Individual Brain Charting). 12 adults watched Partly Cloudy twice in one session (runs dir-ap and dir-pa) and ran the Saxe-lab false-belief vs photo localiser in the same session.
Rules: [PREREG.md](PREREG.md), committed before any movie time series was extracted. Code: [pipe.py](pipe.py), [analyze_ibc.py](analyze_ibc.py), [ibc_internal.py](ibc_internal.py), [lag_ap_pa.py](lag_ap_pa.py).

## Deviation from the frozen rules (registration)
The frozen plan went EPI -> T1 -> MNI. That route failed its quality check, because the T1 couldn't be skull-stripped offline:
- The T1-to-MNI affine came out with determinant 0.39-0.65.
- The template brain warped into EPI space measured 2.7-3.4 L.

I switched to direct EPI -> MNI (ANTs SyN on the masked mean EPI). Overlap of the warped template brain with the EPI brain mask is now Dice 0.88-0.95 for all 12 subjects.
This change was made after a preview on 9 subjects run with the broken registration. The criterion for the change was registration quality, not outcomes.

## Preregistered test: IBC viewings against the ds000228 adult reference — inconclusive
- The alignment curve between the two datasets is a broad plateau rather than a peak: pain network r = 0.40-0.46 for any offset between 11 and 17 TRs.
- The event timings shipped with the two datasets also drift apart over the film.
- Most likely the two copies of the film are not timed identically, so a cross-dataset reference is the wrong tool here.
- With the frozen offset (k = 16), ToM detected 11/24 viewings with group spheres and 7/24 with personal ROIs. Reversed-reference false alarms were 1/24 each. Pain detected 14/24. McNemar personal-only 2 vs group-only 6, p = .29.
- Recorded as a failure of the cross-dataset design. I don't read it as a result about ToM.

## Not preregistered: IBC's own reference (leave one subject out, same viewing direction, no cross-dataset timing)
| ROI | Viewings detected | Reversed false alarms | Median r |
|---|---|---|---|
| ToM, group spheres | 19/24 | 1/24 | 0.30 |
| ToM, personal (top 50 belief>photo voxels within 15 mm) | 20/24 | 1/24 | 0.29 |
| Pain, group spheres | 20/24 | 1/24 | 0.22 |

- About 1 in 5 single viewings miss a healthy adult. That's the same rate as ds000228 adults (6/33).
- Localiser-defined personal ROIs don't change the miss rate (4 vs 5 misses).
- **No subject is missed on both viewings.** sub-04 and sub-14 are missed on dir-pa only, sub-15 on dir-ap only, and sub-07 on dir-ap with personal ROIs only. With two viewings, all 12 adults are detected.
- No consistent timing shift between the two viewings (group ToM peak at lag 0, r = .61). I don't know which run came first.

## Caveats
- 12 subjects. The localiser is weak at 3 mm with no smoothing: mean belief>photo t in the group spheres is 0.1-1.5.
- Within-person test-retest of the ToM time course is low (r = 0.0-0.35).
