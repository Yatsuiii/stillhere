# Partly Cloudy, one person at a time

**Question.** I'm building passive tests for patients who can't respond. A short film or story should show that a network is following it in one person. Before a result like that means anything in a patient, I need to know how often a short viewing misses a healthy person, and how much scanning fixes that.
This is a single-person question. It doesn't test or challenge the group result of Richardson et al. (2018).

**Data.**
- OpenNeuro ds000228: 122 children aged 3-12 and 33 adults, one viewing of Partly Cloudy each, with preprocessing and ROI spheres as shared.
- OpenNeuro ds002685 (IBC): 12 adults, two viewings each, plus a false-belief vs photo localiser in the same session.

**How it was done.** The rules were committed before every analysis ([PREREG.md](PREREG.md), [PREREG2.md](PREREG2.md), [PREREG3.md](PREREG3.md), [PREREG4.md](PREREG4.md)).
- Per person: correlate their ToM network time course (DMPFC, MMPFC, VMPFC, PC, LTPJ, RTPJ) with the average of the other adults.
- Null: 2000 phase-randomised surrogates of that person's series.
- Calibrated cutoff: the cutoff is set so that tests against 20 deliberately mismatched references per person come out "detected" 5% of the time (measured 4.8-5.5%).
- Intervals: Wilson 95% for counts, bootstrap over people for window averages.
- Code: [final.py](final.py) and [final_ibc.py](final_ibc.py). Raw output: [final_ds.txt](final_ds.txt) and [final_ibc_clean.txt](final_ibc_clean.txt).

## 1. One 5.3 minute viewing (ds000228)

| Age | ToM detected | 95% CI |
|---|---|---|
| 3 | 5/17 (29%) | 13-53% |
| 4 | 3/14 (21%) | 8-48% |
| 5 | 24/34 (71%) | 54-83% |
| 7 | 20/23 (87%) | 68-95% |
| 8-12 | 30/34 (88%) | 73-95% |
| Adults | 26/33 (79%) | 62-89% |

About 1 healthy adult in 5 is missed from one viewing.

## 2. How many minutes (ds000228; share of windows detected, 95% CI)

| Age | 1 min | 2 min | 3 min | 4 min | 5.3 min |
|---|---|---|---|---|---|
| 3 | 12% [7-19] | 26% [14-39] | 27% [13-44] | 31% [13-51] | 29% [12-53] |
| 4 | 18% [11-26] | 21% [11-34] | 21% [8-36] | 25% [9-43] | 21% [0-43] |
| 5 | 26% [19-34] | 40% [29-50] | 53% [40-66] | 51% [35-66] | 71% [56-85] |
| 7 | 43% [36-52] | 71% [60-81] | 85% [73-95] | 89% [76-99] | 87% [74-100] |
| 8-12 | 45% [38-52] | 62% [50-73] | 74% [61-85] | 79% [67-90] | 88% [76-97] |
| Adults | 42% [34-51] | 62% [50-74] | 68% [55-80] | 77% [63-89] | 79% [64-91] |

In adults, detection is still rising at the full film length. In 3-4 year olds it stays flat around 20-30% at every length.

## 3. One viewing vs two viewings averaged (IBC, 12 adults)

| Length per viewing | One viewing, ToM spheres | Two averaged, ToM spheres | One viewing, localiser ROIs | Two averaged, localiser ROIs |
|---|---|---|---|---|
| 1 min | 28% [22-36] | 47% [35-59] | 32% [25-39] | 44% [32-56] |
| 2 min | 60% [47-74] | 74% [58-90] | 53% [40-65] | 79% [63-91] |
| 3 min | 67% [51-80] | 87% [73-98] | 68% [54-81] | 90% [78-98] |
| 4 min | 78% [65-91] | 93% [79-100] | 74% [59-87] | 94% [83-100] |
| 5.3 min | 79% [67-92] | 92% [75-100] | 83% [69-96] | 96% [88-100] |

- One viewing in IBC misses about the same share of adults as in ds000228.
- Averaging two viewings raises detection at every length.
- Two 3 minute viewings (87-90%) beat one 5.3 minute viewing (79-83%) for about the same total scan time.
- With only 12 people, the intervals overlap.
- Localiser-defined personal ROIs and group spheres perform about the same.

## 4. The missed adults (ds000228)

| Subject | ToM r (p) | Pain r (p) | Spike TRs | Split half: sphere / personal ROI |
|---|---|---|---|---|
| sub-pixar127 | .18 (.058) | .28 (.001) | 2 | detected in 1 of 2 halves / 1 of 2 |
| sub-pixar132 | -.25 (.94) | .26 (.017) | 0 | negative in both halves with both ROI types |
| sub-pixar134 | .22 (.105) | .43 (.001) | 0 | 0 of 2 / 0 of 2 |
| sub-pixar136 | .09 (.24) | .09 (.26) | 3 | 0 of 2 / **2 of 2** |
| sub-pixar141 | -.05 (.61) | .13 (.14) | 0 | 0 of 2 / 0 of 2 |
| sub-pixar149 | -.13 (.78) | .32 (.005) | 10 | 0 of 2 / 1 of 2 |
| sub-pixar155 | .20 (.135) | .58 (.001) | 3 | 1 of 2 / 1 of 2 |

- Personal ROIs (top 50 voxels picked on the other half) rescue one adult (136) in both halves. That adult's pain network is also flat, so misregistration fits as well as ROI placement.
- At half length, adult detection is 65% with spheres and 76% with personal ROIs. Children show no consistent gain.
- sub-pixar132 is the clearest case: no spikes, a normal pain response, and ToM running against the other adults throughout.

## What failed or was corrected (kept visible)
- **First estimator:** the first pass drew 1000 circular shifts with replacement from only 139 possible. That was noisy, and borderline subjects flipped with the seed. An independent re-check caught it ([independent_review/](independent_review/)), and everything above uses the fixed null. Old numbers: [RESULTS_v1_superseded.md](RESULTS_v1_superseded.md).
- **Withdrawn claim:** "No IBC adult is missed on both viewings." With a 1-in-5 miss rate, chance predicts that for 12 people. With the fixed null, sub-09 is in fact missed on both viewings using spheres.
- **Withdrawn claims about sub-pixar132:** "no timing offset" (lag searches were inconclusive and searching many lags proves nothing) and "the likely cause is group ROIs" (contradicted by sections 3 and 4).
- **IBC registration:** the planned T1 route failed QC and was replaced with direct EPI-to-MNI registration. Dice is 0.88-0.95. The reviewer checked anatomy, with TPJ lateral and MPFC anterior in all 12.
- **IBC preregistered test failed:** testing IBC against the ds000228 adult reference gave no clear alignment peak, so it was recorded as a failure.
- **Weak IBC localiser:** top-50 belief>photo t is 2.7-4.4 in TPJ, so the personal ROIs are partly noise-selected.

## What this means for a passive test
- A negative result from one ~5 minute viewing doesn't show that processing is absent: about 1 healthy adult in 5 is missed.
- Showing the same film twice and averaging looks like the cheapest fix.
- Testing that in children needs a dataset where each child saw the film twice.
