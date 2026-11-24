# Partly Cloudy, one child at a time

Data: OpenNeuro ds000228 (Richardson, Lisandrelli, Riobueno-Naylor & Saxe 2018), 122 children aged 3-12 and 33 adults, one 5.6 minute viewing each.
Rules were frozen and committed before any time series was extracted ([PREREG.md](PREREG.md)). Code: [extract.py](extract.py), [analyze.py](analyze.py). Raw output: [results_stdout.txt](results_stdout.txt), per-subject numbers: [results.json](results.json).

## Question

The 2018 paper shows at the group level that the theory-of-mind (ToM) and pain networks follow the film, and that this becomes more adult-like with age.
Here the question is per person: from one 5.6 minute scan, in how many individual children (and adults) can we say "this person's ToM network is following the film", with a known false-alarm rate?

## Method (as frozen)

- Group ROI spheres shipped with the dataset. ToM: DMPFC, MMPFC, VMPFC, PC, LTPJ, RTPJ. Pain: AMCC, L/R insula, L/R S2, L/R MFG.
- Shared ART + CompCor regressors and a linear trend removed. First 10 TRs (rest + credits) dropped, leaving 158 TRs.
- Per subject: correlate the network time course with the adult average (leaving the subject out if adult).
- Significance: 1000 circular shifts of the subject's own series, p < .05 one-sided.
- False-alarm check: the same test against the time-reversed adult average. This should come out near 5%.

## Update (follow-up checks, rules in [PREREG2.md](PREREG2.md))

- **False alarms:** the 15% in 8-12 year olds was one noisy draw per person. Against 20 mismatched references per person, the false-alarm rate is 6.4-7.6% in every age group. So p < .05 on the circular-shift null is a little liberal, and p < .033 gives 5%. At that threshold, ToM detection is: 3 yo 41%, 4 yo 29%, 5 yo 68%, 7 yo 87%, 8-12 yo 85%, adults 79%.
- **Motion:** removing spike-flagged TRs changes nothing (adults 85%, 7 yo 91%). None of the 6 missed adults is rescued by scrubbing (sub-pixar134 becomes borderline, p = .039).
- **Timing:** taking each person's lag from the pain network and applying it to ToM rescues none of the 6 missed adults. Their pain lag is 0 in 4 of the 6.
- **Personal ROIs, split half:** choosing the top 50 voxels per region on one half and testing on the other raises adult detection from 64% to 73% at half length. Children don't gain. Of the 6 missed adults, only sub-pixar136 is rescued in both halves (r = .43 and .46 vs .08 and .06 with spheres). sub-pixar155 is rescued in one half. sub-pixar132 is negative in both halves with both ROI types (r = -.18 to -.30), while its pain network tracks normally, with no motion and no timing offset.
- **Replication on IBC** (12 adults, two viewings each, false-belief localiser): about 1 in 5 single viewings miss, localiser ROIs don't fix it, and no one is missed on both viewings. See [../ibc_partly_cloudy/RESULTS.md](../ibc_partly_cloudy/RESULTS.md).

## Result (preregistered)

| Age group | n | ToM detected | Pain detected | ToM false alarms (reversed) | Pain false alarms (reversed) | Median ToM r |
|---|---|---|---|---|---|---|
| 3 yo | 17 | 47% | 53% | 6% | 6% | 0.16 |
| 4 yo | 14 | 36% | 71% | 7% | 7% | 0.15 |
| 5 yo | 34 | 71% | 88% | 6% | 9% | 0.30 |
| 7 yo | 23 | 91% | 96% | 0% | 4% | 0.40 |
| 8-12 yo | 34 | 85% | 97% | 15% | 3% | 0.39 |
| Adults | 33 | 82% | 94% | 6% | 0% | 0.40 |

- From age 7 on, one 5.6 minute scan finds a film-following ToM network in about 85-90% of people. At 3-4 it finds it in fewer than half.
- The pain network is easier to detect than the ToM network at every age. The gap is biggest at age 4 (71% vs 36%).
- Motion barely matters among children. ToM detection by tercile of spike regressors: 75%, 73%, 66%.
- ToM specificity, meaning the child's ToM series matches adult ToM better than adult pain: 71-100% by group.

## Failures and caveats, kept visible

- **6 of 33 healthy adults (18%) are missed on ToM** from 5.6 minutes. Two of them correlate negatively with the other adults (sub-pixar132 r = -0.25, sub-pixar149 r = -0.13). Their pain networks track normally (r = 0.26 and 0.32, both p = .001), so the scan data are fine. The likely cause is that group spheres miss these people's own ToM regions. Individual localisers would answer this, and this dataset doesn't include them.
- **False alarms in 8-12 year olds were 15% (5 of 34)** against the reversed reference, above the 5% the null should give. Across everyone it was 7.1% (11 of 155).
- The reference is the adult average. So "not detected" in a 3-year-old can mean "not following the film" or "following it in a non-adult way". The exploratory check below tries to tell these apart.

## Exploratory (decided after seeing the results above, not preregistered)

The reference here is the leave-one-out mean of the child's own age group, not adults:

| Group | ToM detected vs own age group | Reversed false alarms |
|---|---|---|
| 3 yo | 59% | 6% |
| 4 yo | 36% | 14% |
| 5 yo | 85% | 15% |
| 7 yo | 91% | 0% |
| 8-12 yo | 79% | 3% |
| Adults | 85% | 6% |

Against other 3-year-olds, more 3-year-olds are detected (59% vs 47%). Their ToM responses look partly shared with each other but not with adults. The age-group samples are small (14-34), and the false alarms in this check run up to 15%, so this is a lead, not a finding.

## What this means for a passive story test in patients

Even in healthy adults with no motion, about 1 in 5 are missed from ~5 minutes of film when group ROIs are used. A negative result on a short scan is not evidence of absent processing.
