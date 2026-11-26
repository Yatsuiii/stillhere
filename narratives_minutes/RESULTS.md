# Healthy miss rate of a passive story test, by scan length (Narratives)

> **Superseded headline (26 Nov stress test, below).** The 18% five-minute miss rate is a property of our weak readout (one averaged left-hemisphere time course), not of passive story tests. A parcel-by-parcel readout over both hemispheres' language parcels misses 3.5% at 5 minutes. That readout was chosen after seeing results, so it needs a held-out test before any claim.

Rules: [PREREG.md](PREREG.md), committed before any BOLD was downloaded. Code: [extract.py](extract.py), [score.py](score.py). Full output: [score.log](score.log), [summary.json](summary.json).

**Data.** Narratives (Nastase et al. 2021), 7 stories at least 13 minutes long (black, forgot, merlin, sherlock, tunnel, 21styear, slumlord). This gives 226 listener runs from 142 healthy adults, after the dataset's own exclusions.

**Test.** One person against 20 healthy listeners of the same story, with no model. The score is the correlation of the person's left language-network time course with the group's over the first k minutes. Each p value comes from 1000 phase-randomised surrogates. The cutoff is calibrated so that scoring each person against a different story's group gives a 5% false-alarm rate.

## Result

| Minutes of story | Language network: healthy miss rate [95% CI] | Hearing network: miss rate |
|---|---|---|
| 1 | 65% [59, 71] | 37% |
| 2 | 36% [30, 42] | 8% |
| 3 | 29% [24, 35] | 7% |
| **5** | **18% [14, 24]** | 3% |
| 7 | 9% [6, 13] | 1% |
| **10** | **6% [4, 10]** | 0.4% |
| 13 | 4% [2, 7] | 0% |

The CIs are Wilson intervals. A bootstrap that resamples people, since some heard more than one story, gives nearly the same intervals: 13-24% at 5 minutes and 4-9% at 10. The wrong-story false-alarm rate is 5% at every length by construction. Without calibration, p < .05 gives the same detection rates within a few points, and its wrong-story rate stays between 3% and 8%.

Per story at 5 minutes, the language miss rate ranges from 0% (21styear) to 29% (merlin). At 10 minutes it ranges from 0% to 13% (forgot).

## What it means

- With 5 minutes of story, about 1 in 5 healthy young adults does not pass a calibrated language-network test. With 10 minutes, it is about 1 in 16.
- The hearing network passes much sooner, at about 1 in 30 missing at 5 minutes. So a patient whose hearing network passes but whose language network doesn't, on a 5-minute scan, is in a range where many healthy people land too.
- These are healthy, mostly young adults lying still in a research scanner. Patients move more and vary in arousal, so their miss rate at a given length will be higher. Treat these numbers as a floor.
- A pass means only that the language-network time course followed the healthy group's. It says nothing about awareness.

## Stress test (26 Nov, exploratory, after the result was challenged)

Same 226 runs, same wrong-story calibration (5% false alarms). Scripts: [robust1.py](robust1.py), [robust2.py](robust2.py) (parcel series from [extract400.py](extract400.py)); output [robust2.log](robust2.log).

| Readout, first 5 minutes | Miss rate |
|---|---|
| Left language network, averaged (the frozen test) | 18% (16% in the re-extraction) |
| Same, minutes 5-10 instead of 0-5 | 11% |
| Same, all other listeners as reference instead of 20 | 16% |
| Same, 5 reference listeners | 32% |
| Both hemispheres, averaged | 18% |
| Left language parcels, each parcel scored then averaged | 7.5% |
| **Both hemispheres' language parcels, parcel by parcel** | **3.5%** (10 min: 1.3%) |
| All 400 parcels, parcel by parcel | 21% |

Also: misses moved slightly more (median FD 0.15 vs 0.12 mm); 39 of 41 five-minute misses passed the hearing test; 32 of 41 passed the language test at 13 minutes.

**What holds:** with this design, how the network signal is read out, which minutes are used and how many reference listeners exist change the healthy miss rate several-fold. **What doesn't:** "about 1 in 5 healthy people fails a 5-minute passive language test". That number described our readout. The 3.5% figure was picked from 8 variants after seeing them and must be confirmed on held-out stories before it is quoted.
