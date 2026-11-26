# Healthy miss rate of a passive story test, by scan length (Narratives)

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
