# Does a repeat viewing help per-person detection? 30 adults, 6 viewings (OpenNeuro ds001545)

The rules are in [../partly_cloudy/PREREG5.md](../partly_cloudy/PREREG5.md), frozen before download. Code: [aly_pipe.py](aly_pipe.py) and [aly_analyze.py](aly_analyze.py). Output: [aly_results.txt](aly_results.txt).

Data: Aly et al. 2018. 30 adults watched the same 90 s intact clip from The Grand Budapest Hotel 6 times, twice in each of 3 runs. The test uses the same per-person method as for Partly Cloudy: ToM and pain spheres, a leave-one-out reference, a phase-randomised null, and a cutoff calibrated to 5% false alarms (measured 4-6%).

Registration Dice is 0.92-0.96. The field of view cuts off DMPFC in 23 subjects and MMPFC in 7. Those regions are dropped for those subjects, and the network is the mean of the regions available.

## Preregistered result (intact clip)

| | ToM | Pain |
|---|---|---|
| Viewing 1 alone | 11/30 (37%) [22-54] | 16/30 (53%) [36-70] |
| Viewing 2 alone | 10/30 (33%) [19-51] | 7/30 (23%) [12-41] |
| Viewings 3-6 alone | 2-6/30 (7-20%) | 3-5/30 (10-17%) |
| First 2 averaged | 12/30 (40%) [25-58] | 16/30 (53%) [36-70] |
| First 3 averaged | 12/30 (40%) | 19/30 (63%) |
| All 6 averaged | 17/30 (57%) [39-73] | 17/30 (57%) [39-73] |

- **A second viewing adds almost nothing here.** Two averaged give 40% vs 37% for one (ToM), and 53% vs 53% (pain). Only averaging all six raises ToM detection, to 57%.
- **Each extra viewing is worth less than the first.** The pain network falls from 53% on the first viewing to 23% on the second. Viewings 1 and 2 are in the same run, so this isn't fatigue across runs. ToM holds through the second viewing (37% then 33%) and then falls.
- **Control:** the scrambled clip, shown in the same scrambled order each time, is rarely detected in ToM (3-13%). So the intact-clip signal is about the story, not the visuals.

## Exploratory (not preregistered)
- Using a position-matched reference (each viewing against the other people's same-numbered viewing) does not bring later viewings back: ToM 40, 27, 17, 13, 3 and 20% across viewings 1-6.
- So the drop isn't just responses shifting earlier in time. The film-locked response that people share fades with repetition.

## What this does to the IBC two-viewing result
- In IBC (12 adults, a 5.8 minute film, two viewings), averaging raised detection from about 80% to 92-96%.
- Here, with 30 adults and a 90 s clip shown 6 times, the second viewing adds little.
- The two disagree. The likely factors are clip length, how many repeats there are, and n = 12. The two-viewing gain is therefore **not** a safe general recommendation.
- What does replicate is that **the first viewing is the most informative one**, and that pain-network tracking falls off fastest when the film is repeated.
