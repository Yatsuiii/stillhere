# Alice EEG: what the one-segment onset shift does (NEMAR nm000180)

Protocol: [PROTOCOL.md](PROTOCOL.md), with three deviations listed at its end. Code: [analyze.py](analyze.py). All three runs are kept: [run1.log](run1.log), [run2.log](run2.log) and [run3.log](run3.log). The onset check (per_segment.csv) was rerun here on the same pinned files and finds the same 8 shifted recordings as on 23 Nov.

**Read this first.** The protocol changed twice after results were seen.
- Run 1 used no cleaning. It failed a sanity check that doesn't involve the shifted subjects: normal subjects' word ERPs didn't resemble each other.
- Run 2 used the 2019 ICA, channels and retained trials. But those trials were screened at the shifted times, which left unscreened artifacts at the corrected times.
- Run 3 adds one amplitude rejection, applied the same way to both versions.

The numbers below are from run 3 and are exploratory.

## M1: word-onset ERP (model-free)

Each subject's average word ERP is correlated with the mean ERP of the 31 consistent subjects (leave-self-out). The consistent subjects' median is 0.52, with a 10-90% range of 0.03 to 0.71.

| Recording | Analyzed in 2019 | Released onsets | Corrected onsets |
|---|---|---|---|
| sub-002 | no | -0.07 | +0.11 |
| sub-026 | yes | +0.08 | +0.52 |
| sub-027 | no | +0.23 | +0.56 |
| sub-030 | no | -0.18 | +0.08 |
| sub-032 | no | +0.35 | +0.60 |
| sub-034 | yes | +0.27 | +0.33 |
| sub-035 | yes | -0.10 | +0.42 |
| sub-036 | yes | -0.47 | +0.56 |

All 8 move toward the normal range when words are placed on the recorded audio. Run 2 shows the same direction in 6 of 8.

## M2: n-gram surprisal beta map, similarity to the consistent subjects

The consistent subjects' median is 0.33. With corrected onsets, 6 of 8 recordings improve (026, 027, 030, 034, 035, 036). Two get worse (002 and 032).

## M3: group effect in the analyzed sample (our pipeline, not the 2019 one)

The window was picked using only the analyzed consistent subjects.

- **N-gram surprisal** (channel 49, 200-250 ms): t(24) = 4.71 with released onsets and 7.10 with corrected onsets.
- **CFG surprisal beyond n-gram** (channel 48, 500-550 ms): t(20) = -1.51 with released onsets and -1.66 with corrected. It is not significant either way in this pipeline.

## What this supports

- In the 8 recordings, the released onsets lock the epochs to the wrong words. Re-placing the words on the recorded audio restores a normal word ERP in all 8.
- In this pipeline, fixing the 4 analyzed recordings strengthens the group n-gram effect, the opposite of what you'd expect if the error were harmless. This rests on run 3, after deviations.
- It says nothing about whether the 2019 conclusions hold. The 2019 pipeline and statistics weren't rerun.
