# How many real people is Meta's brain model worth?

Raghav, November 2026. Every number below is in this repository's `results/` folder at commit 4ea23d0. Every rule was committed before the scans it scores were downloaded. Healthy volunteers only. Nothing here is a diagnosis or medical advice.

## Why ask

Some people who can't respond after a brain injury still follow what they hear. Bodien et al. (NEJM 2024) found command-following brain activity in about one in four people who showed none at the bedside. One way to look for it is to play a story in the scanner and check whether the person's language areas follow it. But every new story first needs its own healthy reference, usually 12 to 15 volunteers scanned on it, in that language.

Meta's TRIBE v2 forecasts how an average healthy brain responds to any story. If that forecast could stand in for the volunteers, any lab could build a test in any language without first scanning volunteers. So the question is how many real volunteers the forecast is worth.

## How it's scored

For each listener, we compare their language areas with two references. One is the forecast. The other is the average of k other real listeners, for k = 1, 2, 3 and up. A forecast is "worth N volunteers" when it matches a new listener as well as the average of N real ones. The cheap baselines it must beat are word rate (words per second) and the sound envelope.

## Results on stories it was never trained or tested on

| Story (data from) | Listeners | Minutes | Worth (lower bound) | Per person, zero volunteers |
|---|---|---|---|---|
| Paranoia (Finn et al. 2018) | 22 | 20.8 | 3.0 (1.9) | 21 of 22 detected; beat word rate for 22 of 22 |
| Le Petit Prince section 8, held out (Li, Hale & Pallier) | 49 | 4.8 | 5.0 (2.6) | 10% false alarms on a wrong story, over the 7% limit |
| Alice (Bhattasali, Brennan et al.) | 22 | 12.5 | 1.6 (0) | 16 of 22 detected; 8 of 22 on the first 4.8 minutes |
| Emo-FilM, Tears of Steel (Morgenroth et al.) | 16 | 3 | under 1 | language areas no better than word rate |

There was also one pre-registered test where the truth is known for every listener. On Narratives "Not the Fall" (Chien & Honey 2020), 22 listeners heard the story intact and with its sentences shuffled. The forecast matched them better on the intact story (p = 0.005). Word rate alone did not (p = 0.61). Against a coarser scramble, the test did not pass (p = 0.07).

On sedation data (Kandeepan, Owen, Naci, Soddu et al., OpenNeuro ds003171), the main pre-registered bar failed: 9 of 12 people, where 10 were needed. Telling each person's awake scan from their deeply sedated one, the forecast scored AUC 0.80, against 0.86 with real volunteers as the reference.

Across 40 public story clips (23 Narratives stories and 17 Le Petit Prince sections), the forecast followed its own listeners better than other stories from the same dataset on 34. This part is descriptive only: its wrong-label check came out at 6 of 33, over its limit.

## A forecast plus a few volunteers

On Le Petit Prince section 8, adding the forecast to a small real reference helped a lot:

| Reference | Match (z) | False alarms on a wrong story |
|---|---|---|
| 1 volunteer | 0.92 | |
| Forecast + 1 volunteer | 1.71 | 6% |
| 5 volunteers | 1.78 | |
| Forecast + 3 volunteers | 1.88 | 2% |
| 15 volunteers | 2.42 | |

So the forecast plus 1 volunteer does about the job of 5, and the forecast plus 3 about the job of 6. It halves the reference. It does not replace one.

## What this means

- The forecast can't replace a story's volunteers yet, and short scans are where it is weakest.
- On a long scan it detects nearly every healthy listener with no volunteers at all.
- Its value differs a lot from story to story (under 1 to about 5). A lab can't know in advance which kind of story it has, which is why a public benchmark is needed.

## Next

- Score a second, simpler model (a ridge regression on word features) through the same scorer, so TRIBE has something to be ranked against.
- Run the same per-person test on every public story-listening dataset, one row per story and lab.
- Run the Mandarin Le Petit Prince section, so only the language changes.

If your data is in here and something looks wrong, please open an issue. I'd rather fix it before the release.
