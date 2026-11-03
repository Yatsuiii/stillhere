# Forecast card: TRIBE v2 as StillHere uses it

## Summary

StillHere runs stock TRIBE v2 (Meta FAIR; d'Ascoli et al. 2026, arXiv 2605.04326) to forecast the healthy-adult brain response to a
story or a film. On a story outside TRIBE v2's training and evaluation sets (Paranoia, 22 listeners; exploratory), the forecast was
worth **about 3 real volunteers, and at least 1**, and the minutes it ranked highest gave listeners' language areas a closer match than
the minutes it ranked lowest for 17 of 22 listeners (p = 0.0004). On a held-out audiobook from TRIBE v2's own evaluation sets (Le
Petit Prince, section 8, 49 listeners), it was worth about 5, and at least 2. On a different story with real listeners, its
pre-registered intact-versus-shuffled test passed (Narratives notthefall, 22 listeners, p = 0.00516).

StillHere is a research tool, tested only in healthy volunteers. It is not a medical device. Not a diagnosis or medical advice.
TRIBE v2 is CC BY-NC 4.0 and is not included in this repository.

## Intended use

- Piloting a new story for a listening study before booking scanner time: a first healthy reference, and evidence for how far to
  trust it.
- Checking that a control condition is what it claims to be before anyone is scanned (Muse Voice Transcribe reads it back).
- Reproducing or critiquing the reported tests under their committed rules (`docs/rules/`, `PREREG.md`).
- Teaching the forecast-against-scan comparison with public data.

Out of scope: anything about one patient's scan (the planner declines it, with the reason); clinical decisions of any kind; claims
about a person from a forecast; treating a forecast as a scan. Every live run ends on the gray forecast alone: "Forecast, not a
scan."

## Configuration

| Component | Setting |
|---|---|
| Model | stock TRIBE v2 checkpoint |
| Text lane | Llama 3.2 3B in fp16, split between the GPU and the CPU by accelerate |
| Sound lane | w2v-BERT 2.0, on the CPU |
| Video lane | V-JEPA 2: off for audio clips; used for the film forecasts |
| Output | fsaverage5 surface, 20,484 vertices |
| Brain areas | Schaefer-400, 17 networks. Hearing: SomMotB_Aud, both hemispheres. Language: left TempPar, DefaultB_Temp, DefaultB_PFCv and DefaultB_PFCl (`stillhere/heldout.py`) |
| Timing | a scan volume reads the forecast 5 s earlier (TRIBE's own offset) |
| Live forecasts | a laptop RTX 4050 (6 GB) in a sandboxed worker; new seeds run 10:00-14:00 ET, Nov 4-9; a saved forecast otherwise |
| Forecasts scored against scans | private Kaggle GPU notebooks |

## Evaluation data

All public, on OpenNeuro. Each result is shown beside cheap baselines (word rate and the sound envelope; for film, brightness and
motion too).

| Dataset | Test | People | Result |
|---|---|---|---|
| Le Petit Prince (ds003643), section 8 | held-out audiobook | 49 listeners | about 5 real volunteers' worth, at least 2 (first measured Oct 22-23; LPP is one of TRIBE v2's own evaluation sets) |
| Narratives (ds002345), notthefall | intact vs sentence-shuffled, pre-registered (D20) | 22 listeners | pass, p = 0.00516; the secondary did not reach p < 0.05 |
| Paranoia (ds001338, Finn et al. 2018) | a story outside TRIBE v2's training and evaluation sets, the whole story (R2) | 22 listeners | about 3 real volunteers' worth, at least 1; 59% of what the other listeners' average reaches; beaten baselines 4 of 4 (exploratory) |
| Paranoia, the same listeners | which minutes to play in a short scan (R5) | 22 listeners | TRIBE's 7 top-ranked minutes against its 7 bottom-ranked: 17 of 22, p = 0.0004; word rate alone 8 of 22, p = 0.94 (exploratory) |
| The Alice Datasets (ds002322), chapter 1 | a second audiobook outside TRIBE v2's training and evaluation sets | 22 listeners | tracked the listeners' language areas (z 1.42, p = 6e-05), but worth less than one real volunteer at its lower bound (exploratory) |
| The Alice Datasets, the same listeners | which minutes to play, a replication attempt of R5 (R6) | 22 listeners | TRIBE's 4 top-ranked minutes against its 4 bottom-ranked: 12 of 22 (bar 16), p = 0.34, not replicated; word rate alone 16 of 22, p = 0.099 (Exploratory; the Alice scans were read on Oct 30 for the held-out test) |
| Narratives and Le Petit Prince | story picker and story wall, pre-registered (PREREG section 9, D19) | 37 story families in 4 blocks | picker: S1 rho 0.54, p = 0.0007; S3 partial rho 0.63, p = 0.0005; S4 did not pass. Wall: 29 of 33 families, bar met, descriptive (wrong-label check 6 of 33, above its 7% limit) |
| Emo-FilM (ds004892, ds004872), Tears of Steel | intact film | 16 viewers | followed hearing (median z 3.23) and visual areas (2.77) more closely than sound, brightness or motion; no better than word rate in language areas (0.80); worth less than one real viewer in every area (exploratory) |
| Emo-FilM, Tears of Steel | face test | 16 viewers | missed the spin test (12 of 16 above 0, 12 needed; spin p = 0.092): fail (exploratory) |

## Limitations

- Healthy volunteers and public data only.
- The held-out audiobook was first measured Oct 22-23, before this repository began, and LPP is one of TRIBE v2's own evaluation
  sets.
- For notthefall, each version also had its own TRIBE forecast, so a pass does not separate a change in the listeners' brains from a
  change in TRIBE's forecast, and the run order is not public, so it cannot be separated from an order or repeat-listening effect.
- Cheap baselines carry much of what a forecast seems to capture, so every result is shown beside them.
- Never tested in children or patients; it does not map left versus right.
- The live forecast runs on one 6 GB laptop GPU, in set hours only.

Cite with `CITATION.cff`. Every result and its file: the README's "Results so far".
