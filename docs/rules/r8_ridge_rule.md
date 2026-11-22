# R8 rule: a second, simple model through the same Paranoia scorer (exploratory)

Raghav's call, 22 Nov 2026. Committed and pushed before any ridge model is fitted, before any Narratives training scan is downloaded
for it, and before the Paranoia scans are fetched on this machine. The commit time is the record. A change is a new commit that says what
changed and why, made before the step it changes runs. The result is shown whatever it is, labelled exploratory.

## 1. Why

"Worth N volunteers" means little with one model. R8 scores a simple, open encoding model with the exact R2 scorer
(`scripts/r2_paranoia.py score --verify-own`), so TRIBE v2 has something to be ranked against.

## 2. The model (fixed here)

- Features per 1 s bin of story time: the mean 300-d word vector (spaCy `en_core_web_md` 3.7.1, lower-cased token, zero vector when
  out of vocabulary) of the words whose onset falls in the bin, zeros for a bin with none; plus word count in the bin. The "audio"
  forecast uses the sound envelope (mean absolute amplitude per bin, the scorer's `envelope_per_second`) alone. The "shuf" forecast uses
  the full model's weights on the story's words in a shuffled order (one permutation, seed 20261122), timings kept.
- Each feature is z-scored over training bins, then given finite-impulse-response delays of 2, 3, 4, 5, 6, 7 and 8 s.
- Training data: Narratives (OpenNeuro ds002345), the fMRIPrep fsaverage6 surface derivative, first 10,242 vertices per hemisphere
  (fsaverage5). Stories: pieman, lucy, tunnel, slumlordreach, merlin, prettymouth, black, forgot. Per story, up to 20 listeners in
  participant order who pass the dataset's own exclusion list, each run cleaned (linear detrend, high-pass 0.01 Hz, z-score), then
  averaged. Words and timings from the dataset's Gentle alignments. BOLD at TR 1.5 s is linearly interpolated to 1 s bins after
  cleaning; the first 10 s of each story are dropped.
- Ridge regression per vertex, one alpha for all vertices, chosen from 10^1..10^6 (11 log-spaced values) by leave-one-story-out mean
  correlation over all vertices.
- Output: a 1 Hz forecast `P` (seconds by 20,484 vertices) with `starts`, in the R2 forecast file format. Because the ridge forecast
  already contains its delay and the scorer reads every forecast at story time j - 2 - 5 s, the saved forecast at second s is the ridge
  prediction for second s + 5.
- Paranoia words and timings: the word tables published with the R2 release (`paranoia_files.py fetch-release`), checked against their
  pins. No Paranoia scan is read before scoring.

## 3. Scoring

`scripts/r2_paranoia.py score --verify-own DIR` with DIR holding the ridge's nine forecasts under the stock model's file names. Every
other input must equal what R2's scoring read (the script enforces this). Reported: the ridge's volunteer-equivalent in language areas
(point, lower bound), its share of the real-volunteer reference, and its comparisons with word rate and the envelope, beside TRIBE v2's
committed R2 numbers.

## 4. Lines

- "Exploratory. On Paranoia (22 listeners), a ridge model on word vectors, trained on 8 Narratives stories, was worth about {k}
  real listeners in language areas (at least {klow}). Stock TRIBE v2, scored the same way: about 3 (at least 1.9)."
- If any step fails: "R8: rule committed, not run ({reason})."

Healthy volunteers only. Not a diagnosis or medical advice.

## Dated amendment, 22 Nov 2026 (before any training scan is downloaded)

- Training scans: the dataset's own fMRIPrep **fsaverage5** surface files (`space-fsaverage5`, found on the bucket after the rule was
  committed), in place of the first 10,242 fsaverage6 vertices. Paranoia's scorer reads fMRIPrep fsaverage5 too, so this matches it.
- One run per listener: a listener's first run of a story (pieman has a second hearing for some listeners, which is not used).
- spaCy: the `en_core_web_md` 3.7.1 vectors are read with spaCy 3.8.7 (3.7 has no build for this machine's Python 3.13). Only the
  vector table is used, not the pipeline.
