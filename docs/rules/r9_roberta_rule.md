# R9 rule: a third model, ridge on RoBERTa features, through the same Paranoia scorer (exploratory)

23 Nov 2026. Committed and pushed before any RoBERTa feature is computed and before any fit. The commit time is the record. A change
is a new commit that says what changed and why, made before the step it changes runs. The result is shown whatever it is, labelled
exploratory.

## 1. Why

R8 put a ridge model on static word vectors beside TRIBE v2, and it was worth about 0 listeners. That is a floor, not a rival. Ridge
regression on a language model's contextual features is the standard encoding model in the field, so R9 scores one, open and
CPU-sized, the same way.

## 2. The model (fixed here)

Everything is as in R8 (`docs/rules/r8_ridge_rule.md` with its amendment, `scripts/r8_ridge.py`), with one change: the word features.

- Word features: RoBERTa-base last hidden layer, as packaged in spaCy's `en_core_web_trf` 3.8.0 (its own strided windows; read with
  spaCy 3.8.7, offline). Each story's word list is passed as one pre-tokenised document, in order, so word i keeps its onset. A word's
  vector is the mean of its word-piece rows (768 values). A 1 s bin gets the mean vector of the words whose onset falls in it (zeros if
  none), plus the word count, exactly as R8 bins its vectors.
- Same training data, stories, cleaning, z-scoring, delays (2 to 8 s), alpha grid (10^1..10^6, 11 values), leave-one-story-out choice
  of alpha, and the same 5 s offset in the saved forecast.
- "shuf": the story's words in R8's shuffled order (seed 20261122), timings kept, run through RoBERTa in that order, then the full
  model's weights.
- "audio": R8's envelope-only model, reused unchanged (`ridge_E.npz`).

## 3. Scoring

`scripts/r2_paranoia.py score --verify-own DIR`, unchanged, as in R8.

## 4. Lines

- "Exploratory. On Paranoia (22 listeners), a ridge model on RoBERTa-base features, trained on 8 Narratives stories, was worth about
  {k} real listeners in language areas (at least {klow}). Stock TRIBE v2, scored the same way: about 3 (at least 1.9). Word-vector
  ridge (R8): about 0."
- If any step fails: "R9: rule committed, not run ({reason})."

Healthy volunteers only. Not a diagnosis or medical advice.

## Result, 23 Nov 2026 (exploratory)

Exploratory. On Paranoia (22 listeners), a ridge model on RoBERTa-base features, trained on 8 Narratives stories, was worth about 0
real listeners in language areas (at least 0). Stock TRIBE v2, scored the same way: about 3 (at least 1.9). Word-vector ridge (R8):
about 0. Beyond that headline, RoBERTa did better than R8's word vectors: its forecast tracked the listeners' language areas above
chance (language z 0.84, p = 0.0001, though under the bar the scorer sets, and only 3 of 22 listeners detected one at a time), it
followed the intact story better than the shuffled one (p = 0.02), its share of the real-volunteer reference was 12% (TRIBE v2 59%,
R8 0%), and it beat word rate (15 of 22 listeners, p = 0.004) but not the envelope or surprisal (1 of 4 baselines). Leave-one-story-out
training correlation 0.037 at alpha 10^4.5. Files: `results/r9_roberta_paranoia.json`, `results/r9_roberta_fit.json`.
