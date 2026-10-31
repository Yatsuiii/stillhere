# Alice fine-tuned rule: fine-tuned TRIBE v2's forecast against the Alice Datasets listeners (exploratory, R1)

Cloud ruling of Oct 31, 08:45 UTC (R1). Committed and pushed before any fine-tuned Alice forecast exists, so before any fine-tuned
forecast meets Alice brain data; the commit time is the record. Everything below is fixed by this commit. A second commit, after the
three forecasts and before scoring, adds their SHA-256 (section 3) and nothing else. Any other change is a new commit that says what
changed and why, made before the step it changes runs. The result is shown whatever it is, labelled exploratory, on its own line
beside the stock line.

## 1. What is tested

The Alice held-out test (`docs/rules/alice_heldout_rule.md`) with one change: the forecasts come from fine-tuned TRIBE v2 instead of
stock TRIBE v2.

- Weights: `tuned.ckpt`, 708,849,658 bytes, SHA-256 96923860eb47ee8e8f65e597f2fc216992470852dc07af95112d80d32746421e, frozen
  Oct 26 (D3), made by the D22 recipe from the released checkpoint (SHA-256
  9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321). Trained on LPP sections 2-7 and 9 (33 English and 19 French
  listeners) and Narratives (training manifest `d22/TRAINING-MANIFEST.md`, SHA-256
  c63042fc8c2360bb424c55152a89b4f05256fd8751642146208c1ae58f2ba2d3, kept outside the repository). The Alice Datasets are in no
  training list, so Alice stays outside the fine-tuned model's training data as it is outside stock TRIBE v2's.
- Under D3, stock TRIBE v2 stays the primary model. This result is reported separately as "fine-tuned TRIBE v2" and never replaces
  the stock line or any stage number.
- Known before this rule, and disclosed: the stock Alice result is on origin/main (`results/alice_heldout.json`), and the Alice
  derivatives have been on this laptop since Oct 30. Every analysis choice below is the stock rule's, fixed before that result, so no
  choice is made here. Stock is not scored again (cloud ruling: never re-run Alice stock).

## 2. Data, words, volumes, timing and coverage

`docs/rules/alice_heldout_rule.md` sections 2, 3 (stimulus, words, surprisal), 4 and 5, unchanged: the same 22 listeners (sub-53
excluded), the same onsets, windows, indexing and coverage rule. Coverage is decided from the brain data alone, so the same listeners
are scored as in the stock run.

## 3. Forecasts

- Runner: `alice/forecast/run_alice_forecast_tuned.py`. It imports the stock runner (`alice/forecast/run_alice_forecast.py`)
  unchanged and adds only: the weights, loaded strictly into the built model as D3's tuned kernel did (the released checkpoint's pin,
  the tuned checkpoint's pin, a strict load, at least one tensor changed, eval mode); its own output folder,
  `alice/forecast/out_tuned` (kept outside the repository); and records. The check that the weights applied is the count of changed
  tensors, on the model that then predicts; each forecast's largest absolute difference from, and correlation with, the stock
  forecast of the same mode is recorded and must not be zero (forecast against forecast, no brain data).
- Modes: the three that `alice_heldout.py score` reads: audio-only, full, and full with the word texts shuffled across their onsets
  (seed 7). One run each on the laptop (RTX 4050), with the stock runs' placement, launched by `alice/forecast/run_all_tuned.sh`
  (kept outside the repository, SHA-256 9890610a841462c19afa8224c586a6e71df96c4fa0dc97b97c4a63a3ec9ecb2f): the stock launcher
  `run_all.sh` (unchanged, e3574a28...) with only the runner's name and the comment line changed. The runner's analysis-tree copy is
  byte-identical to the repository's.
- The sound and Llama features are read from the stock runs' feature cache. They do not depend on the brain model's weights: the
  fine-tune changed only TRIBE's brain model, and its features came from the stock pipeline (D22 recipe). The sound extractor's cache
  key is the path of the stage's `stim.wav`, so the fine-tuned runs use the stock runs' work folders, where the stock runner's
  `prepare_dir` writes the same bytes again. Each run checks that its work folder is unchanged and that no file was added to the
  feature cache (a computed feature would add one), and its stage file records it.
- A run that fails technically (a crash, out of memory, a non-finite value, a pin mismatch, a forecast equal to stock, a feature
  computed instead of read) is fixed and rerun. No forecast is compared with Alice brain data before the one scoring run, and no run is
  chosen by any comparison with it.
- The second commit pins the three forecasts' SHA-256 here and in `scripts/alice_finetuned.py` (`FORECAST_PINS`); the scorer
  refuses any other file and refuses to run before the pins exist.
- Second commit, Oct 31, after the forecasts (made 10:47 UTC by `run_all_tuned.sh`, all three runs on the first try) and before
  scoring. `alice/forecast/out_tuned/pred_alice_{full,audio,shuf}.npz`, each 744 by 20,484, finite, one row per second, 105 of 108
  tensors changed from stock, eval mode, every sound and Llama feature read from the stock cache with the work folders unchanged;
  correlation with the stock forecast of the same mode 0.870, 0.846 and 0.717. SHA-256: full
  dbdf2ee7558f195d3d1488791d1ac3c5c0ebb4faee798357a170ff8d682a7194, audio
  5593dd410b871671c8046e2af620d76333dfb02d408706ecaa4bf0cf8b24ec2e, shuffled
  fdbfa5b03f27cbf34eaeb57f4864936b295c65a848a3781eab5b1e68745322ec.

## 4. Scoring: the stock rule's, unchanged

`scripts/alice_finetuned.py` calls `alice_heldout.py`'s `score` unchanged, on an overlay folder outside the repository: links to the
same atlas, derivatives, events, stimulus, annotations and surprisal, with the forecast folder linked to `out_tuned`. It replaces only
the writer, so the stock result file is never touched: `results/alice_finetuned.json` holds the same fields as
`results/alice_heldout.json`, with `rule` naming this file, plus `model` and `vs_stock`; the per-listener numbers stay outside the
repository. One scoring
run. Section 6 of the stock rule applies unchanged: lag, cleaning, nulls, M1, the claim through `volunteer_equivalent`, the
baselines and the cheap baselines.

## 5. Comparison with stock (descriptive, decides nothing)

Per listener, fine-tuned minus stock LANGUAGE z and HEARING z, in the claim window at the development lag (-1.25 s), over the
listeners both runs scored: the mean difference, the number of listeners where fine-tuned is higher, and a two-sided Wilcoxon p.
Stock's per-listener numbers are read from the stock run's output file (`alice/score/alice_heldout.json`, kept outside the
repository), only if its recorded forecast SHA-256 are the stock rule's three pins, which rules out the first run's file; the scorer
checks this before it scores. It is written to the results file as `vs_stock`; no line says the fine-tune helped or hurt.

## 6. Outcomes and their lines (written before the result)

- The lines of `docs/rules/alice_heldout_rule.md` section 7, word for word, each starting with "Fine-tuned TRIBE v2: ". The stock
  line stays as recorded, unchanged, beside it.
- Not on origin/main by Mon Nov 2, 12:00 UTC: "Fine-tuned TRIBE v2: Alice Datasets: rule committed, not run."
- In the write-up, the line carries this rule's commit time.

No other line describes this result. Not a diagnosis or medical advice.

## 7. Code committed with this rule

`alice/forecast/run_alice_forecast_tuned.py`, `scripts/alice_finetuned.py` and `tests/test_alice_finetuned.py`. The tests use
synthetic data and a toy model only: the strict load, both pin refusals, an extra key and training mode refused, the
forecast-equals-stock refusal (these need torch); the overlay links, the refusal without pins or with a different forecast, the
stock-file check, the writer leaving `results/alice_heldout.json` untouched, the line prefix, and the paired comparison against
scipy's Wilcoxon (these need scipy only and run in CI). An independent read-only review before this commit found the sound-cache key
issue fixed above. `scripts/alice_heldout.py`, `alice/forecast/run_alice_forecast.py` and `run_all.sh` are unchanged.

## 8. Order and cutoff

This commit on origin; the three forecasts on the laptop, never during a chain step that uses the laptop (gate 0, the film maps,
the picker merge and scoring) and never during Sunday's filming; the second commit with the pins; one scoring run; the results
commit. On origin/main by Mon Nov 2, 12:00 UTC, or the not-run line.


## Dated note, Oct 31, after the result (cloud ruling of Oct 31, 21:30 UTC, items 9-11; the text above is unchanged; the result stands)

- Section 5 says the stock-file pin check rules out the first run's file. It cannot: both stock runs share the same three forecast
  pins (`results/alice_heldout_first_run.json` lists the same SHA-256). The committed `vs_stock` means match the corrected run of record
  exactly: stock LANGUAGE 1.4153 and HEARING 3.0278, equal to `results/alice_heldout.json`'s claim-window group z. From this note
  `stock_scores` also requires the file's claim-window group LANGUAGE and HEARING z to equal `results/alice_heldout.json`'s, so the
  first run's file is refused; the test uses the first run's real pins (`tests/test_alice_finetuned.py`). No number changed and
  nothing was re-scored.
- Forecast provenance committed: `results/alice_finetuned_provenance.json` (`scripts/alice_finetuned_provenance.py`): the three stage
  records (counts, hashes and placement; no brain data, no word text), their UTC start and end (audio 10:46:26-10:46:45, full
  10:46:45-10:47:09, shuffled 10:47:09-10:47:33), the SHA-256 of the runner, the stock runner, `k2/tribe_common.py` and the launcher,
  and asserts that no Llama placement was used (every Llama feature was read from the stock runs' cache).
- The README row carries a numbers-only footnote on the volunteer figures and the one comparison not beaten.
