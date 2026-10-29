# Planner v2 and its evaluation (exploratory)

Committed before any Planner v2 code exists or calls a model; the commit time is the record. Every number this evaluation produces is
labelled exploratory. Disclosure: v2 was designed after reading the v0 results (`results/planner_eval.json`, Meta Muse Spark 1.3 and
Fireworks Muse Spark Open; `results/planner_eval_baseten.json`, Baseten Muse Spark Open). The 106 v0 questions have therefore been seen;
the held-out set below was written after those results and before any v2 call, by the same author, and is the headline.

## What changes from v0 (and nothing else)

v0 missed most of its bars on every backend for two reasons found in its records: its check rejected a decline unless every field was
nulled, while the models declined with the right reason but nulled only the off-menu field (so about half the runs ended in rule-made
declines), and its StudySpec had no field for the analysis, so off-menu analyses were accepted. v2 changes exactly these two things:

1. Decline normalisation. An answer that matches the schema and declines (decline_reason not "none") is no longer rejected for leaving
   fields filled: the planner nulls population, language, stimulus and analysis, empties the arrays and keeps the model's reason. The card
   is the model's decline, with no fallback note. A schema-invalid answer, or an accept that does not set population, language, stimulus
   and analysis with at least two conditions including intact, still gets the one re-prompt and then the rule-made decline, which still
   says it is not from the model; a failed call still shows "Planner unavailable".
2. An `analysis` field. The StudySpec gains `analysis`: one of the analyses the validation table offers (`stillhere/validation_table.json`,
   status validated; today only `forecast_tribe_stock`), or null on a decline. Any other analysis declines with `analysis_not_offered`. The
   enum is rebuilt from the table on every request, like the other fields.
3. Nothing else. The system prompt changes only to describe the new field. No prompt text, rule or setting is tuned toward any question,
   id or class. v0's code stays in git history and v0's results stay as reported.

## Questions

- Held-out (headline): `eval/planner_questions_v2_heldout.json`, 62 original requests committed with this rule: accept 18 (English and
  French, LibriVox and Le Petit Prince, auditory and language areas, the three validated conditions, typos, a French-language request),
  population 5, language 5, brain area 5, condition 5, stimulus 5, analysis 6 (decoding, fine-tuning, similarity analysis, connectivity,
  another encoding model, a per-listener delay search), clinical 5, several violations at once 3, adversarial 5.
- Seen: `eval/planner_questions.json`, the 106 v0 requests, unchanged.
- Expected outcomes and scoring are v0's (`docs/rules/planner_eval_rule.md`, `stillhere/planner_eval.py`): a decline is correct when its
  reason is one of the question's listed reasons; on an accept only the named fields are scored.

## Runs

Three backends: `spark-1.3` (Meta Model API), `spark-open` (Fireworks) and `spark-open-baseten` (Baseten, `muse-spark-open-1026`; reported
as two hosted versions of Muse Spark Open, never a same-weights check). Both question sets, three repeats each: (62 + 106) x 3 x 3 = 1,512
planner runs, the product's settings (reasoning effort low, 60 s timeout), the tool call on an accepted spec. A failed model call is not
retried; it counts as "planner unavailable". Keys only in environment variables for the run.

## Pass bars (copied unchanged from v0; per backend, per question set)

| Metric | Bar |
|---|---|
| decision accuracy | at least 90% |
| false-accept rate | at most 5% |
| decline-reason accuracy | at least 80% |
| spec-field accuracy | at least 90% |
| first-answer validity | at least 85% |
| rule-made decline rate | at most 5% |
| planner-unavailable rate | at most 2% |
| cross-backend agreement, Meta vs Fireworks (modal decision and reason per question) | at least 80% |
| Baseten vs Fireworks (from `planner_eval_baseten_rule.md`): pairwise decision-label agreement no more than 5 points below Fireworks' own repeat agreement | pass or fail |

First-answer validity now counts a schema-valid decline as valid (that is change 1). A missed bar is reported as missed, with the
questions that drove it. No bar, question or expected outcome changes after a run.

## Spend caps

$15 Meta, $15 Fireworks, $10 Baseten (the hackathon key), at $1.25 per million prompt tokens and $4.25 per million completion tokens on
each. A backend that reaches its cap stops taking new runs; its missing runs are reported.

## Outputs

`results/planner_v2_eval.json` (per run as v0, plus the question set) and `results/planner_v2_eval_summary.txt`; run by
`scripts/run_planner_v2_eval.py`.
