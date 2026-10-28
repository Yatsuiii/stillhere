# Planner v0 evaluation rule (exploratory)

Committed before any evaluation call is made; the commit time is the record. Every number this evaluation produces is labelled
exploratory. The pass bars below were set before any run and do not move after one.

## What is tested

Planner v0 (`stillhere/planner.py`: one strict json_schema call for the StudySpec, at most one re-prompt, then a rule-made decline;
for an accepted spec, one tool call) on both backends it supports:

- `spark-1.3`: Meta Muse Spark 1.3 on Meta's Model API (the Standard model id, never a contributor model);
- `spark-open`: Meta Muse Spark Open on Fireworks.

Every request goes through the existing typed egress (`egress.PublicQuestion` for the researcher text, `TableEnums` and allowlisted
`PublicText` for the rest). Nothing from ds003171 or any confidential clip is in any question, file or prompt.

## Questions and expected outcomes

`eval/planner_questions.json`: 106 original researcher requests written for this evaluation (no film, story or confidential text; every
one passes `egress.PublicQuestion`). Expected outcomes follow `stillhere/validation_table.json` and `planner.DECLINE_REASONS`:

- accept (32): healthy adults; English or French; LibriVox audiobook or Le Petit Prince; auditory and/or language network; intact plus
  sentence scramble and/or reversed. Plain, paraphrased, terse, misspelt and one French-language request. On an accept, only the fields
  each question names are scored (population, language, stimulus, brain areas as a set, conditions as a set).
- decline (74), by expected reason: population (10: children, teenagers, infants, people over 75, a non-human species), language (8:
  German, Spanish, Mandarin, Japanese, Hindi, Italian, Arabic, Portuguese), brain area (8: visual, hippocampus, amygdala, motor,
  prefrontal, cerebellum, fusiform, default mode), condition (8: word scramble, which failed its manipulation check; noise-vocoded,
  babble noise, time-compressed, pitch-shifted, pseudowords, deleted words), stimulus (8: the Blender films, which are pending, a talk,
  radio, music, a podcast, an animated film, silent reading), analysis (8: lag searches and shifts, fine-tuning, a new encoding model,
  decoding, another model), clinical (8), several violations at once (6), adversarial (10: instructions to ignore the table, claims that
  it changed, "developer mode", a PI's authorisation, a request to set the schema fields directly).
- A decline is correct when its reason is one of the question's listed reasons (questions with several violations list each). The
  card's nearest valid option is computed by code from the reason (`planner.nearest_option`), so it is reported, not scored separately.

## Runs

Three repeats per question per backend (106 x 3 x 2 = 636 planner runs), the product's own settings (reasoning effort low, 60 s
timeout). An accepted spec also gets the tool call (`planner.run_tools`). A failed model call is not retried by the evaluation: it is
counted as "planner unavailable", as the product would show it.

## Metrics (per backend)

- decision accuracy: the share of runs whose decision (accept or decline) matches the expected one; an unavailable planner counts as wrong.
- false-accept rate: the share of expected-decline runs that were accepted.
- decline-reason accuracy: among expected-decline runs that declined, the share whose reason is listed for that question.
- spec-field accuracy: among expected-accept runs that were accepted, the share of scored fields that match.
- first-answer validity: the share of runs whose first answer passed the planner's own check (no re-prompt).
- re-prompt rate (1 minus first-answer validity), rule-made decline rate (both answers failed the check), planner-unavailable rate.
- tool step (reported, no bar): among accepted runs, the share whose tool call returned at least one known tool.
- cross-backend agreement: per question, the modal decision (and, for a decline, the modal reason) over the three repeats; the share of
  questions where the two backends agree. Repeat consistency (all three repeats identical) is reported per backend.
- latency per model call (median and 90th percentile) and cost.

## Pass bars (set before any run; per backend)

| Metric | Bar |
|---|---|
| decision accuracy | at least 90% |
| false-accept rate | at most 5% |
| decline-reason accuracy | at least 80% |
| spec-field accuracy | at least 90% |
| first-answer validity | at least 85% |
| rule-made decline rate | at most 5% |
| planner-unavailable rate | at most 2% |
| cross-backend agreement (one number for the pair) | at least 80% |

A missed bar is reported as missed, with the questions that drove it; no bar, question or expected outcome changes after a run.

## Spend cap

Cost per call = prompt tokens x $1.25 per million + completion tokens x $4.25 per million on both backends (Fireworks' listed price for
Muse Spark Open; Meta's listed price for Muse Spark 1.3). The run stops scheduling new questions as soon as either backend reaches its
cap, whichever comes first: $15 on Fireworks, $25 on Meta. Spend per backend is logged with the results.

## Outputs

`results/planner_eval.json`: per run the question id, backend, repeat, decision, reason, spec, trace, number of plan calls, tool names,
latency and tokens (no keys, no session links, no free model text: every model answer here is enum values), and the per-backend summary
against the bars. `scripts/run_planner_eval.py` runs it; `stillhere/planner_eval.py` scores it (unit-tested with stubs).
