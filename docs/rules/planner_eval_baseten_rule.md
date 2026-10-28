# Planner evaluation, Baseten addition (exploratory)

Project-thread ruling of Oct 28, 21:45 UTC (item 2). Committed before any planner request is sent to Baseten and before any result of
the running evaluation (`docs/rules/planner_eval_rule.md`, Meta Muse Spark 1.3 and Fireworks Muse Spark Open) has been read; the commit
time is the record. Disclosed: before this rule, two one-word key checks ("Reply with the single word OK.") were sent to Baseten, one per
key; no planner request, schema or question was sent. Every number here is labelled exploratory.

## What is added

A third backend, `spark-open-baseten`: Meta Muse Spark Open served by Baseten (`meta-models/muse-spark-open-1026`, the hackathon key,
OpenAI-compatible endpoint `inference.baseten.co`), through the same typed egress as the other two. `inference.baseten.co` joins the
product's allowed API hosts with a test; Baseten receives exactly what Fireworks receives (never tier-D). The key is read into an
environment variable for the run only and never goes in a log, commit, file or screenshot.

## Same release or two hosted versions

Before the run, Fireworks' `muse-spark-open` and Baseten's `muse-spark-open-1026` are compared by their model cards or version strings.
Only if they name the same release is this reported as a same-weights check across providers. Otherwise every result here is reported
as "two hosted versions" of Muse Spark Open, never as a same-weights check.

## Runs

The same 106 requests (`eval/planner_questions.json`), the same system prompt, schema and sampling settings as the running evaluation
(planner v0, reasoning effort low, 60 s timeout, at most one re-prompt, then a rule-made decline; the tool call on an accepted spec),
three repeats each: 318 planner runs. A failed model call is not retried; it counts as "planner unavailable".

## Pass bar (set before any Baseten call)

Each run's decision label is the in-menu choice for an accept (population, language, stimulus, brain areas as a set, conditions as a
set) or the decline type for a decline ("planner unavailable" is its own label).

- Fireworks repeat agreement: over the 106 questions and the three pairs of Fireworks repeats per question, the share of pairs whose
  labels are identical.
- Baseten-vs-Fireworks agreement: over the 106 questions and the nine Baseten-repeat x Fireworks-repeat pairs per question, the share
  whose labels are identical.
- Pass: Baseten-vs-Fireworks agreement is no more than 5 percentage points below Fireworks repeat agreement.

Reported beside it, with no further bar: the running evaluation's metrics for Baseten (decision accuracy, false-accept rate,
decline-reason accuracy, spec-field accuracy, first-answer validity, re-prompt, rule-made decline and unavailable rates, latency) and
the modal-decision agreement of Baseten with each of the other two backends. A missed bar is reported as missed, with the questions
that drove it; nothing here changes after a run.

## Spend cap

$15 on the hackathon key. Cost per call = prompt tokens x Baseten's listed price for Muse Spark Open, plus completion tokens x its
listed output price, looked up and recorded before the run; if Baseten lists none, Fireworks' price ($1.25 and $4.25 per million) is
the estimate. The run stops scheduling new questions at the cap. Spend is logged with the results.

## Outputs

`results/planner_eval_baseten.json`, in the same record format as `results/planner_eval.json`, plus the release check and the
agreement numbers. Run by `scripts/run_planner_eval.py` with the Baseten backend.
