# R4 rule: planner first-answer validity on Muse Spark 1.3 (exploratory)

Cloud ruling of Oct 31, 08:45 UTC (R4). The held-out requests were committed first (public f19badd, Oct 31 11:01:56 UTC), before
any prompt or schema change and by an agent that saw none. This rule is committed before any call with R4's candidate prompt; the
commit time is the record. The candidate is changed only on development requests; it is frozen by the commit that pins its SHA-256,
before the one final measurement. Every number is labelled exploratory.

## 1. What is measured, and why

Planner v2's first-answer validity on Muse Spark 1.3 was 83.3% on its 62 held-out requests, under v0's bar of 85%
(`results/planner_v2_eval.json`); Spark Open passed it on both hosts. Every invalid first answer in that run was
`decline_inconsistent`: an accept (decline_reason "none") missing a field the check requires (31 of 186 held-out runs, 34 of 318
seen). A development probe on 30 development accept requests whose first answers had failed (one repeat, v2 unchanged, raw answers
recorded; `r4/probe_first_answers.py` in the analysis tree) found the missing field was `analysis` in 23 of the 30 first answers, and
no other field: the requests seldom name an analysis, and `forecast_tribe_stock` is the only one offered.

## 2. The candidate (prompt only; the schema stays v2's)

`stillhere/public_text/planner_system_r4.txt`: v2's system prompt with one sentence replaced. v2's "Choose the analysis only from
the analysis enum; any other analysis declines with analysis_not_offered." becomes: "Choose the analysis only from the analysis enum.
Every accepted plan runs an analysis: when the request names none, set the analysis the enum offers; a request for any other analysis
declines with analysis_not_offered. An accepted StudySpec sets population, language, stimulus and analysis and lists at least two
conditions, one of them intact; otherwise decline with the matching decline_reason." `planner.plan` gains a `system` argument whose
default is v2's prompt, so the product and v2 are unchanged. The live planner (v2.1, its own prompt) is not changed by R4; carrying
a result into it is a later decision.

## 3. Development (development requests only)

The development requests are v0's 106 and v2's 62 (168, all seen before); never an R4 held-out request (the runner checks ids and
texts). At most four passes of the candidate, one repeat each, on Muse Spark 1.3, each recorded (`results/planner_r4_dev_r4.json`
for the last pass and `results/planner_r4_spend.json` for every pass), plus one pass of v2 on the same requests for reference. Any
edit between passes changes only `planner_system_r4.txt` and is listed in the freeze commit.

## 4. The final measurement (once)

The 62 R4 held-out requests (`eval/planner_questions_r4_heldout.json`), three repeats each, Muse Spark 1.3, the product's settings
(reasoning effort low, 60 s timeout), plan calls only, for v2's prompt and for the frozen candidate: 2 x 186 runs, by
`scripts/run_planner_r4_eval.py`. Scored by v0's code and bars (`stillhere/planner_eval.py`, unchanged): decision accuracy at least
90%, false-accept rate at most 5%, decline-reason accuracy at least 80%, spec-field accuracy at least 90%, first-answer validity at
least 85%, rule-made decline rate at most 5%, planner-unavailable rate at most 2%. A failed call is not retried. Unlike v2's
evaluation no after-accept tool call is made; it does not touch any of these bars except cost.

The candidate counts as an improvement only if its first-answer validity is at least 85% and every bar v2 passes on the same runs it
also passes. Both versions' bars are reported side by side, whatever they show.

## 5. Spend

Every call is costed at the planner rule's prices ($1.25 per million prompt tokens, $4.25 per million completion tokens) and counted
across all R4 runs, the probe included; no run starts once the total reaches $13.00 (under 10% of the Meta credit left on Oct 31,
$136.43). Estimate posted before any call: about $7.10. The app's daily caps do not change.

## 6. Lines (written before the result)

- "Planner R4 (rule committed {commit time}), 62 new held-out requests, three repeats, Muse Spark 1.3: first-answer validity {r4}
  with the new prompt against {v2} with v2's (bar 85%)." Then each other bar for both versions, then: "{An improvement by the rule set
  before the run | Not an improvement by the rule set before the run: {the bars it missed}}."

Not a diagnosis or medical advice.
