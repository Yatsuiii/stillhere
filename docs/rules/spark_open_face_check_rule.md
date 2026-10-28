# Spark Open face-edit check (a second check beside Glimmer)

Exploratory. Project-thread ruling of Oct 28, 21:45 UTC (item 3). Committed before any of the 93 edited-test frames is shown to Muse
Spark Open; the commit time is the record. This rule copies `glimmer_face_edit_rule.md`: the same frames, question wording, pass bars
and one retry for an unparsed answer. Thresholds, prompt and frames do not change after a frame has been shown. It is reported next to
Glimmer; neither check replaces the other, and Glimmer still runs on Saturday. Wording: SAM edits the frames; Spark Open and Glimmer
check the edits. Neither "detects faces".

## Model and what is sent

- Meta Muse Spark Open on Baseten (`meta-models/muse-spark-open-1026`, the hackathon key from an environment variable only), through the
  product's typed egress: each frame as an `egress.PublicFrame` (its SHA-256 is on `stillhere/public_frame_allowlist.txt`) and the
  allowlisted prompt texts in `stillhere/public_text/` (`frame_check_system`, `frame_check_question`, `frame_check_retry`). Only Tears of
  Steel bridge frames are sent; nothing from ds003171.
- Image input was confirmed before this rule on one frame that is not among the 93 (release frame 606, 25.25 s), with a neutral
  question about the sky's colour; no face question was asked of any frame before this rule.
- Differences from Glimmer's run, forced by the product client: the answer arrives as a JSON string field (the egress only sends
  structured requests) and is parsed with Glimmer's own parser (`yes`, `no`, else unparsed); the client sets reasoning effort low and no
  temperature or token limit (Glimmer: temperature 0, 512 tokens, 1024 on the retry).

## Frames

The 93 frames of `glimmer_edit_frames` (MANIFEST e31e4b19...): the bridge scene, release t = 25.0, 25.5, ..., 40.0 s (31 instants), in
three versions (intact; faces removed; area matched), built by `scripts/make_glimmer_edit_frames.py` with `stillhere/face_edit.py`.

## Question and retry

"Is any human face recognisable? Answer yes or no." An unparsed answer gets one retry: "Is any human face recognisable? Reply with one
word only: yes or no." A failed call is not retried and counts as an error.

## Pass rule (all three must hold), as Glimmer's

1. Intact frames with a face (SAM 3.1 found at least one face on that frame; all 31): yes on at least 80%. Unparsed or error counts as
   not yes.
2. Faces removed: no on at least 90% of the 31 frames. Unparsed or error counts as yes.
3. Area matched: yes on at least 80% of the 31 frames. Unparsed or error counts as not yes.

A fail is reported as a fail.

## Spend cap

$10 on the hackathon key, at Baseten's listed Muse Spark Open price ($1.25 per million prompt tokens, $4.25 per million completion
tokens; images are billed as prompt tokens). The run stops at the cap. 93 frames plus retries are expected to cost well under $1.

## Output

`results/spark_open_face_check.json`: per frame the version, instant, answer, first answer if retried, tokens and seconds; the three
rates against the bars; spend. Run by `scripts/run_spark_open_face_check.py`.
