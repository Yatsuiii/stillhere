# Edit check of the film versions (exploratory)

Project-thread ruling of Oct 29, 06:00 UTC (item 3c). Committed before any frame of this sample is shown to a model; the commit time is
the record. It extends the committed Glimmer and Spark Open rules (`glimmer_face_edit_rule.md`, `spark_open_face_check_rule.md`) from
the bridge scene to the scored film window, for the face-map test (`face_map_rule.md`). SAM edits the frames; Spark Open and Glimmer
check the edits. Neither "detects faces", and no viewer saw an edited cut, so nothing here is "validated".

## Frames (chosen from SAM's output alone, before any frame is shown)

`scripts/edit_check_sample.py` on the SAM 3.1 run over Tears of Steel release 0-190 s (`faces_api.jsonl`; cut frame c = release frame c);
its output is committed with this rule as `eval/edit_check_sample.json`:

- Bridge frames: the 31 committed frames of the bridge scene (`glimmer_edit_frames`, MANIFEST e31e4b19..., release 25.0-40.0 s at 2 per
  second), unchanged, in their three versions.
- New face frames: 31 frames from film time 10-180 s outside the bridge window, each with at least one SAM face whose box covers at
  least 500 pixels of the 640x360 cut (about 2,000 pixels at 1280x534, fixed before any frame was seen so that a speck does not count as
  a recognisable face); 1,611 frames qualify; one is drawn from each of 31 contiguous equal parts with seed 20261029. These are decoded
  from the FINAL film-version cuts (`pilot/film/film_edits/{intact,faces_removed,area_matched}_0p00_190p00.mp4`, after the one re-render
  the stimulus rule allows and before any edited forecast), in all three versions, and written as JPEG quality 95 at 1280x534 like the
  committed frames.
- No-mask frames: 31 frames from film time 10-180 s where SAM found no face (1,955 qualify), drawn the same way with seed 20261030,
  intact version only, to count faces SAM missed.

## Question, retry, parser

As committed: "Is any human face recognisable? Answer yes or no." An unparsed answer gets one retry ("Is any human face recognisable?
Reply with one word only: yes or no."); Glimmer's parser decides yes, no or unparsed.

## Bars (all four must hold); an unparsed answer or an error counts against the bar every time

1. Intact frames with a face (31 bridge + 31 new = 62): yes on at least 80%.
2. Faces removed (62): no on at least 90%.
3. Area matched (62): yes on at least 80%.
4. No-mask frames, intact (31): at most 3 yes (faces SAM missed).

Bridge and new frames are also reported separately, with no separate bar.

## Who decides, and what a fail means

Spark Open (Baseten, `meta-models/muse-spark-open-1026`, spend cap $10) runs it now and is reported. Glimmer (the committed Kaggle
setup) repeats it on Saturday, in its slot of the GPU order, with the same frames and decides. If Glimmer fails any bar, or cannot
run in that slot, the face-map test is NOT RUN. Nothing here changes after a frame has been shown.

## Sent

Only Tears of Steel frames (Blender Foundation 2012, CC BY 3.0), each on the public-frame SHA-256 allowlist; nothing from ds003171.
