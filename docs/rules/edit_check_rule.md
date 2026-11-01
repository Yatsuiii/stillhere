# Edit check of the film versions (exploratory)

Project-thread ruling of Oct 29, 06:00 UTC (item 3c). Committed before any frame of this sample is shown to a model; the commit time is
the record. It extends the committed Glimmer and Spark Open rules (`glimmer_face_edit_rule.md`, `spark_open_face_check_rule.md`) from
the bridge scene to the scored film window, for the face-map test (`face_map_rule.md`). SAM edits the frames; Spark Open and Glimmer
check the edits. Neither "detects faces", and no viewer saw an edited cut, so nothing here is "validated".

## Frames (chosen from SAM's output alone, before any frame is shown)

`scripts/edit_check_sample.py` on the SAM 3.1 run over Tears of Steel release 0-190 s (`faces_api.jsonl`; cut frame c = release frame c);
its output is committed with this rule as `eval/edit_check_sample.json`:

- Bridge frames: the 31 bridge instants (release 25.0, 25.5, ..., 40.0 s; cut frames 600 + 12k), decoded from the final cuts like the
  new frames, in their three versions. (Amended before any frame of this sample was built or shown, on the coordinator's instruction
  after the one allowed re-render: every edit-check frame comes from the final cuts; the earlier bridge-cut frame set is not used.)
- New face frames: 31 frames from film time 10-180 s outside the bridge window, each with at least one SAM face whose box covers at
  least 500 pixels of the 640x360 cut (about 2,000 pixels at 1280x534, fixed before any frame was seen so that a speck does not count as
  a recognisable face); 1,611 frames qualify; one is drawn from each of 31 contiguous equal parts with seed 20261029. These are decoded
  from the FINAL film-version cuts, film stimuli v2 (`pilot/film/film_stim_v2`: seg_intact.mp4, faces_removed_v2.mp4,
  area_matched_v2.mp4; MANIFEST.sha256 d4e27d55e2c462f8edd9d2373407692d03534893a3fac284a6819897f52a6f44, checked before any frame is
  built), in all three versions, and written as JPEG quality 95 at 1280x534.
- In v2 the fill is the local mean of the surrounding colour, the same for the controls; control patches sit inside the part of the
  frame TRIBE's video reader keeps (release x 404-873, y 32-501). 51 of 2,717 in-view faces (the largest close-ups) have no control,
  so the control covers 0.889 of in-view face pixels (reported per the stimulus rule). Control frames are asked as they are; the bar
  does not change.
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

Dated note, Nov 1 2026 (cloud ruling of Nov 1, 05:05 UTC, item 1; the text above is unchanged): Glimmer ran at 05:15 UTC Oct 31, outside
its chain slot; it reads only SAM frames and does not depend on the forecasts, so this is not "cannot run in that slot". Its record is
`results/edit_check_glimmer.json`: all four bars pass (intact 58 of 62 yes, faces removed 62 of 62 no, area matched 50 of 62 yes, no-mask
0 of 31 yes) on the frames MANIFEST fe3d936d....
