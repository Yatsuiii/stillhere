# Glimmer face-edit check: labels and pass rule

Exploratory. Committed before any edited frame exists or is shown to Glimmer; the commit time is the record. Thresholds, prompt and
frames do not change after an edited frame has been shown.

## Ground-truth labels: the 30 probe-1e frames

- Frames: Tears of Steel (Blender Foundation 2012, CC BY 3.0), `tears_of_steel_720p.mov` (SHA-256 efa9062d...e588e8f),
  `ffmpeg -ss 20 -t 60 -vf fps=0.5 -q:v 2`: 30 frames at film 20, 22, ..., 78 s, exactly as probe 1e.
- Question: is any human face recognisable? A face counts when the eyes, nose or mouth of a person are visible enough to see it as a
  face. Backs of heads, hands, distant figures with no visible features, CG hologram figures and blurred fragments do not count.
- Labels (by Claude in the event session, looking at each frame; file `glimmer_probe_frame_labels.json`): face at 26, 28, 30, 32, 34,
  36, 38, 52, 54, 60 and 62 s (11 frames); no face on the other 19. Borderline: 58 s, an out-of-focus face behind the brain prop and
  mostly cut off, labelled no face.
- Disclosure: probe 1e's Glimmer answers on these intact frames were seen before labelling (10 yes, 19 no, 1 unparsed at 36 s). All 29
  parsed answers match the labels, and the unparsed frame is labelled face. Glimmer's intact rate here (10 of 11, 91%) is therefore
  already known and is not used as a test.

## Edited test frames

- Window: the bridge scene, release 25.00 to 40.25 s (one shot, two people talking on a canal bridge). SAM 3.1 (Meta Model API, concept
  "face") found faces on all 366 frames of the cut, two tracked faces.
- Sampling: 2 frames per second, release t = 25.0, 25.5, ..., 40.0 s (31 frames), the same instants in every version.
- Versions, as defined in `emofilm_bridge_rule.md`:
  1. intact: the release frames.
  2. faces removed: every face removed with SAM 3.1 masks (concept "face"), inpainted or filled with the local mean.
  3. area matched: per frame, the same masked area removed from a non-face region of the same frame, with the same fill.
- Prompt: probe 1e's ("Is any human face recognisable? Answer yes or no."), temperature 0. An unparsed answer gets one retry that asks
  for yes or no only (`pilot/probes/k1e`).

## Pass rule (all three must hold)

1. Intact frames with a face (SAM 3.1 found at least one face on that frame): Glimmer answers yes on at least 80%. Unparsed or error
   counts as not yes.
2. Faces removed: Glimmer answers no on at least 90% of the frames. Unparsed or error counts as yes.
3. Area matched: Glimmer answers yes on at least 80% of the frames. Unparsed or error counts as not yes.

Denominators are all 31 frames per version (for the intact version, those with a face; 31 expected). A pass means Glimmer can serve as
an automatic check that the face edit removed recognisable faces while the matched-area control kept them. A fail is reported as a
fail.
