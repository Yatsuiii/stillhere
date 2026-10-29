# Captions for the page and the video (written before the results; Sunday fills the placeholders)

Placeholders: {k} viewers or frames that met the condition, {N} the total, {p} the p value as computed, {lo} and {hi} interval ends,
{veq} and {veq_lo} the volunteer-equivalent and its lower bound. Each slot has one caption per possible ending; Sunday picks the one
that happened and fills its numbers. No caption changes wording after a result is known.

## Wording rules

- SAM edits the frames; Spark Open and Glimmer check the edits ("edit check", never "face check"); TRIBE forecasts. Nothing "detects
  faces". Nothing is "validated": no viewer saw an edited cut.
- What the edit does, said plainly: "SAM masked the faces and StillHere filled each one with the colour around it." (If the final cuts
  blur instead of fill, this line changes to "SAM masked the faces and StillHere blurred them" before any edited forecast exists.)
- The proof line: "a forecast worth about {veq} real volunteers, and at least {veq_lo}."
- Cheap baselines are never a headline. Each evidence panel carries one row, "What it had to beat: {baselines}".
- Hearing: "agreed with the classic sound test".
- Every result is labelled exploratory unless it is a pre-registered bar; a failed bar is shown as failed.
- Film frames on screen always carry the credit line below.

## Credit line (on screen whenever film frames show)

"Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study."

## Hero

1. Question (typed): "If I remove the faces from this film scene, what changes in the brain?"
2. Plan: "Muse Spark turned the question into a study: the intact film, a version with the faces removed, and an equal-area control with
   the same amount of picture removed somewhere else."
3. Edit: "SAM 3.1 masked every face in three minutes of film; StillHere removed them, and removed the same area elsewhere for the
   control."
4. Edit check:
   - pass: "Edit check: Spark Open and Glimmer saw a face in {k_int} of {N} intact frames, none in {k_rem} of {N} edited ones, and
     still saw faces in {k_am} of {N} control frames."
   - fail: "Edit check failed ({detail}), so the face-map test did not run."
5. Maps, one caption per ending (the test is scored once, at the logged onsets):
   - PASS (also outside the visual network): "SAM masked the faces; TRIBE forecast the change; that forecast lined up with where {N} real
     viewers' brains tracked faces in the intact film ({k} of {N} viewers, p = {p}), and more than for the equal-area control."
   - PASS carried by visual-network parcels: the PASS line, then "Most of the match comes from visual-network areas."
   - FAIL, not face-specific: "TRIBE's forecast lined up with where {N} real viewers' brains tracked faces ({k} of {N}), but no better
     than its forecast for the equal-area control: not face-specific."
   - FAIL: "TRIBE's forecast of the face edit did not line up with where {N} real viewers' brains tracked faces ({k} of {N} viewers,
     p = {p}). We show it because we said we would."
   - UNINFORMATIVE: "The {N} viewers' own face maps did not agree with each other, so this film cannot test the forecast."
   - NOT RUN: "The face-map test did not run: {reason}. It was set to run once, by rules written before any brain data was read."

## Evidence panels

- Held-out speech (LPP section 8): "On a story chapter it never saw, TRIBE's forecast was worth about {veq} real volunteers, and at
  least {veq_lo}. It agreed with the classic sound test." What it had to beat: word rate and the sound envelope. Label: "recomputed with
  code built during the event; first measured Oct 22-23."
- Story wall (Narratives and LPP stories), one per ending:
  - scored: "Across {N} stories from public scans, TRIBE's forecast {wall_result}." What it had to beat: {baselines}.
  - next step (too few stories passed the pre-registered checks): "Too few stories passed the pre-registered data checks to score the
    wall; the rule says to report it as a next step."
  - not run: "Pre-registered, not run (GPU time ran out)."
- notthefall (intact story vs the same words scrambled), one per ending:
  - pass: "Listeners' brains followed the intact story more than the scrambled one where TRIBE said they would ({k} of {N}, p = {p})."
  - fail: "The intact-versus-scrambled test did not pass ({k} of {N}, p = {p})."
  - not run: "Pre-registered, not run ({reason})."
  - What it had to beat: word rate and the sound envelope of each version.
- Planner: "Ask for something it cannot support and it says so, with the reason and the nearest thing it can do." Numbers (v2, on
  62 requests written after v2 was frozen, never seen before): "False accepts: {k} of {N} ({p_upper} upper bound when {k} is 0).
  Declines from the model: {k_model}; from the planner's fixed rules: {k_rules}."
- Sedation case study: "Frozen group statistics only. Its two headline bars did not pass, and the write-up says so."

## Disclosures (page footer and the video's last slide)

- TRIBE v2 (Meta FAIR, CC BY-NC 4.0, not included in this repository) was trained on four datasets (Courtois NeuroMod, BoldMoments,
  Lebel2023, Wen2017; 25 participants) and tested by its authors on four others (NNDb, LPP, Narratives, HCP): d'Ascoli et al. 2026,
  arXiv 2605.04326, Table 1. Emo-FilM and the Blender films are not among them, so the 28 film viewers and Tears of Steel are new to
  the model. LPP and Narratives are TRIBE's own held-out test sets, not its training data.
- SAM 3.1 through Meta's Model API, decoded with meta-sam-parser under Meta's SAM License; we acknowledge the use of SAM in this work.
- Data: Emo-FilM (Morgenroth et al. 2025; OpenNeuro ds004892 and ds004872, CC0), Le Petit Prince (ds003643), Narratives (ds002345),
  LibriVox (public domain), Tears of Steel and Sintel (Blender Foundation, CC BY 3.0).
- Research tool for brain-imaging labs. Healthy volunteers and public data. Not a diagnosis or medical advice.
