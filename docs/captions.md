# Captions, voice-over and slides (written before the results; Sunday fills the placeholders)

Storyline of the project-thread ruling of Oct 29, 06:30 UTC. Placeholders: {k} viewers, listeners or frames that met the condition, {N}
the total, {p} the p value as computed, {reason} why a test did not run. Each results slot has one caption per possible ending; Sunday
picks the ending that happened and fills its numbers. No wording changes after a result is known.

## Wording rules

- What the edit does: "SAM filled the faces it masked with the surrounding colour"; the control patches get exactly the same fill.
- SAM edits the frames; Spark Open and Glimmer check the edits ("edit check"); TRIBE forecasts. Nothing here is "validated": no viewer
  saw an edited cut.
- StillHere is a research tool for labs. The words on the banned list (tests/test_captions_wording.py) never appear in captions,
  voice-over or the write-up.
- Every results slide carries: "Not a diagnosis or medical advice."
- Every slide that shows film frames carries the credit line below.
- Hearing: "agreed with the classic sound test". Cheap baselines are never a headline; a results panel may carry one row, "What it had to
  beat: ...".
- The sedation case study is never in the video.

## Credit line (on screen whenever film frames show)

"Tears of Steel (CC) Blender Foundation | mango.blender.org, CC BY 3.0. Edited for this study."

## a. Opening (spoken word for word: the frozen D23 opener)

Voice-over: "Labs study people who can't respond after brain injury by playing them stories in the scanner. But every new story needs its
own 12-15 healthy volunteers scanned first, in that language. For a new story, StillHere gives labs a first reading with zero volunteer
scans, measured against each listener's own brain. It has been tested only in healthy volunteers."

On-screen captions:
- "1 in 4 brain-injured patients who could not follow commands at the bedside showed command-following brain activity on fMRI or EEG
  (Bodien et al., NEJM 2024)."
- "At university scanner rates ($450-700 an hour), 12-15 volunteers is roughly $5,000-15,000 of scanner time per new story, before
  recruitment."

## b. Why it matters (one line)

"Passive tests play a story or a film, and every new one needs healthy volunteers scanned first (Naci et al., PNAS 2014, used a film)."

## c. Demo beat A: a new story test

1. Ask (typed): "Build a test from this new story, with a scrambled control."
2. Plan: "Muse Spark 1.3 plans it: the intact story and a sentence-scrambled control, the only kinds it has evidence for."
3. Check: "Muse Voice Transcribe checks every version before anyone is scanned: the intact story kept 99% of its words; the sentence
   scramble kept 98% of its words in a new order."
   Broken stimulus caught: "The word scramble kept only 16% of its words, so it was caught as broken before scanning and is not used."
4. Forecast and proof: "TRIBE v2 forecasts the healthy-adult response. On a held-out audiobook, the forecast was worth about 5 real
   volunteers, and at least 2. It agreed with the classic sound test." What it had to beat: word rate and the sound envelope.
   Small print: "Le Petit Prince, section 8, held out: recomputed with code built during the event; first measured Oct 22-23."
5. Real listeners, intact versus scrambled (notthefall), used only if it passes tonight:
   - pass: "With zero same-clip controls, stock TRIBE v2's language readout was stronger when the same healthy listeners heard a
     public story intact than when they heard its sentences shuffled (Narratives `notthefall`, 22 listeners, 9.1 minutes = 365 TRs per
     version; the order of their two runs is not public)." Voice-over: "On a different story, TRIBE's language readout was stronger
     intact than shuffled." Changed after the D20 result to PREREG D20's pre-written line, which predates the result.
   - any other ending: the panel is left out.
6. Planner: "Ask for something it cannot support and it says so, with the reason and the nearest thing it can do. On requests it had
   never seen, the default planner (Muse Spark 1.3) accepted 2.3% it should have declined (upper bound 6.5%). Private mode, Muse Spark
   Open, which can run on your own servers: 7.6% (Fireworks), 6.8% (Baseten)."

## d. Demo beat B: the same loop on video

1. Ask (typed): "If the faces in this film scene are filled in, what changes in the brain?"
2. Edit: "SAM filled the faces it masked with the surrounding colour. The control fills the same area somewhere else, with exactly the
   same fill."
3. Edit check, one per ending:
   - pass: "Edit check: Glimmer saw a face in {k_int} of {N} intact frames, in {k_rem} of {N} filled ones, and in {k_am} of {N}
     control frames." (Glimmer's counts only; Spark Open's are given as Spark Open's.)
   - fail: "The edit check failed ({detail}), so the film test did not run."
4. Forecast against 16 real viewers (28 kept, 11 dropped by the volume check, 1 for motion). Length and caption follow the outcome (the test is scored once, at the logged onsets). Lines fixed
   by the cloud ruling of Nov 1, 05:05 UTC (item 5), before the result; every result line starts "Exploratory." (face_map_rule.md:5).
   {k}, {kstar}, {p} and {spin_p} are B1's (`results/face_map.json`: k_above_0, kstar, wilcoxon_p, spin_p); {k2}, {p2}, {spin_p2}
   are B2's.
   - Before the result (pending): "If the edit check fails or gate 0 stops it, it is NOT RUN; if the viewers' maps are not reliable
     (gate 1), it is UNINFORMATIVE."
   - PASS (also outside the visual network; long panel): "Exploratory. SAM filled the faces it masked; TRIBE forecast the change; that
     forecast lined up with where {N} real viewers' brains tracked faces in the intact film ({k} of {N} viewers, p = {p}), and more than
     for the control."
   - PASS with the lead check failing (long panel): the PASS line, then the rule's words: "The match was carried by visual-network
     parcels."
   - FAIL, not face-specific (short panel): "Exploratory. TRIBE's forecast lined up with where {N} real viewers' brains tracked faces
     ({k} of {N}), but not reliably more than its forecast for the control (B2: {k2} of {N}, p = {p2}, spin p = {spin_p2})."
   - Beside B2 (PASS and FAIL, not face-specific; small print, descriptive): "Descriptive: both forecast maps, the control's especially,
     are concentrated in the seconds around the 51 uncontrolled close-ups (61-67 s)."
   - FAIL (short panel), naming the missed criterion (the Wilcoxon test, the viewer count, the spin test, or all three criteria) with all
     three numbers: "Exploratory. TRIBE's forecast of the filled faces did not line up with where {N} real viewers' brains tracked faces:
     it missed {criterion} ({k} of {N} viewers above 0, {kstar} needed; Wilcoxon p = {p}; spin p = {spin_p}). We show it because we
     said we would." Spoken: "Exploratory. It missed the spin test. We show it because we said we would." (the criterion it missed).
     Beside it when B2 passed (small print; written after the result, cloud ruling of Nov 1, 06:05 UTC, item 3): "B2, more than the
     control, passed, but it is not used once B1 fails; part of its margin came from the control's map running opposite to the
     viewers' (rho with the viewers' average map: R 0.20, C -0.11; on the descriptive controlled-grid maps, R 0.02, C -0.31;
     descriptive, never decides)." and the close-up sentence above.
   - UNINFORMATIVE (short panel): "Exploratory. The {N} viewers' own face maps did not agree with each other, so this film cannot test
     the forecast." If a constant or non-finite parcel row stopped the score after the BOLD was opened: "Exploratory. The face test is
     UNINFORMATIVE: {reason}." (the viewer and the parcel named).
   - The intact film, with every outcome (cloud ruling of Nov 1, 06:05 UTC, item 2; written after the result from emofilm_rule.md's
     conventions, numbers copied from `results/emofilm_film_score.json`): "Exploratory. On a film clip in neither TRIBE v2's training
     list nor its evaluation list (Tears of Steel, film seconds 10-185; Emo-FilM, 16 viewers), TRIBE v2's forecast followed the viewers'
     hearing areas (median z 3.23) and visual areas (2.77) more closely than the sound envelope, brightness or motion did. In language
     areas (0.80) it did no better than word rate. In every area it was worth less than one real viewer." Beside it: "What it had to
     beat: word rate, the sound envelope, brightness and motion (beaten in 3 of 4 primary comparisons; every pair in
     results/emofilm_film_score.json)." and "Not a diagnosis or medical advice."
   - NOT RUN (18 s; the edit and its check have already been shown, and the story beat leads the video): the intact film line above;
     then "The face test did not run: {reason}. It was set to
     run once, by rules written before any brain data was read." Reasons as logged, e.g. "edit not confirmed by Glimmer", "gate 0:
     VIF > 5", "GPU time ran out".
   - The control, wherever its coverage is given: "The same-size patch covers 89% of the face area TRIBE's video model sees; the 51
     largest close-ups, and faces outside that view, have none."
   Small print on every film results slide (the page footer's words): "Emo-FilM is in neither TRIBE v2's training list nor its evaluation
   list: {N} real viewers of the intact film."

## e. Close

"A first healthy reference for a new test, with zero scans, and evidence for how far to trust it."

## Disclosures (close slide and page footer)

- TRIBE v2 (Meta FAIR, CC BY-NC 4.0, not included in this repository) was trained on Courtois NeuroMod, BoldMoments, Lebel2023 and
  Wen2017 (25 participants) and evaluated by its authors on NNDb, LPP, Narratives and HCP (d'Ascoli et al. 2026, arXiv 2605.04326,
  Table 1). LPP and Narratives are therefore among TRIBE v2's own evaluation sets. Emo-FilM, the Blender films and the sedation data
  (ds003171) are in neither list.
- SAM 3.1 through Meta's Model API, decoded with meta-sam-parser under Meta's SAM License; we acknowledge the use of SAM in this work.
- Data: Emo-FilM (Morgenroth et al. 2025; OpenNeuro ds004892 and ds004872, CC0), Le Petit Prince (ds003643), Narratives (ds002345),
  LibriVox (public domain), Tears of Steel (Blender Foundation, CC BY 3.0).
- Research tool for labs. Tested only in healthy volunteers and public data. Not a diagnosis or medical advice.

## Sources for the write-up

- Bodien et al., NEJM 2024 (the 1 in 4 figure; cite by authors and journal, without the title).
- Naci et al., PNAS 2014 (a passive test with a film).
- Scanner rates: University of Michigan fMRI Lab, $620 an hour from Apr 16, 2025 (fmri.research.umich.edu/users/billing.php);
  University of Kentucky MRISC, budget up to $625 an hour for proposals after June 30, 2026 (research.uky.edu/mrisc/rates-policies);
  Yale MRRC, $696 (medicine.yale.edu/mrrc/users/charges; the page lists $696 per hour with a 1-hour minimum, so check the "for 1.5 h"
  reading before quoting it).
