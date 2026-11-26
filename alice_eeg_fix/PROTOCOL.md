# Alice EEG: what the one-segment timing shift does to the word-level effects (frozen 2026-11-26, before any EEG is epoched)

Data: Brennan & Hale (2019) Alice in Wonderland EEG as re-released by NEMAR (nm000180, commit e37db55b334496f1602489963f94f354012ba580), annexed files hash-checked against their git-annex keys (`reproduction-receipts` branch `claude/project-thread-j92rdf`, `alice-eeg-onsets/fetch.py`).

Already known before this file (onset check of 23 Nov, same branch, `check_onsets.py`): in 8 recordings (sub-002, 026, 027, 030, 032, 034, 035, 036) every word onset in `events.tsv` sits one audio segment late, and 4 of them are marked analyzed (026, 034, 035, 036). Nothing in this file has looked at any epoch, ERP or regression.

## Onsets

- **Released**: the word onsets of each subject's `proc/SXX.mat` `trl` (the trials retained in the 2019 preprocessing), as rebuilt in `events.tsv`.
- **Corrected**: the same words, placed at the measured start of their own segment's audio in that recording (`check_onsets.py`'s fine placement, rerun here on the same pinned files) plus the word's onset inside the segment file (`AliceChapterOne-EEG.csv`). Only the 8 shifted subjects change; for the others, released onsets are used as they are (their residuals were within about 5 ms).
- A word whose corrected epoch would run past the end of the recording is dropped from both versions for that subject.

## EEG pipeline (ours, fixed now; not the 2019 FieldTrip pipeline)

MNE 1.10.2. The 61 EEG channels, average reference, FIR band-pass 0.1-30 Hz, epochs -0.3 to 1.0 s, baseline -0.2 to 0 s, no further rejection (the retained-trial list is his). Sub-005 and sub-038 (audio channel all zeros) and the 3 subjects with undetermined audio placement are scored with released onsets only and are not part of any comparison.

## Measures

- **M1 (word-onset ERP, model-free).** Per subject, the mean ERP over all retained words, all channels, -0.1 to 0.6 s. Similarity = Pearson r between that map and the mean map of the consistent subjects (those not among the 8; leave-self-out). Reported for each of the 8, released vs corrected, next to the consistent subjects' distribution.
- **M2 (surprisal effect).** Per subject, at every channel x time point, OLS of amplitude on z-scored [NGRAM, LogFreq, LogFreq_Prev, LogFreq_Next, SndPower, Length, Position, Sentence]. Similarity of the subject's NGRAM beta map (0 to 1 s) to the leave-self-out mean map of the consistent subjects, released vs corrected, for each of the 8.
- **M3 (group effect, the analyzed sample).** The 33 subjects marked analyzed. The window is chosen without the 4 analyzed shifted subjects: the channel x 50-ms-bin with the largest |mean NGRAM beta| over the analyzed consistent subjects, 200-700 ms. Group one-sample t over all 33 at that window, released vs corrected. The same for CFG in a model that adds CFG to M2's predictors (CFG beyond NGRAM, the comparison the 2019 paper cared about), window chosen the same way.

## Pre-written lines

- If corrected onsets bring the 8 subjects' M1 and M2 maps into the consistent subjects' range and released ones don't: "In the 8 recordings, the released onsets lock the epochs to the wrong words: their word ERP and surprisal maps look like noise and recover when the words are placed on the recorded audio."
- M3 is reported as the t change, whatever its size, with the note that 4 of 33 subjects can move a group t only a little.
- If corrected onsets don't help: reported as such, and the shift claim is re-examined before anything is said to anyone.

Never "the paper is wrong". The 2019 results are not re-tested here beyond M3, which uses our pipeline, not his.
