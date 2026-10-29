# Order-map go/no-go (exploratory; forecast only, no brain data)

Cloud plan v2 (Oct 29, 16:20 IST), item 1. It decides one thing: whether the page and the video may show an "order map", the picture of
where TRIBE's forecast changes when a story's sentences are put in a different order. It is a check on the model's forecasts, made
before the order map is drawn anywhere. No listener's scan is read, and nothing frozen changes. This file and its code
(`scripts/order_map.py`, `scripts/order_map_run.py`, `tests/test_order_map.py`) are committed before any forecast on these clips.

## Clips

Three LibriVox recordings of Lewis Carroll's *Alice's Adventures in Wonderland* (public domain; archive.org item
`alice_in_wonderland_librivox`): chapter 1 (read by Kristen McQuillin), chapter 2 (read by Brad Bush) and chapter 3 (reader listed on
the LibriVox catalog page). For each, the first bytes of the archive.org file were fetched (a byte range; the archive was slow), which
holds well over 100 s of audio: `wonderland_ch_01_64kb.mp3` 949,804 bytes (SHA-256 84c3fa62...), `wonderland_ch_02_64kb.mp3`
818,876 bytes (a798fe14...), `wonderland_ch_03_64kb.mp3` 966,170 bytes (508aea31...). Each clip is 40 s from 60 s, mono 16 kHz PCM
(ffmpeg, bit-exact flags). Word timings come from the local faster-whisper base.en model (CPU, offline), as for the Pilot's existing
LibriVox assets. The full table is in `results/order_map_prepare.json`.

| Clip | Words | Sentences | Pieces (s) | Order |
|---|---|---|---|---|
| alice_ch01 | 147 | 4 | two blocks at the midpoint: 0 to 27.36, 27.36 to 40 | 1, 0 |
| alice_ch02 | 117 | 10 | blocks of 15 s or more: 0 to 18.8, 18.8 to 40 | 1, 0 |
| alice_ch03 | 66 | 4 | two blocks at the midpoint: 0 to 12.83, 12.83 to 40 | 1, 0 |

Condition files (SHA-256, all on `stillhere/public_audio_allowlist.txt`; intact is the clip itself):

| Clip | intact | sentence_shuffle | reversed |
|---|---|---|---|
| alice_ch01 | 35fd795aa583... | 9daad12ef37b... | 5a83b8636054... |
| alice_ch02 | 5d0a79f4643f... | cff6e55fbe3f... | 40a4ad9312a1... |
| alice_ch03 | 4c41e4e0e7c1... | e0326b61bd7f... | 050d8a701465... |

Two choices were changed before this commit, and before any forecast on any of these clips: the start moved from 30 s to 60 s,
because chapter 3 at 30 s was still its spoken title and could not be split; and the two-block midpoint fallback below was added after
the sentence lengths of chapters 1 and 3 at 60 s showed no split into two blocks of 15 s or more.

## Conditions

- intact: the clip.
- sentence_shuffle: the pieces in a shuffled order. Pieces are runs of whole sentences (a sentence ends at a word ending in . ! or ?):
  1. if every sentence lasts at least 6 s, the pieces are the sentences;
  2. otherwise, blocks of whole sentences, the most blocks that each last at least 15 s (ties: the smallest longest block);
  3. if no split gives two such blocks, two blocks split at the sentence end nearest the clip's midpoint, each at least 10 s (so at
     least 5 s of each survives the 5 s drop below);
  4. if none of these exists, the clip cannot be shuffled and counts as a clip where the language-area change is not larger.
  The order is `stillhere.conditions`' non-identity permutation with the Pilot's default seed (20261028). Cuts fall between words;
  joins between pieces that were not neighbours get the Pilot's 10 ms equal-power crossfade, and the length is unchanged.
- reversed: the clip played backwards. It is forecast and shown, but it is not part of the go/no-go.

## The forecast

Stock TRIBE v2 (`facebook/tribev2` best.ckpt, checked against the stock SHA-256 pin) on this laptop's RTX 4050, text and audio lanes
on, video lane off (there is no video). The audio lane is w2v-BERT 2.0. The text lane is Llama 3.2 3B, and it must be on. It reads
each condition's word table through TRIBE's own word, sentence and context transforms (the table TRIBE's speech-to-text step would
otherwise write; its sentence step uses TRIBE's default spaCy model, en_core_web_lg 3.8.0): intact uses the clip's word timings; sentence_shuffle moves every word with its piece; reversed has no words, so 0
words reach the text lane (a model input, not a brain result). Llama 3.2 runs in fp16 with accelerate's GPU and CPU placement, as in
the project's Kaggle forecasts; if that cannot run on the 6 GB GPU, 8-bit, then 4-bit. Every clip and condition uses the same
precision, and every result records the precision, the extractors that ran, the words that reached the text lane, timings and peak
VRAM. To fit the 6 GB GPU, the text lane's features for every condition are computed first and kept in TRIBE's own feature cache,
then Llama 3.2 is released before the audio lane and the brain model run (memory only; the features are the same). Each 40 s
clip is one TRIBE window (100 TRs), predicted with TRIBE's built-in predict: one value per second.

## Score

- Regions: the frozen proof's parcel sets, `stillhere.heldout.parcel_sets` on the Schaefer-400 17-network annotation. LANGUAGE = left
  TempPar, DefaultB_Temp, DefaultB_PFCv and DefaultB_PFCl; HEARING = SomMotB_Aud in both hemispheres. A parcel's forecast is the mean
  over its vertices.
- Un-shuffle: each second of the shuffled forecast that lies wholly inside one piece is put back at its original time (piece start
  plus the same offset), and compared with the intact forecast read there (linear between its 1 s samples). Dropped: the first 5 s of
  every piece whose predecessor in the shuffled audio is not its predecessor in the intact (the clip start counts as a predecessor),
  that is, the 5 s after every cut.
- Change, per parcel: the RMS of (un-shuffled minus intact) over the kept seconds, divided by the SD of the intact over the same
  seconds. It is scale-free: it does not depend on a parcel's units or response size. A region's change is the mean over its parcels.
- Per clip: the language-area change is larger when LANGUAGE's change > HEARING's change.
- GO: the language-area change is larger on 3 of 3 clips. The page and the video may then show an order map.
- NO-GO: anything else. The page and the video use time-course panels plus the reversed variant, and no order map.

The outcome is committed either way, in `results/order_map_gonogo.json`: per clip the kept seconds, the change per region, the call,
the extractors and precision, timings and VRAM, and the per-second LANGUAGE and HEARING forecast time courses (region means, raw TRIBE
units) for the page and the video. It is a statement about TRIBE's forecasts only, never about any listener's brain.
