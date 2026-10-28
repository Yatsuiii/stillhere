# Assets

- `librivox_alice_ch01_20s.wav`: 20 s (from 0:20 to 0:40) of LibriVox's *Alice's Adventures in Wonderland*, chapter 1
  (archive.org item `alice_in_wonderland_librivox`, file `wonderland_ch_01_64kb.mp3`), public domain, converted to mono 16 kHz PCM.
  It is the only clip the live laptop TRIBE run plays (PLAN §7 demo beat; LibriVox is the only audio played on stage). Its SHA-256 is
  on `stillhere/public_audio_allowlist.txt`, and the live run refuses any file that is not.

`librivox_alice_ch01_75s.wav`: the same LibriVox recording (chapter 1, read by Kristen McQuillin, public domain), 0:20 to 1:35, mono 16 kHz; `librivox_alice_ch01_75s.words.json` holds its word timings from the local faster-whisper base.en model (offline). `conditions/`: the four exploratory Pilot conditions built from it by `scripts/build_conditions.py` (intact, sentence scramble, word scramble, reversed; 10 ms equal-power raised-cosine crossfades at word boundaries, seed 20261028) and their manifest.

`diarization/two_readers_6turns.wav`: six ~10 s turns alternating between two credited LibriVox readers of the same project (chapter 1, Kristen McQuillin = A; chapter 2, Brad Bush = B; public domain), cut at local faster-whisper word gaps and joined with 10 ms crossfades by `scripts/build_diarization_clip.py`; `two_readers_6turns.truth.json` holds the exact turn table.
