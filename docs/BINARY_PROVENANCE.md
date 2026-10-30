# Binary files and what built them

The hygiene scan reads text only, so every binary file in this repository is listed here with its inputs. None contains anything from
the privately shared sedation stimulus (no audio, frames, words, durations, word counts or predictions), no per-volunteer value and no
subject ID. No binary here shows the sedation case study (OpenNeuro ds003171): the first page's screenshot and the first video cut, which
showed its frozen group statistics, were retired on Oct 29 and are in history only.

| File | Size | Built from |
|---|---|---|
| `assets/librivox_alice_ch01_20s.wav` | 640 KB | LibriVox *Alice's Adventures in Wonderland*, chapter 1, read by Kristen McQuillin (public domain; archive.org item `alice_in_wonderland_librivox`, file `wonderland_ch_01_64kb.mp3`), 0:20 to 0:40, mono 16 kHz |
| `assets/librivox_alice_ch01_75s.wav` | 2.4 MB | the same recording, 0:20 to 1:35, mono 16 kHz |
| `assets/conditions/alice_ch01_75s_intact.wav` | 2.4 MB | `librivox_alice_ch01_75s.wav` by `scripts/build_conditions.py` (word timings from the local faster-whisper base.en model) |
| `assets/conditions/alice_ch01_75s_sentence_scramble.wav` | 2.4 MB | the same input and script, sentences reordered (seed 20261028), 10 ms crossfades |
| `assets/conditions/alice_ch01_75s_word_scramble.wav` | 2.4 MB | the same input and script, words reordered (seed 20261028), 10 ms crossfades |
| `assets/conditions/alice_ch01_75s_reversed.wav` | 2.4 MB | the same input, time-reversed |
| `assets/diarization/two_readers_6turns.wav` | 1.9 MB | LibriVox *Alice* chapter 1 (Kristen McQuillin) and chapter 2 (Brad Bush), same archive.org item, public domain; six turns joined by `scripts/build_diarization_clip.py` |
| `demo/media/alice_75s_{intact,sentence_scramble,word_scramble,reversed}.mp3` | 0.3 MB each | the four `assets/conditions/` clips above, mono 32 kbit/s MP3, by `scripts/build_pilot_media.py` (sources and SHA-256 in `demo/media/MEDIA.json`) |
| `demo/media/{bridge_t032.5,f1237}_{intact,faces_removed,area_matched}.jpg` | 24 KB each | Tears of Steel (Blender Foundation, CC BY 3.0): the final v2 edit-check frames at two instants, intact, faces filled and same-size control, resized to 640 px by `scripts/build_pilot_media.py`; each source frame's SHA-256 is checked against the edit check's frame manifest (`frames_manifest_sha256` in `results/edit_check_spark_open.json`) |
| `web/film/{bridge_t032.5,f1237,f2453}.jpg` | 0.2 MB each | Tears of Steel (Blender Foundation, CC BY 3.0): intact frames of the edit check's frame set (1280x534 release frames), the hosted page's FILM station |
| `web/film/{bridge_t032.5,f1237,f2453}_cut.mp4` | 20 KB each | the same frames as one-frame H.264 clips at the bridge cut's 640x360 letterbox, by ffmpeg, for SAM 3.1 (SHA-256 on `stillhere/public_frame_allowlist.txt`) |
| `web/replay/film.json` | 0.5 MB | one recorded FILM run (SAM 3.1 masks of "face" on those frames, filled; JPEG data inside), by `web/record_replays.py` |
| `docs/assets/stillhere-hero.png` | 0.4 MB | `scripts/build_readme_art.py`: panel A, the menu's two brain areas (as `stillhere/heldout.py` `parcel_sets` defines them, Schaefer-400 annotation) on nilearn's bundled fsaverage5 pial surface, no data values; panel B, the LPP ghost line from `results/ghostline_lpp_s8.json` under `docs/rules/display_rule_ghostline.md` |
| `docs/assets/stillhere-architecture.png` | 0.6 MB | `scripts/build_readme_art.py`: drawn shapes and the submission's wording only |
| `web/public/hero.png` | 0.4 MB | the same picture as `docs/assets/stillhere-hero.png`, written by the same script for the hosted page's header |
| `web/public/fonts/{NotoSerif-Regular,NotoSerif-Italic,NotoSerif-Bold,NotoSansMono-Regular,NotoSansMono-Medium}.woff2` | 16-21 KB each | Noto Serif 2.015 and Noto Sans Mono 2.014 (The Noto Project Authors, SIL OFL 1.1, `web/public/fonts/OFL.txt`), subset to Latin and General Punctuation as WOFF2 with fontTools 4.62.1 (`fontTools.subset --flavor=woff2`) |
