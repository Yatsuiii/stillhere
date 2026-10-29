# Binary files and what built them

The hygiene scan reads text only, so every binary file in this repository is listed here with its inputs. None contains anything from
the privately shared sedation stimulus (no audio, frames, words, durations, word counts or predictions), no per-volunteer value and no
subject ID. The sedation case study (OpenNeuro ds003171) appears only in the page screenshot and the video, and only as the frozen
group statistics in `results/gate1_real.json` (counts per state, k of N with k*, Hodges-Lehmann medians with intervals, p values, the
pre-registered fallback lines, the lag marks and the timing note).

| File | Size | Built from |
|---|---|---|
| `assets/librivox_alice_ch01_20s.wav` | 640 KB | LibriVox *Alice's Adventures in Wonderland*, chapter 1, read by Kristen McQuillin (public domain; archive.org item `alice_in_wonderland_librivox`, file `wonderland_ch_01_64kb.mp3`), 0:20 to 0:40, mono 16 kHz |
| `assets/librivox_alice_ch01_75s.wav` | 2.4 MB | the same recording, 0:20 to 1:35, mono 16 kHz |
| `assets/conditions/alice_ch01_75s_intact.wav` | 2.4 MB | `librivox_alice_ch01_75s.wav` by `scripts/build_conditions.py` (word timings from the local faster-whisper base.en model) |
| `assets/conditions/alice_ch01_75s_sentence_scramble.wav` | 2.4 MB | the same input and script, sentences reordered (seed 20261028), 10 ms crossfades |
| `assets/conditions/alice_ch01_75s_word_scramble.wav` | 2.4 MB | the same input and script, words reordered (seed 20261028), 10 ms crossfades |
| `assets/conditions/alice_ch01_75s_reversed.wav` | 2.4 MB | the same input, time-reversed |
| `assets/diarization/two_readers_6turns.wav` | 1.9 MB | LibriVox *Alice* chapter 1 (Kristen McQuillin) and chapter 2 (Brad Bush), same archive.org item, public domain; six turns joined by `scripts/build_diarization_clip.py` |
| `demo/v1/screenshot.png` | 1.1 MB | headless Chrome render of the first Pilot page, `demo/v1/index.html`, which `scripts/build_demo_page.py` builds from `results/` JSON (section 8 and its baselines from public LPP data; conditions, diarisation and the live laptop forecast from the LibriVox assets; the sedation tab from `results/gate1_real.json` group statistics only), `stillhere/validation_table.json` and `replay/desk_real.jsonl` |
| `demo/media/alice_75s_{intact,sentence_scramble,word_scramble,reversed}.mp3` | 0.3 MB each | the four `assets/conditions/` clips above, mono 32 kbit/s MP3, by `scripts/build_pilot_media.py` (sources and SHA-256 in `demo/media/MEDIA.json`) |
| `demo/media/{bridge_t032.5,f1237}_{intact,faces_removed,area_matched}.jpg` | 24 KB each | Tears of Steel (Blender Foundation, CC BY 3.0): the final v2 edit-check frames at two instants, intact, faces filled and same-size control, resized to 640 px by `scripts/build_pilot_media.py`; each source frame's SHA-256 is checked against the edit check's frame manifest (`frames_manifest_sha256` in `results/edit_check_spark_open.json`) |
| `video/skeleton.mp4` | 6.3 MB | `scripts/render_video.py`: slides rendered from `video/scenes.json` and the same JSON files, with the page's CSS and charts; silent, no audio track; the sedation slide shows group statistics only |
| `video/contact_sheet.png` | 0.7 MB | the same slides, tiled |
