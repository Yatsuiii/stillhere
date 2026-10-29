# StillHere Pilot

Research tool for brain-imaging labs, built during the Meta Global AI Developer Hackathon. Healthy volunteers and public data only.
Not a diagnosis or medical advice.

A researcher types the study they want to run. Meta's models turn it into a study spec, build the stimulus conditions and their
controls, and check them. Meta's TRIBE v2 forecasts the healthy-adult brain response. StillHere then shows how such forecasts did
against real public scans:

- LPP English section 8 (Le Petit Prince, held out): recomputed with code built during the event; first measured Oct 22-23.
  Two numbers for the same 49 listeners: the language result is 1.76 z with word rate and sound partialled out of the forecast, as the
  language gauge does (`results/section8.json`); the comparison with word rate and the sound envelope scores the forecast unpartialled,
  1.65 z over the same 145 time points, so every predictor is scored the same way (`results/section8_baselines.json`).
  `stillhere/heldout.py`: 12 of 215 lines follow pre-event code (language-area parcel list, lag and resampling, one alignment call); the rest was written from Oct 27.
- The story wall across public Narratives and LPP stories (manifest Thursday Oct 29; scored once, Saturday Oct 31).
- notthefall, intact against sentence-shuffled (Narratives).
- An Emo-FilM Blender film (Tears of Steel, CC BY 3.0) and its viewers.
- A sedation case study: [docs/writeup_sedation.md](docs/writeup_sedation.md).

Every new result is labelled exploratory. A forecast is labelled "forecast, not checked against scans".

## What is here

| Part | Files |
|---|---|
| Typed egress: the only way anything reaches a model API (numbers and enums, SHA-256-allowlisted public text, scanned questions, allowlisted public audio and film frames; hosts api.meta.ai, api.fireworks.ai and inference.baseten.co only) | `stillhere/egress.py`, `stillhere/meta_client.py` |
| Models | Meta Muse Spark 1.3 (Meta Model API; the default planner), Meta Muse Spark Open (open weights, served by Fireworks and by Baseten; the private planner mode and a vision edit check), SAM 3.1 (Meta Model API; tracked face masks), Muse Glimmer 30B (open weights, llama.cpp on Kaggle; the deciding edit check), Muse Voice Transcribe (manipulation checks, diarisation), Meta TRIBE v2 (brain-response forecasts; inside it, V-JEPA 2 for video, w2v-BERT for audio and Llama 3.2 for text) |
| Sedation case-study desk: answers only with the lines the case study's frozen bars license (the write-up is `docs/writeup_sedation.md`) | `stillhere/desk.py`, `stillhere/frozen.py`, `stillhere/verifier.py`, `replay/` |
| Stimulus conditions (intact, sentence scramble, word scramble, reversed) and their Voice Transcribe manipulation checks | `stillhere/conditions.py`, `stillhere/manipulation_check.py`, `assets/` |
| Live audio-only TRIBE v2 on a laptop GPU | `stillhere/live_tribe.py`, `scripts/live_tribe_demo.py` |
| Held-out section scorer (LPP section 8) | `stillhere/heldout.py`, `scripts/heldout_section8.py` |
| Table of tested options the study planner is limited to | `stillhere/validation_table.json` |
| Pre-registration guard (git hook and Muse Code Stop hook) and the pre-push hygiene scan | `stillhere/prereg_check.py`, `muse-plugin/`, `scripts/hygiene_scan.py`, `scripts/hooks/` |
| Cheap baselines beside every TRIBE result (HRF x word rate, HRF x sound envelope); section 8 table on the Pilot page, labelled exploratory, rule committed after section 8 was first read (Oct 22-23) | `stillhere/cheap_baselines.py`, `scripts/section8_baselines.py`, `scripts/wall_baselines.py`, `results/section8_baselines.json` |
| Rules committed before each dataset's brain data is read (the commit time is the record) | `docs/rules/` |
| Face edits of Tears of Steel, release 0-190 s: SAM 3.1 masks through Meta's Model API; faces filled with the surrounding colour; equal-area controls with the same fill, inside the part of the frame TRIBE's video model sees; final stimuli checked through TRIBE's own video reader. The edit check (Spark Open now, Glimmer decides) and the face-map test (rule committed before any viewer's brain data is read; forecasts only for the edited versions) | `stillhere/face_edit.py`, `scripts/make_film_stimuli_v2.py`, `scripts/film_stimulus_checks.py`, `scripts/build_edit_check_frames.py`, `scripts/run_edit_check.py`, `stillhere/face_map.py`, `scripts/face_map_runner.py`, `scripts/emofilm_score_runner.py`, `docs/rules/film_stimulus_rule.md`, `docs/rules/edit_check_rule.md`, `docs/rules/face_map_rule.md`, `requirements-face-edits.txt` |
| Pilot page (static, replay only; stored card wording shown in plain words) | `demo/` (built by `scripts/build_pilot_page.py`, media by `scripts/build_pilot_media.py`); the first page was retired on Oct 29 and is in git history |
| Video | `video/`, `scripts/render_video*.py` |
| Sedation case study write-up (frozen group statistics; exploratory variants labelled) | `docs/writeup_sedation.md` |
| Results (public data, or frozen group statistics) | `results/` |

Install the hooks with `git config core.hooksPath scripts/hooks`. The pre-push hygiene scan reads a laptop-only word list kept outside
the repository and refuses to push without it.

## Licences and credits

- StillHere code: MIT (see `LICENSE`). TRIBE v2 is CC BY-NC 4.0 and is not included (see `NOTICE`).
- TRIBE v2: Meta FAIR (d'Ascoli et al. 2026), weights and code released under CC BY-NC 4.0.
- SAM 3.1 (Meta): the face masks for the Tears of Steel edits come from SAM 3.1 through Meta's Model API and are decoded with
  meta-sam-parser (Meta's package, SAM License; a dependency listed in `requirements-face-edits.txt`, not included). We acknowledge
  the use of SAM in this work, as the SAM License requires.
- Data: Le Petit Prince fMRI corpus (OpenNeuro ds003643); Narratives (ds002345); Emo-FilM (ds004892, ds004872, CC0); LibriVox
  recordings (public domain); Tears of Steel and Sintel (Blender Foundation, CC BY 3.0).
