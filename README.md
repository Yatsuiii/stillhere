# StillHere Pilot

Research tool for brain-imaging labs, built during the Meta Global AI Developer Hackathon. Healthy volunteers and public data only.
Not a diagnosis or medical advice.

A researcher types the study they want to run. Meta's models turn it into a study spec, build the stimulus conditions and their
controls, and check them. Meta's TRIBE v2 forecasts the healthy-adult brain response. StillHere then shows how such forecasts did
against real public scans:

- LPP English section 8 (Le Petit Prince, held out): recomputed with code built during the event; first measured Oct 22-23.
- The story wall across public Narratives and LPP stories (scored Thursday Oct 29).
- notthefall, intact against sentence-shuffled (Narratives).
- An Emo-FilM Blender film (Tears of Steel, CC BY 3.0) and its viewers.
- A sedation case study, shown only as frozen group statistics (k of N with k*, Hodges-Lehmann medians with intervals, and the
  pre-registered fallback lines). Its two headline bars did not pass, and the results say so.

Every new result is labelled exploratory. A forecast is labelled "forecast, not validated".

## What is here

| Part | Files |
|---|---|
| Typed egress: the only way anything reaches a model API (numbers and enums, SHA-256-allowlisted public text, scanned questions, allowlisted public audio and film frames; hosts api.meta.ai and api.fireworks.ai only) | `stillhere/egress.py`, `stillhere/meta_client.py` |
| Models | Meta Muse Spark 1.3 (Meta Model API), Meta Muse Spark Open (open weights, served by Fireworks), Muse Voice Transcribe, Meta TRIBE v2 |
| Spark analysis desk: answers only with lines the frozen bars license | `stillhere/desk.py`, `stillhere/frozen.py`, `stillhere/verifier.py`, `replay/` |
| Stimulus conditions (intact, sentence scramble, word scramble, reversed) and their Voice Transcribe manipulation checks | `stillhere/conditions.py`, `stillhere/manipulation_check.py`, `assets/` |
| Live audio-only TRIBE v2 on a laptop GPU | `stillhere/live_tribe.py`, `scripts/live_tribe_demo.py` |
| Held-out section scorer (LPP section 8) | `stillhere/heldout.py`, `scripts/heldout_section8.py` |
| Validation table the study planner is limited to | `stillhere/validation_table.json` |
| Pre-registration guard (git hook and Muse Code Stop hook) and the pre-push hygiene scan | `stillhere/prereg_check.py`, `muse-plugin/`, `scripts/hygiene_scan.py`, `scripts/hooks/` |
| Results (public data, or frozen group statistics) | `results/` |

Install the hooks with `git config core.hooksPath scripts/hooks`. The pre-push hygiene scan reads a laptop-only word list kept outside
the repository and refuses to push without it.

## Licences and credits

- StillHere code: licence not chosen yet (see `LICENSE`, a placeholder).
- TRIBE v2: Meta FAIR (d'Ascoli et al. 2026), weights and code released under CC BY-NC 4.0.
- Data: Le Petit Prince fMRI corpus (OpenNeuro ds003643); Narratives (ds002345); Emo-FilM (ds004892, ds004872, CC0); LibriVox
  recordings (public domain); Tears of Steel and Sintel (Blender Foundation, CC BY 3.0).
