<p align="center">
  <img src="docs/assets/stillhere-hero.png" alt="StillHere. Panel A: the menu's two brain areas, hearing and language, on a contour-line drawing of the left cortex. Panel B: TRIBE v2's forecast as a gray dashed line and 49 public listeners as a coloured line, Le Petit Prince section 8, held out." width="100%">
</p>

# StillHere

### Pilot the story before booking the scanner

> Labs study people who can't respond after brain injury by playing them stories in the scanner. But every new story needs its own
> 12-15 healthy volunteers scanned first, in that language. For a new story, StillHere gives labs a first reading with zero volunteer
> scans, measured against each listener's own brain. It has been tested only in healthy volunteers.

**Research use. Healthy volunteers and public data only. Not a diagnosis or medical advice.**

Built during the Meta Global AI Developer Hackathon (building period Oct 27 to Nov 4, 2026): Indie developer track, theme Push the
Frontier. MIT licence for our code; TRIBE v2 is CC BY-NC 4.0 and is not included.

[The problem](#the-problem) · [What it does](#what-it-does) · [Results so far](#results-so-far) · [Architecture](#architecture) ·
[Meta models](#meta-models-and-the-job-of-each) · [Keeping our own rules](#keeping-our-own-rules) · [Quickstart](#quickstart) ·
[Deployment](#deployment) · [Limitations](#limitations) · [Project layout](#project-layout) · [Documentation](#documentation)

---

## The problem

Passive tests play a story or a film to a person in the scanner and look for the brain response a healthy listener would have. They
matter most for people who cannot answer: 1 in 4 brain-injured patients who could not follow commands at the bedside showed brain
activity on command-following tasks, on fMRI or EEG (Bodien et al., NEJM 2024; 60 of 241). That figure comes from command-following
tests, not passive listening. Passive tests with a story or a film are the other half of the toolkit (Naci et al., PNAS 2014, used a
film).

Every new story or film needs its own healthy reference first. Each story needs 12-15 volunteers: $5-15k of scanner time per
stimulus, per language, at university scanner rates of $450-700 an hour (Michigan $620, Kentucky up to $625, Yale $696), before
recruitment. So a new story, or the same story in a new language, starts with a scanner booking before anyone knows whether it will
work.

**StillHere Pilot lets a lab pilot the story before booking the scanner.** A researcher types the study they want; Muse Spark 1.3
turns it into a plan from a tested menu, or declines with the reason. StillHere builds the conditions, and Muse Voice Transcribe reads
every one back before anyone is scanned. TRIBE v2 forecasts the brain response, live on a laptop. Gray is the forecast; colour is
real listeners: on a held-out audiobook, the forecast was worth about 5 real volunteers, and at least 2.

A first healthy reference for a new story, before any scans, and evidence for how far to trust it.

## What it does

The Pilot page is a rail of six stations. Each one calls a Meta model live, or shows its recorded run, labelled so, when live calls
are off or a budget is spent.

| | Station | What happens | Meta model |
|---|---|---|---|
| 1 | **Plan** | The researcher types a study. The planner turns it into a strict study spec from a tested menu (a JSON schema built from the stimuli, controls and analyses we have evidence for), or declines with a reason code and the nearest tested option. Ask about one patient's scan, and it declines, with the reason. A request can name content notes to avoid; code, not the model, filters the menu on the story cards. | Muse Spark 1.3 (default); Muse Spark Open on Fireworks or Baseten (private mode) |
| 2 | **Build and check** | StillHere cuts the story at sentence boundaries with short crossfades. Muse Voice Transcribe reads the built audio back and fills the checklist: words kept and word order decide whether a control is what it claims to be. A failed check blocks the build. "Rebuild with the old bug" replays the broken word scramble the check caught. | Muse Voice Transcribe |
| 3 | **Audit** | Spark Open reads the plan and leaves notes. Advisory check, not measured: it never blocks anything. | Muse Spark Open |
| 4 | **Forecast** | A judge picks one of 4 screened public stories, a condition and a seed. TRIBE v2 forecasts the brain response, stage by stage, on the laptop GPU, with its Llama 3.2 text lane on (fp16). A saved forecast is shown whenever one exists for your choice. New seeds run live on the laptop, 10:00-14:00 ET, Nov 4-9. Every live run ends on the gray forecast, with no real line drawn over it. No one has scanned this yet. | TRIBE v2 (V-JEPA 2, w2v-BERT, Llama 3.2 inside) |
| 5 | **Evidence** | Did the forecasts hold up? Each forecast against real public scans it never saw. These panels replay committed results files, each beside its cheap baselines. | TRIBE v2 |
| 6 | **Film** | The same loop on video. SAM 3.1 masked the faces in the first 190 s of Tears of Steel; StillHere filled them with the surrounding colour, and filled a same-size patch elsewhere in each frame as a control. Type "face", or another word, to make the same edit on three frames. | SAM 3.1, Muse Glimmer 30B, Muse Spark Open |

The live menu (`docs/rules/story_menu_rule.md`, exploratory; rule committed before any story was downloaded): 4 screened public
LibriVox stories, then the Alice chapters as backup. The screen runs Muse Voice Transcribe with diarization: one voice, no silence
over 5 s, and agreement with the word table TRIBE reads. Spark Open read every screened story and noted what is in it, with the line
and the time. The notes are the union of word-for-word quotes found in the transcript over 3 calls at temperature 0.6. Not a full
review; listen before you use a story.

## Results so far

Not a diagnosis or medical advice. Exploratory unless pre-registered; reported as it came. Each row names the committed file its
numbers come from.

| Test | Data | Result | Status | File |
|---|---|---|---|---|
| Held-out audiobook | Le Petit Prince, section 8, 49 listeners (OpenNeuro ds003643) | The forecast was worth about 5 real volunteers, and at least 2 (volunteer-equivalent 5.0, lower bound 2.58). Language result 1.76 z with word rate and sound partialled out. | Held out; first measured Oct 22-23, recomputed with code built during the event. LPP is one of TRIBE v2's own evaluation sets. | `results/section8.json` |
| Cheap baselines, same section | the same 49 listeners | Beaten in 8 of 8 comparisons (word rate and the sound envelope). In language areas it beat a simple word-rate model in 43 of 49 listeners; the forecast scored unpartialled, 1.65 z over the same 145 time points. | Exploratory: the word-rate rule was written after section 8 was first measured. | `results/section8_baselines.json` |
| Intact vs sentence-shuffled | Narratives notthefall, 22 listeners (OpenNeuro ds002345) | Stock TRIBE v2's language readout was stronger when the same healthy listeners heard the story intact than shuffled: HL 0.0356 [95% CI 0.0125, 0.0588], one-sided Wilcoxon p = 0.00516. | **Pre-registered (PREREG D20, words from D27): pass.** | `results/d20_notthefall.json` |
| Intact vs the coarser scramble | the same story, 21 listeners | HL 0.0215 [95% CI -0.0068, 0.0538], p = 0.0735; did not reach p < 0.05. | Pre-registered secondary: the result of record, a fail. | `results/d20_notthefall.json` |
| Planner v2 | 62 held-out requests, bars set before the run | Muse Spark 1.3: decision accuracy 93.0% (pass); false accepts 3 of 132 (2.3%, exact 95% upper bound 6.5%; pass); first-answer validity 83.3% against a bar of 85% (missed). Muse Spark Open: false accepts 7.6% (Fireworks) and 6.8% (Baseten) against a bar of 5% (missed). | Exploratory rule, `docs/rules/planner_v2_eval_rule.md`. v2.1 adds clip, condition, seed and story tags; not re-measured. | `results/planner_v2_eval.json` |
| Stimulus check | Voice Transcribe on the built conditions | The sentence scramble: 98% of the words heard, reordered. Our first word scramble kept only 16% of its words, so it was caught as broken before scanning and is not used. | | `results/conditions_manipulation_check.json` |
| Order map go/no-go | 3 LibriVox Alice clips, forecast only, no brain data | A new sentence order changed the language-area forecast more than the hearing-area forecast on 1 of 3 clips, short of the 3 of 3 set before any forecast. So no order map is shown. | Exploratory; no-go. | `results/order_map_gonogo.json` |
| Film edit check, Spark Open | 62 frames per version, Tears of Steel | Spark Open (Baseten): face seen in 54 of 62 intact frames, no face seen in 62 of 62 filled frames, face seen in 51 of 62 control frames. | Exploratory; Muse Glimmer 30B decides (Sat Oct 31 run). | `results/edit_check_spark_open.json` |
| Story wall and story picker | Narratives and Le Petit Prince, 42 stories | Every story we scored stays on the wall, weak ones too. | Pre-registered (D19); scored once, from Sat Oct 31. Scoring rule committed Oct 28, 340b74b; 41 of the 42 stories were used before then, for development or to tune a second model. | `results/story_wall.json`, when it lands |
| Film face test | Emo-FilM, 28 viewers of the intact film | Does TRIBE's forecast of the face edit land where viewers' brains follow faces? | Scored once by a rule committed before any viewer's brain data was read (`docs/rules/face_map_rule.md`). | `results/face_map.json`, when it lands |
| Sedation case study | frozen group statistics | The write-up, with exploratory variants labelled. | Frozen fail format. | [docs/writeup_sedation.md](docs/writeup_sedation.md) |

## Architecture

<p align="center">
  <img src="docs/assets/stillhere-architecture.png" alt="StillHere architecture. I, the stations: request, typed egress, plan, build and check, audit, forecast on the laptop GPU, the end screen, receipts and caps. II, evidence: rules first, Kaggle GPUs, public scans, scoring, results files. III, film: Tears of Steel, SAM 3.1 masks, fill and control, the edit check, the face test." width="100%">
</p>

- **One typed egress layer** (`stillhere/egress.py`) is the only way anything reaches a model API: public text, scanned requests,
  allowlisted public audio and film frames, three hosts (`api.meta.ai`, `api.fireworks.ai`, `inference.baseten.co`). Keys live in
  environment variables only; no response carries one.
- **Planner** (`stillhere/planner.py`, `stillhere/planner_v21.py`): Muse Spark 1.3 turns a typed request into a strict study spec, or
  declines with a reason code and the nearest supported option. Planner v2 is the version measured on 62 held-out requests and stays
  as published. In v2.1 the model only plans and our code runs the tools (list stimuli, make conditions, check conditions, forecast).
- **Manipulation check** (`stillhere/conditions.py`, `stillhere/manipulation_check.py`): Muse Voice Transcribe reads every built
  condition back; words kept and word order decide whether a control is what it claims to be.
- **Live forecast** (`worker/`, `docs/worker.md`, `docs/jobs_api.md`): stock TRIBE v2 on a laptop (RTX 4050, 6 GB GPU). The Llama
  3.2 3B text lane runs in fp16, split between the GPU and the CPU by accelerate; w2v-BERT 2.0 for sound and the brain model run on the
  CPU; V-JEPA 2 is off for audio clips. A job is a recipe and nothing else (a story from the live menu, a condition and a seed). The
  worker runs sandboxed on the laptop and pulls jobs over HTTPS; the laptop accepts no inbound connection.
- **Hosted app** (`web/`, `api/index.py`): one WSGI app on Vercel's Python runtime, standard library around the repo's own modules.
  Receipts for every live call (model, time, cost; no request text, no key), spend caps per provider, and 5 live runs an hour per
  visitor. State lives in Upstash Redis.
- **Forecasts against real brains** (`kernels/`, `stillhere/heldout.py`, `stillhere/cheap_baselines.py`): TRIBE v2 runs in private
  Kaggle GPU notebooks. Scoring compares each forecast with each real listener's own brain, against a shuffled null and against cheap
  baselines (word rate and the sound envelope; for film, brightness and motion too) that every result is shown beside.
- **Film** (`stillhere/face_edit.py`, `stillhere/face_map.py`, `web/sam.py`): SAM 3.1 tracked face masks over 4,560 frames of Tears
  of Steel through Meta's Model API, decoded with meta-sam-parser; faces filled with the surrounding colour; an equal-area control
  with the same fill inside the 470 x 470 part of the frame TRIBE's video model sees; every version checked through TRIBE's own video
  reader. Muse Glimmer 30B (open weights, on a Kaggle GPU) decides the edit check; Spark Open's counts are shown beside it.

## Meta models and the job of each

| Model | Where it runs | Its job |
|---|---|---|
| Muse Spark 1.3 | Meta Model API | The default study planner. Plans the study, or declines with the reason. |
| Muse Spark Open | Fireworks and Baseten (reported as two hosted versions, not one model) | Private planner mode, the advisory check, and the content notes on the screened stories; its edit-check counts are shown beside Glimmer's. |
| Muse Voice Transcribe | Meta Model API | Reads every built condition back before anyone is scanned, and screens the live menu's stories. |
| SAM 3.1 | Meta Model API | Masks the faces in the film, for the face edits and their equal-area controls. |
| Muse Glimmer 30B | open weights, Kaggle GPU | The deciding edit check. |
| TRIBE v2 (Meta FAIR), with V-JEPA 2, w2v-BERT and Llama 3.2 inside | laptop GPU (live) and Kaggle GPUs | Forecasts the brain response, for stories and film. |
| Muse Code | the developer's machine | Built the planner (standard model only); runs the pre-registration guard as a Stop hook. |

What we learned about them:

- A strict schema and a tested menu let the model make the decision. In Planner v0 our fixed rules did most of the declining (107 of
  114 for Spark 1.3); in v2 the model did (136 of 139 declines for Spark 1.3).
- Two hosts of the same open model are two models for evaluation: Spark Open's false-accept rate was 7.6% on Fireworks and 6.8% on
  Baseten, so each is reported on its own.
- Cheap baselines carry much of what a forecast seems to capture, so every result is shown beside them.

## Keeping our own rules

- **Rules first.** Every rule and its scoring code is committed and hash-pinned before any brain data it scores is read
  (`docs/rules/`, [PREREG.md](PREREG.md)). The commit time is the record. [PREREG.md](PREREG.md) is the frozen pre-registration
  (D0-D27). Lines it quotes in its Never lists, and lines replaced by D18 and D23, are not claims.
- **Frozen code stays frozen.** `prereg_registry.json` pins the SHA-256 of every entry point that scores a registered test. A git
  pre-commit hook and a Muse Code Stop hook (`muse-plugin/prereg-guard/`) run `stillhere/prereg_check.py` and block changes to frozen
  code. Every later plumbing change is logged with its timing in `docs/frozen_code_changes.md`, made before the relevant brain data
  was read.
- **Go/no-go before the picture.** A go/no-go written before the forecast keeps a good-looking picture honest: the order map did not
  pass its own rule, so it is not shown.
- **Captions before results.** Every caption, for every ending, was written before the result was known (`docs/captions.md`).
- **Nothing private leaves.** A pre-push hygiene scan (`scripts/hygiene_scan.py`) fails closed on keys and tokens, agent-session
  links, absolute paths that name the laptop user, subject IDs in the private sedation dataset's format, em dashes and files over
  10 MB, across every file and every commit message in the history. It reads a laptop-only word list kept outside the repository and
  refuses to push without it. Binary files are listed with what built them in `docs/BINARY_PROVENANCE.md`.

## Quickstart

### 1. Install

Python 3.12.

```sh
git clone https://github.com/Yatsuiii/stillhere.git && cd stillhere
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
git config core.hooksPath scripts/hooks      # the pre-registration guard and the pre-push hygiene scan
```

`requirements.txt` holds the hosted app's runtime dependencies, pinned to the versions it was tested with. The face edits also need
`requirements-face-edits.txt`. TRIBE v2 is not included (CC BY-NC 4.0); the live forecast runs on the laptop worker (step 5).

### 2. Run the Pilot app locally

```sh
python web/serve.py                       # http://127.0.0.1:8780/, in-memory store
STILLHERE_LIVE=0 python web/serve.py      # every station shows its recorded run
```

Live calls happen only where keys are in the environment: `MODEL_API_KEY` for Spark 1.3, Voice Transcribe and SAM 3.1;
`FIREWORKS_API_KEY` and `BASETEN_API_KEY` for Spark Open. Without them every station shows its recorded run. `web/README.md` covers
the stations, the caps and the fallbacks.

### 3. Run the tests

```sh
python -m pytest -q
python web/smoke_local.py                 # the hosted app end to end in production mode, locally
```

No test calls a model or reads a key.

### 4. Rebuild the pages and images

```sh
python scripts/build_pilot_page.py        # demo/index.html: the static, replay-only Pilot page
python scripts/build_readme_art.py        # docs/assets/: the README's hero and architecture images
```

Every number on the static page comes from a committed JSON file. The README images need nilearn, nibabel, matplotlib, scipy and
`rsvg-convert` (see the script's docstring); nothing is downloaded.

### 5. The live worker (optional, a laptop with an NVIDIA GPU)

```sh
sudo bash worker/install_worker.sh <page-url>
```

A systemd service with a dynamic user and a sandbox pulls jobs from the page over HTTPS and runs TRIBE v2 on the local GPU during the
set hours. See `docs/worker.md` and `docs/jobs_api.md`.

## Deployment

The hosted Pilot app runs on Vercel's Python runtime (`vercel.json` pins the Python framework preset; `api/index.py` is the one
entrypoint) with Upstash Redis for receipts, budgets and jobs. Git is not connected, so a push never deploys; each deploy is a
preview first, checked, then promoted. Every step, the environment variables and the smoke checks are in
[docs/deploy_checklist.md](docs/deploy_checklist.md).

**Try it:** the hosted page's address is listed here on Nov 3. Until then, open `demo/index.html` from a clone: the static,
replay-only Pilot page, no script and no network.

## Limitations

- Research use only. Healthy volunteers and public data only. Not a diagnosis or medical advice.
- Every live run is a forecast, not a scan: the end screen draws no real line, because no one has scanned that story yet.
- The held-out audiobook (Le Petit Prince, section 8) was first measured Oct 22-23, before this repository began, and LPP is one of
  TRIBE v2's own evaluation sets. The word-rate comparison is exploratory: its rule was written after section 8 was first measured.
- Intact vs sentence-shuffled: each version also had its own TRIBE forecast, so a pass does not separate a change in the listeners'
  brains from a change in TRIBE's forecast, and the run order is not public, so it cannot be separated from an order or
  repeat-listening effect. The secondary comparison did not reach p < 0.05.
- The private-mode planner (Spark Open) is over its 5% false-accept bar on both hosts; Planner v2.1 is not re-measured.
- Close-up faces leave no room for an equal-area control: the 51 largest close-ups have none, so the controls cover 88.9% of in-view
  face pixels.
- Content notes are not a full review; listen before you use a story.
- A 6 GB laptop GPU runs the live forecast, during set hours only (10:00-14:00 ET, Nov 4-9); outside them a saved forecast is shown
  when one exists.
- GPU time: if all 42 picker stories are not ready by Sun Nov 1, 18:00 UTC, the picker and the wall are reported "pre-registered, not
  run", with no partial scoring.

## Project layout

```text
api/            Vercel entrypoint: index.py exports the hosted app's WSGI callable
web/            the hosted Pilot app: stations, guard, ledger and caps, jobs API, receipts; web/public is the page
stillhere/      the library: typed egress, Meta Model API client, planner, conditions and checks, held-out scorer,
                cheap baselines, face edits and the face map, the sedation desk, the pre-registration check
worker/         the laptop worker: recipe, engine, sandboxed systemd install
kernels/        Kaggle notebook kernels (film forecasts)
scripts/        builders and runners: the static page and its media, conditions, planner evaluations, maps, the
                ghost line, the video, the README images, the hygiene scan; scripts/hooks holds the git hooks
results/        committed results files the Evidence station replays
eval/           the planner's question sets (held-out set included), the demo requests, the edit-check sample
data/           the live clips, the film stimuli's metadata, the forecast cache
assets/         public-domain LibriVox clips and the built conditions
demo/           the static, replay-only Pilot page and its media
docs/           rules, deploy checklist, jobs API, worker, captions, write-ups, binary provenance, images
replay/         the sedation desk's recorded replay
video/          the video's scene files
muse-plugin/    the pre-registration guard as a Muse Code Stop hook
tests/          the pytest suite
PREREG.md       the frozen pre-registration (D0-D27)
prereg_registry.json   SHA-256 of every entry point that scores a registered test
```

## Documentation

| Document | What it is |
|---|---|
| [PREREG.md](PREREG.md) | The frozen pre-registration (D0-D27) |
| [docs/rules/](docs/rules/) | One rule per test, each committed before its data was read: cheap baselines, the ghost-line display, the edit check, Emo-FilM, the face map, the film stimuli, Glimmer's face-edit check, the order-map go/no-go, the planner evaluations, Spark Open's face check, the story menu |
| [docs/frozen_code_changes.md](docs/frozen_code_changes.md) | Every change to frozen code, with its timing and why |
| [docs/deploy_checklist.md](docs/deploy_checklist.md) | Deploying the hosted Pilot app: accounts, environment variables, smoke checks, redeploys |
| [docs/jobs_api.md](docs/jobs_api.md) | The jobs API between the hosted page and the laptop worker |
| [docs/worker.md](docs/worker.md) | The live worker: what a job can ask for, the sandbox, the hours |
| [docs/story_menu_data.md](docs/story_menu_data.md) | The story card on PLAN and BUILD |
| [docs/captions.md](docs/captions.md) | Captions, voice-over and slides, written before the results |
| [docs/vo_script.md](docs/vo_script.md) | The video's voice-over script |
| [docs/writeup_sedation.md](docs/writeup_sedation.md) | The sedation case study: frozen group statistics |
| [docs/BINARY_PROVENANCE.md](docs/BINARY_PROVENANCE.md) | Every binary file and what built it |
| [web/README.md](web/README.md) | The hosted app: stations, caps, fallbacks, how it is built |

## Built during the event

The product repository started on 2026-10-27, inside the building period (Oct 27 to Nov 4); this public repository was cut from it
on 2026-10-28 (first commit 340b74b). Built during the event: the typed egress layer,
the Meta Model API client, the planner (v0, v2 and v2.1) and its evaluation, the Voice Transcribe checks, the live laptop forecast,
the held-out section scorer, the film edits and edit check, the face-map test, the cheap-baseline scoring, the order-map go/no-go, the
pre-registration guard and its Muse Code hook, the Pilot page and the video. Before the event: the pre-registration (Oct 21-26), the
Le Petit Prince calibration and timing harness (Oct 21-23), and the TRIBE tuning pipeline code (Oct 24-25; run Oct 26). The held-out
audiobook result was first measured Oct 22-23 and is recomputed with code built during the event; `stillhere/heldout.py`: 12 of 215
lines follow pre-event code (language-area parcel list, lag and resampling, one alignment call); the rest was written from Oct 27.

## What's next

- A lab's own new story, with its own listeners, intact vs shuffled, scored once by a rule written first.
- Bring the private-mode planner under its 5% false-accept bar before it is offered as a default, and measure v2.1.
- More held-out stories and films, in more languages.
- Let a lab run the forecast on its own GPU (TRIBE v2 is CC BY-NC 4.0 and is not included in the repository).

## Licences and credits

- StillHere code: MIT (see `LICENSE`). TRIBE v2 is CC BY-NC 4.0 and is not included (see `NOTICE`).
- TRIBE v2: Meta FAIR (d'Ascoli et al. 2026), weights and code released under CC BY-NC 4.0.
- SAM 3.1 (Meta): the face masks for the Tears of Steel edits come from SAM 3.1 through Meta's Model API and are decoded with
  meta-sam-parser (Meta's package, SAM License; a dependency listed in `requirements-face-edits.txt`, not included). We acknowledge
  the use of SAM in this work, as the SAM License requires.
- Data: Le Petit Prince fMRI corpus (OpenNeuro ds003643); Narratives (ds002345); Emo-FilM (ds004892, ds004872, CC0); LibriVox
  recordings (public domain); Tears of Steel and Sintel (Blender Foundation, CC BY 3.0).
- Brain surface in the README images: fsaverage5 as bundled with nilearn; areas from the Schaefer 2018 400-parcel, 17-network atlas.
- Epigraph: Lewis Carroll, *Alice's Adventures in Wonderland* (1865), public domain; its chapters are on the story menu.
