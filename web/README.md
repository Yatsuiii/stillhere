# StillHere Pilot, the hosted app (cloud plan v2, items 4-6)

A stations rail: PLAN (planner v2.1 on Muse Spark 1.3, Spark Open selectable), BUILD (the condition cut at sentence boundaries,
Muse Voice Transcribe reads it back and fills the checklist; Reshuffle; "Rebuild with the old bug"), AUDIT (Spark Open, advisory
check, not measured), FORECAST (TRIBE v2 on the laptop GPU through the jobs API, lane bars from real worker events), EVIDENCE (the
static page's panels) and FILM (SAM 3.1 on three allowlisted Tears of Steel frames, our fill and a same-size control). Every live run
ends on the gray forecast line alone: "No public scan of this clip. Forecast, not a scan."

Nothing is deployed. Deploying, the accounts and the keys are Yatsu's steps (below).

## Run it locally (no account)

```sh
python web/serve.py                       # http://127.0.0.1:8780/, in-memory store
STILLHERE_LIVE=0 python web/serve.py      # every station shows its recorded run
```

Live calls happen only where keys are in the environment: `MODEL_API_KEY` (or the laptop's Muse config) for Spark 1.3, Voice
Transcribe and SAM 3.1; `FIREWORKS_API_KEY` for Spark Open. FILM also needs opencv and meta-sam-parser (`web/api/requirements.txt`).
Tests: `pytest tests/test_web_app.py` (no key ever reaches the browser, fallbacks with the APIs blocked, guard, caps and limits, a
fake worker through the jobs API, the evidence endings, 360 px).

## How it is built

| File | Job |
|---|---|
| `app.py` | One WSGI app, standard library only around the repo's modules; session cookie; CSP; never logs request bodies |
| `stations.py` | The stations; every model call goes through `stillhere/egress.py` |
| `ledger.py` | Spend reservations and caps, 5 runs per hour per visitor, one use of each station per run |
| `guard.py` | Every model-written line passes the banned-word list or becomes a template line |
| `jobs.py` | The jobs table and worker API (`docs/jobs_api.md`); recipes and cache keys come from `worker/recipe.py` |
| `store.py` | Upstash Redis over REST when its env vars are set; an in-memory stand-in otherwise |
| `sam.py` | SAM 3.1 on one allowlisted one-frame clip (`film/frames.json`) |
| `replay/` | The recorded runs (`record_replays.py`, from real live calls) every station falls back to, labelled "recorded run" |
| `public/` | The page, `receipt.html`, and their scripts (no inline script; the CSS is the static page's, served as `/pilot.css`) |

Typed text is used for its one model call and never stored or logged; receipts hold hashes, model ids, times, costs and the GPU.
On the server, `egress.PublicQuestion` reads `web/server_question_blocklist.txt` (a salt and no terms: the laptop-only list comes from
the confidential clip and never leaves the laptop, and the server holds nothing from that clip).

## Money

- Hard stops kept in the store, so they hold across restarts and instances: Meta $25 total and $8 a day, Fireworks $15 total and
  $5 a day (env `CAP_*_USD`, `DAILY_CAP_*_USD`). Each call reserves its worst case first; chat calls settle to their token cost at the
  planner rule's prices. Voice Transcribe ($0.10 reserved) and SAM 3.1 ($0.50 reserved per frame) publish no price here, so their
  reservations stand. Baseten is not offered live.
- Provider alerts Yatsu should set where the consoles offer them (whether a hard per-key limit exists is unknown): Meta Model API
  usage or billing alert below the remaining credit; Fireworks spend limit or alert at $15; Upstash max monthly budget (free tier);
  Vercel Hobby has no paid overage.

## Deploy (Yatsu's steps; nothing here has been run)

1. Vercel (Hobby) account; Upstash Redis from the Vercel Marketplace (sets `KV_REST_API_URL` and `KV_REST_API_TOKEN`).
2. Server secrets, entered by Yatsu: `MODEL_API_KEY`, `FIREWORKS_API_KEY`, `STILLHERE_WORKER_TOKEN`; plus
   `STILLHERE_QUESTION_BLOCKLIST=web/server_question_blocklist.txt`, `STILLHERE_USAGE_LOG=/tmp/usage.jsonl`, `WORKER_HOURS_ET`.
3. From the repository root: `vercel deploy -A web/vercel.json` (ASSUMPTION: the Python builder serves the WSGI `app` and takes the
   `includeFiles` list; check on the first preview deploy). Keep the URL unlisted until Nov 3.
4. The worker on the laptop gets the same `STILLHERE_WORKER_TOKEN` and the page's URL (worker fork).

## Worker hours

10:00-14:00 ET, Nov 4-9 (15:00-19:00 UTC) (ruling of Oct 29, 17:35 UTC, item 8; `WORKER_HOURS_ET`). The page says: "A saved forecast is shown whenever one
exists for your choice. New seeds run live on the laptop, 10:00-14:00 ET, Nov 4-9." Outside the hours, a saved run at another seed
is labelled "No saved run for seed {seed}; shown: the saved run at seed 20261028."

## Hard stops (cloud plan v2)

Live parts by Sat Oct 31 12:00 UTC, else ship cache-only, labelled; the whole page by Sun Nov 1 12:00 UTC, else the clean replay
page (`demo/index.html`, built by `scripts/build_pilot_page.py`).
