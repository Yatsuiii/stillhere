# StillHere Pilot, the hosted app (cloud plan v2, items 4-6)

A stations rail: PLAN (planner v2.1 on Muse Spark 1.3, Spark Open selectable), BUILD (the condition cut at sentence boundaries,
Muse Voice Transcribe reads it back and fills the checklist; Reshuffle; "Rebuild with the old bug"), AUDIT (Spark Open, advisory
check, not measured), FORECAST (TRIBE v2 on the laptop GPU through the jobs API, lane bars from real worker events), EVIDENCE (the
static page's panels) and FILM (SAM 3.1 on three allowlisted Tears of Steel frames, our fill and a same-size control). Every live run
ends on the gray forecast line alone, with the clip's own end line from `stillhere/live_menu.json` `public_scan`: "No public
scan of this clip. Forecast, not a scan." for a clip no public dataset played, else for example "A public scan of this chapter exists
(Alice Datasets, OpenNeuro ds002322); not compared here. Forecast, not a scan."

Deployed on Vercel (`docs/deploy_checklist.md`); its address is listed from Nov 3. The accounts and the keys are Yatsu's steps (below).

## Run it locally (no account)

```sh
python web/serve.py                       # http://127.0.0.1:8780/, in-memory store
STILLHERE_LIVE=0 python web/serve.py      # every station shows its recorded run
```

Live calls happen only where keys are in the environment: `MODEL_API_KEY` (or the laptop's Muse config) for Spark 1.3, Voice
Transcribe and SAM 3.1; `FIREWORKS_API_KEY` for Spark Open. FILM also needs opencv and meta-sam-parser (the root `requirements.txt`).
Tests: `pytest tests/test_web_app.py` (no key ever reaches the browser, fallbacks with the APIs blocked, guard, caps and limits, a
fake worker through the jobs API, the evidence endings, 360 px).

## How it is built

| File | Job |
|---|---|
| `app.py` | One WSGI app, standard library only around the repo's modules; session cookie; CSP; never logs request bodies |
| `stations.py` | The stations; every model call goes through `stillhere/egress.py` |
| `ledger.py` | Spend reservations and caps; visitor ids issued only by the page's first call (`POST /api/visitor`, a fresh id each, SET NX, at most 20 per address an hour; "Plan it" waits for it), so a request without one counts as its network address (x-real-ip, an IPv6 /64) and no page load touches the store; per UTC day a visitor and an address each keep at most 20% of each daily cap; 5 runs per true sliding hour per visitor, 20 per address (sorted sets of event times); one use of each station per run; an hourly cap on each station for all visitors together (PLAN and AUDIT 120, BUILD 70, FORECAST 30, FILM 10), of which one address takes a fixed slice of at most a quarter (PLAN and AUDIT 10, BUILD 17, FORECAST 7, FILM 2); every POST from the page first passes a per-address token bucket in each instance (20 at once, then one every 10 s), and a receipt or job poll (GET) a second one (60 at once, then one a second, above the page's 1.5 s poll), each before any store command (past the POST bucket, a 429 carrying the station's recorded run and its why line), claimed after every per-visitor check and given back on any error before the paid call, then its recorded run; with the store failing, every station gives its recorded run ("Live runs are paused") |
| `guard.py` | Every model-written line passes the banned-word list or becomes a template line |
| `jobs.py` | The jobs table and worker API (`docs/jobs_api.md`); recipes and cache keys come from `worker/recipe.py` |
| `store.py` | Upstash Redis over REST when its env vars are set; an in-memory stand-in otherwise (on Vercel, live calls need Upstash) |
| `smoke_local.py` | The local production-mode smoke test (`docs/deploy_checklist.md`, 8a) |
| `sam.py` | SAM 3.1 on one allowlisted one-frame clip (`film/frames.json`) |
| `replay/` | The recorded runs (`record_replays.py`, from real live calls) every station falls back to, labelled "recorded run" |
| `public/` | The page, `receipt.html`, and their scripts (no inline script; the CSS is the static page's, served as `/pilot.css`) |

Typed text is used for its one model call and never stored or logged; receipts hold hashes, model ids, times, costs and the GPU.
On the server, `egress.PublicQuestion` reads `web/server_question_blocklist.txt` (a salt and no terms: the laptop-only list comes from
the confidential clip and never leaves the laptop, and the server holds nothing from that clip).

## Money

- Hard stops kept in the store, so they hold across restarts and instances (cloud plan of Oct 29, 19:20 UTC, 1c): totals Meta $80,
  Fireworks $50, Baseten $50; daily Meta $20, Fireworks $12, Baseten $12 (env `CAP_*_USD`, `DAILY_CAP_*_USD`). Each call reserves its
  worst case first; chat calls settle to their token cost at the planner rule's prices. Voice Transcribe ($0.10 reserved) keeps its
  reservation. SAM 3.1 settles each completed frame to `SAM_USD_PER_FRAME`, set to $0.01: Meta lists video at $0.20 per 1,000
  frames and a one-frame call bills a 50-frame minimum (`docs/deploy_checklist.md`); with it set, a frame reserves twice it ($0.06 a
  run), and unset, $0.50 a frame that it keeps ($1.50 a run; cloud review of Oct 31, 12:20 UTC, item 1). FILM reserves its three frames before the first call, gives each SAM call 60 s a phase and an end 180 s after its first frame (upload, stream and a 10 s best-effort file delete), waits for a call only until that end (the call runs in a thread), and starts no frame after 150 s, so it settles by about 180 s plus one frame's edit, inside the function's 300 s (item 7, and the refuters' items 3 and A); each chat
  call carries `max_tokens` so it cannot cost more than its reservation, and `settle` holds an over-run at the cap.
- Spark Open runs on Fireworks; Baseten (`BASETEN_API_KEY`) is its fallback host when Fireworks has no key, is capped, or fails. The
  receipt names the host that answered.
- The event's Meta credit: no source we could reach gives its end. Checked Oct 29: the event details page (cerebralvalley.ai, saved
  Oct 26: "$150 in Meta Model API credits ... a credit card must be applied to your account"; a "Spend Limit" notification on the
  Billing page), the event's Oct 26 emails, and the claim form ("credits to build with during the hackathon"; claim by Nov 4, 9:00 PM
  ET). Fireworks's grant says the opposite for its own credit: "Once added to your account, the credits don't expire" (Oct 27 email).
  Ruling of Oct 29, 19:40 UTC, item 3: the Meta stations (PLAN on Spark 1.3, BUILD, FILM) stay live through Nov 9 and fall back only
  when a call fails ("The live call failed, so this station shows its recorded run."). After `META_CREDIT_END_UTC` (default Nov 5
  02:00 UTC, the end of the event's ten days of building; set it to the date on the Billing page if one shows), Meta spend gets one
  more total cap of $20 (`POST_CREDIT_CAP_META_USD`), then each Meta station shows its recorded run with "The event's Meta API credit
  has ended and the $20 set aside after it is spent, so this station shows its recorded run."
- Provider alerts Yatsu should set: Meta Model API "Spend Limit" notification on the Billing page (the event page names it); Fireworks
  spend limit or alert at $50; Baseten usage alert at $50; Upstash pay-as-you-go with a $5 budget (since Oct 31). Vercel Hobby has no paid overage.

## Deploy

Yatsu's steps, every environment variable name, and the preview smoke list: `docs/deploy_checklist.md`. The layout follows Vercel's
current Python runtime (checked Oct 30): `vercel.json` pins the Python framework preset, whose one entrypoint is `api/index.py`
exporting the WSGI `app`, and every path reaches it unchanged (no rewrite: under this preset a rewrite becomes the app's path); dependencies in the root `requirements.txt`; `.vercelignore` uploads only what the app reads.
Before any deploy, `python web/smoke_local.py` serves that same upload set in production mode with no account (keys absent, then dummy
keys with the model APIs blocked and a local stand-in for Upstash).

## Worker hours

10:00-14:00 ET, Nov 4-9 (15:00-19:00 UTC) (ruling of Oct 29, 17:35 UTC, item 8; `WORKER_HOURS_ET`). The page says: "A saved forecast is shown whenever one
exists for your choice. New seeds run live on the laptop, 10:00-14:00 ET, Nov 4-9." Outside the hours, a saved run at another seed
is labelled "No saved run for seed {seed}; shown: the saved run at seed 20261028."

## Hard stops (cloud plan v2)

Live parts by Sat Oct 31 12:00 UTC, else ship cache-only, labelled; the whole page by Sun Nov 1 12:00 UTC, else the clean replay
page (`demo/index.html`, built by `scripts/build_pilot_page.py`).
