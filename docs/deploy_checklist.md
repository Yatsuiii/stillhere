# Deploy checklist for the hosted Pilot page (cloud ruling of Oct 30, 04:20 UTC, item 8)

8a is done by the web fork (below). 8b to 8e are Yatsu's: the accounts, the keys and every deploy are his. No key is ever typed into a
chat, an issue, a commit, a command line argument or a terminal that is being recorded; every secret below goes from a file or a
pipe straight into `vercel env add`, which reads stdin. Labels: VERIFIED (checked here), ASSUMPTION (from Vercel's docs or the
dashboard as described; check it when you get there).

## Layout (checked against Vercel's Python runtime docs on Oct 30)

- `api/index.py` at the repository root exports the WSGI `app`. `vercel.json` pins Vercel's Python framework preset (`"framework":
  "python"`, what `vercel link` detects), which finds `api/index.py` as its one entrypoint (candidates: app, index, server, main, wsgi,
  asgi in the root, src/, app/, api/; only api/index.py is uploaded) and sends every path to it unchanged.
- No rewrite (VERIFIED Oct 30): the first deploy had `"rewrites": [{"source": "/(.*)", "destination": "/api/index"}]` and answered
  every path, `/` included, with `{"error": "not_found"}`, because under this preset the runtime copies the rewrite's destination into
  the app's path (vercel/vercel packages/python/src/index.ts, the catch-all after `handle: 'rewrite'`). Check 1 confirms the fix.
- `functions.api/index.py.includeFiles: web/public/**` (VERIFIED Oct 30): the Python builder leaves every `**/public/**` out of the
  function (its predefined excludes, for a CDN `public/` folder), so the second preview served `/api/status` but answered `/`,
  `/app.js` and the receipt page with 404; includeFiles is merged in after the excludes (packages/python/src/index.ts).
- `requirements.txt` at the root holds the runtime dependencies, pinned (numpy pinned so BUILD reproduces the allowlisted builds byte
  for byte). Python: Vercel's default, 3.12 (3.13 also works; `api/index.py` refuses anything older than 3.12).
- `.vercelignore` is an allowlist: only `api/`, `web/`, `stillhere/`, `worker/recipe.py`, `data/`, `results/`,
  `scripts/build_pilot_page.py`, one `assets/conditions` wav, `requirements.txt` and `vercel.json` are uploaded (155 files, 27 MB; the
  function's limit is 500 MB with the dependencies, about 140 MB).
- The earlier `web/vercel.json` used the `builds` property, which Vercel's docs now list under "Legacy ... deprecated", with
  `includeFiles` inside the builder's config; it is gone.

## 8a. Local production-mode smoke test (done by the web fork, Oct 30)

`PYTHONPATH=<site-packages with httpx, pydantic, numpy, cv2, meta_sam_parser> python web/smoke_local.py` copies exactly the upload
set into a temporary folder and serves `api/index.py` from there as Vercel does (working directory = project root, `VERCEL=1`, an
empty HOME, so no laptop key or list can be read). Result: SMOKE PASS.

- Run A, keys absent: `/api/status` shows live off; PLAN (Spark 1.3 and Spark Open), AUDIT, BUILD, the old bug and FILM answer with
  their recorded runs; FORECAST at a seed with no saved run shows the saved run with "No saved run for seed 3; shown: the saved run
  at seed 20261028."; the worker API answers 401 to every call.
- Run B, dummy keys, the model APIs blocked, a local stand-in for Upstash's REST API: PLAN on both models and FILM try live, fail, and
  show "The live call failed, so this station shows its recorded run."; Spark Open tried Fireworks, then Baseten; the ledger's
  reservations went through the REST store and were released; one jobs-API round trip (claim, event, result, done) ran through it.
- Both runs: no key or token in any response body, header or page file; a marker sent in every request body never appeared in the
  server's output, which held request lines only.

## Environment variables the app reads (names only)

| Name | Secret | Set by | Value or default |
|---|---|---|---|
| `MODEL_API_KEY` | yes | Yatsu | Meta Model API key (Spark 1.3, Voice Transcribe, SAM 3.1) |
| `FIREWORKS_API_KEY` | yes | Yatsu | Spark Open on Fireworks |
| `BASETEN_API_KEY` | yes | Yatsu | Spark Open on Baseten, the fallback host |
| `STILLHERE_WORKER_TOKEN` | yes | Yatsu | the laptop worker's job token (same value as `/etc/stillhere-worker/job_token`) |
| `KV_REST_API_URL`, `KV_REST_API_TOKEN` | token yes | the Upstash integration | the shared store; the app also reads `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` |
| `STILLHERE_QUESTION_BLOCKLIST` | no | Yatsu | `web/server_question_blocklist.txt` |
| `STILLHERE_USAGE_LOG` | no | Yatsu (optional) | `/tmp/stillhere_usage.jsonl` (the app uses this on Vercel when unset) |
| `WORKER_HOURS_ET` | no | Yatsu | `10:00-14:00 ET, Nov 4-9` |
| `META_CREDIT_END_UTC` | no | Yatsu | `2026-11-05T02:00:00Z`, or the end date the Meta Billing page shows |
| `CAP_META_USD`, `CAP_FIREWORKS_USD`, `CAP_BASETEN_USD` | no | Yatsu | `80`, `50`, `50` |
| `DAILY_CAP_META_USD`, `DAILY_CAP_FIREWORKS_USD`, `DAILY_CAP_BASETEN_USD` | no | Yatsu | `20`, `12`, `12` |
| `POST_CREDIT_CAP_META_USD` | no | Yatsu | `20` |
| `SAM_USD_PER_FRAME` | no | Yatsu | `0.01` (set Oct 31). Meta's Models page lists SAM 3.1 video at $0.20 per 1,000 frames, and one FILM call on a one-frame clip (`web/film/bridge_t032.5_cut.mp4`, 1 frame by ffprobe) reported `video_frames_processed: 50`, so a call bills 50 frames: $0.01 (VERIFIED Oct 31, one call). Unset: each FILM frame keeps its $0.50 reservation ($1.50 a run) |
| `JOB_EXPIRY_S` | no | optional | `120` |
| `STILLHERE_LIVE` | no | only as a switch | unset; `0` turns every station to its recorded run |
| `VERCEL` | no | Vercel | present on Vercel: live calls then require the shared store |

The question blocklist: the laptop-only list (`~/.config/stillhere/question_blocklist.txt`, salted hashes of terms from the
confidential clip) is outside the repository and outside the upload allowlist, so it is never uploaded. The server reads
`web/server_question_blocklist.txt`, a salt with no terms (the server holds nothing from that clip). If
`STILLHERE_QUESTION_BLOCKLIST` is missing or names a missing file, the egress fails closed: every typed request and every FILM concept
is refused before any model call ("This request can't be sent to the planner. ..." and "This concept can't be sent."), the example
chips included, while BUILD, FORECAST, EVIDENCE and the recorded runs keep working (VERIFIED by `tests/test_web_app.py`).

Without the Upstash variables on Vercel, every station shows its recorded run: serverless instances share no memory, so the caps, the
5-runs-per-hour limit and the jobs table would not hold (VERIFIED by tests).

## 8b. Accounts and secrets

1. Vercel CLI 65.8.0 or newer. The laptop's global CLI is 65.4.0 and crashes (segmentation fault) on Node 26.12.1; 65.8.0 runs
   (VERIFIED Oct 30): `sudo npm install -g vercel@latest`, then `vercel --version`.
2. A clean clone of the public repository at the pushed commit, so nothing untracked can be uploaded:
   `git clone https://github.com/Yatsuiii/stillhere.git stillhere-deploy && cd stillhere-deploy`
3. `vercel login` (Hobby account), then `vercel link` in the clone: a new project, root directory `./`, no Git connection (pushes
   must not deploy production; `vercel git disconnect` if it was connected), settings not customized (it detects Python).
4. Upstash: Vercel dashboard, the project, Storage, create an Upstash Redis database, connect it to Production and Preview. Then
   Settings, Environment Variables must list `KV_REST_API_URL` and `KV_REST_API_TOKEN` (ASSUMPTION about the exact names: if it lists
   `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` instead, that also works). Plan: pay-as-you-go with a $5 budget set in the
   Upstash console (changed Oct 31 from the free plan).
5. Secrets, each Sensitive, for Production and Preview, from a file or pipe (never `--value`):

   ```sh
   python3 -c 'import json,os,sys; sys.stdout.write(json.load(open(os.path.expanduser("~/.config/muse/auth.json")))["providers"]["meta"]["api_key"])' \
     | vercel env add MODEL_API_KEY --environment production,preview --sensitive
   tr -d '\n' < ~/.config/fireworks/api_key | vercel env add FIREWORKS_API_KEY --environment production,preview --sensitive
   tr -d '\n' < <the file holding the Baseten key that serves muse-spark-open-1026> \
     | vercel env add BASETEN_API_KEY --environment production,preview --sensitive
   ```

   The worker token. If the worker is installed, its token is already at `/etc/stillhere-worker/job_token` (root, 0600). If not,
   create it there first, in the installer's own format, without printing it; the installer keeps an existing token:

   ```sh
   sudo install -d -m 0700 -o root -g root /etc/stillhere-worker
   sudo sh -c 'umask 077; [ -s /etc/stillhere-worker/job_token ] || head -c 32 /dev/urandom | od -An -tx1 | tr -d " \n" > /etc/stillhere-worker/job_token'
   sudo cat /etc/stillhere-worker/job_token | vercel env add STILLHERE_WORKER_TOKEN --environment production,preview --sensitive
   ```

   (`docs/worker.md` shows `sudo cat` to the terminal; use the pipe above instead, so the token is never on screen or in a log.)
6. Non-secret variables, Production and Preview:

   ```sh
   for kv in "STILLHERE_QUESTION_BLOCKLIST=web/server_question_blocklist.txt" "STILLHERE_USAGE_LOG=/tmp/stillhere_usage.jsonl" \
             "WORKER_HOURS_ET=10:00-14:00 ET, Nov 4-9" "META_CREDIT_END_UTC=2026-11-05T02:00:00Z" \
             "CAP_META_USD=80" "CAP_FIREWORKS_USD=50" "CAP_BASETEN_USD=50" "DAILY_CAP_META_USD=20" "DAILY_CAP_FIREWORKS_USD=12" \
             "DAILY_CAP_BASETEN_USD=12" "POST_CREDIT_CAP_META_USD=20"; do
     printf '%s' "${kv#*=}" | vercel env add "${kv%%=*}" --environment production,preview --no-sensitive
   done
   vercel env ls        # names and environments only
   ```
7. Provider alerts (`web/README.md`, Money): the Meta Billing page's Spend Limit notification, Fireworks and Baseten spend alerts.

## 8c. Preview deploy and its smoke list

`vercel deploy` from the clone's root (a preview; it prints the URL). Preview URLs sit behind Vercel Authentication by default
(ASSUMPTION: Settings, Deployment Protection shows it on), so open them in the browser where you are logged in to Vercel.

- [ ] 1. The page loads at `/`, `/receipt.html?id=0000000000000000` says not found, `/api/status` shows live on for meta and
      fireworks; every response carries `X-Robots-Tag: noindex`.
- [ ] 2. PLAN on Spark 1.3, live once (the first example chip): the card, the story card of The Black Cat with its archive.org links,
      a receipt.
- [ ] 3. PLAN on Spark Open, live once: the receipt names its host.
- [ ] 4. BUILD on that plan: the Voice Transcribe checklist, live once, with its receipt (the input hash).
- [ ] 5. AUDIT once: "Advisory check, not measured".
- [ ] 6. FILM once ("face"): SAM 3.1 on the three frames, filled and same-size control, the CC BY credit.
- [ ] 7. An avoid request, for example "Plan an exploratory stock TRIBE forecast for healthy English-speaking adults on a story with
      no violence, intact versus sentences shuffled, in the language network.": the card line ends with "The chosen story's card
      lists none of them. Not a full review; listen before you use a story."
- [ ] 8. Each fallback with the keys blocked: `printf 0 | vercel env add STILLHERE_LIVE --environment preview --no-sensitive`,
      `vercel deploy` again, and every station shows its recorded run with "Live calls are off on this server."; then
      `vercel env rm STILLHERE_LIVE preview` and deploy again. (The failed-call path is covered by 8a run B.)
- [ ] 9. No key in any response or page source: DevTools, Network, search all responses for the first characters of each key: no
      match; View Source of `/` and `/app.js`: no key, no Authorization header in any page request.
- [ ] 10. The ledger counts in Upstash: the Upstash console's data browser shows `spend:meta:total`, `spend:fireworks:total` and
      `runs:<session>`, growing by the receipts' costs.
- [ ] 11. No request body in Vercel's logs: the deployment's Runtime Logs hold request lines only; search them for a word typed in
      step 7: no match.
- [ ] 12. One jobs-API round trip: the worker cannot pass Vercel Authentication, so run it against production in 8e (or switch
      Deployment Protection off for previews just for it, and back on after).

## 8d. Production

`vercel deploy --prod` from the same clone and commit. The URL stays unlisted until Nov 3: not in the README, the submission, the
video or any post before then; every response is `noindex`. Repeat checks 1, 2, 9, 10 and 11 on it.

## 8e. Worker reinstall and the round trip

1. `sudo bash worker/install_worker.sh https://<production host>` from the laptop's repository: it keeps
   `/etc/stillhere-worker/job_token`, the same value that is in Vercel as `STILLHERE_WORKER_TOKEN`.
2. `systemctl list-timers stillhere-worker.timer`: next start Wed Nov 4, 15:00 UTC.
3. The round trip, outside the hours, once (`docs/worker.md`): `echo STILLHERE_HOURS_ET=00:00-24:00 | sudo tee
   /etc/stillhere-worker/worker.env`, `sudo systemctl start stillhere-worker`; on the page, a live plan, BUILD, Reshuffle to a seed
   with no saved run, FORECAST: the lane bars move on the worker's events and the forecast lands with its receipt. Then
   `sudo systemctl stop stillhere-worker` and `sudo rm /etc/stillhere-worker/worker.env`.

## Redeploys

Git is not connected, so nothing deploys on a push. After any commit the hosted app needs (a results file from the Saturday chain
included: `results/story_wall.json`, `results/face_map.json`, `results/emofilm_film_score.json`), in `~/stillhere-deploy`:
`git pull --ff-only`, `vercel deploy` (a preview; check what changed through `vercel curl`), then `vercel deploy --prod`.

## Nov 3: listing the URL (ruling of Oct 30, 07:40 UTC, a), in this order

1. Make the repository public (Yatsu): GitHub, Yatsuiii/stillhere, Settings, Danger Zone, Change visibility, Public.
2. Open every github.com link on the hosted page (the PREREG.md link in Evidence, the write-up link) logged out, in a private
   window: each must load. `tests/test_pilot_page.py` checks that every repository path the page links to exists at HEAD.
3. Then list the URL: the README (its "Try it" line), the submission, the video's end card, and the repository's About website
   (`gh repo edit Yatsuiii/stillhere --homepage <url>`; the description and topics were set Oct 30).
4. The README's first-click links, in the badge row right after the tests badge: a "live demo" badge to the URL and a "video"
   badge to the uploaded video (add the video badge once it is uploaded). Judges landing on the repository click these first.
5. A picture of the page running at "A worked example": a screenshot (or a short GIF) of the stations from the production page,
   made after Sunday's filmed run; listed in `docs/BINARY_PROVENANCE.md`. Its caption is new wording: send it for review.

## What can block a deploy

- The laptop's Vercel CLI 65.4.0 crashes; update it (8b.1).
- Function time: Hobby functions stop at 300 s. FILM makes three SAM calls in one request; if SAM is slow the request ends in a 504
  and the page says so (its reservation stays counted).
- A new project's first `vercel deploy` goes to production, not a preview (Vercel's rule; on Oct 30 that deploy served only
  `not_found`, see Layout). Later ones are previews unless `--prod`.
- The Upstash variable names: 8b.4 (VERIFIED Oct 30 with the prefix `KV`: `KV_REST_API_URL`, `KV_REST_API_TOKEN`).
