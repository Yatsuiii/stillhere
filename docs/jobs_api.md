# Jobs API between the hosted page and the laptop worker (agreed Oct 29 by the web and worker forks)

The page's backend (`web/`) keeps a jobs table; the laptop worker pulls jobs over HTTPS, so the laptop accepts no inbound
connection. One job runs at a time. Nothing here carries audio, text, keys or any file from the laptop; only the recipe goes out and
only region values, stage events and timings come back.

## Recipe (the only thing a job can ask for)

```json
{"clip_id": "alice_ch01", "condition": "sentence_scramble", "seed": 20261028}
```

- `clip_id`: one of the ids in `stillhere/live_menu.json` (the planner's live menu: `alice_ch01` to `alice_ch12`, 40 s from 60 s of
  each LibriVox Alice chapter). The server validates recipes against it. No other clip, no uploaded audio or text.
  `data/live_clips.json` (worker fork) holds only what the worker needs per clip (`wav_sha256`, `words_sha256`, `seconds`) and the
  model manifest; a test checks its ids and intact SHA-256s against `live_menu.json`.
- `condition`: `intact`, `sentence_scramble`, `word_scramble` or `reversed` (`stillhere/conditions.py` names).
- `seed`: integer 0 to 2^31 - 1. `intact` and `reversed` ignore it (the server stores it as 0 for them).
- `cache_key`: SHA-256 of the canonical JSON `{"clip_id", "condition", "seed", "models", "wav_sha256", "words_sha256"}`
  (`sort_keys=True`, separators `(",", ":")`), with `models`, `wav_sha256` and `words_sha256` read from `data/live_clips.json`, so a
  changed clip, word table or model changes the key. The server computes it; the worker re-derives it and fails the job with
  `cache_key_mismatch` if it differs.
- The condition audio is always built by `worker/recipe.py` `build_condition(clip_id, condition, seed)` (which calls
  `stillhere/conditions.build` with the clip's committed word table). The BUILD station calls the same function, so its Voice
  Transcribe check reads the same audio TRIBE forecasts.

## Auth

The worker sends `Authorization: Bearer <job token>` on every worker call. The server holds the token as the env var
`STILLHERE_WORKER_TOKEN` and compares in constant time; a wrong or missing token gets 401 and no body. The token is the worker's
only secret. Page calls never see it.

## Worker calls (all POST, JSON in and out, `Content-Type: application/json` or 415)

| Call | Body | Answer |
|---|---|---|
| `/api/worker/claim` | `{"worker": "laptop-4050", "models": {...}}` | `200 {"job": {"id", "recipe", "cache_key", "claimed_utc", "deadline_s": 40}}` (`deadline_s` is the audio cap in seconds), or `204` when no job waits. Every claim also counts as a heartbeat (`worker_last_seen`, 30 s TTL); the page shows the worker online only within it. The worker polls every 2 s, during worker hours only. |
| `/api/worker/jobs/{id}/event` | `{"stage", "lane", "status", "t_utc", "detail"}` | `204` |
| `/api/worker/jobs/{id}/result` | the result object below | `204`; `409` if the job expired or was already finished |
| `/api/worker/jobs/{id}/fail` | `{"reason": "<code>"}`, codes: `timeout`, `oom`, `model_error`, `bad_recipe`, `cache_key_mismatch`, `busy` | `204` |

Events drive the page's lane bars, so the worker posts them as the stages really happen, never after the fact. Order per job:
`build`, `video_lane` off, `text_lane` start and end, `audio_lane` start and end, `predict` start and end, `regions` start and end, then
the result (`upload` is the server's side).

- `stage`: `load`, `build` (the condition audio from the clip and seed), `text_lane`, `audio_lane`, `video_lane`, `predict`,
  `regions`, `upload`. No speech-to-text runs on the worker: word timings are precomputed per clip and committed, and the text lane
  reads the word table moved with the condition.
- `lane`: `w2v-bert` (audio lane), `llama-3.2` (text lane), `v-jepa-2` (video lane) or null.
- `status`: `start`, `end` or `off` (`v-jepa-2` is always `off` on audio).
- `detail` (optional, small): `{"n_words": int}` on the text lane (0 on reversed), `{"precision": "float16, accelerate GPU/CPU split"}`
  for Llama.

## Result object

```json
{"cache_key": "...", "tr_s": 1.0, "seconds": 40,
 "regions": {"language": [..], "hearing": [..]},
 "word_rate": [..], "n_words_text_lane": 117,
 "models": {"tribe": "facebook/tribev2", "w2v_bert": "...", "llama": "meta-llama/Llama-3.2-3B",
            "precision": "float16, accelerate GPU/CPU split", "v_jepa_2": "off"},
 "input_sha256": "<sha256 of the built condition wav>", "peak_vram_gb": 3.5, "worker_version": "...",
 "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU", "timings_s": {"load": 0.9, "build": 0.2, "predict": 2.0, "total": 9.4},
 "computed_utc": "2026-10-30T14:02:11Z"}
```

- `regions`: per second of clip time, the mean over the frozen proof's language parcels and hearing parcels (`stillhere/heldout.py`
  `parcel_sets`) of TRIBE's prediction, raw units. At most 60 points per region, finite numbers only; the server rejects anything
  else with 422.
- `word_rate`: words per second (word onsets in 1 s bins) from the committed word table moved with the condition, same length as
  the regions.
- `input_sha256`, `peak_vram_gb`, `worker_version`: for the receipt.
- No vertex data, no audio, no transcript text.

## Page calls (no token)

| Call | Body | Answer |
|---|---|---|
| `POST /api/forecast` | `{run_id, clip_id, condition, seed}` | `200 {"state": "cached", "result"}` when the cache key is known; `202 {"state": "queued", "job_id"}` when the worker is online; `200 {"state": "offline", "why", "fallback"}` when it is not (no job is queued); `200 {"state": "busy", "job_id"}` while this session already has a job for this same recipe (the page follows it), or `200 {"state": "busy", "why", "fallback"}` when that job is for another recipe; `200 {"state": "busy", "why", "fallback"}` when 6 jobs already wait (no job is queued); a session past its runs, a scramble seed BUILD does not offer, a run whose live plan named another clip or condition or was declined, or a station past its hourly cap gets the saved run only |
| `GET /api/forecast/{job_id}` | | `{"status": "queued" or "running" or "done" or "failed" or "expired", "events": [...], "result"}`, with `"fallback"` and a `"why"` line when failed or expired |

## Timeouts and cache

- A claimed job with no result `JOB_EXPIRY_S` = 120 s after its claim is `expired` (measured on the 4050 with a warm engine: reversed
  about 39 s, intact or scrambled 47 to 56 s; the audio lane runs on CPU while Llama 3.2 stays on the GPU). A queued job also expires
  after `JOB_EXPIRY_S` when no worker is online.
- Every fallback (offline, failed, expired) is the saved run for that clip and condition at seed 20261028, labelled "Cached forecast,
  computed <UTC>"; when the visitor asked for another seed it adds "No saved run for seed {seed}; shown: the saved run at seed 20261028."
- Results are kept by `cache_key` (no TTL). `data/forecast_cache.json` (committed by the worker fork) holds the 12 clips x 4 conditions
  at the default seed, precomputed, in the result format above; the server loads it at start.
- One queued or running job per session; at most 6 jobs waiting in all, 2 of them from one network address (one atomic count: added on submit, removed when the worker claims the job or it expires); at most 5 runs per sliding hour per session and 20 per
  network address (shared with the other live stations); FORECAST's own hourly cap is 30 jobs for all visitors together, counting only
  jobs that reach the queue: a cached, busy, offline or refused forecast gives back both the hour and the run's use.

## Worker hours

10:00-14:00 ET, Nov 4-9 (15:00-19:00 UTC) (ruling of Oct 29, 17:35 UTC, item 8). Published on the page and in the README in ET.
Outside them the page says the worker is offline and serves the saved runs.
