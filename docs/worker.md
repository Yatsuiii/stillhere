# The live worker

The hosted page's FORECAST station runs TRIBE v2 on this laptop's GPU through a worker that pulls jobs over HTTPS (cloud plan v2,
item 5; sandbox ruling of Oct 29, 18:15 IST, item 6). The laptop accepts no inbound connection. The protocol is `docs/jobs_api.md`.
Every result is a forecast, not a scan: TRIBE v2 on the laptop, audio only, no brain data.

## What a job can ask for

A job is a recipe and nothing else: `{clip_id, condition, seed}` (`worker/recipe.py`).

- `clip_id`: one of the 12 clips in `data/live_clips.json`. Each is 40 s from 60 s of one chapter of the LibriVox recording of
  *Alice's Adventures in Wonderland* (public domain; readers named in the file). They are cut and transcribed by
  `worker/make_clips.py` (ffmpeg bitexact cut, local faster-whisper base.en word timings). alice_ch01-03 reproduce the intact
  SHA-256s in `stillhere/live_menu.json`.
- `condition`: intact, sentence_scramble, word_scramble or reversed.
- `seed`: an integer; it is stored as 0 for intact and reversed, which ignore it.

Before anything runs, the worker refuses:

- anything outside that schema;
- any clip not on the list, or a clip whose files do not match their SHA-256s;
- any job whose cache key it does not reproduce (`cache_key_mismatch`).

No audio, text or file ever comes in.

The audio for a recipe comes from `stillhere/conditions.build` on the committed clip, with the committed word table moved along with
the pieces. The BUILD station's Voice Transcribe check and the worker therefore read the same samples. The build is byte-identical
under numpy 2.2.6 and 2.3.5 (168 builds compared). Two edits are made to the word tables, both recorded in `make_clips.py`:
zero-duration words at a clip edge are dropped (they are not timed inside the clip), and the em dash character is written as a hyphen.

## What runs

| Part | Detail |
|---|---|
| Text lane | Stock TRIBE v2 with Llama 3.2 3B in float16, placed by accelerate across GPU and CPU. It reads the moved word table, not speech-to-text. Reversed sends no word to it. |
| Audio lane | w2v-BERT 2.0 on CPU. |
| Brain model | On CPU. |
| Video lane | V-JEPA 2 is off (no video). |

The 6 GB GPU cannot hold Llama 3.2's float16 layers and w2v-BERT at once, which is why the audio lane and brain model run on CPU.
With TRIBE's in-RAM feature cache off, its text extractor loads Llama 3.2 for each job (about 5 s from the page cache) and releases it
after the text lane. w2v-BERT and the brain model stay loaded.

The engine writes each build at a content-addressed path, because TRIBE's feature cache keys audio by file path. Without that, a
second job at the same path would silently reuse the first job's features. The development run hit this and it is now fixed.

Cause of the old mismatch: a cache-purge bug in the worker itself. The first version deleted each job's files from TRIBE's feature
cache (exca) after the job. exca keeps an in-process index into those memmap files, so every later job in the same process read wrong
features. That is why alice_ch02 intact once differed from the order-map run (r = 0.93 language, 0.83 hearing). It also made the first
48-entry cache wrong: 46 of 48 old entries were wrong, median max-abs error 0.40 (raw units). Its alice_ch02 intact reached r = 0.25
(language) and 0.69 (hearing) against the order-map forecast. That cache stays in the history of commit b62bb10; it was discarded and
recomputed. The fix: the feature cache is emptied when the engine loads and never touched while the process runs. It grows by about
36 MB a job. Past 2 GB (`STILLHERE_CACHE_LIMIT_GB`), `serve` exits with code 75, only between jobs (after posting the result), and
systemd (`Restart=on-failure`) starts a fresh worker with an empty cache.

The word table TRIBE reads (`<wav>.tsv`, through pandas) is written with csv quoting and read back before every run. Unquoted, a word starting with a double quote (dialogue in chapters 7, 8
and 12) opened a quoted field and silently swallowed the rows after it. The first precompute lost about half of alice_ch07's words
that way; those two entries were discarded and recomputed. The order-map clips (chapters 1-3) contain no such word, so the
order-map run is unaffected. `n_words_text_lane` is what reached TRIBE's text lane after its own sentence and missing-value steps,
and it can be one below the table: alice_ch06 word scramble gives 145 of 146.

Output is region values only:

- per second, the mean over the frozen proof's language (19 parcels) and hearing (7 parcels) sets of the parcel-mean forecast, in raw
  TRIBE units (`worker/regions.json`, from `stillhere/heldout.py` `parcel_sets`);
- words per second, the counts, timings and devices.

No vertex data, audio or transcript leaves.

`data/live_clips.json` `models` names every model, revision, precision and device, and is part of every cache key. The engine refuses
to start if what it loads differs.

### Repeatability (measured Oct 29, alice_ch02 intact, raw units; region SD about 0.12 language, 0.15 hearing)

| Pair | Language max abs diff, r | Hearing max abs diff, r |
|---|---|---|
| Two fresh processes, no other process on the GPU (same split: 22 Llama modules on GPU, 9 on CPU) | 0.00037, 1.000000 | 0.00027, 1.000000 |
| After another job in the same process vs a fresh process | 0.00013, 1.000000 | 0.00029, 1.000000 |
| Another process holding 1.5 GB of VRAM (split moves to 15 on GPU, 16 on CPU) vs idle | 0.00014, 1.000000 | 0.00017, 1.000000 |
| Fork E's order-map forecast vs a fresh worker run | 0.00026, 1.000000 | 0.00023, 1.000000 |

The device split is not a material cause. It depends on free GPU memory, but moving it changes the forecast only at float16
run-to-run level, so it is recorded, not pinned. Every result carries `llama_placement` (module counts and a hash of the device map).
It is kept out of `models` and the cache key, so free memory cannot change a key. Fork E's numbers fall inside the run-to-run spread.
The earlier r = 0.93 / 0.83 came from the cache-purge bug above, not from the model or the GPU/CPU split.

A live run of a cached recipe therefore reproduces the cached line within 0.002 max abs difference and r >= 0.9999, the tolerance of
the opt-in GPU test `tests/test_worker_gpu.py` (`STILLHERE_GPU_TESTS=1`). That test runs a recipe after another job in the same
process.

## Timings (warm engine, 4050 laptop)

| Job | Time | Breakdown |
|---|---|---|
| Intact or scrambled | 35 to 56 s (median 43 intact, 45 sentence, 41 word scramble) | text lane about 20 to 25 s; audio lane on CPU about 15 to 25 s; predict 2 to 4 s |
| Reversed | 14 to 15 s | no text lane |

The page's job expiry is 120 s. `data/forecast_cache.json` holds the 12 clips x 4 conditions at the default seed, computed by
`python worker/tribe_worker.py precompute` with the same engine (version 2, in one process, Oct 29 16:39-17:07 UTC).

## The sandbox

`worker/stillhere-worker.service.in`, rendered by the install script into `stillhere-worker.service` and
`stillhere-worker-selftest.service`:

| Lines | Effect |
|---|---|
| `DynamicUser=yes`, `NoNewPrivileges=yes`, `CapabilityBoundingSet=` | A throwaway user with no capabilities. |
| `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes` | Read-only system, no home directories: the confidential clip folder, every key and the Kaggle settings live under /home. |
| `TemporaryFileSystem=/run/media` | Hides the whole Windows partition. It is mounted for every local user with every file mode 777, so a separate user alone would not keep the worker out of ds003171. |
| `BindReadOnlyPaths=` | Exactly three folders from that partition, read-only, at new paths under /opt, so nothing under /run/media can be listed: the worker's own model folder (`work/.worker-models`, hard links to TRIBE v2, w2v-BERT 2.0 and Llama 3.2 3B only, built by `worker/link_models.py`), the TRIBE site-packages, and the spaCy English model. |
| `StateDirectory=stillhere-worker` | Its only writable place (feature cache, job scratch). |
| `LoadCredential=job_token:...` | The job token, its only secret. |

The interpreter and the worker's code and clips are copied root-owned into /opt/stillhere-worker. The unit also sets
`HF_HUB_OFFLINE=1` (no model download, no Hugging Face token) and `MemoryMax=11G`. `systemd-analyze security --offline` scores it
3.1 ("OK").

The audit runs before anything is bound (`worker/link_models.py --audit`). It requires:

- the model folder holds 16 files, all weights, configs or tokenizers;
- site-packages and spaCy hold no audio, video or brain-imaging file except the 135 that published wheels list in their RECORD
  (nibabel's and nilearn's test templates);
- no path carries a private-data mark;
- no symlink leaves a folder.

Today's audit: models 16 files (9.47 GB); site-packages 36,801 files, 135 wheel-shipped media files, 0 other; spaCy 41 files; 0
marks; 0 escaping links.

The self-test is `tribe_worker.py self-test`, run inside the same sandbox by the install script. It passes only if:

- listing the ds003171 folder, the work folder, the partition root, the user's home and /home fails;
- writing into each bound folder fails;
- the GPU is visible.

A user-level rehearsal (a transient `systemd-run --user` unit with the same TemporaryFileSystem, ProtectHome, ProtectSystem and
read-only binds, but no DynamicUser, which needs root) passed every check on Oct 29: FileNotFoundError on /run/media paths,
PermissionError on /home, read-only binds, GPU visible. The root install reruns the self-test for real and refuses to finish if it
fails.

## Worker hours

10:00-14:00 ET, Nov 4-9 (EST after Nov 1, so 15:00-19:00 UTC, 20:30-00:30 IST; cloud ruling of Oct 29, 17:35 UTC, item 8).

- `stillhere-worker.timer` starts the worker at `OnCalendar=2026-11-04..09 10:00:00 America/New_York` with `Persistent=true`. A start
  missed while the laptop was off runs at the next boot, and the worker exits at once if that is after 14:00 ET.
- The installer enables only the timer. The service has no `[Install]` section.
- `serve` checks the hours (`STILLHERE_HOURS_ET`, default `10:00-14:00`, America/New_York wall clock) before it loads anything and
  before every poll. Once the hours are over and no job is running, it exits 0. Under `Restart=on-failure`, exit 0 stays down until the
  next timer start, while the cache-limit exit (75) and crashes restart. `RuntimeMaxSec=` is not used: it would end a job mid-run.
- No suspend during the hours: the worker unit pulls in `stillhere-worker-awake.service` (`Wants=`), which is bound to it (`BindsTo=`)
  and stops with it, including at 14:00 ET. That unit runs as root only
  `systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block sleep infinity`, so logind refuses suspend, idle sleep and the lid
  switch while the worker runs. A root unit holds the inhibitor because the sandboxed worker (DynamicUser, no capabilities) may not be
  allowed one by polkit.
- Outside the hours (for example to film a live job on Sunday), Yatsu writes the override, starts the worker, and removes the file
  afterwards: `echo STILLHERE_HOURS_ET=00:00-24:00 | sudo tee /etc/stillhere-worker/worker.env`, then
  `sudo systemctl start stillhere-worker`, then `sudo rm /etc/stillhere-worker/worker.env`.

Outside the hours, the page serves saved forecasts (the page's line is fork A's).

## What Yatsu runs

Order (cloud ruling of Oct 29, 17:35 UTC, item 9): this engine fix, the timer and fork A's labels land first. Then Yatsu reinstalls once,
because any /opt copy installed earlier still holds the old engine. The installer replaces /opt/stillhere-worker, keeps the existing job
token, reruns the self-test and enables only the timer.

```sh
sudo bash worker/install_worker.sh https://<page host>    # checks, audit, copies, token, units, self-test, enables the timer only
sudo cat /etc/stillhere-worker/job_token                    # paste into the page's server secrets as STILLHERE_WORKER_TOKEN
systemctl list-timers stillhere-worker.timer                # next start: Wed Nov 4, 15:00 UTC
sudo systemctl start stillhere-worker                       # by hand inside the hours; stop with: sudo systemctl stop stillhere-worker
sudo bash worker/uninstall_worker.sh                        # removes the timer, the units, /opt/stillhere-worker, the token and the state
```
