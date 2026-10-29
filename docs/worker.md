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
second job at the same path would silently reuse the first job's features. The development run hit this and it is now fixed. The
engine also deletes each job's cached features afterwards.

Output is region values only:

- per second, the mean over the frozen proof's language (19 parcels) and hearing (7 parcels) sets of the parcel-mean forecast, in raw
  TRIBE units (`worker/regions.json`, from `stillhere/heldout.py` `parcel_sets`);
- words per second, the counts, timings and devices.

No vertex data, audio or transcript leaves.

`data/live_clips.json` `models` names every model, revision, precision and device, and is part of every cache key. The engine refuses
to start if what it loads differs.

Check against the order-map run, which used the same clip, a GPU audio lane and a GPU brain model: the worker's reversed alice_ch02
forecast matches it exactly (r = 1.0000, max difference 0.0000 in both regions). The intact forecast differs (language r = 0.93,
hearing r = 0.83). The audio and the word table are identical, so the difference is in the text lane. The most likely cause is that
accelerate places a different share of Llama's float16 layers on the CPU when the brain model is not on the GPU. That placement
depends on free GPU memory when Llama loads, so the cache and live jobs are only comparable when the GPU is otherwise idle.

## Timings (warm engine, 4050 laptop)

| Job | Time | Breakdown |
|---|---|---|
| Intact or scrambled | about 47 to 73 s | text lane 20 to 35 s; audio lane on CPU 23 to 39 s; predict 2 to 4 s |
| Reversed | about 25 to 39 s | no text lane |

The page's job expiry is 120 s. `data/forecast_cache.json` holds the 12 clips x 4 conditions at the default seed, computed by
`python worker/tribe_worker.py precompute` with the same engine.

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

## What Yatsu runs

```sh
sudo bash worker/install_worker.sh https://<page host>    # checks, audit, copies, token, units, self-test; starts nothing
sudo cat /etc/stillhere-worker/job_token                    # paste into the page's server secrets as STILLHERE_WORKER_TOKEN
sudo systemctl start stillhere-worker                       # during worker hours; stop with: sudo systemctl stop stillhere-worker
sudo bash worker/uninstall_worker.sh                        # removes the units, /opt/stillhere-worker, the token and the state
```

Worker hours (ET) are Yatsu's decision: `{worker_hours_et}`. The proposal is 10:00-14:00 ET daily, Nov 4-9. Outside the hours, the
page serves the cache.
