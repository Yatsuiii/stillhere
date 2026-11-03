# Run StillHere on your own machine

For judges and labs who want to check our numbers without our laptop. Four routes, from no GPU to Paranoia end to end. Research use;
healthy volunteers and public data only. Not a diagnosis or medical advice.

TRIBE v2 (Meta FAIR, CC BY-NC 4.0) is not in this repository: the setup script downloads it from Hugging Face at the pinned revision.

## What you need

| Route | GPU | Disk | Keys |
|---|---|---|---|
| 1. The page, no GPU | none | the repository | none |
| 2. One forecast and `verify` | NVIDIA, 6 GB or more | models 9.47 GB, TRIBE environment about 7 GB | a Hugging Face login with Llama 3.2 access |
| 3. The page on your GPU | as route 2 | as route 2 | as route 2 |
| 4a. Paranoia with your forecasts | as route 2 (we used 16 GB T4s) | route 2 plus 2.6 GB of public data | as route 2 |
| 4b. Paranoia re-scored (`--no-gpu`) | none | 2.9 GB: the public scans and our released forecasts | none |

- Python 3.12 for the page (`requirements-dev.txt`) and the scorers (`requirements-score.txt`, the R2 and R5 scoring kernel's pins),
  Python 3.11 for TRIBE v2 (`requirements-tribe.txt`, the laptop
  worker's environment as installed: tribev2 at af58661, torch 2.6.0 with CUDA 12.4). tribev2 needs torch below 2.7, so RTX 50-series
  GPUs, which need a newer CUDA build, are likely unsupported. Your NVIDIA driver must support CUDA 12.4.
- Llama 3.2 3B is gated: accept Meta's licence at https://huggingface.co/meta-llama/Llama-3.2-3B, then log in with `hf auth login`.
  The setup script checks this before it downloads any model (the TRIBE packages install first) and stops with one sentence if
  access is missing.
- RAM: the live worker runs in an 11 GB memory cap on a 14 GB laptop. Route 4a's forecasts need more: on that laptop, with its
  desktop running, story 1's three stages and story 2's audio-only and full stages finished (story 1's full stage peaked at about
  9.7 GB), but story 2's shuffled stage was killed out of memory at about 10 GB of RAM plus 4 GB of swap. Use a machine with 32 GB
  of RAM for route 4a; we made our forecasts on Kaggle T4s (about 30 GB). Route 4b's scoring peaked at about 5.7 GB.

## 1. The page, no GPU

```sh
python web/serve.py          # http://127.0.0.1:8780/
```

With no keys, every station shows its recorded run and FORECAST shows our saved forecasts (`data/forecast_cache.json`). Nothing is
computed on your machine.

## 2. One forecast and `verify` on your GPU

```sh
bash scripts/judge_gpu_setup.sh ~/stillhere-models     # a venv (.venv-tribe), the 3 models (about 9.5 GB), then an audit
export STILLHERE_TRIBE_PYTHON=$PWD/.venv-tribe/bin/python STILLHERE_MODELS_DIR=~/stillhere-models/worker   # as the script prints
$STILLHERE_TRIBE_PYTHON worker/tribe_worker.py verify --out verify.json     # the table, plus the GPU, driver and versions it ran on
$STILLHERE_TRIBE_PYTHON worker/tribe_worker.py run --clip alice_ch01 --condition word_scramble --seed 7 --out my_forecast.json
```

The setup script pins the three models to the revisions the worker was built on (`worker/link_models.py`: facebook/tribev2 f894e78,
facebook/w2v-bert-2.0 da985ba, meta-llama/Llama-3.2-3B 13afe51) and runs everything after the download offline.

`verify` runs one cached recipe per condition (intact, sentence scramble, word scramble, reversed) in one process and prints a table:
each region's largest absolute difference from our saved forecast and the correlation, against the tolerance our own repeats set
(largest difference 0.002 in raw units, r 0.9999; `tests/test_worker_gpu.py`). It exits 1 if any row is out of tolerance. `run`
computes one recipe in the foreground (no network, no hours, no token), writes the region values with `--out`, and compares them with
our saved forecast when that recipe is in `data/forecast_cache.json`. Only one GPU is used (`CUDA_VISIBLE_DEVICES=0` unless you set it).
The tolerance comes from repeats on our RTX 4050; `verify` has not run on any other GPU, so a small excess on another card may be the
hardware, not the setup.

On our 6 GB RTX 4050 laptop GPU a job took 35 to 56 s (intact or scrambled) and 14 to 15 s (reversed), with a peak of about 5.3 GB.

## 3. The page on your GPU

```sh
bash scripts/run_local_gpu.sh        # needs STILLHERE_TRIBE_PYTHON and STILLHERE_MODELS_DIR from route 2
```

It starts the page on http://127.0.0.1:8780/ and the worker against it, at any hour, with a fresh job token in a 0600 file under
`~/.local/state/stillhere-local-gpu` (never printed). With `STILLHERE_LOCAL_GPU=1`, which the script sets and the hosted server
refuses, FORECAST runs live on a recorded plan, so no API key is needed: pick the default study, PLAN shows its recorded plan, then
FORECAST runs on your GPU (Reshuffle picks another offered seed). The page names the worker "local GPU" (`STILLHERE_WORKER_LABEL`).

## 4. Paranoia end to end (R2 and R5)

```sh
python3 -m venv .venv-score && .venv-score/bin/pip install -r requirements-score.txt
export STILLHERE_SCORE_PYTHON=$PWD/.venv-score/bin/python   # checked before anything is downloaded
bash scripts/reproduce_paranoia.sh --dry-run ~/paranoia      # sizes only
bash scripts/reproduce_paranoia.sh ~/paranoia                # 4a: data, your forecasts (route 2's setup), then both scorers on them
bash scripts/reproduce_paranoia.sh --no-gpu ~/paranoia       # 4b: no GPU, no models, no login: re-score our released forecasts
```

It fetches the ds001338 fMRIPrep fsaverage5 derivative (132 GIFTIs, 2.5 GB), the three recordings and the atlas, makes the 9 stock
forecasts with `r2/run_r2_forecast.py`, then scores them with `scripts/r2_paranoia.py --verify-own` and
`scripts/r5_short_scan.py --verify-own`. Every downloaded file must equal the one our scoring read (`results/r2_bold_SHA256.json`,
`results/r2_paranoia_SHA256.json`); your forecasts are hashed but not held to our pins; nothing is written into this repository, and
your numbers print beside ours (volunteer-equivalent, lower bound, share, baselines; R5's k of 22 and p). Our forecasts ran on Kaggle
T4s with every model on the GPU; `run_r2_forecast.py` keeps the sound and brain models on the CPU, so small numeric differences are
possible. Our own scoring took 69 minutes on a Kaggle CPU (R5 another 2), after a 4-minute download.

## What we ran ourselves, on the laptop (Nov 2-3 2026)

From a fresh clone in a new folder, with a new venv and a new models folder (linked from an existing download):

- Route 2: `judge_gpu_setup.sh` (STILLHERE_HF_SOURCE set): 16 model files, 9.47 GB, audit clean, in 15 s. `verify` on the RTX 4050
  Laptop GPU: PASS for all four conditions, largest difference at most 0.0002 against our saved forecasts, r 1.00000; jobs 52 to 61 s
  (reversed 13 s), peak 5.29 GB. A second run with `--out`: PASS again, largest difference 0.0003; its file, to compare yours with,
  is `docs/proof/verify_rtx4050_laptop.json` (driver 615.78.08, CUDA 12.4, torch 2.6.0, tribev2 at af58661). On Nov 3,
  `judge_gpu_setup.sh` installed `requirements-tribe.txt` from scratch into a new venv (about 3 GB of wheels; two attempts hit
  network timeouts, after which the script's longer timeout and resumable venv carried the third run through in 20 minutes): torch 2.6.0+cu124 saw the GPU, the models' audit was clean, and `verify` on that venv passed
  again (largest difference 0.0004, r 1.00000, jobs 47 to 53 s).
- Route 3: `run_local_gpu.sh` with an empty HOME and no keys: "Live models: off", the default plan's recorded run opened FORECAST,
  "Running on local GPU", then "Forecast, live" on the RTX 4050 in 46 s; the job token was 0600 in a 0700 folder. Again after the
  review fixes (8eca1d3), from another fresh clone: "Forecast, live" in 54 s under the chip "Forecast only · audio-only, your GPU"
  and the line "Local GPU mode: every forecast runs on your GPU; saved forecasts are not used."; a request with another Host got 403.
- Route 4b (`reproduce_paranoia.sh --no-gpu`, from a fresh clone, a fresh `requirements-score.txt` venv and our released forecasts
  read from a local copy of the release while the repository was private): all 135 downloaded files (132 GIFTIs, 3 events files)
  equal our download, and both scorers gave exactly our numbers: R2 about 3 volunteers, lower bound 1.88, at least 1, share 59%,
  language z 4.01, baselines beaten 4 of 4; R5 17 of 22, p = 0.0004, word rate 8 of 22, p = 0.94, PASS. Scoring took about 17 minutes
  on the laptop's CPU. The download ran across two connections (at about 160 KB/s, then about 3 MB/s); one dropped transfer resumed
  where it stopped.
- Route 4a (the same steps with the forecasts made on the RTX 4050): 5 of the 9 stages finished (story 1's three, story 2's
  audio-only and full; 282 to 1,786 s each); story 2's shuffled stage was killed out of memory (see RAM above), so route 4a was not
  scored on the laptop.

## What cannot be reproduced here

Only Paranoia (R2 and R5) and the live worker reproduce here.

- Le Petit Prince section 8, the Alice held-out line, the notthefall intact-vs-shuffled test (D20), the story picker and wall, R3, the
  intact film and the film face test: their inputs (pre-event caches, Kaggle kernel outputs, the film edits) are not published, and the
  face test's scorer refuses a second run.
- The fine-tuned TRIBE v2 line: its weights are not public.
