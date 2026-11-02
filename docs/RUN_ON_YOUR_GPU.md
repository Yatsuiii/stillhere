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
| 4. Paranoia end to end | as route 2 (we used 16 GB T4s) | route 2 plus 2.6 GB of public data | as route 2 |

- Python 3.12 for the page and the scorers (`requirements-dev.txt`), Python 3.11 for TRIBE v2 (`requirements-tribe.txt`, the laptop
  worker's environment as installed: tribev2 at af58661, torch 2.6.0 with CUDA 12.4). tribev2 needs torch below 2.7, so RTX 50-series
  GPUs, which need a newer CUDA build, are likely unsupported. Your NVIDIA driver must support CUDA 12.4.
- Llama 3.2 3B is gated: accept Meta's licence at https://huggingface.co/meta-llama/Llama-3.2-3B, then log in with `hf auth login`.
  The setup script checks this before it downloads anything and stops with one sentence if access is missing.
- RAM: the live worker runs in an 11 GB memory cap on a 14 GB laptop. The Paranoia forecasts did not fit that laptop's RAM; we made
  them on Kaggle T4s.

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
$STILLHERE_TRIBE_PYTHON worker/tribe_worker.py verify
$STILLHERE_TRIBE_PYTHON worker/tribe_worker.py run --clip alice_ch01 --condition word_scramble --seed 7 --out my_forecast.json
```

The setup script pins the three models to the revisions the worker was built on (`worker/link_models.py`: facebook/tribev2 f894e78,
facebook/w2v-bert-2.0 da985ba, meta-llama/Llama-3.2-3B 13afe51) and runs everything after the download offline.

`verify` runs one cached recipe per condition (intact, sentence scramble, word scramble, reversed) in one process and prints a table:
each region's largest absolute difference from our saved forecast and the correlation, against the tolerance our own repeats set
(largest difference 0.002 in raw units, r 0.9999; `tests/test_worker_gpu.py`). It exits 1 if any row is out of tolerance. `run`
computes one recipe in the foreground (no network, no hours, no token), writes the region values with `--out`, and compares them with
our saved forecast when that recipe is in `data/forecast_cache.json`. Only one GPU is used (`CUDA_VISIBLE_DEVICES=0` unless you set it).

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
bash scripts/reproduce_paranoia.sh --dry-run ~/paranoia      # sizes only
bash scripts/reproduce_paranoia.sh ~/paranoia                # data, your forecasts, then both scorers on them
bash scripts/reproduce_paranoia.sh --no-gpu ~/paranoia       # no GPU: re-score our released forecasts from the public scans
```

It fetches the ds001338 fMRIPrep fsaverage5 derivative (132 GIFTIs, 2.5 GB), the three recordings and the atlas, makes the 9 stock
forecasts with `r2/run_r2_forecast.py`, then scores them with `scripts/r2_paranoia.py --verify-own` and
`scripts/r5_short_scan.py --verify-own`. Every downloaded file must equal the one our scoring read (`results/r2_bold_SHA256.json`,
`results/r2_paranoia_SHA256.json`); your forecasts are hashed but not held to our pins; nothing is written into this repository, and
your numbers print beside ours (volunteer-equivalent, lower bound, share, baselines; R5's k of 22 and p). Our forecasts ran on Kaggle
T4s with every model on the GPU; `run_r2_forecast.py` keeps the sound and brain models on the CPU, so small numeric differences are
possible. Our own scoring took 69 minutes on a Kaggle CPU (R5 another 2), after a 4-minute download.

## What cannot be reproduced here

- Le Petit Prince section 8, the notthefall intact-vs-shuffled test (D20), the story picker and wall, and the film face test: they ran
  in Kaggle notebooks and the pre-event harness, not from this repository's scripts.
- The fine-tuned TRIBE v2 line: its weights are not public.
