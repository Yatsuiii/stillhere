"""R2 (docs/rules/r2_paranoia_rule.md, section 3): the 18 Paranoia forecasts, stock TRIBE v2 first, then fine-tuned TRIBE v2, before any
Paranoia brain data is downloaded or read. Per story (story1, story2, story3, each heard in its own run): audio-only, full, and full with
the word texts shuffled across their onsets (seed 7).

The harness is the Alice runner's (alice/forecast/run_alice_forecast.py, imported unchanged, as R3 did): k2/tribe_common.py at tribev2
af58661, build_events in the training order (words from the whole audio first, then chunking), stitched 50 s windows, the text lane in
fp16 through accelerate, the sound extractor and TRIBE's brain model on the CPU and Llama 3.2's layers on a fixed 3 GiB GPU budget. The
fine-tuned weights load through R1's loader (alice/forecast/run_alice_forecast_tuned.py apply_tuned: both pins, strict, at least one
tensor changed, eval mode).

Words: the dataset has a full text but no word timings, so they come from TRIBE v2's own speech-to-text (ExtractWordsFromAudio, as
tribev2 af58661 ships it: uvx whisperx large-v3), run once per story inside tribe_common.build_events by the stock full stage, which
writes stim.tsv into its work folder. That file is the story's word table: copied to OUT/words_<story>.tsv, it feeds the full forecasts
(read back by ExtractWordsFromAudio, so no second speech-to-text pass), the shuffled forecasts (tribe_common.write_tsv with seed 7, the
section 8 control), the word-rate regressor and every word baseline. Only text, start and duration reach TRIBE: AddContextToWords runs
with sentence_only=False and AddSentenceToWords rebuilds sentences from the text, so the table's other columns change no forecast. Checks
on every text stage, recorded and required: no block of words repeated at one exact shift (the film kernels' test), the last word
before the audio's end; descriptive only: the word count against the dataset's full text (counts, never text).

Work folders are per story and mode and shared by the stock and fine-tuned stages: the sound extractor's cache key is the stage's
stim.wav path, so a fine-tuned stage reads every sound and Llama feature its stock stage computed (checked: no file added to the
cache, work folder unchanged, forecast different from stock; the fine-tune changed only TRIBE's brain model).
Writes pred_<stage>.npz, stage_<stage>.json and the word tables to OUT; prints counts, shapes and timings, never word text.
Run from the analysis tree's copy of this file (r2/, beside alice/ and k2/); the launcher keeps the tribe environment out of PYTHONPATH
(R2_SITE) so the speech-to-text subprocess uses only its own packages.
usage: python run_r2_forecast.py STAGE   (STAGE: {stock|tuned}_story{1|2|3}_{audio|full|shuf}; all nine stock stages first, and
       within a story audio, full, then shuf)"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path[:0] = [p for p in os.environ.pop("R2_SITE", "").split(":") if p]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                              # the analysis root (work/stillhere)
R2 = ROOT / "r2"
STIM = R2 / "stimuli"
STIM_SHA = {"story1": "e0a674401cd64580aeb50327af4393332f387df87ddbf0a65d68a96eb55a6c3e",   # ds001338 v1.0.0 stimuli/, what the
            "story2": "e8b784671a748ad03f91227d1152d6219d51f713a46d6d6e41c63c17deb98e44",   # listeners heard
            "story3": "a739113b62f1aef92a2ead0fa069d97c4e05e1478123d26f42d223b05f970034"}
FULLTEXT, FULLTEXT_SHA = STIM / "story_fulltext.txt", "c762b04ed6a1141391c9537b9eddb3e77bb18824d1adff968e300d7d70aae82c"
OUT = R2 / "forecast/out"; WORK = R2 / "forecast/tmp"; CACHE = R2 / "forecast/cache"
STORIES, MODES, MODELS = ("story1", "story2", "story3"), ("audio", "full", "shuf"), ("stock", "tuned")
STAGES = [f"{m}_{s}_{k}" for m in MODELS for s in STORIES for k in MODES]
SHUFFLE_SEED = 7
FEATS = {"audio": ["audio"], "full": ["text", "audio", "video"], "shuf": ["text", "audio", "video"]}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def parse(stage) -> tuple:
    m = re.fullmatch(r"(stock|tuned)_(story[123])_(audio|full|shuf)", stage)
    if not m:
        raise SystemExit(f"unknown stage {stage!r}; one of {STAGES}")
    return m.groups()


def missing_before(stage, done) -> list:
    """The stages the order puts before `stage` that have no stage file yet: every stock stage before any fine-tuned one; within a
    model and story, audio, full, then shuf (shuf reads the word table the stock full stage writes)."""
    model, story, mode = parse(stage)
    need = [f"stock_{s}_{k}" for s in STORIES for k in MODES] if model == "tuned" else []
    need += [f"{model}_{story}_{k}" for k in MODES[:MODES.index(mode)]]
    return [s for s in need if s not in done]


def words_table(path) -> list:
    """(text, start, duration) rows of a word table in ExtractWordsFromAudio's format (tab separated, pandas quoting)."""
    with open(path, newline="") as f:
        return [(r["text"], float(r["start"]), float(r["duration"])) for r in csv.DictReader(f, delimiter="\t")]


def shift_copies(texts, starts) -> dict:
    """The film kernels' check for the over-60 s duplication: the most common exact shift (ms) between repeats of the same word; a block
    is flagged when that shift recurs at least max(5, 10% of the words) times."""
    by = {}
    for t, s in zip(texts, starts):
        by.setdefault(t, []).append(round(float(s), 3))
    shifts = Counter(round(b - a, 3) for ts in by.values() for i, a in enumerate(ts) for b in ts[i + 1:] if b > a)
    worst = max(shifts.items(), key=lambda kv: kv[1], default=(None, 0))
    return {"exact_shift_copies": int(worst[1]) if worst[1] >= max(5, 0.1 * len(texts)) else 0, "worst_shift_s": worst[0],
            "worst_shift_count": int(worst[1])}


def fulltext_counts(path) -> dict:
    """Word counts of the dataset's full text, per part: the citation header between the first two dash rules is dropped, parts are
    split at the ---...--- marker lines, and a word is a whitespace token with a letter or digit. Counts only, never text."""
    lines = open(path, encoding="utf-8").read().splitlines()
    rules = [i for i, l in enumerate(lines) if re.fullmatch(r"-{10,}", l.strip())]
    body = lines[rules[1] + 1:] if len(rules) >= 2 else lines
    parts, cur = [], []
    for l in body:
        if re.fullmatch(r"-{3}[^-].*[^-]-{3}", l.strip()):
            parts.append(cur); cur = []
        else:
            cur.append(l)
    parts.append(cur)
    n = [sum(1 for w in " ".join(p).split() if re.search(r"\w", w)) for p in parts]
    return dict(zip(STORIES, n)) if len(n) == len(STORIES) else {"total": sum(n)}


def word_checks(ev, tsv, C) -> dict:
    """tribe_common.event_checks plus the exact-shift test on the Word events; both required checks must hold."""
    rows = words_table(tsv)
    res = C.event_checks(ev, len(rows), sum(1 for r in rows if r[1] > 60))
    wd = ev[ev.type == "Word"]
    res.update(shift_copies(list(wd.text), list(wd.start)))
    res["words_ok"] = bool(len(wd) and res["last_word_before_audio_end"] and res["exact_shift_copies"] == 0)
    assert res["words_ok"], f"word checks failed: {({k: res[k] for k in ('n_words', 'last_word_before_audio_end', 'exact_shift_copies')})}"
    return res


def whisperx_version() -> list:
    """uvx's own pick of whisperx (recorded, not pinned), from the uv cache the launcher sets."""
    return sorted(Path(p).name for p in glob.glob(os.path.join(os.environ.get("UV_CACHE_DIR", ""), "archive-v0/*/whisperx-*.dist-info")))


def prepare(stage, wav, work, C) -> dict:
    """A stock stage's work folder: stim.wav, and for shuf the shuffled word table; full gets no table, so build_events runs TRIBE's
    speech-to-text and writes it (a table left by an earlier attempt of this stage is read instead, and recorded)."""
    _, story, mode = parse(stage)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wav, work / "stim.wav")
    if mode == "full":
        tsv = work / "stim.tsv"
        return {"speech_to_text": "read from an earlier attempt's stim.tsv" if tsv.exists() else "ran (ExtractWordsFromAudio)",
                "whisperx": whisperx_version()}
    if mode == "shuf":
        words = words_table(OUT / f"words_{story}.tsv")
        n = C.write_tsv(words, work / "stim.wav", shuffle_seed=SHUFFLE_SEED)
        return {"shuffle_seed": SHUFFLE_SEED, "n_tsv_words": n, "word_table_sha256": sha(OUT / f"words_{story}.tsv")}
    return {}


def check_shuffled(tsv, table) -> dict:
    """The shuffled table keeps every onset and duration and permutes the texts (section 8's control)."""
    a, b = words_table(tsv), words_table(table)
    same_times = [(x[1], x[2]) for x in a] == [(x[1], x[2]) for x in b]
    perm = sorted(x[0] for x in a) == sorted(x[0] for x in b)
    assert same_times and perm, "the shuffled table does not keep the word table's timings and texts"
    return {"same_onsets_and_durations": same_times, "texts_permuted": perm, "texts_moved": sum(x[0] != y[0] for x, y in zip(a, b))}


def main(stage):
    model, story, mode = parse(stage)
    sys.path.insert(0, str(ROOT / "alice/forecast"))
    import numpy as np
    import torch
    import run_alice_forecast as R
    import run_alice_forecast_tuned as RT
    C = R.C
    R.CACHE = CACHE
    wav = STIM / f"{story}_audio.wav"
    assert sha(wav) == STIM_SHA[story], "stimulus bytes differ from the pinned SHA-256"
    assert sha(FULLTEXT) == FULLTEXT_SHA, "full text bytes differ from the pinned SHA-256"
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    done = {p.stem[len("stage_"):] for p in OUT.glob("stage_*.json")}
    if missing_before(stage, done):
        raise SystemExit(f"run first: {missing_before(stage, done)}")
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    work = WORK / f"{story}_{mode}"
    res = {"stage": stage, "model": model, "story": story, "mode": mode, "wav_sha256": STIM_SHA[story]}
    if model == "tuned":
        work_before, cache_before = RT.files(work), RT.cache_files(CACHE)
        assert work_before, f"the stock work folder {work} is missing"
    else:
        res["prepare"] = prepare(stage, wav, work, C)
    feats = FEATS[mode]
    ev = C.build_events(work / "stim.wav", "english", with_text=("text" in feats))
    if "text" in feats:
        res["words"] = word_checks(ev, work / "stim.tsv", C)
        table = OUT / f"words_{story}.tsv"
        if mode == "full" and model == "stock":
            shutil.copyfile(work / "stim.tsv", table)
            res["words"]["fulltext_words"] = fulltext_counts(FULLTEXT).get(story)
        if mode == "full":
            assert sha(work / "stim.tsv") == sha(table), "the full stage's table differs from the story's word table"
        else:
            res["words"]["shuffled"] = check_shuffled(work / "stim.tsv", table)
        res["words"]["word_table_sha256"] = sha(table)
    else:
        res["audio_end_s"] = float(ev[ev.type == "Audio"].stop.max())
    end = res["words"]["audio_end_s"] if "text" in feats else res["audio_end_s"]
    R.PLACEMENT.clear(); R.pin_llama()
    m = RT._stock_loader(feats)
    if model == "tuned":
        res["weights"] = RT.apply_tuned(m)
    else:
        ck = glob.glob(os.environ.get("HF_HOME", "/tmp/hf") + "/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]
        res["weights"] = {"stock_ckpt_sha256": sha(ck)}
        assert res["weights"]["stock_ckpt_sha256"] == RT.STOCK_PIN, "stock checkpoint differs from the pin"
    starts, P, nwin = C.stitched_predict(m, ev, end)
    res.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),
                "contiguous_seconds": bool((np.diff(starts) == 1).all() and starts[0] == 0), **C.gpu_stats(),
                "seconds": round(time.time() - t0), "placement": {"audio_extractor": "cpu", "brain_model": "cpu", **R.PLACEMENT}})
    if "text" in feats and not R.PLACEMENT:
        res["placement"]["text_features"] = "read from this runner's feature cache"
    assert res["finite"] and res["contiguous_seconds"], "forecast not finite or not one row per second"
    if model == "tuned":
        added = sorted(RT.cache_files(CACHE) - cache_before)
        assert RT.files(work) == work_before, "the fine-tuned stage changed the stock stage's work folder"
        assert not added, f"{len(added)} files added to the feature cache: features were computed, not read"
        res["features"] = {"sound_and_text": "read from the stock stage's cache entries (no file added)", "work_folder_unchanged": True}
        res["vs_stock"] = RT.vs_stock_arrays(P, starts, OUT / f"pred_stock_{story}_{mode}.npz")
    f = OUT / f"pred_{stage}.npz"
    np.savez_compressed(f, P=P.astype("float32"), starts=starts)
    res["pred_sha256"] = sha(f)
    json.dump(res, open(OUT / f"stage_{stage}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "weights"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
