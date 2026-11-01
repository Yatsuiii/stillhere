"""One R2 stock stage inside the kernel's tribe environment (r2_forecast_tpl.py runs it as a separate process per stage, so each stage's
memory is freed): r2/run_r2_forecast.py's main for a stock stage, with every model on the T4.
usage: python r2_stage.py STAGE ROOT OUT CACHE CODE"""
import hashlib, json, sys, time
from pathlib import Path
stage, root, out, cache, code = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
sys.path.insert(0, code)
import numpy as np, torch  # noqa: E401,E402
import tribe_common as C  # noqa: E402
from r2_checks import check_shuffled, fulltext_counts, shift_copies, words_table  # noqa: E402
FEATS = {"audio": ["audio"], "full": ["text", "audio", "video"], "shuf": ["text", "audio", "video"]}       # the runner's
SHUFFLE_SEED, PIN = 7, "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


model, story, mode = stage.split("_")
assert model == "stock" and mode in FEATS, stage
wav, tsv, table = root / story / "stim.wav", root / story / "stim.tsv", out / f"words_{story}.tsv"
t0 = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
res = {"stage": stage, "model": model, "story": story, "mode": mode, "wav_sha256": sha(wav)}
if mode == "shuf":
    n = C.write_tsv(words_table(table), wav, shuffle_seed=SHUFFLE_SEED)
    res["prepare"] = {"shuffle_seed": SHUFFLE_SEED, "n_tsv_words": n, "word_table_sha256": sha(table)}
elif mode == "full":
    assert sha(tsv) == sha(table), "stim.tsv is not the story's word table"
    res["prepare"] = {"speech_to_text": "read from stim.tsv: the story's word table from the kernel stillhere-r2-asr (TRIBE v2's own "
                                        "speech-to-text, tribev2 af58661's function unchanged, on a Kaggle T4)"}
else:
    res["prepare"] = {}
before = sha(tsv) if mode != "audio" else None
ev = C.build_events(wav, "english", with_text=(mode != "audio"))
if mode != "audio":
    assert sha(tsv) == before, "the word table changed while the events were built"
    rows = words_table(tsv)
    w = C.event_checks(ev, len(rows), sum(1 for r in rows if r[1] > 60))
    wd = ev[ev.type == "Word"]
    w.update(shift_copies(list(wd.text), list(wd.start)))
    w["words_ok"] = bool(len(wd) and w["last_word_before_audio_end"] and w["exact_shift_copies"] == 0)
    assert w["words_ok"], f"word checks failed: {({k: w[k] for k in ('n_words', 'last_word_before_audio_end', 'exact_shift_copies')})}"
    if mode == "full":
        w["fulltext_words"] = fulltext_counts(root / "story_fulltext.txt").get(story)
    else:
        w["shuffled"] = check_shuffled(tsv, table)
    w["word_table_sha256"] = sha(table)
    res["words"] = w; end = w["audio_end_s"]
else:
    end = res["audio_end_s"] = float(ev[ev.type == "Audio"].stop.max())
m = C.load_model(FEATS[mode], cache, text_device="accelerate")
res["weights"] = C.apply_weights(m, "stock")
assert res["weights"]["stock_ckpt_sha256"] == PIN, "stock checkpoint differs from the pin"
starts, P, nwin = C.stitched_predict(m, ev, end)
res.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),
            "contiguous_seconds": bool(len(starts) and (np.diff(starts) == 1).all() and starts[0] == 0), **C.gpu_stats(),
            "seconds": round(time.time() - t0),
            "placement": {"route": "Kaggle T4 (rule amendment of Nov 1)", "audio_extractor": "cuda", "brain_model": "cuda",
                          "text_features": "accelerate"}})
assert res["finite"] and res["contiguous_seconds"], "forecast not finite or not one row per second"
f = out / f"pred_{stage}.npz"
np.savez_compressed(f, P=P.astype("float32"), starts=starts)
res["pred_sha256"] = sha(f)
json.dump(res, open(out / f"stage_{stage}.json", "w"), indent=1)
print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "weights"}), flush=True)
